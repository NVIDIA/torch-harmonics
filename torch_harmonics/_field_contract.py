# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#


# The shape contract of a field handed to a layer.
#
# Every layer takes tensors laid out as leading axes followed by the spatial (or spectral)
# axes, and only the leading part differs from layer to layer:
#
#     QuadratureS2, ResampleS2, RealSHT           (..., *grid.shape)
#     RealVectorSHT                               (..., 2, *grid.shape)
#     InverseRealSHT                              (..., lmax, mmax)
#     DISCO, attention, SpectralConvS2            (batch, in_channels, *grid.shape)
#
# On a SphericalProductGrid every one of these gains the radial axis directly in front of
# the angular ones, (..., nr, *angular.shape); a layer on the sphere builds on the angular
# grid and passes the radial one here, as `radial`, only to state that axis.
#
# The trailing part is never spelled out by the layer: it is `grid.shape`, which is
# (nlat, nlon) on a RegularGridS2, (npoints,) on any other grid, and the local block on a
# shard. So a layer states only its leading contract and the descriptor supplies the rest,
# and a new grid family or a new decomposition needs no change here.
#
# Checked once, at the public forward. A wrong extent is not always an error further
# down: a size-1 axis broadcasts against a weight buffer, and a ragged kernel reads by
# flat index. The kernel entry points keep their own checks, which guard the custom ops
# rather than the layer.
#
# The checks go through torch_harmonics.utils.check, so they cost nothing but a guard
# under torch.compile and drop the message while tracing; in eager mode the message
# names the expected and the actual shape.

from typing import Any, Callable, Optional, Tuple, Union

import torch

from torch_harmonics.grid import GridShardS2, PointSetS2
from torch_harmonics.utils import check

#: A leading contract. Each entry is an exact extent (int) or any extent (None); a leading
#: ``...`` admits any number of further axes in front. ``(...,)``, the default, admits any
#: leading axes at all, and ``()`` none.
Leading = Tuple[Any, ...]


def _render(leading: Leading, trailing: Tuple[int, ...]) -> str:
    """The expected shape as the error message prints it, e.g. ``(*, 16, 64, 128)``."""
    names = ["..." if d is Ellipsis else "*" if d is None else str(d) for d in leading]
    return "(" + ", ".join(names + [str(d) for d in trailing]) + ")"


@torch.compiler.assume_constant_result
def _grid_shape(grid: Union[PointSetS2, GridShardS2]) -> Tuple[int, ...]:
    """
    ``grid.shape``, evaluated eagerly while tracing.

    A descriptor is immutable, so its shape is a constant of the compiled graph. Resolving
    it outside the trace keeps dynamo out of the descriptor's internals -- on a shard that
    is the split arithmetic -- which it would otherwise trace on every compile.
    """
    return tuple(int(d) for d in grid.shape)


def _check_shape(x: torch.Tensor, leading: Leading, trailing: Tuple[int, ...], name: str, where: Callable[[], str]) -> None:
    """Check ``x`` against ``(*leading, *trailing)``; shared by :func:`check_field` and :func:`check_coeffs`."""
    variadic = len(leading) > 0 and leading[0] is Ellipsis
    fixed = leading[1:] if variadic else leading
    # a contract is written by the layer, not the caller, so a malformed one is a bug here
    if Ellipsis in fixed:
        raise ValueError(f"'...' may only lead a contract, got {leading}")

    expected = (*fixed, *(int(d) for d in trailing))
    rendered = _render(leading, trailing)

    # one message for every way the shape can be wrong: it shows the whole contract next to
    # the actual shape, which is more use than naming the first axis that disagrees. Built
    # only on failure and only in eager mode, which is also why `where` is deferred: a
    # descriptor's repr is not traceable.
    def message():
        return f"Expected {name} of shape {rendered} {where()}, got {tuple(x.shape)}"

    # rank is static under dynamo, so this resolves at trace time
    ndim = len(expected)
    check(x.dim() >= ndim if variadic else x.dim() == ndim, message)

    # extents may be symbolic under dynamic shapes; checked one at a time, each becomes a
    # runtime assert rather than a guard on the conjunction
    for axis, extent in zip(range(-ndim, 0), expected):
        if extent is not None:
            check(x.shape[axis] == extent, message)


def check_field(x: torch.Tensor, grid: Union[PointSetS2, GridShardS2], leading: Leading = (...,), name: str = "x", radial: Optional[Any] = None) -> None:
    """
    Check that ``x`` is a field on ``grid``, shaped ``(*leading, *grid.shape)``.

    With ``radial``, a field on the product of ``grid`` with that radial grid, shaped
    ``(*leading, nr, *grid.shape)``.

    Parameters
    ----------
    x : torch.Tensor
        The tensor handed to the layer.
    grid : PointSetS2 or GridShardS2
        The grid ``x`` is sampled on, or this rank's shard of it.
    leading : tuple, optional
        The layer's leading contract, see :data:`Leading`, by default ``(...,)``.
    name : str, optional
        Name of the tensor in the error message, by default ``"x"``.
    radial : RadialGrid, optional
        The radial grid of a :class:`~torch_harmonics.SphericalProductGrid` whose angular
        factor is ``grid``, by default ``None``.

    Raises
    ------
    RuntimeError
        If the rank or an extent of ``x`` disagrees with the contract.
    """
    if radial is None:
        _check_shape(x, leading, _grid_shape(grid), name, lambda: f"on {grid!r}")
    else:
        _check_shape(x, leading, (radial.nr, *_grid_shape(grid)), name, lambda: f"on {grid!r} x {radial!r}")


def check_coeffs(x: torch.Tensor, lmax: int, mmax: int, leading: Leading = (...,), name: str = "x", radial: Optional[Any] = None) -> None:
    """
    Check that ``x`` holds spherical harmonic coefficients, shaped ``(*leading, lmax, mmax)``.

    The spectral counterpart of :func:`check_field`. ``lmax`` and ``mmax`` are this rank's
    extents on a distributed transform. With ``radial``, one set of coefficients per
    radial node, ``(*leading, nr, lmax, mmax)``.

    Raises
    ------
    RuntimeError
        If the rank or an extent of ``x`` disagrees with the contract.
    """
    if radial is None:
        _check_shape(x, leading, (lmax, mmax), name, lambda: f"for lmax={lmax}, mmax={mmax}")
    else:
        _check_shape(x, leading, (radial.nr, lmax, mmax), name, lambda: f"for lmax={lmax}, mmax={mmax} on {radial!r}")
