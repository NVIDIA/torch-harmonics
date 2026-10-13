# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
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


import unittest

import torch
from parameterized import parameterized, parameterized_class
from testutils import requires_torch_compile

import torch_harmonics as th
from torch_harmonics import HealpixGrid, as_grid
from torch_harmonics._field_contract import check_coeffs, check_field

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))

_C = 4
_C_OUT = 6
_REGULAR = as_grid("legendre-gauss", nlat=8, nlon=16)
_HEALPIX = HealpixGrid(nside=2)


# Each case names a layer, the grid it runs on and the leading contract it declares. The
# factory builds the layer and reports the trailing input shape, the output shape as a
# function of the input's leading axes, and the input dtype. Accepted and rejected shapes
# are both derived from the contract rather than listed, so every layer is probed in the
# same ways.
def _sht(grid):
    layer = th.RealSHT(grid)
    return layer, grid.shape, lambda lead: (*lead, layer.lmax, layer.mmax), torch.float32


def _isht(grid):
    layer = th.InverseRealSHT(grid)
    return layer, (layer.lmax, layer.mmax), lambda lead: (*lead, *grid.shape), torch.complex64


def _vsht(grid):
    layer = th.RealVectorSHT(grid)
    return layer, grid.shape, lambda lead: (*lead, layer.lmax, layer.mmax), torch.float32


def _ivsht(grid):
    layer = th.InverseRealVectorSHT(grid)
    return layer, (layer.lmax, layer.mmax), lambda lead: (*lead, *grid.shape), torch.complex64


def _quadrature(grid):
    # the spatial axes are reduced and nothing else; in particular no leading axis is added
    return th.QuadratureS2(grid), grid.shape, lambda lead: lead, torch.float32


def _resample(grid):
    grid_out = as_grid("equiangular", nlat=grid.nlat + 2, nlon=grid.nlon)
    return th.ResampleS2(grid, grid_out), grid.shape, lambda lead: (*lead, *grid_out.shape), torch.float32


# the channel-mixing layers map _C to _C_OUT channels, so the output check sees the mapping
def _channels_out(grid):
    return lambda lead: (lead[0], _C_OUT, *grid.shape)


def _spectral_conv(grid):
    return th.SpectralConvS2(grid, grid, in_channels=_C, out_channels=_C_OUT), grid.shape, _channels_out(grid), torch.float32


def _disco(grid):
    layer = th.DiscreteContinuousConvS2(grid, grid, in_channels=_C, out_channels=_C_OUT, kernel_shape=3, optimized_kernel=False)
    return layer, grid.shape, _channels_out(grid), torch.float32


def _disco_transpose(grid):
    layer = th.DiscreteContinuousConvTransposeS2(grid, grid, in_channels=_C, out_channels=_C_OUT, kernel_shape=3, optimized_kernel=False)
    return layer, grid.shape, _channels_out(grid), torch.float32


def _attention(grid):
    return th.AttentionS2(grid, grid, in_channels=_C, out_channels=_C_OUT, num_heads=1), grid.shape, _channels_out(grid), torch.float32


def _neighborhood_attention(grid):
    layer = th.NeighborhoodAttentionS2(grid, grid, in_channels=_C, out_channels=_C_OUT, optimized_kernel=False)
    return layer, grid.shape, _channels_out(grid), torch.float32


# name, factory, grid, leading contract
_CASES = [
    ["sht", _sht, _REGULAR, (...,)],
    ["inverse_sht", _isht, _REGULAR, (...,)],
    ["vector_sht", _vsht, _REGULAR, (..., 2)],
    ["inverse_vector_sht", _ivsht, _REGULAR, (..., 2)],
    ["quadrature", _quadrature, _REGULAR, (...,)],
    ["quadrature_healpix", _quadrature, _HEALPIX, (...,)],
    ["resample", _resample, _REGULAR, (...,)],
    ["spectral_conv", _spectral_conv, _REGULAR, (None, _C)],
    ["disco", _disco, _REGULAR, (None, _C)],
    ["disco_healpix", _disco, _HEALPIX, (None, _C)],
    ["disco_transpose", _disco_transpose, _REGULAR, (None, _C)],
    ["disco_transpose_healpix", _disco_transpose, _HEALPIX, (None, _C)],
    ["attention", _attention, _REGULAR, (None, _C)],
    ["attention_healpix", _attention, _HEALPIX, (None, _C)],
    ["neighborhood_attention", _neighborhood_attention, _REGULAR, (None, _C)],
    ["neighborhood_attention_healpix", _neighborhood_attention, _HEALPIX, (None, _C)],
]


def _accepted_leadings(spec):
    """
    Leading shapes the contract admits: the minimum first, then expansions of it.

    The minimum fills every free extent with 1 and adds no optional axes. A ``...``
    contract then grows by one, two and three leading axes, one of them of size 1;
    an exact-rank contract instead grows its free extents.
    """
    variadic = spec[0] is Ellipsis
    fixed = spec[1:] if variadic else spec
    minimum = tuple(1 if d is None else d for d in fixed)
    if variadic:
        return [minimum, (3, *minimum), (2, 3, *minimum), (2, 1, 3, *minimum)]
    grown = tuple(3 if d is None else d for d in fixed)
    return [minimum] if grown == minimum else [minimum, grown]


def _rejected_shapes(spec, leading, trailing):
    """Shapes around an accepted one that break the contract, each with how it breaks it."""
    variadic = spec[0] is Ellipsis
    fixed = spec[1:] if variadic else spec
    offset = len(leading) - len(fixed)
    bad = []
    for i, extent in enumerate(trailing):
        wrong = list(trailing)
        wrong[i] = extent + 1
        bad.append((f"trailing axis {i} too long", (*leading, *wrong)))
        # the broadcast trap: a size-1 axis that a weight buffer would silently expand
        if extent != 1:
            wrong[i] = 1
            bad.append((f"trailing axis {i} of size 1", (*leading, *wrong)))
    for i, extent in enumerate(fixed):
        if extent is not None:
            wrong = list(leading)
            wrong[offset + i] = extent + 1
            bad.append((f"leading axis {offset + i} wrong", (*wrong, *trailing)))
    # fewer axes than any reading of the contract admits
    bad.append(("too few axes", tuple(trailing[1:])))
    if not variadic:
        bad.append(("extra axis", (1, *leading, *trailing)))
    return bad


def _shape_name(shape):
    return "x".join(str(d) for d in shape) or "none"


_SHAPE_CASES = [[f"{name}_{_shape_name(leading)}", factory, grid, leading] for name, factory, grid, spec in _CASES for leading in _accepted_leadings(spec)]


@parameterized_class(("device",), _devices)
class TestFieldContract(unittest.TestCase):
    """Every layer checks its input against ``(*leading, *grid.shape)`` the same way."""

    @parameterized.expand(_SHAPE_CASES)
    def test_shapes(self, _, factory, grid, leading):
        """Every shape the contract admits runs, from the minimum up, and produces the output shape the layer promises."""
        layer, trailing, out_shape, dtype = factory(grid)
        layer = layer.to(self.device)
        out = layer(torch.randn((*leading, *trailing), device=self.device, dtype=dtype))
        self.assertEqual(tuple(out.shape), out_shape(leading))

    @parameterized.expand(_CASES)
    def test_rejects_every_violation(self, _, factory, grid, spec):
        layer, trailing, _, dtype = factory(grid)
        layer = layer.to(self.device)
        # probe around an expanded shape, so a free leading axis is actually present
        leading = _accepted_leadings(spec)[-1]
        for how, shape in _rejected_shapes(spec, leading, trailing):
            with self.subTest(how=how, shape=shape):
                x = torch.randn(shape, device=self.device, dtype=dtype)
                with self.assertRaisesRegex(RuntimeError, r"Expected .* of shape"):
                    layer(x)


class TestFieldContractHelper(unittest.TestCase):
    """The helper itself: what it accepts, and what it reports."""

    def test_message_names_the_contract_and_the_grid(self):
        x = torch.zeros(2, 5, 8, 15)
        with self.assertRaises(RuntimeError) as ctx:
            check_field(x, _REGULAR, leading=(None, 4), name="query")
        message = str(ctx.exception)
        self.assertIn("query", message)
        self.assertIn("(*, 4, 8, 16)", message)
        self.assertIn(repr(_REGULAR), message)
        self.assertIn("(2, 5, 8, 15)", message)

    def test_message_for_coefficients(self):
        with self.assertRaisesRegex(RuntimeError, r"\(\.\.\., 2, 5, 3\) for lmax=5, mmax=3"):
            check_coeffs(torch.zeros(2, 5, 4), 5, 3, leading=(..., 2))

    def test_shard_supplies_the_local_block(self):
        shard = _REGULAR.shard(polar=(1, 2), azimuth=(0, 2))
        check_field(torch.zeros(3, *shard.shape), shard)
        with self.assertRaises(RuntimeError):
            check_field(torch.zeros(3, *_REGULAR.shape), shard)

    def test_ellipsis_only_leads(self):
        with self.assertRaises(ValueError):
            check_field(torch.zeros(2, 8, 16), _REGULAR, leading=(None, ...))

    def test_empty_leading_is_exact(self):
        check_field(torch.zeros(*_HEALPIX.shape), _HEALPIX, leading=())
        with self.assertRaises(RuntimeError):
            check_field(torch.zeros(1, *_HEALPIX.shape), _HEALPIX, leading=())

    @parameterized.expand([[False], [True]])
    @requires_torch_compile
    def test_traces_under_full_graph_compile(self, dynamic):
        """The check is a guard or a runtime assert, never a graph break, with static or symbolic extents."""
        quad = th.QuadratureS2(_REGULAR, normalize=True)
        compiled = torch.compile(quad, backend="aot_eager", fullgraph=True, dynamic=dynamic)
        for batch in (2, 3):
            out = compiled(torch.ones(batch, *_REGULAR.shape))
            torch.testing.assert_close(out, torch.ones(batch))
        with self.assertRaises(Exception):
            compiled(torch.ones(2, _REGULAR.nlat, _REGULAR.nlon + 1))


if __name__ == "__main__":
    unittest.main()
