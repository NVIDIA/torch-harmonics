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

# Integration of scalar fields over the sphere.
#
# This module sits above torch_harmonics.grid: it consumes a grid descriptor rather
# than computing nodes and weights itself. The quadrature rules live below the
# descriptor, in torch_harmonics.quadrature, giving a strict layering of
#
#     quadrature (rules) -> grid (descriptors) -> integration (layers)
#
# with no cycle, which is why the layer cannot live alongside the rules.

from typing import Optional

import torch
import torch.nn as nn

from torch_harmonics._field_contract import check_field
from torch_harmonics.grid import PointSetS2, _rejects_legacy_signature, require_point_set
from torch_harmonics.radial import SphericalProductGrid, split_radial


class QuadratureS2(nn.Module):
    r"""
    Scalar quadrature on :math:`S^2`.

    Given a signal :math:`f` sampled at the points of a grid, this module
    approximates the surface integral over the sphere as a weighted sum:

    .. math::

        I[f] = \int_{S^2} f \; dA
        \;\approx\; \sum_{i} f(\theta_i, \lambda_i)\, W_i

    where :math:`W_i` are the per-point solid-angle weights reported by
    :attr:`~torch_harmonics.grid.PointSetS2.quad_weights`, which sum to
    :math:`4\pi`.

    On a ring-structured grid those weights factorize as
    :math:`W_i = q_k \cdot 2\pi / N_{\lambda,k}` for a point on ring :math:`k`,
    where :math:`q_k` are the latitudinal weights (which absorb the
    :math:`\sin\theta` Jacobian via the change of variable to
    :math:`\cos\theta`). The number of points :math:`N_{\lambda,k}` may differ
    from ring to ring, as on HEALPix.

    The choice of ``grid`` determines how the nodes :math:`\theta_k` and weights
    :math:`q_k` are computed:

    * ``"legendre-gauss"`` -- Gauss--Legendre quadrature.  Nodes are the roots
      of the Legendre polynomial :math:`P_N(\cos\theta)`.  Exact for
      polynomials of degree up to :math:`2N - 1`.
    * ``"lobatto"`` -- Gauss--Lobatto quadrature.  Nodes include both endpoints
      (poles).  Exact for polynomials of degree up to :math:`2N - 3`.
    * ``"equiangular"`` -- Clenshaw--Curtis quadrature on equiangular nodes.
      Nodes are equally spaced in :math:`\theta`.  Exact for polynomials of
      degree up to approximately :math:`N - 1`.
    * ``"trapezoidal"`` -- Trapezoidal rule on the :math:`\cos\theta` interval.
      Nodes are equally spaced in :math:`\cos\theta`, not in :math:`\theta`.
    * ``"healpix"`` -- the equal-area HEALPix pixelization
      (:class:`~torch_harmonics.healpix.HealpixGrid`): every pixel carries
      :math:`4\pi / N_\text{pix}`. Exact only for constants, but it converges as
      ``nside`` grows. A field on it is flat, ``(..., npix)``.

    When ``normalize=True``, the weights are divided by :math:`4\pi` so that
    the output represents the spherical mean rather than the integral:

    .. math::

        \bar{f} = \frac{1}{4\pi} \int_{S^2} f\; dA

    Parameters
    ----------
    grid : PointSetS2
        Descriptor of the sampling to integrate on. Any :class:`~torch_harmonics.grid.PointSetS2` is
        accepted: integration needs points and weights, not ring structure. The
        descriptor carries both the resolution and the quadrature rule, so no separate
        shape argument is needed. Build one with :func:`torch_harmonics.grid.as_grid`.
        On a :class:`~torch_harmonics.radial.SphericalProductGrid` every shell is integrated
        over the sphere alone, giving one integral per radial node; the volume integral
        takes :attr:`SphericalProductGrid.quad_weights
        <torch_harmonics.radial.SphericalProductGrid.quad_weights>` instead.
    normalize : bool, optional
        If ``True``, divides weights by :math:`4\pi` to return a spherical mean
        instead of an integral, by default ``False``.

    Examples
    --------
    Compute the surface area of the unit sphere (:math:`\int_{S^2} 1\,dA = 4\pi`):

    >>> import torch
    >>> import torch_harmonics as th
    >>> grid = th.as_grid("legendre-gauss", nlat=128, nlon=256)
    >>> quad = th.QuadratureS2(grid)
    >>> ones = torch.ones(1, 1, grid.nlat, grid.nlon)
    >>> round(quad(ones).item(), 5)  # ≈ 4π; the weights buffer is float32
    12.56637

    Compute the spherical mean of a field:

    >>> quad_norm = th.QuadratureS2(grid, normalize=True)
    >>> quad_norm(ones).item()  # ≈ 1.0
    1.0
    """

    @_rejects_legacy_signature(
        'img_shape, grid="equiangular", normalize=False',
        grid="img_shape",
    )
    def __init__(self, grid: PointSetS2, normalize: Optional[bool] = False):
        super().__init__()

        # integration needs points and weights and nothing else -- no ring structure, no
        # uniform longitude stride -- so this is the one guard that can be the weakest
        # on a product grid each shell is integrated alike, the radial axis kept
        grid, self.radial = split_radial(grid)
        self.grid = require_point_set(grid)
        self.normalize = normalize

        # the descriptor already folds the longitudinal factor into the per-point weight,
        # per ring. Deriving it here as a single `2 * pi / nlon` would silently assume every
        # ring carries the same number of longitudes, which is false on a reduced Gaussian
        # or HEALPix grid: the polar rings are shorter, so their points cover more solid
        # angle each and must be weighted more heavily.
        quad_weight = self.grid.quad_weights

        # apply normalization
        if normalize:
            quad_weight = quad_weight / (4.0 * torch.pi)

        # lay the weights out like the field they multiply: (nlat, nlon) on a regular grid,
        # flat (npoints,) otherwise. `grid.shape` is what a field on this grid looks like,
        # so this stays correct for both without branching on the family. No leading
        # singleton axes: broadcasting would add them to inputs with fewer leading dims.
        quad_weight = quad_weight.reshape(*self.grid.shape).to(torch.float32).contiguous()

        # how many trailing axes `forward` reduces over; 2 for a regular grid, 1 for a
        # ragged one. Derived from the same `grid.shape`, so it cannot disagree with the
        # buffer above.
        self.spatial_dims = tuple(range(-len(self.grid.shape), 0))

        # register buffer
        self.register_buffer("quad_weight", quad_weight, persistent=False)

    def extra_repr(self):
        radial = "" if self.radial is None else f",\nradial={self.radial!r}"
        return f"grid={self.grid!r}{radial},\nnormalize={self.normalize}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Integrate a signal over the sphere using the precomputed quadrature.

        Parameters
        ----------
        x : torch.Tensor
            Input signal whose trailing axes match ``grid.shape`` -- ``(..., nlat, nlon)``
            on a regular grid, ``(..., npoints)`` on a ragged or unstructured one.

        Returns
        -------
        torch.Tensor
            Integral of shape ``(...)`` (the input with its spatial axes reduced).
        """
        # the weights broadcast against x, so a size-1 spatial axis would integrate silently
        check_field(x, self.grid, radial=self.radial)

        return self._integrate(x)

    def _integrate(self, x: torch.Tensor) -> torch.Tensor:
        """The sum over the angular axes, unchecked; shared with :class:`QuadratureVolume`."""
        # reduce over the spatial axes only, however many the grid has
        return torch.sum(x * self.quad_weight, dim=self.spatial_dims)


class QuadratureVolume(QuadratureS2):
    r"""
    Volume integral of a signal on a :class:`~torch_harmonics.radial.SphericalProductGrid`.

    .. math::

        \int_V f \, dV = \int \left( \int_{S^2} f(r, \theta, \lambda) \, d\Omega \right) r^2 \, dr
            \approx \sum_k r_k^2 \, w_k \sum_i \omega_i \, f(r_k, x_i)

    The radial sum runs first, against the radial grid's
    :attr:`~torch_harmonics.radial.RadialGrid.volume_weights`, and :class:`QuadratureS2` then
    integrates the result over the sphere. The volume element is separable, so the order
    does not change the result, but radial-first leaves the angular reduction, and on the
    distributed layer its collectives, with nr times less to reduce.

    The domain covered is the radial rule's: :class:`~torch_harmonics.radial.LegendreGaussRadialGrid`
    integrates over the whole of ``[r_inner, r_outer]``, while
    :class:`~torch_harmonics.radial.GeometricRadialGrid` integrates over its nodes' range
    ``[r_min, r_max]`` only, leaving the rest of an unbounded or exterior domain unsampled.

    Parameters
    ----------
    grid : SphericalProductGrid
        The product of the angular grid and the radial grid to integrate on.
    normalize : bool, optional
        If ``True``, divides by the volume the weights integrate, :math:`4\pi \sum_k r_k^2 w_k`,
        to return the volume average instead of the integral, by default ``False``.

    Examples
    --------
    The volume of the spherical shell between radii 1 and 2:

    >>> import math, torch
    >>> import torch_harmonics as th
    >>> grid = th.SphericalProductGrid(th.as_grid("legendre-gauss", nlat=8, nlon=16), th.as_radial_grid("legendre-gauss", nr=4, r_inner=1.0, r_outer=2.0))
    >>> volume = th.QuadratureVolume(grid)(torch.ones(grid.shape))
    >>> math.isclose(volume.item(), 4.0 / 3.0 * math.pi * 7.0, rel_tol=1e-6)
    True
    """

    #: no pre-v1.0.0 signature to reject; see test_every_grid_taking_constructor_rejects_the_old_signature
    _legacy_signature_exempt = "new in v1.0.0rc1"

    def __init__(self, grid: SphericalProductGrid, normalize: Optional[bool] = False):
        if not isinstance(grid, SphericalProductGrid):
            raise TypeError(f"grid must be a SphericalProductGrid; for the integral over the sphere alone use QuadratureS2. Got {grid!r}.")

        # the angular weights, normalized to the sphere's mean if asked; the radial weights
        # below then complete either the integral or the volume average
        super().__init__(grid, normalize=normalize)

        radial_weight = self.radial.volume_weights
        if normalize:
            radial_weight = radial_weight / radial_weight.sum()
        # kept in float64: r^2 w spans the cube of the radial range, which leaves float32's
        # range on astrophysical radii in SI units, and in the other direction on tiny ones.
        # Cast at use, so a float64 field integrates in float64
        self.register_buffer("radial_weight", radial_weight.contiguous(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Integrate a signal over the volume.

        Parameters
        ----------
        x : torch.Tensor
            Input signal of shape ``(..., nr, *angular.shape)``.

        Returns
        -------
        torch.Tensor
            Integral of shape ``(...)``.
        """
        check_field(x, self.grid, radial=self.radial)

        # the radial sum first, then QuadratureS2's angular one on what is left
        weight = self.radial_weight.to(torch.promote_types(x.dtype, torch.float32))
        x = torch.sum(x * weight.reshape(-1, *([1] * len(self.grid.shape))), dim=-1 - len(self.grid.shape))
        return self._integrate(x)
