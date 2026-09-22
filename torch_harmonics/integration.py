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

from torch_harmonics.grid import PointSetS2, require_point_set


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
    :math:`\cos\theta`). Note the :math:`k` on :math:`N_\lambda`: the
    longitudinal spacing is per ring, not global. It is uniform only on a
    :class:`~torch_harmonics.grid.RegularGridS2`; on a reduced Gaussian or HEALPix
    grid the polar rings carry fewer points, so each of their points covers more
    solid angle and is weighted more heavily. This module does not assume
    otherwise -- it takes the weights from the descriptor rather than deriving a
    single :math:`\Delta\lambda`.

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

    When ``normalize=True``, the weights are divided by :math:`4\pi` so that
    the output represents the spherical mean rather than the integral:

    .. math::

        \bar{f} = \frac{1}{4\pi} \int_{S^2} f\; dA

    Parameters
    ----------
    grid : PointSetS2
        Descriptor of the sampling to integrate on. Any :class:`PointSetS2` is
        accepted: integration needs points and weights, not ring structure. The
        descriptor carries both the resolution and the quadrature rule, so no separate
        shape argument is needed. Build one with :func:`torch_harmonics.grid.as_grid`.
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

    def __init__(self, grid: PointSetS2, normalize: Optional[bool] = False):
        super().__init__()

        # integration needs points and weights and nothing else -- no ring structure, no
        # uniform longitude stride -- so this is the one guard that can be the weakest
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
        # so this stays correct for both without branching on the family.
        quad_weight = quad_weight.reshape(1, 1, *self.grid.shape).to(torch.float32).contiguous()

        # how many trailing axes `forward` reduces over; 2 for a regular grid, 1 for a
        # ragged one. Derived from the same `grid.shape`, so it cannot disagree with the
        # buffer above.
        self.spatial_dims = tuple(range(-len(self.grid.shape), 0))

        # register buffer
        self.register_buffer("quad_weight", quad_weight, persistent=False)

    def extra_repr(self):
        return f"grid={self.grid!r},\nnormalize={self.normalize}"

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
        # reduce over the spatial axes only, however many the grid has
        quad = torch.sum(x * self.quad_weight, dim=self.spatial_dims)

        return quad
