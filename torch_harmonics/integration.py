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

from torch_harmonics.grid import RegularGridS2, require_regular_grid


class QuadratureS2(nn.Module):
    r"""
    Scalar quadrature on :math:`S^2` for integrating spherical fields defined on a
    latitude/longitude grid.

    Given a signal :math:`f(\theta, \lambda)` sampled on a latitude--longitude
    grid, this module approximates the surface integral over the sphere:

    .. math::

        I[f] = \int_0^{2\pi}\!\int_0^{\pi}
            f(\theta, \lambda)\,\sin\theta\; d\theta\; d\lambda
        \;\approx\; \sum_{k=0}^{N_\theta - 1} \sum_{j=0}^{N_\lambda - 1}
            f(\theta_k, \lambda_j)\, q_k\, \Delta\lambda

    where :math:`q_k` are the latitudinal quadrature weights (which absorb the
    :math:`\sin\theta` Jacobian via the change of variable to
    :math:`\cos\theta`) and :math:`\Delta\lambda = 2\pi / N_\lambda` is the
    uniform longitudinal spacing.

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
    grid : RegularGridS2
        Descriptor of the grid to integrate on. It carries both the resolution and
        the quadrature rule, so no separate shape argument is needed. Build one with
        :func:`torch_harmonics.grid.as_grid`.
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

    def __init__(self, grid: RegularGridS2, normalize: Optional[bool] = False):
        super().__init__()

        self.grid = require_regular_grid(grid)
        self.nlat, self.nlon = grid.shape
        self.normalize = normalize

        img_shape = grid.shape
        weights = grid.quad_weights
        dlambda = 2 * torch.pi / img_shape[1]
        quad_weight = dlambda * weights.unsqueeze(1)
        quad_weight = quad_weight.tile(1, img_shape[1])

        # apply normalization
        if normalize:
            quad_weight = quad_weight / (4.0 * torch.pi)

        # make it contiguous
        quad_weight = quad_weight.contiguous()

        # reshape
        quad_weight = quad_weight.reshape(1, 1, *img_shape).to(torch.float32).contiguous()

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
            Input signal of shape ``(..., nlat, nlon)``. Integration is over the last two
            (spatial) dimensions.

        Returns
        -------
        torch.Tensor
            Integral of shape ``(...)`` (the input with its last two dimensions reduced).
        """
        # integrate over last two axes only:
        quad = torch.sum(x * self.quad_weight, dim=(-2, -1))

        return quad
