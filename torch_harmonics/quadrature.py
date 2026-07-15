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

import math
from typing import Optional, Tuple

import numpy as np
import torch

from torch_harmonics.cache import lru_cache


def _precompute_quadrature_weights(
    n: int, grid: Optional[str] = "equiangular", a: Optional[float] = 0.0, b: Optional[float] = 1.0, periodic: Optional[bool] = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute grid points and quadrature weights for various quadrature rules.

    Parameters
    -----------
    n : int
        Number of grid points
    grid : str, optional
        Grid type (``"equiangular-trapezoidal"``, ``"legendre-gauss"``, ``"lobatto"``, ``"equiangular"``), by default ``"equiangular"``
    a : float, optional
        Lower bound of interval, by default 0.0
    b : float, optional
        Upper bound of interval, by default 1.0
    periodic : bool, optional
        Whether the grid is periodic (only for equiangular-trapezoidal), by default False

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Grid points and weights

    Raises
    ------
    ValueError
        If periodic is True for non-equiangular-trapezoidal grids or unknown grid type
    """

    if (grid != "equiangular-trapezoidal") and periodic:
        raise ValueError("Periodic grid is only supported on equiangular-trapezoidal grids.")

    # compute coordinates
    if grid == "equiangular-trapezoidal":
        xlg, wlg = trapezoidal_weights(n, a=a, b=b, periodic=periodic)
    elif grid == "legendre-gauss":
        xlg, wlg = legendre_gauss_weights(n, a=a, b=b)
    elif grid == "lobatto":
        xlg, wlg = lobatto_weights(n, a=a, b=b)
    elif grid == "equiangular":
        xlg, wlg = clenshaw_curtiss_weights(n, a=a, b=b)
    else:
        raise ValueError(f"Unknown grid type {grid}")

    return xlg, wlg


@lru_cache(typed=True, copy=True)
def precompute_longitudes(nlon: int):
    r"""
    Return equispaced longitude nodes in :math:`[0, 2\pi)`.

    Parameters
    ----------
    nlon : int
        Number of longitudinal nodes.

    Returns
    -------
    lons : torch.Tensor
        Tensor of longitude values in radians, shape ``(nlon,)``.
    """
    lons = torch.linspace(0, 2 * math.pi, nlon + 1, dtype=torch.float64, requires_grad=False)[:-1]
    return lons


@lru_cache(typed=True, copy=True)
def precompute_latitudes(nlat: int, grid: Optional[str] = "equiangular") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return latitude nodes and quadrature weights for the given grid type.

    Parameters
    ----------
    nlat : int
        Number of latitudinal nodes.
    grid : str, optional
        Quadrature grid type. One of ``"equiangular"`` (Clenshaw–Curtis),
        ``"legendre-gauss"``, ``"lobatto"``, or ``"equiangular-trapezoidal"``.
        Default is ``"equiangular"``.

    Returns
    -------
    lats : torch.Tensor
        Tensor of co-latitude values in radians, shape ``(nlat,)``.
    wlg : torch.Tensor
        Corresponding quadrature weights, shape ``(nlat,)``.
    """
    # compute coordinates in the cosine theta domain
    xlg, wlg = _precompute_quadrature_weights(nlat, grid=grid, a=-1.0, b=1.0, periodic=False)

    # to perform the quadrature and account for the jacobian of the sphere, the quadrature rule
    # is formulated in the cosine theta domain, which is designed to integrate functions of cos theta
    lats = torch.flip(torch.arccos(xlg), dims=(0,)).clone()
    wlg = torch.flip(wlg, dims=(0,)).clone()

    return lats, wlg


def trapezoidal_weights(n: int, a: Optional[float] = -1.0, b: Optional[float] = 1.0, periodic: Optional[bool] = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Helper routine which returns equiangular-trapezoidal nodes with trapezoidal weights
    on the interval [a, b]

    Parameters
    -----------
    n: int
        Number of quadrature nodes
    a: Optional[float]
        Lower bound of the interval
    b: Optional[float]
        Upper bound of the interval
    periodic: Optional[bool]
        Whether the grid is periodic

    Returns
    -------
    xlg: torch.Tensor
        Tensor of quadrature nodes
    wlg: torch.Tensor
        Tensor of quadrature weights
    """

    xlg = torch.as_tensor(np.linspace(a, b, n, endpoint=not periodic))
    wlg = (b - a) / (n - 1 + periodic * 1) * torch.ones(n, requires_grad=False)

    if not periodic:
        wlg[0] *= 0.5
        wlg[-1] *= 0.5

    return xlg, wlg


def legendre_gauss_weights(n: int, a: Optional[float] = -1.0, b: Optional[float] = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Helper routine which returns the Legendre-Gauss nodes and weights
    on the interval [a, b]

    Parameters
    -----------
    n: int
        Number of quadrature nodes
    a: Optional[float]
        Lower bound of the interval
    b: Optional[float]
        Upper bound of the interval

    Returns
    -------
    xlg: torch.Tensor
        Tensor of quadrature nodes
    wlg: torch.Tensor
        Tensor of quadrature weights
    """

    xlg, wlg = np.polynomial.legendre.leggauss(n)
    xlg = torch.as_tensor(xlg).clone()
    wlg = torch.as_tensor(wlg).clone()
    xlg = (b - a) * 0.5 * xlg + (b + a) * 0.5
    wlg = wlg * (b - a) * 0.5

    return xlg, wlg


def lobatto_weights(n: int, a: Optional[float] = -1.0, b: Optional[float] = 1.0, tol: Optional[float] = 1e-16, maxiter: Optional[int] = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Helper routine which returns the Legendre-Gauss-Lobatto nodes and weights
    on the interval [a, b]

    Parameters
    -----------
    n: int
        Number of quadrature nodes
    a: Optional[float]
        Lower bound of the interval
    b: Optional[float]
        Upper bound of the interval
    tol: Optional[float]
        Tolerance for the iteration
    maxiter: Optional[int]
        Maximum number of iterations

    Returns
    -------
    tlg: torch.Tensor
        Tensor of quadrature nodes
    wlg: torch.Tensor
        Tensor of quadrature weights

    """

    wlg = torch.zeros((n,), dtype=torch.float64, requires_grad=False)
    tlg = torch.zeros((n,), dtype=torch.float64, requires_grad=False)
    tmp = torch.zeros((n,), dtype=torch.float64, requires_grad=False)

    # Vandermonde Matrix
    vdm = torch.zeros((n, n), dtype=torch.float64, requires_grad=False)

    # initialize Chebyshev nodes as first guess
    for i in range(n):
        tlg[i] = -math.cos(math.pi * i / (n - 1))

    tmp = 2.0

    for i in range(maxiter):
        tmp = tlg

        vdm[:, 0] = 1.0
        vdm[:, 1] = tlg

        for k in range(2, n):
            vdm[:, k] = ((2 * k - 1) * tlg * vdm[:, k - 1] - (k - 1) * vdm[:, k - 2]) / k

        tlg = tmp - (tlg * vdm[:, n - 1] - vdm[:, n - 2]) / (n * vdm[:, n - 1])

        if max(abs(tlg - tmp).flatten()) < tol:
            break

    wlg = 2.0 / ((n * (n - 1)) * (vdm[:, n - 1] ** 2))

    # rescale
    tlg = (b - a) * 0.5 * tlg + (b + a) * 0.5
    wlg = wlg * (b - a) * 0.5

    return tlg, wlg


def clenshaw_curtiss_weights(n: int, a: Optional[float] = -1.0, b: Optional[float] = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computation of the Clenshaw-Curtis quadrature nodes and weights.
    This implementation follows

    Parameters
    -----------
    n: int
        Number of quadrature nodes
    a: Optional[float]
        Lower bound of the interval
    b: Optional[float]
        Upper bound of the interval

    Returns
    -------
    tcc: torch.Tensor
        Tensor of quadrature nodes
    wcc: torch.Tensor
        Tensor of quadrature weights

    References
    ----------
    :cite:`Waldvogel2006`
    """

    if n <= 1:
        raise ValueError(f"n must be greater than 1, got {n}")

    tcc = torch.cos(torch.linspace(math.pi, 0, n, dtype=torch.float64, requires_grad=False))

    if n == 2:
        wcc = torch.as_tensor([1.0, 1.0], dtype=torch.float64)
    else:

        n1 = n - 1
        N = torch.arange(1, n1, 2, dtype=torch.float64)
        l = len(N)
        m = n1 - l

        v = torch.cat([2 / N / (N - 2), 1 / N[-1:], torch.zeros(m, dtype=torch.float64, requires_grad=False)])
        # v = 0 - v[:-1] - v[-1:0:-1]
        v = 0 - v[:-1] - torch.flip(v[1:], dims=(0,))

        g0 = -torch.ones(n1, dtype=torch.float64, requires_grad=False)
        g0[l] = g0[l] + n1
        g0[m] = g0[m] + n1
        g = g0 / (n1**2 - 1 + (n1 % 2))
        wcc = torch.fft.ifft(v + g).real
        wcc = torch.cat((wcc, wcc[:1]))

    # rescale
    tcc = (b - a) * 0.5 * tcc + (b + a) * 0.5
    wcc = wcc * (b - a) * 0.5

    return tcc, wcc


class QuadratureS2(torch.nn.Module):
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
    * ``"equiangular-trapezoidal"`` -- Trapezoidal rule on equiangular nodes.

    When ``normalize=True``, the weights are divided by :math:`4\pi` so that
    the output represents the spherical mean rather than the integral:

    .. math::

        \bar{f} = \frac{1}{4\pi} \int_{S^2} f\; dA

    Parameters
    -----------
    img_shape: Tuple[int]
        Spatial grid shape ``(nlat, nlon)``.
    grid: str, optional
        Quadrature grid type (``"equiangular"``, ``"legendre-gauss"``,
        ``"lobatto"``, ``"equiangular-trapezoidal"``), by default ``"equiangular"``.
    normalize: bool, optional
        If ``True``, divides weights by :math:`4\pi` to return a spherical mean
        instead of an integral, by default ``False``.

    Examples
    --------
    Compute the surface area of the unit sphere (:math:`\int_{S^2} 1\,dA = 4\pi`):

    >>> import torch
    >>> import torch_harmonics as th
    >>> nlat, nlon = 128, 256
    >>> quad = th.QuadratureS2(img_shape=(nlat, nlon), grid="legendre-gauss")
    >>> ones = torch.ones(1, 1, nlat, nlon)
    >>> quad(ones).item()  # ≈ 4π
    12.566370614359172

    Compute the spherical mean of a field:

    >>> quad_norm = th.QuadratureS2(img_shape=(nlat, nlon), grid="legendre-gauss", normalize=True)
    >>> quad_norm(ones).item()  # ≈ 1.0
    1.0

    Raises
    ------
    ValueError
        If an unknown ``grid`` type is provided.
    """

    def __init__(self, img_shape: Tuple[int], grid: Optional[str] = "equiangular", normalize: Optional[bool] = False):
        super().__init__()

        self.grid = grid
        self.normalize = normalize

        if self.grid == "legendre-gauss":
            _, weights = legendre_gauss_weights(img_shape[0], -1, 1)
            dlambda = 2 * torch.pi / img_shape[1]
            quad_weight = dlambda * weights.unsqueeze(1)
            quad_weight = quad_weight.tile(1, img_shape[1])
        elif self.grid == "lobatto":
            _, weights = lobatto_weights(img_shape[0], -1, 1)
            dlambda = 2 * torch.pi / img_shape[1]
            quad_weight = dlambda * weights.unsqueeze(1)
            quad_weight = quad_weight.tile(1, img_shape[1])
        elif self.grid == "equiangular":
            _, weights = clenshaw_curtiss_weights(img_shape[0], -1, 1)
            dlambda = 2 * torch.pi / img_shape[1]
            quad_weight = dlambda * weights.unsqueeze(1)
            quad_weight = quad_weight.tile(1, img_shape[1])
        elif self.grid == "equiangular-trapezoidal":
            _, weights = trapezoidal_weights(img_shape[0], -1, 1)
            dlambda = 2 * torch.pi / img_shape[1]
            quad_weight = dlambda * weights.unsqueeze(1)
            quad_weight = quad_weight.tile(1, img_shape[1])
        else:
            raise (ValueError("Unknown quadrature mode"))

        # apply normalization
        if normalize:
            quad_weight = quad_weight / (4.0 * torch.pi)

        # make it contiguous
        quad_weight = quad_weight.contiguous()

        # reshape
        quad_weight = quad_weight.reshape(1, 1, *img_shape).to(torch.float32).contiguous()

        # register buffer
        self.register_buffer("quad_weight", quad_weight, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Integrate a signal over the sphere using the precomputed quadrature.

        Parameters
        ----------
        x: torch.Tensor
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
