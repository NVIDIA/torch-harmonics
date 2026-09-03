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
import warnings
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
    ----------
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
    torch.Tensor
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


@lru_cache(typed=True, copy=False)
def compute_latitude_spacing(nlat: int, grid: Optional[str] = "equiangular") -> float:
    r"""
    Return the largest gap between adjacent latitude nodes of a grid.

    This is the grid's own notion of "one latitudinal grid spacing". Only the
    ``equiangular`` grid has uniform spacing, in which case this reduces to the
    familiar :math:`\pi / (N_\theta - 1)`. Gauss--Lobatto nodes cluster towards
    the equator, and ``equiangular-trapezoidal`` nodes are equispaced in
    :math:`\cos\theta` rather than in :math:`\theta`, so both are considerably
    coarser near the poles than a node count alone would suggest.

    Parameters
    ----------
    nlat : int
        Number of latitudinal nodes.
    grid : str, optional
        Quadrature grid type, by default ``"equiangular"``.

    Returns
    -------
    float
        Maximum spacing :math:`\max_k (\theta_{k+1} - \theta_k)` in radians.
    """
    lats, _ = precompute_latitudes(nlat, grid=grid)
    return (lats[1:] - lats[:-1]).max().item()


def compute_theta_cutoff(nlat: int, grid: Optional[str] = "equiangular", scale: Optional[float] = 1.0) -> float:
    r"""
    Default angular cutoff for localized operators on the sphere.

    Both the DISCO convolutions and neighborhood attention need a support radius
    for their filter basis. The heuristic is to take one latitudinal grid
    spacing of the coarser of the two grids involved, so that the basis functions
    of adjacent output points overlap and every output point sees more than the
    single latitude ring it sits on.

    The spacing is taken from the grid's actual node distribution
    (:func:`compute_latitude_spacing`) rather than from ``nlat`` alone. Using
    :math:`\pi / (N_\theta - 1)` is only correct for equiangular grids; on
    ``lobatto`` and ``equiangular-trapezoidal`` grids it underestimates the polar
    spacing by ~21% and ~5x respectively, which collapses the stencil of the
    polar output latitudes to a single latitude ring.

    Parameters
    ----------
    nlat : int
        Number of latitudinal nodes of the grid that sets the cutoff. This is the
        output grid for a forward transform and the input grid for a transpose
        one, mirroring which of the two is the coarser.
    grid : str, optional
        Quadrature grid type, by default ``"equiangular"``.
    scale : float, optional
        Multiplier on the grid spacing, by default 1.0.

    Returns
    -------
    float
        Cutoff angle in radians.

    Warns
    -----
    UserWarning
        On grids whose node spacing is not uniform in :math:`\theta`, where this
        returns a different value than the ``pi / (N_\theta - 1)`` heuristic used
        before v0.9.3. Equiangular grids are unaffected and do not warn.

    Notes
    -----
    This routine is the *rule*: it reports one latitudinal node spacing and
    nothing more. The *policy* on top of it -- applying an explicit
    ``theta_cutoff`` override, rejecting a non-positive radius, and refusing a
    shard, whose own node spacing would differ between ranks -- lives in
    :func:`torch_harmonics.truncate_support`, which is what the layers call.
    Prefer that over calling this directly.

    It keeps taking ``nlat`` and a grid string rather than a
    :class:`~torch_harmonics.grid.GridS2` because it sits *below* the descriptors
    in the layering: ``torch_harmonics.grid`` imports from this module, so a
    descriptor argument here would close an import cycle. Descriptor-based
    callers get the same quantity from
    :attr:`~torch_harmonics.grid.GridS2.max_latitude_spacing`, which derives it
    from the descriptor's own nodes instead of re-deriving it here. That leaves
    two routes to one number, so
    ``test_grid_assumptions.test_theta_cutoff_matches_the_free_function`` pins
    them to agree exactly.
    """
    dlat_max = compute_latitude_spacing(nlat, grid=grid)

    # only the equiangular grid is uniform in theta, so only there does the
    # superseded heuristic still agree (up to arccos roundoff)
    legacy = math.pi / float(nlat - 1)
    if abs(dlat_max - legacy) > 1e-9 * legacy:
        consequence = "the previous value under-covered the poles" if dlat_max > legacy else "the previous value was slightly wider than the grid warrants"
        warnings.warn(
            f"Default theta_cutoff changed in v0.9.3: the '{grid}' grid is not uniform in theta, so the cutoff is now "
            f"its maximum latitudinal node spacing ({dlat_max:.6f}) rather than pi/(nlat-1) ({legacy:.6f}); "
            f"{consequence}. Specify theta_cutoff explicitly to override.",
            UserWarning,
            stacklevel=2,
        )

    return scale * dlat_max


def trapezoidal_weights(n: int, a: Optional[float] = -1.0, b: Optional[float] = 1.0, periodic: Optional[bool] = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Helper routine which returns equiangular-trapezoidal nodes with trapezoidal weights
    on the interval [a, b]

    Parameters
    ----------
    n : int
        Number of quadrature nodes
    a : Optional[float]
        Lower bound of the interval
    b : Optional[float]
        Upper bound of the interval
    periodic : Optional[bool]
        Whether the grid is periodic

    Returns
    -------
    xlg : torch.Tensor
        Tensor of quadrature nodes
    wlg : torch.Tensor
        Tensor of quadrature weights
    """

    xlg = torch.as_tensor(np.linspace(a, b, n, endpoint=not periodic))
    # dtype is explicit: torch.ones defaults to float32, which would make these the only
    # float32 weights in the module and cap the accuracy of every rule derived from them
    wlg = (b - a) / (n - 1 + periodic * 1) * torch.ones(n, dtype=xlg.dtype, requires_grad=False)

    if not periodic:
        wlg[0] *= 0.5
        wlg[-1] *= 0.5

    return xlg, wlg


def legendre_gauss_weights(n: int, a: Optional[float] = -1.0, b: Optional[float] = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Helper routine which returns the Legendre-Gauss nodes and weights
    on the interval [a, b]

    Parameters
    ----------
    n : int
        Number of quadrature nodes
    a : Optional[float]
        Lower bound of the interval
    b : Optional[float]
        Upper bound of the interval

    Returns
    -------
    xlg : torch.Tensor
        Tensor of quadrature nodes
    wlg : torch.Tensor
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
    ----------
    n : int
        Number of quadrature nodes
    a : Optional[float]
        Lower bound of the interval
    b : Optional[float]
        Upper bound of the interval
    tol : Optional[float]
        Tolerance for the iteration
    maxiter : Optional[int]
        Maximum number of iterations

    Returns
    -------
    tlg : torch.Tensor
        Tensor of quadrature nodes
    wlg : torch.Tensor
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
    ----------
    n : int
        Number of quadrature nodes
    a : Optional[float]
        Lower bound of the interval
    b : Optional[float]
        Upper bound of the interval

    Returns
    -------
    tcc : torch.Tensor
        Tensor of quadrature nodes
    wcc : torch.Tensor
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
