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

from typing import Tuple, Optional
from torch_harmonics.cache import lru_cache
import math
import numpy as np
import torch

def _precompute_grid(n: int, grid: Optional[str]="equidistant", a: Optional[float]=0.0, b: Optional[float]=1.0,
                     periodic: Optional[bool]=False) -> Tuple[torch.Tensor, torch.Tensor]:

    if (grid != "equidistant") and periodic:
        raise ValueError(f"Periodic grid is only supported on equidistant grids.")

    # compute coordinates
    if grid == "equidistant":
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
def _precompute_longitudes(nlon: int):
    r"""
    Convenience routine to precompute longitudes
    """
    
    lons = torch.linspace(0, 2 * math.pi, nlon+1, dtype=torch.float64, requires_grad=False)[:-1]
    return lons


@lru_cache(typed=True, copy=True)
def _precompute_latitudes(nlat: int, grid: Optional[str]="equiangular") -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Convenience routine to precompute latitudes
    """
        
    # compute coordinates in the cosine theta domain
    xlg, wlg = _precompute_grid(nlat, grid=grid, a=-1.0, b=1.0, periodic=False)
    
    # to perform the quadrature and account for the jacobian of the sphere, the quadrature rule
    # is formulated in the cosine theta domain, which is designed to integrate functions of cos theta
    lats = torch.flip(torch.arccos(xlg), dims=(0,)).clone()
    wlg = torch.flip(wlg, dims=(0,)).clone()

    return lats, wlg


def trapezoidal_weights(n: int, a: Optional[float]=-1.0, b: Optional[float]=1.0, periodic: Optional[bool]=False) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Helper routine which returns equidistant nodes with trapezoidal weights
    on the interval [a, b]
    """

    xlg = torch.as_tensor(np.linspace(a, b, n, endpoint=periodic))
    wlg = (b - a) / (n - periodic * 1) * torch.ones(n, requires_grad=False)

    if not periodic:
        wlg[0] *= 0.5
        wlg[-1] *= 0.5

    return xlg, wlg


def legendre_gauss_weights(n: int, a: Optional[float]=-1.0, b: Optional[float]=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Helper routine which returns the Legendre-Gauss nodes and weights
    on the interval [a, b]
    """

    xlg, wlg = np.polynomial.legendre.leggauss(n)
    xlg = torch.as_tensor(xlg).clone()
    wlg = torch.as_tensor(wlg).clone()
    xlg = (b - a) * 0.5 * xlg + (b + a) * 0.5
    wlg = wlg * (b - a) * 0.5

    return xlg, wlg


def lobatto_weights(n: int, a: Optional[float]=-1.0, b: Optional[float]=1.0,
                    tol: Optional[float]=1e-16, maxiter: Optional[int]=100) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Helper routine which returns the Legendre-Gauss-Lobatto nodes and weights
    on the interval [a, b]
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


def clenshaw_curtiss_weights(n: int, a: Optional[float]=-1.0, b: Optional[float]=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Computation of the Clenshaw-Curtis quadrature nodes and weights.
    This implementation follows

    [1] Joerg Waldvogel, Fast Construction of the Fejer and Clenshaw-Curtis Quadrature Rules; BIT Numerical Mathematics, Vol. 43, No. 1, pp. 001–018.
    """

    assert n > 1

    tcc = torch.cos(torch.linspace(math.pi, 0, n, dtype=torch.float64, requires_grad=False))

    if n == 2:
        wcc = torch.tensor([1.0, 1.0], dtype=torch.float64)
    else:

        n1 = n - 1
        N = torch.arange(1, n1, 2, dtype=torch.float64)
        l = len(N)
        m = n1 - l

        v = torch.cat([2 / N / (N - 2), 1 / N[-1:], torch.zeros(m, dtype=torch.float64, requires_grad=False)])
        #v = 0 - v[:-1] - v[-1:0:-1]
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


def fejer2_weights(n: int, a: Optional[float]=-1.0, b: Optional[float]=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Computation of the Fejer quadrature nodes and weights.
    This implementation follows

    [1] Joerg Waldvogel, Fast Construction of the Fejer and Clenshaw-Curtis Quadrature Rules; BIT Numerical Mathematics, Vol. 43, No. 1, pp. 001–018.
    """

    assert n > 2

    tcc = torch.cos(torch.linspace(math.pi, 0, n, dtype=torch.float64, requires_grad=False))

    n1 = n - 1
    N = torch.arange(1, n1, 2, dtype=torch.float64)
    l = len(N)
    m = n1 - l

    v = torch.cat([2 / N / (N - 2), 1 / N[-1:], torch.zeros(m, dtype=torch.float64, requires_grad=False)])
    #v = 0 - v[:-1] - v[-1:0:-1]
    v = 0 - v[:-1] - torch.flip(v[1:], dims=(0,))

    wcc = torch.fft.ifft(v).real
    wcc = torch.cat((wcc, wcc[:1]))

    # rescale
    tcc = (b - a) * 0.5 * tcc + (b + a) * 0.5
    wcc = wcc * (b - a) * 0.5

    return tcc, wcc
