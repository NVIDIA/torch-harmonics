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
from typing import Optional

import torch

from torch_harmonics.cache import lru_cache


def clm(l: int, m: int) -> float:
    """Defines the normalization factor to orthonormalize the Spherical Harmonics."""
    return math.sqrt((2 * l + 1) / 4 / math.pi) * math.sqrt(math.factorial(l - m) / math.factorial(l + m))


@torch.no_grad()
def legpoly(mmax: int, lmax: int, x: torch.Tensor, norm: Optional[str] = "ortho", inverse: Optional[bool] = False, csphase: Optional[bool] = True) -> torch.Tensor:
    """
    Computes the values of (-1)^m c^l_m P^l_m(x) at the positions specified by x.
    The resulting tensor has shape (mmax, lmax, len(x)). The Condon-Shortley Phase (-1)^m
    can be turned off optionally.

    The three-term recurrence has a sequential dependence in degree ``l`` (each ``l``
    reads ``l-1`` and ``l-2``), but for fixed ``l`` all orders ``m`` are independent;
    the inner ``m``-loop is therefore vectorized as a single tensor op, turning what
    would be O(nmax^2) kernel launches into O(nmax).

    Parameters
    -----------
    mmax: int
        Maximum order of the spherical harmonics
    lmax: int
        Maximum degree of the spherical harmonics
    x: torch.Tensor
        Tensor of positions at which to evaluate the Legendre polynomials
    norm: Optional[str]
        Normalization of the Legendre polynomials
    inverse: Optional[bool]
        Whether to compute the inverse Legendre polynomials
    csphase: Optional[bool]
        Whether to apply the Condon-Shortley phase (-1)^m

    Returns
    -------
    out: torch.Tensor
        Tensor of Legendre polynomial values

    References
    ----------
    :cite:`Schaeffer2013`, :cite:`Rapp1982`, :cite:`Schrama1984`
    """

    nmax = max(mmax, lmax)
    vdm = torch.zeros((nmax, nmax, len(x)), dtype=torch.float64, device=x.device, requires_grad=False)

    norm_factor = 1.0 if norm == "ortho" else math.sqrt(4 * math.pi)
    norm_factor = 1.0 / norm_factor if inverse else norm_factor

    vdm[0, 0, :] = norm_factor / math.sqrt(4 * math.pi)

    # diagonal and sub-diagonal: inherently sequential in l, but only O(nmax) ops
    for l in range(1, nmax):
        vdm[l - 1, l, :] = math.sqrt(2 * l + 1) * x * vdm[l - 1, l - 1, :]
        vdm[l, l, :] = torch.sqrt((2 * l + 1) * (1 + x) * (1 - x) / 2 / l) * vdm[l - 1, l - 1, :]

    # three-term recurrence: vectorize across m for each fixed l.
    # vdm[m, l] depends only on vdm[m, l-1] and vdm[m, l-2], so all m at fixed l are independent.
    for l in range(2, nmax):
        m = torch.arange(0, l - 1, dtype=torch.float64, device=x.device)
        a_lm = torch.sqrt((2 * l - 1) / (l - m) * (2 * l + 1) / (l + m))
        b_lm = torch.sqrt((l + m - 1) / (l - m) * (2 * l + 1) / (2 * l - 3) * (l - m - 1) / (l + m))
        vdm[: l - 1, l, :] = a_lm.unsqueeze(-1) * x.unsqueeze(0) * vdm[: l - 1, l - 1, :] - b_lm.unsqueeze(-1) * vdm[: l - 1, l - 2, :]

    if norm == "schmidt":
        l_vec = torch.arange(nmax, dtype=torch.float64, device=x.device)
        factor = torch.sqrt(2 * l_vec + 1)
        if inverse:
            vdm = vdm * factor.view(1, -1, 1)
        else:
            vdm = vdm / factor.view(1, -1, 1)

    vdm = vdm[:mmax, :lmax]

    if csphase:
        vdm[1::2] *= -1

    return vdm


@lru_cache(typed=True, copy=True)
@torch.no_grad()
def _precompute_legpoly(mmax: int, lmax: int, t: torch.Tensor, norm: Optional[str] = "ortho", inverse: Optional[bool] = False, csphase: Optional[bool] = True) -> torch.Tensor:
    r"""
    Computes the values of (-1)^m c^l_m P^l_m(\cos \theta) at the positions specified by t (theta).
    The resulting tensor has shape (mmax, lmax, len(t)).

    Parameters
    -----------
    mmax: int
        Maximum order of the spherical harmonics
    lmax: int
        Maximum degree of the spherical harmonics
    t: torch.Tensor
        Tensor of positions at which to evaluate the Legendre polynomials
    norm: Optional[str]
        Normalization of the Legendre polynomials
    inverse: Optional[bool]
        Whether to compute the inverse Legendre polynomials
    csphase: Optional[bool]
        Whether to apply the Condon-Shortley phase (-1)^m

    Returns
    -------
    out: torch.Tensor
        Tensor of Legendre polynomial values
    """
    return legpoly(mmax, lmax, torch.cos(t), norm=norm, inverse=inverse, csphase=csphase)


@lru_cache(typed=True, copy=True)
@torch.no_grad()
def _precompute_dlegpoly(mmax: int, lmax: int, t: torch.Tensor, norm: Optional[str] = "ortho", inverse: Optional[bool] = False, csphase: Optional[bool] = True) -> torch.Tensor:
    r"""
    Computes the values of the derivatives $\frac{d}{d \theta} P^m_l(\cos \theta)$ as well as
    $\frac{1}{\sin \theta} P^m_l(\cos \theta)$ (with the implicit $-jm$ factor stripped),
    needed for the vector spherical harmonics. The resulting tensor has shape
    (2, mmax, lmax, len(t)).

    There is no inter-iteration dependence here -- each entry depends only on values from the
    precomputed associated Legendre table -- so both ``m`` and ``l`` axes are vectorized at once.

    Parameters
    -----------
    mmax: int
        Maximum order of the spherical harmonics
    lmax: int
        Maximum degree of the spherical harmonics
    t: torch.Tensor
        Tensor of positions at which to evaluate the Legendre polynomials
    norm: Optional[str]
        Normalization of the Legendre polynomials
    inverse: Optional[bool]
        Whether to compute the inverse Legendre polynomials
    csphase: Optional[bool]
        Whether to apply the Condon-Shortley phase (-1)^m

    Returns
    -------
    out: torch.Tensor
        Tensor of derivative Legendre polynomial values

    References
    ----------
    :cite:`Wang2018`
    """

    pct = _precompute_legpoly(mmax + 1, lmax + 1, t, norm=norm, inverse=inverse, csphase=False)

    dpct = torch.zeros((2, mmax, lmax, len(t)), dtype=torch.float64, device=t.device, requires_grad=False)

    # (mmax, lmax) coefficient grids
    m_idx = torch.arange(mmax, dtype=torch.float64, device=t.device)
    l_idx = torch.arange(lmax, dtype=torch.float64, device=t.device)
    m_g = m_idx.view(mmax, 1)
    l_g = l_idx.view(1, lmax)

    # advanced indices for pct lookups along the m axis. m_minus_1 is clamped (the m=0
    # column of the result is gated out by the mask below and overwritten afterwards).
    m_minus_1 = (m_idx - 1).clamp(min=0).long()  # (mmax,)
    m_plus_1 = (m_idx + 1).long()  # (mmax,); max value mmax, valid in pct (mmax+1 rows)

    # mask of entries set by the interior+boundary recurrence: 1 <= m <= l.
    # the m=l boundary is naturally produced by the general formula (the (l-m) term vanishes).
    mask = ((m_g >= 1) & (m_g <= l_g)).unsqueeze(-1)

    # --- dpct[0]: d/dtheta P^m_l for 1 <= m <= l ---
    a0 = torch.sqrt(torch.clamp((l_g + m_g) * (l_g - m_g + 1), min=0.0))
    b0 = torch.sqrt(torch.clamp((l_g - m_g) * (l_g + m_g + 1), min=0.0))
    pct_mm1_l = pct[m_minus_1, :lmax]
    pct_mp1_l = pct[m_plus_1, :lmax]
    dpct[0, ...] = mask * 0.5 * (a0.unsqueeze(-1) * pct_mm1_l - b0.unsqueeze(-1) * pct_mp1_l)

    # m=0 row: dpct[0, 0, l] = -sqrt(l(l+1)) * pct[1, l]
    coef_m0 = -torch.sqrt(l_idx * (l_idx + 1))
    dpct[0, 0, :] = coef_m0.unsqueeze(-1) * pct[1, :lmax]

    # --- dpct[1]: -1j m P^m_l / sin(theta) (imag part stripped) for 1 <= m <= l ---
    c1 = torch.sqrt((2 * l_g + 1) / (2 * l_g + 3))
    a1 = torch.sqrt(torch.clamp((l_g - m_g + 1) * (l_g - m_g + 2), min=0.0))
    b1 = torch.sqrt((l_g + m_g + 1) * (l_g + m_g + 2))
    pct_mm1_lp1 = pct[m_minus_1, 1 : lmax + 1]
    pct_mp1_lp1 = pct[m_plus_1, 1 : lmax + 1]
    dpct[1, ...] = mask * 0.5 * c1.unsqueeze(-1) * (a1.unsqueeze(-1) * pct_mm1_lp1 + b1.unsqueeze(-1) * pct_mp1_lp1)

    # schmidt correction for dpct[1] -- the recurrence above was derived for ortho; pct[m, l+1]
    # carries degree-(l+1) schmidt scaling 1/sqrt(2l+3) (forward) or sqrt(2l+3) (inverse), so
    # we rescale back to the proper degree-l schmidt normalization.
    if norm == "schmidt":
        if not inverse:
            correction = torch.sqrt((2 * l_idx + 3) / (2 * l_idx + 1))
        else:
            correction = torch.sqrt((2 * l_idx + 1) / (2 * l_idx + 3))
        dpct[1, ...] = dpct[1, ...] * correction.view(1, -1, 1)

    if csphase:
        dpct[:, 1::2, :] *= -1

    return dpct
