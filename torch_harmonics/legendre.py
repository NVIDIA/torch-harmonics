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
def legpoly(
    mmax: int,
    lmax: int,
    x: torch.Tensor,
    norm: Optional[str] = "ortho",
    inverse: Optional[bool] = False,
    csphase: Optional[bool] = True,
    *,
    mmin: Optional[int] = 0,
    lmin: Optional[int] = 0,
) -> torch.Tensor:
    """
    Computes the values of (-1)^m c^l_m P^l_m(x) at the positions specified by x.
    The resulting tensor has shape (mmax - mmin, lmax - lmin, len(x)). The Condon-Shortley
    Phase (-1)^m can be turned off optionally.

    The three-term recurrence has a sequential dependence in degree ``l`` (each ``l``
    reads ``l-1`` and ``l-2``), but for fixed ``l`` all orders ``m`` are independent;
    the inner ``m``-loop is therefore vectorized as a single tensor op, turning what
    would be O(nmax^2) kernel launches into O(nmax).

    Because of that dependence structure the degree axis is *streamed* rather than
    materialized: only the two previous degrees are carried, so the working set is
    O((mmax - mmin) * len(x)) instead of the O(mmax * lmax * len(x)) table. This is what
    lets a distributed transform build only the block it stores, rather than building the
    whole table and discarding most of it.

    ``mmin`` and ``lmin`` restrict which orders and degrees are *stored*, not which are
    *computed*. Both recurrences have to be walked from the start regardless:
    ``P^m_m`` is reached from ``P^{m-1}_{m-1}``, and ``P^m_l`` from ``P^m_{l-1}``. Only
    the evaluation points are free of this -- they are mutually independent, so
    restricting them needs no argument here, just a shorter ``x``.

    Parameters
    ----------
    mmax : int
        Maximum order of the spherical harmonics (exclusive)
    lmax : int
        Maximum degree of the spherical harmonics (exclusive)
    x : torch.Tensor
        Tensor of positions at which to evaluate the Legendre polynomials
    norm : Optional[str]
        Normalization of the Legendre polynomials
    inverse : Optional[bool]
        Whether to compute the inverse Legendre polynomials
    csphase : Optional[bool]
        Whether to apply the Condon-Shortley phase (-1)^m
    mmin : Optional[int]
        First order to store, by default 0
    lmin : Optional[int]
        First degree to store, by default 0

    Returns
    -------
    torch.Tensor
        Tensor of Legendre polynomial values, shape ``(mmax - mmin, lmax - lmin, len(x))``

    Raises
    ------
    ValueError
        If the requested order or degree range is not a valid half-open interval

    References
    ----------
    :cite:`Schaeffer2013`, :cite:`Rapp1982`, :cite:`Schrama1984`
    """

    if not 0 <= mmin <= mmax:
        raise ValueError(f"Expected 0 <= mmin <= mmax, got mmin={mmin}, mmax={mmax}")
    if not 0 <= lmin <= lmax:
        raise ValueError(f"Expected 0 <= lmin <= lmax, got lmin={lmin}, lmax={lmax}")

    nk = len(x)
    nm = mmax - mmin
    nl = lmax - lmin

    out = torch.zeros((nm, nl, nk), dtype=torch.float64, device=x.device, requires_grad=False)
    if nm == 0 or nl == 0:
        return out

    norm_factor = 1.0 if norm == "ortho" else math.sqrt(4 * math.pi)
    norm_factor = 1.0 / norm_factor if inverse else norm_factor

    # the two previous degrees over the stored order range, vdm[mmin:mmax, l-1] and [..., l-2]
    prev1 = torch.zeros((nm, nk), dtype=torch.float64, device=x.device)
    prev2 = torch.zeros((nm, nk), dtype=torch.float64, device=x.device)
    cur = torch.zeros((nm, nk), dtype=torch.float64, device=x.device)

    # vdm[l, l], the sectoral seed. This is the one quantity that couples orders, so it is
    # carried for every order from 0 upward even when the stored block starts higher.
    diag = torch.empty((nk,), dtype=torch.float64, device=x.device)

    for l in range(lmax):
        cur.zero_()

        if l == 0:
            diag_l = torch.full((nk,), norm_factor / math.sqrt(4 * math.pi), dtype=torch.float64, device=x.device)
        else:
            # sub-diagonal vdm[l-1, l] and diagonal vdm[l, l], both from vdm[l-1, l-1]
            sub_l = math.sqrt(2 * l + 1) * x * diag
            diag_l = torch.sqrt((2 * l + 1) * (1 + x) * (1 - x) / 2 / l) * diag

        # three-term recurrence for the interior orders m <= l-2, vectorized across m
        if l >= 2:
            m_hi = min(mmax - 1, l - 2)
            if m_hi >= mmin:
                m = torch.arange(mmin, m_hi + 1, dtype=torch.float64, device=x.device)
                a_lm = torch.sqrt((2 * l - 1) / (l - m) * (2 * l + 1) / (l + m))
                b_lm = torch.sqrt((l + m - 1) / (l - m) * (2 * l + 1) / (2 * l - 3) * (l - m - 1) / (l + m))
                nr = m_hi - mmin + 1
                cur[:nr] = a_lm.unsqueeze(-1) * x.unsqueeze(0) * prev1[:nr] - b_lm.unsqueeze(-1) * prev2[:nr]

        # the two boundary orders, where they fall inside the stored range
        if l >= 1 and mmin <= l - 1 < mmax:
            cur[l - 1 - mmin] = sub_l
        if mmin <= l < mmax:
            cur[l - mmin] = diag_l

        if l >= lmin:
            if norm == "schmidt":
                factor = math.sqrt(2 * l + 1)
                out[:, l - lmin] = cur * factor if inverse else cur / factor
            else:
                out[:, l - lmin] = cur

        # roll the window: cur becomes l-1, prev1 becomes l-2, prev2's buffer is reused
        prev2, prev1, cur = prev1, cur, prev2
        diag = diag_l

    if csphase:
        # negate odd orders; row r holds order mmin + r
        out[(1 if mmin % 2 == 0 else 0) :: 2] *= -1

    return out


@lru_cache(typed=True, copy=True)
@torch.no_grad()
def _precompute_legpoly(
    mmax: int,
    lmax: int,
    t: torch.Tensor,
    norm: Optional[str] = "ortho",
    inverse: Optional[bool] = False,
    csphase: Optional[bool] = True,
    *,
    mmin: Optional[int] = 0,
    lmin: Optional[int] = 0,
) -> torch.Tensor:
    r"""
    Computes the values of (-1)^m c^l_m P^l_m(\cos \theta) at the positions specified by t (theta).
    The resulting tensor has shape (mmax - mmin, lmax - lmin, len(t)).

    Parameters
    ----------
    mmax : int
        Maximum order of the spherical harmonics (exclusive)
    lmax : int
        Maximum degree of the spherical harmonics (exclusive)
    t : torch.Tensor
        Tensor of positions at which to evaluate the Legendre polynomials. Restricting the
        evaluation points is done by passing a shorter tensor -- they are independent of one
        another, unlike the order and degree ranges below.
    norm : Optional[str]
        Normalization of the Legendre polynomials
    inverse : Optional[bool]
        Whether to compute the inverse Legendre polynomials
    csphase : Optional[bool]
        Whether to apply the Condon-Shortley phase (-1)^m
    mmin : Optional[int]
        First order to store, by default 0
    lmin : Optional[int]
        First degree to store, by default 0

    Returns
    -------
    torch.Tensor
        Tensor of Legendre polynomial values
    """
    return legpoly(mmax, lmax, torch.cos(t), norm=norm, inverse=inverse, csphase=csphase, mmin=mmin, lmin=lmin)


@lru_cache(typed=True, copy=True)
@torch.no_grad()
def _precompute_dlegpoly(
    mmax: int,
    lmax: int,
    t: torch.Tensor,
    norm: Optional[str] = "ortho",
    inverse: Optional[bool] = False,
    csphase: Optional[bool] = True,
    *,
    mmin: Optional[int] = 0,
    lmin: Optional[int] = 0,
) -> torch.Tensor:
    r"""
    Computes the values of the derivatives $\frac{d}{d \theta} P^m_l(\cos \theta)$ as well as
    $\frac{1}{\sin \theta} P^m_l(\cos \theta)$ (with the implicit $-jm$ factor stripped),
    needed for the vector spherical harmonics. The resulting tensor has shape
    (2, mmax - mmin, lmax - lmin, len(t)).

    There is no inter-iteration dependence here -- each entry depends only on values from the
    precomputed associated Legendre table -- so both ``m`` and ``l`` axes are vectorized at once.

    Each output entry reads orders ``m-1`` and ``m+1`` at degrees ``l`` and ``l+1``, so a
    restricted block needs the underlying table widened by one in each direction. That halo is
    requested here rather than by the caller.

    Parameters
    ----------
    mmax : int
        Maximum order of the spherical harmonics (exclusive)
    lmax : int
        Maximum degree of the spherical harmonics (exclusive)
    t : torch.Tensor
        Tensor of positions at which to evaluate the Legendre polynomials
    norm : Optional[str]
        Normalization of the Legendre polynomials
    inverse : Optional[bool]
        Whether to compute the inverse Legendre polynomials
    csphase : Optional[bool]
        Whether to apply the Condon-Shortley phase (-1)^m
    mmin : Optional[int]
        First order to store, by default 0
    lmin : Optional[int]
        First degree to store, by default 0

    Returns
    -------
    torch.Tensor
        Tensor of derivative Legendre polynomial values

    References
    ----------
    :cite:`Wang2018`
    """

    # halo of one order below and above; the degree halo is the extra column at lmax
    pmin = max(0, mmin - 1)
    pct = _precompute_legpoly(mmax + 1, lmax + 1, t, norm=norm, inverse=inverse, csphase=False, mmin=pmin, lmin=lmin)

    nm = mmax - mmin
    nl = lmax - lmin

    dpct = torch.zeros((2, nm, nl, len(t)), dtype=torch.float64, device=t.device, requires_grad=False)
    if nm == 0 or nl == 0:
        return dpct

    # (nm, nl) coefficient grids, carrying the *global* order and degree of each entry
    m_idx = torch.arange(mmin, mmax, dtype=torch.float64, device=t.device)
    l_idx = torch.arange(lmin, lmax, dtype=torch.float64, device=t.device)
    m_g = m_idx.view(nm, 1)
    l_g = l_idx.view(1, nl)

    # advanced indices for pct lookups along the m axis, shifted into the halo's frame.
    # m_minus_1 is clamped (the m=0 column of the result is gated out by the mask below and
    # overwritten afterwards).
    m_minus_1 = (m_idx - 1).clamp(min=0).long() - pmin  # (nm,)
    m_plus_1 = (m_idx + 1).long() - pmin  # (nm,); largest value mmax, the last row of pct

    # mask of entries set by the interior+boundary recurrence: 1 <= m <= l.
    # the m=l boundary is naturally produced by the general formula (the (l-m) term vanishes).
    mask = ((m_g >= 1) & (m_g <= l_g)).unsqueeze(-1)

    # --- dpct[0]: d/dtheta P^m_l for 1 <= m <= l ---
    a0 = torch.sqrt(torch.clamp((l_g + m_g) * (l_g - m_g + 1), min=0.0))
    b0 = torch.sqrt(torch.clamp((l_g - m_g) * (l_g + m_g + 1), min=0.0))
    pct_mm1_l = pct[m_minus_1, :nl]
    pct_mp1_l = pct[m_plus_1, :nl]
    dpct[0, ...] = mask * 0.5 * (a0.unsqueeze(-1) * pct_mm1_l - b0.unsqueeze(-1) * pct_mp1_l)

    # m=0 row: dpct[0, 0, l] = -sqrt(l(l+1)) * pct[1, l], only present if the block starts at m=0
    if mmin == 0:
        coef_m0 = -torch.sqrt(l_idx * (l_idx + 1))
        dpct[0, 0, :] = coef_m0.unsqueeze(-1) * pct[1 - pmin, :nl]

    # --- dpct[1]: -1j m P^m_l / sin(theta) (imag part stripped) for 1 <= m <= l ---
    c1 = torch.sqrt((2 * l_g + 1) / (2 * l_g + 3))
    a1 = torch.sqrt(torch.clamp((l_g - m_g + 1) * (l_g - m_g + 2), min=0.0))
    b1 = torch.sqrt((l_g + m_g + 1) * (l_g + m_g + 2))
    pct_mm1_lp1 = pct[m_minus_1, 1 : nl + 1]
    pct_mp1_lp1 = pct[m_plus_1, 1 : nl + 1]
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
        # negate odd orders; row r holds order mmin + r
        dpct[:, (1 if mmin % 2 == 0 else 0) :: 2, :] *= -1

    return dpct
