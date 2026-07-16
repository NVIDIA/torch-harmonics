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

r"""Wigner (small) d- and D-matrices for rotations of spherical-harmonic fields.

The routines here compute the Wigner d-matrix

    d^l_{m'm}(beta) = <l m'| exp(-i beta J_y) | l m>

and the full Wigner D-matrix

    D^l_{m'm}(alpha, beta, gamma) = exp(-i m' alpha) d^l_{m'm}(beta) exp(-i m gamma)

in the *standard* (Varshalovich / quantum-mechanical) convention, i.e. the one
associated with Condon-Shortley-phased orthonormal spherical harmonics -- which
is exactly the convention used by :class:`torch_harmonics.RealSHT` with its
default ``csphase=True``.  Under an active rotation R = R_z(alpha) R_y(beta)
R_z(gamma), the degree-l coefficients of a field transform as

    a^{rot}_{l m'} = sum_m D^l_{m'm}(alpha, beta, gamma) a_{l m}.

The d-matrix is computed with a three-term recurrence in the degree l, in exact
analogy to the associated-Legendre recurrence used in
:func:`torch_harmonics.legendre.legpoly`.  The seed at the lowest degree
l0 = max(|m'|, |m|) is evaluated in log-gamma space so the routine stays stable
to high l.
"""

from typing import Optional

import torch

__all__ = ["wigner_d", "wigner_D"]


@torch.no_grad()
def wigner_d(lmax: int, beta: torch.Tensor, csphase: Optional[bool] = True) -> torch.Tensor:
    r"""Computes the real Wigner (small) d-matrices d^l_{m'm}(beta).

    The three-term recurrence has a sequential dependence in degree ``l`` (each
    ``l`` reads ``l-1`` and ``l-2``), but for fixed ``l`` all orders ``(m', m)``
    are independent; the inner double order-loop is therefore vectorized as a
    single tensor op, turning what would be O(lmax^3) kernel launches into
    O(lmax).  The result is band-packed into a tensor of shape
    ``(*beta.shape, lmax + 1, 2*lmax + 1, 2*lmax + 1)`` where the entry
    ``d[..., l, m' + lmax, m + lmax]`` holds ``d^l_{m'm}(beta)`` for
    ``|m'|, |m| <= l`` and is zero outside of that band.

    Parameters
    -----------
    lmax: int
        Maximum degree of the spherical harmonics (inclusive)
    beta: torch.Tensor
        Tensor of rotation angles (the second Euler angle) in radians, of
        arbitrary shape; the shape is preserved as leading batch dimensions
    csphase: Optional[bool]
        Whether the target spherical-harmonic basis carries the Condon-Shortley
        phase (default ``True``, matching :class:`torch_harmonics.RealSHT`).  If
        ``False`` the d-matrices are conjugated by ``diag((-1)^m)``.

    Returns
    -------
    out: torch.Tensor
        Tensor of Wigner d-matrix values

    References
    ----------
    [1] Varshalovich, D.A.; Moskalev, A.N.; Khersonskii, V.K.; Quantum Theory of Angular Momentum, World Scientific, 1988.
    """

    if lmax < 0:
        raise ValueError(f"lmax must be non-negative, got {lmax}")

    beta = torch.as_tensor(beta)
    device = beta.device
    batch_shape = beta.shape
    beta = beta.to(torch.float64).reshape(-1)  # (nbatch,), work in double like legpoly
    nbatch = beta.numel()
    dim = 2 * lmax + 1

    # half-angle factors, shape (nbatch, 1, 1)
    cosb = torch.cos(beta).reshape(nbatch, 1, 1)
    cosh_ = torch.cos(0.5 * beta).reshape(nbatch, 1, 1)
    sinh_ = torch.sin(0.5 * beta).reshape(nbatch, 1, 1)

    # m'- and m-grids over the full [-lmax, lmax] range, shape (dim, dim)
    m_vals = torch.arange(-lmax, lmax + 1, device=device, dtype=torch.float64)
    mp = m_vals.reshape(dim, 1).expand(dim, dim)  # rows -> m'
    mm = m_vals.reshape(1, dim).expand(dim, dim)  # cols -> m
    l0 = torch.maximum(mp.abs(), mm.abs())  # lowest degree carrying (m', m)

    # ---------------------------------------------------------------------
    # Seed d^{l0}_{m'm}(beta) for every (m', m).  Using the symmetry relations
    #   d^l_{m'm} = (-1)^{m'-m} d^l_{m m'}      (transpose)
    #   d^l_{m'm} = (-1)^{m'-m} d^l_{-m'-m}     (joint negation)
    # any pair reduces to the extremal row a := l0, b in [-l0, l0].  We seed the
    # *non*-Condon-Shortley extremal row (the (-1)^{l0-b} CS sign is folded into
    # the diag((-1)^m) post-step below, exactly as legpoly does):
    #   d^{l0}_{l0 b}(beta) = sqrt( (2 l0)! / ((l0+b)! (l0-b)!) )
    #                         * cos(beta/2)^{l0+b} * sin(beta/2)^{l0-b}
    # evaluated via lgamma to avoid factorial overflow.
    # ---------------------------------------------------------------------
    a = mp.clone()
    b = mm.clone()
    sign = torch.ones(dim, dim, device=device, dtype=torch.float64)

    # (1) make |a| >= |b| by transposing (m', m) -> (m, m') where needed
    swap = b.abs() > a.abs()
    sign = torch.where(swap, sign * (-1.0) ** (a - b), sign)
    a, b = torch.where(swap, b, a), torch.where(swap, a, b)

    # (2) make a >= 0 by joint negation (a, b) -> (-a, -b) where needed
    neg = a < 0
    sign = torch.where(neg, sign * (-1.0) ** (a - b), sign)
    a, b = torch.where(neg, -a, a), torch.where(neg, -b, b)
    # now a == l0 (>= 0) and |b| <= l0

    # log( sqrt( (2 l0)! / ((l0+b)! (l0-b)!) ) )
    log_norm = 0.5 * (torch.lgamma(2.0 * l0 + 1.0) - torch.lgamma(l0 + b + 1.0) - torch.lgamma(l0 - b + 1.0))
    norm = torch.exp(log_norm).reshape(1, dim, dim)
    exp_c = (l0 + b).reshape(1, dim, dim)  # power of cos(beta/2)
    exp_s = (l0 - b).reshape(1, dim, dim)  # power of sin(beta/2)
    seed = sign.reshape(1, dim, dim) * norm * cosh_.pow(exp_c) * sinh_.pow(exp_s)  # (nbatch, dim, dim)

    # ---------------------------------------------------------------------
    # Three-term recurrence in l:
    #   c1(l) d^{l+1} = c2(l) d^l - c3(l) d^{l-1}
    #   c1(l) = l   sqrt((l+1)^2 - m'^2) sqrt((l+1)^2 - m^2)
    #   c2(l) = (2l+1) ( l(l+1) cos(beta) - m' m )
    #   c3(l) = (l+1) sqrt(l^2 - m'^2) sqrt(l^2 - m^2)
    # Only written where l >= l0; for l < l0 the entry stays zero, and the seed
    # occupies the l0 slot.  Because c3(l0) = 0 the seed alone starts the chain.
    # The (m', m) = (0, 0) column has l0 = 0 where c1(0) = 0, so it additionally
    # gets d^1_{00} = cos(beta) and the loop starts at l = 1.
    # ---------------------------------------------------------------------
    d = torch.zeros(nbatch, lmax + 1, dim, dim, device=device, dtype=torch.float64)

    # place the seed at degree l0 for each (m', m)
    l0_idx = l0.to(torch.long).reshape(1, 1, dim, dim).expand(nbatch, 1, dim, dim)
    d.scatter_(1, l0_idx, seed.reshape(nbatch, 1, dim, dim))

    if lmax >= 1:
        # explicit seed for the (0, 0) column at degree 1: d^1_{00} = cos(beta)
        d[:, 1, lmax, lmax] = cosb.reshape(nbatch)

    mp2 = mp.pow(2).reshape(1, dim, dim)
    mm2 = mm.pow(2).reshape(1, dim, dim)
    mpmm = (mp * mm).reshape(1, dim, dim)

    # sequential in l; vectorized across all (m', m) for each fixed l
    for l in range(1, lmax):
        c1 = l * torch.sqrt(torch.clamp((l + 1.0) ** 2 - mp2, min=0.0)) * torch.sqrt(torch.clamp((l + 1.0) ** 2 - mm2, min=0.0))
        c2 = (2.0 * l + 1.0) * (l * (l + 1.0) * cosb - mpmm)
        c3 = (l + 1.0) * torch.sqrt(torch.clamp(l**2 - mp2, min=0.0)) * torch.sqrt(torch.clamp(l**2 - mm2, min=0.0))
        # c1 == 0 only for l == 0 (handled) or outside the band (masked below)
        safe_c1 = torch.where(c1 == 0.0, torch.ones_like(c1), c1)
        dnext = (c2 * d[:, l] - c3 * d[:, l - 1]) / safe_c1
        # only accept the update where l >= l0 (i.e. degree l+1 is inside the band)
        write = (l >= l0).reshape(1, dim, dim) & (c1 != 0.0)
        d[:, l + 1] = torch.where(write, dnext, d[:, l + 1])

    # The recurrence above builds d in the convention *without* the Condon-
    # Shortley phase.  Toggling it on is a diagonal conjugation by diag((-1)^m),
    # i.e. d^{CS}_{m'm} = (-1)^{m'-m} d_{m'm}, mirroring legpoly which applies the
    # (-1)^m factor as a final step when csphase=True.
    if csphase:
        d = d * ((-1.0) ** (mp - mm)).reshape(1, 1, dim, dim)

    return d.reshape(*batch_shape, lmax + 1, dim, dim)


@torch.no_grad()
def wigner_D(
    lmax: int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    csphase: Optional[bool] = True,
) -> torch.Tensor:
    r"""Computes the complex Wigner D-matrices D^l_{m'm}(alpha, beta, gamma).

    ``D^l_{m'm} = exp(-i m' alpha) d^l_{m'm}(beta) exp(-i m gamma)`` for the
    active rotation ``R_z(alpha) R_y(beta) R_z(gamma)``.  The result is packed
    identically to :func:`wigner_d` but is complex-valued.  ``alpha``, ``beta``
    and ``gamma`` are broadcast against one another.

    Parameters
    -----------
    lmax: int
        Maximum degree of the spherical harmonics (inclusive)
    alpha, beta, gamma: torch.Tensor
        Euler angles (Z-Y-Z convention) in radians, broadcastable to a common shape
    csphase: Optional[bool]
        Whether the target basis carries the Condon-Shortley phase (default ``True``)

    Returns
    -------
    out: torch.Tensor
        Complex tensor of Wigner D-matrix values
    """

    alpha = torch.as_tensor(alpha)
    beta = torch.as_tensor(beta)
    gamma = torch.as_tensor(gamma)
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)

    d = wigner_d(lmax, beta, csphase=csphase).to(torch.complex128)  # (*batch, lmax+1, dim, dim)

    dim = 2 * lmax + 1
    m_vals = torch.arange(-lmax, lmax + 1, device=d.device, dtype=torch.float64)
    a = alpha.to(torch.float64).reshape(*alpha.shape, 1, 1, 1)
    g = gamma.to(torch.float64).reshape(*gamma.shape, 1, 1, 1)

    phase_l = torch.exp(-1j * a * m_vals.reshape(dim, 1))  # (*batch, 1, dim, 1)
    phase_r = torch.exp(-1j * g * m_vals.reshape(1, dim))  # (*batch, 1, 1, dim)

    return phase_l * d * phase_r
