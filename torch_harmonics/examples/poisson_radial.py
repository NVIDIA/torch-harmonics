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

# After a spherical harmonic transform, lap u = f decouples into one radial ODE per
# degree, u'' + (2/r) u' - l(l+1)/r**2 u = f. The homogeneous solutions are r**l and
# r**(-(l+1)), and in Sturm-Liouville form (r**2 u')' - l(l+1) u = r**2 f the
# combination p W = -(2l+1) is constant, giving the exact solution
#
#     u_lm(r) = -1 / (2l+1) int y_in(r_<) r_>**(-(l+1)) f_lm(s) s**2 ds
#
# with the inner solution set by the domain: y_in = r**l on the half-line (regularity
# at the origin), and y_in = r**l - R**(2l+1) r**(-(l+1)) on the exterior domain with
# homogeneous Dirichlet data at R (the image-charge term). The image term leaves the
# Wronskian unchanged, so one code path serves both, and R -> 0 degenerates to the
# half-line. The kernel is assembled in log space, where the exponent is bounded above
# and so can only underflow, never overflow.


import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from torch_harmonics.quadrature import trapezoidal_weights

from .radial_geometry import coordinates, default_grid_kind, to_uniform


def contract(kernel: torch.Tensor, fspec: torch.Tensor, eq: str) -> torch.Tensor:
    """Contract a real kernel with a possibly complex spectral field, without copying the kernel."""

    if not fspec.is_complex():
        return torch.einsum(eq, kernel, fspec)

    return torch.complex(torch.einsum(eq, kernel, fspec.real), torch.einsum(eq, kernel, fspec.imag))


def geometric_grid(vmin: float, vmax: float, n: int, dtype: torch.dtype = torch.float64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Geometric grid on [vmin, vmax], uniform in x = log(v).

    A dilation by a = q**j, where q = exp(dx), acts on this grid as an exact shift of
    the node index by j.

    Parameters
    -----------
    vmin : float
        Lower bound, must be positive
    vmax : float
        Upper bound
    n : int
        Number of nodes
    dtype : torch.dtype, optional
        Floating point type, by default torch.float64

    Returns
    -------
    x : torch.Tensor
        Log-coordinates, uniformly spaced
    v : torch.Tensor
        Nodes, v = exp(x)
    dx : torch.Tensor
        Node spacing in log-coordinates
    w : torch.Tensor
        Trapezoidal weights for the integral over dv
    """

    x, wx = trapezoidal_weights(n, math.log(vmin), math.log(vmax))
    x = x.to(dtype)
    v = torch.exp(x)

    # dv = v dx
    return x, v, x[1] - x[0], v * wx.to(dtype)


def radial_grid(nr: int, vmin: float, vmax: float, domain: str = "half-line", R: Optional[float] = None, nondim: bool = True, dtype: torch.dtype = torch.float64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Radial grid and quadrature weights for either dilation-group domain.

    On the half-line the grid is geometric in r. On the exterior domain it is geometric
    in the shifted coordinate rho = r - R, so that the transported-boundary action
    (r, R) -> (a r, a R) remains an index shift.

    Parameters
    -----------
    nr : int
        Number of radial nodes
    vmin : float
        Lower bound on r, or on rho = r - R for the exterior domain, or on rho / R when
        nondim is set
    vmax : float
        Upper bound, interpreted like vmin
    domain : str, optional
        Either "half-line" or "exterior", by default "half-line"
    R : float, optional
        Inner radius, required for domain="exterior", by default None
    nondim : bool, optional
        Exterior only. Build the grid in rho / R, so the sampled node array is identical
        for every R and arrays for different R compare without interpolation, by default True
    dtype : torch.dtype, optional
        Floating point type, by default torch.float64

    Returns
    -------
    x : torch.Tensor
        Log-coordinates of the nodes, in the reduced variable: log r on the half-line and
        log(rho / R) on the exterior when nondim is set
    r : torch.Tensor
        Radial nodes
    w : torch.Tensor
        Trapezoidal weights for the integral over dr
    """

    kind = default_grid_kind(domain)
    if domain == "exterior" and R is None:
        raise ValueError("R must be given for domain='exterior'")

    # the non-dimensional grid is the same shape scaled by R, so fold R into the bounds
    scale = R if (domain == "exterior" and nondim) else 1.0

    x, wx = trapezoidal_weights(nr, *to_uniform(vmin, vmax, kind))
    x = x.to(dtype)

    # x stays the reduced variable the source generator and the plots are written
    # against; R enters through the scale factor rather than through the exponent
    r, jac = coordinates(x, kind, R, scale=scale)

    # dr = J ds, so the quadrature weight for the integral over dr is J times the
    # trapezoid weight. Note r**2 * w is then m ds trap, the same measure the heat
    # solver's mass matrix uses.
    return x, r, jac * wx.to(dtype)


class RadialPoissonOperator(nn.Module):
    """
    Exact per-degree Green's operator for the radial Poisson equation.

    Solves the radial ODE that lap u = f reduces to under a spherical harmonic
    transform, in closed form. See the note at the top of this file for the derivation.
    Deliberately free of any SHT dependency, so that it can be tested on its own and
    reused by a radial spectral layer.

    Parameters
    -----------
    r : torch.Tensor
        Radial nodes, shape (nr,)
    w : torch.Tensor
        Quadrature weights for the integral over dr, shape (nr,)
    lmax : int
        Number of spherical harmonic degrees
    domain : str, optional
        Either "half-line" or "exterior", by default "half-line"
    R : float, optional
        Inner radius, required for domain="exterior", by default None
    bc : str, optional
        Inner boundary condition, currently only "dirichlet", by default "dirichlet"
    """

    def __init__(self, r: torch.Tensor, w: torch.Tensor, lmax: int, domain: str = "half-line", R: Optional[float] = None, bc: str = "dirichlet"):
        super().__init__()

        if domain not in ("half-line", "exterior"):
            raise NotImplementedError(f"Domain {domain} not implemented")
        if domain == "exterior" and R is None:
            raise ValueError("R must be given for domain='exterior'")
        if bc != "dirichlet":
            # Neumann gives y_in = r**l + l / (l + 1) R**(2l+1) r**(-(l+1)), so the condition
            # enters through one scalar image coefficient and only _assemble changes below.
            # Note that l = 0 Neumann carries a constant nullspace and needs separate handling.
            raise NotImplementedError(f"Boundary condition {bc} not implemented")

        self.nr = r.shape[0]
        self.lmax = lmax
        self.domain = domain
        self.R = R
        self.bc = bc

        logr = torch.log(r)
        # the total per-node source factor is r_j**2 w_j
        srcfac = r**2 * w

        green = self._assemble(logr, logr, srcfac)

        l = torch.arange(0, lmax, dtype=r.dtype).reshape(-1, 1)

        # harmonic lift for inhomogeneous Dirichlet data, h_lm (R / r)**(l + 1)
        blift = torch.exp((l.reshape(1, -1) + 1) * (math.log(R) - logr.reshape(-1, 1))) if domain == "exterior" else torch.zeros(self.nr, lmax, dtype=r.dtype)

        # multipole coefficients C_lm = -1 / (2 l + 1) int f_lm s**(l + 2) ds
        moments = -torch.exp((l + 2) * logr.reshape(1, -1)) * w.reshape(1, -1) / (2 * l + 1)

        self.register_buffer("r", r)
        self.register_buffer("w", w)
        self.register_buffer("srcfac", srcfac)
        self.register_buffer("green", green)
        self.register_buffer("blift", blift)
        self.register_buffer("moments", moments)

    def _assemble(self, logr_out: torch.Tensor, logr_src: torch.Tensor, srcfac: torch.Tensor) -> torch.Tensor:
        """Green's kernel of shape (lmax, len(logr_out), len(logr_src)), assembled in log space."""

        loglo = torch.minimum(logr_out.reshape(-1, 1), logr_src.reshape(1, -1))
        loghi = torch.maximum(logr_out.reshape(-1, 1), logr_src.reshape(1, -1))

        l = torch.arange(0, self.lmax, dtype=logr_src.dtype, device=logr_src.device).reshape(-1, 1, 1)

        core = torch.exp(l * loglo - (l + 1) * loghi)
        if self.domain == "exterior":
            core = core - torch.exp((2 * l + 1) * math.log(self.R) - (l + 1) * (loglo + loghi))

        return -core * srcfac.reshape(1, 1, -1) / (2 * l + 1)

    def extra_repr(self):
        return f"nr={self.nr}, lmax={self.lmax},\n domain={self.domain}, R={self.R}, bc={self.bc}"

    def forward(self, fspec: torch.Tensor, v0spec: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Solve lap u = f in spectral space.

        Parameters
        -----------
        fspec : torch.Tensor
            Spectral coefficients of the source, shape (..., nr, lmax, mmax)
        v0spec : torch.Tensor, optional
            Exterior only. Coefficients of the inner Dirichlet data, shape (..., lmax, mmax),
            by default None

        Returns
        -------
        torch.Tensor
            Spectral coefficients of the solution, shape (..., nr, lmax, mmax)
        """

        uspec = contract(self.green, fspec, "lkj,...jlm->...klm")

        if v0spec is not None:
            uspec = uspec + self.boundary_lift(v0spec)

        return uspec

    def solve_degree(self, f: torch.Tensor, l: int) -> torch.Tensor:
        """
        Apply the operator for a single degree to a radial profile of shape (..., nr).

        SHT-free entry point, for verifying the radial core against manufactured solutions.
        """

        return torch.einsum("kj,...j->...k", self.green[l], f)

    def kernel_row(self, r: float) -> torch.Tensor:
        """
        Green's kernel at an arbitrary radius, shape (lmax, nr).

        The radius need not be a grid node. A grid geometric in rho = r - R never contains
        rho = 0, so this is the only way to reach the inner boundary itself.
        """

        logr = torch.log(torch.as_tensor(r, dtype=self.r.dtype, device=self.r.device))
        return self._assemble(logr.reshape(1), torch.log(self.r), self.srcfac).squeeze(-2)

    def boundary_lift(self, v0spec: torch.Tensor) -> torch.Tensor:
        """
        Harmonic lift of inhomogeneous inner Dirichlet data, h_lm (R / r)**(l + 1).

        Takes coefficients of shape (..., lmax, mmax) to (..., nr, lmax, mmax).
        """

        if self.domain != "exterior":
            raise ValueError("boundary_lift is only defined on the exterior domain")

        return v0spec.unsqueeze(-3) * self.blift.reshape(self.nr, self.lmax, 1).to(v0spec.dtype)

    def multipole_moments(self, fspec: torch.Tensor) -> torch.Tensor:
        """
        Multipole coefficients C_lm of a compactly supported source, shape (..., lmax, mmax).

        Outside the support the solution is exactly u_lm(r) = C_lm r**(-(l + 1)), so
        truncating the grid above the support introduces no error in the ground truth.
        """

        return contract(self.moments, fspec, "lj,...jlm->...lm")
