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


"""
Heat equation on 3D spherical domains, solved by spherical-harmonic transform in
the angular directions and a Sturm-Liouville finite-volume discretization in the
radial direction.

    du/dt = nu * Laplacian(u),   Laplacian = (1/r^2) d_r(r^2 d_r) + (1/r^2) Lap_S2

Since Lap_S2 Y_l^m = -l(l+1) Y_l^m, the PDE decouples into one 1-D radial problem
per degree l (independent of order m):

    d_t a_lm(r,t) = nu * [ (1/r^2) d_r(r^2 d_r) - l(l+1)/r^2 ] a_lm(r,t) = nu * L_l a_lm

Interface deliberately mirrors torch_harmonics.examples.shallow_water_equations,
with the radial axis occupying the slot the SWE solver uses for its physical
channels. Coefficient tensors are (..., n_int, lmax, mmax); grid tensors are
(..., n_int, nlat, nlon).

Discretization
--------------
Write the radial coordinate as s (s = r on a linear grid, s = log r on a geometric
grid, s = log(r - R) on a shifted-geometric one) with Jacobian J = dr/ds. Then L_l is
in Sturm-Liouville form

    L_l u = (1/m) [ d_s( w d_s u ) - l(l+1) q u ],   w = r^2/J,  m = r^2 J,  q = J

    linear             (J = 1):    w = r^2,      m = r^2,      q = 1
    geometric          (J = r):    w = r,        m = r^3,      q = r
    shifted-geometric  (J = rho):  w = r^2/rho,  m = r^2 rho,  q = rho

Discretizing the flux form gives A_l = M^-1 K_l with K_l symmetric tridiagonal and
M = diag(m_i ds) diagonal positive. Hence

    Atilde_l = M^-1/2 K_l M^-1/2   is symmetric, exactly,

which is the discrete statement that the radial Laplacian is self-adjoint in the
r^(d-1) dr measure. Boundary conditions enter as diagonal modifications of K, so
symmetry survives them.

Domains
-------
shell     [r_in, r_out], node-centered, homogeneous Dirichlet at both walls.
          n_int = nr - 2 (the walls carry the boundary data and are not evolved).
half-line (0, inf) truncated to [r_in, r_out] with r_in << 1 << r_out, geometric
          grid, cell-centered. The regularity condition u ~ r^l at the origin is
          imposed as Neumann for l = 0 and Dirichlet for l >= 1; decay at infinity
          is Dirichlet at r_out. n_int = nr (every cell is evolved).

          r_in cannot be pushed arbitrarily small. On a geometric grid the
          symmetrized stiffness has ||Ktilde|| ~ 1/(r_in^2 ds), and `eigh` resolves
          eigenvalues only to an *absolute* eps*||Ktilde||, so the slow modes drown
          in rounding noise long before the r -> 0 truncation error (which is
          O(r_in^l)) becomes the limiting factor. With r_out = 3 and nr = 800,
          r_in = 1e-3 gives 6-digit eigenvalues while r_in = 1e-6 gives none at all.
          `slow_mode_resolution` reports the ratio; the constructor warns above 1e-4.
exterior  [R, inf) truncated to rho = r - R in [rho_min, rho_max], node-centered,
          Dirichlet at both walls, shifted-geometric grid. n_int = nr - 2.

          A geometric grid in rho cannot reach rho = 0, so the inner wall sits at
          r = R + rho_min rather than at R itself; rho_min -> 0 recovers the true
          exterior, the same kind of truncation the half-line makes at r_in. With
          nondim the grid is built in rho/R, so the whole domain including the wall
          dilates with R and the discrete spectrum carries the weight -2 of Lemma 1:
          lam(aR) = lam(R)/a^2 to roundoff.

Time evolution
--------------
The problem is linear and autonomous, so we never time-step it. With
Atilde_l = V diag(lam) V^T (lam <= 0, V orthogonal, from `eigh`), the exact
semigroup is

    exp(nu t A_l) = M^-1/2 V exp(nu t lam) V^T M^1/2

We never form that matrix -- on unbounded domains cond(M^1/2) = (r_out/r_in)^(3/2)
is large -- and instead propagate in the symmetrized variable utilde = M^1/2 u.
Ground-truth trajectories therefore carry *no* time-discretization error; the only
error is the radial finite-volume truncation.

Crank-Nicolson is provided (`cn_propagator`) solely to cross-check against
time-stepping reference implementations.
"""

import math
import warnings

import torch
import torch.nn as nn

import torch_harmonics as th
from torch_harmonics.quadrature import precompute_longitudes

from .radial_geometry import coordinates, sturm_liouville_weights, to_uniform, uniform_nodes


def _sturm_liouville(nr, v_in, v_out, radial_grid, layout, bc_inner, bc_outer, lvals, R=None, scale=1.0):
    """Symmetric stiffness K_l (nl, n_int, n_int) and the diagonal mass (n_int,).

    Boundary conditions are pure diagonal modifications, so K stays symmetric.
    The only thing that distinguishes them is how far the boundary sits from the
    outermost evolved node:

        node-centered Dirichlet : boundary node is a full ds away  -> factor 1
        cell-centered  Dirichlet : boundary face is ds/2 away      -> factor 2
        cell-centered  Neumann   : no flux through the face        -> factor 0

    `bc_inner="regular"` encodes the origin behaviour u ~ r^l of the half-line:
    Neumann for l = 0, Dirichlet for l >= 1.
    """
    s_in, s_out = to_uniform(v_in, v_out, radial_grid)
    s_ev, s_face, ds = uniform_nodes(nr, s_in, s_out, layout)

    r_ev, jac_ev = coordinates(s_ev, radial_grid, R, scale=scale)
    _, m_ev, q_ev = sturm_liouville_weights(r_ev, jac_ev)
    w_face, _, _ = sturm_liouville_weights(*coordinates(s_face, radial_grid, R, scale=scale))

    dirichlet_factor = 1.0 if layout == "node" else 2.0
    left = w_face[:-1] / ds  # face to the left of each evolved node
    right = w_face[1:] / ds  # face to the right

    def _scale(bc):
        if bc == "dirichlet":
            return dirichlet_factor
        if bc == "neumann":
            return 0.0
        raise ValueError(f"unknown bc: {bc}")

    # "regular" starts from Neumann; the l >= 1 Dirichlet term is added below
    left = left.clone()
    right = right.clone()
    left[0] *= _scale("neumann" if bc_inner == "regular" else bc_inner)
    right[-1] *= _scale(bc_outer)

    off = w_face[1:-1] / ds  # couplings between adjacent evolved nodes
    k_diff = torch.diag(-(left + right)) + torch.diag(off, 1) + torch.diag(off, -1)

    cl = (lvals * (lvals + 1)).to(r_ev.dtype)
    k = k_diff.unsqueeze(0) - cl.view(-1, 1, 1) * torch.diag(q_ev * ds).unsqueeze(0)

    if bc_inner == "regular":
        # u ~ r^l vanishes at the origin for l >= 1; l = 0 keeps zero flux
        k[1:, 0, 0] -= dirichlet_factor * w_face[0] / ds

    return k, m_ev * ds, r_ev


class SphericalHeatSolver(nn.Module):
    """Heat equation on a 3D spherical domain, exact in time.

    Parameters
    ----------
    nlat, nlon : int
        Angular grid size.
    nr : int
        Radial grid points. n_int = nr - 2 on the shell, nr on the half-line.
    dt : float
        Reference time step. `timestep(u, n)` advances by exactly `n * dt`.
    lmax, mmax : int, optional
        Spherical-harmonic truncation. NOTE these are *counts of degrees/orders*,
        following the torch-harmonics convention: lmax=17 means l = 0..16.
    grid : str
        Angular quadrature grid, "legendre-gauss" by default (exact quadrature).
    domain : str
        "shell", "half-line" or "exterior". Selects the radial layout and boundary
        conditions, and the default radial grid.
    r_in, r_out : float
        Shell and half-line only. Inner and outer radius. For "half-line" these are
        truncation radii and should satisfy r_in << 1 << r_out relative to the
        diffusion length.
    R : float, optional
        Exterior only. Inner radius, i.e. the radius the Dirichlet wall dilates with.
    rho_min, rho_max : float, optional
        Exterior only. Bounds on the shifted coordinate rho = r - R, in units of R
        when nondim is set. Note that a geometric grid in rho cannot reach rho = 0, so
        the wall sits at r = R + rho_min and rho_min -> 0 recovers the true exterior.
        This is the same kind of truncation the half-line makes at r_in.
    nondim : bool, optional
        Exterior only. Build the radial grid in rho / R, so the sampled node array is
        identical for every R and the wall stays at a fixed multiple of R.
    nu : float
        Diffusivity.
    radial_grid : str, optional
        "linear", "geometric" (log-uniform in r) or "shifted-geometric" (log-uniform
        in rho = r - R). Defaults to the domain's natural choice: linear for the
        shell, geometric for the half-line, shifted-geometric for the exterior.
    """

    def __init__(
        self,
        nlat,
        nlon,
        nr,
        dt,
        lmax=None,
        mmax=None,
        grid="legendre-gauss",
        domain="shell",
        r_in=1.0,
        r_out=2.0,
        R=None,
        rho_min=1e-2,
        rho_max=1e4,
        nondim=True,
        nu=1.0,
        radial_grid=None,
    ):
        super().__init__()

        if domain == "shell":
            layout, bc_inner, bc_outer = "node", "dirichlet", "dirichlet"
            radial_grid = radial_grid or "linear"
            self.n_int = nr - 2
        elif domain == "half-line":
            layout, bc_inner, bc_outer = "cell", "regular", "dirichlet"
            radial_grid = radial_grid or "geometric"
            self.n_int = nr
        elif domain == "exterior":
            # Dirichlet at the inner wall and at the outer truncation; the shell's
            # node layout applies verbatim, only the Jacobian changes
            layout, bc_inner, bc_outer = "node", "dirichlet", "dirichlet"
            radial_grid = radial_grid or "shifted-geometric"
            self.n_int = nr - 2
        else:
            raise NotImplementedError(f"domain={domain!r} not implemented")

        if domain == "exterior":
            if R is None:
                raise ValueError("R must be given for domain='exterior'")
            scale = R if nondim else 1.0
            v_in, v_out = rho_min, rho_max
            r_in, r_out = R, R + scale * rho_max
        else:
            if R is not None:
                raise ValueError(f"R is only meaningful on the exterior domain, not {domain!r}")
            scale = 1.0
            v_in, v_out = r_in, r_out

        self.dt = dt
        self.nlat = nlat
        self.nlon = nlon
        self.nr = nr
        self.grid = grid
        self.domain = domain
        self.radial_grid = radial_grid
        self.R = R
        self.nondim = nondim

        self.register_buffer("nu", torch.as_tensor(nu, dtype=torch.float64))
        self.register_buffer("r_in", torch.as_tensor(r_in, dtype=torch.float64))
        self.register_buffer("r_out", torch.as_tensor(r_out, dtype=torch.float64))

        # SHT. csphase=False to match the SWE solver's convention.
        self.sht = th.RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False)
        self.isht = th.InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False)
        self.lmax = self.sht.lmax
        self.mmax = self.sht.mmax

        # angular grid + quadrature weights, same construction as the SWE solver
        if grid == "legendre-gauss":
            cost, quad_weights = th.quadrature.legendre_gauss_weights(nlat, -1, 1)
        elif grid == "lobatto":
            cost, quad_weights = th.quadrature.lobatto_weights(nlat, -1, 1)
        elif grid == "equiangular":
            cost, quad_weights = th.quadrature.clenshaw_curtiss_weights(nlat, -1, 1)
        else:
            raise ValueError(f"unknown grid: {grid}")
        lats = -torch.arcsin(cost)
        lons = precompute_longitudes(nlon)

        # angular Laplacian eigenvalues on the unit sphere, shape (lmax, mmax)
        lvals = torch.arange(0, self.lmax, dtype=torch.float64)
        lgrid = lvals.reshape(self.lmax, 1).expand(self.lmax, self.mmax)
        lap = -lgrid * (lgrid + 1)

        # radial operator: one symmetric eigenproblem per degree l
        k, mass, r_ev = _sturm_liouville(nr, v_in, v_out, radial_grid, layout, bc_inner, bc_outer, lvals, R=R, scale=scale)

        msqrt = torch.sqrt(mass)
        minv_sqrt = 1.0 / msqrt
        ktilde = minv_sqrt.view(1, -1, 1) * k * minv_sqrt.view(1, 1, -1)
        # symmetrize away the ~1e-16 asymmetry from floating-point rounding
        ktilde = 0.5 * (ktilde + ktilde.transpose(-1, -2))
        evals, evecs = torch.linalg.eigh(ktilde)

        # `eigh` resolves eigenvalues to about eps * ||Ktilde||, an *absolute* error.
        # On a geometric grid ||Ktilde|| ~ 1/(r_in^2 ds), so shrinking r_in drowns the
        # physically relevant slow modes in rounding noise long before it improves the
        # r -> 0 truncation error. Report the ratio and complain when it gets close to
        # spoiling the slowest mode.
        lam_slow = evals[:, -1].max().abs().item()
        self.slow_mode_resolution = torch.finfo(torch.float64).eps * ktilde.abs().amax().item() / lam_slow
        if self.slow_mode_resolution > 1e-4:  # fewer than ~4 good digits in the slowest mode
            warnings.warn(
                f"radial eigenproblem is poorly conditioned: eps*||K||/|lam_slow| = {self.slow_mode_resolution:.1e}. "
                f"The slowest modes are corrupted by rounding. Increase r_in (currently {r_in:g}) or coarsen the radial grid.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.register_buffer("r", r_ev)
        self.register_buffer("lats", lats)
        self.register_buffer("lons", lons)
        self.register_buffer("l", lgrid.clone())
        self.register_buffer("lap", lap)
        self.register_buffer("quad_weights", quad_weights.reshape(-1, 1))
        self.register_buffer("mass", mass)
        self.register_buffer("msqrt", msqrt)
        self.register_buffer("minv_sqrt", minv_sqrt)
        self.register_buffer("evals", evals)  # (lmax, n_int), ascending, all < 0
        self.register_buffer("evecs", evecs)  # (lmax, n_int, n_int), orthogonal

    # -- transforms ---------------------------------------------------------

    def grid2spec(self, ugrid):
        """(..., n_int, nlat, nlon) real -> (..., n_int, lmax, mmax) complex."""
        return self.sht(ugrid)

    def spec2grid(self, uspec):
        """(..., n_int, lmax, mmax) complex -> (..., n_int, nlat, nlon) real."""
        return self.isht(uspec)

    # -- radial operator ----------------------------------------------------

    def radial_matrix(self, l: int) -> torch.Tensor:
        """Dense (n_int, n_int) matrix A_l = M^-1 K_l. For tests and Crank-Nicolson."""
        ktilde = self.evecs[l] @ torch.diag(self.evals[l]) @ self.evecs[l].T
        return self.minv_sqrt.view(-1, 1) * ktilde * self.msqrt.view(1, -1)

    def cn_propagator(self, l: int, dt: float) -> torch.Tensor:
        """One-step Crank-Nicolson propagator (I - cA)^-1 (I + cA), c = nu dt/2.

        Only for cross-checking time-stepping reference implementations; the solver
        itself uses the exact semigroup in `propagate`.
        """
        a = self.radial_matrix(l)
        c = 0.5 * self.nu * dt
        eye = torch.eye(self.n_int, dtype=a.dtype, device=a.device)
        return torch.linalg.solve(eye - c * a, eye + c * a)

    # -- time evolution -----------------------------------------------------

    def propagate(self, uspec: torch.Tensor, t) -> torch.Tensor:
        """Apply the exact solution operator exp(nu t A_l) for arbitrary t >= 0.

        uspec: (..., n_int, lmax, mmax) complex. The radial axis is -3.
        """
        evecs = self.evecs.to(uspec.dtype)
        # exp(nu t lam) underflows to 0 for the stiff end of the spectrum; that is
        # the correct answer (the mode is dead), not a numerical failure.
        decay = torch.exp(self.nu * t * self.evals).T.to(uspec.dtype)  # (n_int, lmax)

        u = uspec * self.msqrt.view(-1, 1, 1)  # to the symmetrized variable
        c = torch.einsum("lik,...ilm->...klm", evecs, u)  # project onto eigenbasis
        c = c * decay.unsqueeze(-1)  # exact per-mode decay
        u = torch.einsum("lik,...klm->...ilm", evecs, c)  # back to nodal
        return u * self.minv_sqrt.view(-1, 1, 1)

    def timestep(self, uspec: torch.Tensor, nsteps: int) -> torch.Tensor:
        """Advance by nsteps * dt. Exact -- there is no loop and no timestep error."""
        return self.propagate(uspec, nsteps * self.dt)

    # -- initial conditions -------------------------------------------------

    def _mask(self, uspec: torch.Tensor, lmax_ic: int) -> torch.Tensor:
        """Enforce m <= l, l <= lmax_ic, and reality of the m = 0 coefficients."""
        uspec = torch.tril(uspec)
        if lmax_ic + 1 < self.lmax:
            uspec[..., lmax_ic + 1 :, :] = 0.0
        uspec[..., 0].imag.zero_()
        return uspec

    def random_initial_condition(self, n_traj=1, lmax_ic=8, smoothness=2.0, generator=None):
        """Gaussian random field built from the radial operator's own eigenbasis.

        Radial coefficients are drawn in the symmetrized eigenbasis and damped as
        (lam_k / lam_0)^-smoothness, where lam_0 is the slowest-decaying mode. This
        is a Matern-type field: smooth, full rank (every mode is excited), and it
        satisfies the domain's boundary conditions exactly for free, because the
        eigenvectors do. Angular amplitudes follow a 1/(1+l)^2 spectrum.

        Note the eigenvectors themselves would be a *bad* basis to draw ICs from:
        each one evolves by a pure scalar exp(nu t lam_k), so trajectories confined
        to a few of them are trivial. Damping a full-rank draw keeps every decay
        rate present.
        """
        device = self.evals.device
        shape = (n_traj, self.n_int, self.lmax, self.mmax)

        # radial spectrum, (lmax, n_int) -> broadcast onto the (n_int, lmax) axes
        ratio = self.evals / self.evals[:, -1:]  # >= 1, since evals < 0 and ascending
        weight = ratio.pow(-smoothness).T.to(torch.complex128)

        # angular spectrum
        ang = (1.0 / (1.0 + torch.arange(self.lmax, dtype=torch.float64, device=device)) ** 2).to(torch.complex128)

        re = torch.randn(shape, dtype=torch.float64, device=device, generator=generator)
        im = torch.randn(shape, dtype=torch.float64, device=device, generator=generator)
        c = torch.complex(re, im) * weight.unsqueeze(-1) * ang.view(1, 1, -1, 1)

        evecs = self.evecs.to(torch.complex128)
        u = torch.einsum("lik,...klm->...ilm", evecs, c) * self.minv_sqrt.view(-1, 1, 1)
        return self._mask(u, lmax_ic)

    def sine_initial_condition(self, n_traj=1, lmax_ic=8, n_radial_ic=4, generator=None):
        """Radial sine basis times random angular amplitudes (shell only).

        The sine modes vanish at both walls, so Dirichlet holds exactly. Kept
        because it matches the classic pyspharm-style reference implementations;
        `random_initial_condition` is the general, domain-agnostic one.
        """
        if self.domain != "shell":
            raise ValueError("sine_initial_condition is only defined on the shell")

        device = self.r.device
        length = self.r_out - self.r_in
        p = torch.arange(1, n_radial_ic + 1, dtype=torch.float64, device=device)
        modes = torch.sin(p.view(-1, 1) * math.pi * (self.r - self.r_in) / length)  # (P, n_int)

        uspec = torch.zeros(n_traj, self.n_int, self.lmax, self.mmax, dtype=torch.complex128, device=device)
        for li in range(min(lmax_ic + 1, self.lmax)):
            ang = 1.0 / (1.0 + li) ** 2
            for m in range(min(li + 1, self.mmax)):
                coeff = torch.randn(n_traj, n_radial_ic, dtype=torch.float64, device=device, generator=generator) / p
                prof = coeff @ modes  # (n_traj, n_int)
                re = torch.randn(n_traj, dtype=torch.float64, device=device, generator=generator)
                im = torch.randn(n_traj, dtype=torch.float64, device=device, generator=generator)
                amp = ang * torch.complex(re, im if m > 0 else torch.zeros_like(im))
                uspec[:, :, li, m] = amp.unsqueeze(-1) * prof.to(torch.complex128)
        return uspec

    # -- diagnostics --------------------------------------------------------

    def integrate_grid(self, ugrid: torch.Tensor) -> torch.Tensor:
        """Volume integral over the 3D domain. The mass matrix *is* the radial weight."""
        dlon = 2 * torch.pi / self.nlon
        return torch.sum(ugrid * self.mass.view(-1, 1, 1) * self.quad_weights * dlon, dim=(-3, -2, -1))

    def spectral_energy(self, uspec: torch.Tensor) -> torch.Tensor:
        """The L2(dV) norm of the field, sum_i m_i sum_lm |a_ilm|^2. Non-increasing in t.

        The mass weight is not decoration. Monotonicity is a theorem in *this* norm and
        only this one: it equals ||M^1/2 u||^2, which is what the orthogonal eigenbasis
        contracts by exp(nu t lam) <= 1. The unweighted nodal sum is not a Lyapunov
        functional -- on a geometric grid, where m spans many decades, it can grow.
        """
        return torch.sum(self.mass.view(-1, 1, 1) * torch.abs(uspec) ** 2, dim=(-3, -2, -1))
