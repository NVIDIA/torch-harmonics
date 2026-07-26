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
import torch.nn as nn

import torch_harmonics as th
from torch_harmonics.quadrature import precompute_longitudes

from .poisson_radial import RadialPoissonOperator, contract, radial_grid


class PoissonSolver(nn.Module):
    """
    Poisson solver on (0, inf) x S2 or [R, inf) x S2.

    Solves lap u = f subject to u -> 0 at infinity, using an exact per-degree Green's
    operator in the radial direction and a spherical harmonic transform in the angular
    directions. The form lap V = 4 pi G rho is exposed by :meth:`solve_density`.

    The closed form imposes no artificial boundary condition at the outer truncation
    radius, so the far field stays uncontaminated. For a compactly supported source the
    solution outside the support is exactly the multipole tail, so truncating the grid
    above the support introduces no error in the ground truth.

    Buffers are float64 and spectral quantities complex128, since this is a ground-truth
    generator rather than a trainable model.

    Parameters
    -----------
    nlat : int
        Number of latitude points
    nlon : int
        Number of longitude points
    nr : int
        Number of radial points
    r_in, r_out : float, optional
        Half-line only. Truncation radii of the radial grid, by default 0.1 and 1000.0
    lmax : int, optional
        Maximum l mode, by default None. Note that upstream truncates equiangular grids to
        (nlat + 1) // 2 when this is left unset
    mmax : int, optional
        Maximum m mode, by default None
    grid : str, optional
        Grid type ("legendre-gauss", "lobatto", "equiangular"), by default "legendre-gauss",
        matching SphericalHeatSolver. Gauss quadrature is exact, which matters for a
        ground-truth generator
    domain : str, optional
        Either "half-line" or "exterior", by default "half-line"
    R : float, optional
        Exterior only. Inner radius, i.e. the boundary carrying the Dirichlet data. Required
        for domain="exterior" and rejected otherwise, by default None
    rho_min, rho_max : float, optional
        Exterior only. Bounds on the shifted coordinate rho = r - R, in units of R when
        nondim is set, by default 1e-2 and 1e4
    nondim : bool, optional
        Exterior only. Build the radial grid in rho / R, so the sampled node array is
        identical for every R, by default True
    bc : str, optional
        Inner boundary condition, currently only "dirichlet", by default "dirichlet"
    gravity : float, optional
        Gravitational constant G used by :meth:`solve_density`, by default 1.0
    """

    def __init__(self, nlat, nlon, nr, r_in=0.1, r_out=1000.0, lmax=None, mmax=None, grid="legendre-gauss", domain="half-line", R=None, rho_min=1e-2, rho_max=1e4, nondim=True, bc="dirichlet", gravity=1.0):
        super().__init__()

        # grid parameters
        self.nlat = nlat
        self.nlon = nlon
        self.nr = nr
        self.grid = grid
        self.domain = domain
        self.R = R
        self.nondim = nondim

        # physical constants
        self.register_buffer("gravity", torch.as_tensor(gravity, dtype=torch.float64))

        # SHT
        self.sht = th.RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False)
        self.isht = th.InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False)

        self.lmax = self.sht.lmax
        self.mmax = self.sht.mmax

        # compute gridpoints
        if self.grid == "legendre-gauss":
            cost, quad_weights = th.quadrature.legendre_gauss_weights(self.nlat, -1, 1)
        elif self.grid == "lobatto":
            cost, quad_weights = th.quadrature.lobatto_weights(self.nlat, -1, 1)
        elif self.grid == "equiangular":
            cost, quad_weights = th.quadrature.clenshaw_curtiss_weights(self.nlat, -1, 1)

        quad_weights = quad_weights.reshape(-1, 1)

        # apply cosine transform and flip them
        lats = -torch.arcsin(cost)
        lons = precompute_longitudes(self.nlon)

        # radial grid and the exact radial Green's operator. Each domain takes its own pair
        # of bounds: the half-line is truncated in r, the exterior is gridded in rho = r - R.
        if domain == "half-line":
            if R is not None:
                raise ValueError("R is only meaningful on the exterior domain")
            x, r, w = radial_grid(nr, r_in, r_out, domain=domain)
        else:
            x, r, w = radial_grid(nr, rho_min, rho_max, domain=domain, R=R, nondim=nondim)
        self.radial = RadialPoissonOperator(r, w, self.lmax, domain=domain, R=R, bc=bc)

        # grid ratio, a dilation by q**j is an exact shift of the radial index by j
        self.dx = (x[1] - x[0]).item()
        self.q = math.exp(self.dx)

        l = torch.arange(0, self.lmax).reshape(self.lmax, 1).double()
        l = l.expand(self.lmax, self.mmax)

        # register all
        self.register_buffer("lats", lats)
        self.register_buffer("lons", lons)
        self.register_buffer("x", x)
        self.register_buffer("r", r)
        self.register_buffer("l", l)
        self.register_buffer("quad_weights", quad_weights)

    def extra_repr(self):
        return f"nlat={self.nlat}, nlon={self.nlon}, nr={self.nr},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, domain={self.domain}, R={self.R}"

    @property
    def ctype(self):
        """Complex dtype matching the real dtype of the solver buffers."""
        return torch.complex128 if self.r.dtype == torch.float64 else torch.complex64

    def grid2spec(self, ugrid):
        """Convert spatial data to spectral coefficients. Broadcasts over a leading radial axis."""
        return self.sht(ugrid)

    def spec2grid(self, uspec):
        """Convert spectral coefficients to spatial data. Broadcasts over a leading radial axis."""
        return self.isht(uspec)

    def solve(self, f: torch.Tensor, v0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Solve lap u = f on the grid.

        Parameters
        -----------
        f : torch.Tensor
            Source on the grid, shape (..., nr, nlat, nlon)
        v0 : torch.Tensor, optional
            Exterior only. Inner Dirichlet data, shape (..., nlat, nlon), by default None

        Returns
        -------
        torch.Tensor
            Solution on the grid, shape (..., nr, nlat, nlon)
        """

        v0spec = None if v0 is None else self.grid2spec(v0)
        return self.spec2grid(self.radial(self.grid2spec(f), v0spec))

    def solve_density(self, rho: torch.Tensor, v0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Solve lap V = 4 pi G rho.

        A point mass M at the origin gives V = -G M / r, so that under the dilation action
        V' = a**2 D_a V is the potential of a mass a**3 M.
        """

        return self.solve(4.0 * math.pi * self.gravity * rho, v0)

    def solve_at(self, f: torch.Tensor, r: float, v0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Solve and evaluate on a single sphere of arbitrary radius, shape (..., nlat, nlon).

        The radius need not be a grid node. On the exterior domain the geometric grid in
        rho = r - R never contains rho = 0, so this is the way to reach the boundary itself.
        """

        uspec = contract(self.radial.kernel_row(r), self.grid2spec(f), "lj,...jlm->...lm")

        if v0 is not None:
            lift = torch.exp((self.l[:, : self.mmax] + 1) * (math.log(self.R) - math.log(r)))
            uspec = uspec + self.grid2spec(v0) * lift.to(uspec.dtype)

        return self.spec2grid(uspec)

    def multipole_moments(self, f: torch.Tensor) -> torch.Tensor:
        """Multipole coefficients C_lm of a source given on the grid, shape (..., lmax, mmax)."""

        return self.radial.multipole_moments(self.grid2spec(f))

    def _random_angular_spec(self, shape, l_src=8, decay=1.0, generator=None) -> torch.Tensor:
        """
        Random band-limited complex coefficients of a real field on the sphere.

        Reuses the sqrt(4 pi / (l (l + 1))) normalization of ShallowWaterSolver. The m = 0
        coefficients of a real field are real, and the inverse transform would silently
        discard a nonzero imaginary part there, so it is zeroed to keep the grid field and
        its spectrum consistent.
        """

        device = self.r.device
        lsrc = min(l_src + 1, self.lmax)
        msrc = min(lsrc, self.mmax)

        aspec = torch.zeros(*shape, self.lmax, self.mmax, dtype=self.ctype, device=device)
        scale = math.sqrt(4.0 * math.pi / lsrc / (lsrc + 1))
        aspec[..., :lsrc, :msrc] = scale * torch.randn(aspec[..., :lsrc, :msrc].shape, dtype=self.ctype, device=device, generator=generator)

        # decaying spectrum, lower-triangular storage, real m = 0 column
        l = torch.arange(0, self.lmax, dtype=self.r.dtype, device=device).reshape(-1, 1)
        aspec = aspec * (1.0 + l).pow(-decay).to(self.ctype)
        aspec = aspec * torch.tril(torch.ones(self.lmax, self.mmax, dtype=self.r.dtype, device=device)).to(self.ctype)
        aspec[..., 0] = aspec[..., 0].real.to(self.ctype)

        return aspec

    def random_source(self, n_traj=1, nblobs=3, l_src=8, decay=1.0, margin=0.25, width=(0.03, 0.10), seed=None, spectral=False) -> torch.Tensor:
        """
        Random, compactly supported, multi-scale source of shape (n_traj, nr, nlat, nlon).

        A sum of nblobs terms, each a radial bump in x = log r times a random band-limited
        angular field. The bumps use a C-infinity compact bump rather than a Gaussian, so the
        support is genuinely compact and the solution outside it is exactly the multipole tail.

        The support is confined to the middle of the radial grid. That margin is what gives the
        dilation tests room to shift the source along the grid, so it is a fraction of the log
        range rather than a fixed number of nodes.

        Parameters
        -----------
        n_traj : int, optional
            Number of independent samples, kept as a leading axis to match
            SphericalHeatSolver.random_initial_condition, by default 1
        nblobs : int, optional
            Number of blobs to superpose, by default 3
        l_src : int, optional
            Angular band limit of the source, by default 8
        decay : float, optional
            Exponent of the (1 + l) decay applied to the angular spectrum, by default 1.0
        margin : float, optional
            Fraction of the log range kept free at each end, by default 0.25
        width : tuple of float, optional
            Range of blob half-widths as a fraction of the log range, by default (0.03, 0.10)
        seed : int, optional
            Seed for a local random generator, by default None
        spectral : bool, optional
            Return spectral coefficients of shape (n_traj, nr, lmax, mmax) instead, by default False
        """

        device = self.r.device
        dtype = self.r.dtype

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        x = self.x
        span = (x[-1] - x[0]).item()
        xlo = x[0].item() + margin * span
        xhi = x[-1].item() - margin * span

        if xhi <= xlo:
            raise ValueError(f"margin={margin} leaves no room for the source support")

        shape = (n_traj, nblobs)
        half = width[0] * span + (width[1] - width[0]) * span * torch.rand(shape, dtype=dtype, device=device, generator=generator)
        half = torch.clamp(half, max=0.5 * (xhi - xlo))
        centers = xlo + half + (xhi - xlo - 2 * half) * torch.rand(shape, dtype=dtype, device=device, generator=generator)

        # C-infinity compact bump, normalized to unit peak
        t = (x.reshape(1, 1, -1) - centers.unsqueeze(-1)) / half.unsqueeze(-1)
        inside = t.abs() < 1.0
        radial = torch.zeros_like(t)
        radial[inside] = torch.exp(1.0 - 1.0 / (1.0 - t[inside] ** 2))

        aspec = self._random_angular_spec(shape, l_src=l_src, decay=decay, generator=generator)

        # f_lm(r) = sum_b A_lm^(b) g_b(r)
        fspec = torch.einsum("nblm,nbk->nklm", aspec, radial.to(aspec.dtype))

        return fspec if spectral else self.spec2grid(fspec)

    def random_boundary_data(self, n_traj=1, l_src=8, decay=1.0, seed=None, spectral=False) -> torch.Tensor:
        """Random band-limited Dirichlet data on the inner sphere, shape (n_traj, nlat, nlon)."""

        if self.domain != "exterior":
            raise ValueError("random_boundary_data is only defined on the exterior domain")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.r.device)
            generator.manual_seed(seed)

        v0spec = self._random_angular_spec((n_traj,), l_src=l_src, decay=decay, generator=generator)

        return v0spec if spectral else self.spec2grid(v0spec)
