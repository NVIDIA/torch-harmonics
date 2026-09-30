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
import torch.nn as nn

import torch_harmonics as th
from torch_harmonics.fft import rfft
from torch_harmonics.grid import require_regular_grid


class ShallowWaterSolver(nn.Module):
    """
    Shallow Water Equations (SWE) solver class for spherical geometry.

    Interface inspired by pyspharm and SHTns. Solves the shallow water equations
    on a rotating sphere using spectral methods.

    Parameters
    ----------
    grid : RegularGridS2
        Grid the fields are sampled on, e.g. ``as_grid("equiangular", nlat=256, nlon=512)``.
        Its quadrature must support an SHT, so ``"equiangular"``, ``"legendre-gauss"`` or
        ``"lobatto"``.
    dt : float
        Time step size
    lmax : int, optional
        Maximum l mode for spherical harmonics, by default None
    mmax : int, optional
        Maximum m mode for spherical harmonics, by default None
    radius : float, optional
        Radius of the sphere in meters, by default 6.37122E6 (Earth radius)
    omega : float, optional
        Angular velocity of rotation in rad/s, by default 7.292E-5 (Earth)
    gravity : float, optional
        Gravitational acceleration in m/s², by default 9.80616
    havg : float, optional
        Average height in meters, by default 10.e3
    hamp : float, optional
        Height amplitude in meters, by default 120.
    """

    def __init__(self, grid, dt, lmax=None, mmax=None, radius=6.37122e6, omega=7.292e-5, gravity=9.80616, havg=10.0e3, hamp=120.0):
        super().__init__()

        # time stepping param
        self.dt = dt

        # grid parameters
        self.grid = require_regular_grid(grid)
        self.nlat, self.nlon = self.grid.shape

        # physical sonstants
        self.register_buffer("radius", torch.as_tensor(radius, dtype=torch.float64))
        self.register_buffer("omega", torch.as_tensor(omega, dtype=torch.float64))
        self.register_buffer("gravity", torch.as_tensor(gravity, dtype=torch.float64))
        self.register_buffer("havg", torch.as_tensor(havg, dtype=torch.float64))
        self.register_buffer("hamp", torch.as_tensor(hamp, dtype=torch.float64))

        # SHT
        self.sht = th.RealSHT(self.grid, lmax=lmax, mmax=mmax, csphase=False)
        self.isht = th.InverseRealSHT(self.grid, lmax=lmax, mmax=mmax, csphase=False)
        self.vsht = th.RealVectorSHT(self.grid, lmax=lmax, mmax=mmax, csphase=False)
        self.ivsht = th.InverseRealVectorSHT(self.grid, lmax=lmax, mmax=mmax, csphase=False)

        # grid points and weights, both from the descriptor, so they are ordered alike --
        # north to south. The weights are per-point solid angles, so integrate_grid does
        # not have to reconstruct the longitudinal factor; kept in (nlat, nlon) so the
        # polar_opt slicing indexes rings on axis -2.
        quad_weights = self.grid.quad_weights.reshape(self.nlat, self.nlon)
        lats = self.grid.lats
        lons = self.grid.lons()

        self.lmax = self.sht.lmax
        self.mmax = self.sht.mmax

        # compute the laplace and inverse laplace operators
        l = torch.arange(0, self.lmax).reshape(self.lmax, 1).double()
        l = l.expand(self.lmax, self.mmax)
        # the laplace operator acting on the coefficients is given by - l (l + 1)
        lap = -l * (l + 1) / self.radius**2
        invlap = -self.radius**2 / l / (l + 1)
        invlap[0] = 0.0

        # compute coriolis force
        coriolis = 2 * self.omega * torch.sin(lats).reshape(self.nlat, 1)

        # hyperdiffusion
        hyperdiff = torch.exp(torch.asarray((-self.dt / 2 / 3600.0) * (lap / lap[-1, 0]) ** 4))

        # register all
        self.register_buffer("lats", lats)
        self.register_buffer("lons", lons)
        self.register_buffer("l", l)
        self.register_buffer("lap", lap)
        self.register_buffer("invlap", invlap)
        self.register_buffer("coriolis", coriolis)
        self.register_buffer("hyperdiff", hyperdiff)
        # non-persistent: derived from the grid descriptor, so there is nothing to restore, and
        # its layout changed with the descriptor (per-point, summing to 4*pi)
        self.register_buffer("quad_weights", quad_weights, persistent=False)

    def grid2spec(self, ugrid):
        """Convert spatial data to spectral coefficients."""
        return self.sht(ugrid)

    def spec2grid(self, uspec):
        """Convert spectral coefficients to spatial data."""
        return self.isht(uspec)

    def vrtdivspec(self, ugrid):
        """Compute vorticity and divergence from velocity field."""
        vrtdivspec = self.lap * self.radius * self.vsht(ugrid)
        return vrtdivspec

    def divspec(self, ugrid):
        """Compute only the divergence from velocity field, i.e. vrtdivspec(ugrid)[1] without the vorticity contractions."""
        x = rfft(ugrid, nmodes=self.vsht.mmax, dim=-1, norm="forward").transpose(-1, -2)
        x_re = x.real.contiguous()
        x_im = x.imag.contiguous()
        w0 = self.vsht.weights[0].to(x_re.dtype)
        w1 = self.vsht.weights[1].to(x_re.dtype)
        # same contractions as the toroidal component in RealVectorSHT.forward
        t_re = -torch.einsum("...mk,mlk->...lm", x_im[..., 0, :, :], w1) - torch.einsum("...mk,mlk->...lm", x_re[..., 1, :, :], w0)
        t_im = torch.einsum("...mk,mlk->...lm", x_re[..., 0, :, :], w1) - torch.einsum("...mk,mlk->...lm", x_im[..., 1, :, :], w0)
        return self.lap * self.radius * torch.complex(t_re, t_im)

    def getuv(self, vrtdivspec):
        """Compute wind vector from spectral coefficients of vorticity and divergence."""
        return self.ivsht(self.invlap * vrtdivspec / self.radius)

    def gethuv(self, uspec):
        """Compute height and wind vector from spectral coefficients."""
        hgrid = self.spec2grid(uspec[:1])
        uvgrid = self.getuv(uspec[1:])
        return torch.cat((hgrid, uvgrid), dim=-3)

    def potential_vorticity(self, uspec):
        """Compute potential vorticity from spectral coefficients."""
        ugrid = self.spec2grid(uspec)
        pvrt = (0.5 * self.havg * self.gravity / self.omega) * (ugrid[1] + self.coriolis) / ugrid[0]
        return pvrt

    def dimensionless(self, uspec):
        """Remove dimensions from variables for dimensionless analysis."""
        uspec[0] = (uspec[0] - self.havg * self.gravity) / self.hamp / self.gravity
        # vorticity is measured in 1/s so we normalize using sqrt(g h) / r
        uspec[1:] = uspec[1:] * self.radius / torch.sqrt(self.gravity * self.havg)
        return uspec

    def dudtspec(self, uspec):
        """Compute time derivatives from solution represented in spectral coefficients."""
        dudtspec = torch.zeros_like(uspec)

        # compute the derivatives - this should be incorporated into the solver.
        # only phi = ugrid[0] and vrt = ugrid[1] are needed on the grid, so the divergence is not transformed
        ugrid = self.spec2grid(uspec[:2])
        uvgrid = self.getuv(uspec[1:])

        tmp = uvgrid * (ugrid[1] + self.coriolis)
        tmpspec = self.vrtdivspec(tmp)
        dudtspec[2] = tmpspec[0]
        dudtspec[1] = -1 * tmpspec[1]

        # only the divergence of the geopotential flux is needed
        dudtspec[0] = -1 * self.divspec(uvgrid * ugrid[0])

        tmpspec = self.grid2spec(ugrid[0] + 0.5 * (uvgrid[0] ** 2 + uvgrid[1] ** 2))
        dudtspec[2] = dudtspec[2] - self.lap * tmpspec

        return dudtspec

    def galewsky_initial_condition(self):
        """Initialize non-linear barotropically unstable shallow water test case."""
        device = self.lap.device

        umax = 80.0
        phi0 = torch.asarray(torch.pi / 7.0, device=device)
        phi1 = torch.asarray(0.5 * torch.pi - phi0, device=device)
        phi2 = 0.25 * torch.pi
        en = torch.exp(torch.asarray(-4.0 / (phi1 - phi0) ** 2, device=device))
        alpha = 1.0 / 3.0
        beta = 1.0 / 15.0

        lats, lons = torch.meshgrid(self.lats, self.lons, indexing="ij")

        mask = torch.logical_and(lats > phi0, lats < phi1)
        # use safe values outside the band to prevent exp from overflowing
        product = torch.where(mask, (lats - phi0) * (lats - phi1), -torch.ones_like(lats))
        u1 = (umax / en) * torch.exp(1.0 / product)
        ugrid = torch.where(mask, u1, torch.zeros(self.nlat, self.nlon, device=device))
        vgrid = torch.zeros((self.nlat, self.nlon), device=device)
        hbump = self.hamp * torch.cos(lats) * torch.exp(-(((lons - torch.pi) / alpha) ** 2)) * torch.exp(-((phi2 - lats) ** 2) / beta)

        # intial velocity field
        ugrid = torch.stack((ugrid, vgrid))
        # intial vorticity/divergence field
        vrtdivspec = self.vrtdivspec(ugrid)
        vrtdivgrid = self.spec2grid(vrtdivspec)

        # solve balance eqn to get initial zonal geopotential with a localized bump (not balanced).
        tmp = ugrid * (vrtdivgrid + self.coriolis)
        tmpspec = self.vrtdivspec(tmp)
        tmpspec[1] = self.grid2spec(0.5 * torch.sum(ugrid**2, dim=0))
        phispec = self.invlap * tmpspec[0] - tmpspec[1] + self.grid2spec(self.gravity * (self.havg + hbump))

        # assemble solution
        uspec = torch.zeros(3, self.lmax, self.mmax, dtype=vrtdivspec.dtype, device=device)
        uspec[0] = phispec
        uspec[1:] = vrtdivspec

        return torch.tril(uspec)

    def random_initial_condition(self, mach=0.1) -> torch.Tensor:
        """Generate random initial condition on the sphere."""
        device = self.lap.device
        ctype = torch.complex128 if self.lap.dtype == torch.float64 else torch.complex64

        # mach number relative to wave speed
        llimit = mlimit = 120

        # hgrid = self.havg + hamp * torch.randn(self.nlat, self.nlon, device=device, dtype=dtype)
        # ugrid = uamp * torch.randn(self.nlat, self.nlon, device=device, dtype=dtype)
        # vgrid = vamp * torch.randn(self.nlat, self.nlon, device=device, dtype=dtype)
        # ugrid = torch.stack((ugrid, vgrid))

        # initial geopotential
        uspec = torch.zeros(3, self.lmax, self.mmax, dtype=ctype, device=self.lap.device)
        uspec[:, :llimit, :mlimit] = torch.sqrt(torch.tensor(4 * torch.pi / llimit / (llimit + 1), device=device, dtype=ctype)) * torch.randn_like(uspec[:, :llimit, :mlimit])

        uspec[0] = self.gravity * self.hamp * uspec[0]
        uspec[0, 0, 0] += torch.sqrt(torch.tensor(4 * torch.pi, device=device, dtype=ctype)) * self.havg * self.gravity
        uspec[1:] = mach * uspec[1:] * torch.sqrt(self.gravity * self.havg) / self.radius
        # uspec[1:] = self.vrtdivspec(self.spec2grid(uspec[1:]) * torch.cos(self.lats.reshape(-1, 1)))

        # # intial velocity field
        # ugrid = uamp * self.spec2grid(uspec[1])
        # vgrid = vamp * self.spec2grid(uspec[2])
        # ugrid = torch.stack((ugrid, vgrid))

        # # intial vorticity/divergence field
        # vrtdivspec = self.vrtdivspec(ugrid)
        # vrtdivgrid = self.spec2grid(vrtdivspec)

        # # solve balance eqn to get initial zonal geopotential with a localized bump (not balanced).
        # tmp = ugrid * (vrtdivgrid + self.coriolis)
        # tmpspec = self.vrtdivspec(tmp)
        # tmpspec[1] = self.grid2spec(0.5 * torch.sum(ugrid**2, dim=0))
        # phispec = self.invlap*tmpspec[0] - tmpspec[1] + self.grid2spec(self.gravity * hgrid)

        # # assemble solution
        # uspec = torch.zeros(3, self.lmax, self.mmax, dtype=phispec.dtype, device=device)
        # uspec[0] = phispec
        # uspec[1:] = vrtdivspec

        return torch.tril(uspec)

    def forward(self, uspec: torch.Tensor, dnow: Optional[torch.Tensor] = None, dold: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Advance the solution by a single third-order Adams-Bashforth step.

        This is the unit to compile: ``solver.compile()`` compiles it for this instance, and
        ``timestep`` then runs the compiled step. With ``mode="reduce-overhead"`` (CUDA graphs)
        each call overwrites the outputs of the previous one, so the caller has to clone the
        returned tensors before feeding them back in.

        Parameters
        ----------
        uspec : torch.Tensor
            Current solution in spectral coefficients
        dnow : torch.Tensor, optional
            Tendency from the previous step. None at the first step, which reduces the scheme to forward Euler.
        dold : torch.Tensor, optional
            Tendency from two steps back. None within the first two steps, which reduces the scheme to
            forward Euler and second-order Adams-Bashforth, respectively.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Updated solution and the tendency computed in this step, which becomes dnow of the next step
        """
        dnew = self.dudtspec(uspec)

        # forward euler, then 2nd-order adams-bashforth time steps to start.
        dnow = dnew if dnow is None else dnow
        dold = dnew if dold is None else dold

        # update vort,div,phiv with third-order adams-bashforth.
        uspec = uspec + self.dt * ((23.0 / 12.0) * dnew - (16.0 / 12.0) * dnow + (5.0 / 12.0) * dold)

        # implicit hyperdiffusion for vort and div. Out of place, as inductor cannot generate the
        # complex slice assignment uspec[1:] = ... in triton.
        uspec = torch.cat((uspec[:1], self.hyperdiff * uspec[1:]))

        return uspec, dnew

    def timestep(self, uspec: torch.Tensor, nsteps: int) -> torch.Tensor:
        """Integrate the solution using Adams-Bashforth / forward Euler for nsteps steps."""
        dnow = dold = None
        for _ in range(nsteps):
            uspec, dnew = self(uspec, dnow, dold)
            dnow, dold = dnew, dnow

        return uspec

    def integrate_grid(self, ugrid, dimensionless=False, polar_opt=0):
        """Integrate the solution on the grid."""
        # no dlon here: self.quad_weights is the per-point solid angle and already
        # carries the longitudinal factor, per ring
        radius = 1 if dimensionless else self.radius
        if polar_opt > 0:
            out = torch.sum(ugrid[..., polar_opt:-polar_opt, :] * self.quad_weights[polar_opt:-polar_opt] * radius**2, dim=(-2, -1))
        else:
            out = torch.sum(ugrid * self.quad_weights * radius**2, dim=(-2, -1))
        return out

    def plot_griddata(self, data, fig, cmap="twilight_shifted", vmax=None, vmin=None, projection="3d", title=None, antialiased=False):
        """Plotting routine for data on the grid. Requires cartopy for 3d plots."""
        import matplotlib.pyplot as plt

        lons = self.lons.squeeze() - torch.pi
        lats = self.lats.squeeze()

        # matplotlib needs host memory, whatever device the solver runs on (CUDA, MPS, ...)
        data = data.detach().cpu()
        lons = lons.cpu()
        lats = lats.cpu()

        Lons, Lats = np.meshgrid(lons, lats)

        if projection == "mollweide":

            # ax = plt.gca(projection=projection)
            ax = fig.add_subplot(projection=projection)
            im = ax.pcolormesh(Lons, Lats, data, cmap=cmap, vmax=vmax, vmin=vmin)
            # ax.set_title("Elevation map of mars")
            ax.grid(True)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.colorbar(im, orientation="horizontal")
            plt.title(title)

        elif projection == "3d":

            import cartopy.crs as ccrs

            proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=25.0)

            # ax = plt.gca(projection=proj, frameon=True)
            ax = fig.add_subplot(projection=proj)
            Lons = Lons * 180 / math.pi
            Lats = Lats * 180 / math.pi

            # contour data over the map.
            im = ax.pcolormesh(Lons, Lats, data, cmap=cmap, transform=ccrs.PlateCarree(), antialiased=antialiased, vmax=vmax, vmin=vmin)
            plt.title(title, y=1.05)

        elif projection == "robinson":

            import cartopy.crs as ccrs

            proj = ccrs.Robinson(central_longitude=0.0)

            # ax = plt.gca(projection=proj, frameon=True)
            ax = fig.add_subplot(projection=proj)
            Lons = Lons * 180 / math.pi
            Lats = Lats * 180 / math.pi

            # contour data over the map.
            im = ax.pcolormesh(Lons, Lats, data, cmap=cmap, transform=ccrs.PlateCarree(), antialiased=antialiased, vmax=vmax, vmin=vmin)
            plt.title(title, y=1.05)

        else:
            raise NotImplementedError

        return im

    def plot_specdata(self, data, fig, **kwargs):
        return self.plot_griddata(self.isht(data), fig, **kwargs)
