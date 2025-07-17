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


import torch
import torch.nn as nn
import torch_harmonics as th
from torch_harmonics.quadrature import _precompute_longitudes

import math
import numpy as np


class ShallowWaterSolver(nn.Module):
    """
    Shallow Water Equations (SWE) solver class for spherical geometry.
    
    Interface inspired by pyspharm and SHTns. Solves the shallow water equations
    on a rotating sphere using spectral methods.
    
    Parameters
    -----------
    nlat : int
        Number of latitude points
    nlon : int
        Number of longitude points
    dt : float
        Time step size
    lmax : int, optional
        Maximum l mode for spherical harmonics, by default None
    mmax : int, optional
        Maximum m mode for spherical harmonics, by default None
    grid : str, optional
        Grid type ("equiangular", "legendre-gauss", "lobatto"), by default "equiangular"
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

    def __init__(self, nlat, nlon, dt, lmax=None, mmax=None, grid="equiangular", radius=6.37122E6, \
                 omega=7.292E-5, gravity=9.80616, havg=10.e3, hamp=120.):
        super().__init__()

        # time stepping param
        self.dt = dt

        # grid parameters
        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid

        # physical sonstants
        self.register_buffer('radius', torch.as_tensor(radius, dtype=torch.float64))
        self.register_buffer('omega', torch.as_tensor(omega, dtype=torch.float64))
        self.register_buffer('gravity', torch.as_tensor(gravity, dtype=torch.float64))
        self.register_buffer('havg', torch.as_tensor(havg, dtype=torch.float64))
        self.register_buffer('hamp', torch.as_tensor(hamp, dtype=torch.float64))

        # SHT
        self.sht = th.RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False)
        self.isht = th.InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False)
        self.vsht = th.RealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False)
        self.ivsht = th.InverseRealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False)

        self.lmax = lmax or self.sht.lmax
        self.mmax = lmax or self.sht.mmax

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
        lons = _precompute_longitudes(self.nlon)

        self.lmax = self.sht.lmax
        self.mmax = self.sht.mmax

        # compute the laplace and inverse laplace operators
        l = torch.arange(0, self.lmax).reshape(self.lmax, 1).double()
        l = l.expand(self.lmax, self.mmax)
        # the laplace operator acting on the coefficients is given by - l (l + 1)
        lap = - l * (l + 1) / self.radius**2
        invlap = - self.radius**2 / l / (l + 1)
        invlap[0] = 0.

        # compute coriolis force
        coriolis = 2 * self.omega * torch.sin(lats).reshape(self.nlat, 1)

        # hyperdiffusion
        hyperdiff = torch.exp(torch.asarray((-self.dt / 2 / 3600.)*(lap / lap[-1, 0])**4))

        # register all
        self.register_buffer('lats', lats)
        self.register_buffer('lons', lons)
        self.register_buffer('l', l)
        self.register_buffer('lap', lap)
        self.register_buffer('invlap', invlap)
        self.register_buffer('coriolis', coriolis)
        self.register_buffer('hyperdiff', hyperdiff)
        self.register_buffer('quad_weights', quad_weights)

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

    def getuv(self, vrtdivspec):
        """Compute wind vector from spectral coefficients of vorticity and divergence."""
        return self.ivsht( self.invlap * vrtdivspec / self.radius)

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

        # compute the derivatives - this should be incorporated into the solver:
        ugrid = self.spec2grid(uspec)
        uvgrid = self.getuv(uspec[1:])

        # phi = ugrid[0]
        # vrtdiv = ugrid[1:]

        tmp = uvgrid * (ugrid[1] + self.coriolis)
        tmpspec = self.vrtdivspec(tmp)
        dudtspec[2] = tmpspec[0]
        dudtspec[1] = -1 * tmpspec[1]

        tmp = uvgrid * ugrid[0]
        tmp = self.vrtdivspec(tmp)
        dudtspec[0] = -1 * tmp[1]

        tmpspec = self.grid2spec(ugrid[0] + 0.5 * (uvgrid[0]**2 + uvgrid[1]**2))
        dudtspec[2] = dudtspec[2] - self.lap * tmpspec

        return dudtspec

    def galewsky_initial_condition(self):
        """Initialize non-linear barotropically unstable shallow water test case."""
        device = self.lap.device

        umax = 80.
        phi0 = torch.asarray(torch.pi / 7., device=device)
        phi1 = torch.asarray(0.5 * torch.pi - phi0, device=device)
        phi2 = 0.25 * torch.pi
        en = torch.exp(torch.asarray(-4.0 / (phi1 - phi0)**2, device=device))
        alpha = 1. / 3.
        beta = 1. / 15.

        lats, lons = torch.meshgrid(self.lats, self.lons)

        u1 = (umax/en)*torch.exp(1./((lats-phi0)*(lats-phi1)))
        ugrid = torch.where(torch.logical_and(lats < phi1, lats > phi0), u1, torch.zeros(self.nlat, self.nlon, device=device))
        vgrid = torch.zeros((self.nlat, self.nlon), device=device)
        hbump = self.hamp * torch.cos(lats) * torch.exp(-((lons-torch.pi)/alpha)**2) * torch.exp(-(phi2-lats)**2/beta)

        # intial velocity field
        ugrid = torch.stack((ugrid, vgrid))
        # intial vorticity/divergence field
        vrtdivspec = self.vrtdivspec(ugrid)
        vrtdivgrid = self.spec2grid(vrtdivspec)

        # solve balance eqn to get initial zonal geopotential with a localized bump (not balanced).
        tmp = ugrid * (vrtdivgrid + self.coriolis)
        tmpspec = self.vrtdivspec(tmp)
        tmpspec[1] = self.grid2spec(0.5 * torch.sum(ugrid**2, dim=0))
        phispec = self.invlap*tmpspec[0] - tmpspec[1] + self.grid2spec(self.gravity*(self.havg + hbump))

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
        uspec[:, :llimit, :mlimit] = torch.sqrt(torch.tensor(4 * torch.pi / llimit / (llimit+1), device=device, dtype=ctype)) * torch.randn_like(uspec[:, :llimit, :mlimit])

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

    def timestep(self, uspec: torch.Tensor, nsteps: int) -> torch.Tensor:
        """Integrate the solution using Adams-Bashforth / forward Euler for nsteps steps."""
        dudtspec = torch.zeros(3, 3, self.lmax, self.mmax, dtype=uspec.dtype, device=uspec.device)

        # pointers to indicate the most current result
        inew = 0
        inow = 1
        iold = 2

        for iter in range(nsteps):
            dudtspec[inew] = self.dudtspec(uspec)

            # update vort,div,phiv with third-order adams-bashforth.
            # forward euler, then 2nd-order adams-bashforth time steps to start.
            if iter == 0:
                dudtspec[inow] = dudtspec[inew]
                dudtspec[iold] = dudtspec[inew]
            elif iter == 1:
                dudtspec[iold] = dudtspec[inew]

            uspec = uspec + self.dt*( (23./12.) * dudtspec[inew] - (16./12.) * dudtspec[inow] + (5./12.) * dudtspec[iold] )

            # implicit hyperdiffusion for vort and div.
            uspec[1:] = self.hyperdiff * uspec[1:]

            # cycle through the indices
            inew = (inew - 1) % 3
            inow = (inow - 1) % 3
            iold = (iold - 1) % 3

        return uspec

    def integrate_grid(self, ugrid, dimensionless=False, polar_opt=0):
        """Integrate the solution on the grid."""
        dlon = 2 * torch.pi / self.nlon
        radius = 1 if dimensionless else self.radius
        if polar_opt > 0:
            out = torch.sum(ugrid[..., polar_opt:-polar_opt, :] * self.quad_weights[polar_opt:-polar_opt] * dlon * radius**2, dim=(-2, -1))
        else:
            out = torch.sum(ugrid * self.quad_weights * dlon * radius**2, dim=(-2, -1))
        return out


    def plot_griddata(self, data, fig, cmap='twilight_shifted', vmax=None, vmin=None, projection='3d', title=None, antialiased=False):
        """Plotting routine for data on the grid. Requires cartopy for 3d plots."""
        import matplotlib.pyplot as plt

        lons = self.lons.squeeze() - torch.pi
        lats = self.lats.squeeze()

        if data.is_cuda:
            data = data.cpu()
            lons = lons.cpu()
            lats = lats.cpu()

        Lons, Lats = np.meshgrid(lons, lats)

        if projection == 'mollweide':

            #ax = plt.gca(projection=projection)
            ax = fig.add_subplot(projection=projection)
            im = ax.pcolormesh(Lons, Lats, data, cmap=cmap, vmax=vmax, vmin=vmin)
            # ax.set_title("Elevation map of mars")
            ax.grid(True)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.colorbar(im, orientation='horizontal')
            plt.title(title)

        elif projection == '3d':

            import cartopy.crs as ccrs

            proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=25.0)

            #ax = plt.gca(projection=proj, frameon=True)
            ax = fig.add_subplot(projection=proj)
            Lons = Lons*180/math.pi
            Lats = Lats*180/math.pi

            # contour data over the map.
            im = ax.pcolormesh(Lons, Lats, data, cmap=cmap, transform=ccrs.PlateCarree(), antialiased=antialiased, vmax=vmax, vmin=vmin)
            plt.title(title, y=1.05)

        elif projection == 'robinson':

            import cartopy.crs as ccrs

            proj = ccrs.Robinson(central_longitude=0.0)

            #ax = plt.gca(projection=proj, frameon=True)
            ax = fig.add_subplot(projection=proj)
            Lons = Lons*180/math.pi
            Lats = Lats*180/math.pi

            # contour data over the map.
            im = ax.pcolormesh(Lons, Lats, data, cmap=cmap, transform=ccrs.PlateCarree(), antialiased=antialiased, vmax=vmax, vmin=vmin)
            plt.title(title, y=1.05)

        else:
            raise NotImplementedError

        return im

    def plot_specdata(self, data, fig, **kwargs):
        return self.plot_griddata(self.isht(data), fig, **kwargs)
