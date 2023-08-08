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
import torch_harmonics as harmonics

import numpy as np


class SphereSolver(nn.Module):
    """
    Solver class on the sphere. Can solve the following PDEs:
    - Allen-Cahn eq
    """

    def __init__(self, nlat, nlon, dt, lmax=None, mmax=None, grid='legendre-gauss', radius=1.0, coeff=0.001):
        super().__init__()

        # time stepping param
        self.dt = dt

        # grid parameters
        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid

        # physical sonstants
        self.register_buffer('radius', torch.as_tensor(radius, dtype=torch.float64))
        self.register_buffer('coeff', torch.as_tensor(coeff, dtype=torch.float64))

        # SHT
        self.sht = harmonics.RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False)
        self.isht = harmonics.InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False)

        self.lmax = lmax or self.sht.lmax
        self.mmax = lmax or self.sht.mmax

        # compute gridpoints
        if self.grid == "legendre-gauss":
            cost, _ = harmonics.quadrature.legendre_gauss_weights(self.nlat, -1, 1)
        elif self.grid == "lobatto":
            cost, _ = harmonics.quadrature.lobatto_weights(self.nlat, -1, 1)
        elif self.grid == "equiangular":
            cost, _ = harmonics.quadrature.clenshaw_curtiss_weights(self.nlat, -1, 1)

        # apply cosine transform and flip them
        lats = -torch.as_tensor(np.arcsin(cost))
        lons = torch.linspace(0, 2*np.pi, self.nlon+1, dtype=torch.float64)[:nlon]

        self.lmax = self.sht.lmax
        self.mmax = self.sht.mmax

        l = torch.arange(0, self.lmax).reshape(self.lmax, 1).cdouble()
        l = l.expand(self.lmax, self.mmax)
        # the laplace operator acting on the coefficients is given by l (l + 1)
        lap = - l * (l + 1) / self.radius**2
        invlap = - self.radius**2 / l / (l + 1)
        invlap[0] = 0.

        # register all
        self.register_buffer('lats', lats)
        self.register_buffer('lons', lons)
        self.register_buffer('l', l)
        self.register_buffer('lap', lap)
        self.register_buffer('invlap', invlap)

    def grid2spec(self, u):
        """spectral coefficients from spatial data"""
        
        return self.sht(u)

    def spec2grid(self, uspec):
        """spatial data from spectral coefficients"""

        return self.isht(uspec)

    def dudtspec(self, uspec, pde='allen-cahn'):

        if pde == 'allen-cahn':
            ugrid = self.spec2grid(uspec)
            u3spec  = self.grid2spec(ugrid**3)
            dudtspec = self.coeff*self.lap*uspec + uspec - u3spec
        elif pde == 'ginzburg-landau':
            ugrid = self.spec2grid(uspec)
            u3spec  = self.grid2spec(ugrid**3)
            dudtspec = uspec + (1. + 2.j)*self.coeff*self.lap*uspec - (1. + 2.j)*u3spec
        else:
            NotImplementedError
        
        return dudtspec

    def randspec(self):
        """random data on the sphere"""

        rspec = torch.randn_like(self.lap) / 4 / torch.pi
        return rspec


    def plot_griddata(self, data, fig, cmap='twilight_shifted', vmax=None, vmin=None, projection='3d', title=None, antialiased=False):
        """
        plotting routine for data on the grid. Requires cartopy for 3d plots.
        """
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
            Lons = Lons*180/np.pi
            Lats = Lats*180/np.pi

            # contour data over the map.
            im = ax.pcolormesh(Lons, Lats, data, cmap=cmap, transform=ccrs.PlateCarree(), antialiased=antialiased, vmax=vmax, vmin=vmin)
            plt.title(title, y=1.05)

        else:
            raise NotImplementedError

        return im

    def plot_specdata(self, data, fig, **kwargs):
        return self.plot_griddata(self.isht(data), fig, **kwargs)