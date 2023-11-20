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

import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs

def plot_sphere(data,
                fig=None,
                cmap="RdBu",
                title=None,
                colorbar=False,
                coastlines=False,
                central_latitude=20,
                central_longitude=20,
                lon=None,
                lat=None,
                **kwargs):
    if fig == None:
        fig = plt.figure()

    nlat = data.shape[-2]
    nlon = data.shape[-1]
    if lon is None:
        lon = np.linspace(0, 2*np.pi, nlon)
    if lat is None:
        lat = np.linspace(np.pi/2., -np.pi/2., nlat)
    Lon, Lat = np.meshgrid(lon, lat)

    proj = ccrs.Orthographic(central_longitude=central_longitude, central_latitude=central_latitude)
    # proj = ccrs.Mollweide(central_longitude=central_longitude)

    ax = fig.add_subplot(projection=proj)
    Lon = Lon*180/np.pi
    Lat = Lat*180/np.pi

    # contour data over the map.
    im = ax.pcolormesh(Lon, Lat, data, cmap=cmap, transform=ccrs.PlateCarree(), antialiased=False, **kwargs)
    if coastlines:
        ax.add_feature(cartopy.feature.COASTLINE, edgecolor='white', facecolor='none', linewidth=1.5)
    if colorbar:
        plt.colorbar(im)
    plt.title(title, y=1.05)

    return im

def plot_data(data,
              fig=None,
              projection=None,
              cmap="RdBu",
              title=None,
              colorbar=False,
              lon=None,
              lat=None,
              **kwargs):
    if fig == None:
        fig = plt.figure()
    
    nlat = data.shape[-2]
    nlon = data.shape[-1]
    if lon is None:
        lon = np.linspace(0, 2*np.pi, nlon)
    if lat is None:
        lat = np.linspace(np.pi/2., -np.pi/2., nlat)
    Lon, Lat = np.meshgrid(lon, lat)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    im = ax.pcolormesh(Lon, Lat, data, cmap=cmap, **kwargs)
    
    if colorbar:
        plt.colorbar(im)
    plt.title(title, y=1.05)

    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return im