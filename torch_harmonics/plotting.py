# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
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
import os

# guarded imports
try:
    import matplotlib.pyplot as plt
except ImportError as err:
    plt = None

try:
    import cartopy
    import cartopy.crs as ccrs
except ImportError as err:
    cartopy = None
    ccrs = None


def check_plotting_dependencies():
    if plt is None:
        raise ImportError("matplotlib is required for plotting functions. Install it with 'pip install matplotlib'")
    if cartopy is None:
        raise ImportError("cartopy is required for map plotting. Install it with 'pip install cartopy'")


def get_projection(
    projection,
    central_latitude=0,
    central_longitude=0,
):
    """
    Get a cartopy projection object for map plotting.
    
    Parameters
    -----------
    projection : str
        Projection type ("orthographic", "robinson", "platecarree", "mollweide")
    central_latitude : float, optional
        Central latitude for the projection, by default 0
    central_longitude : float, optional
        Central longitude for the projection, by default 0
        
    Returns
    -------
    cartopy.crs.Projection
        Cartopy projection object
        
    Raises
    ------
    ValueError
        If projection type is not supported
    """
    if projection == "orthographic":
        proj = ccrs.Orthographic(central_latitude=central_latitude, central_longitude=central_longitude)
    elif projection == "robinson":
        proj = ccrs.Robinson(central_longitude=central_longitude)
    elif projection == "platecarree":
        proj = ccrs.PlateCarree(central_longitude=central_longitude)
    elif projection == "mollweide":
        proj = ccrs.Mollweide(central_longitude=central_longitude)
    else:
        raise ValueError(f"Unknown projection mode {projection}")

    return proj


def plot_sphere(
    data, fig=None, projection="robinson", cmap="RdBu", title=None, colorbar=False, coastlines=False, gridlines=False,  central_latitude=0, central_longitude=0, lon=None, lat=None, **kwargs
):
    """
    Plots a function defined on the sphere using pcolormesh
    
    Parameters
    -----------
    data : numpy.ndarray or torch.Tensor
        Data to plot with shape (nlat, nlon)
    fig : matplotlib.figure.Figure, optional
        Figure to plot on, by default None (creates new figure)
    projection : str, optional
        Map projection type, by default "robinson"
    cmap : str, optional
        Colormap name, by default "RdBu"
    title : str, optional
        Plot title, by default None
    colorbar : bool, optional
        Whether to add a colorbar, by default False
    coastlines : bool, optional
        Whether to add coastlines, by default False
    gridlines : bool, optional
        Whether to add gridlines, by default False
    central_latitude : float, optional
        Central latitude for projection, by default 0
    central_longitude : float, optional
        Central longitude for projection, by default 0
    lon : numpy.ndarray, optional
        Longitude coordinates, by default None (auto-generated)
    lat : numpy.ndarray, optional
        Latitude coordinates, by default None (auto-generated)
    **kwargs
        Additional arguments passed to pcolormesh
        
    Returns
    -------
    matplotlib.collections.QuadMesh
        The plotted image object
    """

    # make sure cartopy exist
    check_plotting_dependencies()

    if fig == None:
        fig = plt.figure()

    nlat = data.shape[-2]
    nlon = data.shape[-1]
    if lon is None:
        lon = np.linspace(0, 2 * np.pi, nlon + 1)[:-1]
    if lat is None:
        lat = np.linspace(np.pi / 2.0, -np.pi / 2.0, nlat)
    Lon, Lat = np.meshgrid(lon, lat)

    # convert radians to degrees
    Lon = Lon * 180 / np.pi
    Lat = Lat * 180 / np.pi

    # get the projection. Latitude is shifted to match plot_sphere
    proj = get_projection(projection, central_latitude=central_latitude, central_longitude=central_longitude)

    ax = fig.add_subplot(projection=proj)

    # contour data over the map.
    im = ax.pcolormesh(Lon, Lat, data, cmap=cmap, transform=ccrs.PlateCarree(), antialiased=False, **kwargs)

    # add features if requested
    if coastlines:
        ax.add_feature(cartopy.feature.COASTLINE, edgecolor="white", facecolor="none", linewidth=1.5)

    # add colorbar if requested
    if colorbar:
        plt.colorbar(im)

    # add gridlines
    if gridlines:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color="gray", alpha=0.6, linestyle="--")

    # add title with smaller font
    plt.title(title, y=1.05, fontsize=8)

    return im


def imshow_sphere(data, fig=None, projection="robinson", title=None, central_latitude=0, central_longitude=0, **kwargs):
    """
    Displays an image on the sphere
    
    Parameters
    -----------
    data : numpy.ndarray or torch.Tensor
        Data to display with shape (nlat, nlon)
    fig : matplotlib.figure.Figure, optional
        Figure to plot on, by default None (creates new figure)
    projection : str, optional
        Map projection type, by default "robinson"
    title : str, optional
        Plot title, by default None
    central_latitude : float, optional
        Central latitude for projection, by default 0
    central_longitude : float, optional
        Central longitude for projection, by default 0
    **kwargs
        Additional arguments passed to imshow
        
    Returns
    -------
    matplotlib.image.AxesImage
        The displayed image object
    """

    # make sure cartopy exist
    check_plotting_dependencies()

    if fig == None:
        fig = plt.figure()

    # get the projection. Latitude is shifted to match plot_sphere
    proj = get_projection(projection, central_latitude=central_latitude, central_longitude=central_longitude + 180)

    ax = fig.add_subplot(projection=proj)

    # contour data over the map.
    im = ax.imshow(data, transform=ccrs.PlateCarree(), **kwargs)

    # add title
    plt.title(title, y=1.05)

    return im


# def plot_data(data,
#                 fig=None,
#                 cmap="RdBu",
#                 title=None,
#                 colorbar=False,
#                 coastlines=False,
#                 central_longitude=0,
#                 lon=None,
#                 lat=None,
#                 **kwargs):
#     if fig == None:
#         fig = plt.figure()

#     nlat = data.shape[-2]
#     nlon = data.shape[-1]
#     if lon is None:
#         lon = np.linspace(0, 2*np.pi, nlon+1)[:-1]
#     if lat is None:
#         lat = np.linspace(np.pi/2., -np.pi/2., nlat)
#     Lon, Lat = np.meshgrid(lon, lat)

#     proj = ccrs.Robinson(central_longitude=central_longitude)
#     # proj = ccrs.Mollweide(central_longitude=central_longitude)

#     ax = fig.add_subplot(projection=proj)
#     Lon = Lon*180/np.pi
#     Lat = Lat*180/np.pi

#     # contour data over the map.
#     im = ax.pcolormesh(Lon, Lat, data, cmap=cmap, transform=ccrs.PlateCarree(), antialiased=False, **kwargs)
#     if coastlines:
#         ax.add_feature(cartopy.feature.COASTLINE, edgecolor='white', facecolor='none', linewidth=1.5)
#     if colorbar:
#         plt.colorbar(im)
#     plt.title(title, y=1.05)

#     return im
