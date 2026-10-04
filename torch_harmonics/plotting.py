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
import torch

from torch_harmonics.grid import GridS2, RegularGridS2, as_grid, require_point_set

# guarded imports
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import cartopy
    import cartopy.crs as ccrs
except ImportError:
    cartopy = None
    ccrs = None


def _check_plotting_dependencies():
    if plt is None:
        raise ImportError("matplotlib is required for plotting functions. Install it with 'pip install matplotlib'")
    if cartopy is None:
        raise ImportError("cartopy is required for map plotting. Install it with 'pip install cartopy'")


def _to_host(data):
    """A tensor on any device (CUDA, MPS, ...), possibly requiring grad, as host memory matplotlib can read."""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu()
        # NumPy has no bfloat16 (nor does matplotlib read it), so widen that one
        if data.dtype == torch.bfloat16:
            data = data.float()
    return data


def _rasterize_rings(grid, data):
    r"""
    Resample a field on a ring-structured grid onto an equiangular image, by nearest point.

    A ragged grid has no rectangular
    mesh for ``pcolormesh``, but every image cell can be given the value of the grid point
    nearest to it, which shows each point as the flat patch it represents. Nearest is
    decided per ring -- first the ring closest in colatitude, then the point closest in
    longitude on it -- which needs only the ring structure of :class:`GridS2`, and is
    exact up to the curved pixel boundaries near the poles, below plotting resolution.

    Returns the image and its cell-centre latitudes and longitudes, in radians.
    """
    # about four image cells per grid point, so the patches keep their shape
    nlat = int(min(2048, max(8, np.ceil(2.0 * np.sqrt(grid.npoints / 2.0)))))
    nlon = 2 * nlat
    theta = (np.arange(nlat) + 0.5) * (np.pi / nlat)
    phi = (np.arange(nlon) + 0.5) * (2.0 * np.pi / nlon)

    # nearest ring: the ring centres are sorted, so the boundaries are their midpoints
    colats = grid.colats.numpy()
    ring = np.searchsorted(0.5 * (colats[1:] + colats[:-1]), theta)

    # nearest point on that ring, whose points sit at 2 pi / n * (j + shift)
    size = grid.nlon_per_lat.numpy()[ring][:, None]
    shift = grid.lon_shifts.numpy()[ring][:, None]
    base = grid.lon_offsets[:-1].numpy()[ring][:, None]
    j = np.floor(phi[None, :] * size / (2.0 * np.pi) - shift + 0.5).astype(np.int64) % size

    image = np.asarray(data)[..., base + j]
    return image, np.pi / 2.0 - theta, phi


def get_projection(
    projection,
    central_latitude=0,
    central_longitude=0,
):
    """
    Get a cartopy projection object for map plotting.

    Parameters
    ----------
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
    data,
    fig=None,
    projection="robinson",
    cmap="RdBu",
    title=None,
    colorbar=False,
    coastlines=False,
    gridlines=False,
    central_latitude=0,
    central_longitude=0,
    lon=None,
    lat=None,
    grid=None,
    **kwargs,
):
    """
    Plots a function defined on the sphere using pcolormesh

    Parameters
    ----------
    data : numpy.ndarray or torch.Tensor
        Data to plot, with shape ``(nlat, nlon)``, or ``(npoints,)`` on a ragged grid such as
        HEALPix. A tensor may live on any device and require grad; it is detached and copied
        to the host.
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
        Longitude coordinates in radians. Cannot be combined with ``grid``.
    lat : numpy.ndarray, optional
        Latitude coordinates in radians. Cannot be combined with ``grid``.
    grid : GridS2 or str, optional
        Descriptor of the grid the data lives on, used to place the samples.
        A string is coerced with :func:`torch_harmonics.grid.as_grid` against
        the shape of ``data``. Prefer this over ``lat``/``lon``: the default
        placement assumes equispaced latitudes, which is only correct for the
        equiangular grid.
        A ragged :class:`~torch_harmonics.grid.GridS2` such as HEALPix takes flat
        ``(npoints,)`` data and is drawn by nearest-point resampling onto an
        equiangular image; a point set without rings is not supported.
    **kwargs
        Additional arguments passed to pcolormesh

    Returns
    -------
    matplotlib.collections.QuadMesh
        The plotted image object

    Notes
    -----
    Rows of ``data`` are ordered north to south, matching the ascending
    co-latitudes of :attr:`torch_harmonics.grid.GridS2.colats`, so the output of a
    transform can be handed over directly without flipping.
    """

    # make sure cartopy exist
    _check_plotting_dependencies()

    data = _to_host(data)

    # the grid is resolved and checked before the data is read as (nlat, nlon): a ragged
    # field is flat, and reading its shape first would fail with an IndexError instead
    if grid is not None:
        if lat is not None or lon is not None:
            raise ValueError("pass either grid or lat/lon, not both: the grid descriptor already carries both coordinate vectors")
        # a name is resolved against the shape of the data; a descriptor carries
        # its own parameters and is taken as given
        if isinstance(grid, str):
            if data.ndim < 2:
                raise ValueError(f"a grid name can only be resolved against (nlat, nlon) data, got shape {tuple(data.shape)}; pass a descriptor")
            grid = as_grid(grid, nlat=data.shape[-2], nlon=data.shape[-1])
        grid = require_point_set(grid)
        if grid.shape != tuple(data.shape[-len(grid.shape) :]):
            raise ValueError(f"grid {grid!r} does not match the shape of the data, which is {tuple(data.shape)}")

        if isinstance(grid, RegularGridS2):
            lat = grid.lats.numpy()
            lon = grid.lons().numpy()
        elif isinstance(grid, GridS2):
            # a ragged grid has no rectangular mesh; draw its nearest-point image instead
            data, lat, lon = _rasterize_rings(grid, data)
        else:
            raise TypeError(
                f"plot_sphere draws fields on ring-structured grids (GridS2); {type(grid).__name__} is a point set without " "latitude rings, which is not supported yet"
            )

    if fig is None:
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

    proj = get_projection(projection, central_latitude=central_latitude, central_longitude=central_longitude)

    ax = fig.add_subplot(projection=proj)

    # contour data over the map.
    im = ax.pcolormesh(Lon, Lat, data, cmap=cmap, transform=ccrs.PlateCarree(), antialiased=False, **kwargs)

    # add features if requested
    if coastlines:
        ax.add_feature(cartopy.feature.COASTLINE, edgecolor="white", facecolor="none", linewidth=1.5)

    # add colorbar if requested. On the figure the axes belongs to, not pyplot's current
    # one: a subfigure of a figure pyplot has already let go of (the inline backend closes
    # figures at the end of each cell) would otherwise get a new, empty figure instead
    if colorbar:
        fig.colorbar(im, ax=ax)

    # add gridlines
    if gridlines:
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color="gray", alpha=0.6, linestyle="--")

    # add title with smaller font, on this axes for the same reason
    ax.set_title(title, y=1.05, fontsize=8)

    return im


def imshow_sphere(data, fig=None, projection="robinson", title=None, central_latitude=0, central_longitude=0, **kwargs):
    """
    Displays an image on the sphere

    Parameters
    ----------
    data : numpy.ndarray or torch.Tensor
        Data to display with shape (nlat, nlon). A tensor may live on any device and require grad; it is
        detached and copied to the host.
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
    _check_plotting_dependencies()

    data = _to_host(data)

    if fig is None:
        fig = plt.figure()

    # get the projection. The longitude is shifted by 180 degrees to match plot_sphere
    proj = get_projection(projection, central_latitude=central_latitude, central_longitude=central_longitude + 180)

    ax = fig.add_subplot(projection=proj)

    # contour data over the map.
    im = ax.imshow(data, transform=ccrs.PlateCarree(), **kwargs)

    # add title
    ax.set_title(title, y=1.05)

    return im
