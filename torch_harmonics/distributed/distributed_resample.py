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

from torch_harmonics._field_contract import check_field
from torch_harmonics.grid import GridS2, RegularGridS2, _rejects_legacy_signature, require_grid
from torch_harmonics.radial import split_radial_pair
from torch_harmonics.resample import _interpolate_on_taps, _ring_taps, _slerp_shortest_arc
from torch_harmonics.utils import recast_buffers_for

from .primitives import (
    compute_split_shapes,
    copy_to_azimuth_region,
    distributed_transpose_azimuth,
    distributed_transpose_polar,
    flatten_and_pad_leading_dims,
    reduce_from_azimuth_region,
    unpad_and_unflatten_leading_dims,
)
from .utils import _point_shapes, _shard_grid_pair


class DistributedResampleS2(nn.Module):
    """
    Distributed resampling module for spherical data on the 2-sphere.

    This module performs distributed resampling of spherical data across multiple processes,
    supporting both upscaling and downscaling operations. The data is distributed across
    polar and azimuthal directions, and the module handles the necessary communication
    and interpolation operations.

    **Scheme.** Between two regular grids the interpolation is separable: an all-to-all
    transpose over the polar group makes the latitudes local, with the channels split, the
    latitudes are interpolated and transposed back, and the same follows for the longitudes
    over the azimuth group. A pair with a :class:`~torch_harmonics.healpix.HealpixGrid` on
    either side is interpolated point by point, as in :class:`~torch_harmonics.ResampleS2`:
    one polar transpose makes the whole flat input field local, with the channels split, the
    serial interpolation runs on it, and a second transpose cuts the output into this rank's
    band. A HEALPix grid decomposes along latitude only, so such a pair requires an azimuth
    group of one rank and raises ``NotImplementedError`` otherwise.

    The transposes split the leading axis, so all leading dimensions (batch, channels and the
    radial axis of a product grid) are flattened into one and zero-padded when it is shorter
    than the larger process group, as in :class:`DistributedRealSHT`. Any batch and channel
    count works.

    .. seealso::
        :class:`torch_harmonics.ResampleS2`
            Serial counterpart with full mathematical description and parameter
            documentation.

    Parameters
    ----------
    grid_in : RegularGridS2 or HealpixGrid
        Descriptor of the **global** input grid, not this rank's shard; each rank
        derives its own slice. A field on a HEALPix grid is this rank's band of the
        flat ``(..., npix)`` axis.
    grid_out : RegularGridS2 or HealpixGrid
        Descriptor of the **global** output grid.
    mode : str, optional
        Interpolation mode (``"bilinear"`` or ``"bilinear-spherical"``), by default ``"bilinear"``
    """

    @_rejects_legacy_signature(
        'nlat_in, nlon_in, nlat_out, nlon_out, grid_in="equiangular", grid_out="equiangular", mode="bilinear"',
        grid_in=("nlat_in", "nlon_in"),
        grid_out=("nlat_out", "nlon_out"),
    )
    def __init__(
        self,
        grid_in: GridS2,
        grid_out: GridS2,
        mode: Optional[str] = "bilinear",
    ):

        super().__init__()

        # currently only bilinear is supported
        if mode in ["bilinear", "bilinear-spherical"]:
            self.mode = mode
        else:
            raise NotImplementedError(f"unknown interpolation mode {mode}")

        # on a pair of product grids every shell is resampled alike, see forward
        grid_in, grid_out, self.radial = split_radial_pair(grid_in, grid_out)
        self.grid_in = require_grid(grid_in, "grid_in")
        self.grid_out = require_grid(grid_out, "grid_out")
        # a dense (nlat, nlon) layout on both sides allows the separable interpolation, which
        # transposes latitudes and longitudes in turn; any other pair is interpolated point by
        # point on the whole field, see ResampleS2
        self.separable = isinstance(self.grid_in, RegularGridS2) and isinstance(self.grid_out, RegularGridS2)

        # the process grid and this rank's shards; the shards know the shapes every other
        # rank holds, which is what the transposes need
        _shard_grid_pair(self)

        # the expansion to the poles is a property of the latitudes alone
        colats_in, colats_out = self.grid_in.colats, self.grid_out.colats
        self.expand_poles = bool((colats_out > colats_in[-1]).any() or (colats_out < colats_in[0]).any())

        if self.separable:
            self._init_separable()
        else:
            # the points every polar rank holds, flat: a regular band is a contiguous run of
            # whole rows, a HEALPix band one of whole rings
            self.points_in_shapes = _point_shapes(self.shard_in)
            self.points_out_shapes = _point_shapes(self.shard_out)
            taps, lon_weights, lat_weights = _ring_taps(self.grid_in, self.grid_out, self.expand_poles)
            self.register_buffer("taps", taps, persistent=False)
            self.register_buffer("tap_lon_weights", lon_weights, persistent=False)
            self.register_buffer("tap_lat_weights", lat_weights, persistent=False)

        # descriptor equality covers the grid type and both extents
        self.skip_resampling = self.grid_in == self.grid_out

    def _init_separable(self):
        self.nlat_in, self.nlon_in = self.grid_in.shape
        self.nlat_out, self.nlon_out = self.grid_out.shape

        self.lat_in_shapes = list(self.shard_in.lat_shapes)
        self.lon_in_shapes = list(self.shard_in.lon_shapes)
        self.lat_out_shapes = list(self.shard_out.lat_shapes)
        self.lon_out_shapes = list(self.shard_out.lon_shapes)

        # for upscaling the latitudes we will use interpolation
        self.colats_in = self.grid_in.colats
        self.lons_in = self.grid_in.lons()
        self.colats_out = self.grid_out.colats
        self.lons_out = self.grid_out.lons()

        # in the case where some points lie outside of the range spanned by colats_in,
        # we need to expand the solution to the poles before interpolating
        if self.expand_poles:
            self.colats_in = torch.cat([torch.tensor([0.0], dtype=torch.float64), self.colats_in, torch.tensor([math.pi], dtype=torch.float64)]).contiguous()

        # prepare the interpolation by computing indices to the left and right of each output latitude
        lat_idx = torch.searchsorted(self.colats_in, self.colats_out, side="right") - 1
        # make sure that we properly treat the last point if they coincide with the pole
        lat_idx = torch.where(self.colats_out == self.colats_in[-1], lat_idx - 1, lat_idx)

        # lat_idx = np.where(self.colats_out > self.colats_in[-1], lat_idx - 1, lat_idx)
        # lat_idx = np.where(self.colats_out < self.colats_in[0], 0, lat_idx)

        # compute the interpolation weights along the latitude
        # the weights stay float64, like the SHT and quadrature weights, and are cast to the precision of
        # the data at use; rounding them to float32 here would cap float64 data at ~1e-9
        lat_weights = (self.colats_out - self.colats_in[lat_idx]) / torch.diff(self.colats_in)[lat_idx]
        lat_weights = lat_weights.unsqueeze(-1)

        # register buffers
        self.register_buffer("lat_idx", lat_idx, persistent=False)
        self.register_buffer("lat_weights", lat_weights, persistent=False)

        # get left and right indices but this time make sure periodicity in the longitude is handled
        lon_idx_left = torch.searchsorted(self.lons_in, self.lons_out, side="right") - 1
        lon_idx_right = torch.where(self.lons_out >= self.lons_in[-1], torch.zeros_like(lon_idx_left), lon_idx_left + 1)

        # get the difference
        diff = self.lons_in[lon_idx_right] - self.lons_in[lon_idx_left]
        # Coincident endpoints span a full period when there is one longitude.
        diff = torch.where(diff <= 0.0, diff + 2 * math.pi, diff)
        lon_weights = (self.lons_out - self.lons_in[lon_idx_left]) / diff

        # register buffers
        self.register_buffer("lon_idx_left", lon_idx_left, persistent=False)
        self.register_buffer("lon_idx_right", lon_idx_right, persistent=False)
        self.register_buffer("lon_weights", lon_weights, persistent=False)

    def _apply(self, fn, recurse=True):
        # float64 buffers cannot move to MPS; see recast_buffers_for
        recast_buffers_for(self, fn)
        return super()._apply(fn, recurse)

    def extra_repr(self):
        radial = "" if self.radial is None else f",\nradial={self.radial!r}"
        return f"grid_in={self.grid_in!r},\ngrid_out={self.grid_out!r}{radial},\nmode={self.mode}"

    def _upscale_longitudes(self, x: torch.Tensor):
        """Upscale the longitude dimension using interpolation."""
        # do the interpolation
        lwgt = self.lon_weights.to(x.dtype)
        if self.mode == "bilinear":
            x = torch.lerp(x[..., self.lon_idx_left], x[..., self.lon_idx_right], lwgt)
        else:
            x = _slerp_shortest_arc(x[..., self.lon_idx_left], x[..., self.lon_idx_right], lwgt)

        return x

    def _expand_poles(self, x: torch.Tensor):
        """Expand the data to include pole values for interpolation."""
        # A pole is a single point, so its value cannot depend on longitude: reduce the
        # adjacent ring over phi. That reduction annihilates every m != 0 mode exactly,
        # which is the right answer there, since every m != 0 mode vanishes at the pole.
        # Longitude is sharded across the azimuth group, so the reduction is collective.
        #
        # Do NOT replace this with an f(theta, phi) -> f(-theta, phi + pi) continuation.
        # That is the correct way to reach *through* a pole (as a neighbourhood stencil
        # does), but here it would leave a phi-dependent -- i.e. multivalued -- value
        # *at* the pole: odd m cancels, but even m reinforces.
        if self.mode == "bilinear":
            north = [x[..., 0, :].sum(dim=-1, keepdims=True)]
            south = [x[..., -1, :].sum(dim=-1, keepdims=True)]
        else:
            # angle-valued field: reduce as directions (sum the unit vectors), not as
            # numbers, otherwise the ring mean of +pi and -pi comes out as 0
            north = [torch.sin(x[..., 0, :]).sum(dim=-1, keepdims=True), torch.cos(x[..., 0, :]).sum(dim=-1, keepdims=True)]
            south = [torch.sin(x[..., -1, :]).sum(dim=-1, keepdims=True), torch.cos(x[..., -1, :]).sum(dim=-1, keepdims=True)]

        if self.comm_size_azimuth > 1:
            north = [reduce_from_azimuth_region(t.contiguous()) for t in north]
            south = [reduce_from_azimuth_region(t.contiguous()) for t in south]

        if self.mode == "bilinear":
            x_count = torch.tensor([x.shape[-1]], dtype=torch.long, device=x.device, requires_grad=False)
            if self.comm_size_azimuth > 1:
                x_count = reduce_from_azimuth_region(x_count)
            x_north = north[0] / x_count
            x_south = south[0] / x_count
        else:
            # atan2 is scale invariant, so the 1/count normalisation cancels here
            x_north = torch.atan2(*north)
            x_south = torch.atan2(*south)

        if self.comm_size_azimuth > 1:
            x_north = copy_to_azimuth_region(x_north)
            x_south = copy_to_azimuth_region(x_south)

        x = nn.functional.pad(x, pad=[0, 0, 1, 1], mode="constant")
        x[..., 0, :] = x_north[...]
        x[..., -1, :] = x_south[...]

        return x

    def _upscale_latitudes(self, x: torch.Tensor):
        """Upscale the latitude dimension using interpolation."""
        # do the interpolation
        lwgt = self.lat_weights.to(x.dtype)
        if self.mode == "bilinear":
            x = torch.lerp(x[..., self.lat_idx, :], x[..., self.lat_idx + 1, :], lwgt)
        else:
            x = _slerp_shortest_arc(x[..., self.lat_idx, :], x[..., self.lat_idx + 1, :], lwgt)

        return x

    def forward(self, x: torch.Tensor):
        # a channel axis is required, as for the distributed SHTs
        check_field(x, self.shard_in, leading=(..., None), radial=self.radial)

        if self.skip_resampling:
            return x

        # the transposes split the leading axis across the ranks of either group. Flatten every
        # leading dim into it, the radial axis of a product field included, since every shell
        # is resampled alike, and zero-pad it if needed (linear in the field, so the padding
        # stays out of the result); restore the original layout before returning
        x, lead_shape, lead_size = flatten_and_pad_leading_dims(x, max(self.comm_size_polar, self.comm_size_azimuth), num_trailing_dims=len(self.shard_in.shape))

        if self.separable:
            x = self._resample_separable(x)
        else:
            x = self._resample_taps(x)

        x = unpad_and_unflatten_leading_dims(x, lead_shape, lead_size, num_trailing_dims=len(self.shard_out.shape))

        return x

    def _resample_taps(self, x: torch.Tensor) -> torch.Tensor:
        """Point-by-point interpolation on the whole field, for a pair with a ragged side."""
        # flatten this rank's band to its run of points, behind the leading axis
        x = x.reshape(x.shape[0], self.shard_in.npoints)
        num_chans = x.shape[0]

        # make the whole input field local, with the channels split
        if self.comm_size_polar > 1:
            channels_shapes = compute_split_shapes(num_chans, self.comm_size_polar)
            x = distributed_transpose_polar(x, (0, 1), self.points_in_shapes)

        x = _interpolate_on_taps(x, self.grid_in, self.taps, self.tap_lon_weights, self.tap_lat_weights, self.mode, self.expand_poles)

        # cut the output into the bands of the polar ranks, with the channels whole again
        if self.comm_size_polar > 1:
            x = distributed_transpose_polar(x, (1, 0), channels_shapes, self.points_out_shapes)

        return x.reshape(x.shape[0], *self.shard_out.shape)

    def _resample_separable(self, x: torch.Tensor) -> torch.Tensor:
        """Latitudes, then longitudes, each made local by a transpose over its group."""

        # transpose data so that h is local, and channels are split
        num_chans = x.shape[-3]

        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_polar > 1:
            channels_shapes = compute_split_shapes(num_chans, self.comm_size_polar)
            x = distributed_transpose_polar(x, (-3, -2), self.lat_in_shapes)

        # expand poles if requested
        if self.expand_poles:
            x = self._expand_poles(x)

        # upscaling
        x = self._upscale_latitudes(x)

        # now, transpose back
        if self.comm_size_polar > 1:
            x = distributed_transpose_polar(x, (-2, -3), channels_shapes)

        # now, transpose in w:
        if self.comm_size_azimuth > 1:
            channels_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = distributed_transpose_azimuth(x, (-3, -1), self.lon_in_shapes)

        # upscale
        x = self._upscale_longitudes(x)

        # transpose back
        if self.comm_size_azimuth > 1:
            x = distributed_transpose_azimuth(x, (-1, -3), channels_shapes)

        return x
