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

from typing import List, Tuple, Union, Optional
import math

import torch
import torch.nn as nn

from torch_harmonics.quadrature import _precompute_latitudes, _precompute_longitudes
from torch_harmonics.distributed import polar_group_size, azimuth_group_size, distributed_transpose_azimuth, distributed_transpose_polar
from torch_harmonics.distributed import polar_group_rank, azimuth_group_rank
from torch_harmonics.distributed import compute_split_shapes


class DistributedResampleS2(nn.Module):
    def __init__(
        self,
        nlat_in: int,
        nlon_in: int,
        nlat_out: int,
        nlon_out: int,
        grid_in: Optional[str] = "equiangular",
        grid_out: Optional[str] = "equiangular",
        mode: Optional[str] = "bilinear",
    ):

        super().__init__()

        # currently only bilinear is supported
        if mode in ["bilinear", "bilinear-spherical"]:
            self.mode = mode
        else:
            raise NotImplementedError(f"unknown interpolation mode {mode}")

        self.nlat_in, self.nlon_in = nlat_in, nlon_in
        self.nlat_out, self.nlon_out = nlat_out, nlon_out

        self.grid_in = grid_in
        self.grid_out = grid_out

        # get the comms grid:
        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # compute splits: is this correct even when expanding the poles?
        self.lat_in_shapes = compute_split_shapes(self.nlat_in, self.comm_size_polar)
        self.lon_in_shapes = compute_split_shapes(self.nlon_in, self.comm_size_azimuth)
        self.lat_out_shapes = compute_split_shapes(self.nlat_out, self.comm_size_polar)
        self.lon_out_shapes = compute_split_shapes(self.nlon_out, self.comm_size_azimuth)

        # for upscaling the latitudes we will use interpolation
        self.lats_in, _ = _precompute_latitudes(nlat_in, grid=grid_in)
        self.lons_in = _precompute_longitudes(nlon_in)
        self.lats_out, _ = _precompute_latitudes(nlat_out, grid=grid_out)
        self.lons_out = _precompute_longitudes(nlon_out)

        # in the case where some points lie outside of the range spanned by lats_in,
        # we need to expand the solution to the poles before interpolating
        self.expand_poles = (self.lats_out > self.lats_in[-1]).any() or (self.lats_out < self.lats_in[0]).any()
        if self.expand_poles:
            self.lats_in = torch.cat([torch.tensor([0.], dtype=torch.float64),
                                      self.lats_in,
                                      torch.tensor([math.pi], dtype=torch.float64)]).contiguous()
            #self.lats_in = np.insert(self.lats_in, 0, 0.0)
            #self.lats_in = np.append(self.lats_in, np.pi)

        # prepare the interpolation by computing indices to the left and right of each output latitude
        lat_idx = torch.searchsorted(self.lats_in, self.lats_out, side="right") - 1
        # make sure that we properly treat the last point if they coincide with the pole
        lat_idx = torch.where(self.lats_out == self.lats_in[-1], lat_idx - 1, lat_idx)

        # lat_idx = np.where(self.lats_out > self.lats_in[-1], lat_idx - 1, lat_idx)
        # lat_idx = np.where(self.lats_out < self.lats_in[0], 0, lat_idx)

        # compute the interpolation weights along the latitude
        lat_weights = ((self.lats_out - self.lats_in[lat_idx]) / torch.diff(self.lats_in)[lat_idx]).to(torch.float32)
        lat_weights = lat_weights.unsqueeze(-1)

        # register buffers
        self.register_buffer("lat_idx", lat_idx, persistent=False)
        self.register_buffer("lat_weights", lat_weights, persistent=False)

        # get left and right indices but this time make sure periodicity in the longitude is handled
        lon_idx_left = torch.searchsorted(self.lons_in, self.lons_out, side="right") - 1
        lon_idx_right = torch.where(self.lons_out >= self.lons_in[-1], torch.zeros_like(lon_idx_left), lon_idx_left + 1)

        # get the difference
        diff = self.lons_in[lon_idx_right] - self.lons_in[lon_idx_left]
        diff = torch.where(diff < 0.0, diff + 2 * math.pi, diff)
        lon_weights = ((self.lons_out - self.lons_in[lon_idx_left]) / diff).to(torch.float32)

        # register buffers
        self.register_buffer("lon_idx_left", lon_idx_left, persistent=False)
        self.register_buffer("lon_idx_right", lon_idx_right, persistent=False)
        self.register_buffer("lon_weights", lon_weights, persistent=False)

        self.skip_resampling = (nlon_in == nlon_out) and (nlat_in == nlat_out) and (grid_in == grid_out)

    def extra_repr(self):
        r"""
        Pretty print module
        """
        return f"in_shape={(self.nlat_in, self.nlon_in)}, out_shape={(self.nlat_out, self.nlon_out)}"

    def _upscale_longitudes(self, x: torch.Tensor):
        # do the interpolation
        if self.mode == "bilinear":
            x = torch.lerp(x[..., self.lon_idx_left], x[..., self.lon_idx_right], self.lon_weights)
        else:
            omega = x[..., self.lon_idx_right] - x[..., self.lon_idx_left]
            somega = torch.sin(omega)
            start_prefac = torch.where(somega > 1e-4, torch.sin((1.0 - self.lon_weights) * omega) / somega, (1.0 - self.lon_weights))
            end_prefac = torch.where(somega > 1e-4, torch.sin(self.lon_weights * omega) / somega, self.lon_weights)
            x = start_prefac * x[..., self.lon_idx_left] + end_prefac * x[..., self.lon_idx_right]

        return x

    def _expand_poles(self, x: torch.Tensor):
        repeats = [1 for _ in x.shape]
        repeats[-1] = x.shape[-1]
        x_north = x[..., 0:1, :].mean(dim=-1, keepdim=True).repeat(*repeats)
        x_south = x[..., -1:, :].mean(dim=-1, keepdim=True).repeat(*repeats)
        x = torch.concatenate((x_north, x, x_south), dim=-2)
        return x

    def _upscale_latitudes(self, x: torch.Tensor):
        # do the interpolation
        if self.mode == "bilinear":
            x = torch.lerp(x[..., self.lat_idx, :], x[..., self.lat_idx + 1, :], self.lat_weights)
        else:
            omega = x[..., self.lat_idx + 1, :] - x[..., self.lat_idx, :]
            somega = torch.sin(omega)
            start_prefac = torch.where(somega > 1e-4, torch.sin((1.0 - self.lat_weights) * omega) / somega, (1.0 - self.lat_weights))
            end_prefac = torch.where(somega > 1e-4, torch.sin(self.lat_weights * omega) / somega, self.lat_weights)
            x = start_prefac * x[..., self.lat_idx, :] + end_prefac * x[..., self.lat_idx + 1, :]

        return x

    def forward(self, x: torch.Tensor):

        if self.skip_resampling:
            return x

        # transpose data so that h is local, and channels are split
        num_chans = x.shape[-3]

        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_polar > 1:
            channels_shapes = compute_split_shapes(num_chans, self.comm_size_polar)
            x = distributed_transpose_polar.apply(x, (-3, -2), self.lat_in_shapes)

        # expand poles if requested
        if self.expand_poles:
            x = self._expand_poles(x)

        # upscaling
        x = self._upscale_latitudes(x)

        # now, transpose back
        if self.comm_size_polar > 1:
            x = distributed_transpose_polar.apply(x, (-2, -3), channels_shapes)

        # now, transpose in w:
        if self.comm_size_azimuth > 1:
            channels_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = distributed_transpose_azimuth.apply(x, (-3, -1), self.lon_in_shapes)

        # upscale
        x = self._upscale_longitudes(x)

        # transpose back
        if self.comm_size_azimuth > 1:
            x = distributed_transpose_azimuth.apply(x, (-1, -3), channels_shapes)

        return x
