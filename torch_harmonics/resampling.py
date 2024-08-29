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
import numpy as np

import torch
import torch.nn as nn

from torch_harmonics.quadrature import _precompute_latitudes


class ResampleS2(nn.Module):
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
        if mode == "bilinear":
            self.mode = mode
        else:
            raise NotImplementedError(f"unknown interpolation mode {mode}")

        self.nlat_in, self.nlon_in = nlat_in, nlon_in
        self.nlat_out, self.nlon_out = nlat_out, nlon_out

        self.grid_in = grid_in
        self.grid_out = grid_out

        # for upscaling the latitudes we will use interpolation
        self.lats_in, _ = _precompute_latitudes(nlat_in, grid=grid_in)
        self.lons_in = np.linspace(0, 2 * math.pi, nlon_in, endpoint=False)
        self.lats_out, _ = _precompute_latitudes(nlat_out, grid=grid_out)
        self.lons_out = np.linspace(0, 2 * math.pi, nlon_out, endpoint=False)

        # prepare the interpolation by computing indices to the left and right of each output latitude
        lat_idx = np.searchsorted(self.lats_in, self.lats_out, side="right") - 1
        # to guarantee everything stays in bounds
        lat_idx = np.where(self.lats_out == self.lats_in[-1], lat_idx - 1, lat_idx)

        # compute the interpolation weights along the latitude
        lat_weights = torch.from_numpy((self.lats_out - self.lats_in[lat_idx]) / np.diff(self.lats_in)[lat_idx]).float()
        lat_weights = lat_weights.unsqueeze(-1)

        # convert to tensor
        lat_idx = torch.LongTensor(lat_idx)

        # register buffers
        self.register_buffer("lat_idx", lat_idx, persistent=False)
        self.register_buffer("lat_weights", lat_weights, persistent=False)

        # get left and right indices but this time make sure periodicity in the longitude is handled
        lon_idx_left = np.searchsorted(self.lons_in, self.lons_out, side="right") - 1
        lon_idx_right = np.where(self.lons_out >= self.lons_in[-1], np.zeros_like(lon_idx_left), lon_idx_left + 1)

        # get the difference
        diff = self.lons_in[lon_idx_right] - self.lons_in[lon_idx_left]
        diff = np.where(diff < 0.0, diff + 2 * math.pi, diff)
        lon_weights = torch.from_numpy((self.lons_out - self.lons_in[lon_idx_left]) / diff).float()

        # convert to tensor
        lon_idx_left = torch.LongTensor(lon_idx_left)
        lon_idx_right = torch.LongTensor(lon_idx_right)

        # register buffers
        self.register_buffer("lon_idx_left", lon_idx_left, persistent=False)
        self.register_buffer("lon_idx_right", lon_idx_right, persistent=False)
        self.register_buffer("lon_weights", lon_weights, persistent=False)

    def extra_repr(self):
        r"""
        Pretty print module
        """
        return f"in_shape={(self.nlat_in, self.nlon_in)}, out_shape={(self.nlat_out, self.nlon_out)}"

    def _upscale_longitudes(self, x: torch.Tensor):
        # do the interpolation
        x = torch.lerp(x[..., self.lon_idx_left], x[..., self.lon_idx_right], self.lon_weights)
        return x

    # old deprecated method with repeat_interleave
    # def _upscale_longitudes(self, x: torch.Tensor):
    #     # for artifact-free upsampling in the longitudinal direction
    #     x = torch.repeat_interleave(x, self.lon_scale_factor, dim=-1)
    #     x = torch.roll(x, - self.lon_shift, dims=-1)
    #     return x

    def _upscale_latitudes(self, x: torch.Tensor):
        # do the interpolation
        x = torch.lerp(x[..., self.lat_idx, :], x[..., self.lat_idx + 1, :], self.lat_weights)
        return x

    def forward(self, x: torch.Tensor):
        x = self._upscale_latitudes(x)
        x = self._upscale_longitudes(x)
        return x
