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
import numpy as np

import torch
import torch.nn as nn

from torch_harmonics.quadrature import _precompute_latitudes

class UpsampleS2(nn.Module):
    def __init__(
        nlat_in: int,
        nlon_in: int,
        nlat_out: int,
        nlon_out: int,
        grid_in: Optional[str] = "equiangular",
        grid_out: Optional[str] = "equiangular",
    ):

        assert nlat_in <= nlat_out
        assert nlon_in <= nlon_out

        super().__init__()

        self.nlat_in, self.nlon_in = nlat_in, nlon_in
        self.nlat_out, self.nlon_out = nlat_out, nlon_out

        self.grid_in = grid_in
        self.grid_out = grid_out

        # for upscaling the latitudes we will use interpolation
        self.lats_in, _ = _precompute_latitudes(nlat_in, grid=grid_in)
        self.lats_out, _ = _precompute_latitudes(nlat_out, grid=grid_out)

        # prepare the interpolation by computing indices to the left and right of each output latitude
        lat_idx = np.searchsorted(self.lats_in, self.lats_out) - 1

        # compute the interpolation weights along the latitude
        lat_weights = torch.from_numpy( (self.lats_out - self.lats_in[j]) / np.diff(self.lats_in)[j] )
        lat_weights = lat_weights.unsqueeze(-1)

        # convert to tensor
        lat_idx = torch.LongTensor(lat_idx)

        # register buffers
        self.register_buffer("lat_idx", lat_idx, persistent=False)
        self.register_buffer("lat_weights", lat_weights, persistent=False)

        # for the longitudes we can use the fact that points are equidistant
        # TODO: add mode modes for upscaling in longitude
        assert nlon_out % nlon_in == 0
        self.lon_scale_factor = nlon_out // nlon_in
        self.lon_shift = (self.lon_scale_factor + 1) // 2 - 1

    def extra_repr(self):
        r"""
        Pretty print module
        """
        return f"in_shape={(self.nlat_in, self.nlon_in)}, out_shape={(self.nlat_out, self.nlon_out)}"

    def _upscale_longitudes(self, x: torch.Tensor):
        # for artifact-free upsampling in the longitudinal direction
        x = torch.repeat_interleave(x, self.lon_scale_factor, dim=-1)
        x = torch.roll(x, - self.lon_shift, dims=-1)

    def _upscale_latitudes(self, x: torch.Tensor):
        # do the interpolation
        x = torch.lerp(x[..., self.lat_idx, :], x[..., self.lat_idx+1, :], self.lat_weights)
        return x

    def forward(self, x: torch.Tensor):
        x = self._upscale_latitudes(x)
        x = self._upscale_longitudes(x)
        return x