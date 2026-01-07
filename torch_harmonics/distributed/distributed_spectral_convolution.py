# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
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

from typing import Tuple, Optional

import torch
import torch.nn as nn
import math

from torch_harmonics.spectral_convolution import _contract_lwise

from torch_harmonics.distributed import DistributedRealSHT, DistributedInverseRealSHT, DistributedQuadratureS2

from torch_harmonics.distributed import polar_group_size, azimuth_group_size
from torch_harmonics.distributed import polar_group_rank, azimuth_group_rank
from torch_harmonics.distributed import copy_to_polar_region, copy_to_azimuth_region


class DistributedSpectralConvS2(nn.Module):
    """
    Spectral Convolution of the Driscoll-Healy type implemented via SHT.
    """

    def __init__(self, 
        in_shape: Tuple[int],
        out_shape: Tuple[int],
        in_channels: int,
        out_channels: int,
        num_groups: Optional[int]=1,
        grid_in: Optional[str]="equiangular",
        grid_out: Optional[str]="equiangular",
        bias: Optional[bool]=False,
    ):
        super().__init__()

        assert in_channels % num_groups == 0
        assert out_channels % num_groups == 0

        # copy inputs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups

        # get the comms grid:
        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # set up sht layers
        self.sht = DistributedRealSHT(*in_shape, grid=grid_in)
        self.isht = DistributedInverseRealSHT(*out_shape, grid=grid_out)

        # extract sht parameters
        self.modes_lat = self.isht.lmax
        self.modes_lon = self.isht.mmax
        self.modes_lat_local = self.isht.l_shapes[self.comm_rank_polar]
        self.modes_lon_local = self.isht.m_shapes[self.comm_rank_azimuth]
        self.nlat_local = self.isht.lat_shapes[self.comm_rank_polar]
        self.nlon_local = self.isht.lon_shapes[self.comm_rank_azimuth]

        self.scale_residual = (in_shape[0] != out_shape[0]) or (in_shape[1] != out_shape[1]) or (grid_in != grid_out)

        # weight shape 
        weight_shape = [num_groups, in_channels // num_groups, out_channels // num_groups, self.modes_lat_local]

        # Compute scaling factor for correct initialization
        scale = math.sqrt(1.0 / (in_channels // num_groups)) * torch.ones(self.modes_lat_local, dtype=torch.complex64)
        # seemingly the first weight is not really complex, so we need to account for that
        scale[0] *= math.sqrt(2.0)
        self.weight = nn.Parameter(scale * torch.randn(*weight_shape, dtype=torch.complex64))

        # get the contraction handle. This should return a pyTorch contraction
        self.contract_handle = _contract_lwise

        if bias == True:
            self.spectral_bias = nn.Parameter(
                torch.zeros(1, self.out_channels, self.modes_lat_local, self.modes_lon_local, dtype=torch.complex64)
            )
            self.quadrature = DistributedQuadratureS2(
                img_shape=in_shape, 
                grid=grid_in, 
                normalize=False
            )

    def forward(self, x):
        dtype = x.dtype
        residual = x
        x = x.float()

        # compute integral in case if bias is used
        if hasattr(self, "bias"):
            integral = self.quadrature(x)
            if self.comm_size_polar > 1:
                integral = copy_to_polar_region(integral)
            if self.comm_size_azimuth > 1:
                integral = copy_to_azimuth_region(integral)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            x = self.sht(x).contiguous()
            if self.scale_residual:
                residual = self.isht(x)
                residual = residual.to(dtype)

        # store the shapes
        B, C, H, W = x.shape

        # deal with bias
        if hasattr(self, "bias"):
            x = x + integral.reshape(B, C, 1, 1) * self.spectral_bias

        # perform contraction
        x = x.reshape(B, self.num_groups, C // self.num_groups, H, W)
        xp = self.contract_handle(x, self.weight)
        x = xp.reshape(B, self.out_channels, H, W).contiguous()

        with torch.amp.autocast(device_type="cuda", enabled=False):
            x = self.isht(x)

        # convert datatype
        x = x.to(dtype=dtype)

        return x, residual