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

from typing import Tuple, Union, Optional

import math

import torch
import torch.nn as nn

from torch_harmonics.quadrature import _precompute_latitudes

# distirbuted stuff
from torch_harmonics.distributed import polar_group_size, azimuth_group_size
from torch_harmonics.distributed import gather_from_copy_to_polar_region, gather_from_copy_to_azimuth_region
from torch_harmonics.distributed import polar_group_rank, azimuth_group_rank
from torch_harmonics.distributed import compute_split_shapes


class DistributedAttentionS2(nn.Module):
    """
    (Global) attention on the 2-sphere.
    Parameters
    -----------
    in_channels: int
        number of channels of the input signal (corresponds to embed_dim in MHA in PyTorch)
    num_heads: int
        number of attention heads
    in_shape: tuple
        shape of the input grid
    out_shape: tuple
        shape of the output grid
    grid_in: str, optional
        input grid type, "equiangular" by default
    grid_out: str, optional
        output grid type, "equiangular" by default
    bias: bool, optional
        if specified, adds bias to input / output projection layers
    k_channels: int
        number of dimensions for interior inner product in the attention matrix (corresponds to kdim in MHA in PyTorch)
    out_channels: int, optional
        number of dimensions for interior inner product in the attention matrix (corresponds to vdim in MHA in PyTorch)
    """

    def __init__(
            self,
            in_channels: int,
            num_heads: int,
            in_shape: Tuple[int],
            out_shape: Tuple[int],
            grid_in: Optional[str] = "equiangular",
            grid_out: Optional[str] = "equiangular",
            scale: Optional[Union[torch.Tensor, float]] = None,
            bias: Optional[bool] = True,
            k_channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            drop_rate: Optional[float]=0.0,
    ):
        super().__init__()

        self.nlat_in, self.nlon_in = in_shape
        self.nlat_out, self.nlon_out = out_shape

        # get the comms grid:
        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # we need those shapes:
        self.lat_in_shapes = compute_split_shapes(self.nlat_in, self.comm_size_polar)
        self.lon_in_shapes = compute_split_shapes(self.nlon_in, self.comm_size_azimuth)
        self.lat_out_shapes = compute_split_shapes(self.nlat_out, self.comm_size_polar)
        self.lon_out_shapes = compute_split_shapes(self.nlon_out, self.comm_size_azimuth)

        # set local shapes according to distributed mode:
        self.nlat_in_local = self.lat_in_shapes[self.comm_rank_polar]
        self.nlon_in_local = self.lon_in_shapes[self.comm_rank_azimuth]
        self.nlat_out_local = self.lat_out_shapes[self.comm_rank_polar]
        self.nlon_out_local = self.lon_out_shapes[self.comm_rank_azimuth]

        # other parameters
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.k_channels = in_channels if k_channels is None else k_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.drop_rate = drop_rate
        self.scale = scale

        # integration weights
        _, wgl = _precompute_latitudes(self.nlat_in, grid=grid_in)
        quad_weights = 2.0 * torch.pi * wgl.to(dtype=torch.float32) / self.nlon_in
        # we need to tile and flatten them accordingly
        quad_weights = torch.tile(quad_weights.reshape(-1, 1), (1, self.nlon_in)).flatten()

        # compute log because they are applied as an addition prior to the softmax ('attn_mask'), which includes an exponential.
        # see https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        # for info on how 'attn_mask' is applied to the attention weights
        log_quad_weights = torch.log(quad_weights).reshape(1,1,-1)
        self.register_buffer("log_quad_weights", log_quad_weights, persistent=False)

        # learnable parameters
        # TODO: double-check that this gives us the correct initialization magnitudes
        # the standard MHA uses xavier uniform, NATTEN uses kaiming. Let's use that for now
        if self.k_channels % self.num_heads != 0:
            raise ValueError(f"Please make sure that number of heads {self.num_heads} divides k_channels {self.k_channels} evenly.")
        if self.out_channels % self.num_heads != 0:
            raise ValueError(f"Please make sure that number of heads {self.num_heads} divides out_channels {self.out_channels} evenly.")
        scale_qkv = math.sqrt(3.0 / self.in_channels)
        self.q_weights = nn.Parameter(scale_qkv * (2 * torch.rand(self.k_channels, self.in_channels, 1, 1) - 1))
        self.k_weights = nn.Parameter(scale_qkv * (2 * torch.rand(self.k_channels, self.in_channels, 1, 1) - 1))
        self.v_weights = nn.Parameter(scale_qkv * (2 * torch.rand(self.out_channels, self.in_channels, 1, 1) - 1))
        scale_proj = math.sqrt(3.0 / self.out_channels)
        self.proj_weights = nn.Parameter(scale_proj * (2 * torch.rand(self.out_channels, self.out_channels, 1, 1) - 1))

        if bias:
            self.q_bias = nn.Parameter(torch.zeros(self.k_channels))
            self.k_bias = nn.Parameter(torch.zeros(self.k_channels))
            self.v_bias = nn.Parameter(torch.zeros(self.out_channels))
            self.proj_bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
            self.proj_bias = None


    def extra_repr(self):
        return f"in_shape={(self.nlat_in, self.nlon_in)}, out_shape={(self.nlat_out, self.nlon_out)}, in_channels={self.in_channels}, out_channels={self.out_channels}, k_channels={self.k_channels}"

    def forward(self, query: torch.Tensor, key: Optional[torch.Tensor] = None, value: Optional[torch.Tensor] = None) -> torch.Tensor:

        # self attention simplification
        if key is None:
            key = query.clone()

        if value is None:
            value = query.clone()

        # change this later to allow arbitrary number of batch dims
        assert (query.dim() == key.dim()) and (key.dim() == value.dim()) and (value.dim() == 4)
        assert (query.shape[2] == self.nlat_out_local) and (query.shape[3] == self.nlon_out_local)
        assert (key.shape[2] == self.nlat_in_local) and (key.shape[3] == self.nlon_in_local)
        assert (value.shape[2] == self.nlat_in_local) and (value.shape[3] == self.nlon_in_local)

        # perform MLP
        query = nn.functional.conv2d(query, self.q_weights, bias=self.q_bias)
        key = nn.functional.conv2d(key, self.k_weights, bias=self.k_bias)
        value = nn.functional.conv2d(value, self.v_weights, bias=self.v_bias)

        # gather key and value and register gradients for backward reduction
        key_full = gather_from_copy_to_polar_region(key, -2, self.lat_in_shapes)
        key_full = gather_from_copy_to_azimuth_region(key_full, -1, self.lon_in_shapes)
        value_full = gather_from_copy_to_polar_region(value, -2, self.lat_in_shapes)
        value_full = gather_from_copy_to_azimuth_region(value_full, -1, self.lon_in_shapes)

        # reshape
        B, _, Hloc, Wloc = query.shape
        query = query.reshape(B, self.num_heads, -1, Hloc, Wloc)
        B, _, H, W = key_full.shape
        key_full = key_full.reshape(B, self.num_heads, -1, H, W)
        value_full = value_full.reshape(B, self.num_heads, -1, H, W)

        # reshape to the right dimensions
        Ci = query.shape[2]
        query = query.permute(0,1,3,4,2).reshape(B, self.num_heads, Hloc*Wloc, Ci)
        Ci = key_full.shape[2]
        key_full = key_full.permute(0,1,3,4,2).reshape(B, self.num_heads, H*W, Ci)
        Co = value_full.shape[2]
        value_full = value_full.permute(0,1,3,4,2).reshape(B, self.num_heads, H*W, Co)

        # multiply the query, key and value tensors
        out = nn.functional.scaled_dot_product_attention(
            query,
            key_full,
            value_full,
            attn_mask=self.log_quad_weights,
            dropout_p=self.drop_rate,
            scale=self.scale
        )

        # reshape
        B, _, _, Co = out.shape
        # (B, heads, Hloc*Wloc, Co)
        out = out.permute(0,1,3,2)
        # (B, heads, Co, Hloc*Wloc)
        out = out.reshape(B, self.num_heads*Co, self.nlat_out_local, self.nlon_out_local)
        # (B, heads*Co, Hloc, Wloc)
        out = nn.functional.conv2d(out, self.proj_weights, bias=self.proj_bias)

        return out