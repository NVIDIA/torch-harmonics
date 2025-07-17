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

from typing import List, Tuple, Union, Optional
from warnings import warn

import math

import torch
import torch.nn as nn

from torch_harmonics.quadrature import _precompute_latitudes
from torch_harmonics.disco.convolution import _precompute_convolution_tensor_s2
from torch_harmonics.attention._neighborhood_attention import _neighborhood_attention_s2_torch, _neighborhood_attention_s2_cuda
from torch_harmonics.filter_basis import get_filter_basis
from attention_helpers import optimized_kernels_is_available


class AttentionS2(nn.Module):
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
            key = query

        if value is None:
            value = query

        # change this later to allow arbitrary number of batch dims
        assert (query.dim() == key.dim()) and (key.dim() == value.dim()) and (value.dim() == 4)

        # perform MLP
        query = nn.functional.conv2d(query, self.q_weights, bias=self.q_bias)
        key = nn.functional.conv2d(key, self.k_weights, bias=self.k_bias)
        value = nn.functional.conv2d(value, self.v_weights, bias=self.v_bias)

        # reshape
        B, _, H, W = query.shape
        query = query.reshape(B, self.num_heads, -1, H, W)
        B, _, H, W = key.shape
        key = key.reshape(B, self.num_heads, -1, H, W)
        B, _, H, W = value.shape
        value = value.reshape(B, self.num_heads, -1, H, W)

        # reshape to the right dimensions
        B, _, C, H, W = query.shape
        query = query.permute(0,1,3,4,2).reshape(B, self.num_heads, H*W, C)
        B, _, C, H, W = key.shape
        key = key.permute(0,1,3,4,2).reshape(B, self.num_heads, H*W, C)
        B, _, C, H, W = value.shape
        value = value.permute(0,1,3,4,2).reshape(B, self.num_heads, H*W, C)

        # multiply the query, key and value tensors
        out = nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=self.log_quad_weights, dropout_p=self.drop_rate, scale=self.scale)

        # reshape
        B, _, _, C = out.shape
        # (B, heads, H*W, C)
        out = out.permute(0,1,3,2)
        # (B, heads, C, H*W)
        out = out.reshape(B, self.num_heads*C, self.nlat_out, self.nlon_out)
        # (B, heads*C, H, W)
        out = nn.functional.conv2d(out, self.proj_weights, bias=self.proj_bias)

        return out


class NeighborhoodAttentionS2(nn.Module):
    """
    Neighborhood attention on the 2-sphere.

    Parameters
    -----------
    in_channels: int
        number of channels of the input signal (corresponds to embed_dim in MHA in PyTorch)
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
    theta_cutoff: float, optional
        neighborhood size
    k_channels: int
        number of dimensions for interior inner product in the attention matrix (corresponds to kdim in MHA in PyTorch)
    out_channels: int, optional
        number of dimensions for interior inner product in the attention matrix (corresponds to vdim in MHA in PyTorch)
    """

    def __init__(
        self,
        in_channels: int,
        in_shape: Tuple[int],
        out_shape: Tuple[int],
        grid_in: Optional[str] = "equiangular",
        grid_out: Optional[str] = "equiangular",
        num_heads: Optional[int] = 1,
        scale: Optional[Union[torch.Tensor, float]] = None,
        bias: Optional[bool] = True,
        theta_cutoff: Optional[float] = None,
        k_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
    ):
        super().__init__()

        self.nlat_in, self.nlon_in = in_shape
        self.nlat_out, self.nlon_out = out_shape

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.k_channels = in_channels if k_channels is None else k_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        # heuristic to compute theta cutoff based on the bandlimit of the input field and overlaps of the basis functions
        if theta_cutoff is None:
            theta_cutoff = torch.pi / float(self.nlat_out - 1)

        if theta_cutoff <= 0.0:
            raise ValueError("Error, theta_cutoff has to be positive.")

        # integration weights
        _, wgl = _precompute_latitudes(self.nlat_in, grid=grid_in)
        quad_weights = 2.0 * torch.pi * wgl.to(dtype=torch.float32) / self.nlon_in
        self.register_buffer("quad_weights", quad_weights, persistent=False)

        # create a dummy filter basis to pass to the construction of the convolution tensor
        # this is to avoid code duplication as the logic of pre-computing the sparsity pattern
        # is identical to convolutions with a constant filter function
        fb = get_filter_basis(kernel_shape=1, basis_type="zernike")

        # precompute the neighborhood sparsity pattern
        idx, _, roff = _precompute_convolution_tensor_s2(
            in_shape,
            out_shape,
            fb,
            grid_in=grid_in,
            grid_out=grid_out,
            theta_cutoff=theta_cutoff,
            transpose_normalization=False,
            basis_norm_mode="none",
            merge_quadrature=True,
        )

        # this is kept for legacy resons in case we want to resuse sorting of these entries
        row_idx = idx[1, ...].contiguous()
        col_idx = idx[2, ...].contiguous()
        roff_idx = roff.contiguous()

        # store some metadata
        self.max_psi_nnz = col_idx.max().item() + 1
        self.register_buffer("psi_row_idx", row_idx, persistent=False)
        self.register_buffer("psi_col_idx", col_idx, persistent=False)
        self.register_buffer("psi_roff_idx", roff_idx, persistent=False)

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

        if scale is not None:
            self.scale = scale
        else:
            self.scale = 1 / math.sqrt(self.k_channels)

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
            key = query

        if value is None:
            value = query

        # change this later to allow arbitrary number of batch dims
        assert (query.dim() == key.dim()) and (key.dim() == value.dim()) and (value.dim() == 4)

        # do the scaling
        query_scaled = query * self.scale

        # TODO: insert dimension checks for input
        if query.is_cuda and optimized_kernels_is_available():

            out = _neighborhood_attention_s2_cuda(
                key,
                value,
                query_scaled,
                self.k_weights,
                self.v_weights,
                self.q_weights,
                self.k_bias,
                self.v_bias,
                self.q_bias,
                self.quad_weights,
                self.psi_col_idx,
                self.psi_roff_idx,
                self.max_psi_nnz,
                self.num_heads,
                self.nlon_in,
                self.nlat_out,
                self.nlon_out,
            )
        else:
            if query.is_cuda:
                warn("couldn't find CUDA extension, falling back to slow PyTorch implementation")

            # call attention
            out = _neighborhood_attention_s2_torch(
                key,
                value,
                query_scaled,
                self.k_weights,
                self.v_weights,
                self.q_weights,
                self.k_bias,
                self.v_bias,
                self.q_bias,
                self.quad_weights,
                self.psi_col_idx,
                self.psi_roff_idx,
                self.num_heads,
                self.nlon_in,
                self.nlat_out,
                self.nlon_out,
            )

        out = nn.functional.conv2d(out, self.proj_weights, bias=self.proj_bias)

        return out
