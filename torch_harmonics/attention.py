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
import numpy as np

from torch_harmonics.quadrature import _precompute_latitudes
from torch_harmonics.convolution import _precompute_convolution_tensor_s2
from torch_harmonics._neighborhood_attention import _neighborhood_attention_s2_torch, _neighborhood_attention_s2_cuda
from torch_harmonics.filter_basis import get_filter_basis

# import custom C++/CUDA extensions
try:
    import attention_cuda_extension

    _cuda_extension_available = True
except ImportError as err:
    attention_cuda_extension = None
    _cuda_extension_available = False

# class AttentionS2(nn.Module):
#     """
#     (Global) attention on the 2-sphere.

#     Parameters
#     -----------
#     in_channels: int
#         number of channels of the input signal (corresponds to embed_dim in MHA in PyTorch)
#     in_shape: tuple
#         shape of the input grid
#     out_shape: tuple
#         shape of the output grid
#     grid_in: str, optional
#         input grid type, "equiangular" by default
#     grid_out: str, optional
#         output grid type, "equiangular" by default
#     bias: bool, optional
#         if specified, adds bias to input / output projection layers
#     k_channels: int
#         number of dimensions for interior inner product in the attention matrix (corresponds to kdim in MHA in PyTorch)
#     out_channels: int, optional
#         number of dimensions for interior inner product in the attention matrix (corresponds to vdim in MHA in PyTorch)
#     """

#     def __init__(
#         self,
#         in_channels: int,
#         in_shape: Tuple[int],
#         out_shape: Tuple[int],
#         grid_in: Optional[str] = "equiangular",
#         grid_out: Optional[str] = "equiangular",
#         bias: Optional[bool] = True,
#         k_channels: Optional[int] = None,
#         out_channels: Optional[int] = None,
#     ):
#         super().__init__()

#         self.nlat_in, self.nlon_in = in_shape
#         self.nlat_out, self.nlon_out = out_shape

#         self.in_channels = in_channels
#         self.k_channels = in_channels if k_channels is None else k_channels
#         self.out_channels = in_channels if out_channels is None else out_channels

#         if theta_cutoff <= 0.0:
#             raise ValueError("Error, theta_cutoff has to be positive.")

#         # integration weights
#         _, wgl = _precompute_latitudes(self.nlat_in, grid=grid_in)
#         quad_weights = 2.0 * torch.pi * wgl.to(dtype=torch.float32) / self.nlon_in
#         self.register_buffer("quad_weights", quad_weights, persistent=False)

#         # learnable parameters
#         # TODO: double-check that this gives us the correct initialization magnitudes
#         scale = math.sqrt(1.0 / self.in_channels)
#         self.q_weights = nn.Parameter(scale * torch.randn(self.k_channels, self.in_channels, 1, 1))
#         self.k_weights = nn.Parameter(scale * torch.randn(self.k_channels, self.in_channels, 1, 1))
#         self.v_weights = nn.Parameter(scale * torch.randn(self.out_channels, self.in_channels, 1, 1))

#         if bias:
#             self.q_bias = nn.Parameter(torch.zeros(self.k_channels))
#             self.k_bias = nn.Parameter(torch.zeros(self.k_channels))
#             self.v_bias = nn.Parameter(torch.zeros(self.out_channels))
#         else:
#             self.q_bias = None
#             self.k_bias = None
#             self.v_bias = None

#     def extra_repr(self):
#         r"""
#         Pretty print module
#         """
#         return f"in_shape={(self.nlat_in, self.nlon_in)}, out_shape={(self.nlat_out, self.nlon_out)}, in_channels={self.in_channels}, out_channels={self.out_channels}, k_channels={self.k_channels}"

#     def forward(self, query: torch.Tensor, key: Optional[torch.Tensor] = None, value: Optional[torch.Tensor] = None) -> torch.Tensor:

#         # self attention simplification
#         if key is None:
#             key = query

#         if value is None:
#             value = query

#         # change this later to allow arbitrary number of batch dims
#         assert (query.dim() == key.dim()) and (key.dim() == value.dim()) and (value.dim() == 4)

#         # add checks if dimensions match

#         # reshape to the right dimensions
#         query = query.permute(0,2,3,1).reshape(-1, self.nlat_out * nlon_out, self.in_channels)
#         key = key.permute(0,2,3,1).reshape(-1, self.nlat_in * nlon_in, self.in_channels)
#         value = value.permute(0,2,3,1).reshape(-1, self.nlat_in * nlon_in, self.in_channels)

#         # multiply the query, key and value tensors

#         out = nn.functional.scaled_dot_product_attention()

#         return out


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
        idx, vals = _precompute_convolution_tensor_s2(
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

        # compute row offsets for more structured traversal.
        # only works if rows are sorted but they are by construction
        row_offset = np.empty(self.nlat_out + 1, dtype=np.int64)
        row_offset[0] = 0
        row = row_idx[0]
        for idz, z in enumerate(range(col_idx.shape[0])):
            if row_idx[z] != row:
                row_offset[row + 1] = idz
                row = row_idx[z]

        # set the last value
        row_offset[row + 1] = idz + 1
        row_offset = torch.from_numpy(row_offset)
        self.max_psi_nnz = col_idx.max().item() + 1

        self.register_buffer("psi_row_idx", row_idx, persistent=False)
        self.register_buffer("psi_col_idx", col_idx, persistent=False)
        self.register_buffer("psi_roff_idx", row_offset, persistent=False)
        # self.register_buffer("psi_vals", vals, persistent=False)

        # learnable parameters
        # TODO: double-check that this gives us the correct initialization magnitudes
        scale = math.sqrt(1.0 / self.in_channels)
        self.q_weights = nn.Parameter(scale * torch.randn(self.k_channels, self.in_channels, 1, 1))
        self.k_weights = nn.Parameter(scale * torch.randn(self.k_channels, self.in_channels, 1, 1))
        self.v_weights = nn.Parameter(scale * torch.randn(self.out_channels, self.in_channels, 1, 1))

        if scale is not None:
            self.scale = scale
        else:
            self.scale = 1 / math.sqrt(k_channels)
        
        if bias:
            self.q_bias = nn.Parameter(torch.zeros(self.k_channels))
            self.k_bias = nn.Parameter(torch.zeros(self.k_channels))
            self.v_bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

    def extra_repr(self):
        r"""
        Pretty print module
        """
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
        if query.is_cuda and _cuda_extension_available:
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
                self.nlon_in,
                self.nlat_out,
                self.nlon_out,
            )

        return out
