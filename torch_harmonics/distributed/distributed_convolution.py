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

import abc
from typing import List, Tuple, Union, Optional
from itertools import accumulate
from warnings import warn

import math

import torch
import torch.nn as nn

from functools import partial

from torch_harmonics.quadrature import _precompute_grid, _precompute_latitudes, _precompute_longitudes
from torch_harmonics._disco_convolution import _get_psi, _disco_s2_contraction_torch, _disco_s2_transpose_contraction_torch
from torch_harmonics._disco_convolution import _disco_s2_contraction_cuda, _disco_s2_transpose_contraction_cuda
from torch_harmonics.filter_basis import get_filter_basis
from torch_harmonics.convolution import (
    _precompute_convolution_tensor_s2,
    DiscreteContinuousConv,
)


from torch_harmonics.distributed import polar_group_size, azimuth_group_size
from torch_harmonics.distributed import distributed_transpose_azimuth, distributed_transpose_polar
from torch_harmonics.distributed import reduce_from_polar_region, scatter_to_polar_region, gather_from_polar_region, copy_to_polar_region
from torch_harmonics.distributed import polar_group_rank, azimuth_group_rank
from torch_harmonics.distributed import compute_split_shapes, split_tensor_along_dim

# import custom C++/CUDA extensions if available
try:
    from disco_helpers import preprocess_psi
    import disco_cuda_extension

    _cuda_extension_available = True
except ImportError as err:
    disco_cuda_extension = None
    _cuda_extension_available = False


def _split_distributed_convolution_tensor_s2(
    idx: torch.Tensor,
    vals: torch.Tensor,
    in_shape: Tuple[int],
    out_shape: Tuple[int],
):
    nlat_in, nlon_in = in_shape
    nlat_out, nlon_out = out_shape

    comm_size_polar = polar_group_size()
    comm_rank_polar = polar_group_rank()
    split_shapes = compute_split_shapes(nlat_in, num_chunks=comm_size_polar)
    offsets = [0] + list(accumulate(split_shapes))
    start_idx = offsets[comm_rank_polar]
    end_idx = offsets[comm_rank_polar + 1]

    # once normalization is done we can throw away the entries which correspond to input latitudes we do not care about
    lats = idx[2] // nlon_in
    lons = idx[2] % nlon_in
    ilats = torch.argwhere((lats < end_idx) & (lats >= start_idx)).squeeze()
    vals = vals[ilats]
    # for the indices we need to recompute them to refer to local indices of the input tenor
    idx = torch.stack([idx[0, ilats], idx[1, ilats], (lats[ilats] - start_idx) * nlon_in + lons[ilats]], dim=0)

    # make results contiguous
    idx = idx.contiguous()
    vals = vals.to(dtype=torch.float32).contiguous()

    return idx, vals


class DistributedDiscreteContinuousConvS2(DiscreteContinuousConv):
    """
    Distributed version of Discrete-continuous convolutions (DISCO) on the 2-Sphere as described in [1].

    [1] Ocampo, Price, McEwen, Scalable and equivariant spherical CNNs by discrete-continuous (DISCO) convolutions, ICLR (2023), arXiv:2209.13603

    We assume the data can be splitted in polar and azimuthal directions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_shape: Tuple[int],
        out_shape: Tuple[int],
        kernel_shape: Union[int, Tuple[int], Tuple[int, int]],
        basis_type: Optional[str] = "piecewise linear",
        basis_norm_mode: Optional[str] = "mean",
        groups: Optional[int] = 1,
        grid_in: Optional[str] = "equiangular",
        grid_out: Optional[str] = "equiangular",
        bias: Optional[bool] = True,
        theta_cutoff: Optional[float] = None,
    ):
        super().__init__(in_channels, out_channels, kernel_shape, basis_type, groups, bias)

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

        # compute theta cutoff based on the bandlimit of the input field
        if theta_cutoff is None:
            theta_cutoff = torch.pi / float(self.nlat_out - 1)

        if theta_cutoff <= 0.0:
            raise ValueError("Error, theta_cutoff has to be positive.")

        # Note that the psi matrix is of shape nlat_out x nlat_in * nlon_in. Since the contraction in nlon direction is a convolution,
        # we will keep local to all nodes and split the computation up along nlat. We further split the input dim because this reduces the number
        # of atomic reduction calls inside the actual kernel

        # set local shapes according to distributed mode:
        self.nlat_in_local = self.lat_in_shapes[self.comm_rank_polar]
        self.nlat_out_local = self.nlat_out

        # compute global convolution tensor
        idx, vals, _ = _precompute_convolution_tensor_s2(
            in_shape,
            out_shape,
            self.filter_basis,
            grid_in=grid_in,
            grid_out=grid_out,
            theta_cutoff=theta_cutoff,
            transpose_normalization=False,
            basis_norm_mode=basis_norm_mode,
            merge_quadrature=True,
        )

        # split the convolution tensor along latitude
        idx, vals = _split_distributed_convolution_tensor_s2(idx, vals, in_shape, out_shape)

        # sort the values
        ker_idx = idx[0, ...].contiguous()
        row_idx = idx[1, ...].contiguous()
        col_idx = idx[2, ...].contiguous()
        vals = vals.contiguous()

        if _cuda_extension_available:
            # preprocessed data-structure for GPU kernel
            roff_idx = preprocess_psi(self.kernel_size, self.nlat_out_local, ker_idx, row_idx, col_idx, vals).contiguous()
            self.register_buffer("psi_roff_idx", roff_idx, persistent=False)

        # save all datastructures
        self.register_buffer("psi_ker_idx", ker_idx, persistent=False)
        self.register_buffer("psi_row_idx", row_idx, persistent=False)
        self.register_buffer("psi_col_idx", col_idx, persistent=False)
        self.register_buffer("psi_vals", vals, persistent=False)

        # store psi jic:
        self.psi = _get_psi(self.kernel_size, self.psi_idx, self.psi_vals, self.nlat_in, self.nlon_in, self.nlat_out, self.nlon_out, self.nlat_in_local, self.nlat_out_local)

    def extra_repr(self):
        r"""
        Pretty print module
        """
        return f"in_shape={(self.nlat_in, self.nlon_in)}, out_shape={(self.nlat_out, self.nlon_out)}, in_chans={self.groupsize * self.groups}, out_chans={self.weight.shape[0]}, filter_basis={self.filter_basis}, kernel_shape={self.kernel_shape}, groups={self.groups}"

    @property
    def psi_idx(self):
        return torch.stack([self.psi_ker_idx, self.psi_row_idx, self.psi_col_idx], dim=0).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # store number of channels
        num_chans = x.shape[1]

        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_azimuth > 1:
            x = distributed_transpose_azimuth.apply(x, (1, -1), self.lon_in_shapes)

        if x.is_cuda and _cuda_extension_available:
            x = _disco_s2_contraction_cuda(
                x, self.psi_roff_idx, self.psi_ker_idx, self.psi_row_idx, self.psi_col_idx, self.psi_vals, self.kernel_size, self.nlat_out_local, self.nlon_out
            )
        else:
            if x.is_cuda:
                warn("couldn't find CUDA extension, falling back to slow PyTorch implementation")

            x = _disco_s2_contraction_torch(x, self.psi.to(x.device), self.nlon_out)

        # perform reduce scatter in polar region
        x = reduce_from_polar_region(x)
        x = scatter_to_polar_region(x, -2)

        # now we can transpose back the result, so that lon is split and channels are local
        if self.comm_size_azimuth > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = distributed_transpose_azimuth.apply(x, (-1, 1), chan_shapes)

        # extract shape
        B, C, K, H, W = x.shape
        x = x.reshape(B, self.groups, self.groupsize, K, H, W)

        # do weight multiplication
        out = torch.einsum("bgckxy,gock->bgoxy", x, self.weight.reshape(self.groups, -1, self.weight.shape[1], self.weight.shape[2])).contiguous()
        out = out.reshape(out.shape[0], -1, H, W)

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)

        return out


class DistributedDiscreteContinuousConvTransposeS2(DiscreteContinuousConv):
    """
    Discrete-continuous transpose convolutions (DISCO) on the 2-Sphere as described in [1].

    [1] Ocampo, Price, McEwen, Scalable and equivariant spherical CNNs by discrete-continuous (DISCO) convolutions, ICLR (2023), arXiv:2209.13603

    We assume the data can be splitted in polar and azimuthal directions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_shape: Tuple[int],
        out_shape: Tuple[int],
        kernel_shape: Union[int, Tuple[int], Tuple[int, int]],
        basis_type: Optional[str] = "piecewise linear",
        basis_norm_mode: Optional[str] = "mean",
        groups: Optional[int] = 1,
        grid_in: Optional[str] = "equiangular",
        grid_out: Optional[str] = "equiangular",
        bias: Optional[bool] = True,
        theta_cutoff: Optional[float] = None,
    ):
        super().__init__(in_channels, out_channels, kernel_shape, basis_type, groups, bias)

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

        # bandlimit
        if theta_cutoff is None:
            theta_cutoff = torch.pi / float(self.nlat_in - 1)

        if theta_cutoff <= 0.0:
            raise ValueError("Error, theta_cutoff has to be positive.")

        # Note that the psi matrix is of shape nlat_out x nlat_in * nlon_in. Since the contraction in nlon direction is a convolution,
        # we will keep local to all nodes and split the computation up along nlat. We further split the input dim because this reduces the number
        # of atomic reduction calls inside the actual kernel

        # set local shapes according to distributed mode:
        self.nlat_in_local = self.nlat_in
        self.nlat_out_local = self.lat_out_shapes[self.comm_rank_polar]

        # compute global convolution tensor
        # switch in_shape and out_shape since we want transpose conv
        # distributed mode here is swapped because of the transpose
        idx, vals, _ = _precompute_convolution_tensor_s2(
            out_shape,
            in_shape,
            self.filter_basis,
            grid_in=grid_out,
            grid_out=grid_in,
            theta_cutoff=theta_cutoff,
            transpose_normalization=True,
            basis_norm_mode=basis_norm_mode,
            merge_quadrature=True,
        )

        # split the convolution tensor along latitude, again, we need to swap the meaning
        # of in_shape and out_shape
        idx, vals = _split_distributed_convolution_tensor_s2(idx, vals, out_shape, in_shape)

        # sort the values
        ker_idx = idx[0, ...].contiguous()
        row_idx = idx[1, ...].contiguous()
        col_idx = idx[2, ...].contiguous()
        vals = vals.contiguous()

        if _cuda_extension_available:
            # preprocessed data-structure for GPU kernel
            roff_idx = preprocess_psi(self.kernel_size, self.nlat_in_local, ker_idx, row_idx, col_idx, vals).contiguous()
            self.register_buffer("psi_roff_idx", roff_idx, persistent=False)

        # save all datastructures
        self.register_buffer("psi_ker_idx", ker_idx, persistent=False)
        self.register_buffer("psi_row_idx", row_idx, persistent=False)
        self.register_buffer("psi_col_idx", col_idx, persistent=False)
        self.register_buffer("psi_vals", vals, persistent=False)

        # store psi as COO
        self.psi_st = _get_psi(self.kernel_size, self.psi_idx, self.psi_vals, self.nlat_in, self.nlon_in, self.nlat_out, self.nlon_out, self.nlat_in_local, self.nlat_out_local, semi_transposed=True)

    def extra_repr(self):
        r"""
        Pretty print module
        """
        return f"in_shape={(self.nlat_in, self.nlon_in)}, out_shape={(self.nlat_out, self.nlon_out)}, in_chans={self.groupsize * self.groups}, out_chans={self.weight.shape[0]}, filter_basis={self.filter_basis}, kernel_shape={self.kernel_shape}, groups={self.groups}"

    @property
    def psi_idx(self):
        return torch.stack([self.psi_ker_idx, self.psi_row_idx, self.psi_col_idx], dim=0).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # extract shape
        B, C, H, W = x.shape
        x = x.reshape(B, self.groups, self.groupsize, H, W)

        # do weight multiplication
        x = torch.einsum("bgcxy,gock->bgokxy", x, self.weight.reshape(self.groups, -1, self.weight.shape[1], self.weight.shape[2])).contiguous()
        x = x.reshape(B, -1, x.shape[-3], H, W)
        num_chans = x.shape[1]

        # transpose such that lon is local, channels are split
        if self.comm_size_azimuth > 1:
            x = distributed_transpose_azimuth.apply(x, (1, -1), self.lon_in_shapes)

        # gather input tensor and set up backward reduction hooks
        x = gather_from_polar_region(x, -2, self.lat_in_shapes)
        x = copy_to_polar_region(x)

        if x.is_cuda and _cuda_extension_available:
            out = _disco_s2_transpose_contraction_cuda(
                x, self.psi_roff_idx, self.psi_ker_idx, self.psi_row_idx, self.psi_col_idx, self.psi_vals, self.kernel_size, self.nlat_out_local, self.nlon_out
            )
        else:
            if x.is_cuda:
                warn("couldn't find CUDA extension, falling back to slow PyTorch implementation")
            out = _disco_s2_transpose_contraction_torch(x, self.psi_st.to(x.device), self.nlon_out)

        # now we can transpose back the result, so that lon is split and channels are local
        if self.comm_size_azimuth > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            out = distributed_transpose_azimuth.apply(out, (-1, 1), chan_shapes)

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)

        return out
