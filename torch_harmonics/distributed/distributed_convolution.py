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

import math

import torch
import torch.nn as nn

from functools import partial

from torch_harmonics.quadrature import _precompute_grid, _precompute_latitudes
from torch_harmonics._disco_convolution import (
    _disco_s2_contraction_torch,
    _disco_s2_transpose_contraction_torch,
    _disco_s2_contraction_triton,
    _disco_s2_transpose_contraction_triton,
)

from torch_harmonics.convolution import (
    _compute_support_vals_isotropic,
    _compute_support_vals_anisotropic,
    _precompute_convolution_tensor_2d,
    DiscreteContinuousConv,
)

from torch_harmonics.distributed import polar_group_size, azimuth_group_size
from torch_harmonics.distributed import distributed_transpose_azimuth, distributed_transpose_polar
from torch_harmonics.distributed import reduce_from_polar_region, scatter_to_polar_region
from torch_harmonics.distributed import polar_group_rank, azimuth_group_rank
from torch_harmonics.distributed import compute_split_shapes, split_tensor_along_dim

def _precompute_distributed_convolution_tensor_s2(in_shape, out_shape, kernel_shape, grid_in="equiangular", grid_out="equiangular",
                                                  theta_cutoff=0.01 * math.pi, distributed_mode="columns"):
    """
    Precomputes the rotated filters at positions $R^{-1}_j \omega_i = R^{-1}_j R_i \nu = Y(-\theta_j)Z(\phi_i - \phi_j)Y(\theta_j)\nu$.
    Assumes a tensorized grid on the sphere with an equidistant sampling in longitude as described in Ocampo et al.
    The output tensor has shape kernel_shape x nlat_out x (nlat_in * nlon_in).

    The rotation of the Euler angles uses the YZY convention, which applied to the northpole $(0,0,1)^T$ yields
    $$
    Y(\alpha) Z(\beta) Y(\gamma) n =
        {\begin{bmatrix} 
            \cos(\gamma)\sin(\alpha) + \cos(\alpha)\cos(\beta)\sin(\gamma) \\
            \sin(\beta)\sin(\gamma) \\
            \cos(\alpha)\cos(\gamma)-\cos(\beta)\sin(\alpha)\sin(\gamma)
        \end{bmatrix}}
    $$

    This is the distributed version: the matrix can either be split column- or row-wise. Column-wise seems better because the kernel has a lot of summation
    atomics concerning the row reductions, which we can combine in a single allreduce.
    """

    assert len(in_shape) == 2
    assert len(out_shape) == 2

    if len(kernel_shape) == 1:
        kernel_handle = partial(_compute_support_vals_isotropic, nr=kernel_shape[0], r_cutoff=theta_cutoff, norm="s2")
    elif len(kernel_shape) == 2:
        kernel_handle = partial(_compute_support_vals_anisotropic, nr=kernel_shape[0], nphi=kernel_shape[1], r_cutoff=theta_cutoff, norm="s2")
    else:
        raise ValueError("kernel_shape should be either one- or two-dimensional.")

    nlat_in, nlon_in = in_shape
    nlat_out, nlon_out = out_shape

    lats_in, _ = _precompute_latitudes(nlat_in, grid=grid_in)
    lats_in = torch.from_numpy(lats_in).float()
    lats_out, _ = _precompute_latitudes(nlat_out, grid=grid_out)
    lats_out = torch.from_numpy(lats_out).float()

    # split the latitude vector:
    comm_size_polar = polar_group_size()
    comm_rank_polar = polar_group_rank()
    if distributed_mode == "columns":
        lats_in = split_tensor_along_dim(lats_in, dim=0, num_chunks=comm_size_polar)[comm_rank_polar]
    elif distributed_mode == "rows":
        lats_out = split_tensor_along_dim(lats_out, dim=0, num_chunks=comm_size_polar)[comm_rank_polar]
        nlat_out = lats_out.shape[0]
    else:
        raise NotImplementedError(f"Error, unknown distributed mode {distributed_mode}.")

    # compute the phi differences
    # It's imporatant to not include the 2 pi point in the longitudes, as it is equivalent to lon=0
    lons_in = torch.linspace(0, 2 * math.pi, nlon_in + 1)[:-1]
    
    out_idx = []
    out_vals = []
    for t in range(nlat_out):
        # the last angle has a negative sign as it is a passive rotation, which rotates the filter around the y-axis
        alpha = -lats_out[t]
        beta = lons_in
        gamma = lats_in.reshape(-1, 1)

        # compute cartesian coordinates of the rotated position
        # This uses the YZY convention of Euler angles, where the last angle (alpha) is a passive rotation,
        # and therefore applied with a negative sign
        z = -torch.cos(beta) * torch.sin(alpha) * torch.sin(gamma) + torch.cos(alpha) * torch.cos(gamma)
        x = torch.cos(alpha) * torch.cos(beta) * torch.sin(gamma) + torch.cos(gamma) * torch.sin(alpha)
        y = torch.sin(beta) * torch.sin(gamma)

        # normalization is emportant to avoid NaNs when arccos and atan are applied
        # this can otherwise lead to spurious artifacts in the solution
        norm = torch.sqrt(x * x + y * y + z * z)
        x = x / norm
        y = y / norm
        z = z / norm

        # compute spherical coordinates, where phi needs to fall into the [0, 2pi) range
        theta = torch.arccos(z)
        phi = torch.arctan2(y, x) + torch.pi

        # find the indices where the rotated position falls into the support of the kernel
        iidx, vals = kernel_handle(theta, phi)

        # add the output latitude and reshape such that psi has dimensions kernel_shape x nlat_out x (nlat_in*nlon_in)
        idx = torch.stack([iidx[:, 0], t * torch.ones_like(iidx[:, 0]), iidx[:, 1] * nlon_in + iidx[:, 2]], dim=0)

        # append indices and values to the COO datastructure
        out_idx.append(idx)
        out_vals.append(vals)

    # concatenate the indices and values
    out_idx = torch.cat(out_idx, dim=-1)
    out_vals = torch.cat(out_vals, dim=-1)

    return out_idx, out_vals

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
        kernel_shape: Union[int, List[int]],
        groups: Optional[int] = 1,
        grid_in: Optional[str] = "equiangular",
        grid_out: Optional[str] = "equiangular",
        bias: Optional[bool] = True,
        theta_cutoff: Optional[float] = None,
    ):
        super().__init__(in_channels, out_channels, kernel_shape, groups, bias)

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
            theta_cutoff = (self.kernel_shape[0] + 1) * torch.pi / float(self.nlat_in - 1)

        if theta_cutoff <= 0.0:
            raise ValueError("Error, theta_cutoff has to be positive.")

        # integration weights
        _, wgl = _precompute_latitudes(self.nlat_in, grid=grid_in)
        quad_weights = 2.0 * torch.pi * torch.from_numpy(wgl).float().reshape(-1, 1) / float(self.nlon_in)
        
        # Note that the psi matrix is of shape nlat_out x nlat_in * nlon_in. Since the contraction in nlon direction is a convolution,
        # we will keep local to all nodes and split the computation up along nlat. We further split the input dim because this reduces the number
        # of atomic reduction calls inside the actual kernel
        distributed_mode = "columns"

        # set local shapes according to distributed mode:
        if distributed_mode == "columns":
            self.nlat_in_local = self.lat_in_shapes[self.comm_rank_polar]
            self.nlat_out_local = self.nlat_out
        elif distributed_mode == "rows":
            self.nlat_in_local = self.nlat_in
            self.nlat_out_local = self.lat_out_shapes[self.comm_rank_polar]
        else:
            raise NotImplementedError(f"Error, unknown distributed mode {distributed_mode}.")
        
        idx, vals = _precompute_distributed_convolution_tensor_s2(in_shape, out_shape, self.kernel_shape, grid_in=grid_in, grid_out=grid_out,
                                                                  theta_cutoff=theta_cutoff, distributed_mode=distributed_mode)
        # split the weight tensor as well
        if distributed_mode == "columns":
            quad_weights = split_tensor_along_dim(quad_weights, dim=0, num_chunks=self.comm_size_polar)[self.comm_rank_polar]

        self.register_buffer("quad_weights", quad_weights, persistent=False)
        self.register_buffer("psi_idx", idx, persistent=False)
        self.register_buffer("psi_vals", vals, persistent=False)

    def get_psi(self):
        psi = torch.sparse_coo_tensor(self.psi_idx, self.psi_vals, size=(self.kernel_size, self.nlat_out_local, self.nlat_in_local * self.nlon_in)).coalesce()
        return psi

    def forward(self, x: torch.Tensor, use_triton_kernel: bool = True) -> torch.Tensor:

        # store number of channels
        num_chans = x.shape[1]
        #print("input shape", x.shape)
        
        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_azimuth > 1:
            x = distributed_transpose_azimuth.apply(x, (1, -1), self.lon_in_shapes)

        #print("transposed shape", x.shape)
            
        # pre-multiply x with the quadrature weights
        x = self.quad_weights * x

        #print("multiplied shape", x.shape)

        psi = self.get_psi()

        #print("psi shape", psi.shape)

        if x.is_cuda and use_triton_kernel:
            x = _disco_s2_contraction_triton(x, psi, self.nlon_out)
        else:
            x = _disco_s2_contraction_torch(x, psi, self.nlon_out)

        #print("psi * x shape", x.shape)

        # allreduce over latitudes: h is still local
        x = reduce_from_polar_region(x)

        #print("reduced shape", x.shape)

        # split tensor along latitudes: h is split
        x = scatter_to_polar_region(x, -2)

        #print("scattered shape", x.shape)

        # now we can transpose back the result, so that lon is split and channels are local
        if self.comm_size_azimuth > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = distributed_transpose_azimuth.apply(x, (-1, 1), chan_shapes)

        # extract shape
        B, C, K, H, W = x.shape
        x = x.reshape(B, self.groups, self.groupsize, K, H, W)

        # do weight multiplication
        out = torch.einsum("bgckxy,gock->bgoxy", x, self.weight.reshape(self.groups, -1, self.weight.shape[1], self.weight.shape[2])).contiguous()
        out = out.reshape(out.shape[0], -1, out.shape[-2], out.shape[-1])

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)

        return out


class DistributedDiscreteContinuousConvTransposeS2(DiscreteContinuousConv):
    """
    Discrete-continuous transpose convolutions (DISCO) on the 2-Sphere as described in [1].

    [1] Ocampo, Price, McEwen, Scalable and equivariant spherical CNNs by discrete-continuous (DISCO) convolutions, ICLR (2023), arXiv:2209.13603
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_shape: Tuple[int],
        out_shape: Tuple[int],
        kernel_shape: Union[int, List[int]],
        groups: Optional[int] = 1,
        grid_in: Optional[str] = "equiangular",
        grid_out: Optional[str] = "equiangular",
        bias: Optional[bool] = True,
        theta_cutoff: Optional[float] = None,
    ):
        super().__init__(in_channels, out_channels, kernel_shape, groups, bias)

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
            theta_cutoff = (self.kernel_shape[0] + 1) * torch.pi / float(self.nlat_in - 1)

        if theta_cutoff <= 0.0:
            raise ValueError("Error, theta_cutoff has to be positive.")

        # integration weights
        _, wgl = _precompute_latitudes(self.nlat_in, grid=grid_in)
        quad_weights = 2.0 * torch.pi * torch.from_numpy(wgl).float().reshape(-1, 1) / self.nlon_in

        # Note that the psi matrix is of shape nlat_out x nlat_in * nlon_in. Since the contraction in nlon direction is a convolution,
        # we will keep local to all nodes and split the computation up along nlat. We further split the input dim because this reduces the number
	# of atomic reduction calls inside the actual kernel
        distributed_mode = "columns"

        # set local shapes according to distributed mode:
        if distributed_mode == "columns":
            self.nlat_in_local = self.lat_in_shapes[self.comm_rank_polar]
            self.nlat_out_local = self.nlat_out
        elif distributed_mode == "rows":
            self.nlat_in_local = self.nlat_in
            self.nlat_out_local = self.lat_out_shapes[self.comm_rank_polar]
        else:
            raise NotImplementedError(f"Error, unknown distributed mode {distributed_mode}.")

        # switch in_shape and out_shape since we want transpose conv
        # distributed mode here is swapped because of the transpose
        idx, vals = _precompute_distributed_convolution_tensor_s2(out_shape, in_shape, self.kernel_shape, grid_in=grid_out, grid_out=grid_in,
                                                                  theta_cutoff=theta_cutoff, distributed_mode="rows" if distributed_mode else "columns")

        ## do partial transpose
        ## we do a semi-transposition to faciliate the computation
        #tout = iidx[2] // self.nlon_out
        #pout = iidx[2] % self.nlon_out
        ## flip the axis of longitudes
        #pout = self.nlon_out - 1 - pout
        #tin = iidx[1]
        #idx = torch.stack([iidx[0], tout, tin*self.nlon_out + pout], dim=0)
        
        # split the weight tensor as well
        if distributed_mode == "columns":
            quad_weights = split_tensor_along_dim(quad_weights, dim=0, num_chunks=self.comm_size_polar)[self.comm_rank_polar]

        # register all buffers
        self.register_buffer("quad_weights", quad_weights, persistent=False)
        self.register_buffer("psi_idx", idx, persistent=False)
        self.register_buffer("psi_vals", vals, persistent=False)

    def get_psi(self, use_triton_kernel=True):
        if not use_triton_kernel:
            # do partial transpose
            # we do a semi-transposition to faciliate the computation
            tout = self.psi_idx[2] // self.nlon_out
            pout = self.psi_idx[2] % self.nlon_out
            # flip the axis of longitudes
            pout = self.nlon_out - 1 - pout
            tin = self.psi_idx[1]
            idx = torch.stack([self.psi_idx[0], tout, tin*self.nlon_out + pout], dim=0)
            psi = torch.sparse_coo_tensor(idx, self.psi_vals, size=(self.kernel_size, self.nlat_out_local, self.nlat_in_local * self.nlon_out)).coalesce()
        else:
            psi = torch.sparse_coo_tensor(self.psi_idx, self.psi_vals, size=(self.kernel_size, self.nlat_in_local, self.nlat_out_local * self.nlon_out)).coalesce()
        return psi
    
    def forward(self, x: torch.Tensor, use_triton_kernel: bool = True) -> torch.Tensor:
        # extract shape
        B, C, H, W = x.shape
        x = x.reshape(B, self.groups, self.groupsize, H, W)

        # do weight multiplication
        x = torch.einsum("bgcxy,gock->bgokxy", x, self.weight.reshape(self.groups, -1, self.weight.shape[1], self.weight.shape[2])).contiguous()
        x = x.reshape(x.shape[0], -1, x.shape[-3], x.shape[-2], x.shape[-1])
        num_chans = x.shape[1]
        
        # transpose such that lon is local, channels are split
        if self.comm_size_azimuth > 1:
            x = distributed_transpose_azimuth.apply(x, (1, -1), self.lon_in_shapes)
        
        # pre-multiply x with the quadrature weights
        x = self.quad_weights * x
        
        if x.is_cuda and use_triton_kernel:
            psi = self.get_psi(True)
            out = _disco_s2_transpose_contraction_triton(x, psi, self.nlon_out)
        else:
            psi = self.get_psi(False)
            out = _disco_s2_transpose_contraction_torch(x, psi, self.nlon_out)
            
        # allreduce over latitudes: h is still local
        out = reduce_from_polar_region(out)

        # split tensor along latitudes: h is split
        out = scatter_to_polar_region(out, -2)

        # now we can transpose back the result, so that lon is split and channels are local
        if self.comm_size_azimuth > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            out = distributed_transpose_azimuth.apply(out, (-1, 1), chan_shapes)

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)

        return out

