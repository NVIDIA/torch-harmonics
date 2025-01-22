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

from torch_harmonics.quadrature import _precompute_grid, _precompute_latitudes
from torch_harmonics._disco_convolution import _disco_s2_contraction_torch, _disco_s2_transpose_contraction_torch
from torch_harmonics._disco_convolution import _disco_s2_contraction_cuda, _disco_s2_transpose_contraction_cuda
from torch_harmonics.filter_basis import get_filter_basis
from torch_harmonics.convolution import (
    _normalize_convolution_tensor_s2,
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


def _precompute_distributed_convolution_tensor_s2(
    in_shape,
    out_shape,
    filter_basis,
    grid_in="equiangular",
    grid_out="equiangular",
    theta_cutoff=0.01 * math.pi,
    theta_eps = 1e-3,
    transpose_normalization=False,
    basis_norm_mode="mean",
    merge_quadrature=False,
):
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
    """

    assert len(in_shape) == 2
    assert len(out_shape) == 2

    kernel_size = filter_basis.kernel_size

    nlat_in, nlon_in = in_shape
    nlat_out, nlon_out = out_shape

    # precompute input and output grids
    lats_in, win = _precompute_latitudes(nlat_in, grid=grid_in)
    lats_out, wout = _precompute_latitudes(nlat_out, grid=grid_out)

    # compute the phi differences
    # It's imporatant to not include the 2 pi point in the longitudes, as it is equivalent to lon=0
    lons_in = torch.linspace(0, 2 * math.pi, nlon_in + 1, dtype=torch.float64)[:-1]

    # compute quadrature weights and merge them into the convolution tensor.
    # These quadrature integrate to 1 over the sphere.
    if transpose_normalization:
        quad_weights = wout.reshape(-1, 1) / nlon_in / 2.0
    else:
        quad_weights = win.reshape(-1, 1) / nlon_in / 2.0

    # effective theta cutoff if multiplied with a fudge factor to avoid aliasing with grid width (especially near poles)
    theta_cutoff_eff = (1.0 + theta_eps) * theta_cutoff

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
        x = torch.cos(alpha) * torch.cos(beta) * torch.sin(gamma) + torch.cos(gamma) * torch.sin(alpha)
        y = torch.sin(beta) * torch.sin(gamma)
        z = -torch.cos(beta) * torch.sin(alpha) * torch.sin(gamma) + torch.cos(alpha) * torch.cos(gamma)

        # normalization is important to avoid NaNs when arccos and atan are applied
        # this can otherwise lead to spurious artifacts in the solution
        norm = torch.sqrt(x * x + y * y + z * z)
        x = x / norm
        y = y / norm
        z = z / norm

        # compute spherical coordinates, where phi needs to fall into the [0, 2pi) range
        theta = torch.arccos(z)
        phi = torch.arctan2(y, x)
        phi = torch.where(phi < 0.0, phi + 2 * torch.pi, phi)

        # find the indices where the rotated position falls into the support of the kernel
        iidx, vals = filter_basis.compute_support_vals(theta, phi, r_cutoff=theta_cutoff_eff)

        # add the output latitude and reshape such that psi has dimensions kernel_shape x nlat_out x (nlat_in*nlon_in)
        idx = torch.stack([iidx[:, 0], t * torch.ones_like(iidx[:, 0]), iidx[:, 1] * nlon_in + iidx[:, 2]], dim=0)

        # append indices and values to the COO datastructure
        out_idx.append(idx)
        out_vals.append(vals)

    # concatenate the indices and values
    out_idx = torch.cat(out_idx, dim=-1)
    out_vals = torch.cat(out_vals, dim=-1)

    out_vals = _normalize_convolution_tensor_s2(
        out_idx,
        out_vals,
        in_shape,
        out_shape,
        kernel_size,
        quad_weights,
        transpose_normalization=transpose_normalization,
        basis_norm_mode=basis_norm_mode,
        merge_quadrature=merge_quadrature,
    )

    # TODO: this part can be split off into it's own function
    # split the latitude indices:
    comm_size_polar = polar_group_size()
    comm_rank_polar = polar_group_rank()
    split_shapes = compute_split_shapes(nlat_in, num_chunks=comm_size_polar)
    offsets = [0] + list(accumulate(split_shapes))
    start_idx = offsets[comm_rank_polar]
    end_idx = offsets[comm_rank_polar + 1]

    # once normalization is done we can throw away the entries which correspond to input latitudes we do not care about
    lats = out_idx[2] // nlon_in
    lons = out_idx[2] % nlon_in
    ilats = torch.argwhere((lats < end_idx) & (lats >= start_idx)).squeeze()
    out_vals = out_vals[ilats]
    # for the indices we need to recompute them to refer to local indices of the input tenor
    out_idx = torch.stack([out_idx[0, ilats], out_idx[1, ilats], (lats[ilats] - start_idx) * nlon_in + lons[ilats]], dim=0)

    out_idx = out_idx.contiguous()
    out_vals = out_vals.to(dtype=torch.float32).contiguous()

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

        idx, vals = _precompute_distributed_convolution_tensor_s2(
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

    def extra_repr(self):
        r"""
        Pretty print module
        """
        return f"in_shape={(self.nlat_in, self.nlon_in)}, out_shape={(self.nlat_out, self.nlon_out)}, in_chans={self.groupsize * self.groups}, out_chans={self.weight.shape[0]}, filter_basis={self.filter_basis}, kernel_shape={self.kernel_shape}, groups={self.groups}"

    @property
    def psi_idx(self):
        return torch.stack([self.psi_ker_idx, self.psi_row_idx, self.psi_col_idx], dim=0).contiguous()

    def get_psi(self):
        psi = torch.sparse_coo_tensor(self.psi_idx, self.psi_vals, size=(self.kernel_size, self.nlat_out_local, self.nlat_in_local * self.nlon_in)).coalesce()
        return psi

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

            psi = self.get_psi()

            x = _disco_s2_contraction_torch(x, psi, self.nlon_out)

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

        # switch in_shape and out_shape since we want transpose conv
        # distributed mode here is swapped because of the transpose
        idx, vals = _precompute_distributed_convolution_tensor_s2(
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

    def extra_repr(self):
        r"""
        Pretty print module
        """
        return f"in_shape={(self.nlat_in, self.nlon_in)}, out_shape={(self.nlat_out, self.nlon_out)}, in_chans={self.groupsize * self.groups}, out_chans={self.weight.shape[0]}, filter_basis={self.filter_basis}, kernel_shape={self.kernel_shape}, groups={self.groups}"

    @property
    def psi_idx(self):
        return torch.stack([self.psi_ker_idx, self.psi_row_idx, self.psi_col_idx], dim=0).contiguous()

    def get_psi(self, semi_transposed: bool = False):
        if semi_transposed:
            # do partial transpose
            # we do a semi-transposition to faciliate the computation
            tout = self.psi_idx[2] // self.nlon_out
            pout = self.psi_idx[2] % self.nlon_out
            # flip the axis of longitudes
            pout = self.nlon_out - 1 - pout
            tin = self.psi_idx[1]
            idx = torch.stack([self.psi_idx[0], tout, tin * self.nlon_out + pout], dim=0)
            psi = torch.sparse_coo_tensor(idx, self.psi_vals, size=(self.kernel_size, self.nlat_out_local, self.nlat_in_local * self.nlon_out)).coalesce()
        else:
            psi = torch.sparse_coo_tensor(self.psi_idx, self.psi_vals, size=(self.kernel_size, self.nlat_in_local, self.nlat_out_local * self.nlon_out)).coalesce()
        return psi

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
            psi = self.get_psi(semi_transposed=True)
            out = _disco_s2_transpose_contraction_torch(x, psi, self.nlon_out)

        # now we can transpose back the result, so that lon is split and channels are local
        if self.comm_size_azimuth > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            out = distributed_transpose_azimuth.apply(out, (-1, 1), chan_shapes)

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)

        return out
