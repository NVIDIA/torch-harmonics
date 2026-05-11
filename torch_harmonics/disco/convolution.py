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
from typing import Tuple, Union, Optional

import math

import torch
import torch.nn as nn

from torch_harmonics.convolution_tensor_s2 import (
    _precompute_convolution_tensor_s2,
    _transpose_convolution_tensor_s2,
)
from ._disco_utils import _get_psi
from .optimized.disco_optimized import _disco_s2_contraction_optimized, _disco_s2_transpose_contraction_optimized
from .kernels_torch.disco_torch import _disco_s2_contraction_torch, _disco_s2_transpose_contraction_torch
from torch_harmonics.filter_basis import get_filter_basis
from disco_helpers import optimized_kernels_is_available, preprocess_psi


class DiscreteContinuousConv(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for discrete-continuous convolutions

    Parameters
    -----------
    in_channels: int
        Number of input channels
    out_channels: int
        Number of output channels
    kernel_shape: Union[int, Tuple[int], Tuple[int, int]]
        Shape of the kernel
    basis_type: Optional[str]
        Type of the basis functions
    groups: Optional[int]
        Number of groups
    bias: Optional[bool]
        Whether to use bias

    Returns
    -------
    out: torch.Tensor
        Output tensor
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_shape: Union[int, Tuple[int], Tuple[int, int]],
        basis_type: Optional[str] = "piecewise linear",
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
        optimized_kernel: Optional[bool] = True,
    ):
        super().__init__()

        self.kernel_shape = kernel_shape
        self.optimized_kernel = optimized_kernel and optimized_kernels_is_available()

        # get the filter basis functions
        self.filter_basis = get_filter_basis(kernel_shape=kernel_shape, basis_type=basis_type)

        # groups
        self.groups = groups

        # weight tensor
        if in_channels % self.groups != 0:
            raise ValueError("Error, the number of input channels has to be an integer multiple of the group size")
        if out_channels % self.groups != 0:
            raise ValueError("Error, the number of output channels has to be an integer multiple of the group size")
        self.groupsize = in_channels // self.groups
        scale = math.sqrt(1.0 / self.groupsize) * self.filter_basis.get_init_factors().reshape(1, 1, -1)
        self.weight = nn.Parameter(scale * torch.randn(out_channels, self.groupsize, self.kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    @property
    def kernel_size(self):
        return self.filter_basis.kernel_size

    @abc.abstractmethod
    def forward(self, x: torch.Tensor):
        raise NotImplementedError


class DiscreteContinuousConvS2(DiscreteContinuousConv):
    """
    Discrete-continuous (DISCO) convolutions on the 2-Sphere as described in [1].

    Parameters
    -----------
    in_channels: int
        Number of input channels
    out_channels: int
        Number of output channels
    in_shape: Tuple[int]
        Input shape of the convolution tensor
    out_shape: Tuple[int]
        Output shape of the convolution tensor
    kernel_shape: Union[int, Tuple[int], Tuple[int, int]]
        Shape of the kernel
    basis_type: Optional[str]
        Type of the basis functions
    basis_norm_mode: Optional[str]
        Mode for basis normalization
    groups: Optional[int]
        Number of groups
    grid_in: Optional[str]
        Input grid type
    grid_out: Optional[str]
        Output grid type
    bias: Optional[bool]
        Whether to use bias
    theta_cutoff: Optional[float]
        Theta cutoff for the filter basis functions
    optimized_kernel: Optional[bool]
        Whether to use the optimized kernel (if available)

    Returns
    -------
    out: torch.Tensor
        Output tensor

    References
    ----------
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
        basis_norm_mode: Optional[str] = "nodal",
        groups: Optional[int] = 1,
        grid_in: Optional[str] = "equiangular",
        grid_out: Optional[str] = "equiangular",
        bias: Optional[bool] = True,
        theta_cutoff: Optional[float] = None,
        optimized_kernel: Optional[bool] = True,
    ):
        super().__init__(in_channels, out_channels, kernel_shape, basis_type, groups, bias, optimized_kernel)

        self.nlat_in, self.nlon_in = in_shape
        self.nlat_out, self.nlon_out = out_shape

        # make sure the p-shift works by checking that longitudes are divisible
        assert self.nlon_in % self.nlon_out == 0, \
            f"nlon_in ({self.nlon_in}) must be an integer multiple of nlon_out ({self.nlon_out}) for the DISCO p-shift to be exact"

        # heuristic to compute theta cutoff based on the bandlimit of the input field and overlaps of the basis functions
        if theta_cutoff is None:
            self.theta_cutoff = torch.pi / float(self.nlat_out - 1)
        else:
            self.theta_cutoff = theta_cutoff

        if self.theta_cutoff <= 0.0:
            raise ValueError("Error, theta_cutoff has to be positive.")

        idx, vals, _ = _precompute_convolution_tensor_s2(
            in_shape,
            out_shape,
            self.filter_basis,
            grid_in=grid_in,
            grid_out=grid_out,
            theta_cutoff=self.theta_cutoff,
            transpose_normalization=False,
            basis_norm_mode=basis_norm_mode,
            merge_quadrature=True,
        )

        # sort the values
        ker_idx = idx[0, ...].contiguous()
        row_idx = idx[1, ...].contiguous()
        col_idx = idx[2, ...].contiguous()
        vals = vals.contiguous()

        if self.optimized_kernel:
            # preprocessed data-structure for forward GPU kernel
            # NOTE: preprocess_psi mutates (ker_idx, row_idx, col_idx, vals) in place
            # to be sorted by (ker, row); we build psi_T from the sorted arrays.
            roff_idx = preprocess_psi(self.kernel_size, self.nlat_out, ker_idx, row_idx, col_idx, vals).contiguous()
            self.register_buffer("psi_roff_idx", roff_idx, persistent=False)

            # psi_T data structure for the gather-based backward kernel.
            # Backward gathers per (b,c,hi,wi) cell from grad_out via psi_T,
            # avoiding the atomicAdd contention of a scatter implementation.
            ker_idx_T, col_idx_T, vals_T, roff_idx_T = _transpose_convolution_tensor_s2(
                ker_idx, row_idx, col_idx, vals,
                in_shape=(self.nlat_in, self.nlon_in),
                out_shape=(self.nlat_out, self.nlon_out),
            )
            self.register_buffer("psi_T_roff_idx", roff_idx_T, persistent=False)
            self.register_buffer("psi_T_ker_idx", ker_idx_T, persistent=False)
            self.register_buffer("psi_T_col_idx", col_idx_T, persistent=False)
            self.register_buffer("psi_T_vals", vals_T, persistent=False)

        # save all datastructures
        self.register_buffer("psi_ker_idx", ker_idx, persistent=False)
        self.register_buffer("psi_row_idx", row_idx, persistent=False)
        self.register_buffer("psi_col_idx", col_idx, persistent=False)
        self.register_buffer("psi_vals", vals, persistent=False)

        # also store psi as COO matrix just in case for torch input
        if not self.optimized_kernel:
            self.psi = _get_psi(self.kernel_size, self.psi_idx, self.psi_vals, self.nlat_in, self.nlon_in, self.nlat_out, self.nlon_out)

    def extra_repr(self):
        return f"in_shape={(self.nlat_in, self.nlon_in)}, out_shape={(self.nlat_out, self.nlon_out)}, in_chans={self.groupsize * self.groups}, out_chans={self.weight.shape[0]}, filter_basis={self.filter_basis}, kernel_shape={self.kernel_shape}, theta_cutoff={self.theta_cutoff}, groups={self.groups}"

    @property
    def psi_idx(self):
        return torch.stack([self.psi_ker_idx, self.psi_row_idx, self.psi_col_idx], dim=0).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.optimized_kernel:
            x = _disco_s2_contraction_optimized(
                x,
                # forward psi
                self.psi_roff_idx, self.psi_ker_idx, self.psi_row_idx,
                self.psi_col_idx, self.psi_vals,
                # backward psi_T (for autograd)
                self.psi_T_roff_idx, self.psi_T_ker_idx,
                self.psi_T_col_idx, self.psi_T_vals,
                self.kernel_size, self.nlat_out, self.nlon_out,
            )
        else:
            x = _disco_s2_contraction_torch(x, self.psi.to(x.device), self.nlon_out)

        # extract shape
        B, C, K, H, W = x.shape
        x = x.reshape(B, self.groups, self.groupsize, K, H, W)

        # do weight multiplication
        out = torch.einsum("bgckxy,gock->bgoxy", x, self.weight.reshape(self.groups, -1, self.weight.shape[1], self.weight.shape[2])).contiguous()
        out = out.reshape(B, -1, H, W)

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)

        return out


class DiscreteContinuousConvTransposeS2(DiscreteContinuousConv):
    """
    Discrete-continuous (DISCO) transpose convolutions on the 2-Sphere as described in [1].

    Parameters
    -----------
    in_channels: int
        Number of input channels
    out_channels: int
        Number of output channels
    in_shape: Tuple[int]
        Input shape of the convolution tensor
    out_shape: Tuple[int]
        Output shape of the convolution tensor
    kernel_shape: Union[int, Tuple[int], Tuple[int, int]]
        Shape of the kernel
    basis_type: Optional[str]
        Type of the basis functions
    basis_norm_mode: Optional[str]
        Mode for basis normalization
    groups: Optional[int]
        Number of groups
    grid_in: Optional[str]
        Input grid type
    grid_out: Optional[str]
        Output grid type
    bias: Optional[bool]
        Whether to use bias
    theta_cutoff: Optional[float]
        Theta cutoff for the filter basis functions
    optimized_kernel: Optional[bool]
        Whether to use the optimized kernel (if available)

    Returns
    --------
    out: torch.Tensor
        Output tensor

    References
    ----------
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
        basis_norm_mode: Optional[str] = "nodal",
        groups: Optional[int] = 1,
        grid_in: Optional[str] = "equiangular",
        grid_out: Optional[str] = "equiangular",
        bias: Optional[bool] = True,
        theta_cutoff: Optional[float] = None,
        optimized_kernel: Optional[bool] = True,
    ):
        super().__init__(in_channels, out_channels, kernel_shape, basis_type, groups, bias, optimized_kernel)

        self.nlat_in, self.nlon_in = in_shape
        self.nlat_out, self.nlon_out = out_shape

        # make sure the p-shift works by checking that longitudes are divisible
        assert self.nlon_out % self.nlon_in == 0, \
            f"nlon_out ({self.nlon_out}) must be an integer multiple of nlon_in ({self.nlon_in}) for the DISCO transpose p-shift to be exact"

        # bandlimit
        if theta_cutoff is None:
            self.theta_cutoff = torch.pi / float(self.nlat_in - 1)
        else:
            self.theta_cutoff = theta_cutoff

        if self.theta_cutoff <= 0.0:
            raise ValueError("Error, theta_cutoff has to be positive.")

        # switch in_shape and out_shape since we want the transpose convolution
        idx, vals, _ = _precompute_convolution_tensor_s2(
            out_shape,
            in_shape,
            self.filter_basis,
            grid_in=grid_out,
            grid_out=grid_in,
            theta_cutoff=self.theta_cutoff,
            transpose_normalization=True,
            basis_norm_mode=basis_norm_mode,
            merge_quadrature=True,
        )

        # sort the values
        ker_idx = idx[0, ...].contiguous()
        row_idx = idx[1, ...].contiguous()
        col_idx = idx[2, ...].contiguous()
        vals = vals.contiguous()

        if self.optimized_kernel:
            # preprocessed data-structure for forward GPU kernel.
            # NOTE: preprocess_psi mutates (ker_idx, row_idx, col_idx, vals)
            # in place to be sorted by (ker, row); psi_T is built from the
            # sorted arrays (it re-sorts by row_T anyway).
            roff_idx = preprocess_psi(self.kernel_size, self.nlat_in, ker_idx, row_idx, col_idx, vals).contiguous()
            self.register_buffer("psi_roff_idx", roff_idx, persistent=False)

            # psi_T for the gather-based backward kernel. The transpose
            # conv's psi was built via _precompute_convolution_tensor_s2(
            # out_shape, in_shape, ...) so its function-local Hi/Wi are the
            # transpose conv's (nlat_out, nlon_out) and Ho/Wo are
            # (nlat_in, nlon_in). _transpose_convolution_tensor_s2's in_shape param names the
            # grid that psi_T's row index spans — i.e., the bigger grid.
            ker_idx_T, col_idx_T, vals_T, roff_idx_T = _transpose_convolution_tensor_s2(
                ker_idx, row_idx, col_idx, vals,
                in_shape=(self.nlat_out, self.nlon_out),
                out_shape=(self.nlat_in, self.nlon_in),
            )
            self.register_buffer("psi_T_roff_idx", roff_idx_T, persistent=False)
            self.register_buffer("psi_T_ker_idx", ker_idx_T, persistent=False)
            self.register_buffer("psi_T_col_idx", col_idx_T, persistent=False)
            self.register_buffer("psi_T_vals", vals_T, persistent=False)

        # save all datastructures
        self.register_buffer("psi_ker_idx", ker_idx, persistent=False)
        self.register_buffer("psi_row_idx", row_idx, persistent=False)
        self.register_buffer("psi_col_idx", col_idx, persistent=False)
        self.register_buffer("psi_vals", vals, persistent=False)

        # also store psi just in case
        if not self.optimized_kernel:
            self.psi_st = _get_psi(self.kernel_size, self.psi_idx, self.psi_vals, self.nlat_in, self.nlon_in, self.nlat_out, self.nlon_out, semi_transposed=True)

    def extra_repr(self):
        return f"in_shape={(self.nlat_in, self.nlon_in)}, out_shape={(self.nlat_out, self.nlon_out)}, in_chans={self.groupsize * self.groups}, out_chans={self.weight.shape[0]}, filter_basis={self.filter_basis}, kernel_shape={self.kernel_shape}, theta_cutoff={self.theta_cutoff}, groups={self.groups}"

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

        if self.optimized_kernel:
            out = _disco_s2_transpose_contraction_optimized(
                x,
                # forward psi (used by autograd backward)
                self.psi_roff_idx, self.psi_ker_idx, self.psi_row_idx,
                self.psi_col_idx, self.psi_vals,
                # backward psi_T (used by the transpose conv's spatial spread)
                self.psi_T_roff_idx, self.psi_T_ker_idx,
                self.psi_T_col_idx, self.psi_T_vals,
                self.kernel_size, self.nlat_out, self.nlon_out,
            )
        else:
            out = _disco_s2_transpose_contraction_torch(x, self.psi_st.to(x.device), self.nlon_out)

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)

        return out
