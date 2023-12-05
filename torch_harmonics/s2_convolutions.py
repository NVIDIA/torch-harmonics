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

from functools import partial

from torch_harmonics.quadrature import _precompute_latitudes
from torch_harmonics.disco_convolutions import (
    _disco_s2_contraction_torch,
    _disco_s2_transpose_contraction_torch,
    _disco_s2_contraction_triton,
    _disco_s2_transpose_contraction_triton,
)


def _compute_support_vals_isotropic(theta: torch.Tensor, phi: torch.Tensor, kernel_size: int, theta_cutoff: float):
    """
    Computes the index set that falls into the isotropic kernel's support and returns both indices and values.
    """

    # compute the support
    dtheta = (theta_cutoff - 0.0) / kernel_size
    ikernel = torch.arange(kernel_size).reshape(-1, 1, 1)
    itheta = ikernel * dtheta

    norm_factor = 2 * math.pi * (1 - math.cos(theta_cutoff))

    # find the indices where the rotated position falls into the support of the kernel
    iidx = torch.argwhere(((theta - itheta).abs() <= dtheta) & (theta <= theta_cutoff))
    vals = (1 - (theta[iidx[:, 1], iidx[:, 2]] - itheta[iidx[:, 0], 0, 0]).abs() / dtheta) / norm_factor
    return iidx, vals


def _precompute_convolution_tensor(
    in_shape, out_shape, kernel_shape, grid_in="equiangular", grid_out="equiangular", theta_cutoff=0.01 * math.pi
):
    """
    Precomputes the rotated filters at positions $R^{-1}_j \omega_i = R^{-1}_j R_i \nu = Y(-\theta_j)Z(\phi_i - \phi_j)Y(\theta_j)\nu$.
    Assumes a tensorized grid on the sphere with an equidistant sampling in longitude as described in Ocampo et al.
    The output tensor has shape kernel_shape x nlat_out x (nlat_in * nlon_in)
    """

    assert len(in_shape) == 2
    assert len(out_shape) == 2

    if len(kernel_shape) == 1:
        kernel_handle = partial(_compute_support_vals_isotropic, kernel_size=kernel_shape[0], theta_cutoff=theta_cutoff)
    else:
        raise ValueError("kernel_shape should be either one- or two-dimensional.")

    nlat_in, nlon_in = in_shape
    nlat_out, nlon_out = out_shape

    lats_in, _ = _precompute_latitudes(nlat_in, grid=grid_in)
    lats_in = torch.from_numpy(lats_in).float()
    lats_out, _ = _precompute_latitudes(nlat_out, grid=grid_out)
    lats_out = torch.from_numpy(lats_out).float()

    # array for accumulating non-zero indices
    out_idx = torch.empty([3, 0], dtype=torch.long)
    out_vals = torch.empty([0], dtype=torch.long)

    # compute the phi differences
    phis = torch.linspace(0, 2 * math.pi, nlon_in)

    for t in range(nlat_out):
        alpha = -lats_in.reshape(-1, 1)
        beta = phis
        gamma = lats_out[t]

        # compute latitude of the rotated position
        z = torch.cos(alpha) * torch.cos(gamma) - torch.cos(beta) * torch.sin(alpha) * torch.sin(gamma)
        theta = torch.arccos(z)

        # compute cartesian coordinates of the rotated position
        x = torch.cos(beta) * torch.sin(alpha) + torch.cos(alpha) * torch.cos(beta) * torch.sin(gamma)
        y = torch.sin(beta) * torch.sin(gamma)
        phi = torch.arctan2(y, x)

        # find the indices where the rotated position falls into the support of the kernel
        iidx, vals = kernel_handle(theta, phi)

        # add the output latitude and reshape such that psi has dimensions kernel_shape x nlat_out x (nlat_in*nlon_in)
        idx = torch.stack([iidx[:, 0], t * torch.ones_like(iidx[:, 0]), iidx[:, 1] * nlon_in + iidx[:, 2]], dim=0)

        # append indices and values to the COO datastructure
        out_idx = torch.cat([out_idx, idx], dim=-1)
        out_vals = torch.cat([out_vals, vals], dim=-1)

    return out_idx, out_vals


# TODO:
# - parameter initialization
# - add anisotropy
class DiscreteContinuousConvS2(nn.Module):
    """
    Discrete-continuous convolutions (DISCO) on the 2-Sphere as described in [1].

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
        super().__init__()

        self.nlat_in, self.nlon_in = in_shape
        self.nlat_out, self.nlon_out = out_shape

        if isinstance(kernel_shape, int):
            kernel_shape = [kernel_shape]
        self.kernel_size = 1
        for kdim in kernel_shape:
            self.kernel_size *= kdim

        # bandlimit
        if theta_cutoff is None:
            theta_cutoff = kernel_shape[0] * torch.pi / float(self.nlat_in - 1)

        if theta_cutoff <= 0.0:
            raise ValueError("Error, theta_cutoff has to be positive.")

        # integration weights
        _, wgl = _precompute_latitudes(self.nlat_in, grid=grid_in)
        quad_weights = 2.0 * torch.pi * torch.from_numpy(wgl).float().reshape(-1, 1) / self.nlon_in
        self.register_buffer("quad_weights", quad_weights, persistent=False)

        idx, vals = _precompute_convolution_tensor(
            in_shape, out_shape, kernel_shape, grid_in=grid_in, grid_out=grid_out, theta_cutoff=theta_cutoff
        )
        psi = torch.sparse_coo_tensor(
            idx, vals, size=(self.kernel_size, self.nlat_out, self.nlat_in * self.nlon_in)
        ).coalesce()
        self.register_buffer("psi", psi, persistent=False)

        # groups
        self.groups = groups

        # weight tensor
        if in_channels % self.groups != 0:
            raise ValueError("Error, the number of input channels has to be an integer multiple of the group size")
        self.groupsize = in_channels // self.groups
        weight = nn.Parameter(torch.ones(out_channels, self.groupsize, kernel_shape[0]))
        self.register_buffer("weight", weight)

        if bias:
            btens = nn.Parameter(torch.zeros(out_channels))
            self.register_buffer("bias", btens)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor, use_triton_kernel: bool = True) -> torch.Tensor:
        # pre-multiply x with the quadrature weights
        x = self.quad_weights * x

        if x.is_cuda and use_triton_kernel:
            x = _disco_s2_contraction_triton(x, self.psi, self.nlon_out)
        else:
            x = _disco_s2_contraction_torch(x, self.psi, self.nlon_out)

        # extract shape
        B, C, K, H, W = x.shape
        x = x.reshape(B, self.groups, self.groupsize, K, H, W)

        # do weight multiplication
        out = torch.einsum("bgckxy,fck->bfxy", x, self.weight)

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)

        return out


class DiscreteContinuousConvTransposeS2(nn.Module):
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
        super().__init__()

        self.nlat_in, self.nlon_in = in_shape
        self.nlat_out, self.nlon_out = out_shape

        if isinstance(kernel_shape, int):
            kernel_shape = [kernel_shape]
        self.kernel_size = 1
        for kdim in kernel_shape:
            self.kernel_size *= kdim

        # bandlimit
        if theta_cutoff is None:
            theta_cutoff = kernel_shape[0] * torch.pi / float(self.nlat_in - 1)

        if theta_cutoff <= 0.0:
            raise ValueError("Error, theta_cutoff has to be positive.")

        # integration weights
        _, wgl = _precompute_latitudes(self.nlat_in, grid=grid_in)
        quad_weights = 2.0 * torch.pi * torch.from_numpy(wgl).float().reshape(-1, 1) / self.nlon_in
        self.register_buffer("quad_weights", quad_weights, persistent=False)

        # switch in_shape and out_shape since we want transpose conv
        idx, vals = _precompute_convolution_tensor(
            out_shape, in_shape, kernel_shape, grid_in=grid_out, grid_out=grid_in, theta_cutoff=theta_cutoff
        )
        psi = torch.sparse_coo_tensor(
            idx, vals, size=(self.kernel_size, self.nlat_in, self.nlat_out * self.nlon_out)
        ).coalesce()
        self.register_buffer("psi", psi, persistent=False)

        # groups
        self.groups = groups

        # weight tensor
        if in_channels % self.groups != 0:
            raise ValueError("Error, the number of input channels has to be an integer multiple of the group size")
        self.groupsize = in_channels // self.groups
        weight = nn.Parameter(torch.ones(out_channels, self.groupsize, kernel_shape[0]))
        self.register_buffer("weight", weight)

        if bias:
            btens = nn.Parameter(torch.zeros(out_channels))
            self.register_buffer("bias", btens)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor, use_triton_kernel: bool = True) -> torch.Tensor:
        # extract shape
        B, F, H, W = x.shape
        x = x.reshape(B, self.groups, self.groupsize, H, W)

        # do weight multiplication
        x = torch.einsum("bgfxy,cfk->bckxy", x, self.weight)

        # pre-multiply x with the quadrature weights
        x = self.quad_weights * x

        if x.is_cuda and use_triton_kernel:
            out = _disco_s2_transpose_contraction_triton(x, self.psi, self.nlon_out)
        else:
            out = _disco_s2_transpose_contraction_torch(x, self.psi, self.nlon_out)

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)

        return out
