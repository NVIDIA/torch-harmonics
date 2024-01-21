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
import torch.nn.functional as F

from functools import partial

from torch_harmonics.quadrature import _precompute_grid, _precompute_latitudes
if torch.cuda.is_available():
    from torch_harmonics._disco_convolution import _disco_s2_contraction_triton, _disco_s2_transpose_contraction_triton


def _compute_support_vals_isotropic(r: torch.Tensor, phi: torch.Tensor, nr: int, r_cutoff: float, norm: str = "s2"):
    """
    Computes the index set that falls into the isotropic kernel's support and returns both indices and values.
    """

    # compute the support
    dr = (r_cutoff - 0.0) / nr
    ikernel = torch.arange(nr).reshape(-1, 1, 1)
    ir = ikernel * dr

    if norm == "none":
        norm_factor = 1.0
    elif norm == "2d":
        norm_factor = math.pi * (r_cutoff * nr / (nr + 1))**2 + math.pi * r_cutoff**2 * (2 * nr / (nr + 1) + 1) / (nr + 1) / 3
    elif norm == "s2":
        norm_factor = 2 * math.pi * (1 - math.cos(r_cutoff - dr) + math.cos(r_cutoff - dr) + (math.sin(r_cutoff - dr) - math.sin(r_cutoff)) / dr)
    else:
        raise ValueError(f"Unknown normalization mode {norm}.")

    # find the indices where the rotated position falls into the support of the kernel
    iidx = torch.argwhere(((r - ir).abs() <= dr) & (r <= r_cutoff))
    vals = (1 - (r[iidx[:, 1], iidx[:, 2]] - ir[iidx[:, 0], 0, 0]).abs() / dr) / norm_factor
    return iidx, vals


def _compute_support_vals_anisotropic(r: torch.Tensor, phi: torch.Tensor, nr: int, nphi: int, r_cutoff: float, norm: str = "s2"):
    """
    Computes the index set that falls into the anisotropic kernel's support and returns both indices and values.
    """

    # compute the support
    dr = (r_cutoff - 0.0) / nr
    dphi = 2.0 * math.pi / nphi
    kernel_size = (nr - 1) * nphi + 1
    ikernel = torch.arange(kernel_size).reshape(-1, 1, 1)
    ir = ((ikernel - 1) // nphi + 1) * dr
    iphi = ((ikernel - 1) % nphi) * dphi

    if norm == "none":
        norm_factor = 1.0
    elif norm == "2d":
        norm_factor = math.pi * (r_cutoff * nr / (nr + 1))**2 + math.pi * r_cutoff**2 * (2 * nr / (nr + 1) + 1) / (nr + 1) / 3
    elif norm == "s2":
        norm_factor = 2 * math.pi * (1 - math.cos(r_cutoff - dr) + math.cos(r_cutoff - dr) + (math.sin(r_cutoff - dr) - math.sin(r_cutoff)) / dr)
    else:
        raise ValueError(f"Unknown normalization mode {norm}.")

    # find the indices where the rotated position falls into the support of the kernel
    cond_r = ((r - ir).abs() <= dr) & (r <= r_cutoff)
    cond_phi = (ikernel == 0) | ((phi - iphi).abs() <= dphi) | ((2 * math.pi - (phi - iphi).abs()) <= dphi)
    iidx = torch.argwhere(cond_r & cond_phi)
    vals = (1 - (r[iidx[:, 1], iidx[:, 2]] - ir[iidx[:, 0], 0, 0]).abs() / dr) / norm_factor
    vals *= torch.where(
        iidx[:, 0] > 0,
        (1 - torch.minimum((phi[iidx[:, 1], iidx[:, 2]] - iphi[iidx[:, 0], 0, 0]).abs(), (2 * math.pi - (phi[iidx[:, 1], iidx[:, 2]] - iphi[iidx[:, 0], 0, 0]).abs())) / dphi),
        1.0,
    )
    return iidx, vals


def _precompute_convolution_tensor_s2(in_shape, out_shape, kernel_shape, grid_in="equiangular", grid_out="equiangular", theta_cutoff=0.01 * math.pi):
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

    # array for accumulating non-zero indices
    out_idx = torch.empty([3, 0], dtype=torch.long)
    out_vals = torch.empty([0], dtype=torch.long)

    # compute the phi differences
    # It's imporatant to not include the 2 pi point in the longitudes, as it is equivalent to lon=0
    lons_in = torch.linspace(0, 2 * math.pi, nlon_in + 1)[:-1]

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
        out_idx = torch.cat([out_idx, idx], dim=-1)
        out_vals = torch.cat([out_vals, vals], dim=-1)

    return out_idx, out_vals


def _precompute_convolution_tensor_2d(grid_in, grid_out, kernel_shape, radius_cutoff=0.01, periodic=False):
    """
    Precomputes the translated filters at positions $T^{-1}_j \omega_i = T^{-1}_j T_i \nu$. Similar to the S2 routine,
    only that it assumes a non-periodic subset of the euclidean plane
    """

    # check that input arrays are valid point clouds in 2D
    assert len(grid_in) == 2
    assert len(grid_out) == 2
    assert grid_in.shape[0] == 2
    assert grid_out.shape[0] == 2

    n_in = grid_in.shape[-1]
    n_out = grid_out.shape[-1]

    if len(kernel_shape) == 1:
        kernel_handle = partial(_compute_support_vals_isotropic, nr=kernel_shape[0], r_cutoff=radius_cutoff, norm="2d")
    elif len(kernel_shape) == 2:
        kernel_handle = partial(_compute_support_vals_anisotropic, nr=kernel_shape[0], nphi=kernel_shape[1], r_cutoff=radius_cutoff, norm="2d")
    else:
        raise ValueError("kernel_shape should be either one- or two-dimensional.")

    grid_in = grid_in.reshape(2, 1, n_in)
    grid_out = grid_out.reshape(2, n_out, 1)

    diffs = grid_in - grid_out
    if periodic:
        periodic_diffs = torch.where(diffs > 0.0, diffs-1, diffs+1)
        diffs = torch.where(diffs.abs() < periodic_diffs.abs(), diffs, periodic_diffs)


    r = torch.sqrt(diffs[0] ** 2 + diffs[1] ** 2)
    phi = torch.arctan2(diffs[1], diffs[0]) + torch.pi

    idx, vals = kernel_handle(r, phi)
    idx = idx.permute(1, 0)

    return idx, vals


class DiscreteContinuousConv(nn.Module, abc.ABC):
    """
    Abstract base class for DISCO convolutions
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_shape: Union[int, List[int]],
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
    ):
        super().__init__()

        if isinstance(kernel_shape, int):
            self.kernel_shape = [kernel_shape]
        else:
            self.kernel_shape = kernel_shape

        if len(self.kernel_shape) == 1:
            self.kernel_size = self.kernel_shape[0]
        elif len(self.kernel_shape) == 2:
            self.kernel_size = (self.kernel_shape[0] - 1) * self.kernel_shape[1] + 1
        else:
            raise ValueError("kernel_shape should be either one- or two-dimensional.")

        # groups
        self.groups = groups

        # weight tensor
        if in_channels % self.groups != 0:
            raise ValueError("Error, the number of input channels has to be an integer multiple of the group size")
        if out_channels % self.groups != 0:
            raise ValueError("Error, the number of output channels has to be an integer multiple of the group size")
        self.groupsize = in_channels // self.groups
        scale = math.sqrt(1.0 / self.groupsize)
        self.weight = nn.Parameter(scale * torch.randn(out_channels, self.groupsize, self.kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    @abc.abstractmethod
    def forward(self, x: torch.Tensor):
        raise NotImplementedError


def _disco_s2_contraction_torch(x: torch.Tensor, psi: torch.Tensor, nlon_out: int):
    """
    Reference implementation of the custom contraction as described in [1]. This requires repeated
    shifting of the input tensor, which can potentially be costly. For an efficient implementation
    on GPU, make sure to use the custom kernel written in Triton.
    """
    assert len(psi.shape) == 3
    assert len(x.shape) == 4
    psi = psi.to(x.device)

    batch_size, n_chans, nlat_in, nlon_in = x.shape
    kernel_size, nlat_out, _ = psi.shape

    assert psi.shape[-1] == nlat_in * nlon_in
    assert nlon_in % nlon_out == 0
    assert nlon_in >= nlat_out
    pscale = nlon_in // nlon_out

    # add a dummy dimension for nkernel and move the batch and channel dims to the end
    x = x.reshape(1, batch_size * n_chans, nlat_in, nlon_in).permute(0, 2, 3, 1)
    x = x.expand(kernel_size, -1, -1, -1)

    y = torch.zeros(nlon_out, kernel_size, nlat_out, batch_size * n_chans, device=x.device, dtype=x.dtype)

    for pout in range(nlon_out):
        # sparse contraction with psi
        y[pout] = torch.bmm(psi, x.reshape(kernel_size, nlat_in * nlon_in, -1))
        # we need to repeatedly roll the input tensor to faciliate the shifted multiplication
        x = torch.roll(x, -pscale, dims=2)

    # reshape y back to expose the correct dimensions
    y = y.permute(3, 1, 2, 0).reshape(batch_size, n_chans, kernel_size, nlat_out, nlon_out)

    return y


def _disco_s2_transpose_contraction_torch(x: torch.Tensor, psi: torch.Tensor, nlon_out: int):
    """
    Reference implementation of the custom contraction as described in [1]. This requires repeated
    shifting of the input tensor, which can potentially be costly. For an efficient implementation
    on GPU, make sure to use the custom kernel written in Triton.
    """
    assert len(psi.shape) == 3
    assert len(x.shape) == 5
    psi = psi.to(x.device)

    batch_size, n_chans, kernel_size, nlat_in, nlon_in = x.shape
    kernel_size, _, n_out = psi.shape

    assert psi.shape[-2] == nlat_in
    assert n_out % nlon_out == 0
    nlat_out = n_out // nlon_out
    assert nlon_out >= nlat_in
    pscale = nlon_out // nlon_in

    # we do a semi-transposition to faciliate the computation
    inz = psi.indices()
    tout = inz[2] // nlon_out
    pout = inz[2] % nlon_out
    # flip the axis of longitudes
    pout = nlon_out - 1 - pout
    tin = inz[1]
    inz = torch.stack([inz[0], tout, tin*nlon_out + pout], dim=0)
    psi_mod = torch.sparse_coo_tensor(inz, psi.values(), size=(kernel_size, nlat_out, nlat_in*nlon_out))

    # interleave zeros along the longitude dimension to allow for fractional offsets to be considered
    x_ext = torch.zeros(kernel_size, nlat_in, nlon_out, batch_size * n_chans, device=x.device, dtype=x.dtype)
    x_ext[:, :, ::pscale, :] = x.reshape(batch_size * n_chans, kernel_size, nlat_in, nlon_in).permute(1, 2, 3, 0)
    # we need to go backwards through the vector, so we flip the axis
    x_ext = x_ext.contiguous()

    y = torch.zeros(kernel_size, nlon_out, nlat_out, batch_size * n_chans, device=x.device, dtype=x.dtype)

    for pout in range(nlon_out):
        # we need to repeatedly roll the input tensor to faciliate the shifted multiplication
        # TODO: double-check why this has to happen first
        x_ext = torch.roll(x_ext, -1, dims=2)
        # sparse contraction with the modified psi
        y[:, pout, :, :] = torch.bmm(psi_mod, x_ext.reshape(kernel_size, nlat_in * nlon_out, -1))

    # sum over the kernel dimension and reshape to the correct output size
    y = y.sum(dim=0).permute(2, 1, 0).reshape(batch_size, n_chans, nlat_out, nlon_out)

    return y

class DiscreteContinuousConvS2(DiscreteContinuousConv):
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
        super().__init__(in_channels, out_channels, kernel_shape, groups, bias)

        self.nlat_in, self.nlon_in = in_shape
        self.nlat_out, self.nlon_out = out_shape

        # compute theta cutoff based on the bandlimit of the input field
        if theta_cutoff is None:
            theta_cutoff = (self.kernel_shape[0] + 1) * torch.pi / float(self.nlat_in - 1)

        if theta_cutoff <= 0.0:
            raise ValueError("Error, theta_cutoff has to be positive.")

        # integration weights
        _, wgl = _precompute_latitudes(self.nlat_in, grid=grid_in)
        quad_weights = 2.0 * torch.pi * torch.from_numpy(wgl).float().reshape(-1, 1) / self.nlon_in
        self.register_buffer("quad_weights", quad_weights, persistent=False)

        idx, vals = _precompute_convolution_tensor_s2(in_shape, out_shape, self.kernel_shape, grid_in=grid_in, grid_out=grid_out, theta_cutoff=theta_cutoff)

        self.register_buffer("psi_idx", idx, persistent=False)
        self.register_buffer("psi_vals", vals, persistent=False)

    def get_psi(self):
        psi = torch.sparse_coo_tensor(self.psi_idx, self.psi_vals, size=(self.kernel_size, self.nlat_out, self.nlat_in * self.nlon_in)).coalesce()
        return psi

    def forward(self, x: torch.Tensor, use_triton_kernel: bool = True) -> torch.Tensor:
        # pre-multiply x with the quadrature weights
        x = self.quad_weights * x

        psi = self.get_psi()

        if x.is_cuda and use_triton_kernel:
            x = _disco_s2_contraction_triton(x, psi, self.nlon_out)
        else:
            x = _disco_s2_contraction_torch(x, psi, self.nlon_out)

        # extract shape
        B, C, K, H, W = x.shape
        x = x.reshape(B, self.groups, self.groupsize, K, H, W)

        # do weight multiplication
        out = torch.einsum("bgckxy,gock->bgoxy", x, self.weight.reshape(self.groups, -1, self.weight.shape[1], self.weight.shape[2]))
        out = out.reshape(out.shape[0], -1, out.shape[-2], out.shape[-1])

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)

        return out


class DiscreteContinuousConvTransposeS2(DiscreteContinuousConv):
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

        # bandlimit
        if theta_cutoff is None:
            theta_cutoff = (self.kernel_shape[0] + 1) * torch.pi / float(self.nlat_in - 1)

        if theta_cutoff <= 0.0:
            raise ValueError("Error, theta_cutoff has to be positive.")

        # integration weights
        _, wgl = _precompute_latitudes(self.nlat_in, grid=grid_in)
        quad_weights = 2.0 * torch.pi * torch.from_numpy(wgl).float().reshape(-1, 1) / self.nlon_in
        self.register_buffer("quad_weights", quad_weights, persistent=False)

        # switch in_shape and out_shape since we want transpose conv
        idx, vals = _precompute_convolution_tensor_s2(out_shape, in_shape, self.kernel_shape, grid_in=grid_out, grid_out=grid_in, theta_cutoff=theta_cutoff)

        self.register_buffer("psi_idx", idx, persistent=False)
        self.register_buffer("psi_vals", vals, persistent=False)

    def get_psi(self):
        psi = torch.sparse_coo_tensor(self.psi_idx, self.psi_vals, size=(self.kernel_size, self.nlat_in, self.nlat_out * self.nlon_out)).coalesce()
        return psi

    def forward(self, x: torch.Tensor, use_triton_kernel: bool = True) -> torch.Tensor:
        # extract shape
        B, C, H, W = x.shape
        x = x.reshape(B, self.groups, self.groupsize, H, W)

        # do weight multiplication
        x = torch.einsum("bgcxy,gock->bgokxy", x, self.weight.reshape(self.groups, -1, self.weight.shape[1], self.weight.shape[2]))
        x = x.reshape(x.shape[0], -1, x.shape[-3], x.shape[-2], x.shape[-1])

        # pre-multiply x with the quadrature weights
        x = self.quad_weights * x

        psi = self.get_psi()

        if x.is_cuda and use_triton_kernel:
            out = _disco_s2_transpose_contraction_triton(x, psi, self.nlon_out)
        else:
            out = _disco_s2_transpose_contraction_torch(x, psi, self.nlon_out)

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)

        return out


class DiscreteContinuousConv2d(DiscreteContinuousConv):
    """
    Discrete-continuous convolutions (DISCO) on arbitrary 2d grids.

    [1] Ocampo, Price, McEwen, Scalable and equivariant spherical CNNs by discrete-continuous (DISCO) convolutions, ICLR (2023), arXiv:2209.13603
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_in: torch.Tensor,
        grid_out: torch.Tensor,
        kernel_shape: Union[int, List[int]],
        n_in: Optional[Tuple[int]] = None,
        n_out: Optional[Tuple[int]] = None,
        quad_weights: Optional[torch.Tensor] = None,
        periodic: Optional[bool] = False,
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
        radius_cutoff: Optional[float] = None,
    ):
        super().__init__(in_channels, out_channels, kernel_shape, groups, bias)

        # the instantiator supports convenience constructors for the input and output grids
        if isinstance(grid_in, torch.Tensor):
            assert isinstance(quad_weights, torch.Tensor)
            assert not periodic
        elif isinstance(grid_in, str):
            assert n_in is not None
            assert len(n_in) == 2
            x, wx = _precompute_grid(n_in[0], grid=grid_in, periodic=periodic)
            y, wy = _precompute_grid(n_in[1], grid=grid_in, periodic=periodic)
            x, y = torch.meshgrid(torch.from_numpy(x), torch.from_numpy(y))
            wx, wy = torch.meshgrid(torch.from_numpy(wx), torch.from_numpy(wy))
            grid_in = torch.stack([x.reshape(-1), y.reshape(-1)])
            quad_weights = (wx * wy).reshape(-1)
        else:
            raise ValueError(f"Unknown grid input type of type {type(grid_in)}")
        
        if isinstance(grid_out, torch.Tensor):
            pass
        elif isinstance(grid_out, str):
            assert n_out is not None
            assert len(n_out) == 2
            x, wx = _precompute_grid(n_out[0], grid=grid_out, periodic=periodic)
            y, wy = _precompute_grid(n_out[1], grid=grid_out, periodic=periodic)
            x, y = torch.meshgrid(torch.from_numpy(x), torch.from_numpy(y))
            grid_out = torch.stack([x.reshape(-1), y.reshape(-1)])
        else:
            raise ValueError(f"Unknown grid output type of type {type(grid_out)}")

        # check that input arrays are valid point clouds in 2D
        assert len(grid_in.shape) == 2
        assert len(grid_out.shape) == 2
        assert len(quad_weights.shape) == 1
        assert grid_in.shape[0] == 2
        assert grid_out.shape[0] == 2

        self.n_in = grid_in.shape[-1]
        self.n_out = grid_out.shape[-1]

        # compute the cutoff radius based on the bandlimit of the input field
        # TODO: this heuristic is ad-hoc! Verify that we do the right one
        if radius_cutoff is None:
            radius_cutoff = 2 * (self.kernel_shape[0] + 1) / float(math.sqrt(self.n_in) - 1)

        if radius_cutoff <= 0.0:
            raise ValueError("Error, radius_cutoff has to be positive.")

        # integration weights
        self.register_buffer("quad_weights", quad_weights, persistent=False)

        idx, vals = _precompute_convolution_tensor_2d(grid_in, grid_out, self.kernel_shape, radius_cutoff=radius_cutoff, periodic=periodic)

        # to improve performance, we make psi a matrix by merging the first two dimensions
        # This has to be accounted for in the forward pass
        idx = torch.stack([idx[0]*self.n_out + idx[1], idx[2]], dim=0)

        self.register_buffer("psi_idx", idx.contiguous(), persistent=False)
        self.register_buffer("psi_vals", vals.contiguous(), persistent=False)

    def get_psi(self):
        psi = torch.sparse_coo_tensor(self.psi_idx, self.psi_vals, size=(self.kernel_size*self.n_out, self.n_in))
        return psi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-multiply x with the quadrature weights
        x = self.quad_weights * x

        psi = self.get_psi()

        # extract shape
        B, C, _ = x.shape

        # bring into the right shape for the bmm and perform it
        x = x.reshape(B * C, self.n_in).permute(1, 0).contiguous()
        x = torch.mm(psi, x)
        x = x.permute(1, 0).reshape(B, C, self.kernel_size, self.n_out)
        x = x.reshape(B, self.groups, self.groupsize, self.kernel_size, self.n_out)

        # do weight multiplication
        out = torch.einsum("bgckx,gock->bgox", x, self.weight.reshape(self.groups, -1, self.weight.shape[1], self.weight.shape[2]))
        out = out.reshape(out.shape[0], -1, out.shape[-1])

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1)

        return out

class DiscreteContinuousConvTranspose2d(DiscreteContinuousConv2d):
    """
    Discrete-continuous convolutions (DISCO) on arbitrary 2d grids.

    [1] Ocampo, Price, McEwen, Scalable and equivariant spherical CNNs by discrete-continuous (DISCO) convolutions, ICLR (2023), arXiv:2209.13603
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_in: torch.Tensor,
        grid_out: torch.Tensor,
        kernel_shape: Union[int, List[int]],
        n_in: Optional[Tuple[int]] = None,
        n_out: Optional[Tuple[int]] = None,
        quad_weights: Optional[torch.Tensor] = None,
        periodic: Optional[bool] = False,
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
        radius_cutoff: Optional[float] = None,
    ):
        # the instantiator supports convenience constructors for the input and output grids
        if isinstance(grid_in, torch.Tensor):
            assert isinstance(quad_weights, torch.Tensor)
            assert not periodic
        elif isinstance(grid_in, str):
            assert n_in is not None
            assert len(n_in) == 2
            x, wx = _precompute_grid(n_in[0], grid=grid_in, periodic=periodic)
            y, wy = _precompute_grid(n_in[1], grid=grid_in, periodic=periodic)
            x, y = torch.meshgrid(torch.from_numpy(x), torch.from_numpy(y))
            wx, wy = torch.meshgrid(torch.from_numpy(wx), torch.from_numpy(wy))
            grid_in = torch.stack([x.reshape(-1), y.reshape(-1)])
            quad_weights = (wx * wy).reshape(-1)
        else:
            raise ValueError(f"Unknown grid input type of type {type(grid_in)}")
        
        if isinstance(grid_out, torch.Tensor):
            pass
        elif isinstance(grid_out, str):
            assert n_out is not None
            assert len(n_out) == 2
            x, wx = _precompute_grid(n_out[0], grid=grid_out, periodic=periodic)
            y, wy = _precompute_grid(n_out[1], grid=grid_out, periodic=periodic)
            x, y = torch.meshgrid(torch.from_numpy(x), torch.from_numpy(y))
            grid_out = torch.stack([x.reshape(-1), y.reshape(-1)])
        else:
            raise ValueError(f"Unknown grid output type of type {type(grid_out)}")

        # just interchange grid_in and grid_out in the constructor
        super().__init__(in_channels, out_channels, grid_out, grid_in, kernel_shape, n_out, n_in, quad_weights, periodic, groups, bias, radius_cutoff)

        # permute the index vector and overwrite it, effectively overwriting the Psi tensor
        l = torch.Tensor([0, 2, 1])
        psi_idx = self.psi_idx[l]
        self.register_buffer("psi_idx", psi_idx, persistent=False)


class EquidistantDiscreteContinuousConv2d(DiscreteContinuousConv):
    """
    Discrete-continuous convolutions (DISCO) on arbitrary 2d grids.

    [1] Ocampo, Price, McEwen, Scalable and equivariant spherical CNNs by discrete-continuous (DISCO) convolutions, ICLR (2023), arXiv:2209.13603
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_shape: Union[int, List[int]],
        in_shape: Tuple[int],
        out_shape: Tuple[int],
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
        radius_cutoff: Optional[float] = None,
        padding_mode: str = "circular",
        **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel_shape, groups, bias)

        self.padding_mode = padding_mode

        # compute the cutoff radius based on the assumption that the grid is [-1, 1]^2
        # this still assumes a quadratic domain
        if radius_cutoff is None:
            radius_cutoff = 2 * (self.kernel_shape[0]) / float(max(*in_shape))
        self.psi_local_size = math.floor(2*radius_cutoff * max(*in_shape) / 2) + 1

        # psi_local is essentially the support of the hat functions evaluated locally
        x = torch.linspace(-radius_cutoff, radius_cutoff, self.psi_local_size)
        x, y = torch.meshgrid(x, x)
        grid_in = torch.stack([x.reshape(-1), y.reshape(-1)])
        grid_out = torch.Tensor([[0.0], [0.0]])

        idx, vals = _precompute_convolution_tensor_2d(grid_in, grid_out, self.kernel_shape, radius_cutoff=radius_cutoff, periodic=False)

        psi_loc = torch.zeros(self.kernel_size, self.psi_local_size*self.psi_local_size)
        for ie in range(len(vals)):
            f = idx[0, ie]; j = idx[2, ie]; v = vals[ie]
            psi_loc[f, j] = v

        # compute local version of the filter matrix
        psi_loc = psi_loc.reshape(self.kernel_size, self.psi_local_size, self.psi_local_size)
        # normalization by the quadrature weights
        psi_loc = 4.0 * psi_loc / float(in_shape[0]*in_shape[1])

        self.register_buffer("psi_loc", psi_loc, persistent=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        kernel = torch.einsum("kxy,ogk->ogxy", self.psi_loc, self.weight)

        left_pad = self.psi_local_size // 2
        right_pad = (self.psi_local_size+1) // 2 - 1
        x = F.pad(x, (left_pad, right_pad, left_pad, right_pad), mode=self.padding_mode)
        out = F.conv2d(x, kernel, self.bias, stride=1, dilation=1, padding=0, groups=self.groups)

        return out