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
from warnings import warn

import math

import torch
import torch.nn as nn

from functools import partial

from torch_harmonics.quadrature import _precompute_grid, _precompute_latitudes
from torch_harmonics._disco_convolution import _disco_s2_contraction_torch, _disco_s2_transpose_contraction_torch
from torch_harmonics._disco_convolution import _disco_s2_contraction_cuda, _disco_s2_transpose_contraction_cuda

# import custom C++/CUDA extensions if available
try:
    from disco_helpers import preprocess_psi
    import disco_cuda_extension
    _cuda_extension_available = True
except ImportError as err:
    disco_cuda_extension = None
    _cuda_extension_available = False


def _compute_support_vals_isotropic(r: torch.Tensor, phi: torch.Tensor, nr: int, r_cutoff: float):
    """
    Computes the index set that falls into the isotropic kernel's support and returns both indices and values.
    """

    kernel_size = (nr // 2) + nr % 2
    ikernel = torch.arange(kernel_size).reshape(-1, 1, 1)
    dr = 2 * r_cutoff / (nr + 1)

    # compute the support
    if nr % 2 == 1:
        ir = ikernel * dr
    else:
        ir = (ikernel + 0.5) * dr

    # find the indices where the rotated position falls into the support of the kernel
    iidx = torch.argwhere(((r - ir).abs() <= dr) & (r <= r_cutoff))
    vals = 1 - (r[iidx[:, 1], iidx[:, 2]] - ir[iidx[:, 0], 0, 0]).abs() / dr

    return iidx, vals


def _compute_support_vals_anisotropic(r: torch.Tensor, phi: torch.Tensor, nr: int, nphi: int, r_cutoff: float):
    """
    Computes the index set that falls into the anisotropic kernel's support and returns both indices and values. Handles the special case
    when there is an uneven number of collocation points across the diameter of the kernel.
    """

    kernel_size = (nr // 2) * nphi + nr % 2
    ikernel = torch.arange(kernel_size).reshape(-1, 1, 1)
    dr = 2 * r_cutoff / (nr + 1)
    dphi = 2.0 * math.pi / nphi

    # disambiguate even and uneven cases and compute the support
    if nr % 2 == 1:
        ir = ((ikernel - 1) // nphi + 1) * dr
        iphi = ((ikernel - 1) % nphi) * dphi
    else:
        ir = (ikernel // nphi + 0.5) * dr
        iphi = (ikernel % nphi) * dphi

    # find the indices where the rotated position falls into the support of the kernel
    if nr % 2 == 1:
        # find the support
        cond_r = ((r - ir).abs() <= dr) & (r <= r_cutoff)
        cond_phi = (ikernel == 0) | ((phi - iphi).abs() <= dphi) | ((2 * math.pi - (phi - iphi).abs()) <= dphi)
        # find indices where conditions are met
        iidx = torch.argwhere(cond_r & cond_phi)
        # compute the distance to the collocation points
        dist_r = (r[iidx[:, 1], iidx[:, 2]] - ir[iidx[:, 0], 0, 0]).abs()
        dist_phi = (phi[iidx[:, 1], iidx[:, 2]] - iphi[iidx[:, 0], 0, 0]).abs()
        # compute the value of the basis functions
        vals = 1 - dist_r / dr
        vals *= torch.where(
            (iidx[:, 0] > 0),
            (1 - torch.minimum(dist_phi, (2 * math.pi - dist_phi)) / dphi),
            1.0,
        )

    else:
        # in the even case, the inner casis functions overlap into areas with a negative areas
        rn = -r
        phin = torch.where(phi + math.pi >= 2 * math.pi, phi - math.pi, phi + math.pi)
        # find the support
        cond_r = ((r - ir).abs() <= dr) & (r <= r_cutoff)
        cond_phi = ((phi - iphi).abs() <= dphi) | ((2 * math.pi - (phi - iphi).abs()) <= dphi)
        cond_rn = ((rn - ir).abs() <= dr) & (rn <= r_cutoff)
        cond_phin = ((phin - iphi).abs() <= dphi) | ((2 * math.pi - (phin - iphi).abs()) <= dphi)
        # find indices where conditions are met
        iidx = torch.argwhere((cond_r & cond_phi) | (cond_rn & cond_phin))
        dist_r = (r[iidx[:, 1], iidx[:, 2]] - ir[iidx[:, 0], 0, 0]).abs()
        dist_phi = (phi[iidx[:, 1], iidx[:, 2]] - iphi[iidx[:, 0], 0, 0]).abs()
        dist_rn = (rn[iidx[:, 1], iidx[:, 2]] - ir[iidx[:, 0], 0, 0]).abs()
        dist_phin = (phin[iidx[:, 1], iidx[:, 2]] - iphi[iidx[:, 0], 0, 0]).abs()
        # compute the value of the basis functions
        vals = cond_r[iidx[:, 0], iidx[:, 1], iidx[:, 2]] * (1 - dist_r / dr)
        vals *= cond_phi[iidx[:, 0], iidx[:, 1], iidx[:, 2]] * (1 - torch.minimum(dist_phi, (2 * math.pi - dist_phi)) / dphi)
        valsn = cond_rn[iidx[:, 0], iidx[:, 1], iidx[:, 2]] * (1 - dist_rn / dr)
        valsn *= cond_phin[iidx[:, 0], iidx[:, 1], iidx[:, 2]] * (1 - torch.minimum(dist_phin, (2 * math.pi - dist_phin)) / dphi)
        vals += valsn

    return iidx, vals


def _normalize_convolution_tensor_s2(psi_idx, psi_vals, in_shape, out_shape, kernel_shape, quad_weights, transpose_normalization=False, merge_quadrature=False, eps=1e-9):
    """
    Discretely normalizes the convolution tensor.
    """

    nlat_in, nlon_in = in_shape
    nlat_out, nlon_out = out_shape

    if len(kernel_shape) == 1:
        kernel_size = math.ceil(kernel_shape[0] / 2)
    elif len(kernel_shape) == 2:
        kernel_size = (kernel_shape[0] // 2) * kernel_shape[1] + kernel_shape[0] % 2

    # reshape the indices implicitly to be ikernel, lat_out, lat_in, lon_in
    idx = torch.stack([psi_idx[0], psi_idx[1], psi_idx[2] // nlon_in, psi_idx[2] % nlon_in], dim=0)

    if transpose_normalization:
        # pre-compute the quadrature weights
        q = quad_weights[idx[1]].reshape(-1)

        # loop through dimensions which require normalization
        for ik in range(kernel_size):
            for ilat in range(nlat_in):
                # get relevant entries
                iidx = torch.argwhere((idx[0] == ik) & (idx[2] == ilat))
                # normalize, while summing also over the input longitude dimension here as this is not available for the output
                vnorm = torch.sum(psi_vals[iidx] * q[iidx])
                if merge_quadrature:
                    # the correction factor accounts for the difference in longitudinal grid points when the input vector is upscaled
                    psi_vals[iidx] = psi_vals[iidx] * q[iidx] * nlon_in / nlon_out / (vnorm + eps)
                else:
                    psi_vals[iidx] = psi_vals[iidx] / (vnorm + eps)
    else:
        # pre-compute the quadrature weights
        q = quad_weights[idx[2]].reshape(-1)

        # loop through dimensions which require normalization
        for ik in range(kernel_size):
            for ilat in range(nlat_out):
                # get relevant entries
                iidx = torch.argwhere((idx[0] == ik) & (idx[1] == ilat))
                # normalize
                vnorm = torch.sum(psi_vals[iidx] * q[iidx])
                if merge_quadrature:
                    psi_vals[iidx] = psi_vals[iidx] * q[iidx] / (vnorm + eps)
                else:
                    psi_vals[iidx] = psi_vals[iidx] / (vnorm + eps)

    return psi_vals


def _precompute_convolution_tensor_s2(
    in_shape, out_shape, kernel_shape, grid_in="equiangular", grid_out="equiangular", theta_cutoff=0.01 * math.pi, transpose_normalization=False, merge_quadrature=False
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

    if len(kernel_shape) == 1:
        kernel_handle = partial(_compute_support_vals_isotropic, nr=kernel_shape[0], r_cutoff=theta_cutoff)
    elif len(kernel_shape) == 2:
        kernel_handle = partial(_compute_support_vals_anisotropic, nr=kernel_shape[0], nphi=kernel_shape[1], r_cutoff=theta_cutoff)
    else:
        raise ValueError("kernel_shape should be either one- or two-dimensional.")

    nlat_in, nlon_in = in_shape
    nlat_out, nlon_out = out_shape

    lats_in, win = _precompute_latitudes(nlat_in, grid=grid_in)
    lats_in = torch.from_numpy(lats_in).float()
    lats_out, wout = _precompute_latitudes(nlat_out, grid=grid_out)
    lats_out = torch.from_numpy(lats_out).float()

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
    out_idx = torch.cat(out_idx, dim=-1).to(torch.long).contiguous()
    out_vals = torch.cat(out_vals, dim=-1).to(torch.float32).contiguous()

    if transpose_normalization:
        quad_weights = 2.0 * torch.pi * torch.from_numpy(wout).float().reshape(-1, 1) / nlon_in
    else:
        quad_weights = 2.0 * torch.pi * torch.from_numpy(win).float().reshape(-1, 1) / nlon_in
    out_vals = _normalize_convolution_tensor_s2(
        out_idx, out_vals, in_shape, out_shape, kernel_shape, quad_weights, transpose_normalization=transpose_normalization, merge_quadrature=merge_quadrature
    )

    return out_idx, out_vals


class DiscreteContinuousConv(nn.Module, metaclass=abc.ABCMeta):
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
            self.kernel_size = math.ceil(self.kernel_shape[0] / 2)
            if self.kernel_shape[0] % 2 == 0:
                warn(
                    "Detected isotropic kernel with even number of collocation points in the radial direction. This feature is only supported out of consistency and may lead to unexpected behavior."
                )
        elif len(self.kernel_shape) == 2:
            self.kernel_size = (self.kernel_shape[0] // 2) * self.kernel_shape[1] + self.kernel_shape[0] % 2
        if len(self.kernel_shape) > 2:
            raise ValueError("kernel_shape should be either one- or two-dimensional.")

        # groups
        self.groups = groups

        # weight tensor
        if in_channels % self.groups != 0:
            raise ValueError("Error, the number of input channels has to be an integer multiple of the group size")
        if out_channels % self.groups != 0:
            raise ValueError("Error, the number of output channels has to be an integer multiple of the group size")
        self.groupsize = in_channels // self.groups
        scale = math.sqrt(1.0 / self.groupsize / self.kernel_size)
        self.weight = nn.Parameter(scale * torch.randn(out_channels, self.groupsize, self.kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    @abc.abstractmethod
    def forward(self, x: torch.Tensor):
        raise NotImplementedError


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

        # heuristic to compute theta cutoff based on the bandlimit of the input field and overlaps of the basis functions
        if theta_cutoff is None:
            theta_cutoff = torch.pi / float(self.nlat_out - 1)

        if theta_cutoff <= 0.0:
            raise ValueError("Error, theta_cutoff has to be positive.")

        idx, vals = _precompute_convolution_tensor_s2(
            in_shape, out_shape, self.kernel_shape, grid_in=grid_in, grid_out=grid_out, theta_cutoff=theta_cutoff, transpose_normalization=False, merge_quadrature=True
        )

        # sort the values
        ker_idx = idx[0, ...].contiguous()
        row_idx = idx[1, ...].contiguous()
        col_idx = idx[2, ...].contiguous()
        vals = vals.contiguous()

        if _cuda_extension_available:
            # preprocessed data-structure for GPU kernel
            roff_idx = preprocess_psi(self.kernel_size, out_shape[0], ker_idx, row_idx, col_idx, vals).contiguous()
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
        return f"in_shape={(self.nlat_in, self.nlon_in)}, out_shape={(self.nlat_out, self.nlon_out)}, in_chans={self.groupsize * self.groups}, out_chans={self.weight.shape[0]}, kernel_shape={self.kernel_shape}, groups={self.groups}"

    @property
    def psi_idx(self):
        return torch.stack([self.psi_ker_idx, self.psi_row_idx, self.psi_col_idx], dim=0).contiguous()

    def get_psi(self):
        psi = torch.sparse_coo_tensor(self.psi_idx, self.psi_vals, size=(self.kernel_size, self.nlat_out, self.nlat_in * self.nlon_in)).coalesce()
        return psi

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.is_cuda and _cuda_extension_available:
            x = _disco_s2_contraction_cuda(
                x, self.psi_roff_idx, self.psi_ker_idx, self.psi_row_idx, self.psi_col_idx, self.psi_vals, self.kernel_size, self.nlat_out, self.nlon_out
            )
        else:
            if x.is_cuda:
                warn("couldn't find CUDA extension, falling back to slow PyTorch implementation")
            psi = self.get_psi()
            x = _disco_s2_contraction_torch(x, psi, self.nlon_out)

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
            theta_cutoff = torch.pi / float(self.nlat_in - 1)

        if theta_cutoff <= 0.0:
            raise ValueError("Error, theta_cutoff has to be positive.")

        # switch in_shape and out_shape since we want transpose conv
        idx, vals = _precompute_convolution_tensor_s2(
            out_shape, in_shape, self.kernel_shape, grid_in=grid_out, grid_out=grid_in, theta_cutoff=theta_cutoff, transpose_normalization=True, merge_quadrature=True
        )

        # sort the values
        ker_idx = idx[0, ...].contiguous()
        row_idx = idx[1, ...].contiguous()
        col_idx = idx[2, ...].contiguous()
        vals = vals.contiguous()

        if _cuda_extension_available:
            # preprocessed data-structure for GPU kernel
            roff_idx = preprocess_psi(self.kernel_size, in_shape[0], ker_idx, row_idx, col_idx, vals).contiguous()
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
        return f"in_shape={(self.nlat_in, self.nlon_in)}, out_shape={(self.nlat_out, self.nlon_out)}, in_chans={self.groupsize * self.groups}, out_chans={self.weight.shape[0]}, kernel_shape={self.kernel_shape}, groups={self.groups}"

    @property
    def psi_idx(self):
        return torch.stack([self.psi_ker_idx, self.psi_row_idx, self.psi_col_idx], dim=0).contiguous()

    def get_psi(self, semi_transposed: bool = False):
        if semi_transposed:
            # we do a semi-transposition to faciliate the computation
            tout = self.psi_idx[2] // self.nlon_out
            pout = self.psi_idx[2] % self.nlon_out
            # flip the axis of longitudes
            pout = self.nlon_out - 1 - pout
            tin = self.psi_idx[1]
            idx = torch.stack([self.psi_idx[0], tout, tin * self.nlon_out + pout], dim=0)
            psi = torch.sparse_coo_tensor(idx, self.psi_vals, size=(self.kernel_size, self.nlat_out, self.nlat_in * self.nlon_out)).coalesce()
        else:
            psi = torch.sparse_coo_tensor(self.psi_idx, self.psi_vals, size=(self.kernel_size, self.nlat_in, self.nlat_out * self.nlon_out)).coalesce()

        return psi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # extract shape
        B, C, H, W = x.shape
        x = x.reshape(B, self.groups, self.groupsize, H, W)

        # do weight multiplication
        x = torch.einsum("bgcxy,gock->bgokxy", x, self.weight.reshape(self.groups, -1, self.weight.shape[1], self.weight.shape[2])).contiguous()
        x = x.reshape(B, -1, x.shape[-3], H, W)

        if x.is_cuda and _cuda_extension_available:
            out = _disco_s2_transpose_contraction_cuda(
                x, self.psi_roff_idx, self.psi_ker_idx, self.psi_row_idx, self.psi_col_idx, self.psi_vals, self.kernel_size, self.nlat_out, self.nlon_out
            )
        else:
            if x.is_cuda:
                warn("couldn't find CUDA extension, falling back to slow PyTorch implementation")
            psi = self.get_psi(semi_transposed=True)
            out = _disco_s2_transpose_contraction_torch(x, psi, self.nlon_out)

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)

        return out
