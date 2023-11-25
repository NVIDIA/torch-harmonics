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

import math

import torch
import torch.nn as nn

from functools import partial

from torch_harmonics.quadrature import _precompute_latitudes
from torch_harmonics.disco_convolutions import _disco_s2_contraction

def _compute_support_vals_isotropic(theta : torch.Tensor, phi : torch.Tensor, kernel_size : int, theta_cutoff : float):
    """
    Computes the index set that falls into the isotropic kernel's support and returns both indices and values.
    """

    # compute the support
    d_theta = (theta_cutoff - 0.0) / kernel_size
    ikernel = torch.arange(kernel_size).reshape(-1, 1, 1)
    itheta = ikernel * d_theta

    # find the indices where the rotated position falls into the support of the kernel
    iidx = torch.argwhere(((theta - itheta).abs() < d_theta) & (theta < theta_cutoff))
    vals = 1 - (theta[iidx[:, 1], iidx[:, 2]] - itheta[iidx[:, 0], 0, 0]).abs() / d_theta

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
        raise ValueError("Kernel shape should be either one- or two-dimensional.")

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
    phis = torch.linspace(0, 2*math.pi, nlon_in)

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
# - weights
# - bias
# - add anisotropy and handle pre-computation via a lambda
class DiscreteContinuousConvS2(nn.Module):
    """
    Discrete-continuous convolutions (DISCO) on the 2-Sphere as described in [1].

    [1] Ocampo, Price, McEwen, Scalable and equivariant spherical CNNs by discrete-continuous (DISCO) convolutions, ICLR (2023), arXiv:2209.13603
    """

    def __init__(
        self,
        in_shape,
        out_shape,
        kernel_shape,
        grid_in="equiangular",
        grid_out="equiangular",
        bias=True,
        theta_cutoff=None,
    ):
        super().__init__()

        self.nlat_in, self.nlon_in = in_shape
        self.nlat_out, self.nlon_out = out_shape

        # bandlimit
        if theta_cutoff is None:
            theta_cutoff = kernel_shape[0] * torch.pi / self.nlat_in

        # integration weights
        _, wgl = _precompute_latitudes(self.nlat_in, grid=grid_in)
        quad_weights = 2.0 * torch.pi * torch.from_numpy(wgl).reshape(-1, 1) / self.nlon_in
        self.register_buffer("quad_weights", quad_weights)

        idx, vals = _precompute_convolution_tensor(
            in_shape, out_shape, kernel_shape, grid_in=grid_in, grid_out=grid_out, theta_cutoff=theta_cutoff
        )
        psi = torch.sparse_coo_tensor(idx, vals).coalesce()

        self.register_buffer("psi", psi)

    # TODO: Refactor this code for better readability
    def _contract_disco_torch(self, x: torch.Tensor):
        """
        Reference implementation of the custom contraction as described in [1]. This requires repeated shifting of the input tensor,
        which can potentially be costly
        """
        self.psi = self.psi.to(x.device)

        B = x.shape[0]
        C = x.shape[1]
        P = self.psi.shape[0]

        scale_factor = self.nlon_in // self.nlon_out

        x = x.reshape(B * C, self.nlat_in, self.nlon_in).permute(1, 2, 0)

        x_out = torch.empty(self.nlon_out, P, self.nlat_out, B * C, device=x.device, dtype=x.dtype)

        for p in range(self.nlon_out):
            x = torch.roll(x, scale_factor, dims=1)
            x_out[p] = torch.bmm(self.psi, x.reshape(1, -1, B * C).expand(P, -1, -1))

        x_out = x_out.permute(3, 1, 2, 0).reshape(B, C, P, self.nlat_out, self.nlon_out)

        return x_out

    def forward(self, x: torch.Tensor):
        # pre-multiply x with the quadrature weights
        x = self.quad_weights * x

        if x.is_cuda:
            out = _disco_s2_contraction(self.psi, x, self.nlon_out)
        else:
            out = self._contract_disco_torch(x)

        return out


# class SpectralConvS2(nn.Module):
#     """
#     Spectral Convolution on the sphere
#     """

#     def __init__(self, forward_transform, inverse_transform, in_channels, out_channels, bias=False):
#         pass

#     def forward(self, x):
#         pass
