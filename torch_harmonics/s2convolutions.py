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

from torch_harmonics.quadrature import _precompute_latitudes


def _precompute_convolution_tensor(
    in_shape,
    out_shape,
    kernel_shape,
    in_grid="equiangular",
    out_grid="equiangular",
    theta_cutoff=0.01*math.pi
):
    """
    Precomputes the rotated filters at positions $R^{-1}_j \omega_i = R^{-1}_j R_i \nu = Y(-\theta_j)Z(\phi_i - \phi_j)Y(\theta_j)\nu$.
    Assumes a tensorized grid on the sphere with an equidistant sampling in longitude as described in Ocampo et al.
    The output tensor has shape kernel_shape x nlat_out x (nlat_in * nlon_in)
    """

    assert len(in_shape) == 2
    assert len(out_shape) == 2
    assert len(kernel_shape) == 1

    nlat_in, nlon_in = in_shape
    nlat_out, nlon_out = out_shape

    lats_in = torch.from_numpy(_precompute_latitudes(nlat_in, grid=in_grid)).float()
    lats_out = torch.from_numpy(_precompute_latitudes(nlat_out, grid=out_grid)).float()

    # array for accumulating non-zero indices
    out_idx = torch.empty([3, 0], dtype=torch.long)
    out_vals = torch.empty([0], dtype=torch.long)

    # compute the phi differences
    phis = torch.linspace(-math.pi, math.pi, nlon_in)

    for t in range(nlat_out):
        alpha = -lats_in.reshape(-1,1)
        beta = phis
        gamma = lats_out[t]

        # compute latitude of the rotated position
        z = torch.cos(alpha)*torch.cos(gamma) - torch.cos(beta)*torch.sin(alpha)*torch.sin(gamma)
        theta = torch.arccos(z)

        # compute cartesian coordinates of the rotated position
        x = torch.cos(beta)*torch.sin(alpha) + torch.cos(alpha)*torch.cos(beta)*torch.sin(gamma)
        y = torch.sin(beta)*torch.sin(gamma)
        phi = torch.arctan2(y, x)

        # compute the support
        d_theta = (theta_cutoff - 0.0) / kernel_shape[0]
        ps = torch.arange(kernel_shape[0]).reshape(-1, 1, 1)
        theta_i = ps * d_theta

        # find the indices where the rotated position falls into the support of the kernel
        iidx = torch.argwhere(((theta - theta_i).abs() < d_theta) & (theta < theta_cutoff))

        # compute  the value of the kernel at these position
        vals = 1 - (theta[iidx[:, 1], iidx[:, 2]] - theta_i[iidx[:, 0], 0, 0]).abs() / d_theta
        # vals = torch.ones(iidx.shape[0])
        out_vals = torch.cat([out_vals, vals], dim=-1)

        # add the output latitude indices
        # TODO: reshape them such that Psi is a sparse
        idx = torch.stack([iidx[:, 0], t*torch.ones_like(iidx[:, 0]), iidx[:, 1]*nlon_in + iidx[:, 2]], dim=0)
        out_idx = torch.cat([out_idx, idx], dim=-1)

    return out_idx, out_vals


class DiscreteContinuousConvS2(nn.Module):
    """
    Discrete-continuous convolutions (DISCO) on the 2-Sphere as described in [1].

    [1] Ocampo, Price, McEwen, Scalable and equivariant spherical CNNs by discrete-continuous (DISCO) convolutions, ICLR (2023), arXiv:2209.13603
    """

    def __init__(self, in_shape, out_shape, kernel_shape, bias=True, theta_cutoff=None):

        super().__init__()

        self.nlat_in, self.nlon_in = in_shape
        self.nlat_out, self.nlon_out = out_shape

        # bandlimit
        if theta_cutoff is None:
            self.theta_cutoff = kernel_shape[0] * torch.pi / min(self.nlat_out, self.nlon_in)
        
        idx, vals = _precompute_convolution_tensor(in_shape, out_shape, kernel_shape)
        psi = torch.sparse_coo_tensor(idx, vals).coalesce()

        self.register_buffer("psi", psi)

    def forward(self, x):
        return self._contract_disco(x)

    def _contract_disco_torch(self, x):
    
        self.psi = self.psi.to(x.device)

        B, C, = x.shape[:2]
        P = self.psi.shape[0]

        scale_factor = self.nlon_in // self.nlon_out

        x = x.reshape(B*C, self.nlat_in, self.nlon_in).permute(1,2,0)

        x_out = torch.empty(self.nlon_out, P, self.nlat_out, B*C, device=x.device, dtype=x.dtype)

        for p in range(self.nlon_out):
            x = torch.roll(x, scale_factor, dims=1)
            x_out[p] = torch.bmm(psi, x.reshape(1, -1, B*C).expand(P, -1, -1))

        x_out = x_out.permute(3, 1, 2, 0).reshape(B, C, P, self.nlat_out, self.nlon_out)

        return x_out


# class SpectralConvS2(nn.Module):
#     """
#     Spectral Convolution on the sphere
#     """

#     def __init__(self, forward_transform, inverse_transform, in_channels, out_channels, bias=False):
#         pass

#     def forward(self, x):
#         pass
