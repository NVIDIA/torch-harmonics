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
import torch.nn.functional as F
import numpy as np

from torch_harmonics.quadrature import _precompute_latitudes
from torch_harmonics._neighborhood_attention import _neighborhood_attention_s2_torch, _neighborhood_attention_s2_cuda

# import custom C++/CUDA extensions
try:
    import attention_cuda_extension
    _cuda_extension_available = True
except ImportError as err:
    attention_cuda_extension = None
    _cuda_extension_available = False

# TODO: probably this can be merged with routines in convolution.py to reduce code duplication
def _precompute_neighborhood_sparsity_s2(
    in_shape, out_shape, grid_in="equiangular", grid_out="equiangular", theta_cutoff=0.01 * math.pi, transpose_normalization=False
):
    """
    This is a modified version of the Psi (rotated filter computation) for the DISCO kernel. Instead of computing the filters we use a
    simple indicator function to check whether the points are contained within a certain radius from the origin (north-pole) at the pre-rotated positions $R^{-1}_j \omega_i = R^{-1}_j R_i \nu = Y(-\theta_j)Z(\phi_i - \phi_j)Y(\theta_j)\nu$.
    The (sparse) output tensor has shape (nlat_out * nlon_out) x (nlat_in * nlon_in).

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

    # the shaspe of the output tensor can still be adjusted depending on what is easier

    assert len(in_shape) == 2
    assert len(out_shape) == 2

    # indicator function returns indicators within the cutoff region
    indicator_handle = lambda theta : torch.argwhere(theta <= theta_cutoff)

    nlat_in, nlon_in = in_shape
    nlat_out, nlon_out = out_shape

    lats_in, win = _precompute_latitudes(nlat_in, grid=grid_in)
    lats_in = torch.from_numpy(lats_in).float()
    lats_out, wout = _precompute_latitudes(nlat_out, grid=grid_out)
    lats_out = torch.from_numpy(lats_out).float()

    # compute the phi differences
    # It's imporatant to not include the 2 pi point in the longitudes, as it is equivalent to lon=0
    lons_in = torch.linspace(0, 2 * math.pi, nlon_in + 1)[:-1]
    lons_out = torch.linspace(0, 2 * math.pi, nlon_out + 1)[:-1]

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
        # potentially kee pthis if we want oto have angular dependency for some reason
        # phi = torch.arctan2(y, x) + torch.pi

        # find the indices where the rotated position falls into the support of the kernel
        iidx = indicator_handle(theta)

        # for simplicity emulate values
        vals = torch.ones_like(iidx[:, 0])

        # add the output latitude and reshape such that psi has dimensions nlat_out x (nlat_in*nlon_in)
        idx = torch.stack([torch.zeros_like(iidx[:, 0]), t * torch.ones_like(iidx[:, 0]), iidx[:, 0] * nlon_in + iidx[:, 1]], dim=0)

        # append indices and values to the COO datastructure
        out_idx.append(idx)
        out_vals.append(vals)

    # concatenate the indices and values
    out_idx = torch.cat(out_idx, dim=-1).to(torch.int64).contiguous()
    out_vals = torch.cat(out_vals, dim=-1).to(torch.float32).contiguous()

    return out_idx, out_vals


class NeighborhoodAttentionS2(nn.Module):
    """
    Neighborhood attention on the sphere

    Parameters:
    =============

    channels: int,
        number of channels of the input signal (Self-attention)
    kdim: int,
        number of dimensions for interior inner product in the attention
    vdim: int,
        number of output channels of the attention
    """

    def __init__(
        self,
        channels: int, # embedding dim
        in_shape: Tuple[int],
        out_shape: Tuple[int],
        grid_in: Optional[str] = "equiangular",
        grid_out: Optional[str] = "equiangular",
        bias: Optional[bool] = True,
        theta_cutoff: Optional[float] = None,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
    ):
        super().__init__()

        self.nlat_in, self.nlon_in = in_shape
        self.nlat_out, self.nlon_out = out_shape

        self.channels = channels
        self.kdim = channels if kdim is None else kdim
        self.vdim = channels if vdim is None else vdim

        # heuristic to compute theta cutoff based on the bandlimit of the input field and overlaps of the basis functions
        if theta_cutoff is None:
            theta_cutoff = torch.pi / float(self.nlat_out - 1)

        if theta_cutoff <= 0.0:
            raise ValueError("Error, theta_cutoff has to be positive.")

        # integration weights
        _, wgl = _precompute_latitudes(self.nlat_in, grid=grid_in)
        quad_weights = 2.0 * torch.pi * torch.from_numpy(wgl).float() / self.nlon_in
        self.register_buffer("quad_weights", quad_weights, persistent=False)

        # precompute the neighborhood sparsity pattern
        idx, vals = _precompute_neighborhood_sparsity_s2(in_shape, out_shape, grid_in=grid_in, grid_out=grid_out, theta_cutoff=theta_cutoff)

        # this is kept for legacy resons in case we want to resuse sorting of these entries
        #ker_idx = idx[0, ...].contiguous()
        row_idx = idx[1, ...].contiguous()
        col_idx = idx[2, ...].contiguous()
        vals = vals.contiguous()

        #if _cuda_extension_available:
        #    device = "cuda"
        #    row_offset = torch.zeros(self.nlat_out+1, dtype=torch.int64).to(device)
        #    _psi_row_count = torch.zeros(self.nlat_out+1, dtype=torch.int64).to(device)
        #    self.max_psi_nnz = attention_cuda_extension.compute_row_offset(col_idx, row_idx, row_offset, _psi_row_count)
        #else:
        # compute row offsets for more structured traversal.
        # only works if rows are sorted but they are by construction
        row_offset = np.empty(self.nlat_out+1, dtype=np.int64)
        row_offset[0] = 0
        row = row_idx[0]
        for idz, z in enumerate(range(col_idx.shape[0])):
            if row_idx[z] != row:
                row_offset[row+1] = idz
                row = row_idx[z]

        # set the last value
        row_offset[row+1] = idz+1
        row_offset = torch.from_numpy(row_offset)
        self.max_psi_nnz = row_idx.max().item() + 1

        self.register_buffer("psi_row_idx", row_idx, persistent=False)
        self.register_buffer("psi_col_idx", col_idx, persistent=False)
        self.register_buffer("psi_vals", vals, persistent=False)
        self.register_buffer("psi_roff_idx", row_offset, persistent=False)

        # learnable parameters
        self.q_weights = nn.Parameter(torch.randn(self.kdim, self.channels, 1, 1))
        self.k_weights = nn.Parameter(torch.randn(self.kdim, self.channels, 1, 1))
        self.v_weights = nn.Parameter(torch.randn(self.vdim, self.channels, 1, 1))

        if bias:
            self.q_bias = nn.Parameter(torch.randn(self.kdim))
            self.k_bias = nn.Parameter(torch.randn(self.kdim))
            self.v_bias = nn.Parameter(torch.randn(self.vdim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

    @property
    def psi_idx(self):
        return torch.stack([self.psi_row_idx, self.psi_col_idx], dim=0).contiguous()

    def get_psi(self):
        psi = torch.sparse_coo_tensor(self.psi_idx, self.psi_vals,
                                      size=(self.nlat_out, self.nlat_in * self.nlon_in)).coalesce()
        return psi
    
    
    # TODO: to be implemented. Maybe we should write th
    def forward(self, qo: torch.Tensor, ki: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:

        # change this later to allow arbitrary number of batch dims
        assert (qo.dim() == ki.dim()) and (ki.dim() == vi.dim()) and (vi.dim() == 4)

        # TODO: insert dimension checks for input
        # TODO: rename h and w to lat lon
        # TODO: rename u to x
        # TODO: rename i and o to _in and _out

        # ui \in R^{..., num_channels, nlat, nlon}
        # V \in R^{vdim, num_channels}
        # compute v
        k = F.conv2d(ki, weight=self.k_weights, bias=self.k_bias)
        q = F.conv2d(qo, weight=self.q_weights, bias=self.q_bias)
        v = F.conv2d(vi, weight=self.v_weights, bias=self.v_bias)

        if x.is_cuda and _cuda_extension_available:
            out = _neighborhood_attention_s2_cuda(k, v, q, self.quad_weights,
                                                  self.psi_col_idx, self.psi_roff_idx, self.max_psi_nnz,
                                                  self.nlon_in, self.nlat_out, self.nlon_out)
        else:
            # call attention
            out = _neighborhood_attention_s2_torch(k, v, q, self.quad_weights,
                                                   self.psi_col_idx, self.psi_roff_idx,
                                                   self.nlon_in, self.nlat_out, self.nlon_out)

        return out
