# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
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

from typing import Tuple, Optional

import torch
import torch.nn as nn
import math

from torch_harmonics.truncation import truncate_sht

from torch_harmonics.distributed import DistributedRealSHT, DistributedInverseRealSHT, DistributedQuadratureS2

from torch_harmonics.distributed import polar_group_size, azimuth_group_size
from torch_harmonics.distributed import polar_group_rank, azimuth_group_rank
from torch_harmonics.distributed import copy_to_polar_region, copy_to_azimuth_region


class DistributedSpectralConvS2(nn.Module):
    """
    Distributed spectral convolution layer on :math:`S^2` implemented with
    distributed real SHT (Driscoll-Healy formulation, see https://api.semanticscholar.org/CorpusID:122817218). 
    Computation is split across polar and azimuth communicator groups.

    Parameters
    -----------
    in_shape: Tuple[int]
        Spatial input grid shape ``(nlat, nlon)``.
    out_shape: Tuple[int]
        Spatial output grid shape ``(nlat, nlon)``.
    in_channels: int
        Number of input channels.
    out_channels: int
        Number of output channels.
    num_groups: int, optional
        Number of channel groups for grouped spectral weights, by default 1.
    grid_in: str, optional
        Grid used for the forward distributed SHT (``"equiangular"``,
        ``"legendre-gauss"``, ``"lobatto"``, ``"equidistant"``), by default
        ``"equiangular"``.
    grid_out: str, optional
        Grid used for the inverse distributed SHT, same options as ``grid_in``.
    bias: bool, optional
        If ``True``, adds a learnable spectral bias computed from the spatial
        integral (replicated across process groups as needed), by default
        ``False``.

    Raises
    ------
    AssertionError
        If ``in_channels`` or ``out_channels`` is not divisible by
        ``num_groups``.

    Returns
    -------
    x: torch.Tensor
        Tensor of shape ``(..., out_channels, out_shape[0], out_shape[1])``.

    Notes
    -----
    The layer truncates ``lmax``/``mmax`` to the distributed SHT limits, and
    uses local ``lmax``/``mmax`` slices when constructing spectral weights. The
    grouped contraction is performed with ``_contract_lwise``.
    """

    def __init__(self, 
        in_shape: Tuple[int],
        out_shape: Tuple[int],
        in_channels: int,
        out_channels: int,
        num_groups: Optional[int]=1,
        grid_in: Optional[str]="equiangular",
        grid_out: Optional[str]="equiangular",
        bias: Optional[bool]=False,
    ):
        super().__init__()

        assert in_channels % num_groups == 0
        assert out_channels % num_groups == 0

        # copy inputs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups

        # get the comms grid:
        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # compute truncation
        lmax_in, mmax_in = truncate_sht(in_shape[0], in_shape[1], grid=grid_in)
        lmax_out, mmax_out = truncate_sht(out_shape[0], out_shape[1], grid=grid_out)

        # compute lmax and lmin
        lmax = min(lmax_in, lmax_out)
        mmax = min(mmax_in, mmax_out)
        self.lmax = min(lmax, mmax)
        self.mmax = self.lmax

        # set up sht layers
        self.sht = DistributedRealSHT(*in_shape, grid=grid_in, lmax=self.lmax, mmax=self.mmax)
        self.isht = DistributedInverseRealSHT(*out_shape, grid=grid_out, lmax=self.lmax, mmax=self.mmax)

        # extract sht parameters
        self.l_shapes = self.isht.l_shapes
        self.m_shapes = self.isht.m_shapes
        self.lmax_local = self.l_shapes[self.comm_rank_polar]
        self.mmax_local = self.m_shapes[self.comm_rank_azimuth]

        # weight shape 
        weight_shape = [num_groups, in_channels // num_groups, out_channels // num_groups, self.lmax_local]

        # Compute scaling factor for correct initialization
        scale = math.sqrt(1.0 / (in_channels // num_groups)) * torch.ones(self.lmax_local, dtype=torch.complex64)
        # seemingly the first weight is not really complex, so we need to account for that
        scale[0] *= math.sqrt(2.0)
        self.weight = nn.Parameter(scale * torch.randn(*weight_shape, dtype=torch.complex64))

        if bias == True:
            self.spectral_bias = nn.Parameter(
                torch.zeros(1, self.out_channels, self.lmax_local, self.mmax_local, dtype=torch.complex64)
            )
            self.quadrature = DistributedQuadratureS2(
                img_shape=in_shape, 
                grid=grid_in, 
                normalize=False
            )

    @torch.compile
    def _contract_lwise(self, ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
        resc = torch.einsum("bgixy,giox->bgoxy", ac, bc)
        return resc

    def forward(self, x):
        dtype = x.dtype
        x = x.float()

        # compute integral in case if bias is used
        if hasattr(self, "spectral_bias"):
            integral = self.quadrature(x)
            if self.comm_size_polar > 1:
                integral = copy_to_polar_region(integral)
            if self.comm_size_azimuth > 1:
                integral = copy_to_azimuth_region(integral)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            x = self.sht(x).contiguous()

        # store the shapes
        B, C, H, W = x.shape

        # deal with bias
        if hasattr(self, "spectral_bias"):
            x = x + integral.reshape(B, C, 1, 1) * self.spectral_bias

        # perform contraction
        x = x.reshape(B, self.num_groups, C // self.num_groups, H, W)
        xp = self._contract_lwise(x, self.weight)
        x = xp.reshape(B, self.out_channels, H, W).contiguous()

        with torch.amp.autocast(device_type="cuda", enabled=False):
            x = self.isht(x)

        # convert datatype
        x = x.to(dtype=dtype)

        return x