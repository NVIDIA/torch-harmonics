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

from torch_harmonics.quadrature import legendre_gauss_weights, lobatto_weights, clenshaw_curtiss_weights

from torch_harmonics.distributed import polar_group_size, azimuth_group_size
from torch_harmonics.distributed import polar_group_rank, azimuth_group_rank
from torch_harmonics.distributed import compute_split_shapes, split_tensor_along_dim
from torch_harmonics.distributed import reduce_from_polar_region, reduce_from_azimuth_region


class DistributedQuadratureS2(torch.nn.Module):
    """
    Distributed scalar quadrature on :math:`S^2` for integrating spherical fields on a
    latitude/longitude grid, with data and weights split across polar and
    azimuth communicator groups.

    Parameters
    -----------
    img_shape: Tuple[int]
        Spatial grid shape ``(nlat, nlon)``.
    grid: str, optional
        Quadrature grid type (``"equiangular"``, ``"legendre-gauss"``,
        ``"lobatto"``, ``"equidistant"``), by default ``"equiangular"``.
    normalize: bool, optional
        If ``True``, divides weights by ``4π`` to return an average instead of
        an integral, by default ``False``.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(..., channels)`` containing the global integral over
        the last two spatial dimensions (reduced across communicator groups).

    Raises
    ------
    ValueError
        If an unknown ``grid`` type is provided.
    """
    def __init__(
        self, 
        img_shape: Tuple[int], 
        grid: Optional[str]="equiangular", 
        normalize: Optional[bool]=False
    ):
        super().__init__()

        # copy input
        self.grid = grid
        self.img_shape = img_shape
        self.normalize = normalize

        # get the comms grid:
        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        if self.grid == "legendre-gauss":
            _, weights = legendre_gauss_weights(img_shape[0], -1, 1)
            dlambda = 2 * torch.pi / img_shape[1]
            quad_weight = dlambda * weights.unsqueeze(1)
            quad_weight = quad_weight.tile(1, img_shape[1])
        elif self.grid == "lobatto":
            _, weights = lobatto_weights(img_shape[0], -1, 1)
            dlambda = 2 * torch.pi / img_shape[1]
            quad_weight = dlambda * weights.unsqueeze(1)
            quad_weight = quad_weight.tile(1, img_shape[1])
        elif self.grid == "equiangular":
            _, weights = clenshaw_curtiss_weights(img_shape[0], -1, 1)
            dlambda = 2 * torch.pi / img_shape[1]
            quad_weight = dlambda * weights.unsqueeze(1)
            quad_weight = quad_weight.tile(1, img_shape[1])
        else:
            raise (ValueError("Unknown quadrature mode"))

        # apply normalization
        if normalize:
            quad_weight = quad_weight / (4.0 * torch.pi)

        # store lat and lon shapes:
        self.lat_shapes = compute_split_shapes(img_shape[0], self.comm_size_polar)
        self.lon_shapes = compute_split_shapes(img_shape[1], self.comm_size_azimuth)

        # make it contiguous
        quad_weight = quad_weight.contiguous().reshape(1, 1, *img_shape)

        # split across latitude and longitude
        if self.comm_size_polar > 1:
            quad_weight = split_tensor_along_dim(quad_weight, dim=-2, num_chunks=self.comm_size_polar)[self.comm_rank_polar]
        if self.comm_size_azimuth > 1:
            quad_weight = split_tensor_along_dim(quad_weight, dim=-1, num_chunks=self.comm_size_azimuth)[self.comm_rank_azimuth]

        # cast to fp32
        quad_weight = quad_weight.to(torch.float32).contiguous()

        # register buffer
        self.register_buffer("quad_weight", quad_weight, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # integrate over last two axes only:
        quad = torch.sum(x * self.quad_weight, dim=(-2, -1))
        if self.comm_size_polar > 1:
            quad = reduce_from_polar_region(quad)
        if self.comm_size_azimuth > 1:
            quad = reduce_from_azimuth_region(quad)

        return quad