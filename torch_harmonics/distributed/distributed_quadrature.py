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

from typing import Optional

import torch

from torch_harmonics.grid import GridS2, require_grid

from .primitives import reduce_from_azimuth_region, reduce_from_polar_region
from .utils import azimuth_group_rank, azimuth_group_size, polar_group_rank, polar_group_size


class DistributedQuadratureS2(torch.nn.Module):
    r"""
    Distributed scalar quadrature on :math:`S^2` for integrating spherical fields on a
    latitude/longitude grid, with data and weights split across polar and
    azimuth communicator groups.

    .. seealso::
        :class:`torch_harmonics.QuadratureS2`
            Serial counterpart with full mathematical description and parameter
            documentation.

    Parameters
    ----------
    grid : GridS2
        Descriptor of the *global* grid to integrate on. It carries both the
        resolution and the quadrature rule; the local shard is derived from it.
    normalize : bool, optional
        If ``True``, divides weights by ``4π`` to return an average instead of
        an integral, by default ``False``.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(..., channels)`` containing the global integral over
        the last two spatial dimensions (reduced across communicator groups).

    """

    def __init__(self, grid: GridS2, normalize: Optional[bool] = False):
        super().__init__()

        # copy input
        self.grid = require_grid(grid)
        self.img_shape = grid.shape
        self.nlat, self.nlon = grid.shape
        self.normalize = normalize

        # get the comms grid:
        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # the grid decomposes itself; the shard carries this rank's extent and its
        # slice of the latitudes, and knows the shapes every other rank holds
        self.shard = self.grid.shard(
            polar=(self.comm_rank_polar, self.comm_size_polar),
            azimuth=(self.comm_rank_azimuth, self.comm_size_azimuth),
        )
        self.lat_shapes = list(self.shard.lat_shapes)
        self.lon_shapes = list(self.shard.lon_shapes)

        # Build the local weights directly rather than materialising the global
        # tensor and slicing it. dlambda is the global longitude spacing, and the
        # weight of a point does not depend on its longitude, so tiling to the
        # local width is exactly this rank's slice.
        dlambda = 2 * torch.pi / self.nlon
        quad_weight = dlambda * self.shard.quad_weights.unsqueeze(1)
        quad_weight = quad_weight.tile(1, self.shard.nlon)

        # apply normalization
        if normalize:
            quad_weight = quad_weight / (4.0 * torch.pi)

        # cast to fp32
        quad_weight = quad_weight.reshape(1, 1, *self.shard.shape).to(torch.float32).contiguous()

        # register buffer
        self.register_buffer("quad_weight", quad_weight, persistent=False)

    def extra_repr(self):
        return f"grid={self.grid!r},\nnormalize={self.normalize}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # integrate over last two axes only:
        quad = torch.sum(x * self.quad_weight, dim=(-2, -1))
        if self.comm_size_polar > 1:
            quad = reduce_from_polar_region(quad)
        if self.comm_size_azimuth > 1:
            quad = reduce_from_azimuth_region(quad)

        return quad
