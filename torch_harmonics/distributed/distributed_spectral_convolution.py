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

import math
from typing import Optional

import torch
import torch.nn as nn

from torch_harmonics._field_contract import check_field
from torch_harmonics.grid import GridS2, _rejects_legacy_signature
from torch_harmonics.radial import split_radial_pair
from torch_harmonics.sht import _require_sht_grid
from torch_harmonics.truncation import _truncate_spectral_conv
from torch_harmonics.utils import compile_if_supported

from .distributed_quadrature import DistributedQuadratureS2
from .distributed_sht import DistributedInverseRealSHT, DistributedRealSHT
from .primitives import copy_to_azimuth_region, copy_to_polar_region
from .utils import azimuth_group_rank, azimuth_group_size, polar_group_rank, polar_group_size


class DistributedSpectralConvS2(nn.Module):
    r"""
    Distributed spectral convolution layer on :math:`S^2` implemented with
    distributed real SHT (Driscoll-Healy formulation, see https://api.semanticscholar.org/CorpusID:122817218).
    Computation is split across polar and azimuth communicator groups.

    **Distribution scheme.**
    The forward and inverse SHTs are performed by
    :class:`DistributedRealSHT` and :class:`DistributedInverseRealSHT`
    respectively — see their docstrings for the all-to-all transpose
    sequence.  After the forward SHT, the spectral coefficients are split so
    that degrees ``l`` are distributed across polar ranks and orders ``m``
    across azimuth ranks.  The learnable spectral weight
    ``K[groups, c_in, c_out, l]`` is stored with its ``l`` dimension sharded
    across polar ranks (each rank holds only its local ``lmax_local`` slice).
    The spectral contraction is therefore fully local — no communication is
    needed for the channel mixing.

    **HEALPix.** Either grid may be a :class:`~torch_harmonics.healpix.HealpixGrid`, whose
    local field is this rank's band of rings of the flat ``(..., npix)`` axis (see
    :class:`~torch_harmonics.healpix.HealpixGridShardS2`). A HEALPix grid decomposes along
    latitude only, so the layer then requires an azimuth group of one rank and raises
    ``NotImplementedError`` otherwise. With one azimuth rank every grid shares the same
    spectral layout, ``l`` split across the polar ranks and all ``m`` local, so HEALPix and
    regular grids mix freely on the two sides.

    .. note::
        When saving a checkpoint to a single file, the weight tensor must be
        gathered across polar ranks along the ``l`` dimension to recover the
        full ``(groups, c_in, c_out, lmax)`` shape.  Likewise, when loading a
        serial checkpoint into the distributed module, the ``l`` dimension must
        be split according to
        :func:`~torch_harmonics.distributed.compute_split_shapes`.

    .. note::
        The spectral weight ``K[g, c_in, c_out, l]`` has no ``m`` dimension —
        it is broadcast over the spectral orders during the contraction.
        Because orders are split across azimuth ranks, each rank computes only
        a partial sum of the weight gradient (over its local ``m`` modes).
        For correct gradients the user must **all-reduce (sum) the weight
        gradients across azimuth ranks**.  This can be implemented via
        :class:`torch.nn.parallel.DistributedDataParallel` communication hooks
        or :meth:`torch.Tensor.register_post_accumulate_grad_hook`.

    .. seealso::
        :class:`torch_harmonics.SpectralConvS2`
            Serial counterpart with full mathematical description and parameter
            documentation.

    Parameters
    ----------
    grid_in : RegularGridS2 or HealpixGrid
        Descriptor of the *global* input grid. It carries the resolution as well as the
        quadrature rule, so no separate shape argument is needed.
    grid_out : RegularGridS2 or HealpixGrid
        Descriptor of the *global* output grid.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    num_groups : int, optional
        Number of channel groups for grouped spectral weights, by default 1.
    bias : bool, optional
        If ``True``, adds a learnable spectral bias computed from the spatial
        integral (replicated across process groups as needed), by default
        ``False``.
    lmax : int, optional
        Maximum spherical harmonic degree (non-inclusive) of the spectral support.
    mmax : int, optional
        Maximum azimuthal order (non-inclusive) of the spectral support.
    lmmax : int, optional
        Maximum degree-minus-order bandwidth (non-inclusive): only modes with
        ``l - m < lmmax`` are retained. ``None`` leaves it unrestricted.

    Raises
    ------
    TypeError
        If a grid is neither a regular grid nor a HEALPix grid.
    ValueError
        If ``in_channels`` or ``out_channels`` is not divisible by
        ``num_groups``.
    NotImplementedError
        If a grid is a HEALPix grid and the azimuth group has more than one rank.

    Returns
    -------
    torch.Tensor
        Local output of shape ``(..., out_channels, nlat_out_local, nlon_out_local)``, or
        ``(..., out_channels, npix_out_local)`` on a HEALPix output grid.

    Notes
    -----
    ``lmax``, ``mmax`` and ``lmmax`` resolve exactly as in
    :class:`~torch_harmonics.SpectralConvS2`, so a serial and a distributed layer
    built with the same arguments share their spectral support. The layer stores
    only this rank's local ``lmax`` slice of the spectral weights.
    """

    @_rejects_legacy_signature(
        'in_shape, out_shape, in_channels, out_channels, num_groups=1, grid_in="equiangular", grid_out="equiangular", bias=False',
        grid_in="in_shape",
        grid_out="out_shape",
    )
    def __init__(
        self,
        grid_in: GridS2,
        grid_out: GridS2,
        in_channels: int,
        out_channels: int,
        num_groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        lmax: Optional[int] = None,
        mmax: Optional[int] = None,
        lmmax: Optional[int] = None,
    ):
        super().__init__()

        if in_channels % num_groups != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by num_groups ({num_groups})")
        if out_channels % num_groups != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by num_groups ({num_groups})")

        # copy inputs
        # on a pair of product grids every shell is convolved alike, see SpectralConvS2.forward
        grid_in, grid_out, self.radial = split_radial_pair(grid_in, grid_out)
        self.grid_in = _require_sht_grid(grid_in)
        self.grid_out = _require_sht_grid(grid_out)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups

        # get the comms grid:
        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # resolve the spectral support shared by the forward and inverse SHT
        trunc = _truncate_spectral_conv(self.grid_in, self.grid_out, lmax, mmax, lmmax)
        self.lmax, self.mmax, self.lmmax = trunc.lmax, trunc.mmax, trunc.lmmax

        # set up sht layers
        self.sht = DistributedRealSHT(self.grid_in, lmax=self.lmax, mmax=self.mmax, lmmax=self.lmmax)
        self.isht = DistributedInverseRealSHT(self.grid_out, lmax=self.lmax, mmax=self.mmax, lmmax=self.lmmax)

        # extract sht parameters
        self.l_shapes = self.isht.l_shapes
        self.m_shapes = self.isht.m_shapes
        self.lmax_local = self.l_shapes[self.comm_rank_polar]
        self.mmax_local = self.m_shapes[self.comm_rank_azimuth]

        # weight shape
        weight_shape = [num_groups, in_channels // num_groups, out_channels // num_groups, self.lmax_local]

        # Compute scaling factor for correct initialization
        scale = math.sqrt(1.0 / (in_channels // num_groups)) * torch.ones(self.lmax_local, dtype=torch.complex64)
        # seemingly the first weight is not really complex, so we need to account for that.
        # That is degree 0, which only the first polar rank holds
        if self.comm_rank_polar == 0:
            scale[0] *= math.sqrt(2.0)
        self.weight = nn.Parameter(scale * torch.randn(*weight_shape, dtype=torch.complex64))

        if bias:
            self.spectral_bias = nn.Parameter(torch.zeros(1, self.in_channels, self.lmax_local, self.mmax_local, dtype=torch.complex64))
            self.quadrature = DistributedQuadratureS2(self.grid_in, normalize=False)

    def extra_repr(self):
        radial = "" if self.radial is None else f",\nradial={self.radial!r}"
        return f"grid_in={self.grid_in!r},\ngrid_out={self.grid_out!r}{radial},\nin_channels={self.in_channels}, out_channels={self.out_channels}, lmax={self.lmax}, mmax={self.mmax}, lmmax={self.lmmax}, num_groups={self.num_groups}"

    @compile_if_supported
    def _contract_lwise(self, ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
        # any axes between the channels and (l, m), i.e. the radial one, ride along
        resc = torch.einsum("bgi...xy,giox->bgo...xy", ac, bc)
        return resc

    def forward(self, x):
        check_field(x, self.sht.shard, leading=(None, self.in_channels), radial=self.radial)

        dtype = x.dtype

        # compute integral in case if bias is used
        if hasattr(self, "spectral_bias"):
            integral = self.quadrature(x)
            if self.comm_size_polar > 1:
                integral = copy_to_polar_region(integral)
            if self.comm_size_azimuth > 1:
                integral = copy_to_azimuth_region(integral)

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = x.to(torch.float32)
            # the SHT result is contiguous by construction, so no .contiguous() here: a copy
            # over a complex buffer cannot be codegen'd by inductor, as triton has no complex
            # type and the kernel signature then fails with KeyError: 'complex64'
            x = self.sht(x)

        # store the shapes; the radial axis of a product field rides along, see SpectralConvS2.forward
        B, C = x.shape[:2]
        rest = x.shape[2:]

        # deal with bias. `integral` is computed from the real-valued input above, so it is a
        # real tensor and broadcasts over the re/im axis: applying the bias on the real view is
        # exact and keeps this pointwise op off the complex dtype (see above).
        if hasattr(self, "spectral_bias"):
            bias = torch.view_as_real(self.spectral_bias)
            if self.radial is not None:
                bias = bias.unsqueeze(2)
            bias = integral.reshape(*integral.shape, 1, 1, 1) * bias
            x = torch.view_as_complex(torch.view_as_real(x) + bias)

        # perform contraction
        x = x.reshape(B, self.num_groups, C // self.num_groups, *rest)
        xp = self._contract_lwise(x, self.weight)
        # merge the group axes on the real view: the einsum output can be non-contiguous, in
        # which case this reshape copies, and that copy must not be over a complex buffer
        x = torch.view_as_complex(torch.view_as_real(xp).reshape(B, self.out_channels, *rest, 2).contiguous())

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = self.isht(x)

        # convert datatype
        x = x.to(dtype=dtype)

        return x
