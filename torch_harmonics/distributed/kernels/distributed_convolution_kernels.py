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

"""
Distributed DISCO conv kernel orchestration (all-to-all algorithm).

Two variants, selected by the public ``fused=`` flag (mirrors the serial
``DiscreteContinuousConvS2``):

  ``fused=False`` : the standard a2a path. The azimuth split is bulk-gathered
             via an all-to-all (channel<->azimuth swap), the sparse psi
             contraction runs against the full nlon_in row, polar
             reduce_scatter completes the H sum, a second a2a swaps back to
             channel-distributed, then a local einsum with the replicated
             weight produces ``(B, O, H_out_local, W_out_local)``. The
             K-expanded intermediate is saved for backward.

  ``fused=True``  : the reordered path. The weight einsum is done FIRST, on
             the local azimuth channel shard, via the custom-autograd fused
             conv op (K-expanded recomputed in backward, not saved); the
             collectives then move only the K-less ``(B, O, H, W)``. Trades
             ~one extra contraction in backward for K× lower activation
             memory and K× less collective volume. Grouped convs are handled
             by padding the local channel shard to whole-group boundaries.

Both entry points take an already-distributed input
``(B, C, H_in_local, W_in_local)``, perform the polar reduce_scatter
internally, and return a polar-reduced ``(B, O, H_out_local, W_out_local)``
tensor. Bias and the weight-gradient reduction (all_reduce over the spatial
groups) are the caller's responsibility — identical for both variants.

``optimized_kernels_is_available()`` (from disco_helpers) gates whether the
optimized CUDA kernels (required by ``fused=True``) are present.
"""

from itertools import accumulate
from typing import List, Optional

import torch
from disco_helpers import optimized_kernels_is_available

from torch_harmonics.disco.kernels_torch.disco_torch import _disco_s2_contraction_torch
from torch_harmonics.disco.optimized.disco_optimized import _disco_s2_contraction_optimized

# The fused and kpacked conv ops are defined inside
# disco_optimized.py's ``if optimized_kernels_is_available():`` block, so they
# exist iff that helper returns True. The fused a2a forward requires them.
if optimized_kernels_is_available():
    from torch_harmonics.disco.optimized.disco_optimized import (
        _disco_s2_contraction_kpacked,
        _disco_s2_fused_conv_kpacked,
        _disco_s2_fused_conv_optimized,
    )
else:
    _disco_s2_contraction_kpacked = None
    _disco_s2_fused_conv_kpacked = None
    _disco_s2_fused_conv_optimized = None

from torch_harmonics.distributed.primitives import (
    compute_split_shapes,
    distributed_transpose_azimuth,
    reduce_from_scatter_to_azimuth_region,
    reduce_from_scatter_to_polar_region,
)

# ---------------------------------------------------------------------------
# A2A entry point
# ---------------------------------------------------------------------------


def _distributed_disco_fwd_a2a(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    psi_roff_idx: Optional[torch.Tensor],
    psi_ker_idx: torch.Tensor,
    psi_row_idx: torch.Tensor,
    psi_col_idx: torch.Tensor,
    psi_vals: torch.Tensor,
    psi_kpacked_idx: Optional[torch.Tensor] = None,
    psi_kpacked_vals: Optional[torch.Tensor] = None,
    psi_kpacked_count: Optional[torch.Tensor] = None,
    psi_kpacked_K_pad: Optional[int] = None,
    kpacked_device_supported: bool = False,
    psi_torch: Optional[torch.Tensor],
    optimized_kernel: bool,
    kernel_size: int,
    nlat_out_local: int,
    nlon_out: int,
    groups: int,
    groupsize: int,
    comm_size_azimuth: int,
    lon_in_shapes: List[int],
) -> torch.Tensor:
    """A2A-based distributed DISCO forward.

    Pattern:
      1. (optional) channel <-> azimuth A2A so W is local.
      2. Sparse psi contraction → K-expanded (B, C, K, H_out, W).
      3. Polar reduce_scatter on H (split H_out across polar group).
      4. (optional) azimuth <-> channel A2A back so W is split, C is local.
      5. Local einsum (C, K) × (O, C, K) → (B, O, H_out_local, W_out_local).

    Returns the polar-reduced output WITHOUT bias.
    """
    num_chans = x.shape[1]

    # h and w split; make w local by transposing into channel dim.
    if comm_size_azimuth > 1:
        x = distributed_transpose_azimuth(x, (1, -1), lon_in_shapes)

    _kpacked_ok = optimized_kernel and psi_kpacked_K_pad in (8, 16) and x.dtype in (torch.float16, torch.bfloat16) and x.is_cuda and kpacked_device_supported
    if _kpacked_ok:
        x = _disco_s2_contraction_kpacked(
            x,
            psi_kpacked_idx,
            psi_kpacked_vals,
            psi_kpacked_count,
            psi_roff_idx,
            psi_ker_idx,
            psi_row_idx,
            psi_col_idx,
            psi_vals,
            kernel_size,
            nlat_out_local,
            nlon_out,
        )
    elif optimized_kernel:
        x = _disco_s2_contraction_optimized(
            x,
            psi_roff_idx,
            psi_ker_idx,
            psi_row_idx,
            psi_col_idx,
            psi_vals,
            kernel_size,
            nlat_out_local,
            nlon_out,
        )
    else:
        x = _disco_s2_contraction_torch(x, psi_torch.to(x.device), nlon_out)

    # Fused reduce_scatter on the polar group — half the comm of
    # reduce_from_polar_region + scatter_to_polar_region; pads short
    # chunks for uneven splits along nlat_out across the polar group.
    x = reduce_from_scatter_to_polar_region(x, -2)

    # Transpose back: lon split, channels local.
    if comm_size_azimuth > 1:
        chan_shapes = compute_split_shapes(num_chans, comm_size_azimuth)
        x = distributed_transpose_azimuth(x, (-1, 1), chan_shapes)

    B, C, K, H, W = x.shape
    x = x.reshape(B, groups, groupsize, K, H, W)
    out_channels = weight.shape[0]
    out_per_group = out_channels // groups

    out = torch.einsum(
        "bgckxy,gock->bgoxy",
        x,
        weight.reshape(groups, out_per_group, weight.shape[1], weight.shape[2]),
    ).contiguous()
    out = out.reshape(out.shape[0], out_channels, H, W)
    return out


# ---------------------------------------------------------------------------
# Reordered + fused A2A entry point (fused=True path)
# ---------------------------------------------------------------------------


def _distributed_disco_fwd_a2a_reordered(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    psi_roff_idx: torch.Tensor,
    psi_ker_idx: torch.Tensor,
    psi_row_idx: torch.Tensor,
    psi_col_idx: torch.Tensor,
    psi_vals: torch.Tensor,
    psi_kpacked_idx: Optional[torch.Tensor] = None,
    psi_kpacked_vals: Optional[torch.Tensor] = None,
    psi_kpacked_count: Optional[torch.Tensor] = None,
    psi_kpacked_K_pad: Optional[int] = None,
    kpacked_device_supported: bool = False,
    kernel_size: int,
    nlat_out_local: int,
    nlon_out: int,
    groups: int,
    groupsize: int,
    comm_size_azimuth: int,
    comm_rank_azimuth: int,
    lon_in_shapes: List[int],
) -> torch.Tensor:
    """Reordered + fused A2A DISCO forward (the ``fused=True`` path).

    The weight einsum (a linear contraction over C, K) commutes with the linear
    collectives, so it is done FIRST — on the local azimuth channel shard, via
    the custom-autograd fused conv op (which recomputes the K-expanded in
    backward instead of saving it). The collectives then move the K-less
    ``(B, O, H, W)`` instead of the K-expanded tensor:

        transpose(W->C) -> fused contraction+einsum(local C-shard) ->
            reduce_scatter(polar, H) -> reduce_scatter(azimuth, W)

    vs the non-reordered a2a (``fused=False``), which keeps the einsum last and
    saves/communicates the K-expanded. This trades ~one extra contraction in
    backward for K x lower activation memory and K x less collective volume.

    Grouped convs are handled by padding the local channel shard out to whole
    group boundaries (zero-fill); a group split across azimuth ranks is summed
    back by the azimuth reduce-scatter (each rank contributes its real channels
    plus zeros). For ``groups == 1`` the rank's channels are treated as one
    group of size ``C_local`` (no padding) and the azimuth reduce-scatter sums
    the per-rank partial channel sums.

    The weight is replicated; the forward slices the rows for this rank's local
    groups. The weight-gradient reduction (all_reduce over polar + azimuth) is
    the caller's responsibility — identical to the non-reordered a2a path,
    because summing each rank's disjoint/overlapping group contributions
    reconstructs the full gradient.

    Returns the polar-reduced ``(B, O, H_out_local, W_out_local)`` WITHOUT bias.
    """
    if _disco_s2_fused_conv_optimized is None:
        raise NotImplementedError("fused=True requires the optimized DISCO CUDA kernels " "(_disco_s2_fused_conv_optimized); rebuild the optimized library " "or use fused=False.")

    out_channels, _, K = weight.shape  # weight: (out_channels, groupsize, K)
    out_per_group = out_channels // groups
    in_channels = groups * groupsize

    # 1. azimuth transpose W->C: full W, even channel shard.
    x = distributed_transpose_azimuth(x, (1, -1), lon_in_shapes) if comm_size_azimuth > 1 else x
    local_in_channels = x.shape[1]
    chan_start = ([0] + list(accumulate(compute_split_shapes(in_channels, comm_size_azimuth)[:-1])))[comm_rank_azimuth] if comm_size_azimuth > 1 else 0
    chan_end = chan_start + local_in_channels

    if groups == 1:
        # within-group channel split: one local group of size local_in_channels, no padding.
        x_padded = x.contiguous()
        weight_local = weight[:, chan_start:chan_end, :].reshape(1, out_channels, local_in_channels, K)
        n_local_groups, local_groupsize, out_channel_offset = 1, local_in_channels, 0
    else:
        # round the channel shard to whole-group boundaries.
        group_lo = chan_start // groupsize  # floor
        group_hi = (chan_end + groupsize - 1) // groupsize  # ceil
        n_local_groups = group_hi - group_lo
        local_groupsize = groupsize
        out_channel_offset = group_lo * out_per_group
        if n_local_groups * groupsize == local_in_channels:
            # shard is already whole groups (e.g. groupsize == 1, or a
            # group-aligned even split) — no padding/copy needed.
            x_padded = x.contiguous()
        else:
            # the even split cut a group; zero-fill the shard out to group
            # boundaries (a split group's halves are summed back by the
            # azimuth reduce-scatter below).
            x_padded = x.new_zeros(x.shape[0], n_local_groups * groupsize, x.shape[2], x.shape[3])
            pad_offset = chan_start - group_lo * groupsize
            x_padded[:, pad_offset : pad_offset + local_in_channels] = x
        # (n_local_groups, out_per_group, groupsize, K)
        weight_local = weight.reshape(groups, out_per_group, groupsize, K)[group_lo:group_hi]

    # 2+3. fused contraction + local weight einsum ->
    #      (B, n_local_groups * out_per_group, H_out_full, W_full).
    _kpacked_ok = psi_kpacked_K_pad in (8, 16) and x_padded.dtype in (torch.float16, torch.bfloat16) and x_padded.is_cuda and kpacked_device_supported
    if _kpacked_ok:
        local_out = _disco_s2_fused_conv_kpacked(
            x_padded,
            weight_local,
            psi_kpacked_idx,
            psi_kpacked_vals,
            psi_kpacked_count,
            psi_roff_idx,
            psi_ker_idx,
            psi_row_idx,
            psi_col_idx,
            psi_vals,
            kernel_size,
            nlat_out_local,
            nlon_out,
            n_local_groups,
            local_groupsize,
        )
    else:
        local_out = _disco_s2_fused_conv_optimized(
            x_padded,
            weight_local,
            psi_roff_idx,
            psi_ker_idx,
            psi_row_idx,
            psi_col_idx,
            psi_vals,
            kernel_size,
            nlat_out_local,
            nlon_out,
            n_local_groups,
            local_groupsize,
        )

    # 4. place into a full output-channel tensor (zeros for groups this rank
    #    doesn't touch; a group split across ranks is summed by the azimuth rs).
    if groups == 1:
        out = local_out  # already full out_channels (partial over C; summed by the azimuth rs)
    else:
        out = local_out.new_zeros(local_out.shape[0], out_channels, local_out.shape[-2], local_out.shape[-1])
        out[:, out_channel_offset : out_channel_offset + n_local_groups * out_per_group] = local_out

    # 5. collectives on the small K-less output.
    out = reduce_from_scatter_to_polar_region(out, -2)

    if comm_size_azimuth > 1:
        out = reduce_from_scatter_to_azimuth_region(out, -1)

    # Force standard (channels-first) contiguity. The reduce-scatters can hand
    # back a channels-last / non-default-strided tensor; without this it
    # propagates downstream and trips DDP's gradient-layout-contract warning on
    # the next layer (e.g. the 1x1 MLP conv gets a channels-last weight grad).
    # The non-reordered a2a path is already contiguous via its einsum. Use a
    # plain .contiguous() (the memory_format= kwarg is silently ignored in some
    # op paths).
    return out.contiguous()
