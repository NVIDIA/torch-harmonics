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
             the local azimuth channel shard, through the backend's fused
             node (K-expanded recomputed in backward, not saved); the
             collectives then move only the K-less ``(B, O, H, W)``. Trades
             ~one extra contraction in backward for K× lower activation
             memory and K× less collective volume. Grouped convs are handled
             by padding the local channel shard to whole-group boundaries.

Both entry points take an already-distributed input
``(B, C, H_in_local, W_in_local)``, perform the polar reduce_scatter
internally, and return a polar-reduced ``(B, O, H_out_local, W_out_local)``
tensor. Bias and the weight-gradient reduction (all_reduce over the spatial
groups) are the caller's responsibility — identical for both variants.

Both evaluate the psi contraction through the layer's DISCO backend
(:mod:`torch_harmonics.disco.backends`), so the kpacked tensor cores, the arc kernels on
CPU or CUDA, and the torch reference all serve either variant; the collectives are all
that is distributed-specific here.
"""

from itertools import accumulate

import torch

from torch_harmonics.distributed.primitives import (
    compute_split_shapes,
    distributed_transpose_azimuth,
    polar_halo_exchange,
    reduce_from_scatter_to_azimuth_region,
    reduce_from_scatter_to_polar_region,
)

# ---------------------------------------------------------------------------
# A2A entry point
# ---------------------------------------------------------------------------


def _distributed_disco_fwd_a2a(layer, x: torch.Tensor) -> torch.Tensor:
    """A2A-based distributed DISCO forward.

    Pattern:
      1. (optional) channel <-> azimuth A2A so W is local.
      2. (halo) borrow r_lat input rows from each polar neighbour.
      3. Sparse psi contraction → K-expanded (B, C, K, H_out, W).
      4. (reduce-scatter only) polar reduce_scatter on H.
      5. (optional) azimuth <-> channel A2A back so W is split, C is local.
      6. Local einsum (C, K) × (O, C, K) → (B, O, H_out_local, W_out_local).

    Two polar strategies, chosen by the layer's ``polar_mode``:

    * ``use_halo``: psi is keyed to this rank's own output rows over a halo-padded input
      band, so step 3 already produces the final rows and step 4 is skipped. The K-expanded
      intermediate is then (B, C, K, H_out/P_polar, W) -- it scales with the polar group,
      which is the whole point: with the reduce-scatter it is pinned at the full H_out no
      matter how many ranks are used, and that is what runs a large model out of memory.
    * otherwise: psi is keyed to the local *input* rows over all output rows, so step 3 is a
      partial sum that step 4 completes. Unrestricted in angular reach, hence the fallback
      when the halo would have to span more than one neighbour.

    Returns the output WITHOUT bias.
    """
    comm_size_polar = layer.comm_size_polar
    comm_size_azimuth = layer.comm_size_azimuth
    use_halo = layer.use_halo

    num_chans = x.shape[1]

    # h and w split; make w local by transposing into channel dim.
    if comm_size_azimuth > 1:
        x = distributed_transpose_azimuth(x, (1, -1), layer.lon_in_shapes)

    # Borrow the input rows this rank's output rows reach into. psi's columns were keyed to
    # the padded band at construction, so the contraction below reads it directly.
    if use_halo and comm_size_polar > 1:
        x = polar_halo_exchange(x, layer.r_lat)

    x = layer.backend.contract(layer, x)

    # Fused reduce_scatter on the polar group — half the comm of
    # reduce_from_polar_region + scatter_to_polar_region; pads short
    # chunks for uneven splits along nlat_out across the polar group.
    # Guarded like the azimuth collectives below: at size 1 the primitive is
    # an identity, and skipping it keeps the `torch.compiler.disable()`d
    # wrapper out of the graph so this path stays fullgraph-compilable.
    # On the halo path the rows are already complete and already local -- nothing to reduce.
    if comm_size_polar > 1 and not use_halo:
        x = reduce_from_scatter_to_polar_region(x, -2)

    # Transpose back: lon split, channels local.
    if comm_size_azimuth > 1:
        chan_shapes = compute_split_shapes(num_chans, comm_size_azimuth)
        x = distributed_transpose_azimuth(x, (-1, 1), chan_shapes)

    B, C, K, H, W = x.shape
    x = x.reshape(B, layer.groups, layer.groupsize, K, H, W)
    out = torch.einsum("bgckxy,gock->bgoxy", x, layer._weight_r()).contiguous()
    return out.reshape(B, layer.groups * layer.out_per_group, H, W)


# ---------------------------------------------------------------------------
# Reordered + fused A2A entry point (fused=True path)
# ---------------------------------------------------------------------------


def _distributed_disco_fwd_a2a_reordered(layer, x: torch.Tensor) -> torch.Tensor:
    """Reordered + fused A2A DISCO forward (the ``fused=True`` path).

    The weight einsum (a linear contraction over C, K) commutes with the linear
    collectives, so it is done FIRST — on the local azimuth channel shard, through
    the backend's fused node (which recomputes the K-expanded in backward instead
    of saving it, where the backend can). The collectives then move the K-less
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
    weight = layer.weight
    groups, groupsize = layer.groups, layer.groupsize
    comm_size_polar = layer.comm_size_polar
    comm_size_azimuth = layer.comm_size_azimuth
    comm_rank_azimuth = layer.comm_rank_azimuth
    use_halo = layer.use_halo

    out_channels, _, K = weight.shape  # weight: (out_channels, groupsize, K)
    out_per_group = out_channels // groups
    in_channels = groups * groupsize

    # 1. azimuth transpose W->C: full W, even channel shard.
    x = distributed_transpose_azimuth(x, (1, -1), layer.lon_in_shapes) if comm_size_azimuth > 1 else x

    # (halo) borrow the input rows this rank's output rows reach into; psi's columns were
    # keyed to the padded band at construction. See _distributed_disco_fwd_a2a.
    if use_halo and comm_size_polar > 1:
        x = polar_halo_exchange(x, layer.r_lat)
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
    local_out = layer.backend.conv(layer, x_padded, weight_local, n_local_groups, local_groupsize, recompute=True)

    # 4. place into a full output-channel tensor (zeros for groups this rank
    #    doesn't touch; a group split across ranks is summed by the azimuth rs).
    if groups == 1:
        out = local_out  # already full out_channels (partial over C; summed by the azimuth rs)
    else:
        out = local_out.new_zeros(local_out.shape[0], out_channels, local_out.shape[-2], local_out.shape[-1])
        out[:, out_channel_offset : out_channel_offset + n_local_groups * out_per_group] = local_out

    # 5. collectives on the small K-less output. On the halo path the polar rows are already
    # complete and local; the azimuth reduce-scatter still runs, since it sums the channel
    # groups a rank holds only part of.
    if comm_size_polar > 1 and not use_halo:
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
