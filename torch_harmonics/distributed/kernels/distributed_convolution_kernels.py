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

# The fused conv op (contraction + weight einsum, with the K-expanded
# recomputed in backward instead of saved) is defined inside
# disco_optimized.py's ``if optimized_kernels_is_available():`` block, so it
# exists iff that helper returns True. The fused a2a forward requires it.
if optimized_kernels_is_available():
    from torch_harmonics.disco.optimized.disco_optimized import _disco_s2_fused_conv_optimized
else:
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

    if optimized_kernel:
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

    out = torch.einsum(
        "bgckxy,gock->bgoxy",
        x,
        weight.reshape(groups, -1, weight.shape[1], weight.shape[2]),
    ).contiguous()
    out = out.reshape(out.shape[0], -1, H, W)
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

    az = comm_size_azimuth
    ar = comm_rank_azimuth
    O, _, K = weight.shape  # weight: (O, groupsize, K)
    Og = O // groups
    C = groups * groupsize

    # 1. azimuth transpose W->C: full W, even channel shard.
    x = distributed_transpose_azimuth(x, (1, -1), lon_in_shapes) if az > 1 else x
    Cloc = x.shape[1]
    cstart = ([0] + list(accumulate(compute_split_shapes(C, az)[:-1])))[ar] if az > 1 else 0
    cend = cstart + Cloc

    if groups == 1:
        # within-group channel split: one local group of size Cloc, no padding.
        xpad = x.contiguous()
        w_loc = weight[:, cstart:cend, :].reshape(1, O, Cloc, K)  # (G=1, Og=O, Cg=Cloc, K)
        lg, gs_eff, og0 = 1, Cloc, 0
    else:
        # round the channel shard to group boundaries; zero-fill the rest.
        g_lo = cstart // groupsize
        g_hi = -(-cend // groupsize)  # ceil
        lg = g_hi - g_lo
        gs_eff = groupsize
        og0 = g_lo * Og
        xpad = x.new_zeros(x.shape[0], lg * groupsize, x.shape[2], x.shape[3])
        off = cstart - g_lo * groupsize
        xpad[:, off : off + Cloc] = x
        w_loc = weight.reshape(groups, Og, groupsize, K)[g_lo:g_hi]  # (lg, Og, gs, K)

    # 2+3. fused contraction + local weight einsum -> (B, lg*Og, H_out_full, W_full).
    op_out = _disco_s2_fused_conv_optimized(
        xpad,
        w_loc,
        psi_roff_idx,
        psi_ker_idx,
        psi_row_idx,
        psi_col_idx,
        psi_vals,
        kernel_size,
        nlat_out_local,
        nlon_out,
        lg,
        gs_eff,
    )

    # 4. place into a full-O tensor (zeros for groups this rank doesn't touch).
    if groups == 1:
        out = op_out  # already full O (partial over C; summed by the azimuth rs)
    else:
        out = op_out.new_zeros(op_out.shape[0], O, op_out.shape[-2], op_out.shape[-1])
        out[:, og0 : og0 + lg * Og] = op_out

    # 5. collectives on the small K-less output.
    out = reduce_from_scatter_to_polar_region(out, -2)
    if az > 1:
        out = reduce_from_scatter_to_azimuth_region(out, -1)
    return out
