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

from itertools import accumulate
from typing import Dict, Optional, Union

import torch
import torch.distributed as dist
from attention_helpers import optimized_kernels_is_available

from torch_harmonics.attention import attention_kernels
from torch_harmonics.attention._attention_utils import _check_dtypes_match, _check_extent, _check_ndim
from torch_harmonics.attention.attention import NeighborhoodAttentionS2
from torch_harmonics.attention.backends import AttentionBackendS2, _ring_weights
from torch_harmonics.distributed._amp_utils import _cast_to_autocast_dtype, _custom_fwd, _custom_setup_context
from torch_harmonics.grid import RegularGridS2, _rejects_legacy_signature, require_regular_grid
from torch_harmonics.quadrature import effective_theta_cutoff

from .primitives import compute_polar_halo_radius, get_group_neighbors, polar_halo_exchange
from .utils import azimuth_group, azimuth_group_rank, azimuth_group_size, polar_group_rank, polar_group_size

# ---------------------------------------------------------------------------
# autograd.Function for the ring-step attention kernel calls
# ---------------------------------------------------------------------------


@torch.compiler.disable()
def _ring_kv(kw_chunk, vw_chunk, az_group, next_nlon_kw, next_nlon_kv):
    """
    Async send current chunks, receive next chunks with known shapes.

    Chunks are physical NHWC ``[B, H, W, C]``. The ring carries them in the layout
    the kernels consume, so a received chunk is usable directly and no step pays a
    conversion; only the local chunk is converted, once, before the loop. The sent
    tensors have to be contiguous for NCCL, which is why the ring exchanges the NHWC
    tensors themselves rather than channels-first views of them.
    """
    send_to, recv_from = get_group_neighbors(az_group)
    B, H, _, C_k = kw_chunk.shape
    B, H, _, C_v = vw_chunk.shape
    recv_kw = torch.empty(B, H, next_nlon_kw, C_k, device=kw_chunk.device, dtype=kw_chunk.dtype)
    recv_vw = torch.empty(B, H, next_nlon_kv, C_v, device=vw_chunk.device, dtype=vw_chunk.dtype)
    ops = [
        dist.P2POp(dist.isend, kw_chunk, send_to, az_group),
        dist.P2POp(dist.irecv, recv_kw, recv_from, az_group),
        dist.P2POp(dist.isend, vw_chunk, send_to, az_group),
        dist.P2POp(dist.irecv, recv_vw, recv_from, az_group),
    ]
    reqs = dist.batch_isend_irecv(ops)
    return recv_kw, recv_vw, reqs


@torch.compiler.disable()
def _ring_grad(dkw_acc, dvw_acc, az_group, next_nlon):
    """Rotate the dkw/dvw accumulators one hop along the same ring as :func:`_ring_kv`.

    Each accumulator travels *with* the lon chunk it belongs to. At step ``s`` rank ``r``
    holds chunk ``(r + s) % P`` and adds its contribution to the accumulator it currently
    carries; the next hop goes to ``send_to == r - 1``, which holds that same chunk at step
    ``s + 1``. So after the ``P - 1`` in-loop hops plus one final hop, every accumulator is
    back at the rank owning its chunk, fully reduced.

    This is a ring reduce-scatter, and it replaces accumulating into a buffer spanning the
    *whole global longitude axis* and all-reducing that. The buffer was the one allocation
    here that did not shrink as azimuth ranks were added -- ``B x H_halo x nlon_global x C``
    in fp32, twice -- so per-rank cost went up with the group size while the slice actually
    kept went down. It also moves less: ``P`` chunk-sized hops against the all_reduce's
    ``2 (P - 1)`` chunk-equivalents.

    A ``None`` accumulator (that branch needs no gradient) is skipped, so the rotation costs
    nothing for branches that were pruned by the autograd contract.

    Shapes follow ``_ring_kv`` exactly: the accumulator's ``W`` is always the current
    chunk's ``nlon``, so ``next_nlon`` is the same value passed there.
    """
    send_to, recv_from = get_group_neighbors(az_group)
    ops = []
    recv_dkw = recv_dvw = None
    if dkw_acc is not None:
        B, H, _, C_k = dkw_acc.shape
        recv_dkw = torch.empty(B, H, next_nlon, C_k, device=dkw_acc.device, dtype=dkw_acc.dtype)
        ops += [
            dist.P2POp(dist.isend, dkw_acc, send_to, az_group),
            dist.P2POp(dist.irecv, recv_dkw, recv_from, az_group),
        ]
    if dvw_acc is not None:
        B, H, _, C_v = dvw_acc.shape
        recv_dvw = torch.empty(B, H, next_nlon, C_v, device=dvw_acc.device, dtype=dvw_acc.dtype)
        ops += [
            dist.P2POp(dist.isend, dvw_acc, send_to, az_group),
            dist.P2POp(dist.irecv, recv_dvw, recv_from, az_group),
        ]
    reqs = dist.batch_isend_irecv(ops) if ops else []
    return recv_dkw, recv_dvw, reqs


def _per_head(packed: torch.Tensor, num_heads: int) -> torch.Tensor:
    """``[B, H, W, nh * C]`` -> ``[B, H, W, nh, C]``: split the packed channel axis. A view."""
    return packed.unflatten(-1, (num_heads, -1))


def _stat_per_head(stat: torch.Tensor) -> torch.Tensor:
    """``[B, nh, H, W]`` softmax statistic -> ``[B, H, W, nh, 1]``, broadcastable against :func:`_per_head`."""
    return stat.permute(0, 2, 3, 1).unsqueeze(-1)


class _RingNeighborhoodAttentionFn(torch.autograd.Function):
    """Forward ring attention + backward ring, gather (self / downsample) direction.

    Channels-last with the heads packed along the channel dim, as in the serial
    kernels; the ring carries these tensors as they are, so no step converts:

    kw, vw : [B, H_halo, W_local, nh * C_k / nh * C_v]  lat-halo-padded
    qw     : [B, H_out_local, W_out_local, nh * C_k]

    State buffers follow the kernels' ABI:
      y_acc        : [B, H_out, W_out, nh * C_v]
      alpha_k/kvw  : [B, H_out, W_out, nh * C_k]
      alpha_sum/qdotk_max/integral : [B, nh, H_out, W_out]
    """

    @staticmethod
    @_custom_fwd(device_type="cuda")
    def forward(
        kw,
        vw,
        qw,
        psi_seg,
        psi_seg_off,
        ring_weights,
        num_heads: int,
        nlon_in: int,
        nlon_out_global: int,
        lon_chunk_starts: list,
        nlon_kx_list: list,
        lat_halo_start: int,
        nlat_out_local: int,
        nlon_out_local: int,
        az_group,
        az_rank: int,
        az_size: int,
    ):
        B = kw.shape[0]
        C_v = vw.shape[-1]
        device = kw.device

        # Capture input dtype so we can cast the user-visible output (y_out)
        # back at the end of forward. The internal accumulators (y_acc,
        # alpha_sum, qdotk_max) stay fp32 — softmax stability requires that,
        # and the saved alpha_sum/qdotk_max feed fp32 math in backward.
        inp_dtype = kw.dtype

        # State buffers in the kernels' ABI: y_acc packed like the activations,
        # one softmax statistic per (batch, head, point).
        y_acc = torch.zeros(B, nlat_out_local, nlon_out_local, C_v, device=device, dtype=torch.float32)
        alpha_sum = torch.zeros(B, num_heads, nlat_out_local, nlon_out_local, device=device, dtype=torch.float32)
        qdotk_max = torch.full((B, num_heads, nlat_out_local, nlon_out_local), float("-inf"), device=device, dtype=torch.float32)

        # already in the kernels' layout: the ring sends and receives these directly
        kw_chunk = kw.contiguous()
        vw_chunk = vw.contiguous()
        qw = qw.contiguous()

        for step in range(az_size):
            src_rank = (az_rank + step) % az_size
            lon_lo_kx = lon_chunk_starts[src_rank]

            # Pre-allocate receive buffers for the NEXT chunk (correct shape)
            if step < az_size - 1:
                next_src = (az_rank + step + 1) % az_size
                recv_kw, recv_vw, reqs = _ring_kv(kw_chunk, vw_chunk, az_group, nlon_kx_list[next_src], nlon_kx_list[next_src])

            attention_kernels.forward_ring_step.default(
                kw_chunk,
                vw_chunk,
                qw,
                y_acc,
                alpha_sum,
                qdotk_max,
                ring_weights,
                psi_seg,
                psi_seg_off,
                num_heads,
                nlon_in,
                nlon_out_global,
                lon_lo_kx,
                lat_halo_start,
                nlat_out_local,
                nlon_out_local,
            )

            if step < az_size - 1:
                for req in reqs:
                    req.wait()
                kw_chunk = recv_kw.clone()
                vw_chunk = recv_vw.clone()

        # Finalize: y = y_acc / alpha_sum, per head. Cast back to the input dtype
        # to keep the op faithful to its input dtype.
        y_out = (_per_head(y_acc, num_heads) / _stat_per_head(alpha_sum)).flatten(-2)  # [B, H, W, nh * C_v]
        y_out = y_out.to(dtype=inp_dtype)

        # alpha_sum and qdotk_max are returned so setup_context can save them;
        # they are marked non-differentiable there, so backward still only
        # receives one gradient argument (dy for y_out).
        return y_out, alpha_sum, qdotk_max

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        (
            kw,
            vw,
            qw,
            psi_seg,
            psi_seg_off,
            ring_weights,
            num_heads,
            nlon_in,
            nlon_out_global,
            lon_chunk_starts,
            nlon_kx_list,
            lat_halo_start,
            nlat_out_local,
            nlon_out_local,
            az_group,
            az_rank,
            az_size,
        ) = inputs
        y_out, alpha_sum, qdotk_max = output
        # alpha_sum and qdotk_max are internal accumulators, not true outputs;
        # marking them non-differentiable keeps backward's signature as (ctx, dy).
        ctx.mark_non_differentiable(alpha_sum, qdotk_max)
        ctx.save_for_backward(kw, vw, qw, psi_seg, psi_seg_off, ring_weights, alpha_sum, qdotk_max)
        ctx.num_heads = num_heads
        ctx.nlon_in = nlon_in
        ctx.nlon_out_global = nlon_out_global
        ctx.lon_chunk_starts = lon_chunk_starts
        ctx.nlon_kx_list = nlon_kx_list
        ctx.lat_halo_start = lat_halo_start
        ctx.nlat_out_local = nlat_out_local
        ctx.nlon_out_local = nlon_out_local
        ctx.az_group = az_group
        ctx.az_rank = az_rank
        ctx.az_size = az_size

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dy, _dalpha_sum, _dqdotk_max):
        # _dalpha_sum and _dqdotk_max are always None (non-differentiable outputs)
        kw, vw, qw, psi_seg, psi_seg_off, ring_weights, fwd_alpha_sum, fwd_qdotk_max = ctx.saved_tensors

        num_heads = ctx.num_heads
        nlon_in = ctx.nlon_in
        nlon_out_global = ctx.nlon_out_global
        lon_chunk_starts = ctx.lon_chunk_starts
        nlon_kx_list = ctx.nlon_kx_list
        lat_halo_start = ctx.lat_halo_start
        nlat_out_local = ctx.nlat_out_local
        nlon_out_local = ctx.nlon_out_local
        az_group = ctx.az_group
        az_rank = ctx.az_rank
        az_size = ctx.az_size

        # Autograd contract: skip per-branch work (kernel calls, ring exchanges) for any
        # of {kw, vw, qw} that doesn't need a gradient, and return None in those slots.
        # This is what lets torch.compile / AOTAutograd prune dead subgraphs (including
        # the NCCL exchanges) from the compiled backward.
        kw_needs_grad = ctx.needs_input_grad[0]
        vw_needs_grad = ctx.needs_input_grad[1]
        qw_needs_grad = ctx.needs_input_grad[2]

        # Defensive: if somehow none of (kw, vw, qw) need grad (e.g., user wired
        # requires_grad onto one of the index buffers), there's nothing to compute.
        if not (kw_needs_grad or vw_needs_grad or qw_needs_grad):
            return (None,) * 17

        B, H_halo, _, C_k = kw.shape
        C_v = vw.shape[-1]
        device = kw.device

        # Capture input dtypes so the returned grads can be cast back. The
        # backward ring kernels consume kw/vw/qw/dy in their native dtype (Tier B:
        # widen fp16/bf16 at load, fp32 compute/accumulation). Keeping them native
        # — instead of upcasting to fp32 here — also keeps the backward ring
        # exchange at 16-bit under AMP (halved K/V comm volume), matching the
        # forward ring. The fp32 accumulators (integral_buf, alpha_k/kvw_buf,
        # dkw/dvw_acc) are unaffected; the returned grads are cast back to the
        # captured input dtypes at the end.
        kw_dtype = kw.dtype
        vw_dtype = vw.dtype
        qw_dtype = qw.dtype
        dy = dy.contiguous()  # [B, H, W, nh * C_v], native dtype
        qw = qw.contiguous()

        # ----------------------------------------------------------------
        # Backward pass 1: re-accumulate {alpha_sum, qdotk_max, integral,
        #                                  alpha_k, alpha_kvw} via ring.
        # Required whenever any of (kw, vw, qw) needs grad: integral feeds
        # pass-2's integral_norm and dqy reads alpha_k/alpha_kvw. The kernel
        # writes all three buffers in one call, so pass-1 cannot be pruned
        # per-branch.
        # ----------------------------------------------------------------
        bwd_alpha_sum = torch.zeros(B, num_heads, nlat_out_local, nlon_out_local, device=device, dtype=torch.float32)
        bwd_qdotk_max = torch.full((B, num_heads, nlat_out_local, nlon_out_local), float("-inf"), device=device, dtype=torch.float32)
        integral_buf = torch.zeros_like(bwd_alpha_sum)
        alpha_k_buf = torch.zeros(B, nlat_out_local, nlon_out_local, C_k, device=device, dtype=torch.float32)
        alpha_kvw_buf = torch.zeros_like(alpha_k_buf)

        kw = kw.contiguous()
        vw = vw.contiguous()
        kw_chunk, vw_chunk = kw, vw

        for step in range(az_size):
            src_rank = (az_rank + step) % az_size
            lon_lo_kx = lon_chunk_starts[src_rank]

            if step < az_size - 1:
                next_src = (az_rank + step + 1) % az_size
                recv_kw, recv_vw, reqs = _ring_kv(kw_chunk, vw_chunk, az_group, nlon_kx_list[next_src], nlon_kx_list[next_src])

            attention_kernels.backward_ring_step_pass1.default(
                kw_chunk,
                vw_chunk,
                qw,
                dy,
                bwd_alpha_sum,
                bwd_qdotk_max,
                integral_buf,
                alpha_k_buf,
                alpha_kvw_buf,
                ring_weights,
                psi_seg,
                psi_seg_off,
                num_heads,
                nlon_in,
                nlon_out_global,
                lon_lo_kx,
                lat_halo_start,
                nlat_out_local,
                nlon_out_local,
            )

            if step < az_size - 1:
                for req in reqs:
                    req.wait()
                kw_chunk = recv_kw.clone()
                vw_chunk = recv_vw.clone()

        # ----------------------------------------------------------------
        # Finalize pass-1 outputs.
        # Use the SAVED forward alpha_sum/qdotk_max (same values, but authoritative).
        # ----------------------------------------------------------------
        alpha_sum_inv = 1.0 / fwd_alpha_sum  # [B, nh, H, W]

        # integral_norm only feeds pass-2; skip if neither kw nor vw needs grad.
        if kw_needs_grad or vw_needs_grad:
            integral_norm = integral_buf * alpha_sum_inv  # [B, nh, H, W]

        # dqy[b,h,w,c] = inv_sq*(alpha_sum*alpha_kvw - integral*alpha_k), per head
        if qw_needs_grad:
            alpha_sum_inv_sq = alpha_sum_inv**2
            dqy = _stat_per_head(alpha_sum_inv_sq) * (
                _stat_per_head(fwd_alpha_sum) * _per_head(alpha_kvw_buf, num_heads) - _stat_per_head(integral_buf) * _per_head(alpha_k_buf, num_heads)
            )
            dqy = dqy.flatten(-2).to(dtype=qw_dtype)  # [B, H, W, nh * C_k]
        else:
            dqy = None

        # ----------------------------------------------------------------
        # Backward pass 2: scatter dkw/dvw contributions.
        # Each GPU computes its contribution to every lon chunk it visits, and a ring
        # reduce-scatter carries each chunk's accumulator back to the rank owning it
        # (see _ring_grad). The accumulator is chunk-sized, so this is the whole local
        # gradient at the end -- no allreduce and no slice.
        # Skip entirely if neither kw nor vw needs grad; a branch that does not
        # need one still gets a scratch output, since the fused kernel writes
        # both in a single call.
        # ----------------------------------------------------------------
        if kw_needs_grad or vw_needs_grad:
            # pass 1 rotated kw_chunk/vw_chunk; reset to the local chunk
            kw_chunk, vw_chunk = kw, vw
            # the accumulator starts on this rank's own chunk, which is the one it holds
            # at step 0, and is re-sized by each hop to match the chunk it then carries
            my_nlon = nlon_kx_list[az_rank]
            dkw_acc = torch.zeros(B, H_halo, my_nlon, C_k, device=device, dtype=torch.float32) if kw_needs_grad else None
            dvw_acc = torch.zeros(B, H_halo, my_nlon, C_v, device=device, dtype=torch.float32) if vw_needs_grad else None

            for step in range(az_size):
                src_rank = (az_rank + step) % az_size
                lon_lo_kx = lon_chunk_starts[src_rank]
                nlon_kx = nlon_kx_list[src_rank]

                # This kernel accumulates into its gradient outputs with atomicAdd and never
                # clears them, so the accumulator goes in directly -- no per-step temp and
                # no add_ afterwards. Its width is the current chunk's by construction.
                # Both arguments are required by the fused signature, so a branch that needs
                # no gradient still gets a scratch buffer, which is then discarded.
                # NOTE: the upsample direction's kernel assigns rather than accumulates, so
                # it must keep the per-step temp; do not mirror this there.
                dkw_out = dkw_acc if kw_needs_grad else torch.zeros(B, H_halo, nlon_kx, C_k, device=device, dtype=torch.float32)
                dvw_out = dvw_acc if vw_needs_grad else torch.zeros(B, H_halo, nlon_kx, C_v, device=device, dtype=torch.float32)

                attention_kernels.backward_ring_step_pass2.default(
                    kw_chunk,
                    vw_chunk,
                    qw,
                    dy,
                    fwd_alpha_sum,
                    fwd_qdotk_max,
                    integral_norm,
                    dkw_out,
                    dvw_out,
                    ring_weights,
                    psi_seg,
                    psi_seg_off,
                    num_heads,
                    nlon_in,
                    nlon_out_global,
                    lon_lo_kx,
                    lat_halo_start,
                    nlat_out_local,
                    nlon_out_local,
                )

                if step < az_size - 1:
                    next_src = (az_rank + step + 1) % az_size
                    recv_kw, recv_vw, reqs = _ring_kv(kw_chunk, vw_chunk, az_group, nlon_kx_list[next_src], nlon_kx_list[next_src])
                    # the accumulators follow the chunks they belong to, so they rotate in
                    # the same direction and to the same per-chunk width
                    recv_dkw, recv_dvw, grad_reqs = _ring_grad(dkw_acc, dvw_acc, az_group, nlon_kx_list[next_src])
                    for req in reqs:
                        req.wait()
                    for req in grad_reqs:
                        req.wait()
                    kw_chunk = recv_kw.clone()
                    vw_chunk = recv_vw.clone()
                    # Cloned for the same reason as kw/vw, and with more at stake: the next
                    # step's kernel atomicAdds *into* the accumulator, so handing it the
                    # irecv destination directly would mutate a buffer the collective owns.
                    # The clone is chunk-sized, so it does not give back the saving above.
                    dkw_acc = recv_dkw.clone() if recv_dkw is not None else None
                    dvw_acc = recv_dvw.clone() if recv_dvw is not None else None

            # One final hop returns each accumulator to the rank owning its chunk: after the
            # loop a rank carries the accumulator for chunk az_rank - 1, which is send_to.
            if az_size > 1:
                recv_dkw, recv_dvw, grad_reqs = _ring_grad(dkw_acc, dvw_acc, az_group, my_nlon)
                for req in grad_reqs:
                    req.wait()
                # Cloned for the same reason as in the loop: .to() below is a no-op for
                # fp32, so without it the returned gradient would alias the irecv
                # destination.
                dkw_acc = recv_dkw.clone() if recv_dkw is not None else None
                dvw_acc = recv_dvw.clone() if recv_dvw is not None else None

            # The accumulator IS the local chunk now, already in the layout of kw/vw.
            # No halo stripping: dkw/dvw must match kw/vw shape (= key_halo/value_halo);
            # the halo exchange's backward returns the halo rows to their owners.
            dkw = dkw_acc.to(dtype=kw_dtype) if kw_needs_grad else None  # [B, H_halo, W_local, nh * C_k]
            dvw = dvw_acc.to(dtype=vw_dtype) if vw_needs_grad else None  # [B, H_halo, W_local, nh * C_v]
        else:
            dkw = None
            dvw = None

        # Return grads for (kw, vw, qw, psi_seg, psi_seg_off, ring_weights,
        #                   num_heads, nlon_in, nlon_out_global, lon_chunk_starts, nlon_kx_list,
        #                   lat_halo_start, nlat_out_local, nlon_out_local,
        #                   az_group, az_rank, az_size)
        return (dkw, dvw, dqy) + (None,) * 14


class _RingNeighborhoodAttentionUpsampleFn(torch.autograd.Function):
    """Forward ring attention + backward ring for the UPSAMPLE (scatter) direction.

    K/V live on the coarse input grid (sharded, halo-padded in lat, rotating
    around the azimuth ring); Q and the output live on the fine output grid and
    stay local. Layouts are those of :class:`_RingNeighborhoodAttentionFn`:

    kw, vw : [B, H_halo, W_in_local, nh * C_k / nh * C_v]  lat-halo-padded
    qw     : [B, H_out_local, W_out_local, nh * C_k]

    State buffers:
      y_acc        : [B, H_out, W_out, nh * C_v]
      alpha_k/kvw  : [B, H_out, W_out, nh * C_k]
      alpha_sum/qdotk_max/integral : [B, nh, H_out, W_out]

    The local psi is built by RingUpsampleBackend.prepare: rows are keyed by the
    halo-padded LOCAL input latitude, arcs are (ho_local, lo, len) on the fine
    output grid with lo pre-shifted by -lon_lo_out.
    """

    @staticmethod
    @_custom_fwd(device_type="cuda")
    def forward(
        kw,
        vw,
        qw,
        psi_seg,
        psi_seg_off,
        ring_weights,
        num_heads: int,
        nlon_in: int,
        nlon_out_global: int,
        lon_chunk_starts: list,
        nlon_kx_list: list,
        lat_halo_start: int,
        nlat_out_local: int,
        nlon_out_local: int,
        az_group,
        az_rank: int,
        az_size: int,
    ):
        B = kw.shape[0]
        C_v = vw.shape[-1]
        device = kw.device

        # Capture input dtype so we can cast the user-visible output (y_out)
        # back at the end of forward. The internal accumulators stay fp32 —
        # softmax stability requires that, and the saved alpha_sum/qdotk_max
        # feed fp32 math in backward.
        inp_dtype = kw.dtype

        # State buffers in the kernels' ABI: y_acc packed like the activations,
        # one softmax statistic per (batch, head, point).
        y_acc = torch.zeros(B, nlat_out_local, nlon_out_local, C_v, device=device, dtype=torch.float32)
        alpha_sum = torch.zeros(B, num_heads, nlat_out_local, nlon_out_local, device=device, dtype=torch.float32)
        qdotk_max = torch.full((B, num_heads, nlat_out_local, nlon_out_local), float("-inf"), device=device, dtype=torch.float32)

        # already in the kernels' layout: the ring sends and receives these directly
        kw_chunk = kw.contiguous()
        vw_chunk = vw.contiguous()
        qw = qw.contiguous()

        for step in range(az_size):
            src_rank = (az_rank + step) % az_size
            lon_lo_kx = lon_chunk_starts[src_rank]

            # Pre-allocate receive buffers for the NEXT chunk (correct shape)
            if step < az_size - 1:
                next_src = (az_rank + step + 1) % az_size
                recv_kw, recv_vw, reqs = _ring_kv(kw_chunk, vw_chunk, az_group, nlon_kx_list[next_src], nlon_kx_list[next_src])

            attention_kernels.forward_ring_step_upsample.default(
                kw_chunk,
                vw_chunk,
                qw,
                y_acc,
                alpha_sum,
                qdotk_max,
                ring_weights,
                psi_seg,
                psi_seg_off,
                num_heads,
                nlon_in,
                nlon_out_global,
                lon_lo_kx,
                lat_halo_start,
                nlat_out_local,
                nlon_out_local,
            )

            if step < az_size - 1:
                for req in reqs:
                    req.wait()
                kw_chunk = recv_kw.clone()
                vw_chunk = recv_vw.clone()

        # Finalize: y = y_acc / alpha_sum, per head. Cast back to the input dtype
        # to keep the op faithful to its input dtype.
        y_out = (_per_head(y_acc, num_heads) / _stat_per_head(alpha_sum)).flatten(-2)  # [B, H, W, nh * C_v]
        y_out = y_out.to(dtype=inp_dtype)

        # alpha_sum and qdotk_max are returned so setup_context can save them;
        # they are marked non-differentiable there, so backward still only
        # receives one gradient argument (dy for y_out).
        return y_out, alpha_sum, qdotk_max

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        (
            kw,
            vw,
            qw,
            psi_seg,
            psi_seg_off,
            ring_weights,
            num_heads,
            nlon_in,
            nlon_out_global,
            lon_chunk_starts,
            nlon_kx_list,
            lat_halo_start,
            nlat_out_local,
            nlon_out_local,
            az_group,
            az_rank,
            az_size,
        ) = inputs
        y_out, alpha_sum, qdotk_max = output
        # alpha_sum and qdotk_max are internal accumulators, not true outputs;
        # marking them non-differentiable keeps backward's signature as (ctx, dy).
        ctx.mark_non_differentiable(alpha_sum, qdotk_max)
        ctx.save_for_backward(kw, vw, qw, psi_seg, psi_seg_off, ring_weights, alpha_sum, qdotk_max)
        ctx.num_heads = num_heads
        ctx.nlon_in = nlon_in
        ctx.nlon_out_global = nlon_out_global
        ctx.lon_chunk_starts = lon_chunk_starts
        ctx.nlon_kx_list = nlon_kx_list
        ctx.lat_halo_start = lat_halo_start
        ctx.nlat_out_local = nlat_out_local
        ctx.nlon_out_local = nlon_out_local
        ctx.az_group = az_group
        ctx.az_rank = az_rank
        ctx.az_size = az_size

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dy, _dalpha_sum, _dqdotk_max):
        # _dalpha_sum and _dqdotk_max are always None (non-differentiable outputs)
        kw, vw, qw, psi_seg, psi_seg_off, ring_weights, fwd_alpha_sum, fwd_qdotk_max = ctx.saved_tensors

        num_heads = ctx.num_heads
        nlon_in = ctx.nlon_in
        nlon_out_global = ctx.nlon_out_global
        lon_chunk_starts = ctx.lon_chunk_starts
        nlon_kx_list = ctx.nlon_kx_list
        lat_halo_start = ctx.lat_halo_start
        nlat_out_local = ctx.nlat_out_local
        nlon_out_local = ctx.nlon_out_local
        az_group = ctx.az_group
        az_rank = ctx.az_rank
        az_size = ctx.az_size

        # Autograd contract: skip per-branch work (kernel calls, ring exchanges) for any
        # of {kw, vw, qw} that doesn't need a gradient, and return None in those slots.
        kw_needs_grad = ctx.needs_input_grad[0]
        vw_needs_grad = ctx.needs_input_grad[1]
        qw_needs_grad = ctx.needs_input_grad[2]

        # Defensive: if somehow none of (kw, vw, qw) need grad, there's nothing to compute.
        if not (kw_needs_grad or vw_needs_grad or qw_needs_grad):
            return (None,) * 17

        B, H_halo, _, C_k = kw.shape
        C_v = vw.shape[-1]
        device = kw.device

        # Capture input dtypes so the returned grads can be cast back. The
        # backward kernels consume kw/vw/qw/dy in their native dtype (widen at
        # load, fp32 compute/accumulation), keeping the backward ring exchange
        # at 16-bit under AMP.
        kw_dtype = kw.dtype
        vw_dtype = vw.dtype
        qw_dtype = qw.dtype
        dy = dy.contiguous()  # [B, H, W, nh * C_v], native dtype
        qw = qw.contiguous()

        # ----------------------------------------------------------------
        # Backward pass 1: re-accumulate {integral, alpha_k, alpha_kvw} via
        # ring, using the SAVED forward alpha_sum/qdotk_max (no max recompute
        # is needed in the upsample direction — the forward-final softmax
        # stats are authoritative). Required whenever any of (kw, vw, qw)
        # needs grad: integral feeds pass-2's integral_norm and dqy reads
        # alpha_k/alpha_kvw. The kernel writes all buffers in one call, so
        # pass-1 cannot be pruned per-branch.
        # ----------------------------------------------------------------
        integral_buf = torch.zeros(B, num_heads, nlat_out_local, nlon_out_local, device=device, dtype=torch.float32)
        alpha_k_buf = torch.zeros(B, nlat_out_local, nlon_out_local, C_k, device=device, dtype=torch.float32)
        alpha_kvw_buf = torch.zeros_like(alpha_k_buf)

        kw = kw.contiguous()
        vw = vw.contiguous()
        kw_chunk, vw_chunk = kw, vw

        for step in range(az_size):
            src_rank = (az_rank + step) % az_size
            lon_lo_kx = lon_chunk_starts[src_rank]

            if step < az_size - 1:
                next_src = (az_rank + step + 1) % az_size
                recv_kw, recv_vw, reqs = _ring_kv(kw_chunk, vw_chunk, az_group, nlon_kx_list[next_src], nlon_kx_list[next_src])

            attention_kernels.backward_ring_step_upsample_pass1.default(
                kw_chunk,
                vw_chunk,
                qw,
                dy,
                fwd_qdotk_max,
                integral_buf,
                alpha_k_buf,
                alpha_kvw_buf,
                ring_weights,
                psi_seg,
                psi_seg_off,
                num_heads,
                nlon_in,
                nlon_out_global,
                lon_lo_kx,
                lat_halo_start,
                nlat_out_local,
                nlon_out_local,
            )

            if step < az_size - 1:
                for req in reqs:
                    req.wait()
                kw_chunk = recv_kw.clone()
                vw_chunk = recv_vw.clone()

        # ----------------------------------------------------------------
        # Finalize pass-1 outputs.
        # ----------------------------------------------------------------
        alpha_sum_inv = 1.0 / fwd_alpha_sum  # [B, nh, H, W]

        # integral_norm only feeds pass-2; skip if neither kw nor vw needs grad.
        if kw_needs_grad or vw_needs_grad:
            integral_norm = integral_buf * alpha_sum_inv  # [B, nh, H, W]

        # dqy[b,h,w,c] = inv_sq*(alpha_sum*alpha_kvw - integral*alpha_k), per head
        if qw_needs_grad:
            alpha_sum_inv_sq = alpha_sum_inv**2
            dqy = _stat_per_head(alpha_sum_inv_sq) * (
                _stat_per_head(fwd_alpha_sum) * _per_head(alpha_kvw_buf, num_heads) - _stat_per_head(integral_buf) * _per_head(alpha_k_buf, num_heads)
            )
            dqy = dqy.flatten(-2).to(dtype=qw_dtype)  # [B, H, W, nh * C_k]
        else:
            dqy = None

        # ----------------------------------------------------------------
        # Backward pass 2: accumulate dkw/dvw contributions per chunk.
        # Each GPU computes its LOCAL outputs' contribution to every lon chunk it visits,
        # and a ring reduce-scatter carries each chunk's accumulator back to its owner
        # (see _ring_grad) -- chunk-sized throughout, so no allreduce and no slice.
        # ----------------------------------------------------------------
        if kw_needs_grad or vw_needs_grad:
            # pass 1 rotated kw_chunk/vw_chunk; reset to the local chunk
            kw_chunk, vw_chunk = kw, vw
            # starts on this rank's own chunk (the one held at step 0), re-sized by each hop
            my_nlon = nlon_kx_list[az_rank]
            dkw_acc = torch.zeros(B, H_halo, my_nlon, C_k, device=device, dtype=torch.float32) if kw_needs_grad else None
            dvw_acc = torch.zeros(B, H_halo, my_nlon, C_v, device=device, dtype=torch.float32) if vw_needs_grad else None

            for step in range(az_size):
                src_rank = (az_rank + step) % az_size
                lon_lo_kx = lon_chunk_starts[src_rank]
                nlon_kx = nlon_kx_list[src_rank]

                # Unlike the gather direction, this kernel *assigns* its gradient outputs
                # (`dkx[chan] = sh_dk[chan]` in attention_cuda_bwd_ring_upsample.cu) rather
                # than accumulating with atomicAdd: in the scatter direction each input cell
                # is written by exactly one row, so it can store instead of reduce. The
                # accumulator therefore cannot be passed in directly -- each step would
                # overwrite the previous ones -- so it keeps a zeroed per-step temp and adds.
                dkw_chunk_cl = torch.zeros(B, H_halo, nlon_kx, C_k, device=device, dtype=torch.float32)
                dvw_chunk_cl = torch.zeros(B, H_halo, nlon_kx, C_v, device=device, dtype=torch.float32)

                attention_kernels.backward_ring_step_upsample_pass2.default(
                    kw_chunk,
                    vw_chunk,
                    qw,
                    dy,
                    fwd_alpha_sum,
                    fwd_qdotk_max,
                    integral_norm,
                    dkw_chunk_cl,
                    dvw_chunk_cl,
                    ring_weights,
                    psi_seg,
                    psi_seg_off,
                    num_heads,
                    nlon_in,
                    nlon_out_global,
                    lon_lo_kx,
                    lat_halo_start,
                    nlat_out_local,
                    nlon_out_local,
                )

                if kw_needs_grad:
                    dkw_acc.add_(dkw_chunk_cl)
                if vw_needs_grad:
                    dvw_acc.add_(dvw_chunk_cl)

                if step < az_size - 1:
                    next_src = (az_rank + step + 1) % az_size
                    recv_kw, recv_vw, reqs = _ring_kv(kw_chunk, vw_chunk, az_group, nlon_kx_list[next_src], nlon_kx_list[next_src])
                    # accumulators follow the chunks they belong to: same direction, same width
                    recv_dkw, recv_dvw, grad_reqs = _ring_grad(dkw_acc, dvw_acc, az_group, nlon_kx_list[next_src])
                    for req in reqs:
                        req.wait()
                    for req in grad_reqs:
                        req.wait()
                    kw_chunk = recv_kw.clone()
                    vw_chunk = recv_vw.clone()
                    # Cloned for the same reason as kw/vw, and with more at stake: the next
                    # step's add_ writes *into* the accumulator, so keeping the irecv
                    # destination would mutate a buffer the collective owns.
                    # The clone is chunk-sized, so it does not give back the saving above.
                    dkw_acc = recv_dkw.clone() if recv_dkw is not None else None
                    dvw_acc = recv_dvw.clone() if recv_dvw is not None else None

            # Final hop home: after the loop a rank carries the accumulator for chunk
            # az_rank - 1, which is exactly send_to.
            if az_size > 1:
                recv_dkw, recv_dvw, grad_reqs = _ring_grad(dkw_acc, dvw_acc, az_group, my_nlon)
                for req in grad_reqs:
                    req.wait()
                # Cloned for the same reason as in the loop: .to() below is a no-op for
                # fp32, so without it the returned gradient would alias the irecv
                # destination.
                dkw_acc = recv_dkw.clone() if recv_dkw is not None else None
                dvw_acc = recv_dvw.clone() if recv_dvw is not None else None

            # The accumulator IS the local chunk now, already in the layout of kw/vw.
            # No halo stripping: dkw/dvw must match kw/vw shape (= key_halo/value_halo).
            dkw = dkw_acc.to(dtype=kw_dtype) if kw_needs_grad else None  # [B, H_halo, W_local, nh * C_k]
            dvw = dvw_acc.to(dtype=vw_dtype) if vw_needs_grad else None  # [B, H_halo, W_local, nh * C_v]
        else:
            dkw = None
            dvw = None

        # Return grads for (kw, vw, qw, psi_seg, psi_seg_off, ring_weights,
        #                   num_heads, nlon_in, nlon_out_global, lon_chunk_starts,
        #                   nlon_kx_list, lat_halo_start, nlat_out_local, nlon_out_local,
        #                   az_group, az_rank, az_size)
        return (dkw, dvw, dqy) + (None,) * 14


# ---------------------------------------------------------------------------
# Ring backends
# ---------------------------------------------------------------------------


def _cast_and_halo(layer: "DistributedNeighborhoodAttentionS2", key: torch.Tensor, value: torch.Tensor, query_scaled: torch.Tensor):
    """
    What both ring backends do before the ring: the autocast cast, then the latitude halo.

    Under autocast, k/v/q are cast to the autocast dtype before ``.apply()`` -- mirrors
    PyTorch's autocast-eligible-op dataflow. Upstream projections under autocast already
    produce the autocast dtype, so this is usually a no-op; it covers an fp32-producing
    upstream. Casting first means the halo rows travel in that dtype too, not in fp32.

    key/value arrive as ``[B, H_in_local, W_in_local, C]``, latitude on dim 1. The halo
    exchange is differentiable and dtype-agnostic, and only runs when there is an actual
    polar split; otherwise it is the identity.
    """
    key, value, query_scaled = _cast_to_autocast_dtype(key, value, query_scaled)

    if layer.r_lat > 0 and layer.comm_size_polar > 1:
        key = polar_halo_exchange(key, layer.r_lat, lat_dim=1)
        value = polar_halo_exchange(value, layer.r_lat, lat_dim=1)

    return key, value, query_scaled


def _require_halo_covers(layer: "DistributedNeighborhoodAttentionS2", hi_global: torch.Tensor) -> None:
    """
    Every input latitude a local output row reaches must be in the halo-padded chunk.

    The ring kernels skip an arc whose latitude falls outside the chunk, as they must for
    pole padding, so a halo sized too small would not fail -- it would drop neighbours and
    return a plausible, wrong result. The halo is derived from the geometry by
    compute_polar_halo_radius; this holds that derivation to the pattern it has to serve.
    Construction time only, so none of it is traced.
    """
    lo = layer.lat_halo_start
    hi = lo + layer.nlat_in_local + 2 * layer.r_lat
    outside = (hi_global < lo) | (hi_global >= hi)
    if bool(outside.any()):
        missing = sorted(set(hi_global[outside].tolist()))
        raise RuntimeError(
            f"DistributedNeighborhoodAttentionS2: the latitude halo (r_lat={layer.r_lat}) does not cover the input "
            f"latitudes {missing[:8]}{'...' if len(missing) > 8 else ''} that this rank's output rows reach; the chunk "
            f"spans [{lo}, {hi}). This is a bug in the halo derivation, not in the arguments."
        )


class RingGatherBackend(AttentionBackendS2):
    """
    The ring kernels in the gather direction: self-attention and downsampling.

    State is this rank's output rows of the serial arcs. The arcs are folded -- one row
    per output latitude, the other longitudes of the ring reached by the p-shift -- so
    those rows are a contiguous slice, and nothing global is ever expanded. The arc
    starts are canonical at global wo = 0 and the kernels shift them by pscale * wo with
    the LOCAL wo, so the missing pscale * lon_lo_out is added here, once:

        (lo + pscale * (lon_lo_out + wo_local)) % nlon_in
            == ((lo + pscale * lon_lo_out) % nlon_in + pscale * wo_local) % nlon_in

    pscale = 1 when nlon_in == nlon_out (same-shape case).
    """

    name = "ring-gather"

    # No device test, like RaggedOptimizedBackend: the ring ops exist only for CUDA, and
    # a layer built on CPU is normally moved there before it runs, so construction must
    # not refuse. Called on CPU, the op dispatcher raises.
    @classmethod
    def available(cls, layer: "DistributedNeighborhoodAttentionS2", device: torch.device) -> bool:
        return not layer.upsample

    def prepare(self, layer: "DistributedNeighborhoodAttentionS2", device: torch.device) -> Dict[str, torch.Tensor]:
        lat_lo = layer.lat_lo_out
        lat_hi = lat_lo + layer.nlat_out_local

        arcs = layer._neighborhood_arcs()
        start = int(arcs.offsets[lat_lo])
        end = int(arcs.offsets[lat_hi])

        seg = arcs.segments[start:end].clone()
        seg_off = arcs.offsets[lat_lo : lat_hi + 1] - arcs.offsets[lat_lo]

        # the arcs' input latitudes are global; each must be inside the chunk
        _require_halo_covers(layer, seg[:, 0].to(torch.int64))

        pscale = layer.nlon_in // layer.nlon_out
        seg[:, 1] = (seg[:, 1] + pscale * layer.lon_lo_out) % layer.nlon_in

        return {"ring_weights": _ring_weights(layer, device), "psi_seg": seg.contiguous().to(device), "psi_seg_off": seg_off.contiguous().to(device)}

    def __call__(self, layer, key, value, query_scaled):
        key, value, query_scaled = _cast_and_halo(layer, key, value, query_scaled)

        out, _, _ = _RingNeighborhoodAttentionFn.apply(
            key,
            value,
            query_scaled,
            layer.psi_seg,
            layer.psi_seg_off,
            layer.ring_weights,
            layer.num_heads,
            layer.nlon_in,
            layer.nlon_out,
            layer.lon_in_starts,  # lon chunk starts for kv (same as lon_in)
            layer.lon_in_shapes,  # lon chunk sizes for kv
            layer.lat_halo_start,
            layer.nlat_out_local,
            layer.nlon_out_local,
            azimuth_group(),
            layer.comm_rank_azimuth,
            layer.comm_size_azimuth,
        )

        # [B, H_out_local, W_out_local, nh * C_v]
        return out


class RingUpsampleBackend(AttentionBackendS2):
    """
    The ring kernels in the scatter direction: upsampling.

    The serial arcs have rows keyed by the input lat hi in [0, nlat_in) and arcs
    (ho, lo, len) on the fine output grid, with lo canonical at wi = 0. This rank's
    state

      * re-keys the rows to the halo-padded LOCAL input lat range
        [lat_halo_start, lat_halo_start + nlat_halo); pole-padding rows
        (hi outside the global grid) are empty,
      * keeps only arcs whose output row ho falls into the local output shard
        and re-keys ho to ho_local = ho - lat_lo_out,
      * pre-shifts lo by -lon_lo_out (mod nlon_out), so the kernel's serial shift
        (lo + pscale_out * wi_global) mod nlon_out lands relative to this rank's
        first output longitude, and clipping to [0, nlon_out_local) keeps the
        part it owns.
    """

    name = "ring-upsample"

    # no device test; see RingGatherBackend
    @classmethod
    def available(cls, layer: "DistributedNeighborhoodAttentionS2", device: torch.device) -> bool:
        return layer.upsample

    def prepare(self, layer: "DistributedNeighborhoodAttentionS2", device: torch.device) -> Dict[str, torch.Tensor]:
        nlon_out = layer.nlon_out
        lat_lo_out = layer.lat_lo_out
        lat_hi_out = lat_lo_out + layer.nlat_out_local
        lon_lo_out = layer.lon_lo_out

        nlat_halo = layer.nlat_in_local + 2 * layer.r_lat

        arcs = layer._neighborhood_arcs()
        segments = arcs.segments
        offsets = arcs.offsets.to(torch.int64)

        # input-lat row index of every arc
        hi_of_seg = torch.repeat_interleave(torch.arange(layer.nlat_in, dtype=torch.int64), offsets.diff())
        ho = segments[:, 0].to(torch.int64)

        # every arc feeding a local output row must come from an input row in the chunk;
        # the in-range test below then only discards arcs of other ranks' output rows
        local_out = (ho >= lat_lo_out) & (ho < lat_hi_out)
        _require_halo_covers(layer, hi_of_seg[local_out])

        # keep arcs whose input row lies in the halo-padded local range and whose
        # output row is owned by this polar rank
        hi_local = hi_of_seg - layer.lat_halo_start
        mask = local_out & (hi_local >= 0) & (hi_local < nlat_halo)

        seg = segments[mask].clone()
        seg[:, 0] -= lat_lo_out
        seg[:, 1] = (seg[:, 1] - lon_lo_out) % nlon_out

        # rebuild the row offsets over the halo-padded local rows; masked selection
        # preserves the row-major order, so seg is already consistent with seg_off
        counts = torch.bincount(hi_local[mask], minlength=nlat_halo)
        seg_off = torch.zeros(nlat_halo + 1, dtype=arcs.offsets.dtype)
        seg_off[1:] = torch.cumsum(counts, dim=0)

        return {"ring_weights": _ring_weights(layer, device), "psi_seg": seg.contiguous().to(device), "psi_seg_off": seg_off.contiguous().to(device)}

    def __call__(self, layer, key, value, query_scaled):
        key, value, query_scaled = _cast_and_halo(layer, key, value, query_scaled)

        out, _, _ = _RingNeighborhoodAttentionUpsampleFn.apply(
            key,
            value,
            query_scaled,
            layer.psi_seg,
            layer.psi_seg_off,
            layer.ring_weights,
            layer.num_heads,
            layer.nlon_in,
            layer.nlon_out,
            layer.lon_in_starts,  # lon chunk starts for kv (same as lon_in)
            layer.lon_in_shapes,  # lon chunk sizes for kv
            layer.lat_halo_start,
            layer.nlat_out_local,
            layer.nlon_out_local,
            azimuth_group(),
            layer.comm_rank_azimuth,
            layer.comm_size_azimuth,
        )

        # [B, H_out_local, W_out_local, nh * C_v]
        return out


# ---------------------------------------------------------------------------
# Distributed Neighborhood Attention on the 2-sphere
# ---------------------------------------------------------------------------


class DistributedNeighborhoodAttentionS2(NeighborhoodAttentionS2):
    """
    Distributed neighborhood attention on the 2-sphere using a ring exchange
    strategy for the longitude dimension and halo exchange for the latitude
    dimension.

    Data is assumed to be split along both the latitude (polar group) and
    longitude (azimuth group) dimensions.  The forward pass uses ring exchange
    of key/value chunks over the azimuth group so that every output point can
    attend to its full spherical neighborhood.

    Self-attention (``grid_in == grid_out``), downsampling
    (``nlon_in % nlon_out == 0``) and upsampling (``nlon_out % nlon_in == 0``)
    cross-attention are supported. In all cases keys and values circulate around the
    azimuth ranks while queries and the softmax state stay local. Parameters are the
    same as for the serial layer.

    Requires the compiled kernels; ``optimized_kernel`` must be ``True``.

    .. seealso::
        :class:`torch_harmonics.NeighborhoodAttentionS2`
            Serial counterpart with full parameter documentation.
    """

    _backends = (RingGatherBackend, RingUpsampleBackend)

    @_rejects_legacy_signature
    def __init__(
        self,
        grid_in: RegularGridS2,
        grid_out: RegularGridS2,
        in_channels: int,
        num_heads: Optional[int] = 1,
        scale: Optional[Union[torch.Tensor, float]] = None,
        use_qknorm: Optional[bool] = False,
        bias: Optional[bool] = True,
        theta_cutoff: Optional[float] = None,
        k_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        optimized_kernel: Optional[bool] = True,
    ):
        # the ring exchange is implemented only in the compiled kernels; refuse a request for
        # the reference path rather than silently running the optimized one instead
        if not optimized_kernel:
            raise ValueError("DistributedNeighborhoodAttentionS2 has no reference implementation; optimized_kernel=False is not supported.")
        if not optimized_kernels_is_available():
            raise RuntimeError("Optimized kernels are required to run DistributedNeighborhoodAttentionS2.")

        # The base class accepts any GridS2, because the serial layer has a ragged path.
        # This one does not: the ring backends re-shift arc starts by pscale * lon_lo,
        # and the ring kernels shift them by pscale * wo, which is the p-shift of a
        # regular grid. On a grid whose rings differ in length none of that arithmetic
        # means anything -- it would not raise, it would silently address the wrong
        # points. So the assumption is stated here rather than inherited, as it is in
        # DistributedDiscreteContinuousConvS2.
        require_regular_grid(grid_in, "grid_in")
        require_regular_grid(grid_out, "grid_out")

        # the decomposition is settled in _setup, which the base class calls just before
        # it selects a backend
        super().__init__(
            grid_in,
            grid_out,
            in_channels,
            num_heads=num_heads,
            scale=scale,
            use_qknorm=use_qknorm,
            bias=bias,
            theta_cutoff=theta_cutoff,
            k_channels=k_channels,
            out_channels=out_channels,
            optimized_kernel=True,
        )

    def _setup(self) -> None:
        """Split the grids across the polar and azimuth groups, and size the latitude halo."""

        # ---- distributed info ----
        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # each grid decomposes itself. The ring and the halo exchange need every
        # rank's extent, not just this one's, which is what the shape lists carry.
        self.shard_in = self.grid_in.shard(
            polar=(self.comm_rank_polar, self.comm_size_polar),
            azimuth=(self.comm_rank_azimuth, self.comm_size_azimuth),
        )
        self.shard_out = self.grid_out.shard(
            polar=(self.comm_rank_polar, self.comm_size_polar),
            azimuth=(self.comm_rank_azimuth, self.comm_size_azimuth),
        )
        self.lat_in_shapes = list(self.shard_in.lat_shapes)
        self.lon_in_shapes = list(self.shard_in.lon_shapes)
        self.lat_out_shapes = list(self.shard_out.lat_shapes)
        self.lon_out_shapes = list(self.shard_out.lon_shapes)

        # local sizes for this rank
        self.nlat_in_local = self.shard_in.nlat
        self.nlon_in_local = self.shard_in.nlon
        self.nlat_out_local = self.shard_out.nlat
        self.nlon_out_local = self.shard_out.nlon

        # Uniform-pscale invariant: every azimuth rank must carry the same lon pscale.
        # The global divisibility check is inherited from the serial
        # NeighborhoodAttentionS2.__init__, but that is not sufficient in distributed:
        # if the grid hands different ranks different local pscales
        # (e.g. nlon_in=12, nlon_out=4, comm_size_azimuth=3 -> [4,4,4] vs [2,1,1]),
        # the p-shift mapping in the ring exchange is ill-defined.
        if self.upsample:
            pscale_lon = self.nlon_out // self.nlon_in
            for r, (lon_in_r, lon_out_r) in enumerate(zip(self.lon_in_shapes, self.lon_out_shapes)):
                if lon_out_r != pscale_lon * lon_in_r:
                    raise ValueError(
                        f"DistributedNeighborhoodAttentionS2: inconsistent azimuth split at rank {r}: "
                        f"nlon_in_local={lon_in_r}, nlon_out_local={lon_out_r}. "
                        f"Every azimuth rank must satisfy nlon_out_local == (nlon_out // nlon_in) * nlon_in_local "
                        f"= {pscale_lon} * nlon_in_local. "
                        f"Choose (nlon_in, nlon_out, comm_size_azimuth) so that the azimuth split "
                        f"produces uniform local pscale."
                    )
        else:
            pscale_lon = self.nlon_in // self.nlon_out
            for r, (lon_in_r, lon_out_r) in enumerate(zip(self.lon_in_shapes, self.lon_out_shapes)):
                if lon_in_r != pscale_lon * lon_out_r:
                    raise ValueError(
                        f"DistributedNeighborhoodAttentionS2: inconsistent azimuth split at rank {r}: "
                        f"nlon_in_local={lon_in_r}, nlon_out_local={lon_out_r}. "
                        f"Every azimuth rank must satisfy nlon_in_local == (nlon_in // nlon_out) * nlon_out_local "
                        f"= {pscale_lon} * nlon_out_local. "
                        f"Choose (nlon_in, nlon_out, comm_size_azimuth) so that the azimuth split "
                        f"produces uniform local pscale."
                    )

        # global lon offset of every rank's kv chunk, which the ring walks through
        self.lon_in_starts = list(accumulate([0] + self.lon_in_shapes[:-1]))

        self.lon_lo_out = self.shard_out.lon_offset
        self.lat_lo_out = self.shard_out.lat_offset

        # ---- lat halo size ----
        # Derived from the grid geometry rather than measured off the psi: an output latitude
        # can only reach input latitudes within theta_cutoff of it, and the halo is how far
        # that band runs past a rank's own input range. The criterion is symmetric in the two
        # grids, so the same call covers the gather and scatter directions -- the upsample path
        # builds its psi with the shapes swapped, but the latitudes it needs are the same ones.
        # It also raises if the halo outgrows a local chunk, which the immediate-neighbour
        # exchange could not serve.
        colats_in, colats_out = self.grid_in.colats, self.grid_out.colats
        self.r_lat = compute_polar_halo_radius(
            colats_in,
            colats_out,
            effective_theta_cutoff(self.theta_cutoff),
            self.lat_in_shapes,
            self.lat_out_shapes,
        )

        # global lat index of the first halo row, which is where the K/V chunks start
        self.lat_halo_start = self.shard_in.lat_offset - self.r_lat

    def _check_inputs(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """The serial contract, against this rank's shard rather than the global grid."""
        _check_ndim(query, 4, "query")
        _check_ndim(key, 4, "key")
        _check_ndim(value, 4, "value")
        _check_dtypes_match((query, key, value))
        _check_extent(query, -2, self.nlat_out_local, "query latitudes")
        _check_extent(query, -1, self.nlon_out_local, "query longitudes")
        _check_extent(key, -2, self.nlat_in_local, "key latitudes")
        _check_extent(key, -1, self.nlon_in_local, "key longitudes")
        _check_extent(value, -2, self.nlat_in_local, "value latitudes")
        _check_extent(value, -1, self.nlon_in_local, "value longitudes")
