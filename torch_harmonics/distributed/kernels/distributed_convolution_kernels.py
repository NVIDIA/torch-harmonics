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
Distributed DISCO conv kernel orchestration.

Two algorithms, dispatched via the public ``method=`` flag:

  ``a2a``  : the original shipped path. The azimuth split is bulk-gathered
             via an all-to-all (channel<->azimuth swap), the sparse psi
             contraction runs against the full nlon_in row, polar
             reduce_scatter completes the H sum, a second a2a swaps back
             to channel-distributed, then a local einsum with the
             replicated weight produces ``(B, O, H_out_local, W_out_local)``.
             Lower memory footprint than the legacy reduce-then-scatter
             variant; matches DistributedDiscreteContinuousConvS2's prior
             behavior exactly.

  ``ring`` : nearest-neighbor rotation along azimuth. Each rank rotates
             its W input chunk through P_az-1 P2P steps and accumulates
             partial contributions into its local W output range. The
             K-expanded intermediate never crosses any collective.
             Optional ``fused=True`` wraps the ring fwd + weight einsum
             into one autograd Function so only ``x`` (not the K-expanded
             tensor) is saved across the bwd boundary — bwd recomputes
             ``y_acc`` via a second ring fwd. Trades ~one extra ring fwd
             of compute for K× lower activation memory.

Both entry points take an already-distributed input
``(B, C, H_in_local, W_in_local)``, perform the polar reduce_scatter
internally, and return a polar-reduced
``(B, O, H_out_local, W_out_local)`` tensor. Bias is the caller's
responsibility.

Ring autograd Functions and their P2P helpers live below the public
entry points. ``optimized_kernels_is_available()`` (from disco_helpers)
gates whether the per-step CUDA kernels are present in this build.
"""

from typing import List, Optional

import torch
import torch.distributed as dist
from disco_helpers import optimized_kernels_is_available

from torch_harmonics.disco._disco_utils import _compute_dtype
from torch_harmonics.disco.kernels_torch.disco_torch import _disco_s2_contraction_torch
from torch_harmonics.disco.optimized.disco_optimized import _disco_s2_contraction_optimized
from torch_harmonics.distributed._amp_utils import _cast_to_autocast_dtype, _custom_setup_context

# The optimized ring-step CUDA wrappers are defined inside
# disco_optimized.py's ``if optimized_kernels_is_available():`` block, so
# they exist iff that helper returns True. Gate the import the same way
# instead of try/except — the public class's construction-time assert
# uses ``optimized_kernels_is_available()`` directly.
if optimized_kernels_is_available():
    from torch_harmonics.disco.optimized.disco_optimized import (
        _disco_s2_contraction_ring_step_optimized,
        _disco_s2_transpose_contraction_ring_step_optimized,
    )
else:
    _disco_s2_contraction_ring_step_optimized = None
    _disco_s2_transpose_contraction_ring_step_optimized = None

from torch_harmonics.distributed.primitives import (
    compute_split_shapes,
    distributed_transpose_azimuth,
    get_group_neighbors,
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
# Ring P2P helpers
# ---------------------------------------------------------------------------


@torch.compiler.disable()
def _ring_x_chunk(x_chunk, az_group, next_nlon_in_local, recv_buf=None):
    """Async send current x_chunk to the next azimuth neighbor, receive the
    chunk from the previous neighbor with known target W size. Returns
    ``(recv_buffer, P2P request list)`` — caller must wait on the requests
    before consuming recv_buffer.

    If ``recv_buf`` is provided, it is used as the receive destination
    (no allocation). The caller is responsible for ensuring it has the
    right shape/dtype/device. When ``recv_buf`` is None the function
    allocates a fresh tensor per call (legacy behavior).
    """
    send_to, recv_from = get_group_neighbors(az_group)
    if recv_buf is None:
        B, C, H_in_local, _ = x_chunk.shape
        recv_buf = torch.empty(
            B,
            C,
            H_in_local,
            next_nlon_in_local,
            device=x_chunk.device,
            dtype=x_chunk.dtype,
        )
    ops = [
        dist.P2POp(dist.isend, x_chunk, send_to, az_group),
        dist.P2POp(dist.irecv, recv_buf, recv_from, az_group),
    ]
    reqs = dist.batch_isend_irecv(ops)
    return recv_buf, reqs


@torch.compiler.disable()
def _ring_y_chunk(y_chunk, az_group, next_nlon_out_local, recv_buf=None):
    """Same as ``_ring_x_chunk`` but for a K-expanded 5D tensor — used by
    the non-fused ring backward to rotate ``grad_y`` chunks.

    If ``recv_buf`` is provided, it is used as the receive destination;
    otherwise a fresh tensor is allocated per call (legacy behavior).
    """
    send_to, recv_from = get_group_neighbors(az_group)
    if recv_buf is None:
        B, C, K, H_out_local, _ = y_chunk.shape
        recv_buf = torch.empty(
            B,
            C,
            K,
            H_out_local,
            next_nlon_out_local,
            device=y_chunk.device,
            dtype=y_chunk.dtype,
        )
    ops = [
        dist.P2POp(dist.isend, y_chunk, send_to, az_group),
        dist.P2POp(dist.irecv, recv_buf, recv_from, az_group),
    ]
    reqs = dist.batch_isend_irecv(ops)
    return recv_buf, reqs


# ---------------------------------------------------------------------------
# Ring autograd Functions
# ---------------------------------------------------------------------------
#
# Two variants:
#   _RingDiscoConvS2Fn       — unfused. Returns the K-expanded (B,C,K,H,W)
#                              tensor; the caller does the einsum with the
#                              replicated weight. Saves the K-expanded
#                              activation across fwd/bwd, so memory tracks
#                              the size of that intermediate.
#
#   _RingDiscoConvFusedFn    — fwd-ring + einsum fused into one autograd
#                              op. Saves ``x`` (not the K-expanded buffer)
#                              across the bwd boundary; bwd recomputes
#                              ``y_acc`` via a second ring fwd in the
#                              grad_w path. Memory drops by K× (minus the
#                              polar P factor we already pay), at the cost
#                              of one extra ring fwd of compute.


class _RingDiscoConvS2Fn(torch.autograd.Function):
    """Ring-exchange-based DISCO contraction over the azimuth communicator.

    Forward: rotates x chunks through P_az steps; each step accumulates the
    in-range psi entries' contribution into the local K-expanded output.

    Backward: rotates grad_y chunks through P_az steps in the same direction;
    each step accumulates the transpose contraction's contribution into the
    local grad_x.

    DISCO's per-step contribution is a pure additive accumulation (unlike
    attention's softmax, which needs a two-pass backward), so one kernel
    call per ring step is sufficient in each direction.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        x,
        psi_roff_idx,
        psi_ker_idx,
        psi_row_idx,
        psi_col_idx,
        psi_vals,
        kernel_size: int,
        nlat_out: int,
        nlon_out_local_self: int,
        nlon_in_global: int,
        pscale: int,
        lon_in_chunk_starts: List[int],
        lon_in_local_sizes: List[int],
        lon_out_chunk_starts: List[int],
        lon_out_local_sizes: List[int],
        az_group,
        az_rank: int,
        az_size: int,
        use_p2p_buffer: bool = False,
        comm_stream: Optional[torch.cuda.Stream] = None,
        recv_pool_fwd: Optional[List[torch.Tensor]] = None,
        recv_pool_bwd_grad_y_ke: Optional[List[torch.Tensor]] = None,
    ):
        if not optimized_kernels_is_available():
            raise NotImplementedError(
                "ring DISCO step kernel is not built. Add " "_disco_s2_contraction_ring_step_optimized + transpose variant " "to torch_harmonics/disco/optimized/."
            )

        B, C, H_in_local, _ = x.shape
        # K-expanded output accumulator; partial in polar. Allocated in
        # compute_dtype (fp32 under AMP) because the fwd ring-step kernel
        # uses a single STORAGE_T for both inp and out and the wrapper
        # upcasts inp to compute_dtype — out must match. Keeping the
        # accumulator in fp32 across ring steps also avoids the bf16/fp16
        # rounding-during-accumulation that would amplify cross-rank
        # cancellation (see _RingDiscoConvFwdFn.backward's grad_x_acc).
        compute_dtype = _compute_dtype(x.dtype)
        y_acc = torch.zeros(
            B,
            C,
            kernel_size,
            nlat_out,
            nlon_out_local_self,
            device=x.device,
            dtype=compute_dtype,
        )

        # Hoist the psi_vals cast out of the per-step wrapper (kernel
        # requires compute_dtype/fp32). The wrapper's defensive .to() is
        # then a no-op every ring step.
        psi_vals_c = psi_vals.to(compute_dtype)

        x_chunk = x.contiguous()

        # P2P recv buffers: prefer module-owned ones passed in via
        # ``recv_pool_fwd``; they survive past forward and avoid the
        # caching allocator recycling memory while NCCL's internal-stream
        # writes are still in flight. If the caller didn't pre-allocate
        # but use_p2p_buffer is set, fall back to per-call allocation
        # (legacy behavior — works for non-stacked single-shot calls).
        if recv_pool_fwd is not None:
            recv_pool = recv_pool_fwd
        elif use_p2p_buffer and az_size > 1:
            recv_pool = [
                torch.empty(
                    B,
                    C,
                    H_in_local,
                    lon_in_local_sizes[(az_rank + s + 1) % az_size],
                    device=x.device,
                    dtype=x.dtype,
                )
                for s in range(az_size - 1)
            ]
        else:
            recv_pool = None

        # Ring loop. When ``comm_stream`` is provided we post send/recv on
        # it so NCCL's at-issue event sync gates only against the comm
        # stream (not compute_stream's pending kernels), letting comm[i]
        # overlap with kernel[i]. The cs.wait_stream() / ev.record() /
        # compute_stream.wait_event() calls collapse to no-ops when
        # ``cs is compute_stream`` so the single-stream case matches the
        # pre-session behavior.
        compute_stream = torch.cuda.current_stream(x.device)
        cs = comm_stream if comm_stream is not None else compute_stream
        cs.wait_stream(compute_stream)
        ev = torch.cuda.Event() if az_size > 1 else None

        for step in range(az_size):
            src_rank = (az_rank + step) % az_size
            lon_lo_src = lon_in_chunk_starts[src_rank]
            nlon_in_local_src = lon_in_local_sizes[src_rank]

            if step < az_size - 1:
                next_src = (az_rank + step + 1) % az_size
                # Bring cs up to compute_stream's current state so the next
                # post sees any compute-side ops queued in the previous
                # iteration (notably ``recv_x.clone()`` on the no-pool path).
                cs.wait_stream(compute_stream)
                with torch.cuda.stream(cs):
                    recv_x, reqs = _ring_x_chunk(
                        x_chunk,
                        az_group,
                        lon_in_local_sizes[next_src],
                        recv_buf=recv_pool[step] if recv_pool is not None else None,
                    )

            _disco_s2_contraction_ring_step_optimized(
                x_chunk,
                y_acc,
                psi_roff_idx,
                psi_ker_idx,
                psi_row_idx,
                psi_col_idx,
                psi_vals_c,
                kernel_size,
                nlat_out,
                nlon_out_local_self,
                nlon_in_global,
                pscale,
                lon_lo_src,
                nlon_in_local_src,
            )

            if step < az_size - 1:
                with torch.cuda.stream(cs):
                    for req in reqs:
                        req.wait()
                    ev.record(cs)
                compute_stream.wait_event(ev)
                x_chunk = recv_x if recv_pool is not None else recv_x.clone()

        # Cast back to input dtype so the downstream einsum (and the bwd
        # K-expanded grad) sees the same dtype the caller expects.
        y_acc = y_acc.to(x.dtype)

        return y_acc

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        (
            x,
            psi_roff_idx,
            psi_ker_idx,
            psi_row_idx,
            psi_col_idx,
            psi_vals,
            kernel_size,
            nlat_out,
            nlon_out_local_self,
            nlon_in_global,
            pscale,
            lon_in_chunk_starts,
            lon_in_local_sizes,
            lon_out_chunk_starts,
            lon_out_local_sizes,
            az_group,
            az_rank,
            az_size,
            use_p2p_buffer,
            comm_stream,
            recv_pool_fwd,
            recv_pool_bwd_grad_y_ke,
        ) = inputs

        ctx.save_for_backward(
            psi_roff_idx,
            psi_ker_idx,
            psi_row_idx,
            psi_col_idx,
            psi_vals,
        )
        ctx.kernel_size = kernel_size
        ctx.nlat_out = nlat_out
        ctx.nlon_out_local_self = nlon_out_local_self
        ctx.nlon_in_global = nlon_in_global
        ctx.pscale = pscale
        ctx.lon_in_chunk_starts = lon_in_chunk_starts
        ctx.lon_in_local_sizes = lon_in_local_sizes
        ctx.lon_out_chunk_starts = lon_out_chunk_starts
        ctx.lon_out_local_sizes = lon_out_local_sizes
        ctx.az_group = az_group
        ctx.az_rank = az_rank
        ctx.az_size = az_size
        ctx.x_shape = tuple(x.shape)
        ctx.x_dtype = x.dtype
        ctx.x_device = x.device
        ctx.use_p2p_buffer = use_p2p_buffer
        ctx.comm_stream = comm_stream
        # Module-owned recv pool for the bwd grad_y rotation. None when
        # p2p_buffer is off or az_size <= 1; bwd falls back to per-call
        # allocation in that case.
        ctx.recv_pool_bwd_grad_y_ke = recv_pool_bwd_grad_y_ke

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_y_acc):
        if not optimized_kernels_is_available():
            raise NotImplementedError("ring DISCO transpose step kernel is not built.")

        (
            psi_roff_idx,
            psi_ker_idx,
            psi_row_idx,
            psi_col_idx,
            psi_vals,
        ) = ctx.saved_tensors

        kernel_size = ctx.kernel_size
        nlon_in_global = ctx.nlon_in_global
        pscale = ctx.pscale
        lon_in_chunk_starts = ctx.lon_in_chunk_starts
        lon_out_chunk_starts = ctx.lon_out_chunk_starts
        lon_out_local_sizes = ctx.lon_out_local_sizes
        az_group = ctx.az_group
        az_rank = ctx.az_rank
        az_size = ctx.az_size

        B, C, H_in_local, nlon_in_local_self = ctx.x_shape

        # grad_x_acc accumulates the input gradient for THIS rank's W range.
        # The transpose ring-step kernel atomicAdds into this buffer, so we
        # allocate it in compute_dtype (fp32 for fp16/bf16 inputs) for
        # precision; cast back to the input dtype after the ring loop.
        compute_dtype = _compute_dtype(ctx.x_dtype)
        grad_x_acc = torch.zeros(
            B,
            C,
            H_in_local,
            nlon_in_local_self,
            device=ctx.x_device,
            dtype=compute_dtype,
        )

        # psi_vals must be in compute_dtype for the kernel; cast once.
        psi_vals_c = psi_vals.to(compute_dtype)

        lon_lo_out_self = lon_out_chunk_starts[az_rank]
        grad_y_chunk = grad_y_acc.contiguous()
        lon_lo_in_self = lon_in_chunk_starts[az_rank]

        # Recv pool for grad_y (K-expanded) rotation. Prefer the
        # module-owned pool when use_p2p_buffer is on; otherwise fall
        # back to per-call allocation. See the fwd path for why module
        # ownership matters (caching allocator vs NCCL internal stream).
        K_dim = grad_y_acc.shape[2]
        H_out_local = grad_y_acc.shape[3]
        recv_pool_bwd = getattr(ctx, "recv_pool_bwd_grad_y_ke", None)
        if recv_pool_bwd is not None:
            recv_pool = recv_pool_bwd
        elif ctx.use_p2p_buffer and az_size > 1:
            recv_pool = [
                torch.empty(
                    B,
                    C,
                    K_dim,
                    H_out_local,
                    lon_out_local_sizes[(az_rank + s + 1) % az_size],
                    device=ctx.x_device,
                    dtype=grad_y_acc.dtype,
                )
                for s in range(az_size - 1)
            ]
        else:
            recv_pool = None

        # Same unified comm/compute stream pattern as the fwd ring loop.
        compute_stream = torch.cuda.current_stream(ctx.x_device)
        cs = ctx.comm_stream if ctx.comm_stream is not None else compute_stream
        cs.wait_stream(compute_stream)
        ev = torch.cuda.Event() if az_size > 1 else None

        for step in range(az_size):
            src_rank = (az_rank + step) % az_size
            lon_lo_src_out = lon_out_chunk_starts[src_rank]
            nlon_out_local_src = lon_out_local_sizes[src_rank]

            # pscale * (lon_lo_src_out - lon_lo_out_self): kernel input
            # absorbing the wi-shift mismatch between psi (built for
            # lon_lo_out_self) and this step's source W output range.
            pscale_wo_offset = pscale * (lon_lo_src_out - lon_lo_out_self)

            if step < az_size - 1:
                next_src = (az_rank + step + 1) % az_size
                # Bring cs up to compute_stream's state so the next post
                # sees the prev iter's ``recv_gy.clone()`` on the no-pool path.
                cs.wait_stream(compute_stream)
                with torch.cuda.stream(cs):
                    recv_gy, reqs = _ring_y_chunk(
                        grad_y_chunk,
                        az_group,
                        lon_out_local_sizes[next_src],
                        recv_buf=recv_pool[step] if recv_pool is not None else None,
                    )

            _disco_s2_transpose_contraction_ring_step_optimized(
                grad_y_chunk,
                grad_x_acc,
                psi_roff_idx,
                psi_ker_idx,
                psi_row_idx,
                psi_col_idx,
                psi_vals_c,
                kernel_size,
                H_in_local,
                nlon_in_local_self,
                nlon_in_global,
                pscale,
                pscale_wo_offset,
                lon_lo_in_self,
                nlon_out_local_src,
            )

            if step < az_size - 1:
                with torch.cuda.stream(cs):
                    for req in reqs:
                        req.wait()
                    ev.record(cs)
                compute_stream.wait_event(ev)
                grad_y_chunk = recv_gy if recv_pool is not None else recv_gy.clone()

        if compute_dtype != ctx.x_dtype:
            grad_x = grad_x_acc.to(ctx.x_dtype)
        else:
            grad_x = grad_x_acc

        # 21 non-tensor positional inputs; gradients return None for them.
        return (
            grad_x,
            None,
            None,
            None,
            None,
            None,  # psi tensors
            None,
            None,
            None,  # kernel_size, nlat_out, nlon_out_local_self
            None,
            None,  # nlon_in_global, pscale
            None,
            None,  # lon_in_chunk_starts, lon_in_local_sizes
            None,
            None,  # lon_out_chunk_starts, lon_out_local_sizes
            None,
            None,
            None,  # az_group, az_rank, az_size
            None,  # use_p2p_buffer
            None,  # comm_stream
            None,  # recv_pool_fwd
            None,  # recv_pool_bwd_grad_y_ke
        )


class _RingDiscoConvFusedFn(torch.autograd.Function):
    """Ring DISCO + weight contraction fused into one autograd op.

    Saves ``x`` (and ``weight`` + psi) across the fwd/bwd boundary — the
    K-expanded intermediate ``y_acc`` is reconstructed by a second ring
    fwd loop during backward's ``grad_w`` path.

    Returns ``(B, O, H_out, W_out_local)`` — still partial in polar; the
    caller's polar reduce_scatter completes the polar sum and splits H.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        x,
        weight,
        psi_roff_idx,
        psi_ker_idx,
        psi_row_idx,
        psi_col_idx,
        psi_vals,
        kernel_size: int,
        nlat_out: int,
        nlon_out_local_self: int,
        nlon_in_global: int,
        pscale: int,
        lon_in_chunk_starts: List[int],
        lon_in_local_sizes: List[int],
        lon_out_chunk_starts: List[int],
        lon_out_local_sizes: List[int],
        az_group,
        az_rank: int,
        az_size: int,
        groups: int,
        groupsize: int,
        use_p2p_buffer: bool = False,
        comm_stream: Optional[torch.cuda.Stream] = None,
        recv_pool_fwd: Optional[List[torch.Tensor]] = None,
        recv_pool_bwd_grad_out: Optional[List[torch.Tensor]] = None,
    ):
        if not optimized_kernels_is_available():
            raise NotImplementedError("ring DISCO step kernel is not built (fused path).")

        B, C, H_in_local, _ = x.shape

        # K-expanded accumulator — transient, NOT saved. Allocated in
        # compute_dtype (fp32 under AMP); see note in
        # _RingDiscoConvFwdFn.forward about the kernel's STORAGE_T contract
        # and fp32-across-ring-steps precision intent.
        compute_dtype = _compute_dtype(x.dtype)
        y_acc = torch.zeros(
            B,
            C,
            kernel_size,
            nlat_out,
            nlon_out_local_self,
            device=x.device,
            dtype=compute_dtype,
        )

        psi_vals_c = psi_vals.to(compute_dtype)

        x_chunk = x.contiguous()

        # P2P recv buffers: prefer module-owned ones passed in via
        # ``recv_pool_fwd``. See _RingDiscoConvS2Fn.forward for the
        # rationale (caching allocator vs. NCCL internal-stream writes).
        if recv_pool_fwd is not None:
            recv_pool = recv_pool_fwd
        elif use_p2p_buffer and az_size > 1:
            recv_pool = [
                torch.empty(
                    B,
                    C,
                    H_in_local,
                    lon_in_local_sizes[(az_rank + s + 1) % az_size],
                    device=x.device,
                    dtype=x.dtype,
                )
                for s in range(az_size - 1)
            ]
        else:
            recv_pool = None

        # See _RingDiscoConvS2Fn.forward for the comm/compute stream rationale.
        compute_stream = torch.cuda.current_stream(x.device)
        cs = comm_stream if comm_stream is not None else compute_stream
        cs.wait_stream(compute_stream)
        ev = torch.cuda.Event() if az_size > 1 else None

        for step in range(az_size):
            src_rank = (az_rank + step) % az_size
            lon_lo_src = lon_in_chunk_starts[src_rank]
            nlon_in_local_src = lon_in_local_sizes[src_rank]

            if step < az_size - 1:
                next_src = (az_rank + step + 1) % az_size
                # Bring cs up to compute_stream's current state so the next
                # post sees any compute-side ops queued in the previous
                # iteration (notably ``recv_x.clone()`` on the no-pool path).
                cs.wait_stream(compute_stream)
                with torch.cuda.stream(cs):
                    recv_x, reqs = _ring_x_chunk(
                        x_chunk,
                        az_group,
                        lon_in_local_sizes[next_src],
                        recv_buf=recv_pool[step] if recv_pool is not None else None,
                    )

            _disco_s2_contraction_ring_step_optimized(
                x_chunk,
                y_acc,
                psi_roff_idx,
                psi_ker_idx,
                psi_row_idx,
                psi_col_idx,
                psi_vals_c,
                kernel_size,
                nlat_out,
                nlon_out_local_self,
                nlon_in_global,
                pscale,
                lon_lo_src,
                nlon_in_local_src,
            )

            if step < az_size - 1:
                with torch.cuda.stream(cs):
                    for req in reqs:
                        req.wait()
                    ev.record(cs)
                compute_stream.wait_event(ev)
                x_chunk = recv_x if recv_pool is not None else recv_x.clone()

        # Cast back to input dtype before the einsum so the contraction
        # runs in the autocast-active dtype (matches the non-fused path
        # and the bwd grad_w recompute which casts to grad_out.dtype).
        y_acc = y_acc.to(x.dtype)

        # Weight contraction. y_acc dropped after this scope.
        H, W = nlat_out, nlon_out_local_self
        y_acc_r = y_acc.reshape(B, groups, groupsize, kernel_size, H, W)
        weight_r = weight.reshape(groups, -1, weight.shape[1], weight.shape[2])
        out = torch.einsum("bgckxy,gock->bgoxy", y_acc_r, weight_r).contiguous()
        out = out.reshape(B, -1, H, W)
        return out

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        (
            x,
            weight,
            psi_roff_idx,
            psi_ker_idx,
            psi_row_idx,
            psi_col_idx,
            psi_vals,
            kernel_size,
            nlat_out,
            nlon_out_local_self,
            nlon_in_global,
            pscale,
            lon_in_chunk_starts,
            lon_in_local_sizes,
            lon_out_chunk_starts,
            lon_out_local_sizes,
            az_group,
            az_rank,
            az_size,
            groups,
            groupsize,
            use_p2p_buffer,
            comm_stream,
            recv_pool_fwd,
            recv_pool_bwd_grad_out,
        ) = inputs

        # We save x — the whole point of the fused path. y_acc is NOT saved.
        ctx.save_for_backward(
            x,
            weight,
            psi_roff_idx,
            psi_ker_idx,
            psi_row_idx,
            psi_col_idx,
            psi_vals,
        )
        ctx.kernel_size = kernel_size
        ctx.nlat_out = nlat_out
        ctx.nlon_out_local_self = nlon_out_local_self
        ctx.nlon_in_global = nlon_in_global
        ctx.pscale = pscale
        ctx.lon_in_chunk_starts = lon_in_chunk_starts
        ctx.lon_in_local_sizes = lon_in_local_sizes
        ctx.lon_out_chunk_starts = lon_out_chunk_starts
        ctx.lon_out_local_sizes = lon_out_local_sizes
        ctx.az_group = az_group
        ctx.az_rank = az_rank
        ctx.az_size = az_size
        ctx.groups = groups
        ctx.groupsize = groupsize
        ctx.use_p2p_buffer = use_p2p_buffer
        ctx.comm_stream = comm_stream
        # Module-owned recv pools for the bwd. recv_pool_fwd is reused
        # by the grad_w recompute (same x shape as fwd); recv_pool_bwd_grad_out
        # is for the grad_x path's grad_out rotation.
        ctx.recv_pool_fwd = recv_pool_fwd
        ctx.recv_pool_bwd_grad_out = recv_pool_bwd_grad_out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        if not optimized_kernels_is_available():
            raise NotImplementedError("ring DISCO transpose step kernel is not built (fused path).")

        (
            x,
            weight,
            psi_roff_idx,
            psi_ker_idx,
            psi_row_idx,
            psi_col_idx,
            psi_vals,
        ) = ctx.saved_tensors

        K = ctx.kernel_size
        H = ctx.nlat_out
        W = ctx.nlon_out_local_self
        G = ctx.groups
        Cg = ctx.groupsize
        # weight stored as (out_channels, groupsize, kernel_size); reshape
        # treats it as (G, Og, Cg, K) where Og = out_channels // G.
        Og = weight.shape[0] // G
        B = grad_out.shape[0]
        C = x.shape[1]
        H_in_local = x.shape[2]
        nlon_in_local_self = x.shape[3]
        nlon_in_global = ctx.nlon_in_global
        pscale = ctx.pscale
        lon_in_chunk_starts = ctx.lon_in_chunk_starts
        lon_in_local_sizes = ctx.lon_in_local_sizes
        lon_out_chunk_starts = ctx.lon_out_chunk_starts
        lon_out_local_sizes = ctx.lon_out_local_sizes
        az_group = ctx.az_group
        az_rank = ctx.az_rank
        az_size = ctx.az_size

        compute_dtype = _compute_dtype(x.dtype)

        grad_x = None
        grad_weight = None

        # Both bwd einsums run in grad_out's dtype: under autocast bf16,
        # weight (fp32) and y_acc (fp32) get downcast to bf16 here so the
        # einsum hits tensor cores. Output is cast back to weight.dtype on
        # return for DDP/parameter-dtype compatibility.
        weight_r = weight.reshape(G, Og, Cg, K).to(grad_out.dtype)
        grad_out_r = grad_out.reshape(B, G, Og, H, W)

        # Hoist the psi_vals cast out of the per-step wrappers (both
        # branches use it). Saves one alloc+copy per ring step.
        psi_vals_c = psi_vals.to(compute_dtype)

        # ---- grad_x path ----
        # Rotate grad_out (O-channel, small) through the ring and compute
        # the K-expanded gradient chunk LOCALLY per step. Shrinks per-step
        # P2P traffic from B*C*K*H*W_chunk down to B*O*H*W_chunk — factor
        # of C·K/O (e.g. ~9× when C≈O, K=9). _ring_x_chunk's helper is
        # shape-generic, so the same rotation primitive works for the
        # O-channel grad_out tensor.
        if ctx.needs_input_grad[0]:
            grad_x_acc = torch.zeros(
                B,
                C,
                H_in_local,
                nlon_in_local_self,
                device=x.device,
                dtype=compute_dtype,
            )

            lon_lo_out_self = lon_out_chunk_starts[az_rank]
            lon_lo_in_self = lon_in_chunk_starts[az_rank]

            grad_out_chunk = grad_out.contiguous()
            # grad_out has shape (B, O_total, H, W_chunk); O_total = G*Og.
            O_total = grad_out_chunk.shape[1]

            # Recv pool for grad_out rotation. Prefer the module-owned pool.
            recv_pool_bwd = getattr(ctx, "recv_pool_bwd_grad_out", None)
            if recv_pool_bwd is not None:
                recv_pool = recv_pool_bwd
            elif ctx.use_p2p_buffer and az_size > 1:
                recv_pool = [
                    torch.empty(
                        B,
                        O_total,
                        H,
                        lon_out_local_sizes[(az_rank + s + 1) % az_size],
                        device=x.device,
                        dtype=grad_out.dtype,
                    )
                    for s in range(az_size - 1)
                ]
            else:
                recv_pool = None

            # Unified comm/compute stream pattern, mirroring the fwd ring loop.
            compute_stream = torch.cuda.current_stream(x.device)
            cs = ctx.comm_stream if ctx.comm_stream is not None else compute_stream
            cs.wait_stream(compute_stream)
            ev = torch.cuda.Event() if az_size > 1 else None

            for step in range(az_size):
                src_rank = (az_rank + step) % az_size
                lon_lo_src_out = lon_out_chunk_starts[src_rank]
                nlon_out_local_src = lon_out_local_sizes[src_rank]
                pscale_wo_offset = pscale * (lon_lo_src_out - lon_lo_out_self)

                if step < az_size - 1:
                    next_src = (az_rank + step + 1) % az_size
                    # See _RingDiscoConvS2Fn.forward for why this wait_stream
                    # is needed (no-pool clone is on compute_stream).
                    cs.wait_stream(compute_stream)
                    with torch.cuda.stream(cs):
                        recv_go, reqs = _ring_x_chunk(
                            grad_out_chunk,
                            az_group,
                            lon_out_local_sizes[next_src],
                            recv_buf=recv_pool[step] if recv_pool is not None else None,
                        )

                B_, O_, H_, W_chunk = grad_out_chunk.shape
                grad_out_chunk_r = grad_out_chunk.reshape(B_, G, Og, H_, W_chunk)
                grad_y_ke_chunk = torch.einsum(
                    "bgoxy,gock->bgckxy",
                    grad_out_chunk_r,
                    weight_r,
                ).contiguous()
                grad_y_ke_chunk = grad_y_ke_chunk.reshape(B_, C, K, H_, W_chunk)

                _disco_s2_transpose_contraction_ring_step_optimized(
                    grad_y_ke_chunk,
                    grad_x_acc,
                    psi_roff_idx,
                    psi_ker_idx,
                    psi_row_idx,
                    psi_col_idx,
                    psi_vals_c,
                    K,
                    H_in_local,
                    nlon_in_local_self,
                    nlon_in_global,
                    pscale,
                    pscale_wo_offset,
                    lon_lo_in_self,
                    nlon_out_local_src,
                )

                # Drop the K-expanded chunk before next iter allocates a new one.
                del grad_y_ke_chunk, grad_out_chunk_r

                if step < az_size - 1:
                    with torch.cuda.stream(cs):
                        for req in reqs:
                            req.wait()
                        ev.record(cs)
                    compute_stream.wait_event(ev)
                    grad_out_chunk = recv_go if recv_pool is not None else recv_go.clone()

            grad_x = grad_x_acc.to(x.dtype) if compute_dtype != x.dtype else grad_x_acc

            del grad_out_chunk

        # ---- grad_w path: recompute y_acc via a second ring fwd ----
        if ctx.needs_input_grad[1]:
            # Allocate in compute_dtype to match the fwd ring kernel's
            # STORAGE_T contract (the wrapper upcasts inp to fp32) and to
            # keep accumulation in fp32 across ring steps. The downstream
            # einsum already casts y_acc_r to grad_out.dtype below.
            y_acc = torch.zeros(
                B,
                C,
                K,
                H,
                W,
                device=x.device,
                dtype=compute_dtype,
            )
            x_chunk = x.contiguous()

            # Recv pool for the x_chunk rotation. Reuse the module-owned
            # fwd pool when available (same shape as fwd); otherwise fall
            # back to per-call allocation. The fwd pool was saved on ctx
            # during setup_context.
            recv_pool_fwd_ref = getattr(ctx, "recv_pool_fwd", None)
            if recv_pool_fwd_ref is not None:
                recv_pool = recv_pool_fwd_ref
            elif ctx.use_p2p_buffer and az_size > 1:
                recv_pool = [
                    torch.empty(
                        B,
                        C,
                        H_in_local,
                        lon_in_local_sizes[(az_rank + s + 1) % az_size],
                        device=x.device,
                        dtype=x.dtype,
                    )
                    for s in range(az_size - 1)
                ]
            else:
                recv_pool = None

            # Unified comm/compute stream pattern, mirroring the fwd ring loop.
            compute_stream = torch.cuda.current_stream(x.device)
            cs = ctx.comm_stream if ctx.comm_stream is not None else compute_stream
            cs.wait_stream(compute_stream)
            ev = torch.cuda.Event() if az_size > 1 else None

            for step in range(az_size):
                src_rank = (az_rank + step) % az_size
                lon_lo_src = lon_in_chunk_starts[src_rank]
                nlon_in_local_src = lon_in_local_sizes[src_rank]

                if step < az_size - 1:
                    next_src = (az_rank + step + 1) % az_size
                    cs.wait_stream(compute_stream)
                    with torch.cuda.stream(cs):
                        recv_x, reqs = _ring_x_chunk(
                            x_chunk,
                            az_group,
                            lon_in_local_sizes[next_src],
                            recv_buf=recv_pool[step] if recv_pool is not None else None,
                        )

                _disco_s2_contraction_ring_step_optimized(
                    x_chunk,
                    y_acc,
                    psi_roff_idx,
                    psi_ker_idx,
                    psi_row_idx,
                    psi_col_idx,
                    psi_vals_c,
                    K,
                    H,
                    W,
                    nlon_in_global,
                    pscale,
                    lon_lo_src,
                    nlon_in_local_src,
                )

                if step < az_size - 1:
                    with torch.cuda.stream(cs):
                        for req in reqs:
                            req.wait()
                        ev.record(cs)
                    compute_stream.wait_event(ev)
                    x_chunk = recv_x if recv_pool is not None else recv_x.clone()

            y_acc_r = y_acc.reshape(B, G, Cg, K, H, W).to(grad_out.dtype)
            grad_weight = torch.einsum("bgoxy,bgckxy->gock", grad_out_r, y_acc_r)
            grad_weight = grad_weight.reshape(weight.shape).to(weight.dtype).contiguous()

        # 25 positional inputs total; gradients are None except for x, weight.
        return (
            grad_x,
            grad_weight,
            None,
            None,
            None,
            None,
            None,  # psi tensors (5)
            None,
            None,
            None,  # kernel_size, nlat_out, nlon_out_local_self
            None,
            None,  # nlon_in_global, pscale
            None,
            None,  # lon_in_chunk_starts, lon_in_local_sizes
            None,
            None,  # lon_out_chunk_starts, lon_out_local_sizes
            None,
            None,
            None,  # az_group, az_rank, az_size
            None,
            None,  # groups, groupsize
            None,  # use_p2p_buffer
            None,  # comm_stream
            None,  # recv_pool_fwd
            None,  # recv_pool_bwd_grad_out
        )


# ---------------------------------------------------------------------------
# Ring entry point
# ---------------------------------------------------------------------------


def _distributed_disco_fwd_ring(
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
    nlon_out_local: int,
    nlon_in: int,
    pscale: int,
    lon_in_chunk_starts: List[int],
    lon_in_shapes: List[int],
    lon_out_chunk_starts: List[int],
    lon_out_shapes: List[int],
    az_group,
    az_rank: int,
    az_size: int,
    comm_size_azimuth: int,
    groups: int,
    groupsize: int,
    fused: bool,
    use_p2p_buffer: bool = False,
    comm_stream: Optional[torch.cuda.Stream] = None,
    recv_pool_fwd: Optional[List[torch.Tensor]] = None,
    recv_pool_bwd_grad_y_ke: Optional[List[torch.Tensor]] = None,
    recv_pool_bwd_grad_out: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """Ring-exchange distributed DISCO forward.

    Pattern (fused=False):
      1. Ring loop: rotate x chunks, accumulate K-expanded local output.
      2. Local einsum (C,K) × (O,C,K) → (B, O, H_out, W_out_local).
      3. Polar reduce_scatter on H.

    With fused=True the ring loop + einsum live inside one autograd op
    that only saves ``x`` across the bwd boundary (recomputes y_acc in
    bwd's grad_w path).

    Returns the polar-reduced output WITHOUT bias.
    """
    if not optimized_kernels_is_available():
        raise NotImplementedError('Ring DISCO step kernels are not built. Use method="a2a" until ' "the optimized ring kernels are added to torch_harmonics/disco/optimized/.")

    # Under autocast, cast activation + weight to the autocast dtype before
    # .apply() — matches the dataflow of PyTorch's built-in autocast-eligible
    # ops. Setup_context will then save the cast tensors and ring P2P chunks
    # travel at the smaller dtype. Internal accumulators stay fp32 per
    # _RingDiscoConv*Fn.forward.
    x, weight = _cast_to_autocast_dtype(x, weight)

    if fused:
        # Fused ring fwd + einsum, saving x (not the K-expanded
        # intermediate). bwd recomputes y_acc for the grad_w path.
        out = _RingDiscoConvFusedFn.apply(
            x,
            weight,
            psi_roff_idx,
            psi_ker_idx,
            psi_row_idx,
            psi_col_idx,
            psi_vals,
            kernel_size,
            nlat_out_local,
            nlon_out_local,
            nlon_in,
            pscale,
            lon_in_chunk_starts,
            lon_in_shapes,
            lon_out_chunk_starts,
            lon_out_shapes,
            az_group,
            az_rank,
            az_size,
            groups,
            groupsize,
            use_p2p_buffer,
            comm_stream,
            recv_pool_fwd,
            recv_pool_bwd_grad_out,
        )
    else:
        # If azimuth is not actually distributed, the ring degenerates to
        # one local kernel call — call the serial DISCO directly instead.
        if comm_size_azimuth == 1:
            x_ke = _disco_s2_contraction_optimized(
                x,
                psi_roff_idx,
                psi_ker_idx,
                psi_row_idx,
                psi_col_idx,
                psi_vals,
                kernel_size,
                nlat_out_local,
                nlon_out_local,
            )
        else:
            x_ke = _RingDiscoConvS2Fn.apply(
                x,
                psi_roff_idx,
                psi_ker_idx,
                psi_row_idx,
                psi_col_idx,
                psi_vals,
                kernel_size,
                nlat_out_local,
                nlon_out_local,
                nlon_in,
                pscale,
                lon_in_chunk_starts,
                lon_in_shapes,
                lon_out_chunk_starts,
                lon_out_shapes,
                az_group,
                az_rank,
                az_size,
                use_p2p_buffer,
                comm_stream,
                recv_pool_fwd,
                recv_pool_bwd_grad_y_ke,
            )

        # Local einsum with the replicated weight, run BEFORE the polar
        # reduce_scatter. The einsum is linear over both C and K, so it
        # commutes with the polar sum. Doing it here means the polar
        # collective below operates on (B, O, H_out, W_out_local) — no K
        # factor — which shrinks polar comm volume by C·K/O.
        B, C, K, H, W = x_ke.shape
        x_ke = x_ke.reshape(B, groups, groupsize, K, H, W)
        out = torch.einsum(
            "bgckxy,gock->bgoxy",
            x_ke,
            weight.reshape(groups, -1, weight.shape[1], weight.shape[2]),
        ).contiguous()
        out = out.reshape(B, -1, H, W)

    # Polar reduce_scatter on H — operates on the post-einsum tensor in
    # both fused and non-fused paths.
    out = reduce_from_scatter_to_polar_region(out, -2)
    return out
