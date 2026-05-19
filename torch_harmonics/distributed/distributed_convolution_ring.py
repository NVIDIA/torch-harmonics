# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Ring-exchange-based distributed DISCO convolution.

Algorithm
---------
The azimuth direction is NOT bulk-gathered via all-to-all. Instead, each
rank rotates its longitude chunk of the input through the azimuth
communicator over P_az steps, accumulating partial output contributions
into its local W output range. This mirrors the structure of distributed
neighborhood attention (_RingNeighborhoodAttentionFn): same per-rank
input chunk shape, same async P2P send/recv overlap pattern, but with the
DISCO sparse contraction as the per-step kernel instead of attention scoring.

Compared to the all-to-all variant:

  - The K-expanded intermediate never crosses any collective.
  - Comm pattern is P_az-1 nearest-neighbor sends instead of one bulk A2A.
  - No channel sharding required — works uniformly for groups=1, groups>1,
    and the depthwise case (groups == in_channels) where the A2A variant
    is incompatible.

The polar (latitude) direction still uses reduce_scatter for the H split,
exactly as in the existing DistributedDiscreteContinuousConvS2.

Psi preprocessing
-----------------
Beyond the polar-latitude split (handled by
``_split_distributed_convolution_tensor_s2``) we apply the same ``wi``
pre-shift that distributed_attention performs in its ``_build_local_psi``.
The shift absorbs this rank's azimuth output offset into ``col_idx`` so
the ring-step kernel does not need a ``lon_lo_self`` runtime parameter:

    col_idx natively stores  hi_global * nlon_in + wi_canonical
    we replace wi_canonical with
        wi_shifted = (wi_canonical + pscale * lon_lo_out_self) mod nlon_in
    so col_idx becomes  hi_global * nlon_in + wi_shifted

The kernel then evaluates:

    w_in_global = (wi_shifted + pscale * w_out_local) mod nlon_in_global
                = (wi_canonical + pscale * (lon_lo_out_self + w_out_local)) mod nlon_in
                = (wi_canonical + pscale * w_out_global) mod nlon_in

which matches the serial DISCO operation. With the wi shift baked into
psi at construction the kernel signature is one parameter simpler and
mirrors ``attention_kernels.forward_ring_step`` closely. This is handled
by ``_build_local_psi`` (analogous to the attention helper of the same
name).

Per-step kernel (CUDA — to be added separately)
-----------------------------------------------
    _disco_s2_contraction_ring_step_optimized(
        x_chunk,                                        # (B, C, H_in_local, nlon_in_local_src)
        y_acc,                                          # accumulator: (B, C, K, H_out, nlon_out_local_self)
        psi_roff_idx, psi_ker_idx, psi_row_idx,
        psi_col_idx,                                    # col = hi_global * nlon_in + wi_shifted (PRE-SHIFTED)
        psi_vals,
        kernel_size, H_out_full, nlon_out_local_self,
        nlon_in_global, pscale,
        lon_lo_src, nlon_in_local_src,                  # range of x_chunk in global lon
    )

    For each (k, h_out, w_out_local) and each psi entry (wi_shifted, h_in):
        w_in_global = (wi_shifted + pscale * w_out_local) mod nlon_in_global
        if lon_lo_src <= w_in_global < lon_lo_src + nlon_in_local_src:
            w_in_local_src = w_in_global - lon_lo_src
            y_acc[..., k, h_out, w_out_local] += psi_val * x_chunk[..., h_in, w_in_local_src]

The transpose variant has the symmetric shape — it accumulates into
grad_x_acc using grad_y_chunk as the held tensor; the inner check
becomes "is w_out_global in src's W output range" instead of "is
w_in_global in src's W input range". The same wi pre-shift in psi
applies, so the transpose kernel also does not need lon_lo_self.

Until the optimized kernel exists, _RING_KERNELS_AVAILABLE is False and
forward() raises NotImplementedError — callers should fall back to the
A2A-based DistributedDiscreteContinuousConvS2.
"""

from typing import Tuple, Union, Optional, List
from itertools import accumulate

import torch
import torch.distributed as dist

from torch_harmonics.disco._disco_utils import _get_psi, _compute_dtype
from torch_harmonics.disco.kernels_torch.disco_torch import _disco_s2_contraction_torch
from disco_helpers import optimized_kernels_is_available, preprocess_psi
from torch_harmonics.disco.convolution import (
    _precompute_convolution_tensor_s2,
    DiscreteContinuousConv,
)

# Optimized ring-step kernels are added separately in
# torch_harmonics/disco/optimized/. Probe for them at import time; the
# Python-side scaffolding works either way (and the class raises a clear
# error at construction if neither path is available).
try:
    from torch_harmonics.disco.optimized.disco_optimized import (
        _disco_s2_contraction_ring_step_optimized,
        _disco_s2_transpose_contraction_ring_step_optimized,
    )
    _RING_KERNELS_AVAILABLE = True
except ImportError:
    _disco_s2_contraction_ring_step_optimized = None
    _disco_s2_transpose_contraction_ring_step_optimized = None
    _RING_KERNELS_AVAILABLE = False

from torch_harmonics.distributed import (
    polar_group_size,
    polar_group_rank,
    azimuth_group,
    azimuth_group_size,
    azimuth_group_rank,
    compute_split_shapes,
    reduce_from_scatter_to_polar_region,
)
from torch_harmonics.distributed.primitives import get_group_neighbors
from torch_harmonics.distributed.distributed_convolution import _split_distributed_convolution_tensor_s2


# ---------------------------------------------------------------------------
# Async P2P helper — mirror of _ring_kv in distributed_attention but for a
# single tensor (no kw/vw split).
# ---------------------------------------------------------------------------

@torch.compiler.disable()
def _ring_x_chunk(x_chunk, az_group, next_nlon_in_local):
    """Async send current x_chunk to next neighbor, receive the chunk from
    the previous neighbor (with known target size). Returns (recv_buffer,
    P2P request list)."""
    send_to, recv_from = get_group_neighbors(az_group)
    B, C, H_in_local, _ = x_chunk.shape
    recv_x = torch.empty(
        B, C, H_in_local, next_nlon_in_local,
        device=x_chunk.device, dtype=x_chunk.dtype,
    )
    ops = [
        dist.P2POp(dist.isend, x_chunk, send_to,   az_group),
        dist.P2POp(dist.irecv, recv_x,  recv_from, az_group),
    ]
    reqs = dist.batch_isend_irecv(ops)
    return recv_x, reqs


@torch.compiler.disable()
def _ring_y_chunk(y_chunk, az_group, next_nlon_out_local):
    """Same as _ring_x_chunk but for the K-expanded output gradient
    tensor rotated through the ring in backward."""
    send_to, recv_from = get_group_neighbors(az_group)
    B, C, K, H_out_local, _ = y_chunk.shape
    recv_y = torch.empty(
        B, C, K, H_out_local, next_nlon_out_local,
        device=y_chunk.device, dtype=y_chunk.dtype,
    )
    ops = [
        dist.P2POp(dist.isend, y_chunk, send_to,   az_group),
        dist.P2POp(dist.irecv, recv_y,  recv_from, az_group),
    ]
    reqs = dist.batch_isend_irecv(ops)
    return recv_y, reqs


# ---------------------------------------------------------------------------
# Ring DISCO autograd Function
# ---------------------------------------------------------------------------

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
    def forward(
        x,
        psi_roff_idx, psi_ker_idx, psi_row_idx, psi_col_idx, psi_vals,
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
    ):
        if not _RING_KERNELS_AVAILABLE:
            raise NotImplementedError(
                "ring DISCO step kernel is not built. Add "
                "_disco_s2_contraction_ring_step_optimized + transpose variant "
                "to torch_harmonics/disco/optimized/. See module docstring for "
                "the expected signature."
            )

        B, C, H_in_local, _ = x.shape
        # K-expanded output accumulator; partial in polar.
        y_acc = torch.zeros(
            B, C, kernel_size, nlat_out, nlon_out_local_self,
            device=x.device, dtype=x.dtype,
        )

        x_chunk = x.contiguous()

        for step in range(az_size):
            src_rank   = (az_rank + step) % az_size
            lon_lo_src = lon_in_chunk_starts[src_rank]
            nlon_in_local_src = lon_in_local_sizes[src_rank]

            # Pre-post the async send/recv for the NEXT step's chunk.
            if step < az_size - 1:
                next_src = (az_rank + step + 1) % az_size
                recv_x, reqs = _ring_x_chunk(
                    x_chunk, az_group, lon_in_local_sizes[next_src],
                )

            # Accumulate this step's contribution. Psi's col_idx is already
            # pre-shifted by pscale * lon_lo_out_self (see _build_local_psi),
            # so the kernel does not need lon_lo_self.
            _disco_s2_contraction_ring_step_optimized(
                x_chunk, y_acc,
                psi_roff_idx, psi_ker_idx, psi_row_idx, psi_col_idx, psi_vals,
                kernel_size, nlat_out, nlon_out_local_self,
                nlon_in_global, pscale,
                lon_lo_src, nlon_in_local_src,
            )

            if step < az_size - 1:
                for req in reqs:
                    req.wait()
                x_chunk = recv_x.clone()

        return y_acc

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x,
         psi_roff_idx, psi_ker_idx, psi_row_idx, psi_col_idx, psi_vals,
         kernel_size, nlat_out, nlon_out_local_self,
         nlon_in_global, pscale,
         lon_in_chunk_starts, lon_in_local_sizes,
         lon_out_chunk_starts, lon_out_local_sizes,
         az_group, az_rank, az_size) = inputs

        ctx.save_for_backward(
            psi_roff_idx, psi_ker_idx, psi_row_idx, psi_col_idx, psi_vals,
        )
        ctx.kernel_size            = kernel_size
        ctx.nlat_out               = nlat_out
        ctx.nlon_out_local_self    = nlon_out_local_self
        ctx.nlon_in_global         = nlon_in_global
        ctx.pscale                 = pscale
        ctx.lon_in_chunk_starts    = lon_in_chunk_starts
        ctx.lon_in_local_sizes     = lon_in_local_sizes
        ctx.lon_out_chunk_starts   = lon_out_chunk_starts
        ctx.lon_out_local_sizes    = lon_out_local_sizes
        ctx.az_group               = az_group
        ctx.az_rank                = az_rank
        ctx.az_size                = az_size
        ctx.x_shape                = tuple(x.shape)
        ctx.x_dtype                = x.dtype
        ctx.x_device               = x.device

    @staticmethod
    def backward(ctx, grad_y_acc):
        if not _RING_KERNELS_AVAILABLE:
            raise NotImplementedError(
                "ring DISCO transpose step kernel is not built. See module "
                "docstring for the expected signature."
            )

        (psi_roff_idx, psi_ker_idx, psi_row_idx, psi_col_idx, psi_vals,
         ) = ctx.saved_tensors

        kernel_size           = ctx.kernel_size
        nlat_out              = ctx.nlat_out
        nlon_in_global        = ctx.nlon_in_global
        pscale                = ctx.pscale
        lon_in_chunk_starts   = ctx.lon_in_chunk_starts
        lon_in_local_sizes    = ctx.lon_in_local_sizes
        lon_out_chunk_starts  = ctx.lon_out_chunk_starts
        lon_out_local_sizes   = ctx.lon_out_local_sizes
        az_group              = ctx.az_group
        az_rank               = ctx.az_rank
        az_size               = ctx.az_size

        B, C, H_in_local, nlon_in_local_self = ctx.x_shape

        # grad_x_acc accumulates the input gradient for THIS rank's W range.
        # The transpose ring-step kernel does atomicAdds into this buffer, so
        # we allocate it in the compute dtype (fp32 for fp16/bf16 inputs) for
        # precision; we cast back to the input dtype after the ring loop.
        compute_dtype = _compute_dtype(ctx.x_dtype)
        grad_x_acc = torch.zeros(
            B, C, H_in_local, nlon_in_local_self,
            device=ctx.x_device, dtype=compute_dtype,
        )

        # Precomputed offset for the wi-shift compensation in the kernel.
        # See module docstring: psi was wi-shifted for THIS rank's
        # lon_lo_out_self at construction (mirroring attention's
        # _build_local_psi), so when bwd consumes src's wo range we need to
        # add pscale * (lon_lo_src_out - lon_lo_out_self) to the resolved
        # wi index. This is constant per ring step.
        lon_lo_out_self = lon_out_chunk_starts[az_rank]

        # Rotate grad_y chunks through the ring in the same direction as
        # forward. At step k we hold the grad_y chunk that belongs to
        # src_rank's W output range; the transpose contraction kernel
        # contributes the (src's w_out → my w_in) terms to grad_x_acc.
        grad_y_chunk = grad_y_acc.contiguous()

        # This rank's input W offset — entries that don't resolve to a
        # wi_global in [lon_lo_in_self, lon_lo_in_self + nlon_in_local_self)
        # are skipped for this src chunk.
        lon_lo_in_self = lon_in_chunk_starts[az_rank]

        for step in range(az_size):
            src_rank          = (az_rank + step) % az_size
            lon_lo_src_out    = lon_out_chunk_starts[src_rank]
            nlon_out_local_src = lon_out_local_sizes[src_rank]

            # pscale * (lon_lo_src_out - lon_lo_out_self), passed to the
            # kernel to absorb the wi-shift mismatch.
            pscale_wo_offset = pscale * (lon_lo_src_out - lon_lo_out_self)

            if step < az_size - 1:
                next_src = (az_rank + step + 1) % az_size
                recv_gy, reqs = _ring_y_chunk(
                    grad_y_chunk, az_group, lon_out_local_sizes[next_src],
                )

            _disco_s2_transpose_contraction_ring_step_optimized(
                grad_y_chunk, grad_x_acc,
                psi_roff_idx, psi_ker_idx, psi_row_idx, psi_col_idx, psi_vals,
                kernel_size, H_in_local, nlon_in_local_self,
                nlon_in_global, pscale,
                pscale_wo_offset, lon_lo_in_self,
                nlon_out_local_src,
            )

            if step < az_size - 1:
                for req in reqs:
                    req.wait()
                grad_y_chunk = recv_gy.clone()

        # Cast back to the input dtype. The kernel accumulated in compute_t
        # for atomicAdd precision; downstream consumers expect ctx.x_dtype.
        if compute_dtype != ctx.x_dtype:
            grad_x = grad_x_acc.to(ctx.x_dtype)
        else:
            grad_x = grad_x_acc

        # 17 non-tensor positional inputs; gradients return None for them.
        return (
            grad_x,
            None, None, None, None, None,   # psi tensors
            None, None, None,               # kernel_size, nlat_out, nlon_out_local_self
            None, None,                     # nlon_in_global, pscale
            None, None,                     # lon_in_chunk_starts, lon_in_local_sizes
            None, None,                     # lon_out_chunk_starts, lon_out_local_sizes
            None, None, None,               # az_group, az_rank, az_size
        )


# ---------------------------------------------------------------------------
# Fused ring DISCO autograd Function
# ---------------------------------------------------------------------------

class _RingDiscoConvFusedFn(torch.autograd.Function):
    """Ring DISCO + weight contraction fused into one autograd op.

    Saves only ``x`` (and ``weight`` + psi) across the fwd/bwd boundary —
    the K-expanded intermediate ``y_acc`` is reconstructed by a second
    ring fwd loop during backward's ``grad_w`` path. This mirrors the
    recomputation pattern of the serial fused DISCO op
    (``_disco_s2_fused_conv_optimized`` on the tkurth/fused-disco branch).

    Trade: saved-activation memory shrinks from ``(B, C, K, H_out, W_local)``
    down to ``(B, C, H_in_local, W_local)`` (K× plus the polar P factor
    we already pay in the reordered path); bwd does one extra ring fwd
    pass for the grad_w recompute when both gradients are needed.

    Returns ``(B, O, H_out, W_out_local)`` — still partial in polar; the
    caller's polar reduce_scatter completes the polar sum and splits H.
    """

    @staticmethod
    def forward(
        x, weight,
        psi_roff_idx, psi_ker_idx, psi_row_idx, psi_col_idx, psi_vals,
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
    ):
        if not _RING_KERNELS_AVAILABLE:
            raise NotImplementedError(
                "ring DISCO step kernel is not built (fused path)."
            )

        B, C, H_in_local, _ = x.shape

        # K-expanded accumulator — transient, NOT saved.
        y_acc = torch.zeros(
            B, C, kernel_size, nlat_out, nlon_out_local_self,
            device=x.device, dtype=x.dtype,
        )

        x_chunk = x.contiguous()
        for step in range(az_size):
            src_rank          = (az_rank + step) % az_size
            lon_lo_src        = lon_in_chunk_starts[src_rank]
            nlon_in_local_src = lon_in_local_sizes[src_rank]

            if step < az_size - 1:
                next_src = (az_rank + step + 1) % az_size
                recv_x, reqs = _ring_x_chunk(
                    x_chunk, az_group, lon_in_local_sizes[next_src],
                )

            _disco_s2_contraction_ring_step_optimized(
                x_chunk, y_acc,
                psi_roff_idx, psi_ker_idx, psi_row_idx, psi_col_idx, psi_vals,
                kernel_size, nlat_out, nlon_out_local_self,
                nlon_in_global, pscale,
                lon_lo_src, nlon_in_local_src,
            )

            if step < az_size - 1:
                for req in reqs:
                    req.wait()
                x_chunk = recv_x.clone()

        # Weight contraction. y_acc dropped after this scope.
        H, W = nlat_out, nlon_out_local_self
        y_acc_r = y_acc.reshape(B, groups, groupsize, kernel_size, H, W)
        weight_r = weight.reshape(groups, -1, weight.shape[1], weight.shape[2])
        out = torch.einsum("bgckxy,gock->bgoxy", y_acc_r, weight_r).contiguous()
        out = out.reshape(B, -1, H, W)
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x, weight,
         psi_roff_idx, psi_ker_idx, psi_row_idx, psi_col_idx, psi_vals,
         kernel_size, nlat_out, nlon_out_local_self,
         nlon_in_global, pscale,
         lon_in_chunk_starts, lon_in_local_sizes,
         lon_out_chunk_starts, lon_out_local_sizes,
         az_group, az_rank, az_size,
         groups, groupsize) = inputs

        # We save x — the whole point of the fused path. y_acc is NOT saved.
        ctx.save_for_backward(
            x, weight,
            psi_roff_idx, psi_ker_idx, psi_row_idx, psi_col_idx, psi_vals,
        )
        ctx.kernel_size            = kernel_size
        ctx.nlat_out               = nlat_out
        ctx.nlon_out_local_self    = nlon_out_local_self
        ctx.nlon_in_global         = nlon_in_global
        ctx.pscale                 = pscale
        ctx.lon_in_chunk_starts    = lon_in_chunk_starts
        ctx.lon_in_local_sizes     = lon_in_local_sizes
        ctx.lon_out_chunk_starts   = lon_out_chunk_starts
        ctx.lon_out_local_sizes    = lon_out_local_sizes
        ctx.az_group               = az_group
        ctx.az_rank                = az_rank
        ctx.az_size                = az_size
        ctx.groups                 = groups
        ctx.groupsize              = groupsize

    @staticmethod
    def backward(ctx, grad_out):
        if not _RING_KERNELS_AVAILABLE:
            raise NotImplementedError(
                "ring DISCO transpose step kernel is not built (fused path)."
            )

        (x, weight,
         psi_roff_idx, psi_ker_idx, psi_row_idx, psi_col_idx, psi_vals,
         ) = ctx.saved_tensors

        K  = ctx.kernel_size
        H  = ctx.nlat_out
        W  = ctx.nlon_out_local_self
        G  = ctx.groups
        Cg = ctx.groupsize
        # weight is stored as (out_channels, groupsize, kernel_size); reshape
        # treats it as (G, Og, Cg, K) where Og = out_channels // G.
        Og = weight.shape[0] // G
        B  = grad_out.shape[0]
        C  = x.shape[1]
        H_in_local         = x.shape[2]
        nlon_in_local_self = x.shape[3]
        nlon_in_global     = ctx.nlon_in_global
        pscale             = ctx.pscale
        lon_in_chunk_starts  = ctx.lon_in_chunk_starts
        lon_in_local_sizes   = ctx.lon_in_local_sizes
        lon_out_chunk_starts = ctx.lon_out_chunk_starts
        lon_out_local_sizes  = ctx.lon_out_local_sizes
        az_group = ctx.az_group
        az_rank  = ctx.az_rank
        az_size  = ctx.az_size

        compute_dtype = _compute_dtype(x.dtype)

        grad_x      = None
        grad_weight = None

        weight_r   = weight.reshape(G, Og, Cg, K).to(grad_out.dtype)
        grad_out_r = grad_out.reshape(B, G, Og, H, W)

        # ---- grad_x path ----
        # Reordered comm: rotate grad_out (O-channel, small) through the
        # ring and compute the K-expanded gradient chunk LOCALLY per step.
        # Shrinks per-step P2P traffic from B*C*K*H*W_chunk down to
        # B*O*H*W_chunk — factor of C·K/O (e.g. ~9× when C≈O, K=9).
        # _ring_x_chunk's helper is shape-generic (B, *, *, W), so the
        # same rotation primitive works for the O-channel grad_out tensor.
        if ctx.needs_input_grad[0]:
            grad_x_acc = torch.zeros(
                B, C, H_in_local, nlon_in_local_self,
                device=x.device, dtype=compute_dtype,
            )

            lon_lo_out_self = lon_out_chunk_starts[az_rank]
            lon_lo_in_self  = lon_in_chunk_starts[az_rank]

            grad_out_chunk = grad_out.contiguous()
            for step in range(az_size):
                src_rank           = (az_rank + step) % az_size
                lon_lo_src_out     = lon_out_chunk_starts[src_rank]
                nlon_out_local_src = lon_out_local_sizes[src_rank]
                pscale_wo_offset   = pscale * (lon_lo_src_out - lon_lo_out_self)

                # Async P2P for the NEXT step's grad_out chunk. Note we use
                # _ring_x_chunk (generic 4D rotator); the "x" in the name is
                # historical — the helper doesn't care whether dim 1 means
                # input channels or output channels.
                if step < az_size - 1:
                    next_src = (az_rank + step + 1) % az_size
                    recv_go, reqs = _ring_x_chunk(
                        grad_out_chunk, az_group, lon_out_local_sizes[next_src],
                    )

                # Local einsum bwd: (B,O,H,W_chunk) x (G,Og,Cg,K) -> K-expanded chunk.
                B_, O_, H_, W_chunk = grad_out_chunk.shape
                grad_out_chunk_r = grad_out_chunk.reshape(B_, G, Og, H_, W_chunk)
                grad_y_ke_chunk = torch.einsum(
                    "bgoxy,gock->bgckxy", grad_out_chunk_r, weight_r,
                ).contiguous()
                grad_y_ke_chunk = grad_y_ke_chunk.reshape(B_, C, K, H_, W_chunk)

                _disco_s2_transpose_contraction_ring_step_optimized(
                    grad_y_ke_chunk, grad_x_acc,
                    psi_roff_idx, psi_ker_idx, psi_row_idx, psi_col_idx, psi_vals,
                    K, H_in_local, nlon_in_local_self,
                    nlon_in_global, pscale,
                    pscale_wo_offset, lon_lo_in_self,
                    nlon_out_local_src,
                )

                # Drop the K-expanded chunk before next iter allocates a new one.
                del grad_y_ke_chunk, grad_out_chunk_r

                if step < az_size - 1:
                    for req in reqs:
                        req.wait()
                    grad_out_chunk = recv_go.clone()

            grad_x = grad_x_acc.to(x.dtype) if compute_dtype != x.dtype else grad_x_acc

            del grad_out_chunk

        # ---- grad_w path: recompute y_acc via a second ring fwd ----
        if ctx.needs_input_grad[1]:
            y_acc = torch.zeros(
                B, C, K, H, W,
                device=x.device, dtype=x.dtype,
            )
            x_chunk = x.contiguous()
            for step in range(az_size):
                src_rank          = (az_rank + step) % az_size
                lon_lo_src        = lon_in_chunk_starts[src_rank]
                nlon_in_local_src = lon_in_local_sizes[src_rank]

                if step < az_size - 1:
                    next_src = (az_rank + step + 1) % az_size
                    recv_x, reqs = _ring_x_chunk(
                        x_chunk, az_group, lon_in_local_sizes[next_src],
                    )

                _disco_s2_contraction_ring_step_optimized(
                    x_chunk, y_acc,
                    psi_roff_idx, psi_ker_idx, psi_row_idx, psi_col_idx, psi_vals,
                    K, H, W,
                    nlon_in_global, pscale,
                    lon_lo_src, nlon_in_local_src,
                )

                if step < az_size - 1:
                    for req in reqs:
                        req.wait()
                    x_chunk = recv_x.clone()

            y_acc_r = y_acc.reshape(B, G, Cg, K, H, W)
            grad_weight = torch.einsum("bgoxy,bgckxy->gock", grad_out_r, y_acc_r)
            grad_weight = grad_weight.reshape(weight.shape).contiguous()

        # 21 positional inputs total; gradients are None except for x, weight.
        return (
            grad_x, grad_weight,
            None, None, None, None, None,   # psi tensors (5)
            None, None, None,               # kernel_size, nlat_out, nlon_out_local_self
            None, None,                     # nlon_in_global, pscale
            None, None,                     # lon_in_chunk_starts, lon_in_local_sizes
            None, None,                     # lon_out_chunk_starts, lon_out_local_sizes
            None, None, None,               # az_group, az_rank, az_size
            None, None,                     # groups, groupsize
        )


# ---------------------------------------------------------------------------
# DistributedDiscreteContinuousConvS2Ring
# ---------------------------------------------------------------------------

class DistributedDiscreteContinuousConvS2Ring(DiscreteContinuousConv):
    """
    Ring-exchange-based distributed DISCO convolution. Same outward
    interface as DistributedDiscreteContinuousConvS2 (no A2A version)
    but parallelizes the azimuth direction by rotating input W chunks
    through the azimuth communicator over P_az steps. Polar direction
    still uses reduce_scatter (same as the A2A variant).

    Weight is REPLICATED across both polar and azimuth — no per-rank
    sharding. Each rank's autograd-computed weight.grad must be
    all-reduced across the full layer world group (polar AND azimuth).

    Parameters
    ----------
    Same as DistributedDiscreteContinuousConvS2.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_shape: Tuple[int],
        out_shape: Tuple[int],
        kernel_shape: Union[int, Tuple[int], Tuple[int, int]],
        basis_type: Optional[str] = "piecewise linear",
        basis_norm_mode: Optional[str] = "mean",
        groups: Optional[int] = 1,
        grid_in: Optional[str] = "equiangular",
        grid_out: Optional[str] = "equiangular",
        bias: Optional[bool] = True,
        theta_cutoff: Optional[float] = None,
        optimized_kernel: Optional[bool] = True,
        fused: Optional[bool] = False,
    ):
        super().__init__(in_channels, out_channels, kernel_shape, basis_type, groups, bias, optimized_kernel)

        # Fused mode wraps ring fwd + einsum into one autograd Function so
        # the K-expanded intermediate is NOT saved across fwd/bwd; bwd
        # recomputes it via a second ring fwd loop. Trades activation
        # memory (K× plus the polar factor) for one extra ring fwd pass
        # in backward.
        self.fused = bool(fused)

        # CUDA-only path: matches the policy used for distributed attention
        # (CPU torch doesn't have working P2P send/recv for the ring loop,
        # and the per-step kernels exist only as optimized CUDA ops). The
        # _RING_KERNELS_AVAILABLE flag also covers the case where the
        # optimized library hasn't been built.
        if not torch.cuda.is_available():
            raise NotImplementedError(
                "DistributedDiscreteContinuousConvS2Ring is CUDA-only "
                "(the ring loop relies on dist.batch_isend_irecv which "
                "only works on NCCL, and the per-step kernels are CUDA "
                "ops). Use the A2A variant (DistributedDiscreteContinuousConvS2) "
                "for CPU/gloo backends."
            )
        if not _RING_KERNELS_AVAILABLE:
            raise NotImplementedError(
                "DistributedDiscreteContinuousConvS2Ring requires the "
                "_disco_s2_contraction_ring_step_optimized CUDA kernel "
                "(and its transpose variant). They are not present in this "
                "build. Use DistributedDiscreteContinuousConvS2 (A2A) until "
                "they are added."
            )

        self.nlat_in, self.nlon_in = in_shape
        self.nlat_out, self.nlon_out = out_shape

        # comms grid
        self.comm_size_polar    = polar_group_size()
        self.comm_rank_polar    = polar_group_rank()
        self.comm_size_azimuth  = azimuth_group_size()
        self.comm_rank_azimuth  = azimuth_group_rank()

        # split shapes
        self.lat_in_shapes  = compute_split_shapes(self.nlat_in,  self.comm_size_polar)
        self.lon_in_shapes  = compute_split_shapes(self.nlon_in,  self.comm_size_azimuth)
        self.lat_out_shapes = compute_split_shapes(self.nlat_out, self.comm_size_polar)
        self.lon_out_shapes = compute_split_shapes(self.nlon_out, self.comm_size_azimuth)

        # offsets into the global longitude grid for each azimuth rank
        self.lon_in_chunk_starts  = [0] + list(accumulate(self.lon_in_shapes[:-1]))
        self.lon_out_chunk_starts = [0] + list(accumulate(self.lon_out_shapes[:-1]))

        # rank-local sizes
        self.nlat_in_local   = self.lat_in_shapes[self.comm_rank_polar]
        self.nlat_out_local  = self.nlat_out
        self.nlon_in_local   = self.lon_in_shapes[self.comm_rank_azimuth]
        self.nlon_out_local  = self.lon_out_shapes[self.comm_rank_azimuth]
        self.lon_lo_in_self  = self.lon_in_chunk_starts[self.comm_rank_azimuth]
        self.lon_lo_out_self = self.lon_out_chunk_starts[self.comm_rank_azimuth]

        # theta cutoff
        if theta_cutoff is None:
            self.theta_cutoff = torch.pi / float(self.nlat_out - 1)
        else:
            self.theta_cutoff = theta_cutoff
        if self.theta_cutoff <= 0.0:
            raise ValueError("Error, theta_cutoff has to be positive.")

        # pscale used by the ring kernel: integer ratio of global input lon
        # to global output lon (same as the serial DISCO kernel's pscale).
        if self.nlon_in % self.nlon_out != 0:
            raise ValueError(
                f"nlon_in ({self.nlon_in}) must be a multiple of nlon_out "
                f"({self.nlon_out}) for the DISCO pshift to be exact."
            )
        self.pscale = self.nlon_in // self.nlon_out

        if not self.optimized_kernel:
            # The torch reference path is not wired into the ring algorithm
            # — the per-step kernel signature is CUDA-only by design.
            raise NotImplementedError(
                "DistributedDiscreteContinuousConvS2Ring currently requires "
                "optimized_kernel=True (the ring step is CUDA-only)."
            )

        # Global convolution tensor — same as the A2A variant, split along
        # polar (latitude) but NOT along azimuth. The ``wi`` shift that
        # bakes ``lon_lo_out_self`` into col_idx happens inside
        # ``_build_local_psi``, mirroring the attention class.
        idx, vals, _ = _precompute_convolution_tensor_s2(
            in_shape, out_shape, self.filter_basis,
            grid_in=grid_in, grid_out=grid_out,
            theta_cutoff=self.theta_cutoff,
            transpose_normalization=False,
            basis_norm_mode=basis_norm_mode,
            merge_quadrature=True,
        )
        idx, vals = _split_distributed_convolution_tensor_s2(idx, vals, in_shape, out_shape)

        self._build_local_psi(idx, vals)

    # -----------------------------------------------------------------------

    def _build_local_psi(self, idx: torch.Tensor, vals: torch.Tensor):
        """Apply the azimuth ``wi`` pre-shift to the polar-split psi and
        register the buffers the ring-step kernel will read.

        Mirrors ``distributed_attention.DistributedNeighborhoodAttentionS2.
        _build_local_psi``: the polar filter happened upstream (via
        ``_split_distributed_convolution_tensor_s2``); here we absorb this
        rank's ``lon_lo_out_self`` into ``col_idx`` so the kernel does not
        need it as a runtime parameter.
        """
        ker_idx = idx[0, ...].contiguous()
        row_idx = idx[1, ...].contiguous()
        col_idx = idx[2, ...].contiguous()
        vals    = vals.contiguous()

        # ---- wi pre-shift on col_idx ----
        # col_idx natively stores hi_global * nlon_in + wi_canonical.
        # We replace wi_canonical with
        #     wi_shifted = (wi_canonical + pscale * lon_lo_out_self) mod nlon_in
        # so the kernel can compute w_in_global = (wi_shifted + pscale * w_out_local) mod nlon_in
        # without seeing lon_lo_out_self at all.
        nlon_in = self.nlon_in
        if self.comm_size_azimuth > 1 and self.lon_lo_out_self != 0:
            hi_global  = col_idx // nlon_in
            wi_canon   = col_idx - hi_global * nlon_in
            wi_shifted = (wi_canon + self.pscale * self.lon_lo_out_self) % nlon_in
            col_idx    = (hi_global * nlon_in + wi_shifted).contiguous()

        # ---- preprocess: row offsets / sort permutation ----
        # preprocess_psi sorts by per-row nnz and emits roff_idx for the
        # local nlat_out range.
        roff_idx = preprocess_psi(self.kernel_size, self.nlat_out_local,
                                  ker_idx, row_idx, col_idx, vals).contiguous()

        self.register_buffer("psi_ker_idx",  ker_idx,  persistent=False)
        self.register_buffer("psi_row_idx",  row_idx,  persistent=False)
        self.register_buffer("psi_col_idx",  col_idx,  persistent=False)
        self.register_buffer("psi_vals",     vals,     persistent=False)
        self.register_buffer("psi_roff_idx", roff_idx, persistent=False)

    @property
    def psi_idx(self):
        return torch.stack([self.psi_ker_idx, self.psi_row_idx, self.psi_col_idx], dim=0).contiguous()

    def extra_repr(self):
        return (f"in_shape={(self.nlat_in, self.nlon_in)}, out_shape={(self.nlat_out, self.nlon_out)}, "
                f"in_chans={self.groupsize * self.groups}, out_chans={self.weight.shape[0]}, "
                f"filter_basis={self.filter_basis}, kernel_shape={self.kernel_shape}, "
                f"theta_cutoff={self.theta_cutoff}, groups={self.groups} [ring]")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H_in_local, W_in_local) on each rank

        if self.fused:
            # Fused ring fwd + einsum, saving x (not the K-expanded
            # intermediate). bwd recomputes y_acc for the grad_w path.
            # Works for any comm_size_azimuth (loop runs az_size steps).
            out = _RingDiscoConvFusedFn.apply(
                x, self.weight,
                self.psi_roff_idx, self.psi_ker_idx, self.psi_row_idx, self.psi_col_idx, self.psi_vals,
                self.kernel_size, self.nlat_out_local, self.nlon_out_local,
                self.nlon_in, self.pscale,
                self.lon_in_chunk_starts, self.lon_in_shapes,
                self.lon_out_chunk_starts, self.lon_out_shapes,
                azimuth_group(), self.comm_rank_azimuth, self.comm_size_azimuth,
                self.groups, self.groupsize,
            )
            # out: (B, O, H_out, W_out_local) — partial in polar.
        else:
            # If azimuth is not actually distributed, fall back to a single
            # local kernel call — the ring degenerates to one step.
            if self.comm_size_azimuth == 1:
                # Equivalent to serial DISCO over local polar slice + polar RS.
                from torch_harmonics.disco.optimized.disco_optimized import _disco_s2_contraction_optimized
                x_ke = _disco_s2_contraction_optimized(
                    x, self.psi_roff_idx, self.psi_ker_idx, self.psi_row_idx, self.psi_col_idx, self.psi_vals,
                    self.kernel_size, self.nlat_out_local, self.nlon_out,
                )
            else:
                x_ke = _RingDiscoConvS2Fn.apply(
                    x,
                    self.psi_roff_idx, self.psi_ker_idx, self.psi_row_idx, self.psi_col_idx, self.psi_vals,
                    self.kernel_size, self.nlat_out_local, self.nlon_out_local,
                    self.nlon_in, self.pscale,
                    self.lon_in_chunk_starts, self.lon_in_shapes,
                    self.lon_out_chunk_starts, self.lon_out_shapes,
                    azimuth_group(), self.comm_rank_azimuth, self.comm_size_azimuth,
                )
            # x_ke: (B, C, K, H_out, W_out_local) — partial in polar.

            # Local einsum with the replicated weight, run BEFORE the polar
            # reduce_scatter. The einsum is linear over both C and K, so it
            # commutes with the polar sum. Doing it here means the polar
            # collective below operates on (B, O, H_out, W_out_local) — no
            # K factor — which shrinks the polar comm volume by C·K/O.
            B, C, K, H, W = x_ke.shape
            x_ke = x_ke.reshape(B, self.groups, self.groupsize, K, H, W)
            out = torch.einsum(
                "bgckxy,gock->bgoxy",
                x_ke,
                self.weight.reshape(self.groups, -1, self.weight.shape[1], self.weight.shape[2]),
            ).contiguous()
            out = out.reshape(B, -1, H, W)
            # out: (B, O, H_out, W_out_local) — still partial in polar.

        # Polar reduce_scatter on H — operates on the post-einsum tensor
        # in both paths.
        out = reduce_from_scatter_to_polar_region(out, -2)
        # out: (B, O, H_out_local, W_out_local)

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)

        return out
