# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
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

from typing import Tuple

import torch
from attention_helpers import optimized_kernels_is_available

from torch_harmonics.utils import check

from .. import attention_kernels
from .._attention_utils import _setup_context_attention_ragged_backward, _setup_context_attention_regular_optimized_backward

# Operand checks for the fake implementations. They mirror the host-side TORCH_CHECKs in
# attention_checks.h -- devices, dtypes, ranks, sizes -- so that under torch.compile a
# malformed call fails at trace time with the same message instead of only when the graph
# runs. Strides stay on the C++ side: with dynamic shapes they are symbolic, and the fakes
# promise nothing about layout.
#
# Every check goes through `check`, i.e. torch._check, and holds at most one comparison of
# sizes: those may be symbolic, and combining them with `and`, all() or a tuple comparison
# would call bool() on a SymBool, which makes Dynamo guard -- specialize -- on it. Checks
# on plain Python values (dtypes, ranks, int arguments) are combined freely.


def _check_device(t: torch.Tensor, ref: torch.Tensor, name: str) -> None:
    check(t.device == ref.device, lambda: f"{name} must be on the same device as the activations ({ref.device}), got {t.device}")


def _check_same_dtype(t: torch.Tensor, ref: torch.Tensor, name: str, ref_name: str) -> None:
    # dtypes are plain values, so this adds no guard
    check(t.dtype == ref.dtype, lambda: f"{name} dtype ({t.dtype}) must match {ref_name} dtype ({ref.dtype})")


def _check_size(t: torch.Tensor, dim: int, expected, what: str) -> None:
    # one symbolic comparison per call, see above
    check(t.shape[dim] == expected, lambda: f"{what}: expected {expected}, got {t.shape[dim]} (shape {tuple(t.shape)})")


def _check_shape(t: torch.Tensor, shape, name: str) -> None:
    check(t.dim() == len(shape), lambda: f"{name} must have {len(shape)} dims, got shape {tuple(t.shape)}")
    for d, n in enumerate(shape):
        _check_size(t, d, n, f"{name} dim {d}")


def _check_arc_pattern(seg: torch.Tensor, seg_off: torch.Tensor, nrows) -> None:
    check(seg.dtype == torch.int32 and seg.dim() == 2, lambda: f"psi_seg must be an int32 (nsegs, 3) tensor, got {seg.dtype} of shape {tuple(seg.shape)}")
    _check_size(seg, 1, 3, "psi_seg columns")
    check(seg_off.dtype == torch.int32 and seg_off.dim() == 1, lambda: f"psi_seg_off must be an int32 vector, got {seg_off.dtype} of shape {tuple(seg_off.shape)}")
    _check_size(seg_off, 0, nrows + 1, "psi_seg_off row offsets")


def _check_heads(kx: torch.Tensor, vx: torch.Tensor, qy: torch.Tensor, num_heads: int) -> None:
    check(num_heads >= 1, lambda: f"num_heads must be positive, got {num_heads}")
    _check_same_dtype(kx, qy, "k", "q")
    _check_same_dtype(vx, qy, "v", "q")
    _check_size(qy, -1, kx.shape[-1], "q channels, which must equal k's")
    check(qy.shape[-1] % num_heads == 0, lambda: f"q/k channel count ({qy.shape[-1]}) must be divisible by num_heads ({num_heads})")
    check(vx.shape[-1] % num_heads == 0, lambda: f"v channel count ({vx.shape[-1]}) must be divisible by num_heads ({num_heads})")


def _check_output_grad(dy: torch.Tensor, kx: torch.Tensor, vx: torch.Tensor, qy: torch.Tensor) -> None:
    # dy is the output's gradient: shaped like qy, with v's channel count
    _check_device(dy, kx, "dy")
    _check_same_dtype(dy, qy, "dy", "q")
    _check_shape(dy, (*qy.shape[:-1], vx.shape[-1]), "dy")


def _check_state_buffer(t: torch.Tensor, kx: torch.Tensor, name: str, shape) -> None:
    _check_device(t, kx, name)
    check(t.dtype == torch.float32, lambda: f"{name} must be float32, got {t.dtype}")
    _check_shape(t, shape, name)


def _check_float_vector(t: torch.Tensor, name: str) -> None:
    check(t.dtype == torch.float32 and t.dim() == 1, lambda: f"{name} must be a float32 vector, got {t.dtype} of shape {tuple(t.shape)}")


def _check_regular_inputs(kx, vx, qy, ring_weights, seg, seg_off, num_heads, nlon_in, nlat_out, nlon_out) -> None:
    """The serial product-grid ops; see check_regular_attention_inputs in attention_checks.h."""
    check(kx.dim() == 4 and vx.dim() == 4 and qy.dim() == 4, lambda: f"kx, vx and qy must be 4-D, got {kx.dim()}, {vx.dim()} and {qy.dim()} dims")
    check(nlon_in > 0 and nlon_out > 0 and nlat_out > 0, lambda: f"nlon_in, nlat_out and nlon_out must be positive, got {nlon_in}, {nlat_out} and {nlon_out}")
    # before the arc rows, which depend on the direction
    check(nlon_in % nlon_out == 0 or nlon_out % nlon_in == 0, lambda: f"either nlon_in ({nlon_in}) must be an integer multiple of nlon_out ({nlon_out}), or vice versa")
    for t, name in ((vx, "vx"), (qy, "qy"), (ring_weights, "ring_weights"), (seg, "psi_seg"), (seg_off, "psi_seg_off")):
        _check_device(t, kx, name)
    _check_size(kx, 2, nlon_in, "kx longitudes (nlon_in)")
    for d in range(3):
        _check_size(vx, d, kx.shape[d], f"vx dim {d}, which must equal kx's")
    _check_size(qy, 0, kx.shape[0], "qy batch")
    _check_size(qy, 1, nlat_out, "qy latitudes (nlat_out)")
    _check_size(qy, 2, nlon_out, "qy longitudes (nlon_out)")
    _check_heads(kx, vx, qy, num_heads)
    _check_float_vector(ring_weights, "ring_weights")
    _check_size(ring_weights, 0, kx.shape[1], "ring_weights, one per input ring")
    # gather keys the arcs by output ring, scatter by input ring
    _check_arc_pattern(seg, seg_off, nlat_out if nlon_in % nlon_out == 0 else kx.shape[1])


def _check_ring_inputs(kx, vx, qy, ring_weights, seg, seg_off, num_heads, nlat_out, nlon_out, seg_rows) -> None:
    """One ring step; see check_ring_step_inputs in attention_checks.h."""
    check(kx.dim() == 4 and vx.dim() == 4 and qy.dim() == 4, lambda: f"kx, vx and qy must be 4-D, got {kx.dim()}, {vx.dim()} and {qy.dim()} dims")
    for t, name in ((vx, "vx"), (qy, "qy"), (ring_weights, "ring_weights"), (seg, "psi_seg"), (seg_off, "psi_seg_off")):
        _check_device(t, kx, name)
    for d in range(3):
        _check_size(vx, d, kx.shape[d], f"vx dim {d}, which must equal kx's")
    _check_size(qy, 0, kx.shape[0], "qy batch")
    _check_size(qy, 1, nlat_out, "qy latitudes (nlat_out)")
    _check_size(qy, 2, nlon_out, "qy longitudes (nlon_out)")
    _check_heads(kx, vx, qy, num_heads)
    _check_float_vector(ring_weights, "ring_weights")
    _check_arc_pattern(seg, seg_off, seg_rows)


def _check_ragged_inputs(kx, vx, qy, ring_weights, seg, seg_off, ring_base, ring_size, num_heads, npoints_out) -> None:
    """The ragged ops; see check_ragged_attention_inputs in attention_checks.h."""
    check(kx.dim() == 3 and vx.dim() == 3 and qy.dim() == 3, lambda: f"kx, vx and qy must be 3-D, got {kx.dim()}, {vx.dim()} and {qy.dim()} dims")
    for t, name in ((vx, "vx"), (qy, "qy"), (ring_weights, "ring_weights"), (seg, "psi_seg"), (seg_off, "psi_seg_off"), (ring_base, "ring_base"), (ring_size, "ring_size")):
        _check_device(t, kx, name)
    for d in range(2):
        _check_size(vx, d, kx.shape[d], f"vx dim {d}, which must equal kx's")
    _check_size(qy, 0, kx.shape[0], "qy batch")
    _check_size(qy, 1, npoints_out, "qy points (npoints_out)")
    _check_heads(kx, vx, qy, num_heads)
    for t, name in ((ring_base, "ring_base"), (ring_size, "ring_size")):
        check(t.dtype == torch.int64 and t.dim() == 1, lambda: f"{name} must be an int64 vector, got {t.dtype} of shape {tuple(t.shape)}")
    _check_size(ring_size, 0, ring_base.shape[0], "ring_size, one per input ring")
    _check_float_vector(ring_weights, "ring_weights")
    _check_size(ring_weights, 0, ring_base.shape[0], "ring_weights, one per input ring")
    _check_arc_pattern(seg, seg_off, npoints_out)


# fake (meta) implementations and autograd for the compiled ops, CPU and CUDA alike
if optimized_kernels_is_available():
    # raw forward fake
    @torch.library.register_fake("attention_kernels::forward_regular")
    def _(
        kw: torch.Tensor,
        vw: torch.Tensor,
        qw: torch.Tensor,
        ring_weights: torch.Tensor,
        seg: torch.Tensor,
        seg_off: torch.Tensor,
        num_heads: int,
        nlon_in: int,
        nlat_out: int,
        nlon_out: int,
    ) -> torch.Tensor:
        _check_regular_inputs(kw, vw, qw, ring_weights, seg, seg_off, num_heads, nlon_in, nlat_out, nlon_out)
        # NHWC: (B, nlat_out, nlon_out, num_heads * C_v). The channel extent is
        # taken from vw, which already carries the packed (num_heads * C_v) width.
        out_shape = (kw.shape[0], nlat_out, nlon_out, vw.shape[3])
        return torch.empty(out_shape, dtype=kw.dtype, device=kw.device)

    # raw backward fake
    @torch.library.register_fake("attention_kernels::backward_regular")
    def _(
        kw: torch.Tensor,
        vw: torch.Tensor,
        qw: torch.Tensor,
        grad_output: torch.Tensor,
        ring_weights: torch.Tensor,
        seg: torch.Tensor,
        seg_off: torch.Tensor,
        num_heads: int,
        nlon_in: int,
        nlat_out: int,
        nlon_out: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _check_regular_inputs(kw, vw, qw, ring_weights, seg, seg_off, num_heads, nlon_in, nlat_out, nlon_out)
        _check_output_grad(grad_output, kw, vw, qw)
        dk = torch.empty_like(kw)
        dv = torch.empty_like(vw)
        dq = torch.empty_like(qw)
        return dk, dv, dq

    # fake implementations for ring step ops
    @torch.library.register_fake("attention_kernels::forward_ring_step")
    def _(
        kx: torch.Tensor,
        vx: torch.Tensor,
        qy: torch.Tensor,
        y_acc: torch.Tensor,
        alpha_sum_buf: torch.Tensor,
        qdotk_max_buf: torch.Tensor,
        ring_weights: torch.Tensor,
        seg: torch.Tensor,
        seg_off: torch.Tensor,
        num_heads: int,
        nlon_in: int,
        nlon_out_global: int,
        lon_lo_kx: int,
        lat_halo_start: int,
        nlat_out: int,
        nlon_out: int,
    ) -> None:
        _check_ring_inputs(kx, vx, qy, ring_weights, seg, seg_off, num_heads, nlat_out, nlon_out, nlat_out)
        _check_state_buffer(y_acc, kx, "y_acc", (kx.shape[0], nlat_out, nlon_out, vx.shape[3]))
        _check_state_buffer(alpha_sum_buf, kx, "alpha_sum_buf", (kx.shape[0], num_heads, nlat_out, nlon_out))
        _check_state_buffer(qdotk_max_buf, kx, "qdotk_max_buf", (kx.shape[0], num_heads, nlat_out, nlon_out))

    @torch.library.register_fake("attention_kernels::backward_ring_step_pass1")
    def _(
        kx: torch.Tensor,
        vx: torch.Tensor,
        qy: torch.Tensor,
        dy: torch.Tensor,
        alpha_sum_buf: torch.Tensor,
        qdotk_max_buf: torch.Tensor,
        integral_buf: torch.Tensor,
        alpha_k_buf: torch.Tensor,
        alpha_kvw_buf: torch.Tensor,
        ring_weights: torch.Tensor,
        seg: torch.Tensor,
        seg_off: torch.Tensor,
        num_heads: int,
        nlon_in: int,
        nlon_out_global: int,
        lon_lo_kx: int,
        lat_halo_start: int,
        nlat_out: int,
        nlon_out: int,
    ) -> None:
        _check_ring_inputs(kx, vx, qy, ring_weights, seg, seg_off, num_heads, nlat_out, nlon_out, nlat_out)
        _check_output_grad(dy, kx, vx, qy)
        _check_state_buffer(alpha_sum_buf, kx, "alpha_sum_buf", (kx.shape[0], num_heads, nlat_out, nlon_out))
        _check_state_buffer(qdotk_max_buf, kx, "qdotk_max_buf", (kx.shape[0], num_heads, nlat_out, nlon_out))
        _check_state_buffer(integral_buf, kx, "integral_buf", (kx.shape[0], num_heads, nlat_out, nlon_out))
        _check_state_buffer(alpha_k_buf, kx, "alpha_k_buf", (kx.shape[0], nlat_out, nlon_out, qy.shape[3]))
        _check_state_buffer(alpha_kvw_buf, kx, "alpha_kvw_buf", (kx.shape[0], nlat_out, nlon_out, qy.shape[3]))

    @torch.library.register_fake("attention_kernels::backward_ring_step_pass2")
    def _(
        kx: torch.Tensor,
        vx: torch.Tensor,
        qy: torch.Tensor,
        dy: torch.Tensor,
        alpha_sum_buf: torch.Tensor,
        qdotk_max_buf: torch.Tensor,
        integral_norm_buf: torch.Tensor,
        dkx: torch.Tensor,
        dvx: torch.Tensor,
        ring_weights: torch.Tensor,
        seg: torch.Tensor,
        seg_off: torch.Tensor,
        num_heads: int,
        nlon_in: int,
        nlon_out_global: int,
        lon_lo_kx: int,
        lat_halo_start: int,
        nlat_out: int,
        nlon_out: int,
    ) -> None:
        _check_ring_inputs(kx, vx, qy, ring_weights, seg, seg_off, num_heads, nlat_out, nlon_out, nlat_out)
        _check_output_grad(dy, kx, vx, qy)
        _check_state_buffer(alpha_sum_buf, kx, "alpha_sum_buf", (kx.shape[0], num_heads, nlat_out, nlon_out))
        _check_state_buffer(qdotk_max_buf, kx, "qdotk_max_buf", (kx.shape[0], num_heads, nlat_out, nlon_out))
        _check_state_buffer(integral_norm_buf, kx, "integral_norm_buf", (kx.shape[0], num_heads, nlat_out, nlon_out))
        _check_state_buffer(dkx, kx, "dkx", tuple(kx.shape))
        _check_state_buffer(dvx, kx, "dvx", tuple(vx.shape))

    # fake implementations for the upsample (scatter) ring step ops
    @torch.library.register_fake("attention_kernels::forward_ring_step_upsample")
    def _(
        kx: torch.Tensor,
        vx: torch.Tensor,
        qy: torch.Tensor,
        y_acc: torch.Tensor,
        alpha_sum_buf: torch.Tensor,
        qdotk_max_buf: torch.Tensor,
        ring_weights: torch.Tensor,
        seg: torch.Tensor,
        seg_off: torch.Tensor,
        num_heads: int,
        nlon_in: int,
        nlon_out_global: int,
        lon_lo_kx: int,
        lat_halo_start: int,
        nlat_out: int,
        nlon_out: int,
    ) -> None:
        _check_ring_inputs(kx, vx, qy, ring_weights, seg, seg_off, num_heads, nlat_out, nlon_out, kx.shape[1])
        _check_state_buffer(y_acc, kx, "y_acc", (kx.shape[0], nlat_out, nlon_out, vx.shape[3]))
        _check_state_buffer(alpha_sum_buf, kx, "alpha_sum_buf", (kx.shape[0], num_heads, nlat_out, nlon_out))
        _check_state_buffer(qdotk_max_buf, kx, "qdotk_max_buf", (kx.shape[0], num_heads, nlat_out, nlon_out))

    @torch.library.register_fake("attention_kernels::backward_ring_step_upsample_pass1")
    def _(
        kx: torch.Tensor,
        vx: torch.Tensor,
        qy: torch.Tensor,
        dy: torch.Tensor,
        qdotk_max_buf: torch.Tensor,
        integral_buf: torch.Tensor,
        alpha_k_buf: torch.Tensor,
        alpha_kvw_buf: torch.Tensor,
        ring_weights: torch.Tensor,
        seg: torch.Tensor,
        seg_off: torch.Tensor,
        num_heads: int,
        nlon_in: int,
        nlon_out_global: int,
        lon_lo_kx: int,
        lat_halo_start: int,
        nlat_out: int,
        nlon_out: int,
    ) -> None:
        _check_ring_inputs(kx, vx, qy, ring_weights, seg, seg_off, num_heads, nlat_out, nlon_out, kx.shape[1])
        _check_output_grad(dy, kx, vx, qy)
        _check_state_buffer(qdotk_max_buf, kx, "qdotk_max_buf", (kx.shape[0], num_heads, nlat_out, nlon_out))
        _check_state_buffer(integral_buf, kx, "integral_buf", (kx.shape[0], num_heads, nlat_out, nlon_out))
        _check_state_buffer(alpha_k_buf, kx, "alpha_k_buf", (kx.shape[0], nlat_out, nlon_out, qy.shape[3]))
        _check_state_buffer(alpha_kvw_buf, kx, "alpha_kvw_buf", (kx.shape[0], nlat_out, nlon_out, qy.shape[3]))

    @torch.library.register_fake("attention_kernels::backward_ring_step_upsample_pass2")
    def _(
        kx: torch.Tensor,
        vx: torch.Tensor,
        qy: torch.Tensor,
        dy: torch.Tensor,
        alpha_sum_buf: torch.Tensor,
        qdotk_max_buf: torch.Tensor,
        integral_norm_buf: torch.Tensor,
        dkx: torch.Tensor,
        dvx: torch.Tensor,
        ring_weights: torch.Tensor,
        seg: torch.Tensor,
        seg_off: torch.Tensor,
        num_heads: int,
        nlon_in: int,
        nlon_out_global: int,
        lon_lo_kx: int,
        lat_halo_start: int,
        nlat_out: int,
        nlon_out: int,
    ) -> None:
        _check_ring_inputs(kx, vx, qy, ring_weights, seg, seg_off, num_heads, nlat_out, nlon_out, kx.shape[1])
        _check_output_grad(dy, kx, vx, qy)
        _check_state_buffer(alpha_sum_buf, kx, "alpha_sum_buf", (kx.shape[0], num_heads, nlat_out, nlon_out))
        _check_state_buffer(qdotk_max_buf, kx, "qdotk_max_buf", (kx.shape[0], num_heads, nlat_out, nlon_out))
        _check_state_buffer(integral_norm_buf, kx, "integral_norm_buf", (kx.shape[0], num_heads, nlat_out, nlon_out))
        _check_state_buffer(dkx, kx, "dkx", tuple(kx.shape))
        _check_state_buffer(dvx, kx, "dvx", tuple(vx.shape))

    # forward
    @torch.library.custom_op("attention_kernels::_neighborhood_s2_attention_regular_optimized", mutates_args=())
    def _neighborhood_s2_attention_regular_optimized(
        kw: torch.Tensor,
        vw: torch.Tensor,
        qw: torch.Tensor,
        ring_weights: torch.Tensor,
        seg: torch.Tensor,
        seg_off: torch.Tensor,
        nh: int,
        nlon_in: int,
        nlat_out: int,
        nlon_out: int,
    ) -> torch.Tensor:

        # NHWC in, NHWC out, heads packed along the channel dimension. There is no
        # reshape to fold heads into the batch dimension any more: the head axis is
        # interior in this layout, so folding it would materialize a copy. The
        # kernels address a head in place instead, which is why nh is passed down.
        #
        # The native dtype is kept: the CUDA op handles fp16/bf16/fp32 natively,
        # widening to fp32 only at the load site.
        kw = kw.contiguous()
        vw = vw.contiguous()
        qw = qw.contiguous()

        return attention_kernels.forward_regular.default(kw, vw, qw, ring_weights, seg, seg_off, nh, nlon_in, nlat_out, nlon_out)

    @torch.library.register_fake("attention_kernels::_neighborhood_s2_attention_regular_optimized")
    def _(
        kw: torch.Tensor,
        vw: torch.Tensor,
        qw: torch.Tensor,
        ring_weights: torch.Tensor,
        seg: torch.Tensor,
        seg_off: torch.Tensor,
        nh: int,
        nlon_in: int,
        nlat_out: int,
        nlon_out: int,
    ) -> torch.Tensor:
        _check_regular_inputs(kw, vw, qw, ring_weights, seg, seg_off, nh, nlon_in, nlat_out, nlon_out)
        out_shape = (kw.shape[0], nlat_out, nlon_out, vw.shape[3])
        return torch.empty(out_shape, dtype=kw.dtype, device=kw.device)

else:
    # Without the extension the op is not defined, but the name still exists: the attention
    # backends import it, and a failing import would make the torch reference unreachable.
    # The backend that calls it requires layer.optimized_kernel, which is False on such a
    # build, so it is never selected.
    _neighborhood_s2_attention_regular_optimized = None


def _neighborhood_s2_attention_regular_bwd_optimized(ctx, grad_output):
    seg, seg_off, ring_weights, kw, vw, qw = ctx.saved_tensors
    nh = ctx.nh
    nlon_in = ctx.nlon_in
    nlat_out = ctx.nlat_out
    nlon_out = ctx.nlon_out

    # NHWC throughout, heads packed along channels -- no folding, see the forward.
    # The CUDA backward accumulates gradients in fp32 internally and casts back at
    # the op boundary.
    kw = kw.contiguous()
    vw = vw.contiguous()
    qw = qw.contiguous()
    grad_output = grad_output.contiguous()

    dkw, dvw, dqw = attention_kernels.backward_regular.default(kw, vw, qw, grad_output, ring_weights, seg, seg_off, nh, nlon_in, nlat_out, nlon_out)

    # one gradient per forward input: kw, vw, qw, then None for ring_weights,
    # seg, seg_off, nh, nlon_in, nlat_out, nlon_out
    return dkw, dvw, dqw, None, None, None, None, None, None, None


# register backward
if optimized_kernels_is_available():
    torch.library.register_autograd(
        "attention_kernels::_neighborhood_s2_attention_regular_optimized",
        _neighborhood_s2_attention_regular_bwd_optimized,
        setup_context=_setup_context_attention_regular_optimized_backward,
    )

    # Autocast: register at the dispatcher's Autocast{CUDA,CPU} keys (not via
    # register_autocast — that API hard-codes ``cast_inputs`` and can't follow
    # the active autocast dtype). Index tensors and ring_weights pass through.
    #
    # Both keys are needed. The kernels dispatch once on q's scalar type and then
    # reinterpret every activation pointer as that type, so they require k, v and q
    # to share a dtype and check it explicitly. Autocast does not guarantee that on
    # its own: it casts some ops and not others, so a module mixing projections with
    # normalization can hand the op an fp32 q next to an fp16 v. Normalizing here is
    # what makes the requirement hold.
    def _make_autocast_impl(device_type):
        @torch.library.impl("attention_kernels::_neighborhood_s2_attention_regular_optimized", f"Autocast{device_type.upper()}")
        def _(kw, vw, qw, ring_weights, seg, seg_off, nh, nlon_in, nlat_out, nlon_out):
            cast_dtype = torch.get_autocast_dtype(device_type)
            with torch.amp.autocast(device_type, enabled=False):
                return _neighborhood_s2_attention_regular_optimized(
                    kw.to(cast_dtype), vw.to(cast_dtype), qw.to(cast_dtype), ring_weights, seg, seg_off, nh, nlon_in, nlat_out, nlon_out
                )

        return _

    _make_autocast_impl("cuda")
    _make_autocast_impl("cpu")


# define the ragged NA op, for a grid whose rings differ in length (HEALPix, reduced
# Gaussian). Registered for CPU and CUDA, both reading the arc form (kernels_cpu/ragged,
# kernels_cuda/ragged); without the compiled kernels a ragged layer falls back to the
# torch reference.
if optimized_kernels_is_available():
    # raw forward fake. The three trailing outputs are softmax bookkeeping the backward
    # consumes; they are fp32 whatever the activations are, because that is what the
    # kernel writes and what the backward reads unconditionally.
    @torch.library.register_fake("attention_kernels::forward_ragged")
    def _(
        kx: torch.Tensor,
        vx: torch.Tensor,
        qy: torch.Tensor,
        ring_weights: torch.Tensor,
        psi_seg: torch.Tensor,
        psi_seg_off: torch.Tensor,
        ring_base: torch.Tensor,
        ring_size: torch.Tensor,
        num_heads: int,
        npoints_out: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _check_ragged_inputs(kx, vx, qy, ring_weights, psi_seg, psi_seg_off, ring_base, ring_size, num_heads, npoints_out)
        # flat field: one point axis in place of the regular path's (nlat, nlon)
        out_shape = (kx.shape[0], npoints_out, vx.shape[2])
        stat_shape = (kx.shape[0], num_heads, npoints_out)
        f32 = dict(dtype=torch.float32, device=kx.device)
        # y_hi carries the fp32 output only for bfloat16, whose mantissa is too short to
        # hold what the backward re-reads; otherwise the kernels return an empty
        # placeholder. dtype is FakeTensor metadata, so this branch resolves at trace
        # time -- and getting it wrong is invisible until something reads the fake,
        # which is what opcheck's aot_dispatch utilities do.
        y_hi_shape = out_shape if kx.dtype == torch.bfloat16 else (0,)
        return (
            torch.empty(out_shape, dtype=kx.dtype, device=kx.device),
            torch.empty(y_hi_shape, **f32),
            torch.empty(stat_shape, **f32),
            torch.empty(stat_shape, **f32),
        )

    # raw backward fake
    @torch.library.register_fake("attention_kernels::backward_ragged")
    def _(
        kx: torch.Tensor,
        vx: torch.Tensor,
        qy: torch.Tensor,
        dy: torch.Tensor,
        y: torch.Tensor,
        y_hi: torch.Tensor,
        alpha_sum: torch.Tensor,
        qdotk_max: torch.Tensor,
        ring_weights: torch.Tensor,
        psi_seg: torch.Tensor,
        psi_seg_off: torch.Tensor,
        ring_base: torch.Tensor,
        ring_size: torch.Tensor,
        num_heads: int,
        npoints_out: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _check_ragged_inputs(kx, vx, qy, ring_weights, psi_seg, psi_seg_off, ring_base, ring_size, num_heads, npoints_out)
        _check_output_grad(dy, kx, vx, qy)
        _check_device(y, kx, "y")
        _check_same_dtype(y, qy, "y", "q")
        _check_shape(y, tuple(dy.shape), "y")
        # y_hi is empty, (0,), or the fp32 output: told apart by rank, a plain int, rather
        # than by numel(), which is symbolic and would make Dynamo guard on it
        if y_hi.dim() != 1:
            _check_state_buffer(y_hi, kx, "y_hi", tuple(y.shape))
        else:
            _check_size(y_hi, 0, 0, "y_hi, which must be empty or shaped like y")
        _check_state_buffer(alpha_sum, kx, "alpha_sum", (kx.shape[0], num_heads, npoints_out))
        _check_state_buffer(qdotk_max, kx, "qdotk_max", (kx.shape[0], num_heads, npoints_out))
        return (torch.empty_like(kx), torch.empty_like(vx), torch.empty_like(qy))

    # forward
    @torch.library.custom_op("attention_kernels::_neighborhood_s2_attention_ragged_optimized", mutates_args=())
    def _neighborhood_s2_attention_ragged_optimized(
        kw: torch.Tensor,
        vw: torch.Tensor,
        qw: torch.Tensor,
        ring_weights: torch.Tensor,
        psi_seg: torch.Tensor,
        psi_seg_off: torch.Tensor,
        ring_base: torch.Tensor,
        ring_size: torch.Tensor,
        nh: int,
        npoints_out: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # NHC in, NHC out, heads packed along the channel dimension -- the regular op's
        # layout with the two spatial axes collapsed to one. As there, the head axis is
        # interior and is addressed in place rather than folded into the batch, which is
        # why nh is passed down. The native dtype is kept and widened at the load site.
        #
        # Unlike the regular op this returns four tensors, not one. The regular backward
        # can rebuild its softmax statistics by re-walking a row it can address
        # arithmetically; here the neighbour list is explicit and re-walking it is the
        # expensive part, so the forward hands the statistics on instead. They carry no
        # gradient -- see _setup_context_attention_ragged_backward.
        kw = kw.contiguous()
        vw = vw.contiguous()
        qw = qw.contiguous()

        return attention_kernels.forward_ragged.default(kw, vw, qw, ring_weights, psi_seg, psi_seg_off, ring_base, ring_size, nh, npoints_out)

    @torch.library.register_fake("attention_kernels::_neighborhood_s2_attention_ragged_optimized")
    def _(
        kw: torch.Tensor,
        vw: torch.Tensor,
        qw: torch.Tensor,
        ring_weights: torch.Tensor,
        psi_seg: torch.Tensor,
        psi_seg_off: torch.Tensor,
        ring_base: torch.Tensor,
        ring_size: torch.Tensor,
        nh: int,
        npoints_out: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _check_ragged_inputs(kw, vw, qw, ring_weights, psi_seg, psi_seg_off, ring_base, ring_size, nh, npoints_out)
        out_shape = (kw.shape[0], npoints_out, vw.shape[2])
        stat_shape = (kw.shape[0], nh, npoints_out)
        f32 = dict(dtype=torch.float32, device=kw.device)
        y_hi_shape = out_shape if kw.dtype == torch.bfloat16 else (0,)
        return (
            torch.empty(out_shape, dtype=kw.dtype, device=kw.device),
            torch.empty(y_hi_shape, **f32),
            torch.empty(stat_shape, **f32),
            torch.empty(stat_shape, **f32),
        )

else:
    # Without the extension the op is not defined, but the name still exists: the attention
    # backends import it, and a failing import would make the torch reference unreachable.
    # The backend that calls it requires layer.optimized_kernel, which is False on such a
    # build, so it is never selected.
    _neighborhood_s2_attention_ragged_optimized = None


def _neighborhood_s2_attention_ragged_bwd_optimized(ctx, grad_output, grad_y_hi, grad_alpha_sum, grad_qdotk_max):
    """
    One incoming gradient per forward output. Only the first is used: the other three
    are marked non-differentiable in setup_context, so autograd never routes anything
    through them and they arrive as None.
    """
    psi_seg, psi_seg_off, ring_base, ring_size, ring_weights, kw, vw, qw, y, y_hi, alpha_sum, qdotk_max = ctx.saved_tensors
    nh = ctx.nh
    npoints_out = ctx.npoints_out

    kw = kw.contiguous()
    vw = vw.contiguous()
    qw = qw.contiguous()
    grad_output = grad_output.contiguous()

    dkw, dvw, dqw = attention_kernels.backward_ragged.default(
        kw, vw, qw, grad_output, y, y_hi, alpha_sum, qdotk_max, ring_weights, psi_seg, psi_seg_off, ring_base, ring_size, nh, npoints_out
    )

    # one gradient per forward input: kw, vw, qw, then None for ring_weights, psi_seg,
    # psi_seg_off, ring_base, ring_size, nh, npoints_out
    return dkw, dvw, dqw, None, None, None, None, None, None, None


# register backward
if optimized_kernels_is_available():
    torch.library.register_autograd(
        "attention_kernels::_neighborhood_s2_attention_ragged_optimized",
        _neighborhood_s2_attention_ragged_bwd_optimized,
        setup_context=_setup_context_attention_ragged_backward,
    )

    # Autocast, for the same reason as the regular op: the kernel dispatches once on q's
    # scalar type and then reinterprets every activation pointer as that type, so k, v
    # and q must agree. Autocast does not guarantee that on its own. Index tensors and
    # ring_weights pass through untouched.
    #
    # Registered for both keys, as the regular op is: the CPU ragged kernel requires the
    # three dtypes to agree just as the CUDA one does, and RaggedOptimizedBackend selects
    # it on CPU.
    def _make_ragged_autocast_impl(device_type):
        @torch.library.impl("attention_kernels::_neighborhood_s2_attention_ragged_optimized", f"Autocast{device_type.upper()}")
        def _(kw, vw, qw, ring_weights, psi_seg, psi_seg_off, ring_base, ring_size, nh, npoints_out):
            cast_dtype = torch.get_autocast_dtype(device_type)
            with torch.amp.autocast(device_type, enabled=False):
                return _neighborhood_s2_attention_ragged_optimized(
                    kw.to(cast_dtype), vw.to(cast_dtype), qw.to(cast_dtype), ring_weights, psi_seg, psi_seg_off, ring_base, ring_size, nh, npoints_out
                )

        return _

    _make_ragged_autocast_impl("cuda")
    _make_ragged_autocast_impl("cpu")
