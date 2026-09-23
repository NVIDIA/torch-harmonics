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

from .. import attention_kernels
from .._attention_utils import _setup_context_attention_backward, _setup_context_attention_ragged_backward


def _op_is_declared(name: str) -> bool:
    """
    Whether the extension declared a schema for ``attention_kernels::<name>``.

    A build can have the kernels compiled and still not expose a given operator: the
    ragged pair, for one, was registering CUDA implementations before any schema
    declared them. Tests that exercise an op need to gate on the op, not merely on
    ``optimized_kernels_is_available``, or they fail with an attribute error on a build
    where the rest of the library is fine.
    """
    try:
        return hasattr(torch.ops.attention_kernels, name)
    except (AttributeError, RuntimeError):
        # the namespace itself is absent, i.e. the extension never loaded
        return False


# define NA op for CUDA
if optimized_kernels_is_available():
    # raw forward fake
    @torch.library.register_fake("attention_kernels::forward_regular")
    def _(
        kw: torch.Tensor,
        vw: torch.Tensor,
        qw: torch.Tensor,
        ring_weights: torch.Tensor,
        col_idx: torch.Tensor,
        row_off: torch.Tensor,
        seg: torch.Tensor,
        seg_off: torch.Tensor,
        num_heads: int,
        nlon_in: int,
        nlat_out: int,
        nlon_out: int,
    ) -> torch.Tensor:
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
        col_idx: torch.Tensor,
        row_off: torch.Tensor,
        seg: torch.Tensor,
        seg_off: torch.Tensor,
        num_heads: int,
        nlon_in: int,
        nlat_out: int,
        nlon_out: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        col_idx: torch.Tensor,
        row_off: torch.Tensor,
        row_idx: torch.Tensor,
        nlon_in: int,
        pscale: int,
        lon_lo_kx: int,
        lat_halo_start: int,
        nlat_out: int,
        nlon_out: int,
        n_long_rows: int,
        max_row_len: int,
        mid_row_len: int,
    ) -> None:
        pass

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
        col_idx: torch.Tensor,
        row_off: torch.Tensor,
        row_idx: torch.Tensor,
        nlon_in: int,
        pscale: int,
        lon_lo_kx: int,
        lat_halo_start: int,
        nlat_out: int,
        nlon_out: int,
        n_long_rows: int,
        max_row_len: int,
        mid_row_len: int,
    ) -> None:
        pass

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
        col_idx: torch.Tensor,
        row_off: torch.Tensor,
        row_idx: torch.Tensor,
        nlon_in: int,
        pscale: int,
        lon_lo_kx: int,
        lat_halo_start: int,
        nlat_out: int,
        nlon_out: int,
        n_long_rows: int,
        max_row_len: int,
        mid_row_len: int,
    ) -> None:
        pass

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
        col_idx: torch.Tensor,
        row_off: torch.Tensor,
        nlon_in: int,
        nlon_out_global: int,
        pscale_out: int,
        lon_lo_kx: int,
        lat_halo_start: int,
        nlat_out: int,
        nlon_out: int,
    ) -> None:
        pass

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
        col_idx: torch.Tensor,
        row_off: torch.Tensor,
        nlon_in: int,
        nlon_out_global: int,
        pscale_out: int,
        lon_lo_kx: int,
        lat_halo_start: int,
        nlat_out: int,
        nlon_out: int,
    ) -> None:
        pass

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
        col_idx: torch.Tensor,
        row_off: torch.Tensor,
        nlon_in: int,
        nlon_out_global: int,
        pscale_out: int,
        lon_lo_kx: int,
        lat_halo_start: int,
        nlat_out: int,
        nlon_out: int,
    ) -> None:
        pass

    # forward
    @torch.library.custom_op("attention_kernels::_neighborhood_s2_attention_regular_optimized", mutates_args=())
    def _neighborhood_s2_attention_regular_optimized(
        kw: torch.Tensor,
        vw: torch.Tensor,
        qw: torch.Tensor,
        ring_weights: torch.Tensor,
        col_idx: torch.Tensor,
        row_off: torch.Tensor,
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
        # The native dtype is kept: the CUDA op handles fp16/bf16/fp32 natively
        # (Tier B storage refactor), widening to fp32 only at the load site.
        kw = kw.contiguous()
        vw = vw.contiguous()
        qw = qw.contiguous()

        return attention_kernels.forward_regular.default(kw, vw, qw, ring_weights, col_idx, row_off, seg, seg_off, nh, nlon_in, nlat_out, nlon_out)

    @torch.library.register_fake("attention_kernels::_neighborhood_s2_attention_regular_optimized")
    def _(
        kw: torch.Tensor,
        vw: torch.Tensor,
        qw: torch.Tensor,
        ring_weights: torch.Tensor,
        col_idx: torch.Tensor,
        row_off: torch.Tensor,
        seg: torch.Tensor,
        seg_off: torch.Tensor,
        nh: int,
        nlon_in: int,
        nlat_out: int,
        nlon_out: int,
    ) -> torch.Tensor:
        out_shape = (kw.shape[0], nlat_out, nlon_out, vw.shape[3])
        return torch.empty(out_shape, dtype=kw.dtype, device=kw.device)


def _neighborhood_s2_attention_regular_bwd_optimized(ctx, grad_output):
    col_idx, row_off, seg, seg_off, ring_weights, kw, vw, qw = ctx.saved_tensors
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

    dkw, dvw, dqw = attention_kernels.backward_regular.default(kw, vw, qw, grad_output, ring_weights, col_idx, row_off, seg, seg_off, nh, nlon_in, nlat_out, nlon_out)

    # one gradient per forward input: kw, vw, qw, then None for ring_weights,
    # col_idx, row_off, seg, seg_off, nh, nlon_in, nlat_out, nlon_out
    return dkw, dvw, dqw, None, None, None, None, None, None, None, None, None


# register backward
if optimized_kernels_is_available():
    torch.library.register_autograd(
        "attention_kernels::_neighborhood_s2_attention_regular_optimized", _neighborhood_s2_attention_regular_bwd_optimized, setup_context=_setup_context_attention_backward
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
        def _(kw, vw, qw, ring_weights, col_idx, row_off, seg, seg_off, nh, nlon_in, nlat_out, nlon_out):
            cast_dtype = torch.get_autocast_dtype(device_type)
            with torch.amp.autocast(device_type, enabled=False):
                return _neighborhood_s2_attention_regular_optimized(
                    kw.to(cast_dtype), vw.to(cast_dtype), qw.to(cast_dtype), ring_weights, col_idx, row_off, seg, seg_off, nh, nlon_in, nlat_out, nlon_out
                )

        return _

    _make_autocast_impl("cuda")
    _make_autocast_impl("cpu")


# define the ragged NA op, for a grid whose rings differ in length (HEALPix, reduced
# Gaussian). CUDA only: there is no CPU ragged kernel, so a ragged grid on CPU falls
# back to the torch reference -- see optimized/kernels_cpu/ragged/README.md.
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
    # Only the CUDA key is registered here. The CPU key would be dead -- there is no CPU
    # ragged kernel, so a ragged grid on CPU never reaches this op at all.
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
