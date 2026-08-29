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
from .._attention_utils import _setup_context_attention_backward

# define NA op for CUDA
if optimized_kernels_is_available():
    # raw forward fake
    @torch.library.register_fake("attention_kernels::forward")
    def _(
        kw: torch.Tensor,
        vw: torch.Tensor,
        qw: torch.Tensor,
        quad_weights: torch.Tensor,
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
    @torch.library.register_fake("attention_kernels::backward")
    def _(
        kw: torch.Tensor,
        vw: torch.Tensor,
        qw: torch.Tensor,
        grad_output: torch.Tensor,
        quad_weights: torch.Tensor,
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
        quad_weights: torch.Tensor,
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
        quad_weights: torch.Tensor,
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
        quad_weights: torch.Tensor,
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
        quad_weights: torch.Tensor,
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
        quad_weights: torch.Tensor,
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
        quad_weights: torch.Tensor,
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
    @torch.library.custom_op("attention_kernels::_neighborhood_s2_attention_optimized", mutates_args=())
    def _neighborhood_s2_attention_optimized(
        kw: torch.Tensor,
        vw: torch.Tensor,
        qw: torch.Tensor,
        quad_weights: torch.Tensor,
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

        return attention_kernels.forward.default(kw, vw, qw, quad_weights, col_idx, row_off, seg, seg_off, nh, nlon_in, nlat_out, nlon_out)

    @torch.library.register_fake("attention_kernels::_neighborhood_s2_attention_optimized")
    def _(
        kw: torch.Tensor,
        vw: torch.Tensor,
        qw: torch.Tensor,
        quad_weights: torch.Tensor,
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


def _neighborhood_s2_attention_bwd_optimized(ctx, grad_output):
    col_idx, row_off, seg, seg_off, quad_weights, kw, vw, qw = ctx.saved_tensors
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

    dkw, dvw, dqw = attention_kernels.backward.default(kw, vw, qw, grad_output, quad_weights, col_idx, row_off, seg, seg_off, nh, nlon_in, nlat_out, nlon_out)

    # one gradient per forward input: kw, vw, qw, then None for quad_weights,
    # col_idx, row_off, seg, seg_off, nh, nlon_in, nlat_out, nlon_out
    return dkw, dvw, dqw, None, None, None, None, None, None, None, None, None


# register backward
if optimized_kernels_is_available():
    torch.library.register_autograd(
        "attention_kernels::_neighborhood_s2_attention_optimized", _neighborhood_s2_attention_bwd_optimized, setup_context=_setup_context_attention_backward
    )

    # Autocast: register at the dispatcher's Autocast{CUDA,CPU} keys (not via
    # register_autocast — that API hard-codes ``cast_inputs`` and can't follow
    # the active autocast dtype). Index tensors and quad_weights pass through.
    #
    # Both keys are needed. The kernels dispatch once on q's scalar type and then
    # reinterpret every activation pointer as that type, so they require k, v and q
    # to share a dtype and check it explicitly. Autocast does not guarantee that on
    # its own: it casts some ops and not others, so a module mixing projections with
    # normalization can hand the op an fp32 q next to an fp16 v. Normalizing here is
    # what makes the requirement hold.
    def _make_autocast_impl(device_type):
        @torch.library.impl("attention_kernels::_neighborhood_s2_attention_optimized", f"Autocast{device_type.upper()}")
        def _(kw, vw, qw, quad_weights, col_idx, row_off, seg, seg_off, nh, nlon_in, nlat_out, nlon_out):
            cast_dtype = torch.get_autocast_dtype(device_type)
            with torch.amp.autocast(device_type, enabled=False):
                return _neighborhood_s2_attention_optimized(
                    kw.to(cast_dtype), vw.to(cast_dtype), qw.to(cast_dtype), quad_weights, col_idx, row_off, seg, seg_off, nh, nlon_in, nlat_out, nlon_out
                )

        return _

    _make_autocast_impl("cuda")
    _make_autocast_impl("cpu")
