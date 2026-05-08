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
        kw: torch.Tensor, vw: torch.Tensor, qw: torch.Tensor, quad_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor, nlon_in: int, nlat_out: int, nlon_out: int
    ) -> torch.Tensor:
        out_shape = (kw.shape[0], vw.shape[1], nlat_out, nlon_out)
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
        max_psi_nnz: int,
        nh: int,
        nlon_in: int,
        nlat_out: int,
        nlon_out: int,
    ) -> torch.Tensor:

        # reshape, folding num heads into batch dim
        B, _, H, W = kw.shape
        kw = kw.reshape(B * nh, -1, H, W)
        B, _, H, W = vw.shape
        vw = vw.reshape(B * nh, -1, H, W)
        B, _, H, W = qw.shape
        qw = qw.reshape(B * nh, -1, H, W)

        # convert to float32
        inp_dtype = kw.dtype
        kw = kw.to(torch.float32).contiguous()
        vw = vw.to(torch.float32).contiguous()
        qw = qw.to(torch.float32).contiguous()

        output = attention_kernels.forward.default(kw, vw, qw, quad_weights, col_idx, row_off, nlon_in, nlat_out, nlon_out)

        _, C, H, W = output.shape
        output = output.reshape(B, -1, H, W)

        # convert back precision
        output = output.to(dtype=inp_dtype)

        return output

    @torch.library.register_fake("attention_kernels::_neighborhood_s2_attention_optimized")
    def _(
        kw: torch.Tensor,
        vw: torch.Tensor,
        qw: torch.Tensor,
        quad_weights: torch.Tensor,
        col_idx: torch.Tensor,
        row_off: torch.Tensor,
        max_psi_nnz: int,
        nh: int,
        nlon_in: int,
        nlat_out: int,
        nlon_out: int,
    ) -> torch.Tensor:
        out_shape = (kw.shape[0], vw.shape[1], nlat_out, nlon_out)
        return torch.empty(out_shape, dtype=kw.dtype, device=kw.device)


def _neighborhood_s2_attention_bwd_optimized(ctx, grad_output):
    col_idx, row_off, quad_weights, kw, vw, qw = ctx.saved_tensors
    nh = ctx.nh
    nlon_in = ctx.nlon_in
    nlat_out = ctx.nlat_out
    nlon_out = ctx.nlon_out

    # reshape, folding num heads into batch dim
    B, _, H, W = kw.shape
    kw = kw.reshape(B * nh, -1, H, W)
    B, _, H, W = vw.shape
    vw = vw.reshape(B * nh, -1, H, W)
    B, _, H, W = qw.shape
    qw = qw.reshape(B * nh, -1, H, W)
    B, _, H, W = grad_output.shape
    grad_output = grad_output.reshape(B * nh, -1, H, W)

    # save type and convert to float32
    kw_dtype = kw.dtype
    vw_dtype = vw.dtype
    qw_dtype = qw.dtype

    kw = kw.to(torch.float32).contiguous()
    vw = vw.to(torch.float32).contiguous()
    qw = qw.to(torch.float32).contiguous()
    grad_output = grad_output.to(torch.float32).contiguous()

    dkw, dvw, dqw = attention_kernels.backward.default(kw, vw, qw, grad_output, quad_weights, col_idx, row_off, nlon_in, nlat_out, nlon_out)

    # reshape back to original batch dim and convert back precision
    _, _, Hk, Wk = dkw.shape
    dkw = dkw.reshape(B, -1, Hk, Wk).to(dtype=kw_dtype)
    _, _, Hv, Wv = dvw.shape
    dvw = dvw.reshape(B, -1, Hv, Wv).to(dtype=vw_dtype)
    _, _, Hq, Wq = dqw.shape
    dqw = dqw.reshape(B, -1, Hq, Wq).to(dtype=qw_dtype)

    return dkw, dvw, dqw, None, None, None, None, None, None, None, None


# register backward
if optimized_kernels_is_available():
    torch.library.register_autograd(
        "attention_kernels::_neighborhood_s2_attention_optimized", _neighborhood_s2_attention_bwd_optimized, setup_context=_setup_context_attention_backward
    )
