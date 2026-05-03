# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
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

import torch
from disco_helpers import optimized_kernels_is_available
from .. import disco_kernels
from .._disco_utils import _compute_dtype

# custom kernels
if optimized_kernels_is_available():
    # raw CSR forward fake
    @torch.library.register_fake("disco_kernels::forward_csr")
    def _(inp: torch.Tensor, roff_idx: torch.Tensor, ker_idx: torch.Tensor,
          row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor,
          kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        out_shape = (inp.shape[0], inp.shape[1], kernel_size, nlat_out, nlon_out)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # raw CSR backward fake
    @torch.library.register_fake("disco_kernels::backward_csr")
    def _(inp: torch.Tensor, roff_idx: torch.Tensor, ker_idx: torch.Tensor,
          row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor,
          kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        out_shape = (inp.shape[0], inp.shape[1], nlat_out, nlon_out)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # raw dense forward fake
    @torch.library.register_fake("disco_kernels::forward_dense")
    def _(inp: torch.Tensor, pack_idx: torch.Tensor, pack_val: torch.Tensor,
          pack_count: torch.Tensor, kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        out_shape = (inp.shape[0], inp.shape[1], kernel_size, nlat_out, nlon_out)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # forward
    @torch.library.custom_op("disco_kernels::_disco_s2_contraction_optimized", mutates_args=())
    def _disco_s2_contraction_optimized(
        inp: torch.Tensor, roff_idx: torch.Tensor, ker_idx: torch.Tensor,
        row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor,
        kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        itype = inp.dtype
        cdtype = _compute_dtype(itype)
        inp = inp.to(cdtype).contiguous()
        vals = vals.to(cdtype)
        out = disco_kernels.forward_csr.default(inp, roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out)
        out = out.to(itype)
        return out

    # dense forward (consumes packed psi for fwd; saves CSR psi for bwd, which
    # routes through the existing transpose contraction kernel)
    @torch.library.custom_op("disco_kernels::_disco_s2_contraction_dense_optimized", mutates_args=())
    def _disco_s2_contraction_dense_optimized(
        inp: torch.Tensor,
        psi_packed_idx: torch.Tensor, psi_packed_vals: torch.Tensor, psi_packed_count: torch.Tensor,
        psi_roff_idx: torch.Tensor, psi_ker_idx: torch.Tensor, psi_row_idx: torch.Tensor,
        psi_col_idx: torch.Tensor, psi_vals: torch.Tensor,
        kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        itype = inp.dtype
        cdtype = _compute_dtype(itype)
        inp = inp.to(cdtype).contiguous()
        psi_packed_vals = psi_packed_vals.to(cdtype)
        out = disco_kernels.forward_dense.default(
            inp, psi_packed_idx, psi_packed_vals, psi_packed_count,
            kernel_size, nlat_out, nlon_out)
        out = out.to(itype)
        return out

    # transpose
    @torch.library.custom_op("disco_kernels::_disco_s2_transpose_contraction_optimized", mutates_args=())
    def _disco_s2_transpose_contraction_optimized(
        inp: torch.Tensor, roff_idx: torch.Tensor, ker_idx: torch.Tensor,
        row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor,
        kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        itype = inp.dtype
        cdtype = _compute_dtype(itype)
        inp = inp.to(cdtype).contiguous()
        vals = vals.to(cdtype)
        out = disco_kernels.backward_csr.default(inp, roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out)
        out = out.to(itype)
        return out

    # forward fake
    @torch.library.register_fake("disco_kernels::_disco_s2_contraction_optimized")
    def _(inp: torch.Tensor, roff_idx: torch.Tensor, ker_idx: torch.Tensor,
          row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor,
          kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        out_shape = (inp.shape[0], inp.shape[1], kernel_size, nlat_out, nlon_out)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # dense forward fake
    @torch.library.register_fake("disco_kernels::_disco_s2_contraction_dense_optimized")
    def _(inp: torch.Tensor,
          psi_packed_idx: torch.Tensor, psi_packed_vals: torch.Tensor, psi_packed_count: torch.Tensor,
          psi_roff_idx: torch.Tensor, psi_ker_idx: torch.Tensor, psi_row_idx: torch.Tensor,
          psi_col_idx: torch.Tensor, psi_vals: torch.Tensor,
          kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        out_shape = (inp.shape[0], inp.shape[1], kernel_size, nlat_out, nlon_out)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # transpose fake
    @torch.library.register_fake("disco_kernels::_disco_s2_transpose_contraction_optimized")
    def _(inp: torch.Tensor, roff_idx: torch.Tensor, ker_idx: torch.Tensor,
          row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor,
        kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        out_shape = (inp.shape[0], inp.shape[1], nlat_out, nlon_out)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

#general routines: this is the same for forward and transpose
def _setup_context_conv_backward(ctx, inputs, output):
    inp, roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out = inputs
    ctx.save_for_backward(roff_idx, ker_idx, row_idx, col_idx, vals)
    ctx.kernel_size = kernel_size
    ctx.nlat_in = inp.shape[-2]
    ctx.nlon_in = inp.shape[-1]

# convolution related
def _disco_s2_contraction_bwd_optimized(ctx, grad_output):
    roff_idx, ker_idx, row_idx, col_idx, vals = ctx.saved_tensors

    if ctx.needs_input_grad[0]:
        gtype = grad_output.dtype
        cdtype = _compute_dtype(gtype)
        grad_output = grad_output.to(cdtype).contiguous()
        vals = vals.to(cdtype)
        grad_input = disco_kernels.backward_csr.default(grad_output, roff_idx, ker_idx, row_idx, col_idx, vals,
                                                    ctx.kernel_size, ctx.nlat_in, ctx.nlon_in)
        grad_input = grad_input.to(gtype)
    else:
        grad_input = None

    return grad_input, None, None, None, None, None, None, None, None

if optimized_kernels_is_available():
    torch.library.register_autograd(
        "disco_kernels::_disco_s2_contraction_optimized", _disco_s2_contraction_bwd_optimized, setup_context=_setup_context_conv_backward)

# dense convolution forward — bwd routes through the existing CSR transpose
# contraction kernel. The autograd context only needs to retain the CSR psi
# (the packed buffers are an fwd-only optimization).
def _setup_context_conv_dense_backward(ctx, inputs, output):
    inp, _pi, _pv, _pc, roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out = inputs
    ctx.save_for_backward(roff_idx, ker_idx, row_idx, col_idx, vals)
    ctx.kernel_size = kernel_size
    ctx.nlat_in = inp.shape[-2]
    ctx.nlon_in = inp.shape[-1]

def _disco_s2_contraction_dense_bwd_optimized(ctx, grad_output):
    roff_idx, ker_idx, row_idx, col_idx, vals = ctx.saved_tensors

    if ctx.needs_input_grad[0]:
        gtype = grad_output.dtype
        cdtype = _compute_dtype(gtype)
        grad_output = grad_output.to(cdtype).contiguous()
        vals = vals.to(cdtype)
        grad_input = disco_kernels.backward_csr.default(grad_output, roff_idx, ker_idx, row_idx, col_idx, vals,
                                                    ctx.kernel_size, ctx.nlat_in, ctx.nlon_in)
        grad_input = grad_input.to(gtype)
    else:
        grad_input = None

    # 12 inputs: inp, psi_packed_idx, psi_packed_vals, psi_packed_count,
    #            psi_roff_idx, psi_ker_idx, psi_row_idx, psi_col_idx, psi_vals,
    #            kernel_size, nlat_out, nlon_out
    return grad_input, None, None, None, None, None, None, None, None, None, None, None

if optimized_kernels_is_available():
    torch.library.register_autograd(
        "disco_kernels::_disco_s2_contraction_dense_optimized",
        _disco_s2_contraction_dense_bwd_optimized,
        setup_context=_setup_context_conv_dense_backward)

# Transpose convolution related
def _disco_s2_transpose_contraction_bwd_optimized(ctx, grad_output):
    roff_idx, ker_idx, row_idx, col_idx, vals = ctx.saved_tensors

    if ctx.needs_input_grad[0]:
        gtype = grad_output.dtype
        cdtype = _compute_dtype(gtype)
        grad_output = grad_output.to(cdtype).contiguous()
        vals = vals.to(cdtype)
        grad_input = disco_kernels.forward_csr.default(grad_output, roff_idx, ker_idx, row_idx, col_idx, vals,
                                                    ctx.kernel_size, ctx.nlat_in, ctx.nlon_in)
        grad_input = grad_input.to(gtype)
    else:
        grad_input = None

    return grad_input, None, None, None, None, None, None, None, None

if optimized_kernels_is_available():
    torch.library.register_autograd(
        "disco_kernels::_disco_s2_transpose_contraction_optimized", _disco_s2_transpose_contraction_bwd_optimized, setup_context=_setup_context_conv_backward)
