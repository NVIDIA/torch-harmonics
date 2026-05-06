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

    # raw dense backward fake
    @torch.library.register_fake("disco_kernels::backward_dense")
    def _(inp: torch.Tensor, pack_idx: torch.Tensor, pack_val: torch.Tensor,
          pack_count: torch.Tensor, kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        out_shape = (inp.shape[0], inp.shape[1], nlat_out, nlon_out)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # raw K-packed dense forward fake. pack_idx/val/count have a different shape
    # than the per-k_kern dense path (see _maybe_kpack_psi in convolution.py):
    #   pack_idx   : [Ho, NBR_PAD, 2]
    #   pack_val   : [Ho, NBR_PAD, K_PAD]
    #   pack_count : [Ho]
    @torch.library.register_fake("disco_kernels::forward_dense_kpacked")
    def _(inp: torch.Tensor, pack_idx: torch.Tensor, pack_val: torch.Tensor,
          pack_count: torch.Tensor, kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        out_shape = (inp.shape[0], inp.shape[1], kernel_size, nlat_out, nlon_out)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # forward
    @torch.library.custom_op("disco_kernels::_disco_s2_contraction_optimized_csr", mutates_args=())
    def _disco_s2_contraction_optimized_csr(
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

    # dense forward — consumes packed psi for both fwd and bwd. Dtype handling
    # (bf16/fp16 → fp32 for the FMA, cast back to storage at write) lives inside
    # the dense backend (CUDA: STORAGE_T/COMPUTE_T templated kernel; CPU: host
    # wrapper upcasts before the OMP loop). The wrapper just makes inp contiguous.
    @torch.library.custom_op("disco_kernels::_disco_s2_contraction_optimized_dense", mutates_args=())
    def _disco_s2_contraction_optimized_dense(
        inp: torch.Tensor,
        psi_packed_idx: torch.Tensor, psi_packed_vals: torch.Tensor, psi_packed_count: torch.Tensor,
        kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        inp = inp.contiguous()
        return disco_kernels.forward_dense.default(
            inp, psi_packed_idx, psi_packed_vals, psi_packed_count,
            kernel_size, nlat_out, nlon_out)

    # transpose
    @torch.library.custom_op("disco_kernels::_disco_s2_transpose_contraction_optimized_csr", mutates_args=())
    def _disco_s2_transpose_contraction_optimized_csr(
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

    # transpose, dense — fwd uses backward_dense, bwd uses forward_dense. Dtype
    # handling lives in the backend (see _disco_s2_contraction_optimized_dense).
    @torch.library.custom_op("disco_kernels::_disco_s2_transpose_contraction_optimized_dense", mutates_args=())
    def _disco_s2_transpose_contraction_optimized_dense(
        inp: torch.Tensor,
        psi_packed_idx: torch.Tensor, psi_packed_vals: torch.Tensor, psi_packed_count: torch.Tensor,
        kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        inp = inp.contiguous()
        return disco_kernels.backward_dense.default(
            inp, psi_packed_idx, psi_packed_vals, psi_packed_count,
            kernel_size, nlat_out, nlon_out)

    # forward fake
    @torch.library.register_fake("disco_kernels::_disco_s2_contraction_optimized_csr")
    def _(inp: torch.Tensor, roff_idx: torch.Tensor, ker_idx: torch.Tensor,
          row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor,
          kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        out_shape = (inp.shape[0], inp.shape[1], kernel_size, nlat_out, nlon_out)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # dense forward fake
    @torch.library.register_fake("disco_kernels::_disco_s2_contraction_optimized_dense")
    def _(inp: torch.Tensor,
          psi_packed_idx: torch.Tensor, psi_packed_vals: torch.Tensor, psi_packed_count: torch.Tensor,
          kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        out_shape = (inp.shape[0], inp.shape[1], kernel_size, nlat_out, nlon_out)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # transpose fake
    @torch.library.register_fake("disco_kernels::_disco_s2_transpose_contraction_optimized_csr")
    def _(inp: torch.Tensor, roff_idx: torch.Tensor, ker_idx: torch.Tensor,
          row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor,
        kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        out_shape = (inp.shape[0], inp.shape[1], nlat_out, nlon_out)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # transpose dense fake
    @torch.library.register_fake("disco_kernels::_disco_s2_transpose_contraction_optimized_dense")
    def _(inp: torch.Tensor,
          psi_packed_idx: torch.Tensor, psi_packed_vals: torch.Tensor, psi_packed_count: torch.Tensor,
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
        "disco_kernels::_disco_s2_contraction_optimized_csr", _disco_s2_contraction_bwd_optimized, setup_context=_setup_context_conv_backward)

# dense convolution forward — bwd routes through backward_dense, consuming the
# same packed psi as the fwd kernel.
def _setup_context_conv_dense_backward(ctx, inputs, output):
    inp, pack_idx, pack_val, pack_count, kernel_size, nlat_out, nlon_out = inputs
    ctx.save_for_backward(pack_idx, pack_val, pack_count)
    ctx.kernel_size = kernel_size
    ctx.nlat_in = inp.shape[-2]
    ctx.nlon_in = inp.shape[-1]

def _disco_s2_contraction_dense_bwd_optimized(ctx, grad_output):
    pack_idx, pack_val, pack_count = ctx.saved_tensors

    if ctx.needs_input_grad[0]:
        # Dtype handling (bf16/fp16 → fp32 compute, cast back) lives in the
        # backend.
        grad_input = disco_kernels.backward_dense.default(grad_output.contiguous(),
                                                         pack_idx, pack_val, pack_count,
                                                         ctx.kernel_size, ctx.nlat_in, ctx.nlon_in)
    else:
        grad_input = None

    # 7 inputs: inp, psi_packed_idx, psi_packed_vals, psi_packed_count,
    #           kernel_size, nlat_out, nlon_out
    return grad_input, None, None, None, None, None, None

if optimized_kernels_is_available():
    torch.library.register_autograd(
        "disco_kernels::_disco_s2_contraction_optimized_dense",
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
        "disco_kernels::_disco_s2_transpose_contraction_optimized_csr", _disco_s2_transpose_contraction_bwd_optimized, setup_context=_setup_context_conv_backward)

# Transpose convolution related, dense
def _disco_s2_transpose_contraction_dense_bwd_optimized(ctx, grad_output):
    pack_idx, pack_val, pack_count = ctx.saved_tensors

    if ctx.needs_input_grad[0]:
        # Dtype handling (bf16/fp16 → fp32 compute, cast back) lives in the
        # backend.
        grad_input = disco_kernels.forward_dense.default(grad_output.contiguous(),
                                                        pack_idx, pack_val, pack_count,
                                                        ctx.kernel_size, ctx.nlat_in, ctx.nlon_in)
    else:
        grad_input = None

    # 7 inputs: inp, psi_packed_idx, psi_packed_vals, psi_packed_count,
    #           kernel_size, nlat_out, nlon_out
    return grad_input, None, None, None, None, None, None

if optimized_kernels_is_available():
    torch.library.register_autograd(
        "disco_kernels::_disco_s2_transpose_contraction_optimized_dense",
        _disco_s2_transpose_contraction_dense_bwd_optimized,
        setup_context=_setup_context_conv_dense_backward)


# ---------------------------------------------------------------------------
# Autocast support — dense ops only.
# ---------------------------------------------------------------------------
#
# The dense backend's STORAGE_T/COMPUTE_T split lets the kernel run with bf16/
# fp16 storage and fp32 accumulators, so casting a fp32 input down to the
# autocast dtype is a real bandwidth win. We register Autocast{CUDA,CPU}
# kernels that cast only the data tensor (`inp`/`grad_output`) — pack_val stays
# fp32 so the host wrapper's "is pack_val already in compute_t" check skips
# the redundant upcast.
#
# CSR ops are deliberately not registered for autocast: their wrapper
# unconditionally upcasts to fp32, so an autocast hop would just add a
# fp32→bf16→fp32 round-trip with no compute benefit.
#
# Inside the autocast impl we redispatch with `autocast(enabled=False)` to
# drop the Autocast key — the call then goes straight to the CPU/CUDA impl.
def _make_autocast_first_arg(custom_op, device_type: str):
    def _impl(inp, *rest):
        cast_dtype = torch.get_autocast_dtype(device_type)
        with torch.amp.autocast(device_type=device_type, enabled=False):
            return custom_op(inp.to(cast_dtype), *rest)
    return _impl


if optimized_kernels_is_available():
    _DISCO_DENSE_AUTOCAST_OPS = (
        ("disco_kernels::_disco_s2_contraction_optimized_dense",
         _disco_s2_contraction_optimized_dense),
        ("disco_kernels::_disco_s2_transpose_contraction_optimized_dense",
         _disco_s2_transpose_contraction_optimized_dense),
    )
    for _qualname, _custom_op in _DISCO_DENSE_AUTOCAST_OPS:
        for _device_type, _key in (("cuda", "AutocastCUDA"), ("cpu", "AutocastCPU")):
            torch.library.impl(_qualname, _key)(
                _make_autocast_first_arg(_custom_op, _device_type)
            )
