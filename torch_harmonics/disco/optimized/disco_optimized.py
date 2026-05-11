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

# Both `forward` and `backward` ops are now gather-style. The
# scatter-with-atomicAdd backward has been replaced; the conv module
# materializes psi_T (transposed CSR over the bigger grid) in addition to psi,
# and the wrappers thread both through so:
#   Conv.fwd          → forward(inp, psi)            [gather, K free]
#   Conv.bwd          → backward(grad_out, psi_T)    [gather, K contracted]
#   TransposeConv.fwd → backward(inp, psi_T)         [gather, K contracted]
#   TransposeConv.bwd → forward(grad_out, psi)       [gather, K free]

import torch
from disco_helpers import optimized_kernels_is_available
from .. import disco_kernels
from .._disco_utils import _compute_dtype


# --------------------------------------------------------------------------- #
# raw op fakes
# --------------------------------------------------------------------------- #
if optimized_kernels_is_available():

    @torch.library.register_fake("disco_kernels::forward")
    def _(inp: torch.Tensor, roff_idx: torch.Tensor, ker_idx: torch.Tensor,
          row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor,
          kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        out_shape = (inp.shape[0], inp.shape[1], kernel_size, nlat_out, nlon_out)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    @torch.library.register_fake("disco_kernels::backward")
    def _(inp: torch.Tensor, roff_idx: torch.Tensor, ker_idx: torch.Tensor,
          col_idx: torch.Tensor, vals: torch.Tensor,
          kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        out_shape = (inp.shape[0], inp.shape[1], nlat_out, nlon_out)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)


# --------------------------------------------------------------------------- #
# Conv (S2): forward uses `forward(psi)`, autograd backward uses `backward(psi_T)`.
# --------------------------------------------------------------------------- #
if optimized_kernels_is_available():

    @torch.library.custom_op("disco_kernels::_disco_s2_contraction_optimized", mutates_args=())
    def _disco_s2_contraction_optimized(
        inp: torch.Tensor,
        # forward psi (gather, K free)
        roff_idx: torch.Tensor, ker_idx: torch.Tensor,
        row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor,
        # backward psi_T (gather, K contracted) — saved on ctx for autograd
        roff_idx_T: torch.Tensor, ker_idx_T: torch.Tensor,
        col_idx_T: torch.Tensor, vals_T: torch.Tensor,
        kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        itype = inp.dtype
        cdtype = _compute_dtype(itype)
        inp = inp.to(cdtype).contiguous()
        vals = vals.to(cdtype)
        out = disco_kernels.forward.default(inp, roff_idx, ker_idx, row_idx, col_idx, vals,
                                            kernel_size, nlat_out, nlon_out)
        out = out.to(itype)
        return out

    @torch.library.register_fake("disco_kernels::_disco_s2_contraction_optimized")
    def _(inp: torch.Tensor,
          roff_idx: torch.Tensor, ker_idx: torch.Tensor,
          row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor,
          roff_idx_T: torch.Tensor, ker_idx_T: torch.Tensor,
          col_idx_T: torch.Tensor, vals_T: torch.Tensor,
          kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        out_shape = (inp.shape[0], inp.shape[1], kernel_size, nlat_out, nlon_out)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)


# --------------------------------------------------------------------------- #
# TransposeConv (S2): forward uses `backward(psi_T)`, autograd backward uses `forward(psi)`.
# --------------------------------------------------------------------------- #
if optimized_kernels_is_available():

    @torch.library.custom_op("disco_kernels::_disco_s2_transpose_contraction_optimized", mutates_args=())
    def _disco_s2_transpose_contraction_optimized(
        inp: torch.Tensor,
        # forward psi (gather, K free) — saved on ctx for autograd
        roff_idx: torch.Tensor, ker_idx: torch.Tensor,
        row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor,
        # backward psi_T (gather, K contracted)
        roff_idx_T: torch.Tensor, ker_idx_T: torch.Tensor,
        col_idx_T: torch.Tensor, vals_T: torch.Tensor,
        kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        itype = inp.dtype
        cdtype = _compute_dtype(itype)
        inp = inp.to(cdtype).contiguous()
        vals_T = vals_T.to(cdtype)
        out = disco_kernels.backward.default(inp, roff_idx_T, ker_idx_T, col_idx_T, vals_T,
                                             kernel_size, nlat_out, nlon_out)
        out = out.to(itype)
        return out

    @torch.library.register_fake("disco_kernels::_disco_s2_transpose_contraction_optimized")
    def _(inp: torch.Tensor,
          roff_idx: torch.Tensor, ker_idx: torch.Tensor,
          row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor,
          roff_idx_T: torch.Tensor, ker_idx_T: torch.Tensor,
          col_idx_T: torch.Tensor, vals_T: torch.Tensor,
          kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        out_shape = (inp.shape[0], inp.shape[1], nlat_out, nlon_out)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)


# --------------------------------------------------------------------------- #
# Autograd plumbing
# --------------------------------------------------------------------------- #
def _setup_context_conv(ctx, inputs, output):
    """Conv: forward uses psi; backward uses psi_T. Save psi_T."""
    (inp,
     roff_idx, ker_idx, row_idx, col_idx, vals,
     roff_idx_T, ker_idx_T, col_idx_T, vals_T,
     kernel_size, nlat_out, nlon_out) = inputs
    ctx.save_for_backward(roff_idx_T, ker_idx_T, col_idx_T, vals_T)
    ctx.kernel_size = kernel_size
    ctx.nlat_in = inp.shape[-2]
    ctx.nlon_in = inp.shape[-1]


def _setup_context_transpose(ctx, inputs, output):
    """TransposeConv: forward uses psi_T; backward uses psi. Save psi."""
    (inp,
     roff_idx, ker_idx, row_idx, col_idx, vals,
     roff_idx_T, ker_idx_T, col_idx_T, vals_T,
     kernel_size, nlat_out, nlon_out) = inputs
    ctx.save_for_backward(roff_idx, ker_idx, row_idx, col_idx, vals)
    ctx.kernel_size = kernel_size
    ctx.nlat_in = inp.shape[-2]
    ctx.nlon_in = inp.shape[-1]


_NUM_INPUTS = 13  # inp + 5 psi tensors + 4 psi_T tensors + 3 ints
_NONE_TAIL = (None,) * (_NUM_INPUTS - 1)


def _disco_s2_contraction_bwd_optimized(ctx, grad_output):
    """Conv.backward: backward.default(grad_output, psi_T_args) — gather, no atomics."""
    roff_idx_T, ker_idx_T, col_idx_T, vals_T = ctx.saved_tensors

    if ctx.needs_input_grad[0]:
        gtype = grad_output.dtype
        cdtype = _compute_dtype(gtype)
        grad_output = grad_output.to(cdtype).contiguous()
        vals_T = vals_T.to(cdtype)
        grad_input = disco_kernels.backward.default(
            grad_output, roff_idx_T, ker_idx_T, col_idx_T, vals_T,
            ctx.kernel_size, ctx.nlat_in, ctx.nlon_in)
        grad_input = grad_input.to(gtype)
    else:
        grad_input = None

    return (grad_input,) + _NONE_TAIL


def _disco_s2_transpose_contraction_bwd_optimized(ctx, grad_output):
    """TransposeConv.backward: forward.default(grad_output, psi_args)."""
    roff_idx, ker_idx, row_idx, col_idx, vals = ctx.saved_tensors

    if ctx.needs_input_grad[0]:
        gtype = grad_output.dtype
        cdtype = _compute_dtype(gtype)
        grad_output = grad_output.to(cdtype).contiguous()
        vals = vals.to(cdtype)
        grad_input = disco_kernels.forward.default(
            grad_output, roff_idx, ker_idx, row_idx, col_idx, vals,
            ctx.kernel_size, ctx.nlat_in, ctx.nlon_in)
        grad_input = grad_input.to(gtype)
    else:
        grad_input = None

    return (grad_input,) + _NONE_TAIL


if optimized_kernels_is_available():
    torch.library.register_autograd(
        "disco_kernels::_disco_s2_contraction_optimized",
        _disco_s2_contraction_bwd_optimized,
        setup_context=_setup_context_conv)

    torch.library.register_autograd(
        "disco_kernels::_disco_s2_transpose_contraction_optimized",
        _disco_s2_transpose_contraction_bwd_optimized,
        setup_context=_setup_context_transpose)
