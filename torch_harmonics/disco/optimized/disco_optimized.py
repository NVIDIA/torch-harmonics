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
    # raw forward fake
    @torch.library.register_fake("disco_kernels::forward")
    def _(
        inp: torch.Tensor,
        roff_idx: torch.Tensor,
        ker_idx: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        vals: torch.Tensor,
        kernel_size: int,
        nlat_out: int,
        nlon_out: int,
    ) -> torch.Tensor:
        out_shape = (inp.shape[0], inp.shape[1], kernel_size, nlat_out, nlon_out)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # raw backward fake
    @torch.library.register_fake("disco_kernels::backward")
    def _(
        inp: torch.Tensor,
        roff_idx: torch.Tensor,
        ker_idx: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        vals: torch.Tensor,
        kernel_size: int,
        nlat_out: int,
        nlon_out: int,
    ) -> torch.Tensor:
        out_shape = (inp.shape[0], inp.shape[1], nlat_out, nlon_out)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # forward
    @torch.library.custom_op("disco_kernels::_disco_s2_contraction_optimized", mutates_args=())
    def _disco_s2_contraction_optimized(
        inp: torch.Tensor,
        roff_idx: torch.Tensor,
        ker_idx: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        vals: torch.Tensor,
        kernel_size: int,
        nlat_out: int,
        nlon_out: int,
    ) -> torch.Tensor:
        itype = inp.dtype
        cdtype = _compute_dtype(itype)
        inp = inp.to(cdtype).contiguous()
        vals = vals.to(cdtype)
        out = disco_kernels.forward.default(inp, roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out)
        out = out.to(itype)
        return out

    # transpose
    @torch.library.custom_op("disco_kernels::_disco_s2_transpose_contraction_optimized", mutates_args=())
    def _disco_s2_transpose_contraction_optimized(
        inp: torch.Tensor,
        roff_idx: torch.Tensor,
        ker_idx: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        vals: torch.Tensor,
        kernel_size: int,
        nlat_out: int,
        nlon_out: int,
    ) -> torch.Tensor:
        itype = inp.dtype
        cdtype = _compute_dtype(itype)
        inp = inp.to(cdtype).contiguous()
        vals = vals.to(cdtype)
        out = disco_kernels.backward.default(inp, roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out)
        out = out.to(itype)
        return out

    # forward fake
    @torch.library.register_fake("disco_kernels::_disco_s2_contraction_optimized")
    def _(
        inp: torch.Tensor,
        roff_idx: torch.Tensor,
        ker_idx: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        vals: torch.Tensor,
        kernel_size: int,
        nlat_out: int,
        nlon_out: int,
    ) -> torch.Tensor:
        out_shape = (inp.shape[0], inp.shape[1], kernel_size, nlat_out, nlon_out)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # transpose fake
    @torch.library.register_fake("disco_kernels::_disco_s2_transpose_contraction_optimized")
    def _(
        inp: torch.Tensor,
        roff_idx: torch.Tensor,
        ker_idx: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        vals: torch.Tensor,
        kernel_size: int,
        nlat_out: int,
        nlon_out: int,
    ) -> torch.Tensor:
        out_shape = (inp.shape[0], inp.shape[1], nlat_out, nlon_out)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # ring-step forward fake — schema returns None (op mutates ``out``)
    @torch.library.register_fake("disco_kernels::forward_ring_step")
    def _(
        inp: torch.Tensor,
        out: torch.Tensor,
        roff_idx: torch.Tensor,
        ker_idx: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        vals: torch.Tensor,
        kernel_size: int,
        nlat_out: int,
        nlon_out_local_self: int,
        nlon_in_global: int,
        pscale: int,
        lon_lo_src: int,
        nlon_in_local_src: int,
    ) -> None:
        return None

    # ring-step backward fake — same shape contract; mutates ``out``
    @torch.library.register_fake("disco_kernels::backward_ring_step")
    def _(
        inp: torch.Tensor,
        out: torch.Tensor,
        roff_idx: torch.Tensor,
        ker_idx: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        vals: torch.Tensor,
        kernel_size: int,
        nlat_in: int,
        nlon_in_local_self: int,
        nlon_in_global: int,
        pscale: int,
        pscale_wo_offset: int,
        lon_lo_in_self: int,
        nlon_out_local_src: int,
    ) -> None:
        return None

    # ring-step forward Python wrapper. Performs the dtype dance the
    # serial forward op does and forwards into the in-place CUDA kernel.
    # Note this is a plain function — not a custom_op — because the
    # underlying op mutates ``out`` and we want the autograd Function in
    # distributed_convolution_ring.py to drive the ring loop and zero-init.
    def _disco_s2_contraction_ring_step_optimized(
        inp: torch.Tensor,
        out: torch.Tensor,
        roff_idx: torch.Tensor,
        ker_idx: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        vals: torch.Tensor,
        kernel_size: int,
        nlat_out: int,
        nlon_out_local_self: int,
        nlon_in_global: int,
        pscale: int,
        lon_lo_src: int,
        nlon_in_local_src: int,
    ) -> None:
        itype = inp.dtype
        cdtype = _compute_dtype(itype)
        # Cast inputs/vals to the compute dtype to keep accumulation precision
        # at fp32 when inputs are fp16/bf16. The kernel uses a single STORAGE_T
        # for both ``inp`` and ``out``, so callers MUST pass ``out`` in the
        # same compute dtype (the autograd Function allocates it that way).
        inp_c = inp.to(cdtype).contiguous()
        vals_c = vals.to(cdtype)
        disco_kernels.forward_ring_step.default(
            inp_c,
            out,
            roff_idx,
            ker_idx,
            row_idx,
            col_idx,
            vals_c,
            kernel_size,
            nlat_out,
            nlon_out_local_self,
            nlon_in_global,
            pscale,
            lon_lo_src,
            nlon_in_local_src,
        )

    # ring-step transpose Python wrapper. Mirrors the forward wrapper but
    # the underlying op accumulates into a compute_t (fp32) buffer; the
    # autograd Function in distributed_convolution_ring.py handles the
    # final cast back to the input dtype.
    #
    # Unlike the forward wrapper, we DON'T upcast ``inp`` here: the kernel
    # template's storage_t/compute_t split casts STORAGE_T -> COMPUTE_T on
    # load and accumulates in fp32 regardless, so a Python-side upcast is
    # pure overhead (an extra alloc + copy per ring step under AMP).
    def _disco_s2_transpose_contraction_ring_step_optimized(
        inp: torch.Tensor,
        out: torch.Tensor,
        roff_idx: torch.Tensor,
        ker_idx: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        vals: torch.Tensor,
        kernel_size: int,
        nlat_in: int,
        nlon_in_local_self: int,
        nlon_in_global: int,
        pscale: int,
        pscale_wo_offset: int,
        lon_lo_in_self: int,
        nlon_out_local_src: int,
    ) -> None:
        # vals must be in compute_t (fp32) — kernel reads them as such.
        # ``out`` is grad_x_acc and is already fp32 (compute_dtype).
        vals_c = vals.to(_compute_dtype(inp.dtype))
        disco_kernels.backward_ring_step.default(
            inp.contiguous(),
            out,
            roff_idx,
            ker_idx,
            row_idx,
            col_idx,
            vals_c,
            kernel_size,
            nlat_in,
            nlon_in_local_self,
            nlon_in_global,
            pscale,
            pscale_wo_offset,
            lon_lo_in_self,
            nlon_out_local_src,
        )


# general routines: this is the same for forward and transpose
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
        grad_input = disco_kernels.backward.default(grad_output, roff_idx, ker_idx, row_idx, col_idx, vals, ctx.kernel_size, ctx.nlat_in, ctx.nlon_in)
        grad_input = grad_input.to(gtype)
    else:
        grad_input = None

    return grad_input, None, None, None, None, None, None, None, None


if optimized_kernels_is_available():
    torch.library.register_autograd("disco_kernels::_disco_s2_contraction_optimized", _disco_s2_contraction_bwd_optimized, setup_context=_setup_context_conv_backward)

    # Autocast: cast activation to the autocast dtype before calling the op.
    # The op body already upcasts to fp32 internally for precision and casts
    # the output back, so this just establishes the autocast boundary.
    @torch.library.register_autocast("disco_kernels::_disco_s2_contraction_optimized", "cuda", None)
    def _(inp, roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out):
        cast_dtype = torch.get_autocast_dtype("cuda")
        return _disco_s2_contraction_optimized(inp.to(cast_dtype), roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out)


# Transpose convolution related
def _disco_s2_transpose_contraction_bwd_optimized(ctx, grad_output):
    roff_idx, ker_idx, row_idx, col_idx, vals = ctx.saved_tensors

    if ctx.needs_input_grad[0]:
        gtype = grad_output.dtype
        cdtype = _compute_dtype(gtype)
        grad_output = grad_output.to(cdtype).contiguous()
        vals = vals.to(cdtype)
        grad_input = disco_kernels.forward.default(grad_output, roff_idx, ker_idx, row_idx, col_idx, vals, ctx.kernel_size, ctx.nlat_in, ctx.nlon_in)
        grad_input = grad_input.to(gtype)
    else:
        grad_input = None

    return grad_input, None, None, None, None, None, None, None, None


if optimized_kernels_is_available():
    torch.library.register_autograd(
        "disco_kernels::_disco_s2_transpose_contraction_optimized", _disco_s2_transpose_contraction_bwd_optimized, setup_context=_setup_context_conv_backward
    )

    @torch.library.register_autocast("disco_kernels::_disco_s2_transpose_contraction_optimized", "cuda", None)
    def _(inp, roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out):
        cast_dtype = torch.get_autocast_dtype("cuda")
        return _disco_s2_transpose_contraction_optimized(inp.to(cast_dtype), roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out)
