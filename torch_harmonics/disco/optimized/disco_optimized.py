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

import functools

import torch
from disco_helpers import (
    kpacked_sm90_kernels_is_available,
    kpacked_sm100_kernels_is_available,
    optimized_kernels_is_available,
)

from .. import disco_kernels
from .._disco_utils import _compute_dtype


@functools.lru_cache(maxsize=None)
def _kpacked_supported_on_device(device_index: int) -> bool:
    """Return True if the kpacked kernel is supported on this CUDA device.

    SM_90a (Hopper)  — WGMMA path in disco_cuda_fwd_dense_kpacked_sm90.cu.
    SM_100a (Blackwell) — tcgen05 path in disco_cuda_fwd_dense_kpacked_sm100.cu.
    """
    major, _ = torch.cuda.get_device_capability(device_index)
    if major == 9:
        return kpacked_sm90_kernels_is_available()
    if major == 10:
        return kpacked_sm100_kernels_is_available()
    return False


def _maybe_kpack_psi(psi_packed_idx, psi_packed_vals, psi_packed_count, n_align: int = 8):
    """Convert pack_psi_dense output [K,Ho,NBR_PAD,*] to [Ho,NBR_PAD,K_PAD] kpacked layout.

    Returns (kpacked_idx, kpacked_vals, kpacked_count, K_pad) or None if the
    per-k support sets differ across k_kern (layout mismatch).

    Inputs (from pack_psi_dense):
        psi_packed_idx   : [K, Ho, NBR_PAD, 2]   int64
        psi_packed_vals  : [K, Ho, NBR_PAD]      fp32
        psi_packed_count : [K, Ho]               int64
    Outputs:
        kpacked_idx      : [Ho, NBR_PAD, 2]      int64   (== psi_packed_idx[0])
        kpacked_vals     : [Ho, NBR_PAD, K_pad]  fp32    (permute(1,2,0), zero-padded)
        kpacked_count    : [Ho]                  int64   (== psi_packed_count[0])
        K_pad            : int  (K rounded up to next multiple of n_align)
    """
    K = int(psi_packed_count.shape[0])
    K_pad = ((K + n_align - 1) // n_align) * n_align

    if psi_packed_count.shape[0] <= 1:
        kpacked_idx = psi_packed_idx[0].contiguous()
        kpacked_count = psi_packed_count[0].contiguous()
        Ho = psi_packed_vals.shape[1]
        NBR_PAD = psi_packed_vals.shape[2]
        kpacked_vals = torch.zeros(Ho, NBR_PAD, K_pad, dtype=psi_packed_vals.dtype, device=psi_packed_vals.device)
        kpacked_vals[:, :, :K] = psi_packed_vals.permute(1, 2, 0)
        return kpacked_idx, kpacked_vals.contiguous(), kpacked_count, K_pad

    # Verify that all k have the same support indices (required for the
    # K-packed layout to be valid: one idx/count per ho shared across all k).
    if not torch.equal(psi_packed_count, psi_packed_count[0:1].expand_as(psi_packed_count)):
        return None
    if not torch.equal(psi_packed_idx, psi_packed_idx[0:1].expand_as(psi_packed_idx)):
        return None

    kpacked_idx = psi_packed_idx[0].contiguous()
    kpacked_count = psi_packed_count[0].contiguous()

    vals_perm = psi_packed_vals.permute(1, 2, 0)  # [Ho, NBR_PAD, K]
    if K_pad == K:
        kpacked_vals = vals_perm.contiguous()
    else:
        Ho = psi_packed_vals.shape[1]
        NBR_PAD = psi_packed_vals.shape[2]
        kpacked_vals = torch.zeros(Ho, NBR_PAD, K_pad, dtype=psi_packed_vals.dtype, device=psi_packed_vals.device)
        kpacked_vals[:, :, :K] = vals_perm
        kpacked_vals = kpacked_vals.contiguous()

    return kpacked_idx, kpacked_vals, kpacked_count, K_pad


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
        # keep the activation native (storage dtype) so fp16/bf16 reach the kernel,
        # which accumulates in fp32 internally; vals stays compute type (fp32 for
        # half), matching the kernel's val.data_ptr<compute_t>(). fp32/fp64 unchanged.
        inp = inp.contiguous()
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
        # keep the activation native (storage dtype) so fp16/bf16 reach the kernel,
        # which accumulates in fp32 internally; vals stays compute type (fp32 for
        # half), matching the kernel's val.data_ptr<compute_t>(). fp32/fp64 unchanged.
        inp = inp.contiguous()
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
        # keep grad native (storage dtype); kernel accumulates in fp32. vals stays
        # compute type (fp32 for half), matching val.data_ptr<compute_t>().
        grad_output = grad_output.contiguous()
        vals = vals.to(cdtype)
        grad_input = disco_kernels.backward.default(grad_output, roff_idx, ker_idx, row_idx, col_idx, vals, ctx.kernel_size, ctx.nlat_in, ctx.nlon_in)
        grad_input = grad_input.to(gtype)
    else:
        grad_input = None

    return grad_input, None, None, None, None, None, None, None, None


if optimized_kernels_is_available():
    torch.library.register_autograd("disco_kernels::_disco_s2_contraction_optimized", _disco_s2_contraction_bwd_optimized, setup_context=_setup_context_conv_backward)

    # Autocast: register at the dispatcher's AutocastCUDA key. We use
    # torch.library.impl (not register_autocast) because register_autocast's
    # ``cast_inputs`` hard-codes a single dtype and can't follow the active
    # autocast dtype. The `autocast(enabled=False)` guard on the inner call
    # excludes the AutocastCUDA key from the dispatch set so the inner call
    # routes to the regular CUDA kernel (the op body) instead of recursing
    # back into this autocast kernel.
    @torch.library.impl("disco_kernels::_disco_s2_contraction_optimized", "AutocastCUDA")
    def _(inp, roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out):
        cast_dtype = torch.get_autocast_dtype("cuda")
        with torch.amp.autocast("cuda", enabled=False):
            return _disco_s2_contraction_optimized(inp.to(cast_dtype), roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out)


# Transpose convolution related
def _disco_s2_transpose_contraction_bwd_optimized(ctx, grad_output):
    roff_idx, ker_idx, row_idx, col_idx, vals = ctx.saved_tensors

    if ctx.needs_input_grad[0]:
        gtype = grad_output.dtype
        cdtype = _compute_dtype(gtype)
        # keep grad native (storage dtype); kernel accumulates in fp32. vals stays
        # compute type (fp32 for half), matching val.data_ptr<compute_t>().
        grad_output = grad_output.contiguous()
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

    @torch.library.impl("disco_kernels::_disco_s2_transpose_contraction_optimized", "AutocastCUDA")
    def _(inp, roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out):
        cast_dtype = torch.get_autocast_dtype("cuda")
        with torch.amp.autocast("cuda", enabled=False):
            return _disco_s2_transpose_contraction_optimized(inp.to(cast_dtype), roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out)


# Fused convolution + weight contraction.
# Avoids storing the K-expanded intermediate (B, C, K, H, W) in the autograd graph
# by recomputing the contraction during backward instead.
if optimized_kernels_is_available():

    @torch.library.custom_op("disco_kernels::_disco_s2_fused_conv_optimized", mutates_args=())
    def _disco_s2_fused_conv_optimized(
        inp: torch.Tensor,
        weight: torch.Tensor,
        roff_idx: torch.Tensor,
        ker_idx: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        vals: torch.Tensor,
        kernel_size: int,
        nlat_out: int,
        nlon_out: int,
        groups: int,
        groupsize: int,
    ) -> torch.Tensor:

        # sparse contraction: (B, C, H_in, W_in) -> (B, C, K, H_out, W_out)
        itype = inp.dtype
        cdtype = _compute_dtype(itype)
        x_expanded = disco_kernels.forward.default(inp.contiguous(), roff_idx, ker_idx, row_idx, col_idx, vals.to(cdtype), kernel_size, nlat_out, nlon_out)
        x_expanded = x_expanded.to(itype)

        # weight contraction: (B, G, Cg, K, H, W) x (G, Og, Cg, K) -> (B, O, H, W)
        B, C, K, H, W = x_expanded.shape
        x_expanded = x_expanded.reshape(B, groups, groupsize, K, H, W)
        out = torch.einsum("bgckxy,gock->bgoxy", x_expanded, weight.to(itype)).contiguous()
        out = out.reshape(B, groups * weight.shape[1], H, W)
        return out

    @torch.library.register_fake("disco_kernels::_disco_s2_fused_conv_optimized")
    def _(
        inp: torch.Tensor,
        weight: torch.Tensor,
        roff_idx: torch.Tensor,
        ker_idx: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        vals: torch.Tensor,
        kernel_size: int,
        nlat_out: int,
        nlon_out: int,
        groups: int,
        groupsize: int,
    ) -> torch.Tensor:
        out_channels = groups * weight.shape[1]
        return torch.empty(inp.shape[0], out_channels, nlat_out, nlon_out, dtype=inp.dtype, device=inp.device)


def _setup_context_fused_conv_backward(ctx, inputs, output):
    inp, weight, roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out, groups, groupsize = inputs
    ctx.save_for_backward(inp, weight, roff_idx, ker_idx, row_idx, col_idx, vals)
    ctx.kernel_size = kernel_size
    ctx.nlat_out = nlat_out
    ctx.nlon_out = nlon_out
    ctx.groups = groups
    ctx.groupsize = groupsize


def _disco_s2_fused_conv_bwd_optimized(ctx, grad_output):
    inp, weight, roff_idx, ker_idx, row_idx, col_idx, vals = ctx.saved_tensors

    itype = grad_output.dtype
    cdtype = _compute_dtype(itype)
    vals_c = vals.to(cdtype)

    K = ctx.kernel_size
    G, Cg = ctx.groups, ctx.groupsize
    H, W = ctx.nlat_out, ctx.nlon_out
    Og = weight.shape[1]
    B = grad_output.shape[0]
    grad_output_r = grad_output.reshape(B, G, Og, H, W)

    grad_inp = None
    grad_weight = None

    if ctx.needs_input_grad[0]:
        # einsum backward: expand grad into K-space
        # (B, G, Og, H, W) x (G, Og, Cg, K) -> (B, G, Cg, K, H, W)
        grad_x_expanded = torch.einsum("bgoxy,gock->bgckxy", grad_output_r, weight.to(itype))
        grad_x_expanded = grad_x_expanded.reshape(B, G * Cg, K, H, W).contiguous()

        # transpose contraction back to input space
        grad_inp = disco_kernels.backward.default(grad_x_expanded.contiguous(), roff_idx, ker_idx, row_idx, col_idx, vals_c, K, inp.shape[-2], inp.shape[-1])
        grad_inp = grad_inp.to(itype)

    if ctx.needs_input_grad[1]:
        # recompute x_expanded from inp (the trade: one extra forward pass)
        x_expanded = disco_kernels.forward.default(inp.contiguous(), roff_idx, ker_idx, row_idx, col_idx, vals_c, K, H, W)
        x_expanded = x_expanded.to(itype).reshape(B, G, Cg, K, H, W)

        # weight gradient: (B, G, Og, H, W) x (B, G, Cg, K, H, W) -> (G, Og, Cg, K)
        grad_weight = torch.einsum("bgoxy,bgckxy->gock", grad_output_r, x_expanded)

    return (grad_inp, grad_weight, None, None, None, None, None, None, None, None, None, None)


if optimized_kernels_is_available():
    torch.library.register_autograd(
        "disco_kernels::_disco_s2_fused_conv_optimized",
        _disco_s2_fused_conv_bwd_optimized,
        setup_context=_setup_context_fused_conv_backward,
    )

    @torch.library.impl("disco_kernels::_disco_s2_fused_conv_optimized", "AutocastCUDA")
    def _(inp, weight, roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out, groups, groupsize):
        cast_dtype = torch.get_autocast_dtype("cuda")
        with torch.amp.autocast("cuda", enabled=False):
            return _disco_s2_fused_conv_optimized(
                inp.to(cast_dtype), weight.to(cast_dtype), roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out, groups, groupsize
            )


# ---------------------------------------------------------------------------
# K-packed dense ops (WGMMA forward + CSR backward)
# ---------------------------------------------------------------------------
# Forward uses the WGMMA kpacked kernel (disco_kernels::forward_kpacked).
# Backward uses the CSR kernel (disco_kernels::backward) — the gather
# formulation is input-pixel-parallel with no cross-CTA atomics, which is
# the correct algorithm for overlapping support sets (same reason cuDNN uses
# implicit GEMM instead of col2im scatter for strided convolutions).
# ---------------------------------------------------------------------------

if optimized_kernels_is_available():

    @torch.library.register_fake("disco_kernels::forward_kpacked")
    def _(inp: torch.Tensor, pack_idx: torch.Tensor, pack_val: torch.Tensor, pack_count: torch.Tensor, kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        out_shape = (inp.shape[0], inp.shape[1], kernel_size, nlat_out, nlon_out)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    @torch.library.impl("disco_kernels::forward_kpacked", "AutocastCUDA")
    def _(inp, pack_idx, pack_val, pack_count, kernel_size, nlat_out, nlon_out):
        cast_dtype = torch.get_autocast_dtype("cuda")
        with torch.amp.autocast("cuda", enabled=False):
            return disco_kernels.forward_kpacked.default(inp.to(cast_dtype), pack_idx, pack_val, pack_count, kernel_size, nlat_out, nlon_out)


class _DiscoKpackedFn(torch.autograd.Function):
    """WGMMA forward + CSR backward for the K-packed dense contraction."""

    @staticmethod
    def forward(ctx, inp, pack_idx, pack_val, pack_count, roff_idx, ker_idx, row_idx, col_idx, csr_vals, kernel_size, nlat_out, nlon_out):
        ctx.save_for_backward(roff_idx, ker_idx, row_idx, col_idx, csr_vals)
        ctx.kernel_size = kernel_size
        ctx.nlat_in = inp.shape[-2]
        ctx.nlon_in = inp.shape[-1]
        return disco_kernels.forward_kpacked.default(inp.contiguous(), pack_idx, pack_val, pack_count, kernel_size, nlat_out, nlon_out)

    @staticmethod
    def backward(ctx, grad_output):
        roff_idx, ker_idx, row_idx, col_idx, csr_vals = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            gtype = grad_output.dtype
            cdtype = _compute_dtype(gtype)
            grad_input = disco_kernels.backward.default(
                grad_output.contiguous(),
                roff_idx,
                ker_idx,
                row_idx,
                col_idx,
                csr_vals.to(cdtype),
                ctx.kernel_size,
                ctx.nlat_in,
                ctx.nlon_in,
            ).to(gtype)
        # inp, pack_idx, pack_val, pack_count, roff_idx, ker_idx, row_idx, col_idx,
        # csr_vals, kernel_size, nlat_out, nlon_out
        return (grad_input,) + (None,) * 11


def _disco_s2_contraction_kpacked(inp, pack_idx, pack_val, pack_count, roff_idx, ker_idx, row_idx, col_idx, csr_vals, kernel_size, nlat_out, nlon_out):
    return _DiscoKpackedFn.apply(inp, pack_idx, pack_val, pack_count, roff_idx, ker_idx, row_idx, col_idx, csr_vals, kernel_size, nlat_out, nlon_out)


class _DiscoKpackedFusedFn(torch.autograd.Function):
    """WGMMA forward + CSR backward for the fused (contraction + weight einsum) path.

    Forward: WGMMA kpacked contraction → weight einsum (no (B,C,K,H,W) intermediate saved).
    Backward: CSR forward recompute to get x_expanded for grad_weight; CSR backward for grad_inp.
    """

    @staticmethod
    def forward(ctx, inp, weight, pack_idx, pack_val, pack_count, roff_idx, ker_idx, row_idx, col_idx, csr_vals, kernel_size, nlat_out, nlon_out, groups, groupsize):
        ctx.save_for_backward(inp, weight, roff_idx, ker_idx, row_idx, col_idx, csr_vals)
        ctx.kernel_size = kernel_size
        ctx.nlat_out = nlat_out
        ctx.nlon_out = nlon_out
        ctx.groups = groups
        ctx.groupsize = groupsize

        itype = inp.dtype
        x_expanded = disco_kernels.forward_kpacked.default(inp.contiguous(), pack_idx, pack_val, pack_count, kernel_size, nlat_out, nlon_out)
        B, C, K, H, W = x_expanded.shape
        x_expanded = x_expanded.reshape(B, groups, groupsize, K, H, W)
        out = torch.einsum("bgckxy,gock->bgoxy", x_expanded, weight.to(itype)).contiguous()
        return out.reshape(B, groups * weight.shape[1], H, W)

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight, roff_idx, ker_idx, row_idx, col_idx, csr_vals = ctx.saved_tensors

        itype = grad_output.dtype
        cdtype = _compute_dtype(itype)
        vals_c = csr_vals.to(cdtype)

        K = ctx.kernel_size
        G = ctx.groups
        Cg = ctx.groupsize
        H, W = ctx.nlat_out, ctx.nlon_out
        Og = weight.shape[1]
        B = grad_output.shape[0]
        grad_output_r = grad_output.reshape(B, G, Og, H, W)

        grad_inp = None
        grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_x_expanded = torch.einsum("bgoxy,gock->bgckxy", grad_output_r, weight.to(itype))
            grad_x_expanded = grad_x_expanded.reshape(B, G * Cg, K, H, W).contiguous()
            grad_inp = disco_kernels.backward.default(grad_x_expanded, roff_idx, ker_idx, row_idx, col_idx, vals_c, K, inp.shape[-2], inp.shape[-1]).to(itype)

        if ctx.needs_input_grad[1]:
            x_expanded = disco_kernels.forward.default(inp.contiguous(), roff_idx, ker_idx, row_idx, col_idx, vals_c, K, H, W).to(itype).reshape(B, G, Cg, K, H, W)
            grad_weight = torch.einsum("bgoxy,bgckxy->gock", grad_output_r, x_expanded)

        # inp, weight, pack_idx, pack_val, pack_count, roff_idx, ker_idx, row_idx,
        # col_idx, csr_vals, kernel_size, nlat_out, nlon_out, groups, groupsize
        return (grad_inp, grad_weight) + (None,) * 13


def _disco_s2_fused_conv_kpacked(inp, weight, pack_idx, pack_val, pack_count, roff_idx, ker_idx, row_idx, col_idx, csr_vals, kernel_size, nlat_out, nlon_out, groups, groupsize):
    return _DiscoKpackedFusedFn.apply(
        inp, weight, pack_idx, pack_val, pack_count, roff_idx, ker_idx, row_idx, col_idx, csr_vals, kernel_size, nlat_out, nlon_out, groups, groupsize
    )
