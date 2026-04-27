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

"""
Pure-PyTorch reference implementations of the neighborhood S2 attention kernels,
plus the matching ``torch.library`` custom_op + autograd registration that
exposes them under the ``attention_kernels::_neighborhood_s2_attention_torch``
operator name.

These mirror the C++/CUDA kernels in ``optimized/kernels_cpu`` and
``optimized/kernels_cuda``. Performance is intentionally not the focus — the
implementations use the qdotk_max online-softmax trick (see
https://arxiv.org/abs/1805.02867) but are written in plain PyTorch loops, which
makes them slow but easy to read and useful as a correctness reference for the
optimized paths.
"""

import torch

from .. import attention_kernels
from .._attention_utils import _setup_context_attention_backward


# =====================================================================================
# SELF / DOWNSAMPLE (gather-style) torch reference
# =====================================================================================
#
# Symmetry with DISCO's forward module:
#   psi is precomputed by calling _precompute_convolution_tensor_s2 with in_shape
#   and out_shape in their natural order, so rows of psi index the (smaller or
#   equal) output grid and cols encode (hi, wi_ref) on the (larger or equal)
#   input grid as hi * nlon_in + wi_ref. Self-attention is the degenerate case
#   nlon_in == nlon_out (pscale == 1).
#
# Translation invariance: psi is built for output (ho, wo=0).
# For wo > 0 we shift the stored input column index by pscale * wo
# where pscale = nlon_in / nlon_out.
#
# Forward is a single-pass online softmax using the qdotk_max update trick:
#   for each output (ho, wo), iterate the neighbor list once; on each step we
#   update qdotk_max, rescale alpha_sum and y_acc by exp(old_max - new_max),
#   then accumulate the new alpha and alpha * v term. Finalize y = y_acc / alpha_sum.
#   See e.g. https://arxiv.org/abs/1805.02867 and
#   https://alexdremov.me/understanding-flash-attention-writing-the-algorithm-from-scratch-in-triton/
# The backward references rebuild the softmax state on the fly (multiple passes
# over the neighbor list); reference code, not meant to be fast.
def _neighborhood_s2_attention_fwd_torch(kx: torch.Tensor, vx: torch.Tensor, qy: torch.Tensor,
                                         quad_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor,
                                         nlon_in: int, nlat_out: int, nlon_out: int) -> torch.Tensor:

    # one output lon step corresponds to pscale input lon steps; require an integer ratio
    assert nlon_in % nlon_out == 0, f"nlon_in ({nlon_in}) must be an integer multiple of nlon_out ({nlon_out})"
    pscale = nlon_in // nlon_out

    # prepare result tensor
    out_shape = (qy.shape[0], vx.shape[1], nlat_out, nlon_out)
    y = torch.zeros(out_shape, dtype=qy.dtype, device=qy.device)

    for ho in range(nlat_out):

	    # get number of nonzeros
        zstart = row_off[ho]
        zend = row_off[ho+1]

        for wo in range(nlon_out):

            alpha_sum = torch.zeros((y.shape[0],), dtype=y.dtype, device=y.device)
            qdotk_max = torch.zeros((y.shape[0],), dtype=y.dtype, device=y.device)

            for idz in range(zstart, zend):
                nz_col_idx = col_idx[idz]

                # compute input indices from psi datastructure
                hi = nz_col_idx // nlon_in
                # account for output shift and ensure positive index due to circular condition
                wi = nz_col_idx % nlon_in
                wip = (wi + pscale * wo) % nlon_in

                # compute correlation & softmax numerator
                q_ho_wo = qy[:, :, ho, wo]
                k_hi_wip = kx[:, :, hi, wip]
                qdotk = torch.sum(q_ho_wo * k_hi_wip, dim=1)

                # tmp max
                qdotk_max_tmp = torch.maximum(qdotk_max, qdotk)

                # alpha sum update
                alpha = torch.exp(qdotk - qdotk_max_tmp) * quad_weights[hi]
                alpha_sum = alpha + alpha_sum * torch.exp(qdotk_max - qdotk_max_tmp)
                # update output
                y[:,:,ho,wo] = y[:,:,ho,wo] * torch.exp(qdotk_max - qdotk_max_tmp).unsqueeze(1) + alpha[:, None] * vx[:,:,hi,wip]

                # define new max
                qdotk_max = qdotk_max_tmp

            y[:,:,ho,wo] = y[:,:,ho,wo] / alpha_sum[:, None]

    return y

# Explicit gradient w.r.t. vx: dM/dv
# provided as a reference for CUDA & other hand-written gradients
def _neighborhood_s2_attention_bwd_dv_torch(kx: torch.Tensor, vx: torch.Tensor, qy: torch.Tensor, dy: torch.Tensor,
                                            quad_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor,
                                            nlon_in: int, nlat_out: int, nlon_out: int):

    # shapes:
    # input
    # kx: B, C, Hi, Wi
    # vx: B, Cout, Hi, Wi
    # qy: B, Cout, Ho, Wo
    # quad_weights: Hi
    # output
    # dvx: B, Cout, Hi, Wi

    assert nlon_in % nlon_out == 0, f"nlon_in ({nlon_in}) must be an integer multiple of nlon_out ({nlon_out})"
    pscale = nlon_in // nlon_out

    dvx = torch.zeros_like(vx)
    batch_size = dy.shape[0]

    for ho in range(nlat_out):

        # get number of nonzeros
        zstart = row_off[ho]
        zend = row_off[ho+1]

        for wo in range(nlon_out):

            alpha_nz = torch.zeros((batch_size, zend-zstart), dtype=dy.dtype, device=dy.device)
            qdotk_nz = torch.zeros((batch_size, zend-zstart), dtype=dy.dtype, device=dy.device)
            alpha_sum = torch.zeros((batch_size,), dtype=dy.dtype, device=dy.device)
            for idz in range(zstart, zend):
                nz_col_idx = col_idx[idz]

                # compute input indices from psi datastructure
                hi = nz_col_idx // nlon_in
                # account for output shift and ensure positive index due to circular condition
                wi = nz_col_idx % nlon_in
                wip = (wi + pscale * wo) % nlon_in

                # compute correlation & softmax numerator
                q_ho_wo = qy[:, :, ho, wo]
                k_hi_wi = kx[:, :, hi, wip]
                qdotk_nz[:,idz-zstart] = torch.sum(q_ho_wo * k_hi_wi, dim=1)

            qdotk_max, _ = torch.max(qdotk_nz, dim=1)

            for idz in range(zstart, zend):
                nz_col_idx = col_idx[idz]

                # compute input indices from psi datastructure
                hi = nz_col_idx // nlon_in
                # account for output shift and ensure positive index due to circular condition
                wi = nz_col_idx % nlon_in
                wip = (wi + pscale * wo) % nlon_in
                alpha_nz[:,idz-zstart] = torch.exp(qdotk_nz[:,idz-zstart] - qdotk_max) * quad_weights[hi]
                alpha_sum[:] += alpha_nz[:,idz-zstart]

            for idz in range(zstart, zend):
                nz_col_idx = col_idx[idz]

                # compute input indices from psi datastructure
                hi = nz_col_idx // nlon_in
                # account for output shift and ensure positive index due to circular condition
                wi = nz_col_idx % nlon_in
                wip = (wi + pscale * wo) % nlon_in
                dvx[:,:,hi, wip] += (alpha_nz[:, None, idz-zstart] / alpha_sum[:, None]) * dy[:,:,ho,wo]

    return dvx


# Explicit gradient w.r.t. kx: dM/dk
# provided as a reference for CUDA & other hand-written gradients
def _neighborhood_s2_attention_bwd_dk_torch(kx: torch.Tensor, vx: torch.Tensor, qy: torch.Tensor, dy: torch.Tensor,
                                            quad_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor,
                                            nlon_in: int, nlat_out: int, nlon_out: int):

    # shapes:
    # input
    # kx: B, C, Hi, Wi
    # vx: B, Cout, Hi, Wi
    # qy: B, C, Ho, Wo
    # quad_weights: Hi
    # output
    # dkx: B, C, Hi, Wi

    assert nlon_in % nlon_out == 0, f"nlon_in ({nlon_in}) must be an integer multiple of nlon_out ({nlon_out})"
    pscale = nlon_in // nlon_out

    dkx = torch.zeros_like(kx)
    batch_size = dy.shape[0]

    for ho in range(nlat_out):

        # get number of nonzeros
        zstart = row_off[ho]
        zend = row_off[ho+1]

        for wo in range(nlon_out):

            qdotk_nz = torch.zeros((batch_size, zend-zstart), dtype=dy.dtype, device=dy.device)
            integral = torch.zeros((batch_size,), dtype=dy.dtype, device=dy.device)
            alpha = torch.zeros((batch_size, zend-zstart), dtype=dy.dtype, device=dy.device)
            alpha_sum = torch.zeros((batch_size,), dtype=dy.dtype, device=dy.device)
            for idz in range(zstart, zend):
                nz_col_idx = col_idx[idz]

                # compute input indices from psi datastructure
                hj = nz_col_idx // nlon_in
                # account for output shift and ensure positive index due to circular condition
                wj = nz_col_idx % nlon_in
                wjp = (wj + pscale * wo) % nlon_in

                # compute correlation & softmax numerator
                q_ho_wo = qy[:, :, ho, wo]
                k_hj_wjp = kx[:, :, hj, wjp]
                qdotk_nz[:,idz-zstart] = torch.sum(q_ho_wo * k_hj_wjp, dim=1)

            qdotk_max, _ = torch.max(qdotk_nz, dim=1)

            for idz in range(zstart, zend):
                nz_col_idx = col_idx[idz]

                # compute input indices from psi datastructure
                hj = nz_col_idx // nlon_in
                # account for output shift and ensure positive index due to circular condition
                wj = nz_col_idx % nlon_in
                wjp = (wj + pscale * wo) % nlon_in

                alpha[:, idz-zstart] = torch.exp(qdotk_nz[:,idz-zstart] - qdotk_max) * quad_weights[hj]
                alpha_sum[:] += alpha[:, idz-zstart]

                # input dot
                gdotv = torch.sum(dy[:,:,ho, wo] * vx[:,:,hj, wjp], dim=1)

                # integral term
                integral[:] += alpha[:, idz-zstart] * gdotv[:]

            integral[:] = integral[:] / alpha_sum[:]

            for idz in range(zstart, zend):
                nz_col_idx = col_idx[idz]

                # compute input indices from psi datastructure
                hi = nz_col_idx // nlon_in
                # account for output shift and ensure positive index due to circular condition
                wi = nz_col_idx % nlon_in
                wip = (wi + pscale * wo) % nlon_in

                # compute correlation & softmax numerator
                gdotv = torch.sum(dy[:,:,ho, wo] * vx[:,:,hi, wip], dim=1)

                dkx[:,:,hi,wip] += qy[:, :, ho, wo] * (alpha[:, None, idz-zstart] / alpha_sum[:, None]) * (gdotv[:, None] - integral[:, None])

    return dkx

# Explicit gradient w.r.t. qy: dM/dq
# provided as a reference for CUDA & other hand-written gradients
def _neighborhood_s2_attention_bwd_dq_torch(kx: torch.Tensor, vx: torch.Tensor, qy: torch.Tensor, dy: torch.Tensor,
                                            quad_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor,
                                            nlon_in: int, nlat_out: int, nlon_out: int):

    # shapes:
    # input
    # kx: B, C, Hi, Wi
    # vx: B, Cout, Hi, Wi
    # qy: B, C, Ho, Wo
    # quad_weights: Hi
    # output
    # dq: B, C, Ho, Wo

    assert nlon_in % nlon_out == 0, f"nlon_in ({nlon_in}) must be an integer multiple of nlon_out ({nlon_out})"
    pscale = nlon_in // nlon_out

    batch_size = dy.shape[0]
    channels_in = kx.shape[1]
    channels_out = vx.shape[1]

    dqy = torch.zeros_like(qy)

    for ho in range(nlat_out):

        # get number of nonzeros
        zstart = row_off[ho]
        zend = row_off[ho+1]

        for wo in range(nlon_out):

            alpha = torch.zeros((batch_size, zend-zstart), dtype=dy.dtype, device=dy.device)
            qdotk_nz = torch.zeros((batch_size, zend-zstart), dtype=dy.dtype, device=dy.device)
            alpha_k = torch.zeros((batch_size, channels_in), dtype=dy.dtype, device=dy.device)
            alpha_vw = torch.zeros((batch_size,), dtype=dy.dtype, device=dy.device)
            alpha_kvw = torch.zeros((batch_size, channels_in), dtype=dy.dtype, device=dy.device)
            alpha_sum = torch.zeros((batch_size,), dtype=dy.dtype, device=dy.device)
            alpha_sum2 = torch.zeros((batch_size,), dtype=dy.dtype, device=dy.device)
            for idz in range(zstart, zend):
                nz_col_idx = col_idx[idz]

                # compute input indices from psi datastructure
                hi = nz_col_idx // nlon_in
                # account for output shift and ensure positive index due to circular condition
                wi = nz_col_idx % nlon_in
                wip = (wi + pscale * wo) % nlon_in

                idz_i = idz-zstart

                # compute correlation & softmax numerator
                q_ho_wo = qy[:, :, ho, wo]
                k_hi_wi = kx[:, :, hi, wip]
                qdotk_nz[:,idz-zstart] = torch.sum(q_ho_wo * k_hi_wi, dim=1)

            qdotk_max,_ = qdotk_nz.max(dim=1)

            for idz in range(zstart, zend):
                nz_col_idx = col_idx[idz]

                # compute input indices from psi datastructure
                hi = nz_col_idx // nlon_in
                # account for output shift and ensure positive index due to circular condition
                wi = nz_col_idx % nlon_in
                wip = (wi + pscale * wo) % nlon_in

                q_ho_wo = qy[:, :, ho, wo]
                k_hi_wi = kx[:, :, hi, wip]
                idz_i = idz-zstart
                alpha[:, idz_i] = torch.exp(qdotk_nz[:,idz-zstart] - qdotk_max) * quad_weights[hi]
                alpha_sum[:] += alpha[:, idz_i]

                gdotv = torch.sum(dy[:,:,ho, wo] * vx[:,:,hi, wip], dim=1)
                alpha_k[:,:] += alpha[:, None, idz_i] * k_hi_wi
                alpha_vw[:] += alpha[:, idz_i] * gdotv[:]
                alpha_kvw[:,:] += alpha[:, None, idz_i] * k_hi_wi * gdotv[:,None]

            dqy[:,:,ho,wo] = (alpha_kvw * alpha_sum[:,None] - alpha_vw[:, None] * alpha_k) / (alpha_sum[:,None] * alpha_sum[:,None])

    return dqy

@torch.library.custom_op("attention_kernels::_neighborhood_s2_attention_torch", mutates_args=())
def _neighborhood_s2_attention_torch(kw: torch.Tensor, vw: torch.Tensor, qw: torch.Tensor,
                                     quad_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor,
                                     max_psi_nnz: int, nh: int, nlon_in: int, nlat_out: int, nlon_out: int) -> torch.Tensor:

    # reshape, folding num heads into batch dim
    B, _, H, W = kw.shape
    kw = kw.reshape(B*nh, -1, H, W)
    B, _, H, W = vw.shape
    vw = vw.reshape(B*nh, -1, H, W)
    B, _, H, W = qw.shape
    qw = qw.reshape(B*nh, -1, H, W)

    kw = kw.to(torch.float32)
    vw = vw.to(torch.float32)
    qw = qw.to(torch.float32)

    # direction selection: gather (self / downsample) iff nlon_in is an integer
    # multiple of nlon_out; scatter (upsample) iff nlon_out is an integer multiple
    # of nlon_in. Self-attention (nlon_in == nlon_out) satisfies both and falls
    # through the gather path with pscale == 1.
    if nlon_in % nlon_out == 0:
        output = _neighborhood_s2_attention_fwd_torch(kw, vw, qw, quad_weights,
                                                      col_idx, row_off,
                                                      nlon_in, nlat_out, nlon_out)
    elif nlon_out % nlon_in == 0:
        output = _neighborhood_s2_attention_upsample_fwd_torch(kw, vw, qw, quad_weights,
                                                               col_idx, row_off,
                                                               nlon_in, nlat_out, nlon_out)
    else:
        raise ValueError(f"either nlon_in ({nlon_in}) must be an integer multiple of nlon_out ({nlon_out}), or vice versa")

    _, _, H, W = output.shape
    output = output.reshape(B, -1, H, W)

    return output

@torch.library.register_fake("attention_kernels::_neighborhood_s2_attention_torch")
def _(kw: torch.Tensor, vw: torch.Tensor, qw: torch.Tensor,
      quad_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor,
      max_psi_nnz: int, nh: int, nlon_in: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
    out_shape = (kw.shape[0], vw.shape[1], nlat_out, nlon_out)
    return torch.empty(out_shape, dtype=kw.dtype, device=kw.device)

def _neighborhood_s2_attention_bwd_torch(ctx, grad_output):
    col_idx, row_off, quad_weights, kw, vw, qw = ctx.saved_tensors
    nh = ctx.nh
    nlon_in = ctx.nlon_in
    nlat_out = ctx.nlat_out
    nlon_out = ctx.nlon_out

    # check if we need the grads at all
    kw_needs_grad = ctx.needs_input_grad[0]
    vw_needs_grad = ctx.needs_input_grad[1]
    qw_needs_grad = ctx.needs_input_grad[2]

    # reshape, folding num heads into batch dim
    B, _, H, W = kw.shape
    kw = kw.reshape(B*nh, -1, H, W)
    B, _, H, W = vw.shape
    vw = vw.reshape(B*nh, -1, H, W)
    B, _, H, W = qw.shape
    qw = qw.reshape(B*nh, -1, H, W)
    B, _, H, W  = grad_output.shape
    grad_output = grad_output.reshape(B*nh, -1, H, W)

    # direction selection — same convention as the forward op.
    if nlon_in % nlon_out == 0:
        dv_fn = _neighborhood_s2_attention_bwd_dv_torch
        dk_fn = _neighborhood_s2_attention_bwd_dk_torch
        dq_fn = _neighborhood_s2_attention_bwd_dq_torch
    elif nlon_out % nlon_in == 0:
        dv_fn = _neighborhood_s2_attention_upsample_bwd_dv_torch
        dk_fn = _neighborhood_s2_attention_upsample_bwd_dk_torch
        dq_fn = _neighborhood_s2_attention_upsample_bwd_dq_torch
    else:
        raise ValueError(f"either nlon_in ({nlon_in}) must be an integer multiple of nlon_out ({nlon_out}), or vice versa")

    if vw_needs_grad:
        dvw = dv_fn(kw, vw, qw, grad_output, quad_weights, col_idx, row_off, nlon_in, nlat_out, nlon_out)
        _, _, H, W = dvw.shape
        dvw = dvw.reshape(B, -1, H, W)
    else:
        dvw = None

    if kw_needs_grad:
        dkw = dk_fn(kw, vw, qw, grad_output, quad_weights, col_idx, row_off, nlon_in, nlat_out, nlon_out)
        _, _, H, W = dkw.shape
        dkw = dkw.reshape(B, -1, H, W)
    else:
        dkw = None

    if qw_needs_grad:
        dqw = dq_fn(kw, vw, qw, grad_output, quad_weights, col_idx, row_off, nlon_in, nlat_out, nlon_out)
        _, _, H, W = dqw.shape
        dqw = dqw.reshape(B, -1, H, W)
    else:
        dqw = None

    return dkw, dvw, dqw, \
            None, None, None, None, None, None, None, None

# register backward
torch.library.register_autograd("attention_kernels::_neighborhood_s2_attention_torch", _neighborhood_s2_attention_bwd_torch, setup_context=_setup_context_attention_backward)

# =====================================================================================
# UPSAMPLE (scatter-style) torch reference
# =====================================================================================
#
# Symmetry with DISCO's transpose module:
#   psi is precomputed by calling _precompute_convolution_tensor_s2 with in_shape
#   and out_shape swapped, so rows of psi index the (smaller) input grid and
#   cols encode (ho_big, wo_big_ref) on the (larger) output grid as
#   ho_big * nlon_out + wo_big_ref.
#
# Translation invariance: psi is built for input (hi_small, wi_small=0).
# For wi_small > 0 we shift the stored output column index by pscale_out * wi_small
# where pscale_out = nlon_out / nlon_in.
#
# Forward is a 2-pass scatter softmax (classical, not online):
#   pass 1: for every (input, output-neighbor) pair, update qdotk_max[ho, wo]
#   pass 2: with max fixed, accumulate alpha_sum[ho, wo] and y_acc[ho, wo]
#   finalize: y = y_acc / alpha_sum
# The backward references rebuild the softmax state on the fly (same pattern as
# the downsample torch references; reference code, not meant to be fast).
def _neighborhood_s2_attention_upsample_fwd_torch(kx: torch.Tensor, vx: torch.Tensor, qy: torch.Tensor,
                                                   quad_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor,
                                                   nlon_in: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
    """Scatter-style attention forward for the upsample direction.

    kx, vx: [B, C_k/C_v, nlat_in, nlon_in]   (small grid = INPUT)
    qy:     [B, C_k,     nlat_out, nlon_out] (large grid = OUTPUT)
    col_idx/row_off: psi built with swapped shapes; rows=nlat_in, cols=(ho*nlon_out + wo_ref)
    """

    assert nlon_out % nlon_in == 0, f"nlon_out ({nlon_out}) must be an integer multiple of nlon_in ({nlon_in})"
    pscale_out = nlon_out // nlon_in

    B = qy.shape[0]
    C_v = vx.shape[1]
    nlat_in = kx.shape[-2]
    device, dtype = qy.device, qy.dtype

    # state buffers (channels-first)
    qdotk_max = torch.full((B, nlat_out, nlon_out), float('-inf'), dtype=dtype, device=device)
    alpha_sum = torch.zeros((B, nlat_out, nlon_out), dtype=dtype, device=device)
    y_acc = torch.zeros((B, C_v, nlat_out, nlon_out), dtype=dtype, device=device)

    # pass 1: compute qdotk_max over all (input, output-neighbor) pairs
    for hi in range(nlat_in):
        zstart = int(row_off[hi])
        zend = int(row_off[hi + 1])
        for wi in range(nlon_in):
            for idz in range(zstart, zend):
                col = int(col_idx[idz])
                ho = col // nlon_out
                wo_ref = col % nlon_out
                wo = (wo_ref + pscale_out * wi) % nlon_out

                qdotk = torch.sum(qy[:, :, ho, wo] * kx[:, :, hi, wi], dim=1)   # [B]
                qdotk_max[:, ho, wo] = torch.maximum(qdotk_max[:, ho, wo], qdotk)

    # pass 2: with max fixed, accumulate alpha_sum and y_acc
    for hi in range(nlat_in):
        zstart = int(row_off[hi])
        zend = int(row_off[hi + 1])
        for wi in range(nlon_in):
            for idz in range(zstart, zend):
                col = int(col_idx[idz])
                ho = col // nlon_out
                wo_ref = col % nlon_out
                wo = (wo_ref + pscale_out * wi) % nlon_out

                qdotk = torch.sum(qy[:, :, ho, wo] * kx[:, :, hi, wi], dim=1)   # [B]
                alpha = torch.exp(qdotk - qdotk_max[:, ho, wo]) * quad_weights[hi]
                alpha_sum[:, ho, wo] += alpha
                y_acc[:, :, ho, wo] += alpha.unsqueeze(1) * vx[:, :, hi, wi]

    # finalize
    y = y_acc / alpha_sum.unsqueeze(1)
    return y


def _neighborhood_s2_attention_upsample_bwd_dv_torch(kx: torch.Tensor, vx: torch.Tensor, qy: torch.Tensor, dy: torch.Tensor,
                                                      quad_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor,
                                                      nlon_in: int, nlat_out: int, nlon_out: int):
    """Scatter gradient w.r.t. vx.

    dvx[b, c, hi, wi] = sum over output neighbors (ho, wo) of  alpha_norm * dy[b, c, ho, wo]
    where alpha_norm = alpha[ho,wo,hi,wi] / alpha_sum[ho,wo].
    """

    assert nlon_out % nlon_in == 0
    pscale_out = nlon_out // nlon_in

    B = qy.shape[0]
    nlat_in = kx.shape[-2]
    device, dtype = qy.device, qy.dtype

    # recompute qdotk_max and alpha_sum per output cell
    qdotk_max = torch.full((B, nlat_out, nlon_out), float('-inf'), dtype=dtype, device=device)
    for hi in range(nlat_in):
        zstart, zend = int(row_off[hi]), int(row_off[hi + 1])
        for wi in range(nlon_in):
            for idz in range(zstart, zend):
                col = int(col_idx[idz])
                ho = col // nlon_out
                wo = (col % nlon_out + pscale_out * wi) % nlon_out
                qdotk = torch.sum(qy[:, :, ho, wo] * kx[:, :, hi, wi], dim=1)
                qdotk_max[:, ho, wo] = torch.maximum(qdotk_max[:, ho, wo], qdotk)

    alpha_sum = torch.zeros((B, nlat_out, nlon_out), dtype=dtype, device=device)
    for hi in range(nlat_in):
        zstart, zend = int(row_off[hi]), int(row_off[hi + 1])
        for wi in range(nlon_in):
            for idz in range(zstart, zend):
                col = int(col_idx[idz])
                ho = col // nlon_out
                wo = (col % nlon_out + pscale_out * wi) % nlon_out
                qdotk = torch.sum(qy[:, :, ho, wo] * kx[:, :, hi, wi], dim=1)
                alpha_sum[:, ho, wo] += torch.exp(qdotk - qdotk_max[:, ho, wo]) * quad_weights[hi]

    # scatter dvx
    dvx = torch.zeros_like(vx)
    for hi in range(nlat_in):
        zstart, zend = int(row_off[hi]), int(row_off[hi + 1])
        for wi in range(nlon_in):
            for idz in range(zstart, zend):
                col = int(col_idx[idz])
                ho = col // nlon_out
                wo = (col % nlon_out + pscale_out * wi) % nlon_out
                qdotk = torch.sum(qy[:, :, ho, wo] * kx[:, :, hi, wi], dim=1)
                alpha = torch.exp(qdotk - qdotk_max[:, ho, wo]) * quad_weights[hi]
                alpha_norm = alpha / alpha_sum[:, ho, wo]                         # [B]
                dvx[:, :, hi, wi] += alpha_norm.unsqueeze(1) * dy[:, :, ho, wo]

    return dvx


def _neighborhood_s2_attention_upsample_bwd_dk_torch(kx: torch.Tensor, vx: torch.Tensor, qy: torch.Tensor, dy: torch.Tensor,
                                                      quad_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor,
                                                      nlon_in: int, nlat_out: int, nlon_out: int):
    """Scatter gradient w.r.t. kx.

    dkx[b, c, hi, wi] = sum over output neighbors (ho, wo) of
         qy[b, c, ho, wo] * alpha_norm * (dy . v[hi, wi] - integral[ho, wo])
    where integral[ho, wo] = sum_j alpha_norm_j * (dy . v_j).
    """

    assert nlon_out % nlon_in == 0
    pscale_out = nlon_out // nlon_in

    B = qy.shape[0]
    nlat_in = kx.shape[-2]
    device, dtype = qy.device, qy.dtype

    # --- pass 1: qdotk_max per output ---
    qdotk_max = torch.full((B, nlat_out, nlon_out), float('-inf'), dtype=dtype, device=device)
    for hi in range(nlat_in):
        zstart, zend = int(row_off[hi]), int(row_off[hi + 1])
        for wi in range(nlon_in):
            for idz in range(zstart, zend):
                col = int(col_idx[idz])
                ho = col // nlon_out
                wo = (col % nlon_out + pscale_out * wi) % nlon_out
                qdotk = torch.sum(qy[:, :, ho, wo] * kx[:, :, hi, wi], dim=1)
                qdotk_max[:, ho, wo] = torch.maximum(qdotk_max[:, ho, wo], qdotk)

    # --- pass 2: alpha_sum and integral per output ---
    alpha_sum = torch.zeros((B, nlat_out, nlon_out), dtype=dtype, device=device)
    integral = torch.zeros((B, nlat_out, nlon_out), dtype=dtype, device=device)
    for hi in range(nlat_in):
        zstart, zend = int(row_off[hi]), int(row_off[hi + 1])
        for wi in range(nlon_in):
            for idz in range(zstart, zend):
                col = int(col_idx[idz])
                ho = col // nlon_out
                wo = (col % nlon_out + pscale_out * wi) % nlon_out
                qdotk = torch.sum(qy[:, :, ho, wo] * kx[:, :, hi, wi], dim=1)
                alpha = torch.exp(qdotk - qdotk_max[:, ho, wo]) * quad_weights[hi]
                alpha_sum[:, ho, wo] += alpha
                gdotv = torch.sum(dy[:, :, ho, wo] * vx[:, :, hi, wi], dim=1)     # [B]
                integral[:, ho, wo] += alpha * gdotv
    integral = integral / alpha_sum

    # --- pass 3: scatter dkx ---
    dkx = torch.zeros_like(kx)
    for hi in range(nlat_in):
        zstart, zend = int(row_off[hi]), int(row_off[hi + 1])
        for wi in range(nlon_in):
            for idz in range(zstart, zend):
                col = int(col_idx[idz])
                ho = col // nlon_out
                wo = (col % nlon_out + pscale_out * wi) % nlon_out
                qdotk = torch.sum(qy[:, :, ho, wo] * kx[:, :, hi, wi], dim=1)
                alpha = torch.exp(qdotk - qdotk_max[:, ho, wo]) * quad_weights[hi]
                alpha_norm = alpha / alpha_sum[:, ho, wo]                         # [B]
                gdotv = torch.sum(dy[:, :, ho, wo] * vx[:, :, hi, wi], dim=1)     # [B]
                dkx[:, :, hi, wi] += qy[:, :, ho, wo] * (alpha_norm * (gdotv - integral[:, ho, wo])).unsqueeze(1)

    return dkx


def _neighborhood_s2_attention_upsample_bwd_dq_torch(kx: torch.Tensor, vx: torch.Tensor, qy: torch.Tensor, dy: torch.Tensor,
                                                      quad_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor,
                                                      nlon_in: int, nlat_out: int, nlon_out: int):
    """Scatter gradient w.r.t. qy.

    dqy[b, c, ho, wo] = (alpha_kvw * alpha_sum - alpha_vw * alpha_k) / alpha_sum^2
    where the alpha_* accumulators are sums over (hi, wi) neighbors of (ho, wo).
    We accumulate them per output cell by scattering from all (hi, wi) inputs.
    """

    assert nlon_out % nlon_in == 0
    pscale_out = nlon_out // nlon_in

    B = qy.shape[0]
    C_k = kx.shape[1]
    nlat_in = kx.shape[-2]
    device, dtype = qy.device, qy.dtype

    # --- pass 1: qdotk_max per output ---
    qdotk_max = torch.full((B, nlat_out, nlon_out), float('-inf'), dtype=dtype, device=device)
    for hi in range(nlat_in):
        zstart, zend = int(row_off[hi]), int(row_off[hi + 1])
        for wi in range(nlon_in):
            for idz in range(zstart, zend):
                col = int(col_idx[idz])
                ho = col // nlon_out
                wo = (col % nlon_out + pscale_out * wi) % nlon_out
                qdotk = torch.sum(qy[:, :, ho, wo] * kx[:, :, hi, wi], dim=1)
                qdotk_max[:, ho, wo] = torch.maximum(qdotk_max[:, ho, wo], qdotk)

    # --- pass 2: alpha_sum, alpha_k, alpha_vw, alpha_kvw per output ---
    alpha_sum = torch.zeros((B, nlat_out, nlon_out), dtype=dtype, device=device)
    alpha_vw = torch.zeros((B, nlat_out, nlon_out), dtype=dtype, device=device)
    alpha_k = torch.zeros((B, C_k, nlat_out, nlon_out), dtype=dtype, device=device)
    alpha_kvw = torch.zeros((B, C_k, nlat_out, nlon_out), dtype=dtype, device=device)

    for hi in range(nlat_in):
        zstart, zend = int(row_off[hi]), int(row_off[hi + 1])
        for wi in range(nlon_in):
            for idz in range(zstart, zend):
                col = int(col_idx[idz])
                ho = col // nlon_out
                wo = (col % nlon_out + pscale_out * wi) % nlon_out
                qdotk = torch.sum(qy[:, :, ho, wo] * kx[:, :, hi, wi], dim=1)
                alpha = torch.exp(qdotk - qdotk_max[:, ho, wo]) * quad_weights[hi]   # [B]
                gdotv = torch.sum(dy[:, :, ho, wo] * vx[:, :, hi, wi], dim=1)         # [B]

                alpha_sum[:, ho, wo] += alpha
                alpha_vw[:, ho, wo] += alpha * gdotv
                alpha_k[:, :, ho, wo] += alpha.unsqueeze(1) * kx[:, :, hi, wi]
                alpha_kvw[:, :, ho, wo] += (alpha * gdotv).unsqueeze(1) * kx[:, :, hi, wi]

    alpha_sum_1 = alpha_sum.unsqueeze(1)
    dqy = (alpha_kvw * alpha_sum_1 - alpha_vw.unsqueeze(1) * alpha_k) / (alpha_sum_1 * alpha_sum_1)
    return dqy


