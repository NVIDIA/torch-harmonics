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

# Shared backward-context helper used by both the torch reference kernels
# (in kernels_torch/) and the optimized custom_op path (in optimized/).
def _setup_context_attention_backward(ctx, inputs, output):
    kw, vw, qw, quad_weights, col_idx, row_off, max_psi_nnz, nh, nlon_in, nlat_out, nlon_out = inputs
    ctx.save_for_backward(col_idx, row_off, quad_weights, kw, vw, qw)
    ctx.nh = nh
    ctx.max_psi_nnz = max_psi_nnz
    ctx.nlon_in = nlon_in
    ctx.nlat_out = nlat_out
    ctx.nlon_out = nlon_out

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


# HELPER: same shape as _setup_context_attention_backward
def _setup_context_attention_upsample_backward(ctx, inputs, output):
    k, v, q, wk, wv, wq, bk, bv, bq, quad_weights, col_idx, row_off, max_psi_nnz, nh, nlon_in, nlat_out, nlon_out = inputs
    ctx.save_for_backward(col_idx, row_off, quad_weights, k, v, q, wk, wv, wq, bk, bv, bq)
    ctx.nh = nh
    ctx.max_psi_nnz = max_psi_nnz
    ctx.nlon_in = nlon_in
    ctx.nlat_out = nlat_out
    ctx.nlon_out = nlon_out


@torch.library.custom_op("attention_kernels::_neighborhood_s2_attention_upsample_torch", mutates_args=())
def _neighborhood_s2_attention_upsample_torch(k: torch.Tensor, v: torch.Tensor, q: torch.Tensor,
                                              wk: torch.Tensor, wv: torch.Tensor, wq: torch.Tensor,
                                              bk: Union[torch.Tensor, None], bv: Union[torch.Tensor, None], bq: Union[torch.Tensor, None],
                                              quad_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor,
                                              max_psi_nnz: int, nh: int, nlon_in: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
    kw = F.conv2d(k, weight=wk, bias=bk)
    vw = F.conv2d(v, weight=wv, bias=bv)
    qw = F.conv2d(q, weight=wq, bias=bq)

    # fold num_heads into batch
    B, _, H, W = kw.shape
    kw = kw.reshape(B * nh, -1, H, W)
    B, _, H, W = vw.shape
    vw = vw.reshape(B * nh, -1, H, W)
    B, _, H, W = qw.shape
    qw = qw.reshape(B * nh, -1, H, W)

    kw = kw.to(torch.float32)
    vw = vw.to(torch.float32)
    qw = qw.to(torch.float32)

    output = _neighborhood_s2_attention_upsample_fwd_torch(kw, vw, qw, quad_weights,
                                                           col_idx, row_off,
                                                           nlon_in, nlat_out, nlon_out)

    _, C, H, W = output.shape
    output = output.reshape(B, -1, H, W)

    return output


@torch.library.register_fake("attention_kernels::_neighborhood_s2_attention_upsample_torch")
def _(k: torch.Tensor, v: torch.Tensor, q: torch.Tensor,
      wk: torch.Tensor, wv: torch.Tensor, wq: torch.Tensor,
      bk: Union[torch.Tensor, None], bv: Union[torch.Tensor, None], bq: Union[torch.Tensor, None],
      quad_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor,
      max_psi_nnz: int, nh: int, nlon_in: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
    out_shape = (k.shape[0], wv.shape[0], nlat_out, nlon_out)
    return torch.empty(out_shape, dtype=k.dtype, device=k.device)


def _neighborhood_s2_attention_upsample_bwd_torch(ctx, grad_output):
    col_idx, row_off, quad_weights, k, v, q, wk, wv, wq, bk, bv, bq = ctx.saved_tensors
    nh = ctx.nh
    nlon_in = ctx.nlon_in
    nlat_out = ctx.nlat_out
    nlon_out = ctx.nlon_out

    k_needs_grad = ctx.needs_input_grad[0]
    v_needs_grad = ctx.needs_input_grad[1]
    q_needs_grad = ctx.needs_input_grad[2]
    wk_needs_grad = ctx.needs_input_grad[3]
    wv_needs_grad = ctx.needs_input_grad[4]
    wq_needs_grad = ctx.needs_input_grad[5]
    bk_needs_grad = ctx.needs_input_grad[6]
    bv_needs_grad = ctx.needs_input_grad[7]
    bq_needs_grad = ctx.needs_input_grad[8]

    kw = F.conv2d(k, weight=wk, bias=bk)
    vw = F.conv2d(v, weight=wv, bias=bv)
    qw = F.conv2d(q, weight=wq, bias=bq)

    B, _, H, W = kw.shape
    kw = kw.reshape(B * nh, -1, H, W)
    B, _, H, W = vw.shape
    vw = vw.reshape(B * nh, -1, H, W)
    B, _, H, W = qw.shape
    qw = qw.reshape(B * nh, -1, H, W)
    B, _, H, W = grad_output.shape
    grad_output = grad_output.reshape(B * nh, -1, H, W)

    if v_needs_grad or wv_needs_grad or bv_needs_grad:
        dvw = _neighborhood_s2_attention_upsample_bwd_dv_torch(kw, vw, qw, grad_output,
                                                                quad_weights, col_idx, row_off,
                                                                nlon_in, nlat_out, nlon_out)
        _, C, H, W = dvw.shape
        dvw = dvw.reshape(B, -1, H, W)
    else:
        dvw = None

    if k_needs_grad or wk_needs_grad or bk_needs_grad:
        dkw = _neighborhood_s2_attention_upsample_bwd_dk_torch(kw, vw, qw, grad_output,
                                                                quad_weights, col_idx, row_off,
                                                                nlon_in, nlat_out, nlon_out)
        _, C, H, W = dkw.shape
        dkw = dkw.reshape(B, -1, H, W)
    else:
        dkw = None

    if q_needs_grad or wq_needs_grad or bq_needs_grad:
        dqw = _neighborhood_s2_attention_upsample_bwd_dq_torch(kw, vw, qw, grad_output,
                                                                quad_weights, col_idx, row_off,
                                                                nlon_in, nlat_out, nlon_out)
        _, C, H, W = dqw.shape
        dqw = dqw.reshape(B, -1, H, W)
    else:
        dqw = None

    # input grads via the 1x1 conv adjoint (identical to downsample)
    dv = torch.nn.functional.conv2d(dvw, weight=wv.permute([1, 0, 2, 3]), bias=None) if v_needs_grad else None
    dk = torch.nn.functional.conv2d(dkw, weight=wk.permute([1, 0, 2, 3]), bias=None) if k_needs_grad else None
    dq = torch.nn.functional.conv2d(dqw, weight=wq.permute([1, 0, 2, 3]), bias=None) if q_needs_grad else None

    dwv = torch.einsum("bchw,bfhw->cf", dvw, v).reshape(*wv.shape).contiguous() if wv_needs_grad else None
    dwk = torch.einsum("bchw,bfhw->cf", dkw, k).reshape(*wk.shape).contiguous() if wk_needs_grad else None
    dwq = torch.einsum("bchw,bfhw->cf", dqw, q).reshape(*wq.shape).contiguous() if wq_needs_grad else None

    dbv = torch.sum(dvw, dim=(0, 2, 3)) if bv_needs_grad else None
    dbk = torch.sum(dkw, dim=(0, 2, 3)) if bk_needs_grad else None
    dbq = torch.sum(dqw, dim=(0, 2, 3)) if bq_needs_grad else None

    return dk, dv, dq, dwk, dwv, dwq, dbk, dbv, dbq, \
            None, None, None, None, None, None, None, None


torch.library.register_autograd("attention_kernels::_neighborhood_s2_attention_upsample_torch",
                                _neighborhood_s2_attention_upsample_bwd_torch,
                                setup_context=_setup_context_attention_upsample_backward)
