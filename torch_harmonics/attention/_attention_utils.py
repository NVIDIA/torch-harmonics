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

import math
from typing import Union

import torch
import torch.nn.functional as F
from torch.amp import custom_fwd, custom_bwd
from attention_helpers import optimized_kernels_is_available
from . import attention_kernels

# HELPER ROUTINE FOR BACKWARD setup_context
def _setup_context_attention_backward(ctx, inputs, output):
    k, v, q, wk, wv, wq, bk, bv, bq, quad_weights, col_idx, row_off, max_psi_nnz, nh, nlon_in, nlat_out, nlon_out = inputs
    ctx.save_for_backward(col_idx, row_off, quad_weights, k, v, q, wk, wv, wq, bk, bv, bq)
    ctx.nh = nh
    ctx.max_psi_nnz = max_psi_nnz
    ctx.nlon_in = nlon_in
    ctx.nlat_out = nlat_out
    ctx.nlon_out = nlon_out

# define NA op for CUDA
@torch.library.custom_op("attention_kernels::_neighborhood_s2_attention_optimized", mutates_args=())
def _neighborhood_s2_attention_optimized(k: torch.Tensor, v: torch.Tensor, q: torch.Tensor,
                                         wk: torch.Tensor, wv: torch.Tensor, wq: torch.Tensor,
                                         bk: Union[torch.Tensor, None], bv: Union[torch.Tensor, None], bq: Union[torch.Tensor, None],
                                         quad_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor,
                                         max_psi_nnz: int, nh: int, nlon_in: int, nlat_out: int, nlon_out: int) -> torch.Tensor:

    kw = F.conv2d(k, weight=wk, bias=bk)
    vw = F.conv2d(v, weight=wv, bias=bv)
    qw = F.conv2d(q, weight=wq, bias=bq)

    # reshape, folding num heads into batch dim
    B, _, H, W = kw.shape
    kw = kw.reshape(B*nh, -1, H, W)
    B, _, H, W = vw.shape
    vw = vw.reshape(B*nh, -1, H, W)
    B, _, H, W = qw.shape
    qw = qw.reshape(B*nh, -1, H, W)

    # convert to float32
    inp_dtype = kw.dtype
    kw = kw.to(torch.float32).contiguous()
    vw = vw.to(torch.float32).contiguous()
    qw = qw.to(torch.float32).contiguous()

    output = attention_kernels.forward.default(kw, vw, qw, quad_weights,
                                               col_idx, row_off,
                                               nlon_in, nlat_out, nlon_out)

    _, C, H, W = output.shape
    output = output.reshape(B, -1, H, W)

    # convert back precision
    output = output.to(dtype=inp_dtype)

    return output

@torch.library.register_fake("attention_kernels::_neighborhood_s2_attention_optimized")
def _(k: torch.Tensor, v: torch.Tensor, q: torch.Tensor,
      wk: torch.Tensor, wv: torch.Tensor, wq: torch.Tensor,
      bk: Union[torch.Tensor, None], bv: Union[torch.Tensor, None], bq: Union[torch.Tensor, None],
      quad_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor,
      max_psi_nnz: int, nh: int, nlon_in: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
    out_shape = (k.shape[0], wv.shape[0], nlat_out, nlon_out)
    return torch.empty(out_shape, dtype=k.dtype, device=k.device)

def _neighborhood_s2_attention_bwd_optimized(ctx, grad_output):
    col_idx, row_off, quad_weights, k, v, q, wk, wv, wq, bk, bv, bq = ctx.saved_tensors
    nh = ctx.nh
    max_psi_nnz = ctx.max_psi_nnz
    nlon_in = ctx.nlon_in
    nlat_out = ctx.nlat_out
    nlon_out = ctx.nlon_out

    # check if we need the grads at all
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

    # reshape, folding num heads into batch dim
    B, _, H, W = kw.shape
    kw = kw.reshape(B*nh, -1, H, W)
    B, _, H, W = vw.shape
    vw = vw.reshape(B*nh, -1, H, W)
    B, _, H, W = qw.shape
    qw = qw.reshape(B*nh, -1, H, W)
    B, _, H, W  = grad_output.shape
    grad_output = grad_output.reshape(B*nh, -1, H, W)

    # save type and convert to float32
    kw_dtype = kw.dtype
    vw_dtype = vw.dtype
    qw_dtype = qw.dtype

    kw = kw.to(torch.float32).contiguous()
    vw = vw.to(torch.float32).contiguous()
    qw = qw.to(torch.float32).contiguous()
    grad_output = grad_output.to(torch.float32).contiguous()

    dkw, dvw, dqw = attention_kernels.backward.default(kw, vw, qw, grad_output,
                                                       quad_weights,
                                                       col_idx, row_off,
                                                       nlon_in, nlat_out, nlon_out)

    # weight grads
    if wk_needs_grad:
        _, C, H, W = dkw.shape
        dkw = dkw.reshape(B, -1, H, W)
        dkw = dkw.to(dtype=kw_dtype)
        dwk = torch.einsum("bchw,bfhw->cf", dkw, k).reshape(*wk.shape).contiguous()
    else:
        dwk = None

    if wv_needs_grad:
        _, C, H, W = dvw.shape
        dvw = dvw.reshape(B, -1, H, W)
        dvw = dvw.to(dtype=vw_dtype)
        dwv = torch.einsum("bchw,bfhw->cf", dvw, v).reshape(*wv.shape).contiguous()
    else:
        dwv = None

    if wq_needs_grad:
        _, C, H, W = dqw.shape
        dqw = dqw.reshape(B, -1, H, W)
        dqw = dqw.to(dtype=qw_dtype)
        dwq = torch.einsum("bchw,bfhw->cf", dqw, q).reshape(*wq.shape).contiguous()
    else:
        dwq = None
   
    # input grads
    if v_needs_grad:
        dv = torch.nn.functional.conv2d(dvw, weight=wv.permute([1,0,2,3]), bias=None)
    else:
        dv = None

    if k_needs_grad:
        dk = torch.nn.functional.conv2d(dkw, weight=wk.permute([1,0,2,3]), bias=None)
    else:
        dk = None

    if q_needs_grad:
        dq = torch.nn.functional.conv2d(dqw, weight=wq.permute([1,0,2,3]), bias=None)
    else:
        dq = None

    # bias grads:
    if bv_needs_grad
        dbv = torch.sum(dvw, dim=(0,2,3))
    else:
        dbv = None

    if bk_needs_grad
        dbk = torch.sum(dkw, dim=(0,2,3))
    else:
        dbk = None

    if bq_needs_grad:
        dbq = torch.sum(dqw, dim=(0,2,3))
    else:
        dbq = None

    return dk, dv, dq, dwk, dwv, dwq, dbk, dbv, dbq, \
            None, None, None, None, None, None, None, None

# register backward
if optimized_kernels_is_available():
    torch.library.register_autograd("attention_kernels::_neighborhood_s2_attention_optimized", _neighborhood_s2_attention_bwd_optimized, setup_context=_setup_context_attention_backward)


# torch kernels
# uses qdotk_max update trick to avoid two loops when computing the softmax
# see e.g., https://arxiv.org/abs/1805.02867
# and https://alexdremov.me/understanding-flash-attention-writing-the-algorithm-from-scratch-in-triton/
def _neighborhood_s2_attention_fwd_torch(kx: torch.Tensor, vx: torch.Tensor, qy: torch.Tensor,
                                         quad_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor,
                                         nlon_in: int, nlat_out: int, nlon_out: int) -> torch.Tensor:


    # prepare result tensor
    y = torch.zeros_like(qy)

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
                wip = (wi + wo) % nlon_in

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
    # vx: B, C, Hi, Wi
    # qy: B, C, Ho, Wo
    # quad_weights: Hi
    # output
    # dvx: B, C, Hi, Wi

    dvx = torch.zeros_like(vx)

    for ho in range(nlat_out):

        # get number of nonzeros
        zstart = row_off[ho]
        zend = row_off[ho+1]

        for wo in range(nlon_out):

            alpha_nz = torch.zeros((dy.shape[0], zend-zstart), dtype=dy.dtype, device=dy.device)
            qdotk_nz = torch.zeros((dy.shape[0], zend-zstart), dtype=dy.dtype, device=dy.device)
            alpha_sum = torch.zeros((dy.shape[0],), dtype=dy.dtype, device=dy.device)
            for idz in range(zstart, zend):
                nz_col_idx = col_idx[idz]

                # compute input indices from psi datastructure
                hi = nz_col_idx // nlon_in
                # account for output shift and ensure positive index due to circular condition
                wi = nz_col_idx % nlon_in
                wip = (wi+wo) % nlon_in

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
                wip = (wi+wo) % nlon_in
                alpha_nz[:,idz-zstart] = torch.exp(qdotk_nz[:,idz-zstart] - qdotk_max) * quad_weights[hi]
                alpha_sum[:] += alpha_nz[:,idz-zstart]

            for idz in range(zstart, zend):
                nz_col_idx = col_idx[idz]

                # compute input indices from psi datastructure
                hi = nz_col_idx // nlon_in
                # account for output shift and ensure positive index due to circular condition
                wi = nz_col_idx % nlon_in
                wip = (wi+wo) % nlon_in
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
    # vx: B, C, Hi, Wi
    # qy: B, C, Ho, Wo
    # quad_weights: Hi
    # output
    # dkx: B, C, Hi, Wi

    dkx = torch.zeros_like(kx)

    for ho in range(nlat_out):

        # get number of nonzeros
        zstart = row_off[ho]
        zend = row_off[ho+1]

        for wo in range(nlon_out):

            qdotk_nz = torch.zeros((dy.shape[0], zend-zstart), dtype=dy.dtype, device=dy.device)
            integral = torch.zeros((dy.shape[0],), dtype=dy.dtype, device=dy.device)
            alpha = torch.zeros((dy.shape[0], zend-zstart), dtype=dy.dtype, device=dy.device)
            alpha_sum = torch.zeros((dy.shape[0],), dtype=dy.dtype, device=dy.device)
            for idz in range(zstart, zend):
                nz_col_idx = col_idx[idz]

                # compute input indices from psi datastructure
                hj = nz_col_idx // nlon_in
                # account for output shift and ensure positive index due to circular condition
                wj = nz_col_idx % nlon_in
                wjp = (wj+wo) % nlon_in

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
                wjp = (wj+wo) % nlon_in

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
                wip = (wi+wo) % nlon_in

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
    # vx: B, C, Hi, Wi
    # qy: B, C, Ho, Wo
    # quad_weights: Hi
    # output
    # dvx: B, C, Hi, Wi

    dqy = torch.zeros_like(qy)

    for ho in range(nlat_out):

        # get number of nonzeros
        zstart = row_off[ho]
        zend = row_off[ho+1]

        for wo in range(nlon_out):

            alpha = torch.zeros((dy.shape[0], zend-zstart), dtype=dy.dtype, device=dy.device)
            qdotk_nz = torch.zeros((dy.shape[0], zend-zstart), dtype=dy.dtype, device=dy.device)
            alpha_k = torch.zeros((dy.shape[0], dy.shape[1]), dtype=dy.dtype, device=dy.device)
            alpha_vw = torch.zeros((dy.shape[0], dy.shape[1]), dtype=dy.dtype, device=dy.device)
            alpha_kvw = torch.zeros((dy.shape[0], dy.shape[1]), dtype=dy.dtype, device=dy.device)
            alpha_sum = torch.zeros((dy.shape[0],), dtype=dy.dtype, device=dy.device)
            alpha_sum2 = torch.zeros((dy.shape[0],), dtype=dy.dtype, device=dy.device)
            for idz in range(zstart, zend):
                nz_col_idx = col_idx[idz]

                # compute input indices from psi datastructure
                hi = nz_col_idx // nlon_in
                # account for output shift and ensure positive index due to circular condition
                wi = nz_col_idx % nlon_in
                wip = (wi+wo) % nlon_in

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
                wip = (wi+wo) % nlon_in

                q_ho_wo = qy[:, :, ho, wo]
                k_hi_wi = kx[:, :, hi, wip]
                idz_i = idz-zstart
                alpha[:, idz_i] = torch.exp(qdotk_nz[:,idz-zstart] - qdotk_max) * quad_weights[hi]
                alpha_sum[:] += alpha[:, idz_i]

                gdotv = torch.sum(dy[:,:,ho, wo] * vx[:,:,hi, wip], dim=1)
                alpha_k[:,:] += alpha[:, None, idz_i] * k_hi_wi
                alpha_vw[:,:] += alpha[:, None, idz_i] * gdotv[:,None]
                alpha_kvw[:,:] += alpha[:, None, idz_i] * k_hi_wi * gdotv[:,None]

            dqy[:,:,ho,wo] = (alpha_kvw*alpha_sum[:,None] - alpha_vw*alpha_k) / (alpha_sum[:,None]*alpha_sum[:,None])

    return dqy

# this is legacy code and will be updated soon
class _NeighborhoodAttentionS2(torch.autograd.Function):

    @staticmethod
    @custom_fwd(device_type="cpu")
    def forward(ctx, k: torch.Tensor, v: torch.Tensor, q: torch.Tensor,
                wk: torch.Tensor, wv: torch.Tensor, wq: torch.Tensor,
                bk: Union[torch.Tensor, None], bv: Union[torch.Tensor, None], bq: Union[torch.Tensor, None],
                quad_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor,
                nh: int, nlon_in: int, nlat_out: int, nlon_out: int):

        ctx.save_for_backward(col_idx, row_off, quad_weights, k, v, q, wk, wv, wq, bk, bv, bq)
        ctx.nh = nh
        ctx.nlon_in = nlon_in
        ctx.nlat_out = nlat_out
        ctx.nlon_out = nlon_out

        kw = F.conv2d(k, weight=wk, bias=bk)
        vw = F.conv2d(v, weight=wv, bias=bv)
        qw = F.conv2d(q, weight=wq, bias=bq)

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

        output = _neighborhood_attention_s2_fwd_torch(kw, vw, qw, quad_weights,
                                                      col_idx, row_off,
                                                      nlon_in, nlat_out, nlon_out)

        _, C, H, W = output.shape
        output = output.reshape(B, -1, H, W)

        return output

    @staticmethod
    @custom_bwd(device_type="cpu")
    def backward(ctx, grad_output):
        col_idx, row_off, quad_weights, k, v, q, wk, wv, wq, bk, bv, bq = ctx.saved_tensors
        nh = ctx.nh
        nlon_in = ctx.nlon_in
        nlat_out = ctx.nlat_out
        nlon_out = ctx.nlon_out

        kw = F.conv2d(k, weight=wk, bias=bk)
        vw = F.conv2d(v, weight=wv, bias=bv)
        qw = F.conv2d(q, weight=wq, bias=bq)

        # reshape, folding num heads into batch dim
        B, _, H, W = kw.shape
        kw = kw.reshape(B*nh, -1, H, W)
        B, _, H, W = vw.shape
        vw = vw.reshape(B*nh, -1, H, W)
        B, _, H, W = qw.shape
        qw = qw.reshape(B*nh, -1, H, W)
        B, _, H, W  = grad_output.shape
        grad_output = grad_output.reshape(B*nh, -1, H, W)

        dvw = _neighborhood_attention_s2_bwd_dv_torch(kw, vw, qw, grad_output,
                                                      quad_weights,
                                                      col_idx, row_off,
                                                      nlon_in, nlat_out, nlon_out)

        dkw = _neighborhood_attention_s2_bwd_dk_torch(kw, vw, qw, grad_output,
                                                      quad_weights,
                                                      col_idx, row_off,
                                                      nlon_in, nlat_out, nlon_out)

        dqw = _neighborhood_attention_s2_bwd_dq_torch(kw, vw, qw, grad_output,
                                                      quad_weights,
                                                      col_idx, row_off,
                                                      nlon_in, nlat_out, nlon_out)

        # reshape again
        _, C, H, W = dkw.shape
        dkw = dkw.reshape(B, -1, H, W)
        _, C, H, W = dvw.shape
        dvw = dvw.reshape(B, -1, H, W)
        _, C, H, W = dqw.shape
        dqw = dqw.reshape(B, -1, H, W)

        # input grads
        dv = torch.nn.functional.conv2d(dvw, weight=wv.permute([1,0,2,3]), bias=None)
        dk = torch.nn.functional.conv2d(dkw, weight=wk.permute([1,0,2,3]), bias=None)
        dq = torch.nn.functional.conv2d(dqw, weight=wq.permute([1,0,2,3]), bias=None)

        # weight grads
        dwv = torch.einsum("bchw,bfhw->cf", dvw, v).reshape(*wv.shape).contiguous()
        dwk = torch.einsum("bchw,bfhw->cf", dkw, k).reshape(*wk.shape).contiguous()
        dwq = torch.einsum("bchw,bfhw->cf", dqw, q).reshape(*wq.shape).contiguous()

        # bias grads:
        if bv is not None:
            dbv = torch.sum(dvw, dim=(0,2,3))
        else:
            dbv = None

        if bk is not None:
            dbk = torch.sum(dkw, dim=(0,2,3))
        else:
            dbk = None

        if bq is not None:
            dbq = torch.sum(dqw, dim=(0,2,3))
        else:
            dbq = None

        return dk, dv, dq, dwk, dwv, dwq, dbk, dbv, dbq, \
                None, None, None, None, None, None, None


def _neighborhood_s2_attention_torch(k: torch.Tensor, v: torch.Tensor, q: torch.Tensor,
                                     wk: torch.Tensor, wv: torch.Tensor, wq: torch.Tensor,
                                     bk: Union[torch.Tensor, None], bv: Union[torch.Tensor, None],
                                     bq: Union[torch.Tensor, None], quad_weights: torch.Tensor,
                                     col_idx: torch.Tensor, row_off: torch.Tensor,
                                     nh: int, nlon_in: int, nlat_out: int, nlon_out: int) -> torch.Tensor:

    return _NeighborhoodAttentionS2.apply(k, v, q, wk, wv, wq, bk, bv, bq,
                                          quad_weights, col_idx, row_off,
                                          nh, nlon_in, nlat_out, nlon_out)