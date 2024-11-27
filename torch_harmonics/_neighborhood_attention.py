# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2024 The torch-harmonics Authors. All rights reserved.
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

import torch
from torch.amp import custom_fwd, custom_bwd

try:
    import attention_cuda_extension
    _cuda_extension_available = True
except ImportError as err:
    attention_cuda_extension = None
    _cuda_extension_available = False


def _neighborhood_attention_s2_torch(kx: torch.Tensor, vx: torch.Tensor, qy: torch.Tensor,
                                     quad_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor,
                                     nlon_in: int, nlat_out: int, nlon_out: int) -> torch.Tensor:


    # prepare result tensor
    y = torch.empty_like(qy)

    for ho in range(nlat_out):

        # get number of nonzeros
        zstart = row_off[ho]
        zend = row_off[ho+1]

        for wo in range(nlon_out):

            alpha_sum = torch.zeros((y.shape[0],), dtype=y.dtype, device=y.device)
            qdotk_nz = torch.zeros((y.shape[0], zend-zstart,), dtype=y.dtype, device=y.device)

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
                qdotk_nz[:,idz-zstart] = torch.sum(q_ho_wo * k_hi_wip, dim=1)

            qdotk_max,_ = qdotk_nz.max(dim=1)

            for idz in range(zstart, zend):
                nz_col_idx = col_idx[idz]

                # compute input indices from psi datastructure
                hi = nz_col_idx // nlon_in
                # account for output shift and ensure positive index due to circular condition
                wi = nz_col_idx % nlon_in
                wip = (wi + wo) % nlon_in
                alpha = torch.exp(qdotk_nz[:,idz-zstart] - qdotk_max)
                # softmax denominator
                alpha_sum[:] += alpha[:]

                y[:,:,ho,wo] += alpha[:, None] * vx[:,:,hi,wip] * quad_weights[hi]


            y[:,:,ho,wo] = y[:,:,ho,wo] / alpha_sum[:, None]

    return y


# Explicit gradient w.r.t. vx: dM/dv
# provided as a reference for CUDA & other hand-written gradients
def _disco_att_bwd_dv_torch(kx: torch.Tensor, vx: torch.Tensor, qy: torch.Tensor, dy: torch.Tensor,
                           quad_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor,
                           nlat_in: int, nlon_in: int, nlat_out: int, nlon_out: int):

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

            qdotk_max,_ = qdotk_nz.max(dim=1)

            for idz in range(zstart, zend):
                nz_col_idx = col_idx[idz]

                # compute input indices from psi datastructure
                hi = nz_col_idx // nlon_in
                # account for output shift and ensure positive index due to circular condition
                wi = nz_col_idx % nlon_in
                wip = (wi+wo) % nlon_in
                alpha_nz[:,idz-zstart] = torch.exp(qdotk_nz[:,idz-zstart] - qdotk_max)
                alpha_sum[:] += alpha_nz[:,idz-zstart]

            for idz in range(zstart, zend):
		nz_col_idx = col_idx[idz]

                # compute input indices from psi datastructure
                hi = nz_col_idx // nlon_in
                # account for output shift and ensure positive index due to circular condition
                wi = nz_col_idx % nlon_in
                wip = (wi+wo) % nlon_in
                dvx[:,:,hi, wip] += (alpha_nz[:, None, idz-zstart] / alpha_sum[:, None]) * dy[:,:,ho,wo] * quad_weights[hi]

    return dvx


# Explicit gradient w.r.t. kx: dM/dk
# provided as a reference for CUDA & other hand-written gradients
def _disco_att_bwd_dk_torch(kx: torch.Tensor, vx: torch.Tensor, qy: torch.Tensor, dy: torch.Tensor,
                           quad_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor,
                           nlat_in: int, nlon_in: int, nlat_out: int, nlon_out: int):

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

            qdotk_max,_ = qdotk_nz.max(dim=1)

            for idz in range(zstart, zend):
                nz_col_idx = col_idx[idz]

                # compute input indices from psi datastructure
                hj = nz_col_idx // nlon_in
                # account for output shift and ensure positive index due to circular condition
		wj = nz_col_idx % nlon_in
		wjp = (wj+wo) % nlon_in

		alpha[:, idz-zstart] = torch.exp(qdotk_nz[:,idz-zstart] - qdotk_max)
                alpha_sum[:] += alpha[:, idz-zstart]

                # input dot
                gdotv = torch.sum(dy[:,:,ho, wo] * vx[:,:,hj, wjp], dim=1)

                # integral term
                integral[:] += alpha[:, idz-zstart] * gdotv[:] * quad_weights[hj]

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

		dkx[:,:,hi,wip] += qy[:, :, ho, wo] * (alpha[:, None, idz-zstart] / alpha_sum[:, None]) * (quad_weights[hi] * gdotv[:, None] - integral[:, None])

    return dkx

# Explicit gradient w.r.t. qy: dM/dq
# provided as a reference for CUDA & other hand-written gradients
def _disco_att_bwd_dq_torch(kx: torch.Tensor, vx: torch.Tensor, qy: torch.Tensor, dy: torch.Tensor,
                           quad_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor,
                           nlat_in: int, nlon_in: int, nlat_out: int, nlon_out: int):

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
                alpha[:, idz_i] = torch.exp(qdotk_nz[:,idz-zstart] - qdotk_max)
                alpha_sum[:] += alpha[:, idz_i]

                gdotv = torch.sum(dy[:,:,ho, wo] * vx[:,:,hi, wip], dim=1)
                alpha_k[:,:] += alpha[:, None, idz_i] * k_hi_wi
                alpha_vw[:,:] += alpha[:, None, idz_i] * gdotv[:,None] * quad_weights[hi]
                alpha_kvw[:,:] += alpha[:, None, idz_i] * k_hi_wi * gdotv[:,None] * quad_weights[hi]

            dqy[:,:,ho,wo] = (alpha_kvw*alpha_sum[:,None] - alpha_vw*alpha_k) / (alpha_sum[:,None]*alpha_sum[:,None])

    return dqy


class _NeighborhoodAttentionS2Cuda(torch.autograd.Function):

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, kx: torch.Tensor, vx: torch.Tensor, qy: torch.Tensor,
                quad_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor,
                max_psi_nnz: int, nlon_in: int, nlat_out: int, nlon_out: int):

        ctx.save_for_backward(col_idx, row_off, quad_weights, kx, vx, qy)
        ctx.max_psi_nnz = max_psi_nnz
        ctx.nlon_in = nlon_in
        ctx.nlat_out = nlat_out
        ctx.nlon_out = nlon_out

        kx = kx.to(torch.float32)
        vx = vx.to(torch.float32)
        qy = qy.to(torch.float32)

        output = torch.empty_like(qy)
        
        attention_cuda_extension.s2_attention_fwd(kx, vx, qy, quad_weights,
                                                  col_idx, row_off,
                                                  max_psi_nnz, nlon_in, nlat_out, nlon_out,
                                                  output)

        return output

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        col_idx, row_off, quad_weights, kx, vx, qy = ctx.saved_tensors
        max_psi_nnz = ctx.max_psi_nnz
        nlon_in = ctx.nlon_in
        nlat_out = ctx.nlat_out
        nlon_out = ctx.nlon_out

        dv = torch.empty_like(vx)
        attention_cuda_extension.s2_attention_bwd_dv_cuda(kx, vx, qy, quad_weights,
                                                          col_idx, row_off,
                                                          max_psi_nnz,
                                                          grad_output,
                                                          nlon_in, nlat_out, nlon_out,
                                                          dv)

        dk = torch.empty_like(kx)
        attention_cuda_extension.s2_attention_bwd_dk_cuda(kx, vx, qy, quad_weights,
                                                          col_idx, row_off,
                                                          max_psi_nnz,
                                                          grad_output,
                                                          nlon_in, nlat_out, nlon_out,
                                                          dk)

        dq = torch.empty_like(qy)
        attention_cuda_extension.s2_attention_bwd_dq_cuda(kx, vx, qy, quad_weights,
                                                          col_idx, row_off,
                                                          max_psi_nnz,
                                                          grad_output,
                                                          nlon_in, nlat_out, nlon_out,
                                                          dq)

        return dk, dv, dq, None, None, None, None, None, None, None


def _neighborhood_attention_s2_cuda(kx: torch.Tensor, vx: torch.Tensor, qy: torch.Tensor, quad_weights: torch.Tensor,
                                    col_idx: torch.Tensor, row_off: torch.Tensor, max_psi_nnz: int,
                                    nlon_in: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
    
    return _NeighborhoodAttentionS2Cuda.apply(kx, vx, qy, quad_weights,
                                              col_idx, row_off, max_psi_nnz,
                                              nlon_in, nlat_out, nlon_out)
