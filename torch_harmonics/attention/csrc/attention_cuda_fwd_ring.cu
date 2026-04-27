// coding=utf-8
//
// SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "attention_cuda.cuh"
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/CUDAUtils.h>

#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

#include "cudamacro.h"
#include "attention_cuda_utils.cuh"

#define THREADS (64)

// Ring-step variant of the forward attention kernel. Used by
// DistributedNeighborhoodAttentionS2: K/V are sharded along longitude across
// an azimuth process group; each call processes one rotating chunk and
// accumulates softmax state into externally-allocated buffers (y_acc,
// alpha_sum_buf, qdotk_max_buf). col_idx must have wi pre-shifted by
// pscale * lon_lo_out (see _build_local_psi in distributed_attention.py).

namespace attention_kernels {

template<int BDIM_X,
         typename FLOATV_T>
__global__
__launch_bounds__(BDIM_X)
void s2_attn_fwd_ring_step_generic_vec_k(
    int nchan_in,         // no. of FLOATV_T elements along channel dim
    int nchan_out,        // no. of FLOATV_T elements along channel dim
    int nlat_halo,        // number of lat rows in kx/vx chunk (with halo)
    int nlon_kx,          // number of lon columns in kx/vx chunk
    int nlon_in,          // GLOBAL nlon_in (for modular arithmetic)
    int pscale,           // GLOBAL pscale = nlon_in / nlon_out_global
    int lon_lo_kx,        // global lon start of kx chunk
    int lat_halo_start,   // global lat index of first row in kx chunk
    int nlat_out,         // local output lat size
    int nlon_out,         // local output lon size
    const FLOATV_T *__restrict__ kx,           // [batch][nlat_halo][nlon_kx][nchan_in]
    const FLOATV_T *__restrict__ vx,           // [batch][nlat_halo][nlon_kx][nchan_out]
    const FLOATV_T *__restrict__ qy,           // [batch][nlat_out][nlon_out][nchan_in]
    const int32_t  *__restrict__ row_idx,
    const int64_t  *__restrict__ row_off,
    const int64_t  *__restrict__ col_idx,      // wi already shifted by pscale * lon_lo_out
    const float    *__restrict__ quad_weights, // [nlat_in_global]
    FLOATV_T *__restrict__ y_acc,              // [batch][nlat_out][nlon_out][nchan_out] (in/out)
    float    *__restrict__ alpha_sum_buf,      // [batch][nlat_out][nlon_out] (in/out)
    float    *__restrict__ qdotk_max_buf       // [batch][nlat_out][nlon_out] (in/out)
) {
    extern __shared__ __align__(sizeof(float4)) float shext[];
    FLOATV_T *shy = reinterpret_cast<FLOATV_T *>(shext) + threadIdx.y * nchan_out;

    const int batch = blockIdx.y;
    const int wid   = blockIdx.x * blockDim.y + threadIdx.y;
    if (wid >= nlat_out * nlon_out) return;

    const int tidx  = threadIdx.x;
    const int h     = wid / nlon_out;
    const int wo    = wid - (h * nlon_out);   // LOCAL wo
    const int ho    = row_idx[h];

    kx += int64_t(batch) * nlat_halo * nlon_kx * nchan_in;
    vx += int64_t(batch) * nlat_halo * nlon_kx * nchan_out;
    qy += int64_t(batch) * nlat_out  * nlon_out * nchan_in
        + int64_t(ho)    * nlon_out  * nchan_in
        + int64_t(wo)    * nchan_in;

    const int64_t out_flat = int64_t(batch) * nlat_out * nlon_out
                           + int64_t(ho)    * nlon_out + wo;
    y_acc         += out_flat * nchan_out;
    alpha_sum_buf += out_flat;
    qdotk_max_buf += out_flat;

    // Load current state from buffers
    float alpha_sum = alpha_sum_buf[0];
    float qdotk_max = qdotk_max_buf[0];
    for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
        shy[chan] = y_acc[chan];
    }

    // `pscale` is the GLOBAL ratio nlon_in / nlon_out_global, passed by the caller.
    // Computing it here as `nlon_in / nlon_out` would be wrong because the kernel's
    // `nlon_out` is the LOCAL output width (nlon_out_local), not the global one.

    const int64_t rbeg = row_off[ho];
    const int64_t rend = row_off[ho + 1];
    col_idx += rbeg;
    const int rlen = rend - rbeg;

    for (int off = 0; off < rlen; off++) {
        const int64_t col = col_idx[off];

        // col_idx stores hi_global * nlon_in + wi_shifted
        // where wi_shifted = (wi_canonical + pscale * lon_lo_out) % nlon_in (baked in at Python __init__)
        const int hi_global = col / nlon_in;
        const int wi        = col - (hi_global * nlon_in);
        // wip = (wi + pscale * wo_local) % nlon_in
        //     = (wi_canonical + pscale * (lon_lo_out + wo_local)) % nlon_in
        const int wi_wo     = wi + pscale * wo;
        const int wip       = wi_wo - (wi_wo / nlon_in) * nlon_in;

        // Skip neighbors not in current kx chunk
        if (wip < lon_lo_kx || wip >= lon_lo_kx + nlon_kx) continue;

        const int hi_local  = hi_global - lat_halo_start;
        // Skip neighbors outside the halo-padded lat range (needed for distributed case)
        if (hi_local < 0 || hi_local >= nlat_halo) continue;
        const int wip_local = wip - lon_lo_kx;

        const FLOATV_T *_kx = kx + int64_t(hi_local) * nlon_kx * nchan_in  + int64_t(wip_local) * nchan_in;
        const FLOATV_T *_vx = vx + int64_t(hi_local) * nlon_kx * nchan_out + int64_t(wip_local) * nchan_out;

        FLOATV_T qdotkv = __vset<FLOATV_T>(0.f);
        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
            qdotkv = __vadd(qdotkv, __vmul(qy[chan], _kx[chan]));
        }
        float qdotk = __warp_sum(__vred(qdotkv));

        const float qdotk_max_tmp = max(qdotk_max, qdotk);
        const float alpha          = expf(qdotk - qdotk_max_tmp) * quad_weights[hi_global];
        const float exp_save       = expf(qdotk_max - qdotk_max_tmp);

        alpha_sum = alpha + alpha_sum * exp_save;
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
            shy[chan] = __vadd(__vscale(exp_save, shy[chan]),
                               __vscale(alpha,    _vx[chan]));
        }
        qdotk_max = qdotk_max_tmp;
    }

    // Store updated state back to buffers
    alpha_sum_buf[0] = alpha_sum;
    qdotk_max_buf[0] = qdotk_max;
    for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
        y_acc[chan] = shy[chan];
    }
}

static void s2_attn_fwd_ring_step_dispatch(
    int64_t batch_size,
    int64_t nchans_in,
    int64_t nchans_out,
    int64_t nlon_in,
    int64_t pscale,
    int64_t nlat_halo,
    int64_t nlon_kx,
    int64_t lon_lo_kx,
    int64_t lat_halo_start,
    int64_t nlat_out,
    int64_t nlon_out,
    at::Tensor kxP,
    at::Tensor vxP,
    at::Tensor qyP,
    at::Tensor row_idx,
    at::Tensor row_off,
    at::Tensor col_idx,
    at::Tensor quad_weights,
    at::Tensor y_acc,
    at::Tensor alpha_sum_buf,
    at::Tensor qdotk_max_buf) {

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    dim3 block(WARP_SIZE, THREADS / WARP_SIZE);
    dim3 grid(DIV_UP(nlat_out * nlon_out, block.y), batch_size);

    float *_kxp          = reinterpret_cast<float *>(kxP.data_ptr());
    float *_vxp          = reinterpret_cast<float *>(vxP.data_ptr());
    float *_qyp          = reinterpret_cast<float *>(qyP.data_ptr());
    float *_y_acc        = reinterpret_cast<float *>(y_acc.data_ptr());
    float *_alpha_sum    = reinterpret_cast<float *>(alpha_sum_buf.data_ptr());
    float *_qdotk_max   = reinterpret_cast<float *>(qdotk_max_buf.data_ptr());
    int32_t *_row_idx    = reinterpret_cast<int32_t *>(row_idx.data_ptr());
    int64_t *_row_off    = reinterpret_cast<int64_t *>(row_off.data_ptr());
    int64_t *_col_idx    = reinterpret_cast<int64_t *>(col_idx.data_ptr());
    float *_quad_weights = reinterpret_cast<float *>(quad_weights.data_ptr());

    constexpr int VEC_SIZE = sizeof(float4) / sizeof(float);

    if (!is_aligned<sizeof(float4)>(_kxp) ||
        !is_aligned<sizeof(float4)>(_vxp) ||
        !is_aligned<sizeof(float4)>(_qyp) ||
        !is_aligned<sizeof(float4)>(_y_acc) ||
        (nchans_in  % VEC_SIZE) != 0 ||
        (nchans_out % VEC_SIZE) != 0) {

        size_t shsize = sizeof(float) * nchans_out * block.y;
        s2_attn_fwd_ring_step_generic_vec_k<THREADS, float>
            <<<grid, block, shsize, stream>>>(
                nchans_in, nchans_out, nlat_halo, nlon_kx,
                nlon_in, pscale, lon_lo_kx, lat_halo_start, nlat_out, nlon_out,
                _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights,
                _y_acc, _alpha_sum, _qdotk_max);
        CHECK_ERROR("s2_attn_fwd_ring_step_generic_vec_k<float>");

    } else {

        float4 *_kxp4  = reinterpret_cast<float4 *>(_kxp);
        float4 *_vxp4  = reinterpret_cast<float4 *>(_vxp);
        float4 *_qyp4  = reinterpret_cast<float4 *>(_qyp);
        float4 *_yacc4 = reinterpret_cast<float4 *>(_y_acc);

        size_t shsize = sizeof(float4) * (nchans_out / VEC_SIZE) * block.y;
        s2_attn_fwd_ring_step_generic_vec_k<THREADS, float4>
            <<<grid, block, shsize, stream>>>(
                nchans_in / VEC_SIZE, nchans_out / VEC_SIZE,
                nlat_halo, nlon_kx, nlon_in, pscale, lon_lo_kx, lat_halo_start,
                nlat_out, nlon_out,
                _kxp4, _vxp4, _qyp4, _row_idx, _row_off, _col_idx, _quad_weights,
                _yacc4, _alpha_sum, _qdotk_max);
        CHECK_ERROR("s2_attn_fwd_ring_step_generic_vec_k<float4>");
    }
}

void s2_attention_fwd_ring_step_cuda(
    at::Tensor kx,
    at::Tensor vx,
    at::Tensor qy,
    at::Tensor y_acc,
    at::Tensor alpha_sum_buf,
    at::Tensor qdotk_max_buf,
    at::Tensor quad_weights,
    at::Tensor psi_col_idx,
    at::Tensor psi_row_off,
    at::Tensor psi_row_idx,
    int64_t nlon_in,
    int64_t pscale,
    int64_t lon_lo_kx,
    int64_t lat_halo_start,
    int64_t nlat_out,
    int64_t nlon_out)
{
    CHECK_CUDA_INPUT_TENSOR(kx);
    CHECK_CUDA_INPUT_TENSOR(vx);
    CHECK_CUDA_INPUT_TENSOR(qy);
    CHECK_CUDA_TENSOR(y_acc);
    CHECK_CUDA_TENSOR(alpha_sum_buf);
    CHECK_CUDA_TENSOR(qdotk_max_buf);
    CHECK_CUDA_TENSOR(quad_weights);
    CHECK_CUDA_TENSOR(psi_col_idx);
    CHECK_CUDA_TENSOR(psi_row_off);
    CHECK_CUDA_TENSOR(psi_row_idx);

    const int batch_size = kx.size(0);
    const int nlat_halo  = kx.size(2);   // kx is [B,C,H,W], H is dim 2
    const int nlon_kx    = kx.size(3);   // W is dim 3
    const size_t nchans_in  = qy.size(1);
    const size_t nchans_out = vx.size(1);

    torch::Tensor kxP = kx.to(torch::kFloat32);
    torch::Tensor vxP = vx.to(torch::kFloat32);
    torch::Tensor qyP = qy.to(torch::kFloat32);

    bool kx_is_channels_last = kxP.strides()[1] == 1;
    bool vx_is_channels_last = vxP.strides()[1] == 1;
    bool qy_is_channels_last = qyP.strides()[1] == 1;

    if (!kx_is_channels_last) { kxP = permute_4D_to0231(kxP); }
    if (!vx_is_channels_last) { vxP = permute_4D_to0231(vxP); }
    if (!qy_is_channels_last) { qyP = permute_4D_to0231(qyP); }

    s2_attn_fwd_ring_step_dispatch(
        batch_size, nchans_in, nchans_out,
        nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start,
        nlat_out, nlon_out,
        kxP, vxP, qyP,
        psi_row_idx, psi_row_off, psi_col_idx,
        quad_weights,
        y_acc, alpha_sum_buf, qdotk_max_buf);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY_IMPL(attention_kernels, CUDA, m)
{
    m.impl("forward_ring_step", &s2_attention_fwd_ring_step_cuda);
}

}
