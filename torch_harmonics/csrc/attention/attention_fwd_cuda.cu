// coding=utf-8
//
// SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
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

#include "attention.cuh"
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/CUDAUtils.h>

#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

#include "cudamacro.h"
#include "attention_utils.cuh"

#define THREADS (64)

#define MAX_LOCAL_ARR_LEN (16)

// BEGIN - forward kernels and functions

// called with (blockDim.x=32 and blockDim.y>1, BDIM_X=blockDim.x*blockDim.y)
template<int BDIM_X,
         typename FLOATV_T> // either float or float4
__global__
__launch_bounds__(BDIM_X)
void s2_attn_fwd_generic_vec_k(int nchan,  // no. of FLOATV_T elements along channel dim
                               int nlat_in,
                               int nlon_in,
                               int nlat_out,
                               int nlon_out,
                               const FLOATV_T *__restrict__ kx,
                               const FLOATV_T *__restrict__ vx,
                               const FLOATV_T *__restrict__ qy,
                               const int32_t *__restrict__ row_idx,
                               const int64_t *__restrict__ row_off,
                               const int64_t *__restrict__ col_idx,
                               const   float *__restrict__ quad_weights,
                                     FLOATV_T *__restrict__ y) {

    extern __shared__ __align__(sizeof(float4)) float shext[];
    FLOATV_T *shy = reinterpret_cast<FLOATV_T *>(shext) + threadIdx.y*nchan;

    const int batch = blockIdx.y;
    const int wid = blockIdx.x*blockDim.y + threadIdx.y;

    if (wid >= nlat_out*nlon_out) {
        return;
    }

    const int tidx = threadIdx.x;

    const int h = wid / nlon_out;
    const int wo = wid - (h*nlon_out);
    const int ho = row_idx[h];

    for(int chan = tidx; chan < nchan; chan += WARP_SIZE) {
        shy[chan] = __vset<FLOATV_T>(0.f);
    }

    kx += int64_t(batch)*nlat_in*nlon_in*nchan;
    vx += int64_t(batch)*nlat_in*nlon_in*nchan;
    qy += int64_t(batch)*nlat_out*nlon_out*nchan + int64_t(ho)*nchan*nlon_out + int64_t(wo)*nchan;
    y  += int64_t(batch)*nlat_out*nlon_out*nchan + int64_t(ho)*nchan*nlon_out + int64_t(wo)*nchan;

    float alpha_sum = 0.0f;
    float qdotk_max = -FLT_MAX;

    const int64_t rbeg = row_off[ho];
    const int64_t rend = row_off[ho+1];

    col_idx += rbeg;

    const int rlen = rend-rbeg;

    for(int off = 0; off < rlen; off++) {

        const int64_t col = col_idx[off];

        const int hi = col / nlon_in;
        const int wi = col - (hi*nlon_in);
        const int wip = (wi+wo) - ((wi+wo) / nlon_in) * nlon_in;

        const FLOATV_T *_kx = kx + int64_t(hi)*nlon_in*nchan + int64_t(wip)*nchan;
        const FLOATV_T *_vx = vx + int64_t(hi)*nlon_in*nchan + int64_t(wip)*nchan;

        FLOATV_T qdotkv = __vset<FLOATV_T>(0.f);

        for(int chan = tidx; chan < nchan; chan += WARP_SIZE) {
            qdotkv = __vadd(qdotkv,
                            __vmul( qy[chan],
                                   _kx[chan]));
        }

        float qdotk = __warp_sum(__vred(qdotkv));

        float qdotk_max_tmp;
        float alpha;
        float exp_save;

        qdotk_max_tmp = max(qdotk_max, qdotk);
        alpha = expf(qdotk - qdotk_max_tmp) * quad_weights[hi];
        exp_save = expf(qdotk_max - qdotk_max_tmp);

        alpha_sum = alpha + alpha_sum*exp_save;

        for(int chan = tidx; chan < nchan; chan += WARP_SIZE) {
            shy[chan] = __vadd(__vscale(exp_save, shy[chan]),
                               __vscale(   alpha, _vx[chan]));
        }
        qdotk_max = qdotk_max_tmp;
    }

    alpha_sum = 1.0f / alpha_sum;
    for(int chan = tidx; chan < nchan; chan += WARP_SIZE) {
        y[chan] = __vscale(alpha_sum, shy[chan]);
    }

    return;
}

// called with either (BDIM_X=32 and BDIM_Y>1) || (2^K=BDIM_X > 32 and BDIM_Y=1)
template<int BDIM_X,
         int BDIM_Y,
         int NLOC,
         typename FLOATV_T> // either float or float4
__global__
__launch_bounds__(BDIM_X*BDIM_Y)
void s2_attn_fwd_special_vec_k(int nchan, // no. of FLOATV_T elements along channel dim
                               int nlat_in,
                               int nlon_in,
                               int nlat_out,
                               int nlon_out,
                               const FLOATV_T *__restrict__ kx,
                               const FLOATV_T *__restrict__ vx,
                               const FLOATV_T *__restrict__ qy,
                               const int32_t *__restrict__ row_idx,
                               const int64_t *__restrict__ row_off,
                               const int64_t *__restrict__ col_idx,
                               const   float *__restrict__ quad_weights,
                                     FLOATV_T *__restrict__ y) {

    static_assert(0 == (BDIM_X & (BDIM_X-1)));
    static_assert(0 == (BDIM_Y & (BDIM_Y-1)));
    static_assert((BDIM_X == 32 && BDIM_Y  > 1) ||
                  (BDIM_X  > 32 && BDIM_Y == 1)) ;

    constexpr int NLOC_M1 = NLOC-1;

    const int tidx = threadIdx.x;
    const int batch = blockIdx.y;
    const int ctaid = blockIdx.x*blockDim.y + threadIdx.y;

    if (ctaid >= nlat_out*nlon_out) {
        return;
    }

    FLOATV_T locy[NLOC];

    extern __shared__ __align__(sizeof(float4)) float shext[];
    FLOATV_T *shq = reinterpret_cast<FLOATV_T *>(shext) + threadIdx.y*nchan + tidx;

    const int h = ctaid / nlon_out;
    const int wo = ctaid - (h*nlon_out);
    const int ho = row_idx[h];

    kx += int64_t(batch)*nlat_in*nlon_in*nchan + tidx;
    vx += int64_t(batch)*nlat_in*nlon_in*nchan + tidx;
    qy += int64_t(batch)*nlat_out*nlon_out*nchan + int64_t(ho)*nlon_out*nchan + int64_t(wo)*nchan + tidx;
    y  += int64_t(batch)*nlat_out*nlon_out*nchan + int64_t(ho)*nlon_out*nchan + int64_t(wo)*nchan + tidx;

    #pragma unroll
    for(int i = 0; i < NLOC; i++) {
        locy[i] = __vset<FLOATV_T>(0.f);
    }

    #pragma unroll
    for(int i = 0; i < NLOC_M1; i++) {
        shq[i*BDIM_X] = qy[i*BDIM_X];
    }
    if (NLOC_M1*BDIM_X+tidx < nchan) {
        shq[NLOC_M1*BDIM_X] = qy[NLOC_M1*BDIM_X];
    }

    float alpha_sum = 0.0f;
    float qdotk_max = -FLT_MAX;

    const int64_t rbeg = row_off[ho];
    const int64_t rend = row_off[ho+1];

    col_idx += rbeg;

    const int rlen = rend-rbeg;

    for(int off = 0; off < rlen; off++) {

        const int64_t col = col_idx[off];

        const int hi = col / nlon_in;
        const int wi = col - (hi*nlon_in);
        const int wip = (wi+wo) - ((wi+wo) / nlon_in) * nlon_in;

        const FLOATV_T *_kx = kx + int64_t(hi)*nlon_in*nchan + int64_t(wip)*nchan;
        const FLOATV_T *_vx = vx + int64_t(hi)*nlon_in*nchan + int64_t(wip)*nchan;

        FLOATV_T qdotkv = __vset<FLOATV_T>(0.f);

        #pragma unroll
        for(int i = 0; i < NLOC_M1; i++) {
            qdotkv = __vadd(qdotkv,
                            __vmul(shq[i*BDIM_X],
                                   _kx[i*BDIM_X]));
        }
        if (NLOC_M1*BDIM_X+tidx < nchan) {
            qdotkv = __vadd(qdotkv,
                            __vmul(shq[NLOC_M1*BDIM_X],
                                   _kx[NLOC_M1*BDIM_X]));
        }

        float qdotk = __vred(qdotkv);
        if constexpr(BDIM_X == 32) { qdotk =          __warp_sum(qdotk); }
        else                       { qdotk = __block_sum<BDIM_X>(qdotk); }

        float qdotk_max_tmp;
        float alpha;
        float exp_save;

        qdotk_max_tmp = max(qdotk_max, qdotk);
        alpha = expf(qdotk - qdotk_max_tmp) * quad_weights[hi];
        exp_save = expf(qdotk_max - qdotk_max_tmp);

        alpha_sum = alpha + alpha_sum*exp_save;

        #pragma unroll
        for(int i = 0; i < NLOC_M1; i++) {
            locy[i] = __vadd(__vscale(exp_save, locy[i]),
                             __vscale(alpha, _vx[i*BDIM_X]));
        }
        if (NLOC_M1*BDIM_X+tidx < nchan) {
            locy[NLOC_M1] = __vadd(__vscale(exp_save, locy[NLOC_M1]),
                                   __vscale(alpha, _vx[NLOC_M1*BDIM_X]));
        }

        qdotk_max = qdotk_max_tmp;
    }

    alpha_sum = 1.0f / alpha_sum;

    #pragma unroll
    for(int i = 0; i < NLOC_M1; i++) {
        y[i*BDIM_X] = __vscale(alpha_sum, locy[i]);
    }
    if (NLOC_M1*BDIM_X+tidx < nchan) {
        y[NLOC_M1*BDIM_X] = __vscale(alpha_sum, locy[NLOC_M1]);
    }

    return;
}

template<typename FLOATV_T>
void launch_gen_attn_fwd(int batch_size,
                         int nchans,
                         int nlat_in,
                         int nlon_in,
                         int nlat_out,
                         int nlon_out,
                         FLOATV_T *__restrict__ _kxp,
                         FLOATV_T *__restrict__ _vxp,
                         FLOATV_T *__restrict__ _qyp,
                         int32_t *_row_idx,
                         int64_t *_row_off,
                         int64_t *_col_idx,
                         float *_quad_weights,
                         FLOATV_T *__restrict__ _yp,
                         cudaStream_t stream) {

    dim3 block(WARP_SIZE, THREADS/WARP_SIZE);
    dim3 grid(DIV_UP(nlat_out*nlon_out, block.y), batch_size);

    size_t shsize = sizeof(FLOATV_T)*nchans * block.y;

    s2_attn_fwd_generic_vec_k<THREADS>
                             <<<grid, block, shsize, stream>>>(nchans, nlat_in, nlon_in, nlat_out, nlon_out,
                                                               _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp);
    CHECK_ERROR("s2_attn_fwd_generic_vec_k");

    return;
}

template<int BDIM_X,
         int BDIM_Y,
         int CUR_LOC_SIZE,
         int MAX_LOC_SIZE, // max size of FLOATV_T[] local array
         typename FLOATV_T>
void launch_spc_attn_fwd(int batch_size,
                         int nloc,      // "BDIM_X*nloc" >= nchans
                         int nchans,
                         int nlat_in,
                         int nlon_in,
                         int nlat_out,
                         int nlon_out,
                         FLOATV_T *__restrict__ _kxp,
                         FLOATV_T *__restrict__ _vxp,
                         FLOATV_T *__restrict__ _qyp,
                         int32_t *_row_idx,
                         int64_t *_row_off,
                         int64_t *_col_idx,
                         float *_quad_weights,
                         FLOATV_T *__restrict__ _yp,
                         cudaStream_t stream) {

    if (CUR_LOC_SIZE == nloc) {

        dim3 block(BDIM_X, BDIM_Y);
        dim3 grid(DIV_UP(nlat_out*nlon_out, block.y), batch_size);

        size_t shsize = sizeof(FLOATV_T)*nchans * block.y; // block.y > 1 iif block.x==32

        s2_attn_fwd_special_vec_k<BDIM_X, BDIM_Y, CUR_LOC_SIZE>
                                 <<<grid, block, shsize, stream>>>(nchans, nlat_in, nlon_in, nlat_out, nlon_out,
                                                                   _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp);
        CHECK_ERROR("s2_attn_fwd_special_vec_k");

        return;
    }
    if constexpr(CUR_LOC_SIZE < MAX_LOC_SIZE) {
        launch_spc_attn_fwd<BDIM_X,
                             BDIM_Y,
                             CUR_LOC_SIZE+1,
                             MAX_LOC_SIZE>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out,
                                           _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp,
                                           stream);
    }
    return;
}

static void s2_attn_fwd_dispatch(int batch_size,
                                 int nchans,
                                 int nlon_in,
                                 int nlat_out,
                                 int nlon_out,
                                 at::Tensor kxP,
                                 at::Tensor vxP,
                                 at::Tensor qyP,
                                 at::Tensor row_off,
                                 at::Tensor col_idx,
                                 at::Tensor quad_weights,
                                 at::Tensor yP,
                                 cudaStream_t stream) {

    static_assert(0 == (MAX_LOCAL_ARR_LEN & (MAX_LOCAL_ARR_LEN-1)));

    // sort row indices (ho-s) in descending order
    // based on (row_off[ho+1]-row_off[ho])
    at::Tensor row_idx = sortRows(nlat_out, row_off, stream);

    const int nlat_in = kxP.size(1);

    // smallest power of two "bdimx" (>=32) s.t. bdimx*MAX_LOCAL_ARR_LEN >= nchans
    int bdimx;
    bdimx = DIV_UP(nchans, MAX_LOCAL_ARR_LEN);
    bdimx = max(bdimx, WARP_SIZE);
    bdimx = next_pow2(bdimx);

    float *_kxp = reinterpret_cast<float *>(kxP.data_ptr());
    float *_vxp = reinterpret_cast<float *>(vxP.data_ptr());
    float *_qyp = reinterpret_cast<float *>(qyP.data_ptr());
    float *_yp  = reinterpret_cast<float *>(yP.data_ptr());

    int32_t *_row_idx = reinterpret_cast<int32_t *>(row_idx.data_ptr());
    int64_t *_row_off = reinterpret_cast<int64_t *>(row_off.data_ptr());
    int64_t *_col_idx = reinterpret_cast<int64_t *>(col_idx.data_ptr());
    float *_quad_weights = reinterpret_cast<float *>(quad_weights.data_ptr());

    constexpr int VEC_SIZE = sizeof(float4) / sizeof(float);

    if (!is_aligned<sizeof(float4)>(_kxp) ||
        !is_aligned<sizeof(float4)>(_vxp) ||
        !is_aligned<sizeof(float4)>(_qyp) ||
        !is_aligned<sizeof(float4)>(_yp)  ||
        (nchans % VEC_SIZE) != 0) {

        const int nloc = DIV_UP(nchans, bdimx);
        
        // to avoid the compilation of unused template instances;
        // we use a block size BDIM_X that is the smallest power of 2
        // such that BDIM_X*MAX_LOCAL_ARR_LEN >= nchans, so
        // BDIM_X > 32 are used only for:
        //
        //  (BDIM_X-1)*MAX_LOCAL_ARR_LEN < nchans <= BDIM_X*MAX_LOCAL_ARR_LEN
        constexpr int MIN_LOC_ARR_LEN = MAX_LOCAL_ARR_LEN/2+1;

        // use 2D blocks only if 32 threads are enough
        switch(bdimx) {
            case   32: launch_spc_attn_fwd<  32, 2,               1, MAX_LOCAL_ARR_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp, stream); break;
            case   64: launch_spc_attn_fwd<  64, 1, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp, stream); break;
            case  128: launch_spc_attn_fwd< 128, 1, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp, stream); break;
            case  256: launch_spc_attn_fwd< 256, 1, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp, stream); break;
            case  512: launch_spc_attn_fwd< 512, 1, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp, stream); break;
            case 1024: launch_spc_attn_fwd<1024, 1, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp, stream); break;
            default:   launch_gen_attn_fwd                                             (batch_size,       nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp, stream); break;
        }

    } else {

        float4 *_kxp4 = reinterpret_cast<float4 *>(_kxp);
        float4 *_vxp4 = reinterpret_cast<float4 *>(_vxp);
        float4 *_qyp4 = reinterpret_cast<float4 *>(_qyp);
        float4 *_yp4  = reinterpret_cast<float4 *>(_yp);

        nchans /= VEC_SIZE;
        const int nloc = DIV_UP(nchans, bdimx);

        constexpr int MAX_LOCAL_VEC_LEN = MAX_LOCAL_ARR_LEN / VEC_SIZE;
        
        constexpr int MIN_LOC_VEC_LEN = MAX_LOCAL_VEC_LEN/2+1;

        // use 2D blocks only if 32 threads are enough
        switch(bdimx) {
            case   32: launch_spc_attn_fwd<  32, 2,               1, MAX_LOCAL_VEC_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, _row_idx, _row_off, _col_idx, _quad_weights, _yp4, stream); break;
            case   64: launch_spc_attn_fwd<  64, 1, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, _row_idx, _row_off, _col_idx, _quad_weights, _yp4, stream); break;
            case  128: launch_spc_attn_fwd< 128, 1, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, _row_idx, _row_off, _col_idx, _quad_weights, _yp4, stream); break;
            case  256: launch_spc_attn_fwd< 256, 1, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, _row_idx, _row_off, _col_idx, _quad_weights, _yp4, stream); break;
            case  512: launch_spc_attn_fwd< 512, 1, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, _row_idx, _row_off, _col_idx, _quad_weights, _yp4, stream); break;
            case 1024: launch_spc_attn_fwd<1024, 1, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, _row_idx, _row_off, _col_idx, _quad_weights, _yp4, stream); break;
            default:   launch_gen_attn_fwd                                             (batch_size,       nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, _row_idx, _row_off, _col_idx, _quad_weights, _yp4, stream); break;
        }
    }

    return;
}

// END - forward kernels and functions

torch::Tensor s2_attention_fwd_cuda(at::Tensor kx,
                                    at::Tensor vx,
                                    at::Tensor qy,
                                    at::Tensor quad_weights,
                                    at::Tensor psi_col_idx,
                                    at::Tensor psi_row_off,
                                    int nlon_in,
                                    int nlat_out,
                                    int nlon_out) {
    CHECK_CUDA_TENSOR(kx);
    CHECK_CUDA_TENSOR(vx);
    CHECK_CUDA_TENSOR(qy);
    CHECK_CUDA_TENSOR(quad_weights);
    CHECK_CUDA_TENSOR(psi_col_idx);
    CHECK_CUDA_TENSOR(psi_row_off);

    // TODO: check sizes

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    size_t uo_num_channels = kx.size(1);

    const int batch_size = kx.size(0);

    torch::Tensor kxP = kx;
    torch::Tensor vxP = vx;
    torch::Tensor qyP = qy;

    auto k_channel_first = kx.strides()[1] == 1;
    auto v_channel_first = vx.strides()[1] == 1;
    auto q_channel_first = qy.strides()[1] == 1;

    if (!k_channel_first) { kxP = permute_4D_floatT_to0231(kx, stream); }
    if (!v_channel_first) { vxP = permute_4D_floatT_to0231(vx, stream); }
    if (!q_channel_first) { qyP = permute_4D_floatT_to0231(qy, stream); }

    torch::Tensor yP = torch::empty_like(qyP);

    s2_attn_fwd_dispatch(batch_size,
                         uo_num_channels,
                         nlon_in,
                         nlat_out,
                         nlon_out,
                         kxP, vxP, qyP,
                         psi_row_off,
                         psi_col_idx,
                         quad_weights,
                         yP, // out tensor
                         stream);

    torch::Tensor y = yP;
    if (!q_channel_first) { y = permute_4D_floatT_to0312(yP, stream); }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}
