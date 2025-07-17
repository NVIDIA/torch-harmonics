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
#include "c10/core/MemoryFormat.h"

#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/CUDAUtils.h>

#include <ctime>
#include <cub/cub.cuh>
#include <limits>

#include "cudamacro.h"
#include "attention_utils.cuh"

#include <iostream>
#include <chrono>
#include <string>

#define THREADS (64)

#define MAX_LOCAL_ARR_LEN (16)

namespace attention_kernels {

#if 0
class ScopeTimer
{
  public:
    explicit ScopeTimer(const std::string &label = "") :
        label_(label), start_(std::chrono::high_resolution_clock::now())
    {
    }

    ~ScopeTimer()
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
        std::cout << label_ << "Elapsed time: " << elapsed.count() << " ms" << std::endl;
    }

  private:
    std::string label_;
    std::chrono::high_resolution_clock::time_point start_;
};

// easier to understand version of manual shfl_xor_sync, performance appears similar
static __device__ float __warp_sum_cub(float val)
{
    // use cub to reduce within a warp
    __shared__ typename cub::WarpReduce<float>::TempStorage temp_storage;

    // 1. Compute sum (initially only in lane 0)
    float sum = cub::WarpReduce<float>(temp_storage).Sum(val);
    // 2. Broadcast sum to all threads
    sum = __shfl_sync(0xFFFFFFFF, sum, 0);
    return sum;
}

// This kernel computes the backward pass for the S2 attention mechanism, using
// shared memory as a cache and one warp per output point, warp-parallel over
// channels, which should be layed out in the fastest dimension for coalesced
// memory access.
template <int BDIM_X>
__global__ __launch_bounds__(BDIM_X) void s2_attention_bwd_dkvq_kernel(
    int num_channels, int nlon_in, int nlat_out, int nlon_out,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> kx,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> vx,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> qy,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> dy,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> dydk,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> dydv,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> dydq,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> psi_col_idx,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> psi_row_offset,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> quad_weights)
{

    extern __shared__ float sh[];
    float *sh_alpha_k = sh + threadIdx.y * num_channels * 5;
    float *sh_alpha_vw = sh_alpha_k + num_channels;
    float *sh_alpha_kvw = sh_alpha_vw + num_channels;
    float *sh_dy = sh_alpha_kvw + num_channels;
    float *sh_qy = sh_dy + num_channels;
    // (optionally, could use more shared memory for other intermediates)

    const uint64_t batchId = blockIdx.y;
    const uint64_t wid = uint64_t(blockIdx.x) * blockDim.y + threadIdx.y;
    if (wid >= uint64_t(nlat_out) * nlon_in) return;
    const int tidx = threadIdx.x;
    const int ho = wid / nlon_out;
    const int wo = wid - (ho * nlon_out);

    // Zero shared memory
    for (int chan = tidx; chan < num_channels; chan += WARP_SIZE) {
        sh_alpha_k[chan] = 0.0f;
        sh_alpha_vw[chan] = 0.0f;
        sh_alpha_kvw[chan] = 0.0f;
        sh_dy[chan] = dy[batchId][chan][ho][wo];
        sh_qy[chan] = qy[batchId][chan][ho][wo];
    }
    float alpha_sum = 0.0f;
    float qdotk_max = -FLT_MAX;
    float integral = 0.0f;
    __syncthreads();

    const int64_t rbeg = psi_row_offset[ho];
    const int64_t rend = psi_row_offset[ho + 1];
    const int rlen = rend - rbeg;

    // 1st pass: accumulate alpha_sum, integral, and shared stats, along with a progressively computed qdotk_max.
    for (int off = 0; off < rlen; off++) {
        const int64_t col = psi_col_idx[rbeg + off];
        const int hi = col / nlon_in;
        const int wi = col - (hi * nlon_in);
        const int wip = (wi + wo) - ((wi + wo) / nlon_in) * nlon_in;
        float qdotk = 0.0f, gdotv = 0.0f;
        for (int chan = tidx; chan < num_channels; chan += WARP_SIZE) {
            qdotk += sh_qy[chan] * kx[batchId][chan][hi][wip];
            gdotv += sh_dy[chan] * vx[batchId][chan][hi][wip];
        }
        qdotk = __warp_sum_cub(qdotk);
        gdotv = __warp_sum_cub(gdotv);
        float qdotk_max_tmp = max(qdotk_max, qdotk);
        float alpha_inz = expf(qdotk - qdotk_max_tmp) * quad_weights[hi];
        float max_correction = expf(qdotk_max - qdotk_max_tmp);
        alpha_sum = alpha_sum * max_correction + alpha_inz;
        integral = integral * max_correction + alpha_inz * gdotv;
        for (int chan = tidx; chan < num_channels; chan += WARP_SIZE) {
            float kxval = kx[batchId][chan][hi][wip];
            sh_alpha_k[chan] = sh_alpha_k[chan] * max_correction + alpha_inz * kxval;
            sh_alpha_vw[chan] = sh_alpha_vw[chan] * max_correction + alpha_inz * gdotv;
            sh_alpha_kvw[chan] = sh_alpha_kvw[chan] * max_correction + alpha_inz * kxval * gdotv;
        }
        qdotk_max = qdotk_max_tmp;
    }

    integral /= alpha_sum;

    // Write dydq
    for (int chan = tidx; chan < num_channels; chan += WARP_SIZE) {
        dydq[batchId][chan][ho][wo]
            = (sh_alpha_kvw[chan] * alpha_sum - sh_alpha_vw[chan] * sh_alpha_k[chan]) / (alpha_sum * alpha_sum);
    }

    // Third pass: accumulate gradients for k and v
    for (int off = 0; off < rlen; off++) {
        const int64_t col = psi_col_idx[rbeg + off];
        const int hi = col / nlon_in;
        const int wi = col - (hi * nlon_in);
        const int wip = (wi + wo) - ((wi + wo) / nlon_in) * nlon_in;
        float qdotk = 0.0f, gdotv = 0.0f;
        for (int chan = tidx; chan < num_channels; chan += WARP_SIZE) {
            qdotk += qy[batchId][chan][ho][wo] * kx[batchId][chan][hi][wip];
            gdotv += sh_dy[chan] * vx[batchId][chan][hi][wip];
        }
        qdotk = __warp_sum_cub(qdotk);
        gdotv = __warp_sum_cub(gdotv);
        float alpha_inz = expf(qdotk - qdotk_max) * quad_weights[hi];
        for (int chan = tidx; chan < num_channels; chan += WARP_SIZE) {
            float qyval = qy[batchId][chan][ho][wo];
            float dyval = sh_dy[chan];
            atomicAdd(&dydk[batchId][chan][hi][wip], qyval * (alpha_inz / alpha_sum) * (gdotv - integral));
            atomicAdd(&dydv[batchId][chan][hi][wip], (alpha_inz / alpha_sum) * dyval);
        }
    }
}
#endif

// BEGIN backward kernels and functions

// called with (blockDim.x=32 and blockDim.y>1, BDIM=blockDim.x*blockDim.y)
template<int BDIM_X,
         typename FLOATV_T> // either float or float4
__global__
__launch_bounds__(BDIM_X)
void s2_attn_bwd_generic_vec_k(int nchan,  // no. of FLOATV_T elements along channel dim
                               int nlat_in,
                               int nlon_in,
                               int nlat_out,
                               int nlon_out,
                               const FLOATV_T *__restrict__ kx,
                               const FLOATV_T *__restrict__ vx,
                               const FLOATV_T *__restrict__ qy,
                               const FLOATV_T *__restrict__ dy,
                               const int32_t *__restrict__ row_idx,
                               const int64_t *__restrict__ row_off,
                               const int64_t *__restrict__ col_idx,
                               const   float *__restrict__ quad_weights,
                                     FLOATV_T *__restrict__ dkx,
                                     FLOATV_T *__restrict__ dvx,
                                     FLOATV_T *__restrict__ dqy) {

    extern __shared__ __align__(sizeof(float4)) float shext[];

    // for dqy
    FLOATV_T *sh_alpha_k__ = reinterpret_cast<FLOATV_T *>(shext) + threadIdx.y * nchan*5;
    FLOATV_T *sh_alpha_vw_ = sh_alpha_k__ + nchan;
    FLOATV_T *sh_alpha_kvw = sh_alpha_vw_ + nchan;

    FLOATV_T *sh_dy        = sh_alpha_kvw + nchan;
    FLOATV_T *sh_qy        = sh_dy        + nchan;

    const int batch = blockIdx.y;

    const uint64_t wid = uint64_t(blockIdx.x) * blockDim.y + threadIdx.y;
    if (wid >= uint64_t(nlat_out)*nlon_in) {
        return;
    }

    const int tidx = threadIdx.x;

    // use permuted rows
    const int h = wid / nlon_out;
    const int wo = wid - (h*nlon_out);
    const int ho = row_idx[h];

    // offset input tensors
    kx += int64_t(batch)*nlat_in*nlon_in*nchan;
    vx += int64_t(batch)*nlat_in*nlon_in*nchan;
    qy += int64_t(batch)*nlat_out*nlon_out*nchan + int64_t(ho)*nlon_out*nchan + int64_t(wo)*nchan;
    dy += int64_t(batch)*nlat_out*nlon_out*nchan + int64_t(ho)*nlon_out*nchan + int64_t(wo)*nchan;

    // offset output tensors
    dkx += int64_t(batch)*nlat_in*nlon_in*nchan;
    dvx += int64_t(batch)*nlat_in*nlon_in*nchan;
    dqy += int64_t(batch)*nlat_out*nlon_out*nchan + int64_t(ho)*nlon_out*nchan + int64_t(wo)*nchan;

    // zero/init shared memory
    for (int chan = tidx; chan < nchan; chan += WARP_SIZE) {

        sh_alpha_k__[chan] = __vset<FLOATV_T>(0.0f);
        sh_alpha_vw_[chan] = __vset<FLOATV_T>(0.0f);
        sh_alpha_kvw[chan] = __vset<FLOATV_T>(0.0f);

        sh_dy[chan] = dy[chan];
        sh_qy[chan] = qy[chan];
    }

#if __CUDA_ARCH__ < 900
    // for architectures < 9.0, sh_dy and sh_qy will be read
    // as individual floats at the end of the kernel, which
    // breaks the assumption that each FLOATV_T location is
    // written to and read by the same thread throughout the 
    // kernel, in the case FLOATV_T==float4
    if constexpr(std::is_same<FLOATV_T, float4>::value) { __syncwarp(); }
#endif

    // for dkx, dvx, dqy
    float alpha_sum = 0.0f;
    float qdotk_max = -FLT_MAX;
    
    // for dkx
    float integral = 0.0f;

    const int64_t rbeg = row_off[ho];
    const int64_t rend = row_off[ho+1];

    col_idx += rbeg;

    const int rlen = rend - rbeg;

    // accumulate alpha_sum, integral, and shared stats, 
    // along with a progressively computed qdotk_max.
    for (int off = 0; off < rlen; off++) {
        
        const int64_t col = col_idx[off];

        const int hi = col / nlon_in;
        const int wi = col - (hi * nlon_in);
        const int wip = (wi + wo) - ((wi + wo) / nlon_in) * nlon_in;

        const FLOATV_T *_kx = kx + int64_t(hi)*nlon_in*nchan + int64_t(wip)*nchan;
        const FLOATV_T *_vx = vx + int64_t(hi)*nlon_in*nchan + int64_t(wip)*nchan;

        FLOATV_T qdotk_v = __vset<FLOATV_T>(0.0f);
        FLOATV_T gdotv_v = __vset<FLOATV_T>(0.0f);

        for(int chan = tidx; chan < nchan; chan += WARP_SIZE) {

            qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[chan], _kx[chan]));
            gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[chan], _vx[chan]));
        }

        const float qdotk = __warp_sum(__vred(qdotk_v));
        const float gdotv = __warp_sum(__vred(gdotv_v));

        const float qdotk_max_tmp = max(qdotk_max, qdotk);
        const float alpha_inz = expf(qdotk - qdotk_max_tmp) * quad_weights[hi];
        const float max_correction = expf(qdotk_max - qdotk_max_tmp);
        alpha_sum = alpha_sum * max_correction + alpha_inz;

        integral = integral * max_correction + alpha_inz * gdotv;

        const float ainz_gdotv = alpha_inz * gdotv;

        for (int chan = tidx; chan < nchan; chan += WARP_SIZE) {

            const FLOATV_T kxval = _kx[chan];

            sh_alpha_k__[chan] = __vadd(__vscale(max_correction, sh_alpha_k__[chan]), __vscale(alpha_inz, kxval));
            sh_alpha_vw_[chan] = __vadd(__vscale(max_correction, sh_alpha_vw_[chan]), __vset<FLOATV_T>(ainz_gdotv));
            sh_alpha_kvw[chan] = __vadd(__vscale(max_correction, sh_alpha_kvw[chan]), __vscale(ainz_gdotv, kxval));
        }
        qdotk_max = qdotk_max_tmp;
    }

    const float alpha_sum_inv = 1.0f / alpha_sum;

    integral *= alpha_sum_inv;

    // Write dqy
    for (int chan = tidx; chan < nchan; chan += WARP_SIZE) {

        dqy[chan] = __vscale(alpha_sum_inv * alpha_sum_inv,
                             __vsub(__vscale(alpha_sum, sh_alpha_kvw[chan]),
                                    __vmul(sh_alpha_vw_[chan], sh_alpha_k__[chan])));
    }

    // accumulate gradients for k and v
    for (int off = 0; off < rlen; off++) {

        const int64_t col = col_idx[off];
        const int hi = col / nlon_in;
        const int wi = col - (hi * nlon_in);
        const int wip = (wi + wo) - ((wi + wo) / nlon_in) * nlon_in;

        const FLOATV_T *_kx = kx + int64_t(hi)*nlon_in*nchan + int64_t(wip)*nchan;
        const FLOATV_T *_vx = vx + int64_t(hi)*nlon_in*nchan + int64_t(wip)*nchan;

        FLOATV_T qdotk_v = __vset<FLOATV_T>(0.0f);
        FLOATV_T gdotv_v = __vset<FLOATV_T>(0.0f);

        for (int chan = tidx; chan < nchan; chan += WARP_SIZE) {

            qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[chan], _kx[chan]));
            gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[chan], _vx[chan]));
        }

        const float qdotk = __warp_sum(__vred(qdotk_v));
        const float gdotv = __warp_sum(__vred(gdotv_v));

        const float alpha_inz = expf(qdotk - qdotk_max) * quad_weights[hi];

        FLOATV_T *_dkx = dkx + int64_t(hi)*nlon_in*nchan + int64_t(wip)*nchan;
        FLOATV_T *_dvx = dvx + int64_t(hi)*nlon_in*nchan + int64_t(wip)*nchan;

        const float alpha_mul = alpha_inz * alpha_sum_inv;

        const float scale_fact_qy = (gdotv - integral)*alpha_mul;
        const float scale_fact_dy =                    alpha_mul;

        // float4, 128-bit atomics are only supported by devices of compute 
        // capability 9.x+, so on older devices we resort to 32-bit atomics

#if __CUDA_ARCH__ < 900
        // to use 32-bit operations on consecutve addresses
        float *sh_qy_scl = reinterpret_cast<float *>(sh_qy);
        float *sh_dy_scl = reinterpret_cast<float *>(sh_dy);

        float *_dkx_scl = reinterpret_cast<float *>(_dkx);
        float *_dvx_scl = reinterpret_cast<float *>(_dvx);

        constexpr int VEC_SIZE = sizeof(FLOATV_T)/sizeof(float);

        // 32-bit, consecutive atomics to glmem;
        // strided atomics results in a severe slowdown
        for (int chan = tidx; chan < nchan*VEC_SIZE; chan += WARP_SIZE) {

            atomicAdd(_dkx_scl + chan, scale_fact_qy * sh_qy_scl[chan]);
            atomicAdd(_dvx_scl + chan, scale_fact_dy * sh_dy_scl[chan]);
        }
#else
        // 128-bit, consecutive atomics to glmem
        for (int chan = tidx; chan < nchan; chan += WARP_SIZE) {

            atomicAdd(_dkx + chan, __vscale(scale_fact_qy, sh_qy[chan]));
            atomicAdd(_dvx + chan, __vscale(scale_fact_dy, sh_dy[chan]));
        }
#endif
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
void s2_attn_bwd_special_vec_k(int nchan,  // no. of FLOATV_T elements along channel dim
                               int nlat_in,
                               int nlon_in,
                               int nlat_out,
                               int nlon_out,
                               const FLOATV_T *__restrict__ kx,
                               const FLOATV_T *__restrict__ vx,
                               const FLOATV_T *__restrict__ qy,
                               const FLOATV_T *__restrict__ dy,
                               const int32_t *__restrict__ row_idx,
                               const int64_t *__restrict__ row_off,
                               const int64_t *__restrict__ col_idx,
                               const   float *__restrict__ quad_weights,
                                     FLOATV_T *__restrict__ dkx,
                                     FLOATV_T *__restrict__ dvx,
                                     FLOATV_T *__restrict__ dqy) {

    static_assert(0 == (BDIM_X & (BDIM_X-1)));
    static_assert(0 == (BDIM_Y & (BDIM_Y-1)));
    static_assert((BDIM_X == 32 && BDIM_Y  > 1) ||
                  (BDIM_X  > 32 && BDIM_Y == 1)) ;

    constexpr int NLOC_M1 = NLOC-1;

    const int tidx = threadIdx.x;
    const int batch = blockIdx.y;
    const uint64_t ctaid = uint64_t(blockIdx.x) * blockDim.y + threadIdx.y;
    
    if (ctaid >= uint64_t(nlat_out)*nlon_in) {
        return;
    }

    extern __shared__ __align__(sizeof(float4)) float shext[];

    FLOATV_T *sh_dy = reinterpret_cast<FLOATV_T *>(shext) + threadIdx.y*nchan*2 + tidx;
    FLOATV_T *sh_qy = sh_dy + nchan;

    // for dqy
    FLOATV_T loc_k__[NLOC];
    FLOATV_T loc_vw_[NLOC];
    FLOATV_T loc_kvw[NLOC];

    // use permuted rows
    const int h = ctaid / nlon_out;
    const int wo = ctaid - (h*nlon_out);
    const int ho = row_idx[h];

    // offset input tensors
    kx += int64_t(batch)*nlat_in*nlon_in*nchan + tidx;
    vx += int64_t(batch)*nlat_in*nlon_in*nchan + tidx;
    qy += int64_t(batch)*nlat_out*nlon_out*nchan + int64_t(ho)*nlon_out*nchan + int64_t(wo)*nchan + tidx;
    dy += int64_t(batch)*nlat_out*nlon_out*nchan + int64_t(ho)*nlon_out*nchan + int64_t(wo)*nchan + tidx;

    // offset output tensors
    dkx += int64_t(batch)*nlat_in*nlon_in*nchan + tidx;
    dvx += int64_t(batch)*nlat_in*nlon_in*nchan + tidx;
    dqy += int64_t(batch)*nlat_out*nlon_out*nchan + int64_t(ho)*nlon_out*nchan + int64_t(wo)*nchan + tidx;

    #pragma unroll
    for(int i = 0; i < NLOC; i++) {
        loc_k__[i] = __vset<FLOATV_T>(0.0f);
        loc_vw_[i] = __vset<FLOATV_T>(0.0f);
        loc_kvw[i] = __vset<FLOATV_T>(0.0f);
    }

    #pragma unroll
    for(int i = 0; i < NLOC_M1; i++) {
        sh_dy[i*BDIM_X] = dy[i*BDIM_X];
        sh_qy[i*BDIM_X] = qy[i*BDIM_X];
    }
    if (NLOC_M1*BDIM_X+tidx < nchan) {
        sh_dy[NLOC_M1*BDIM_X] = dy[NLOC_M1*BDIM_X];
        sh_qy[NLOC_M1*BDIM_X] = qy[NLOC_M1*BDIM_X];
    }

#if __CUDA_ARCH__ < 900
    // for architectures < 9.0, sh_dy and sh_qy will be read
    // as individual floats at the end of the kernel, which
    // breaks the assumption that each FLOATV_T location is
    // written to and read by the same thread throughout the 
    // kernel, in the case FLOATV_T==float4
    if constexpr(std::is_same<FLOATV_T, float4>::value) {
        if constexpr(BDIM_X == 32) {    __syncwarp(); }
        else                       { __syncthreads(); }
    }
#endif

    // for dkx, dvx, dqy
    float alpha_sum = 0.0f;
    float qdotk_max = -FLT_MAX;
    
    // for dkx
    float integral = 0.0f;

    const int64_t rbeg = row_off[ho];
    const int64_t rend = row_off[ho+1];

    col_idx += rbeg;

    const int rlen = rend - rbeg;

    // accumulate alpha_sum, integral, and shared stats, 
    // along with a progressively computed qdotk_max.
    for (int off = 0; off < rlen; off++) {
        
        const int64_t col = col_idx[off];

        const int hi = col / nlon_in;
        const int wi = col - (hi * nlon_in);
        const int wip = (wi + wo) - ((wi + wo) / nlon_in) * nlon_in;

        const FLOATV_T *_kx = kx + int64_t(hi)*nlon_in*nchan + int64_t(wip)*nchan;
        const FLOATV_T *_vx = vx + int64_t(hi)*nlon_in*nchan + int64_t(wip)*nchan;

        FLOATV_T qdotk_v = __vset<FLOATV_T>(0.0f);
        FLOATV_T gdotv_v = __vset<FLOATV_T>(0.0f);

        #pragma unroll
        for(int i = 0; i < NLOC_M1; i++) {
            qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[i*BDIM_X], _kx[i*BDIM_X]));
            gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[i*BDIM_X], _vx[i*BDIM_X]));
        }
        if (NLOC_M1*BDIM_X+tidx < nchan) {
            qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[NLOC_M1*BDIM_X], _kx[NLOC_M1*BDIM_X]));
            gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[NLOC_M1*BDIM_X], _vx[NLOC_M1*BDIM_X]));
        }

        float qdotk = __vred(qdotk_v);
        float gdotv = __vred(gdotv_v);

        if constexpr(BDIM_X == 32) { 
            qdotk = __warp_sum(qdotk); 
            gdotv = __warp_sum(gdotv); 
        } else {
            qdotk = __block_sum<BDIM_X>(qdotk);
            gdotv = __block_sum<BDIM_X>(gdotv);
        }

        const float qdotk_max_tmp = max(qdotk_max, qdotk);
        const float alpha_inz = expf(qdotk - qdotk_max_tmp) * quad_weights[hi];
        const float max_correction = expf(qdotk_max - qdotk_max_tmp);

        alpha_sum = alpha_sum * max_correction + alpha_inz;
        integral  = integral * max_correction + alpha_inz * gdotv;

        const float ainz_gdotv = alpha_inz * gdotv;

        #pragma unroll
        for(int i = 0; i < NLOC_M1; i++) {
            const FLOATV_T kxval = _kx[i*BDIM_X];
            loc_k__[i] = __vadd(__vscale(max_correction, loc_k__[i]), __vscale(alpha_inz, kxval));
            loc_vw_[i] = __vadd(__vscale(max_correction, loc_vw_[i]), __vset<FLOATV_T>(ainz_gdotv));
            loc_kvw[i] = __vadd(__vscale(max_correction, loc_kvw[i]), __vscale(ainz_gdotv, kxval));
        }
        if (NLOC_M1*BDIM_X+tidx < nchan) {
            const FLOATV_T kxval = _kx[NLOC_M1*BDIM_X];
            loc_k__[NLOC_M1] = __vadd(__vscale(max_correction, loc_k__[NLOC_M1]), __vscale(alpha_inz, kxval));
            loc_vw_[NLOC_M1] = __vadd(__vscale(max_correction, loc_vw_[NLOC_M1]), __vset<FLOATV_T>(ainz_gdotv));
            loc_kvw[NLOC_M1] = __vadd(__vscale(max_correction, loc_kvw[NLOC_M1]), __vscale(ainz_gdotv, kxval));
        }

        qdotk_max = qdotk_max_tmp;
    }

    const float alpha_sum_inv = 1.0f / alpha_sum;

    integral *= alpha_sum_inv;

    // Write dqy
    const float alpha_sum_inv_sq = alpha_sum_inv*alpha_sum_inv;

    #pragma unroll
    for(int i = 0; i < NLOC_M1; i++) {
        dqy[i*BDIM_X] = __vscale(alpha_sum_inv_sq,
                                 __vsub(__vscale(alpha_sum, loc_kvw[i]),
                                        __vmul(loc_vw_[i], loc_k__[i])));
    }
    if (NLOC_M1*BDIM_X+tidx < nchan) {
        dqy[NLOC_M1*BDIM_X] = __vscale(alpha_sum_inv_sq,
                                       __vsub(__vscale(alpha_sum, loc_kvw[NLOC_M1]),
                                              __vmul(loc_vw_[NLOC_M1], loc_k__[NLOC_M1])));
    }

    // accumulate gradients for k and v
    for (int off = 0; off < rlen; off++) {

        const int64_t col = col_idx[off];

        const int hi = col / nlon_in;
        const int wi = col - (hi * nlon_in);
        const int wip = (wi + wo) - ((wi + wo) / nlon_in) * nlon_in;

        const FLOATV_T *_kx = kx + int64_t(hi)*nlon_in*nchan + int64_t(wip)*nchan;
        const FLOATV_T *_vx = vx + int64_t(hi)*nlon_in*nchan + int64_t(wip)*nchan;

        FLOATV_T qdotk_v = __vset<FLOATV_T>(0.0f);
        FLOATV_T gdotv_v = __vset<FLOATV_T>(0.0f);

        #pragma unroll
        for(int i = 0; i < NLOC_M1; i++) {
            qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[i*BDIM_X], _kx[i*BDIM_X]));
            gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[i*BDIM_X], _vx[i*BDIM_X]));
        }
        if (NLOC_M1*BDIM_X+tidx < nchan) {
            qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[NLOC_M1*BDIM_X], _kx[NLOC_M1*BDIM_X]));
            gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[NLOC_M1*BDIM_X], _vx[NLOC_M1*BDIM_X]));
        }

        float qdotk = __vred(qdotk_v);
        float gdotv = __vred(gdotv_v);

        if constexpr(BDIM_X == 32) { 
            qdotk = __warp_sum(qdotk); 
            gdotv = __warp_sum(gdotv); 
        } else {
            qdotk = __block_sum<BDIM_X>(qdotk);
            gdotv = __block_sum<BDIM_X>(gdotv);
        }

        const float alpha_inz = expf(qdotk - qdotk_max) * quad_weights[hi];

        FLOATV_T *_dkx = dkx + int64_t(hi)*nlon_in*nchan + int64_t(wip)*nchan;
        FLOATV_T *_dvx = dvx + int64_t(hi)*nlon_in*nchan + int64_t(wip)*nchan;

        const float alpha_mul = alpha_inz * alpha_sum_inv;

        const float scale_fact_qy = (gdotv - integral)*alpha_mul;
        const float scale_fact_dy =                    alpha_mul;

        // float4, 128-bit atomics are only supported by devices of compute 
        // capability 9.x+, so on older devices we resort to 32-bit atomics

#if __CUDA_ARCH__ < 900
        // making the loop count known at compile time doesn't seem
        // to make any difference here so let's keep this (much) 
        // simpler version
        float *sh_qy_scl = reinterpret_cast<float *>(sh_qy - tidx);
        float *sh_dy_scl = reinterpret_cast<float *>(sh_dy - tidx);

        float *_dkx_scl = reinterpret_cast<float *>(_dkx - tidx);
        float *_dvx_scl = reinterpret_cast<float *>(_dvx - tidx);

        constexpr int VEC_SIZE = sizeof(FLOATV_T)/sizeof(float);
        
        // 32-bit, consecutive atomics to glmem
        // strided atomics results in a severe slowdown
        for (int chan = tidx; chan < nchan*VEC_SIZE; chan += BDIM_X) {

            atomicAdd(_dkx_scl + chan, scale_fact_qy * sh_qy_scl[chan]);
            atomicAdd(_dvx_scl + chan, scale_fact_dy * sh_dy_scl[chan]);
        }
#else
        #pragma unroll
        for(int i = 0; i < NLOC_M1; i++) {
            atomicAdd(_dkx + i*BDIM_X, __vscale(scale_fact_qy, sh_qy[i*BDIM_X]));
            atomicAdd(_dvx + i*BDIM_X, __vscale(scale_fact_dy, sh_dy[i*BDIM_X]));
        }
        if (NLOC_M1*BDIM_X+tidx < nchan) {
            atomicAdd(_dkx + NLOC_M1*BDIM_X, __vscale(scale_fact_qy, sh_qy[NLOC_M1*BDIM_X]));
            atomicAdd(_dvx + NLOC_M1*BDIM_X, __vscale(scale_fact_dy, sh_dy[NLOC_M1*BDIM_X]));
        }
#endif
    }

    return;
}

template<typename FLOATV_T>
void launch_gen_attn_bwd(int batch_size,
                         int nchans,
                         int nlat_in,
                         int nlon_in,
                         int nlat_out,
                         int nlon_out,
                         FLOATV_T *_kxp,
                         FLOATV_T *_vxp,
                         FLOATV_T *_qyp,
                         FLOATV_T *_dyp,
                         int32_t *_row_idx,
                         int64_t *_row_off,
                         int64_t *_col_idx,
                         float *_quad_weights,
                         FLOATV_T *_dkxp,
                         FLOATV_T *_dvxp,
                         FLOATV_T *_dqyp,
                         cudaStream_t stream) {

    dim3 block(WARP_SIZE, THREADS / WARP_SIZE);
    dim3 grid(DIV_UP(nlat_out*nlon_out, block.y), batch_size);

    size_t shsize = sizeof(FLOATV_T)*nchans*5 * block.y; // 5 arrays per warp

    s2_attn_bwd_generic_vec_k<THREADS>
                             <<<grid, block, shsize, stream>>>(nchans, nlat_in, nlon_in, nlat_out, nlon_out,
                                                               _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx,
                                                               _quad_weights, _dkxp, _dvxp, _dqyp);
    CHECK_ERROR("s2_attn_bwd_generic_vec_k");

    return;
}

template<int BDIM_X,
         int BDIM_Y,
         int CUR_LOC_SIZE,
         int MAX_LOC_SIZE, // max size of FLOATV_T[] local array
         typename FLOATV_T>
void launch_spc_attn_bwd(int batch_size,
                         int nloc,      // "BDIM_X*nloc" >= nchans
                         int nchans,
                         int nlat_in,
                         int nlon_in,
                         int nlat_out,
                         int nlon_out,
                         FLOATV_T *_kxp,
                         FLOATV_T *_vxp,
                         FLOATV_T *_qyp,
                         FLOATV_T *_dyp,
                         int32_t *_row_idx,
                         int64_t *_row_off,
                         int64_t *_col_idx,
                         float *_quad_weights,
                         FLOATV_T *_dkxp,
                         FLOATV_T *_dvxp,
                         FLOATV_T *_dqyp,
                         cudaStream_t stream) {

    if (CUR_LOC_SIZE == nloc) {

        dim3 block(BDIM_X, BDIM_Y);
        dim3 grid(DIV_UP(nlat_out*nlon_out, block.y), batch_size);

        size_t shsize = sizeof(FLOATV_T)*nchans*2 * block.y; // 2 arrays per cta, block.y > 1 iif block.x==32

        s2_attn_bwd_special_vec_k<BDIM_X, BDIM_Y, CUR_LOC_SIZE>
                                 <<<grid, block, shsize, stream>>>(nchans, nlat_in, nlon_in, nlat_out, nlon_out,
                                                                   _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx,
                                                                   _quad_weights, _dkxp, _dvxp, _dqyp);
        CHECK_ERROR("s2_attn_bwd_special_vec_k");

        return;
    }
    if constexpr(CUR_LOC_SIZE < MAX_LOC_SIZE) {
        launch_spc_attn_bwd<BDIM_X,
                            BDIM_Y,
                            CUR_LOC_SIZE+1,
                            MAX_LOC_SIZE>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out,
                                          _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights,
                                          _dkxp, _dvxp, _dqyp, stream);
    }
    return;
}

static void s2_attn_bwd_dispatch(int64_t batch_size,
                                 int64_t nchans,
                                 int64_t nlon_in,
                                 int64_t nlat_out,
                                 int64_t nlon_out,
                                 at::Tensor kxP,
                                 at::Tensor vxP,
                                 at::Tensor qyP,
                                 at::Tensor dyP,
                                 at::Tensor row_off,
                                 at::Tensor col_idx,
                                 at::Tensor quad_weights,
                                 at::Tensor dkxP,
                                 at::Tensor dvxP,
                                 at::Tensor dqyP) {

    static_assert(0 == (MAX_LOCAL_ARR_LEN & (MAX_LOCAL_ARR_LEN-1)));

    // get stream
    auto stream = at::cuda::getCurrentCUDAStream().stream();

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
    float *_dyp = reinterpret_cast<float *>(dyP.data_ptr());

    float *_dkxp = reinterpret_cast<float *>(dkxP.data_ptr());
    float *_dvxp = reinterpret_cast<float *>(dvxP.data_ptr());
    float *_dqyp = reinterpret_cast<float *>(dqyP.data_ptr());

    int32_t *_row_idx = reinterpret_cast<int32_t *>(row_idx.data_ptr());
    int64_t *_row_off = reinterpret_cast<int64_t *>(row_off.data_ptr());
    int64_t *_col_idx = reinterpret_cast<int64_t *>(col_idx.data_ptr());
    float *_quad_weights = reinterpret_cast<float *>(quad_weights.data_ptr());

    constexpr int VEC_SIZE = sizeof(float4) / sizeof(float);

    if (!is_aligned<sizeof(float4)>(_kxp) ||
        !is_aligned<sizeof(float4)>(_vxp) ||
        !is_aligned<sizeof(float4)>(_qyp) ||
        !is_aligned<sizeof(float4)>(_dyp) ||
        !is_aligned<sizeof(float4)>(_dkxp) ||
        !is_aligned<sizeof(float4)>(_dvxp) ||
        !is_aligned<sizeof(float4)>(_dqyp) ||
        (nchans % VEC_SIZE) != 0) {

        const int nloc = DIV_UP(nchans, bdimx);

        // to avoid the compilation of unused template instances;
        // we use a block size BDIM_X that is the smallest power of 2
        // such that BDIM_X*MAX_LOCAL_ARR_LEN >= nchans, so
        // BDIM_X > 32 are used only for:
        //
        //  (BDIM_X-1)*MAX_LOCAL_ARR_LEN < nchans <= BDIM_X*MAX_LOCAL_ARR_LEN
        constexpr int MIN_LOC_ARR_LEN = MAX_LOCAL_ARR_LEN/2+1;

        // use 2D blocks only if 32 threads are enough; w.r.t fowrard,
        // we use the special kernel only up to BDIM_X=512 as with 1024
        // each thread cannot use more than 64 registers, resulting in
        // large amounts of registers spills
        switch(bdimx) {
            case   32: launch_spc_attn_bwd< 32, 2,               1, MAX_LOCAL_ARR_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp, stream); break;
            case   64: launch_spc_attn_bwd< 64, 1, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp, stream); break;
            case  128: launch_spc_attn_bwd<128, 1, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp, stream); break;
            case  256: launch_spc_attn_bwd<256, 1, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp, stream); break;
            case  512: launch_spc_attn_bwd<512, 1, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp, stream); break;
            default:   launch_gen_attn_bwd                                            (batch_size,       nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp, stream); break;
        }

    } else {

        float4 *_kxp4 = reinterpret_cast<float4 *>(kxP.data_ptr());
        float4 *_vxp4 = reinterpret_cast<float4 *>(vxP.data_ptr());
        float4 *_qyp4 = reinterpret_cast<float4 *>(qyP.data_ptr());
        float4 *_dyp4 = reinterpret_cast<float4 *>(dyP.data_ptr());

        float4 *_dkxp4 = reinterpret_cast<float4 *>(dkxP.data_ptr());
        float4 *_dvxp4 = reinterpret_cast<float4 *>(dvxP.data_ptr());
        float4 *_dqyp4 = reinterpret_cast<float4 *>(dqyP.data_ptr());

        nchans /= VEC_SIZE;
        const int nloc = DIV_UP(nchans, bdimx);

        constexpr int MAX_LOCAL_VEC_LEN = MAX_LOCAL_ARR_LEN / VEC_SIZE;
        
        constexpr int MIN_LOC_VEC_LEN = MAX_LOCAL_VEC_LEN/2+1;

        // use 2D blocks only if 32 threads are enough
        switch(bdimx) {
            case   32: launch_spc_attn_bwd< 32, 2,               1, MAX_LOCAL_VEC_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, _dyp4, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp4, _dvxp4, _dqyp4, stream); break;
            case   64: launch_spc_attn_bwd< 64, 1, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, _dyp4, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp4, _dvxp4, _dqyp4, stream); break;
            case  128: launch_spc_attn_bwd<128, 1, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, _dyp4, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp4, _dvxp4, _dqyp4, stream); break;
            case  256: launch_spc_attn_bwd<256, 1, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, _dyp4, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp4, _dvxp4, _dqyp4, stream); break;
            case  512: launch_spc_attn_bwd<512, 1, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, _dyp4, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp4, _dvxp4, _dqyp4, stream); break;
            default:   launch_gen_attn_bwd                                            (batch_size,       nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, _dyp4, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp4, _dvxp4, _dqyp4, stream); break;
        }
    }

    return;
}

// END backward kernels and functions

std::tuple<at::Tensor, at::Tensor, at::Tensor> s2_attention_bwd_dkvq_cuda(at::Tensor kx, at::Tensor vx, at::Tensor qy,
                                                                          at::Tensor dy, at::Tensor quad_weights,
                                                                          at::Tensor psi_col_idx, at::Tensor psi_row_off,
                                                                          int64_t nlon_in, int64_t nlat_out, int64_t nlon_out)
{

    CHECK_CUDA_INPUT_TENSOR(kx);
    CHECK_CUDA_INPUT_TENSOR(vx);
    CHECK_CUDA_INPUT_TENSOR(qy);
    CHECK_CUDA_INPUT_TENSOR(dy);
    CHECK_CUDA_TENSOR(quad_weights);
    CHECK_CUDA_TENSOR(psi_col_idx);
    CHECK_CUDA_TENSOR(psi_row_off);
    
    const size_t uo_num_channels = kx.size(1);
    const int batch_size = kx.size(0);

    // extract dtype
    auto kx_type = kx.dtype();
    auto vx_type = vx.dtype();
    auto qy_type = qy.dtype();
    auto dy_type = dy.dtype();

    torch::Tensor kxP = kx.to(torch::kFloat32);
    torch::Tensor vxP = vx.to(torch::kFloat32);
    torch::Tensor qyP = qy.to(torch::kFloat32);
    torch::Tensor dyP = dy.to(torch::kFloat32);

    // exract memory format: this is much safer than checking is_contiguous(at::MemoryFormat::ChannelsLast)
    // the former fails for num_channels == 1
    bool kx_is_channels_last = kxP.strides()[1] == 1;
    bool vx_is_channels_last = vxP.strides()[1] == 1;
    bool qy_is_channels_last = qyP.strides()[1] == 1;
    bool dy_is_channels_last = dyP.strides()[1] == 1;

    // transpose if required
    if (!kx_is_channels_last) { kxP = permute_4D_to0231(kxP); }
    if (!vx_is_channels_last) { vxP = permute_4D_to0231(vxP); }
    if (!qy_is_channels_last) { qyP = permute_4D_to0231(qyP); }
    if (!dy_is_channels_last) { dyP = permute_4D_to0231(dyP); }

    torch::Tensor dkxP = torch::zeros_like(kxP);
    torch::Tensor dvxP = torch::zeros_like(vxP);
    torch::Tensor dqyP = torch::zeros_like(qyP);

    s2_attn_bwd_dispatch(batch_size,
                         uo_num_channels,
                         nlon_in,
                         nlat_out,
                         nlon_out,
                         kxP, vxP, qyP, dyP,
                         psi_row_off,
                         psi_col_idx,
                         quad_weights,
                         dkxP, dvxP, dqyP);

    torch::Tensor dkx = dkxP;
    torch::Tensor dvx = dvxP;
    torch::Tensor dqy = dqyP;

    if (!kx_is_channels_last) { dkx = permute_4D_to0312(dkx); }
    if (!vx_is_channels_last) { dvx = permute_4D_to0312(dvx); }
    if (!qy_is_channels_last) { dqy = permute_4D_to0312(dqy); }

    // convert precision back to starting
    dkx = dkx.to(kx_type);
    dvx = dvx.to(vx_type);
    dqy = dqy.to(qy_type);

    return std::make_tuple(dkx, dvx, dqy);
}

TORCH_LIBRARY_IMPL(attention_kernels, CUDA, m)
{
    m.impl("backward",  &s2_attention_bwd_dkvq_cuda);
}

}
