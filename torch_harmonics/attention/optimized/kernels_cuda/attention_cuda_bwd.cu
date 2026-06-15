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

#include "attention_cuda.cuh"
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
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
#include "attention_cuda_utils.cuh"

#include <iostream>
#include <chrono>
#include <string>

#define THREADS (64)

#define MAX_LOCAL_ARR_LEN (16)

namespace attention_kernels
{

    // scatter-direction dispatcher, defined in attention_cuda_bwd_upsample.cu;
    // called by s2_attention_bwd_dkvq_cuda when nlon_out % nlon_in == 0.
    void s2_attn_bwd_upsample_dispatch(int batch_size, size_t nchans_in, size_t nchans_out, int64_t nlon_in,
                                       int64_t nlat_in, int64_t nlat_out, int64_t nlon_out, torch::Tensor kxP,
                                       torch::Tensor vxP, torch::Tensor qyP, torch::Tensor dyP,
                                       torch::Tensor psi_row_off, torch::Tensor psi_col_idx, torch::Tensor quad_weights,
                                       torch::Tensor dkxP, torch::Tensor dvxP, torch::Tensor dqyP);

    // BEGIN backward kernels and functions

    // called with (blockDim.x=32 and blockDim.y>1, BDIM=blockDim.x*blockDim.y)
    //
    // STORAGE_T is the global-memory element type of the INPUTS (kx/vx/qy/dy):
    // float4 for the fp32 vectorized path, float / c10::Half / c10::BFloat16 for
    // the scalar path. COMPUTE_T (float4 or float) is the arithmetic type and the
    // type of the gradient OUTPUTS dkx/dvx/dqy — these stay fp32 because dkx/dvx
    // are atomically scatter-accumulated across overlapping neighborhoods (fp16
    // atomics would lose precision / aren't well supported). The wrapper allocates
    // dkx/dvx/dqy in fp32 and casts to the input dtype at the end. vload widens
    // STORAGE_T inputs to COMPUTE_T at the load site.
    template <int BDIM_X, typename STORAGE_T>
    __global__ __launch_bounds__(BDIM_X) void s2_attn_bwd_generic_vec_k(
        int nchans_in,  // no. of elements along channel dim
        int nchans_out, // no. of elements along channel dim
        int nlat_in, int nlon_in, int nlat_out, int nlon_out,
        const STORAGE_T *__restrict__ kx, // [batch][nlat_in][nlon_in][nchan_in]
        const STORAGE_T *__restrict__ vx, // [batch][nlat_in][nlon_in][nchan_out]
        const STORAGE_T *__restrict__ qy, // [batch][nlat_out][nlon_out][nchan_in]
        const STORAGE_T *__restrict__ dy, // [batch][nlat_out][nlon_out][nchan_out]
        const int32_t *__restrict__ row_idx, const int64_t *__restrict__ row_off, const int64_t *__restrict__ col_idx,
        const float *__restrict__ quad_weights,
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dkx, // [batch][nlat_in][nlon_in][nchan_in]
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dvx, // [batch][nlat_in][nlon_in][nchan_out]
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dqy)
    { // [batch][nlat_out][nlon_out][nchan_in]
        using COMPUTE_T = typename vec_traits<STORAGE_T>::compute_t;

        extern __shared__ __align__(sizeof(float4)) float shext[];

        // for dqy
        COMPUTE_T *sh_alpha_k__ = reinterpret_cast<COMPUTE_T *>(shext) + threadIdx.y * (nchans_in * 4 + nchans_out);
        COMPUTE_T *sh_alpha_vw_ = sh_alpha_k__ + nchans_in;
        COMPUTE_T *sh_alpha_kvw = sh_alpha_vw_ + nchans_in;

        COMPUTE_T *sh_dy = sh_alpha_kvw + nchans_in;
        COMPUTE_T *sh_qy = sh_dy + nchans_out;
        // sh_alpha_k__[nchan_in], sh_alpha_vw_[nchan_in], sh_alpha_kvw[nchan_in]
        // sh_dy[nchan_out], sh_qy[nchan_in]

        const int batch = blockIdx.y;

        const uint64_t wid = uint64_t(blockIdx.x) * blockDim.y + threadIdx.y;
        if (wid >= uint64_t(nlat_out) * nlon_out) { return; }

        const int tidx = threadIdx.x;

        // use permuted rows
        const int h = wid / nlon_out;
        const int wo = wid - (h * nlon_out);
        const int ho = row_idx[h];

        // one output lon step corresponds to pscale input lon steps (requires nlon_in % nlon_out == 0)
        const int pscale = nlon_in / nlon_out;

        // offset input tensors
        kx += int64_t(batch) * nlat_in * nlon_in * nchans_in;
        qy += int64_t(batch) * nlat_out * nlon_out * nchans_in + int64_t(ho) * nlon_out * nchans_in
            + int64_t(wo) * nchans_in;

        vx += int64_t(batch) * nlat_in * nlon_in * nchans_out;
        dy += int64_t(batch) * nlat_out * nlon_out * nchans_out + int64_t(ho) * nlon_out * nchans_out
            + int64_t(wo) * nchans_out;

        // offset output tensors
        dkx += int64_t(batch) * nlat_in * nlon_in * nchans_in;
        dvx += int64_t(batch) * nlat_in * nlon_in * nchans_out;
        dqy += int64_t(batch) * nlat_out * nlon_out * nchans_in + int64_t(ho) * nlon_out * nchans_in
            + int64_t(wo) * nchans_in;

        // zero/init shared memory
        for (int chan = tidx; chan < nchans_in; chan += WARP_SIZE) {
            sh_alpha_k__[chan] = __vset<COMPUTE_T>(0.0f);
            sh_alpha_vw_[chan] = __vset<COMPUTE_T>(0.0f);
            sh_alpha_kvw[chan] = __vset<COMPUTE_T>(0.0f);

            sh_qy[chan] = vload(qy, chan);
        }
        for (int chan = tidx; chan < nchans_out; chan += WARP_SIZE) { sh_dy[chan] = vload(dy, chan); }

#if __CUDA_ARCH__ < 900
        // for architectures < 9.0, sh_dy and sh_qy will be read
        // as individual floats at the end of the kernel, which
        // breaks the assumption that each COMPUTE_T location is
        // written to and read by the same thread throughout the
        // kernel, in the case COMPUTE_T==float4
        if constexpr (std::is_same<COMPUTE_T, float4>::value) { __syncwarp(); }
#endif

        // for dkx, dvx, dqy
        float alpha_sum = 0.0f;
        float qdotk_max = -FLT_MAX;

        // for dkx
        float integral = 0.0f;

        const int64_t rbeg = row_off[ho];
        const int64_t rend = row_off[ho + 1];

        col_idx += rbeg;

        const int rlen = rend - rbeg;

        // accumulate alpha_sum, integral, and shared stats,
        // along with a progressively computed qdotk_max.
        for (int off = 0; off < rlen; off++) {

            const int64_t col = col_idx[off];

            const int hi = col / nlon_in;
            const int wi = col - (hi * nlon_in);
            const int wi_wo = wi + pscale * wo;
            const int wip = wi_wo - (wi_wo / nlon_in) * nlon_in;

            const STORAGE_T *_kx = kx + int64_t(hi) * nlon_in * nchans_in + int64_t(wip) * nchans_in;
            const STORAGE_T *_vx = vx + int64_t(hi) * nlon_in * nchans_out + int64_t(wip) * nchans_out;

            COMPUTE_T qdotk_v = __vset<COMPUTE_T>(0.0f);
            COMPUTE_T gdotv_v = __vset<COMPUTE_T>(0.0f);

            for (int chan = tidx; chan < nchans_in; chan += WARP_SIZE) {
                qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[chan], vload(_kx, chan)));
            }
            for (int chan = tidx; chan < nchans_out; chan += WARP_SIZE) {
                gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[chan], vload(_vx, chan)));
            }

            const float qdotk = __warp_sum(__vred(qdotk_v));
            const float gdotv = __warp_sum(__vred(gdotv_v));

            const float qdotk_max_tmp = max(qdotk_max, qdotk);
            const float alpha_inz = expf(qdotk - qdotk_max_tmp) * quad_weights[hi];
            const float max_correction = expf(qdotk_max - qdotk_max_tmp);
            alpha_sum = alpha_sum * max_correction + alpha_inz;

            integral = integral * max_correction + alpha_inz * gdotv;

            const float ainz_gdotv = alpha_inz * gdotv;

            for (int chan = tidx; chan < nchans_in; chan += WARP_SIZE) {

                const COMPUTE_T kxval = vload(_kx, chan);

                sh_alpha_k__[chan] = __vadd(__vscale(max_correction, sh_alpha_k__[chan]), __vscale(alpha_inz, kxval));
                sh_alpha_vw_[chan] = __vadd(__vscale(max_correction, sh_alpha_vw_[chan]), __vset<COMPUTE_T>(ainz_gdotv));
                sh_alpha_kvw[chan] = __vadd(__vscale(max_correction, sh_alpha_kvw[chan]), __vscale(ainz_gdotv, kxval));
            }
            qdotk_max = qdotk_max_tmp;
        }

        const float alpha_sum_inv = 1.0f / alpha_sum;

        integral *= alpha_sum_inv;

        // Write dqy (fp32 output)
        for (int chan = tidx; chan < nchans_in; chan += WARP_SIZE) {

            dqy[chan] = __vscale(
                alpha_sum_inv * alpha_sum_inv,
                __vsub(__vscale(alpha_sum, sh_alpha_kvw[chan]), __vmul(sh_alpha_vw_[chan], sh_alpha_k__[chan])));
        }

        // accumulate gradients for k and v
        for (int off = 0; off < rlen; off++) {

            const int64_t col = col_idx[off];
            const int hi = col / nlon_in;
            const int wi = col - (hi * nlon_in);
            const int wi_wo = wi + pscale * wo;
            const int wip = wi_wo - (wi_wo / nlon_in) * nlon_in;

            const STORAGE_T *_kx = kx + int64_t(hi) * nlon_in * nchans_in + int64_t(wip) * nchans_in;
            const STORAGE_T *_vx = vx + int64_t(hi) * nlon_in * nchans_out + int64_t(wip) * nchans_out;

            COMPUTE_T qdotk_v = __vset<COMPUTE_T>(0.0f);
            COMPUTE_T gdotv_v = __vset<COMPUTE_T>(0.0f);

            for (int chan = tidx; chan < nchans_in; chan += WARP_SIZE) {
                qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[chan], vload(_kx, chan)));
            }
            for (int chan = tidx; chan < nchans_out; chan += WARP_SIZE) {
                gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[chan], vload(_vx, chan)));
            }

            const float qdotk = __warp_sum(__vred(qdotk_v));
            const float gdotv = __warp_sum(__vred(gdotv_v));

            const float alpha_inz = expf(qdotk - qdotk_max) * quad_weights[hi];

            // _dkx / _dvx are COMPUTE_T (fp32) gradient buffers, accumulated atomically.
            COMPUTE_T *_dkx = dkx + int64_t(hi) * nlon_in * nchans_in + int64_t(wip) * nchans_in;
            COMPUTE_T *_dvx = dvx + int64_t(hi) * nlon_in * nchans_out + int64_t(wip) * nchans_out;

            const float alpha_mul = alpha_inz * alpha_sum_inv;

            const float scale_fact_qy = (gdotv - integral) * alpha_mul;
            const float scale_fact_dy = alpha_mul;

            // float4, 128-bit atomics are only supported by devices of compute
            // capability 9.x+, so on older devices we resort to 32-bit atomics

#if __CUDA_ARCH__ < 900
            // to use 32-bit operations on consecutve addresses
            float *sh_qy_scl = reinterpret_cast<float *>(sh_qy);
            float *sh_dy_scl = reinterpret_cast<float *>(sh_dy);

            float *_dkx_scl = reinterpret_cast<float *>(_dkx);
            float *_dvx_scl = reinterpret_cast<float *>(_dvx);

            constexpr int VEC_SIZE = sizeof(COMPUTE_T) / sizeof(float);

            // 32-bit, consecutive atomics to glmem;
            // strided atomics results in a severe slowdown
            for (int chan = tidx; chan < nchans_in * VEC_SIZE; chan += WARP_SIZE) {
                atomicAdd(_dkx_scl + chan, scale_fact_qy * sh_qy_scl[chan]);
            }
            for (int chan = tidx; chan < nchans_out * VEC_SIZE; chan += WARP_SIZE) {
                atomicAdd(_dvx_scl + chan, scale_fact_dy * sh_dy_scl[chan]);
            }
#else
            // 128-bit, consecutive atomics to glmem
            for (int chan = tidx; chan < nchans_in; chan += WARP_SIZE) {
                atomicAdd(_dkx + chan, __vscale(scale_fact_qy, sh_qy[chan]));
            }
            for (int chan = tidx; chan < nchans_out; chan += WARP_SIZE) {
                atomicAdd(_dvx + chan, __vscale(scale_fact_dy, sh_dy[chan]));
            }
#endif
        }

        return;
    }

    // called with either (BDIM_X=32 and BDIM_Y>1) || (2^K=BDIM_X > 32 and BDIM_Y=1)
    template <int BDIM_X, int BDIM_Y,
              int CHOUT_AS_IN, // 1 iif "BDIM_X*(NLOC-1) <= nchan_out <= BDIM_X*NLOC" else 0
              int NLOC,        // smallest int such that BDIM_X*NLOC >= nchan_in
              typename STORAGE_T>
    __global__ __launch_bounds__(BDIM_X *BDIM_Y) void s2_attn_bwd_special_vec_k(
        int nchan_in,  // no. of elements along channel dim
        int nchan_out, // no. of elements along channel dim
        int nlat_in, int nlon_in, int nlat_out, int nlon_out,
        const STORAGE_T *__restrict__ kx, // [batch][nlat_in][nlon_in][nchan_in]
        const STORAGE_T *__restrict__ vx, // [batch][nlat_in][nlon_in][nchan_out]
        const STORAGE_T *__restrict__ qy, // [batch][nlat_out][nlon_out][nchan_in]
        const STORAGE_T *__restrict__ dy, // [batch][nlat_out][nlon_out][nchan_out]
        const int32_t *__restrict__ row_idx, const int64_t *__restrict__ row_off, const int64_t *__restrict__ col_idx,
        const float *__restrict__ quad_weights,
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dkx, // [batch][nlat_in][nlon_in][nchan_in]
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dvx, // [batch][nlat_in][nlon_in][nchan_out]
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dqy)
    { // [batch][nlat_out][nlon_out][nchan_in]
        using COMPUTE_T = typename vec_traits<STORAGE_T>::compute_t;

        static_assert(0 == (BDIM_X & (BDIM_X - 1)));
        static_assert(0 == (BDIM_Y & (BDIM_Y - 1)));
        static_assert((BDIM_X == 32 && BDIM_Y > 1) || (BDIM_X > 32 && BDIM_Y == 1));

        constexpr int NLOC_M1 = NLOC - 1;

        const int tidx = threadIdx.x;
        const int batch = blockIdx.y;
        const uint64_t ctaid = uint64_t(blockIdx.x) * blockDim.y + threadIdx.y;

        if (ctaid >= uint64_t(nlat_out) * nlon_out) { return; }

        extern __shared__ __align__(sizeof(float4)) float shext[];

        // sh_dy[nchan_out], sh_qy[nchan_in]
        COMPUTE_T *sh_dy = reinterpret_cast<COMPUTE_T *>(shext) + threadIdx.y * (nchan_in + nchan_out); // + tidx;
        COMPUTE_T *sh_qy = sh_dy + nchan_out + tidx;

        if constexpr (CHOUT_AS_IN) { sh_dy += tidx; }

        // for dqy
        COMPUTE_T loc_k__[NLOC];
        COMPUTE_T loc_vw_[NLOC];
        COMPUTE_T loc_kvw[NLOC];

        // use permuted rows
        const int h = ctaid / nlon_out;
        const int wo = ctaid - (h * nlon_out);
        const int ho = row_idx[h];

        // one output lon step corresponds to pscale input lon steps (requires nlon_in % nlon_out == 0)
        const int pscale = nlon_in / nlon_out;

        // offset input tensors
        kx += int64_t(batch) * nlat_in * nlon_in * nchan_in + tidx;
        qy += int64_t(batch) * nlat_out * nlon_out * nchan_in + int64_t(ho) * nlon_out * nchan_in
            + int64_t(wo) * nchan_in + tidx;

        vx += int64_t(batch) * nlat_in * nlon_in * nchan_out; // + tidx;
        dy += int64_t(batch) * nlat_out * nlon_out * nchan_out + int64_t(ho) * nlon_out * nchan_out
            + int64_t(wo) * nchan_out; // + tidx;
        if constexpr (CHOUT_AS_IN) {
            vx += tidx;
            dy += tidx;
        }

        // offset output tensors
        dkx += int64_t(batch) * nlat_in * nlon_in * nchan_in + tidx;
        dvx += int64_t(batch) * nlat_in * nlon_in * nchan_out; // + tidx;
        if constexpr (CHOUT_AS_IN) { dvx += tidx; }
        dqy += int64_t(batch) * nlat_out * nlon_out * nchan_in + int64_t(ho) * nlon_out * nchan_in
            + int64_t(wo) * nchan_in + tidx;

#pragma unroll
        for (int i = 0; i < NLOC; i++) {
            loc_k__[i] = __vset<COMPUTE_T>(0.0f);
            loc_vw_[i] = __vset<COMPUTE_T>(0.0f);
            loc_kvw[i] = __vset<COMPUTE_T>(0.0f);
        }

#pragma unroll
        for (int i = 0; i < NLOC_M1; i++) { sh_qy[i * BDIM_X] = vload(qy, i * BDIM_X); }
        if (NLOC_M1 * BDIM_X + tidx < nchan_in) { sh_qy[NLOC_M1 * BDIM_X] = vload(qy, NLOC_M1 * BDIM_X); }

        if constexpr (CHOUT_AS_IN) {
#pragma unroll
            for (int i = 0; i < NLOC_M1; i++) { sh_dy[i * BDIM_X] = vload(dy, i * BDIM_X); }
            if (NLOC_M1 * BDIM_X + tidx < nchan_out) { sh_dy[NLOC_M1 * BDIM_X] = vload(dy, NLOC_M1 * BDIM_X); }
        } else {
            for (int chan = tidx; chan < nchan_out; chan += BDIM_X) { sh_dy[chan] = vload(dy, chan); }
        }

#if __CUDA_ARCH__ < 900
        // for architectures < 9.0, sh_dy and sh_qy will be read
        // as individual floats at the end of the kernel, which
        // breaks the assumption that each COMPUTE_T location is
        // written to and read by the same thread throughout the
        // kernel, in the case COMPUTE_T==float4
        if constexpr (std::is_same<COMPUTE_T, float4>::value) {
            if constexpr (BDIM_X == 32) {
                __syncwarp();
            } else {
                __syncthreads();
            }
        }
#endif

        // for dkx, dvx, dqy
        float alpha_sum = 0.0f;
        float qdotk_max = -FLT_MAX;

        // for dkx
        float integral = 0.0f;

        const int64_t rbeg = row_off[ho];
        const int64_t rend = row_off[ho + 1];

        col_idx += rbeg;

        const int rlen = rend - rbeg;

        // accumulate alpha_sum, integral, and shared stats,
        // along with a progressively computed qdotk_max.
        for (int off = 0; off < rlen; off++) {

            const int64_t col = col_idx[off];

            const int hi = col / nlon_in;
            const int wi = col - (hi * nlon_in);
            const int wi_wo = wi + pscale * wo;
            const int wip = wi_wo - (wi_wo / nlon_in) * nlon_in;

            const STORAGE_T *_kx = kx + int64_t(hi) * nlon_in * nchan_in + int64_t(wip) * nchan_in;
            const STORAGE_T *_vx = vx + int64_t(hi) * nlon_in * nchan_out + int64_t(wip) * nchan_out;

            COMPUTE_T qdotk_v = __vset<COMPUTE_T>(0.0f);
            COMPUTE_T gdotv_v = __vset<COMPUTE_T>(0.0f);

#pragma unroll
            for (int i = 0; i < NLOC_M1; i++) {
                qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[i * BDIM_X], vload(_kx, i * BDIM_X)));
            }
            if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[NLOC_M1 * BDIM_X], vload(_kx, NLOC_M1 * BDIM_X)));
            }
            if constexpr (CHOUT_AS_IN) {
#pragma unroll
                for (int i = 0; i < NLOC_M1; i++) {
                    gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[i * BDIM_X], vload(_vx, i * BDIM_X)));
                }
                if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
                    gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[NLOC_M1 * BDIM_X], vload(_vx, NLOC_M1 * BDIM_X)));
                }
            } else {
                for (int chan = tidx; chan < nchan_out; chan += BDIM_X) {
                    gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[chan], vload(_vx, chan)));
                }
            }

            float qdotk = __vred(qdotk_v);
            float gdotv = __vred(gdotv_v);

            if constexpr (BDIM_X == 32) {
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
            integral = integral * max_correction + alpha_inz * gdotv;

            const float ainz_gdotv = alpha_inz * gdotv;

#pragma unroll
            for (int i = 0; i < NLOC_M1; i++) {
                const COMPUTE_T kxval = vload(_kx, i * BDIM_X);
                loc_k__[i] = __vadd(__vscale(max_correction, loc_k__[i]), __vscale(alpha_inz, kxval));
                loc_vw_[i] = __vadd(__vscale(max_correction, loc_vw_[i]), __vset<COMPUTE_T>(ainz_gdotv));
                loc_kvw[i] = __vadd(__vscale(max_correction, loc_kvw[i]), __vscale(ainz_gdotv, kxval));
            }
            if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                const COMPUTE_T kxval = vload(_kx, NLOC_M1 * BDIM_X);
                loc_k__[NLOC_M1] = __vadd(__vscale(max_correction, loc_k__[NLOC_M1]), __vscale(alpha_inz, kxval));
                loc_vw_[NLOC_M1] = __vadd(__vscale(max_correction, loc_vw_[NLOC_M1]), __vset<COMPUTE_T>(ainz_gdotv));
                loc_kvw[NLOC_M1] = __vadd(__vscale(max_correction, loc_kvw[NLOC_M1]), __vscale(ainz_gdotv, kxval));
            }

            qdotk_max = qdotk_max_tmp;
        }

        const float alpha_sum_inv = 1.0f / alpha_sum;

        integral *= alpha_sum_inv;

        // Write dqy
        const float alpha_sum_inv_sq = alpha_sum_inv * alpha_sum_inv;

#pragma unroll
        for (int i = 0; i < NLOC_M1; i++) {
            dqy[i * BDIM_X]
                = __vscale(alpha_sum_inv_sq, __vsub(__vscale(alpha_sum, loc_kvw[i]), __vmul(loc_vw_[i], loc_k__[i])));
        }
        if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
            dqy[NLOC_M1 * BDIM_X]
                = __vscale(alpha_sum_inv_sq,
                           __vsub(__vscale(alpha_sum, loc_kvw[NLOC_M1]), __vmul(loc_vw_[NLOC_M1], loc_k__[NLOC_M1])));
        }

        // accumulate gradients for k and v
        for (int off = 0; off < rlen; off++) {

            const int64_t col = col_idx[off];

            const int hi = col / nlon_in;
            const int wi = col - (hi * nlon_in);
            const int wi_wo = wi + pscale * wo;
            const int wip = wi_wo - (wi_wo / nlon_in) * nlon_in;

            const STORAGE_T *_kx = kx + int64_t(hi) * nlon_in * nchan_in + int64_t(wip) * nchan_in;
            const STORAGE_T *_vx = vx + int64_t(hi) * nlon_in * nchan_out + int64_t(wip) * nchan_out;

            COMPUTE_T qdotk_v = __vset<COMPUTE_T>(0.0f);
            COMPUTE_T gdotv_v = __vset<COMPUTE_T>(0.0f);

#pragma unroll
            for (int i = 0; i < NLOC_M1; i++) {
                qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[i * BDIM_X], vload(_kx, i * BDIM_X)));
            }
            if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[NLOC_M1 * BDIM_X], vload(_kx, NLOC_M1 * BDIM_X)));
            }
            if constexpr (CHOUT_AS_IN) {
#pragma unroll
                for (int i = 0; i < NLOC_M1; i++) {
                    gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[i * BDIM_X], vload(_vx, i * BDIM_X)));
                }
                if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
                    gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[NLOC_M1 * BDIM_X], vload(_vx, NLOC_M1 * BDIM_X)));
                }
            } else {
                for (int chan = tidx; chan < nchan_out; chan += BDIM_X) {
                    gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[chan], vload(_vx, chan)));
                }
            }

            float qdotk = __vred(qdotk_v);
            float gdotv = __vred(gdotv_v);

            if constexpr (BDIM_X == 32) {
                qdotk = __warp_sum(qdotk);
                gdotv = __warp_sum(gdotv);
            } else {
                qdotk = __block_sum<BDIM_X>(qdotk);
                gdotv = __block_sum<BDIM_X>(gdotv);
            }

            const float alpha_inz = expf(qdotk - qdotk_max) * quad_weights[hi];

            COMPUTE_T *_dkx = dkx + int64_t(hi) * nlon_in * nchan_in + int64_t(wip) * nchan_in;
            COMPUTE_T *_dvx = dvx + int64_t(hi) * nlon_in * nchan_out + int64_t(wip) * nchan_out;

            const float alpha_mul = alpha_inz * alpha_sum_inv;

            const float scale_fact_qy = (gdotv - integral) * alpha_mul;
            const float scale_fact_dy = alpha_mul;

            // float4, 128-bit atomics are only supported by devices of compute
            // capability 9.x+, so on older devices we resort to 32-bit atomics

#if __CUDA_ARCH__ < 900
            constexpr int VEC_SIZE = sizeof(COMPUTE_T) / sizeof(float);

            // making the loop count known at compile time doesn't seem
            // to make any difference here so let's keep this (much)
            // simpler version
            float *sh_qy_scl = reinterpret_cast<float *>(sh_qy);
            float *sh_dy_scl = reinterpret_cast<float *>(sh_dy);

            float *_dkx_scl = reinterpret_cast<float *>(_dkx);
            float *_dvx_scl = reinterpret_cast<float *>(_dvx);

            sh_qy_scl -= tidx * VEC_SIZE;
            _dkx_scl -= tidx * VEC_SIZE;
            if constexpr (CHOUT_AS_IN) {
                sh_dy_scl -= tidx * VEC_SIZE;
                _dvx_scl -= tidx * VEC_SIZE;
            }

            // 32-bit, consecutive atomics to glmem
            // strided atomics results in a severe slowdown
            for (int chan = tidx; chan < nchan_in * VEC_SIZE; chan += BDIM_X) {
                atomicAdd(_dkx_scl + chan, scale_fact_qy * sh_qy_scl[chan]);
            }
            for (int chan = tidx; chan < nchan_out * VEC_SIZE; chan += BDIM_X) {
                atomicAdd(_dvx_scl + chan, scale_fact_dy * sh_dy_scl[chan]);
            }
#else
#pragma unroll
            for (int i = 0; i < NLOC_M1; i++) {
                atomicAdd(_dkx + i * BDIM_X, __vscale(scale_fact_qy, sh_qy[i * BDIM_X]));
            }
            if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                atomicAdd(_dkx + NLOC_M1 * BDIM_X, __vscale(scale_fact_qy, sh_qy[NLOC_M1 * BDIM_X]));
            }
            if constexpr (CHOUT_AS_IN) {
#pragma unroll
                for (int i = 0; i < NLOC_M1; i++) {
                    atomicAdd(_dvx + i * BDIM_X, __vscale(scale_fact_dy, sh_dy[i * BDIM_X]));
                }
                if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
                    atomicAdd(_dvx + NLOC_M1 * BDIM_X, __vscale(scale_fact_dy, sh_dy[NLOC_M1 * BDIM_X]));
                }
            } else {
                for (int chan = tidx; chan < nchan_out; chan += BDIM_X) {
                    atomicAdd(_dvx + chan, __vscale(scale_fact_dy, sh_dy[chan]));
                }
            }
#endif
        }

        return;
    }

    template <typename STORAGE_T>
    void launch_gen_attn_bwd(int batch_size, int nchans_in, int nchans_out, int nlat_in, int nlon_in, int nlat_out,
                             int nlon_out, STORAGE_T *_kxp, STORAGE_T *_vxp, STORAGE_T *_qyp, STORAGE_T *_dyp,
                             int32_t *_row_idx, int64_t *_row_off, int64_t *_col_idx, float *_quad_weights,
                             typename vec_traits<STORAGE_T>::compute_t *_dkxp,
                             typename vec_traits<STORAGE_T>::compute_t *_dvxp,
                             typename vec_traits<STORAGE_T>::compute_t *_dqyp, cudaStream_t stream)
    {

        dim3 block(WARP_SIZE, THREADS / WARP_SIZE);
        dim3 grid(DIV_UP(nlat_out * nlon_out, block.y), batch_size);

        // shared memory holds compute-type (COMPUTE_T) data, not STORAGE_T. 5 arrays per warp.
        size_t shsize = sizeof(typename vec_traits<STORAGE_T>::compute_t) * (nchans_in * 4 + nchans_out) * block.y;

        s2_attn_bwd_generic_vec_k<THREADS><<<grid, block, shsize, stream>>>(
            nchans_in, nchans_out, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off,
            _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp);
        CHECK_ERROR("s2_attn_bwd_generic_vec_k");

        return;
    }

    template <int BDIM_X, int BDIM_Y, int CUR_LOC_SIZE,
              int MAX_LOC_SIZE, // max size of COMPUTE_T[] local array
              typename STORAGE_T>
    void launch_spc_attn_bwd(int nloc, // "BDIM_X*nloc" >= nchans_out
                             int batch_size, int nchans_in, int nchans_out, int nlat_in, int nlon_in, int nlat_out,
                             int nlon_out, STORAGE_T *_kxp, STORAGE_T *_vxp, STORAGE_T *_qyp, STORAGE_T *_dyp,
                             int32_t *_row_idx, int64_t *_row_off, int64_t *_col_idx, float *_quad_weights,
                             typename vec_traits<STORAGE_T>::compute_t *_dkxp,
                             typename vec_traits<STORAGE_T>::compute_t *_dvxp,
                             typename vec_traits<STORAGE_T>::compute_t *_dqyp, cudaStream_t stream)
    {

        if (CUR_LOC_SIZE == nloc) {

            dim3 block(BDIM_X, BDIM_Y);
            dim3 grid(DIV_UP(nlat_out * nlon_out, block.y), batch_size);

            // shared memory holds compute-type (COMPUTE_T) data, not STORAGE_T.
            // 2 arrays per cta, block.y > 1 iif block.x==32
            size_t shsize = sizeof(typename vec_traits<STORAGE_T>::compute_t) * (nchans_in + nchans_out) * block.y;

            // nloc determines the size of local arrays used to store
            // temporary buffers loc_k__[], loc_vw_[] and loc_kvw[],
            // of size nchans_in each;
            // if nchans_out is >= BDIM_X*(nloc-1) and <= BDIM_X*nloc
            // then we can use the same compile-time known loops used
            // for input channels, with the exception of testing
            // whether to execute the last iteration based on "nchans_out"
            // instead of "nchans_in"; in this way as long as the
            // difference between the number of input and output channels
            // is <= BDIM_X we can use the faster path
            if (nchans_out >= BDIM_X * (CUR_LOC_SIZE - 1) && nchans_out <= BDIM_X * CUR_LOC_SIZE) {
                s2_attn_bwd_special_vec_k<BDIM_X, BDIM_Y, 1, CUR_LOC_SIZE><<<grid, block, shsize, stream>>>(
                    nchans_in, nchans_out, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx,
                    _row_off, _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp);
            } else {
                s2_attn_bwd_special_vec_k<BDIM_X, BDIM_Y, 0, CUR_LOC_SIZE><<<grid, block, shsize, stream>>>(
                    nchans_in, nchans_out, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx,
                    _row_off, _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp);
            }
            CHECK_ERROR("s2_attn_bwd_special_vec_k");

            return;
        }
        if constexpr (CUR_LOC_SIZE < MAX_LOC_SIZE) {
            launch_spc_attn_bwd<BDIM_X, BDIM_Y, CUR_LOC_SIZE + 1, MAX_LOC_SIZE>(
                nloc, batch_size, nchans_in, nchans_out, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp,
                _row_idx, _row_off, _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp, stream);
        }
        return;
    }

    // Picks the block size (BDIM_X) instance and launches the backward gather
    // kernel for a given input storage vector type SV. Inputs are SV*; gradient
    // outputs are COMPUTE_T* (fp32) — see s2_attn_bwd_*_vec_k. Backward uses the
    // special kernel only up to BDIM_X=512 (1024 spills); larger falls to generic.
    template <int MAX_LOC, int MIN_LOC, typename SV>
    static void bwd_dispatch_bdimx(int bdimx, int nloc, int64_t batch_size, int64_t nchans_in, int64_t nchans_out,
                                   int nlat_in, int64_t nlon_in, int64_t nlat_out, int64_t nlon_out, SV *_kxp, SV *_vxp,
                                   SV *_qyp, SV *_dyp, int32_t *_row_idx, int64_t *_row_off, int64_t *_col_idx,
                                   float *_quad_weights, typename vec_traits<SV>::compute_t *_dkxp,
                                   typename vec_traits<SV>::compute_t *_dvxp, typename vec_traits<SV>::compute_t *_dqyp,
                                   cudaStream_t stream)
    {
        switch (bdimx) {
        case 32:
            launch_spc_attn_bwd<32, 2, 1, MAX_LOC>(nloc, batch_size, nchans_in, nchans_out, nlat_in, nlon_in, nlat_out,
                                                   nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx,
                                                   _quad_weights, _dkxp, _dvxp, _dqyp, stream);
            break;
        case 64:
            launch_spc_attn_bwd<64, 1, MIN_LOC, MAX_LOC>(nloc, batch_size, nchans_in, nchans_out, nlat_in, nlon_in,
                                                         nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off,
                                                         _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp, stream);
            break;
        case 128:
            launch_spc_attn_bwd<128, 1, MIN_LOC, MAX_LOC>(nloc, batch_size, nchans_in, nchans_out, nlat_in, nlon_in,
                                                          nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off,
                                                          _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp, stream);
            break;
        case 256:
            launch_spc_attn_bwd<256, 1, MIN_LOC, MAX_LOC>(nloc, batch_size, nchans_in, nchans_out, nlat_in, nlon_in,
                                                          nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off,
                                                          _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp, stream);
            break;
        case 512:
            launch_spc_attn_bwd<512, 1, MIN_LOC, MAX_LOC>(nloc, batch_size, nchans_in, nchans_out, nlat_in, nlon_in,
                                                          nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off,
                                                          _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp, stream);
            break;
        default:
            launch_gen_attn_bwd(batch_size, nchans_in, nchans_out, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp,
                                _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp, stream);
            break;
        }
    }

    // Templated on the input storage element type (float / c10::Half / c10::BFloat16).
    // Gradient outputs dkx/dvx/dqy are always fp32 (COMPUTE_T) — dkx/dvx are atomic
    // scatter-accumulated, so they cannot be reduced precision; the wrapper casts
    // them to the input dtype at the end. fp32 keeps the float4 vectorized path;
    // fp16/bf16 (and unaligned fp32) take the scalar STORAGE_T path.
    template <typename scalar_t>
    static void s2_attn_bwd_dispatch(int64_t batch_size, int64_t nchans_in, int64_t nchans_out, int64_t nlon_in,
                                     int64_t nlat_out, int64_t nlon_out, at::Tensor kxP, at::Tensor vxP, at::Tensor qyP,
                                     at::Tensor dyP, at::Tensor row_off, at::Tensor col_idx, at::Tensor quad_weights,
                                     at::Tensor dkxP, at::Tensor dvxP, at::Tensor dqyP)
    {

        static_assert(0 == (MAX_LOCAL_ARR_LEN & (MAX_LOCAL_ARR_LEN - 1)));

        // get stream
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        // sort row indices (ho-s) in descending order
        // based on (row_off[ho+1]-row_off[ho])
        at::Tensor row_idx = sortRows(nlat_out, row_off, stream);

        const int nlat_in = kxP.size(1);

        // smallest power of two "bdimx" (>=32) s.t. bdimx*MAX_LOCAL_ARR_LEN >= nchans_in
        int bdimx;
        bdimx = DIV_UP(nchans_in, MAX_LOCAL_ARR_LEN);
        bdimx = max(bdimx, WARP_SIZE);
        bdimx = next_pow2(bdimx);

        scalar_t *_kxp = reinterpret_cast<scalar_t *>(kxP.data_ptr());
        scalar_t *_vxp = reinterpret_cast<scalar_t *>(vxP.data_ptr());
        scalar_t *_qyp = reinterpret_cast<scalar_t *>(qyP.data_ptr());
        scalar_t *_dyp = reinterpret_cast<scalar_t *>(dyP.data_ptr());

        // gradient outputs are fp32
        float *_dkxp = reinterpret_cast<float *>(dkxP.data_ptr());
        float *_dvxp = reinterpret_cast<float *>(dvxP.data_ptr());
        float *_dqyp = reinterpret_cast<float *>(dqyP.data_ptr());

        int32_t *_row_idx = reinterpret_cast<int32_t *>(row_idx.data_ptr());
        int64_t *_row_off = reinterpret_cast<int64_t *>(row_off.data_ptr());
        int64_t *_col_idx = reinterpret_cast<int64_t *>(col_idx.data_ptr());
        float *_quad_weights = reinterpret_cast<float *>(quad_weights.data_ptr());

        constexpr int MIN_LOC_ARR_LEN = MAX_LOCAL_ARR_LEN / 2 + 1;

        if constexpr (std::is_same<scalar_t, float>::value) {
            // fp32: float4 vectorized when 16B-aligned + 4-divisible, else scalar.
            constexpr int VEC_SIZE = sizeof(float4) / sizeof(float); // 4
            const bool use_vec = is_aligned<16>(_kxp) && is_aligned<16>(_vxp) && is_aligned<16>(_qyp)
                && is_aligned<16>(_dyp) && is_aligned<16>(_dkxp) && is_aligned<16>(_dvxp) && is_aligned<16>(_dqyp)
                && (nchans_in % VEC_SIZE) == 0 && (nchans_out % VEC_SIZE) == 0;

            if (use_vec) {
                constexpr int MAX_VEC = MAX_LOCAL_ARR_LEN / VEC_SIZE;
                constexpr int MIN_VEC = MAX_VEC / 2 + 1;
                const int64_t nci = nchans_in / VEC_SIZE;
                const int64_t nco = nchans_out / VEC_SIZE;
                bwd_dispatch_bdimx<MAX_VEC, MIN_VEC, float4>(
                    bdimx, DIV_UP(nci, bdimx), batch_size, nci, nco, nlat_in, nlon_in, nlat_out, nlon_out,
                    reinterpret_cast<float4 *>(_kxp), reinterpret_cast<float4 *>(_vxp),
                    reinterpret_cast<float4 *>(_qyp), reinterpret_cast<float4 *>(_dyp), _row_idx, _row_off, _col_idx,
                    _quad_weights, reinterpret_cast<float4 *>(_dkxp), reinterpret_cast<float4 *>(_dvxp),
                    reinterpret_cast<float4 *>(_dqyp), stream);
            } else {
                bwd_dispatch_bdimx<MAX_LOCAL_ARR_LEN, MIN_LOC_ARR_LEN, float>(
                    bdimx, DIV_UP(nchans_in, bdimx), batch_size, nchans_in, nchans_out, nlat_in, nlon_in, nlat_out,
                    nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp,
                    stream);
            }
        } else {
            // fp16/bf16: scalar STORAGE_T inputs, fp32 outputs; fp32 compute/accumulation.
            bwd_dispatch_bdimx<MAX_LOCAL_ARR_LEN, MIN_LOC_ARR_LEN, scalar_t>(
                bdimx, DIV_UP(nchans_in, bdimx), batch_size, nchans_in, nchans_out, nlat_in, nlon_in, nlat_out, nlon_out,
                _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp, stream);
        }

        return;
    }

    // END backward kernels and functions

    std::tuple<at::Tensor, at::Tensor, at::Tensor>
    s2_attention_bwd_dkvq_cuda(at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor dy, at::Tensor quad_weights,
                               at::Tensor psi_col_idx, at::Tensor psi_row_off, int64_t nlon_in, int64_t nlat_out,
                               int64_t nlon_out)
    {

        CHECK_CUDA_INPUT_TENSOR(kx);
        CHECK_CUDA_INPUT_TENSOR(vx);
        CHECK_CUDA_INPUT_TENSOR(qy);
        CHECK_CUDA_INPUT_TENSOR(dy);
        CHECK_CUDA_TENSOR(quad_weights);
        CHECK_CUDA_TENSOR(psi_col_idx);
        CHECK_CUDA_TENSOR(psi_row_off);

        // direction selection: gather (self / downsample) iff nlon_in is an integer
        // multiple of nlon_out; scatter (upsample) iff nlon_out is an integer multiple
        // of nlon_in. Self-attention satisfies both and routes through the gather path.
        const bool downsample = (nlon_in % nlon_out == 0);
        const bool upsample = (nlon_out % nlon_in == 0);
        TORCH_CHECK(downsample || upsample, "either nlon_in (", nlon_in, ") must be an integer multiple of nlon_out (",
                    nlon_out, "), or vice versa");

        // const size_t uo_num_channels = kx.size(1);
        size_t nchans_in = qy.size(1); // or kx.size(1)
        size_t nchans_out = vx.size(1);

        const int batch_size = kx.size(0);
        const int64_t nlat_in = kx.size(2);

        // extract dtype
        auto kx_type = kx.dtype(); // nchans_in
        auto qy_type = qy.dtype();
        auto vx_type = vx.dtype(); // ncahn_out
        auto dy_type = dy.dtype();

        torch::Tensor dkx, dvx, dqy;

        // ATen dispatch over the input dtype.
        //
        // Gather (downsample / self) path: native storage. kx/vx/qy/dy keep their
        // dtype; the kernel widens to fp32 at load (Tier B). Gradient buffers
        // dkx/dvx/dqy are allocated fp32 because dkx/dvx are atomically
        // scatter-accumulated (reduced-precision atomics would lose precision);
        // they are cast back to the input dtype at the end.
        //
        // Scatter (upsample) path: scatter kernels are not yet on native storage,
        // so we still upcast inputs to fp32. fp32 inputs are unaffected either way.
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, qy.scalar_type(), "s2_attention_bwd_dkvq_cuda", [&] {
            using storage_t = scalar_t;

            const auto f32_like
                = [](const torch::Tensor &t) { return torch::zeros_like(t, t.options().dtype(torch::kFloat32)); };

            if (downsample) {
                // native-storage gather path
                torch::Tensor kxP = kx;
                torch::Tensor vxP = vx;
                torch::Tensor qyP = qy;
                torch::Tensor dyP = dy;

                // safer than is_contiguous(ChannelsLast), which fails for num_channels == 1
                bool kx_is_channels_last = kxP.strides()[1] == 1;
                bool vx_is_channels_last = vxP.strides()[1] == 1;
                bool qy_is_channels_last = qyP.strides()[1] == 1;
                bool dy_is_channels_last = dyP.strides()[1] == 1;

                if (!kx_is_channels_last) { kxP = permute_4D_to0231(kxP); }
                if (!vx_is_channels_last) { vxP = permute_4D_to0231(vxP); }
                if (!qy_is_channels_last) { qyP = permute_4D_to0231(qyP); }
                if (!dy_is_channels_last) { dyP = permute_4D_to0231(dyP); }

                // fp32 gradient buffers (same shape/layout as the native inputs)
                torch::Tensor dkxP = f32_like(kxP);
                torch::Tensor dvxP = f32_like(vxP);
                torch::Tensor dqyP = f32_like(qyP);

                s2_attn_bwd_dispatch<storage_t>(batch_size, nchans_in, nchans_out, nlon_in, nlat_out, nlon_out, kxP,
                                                vxP, qyP, dyP, psi_row_off, psi_col_idx, quad_weights, dkxP, dvxP, dqyP);

                dkx = dkxP;
                dvx = dvxP;
                dqy = dqyP;

                if (!kx_is_channels_last) { dkx = permute_4D_to0312(dkx); }
                if (!vx_is_channels_last) { dvx = permute_4D_to0312(dvx); }
                if (!qy_is_channels_last) { dqy = permute_4D_to0312(dqy); }
            } else {
                // upsample path: still fp32 internally (Tier B not applied to scatter kernels)
                torch::Tensor kxP = kx.to(torch::kFloat32);
                torch::Tensor vxP = vx.to(torch::kFloat32);
                torch::Tensor qyP = qy.to(torch::kFloat32);
                torch::Tensor dyP = dy.to(torch::kFloat32);

                bool kx_is_channels_last = kxP.strides()[1] == 1;
                bool vx_is_channels_last = vxP.strides()[1] == 1;
                bool qy_is_channels_last = qyP.strides()[1] == 1;
                bool dy_is_channels_last = dyP.strides()[1] == 1;

                if (!kx_is_channels_last) { kxP = permute_4D_to0231(kxP); }
                if (!vx_is_channels_last) { vxP = permute_4D_to0231(vxP); }
                if (!qy_is_channels_last) { qyP = permute_4D_to0231(qyP); }
                if (!dy_is_channels_last) { dyP = permute_4D_to0231(dyP); }

                torch::Tensor dkxP = f32_like(kxP);
                torch::Tensor dvxP = f32_like(vxP);
                torch::Tensor dqyP = f32_like(qyP);

                s2_attn_bwd_upsample_dispatch(batch_size, nchans_in, nchans_out, nlon_in, nlat_in, nlat_out, nlon_out,
                                              kxP, vxP, qyP, dyP, psi_row_off, psi_col_idx, quad_weights, dkxP, dvxP,
                                              dqyP);

                dkx = dkxP;
                dvx = dvxP;
                dqy = dqyP;

                if (!kx_is_channels_last) { dkx = permute_4D_to0312(dkx); }
                if (!vx_is_channels_last) { dvx = permute_4D_to0312(dvx); }
                if (!qy_is_channels_last) { dqy = permute_4D_to0312(dqy); }
            }
        });

        // convert precision back to starting dtype (no-op for fp32; narrows for fp16/bf16)
        dkx = dkx.to(kx_type);
        dvx = dvx.to(vx_type);
        dqy = dqy.to(qy_type);

        return std::make_tuple(dkx, dvx, dqy);
    }

    TORCH_LIBRARY_IMPL(attention_kernels, CUDA, m) { m.impl("backward", &s2_attention_bwd_dkvq_cuda); }

} // namespace attention_kernels
