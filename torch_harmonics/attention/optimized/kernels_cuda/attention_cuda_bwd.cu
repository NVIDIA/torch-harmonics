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


    // **************** start generic kernel ****************

    // called with (blockDim.x=32 and blockDim.y>1, BDIM=blockDim.x*blockDim.y)
    template <int BDIM_X,
              typename FLOATV_T> // either float or float4
    __global__ __launch_bounds__(BDIM_X) void s2_attn_bwd_generic_vec_k(
        const __grid_constant__ attn_params_t p,
        const FLOATV_T *__restrict__ kx, // [batch][nlat_in][nlon_in][nchan_in]
        const FLOATV_T *__restrict__ vx, // [batch][nlat_in][nlon_in][nchan_out]
        const FLOATV_T *__restrict__ qy, // [batch][nlat_out][nlon_out][nchan_in]
        const FLOATV_T *__restrict__ dy, // [batch][nlat_out][nlon_out][nchan_out]
        const int32_t *__restrict__ row_idx, const int64_t *__restrict__ row_off, const int64_t *__restrict__ col_idx,
        const float *__restrict__ quad_weights,
        FLOATV_T *__restrict__ dkx, // [batch][nlat_in][nlon_in][nchan_in]
        FLOATV_T *__restrict__ dvx, // [batch][nlat_in][nlon_in][nchan_out]
        FLOATV_T *__restrict__ dqy)
    { // [batch][nlat_out][nlon_out][nchan_in]

        const int batch = blockIdx.y;

        const int &nchan_in = p.nchan_in;
        const int &nchan_out = p.nchan_out;
        const int &nlat_in = p.nlat_in;
        const int &nlon_in = p.nlon_in;
        const int &pscale = p.pscale;
        const int &nlat_out = p.nlat_out;
        const int &nlon_out = p.nlon_out;

        const uint64_t wid = uint64_t(blockIdx.x) * blockDim.y + threadIdx.y;
        if (wid >= uint64_t(nlat_out) * nlon_out) { return; }

        extern __shared__ __align__(sizeof(float4)) float shext[];

        // for dqy
        FLOATV_T *sh_alpha_k__ = reinterpret_cast<FLOATV_T *>(shext) + threadIdx.y * (nchan_in * 4 + nchan_out);
        FLOATV_T *sh_alpha_vw_ = sh_alpha_k__ + nchan_in;
        FLOATV_T *sh_alpha_kvw = sh_alpha_vw_ + nchan_in;

        FLOATV_T *sh_dy = sh_alpha_kvw + nchan_in;
        FLOATV_T *sh_qy = sh_dy + nchan_out;
        // sh_alpha_k__[nchan_in], sh_alpha_vw_[nchan_in], sh_alpha_kvw[nchan_in]
        // sh_dy[nchan_out], sh_qy[nchan_in]

        const int tidx = threadIdx.x;

        // use permuted rows
        const int h = wid / nlon_out;
        const int wo = wid - (h * nlon_out);
        const int ho = row_idx[h];

        // one output lon step corresponds to pscale input lon steps (requires nlon_in % nlon_out == 0)
        //const int pscale = nlon_in / nlon_out;

        // offset input tensors
        kx += int64_t(batch) * nlat_in * nlon_in * nchan_in;
        qy += int64_t(batch) * nlat_out * nlon_out * nchan_in + int64_t(ho) * nlon_out * nchan_in
            + int64_t(wo) * nchan_in;

        vx += int64_t(batch) * nlat_in * nlon_in * nchan_out;
        dy += int64_t(batch) * nlat_out * nlon_out * nchan_out + int64_t(ho) * nlon_out * nchan_out
            + int64_t(wo) * nchan_out;

        // offset output tensors
        dkx += int64_t(batch) * nlat_in * nlon_in * nchan_in;
        dvx += int64_t(batch) * nlat_in * nlon_in * nchan_out;
        dqy += int64_t(batch) * nlat_out * nlon_out * nchan_in + int64_t(ho) * nlon_out * nchan_in
            + int64_t(wo) * nchan_in;

        // zero/init shared memory
        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
            sh_alpha_k__[chan] = __vset<FLOATV_T>(0.0f);
            sh_alpha_vw_[chan] = __vset<FLOATV_T>(0.0f);
            sh_alpha_kvw[chan] = __vset<FLOATV_T>(0.0f);

            sh_qy[chan] = qy[chan];
        }
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { sh_dy[chan] = dy[chan]; }

#if __CUDA_ARCH__ < 900
        // for architectures < 9.0, sh_dy and sh_qy will be read
        // as individual floats at the end of the kernel, which
        // breaks the assumption that each FLOATV_T location is
        // written to and read by the same thread throughout the
        // kernel, in the case FLOATV_T==float4
        if constexpr (std::is_same<FLOATV_T, float4>::value) { __syncwarp(); }
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

            const FLOATV_T *_kx = kx + int64_t(hi) * nlon_in * nchan_in + int64_t(wip) * nchan_in;
            const FLOATV_T *_vx = vx + int64_t(hi) * nlon_in * nchan_out + int64_t(wip) * nchan_out;

            FLOATV_T qdotk_v = __vset<FLOATV_T>(0.0f);
            FLOATV_T gdotv_v = __vset<FLOATV_T>(0.0f);

            for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
                qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[chan], _kx[chan]));
            }
            for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
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

            for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {

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
        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {

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

            const FLOATV_T *_kx = kx + int64_t(hi) * nlon_in * nchan_in + int64_t(wip) * nchan_in;
            const FLOATV_T *_vx = vx + int64_t(hi) * nlon_in * nchan_out + int64_t(wip) * nchan_out;

            FLOATV_T qdotk_v = __vset<FLOATV_T>(0.0f);
            FLOATV_T gdotv_v = __vset<FLOATV_T>(0.0f);

            for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
                qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[chan], _kx[chan]));
            }
            for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
                gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[chan], _vx[chan]));
            }

            const float qdotk = __warp_sum(__vred(qdotk_v));
            const float gdotv = __warp_sum(__vred(gdotv_v));

            const float alpha_inz = expf(qdotk - qdotk_max) * quad_weights[hi];

            FLOATV_T *_dkx = dkx + int64_t(hi) * nlon_in * nchan_in + int64_t(wip) * nchan_in;
            FLOATV_T *_dvx = dvx + int64_t(hi) * nlon_in * nchan_out + int64_t(wip) * nchan_out;

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

            constexpr int VEC_SIZE = sizeof(FLOATV_T) / sizeof(float);

            // 32-bit, consecutive atomics to glmem;
            // strided atomics results in a severe slowdown
            for (int chan = tidx; chan < nchan_in * VEC_SIZE; chan += WARP_SIZE) {
                atomicAdd(_dkx_scl + chan, scale_fact_qy * sh_qy_scl[chan]);
            }
            for (int chan = tidx; chan < nchan_out * VEC_SIZE; chan += WARP_SIZE) {
                atomicAdd(_dvx_scl + chan, scale_fact_dy * sh_dy_scl[chan]);
            }
#else
            // 128-bit, consecutive atomics to glmem
            for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
                atomicAdd(_dkx + chan, __vscale(scale_fact_qy, sh_qy[chan]));
            }
            for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
                atomicAdd(_dvx + chan, __vscale(scale_fact_dy, sh_dy[chan]));
            }
#endif
        }

        return;
    }
    
    template <typename FLOATV_T>
    void launch_gen_attn_bwd(attn_params_t params, int batch_size, 
                             FLOATV_T *_kxp, FLOATV_T *_vxp, FLOATV_T *_qyp, FLOATV_T *_dyp,
                             int32_t *_row_idx, int64_t *_row_off, int64_t *_col_idx, float *_quad_weights,
                             FLOATV_T *_dkxp, FLOATV_T *_dvxp, FLOATV_T *_dqyp, cudaStream_t stream)
    {

        const int nlat_out  = params.nlat_out;
        const int nlon_out  = params.nlon_out;
        const int nchan_in  = params.nchan_in;
        const int nchan_out = params.nchan_out;

        dim3 block(WARP_SIZE, THREADS / WARP_SIZE);
        dim3 grid(DIV_UP(nlat_out*nlon_out, block.y), batch_size);

        size_t shsize = sizeof(FLOATV_T) * (nchan_in * 4 + nchan_out) * block.y; // 5 arrays per warp
        
        auto kern = &s2_attn_bwd_generic_vec_k<THREADS, FLOATV_T>;
        ensure_dyn_shmem(reinterpret_cast<const void *>(kern), shsize);

        kern<<<grid, block, shsize, stream>>>(params, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp);
        CHECK_ERROR("s2_attn_bwd_generic_vec_k");

        return;
    }

    // **************** end generic kernel ****************

    // **************** start long-rows specific kernels ****************

    // called with either (BDIM_X=32 and BDIM_Y>1) || (2^K=BDIM_X > 32 and BDIM_Y=1)
    template<int BDIM_X,
             int BDIM_Y,
             int NLOC,          // smallest int such that BDIM_X*NLOC >= nchan_in
             typename FLOATV_T>
    __global__
    __launch_bounds__(BDIM_X*BDIM_Y)
    void s2_attn_bwd_softmax_k(const __grid_constant__ attn_params_t p,
                               const int shcol_len_max,
                               const int nlat_max, // = n_long_rows
                               const FLOATV_T *__restrict__ kx,
                               const FLOATV_T *__restrict__ qy,
                               const int32_t  *__restrict__ row_idx,
                               const int64_t  *__restrict__ row_off,
                               const int64_t  *__restrict__ col_idx,
                                     float    *__restrict__ qdotk_max_buf) {
 
        static_assert(0 == (BDIM_X & (BDIM_X-1)));
        static_assert(0 == (BDIM_Y & (BDIM_Y-1)));
        static_assert((BDIM_X == WARP_SIZE && BDIM_Y  > 1) ||
                      (BDIM_X  > WARP_SIZE && BDIM_Y == 1));
 
        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;

        const int blk_per_row = gridDim.y; // blocks along Y process the same (ho,wo)
                                           // point by iteration over the (same) CSR
                                           // row in an interleaved fashion
        const int blk_split_id = blockIdx.y;

        const int batch = blockIdx.z;
        const uint64_t ctaid = uint64_t(blockIdx.x)*blockDim.y + threadIdx.y;
        
        const int &nchan_in = p.nchan_in;
        const int &nchan_out = p.nchan_out;
        const int &nlat_in = p.nlat_in;
        const int &nlon_in = p.nlon_in;
        const int &pscale = p.pscale;
        const int &nlat_out = p.nlat_out;
        const int &nlon_out = p.nlon_out;
 
        if (ctaid >= uint64_t(nlat_max)*nlon_out) return;
 
        alignas(float4) extern __shared__ float shext[];
#if 1
        using FLOATV_PTR_T = const FLOATV_T *;

        // chunked into 2 arrays: FLOATV_T sh_qy[BDIM_Y][nchan_in]
        //                        FLOATV_T *shkx_ptr[BDIM_Y][shcol_len_max]
        FLOATV_T     *base_fltv     = NULL;
        FLOATV_PTR_T *base_fltv_ptr = NULL;

        if constexpr (sizeof(FLOATV_T) > sizeof(FLOATV_PTR_T)) {
            base_fltv     = reinterpret_cast<FLOATV_T     *>(shext);
            base_fltv_ptr = reinterpret_cast<FLOATV_PTR_T *>(base_fltv+ BDIM_Y*nchan_in);
        } else {
            base_fltv_ptr = reinterpret_cast<FLOATV_PTR_T *>(shext);
            base_fltv     = reinterpret_cast<FLOATV_T     *>(base_fltv_ptr + BDIM_Y*shcol_len_max);
        }

        FLOATV_T        *sh_qy = base_fltv     + tidy*nchan_in + tidx;  // [nchan_in]
        FLOATV_PTR_T *shkx_ptr = base_fltv_ptr + tidy*shcol_len_max;    // [shcol_len_max]
#else
        FLOATV_T *sh_qy = reinterpret_cast<FLOATV_T *>(shext) + tidy*nchan_in + tidx;
#endif
        const int h  = ctaid / nlon_out;        // position in long-row list [0, nlat_max)
        const int wo = ctaid - h*nlon_out;
        const int ho = row_idx[h];              // actual row index [0, nlat_out)
 
        //const int pscale = nlon_in / nlon_out;
 
        kx += int64_t(batch)*nlat_in*nlon_in*nchan_in;// + tidx;
        qy += int64_t(batch)*nlat_out*nlon_out*nchan_in + int64_t(ho)*nlon_out*nchan_in + int64_t(wo)*nchan_in + tidx;
 
        const int64_t scratch_flat = int64_t(batch)*nlat_max*nlon_out + int64_t(h)*nlon_out + wo;

        qdotk_max_buf += scratch_flat;
    
        strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { sh_qy[i*BDIM_X] = qy[i*BDIM_X]; });
 
        float qdotk_max_local = -FLT_MAX;
 
        const int64_t rbeg = row_off[ho];
        const int64_t rend = row_off[ho + 1];
        const int     rlen = rend - rbeg;
#if 1
        const int rlen_div = rlen / blk_per_row;
        const int rlen_mod = rlen % blk_per_row;

        int n = rlen_div + (blk_split_id < rlen_mod);
        
        col_idx += rbeg + blk_split_id;

        for (int i = tidx; i < n; i += BDIM_X) {

            const int64_t col = col_idx[i*blk_per_row];

            const int hi = col / nlon_in;
            const int wi = col - (hi*nlon_in);
            const int wi_wo = wi + pscale*wo;
            const int wip = wi_wo - (wi_wo / nlon_in)*nlon_in;

            shkx_ptr[i] = kx + int64_t(hi)*nlon_in*nchan_in  + int64_t(wip)*nchan_in;
        }
        __group_sync<BDIM_X>();

        for (int i = 0; i < n; i++) {

            const FLOATV_T *_kx = shkx_ptr[i] + tidx;
#else
        col_idx += rbeg;

        for (int off = blk_split_id; off < rlen; off += blk_per_row) {

            const int64_t col = col_idx[off];

            const int hi    = col / nlon_in;
            const int wi    = col - hi*nlon_in;
            const int wi_wo = wi + pscale*wo;
            const int wip   = wi_wo - (wi_wo/nlon_in)*nlon_in;
 
            const FLOATV_T *_kx = kx + int64_t(hi)*nlon_in*nchan_in + int64_t(wip)*nchan_in;
#endif 
            FLOATV_T qdotk_v = __vset<FLOATV_T>(0.f);

            strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[i*BDIM_X], _kx[i*BDIM_X])); });
 
            float qdotk = __vred(qdotk_v);

            __group_sum<BDIM_X, BDIM_Y>(qdotk);

            qdotk_max_local = max(qdotk_max_local, qdotk);
        }
 
        if (!tidx) {
            atomicMax(qdotk_max_buf, qdotk_max_local);
        }

        return;
    }

    template<int BDIM_X,
             int BDIM_Y,
             int CHOUT_AS_IN,   // 1 iif BDIM_X*(NLOC-1) <= nchan_out <= BDIM_X*NLOC
             int NLOC,          // smallest int such that BDIM_X*NLOC >= nchan_in
             typename FLOATV_T>
    __global__
    __launch_bounds__(BDIM_X*BDIM_Y)
    void s2_attn_bwd_finalize_k(const __grid_constant__ attn_params_t p,
                                const int shcol_len_max,
                                const int nlat_max, // = n_long_rows
                                const FLOATV_T *__restrict__ kx,
                                const FLOATV_T *__restrict__ vx,
                                const FLOATV_T *__restrict__ qy,
                                const FLOATV_T *__restrict__ dy,
                                const int32_t  *__restrict__ row_idx,
                                const int64_t  *__restrict__ row_off,
                                const int64_t  *__restrict__ col_idx,
                                const float    *__restrict__ quad_weights,
                                const float    *__restrict__ qdotk_max_buf,
                                      float    *__restrict__ alpha_sum_buf,
                                      float    *__restrict__ integral_buf,    // unnormalized
                                      FLOATV_T *__restrict__ alpha_k_buf,     // [B,nlat_max,nlon_out,nchan_in]
                                      FLOATV_T *__restrict__ alpha_kvw_buf,
                                      int      *__restrict__ cta_done_buf,
                                      FLOATV_T *__restrict__ dqy) {
 
        static_assert(0 == (BDIM_X & (BDIM_X-1)));
        static_assert(0 == (BDIM_Y & (BDIM_Y-1)));
        static_assert((BDIM_X == WARP_SIZE && BDIM_Y  > 1) ||
                      (BDIM_X  > WARP_SIZE && BDIM_Y == 1));
 
        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;

        const int blk_per_row = gridDim.y; // blocks along Y process the same (ho,wo)
                                           // point by iteration over the (same) CSR
                                           // row in an interleaved fashion
        const int blk_split_id = blockIdx.y;

        const int batch = blockIdx.z;
        const uint64_t ctaid = uint64_t(blockIdx.x)*blockDim.y + threadIdx.y;
        
        const int &nchan_in  = p.nchan_in;
        const int &nchan_out = p.nchan_out;
        const int &nlat_in   = p.nlat_in;
        const int &nlon_in   = p.nlon_in;
        const int &pscale    = p.pscale;
        const int &nlat_out  = p.nlat_out;
        const int &nlon_out  = p.nlon_out;

        if (ctaid >= uint64_t(nlat_max)*nlon_out) return;
 
        // sh_dy[BDIM_Y][nchan_out], sh_qy[BDIM_Y][nchan_in]
        alignas(float4) extern __shared__ float shext[];
#if 1
        using FLOATV_PTR_T = const FLOATV_T *;

        // chunked into 5 arrays: FLOATV_T sh_dy[BDIM_Y][nchan_out]
        //                        FLOATV_T sh_qy[BDIM_Y][nchan_in]
        //                        FLOATV_T *shkx_ptr[BDIM_Y][shcol_len_max]
        //                        FLOATV_T *shvx_ptr[BDIM_Y][shcol_len_max]
        //                        float     shweight[BDIM_Y][shcol_len_max]
        FLOATV_T     *base_fltv     = NULL;
        FLOATV_PTR_T *base_fltv_ptr = NULL;
        float        *base_flt      = NULL;

        if constexpr (sizeof(FLOATV_T) > sizeof(FLOATV_PTR_T)) {
            base_fltv     = reinterpret_cast<FLOATV_T *>(shext);
            base_fltv_ptr = reinterpret_cast<FLOATV_PTR_T *>(base_fltv + BDIM_Y*(nchan_in + nchan_out));
            base_flt      = reinterpret_cast<float *>(base_fltv_ptr + BDIM_Y*2*shcol_len_max);
        } else {
            base_fltv_ptr = reinterpret_cast<FLOATV_PTR_T *>(shext);
            base_fltv     = reinterpret_cast<FLOATV_T *>(base_fltv_ptr + BDIM_Y*2*shcol_len_max);
            base_flt      = reinterpret_cast<float *>(base_fltv + BDIM_Y*(nchan_in + nchan_out));
        }

        FLOATV_T        *sh_dy = base_fltv                            + tidy*nchan_out;     // [nchan_out]
        FLOATV_T        *sh_qy = base_fltv     + BDIM_Y*nchan_out     + tidy*nchan_in;      // [nchan_in]
        FLOATV_PTR_T *shkx_ptr = base_fltv_ptr                        + tidy*shcol_len_max; // [shcol_len_max]
        FLOATV_PTR_T *shvx_ptr = base_fltv_ptr + BDIM_Y*shcol_len_max + tidy*shcol_len_max; // [shcol_len_max]
        float        *shweight = base_flt                             + tidy*shcol_len_max; // [shcol_len_max]
#else
        FLOATV_T *sh_dy = reinterpret_cast<FLOATV_T *>(shext) + tidy*(nchan_in + nchan_out);
        FLOATV_T *sh_qy = sh_dy + nchan_out;
#endif
        const int h  = ctaid / nlon_out;
        const int wo = ctaid - h*nlon_out;
        const int ho = row_idx[h];
 
        kx += int64_t(batch)*nlat_in*nlon_in*nchan_in;
        vx += int64_t(batch)*nlat_in*nlon_in*nchan_out;

        qy += int64_t(batch)*nlat_out*nlon_out*nchan_in  + int64_t(ho)*nlon_out*nchan_in  + int64_t(wo)*nchan_in;
        dy += int64_t(batch)*nlat_out*nlon_out*nchan_out + int64_t(ho)*nlon_out*nchan_out + int64_t(wo)*nchan_out;
            
        dqy += int64_t(batch)*nlat_out*nlon_out*nchan_in + int64_t(ho)*nlon_out*nchan_in + int64_t(wo)*nchan_in;

        const int64_t scratch_flat = int64_t(batch)*nlat_max*nlon_out + int64_t(h)*nlon_out + wo;

        qdotk_max_buf += scratch_flat;
        alpha_sum_buf += scratch_flat;
        integral_buf  += scratch_flat;

        cta_done_buf  += scratch_flat;

        alpha_k_buf   += scratch_flat*nchan_in;
        alpha_kvw_buf += scratch_flat*nchan_in;
 
        const float qdotk_max = qdotk_max_buf[0];
 
        // Stage qy and dy.
        strided_op<BDIM_X,               NLOC    >(nchan_in,  [&](int i) { sh_qy[i*BDIM_X + tidx] = qy[i*BDIM_X + tidx]; });
        strided_op<BDIM_X, CHOUT_AS_IN ? NLOC : 0>(nchan_out, [&](int i) { sh_dy[i*BDIM_X + tidx] = dy[i*BDIM_X + tidx]; });
 
        // Local accumulators (per-stripe).
        FLOATV_T loc_k__[NLOC];
        FLOATV_T loc_kvw[NLOC];

        strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { loc_k__[i] = __vset<FLOATV_T>(0.f); });
        strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { loc_kvw[i] = __vset<FLOATV_T>(0.f); });

        float alpha_sum_local = 0.f;
        float integral_local  = 0.f;
 
        const int64_t rbeg = row_off[ho];
        const int64_t rend = row_off[ho + 1];
        const int     rlen = rend - rbeg;
#if 1
        const int rlen_div = rlen / blk_per_row;
        const int rlen_mod = rlen % blk_per_row;

        int n = rlen_div + (blk_split_id < rlen_mod);
        
        col_idx += rbeg + blk_split_id;

        for (int i = tidx; i < n; i += BDIM_X) {

            const int64_t col = col_idx[i*blk_per_row];

            const int hi = col / nlon_in;
            const int wi = col - (hi*nlon_in);
            const int wi_wo = wi + pscale*wo;
            const int wip = wi_wo - (wi_wo / nlon_in)*nlon_in;

            shkx_ptr[i] = kx + int64_t(hi)*nlon_in*nchan_in  + int64_t(wip)*nchan_in;
            shvx_ptr[i] = vx + int64_t(hi)*nlon_in*nchan_out + int64_t(wip)*nchan_out;
            shweight[i] = quad_weights[hi];
        }
        __group_sync<BDIM_X>();

        for (int i = 0; i < n; i++) {

            const FLOATV_T *_kx = shkx_ptr[i];// + tidx;
            const FLOATV_T *_vx = shvx_ptr[i];// + tidx;
#else
        for (int off = blk_split_id; off < rlen; off += blk_per_row) {

            const int64_t col = col_idx[off];

            const int hi    = col / nlon_in;
            const int wi    = col - hi*nlon_in;
            const int wi_wo = wi + pscale*wo;
            const int wip   = wi_wo - (wi_wo/nlon_in)*nlon_in;
 
            const FLOATV_T *_kx = kx + int64_t(hi)*nlon_in*nchan_in  + int64_t(wip)*nchan_in;
            const FLOATV_T *_vx = vx + int64_t(hi)*nlon_in*nchan_out + int64_t(wip)*nchan_out;
#endif
            // cache kx in registers (reused for qdotk and for loc_k__loc_kvw updates).
            FLOATV_T loc_kx[NLOC];
            strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { loc_kx[i] = _kx[i*BDIM_X + tidx]; });
 
            FLOATV_T qdotk_v = __vset<FLOATV_T>(0.f);
            FLOATV_T gdotv_v = __vset<FLOATV_T>(0.f);
 
            strided_op<BDIM_X,               NLOC    >(nchan_in,  [&](int i) { qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[i*BDIM_X + tidx], loc_kx[i])); });
            strided_op<BDIM_X, CHOUT_AS_IN ? NLOC : 0>(nchan_out, [&](int i) { gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[i*BDIM_X + tidx], _vx[i*BDIM_X + tidx])); }); 

            float qdotk = __vred(qdotk_v);
            float gdotv = __vred(gdotv_v);

            __group_sum<BDIM_X, BDIM_Y>(qdotk, gdotv);
 
            const float alpha_inz  = expf(qdotk - qdotk_max) * shweight[i]; //quad_weights[hi];
            const float ainz_gdotv = alpha_inz * gdotv;
 
            alpha_sum_local += alpha_inz;
            integral_local  += ainz_gdotv;
    
            strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { loc_k__[i] = __vadd(loc_k__[i], __vscale(alpha_inz,  loc_kx[i])); });
            strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { loc_kvw[i] = __vadd(loc_kvw[i], __vscale(ainz_gdotv, loc_kx[i])); });
        }
 
        // add partials
        if (!tidx) {
            atomicAdd(alpha_sum_buf, alpha_sum_local);
            atomicAdd(integral_buf,  integral_local);
        }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
        constexpr bool DO_SPLIT_VEC = sizeof(FLOATV_T) / sizeof(float) == 4;
#else
        constexpr bool DO_SPLIT_VEC = false;
#endif
        if constexpr (DO_SPLIT_VEC) {
            strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) {
                atomicAdd(&alpha_k_buf[i*BDIM_X + tidx].x, loc_k__[i].x);
                atomicAdd(&alpha_k_buf[i*BDIM_X + tidx].y, loc_k__[i].y);
                atomicAdd(&alpha_k_buf[i*BDIM_X + tidx].z, loc_k__[i].z);
                atomicAdd(&alpha_k_buf[i*BDIM_X + tidx].w, loc_k__[i].w);
            } );
            
            strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) {
                atomicAdd(&alpha_kvw_buf[i*BDIM_X + tidx].x, loc_kvw[i].x);
                atomicAdd(&alpha_kvw_buf[i*BDIM_X + tidx].y, loc_kvw[i].y);
                atomicAdd(&alpha_kvw_buf[i*BDIM_X + tidx].z, loc_kvw[i].z);
                atomicAdd(&alpha_kvw_buf[i*BDIM_X + tidx].w, loc_kvw[i].w);
            });
        } else {
            strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { atomicAdd(alpha_k_buf   + i*BDIM_X + tidx, loc_k__[i]); });
            strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { atomicAdd(alpha_kvw_buf + i*BDIM_X + tidx, loc_kvw[i]); });
        }
    
        // last cta for each point writes dqy
        __threadfence();

        __shared__ int n_done_cta[BDIM_Y];

        if (!tidx) {
            n_done_cta[tidy] = atomicAdd(cta_done_buf, 1);
        }
        __group_sync<BDIM_X>();

        if (n_done_cta[tidy] == blk_per_row-1) {

            const float as      = alpha_sum_buf[0];
            const float intgr   = integral_buf [0];     // unnormalized
            const float as_inv  = (as > 0.f) ? 1.f/as : 0.f;
            const float as_inv_sq = as_inv*as_inv;

            strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) {
                dqy[i*BDIM_X + tidx] = __vscale(as_inv_sq, __vsub(__vscale(as, alpha_kvw_buf[i*BDIM_X + tidx]), __vscale(intgr, alpha_k_buf[i*BDIM_X + tidx])));
            });
        }
    }

    template<int BDIM_X,
             int BDIM_Y,
             int CHOUT_AS_IN,
             int NLOC,
             typename FLOATV_T>
    __global__
    __launch_bounds__(BDIM_X*BDIM_Y)
    void s2_attn_bwd_scatter_k(const __grid_constant__ attn_params_t p,
                               const int shcol_len_max,
                               const int nlat_max, // = n_long_rows
                               const FLOATV_T *__restrict__ kx,
                               const FLOATV_T *__restrict__ vx,
                               const FLOATV_T *__restrict__ qy,
                               const FLOATV_T *__restrict__ dy,
                               const int32_t  *__restrict__ row_idx,
                               const int64_t  *__restrict__ row_off,
                               const int64_t  *__restrict__ col_idx,
                               const float    *__restrict__ quad_weights,
                               const float    *__restrict__ qdotk_max_buf,
                               const float    *__restrict__ alpha_sum_buf,
                               const float    *__restrict__ integral_buf,    // unnormalized
                                     FLOATV_T *__restrict__ dkx,
                                     FLOATV_T *__restrict__ dvx) {
 
        static_assert(0 == (BDIM_X & (BDIM_X-1)));
        static_assert(0 == (BDIM_Y & (BDIM_Y-1)));
        static_assert((BDIM_X == WARP_SIZE && BDIM_Y  > 1) ||
                      (BDIM_X  > WARP_SIZE && BDIM_Y == 1));
 
        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;

        const int blk_per_row = gridDim.y; // blocks along Y process the same (ho,wo)
                                           // point by iteration over the (same) CSR
                                           // row in an interleaved fashion
        const int blk_split_id = blockIdx.y;

        const int batch = blockIdx.z;
        const uint64_t ctaid = uint64_t(blockIdx.x)*blockDim.y + threadIdx.y;
        
        const int &nchan_in = p.nchan_in;
        const int &nchan_out = p.nchan_out;
        const int &nlat_in = p.nlat_in;
        const int &nlon_in = p.nlon_in;
        const int &pscale = p.pscale;
        const int &nlat_out = p.nlat_out;
        const int &nlon_out = p.nlon_out;
 
        if (ctaid >= uint64_t(nlat_max)*nlon_out) return;
 
        // sh_dy[BDIM_Y][nchan_out], sh_qy[BDIM_Y][nchan_in]
        alignas(float4) extern __shared__ float shext[];
#if 1
        // chunked into 4 arrays: FLOATV_T sh_dy[BDIM_Y][nchan_out]
        //                        FLOATV_T sh_qy[BDIM_Y][nchan_in]
        //                        int64_t *shoff[BDIM_Y][shcol_len_max]
        //                        float     shweight[BDIM_Y][shcol_len_max]
        FLOATV_T *base_fltv = NULL;
        int64_t  *base_i64  = NULL;
        float    *base_flt  = NULL;

        if constexpr (sizeof(FLOATV_T) > sizeof(int64_t)) {
            base_fltv = reinterpret_cast<FLOATV_T *>(shext);
            base_i64  = reinterpret_cast<int64_t  *>(base_fltv + BDIM_Y*(nchan_in + nchan_out));
            base_flt  = reinterpret_cast<float    *>(base_i64  + BDIM_Y*shcol_len_max);
        } else {
            base_i64  = reinterpret_cast<int64_t  *>(shext);
            base_fltv = reinterpret_cast<FLOATV_T *>(base_i64  + BDIM_Y*shcol_len_max);
            base_flt  = reinterpret_cast<float    *>(base_fltv + BDIM_Y*(nchan_in + nchan_out));
        }

        FLOATV_T *sh_dy    = base_fltv                        + tidy*nchan_out;     // [nchan_out]
        FLOATV_T *sh_qy    = base_fltv + BDIM_Y*nchan_out     + tidy*nchan_in;      // [nchan_in]
        int64_t  *shoff    = base_i64                         + tidy*shcol_len_max; // [shcol_len_max]
        float    *shweight = base_flt                         + tidy*shcol_len_max; // [shcol_len_max]
#else
        FLOATV_T *sh_dy = reinterpret_cast<FLOATV_T *>(shext) + tidy*(nchan_in + nchan_out);
        FLOATV_T *sh_qy = sh_dy + nchan_out;
#endif
        const int h  = ctaid / nlon_out;
        const int wo = ctaid - h*nlon_out;
        const int ho = row_idx[h];
    
        kx  += int64_t(batch)*nlat_in*nlon_in*nchan_in;
        vx  += int64_t(batch)*nlat_in*nlon_in*nchan_out;
 
        qy += int64_t(batch)*nlat_out*nlon_out*nchan_in  + int64_t(ho)*nlon_out*nchan_in  + int64_t(wo)*nchan_in; 
        dy += int64_t(batch)*nlat_out*nlon_out*nchan_out + int64_t(ho)*nlon_out*nchan_out + int64_t(wo)*nchan_out;
 
        dkx += int64_t(batch)*nlat_in*nlon_in*nchan_in;
        dvx += int64_t(batch)*nlat_in*nlon_in*nchan_out;
 
        const int64_t scratch_flat = int64_t(batch)*nlat_max*nlon_out + int64_t(h)*nlon_out + wo;
 
        const float qdotk_max     = qdotk_max_buf[scratch_flat];
        const float alpha_sum     = alpha_sum_buf[scratch_flat];
        const float integral_un   = integral_buf [scratch_flat];
        const float alpha_sum_inv = (alpha_sum > 0.f) ? 1.f / alpha_sum : 0.f;
        const float integral_norm = integral_un * alpha_sum_inv;
 
        // Stage qy and dy.
        strided_op<BDIM_X,               NLOC    >(nchan_in,  [&](int i) { sh_qy[i*BDIM_X + tidx] = qy[i*BDIM_X + tidx]; });
        strided_op<BDIM_X, CHOUT_AS_IN ? NLOC : 0>(nchan_out, [&](int i) { sh_dy[i*BDIM_X + tidx] = dy[i*BDIM_X + tidx]; });

#if __CUDA_ARCH__ < 900
        if constexpr (std::is_same<FLOATV_T, float4>::value) {
            __group_sync<BDIM_X>();
        }
#endif
    
        const int64_t rbeg = row_off[ho];
        const int64_t rend = row_off[ho + 1];
        const int     rlen = rend - rbeg;
#if 1
        const int rlen_div = rlen / blk_per_row;
        const int rlen_mod = rlen % blk_per_row;

        int n = rlen_div + (blk_split_id < rlen_mod);

        col_idx += rbeg + blk_split_id;

        for (int i = tidx; i < n; i += BDIM_X) {

            const int64_t col = col_idx[i*blk_per_row];

            const int hi = col / nlon_in;
            const int wi = col - (hi*nlon_in);
            const int wi_wo = wi + pscale*wo;
            const int wip = wi_wo - (wi_wo / nlon_in)*nlon_in;

            shoff[i] = int64_t(hi)*nlon_in  + int64_t(wip);
            shweight[i] = quad_weights[hi];
        }
        __group_sync<BDIM_X>();

        for (int i = 0; i < n; i++) {

            const FLOATV_T *_kx = kx + shoff[i]*nchan_in;
            const FLOATV_T *_vx = vx + shoff[i]*nchan_out;
#else
        col_idx += rbeg;
 
        for (int i = blk_split_id; i < rlen; i += blk_per_row) {

            const int64_t col = col_idx[i];

            const int hi    = col / nlon_in;
            const int wi    = col - hi*nlon_in;
            const int wi_wo = wi + pscale*wo;
            const int wip   = wi_wo - (wi_wo/nlon_in)*nlon_in;
 
            const FLOATV_T *_kx = kx + int64_t(hi)*nlon_in*nchan_in  + int64_t(wip)*nchan_in;
            const FLOATV_T *_vx = vx + int64_t(hi)*nlon_in*nchan_out + int64_t(wip)*nchan_out;
#endif 
            FLOATV_T qdotk_v = __vset<FLOATV_T>(0.f);
            FLOATV_T gdotv_v = __vset<FLOATV_T>(0.f);
 
            strided_op<BDIM_X,               NLOC    >(nchan_in,  [&](int i) { qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[i*BDIM_X + tidx], _kx[i*BDIM_X + tidx])); });
            strided_op<BDIM_X, CHOUT_AS_IN ? NLOC : 0>(nchan_out, [&](int i) { gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[i*BDIM_X + tidx], _vx[i*BDIM_X + tidx])); });
 
            float qdotk = __vred(qdotk_v);
            float gdotv = __vred(gdotv_v);

            __group_sum<BDIM_X, BDIM_Y>(qdotk, gdotv);
 
            const float alpha_inz     = expf(qdotk - qdotk_max) * shweight[i]; //quad_weights[hi];
            const float alpha_mul     = alpha_inz * alpha_sum_inv;
            const float scale_fact_qy = (gdotv - integral_norm) * alpha_mul;
            const float scale_fact_dy =                           alpha_mul;
#if 1
            FLOATV_T *_dkx = dkx + shoff[i]*nchan_in; 
            FLOATV_T *_dvx = dvx + shoff[i]*nchan_out;
#else 
            FLOATV_T *_dkx = dkx + int64_t(hi)*nlon_in*nchan_in  + int64_t(wip)*nchan_in;
            FLOATV_T *_dvx = dvx + int64_t(hi)*nlon_in*nchan_out + int64_t(wip)*nchan_out;
#endif
            constexpr int VEC_SIZE = sizeof(FLOATV_T)/sizeof(float);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
            constexpr bool DO_SPLIT_VEC = VEC_SIZE == 4;
#else
            constexpr bool DO_SPLIT_VEC = false;
#endif
            if constexpr (DO_SPLIT_VEC) {
                float *sh_qy_scl = reinterpret_cast<float *>(sh_qy);
                float *sh_dy_scl = reinterpret_cast<float *>(sh_dy);
                float *_dkx_scl  = reinterpret_cast<float *>(_dkx);
                float *_dvx_scl  = reinterpret_cast<float *>(_dvx);

                for (int chan = tidx; chan < nchan_in*VEC_SIZE; chan += BDIM_X) {
                    atomicAdd(_dkx_scl + chan, scale_fact_qy * sh_qy_scl[chan]);
                }
                for (int chan = tidx; chan < nchan_out*VEC_SIZE; chan += BDIM_X) {
                    atomicAdd(_dvx_scl + chan, scale_fact_dy * sh_dy_scl[chan]);
                }
            } else {
                strided_op<BDIM_X,               NLOC    >(nchan_in,  [&](int i) { atomicAdd(_dkx + i*BDIM_X + tidx, __vscale(scale_fact_qy, sh_qy[i*BDIM_X + tidx])); });
                strided_op<BDIM_X, CHOUT_AS_IN ? NLOC : 0>(nchan_out, [&](int i) { atomicAdd(_dvx + i*BDIM_X + tidx, __vscale(scale_fact_dy, sh_dy[i*BDIM_X + tidx])); });
            }
        }
        return;
    }

    template<int BDIM_X,
             int LOC_SIZE,
             typename FLOATV_T>
    void spc_attn_bwd_long_rows(attn_params_t params,
                                int64_t n_long_rows,
                                int64_t max_row_len,
                                int batch_size, 
                                FLOATV_T *_kxp, FLOATV_T *_vxp, FLOATV_T *_qyp, FLOATV_T *_dyp,
                                int32_t *_row_idx, int64_t *_row_off, int64_t *_col_idx, float *_quad_weights,
                                FLOATV_T *_dkxp, FLOATV_T *_dvxp, FLOATV_T *_dqyp, cudaStream_t stream) {

        static_assert(0 == (BDIM_X & (BDIM_X - 1)));
        static_assert(std::is_same<FLOATV_T, float>::value ||
                      std::is_same<FLOATV_T, float4>::value);

        if (!n_long_rows) {
            return;
        }

        //const int nlat_oublockIdx.yt = params.nlat_out;
        const int nlat_out  = params.nlat_out;
        const int nlon_out  = params.nlon_out;
        const int nchan_in  = params.nchan_in;
        const int nchan_out = params.nchan_out;

        const bool chout_as_in = (nchan_out >= BDIM_X * (LOC_SIZE-1) &&
                                  nchan_out <= BDIM_X *  LOC_SIZE   );

        constexpr int BDIM_Y = (BDIM_X <= WARP_SIZE) ? THREADS / BDIM_X : 1;

        dim3 block(BDIM_X, BDIM_Y);

        // temporary, should this be passed into the module like qdotk_max_buf?

        constexpr int VEC_SIZE = sizeof(FLOATV_T) / sizeof(float);

        auto opts_f = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
        auto opts_i = at::TensorOptions().dtype(at::kInt).device(at::kCUDA);

        at::Tensor qdotk_max_t = at::full ({batch_size, n_long_rows, nlon_out}, -FLT_MAX, opts_f);
        at::Tensor alpha_sum_t = at::zeros({batch_size, n_long_rows, nlon_out}, opts_f);
        at::Tensor integral_t  = at::zeros({batch_size, n_long_rows, nlon_out}, opts_f);
        at::Tensor alpha_k_t   = at::zeros({batch_size, n_long_rows, nlon_out, nchan_in * VEC_SIZE}, opts_f);
        at::Tensor alpha_kvw_t = at::zeros({batch_size, n_long_rows, nlon_out, nchan_in * VEC_SIZE}, opts_f);
        at::Tensor done_cnt_t  = at::zeros({batch_size, n_long_rows, nlon_out},  opts_i);

        float    *_qdotk_max = qdotk_max_t.data_ptr<float>();
        float    *_alpha_sum  = alpha_sum_t.data_ptr<float>();
        float    *_integral   = integral_t .data_ptr<float>();
        FLOATV_T *_alpha_k    = reinterpret_cast<FLOATV_T *>(alpha_k_t  .data_ptr<float>());
        FLOATV_T *_alpha_kvw  = reinterpret_cast<FLOATV_T *>(alpha_kvw_t.data_ptr<float>());
        int      *_done_count = done_cnt_t .data_ptr<int>();

        const int cta_per_row = min(int64_t(SPLIT_LONG_ROW_MAX_BLK_X_ROW),
                                    DIV_UP(max_row_len, SPLIT_LONG_ROW_MIN_WORK_X_BLK));

        dim3 grid_lr(DIV_UP(n_long_rows*nlon_out, block.y), cta_per_row, batch_size); // softmax+finalize+scatter grid

        const int max_niter_cta = DIV_UP(max_row_len, cta_per_row);
#if 0
        //printf("getPtxver(): %d\n", getPtxver());
        printf("n_long_rows: %ld, max_row_len: %ld, cta_per_row: %d, max_niter_cta: %d\n",
                n_long_rows, max_row_len, cta_per_row, max_niter_cta);
        printf("Launching s2_attn_fwd_softmax_k<%d, %d, %d><<<(%u, %u, %u), (%u, %u), ...>>>\n",
                BDIM_X, BDIM_Y, LOC_SIZE, grid_lr.x, grid_lr.y, grid_lr.z, block.x, block.y);
#endif

        { // softmax
            size_t shsize = (sizeof(FLOATV_T)*nchan_in + sizeof(FLOATV_T *)*max_niter_cta) * block.y;

            auto kern = &s2_attn_bwd_softmax_k<BDIM_X, BDIM_Y, LOC_SIZE, FLOATV_T>;
            ensure_dyn_shmem(reinterpret_cast<const void *>(kern), shsize);

            kern<<<grid_lr, block, shsize, stream>>>(params,
                                                     max_niter_cta, n_long_rows,
                                                     _kxp, _qyp,
                                                     _row_idx, _row_off, _col_idx,
                                                     _qdotk_max);
            CHECK_ERROR("s2_attn_fwd_softmax_k");
        }
        { // finalize: write dqy + atomic_accumul _alpha_k, _alpha_kvw
            size_t shsize = (sizeof(FLOATV_T)*(nchan_in + nchan_out) +
                             sizeof(FLOATV_T *)*max_niter_cta*2 +
                             sizeof(float)*max_niter_cta) * block.y;
            if (chout_as_in) {

                auto kern = &s2_attn_bwd_finalize_k<BDIM_X, BDIM_Y, 1, LOC_SIZE, FLOATV_T>;
                ensure_dyn_shmem(reinterpret_cast<const void*>(kern), shsize);

                kern<<<grid_lr, block, shsize, stream>>>(params, max_niter_cta, n_long_rows,
                        _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights,
                        _qdotk_max, _alpha_sum, _integral, _alpha_k, _alpha_kvw, _done_count,
                        _dqyp);
            } else {

                auto kern = &s2_attn_bwd_finalize_k<BDIM_X, BDIM_Y, 0, LOC_SIZE, FLOATV_T>;
                ensure_dyn_shmem(reinterpret_cast<const void*>(kern), shsize);

                kern<<<grid_lr, block, shsize, stream>>>(params, max_niter_cta, n_long_rows,
                        _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights,
                        _qdotk_max, _alpha_sum, _integral, _alpha_k, _alpha_kvw, _done_count,
                        _dqyp);
            }
            CHECK_ERROR("s2_attn_bwd_finalize_k");
        }
        { // scatter into _dkxp, _dvxp
            size_t shsize = (sizeof(FLOATV_T)*(nchan_in + nchan_out) +
                             sizeof(int64_t)*max_niter_cta /* *2 */ +
                             sizeof(float)*max_niter_cta) * block.y;
            if (chout_as_in) {

                auto kern = &s2_attn_bwd_scatter_k<BDIM_X, BDIM_Y, 1, LOC_SIZE, FLOATV_T>;
                ensure_dyn_shmem(reinterpret_cast<const void*>(kern), shsize);

                kern<<<grid_lr, block, shsize, stream>>>(params, max_niter_cta, n_long_rows,
                        _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights,
                        _qdotk_max, _alpha_sum, _integral, _dkxp, _dvxp);
            } else {

                auto kern = &s2_attn_bwd_scatter_k<BDIM_X, BDIM_Y, 0, LOC_SIZE, FLOATV_T>;
                ensure_dyn_shmem(reinterpret_cast<const void*>(kern), shsize);

                kern<<<grid_lr, block, shsize, stream>>>(params, max_niter_cta, n_long_rows,
                        _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights,
                        _qdotk_max, _alpha_sum, _integral, _dkxp, _dvxp);
            }
            CHECK_ERROR("s2_attn_bwd_scatter_k");
        }
        return;
    }

    // **************** end long-rows specific kernels ****************

    
    // **************** start specialized kernel ****************

    // called with either (BDIM_X=32 and BDIM_Y>1) || (2^K=BDIM_X > 32 and BDIM_Y=1)
    template <int BDIM_X, int BDIM_Y,
              int CHOUT_AS_IN,   // 1 iif "BDIM_X*(NLOC-1) <= nchan_out <= BDIM_X*NLOC" else 0
              int NLOC,          // smallest int such that BDIM_X*NLOC >= nchan_in
              typename FLOATV_T> // either float or float4
    __global__ __launch_bounds__(BDIM_X *BDIM_Y)
    void s2_attn_bwd_special_vec_k(const __grid_constant__ attn_params_t p,
                                   const int shcol_len_max,
                                   const int nlat_max, 
                                   const FLOATV_T *__restrict__ kx,    // [batch][nlat_in][nlon_in][nchan_in]
                                   const FLOATV_T *__restrict__ vx,    // [batch][nlat_in][nlon_in][nchan_out]
                                   const FLOATV_T *__restrict__ qy,    // [batch][nlat_out][nlon_out][nchan_in]
                                   const FLOATV_T *__restrict__ dy,    // [batch][nlat_out][nlon_out][nchan_out]
                                   const int32_t *__restrict__ row_idx,
                                   const int64_t *__restrict__ row_off,
                                   const int64_t *__restrict__ col_idx,
                                   const float *__restrict__ quad_weights,
                                         FLOATV_T *__restrict__ dkx,   // [batch][nlat_in][nlon_in][nchan_in]
                                         FLOATV_T *__restrict__ dvx,   // [batch][nlat_in][nlon_in][nchan_out]
                                         FLOATV_T *__restrict__ dqy) { // [batch][nlat_out][nlon_out][nchan_in]

        static_assert(0 == (BDIM_X & (BDIM_X - 1)));
        static_assert(0 == (BDIM_Y & (BDIM_Y - 1)));
        static_assert((BDIM_X == 32 && BDIM_Y > 1) || (BDIM_X > 32 && BDIM_Y == 1));

        constexpr int NLOC_M1 = NLOC - 1;

        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;

        const int batch = blockIdx.y;
        const uint64_t ctaid = uint64_t(blockIdx.x)*blockDim.y + threadIdx.y;

        const int &nchan_in = p.nchan_in;
        const int &nchan_out = p.nchan_out;
        const int &nlat_in = p.nlat_in;
        const int &nlon_in = p.nlon_in;
        const int &pscale = p.pscale;
        const int &nlat_out = p.nlat_out;
        const int &nlon_out = p.nlon_out;

        if (ctaid >= uint64_t(/*nlat_out*/nlat_max)*nlon_out) { return; }

        extern __shared__ __align__(sizeof(float4)) float shext[];
#if 1
        // chunked into 4 arrays: FLOATV_T sh_dy[BDIM_Y][nchan_out]
        //                        FLOATV_T sh_qy[BDIM_Y][nchan_in]
        //                        int64_t *shoff[BDIM_Y][shcol_len_max]
        //                        float shweight[BDIM_Y][shcol_len_max]
        FLOATV_T *base_fltv = NULL;
        int64_t  *base_i64  = NULL;
        float    *base_flt  = NULL;

        if constexpr (sizeof(FLOATV_T) > sizeof(int64_t)) {
            base_fltv = reinterpret_cast<FLOATV_T *>(shext);
            base_i64  = reinterpret_cast<int64_t  *>(base_fltv + BDIM_Y*(nchan_in + nchan_out));
            base_flt  = reinterpret_cast<float    *>(base_i64  + BDIM_Y*shcol_len_max);
        } else {
            base_i64  = reinterpret_cast<int64_t  *>(shext);
            base_fltv = reinterpret_cast<FLOATV_T *>(base_i64  + BDIM_Y*shcol_len_max);
            base_flt  = reinterpret_cast<float    *>(base_fltv + BDIM_Y*(nchan_in + nchan_out));
        }

        FLOATV_T *sh_dy    = base_fltv                        + tidy*nchan_out;     // [nchan_out]
        FLOATV_T *sh_qy    = base_fltv + BDIM_Y*nchan_out     + tidy*nchan_in;      // [nchan_in]
        int64_t  *shoff    = base_i64                         + tidy*shcol_len_max; // [shcol_len_max]
        float    *shweight = base_flt                         + tidy*shcol_len_max; // [shcol_len_max]
#else
        // sh_dy[nchan_out], sh_qy[nchan_in]
        FLOATV_T *sh_dy = reinterpret_cast<FLOATV_T *>(shext) + threadIdx.y*(nchan_in + nchan_out);
        FLOATV_T *sh_qy = sh_dy + nchan_out;
#endif
        // for dqy
        FLOATV_T loc_k__[NLOC];
        FLOATV_T loc_vw_[NLOC];
        FLOATV_T loc_kvw[NLOC];

        // use permuted rows
        const int h = ctaid / nlon_out;
        const int wo = ctaid - (h*nlon_out);
        const int ho = row_idx[h];

        // one output lon step corresponds to pscale input lon steps (requires nlon_in % nlon_out == 0)
        //const int pscale = nlon_in / nlon_out;

        // offset input tensors
        kx += int64_t(batch)*nlat_in*nlon_in*nchan_in;
        qy += int64_t(batch)*nlat_out*nlon_out*nchan_in + int64_t(ho)*nlon_out*nchan_in + int64_t(wo)*nchan_in;

        vx += int64_t(batch)*nlat_in*nlon_in*nchan_out;
        dy += int64_t(batch)*nlat_out*nlon_out*nchan_out + int64_t(ho)*nlon_out*nchan_out + int64_t(wo)*nchan_out;

        // offset output tensors
        dkx += int64_t(batch)*nlat_in*nlon_in*nchan_in;
        dvx += int64_t(batch)*nlat_in*nlon_in*nchan_out;
        
        dqy += int64_t(batch)*nlat_out*nlon_out*nchan_in + int64_t(ho)*nlon_out*nchan_in + int64_t(wo)*nchan_in;

        #pragma unroll
        for (int i = 0; i < NLOC; i++) {
            loc_k__[i] = __vset<FLOATV_T>(0.0f);
            loc_vw_[i] = __vset<FLOATV_T>(0.0f);
            loc_kvw[i] = __vset<FLOATV_T>(0.0f);
        }

        strided_op<BDIM_X,               NLOC    >(nchan_in,  [&](int i) { sh_qy[i*BDIM_X + tidx] = qy[i*BDIM_X + tidx]; });
        strided_op<BDIM_X, CHOUT_AS_IN ? NLOC : 0>(nchan_out, [&](int i) { sh_dy[i*BDIM_X + tidx] = dy[i*BDIM_X + tidx]; });

#if __CUDA_ARCH__ < 900
        // for architectures < 9.0, sh_dy and sh_qy will be read
        // as individual floats at the end of the kernel, which
        // breaks the assumption that each FLOATV_T location is
        // written to and read by the same thread throughout the
        // kernel, in the case FLOATV_T==float4
        if constexpr (std::is_same<FLOATV_T, float4>::value) {
            __group_sync<BDIM_X>();
        }
#endif
        // for dkx, dvx, dqy
        float alpha_sum = 0.0f;
        float qdotk_max = -FLT_MAX;

        // for dkx
        float integral = 0.0f;

        const int64_t rbeg = row_off[ho];
        const int64_t rend = row_off[ho + 1];
        const int     rlen = rend - rbeg;
#if 1
        col_idx += rbeg;

        for (int i = tidx; i < rlen; i += BDIM_X) {

            const int64_t col = col_idx[i];

            const int hi = col / nlon_in;
            const int wi = col - (hi*nlon_in);
            const int wi_wo = wi + pscale*wo;
            const int wip = wi_wo - (wi_wo / nlon_in)*nlon_in;

            shoff[i] = int64_t(hi)*nlon_in  + int64_t(wip);

            shweight[i] = quad_weights[hi];
        }
        __group_sync<BDIM_X>();

        for (int i = 0; i < rlen; i++) {

            const FLOATV_T *_kx = kx + shoff[i]*nchan_in;
            const FLOATV_T *_vx = vx + shoff[i]*nchan_out;
#else
        col_idx += rbeg;

        // accumulate alpha_sum, integral, and shared stats,
        // along with a progressively computed qdotk_max.
        for (int off = 0; off < rlen; off++) {

            const int64_t col = col_idx[off];

            const int hi = col / nlon_in;
            const int wi = col - (hi*nlon_in);
            const int wi_wo = wi + pscale*wo;
            const int wip = wi_wo - (wi_wo / nlon_in)*nlon_in;

            const FLOATV_T *_kx = kx + int64_t(hi)*nlon_in*nchan_in  + int64_t(wip)*nchan_in;
            const FLOATV_T *_vx = vx + int64_t(hi)*nlon_in*nchan_out + int64_t(wip)*nchan_out;
#endif
            FLOATV_T qdotk_v = __vset<FLOATV_T>(0.0f);
            FLOATV_T gdotv_v = __vset<FLOATV_T>(0.0f);

            strided_op<BDIM_X,               NLOC    >(nchan_in,  [&](int i) { qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[i*BDIM_X + tidx], _kx[i*BDIM_X + tidx])); });
            strided_op<BDIM_X, CHOUT_AS_IN ? NLOC : 0>(nchan_out, [&](int i) { gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[i*BDIM_X + tidx], _vx[i*BDIM_X + tidx])); });

            float qdotk = __vred(qdotk_v);
            float gdotv = __vred(gdotv_v);

            __group_sum<BDIM_X, BDIM_Y>(qdotk, gdotv);

            const float qdotk_max_tmp = max(qdotk_max, qdotk);
            const float alpha_inz = expf(qdotk - qdotk_max_tmp)*shweight[i]; //quad_weights[hi];
            const float max_correction = expf(qdotk_max - qdotk_max_tmp);

            alpha_sum = alpha_sum*max_correction + alpha_inz;
            integral = integral*max_correction + alpha_inz*gdotv;

            const float ainz_gdotv = alpha_inz*gdotv;

            strided_op<BDIM_X, NLOC>(nchan_in,  [&](int i) { loc_k__[i] = __vadd(__vscale(max_correction, loc_k__[i]), __vscale(alpha_inz,  _kx[i*BDIM_X + tidx]));  });
            strided_op<BDIM_X, NLOC>(nchan_in,  [&](int i) { loc_kvw[i] = __vadd(__vscale(max_correction, loc_kvw[i]), __vscale(ainz_gdotv, _kx[i*BDIM_X + tidx])); });
            strided_op<BDIM_X, NLOC>(nchan_in,  [&](int i) { loc_vw_[i] = __vadd(__vscale(max_correction, loc_vw_[i]), __vset<FLOATV_T>(ainz_gdotv));        });

            qdotk_max = qdotk_max_tmp;
        }

        const float alpha_sum_inv = 1.0f / alpha_sum;

        integral *= alpha_sum_inv;

        const float alpha_sum_inv_sq = alpha_sum_inv*alpha_sum_inv;

        strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) {
            dqy[i*BDIM_X + tidx] = __vscale(alpha_sum_inv_sq, __vsub(__vscale(alpha_sum, loc_kvw[i]), __vmul(loc_vw_[i], loc_k__[i])));
        });
#if 1
        for (int i = 0; i < rlen; i++) {

            const FLOATV_T *_kx = kx + shoff[i]*nchan_in;
            const FLOATV_T *_vx = vx + shoff[i]*nchan_out;
#else
        // accumulate gradients for k and v
        for (int off = 0; off < rlen; off++) {

            const int64_t col = col_idx[off];

            const int hi = col / nlon_in;
            const int wi = col - (hi*nlon_in);
            const int wi_wo = wi + pscale*wo;
            const int wip = wi_wo - (wi_wo / nlon_in)*nlon_in;

            const FLOATV_T *_kx = kx + int64_t(hi)*nlon_in*nchan_in + int64_t(wip)*nchan_in;
            const FLOATV_T *_vx = vx + int64_t(hi)*nlon_in*nchan_out + int64_t(wip)*nchan_out;
#endif
            FLOATV_T qdotk_v = __vset<FLOATV_T>(0.0f);
            FLOATV_T gdotv_v = __vset<FLOATV_T>(0.0f);

            strided_op<BDIM_X,               NLOC    >(nchan_in,  [&](int i) { qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[i*BDIM_X + tidx], _kx[i*BDIM_X + tidx])); });
            strided_op<BDIM_X, CHOUT_AS_IN ? NLOC : 0>(nchan_out, [&](int i) { gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[i*BDIM_X + tidx], _vx[i*BDIM_X + tidx])); });

            float qdotk = __vred(qdotk_v);
            float gdotv = __vred(gdotv_v);

            __group_sum<BDIM_X, BDIM_Y>(qdotk, gdotv);

            const float alpha_inz = expf(qdotk - qdotk_max)*shweight[i]; //quad_weights[hi];
#if 1
            FLOATV_T *_dkx = dkx + shoff[i]*nchan_in;
            FLOATV_T *_dvx = dvx + shoff[i]*nchan_out;
#else
            FLOATV_T *_dkx = dkx + int64_t(hi)*nlon_in*nchan_in + int64_t(wip)*nchan_in;
            FLOATV_T *_dvx = dvx + int64_t(hi)*nlon_in*nchan_out + int64_t(wip)*nchan_out;
#endif
            const float alpha_mul = alpha_inz*alpha_sum_inv;

            const float scale_fact_qy = (gdotv - integral)*alpha_mul;
            const float scale_fact_dy = alpha_mul;

            // float4, 128-bit atomics are only supported by devices of compute
            // capability 9.x+, so on older devices we resort to 32-bit atomics

            constexpr int VEC_SIZE = sizeof(FLOATV_T) / sizeof(float);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
            constexpr bool DO_SPLIT_VEC = VEC_SIZE == 4;
#else
            constexpr bool DO_SPLIT_VEC = false;
#endif
            if constexpr(DO_SPLIT_VEC) {
                // making the loop count known at compile time doesn't seem
                // to make any difference here so let's keep this (much)
                // simpler version
                float *sh_qy_scl = reinterpret_cast<float *>(sh_qy);
                float *sh_dy_scl = reinterpret_cast<float *>(sh_dy);

                float *_dkx_scl = reinterpret_cast<float *>(_dkx);
                float *_dvx_scl = reinterpret_cast<float *>(_dvx);

                // 32-bit, consecutive atomics to glmem
                // strided atomics results in a severe slowdown
                for (int chan = tidx; chan < nchan_in*VEC_SIZE; chan += BDIM_X) {
                    atomicAdd(_dkx_scl + chan, scale_fact_qy*sh_qy_scl[chan]);
                }
                for (int chan = tidx; chan < nchan_out*VEC_SIZE; chan += BDIM_X) {
                    atomicAdd(_dvx_scl + chan, scale_fact_dy*sh_dy_scl[chan]);
                }
            } else {

                strided_op<BDIM_X,               NLOC    >(nchan_in,  [&](int i) { atomicAdd(_dkx + i*BDIM_X + tidx, __vscale(scale_fact_qy, sh_qy[i*BDIM_X + tidx])); });
                strided_op<BDIM_X, CHOUT_AS_IN ? NLOC : 0>(nchan_out, [&](int i) { atomicAdd(_dvx + i*BDIM_X + tidx, __vscale(scale_fact_dy, sh_dy[i*BDIM_X + tidx])); });
            }
        }

        return;
    }

    template <int BDIM_X, int CUR_LOC_SIZE,
              int MAX_LOC_SIZE, // max size of FLOATV_T[] local array
              typename FLOATV_T>
    void launch_spc_attn_bwd(attn_params_t params, int nloc, // "BDIM_X*nloc" >= nchans_out
                             int batch_size, 
                             FLOATV_T *_kxp, FLOATV_T *_vxp, FLOATV_T *_qyp, FLOATV_T *_dyp,
                             int32_t *_row_idx, int64_t *_row_off, int64_t *_col_idx, float *_quad_weights,
                             FLOATV_T *_dkxp, FLOATV_T *_dvxp, FLOATV_T *_dqyp, cudaStream_t stream)
    {

        if (CUR_LOC_SIZE == nloc) {

            const int nlat_out  = params.nlat_out;
            const int nlon_out  = params.nlon_out;
            const int nchan_in  = params.nchan_in;
            const int nchan_out = params.nchan_out;

            int64_t n_long_rows;
            int64_t max_row_len;
            int64_t mid_row_len;

            // splits the rows in "long" and "short" rows; long rows have
            // a length >= max(SPLIT_LONG_ROW_MIN_LEN(1024), len(row_0))
            // (since the rows are sorted in decreasing order, row_0 is the
            // longest row); short rows are the remaining rows.
            // If there are long rows, they are processed with a separate
            // kernels using multiple blocks per row, in order to mitigate
            // the imbalance causing long temporal tails.
            split_csr_rows(SPLIT_ROW_LENGTH_THRES, SPLIT_LONG_ROW_MIN_LEN,
                           nlat_out, _row_idx, _row_off, 
                           &n_long_rows, &max_row_len, &mid_row_len);

            //printf("%s:%d: n_long_rows: %ld, max_row_len: %ld, mid_row_len: %ld\n", __func__, __LINE__, n_long_rows, max_row_len, mid_row_len);

            if (n_long_rows > 0) {
                // processes the "long rows", from _row_idx[0] to _row_idx[n_long_rows-1]
                spc_attn_bwd_long_rows<BDIM_X, CUR_LOC_SIZE>(params, n_long_rows, max_row_len, batch_size, _kxp,
                                                             _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx,
                                                             _quad_weights, _dkxp, _dvxp, _dqyp, stream);
            }

            int64_t n_reg_rows = nlat_out - n_long_rows;
            if (!n_reg_rows) { return; }
            // process the "short rows", from _row_idx[n_long_rows] to _row_idx[nlat_out-1]

            // nloc determines the size of local arrays used to store
            // temporary buffers loc_k__[], loc_vw_[] and loc_kvw[],
            // of size nchan_in each;
            // if nchan_out is >= BDIM_X*(nloc-1) and <= BDIM_X*nloc
            // then we can use the same compile-time known loops used
            // for input channels, with the exception of testing
            // whether to execute the last iteration based on "nchan_out"
            // instead of "nchan_in"; in this way as long as the
            // difference between the number of input and output channels
            // is <= BDIM_X we can use the faster path
            const bool chout_as_in = (nchan_out >= BDIM_X * (CUR_LOC_SIZE-1) &&
                                      nchan_out <= BDIM_X *  CUR_LOC_SIZE   );

            constexpr int BDIM_Y = (BDIM_X <= WARP_SIZE) ? THREADS / BDIM_X : 1;

            dim3 block(BDIM_X, BDIM_Y);
            dim3 grid(DIV_UP(/*nlat_out*/n_reg_rows*nlon_out, block.y), batch_size);
            
            size_t shsize = (sizeof(FLOATV_T)*(nchan_in + nchan_out) +
                             sizeof(int64_t)*mid_row_len +
                             sizeof(float  )*mid_row_len)* block.y; 

            if (chout_as_in) {
                auto kern = &s2_attn_bwd_special_vec_k<BDIM_X, BDIM_Y, 1, CUR_LOC_SIZE, FLOATV_T>;
                ensure_dyn_shmem(reinterpret_cast<const void *>(kern), shsize);

                kern<<<grid, block, shsize, stream>>>(params, mid_row_len, n_reg_rows, 
                                                      _kxp, _vxp, _qyp, _dyp, _row_idx+n_long_rows,
                                                      _row_off, _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp);
            } else {
                auto kern = &s2_attn_bwd_special_vec_k<BDIM_X, BDIM_Y, 0, CUR_LOC_SIZE, FLOATV_T>;
                ensure_dyn_shmem(reinterpret_cast<const void *>(kern), shsize);

                kern<<<grid, block, shsize, stream>>>(params, mid_row_len, n_reg_rows,
                                                      _kxp, _vxp, _qyp, _dyp, _row_idx+n_long_rows,
                                                      _row_off, _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp);
            }
            CHECK_ERROR("s2_attn_bwd_special_vec_k");

            return;
        }
        if constexpr (CUR_LOC_SIZE < MAX_LOC_SIZE) {
            launch_spc_attn_bwd<BDIM_X, CUR_LOC_SIZE + 1, MAX_LOC_SIZE>(params, nloc, batch_size,
                                                                        _kxp, _vxp, _qyp, _dyp,
                                                                        _row_idx, _row_off, _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp, stream);
        }
        return;
    }
    
    // **************** end specialized kernel ****************

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

        attn_params_t params = {0};

        params.nchan_in = nchans_in;
        params.nchan_out = nchans_out;
        params.nlat_in = nlat_in;
        params.nlon_in = nlon_in;
        params.nlat_out = nlat_out;
        params.nlon_out = nlon_out;
        params.pscale = nlon_in / nlon_out;

        if (!is_aligned<sizeof(float4)>(_kxp) || !is_aligned<sizeof(float4)>(_vxp) || !is_aligned<sizeof(float4)>(_qyp)
            || !is_aligned<sizeof(float4)>(_dyp) || !is_aligned<sizeof(float4)>(_dkxp)
            || !is_aligned<sizeof(float4)>(_dvxp) || !is_aligned<sizeof(float4)>(_dqyp) || (nchans_in % VEC_SIZE) != 0
            || (nchans_out % VEC_SIZE) != 0) {

            const int nloc = DIV_UP(nchans_in, bdimx);

            // to avoid the compilation of unused template instances;
            // we use a block size BDIM_X that is the smallest power of 2
            // such that BDIM_X*MAX_LOCAL_ARR_LEN >= nchans_in, so
            // BDIM_X > 32 are used only for:
            //
            //  (BDIM_X-1)*MAX_LOCAL_ARR_LEN < nchans_in <= BDIM_X*MAX_LOCAL_ARR_LEN
            constexpr int MIN_LOC_ARR_LEN = MAX_LOCAL_ARR_LEN / 2 + 1;

            // use 2D blocks only if 32 threads are enough; w.r.t fowrard,
            // we use the special kernel only up to BDIM_X=512 as with 1024
            // each thread cannot use more than 64 registers, resulting in
            // large amounts of registers spills
            switch (bdimx) {
                case  32: launch_spc_attn_bwd< 32,               1, MAX_LOCAL_ARR_LEN>(params, nloc, batch_size, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp, stream); break;
                case  64: launch_spc_attn_bwd< 64, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(params, nloc, batch_size, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp, stream); break;
                case 128: launch_spc_attn_bwd<128, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(params, nloc, batch_size, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp, stream); break;
                case 256: launch_spc_attn_bwd<256, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(params, nloc, batch_size, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp, stream); break;
                case 512: launch_spc_attn_bwd<512, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(params, nloc, batch_size, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp, stream); break;
                default:  launch_gen_attn_bwd                                         (params,       batch_size, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp, stream); break;
            }

        } else {

            float4 *_kxp4 = reinterpret_cast<float4 *>(kxP.data_ptr());
            float4 *_vxp4 = reinterpret_cast<float4 *>(vxP.data_ptr());
            float4 *_qyp4 = reinterpret_cast<float4 *>(qyP.data_ptr());
            float4 *_dyp4 = reinterpret_cast<float4 *>(dyP.data_ptr());

            float4 *_dkxp4 = reinterpret_cast<float4 *>(dkxP.data_ptr());
            float4 *_dvxp4 = reinterpret_cast<float4 *>(dvxP.data_ptr());
            float4 *_dqyp4 = reinterpret_cast<float4 *>(dqyP.data_ptr());

            nchans_in /= VEC_SIZE;
            nchans_out /= VEC_SIZE;
            
            params.nchan_in = nchans_in;
            params.nchan_out = nchans_out;

            const int nloc = DIV_UP(nchans_in, bdimx);

            constexpr int MAX_LOCAL_VEC_LEN = MAX_LOCAL_ARR_LEN / VEC_SIZE;

            constexpr int MIN_LOC_VEC_LEN = MAX_LOCAL_VEC_LEN / 2 + 1;

            // use 2D blocks only if 32 threads are enough
            switch (bdimx) {
                case  32: launch_spc_attn_bwd< 32,               1, MAX_LOCAL_VEC_LEN>(params, nloc, batch_size, _kxp4, _vxp4, _qyp4, _dyp4, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp4, _dvxp4, _dqyp4, stream); break;
                case  64: launch_spc_attn_bwd< 64, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(params, nloc, batch_size, _kxp4, _vxp4, _qyp4, _dyp4, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp4, _dvxp4, _dqyp4, stream); break;
                case 128: launch_spc_attn_bwd<128, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(params, nloc, batch_size, _kxp4, _vxp4, _qyp4, _dyp4, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp4, _dvxp4, _dqyp4, stream); break;
                case 256: launch_spc_attn_bwd<256, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(params, nloc, batch_size, _kxp4, _vxp4, _qyp4, _dyp4, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp4, _dvxp4, _dqyp4, stream); break;
                case 512: launch_spc_attn_bwd<512, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(params, nloc, batch_size, _kxp4, _vxp4, _qyp4, _dyp4, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp4, _dvxp4, _dqyp4, stream); break;
                default:  launch_gen_attn_bwd                                         (params,       batch_size, _kxp4, _vxp4, _qyp4, _dyp4, _row_idx, _row_off, _col_idx, _quad_weights, _dkxp4, _dvxp4, _dqyp4, stream); break;
            }
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

        if (downsample) {
            s2_attn_bwd_dispatch(batch_size, nchans_in, nchans_out, nlon_in, nlat_out, nlon_out, kxP, vxP, qyP, dyP,
                                 psi_row_off, psi_col_idx, quad_weights, dkxP, dvxP, dqyP);
        } else {
            s2_attn_bwd_upsample_dispatch(batch_size, nchans_in, nchans_out, nlon_in, nlat_in, nlat_out, nlon_out, kxP,
                                          vxP, qyP, dyP, psi_row_off, psi_col_idx, quad_weights, dkxP, dvxP, dqyP);
        }

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

    TORCH_LIBRARY_IMPL(attention_kernels, CUDA, m) { m.impl("backward", &s2_attention_bwd_dkvq_cuda); }

} // namespace attention_kernels
