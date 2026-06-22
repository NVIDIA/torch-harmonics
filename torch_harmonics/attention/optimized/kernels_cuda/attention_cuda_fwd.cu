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

#define MAX_LOCAL_ARR_LEN (16)

//#define USE_SPLIT_ROW_FWD

// BEGIN - forward kernels and functions

namespace attention_kernels
{

    void dump_csr_linear(const char *fname, int64_t nrows, at::Tensor row_idx, at::Tensor row_off, at::Tensor col_idx);

    // scatter-direction launcher, defined in attention_cuda_fwd_upsample.cu;
    // called by s2_attention_fwd_cuda when nlon_out % nlon_in == 0.
    void s2_attn_fwd_upsample_dispatch(int batch_size, size_t nchans_in, size_t nchans_out, int64_t nlon_in,
                                       int64_t nlat_in, int64_t nlat_out, int64_t nlon_out, torch::Tensor kxP,
                                       torch::Tensor vxP, torch::Tensor qyP, torch::Tensor psi_row_off,
                                       torch::Tensor psi_col_idx, torch::Tensor quad_weights, torch::Tensor yP);

    // **************** start generic kernel ****************

    // called with (blockDim.x=32 and blockDim.y>1, BDIM_X=blockDim.x*blockDim.y)
    template <int BDIM_X,
              typename FLOATV_T> // either float or float4
    __global__ __launch_bounds__(BDIM_X) void s2_attn_fwd_generic_k(
        const __grid_constant__ attn_params_t p,
        const FLOATV_T *__restrict__ kx,
        const FLOATV_T *__restrict__ vx, const FLOATV_T *__restrict__ qy, const int32_t *__restrict__ row_idx,
        const int64_t *__restrict__ row_off, const int64_t *__restrict__ col_idx,
        const float *__restrict__ quad_weights, FLOATV_T *__restrict__ y)
    {
        const int &nchan_in = p.nchan_in;
        const int &nchan_out = p.nchan_out;
        const int &nlat_in = p.nlat_in;
        const int &nlon_in = p.nlon_in;
        const int &pscale = p.pscale;
        const int &nlat_out = p.nlat_out;
        const int &nlon_out = p.nlon_out;

        extern __shared__ __align__(sizeof(float4)) float shext[];
        FLOATV_T *shy = reinterpret_cast<FLOATV_T *>(shext) + threadIdx.y*nchan_out;

        const int batch = blockIdx.y;
        const int wid = blockIdx.x*blockDim.y + threadIdx.y;

        if (wid >= nlat_out*nlon_out) { return; }

        const int tidx = threadIdx.x;

        const int h = wid / nlon_out;
        const int wo = wid - (h*nlon_out);
        const int ho = row_idx[h];

        // one output lon step corresponds to pscale input lon steps (requires nlon_in % nlon_out == 0)

        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { shy[chan] = __vset<FLOATV_T>(0.f); }

        kx += int64_t(batch)*nlat_in*nlon_in*nchan_in;
        qy += int64_t(batch)*nlat_out*nlon_out*nchan_in + int64_t(ho)*nchan_in*nlon_out
            + int64_t(wo)*nchan_in;

        vx += int64_t(batch)*nlat_in*nlon_in*nchan_out;
        y += int64_t(batch)*nlat_out*nlon_out*nchan_out + int64_t(ho)*nchan_out*nlon_out
            + int64_t(wo)*nchan_out;

        float alpha_sum = 0.0f;
        float qdotk_max = -FLT_MAX;

        const int64_t rbeg = row_off[ho];
        const int64_t rend = row_off[ho + 1];

        col_idx += rbeg;

        const int rlen = rend - rbeg;

        for (int off = 0; off < rlen; off++) {

            const int64_t col = col_idx[off];

            const int hi = col / nlon_in;
            const int wi = col - (hi*nlon_in);
            const int wi_wo = wi + pscale*wo;
            const int wip = wi_wo - (wi_wo / nlon_in)*nlon_in;

            const FLOATV_T *_kx = kx + int64_t(hi)*nlon_in*nchan_in + int64_t(wip)*nchan_in;
            const FLOATV_T *_vx = vx + int64_t(hi)*nlon_in*nchan_out + int64_t(wip)*nchan_out;

            FLOATV_T qdotkv = __vset<FLOATV_T>(0.f);

            for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
                qdotkv = __vadd(qdotkv, __vmul(qy[chan], _kx[chan]));
            }

            float qdotk = __warp_sum(__vred(qdotkv));

            float qdotk_max_tmp;
            float alpha;
            float exp_save;

            qdotk_max_tmp = max(qdotk_max, qdotk);
            alpha = expf(qdotk - qdotk_max_tmp)*quad_weights[hi];
            exp_save = expf(qdotk_max - qdotk_max_tmp);

            alpha_sum = alpha + alpha_sum*exp_save;

            for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
                shy[chan] = __vadd(__vscale(exp_save, shy[chan]), __vscale(alpha, _vx[chan]));
            }
            qdotk_max = qdotk_max_tmp;
        }

        alpha_sum = 1.0f / alpha_sum;
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { y[chan] = __vscale(alpha_sum, shy[chan]); }

        return;
    }

    template <typename FLOATV_T>
    void launch_gen_attn_fwd(attn_params_t params, int batch_size,
                             FLOATV_T *__restrict__ _kxp, FLOATV_T *__restrict__ _vxp,
                             FLOATV_T *__restrict__ _qyp, int32_t *_row_idx, int64_t *_row_off, int64_t *_col_idx,
                             float *_quad_weights, FLOATV_T *__restrict__ _yp, cudaStream_t stream)
    {
        const int nlat_out = params.nlat_out;
        const int nlon_out = params.nlon_out;
        const int nchans_out = params.nchan_out;

        dim3 block(WARP_SIZE, THREADS / WARP_SIZE);
        dim3 grid(DIV_UP(nlat_out*nlon_out, block.y), batch_size);

        size_t shsize = sizeof(FLOATV_T)*nchans_out*block.y;
        
        auto kern = &s2_attn_fwd_generic_k<THREADS, FLOATV_T>;
        ensure_dyn_shmem(reinterpret_cast<const void *>(kern), shsize);

        kern<<<grid, block, shsize, stream>>>(params, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp);
        CHECK_ERROR("s2_attn_fwd_generic_k");

        return;
    }
    
    // **************** end generic kernel ****************

    // **************** start long-rows specific kernels ****************

    template <int BDIM_X,
              int BDIM_Y,
              int NLOC,          // smallest int such that BDIM_X*NLOC >= nchan_out
              typename FLOATV_T> // either float or float4
    __global__ __launch_bounds__(BDIM_X *BDIM_Y)
    void s2_attn_fwd_softmax_k(const __grid_constant__ attn_params_t p,
                               const int shcol_len_max,
                               const int nlat_max,
                               const FLOATV_T *__restrict__ kx,         // [batch][nlat_in][nlon_in][nchan_in]
                               const FLOATV_T *__restrict__ qy,         // [batch][nlat_out][nlon_out][nchan_in]
                               const int32_t  *__restrict__ row_idx,
                               const int64_t  *__restrict__ row_off,
                               const int64_t  *__restrict__ col_idx,
                                     float    *__restrict__ qdotk_max_buf,
                                     FLOATV_T *__restrict__ y) {        // [batch][nlat_out][nlon_out][nchan_out] (out)

        static_assert(0 == (BDIM_X & (BDIM_X - 1)));
        static_assert(0 == (BDIM_Y & (BDIM_Y - 1)));
        static_assert((BDIM_X == WARP_SIZE && BDIM_Y > 1) || (BDIM_X > WARP_SIZE && BDIM_Y == 1));

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

        if (ctaid >= uint64_t(nlat_max)*nlon_out) { return; }

        const int h  = ctaid / nlon_out;
        const int wo = ctaid - (h*nlon_out);
        const int ho = row_idx[h];

        kx += int64_t(batch)*nlat_in*nlon_in*nchan_in;
        qy += int64_t(batch)*nlat_out*nlon_out*nchan_in  + int64_t(ho)*nlon_out*nchan_in  + int64_t(wo)*nchan_in  + tidx;
        y  += int64_t(batch)*nlat_out*nlon_out*nchan_out + int64_t(ho)*nlon_out*nchan_out + int64_t(wo)*nchan_out + tidx;
        
        // zero y vectors for finalize kernel...
        if (blk_split_id == 0) {
            strided_op<BDIM_X, NLOC>(nchan_out, [&](int i) { y[i*BDIM_X] = __vset<FLOATV_T>(0); });
        }

        alignas(float4) extern __shared__ float shext[];

        // just to simplify the seatup of the shared memory layout
        using FLOATV_PTR_T = const FLOATV_T *;

        FLOATV_T *shqy = NULL;
        FLOATV_PTR_T *shkx_ptr = NULL;

        if constexpr (sizeof(FLOATV_T) > sizeof(FLOATV_PTR_T)) {
            FLOATV_T *base = reinterpret_cast<FLOATV_T *>(shext);
            shqy = base + tidy*nchan_in;
            shkx_ptr = reinterpret_cast<FLOATV_PTR_T *>(base + BDIM_Y*nchan_in) + tidy*shcol_len_max;
        } else {
            FLOATV_PTR_T *base = reinterpret_cast<FLOATV_PTR_T *>(shext);
            shkx_ptr = base + tidy*shcol_len_max;
            shqy = reinterpret_cast<FLOATV_T *>(base + BDIM_Y*shcol_len_max) + tidy*nchan_in;
        }
        shqy += tidx;

        strided_op<BDIM_X, 0>(nchan_in, [&](int i) { shqy[i*BDIM_X] = qy[i*BDIM_X]; });

        //const int64_t out_flat = int64_t(batch)*nlat_out*nlon_out + int64_t(ho)*nlon_out + wo;
        const int64_t out_flat = int64_t(batch)*nlat_max*nlon_out + int64_t(h)*nlon_out + wo;

        qdotk_max_buf += out_flat;

        float qdotk_max = -FLT_MAX;

        const int64_t rbeg = row_off[ho];
        const int64_t rend = row_off[ho + 1];
        const int rlen = rend - rbeg;

        col_idx += rbeg + blk_split_id;

        const int rlen_div = rlen / blk_per_row;
        const int rlen_mod = rlen % blk_per_row;

        int n = rlen_div + (blk_split_id < rlen_mod);

        for (int i = tidx; i < n; i += BDIM_X) {

            const int64_t col = col_idx[i*blk_per_row];

            const int hi = col / nlon_in;
            const int wi = col - (hi*nlon_in);
            const int wi_wo = wi + pscale*wo;
            const int wip = wi_wo - (wi_wo / nlon_in)*nlon_in;

            shkx_ptr[i] = kx + int64_t(hi)*nlon_in*nchan_in + int64_t(wip)*nchan_in;
        }

        __group_sync<BDIM_X>();

        for (int i = 0; i < n; i++) {

            const FLOATV_T *__restrict__ _kx = shkx_ptr[i] + tidx;

            FLOATV_T qdotk_v = __vset<FLOATV_T>(0.0f);

            strided_op<BDIM_X, 0>(nchan_in,
                                  [&](int i) { qdotk_v = __vadd(qdotk_v, __vmul(shqy[i*BDIM_X], _kx[i*BDIM_X])); });

            float qdotk = __vred(qdotk_v);
            __group_sum<BDIM_X, BDIM_Y>(qdotk);
            qdotk_max = max(qdotk_max, qdotk);
        }

        if (!tidx) {
                atomicMax(qdotk_max_buf, qdotk_max);
        }

        return;
    }

    template <int BDIM_X,
              int BDIM_Y,
              int CHIN_AS_OUT,   // 1 iif "BDIM_X*(NLOC-1) <= nchan_out <= BDIM_X*NLOC" else 0
              int NLOC,          // smallest int such that BDIM_X*NLOC >= nchan_in
              typename FLOATV_T> // either float or float4
    __global__ __launch_bounds__(BDIM_X *BDIM_Y)
    void s2_attn_fwd_finalize_k(const __grid_constant__ attn_params_t p,
                                const int shcol_len_max,
                                const int nlat_max,
                                const FLOATV_T *__restrict__ kx,         // [batch][nlat_halo][nlon_kx][nchan_in]
                                const FLOATV_T *__restrict__ vx,         // [batch][nlat_halo][nlon_kx][nchan_out]
                                const FLOATV_T *__restrict__ qy,         // [batch][nlat_out][nlon_out][nchan_in]
                                const int32_t *__restrict__ row_idx,   
                                const int64_t *__restrict__ row_off,
                                const int64_t *__restrict__ col_idx,
                                const float *__restrict__ quad_weights,
                                const float *__restrict__ qdotk_max_buf, // [batch][nlat_out][nlon_out] (in)
                                      float *__restrict__ alpha_sum_buf, // [batch][nlat_out][nlon_out] (in)
                                      int   *__restrict__ cta_done_buf,  // [batch][nlat_out][nlon_out] (in)
                                      FLOATV_T *__restrict__ y) {        // [batch][nlat_out][nlon_out][nchan_out] (in/out)

        static_assert(0 == (BDIM_X & (BDIM_X - 1)));
        static_assert(0 == (BDIM_Y & (BDIM_Y - 1)));
        static_assert((BDIM_X == WARP_SIZE && BDIM_Y > 1) || (BDIM_X > WARP_SIZE && BDIM_Y == 1));

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

        if (ctaid >= uint64_t(nlat_max)*nlon_out) { return; }

        alignas(float4) extern __shared__ float shext[];

        // just to simplify the seatup of the shared memory layout
        using FLOATV_PTR_T = const FLOATV_T *;

        // chunked into 3 arrays: FLOATV_T shq[BDIM_Y][nchan_in]
        //                        int64_t  shoff[BDIM_Y][shcol_len_max]
        //                        float    shweight[BDIM_Y][shcol_len_max]
        FLOATV_T *base_fltv     = NULL;
        FLOATV_PTR_T *base_fltv_ptr = NULL;
        //int64_t  *base_i64 = NULL;
        float    *base_flt      = NULL;

        if constexpr (sizeof(FLOATV_T) > sizeof(FLOATV_PTR_T)) {
            base_fltv     = reinterpret_cast<FLOATV_T     *>(shext);
            base_fltv_ptr = reinterpret_cast<FLOATV_PTR_T *>(base_fltv     + BDIM_Y*nchan_in);
            base_flt      = reinterpret_cast<float        *>(base_fltv_ptr + BDIM_Y*2*shcol_len_max);
            //base_i64 = reinterpret_cast<int64_t *>(base_fltv     + BDIM_Y*nchan_in);
            //base_flt      = reinterpret_cast<float        *>(base_i64 + BDIM_Y*shcol_len_max);
        } else {
            base_fltv_ptr = reinterpret_cast<FLOATV_PTR_T *>(shext);
            base_fltv     = reinterpret_cast<FLOATV_T     *>(base_fltv_ptr + BDIM_Y*2*shcol_len_max);
            //base_i64 = reinterpret_cast<int64_t *>(shext);
            //base_fltv     = reinterpret_cast<FLOATV_T     *>(base_i64 + BDIM_Y*shcol_len_max);
            base_flt      = reinterpret_cast<float        *>(base_fltv     + BDIM_Y*nchan_in);
        }

        FLOATV_T *shq      = base_fltv + tidy*nchan_in;      // [nchan_in]
        FLOATV_PTR_T *shkx_ptr = base_fltv_ptr +                        tidy*shcol_len_max; // [shcol_len_max]
        FLOATV_PTR_T *shvx_ptr = base_fltv_ptr + BDIM_Y*shcol_len_max + tidy*shcol_len_max; // [shcol_len_max]
        //int64_t  *shoff    = base_i64  + tidy*shcol_len_max; // [shcol_len_max]
        float    *shweight = base_flt  + tidy*shcol_len_max; // [shcol_len_max]

        shq += tidx;

        const int h  = ctaid / nlon_out;
        const int wo = ctaid - (h*nlon_out);
        const int ho = row_idx[h];

        kx += int64_t(batch)*nlat_in*nlon_in*nchan_in;
        vx += int64_t(batch)*nlat_in*nlon_in*nchan_out;

        qy += int64_t(batch)*nlat_out*nlon_out*nchan_in  + int64_t(ho)*nlon_out*nchan_in  + int64_t(wo)*nchan_in  + tidx;
        y  += int64_t(batch)*nlat_out*nlon_out*nchan_out + int64_t(ho)*nlon_out*nchan_out + int64_t(wo)*nchan_out + tidx;

        //const int64_t out_flat = int64_t(batch)*nlat_out*nlon_out + int64_t(ho)*nlon_out + wo;
        const int64_t out_flat = int64_t(batch)*nlat_max*nlon_out + int64_t(h)*nlon_out + wo;

        qdotk_max_buf += out_flat;
        alpha_sum_buf += out_flat;
        cta_done_buf  += out_flat;

        // Load current state
        float alpha_sum = 0;
        const float qdotk_max = qdotk_max_buf[0];

        FLOATV_T locy[NLOC];
        strided_op<BDIM_X, NLOC>(nchan_out, [&](int i) { locy[i] = __vset<FLOATV_T>(0); });

        strided_op<BDIM_X, CHIN_AS_OUT ? NLOC : 0>(nchan_in, [&](int i) { shq[i*BDIM_X] = qy[i*BDIM_X]; });

        // `pscale` is the GLOBAL ratio nlon_in / nlon_out_global, passed by the caller.
        // Computing it here as `nlon_in / nlon_out` would be wrong because the kernel's
        // `nlon_out` is the LOCAL output width (nlon_out_local), not the global one.

        const int64_t rbeg = row_off[ho];
        const int64_t rend = row_off[ho + 1];
        const int     rlen = rend - rbeg;

        const int rlen_div = rlen / blk_per_row;
        const int rlen_mod = rlen % blk_per_row;

        int n = rlen_div + (blk_split_id < rlen_mod);
        
        col_idx += rbeg + blk_split_id;
        //col_idx += rbeg + blk_split_id*rlen_div + min(blk_split_id, rlen_mod);

        for (int i = tidx; i < n; i += BDIM_X) {

            const int64_t col = col_idx[i*blk_per_row];

            const int hi = col / nlon_in;
            const int wi = col - (hi*nlon_in);
            const int wi_wo = wi + pscale*wo;
            const int wip = wi_wo - (wi_wo / nlon_in)*nlon_in;

            shkx_ptr[i] = kx + int64_t(hi)*nlon_in*nchan_in  + int64_t(wip)*nchan_in;
            shvx_ptr[i] = vx + int64_t(hi)*nlon_in*nchan_out + int64_t(wip)*nchan_out;
            //shoff[i] = int64_t(hi)*nlon_in + int64_t(wip); 

            shweight[i] = quad_weights[hi];
        }
        __group_sync<BDIM_X>();

        for (int i = 0; i < n; i++) {

            const FLOATV_T *_kx = shkx_ptr[i] + tidx;
            const FLOATV_T *_vx = shvx_ptr[i] + tidx;
            //const FLOATV_T *_kx = kx + shoff[i]*nchan_in  + tidx;
            //const FLOATV_T *_vx = vx + shoff[i]*nchan_out + tidx;

            FLOATV_T qdotkv = __vset<FLOATV_T>(0.f);
            strided_op<BDIM_X, CHIN_AS_OUT ? NLOC : 0>(nchan_in, [&](int i) { qdotkv = __vadd(qdotkv, __vmul(shq[i*BDIM_X], _kx[i*BDIM_X])); });

            float qdotk = __vred(qdotkv);
            __group_sum<BDIM_X, BDIM_Y>(qdotk);

            const float alpha = expf(qdotk - qdotk_max)*shweight[i];
            //const float alpha = expf(qdotk - qdotk_max)*quad_weights[hi];

            alpha_sum += alpha;

            strided_op<BDIM_X, NLOC>(nchan_out,
                                     [&](int i) { locy[i] = __vadd(locy[i], __vscale(alpha, _vx[i*BDIM_X])); });
        }

        // Store updated state
        if (!tidx) {
            atomicAdd(alpha_sum_buf, alpha_sum);
        }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
        constexpr bool DO_SPLIT_VEC = sizeof(FLOATV_T) / sizeof(float) == 4;
#else
        constexpr bool DO_SPLIT_VEC = false;
#endif
        if constexpr (DO_SPLIT_VEC) {
            strided_op<BDIM_X, NLOC>(nchan_out, [&](int i) {
                atomicAdd(&y[i*BDIM_X].x, locy[i].x);
                atomicAdd(&y[i*BDIM_X].y, locy[i].y);
                atomicAdd(&y[i*BDIM_X].z, locy[i].z);
                atomicAdd(&y[i*BDIM_X].w, locy[i].w);
            });
        } else {
            strided_op<BDIM_X, NLOC>(nchan_out, [&](int i) { atomicAdd(y + i*BDIM_X, locy[i]); });
        }

        // last cta for each point performs the normalization
        __threadfence();

        __shared__ int n_done_cta[BDIM_Y];

        if (!tidx) {
            n_done_cta[tidy] = atomicAdd(cta_done_buf, 1);
        }
        __group_sync<BDIM_X>();

        if (n_done_cta[tidy] == blk_per_row-1) {
            const float as     = alpha_sum_buf[0];
            const float as_inv = (as > 0.f) ? 1.f / as : 0.f;
            strided_op<BDIM_X, NLOC>(nchan_out, [&](int i) { y[i*BDIM_X] = __vscale(as_inv, y[i*BDIM_X]); });
        }

        return;
    }

    template<int BDIM_X,
             int LOC_SIZE,
             typename FLOATV_T>
    void spc_attn_fwd_long_rows(attn_params_t params,
                                int64_t n_long_rows,
                                int64_t max_row_len,
                                int64_t batch_size,
                                FLOATV_T *_kxp, FLOATV_T *_vxp, FLOATV_T *_qyp, int32_t *_row_idx,
                                int64_t *_row_off, int64_t *_col_idx, float *_quad_weights, FLOATV_T *_yp,
                                cudaStream_t stream) {

        static_assert(0 == (BDIM_X & (BDIM_X - 1)));
        static_assert(std::is_same<FLOATV_T, float>::value || std::is_same<FLOATV_T, float4>::value);

        if (!n_long_rows) {
            return;
        }

        //const int nlat_oublockIdx.yt = params.nlat_out;
        const int nlon_out   = params.nlon_out;
        const int nchans_in  = params.nchan_in;
        const int nchans_out = params.nchan_out;

        const bool chin_as_out = (nchans_in >= BDIM_X*(LOC_SIZE-1) &&
                                  nchans_in <= BDIM_X* LOC_SIZE   );

        constexpr int BDIM_Y = (BDIM_X <= WARP_SIZE) ? THREADS / BDIM_X : 1;

        // temporary, should this be passed into the module like qdotk_max_buf?
        auto opts_f = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
        auto opts_i = at::TensorOptions().dtype(at::kInt).device(at::kCUDA);

        at::Tensor qdotk_max_t = at::full ({batch_size, n_long_rows, nlon_out}, -FLT_MAX, opts_f);
        at::Tensor alpha_sum_t = at::zeros({batch_size, n_long_rows, nlon_out},           opts_f);
        at::Tensor cta_done_t  = at::zeros({batch_size, n_long_rows, nlon_out},           opts_i);

        float *_alpha_sum = alpha_sum_t.data_ptr<float>();
        int    *_cta_done = cta_done_t.data_ptr<   int>();

        float *_qdotk_max = qdotk_max_t.data_ptr<float>();

        const int cta_per_row = min( int64_t(SPLIT_LONG_ROW_MAX_BLK_X_ROW),
                                     max(1l, DIV_UP(max_row_len, SPLIT_LONG_ROW_MIN_WORK_X_BLK)) );

        dim3 block(BDIM_X, BDIM_Y);
        dim3 grid(DIV_UP(n_long_rows*nlon_out, block.y), cta_per_row, batch_size); // softmax+finalize grid

        const int max_niter_cta = DIV_UP(max_row_len, cta_per_row);
#if 0
        //printf("getPtxver(): %d\n", getPtxver());
        printf("n_long_rows: %ld, max_row_len: %ld, cta_per_row: %d, max_niter_cta: %d\n",
                n_long_rows, max_row_len, cta_per_row, max_niter_cta);
        printf("Launching s2_attn_fwd_softmax_k<%d, %d, %d><<<(%u, %u, %u), (%u, %u), ...>>>\n",
                BDIM_X, BDIM_Y, LOC_SIZE, grid.x, grid.y, grid.z, block.x, block.y);
#endif
        size_t shsize = (sizeof(FLOATV_T)*nchans_in + sizeof(FLOATV_T *)*max_niter_cta) * block.y;

        auto kern = &s2_attn_fwd_softmax_k<BDIM_X, BDIM_Y, LOC_SIZE, FLOATV_T>;
        ensure_dyn_shmem(reinterpret_cast<const void *>(kern), shsize);

        kern<<<grid, block, shsize, stream>>>(params,
                                              max_niter_cta, n_long_rows,
                                              _kxp, _qyp, _row_idx, _row_off, _col_idx, _qdotk_max, _yp);
        CHECK_ERROR("s2_attn_fwd_softmax_k");

        shsize = (sizeof(FLOATV_T)*nchans_in + sizeof(FLOATV_T *)*max_niter_cta*2 + sizeof(float)*max_niter_cta) * block.y;
        //shsize = (sizeof(FLOATV_T)*nchans_in + sizeof(int64_t)*max_niter_cta + sizeof(float)*max_niter_cta) * block.y;

        // atomicAdd into alpha_sum_t and _yp (as y_acc)
        if (chin_as_out) {
            auto kern = &s2_attn_fwd_finalize_k<BDIM_X, BDIM_Y, 1, LOC_SIZE, FLOATV_T>;
            ensure_dyn_shmem(reinterpret_cast<const void *>(kern), shsize);

            kern<<<grid, block, shsize, stream>>>(params, max_niter_cta, n_long_rows, _kxp, _vxp, _qyp, _row_idx,
                                                  _row_off, _col_idx, _quad_weights, _qdotk_max, _alpha_sum, _cta_done, _yp);
        } else {
            auto kern = &s2_attn_fwd_finalize_k<BDIM_X, BDIM_Y, 0, LOC_SIZE, FLOATV_T>;
            ensure_dyn_shmem(reinterpret_cast<const void *>(kern), shsize);

            kern<<<grid, block, shsize, stream>>>(params, max_niter_cta, n_long_rows, _kxp, _vxp, _qyp, _row_idx,
                                                  _row_off, _col_idx, _quad_weights, _qdotk_max, _alpha_sum, _cta_done, _yp);
        }
        CHECK_ERROR("s2_attn_fwd_finalize_k");

        return;
    }

    // **************** end long-rows specific kernels ****************

    // **************** start specialized kernel ****************

    // called with either (BDIM_X=32 and BDIM_Y>1) || (2^K=BDIM_X > 32 and BDIM_Y=1)
    template <int BDIM_X,
              int BDIM_Y,
              int CHIN_AS_OUT,   // 1 iif "BDIM_X*(NLOC-1) <= nchan_in <= BDIM_X*NLOC" else 0
              int NLOC,          // smallest int such that BDIM_X*NLOC >= nchan_out
              typename FLOATV_T> // either float or float4
    __global__ __launch_bounds__(BDIM_X*BDIM_Y)
    void s2_attn_fwd_special_k(const __grid_constant__ attn_params_t p,
                               const int shcol_len_max,
                               const int nlat_max, 
                               const FLOATV_T *__restrict__ kx,
                               const FLOATV_T *__restrict__ vx,
                               const FLOATV_T *__restrict__ qy,
                               const int32_t *__restrict__ row_idx,
                               const int64_t *__restrict__ row_off,
                               const int64_t *__restrict__ col_idx,
                               const float *__restrict__ quad_weights,
                                     FLOATV_T *__restrict__ y) {

        static_assert(0 == (BDIM_X & (BDIM_X - 1)));
        static_assert(0 == (BDIM_Y & (BDIM_Y - 1)));
        static_assert((BDIM_X == 32 && BDIM_Y > 1) || (BDIM_X > 32 && BDIM_Y == 1));

        constexpr int NLOC_M1 = NLOC - 1;

        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;

        const int batch = blockIdx.y;
        const int ctaid = blockIdx.x*blockDim.y + threadIdx.y;

        const int &nchan_in = p.nchan_in;
        const int &nchan_out = p.nchan_out;
        const int &nlat_in = p.nlat_in;
        const int &nlon_in = p.nlon_in;
        const int &pscale = p.pscale;
        const int &nlat_out = p.nlat_out;
        const int &nlon_out = p.nlon_out;

        if (ctaid >= /*nlat_out*/nlat_max*nlon_out) { return; }

        FLOATV_T locy[NLOC];

        extern __shared__ __align__(sizeof(float4)) float shext[];
#ifdef USE_SPLIT_ROW_FWD
        // just to simplify the seatup of the shared memory layout
        //using FLOATV_PTR_T = const FLOATV_T *;

        // chunked into 3 arrays: FLOATV_T shq[BDIM_Y][nchan_in]
        //                        int64_t  shoff[BDIM_Y][shcol_len_max]
        //                        float    shweight[BDIM_Y][shcol_len_max]
        FLOATV_T     *base_fltv     = NULL;
        int64_t      *base_i64 = NULL;
        float        *base_flt      = NULL;

        if constexpr(sizeof(FLOATV_T) > sizeof(int64_t)) {
            base_fltv     = reinterpret_cast<FLOATV_T     *>(shext);
            base_i64      = reinterpret_cast<int64_t      *>(base_fltv     + BDIM_Y*nchan_in);
            base_flt      = reinterpret_cast<float        *>(base_i64 + BDIM_Y*shcol_len_max);
        } else {
            base_i64      = reinterpret_cast<int64_t *>(shext);
            base_fltv     = reinterpret_cast<FLOATV_T     *>(base_i64 + BDIM_Y*shcol_len_max);
            base_flt      = reinterpret_cast<float        *>(base_fltv     + BDIM_Y*nchan_in);
        }

        FLOATV_T *shq      = base_fltv + tidy*nchan_in;        // [nchan_in]
        int64_t  *shoff    = base_i64  + tidy*shcol_len_max;   // [shcol_len_max]
        float    *shweight = base_flt  + tidy*shcol_len_max;   // [shcol_len_max]
#else
        FLOATV_T *shq = reinterpret_cast<FLOATV_T *>(shext) + threadIdx.y*nchan_in;
#endif
        const int h = ctaid / nlon_out;
        const int wo = ctaid - (h*nlon_out);
        const int ho = row_idx[h];

        kx += int64_t(batch)*nlat_in*nlon_in*nchan_in;
        vx += int64_t(batch)*nlat_in*nlon_in*nchan_out;

        qy += int64_t(batch)*nlat_out*nlon_out*nchan_in  + int64_t(ho)*nlon_out*nchan_in  + int64_t(wo)*nchan_in;
        y  += int64_t(batch)*nlat_out*nlon_out*nchan_out + int64_t(ho)*nlon_out*nchan_out + int64_t(wo)*nchan_out;

        #pragma unroll
        for (int i = 0; i < NLOC; i++) {
            locy[i] = __vset<FLOATV_T>(0.f);
        }

        strided_op<BDIM_X, CHIN_AS_OUT ? NLOC : 0>(nchan_in, [&](int i) { shq[i*BDIM_X+tidx] = qy[i*BDIM_X+tidx]; });

        float alpha_sum = 0.0f;
        float qdotk_max = -FLT_MAX;

        const int64_t rbeg = row_off[ho];
        const int64_t rend = row_off[ho + 1];
        const int     rlen = rend - rbeg;
        
        col_idx += rbeg;
#ifdef USE_SPLIT_ROW_FWD
        for (int i = tidx; i < rlen; i += BDIM_X) {

            const int64_t col = col_idx[i];

            const int hi = col / nlon_in;
            const int wi = col - (hi*nlon_in);
            const int wi_wo = wi + pscale*wo;
            const int wip = wi_wo - (wi_wo / nlon_in)*nlon_in;

            shoff[i] = int64_t(hi)*nlon_in + int64_t(wip);

            shweight[i] = quad_weights[hi];
        }
        __group_sync<BDIM_X>();

        for (int i = 0; i < rlen; i++) {

            const FLOATV_T *_kx = kx + shoff[i]*nchan_in;
            const FLOATV_T *_vx = vx + shoff[i]*nchan_out;
#else
        for (int off = 0; off < rlen; off++) {

            const int64_t col = col_idx[off];

            const int hi = col / nlon_in;
            const int wi = col - (hi*nlon_in);
            const int wi_wo = wi + pscale*wo;
            const int wip = wi_wo - (wi_wo / nlon_in)*nlon_in;

            const FLOATV_T *_kx = kx + int64_t(hi)*nlon_in*nchan_in + int64_t(wip)*nchan_in;
            const FLOATV_T *_vx = vx + int64_t(hi)*nlon_in*nchan_out + int64_t(wip)*nchan_out;
#endif
            FLOATV_T qdotkv = __vset<FLOATV_T>(0.f);

            strided_op<BDIM_X, CHIN_AS_OUT ? NLOC : 0>(nchan_in, [&](int i) { qdotkv = __vadd(qdotkv, __vmul(shq[i*BDIM_X+tidx], _kx[i*BDIM_X+tidx])); }); 

            float qdotk = __vred(qdotkv);
            __group_sum<BDIM_X, BDIM_Y>(qdotk);

            float qdotk_max_tmp;
            float alpha;
            float exp_save;

            qdotk_max_tmp = max(qdotk_max, qdotk);
#ifdef USE_SPLIT_ROW_FWD
            alpha = expf(qdotk - qdotk_max_tmp)*shweight[i];
#else
            alpha = expf(qdotk - qdotk_max_tmp)*quad_weights[hi];
#endif
            exp_save = expf(qdotk_max - qdotk_max_tmp);
            alpha_sum = alpha + alpha_sum*exp_save;

            strided_op<BDIM_X, NLOC>(nchan_out, [&](int i) { locy[i] = __vadd(__vscale(exp_save, locy[i]), __vscale(alpha, _vx[i*BDIM_X+tidx])); });

            qdotk_max = qdotk_max_tmp;
        }

        alpha_sum = 1.0f / alpha_sum;
        strided_op<BDIM_X, NLOC>(nchan_out, [&](int i) { y[i*BDIM_X+tidx] = __vscale(alpha_sum, locy[i]); });

        return;
    }

    template <int BDIM_X, int CUR_LOC_SIZE,
              int MAX_LOC_SIZE, // max size of FLOATV_T[] local array
              typename FLOATV_T>
    void launch_spc_attn_fwd(attn_params_t params, int nloc, // "BDIM_X*nloc" >= nchans_out
                             int batch_size, FLOATV_T *__restrict__ _kxp, FLOATV_T *__restrict__ _vxp,
                             FLOATV_T *__restrict__ _qyp, int32_t *_row_idx, int64_t *_row_off, int64_t *_col_idx,
                             float *_quad_weights, FLOATV_T *__restrict__ _yp, cudaStream_t stream)
    {

        if (CUR_LOC_SIZE == nloc) {

            const int nlat_out = params.nlat_out;
            const int nlon_out = params.nlon_out;
            const int nchan_in = params.nchan_in;

            int64_t n_long_rows = 0;
            int64_t max_row_len = 0;
            int64_t mid_row_len = 0;

#ifdef USE_SPLIT_ROW_FWD
            // splits the rows in "long" and "short" rows; long rows have
            // a length >= max(SPLIT_LONG_ROW_MIN_LEN(1024), len(row_0))
            // (since the rows are sorted in decreasing order, row_0 is the
            // longest row); short rows are the remaining rows.
            // If there are long rows, they are processed with a separate
            // kernels using multiple blocks per row, in order to mitigate
            // the imbalance causing long temporal tails.
            split_csr_rows(SPLIT_ROW_LENGTH_THRES, SPLIT_LONG_ROW_MIN_LEN, nlat_out, _row_idx, _row_off, &n_long_rows,
                           &max_row_len, &mid_row_len);

            //printf("n_long_rows: %ld, max_row_len: %ld, mid_row_len: %ld\n", n_long_rows, max_row_len, mid_row_len);

            if (n_long_rows > 0) {
                // processes the "long rows", from _row_idx[0] to _row_idx[n_long_rows-1]
                 spc_attn_fwd_long_rows<BDIM_X, CUR_LOC_SIZE>(params, n_long_rows, max_row_len, batch_size, _kxp,
                                                             _vxp, _qyp, _row_idx, _row_off, _col_idx,
                                                             _quad_weights, _yp, stream);
            }
#endif
            int64_t n_reg_rows = nlat_out - n_long_rows;
            if (!n_reg_rows) { return; }
            // process the "short rows", from _row_idx[n_long_rows] to _row_idx[nlat_out-1]

            // nloc determines the size of local arrays used to store
            // y vectors, of length nchan_out;
            // if nchan_in is >= BDIM_X*(nloc-1) and <= BDIM_X*nloc
            // then we can use the same compile-time known loops used
            // for output channels, with the execpetion of testing
            // whether to execute the last iteration based on "nchan_in"
            // rather than on "nchan_out"; in this way as long as the
            // difference between the number of input and output channels
            // is <= BDIM_X we can use the faster path
            const bool chin_as_out = (nchan_in >= BDIM_X*(CUR_LOC_SIZE - 1) &&
                                      nchan_in <= BDIM_X* CUR_LOC_SIZE     );

            constexpr int BDIM_Y = (BDIM_X <= WARP_SIZE) ? THREADS / BDIM_X : 1;

            dim3 block(BDIM_X, BDIM_Y);
            dim3 grid(DIV_UP(n_reg_rows*nlon_out, block.y), batch_size);
#ifdef USE_SPLIT_ROW_FWD
            size_t shsize = (sizeof(FLOATV_T)*nchan_in +
                             sizeof(int64_t)*mid_row_len + 
                             sizeof(float)*mid_row_len) * block.y;
#else
            size_t shsize = sizeof(FLOATV_T)*nchan_in*block.y; // block.y > 1 iif block.x==32
#endif
            if (chin_as_out) {
                auto kern = &s2_attn_fwd_special_k<BDIM_X, BDIM_Y, 1, CUR_LOC_SIZE, FLOATV_T>;
                ensure_dyn_shmem(reinterpret_cast<const void *>(kern), shsize);

                kern<<<grid, block, shsize, stream>>>(params, mid_row_len, n_reg_rows, _kxp, _vxp, _qyp, _row_idx + n_long_rows, _row_off, _col_idx, _quad_weights, _yp);
            } else {
                auto kern = &s2_attn_fwd_special_k<BDIM_X, BDIM_Y, 0, CUR_LOC_SIZE, FLOATV_T>;
                ensure_dyn_shmem(reinterpret_cast<const void *>(kern), shsize);
                
                kern<<<grid, block, shsize, stream>>>(params, mid_row_len, n_reg_rows, _kxp, _vxp, _qyp, _row_idx + n_long_rows, _row_off, _col_idx, _quad_weights, _yp);
            }
            CHECK_ERROR("s2_attn_fwd_special_k");

            return;
        }
        if constexpr (CUR_LOC_SIZE < MAX_LOC_SIZE) {
            launch_spc_attn_fwd<BDIM_X, CUR_LOC_SIZE + 1, MAX_LOC_SIZE>(
                params, nloc, batch_size, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp, stream);
        }
        return;
    }
    
    // **************** end specialized kernel ****************

    static void s2_attn_fwd_dispatch(int64_t batch_size, int64_t nchans_in, int64_t nchans_out, int64_t nlon_in,
                                     int64_t nlat_out, int64_t nlon_out, at::Tensor kxP, at::Tensor vxP, at::Tensor qyP,
                                     at::Tensor row_off, at::Tensor col_idx, at::Tensor quad_weights, at::Tensor yP)
    {

        static_assert(0 == (MAX_LOCAL_ARR_LEN & (MAX_LOCAL_ARR_LEN - 1)));

        // get stream
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        // sort row indices (ho-s) in descending order
        // based on (row_off[ho+1]-row_off[ho])
        at::Tensor row_idx = sortRows(nlat_out, row_off, stream);
#if 0
        dump_csr_linear("csr_attn",  nlat_out, row_idx, row_off, col_idx);
#endif
        const int nlat_in = kxP.size(1);

        // smallest power of two "bdimx" (>=32) s.t. bdimx*MAX_LOCAL_ARR_LEN >= nchans_out
        int bdimx;
        bdimx = DIV_UP(nchans_out, MAX_LOCAL_ARR_LEN);
        bdimx = max(bdimx, WARP_SIZE);
        bdimx = next_pow2(bdimx);

        float *_kxp = reinterpret_cast<float *>(kxP.data_ptr());
        float *_vxp = reinterpret_cast<float *>(vxP.data_ptr());
        float *_qyp = reinterpret_cast<float *>(qyP.data_ptr());
        float *_yp = reinterpret_cast<float *>(yP.data_ptr());

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
            || !is_aligned<sizeof(float4)>(_yp) || (nchans_in % VEC_SIZE) != 0 || (nchans_out % VEC_SIZE) != 0) {

            const int nloc = DIV_UP(nchans_out, bdimx);

            // to avoid the compilation of unused template instances;
            // we use a block size BDIM_X that is the smallest power of 2
            // such that BDIM_X*MAX_LOCAL_ARR_LEN >= nchans_out, so
            // BDIM_X > 32 are used only for:
            //
            //  (BDIM_X-1)*MAX_LOCAL_ARR_LEN < nchans_out <= BDIM_X*MAX_LOCAL_ARR_LEN
            constexpr int MIN_LOC_ARR_LEN = MAX_LOCAL_ARR_LEN / 2 + 1;

            // use 2D blocks only if 32 threads are enough
            switch (bdimx) {
                case   32: launch_spc_attn_fwd<  32,               1, MAX_LOCAL_ARR_LEN>(params, nloc, batch_size, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp, stream); break;
                case   64: launch_spc_attn_fwd<  64, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(params, nloc, batch_size, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp, stream); break;
                case  128: launch_spc_attn_fwd< 128, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(params, nloc, batch_size, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp, stream); break;
                case  256: launch_spc_attn_fwd< 256, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(params, nloc, batch_size, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp, stream); break;
                case  512: launch_spc_attn_fwd< 512, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(params, nloc, batch_size, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp, stream); break;
                case 1024: launch_spc_attn_fwd<1024, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(params, nloc, batch_size, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp, stream); break;
                default:   launch_gen_attn_fwd                                          (params,       batch_size, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp, stream); break;
            }
        } else {

            float4 *_kxp4 = reinterpret_cast<float4 *>(_kxp);
            float4 *_vxp4 = reinterpret_cast<float4 *>(_vxp);
            float4 *_qyp4 = reinterpret_cast<float4 *>(_qyp);
            float4 *_yp4 = reinterpret_cast<float4 *>(_yp);

            nchans_in /= VEC_SIZE;
            nchans_out /= VEC_SIZE;
        
            params.nchan_in = nchans_in;
            params.nchan_out = nchans_out;

            const int nloc = DIV_UP(nchans_out, bdimx);

            constexpr int MAX_LOCAL_VEC_LEN = MAX_LOCAL_ARR_LEN / VEC_SIZE;
            constexpr int MIN_LOC_VEC_LEN = MAX_LOCAL_VEC_LEN / 2 + 1;

            // use 2D blocks only if 32 threads are enough
            switch (bdimx) {
                case   32: launch_spc_attn_fwd<  32,               1, MAX_LOCAL_VEC_LEN>(params, nloc, batch_size, _kxp4, _vxp4, _qyp4, _row_idx, _row_off, _col_idx, _quad_weights, _yp4, stream); break;
                case   64: launch_spc_attn_fwd<  64, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(params, nloc, batch_size, _kxp4, _vxp4, _qyp4, _row_idx, _row_off, _col_idx, _quad_weights, _yp4, stream); break;
                case  128: launch_spc_attn_fwd< 128, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(params, nloc, batch_size, _kxp4, _vxp4, _qyp4, _row_idx, _row_off, _col_idx, _quad_weights, _yp4, stream); break;
                case  256: launch_spc_attn_fwd< 256, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(params, nloc, batch_size, _kxp4, _vxp4, _qyp4, _row_idx, _row_off, _col_idx, _quad_weights, _yp4, stream); break;
                case  512: launch_spc_attn_fwd< 512, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(params, nloc, batch_size, _kxp4, _vxp4, _qyp4, _row_idx, _row_off, _col_idx, _quad_weights, _yp4, stream); break;
                case 1024: launch_spc_attn_fwd<1024, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(params, nloc, batch_size, _kxp4, _vxp4, _qyp4, _row_idx, _row_off, _col_idx, _quad_weights, _yp4, stream); break;
                default:   launch_gen_attn_fwd                                          (params,       batch_size, _kxp4, _vxp4, _qyp4, _row_idx, _row_off, _col_idx, _quad_weights, _yp4, stream); break;
            }
        }

        return;
    }

    // END - forward kernels and functions

    torch::Tensor s2_attention_fwd_cuda(at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor quad_weights,
                                        at::Tensor psi_col_idx, at::Tensor psi_row_off, int64_t nlon_in,
                                        int64_t nlat_out, int64_t nlon_out)
    {
        CHECK_CUDA_INPUT_TENSOR(kx);
        CHECK_CUDA_INPUT_TENSOR(vx);
        CHECK_CUDA_INPUT_TENSOR(qy);
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

        size_t nchans_in = qy.size(1); // or kx.size(1)
        size_t nchans_out = vx.size(1);

        const int batch_size = kx.size(0);
        const int64_t nlat_in = kx.size(2);

        // extract dtype
        auto qy_type = qy.dtype();

        torch::Tensor kxP = kx.to(torch::kFloat32);
        torch::Tensor vxP = vx.to(torch::kFloat32);
        torch::Tensor qyP = qy.to(torch::kFloat32);

        // these are much safer than checking is_contiguous(at::MemoryFormat::ChannelsLast)
        // the former fails for num_channels == 1
        bool kx_is_channels_last = kxP.strides()[1] == 1;
        bool vx_is_channels_last = vxP.strides()[1] == 1;
        bool qy_is_channels_last = qyP.strides()[1] == 1;

        if (!kx_is_channels_last) { kxP = permute_4D_to0231(kxP); }
        if (!vx_is_channels_last) { vxP = permute_4D_to0231(vxP); }
        if (!qy_is_channels_last) { qyP = permute_4D_to0231(qyP); }

        int64_t out_dims[] = {batch_size, nlat_out, nlon_out, nchans_out};
        torch::Tensor yP = torch::empty(out_dims, kxP.options());

        if (downsample) {
            s2_attn_fwd_dispatch(batch_size, nchans_in, nchans_out, nlon_in, nlat_out, nlon_out, kxP, vxP, qyP,
                                 psi_row_off, psi_col_idx, quad_weights, yP);
        } else {
            s2_attn_fwd_upsample_dispatch(batch_size, nchans_in, nchans_out, nlon_in, nlat_in, nlat_out, nlon_out, kxP,
                                          vxP, qyP, psi_row_off, psi_col_idx, quad_weights, yP);
        }

        torch::Tensor y = yP;
        if (!qy_is_channels_last) { y = permute_4D_to0312(y); }

        // convert precision back to starting
        y = y.to(qy_type);

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return y;
    }

    TORCH_LIBRARY_IMPL(attention_kernels, CUDA, m) { m.impl("forward", &s2_attention_fwd_cuda); }

void dump_csr_linear(const char *fname,
                     int64_t nrows,
                     at::Tensor row_idx,
                     at::Tensor row_off,
                     at::Tensor col_idx) {

        int64_t nnz = col_idx.size(0);

        int32_t *row_idx_h = new int32_t[nrows];
        int64_t *row_off_h = new int64_t[nrows+1];
        int64_t *col_idx_h = new int64_t[nnz];

        int32_t *row_idx_d = row_idx.data_ptr<int32_t>();
        int64_t *row_off_d = row_off.data_ptr<int64_t>();
        int64_t *col_idx_d = col_idx.data_ptr<int64_t>();

        CHECK_CUDA(cudaMemcpy(row_idx_h, row_idx_d, sizeof(*row_idx_h)*nrows    , cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(row_off_h, row_off_d, sizeof(*row_off_h)*(nrows+1), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(col_idx_h, col_idx_d, sizeof(*col_idx_h)*nnz      , cudaMemcpyDeviceToHost));

        printf("Writing data to file...");

        static int count = 0;

        char file_name[256];
        snprintf(file_name, sizeof(file_name), "%s_%d.txt", fname, count);
        count++;

        FILE *fp = fopen(file_name, "w");
        if (!fp) {
                fprintf(stderr, "Cannot open file %s for writing!\n", fname);
                exit(EXIT_FAILURE);
        }
        
        fprintf(fp, "nrows: %ld, row_idx.size(0): %ld, row_off.size(0): %ld, col_idx.size(0): %ld\n",
                nrows, row_idx.size(0), row_off.size(0), col_idx.size(0));

        fprintf(fp, "CSR:\n");

        for(int64_t i = 0; i < nrows; i++) {

                int32_t r = row_idx_h[i];
                
                fprintf(fp, "%6ld, row: %6d, len: %6ld - ", i, r, row_off_h[r+1]-row_off_h[r]);

                for(int64_t o = row_off_h[r]; o < row_off_h[r+1]; o++) {
                        fprintf(fp, "%10ld", col_idx_h[o]);
                }
                fprintf(fp, "\n");
        }
        fclose(fp);
        printf("done\n");

        delete [] row_idx_h;
        delete [] row_off_h;
        delete [] col_idx_h;
}

} // namespace attention_kernels
