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

// BEGIN - forward kernels and functions

namespace attention_kernels
{

    // scatter-direction launcher, defined in attention_cuda_fwd_upsample.cu;
    // called by s2_attention_fwd_cuda when nlon_out % nlon_in == 0.
    void s2_attn_fwd_upsample_dispatch(int batch_size, size_t nchans_in, size_t nchans_out, int64_t nlon_in,
                                       int64_t nlat_in, int64_t nlat_out, int64_t nlon_out, torch::Tensor kxP,
                                       torch::Tensor vxP, torch::Tensor qyP, torch::Tensor psi_row_off,
                                       torch::Tensor psi_col_idx, torch::Tensor quad_weights, torch::Tensor yP);

    // called with (blockDim.x=32 and blockDim.y>1, BDIM_X=blockDim.x*blockDim.y)
    //
    // STORAGE_T is the global-memory element type (float4 for the fp32 vectorized
    // path; float / c10::Half / c10::BFloat16 for the scalar path). COMPUTE_T is
    // the arithmetic type (float4 for the vectorized path, float otherwise) — all
    // dot products, softmax and accumulation happen in COMPUTE_T. vload/vstore
    // widen/narrow at the memory boundary.
    template <int BDIM_X, typename STORAGE_T>
    __global__ __launch_bounds__(BDIM_X) void s2_attn_fwd_generic_vec_k(
        int nchan_in,  // no. of STORAGE_T elements along channel dim
        int nchan_out, // no. of STORAGE_T elements along channel dim
        int nlat_in, int nlon_in, int nlat_out, int nlon_out, const STORAGE_T *__restrict__ kx,
        const STORAGE_T *__restrict__ vx, const STORAGE_T *__restrict__ qy, const int32_t *__restrict__ row_idx,
        const int64_t *__restrict__ row_off, const int64_t *__restrict__ col_idx,
        const float *__restrict__ quad_weights, STORAGE_T *__restrict__ y)
    {
        using COMPUTE_T = typename vec_traits<STORAGE_T>::compute_t;

        extern __shared__ __align__(sizeof(float4)) float shext[];
        COMPUTE_T *shy = reinterpret_cast<COMPUTE_T *>(shext) + threadIdx.y * nchan_out;

        const int batch = blockIdx.y;
        const int wid = blockIdx.x * blockDim.y + threadIdx.y;

        if (wid >= nlat_out * nlon_out) { return; }

        const int tidx = threadIdx.x;

        const int h = wid / nlon_out;
        const int wo = wid - (h * nlon_out);
        const int ho = row_idx[h];

        // one output lon step corresponds to pscale input lon steps (requires nlon_in % nlon_out == 0)
        const int pscale = nlon_in / nlon_out;

        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { shy[chan] = __vset<COMPUTE_T>(0.f); }

        kx += int64_t(batch) * nlat_in * nlon_in * nchan_in;
        qy += int64_t(batch) * nlat_out * nlon_out * nchan_in + int64_t(ho) * nchan_in * nlon_out
            + int64_t(wo) * nchan_in;

        vx += int64_t(batch) * nlat_in * nlon_in * nchan_out;
        y += int64_t(batch) * nlat_out * nlon_out * nchan_out + int64_t(ho) * nchan_out * nlon_out
            + int64_t(wo) * nchan_out;

        float alpha_sum = 0.0f;
        float qdotk_max = -FLT_MAX;

        const int64_t rbeg = row_off[ho];
        const int64_t rend = row_off[ho + 1];

        col_idx += rbeg;

        const int rlen = rend - rbeg;

        for (int off = 0; off < rlen; off++) {

            const int64_t col = col_idx[off];

            const int hi = col / nlon_in;
            const int wi = col - (hi * nlon_in);
            const int wi_wo = wi + pscale * wo;
            const int wip = wi_wo - (wi_wo / nlon_in) * nlon_in;

            const STORAGE_T *_kx = kx + int64_t(hi) * nlon_in * nchan_in + int64_t(wip) * nchan_in;
            const STORAGE_T *_vx = vx + int64_t(hi) * nlon_in * nchan_out + int64_t(wip) * nchan_out;

            COMPUTE_T qdotkv = __vset<COMPUTE_T>(0.f);

            for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
                qdotkv = __vadd(qdotkv, __vmul(vload(qy, chan), vload(_kx, chan)));
            }

            float qdotk = __warp_sum(__vred(qdotkv));

            float qdotk_max_tmp;
            float alpha;
            float exp_save;

            qdotk_max_tmp = max(qdotk_max, qdotk);
            alpha = expf(qdotk - qdotk_max_tmp) * quad_weights[hi];
            exp_save = expf(qdotk_max - qdotk_max_tmp);

            alpha_sum = alpha + alpha_sum * exp_save;

            for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
                shy[chan] = __vadd(__vscale(exp_save, shy[chan]), __vscale(alpha, vload(_vx, chan)));
            }
            qdotk_max = qdotk_max_tmp;
        }

        alpha_sum = 1.0f / alpha_sum;
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { vstore(y, chan, __vscale(alpha_sum, shy[chan])); }

        return;
    }

    // called with either (BDIM_X=32 and BDIM_Y>1) || (2^K=BDIM_X > 32 and BDIM_Y=1)
    template <int BDIM_X, int BDIM_Y,
              int CHIN_AS_OUT, // 1 iif "BDIM_X*(NLOC-1) <= nchan_in <= BDIM_X*NLOC" else 0
              int NLOC,        // smallest int such that BDIM_X*NLOC >= nchan_out
              typename STORAGE_T>
    __global__ __launch_bounds__(BDIM_X *BDIM_Y) void s2_attn_fwd_special_vec_k(
        int nchan_in,  // no. of STORAGE_T elements along channel dim
        int nchan_out, // no. of STORAGE_T elements along channel dim
        int nlat_in, int nlon_in, int nlat_out, int nlon_out, const STORAGE_T *__restrict__ kx,
        const STORAGE_T *__restrict__ vx, const STORAGE_T *__restrict__ qy, const int32_t *__restrict__ row_idx,
        const int64_t *__restrict__ row_off, const int64_t *__restrict__ col_idx,
        const float *__restrict__ quad_weights, STORAGE_T *__restrict__ y)
    {
        using COMPUTE_T = typename vec_traits<STORAGE_T>::compute_t;

        static_assert(0 == (BDIM_X & (BDIM_X - 1)));
        static_assert(0 == (BDIM_Y & (BDIM_Y - 1)));
        static_assert((BDIM_X == 32 && BDIM_Y > 1) || (BDIM_X > 32 && BDIM_Y == 1));

        constexpr int NLOC_M1 = NLOC - 1;

        const int tidx = threadIdx.x;
        const int batch = blockIdx.y;
        const int ctaid = blockIdx.x * blockDim.y + threadIdx.y;

        if (ctaid >= nlat_out * nlon_out) { return; }

        COMPUTE_T locy[NLOC];

        // shq holds q already widened to COMPUTE_T (converted once on load, reused
        // across the neighbor loop).
        extern __shared__ __align__(sizeof(float4)) float shext[];
        COMPUTE_T *shq = reinterpret_cast<COMPUTE_T *>(shext) + threadIdx.y * nchan_in;

        if constexpr (CHIN_AS_OUT) { shq += tidx; }

        const int h = ctaid / nlon_out;
        const int wo = ctaid - (h * nlon_out);
        const int ho = row_idx[h];

        // one output lon step corresponds to pscale input lon steps (requires nlon_in % nlon_out == 0)
        const int pscale = nlon_in / nlon_out;

        kx += int64_t(batch) * nlat_in * nlon_in * nchan_in;
        qy += int64_t(batch) * nlat_out * nlon_out * nchan_in + int64_t(ho) * nlon_out * nchan_in
            + int64_t(wo) * nchan_in;
        if constexpr (CHIN_AS_OUT) {
            kx += tidx;
            qy += tidx;
        }

        vx += int64_t(batch) * nlat_in * nlon_in * nchan_out + tidx;
        y += int64_t(batch) * nlat_out * nlon_out * nchan_out + int64_t(ho) * nlon_out * nchan_out
            + int64_t(wo) * nchan_out + tidx;

#pragma unroll
        for (int i = 0; i < NLOC; i++) { locy[i] = __vset<COMPUTE_T>(0.f); }

        if constexpr (CHIN_AS_OUT) {
#pragma unroll
            for (int i = 0; i < NLOC_M1; i++) { shq[i * BDIM_X] = vload(qy, i * BDIM_X); }
            if (NLOC_M1 * BDIM_X + tidx < nchan_in) { shq[NLOC_M1 * BDIM_X] = vload(qy, NLOC_M1 * BDIM_X); }
        } else {
            for (int chan = tidx; chan < nchan_in; chan += BDIM_X) { shq[chan] = vload(qy, chan); }
        }

        float alpha_sum = 0.0f;
        float qdotk_max = -FLT_MAX;

        const int64_t rbeg = row_off[ho];
        const int64_t rend = row_off[ho + 1];

        col_idx += rbeg;

        const int rlen = rend - rbeg;

        for (int off = 0; off < rlen; off++) {

            const int64_t col = col_idx[off];

            const int hi = col / nlon_in;
            const int wi = col - (hi * nlon_in);
            const int wi_wo = wi + pscale * wo;
            const int wip = wi_wo - (wi_wo / nlon_in) * nlon_in;

            const STORAGE_T *_kx = kx + int64_t(hi) * nlon_in * nchan_in + int64_t(wip) * nchan_in;
            const STORAGE_T *_vx = vx + int64_t(hi) * nlon_in * nchan_out + int64_t(wip) * nchan_out;

            COMPUTE_T qdotkv = __vset<COMPUTE_T>(0.f);

            if constexpr (CHIN_AS_OUT) {
#pragma unroll
                for (int i = 0; i < NLOC_M1; i++) {
                    qdotkv = __vadd(qdotkv, __vmul(shq[i * BDIM_X], vload(_kx, i * BDIM_X)));
                }
                if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                    qdotkv = __vadd(qdotkv, __vmul(shq[NLOC_M1 * BDIM_X], vload(_kx, NLOC_M1 * BDIM_X)));
                }
            } else {
                for (int chan = tidx; chan < nchan_in; chan += BDIM_X) {
                    qdotkv = __vadd(qdotkv, __vmul(shq[chan], vload(_kx, chan)));
                }
            }

            float qdotk = __vred(qdotkv);
            if constexpr (BDIM_X == 32) {
                qdotk = __warp_sum(qdotk);
            } else {
                qdotk = __block_sum<BDIM_X>(qdotk);
            }

            float qdotk_max_tmp;
            float alpha;
            float exp_save;

            qdotk_max_tmp = max(qdotk_max, qdotk);
            alpha = expf(qdotk - qdotk_max_tmp) * quad_weights[hi];
            exp_save = expf(qdotk_max - qdotk_max_tmp);

            alpha_sum = alpha + alpha_sum * exp_save;

#pragma unroll
            for (int i = 0; i < NLOC_M1; i++) {
                locy[i] = __vadd(__vscale(exp_save, locy[i]), __vscale(alpha, vload(_vx, i * BDIM_X)));
            }
            if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
                locy[NLOC_M1] = __vadd(__vscale(exp_save, locy[NLOC_M1]), __vscale(alpha, vload(_vx, NLOC_M1 * BDIM_X)));
            }

            qdotk_max = qdotk_max_tmp;
        }

        alpha_sum = 1.0f / alpha_sum;

#pragma unroll
        for (int i = 0; i < NLOC_M1; i++) { vstore(y, i * BDIM_X, __vscale(alpha_sum, locy[i])); }
        if (NLOC_M1 * BDIM_X + tidx < nchan_out) { vstore(y, NLOC_M1 * BDIM_X, __vscale(alpha_sum, locy[NLOC_M1])); }

        return;
    }

    template <typename STORAGE_T>
    void launch_gen_attn_fwd(int batch_size, int nchans_in, int nchans_out, int nlat_in, int nlon_in, int nlat_out,
                             int nlon_out, STORAGE_T *__restrict__ _kxp, STORAGE_T *__restrict__ _vxp,
                             STORAGE_T *__restrict__ _qyp, int32_t *_row_idx, int64_t *_row_off, int64_t *_col_idx,
                             float *_quad_weights, STORAGE_T *__restrict__ _yp, cudaStream_t stream)
    {

        dim3 block(WARP_SIZE, THREADS / WARP_SIZE);
        dim3 grid(DIV_UP(nlat_out * nlon_out, block.y), batch_size);

        // shared memory holds compute-type (COMPUTE_T) data, not STORAGE_T.
        size_t shsize = sizeof(typename vec_traits<STORAGE_T>::compute_t) * nchans_out * block.y;

        s2_attn_fwd_generic_vec_k<THREADS>
            <<<grid, block, shsize, stream>>>(nchans_in, nchans_out, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp,
                                              _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp);
        CHECK_ERROR("s2_attn_fwd_generic_vec_k");

        return;
    }

    template <int BDIM_X, int BDIM_Y, int CUR_LOC_SIZE,
              int MAX_LOC_SIZE, // max size of COMPUTE_T[] local array
              typename STORAGE_T>
    void launch_spc_attn_fwd(int nloc, // "BDIM_X*nloc" >= nchans_out
                             int batch_size, int nchans_in, int nchans_out, int nlat_in, int nlon_in, int nlat_out,
                             int nlon_out, STORAGE_T *__restrict__ _kxp, STORAGE_T *__restrict__ _vxp,
                             STORAGE_T *__restrict__ _qyp, int32_t *_row_idx, int64_t *_row_off, int64_t *_col_idx,
                             float *_quad_weights, STORAGE_T *__restrict__ _yp, cudaStream_t stream)
    {

        if (CUR_LOC_SIZE == nloc) {

            dim3 block(BDIM_X, BDIM_Y);
            dim3 grid(DIV_UP(nlat_out * nlon_out, block.y), batch_size);

            // shared memory holds compute-type (COMPUTE_T) data, not STORAGE_T.
            // block.y > 1 iif block.x==32
            size_t shsize = sizeof(typename vec_traits<STORAGE_T>::compute_t) * nchans_in * block.y;

            // nloc determines the size of local arrays used to store
            // y vectors, of length nchans_out;
            // if nchans_in is >= BDIM_X*(nloc-1) and <= BDIM_X*nloc
            // then we can use the same compile-time known loops used
            // for output channels, with the execpetion of testing
            // whether to execute the last iteration based on "nchans_in"
            // rather than on "nchans_out"; in this way as long as the
            // difference between the number of input and output channels
            // is <= BDIM_X we can use the faster path
            if (nchans_in >= BDIM_X * (CUR_LOC_SIZE - 1) && nchans_in <= BDIM_X * CUR_LOC_SIZE) {

                s2_attn_fwd_special_vec_k<BDIM_X, BDIM_Y, 1, CUR_LOC_SIZE>
                    <<<grid, block, shsize, stream>>>(nchans_in, nchans_out, nlat_in, nlon_in, nlat_out, nlon_out, _kxp,
                                                      _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp);
            } else {

                s2_attn_fwd_special_vec_k<BDIM_X, BDIM_Y, 0, CUR_LOC_SIZE>
                    <<<grid, block, shsize, stream>>>(nchans_in, nchans_out, nlat_in, nlon_in, nlat_out, nlon_out, _kxp,
                                                      _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp);
            }
            CHECK_ERROR("s2_attn_fwd_special_vec_k");

            return;
        }
        if constexpr (CUR_LOC_SIZE < MAX_LOC_SIZE) {
            launch_spc_attn_fwd<BDIM_X, BDIM_Y, CUR_LOC_SIZE + 1, MAX_LOC_SIZE>(
                nloc, batch_size, nchans_in, nchans_out, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp,
                _row_idx, _row_off, _col_idx, _quad_weights, _yp, stream);
        }
        return;
    }

    // Picks the block size (BDIM_X) instance and launches the gather kernel for a
    // given storage vector type SV. MAX_LOC / MIN_LOC bound the per-thread local
    // array length (in COMPUTE_T units). nci / nco are channel counts in SV units.
    template <int MAX_LOC, int MIN_LOC, typename SV>
    static void fwd_dispatch_bdimx(int bdimx, int nloc, int64_t batch_size, int64_t nci, int64_t nco, int nlat_in,
                                   int64_t nlon_in, int64_t nlat_out, int64_t nlon_out, SV *_kxp, SV *_vxp, SV *_qyp,
                                   int32_t *_row_idx, int64_t *_row_off, int64_t *_col_idx, float *_quad_weights,
                                   SV *_yp, cudaStream_t stream)
    {
        // use 2D blocks only if 32 threads are enough
        switch (bdimx) {
        case 32:
            launch_spc_attn_fwd<32, 2, 1, MAX_LOC>(nloc, batch_size, nci, nco, nlat_in, nlon_in, nlat_out, nlon_out, _kxp,
                                                   _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp, stream);
            break;
        case 64:
            launch_spc_attn_fwd<64, 1, MIN_LOC, MAX_LOC>(nloc, batch_size, nci, nco, nlat_in, nlon_in, nlat_out,
                                                         nlon_out, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx,
                                                         _quad_weights, _yp, stream);
            break;
        case 128:
            launch_spc_attn_fwd<128, 1, MIN_LOC, MAX_LOC>(nloc, batch_size, nci, nco, nlat_in, nlon_in, nlat_out,
                                                          nlon_out, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx,
                                                          _quad_weights, _yp, stream);
            break;
        case 256:
            launch_spc_attn_fwd<256, 1, MIN_LOC, MAX_LOC>(nloc, batch_size, nci, nco, nlat_in, nlon_in, nlat_out,
                                                          nlon_out, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx,
                                                          _quad_weights, _yp, stream);
            break;
        case 512:
            launch_spc_attn_fwd<512, 1, MIN_LOC, MAX_LOC>(nloc, batch_size, nci, nco, nlat_in, nlon_in, nlat_out,
                                                          nlon_out, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx,
                                                          _quad_weights, _yp, stream);
            break;
        case 1024:
            launch_spc_attn_fwd<1024, 1, MIN_LOC, MAX_LOC>(nloc, batch_size, nci, nco, nlat_in, nlon_in, nlat_out,
                                                           nlon_out, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx,
                                                           _quad_weights, _yp, stream);
            break;
        default:
            launch_gen_attn_fwd(batch_size, nci, nco, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, _row_idx,
                                _row_off, _col_idx, _quad_weights, _yp, stream);
            break;
        }
    }

    // Templated on the storage element type (float / c10::Half / c10::BFloat16).
    // Path selection (compute / accumulation are fp32 in every case):
    //   - fp32, 16B-aligned, nchans % 4 == 0  -> float4 vectorized (LDG.128)
    //   - fp16/bf16, 16B-aligned, nchans % 8 == 0 -> half8/bf168 vectorized (LDG.128)
    //   - otherwise -> scalar STORAGE_T path
    template <typename scalar_t>
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

        const int nlat_in = kxP.size(1);

        // smallest power of two "bdimx" (>=32) s.t. bdimx*MAX_LOCAL_ARR_LEN >= nchans_out
        int bdimx;
        bdimx = DIV_UP(nchans_out, MAX_LOCAL_ARR_LEN);
        bdimx = max(bdimx, WARP_SIZE);
        bdimx = next_pow2(bdimx);

        scalar_t *_kxp = reinterpret_cast<scalar_t *>(kxP.data_ptr());
        scalar_t *_vxp = reinterpret_cast<scalar_t *>(vxP.data_ptr());
        scalar_t *_qyp = reinterpret_cast<scalar_t *>(qyP.data_ptr());
        scalar_t *_yp = reinterpret_cast<scalar_t *>(yP.data_ptr());

        int32_t *_row_idx = reinterpret_cast<int32_t *>(row_idx.data_ptr());
        int64_t *_row_off = reinterpret_cast<int64_t *>(row_off.data_ptr());
        int64_t *_col_idx = reinterpret_cast<int64_t *>(col_idx.data_ptr());
        float *_quad_weights = reinterpret_cast<float *>(quad_weights.data_ptr());

        constexpr int MIN_LOC_ARR_LEN = MAX_LOCAL_ARR_LEN / 2 + 1;

        if constexpr (std::is_same<scalar_t, float>::value) {
            // fp32: float4 vectorized when 16B-aligned + 4-divisible, else scalar.
            constexpr int VEC_SIZE = sizeof(float4) / sizeof(float); // 4
            const bool use_vec = is_aligned<16>(_kxp) && is_aligned<16>(_vxp) && is_aligned<16>(_qyp)
                && is_aligned<16>(_yp) && (nchans_in % VEC_SIZE) == 0 && (nchans_out % VEC_SIZE) == 0;

            if (use_vec) {
                constexpr int MAX_VEC = MAX_LOCAL_ARR_LEN / VEC_SIZE;
                constexpr int MIN_VEC = MAX_VEC / 2 + 1;
                const int64_t nci = nchans_in / VEC_SIZE;
                const int64_t nco = nchans_out / VEC_SIZE;
                fwd_dispatch_bdimx<MAX_VEC, MIN_VEC, float4>(
                    bdimx, DIV_UP(nco, bdimx), batch_size, nci, nco, nlat_in, nlon_in, nlat_out, nlon_out,
                    reinterpret_cast<float4 *>(_kxp), reinterpret_cast<float4 *>(_vxp), reinterpret_cast<float4 *>(_qyp),
                    _row_idx, _row_off, _col_idx, _quad_weights, reinterpret_cast<float4 *>(_yp), stream);
            } else {
                fwd_dispatch_bdimx<MAX_LOCAL_ARR_LEN, MIN_LOC_ARR_LEN, float>(
                    bdimx, DIV_UP(nchans_out, bdimx), batch_size, nchans_in, nchans_out, nlat_in, nlon_in, nlat_out,
                    nlon_out, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp, stream);
            }
        } else {
            // fp16/bf16: scalar STORAGE_T path (widen at load, narrow at store; fp32
            // compute/accumulation). A vectorized 8-wide path was tried and reverted:
            // it raised register pressure and lowered occupancy, and ncu shows this
            // kernel is latency/occupancy-bound (DRAM ~25%), not bandwidth-bound, so
            // vectorizing reduced precision only hurt. See the AMP refactor notes.
            fwd_dispatch_bdimx<MAX_LOCAL_ARR_LEN, MIN_LOC_ARR_LEN, scalar_t>(
                bdimx, DIV_UP(nchans_out, bdimx), batch_size, nchans_in, nchans_out, nlat_in, nlon_in, nlat_out,
                nlon_out, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp, stream);
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

        const int64_t out_dims[] = {batch_size, nlat_out, nlon_out, nchans_out};
        torch::Tensor y;

        // ATen dispatch over the input dtype.
        //
        // Gather (downsample / self) path: native storage. kx/vx/qy keep their
        // dtype; the kernel widens to fp32 at load and narrows back at store
        // (Tier B), so there is no whole-tensor fp32 copy and the read bandwidth
        // for fp16/bf16 is halved. Compute/accumulation are fp32 in-kernel.
        //
        // Scatter (upsample) path: the scatter kernels are not yet on the native
        // storage path, so we still upcast to fp32 here and downcast the result
        // at the end (the trailing y.to(qy_type)). fp32 inputs are unaffected
        // either way (the .to(kFloat32) is a no-op).
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, qy.scalar_type(), "s2_attention_fwd_cuda", [&] {
            using storage_t = scalar_t;

            if (downsample) {
                // native-storage gather path
                torch::Tensor kxP = kx;
                torch::Tensor vxP = vx;
                torch::Tensor qyP = qy;

                // safer than is_contiguous(ChannelsLast), which fails for num_channels == 1
                bool kx_is_channels_last = kxP.strides()[1] == 1;
                bool vx_is_channels_last = vxP.strides()[1] == 1;
                bool qy_is_channels_last = qyP.strides()[1] == 1;

                if (!kx_is_channels_last) { kxP = permute_4D_to0231(kxP); }
                if (!vx_is_channels_last) { vxP = permute_4D_to0231(vxP); }
                if (!qy_is_channels_last) { qyP = permute_4D_to0231(qyP); }

                torch::Tensor yP = torch::empty(out_dims, kxP.options()); // native dtype

                s2_attn_fwd_dispatch<storage_t>(batch_size, nchans_in, nchans_out, nlon_in, nlat_out, nlon_out, kxP,
                                                vxP, qyP, psi_row_off, psi_col_idx, quad_weights, yP);

                y = yP;
                if (!qy_is_channels_last) { y = permute_4D_to0312(y); }
            } else {
                // upsample (scatter) path: native storage. s2_attn_fwd_upsample_dispatch
                // does its own AT_DISPATCH and widens fp16/bf16 at load (fp32 compute),
                // narrowing the output at store — same as the gather path.
                torch::Tensor kxP = kx;
                torch::Tensor vxP = vx;
                torch::Tensor qyP = qy;

                bool kx_is_channels_last = kxP.strides()[1] == 1;
                bool vx_is_channels_last = vxP.strides()[1] == 1;
                bool qy_is_channels_last = qyP.strides()[1] == 1;

                if (!kx_is_channels_last) { kxP = permute_4D_to0231(kxP); }
                if (!vx_is_channels_last) { vxP = permute_4D_to0231(vxP); }
                if (!qy_is_channels_last) { qyP = permute_4D_to0231(qyP); }

                torch::Tensor yP = torch::empty(out_dims, kxP.options()); // native dtype

                s2_attn_fwd_upsample_dispatch(batch_size, nchans_in, nchans_out, nlon_in, nlat_in, nlat_out, nlon_out,
                                              kxP, vxP, qyP, psi_row_off, psi_col_idx, quad_weights, yP);

                y = yP;
                if (!qy_is_channels_last) { y = permute_4D_to0312(y); }
            }
        });

        // convert precision back to starting dtype. No-op now that both the gather
        // and upsample paths produce native-dtype output; kept as a safety net.
        y = y.to(qy_type);

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return y;
    }

    TORCH_LIBRARY_IMPL(attention_kernels, CUDA, m) { m.impl("forward", &s2_attention_fwd_cuda); }

} // namespace attention_kernels
