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
#include <c10/cuda/CUDAException.h>

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
    void s2_attn_fwd_upsample_dispatch(int batch_size, int64_t num_heads, size_t nchans_in, size_t nchans_out,
                                       int64_t nlon_in, int64_t nlat_in, int64_t nlat_out, int64_t nlon_out,
                                       torch::Tensor kxP, torch::Tensor vxP, torch::Tensor qyP, torch::Tensor psi_row_off,
                                       torch::Tensor psi_col_idx, torch::Tensor quad_weights, torch::Tensor yP);

    // called with (blockDim.x=32 and blockDim.y>1, BDIM_X=blockDim.x*blockDim.y)
    //
    // STORAGE_T is the global-memory element type (float4 for the fp32 vectorized
    // path; float / c10::Half / c10::BFloat16 for the scalar path). COMPUTE_T is
    // the arithmetic type (float4 for the vectorized path, float otherwise) — all
    // dot products, softmax and accumulation happen in COMPUTE_T. vload/vstore
    // widen/narrow at the memory boundary.
    // Heads are NOT folded into the batch dimension. The tensors are physically
    // (B, nlat, nlon, nheads * nchan), and this kernel addresses one head of one
    // batch element: gridDim.y spans batch * nheads, and the head selects a
    // contiguous nchan-wide slice within each spatial point.
    //
    // Consequently nchan_in / nchan_out keep their meaning as the number of
    // channels this kernel iterates over, while the distance between adjacent
    // spatial points becomes ldi / ldo. Folding heads into batch would have been
    // free in a channels-first layout, but here the head axis sits inside the
    // innermost dimension, so pre-folding would cost a full copy.
    template <int BDIM_X, typename STORAGE_T>
    __global__ __launch_bounds__(BDIM_X) void s2_attn_fwd_generic_vec_k(
        int nheads,    // no. of attention heads packed along the channel dim
        int nchan_in,  // no. of STORAGE_T elements along channel dim, per head
        int nchan_out, // no. of STORAGE_T elements along channel dim, per head
        int nlat_in, int nlon_in, int nlat_out, int nlon_out, const STORAGE_T *__restrict__ kx,
        const STORAGE_T *__restrict__ vx, const STORAGE_T *__restrict__ qy, const int32_t *__restrict__ row_idx,
        const int64_t *__restrict__ row_off, const int64_t *__restrict__ col_idx,
        const float *__restrict__ quad_weights, STORAGE_T *__restrict__ y)
    {
        using COMPUTE_T = typename vec_traits<STORAGE_T>::compute_t;

        extern __shared__ __align__(sizeof(float4)) float shext[];
        COMPUTE_T *shy = reinterpret_cast<COMPUTE_T *>(shext) + threadIdx.y * nchan_out;

        const int bh = blockIdx.y;
        const int batch = bh / nheads;
        const int head = bh - (batch * nheads);

        // leading dimensions: elements between adjacent spatial points
        const int64_t ldi = int64_t(nheads) * nchan_in;
        const int64_t ldo = int64_t(nheads) * nchan_out;

        const int wid = blockIdx.x * blockDim.y + threadIdx.y;

        if (wid >= nlat_out * nlon_out) { return; }

        const int tidx = threadIdx.x;

        const int h = wid / nlon_out;
        const int wo = wid - (h * nlon_out);
        const int ho = row_idx[h];

        // one output lon step corresponds to pscale input lon steps (requires nlon_in % nlon_out == 0)
        const int pscale = nlon_in / nlon_out;

        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { shy[chan] = __vset<COMPUTE_T>(0.f); }

        kx += int64_t(batch) * nlat_in * nlon_in * ldi + int64_t(head) * nchan_in;
        qy += int64_t(batch) * nlat_out * nlon_out * ldi + int64_t(head) * nchan_in + int64_t(ho) * ldi * nlon_out
            + int64_t(wo) * ldi;

        vx += int64_t(batch) * nlat_in * nlon_in * ldo + int64_t(head) * nchan_out;
        y += int64_t(batch) * nlat_out * nlon_out * ldo + int64_t(head) * nchan_out + int64_t(ho) * ldo * nlon_out
            + int64_t(wo) * ldo;

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
            const int wip = wrap_lon(wi_wo, nlon_in);

            // stride between spatial points is ldi/ldo; the head offset is
            // already baked into kx/vx above
            const STORAGE_T *_kx = kx + int64_t(hi) * nlon_in * ldi + int64_t(wip) * ldi;
            const STORAGE_T *_vx = vx + int64_t(hi) * nlon_in * ldo + int64_t(wip) * ldo;

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
        int nheads,    // no. of attention heads packed along the channel dim
        int nchan_in,  // no. of STORAGE_T elements along channel dim, per head
        int nchan_out, // no. of STORAGE_T elements along channel dim, per head
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

        // see s2_attn_fwd_generic_vec_k: gridDim.y spans batch * nheads, and the
        // head selects an nchan-wide slice at each spatial point. nchan_in /
        // nchan_out remain per-head counts (they bound the loops and size shq);
        // only the distance between spatial points becomes ldi / ldo.
        const int bh = blockIdx.y;
        const int batch = bh / nheads;
        const int head = bh - (batch * nheads);

        const int64_t ldi = int64_t(nheads) * nchan_in;
        const int64_t ldo = int64_t(nheads) * nchan_out;

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

        kx += int64_t(batch) * nlat_in * nlon_in * ldi + int64_t(head) * nchan_in;
        qy += int64_t(batch) * nlat_out * nlon_out * ldi + int64_t(head) * nchan_in + int64_t(ho) * nlon_out * ldi
            + int64_t(wo) * ldi;
        if constexpr (CHIN_AS_OUT) {
            kx += tidx;
            qy += tidx;
        }

        vx += int64_t(batch) * nlat_in * nlon_in * ldo + int64_t(head) * nchan_out + tidx;
        y += int64_t(batch) * nlat_out * nlon_out * ldo + int64_t(head) * nchan_out + int64_t(ho) * nlon_out * ldo
            + int64_t(wo) * ldo + tidx;

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

        // Neighbors are processed in groups of NB.
        //
        // The one-at-a-time form this replaces had a fully serial dependency chain per
        // neighbor: col_idx -> address -> load k -> warp reduce -> softmax -> load v.
        // Nothing was in flight while a ~500-cycle k load resolved, and with nchan_in
        // = 64 over BDIM_X = 32 lanes there were only two independent loads inside the
        // channel loop to hide it with. Measured 1.6 TFLOP/s on H100 -- about 2% of
        // this GPU's scalar fp32 peak and ~300x off its bandwidth roofline, i.e. the
        // kernel was latency-bound, not compute- or bandwidth-bound.
        //
        // Grouping issues NB independent k loads (and later NB v loads) before any is
        // consumed, which is the entire point: memory-level parallelism, not fewer
        // flops. The online softmax is applied once per group instead of once per
        // neighbor. That is algebraically the same reduction -- one running max, one
        // rescale of locy per group rather than NB of them -- and is if anything
        // better conditioned, since it rescales less often.
        constexpr int NB = 4;

        int off = 0;
        for (; off + NB <= rlen; off += NB) {

            const STORAGE_T *kp[NB];
            const STORAGE_T *vp[NB];
            float qw[NB];

#pragma unroll
            for (int u = 0; u < NB; u++) {
                const int64_t col = col_idx[off + u];
                const int hi = col / nlon_in;
                const int wi = col - (hi * nlon_in);
                const int wi_wo = wi + pscale * wo;
                const int wip = wrap_lon(wi_wo, nlon_in);
                kp[u] = kx + int64_t(hi) * nlon_in * ldi + int64_t(wip) * ldi;
                vp[u] = vx + int64_t(hi) * nlon_in * ldo + int64_t(wip) * ldo;
                qw[u] = quad_weights[hi];
            }

            COMPUTE_T acc[NB];
#pragma unroll
            for (int u = 0; u < NB; u++) { acc[u] = __vset<COMPUTE_T>(0.f); }

            // one channel loop feeding NB accumulators, so NB loads are outstanding
            // per channel step rather than one
            if constexpr (CHIN_AS_OUT) {
#pragma unroll
                for (int i = 0; i < NLOC_M1; i++) {
                    const COMPUTE_T q = shq[i * BDIM_X];
#pragma unroll
                    for (int u = 0; u < NB; u++) { acc[u] = __vadd(acc[u], __vmul(q, vload(kp[u], i * BDIM_X))); }
                }
                if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                    const COMPUTE_T q = shq[NLOC_M1 * BDIM_X];
#pragma unroll
                    for (int u = 0; u < NB; u++) { acc[u] = __vadd(acc[u], __vmul(q, vload(kp[u], NLOC_M1 * BDIM_X))); }
                }
            } else {
                for (int chan = tidx; chan < nchan_in; chan += BDIM_X) {
                    const COMPUTE_T q = shq[chan];
#pragma unroll
                    for (int u = 0; u < NB; u++) { acc[u] = __vadd(acc[u], __vmul(q, vload(kp[u], chan))); }
                }
            }

            float qdotk[NB];
#pragma unroll
            for (int u = 0; u < NB; u++) {
                float t = __vred(acc[u]);
                if constexpr (BDIM_X == 32) {
                    t = __warp_sum(t);
                } else {
                    t = __block_sum<BDIM_X>(t);
                }
                qdotk[u] = t;
            }

            // group-wise online softmax: one running max and one rescale for all NB
            float qdotk_max_tmp = qdotk_max;
#pragma unroll
            for (int u = 0; u < NB; u++) { qdotk_max_tmp = max(qdotk_max_tmp, qdotk[u]); }
            const float exp_save = expf(qdotk_max - qdotk_max_tmp);

            float alpha[NB];
            float alpha_grp = 0.0f;
#pragma unroll
            for (int u = 0; u < NB; u++) {
                alpha[u] = expf(qdotk[u] - qdotk_max_tmp) * qw[u];
                alpha_grp += alpha[u];
            }
            alpha_sum = alpha_grp + alpha_sum * exp_save;

#pragma unroll
            for (int i = 0; i < NLOC_M1; i++) {
                COMPUTE_T t = __vscale(exp_save, locy[i]);
#pragma unroll
                for (int u = 0; u < NB; u++) { t = __vadd(t, __vscale(alpha[u], vload(vp[u], i * BDIM_X))); }
                locy[i] = t;
            }
            if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
                COMPUTE_T t = __vscale(exp_save, locy[NLOC_M1]);
#pragma unroll
                for (int u = 0; u < NB; u++) { t = __vadd(t, __vscale(alpha[u], vload(vp[u], NLOC_M1 * BDIM_X))); }
                locy[NLOC_M1] = t;
            }

            qdotk_max = qdotk_max_tmp;
        }

        // remainder: fewer than NB neighbors left, original one-at-a-time path
        for (; off < rlen; off++) {

            const int64_t col = col_idx[off];

            const int hi = col / nlon_in;
            const int wi = col - (hi * nlon_in);
            const int wi_wo = wi + pscale * wo;
            const int wip = wrap_lon(wi_wo, nlon_in);

            const STORAGE_T *_kx = kx + int64_t(hi) * nlon_in * ldi + int64_t(wip) * ldi;
            const STORAGE_T *_vx = vx + int64_t(hi) * nlon_in * ldo + int64_t(wip) * ldo;

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
    void launch_gen_attn_fwd(int batch_size, int nheads, int nchans_in, int nchans_out, int nlat_in, int nlon_in,
                             int nlat_out, int nlon_out, STORAGE_T *__restrict__ _kxp, STORAGE_T *__restrict__ _vxp,
                             STORAGE_T *__restrict__ _qyp, int32_t *_row_idx, int64_t *_row_off, int64_t *_col_idx,
                             float *_quad_weights, STORAGE_T *__restrict__ _yp, cudaStream_t stream)
    {

        dim3 block(WARP_SIZE, THREADS / WARP_SIZE);
        // one block row per (batch, head) pair
        dim3 grid(DIV_UP(nlat_out * nlon_out, block.y), batch_size * nheads);

        // shared memory holds compute-type (COMPUTE_T) data, not STORAGE_T.
        // sized from the per-head channel count, so it does not scale with nheads
        size_t shsize = sizeof(typename vec_traits<STORAGE_T>::compute_t) * nchans_out * block.y;

        s2_attn_fwd_generic_vec_k<THREADS>
            <<<grid, block, shsize, stream>>>(nheads, nchans_in, nchans_out, nlat_in, nlon_in, nlat_out, nlon_out, _kxp,
                                              _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp);
        CHECK_ERROR("s2_attn_fwd_generic_vec_k");

        return;
    }

    template <int BDIM_X, int BDIM_Y, int CUR_LOC_SIZE,
              int MAX_LOC_SIZE, // max size of COMPUTE_T[] local array
              typename STORAGE_T>
    void launch_spc_attn_fwd(int nloc, // "BDIM_X*nloc" >= nchans_out
                             int batch_size, int nheads, int nchans_in, int nchans_out, int nlat_in, int nlon_in,
                             int nlat_out, int nlon_out, STORAGE_T *__restrict__ _kxp, STORAGE_T *__restrict__ _vxp,
                             STORAGE_T *__restrict__ _qyp, int32_t *_row_idx, int64_t *_row_off, int64_t *_col_idx,
                             float *_quad_weights, STORAGE_T *__restrict__ _yp, cudaStream_t stream)
    {

        if (CUR_LOC_SIZE == nloc) {

            dim3 block(BDIM_X, BDIM_Y);
            // one block row per (batch, head) pair
            dim3 grid(DIV_UP(nlat_out * nlon_out, block.y), batch_size * nheads);

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

                s2_attn_fwd_special_vec_k<BDIM_X, BDIM_Y, 1, CUR_LOC_SIZE><<<grid, block, shsize, stream>>>(
                    nheads, nchans_in, nchans_out, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, _row_idx,
                    _row_off, _col_idx, _quad_weights, _yp);
            } else {

                s2_attn_fwd_special_vec_k<BDIM_X, BDIM_Y, 0, CUR_LOC_SIZE><<<grid, block, shsize, stream>>>(
                    nheads, nchans_in, nchans_out, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, _row_idx,
                    _row_off, _col_idx, _quad_weights, _yp);
            }
            CHECK_ERROR("s2_attn_fwd_special_vec_k");

            return;
        }
        if constexpr (CUR_LOC_SIZE < MAX_LOC_SIZE) {
            launch_spc_attn_fwd<BDIM_X, BDIM_Y, CUR_LOC_SIZE + 1, MAX_LOC_SIZE>(
                nloc, batch_size, nheads, nchans_in, nchans_out, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp,
                _row_idx, _row_off, _col_idx, _quad_weights, _yp, stream);
        }
        return;
    }

    // Picks the block size (BDIM_X) instance and launches the gather kernel for a
    // given storage vector type SV. MAX_LOC / MIN_LOC bound the per-thread local
    // array length (in COMPUTE_T units). nci / nco are channel counts in SV units.
    template <int MAX_LOC, int MIN_LOC, typename SV>
    static void fwd_dispatch_bdimx(int bdimx, int nloc, int64_t batch_size, int64_t nheads, int64_t nci, int64_t nco,
                                   int nlat_in, int64_t nlon_in, int64_t nlat_out, int64_t nlon_out, SV *_kxp, SV *_vxp,
                                   SV *_qyp, int32_t *_row_idx, int64_t *_row_off, int64_t *_col_idx,
                                   float *_quad_weights, SV *_yp, cudaStream_t stream)
    {
        // use 2D blocks only if 32 threads are enough
        switch (bdimx) {
        case 32:
            launch_spc_attn_fwd<32, 2, 1, MAX_LOC>(nloc, batch_size, nheads, nci, nco, nlat_in, nlon_in, nlat_out,
                                                   nlon_out, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx,
                                                   _quad_weights, _yp, stream);
            break;
        case 64:
            launch_spc_attn_fwd<64, 1, MIN_LOC, MAX_LOC>(nloc, batch_size, nheads, nci, nco, nlat_in, nlon_in, nlat_out,
                                                         nlon_out, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx,
                                                         _quad_weights, _yp, stream);
            break;
        case 128:
            launch_spc_attn_fwd<128, 1, MIN_LOC, MAX_LOC>(nloc, batch_size, nheads, nci, nco, nlat_in, nlon_in,
                                                          nlat_out, nlon_out, _kxp, _vxp, _qyp, _row_idx, _row_off,
                                                          _col_idx, _quad_weights, _yp, stream);
            break;
        case 256:
            launch_spc_attn_fwd<256, 1, MIN_LOC, MAX_LOC>(nloc, batch_size, nheads, nci, nco, nlat_in, nlon_in,
                                                          nlat_out, nlon_out, _kxp, _vxp, _qyp, _row_idx, _row_off,
                                                          _col_idx, _quad_weights, _yp, stream);
            break;
        case 512:
            launch_spc_attn_fwd<512, 1, MIN_LOC, MAX_LOC>(nloc, batch_size, nheads, nci, nco, nlat_in, nlon_in,
                                                          nlat_out, nlon_out, _kxp, _vxp, _qyp, _row_idx, _row_off,
                                                          _col_idx, _quad_weights, _yp, stream);
            break;
        case 1024:
            launch_spc_attn_fwd<1024, 1, MIN_LOC, MAX_LOC>(nloc, batch_size, nheads, nci, nco, nlat_in, nlon_in,
                                                           nlat_out, nlon_out, _kxp, _vxp, _qyp, _row_idx, _row_off,
                                                           _col_idx, _quad_weights, _yp, stream);
            break;
        default:
            launch_gen_attn_fwd(batch_size, nheads, nci, nco, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp,
                                _row_idx, _row_off, _col_idx, _quad_weights, _yp, stream);
            break;
        }
    }

    // Templated on the storage element type (float / c10::Half / c10::BFloat16).
    // Path selection (compute / accumulation are fp32 in every case):
    //   - fp32, 16B-aligned, nchans % 4 == 0  -> float4 vectorized (LDG.128)
    //   - fp16/bf16, 16B-aligned, nchans % 8 == 0 -> half8/bf168 vectorized (LDG.128)
    //   - otherwise -> scalar STORAGE_T path
    template <typename scalar_t>
    static void s2_attn_fwd_dispatch(int64_t batch_size, int64_t nheads, int64_t nchans_in, int64_t nchans_out,
                                     int64_t nlon_in, int64_t nlat_out, int64_t nlon_out, at::Tensor kxP,
                                     at::Tensor vxP, at::Tensor qyP, at::Tensor row_off, at::Tensor col_idx,
                                     at::Tensor quad_weights, at::Tensor yP)
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
                    bdimx, DIV_UP(nco, bdimx), batch_size, nheads, nci, nco, nlat_in, nlon_in, nlat_out, nlon_out,
                    reinterpret_cast<float4 *>(_kxp), reinterpret_cast<float4 *>(_vxp), reinterpret_cast<float4 *>(_qyp),
                    _row_idx, _row_off, _col_idx, _quad_weights, reinterpret_cast<float4 *>(_yp), stream);
            } else {
                fwd_dispatch_bdimx<MAX_LOCAL_ARR_LEN, MIN_LOC_ARR_LEN, float>(
                    bdimx, DIV_UP(nchans_out, bdimx), batch_size, nheads, nchans_in, nchans_out, nlat_in, nlon_in,
                    nlat_out, nlon_out, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp, stream);
            }
        } else {
            // fp16/bf16: scalar STORAGE_T path (widen at load, narrow at store; fp32
            // compute/accumulation). A vectorized 8-wide path was tried and reverted:
            // it raised register pressure and lowered occupancy, and ncu shows this
            // kernel is latency/occupancy-bound (DRAM ~25%), not bandwidth-bound, so
            // vectorizing reduced precision only hurt. See the AMP refactor notes.
            fwd_dispatch_bdimx<MAX_LOCAL_ARR_LEN, MIN_LOC_ARR_LEN, scalar_t>(
                bdimx, DIV_UP(nchans_out, bdimx), batch_size, nheads, nchans_in, nchans_out, nlat_in, nlon_in, nlat_out,
                nlon_out, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp, stream);
        }

        return;
    }

    // END - forward kernels and functions

    // NHWC ABI: kx, vx, qy are physically (B, nlat, nlon, num_heads * nchan) and
    // contiguous. Layout is never inferred from strides here -- the caller states
    // it by construction (see attention/_layout.py). Heads are packed along the
    // channel dimension rather than folded into the batch dimension, because
    // folding is not free in this layout.
    torch::Tensor s2_attention_fwd_cuda(at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor quad_weights,
                                        at::Tensor psi_col_idx, at::Tensor psi_row_off, int64_t num_heads,
                                        int64_t nlon_in, int64_t nlat_out, int64_t nlon_out)
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

        TORCH_CHECK(num_heads >= 1, "num_heads must be positive, got ", num_heads);
        TORCH_CHECK(qy.size(3) % num_heads == 0, "q/k channel count (", qy.size(3),
                    ") must be divisible by num_heads (", num_heads, ")");
        TORCH_CHECK(vx.size(3) % num_heads == 0, "v channel count (", vx.size(3), ") must be divisible by num_heads (",
                    num_heads, ")");

        // Every activation must share one dtype: the dispatch below selects a single
        // scalar_t from qy and the launchers reinterpret_cast k/v/q (and dy) to it, so
        // a mismatched input would be reinterpreted rather than converted.
        TORCH_CHECK(kx.scalar_type() == qy.scalar_type(), "k dtype (", kx.scalar_type(), ") must match q dtype (",
                    qy.scalar_type(), ")");
        TORCH_CHECK(vx.scalar_type() == qy.scalar_type(), "v dtype (", vx.scalar_type(), ") must match q dtype (",
                    qy.scalar_type(), ")");

        // per-head channel counts; the packed extent is num_heads times these
        size_t nchans_in = qy.size(3) / num_heads; // or kx.size(3) / num_heads
        size_t nchans_out = vx.size(3) / num_heads;

        const int batch_size = kx.size(0);
        const int64_t nlat_in = kx.size(1);

        // extract dtype
        auto qy_type = qy.dtype();

        const int64_t out_dims[] = {batch_size, nlat_out, nlon_out, int64_t(nchans_out) * num_heads};
        torch::Tensor y;

        // ATen dispatch over the input dtype.
        //
        // Both paths are on native storage: kx/vx/qy keep their dtype and y is
        // allocated in it, so there is no whole-tensor fp32 copy anywhere and the
        // read bandwidth for fp16/bf16 is halved. The kernels widen to fp32 at
        // load and narrow back at store (Tier B); compute and accumulation are
        // fp32 in-kernel either way.
        //
        // The scatter (upsample) path additionally keeps its softmax reduction
        // scratch -- numer/denom/maxbuf -- in fp32 regardless of the input dtype
        // (see launch_attn_fwd_upsample_scatter). Those are private accumulators,
        // not converted copies of the activations: a scatter accumulates an
        // unbounded number of contributions per output element, so narrowing them
        // would lose precision that the gather path never risks.
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, qy.scalar_type(), "s2_attention_fwd_cuda", [&] {
            using storage_t = scalar_t;

            // No layout conversion here any more: inputs are NHWC by contract and
            // the output is produced NHWC. The conversion, when a caller needs
            // one, happens once at the module boundary via the permute_to_nhwc /
            // permute_to_nchw ops rather than once per launcher.
            torch::Tensor y_nhwc = torch::empty(out_dims, kx.options()); // native dtype

            if (downsample) {
                s2_attn_fwd_dispatch<storage_t>(batch_size, num_heads, nchans_in, nchans_out, nlon_in, nlat_out,
                                                nlon_out, kx, vx, qy, psi_row_off, psi_col_idx, quad_weights, y_nhwc);
            } else {
                // upsample (scatter) path: s2_attn_fwd_upsample_dispatch does its own
                // AT_DISPATCH and widens fp16/bf16 at load (fp32 compute), narrowing
                // the output at store — same as the gather path.
                s2_attn_fwd_upsample_dispatch(batch_size, num_heads, nchans_in, nchans_out, nlon_in, nlat_in, nlat_out,
                                              nlon_out, kx, vx, qy, psi_row_off, psi_col_idx, quad_weights, y_nhwc);
            }

            y = y_nhwc;
        });

        // A no-op whenever kx and qy share a dtype, which is the only case this
        // function supports: the dispatch above selects scalar_t from qy, while y
        // is allocated from kx.options() and the launchers reinterpret_cast every
        // activation pointer to that one scalar_t. Mismatched input dtypes would
        // therefore misread storage long before reaching this line -- so this is a
        // shape-preserving formality, not a guard against that case.
        y = y.to(qy_type);

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return y;
    }

    TORCH_LIBRARY_IMPL(attention_kernels, CUDA, m) { m.impl("forward", &s2_attention_fwd_cuda); }

} // namespace attention_kernels
