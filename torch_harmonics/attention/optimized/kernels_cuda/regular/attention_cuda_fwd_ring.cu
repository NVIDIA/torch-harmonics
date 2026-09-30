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

#include "../common/attention_cuda.cuh"
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

#include "../common/cudamacro.h"
#include "../common/attention_cuda_utils.cuh"

#define THREADS (64)

#define MAX_LOCAL_ARR_LEN (16)

// Ring-step variant of the forward attention kernels in attention_cuda_fwd.cu, used by
// DistributedNeighborhoodAttentionS2. K/V are sharded along longitude across an
// azimuth process group and rotate around it; each call processes the chunk currently
// held and folds it into the online softmax state kept in y_acc / alpha_sum_buf /
// qdotk_max_buf, which persist across the ring steps.
//
// The kernels are the serial ones with two changes, and are kept that way on purpose:
//
//   - the state is loaded at the start and stored at the end, instead of being
//     initialized and normalized, and
//   - every arc is clipped to the chunk (clip_arc), and to the halo-padded latitude
//     range it holds, before it is walked.
//
// psi is the serial arc form, sliced to this rank's output rows, with the arc starts
// pre-shifted by pscale * lon_lo_out so that the local output longitude addresses it
// exactly as the serial kernels do (see RingGatherBackend in distributed_attention.py).

namespace attention_kernels
{

    // see s2_attn_fwd_generic_vec_k
    template <int BDIM_X, typename STORAGE_T>
    __global__ __launch_bounds__(BDIM_X) void s2_attn_fwd_ring_generic_vec_k(
        const __grid_constant__ attn_params_t p,
        const STORAGE_T *__restrict__ kx, // [batch][nlat_halo][nlon_kx][nheads*nchan_in]
        const STORAGE_T *__restrict__ vx, // [batch][nlat_halo][nlon_kx][nheads*nchan_out]
        const STORAGE_T *__restrict__ qy, // [batch][nlat_out][nlon_out][nheads*nchan_in]
        const int32_t *__restrict__ row_idx, const int32_t *__restrict__ seg, const int32_t *__restrict__ seg_off,
        const float *__restrict__ ring_weights,
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ y_acc, // [batch][nlat_out][nlon_out][nheads*nchan_out] (in/out)
        float *__restrict__ alpha_sum_buf,                             // [batch][nheads][nlat_out][nlon_out] (in/out)
        float *__restrict__ qdotk_max_buf)                             // [batch][nheads][nlat_out][nlon_out] (in/out)
    {
        using COMPUTE_T = typename vec_traits<STORAGE_T>::compute_t;

        const int &nheads = p.nheads;
        const int &nchan_in = p.nchan_in;
        const int &nchan_out = p.nchan_out;
        const int &nlat_halo = p.nlat_halo;
        const int &nlon_kx = p.nlon_kx;
        const int &nlon_in = p.nlon_in;
        const int &pscale = p.pscale;
        const int &nlat_out = p.nlat_out;
        const int &nlon_out = p.nlon_out;

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
        const int wo = wid - (h * nlon_out); // LOCAL wo
        const int ho = row_idx[h];

        kx += int64_t(batch) * nlat_halo * nlon_kx * ldi + int64_t(head) * nchan_in;
        qy += int64_t(batch) * nlat_out * nlon_out * ldi + int64_t(head) * nchan_in + int64_t(ho) * ldi * nlon_out
            + int64_t(wo) * ldi;

        vx += int64_t(batch) * nlat_halo * nlon_kx * ldo + int64_t(head) * nchan_out;
        y_acc += int64_t(batch) * nlat_out * nlon_out * ldo + int64_t(head) * nchan_out + int64_t(ho) * ldo * nlon_out
            + int64_t(wo) * ldo;

        // softmax state is one value per (batch, head, point)
        const int64_t stat_off = int64_t(bh) * nlat_out * nlon_out + int64_t(ho) * nlon_out + wo;
        alpha_sum_buf += stat_off;
        qdotk_max_buf += stat_off;

        // resume the online softmax where the previous ring step left it
        float alpha_sum = alpha_sum_buf[0];
        float qdotk_max = qdotk_max_buf[0];
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { shy[chan] = y_acc[chan]; }

        const int seg_beg = seg_off[ho];
        const int seg_end = seg_off[ho + 1];

        for (int sg = seg_beg; sg < seg_end; sg++) {

            const int hi = seg[3 * sg + 0];
            const int lo = seg[3 * sg + 1];
            const int len = seg[3 * sg + 2];

            // the chunk holds only the halo-padded latitudes of this polar rank
            const int hi_local = hi - p.lat_halo_start;
            if (hi_local < 0 || hi_local >= nlat_halo) { continue; }

            const float qw = ring_weights[hi];

            const STORAGE_T *kx_row = kx + int64_t(hi_local) * nlon_kx * ldi;
            const STORAGE_T *vx_row = vx + int64_t(hi_local) * nlon_kx * ldo;

            // the part of the arc inside the chunk, as at most two contiguous pieces
            int2 piece[2];
            const int npiece = clip_arc(wrap_lon(lo + pscale * wo, nlon_in), len, nlon_in, p.lon_lo_kx, nlon_kx, piece);

            for (int pc = 0; pc < npiece; pc++) {

                int wip = piece[pc].x;

                for (int j = 0; j < piece[pc].y; j++) {

                    const STORAGE_T *_kx = kx_row + int64_t(wip) * ldi;
                    const STORAGE_T *_vx = vx_row + int64_t(wip) * ldo;

                    COMPUTE_T qdotkv = __vset<COMPUTE_T>(0.f);

                    for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
                        qdotkv = __vadd(qdotkv, __vmul(vload(qy, chan), vload(_kx, chan)));
                    }

                    float qdotk = __warp_sum(__vred(qdotkv));

                    float qdotk_max_tmp;
                    float alpha;
                    float exp_save;

                    qdotk_max_tmp = max(qdotk_max, qdotk);
                    alpha = expf(qdotk - qdotk_max_tmp) * qw;
                    exp_save = expf(qdotk_max - qdotk_max_tmp);

                    alpha_sum = alpha + alpha_sum * exp_save;

                    for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
                        shy[chan] = __vadd(__vscale(exp_save, shy[chan]), __vscale(alpha, vload(_vx, chan)));
                    }
                    qdotk_max = qdotk_max_tmp;

                    // pieces do not wrap: clip_arc split the arc at the seam
                    wip++;
                }
            }
        }

        // hand the state on to the next ring step; normalization happens after the last
        if (!tidx) {
            alpha_sum_buf[0] = alpha_sum;
            qdotk_max_buf[0] = qdotk_max;
        }
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { y_acc[chan] = shy[chan]; }

        return;
    }

    // see s2_attn_fwd_special_vec_k
    template <int BDIM_X, int BDIM_Y,
              int CHIN_AS_OUT, // 1 iif "BDIM_X*(NLOC-1) <= nchan_in <= BDIM_X*NLOC" else 0
              int NLOC,        // smallest int such that BDIM_X*NLOC >= nchan_out
              typename STORAGE_T>
    __global__ __launch_bounds__(BDIM_X *BDIM_Y) void s2_attn_fwd_ring_special_vec_k(
        const __grid_constant__ attn_params_t p,
        const STORAGE_T *__restrict__ kx, // [batch][nlat_halo][nlon_kx][nheads*nchan_in]
        const STORAGE_T *__restrict__ vx, // [batch][nlat_halo][nlon_kx][nheads*nchan_out]
        const STORAGE_T *__restrict__ qy, // [batch][nlat_out][nlon_out][nheads*nchan_in]
        const int32_t *__restrict__ row_idx, const int32_t *__restrict__ seg, const int32_t *__restrict__ seg_off,
        const float *__restrict__ ring_weights,
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ y_acc, // [batch][nlat_out][nlon_out][nheads*nchan_out] (in/out)
        float *__restrict__ alpha_sum_buf,                             // [batch][nheads][nlat_out][nlon_out] (in/out)
        float *__restrict__ qdotk_max_buf)                             // [batch][nheads][nlat_out][nlon_out] (in/out)
    {
        using COMPUTE_T = typename vec_traits<STORAGE_T>::compute_t;

        static_assert(0 == (BDIM_X & (BDIM_X - 1)));
        static_assert(0 == (BDIM_Y & (BDIM_Y - 1)));
        static_assert((BDIM_X == 32 && BDIM_Y > 1) || (BDIM_X > 32 && BDIM_Y == 1));

        constexpr int NLOC_M1 = NLOC - 1;

        const int &nheads = p.nheads;
        const int &nchan_in = p.nchan_in;
        const int &nchan_out = p.nchan_out;
        const int &nlat_halo = p.nlat_halo;
        const int &nlon_kx = p.nlon_kx;
        const int &nlon_in = p.nlon_in;
        const int &pscale = p.pscale;
        const int &nlat_out = p.nlat_out;
        const int &nlon_out = p.nlon_out;

        const int tidx = threadIdx.x;

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
        const int wo = ctaid - (h * nlon_out); // LOCAL wo
        const int ho = row_idx[h];

        kx += int64_t(batch) * nlat_halo * nlon_kx * ldi + int64_t(head) * nchan_in;
        qy += int64_t(batch) * nlat_out * nlon_out * ldi + int64_t(head) * nchan_in + int64_t(ho) * nlon_out * ldi
            + int64_t(wo) * ldi;
        if constexpr (CHIN_AS_OUT) {
            kx += tidx;
            qy += tidx;
        }

        vx += int64_t(batch) * nlat_halo * nlon_kx * ldo + int64_t(head) * nchan_out + tidx;
        y_acc += int64_t(batch) * nlat_out * nlon_out * ldo + int64_t(head) * nchan_out + int64_t(ho) * nlon_out * ldo
            + int64_t(wo) * ldo + tidx;

        // softmax state is one value per (batch, head, point)
        const int64_t stat_off = int64_t(bh) * nlat_out * nlon_out + int64_t(ho) * nlon_out + wo;
        alpha_sum_buf += stat_off;
        qdotk_max_buf += stat_off;

        // resume the online softmax where the previous ring step left it
#pragma unroll
        for (int i = 0; i < NLOC_M1; i++) { locy[i] = y_acc[i * BDIM_X]; }
        locy[NLOC_M1] = __vset<COMPUTE_T>(0.f);
        if (NLOC_M1 * BDIM_X + tidx < nchan_out) { locy[NLOC_M1] = y_acc[NLOC_M1 * BDIM_X]; }

        if constexpr (CHIN_AS_OUT) {
#pragma unroll
            for (int i = 0; i < NLOC_M1; i++) { shq[i * BDIM_X] = vload(qy, i * BDIM_X); }
            if (NLOC_M1 * BDIM_X + tidx < nchan_in) { shq[NLOC_M1 * BDIM_X] = vload(qy, NLOC_M1 * BDIM_X); }
        } else {
            for (int chan = tidx; chan < nchan_in; chan += BDIM_X) { shq[chan] = vload(qy, chan); }
        }

        float alpha_sum = alpha_sum_buf[0];
        float qdotk_max = qdotk_max_buf[0];

        // Neighbours per group; see s2_attn_fwd_special_vec_k, whose override this shares.
#ifndef TH_ATTENTION_FWD_NB
#define TH_ATTENTION_FWD_NB 4
#endif
        constexpr int NB = TH_ATTENTION_FWD_NB;

        const int seg_beg = seg_off[ho];
        const int seg_end = seg_off[ho + 1];

        for (int sg = seg_beg; sg < seg_end; sg++) {

            const int hi = seg[3 * sg + 0];
            const int lo = seg[3 * sg + 1];
            const int len = seg[3 * sg + 2];

            // the chunk holds only the halo-padded latitudes of this polar rank
            const int hi_local = hi - p.lat_halo_start;
            if (hi_local < 0 || hi_local >= nlat_halo) { continue; }

            const float qw_seg = ring_weights[hi];
            const STORAGE_T *kx_row = kx + int64_t(hi_local) * nlon_kx * ldi;
            const STORAGE_T *vx_row = vx + int64_t(hi_local) * nlon_kx * ldo;

            // the part of the arc inside the chunk, as at most two contiguous pieces
            int2 piece[2];
            const int npiece = clip_arc(wrap_lon(lo + pscale * wo, nlon_in), len, nlon_in, p.lon_lo_kx, nlon_kx, piece);

            for (int pc = 0; pc < npiece; pc++) {

                int wip = piece[pc].x;
                const int plen = piece[pc].y;

                int j = 0;
                for (; j + NB <= plen; j += NB) {

                    const STORAGE_T *kp[NB];
                    const STORAGE_T *vp[NB];

#pragma unroll
                    for (int u = 0; u < NB; u++) {
                        kp[u] = kx_row + int64_t(wip) * ldi;
                        vp[u] = vx_row + int64_t(wip) * ldo;
                        wip++;
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
                            for (int u = 0; u < NB; u++) {
                                acc[u] = __vadd(acc[u], __vmul(q, vload(kp[u], i * BDIM_X)));
                            }
                        }
                        if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                            const COMPUTE_T q = shq[NLOC_M1 * BDIM_X];
#pragma unroll
                            for (int u = 0; u < NB; u++) {
                                acc[u] = __vadd(acc[u], __vmul(q, vload(kp[u], NLOC_M1 * BDIM_X)));
                            }
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
                        alpha[u] = expf(qdotk[u] - qdotk_max_tmp) * qw_seg;
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
                        for (int u = 0; u < NB; u++) {
                            t = __vadd(t, __vscale(alpha[u], vload(vp[u], NLOC_M1 * BDIM_X)));
                        }
                        locy[NLOC_M1] = t;
                    }

                    qdotk_max = qdotk_max_tmp;
                }

                // remainder: fewer than NB neighbours left in this piece
                for (; j < plen; j++) {

                    const STORAGE_T *_kx = kx_row + int64_t(wip) * ldi;
                    const STORAGE_T *_vx = vx_row + int64_t(wip) * ldo;

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
                    alpha = expf(qdotk - qdotk_max_tmp) * qw_seg;
                    exp_save = expf(qdotk_max - qdotk_max_tmp);

                    alpha_sum = alpha + alpha_sum * exp_save;

#pragma unroll
                    for (int i = 0; i < NLOC_M1; i++) {
                        locy[i] = __vadd(__vscale(exp_save, locy[i]), __vscale(alpha, vload(_vx, i * BDIM_X)));
                    }
                    if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
                        locy[NLOC_M1]
                            = __vadd(__vscale(exp_save, locy[NLOC_M1]), __vscale(alpha, vload(_vx, NLOC_M1 * BDIM_X)));
                    }

                    qdotk_max = qdotk_max_tmp;

                    wip++;
                }
            }
        }

        // hand the state on to the next ring step; normalization happens after the last
        if (!tidx) {
            alpha_sum_buf[0] = alpha_sum;
            qdotk_max_buf[0] = qdotk_max;
        }

#pragma unroll
        for (int i = 0; i < NLOC_M1; i++) { y_acc[i * BDIM_X] = locy[i]; }
        if (NLOC_M1 * BDIM_X + tidx < nchan_out) { y_acc[NLOC_M1 * BDIM_X] = locy[NLOC_M1]; }

        return;
    }

    template <typename STORAGE_T>
    void launch_gen_attn_ring_fwd(int batch_size, const attn_params_t &params, STORAGE_T *_kxp, STORAGE_T *_vxp,
                                  STORAGE_T *_qyp, int32_t *_row_idx, const int32_t *_seg, const int32_t *_seg_off,
                                  float *_quad_weights, typename vec_traits<STORAGE_T>::compute_t *_y_acc,
                                  float *_alpha_sum, float *_qdotk_max, cudaStream_t stream)
    {

        dim3 block(WARP_SIZE, THREADS / WARP_SIZE);
        // one block row per (batch, head) pair
        dim3 grid(DIV_UP(params.nlat_out * params.nlon_out, block.y), batch_size * params.nheads);

        // shared memory holds compute-type (COMPUTE_T) data, not STORAGE_T.
        // sized from the per-head channel count, so it does not scale with nheads
        size_t shsize = sizeof(typename vec_traits<STORAGE_T>::compute_t) * params.nchan_out * block.y;

        launch_dyn_shmem(&s2_attn_fwd_ring_generic_vec_k<THREADS, STORAGE_T>, grid, block, shsize, stream, params, _kxp,
                         _vxp, _qyp, _row_idx, _seg, _seg_off, _quad_weights, _y_acc, _alpha_sum, _qdotk_max);
        CHECK_ERROR("s2_attn_fwd_ring_generic_vec_k");

        return;
    }

    template <int BDIM_X, int BDIM_Y, int CUR_LOC_SIZE,
              int MAX_LOC_SIZE, // max size of COMPUTE_T[] local array
              typename STORAGE_T>
    void launch_spc_attn_ring_fwd(int nloc, // "BDIM_X*nloc" >= nchans_out
                                  int batch_size, const attn_params_t &params, STORAGE_T *_kxp, STORAGE_T *_vxp,
                                  STORAGE_T *_qyp, int32_t *_row_idx, const int32_t *_seg, const int32_t *_seg_off,
                                  float *_quad_weights, typename vec_traits<STORAGE_T>::compute_t *_y_acc,
                                  float *_alpha_sum, float *_qdotk_max, cudaStream_t stream)
    {

        if (CUR_LOC_SIZE == nloc) {

            dim3 block(BDIM_X, BDIM_Y);
            // one block row per (batch, head) pair
            dim3 grid(DIV_UP(params.nlat_out * params.nlon_out, block.y), batch_size * params.nheads);

            // shared memory holds compute-type (COMPUTE_T) data, not STORAGE_T.
            // block.y > 1 iif block.x==32
            size_t shsize = sizeof(typename vec_traits<STORAGE_T>::compute_t) * params.nchan_in * block.y;

            // see launch_spc_attn_fwd
            if (params.nchan_in >= BDIM_X * (CUR_LOC_SIZE - 1) && params.nchan_in <= BDIM_X * CUR_LOC_SIZE) {

                launch_dyn_shmem(&s2_attn_fwd_ring_special_vec_k<BDIM_X, BDIM_Y, 1, CUR_LOC_SIZE, STORAGE_T>, grid,
                                 block, shsize, stream, params, _kxp, _vxp, _qyp, _row_idx, _seg, _seg_off,
                                 _quad_weights, _y_acc, _alpha_sum, _qdotk_max);
            } else {

                launch_dyn_shmem(&s2_attn_fwd_ring_special_vec_k<BDIM_X, BDIM_Y, 0, CUR_LOC_SIZE, STORAGE_T>, grid,
                                 block, shsize, stream, params, _kxp, _vxp, _qyp, _row_idx, _seg, _seg_off,
                                 _quad_weights, _y_acc, _alpha_sum, _qdotk_max);
            }
            CHECK_ERROR("s2_attn_fwd_ring_special_vec_k");

            return;
        }
        if constexpr (CUR_LOC_SIZE < MAX_LOC_SIZE) {
            launch_spc_attn_ring_fwd<BDIM_X, BDIM_Y, CUR_LOC_SIZE + 1, MAX_LOC_SIZE>(
                nloc, batch_size, params, _kxp, _vxp, _qyp, _row_idx, _seg, _seg_off, _quad_weights, _y_acc, _alpha_sum,
                _qdotk_max, stream);
        }
        return;
    }

    // see fwd_dispatch_bdimx
    template <int MAX_LOC, int MIN_LOC, typename SV>
    static void fwd_ring_dispatch_bdimx(int bdimx, int nloc, int64_t batch_size, const attn_params_t &params, SV *_kxp,
                                        SV *_vxp, SV *_qyp, int32_t *_row_idx, const int32_t *_seg,
                                        const int32_t *_seg_off, float *_quad_weights,
                                        typename vec_traits<SV>::compute_t *_y_acc, float *_alpha_sum,
                                        float *_qdotk_max, cudaStream_t stream)
    {
        // use 2D blocks only if 32 threads are enough
        switch (bdimx) {
        case 32:
            launch_spc_attn_ring_fwd<32, 2, 1, MAX_LOC>(nloc, batch_size, params, _kxp, _vxp, _qyp, _row_idx, _seg,
                                                        _seg_off, _quad_weights, _y_acc, _alpha_sum, _qdotk_max, stream);
            break;
        case 64:
            launch_spc_attn_ring_fwd<64, 1, MIN_LOC, MAX_LOC>(nloc, batch_size, params, _kxp, _vxp, _qyp, _row_idx,
                                                              _seg, _seg_off, _quad_weights, _y_acc, _alpha_sum,
                                                              _qdotk_max, stream);
            break;
        case 128:
            launch_spc_attn_ring_fwd<128, 1, MIN_LOC, MAX_LOC>(nloc, batch_size, params, _kxp, _vxp, _qyp, _row_idx,
                                                               _seg, _seg_off, _quad_weights, _y_acc, _alpha_sum,
                                                               _qdotk_max, stream);
            break;
        case 256:
            launch_spc_attn_ring_fwd<256, 1, MIN_LOC, MAX_LOC>(nloc, batch_size, params, _kxp, _vxp, _qyp, _row_idx,
                                                               _seg, _seg_off, _quad_weights, _y_acc, _alpha_sum,
                                                               _qdotk_max, stream);
            break;
        case 512:
            launch_spc_attn_ring_fwd<512, 1, MIN_LOC, MAX_LOC>(nloc, batch_size, params, _kxp, _vxp, _qyp, _row_idx,
                                                               _seg, _seg_off, _quad_weights, _y_acc, _alpha_sum,
                                                               _qdotk_max, stream);
            break;
        case 1024:
            launch_spc_attn_ring_fwd<1024, 1, MIN_LOC, MAX_LOC>(nloc, batch_size, params, _kxp, _vxp, _qyp, _row_idx,
                                                                _seg, _seg_off, _quad_weights, _y_acc, _alpha_sum,
                                                                _qdotk_max, stream);
            break;
        default:
            launch_gen_attn_ring_fwd(batch_size, params, _kxp, _vxp, _qyp, _row_idx, _seg, _seg_off, _quad_weights,
                                     _y_acc, _alpha_sum, _qdotk_max, stream);
            break;
        }
    }

    // see s2_attn_fwd_dispatch; the path selection is the same
    template <typename scalar_t>
    static void s2_attn_fwd_ring_dispatch(int64_t batch_size, attn_params_t params, at::Tensor kxP, at::Tensor vxP,
                                          at::Tensor qyP, at::Tensor row_off, at::Tensor seg, at::Tensor seg_off,
                                          at::Tensor ring_weights, at::Tensor y_acc, at::Tensor alpha_sum_buf,
                                          at::Tensor qdotk_max_buf)
    {

        static_assert(0 == (MAX_LOCAL_ARR_LEN & (MAX_LOCAL_ARR_LEN - 1)));

        // get stream
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        // sort row indices (ho-s) in descending order
        // based on (row_off[ho+1]-row_off[ho])
        at::Tensor row_idx = sortRows(params.nlat_out, row_off, stream);

        const int64_t nchans_in = params.nchan_in;
        const int64_t nchans_out = params.nchan_out;

        // smallest power of two "bdimx" (>=32) s.t. bdimx*MAX_LOCAL_ARR_LEN >= nchans_out
        int bdimx;
        bdimx = DIV_UP(nchans_out, MAX_LOCAL_ARR_LEN);
        bdimx = max(bdimx, WARP_SIZE);
        bdimx = next_pow2(bdimx);

        scalar_t *_kxp = reinterpret_cast<scalar_t *>(kxP.data_ptr());
        scalar_t *_vxp = reinterpret_cast<scalar_t *>(vxP.data_ptr());
        scalar_t *_qyp = reinterpret_cast<scalar_t *>(qyP.data_ptr());

        int32_t *_row_idx = reinterpret_cast<int32_t *>(row_idx.data_ptr());
        const int32_t *_seg = reinterpret_cast<const int32_t *>(seg.data_ptr());
        const int32_t *_seg_off = reinterpret_cast<const int32_t *>(seg_off.data_ptr());
        float *_quad_weights = reinterpret_cast<float *>(ring_weights.data_ptr());

        // the state is fp32 whatever the activations are
        float *_y_acc = reinterpret_cast<float *>(y_acc.data_ptr());
        float *_alpha_sum = reinterpret_cast<float *>(alpha_sum_buf.data_ptr());
        float *_qdotk_max = reinterpret_cast<float *>(qdotk_max_buf.data_ptr());

        constexpr int MIN_LOC_ARR_LEN = MAX_LOCAL_ARR_LEN / 2 + 1;

        // the vectorized forms read and write y_acc as float4 too
        constexpr int VEC_SIZE = 4;
        const bool vec_fills_block = (nchans_in % VEC_SIZE) == 0 && (nchans_out % VEC_SIZE) == 0
            && (nchans_in / VEC_SIZE) >= bdimx && (nchans_out / VEC_SIZE) >= bdimx && is_aligned<16>(_y_acc);

        if constexpr (std::is_same<scalar_t, float>::value) {
            const bool use_vec = vec_fills_block && is_aligned<16>(_kxp) && is_aligned<16>(_vxp) && is_aligned<16>(_qyp);

            if (use_vec) {
                constexpr int MAX_VEC = MAX_LOCAL_ARR_LEN / VEC_SIZE;
                constexpr int MIN_VEC = MAX_VEC / 2 + 1;
                params.nchan_in = nchans_in / VEC_SIZE;
                params.nchan_out = nchans_out / VEC_SIZE;
                fwd_ring_dispatch_bdimx<MAX_VEC, MIN_VEC, float4>(
                    bdimx, DIV_UP(params.nchan_out, bdimx), batch_size, params, reinterpret_cast<float4 *>(_kxp),
                    reinterpret_cast<float4 *>(_vxp), reinterpret_cast<float4 *>(_qyp), _row_idx, _seg, _seg_off,
                    _quad_weights, reinterpret_cast<float4 *>(_y_acc), _alpha_sum, _qdotk_max, stream);
            } else {
                fwd_ring_dispatch_bdimx<MAX_LOCAL_ARR_LEN, MIN_LOC_ARR_LEN, float>(
                    bdimx, DIV_UP(nchans_out, bdimx), batch_size, params, _kxp, _vxp, _qyp, _row_idx, _seg, _seg_off,
                    _quad_weights, _y_acc, _alpha_sum, _qdotk_max, stream);
            }
        } else {
            const bool use_vec = vec_fills_block && is_aligned<8>(_kxp) && is_aligned<8>(_vxp) && is_aligned<8>(_qyp);

            if (use_vec) {
                using vec_t = std::conditional_t<std::is_same<scalar_t, at::Half>::value, half4, bf164>;
                constexpr int MAX_VEC = MAX_LOCAL_ARR_LEN / VEC_SIZE;
                constexpr int MIN_VEC = MAX_VEC / 2 + 1;
                params.nchan_in = nchans_in / VEC_SIZE;
                params.nchan_out = nchans_out / VEC_SIZE;
                fwd_ring_dispatch_bdimx<MAX_VEC, MIN_VEC, vec_t>(
                    bdimx, DIV_UP(params.nchan_out, bdimx), batch_size, params, reinterpret_cast<vec_t *>(_kxp),
                    reinterpret_cast<vec_t *>(_vxp), reinterpret_cast<vec_t *>(_qyp), _row_idx, _seg, _seg_off,
                    _quad_weights, reinterpret_cast<float4 *>(_y_acc), _alpha_sum, _qdotk_max, stream);
            } else {
                fwd_ring_dispatch_bdimx<MAX_LOCAL_ARR_LEN, MIN_LOC_ARR_LEN, scalar_t>(
                    bdimx, DIV_UP(nchans_out, bdimx), batch_size, params, _kxp, _vxp, _qyp, _row_idx, _seg, _seg_off,
                    _quad_weights, _y_acc, _alpha_sum, _qdotk_max, stream);
            }
        }

        return;
    }

    // NHWC ABI, heads packed along channels -- see s2_attention_fwd_cuda. The state
    // follows the same convention: y_acc is (B, H, W, num_heads * nchan_out), and
    // alpha_sum / qdotk_max are (B, num_heads, H, W).
    void s2_attention_fwd_ring_step_cuda(at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor y_acc,
                                         at::Tensor alpha_sum_buf, at::Tensor qdotk_max_buf, at::Tensor ring_weights,
                                         at::Tensor psi_seg, at::Tensor psi_seg_off, int64_t num_heads, int64_t nlon_in,
                                         int64_t nlon_out_global, int64_t lon_lo_kx, int64_t lat_halo_start,
                                         int64_t nlat_out, int64_t nlon_out)
    {
        CHECK_CUDA_INPUT_TENSOR(kx);
        CHECK_CUDA_INPUT_TENSOR(vx);
        CHECK_CUDA_INPUT_TENSOR(qy);
        CHECK_CUDA_TENSOR(y_acc);
        CHECK_CUDA_TENSOR(alpha_sum_buf);
        CHECK_CUDA_TENSOR(qdotk_max_buf);
        CHECK_CUDA_TENSOR(ring_weights);

        // run on the inputs' device: without this, the current stream, the scratch
        // allocations and the per-device queries (ensure_dyn_shmem, getPtxver) would all
        // resolve to whichever CUDA device happens to be current
        const at::cuda::OptionalCUDAGuard device_guard(kx.device());
        // devices, shapes, index and weight dtypes, dense layouts; the arc rows are this
        // rank's output latitudes
        check_ring_step_inputs(kx, vx, qy, ring_weights, psi_seg, psi_seg_off, num_heads, nlat_out, nlon_out, nlat_out);
        check_state_buffer(y_acc, kx, "y_acc", {kx.size(0), nlat_out, nlon_out, vx.size(3)});
        check_state_buffer(alpha_sum_buf, kx, "alpha_sum_buf", {kx.size(0), num_heads, nlat_out, nlon_out});
        check_state_buffer(qdotk_max_buf, kx, "qdotk_max_buf", {kx.size(0), num_heads, nlat_out, nlon_out});

        // row offsets from the arcs, for sortRows -- see s2_attention_fwd_cuda
        auto seg_cum = torch::zeros({psi_seg.size(0) + 1}, psi_seg.options().dtype(torch::kInt64));
        seg_cum.slice(0, 1).copy_(torch::cumsum(psi_seg.select(1, 2).to(torch::kInt64), 0));
        const auto psi_row_off = seg_cum.index_select(0, psi_seg_off.to(torch::kInt64));

        // the p-shift ratio, derived as the serial ops derive it but from the GLOBAL
        // output width: nlon_out is this rank's
        TORCH_CHECK(nlon_out_global > 0 && nlon_in % nlon_out_global == 0, "nlon_in (", nlon_in,
                    ") must be an integer multiple of nlon_out_global (", nlon_out_global, ")");

        attn_params_t params;
        params.nheads = num_heads;
        // per-head channel counts; the packed extent is num_heads times these
        params.nchan_in = qy.size(3) / num_heads;
        params.nchan_out = vx.size(3) / num_heads;
        params.nlat_halo = kx.size(1);
        params.nlon_kx = kx.size(2);
        params.nlon_in = nlon_in;
        params.pscale = nlon_in / nlon_out_global;
        params.lon_lo_kx = lon_lo_kx;
        params.lat_halo_start = lat_halo_start;
        params.nlat_out = nlat_out;
        params.nlon_out = nlon_out;

        const int batch_size = kx.size(0);

        // Native storage: kx/vx/qy keep their dtype and are widened at load; the state
        // is allocated fp32 in Python and stays fp32.
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, qy.scalar_type(), "s2_attention_fwd_ring_step_cuda", [&] {
            s2_attn_fwd_ring_dispatch<scalar_t>(batch_size, params, kx, vx, qy, psi_row_off, psi_seg, psi_seg_off,
                                                ring_weights, y_acc, alpha_sum_buf, qdotk_max_buf);
        });

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    TORCH_LIBRARY_IMPL(attention_kernels, CUDA, m) { m.impl("forward_ring_step", &s2_attention_fwd_ring_step_cuda); }

} // namespace attention_kernels
