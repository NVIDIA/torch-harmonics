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
#include <array>
#include <limits>

#include "../common/cudamacro.h"
#include "../common/attention_cuda_utils.cuh"

#define THREADS (64)

#define MAX_LOCAL_ARR_LEN (16)

// Ring-step variant of the backward attention kernels in attention_cuda_bwd.cu, used by
// DistributedNeighborhoodAttentionS2. The serial backward is one kernel with two
// neighbour loops: the first accumulates the softmax statistics and yields dqy, the
// second scatters dkx/dvx with the finalized statistics. On the ring those loops
// cannot share a kernel, because the statistics are only final once every chunk has
// been seen, so each becomes a pass run once per ring step:
//
//   pass1: the first loop. Accumulates alpha_sum, qdotk_max, integral and the
//          per-output alpha_k, alpha_kvw into state that persists across ring
//          steps; dqy is finalized from it in Python after the last step.
//   pass2: the second loop. Scatters dkx/dvx into the current chunk, reading the
//          finalized alpha_sum, qdotk_max and integral_norm.
//
// As in the forward ring kernels, the loops are the serial ones except that the
// state is loaded and stored rather than initialized, and every arc is clipped to the
// chunk (clip_arc) and to its halo-padded latitude range. One simplification against
// the serial kernel: its per-channel alpha_vw is the scalar integral broadcast across
// channels, so pass1 carries the scalar.

namespace attention_kernels
{

    ///////////////// BEGIN PASS1 SECTION

    // see the first neighbour loop of s2_attn_bwd_generic_vec_k
    template <int BDIM_X, typename STORAGE_T>
    __global__ __launch_bounds__(BDIM_X) void s2_attn_bwd_ring_pass1_generic_vec_k(
        const __grid_constant__ attn_params_t p,
        const STORAGE_T *__restrict__ kx, // [batch][nlat_halo][nlon_kx][nheads*nchan_in]
        const STORAGE_T *__restrict__ vx, // [batch][nlat_halo][nlon_kx][nheads*nchan_out]
        const STORAGE_T *__restrict__ qy, // [batch][nlat_out][nlon_out][nheads*nchan_in]
        const STORAGE_T *__restrict__ dy, // [batch][nlat_out][nlon_out][nheads*nchan_out]
        const int32_t *__restrict__ row_idx, const int32_t *__restrict__ seg, const int32_t *__restrict__ seg_off,
        const float *__restrict__ ring_weights,
        float *__restrict__ alpha_sum_buf, // [batch][nheads][nlat_out][nlon_out] (in/out)
        float *__restrict__ qdotk_max_buf, // [batch][nheads][nlat_out][nlon_out] (in/out)
        float *__restrict__ integral_buf,  // [batch][nheads][nlat_out][nlon_out] (in/out)
        typename vec_traits<STORAGE_T>::compute_t
            *__restrict__ alpha_k_buf, // [batch][nlat_out][nlon_out][nheads*nchan_in] (in/out)
        typename vec_traits<STORAGE_T>::compute_t
            *__restrict__ alpha_kvw_buf) // [batch][nlat_out][nlon_out][nheads*nchan_in] (in/out)
    {
        using COMPUTE_T = typename vec_traits<STORAGE_T>::compute_t;

        const int &nheads = p.nheads;
        const int &nchans_in = p.nchan_in;
        const int &nchans_out = p.nchan_out;
        const int &nlat_halo = p.nlat_halo;
        const int &nlon_kx = p.nlon_kx;
        const int &nlon_in = p.nlon_in;
        const int &pscale = p.pscale;
        const int &nlat_out = p.nlat_out;
        const int &nlon_out = p.nlon_out;

        extern __shared__ __align__(sizeof(float4)) float shext[];

        // sh_alpha_k__[nchan_in], sh_alpha_kvw[nchan_in], sh_dy[nchan_out], sh_qy[nchan_in]
        COMPUTE_T *sh_alpha_k__ = reinterpret_cast<COMPUTE_T *>(shext) + threadIdx.y * (nchans_in * 3 + nchans_out);
        COMPUTE_T *sh_alpha_kvw = sh_alpha_k__ + nchans_in;

        COMPUTE_T *sh_dy = sh_alpha_kvw + nchans_in;
        COMPUTE_T *sh_qy = sh_dy + nchans_out;

        const int bh = blockIdx.y;
        const int batch = bh / nheads;
        const int head = bh - (batch * nheads);

        const int64_t ldi = int64_t(nheads) * nchans_in;
        const int64_t ldo = int64_t(nheads) * nchans_out;

        const uint64_t wid = uint64_t(blockIdx.x) * blockDim.y + threadIdx.y;
        if (wid >= uint64_t(nlat_out) * nlon_out) { return; }

        const int tidx = threadIdx.x;

        // use permuted rows
        const int h = wid / nlon_out;
        const int wo = wid - (h * nlon_out); // LOCAL wo
        const int ho = row_idx[h];

        // offset input tensors
        kx += int64_t(batch) * nlat_halo * nlon_kx * ldi + int64_t(head) * nchans_in;
        qy += int64_t(batch) * nlat_out * nlon_out * ldi + int64_t(head) * nchans_in + int64_t(ho) * nlon_out * ldi
            + int64_t(wo) * ldi;

        vx += int64_t(batch) * nlat_halo * nlon_kx * ldo + int64_t(head) * nchans_out;
        dy += int64_t(batch) * nlat_out * nlon_out * ldo + int64_t(head) * nchans_out + int64_t(ho) * nlon_out * ldo
            + int64_t(wo) * ldo;

        // offset state (per-output vectors packed like qy; scalars per (batch, head, point))
        alpha_k_buf += int64_t(batch) * nlat_out * nlon_out * ldi + int64_t(head) * nchans_in
            + int64_t(ho) * nlon_out * ldi + int64_t(wo) * ldi;
        alpha_kvw_buf += int64_t(batch) * nlat_out * nlon_out * ldi + int64_t(head) * nchans_in
            + int64_t(ho) * nlon_out * ldi + int64_t(wo) * ldi;

        const int64_t stat_off = int64_t(bh) * nlat_out * nlon_out + int64_t(ho) * nlon_out + wo;
        alpha_sum_buf += stat_off;
        qdotk_max_buf += stat_off;
        integral_buf += stat_off;

        // resume the state where the previous ring step left it
        for (int chan = tidx; chan < nchans_in; chan += WARP_SIZE) {
            sh_alpha_k__[chan] = alpha_k_buf[chan];
            sh_alpha_kvw[chan] = alpha_kvw_buf[chan];

            sh_qy[chan] = vload(qy, chan);
        }
        for (int chan = tidx; chan < nchans_out; chan += WARP_SIZE) { sh_dy[chan] = vload(dy, chan); }

        float alpha_sum = alpha_sum_buf[0];
        float qdotk_max = qdotk_max_buf[0];
        float integral = integral_buf[0];

        const int seg_beg = seg_off[ho];
        const int seg_end = seg_off[ho + 1];

        // accumulate alpha_sum, integral, and shared stats,
        // along with a progressively computed qdotk_max.
        for (int sg = seg_beg; sg < seg_end; sg++) {

            const int hi = seg[3 * sg + 0];
            const int seg_lo = seg[3 * sg + 1];
            const int seg_len = seg[3 * sg + 2];

            // the chunk holds only the halo-padded latitudes of this polar rank
            const int hi_local = hi - p.lat_halo_start;
            if (hi_local < 0 || hi_local >= nlat_halo) { continue; }

            const float qw_seg = ring_weights[hi];

            const STORAGE_T *kx_row = kx + int64_t(hi_local) * nlon_kx * ldi;
            const STORAGE_T *vx_row = vx + int64_t(hi_local) * nlon_kx * ldo;

            // the part of the arc inside the chunk, as at most two contiguous pieces
            int2 piece[2];
            const int npiece
                = clip_arc(wrap_lon(seg_lo + pscale * wo, nlon_in), seg_len, nlon_in, p.lon_lo_kx, nlon_kx, piece);

            for (int pc = 0; pc < npiece; pc++) {

                int wip = piece[pc].x;

                for (int j = 0; j < piece[pc].y; j++) {

                    const STORAGE_T *_kx = kx_row + int64_t(wip) * ldi;
                    const STORAGE_T *_vx = vx_row + int64_t(wip) * ldo;

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
                    const float alpha_inz = expf(qdotk - qdotk_max_tmp) * qw_seg;
                    const float max_correction = expf(qdotk_max - qdotk_max_tmp);
                    alpha_sum = alpha_sum * max_correction + alpha_inz;

                    integral = integral * max_correction + alpha_inz * gdotv;

                    const float ainz_gdotv = alpha_inz * gdotv;

                    for (int chan = tidx; chan < nchans_in; chan += WARP_SIZE) {

                        const COMPUTE_T kxval = vload(_kx, chan);

                        sh_alpha_k__[chan]
                            = __vadd(__vscale(max_correction, sh_alpha_k__[chan]), __vscale(alpha_inz, kxval));
                        sh_alpha_kvw[chan]
                            = __vadd(__vscale(max_correction, sh_alpha_kvw[chan]), __vscale(ainz_gdotv, kxval));
                    }
                    qdotk_max = qdotk_max_tmp;

                    // pieces do not wrap: clip_arc split the arc at the seam
                    wip++;
                }
            }
        }

        // hand the state on to the next ring step; dqy is finalized after the last
        if (!tidx) {
            alpha_sum_buf[0] = alpha_sum;
            qdotk_max_buf[0] = qdotk_max;
            integral_buf[0] = integral;
        }
        for (int chan = tidx; chan < nchans_in; chan += WARP_SIZE) {
            alpha_k_buf[chan] = sh_alpha_k__[chan];
            alpha_kvw_buf[chan] = sh_alpha_kvw[chan];
        }

        return;
    }

    // see the first neighbour loop of s2_attn_bwd_special_vec_k
    template <int BDIM_X, int BDIM_Y,
              int CHOUT_AS_IN, // 1 iif "BDIM_X*(NLOC-1) <= nchan_out <= BDIM_X*NLOC" else 0
              int NLOC,        // smallest int such that BDIM_X*NLOC >= nchan_in
              typename STORAGE_T>
    __global__ __launch_bounds__(BDIM_X *BDIM_Y) void s2_attn_bwd_ring_pass1_special_vec_k(
        const __grid_constant__ attn_params_t p,
        const STORAGE_T *__restrict__ kx, // [batch][nlat_halo][nlon_kx][nheads*nchan_in]
        const STORAGE_T *__restrict__ vx, // [batch][nlat_halo][nlon_kx][nheads*nchan_out]
        const STORAGE_T *__restrict__ qy, // [batch][nlat_out][nlon_out][nheads*nchan_in]
        const STORAGE_T *__restrict__ dy, // [batch][nlat_out][nlon_out][nheads*nchan_out]
        const int32_t *__restrict__ row_idx, const int32_t *__restrict__ seg, const int32_t *__restrict__ seg_off,
        const float *__restrict__ ring_weights,
        float *__restrict__ alpha_sum_buf, // [batch][nheads][nlat_out][nlon_out] (in/out)
        float *__restrict__ qdotk_max_buf, // [batch][nheads][nlat_out][nlon_out] (in/out)
        float *__restrict__ integral_buf,  // [batch][nheads][nlat_out][nlon_out] (in/out)
        typename vec_traits<STORAGE_T>::compute_t
            *__restrict__ alpha_k_buf, // [batch][nlat_out][nlon_out][nheads*nchan_in] (in/out)
        typename vec_traits<STORAGE_T>::compute_t
            *__restrict__ alpha_kvw_buf) // [batch][nlat_out][nlon_out][nheads*nchan_in] (in/out)
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

        const uint64_t ctaid = uint64_t(blockIdx.x) * blockDim.y + threadIdx.y;

        if (ctaid >= uint64_t(nlat_out) * nlon_out) { return; }

        extern __shared__ __align__(sizeof(float4)) float shext[];

        // sh_dy[nchan_out], only read when the dy slice does not fit the register loop
        COMPUTE_T *sh_dy = reinterpret_cast<COMPUTE_T *>(shext) + threadIdx.y * nchan_out;

        if constexpr (CHOUT_AS_IN) { sh_dy += tidx; }

        // for dqy
        COMPUTE_T loc_k__[NLOC];
        COMPUTE_T loc_kvw[NLOC];

        // register copies of this thread's slice of qy / dy; see s2_attn_bwd_special_vec_k
        COMPUTE_T loc_qy[NLOC];
        COMPUTE_T loc_dy[NLOC];
#pragma unroll
        for (int i = 0; i < NLOC; i++) {
            loc_qy[i] = __vset<COMPUTE_T>(0.0f);
            loc_dy[i] = __vset<COMPUTE_T>(0.0f);
            loc_k__[i] = __vset<COMPUTE_T>(0.0f);
            loc_kvw[i] = __vset<COMPUTE_T>(0.0f);
        }

        // use permuted rows
        const int h = ctaid / nlon_out;
        const int wo = ctaid - (h * nlon_out); // LOCAL wo
        const int ho = row_idx[h];

        // offset input tensors
        kx += int64_t(batch) * nlat_halo * nlon_kx * ldi + int64_t(head) * nchan_in + tidx;
        qy += int64_t(batch) * nlat_out * nlon_out * ldi + int64_t(head) * nchan_in + int64_t(ho) * nlon_out * ldi
            + int64_t(wo) * ldi + tidx;

        vx += int64_t(batch) * nlat_halo * nlon_kx * ldo + int64_t(head) * nchan_out; // + tidx;
        dy += int64_t(batch) * nlat_out * nlon_out * ldo + int64_t(head) * nchan_out + int64_t(ho) * nlon_out * ldo
            + int64_t(wo) * ldo; // + tidx;
        if constexpr (CHOUT_AS_IN) {
            vx += tidx;
            dy += tidx;
        }

        // offset state (per-output vectors packed like qy; scalars per (batch, head, point))
        alpha_k_buf += int64_t(batch) * nlat_out * nlon_out * ldi + int64_t(head) * nchan_in
            + int64_t(ho) * nlon_out * ldi + int64_t(wo) * ldi + tidx;
        alpha_kvw_buf += int64_t(batch) * nlat_out * nlon_out * ldi + int64_t(head) * nchan_in
            + int64_t(ho) * nlon_out * ldi + int64_t(wo) * ldi + tidx;

        const int64_t stat_off = int64_t(bh) * nlat_out * nlon_out + int64_t(ho) * nlon_out + wo;
        alpha_sum_buf += stat_off;
        qdotk_max_buf += stat_off;
        integral_buf += stat_off;

        // resume the state where the previous ring step left it
#pragma unroll
        for (int i = 0; i < NLOC_M1; i++) {
            loc_qy[i] = vload(qy, i * BDIM_X);
            loc_k__[i] = alpha_k_buf[i * BDIM_X];
            loc_kvw[i] = alpha_kvw_buf[i * BDIM_X];
        }
        if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
            loc_qy[NLOC_M1] = vload(qy, NLOC_M1 * BDIM_X);
            loc_k__[NLOC_M1] = alpha_k_buf[NLOC_M1 * BDIM_X];
            loc_kvw[NLOC_M1] = alpha_kvw_buf[NLOC_M1 * BDIM_X];
        }

        if constexpr (CHOUT_AS_IN) {
#pragma unroll
            for (int i = 0; i < NLOC_M1; i++) { loc_dy[i] = vload(dy, i * BDIM_X); }
            if (NLOC_M1 * BDIM_X + tidx < nchan_out) { loc_dy[NLOC_M1] = vload(dy, NLOC_M1 * BDIM_X); }
        } else {
            for (int chan = tidx; chan < nchan_out; chan += BDIM_X) { sh_dy[chan] = vload(dy, chan); }
        }

        float alpha_sum = alpha_sum_buf[0];
        float qdotk_max = qdotk_max_buf[0];
        float integral = integral_buf[0];

        const int seg_beg = seg_off[ho];
        const int seg_end = seg_off[ho + 1];

        // accumulate alpha_sum, integral, and shared stats,
        // along with a progressively computed qdotk_max.
        for (int sg = seg_beg; sg < seg_end; sg++) {

            const int hi = seg[3 * sg + 0];
            const int seg_lo = seg[3 * sg + 1];
            const int seg_len = seg[3 * sg + 2];

            // the chunk holds only the halo-padded latitudes of this polar rank
            const int hi_local = hi - p.lat_halo_start;
            if (hi_local < 0 || hi_local >= nlat_halo) { continue; }

            const float qw_seg = ring_weights[hi];

            const STORAGE_T *kx_row = kx + int64_t(hi_local) * nlon_kx * ldi;
            const STORAGE_T *vx_row = vx + int64_t(hi_local) * nlon_kx * ldo;

            // the part of the arc inside the chunk, as at most two contiguous pieces
            int2 piece[2];
            const int npiece
                = clip_arc(wrap_lon(seg_lo + pscale * wo, nlon_in), seg_len, nlon_in, p.lon_lo_kx, nlon_kx, piece);

            for (int pc = 0; pc < npiece; pc++) {

                int wip = piece[pc].x;

                for (int j = 0; j < piece[pc].y; j++) {

                    const STORAGE_T *_kx = kx_row + int64_t(wip) * ldi;
                    const STORAGE_T *_vx = vx_row + int64_t(wip) * ldo;

                    COMPUTE_T qdotk_v = __vset<COMPUTE_T>(0.0f);
                    COMPUTE_T gdotv_v = __vset<COMPUTE_T>(0.0f);

#pragma unroll
                    for (int i = 0; i < NLOC_M1; i++) {
                        qdotk_v = __vadd(qdotk_v, __vmul(loc_qy[i], vload(_kx, i * BDIM_X)));
                    }
                    if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                        qdotk_v = __vadd(qdotk_v, __vmul(loc_qy[NLOC_M1], vload(_kx, NLOC_M1 * BDIM_X)));
                    }
                    if constexpr (CHOUT_AS_IN) {
#pragma unroll
                        for (int i = 0; i < NLOC_M1; i++) {
                            gdotv_v = __vadd(gdotv_v, __vmul(loc_dy[i], vload(_vx, i * BDIM_X)));
                        }
                        if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
                            gdotv_v = __vadd(gdotv_v, __vmul(loc_dy[NLOC_M1], vload(_vx, NLOC_M1 * BDIM_X)));
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
                    const float alpha_inz = expf(qdotk - qdotk_max_tmp) * qw_seg;
                    const float max_correction = expf(qdotk_max - qdotk_max_tmp);

                    alpha_sum = alpha_sum * max_correction + alpha_inz;
                    integral = integral * max_correction + alpha_inz * gdotv;

                    const float ainz_gdotv = alpha_inz * gdotv;

#pragma unroll
                    for (int i = 0; i < NLOC_M1; i++) {
                        const COMPUTE_T kxval = vload(_kx, i * BDIM_X);
                        loc_k__[i] = __vadd(__vscale(max_correction, loc_k__[i]), __vscale(alpha_inz, kxval));
                        loc_kvw[i] = __vadd(__vscale(max_correction, loc_kvw[i]), __vscale(ainz_gdotv, kxval));
                    }
                    if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                        const COMPUTE_T kxval = vload(_kx, NLOC_M1 * BDIM_X);
                        loc_k__[NLOC_M1] = __vadd(__vscale(max_correction, loc_k__[NLOC_M1]), __vscale(alpha_inz, kxval));
                        loc_kvw[NLOC_M1]
                            = __vadd(__vscale(max_correction, loc_kvw[NLOC_M1]), __vscale(ainz_gdotv, kxval));
                    }

                    qdotk_max = qdotk_max_tmp;

                    // pieces do not wrap: clip_arc split the arc at the seam
                    wip++;
                }
            }
        }

        // hand the state on to the next ring step; dqy is finalized after the last
        if (!tidx) {
            alpha_sum_buf[0] = alpha_sum;
            qdotk_max_buf[0] = qdotk_max;
            integral_buf[0] = integral;
        }

#pragma unroll
        for (int i = 0; i < NLOC_M1; i++) {
            alpha_k_buf[i * BDIM_X] = loc_k__[i];
            alpha_kvw_buf[i * BDIM_X] = loc_kvw[i];
        }
        if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
            alpha_k_buf[NLOC_M1 * BDIM_X] = loc_k__[NLOC_M1];
            alpha_kvw_buf[NLOC_M1 * BDIM_X] = loc_kvw[NLOC_M1];
        }

        return;
    }

    ///////////////// END PASS1 SECTION

    ///////////////// BEGIN PASS2 SECTION

    // see the second neighbour loop of s2_attn_bwd_generic_vec_k
    template <int BDIM_X, typename STORAGE_T>
    __global__ __launch_bounds__(BDIM_X) void s2_attn_bwd_ring_pass2_generic_vec_k(
        const __grid_constant__ attn_params_t p,
        const STORAGE_T *__restrict__ kx, // [batch][nlat_halo][nlon_kx][nheads*nchan_in]
        const STORAGE_T *__restrict__ vx, // [batch][nlat_halo][nlon_kx][nheads*nchan_out]
        const STORAGE_T *__restrict__ qy, // [batch][nlat_out][nlon_out][nheads*nchan_in]
        const STORAGE_T *__restrict__ dy, // [batch][nlat_out][nlon_out][nheads*nchan_out]
        const int32_t *__restrict__ row_idx, const int32_t *__restrict__ seg, const int32_t *__restrict__ seg_off,
        const float *__restrict__ ring_weights,
        const float *__restrict__ alpha_sum_buf,     // finalized [batch][nheads][nlat_out][nlon_out]
        const float *__restrict__ qdotk_max_buf,     // finalized [batch][nheads][nlat_out][nlon_out]
        const float *__restrict__ integral_norm_buf, // finalized, normalized [batch][nheads][nlat_out][nlon_out]
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dkx, // [batch][nlat_halo][nlon_kx][nheads*nchan_in]
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dvx) // [batch][nlat_halo][nlon_kx][nheads*nchan_out]
    {
        using COMPUTE_T = typename vec_traits<STORAGE_T>::compute_t;

        const int &nheads = p.nheads;
        const int &nchans_in = p.nchan_in;
        const int &nchans_out = p.nchan_out;
        const int &nlat_halo = p.nlat_halo;
        const int &nlon_kx = p.nlon_kx;
        const int &nlon_in = p.nlon_in;
        const int &pscale = p.pscale;
        const int &nlat_out = p.nlat_out;
        const int &nlon_out = p.nlon_out;

        extern __shared__ __align__(sizeof(float4)) float shext[];

        // sh_dy[nchan_out], sh_qy[nchan_in]
        COMPUTE_T *sh_dy = reinterpret_cast<COMPUTE_T *>(shext) + threadIdx.y * (nchans_in + nchans_out);
        COMPUTE_T *sh_qy = sh_dy + nchans_out;

        const int bh = blockIdx.y;
        const int batch = bh / nheads;
        const int head = bh - (batch * nheads);

        const int64_t ldi = int64_t(nheads) * nchans_in;
        const int64_t ldo = int64_t(nheads) * nchans_out;

        const uint64_t wid = uint64_t(blockIdx.x) * blockDim.y + threadIdx.y;
        if (wid >= uint64_t(nlat_out) * nlon_out) { return; }

        const int tidx = threadIdx.x;

        // use permuted rows
        const int h = wid / nlon_out;
        const int wo = wid - (h * nlon_out); // LOCAL wo
        const int ho = row_idx[h];

        // offset input tensors
        kx += int64_t(batch) * nlat_halo * nlon_kx * ldi + int64_t(head) * nchans_in;
        qy += int64_t(batch) * nlat_out * nlon_out * ldi + int64_t(head) * nchans_in + int64_t(ho) * nlon_out * ldi
            + int64_t(wo) * ldi;

        vx += int64_t(batch) * nlat_halo * nlon_kx * ldo + int64_t(head) * nchans_out;
        dy += int64_t(batch) * nlat_out * nlon_out * ldo + int64_t(head) * nchans_out + int64_t(ho) * nlon_out * ldo
            + int64_t(wo) * ldo;

        // offset output tensors (same packed layout as their inputs)
        dkx += int64_t(batch) * nlat_halo * nlon_kx * ldi + int64_t(head) * nchans_in;
        dvx += int64_t(batch) * nlat_halo * nlon_kx * ldo + int64_t(head) * nchans_out;

        for (int chan = tidx; chan < nchans_in; chan += WARP_SIZE) { sh_qy[chan] = vload(qy, chan); }
        for (int chan = tidx; chan < nchans_out; chan += WARP_SIZE) { sh_dy[chan] = vload(dy, chan); }

#if __CUDA_ARCH__ < 900
        // see s2_attn_bwd_generic_vec_k: sh_dy and sh_qy are read as individual floats
        // below, which breaks the one-thread-per-COMPUTE_T-slot assumption for float4
        if constexpr (std::is_same<COMPUTE_T, float4>::value) { __syncwarp(); }
#endif

        // the statistics are final: every chunk has been seen by pass1
        const int64_t stat_off = int64_t(bh) * nlat_out * nlon_out + int64_t(ho) * nlon_out + wo;
        const float alpha_sum_inv = 1.0f / alpha_sum_buf[stat_off];
        const float qdotk_max = qdotk_max_buf[stat_off];
        const float integral = integral_norm_buf[stat_off];

        const int seg_beg = seg_off[ho];
        const int seg_end = seg_off[ho + 1];

        // accumulate gradients for k and v
        for (int sg = seg_beg; sg < seg_end; sg++) {

            const int hi = seg[3 * sg + 0];
            const int seg_lo = seg[3 * sg + 1];
            const int seg_len = seg[3 * sg + 2];

            // the chunk holds only the halo-padded latitudes of this polar rank
            const int hi_local = hi - p.lat_halo_start;
            if (hi_local < 0 || hi_local >= nlat_halo) { continue; }

            const float qw_seg = ring_weights[hi];

            const STORAGE_T *kx_row = kx + int64_t(hi_local) * nlon_kx * ldi;
            const STORAGE_T *vx_row = vx + int64_t(hi_local) * nlon_kx * ldo;
            COMPUTE_T *dkx_row = dkx + int64_t(hi_local) * nlon_kx * ldi;
            COMPUTE_T *dvx_row = dvx + int64_t(hi_local) * nlon_kx * ldo;

            // the part of the arc inside the chunk, as at most two contiguous pieces
            int2 piece[2];
            const int npiece
                = clip_arc(wrap_lon(seg_lo + pscale * wo, nlon_in), seg_len, nlon_in, p.lon_lo_kx, nlon_kx, piece);

            for (int pc = 0; pc < npiece; pc++) {

                int wip = piece[pc].x;

                for (int j = 0; j < piece[pc].y; j++) {

                    const STORAGE_T *_kx = kx_row + int64_t(wip) * ldi;
                    const STORAGE_T *_vx = vx_row + int64_t(wip) * ldo;

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

                    const float alpha_inz = expf(qdotk - qdotk_max) * qw_seg;

                    // _dkx / _dvx are COMPUTE_T (fp32) gradient buffers, accumulated atomically.
                    COMPUTE_T *_dkx = dkx_row + int64_t(wip) * ldi;
                    COMPUTE_T *_dvx = dvx_row + int64_t(wip) * ldo;

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
                    // pieces do not wrap: clip_arc split the arc at the seam
                    wip++;
                }
            }
        }

        return;
    }

    // see the second neighbour loop of s2_attn_bwd_special_vec_k
    template <int BDIM_X, int BDIM_Y,
              int CHOUT_AS_IN, // 1 iif "BDIM_X*(NLOC-1) <= nchan_out <= BDIM_X*NLOC" else 0
              int NLOC,        // smallest int such that BDIM_X*NLOC >= nchan_in
              typename STORAGE_T>
    __global__ __launch_bounds__(BDIM_X *BDIM_Y) void s2_attn_bwd_ring_pass2_special_vec_k(
        const __grid_constant__ attn_params_t p,
        const STORAGE_T *__restrict__ kx, // [batch][nlat_halo][nlon_kx][nheads*nchan_in]
        const STORAGE_T *__restrict__ vx, // [batch][nlat_halo][nlon_kx][nheads*nchan_out]
        const STORAGE_T *__restrict__ qy, // [batch][nlat_out][nlon_out][nheads*nchan_in]
        const STORAGE_T *__restrict__ dy, // [batch][nlat_out][nlon_out][nheads*nchan_out]
        const int32_t *__restrict__ row_idx, const int32_t *__restrict__ seg, const int32_t *__restrict__ seg_off,
        const float *__restrict__ ring_weights,
        const float *__restrict__ alpha_sum_buf,     // finalized [batch][nheads][nlat_out][nlon_out]
        const float *__restrict__ qdotk_max_buf,     // finalized [batch][nheads][nlat_out][nlon_out]
        const float *__restrict__ integral_norm_buf, // finalized, normalized [batch][nheads][nlat_out][nlon_out]
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dkx, // [batch][nlat_halo][nlon_kx][nheads*nchan_in]
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dvx) // [batch][nlat_halo][nlon_kx][nheads*nchan_out]
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

        const uint64_t ctaid = uint64_t(blockIdx.x) * blockDim.y + threadIdx.y;

        if (ctaid >= uint64_t(nlat_out) * nlon_out) { return; }

        extern __shared__ __align__(sizeof(float4)) float shext[];

        // sh_dy[nchan_out], sh_qy[nchan_in]
        COMPUTE_T *sh_dy = reinterpret_cast<COMPUTE_T *>(shext) + threadIdx.y * (nchan_in + nchan_out); // + tidx;
        COMPUTE_T *sh_qy = sh_dy + nchan_out + tidx;

        if constexpr (CHOUT_AS_IN) { sh_dy += tidx; }

        // register copies of this thread's slice of qy / dy; see s2_attn_bwd_special_vec_k
        COMPUTE_T loc_qy[NLOC];
        COMPUTE_T loc_dy[NLOC];
#pragma unroll
        for (int i = 0; i < NLOC; i++) {
            loc_qy[i] = __vset<COMPUTE_T>(0.0f);
            loc_dy[i] = __vset<COMPUTE_T>(0.0f);
        }

        // use permuted rows
        const int h = ctaid / nlon_out;
        const int wo = ctaid - (h * nlon_out); // LOCAL wo
        const int ho = row_idx[h];

        // offset input tensors
        kx += int64_t(batch) * nlat_halo * nlon_kx * ldi + int64_t(head) * nchan_in + tidx;
        qy += int64_t(batch) * nlat_out * nlon_out * ldi + int64_t(head) * nchan_in + int64_t(ho) * nlon_out * ldi
            + int64_t(wo) * ldi + tidx;

        vx += int64_t(batch) * nlat_halo * nlon_kx * ldo + int64_t(head) * nchan_out; // + tidx;
        dy += int64_t(batch) * nlat_out * nlon_out * ldo + int64_t(head) * nchan_out + int64_t(ho) * nlon_out * ldo
            + int64_t(wo) * ldo; // + tidx;
        if constexpr (CHOUT_AS_IN) {
            vx += tidx;
            dy += tidx;
        }

        // offset output tensors
        dkx += int64_t(batch) * nlat_halo * nlon_kx * ldi + int64_t(head) * nchan_in + tidx;
        dvx += int64_t(batch) * nlat_halo * nlon_kx * ldo + int64_t(head) * nchan_out; // + tidx;
        if constexpr (CHOUT_AS_IN) { dvx += tidx; }

#pragma unroll
        for (int i = 0; i < NLOC_M1; i++) {
            loc_qy[i] = vload(qy, i * BDIM_X);
            sh_qy[i * BDIM_X] = loc_qy[i];
        }
        if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
            loc_qy[NLOC_M1] = vload(qy, NLOC_M1 * BDIM_X);
            sh_qy[NLOC_M1 * BDIM_X] = loc_qy[NLOC_M1];
        }

        if constexpr (CHOUT_AS_IN) {
#pragma unroll
            for (int i = 0; i < NLOC_M1; i++) {
                loc_dy[i] = vload(dy, i * BDIM_X);
                sh_dy[i * BDIM_X] = loc_dy[i];
            }
            if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
                loc_dy[NLOC_M1] = vload(dy, NLOC_M1 * BDIM_X);
                sh_dy[NLOC_M1 * BDIM_X] = loc_dy[NLOC_M1];
            }
        } else {
            for (int chan = tidx; chan < nchan_out; chan += BDIM_X) { sh_dy[chan] = vload(dy, chan); }
        }

#if __CUDA_ARCH__ < 900
        // see s2_attn_bwd_special_vec_k: sh_dy and sh_qy are read as individual floats
        // below, which breaks the one-thread-per-COMPUTE_T-slot assumption for float4
        if constexpr (std::is_same<COMPUTE_T, float4>::value) {
            if constexpr (BDIM_X == 32) {
                __syncwarp();
            } else {
                __syncthreads();
            }
        }
#endif

        // the statistics are final: every chunk has been seen by pass1
        const int64_t stat_off = int64_t(bh) * nlat_out * nlon_out + int64_t(ho) * nlon_out + wo;
        const float alpha_sum_inv = 1.0f / alpha_sum_buf[stat_off];
        const float qdotk_max = qdotk_max_buf[stat_off];
        const float integral = integral_norm_buf[stat_off];

        const int seg_beg = seg_off[ho];
        const int seg_end = seg_off[ho + 1];

        // accumulate gradients for k and v
        for (int sg = seg_beg; sg < seg_end; sg++) {

            const int hi = seg[3 * sg + 0];
            const int seg_lo = seg[3 * sg + 1];
            const int seg_len = seg[3 * sg + 2];

            // the chunk holds only the halo-padded latitudes of this polar rank
            const int hi_local = hi - p.lat_halo_start;
            if (hi_local < 0 || hi_local >= nlat_halo) { continue; }

            const float qw_seg = ring_weights[hi];

            const STORAGE_T *kx_row = kx + int64_t(hi_local) * nlon_kx * ldi;
            const STORAGE_T *vx_row = vx + int64_t(hi_local) * nlon_kx * ldo;
            COMPUTE_T *dkx_row = dkx + int64_t(hi_local) * nlon_kx * ldi;
            COMPUTE_T *dvx_row = dvx + int64_t(hi_local) * nlon_kx * ldo;

            // the part of the arc inside the chunk, as at most two contiguous pieces
            int2 piece[2];
            const int npiece
                = clip_arc(wrap_lon(seg_lo + pscale * wo, nlon_in), seg_len, nlon_in, p.lon_lo_kx, nlon_kx, piece);

            for (int pc = 0; pc < npiece; pc++) {

                int wip = piece[pc].x;

                for (int j = 0; j < piece[pc].y; j++) {

                    const STORAGE_T *_kx = kx_row + int64_t(wip) * ldi;
                    const STORAGE_T *_vx = vx_row + int64_t(wip) * ldo;

                    COMPUTE_T qdotk_v = __vset<COMPUTE_T>(0.0f);
                    COMPUTE_T gdotv_v = __vset<COMPUTE_T>(0.0f);

#pragma unroll
                    for (int i = 0; i < NLOC_M1; i++) {
                        qdotk_v = __vadd(qdotk_v, __vmul(loc_qy[i], vload(_kx, i * BDIM_X)));
                    }
                    if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                        qdotk_v = __vadd(qdotk_v, __vmul(loc_qy[NLOC_M1], vload(_kx, NLOC_M1 * BDIM_X)));
                    }
                    if constexpr (CHOUT_AS_IN) {
#pragma unroll
                        for (int i = 0; i < NLOC_M1; i++) {
                            gdotv_v = __vadd(gdotv_v, __vmul(loc_dy[i], vload(_vx, i * BDIM_X)));
                        }
                        if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
                            gdotv_v = __vadd(gdotv_v, __vmul(loc_dy[NLOC_M1], vload(_vx, NLOC_M1 * BDIM_X)));
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

                    const float alpha_inz = expf(qdotk - qdotk_max) * qw_seg;

                    COMPUTE_T *_dkx = dkx_row + int64_t(wip) * ldi;
                    COMPUTE_T *_dvx = dvx_row + int64_t(wip) * ldo;

                    const float alpha_mul = alpha_inz * alpha_sum_inv;

                    const float scale_fact_qy = (gdotv - integral) * alpha_mul;
                    const float scale_fact_dy = alpha_mul;

                    // float4, 128-bit atomics are only supported by devices of compute
                    // capability 9.x+, so on older devices we resort to 32-bit atomics

#if __CUDA_ARCH__ < 900
                    constexpr int VEC_SIZE = sizeof(COMPUTE_T) / sizeof(float);

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
                        atomicAdd(_dkx + i * BDIM_X, __vscale(scale_fact_qy, loc_qy[i]));
                    }
                    if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                        atomicAdd(_dkx + NLOC_M1 * BDIM_X, __vscale(scale_fact_qy, loc_qy[NLOC_M1]));
                    }
                    if constexpr (CHOUT_AS_IN) {
#pragma unroll
                        for (int i = 0; i < NLOC_M1; i++) {
                            atomicAdd(_dvx + i * BDIM_X, __vscale(scale_fact_dy, loc_dy[i]));
                        }
                        if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
                            atomicAdd(_dvx + NLOC_M1 * BDIM_X, __vscale(scale_fact_dy, loc_dy[NLOC_M1]));
                        }
                    } else {
                        for (int chan = tidx; chan < nchan_out; chan += BDIM_X) {
                            atomicAdd(_dvx + chan, __vscale(scale_fact_dy, sh_dy[chan]));
                        }
                    }
#endif
                    // pieces do not wrap: clip_arc split the arc at the seam
                    wip++;
                }
            }
        }

        return;
    }

    ///////////////// END PASS2 SECTION

    ///////////////// BEGIN LAUNCHERS

    // Both passes share the serial backward's launch configuration: bdimx from
    // nchan_in, the special kernels up to BDIM_X=512 (1024 spills), the generic one
    // above. PASS picks the kernel pair; the state pointers are passed through
    // untouched, which is why they are typed by the pass.
    template <int PASS, typename SV> struct bwd_ring_state_t;
    template <typename SV> struct bwd_ring_state_t<1, SV> {
        float *alpha_sum, *qdotk_max, *integral;
        typename vec_traits<SV>::compute_t *alpha_k, *alpha_kvw;
    };
    template <typename SV> struct bwd_ring_state_t<2, SV> {
        const float *alpha_sum, *qdotk_max, *integral_norm;
        typename vec_traits<SV>::compute_t *dkx, *dvx;
    };

    template <int PASS, typename SV>
    void launch_gen_attn_ring_bwd(int batch_size, const attn_params_t &params, SV *_kxp, SV *_vxp, SV *_qyp, SV *_dyp,
                                  int32_t *_row_idx, int32_t *_seg, int32_t *_seg_off, float *_quad_weights,
                                  const bwd_ring_state_t<PASS, SV> &st, cudaStream_t stream)
    {
        using COMPUTE_T = typename vec_traits<SV>::compute_t;

        dim3 block(WARP_SIZE, THREADS / WARP_SIZE);
        dim3 grid(DIV_UP(params.nlat_out * params.nlon_out, block.y), batch_size * params.nheads);

        if constexpr (PASS == 1) {
            // shared memory holds compute-type (COMPUTE_T) data, not STORAGE_T. 4 arrays per warp.
            size_t shsize = sizeof(COMPUTE_T) * (params.nchan_in * 3 + params.nchan_out) * block.y;
            launch_dyn_shmem(&s2_attn_bwd_ring_pass1_generic_vec_k<THREADS, SV>, grid, block, shsize, stream, params,
                             _kxp, _vxp, _qyp, _dyp, _row_idx, _seg, _seg_off, _quad_weights, st.alpha_sum,
                             st.qdotk_max, st.integral, st.alpha_k, st.alpha_kvw);
            CHECK_ERROR("s2_attn_bwd_ring_pass1_generic_vec_k");
        } else {
            // 2 arrays per warp
            size_t shsize = sizeof(COMPUTE_T) * (params.nchan_in + params.nchan_out) * block.y;
            launch_dyn_shmem(&s2_attn_bwd_ring_pass2_generic_vec_k<THREADS, SV>, grid, block, shsize, stream, params,
                             _kxp, _vxp, _qyp, _dyp, _row_idx, _seg, _seg_off, _quad_weights, st.alpha_sum,
                             st.qdotk_max, st.integral_norm, st.dkx, st.dvx);
            CHECK_ERROR("s2_attn_bwd_ring_pass2_generic_vec_k");
        }

        return;
    }

    template <int PASS, int BDIM_X, int BDIM_Y, int CUR_LOC_SIZE,
              int MAX_LOC_SIZE, // max size of COMPUTE_T[] local array
              typename SV>
    void launch_spc_attn_ring_bwd(int nloc, // "BDIM_X*nloc" >= nchans_in
                                  int batch_size, const attn_params_t &params, SV *_kxp, SV *_vxp, SV *_qyp, SV *_dyp,
                                  int32_t *_row_idx, int32_t *_seg, int32_t *_seg_off, float *_quad_weights,
                                  const bwd_ring_state_t<PASS, SV> &st, cudaStream_t stream)
    {
        using COMPUTE_T = typename vec_traits<SV>::compute_t;

        if (CUR_LOC_SIZE == nloc) {

            dim3 block(BDIM_X, BDIM_Y);
            dim3 grid(DIV_UP(params.nlat_out * params.nlon_out, block.y), batch_size * params.nheads);

            // see launch_spc_attn_bwd
            const bool chout_as_in
                = params.nchan_out >= BDIM_X * (CUR_LOC_SIZE - 1) && params.nchan_out <= BDIM_X * CUR_LOC_SIZE;

            if constexpr (PASS == 1) {
                // pass1 keeps qy in registers and needs no shared copy of it; dy is
                // staged in shared only when it does not fit the register loop
                size_t shsize = sizeof(COMPUTE_T) * params.nchan_out * block.y;
                if (chout_as_in) {
                    launch_dyn_shmem(&s2_attn_bwd_ring_pass1_special_vec_k<BDIM_X, BDIM_Y, 1, CUR_LOC_SIZE, SV>, grid,
                                     block, shsize, stream, params, _kxp, _vxp, _qyp, _dyp, _row_idx, _seg, _seg_off,
                                     _quad_weights, st.alpha_sum, st.qdotk_max, st.integral, st.alpha_k, st.alpha_kvw);
                } else {
                    launch_dyn_shmem(&s2_attn_bwd_ring_pass1_special_vec_k<BDIM_X, BDIM_Y, 0, CUR_LOC_SIZE, SV>, grid,
                                     block, shsize, stream, params, _kxp, _vxp, _qyp, _dyp, _row_idx, _seg, _seg_off,
                                     _quad_weights, st.alpha_sum, st.qdotk_max, st.integral, st.alpha_k, st.alpha_kvw);
                }
                CHECK_ERROR("s2_attn_bwd_ring_pass1_special_vec_k");
            } else {
                // 2 arrays per cta, block.y > 1 iif block.x==32
                size_t shsize = sizeof(COMPUTE_T) * (params.nchan_in + params.nchan_out) * block.y;
                if (chout_as_in) {
                    launch_dyn_shmem(&s2_attn_bwd_ring_pass2_special_vec_k<BDIM_X, BDIM_Y, 1, CUR_LOC_SIZE, SV>, grid,
                                     block, shsize, stream, params, _kxp, _vxp, _qyp, _dyp, _row_idx, _seg, _seg_off,
                                     _quad_weights, st.alpha_sum, st.qdotk_max, st.integral_norm, st.dkx, st.dvx);
                } else {
                    launch_dyn_shmem(&s2_attn_bwd_ring_pass2_special_vec_k<BDIM_X, BDIM_Y, 0, CUR_LOC_SIZE, SV>, grid,
                                     block, shsize, stream, params, _kxp, _vxp, _qyp, _dyp, _row_idx, _seg, _seg_off,
                                     _quad_weights, st.alpha_sum, st.qdotk_max, st.integral_norm, st.dkx, st.dvx);
                }
                CHECK_ERROR("s2_attn_bwd_ring_pass2_special_vec_k");
            }

            return;
        }
        if constexpr (CUR_LOC_SIZE < MAX_LOC_SIZE) {
            launch_spc_attn_ring_bwd<PASS, BDIM_X, BDIM_Y, CUR_LOC_SIZE + 1, MAX_LOC_SIZE>(
                nloc, batch_size, params, _kxp, _vxp, _qyp, _dyp, _row_idx, _seg, _seg_off, _quad_weights, st, stream);
        }
        return;
    }

    // see bwd_dispatch_bdimx
    template <int PASS, int MAX_LOC, int MIN_LOC, typename SV>
    static void bwd_ring_dispatch_bdimx(int bdimx, int nloc, int64_t batch_size, const attn_params_t &params, SV *_kxp,
                                        SV *_vxp, SV *_qyp, SV *_dyp, int32_t *_row_idx, int32_t *_seg, int32_t *_seg_off,
                                        float *_quad_weights, const bwd_ring_state_t<PASS, SV> &st, cudaStream_t stream)
    {
        switch (bdimx) {
        case 32:
            launch_spc_attn_ring_bwd<PASS, 32, 2, 1, MAX_LOC>(nloc, batch_size, params, _kxp, _vxp, _qyp, _dyp,
                                                              _row_idx, _seg, _seg_off, _quad_weights, st, stream);
            break;
        case 64:
            launch_spc_attn_ring_bwd<PASS, 64, 1, MIN_LOC, MAX_LOC>(nloc, batch_size, params, _kxp, _vxp, _qyp, _dyp,
                                                                    _row_idx, _seg, _seg_off, _quad_weights, st, stream);
            break;
        case 128:
            launch_spc_attn_ring_bwd<PASS, 128, 1, MIN_LOC, MAX_LOC>(
                nloc, batch_size, params, _kxp, _vxp, _qyp, _dyp, _row_idx, _seg, _seg_off, _quad_weights, st, stream);
            break;
        case 256:
            launch_spc_attn_ring_bwd<PASS, 256, 1, MIN_LOC, MAX_LOC>(
                nloc, batch_size, params, _kxp, _vxp, _qyp, _dyp, _row_idx, _seg, _seg_off, _quad_weights, st, stream);
            break;
        case 512:
            launch_spc_attn_ring_bwd<PASS, 512, 1, MIN_LOC, MAX_LOC>(
                nloc, batch_size, params, _kxp, _vxp, _qyp, _dyp, _row_idx, _seg, _seg_off, _quad_weights, st, stream);
            break;
        default:
            launch_gen_attn_ring_bwd<PASS>(batch_size, params, _kxp, _vxp, _qyp, _dyp, _row_idx, _seg, _seg_off,
                                           _quad_weights, st, stream);
            break;
        }
    }

    // see s2_attn_bwd_dispatch; the path selection is the same. `state` holds the
    // pass's five buffers in schema order.
    template <int PASS, typename scalar_t>
    static void s2_attn_bwd_ring_dispatch(int64_t batch_size, attn_params_t params, at::Tensor kxP, at::Tensor vxP,
                                          at::Tensor qyP, at::Tensor dyP, at::Tensor row_off, at::Tensor seg,
                                          at::Tensor seg_off, at::Tensor ring_weights, std::array<at::Tensor, 5> state)
    {

        static_assert(0 == (MAX_LOCAL_ARR_LEN & (MAX_LOCAL_ARR_LEN - 1)));

        // get stream
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        // sort row indices (ho-s) in descending order
        // based on (row_off[ho+1]-row_off[ho])
        at::Tensor row_idx = sortRows(params.nlat_out, row_off, stream);

        const int64_t nchans_in = params.nchan_in;
        const int64_t nchans_out = params.nchan_out;

        // smallest power of two "bdimx" (>=32) s.t. bdimx*MAX_LOCAL_ARR_LEN >= nchans_in
        int bdimx;
        bdimx = DIV_UP(nchans_in, MAX_LOCAL_ARR_LEN);
        bdimx = max(bdimx, WARP_SIZE);
        bdimx = next_pow2(bdimx);

        scalar_t *_kxp = reinterpret_cast<scalar_t *>(kxP.data_ptr());
        scalar_t *_vxp = reinterpret_cast<scalar_t *>(vxP.data_ptr());
        scalar_t *_qyp = reinterpret_cast<scalar_t *>(qyP.data_ptr());
        scalar_t *_dyp = reinterpret_cast<scalar_t *>(dyP.data_ptr());

        int32_t *_row_idx = reinterpret_cast<int32_t *>(row_idx.data_ptr());
        int32_t *_seg = reinterpret_cast<int32_t *>(seg.data_ptr());
        int32_t *_seg_off = reinterpret_cast<int32_t *>(seg_off.data_ptr());
        float *_quad_weights = reinterpret_cast<float *>(ring_weights.data_ptr());

        // the state is fp32 whatever the activations are: three scalars per point, then
        // two per-channel buffers (alpha_k / alpha_kvw in pass1, dkx / dvx in pass2)
        float *_s[5];
        for (int i = 0; i < 5; i++) { _s[i] = reinterpret_cast<float *>(state[i].data_ptr()); }

        const auto make_state = [&](auto vec_tag) {
            using V = decltype(vec_tag);
            using C = typename vec_traits<V>::compute_t;
            return bwd_ring_state_t<PASS, V> {_s[0], _s[1], _s[2], reinterpret_cast<C *>(_s[3]),
                                              reinterpret_cast<C *>(_s[4])};
        };

        constexpr int MIN_LOC_ARR_LEN = MAX_LOCAL_ARR_LEN / 2 + 1;

        if constexpr (std::is_same<scalar_t, float>::value) {
            // fp32: float4 vectorized when 16B-aligned + 4-divisible, else scalar.
            constexpr int VEC_SIZE = sizeof(float4) / sizeof(float); // 4
            const bool use_vec = is_aligned<16>(_kxp) && is_aligned<16>(_vxp) && is_aligned<16>(_qyp)
                && is_aligned<16>(_dyp) && is_aligned<16>(_s[3]) && is_aligned<16>(_s[4]) && (nchans_in % VEC_SIZE) == 0
                && (nchans_out % VEC_SIZE) == 0;

            if (use_vec) {
                constexpr int MAX_VEC = MAX_LOCAL_ARR_LEN / VEC_SIZE;
                constexpr int MIN_VEC = MAX_VEC / 2 + 1;
                params.nchan_in = nchans_in / VEC_SIZE;
                params.nchan_out = nchans_out / VEC_SIZE;
                bwd_ring_dispatch_bdimx<PASS, MAX_VEC, MIN_VEC, float4>(
                    bdimx, DIV_UP(params.nchan_in, bdimx), batch_size, params, reinterpret_cast<float4 *>(_kxp),
                    reinterpret_cast<float4 *>(_vxp), reinterpret_cast<float4 *>(_qyp), reinterpret_cast<float4 *>(_dyp),
                    _row_idx, _seg, _seg_off, _quad_weights, make_state(float4 {}), stream);
            } else {
                bwd_ring_dispatch_bdimx<PASS, MAX_LOCAL_ARR_LEN, MIN_LOC_ARR_LEN, float>(
                    bdimx, DIV_UP(nchans_in, bdimx), batch_size, params, _kxp, _vxp, _qyp, _dyp, _row_idx, _seg,
                    _seg_off, _quad_weights, make_state(float {}), stream);
            }
        } else {
            // fp16/bf16: scalar STORAGE_T inputs, fp32 state; fp32 compute/accumulation.
            bwd_ring_dispatch_bdimx<PASS, MAX_LOCAL_ARR_LEN, MIN_LOC_ARR_LEN, scalar_t>(
                bdimx, DIV_UP(nchans_in, bdimx), batch_size, params, _kxp, _vxp, _qyp, _dyp, _row_idx, _seg, _seg_off,
                _quad_weights, make_state(scalar_t {}), stream);
        }

        return;
    }

    ///////////////// END LAUNCHERS

    // Host entry shared by both passes: the checks, the row offsets for sortRows and
    // the params are identical, only the state and the kernel pair differ.
    template <int PASS>
    static void s2_attention_bwd_ring_step_cuda(at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor dy,
                                                std::array<at::Tensor, 5> state, at::Tensor ring_weights,
                                                at::Tensor psi_seg, at::Tensor psi_seg_off, int64_t num_heads,
                                                int64_t nlon_in, int64_t nlon_out_global, int64_t lon_lo_kx,
                                                int64_t lat_halo_start, int64_t nlat_out, int64_t nlon_out)
    {
        CHECK_CUDA_INPUT_TENSOR(kx);
        CHECK_CUDA_INPUT_TENSOR(vx);
        CHECK_CUDA_INPUT_TENSOR(qy);
        CHECK_CUDA_INPUT_TENSOR(dy);
        for (auto &t : state) { CHECK_CUDA_TENSOR(t); }
        CHECK_CUDA_TENSOR(ring_weights);

        // run on the inputs' device: without this, the current stream, the scratch
        // allocations and the per-device queries (ensure_dyn_shmem, getPtxver) would all
        // resolve to whichever CUDA device happens to be current
        const at::cuda::OptionalCUDAGuard device_guard(kx.device());
        // devices, shapes, index and weight dtypes, dense layouts; the arc rows are this
        // rank's output latitudes
        check_ring_step_inputs(kx, vx, qy, ring_weights, psi_seg, psi_seg_off, num_heads, nlat_out, nlon_out, nlat_out);
        check_output_grad(dy, kx, vx, qy);
        // state, in schema order: alpha_sum, qdotk_max, then integral / alpha_k / alpha_kvw
        // (pass 1) or integral_norm / dkx / dvx (pass 2), the gradients shaped like the chunk
        check_state_buffer(state[0], kx, "alpha_sum_buf", {kx.size(0), num_heads, nlat_out, nlon_out});
        check_state_buffer(state[1], kx, "qdotk_max_buf", {kx.size(0), num_heads, nlat_out, nlon_out});
        check_state_buffer(state[2], kx, PASS == 1 ? "integral_buf" : "integral_norm_buf",
                           {kx.size(0), num_heads, nlat_out, nlon_out});
        if constexpr (PASS == 1) {
            check_state_buffer(state[3], kx, "alpha_k_buf", {kx.size(0), nlat_out, nlon_out, qy.size(3)});
            check_state_buffer(state[4], kx, "alpha_kvw_buf", {kx.size(0), nlat_out, nlon_out, qy.size(3)});
        } else {
            check_state_buffer(state[3], kx, "dkx", kx.sizes());
            check_state_buffer(state[4], kx, "dvx", vx.sizes());
        }

        // row offsets from the arcs, for sortRows -- see s2_attention_bwd_dkvq_cuda
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

        // Native storage: kx/vx/qy/dy keep their dtype and are widened at load; the
        // state and the gradient buffers are allocated fp32 in Python and stay fp32
        // (dkx/dvx are atomically scatter-accumulated).
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, qy.scalar_type(), "s2_attention_bwd_ring_step_cuda", [&] {
            s2_attn_bwd_ring_dispatch<PASS, scalar_t>(batch_size, params, kx, vx, qy, dy, psi_row_off, psi_seg,
                                                      psi_seg_off, ring_weights, state);
        });

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    void s2_attention_bwd_ring_step_pass1_cuda(at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor dy,
                                               at::Tensor alpha_sum_buf, at::Tensor qdotk_max_buf,
                                               at::Tensor integral_buf, at::Tensor alpha_k_buf,
                                               at::Tensor alpha_kvw_buf, at::Tensor ring_weights, at::Tensor psi_seg,
                                               at::Tensor psi_seg_off, int64_t num_heads, int64_t nlon_in,
                                               int64_t nlon_out_global, int64_t lon_lo_kx, int64_t lat_halo_start,
                                               int64_t nlat_out, int64_t nlon_out)
    {
        s2_attention_bwd_ring_step_cuda<1>(
            kx, vx, qy, dy, {alpha_sum_buf, qdotk_max_buf, integral_buf, alpha_k_buf, alpha_kvw_buf}, ring_weights,
            psi_seg, psi_seg_off, num_heads, nlon_in, nlon_out_global, lon_lo_kx, lat_halo_start, nlat_out, nlon_out);
    }

    void s2_attention_bwd_ring_step_pass2_cuda(at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor dy,
                                               at::Tensor alpha_sum_buf, at::Tensor qdotk_max_buf,
                                               at::Tensor integral_norm_buf, at::Tensor dkx, at::Tensor dvx,
                                               at::Tensor ring_weights, at::Tensor psi_seg, at::Tensor psi_seg_off,
                                               int64_t num_heads, int64_t nlon_in, int64_t nlon_out_global,
                                               int64_t lon_lo_kx, int64_t lat_halo_start, int64_t nlat_out,
                                               int64_t nlon_out)
    {
        s2_attention_bwd_ring_step_cuda<2>(kx, vx, qy, dy, {alpha_sum_buf, qdotk_max_buf, integral_norm_buf, dkx, dvx},
                                           ring_weights, psi_seg, psi_seg_off, num_heads, nlon_in, nlon_out_global,
                                           lon_lo_kx, lat_halo_start, nlat_out, nlon_out);
    }

    TORCH_LIBRARY_IMPL(attention_kernels, CUDA, m)
    {
        m.impl("backward_ring_step_pass1", &s2_attention_bwd_ring_step_pass1_cuda);
        m.impl("backward_ring_step_pass2", &s2_attention_bwd_ring_step_pass2_cuda);
    }

} // namespace attention_kernels
