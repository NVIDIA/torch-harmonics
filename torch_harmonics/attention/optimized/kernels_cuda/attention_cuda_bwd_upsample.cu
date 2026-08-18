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

// =====================================================================================
// Upsample (scatter-style) attention backward — CUDA
// =====================================================================================
//
// Mirrors the structure of the downsample backward (see attention_cuda_bwd.cu),
// just with the output-centric inverse-shift logic in the inner scan:
//   - psi rows are indexed by hi; each output (b, ho, wo) scans every psi[hi]
//     row, skipping entries where ho_neigh != ho or where the residue test
//     (wo - wo_canonical) mod pscale_out != 0 fails. For surviving entries,
//     wi = (wo - wo_canonical) / pscale_out.
//
// Single kernel computes dqy, dkx, dvx for one output cell:
//   pass 1 — online softmax over contributors; accumulates per-channel
//            shared-memory state (sh_alpha_k__, sh_alpha_vw_, sh_alpha_kvw)
//            and scalar alpha_sum, integral, qdotk_max. After the scan, the
//            warp writes dqy[b, ho, wo, :] = (alpha_kvw * alpha_sum - alpha_vw *
//            alpha_k) / alpha_sum^2.
//   pass 2 — scan again with the finalized softmax stats; for each contributor
//            atomicAdd into dkx[b, hi, wi, :] += qy * (gdotv - integral) *
//            alpha_norm and dvx[b, hi, wi, :] += dy * alpha_norm. atomics are
//            required because many output cells can scatter into the same
//            input cell (one (hi, wi) is reachable from multiple (ho, wo)
//            via the residue map).
// Generic-only for now; no specialized channel-size variant. Rows ARE sorted
// (sortRows) and the kernels walk psi's arc segments.
// load-balancing (correctness path, not perf path).
// =====================================================================================

#include "attention_cuda.cuh"
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>
#include <cfloat>

#include "cudamacro.h"
#include "attention_cuda_utils.cuh"

#define THREADS (64)

namespace attention_kernels
{

    // =====================================================================================
    // SCATTER (input-keyed) backward — the adjoint of the scatter forward. One warp per
    // COARSE INPUT cell (hi, wi) walks ONLY its real psi row, so the O(out_cells * nnz)
    // scan of the old output-centric kernel (15-78x slower) is replaced by O(nnz) work.
    //
    //   dkx/dvx live at the coarse cell (= the warp's row key), so they accumulate
    //   locally over the row and are written directly — NO atomics (gather-fwd-like).
    //
    //   dqy + the softmax stats live at the fine output cell (NOT the row key), so they
    //   are cross-warp reductions done with atomics, mirroring the scatter forward:
    //     pass 1 (max):   scatter q.k -> atomicMax m[b,ho,wo]
    //     pass 2 (stats): scatter alpha -> S, alpha*g -> Avw, alpha*k -> Ak[:],
    //                     alpha*g*k -> Akvw[:]   (g = dy.v ; alpha = exp(q.k-m)*w)
    //     finalize dq:    dqy = (S*Akvw - Avw*Ak) / S^2     (per fine cell)
    //     pass 3 (dkdv):  per coarse cell, local accumulate dvx += (alpha/S)*dy and
    //                     dkx += q*(g - Avw/S)*(alpha/S) over the row; direct write.
    // All reduction buffers (m/S/Avw/Ak/Akvw) and the grads are fp32; Ak/Akvw are
    // [b, ho, wo, nchan_in] — the memory price of keeping psi input-keyed.
    // =====================================================================================

    __device__ __forceinline__ float bwd_atomicMaxf(float *addr, float val)
    {
        int *ai = reinterpret_cast<int *>(addr);
        int old = *ai;
        while (val > __int_as_float(old)) {
            const int assumed = old;
            old = atomicCAS(ai, assumed, __float_as_int(val));
            if (old == assumed) { break; }
        }
        return __int_as_float(old);
    }

    __device__ __forceinline__ int bwd_scatter_wo(int wo_canonical, int wi, int pscale_out, int nlon_out)
    {
        int wo = wo_canonical + pscale_out * wi; // < 2*nlon_out
        if (wo >= nlon_out) { wo -= nlon_out; }
        return wo;
    }

    // pass 1: per coarse cell, scatter q.k into the per-output-cell running max.
    template <int THREADS_PER_BLOCK, typename STORAGE_T>
    __global__ __launch_bounds__(THREADS_PER_BLOCK) void s2_attn_bwd_upsample_scatter_max_k(
        int nheads, int nchan_in, int nlat_in, int nlon_in, int nlat_out, int nlon_out,
        const STORAGE_T *__restrict__ kx, const STORAGE_T *__restrict__ qy, const int32_t *__restrict__ row_idx,
        const int32_t *__restrict__ seg, const int32_t *__restrict__ seg_off, float *__restrict__ maxbuf)
    {
        extern __shared__ float shext[];
        float *sh_k = shext + threadIdx.y * nchan_in;

        // gridDim.y spans batch * nheads. User tensors are packed
        // (B, nlat, nlon, nheads*nchan) so they use ldi/ldo plus a head offset;
        // the reduction scratch allocated by this launcher keeps (batch, head) as
        // its leading index instead.
        const int bh = blockIdx.y;
        const int batch = bh / nheads;
        const int head = bh - (batch * nheads);

        const int64_t ldi = int64_t(nheads) * nchan_in;

        const int wid = blockIdx.x * blockDim.y + threadIdx.y;
        if (wid >= nlat_in * nlon_in) { return; }
        const int tidx = threadIdx.x;
        // Rows visited longest-first. psi's density is heavily skewed with latitude -- a
        // polar row's neighborhood spans the whole circle -- and every wi within a
        // latitude does identical work, so in natural order entire latitudes of blocks
        // straggle at the end. ncu measured 29.95% achieved occupancy against 50%
        // theoretical, which is what that imbalance looks like. The gather path has sorted
        // rows for exactly this reason; this one never did.
        const int hi_c = wid / nlon_in;
        const int hi = row_idx[hi_c];
        const int wi = wid - hi_c * nlon_in;
        const int pscale_out = nlon_out / nlon_in;

        kx += int64_t(batch) * nlat_in * nlon_in * ldi + int64_t(head) * nchan_in + (int64_t(hi) * nlon_in + wi) * ldi;
        qy += int64_t(batch) * nlat_out * nlon_out * ldi + int64_t(head) * nchan_in;
        maxbuf += int64_t(bh) * nlat_out * nlon_out;

        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) { sh_k[chan] = vload(kx, chan); }

        // Arc segments instead of the flat column list: a neighbor's (output lat, output
        // lon) is derived by counting along a contiguous arc, so the 64-bit division that
        // decoded col / nlon_out per neighbor is gone. The GPU has no integer-divide
        // instruction, so each cost ~70-100 emulated ones.
        const int seg_beg = seg_off[hi];
        const int seg_end = seg_off[hi + 1];

        for (int sg = seg_beg; sg < seg_end; sg++) {

            const int ho = seg[3 * sg + 0];
            const int seg_lo = seg[3 * sg + 1];
            const int seg_len = seg[3 * sg + 2];

            // one shift at the arc start, then a counted walk: no modulus inside
            int wo = bwd_scatter_wo(seg_lo, wi, pscale_out, nlon_out);

            for (int j = 0; j < seg_len; j++) {
                const STORAGE_T *_qy = qy + (int64_t(ho) * nlon_out + wo) * ldi;

                float qd = 0.f;
                for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) { qd += sh_k[chan] * vload(_qy, chan); }
                qd = __warp_sum(qd);
                if (tidx == 0) { bwd_atomicMaxf(&maxbuf[int64_t(ho) * nlon_out + wo], qd); }
                if (++wo == nlon_out) { wo = 0; }
            }
        }
    }

    // pass 2: per coarse cell, scatter the softmax stats (S, Avw, Ak, Akvw) to fine cells.
    template <int THREADS_PER_BLOCK, typename STORAGE_T>
    __global__ __launch_bounds__(THREADS_PER_BLOCK) void s2_attn_bwd_upsample_scatter_stats_k(
        int nheads, int nchan_in, int nchan_out, int nlat_in, int nlon_in, int nlat_out, int nlon_out,
        const STORAGE_T *__restrict__ kx, const STORAGE_T *__restrict__ vx, const STORAGE_T *__restrict__ qy,
        const STORAGE_T *__restrict__ dy, const int32_t *__restrict__ row_idx, const int32_t *__restrict__ seg,
        const int32_t *__restrict__ seg_off, const float *__restrict__ quad_weights, const float *__restrict__ maxbuf,
        float *__restrict__ S, float *__restrict__ Avw, float *__restrict__ Ak, float *__restrict__ Akvw)
    {
        extern __shared__ float shext[];
        float *sh_k = shext + threadIdx.y * (nchan_in + nchan_out);
        float *sh_v = sh_k + nchan_in;

        // gridDim.y spans batch * nheads. User tensors are packed
        // (B, nlat, nlon, nheads*nchan) so they use ldi/ldo plus a head offset;
        // the reduction scratch allocated by this launcher keeps (batch, head) as
        // its leading index instead.
        const int bh = blockIdx.y;
        const int batch = bh / nheads;
        const int head = bh - (batch * nheads);

        const int64_t ldi = int64_t(nheads) * nchan_in;
        const int64_t ldo = int64_t(nheads) * nchan_out;

        const int wid = blockIdx.x * blockDim.y + threadIdx.y;
        if (wid >= nlat_in * nlon_in) { return; }
        const int tidx = threadIdx.x;
        // Rows visited longest-first. psi's density is heavily skewed with latitude -- a
        // polar row's neighborhood spans the whole circle -- and every wi within a
        // latitude does identical work, so in natural order entire latitudes of blocks
        // straggle at the end. ncu measured 29.95% achieved occupancy against 50%
        // theoretical, which is what that imbalance looks like. The gather path has sorted
        // rows for exactly this reason; this one never did.
        const int hi_c = wid / nlon_in;
        const int hi = row_idx[hi_c];
        const int wi = wid - hi_c * nlon_in;
        const int pscale_out = nlon_out / nlon_in;

        kx += int64_t(batch) * nlat_in * nlon_in * ldi + int64_t(head) * nchan_in + (int64_t(hi) * nlon_in + wi) * ldi;
        vx += int64_t(batch) * nlat_in * nlon_in * ldo + int64_t(head) * nchan_out + (int64_t(hi) * nlon_in + wi) * ldo;
        qy += int64_t(batch) * nlat_out * nlon_out * ldi + int64_t(head) * nchan_in;
        dy += int64_t(batch) * nlat_out * nlon_out * ldo + int64_t(head) * nchan_out;
        maxbuf += int64_t(bh) * nlat_out * nlon_out;
        S += int64_t(bh) * nlat_out * nlon_out;
        Avw += int64_t(bh) * nlat_out * nlon_out;
        Ak += int64_t(bh) * nlat_out * nlon_out * nchan_in;
        Akvw += int64_t(bh) * nlat_out * nlon_out * nchan_in;

        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) { sh_k[chan] = vload(kx, chan); }
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { sh_v[chan] = vload(vx, chan); }

        const float qw = quad_weights[hi];
        // Arc segments instead of the flat column list: a neighbor's (output lat, output
        // lon) is derived by counting along a contiguous arc, so the 64-bit division that
        // decoded col / nlon_out per neighbor is gone. The GPU has no integer-divide
        // instruction, so each cost ~70-100 emulated ones.
        const int seg_beg = seg_off[hi];
        const int seg_end = seg_off[hi + 1];

        for (int sg = seg_beg; sg < seg_end; sg++) {

            const int ho = seg[3 * sg + 0];
            const int seg_lo = seg[3 * sg + 1];
            const int seg_len = seg[3 * sg + 2];

            // one shift at the arc start, then a counted walk: no modulus inside
            int wo = bwd_scatter_wo(seg_lo, wi, pscale_out, nlon_out);

            for (int j = 0; j < seg_len; j++) {
                const int64_t cell = int64_t(ho) * nlon_out + wo;
                const STORAGE_T *_qy = qy + cell * ldi;
                const STORAGE_T *_dy = dy + cell * ldo;

                float qd = 0.f, gd = 0.f;
                for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) { qd += sh_k[chan] * vload(_qy, chan); }
                for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { gd += sh_v[chan] * vload(_dy, chan); }
                qd = __warp_sum(qd);
                gd = __warp_sum(gd);

                const float alpha = expf(qd - maxbuf[cell]) * qw;
                const float ag = alpha * gd;
                if (tidx == 0) {
                    atomicAdd(&S[cell], alpha);
                    atomicAdd(&Avw[cell], ag);
                }
                float *_Ak = Ak + cell * nchan_in;
                float *_Akvw = Akvw + cell * nchan_in;
                for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
                    atomicAdd(&_Ak[chan], alpha * sh_k[chan]);
                    atomicAdd(&_Akvw[chan], ag * sh_k[chan]);
                }
                if (++wo == nlon_out) { wo = 0; }
            }
        }
    }

    // finalize: dqy = (S*Akvw - Avw*Ak) / S^2 (one warp per fine output cell).
    template <int THREADS_PER_BLOCK>
    __global__ __launch_bounds__(THREADS_PER_BLOCK) void s2_attn_bwd_upsample_scatter_dq_k(
        int nheads, int nchan_in, int nlat_out, int nlon_out, const float *__restrict__ S, const float *__restrict__ Avw,
        const float *__restrict__ Ak, const float *__restrict__ Akvw, float *__restrict__ dqy)
    {
        const int bh = blockIdx.y;
        const int batch = bh / nheads;
        const int head = bh - (batch * nheads);

        const int64_t ldi = int64_t(nheads) * nchan_in;

        const int wid = blockIdx.x * blockDim.y + threadIdx.y;
        if (wid >= nlat_out * nlon_out) { return; }
        const int tidx = threadIdx.x;

        // scratch is indexed by (batch, head); dqy is packed along channels
        const int64_t scell = int64_t(bh) * nlat_out * nlon_out + wid;
        const float s = S[scell];
        const float s_inv = 1.0f / s;
        const float avw = Avw[scell];
        const float *_Ak = Ak + scell * nchan_in;
        const float *_Akvw = Akvw + scell * nchan_in;
        float *_dqy = dqy + (int64_t(batch) * nlat_out * nlon_out + wid) * ldi + int64_t(head) * nchan_in;
        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
            _dqy[chan] = s_inv * s_inv * (s * _Akvw[chan] - avw * _Ak[chan]);
        }
    }

    // pass 3: per coarse cell, accumulate dkx/dvx locally over its row (no atomics).
    template <int THREADS_PER_BLOCK, typename STORAGE_T>
    __global__ __launch_bounds__(THREADS_PER_BLOCK) void s2_attn_bwd_upsample_scatter_dkv_k(
        int nheads, int nchan_in, int nchan_out, int nlat_in, int nlon_in, int nlat_out, int nlon_out,
        const STORAGE_T *__restrict__ kx, const STORAGE_T *__restrict__ vx, const STORAGE_T *__restrict__ qy,
        const STORAGE_T *__restrict__ dy, const int32_t *__restrict__ row_idx, const int32_t *__restrict__ seg,
        const int32_t *__restrict__ seg_off, const float *__restrict__ quad_weights, const float *__restrict__ maxbuf,
        const float *__restrict__ S, const float *__restrict__ Avw, float *__restrict__ dkx, float *__restrict__ dvx)
    {
        extern __shared__ float shext[];
        float *sh_k = shext + threadIdx.y * (2 * nchan_in + 2 * nchan_out);
        float *sh_v = sh_k + nchan_in;
        float *sh_dk = sh_v + nchan_out;
        float *sh_dv = sh_dk + nchan_in;

        const int bh = blockIdx.y;
        const int batch = bh / nheads;
        const int head = bh - (batch * nheads);

        const int64_t ldi = int64_t(nheads) * nchan_in;
        const int64_t ldo = int64_t(nheads) * nchan_out;

        const int wid = blockIdx.x * blockDim.y + threadIdx.y;
        if (wid >= nlat_in * nlon_in) { return; }
        const int tidx = threadIdx.x;
        // Rows visited longest-first. psi's density is heavily skewed with latitude -- a
        // polar row's neighborhood spans the whole circle -- and every wi within a
        // latitude does identical work, so in natural order entire latitudes of blocks
        // straggle at the end. ncu measured 29.95% achieved occupancy against 50%
        // theoretical, which is what that imbalance looks like. The gather path has sorted
        // rows for exactly this reason; this one never did.
        const int hi_c = wid / nlon_in;
        const int hi = row_idx[hi_c];
        const int wi = wid - hi_c * nlon_in;
        const int pscale_out = nlon_out / nlon_in;

        kx += int64_t(batch) * nlat_in * nlon_in * ldi + int64_t(head) * nchan_in + (int64_t(hi) * nlon_in + wi) * ldi;
        vx += int64_t(batch) * nlat_in * nlon_in * ldo + int64_t(head) * nchan_out + (int64_t(hi) * nlon_in + wi) * ldo;
        qy += int64_t(batch) * nlat_out * nlon_out * ldi + int64_t(head) * nchan_in;
        dy += int64_t(batch) * nlat_out * nlon_out * ldo + int64_t(head) * nchan_out;
        maxbuf += int64_t(bh) * nlat_out * nlon_out;
        S += int64_t(bh) * nlat_out * nlon_out;
        Avw += int64_t(bh) * nlat_out * nlon_out;
        dkx += int64_t(batch) * nlat_in * nlon_in * ldi + int64_t(head) * nchan_in + (int64_t(hi) * nlon_in + wi) * ldi;
        dvx += int64_t(batch) * nlat_in * nlon_in * ldo + int64_t(head) * nchan_out + (int64_t(hi) * nlon_in + wi) * ldo;

        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
            sh_k[chan] = vload(kx, chan);
            sh_dk[chan] = 0.f;
        }
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
            sh_v[chan] = vload(vx, chan);
            sh_dv[chan] = 0.f;
        }

        const float qw = quad_weights[hi];
        // Arc segments instead of the flat column list: a neighbor's (output lat, output
        // lon) is derived by counting along a contiguous arc, so the 64-bit division that
        // decoded col / nlon_out per neighbor is gone. The GPU has no integer-divide
        // instruction, so each cost ~70-100 emulated ones.
        const int seg_beg = seg_off[hi];
        const int seg_end = seg_off[hi + 1];

        for (int sg = seg_beg; sg < seg_end; sg++) {

            const int ho = seg[3 * sg + 0];
            const int seg_lo = seg[3 * sg + 1];
            const int seg_len = seg[3 * sg + 2];

            // one shift at the arc start, then a counted walk: no modulus inside
            int wo = bwd_scatter_wo(seg_lo, wi, pscale_out, nlon_out);

            for (int j = 0; j < seg_len; j++) {
                const int64_t cell = int64_t(ho) * nlon_out + wo;
                const STORAGE_T *_qy = qy + cell * ldi;
                const STORAGE_T *_dy = dy + cell * ldo;

                float qd = 0.f, gd = 0.f;
                for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) { qd += sh_k[chan] * vload(_qy, chan); }
                for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { gd += sh_v[chan] * vload(_dy, chan); }
                qd = __warp_sum(qd);
                gd = __warp_sum(gd);

                const float s = S[cell];
                const float alpha_mul = expf(qd - maxbuf[cell]) * qw / s;
                const float integral = Avw[cell] / s;
                const float scale_dk = (gd - integral) * alpha_mul;

                for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
                    sh_dv[chan] += alpha_mul * vload(_dy, chan);
                }
                for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
                    sh_dk[chan] += scale_dk * vload(_qy, chan);
                }
                if (++wo == nlon_out) { wo = 0; }
            }
        }

        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) { dkx[chan] = sh_dk[chan]; }
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { dvx[chan] = sh_dv[chan]; }
    }

    // host launcher for the scatter backward (allocates the fp32 reduction buffers,
    // runs the four passes). STORAGE_T deduces from the activation pointers.
    template <typename STORAGE_T>
    static void launch_attn_bwd_upsample_scatter(int batch_size, int nheads, int nchans_in, int nchans_out, int nlat_in,
                                                 int nlon_in, int nlat_out, int nlon_out, STORAGE_T *_kxp,
                                                 STORAGE_T *_vxp, STORAGE_T *_qyp, STORAGE_T *_dyp, int32_t *_row_idx,
                                                 int32_t *_seg, int32_t *_seg_off, float *_quad_weights, float *_dkxp,
                                                 float *_dvxp, float *_dqyp, cudaStream_t stream)
    {
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        // reduction scratch carries (batch, head) as its leading index -- private to
        // this launcher, so it need not match the packed activation layout
        const int bh_size = batch_size * nheads;

        torch::Tensor S = torch::zeros({bh_size, nlat_out, nlon_out}, opts);
        torch::Tensor Avw = torch::zeros({bh_size, nlat_out, nlon_out}, opts);
        torch::Tensor maxbuf = torch::full({bh_size, nlat_out, nlon_out}, -FLT_MAX, opts);
        torch::Tensor Ak = torch::zeros({bh_size, nlat_out, nlon_out, nchans_in}, opts);
        torch::Tensor Akvw = torch::zeros({bh_size, nlat_out, nlon_out, nchans_in}, opts);

        float *_S = reinterpret_cast<float *>(S.data_ptr());
        float *_Avw = reinterpret_cast<float *>(Avw.data_ptr());
        float *_maxbuf = reinterpret_cast<float *>(maxbuf.data_ptr());
        float *_Ak = reinterpret_cast<float *>(Ak.data_ptr());
        float *_Akvw = reinterpret_cast<float *>(Akvw.data_ptr());

        dim3 block(WARP_SIZE, THREADS / WARP_SIZE);
        dim3 grid_in(DIV_UP(nlat_in * nlon_in, block.y), bh_size);
        dim3 grid_out(DIV_UP(nlat_out * nlon_out, block.y), bh_size);

        const size_t sh_max = sizeof(float) * nchans_in * block.y;
        const size_t sh_stats = sizeof(float) * (nchans_in + nchans_out) * block.y;
        const size_t sh_dkv = sizeof(float) * (2 * nchans_in + 2 * nchans_out) * block.y;

        s2_attn_bwd_upsample_scatter_max_k<THREADS><<<grid_in, block, sh_max, stream>>>(
            nheads, nchans_in, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _qyp, _row_idx, _seg, _seg_off, _maxbuf);
        CHECK_ERROR("s2_attn_bwd_upsample_scatter_max_k");

        s2_attn_bwd_upsample_scatter_stats_k<THREADS><<<grid_in, block, sh_stats, stream>>>(
            nheads, nchans_in, nchans_out, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _seg,
            _seg_off, _quad_weights, _maxbuf, _S, _Avw, _Ak, _Akvw);
        CHECK_ERROR("s2_attn_bwd_upsample_scatter_stats_k");

        s2_attn_bwd_upsample_scatter_dq_k<THREADS>
            <<<grid_out, block, 0, stream>>>(nheads, nchans_in, nlat_out, nlon_out, _S, _Avw, _Ak, _Akvw, _dqyp);
        CHECK_ERROR("s2_attn_bwd_upsample_scatter_dq_k");

        s2_attn_bwd_upsample_scatter_dkv_k<THREADS><<<grid_in, block, sh_dkv, stream>>>(
            nheads, nchans_in, nchans_out, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _seg,
            _seg_off, _quad_weights, _maxbuf, _S, _Avw, _dkxp, _dvxp);
        CHECK_ERROR("s2_attn_bwd_upsample_scatter_dkv_k");
    }

    // -----------------------------------------------------------------------------
    // host dispatcher — called from s2_attention_bwd_dkvq_cuda when the direction is
    // upsample (nlon_out % nlon_in == 0). Native-storage (Tier B): kept NON-templated
    // (called from a different TU, attention_cuda_bwd.cu) and does its own AT_DISPATCH
    // over the activation dtype, then routes to the input-keyed scatter backward
    // (scalar path for every dtype; activations widen at load, fp32 compute). The
    // gradient tensors dkx/dvx/dqy are always fp32 — same as the gather backward.
    // -----------------------------------------------------------------------------
    void s2_attn_bwd_upsample_dispatch(int batch_size, int64_t num_heads, size_t nchans_in, size_t nchans_out,
                                       int64_t nlon_in, int64_t nlat_in, int64_t nlat_out, int64_t nlon_out,
                                       torch::Tensor kxP, torch::Tensor vxP, torch::Tensor qyP, torch::Tensor dyP,
                                       torch::Tensor psi_row_off, torch::Tensor psi_seg, torch::Tensor psi_seg_off,
                                       torch::Tensor quad_weights, torch::Tensor dkxP, torch::Tensor dvxP,
                                       torch::Tensor dqyP)
    {

        auto stream = at::cuda::getCurrentCUDAStream().stream();

        // gradient outputs are always fp32
        float *_dkxp = reinterpret_cast<float *>(dkxP.data_ptr());
        float *_dvxp = reinterpret_cast<float *>(dvxP.data_ptr());
        float *_dqyp = reinterpret_cast<float *>(dqyP.data_ptr());

        // Compacted-row -> input-latitude map, sorted longest-first. psi's rows here are
        // keyed by INPUT latitude (the scatter psi), so this balances across latitudes;
        // every wi within a latitude does identical work. row_off survives only to feed
        // this -- the kernels read seg / seg_off instead.
        at::Tensor row_idx = sortRows(static_cast<int>(nlat_in), psi_row_off, stream);
        int32_t *_row_idx = reinterpret_cast<int32_t *>(row_idx.data_ptr());
        int32_t *_seg = reinterpret_cast<int32_t *>(psi_seg.data_ptr());
        int32_t *_seg_off = reinterpret_cast<int32_t *>(psi_seg_off.data_ptr());
        float *_quad_weights = reinterpret_cast<float *>(quad_weights.data_ptr());

        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, qyP.scalar_type(), "s2_attn_bwd_upsample", [&] {
            scalar_t *_kxp = reinterpret_cast<scalar_t *>(kxP.data_ptr());
            scalar_t *_vxp = reinterpret_cast<scalar_t *>(vxP.data_ptr());
            scalar_t *_qyp = reinterpret_cast<scalar_t *>(qyP.data_ptr());
            scalar_t *_dyp = reinterpret_cast<scalar_t *>(dyP.data_ptr());

            launch_attn_bwd_upsample_scatter(batch_size, static_cast<int>(num_heads), static_cast<int>(nchans_in),
                                             static_cast<int>(nchans_out), static_cast<int>(nlat_in),
                                             static_cast<int>(nlon_in), static_cast<int>(nlat_out),
                                             static_cast<int>(nlon_out), _kxp, _vxp, _qyp, _dyp, _row_idx, _seg,
                                             _seg_off, _quad_weights, _dkxp, _dvxp, _dqyp, stream);
        });

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

} // namespace attention_kernels
