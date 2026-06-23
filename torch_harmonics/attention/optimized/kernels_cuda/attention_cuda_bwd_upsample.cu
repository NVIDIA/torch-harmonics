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
// Generic-only for now; no specialized channel-size variant or sortRows
// load-balancing (correctness path, not perf path).
// =====================================================================================

#include "attention_cuda.cuh"
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAUtils.h>

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
        int nchan_in, int nlat_in, int nlon_in, int nlat_out, int nlon_out, const STORAGE_T *__restrict__ kx,
        const STORAGE_T *__restrict__ qy, const int64_t *__restrict__ row_off, const int64_t *__restrict__ col_idx,
        float *__restrict__ maxbuf)
    {
        extern __shared__ float shext[];
        float *sh_k = shext + threadIdx.y * nchan_in;

        const int batch = blockIdx.y;
        const int wid = blockIdx.x * blockDim.y + threadIdx.y;
        if (wid >= nlat_in * nlon_in) { return; }
        const int tidx = threadIdx.x;
        const int hi = wid / nlon_in;
        const int wi = wid - hi * nlon_in;
        const int pscale_out = nlon_out / nlon_in;

        kx += int64_t(batch) * nlat_in * nlon_in * nchan_in + (int64_t(hi) * nlon_in + wi) * nchan_in;
        qy += int64_t(batch) * nlat_out * nlon_out * nchan_in;
        maxbuf += int64_t(batch) * nlat_out * nlon_out;

        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) { sh_k[chan] = vload(kx, chan); }

        const int64_t rbeg = row_off[hi];
        const int rlen = static_cast<int>(row_off[hi + 1] - rbeg);
        const int64_t *col_hi = col_idx + rbeg;

        for (int off = 0; off < rlen; off++) {
            const int64_t col = col_hi[off];
            const int ho = static_cast<int>(col / nlon_out);
            const int wo = bwd_scatter_wo(static_cast<int>(col - int64_t(ho) * nlon_out), wi, pscale_out, nlon_out);
            const STORAGE_T *_qy = qy + (int64_t(ho) * nlon_out + wo) * nchan_in;

            float qd = 0.f;
            for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) { qd += sh_k[chan] * vload(_qy, chan); }
            qd = __warp_sum(qd);
            if (tidx == 0) { bwd_atomicMaxf(&maxbuf[int64_t(ho) * nlon_out + wo], qd); }
        }
    }

    // pass 2: per coarse cell, scatter the softmax stats (S, Avw, Ak, Akvw) to fine cells.
    template <int THREADS_PER_BLOCK, typename STORAGE_T>
    __global__ __launch_bounds__(THREADS_PER_BLOCK) void s2_attn_bwd_upsample_scatter_stats_k(
        int nchan_in, int nchan_out, int nlat_in, int nlon_in, int nlat_out, int nlon_out,
        const STORAGE_T *__restrict__ kx, const STORAGE_T *__restrict__ vx, const STORAGE_T *__restrict__ qy,
        const STORAGE_T *__restrict__ dy, const int64_t *__restrict__ row_off, const int64_t *__restrict__ col_idx,
        const float *__restrict__ quad_weights, const float *__restrict__ maxbuf, float *__restrict__ S,
        float *__restrict__ Avw, float *__restrict__ Ak, float *__restrict__ Akvw)
    {
        extern __shared__ float shext[];
        float *sh_k = shext + threadIdx.y * (nchan_in + nchan_out);
        float *sh_v = sh_k + nchan_in;

        const int batch = blockIdx.y;
        const int wid = blockIdx.x * blockDim.y + threadIdx.y;
        if (wid >= nlat_in * nlon_in) { return; }
        const int tidx = threadIdx.x;
        const int hi = wid / nlon_in;
        const int wi = wid - hi * nlon_in;
        const int pscale_out = nlon_out / nlon_in;

        kx += int64_t(batch) * nlat_in * nlon_in * nchan_in + (int64_t(hi) * nlon_in + wi) * nchan_in;
        vx += int64_t(batch) * nlat_in * nlon_in * nchan_out + (int64_t(hi) * nlon_in + wi) * nchan_out;
        qy += int64_t(batch) * nlat_out * nlon_out * nchan_in;
        dy += int64_t(batch) * nlat_out * nlon_out * nchan_out;
        maxbuf += int64_t(batch) * nlat_out * nlon_out;
        S += int64_t(batch) * nlat_out * nlon_out;
        Avw += int64_t(batch) * nlat_out * nlon_out;
        Ak += int64_t(batch) * nlat_out * nlon_out * nchan_in;
        Akvw += int64_t(batch) * nlat_out * nlon_out * nchan_in;

        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) { sh_k[chan] = vload(kx, chan); }
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { sh_v[chan] = vload(vx, chan); }

        const float qw = quad_weights[hi];
        const int64_t rbeg = row_off[hi];
        const int rlen = static_cast<int>(row_off[hi + 1] - rbeg);
        const int64_t *col_hi = col_idx + rbeg;

        for (int off = 0; off < rlen; off++) {
            const int64_t col = col_hi[off];
            const int ho = static_cast<int>(col / nlon_out);
            const int wo = bwd_scatter_wo(static_cast<int>(col - int64_t(ho) * nlon_out), wi, pscale_out, nlon_out);
            const int64_t cell = int64_t(ho) * nlon_out + wo;
            const STORAGE_T *_qy = qy + cell * nchan_in;
            const STORAGE_T *_dy = dy + cell * nchan_out;

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
        }
    }

    // finalize: dqy = (S*Akvw - Avw*Ak) / S^2 (one warp per fine output cell).
    template <int THREADS_PER_BLOCK>
    __global__ __launch_bounds__(THREADS_PER_BLOCK) void s2_attn_bwd_upsample_scatter_dq_k(
        int nchan_in, int nlat_out, int nlon_out, const float *__restrict__ S, const float *__restrict__ Avw,
        const float *__restrict__ Ak, const float *__restrict__ Akvw, float *__restrict__ dqy)
    {
        const int batch = blockIdx.y;
        const int wid = blockIdx.x * blockDim.y + threadIdx.y;
        if (wid >= nlat_out * nlon_out) { return; }
        const int tidx = threadIdx.x;

        const int64_t gcell = int64_t(batch) * nlat_out * nlon_out + wid;
        const float s = S[gcell];
        const float s_inv = 1.0f / s;
        const float avw = Avw[gcell];
        const float *_Ak = Ak + gcell * nchan_in;
        const float *_Akvw = Akvw + gcell * nchan_in;
        float *_dqy = dqy + gcell * nchan_in;
        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
            _dqy[chan] = s_inv * s_inv * (s * _Akvw[chan] - avw * _Ak[chan]);
        }
    }

    // pass 3: per coarse cell, accumulate dkx/dvx locally over its row (no atomics).
    template <int THREADS_PER_BLOCK, typename STORAGE_T>
    __global__ __launch_bounds__(THREADS_PER_BLOCK) void s2_attn_bwd_upsample_scatter_dkv_k(
        int nchan_in, int nchan_out, int nlat_in, int nlon_in, int nlat_out, int nlon_out,
        const STORAGE_T *__restrict__ kx, const STORAGE_T *__restrict__ vx, const STORAGE_T *__restrict__ qy,
        const STORAGE_T *__restrict__ dy, const int64_t *__restrict__ row_off, const int64_t *__restrict__ col_idx,
        const float *__restrict__ quad_weights, const float *__restrict__ maxbuf, const float *__restrict__ S,
        const float *__restrict__ Avw, float *__restrict__ dkx, float *__restrict__ dvx)
    {
        extern __shared__ float shext[];
        float *sh_k = shext + threadIdx.y * (2 * nchan_in + 2 * nchan_out);
        float *sh_v = sh_k + nchan_in;
        float *sh_dk = sh_v + nchan_out;
        float *sh_dv = sh_dk + nchan_in;

        const int batch = blockIdx.y;
        const int wid = blockIdx.x * blockDim.y + threadIdx.y;
        if (wid >= nlat_in * nlon_in) { return; }
        const int tidx = threadIdx.x;
        const int hi = wid / nlon_in;
        const int wi = wid - hi * nlon_in;
        const int pscale_out = nlon_out / nlon_in;

        kx += int64_t(batch) * nlat_in * nlon_in * nchan_in + (int64_t(hi) * nlon_in + wi) * nchan_in;
        vx += int64_t(batch) * nlat_in * nlon_in * nchan_out + (int64_t(hi) * nlon_in + wi) * nchan_out;
        qy += int64_t(batch) * nlat_out * nlon_out * nchan_in;
        dy += int64_t(batch) * nlat_out * nlon_out * nchan_out;
        maxbuf += int64_t(batch) * nlat_out * nlon_out;
        S += int64_t(batch) * nlat_out * nlon_out;
        Avw += int64_t(batch) * nlat_out * nlon_out;
        dkx += int64_t(batch) * nlat_in * nlon_in * nchan_in + (int64_t(hi) * nlon_in + wi) * nchan_in;
        dvx += int64_t(batch) * nlat_in * nlon_in * nchan_out + (int64_t(hi) * nlon_in + wi) * nchan_out;

        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
            sh_k[chan] = vload(kx, chan);
            sh_dk[chan] = 0.f;
        }
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
            sh_v[chan] = vload(vx, chan);
            sh_dv[chan] = 0.f;
        }

        const float qw = quad_weights[hi];
        const int64_t rbeg = row_off[hi];
        const int rlen = static_cast<int>(row_off[hi + 1] - rbeg);
        const int64_t *col_hi = col_idx + rbeg;

        for (int off = 0; off < rlen; off++) {
            const int64_t col = col_hi[off];
            const int ho = static_cast<int>(col / nlon_out);
            const int wo = bwd_scatter_wo(static_cast<int>(col - int64_t(ho) * nlon_out), wi, pscale_out, nlon_out);
            const int64_t cell = int64_t(ho) * nlon_out + wo;
            const STORAGE_T *_qy = qy + cell * nchan_in;
            const STORAGE_T *_dy = dy + cell * nchan_out;

            float qd = 0.f, gd = 0.f;
            for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) { qd += sh_k[chan] * vload(_qy, chan); }
            for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { gd += sh_v[chan] * vload(_dy, chan); }
            qd = __warp_sum(qd);
            gd = __warp_sum(gd);

            const float s = S[cell];
            const float alpha_mul = expf(qd - maxbuf[cell]) * qw / s;
            const float integral = Avw[cell] / s;
            const float scale_dk = (gd - integral) * alpha_mul;

            for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { sh_dv[chan] += alpha_mul * vload(_dy, chan); }
            for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) { sh_dk[chan] += scale_dk * vload(_qy, chan); }
        }

        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) { dkx[chan] = sh_dk[chan]; }
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { dvx[chan] = sh_dv[chan]; }
    }

    // host launcher for the scatter backward (allocates the fp32 reduction buffers,
    // runs the four passes). STORAGE_T deduces from the activation pointers.
    template <typename STORAGE_T>
    static void launch_attn_bwd_upsample_scatter(int batch_size, int nchans_in, int nchans_out, int nlat_in,
                                                 int nlon_in, int nlat_out, int nlon_out, STORAGE_T *_kxp,
                                                 STORAGE_T *_vxp, STORAGE_T *_qyp, STORAGE_T *_dyp, int64_t *_row_off,
                                                 int64_t *_col_idx, float *_quad_weights, float *_dkxp, float *_dvxp,
                                                 float *_dqyp, cudaStream_t stream)
    {
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        torch::Tensor S = torch::zeros({batch_size, nlat_out, nlon_out}, opts);
        torch::Tensor Avw = torch::zeros({batch_size, nlat_out, nlon_out}, opts);
        torch::Tensor maxbuf = torch::full({batch_size, nlat_out, nlon_out}, -FLT_MAX, opts);
        torch::Tensor Ak = torch::zeros({batch_size, nlat_out, nlon_out, nchans_in}, opts);
        torch::Tensor Akvw = torch::zeros({batch_size, nlat_out, nlon_out, nchans_in}, opts);

        float *_S = reinterpret_cast<float *>(S.data_ptr());
        float *_Avw = reinterpret_cast<float *>(Avw.data_ptr());
        float *_maxbuf = reinterpret_cast<float *>(maxbuf.data_ptr());
        float *_Ak = reinterpret_cast<float *>(Ak.data_ptr());
        float *_Akvw = reinterpret_cast<float *>(Akvw.data_ptr());

        dim3 block(WARP_SIZE, THREADS / WARP_SIZE);
        dim3 grid_in(DIV_UP(nlat_in * nlon_in, block.y), batch_size);
        dim3 grid_out(DIV_UP(nlat_out * nlon_out, block.y), batch_size);

        const size_t sh_max = sizeof(float) * nchans_in * block.y;
        const size_t sh_stats = sizeof(float) * (nchans_in + nchans_out) * block.y;
        const size_t sh_dkv = sizeof(float) * (2 * nchans_in + 2 * nchans_out) * block.y;

        s2_attn_bwd_upsample_scatter_max_k<THREADS><<<grid_in, block, sh_max, stream>>>(
            nchans_in, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _qyp, _row_off, _col_idx, _maxbuf);
        CHECK_ERROR("s2_attn_bwd_upsample_scatter_max_k");

        s2_attn_bwd_upsample_scatter_stats_k<THREADS><<<grid_in, block, sh_stats, stream>>>(
            nchans_in, nchans_out, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_off, _col_idx,
            _quad_weights, _maxbuf, _S, _Avw, _Ak, _Akvw);
        CHECK_ERROR("s2_attn_bwd_upsample_scatter_stats_k");

        s2_attn_bwd_upsample_scatter_dq_k<THREADS>
            <<<grid_out, block, 0, stream>>>(nchans_in, nlat_out, nlon_out, _S, _Avw, _Ak, _Akvw, _dqyp);
        CHECK_ERROR("s2_attn_bwd_upsample_scatter_dq_k");

        s2_attn_bwd_upsample_scatter_dkv_k<THREADS><<<grid_in, block, sh_dkv, stream>>>(
            nchans_in, nchans_out, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_off, _col_idx,
            _quad_weights, _maxbuf, _S, _Avw, _dkxp, _dvxp);
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
    void s2_attn_bwd_upsample_dispatch(int batch_size, size_t nchans_in, size_t nchans_out, int64_t nlon_in,
                                       int64_t nlat_in, int64_t nlat_out, int64_t nlon_out, torch::Tensor kxP,
                                       torch::Tensor vxP, torch::Tensor qyP, torch::Tensor dyP,
                                       torch::Tensor psi_row_off, torch::Tensor psi_col_idx, torch::Tensor quad_weights,
                                       torch::Tensor dkxP, torch::Tensor dvxP, torch::Tensor dqyP)
    {

        auto stream = at::cuda::getCurrentCUDAStream().stream();

        // gradient outputs are always fp32
        float *_dkxp = reinterpret_cast<float *>(dkxP.data_ptr());
        float *_dvxp = reinterpret_cast<float *>(dvxP.data_ptr());
        float *_dqyp = reinterpret_cast<float *>(dqyP.data_ptr());

        int64_t *_row_off = reinterpret_cast<int64_t *>(psi_row_off.data_ptr());
        int64_t *_col_idx = reinterpret_cast<int64_t *>(psi_col_idx.data_ptr());
        float *_quad_weights = reinterpret_cast<float *>(quad_weights.data_ptr());

        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, qyP.scalar_type(), "s2_attn_bwd_upsample", [&] {
            scalar_t *_kxp = reinterpret_cast<scalar_t *>(kxP.data_ptr());
            scalar_t *_vxp = reinterpret_cast<scalar_t *>(vxP.data_ptr());
            scalar_t *_qyp = reinterpret_cast<scalar_t *>(qyP.data_ptr());
            scalar_t *_dyp = reinterpret_cast<scalar_t *>(dyP.data_ptr());

            launch_attn_bwd_upsample_scatter(batch_size, static_cast<int>(nchans_in), static_cast<int>(nchans_out),
                                             static_cast<int>(nlat_in), static_cast<int>(nlon_in),
                                             static_cast<int>(nlat_out), static_cast<int>(nlon_out), _kxp, _vxp, _qyp,
                                             _dyp, _row_off, _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp, stream);
        });
    }

} // namespace attention_kernels
