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
#include <type_traits>

#include "cudamacro.h"
#include "attention_cuda_utils.cuh"

#define THREADS (64)

namespace attention_kernels
{

    // Output-centric backward kernel for the scatter direction.
    // Called with (blockDim.x = WARP_SIZE, blockDim.y = THREADS_PER_BLOCK / WARP_SIZE).
    //
    // STORAGE_T is the global-memory element type of the activations (float4 for the
    // fp32 vectorized path; float / c10::Half / c10::BFloat16 for the scalar path).
    // COMPUTE_T (= vec_traits<STORAGE_T>::compute_t) is the arithmetic / accumulator
    // type and also the gradient-output type: dkx/dvx/dqy stay fp32 (atomics +
    // accumulation), matching the gather backward. vload widens activations at load.
    template <int THREADS_PER_BLOCK, typename STORAGE_T>
    __global__ __launch_bounds__(THREADS_PER_BLOCK) void s2_attn_bwd_upsample_generic_vec_k(
        int nchan_in, int nchan_out, int nlat_in, int nlon_in, int nlat_out, int nlon_out,
        const STORAGE_T *__restrict__ kx, // [B][nlat_in][nlon_in][nchan_in]
        const STORAGE_T *__restrict__ vx, // [B][nlat_in][nlon_in][nchan_out]
        const STORAGE_T *__restrict__ qy, // [B][nlat_out][nlon_out][nchan_in]
        const STORAGE_T *__restrict__ dy, // [B][nlat_out][nlon_out][nchan_out]
        const int64_t *__restrict__ row_off, const int64_t *__restrict__ col_idx, const float *__restrict__ quad_weights,
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dkx, // [B][nlat_in][nlon_in][nchan_in]
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dvx, // [B][nlat_in][nlon_in][nchan_out]
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dqy)
    { // [B][nlat_out][nlon_out][nchan_in]

        using COMPUTE_T = typename vec_traits<STORAGE_T>::compute_t;

        extern __shared__ __align__(sizeof(float4)) float shext[];

        // shared per-warp accumulators / cached (qy, dy), all in COMPUTE_T:
        //   sh_alpha_k__[nchan_in], sh_alpha_vw_[nchan_in], sh_alpha_kvw[nchan_in]
        //   sh_dy[nchan_out], sh_qy[nchan_in]
        COMPUTE_T *sh_alpha_k__ = reinterpret_cast<COMPUTE_T *>(shext) + threadIdx.y * (nchan_in * 4 + nchan_out);
        COMPUTE_T *sh_alpha_vw_ = sh_alpha_k__ + nchan_in;
        COMPUTE_T *sh_alpha_kvw = sh_alpha_vw_ + nchan_in;
        COMPUTE_T *sh_dy = sh_alpha_kvw + nchan_in;
        COMPUTE_T *sh_qy = sh_dy + nchan_out;

        const int batch = blockIdx.y;
        const uint64_t wid = uint64_t(blockIdx.x) * blockDim.y + threadIdx.y;
        if (wid >= uint64_t(nlat_out) * nlon_out) { return; }

        const int tidx = threadIdx.x;

        const int ho = wid / nlon_out;
        const int wo = wid - (ho * nlon_out);

        // one input lon step corresponds to pscale_out output lon steps
        const int pscale_out = nlon_out / nlon_in;

        // base-pointer offsets
        kx += int64_t(batch) * nlat_in * nlon_in * nchan_in;
        qy += int64_t(batch) * nlat_out * nlon_out * nchan_in + int64_t(ho) * nlon_out * nchan_in
            + int64_t(wo) * nchan_in;
        vx += int64_t(batch) * nlat_in * nlon_in * nchan_out;
        dy += int64_t(batch) * nlat_out * nlon_out * nchan_out + int64_t(ho) * nlon_out * nchan_out
            + int64_t(wo) * nchan_out;

        dkx += int64_t(batch) * nlat_in * nlon_in * nchan_in;
        dvx += int64_t(batch) * nlat_in * nlon_in * nchan_out;
        dqy += int64_t(batch) * nlat_out * nlon_out * nchan_in + int64_t(ho) * nlon_out * nchan_in
            + int64_t(wo) * nchan_in;

        // init per-channel shared accumulators; cache qy and dy for this output cell.
        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
            sh_alpha_k__[chan] = __vset<COMPUTE_T>(0.f);
            sh_alpha_vw_[chan] = __vset<COMPUTE_T>(0.f);
            sh_alpha_kvw[chan] = __vset<COMPUTE_T>(0.f);
            sh_qy[chan] = vload(qy, chan);
        }
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { sh_dy[chan] = vload(dy, chan); }

#if __CUDA_ARCH__ < 900
        // matches the downsample kernel: on pre-sm_9.0 we fall back to 32-bit
        // atomics on consecutive lanes for the dkx/dvx scatter, which breaks the
        // per-thread ownership of each COMPUTE_T slot. Sync the warp here so the
        // shared loads above are visible across lanes when the scatter pass reads
        // them as plain floats.
        if constexpr (std::is_same<COMPUTE_T, float4>::value) { __syncwarp(); }
#endif

        // running scalars across the contributor scan
        float alpha_sum = 0.0f;
        float qdotk_max = -FLT_MAX;
        float integral = 0.0f;

        // -----------------------------------------------------------------------
        // pass 1: scan all psi[hi] rows; for entries with ho_neigh == ho and the
        // residue test satisfied, do the online-softmax update.
        // -----------------------------------------------------------------------
        for (int hi = 0; hi < nlat_in; hi++) {

            const int64_t rbeg = row_off[hi];
            const int64_t rend = row_off[hi + 1];
            const int rlen = static_cast<int>(rend - rbeg);
            const int64_t *col_idx_hi = col_idx + rbeg;

            for (int off = 0; off < rlen; off++) {

                const int64_t col = col_idx_hi[off];
                const int ho_neigh = static_cast<int>(col / nlon_out);
                if (ho_neigh != ho) continue;

                const int wo_canonical = static_cast<int>(col - int64_t(ho_neigh) * nlon_out);

                int wo_diff = wo - wo_canonical;
                if (wo_diff < 0) wo_diff += nlon_out;
                if ((wo_diff % pscale_out) != 0) continue;
                const int wi = wo_diff / pscale_out;

                const STORAGE_T *_kx = kx + int64_t(hi) * nlon_in * nchan_in + int64_t(wi) * nchan_in;
                const STORAGE_T *_vx = vx + int64_t(hi) * nlon_in * nchan_out + int64_t(wi) * nchan_out;

                // qdotk = <qy[ho, wo, :], kx[hi, wi, :]>
                // gdotv = <dy[ho, wo, :], vx[hi, wi, :]>
                COMPUTE_T qdotk_v = __vset<COMPUTE_T>(0.f);
                COMPUTE_T gdotv_v = __vset<COMPUTE_T>(0.f);
                for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
                    qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[chan], vload(_kx, chan)));
                }
                for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
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

                for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
                    const COMPUTE_T kxval = vload(_kx, chan);
                    sh_alpha_k__[chan] = __vadd(__vscale(max_correction, sh_alpha_k__[chan]), __vscale(alpha_inz, kxval));
                    sh_alpha_vw_[chan]
                        = __vadd(__vscale(max_correction, sh_alpha_vw_[chan]), __vset<COMPUTE_T>(ainz_gdotv));
                    sh_alpha_kvw[chan]
                        = __vadd(__vscale(max_correction, sh_alpha_kvw[chan]), __vscale(ainz_gdotv, kxval));
                }
                qdotk_max = qdotk_max_tmp;
            }
        }

        const float alpha_sum_inv = 1.0f / alpha_sum;
        integral *= alpha_sum_inv;

        // -----------------------------------------------------------------------
        // write dqy[b, ho, wo, :]
        //   dqy[chan] = (alpha_kvw[chan] * alpha_sum - alpha_vw_[chan] * alpha_k__[chan]) / alpha_sum^2
        // -----------------------------------------------------------------------
        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
            dqy[chan] = __vscale(
                alpha_sum_inv * alpha_sum_inv,
                __vsub(__vscale(alpha_sum, sh_alpha_kvw[chan]), __vmul(sh_alpha_vw_[chan], sh_alpha_k__[chan])));
        }

        // -----------------------------------------------------------------------
        // pass 2: scan again, scatter dkx and dvx with finalized softmax stats.
        // -----------------------------------------------------------------------
        for (int hi = 0; hi < nlat_in; hi++) {

            const int64_t rbeg = row_off[hi];
            const int64_t rend = row_off[hi + 1];
            const int rlen = static_cast<int>(rend - rbeg);
            const int64_t *col_idx_hi = col_idx + rbeg;

            for (int off = 0; off < rlen; off++) {

                const int64_t col = col_idx_hi[off];
                const int ho_neigh = static_cast<int>(col / nlon_out);
                if (ho_neigh != ho) continue;

                const int wo_canonical = static_cast<int>(col - int64_t(ho_neigh) * nlon_out);

                int wo_diff = wo - wo_canonical;
                if (wo_diff < 0) wo_diff += nlon_out;
                if ((wo_diff % pscale_out) != 0) continue;
                const int wi = wo_diff / pscale_out;

                const STORAGE_T *_kx = kx + int64_t(hi) * nlon_in * nchan_in + int64_t(wi) * nchan_in;
                const STORAGE_T *_vx = vx + int64_t(hi) * nlon_in * nchan_out + int64_t(wi) * nchan_out;

                // recompute qdotk and gdotv for this neighbor
                COMPUTE_T qdotk_v = __vset<COMPUTE_T>(0.f);
                COMPUTE_T gdotv_v = __vset<COMPUTE_T>(0.f);
                for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
                    qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[chan], vload(_kx, chan)));
                }
                for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
                    gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[chan], vload(_vx, chan)));
                }
                const float qdotk = __warp_sum(__vred(qdotk_v));
                const float gdotv = __warp_sum(__vred(gdotv_v));

                const float alpha_inz = expf(qdotk - qdotk_max) * quad_weights[hi];
                const float alpha_mul = alpha_inz * alpha_sum_inv;

                COMPUTE_T *_dkx = dkx + int64_t(hi) * nlon_in * nchan_in + int64_t(wi) * nchan_in;
                COMPUTE_T *_dvx = dvx + int64_t(hi) * nlon_in * nchan_out + int64_t(wi) * nchan_out;

                const float scale_fact_qy = (gdotv - integral) * alpha_mul;
                const float scale_fact_dy = alpha_mul;

#if __CUDA_ARCH__ < 900
                // 32-bit atomics on consecutive lanes (pre-sm_9.0 has no float4 atomicAdd)
                float *sh_qy_scl = reinterpret_cast<float *>(sh_qy);
                float *sh_dy_scl = reinterpret_cast<float *>(sh_dy);
                float *_dkx_scl = reinterpret_cast<float *>(_dkx);
                float *_dvx_scl = reinterpret_cast<float *>(_dvx);

                constexpr int VEC_SIZE = sizeof(COMPUTE_T) / sizeof(float);

                for (int chan = tidx; chan < nchan_in * VEC_SIZE; chan += WARP_SIZE) {
                    atomicAdd(_dkx_scl + chan, scale_fact_qy * sh_qy_scl[chan]);
                }
                for (int chan = tidx; chan < nchan_out * VEC_SIZE; chan += WARP_SIZE) {
                    atomicAdd(_dvx_scl + chan, scale_fact_dy * sh_dy_scl[chan]);
                }
#else
                // 128-bit float4 atomics on sm_9.0+
                for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
                    atomicAdd(_dkx + chan, __vscale(scale_fact_qy, sh_qy[chan]));
                }
                for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
                    atomicAdd(_dvx + chan, __vscale(scale_fact_dy, sh_dy[chan]));
                }
#endif
            }
        }
    }

    // STORAGE_T is deduced from the activation pointers; the gradient-output type
    // (vec_traits<STORAGE_T>::compute_t) is a non-deduced context that resolves once
    // STORAGE_T is fixed — exactly like vload(). Call sites MUST NOT pass explicit
    // template args: nvcc mishandles non-deduced parameter substitution when
    // STORAGE_T is given explicitly. Always let it deduce from the pointer types.
    template <typename STORAGE_T>
    static void launch_gen_attn_bwd_upsample(int batch_size, int nchans_in, int nchans_out, int nlat_in, int nlon_in,
                                             int nlat_out, int nlon_out, STORAGE_T *_kxp, STORAGE_T *_vxp,
                                             STORAGE_T *_qyp, STORAGE_T *_dyp, int64_t *_row_off, int64_t *_col_idx,
                                             float *_quad_weights, typename vec_traits<STORAGE_T>::compute_t *_dkxp,
                                             typename vec_traits<STORAGE_T>::compute_t *_dvxp,
                                             typename vec_traits<STORAGE_T>::compute_t *_dqyp, cudaStream_t stream)
    {

        dim3 block(WARP_SIZE, THREADS / WARP_SIZE);
        dim3 grid(DIV_UP(nlat_out * nlon_out, block.y), batch_size);

        // shared per-warp (COMPUTE_T): nchan_in * 3 (alpha_k, alpha_vw, alpha_kvw) + nchan_out (sh_dy) + nchan_in (sh_qy)
        const size_t shsize = sizeof(typename vec_traits<STORAGE_T>::compute_t) * (nchans_in * 4 + nchans_out) * block.y;

        s2_attn_bwd_upsample_generic_vec_k<THREADS>
            <<<grid, block, shsize, stream>>>(nchans_in, nchans_out, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp,
                                              _qyp, _dyp, _row_off, _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp);
        CHECK_ERROR("s2_attn_bwd_upsample_generic_vec_k");
    }

    // -----------------------------------------------------------------------------
    // host dispatcher — called from s2_attention_bwd_dkvq_cuda when the direction
    // is upsample (nlon_out % nlon_in == 0). Native-storage (Tier B): kept
    // NON-templated (called from a different TU, attention_cuda_bwd.cu) and does its
    // own AT_DISPATCH over the activation dtype. fp16/bf16 take the scalar STORAGE_T
    // path (widen at load, fp32 compute); fp32 keeps the float4 vectorized path. The
    // gradient tensors dkx/dvx/dqy are always fp32 (COMPUTE_T) — same as gather bwd.
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

            const int b = batch_size;
            const int nci = static_cast<int>(nchans_in);
            const int nco = static_cast<int>(nchans_out);
            const int nli = static_cast<int>(nlat_in);
            const int nlonI = static_cast<int>(nlon_in);
            const int nlo = static_cast<int>(nlat_out);
            const int nlonO = static_cast<int>(nlon_out);

            if constexpr (std::is_same<scalar_t, float>::value) {
                constexpr int VEC_SIZE = sizeof(float4) / sizeof(float);
                const bool use_vec = is_aligned<sizeof(float4)>(_kxp) && is_aligned<sizeof(float4)>(_vxp)
                    && is_aligned<sizeof(float4)>(_qyp) && is_aligned<sizeof(float4)>(_dyp)
                    && is_aligned<sizeof(float4)>(_dkxp) && is_aligned<sizeof(float4)>(_dvxp)
                    && is_aligned<sizeof(float4)>(_dqyp) && (nchans_in % VEC_SIZE) == 0 && (nchans_out % VEC_SIZE) == 0;
                if (use_vec) {
                    // STORAGE_T deduces to float4 from the pointers; grads are float4 too.
                    launch_gen_attn_bwd_upsample(b, nci / VEC_SIZE, nco / VEC_SIZE, nli, nlonI, nlo, nlonO,
                                                 reinterpret_cast<float4 *>(_kxp), reinterpret_cast<float4 *>(_vxp),
                                                 reinterpret_cast<float4 *>(_qyp), reinterpret_cast<float4 *>(_dyp),
                                                 _row_off, _col_idx, _quad_weights, reinterpret_cast<float4 *>(_dkxp),
                                                 reinterpret_cast<float4 *>(_dvxp), reinterpret_cast<float4 *>(_dqyp),
                                                 stream);
                } else {
                    // STORAGE_T deduces to float; grads are float.
                    launch_gen_attn_bwd_upsample(b, nci, nco, nli, nlonI, nlo, nlonO, _kxp, _vxp, _qyp, _dyp, _row_off,
                                                 _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp, stream);
                }
            } else {
                // fp16/bf16 (and double): STORAGE_T deduces to scalar_t, grads stay fp32.
                launch_gen_attn_bwd_upsample(b, nci, nco, nli, nlonI, nlo, nlonO, _kxp, _vxp, _qyp, _dyp, _row_off,
                                             _col_idx, _quad_weights, _dkxp, _dvxp, _dqyp, stream);
            }
        });
    }

} // namespace attention_kernels
