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
// Upsample (scatter-style) attention forward — CUDA
// =====================================================================================
//
// K, V live on the input (smaller) grid; Q lives on the output (larger) grid.
// psi is built with rows indexed by hi and cols encoding (ho, wo_canonical) on
// the output grid as ho * nlon_out + wo_canonical (canonical at wi=0). For
// wi > 0 the actual output column is (wo_canonical + pscale_out * wi) mod
// nlon_out, with pscale_out = nlon_out / nlon_in. Requires nlon_out % nlon_in
// == 0.
//
// Algorithm: same online softmax as the downsample kernel (single pass over
// contributors per output cell), but output-centric: each warp owns a unique
// (b, ho, wo) cell and scans every psi[hi] row, skipping entries whose stored
// ho_neigh doesn't match this output's ho and entries whose stored
// wo_canonical doesn't satisfy the divisibility wo ≡ wo_canonical (mod
// pscale_out). The mapping wi -> (wo_canonical + pscale_out * wi) mod nlon_out
// is a bijection within each pscale_out residue class, so for every (ho, wo)
// each contributing psi entry yields a unique wi = (wo - wo_canonical) /
// pscale_out within [0, nlon_in). The redundant scan over non-matching psi
// entries is the price for keeping the parallelization atomics-free, mirroring
// the CPU upsample kernel.
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

    // Output-centric online-softmax kernel for the scatter direction.
    // Called with (blockDim.x = WARP_SIZE, blockDim.y = THREADS_PER_BLOCK / WARP_SIZE).
    //
    // STORAGE_T is the global-memory element type (float4 for the fp32 vectorized
    // path; float / c10::Half / c10::BFloat16 for the scalar path). COMPUTE_T is the
    // arithmetic type (float4 / float); all dot-products, softmax and accumulation
    // happen in COMPUTE_T. vload/vstore widen/narrow at the memory boundary.
    template <int THREADS_PER_BLOCK, typename STORAGE_T>
    __global__ __launch_bounds__(THREADS_PER_BLOCK) void s2_attn_fwd_upsample_generic_vec_k(
        int nchan_in, int nchan_out, int nlat_in, int nlon_in, int nlat_out, int nlon_out,
        const STORAGE_T *__restrict__ kx, const STORAGE_T *__restrict__ vx, const STORAGE_T *__restrict__ qy,
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

        const int ho = wid / nlon_out;
        const int wo = wid - (ho * nlon_out);

        // one input lon step corresponds to pscale_out output lon steps
        const int pscale_out = nlon_out / nlon_in;

        // initialize per-cell shared accumulator
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { shy[chan] = __vset<COMPUTE_T>(0.f); }

        // batch-shift base pointers (channels-last layout)
        kx += int64_t(batch) * nlat_in * nlon_in * nchan_in;
        qy += int64_t(batch) * nlat_out * nlon_out * nchan_in + int64_t(ho) * nlon_out * nchan_in
            + int64_t(wo) * nchan_in;
        vx += int64_t(batch) * nlat_in * nlon_in * nchan_out;
        y += int64_t(batch) * nlat_out * nlon_out * nchan_out + int64_t(ho) * nlon_out * nchan_out
            + int64_t(wo) * nchan_out;

        float alpha_sum = 0.0f;
        float qdotk_max = -FLT_MAX;

        // outer scan over all input rows; only entries with ho_neigh == ho contribute
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

                // wi such that (wo_canonical + pscale_out * wi) mod nlon_out == wo;
                // exists iff (wo - wo_canonical) is divisible by pscale_out.
                int wo_diff = wo - wo_canonical;
                if (wo_diff < 0) wo_diff += nlon_out;
                if ((wo_diff % pscale_out) != 0) continue;
                const int wi = wo_diff / pscale_out;

                const STORAGE_T *_kx = kx + int64_t(hi) * nlon_in * nchan_in + int64_t(wi) * nchan_in;
                const STORAGE_T *_vx = vx + int64_t(hi) * nlon_in * nchan_out + int64_t(wi) * nchan_out;

                // qdotk = <qy[ho, wo, :], kx[hi, wi, :]>
                COMPUTE_T qdotkv = __vset<COMPUTE_T>(0.f);
                for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
                    qdotkv = __vadd(qdotkv, __vmul(vload(qy, chan), vload(_kx, chan)));
                }
                const float qdotk = __warp_sum(__vred(qdotkv));

                // online softmax update (single pass; same as the gather kernel)
                const float qdotk_max_tmp = max(qdotk_max, qdotk);
                const float alpha = expf(qdotk - qdotk_max_tmp) * quad_weights[hi];
                const float exp_save = expf(qdotk_max - qdotk_max_tmp);

                alpha_sum = alpha + alpha_sum * exp_save;

                for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
                    shy[chan] = __vadd(__vscale(exp_save, shy[chan]), __vscale(alpha, vload(_vx, chan)));
                }
                qdotk_max = qdotk_max_tmp;
            }
        }

        // finalize: y[b, ho, wo, :] = shy[:] / alpha_sum
        const float inv_alpha = 1.0f / alpha_sum;
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { vstore(y, chan, __vscale(inv_alpha, shy[chan])); }
    }

    template <typename STORAGE_T>
    static void launch_gen_attn_fwd_upsample(int batch_size, int nchans_in, int nchans_out, int nlat_in, int nlon_in,
                                             int nlat_out, int nlon_out, STORAGE_T *_kxp, STORAGE_T *_vxp,
                                             STORAGE_T *_qyp, int64_t *_row_off, int64_t *_col_idx,
                                             float *_quad_weights, STORAGE_T *_yp, cudaStream_t stream)
    {

        dim3 block(WARP_SIZE, THREADS / WARP_SIZE);
        dim3 grid(DIV_UP(nlat_out * nlon_out, block.y), batch_size);

        // shared memory holds compute-type (COMPUTE_T) data, not STORAGE_T
        const size_t shsize = sizeof(typename vec_traits<STORAGE_T>::compute_t) * nchans_out * block.y;

        s2_attn_fwd_upsample_generic_vec_k<THREADS>
            <<<grid, block, shsize, stream>>>(nchans_in, nchans_out, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp,
                                              _qyp, _row_off, _col_idx, _quad_weights, _yp);
        CHECK_ERROR("s2_attn_fwd_upsample_generic_vec_k");
    }

    // -----------------------------------------------------------------------------
    // host dispatcher — called from s2_attention_fwd_cuda when the direction is
    // upsample (nlon_out % nlon_in == 0). Vec/non-vec branching mirrors the
    // downsample dispatcher; only the generic kernel is instantiated for now.
    // -----------------------------------------------------------------------------
    // Native-storage dispatch (Tier B). Kept NON-templated (it is called from a
    // different TU, attention_cuda_fwd.cu) and does its own AT_DISPATCH over the
    // input dtype: fp16/bf16 take the scalar STORAGE_T path (widen at load, fp32
    // compute, narrow the output at store); fp32 keeps the float4 vectorized path.
    void s2_attn_fwd_upsample_dispatch(int batch_size, size_t nchans_in, size_t nchans_out, int64_t nlon_in,
                                       int64_t nlat_in, int64_t nlat_out, int64_t nlon_out, torch::Tensor kxP,
                                       torch::Tensor vxP, torch::Tensor qyP, torch::Tensor psi_row_off,
                                       torch::Tensor psi_col_idx, torch::Tensor quad_weights, torch::Tensor yP)
    {

        auto stream = at::cuda::getCurrentCUDAStream().stream();

        int64_t *_row_off = reinterpret_cast<int64_t *>(psi_row_off.data_ptr());
        int64_t *_col_idx = reinterpret_cast<int64_t *>(psi_col_idx.data_ptr());
        float *_quad_weights = reinterpret_cast<float *>(quad_weights.data_ptr());

        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, qyP.scalar_type(), "s2_attn_fwd_upsample", [&] {
            scalar_t *_kxp = reinterpret_cast<scalar_t *>(kxP.data_ptr());
            scalar_t *_vxp = reinterpret_cast<scalar_t *>(vxP.data_ptr());
            scalar_t *_qyp = reinterpret_cast<scalar_t *>(qyP.data_ptr());
            scalar_t *_yp = reinterpret_cast<scalar_t *>(yP.data_ptr());

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
                    && is_aligned<sizeof(float4)>(_qyp) && is_aligned<sizeof(float4)>(_yp)
                    && (nchans_in % VEC_SIZE) == 0 && (nchans_out % VEC_SIZE) == 0;
                if (use_vec) {
                    // STORAGE_T deduces to float4 from the pointers (NO explicit template
                    // arg: nvcc mishandles explicit-arg substitution with vec_traits in body).
                    launch_gen_attn_fwd_upsample(b, nci / VEC_SIZE, nco / VEC_SIZE, nli, nlonI, nlo, nlonO,
                                                 reinterpret_cast<float4 *>(_kxp), reinterpret_cast<float4 *>(_vxp),
                                                 reinterpret_cast<float4 *>(_qyp), _row_off, _col_idx, _quad_weights,
                                                 reinterpret_cast<float4 *>(_yp), stream);
                } else {
                    launch_gen_attn_fwd_upsample(b, nci, nco, nli, nlonI, nlo, nlonO, _kxp, _vxp, _qyp, _row_off,
                                                 _col_idx, _quad_weights, _yp, stream);
                }
            } else {
                // STORAGE_T deduces to scalar_t (fp16/bf16/double).
                launch_gen_attn_fwd_upsample(b, nci, nco, nli, nlonI, nlo, nlonO, _kxp, _vxp, _qyp, _row_off, _col_idx,
                                             _quad_weights, _yp, stream);
            }
        });
    }

} // namespace attention_kernels
