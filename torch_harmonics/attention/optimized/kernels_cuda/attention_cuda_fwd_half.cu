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
// Gather (downsample / self) attention forward — fp16/bf16 PACKED specialization
// =====================================================================================
//
// A bf16/fp16-specialized variant of the generic gather forward (warp-per-output
// cell). Motivation: stored as a scalar, each c10::Half lands in its own 32-bit
// register (top 16 bits wasted), so a wide fp16 load doubles register pressure
// versus fp32 and tips this (register-limited) kernel into spills — which is why
// the earlier scalar-fp16 "float8" experiment regressed. Here the activations are
// loaded 8-wide (LDG.128) as 4x __half2 / __nv_bfloat162 and the q.k dot product
// runs on packed __hfma2, so two halves share one register.
//
// Precision is unchanged from the scalar path: the q.k accumulator is packed
// half2 only for the handful of per-lane terms (nchan/8/32), then reduced in fp32;
// the softmax and the per-channel value accumulator `shy` stay fp32. Only the
// loads + the dot product are packed; alpha*v accumulates in fp32 as before.
//
// bf16 packed FMA is native on sm_80+; on older arches the traits fall back to a
// float-based fma2 so the TU still compiles, and the host dispatch only routes
// bf16 here when the device is sm_80+ (fp16 packs on Volta+).
// =====================================================================================

#include "attention_cuda.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cfloat>

#include "cudamacro.h"
#include "attention_cuda_utils.cuh"

#define THREADS (64)

namespace attention_kernels
{

    // packed-half traits: maps a c10 element type to its 2-wide vector type, an
    // 8-wide (16-byte, LDG.128) load struct, and the pack/unpack/fma2 primitives.
    template <typename ELEM> struct pack8;

    template <> struct pack8<c10::Half> {
        using v2_t = __half2;
        struct v8_t {
            __half2 a, b, c, d;
        };
        static __device__ __forceinline__ v2_t zero() { return __floats2half2_rn(0.f, 0.f); }
        static __device__ __forceinline__ v2_t fma2(v2_t x, v2_t y, v2_t acc) { return __hfma2(x, y, acc); }
        static __device__ __forceinline__ float2 to_f2(v2_t v) { return __half22float2(v); }
        static __device__ __forceinline__ v2_t pack(float lo, float hi) { return __floats2half2_rn(lo, hi); }
    };

    template <> struct pack8<c10::BFloat16> {
        using v2_t = __nv_bfloat162;
        struct v8_t {
            __nv_bfloat162 a, b, c, d;
        };
        static __device__ __forceinline__ v2_t zero() { return __floats2bfloat162_rn(0.f, 0.f); }
        static __device__ __forceinline__ v2_t fma2(v2_t x, v2_t y, v2_t acc)
        {
#if __CUDA_ARCH__ >= 800
            return __hfma2(x, y, acc);
#else
            // pre-sm_80: bf16 packed FMA is not native. This path exists only so the
            // TU compiles; the host dispatch never routes bf16 here below sm_80.
            const float2 xf = __bfloat1622float2(x);
            const float2 yf = __bfloat1622float2(y);
            const float2 af = __bfloat1622float2(acc);
            return __floats2bfloat162_rn(xf.x * yf.x + af.x, xf.y * yf.y + af.y);
#endif
        }
        static __device__ __forceinline__ float2 to_f2(v2_t v) { return __bfloat1622float2(v); }
        static __device__ __forceinline__ v2_t pack(float lo, float hi) { return __floats2bfloat162_rn(lo, hi); }
    };

    // warp-per-output-cell gather forward, packed fp16/bf16. nchan_in/nchan_out are
    // in ELEM units and must be multiples of 8 (checked by the host dispatch).
    template <int BDIM_X, typename ELEM>
    __global__ __launch_bounds__(BDIM_X) void s2_attn_fwd_generic_half2_k(
        int nchan_in, int nchan_out, int nlat_in, int nlon_in, int nlat_out, int nlon_out, const ELEM *__restrict__ kx,
        const ELEM *__restrict__ vx, const ELEM *__restrict__ qy, const int32_t *__restrict__ row_idx,
        const int64_t *__restrict__ row_off, const int64_t *__restrict__ col_idx,
        const float *__restrict__ quad_weights, ELEM *__restrict__ y)
    {
        using PK = pack8<ELEM>;
        using v2_t = typename PK::v2_t;
        using v8_t = typename PK::v8_t;

        const int nchan8_in = nchan_in >> 3;
        const int nchan8_out = nchan_out >> 3;

        // fp32 per-channel value accumulator (precision unchanged from scalar path).
        // Each lane owns the 8 contiguous channels of one v8_t; storing them at
        // stride SHY_STRIDE (=9) per c8 block makes the per-j cross-lane access
        // bank-conflict free (gcd(9,32)==1), since lane t writes shy[c8*9 + j] and
        // c8 = t + 32*m  ->  bank (9*t + j) mod 32 is a bijection over the warp.
        constexpr int SHY_STRIDE = 9;
        extern __shared__ __align__(sizeof(float4)) float shext[];
        float *shy = shext + threadIdx.y * (SHY_STRIDE * nchan8_out);

        const int batch = blockIdx.y;
        const int wid = blockIdx.x * blockDim.y + threadIdx.y;
        if (wid >= nlat_out * nlon_out) { return; }

        const int tidx = threadIdx.x;
        const int h = wid / nlon_out;
        const int wo = wid - (h * nlon_out);
        const int ho = row_idx[h];
        const int pscale = nlon_in / nlon_out;

        for (int i = tidx; i < SHY_STRIDE * nchan8_out; i += WARP_SIZE) { shy[i] = 0.f; }

        kx += int64_t(batch) * nlat_in * nlon_in * nchan_in;
        qy += int64_t(batch) * nlat_out * nlon_out * nchan_in + int64_t(ho) * nchan_in * nlon_out
            + int64_t(wo) * nchan_in;
        vx += int64_t(batch) * nlat_in * nlon_in * nchan_out;
        y += int64_t(batch) * nlat_out * nlon_out * nchan_out + int64_t(ho) * nchan_out * nlon_out
            + int64_t(wo) * nchan_out;

        const v8_t *qy8 = reinterpret_cast<const v8_t *>(qy);

        float alpha_sum = 0.0f;
        float qdotk_max = -FLT_MAX;

        const int64_t rbeg = row_off[ho];
        const int64_t rend = row_off[ho + 1];
        col_idx += rbeg;
        const int rlen = static_cast<int>(rend - rbeg);

        for (int off = 0; off < rlen; off++) {
            const int64_t col = col_idx[off];
            const int hi = col / nlon_in;
            const int wi = col - (hi * nlon_in);
            const int wi_wo = wi + pscale * wo;
            const int wip = wi_wo - (wi_wo / nlon_in) * nlon_in;

            const v8_t *kx8
                = reinterpret_cast<const v8_t *>(kx + int64_t(hi) * nlon_in * nchan_in + int64_t(wip) * nchan_in);
            const v8_t *vx8
                = reinterpret_cast<const v8_t *>(vx + int64_t(hi) * nlon_in * nchan_out + int64_t(wip) * nchan_out);

            // q.k: packed half2 FMA per lane, reduced to fp32 across the warp.
            v2_t acc = PK::zero();
            for (int c8 = tidx; c8 < nchan8_in; c8 += WARP_SIZE) {
                const v8_t q = qy8[c8];
                const v8_t k = kx8[c8];
                acc = PK::fma2(q.a, k.a, acc);
                acc = PK::fma2(q.b, k.b, acc);
                acc = PK::fma2(q.c, k.c, acc);
                acc = PK::fma2(q.d, k.d, acc);
            }
            const float2 accf = PK::to_f2(acc);
            const float qdotk = __warp_sum(accf.x + accf.y);

            const float qdotk_max_tmp = max(qdotk_max, qdotk);
            const float alpha = expf(qdotk - qdotk_max_tmp) * quad_weights[hi];
            const float exp_save = expf(qdotk_max - qdotk_max_tmp);
            alpha_sum = alpha + alpha_sum * exp_save;

            // value accumulate: load v packed, convert to fp32, accumulate in fp32.
            for (int c8 = tidx; c8 < nchan8_out; c8 += WARP_SIZE) {
                const v8_t v = vx8[c8];
                const float2 a = PK::to_f2(v.a);
                const float2 b = PK::to_f2(v.b);
                const float2 c = PK::to_f2(v.c);
                const float2 d = PK::to_f2(v.d);
                float *sh = shy + c8 * SHY_STRIDE;
                sh[0] = exp_save * sh[0] + alpha * a.x;
                sh[1] = exp_save * sh[1] + alpha * a.y;
                sh[2] = exp_save * sh[2] + alpha * b.x;
                sh[3] = exp_save * sh[3] + alpha * b.y;
                sh[4] = exp_save * sh[4] + alpha * c.x;
                sh[5] = exp_save * sh[5] + alpha * c.y;
                sh[6] = exp_save * sh[6] + alpha * d.x;
                sh[7] = exp_save * sh[7] + alpha * d.y;
            }
            qdotk_max = qdotk_max_tmp;
        }

        const float inv = 1.0f / alpha_sum;
        v8_t *y8 = reinterpret_cast<v8_t *>(y);
        for (int c8 = tidx; c8 < nchan8_out; c8 += WARP_SIZE) {
            const float *sh = shy + c8 * SHY_STRIDE;
            v8_t out;
            out.a = PK::pack(sh[0] * inv, sh[1] * inv);
            out.b = PK::pack(sh[2] * inv, sh[3] * inv);
            out.c = PK::pack(sh[4] * inv, sh[5] * inv);
            out.d = PK::pack(sh[6] * inv, sh[7] * inv);
            y8[c8] = out;
        }
    }

    // -----------------------------------------------------------------------------
    // host dispatch — called from s2_attn_fwd_dispatch (attention_cuda_fwd.cu) for
    // the fp16/bf16 gather/self path. Returns true if it launched the packed kernel,
    // false when not eligible (caller then takes the scalar fallback). Eligibility:
    // dtype in {Half, BFloat16}, nchans % 8 == 0, 16-byte aligned, and (for bf16)
    // device sm_80+.
    // -----------------------------------------------------------------------------
    bool s2_attn_fwd_half2_dispatch(int64_t batch_size, int64_t nchans_in, int64_t nchans_out, int64_t nlat_in,
                                    int64_t nlon_in, int64_t nlat_out, int64_t nlon_out, at::Tensor kxP, at::Tensor vxP,
                                    at::Tensor qyP, at::Tensor row_idx, at::Tensor row_off, at::Tensor col_idx,
                                    at::Tensor quad_weights, at::Tensor yP)
    {
        if ((nchans_in % 8) != 0 || (nchans_out % 8) != 0) { return false; }

        const auto dt = qyP.scalar_type();
        const bool is_half = (dt == at::kHalf);
        const bool is_bf16 = (dt == at::kBFloat16);
        if (!is_half && !is_bf16) { return false; }

        // bf16 packed FMA is native only on sm_80+; below that, defer to the scalar path.
        if (is_bf16 && at::cuda::getCurrentDeviceProperties()->major < 8) { return false; }

        void *kxp = kxP.data_ptr();
        void *vxp = vxP.data_ptr();
        void *qyp = qyP.data_ptr();
        void *yp = yP.data_ptr();
        if (!is_aligned<16>(kxp) || !is_aligned<16>(vxp) || !is_aligned<16>(qyp) || !is_aligned<16>(yp)) {
            return false;
        }

        auto stream = at::cuda::getCurrentCUDAStream().stream();

        int32_t *_row_idx = reinterpret_cast<int32_t *>(row_idx.data_ptr());
        int64_t *_row_off = reinterpret_cast<int64_t *>(row_off.data_ptr());
        int64_t *_col_idx = reinterpret_cast<int64_t *>(col_idx.data_ptr());
        float *_quad_weights = reinterpret_cast<float *>(quad_weights.data_ptr());

        dim3 block(WARP_SIZE, THREADS / WARP_SIZE);
        dim3 grid(DIV_UP(nlat_out * nlon_out, block.y), batch_size);
        // padded fp32 accumulator: 9 floats per 8-channel block (bank-conflict free)
        const size_t shsize = sizeof(float) * 9 * (nchans_out / 8) * block.y;

        const int nci = static_cast<int>(nchans_in);
        const int nco = static_cast<int>(nchans_out);
        const int nli = static_cast<int>(nlat_in);
        const int nlonI = static_cast<int>(nlon_in);
        const int nlo = static_cast<int>(nlat_out);
        const int nlonO = static_cast<int>(nlon_out);

        if (is_half) {
            s2_attn_fwd_generic_half2_k<THREADS, c10::Half><<<grid, block, shsize, stream>>>(
                nci, nco, nli, nlonI, nlo, nlonO, reinterpret_cast<const c10::Half *>(kxp),
                reinterpret_cast<const c10::Half *>(vxp), reinterpret_cast<const c10::Half *>(qyp), _row_idx, _row_off,
                _col_idx, _quad_weights, reinterpret_cast<c10::Half *>(yp));
        } else {
            s2_attn_fwd_generic_half2_k<THREADS, c10::BFloat16><<<grid, block, shsize, stream>>>(
                nci, nco, nli, nlonI, nlo, nlonO, reinterpret_cast<const c10::BFloat16 *>(kxp),
                reinterpret_cast<const c10::BFloat16 *>(vxp), reinterpret_cast<const c10::BFloat16 *>(qyp), _row_idx,
                _row_off, _col_idx, _quad_weights, reinterpret_cast<c10::BFloat16 *>(yp));
        }
        CHECK_ERROR("s2_attn_fwd_generic_half2_k");
        return true;
    }

} // namespace attention_kernels
