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
// Gather / self attention forward — Hopper (sm_90a) WGMMA, fp16, EXPERIMENTAL v2
// =====================================================================================
//
// v2: N-TILED ONLINE SOFTMAX (FlashAttention-shaped). Handles arbitrary neighbor
// counts — crucially the polar rows, whose neighborhood spans the full longitude
// dimension (rlen ~ W, and a few x W across rings) because longitudes converge at
// the poles. The earlier single-pass v1 (rlen <= NHALO) could not.
//
// Built on the VALIDATED WGMMA primitives from the DISCO tensor-core branch
// (attention_cuda_ptx.cuh): descriptor, m64n16k16 fp16 mma (predicate scale_D),
// Major::MN A/B core layouts, and the accumulator -> (m,n) epilogue.
//
// Structure (one warpgroup per (ho-row tile of TM queries, batch)):
//   O[TM x Cph] (fp32, shared), running max m[TM], running denom l[TM].
//   for n0 in 0..rlen step NT:
//     QK^T  : S_tile[TM x NT] = Q . Khalo_tile^T   (WGMMA, contract Cph in k=16)
//     online: m_new = max(m, rowmax(S_tile)); corr = exp(m - m_new);
//             l = l*corr + sum exp(S_tile - m_new)*w;  O *= corr;  P = exp(S - m_new)*w
//     PV    : O[:, d] += P_tile . Vhalo_tile[:, d]  (WGMMA, channel-tiled DTILE)
//   finalize: O /= l, narrow to fp16, store.
//
// PROVEN here: WGMMA mechanics + the online-softmax / N-tiling / channel-tiling
// control flow. STILL "VALIDATE ON HOPPER": the per-query longitude-shift HALO +
// MASK (v2 stages neighbors at the tile-base longitude, exact only for rows whose
// neighbor longitudes are wo-independent; the general union-band halo + per-(m,n)
// mask is the remaining keying item). O resident in shared caps Cph ~ 800 on
// Hopper; larger C wants the 2-pass variant (pass A: stats; pass B: PV).
//
// BUILD: TORCH_CUDA_ARCH_LIST="9.0a+PTX". Gated by TORCH_HARMONICS_ATTN_WGMMA=1.
// =====================================================================================

#include "attention_cuda.cuh"

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include "cudamacro.h"
#include "attention_cuda_utils.cuh"
#include "attention_cuda_ptx.cuh"

namespace attention_kernels
{

    static constexpr int WG_THREADS = 128;
    static constexpr int TM = 64;    // queries per row-tile (WGMMA M)
    static constexpr int NT = 16;    // neighbor tile width (WGMMA N for QK^T)
    static constexpr int DTILE = 16; // channel tile for PV (WGMMA N of the PV GEMM)
    static constexpr int KTILE = 16; // WGMMA K
    static constexpr int CPH_MAX = 1024;

    // Major::MN core-matrix element indexers (see attention_cuda_ptx.cuh):
    //   A [M,K] M-fast: (m/8)*128 + (m%8) + 8*k     ; B [K,N] N-fast: (n/8)*128 + k*8 + (n%8)
    __host__ __device__ __forceinline__ int aIdx(int m, int k) { return (m / 8) * 128 + (m % 8) + 8 * k; }
    __host__ __device__ __forceinline__ int bIdx(int k, int n) { return (n / 8) * 128 + k * 8 + (n % 8); }

#if defined(__CUDA_ARCH_FEAT_SM90_ALL)

    // descriptors are dtype-agnostic (fp16 and bf16 are both 2-byte; the byte strides
    // 128/256 are identical), so a void* address is all that's needed.
    __device__ __forceinline__ uint64_t descA(const void *p) { return make_wgmma_desc(p, 8 * 16, 16 * 16); }
    __device__ __forceinline__ uint64_t descB(const void *p) { return make_wgmma_desc(p, 8 * 16, 16 * 16); }

    // float -> element conversion and the m64n16k16 mma, selected by the storage type.
    template <typename T> __device__ __forceinline__ T f2e(float x);
    template <> __device__ __forceinline__ __half f2e<__half>(float x) { return __float2half(x); }
    template <> __device__ __forceinline__ __nv_bfloat16 f2e<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

    template <typename T> __device__ __forceinline__ void wgmma_n16(float (&d)[8], uint64_t da, uint64_t db)
    {
        if constexpr (std::is_same<T, __half>::value) {
            wgmma_m64n16k16_acc_fp16(d, da, db);
        } else {
            wgmma_m64n16k16_acc_bf16(d, da, db);
        }
    }

    // m64n16k16 accumulator -> (m,n) writeback (NHALO/NT=16 -> 8 cells/thread).
    template <typename OP>
    __device__ __forceinline__ void epilogue_n16(const float (&acc)[8], int warp_id, int lane, OP op)
    {
        const int m01 = warp_id * 16 + (lane >> 2);
        const int m23 = m01 + 8;
        const int n_a = (lane & 3) * 2;
        const int n_b = n_a + 1;
        const int ng = 0; // single n-group for N=16
        op(m01, n_a + 8 * ng, acc[ng * 4 + 0]);
        op(m01, n_b + 8 * ng, acc[ng * 4 + 1]);
        op(m23, n_a + 8 * ng, acc[ng * 4 + 2]);
        op(m23, n_b + 8 * ng, acc[ng * 4 + 3]);
        const int ng1 = 1;
        op(m01, n_a + 8 * ng1, acc[ng1 * 4 + 0]);
        op(m01, n_b + 8 * ng1, acc[ng1 * 4 + 1]);
        op(m23, n_a + 8 * ng1, acc[ng1 * 4 + 2]);
        op(m23, n_b + 8 * ng1, acc[ng1 * 4 + 3]);
    }

    // NOTE: no __restrict__ on the parameters — nvcc/cudafe device-stub generation for
    // a TEMPLATED __global__ mishandles __restrict__ qualifiers ("template-id ... does
    // not match any template declaration"). The non-templated kernels keep __restrict__.
    template <typename T>
    __global__ __launch_bounds__(WG_THREADS) void s2_attn_fwd_wgmma_sm90_k(int nchan, int nlat_in, int nlon_in,
                                                                           int nlat_out, int nlon_out,
                                                                           int tiles_per_row, const T *kx, const T *vx,
                                                                           const T *qy, const int32_t *row_idx,
                                                                           const int64_t *row_off, const int64_t *col_idx,
                                                                           const float *quad_weights, T *y)
    {
        const int tid = threadIdx.x;
        const int warp_id = tid / 32;
        const int lane = tid - warp_id * 32;

        const int batch = blockIdx.y;
        const int row = blockIdx.x / tiles_per_row;
        const int tile = blockIdx.x - row * tiles_per_row;
        const int wo_base = tile * TM;
        if (row >= nlat_out || wo_base >= nlon_out) { return; }

        const int ho = row_idx[row];
        const int pscale = nlon_in / nlon_out;
        const int tm = min(TM, nlon_out - wo_base);
        const int64_t rbeg = row_off[ho];
        const int rlen = static_cast<int>(row_off[ho + 1] - rbeg);

        // ---- shared ----
        extern __shared__ __align__(128) char smem_raw[];
        float *shO = reinterpret_cast<float *>(smem_raw);  // [TM x nchan] fp32 running output
        float *shM = shO + TM * nchan;                     // [TM] running max
        float *shL = shM + TM;                             // [TM] running denom
        float *shStile = shL + TM;                         // [TM x NT] fp32 scores
        T *shQ = reinterpret_cast<T *>(shStile + TM * NT); // [TM x KTILE]
        T *shK = shQ + TM * KTILE;                         // [KTILE x NT]
        T *shP = shK + KTILE * NT;                         // [TM x NT] probs (aIdx)
        T *shV = shP + TM * NT;                            // [NT x DTILE]

        for (int i = tid; i < TM * nchan; i += WG_THREADS) { shO[i] = 0.f; }
        for (int i = tid; i < TM; i += WG_THREADS) {
            shM[i] = -FLT_MAX;
            shL[i] = 0.f;
        }
        __syncthreads();

        // ---- build the canonical membership mask + latitude band (the HALO). The
        // neighborhood spans latitudes [hi_min, hi_max]; per query the longitudes shift
        // by pscale*wo but the canonical (wo=0) membership is fixed. We stage the
        // full-longitude band halo (absolute input cells, no shift) and mask each query
        // by de-shifting a halo column back to canonical and testing membership. ----
        char *member = reinterpret_cast<char *>(shV + NT * DTILE); // [nhi <= nlat_in][nlon_in] (after the T tiles)

        __shared__ int sh_himin;
        __shared__ int sh_himax;
        if (tid == 0) {
            sh_himin = nlat_in;
            sh_himax = -1;
        }
        __syncthreads();
        for (int off = tid; off < rlen; off += WG_THREADS) {
            const int hi = static_cast<int>(col_idx[rbeg + off] / nlon_in);
            // ::atomic* — the global int built-ins; the namespace has a custom
            // atomicMax(float*,float) that would otherwise shadow these.
            ::atomicMin(&sh_himin, hi);
            ::atomicMax(&sh_himax, hi);
        }
        __syncthreads();
        const int hi_min = sh_himin;
        const int nhi = sh_himax - sh_himin + 1;
        const int Nhalo = nhi * nlon_in;

        for (int i = tid; i < Nhalo; i += WG_THREADS) { member[i] = 0; }
        __syncthreads();
        for (int off = tid; off < rlen; off += WG_THREADS) {
            const int64_t col = col_idx[rbeg + off];
            const int hi = static_cast<int>(col / nlon_in);
            const int wi = static_cast<int>(col - int64_t(hi) * nlon_in);
            member[(hi - hi_min) * nlon_in + wi] = 1;
        }
        __syncthreads();

        // halo staging: absolute input cell (hi_min+hi_idx, wi); no per-query shift.
        auto stage_halo = [&](const T *src, int nchan_src, int hc, int c) -> T {
            const int hi_idx = hc / nlon_in;
            const int wi = hc - hi_idx * nlon_in;
            const int64_t g = int64_t(batch) * nlat_in * nlon_in * nchan_src
                + (int64_t(hi_min + hi_idx) * nlon_in + wi) * nchan_src + c;
            return src[g];
        };

        // ============================ halo N-tile loop (online softmax) ===============
        for (int n0 = 0; n0 < Nhalo; n0 += NT) {
            const int nt = min(NT, Nhalo - n0);

            // ---- QK^T: S[TM x NT] = Q . Khalo^T (contract nchan, k=16) ----
            float sacc[8];
#pragma unroll
            for (int i = 0; i < 8; i++) { sacc[i] = 0.f; }
            for (int k0 = 0; k0 < nchan; k0 += KTILE) {
                for (int idx = tid; idx < TM * KTILE; idx += WG_THREADS) {
                    const int m = idx / KTILE, k = idx - m * KTILE;
                    T v = f2e<T>(0.f);
                    if (m < tm) {
                        const int wo = wo_base + m;
                        const int64_t g = int64_t(batch) * nlat_out * nlon_out * nchan
                            + (int64_t(ho) * nlon_out + wo) * nchan + (k0 + k);
                        v = qy[g];
                    }
                    shQ[aIdx(m, k)] = v;
                }
                for (int idx = tid; idx < KTILE * NT; idx += WG_THREADS) {
                    const int k = idx / NT, n = idx - k * NT;
                    shK[bIdx(k, n)] = (n < nt) ? stage_halo(kx, nchan, n0 + n, k0 + k) : f2e<T>(0.f);
                }
                // promote the generic-proxy shQ/shK stores to the WGMMA async proxy
                // (required; wgmma.fence does not cover shared operands).
                fence_proxy_async_shared_cta();
                __syncthreads();
                // synchronous per k-tile: the async mma must finish reading shQ/shK
                // (commit+wait) before the next iteration overwrites them.
                wgmma_fence();
                wgmma_n16<T>(sacc, descA(shQ), descB(shK));
                wgmma_commit_group();
                wgmma_wait_group<0>();
                __syncthreads();
            }

            for (int i = tid; i < TM * NT; i += WG_THREADS) { shStile[i] = -FLT_MAX; }
            __syncthreads();
            epilogue_n16(sacc, warp_id, lane, [&](int m, int n, float vv) {
                if (m < tm && n < nt) { shStile[m * NT + n] = vv; }
            });
            __syncthreads();

            // ---- masked online softmax (per query: de-shift halo col -> canonical,
            // test membership). The score Q.K at the absolute cell is already correct;
            // the mask just selects which halo cells are THIS query's neighbors. ----
            for (int m = tid; m < tm; m += WG_THREADS) {
                const int wo = wo_base + m;
                const int shift = (pscale * wo) % nlon_in;
                float tmax = -FLT_MAX;
                for (int n = 0; n < nt; n++) {
                    const int hc = n0 + n;
                    const int hi_idx = hc / nlon_in;
                    int cw = (hc - hi_idx * nlon_in) - shift;
                    if (cw < 0) cw += nlon_in;
                    if (member[hi_idx * nlon_in + cw]) { tmax = fmaxf(tmax, shStile[m * NT + n]); }
                }
                const float m_new = fmaxf(shM[m], tmax);
                const float corr = expf(shM[m] - m_new);
                float lsum = 0.f;
                for (int n = 0; n < nt; n++) {
                    const int hc = n0 + n;
                    const int hi_idx = hc / nlon_in;
                    int cw = (hc - hi_idx * nlon_in) - shift;
                    if (cw < 0) cw += nlon_in;
                    float p = 0.f;
                    if (member[hi_idx * nlon_in + cw]) {
                        p = expf(shStile[m * NT + n] - m_new) * quad_weights[hi_min + hi_idx];
                    }
                    shP[aIdx(m, n)] = f2e<T>(p);
                    lsum += p;
                }
                for (int n = nt; n < NT; n++) { shP[aIdx(m, n)] = f2e<T>(0.f); }
                shL[m] = shL[m] * corr + lsum;
                for (int c = 0; c < nchan; c++) { shO[m * nchan + c] *= corr; }
                shM[m] = m_new;
            }
            __syncthreads();

            // ---- PV: O[:, d] += P . Vhalo[:, d] (contract NT halo cells, channel-tiled) ----
            for (int d0 = 0; d0 < nchan; d0 += DTILE) {
                for (int idx = tid; idx < NT * DTILE; idx += WG_THREADS) {
                    const int k = idx / DTILE, n = idx - k * DTILE;
                    shV[bIdx(k, n)] = (k < nt && (d0 + n) < nchan) ? stage_halo(vx, nchan, n0 + k, d0 + n) : f2e<T>(0.f);
                }
                // promote the generic-proxy shP (softmax) + shV stores to the WGMMA
                // async proxy before the PV mma reads them.
                fence_proxy_async_shared_cta();
                __syncthreads();
                float oacc[8];
#pragma unroll
                for (int i = 0; i < 8; i++) { oacc[i] = 0.f; }
                wgmma_fence();
                wgmma_n16<T>(oacc, descA(shP), descB(shV));
                wgmma_commit_group();
                wgmma_wait_group<0>();
                epilogue_n16(oacc, warp_id, lane, [&](int m, int n, float vv) {
                    const int c = d0 + n;
                    if (m < tm && c < nchan) { shO[m * nchan + c] += vv; }
                });
                __syncthreads();
            }
        }

        // ---- finalize: O /= l, store ----
        for (int m = tid; m < tm; m += WG_THREADS) {
            const float inv = (shL[m] > 0.f) ? 1.f / shL[m] : 0.f;
            for (int c = 0; c < nchan; c++) {
                const int wo = wo_base + m;
                const int64_t g
                    = int64_t(batch) * nlat_out * nlon_out * nchan + (int64_t(ho) * nlon_out + wo) * nchan + c;
                y[g] = f2e<T>(shO[m * nchan + c] * inv);
            }
        }
    }

    // ---------------------------------------------------------------------------------
    // DEBUG micro-kernel: D[64x16] = A[64x16] @ B[16x16], one m64n16k16, using the EXACT
    // descriptor / aIdx-bIdx staging / epilogue the real kernel uses. A,B,D are plain
    // row-major fp16 in global memory. Isolates the WGMMA mechanics from all attention
    // logic so they can be unit-tested against a host A@B. One warpgroup, one block.
    // ---------------------------------------------------------------------------------
    __global__ __launch_bounds__(WG_THREADS) void s2_wgmma_gemm_debug_k(const __half *__restrict__ A,
                                                                        const __half *__restrict__ B,
                                                                        __half *__restrict__ D)
    {
        const int tid = threadIdx.x;
        const int warp_id = tid / 32;
        const int lane = tid - warp_id * 32;

        // dynamic shared (matches DISCO + the real kernel — GMMA descriptors must
        // address dynamic smem the same way; static __shared__ here read as zero).
        extern __shared__ __align__(128) char smem_raw[];
        __half *shA = reinterpret_cast<__half *>(smem_raw); // [64 x 16] in aIdx layout
        __half *shB = shA + TM * KTILE;                     // [16 x 16] in bIdx layout

        for (int idx = tid; idx < TM * KTILE; idx += WG_THREADS) {
            const int m = idx / KTILE, k = idx - m * KTILE;
            shA[aIdx(m, k)] = A[m * KTILE + k]; // A row-major [64,16]
        }
        for (int idx = tid; idx < KTILE * NT; idx += WG_THREADS) {
            const int k = idx / NT, n = idx - k * NT;
            shB[bIdx(k, n)] = B[k * NT + n]; // B row-major [16,16]
        }
        fence_proxy_async_shared_cta();
        __syncthreads();

        float acc[8];
#pragma unroll
        for (int i = 0; i < 8; i++) { acc[i] = 0.f; }
        wgmma_fence();
        wgmma_m64n16k16_acc_fp16(acc, descA(shA), descB(shB));
        wgmma_commit_group();
        wgmma_wait_group<0>();

        epilogue_n16(acc, warp_id, lane, [&](int m, int n, float v) { D[m * NT + n] = __float2half(v); });
    }

#else
    template <typename T>
    __global__ void s2_attn_fwd_wgmma_sm90_k(int, int, int, int, int, int, const T *, const T *, const T *,
                                             const int32_t *, const int64_t *, const int64_t *, const float *, T *)
    {
        // matches the real kernel's (unqualified) parameter signature.
    }
    __global__ void s2_wgmma_gemm_debug_k(const __half *, const __half *, __half *) { }
#endif

    // -----------------------------------------------------------------------------
    // host dispatch — true if launched. Gated: env, sm_90+, fp16, self/gather,
    // Cph%16==0, Cph<=CPH_MAX. v2 N-tiles the neighbors so there is NO rlen cap.
    // -----------------------------------------------------------------------------
    bool s2_attn_fwd_wgmma_dispatch(int64_t batch_size, int64_t nchans_in, int64_t nchans_out, int64_t nlat_in,
                                    int64_t nlon_in, int64_t nlat_out, int64_t nlon_out, at::Tensor kxP, at::Tensor vxP,
                                    at::Tensor qyP, at::Tensor row_idx, at::Tensor row_off, at::Tensor col_idx,
                                    at::Tensor quad_weights, at::Tensor yP)
    {
        // diagnostics gated behind TORCH_HARMONICS_WGMMA_VERBOSE so bench runs are quiet.
        const char *vv = std::getenv("TORCH_HARMONICS_WGMMA_VERBOSE");
        const bool verbose = (vv != nullptr && std::atoi(vv) != 0);
        const auto dt = qyP.scalar_type();
        const bool is_half = (dt == at::kHalf);
        const bool is_bf16 = (dt == at::kBFloat16);
        if (verbose) {
            std::fprintf(stderr, "[torch-harmonics] wgmma_dispatch reached: half=%d bf16=%d nin=%ld nout=%ld major=%d\n",
                         static_cast<int>(is_half), static_cast<int>(is_bf16), static_cast<long>(nchans_in),
                         static_cast<long>(nchans_out), at::cuda::getCurrentDeviceProperties()->major);
        }
        if (!is_half && !is_bf16) { return false; }
        if (nchans_in != nchans_out) { return false; }
        if ((nchans_in % KTILE) != 0 || nchans_in > CPH_MAX) { return false; }
        if (at::cuda::getCurrentDeviceProperties()->major < 9) { return false; }

        if (verbose) {
            const int64_t max_rlen = row_off.diff().max().item<int64_t>();
            std::fprintf(
                stderr,
                "[torch-harmonics] attention WGMMA sm90 kernel LAUNCHED (dtype=%s head_dim=%ld max_row_len=%ld)\n",
                is_half ? "fp16" : "bf16", static_cast<long>(nchans_in), static_cast<long>(max_rlen));
        }

        const int nchan = static_cast<int>(nchans_in);
        const int tiles_per_row = static_cast<int>(DIV_UP(nlon_out, TM));
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        const int32_t *_row_idx = reinterpret_cast<const int32_t *>(row_idx.data_ptr());
        const int64_t *_row_off = reinterpret_cast<const int64_t *>(row_off.data_ptr());
        const int64_t *_col_idx = reinterpret_cast<const int64_t *>(col_idx.data_ptr());
        const float *_qw = reinterpret_cast<const float *>(quad_weights.data_ptr());

        dim3 block(WG_THREADS);
        dim3 grid(nlat_out * tiles_per_row, batch_size);

        // fp16 and bf16 are both 2-byte; the K/V/Q/P tiles are 2 bytes/elem. + membership
        // mask (halo): nlat_in*nlon_in bytes (worst-case latitude band).
        const size_t sh_bytes = sizeof(float) * (TM * nchan + 2 * TM + TM * NT)
            + size_t(2) * (TM * KTILE + KTILE * NT + TM * NT + NT * DTILE)
            + static_cast<size_t>(nlat_in) * static_cast<size_t>(nlon_in);
        // Hopper smem ceiling ~227 KB; fall back if the (nlat_in*nlon_in) membership
        // mask makes the tile too large (large grids — the full-lon halo is the v1
        // limit; an arc-band halo would shrink this).
        if (sh_bytes > 220000) { return false; }

        // launch the kernel instantiated for the storage type (fp16 or bf16). Written
        // out explicitly (not via a generic lambda) — nvcc's device-stub generation
        // mishandles launching a templated __global__ from inside an `auto` lambda.
        const int nli = static_cast<int>(nlat_in), nlonI = static_cast<int>(nlon_in);
        const int nlo = static_cast<int>(nlat_out), nlonO = static_cast<int>(nlon_out);
        if (is_half) {
            cudaFuncSetAttribute(reinterpret_cast<const void *>(s2_attn_fwd_wgmma_sm90_k<__half>),
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sh_bytes));
            s2_attn_fwd_wgmma_sm90_k<__half><<<grid, block, sh_bytes, stream>>>(
                nchan, nli, nlonI, nlo, nlonO, tiles_per_row, reinterpret_cast<const __half *>(kxP.data_ptr()),
                reinterpret_cast<const __half *>(vxP.data_ptr()), reinterpret_cast<const __half *>(qyP.data_ptr()),
                _row_idx, _row_off, _col_idx, _qw, reinterpret_cast<__half *>(yP.data_ptr()));
        } else {
            cudaFuncSetAttribute(reinterpret_cast<const void *>(s2_attn_fwd_wgmma_sm90_k<__nv_bfloat16>),
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sh_bytes));
            s2_attn_fwd_wgmma_sm90_k<__nv_bfloat16><<<grid, block, sh_bytes, stream>>>(
                nchan, nli, nlonI, nlo, nlonO, tiles_per_row, reinterpret_cast<const __nv_bfloat16 *>(kxP.data_ptr()),
                reinterpret_cast<const __nv_bfloat16 *>(vxP.data_ptr()),
                reinterpret_cast<const __nv_bfloat16 *>(qyP.data_ptr()), _row_idx, _row_off, _col_idx, _qw,
                reinterpret_cast<__nv_bfloat16 *>(yP.data_ptr()));
        }
        CHECK_ERROR("s2_attn_fwd_wgmma_sm90_k");
        return true;
    }

    // -----------------------------------------------------------------------------
    // Standalone forward op for isolated benchmarking of the WGMMA (sm_90a) path.
    // Mirrors s2_attention_fwd_cuda's channels-last handling, allocates the output,
    // and launches the WGMMA kernel via the dispatch above. Registered as
    // attention_kernels::forward_wgmma. Not routed by the module.
    // -----------------------------------------------------------------------------
    torch::Tensor s2_attn_fwd_wgmma_op(at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor quad_weights,
                                       at::Tensor col_idx, at::Tensor row_off, int64_t nlon_in, int64_t nlat_out,
                                       int64_t nlon_out)
    {
        CHECK_CUDA_INPUT_TENSOR(kx);
        CHECK_CUDA_INPUT_TENSOR(vx);
        CHECK_CUDA_INPUT_TENSOR(qy);
        CHECK_CUDA_TENSOR(quad_weights);
        CHECK_CUDA_TENSOR(col_idx);
        CHECK_CUDA_TENSOR(row_off);

        const int batch_size = kx.size(0);
        const int64_t nlat_in = kx.size(2);
        const size_t nchans_in = qy.size(1);
        const size_t nchans_out = vx.size(1);
        auto qy_type = qy.dtype();
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        // compacted-row -> ho map, sorted by row length for load balance — derived
        // internally exactly like the production forward (s2_attention_fwd_cuda).
        at::Tensor row_idx = sortRows(static_cast<int>(nlat_out), row_off, stream);

        // The WGMMA kernel indexes activations as channels-last (NHWC) contiguous.
        torch::Tensor kxP = kx, vxP = vx, qyP = qy;
        const bool qy_is_channels_last = (qyP.strides()[1] == 1);
        if (kxP.strides()[1] != 1) { kxP = permute_4D_to0231(kxP); }
        if (vxP.strides()[1] != 1) { vxP = permute_4D_to0231(vxP); }
        if (!qy_is_channels_last) { qyP = permute_4D_to0231(qyP); }

        const int64_t out_dims[] = {batch_size, nlat_out, nlon_out, static_cast<int64_t>(nchans_out)};
        torch::Tensor yP = torch::empty(out_dims, kxP.options());

        const bool ok = s2_attn_fwd_wgmma_dispatch(batch_size, nchans_in, nchans_out, nlat_in, nlon_in, nlat_out,
                                                   nlon_out, kxP, vxP, qyP, row_idx, row_off, col_idx, quad_weights, yP);
        TORCH_CHECK(ok,
                    "forward_wgmma: WGMMA kernel not launched (needs sm_90a build + fp16/bf16 + "
                    "nchans_in==nchans_out + nchans%16==0 + nchans<=CPH_MAX + arc-band fits smem)");

        // PyTorch C10 launch check — surfaces any async launch error from the kernel
        // (matches s2_attention_fwd_cuda). CHECK_ERROR inside the dispatch already checks
        // cudaGetLastError post-launch; this is the canonical torch-side guard.
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        torch::Tensor y = yP;
        if (!qy_is_channels_last) { y = permute_4D_to0312(y); }
        return y.to(qy_type);
    }

    TORCH_LIBRARY_IMPL(attention_kernels, CUDA, m) { m.impl("forward_wgmma", &s2_attn_fwd_wgmma_op); }

} // namespace attention_kernels
