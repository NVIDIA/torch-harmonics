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

// DISCO S2 forward contraction (K-packed psi), fp16 in / fp32 accumulate / fp16 out.
//
// Formulation (per output latitude ho, per support segment = one input latitude hi with a contiguous
// longitude window [s, s+w) mod Wi):
//   out[c, k, wo] += sum_j f[k][j] * X[c, hi, s + j + wo*p]
// For a tile of Tw consecutive output longitudes this is a Toeplitz GEMM
//   D[(k,wo_l), c] = sum_u F[(k,wo_l), u] * X[c, hi, s + wo0*p + u],  F[(k,wo_l),u] = f[k][u - wo_l*p]
// with M = K*Tw <= 64 rows (Tw = 64/K), N = channels, and K-dim u in [0, w + (Tw-1)p).
// The A operand (F) is generated in registers from f^T staged in smem and reused for all wo tiles of the
// CTA; the B operand (X, channel-innermost) is read by wgmma straight from smem with per-k-step
// descriptors, so k-steps can concatenate 8-row blocks of different segments.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <string.h>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#include "leader_small.cuh"

namespace ffi = tvm::ffi;

struct __align__(16) Seg {
    int hi, s, w, n1, len1, n2, pad0, pad1;
};

// ------------------------------------------------------------------------------------------------
// small PTX helpers
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar)
{
    asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.release.cta.shared::cta.b64 st, [%0];\n}" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes)
{
    asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 st, [%0], %1;\n}" ::"r"(bar),
                 "r"(bytes)
                 : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity)
{
    asm volatile("{\n .reg .pred P1;\n"
                 "WAIT_%=:\n"
                 " mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n"
                 " @!P1 bra WAIT_%=;\n"
                 "}\n" ::"r"(bar),
                 "r"(parity)
                 : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void *src, uint32_t bytes, uint32_t bar)
{
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
                 "l"(src), "r"(bytes), "r"(bar)
                 : "memory");
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void *src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
// the mbarrier tracks completion of this thread's prior cp.async ops (pending count pre-incremented)
__device__ __forceinline__ void cp_async_mbar_arrive(uint32_t bar)
{
    asm volatile("cp.async.mbarrier.arrive.shared::cta.b64 [%0];" ::"r"(bar) : "memory");
}
// TMA tensor store (smem tile -> 3-D box of out viewed as [B*C*K][Ho][Wo]), clipped at the tensor bounds
__device__ __forceinline__ void tma_store_3d(const void *map, uint32_t src, int x, int y, int z)
{
    asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, %4}], [%1];" ::"l"(
                     reinterpret_cast<uint64_t>(map)),
                 "r"(src), "r"(x), "r"(y), "r"(z)
                 : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_read0() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait0() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ void named_bar(int id, int n)
{
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ uint32_t lds32(uint32_t addr)
{
    uint32_t v;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(addr));
    return v;
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait()
{
    asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}
__device__ __forceinline__ void fence_u32(uint32_t &r) { asm volatile("" : "+r"(r)::"memory"); }
__device__ __forceinline__ void fence_f32(float &r) { asm volatile("" : "+f"(r)::"memory"); }
template <int R> __device__ __forceinline__ void reg_alloc()
{
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(R));
}
template <int R> __device__ __forceinline__ void reg_dealloc()
{
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(R));
}
__device__ __forceinline__ uint64_t make_desc(uint32_t addr, uint32_t lbo, uint32_t sbo)
{
    return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)((lbo & 0x3FFFF) >> 4) << 16)
        | ((uint64_t)((sbo & 0x3FFFF) >> 4) << 32);
}

__device__ __forceinline__ void wgmma_rs_n8(float *d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t descB)
{
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %9, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
                 "{%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, %8, p, 1, 1, 1;\n"
                 "}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(descB), "r"(1));
}

__device__ __forceinline__ void wgmma_rs_n16(float *d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t descB)
{
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %13, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, "
                 "{%8, %9, %10, %11}, %12, p, 1, 1, 1;\n"
                 "}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(descB), "r"(1));
}

__device__ __forceinline__ void wgmma_rs_n24(float *d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t descB)
{
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %17, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, "
                 "{%12, %13, %14, %15}, %16, p, 1, 1, 1;\n"
                 "}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                   "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(descB), "r"(1));
}

__device__ __forceinline__ void wgmma_rs_n32(float *d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t descB)
{
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %21, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
                 "{%16, %17, %18, %19}, %20, p, 1, 1, 1;\n"
                 "}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                   "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(descB), "r"(1));
}

__device__ __forceinline__ void wgmma_rs_n64(float *d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t descB)
{
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %37, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, "
                 "%22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
                 "{%32, %33, %34, %35}, %36, p, 1, 1, 1;\n"
                 "}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                   "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
                   "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
                   "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(descB), "r"(1));
}

__device__ __forceinline__ void wgmma_rs_n72(float *d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t descB)
{
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %41, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n72k16.f32.f16.f16 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, "
                 "%22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}, "
                 "{%36, %37, %38, %39}, %40, p, 1, 1, 1;\n"
                 "}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                   "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
                   "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]),
                   "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),
                   "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(descB), "r"(1));
}

template <int N>
__device__ __forceinline__ void wgmma_rs(float *d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc)
{
    if constexpr (N == 8)
        wgmma_rs_n8(d, a0, a1, a2, a3, desc);
    else if constexpr (N == 16)
        wgmma_rs_n16(d, a0, a1, a2, a3, desc);
    else if constexpr (N == 24)
        wgmma_rs_n24(d, a0, a1, a2, a3, desc);
    else if constexpr (N == 32)
        wgmma_rs_n32(d, a0, a1, a2, a3, desc);
    else if constexpr (N == 64)
        wgmma_rs_n64(d, a0, a1, a2, a3, desc);
    else if constexpr (N == 72)
        wgmma_rs_n72(d, a0, a1, a2, a3, desc);
}

// ------------------------------------------------------------------------------------------------
// kernel 1: inp [BC][Hi*Wi] -> xt [G][Hi*Wi][8]  (channel-innermost, 16 B per position)
// ------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
    transpose_kernel(const uint16_t *__restrict__ inp, uint4 *__restrict__ xt, int BC, long long HW)
{
    long long pos = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int gi = blockIdx.y;
    if (pos >= HW) return;
    uint32_t v[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        int ch = gi * 8 + c;
        v[c] = ch < BC ? (uint32_t)__ldg(inp + (size_t)ch * HW + pos) : 0u;
    }
    uint4 o;
    o.x = v[0] | (v[1] << 16);
    o.y = v[2] | (v[3] << 16);
    o.z = v[4] | (v[5] << 16);
    o.w = v[6] | (v[7] << 16);
    xt[(size_t)gi * HW + pos] = o;
}

// ------------------------------------------------------------------------------------------------
// kernel 2: per output row, split the support into segments: maximal runs of points with the same hi
// and consecutive wi; a run ending at the seam (wi = Wi-1) is merged with the run of the same hi
// starting at wi = 0.  nseg[ho] = -1 marks rows handled by the generic fallback.
// ------------------------------------------------------------------------------------------------
#define PREP_THREADS 512
#define SEGCAP_PREP 128
struct FtParams {
    const __half *val;
    uint32_t *ftg; // [totcol][K]: half2 pair (f[k][col-OFF], f[k][col+1-OFF])
    long long totcol;
    int K, KP, OFF, segcap;
};

__device__ __forceinline__ void block_scan_excl(int v, int &excl, int &total, int *s_wsum)
{
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    int incl = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int y = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += y;
    }
    if (lane == 31) s_wsum[wid] = incl;
    __syncthreads();
    if (wid == 0) {
        int x = lane < PREP_THREADS / 32 ? s_wsum[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int y = __shfl_up_sync(0xffffffffu, x, o);
            if (lane >= o) x += y;
        }
        if (lane < PREP_THREADS / 32) s_wsum[lane] = x;
    }
    __syncthreads();
    excl = (wid ? s_wsum[wid - 1] : 0) + incl - v;
    total = s_wsum[PREP_THREADS / 32 - 1];
    __syncthreads();
}

__global__ void __launch_bounds__(PREP_THREADS)
    seg_prep_kernel(const long long *__restrict__ idx, const long long *__restrict__ off, Seg *__restrict__ segs,
                    int *__restrict__ nseg_out, int Hi, int Wi, const FtParams F)
{
    __shared__ int s_n[PREP_THREADS], s_hi[PREP_THREADS], s_wi[PREP_THREADS];
    __shared__ int s_wsum[PREP_THREADS / 32];
    const int segcap = F.segcap;
    const int ho = blockIdx.x;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const long long a = off[ho], b = off[ho + 1];
    int cnt = 0;
    bool bad = false;
    for (long long base = a; base < b; base += PREP_THREADS * 4) {
        long long n0 = base + (long long)tid * 4;
        int hh[4], ww[4], fl[4];
        int ph = -1, pw = -2;
        if (n0 - 1 >= a && n0 - 1 < b) {
            ph = (int)idx[2 * (n0 - 1)];
            pw = (int)idx[2 * (n0 - 1) + 1];
        }
        int c = 0;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            long long n = n0 + q;
            fl[q] = 0;
            hh[q] = 0;
            ww[q] = 0;
            if (n < b) {
                int h = (int)idx[2 * n], w = (int)idx[2 * n + 1];
                if (h < 0 || h >= Hi || w < 0 || w >= Wi) bad = true;
                fl[q] = (n == a) || h != ph || w != pw + 1;
                hh[q] = h;
                ww[q] = w;
                ph = h;
                pw = w;
            }
            c += fl[q];
        }
        int incl = c;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int y = __shfl_up_sync(0xffffffffu, incl, o);
            if (lane >= o) incl += y;
        }
        if (lane == 31) s_wsum[wid] = incl;
        __syncthreads();
        if (wid == 0) {
            int x = lane < PREP_THREADS / 32 ? s_wsum[lane] : 0;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                int y = __shfl_up_sync(0xffffffffu, x, o);
                if (lane >= o) x += y;
            }
            if (lane < PREP_THREADS / 32) s_wsum[lane] = x;
        }
        __syncthreads();
        int excl = cnt + (wid ? s_wsum[wid - 1] : 0) + incl - c;
        int total = s_wsum[PREP_THREADS / 32 - 1];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            if (fl[q]) {
                if (excl < PREP_THREADS) {
                    s_n[excl] = (int)(n0 + q - a);
                    s_hi[excl] = hh[q];
                    s_wi[excl] = ww[q];
                }
                ++excl;
            }
        }
        cnt += total;
        __syncthreads();
    }
    bad = __syncthreads_or(bad);
    if (bad || cnt > PREP_THREADS || cnt > segcap) {
        if (tid == 0) nseg_out[ho] = -1;
        return;
    }
    const int nrow = (int)(b - a);
    const int i = tid;
    int emit = 0;
    Seg S;
    if (i < cnt) {
        int hi = s_hi[i];
        int len = (i + 1 < cnt ? s_n[i + 1] : nrow) - s_n[i];
        int F = i;
        while (F > 0 && s_hi[F - 1] == hi) --F;
        int L = i;
        while (L + 1 < cnt && s_hi[L + 1] == hi) ++L;
        int lenF = (F + 1 < cnt ? s_n[F + 1] : nrow) - s_n[F];
        int lenL = (L + 1 < cnt ? s_n[L + 1] : nrow) - s_n[L];
        bool merge = (L > F) && s_wi[F] == 0 && (s_wi[L] + lenL == Wi);
        emit = !(merge && i == F);
        S.hi = hi;
        S.pad0 = 0;
        S.pad1 = 0;
        if (merge && i == L) {
            S.s = s_wi[L];
            S.w = lenL + lenF;
            S.n1 = (int)a + s_n[L];
            S.len1 = lenL;
            S.n2 = (int)a + s_n[F];
        } else {
            S.s = s_wi[i];
            S.w = len;
            S.n1 = (int)a + s_n[i];
            S.len1 = len;
            S.n2 = 0;
        }
    }
    const int OFF = F.OFF;
    const int CL = emit ? ((S.w + 2 * OFF + 8 + 7) & ~7) : 0;
    int pos, nout, cpos, ctot;
    block_scan_excl(emit, pos, nout, s_wsum);
    block_scan_excl(CL, cpos, ctot, s_wsum);
    // row base column in the global f^T (closed form, 8-aligned, disjoint across rows)
    const long long rb = ((a + (long long)ho * ((long long)segcap * (2 * OFF + 16) + 8)) + 7) & ~7LL;
    if (emit) {
        S.pad0 = (int)(rb + cpos);
        segs[a + pos] = S;
    }
    if (tid == 0) nseg_out[ho] = nout;
}

// ------------------------------------------------------------------------------------------------
// kernel 2b: fill the global f^T of every segment; FILL_SPLIT blocks per row share the row's columns
// ------------------------------------------------------------------------------------------------
#define FILL_SPLIT 8
__device__ __forceinline__ void load_val_row(const __half *__restrict__ val, long long n, int KP, int K, bool in,
                                             uint16_t (&v)[16])
{
    if (in && KP == 16) {
        const uint4 *r = reinterpret_cast<const uint4 *>(val + n * 16);
        const uint4 x0 = __ldg(r), x1 = __ldg(r + 1);
        const uint32_t w[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
        for (int k = 0; k < 16; ++k) v[k] = (uint16_t)(w[k >> 1] >> ((k & 1) * 16));
    } else {
#pragma unroll
        for (int k = 0; k < 16; ++k) v[k] = (in && k < K) ? __half_as_ushort(__ldg(val + n * KP + k)) : (uint16_t)0;
    }
}

__global__ void __launch_bounds__(256) ft_fill_kernel(const long long *__restrict__ off, const Seg *__restrict__ segs,
                                                      const int *__restrict__ nseg_in, const FtParams F)
{
    __shared__ Seg s_sg[SEGCAP_PREP];
    const int ho = blockIdx.x / FILL_SPLIT, part = blockIdx.x % FILL_SPLIT;
    const int nseg = nseg_in[ho];
    if (nseg <= 0 || nseg > SEGCAP_PREP) return;
    const long long a = off[ho];
    for (int i = threadIdx.x; i < nseg; i += blockDim.x) s_sg[i] = segs[a + i];
    __syncthreads();
    const int OFF = F.OFF, K = F.K, KP = F.KP;
    const int c0 = s_sg[0].pad0;
    const int total = s_sg[nseg - 1].pad0 + ((s_sg[nseg - 1].w + 2 * OFF + 8 + 7) & ~7) - c0;
    const int beg = (int)((long long)total * part / FILL_SPLIT);
    const int end = (int)((long long)total * (part + 1) / FILL_SPLIT);
    for (int q0 = beg + threadIdx.x; q0 < end; q0 += 2 * blockDim.x) {
        uint32_t pk[2][16];
        bool ok[2];
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const int q = q0 + u * blockDim.x;
            ok[u] = q < end;
            int lo = 0, hi = nseg - 1; // segment containing column q (segments are back to back)
            while (lo < hi) {
                const int mid = (lo + hi + 1) >> 1;
                if (s_sg[mid].pad0 - c0 <= q)
                    lo = mid;
                else
                    hi = mid - 1;
            }
            const Seg sg = s_sg[lo];
            const int j0 = q - (sg.pad0 - c0) - OFF, j1 = j0 + 1;
            const bool in0 = ok[u] && j0 >= 0 && j0 < sg.w, in1 = ok[u] && j1 >= 0 && j1 < sg.w;
            const long long n0 = in0 ? ((j0 < sg.len1) ? (long long)sg.n1 + j0 : (long long)sg.n2 + (j0 - sg.len1)) : 0;
            const long long n1 = in1 ? ((j1 < sg.len1) ? (long long)sg.n1 + j1 : (long long)sg.n2 + (j1 - sg.len1)) : 0;
            uint16_t va[16], vb[16];
            load_val_row(F.val, n0, KP, K, in0, va);
            load_val_row(F.val, n1, KP, K, in1, vb);
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                const uint32_t lo16 = in0 ? va[k] : 0u, hi16 = in1 ? vb[k] : 0u;
                pk[u][k] = lo16 | (hi16 << 16);
            }
        }
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            if (!ok[u]) continue;
            uint32_t *dst = F.ftg + ((long long)c0 + q0 + u * blockDim.x) * K;
#pragma unroll
            for (int k = 0; k < 16; ++k)
                if (k < K) dst[k] = pk[u][k];
        }
    }
}

// ------------------------------------------------------------------------------------------------
// kernel 3 (fallback): direct CUDA-core evaluation for rows flagged by seg_prep (rowflag[ho] < 0) or
// for all rows when rowflag == nullptr.  One block per output row.
// ------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
    generic_kernel(const __half *__restrict__ inp, const long long *__restrict__ idx, const __half *__restrict__ val,
                   const long long *__restrict__ off, __half *__restrict__ out, const int *__restrict__ rowflag, int BC,
                   int Hi, int Wi, int Ho, int Wo, int K, int KP, int p)
{
    const int ho = blockIdx.x;
    if (rowflag && rowflag[ho] >= 0) return;
    const long long a = off[ho], b = off[ho + 1];
    const long long total = (long long)BC * Wo;
    for (long long q = threadIdx.x; q < total; q += blockDim.x) {
        int bc = (int)(q / Wo), wo = (int)(q % Wo);
        float acc[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) acc[k] = 0.f;
        const __half *xr = inp + (size_t)bc * Hi * Wi;
        for (long long n = a; n < b; ++n) {
            int hi = (int)idx[2 * n], wi = (int)idx[2 * n + 1];
            float x = __half2float(xr[(size_t)hi * Wi + (int)(((long long)wi + (long long)wo * p) % Wi)]);
#pragma unroll
            for (int k = 0; k < 16; ++k)
                if (k < K) acc[k] += __half2float(val[n * KP + k]) * x;
        }
#pragma unroll
        for (int k = 0; k < 16; ++k)
            if (k < K) out[(((size_t)bc * K + k) * Ho + ho) * Wo + wo] = __float2half_rn(acc[k]);
    }
}

// ------------------------------------------------------------------------------------------------
// kernel 4: main warp-specialized Toeplitz WGMMA kernel
// ------------------------------------------------------------------------------------------------
#define MAXSTAGE 8
#define KBMAX 32
#define SEGCAP 128
#define FT_BYTES_BUDGET 12288

struct Piece {
    short e, b0, nb, x0, fo, contig, rows, blk;
};
struct StageD {
    short pb, pe, nblk, fin;
    int bytes, pad;
};
#define PIECECAP 192
#define STAGECAP 128

struct MainParams {
    const __half *val;
    const int *nseg;
    const Seg *segs;
    const long long *off;
    const uint4 *xt;
    const uint32_t *ftg;
    long long totcol;
    __half *out;
    int BC, G, Hi, Wi, Ho, Wo, p, K, KP, Tw;
    int CT, WT, WTC, NWC, units;
    int ROWS, FJS, OFF, nstage;
    uint32_t ft_off, tab_off, stage_bytes, stg_off, stg_bytes, seg_off, piece_off, stage_off, bar_off;
    int tma_store, boxR, nbox; // epilogue through TMA tensor stores of boxR staging rows each
};

template <int NG, int T, int KF, int PF>
__global__ void __launch_bounds__(384, 1) disco_main_kernel(const MainParams P, const __grid_constant__ CUtensorMap omap)
{
    extern __shared__ __align__(1024) uint8_t smem[];
    constexpr int N = 8 * NG;
    constexpr int TT = 2 * T;
    const int Tw = KF ? (64 / KF) : P.Tw;
    const int K = KF ? KF : P.K;
    const int p = PF ? PF : P.p;
    const int OFF = (KF && PF) ? (((Tw - 1) * p + 7) & ~7) : P.OFF;
    const int ROWS = P.ROWS, FJS = P.FJS;

    const uint32_t sbase = smem_u32(smem);
    const uint32_t bar_full = sbase + P.bar_off;
    const int NSTAGE = P.nstage;
    const uint32_t bar_empty = bar_full + MAXSTAGE * 8;
    Seg *s_seg = reinterpret_cast<Seg *>(smem + P.seg_off);
    Piece *s_piece = reinterpret_cast<Piece *>(smem + P.piece_off);
    int *s_plan = reinterpret_cast<int *>(smem + P.bar_off + 2 * MAXSTAGE * 8);

    if (threadIdx.x == 0) {
        for (int s = 0; s < NSTAGE; ++s) {
            mbar_init(bar_full + 8 * s, 128);
            mbar_init(bar_empty + 8 * s, 256);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();

    const int wgi = threadIdx.x >> 7;
    if (wgi == 0) {
        // ============================== producer warpgroup ==============================
        reg_dealloc<40>();
        const int ptid = threadIdx.x;
        StageD *s_stage = reinterpret_cast<StageD *>(smem + P.stage_off);
        int it = 0;
        // greedy stage planner (thread 0 only); independent of the wo tile, so a row is planned once per unit
        auto plan_stage = [&](int nseg, int &e, int &b, int &np, int npcap, StageD &sd) {
            int nblk = 0, xr = 0, fc = 0;
            uint32_t bytes = 0;
            const int pb = np;
            const int halo = (TT - 1) * Tw * p;
            while (e < nseg && nblk < KBMAX && np < npcap) {
                const Seg sg = s_seg[e];
                const int U = sg.w + (Tw - 1) * p;
                const int nbe = (U + 7) >> 3;
                const int remb = nbe - b;
                const int availR = ROWS - xr, availF = FJS - fc;
                int nbc = (availR - halo) / 8;
                if (availR - halo < 0) nbc = -1;
                const int nbs = availR / (8 * TT);
                const int nbr = max(nbc, nbs);
                int nbf = availF - OFF;
                nbf = nbf < 0 ? -1 : nbf / 8;
                const int nb = min(min(remb, KBMAX - nblk), min(nbr, nbf));
                if (nb <= 0) break;
                const int rc = 8 * nb + halo, rsep = 8 * nb * TT;
                const int contig = rc <= rsep;
                const int rows = contig ? rc : rsep;
                const int FJ = 8 * nb + OFF;
                Piece pc;
                pc.e = (short)e;
                pc.b0 = (short)b;
                pc.nb = (short)nb;
                pc.x0 = (short)xr;
                pc.fo = (short)fc;
                pc.contig = (short)contig;
                pc.rows = (short)rows;
                pc.blk = (short)nblk;
                s_piece[np++] = pc;
                nblk += nb;
                xr += rows;
                fc += FJ;
                bytes += (uint32_t)(NG * rows * 16);
                b += nb;
                if (b == nbe) {
                    ++e;
                    b = 0;
                }
            }
            sd.pb = (short)pb;
            sd.pe = (short)np;
            sd.nblk = (short)nblk;
            sd.fin = (short)(e == nseg);
            sd.bytes = (int)bytes;
        };
        // all 128 producer threads: issue the bulk copies + block table of one stage, then arrive
        auto issue_stage = [&](const StageD sd, int s, int base_shift, int ct) {
            const uint32_t xs = sbase + s * P.stage_bytes;
            const uint32_t fts_p = xs + P.ft_off;
            int *tab = reinterpret_cast<int *>(smem + s * P.stage_bytes + P.tab_off);
            constexpr int CPP = NG;
            const int nq = (sd.pe - sd.pb) * CPP;
            // bulk copies issue one lane at a time per warp: spread the tasks over the 4 producer warps
            for (int q = ((ptid & 31) << 2) | (ptid >> 5); q < nq; q += 128) {
                const int pi = sd.pb + q / CPP, sub = q % CPP;
                const Piece pc = s_piece[pi];
                const Seg sg = s_seg[pc.e];
                if (sub < NG) {
                    const int nwin = pc.contig ? 1 : TT;
                    const int L = pc.contig ? pc.rows : 8 * pc.nb;
                    int P0;
                    if constexpr (KF == 9 && PF == 1) {
                        // pscale=1 implies Wi=Wo.  The tile base is already in [0, Wi), and a
                        // segment contributes at most one longitude ring, so two bounded
                        // subtractions replace the integer remainder in this hot loop.
                        uint64_t z = (uint32_t)sg.s + (uint32_t)base_shift + 8ull * (uint16_t)pc.b0;
#pragma unroll
                        for (int r = 0; r < 2; ++r)
                            if (z >= (uint32_t)P.Wi) z -= (uint32_t)P.Wi;
                        P0 = (int)z;
                    } else {
                        P0 = (sg.s + base_shift + 8 * pc.b0) % P.Wi;
                    }
                    const int cg = min(ct * NG + sub, P.G - 1);
                    const uint4 *src = P.xt + ((size_t)cg * P.Hi + sg.hi) * P.Wi;
                    int win_pos = P0;
                    for (int wv = 0; wv < nwin; ++wv) {
                        int pos;
                        if constexpr (KF == 9 && PF == 1) {
                            pos = win_pos;
                        } else {
                            pos = (P0 + wv * Tw * p) % P.Wi;
                        }
                        uint32_t dst = xs + (uint32_t)(sub * ROWS + pc.x0 + wv * L) * 16u;
                        int remr = L;
                        while (remr > 0) {
                            const int len = min(remr, P.Wi - pos);
                            bulk_g2s(dst, src + pos, (uint32_t)len * 16u, bar_full + 8 * s);
                            dst += (uint32_t)len * 16u;
                            remr -= len;
                            pos = 0;
                        }
                        if constexpr (KF == 9 && PF == 1) {
                            win_pos += 7;
                            if (win_pos >= P.Wi) win_pos -= P.Wi;
                        }
                    }
                }
                const int TS = pc.contig ? Tw * p : 8 * pc.nb;
                for (int bb = sub; bb < pc.nb; bb += CPP) {
                    tab[2 + 2 * (pc.blk + bb)] = (pc.x0 + 8 * bb) | (TS << 16);
                    tab[2 + 2 * (pc.blk + bb) + 1] = (pc.fo + 8 * bb) * K * 4;
                }
            }
            // filter rows (f^T) of every piece: classic 16-byte cp.async spread over all producer threads
            // (every piece's f^T run is a whole number of 16-byte chunks: columns come in multiples of 8)
            for (int pi = sd.pb + (ptid >> 5); pi < sd.pe; pi += 4) { // one piece per producer warp
                const Piece pc = s_piece[pi];
                const int pad0 = s_seg[pc.e].pad0;
                const int nch = ((8 * pc.nb + OFF) * K) >> 2;
                const uint32_t *fsrc = P.ftg + ((long long)pad0 + 8 * pc.b0) * K;
                const uint32_t fdst = fts_p + (uint32_t)(pc.fo * K) * 4u;
                for (int c = ptid & 31; c < nch; c += 32) cp_async16(fdst + 16u * (uint32_t)c, fsrc + 4 * c);
            }
            cp_async_mbar_arrive(bar_full + 8 * s);
            if (ptid == 0) {
                tab[0] = sd.nblk;
                tab[1] = sd.fin;
                mbar_arrive_expect_tx(bar_full + 8 * s, (uint32_t)sd.bytes);
            } else {
                mbar_arrive(bar_full + 8 * s);
            }
        };

        for (int unit = blockIdx.x; unit < P.units; unit += gridDim.x) {
            const int per_row = P.CT * P.NWC;
            const int rr = unit / per_row, rem = unit - rr * per_row;
            const int ho = (rr & 1) ? (P.Ho - 1 - (rr >> 1)) : (rr >> 1);
            const int ct = rem / P.NWC, wc = rem - ct * P.NWC;
            const int nseg = P.nseg[ho];
            if (nseg < 0) continue;
            const long long a = P.off[ho];
            named_bar(1, 128);
            for (int i = ptid; i < nseg; i += 128) s_seg[i] = P.segs[a + i];
            named_bar(1, 128);
            // thread-0 planner state for this unit; the first wo tile plans stage by stage and fills the cache
            int pe_ = 0, pb_ = 0, pnp = 0, pns = 0, pok = 1;
            const int wt0 = wc * P.WTC;
            const int wt_end = min(P.WT, (wc + 1) * P.WTC);
            int base_fast = wt0 * (TT * 7);
            for (int wt = wt0; wt < wt_end; ++wt) {
                int base_shift;
                if constexpr (KF == 9 && PF == 1) {
                    // Since Wi=Wo and wt<P.WT=ceil(Wo/(TT*7)), every consumed tile base
                    // is strictly below Wi; no modulo is needed.
                    base_shift = base_fast;
                } else {
                    base_shift = (int)(((long long)wt * TT * Tw * p) % P.Wi);
                }
                const bool use_cache = (wt > wt0) && s_plan[0];
                if (use_cache) {
                    const int ns = s_plan[1];
                    for (int st = 0; st < ns; ++st) {
                        const int s = it % NSTAGE;
                        const uint32_t ph = (it / NSTAGE) & 1;
                        mbar_wait(bar_empty + 8 * s, ph ^ 1);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        issue_stage(s_stage[st], s, base_shift, ct);
                        ++it;
                    }
                } else {
                    if (ptid == 0) {
                        pe_ = 0;
                        pb_ = 0;
                    }
                    const bool first = (wt == wt0);
                    while (true) {
                        const int s = it % NSTAGE;
                        const uint32_t ph = (it / NSTAGE) & 1;
                        mbar_wait(bar_empty + 8 * s, ph ^ 1);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        if (ptid == 0) {
                            StageD sd;
                            const bool cache = first && pok && (pnp + 2 * KBMAX <= PIECECAP) && (pns < STAGECAP - 1);
                            if (cache) {
                                plan_stage(nseg, pe_, pb_, pnp, PIECECAP, sd);
                                s_stage[pns] = sd;
                                s_plan[2] = pns++;
                            } else {
                                if (first) pok = 0;
                                int npx = PIECECAP - KBMAX;
                                plan_stage(nseg, pe_, pb_, npx, PIECECAP, sd);
                                s_stage[STAGECAP - 1] = sd;
                                s_plan[2] = STAGECAP - 1;
                            }
                            if (sd.fin) {
                                s_plan[0] = pok;
                                s_plan[1] = pns;
                            }
                        }
                        named_bar(1, 128);
                        const StageD sd = s_stage[s_plan[2]];
                        issue_stage(sd, s, base_shift, ct);
                        named_bar(1, 128);
                        ++it;
                        if (sd.fin) break;
                    }
                }
                if constexpr (KF == 9 && PF == 1) {
                    if (wt + 1 < wt_end) base_fast += TT * 7;
                }
            }
        }
    } else {
        // ============================== consumer warpgroups ==============================
        reg_alloc<232>();
        const int wg = wgi - 1;
        const int ltid = threadIdx.x & 127;
        const int w4 = ltid >> 5, lane = ltid & 31;
        const int g = lane >> 2, t = lane & 3;
        const int KR = K * Tw;
        const int r0 = 16 * w4 + g, r1 = r0 + 8;
        const bool v0 = r0 < KR, v1 = r1 < KR;
        const int k0 = v0 ? r0 / Tw : 0, wl0 = v0 ? r0 % Tw : 0;
        const int k1 = v1 ? r1 / Tw : 0, wl1 = v1 ? r1 % Tw : 0;
        // byte offsets inside the stage f^T region: pair (column 2t + OFF - wl*p, k) of a [col][K] uint32 array
        const uint32_t ro0 = 4u * (uint32_t)((2 * t + OFF - wl0 * p) * K + k0);
        const uint32_t ro1 = 4u * (uint32_t)((2 * t + OFF - wl1 * p) * K + k1);
        const uint32_t m0 = v0 ? 0xffffffffu : 0u, m1 = v1 ? 0xffffffffu : 0u;
        const uint32_t dhi = (uint32_t)ROWS; // SBO >> 4 (group stride ROWS*16 bytes)

        float acc[T][N / 2];
        uint32_t A0[4], A1[4];
        int it = 0;
        int pend = -1;
        int ap = 0;
        __half *stg = reinterpret_cast<__half *>(smem + P.stg_off + wg * P.stg_bytes);
        const int TTw = T * Tw;

        for (int unit = blockIdx.x; unit < P.units; unit += gridDim.x) {
            const int per_row = P.CT * P.NWC;
            const int rr = unit / per_row, rem = unit - rr * per_row;
            const int ho = (rr & 1) ? (P.Ho - 1 - (rr >> 1)) : (rr >> 1);
            const int ct = rem / P.NWC, wc = rem - ct * P.NWC;
            if (P.nseg[ho] < 0) continue;
            const int wt_end = min(P.WT, (wc + 1) * P.WTC);
            for (int wt = wc * P.WTC; wt < wt_end; ++wt) {
#pragma unroll
                for (int tt = 0; tt < T; ++tt)
#pragma unroll
                    for (int j = 0; j < N / 2; ++j) acc[tt][j] = 0.f;

                while (true) {
                    const int s = it % NSTAGE;
                    const uint32_t ph = (it / NSTAGE) & 1;
                    mbar_wait(bar_full + 8 * s, ph);
                    const int *tab = reinterpret_cast<const int *>(smem + s * P.stage_bytes + P.tab_off);
                    const int nblk = tab[0], fin = tab[1];
                    const int2 *ent = reinterpret_cast<const int2 *>(tab + 2);
                    const uint32_t xs = sbase + s * P.stage_bytes;
                    const uint32_t fts = sbase + s * P.stage_bytes + P.ft_off;
                    const int nks = (nblk + 1) >> 1;

                    auto step = [&](int ks, uint32_t(&A)[4], uint32_t(&Aother)[4]) {
                        const int2 ea = ent[2 * ks];
                        const bool hasb = (2 * ks + 1) < nblk;
                        const int2 eb = hasb ? ent[2 * ks + 1] : ea;
                        const uint32_t fa = fts + (uint32_t)ea.y, fb = fts + (uint32_t)eb.y;
                        const uint32_t mb = hasb ? 0xffffffffu : 0u;
                        A[0] = lds32(fa + ro0) & m0;
                        A[1] = lds32(fa + ro1) & m1;
                        A[2] = lds32(fb + ro0) & (m0 & mb);
                        A[3] = lds32(fb + ro1) & (m1 & mb);
                        const uint32_t xa = (uint32_t)ea.x & 0xffffu, tsa = (uint32_t)ea.x >> 16;
                        const uint32_t xb = (uint32_t)eb.x & 0xffffu, tsb = (uint32_t)eb.x >> 16;
                        // desc lo word = (addr>>4) | (LBO>>4)<<16 ; both fields are linear in the tile index
                        const uint32_t dstep = tsa + ((tsb - tsa) << 16);
                        uint32_t dlo = (xs >> 4) + xa + ((xb - xa) << 16) + (uint32_t)(wg * T) * dstep;
                        wg_fence();
#pragma unroll
                        for (int tt = 0; tt < T; ++tt) {
                            const uint64_t desc = ((uint64_t)dhi << 32) | (uint64_t)dlo;
                            wgmma_rs<N>(acc[tt], A[0], A[1], A[2], A[3], desc);
                            dlo += dstep;
                        }
                        wg_commit();
                        wg_wait<1>();
#pragma unroll
                        for (int q = 0; q < 4; ++q) fence_u32(Aother[q]);
                        if (pend >= 0) { // all groups of the previous stage are complete now
                            mbar_arrive(bar_empty + 8 * pend);
                            pend = -1;
                        }
                    };

                    // A register sets alternate globally (not per stage): the set loaded now was last read by the
                    // group two steps back, which the previous step's wait<1> retired.
                    for (int ks = 0; ks < nks; ++ks) {
                        if (ap == 0)
                            step(ks, A0, A1);
                        else
                            step(ks, A1, A0);
                        ap ^= 1;
                    }
                    wg_wait<0>();
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        fence_u32(A0[q]);
                        fence_u32(A1[q]);
                    }
                    if (pend >= 0) mbar_arrive(bar_empty + 8 * pend);
                    pend = -1;
                    mbar_arrive(bar_empty + 8 * s);
                    ++it;
                    if (fin) break;
                }
#pragma unroll
                for (int tt = 0; tt < T; ++tt)
#pragma unroll
                    for (int j = 0; j < N / 2; ++j) fence_f32(acc[tt][j]);

                // ---- epilogue: registers -> smem [c][k][wo_local] -> global (TMA tensor store or vector stores)
                if (P.tma_store && ltid == 0) bulk_wait_read0(); // previous tile's store has read the staging
                named_bar(2 + wg, 128);
                {
                    const int KTT = K * TTw;
                    const int sb0 = (2 * t * K + k0) * TTw + wl0;
                    const int sb1 = (2 * t * K + k1) * TTw + wl1;
#pragma unroll
                    for (int tt = 0; tt < T; ++tt) {
#pragma unroll
                        for (int j = 0; j < N / 8; ++j) {
                            const int cb = 8 * j * KTT + tt * Tw;
                            if (v0) {
                                stg[sb0 + cb] = __float2half_rn(acc[tt][4 * j + 0]);
                                stg[sb0 + cb + KTT] = __float2half_rn(acc[tt][4 * j + 1]);
                            }
                            if (v1) {
                                stg[sb1 + cb] = __float2half_rn(acc[tt][4 * j + 2]);
                                stg[sb1 + cb + KTT] = __float2half_rn(acc[tt][4 * j + 3]);
                            }
                        }
                    }
                }
                if (P.tma_store) {
                    // generic-proxy staging writes -> async-proxy reads of the TMA store
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    named_bar(2 + wg, 128);
                    if (ltid == 0) {
                        const int row0 = ct * N * K;
                        const int wbase = wt * TT * Tw + wg * TTw;
                        const uint32_t sa = smem_u32(stg);
                        for (int b = 0; b < P.nbox; ++b)
                            tma_store_3d(&omap, sa + (uint32_t)(b * P.boxR * TTw * 2), wbase, ho, row0 + b * P.boxR);
                        bulk_commit();
                    }
                } else {
                    named_bar(2 + wg, 128);
                    {
                        // rows of the staging tile are (c*K + k); global row index = ch0*K + row, so
                        // address = out + row * Ho*Wo + (ch0*K*Ho + ho)*Wo + wbase + chunk*CH  (no div/mod per element)
                        const int CH = ((TTw & 7) == 0 && (P.Wo & 7) == 0) ? 8 :
                            ((TTw & 3) == 0 && (P.Wo & 3) == 0)            ? 4 :
                            ((TTw & 1) == 0 && (P.Wo & 1) == 0)            ? 2 :
                                                                             1;
                        const int CPR = TTw / CH;
                        const int ch0 = ct * N;
                        const int wbase = wt * TT * Tw + wg * TTw;
                        const int nrow_valid = min(N, P.BC - ch0) * K;
                        int cvalid = (P.Wo - wbase + CH - 1) / CH;
                        cvalid = cvalid < 0 ? 0 : (cvalid > CPR ? CPR : cvalid);
                        const size_t HoWo = (size_t)P.Ho * P.Wo;
                        __half *gbase = P.out + ((size_t)ch0 * K * P.Ho + ho) * P.Wo + wbase;
                        const int nchunk = nrow_valid * CPR;
                        const int q128 = 128 / CPR, r128 = 128 - q128 * CPR;
                        int row = ltid / CPR, chk = ltid - row * CPR;
                        if constexpr (KF == 9 && PF == 1) {
                            if (wbase + TTw <= P.Wo) {
                                for (int lin = ltid; lin < nchunk; lin += 128) {
                                    __half *gp = gbase + (size_t)row * HoWo + chk * 8;
                                    const __half *sp = stg + row * TTw + chk * 8;
                                    *reinterpret_cast<uint4 *>(gp) = *reinterpret_cast<const uint4 *>(sp);
                                    chk += r128;
                                    row += q128;
                                    if (chk >= CPR) {
                                        chk -= CPR;
                                        ++row;
                                    }
                                }
                            } else {
                                for (int lin = ltid; lin < nchunk; lin += 128) {
                                    if (chk < cvalid) {
                                        __half *gp = gbase + (size_t)row * HoWo + chk * 8;
                                        const __half *sp = stg + row * TTw + chk * 8;
                                        *reinterpret_cast<uint4 *>(gp) = *reinterpret_cast<const uint4 *>(sp);
                                    }
                                    chk += r128;
                                    row += q128;
                                    if (chk >= CPR) {
                                        chk -= CPR;
                                        ++row;
                                    }
                                }
                            }
                        } else {
                            for (int lin = ltid; lin < nchunk; lin += 128) {
                                if (chk < cvalid) {
                                    __half *gp = gbase + (size_t)row * HoWo + chk * CH;
                                    const __half *sp = stg + row * TTw + chk * CH;
                                    if (CH == 8)
                                        *reinterpret_cast<uint4 *>(gp) = *reinterpret_cast<const uint4 *>(sp);
                                    else if (CH == 4)
                                        *reinterpret_cast<uint2 *>(gp) = *reinterpret_cast<const uint2 *>(sp);
                                    else if (CH == 2)
                                        *reinterpret_cast<uint32_t *>(gp) = *reinterpret_cast<const uint32_t *>(sp);
                                    else
                                        *gp = *sp;
                                }
                                chk += r128;
                                row += q128;
                                if (chk >= CPR) {
                                    chk -= CPR;
                                    ++row;
                                }
                            }
                        }
                    }
                }
            } // wt
        }
        if (P.tma_store && ltid == 0) bulk_wait0(); // staging must outlive the last TMA store
    }
}

// ------------------------------------------------------------------------------------------------
// host
// ------------------------------------------------------------------------------------------------
static void *g_ws = nullptr;
static size_t g_ws_size = 0;
static void *ws_get(size_t bytes, cudaStream_t st)
{
    if (bytes > g_ws_size) {
        if (g_ws) {
            cudaStreamSynchronize(st);
            cudaFree(g_ws);
        }
        size_t sz = bytes + (bytes >> 3);
        if (cudaMalloc(&g_ws, sz) != cudaSuccess) {
            g_ws = nullptr;
            g_ws_size = 0;
            return nullptr;
        }
        g_ws_size = sz;
    }
    return g_ws;
}

static int g_num_sms = 0;

template <int NG, int T> static bool prepare_main(MainParams &P, uint32_t &smem)
{
    constexpr int TT = 2 * T;
    if constexpr (NG == 4) {
        if (P.G >= 100) {
            P.ROWS = 501;
            P.FJS = 192;
        } else {
            P.ROWS = 463;
            P.FJS = 256;
        }
    } else if constexpr (NG == 3) {
        P.ROWS = 743;
        P.FJS = 208;
    } else {
        const int X_BYTES = (NG == 9) ? 32768 : 24576;
        P.ROWS = X_BYTES / (NG * 16);
        if ((P.ROWS & 1) == 0) P.ROWS -= 1; // odd row count: SBO = ROWS*16 is not a multiple of 128 B
        P.FJS = (FT_BYTES_BUDGET / (4 * P.K)) & ~7;
    }
    P.OFF = (((P.Tw - 1) * P.p) + 7) & ~7;
    if (P.ROWS < 8 * TT) return false;
    if (8 + P.OFF > P.FJS) return false;
    if ((long long)P.Tw * P.p >= 65536) return false;
    P.CT = (P.G + NG - 1) / NG;
    P.WT = (P.Wo + TT * P.Tw - 1) / (TT * P.Tw);
    {
        // work unit = (row, channel tile, chunk of WTC wo tiles); the row's stage plan is reused across the chunk
        const long long items = (long long)P.Ho * P.CT * P.WT;
        long long wtc = items / (32LL * (g_num_sms > 0 ? g_num_sms : 132));
        if (wtc < 1) wtc = 1;
        if (wtc > P.WT) wtc = P.WT;
        P.WTC = (int)wtc;
        P.NWC = (P.WT + P.WTC - 1) / P.WTC;
        const long long units = (long long)P.Ho * P.CT * P.NWC;
        if (units > 0x7fffffff) return false;
        P.units = (int)units;
    }
    const uint32_t xb = (uint32_t)(NG * P.ROWS * 16);
    P.ft_off = (xb + 127u) & ~127u;
    const uint32_t ftb = (uint32_t)(4 * P.K * P.FJS); // [FJS][K] uint32 pairs
    P.tab_off = (P.ft_off + ftb + 15u) & ~15u;
    P.stage_bytes = (P.tab_off + 8 + 8 * KBMAX + 127u) & ~127u;
    P.stg_bytes = ((uint32_t)(8 * NG * P.K * T * P.Tw * 2) + 127u) & ~127u;
    const uint32_t fixed = 2 * P.stg_bytes + SEGCAP * sizeof(Seg) + PIECECAP * sizeof(Piece) + STAGECAP * sizeof(StageD)
        + 2 * MAXSTAGE * 8 + 64;
    if (fixed + 2 * P.stage_bytes > 232448) return false;
    P.nstage = (int)((232448 - fixed) / P.stage_bytes);
    if (P.nstage > MAXSTAGE) P.nstage = MAXSTAGE;
    P.stg_off = P.nstage * P.stage_bytes;
    P.seg_off = P.stg_off + 2 * P.stg_bytes;
    P.piece_off = P.seg_off + SEGCAP * sizeof(Seg);
    P.stage_off = P.piece_off + PIECECAP * sizeof(Piece);
    P.bar_off = P.stage_off + STAGECAP * sizeof(StageD);
    smem = P.bar_off + 2 * MAXSTAGE * 8 + 64;
    return smem <= 232448;
}

template <int NG, int T, int KF, int PF>
static void launch_main_impl(const MainParams &P, const CUtensorMap &omap, uint32_t smem, cudaStream_t st)
{
    auto kern = disco_main_kernel<NG, T, KF, PF>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    int grid = P.units < g_num_sms ? P.units : g_num_sms;
    if (grid <= 0) return;
    kern<<<grid, 384, smem, st>>>(P, omap);
}

// out viewed as [B*C*K][Ho][Wo]; one box = TTw longitudes x 1 latitude x boxR (channel, k) rows of a
// consumer warpgroup's staging tile.  Falls back to vector stores when the shapes do not allow it.
template <int NG, int T> static void setup_out_map(MainParams &P, CUtensorMap &omap)
{
    P.tma_store = 0;
    P.boxR = 0;
    P.nbox = 0;
    const int TTw = T * P.Tw;
    const int rows = 8 * NG * P.K;
    const int nbox = (rows + 255) / 256;
    if (rows % nbox) return;
    const int boxR = rows / nbox;
    if ((TTw * 2) % 16 || TTw > 256 || ((long long)P.Wo * 2) % 16 || ((long long)boxR * TTw * 2) % 128) return;
    cuuint64_t dims[3] = {(cuuint64_t)P.Wo, (cuuint64_t)P.Ho, (cuuint64_t)P.BC * P.K};
    cuuint64_t strides[2] = {(cuuint64_t)P.Wo * 2, (cuuint64_t)P.Ho * P.Wo * 2};
    cuuint32_t box[3] = {(cuuint32_t)TTw, 1u, (cuuint32_t)boxR};
    cuuint32_t estr[3] = {1u, 1u, 1u};
    CUresult r = cuTensorMapEncodeTiled(&omap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, (void *)P.out, dims, strides, box,
                                        estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) return;
    P.tma_store = 1;
    P.boxR = boxR;
    P.nbox = nbox;
}

template <int NG, int T> static void launch_main(MainParams P, uint32_t smem, cudaStream_t st)
{
    CUtensorMap omap;
    memset(&omap, 0, sizeof(omap));
    setup_out_map<NG, T>(P, omap);
    if (P.K == 9 && P.Tw == 7) {
        if (P.p == 1) return launch_main_impl<NG, T, 9, 1>(P, omap, smem, st);
    }
    launch_main_impl<NG, T, 0, 0>(P, omap, smem, st);
}

void Run(ffi::TensorView inp, ffi::TensorView pack_idx, ffi::TensorView pack_val, ffi::TensorView pack_offset,
         int64_t kernel_size, int64_t nlon_out, ffi::TensorView out)
{
    const int B = (int)inp.size(0), C = (int)inp.size(1), Hi = (int)inp.size(2), Wi = (int)inp.size(3);
    const long long NNZ = pack_idx.size(0);
    const int KP = (int)pack_val.size(1);
    const int Ho = (int)pack_offset.size(0) - 1;
    const int K = (int)kernel_size, Wo = (int)nlon_out;
    const int BC = B * C;
    // Low channel counts: the (channel, longitude-block) M packing of leader_small wastes no channel
    // padding; from a few m64 tiles of channels on, the channel-N Toeplitz path below is faster.
    // (Upper bound on the channel count only; any pscale/K the leader path cannot take goes below.)
    if (BC < 128 && leader_small::supported(Wi, Wo, K)) {
        leader_small::disco_run(inp, pack_idx, pack_val, pack_offset, kernel_size, nlon_out, out);
        return;
    }
    cudaStream_t st = (cudaStream_t)TVMFFIEnvGetStream(inp.device().device_type, inp.device().device_id);
    if (Ho <= 0 || BC <= 0 || Wo <= 0 || K <= 0) return;
    if (g_num_sms == 0) {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&g_num_sms, cudaDevAttrMultiProcessorCount, dev);
    }
    const __half *x = reinterpret_cast<const __half *>(inp.data_ptr());
    const long long *idx = reinterpret_cast<const long long *>(pack_idx.data_ptr());
    const __half *val = reinterpret_cast<const __half *>(pack_val.data_ptr());
    const long long *off = reinterpret_cast<const long long *>(pack_offset.data_ptr());
    __half *o = reinterpret_cast<__half *>(out.data_ptr());
    const int p = Wi / Wo;
    const int G = (BC + 7) / 8;

    bool fast = K <= 16 && K <= KP && p >= 1 && (long long)p * Wo == Wi && NNZ > 0 && NNZ < (1LL << 31) && Hi > 0
        && (long long)G * Hi * Wi < (1LL << 40);
    bool launched = false;
    if (fast) {
        MainParams P;
        P.val = val;
        P.off = off;
        P.out = o;
        P.BC = BC;
        P.G = G;
        P.Hi = Hi;
        P.Wi = Wi;
        P.Ho = Ho;
        P.Wo = Wo;
        P.p = p;
        P.K = K;
        P.KP = KP;
        P.Tw = 64 / K;
        // channel groups (8 channels each) per CTA tile: all of them up to 4 (N = 32 channels per wgmma)
        const int NGsel = G < 4 ? G : 4;
        uint32_t smem = 0;
        bool ok = false;
        switch (NGsel) {
        case 4: ok = prepare_main<4, 8>(P, smem); break;
        case 3: ok = prepare_main<3, 8>(P, smem); break;
        case 2: ok = prepare_main<2, 8>(P, smem); break;
        default: ok = prepare_main<1, 8>(P, smem); break;
        }
        if (ok) {
            const long long totcol = ((NNZ + (long long)Ho * ((long long)SEGCAP * (2 * P.OFF + 16) + 8) + 7) & ~7LL) + 8;
            const size_t xt_bytes = (size_t)G * Hi * Wi * 16;
            const size_t seg_bytes = (size_t)NNZ * sizeof(Seg);
            const size_t nseg_bytes = (size_t)Ho * sizeof(int);
            const size_t ft_bytes = (size_t)K * totcol * 4;
            auto al = [](size_t v) { return (v + 255) & ~(size_t)255; };
            uint8_t *ws = (uint8_t *)ws_get(al(xt_bytes) + al(seg_bytes) + al(nseg_bytes) + al(ft_bytes), st);
            if (ws) {
                uint4 *xt = (uint4 *)ws;
                Seg *segs = (Seg *)(ws + al(xt_bytes));
                int *nseg = (int *)(ws + al(xt_bytes) + al(seg_bytes));
                uint32_t *ftg = (uint32_t *)(ws + al(xt_bytes) + al(seg_bytes) + al(nseg_bytes));
                P.nseg = nseg;
                P.segs = segs;
                P.xt = xt;
                P.ftg = ftg;
                P.totcol = totcol;
                const long long HW = (long long)Hi * Wi;
                dim3 tg((unsigned)((HW + 255) / 256), (unsigned)G);
                transpose_kernel<<<tg, 256, 0, st>>>(reinterpret_cast<const uint16_t *>(x), xt, BC, HW);
                FtParams F;
                F.val = val;
                F.ftg = ftg;
                F.totcol = totcol;
                F.K = K;
                F.KP = KP;
                F.OFF = P.OFF;
                F.segcap = SEGCAP;
                seg_prep_kernel<<<Ho, PREP_THREADS, 0, st>>>(idx, off, segs, nseg, Hi, Wi, F);
                ft_fill_kernel<<<Ho * FILL_SPLIT, 256, 0, st>>>(off, segs, nseg, F);
                switch (NGsel) {
                case 4: launch_main<4, 8>(P, smem, st); break;
                case 3: launch_main<3, 8>(P, smem, st); break;
                case 2: launch_main<2, 8>(P, smem, st); break;
                default: launch_main<1, 8>(P, smem, st); break;
                }
                generic_kernel<<<Ho, 256, 0, st>>>(x, idx, val, off, o, nseg, BC, Hi, Wi, Ho, Wo, K, KP, p);
                launched = true;
            }
        }
    }
    if (!launched) { generic_kernel<<<Ho, 256, 0, st>>>(x, idx, val, off, o, nullptr, BC, Hi, Wi, Ho, Wo, K, KP, p); }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
