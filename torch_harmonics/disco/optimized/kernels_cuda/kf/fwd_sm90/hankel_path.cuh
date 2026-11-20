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

// Hankel-formulation low-pscale path, adopted from campaign candidate 4e5107e1 (rows = (channel, 8-longitude
// block), cols = 9 basis x 8 shifts, RS wgmma m64n72k16; see the comments below).
#include <algorithm>

#include "fft_path.cuh"

namespace hankel
{
    // DISCO S2 forward contraction (K-packed psi), fp16 in / fp32 accumulate / fp16 out.
    //
    // Hankel formulation.  For one output latitude ho and one support piece i (input latitude hi_i, a
    // contiguous run of L_i longitudes starting at w0_i, filter f_i[k][j], j < L_i):
    //   out[c, k, 8q + r] += sum_j f_i[k][j] * x[c, hi_i, w0_i + j + p*(8q + r)]
    // is a GEMM with rows (c, q), columns n = 8k + r (N = 72) and a K index u over the piece window:
    //   A[(c,q), u] = x[c, hi_i, w0_i - a_i + u + 8pq]      Hankel rows (stride 8p), read by LDS.32 from
    //                                                       input windows staged in smem ring slots
    //   B[u, (k,r)] = f_i[k][u - a_i - r*p]                 Toeplitz filter, shared by every (c, q)
    // (a_i = w0_i & 1 keeps every K pair 4-byte aligned).  All pieces of a row are concatenated along K at
    // 2-element granularity, so one output row is a single GEMM  M = BC*Wo/8, N = 72, K = sum_i ext_i.
    // A prep kernel builds B per 16-deep k-step in the wgmma K-major smem layout plus a header locating
    // each K pair (piece, window offset).  The main kernel is persistent: per CTA a producer warp
    // bulk-copies B k-step groups and per-piece input windows into mbarrier rings, one consumer warpgroup
    // issues wgmma m64n72k16 (A from registers) on up to 4 row tiles with fp32 accumulators; the
    // accumulator fragment maps directly onto 128-byte coalesced fp16 stores of out[c, k, ho, 8q..8q+63].

    namespace ffi = tvm::ffi;

#define KPMAX 240    // max support points per piece (long runs are split)
#define SK 2         // k-steps per B stage
#define NBMAX 8      // B stages (runtime NB = 1 << NBL <= NBMAX)
#define RSMAX 16     // input window ring slots (runtime RS = 1 << RSL, >= 8: 2*SK + 3 pieces can be live)
#define CHUNK 2336   // bytes per k-step: 2304 (16 x 72 fp16, K-major core matrices) + 32 header
#define RUNCAP 1024  // runs per row handled by the fast path (more -> generic path for that row)
#define NTMAX 6      // 64-row tiles per unit (3 per consumer warpgroup)
#define PT 512       // prep threads
#define SORTCAP 4096 // rows sorted by cost (more -> natural order)
#define NSPLIT 64    // bfill blocks per row

    struct __align__(16) Piece {
        int hi, base, w0a, Ms; // base: 8-aligned window start, w0a = w0 - a, Ms: K offset in the row
        int ext, L, n1, len1;  // ext: K extent; point j -> n1 + j (j < len1) else n2 + (j - len1)
        int n2, a, pad0, pad1;
    };

    // ------------------------------------------------------------------------------------------------
    // PTX helpers
    // ------------------------------------------------------------------------------------------------
    __device__ __forceinline__ uint32_t smem_u32(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }
    __device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count)
    {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
    }
    __device__ __forceinline__ void mbar_arrive(uint32_t bar)
    {
        asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.release.cta.shared::cta.b64 st, [%0];\n}" ::"r"(bar)
                     : "memory");
    }
    __device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes)
    {
        asm volatile(
            "{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 st, [%0], %1;\n}" ::"r"(bar),
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
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
                     "l"(src), "r"(bytes), "r"(bar)
                     : "memory");
    }
    __device__ __forceinline__ void stsm_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3)
    {
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(r0), "r"(r1),
                     "r"(r2), "r"(r3)
                     : "memory");
    }
    __device__ __forceinline__ void stsm_x2(uint32_t addr, uint32_t r0, uint32_t r1)
    {
        asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};" ::"r"(addr), "r"(r0), "r"(r1)
                     : "memory");
    }
    __device__ __forceinline__ void bulk_s2g(void *dst, uint32_t src, uint32_t bytes)
    {
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst), "r"(src), "r"(bytes)
                     : "memory");
    }
    __device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
    __device__ __forceinline__ void bulk_wait_read0() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
    __device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
    __device__ __forceinline__ void wg_bar(int id) { asm volatile("bar.sync %0, 128;" ::"r"(id) : "memory"); }
    __device__ __forceinline__ uint32_t pack_h2(float a, float b)
    {
        const __half2 v = __floats2half2_rn(a, b);
        return *reinterpret_cast<const uint32_t *>(&v);
    }
    __device__ __forceinline__ uint32_t lds32(uint32_t addr)
    {
        uint32_t v;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(addr) : "memory");
        return v;
    }
    __device__ __forceinline__ void stg32(void *p, uint32_t v)
    {
        asm volatile("st.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
    }
    __device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
    __device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
    template <int N> __device__ __forceinline__ void wg_wait()
    {
        asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
    }
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
    // D(64x72, fp32) (+)= A(64x16, fp16 regs) * B(16x72, fp16 smem, K-major)
    __device__ __forceinline__ void wgmma_rs_n72(float *d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                 uint64_t descB, int scale_d)
    {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, %41, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n72k16.f32.f16.f16 "
            "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
            "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}, "
            "{%36, %37, %38, %39}, %40, p, 1, 1, 0;\n"
            "}\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
              "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
              "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
              "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
              "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(descB), "r"(scale_d));
    }

    // exclusive block scan (blockDim.x a multiple of 32, <= 1024); total returned to every thread
    __device__ __forceinline__ void block_scan(int v, int &excl, int &total, int *s_wsum)
    {
        const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
        int incl = v;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, incl, o);
            if (lane >= o) incl += y;
        }
        if (lane == 31) s_wsum[wid] = incl;
        __syncthreads();
        if (wid == 0) {
            int x = lane < nw ? s_wsum[lane] : 0;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int y = __shfl_up_sync(0xffffffffu, x, o);
                if (lane >= o) x += y;
            }
            if (lane < nw) s_wsum[lane] = x;
        }
        __syncthreads();
        excl = (wid ? s_wsum[wid - 1] : 0) + incl - v;
        total = s_wsum[nw - 1];
        __syncthreads();
    }

    // ------------------------------------------------------------------------------------------------
    // prep 1: per output row, split the support into pieces (maximal runs of consecutive longitudes on one
    // input latitude; the run ending at the seam is merged with the run starting at wi = 0; runs longer
    // than KPMAX are split) and compute their K extents.  nks[ho] = -1 marks a row for the generic path.
    // The last block to finish computes the k-step offsets, the cost-sorted row order and the unit plan.
    // ------------------------------------------------------------------------------------------------
    struct PrepParams {
        const long long *idx;
        const long long *off;
        Piece *pieces;
        int *npieces;
        int *nks;
        int *ks_off;    // [Ho + 1]
        int *row_order; // [Ho]
        int *gp;        // [2] units, [3] total k-steps
        int *urow_off;  // [Ho] first unit of each row
        int *urow_nt;   // [Ho] tiles per unit of each row
        int4 *rinfo;    // [Ho] nks, npieces, ks_off, off
        unsigned *counter;
        const int *fftflag; // rows evaluated by the FFT path (nullptr: none)
        int Ho, Wi, p, BC, Q, nslots, ntmax;
    };

    __global__ void __launch_bounds__(PT) prep_rows_kernel(const PrepParams P)
    {
        __shared__ __align__(16) unsigned char s_buf[SORTCAP * 8];
        __shared__ int s_wsum[PT / 32];
        __shared__ int s_flag;
        int *s_n = reinterpret_cast<int *>(s_buf); // [RUNCAP + 1]
        int *s_hi = s_n + RUNCAP + 1;              // [RUNCAP]
        int *s_wi = s_hi + RUNCAP;                 // [RUNCAP]
        const int ho = blockIdx.x, tid = threadIdx.x;
        const long long a = P.off[ho], b = P.off[ho + 1];
        const int n = (int)(b - a);
        const int Wi = P.Wi, p = P.p;
        if (P.fftflag && P.fftflag[ho]) { // FFT path row: no pieces, no units
            if (tid == 0) {
                P.npieces[ho] = 0;
                P.nks[ho] = -2;
            }
        } else if (n > 0) {
            // pass 1: run starts (4 consecutive points per thread, one 16-byte load per point)
            int cnt = 0;
            const longlong2 *ip = reinterpret_cast<const longlong2 *>(P.idx) + a;
            for (int base = 0; base < n; base += 4 * PT) {
                const int i0 = base + 4 * tid;
                int hh[4], ww[4], fl[4];
                int ph = -1, pw = -2;
                if (i0 > 0 && i0 - 1 < n) {
                    const longlong2 v = ip[i0 - 1];
                    ph = (int)v.x;
                    pw = (int)v.y;
                }
                int c4 = 0;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int i = i0 + k;
                    fl[k] = 0;
                    hh[k] = 0;
                    ww[k] = 0;
                    if (i < n) {
                        const longlong2 v = ip[i];
                        hh[k] = (int)v.x;
                        ww[k] = (int)v.y;
                        fl[k] = (i == 0) || hh[k] != ph || ww[k] != pw + 1;
                        ph = hh[k];
                        pw = ww[k];
                    }
                    c4 += fl[k];
                }
                int excl, tot;
                block_scan(c4, excl, tot, s_wsum);
                int pos = cnt + excl;
#pragma unroll
                for (int k = 0; k < 4; ++k)
                    if (fl[k]) {
                        if (pos < RUNCAP) {
                            s_n[pos] = i0 + k;
                            s_hi[pos] = hh[k];
                            s_wi[pos] = ww[k];
                        }
                        ++pos;
                    }
                cnt += tot;
            }
            if (cnt > RUNCAP) {
                if (tid == 0) {
                    P.npieces[ho] = 0;
                    P.nks[ho] = -1;
                }
            } else {
                if (tid == 0) s_n[cnt] = n;
                __syncthreads();
                // pass 2: merged runs -> pieces (without Ms)
                int npc = 0;
                for (int rb = 0; rb < cnt; rb += PT) {
                    const int i = rb + tid;
                    int np = 0, hi = 0, w0 = 0, Lt = 0, n1 = 0, len1 = 0, n2 = 0;
                    if (i < cnt) {
                        hi = s_hi[i];
                        int F = i;
                        while (F > 0 && s_hi[F - 1] == hi) --F;
                        int Lr = i;
                        while (Lr + 1 < cnt && s_hi[Lr + 1] == hi) ++Lr;
                        const int lenL = s_n[Lr + 1] - s_n[Lr];
                        const bool merge = (Lr > F) && s_wi[F] == 0 && (s_wi[Lr] + lenL == Wi);
                        const bool emit = !(merge && i == F);
                        if (emit) {
                            if (merge && i == Lr) {
                                w0 = s_wi[Lr];
                                len1 = lenL;
                                n1 = (int)a + s_n[Lr];
                                n2 = (int)a + s_n[F];
                                Lt = lenL + (s_n[F + 1] - s_n[F]);
                            } else {
                                w0 = s_wi[i];
                                len1 = s_n[i + 1] - s_n[i];
                                n1 = (int)a + s_n[i];
                                n2 = 0;
                                Lt = len1;
                            }
                            np = (Lt + KPMAX - 1) / KPMAX;
                        }
                    }
                    int excl, tot;
                    block_scan(np, excl, tot, s_wsum);
                    for (int k = 0; k < np; ++k) {
                        const int j0 = k * KPMAX, l = min(KPMAX, Lt - j0);
                        Piece pc;
                        pc.hi = hi;
                        const int w = (w0 + j0) % Wi;
                        pc.a = w & 3;
                        pc.w0a = w - pc.a;
                        pc.base = pc.w0a & ~7;
                        if (j0 < len1) {
                            pc.n1 = n1 + j0;
                            pc.len1 = len1 - j0;
                            pc.n2 = n2;
                        } else {
                            pc.n1 = n2 + (j0 - len1);
                            pc.len1 = l;
                            pc.n2 = 0;
                        }
                        pc.L = l;
                        pc.ext = max(16, (pc.a + l + 7 * p + 3) & ~3); // >= 16: at most 5 pieces in any 64-deep K span
                        pc.Ms = 0;
                        pc.pad0 = pc.pad1 = 0;
                        P.pieces[a + npc + excl + k] = pc;
                    }
                    npc += tot;
                }
                __syncthreads();
                // pass 3: K offsets
                int msum = 0;
                for (int pb = 0; pb < npc; pb += PT) {
                    const int i = pb + tid;
                    const int e = i < npc ? __ldcg(&P.pieces[a + i].ext) : 0;
                    int excl, tot;
                    block_scan(e, excl, tot, s_wsum);
                    if (i < npc) P.pieces[a + i].Ms = msum + excl;
                    msum += tot;
                }
                if (tid == 0) {
                    const bool ok = npc < 65536;
                    P.npieces[ho] = ok ? npc : 0;
                    P.nks[ho] = ok ? ((msum + 31) / 32) * 2 : -1; // whole B stages (SK = 2)
                }
            }
        } else if (tid == 0) {
            P.npieces[ho] = 0;
            P.nks[ho] = 0;
        }

        // ---- last block: k-step offsets, cost-sorted row order, unit plan ----
        __syncthreads();
        if (tid == 0) {
            __threadfence();
            const unsigned prev = atomicAdd(P.counter, 1u);
            s_flag = (prev == gridDim.x - 1);
        }
        __syncthreads();
        if (!s_flag) return;
        __threadfence();
        const int Ho = P.Ho;
        int run = 0, mx = 0;
        long long sum = 0;
        for (int rb = 0; rb < Ho; rb += PT) {
            const int i = rb + tid;
            const int v = i < Ho ? max(__ldcg(&P.nks[i]), 0) : 0;
            int excl, tot;
            block_scan(v, excl, tot, s_wsum);
            if (i < Ho) {
                P.ks_off[i] = run + excl;
                P.rinfo[i] = make_int4(__ldcg(&P.nks[i]), __ldcg(&P.npieces[i]), run + excl, (int)P.off[i]);
            }
            run += tot;
            mx = max(mx, v);
            sum += v;
        }
        // block reductions of mx and sum
        __shared__ int s_mx[PT / 32];
        __shared__ long long s_sum[PT / 32];
        {
            const int lane = tid & 31, wid = tid >> 5;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, o));
                sum += __shfl_xor_sync(0xffffffffu, sum, o);
            }
            if (lane == 0) {
                s_mx[wid] = mx;
                s_sum[wid] = sum;
            }
            __syncthreads();
            mx = 0;
            sum = 0;
            for (int w = 0; w < PT / 32; ++w) {
                mx = max(mx, s_mx[w]);
                sum += s_sum[w];
            }
        }
        // unit plan: a unit is nt (1..ntmax) 64-row tiles of one row; heavy rows get small units so that no
        // unit exceeds half of the average per-pipeline load
        const long long rows_per = (long long)P.BC * P.Q;
        const long long tpr = (rows_per + 63) / 64;
        long long target = (sum * tpr) / (2LL * P.nslots);
        if (target < 1) target = 1;
        if (tid == 0) P.ks_off[Ho] = run;
        // cost-sorted row order (descending k-steps, ascending row)
        const bool sorted = Ho <= SORTCAP;
        if (sorted) {
            unsigned long long *key = reinterpret_cast<unsigned long long *>(s_buf);
            int np2 = 1;
            while (np2 < Ho) np2 <<= 1;
            for (int i = tid; i < np2; i += PT)
                key[i] = i < Ho ? ((unsigned long long)(0x7fffffff - max(__ldcg(&P.nks[i]), 0)) << 32) | (unsigned)i :
                                  ~0ULL;
            __syncthreads();
            for (int k = 2; k <= np2; k <<= 1) {
                for (int j = k >> 1; j > 0; j >>= 1) {
                    for (int i = tid; i < np2; i += PT) {
                        const int l = i ^ j;
                        if (l > i) {
                            const unsigned long long x = key[i], y = key[l];
                            const bool up = (i & k) == 0;
                            if ((x > y) == up) {
                                key[i] = y;
                                key[l] = x;
                            }
                        }
                    }
                    __syncthreads();
                }
            }
            for (int i = tid; i < Ho; i += PT) P.row_order[i] = (int)(key[i] & 0xffffffffu);
        } else {
            for (int i = tid; i < Ho; i += PT) P.row_order[i] = i;
        }
        __syncthreads();
        int ubase = 0;
        for (int rb = 0; rb < Ho; rb += PT) {
            const int i = rb + tid;
            int units = 0, row = 0, nt = P.ntmax;
            if (i < Ho) {
                row = __ldcg(&P.row_order[i]);
                const int nk = __ldcg(&P.nks[row]);
                if (nk > 0) {
                    long long v = target / nk;
                    v = v < 2 ? 2 : v;
                    v = v > P.ntmax ? P.ntmax : v;
                    nt = (int)v;
                }
                units = nk == -2 ? 0 : (int)((rows_per + 64LL * nt - 1) / (64LL * nt));
            }
            int excl, tot;
            block_scan(units, excl, tot, s_wsum);
            if (i < Ho) {
                P.urow_off[row] = ubase + excl;
                P.urow_nt[row] = nt;
            }
            ubase += tot;
        }
        if (tid == 0) {
            P.gp[2] = ubase;
            P.gp[3] = run;
            *P.counter = 0u;
        }
    }

    // ------------------------------------------------------------------------------------------------
    // prep 2: B (Toeplitz filter) of every k-step of every fast row, in the wgmma K-major no-swizzle
    // layout (byte (kk/8)*1152 + n*16 + (kk%8)*2 for K index kk, column n = 8k + r), plus the 8 K-pair
    // header words (piece index << 16 | offset into the piece's input window).
    // ------------------------------------------------------------------------------------------------
    struct FillParams {
        const __half *val;
        const int *urow_off;
        const int *urow_nt;
        int2 *utab; // [units] (row, R0 << 3 | nt)
        long long rows_per;
        const long long *off;
        const Piece *pieces;
        const int *npieces;
        const int *nks;
        const int *ks_off;
        uint8_t *Bg;
        int K, KP, p;
    };

#define MSCAP 2048
    // K slot kk of a k-step holds K offset perm(kk) so that A-fragment registers a0 (slots 2t, 2t+1) and
    // a2 (slots 2t+8, 2t+9) are the 4 consecutive K elements 4t..4t+3 (one LDS.64 per row).
    __global__ void __launch_bounds__(128) bfill_kernel(const FillParams F)
    {
        __shared__ __align__(16) uint16_t sB[1152];
        __shared__ __align__(16) uint32_t sH[8];
        __shared__ int sMs[MSCAP];
        const int ho = blockIdx.x;
        const int tid = threadIdx.x;
        if (blockIdx.y == 0) {
            const int nt = F.urow_nt[ho], u0 = F.urow_off[ho];
            const int units = F.nks[ho] == -2 ? 0 : (int)((F.rows_per + 64LL * nt - 1) / (64LL * nt));
            for (int k = tid; k < units; k += 128) F.utab[u0 + k] = make_int2(ho, ((k * nt * 64) << 3) | nt);
        }
        const int nk = F.nks[ho];
        if (nk <= 0 || (int)blockIdx.y >= nk) return;
        const int kk = tid >> 3, r = tid & 7;
        const int kperm = kk < 8 ? 4 * (kk >> 1) + (kk & 1) : 4 * ((kk - 8) >> 1) + 2 + (kk & 1);
        const long long pb = F.off[ho];
        const int npc = F.npieces[ho];
        const int ks0 = F.ks_off[ho];
        const bool smem_ms = npc <= MSCAP;
        if (smem_ms)
            for (int i = tid; i < npc; i += 128) sMs[i] = F.pieces[pb + i].Ms;
        __syncthreads();
        const int K = F.K, KP = F.KP, p = F.p;
        for (int s = blockIdx.y; s < nk; s += NSPLIT) {
            const int m = 16 * s + kperm;
            int lo = 0, hi = npc - 1;
            while (lo < hi) {
                const int mid = (lo + hi + 1) >> 1;
                const int ms = smem_ms ? sMs[mid] : F.pieces[pb + mid].Ms;
                if (ms <= m)
                    lo = mid;
                else
                    hi = mid - 1;
            }
            const Piece pc = F.pieces[pb + lo];
            const bool inside = m < pc.Ms + pc.ext;
            const int j = m - pc.Ms - pc.a - r * p;
            uint16_t v[16];
#pragma unroll
            for (int k = 0; k < 16; ++k) v[k] = 0;
            if (inside && j >= 0 && j < pc.L) {
                const long long pt = j < pc.len1 ? (long long)pc.n1 + j : (long long)pc.n2 + (j - pc.len1);
                if (KP == 16) {
                    const uint4 *rr = reinterpret_cast<const uint4 *>(F.val + pt * 16);
                    const uint4 x0 = __ldg(rr), x1 = __ldg(rr + 1);
                    const uint32_t w[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
                    for (int k = 0; k < 16; ++k) v[k] = (uint16_t)(w[k >> 1] >> ((k & 1) * 16));
                } else {
#pragma unroll
                    for (int k = 0; k < 16; ++k)
                        if (k < KP) v[k] = __half_as_ushort(F.val[pt * KP + k]);
                }
            }
#pragma unroll
            for (int k = 0; k < 9; ++k) sB[(kk >> 3) * 576 + (8 * k + r) * 8 + (kk & 7)] = k < K ? v[k] : (uint16_t)0;
            if (r == 0 && kk < 8 && (kk & 1) == 0) { // quad kk/2 starts at K offset 4*(kk/2) = kperm
                const int o = inside ? (pc.w0a - pc.base) + (m - pc.Ms) : 0;
                sH[kk >> 1] = ((uint32_t)lo << 16) | (uint32_t)o;
                sH[4 + (kk >> 1)] = 0u;
            }
            __syncthreads();
            uint4 *dst = reinterpret_cast<uint4 *>(F.Bg + (size_t)(ks0 + s) * CHUNK);
            for (int e = tid; e < CHUNK / 16; e += 128)
                dst[e] = e < 144 ? reinterpret_cast<const uint4 *>(sB)[e] : reinterpret_cast<const uint4 *>(sH)[e - 144];
            __syncthreads();
        }
    }

    // ------------------------------------------------------------------------------------------------
    // main persistent kernel
    // ------------------------------------------------------------------------------------------------
    struct MainParams {
        const __half *x;
        __half *out;
        const __half *val;
        const long long *idx;
        const long long *off;
        const Piece *pieces;
        const int *npieces;
        const int *nks;
        const int *ks_off;
        const int2 *utab;
        const int4 *rinfo; // per row: nks, npieces, ks_off, off
        const int *gp;
        const uint8_t *Bg;
        int BC, Hi, Wi, Ho, Wo, Q, p, K, KP;
        int EXTW;     // per-channel window slack in elements (multiple of 8)
        int SLOTB;    // ring slot bytes
        int PIPEB;    // smem bytes per pipeline
        int NBL, RSL; // log2 of B stages / ring slots
    };

#define STAGEB (SK * CHUNK)

    // producer of input windows: three warps split the channel windows of every piece (channel index mod 3;
    // all arrive on the slot's full barrier with their own byte counts)
    __device__ __forceinline__ void produce_pieces(const MainParams &P, uint32_t ring0, uint32_t barFR, uint32_t barER,
                                                   int lane, int gpipe, int npipe, int part)
    {
        const int nunits = P.gp[2], TOT = P.BC * P.Q;
        const int p8 = 8 * P.p, Wi = P.Wi;
        const uint32_t rsm = (1u << P.RSL) - 1u, rsl = (uint32_t)P.RSL;
        uint32_t pseq = 0;
        int u = gpipe;
        int2 ud0 = u < nunits ? P.utab[u] : make_int2(0, 0);
        int2 ud1 = u + npipe < nunits ? P.utab[u + npipe] : make_int2(0, 0);
        int4 ri0 = u < nunits ? P.rinfo[ud0.x] : make_int4(0, 0, 0, 0);
        for (; u < nunits; u += npipe) {
            const int4 ri1 = u + npipe < nunits ? P.rinfo[ud1.x] : make_int4(0, 0, 0, 0);
            const int2 ud2 = u + 2 * npipe < nunits ? P.utab[u + 2 * npipe] : make_int2(0, 0);
            const int nk = ri0.x;
            if (nk > 0) {
                const int R0 = ud0.y >> 3, RU = (ud0.y & 7) * 64;
                const int rows_u = min(RU, TOT - R0);
                const int npc = ri0.y;
                const long long pb = ri0.w;
                const int c_first = R0 / P.Q, c_last = (R0 + rows_u - 1) / P.Q;
                const int ci = 3 * lane + part;
                const int c = c_first + ci;
                const bool cv = c <= c_last;
                int qa = 0, nq = 0;
                uint32_t dsto = 0;
                if (cv) {
                    const int r_lo = max(R0, c * P.Q), r_hi = min(R0 + rows_u, (c + 1) * P.Q);
                    qa = r_lo - c * P.Q;
                    nq = r_hi - r_lo;
                    dsto = 2u * (uint32_t)(ci * P.EXTW + p8 * (r_lo - R0));
                }
                const __half *xc = P.x + (size_t)(cv ? c : c_first) * P.Hi * Wi;
                for (int pi0 = 0; pi0 < npc; pi0 += 32) {
                    int my_hi = 0, my_base = 0, my_ext = 0;
                    if (pi0 + lane < npc) {
                        const int4 v0 = __ldg(reinterpret_cast<const int4 *>(P.pieces + pb + pi0 + lane));
                        my_hi = v0.x;
                        my_base = v0.y;
                        my_ext = __ldg(&P.pieces[pb + pi0 + lane].ext);
                    }
                    const int nb = min(32, npc - pi0);
                    for (int k = 0; k < nb; ++k) {
                        const int hi = __shfl_sync(0xffffffffu, my_hi, k);
                        const int base = __shfl_sync(0xffffffffu, my_base, k);
                        const int ext = __shfl_sync(0xffffffffu, my_ext, k);
                        const uint32_t slot = pseq & rsm;
                        const int span = p8 * nq + ((ext + 15) & ~7);
                        const uint32_t total = __reduce_add_sync(0xffffffffu, cv ? 2u * (uint32_t)span : 0u);
                        if (lane == 0) {
                            mbar_wait(barER + 8 * slot, ((pseq >> rsl) & 1) ^ 1);
                            mbar_arrive_expect_tx(barFR + 8 * slot, total);
                        }
                        __syncwarp();
                        if (cv) {
                            int pos = base + p8 * qa;
                            if (pos >= Wi) pos -= Wi;
                            int rem = span;
                            uint32_t dst = ring0 + slot * (uint32_t)P.SLOTB + dsto;
                            const __half *srow = xc + (size_t)hi * Wi;
                            while (rem > 0) {
                                const int len = min(rem, Wi - pos);
                                bulk_g2s(dst, srow + pos, 2u * (uint32_t)len, barFR + 8 * slot);
                                dst += 2u * (uint32_t)len;
                                rem -= len;
                                pos = 0;
                            }
                        }
                        ++pseq;
                    }
                }
            }
            ud0 = ud1;
            ri0 = ri1;
            ud1 = ud2;
        }
    }

    // producer of B k-step groups
    __device__ __forceinline__ void produce_B(const MainParams &P, uint32_t bst0, uint32_t barFB, uint32_t barEB,
                                              int lane, int gpipe, int npipe)
    {
        const int nunits = P.gp[2];
        const uint32_t nbm = (1u << P.NBL) - 1u, nbl = (uint32_t)P.NBL;
        uint32_t sseq = 0;
        int u = gpipe;
        int2 ud1 = u + npipe < nunits ? P.utab[u + npipe] : make_int2(0, 0);
        int4 ri0 = u < nunits ? P.rinfo[P.utab[u].x] : make_int4(0, 0, 0, 0);
        for (; u < nunits; u += npipe) {
            const int4 ri1 = u + npipe < nunits ? P.rinfo[ud1.x] : make_int4(0, 0, 0, 0);
            const int2 ud2 = u + 2 * npipe < nunits ? P.utab[u + 2 * npipe] : make_int2(0, 0);
            const int nk = ri0.x;
            if (nk > 0 && lane == 0) {
                const int ks0 = ri0.z;
                const int nstg = (nk + SK - 1) / SK;
                for (int gs = 0; gs < nstg; ++gs) {
                    const uint32_t st = sseq & nbm;
                    mbar_wait(barEB + 8 * st, ((sseq >> nbl) & 1) ^ 1);
                    const int nkg = min(SK, nk - gs * SK);
                    mbar_arrive_expect_tx(barFB + 8 * st, (uint32_t)(nkg * CHUNK));
                    bulk_g2s(bst0 + st * STAGEB, P.Bg + (size_t)(ks0 + gs * SK) * CHUNK, (uint32_t)(nkg * CHUNK),
                             barFB + 8 * st);
                    ++sseq;
                }
            }
            sseq = __shfl_sync(0xffffffffu, sseq, 0);
            ri0 = ri1;
            ud1 = ud2;
        }
    }

    // one B stage: NT tiles x 2 k-steps accumulated with scale-d = 1 (accumulators are zeroed per unit)
    template <int NT>
    __device__ __forceinline__ void issue_stage(float (&acc)[NT][36], const uint32_t (&X)[NT][4],
                                                const uint32_t (&Y)[NT][4], uint64_t d0, uint64_t d1);
    template <>
    __device__ __forceinline__ void issue_stage<1>(float (&acc)[1][36], const uint32_t (&X)[1][4],
                                                   const uint32_t (&Y)[1][4], uint64_t d0, uint64_t d1)
    {
        asm volatile("wgmma.mma_async.sync.aligned.m64n72k16.f32.f16.f16 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%"
                     "26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%36,%37,%38,%39}, %44, 1, 1, 1, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n72k16.f32.f16.f16 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%"
                     "26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%40,%41,%42,%43}, %45, 1, 1, 1, 0;\n"
                     : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[0][4]),
                       "+f"(acc[0][5]), "+f"(acc[0][6]), "+f"(acc[0][7]), "+f"(acc[0][8]), "+f"(acc[0][9]),
                       "+f"(acc[0][10]), "+f"(acc[0][11]), "+f"(acc[0][12]), "+f"(acc[0][13]), "+f"(acc[0][14]),
                       "+f"(acc[0][15]), "+f"(acc[0][16]), "+f"(acc[0][17]), "+f"(acc[0][18]), "+f"(acc[0][19]),
                       "+f"(acc[0][20]), "+f"(acc[0][21]), "+f"(acc[0][22]), "+f"(acc[0][23]), "+f"(acc[0][24]),
                       "+f"(acc[0][25]), "+f"(acc[0][26]), "+f"(acc[0][27]), "+f"(acc[0][28]), "+f"(acc[0][29]),
                       "+f"(acc[0][30]), "+f"(acc[0][31]), "+f"(acc[0][32]), "+f"(acc[0][33]), "+f"(acc[0][34]),
                       "+f"(acc[0][35])
                     : "r"(X[0][0]), "r"(X[0][1]), "r"(X[0][2]), "r"(X[0][3]), "r"(Y[0][0]), "r"(Y[0][1]), "r"(Y[0][2]),
                       "r"(Y[0][3]), "l"(d0), "l"(d1));
    }
    template <>
    __device__ __forceinline__ void issue_stage<2>(float (&acc)[2][36], const uint32_t (&X)[2][4],
                                                   const uint32_t (&Y)[2][4], uint64_t d0, uint64_t d1)
    {
        asm volatile("wgmma.mma_async.sync.aligned.m64n72k16.f32.f16.f16 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%"
                     "26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%72,%73,%74,%75}, %88, 1, 1, 1, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n72k16.f32.f16.f16 "
                     "{%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,"
                     "%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71}, {%76,%77,%78,%79}, %88, 1, 1, 1, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n72k16.f32.f16.f16 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%"
                     "26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%80,%81,%82,%83}, %89, 1, 1, 1, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n72k16.f32.f16.f16 "
                     "{%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,"
                     "%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71}, {%84,%85,%86,%87}, %89, 1, 1, 1, 0;\n"
                     : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[0][4]),
                       "+f"(acc[0][5]), "+f"(acc[0][6]), "+f"(acc[0][7]), "+f"(acc[0][8]), "+f"(acc[0][9]),
                       "+f"(acc[0][10]), "+f"(acc[0][11]), "+f"(acc[0][12]), "+f"(acc[0][13]), "+f"(acc[0][14]),
                       "+f"(acc[0][15]), "+f"(acc[0][16]), "+f"(acc[0][17]), "+f"(acc[0][18]), "+f"(acc[0][19]),
                       "+f"(acc[0][20]), "+f"(acc[0][21]), "+f"(acc[0][22]), "+f"(acc[0][23]), "+f"(acc[0][24]),
                       "+f"(acc[0][25]), "+f"(acc[0][26]), "+f"(acc[0][27]), "+f"(acc[0][28]), "+f"(acc[0][29]),
                       "+f"(acc[0][30]), "+f"(acc[0][31]), "+f"(acc[0][32]), "+f"(acc[0][33]), "+f"(acc[0][34]),
                       "+f"(acc[0][35]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]),
                       "+f"(acc[1][4]), "+f"(acc[1][5]), "+f"(acc[1][6]), "+f"(acc[1][7]), "+f"(acc[1][8]),
                       "+f"(acc[1][9]), "+f"(acc[1][10]), "+f"(acc[1][11]), "+f"(acc[1][12]), "+f"(acc[1][13]),
                       "+f"(acc[1][14]), "+f"(acc[1][15]), "+f"(acc[1][16]), "+f"(acc[1][17]), "+f"(acc[1][18]),
                       "+f"(acc[1][19]), "+f"(acc[1][20]), "+f"(acc[1][21]), "+f"(acc[1][22]), "+f"(acc[1][23]),
                       "+f"(acc[1][24]), "+f"(acc[1][25]), "+f"(acc[1][26]), "+f"(acc[1][27]), "+f"(acc[1][28]),
                       "+f"(acc[1][29]), "+f"(acc[1][30]), "+f"(acc[1][31]), "+f"(acc[1][32]), "+f"(acc[1][33]),
                       "+f"(acc[1][34]), "+f"(acc[1][35])
                     : "r"(X[0][0]), "r"(X[0][1]), "r"(X[0][2]), "r"(X[0][3]), "r"(X[1][0]), "r"(X[1][1]), "r"(X[1][2]),
                       "r"(X[1][3]), "r"(Y[0][0]), "r"(Y[0][1]), "r"(Y[0][2]), "r"(Y[0][3]), "r"(Y[1][0]), "r"(Y[1][1]),
                       "r"(Y[1][2]), "r"(Y[1][3]), "l"(d0), "l"(d1));
    }
    template <>
    __device__ __forceinline__ void issue_stage<3>(float (&acc)[3][36], const uint32_t (&X)[3][4],
                                                   const uint32_t (&Y)[3][4], uint64_t d0, uint64_t d1)
    {
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n72k16.f32.f16.f16 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%"
            "28,%29,%30,%31,%32,%33,%34,%35}, {%108,%109,%110,%111}, %132, 1, 1, 1, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n72k16.f32.f16.f16 "
            "{%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%"
            "62,%63,%64,%65,%66,%67,%68,%69,%70,%71}, {%112,%113,%114,%115}, %132, 1, 1, 1, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n72k16.f32.f16.f16 "
            "{%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%"
            "98,%99,%100,%101,%102,%103,%104,%105,%106,%107}, {%116,%117,%118,%119}, %132, 1, 1, 1, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n72k16.f32.f16.f16 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%"
            "28,%29,%30,%31,%32,%33,%34,%35}, {%120,%121,%122,%123}, %133, 1, 1, 1, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n72k16.f32.f16.f16 "
            "{%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%"
            "62,%63,%64,%65,%66,%67,%68,%69,%70,%71}, {%124,%125,%126,%127}, %133, 1, 1, 1, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n72k16.f32.f16.f16 "
            "{%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%"
            "98,%99,%100,%101,%102,%103,%104,%105,%106,%107}, {%128,%129,%130,%131}, %133, 1, 1, 1, 0;\n"
            : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[0][4]), "+f"(acc[0][5]),
              "+f"(acc[0][6]), "+f"(acc[0][7]), "+f"(acc[0][8]), "+f"(acc[0][9]), "+f"(acc[0][10]), "+f"(acc[0][11]),
              "+f"(acc[0][12]), "+f"(acc[0][13]), "+f"(acc[0][14]), "+f"(acc[0][15]), "+f"(acc[0][16]),
              "+f"(acc[0][17]), "+f"(acc[0][18]), "+f"(acc[0][19]), "+f"(acc[0][20]), "+f"(acc[0][21]),
              "+f"(acc[0][22]), "+f"(acc[0][23]), "+f"(acc[0][24]), "+f"(acc[0][25]), "+f"(acc[0][26]),
              "+f"(acc[0][27]), "+f"(acc[0][28]), "+f"(acc[0][29]), "+f"(acc[0][30]), "+f"(acc[0][31]),
              "+f"(acc[0][32]), "+f"(acc[0][33]), "+f"(acc[0][34]), "+f"(acc[0][35]), "+f"(acc[1][0]), "+f"(acc[1][1]),
              "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[1][4]), "+f"(acc[1][5]), "+f"(acc[1][6]), "+f"(acc[1][7]),
              "+f"(acc[1][8]), "+f"(acc[1][9]), "+f"(acc[1][10]), "+f"(acc[1][11]), "+f"(acc[1][12]), "+f"(acc[1][13]),
              "+f"(acc[1][14]), "+f"(acc[1][15]), "+f"(acc[1][16]), "+f"(acc[1][17]), "+f"(acc[1][18]),
              "+f"(acc[1][19]), "+f"(acc[1][20]), "+f"(acc[1][21]), "+f"(acc[1][22]), "+f"(acc[1][23]), "+f"(acc[1][24]),
              "+f"(acc[1][25]), "+f"(acc[1][26]), "+f"(acc[1][27]), "+f"(acc[1][28]), "+f"(acc[1][29]), "+f"(acc[1][30]),
              "+f"(acc[1][31]), "+f"(acc[1][32]), "+f"(acc[1][33]), "+f"(acc[1][34]), "+f"(acc[1][35]), "+f"(acc[2][0]),
              "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[2][4]), "+f"(acc[2][5]), "+f"(acc[2][6]),
              "+f"(acc[2][7]), "+f"(acc[2][8]), "+f"(acc[2][9]), "+f"(acc[2][10]), "+f"(acc[2][11]), "+f"(acc[2][12]),
              "+f"(acc[2][13]), "+f"(acc[2][14]), "+f"(acc[2][15]), "+f"(acc[2][16]), "+f"(acc[2][17]),
              "+f"(acc[2][18]), "+f"(acc[2][19]), "+f"(acc[2][20]), "+f"(acc[2][21]), "+f"(acc[2][22]), "+f"(acc[2][23]),
              "+f"(acc[2][24]), "+f"(acc[2][25]), "+f"(acc[2][26]), "+f"(acc[2][27]), "+f"(acc[2][28]), "+f"(acc[2][29]),
              "+f"(acc[2][30]), "+f"(acc[2][31]), "+f"(acc[2][32]), "+f"(acc[2][33]), "+f"(acc[2][34]), "+f"(acc[2][35])
            : "r"(X[0][0]), "r"(X[0][1]), "r"(X[0][2]), "r"(X[0][3]), "r"(X[1][0]), "r"(X[1][1]), "r"(X[1][2]),
              "r"(X[1][3]), "r"(X[2][0]), "r"(X[2][1]), "r"(X[2][2]), "r"(X[2][3]), "r"(Y[0][0]), "r"(Y[0][1]),
              "r"(Y[0][2]), "r"(Y[0][3]), "r"(Y[1][0]), "r"(Y[1][1]), "r"(Y[1][2]), "r"(Y[1][3]), "r"(Y[2][0]),
              "r"(Y[2][1]), "r"(Y[2][2]), "r"(Y[2][3]), "l"(d0), "l"(d1));
    }

    template <int NT>
    __device__ __forceinline__ void issue(float (&acc)[NT][36], const uint32_t (&A)[NT][4], uint64_t desc, int sc)
    {
#pragma unroll
        for (int i = 0; i < NT; ++i) wgmma_rs_n72(acc[i], A[i][0], A[i][1], A[i][2], A[i][3], desc, sc);
    }

    __device__ __forceinline__ void lds64(uint32_t addr, uint32_t &lo, uint32_t &hi)
    {
        asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];" : "=r"(lo), "=r"(hi) : "r"(addr) : "memory");
    }

    // this warpgroup's NT (0..3) tiles start at tile tb of the unit; NT == 0 only keeps the barrier phases
    template <int NT>
    __device__ __forceinline__ void
    consume(const MainParams &P, uint32_t bst0, uint32_t ring0, uint32_t barFB, uint32_t barEB, uint32_t barFR,
            uint32_t barER, uint32_t pseq0, uint32_t sseq0, int nk, int npc, int row, int R0, int rows_u, int c_first,
            int cw, int lane, uint32_t stg0, uint32_t barSF, uint32_t barSE, uint32_t epar, int tb)
    {
        constexpr int NA = NT > 0 ? NT : 1;
        const int g = lane >> 2, t = lane & 3;
        const int Q = P.Q;
        float acc[NA][36];
        // rows of this thread: rl = (tb + i)*64 + cw*16 + g + 8h ; (c, q) tracked incrementally from one division
        uint32_t Rw[NA][2];
        const int r00 = R0 + tb * 64 + cw * 16 + g; // tile i of this warpgroup is unit tile tb + 2i
        const int c00 = r00 / Q, q00 = r00 - c00 * Q;
        {
            int c = c00, q = q00;
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int rl = (tb + 2 * i) * 64 + cw * 16 + g + 8 * h;
                    const int rr = rl < rows_u ? rl : 0;
                    const int cc = rl < rows_u ? c : c_first;
                    Rw[i][h] = 2u * (uint32_t)((cc - c_first) * P.EXTW + 8 * P.p * rr);
                    q += (h == 0) ? 8 : 120;
                    while (q >= Q) {
                        q -= Q;
                        ++c;
                    }
                }
        }
        if (nk > 0) {
            // nk is even: B stage gs holds k-steps 2gs (chunk 0) and 2gs+1 (chunk 1); one wgmma group per stage.
            // While group gs runs, stage gs+1 is waited for and its A fragments are loaded into the other buffer set.
            const uint32_t SLOTB = (uint32_t)P.SLOTB;
            const uint32_t nbm = (1u << P.NBL) - 1u, nbl = (uint32_t)P.NBL;
            const uint32_t rsm = (1u << P.RSL) - 1u, rsl = (uint32_t)P.RSL;
            const int nstg = nk >> 1;
            int rdy = -1, rel = 0;
            uint32_t A0[NA][4], A1[NA][4], B0[NA][4], B1[NA][4];
            const uint64_t desc0 = make_desc(bst0, 1152, 128);
            // wait for stage gs and its pieces, load both k-steps' A fragments; returns ri_lo(2gs)
            auto stage_in = [&](int gs, uint32_t(&X)[NA][4], uint32_t(&Y)[NA][4]) -> int {
                const uint32_t sq = sseq0 + (uint32_t)gs;
                const uint32_t st = sq & nbm;
                mbar_wait(barFB + 8 * st, (sq >> nbl) & 1);
                const uint32_t hb = bst0 + st * STAGEB + 2304;
                const uint32_t h0 = lds32(hb + 4 * t), h1 = lds32(hb + CHUNK + 4 * t);
                const uint32_t hh = lds32(hb + CHUNK + 12), hl = lds32(hb);
                const int rhi = (int)(hh >> 16);
                while (rdy < rhi) {
                    ++rdy;
                    const uint32_t q = pseq0 + (uint32_t)rdy;
                    mbar_wait(barFR + 8 * (q & rsm), (q >> rsl) & 1);
                }
                const uint32_t pa = ring0 + ((pseq0 + (h0 >> 16)) & rsm) * SLOTB + 2 * (h0 & 0xffffu);
                const uint32_t pb = ring0 + ((pseq0 + (h1 >> 16)) & rsm) * SLOTB + 2 * (h1 & 0xffffu);
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    lds64(pa + Rw[i][0], X[i][0], X[i][2]);
                    lds64(pa + Rw[i][1], X[i][1], X[i][3]);
                    lds64(pb + Rw[i][0], Y[i][0], Y[i][2]);
                    lds64(pb + Rw[i][1], Y[i][1], Y[i][3]);
                }
                return (int)(hl >> 16);
            };
            // issue stage gs with (X, Y); then (stage gs-1 complete) release it and prefetch stage gs+1 into (Xn, Yn)
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int e = 0; e < 36; ++e) acc[i][e] = 0.f;
            int rlo_next = stage_in(0, A0, A1);
            auto body = [&](int gs, uint32_t(&X)[NA][4], uint32_t(&Y)[NA][4], uint32_t(&Xn)[NA][4],
                            uint32_t(&Yn)[NA][4]) -> bool {
                const uint64_t d = desc0 + (uint64_t)((((sseq0 + (uint32_t)gs) & nbm) * STAGEB) >> 4);
                const int rlo_cur = rlo_next;
                if constexpr (NT > 0) {
                    wg_fence();
                    issue_stage<NT>(acc, X, Y, d, d + (uint64_t)(CHUNK >> 4));
                    wg_commit();
                    wg_wait<1>();
                }
                if (gs > 0) {
                    fence_async_smem();
                    __syncwarp();
                    if (lane == 0) {
                        mbar_arrive(barEB + 8 * ((sseq0 + (uint32_t)gs - 1u) & nbm));
                        for (int r = rel; r < rlo_cur; ++r) mbar_arrive(barER + 8 * ((pseq0 + (uint32_t)r) & rsm));
                    }
                    rel = max(rel, rlo_cur);
                }
                if (gs + 1 < nstg) {
                    rlo_next = stage_in(gs + 1, Xn, Yn);
                    return true;
                }
                return false;
            };
            for (int gs = 0;; gs += 2) {
                if (!body(gs, A0, A1, B0, B1)) break;
                if (!body(gs + 1, B0, B1, A0, A1)) break;
            }
            wg_wait<0>();
            fence_async_smem();
            __syncwarp();
            if (lane == 0) {
                mbar_arrive(barEB + 8 * ((sseq0 + (uint32_t)nstg - 1u) & nbm));
                for (int r = rel; r < npc; ++r) mbar_arrive(barER + 8 * ((pseq0 + (uint32_t)r) & rsm));
            }
        } else {
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int e = 0; e < 36; ++e) acc[i][e] = 0.f;
        }
        // epilogue: accumulators -> fp16 staging tile [i][j][64 rows][8 halfs] via stmatrix; the storer warps
        // drain it to out[c, k, ho, 8q .. 8q + 7] while this warpgroup starts the next unit
        mbar_wait(barSE, epar ^ 1u);
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            const uint32_t tb = stg0 + (uint32_t)i * 9216u;
            const uint32_t rowa = (uint32_t)(cw * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * 16u;
#pragma unroll
            for (int j = 0; j < 8; j += 2) {
                const uint32_t addr = tb + (uint32_t)(j + (lane >> 4)) * 1024u + rowa;
                stsm_x4(addr, pack_h2(acc[i][4 * j], acc[i][4 * j + 1]), pack_h2(acc[i][4 * j + 2], acc[i][4 * j + 3]),
                        pack_h2(acc[i][4 * j + 4], acc[i][4 * j + 5]), pack_h2(acc[i][4 * j + 6], acc[i][4 * j + 7]));
            }
            stsm_x2(tb + 8u * 1024u + rowa, pack_h2(acc[i][32], acc[i][33]), pack_h2(acc[i][34], acc[i][35]));
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(barSF);
    }

    // storer: drains the staging tiles of every unit of the pipeline (2 warps, one row of a tile per lane)
    __device__ __forceinline__ void store_units(const MainParams &P, uint32_t stg0, uint32_t barSF, uint32_t barSE,
                                                int sw, int lane, int gpipe, int npipe, int hw)
    {
        const int nunits = P.gp[2], TOT = P.BC * P.Q, Q = P.Q, K = P.K;
        const size_t HoWo = (size_t)P.Ho * P.Wo;
        uint32_t par = 0;
        for (int u = gpipe; u < nunits; u += npipe) {
            const int2 ud = P.utab[u];
            const int row = ud.x;
            if (P.nks[row] < 0) continue;
            const int R0 = ud.y >> 3;
            const int rows_u = min((ud.y & 7) * 64, TOT - R0);
            const int nt = min(3, max(0, (((rows_u + 63) >> 6) - hw + 1) >> 1));
            mbar_wait(barSF, par);
            for (int i = 0; i < nt; ++i) {
                const int rl = (hw + 2 * i) * 64 + sw * 32 + lane;
                if (rl < rows_u) {
                    const int r = R0 + rl;
                    const int c = r / Q, q = r - c * Q;
                    __half *o = P.out + ((size_t)c * K * P.Ho + row) * P.Wo + 8 * q;
                    const uint32_t sa = stg0 + (uint32_t)i * 9216u + (uint32_t)(sw * 32 + lane) * 16u;
#pragma unroll
                    for (int j0 = 0; j0 < 9; j0 += 3) {
                        uint4 v[3];
#pragma unroll
                        for (int j = 0; j < 3; ++j)
                            asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                                         : "=r"(v[j].x), "=r"(v[j].y), "=r"(v[j].z), "=r"(v[j].w)
                                         : "r"(sa + (uint32_t)(j0 + j) * 1024u));
#pragma unroll
                        for (int j = 0; j < 3; ++j)
                            if (j0 + j < K) *reinterpret_cast<uint4 *>(o + (j0 + j) * HoWo) = v[j];
                    }
                }
            }
            __syncwarp();
            if (lane == 0) mbar_arrive(barSE);
            par ^= 1u;
        }
    }

    // rows of the unit computed directly from psi (rows the fast prep rejected)
    __device__ void generic_unit(const MainParams &P, int row, int R0, int rows_u, int tid128, int nthr = 128)
    {
        const long long a = P.off[row], b = P.off[row + 1];
        const size_t HoWo = (size_t)P.Ho * P.Wo;
        for (int e = tid128; e < rows_u * 8; e += nthr) {
            const int r = R0 + (e >> 3);
            const int c = r / P.Q, q = r - c * P.Q;
            const int wo = 8 * q + (e & 7);
            float acc[16];
#pragma unroll
            for (int k = 0; k < 16; ++k) acc[k] = 0.f;
            const __half *xc = P.x + (size_t)c * P.Hi * P.Wi;
            for (long long nn = a; nn < b; ++nn) {
                const int hi = (int)P.idx[2 * nn], wi = (int)P.idx[2 * nn + 1];
                const float xv = __half2float(xc[(size_t)hi * P.Wi + (wi + (long long)wo * P.p) % P.Wi]);
#pragma unroll
                for (int k = 0; k < 16; ++k)
                    if (k < P.K) acc[k] += __half2float(P.val[nn * P.KP + k]) * xv;
            }
#pragma unroll
            for (int k = 0; k < 16; ++k)
                if (k < P.K) P.out[((size_t)c * P.K + k) * HoWo + (size_t)row * P.Wo + wo] = __float2half_rn(acc[k]);
        }
    }

    // 512 threads, one pipeline per CTA: warpgroup 0 = producers (warps 0/1/3: input windows split by channel
    // index mod 3, warp 2: B groups), warpgroups 1 and 2 = consumers of tiles 0-2 / 3-5 of every unit,
    // warpgroup 3 = output storers (warps 12,13 drain warpgroup 1's staging, 14,15 warpgroup 2's).
    __global__ void __launch_bounds__(512, 1) main_kernel(const __grid_constant__ MainParams P)
    {
        extern __shared__ __align__(128) uint8_t smem[];
        const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
        const uint32_t sbase = smem_u32(smem);
        const uint32_t nb = 1u << P.NBL, rs = 1u << P.RSL;
        const uint32_t bst0 = sbase, ring0 = sbase + nb * STAGEB;
        const uint32_t bar0 = ring0 + rs * (uint32_t)P.SLOTB;
        const uint32_t barFB = bar0, barEB = bar0 + 8 * nb, barFR = bar0 + 16 * nb, barER = barFR + 8 * rs;
        const uint32_t barS
            = bar0 + 16 * nb + 16 * rs; // per consumer warpgroup h: SF at barS + 16h, SE at barS + 16h + 8
        const uint32_t stg = (barS + 32 + 127) & ~127u; // per consumer warpgroup: 3 tiles x 9216 B
        if (warp == 0 && lane == 0) {
            for (uint32_t i = 0; i < nb; ++i) {
                mbar_init(barFB + 8 * i, 1);
                mbar_init(barEB + 8 * i, 8);
            }
            for (uint32_t i = 0; i < rs; ++i) {
                mbar_init(barFR + 8 * i, 3);
                mbar_init(barER + 8 * i, 8);
            }
            for (int h = 0; h < 2; ++h) {
                mbar_init(barS + 16 * h, 4);
                mbar_init(barS + 16 * h + 8, 2);
            }
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
        __syncthreads();
        const int gpipe = blockIdx.x, npipe = gridDim.x;
        if (warp >= 12) {
            reg_dealloc<40>();
            const int hw = (warp - 12) >> 1;
            store_units(P, stg + (uint32_t)hw * 27648u, barS + 16 * hw, barS + 16 * hw + 8, (warp - 12) & 1, lane,
                        gpipe, npipe, hw);
            return;
        }
        if (warp < 4) {
            reg_dealloc<40>();
            if (warp == 2)
                produce_B(P, bst0, barFB, barEB, lane, gpipe, npipe);
            else
                produce_pieces(P, ring0, barFR, barER, lane, gpipe, npipe, warp == 3 ? 2 : warp);
            return;
        }
        reg_alloc<216>();
        const int cw = (warp - 4) & 3, hw = (warp - 4) >> 2;
        const uint32_t stg0 = stg + (uint32_t)hw * 27648u, barSF = barS + 16 * hw, barSE = barSF + 8;
        const int nunits = P.gp[2], TOT = P.BC * P.Q;
        uint32_t pseq = 0, sseq = 0, epar = 0;
        int2 ud0 = gpipe < nunits ? P.utab[gpipe] : make_int2(0, 0);
        int2 ud1 = gpipe + npipe < nunits ? P.utab[gpipe + npipe] : make_int2(0, 0);
        int4 ri0 = gpipe < nunits ? P.rinfo[ud0.x] : make_int4(0, 0, 0, 0);
        for (int u = gpipe; u < nunits; u += npipe) {
            const int4 ri1 = u + npipe < nunits ? P.rinfo[ud1.x] : make_int4(0, 0, 0, 0);
            const int2 ud2 = u + 2 * npipe < nunits ? P.utab[u + 2 * npipe] : make_int2(0, 0);
            const int2 ud = ud0;
            const int4 ri = ri0;
            ud0 = ud1;
            ri0 = ri1;
            ud1 = ud2;
            const int row = ud.x;
            const int R0 = ud.y >> 3, nt_u = ud.y & 7;
            const int rows_u = min(nt_u * 64, TOT - R0);
            const int nk = ri.x;
            if (nk < 0) {
                generic_unit(P, row, R0, rows_u, (tid - 128) & 255, 256);
                continue;
            }
            const int npc = ri.y;
            const int c_first = R0 / P.Q;
            const int mine = min(3, max(0, (((rows_u + 63) >> 6) - hw + 1) >> 1));
            switch (mine) {
            case 0:
                consume<0>(P, bst0, ring0, barFB, barEB, barFR, barER, pseq, sseq, nk, npc, row, R0, rows_u, c_first,
                           cw, lane, stg0, barSF, barSE, epar, hw);
                break;
            case 1:
                consume<1>(P, bst0, ring0, barFB, barEB, barFR, barER, pseq, sseq, nk, npc, row, R0, rows_u, c_first,
                           cw, lane, stg0, barSF, barSE, epar, hw);
                break;
            case 2:
                consume<2>(P, bst0, ring0, barFB, barEB, barFR, barER, pseq, sseq, nk, npc, row, R0, rows_u, c_first,
                           cw, lane, stg0, barSF, barSE, epar, hw);
                break;
            default:
                consume<3>(P, bst0, ring0, barFB, barEB, barFR, barER, pseq, sseq, nk, npc, row, R0, rows_u, c_first,
                           cw, lane, stg0, barSF, barSE, epar, hw);
                break;
            }
            if (nk > 0) {
                pseq += (uint32_t)npc;
                sseq += (uint32_t)((nk + SK - 1) / SK);
            }
            epar ^= 1u;
        }
    }

    // ------------------------------------------------------------------------------------------------
    // generic path (shapes outside the fast path): direct CUDA-core evaluation
    // ------------------------------------------------------------------------------------------------
    __global__ void __launch_bounds__(256)
        generic_kernel(const __half *__restrict__ x, const long long *__restrict__ idx, const __half *__restrict__ val,
                       const long long *__restrict__ off, __half *__restrict__ out, int BC, int Hi, int Wi, int Ho,
                       int Wo, int K, int KP, int p)
    {
        const int ho = blockIdx.x;
        const long long a = off[ho], b = off[ho + 1];
        const size_t HoWo = (size_t)Ho * Wo;
        for (long long e = (long long)blockIdx.y * blockDim.x + threadIdx.x; e < (long long)BC * Wo;
             e += (long long)gridDim.y * blockDim.x) {
            const int bc = (int)(e / Wo), wo = (int)(e % Wo);
            float acc[16];
#pragma unroll
            for (int k = 0; k < 16; ++k) acc[k] = 0.f;
            const __half *xc = x + (size_t)bc * Hi * Wi;
            for (long long n = a; n < b; ++n) {
                const int hi = (int)idx[2 * n], wi = (int)idx[2 * n + 1];
                const float xv = __half2float(xc[(size_t)hi * Wi + (wi + (long long)wo * p) % Wi]);
#pragma unroll
                for (int k = 0; k < 16; ++k)
                    if (k < K) acc[k] += __half2float(val[n * KP + k]) * xv;
            }
            for (int k = 0; k < K; ++k)
                out[((size_t)bc * K + k) * HoWo + (size_t)ho * Wo + wo] = __float2half_rn(acc[k]);
        }
    }

    // ------------------------------------------------------------------------------------------------
    // host
    // ------------------------------------------------------------------------------------------------
    static void *g_ws = nullptr;
    static size_t g_ws_size = 0;
    static int g_num_sms = 0;
    static bool g_attr = false;

    static void *ws_get(size_t bytes, cudaStream_t st)
    {
        if (bytes > g_ws_size) {
            if (g_ws) {
                cudaStreamSynchronize(st);
                cudaFree(g_ws);
            }
            if (cudaMalloc(&g_ws, bytes) != cudaSuccess) {
                g_ws = nullptr;
                g_ws_size = 0;
                return nullptr;
            }
            cudaMemsetAsync(g_ws, 0, 256, st); // counter lives in the first 256 bytes
            g_ws_size = bytes;
        }
        return g_ws;
    }

    // true when disco_run takes the Hankel fast path for these sizes (mirrors its plan search)
    static inline bool supported(int BC, int Hi, int Wi, int Wo, int Ho, int K, int KP, long long NNZ)
    {
        (void)Hi;
        if (Ho <= 0 || BC <= 0 || Wo <= 0 || K <= 0 || Wo % 8 != 0) return false;
        const int p = Wi / Wo;
        const int Q = Wo / 8;
        if (!(K <= 9 && K <= KP && p >= 1 && p <= 8 && (long long)p * Wo == Wi && NNZ > 0 && NNZ < (1LL << 31) - 65536
              && (long long)BC * Q < (1LL << 30)))
            return false;
        const int EXT_MAX = (3 + KPMAX + 7 * p + 3) & ~3;
        const int EXTW = (EXT_MAX + 8 + 7) & ~7;
        const int plans[8][3]
            = {{6, 8, 16}, {6, 4, 16}, {6, 8, 8}, {6, 4, 8}, {4, 8, 8}, {4, 4, 8}, {2, 4, 8}, {1, 4, 8}};
        for (int pl = 0; pl < 8; ++pl) {
            const int nt = plans[pl][0], nbv = plans[pl][1], rsv = plans[pl][2];
            const long long ru = 64LL * nt;
            const long long nchw = std::min<long long>(BC, (ru + Q - 2) / Q + 1);
            const long long sb = (2LL * (8LL * p * ru + nchw * EXTW) + 15) & ~15LL;
            const long long pbytes = ((((long long)nbv * STAGEB + rsv * sb + (2 * nbv + 2 * rsv + 4) * 8 + 127) & ~127LL)
                                      + 2 * 27648LL + 127)
                & ~127LL;
            if (nchw <= 96 && pbytes <= 227 * 1024) return true;
        }
        return false;
    }

    void disco_run(ffi::TensorView inp, ffi::TensorView pack_idx, ffi::TensorView pack_val, ffi::TensorView pack_offset,
                   int64_t kernel_size, int64_t nlon_out, ffi::TensorView out)
    {
        const int B = (int)inp.size(0), C = (int)inp.size(1), Hi = (int)inp.size(2), Wi = (int)inp.size(3);
        const long long NNZ = pack_idx.size(0);
        const int KP = (int)pack_val.size(1);
        const int Ho = (int)pack_offset.size(0) - 1;
        const int K = (int)kernel_size, Wo = (int)nlon_out;
        const int BC = B * C;
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
        const int Q = Wo / 8;

        bool fast = K <= 9 && K <= KP && Wo % 8 == 0 && p >= 1 && p <= 8 && (long long)p * Wo == Wi && NNZ > 0
            && NNZ < (1LL << 31) - 65536 && (long long)BC * Q < (1LL << 30);
        // smem plan per pipeline: ring slot = per channel window (8p*rows + EXTW elements), <= NCHW channels
        const int EXT_MAX = (3 + KPMAX + 7 * p + 3) & ~3; // max piece K extent (alignment pad a <= 3)
        const int EXTW = (EXT_MAX + 8 + 7) & ~7;
        int ntmax = 0, nbl = 0, rsl = 0;
        long long slotb = 0, pipeb = 0;
        // preference: deep rings first, then wide units
        const int plans[8][3]
            = {{6, 8, 16}, {6, 4, 16}, {6, 8, 8}, {6, 4, 8}, {4, 8, 8}, {4, 4, 8}, {2, 4, 8}, {1, 4, 8}};
        for (int pl = 0; pl < 8 && fast; ++pl) {
            const int nt = plans[pl][0], nbv = plans[pl][1], rsv = plans[pl][2];
            const long long ru = 64LL * nt;
            const long long nchw = std::min<long long>(BC, (ru + Q - 2) / Q + 1);
            const long long sb = (2LL * (8LL * p * ru + nchw * EXTW) + 15) & ~15LL;
            const long long pbytes = ((((long long)nbv * STAGEB + rsv * sb + (2 * nbv + 2 * rsv + 4) * 8 + 127) & ~127LL)
                                      + 2 * 27648LL + 127)
                & ~127LL;
            if (nchw <= 96 && pbytes <= 227 * 1024) {
                ntmax = nt;
                nbl = nbv == 8 ? 3 : 2;
                rsl = rsv == 16 ? 4 : 3;
                slotb = sb;
                pipeb = pbytes;
                break;
            }
        }
        if (ntmax == 0) fast = false;

        if (fast) {
            const long long ks_bound = (NNZ * (7LL * p + 3)) / 16 + 2LL * Ho + 2;
            auto al = [](size_t v) { return (v + 255) & ~(size_t)255; };
            const size_t sz_ctr = 256;
            const size_t sz_pc = al((size_t)NNZ * sizeof(Piece));
            const size_t sz_i = al((size_t)(Ho + 1) * sizeof(int));
            const size_t sz_B = al((size_t)ks_bound * CHUNK);
            const size_t sz_u = al((size_t)Ho * (size_t)((BC * (long long)Q + 63) / 64) * sizeof(int2));
            const size_t sz_r = al((size_t)Ho * sizeof(int4));
            uint8_t *ws = (uint8_t *)ws_get(sz_ctr + sz_pc + 7 * sz_i + 256 + sz_u + sz_r + sz_B, st);
            if (ws) {
                unsigned *counter = (unsigned *)ws;
                Piece *pieces = (Piece *)(ws + sz_ctr);
                int *npieces = (int *)(ws + sz_ctr + sz_pc);
                int *nks = (int *)(ws + sz_ctr + sz_pc + sz_i);
                int *ks_off = (int *)(ws + sz_ctr + sz_pc + 2 * sz_i);
                int *row_order = (int *)(ws + sz_ctr + sz_pc + 3 * sz_i);
                int *gp = (int *)(ws + sz_ctr + sz_pc + 4 * sz_i);
                int *urow_off = (int *)(ws + sz_ctr + sz_pc + 5 * sz_i);
                int *urow_nt = (int *)(ws + sz_ctr + sz_pc + 6 * sz_i);
                int2 *utab = (int2 *)(ws + sz_ctr + sz_pc + 7 * sz_i + 256);
                int4 *rinfo = (int4 *)(ws + sz_ctr + sz_pc + 7 * sz_i + 256 + sz_u);
                uint8_t *Bg = ws + sz_ctr + sz_pc + 7 * sz_i + 256 + sz_u + sz_r;
                const int nslots = g_num_sms; // one pipeline per CTA
                PrepParams PP;
                PP.idx = idx;
                PP.off = off;
                PP.pieces = pieces;
                PP.npieces = npieces;
                PP.nks = nks;
                PP.ks_off = ks_off;
                PP.row_order = row_order;
                PP.gp = gp;
                PP.urow_off = urow_off;
                PP.urow_nt = urow_nt;
                PP.rinfo = rinfo;
                PP.counter = counter;
                PP.Ho = Ho;
                PP.Wi = Wi;
                PP.p = p;
                PP.BC = BC;
                PP.Q = Q;
                PP.nslots = nslots;
                PP.ntmax = ntmax;
                // heavy rows (whole rings near the poles) go to the FFT path when its cost model beats this path;
                // cdir = SM cycles per K column per channel (9 x Wo MACs per K column at the calibrated rate)
                PP.fftflag = fftp::run(x, idx, val, off, o, BC, Hi, Wi, Ho, Wo, K, KP, NNZ, 7 * p,
                                       64.f * (float)((Wo + 6) / 7) / 1190.f, g_num_sms, st);
                prep_rows_kernel<<<Ho, PT, 0, st>>>(PP);
                FillParams FP;
                FP.val = val;
                FP.urow_off = urow_off;
                FP.urow_nt = urow_nt;
                FP.utab = utab;
                FP.rows_per = (long long)BC * Q;
                FP.off = off;
                FP.pieces = pieces;
                FP.npieces = npieces;
                FP.nks = nks;
                FP.ks_off = ks_off;
                FP.Bg = Bg;
                FP.K = K;
                FP.KP = KP;
                FP.p = p;
                bfill_kernel<<<dim3(Ho, NSPLIT), 128, 0, st>>>(FP);
                MainParams MP;
                MP.x = x;
                MP.out = o;
                MP.val = val;
                MP.idx = idx;
                MP.off = off;
                MP.pieces = pieces;
                MP.npieces = npieces;
                MP.nks = nks;
                MP.ks_off = ks_off;
                MP.utab = utab;
                MP.gp = gp;
                MP.Bg = Bg;
                MP.BC = BC;
                MP.Hi = Hi;
                MP.Wi = Wi;
                MP.Ho = Ho;
                MP.Wo = Wo;
                MP.Q = Q;
                MP.p = p;
                MP.K = K;
                MP.KP = KP;
                MP.EXTW = EXTW;
                MP.SLOTB = (int)slotb;
                MP.PIPEB = (int)pipeb;
                MP.NBL = nbl;
                MP.RSL = rsl;
                MP.rinfo = rinfo;
                if (!g_attr) {
                    cudaFuncSetAttribute(main_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
                    g_attr = true;
                }
                main_kernel<<<g_num_sms, 512, (size_t)pipeb, st>>>(MP);
                return;
            }
        }
        dim3 gg(Ho, (unsigned)std::max<long long>(1, std::min<long long>(((long long)BC * Wo + 255) / 256, 64)));
        generic_kernel<<<gg, 256, 0, st>>>(x, idx, val, off, o, BC, Hi, Wi, Ho, Wo, K, KP, p);
    }

} // namespace hankel
#undef KPMAX
#undef SK
#undef NBMAX
#undef RSMAX
#undef CHUNK
#undef RUNCAP
#undef NTMAX
#undef PT
#undef SORTCAP
#undef NSPLIT
#undef MSCAP
#undef STAGEB
