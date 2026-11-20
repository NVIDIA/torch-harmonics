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

// DISCO S2 forward contraction, psi in the K-packed layout, fp16 in / fp32 accumulate / fp16 out.
//
// Reformulation used by the fast path. For a row ho the points are grouped into segments: runs of
// points with the same input latitude hi and consecutive longitudes (a run that crosses the
// longitude seam is merged into one segment). With S output longitudes per tile and S * pscale == 8,
//   out[c, k, ho, S*t + s] = sum_seg sum_x G_seg[x, (k, s)] * inp[c, hi_seg, (B_seg + 8 t + x) mod Wi]
// with G_seg[x, (k, s)] = V_seg[k][x - s * pscale] (the dense, zero padded filter of the segment).
// This is a dense GEMM per row: M = (channel, t), N = (k, s), K = concatenated segment windows.
// A is read straight from the staged input window (row t+1 is 8 elements = 16 bytes after row t,
// which is exactly the canonical no-swizzle K-major core-matrix layout), B is built in shared
// memory from the dense filter once per stage and shared by the whole CTA. MMA: wgmma m64nNk16.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <algorithm>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/error.h>

namespace leader_small
{

    namespace ffi = tvm::ffi;

#define GMAX 2   // largest longitude gap kept inside one segment
#define PPP 24   // dense-filter positions reserved per point (upper bound)
#define NPMAX 4  // pieces (segment runs) per A-stage
#define NBMAX 24 // 8-row K blocks per filter stage
#define INFOB 96 // bytes of per-stage info (nb, flags, 16-bit block offsets)
#define RMAX 32  // output rows per CTA (upper bound)
#define RCAP 64  // stage records cached in shared memory per CTA
#ifndef NCW_MAX
#define NCW_MAX 1 // consumer warpgroups per CTA (1 or 2)
#endif
#define NCW2_DISCOUNT 0.85

    // ------------------------------------------------------------------------------------------------
    // small helpers
    // ------------------------------------------------------------------------------------------------
    __device__ __forceinline__ uint32_t smem_u32(const void *p)
    {
        return static_cast<uint32_t>(__cvta_generic_to_shared(p));
    }
    __device__ __forceinline__ void cp_async16(uint32_t dst, const void *src)
    {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
    }
    __device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
    template <int N> __device__ __forceinline__ void cp_async_wait()
    {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
    }
    __device__ __forceinline__ void fence_proxy_async()
    {
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    }
    __device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
    __device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
    template <int N> __device__ __forceinline__ void wgmma_wait()
    {
        asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
    }
    __device__ __forceinline__ uint64_t make_desc(uint32_t addr, uint32_t lbo, uint32_t sbo)
    {
        uint64_t d = 0;
        d |= (uint64_t)((addr & 0x3FFFF) >> 4);
        d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
        d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
        return d; // no swizzle, base offset 0
    }

    // halves [d, d + 8) of the 16-half concatenation lo:hi (0 <= d < 8)
    __device__ __forceinline__ uint4 extract8(uint4 lo, uint4 hi, int d)
    {
        uint32_t w0 = lo.x, w1 = lo.y, w2 = lo.z, w3 = lo.w, w4 = hi.x, w5 = hi.y, w6 = hi.z, w7 = hi.w;
        const int q = d >> 1;
        uint32_t a0, a1, a2, a3, a4;
        switch (q) {
        case 0:
            a0 = w0;
            a1 = w1;
            a2 = w2;
            a3 = w3;
            a4 = w4;
            break;
        case 1:
            a0 = w1;
            a1 = w2;
            a2 = w3;
            a3 = w4;
            a4 = w5;
            break;
        case 2:
            a0 = w2;
            a1 = w3;
            a2 = w4;
            a3 = w5;
            a4 = w6;
            break;
        default:
            a0 = w3;
            a1 = w4;
            a2 = w5;
            a3 = w6;
            a4 = w7;
            break;
        }
        uint4 r;
        if (d & 1) {
            r.x = __byte_perm(a0, a1, 0x5432);
            r.y = __byte_perm(a1, a2, 0x5432);
            r.z = __byte_perm(a2, a3, 0x5432);
            r.w = __byte_perm(a3, a4, 0x5432);
        } else {
            r.x = a0;
            r.y = a1;
            r.z = a2;
            r.w = a3;
        }
        return r;
    }

// ------------------------------------------------------------------------------------------------
// prep, part 1: one 1024-thread CTA per output row.
//   segments (runs of points with equal hi and consecutive wi, seam-crossing runs merged),
//   dense-filter positions, zeroed dense-filter region and the stage schedule.
// Stage record (16 ints): R (Vd position of block 0), nb, np, -, hi[4], wbase[4], nbp[4]
// prep, part 2 (disco_scatter_kernel): one thread per point writes its K values into Vd.
// ------------------------------------------------------------------------------------------------
#define PREP_THREADS 1024
#define PREP_WARPS (PREP_THREADS / 32)

    __device__ __forceinline__ int block_incl_scan(int v, int *s_warp, int &total)
    {
        const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int t = __shfl_up_sync(0xffffffffu, v, o);
            if (lane >= o) v += t;
        }
        if (lane == 31) s_warp[w] = v;
        __syncthreads();
        if (w == 0) {
            int x = lane < PREP_WARPS ? s_warp[lane] : 0;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                int t = __shfl_up_sync(0xffffffffu, x, o);
                if (lane >= o) x += t;
            }
            if (lane < PREP_WARPS) s_warp[32 + lane] = x; // inclusive warp prefix
        }
        __syncthreads();
        const int pre = w > 0 ? s_warp[32 + w - 1] : 0;
        total = s_warp[32 + PREP_WARPS - 1];
        __syncthreads();
        return v + pre;
    }

    __device__ __forceinline__ int64_t row_start_pos(const int64_t *poff, int ho)
    {
        return ((poff[ho] * PPP + 7) & ~7LL) + 16LL * ho;
    }

    __global__ void __launch_bounds__(PREP_THREADS)
        disco_prep_kernel(const int64_t *__restrict__ pidx, const int64_t *__restrict__ poff, int K, int Wi, int ps,
                          int Tc, int chunkmax, int64_t TOTPOS,
                          int *__restrict__ gsid,     // [NNZ] global segment index of every point
                          int *__restrict__ seghead,  // [NNZ] (row-local) head point of every segment
                          int *__restrict__ segpos,   // [NNZ] Vd position of wi == 0 for the points of every segment
                          int4 *__restrict__ seginfo, // [NNZ] hi, B, block offset, nblocks
                          __half *__restrict__ Vd, int *__restrict__ stages, int *__restrict__ nstages)
    {
        __shared__ int s_warp[64];
        const int ho = blockIdx.x;
        const int tid = threadIdx.x;
        const int64_t a = poff[ho];
        const int cnt = (int)(poff[ho + 1] - a);
        const int64_t rowstart = row_start_pos(poff, ho);
        const int64_t rowbase = rowstart + 8;
        const int64_t *P = pidx + 2 * a;
        int *shead = seghead + a;
        int *spos = segpos + a;
        int4 *sinfo = seginfo + a;

        // 1. segment starts + ids
        int nseg = 0;
        for (int base = 0; base < cnt; base += PREP_THREADS) {
            const int i = base + tid;
            int f = 0;
            if (i < cnt) {
                const longlong2 cur = *reinterpret_cast<const longlong2 *>(P + 2 * i);
                if (i == 0) {
                    f = 1;
                } else {
                    const longlong2 prv = *reinterpret_cast<const longlong2 *>(P + 2 * i - 2);
                    const int d = (int)(cur.y - prv.y);
                    f = (cur.x != prv.x) || d < 1 || d > GMAX;
                }
            }
            int tot;
            const int inc = block_incl_scan(f, s_warp, tot);
            if (i < cnt) {
                const int s = nseg + inc - 1;
                gsid[a + i] = (int)a + s;
                if (f) shead[s] = i;
            }
            nseg += tot;
        }
        __syncthreads();

        // 2. seam merge + spans. A hi-run's first segment F is absorbed into its last segment L when
        //    the gap across the longitude seam is <= GMAX.
        //    sinfo.w temporarily holds nblocks, sinfo.y = B, sinfo.x = hi, sinfo.z = absorbing segment or -1
        for (int s = tid; s < nseg; s += PREP_THREADS) {
            const int h0 = shead[s];
            const int t0 = (s + 1 < nseg ? shead[s + 1] : cnt) - 1;
            const int hiv = (int)P[2 * h0];
            const bool isF = (h0 == 0) || ((int)P[2 * h0 - 2] != hiv);
            const bool isL = (t0 == cnt - 1) || ((int)P[2 * t0 + 2] != hiv);
            int absorbed = -1, tailseg = -1;
            if (isF && !isL) {
                int l = s + 1;
                while (true) {
                    const int lt = (l + 1 < nseg ? shead[l + 1] : cnt) - 1;
                    if (lt == cnt - 1 || (int)P[2 * lt + 2] != hiv) break;
                    l++;
                }
                const int lt = (l + 1 < nseg ? shead[l + 1] : cnt) - 1;
                if ((int)P[2 * lt + 1] + GMAX >= (int)P[2 * h0 + 1] + Wi) absorbed = l;
            }
            if (isL && !isF) {
                int f = s - 1;
                while (true) {
                    const int fh = shead[f];
                    if (fh == 0 || (int)P[2 * fh - 2] != hiv) break;
                    f--;
                }
                const int fh = shead[f];
                if ((int)P[2 * t0 + 1] + GMAX >= (int)P[2 * fh + 1] + Wi) tailseg = f;
            }
            const int wh = (int)P[2 * h0 + 1];
            int wend = (int)P[2 * t0 + 1];
            if (tailseg >= 0) {
                const int ft = (tailseg + 1 < nseg ? shead[tailseg + 1] : cnt) - 1;
                wend = (int)P[2 * ft + 1] + Wi;
            }
            const int B = wh & ~7;
            const int span = wend - wh + 1;
            const int kseg = ((wh - B) + span + 8 - ps + 7) & ~7;
            sinfo[s] = make_int4(hiv, B, absorbed, absorbed >= 0 ? 0 : (kseg >> 3));
        }
        __syncthreads();

        // 3. block offsets (exclusive scan over segments)
        int NB = 0;
        for (int base = 0; base < nseg; base += PREP_THREADS) {
            const int s = base + tid;
            const int v = s < nseg ? sinfo[s].w : 0;
            int tot;
            const int inc = block_incl_scan(v, s_warp, tot);
            if (s < nseg) spos[s] = NB + inc - v; // temporarily: block offset
            NB += tot;
        }
        __syncthreads();
        // 4. finalize: seginfo.z = block offset, segpos = Vd position of wi == 0
        for (int s = tid; s < nseg; s += PREP_THREADS) {
            const int4 si = sinfo[s];
            const int ab = si.z;
            int pos;
            if (ab >= 0)
                pos = (int)(rowbase + 8LL * spos[ab]) - sinfo[ab].y + Wi;
            else
                pos = (int)(rowbase + 8LL * spos[s]) - si.y;
            seghead[a + s] = pos; // reuse the head storage (heads are no longer needed)
        }
        __syncthreads();
        for (int s = tid; s < nseg; s += PREP_THREADS) {
            const int boff = spos[s];
            spos[s] = seghead[a + s];
            int4 si = sinfo[s];
            si.z = boff;
            sinfo[s] = si;
        }

        // 5. zero the dense filter of the row: positions [rowstart, rowbase + 8 NB)
        {
            const int nchunk = 1 + NB; // 8-position chunks
            for (int idx = tid; idx < K * nchunk; idx += PREP_THREADS) {
                const int k = idx / nchunk, j = idx - k * nchunk;
                *reinterpret_cast<uint4 *>(Vd + (size_t)k * TOTPOS + rowstart + 8 * j) = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // 6. A-stage schedule (single thread). An A-stage is a set of <= NPMAX pieces (runs of blocks of one
        //    segment) whose input windows fit the stage buffer: sum over pieces of (Tc + nbp - 1) <= chunkmax.
        if (tid == 0) {
            int *st = stages + (size_t)(rowstart >> 3) * 16;
            int ns = 0, ngs = 0, cnb = 0, cnp = 0, cch = 0;
            int rec[16];
#pragma unroll
            for (int q = 0; q < 16; q++) rec[q] = 0;
            int4 nxt = nseg > 0 ? sinfo[0] : make_int4(0, 0, 0, 0);
            for (int s = 0; s < nseg; s++) {
                const int4 si = nxt;
                if (s + 1 < nseg) nxt = sinfo[s + 1];
                const int nb = si.w;
                if (nb == 0) continue;
                const int hiv = si.x, B = si.y, bo = si.z;
                int done = 0;
                while (done < nb) {
                    int take = min(nb - done, chunkmax - cch - (Tc - 1));
                    if (take <= 0 || cnp == NPMAX) {
                        // close the current stage
                        rec[1] = cnb;
                        rec[2] = cnp;
                        int4 *o = reinterpret_cast<int4 *>(st + 16 * ns);
                        o[0] = make_int4(rec[0], rec[1], rec[2], rec[3]);
                        o[1] = make_int4(rec[4], rec[5], rec[6], rec[7]);
                        o[2] = make_int4(rec[8], rec[9], rec[10], rec[11]);
                        o[3] = make_int4(rec[12], rec[13], rec[14], rec[15]);
                        ns++;
                        ngs += (cnb + NBMAX - 1) / NBMAX;
                        cnb = 0;
                        cnp = 0;
                        cch = 0;
                        continue;
                    }
                    if (cnb == 0) {
                        rec[0] = (int)(rowbase + 8LL * (bo + done));
#pragma unroll
                        for (int q = 1; q < 16; q++) rec[q] = 0;
                    }
                    int wb = B + 8 * done;
                    wb %= Wi;
                    switch (cnp) {
                    case 0:
                        rec[4] = hiv;
                        rec[8] = wb;
                        rec[12] = take;
                        break;
                    case 1:
                        rec[5] = hiv;
                        rec[9] = wb;
                        rec[13] = take;
                        break;
                    case 2:
                        rec[6] = hiv;
                        rec[10] = wb;
                        rec[14] = take;
                        break;
                    default:
                        rec[7] = hiv;
                        rec[11] = wb;
                        rec[15] = take;
                        break;
                    }
                    cnp++;
                    cnb += take;
                    cch += Tc - 1 + take;
                    done += take;
                }
            }
            if (cnb > 0) {
                rec[1] = cnb;
                rec[2] = cnp;
                int4 *o = reinterpret_cast<int4 *>(st + 16 * ns);
                o[0] = make_int4(rec[0], rec[1], rec[2], rec[3]);
                o[1] = make_int4(rec[4], rec[5], rec[6], rec[7]);
                o[2] = make_int4(rec[8], rec[9], rec[10], rec[11]);
                o[3] = make_int4(rec[12], rec[13], rec[14], rec[15]);
                ns++;
                ngs += (cnb + NBMAX - 1) / NBMAX;
            }
            reinterpret_cast<int4 *>(nstages)[ho] = make_int4((int)(rowstart >> 3), ns, ngs, 0);
        }
    }

    __global__ void __launch_bounds__(256)
        disco_scatter_kernel(const int64_t *__restrict__ pidx, const __half *__restrict__ pval,
                             const int *__restrict__ gsid, const int *__restrict__ segpos, int64_t NNZ, int K, int KP,
                             int64_t TOTPOS, __half *__restrict__ Vd)
    {
        const int64_t n = (int64_t)blockIdx.x * 256 + threadIdx.x;
        if (n >= NNZ) return;
        const int64_t pos = (int64_t)segpos[gsid[n]] + pidx[2 * n + 1];
        const __half *v = pval + (size_t)n * KP;
        for (int k = 0; k < K; k++) Vd[(size_t)k * TOTPOS + pos] = v[k];
    }

    // ------------------------------------------------------------------------------------------------
    // main kernel (warp specialized)
    // CTA = 2 warpgroups (wgmma, 2 m64 tiles each -> 256 GEMM rows). Warp 0 also acts as producer:
    // between its asynchronous wgmma batches it streams every stage with 1-D bulk copies (input windows + dense filter rows),
    // builds the shifted filter B = G[(block, n)][8] in smem and hands the stage over with mbarriers.
    // Tile layouts:
    //   mode 0 (channel-major): Cc = 8*gc channels x Tc = 8*gt tiles; core matrix j = channel
    //   mode 1 (t-major):       Cc = gc channels     x Tc = 64*gt tiles; core matrix j = 8 t-tiles
    // ------------------------------------------------------------------------------------------------
    struct MainParams {
        const __half *inp;
        const __half *Vd;
        const int *stages;
        const int *nstages;
        const int64_t *poff;
        __half *out;
        int64_t TOTPOS;
        int BC, Hi, Wi, Wo, Ho, K, ps, T;
        int mode, Cc, Tc, nTB;
        int CW;         // bytes per channel row of the A stage buffer
        int ABYTES;     // Cc * CW
        int VBYTES;     // K * (NBMAX + 1) * 16
        int na, ng, nv; // A-window slots, filter (B) slots, dense-filter-row slots
        int R;          // output rows per CTA
        int nCB, nItems;
    };

    template <int N> struct WG; // wgmma m64nNk16 f32 += f16 * f16, both operands from smem descriptors

    template <> struct WG<8> {
        __device__ static __forceinline__ void mma(float (&d)[4], uint64_t da, uint64_t db)
        {
            asm volatile(
                "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
                "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, %4, %5, p, 1, 1, 0, 0;\n}\n"
                : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                : "l"(da), "l"(db), "r"(1));
        }
    };

    template <> struct WG<16> {
        __device__ static __forceinline__ void mma(float (&d)[8], uint64_t da, uint64_t db)
        {
            asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\n"
                         "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, "
                         "p, 1, 1, 0, 0;\n}\n"
                         : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                         : "l"(da), "l"(db), "r"(1));
        }
    };

    template <> struct WG<24> {
        __device__ static __forceinline__ void mma(float (&d)[12], uint64_t da, uint64_t db)
        {
            asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %14, 0;\n"
                         "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, "
                         "%10, %11}, %12, %13, p, 1, 1, 0, 0;\n}\n"
                         : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
                           "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
                         : "l"(da), "l"(db), "r"(1));
        }
    };

    template <> struct WG<32> {
        __device__ static __forceinline__ void mma(float (&d)[16], uint64_t da, uint64_t db)
        {
            asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %18, 0;\n"
                         "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, "
                         "%10, %11, %12, %13, %14, %15}, %16, %17, p, 1, 1, 0, 0;\n}\n"
                         : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
                           "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
                           "+f"(d[14]), "+f"(d[15])
                         : "l"(da), "l"(db), "r"(1));
        }
    };

    template <> struct WG<40> {
        __device__ static __forceinline__ void mma(float (&d)[20], uint64_t da, uint64_t db)
        {
            asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %22, 0;\n"
                         "wgmma.mma_async.sync.aligned.m64n40k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, "
                         "%10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, %20, %21, p, 1, 1, 0, 0;\n}\n"
                         : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
                           "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
                           "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19])
                         : "l"(da), "l"(db), "r"(1));
        }
    };

    template <> struct WG<64> {
        __device__ static __forceinline__ void mma(float (&d)[32], uint64_t da, uint64_t db)
        {
            asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
                         "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, "
                         "%10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
                         "%28, %29, %30, %31}, %32, %33, p, 1, 1, 0, 0;\n}\n"
                         : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
                           "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
                           "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
                           "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
                           "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                         : "l"(da), "l"(db), "r"(1));
        }
    };

    template <> struct WG<72> {
        __device__ static __forceinline__ void mma(float (&d)[36], uint64_t da, uint64_t db)
        {
            asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %38, 0;\n"
                         "wgmma.mma_async.sync.aligned.m64n72k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, "
                         "%10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
                         "%28, %29, %30, %31, %32, %33, %34, %35}, %36, %37, p, 1, 1, 0, 0;\n}\n"
                         : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                           "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]),
                           "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]),
                           "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]),
                           "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
                         : "l"(da), "l"(db), "r"(1));
        }
    };

    static inline bool fast_pair_supported(int N, int S)
    {
        return (N == 72 && S == 8) || (N == 40 && S == 4) || (N == 24 && S == 2) || (N == 16 && S == 1)
            || (N == 64 && S == 8) || (N == 32 && S == 4) || (N == 16 && S == 2) || (N == 8 && S == 1);
    }

#define DISPATCH_MAIN(N_, S_, mp_, ncb_, st_)                                                                          \
    do {                                                                                                               \
        if ((N_) == 72 && (S_) == 8) {                                                                                 \
            launch_main<72, 8, 1>(mp_, ncb_, st_);                                                                     \
            break;                                                                                                     \
        }                                                                                                              \
        if ((N_) == 40 && (S_) == 4) {                                                                                 \
            launch_main<40, 4, 1>(mp_, ncb_, st_);                                                                     \
            break;                                                                                                     \
        }                                                                                                              \
        if ((N_) == 24 && (S_) == 2) {                                                                                 \
            launch_main<24, 2, 1>(mp_, ncb_, st_);                                                                     \
            break;                                                                                                     \
        }                                                                                                              \
        if ((N_) == 16 && (S_) == 1) {                                                                                 \
            launch_main<16, 1, 1>(mp_, ncb_, st_);                                                                     \
            break;                                                                                                     \
        }                                                                                                              \
        if ((N_) == 64 && (S_) == 8) {                                                                                 \
            launch_main<64, 8, 1>(mp_, ncb_, st_);                                                                     \
            break;                                                                                                     \
        }                                                                                                              \
        if ((N_) == 32 && (S_) == 4) {                                                                                 \
            launch_main<32, 4, 1>(mp_, ncb_, st_);                                                                     \
            break;                                                                                                     \
        }                                                                                                              \
        if ((N_) == 16 && (S_) == 2) {                                                                                 \
            launch_main<16, 2, 1>(mp_, ncb_, st_);                                                                     \
            break;                                                                                                     \
        }                                                                                                              \
        if ((N_) == 8 && (S_) == 1) {                                                                                  \
            launch_main<8, 1, 1>(mp_, ncb_, st_);                                                                      \
            break;                                                                                                     \
        }                                                                                                              \
    } while (0)
#define DISPATCH_MAIN2(N_, S_, mp_, ncb_, st_)                                                                         \
    do {                                                                                                               \
        if ((N_) == 72 && (S_) == 8) {                                                                                 \
            launch_main<72, 8, 2>(mp_, ncb_, st_);                                                                     \
            break;                                                                                                     \
        }                                                                                                              \
        if ((N_) == 40 && (S_) == 4) {                                                                                 \
            launch_main<40, 4, 2>(mp_, ncb_, st_);                                                                     \
            break;                                                                                                     \
        }                                                                                                              \
        if ((N_) == 24 && (S_) == 2) {                                                                                 \
            launch_main<24, 2, 2>(mp_, ncb_, st_);                                                                     \
            break;                                                                                                     \
        }                                                                                                              \
        if ((N_) == 16 && (S_) == 1) {                                                                                 \
            launch_main<16, 1, 2>(mp_, ncb_, st_);                                                                     \
            break;                                                                                                     \
        }                                                                                                              \
        if ((N_) == 64 && (S_) == 8) {                                                                                 \
            launch_main<64, 8, 2>(mp_, ncb_, st_);                                                                     \
            break;                                                                                                     \
        }                                                                                                              \
        if ((N_) == 32 && (S_) == 4) {                                                                                 \
            launch_main<32, 4, 2>(mp_, ncb_, st_);                                                                     \
            break;                                                                                                     \
        }                                                                                                              \
        if ((N_) == 16 && (S_) == 2) {                                                                                 \
            launch_main<16, 2, 2>(mp_, ncb_, st_);                                                                     \
            break;                                                                                                     \
        }                                                                                                              \
        if ((N_) == 8 && (S_) == 1) {                                                                                  \
            launch_main<8, 1, 2>(mp_, ncb_, st_);                                                                      \
            break;                                                                                                     \
        }                                                                                                              \
    } while (0)

    __device__ __forceinline__ void mbar_init(uint32_t bar, int count)
    {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count));
    }
    __device__ __forceinline__ void mbar_arrive(uint32_t bar)
    {
        asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" ::"r"(bar) : "memory");
    }
    __device__ __forceinline__ void mbar_arrive_tx(uint32_t bar, uint32_t tx)
    {
        asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}\n" ::"r"(bar), "r"(tx)
                     : "memory");
    }
    __device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity)
    {
        asm volatile("{\n.reg .pred p;\nWAIT_%=:\n"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, 0x989680;\n"
                     "@!p bra WAIT_%=;\n}\n" ::"r"(bar),
                     "r"(parity)
                     : "memory");
    }
    __device__ __forceinline__ void leader_bulk_g2s(uint32_t dst, const void *src, uint32_t bytes, uint32_t bar)
    {
        asm volatile(
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dst),
            "l"(src), "r"(bytes), "r"(bar)
            : "memory");
    }

    // halves [D, D + 8) of lo:hi for a compile-time D in [1, 7]
    template <int D> __device__ __forceinline__ uint4 extract8c(const uint4 &lo, const uint4 &hi)
    {
        const uint32_t w[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        constexpr int q = D >> 1;
        uint4 r;
        if constexpr (D & 1) {
            r.x = __byte_perm(w[q], w[q + 1], 0x5432);
            r.y = __byte_perm(w[q + 1], w[q + 2], 0x5432);
            r.z = __byte_perm(w[q + 2], w[q + 3], 0x5432);
            r.w = __byte_perm(w[q + 3], w[q + 4], 0x5432);
        } else {
            r.x = w[q];
            r.y = w[q + 1];
            r.z = w[q + 2];
            r.w = w[q + 3];
        }
        return r;
    }

    template <int S, int s> __device__ __forceinline__ void put_shifts(uint8_t *gdst, const uint4 &lo, const uint4 &hi)
    {
        if constexpr (s < S) {
            constexpr int PS = 8 / S;
            uint4 v;
            if constexpr (s == 0)
                v = hi;
            else
                v = extract8c<8 - s * PS>(lo, hi);
            *reinterpret_cast<uint4 *>(gdst + s * 16) = v;
            put_shifts<S, s + 1>(gdst, lo, hi);
        }
    }

    template <int N, int S, int NCW>
    __global__ void __launch_bounds__(128 * (NCW + 1), NCW == 1 ? 2 : 1) disco_main_kernel(const MainParams p)
    {
        constexpr int NT = 128 * (NCW + 1); // threads: NCW consumer warpgroups + 1 producer warpgroup
        constexpr int NR = N / 2;           // accumulator registers per m64 tile
        constexpr int GST = N * 16 + 16;    // bytes per 8-row block of B (padded: conflict-free building)
        extern __shared__ __align__(128) uint8_t smem[];
        const int tid = threadIdx.x;
        const int warp_id = __shfl_sync(0xffffffffu, tid >> 5, 0), lane = tid & 31;
        const int NA = p.na, NG = p.ng;
        const int Cc = p.Cc, Tc = p.Tc, CW = p.CW;
        const int Kk = p.K;

        // this CTA: one (channel, longitude) tile for the output rows of one row group
        const int tb = blockIdx.x % p.nTB;
        const int cb = blockIdx.x / p.nTB;
        const int c0 = cb * Cc;
        const int t0 = tb * Tc;
        const int nGroups = (p.Ho + p.R - 1) / p.R;
        const int yi = blockIdx.y; // heavy-first: groups from both ends towards the middle
        const int grp = (yi & 1) ? (nGroups - 1 - (yi >> 1)) : (yi >> 1);
        const int ho0 = grp * p.R;
        const int nR = min(p.R, p.Ho - ho0);

        // layout: [NA A slots][NG G slots][NV V slots][zero block][stage records[RCAP]][row info[RMAX]][lastG[8]][barriers]
        const int NV = p.nv;
        const int GSB = NBMAX * GST + INFOB; // G slot: B blocks, info
        uint8_t *gring = smem + NA * p.ABYTES;
        uint8_t *vring = gring + NG * GSB;
        uint8_t *zblk = vring + NV * p.VBYTES;
        int4 *rcache = reinterpret_cast<int4 *>(zblk + GST);
        int4 *rs = rcache + 4 * RCAP;
        int *lastG = reinterpret_cast<int *>(rs + RMAX);
        const uint32_t bar0 = smem_u32(lastG + 8);
#define A_FULL(a) (bar0 + 8 * (a))
#define A_EMPTY(a) (bar0 + 8 * (NA + (a)))
#define V_LOAD(v) (bar0 + 8 * (2 * NA + (v)))
#define V_FREE(v) (bar0 + 8 * (2 * NA + NV + (v)))
#define G_FULL(g) (bar0 + 8 * (2 * NA + 2 * NV + (g)))
#define G_EMPTY(g) (bar0 + 8 * (2 * NA + 2 * NV + NG + (g)))

        // row info (stage slot, nA, nG, first cached record) and the stage records of the group's rows
        if (tid < nR) rs[tid] = __ldg(reinterpret_cast<const int4 *>(p.nstages) + ho0 + tid);
        __syncthreads();
        if (tid == 0) {
            int tot = 0;
            for (int r = 0; r < nR; r++) {
                rs[r].w = tot;
                tot += rs[r].y;
            }
            lastG[7] = tot;
        }
        __syncthreads();
        const bool cached = lastG[7] <= RCAP;
        if (cached) {
            for (int r = 0; r < nR; r++) {
                const int4 ri = rs[r];
                for (int i = tid; i < 4 * ri.y; i += NT)
                    rcache[4 * ri.w + i] = __ldg(reinterpret_cast<const int4 *>(p.stages) + (size_t)ri.x * 4 + i);
            }
        }
        if (tid < 2 * NA + 2 * NV + 2 * NG) {
            // A_FULL, V_LOAD: 1 (expect_tx); V_FREE, G_FULL: 4 producer warps; A_EMPTY, G_EMPTY: all consumer warps
            const bool one = tid < NA || (tid >= 2 * NA && tid < 2 * NA + NV);
            const bool prod = (tid >= 2 * NA + NV && tid < 2 * NA + 2 * NV)
                || (tid >= 2 * NA + 2 * NV && tid < 2 * NA + 2 * NV + NG);
            mbar_init(bar0 + 8 * tid, one ? 1 : prod ? 3 : 4 * NCW);
        }
        for (int i = tid; i < GST / 16; i += NT) reinterpret_cast<uint4 *>(zblk)[i] = make_uint4(0, 0, 0, 0);
        // padding columns n in [K*S, N) of every B buffer are constant zeros
        if (Kk * S < N) {
            const int npad = N - Kk * S;
            for (int i = tid; i < NG * NBMAX * npad; i += NT) {
                const int g = i / (NBMAX * npad), r = i - g * (NBMAX * npad);
                const int blk = r / npad, n = Kk * S + (r - blk * npad);
                *reinterpret_cast<uint4 *>(gring + g * GSB + blk * GST + n * 16) = make_uint4(0, 0, 0, 0);
            }
        }
        fence_proxy_async();
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        __syncthreads();

        if (warp_id >= 4 * NCW) {
            // ============================== producer warpgroup ==============================
            if constexpr (NCW == 1)
                asm volatile("setmaxnreg.dec.sync.aligned.u32 56;\n" ::: "memory");
            else
                asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
            const int pw = warp_id - 4 * NCW;
            const size_t chan_stride = (size_t)p.Hi * p.Wi;
            auto recp = [&](int r, int ia) -> const int * {
                return cached ? reinterpret_cast<const int *>(rcache + 4 * (rs[r].w + ia)) :
                                p.stages + ((size_t)rs[r].x + ia) * 16;
            };
            // flattened cursor over the A-stages (and their 16-block G-stages) of the rows of the group
            struct Cur {
                int r, ia, gs;
            };
            auto norm = [&](Cur &c) {
                while (c.r < nR && c.ia >= rs[c.r].y) {
                    c.r++;
                    c.ia = 0;
                }
            };
            // A-stage `ag` (global index) -> slot ag % NA: input windows of its pieces for the Cc channels
            auto load_A = [&](const Cur &c, int ag) {
                const int a = ag % NA;
                const int *rp = recp(c.r, c.ia);
                const int4 r0 = reinterpret_cast<const int4 *>(rp)[0], rh = reinterpret_cast<const int4 *>(rp)[1];
                const int4 rw = reinterpret_cast<const int4 *>(rp)[2], rn = reinterpret_cast<const int4 *>(rp)[3];
                const int np = r0.z;
                const int L0 = np > 0 ? Tc + rn.x - 1 : 0, L1 = np > 1 ? Tc + rn.y - 1 : 0;
                const int L2 = np > 2 ? Tc + rn.z - 1 : 0, L3 = np > 3 ? Tc + rn.w - 1 : 0;
                const uint32_t tx = (uint32_t)(Cc * (L0 + L1 + L2 + L3) * 16);
                const uint32_t fl = A_FULL(a);
                if (lane == 0) mbar_arrive_tx(fl, tx);
                __syncwarp();
                const uint32_t aBase = smem_u32(smem + a * p.ABYTES);
                const int npairs = np * Cc;
                for (int pr = lane; pr < npairs; pr += 32) {
                    const int pc = pr / Cc, cl = pr - pc * Cc;
                    const int hi = pc == 0 ? rh.x : pc == 1 ? rh.y : pc == 2 ? rh.z : rh.w;
                    const int wb = pc == 0 ? rw.x : pc == 1 ? rw.y : pc == 2 ? rw.z : rw.w;
                    const int off = (pc > 0 ? L0 : 0) + (pc > 1 ? L1 : 0) + (pc > 2 ? L2 : 0);
                    const int L = pc == 0 ? L0 : pc == 1 ? L1 : pc == 2 ? L2 : L3;
                    int c = c0 + cl;
                    c = c < p.BC ? c : p.BC - 1;
                    const __half *row = p.inp + (size_t)c * chan_stride + (size_t)hi * p.Wi;
                    int pos = (wb + 8 * t0) % p.Wi;
                    int rem = L * 8; // positions
                    uint32_t dst = aBase + cl * CW + off * 16;
                    while (rem > 0) {
                        const int n = min(rem, p.Wi - pos);
                        leader_bulk_g2s(dst, row + pos, (uint32_t)n * 2, fl);
                        dst += n * 2;
                        rem -= n;
                        pos = 0;
                    }
                }
            };
            // G-stage j (= sub-stage c.gs of an A-stage) -> V slot j % NV: dense filter rows of its blocks
            auto load_V = [&](const Cur &c, int j) {
                const int v = j % NV;
                const int *rp = recp(c.r, c.ia);
                const int R = rp[0] + 8 * NBMAX * c.gs;
                const int nb = min(NBMAX, rp[1] - NBMAX * c.gs);
                const uint32_t fl = V_LOAD(v);
                if (lane == 0) mbar_arrive_tx(fl, (uint32_t)(Kk * (nb + 1) * 16));
                __syncwarp();
                const uint32_t vBase = smem_u32(vring + v * p.VBYTES);
                for (int k = lane; k < Kk; k += 32)
                    leader_bulk_g2s(vBase + k * (NBMAX + 1) * 16, p.Vd + (size_t)k * p.TOTPOS + (R - 8),
                                    (uint32_t)(nb + 1) * 16, fl);
            };
            // every producer warp builds filter rows k = pw + 4 i; warp 4 adds the info
            auto build_G = [&](const Cur &c, int j, int ag) {
                const int g = j % NG;
                const int *rp = recp(c.r, c.ia);
                const int4 r0 = reinterpret_cast<const int4 *>(rp)[0];
                const int nbA = r0.y;
                const int nb = min(NBMAX, nbA - NBMAX * c.gs);
                if (j >= NG) mbar_wait(G_EMPTY(g), ((j / NG) - 1) & 1); // consumers are done with stage j - NG
                const int v = j % NV;
                mbar_wait(V_LOAD(v), (j / NV) & 1);
                const uint8_t *V = vring + v * p.VBYTES;
                uint8_t *G = gring + g * GSB;
                for (int task = lane; task < NBMAX * 6; task += 32) {
                    const int blk = task % NBMAX;
                    const int k = (pw - 1) + 3 * (task / NBMAX); // K <= 16 (6 rows of blocks per warp cover 18 rows)
                    if (blk < nb && k < Kk) {
                        const uint4 *src = reinterpret_cast<const uint4 *>(V + (k * (NBMAX + 1) + blk) * 16);
                        const uint4 lo = src[0], hi = src[1];
                        put_shifts<S, 0>(G + blk * GST + k * S * 16, lo, hi);
                    }
                }
                // info: nb, A slot/phase/first/last, chunk offset of every block (16 bit each)
                if (pw == 1 && lane < NBMAX) {
                    const int4 rn = reinterpret_cast<const int4 *>(rp)[3];
                    const int np = r0.z;
                    const int j8 = NBMAX * c.gs + (lane < nb ? lane : nb - 1);
                    int off = 0, bs = 0, bo = 0;
#pragma unroll
                    for (int pc = 0; pc < NPMAX; pc++) {
                        const int nbp = (pc < np) ? (pc == 0 ? rn.x : pc == 1 ? rn.y : pc == 2 ? rn.z : rn.w) : 0;
                        if (j8 >= bs && j8 < bs + nbp) bo = off + (j8 - bs);
                        off += nbp ? (Tc + nbp - 1) : 0;
                        bs += nbp;
                    }
                    int *info = reinterpret_cast<int *>(G + NBMAX * GST);
                    if (lane == 0) {
                        info[0] = nb;
                        info[1] = (ag % NA) | (((ag / NA) & 1) << 8) | ((c.gs == 0) << 16)
                            | ((NBMAX * (c.gs + 1) >= nbA) << 17);
                    }
                    reinterpret_cast<unsigned short *>(info + 4)[lane] = (unsigned short)bo;
                }
                fence_proxy_async();
                __syncwarp();
                if (lane == 0) {
                    mbar_arrive(G_FULL(g));
                    mbar_arrive(V_FREE(v));
                }
            };
            // advance a G cursor; returns true when it moved to the next A-stage
            auto advG = [&](Cur &c) -> bool {
                if (NBMAX * (++c.gs) >= recp(c.r, c.ia)[1]) {
                    c.gs = 0;
                    c.ia++;
                    norm(c);
                    return true;
                }
                return false;
            };

            int totG = 0;
            for (int r = 0; r < nR; r++) totG += rs[r].z;
            Cur ca = {0, 0, 0}, cv = {0, 0, 0}, cbd = {0, 0, 0};
            norm(ca);
            cv = ca;
            cbd = ca;
            int aL = 0, gfirst = 0;                 // A-stages issued, first G-stage of A-stage aL
            int vJ = 0, bJ = 0, bAg = 0, done = -1; // V loads issued, stages built, A index of build cursor
            // warp 4 only issues bulk copies (it never executes the proxy fence, which would make it wait for
            // its own in-flight copies); warps 5..7 build the shifted filters and hand them to the consumers
            while (bJ < totG) {
                if (pw == 0) {
                    // dense-filter rows: up to NV stages ahead of the builders (V slots are freed by the builds)
                    while (vJ < totG && vJ <= bJ + NV - 1) {
                        if (vJ >= NV) mbar_wait(V_FREE(vJ % NV), ((vJ / NV) - 1) & 1);
                        load_V(cv, vJ);
                        advG(cv);
                        vJ++;
                    }
                    // A windows: load an A-stage once its first G-stage is inside the V lookahead. Blocking on the
                    // slot is safe: all stages of the previous occupant already have their V rows in flight.
                    while (ca.r < nR) {
                        const int a = aL % NA;
                        if (aL >= NA) {
                            if (gfirst > bJ + NV - 1) break;
                            mbar_wait(A_EMPTY(a), ((aL / NA) - 1) & 1);
                        }
                        const int nbA = recp(ca.r, ca.ia)[1];
                        const int ng = (nbA + NBMAX - 1) / NBMAX;
                        load_A(ca, aL);
                        gfirst += ng;
                        aL++;
                        ca.ia++;
                        norm(ca);
                    }
                    bJ++; // pacing counter: the loader runs up to NV stages ahead of the builders
                } else {
                    build_G(cbd, bJ, bAg);
                    if (advG(cbd)) bAg++;
                    bJ++;
                }
            }
        } else {
            // ============================== consumer warpgroups ==============================
            if constexpr (NCW == 1)
                asm volatile("setmaxnreg.inc.sync.aligned.u32 184;\n" ::: "memory");
            else
                asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
            const int cw = warp_id >> 2; // consumer warpgroup: tiles 4 cw .. 4 cw + 3
            int tile_off[4];
            uint32_t sbo;
#pragma unroll
            for (int qq = 0; qq < 4; qq++) {
                const int q = 4 * cw + qq;
                int &to = tile_off[qq];
                if (p.mode == 0) {
                    const int gt = Tc >> 3;
                    const int cg = q / gt, tg = q - cg * gt;
                    to = cg * 8 * CW + tg * 8 * 16;
                } else {
                    const int gt = Tc >> 6;
                    const int c = q / gt, tg = q - c * gt;
                    to = c * CW + tg * 64 * 16;
                }
            }
            sbo = (p.mode == 0) ? (uint32_t)CW : 128u;
            const uint32_t zaddr = smem_u32(zblk);
            const int warp = warp_id & 3;
            const size_t HW = (size_t)p.Ho * p.Wo;

            int j = 0;           // global G-stage
            int prev_flags = -1; // flags of the previous, not yet released G-stage (-1: none)
            for (int r = 0; r < nR; r++) {
                const int ho = ho0 + r;
                const int nG = rs[r].z;
                float acc[4][NR];
#pragma unroll
                for (int q = 0; q < 4; q++)
#pragma unroll
                    for (int x = 0; x < NR; x++) acc[q][x] = 0.f;

                for (int jj = 0; jj < nG; jj++, j++) {
                    const int g = j % NG;
                    mbar_wait(G_FULL(g), (j / NG) & 1);
                    uint8_t *sb = gring + g * GSB;
                    const uint32_t gBase = smem_u32(sb);
                    const int *info = reinterpret_cast<const int *>(sb + NBMAX * GST);
                    const int nb = __shfl_sync(0xffffffffu, info[0], 0);
                    const int flags = __shfl_sync(0xffffffffu, info[1], 0);
                    const int a = flags & 0xff;
                    if (flags & (1 << 16)) mbar_wait(A_FULL(a), (flags >> 8) & 1);
                    const uint32_t aBase = smem_u32(smem + a * p.ABYTES);
                    const int nsteps = (nb + 1) >> 1;
                    const uint4 bov0 = reinterpret_cast<const uint4 *>(info + 4)[0];
                    const uint4 bov1 = reinterpret_cast<const uint4 *>(info + 4)[1];
                    const uint4 bov2 = reinterpret_cast<const uint4 *>(info + 4)[2];
                    const uint32_t bw[12] = {bov0.x, bov0.y, bov0.z, bov0.w, bov1.x, bov1.y,
                                             bov1.z, bov1.w, bov2.x, bov2.y, bov2.z, bov2.w};
                    wgmma_fence();
#pragma unroll
                    for (int u = 0; u < NBMAX / 2; u++) {
                        if (u < nsteps) {
                            const int boa = bw[u] & 0xffff, bob = bw[u] >> 16;
                            const bool pair = 2 * u + 1 < nb;
                            const uint32_t ga = gBase + 2 * u * GST;
                            const uint32_t lbo_b = pair ? (uint32_t)GST : (zaddr - ga);
                            const uint32_t lbo_a = pair ? (uint32_t)(bob - boa) * 16u : 0u;
                            const uint64_t db = make_desc(ga, lbo_b, 128);
#pragma unroll
                            for (int q = 0; q < 4; q++) {
                                const uint64_t da = make_desc(aBase + tile_off[q] + boa * 16, lbo_a, sbo);
                                WG<N>::mma(acc[q], da, db);
                            }
                        }
                    }
                    wgmma_commit();
                    wgmma_wait<1>();
                    if (prev_flags >= 0 && lane == 0) {
                        // G-stage j-1 has completed in this warp
                        mbar_arrive(G_EMPTY((j - 1) % NG));
                        if (prev_flags & (1 << 17)) mbar_arrive(A_EMPTY(prev_flags & 0xff));
                    }
                    prev_flags = flags;
                }
                wgmma_wait<0>();
                if (prev_flags >= 0 && lane == 0) {
                    mbar_arrive(G_EMPTY((j - 1) % NG));
                    if (prev_flags & (1 << 17)) mbar_arrive(A_EMPTY(prev_flags & 0xff));
                }
                prev_flags = -1;

                // epilogue of row ho
#pragma unroll
                for (int qq = 0; qq < 4; qq++) {
                    const int q = 4 * cw + qq;
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const int rr = 16 * warp + (lane >> 2) + 8 * h;
                        int cl, tl;
                        if (p.mode == 0) {
                            const int gt = Tc >> 3;
                            const int cg = q / gt, tg = q - cg * gt;
                            cl = cg * 8 + (rr >> 3);
                            tl = tg * 8 + (rr & 7);
                        } else {
                            const int gt = Tc >> 6;
                            const int c = q / gt, tg = q - c * gt;
                            cl = c;
                            tl = tg * 64 + rr;
                        }
                        const int ch = c0 + cl, t = t0 + tl;
                        if (ch >= p.BC || t >= p.T) continue;
                        __half *ob = p.out + (size_t)ch * Kk * HW + (size_t)ho * p.Wo + (size_t)t * S;
#pragma unroll
                        for (int gg = 0; gg < N / 8; gg++) {
                            const int n0 = 8 * gg + 2 * (lane & 3);
                            const float v0 = acc[qq][4 * gg + 2 * h], v1 = acc[qq][4 * gg + 2 * h + 1];
                            if constexpr (S >= 2) {
                                const int k = n0 / S, sh = n0 - k * S;
                                if (k < Kk)
                                    *reinterpret_cast<__half2 *>(ob + (size_t)k * HW + sh) = __floats2half2_rn(v0, v1);
                            } else {
                                if (n0 < Kk) ob[(size_t)n0 * HW] = __float2half_rn(v0);
                                if (n0 + 1 < Kk) ob[(size_t)(n0 + 1) * HW] = __float2half_rn(v1);
                            }
                        }
                    }
                }
            }
        }
#undef A_FULL
#undef A_EMPTY
#undef V_LOAD
#undef V_FREE
#undef G_FULL
#undef G_EMPTY
    }

    // ------------------------------------------------------------------------------------------------
    // generic path (any pscale / K): one thread per output longitude, fp32 accumulation
    // ------------------------------------------------------------------------------------------------
    __global__ void disco_generic_kernel(const __half *__restrict__ inp, const int64_t *__restrict__ pidx,
                                         const __half *__restrict__ pval, const int64_t *__restrict__ poff,
                                         __half *__restrict__ out, int BC, int Hi, int Wi, int Wo, int Ho, int K,
                                         int KP, int ps)
    {
        const int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        const int64_t total = (int64_t)BC * Ho * Wo;
        if (gid >= total) return;
        const int wo = (int)(gid % Wo);
        const int ho = (int)((gid / Wo) % Ho);
        const int bc = (int)(gid / ((int64_t)Wo * Ho));
        float acc[16];
#pragma unroll
        for (int k = 0; k < 16; k++) acc[k] = 0.f;
        const __half *src = inp + (size_t)bc * Hi * Wi;
        for (int64_t n = poff[ho]; n < poff[ho + 1]; n++) {
            const int hi = (int)pidx[2 * n], wi = (int)pidx[2 * n + 1];
            const float x = __half2float(src[(size_t)hi * Wi + (wi + (int64_t)wo * ps) % Wi]);
            const __half *v = pval + (size_t)n * KP;
#pragma unroll
            for (int k = 0; k < 16; k++)
                if (k < K) acc[k] += __half2float(v[k]) * x;
        }
        const size_t HW = (size_t)Ho * Wo;
#pragma unroll
        for (int k = 0; k < 16; k++)
            if (k < K) out[((size_t)bc * K + k) * HW + (size_t)ho * Wo + wo] = __float2half_rn(acc[k]);
    }

    // ------------------------------------------------------------------------------------------------
    // host
    // ------------------------------------------------------------------------------------------------
    static ffi::Tensor alloc_bytes(int64_t nbytes, DLDevice dev)
    {
        if (nbytes < 16) nbytes = 16;
        return ffi::Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, ffi::Shape({nbytes}), DLDataType {kDLInt, 8, 1}, dev);
    }

    template <int N, int S, int NCW> static void launch_main(MainParams mp, int nCB, cudaStream_t stream)
    {
        constexpr int GST = N * 16 + 16;
        const int GSB = NBMAX * GST + INFOB;
        auto smem_of = [&](int na, int ng, int nv) {
            return na * mp.ABYTES + ng * GSB + nv * mp.VBYTES + GST + 64 * RCAP + 16 * RMAX + 32
                + 8 * (2 * na + 2 * nv + 2 * ng);
        };
        // largest rings that fit (two CTAs per SM for one consumer warpgroup, one CTA otherwise)
        const int budget = NCW == 1 ? 228 * 1024 / 2 - 1024 : 227 * 1024;
        const int cand[9][3]
            = {{4, 3, 8}, {3, 3, 8}, {3, 3, 6}, {2, 3, 8}, {2, 3, 6}, {2, 3, 4}, {2, 2, 4}, {2, 2, 3}, {2, 2, 2}};
        int pick = 8;
        for (int i = 0; i < 9; i++)
            if (smem_of(cand[i][0], cand[i][1], cand[i][2]) <= budget) {
                pick = i;
                break;
            }
        mp.na = cand[pick][0];
        mp.ng = cand[pick][1];
        mp.nv = cand[pick][2];
        const int smem = smem_of(mp.na, mp.ng, mp.nv);
        static bool configured = false;
        if (!configured) {
            cudaFuncSetAttribute(disco_main_kernel<N, S, NCW>, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
            configured = true;
        }
        // rows per CTA: amortize the pipeline fill while keeping >= ~6 CTAs per SM slot
        const int tiles = nCB * mp.nTB;
        const int slots = (NCW == 1 ? 2 : 1) * 132;
        int R = 1;
        while (R < RMAX && (int64_t)tiles * ((mp.Ho + 2 * R - 1) / (2 * R)) >= 6 * slots) R *= 2;
        mp.R = R;
        dim3 grid(tiles, (mp.Ho + R - 1) / R);
        disco_main_kernel<N, S, NCW><<<grid, 128 * (NCW + 1), smem, stream>>>(mp);
    }

    static inline bool supported(int Wi, int Wo, int K)
    {
        if (Wo <= 0 || Wi % Wo != 0 || (Wo % 8) != 0 || K < 1 || K > 16) return false;
        const int ps = Wi / Wo;
        const int S = ps == 1 ? 8 : ps == 2 ? 4 : ps == 4 ? 2 : ps == 8 ? 1 : 0;
        return S > 0 && fast_pair_supported(((K * S + 7) / 8) * 8, S);
    }

    void disco_run(ffi::TensorView inp, ffi::TensorView pack_idx, ffi::TensorView pack_val, ffi::TensorView pack_offset,
                   int64_t kernel_size, int64_t nlon_out, ffi::TensorView out)
    {
        const int B = (int)inp.size(0), C = (int)inp.size(1), Hi = (int)inp.size(2), Wi = (int)inp.size(3);
        const int BC = B * C;
        const int K = (int)kernel_size, Wo = (int)nlon_out;
        const int Ho = (int)pack_offset.size(0) - 1;
        const int64_t NNZ = pack_idx.size(0);
        const int KP = (int)pack_val.size(1);
        const int ps = Wi / Wo;
        DLDevice dev = inp.device();
        cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));

        const __half *d_inp = static_cast<const __half *>(inp.data_ptr());
        const int64_t *d_idx = static_cast<const int64_t *>(pack_idx.data_ptr());
        const __half *d_val = static_cast<const __half *>(pack_val.data_ptr());
        const int64_t *d_off = static_cast<const int64_t *>(pack_offset.data_ptr());
        __half *d_out = static_cast<__half *>(out.data_ptr());
        if (Ho <= 0 || BC <= 0 || Wo <= 0) return;

        int S = 0;
        if (ps == 1)
            S = 8;
        else if (ps == 2)
            S = 4;
        else if (ps == 4)
            S = 2;
        else if (ps == 8)
            S = 1;
        const int N = ((K * S + 7) / 8) * 8;
        const int64_t TOTPOS = ((NNZ * PPP + 7) & ~7LL) + 16LL * Ho + 16;
        bool fast = S > 0 && K >= 1 && K <= 16 && (Wo % 8) == 0 && Wi == ps * Wo && TOTPOS < (1LL << 31)
            && fast_pair_supported(N, S);

        if (!fast) {
            const int64_t total = (int64_t)BC * Ho * Wo;
            const int threads = 256;
            const int64_t blocks = (total + threads - 1) / threads;
            disco_generic_kernel<<<(unsigned)blocks, threads, 0, stream>>>(d_inp, d_idx, d_val, d_off, d_out, BC, Hi,
                                                                           Wi, Wo, Ho, K, KP, ps);
            return;
        }

        // tile selection: padded GEMM rows plus input-window staging overhead; 4 m64 tiles per consumer
        // warpgroup, 1 (256 rows, two CTAs per SM) or 2 (512 rows, one CTA per SM) consumer warpgroups
        const int T = Wo / S;
        struct Cfg {
            int ncw, mode, Cc, Tc;
        };
        const Cfg cfgs[12]
            = {{1, 0, 32, 8},  {1, 0, 16, 16}, {1, 0, 8, 32}, {1, 1, 4, 64}, {1, 1, 2, 128}, {1, 1, 1, 256},
               {2, 0, 32, 16}, {2, 0, 16, 32}, {2, 0, 8, 64}, {2, 1, 8, 64}, {2, 1, 4, 128}, {2, 1, 2, 256}};
        int best = 0;
        double bestCost = -1.0;
        for (int i = 0; i < 12; i++) {
            if (cfgs[i].ncw > NCW_MAX) continue;
            const int64_t rows = (int64_t)((BC + cfgs[i].Cc - 1) / cfgs[i].Cc) * cfgs[i].Cc
                * ((T + cfgs[i].Tc - 1) / cfgs[i].Tc) * cfgs[i].Tc;
            const double cost = (double)rows * (1.0 + 0.25 * 2.0 / cfgs[i].Tc) * (cfgs[i].ncw == 2 ? NCW2_DISCOUNT : 1.0);
            if (bestCost < 0 || cost < bestCost) {
                best = i;
                bestCost = cost;
            }
        }
        const int Tc = cfgs[best].Tc;
        const int ncw = cfgs[best].ncw;
        const int chunkmax = NPMAX * (Tc - 1) + 2 * NBMAX;

        const int64_t nnz1 = NNZ > 0 ? NNZ : 1;
        ffi::Tensor t_seg = alloc_bytes(nnz1 * (4 + 4 + 4 + 16), dev);
        ffi::Tensor t_vd = alloc_bytes((int64_t)K * TOTPOS * 2, dev);
        ffi::Tensor t_st = alloc_bytes((TOTPOS / 8) * 64 + 64, dev);
        ffi::Tensor t_ns = alloc_bytes((int64_t)Ho * 16, dev);
        int4 *seginfo = static_cast<int4 *>(t_seg.data_ptr()); // first: 16-byte aligned
        int *gsid = reinterpret_cast<int *>(seginfo + nnz1);
        int *seghead = gsid + nnz1;
        int *segpos = seghead + nnz1;
        __half *Vd = static_cast<__half *>(t_vd.data_ptr());
        int *stages = static_cast<int *>(t_st.data_ptr());
        int *nst = static_cast<int *>(t_ns.data_ptr());

        disco_prep_kernel<<<Ho, PREP_THREADS, 0, stream>>>(d_idx, d_off, K, Wi, ps, Tc, chunkmax, TOTPOS, gsid, seghead,
                                                           segpos, seginfo, Vd, stages, nst);
        if (NNZ > 0)
            disco_scatter_kernel<<<(unsigned)((NNZ + 255) / 256), 256, 0, stream>>>(d_idx, d_val, gsid, segpos, NNZ, K,
                                                                                    KP, TOTPOS, Vd);

        MainParams mp;
        mp.inp = d_inp;
        mp.Vd = Vd;
        mp.stages = stages;
        mp.nstages = nst;
        mp.poff = d_off;
        mp.out = d_out;
        mp.TOTPOS = TOTPOS;
        mp.BC = BC;
        mp.Hi = Hi;
        mp.Wi = Wi;
        mp.Wo = Wo;
        mp.Ho = Ho;
        mp.K = K;
        mp.ps = ps;
        mp.T = T;
        mp.mode = cfgs[best].mode;
        mp.Cc = cfgs[best].Cc;
        mp.Tc = Tc;
        mp.nTB = (T + mp.Tc - 1) / mp.Tc;
        mp.CW = 16 * chunkmax;
        mp.ABYTES = mp.Cc * mp.CW;
        mp.VBYTES = K * (NBMAX + 1) * 16;
        const int nCB = (BC + mp.Cc - 1) / mp.Cc;
        mp.nCB = nCB;
        mp.nItems = nCB * mp.nTB * Ho;

        if (ncw == 2)
            DISPATCH_MAIN2(N, S, mp, nCB, stream);
        else
            DISPATCH_MAIN(N, S, mp, nCB, stream);
    }

} // namespace leader_small

#undef GMAX
#undef PPP
#undef NPMAX
#undef NBMAX
#undef INFOB
#undef RMAX
#undef RCAP
#undef NCW_MAX
#undef NCW2_DISCOUNT
#undef PREP_THREADS
#undef PREP_WARPS
#undef DISPATCH_MAIN
#undef DISPATCH_MAIN2
