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

// =====================================================================================================================
// The parts of the DISCO Hankel/Toeplitz tensor-core forward that sm_100a (disco_cuda_fwd_hankel_sm100.cu) and sm_90a
// (disco_cuda_fwd_hankel_sm90.cu) share: the pieces and their planners, the Toeplitz builder, the longitude-extended
// input copy and its co-resident protocol, the mbarrier / bulk-copy / TMA helpers, and the main kernels' parameters.
// Each arch file includes it once and adds its MMA and accumulator parts (tcgen05 / wgmma); see there for the
// formulation.
// =====================================================================================================================

#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <vector>

#include "../../disco.h"
#include "../../disco_checks.h"

#define KB 9 // basis functions handled by the tensor-core path

// A piece is chunks [jb / 8, jb / 8 + nc) of the window of one run (RG = 1: an arc of n points,
// points p1.. (n1 of them) then p2..; RG = 2: a union window of n longitudes, rows in PieceX).
// Long windows are cut at chunk boundaries, so the window halo is paid once per run.
// S = window start + jb (mod Wi): S >> 3 is the piece's input block, S & 7 the run's offset.
struct __align__(16) Piece {
    int hi, S, n, p1, n1, p2, toff, nc; // nc: 8-longitude window chunks of this piece
    int jb, pad0, pad1, pad2;           // jb: 8 * first chunk of the piece within the window
};

// Programmatic dependent launch: a kernel launched with programmatic stream serialization may
// start before its predecessor finishes; pdl_wait() blocks until the predecessor grid has
// completed and its memory is visible, pdl_trigger() lets the successor grid launch early.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }

// Completion stamps (co-resident input extension, see xcopy_kernel).  A count is accumulated with
// atomicAdd; the adder that completes it resets the count to zero and release-stores the call's
// epoch (calls on a workspace are numbered from 1) into the stamp, so stamps left by earlier calls
// never match and nothing needs a per-call reset.
__device__ __forceinline__ void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;\n" ::: "memory"); }
__device__ __forceinline__ void fence_acq_rel_gpu() { asm volatile("fence.acq_rel.gpu;\n" ::: "memory"); }
// two fp32 accumulators rounded to the activations' dtype, packed low to high (torch-harmonics)
__device__ __forceinline__ unsigned pack2(float a, float b, int bf16)
{
    if (bf16) {
        const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
        return *reinterpret_cast<const unsigned *>(&h);
    }
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const unsigned *>(&h);
}
__device__ __forceinline__ unsigned ld_relaxed_u32(const unsigned *p)
{
    unsigned v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ unsigned ld_acquire_u32(const unsigned *p)
{
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_release_u32(unsigned *p, unsigned v)
{
    asm volatile("st.release.gpu.global.u32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}
// thread 0, after a release fence: add n; the completing adder publishes the epoch
__device__ __forceinline__ bool count_add(unsigned *cnt, unsigned *stamp, unsigned n, unsigned target, unsigned epoch)
{
    const unsigned old = atomicAdd(cnt, n);
    if (old + n != target) return false;
    __threadfence(); // acquire every earlier adder's release (they form one RMW chain)
    *reinterpret_cast<volatile unsigned *>(cnt) = 0u;
    st_release_u32(stamp, epoch);
    return true;
}
// every thread of the block publishes its prior global writes (generic and async proxies), then
// thread 0 counts n towards target
__device__ __forceinline__ void block_count(unsigned *cnt, unsigned *stamp, unsigned n, unsigned target, unsigned epoch)
{
    fence_proxy_async_global();
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence(); // release: orders the block's writes (via the barrier) before the count
        count_add(cnt, stamp, n, target, epoch);
    }
}
#define ST_TCNT 0 // Toeplitz builder blocks done (count / stamp)
#define ST_TSTAMP 1
#define ST_XCNT 2 // input-extension tasks done
#define ST_XSTAMP 3
#define ST_XCLAIM 4 // extension task claims (zeroed by the call's first kernel)
#define ST_FCNT 5   // flag blocks done: the extension ranges below are final
#define ST_FSTAMP 6
#define ST_ICTR 8  // the main kernel's dynamic item counter (torch-harmonics: was the planner's counter[1])
#define ST_LCNT 16 // + hi: channel rows of latitude hi extended
#define CO_MAXHI 8000
#define ST_LSTAMP (ST_LCNT + CO_MAXHI) // + hi
// + t: claim word of reserved extension task t (holds the epoch of the call that claimed it)
#define ST_RCLAIM (ST_LSTAMP + CO_MAXHI)
#define XC_MAXRES 256
#define STAMP_WORDS_BYTES 65536
// After the stamp words: per latitude the range of extended-copy chunks this call's windows can
// read, as two 64-bit words {epoch << 32 | ~lo, epoch << 32 | hx} accumulated with atomicMax, so a
// newer call's first update always replaces an older call's value (no per-call reset).
#define STAMP_BYTES (STAMP_WORDS_BYTES + 16 * CO_MAXHI)

// Extended copy layout: xe[c, hi, q] = x[c, hi, chunk (q + hrot) mod nb] with hrot = nb / 2, i.e.
// the ring is rotated by half a turn so that the windows around the seam (where every row's support
// is centred) start mid-row; a window starting at input chunk B8 starts at q = (B8 - hrot) mod nb.
// Only the positions some window of the latitude can read are written.
__device__ __forceinline__ bool lat_range(const unsigned long long *R, int h, unsigned epoch, int Lc, int &lo, int &hx)
{
    lo = 0;
    hx = Lc;
    if (!R) return true;
    const unsigned long long a = __ldcg(R + 2 * h), b = __ldcg(R + 2 * h + 1);
    if ((unsigned)(a >> 32) != epoch) return false; // no point of this call on latitude h
    lo = (int)~(unsigned)a;
    hx = min((int)(unsigned)b, Lc);
    return true;
}
// copies positions [lo, hx) of one extended row (NT cooperating threads, thread index t): every
// input chunk is loaded once (all of a thread's loads before its stores) and stored at each of its
// positions in the range
template <int NT>
__device__ __forceinline__ void copy_range(const uint4 *__restrict__ src, uint4 *__restrict__ dst, int lo, int hx,
                                           int nb, int hrot, int t)
{
    for (int j0 = t; j0 < nb; j0 += 4 * NT) {
        uint4 v[4];
#pragma unroll
        for (int u = 0; u < 4; ++u)
            if (j0 + u * NT < nb) v[u] = __ldg(src + j0 + u * NT);
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int j = j0 + u * NT;
            if (j < nb) {
                int q = j - hrot;
                if (q < 0) q += nb;
                if (q < lo) q += nb;
                for (; q < hx; q += nb) dst[q] = v[u];
            }
        }
    }
}

__device__ __forceinline__ int ld_hi(const int64_t *idx, int64_t p) { return (int)idx[2 * p]; }
__device__ __forceinline__ int ld_wi(const int64_t *idx, int64_t p) { return (int)idx[2 * p + 1]; }

// exclusive block scan (256 threads); returns exclusive prefix, *total = sum
__device__ __forceinline__ int block_exscan256(int v, int *sh, int *total)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int incl = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += u;
    }
    if (lane == 31) sh[warp] = incl;
    __syncthreads();
    if (warp == 0) {
        int w = lane < 8 ? sh[lane] : 0;
        int wi = w;
#pragma unroll
        for (int o = 1; o < 8; o <<= 1) {
            const int u = __shfl_up_sync(0xffffffffu, wi, o);
            if (lane >= o) wi += u;
        }
        if (lane < 8) sh[8 + lane] = wi - w;
        if (lane == 7) sh[16] = wi;
    }
    __syncthreads();
    const int r = sh[8 + warp] + incl - v;
    *total = sh[16];
    __syncthreads();
    return r;
}

// flags[p]: bit0 = starts a new run (different hi or non-consecutive wi), bit1 = different hi
// than point p-1 (row starts are handled by the row kernel).
__global__ void __launch_bounds__(256) flag_kernel(const int64_t *__restrict__ idx, int64_t NNZ, int Wi,
                                                   unsigned char *__restrict__ flags, unsigned *__restrict__ counter)
{
    pdl_trigger();
    const int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= NNZ) return;
    if (p == 0) {
        counter[0] = 0;
        counter[2] = 0;
    }
    const longlong2 cur = reinterpret_cast<const longlong2 *>(idx)[p];
    const unsigned amask = __activemask();
    longlong2 prv;
    prv.x = __shfl_up_sync(amask, cur.x, 1);
    prv.y = __shfl_up_sync(amask, cur.y, 1);
    if ((threadIdx.x & 31) == 0 && p > 0) prv = reinterpret_cast<const longlong2 *>(idx)[p - 1];
    unsigned char f = 3;
    if (p > 0) {
        long long nx = prv.y + 1;
        if (nx == Wi) nx = 0;
        const bool dh = cur.x != prv.x;
        f = (unsigned char)((dh || cur.y != nx) ? 1 : 0) | (unsigned char)(dh ? 2 : 0);
    }
    flags[p] = f;
}

// Every latitude's extended-copy range (see lat_range): every piece window of the latitude starts
// at a chunk B8 in [w - nc_cap - 1, w + 2] for some point chunk w of it (a piece starts at or one
// chunk after a chunk holding one of its points, the half-block shift included; a trailing
// halo-only piece at most two chunks after the run's last point), and a window starting at B8
// reads RPC chunks.  The call's first kernel: it triggers its programmatic dependents at once, so
// the flag pass and the planner overlap it; only the extension copies wait for its stamp.
__global__ void __launch_bounds__(256)
    range_kernel(const int64_t *__restrict__ idx, int64_t NNZ, unsigned long long *__restrict__ R, int nb, int hrot,
                 int nc_cap, int RPC, unsigned epoch)
{
    pdl_trigger();
    const int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int h = -1;
    unsigned lo = 0xFFFFFFFFu, hx = 0u;
    if (p < NNZ) {
        const longlong2 cur = reinterpret_cast<const longlong2 *>(idx)[p];
        h = (int)cur.x;
        // Every chunk in a piece's point run contains a point, and a piece starts no earlier than
        // one chunk before such a point.  Copy only the starts that any planned window can reach.
        int a0 = (int)(cur.y >> 3) - 1 - hrot;
        while (a0 < 0) a0 += nb;
        if (a0 + 3 < nb) {
            lo = (unsigned)a0;
            hx = (unsigned)(a0 + 3 + RPC);
        } else {
            lo = 0u;
            hx = (unsigned)(nb - 1 + RPC);
        }
    }
    // one update per latitude and warp
    const unsigned m = __match_any_sync(0xffffffffu, h);
    const unsigned glo = __reduce_min_sync(m, lo);
    const unsigned ghx = __reduce_max_sync(m, hx);
    if (h >= 0 && (threadIdx.x & 31) == __ffs(m) - 1) {
        const unsigned long long tag = (unsigned long long)epoch << 32;
        atomicMax(R + 2 * h, tag | (unsigned long long)(~glo));
        atomicMax(R + 2 * h + 1, tag | (unsigned long long)ghx);
    }
}

// window chunks (8 longitudes each) of a piece with l points starting at longitude sq
__device__ __forceinline__ int piece_nc(int l, int sq, int ps) { return ((sq & 7) + l + 8 - ps + 7) >> 3; }
// Toeplitz tile k-steps stored per piece: chunks nc + 1 rounded up to even (chunk nc is zero, so the
// last chunk can always be paired with a zero chunk)
__device__ __forceinline__ int piece_tks(int nc) { return (nc + 2) >> 1; }

// ------------------------------------------------------------------------------------------
// Row groups (RG = 2): two neighbouring output rows share every input-latitude window.  A
// row's run at latitude hi (an arc of longitudes, merged across the seam) is united with the
// other row's overlapping arc at the same latitude; the union is cut into windows and one
// Toeplitz tile carries both rows' filters side by side along N (columns r * MN1 + k * DW + dwo),
// so the input window is loaded and multiplied once for both rows.  Arcs that find no partner
// form windows of their own (their other row's columns are zero), so any psi is handled.
// ------------------------------------------------------------------------------------------
struct __align__(16) Arc {
    int hi, S, n, p1, n1, p2, row, pad;
};
// per-row arcs of a group piece (n[r] == 0: row r has no points in this piece)
struct PieceX {
    int S[2], n[2], p1[2], n1[2], p2[2];
};

template <int NT> __device__ __forceinline__ int block_exscan_t(int v, int *sh, int *total)
{
    constexpr int NW = NT / 32;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int incl = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += u;
    }
    if (lane == 31) sh[warp] = incl;
    __syncthreads();
    if (warp == 0) {
        const int w = lane < NW ? sh[lane] : 0;
        int wi = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int u = __shfl_up_sync(0xffffffffu, wi, o);
            if (lane >= o) wi += u;
        }
        if (lane < NW) sh[32 + lane] = wi - w;
        if (lane == NW - 1) sh[64] = wi;
    }
    __syncthreads();
    const int r = sh[32 + warp] + incl - v;
    *total = sh[64];
    __syncthreads();
    return r;
}

// circular union of arcs [Sa, Sa + na) and [Sb, Sb + nb) when they overlap or touch
__device__ __forceinline__ bool arc_union(int Sa, int na, int Sb, int nb, int Wi, int &U, int &L)
{
    int d = Sb - Sa;
    if (d < 0) d += Wi;
    if (d <= na) {
        U = Sa;
        L = max(na, d + nb);
    } else {
        int d2 = Sa - Sb;
        if (d2 < 0) d2 += Wi;
        if (d2 > nb) return false;
        U = Sb;
        L = max(nb, d2 + na);
    }
    if (L > Wi) L = Wi;
    return true;
}

#define GT 512    // group planner threads
#define GMAX 1024 // arcs matched per group (larger groups keep their arcs unpaired)
__global__ void __launch_bounds__(GT)
    group_kernel(const int64_t *__restrict__ idx, const int64_t *__restrict__ off,
                 const unsigned char *__restrict__ flags, int Wi, int ps, int nc_cap, int nunit, int Ho,
                 int *__restrict__ substart, Arc *__restrict__ arcs, Piece *__restrict__ pieces,
                 PieceX *__restrict__ pxs, int *__restrict__ npieces, unsigned *__restrict__ counter, long long cap,
                 int *__restrict__ flat, int *__restrict__ rowsig)
{
    __shared__ int sh[72];
    __shared__ int csum[2];
    __shared__ int mj[GMAX]; // row-1 arc -> smallest row-0 arc that chose it
    __shared__ int pr[GMAX]; // row-0 arc -> its row-1 partner
    __shared__ long long sbase;
    __shared__ int fbase;
    pdl_trigger(); // the input-extension copy of the next kernel does not depend on this one
    pdl_wait();    // flags / counters
    const int g = blockIdx.x;
    const int tid = threadIdx.x;
    const int r0 = 2 * g;
    const int nr = min(2, Ho - r0);
    const int64_t a = off[r0];
    const int m = (int)(off[r0 + 1] - a); // first point of row 1 (== len when nr == 1)
    const int len = (int)(off[r0 + nr] - a);
    int *ss = substart + a;
    const unsigned char *fl = flags + a;
    Arc *ar = arcs + a;
    // pass 1: run starts (row starts, new latitude or non-consecutive longitude)
    const int IT = (len + GT - 1) / GT;
    const int q0 = min(tid * IT, len), q1 = min(q0 + IT, len);
    int c = 0;
    for (int p0 = q0; p0 < q1; p0 += 16) {
        unsigned char f[16];
#pragma unroll
        for (int u = 0; u < 16; ++u) f[u] = (p0 + u < q1) ? fl[p0 + u] : (unsigned char)0;
#pragma unroll
        for (int u = 0; u < 16; ++u) {
            const int p = p0 + u;
            if (p < q1 && (p == 0 || p == m || (f[u] & 1))) ++c;
        }
    }
    int cnt;
    int w = block_exscan_t<GT>(c, sh, &cnt);
    for (int p = q0; p < q1; ++p)
        if (p == 0 || p == m || (fl[p] & 1)) ss[w++] = p;
    __syncthreads();
    // pass 2: arcs (first and last run of a row's latitude merged when they meet at the seam)
    auto arc_of = [&](int s, Arc &o) -> bool {
        const int st = ss[s];
        const int en = (s + 1 < cnt) ? ss[s + 1] : len;
        const int row = (st >= m) ? 1 : 0;
        const int rs = row ? m : 0, re = row ? len : m;
        const bool isFirst = (st == rs) || (fl[st] & 2);
        const bool isLast = (en == re) || (fl[en] & 2);
        const int S = ld_wi(idx, a + st);
        o.hi = ld_hi(idx, a + st);
        o.S = S;
        o.p1 = (int)(a + st);
        o.n1 = en - st;
        o.p2 = o.p1;
        o.n = o.n1;
        o.row = row;
        o.pad = 0;
        if (isFirst && !isLast) {
            int t = s + 1;
            int ten = (t + 1 < cnt) ? ss[t + 1] : len;
            while (!((ten == re) || (fl[ten] & 2))) {
                ++t;
                ten = (t + 1 < cnt) ? ss[t + 1] : len;
            }
            int nx = ld_wi(idx, a + ten - 1) + 1;
            if (nx == Wi) nx = 0;
            if (nx == S && o.n1 + (ten - ss[t]) <= Wi)
                return false; // merged into the last run only if every longitude remains unique
        } else if (!isFirst && isLast) {
            int t = s - 1;
            int tst = ss[t];
            while (!((tst == rs) || (fl[tst] & 2))) {
                --t;
                tst = ss[t];
            }
            const int ten = ss[t + 1];
            int nx = ld_wi(idx, a + en - 1) + 1;
            if (nx == Wi) nx = 0;
            if (nx == ld_wi(idx, a + tst) && o.n1 + (ten - tst) <= Wi) {
                o.p2 = (int)(a + tst);
                o.n = o.n1 + (ten - tst);
            }
        }
        return true;
    };
    const int IS = (cnt + GT - 1) / GT;
    const int s0 = min(tid * IS, cnt), s1 = min(s0 + IS, cnt);
    int kept = 0;
    for (int s = s0; s < s1; ++s) {
        Arc o;
        if (arc_of(s, o)) ++kept;
    }
    int nar;
    int kp = block_exscan_t<GT>(kept, sh, &nar);
    for (int s = s0; s < s1; ++s) {
        Arc o;
        if (arc_of(s, o)) ar[kp++] = o;
    }
    __syncthreads();
    // pass 3: pair each row-0 arc with an overlapping row-1 arc of the same latitude
    const bool domatch = nr == 2 && nar <= GMAX;
    if (domatch) {
        for (int i = tid; i < nar; i += GT) {
            mj[i] = 0x7fffffff;
            pr[i] = -1;
        }
        __syncthreads();
        for (int i = tid; i < nar; i += GT) {
            const Arc ai = ar[i];
            if (ai.row != 0) continue;
            for (int j = 0; j < nar; ++j) {
                const Arc aj = ar[j];
                int U, L;
                if (aj.row == 1 && aj.hi == ai.hi && arc_union(ai.S, ai.n, aj.S, aj.n, Wi, U, L)) {
                    atomicMin(&mj[j], i);
                    break;
                }
            }
        }
        __syncthreads();
        for (int j = tid; j < nar; j += GT)
            if (mj[j] != 0x7fffffff) pr[mj[j]] = j;
        __syncthreads();
    }
    // pass 4: one window run per row-0 arc (with its partner) and per unpaired row-1 arc, cut
    // into pieces of at most pmax longitudes
    const int IU = (nar + GT - 1) / GT;
    const int u0 = min(tid * IU, nar), u1 = min(u0 + IU, nar);
    int npl = 0, nwl = 0;
    // half-block output shift of the row pair (see row_kernel), chosen over the union windows
    const bool shiftable = ps < 8;
    int lc0 = 0, lc1 = 0;
    int sig = 0;
    if (tid < 2) csum[tid] = 0;
    for (int pass = -1; pass < 2; ++pass) {
        int wpos = 0;
        long long wword = 0;
        if (pass == 0) {
            __syncthreads();
            if (lc0) atomicAdd(&csum[0], lc0);
            if (lc1) atomicAdd(&csum[1], lc1);
            __syncthreads();
            // At pscale 1, a shifted row pair pays split, half-sector output stores.  Require a material
            // chunk reduction before accepting that cost; single-row and pscale-2 policies stay unchanged.
            if (ps == 1)
                sig = (shiftable && csum[1] + 4 < csum[0]) ? 4 : 0;
            else
                sig = (shiftable && csum[1] < csum[0]) ? 4 : 0;
            if (tid == 0) rowsig[g] = sig;
        }
        if (pass == 1) {
            int tot, totw;
            wpos = block_exscan_t<GT>(npl, sh, &tot);
            const int woff = block_exscan_t<GT>(nwl, sh, &totw);
            if (tid == 0) {
                npieces[g] = tot;
                fbase = (int)atomicAdd(counter + 2, (unsigned)tot);
                long long b = -1;
                if (totw > 0) {
                    b = (long long)atomicAdd(counter, (unsigned)totw);
                    if (b + totw > cap) b = -1;
                }
                sbase = b;
            }
            __syncthreads();
            wword = sbase < 0 ? -1 : sbase + woff;
        }
        for (int u = u0; u < u1; ++u) {
            const Arc au = ar[u];
            int partner = -1;
            if (domatch) {
                if (au.row == 1 && mj[u] != 0x7fffffff) continue; // emitted with its row-0 partner
                if (au.row == 0) partner = pr[u];
            }
            int U = au.S, L = au.n;
            Arc ap;
            if (partner >= 0) {
                ap = ar[partner];
                arc_union(au.S, au.n, ap.S, ap.n, Wi, U, L);
            }
            if (pass < 0) {
                lc0 += piece_nc(L, U, ps);
                lc1 += piece_nc(L, U + 4, ps);
                continue;
            }
            int np = 0;
            const int NCT = piece_nc(L, U + sig, ps); // chunks of the whole (shifted) window
            for (int h0 = 0; h0 < NCT; h0 += nc_cap, ++np) {
                const int nc = min(nc_cap, NCT - h0);
                const int nw = ((nc + 1) * nunit) >> 1;
                if (pass == 0) {
                    nwl += nw;
                    continue;
                }
                int sq = U + sig + 8 * h0;
                while (sq >= Wi) sq -= Wi;
                Piece pc;
                pc.hi = au.hi;
                pc.S = sq;
                pc.n = L;
                pc.p1 = pc.n1 = pc.p2 = 0;
                pc.toff = wword < 0 ? -1 : (int)wword;
                pc.nc = nc;
                pc.jb = 8 * h0;
                pc.pad0 = pc.pad1 = pc.pad2 = 0;
                if (wword >= 0) wword += nw;
                PieceX px = {};
                // row arcs in the shifted frame (window and arcs move together by sig)
                px.S[au.row] = au.S + sig >= Wi ? au.S + sig - Wi : au.S + sig;
                px.n[au.row] = au.n;
                px.p1[au.row] = au.p1;
                px.n1[au.row] = au.n1;
                px.p2[au.row] = au.p2;
                if (partner >= 0) {
                    px.S[1] = ap.S + sig >= Wi ? ap.S + sig - Wi : ap.S + sig;
                    px.n[1] = ap.n;
                    px.p1[1] = ap.p1;
                    px.n1[1] = ap.n1;
                    px.p2[1] = ap.p2;
                }
                pieces[a + wpos + np] = pc;
                pxs[a + wpos + np] = px;
                flat[fbase + wpos + np] = (int)(a + wpos + np);
            }
            if (pass == 0)
                npl += np;
            else
                wpos += np;
        }
    }
}

// Per output row: split the support into pieces (runs merged across the seam, cut at pmax
// points) and allocate each piece's Toeplitz tiles (ks * 2N 16-byte units) in scratch.
__global__ void __launch_bounds__(256)
    row_kernel(const int64_t *__restrict__ idx, const int64_t *__restrict__ off,
               const unsigned char *__restrict__ flags, int Wi, int ps, int nc_cap, int nunit,
               int *__restrict__ substart, Piece *__restrict__ pieces, int *__restrict__ npieces,
               unsigned *__restrict__ counter, long long cap, int *__restrict__ flat, int *__restrict__ rowsig)
{
    __shared__ int sh[32];
    __shared__ long long sbase;
    __shared__ int fbase;
    __shared__ int csum[2];
    pdl_trigger(); // the input-extension copy of the next kernel does not depend on this one
    pdl_wait();    // flags / counters
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t a = off[row];
    const int len = (int)(off[row + 1] - a);
    int *ss = substart + a;
    const unsigned char *fl = flags + a;
    // pass 1: run starts
    const int IT = (len + 255) >> 8;
    const int q0 = min(tid * IT, len), q1 = min(q0 + IT, len);
    int c = 0;
    for (int p = q0; p < q1; ++p) c += (p == 0 || (fl[p] & 1)) ? 1 : 0;
    int cnt;
    int w = block_exscan256(c, sh, &cnt);
    for (int p = q0; p < q1; ++p)
        if (p == 0 || (fl[p] & 1)) ss[w++] = p;
    __syncthreads();
    // pass 2: merge the first and last run of a latitude when they connect across the seam,
    // then cut into pieces of at most pmax points.
    const int IS = (cnt + 255) >> 8;
    const int s0 = min(tid * IS, cnt), s1 = min(s0 + IS, cnt);
    int npl = 0, nwl = 0;
    // Output blocks of the row may start half a block late (sig = 4 input longitudes, i.e. GEMM row
    // b holds outputs DW b + DW / 2 + dwo): pass -1 keeps the alignment with fewer window chunks
    // (each arc pays ceil(((S + sig) & 7) + n + 8 - ps) / 8 chunks).
    const bool shiftable = ps < 8;
    int lc0 = 0, lc1 = 0;
    int sig = 0;
    if (tid < 2) csum[tid] = 0;
    for (int pass = -1; pass < 2; ++pass) {
        int wpos = 0;
        long long wword = 0;
        if (pass == 0) {
            __syncthreads();
            if (lc0) atomicAdd(&csum[0], lc0);
            if (lc1) atomicAdd(&csum[1], lc1);
            __syncthreads();
            sig = (shiftable && csum[1] < csum[0]) ? 4 : 0;
            if (tid == 0) rowsig[row] = sig;
        }
        if (pass == 1) {
            int tot, totw;
            wpos = block_exscan256(npl, sh, &tot);
            const int woff = block_exscan256(nwl, sh, &totw);
            if (tid == 0) {
                npieces[row] = tot;
                fbase = (int)atomicAdd(counter + 2, (unsigned)tot); // this row's slots in the flat list
                long long b = -1;
                if (totw > 0) {
                    b = (long long)atomicAdd(counter, (unsigned)totw);
                    if (b + totw > cap) b = -1;
                }
                sbase = b;
            }
            __syncthreads();
            wword = sbase < 0 ? -1 : sbase + woff;
        }
        for (int s = s0; s < s1; ++s) {
            const int st = ss[s];
            const int en = (s + 1 < cnt) ? ss[s + 1] : len;
            const int hi = ld_hi(idx, a + st);
            const bool isFirst = (st == 0) || (fl[st] & 2);
            const bool isLast = (en == len) || (fl[en] & 2);
            bool emit = true;
            const int S = ld_wi(idx, a + st);
            const int p1 = (int)(a + st);
            const int n1 = en - st;
            int p2 = p1;
            int n = n1;
            if (isFirst && !isLast) {
                int t = s + 1;
                int ten = (t + 1 < cnt) ? ss[t + 1] : len;
                while (!((ten == len) || (fl[ten] & 2))) {
                    ++t;
                    ten = (t + 1 < cnt) ? ss[t + 1] : len;
                }
                int nx = ld_wi(idx, a + ten - 1) + 1;
                if (nx == Wi) nx = 0;
                if (nx == S) emit = false; // merged into the last run of this latitude
            } else if (!isFirst && isLast) {
                int t = s - 1;
                int tst = ss[t];
                while (!((tst == 0) || (fl[tst] & 2))) {
                    --t;
                    tst = ss[t];
                }
                const int ten = ss[t + 1];
                int nx = ld_wi(idx, a + en - 1) + 1;
                if (nx == Wi) nx = 0;
                if (nx == ld_wi(idx, a + tst)) {
                    p2 = (int)(a + tst);
                    n = n1 + (ten - tst);
                }
            }
            if (!emit) continue;
            if (pass < 0) {
                lc0 += piece_nc(n, S, ps);
                lc1 += piece_nc(n, S + 4, ps);
                continue;
            }
            int np = 0;
            const int NCT = piece_nc(n, S + sig, ps); // chunks of the whole (shifted) arc window
            for (int h0 = 0; h0 < NCT; h0 += nc_cap, ++np) {
                const int nc = min(nc_cap, NCT - h0);
                const int nw = ((nc + 1) * nunit) >> 1;
                if (pass == 0) {
                    nwl += nw;
                    continue;
                }
                int sq = S + sig + 8 * h0;
                while (sq >= Wi) sq -= Wi;
                Piece pc;
                pc.hi = hi;
                pc.S = sq;
                pc.n = n;
                pc.p1 = p1;
                pc.n1 = n1;
                pc.p2 = p2;
                pc.toff = wword < 0 ? -1 : (int)wword;
                pc.nc = nc;
                pc.jb = 8 * h0;
                pc.pad0 = pc.pad1 = pc.pad2 = 0;
                if (wword >= 0) wword += nw;
                pieces[a + wpos + np] = pc;
                flat[fbase + wpos + np] = (int)(a + wpos + np);
            }
            if (pass == 0)
                npl += np;
            else
                wpos += np;
        }
    }
}

// 8 consecutive Toeplitz entries T[u0 .. u0+8, (k, dwo)] of a piece, packed as fp16x8
__device__ __forceinline__ uint4 toeplitz8(const uint16_t *__restrict__ val, const Piece &pc, int u0, int k, int dwo,
                                           int ps, int K, int kpad)
{
    const int j0 = u0 - (pc.S & 7) - ps * dwo + pc.jb; // position in the arc
    unsigned short hv[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        const int j = j0 + q;
        hv[q] = 0;
        if (k < K && j >= 0 && j < pc.n) {
            const int pt = (j < pc.n1) ? (pc.p1 + j) : (pc.p2 + (j - pc.n1));
            hv[q] = val[(size_t)pt * kpad + k];
        }
    }
    return make_uint4((unsigned)hv[0] | ((unsigned)hv[1] << 16), (unsigned)hv[2] | ((unsigned)hv[3] << 16),
                      (unsigned)hv[4] | ((unsigned)hv[5] << 16), (unsigned)hv[6] | ((unsigned)hv[7] << 16));
}

// same for row r of a group piece (window [W, W + l) of longitudes, row arcs in px)
__device__ __forceinline__ uint4 toeplitz8x(const uint16_t *__restrict__ val, int W, int l, int jb, const PieceX &px,
                                            int r, int u0, int k, int dwo, int ps, int K, int kpad, int Wi)
{
    const int j0 = u0 - (W & 7) - ps * dwo; // relative to the piece start W (window position jb + j)
    int base = W - px.S[r];
    if (base < 0) base += Wi;
    unsigned short hv[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        const int j = j0 + q;
        hv[q] = 0;
        if (k < K && j + jb >= 0 && j + jb < l && px.n[r] > 0) {
            int jr = base + j;
            if (jr < 0)
                jr += Wi;
            else if (jr >= Wi)
                jr -= Wi;
            if (jr < px.n[r]) {
                const int pt = (jr < px.n1[r]) ? (px.p1[r] + jr) : (px.p2[r] + (jr - px.n1[r]));
                hv[q] = val[(size_t)pt * kpad + k];
            }
        }
    }
    return make_uint4((unsigned)hv[0] | ((unsigned)hv[1] << 16), (unsigned)hv[2] | ((unsigned)hv[3] << 16),
                      (unsigned)hv[4] | ((unsigned)hv[5] << 16), (unsigned)hv[6] | ((unsigned)hv[7] << 16));
}

// Per-piece Toeplitz B tiles, UMMA K-major no-swizzle layout per k-step s:
//   tile[s][h][n][e] = T[u = 16 s + 8 h + e, n = k * DW + dwo]   (h = K chunk, N rows of 16 B)
#define FQL 176 // 16 leading zeros + 128 points + the largest Toeplitz tail
template <int RG, int DW, int KPAD, bool Z = false>
__global__ void __launch_bounds__(128)
    tfill_kernel(const uint16_t *__restrict__ val, const Piece *__restrict__ pieces, const PieceX *__restrict__ pxs,
                 int Wi, int ps, int K, uint4 *__restrict__ T, int Ho, int TY, const unsigned *__restrict__ counter,
                 const int *__restrict__ flat)
{
    constexpr int MN1 = KB * DW;
    constexpr int MN = ((RG * MN1 + 7) / 8) * 8;
    // the piece's filter values are staged once in shared memory (zero-padded rows), then each
    // 16-byte Toeplitz row segment is assembled from shared memory
    __shared__ unsigned short fs[RG][KB][FQL];
    const int tid = threadIdx.x;
    // Toeplitz blocks walk the flat piece list (all rows), grid-stride
    pdl_wait(); // pieces, flat list and counters from the planner
    const int ntot = (int)counter[2];
    for (int i = blockIdx.x; i < ntot; i += Ho * TY) {
        const int pi = flat[i];
        const Piece pc = pieces[pi];
        if (pc.toff < 0) continue;
        __syncthreads();
        for (int e = tid; e < RG * KB * FQL / 2; e += 128) reinterpret_cast<unsigned *>(&fs[0][0][0])[e] = 0u;
        __syncthreads();
        if constexpr (RG > 1) {
            const PieceX px = pxs[pi];
#pragma unroll
            for (int r = 0; r < RG; ++r) {
                if (px.n[r] == 0) continue;
                // local position j (window position jb + j) maps to offset base + j of row r's arc
                int base = pc.S - px.S[r];
                if (base < 0) base += Wi;
                const int jlo = max(-16, -pc.jb), jhi = min(8 * pc.nc, pc.n - pc.jb);
                if constexpr ((KPAD & 7) == 0) {
                    constexpr int nh = KPAD >> 3;
                    const int nv = min(nh, 2);
                    for (int e = tid; e < (jhi - jlo) * nv; e += 128) {
                        const int j = jlo + e / nv;
                        const int h = e - (j - jlo) * nv;
                        int jr = base + j;
                        if (jr < 0)
                            jr += Wi;
                        else if (jr >= Wi)
                            jr -= Wi;
                        if (jr >= px.n[r]) continue;
                        const int pt = (jr < px.n1[r]) ? (px.p1[r] + jr) : (px.p2[r] + (jr - px.n1[r]));
                        const uint4 v = reinterpret_cast<const uint4 *>(val)[(size_t)pt * nh + h];
                        const unsigned w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                        for (int kk = 0; kk < 8; ++kk) {
                            const int k = 8 * h + kk;
                            if (k < K)
                                fs[r][k][16 + j]
                                    = (unsigned short)((kk & 1) ? (w[kk >> 1] >> 16) : (w[kk >> 1] & 0xFFFFu));
                        }
                    }
                } else {
                    for (int e = tid; e < (jhi - jlo) * K; e += 128) {
                        const int j = jlo + e / K;
                        const int k = e - (j - jlo) * K;
                        int jr = base + j;
                        if (jr < 0)
                            jr += Wi;
                        else if (jr >= Wi)
                            jr -= Wi;
                        if (jr >= px.n[r]) continue;
                        const int pt = (jr < px.n1[r]) ? (px.p1[r] + jr) : (px.p2[r] + (jr - px.n1[r]));
                        fs[r][k][16 + j] = val[(size_t)pt * KPAD + k];
                    }
                }
            }
        } else if constexpr ((KPAD & 7) == 0) {
            // local positions j in [jlo, jhi) are arc points jb + j
            const int jlo = max(-16, -pc.jb), jhi = min(8 * pc.nc, pc.n - pc.jb);
            constexpr int nh = KPAD >> 3;
            const int nv = min(nh, 2);
            for (int e = tid; e < (jhi - jlo) * nv; e += 128) {
                const int j = jlo + e / nv;
                const int h = e - (j - jlo) * nv;
                const int ja = pc.jb + j;
                const int pt = (ja < pc.n1) ? (pc.p1 + ja) : (pc.p2 + (ja - pc.n1));
                const uint4 v = reinterpret_cast<const uint4 *>(val)[(size_t)pt * nh + h];
                const unsigned w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int kk = 0; kk < 8; ++kk) {
                    const int k = 8 * h + kk;
                    if (k < K)
                        fs[0][k][16 + j] = (unsigned short)((kk & 1) ? (w[kk >> 1] >> 16) : (w[kk >> 1] & 0xFFFFu));
                }
            }
        } else {
            const int jlo = max(-16, -pc.jb), jhi = min(8 * pc.nc, pc.n - pc.jb);
            for (int e = tid; e < (jhi - jlo) * K; e += 128) {
                const int j = jlo + e / K;
                const int k = e - (j - jlo) * K;
                const int ja = pc.jb + j;
                const int pt = (ja < pc.n1) ? (pc.p1 + ja) : (pc.p2 + (ja - pc.n1));
                fs[0][k][16 + j] = val[(size_t)pt * KPAD + k];
            }
        }
        __syncthreads();
        uint4 *t = T + pc.toff;
        const int tot = (pc.nc + (Z ? 0 : 1)) * MN;
        const int delta = pc.S & 7;
        for (int e = tid; e < tot; e += 128) {
            const int kc = e / MN;
            const int n0 = e - kc * MN;
            const int r = RG > 1 ? n0 / MN1 : 0;
            const int n = n0 - r * MN1;
            const int k = n / DW;
            const int q0 = 16 + 8 * kc - delta - ps * (n - k * DW);
            unsigned w[4] = {0u, 0u, 0u, 0u};
            // chunk nc pads an odd last chunk and must be zero (for an interior piece of a long window
            // it would otherwise hold the next piece's first chunk)
            if (k < K && kc < pc.nc) {
#pragma unroll
                for (int h = 0; h < 4; ++h)
                    w[h] = (unsigned)fs[r][k][q0 + 2 * h] | ((unsigned)fs[r][k][q0 + 2 * h + 1] << 16);
            }
            t[e] = make_uint4(w[0], w[1], w[2], w[3]);
        }
    }
}

// The longitude-extended input copy ahead of the main kernel: xe[c, hi, q] for the positions some window of
// latitude hi reads (see lat_range), 8 channel rows per block.  In co-resident mode only the first xpre
// latitudes of the extension order (the polar rows the first items need; xcopy_kernel and the producers
// do the rest), else every row.  (torch-harmonics: the delivered kernel ran these blocks behind the
// Toeplitz builder's, in one grid; the builder is part of the plan now.)
__global__ void __launch_bounds__(128)
    xpre_kernel(const uint4 *__restrict__ x, uint4 *__restrict__ xe, int nb, int Lc, long long nrows, int Hi, int xpre,
                const unsigned long long *__restrict__ R, int hrot)
{
    pdl_trigger();
    const long long row0 = (long long)blockIdx.x * 8;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    for (int r = warp; r < 8; r += 4) {
        long long row = row0 + r;
        if (row >= nrows) continue;
        int h;
        if (xpre > 0) {
            const long long c = row / xpre;
            const int j = (int)(row - c * xpre);
            h = (j & 1) ? Hi - 1 - (j >> 1) : (j >> 1);
            row = c * Hi + h;
        } else {
            h = (int)(row % Hi);
        }
        int lo, hx;
        if (!lat_range(R, h, 1u, Lc, lo, hx)) continue;
        copy_range<32>(x + row * nb, xe + row * Lc, lo, hx, nb, hrot, lane);
    }
}

// The per-call words of the co-resident protocol: every stamp and count zero (a call's epoch is 1),
// the latitude ranges published (they are the plan's), the extension claims past the reserved tasks,
// and the main kernel's dynamic item counter.  (torch-harmonics: the delivered kernel kept these words
// across calls and told calls apart by an epoch, which a CUDA graph freezes at capture.)
constexpr int INIT_THREADS = 256;
__global__ void __launch_bounds__(INIT_THREADS)
    init_kernel(unsigned *__restrict__ stamps, int nwords, int xreserve, int item_base)
{
    for (int i = threadIdx.x; i < nwords; i += blockDim.x) {
        unsigned v = 0u;
        if (i == ST_FSTAMP)
            v = 1u;
        else if (i == ST_XCLAIM)
            v = (unsigned)xreserve;
        else if (i == ST_ICTR)
            v = (unsigned)item_base;
        stamps[i] = v;
    }
}

// Longitude extension of the input running alongside the main kernel (co-resident mode).  The
// main kernel triggers this one once its CTAs are resident, and one block of it fits next to them
// on every SM (small register / shared-memory footprint), so the extension no longer serializes
// in front of the main kernel.  Tasks are (latitude, 32-channel slice) in the main kernel's item
// order (latitudes alternately from both poles), claimed dynamically; a finished task adds its
// channel rows to its latitude's count, whose completion stamp the main kernel's producer checks
// before it loads a window of that latitude.  A producer that finds its latitude incomplete claims
// and copies tasks itself, so progress never depends on this kernel becoming resident.
#define XC_THREADS 128 // one 32-register warp per SM sub-partition (fits next to the main CTAs)
#define XC_RPT 32
#define XC_U 4 // channel rows in flight per thread
// copies task t with NT cooperating threads (thread index tid); returns its latitude / row count
template <int NT>
__device__ __forceinline__ void xcopy_task(const uint4 *__restrict__ x, uint4 *__restrict__ xe, int nb, int Lc, int Hi,
                                           int BC, int tpl, int xpre, int t, int tid, int &hi, int &nr,
                                           const unsigned long long *__restrict__ R, unsigned epoch, int hrot)
{
    const int pos = xpre + t / tpl;
    const int c0 = (t - (pos - xpre) * tpl) * XC_RPT;
    hi = (pos & 1) ? Hi - 1 - (pos >> 1) : (pos >> 1);
    nr = min(BC, c0 + XC_RPT) - c0;
    int lo, hx;
    if (!lat_range(R, hi, epoch, Lc, lo, hx)) return;
    const size_t sstr = (size_t)Hi * nb, dstr = (size_t)Hi * Lc; // channel strides (16-byte units)
    const uint4 *src = x + (size_t)c0 * sstr + (size_t)hi * nb;
    uint4 *dst = xe + (size_t)c0 * dstr + (size_t)hi * Lc;
    for (int j = tid; j < nb; j += NT) {
        int q1 = j - hrot; // first position of input chunk j in the range
        if (q1 < 0) q1 += nb;
        if (q1 < lo) q1 += nb;
        for (int r0 = 0; r0 < nr; r0 += XC_U) {
            const int nu = min(XC_U, nr - r0);
            uint4 v[XC_U];
            const uint4 *sp = src + (size_t)r0 * sstr + j;
#pragma unroll
            for (int u = 0; u < XC_U; ++u) {
                if (u < nu) v[u] = __ldg(sp);
                sp += sstr;
            }
            for (int q = q1; q < hx; q += nb) {
                uint4 *dp = dst + (size_t)r0 * dstr + q;
#pragma unroll
                for (int u = 0; u < XC_U; ++u) {
                    if (u < nu) *dp = v[u];
                    dp += dstr;
                }
            }
        }
    }
}
__device__ __forceinline__ int xcopy_ntask(int Hi, int tpl, int xpre) { return (Hi - xpre) * tpl; }

// one warp claims and copies an extension task (false: every task is claimed already)
__device__ __noinline__ bool xcopy_help(const uint4 *__restrict__ x, uint4 *__restrict__ xe, int nb, int Lc, int Hi,
                                        int BC, int tpl, int xpre, unsigned *__restrict__ stamps, unsigned epoch,
                                        const unsigned long long *__restrict__ R, int hrot, int xres)
{
    const int lane = threadIdx.x & 31;
    const int ntask = xcopy_ntask(Hi, tpl, xpre);
    int t = 0;
    if (lane == 0) t = (int)atomicAdd(stamps + ST_XCLAIM, 1u);
    t = __shfl_sync(0xffffffffu, t, 0);
    if (t >= ntask) {
        // every dynamic task is claimed: take a reserved task whose copy block has not started (the
        // copy kernel may be unable to run next to this grid), so no latitude waits on a block that
        // never becomes resident.  The claim words are checked 32 at a time.
        t = ntask;
        for (int r0 = 0; r0 < xres && t == ntask; r0 += 32) {
            const int r = r0 + lane;
            unsigned open = __ballot_sync(0xffffffffu, r < xres && ld_relaxed_u32(stamps + ST_RCLAIM + r) != epoch);
            while (open) {
                const int l = __ffs(open) - 1;
                int got = 0;
                if (lane == l) got = atomicExch(stamps + ST_RCLAIM + r, epoch) != epoch ? 1 : 0;
                if (__shfl_sync(0xffffffffu, got, l)) {
                    t = r0 + l;
                    break;
                }
                open &= open - 1u;
            }
        }
    }
    if (t >= ntask) return false;
    int hi, nr;
    xcopy_task<32>(x, xe, nb, Lc, Hi, BC, tpl, xpre, t, lane, hi, nr, R, epoch, hrot);
    fence_proxy_async_global();
    __threadfence();
    __syncwarp();
    if (lane == 0) {
        __threadfence();
        // Publish global completion once per latitude, not once per channel-slice task.  The latitude
        // counter reaches BC only after every slice's writes are visible through the RMW release chain.
        if (count_add(stamps + ST_LCNT + hi, stamps + ST_LSTAMP + hi, (unsigned)nr, (unsigned)BC, epoch))
            count_add(stamps + ST_XCNT, stamps + ST_XSTAMP, 1u, (unsigned)(Hi - xpre), epoch);
    }
    __syncwarp();
    return true;
}

__global__ void __launch_bounds__(XC_THREADS, 16)
    xcopy_kernel(const uint4 *__restrict__ x, uint4 *__restrict__ xe, int nb, int Lc, int Hi, int BC, int tpl, int xpre,
                 unsigned *__restrict__ stamps, unsigned epoch, const unsigned long long *__restrict__ R, int hrot,
                 int xres)
{
    __shared__ int s_t[2];
    const int tid = threadIdx.x;
    if (R && tid == 0)
        while (ld_acquire_u32(stamps + ST_FSTAMP) != epoch) __nanosleep(64); // latitude ranges
    const int ntask = xcopy_ntask(Hi, tpl, xpre); // the first xpre positions: the builder's part
    // Reserve one early task per dedicated copy block before producer helpers begin claiming.
    // This keeps latency-critical prefix tasks on the 128-thread copy path.
    if (tid == 0) {
        // the block's reserved task, unless a producer helper has taken it already
        const bool mine = (int)blockIdx.x < xres && atomicExch(stamps + ST_RCLAIM + blockIdx.x, epoch) != epoch;
        s_t[0] = mine ? (int)blockIdx.x : (int)atomicAdd(stamps + ST_XCLAIM, 1u);
    }
    __syncthreads();
    for (int it = 0;; ++it) {
        const int t = s_t[it & 1];
        if (t >= ntask) break;
        if (tid == 0) s_t[(it + 1) & 1] = (int)atomicAdd(stamps + ST_XCLAIM, 1u);
        if (tid < XC_RPT && t + (int)gridDim.x < ntask) {
            // pull a probable later claim's rows into L2 while this task is copied
            const int t2 = t + (int)gridDim.x;
            const int pos = xpre + t2 / tpl;
            const int d0 = (t2 - (pos - xpre) * tpl) * XC_RPT;
            const int h2 = (pos & 1) ? Hi - 1 - (pos >> 1) : (pos >> 1);
            if (d0 + tid < BC)
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(x + ((size_t)(d0 + tid) * Hi + h2) * nb),
                             "r"((unsigned)nb * 16u)
                             : "memory");
        }
        int hi, nr;
        xcopy_task<XC_THREADS>(x, xe, nb, Lc, Hi, BC, tpl, xpre, t, tid, hi, nr, R, epoch, hrot);
        fence_proxy_async_global();
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            __threadfence();
            if (count_add(stamps + ST_LCNT + hi, stamps + ST_LSTAMP + hi, (unsigned)nr, (unsigned)BC, epoch))
                count_add(stamps + ST_XCNT, stamps + ST_XSTAMP, 1u, (unsigned)(Hi - xpre), epoch);
        }
    }
}

// ------------------------------------------------------------------------------------------
// mbarrier and bulk-copy helpers
// ------------------------------------------------------------------------------------------
__device__ __forceinline__ unsigned smem_u32(const void *p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(unsigned bar, unsigned cnt)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(cnt));
}
__device__ __forceinline__ void mbar_arrive(unsigned bar)
{
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(unsigned bar, unsigned bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned bar, unsigned parity)
{
    unsigned ok = 0;
    do {
        asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                     " selp.u32 %0, 1, 0, p;\n}\n"
                     : "=r"(ok)
                     : "r"(bar), "r"(parity)
                     : "memory");
    } while (!ok);
}
__device__ __forceinline__ bool mbar_test(unsigned bar, unsigned parity)
{
    unsigned ok = 0;
    asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 " selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok)
                 : "r"(bar), "r"(parity)
                 : "memory");
    return ok != 0;
}
__device__ __forceinline__ void bulk_load(unsigned dst, const void *src, unsigned bytes, unsigned bar)
{
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dst),
                 "l"(src), "r"(bytes), "r"(bar)
                 : "memory");
}
__device__ __forceinline__ void bulk_load_reuse(unsigned dst, const void *src, unsigned bytes, unsigned bar,
                                                uint64_t policy)
{
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                 " [%0], [%1], %2, [%3], %4;\n" ::"r"(dst),
                 "l"(src), "r"(bytes), "r"(bar), "l"(policy)
                 : "memory");
}
__device__ __forceinline__ uint64_t reuse_policy()
{
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_last.L2::evict_unchanged.b64 %0, 1.0;\n" : "=l"(p));
    return p;
}

#define QN 4 // item queue depth (producer -> the item's consumers)

struct HkParams {
    const uint16_t *x;   // fp16 or bf16 bits
    const uint16_t *val; // the filter values, in the activations' dtype
    const Piece *pieces;
    const int *npieces;
    const int64_t *off;
    const uint4 *T;
    const PieceX *pxs;
    uint16_t *out;
    int bf16; // activations are bf16 (else fp16): the MMA's operand format and the epilogue's rounding
    int BC, Hi, Wi, Ho, Wo, ps, K, kpad;
    int nb, RPC, HB, G, NCG, nitems, NG, ticket; // NG: item rows (row groups)
    unsigned ncg_recip;                          // floor(2^32 / NCG), saturated for NCG == 1
    int nvs;                                     // log2 of the number of views (3 or 5)
    int nst, xs_bytes, stage_bytes, bar_off;
    int z_off;         // one all-zero Toeplitz chunk after the barrier area
    int *ictr;         // dynamic item counter (zeroed per call)
    const int *rowsig; // per item row (group): 0 or 4, the input shift of the GEMM-row output blocks
    // co-resident input extension (co != 0): completion stamps instead of the grid dependency
    unsigned *stamps;
    unsigned epoch;
    int co, xpre; // xpre: extension positions done by the builder (ready once it has finished)
    int xres;     // extension tasks reserved for the copy blocks (claim words ST_RCLAIM)
    uint4 *xe;    // extended input (co: the producer helps with unclaimed extension tasks)
    int Lc, tpl;
    const unsigned long long *R; // extended-copy latitude ranges (null: whole rows)
    int hrot;                    // extended-copy rotation (see lat_range)
};

__device__ __forceinline__ void decode_item(const HkParams &p, int item, int &ho, int &cg)
{
    const unsigned q0 = __umulhi((unsigned)item, p.ncg_recip);
    const int r = (int)(q0 + ((unsigned)item - q0 * (unsigned)p.NCG >= (unsigned)p.NCG));
    cg = item - r * p.NCG;
    ho = (r & 1) ? (p.NG - 1 - (r >> 1)) : (r >> 1);
}

__device__ __forceinline__ void tma_load_4d(unsigned dst, const void *tmap, int c0, int c1, int c2, int c3, unsigned bar)
{
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3, %4, %5}], [%6];\n" ::"r"(dst),
                 "l"(tmap), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(bar)
                 : "memory");
}
__device__ __forceinline__ void prefetch_tmap(const void *tmap)
{
    asm volatile("prefetch.tensormap [%0];\n" ::"l"(tmap) : "memory");
}

// nv (8 or 32) views of the extended copy, view j starting 16 j bytes in, with rows of nv chunks
// (128 or 512 bytes): a window starting at chunk B8 is box row B8 / nv of view B8 % nv.  The
// TMA engine handles a window in G * RPC / nv rows, so the widest rows the halo admits are used.
#define NVMAX 32
template <int NV> struct TmapX {
    CUtensorMap m[NV];
};
