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
// The DISCO forward contraction on the tensor cores of sm_90a (H100), as GEMMs of a Hankel matrix of input longitudes
// and a Toeplitz matrix of filter values; float16 or bfloat16 activations, psi in the kpacked layout.
//
// A port of the sm_100a kernel disco_cuda_fwd_hankel_sm100.cu (Kernel Factory campaign disco-fwd-b200-v2, candidate
// 2c7ddc19, with torch-harmonics' changes listed there) to wgmma. Unchanged: the plan (pieces, Toeplitz tiles,
// latitude ranges), the input extension, the producer warps, the shared-memory stages and the operand layouts,
// which are the K-major no-swizzle core-matrix layouts both MMA generations read (the Hankel A operand is a
// descriptor over overlapping core matrices; a piece's odd last chunk pairs with the next piece's first through a
// K-direction stride that spans the two stages). Changed, as Hopper has no tensor memory:
//
// * the accumulators are registers of the warpgroup that issues the MMAs: consumer warpgroup 1 + m owns M tile m
//   (two m64 halves), accumulates an item over its pieces and stores it (the sm_100a kernel's separate MMA
//   issuer and epilogue warps, and its double-buffered accumulators, are gone);
// * a stage is released once the wgmma group that read it has completed (wgmma.wait_group), one per warp;
// * the families are <DW, MT, CPS, RG> with MT = 2 at one CTA per SM or MT = 1 at two: the row pairs at DW 8
//   (MMA width 144) and the three-CTA small-channel family of the sm_100a kernel are not taken.
//
// As there, NaN or Inf in an input longitude the stencil does not reach may still reach outputs (the Toeplitz
// zeros multiply it); accepted.
//
// The sm_100a candidate's description (its MMA, accumulator and epilogue parts are those of the sm_100a kernel):
//
// DISCO S2 forward contraction (K-packed psi), fp16 in / fp32 accumulate / fp16 out.
//
// Formulation.  For every output row ho the support is split into "pieces": runs of points on
// one input latitude hi with consecutive longitudes (mod Wi).  For a piece with start S, n points
// and filter h_k[j] = val[point j][k]:
//   out[c,k,ho,wo] += sum_j h_k[j] * x[c, hi, S + j + wo*ps].
// Write wo*ps = 8*b + ps*dwo with dwo in [0, DW), DW = 8/ps, and S = B + delta (B a multiple of 8).
// Then x[c, hi, S + j + wo*ps] = x[c, hi, B + 8*b + u] with u = delta + j + ps*dwo, so
//   D[(c,b), (k,dwo)] = sum_u A[(c,b), u] * T[u, (k,dwo)],
//   A[(c,b), u] = x[c, hi, B + 8*b + u]      (Hankel in (b, u) at 8-element granularity)
//   T[u, (k,dwo)] = h_k[u - delta - ps*dwo]  (Toeplitz, zero outside the piece).
// A is a UMMA operand straight from a contiguous longitude run in shared memory: in the K-major
// no-swizzle core-matrix layout row r sits at 16*r bytes and the next 8-element K chunk at
// LBO = 16 bytes, so one copy of the longitude run serves all 128 rows (blocks) of an M tile.
// GEMM rows are (channel, block) pairs, so the tensor-core M dimension is fully used for any
// channel count; every block holds DW outputs, so outputs leave as DW-wide vectors.
// Rows of a channel are padded with HB halo blocks so that every block's window stays inside the
// channel's (wrapped) longitude run.

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
// mbarrier, bulk-copy and wgmma helpers
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
// K-major, no-swizzle wgmma operand descriptor from a 16-byte-unit start address and K-direction stride: core
// matrices of 8 rows x 16 B, the next 8 rows 128 B on (the layout and fields of the delivered tcgen05 descriptor,
// without its sm_100 version bit)
__device__ __forceinline__ uint64_t mkdesc(unsigned start16, unsigned lbo16)
{
    return (uint64_t)((start16 & 0x3FFFu) | ((lbo16 & 0x3FFFu) << 16)) | (8ull << 32);
}
__device__ __forceinline__ bool elect_one()
{
    unsigned pred = 0;
    asm volatile("{\n .reg .pred p;\n elect.sync _|p, 0xffffffff;\n selp.u32 %0, 1, 0, p;\n}\n" : "=r"(pred));
    return pred != 0;
}
// ------------------------------------------------------------------------------------------
// wgmma (sm_90a): D[64 x N] (+)= A[64 x 16] * B[16 x N], A and B K-major in shared memory, D fp32 in the
// registers of the issuing warpgroup. Thread t of the warpgroup holds, for 8-column block j, the columns
// 8 j + 2 (t % 4) + {0, 1} of the rows 16 (t / 32) + (t % 32) / 4 (regs 4 j, 4 j + 1) and of the row 8 below
// (regs 4 j + 2, 4 j + 3). scale_d = 0 overwrites D, 1 accumulates.
// ------------------------------------------------------------------------------------------
template <int N> struct Wgmma;
template <> struct Wgmma<16> {
    static constexpr int R = 8;
    __device__ __forceinline__ static void mma(float (&d)[8], uint64_t a, uint64_t b, int scale_d, bool bf16)
    {
        if (bf16) {
            asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %10, 0;\n"
                         " wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, "
                         "1, 1, 0, 0;\n}\n"
                         : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                         : "l"(a), "l"(b), "r"(scale_d)
                         : "memory");
        } else {
            asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %10, 0;\n"
                         " wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, 1, "
                         "1, 0, 0;\n}\n"
                         : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                         : "l"(a), "l"(b), "r"(scale_d)
                         : "memory");
        }
    }
};
template <> struct Wgmma<24> {
    static constexpr int R = 12;
    __device__ __forceinline__ static void mma(float (&d)[12], uint64_t a, uint64_t b, int scale_d, bool bf16)
    {
        if (bf16) {
            asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %14, 0;\n"
                         " wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 "
                         "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11}, %12, %13, p, 1, 1, 0, 0;\n}\n"
                         : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
                           "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
                         : "l"(a), "l"(b), "r"(scale_d)
                         : "memory");
        } else {
            asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %14, 0;\n"
                         " wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11}, "
                         "%12, %13, p, 1, 1, 0, 0;\n}\n"
                         : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
                           "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
                         : "l"(a), "l"(b), "r"(scale_d)
                         : "memory");
        }
    }
};
template <> struct Wgmma<40> {
    static constexpr int R = 20;
    __device__ __forceinline__ static void mma(float (&d)[20], uint64_t a, uint64_t b, int scale_d, bool bf16)
    {
        if (bf16) {
            asm volatile(
                "{\n .reg .pred p;\n setp.ne.b32 p, %22, 0;\n"
                " wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19}, %20, %21, p, 1, 1, 0, 0;\n}\n"
                : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                  "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
                  "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19])
                : "l"(a), "l"(b), "r"(scale_d)
                : "memory");
        } else {
            asm volatile(
                "{\n .reg .pred p;\n setp.ne.b32 p, %22, 0;\n"
                " wgmma.mma_async.sync.aligned.m64n40k16.f32.f16.f16 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19}, %20, %21, p, 1, 1, 0, 0;\n}\n"
                : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                  "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
                  "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19])
                : "l"(a), "l"(b), "r"(scale_d)
                : "memory");
        }
    }
};
template <> struct Wgmma<72> {
    static constexpr int R = 36;
    __device__ __forceinline__ static void mma(float (&d)[36], uint64_t a, uint64_t b, int scale_d, bool bf16)
    {
        if (bf16) {
            asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %38, 0;\n"
                         " wgmma.mma_async.sync.aligned.m64n72k16.f32.bf16.bf16 "
                         "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%"
                         "25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, %36, %37, p, 1, 1, 0, 0;\n}\n"
                         : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                           "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]),
                           "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]),
                           "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]),
                           "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
                         : "l"(a), "l"(b), "r"(scale_d)
                         : "memory");
        } else {
            asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %38, 0;\n"
                         " wgmma.mma_async.sync.aligned.m64n72k16.f32.f16.f16 "
                         "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%"
                         "25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, %36, %37, p, 1, 1, 0, 0;\n}\n"
                         : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                           "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]),
                           "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]),
                           "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]),
                           "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
                         : "l"(a), "l"(b), "r"(scale_d)
                         : "memory");
        }
    }
};
template <> struct Wgmma<80> {
    static constexpr int R = 40;
    __device__ __forceinline__ static void mma(float (&d)[40], uint64_t a, uint64_t b, int scale_d, bool bf16)
    {
        if (bf16) {
            asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %42, 0;\n"
                         " wgmma.mma_async.sync.aligned.m64n80k16.f32.bf16.bf16 "
                         "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%"
                         "25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39}, %40, %41, p, 1, 1, 0, 0;\n}\n"
                         : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
                           "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
                           "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
                           "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
                           "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]),
                           "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39])
                         : "l"(a), "l"(b), "r"(scale_d)
                         : "memory");
        } else {
            asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %42, 0;\n"
                         " wgmma.mma_async.sync.aligned.m64n80k16.f32.f16.f16 "
                         "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%"
                         "25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39}, %40, %41, p, 1, 1, 0, 0;\n}\n"
                         : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
                           "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
                           "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
                           "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
                           "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]),
                           "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39])
                         : "l"(a), "l"(b), "r"(scale_d)
                         : "memory");
        }
    }
};
template <> struct Wgmma<144> {
    static constexpr int R = 72;
    __device__ __forceinline__ static void mma(float (&d)[72], uint64_t a, uint64_t b, int scale_d, bool bf16)
    {
        if (bf16) {
            asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %74, 0;\n"
                         " wgmma.mma_async.sync.aligned.m64n144k16.f32.bf16.bf16 "
                         "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%"
                         "25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%"
                         "48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%"
                         "71}, %72, %73, p, 1, 1, 0, 0;\n}\n"
                         : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
                           "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
                           "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
                           "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
                           "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]),
                           "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]),
                           "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
                           "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
                           "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]),
                           "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]),
                           "+f"(d[70]), "+f"(d[71])
                         : "l"(a), "l"(b), "r"(scale_d)
                         : "memory");
        } else {
            asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %74, 0;\n"
                         " wgmma.mma_async.sync.aligned.m64n144k16.f32.f16.f16 "
                         "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%"
                         "25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%"
                         "48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%"
                         "71}, %72, %73, p, 1, 1, 0, 0;\n}\n"
                         : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
                           "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
                           "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
                           "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
                           "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]),
                           "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]),
                           "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
                           "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
                           "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]),
                           "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]),
                           "+f"(d[70]), "+f"(d[71])
                         : "l"(a), "l"(b), "r"(scale_d)
                         : "memory");
        }
    }
};
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int NP> __device__ __forceinline__ void wgmma_wait()
{
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(NP) : "memory");
}
// keeps the compiler from moving accesses to accumulator registers across a wgmma wait (as CUTLASS's
// warpgroup_fence_operand)
template <int R> __device__ __forceinline__ void fence_acc(float (&d)[R])
{
#pragma unroll
    for (int i = 0; i < R; ++i) asm volatile("" : "+f"(d[i])::"memory");
}

// ------------------------------------------------------------------------------------------
// Main kernel.  Item = (output row ho, group of G channels); GEMM row g = (c - c0) * RPC + b
// (block b of channel c; b >= nb are halo rows, discarded), M tiles of 128 rows, G * RPC <= 128 MT.
// Shared-memory slot g = (c - c0) * RPC + i holds x[c, hi, B + 8 i (mod Wi)] for the current
// piece, so the A operand of M tile m starts at slot 128 m.  The slots come from one 4-D TMA box
// of the longitude-extended input copy (TX) or from per-channel bulk copies (two runs at the seam).
// Warp roles (sm_90a): warpgroup 0 holds the producer warp(s) (warps 0 .. NPW-1, the rest idle); warpgroup
// 1 + m owns M tile m: it issues the tile's wgmmas (two m64 halves per k-step), keeps the accumulators in its
// registers through the item and stores them (fp32 -> fp16 / bf16 -> global).
// ------------------------------------------------------------------------------------------
#define QN 4 // item queue depth (producer -> consumer warpgroups)

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

// The configuration of a main-kernel family <DW, MT, CPS, RG>: MT M tiles of 128 GEMM rows per item, one consumer
// warpgroup each, CPS CTAs per SM.  The accumulators are registers (MN / 2 per thread and m64 half), so the widest
// families of the sm_100a kernel (row pairs at DW 8, MN 144) are not taken.
template <int DW, int MT_, int CPS = 1, int RG = 1> struct HkCfg {
    static constexpr int MN1 = KB * DW;                 // logical columns per output row
    static constexpr int MN = ((RG * MN1 + 7) / 8) * 8; // exact MMA width across row groups
    static_assert(MN <= 80, "the accumulators of wider MMAs do not fit the register file");
    static constexpr int MT = MT_;
    static constexpr int NWG = MT; // consumer warpgroups
    // One-CTA kernels use a second producer so adjacent pieces can issue their TMA loads in
    // parallel; other configurations keep a single producer to preserve occupancy.
    static constexpr int NPW = CPS == 1 ? 2 : 1;
    static constexpr int THREADS = 128 * (1 + NWG);
};

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

template <int DW, bool TX, int MT_, int CPS, int RG, bool LP, int NV, bool Z>
__device__ __forceinline__ void hk_body(const TmapX<NV> &tmap_x, const HkParams &p)
{
    static_assert(NV == 8 || NV == NVMAX);
    static_assert(!LP || RG == 1);
    using Cfg = HkCfg<DW, MT_, CPS, RG>;
    constexpr int MN = Cfg::MN;
    constexpr int MN1 = Cfg::MN1;
    constexpr int NWG = Cfg::NWG;
    constexpr int NTH = Cfg::THREADS;
    constexpr int NPW = Cfg::NPW;
    extern __shared__ __align__(1024) unsigned char smem[];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if constexpr (TX)
        if (warp == 0 && lane < (1 << p.nvs)) prefetch_tmap(&tmap_x.m[lane]);
    const int nst = p.nst;
    unsigned char *bar_area = smem + p.bar_off;
    uint64_t *bars = reinterpret_cast<uint64_t *>(bar_area);
    // [0,nst) full, [nst,2nst) empty, then item queue full/empty
    int *sks = reinterpret_cast<int *>(bar_area + 8 * (2 * nst + 2 * QN));
    int2 *qbuf = reinterpret_cast<int2 *>(bar_area + 8 * (2 * nst + 2 * QN) + 4 * 16);
    long long *qpb = reinterpret_cast<long long *>(bar_area + 512);
    const unsigned bar0 = smem_u32(bars);
    auto FULL = [&](int s) { return bar0 + 8u * s; };
    auto EMPTY = [&](int s) { return bar0 + 8u * (nst + s); };
    auto QFULL = [&](int q) { return bar0 + 8u * (2 * nst + q); };
    auto QEMPTY = [&](int q) { return bar0 + 8u * (2 * nst + QN + q); };
    const int grows = p.G * p.RPC; // GEMM rows filled from the slots

    if (tid == 0) {
        for (int s = 0; s < nst; ++s) {
            mbar_init(FULL(s), 1);
            // one arrival per consumer warp, once its wgmmas reading the stage have completed
            mbar_init(EMPTY(s), 4 * NWG);
        }
        for (int q = 0; q < QN; ++q) {
            mbar_init(QFULL(q), 1);
            // the consumer warps and the second producer
            mbar_init(QEMPTY(q), 4 * NWG + NPW - 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    if constexpr (Z)
        for (int e = tid; e < MN; e += NTH) reinterpret_cast<uint4 *>(smem + p.z_off)[e] = make_uint4(0, 0, 0, 0);
    // slots past the loaded rows are read only by discarded GEMM rows: keep them finite
    for (int s = 0; s < nst; ++s) {
        uint4 *xs = reinterpret_cast<uint4 *>(smem + (size_t)s * p.stage_bytes);
        for (int e = grows + tid; e < p.xs_bytes / 16; e += NTH) xs[e] = make_uint4(0, 0, 0, 0);
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();
    const int ps = p.ps;
    pdl_wait(); // pieces, Toeplitz tiles, extended input (or its builder part) and the item counter
    // co-resident extension: the extension kernel is this kernel's programmatic dependent; it can
    // only become resident once every CTA of this grid is, so it never displaces one
    if (p.co) pdl_trigger();

    if (warp < NPW) {
        // ---------------- producer(s) ----------------
        const uint64_t tpolicy = reuse_policy();
        // co-resident extension: before a window of latitude hi is loaded, its channel rows must have
        // been extended.  Until the extension kernel has finished every task, each item's latitudes
        // are checked (one relaxed poll per lane, then an acquire fence and a proxy fence so that the
        // TMA reads observe the generic-proxy writes).
        // (compiled only into variants the host can run in co-resident mode: TMA input, CPS < 3)
        constexpr bool COK = TX && CPS < 3;
        bool xall = !COK || p.co == 0;
        bool xhelp_done = false;
        auto lat_check = [&](int h, bool valid) { // warp-collective
            if constexpr (COK) {
                if (xall) return;
                const unsigned xd = ld_relaxed_u32(p.stamps + ST_XSTAMP);
                // extension position of latitude h (alternately from both poles); the builder's prefix is
                // complete once the main kernel has started
                const int lpos = 2 * h < p.Hi ? 2 * h : 2 * (p.Hi - 1 - h) + 1;
                bool ok = !valid || lpos < p.xpre || ld_relaxed_u32(p.stamps + ST_LSTAMP + h) == p.epoch;
                // every lane observed the final value itself, so its acquire fence below synchronizes
                const bool fin = __all_sync(0xffffffffu, xd == p.epoch);
                if (!fin) {
                    while (!__all_sync(0xffffffffu, ok)) {
                        // help: claim and copy an extension task (claims run in priority order, so the missing
                        // latitude is reached); once every task is claimed, the claimants are running and finish
                        if (xhelp_done
                            || !xcopy_help(reinterpret_cast<const uint4 *>(p.x), p.xe, p.nb, p.Lc, p.Hi, p.BC, p.tpl,
                                           p.xpre, p.stamps, p.epoch, p.R, p.hrot, p.xres)) {
                            xhelp_done = true;
                            __nanosleep(128);
                        }
                        if (!ok) ok = ld_relaxed_u32(p.stamps + ST_LSTAMP + h) == p.epoch;
                    }
                }
                // each lane acquires what it observed; the warp barrier then orders every lane's acquire
                // before any lane's TMA issue (lane 0 issues windows of other lanes' pieces)
                fence_acq_rel_gpu();
                __syncwarp();
                fence_proxy_async_global();
                if (fin) xall = true;
            }
        };
        if constexpr (LP) {
            // For short input rows a single lane can issue each piece's independent TMA/bulk-copy
            // transaction.  Issue a prefix of ready stages in parallel across the producer warp;
            // scratch-overflow pieces still fall back to a whole-warp Toeplitz build.
            auto meta = [&](int it_, int &np_, int64_t &pb_) {
                np_ = 0;
                pb_ = 0;
                if (it_ < p.nitems) {
                    int h_, t_;
                    decode_item(p, it_, h_, t_);
                    np_ = p.npieces[h_];
                    pb_ = p.off[h_];
                }
            };
            int ticket_next = 0, ticket_left = 0;
            auto claim_one = [&]() {
                if (ticket_left == 0) {
                    int base = 0;
                    if (lane == 0) base = atomicAdd(p.ictr, p.ticket);
                    ticket_next = __shfl_sync(0xffffffffu, base, 0);
                    ticket_left = p.ticket;
                }
                --ticket_left;
                return ticket_next++;
            };

            int s = 0, P = 0, qs = 0, nq = 0;
            unsigned ph = 0, qph = 0;
            int item = (int)blockIdx.x;
            int item1 = claim_one();
            int np, np1;
            int64_t pbase, pb1;
            meta(item, np, pbase);
            meta(item1, np1, pb1);
            Piece mine;
            if (lane < np) mine = p.pieces[pbase + lane];
            int fut = claim_one();
            const unsigned xb = (unsigned)grows * 16u;
            const unsigned rpcb = (unsigned)p.RPC * 16u;

            while (true) {
                if (nq >= QN) mbar_wait(QEMPTY(qs), qph ^ 1u);
                if (lane == 0) {
                    qbuf[qs] = make_int2(item, np);
                    mbar_arrive(QFULL(qs));
                }
                __syncwarp();
                if (++qs == QN) {
                    qs = 0;
                    qph ^= 1u;
                }
                ++nq;
                if (item >= p.nitems) break;
                lat_check(lane < np ? mine.hi : 0, lane < np);

                int ho, cg;
                decode_item(p, item, ho, cg);
                const int c0 = cg * p.G;
                Piece nmine;
                if (lane < np1) nmine = p.pieces[pb1 + lane];
                const int item2 = fut;
                int np2;
                int64_t pb2;
                meta(item2, np2, pb2);
                fut = claim_one();

                for (int q0 = 0; q0 < np; q0 += 32) {
                    const int npass = min(32, np - q0);
                    if (q0 > 0) {
                        if (lane < npass) mine = p.pieces[pbase + q0 + lane];
                        lat_check(lane < npass ? mine.hi : 0, lane < npass);
                    }
                    int sl = s + lane;
                    unsigned phl = ph;
                    while (sl >= nst) {
                        sl -= nst;
                        phl ^= 1u;
                    }
                    const bool reused = P + lane >= nst;
                    const unsigned bad = __ballot_sync(0xffffffffu, lane < npass && mine.toff < 0);
                    int done = 0;
                    while (done < npass) {
                        const unsigned remaining_bad = bad & (0xffffffffu << done);
                        if (remaining_bad & (1u << done)) {
                            const int sd = __shfl_sync(0xffffffffu, sl, done);
                            const unsigned phd = __shfl_sync(0xffffffffu, phl, done);
                            if (__shfl_sync(0xffffffffu, (int)reused, done)) mbar_wait(EMPTY(sd), phd ^ 1u);

                            Piece pc;
                            pc.hi = __shfl_sync(0xffffffffu, mine.hi, done);
                            pc.S = __shfl_sync(0xffffffffu, mine.S, done);
                            pc.n = __shfl_sync(0xffffffffu, mine.n, done);
                            pc.p1 = __shfl_sync(0xffffffffu, mine.p1, done);
                            pc.n1 = __shfl_sync(0xffffffffu, mine.n1, done);
                            pc.p2 = __shfl_sync(0xffffffffu, mine.p2, done);
                            pc.jb = __shfl_sync(0xffffffffu, mine.jb, done);
                            pc.nc = __shfl_sync(0xffffffffu, mine.nc, done);
                            unsigned char *st = smem + (size_t)sd * p.stage_bytes;
                            const unsigned xa = smem_u32(st);
                            uint4 *ts = reinterpret_cast<uint4 *>(st + p.xs_bytes);
                            for (int e = lane; e < (pc.nc + (Z ? 0 : 1)) * MN; e += 32) {
                                const int kc = e / MN;
                                const int n = e - kc * MN;
                                const int k = n / DW;
                                ts[e] = kc < pc.nc ? toeplitz8(p.val, pc, 8 * kc, k, n - k * DW, ps, p.K, p.kpad) :
                                                     make_uint4(0, 0, 0, 0);
                            }
                            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                            __syncwarp();
                            if (lane == 0) {
                                sks[sd] = pc.nc;
                                mbar_expect_tx(FULL(sd), xb);
                            }
                            __syncwarp();
                            const int B8 = pc.S >> 3;
                            if constexpr (TX) {
                                int q0 = B8 - p.hrot;
                                if (q0 < 0) q0 += p.nb;
                                if (lane == 0)
                                    tma_load_4d(xa, &tmap_x.m[q0 & ((1 << p.nvs) - 1)], 0, q0 >> p.nvs, pc.hi, c0,
                                                FULL(sd));
                            } else {
                                for (int cl = lane; cl < p.G; cl += 32) {
                                    const int c = c0 + cl;
                                    int ch = B8;
                                    const uint16_t *src = p.x + ((size_t)(c < p.BC ? c : p.BC - 1) * p.Hi + pc.hi) * p.Wi;
                                    unsigned dst = xa + (unsigned)cl * rpcb;
                                    for (int rem = p.RPC; rem;) {
                                        const int len = min(rem, p.nb - ch);
                                        bulk_load(dst, src + 8 * ch, (unsigned)len * 16u, FULL(sd));
                                        dst += (unsigned)len * 16u;
                                        rem -= len;
                                        ch = 0;
                                    }
                                }
                            }
                            __syncwarp();
                            ++done;
                            continue;
                        }

                        int lim = remaining_bad ? (__ffs(remaining_bad) - 1) : npass;
                        lim = min(lim, done + nst);
                        bool ready = false;
                        if (lane >= done && lane < lim) ready = !reused || mbar_test(EMPTY(sl), phl ^ 1u);
                        const unsigned ready_mask = __ballot_sync(0xffffffffu, ready);
                        const unsigned first_not_ready = ~(ready_mask >> done);
                        const int count = first_not_ready ? (__ffs(first_not_ready) - 1) : (32 - done);
                        if (count == 0) {
                            if (lane == done) mbar_wait(EMPTY(sl), phl ^ 1u);
                            __syncwarp();
                            continue;
                        }

                        if (lane >= done && lane < done + count) {
                            const unsigned tb = (unsigned)(mine.nc + (Z ? 0 : 1)) * MN * 16u;
                            const unsigned xa = smem_u32(smem + (size_t)sl * p.stage_bytes);
                            sks[sl] = mine.nc;
                            mbar_expect_tx(FULL(sl), xb + tb);
                            bulk_load_reuse(xa + (unsigned)p.xs_bytes, p.T + mine.toff, tb, FULL(sl), tpolicy);
                            const int B8 = mine.S >> 3;
                            if constexpr (TX) {
                                int q0 = B8 - p.hrot;
                                if (q0 < 0) q0 += p.nb;
                                tma_load_4d(xa, &tmap_x.m[q0 & ((1 << p.nvs) - 1)], 0, q0 >> p.nvs, mine.hi, c0,
                                            FULL(sl));
                            } else {
                                for (int cl = 0; cl < p.G; ++cl) {
                                    const int c = c0 + cl;
                                    int ch = B8;
                                    const uint16_t *src
                                        = p.x + ((size_t)(c < p.BC ? c : p.BC - 1) * p.Hi + mine.hi) * p.Wi;
                                    unsigned dst = xa + (unsigned)cl * rpcb;
                                    for (int rem = p.RPC; rem;) {
                                        const int len = min(rem, p.nb - ch);
                                        bulk_load(dst, src + 8 * ch, (unsigned)len * 16u, FULL(sl));
                                        dst += (unsigned)len * 16u;
                                        rem -= len;
                                        ch = 0;
                                    }
                                }
                            }
                        }
                        __syncwarp();
                        done += count;
                    }
                    P += npass;
                    s += npass;
                    while (s >= nst) {
                        s -= nst;
                        ph ^= 1u;
                    }
                }

                item = item1;
                np = np1;
                pbase = pb1;
                mine = nmine;
                item1 = item2;
                np1 = np2;
                pb1 = pb2;
            }
        } else {
            auto meta = [&](int it_, int &np_, int64_t &pb_) {
                np_ = 0;
                pb_ = 0;
                if (it_ < p.nitems) {
                    int h_, t_;
                    decode_item(p, it_, h_, t_);
                    np_ = p.npieces[h_];
                    pb_ = p.off[h_ * RG];
                }
            };
            int s = 0;
            unsigned ph = 0;
            int P = 0;
            int qs = 0, nq = 0;
            unsigned qph = 0;
            const unsigned xbytes = (unsigned)grows * 16u;
            // Both producer warps walk the same piece stream, but each issues alternating stages.
            auto issue_item = [&](int np, int64_t pbase, int c0, Piece &mine) {
                for (int q = 0; q < np; ++q, ++P) {
                    if (q > 0 && (q & 31) == 0) {
                        if (q + lane < np) mine = p.pieces[pbase + q + lane];
                        lat_check(q + lane < np ? mine.hi : 0, q + lane < np);
                    }
                    if (NPW > 1 && (P & 1) != warp) {
                        if (++s == nst) {
                            s = 0;
                            ph ^= 1u;
                        }
                        continue;
                    }
                    const int hi = __shfl_sync(0xffffffffu, mine.hi, q & 31);
                    const int S = __shfl_sync(0xffffffffu, mine.S, q & 31);
                    const int toff = __shfl_sync(0xffffffffu, mine.toff, q & 31);
                    const int nc = __shfl_sync(0xffffffffu, mine.nc, q & 31);
                    if (P >= nst) mbar_wait(EMPTY(s), ph ^ 1u);
                    unsigned char *st = smem + (size_t)s * p.stage_bytes;
                    const unsigned xa = smem_u32(st);
                    const unsigned ta = xa + (unsigned)p.xs_bytes;
                    const unsigned tbytes = (unsigned)((nc + (Z ? 0 : 1)) * MN * 16);
                    if (toff < 0) {
                        Piece pc;
                        pc.hi = hi;
                        pc.S = S;
                        pc.n = __shfl_sync(0xffffffffu, mine.n, q & 31);
                        pc.p1 = __shfl_sync(0xffffffffu, mine.p1, q & 31);
                        pc.n1 = __shfl_sync(0xffffffffu, mine.n1, q & 31);
                        pc.p2 = __shfl_sync(0xffffffffu, mine.p2, q & 31);
                        pc.jb = __shfl_sync(0xffffffffu, mine.jb, q & 31);
                        uint4 *ts = reinterpret_cast<uint4 *>(st + p.xs_bytes);
                        if (RG == 1) {
                            for (int e = lane; e < (nc + (Z ? 0 : 1)) * MN; e += 32) {
                                const int kc = e / MN;
                                const int n = e - kc * MN;
                                const int k = n / DW;
                                ts[e] = kc < nc ? toeplitz8(p.val, pc, 8 * kc, k, n - k * DW, ps, p.K, p.kpad) :
                                                  make_uint4(0, 0, 0, 0);
                            }
                        } else {
                            const PieceX px = p.pxs[pbase + q];
                            for (int e = lane; e < (nc + (Z ? 0 : 1)) * MN; e += 32) {
                                const int kc = e / MN;
                                const int n0 = e - kc * MN;
                                const int r = n0 / MN1;
                                const int n = n0 - r * MN1;
                                const int k = n / DW;
                                ts[e] = kc < nc ? toeplitz8x(p.val, pc.S, pc.n, pc.jb, px, r, 8 * kc, k, n - k * DW, ps,
                                                             p.K, p.kpad, p.Wi) :
                                                  make_uint4(0, 0, 0, 0);
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    }
                    if (toff < 0) __syncwarp();
                    const int B8 = S >> 3;
                    if (elect_one()) {
                        sks[s] = nc;
                        mbar_expect_tx(FULL(s), xbytes + (toff >= 0 ? tbytes : 0u));
                        if (toff >= 0) bulk_load_reuse(ta, p.T + toff, tbytes, FULL(s), tpolicy);
                        if (TX) {
                            int q0 = B8 - p.hrot;
                            if (q0 < 0) q0 += p.nb;
                            tma_load_4d(xa, &tmap_x.m[q0 & ((1 << p.nvs) - 1)], 0, q0 >> p.nvs, hi, c0, FULL(s));
                        }
                    }
                    if (!TX) {
                        for (int cl = lane; cl < p.G; cl += 32) {
                            const int c = c0 + cl;
                            int ch = B8;
                            const uint16_t *src = p.x + ((size_t)(c < p.BC ? c : p.BC - 1) * p.Hi + hi) * p.Wi;
                            unsigned d = xa + (unsigned)(cl * p.RPC) * 16u;
                            int rem = p.RPC;
                            while (rem > 0) {
                                const int l = min(rem, p.nb - ch);
                                bulk_load(d, src + 8 * ch, (unsigned)l * 16u, FULL(s));
                                d += (unsigned)l * 16u;
                                rem -= l;
                                ch = 0;
                            }
                        }
                    }
                    if (!TX) __syncwarp();
                    if (++s == nst) {
                        s = 0;
                        ph ^= 1u;
                    }
                }
            };
            if (NPW > 1 && warp == 1) {
                for (;;) {
                    mbar_wait(QFULL(qs), qph);
                    const int2 e = qbuf[qs];
                    const int64_t pbase = qpb[qs];
                    __syncwarp();
                    if (lane == 0) mbar_arrive(QEMPTY(qs));
                    if (++qs == QN) {
                        qs = 0;
                        qph ^= 1u;
                    }
                    const int item = e.x;
                    if (item >= p.nitems) break;
                    const int np = e.y;
                    int ho, cg;
                    decode_item(p, item, ho, cg);
                    Piece mine;
                    if (lane < np) mine = p.pieces[pbase + lane];
                    lat_check(lane < np ? mine.hi : 0, lane < np);
                    issue_item(np, pbase, cg * p.G, mine);
                }
            } else {
                int ticket_next = 0, ticket_left = 0;
                auto claim = [&]() {
                    if (ticket_left == 0) {
                        int base = 0;
                        if (lane == 0) base = atomicAdd(p.ictr, p.ticket);
                        ticket_next = __shfl_sync(0xffffffffu, base, 0);
                        ticket_left = p.ticket;
                    }
                    --ticket_left;
                    return ticket_next++;
                };
                // Item pipeline: while item i is issued, item i+1's pieces, item i+2's row metadata and the
                // claim of item i+3 are in flight (no dependent global round trips at item boundaries).
                int item = (int)blockIdx.x;
                int item1 = claim();
                int np, np1;
                int64_t pbase, pb1;
                meta(item, np, pbase);
                meta(item1, np1, pb1);
                Piece mine;
                if (lane < np) mine = p.pieces[pbase + lane];
                int fut = claim();
                while (true) {
                    // publish the item to the MMA, epilogue and second-producer warps
                    if (nq >= QN) mbar_wait(QEMPTY(qs), qph ^ 1u);
                    if (lane == 0) {
                        qbuf[qs] = make_int2(item, np);
                        if (NPW > 1) qpb[qs] = pbase;
                        mbar_arrive(QFULL(qs));
                    }
                    __syncwarp();
                    if (++qs == QN) {
                        qs = 0;
                        qph ^= 1u;
                    }
                    ++nq;
                    if (item >= p.nitems) break;
                    lat_check(lane < np ? mine.hi : 0, lane < np);
                    int ho, cg;
                    decode_item(p, item, ho, cg);
                    const int c0 = cg * p.G;
                    Piece nmine;
                    if (lane < np1) nmine = p.pieces[pb1 + lane];
                    const int item2 = fut;
                    int np2;
                    int64_t pb2;
                    meta(item2, np2, pb2);
                    fut = claim();
                    issue_item(np, pbase, c0, mine);
                    item = item1;
                    np = np1;
                    pbase = pb1;
                    mine = nmine;
                    item1 = item2;
                    np1 = np2;
                    pb1 = pb2;
                }
            }
        }
    } else if (warp >= 4) {
        // ---------------- consumer warpgroup 1 + m: M tile m ----------------
        // Every warp waits for each stage and releases it (one arrival per warp) once the wgmma group that read
        // it has completed; one piece's group stays in flight behind the next one's issue.
        using W = Wgmma<MN>;
        constexpr int R = W::R;        // accumulator registers per m64 half
        const int m = (warp >> 2) - 1; // this warpgroup's M tile
        const int wq = warp & 3;       // its warp: rows 16 wq .. 16 wq + 15 of each m64 half
        const bool bf = p.bf16 != 0;
        const unsigned z16 = (smem_u32(smem) + (unsigned)p.z_off) >> 4;
        const unsigned off = (unsigned)(m * 128); // the tile's first slot
        // Output offsets: 64-bit to the channel, 32-bit within it (the host checks K * Ho * Wo < 2^32), so the
        // epilogue keeps 32-bit address arithmetic while the whole output may exceed 2^32 elements.
        const unsigned kstride = (unsigned)p.Ho * (unsigned)p.Wo;
        float acc[2][R];
        int s = 0;
        unsigned ph = 0;
        int qs = 0;
        unsigned qph = 0;
        for (int it = 0;; ++it) {
            mbar_wait(QFULL(qs), qph);
            const int2 e = qbuf[qs];
            __syncwarp();
            if (lane == 0) mbar_arrive(QEMPTY(qs));
            if (++qs == QN) {
                qs = 0;
                qph ^= 1u;
            }
            const int item = e.x;
            if (item >= p.nitems) break;
            const int np = e.y;
            int ho, cg;
            decode_item(p, item, ho, cg);
            const int c0 = cg * p.G;
            const int nch = min(p.G, p.BC - c0);
            // The last channel's halo rows are never stored, so exclude them from the tile count.
            const int nrow = (nch - 1) * p.RPC + p.nb;
            const int mtc = (nrow + 127) >> 7;
            // m64 halves of this tile that hold stored rows
            const int nh = m < mtc ? (nrow - 128 * m > 64 ? 2 : 1) : 0;

            // the tile's MMAs of one k-step: A from slot start a16 (K-direction stride alb), B from descriptor bd
            bool init = false; // whether the accumulators hold this item's partial sums
            auto mma = [&](unsigned a16, unsigned alb, uint64_t bd) {
                const int sc = init ? 1 : 0;
                W::mma(acc[0], mkdesc(a16, alb), bd, sc, bf);
                if (nh > 1) W::mma(acc[1], mkdesc(a16 + 64u, alb), bd, sc, bf);
                init = true;
            };
            // Chunk-packed issue: a k-step covers two 8-longitude chunks.  A piece's odd last chunk is
            // paired with the next piece's first chunk (both operands via an LBO spanning the two
            // stages); a ring wrap reverses both K halves, which leaves A*B unchanged and keeps the LBO positive.
            int pend = -1;            // stage whose last chunk is still pending
            unsigned pa = 0, pb = 0;  // its A / B start (16-byte units)
            int rel0 = -1, rel1 = -1; // the stages the previous piece's wgmma group frees
            for (int q = 0; q < np; ++q) {
                mbar_wait(FULL(s), ph); // TMA-written operands: the mbarrier completion orders them
                const int nc = sks[s];
                const unsigned xs16 = (smem_u32(smem) + (unsigned)(s * p.stage_bytes)) >> 4;
                const unsigned ts16 = xs16 + (unsigned)(p.xs_bytes >> 4);
                auto bdesc = [&](unsigned c) -> uint64_t {
                    const unsigned a = ts16 + c * (unsigned)MN;
                    if constexpr (Z)
                        return (int)c + 1 == nc ? mkdesc(a, z16 - a) : mkdesc(a, (unsigned)MN);
                    else
                        return mkdesc(a, (unsigned)MN);
                };
                const bool comb = pend >= 0;
                const int cs = comb ? 1 : 0;
                const int rem = nc - cs;
                // On the first piece, keep at least one pair so D is initialized before the cross-piece MMA
                // accumulates into it.
                const bool carry = (rem & 1) && (q > 0 || nc > 1) && (q + 1 < np);
                const int npair = carry ? (rem >> 1) : ((rem + 1) >> 1);
                if (nh > 0) {
                    wgmma_fence();
                    if (comb) {
                        const bool fwd = ts16 > pb;
                        const uint64_t bd = fwd ? mkdesc(pb, ts16 - pb) : mkdesc(ts16, pb - ts16);
                        if (fwd)
                            mma(pa + off, xs16 - pa, bd);
                        else
                            mma(xs16 + off, pa - xs16, bd);
                    }
                    for (int j = 0; j < npair; ++j) {
                        const unsigned c = (unsigned)(cs + 2 * j);
                        mma(xs16 + c + off, 1u, bdesc(c));
                    }
                }
                wgmma_commit();
                const int r0 = comb ? pend : -1;
                const int r1 = carry ? -1 : s;
                if (carry) {
                    pend = s;
                    pa = xs16 + (unsigned)(nc - 1);
                    pb = ts16 + (unsigned)((nc - 1) * MN);
                } else {
                    pend = -1;
                }
                // the previous piece's group has completed: its stages are free
                wgmma_wait<1>();
                if (lane == 0) {
                    if (rel0 >= 0) mbar_arrive(EMPTY(rel0));
                    if (rel1 >= 0) mbar_arrive(EMPTY(rel1));
                }
                rel0 = r0;
                rel1 = r1;
                if (++s == nst) {
                    s = 0;
                    ph ^= 1u;
                }
            }
            wgmma_wait<0>();
            if (lane == 0) {
                if (rel0 >= 0) mbar_arrive(EMPTY(rel0));
                if (rel1 >= 0) mbar_arrive(EMPTY(rel1));
            }
            fence_acc(acc[0]);
            fence_acc(acc[1]);
            if (!init) {
                // no pieces: the outputs are zero
#pragma unroll
                for (int i = 0; i < R; ++i) acc[0][i] = acc[1][i] = 0.f;
            }

            // ---------------- epilogue: registers -> fp16 / bf16 -> global ----------------
            // half-block shift: GEMM row b holds outputs DW b + DW / 2 + dwo (mod Wo)
            const int sh = (DW >= 2 && p.rowsig[ho] != 0) ? DW / 2 : 0;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                if (h >= nh) break;
#pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    const int g = m * 128 + h * 64 + wq * 16 + (lane >> 2) + 8 * sub; // GEMM row
                    const int cl = p.G == 2 ? (int)(g >= p.RPC) : g / p.RPC;
                    const int b = g - cl * p.RPC;
                    if (cl >= nch || b >= p.nb) continue;
                    uint16_t *chan = p.out + (size_t)(c0 + cl) * ((size_t)p.K * kstride);
#pragma unroll
                    for (int j = 0; j < MN / 8; ++j) {
                        const float v0 = acc[h][4 * j + 2 * sub], v1 = acc[h][4 * j + 2 * sub + 1];
                        const int n = 8 * j + 2 * (lane & 3); // the pair's first column
                        if constexpr (DW >= 2) {
                            // DW even: the pair lies in one (row r, basis k) block
                            const int r = n / MN1;
                            if (r >= RG) continue; // padding columns
                            const int rn = n - r * MN1;
                            const int k = rn / DW;
                            const int hr = ho * RG + r;
                            if (k >= p.K || hr >= p.Ho) continue;
                            int w = DW * b + sh + (rn - k * DW);
                            if (w >= p.Wo) w -= p.Wo;
                            uint16_t *dst = chan + ((unsigned)k * kstride + (unsigned)hr * (unsigned)p.Wo);
                            const unsigned pk = pack2(v0, v1, p.bf16);
                            if (DW == 2 && sh != 0) {
                                // odd start: the pair straddles a 4-byte boundary, and the seam for the last block
                                dst[w] = (uint16_t)(pk & 0xFFFFu);
                                dst[w + 1 == p.Wo ? 0 : w + 1] = (uint16_t)(pk >> 16);
                            } else {
                                *reinterpret_cast<unsigned *>(dst + w) = pk;
                            }
                        } else {
                            // DW 1: one column per (row, basis function)
                            const unsigned pk = pack2(v0, v1, p.bf16);
#pragma unroll
                            for (int t = 0; t < 2; ++t) {
                                const int r = (n + t) / MN1;
                                if (r >= RG) continue;
                                const int k = n + t - r * MN1;
                                const int hr = ho * RG + r;
                                if (k >= p.K || hr >= p.Ho) continue;
                                chan[(unsigned)k * kstride + (unsigned)hr * (unsigned)p.Wo + (unsigned)b]
                                    = (uint16_t)(t ? (pk >> 16) : (pk & 0xFFFFu));
                            }
                        }
                    }
                }
            }
        }
    }
    // warps NPW .. 3 of warpgroup 0 have no role
}

template <int DW, bool TX, int MT, int CPS, int RG, bool LP, int NV, bool Z>
__global__ void __launch_bounds__(HkCfg<DW, MT, CPS, RG>::THREADS, CPS)
    disco_hk_wg(const __grid_constant__ TmapX<NV> tmap_x, const HkParams p)
{
    hk_body<DW, TX, MT, CPS, RG, LP, NV, Z>(tmap_x, p);
}
// the main-kernel entry of a configuration
template <int DW, bool TX, int MT, int CPS, int RG, bool LP = false, int NV = NVMAX, bool Z = false>
static constexpr auto disco_hk_kernel()
{
    return &disco_hk_wg<DW, TX, MT, CPS, RG, LP, NV, Z>;
}

// =====================================================================================================================
// Host (torch-harmonics)
//
// The delivered host side (one call: preprocess psi, extend the input, run the main kernel, all on scratch from a
// process-wide workspace cache told apart by an epoch) is replaced by two ops:
//
// forward_hankel_sm90_plan
//     Everything the forward derives from psi for B * C planes of a dtype: the filter values in that dtype, the
//     pieces (flag pass and row / row-group planner), the Toeplitz tiles and the latitude ranges of the extended
//     copy. A byte buffer, built once per key by the plan of the DISCO backend and kept (disco_plan.py).
// forward_hankel_sm90_run
//     The forward through such a plan: per-call stamps and the extended input copy from torch's allocator on the
//     current stream, zeroed / published by init_kernel, so the call is safe to capture in a CUDA graph and to
//     replay; then the input extension, the main kernel and the co-resident extension, chained by programmatic
//     dependent launch as delivered.
//
// The launch configuration is a function of the sizes and the device alone (make_config): the plan and the run
// compute the same one. Sizes the tensor-core path does not take (pscale 3 or above 8, kernel_size above 9,
// nlon_out not a multiple of 8, ...) run the shipped kpacked forward instead, as does a plan of zero bytes.
// =====================================================================================================================

#if (defined(DISCO_TARGET_SM100A) && DISCO_TARGET_SM100A) || (defined(DISCO_TARGET_SM103A) && DISCO_TARGET_SM103A)
#error "disco_cuda_fwd_hankel_sm90.cu is compiled for sm_90a only, in torch_harmonics.disco._C_sm90"
#endif

namespace
{

    constexpr int STAMP_WORDS = STAMP_WORDS_BYTES / 4;
    constexpr size_t MAIN_SMEM_ATTR = 227 * 1024;

    inline size_t align256(size_t v) { return ((v + 255) / 256) * 256; }

    // Piece length: the longest of {128, 96, 80, 64, 48, 32} points whose halo lets the channel groups
    // fill >= 95% of the M tiles (best fill otherwise).  Longer pieces waste less window per point.
    // Returns the channel group size G (0 if none fits) and the chosen pmax / k-step bound. (As delivered.)
    int choose_group(int nb, int ps, int MT, int BC, int &pmax, int &kstep_max, bool *capped = nullptr)
    {
        const int cands[6] = {128, 96, 80, 64, 48, 32};
        double best = -1.0;
        pmax = 0;
        kstep_max = 0;
        for (int ci = 0; ci < 6; ++ci) {
            const int nc = (7 + cands[ci] + 8 - ps + 7) / 8;
            const int ksm = (nc + 1) / 2;
            const int rpc = (nb + 2 * ksm - 1 + 7) & ~7;
            const int g = (128 * MT) / rpc;
            if (g < 1) continue;
            const double fill = (double)(g * rpc) / (128.0 * MT);
            if (fill >= 0.95) {
                pmax = cands[ci];
                kstep_max = ksm;
                break;
            }
            if (fill > best + 0.02) {
                best = fill;
                pmax = cands[ci];
                kstep_max = ksm;
            }
        }
        if (pmax == 0) return 0;
        int G = (128 * MT) / ((nb + 2 * kstep_max - 1 + 7) & ~7);
        // keep at least ~6 channel groups per row when channels are few, so that the long polar rows
        // are spread over several items (dynamic scheduling then balances them)
        const int gsmall = (BC + 5) / 6;
        if (capped) *capped = G > gsmall;
        if (G > gsmall) G = gsmall;
        return G;
    }

    // which main kernel of a family <DW, MT, CPS, RG> runs
    enum MainVariant : int { MV_LP_NV = 0, MV_LP_8 = 1, MV_TX_NV = 2, MV_TX_8 = 3, MV_BULK = 4 };

    struct HkConfig {
        bool ok = false;
        // sizes
        int BC = 0, Hi = 0, Wi = 0, Ho = 0, Wo = 0, K = 0, kpad = 0, ps = 0;
        int64_t NNZ = 0;
        int nsm = 0;
        // main kernel family and variant
        int DW = 0, CPS = 0, RG = 0;
        int MT = 0;
        int variant = MV_BULK;
        bool tx = false, shared_z = false, pdl = false, co = false, use_R = false;
        // launch parameters (HkParams and the preprocessing)
        int nb = 0, HB = 0, RPC = 0, G = 0, NCG = 0, NG = 0, nitems = 0, ticket = 1, nvs = 3;
        unsigned ncg_recip = 0;
        int nst = 0, xs_bytes = 0, stage_bytes = 0, bar_off = 0, z_off = 0;
        int nc_cap = 0, nunit = 0, grid = 0, TY = 0, tpl = 0, xpre = 0, xres = 0, hrot = 0, Lc = 0, threads = 0;
        long long cap = 0;
        size_t smem = 0;
        // plan layout (bytes)
        size_t o_val = 0, o_pieces = 0, o_sub = 0, o_np = 0, o_flags = 0, o_cnt = 0, o_T = 0, o_flat = 0, o_arcs = 0,
               o_pxs = 0, o_sig = 0, o_R = 0, plan_bytes = 0;
        // per-call scratch: the stamp words, then the extended input copy
        size_t xe_bytes = 0, scratch_bytes = 0;
    };

    // The main kernel of a configuration (as the delivered launch_hk2 chose it).
    template <int DW, int MT, int CPS, int RG> const void *main_kernel(const HkConfig &c)
    {
        constexpr bool ZCFG = CPS != 3;
        switch (c.variant) {
        case MV_LP_NV:
        case MV_LP_8:
            if constexpr (RG == 1) {
                if (c.variant == MV_LP_NV)
                    return (const void *)disco_hk_kernel<DW, true, MT, CPS, RG, true, NVMAX, ZCFG>();
                return (const void *)disco_hk_kernel<DW, true, MT, CPS, RG, true, 8, ZCFG>();
            }
            return nullptr;
        case MV_TX_NV: return (const void *)disco_hk_kernel<DW, true, MT, CPS, RG, false, NVMAX, ZCFG>();
        case MV_TX_8: return (const void *)disco_hk_kernel<DW, true, MT, CPS, RG, false, 8, ZCFG>();
        default: return (const void *)disco_hk_kernel<DW, false, MT, CPS, RG, false, 8>();
        }
    }

    // Can one extension block share an SM with CPS main-kernel CTAs?  (Only a performance question:
    // the extension never waits for the main kernel, so without room the main CTAs simply start later.)
    // As delivered, with the device's limits from torch's cached properties.
    bool co_fits(const void *kfn, int nth, int cps, size_t smem, const cudaDeviceProp &prop)
    {
        cudaFuncAttributes fa, fx;
        C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, kfn));
        C10_CUDA_CHECK(cudaFuncGetAttributes(&fx, (const void *)xcopy_kernel));
        const int smem_sm = (int)prop.sharedMemPerMultiprocessor, regs_sm = prop.regsPerMultiprocessor;
        const int thr_sm = prop.maxThreadsPerMultiProcessor, res = (int)prop.reservedSharedMemPerBlock;
        // registers are allocated per warp (multiples of 256) from the four sub-partitions' files, the
        // warps of a CTA spread across them
        auto wregs = [](int r) { return ((r * 32 + 255) / 256) * 256; };
        const int mw = (nth + 31) / 32, xw = (XC_THREADS + 31) / 32;
        const long long regs = 4LL
            * ((long long)cps * ((mw + 3) / 4) * wregs(fa.numRegs) + (long long)((xw + 3) / 4) * wregs(fx.numRegs));
        const long long sm
            = (long long)cps * ((long long)smem + fa.sharedSizeBytes + res) + (long long)fx.sharedSizeBytes + res;
        return regs <= regs_sm && sm <= smem_sm && cps * nth + XC_THREADS <= thr_sm;
    }

    // The configuration of the family <DW, MT, CPS, RG> (the delivered launch_hk2 up to its launches). Leaves
    // c.ok false where the delivered kernel would have fallen back.
    template <int DW, int MT_, int CPS, int RG> void configure(HkConfig &c, const cudaDeviceProp &prop)
    {
        using Cfg = HkCfg<DW, MT_, CPS, RG>;
        constexpr int MN = Cfg::MN;
        constexpr int MT = Cfg::MT;
        const int BC = c.BC, Hi = c.Hi, Wi = c.Wi, Ho = c.Ho, ps = c.ps;
        const int64_t NNZ = c.NNZ;
        c.DW = DW;
        c.MT = MT;
        c.CPS = CPS;
        c.RG = RG;
        c.threads = Cfg::THREADS;
        c.nunit = 2 * MN; // 16-byte units per k-step tile
        // The paired-row planner can reserve up to three MN-wide records per input point.  Avoid
        // signed reservation wrap on unusually large but otherwise valid sparse layouts.
        if (3ull * (uint64_t)MN * (uint64_t)NNZ > (uint64_t)INT32_MAX) return;
        c.nb = Wi / 8;
        int pmax = 0, kstep_max = 0;
        c.G = choose_group(c.nb, ps, MT, BC, pmax, kstep_max);
        if (c.G < 1) return;
        // The deep-ring trim needs kstep_max >= 4 so that pieces keep nc_cap >= 2 chunks: with
        // nc_cap >= 2 a run of L >= 1 points emits at most ceil(((L + 21) / 8) / 2) <= L pieces, which
        // bounds the planner's Piece / PieceX / flat arrays by NNZ.
        const bool deep_wide_ps1 = CPS == 1 && RG == 2 && ps == 1 && c.nb >= 128 && kstep_max >= 4;
        c.HB = 2 * kstep_max - 1 - (deep_wide_ps1 ? 5 : 0);
        c.RPC = (c.nb + c.HB + 7) & ~7; // per-channel slot stride (whole 128-byte TMA rows)
        // One-CTA row pairs trade two terminal chunks for a seventh producer/MMA stage.
        c.nc_cap = 2 * kstep_max - ((CPS == 1 && RG == 2 && kstep_max >= 2) ? (deep_wide_ps1 ? 5 : 2) : 0);
        if (c.nc_cap < 2 || c.HB < c.nc_cap - 1) return; // planner capacity / halo invariants
        c.NCG = (BC + c.G - 1) / c.G;
        c.ncg_recip = c.NCG == 1 ? UINT32_MAX : (unsigned)((1ull << 32) / (unsigned)c.NCG);
        c.NG = (Ho + RG - 1) / RG;
        const long long nitems64 = (long long)c.NG * c.NCG;
        if (nitems64 > INT32_MAX) return;
        c.nitems = (int)nitems64;
        c.tx = c.RPC / 8 <= 256 && c.G <= 256;
        // 512-byte TMA rows when the channel pitch is a whole number of them, else 128-byte rows
        c.nvs = (c.RPC % 32 == 0) ? 5 : 3;
        const bool lp_ok = RG == 1 && c.tx && c.G >= 4 && c.RPC <= 128;
        constexpr bool ZCFG = CPS != 3;
        c.shared_z = c.tx && ZCFG;
        const int nv = 1 << c.nvs;
        c.Lc = ((c.nb + nv - 1) & ~(nv - 1)) + c.RPC; // minimal whole-TMA-row extension

        const size_t nn = (size_t)(NNZ > 0 ? NNZ : 1);
        c.cap = (long long)c.nunit * ((long long)nn / 8 + (long long)Ho * 64 + 4096);
        if (c.cap > INT32_MAX) return; // Piece::toff and the planner's allocation counter are int32
        const uint64_t xrows = (uint64_t)BC * (uint64_t)Hi;
        if (c.tx && xrows > SIZE_MAX / ((size_t)c.Lc * 16)) return;

        c.nsm = prop.multiProcessorCount;
        c.grid = c.nitems < CPS * c.nsm ? c.nitems : CPS * c.nsm;
        const size_t limit = (CPS == 1 ? 227 : (CPS == 2 ? 113 : (CPS == 3 ? 75 : 56))) * 1024;
        c.xs_bytes = (((128 * MT + c.HB) * 16 + 127) / 128) * 128;
        const int ts_bytes = (c.nc_cap + (c.shared_z ? 0 : 1)) * MN * 16;
        c.stage_bytes = ((c.xs_bytes + ts_bytes + 127) / 128) * 128;
        const int bar_bytes = 1024;
        const int z_bytes = c.shared_z ? ((MN * 16 + 127) / 128) * 128 : 0;
        int nst = (int)((limit - bar_bytes - z_bytes) / c.stage_bytes);
        if (nst > 16) nst = 16;
        if (nst < 2) return;
        c.nst = nst;
        c.bar_off = nst * c.stage_bytes;
        c.z_off = c.bar_off + bar_bytes;
        c.smem = (size_t)c.bar_off + bar_bytes + z_bytes;
        c.ticket = c.nitems >= 64 * c.grid ? 4 : 1;

        bool lp = false;
        if constexpr (RG == 1) lp = CPS != 1 && lp_ok;
        if (lp)
            c.variant = c.nvs == 5 ? MV_LP_NV : MV_LP_8;
        else if (c.tx && c.nvs == 5)
            c.variant = MV_TX_NV;
        else if (c.tx)
            c.variant = MV_TX_8;
        else
            c.variant = MV_BULK;

        // Overlapping the planner, the Toeplitz build and the main kernel's prologue with their
        // predecessors pays on the regular paths; the three-CTA small-channel path runs strictly serialized.
        c.pdl = CPS != 3;
        // Co-resident extension: the longitude-extended input is written by a separate kernel that runs
        // next to the main kernel instead of in front of it.
        c.co = c.pdl && c.tx && Hi <= CO_MAXHI
            && co_fits(main_kernel<DW, MT, CPS, RG>(c), Cfg::THREADS, CPS, c.smem, prop);
        c.TY = CPS == 3 ? 8 : 6;
        c.tpl = (BC + XC_RPT - 1) / XC_RPT;
        // co-resident mode: the input extension ahead of the main kernel covers the first fifth of the latitude
        // order (the polar rows the first items need), the extension kernel the rest while the main kernel runs
        // (half when the support is sparse relative to the input grid)
        const double xdens = (double)NNZ * (double)c.Wo / ((double)Hi * (double)Wi);
        c.xpre = c.co ? (xdens < 256.0 ? (Hi + 1) / 2 : (Hi + 4) / 5) : 0;
        c.xres = c.co ? std::min(std::min(c.nsm, XC_MAXRES), (Hi - c.xpre) * c.tpl) : 0;
        c.hrot = c.tx ? c.nb / 2 : 0;
        // extended copy restricted to the windows' latitude ranges (rotated by half a turn)
        c.use_R = c.tx && c.pdl && Hi <= CO_MAXHI;

        // the plan
        size_t o = 0;
        auto put = [&](size_t bytes) {
            const size_t at = o;
            o += align256(bytes);
            return at;
        };
        c.o_val = put((size_t)NNZ * c.kpad * 2);
        c.o_pieces = put(nn * sizeof(Piece));
        c.o_sub = put(nn * 4);
        c.o_np = put((size_t)Ho * 4);
        c.o_flags = put(nn);
        c.o_cnt = put(256);
        c.o_T = put((size_t)c.cap * 16);
        c.o_flat = put(nn * 4);
        c.o_arcs = put(RG > 1 ? nn * sizeof(Arc) : 0);
        c.o_pxs = put(RG > 1 ? nn * sizeof(PieceX) : 0);
        c.o_sig = put((size_t)Ho * 4);
        c.o_R = put(c.use_R ? (size_t)16 * Hi : 0);
        c.plan_bytes = o;
        c.xe_bytes = c.tx ? align256((size_t)BC * Hi * c.Lc * 16) : 0;
        c.scratch_bytes = (size_t)STAMP_WORDS_BYTES + c.xe_bytes;
        c.ok = true;
    }

    // The CTA shape of a (DW, RG): two CTAs per SM with one M tile each, which overlap one CTA's epilogue with the
    // other's MMAs (the accumulators are registers, not double-buffered), when the channels fill their items;
    // otherwise one CTA per SM with two M tiles, whose items cover twice the channels (as the sm_100a kernel's
    // choice between its two- and one-CTA families).
    template <int DW, int RG = 1> void configure_hk(HkConfig &c, const cudaDeviceProp &prop)
    {
        const int BC = c.BC, Wi = c.Wi, ps = c.ps;
        int pm, ksm;
        bool capped = true;
        const int g2 = choose_group(Wi / 8, ps, HkCfg<DW, 1, 2, RG>::MT, BC, pm, ksm, &capped);
        if (g2 >= 1 && !capped) return configure<DW, 1, 2, RG>(c, prop);
        if (choose_group(Wi / 8, ps, HkCfg<DW, 2, 1, RG>::MT, BC, pm, ksm) >= 1)
            return configure<DW, 2, 1, RG>(c, prop);
        return configure<DW, 1, 2, RG>(c, prop);
    }

    // The configuration for (BC, K, Hi, Wi) -> (BC, K, Ho, Wo) through NNZ points padded to kpad values: the delivered
    // Run's dispatch.
    HkConfig make_config(int64_t BC64, int64_t K64, int64_t kpad64, int64_t Hi64, int64_t Wi64, int64_t Ho64,
                         int64_t Wo64, int64_t NNZ, const cudaDeviceProp &prop)
    {
        HkConfig c;
        if (Ho64 <= 0 || BC64 <= 0 || Wo64 <= 0 || K64 <= 0) return c;
        const int64_t ps64 = Wi64 / Wo64;
        const bool fits_i32 = BC64 <= INT32_MAX && Hi64 <= INT32_MAX && Wi64 <= INT32_MAX && Ho64 <= INT32_MAX
            && Wo64 <= INT32_MAX && NNZ <= INT32_MAX;
        // offsets within one channel stay 32-bit (see the epilogue)
        const bool chan_i32 = (uint64_t)K64 * (uint64_t)Ho64 * (uint64_t)Wo64 <= (uint64_t)UINT32_MAX;
        const bool hk_ok = fits_i32 && chan_i32 && ps64 >= 1 && Wi64 == ps64 * Wo64 && Wo64 % 8 == 0 && 8 % ps64 == 0
            && K64 <= KB && kpad64 >= K64 && (kpad64 == 8 || kpad64 == 16) && NNZ > 0;
        if (!hk_ok) return c;
        c.BC = (int)BC64;
        c.Hi = (int)Hi64;
        c.Wi = (int)Wi64;
        c.Ho = (int)Ho64;
        c.Wo = (int)Wo64;
        c.K = (int)K64;
        c.kpad = (int)kpad64;
        c.ps = (int)ps64;
        c.NNZ = NNZ;
        const int BC = c.BC, Wi = c.Wi, Ho = c.Ho, ps = c.ps;
        const int nsm = prop.multiProcessorCount;
        if (ps == 1) {
            // single rows: the row pairs of the sm_100a kernel (MMA width 144) do not fit the registers
            configure_hk<8>(c, prop);
        } else if (ps == 2) {
            int pm, ksm;
            const int g2 = choose_group(Wi / 8, ps, HkCfg<4, 1, 2, 2>::MT, BC, pm, ksm);
            const long long items2 = g2 >= 1 ? (long long)((Ho + 1) / 2) * ((BC + g2 - 1) / g2) : 0;
            if (items2 >= 8LL * nsm)
                configure_hk<4, 2>(c, prop);
            else
                configure_hk<4, 1>(c, prop);
        } else if (ps == 4) {
            configure_hk<2>(c, prop);
        } else {
            configure_hk<1>(c, prop);
        }
        return c;
    }

    // f(std::integral_constant...) for the family of c: every family make_config can choose.
    template <class F> void with_family(const HkConfig &c, F &&f)
    {
#define DISCO_HK_FAMILY(DW_, MT_, CPS_, RG_)                                                                           \
    if (c.DW == DW_ && c.MT == MT_ && c.CPS == CPS_ && c.RG == RG_) return f.template operator()<DW_, MT_, CPS_, RG_>();
#define DISCO_HK_FAMILIES(DW_, RG_)                                                                                    \
    DISCO_HK_FAMILY(DW_, 1, 2, RG_)                                                                                    \
    DISCO_HK_FAMILY(DW_, 2, 1, RG_)
        DISCO_HK_FAMILIES(8, 1)
        DISCO_HK_FAMILIES(4, 1)
        DISCO_HK_FAMILIES(4, 2)
        DISCO_HK_FAMILIES(2, 1)
        DISCO_HK_FAMILIES(1, 1)
#undef DISCO_HK_FAMILIES
#undef DISCO_HK_FAMILY
        TORCH_CHECK(false, "forward_hankel_sm90: no main kernel for DW ", c.DW, ", MT ", c.MT, ", CPS ", c.CPS, ", RG ",
                    c.RG);
    }

    // launch with programmatic stream serialization when pdl is set (the kernel calls pdl_wait before
    // consuming the predecessor's results; without the attribute pdl_wait/pdl_trigger are no-ops)
    template <typename... KArgs, typename... Args>
    void launch_pdl(bool pdl, void (*k)(KArgs...), unsigned grid, unsigned block, size_t smem, cudaStream_t st,
                    Args... args)
    {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(grid);
        cfg.blockDim = dim3(block);
        cfg.dynamicSmemBytes = smem;
        cfg.stream = st;
        cudaLaunchAttribute a[1];
        a[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        a[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = a;
        cfg.numAttrs = pdl ? 1 : 0;
        C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k, args...));
    }

    // cuTensorMapEncodeTiled, from the driver the runtime uses (no link against libcuda)
    using EncodeTiledFn = CUresult (*)(CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *, const cuuint64_t *,
                                       const cuuint64_t *, const cuuint32_t *, const cuuint32_t *, CUtensorMapInterleave,
                                       CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

    EncodeTiledFn encode_tiled()
    {
        static const EncodeTiledFn fn = [] {
            void *p = nullptr;
            cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
#if CUDART_VERSION >= 12050
            C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q));
#else
            C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q));
#endif
            TORCH_CHECK(q == cudaDriverEntryPointSuccess && p != nullptr,
                        "forward_hankel_sm90: the driver has no cuTensorMapEncodeTiled");
            return reinterpret_cast<EncodeTiledFn>(p);
        }();
        return fn;
    }

    void check_plan_inputs(const at::Tensor &pack_idx, const at::Tensor &pack_val, const at::Tensor &pack_offset)
    {
        TORCH_CHECK(pack_idx.is_cuda(), "forward_hankel_sm90: psi must be on a CUDA device, got ", pack_idx.device());
        disco_kernels::check_same_device(pack_val, pack_idx, "pack_val");
        disco_kernels::check_same_device(pack_offset, pack_idx, "pack_offset");
        TORCH_CHECK(pack_idx.scalar_type() == at::kLong && pack_idx.dim() == 2 && pack_idx.size(1) == 2,
                    "pack_idx must be an int64 (nnz, 2) tensor, got ", pack_idx.scalar_type(), " of shape ",
                    pack_idx.sizes());
        TORCH_CHECK(pack_val.dim() == 2 && pack_val.size(0) == pack_idx.size(0)
                        && at::isFloatingType(pack_val.scalar_type()),
                    "pack_val must be a floating (nnz, K_pad) tensor with one row per pack_idx entry, got ",
                    pack_val.scalar_type(), " of shape ", pack_val.sizes());
        disco_kernels::check_index_vector(pack_offset, at::kLong, "pack_offset");
        disco_kernels::check_dense(pack_idx, "pack_idx");
        disco_kernels::check_dense(pack_val, "pack_val");
    }

    void check_device(int device)
    {
        const cudaDeviceProp *prop = at::cuda::getDeviceProperties(device);
        TORCH_CHECK(prop->major == 9 && prop->minor == 0,
                    "forward_hankel_sm90 requires a compute capability 9.0 device (Hopper); got ", prop->major, ".",
                    prop->minor);
    }

    // the Toeplitz builder of a family (plan)
    struct TfillLaunch {
        const HkConfig &c;
        cudaStream_t st;
        const uint16_t *val;
        const Piece *pieces;
        const PieceX *pxs;
        uint4 *T;
        const unsigned *counter;
        const int *flat;

        template <int RG, int DW, int KPAD, bool Z> void go() const
        {
            tfill_kernel<RG, DW, KPAD, Z><<<(unsigned)(c.nsm * c.TY), 128, 0, st>>>(val, pieces, pxs, c.Wi, c.ps, c.K,
                                                                                    T, c.nsm, c.TY, counter, flat);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        template <int DW, int MT, int CPS, int RG> void operator()() const
        {
            if (c.kpad == 16) {
                if (c.shared_z)
                    go<RG, DW, 16, true>();
                else
                    go<RG, DW, 16, false>();
            } else {
                if (c.shared_z)
                    go<RG, DW, 8, true>();
                else
                    go<RG, DW, 8, false>();
            }
        }
    };

    // the main kernel of a family (run)
    struct MainLaunch {
        const HkConfig &c;
        cudaStream_t st;
        TmapX<NVMAX> *tmap;
        HkParams *prm;

        template <int DW, int MT, int CPS, int RG> void operator()() const
        {
            const void *kfn = main_kernel<DW, MT, CPS, RG>(c);
            C10_CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)MAIN_SMEM_ATTR));
            cudaLaunchConfig_t cfg = {};
            cfg.gridDim = dim3((unsigned)c.grid);
            cfg.blockDim = dim3((unsigned)c.threads);
            cfg.dynamicSmemBytes = c.smem;
            cfg.stream = st;
            cudaLaunchAttribute a[1];
            a[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            a[0].val.programmaticStreamSerializationAllowed = 1;
            cfg.attrs = a;
            cfg.numAttrs = c.pdl ? 1 : 0;
            void *args[2] = {(void *)tmap, (void *)prm};
            C10_CUDA_CHECK(cudaLaunchKernelExC(&cfg, kfn, args));
        }
    };

} // namespace

namespace disco_kernels
{

    // the shipped wgmma kpacked forward (disco_cuda_fwd_dense_kpacked_sm90.cu): the sizes this kernel does not take
    torch::Tensor disco_cuda_fwd_kpacked_sm90(torch::Tensor inp, torch::Tensor pack_idx, torch::Tensor pack_val,
                                              torch::Tensor pack_offset, int64_t K, int64_t Ho, int64_t Wo);

    // The plan of the forward of BC planes of dtype (fp16 / bf16) on (Hi, Wi) through psi in the kpacked layout; zero
    // bytes where the shipped kpacked forward runs instead.
    at::Tensor forward_hankel_sm90_plan(at::Tensor pack_idx, at::Tensor pack_val, at::Tensor pack_offset, int64_t BC,
                                        int64_t K, int64_t Hi, int64_t Wi, int64_t Wo, at::ScalarType dtype)
    {
        check_plan_inputs(pack_idx, pack_val, pack_offset);
        TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "forward_hankel_sm90 takes fp16 or bf16, got ", dtype);
        const c10::cuda::CUDAGuard guard(pack_idx.device());
        check_device(pack_idx.get_device());
        const cudaDeviceProp &prop = *at::cuda::getDeviceProperties(pack_idx.get_device());
        const int64_t Ho = pack_offset.size(0) - 1;
        const HkConfig c = make_config(BC, K, pack_val.size(1), Hi, Wi, Ho, Wo, pack_idx.size(0), prop);
        auto plan = at::empty({c.ok ? (int64_t)c.plan_bytes : 0}, pack_idx.options().dtype(at::kByte));
        if (!c.ok) return plan;

        cudaStream_t st = at::cuda::getCurrentCUDAStream();
        unsigned char *base = static_cast<unsigned char *>(plan.data_ptr());
        uint16_t *val = reinterpret_cast<uint16_t *>(base + c.o_val);
        Piece *pieces = reinterpret_cast<Piece *>(base + c.o_pieces);
        int *substart = reinterpret_cast<int *>(base + c.o_sub);
        int *npieces = reinterpret_cast<int *>(base + c.o_np);
        unsigned char *flags = base + c.o_flags;
        unsigned *counter = reinterpret_cast<unsigned *>(base + c.o_cnt);
        uint4 *T = reinterpret_cast<uint4 *>(base + c.o_T);
        int *flat = reinterpret_cast<int *>(base + c.o_flat);
        Arc *arcs = reinterpret_cast<Arc *>(base + c.o_arcs);
        PieceX *pxs = reinterpret_cast<PieceX *>(base + c.o_pxs);
        int *rowsig = reinterpret_cast<int *>(base + c.o_sig);
        unsigned long long *R = c.use_R ? reinterpret_cast<unsigned long long *>(base + c.o_R) : nullptr;
        const int64_t *idx = pack_idx.data_ptr<int64_t>();
        const int64_t *off = pack_offset.data_ptr<int64_t>();

        // the filter values in the activations' dtype
        const at::Tensor vals = pack_val.to(dtype).contiguous();
        C10_CUDA_CHECK(cudaMemcpyAsync(val, vals.data_ptr(), (size_t)c.NNZ * c.kpad * 2, cudaMemcpyDeviceToDevice, st));

        const unsigned fgrid = (unsigned)((c.NNZ + 255) / 256);
        if (R) {
            // the latitude ranges of the extended copy (a call's epoch is 1, see init_kernel)
            C10_CUDA_CHECK(cudaMemsetAsync(R, 0, (size_t)16 * c.Hi, st));
            range_kernel<<<fgrid, 256, 0, st>>>(idx, c.NNZ, R, c.nb, c.hrot, c.nc_cap, c.RPC, 1u);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        flag_kernel<<<fgrid, 256, 0, st>>>(idx, c.NNZ, c.Wi, flags, counter);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (c.RG == 1)
            row_kernel<<<(unsigned)c.Ho, 256, 0, st>>>(idx, off, flags, c.Wi, c.ps, c.nc_cap, c.nunit, substart, pieces,
                                                       npieces, counter, c.cap, flat, rowsig);
        else
            group_kernel<<<(unsigned)c.NG, GT, 0, st>>>(idx, off, flags, c.Wi, c.ps, c.nc_cap, c.nunit, c.Ho, substart,
                                                        arcs, pieces, pxs, npieces, counter, c.cap, flat, rowsig);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // the Toeplitz tiles
        with_family(c, TfillLaunch {c, st, val, pieces, pxs, T, counter, flat});
        return plan;
    }

    // The forward of the contiguous fp16 / bf16 (B, C, Hi, Wi) inp to (B, C, K, Ho, Wo) through the plan built for
    // B*C planes of its dtype.
    at::Tensor forward_hankel_sm90_run(at::Tensor inp, at::Tensor plan, at::Tensor pack_idx, at::Tensor pack_val,
                                       at::Tensor pack_offset, int64_t K, int64_t Ho, int64_t Wo)
    {
        TORCH_CHECK(inp.is_cuda(), "inp must be a CUDA tensor, got ", inp.device());
        check_kpacked_inputs(inp, pack_idx, pack_val, pack_offset, K, Ho, Wo);
        check_same_device(plan, inp, "plan");
        const c10::cuda::CUDAGuard guard(inp.device());
        check_device(inp.get_device());
        const cudaDeviceProp &prop = *at::cuda::getDeviceProperties(inp.get_device());
        const int64_t B = inp.size(0), C = inp.size(1), Hi = inp.size(2), Wi = inp.size(3);
        const HkConfig c = make_config(B * C, K, pack_val.size(1), Hi, Wi, Ho, Wo, pack_idx.size(0), prop);
        if (!c.ok || plan.numel() == 0)
            return disco_cuda_fwd_kpacked_sm90(inp, pack_idx, pack_val, pack_offset, K, Ho, Wo);
        TORCH_CHECK(plan.scalar_type() == at::kByte && plan.dim() == 1 && (size_t)plan.numel() == c.plan_bytes,
                    "forward_hankel_sm90: the plan (", plan.numel(), " bytes) was not built for these sizes (",
                    c.plan_bytes, " bytes); build it with forward_hankel_sm90_plan for B * C = ", B * C);
        // the input rows are read in 16-byte chunks
        TORCH_CHECK(reinterpret_cast<uintptr_t>(inp.data_ptr()) % 16 == 0,
                    "forward_hankel_sm90: inp must be 16-byte aligned");

        auto out = at::empty({B, C, K, Ho, Wo}, inp.options());
        auto scratch = at::empty({(int64_t)c.scratch_bytes}, inp.options().dtype(at::kByte));
        cudaStream_t st = at::cuda::getCurrentCUDAStream();

        unsigned char *base = static_cast<unsigned char *>(plan.data_ptr());
        unsigned *stamps = static_cast<unsigned *>(scratch.data_ptr());
        uint4 *xe = c.tx ?
            reinterpret_cast<uint4 *>(static_cast<unsigned char *>(scratch.data_ptr()) + STAMP_WORDS_BYTES) :
            nullptr;
        const unsigned long long *R = c.use_R ? reinterpret_cast<const unsigned long long *>(base + c.o_R) : nullptr;
        const uint4 *x = reinterpret_cast<const uint4 *>(inp.data_ptr());
        const bool bf16 = inp.scalar_type() == at::kBFloat16;

        HkParams prm;
        memset(&prm, 0, sizeof(prm));
        prm.x = reinterpret_cast<const uint16_t *>(inp.data_ptr());
        prm.val = reinterpret_cast<const uint16_t *>(base + c.o_val);
        prm.pieces = reinterpret_cast<const Piece *>(base + c.o_pieces);
        prm.npieces = reinterpret_cast<const int *>(base + c.o_np);
        prm.off = pack_offset.data_ptr<int64_t>();
        prm.T = reinterpret_cast<const uint4 *>(base + c.o_T);
        prm.pxs = reinterpret_cast<const PieceX *>(base + c.o_pxs);
        prm.out = reinterpret_cast<uint16_t *>(out.data_ptr());
        prm.bf16 = bf16 ? 1 : 0;
        prm.BC = c.BC;
        prm.Hi = c.Hi;
        prm.Wi = c.Wi;
        prm.Ho = c.Ho;
        prm.Wo = c.Wo;
        prm.ps = c.ps;
        prm.K = c.K;
        prm.kpad = c.kpad;
        prm.nb = c.nb;
        prm.RPC = c.RPC;
        prm.HB = c.HB;
        prm.G = c.G;
        prm.NCG = c.NCG;
        prm.nitems = c.nitems;
        prm.NG = c.NG;
        prm.ticket = c.ticket;
        prm.ncg_recip = c.ncg_recip;
        prm.nvs = c.nvs;
        prm.nst = c.nst;
        prm.xs_bytes = c.xs_bytes;
        prm.stage_bytes = c.stage_bytes;
        prm.bar_off = c.bar_off;
        prm.z_off = c.z_off;
        prm.ictr = reinterpret_cast<int *>(stamps + ST_ICTR);
        prm.rowsig = reinterpret_cast<const int *>(base + c.o_sig);
        prm.stamps = stamps;
        prm.epoch = 1u;
        prm.co = c.co ? 1 : 0;
        prm.xpre = c.xpre;
        prm.xres = c.xres;
        prm.xe = xe;
        prm.Lc = c.Lc;
        prm.tpl = c.tpl;
        prm.R = R;
        prm.hrot = c.hrot;

        // the call's stamps, claims and item counter
        init_kernel<<<1, INIT_THREADS, 0, st>>>(stamps, STAMP_WORDS, c.xres, c.grid);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // the extended input ahead of the main kernel: every row, or in co-resident mode the first xpre latitudes
        if (c.tx) {
            const long long nxrows = (long long)c.BC * (c.co ? c.xpre : c.Hi);
            const long long nxb = (nxrows + 7) / 8;
            if (nxb > 0) {
                TORCH_CHECK(nxb <= INT32_MAX, "forward_hankel_sm90: input too large");
                xpre_kernel<<<(unsigned)nxb, 128, 0, st>>>(x, xe, c.nb, c.Lc, nxrows, c.Hi, c.co ? c.xpre : 0, R, c.hrot);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        }

        // the views of the extended copy for the main kernel's TMA loads
        TmapX<NVMAX> tmap;
        memset(&tmap, 0, sizeof(tmap));
        if (c.tx) {
            const int nv = 1 << c.nvs;
            const cuuint64_t gstr[3] = {(cuuint64_t)(16 * nv), (cuuint64_t)c.Lc * 16, (cuuint64_t)c.Hi * c.Lc * 16};
            const cuuint32_t box[4] = {(cuuint32_t)(8 * nv), (cuuint32_t)(c.RPC / nv), 1, (cuuint32_t)c.G};
            const cuuint32_t es[4] = {1, 1, 1, 1};
            const CUtensorMapDataType dt = bf16 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
            for (int j = 0; j < nv; ++j) {
                cuuint64_t gdim[4] = {(cuuint64_t)(8 * nv), (cuuint64_t)(c.Lc / nv - (j > 0 ? 1 : 0)), (cuuint64_t)c.Hi,
                                      (cuuint64_t)c.BC};
                const CUresult r = encode_tiled()(&tmap.m[j], dt, 4, xe + j, gdim, gstr, box, es,
                                                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
                TORCH_CHECK(r == CUDA_SUCCESS, "forward_hankel_sm90: cuTensorMapEncodeTiled failed (", (int)r, ")");
            }
        }

        // the main kernel; a kernel with 8 views reads the first 8 maps (TmapX<8> is a prefix of TmapX<NVMAX>)
        with_family(c, MainLaunch {c, st, &tmap, &prm});

        if (c.co) {
            // the extension blocks must leave each SM configured for the main kernel's shared memory
            C10_CUDA_CHECK(cudaFuncSetAttribute((const void *)xcopy_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                                (int)cudaSharedmemCarveoutMaxShared));
            launch_pdl(true, xcopy_kernel, (unsigned)c.nsm, (unsigned)XC_THREADS, 0, st, x, xe, c.nb, c.Lc, c.Hi, c.BC,
                       c.tpl, c.xpre, stamps, 1u, R, c.hrot, c.xres);
        }
        return out;
    }

    // The configuration the forward takes for these sizes on the current device, for tests:
    // [ok, DW, MT, CPS, RG, variant, co, plan_bytes, scratch_bytes]
    std::vector<int64_t> forward_hankel_sm90_config(int64_t BC, int64_t K, int64_t kpad, int64_t Hi, int64_t Wi,
                                                    int64_t Ho, int64_t Wo, int64_t nnz)
    {
        const cudaDeviceProp &prop = *at::cuda::getCurrentDeviceProperties();
        const HkConfig c = make_config(BC, K, kpad, Hi, Wi, Ho, Wo, nnz, prop);
        return {c.ok, c.DW, c.MT, c.CPS, c.RG, c.variant, c.co, (int64_t)c.plan_bytes, (int64_t)c.scratch_bytes};
    }

    TORCH_LIBRARY_FRAGMENT(disco_kernels, m)
    {
        m.def("forward_hankel_sm90_plan(Tensor pack_idx, Tensor pack_val, Tensor pack_offset, int bc, int kernel_size, "
              "int nlat_in, int nlon_in, int nlon_out, ScalarType dtype) -> Tensor");
        m.def("forward_hankel_sm90_run(Tensor inp, Tensor plan, Tensor pack_idx, Tensor pack_val, Tensor pack_offset, "
              "int kernel_size, int nlat_out, int nlon_out) -> Tensor",
              {at::Tag::pt2_compliant_tag});
        m.def("forward_hankel_sm90_config(int bc, int kernel_size, int kpad, int nlat_in, int nlon_in, int nlat_out, "
              "int nlon_out, int nnz) -> int[]");
    }

    TORCH_LIBRARY_IMPL(disco_kernels, CUDA, m)
    {
        m.impl("forward_hankel_sm90_plan", &forward_hankel_sm90_plan);
        m.impl("forward_hankel_sm90_run", &forward_hankel_sm90_run);
    }

    TORCH_LIBRARY_IMPL(disco_kernels, CompositeExplicitAutograd, m)
    {
        m.impl("forward_hankel_sm90_config", &forward_hankel_sm90_config);
    }

} // namespace disco_kernels
