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

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <string.h>
#include <mutex>
#include <vector>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

namespace ffi = tvm::ffi;

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
__global__ void __launch_bounds__(256)
    flag_kernel(const int64_t *__restrict__ idx, int64_t NNZ, int Wi, unsigned char *__restrict__ flags,
                unsigned *__restrict__ counter, int item_base)
{
    pdl_trigger();
    const int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= NNZ) return;
    if (p == 0) {
        counter[0] = 0;
        counter[1] = (unsigned)item_base;
        counter[2] = 0;
    }
    unsigned char f = 3;
    if (p > 0) {
        const longlong2 cur = reinterpret_cast<const longlong2 *>(idx)[p];
        const longlong2 prv = reinterpret_cast<const longlong2 *>(idx)[p - 1];
        long long nx = prv.y + 1;
        if (nx == Wi) nx = 0;
        const bool dh = cur.x != prv.x;
        f = (unsigned char)((dh || cur.y != nx) ? 1 : 0) | (unsigned char)(dh ? 2 : 0);
    }
    flags[p] = f;
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
            if (nx == S) return false; // merged into the last run of this latitude
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
            if (nx == ld_wi(idx, a + tst)) {
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
__device__ __forceinline__ uint4 toeplitz8(const __half *__restrict__ val, const Piece &pc, int u0, int k, int dwo,
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
            hv[q] = __half_as_ushort(val[(size_t)pt * kpad + k]);
        }
    }
    return make_uint4((unsigned)hv[0] | ((unsigned)hv[1] << 16), (unsigned)hv[2] | ((unsigned)hv[3] << 16),
                      (unsigned)hv[4] | ((unsigned)hv[5] << 16), (unsigned)hv[6] | ((unsigned)hv[7] << 16));
}

// same for row r of a group piece (window [W, W + l) of longitudes, row arcs in px)
__device__ __forceinline__ uint4 toeplitz8x(const __half *__restrict__ val, int W, int l, int jb, const PieceX &px,
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
                hv[q] = __half_as_ushort(val[(size_t)pt * kpad + k]);
            }
        }
    }
    return make_uint4((unsigned)hv[0] | ((unsigned)hv[1] << 16), (unsigned)hv[2] | ((unsigned)hv[3] << 16),
                      (unsigned)hv[4] | ((unsigned)hv[5] << 16), (unsigned)hv[6] | ((unsigned)hv[7] << 16));
}

// Per-piece Toeplitz B tiles, UMMA K-major no-swizzle layout per k-step s:
//   tile[s][h][n][e] = T[u = 16 s + 8 h + e, n = k * DW + dwo]   (h = K chunk, N rows of 16 B)
#define FQL 176 // 16 leading zeros + 128 points + the largest Toeplitz tail
// Blocks past the first Ho * TY build the longitude-extended input copy instead (8 rows each):
// xe[c, hi, j] = x[c, hi, j mod Wi] in 16-byte chunks, Lc chunks per row.  The two jobs are
// independent, so the bandwidth-bound copy overlaps the latency-bound Toeplitz fill.
template <int RG>
__global__ void __launch_bounds__(128)
    tfill_kernel(const __half *__restrict__ val, const Piece *__restrict__ pieces, const PieceX *__restrict__ pxs,
                 int Wi, int MN1, int ps, int K, int kpad, int N, int DW, uint4 *__restrict__ T, int Ho, int TY,
                 const unsigned *__restrict__ counter, const int *__restrict__ flat, const uint4 *__restrict__ x,
                 uint4 *__restrict__ xe, int nb, int Lc, long long nrows)
{
    // the piece's filter values are staged once in shared memory (zero-padded rows), then each
    // 16-byte Toeplitz row segment is assembled from shared memory
    __shared__ unsigned short fs[RG][KB][FQL];
    const int tid = threadIdx.x;
    pdl_trigger();
    if (blockIdx.x >= (unsigned)(Ho * TY)) {
        // reads only the input and writes only the extended copy (last read by the previous call's
        // main kernel, which completed before this call's first, fully serialized kernel)
        const long long row0 = (long long)(blockIdx.x - Ho * TY) * 8;
        const int warp = tid >> 5;
        const int lane = tid & 31;
        for (int r = warp; r < 8; r += 4) {
            const long long row = row0 + r;
            if (row >= nrows) continue;
            for (int j = lane; j < nb; j += 32) {
                const uint4 v = x[row * nb + j];
                for (int q = j; q < Lc; q += nb) xe[row * Lc + q] = v;
            }
        }
        return;
    }
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
        if (RG > 1) {
            const PieceX px = pxs[pi];
#pragma unroll
            for (int r = 0; r < RG; ++r) {
                if (px.n[r] == 0) continue;
                // local position j (window position jb + j) maps to offset base + j of row r's arc
                int base = pc.S - px.S[r];
                if (base < 0) base += Wi;
                const int jlo = max(-16, -pc.jb), jhi = min(8 * pc.nc, pc.n - pc.jb);
                if ((kpad & 7) == 0) {
                    const int nh = kpad >> 3;
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
                        fs[r][k][16 + j] = __half_as_ushort(val[(size_t)pt * kpad + k]);
                    }
                }
            }
        } else if ((kpad & 7) == 0) {
            // local positions j in [jlo, jhi) are arc points jb + j
            const int jlo = max(-16, -pc.jb), jhi = min(8 * pc.nc, pc.n - pc.jb);
            const int nh = kpad >> 3;
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
                fs[0][k][16 + j] = __half_as_ushort(val[(size_t)pt * kpad + k]);
            }
        }
        __syncthreads();
        uint4 *t = T + pc.toff;
        const int tot = (pc.nc + 1) * N;
        const int delta = pc.S & 7;
        for (int e = tid; e < tot; e += 128) {
            const int kc = e / N;
            const int n0 = e - kc * N;
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

// ------------------------------------------------------------------------------------------
// tcgen05 helpers
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
__device__ __forceinline__ uint64_t umma_desc(unsigned saddr, unsigned lbo, unsigned sbo)
{
    // K-major, no swizzle: core matrices of 8 rows x 16 B; lbo = K-direction stride,
    // sbo = M/N-direction stride (between 8-row groups)
    return (uint64_t)((saddr >> 4) & 0x3FFFu) | ((uint64_t)((lbo >> 4) & 0x3FFFu) << 16)
        | ((uint64_t)((sbo >> 4) & 0x3FFFu) << 32) | (1ull << 46);
}
// K-major no-swizzle descriptor from 16-byte-unit start address and K-direction stride
__device__ __forceinline__ uint64_t mkdesc(unsigned start16, unsigned lbo16)
{
    return (uint64_t)(start16 & 0x3FFFu) | ((uint64_t)(lbo16 & 0x3FFFu) << 16) | (8ull << 32) | (1ull << 46);
}
__device__ __forceinline__ void umma_f16(unsigned dtmem, uint64_t adesc, uint64_t bdesc, unsigned idesc, unsigned acc)
{
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
                 " tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n" ::"r"(dtmem),
                 "l"(adesc), "l"(bdesc), "r"(idesc), "r"(acc));
}
__device__ __forceinline__ void umma_commit(unsigned bar)
{
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void tmem_ld16(unsigned taddr, unsigned *r)
{
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,"
                 "%15}, [%16];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
                 : "r"(taddr));
}
__device__ __forceinline__ void tmem_ld8(unsigned taddr, unsigned *r)
{
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                 : "r"(taddr));
}
__device__ __forceinline__ bool elect_one()
{
    unsigned pred = 0;
    asm volatile("{\n .reg .pred p;\n elect.sync _|p, 0xffffffff;\n selp.u32 %0, 1, 0, p;\n}\n" : "=r"(pred));
    return pred != 0;
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory"); }

// ------------------------------------------------------------------------------------------
// Main kernel.  Item = (output row ho, group of G channels); GEMM row g = (c - c0) * RPC + b
// (block b of channel c; b >= nb are halo rows, discarded), M tiles of 128 rows, G * RPC <= 128 MT.
// Shared-memory slot g = (c - c0) * RPC + i holds x[c, hi, B + 8 i (mod Wi)] for the current
// piece, so the A operand of M tile m starts at slot 128 m.  The slots come from one 4-D TMA box
// of the longitude-extended input copy (TX) or from per-channel bulk copies (two runs at the seam).
// Warp roles: 0 producer, 1 MMA issuer (+TMEM), 2..5 epilogue (TMEM -> fp16 -> global).
// TMEM holds two accumulator sets (MT x N columns each).
// ------------------------------------------------------------------------------------------
// warp 0 producer, warp 1 MMA, then 8 (one CTA per SM) or 4 (two CTAs per SM) epilogue warps
#define QN 4 // item queue depth (producer -> MMA / epilogue)

struct HkParams {
    const __half *x;
    const __half *val;
    const Piece *pieces;
    const int *npieces;
    const int64_t *off;
    const uint4 *T;
    const PieceX *pxs;
    __half *out;
    int BC, Hi, Wi, Ho, Wo, ps, K, kpad;
    int nb, RPC, HB, G, NCG, nitems, NG, ticket; // NG: item rows (row groups)
    int nvs;                                     // log2 of the number of views (3 or 5)
    int nst, xs_bytes, stage_bytes, bar_off;
    int *ictr;         // dynamic item counter (zeroed per call)
    const int *rowsig; // per item row (group): 0 or 4, the input shift of the GEMM-row output blocks
};

__device__ __forceinline__ void decode_item(const HkParams &p, int item, int &ho, int &cg)
{
    const int r = item / p.NCG;
    cg = item - r * p.NCG;
    ho = (r & 1) ? (p.NG - 1 - (r >> 1)) : (r >> 1);
}

// SB: one TMEM accumulator set with per-tile release (more M tiles per item, used for wide N);
// otherwise two accumulator sets (the next item never waits for the epilogue).
template <int DW, bool SB_, int CPS = 1, int RG = 1> struct HkCfg {
    static constexpr int MN1 = KB * DW;                 // logical columns per output row
    static constexpr int MN = ((RG * MN1 + 7) / 8) * 8; // exact MMA width across row groups
    static constexpr int N = ((MN + 15) / 16) * 16;     // TMEM columns per M tile
    static constexpr bool SB = SB_;
    static constexpr int TCOLS = CPS == 3 ? 128 : 512 / CPS; // power-of-two TMEM allocation
    static constexpr int EPIW = CPS == 1 ? 8 : 4;            // epilogue warps
    static constexpr int MT0 = SB ? TCOLS / N : TCOLS / (2 * N);
    static constexpr int MT = MT0 > (SB ? 10 : 8) ? (SB ? 10 : 8) : MT0; // M tiles per item
    // MMA-issuing warps (tile m is issued by warp m % NMW): a single issuer spends most of a piece on
    // bookkeeping while the tensor core drains its few queued MMAs, so the tiles are spread over
    // several issuers that feed the tensor core in parallel
    static constexpr int NMW = MT < 3 ? MT : 3;
    static constexpr int THREADS = 32 * (1 + NMW + EPIW);
    static constexpr int NACC = SB ? MT : 2; // acc_empty barriers
    static constexpr int NACF = SB ? MT : 2; // acc_full barriers (SB: one per M tile)
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
struct TmapX {
    CUtensorMap m[NVMAX];
};

template <int DW, bool TX, bool SB_, int CPS, int RG, bool LP = false>
__global__ void __launch_bounds__(HkCfg<DW, SB_, CPS, RG>::THREADS, CPS)
    disco_hk_kernel(const __grid_constant__ TmapX tmap_x, const HkParams p)
{
    static_assert(!LP || RG == 1);
    using Cfg = HkCfg<DW, SB_, CPS, RG>;
    constexpr int N = Cfg::N;
    constexpr int MN = Cfg::MN;
    constexpr int MN1 = Cfg::MN1;
    constexpr int MT = Cfg::MT;
    constexpr bool SB = SB_;
    constexpr int NACC = Cfg::NACC;
    constexpr int NACF = Cfg::NACF;
    constexpr int EPIW = Cfg::EPIW;
    constexpr int NTH = Cfg::THREADS;
    constexpr int TCOLS = Cfg::TCOLS;
    constexpr int EG = EPIW / 4; // epilogue warps per TMEM lane quadrant
    constexpr int NMW = Cfg::NMW;
    extern __shared__ __align__(1024) unsigned char smem[];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if constexpr (TX)
        if (warp == 0 && lane < (1 << p.nvs)) prefetch_tmap(&tmap_x.m[lane]);
    const int nst = p.nst;
    unsigned char *bar_area = smem + p.bar_off;
    uint64_t *bars = reinterpret_cast<uint64_t *>(bar_area);
    // [0,nst) full, [nst,2nst) empty, 2nst+b acc_full (b < NACF), 2nst+NACF+j acc_empty (j < NACC),
    // then item queue full/empty
    int *sks = reinterpret_cast<int *>(bar_area + 8 * (2 * nst + NACF + NACC + 2 * QN));
    int2 *qbuf = reinterpret_cast<int2 *>(bar_area + 8 * (2 * nst + NACF + NACC + 2 * QN) + 4 * 16);
    unsigned *tmem_slot = reinterpret_cast<unsigned *>(qbuf + QN);
    const unsigned bar0 = smem_u32(bars);
    auto FULL = [&](int s) { return bar0 + 8u * s; };
    auto EMPTY = [&](int s) { return bar0 + 8u * (nst + s); };
    auto ACCF = [&](int b) { return bar0 + 8u * (2 * nst + b); };
    auto ACCE = [&](int b) { return bar0 + 8u * (2 * nst + NACF + b); };
    auto QFULL = [&](int q) { return bar0 + 8u * (2 * nst + NACF + NACC + q); };
    auto QEMPTY = [&](int q) { return bar0 + 8u * (2 * nst + NACF + NACC + QN + q); };
    const int grows = p.G * p.RPC; // GEMM rows filled from the slots

    if (tid == 0) {
        for (int s = 0; s < nst; ++s) {
            mbar_init(FULL(s), 1);
            // per MMA warp: thread arrival (stage metadata read) + tcgen05.commit (MMAs done)
            mbar_init(EMPTY(s), 2 * NMW);
        }
        for (int b = 0; b < NACF; ++b) mbar_init(ACCF(b), SB ? 1 : NMW);
        // SB: tile m is drained by the 4 epilogue warps of half (m & 1); else all 8 warps per set
        for (int j = 0; j < NACC; ++j) mbar_init(ACCE(j), SB ? 4 : EPIW);
        for (int q = 0; q < QN; ++q) {
            mbar_init(QFULL(q), 1);
            mbar_init(QEMPTY(q), NMW + EPIW); // MMA warps + epilogue warps
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    if (warp == 1) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(smem_u32(tmem_slot)),
                     "n"(TCOLS)
                     : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
    }
    // slots past the loaded rows are read only by discarded GEMM rows: keep them finite
    for (int s = 0; s < nst; ++s) {
        uint4 *xs = reinterpret_cast<uint4 *>(smem + (size_t)s * p.stage_bytes);
        for (int e = grows + tid; e < p.xs_bytes / 16; e += NTH) xs[e] = make_uint4(0, 0, 0, 0);
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const unsigned tmem = *tmem_slot;
    const int ps = p.ps;
    pdl_wait(); // pieces, Toeplitz tiles, extended input and the item counter

    if (warp == 0) {
        // ---------------- producer ----------------
        const uint64_t tpolicy = reuse_policy();
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
            auto claim_one = [&]() {
                int v = 0;
                if (lane == 0) v = atomicAdd(p.ictr, 1);
                return __shfl_sync(0xffffffffu, v, 0);
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
            int fut = 0;
            if (lane == 0) fut = atomicAdd(p.ictr, 1);
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

                int ho, cg;
                decode_item(p, item, ho, cg);
                const int c0 = cg * p.G;
                Piece nmine;
                if (lane < np1) nmine = p.pieces[pb1 + lane];
                const int item2 = __shfl_sync(0xffffffffu, fut, 0);
                int np2;
                int64_t pb2;
                meta(item2, np2, pb2);
                if (lane == 0) fut = atomicAdd(p.ictr, 1);

                for (int q0 = 0; q0 < np; q0 += 32) {
                    const int npass = min(32, np - q0);
                    if (q0 > 0 && lane < npass) mine = p.pieces[pbase + q0 + lane];
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
                            for (int e = lane; e < (pc.nc + 1) * MN; e += 32) {
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
                                if (lane == 0)
                                    tma_load_4d(xa, &tmap_x.m[B8 & ((1 << p.nvs) - 1)], 0, B8 >> p.nvs, pc.hi, c0,
                                                FULL(sd));
                            } else {
                                for (int cl = lane; cl < p.G; cl += 32) {
                                    const int c = c0 + cl;
                                    int ch = B8;
                                    const __half *src = p.x + ((size_t)(c < p.BC ? c : p.BC - 1) * p.Hi + pc.hi) * p.Wi;
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
                            const unsigned tb = (unsigned)(mine.nc + 1) * MN * 16u;
                            const unsigned xa = smem_u32(smem + (size_t)sl * p.stage_bytes);
                            sks[sl] = mine.nc;
                            mbar_expect_tx(FULL(sl), xb + tb);
                            bulk_load_reuse(xa + (unsigned)p.xs_bytes, p.T + mine.toff, tb, FULL(sl), tpolicy);
                            const int B8 = mine.S >> 3;
                            if constexpr (TX) {
                                tma_load_4d(xa, &tmap_x.m[B8 & ((1 << p.nvs) - 1)], 0, B8 >> p.nvs, mine.hi, c0,
                                            FULL(sl));
                            } else {
                                for (int cl = 0; cl < p.G; ++cl) {
                                    const int c = c0 + cl;
                                    int ch = B8;
                                    const __half *src = p.x + ((size_t)(c < p.BC ? c : p.BC - 1) * p.Hi + mine.hi) * p.Wi;
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
            const unsigned xbytes = (unsigned)grows * 16u;
            while (true) {
                // publish the item to the MMA and epilogue warps
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
                for (int q = 0; q < np; ++q, ++P) {
                    if (q > 0 && (q & 31) == 0 && q + lane < np) mine = p.pieces[pbase + q + lane];
                    const int hi = __shfl_sync(0xffffffffu, mine.hi, q & 31);
                    const int S = __shfl_sync(0xffffffffu, mine.S, q & 31);
                    const int toff = __shfl_sync(0xffffffffu, mine.toff, q & 31);
                    const int nc = __shfl_sync(0xffffffffu, mine.nc, q & 31);
                    const int ks = piece_tks(nc);
                    if (P >= nst) mbar_wait(EMPTY(s), ph ^ 1u);
                    unsigned char *st = smem + (size_t)s * p.stage_bytes;
                    const unsigned xa = smem_u32(st);
                    const unsigned ta = xa + (unsigned)p.xs_bytes;
                    const unsigned tbytes = (unsigned)((nc + 1) * MN * 16);
                    if (toff < 0) {
                        // Toeplitz tiles not precomputed (scratch overflow): build them here
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
                            for (int e = lane; e < (nc + 1) * MN; e += 32) {
                                const int kc = e / MN;
                                const int n = e - kc * MN;
                                const int k = n / DW;
                                ts[e] = kc < nc ? toeplitz8(p.val, pc, 8 * kc, k, n - k * DW, ps, p.K, p.kpad) :
                                                  make_uint4(0, 0, 0, 0);
                            }
                        } else {
                            const PieceX px = p.pxs[pbase + q];
                            for (int e = lane; e < (nc + 1) * MN; e += 32) {
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
                    __syncwarp();
                    const int B8 = S >> 3;
                    if (lane == 0) {
                        sks[s] = nc;
                        mbar_expect_tx(FULL(s), xbytes + (toff >= 0 ? tbytes : 0u));
                        if (toff >= 0) bulk_load_reuse(ta, p.T + toff, tbytes, FULL(s), tpolicy);
                        if (TX) tma_load_4d(xa, &tmap_x.m[B8 & ((1 << p.nvs) - 1)], 0, B8 >> p.nvs, hi, c0, FULL(s));
                    }
                    if (!TX) {
                        for (int cl = lane; cl < p.G; cl += 32) {
                            const int c = c0 + cl;
                            int ch = B8;
                            const __half *src = p.x + ((size_t)(c < p.BC ? c : p.BC - 1) * p.Hi + hi) * p.Wi;
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
                    __syncwarp();
                    if (++s == nst) {
                        s = 0;
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
        }
    } else if (warp <= NMW) {
        // ---------------- MMA issuers: each warp walks the schedule (uniform values) for its tiles
        // m = mw (mod NMW), one elected lane issues.  Descriptors are advanced by adding
        // (byte offset >> 4) to the base.
        const int mw = warp - 1;
        const unsigned idesc = (1u << 4) | ((unsigned)(MN >> 3) << 17) | ((128u >> 4) << 24);
        const uint64_t adesc0 = umma_desc(smem_u32(smem), 16u, 128u);
        const uint64_t bdesc0 = umma_desc(smem_u32(smem) + (unsigned)p.xs_bytes, (unsigned)(MN * 16), 128u);
        const bool leader = elect_one();
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
            const int mtc = (min(p.G, p.BC - cg * p.G) * p.RPC + 127) >> 7;
            const int ab = SB ? 0 : (it & 1);
            if (!SB && it >= 2) {
                mbar_wait(ACCE(ab), ((it >> 1) - 1) & 1);
                tc_fence_after();
            }
            if (np == 0) {
                if (SB && it > 0)
                    for (int m = mw; m < MT; m += NMW) mbar_wait(ACCE(m), (it - 1) & 1);
                if (leader) {
                    if (SB)
                        for (int m = mw; m < MT; m += NMW) mbar_arrive(ACCF(m));
                    else
                        mbar_arrive(ACCF(ab));
                }
                __syncwarp();
                continue;
            }
            const unsigned dbase = tmem + (unsigned)(ab * MT * N);
            // Chunk-packed issue: a k-step covers two 8-longitude chunks.  A piece's odd last chunk is
            // paired with the next piece's first chunk (both operands via an LBO spanning the two
            // stages) when the next stage lies at a higher shared-memory address.
            int pend = -1;           // stage whose last chunk is still pending
            unsigned pa = 0, pb = 0; // its A / B start (16-byte units)
            for (int q = 0; q < np; ++q) {
                mbar_wait(FULL(s), ph); // TMA-written operands: the mbarrier completion orders them
                const int nc = sks[s];
                __syncwarp();
                if (leader) mbar_arrive(EMPTY(s));
                const unsigned xs16 = (smem_u32(smem) + (unsigned)(s * p.stage_bytes)) >> 4;
                const unsigned ts16 = xs16 + (unsigned)(p.xs_bytes >> 4);
                const bool comb = pend >= 0;
                const int cs = comb ? 1 : 0;
                const int rem = nc - cs;
                // Pair an odd terminal chunk with the next piece instead of multiplying it by the
                // stored zero chunk.  On the first piece, keep at least one pair so D is initialized
                // before the cross-piece MMA accumulates into it.  Ring wrap is handled below by
                // reversing both K halves, which leaves A*B unchanged and keeps descriptor LBO positive.
                const bool carry = (rem & 1) && (q > 0 || nc > 1) && (q + 1 < np);
                const int npair = carry ? (rem >> 1) : ((rem + 1) >> 1);
                // SB, last piece: tile-major with a completion signal per tile, so the epilogue drains
                // tile m while the later tiles still compute
                const bool lastq = SB && q + 1 == np;
                if (SB && q == 0) {
                    // first piece: tile-major, so tile m starts as soon as the epilogue released it
                    for (int m = mw; m < MT; m += NMW) {
                        if (it > 0) {
                            mbar_wait(ACCE(m), (it - 1) & 1);
                            tc_fence_after();
                        }
                        if (leader && m < mtc)
                            for (int j = 0; j < npair; ++j)
                                umma_f16(dbase + (unsigned)(m * N), mkdesc(xs16 + (unsigned)(2 * j + m * 128), 1u),
                                         mkdesc(ts16 + (unsigned)(2 * j * MN), (unsigned)MN), idesc, j > 0 ? 1u : 0u);
                        if (leader && lastq) umma_commit(ACCF(m));
                        __syncwarp();
                    }
                } else if (leader) {
                    if (comb) {
                        const bool fwd = ts16 > pb;
                        const uint64_t bd = fwd ? mkdesc(pb, ts16 - pb) : mkdesc(ts16, pb - ts16);
#pragma unroll
                        for (int m = mw; m < MT; m += NMW)
                            if (m < mtc)
                                umma_f16(dbase + (unsigned)(m * N),
                                         fwd ? mkdesc(pa + (unsigned)(m * 128), xs16 - pa) :
                                               mkdesc(xs16 + (unsigned)(m * 128), pa - xs16),
                                         bd, idesc, 1u);
                        umma_commit(EMPTY(pend));
                    }
                    if (lastq) {
                        for (int m = mw; m < MT; m += NMW) {
                            if (m < mtc)
                                for (int j = 0; j < npair; ++j) {
                                    const unsigned c = (unsigned)(cs + 2 * j);
                                    umma_f16(dbase + (unsigned)(m * N), mkdesc(xs16 + c + (unsigned)(m * 128), 1u),
                                             mkdesc(ts16 + c * (unsigned)MN, (unsigned)MN), idesc, 1u);
                                }
                            umma_commit(ACCF(m));
                        }
                    } else {
                        for (int j = 0; j < npair; ++j) {
                            const unsigned c = (unsigned)(cs + 2 * j);
                            const uint64_t bd = mkdesc(ts16 + c * (unsigned)MN, (unsigned)MN);
                            const unsigned acc = (q > 0 || j > 0) ? 1u : 0u;
#pragma unroll
                            for (int m = mw; m < MT; m += NMW)
                                if (m < mtc)
                                    umma_f16(dbase + (unsigned)(m * N), mkdesc(xs16 + c + (unsigned)(m * 128), 1u), bd,
                                             idesc, acc);
                        }
                    }
                }
                if (carry) {
                    pend = s;
                    pa = xs16 + (unsigned)(nc - 1);
                    pb = ts16 + (unsigned)((nc - 1) * MN);
                } else {
                    pend = -1;
                    if (leader) umma_commit(EMPTY(s));
                }
                __syncwarp();
                if (++s == nst) {
                    s = 0;
                    ph ^= 1u;
                }
            }
            if (!SB && leader) umma_commit(ACCF(ab));
            __syncwarp();
        }
    } else {
        // ---------------- epilogue ----------------
        const int quad = warp & 3;              // TMEM lane quadrant accessible to this warp
        const int half = (warp - 1 - NMW) >> 2; // this warp drains tiles m = half (mod EG)
        const int lrow = quad * 32 + lane;
        const size_t kstride = (size_t)p.Ho * p.Wo;
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
            const int mtc = (min(p.G, p.BC - c0) * p.RPC + 127) >> 7;
            const int ab = SB ? 0 : (it & 1);
            if (!SB) {
                mbar_wait(ACCF(ab), (it >> 1) & 1);
                tc_fence_after();
            }
            // Each of this warp's tiles (m = half + EG i) is drained into packed fp16 registers and
            // released to the MMA warps (SB) before its global stores.
            constexpr int TPH = (MT + EG - 1) / EG;
            constexpr int NWT = N / 2;
            // half-block shift: GEMM row b holds outputs DW b + DW / 2 + dwo (mod Wo), stored as two
            // naturally aligned half vectors (the second one wraps at the seam for the last block)
            const bool hshift = DW >= 2 && p.rowsig[ho] != 0;
#pragma unroll
            for (int i = 0; i < TPH; ++i) {
                const int m = half + EG * i;
                if (m < MT) {
                    unsigned hp[NWT];
                    if (SB) {
                        mbar_wait(ACCF(m), it & 1);
                        tc_fence_after();
                    }
                    if (np > 0 && m < mtc) {
                        const unsigned ta = tmem + ((unsigned)(quad * 32) << 16) + (unsigned)(ab * MT * N + m * N);
                        if constexpr ((MN & 15) != 0) {
                            static_assert((MN & 15) == 8);
                            constexpr int NFULL = MN / 16;
                            constexpr int HEAD = ((NFULL - 1) / 3) * 3;
#pragma unroll
                            for (int g0 = 0; g0 < HEAD; g0 += 3) {
                                unsigned r[48];
#pragma unroll
                                for (int g = 0; g < 3; ++g) tmem_ld16(ta + (unsigned)(16 * (g0 + g)), r + 16 * g);
                                tmem_wait_ld();
#pragma unroll
                                for (int g = 0; g < 3; ++g)
#pragma unroll
                                    for (int w = 0; w < 8; ++w) {
                                        __half2 h = __floats2half2_rn(__uint_as_float(r[16 * g + 2 * w]),
                                                                      __uint_as_float(r[16 * g + 2 * w + 1]));
                                        hp[8 * (g0 + g) + w] = *reinterpret_cast<unsigned *>(&h);
                                    }
                            }
                            constexpr int REM = NFULL - HEAD;
                            unsigned r[16 * REM + 8];
#pragma unroll
                            for (int g = 0; g < REM; ++g) tmem_ld16(ta + (unsigned)(16 * (HEAD + g)), r + 16 * g);
                            tmem_ld8(ta + (unsigned)(MN & ~15), r + 16 * REM);
                            tmem_wait_ld();
#pragma unroll
                            for (int g = 0; g < REM; ++g)
#pragma unroll
                                for (int w = 0; w < 8; ++w) {
                                    __half2 h = __floats2half2_rn(__uint_as_float(r[16 * g + 2 * w]),
                                                                  __uint_as_float(r[16 * g + 2 * w + 1]));
                                    hp[8 * (HEAD + g) + w] = *reinterpret_cast<unsigned *>(&h);
                                }
#pragma unroll
                            for (int w = 0; w < 4; ++w) {
                                __half2 h = __floats2half2_rn(__uint_as_float(r[16 * REM + 2 * w]),
                                                              __uint_as_float(r[16 * REM + 2 * w + 1]));
                                hp[8 * NFULL + w] = *reinterpret_cast<unsigned *>(&h);
                            }
                        } else {
#pragma unroll
                            for (int g0 = 0; g0 < MN / 16; g0 += 3) {
                                unsigned r[48];
#pragma unroll
                                for (int g = 0; g < 3; ++g)
                                    if (g0 + g < MN / 16) tmem_ld16(ta + (unsigned)(16 * (g0 + g)), r + 16 * g);
                                tmem_wait_ld();
#pragma unroll
                                for (int g = 0; g < 3; ++g)
                                    if (g0 + g < MN / 16)
#pragma unroll
                                        for (int w = 0; w < 8; ++w) {
                                            __half2 h = __floats2half2_rn(__uint_as_float(r[16 * g + 2 * w]),
                                                                          __uint_as_float(r[16 * g + 2 * w + 1]));
                                            hp[8 * (g0 + g) + w] = *reinterpret_cast<unsigned *>(&h);
                                        }
                            }
                        }
                    } else {
#pragma unroll
                        for (int w = 0; w < NWT; ++w) hp[w] = 0u;
                    }
                    if (SB) {
                        tc_fence_before();
                        __syncwarp();
                        if (lane == 0) mbar_arrive(ACCE(m));
                    }
                    if (m < mtc) do {
                            const int g = m * 128 + lrow;
                            const int cl = g / p.RPC;
                            const int b = g - cl * p.RPC;
                            const int c = c0 + cl;
                            if constexpr (DW >= 2) {
                                if (hshift) {
                                    if (cl < p.G && c < p.BC && b < p.nb) {
                                        const int w1 = DW * b + DW / 2;
                                        const int w2 = (DW * b + DW == p.Wo) ? 0 : DW * b + DW;
#pragma unroll
                                        for (int r = 0; r < RG; ++r) {
                                            const int hr = ho * RG + r;
                                            if (hr >= p.Ho) break;
                                            __half *rowp = p.out + (((size_t)c * p.K) * p.Ho + hr) * (size_t)p.Wo;
#pragma unroll
                                            for (int k = 0; k < KB; ++k) {
                                                const int c0w = r * MN1 + k * DW;
                                                if (k < p.K) {
                                                    __half *dk = rowp + k * kstride;
                                                    if constexpr (DW == 8) {
                                                        *reinterpret_cast<uint2 *>(dk + w1)
                                                            = make_uint2(hp[c0w / 2], hp[c0w / 2 + 1]);
                                                        *reinterpret_cast<uint2 *>(dk + w2)
                                                            = make_uint2(hp[c0w / 2 + 2], hp[c0w / 2 + 3]);
                                                    } else if constexpr (DW == 4) {
                                                        *reinterpret_cast<unsigned *>(dk + w1) = hp[c0w / 2];
                                                        *reinterpret_cast<unsigned *>(dk + w2) = hp[c0w / 2 + 1];
                                                    } else {
                                                        const unsigned w = hp[c0w / 2];
                                                        dk[w1] = __ushort_as_half((unsigned short)(w & 0xFFFFu));
                                                        dk[w2] = __ushort_as_half((unsigned short)(w >> 16));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    break;
                                }
                            }
                            if (cl < p.G && c < p.BC && b < p.nb) {
#pragma unroll
                                for (int r = 0; r < RG; ++r) {
                                    const int hr = ho * RG + r;
                                    if (hr >= p.Ho) break;
                                    __half *dst = p.out + (((size_t)c * p.K) * p.Ho + hr) * (size_t)p.Wo + (size_t)DW * b;
#pragma unroll
                                    for (int k = 0; k < KB; ++k) {
                                        const int c0w = r * MN1 + k * DW; // first column of (row r, basis k)
                                        if (k < p.K) {
                                            if (DW == 8) {
                                                *reinterpret_cast<uint4 *>(dst + k * kstride) = make_uint4(
                                                    hp[c0w / 2], hp[c0w / 2 + 1], hp[c0w / 2 + 2], hp[c0w / 2 + 3]);
                                            } else if (DW == 4) {
                                                *reinterpret_cast<uint2 *>(dst + k * kstride)
                                                    = make_uint2(hp[c0w / 2], hp[c0w / 2 + 1]);
                                            } else if (DW == 2) {
                                                *reinterpret_cast<unsigned *>(dst + k * kstride) = hp[c0w / 2];
                                            } else {
                                                const unsigned w = hp[c0w >> 1];
                                                dst[k * kstride] = __ushort_as_half(
                                                    (unsigned short)((c0w & 1) ? (w >> 16) : (w & 0xFFFFu)));
                                            }
                                        }
                                    }
                                }
                            }
                        } while (0);
                }
            }
            if (!SB) {
                tc_fence_before();
                __syncwarp();
                if (lane == 0) mbar_arrive(ACCE(ab));
            }
        }
    }

    __syncthreads();
    if (warp == 1) {
        tc_fence_after();
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(tmem), "n"(TCOLS) : "memory");
    }
}

// Longitude-extended input copy: xe[c, hi, j] = x[c, hi, j mod Wi], j < 8 * Lc (16-byte chunks).
__global__ void __launch_bounds__(256)
    xext_kernel(const uint4 *__restrict__ x, uint4 *__restrict__ xe, int nb, int Lc, long long nrows)
{
    // 8 rows per block, 4 chunks in flight per thread
    const long long row0 = (long long)blockIdx.x * 8;
    const int tot = 8 * Lc;
    for (int e0 = threadIdx.x; e0 < tot; e0 += 4 * 256) {
        uint4 v[4];
        long long d[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int e = e0 + u * 256;
            d[u] = -1;
            if (e < tot) {
                const int r = e / Lc;
                const int j = e - r * Lc;
                if (row0 + r < nrows) {
                    int q = j;
                    while (q >= nb) q -= nb;
                    v[u] = x[(row0 + r) * nb + q];
                    d[u] = (row0 + r) * Lc + j;
                }
            }
        }
#pragma unroll
        for (int u = 0; u < 4; ++u)
            if (d[u] >= 0) xe[d[u]] = v[u];
    }
}

// ------------------------------------------------------------------------------------------
// General fallback (CUDA cores, fp32): one thread per (bc, ho, wo).
// ------------------------------------------------------------------------------------------
__global__ void disco_fallback_kernel(const __half *__restrict__ x, const int64_t *__restrict__ idx,
                                      const __half *__restrict__ val, const int64_t *__restrict__ off,
                                      __half *__restrict__ out, int BC, int Hi, int Wi, int Ho, int Wo, int ps, int K,
                                      int kpad)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)BC * Ho * Wo;
    if (t >= total) return;
    const int wo = (int)(t % Wo);
    const int ho = (int)((t / Wo) % Ho);
    const int bc = (int)(t / ((long long)Wo * Ho));
    const long long a = off[ho], b = off[ho + 1];
    const long long sh = (long long)wo * ps;
    for (int k0 = 0; k0 < K; k0 += 16) {
        float acc[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) acc[k] = 0.f;
        for (long long p = a; p < b; ++p) {
            const long long hi = idx[2 * p];
            const long long wi = (idx[2 * p + 1] + sh) % Wi;
            const float xv = __half2float(x[((size_t)bc * Hi + hi) * Wi + wi]);
#pragma unroll
            for (int k = 0; k < 16; ++k)
                if (k0 + k < K && k0 + k < kpad) acc[k] += __half2float(val[p * kpad + k0 + k]) * xv;
        }
        for (int k = 0; k < 16 && k0 + k < K; ++k)
            out[(((size_t)bc * K + k0 + k) * Ho + ho) * (size_t)Wo + wo] = __float2half_rn(acc[k]);
    }
}

// ------------------------------------------------------------------------------------------
// Host
// ------------------------------------------------------------------------------------------
// Scratch is reusable only by calls on the same stream: stream order keeps the previous call's
// kernels ahead of the next call's preprocessing.  Other streams/devices receive disjoint storage.
// The mutex also makes each multi-kernel host enqueue sequence atomic, so two host threads using
// the same stream cannot interleave their preprocessing and main-kernel launches.
struct Workspace {
    int device;
    cudaStream_t stream;
    void *ptr;
    size_t bytes;
};
static std::recursive_mutex g_workspace_mutex;
static std::vector<Workspace> g_workspaces;

static void *get_scratch(size_t bytes, int device, cudaStream_t stream)
{
    std::lock_guard<std::recursive_mutex> lock(g_workspace_mutex);
    for (Workspace &w : g_workspaces) {
        if (w.device != device || w.stream != stream) continue;
        if (w.bytes >= bytes) return w.ptr;
        void *p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
        if (w.ptr) cudaFree(w.ptr);
        w.ptr = p;
        w.bytes = bytes;
        return p;
    }
    void *p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
    g_workspaces.push_back({device, stream, p, bytes});
    return p;
}

// launch with programmatic stream serialization when pdl is set (the kernel calls pdl_wait before
// consuming the predecessor's results; without the attribute pdl_wait/pdl_trigger are no-ops)
template <typename... KArgs, typename... Args>
static void launch_pdl(bool pdl, void (*k)(KArgs...), unsigned grid, unsigned block, size_t smem, cudaStream_t st,
                       Args... args)
{
    cudaLaunchConfig_t c = {};
    c.gridDim = dim3(grid);
    c.blockDim = dim3(block);
    c.dynamicSmemBytes = smem;
    c.stream = st;
    cudaLaunchAttribute a[1];
    a[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    a[0].val.programmaticStreamSerializationAllowed = 1;
    c.attrs = a;
    c.numAttrs = pdl ? 1 : 0;
    cudaLaunchKernelEx(&c, k, args...);
}

static int num_sms()
{
    std::lock_guard<std::recursive_mutex> lock(g_workspace_mutex);
    static int cached[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if ((unsigned)dev < 64 && cached[dev]) return cached[dev];
    int nsm = 0;
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    if ((unsigned)dev < 64) cached[dev] = nsm;
    return nsm;
}

// Piece length: the longest of {128, 96, 80, 64, 48, 32} points whose halo lets the channel groups
// fill >= 95% of the M tiles (best fill otherwise).  Longer pieces waste less window per point.
// Returns the channel group size G (0 if none fits) and the chosen pmax / k-step bound.
static int choose_group(int nb, int ps, int MT, int BC, int &pmax, int &kstep_max, bool *capped = nullptr)
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

template <int DW, bool SB, int CPS, int RG>
static bool launch_hk2(cudaStream_t stream, const __half *x, const int64_t *idx, const __half *val, const int64_t *off,
                       __half *o, int BC, int Hi, int Wi, int Ho, int Wo, int ps, int K, int kpad, int64_t NNZ)
{
    using Cfg = HkCfg<DW, SB, CPS, RG>;
    constexpr int MN = Cfg::MN;
    constexpr int MT = Cfg::MT;
    constexpr int NTH = Cfg::THREADS;
    const int nunit = 2 * MN; // 16-byte units per k-step tile
    HkParams prm;
    prm.nb = Wi / 8;
    int pmax = 0, kstep_max = 0;
    prm.G = choose_group(prm.nb, ps, MT, BC, pmax, kstep_max);
    if (prm.G < 1) return false;
    prm.HB = 2 * kstep_max - 1;
    prm.RPC = (prm.nb + prm.HB + 7) & ~7; // per-channel slot stride (whole 128-byte TMA rows)
    const int nc_cap = 2 * kstep_max;
    prm.NCG = (BC + prm.G - 1) / prm.G;
    prm.NG = (Ho + RG - 1) / RG;
    prm.nitems = prm.NG * prm.NCG;
    const bool tx = prm.RPC / 8 <= 256 && prm.G <= 256;
    // 512-byte TMA rows when the channel pitch is a whole number of them, else 128-byte rows
    prm.nvs = (prm.RPC % 32 == 0) ? 5 : 3;
    const int nv = 1 << prm.nvs;
    const int Lc = ((prm.nb + nv - 1) & ~(nv - 1)) + prm.RPC; // minimal whole-TMA-row extension

    const size_t nn = (size_t)(NNZ > 0 ? NNZ : 1);
    auto al = [](size_t v) { return ((v + 255) / 256) * 256; };
    const long long cap = (long long)nunit * ((long long)nn / 8 + (long long)Ho * 64 + 4096);
    const size_t pieces_bytes = al(nn * sizeof(Piece));
    const size_t sub_bytes = al(nn * 4);
    const size_t np_bytes = al((size_t)Ho * 4);
    const size_t fl_bytes = al(nn);
    const size_t cnt_bytes = 256;
    const size_t t_bytes = al((size_t)cap * 16);
    const size_t xe_bytes = tx ? al((size_t)BC * Hi * Lc * 16) : 0;
    const size_t flat_bytes = al(nn * 4);
    const size_t arc_bytes = RG > 1 ? al(nn * sizeof(Arc)) : 0;
    const size_t px_bytes = RG > 1 ? al(nn * sizeof(PieceX)) : 0;
    const size_t sig_bytes = al((size_t)Ho * 4);
    int device = 0;
    cudaGetDevice(&device);
    unsigned char *scr
        = static_cast<unsigned char *>(get_scratch(pieces_bytes + sub_bytes + np_bytes + fl_bytes + cnt_bytes + t_bytes
                                                       + xe_bytes + flat_bytes + arc_bytes + px_bytes + sig_bytes,
                                                   device, stream));
    if (!scr) return false;
    Piece *pieces = reinterpret_cast<Piece *>(scr);
    int *substart = reinterpret_cast<int *>(scr + pieces_bytes);
    int *npieces = reinterpret_cast<int *>(scr + pieces_bytes + sub_bytes);
    unsigned char *flags = scr + pieces_bytes + sub_bytes + np_bytes;
    unsigned *counter = reinterpret_cast<unsigned *>(scr + pieces_bytes + sub_bytes + np_bytes + fl_bytes);
    uint4 *T = reinterpret_cast<uint4 *>(scr + pieces_bytes + sub_bytes + np_bytes + fl_bytes + cnt_bytes);
    uint4 *xe = reinterpret_cast<uint4 *>(scr + pieces_bytes + sub_bytes + np_bytes + fl_bytes + cnt_bytes + t_bytes);
    int *flat
        = reinterpret_cast<int *>(scr + pieces_bytes + sub_bytes + np_bytes + fl_bytes + cnt_bytes + t_bytes + xe_bytes);
    Arc *arcs = reinterpret_cast<Arc *>(reinterpret_cast<unsigned char *>(flat) + flat_bytes);
    PieceX *pxs = reinterpret_cast<PieceX *>(reinterpret_cast<unsigned char *>(arcs) + arc_bytes);
    int *rowsig = reinterpret_cast<int *>(reinterpret_cast<unsigned char *>(pxs) + px_bytes);
    const int nsm = num_sms();
    const int grid = prm.nitems < CPS * nsm ? prm.nitems : CPS * nsm;

    TmapX tmap;
    memset(&tmap, 0, sizeof(tmap));
    if (tx) {
        for (int j = 0; j < nv; ++j) {
            cuuint64_t gdim[4]
                = {(cuuint64_t)(8 * nv), (cuuint64_t)(Lc / nv - (j > 0 ? 1 : 0)), (cuuint64_t)Hi, (cuuint64_t)BC};
            cuuint64_t gstr[3] = {(cuuint64_t)(16 * nv), (cuuint64_t)Lc * 16, (cuuint64_t)Hi * Lc * 16};
            cuuint32_t box[4] = {(cuuint32_t)(8 * nv), (cuuint32_t)(prm.RPC / nv), 1, (cuuint32_t)prm.G};
            cuuint32_t es[4] = {1, 1, 1, 1};
            if (cuTensorMapEncodeTiled(&tmap.m[j], CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 4, xe + j, gdim, gstr, box, es,
                                       CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                       CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
                != CUDA_SUCCESS)
                return false;
        }
    }

    // Overlapping the planner, the Toeplitz build and the main kernel's prologue with their
    // predecessors pays on the regular paths; the three-CTA small-channel path runs strictly serialized
    // (its short kernels lose more to co-residency than they gain).
    const bool pdl = CPS != 3;
    flag_kernel<<<(unsigned)((NNZ + 255) / 256), 256, 0, stream>>>(idx, NNZ, Wi, flags, counter, grid);
    if (RG == 1)
        launch_pdl(pdl, row_kernel, (unsigned)Ho, 256u, 0, stream, idx, off, (const unsigned char *)flags, Wi, ps,
                   nc_cap, nunit, substart, pieces, npieces, counter, cap, flat, rowsig);
    else
        launch_pdl(pdl, group_kernel, (unsigned)prm.NG, (unsigned)GT, 0, stream, idx, off, (const unsigned char *)flags,
                   Wi, ps, nc_cap, nunit, Ho, substart, arcs, pieces, pxs, npieces, counter, cap, flat, rowsig);
    {
        // Toeplitz-builder blocks per SM, followed by the extended-copy blocks.
        const int TY = 6;
        const long long nxb = tx ? ((long long)BC * Hi + 7) / 8 : 0;
        launch_pdl(pdl, tfill_kernel<RG>, (unsigned)(nsm * TY + nxb), 128u, 0, stream, val, (const Piece *)pieces,
                   (const PieceX *)pxs, Wi, Cfg::MN1, ps, K, kpad, MN, DW, T, nsm, TY, (const unsigned *)counter,
                   (const int *)flat, reinterpret_cast<const uint4 *>(x), xe, prm.nb, Lc, (long long)BC * Hi);
    }

    prm.x = x;
    prm.val = val;
    prm.pieces = pieces;
    prm.npieces = npieces;
    prm.off = off;
    prm.T = T;
    prm.pxs = pxs;
    prm.out = o;
    prm.ictr = reinterpret_cast<int *>(counter + 1);
    prm.rowsig = rowsig;
    prm.BC = BC;
    prm.Hi = Hi;
    prm.Wi = Wi;
    prm.Ho = Ho;
    prm.Wo = Wo;
    prm.ps = ps;
    prm.K = K;
    prm.kpad = kpad;
    const size_t limit = (CPS == 1 ? 227 : (CPS == 2 ? 113 : (CPS == 3 ? 75 : 56))) * 1024;
    prm.xs_bytes = (((128 * MT + prm.HB) * 16 + 127) / 128) * 128;
    const int ts_bytes = (nc_cap + 1) * MN * 16;
    prm.stage_bytes = ((prm.xs_bytes + ts_bytes + 127) / 128) * 128;
    const int bar_bytes = 1024;
    int nst = (int)((limit - bar_bytes) / prm.stage_bytes);
    if (nst > 16) nst = 16;
    if (nst < 2) return false;
    prm.nst = nst;
    prm.bar_off = nst * prm.stage_bytes;
    const size_t smem = (size_t)prm.bar_off + bar_bytes;

    static uint64_t attr_devices = 0;
    const uint64_t dev_bit = (unsigned)device < 64 ? (1ull << device) : 0;
    if (!dev_bit || !(attr_devices & dev_bit)) {
        cudaFuncSetAttribute(disco_hk_kernel<DW, true, SB, CPS, RG, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             227 * 1024);
        cudaFuncSetAttribute(disco_hk_kernel<DW, false, SB, CPS, RG, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
        if constexpr (RG == 1) {
            cudaFuncSetAttribute(disco_hk_kernel<DW, true, SB, CPS, RG, true>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
            cudaFuncSetAttribute(disco_hk_kernel<DW, false, SB, CPS, RG, true>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
        }
        if (dev_bit) attr_devices |= dev_bit;
    }
    prm.ticket = prm.nitems >= 64 * grid ? 4 : 1;
    if constexpr (RG == 1) {
        if (prm.G >= 4 && prm.RPC <= 128) {
            if (tx)
                launch_pdl(pdl, disco_hk_kernel<DW, true, SB, CPS, RG, true>, (unsigned)grid, (unsigned)NTH, smem,
                           stream, tmap, prm);
            else
                launch_pdl(pdl, disco_hk_kernel<DW, false, SB, CPS, RG, true>, (unsigned)grid, (unsigned)NTH, smem,
                           stream, tmap, prm);
            return true;
        }
    }
    if (tx)
        launch_pdl(pdl, disco_hk_kernel<DW, true, SB, CPS, RG, false>, (unsigned)grid, (unsigned)NTH, smem, stream,
                   tmap, prm);
    else
        launch_pdl(pdl, disco_hk_kernel<DW, false, SB, CPS, RG, false>, (unsigned)grid, (unsigned)NTH, smem, stream,
                   tmap, prm);
    return true;
}

// One TMEM accumulator set with per-tile release (SB) lets an item hold twice the M tiles, i.e.
// more channels share each Toeplitz tile and stage; it pays when that actually enlarges the
// channel group, otherwise the double-buffered set (no item-boundary bubble) is used.
template <int DW, int RG = 1>
static bool launch_hk(cudaStream_t stream, const __half *x, const int64_t *idx, const __half *val, const int64_t *off,
                      __half *o, int BC, int Hi, int Wi, int Ho, int Wo, int ps, int K, int kpad, int64_t NNZ)
{
    int pm, ksm;
    // Small channel counts expose too little independent work per CTA.  Three single-buffered CTAs
    // retain the same two-M-tile item geometry while increasing the number of resident pipelines.
    if (BC <= 32 && choose_group(Wi / 8, ps, HkCfg<DW, true, 3, RG>::MT, BC, pm, ksm) >= 1)
        return launch_hk2<DW, true, 3, RG>(stream, x, idx, val, off, o, BC, Hi, Wi, Ho, Wo, ps, K, kpad, NNZ);
    // Two CTAs per SM (half the TMEM / shared memory each) overlap one CTA's pipeline bubbles with
    // the other's MMAs; used when channels are plentiful.  With few channels the items are few and
    // long (polar rows), and one CTA per SM with the larger tile set is better.
    // Few channels: items are small, so double-buffered accumulators (no item-boundary bubble) with
    // two CTAs per SM work best.
    bool capped = true;
    const int g2 = choose_group(Wi / 8, ps, HkCfg<DW, true, 2, RG>::MT, BC, pm, ksm, &capped);
    if (g2 >= 1 && !capped)
        return launch_hk2<DW, true, 2, RG>(stream, x, idx, val, off, o, BC, Hi, Wi, Ho, Wo, ps, K, kpad, NNZ);
    if (choose_group(Wi / 8, ps, HkCfg<DW, false, 2, RG>::MT, BC, pm, ksm) >= 1)
        return launch_hk2<DW, false, 2, RG>(stream, x, idx, val, off, o, BC, Hi, Wi, Ho, Wo, ps, K, kpad, NNZ);
    const int gs = choose_group(Wi / 8, ps, HkCfg<DW, true, 1, RG>::MT, BC, pm, ksm);
    const int gd = choose_group(Wi / 8, ps, HkCfg<DW, false, 1, RG>::MT, BC, pm, ksm);
    if (gs >= 1 && 2 * gs >= 3 * gd)
        return launch_hk2<DW, true, 1, RG>(stream, x, idx, val, off, o, BC, Hi, Wi, Ho, Wo, ps, K, kpad, NNZ);
    return launch_hk2<DW, false, 1, RG>(stream, x, idx, val, off, o, BC, Hi, Wi, Ho, Wo, ps, K, kpad, NNZ);
}

void Run(ffi::TensorView inp, ffi::TensorView pack_idx, ffi::TensorView pack_val, ffi::TensorView pack_offset,
         int64_t kernel_size, int64_t nlon_out, ffi::TensorView out)
{
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, inp.device().device_id));
    const int B = (int)inp.size(0), C = (int)inp.size(1), Hi = (int)inp.size(2), Wi = (int)inp.size(3);
    const int BC = B * C;
    const int K = (int)kernel_size;
    const int Wo = (int)nlon_out;
    const int Ho = (int)pack_offset.size(0) - 1;
    const int64_t NNZ = pack_idx.size(0);
    const int kpad = (int)pack_val.size(1);
    const int ps = Wo > 0 ? Wi / Wo : 0;

    const __half *x = static_cast<const __half *>(inp.data_ptr());
    const int64_t *idx = static_cast<const int64_t *>(pack_idx.data_ptr());
    const __half *val = static_cast<const __half *>(pack_val.data_ptr());
    const int64_t *off = static_cast<const int64_t *>(pack_offset.data_ptr());
    __half *o = static_cast<__half *>(out.data_ptr());
    if (Ho <= 0 || BC <= 0 || Wo <= 0 || K <= 0) return;

    const bool hk_ok
        = (ps >= 1) && (Wi == ps * Wo) && (Wo % 8 == 0) && (8 % ps == 0) && (K <= KB) && (kpad >= K) && (NNZ > 0);
    if (hk_ok) {
        std::unique_lock<std::recursive_mutex> enqueue_lock(g_workspace_mutex);
        bool done;
        if (ps == 1) {
            // Row pairs (N = 144): half the windows and MMAs per output, three M tiles of 144 TMEM
            // columns at one CTA per SM; they pay when there are enough row-pair items to balance the
            // long polar ones over the SMs.  Rings that fit one 128-row tile (short runs, many pieces
            // per row) stay on single rows, whose producer issues the pieces lane-parallel.
            int pm, ksm;
            const int g1 = choose_group(Wi / 8, ps, HkCfg<8, true, 1, 2>::MT, BC, pm, ksm);
            const long long items2 = g1 >= 1 ? (long long)((Ho + 1) / 2) * ((BC + g1 - 1) / g1) : 0;
            const int gs = choose_group(Wi / 8, ps, HkCfg<8, true, 2, 1>::MT, BC, pm, ksm);
            const bool lane_par = gs >= 4 && ((Wi / 8 + 2 * ksm - 1 + 7) & ~7) <= 128;
            if (!lane_par && items2 >= 8LL * num_sms())
                done = launch_hk2<8, true, 1, 2>(stream, x, idx, val, off, o, BC, Hi, Wi, Ho, Wo, ps, K, kpad, NNZ);
            else
                done = launch_hk<8>(stream, x, idx, val, off, o, BC, Hi, Wi, Ho, Wo, ps, K, kpad, NNZ);
        } else if (ps == 2) {
            // Row pairs halve the windows and MMAs per output but make every item twice as long; they
            // pay when there are enough items to balance the long polar ones (several per CTA).
            int pm, ksm;
            const int g2 = choose_group(Wi / 8, ps, HkCfg<4, true, 2, 2>::MT, BC, pm, ksm);
            const long long items2 = g2 >= 1 ? (long long)((Ho + 1) / 2) * ((BC + g2 - 1) / g2) : 0;
            if (items2 >= 8LL * num_sms())
                done = launch_hk<4, 2>(stream, x, idx, val, off, o, BC, Hi, Wi, Ho, Wo, ps, K, kpad, NNZ);
            else
                done = launch_hk<4, 1>(stream, x, idx, val, off, o, BC, Hi, Wi, Ho, Wo, ps, K, kpad, NNZ);
        } else if (ps == 4)
            done = launch_hk<2>(stream, x, idx, val, off, o, BC, Hi, Wi, Ho, Wo, ps, K, kpad, NNZ);
        else
            done = launch_hk<1>(stream, x, idx, val, off, o, BC, Hi, Wi, Ho, Wo, ps, K, kpad, NNZ);
        if (done) return;
        enqueue_lock.unlock();
    }
    const long long total = (long long)BC * Ho * Wo;
    const int threads = 128;
    const long long blocks = (total + threads - 1) / threads;
    disco_fallback_kernel<<<(unsigned)blocks, threads, 0, stream>>>(x, idx, val, off, o, BC, Hi, Wi, Ho, Wo, ps, K, kpad);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
