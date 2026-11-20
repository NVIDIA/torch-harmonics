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

// Channel-row tensor-core kernel for the DISCO transpose (used for many channels), plus the general fp32
// atomic-scatter fallback. See the comment at the top of kernel.cu.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <algorithm>
#include <cstring>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/error.h>

namespace ffi = tvm::ffi;

namespace cr
{
    // DISCO spherical convolution transpose (arc scatter), fp16 activations, fp32 psi / accumulation.
    //
    // Fast path (pscale P = 1 or 2): gather by output ring on tensor cores, channels in the MMA rows.
    //   For one arc (row r, ring h) with taps a in [a_lo, a_hi] and residue b of the output longitude o = P*m + b:
    //     out[h, P*m + b] += sum_a v_b(a) * x_r[(m - a) mod Wi].
    //   With Ph = 8 / P phases, m = Ph*mu + ph and the N index n = P*ph + b, the output longitude is o = 8*mu + n, and
    //     D[row][mu][n] = sum_k A[row][mu][k] * B[k][n],   A[row][mu][k] = x_row[Ph*mu + koff + k],
    //     B[k][n] = v_b(ph - k - koff)  (Toeplitz, shared by every row and mu).
    //   MMA rows are 16 (channel, longitude split) pairs; a warp owns NMU consecutive mu of one ring, so the A
    //   fragment of (mu, k-block kb) equals the one of (mu + 8/Ph, kb - 1): consecutive mu reuse the same registers.
    //   psi (fp32) is split into two fp16 terms after a global power-of-two scale (about 22 mantissa bits); both
    //   products are accumulated in fp32 tensor-core accumulators and the output is rounded to fp16 once.
    //   A CTA owns G consecutive output rings x 16 MMA rows x all longitudes; per (input row, ring group) "entry" a
    //   producer warp streams the input windows (TMA bulk copies), psi operand and arc records into a stage ring.
    //   Long arcs are cut into sub-arcs of at most KPMAX taps so that every window fits a fixed stage.
    // General path (anything else): fp32 atomic scatter + fp16 narrowing.
    //
    // Every call recomputes all psi bookkeeping (sub-arcs, entries, ordering, scale, Toeplitz operands) on the GPU.

#define NSPLIT 2    // scaled fp16 terms per fp32 psi value
#define NMU 15      // mu per warp
#define KPMAX 512   // max K extent of a sub-arc
#define CR_SEGMAX 8 // max sub-arcs per entry
#define NBMAX 16    // row buckets (entry ordering within a ring group)
#define MAXST 16    // max pipeline stages
#define MAXMW 12    // max MMA warps per CTA
#define NPW 4       // producer warps per CTA
#define MAXSPLIT 16
#define MAXSI 48 // max split work items
#ifndef PAIR2
#define PAIR2 0
#endif

    struct Cfg {
        int P, Ph, Wi, Wo, Ho, Hi, K, BC;
        int S, CT, MS, NWL, G, NG, NB, RB, CLW;
        int CHS, XLMAX, SPMAX;
        int off_ps, off_hot, off_ent, stage_bytes, nst, ntiles;
        int sig_lo, sig_hi;
        int rowch[16], rowsp[16];
    };

    struct __align__(16) Entry {
        int xrow, u0, nsub, WB;
        int q0, spw, len1, len2; // input window copy: [q0, q0 + len1) then [0, len2) then [0, len3) (halfs)
        long long spg;
        int len3, grp;
    };

    struct __align__(16) BuildRec {
        int stn, len, koff, kp;
        long long voff, spg;
    };

    struct __align__(16) Item {
        int g, e0, e1, slot;     // slot: first scratch slice of the group's splits, -1 if not split
        int ns, sid, pad0, pad1; // ns splits, sid: split-group id (counter index)
    };

    __host__ __device__ __forceinline__ int floordiv_i(int a, int b)
    {
        int q = a / b;
        if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
        return q;
    }
    __host__ __device__ __forceinline__ int nk_words(int P, int kp)
    {
        if (P == 1) return kp + 8;
        return (kp & 8) ? kp + 16 : kp + 8; // >= kp + 4 and == 8 (mod 16): residue arrays on disjoint banks
    }

    // Walk the arcs of row r, cut them into sub-arcs and group consecutive sub-arcs of one ring group into entries.
    template <class ESub, class EEnt>
    __device__ __forceinline__ void walk_row(int r, const int64_t *__restrict__ seg_off, const int *__restrict__ seg,
                                             const int64_t *__restrict__ val_off, const Cfg &c, ESub esub, EEnt eent,
                                             int &nsub_row, int &spw_row)
    {
        const long long sb = seg_off[r], se = seg_off[r + 1];
        long long voff = val_off[r];
        int u = 0, rsp = 0;
        bool open = false;
        int eg = 0, elo = 0, ehi = 0, en = 0, espw = 0, eu0 = 0, ersp = 0, ework = 0;
        const int span0 = c.Ph * c.MS - c.Ph;
        auto xl_of = [&](int lo, int hi) { return span0 + hi - (lo & ~7); };
        for (long long s = sb; s < se; ++s) {
            const int ring = seg[3 * s], start = seg[3 * s + 1], len = seg[3 * s + 2];
            const long long myv = voff;
            if (len > 0) voff += len;
            if (len <= 0 || ring < 0 || ring >= c.Ho) continue;
            const int g = ring / c.G;
            int st0 = start % c.Wo;
            if (st0 < 0) st0 += c.Wo;
            for (int j0 = 0; j0 < len; j0 += c.CLW) {
                const int lj = min(c.CLW, len - j0);
                int stn = (int)(((long long)st0 + j0) % c.Wo);
                if (stn > c.Wo / 2) stn -= c.Wo;
                const int a_lo = floordiv_i(stn, c.P), a_hi = floordiv_i(stn + lj - 1, c.P);
                const int koff = -2 * floordiv_i(a_hi + 1, 2);
                const int kp = (c.Ph - a_lo - koff + 7) & ~7;
                const int spw = c.P * NSPLIT * nk_words(c.P, kp);
                if (open) {
                    const int nlo = min(elo, koff), nhi = max(ehi, koff + kp);
                    if (g != eg || en >= CR_SEGMAX || xl_of(nlo, nhi) > c.XLMAX || espw + spw > c.SPMAX) {
                        eent(eg, eu0, en, elo & ~7, xl_of(elo, ehi), espw, ersp, ework);
                        open = false;
                    } else {
                        elo = nlo;
                        ehi = nhi;
                    }
                }
                if (!open) {
                    open = true;
                    eg = g;
                    elo = koff;
                    ehi = koff + kp;
                    en = 0;
                    espw = 0;
                    eu0 = u;
                    ersp = rsp;
                    ework = 4;
                }
                esub(u, ring, stn, lj, myv + j0, koff, kp, espw, rsp);
                espw += spw;
                rsp += spw;
                ++en;
                ++u;
                ework += (kp >> 3) + 2;
            }
        }
        if (open) eent(eg, eu0, en, elo & ~7, xl_of(elo, ehi), espw, ersp, ework);
        nsub_row = u;
        spw_row = rsp;
    }

    __global__ void k_count(int nrows, const int64_t *__restrict__ seg_off, const int *__restrict__ seg,
                            const int64_t *__restrict__ val_off, Cfg c, int *__restrict__ bcount,
                            unsigned long long *__restrict__ gwork, int *__restrict__ row_nsub, int *__restrict__ row_spw)
    {
        const int r = blockIdx.x * blockDim.x + threadIdx.x;
        if (r >= nrows) return;
        const int b = r / c.RB;
        int ns, sw;
        walk_row(
            r, seg_off, seg, val_off, c, [&](int, int, int, int, long long, int, int, int, int) {},
            [&](int g, int, int, int, int, int, int, int work) {
                atomicAdd(&bcount[g * c.NB + b], 1);
                atomicAdd(&gwork[g], (unsigned long long)work);
            },
            ns, sw);
        row_nsub[r] = ns;
        row_spw[r] = sw;
    }

    __global__ void k_place(int nrows, const int64_t *__restrict__ seg_off, const int *__restrict__ seg,
                            const int64_t *__restrict__ val_off, const int *__restrict__ row_ker,
                            const int *__restrict__ row_lat, Cfg c, const int *__restrict__ bstart,
                            int *__restrict__ bcursor, const long long *__restrict__ row_suboff,
                            const long long *__restrict__ row_spoff, Entry *__restrict__ ents, int4 *__restrict__ hot,
                            BuildRec *__restrict__ brec)
    {
        const int r = blockIdx.x * blockDim.x + threadIdx.x;
        if (r >= nrows) return;
        const int b = r / c.RB;
        const int xrow = row_ker[r] * c.Hi + row_lat[r];
        const long long u0 = row_suboff[r], sp0 = row_spoff[r];
        int ns, sw;
        walk_row(
            r, seg_off, seg, val_off, c,
            [&](int u, int ring, int stn, int lj, long long voff, int koff, int kp, int spoff, int rsp) {
                hot[u0 + u] = make_int4(ring, koff, kp, spoff);
                BuildRec br;
                br.stn = stn;
                br.len = lj;
                br.koff = koff;
                br.kp = kp;
                br.voff = voff;
                br.spg = sp0 + rsp;
                brec[u0 + u] = br;
            },
            [&](int g, int eu0, int en, int WB, int XLb, int espw, int ersp, int) {
                const int pos = bstart[g * c.NB + b] + atomicAdd(&bcursor[g * c.NB + b], 1);
                Entry e;
                e.xrow = xrow;
                e.u0 = (int)(u0 + eu0);
                e.nsub = en;
                e.WB = WB;
                const int xlc = (XLb + (c.S - 1) * (c.Wi / c.S) + 7) & ~7;
                int q0 = WB % c.Wi;
                if (q0 < 0) q0 += c.Wi;
                const int l1 = min(xlc, c.Wi - q0), l2 = min(xlc - l1, c.Wi), l3 = xlc - l1 - l2;
                e.q0 = q0;
                e.spw = espw;
                e.len1 = l1;
                e.len2 = l2;
                e.spg = sp0 + ersp;
                e.len3 = l3;
                e.grp = g;
                ents[pos] = e;
            },
            ns, sw);
    }

    // Psi operand of one sub-arc: for residue b and split term s, word[i] = (R[i], R[i+1]) (fp16 pair),
    // R[i] = 2^sc * v_b(Ph - i - koff), 0 outside the sub-arc.
    template <int P>
    __global__ void k_build_sp(const long long *__restrict__ ntot, const BuildRec *__restrict__ brec,
                               const float *__restrict__ vals, const float *__restrict__ gscale,
                               uint32_t *__restrict__ spglob)
    {
        const long long u = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        const int lane = threadIdx.x & 31;
        if (u >= *ntot) return;
        const BuildRec br = brec[u];
        constexpr int Ph = 8 / P;
        const float scale = gscale[0];
        const int nk = nk_words(P, br.kp);
        uint32_t *dst = spglob + br.spg;
        const float *v = vals + br.voff;
        for (int i = lane; i < nk; i += 32) {
#pragma unroll
            for (int b = 0; b < P; ++b) {
                float y[2];
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int a = Ph - (i + e) - br.koff;
                    const int jj = P * a + b - br.stn;
                    y[e] = (jj >= 0 && jj < br.len) ? __ldg(v + jj) * scale : 0.f;
                }
#pragma unroll
                for (int s = 0; s < NSPLIT; ++s) {
                    const __half h0 = __float2half_rn(y[0]), h1 = __float2half_rn(y[1]);
                    y[0] -= __half2float(h0);
                    y[1] -= __half2float(h1);
                    __half2 w = __halves2half2(h0, h1);
                    dst[(b * NSPLIT + s) * nk + i] = *reinterpret_cast<uint32_t *>(&w);
                }
            }
        }
    }

    __global__ void k_absmax(const float *__restrict__ v, long long n, int *__restrict__ gmax)
    {
        __shared__ float wm[32];
        float m = 0.f;
        for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += (long long)gridDim.x * blockDim.x)
            m = fmaxf(m, fabsf(__ldg(v + i)));
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if ((threadIdx.x & 31) == 0) wm[threadIdx.x >> 5] = m;
        __syncthreads();
        if (threadIdx.x < 32) {
            m = (threadIdx.x < (blockDim.x >> 5)) ? wm[threadIdx.x] : 0.f;
            for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
            if (threadIdx.x == 0) atomicMax(gmax, __float_as_int(m));
        }
    }

    // Block-wide exclusive scan (1024 threads) of n values; out[n] = total.
    template <class TIn, class TOut>
    __device__ void block_scan_excl(const TIn *__restrict__ in, TOut *__restrict__ out, int n, TOut *wsum, TOut *carry)
    {
        const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
        if (tid == 0) *carry = 0;
        __syncthreads();
        for (int base = 0; base < n; base += 1024) {
            const int i = base + tid;
            const TOut v = (i < n) ? (TOut)in[i] : (TOut)0;
            TOut x = v;
            for (int o = 1; o < 32; o <<= 1) {
                const TOut y = __shfl_up_sync(0xffffffffu, x, o);
                if (lane >= o) x += y;
            }
            if (lane == 31) wsum[wid] = x;
            __syncthreads();
            if (wid == 0) {
                TOut w = wsum[lane];
                for (int o = 1; o < 32; o <<= 1) {
                    const TOut y = __shfl_up_sync(0xffffffffu, w, o);
                    if (lane >= o) w += y;
                }
                wsum[lane] = w;
            }
            __syncthreads();
            const TOut excl = x - v + (wid > 0 ? wsum[wid - 1] : (TOut)0) + *carry;
            if (i < n) out[i] = excl;
            __syncthreads();
            if (tid == 1023) *carry = excl + v;
            __syncthreads();
        }
        if (tid == 0) out[n] = *carry;
        __syncthreads();
    }

// Single CTA: scans, work ordering (heavy groups first, split for low channel-tile counts; the rest in ring
// order), psi scale.
#define SORT_MAX 4096
    __global__ void __launch_bounds__(1024)
        k_scan(Cfg c, const int *__restrict__ bcount, const unsigned long long *__restrict__ gwork,
               int *__restrict__ bstart, const int *__restrict__ row_nsub, long long *__restrict__ row_suboff,
               const int *__restrict__ row_spw, long long *__restrict__ row_spoff, int nrows,
               const int *__restrict__ gmax, float *__restrict__ gscale, int *__restrict__ ginfo,
               Item *__restrict__ items, int *__restrict__ gnorm, int nslots_cta)
    {
        __shared__ int wsum[32];
        __shared__ long long wsum64[32];
        __shared__ int carry;
        __shared__ long long carry64;
        __shared__ unsigned long long keys[SORT_MAX];
        __shared__ unsigned long long s_tot;
        __shared__ int s_nh, s_nsi, s_nhi;
        const int tid = threadIdx.x;
        const int NG = c.NG;
        block_scan_excl<int, int>(bcount, bstart, NG * c.NB, wsum, &carry);
        block_scan_excl<int, long long>(row_nsub, row_suboff, nrows, wsum64, &carry64);
        block_scan_excl<int, long long>(row_spw, row_spoff, nrows, wsum64, &carry64);
        if (tid == 0) {
            s_tot = 0;
            s_nh = 0;
            s_nsi = 0;
            s_nhi = 0;
        }
        __syncthreads();
        const bool sortable = NG <= SORT_MAX;
        int n2 = 1;
        while (n2 < NG) n2 <<= 1;
        if (sortable) {
            for (int i = tid; i < n2; i += 1024) {
                if (i < NG) {
                    const unsigned long long w = gwork[i];
                    const unsigned int w32 = w > 0xffffffffull ? 0xffffffffu : (unsigned int)w;
                    keys[i] = ((unsigned long long)(~w32) << 32) | (unsigned int)i;
                } else {
                    keys[i] = ~0ull;
                }
            }
            __syncthreads();
            for (int k = 2; k <= n2; k <<= 1) {
                for (int j = k >> 1; j > 0; j >>= 1) {
                    for (int i = tid; i < n2; i += 1024) {
                        const int ixj = i ^ j;
                        if (ixj > i) {
                            const unsigned long long a = keys[i], b = keys[ixj];
                            const bool asc = ((i & k) == 0);
                            if ((a > b) == asc) {
                                keys[i] = b;
                                keys[ixj] = a;
                            }
                        }
                    }
                    __syncthreads();
                }
            }
            unsigned long long part = 0;
            for (int i = tid; i < NG; i += 1024) part += (unsigned long long)(~(unsigned int)(keys[i] >> 32));
            for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o);
            if ((tid & 31) == 0 && part) atomicAdd(&s_tot, part);
            __syncthreads();
            // heavy: more than twice the mean group work (at most 1024 of them)
            const unsigned long long tot = s_tot;
            int cnt = 0;
            for (int i = tid; i < min(NG, 1024); i += 1024)
                cnt += ((unsigned long long)(~(unsigned int)(keys[i] >> 32)) * NG > 2 * tot) ? 1 : 0;
            for (int o = 16; o > 0; o >>= 1) cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
            if ((tid & 31) == 0 && cnt) atomicAdd(&s_nh, cnt);
            __syncthreads();
            const int nh = s_nh;
            // split decision (warp 0, heaviest 32 heavy groups): only when there are few channel tiles
            if (tid < 32) {
                int g = 0, e0 = 0, cntg = 0, ns = 1;
                if (tid < min(nh, 32)) {
                    g = (int)(keys[tid] & 0xffffffffu);
                    e0 = bstart[g * c.NB];
                    cntg = bstart[(g + 1) * c.NB] - e0;
                    if (c.ntiles <= 32 && tot > 0) {
                        const double target = (double)tot * c.ntiles / (2.0 * nslots_cta);
                        ns = (int)ceil((double)(~(unsigned int)(keys[tid] >> 32)) / target);
                        ns = max(1, min(min(ns, MAXSPLIT), cntg));
                    }
                }
                int sl = (ns > 1) ? ns : 0, inc = sl;
                for (int o = 1; o < 32; o <<= 1) {
                    const int y = __shfl_up_sync(0xffffffffu, inc, o);
                    if (tid >= o) inc += y;
                }
                if (inc > MAXSI) {
                    ns = 1;
                    sl = 0;
                }
                // recompute with the rejected ones removed
                inc = sl;
                for (int o = 1; o < 32; o <<= 1) {
                    const int y = __shfl_up_sync(0xffffffffu, inc, o);
                    if (tid >= o) inc += y;
                }
                const int slot0 = inc - sl;
                int sp = (ns > 1) ? 1 : 0, sinc = sp;
                for (int o = 1; o < 32; o <<= 1) {
                    const int y = __shfl_up_sync(0xffffffffu, sinc, o);
                    if (tid >= o) sinc += y;
                }
                const int sid = sinc - sp;
                int nit = (tid < min(nh, 32)) ? ns : 0, pinc = nit;
                for (int o = 1; o < 32; o <<= 1) {
                    const int y = __shfl_up_sync(0xffffffffu, pinc, o);
                    if (tid >= o) pinc += y;
                }
                const int pos = pinc - nit;
                for (int k = 0; k < nit; ++k) {
                    Item it;
                    it.g = g;
                    it.e0 = e0 + (int)((long long)k * cntg / ns);
                    it.e1 = e0 + (int)((long long)(k + 1) * cntg / ns);
                    it.slot = (ns > 1) ? slot0 : -1;
                    it.ns = ns;
                    it.sid = (ns > 1) ? sid : -1;
                    it.pad0 = k;
                    it.pad1 = 0;
                    items[pos + k] = it;
                }
                if (tid == 31) s_nhi = pinc;
            }
            __syncthreads();
            const int n32 = s_nhi;
            for (int i = 32 + tid; i < nh; i += 1024) {
                const int g = (int)(keys[i] & 0xffffffffu);
                Item it;
                it.g = g;
                it.e0 = bstart[g * c.NB];
                it.e1 = bstart[(g + 1) * c.NB];
                it.slot = -1;
                it.ns = 1;
                it.sid = -1;
                it.pad0 = 0;
                it.pad1 = 0;
                items[n32 + i - 32] = it;
            }
            if (tid == 0) {
                ginfo[0] = nh;
                ginfo[1] = n32 + max(nh - 32, 0);
            }
        } else {
            if (tid == 0) {
                ginfo[0] = 0;
                ginfo[1] = 0;
            }
        }
        __syncthreads();
        // normal groups (not heavy) in ring order: compact with a block scan over flags
        {
            __shared__ unsigned char heavy[SORT_MAX];
            const int nh = sortable ? ginfo[0] : 0;
            for (int i = tid; i < min(NG, SORT_MAX); i += 1024) heavy[i] = 0;
            __syncthreads();
            if (sortable)
                for (int i = tid; i < nh; i += 1024) heavy[(int)(keys[i] & 0xffffffffu)] = 1;
            __syncthreads();
            __shared__ int ccarry;
            if (tid == 0) ccarry = 0;
            __syncthreads();
            for (int base = 0; base < NG; base += 1024) {
                const int i = base + tid;
                const int f = (i < NG && !(i < SORT_MAX && heavy[i])) ? 1 : 0;
                const unsigned bal = __ballot_sync(0xffffffffu, f);
                const int lane = tid & 31, wid = tid >> 5;
                if (lane == 0) wsum[wid] = __popc(bal);
                __syncthreads();
                if (wid == 0) {
                    int w = wsum[lane];
                    for (int o = 1; o < 32; o <<= 1) {
                        const int y = __shfl_up_sync(0xffffffffu, w, o);
                        if (lane >= o) w += y;
                    }
                    wsum[lane] = w;
                }
                __syncthreads();
                const int pos = ccarry + (wid > 0 ? wsum[wid - 1] : 0) + __popc(bal & ((1u << lane) - 1u));
                if (f) gnorm[pos] = i;
                __syncthreads();
                if (tid == 0) ccarry += wsum[31];
                __syncthreads();
            }
            if (tid == 0) ginfo[2] = ccarry;
        }
        if (tid == 0) {
            const float m = __int_as_float(*gmax);
            int sc = 0;
            if (m > 0.f && m <= 3.0e38f) {
                int e;
                frexpf(m, &e); // m < 2^e
                sc = 15 - e;   // max |psi| * 2^sc < 2^15
                sc = max(-100, min(100, sc));
            }
            gscale[0] = ldexpf(1.f, sc);
            gscale[1] = ldexpf(1.f, -sc);
        }
    }

    // ----------------------------------------------------------------------------------------------------------
    // Tensor-core main kernel
    // ----------------------------------------------------------------------------------------------------------
    __device__ __forceinline__ unsigned smem_u32(const void *p) { return (unsigned)__cvta_generic_to_shared(p); }

    __device__ __forceinline__ void mbar_init(uint64_t *bar, unsigned count)
    {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count));
    }
    __device__ __forceinline__ void mbar_expect_tx(uint64_t *bar, unsigned bytes)
    {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes)
                     : "memory");
    }
    __device__ __forceinline__ void mbar_arrive(uint64_t *bar)
    {
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
    }
    __device__ __forceinline__ void mbar_wait(uint64_t *bar, unsigned parity)
    {
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "WAIT_%=:\n"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                     "@!p bra WAIT_%=;\n"
                     "}\n" ::"r"(smem_u32(bar)),
                     "r"(parity)
                     : "memory");
    }
    __device__ __forceinline__ void bulk_g2s(void *dst, const void *src, unsigned bytes, uint64_t *bar)
    {
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
                         smem_u32(dst)),
                     "l"(src), "r"(bytes), "r"(smem_u32(bar))
                     : "memory");
    }

    __device__ __forceinline__ void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint32_t b0, uint32_t b1)
    {
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%0,%1,%2,%3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    __device__ __forceinline__ void mma1688(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0)
    {
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a1), "r"(b0));
    }

    // One sub-arc for one warp: NP k16 steps (+ optional k8 tail).
    //   A(mu, kb) word offset (from pa0/pa1) = CS2*mu + 4*kb;  B(k16 step p, split s) words: pb[s*NK + 16p + 8j].
    //   Identical A addresses for different (mu, kb) are merged by the compiler (register reuse along mu).
    template <int P>
    __device__ __forceinline__ void arc_mma(float (&acc)[NMU][4], const uint32_t *__restrict__ pa0,
                                            const uint32_t *__restrict__ pa1, const uint32_t *__restrict__ pb, int nk,
                                            int np, bool tail)
    {
        constexpr int CS2 = 4 / P; // words per mu step
        int p = 0;
#if PAIR2
        for (; p + 2 <= np; p += 2) {
            uint32_t b[2][NSPLIT][2];
#pragma unroll
            for (int q = 0; q < 2; ++q)
#pragma unroll
                for (int s = 0; s < NSPLIT; ++s)
#pragma unroll
                    for (int j = 0; j < 2; ++j) b[q][s][j] = pb[s * nk + 16 * (p + q) + 8 * j];
            const uint32_t *x0 = pa0 + 8 * p;
            const uint32_t *x1 = pa1 + 8 * p;
#pragma unroll
            for (int mg = 0; mg < NMU; mg += 3) {
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    uint32_t a[3][4];
#pragma unroll
                    for (int u = 0; u < 3; ++u) {
                        const int w0 = CS2 * (mg + u) + 8 * q, w1 = w0 + 4;
                        a[u][0] = x0[w0];
                        a[u][1] = x1[w0];
                        a[u][2] = x0[w1];
                        a[u][3] = x1[w1];
                    }
#pragma unroll
                    for (int s = 0; s < NSPLIT; ++s)
#pragma unroll
                        for (int u = 0; u < 3; ++u)
                            mma16816(acc[mg + u], a[u][0], a[u][1], a[u][2], a[u][3], b[q][s][0], b[q][s][1]);
                }
            }
        }
#endif
        for (; p < np; ++p) {
            uint32_t b[NSPLIT][2];
#pragma unroll
            for (int s = 0; s < NSPLIT; ++s)
#pragma unroll
                for (int j = 0; j < 2; ++j) b[s][j] = pb[s * nk + 16 * p + 8 * j];
            const uint32_t *x0 = pa0 + 8 * p;
            const uint32_t *x1 = pa1 + 8 * p;
#pragma unroll
            for (int mg = 0; mg < NMU; mg += 3) {
                uint32_t a[3][4];
#pragma unroll
                for (int u = 0; u < 3; ++u) {
                    const int w0 = CS2 * (mg + u), w1 = w0 + 4;
                    a[u][0] = x0[w0];
                    a[u][1] = x1[w0];
                    a[u][2] = x0[w1];
                    a[u][3] = x1[w1];
                }
#pragma unroll
                for (int s = 0; s < NSPLIT; ++s)
#pragma unroll
                    for (int u = 0; u < 3; ++u)
                        mma16816(acc[mg + u], a[u][0], a[u][1], a[u][2], a[u][3], b[s][0], b[s][1]);
            }
        }
        if (tail) {
            uint32_t b[NSPLIT];
#pragma unroll
            for (int s = 0; s < NSPLIT; ++s) b[s] = pb[s * nk + 16 * np];
            const uint32_t *x0 = pa0 + 8 * np;
            const uint32_t *x1 = pa1 + 8 * np;
#pragma unroll
            for (int mg = 0; mg < NMU; mg += 3) {
                uint32_t a[3][2];
#pragma unroll
                for (int u = 0; u < 3; ++u) {
                    const int w0 = CS2 * (mg + u);
                    a[u][0] = x0[w0];
                    a[u][1] = x1[w0];
                }
#pragma unroll
                for (int s = 0; s < NSPLIT; ++s)
#pragma unroll
                    for (int u = 0; u < 3; ++u) mma1688(acc[mg + u], a[u][0], a[u][1], b[s]);
            }
        }
    }

    template <int P>
    __global__ void __launch_bounds__(32 * (MAXMW + NPW), 1)
        k_disco_tc(const __half *__restrict__ inp, const uint32_t *__restrict__ spglob, const int4 *__restrict__ hot,
                   const Entry *__restrict__ ents, const int *__restrict__ bstart, const int *__restrict__ ginfo,
                   const Item *__restrict__ items, const int *__restrict__ gnorm, const float *__restrict__ gscale,
                   __half *__restrict__ out, float *__restrict__ scratch, int *__restrict__ scnt, const Cfg c)
    {
        extern __shared__ __align__(128) unsigned char smem[];
        __shared__ __align__(8) uint64_t full_bar[MAXST];
        __shared__ __align__(8) uint64_t empty_bar[MAXST];
        __shared__ int s_last;
        constexpr int Ph = 8 / P;
        const int tid = threadIdx.x;
        const int lane = tid & 31, warp = tid >> 5;
        const int nwc = c.G * c.NWL;
        const int nst = c.nst;

        // work item: heavy items (tile fastest) first, then normal groups tile-major in ring order
        const int nhi = ginfo[1];
        int tile, gr, e0, n, slot = -1, nsplit = 1, sid = -1, kidx = 0;
        if ((int)blockIdx.x < nhi * c.ntiles) {
            const Item it = items[blockIdx.x / c.ntiles];
            tile = blockIdx.x % c.ntiles;
            gr = it.g;
            e0 = it.e0;
            n = it.e1 - it.e0;
            if (it.ns > 1) {
                slot = it.slot;
                nsplit = it.ns;
                sid = it.sid;
                kidx = it.pad0;
            }
        } else {
            const int r = blockIdx.x - nhi * c.ntiles, nn = ginfo[2];
            if (r >= nn * c.ntiles) return; // spare CTA (grid sized for the maximum number of items)
            tile = r / nn;
            gr = gnorm[r % nn];
            e0 = bstart[gr * c.NB];
            n = bstart[(gr + 1) * c.NB] - e0;
        }

        // padded channel rows (channel >= BC) are never loaded: zero them once in every stage
        const int c0 = tile * c.CT;
#pragma unroll 1
        for (int i = max(0, c.BC - c0); i < c.CT; ++i) {
#pragma unroll 1
            for (int st = 0; st < nst; ++st) {
                uint32_t *rowp = reinterpret_cast<uint32_t *>(smem + st * c.stage_bytes) + i * c.CHS;
                for (int w = tid; w < c.CHS; w += blockDim.x) rowp[w] = 0u;
            }
        }
        if (tid == 0) {
            for (int s = 0; s < nst; ++s) {
                mbar_init(&full_bar[s], NPW);
                mbar_init(&empty_bar[s], nwc);
            }
            asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        }
        __syncthreads();

        if (warp >= nwc) {
            // ---------------- producers: warp pw copies channels pw, pw + NPW, ... and one metadata block ----------------
            const int pw = warp - nwc;
            const size_t cstride = (size_t)c.K * c.Hi * c.Wi;
            // lane j of producer warp pw owns channel row pw + NPW * j
            const int chl = pw + NPW * lane;
            const int ch = c0 + chl;
            const bool rvalid = (chl < c.CT) && (ch < c.BC);
            int nmine = 0;
            for (int k = pw; k < c.CT; k += NPW) nmine += (c0 + k < c.BC) ? 1 : 0;
            const __half *chbase = inp + (rvalid ? (size_t)ch * cstride : 0);
            __half *rowdst = reinterpret_cast<__half *>(smem) + 2 * chl * c.CHS;
            // entries are fetched 32 at a time (lane j holds entry base + j), one batch ahead
            Entry Ecur {}, Enxt {};
            if (lane < n) Ecur = ents[e0 + lane];
            if (32 + lane < n) Enxt = ents[e0 + 32 + lane];
            int s = 0, rnd = 0;
            for (int i = 0; i < n; ++i) {
                const int j = i & 31;
                if (j == 0 && i > 0) {
                    Ecur = Enxt;
                    if (i + 32 + lane < n) Enxt = ents[e0 + i + 32 + lane];
                }
                const int xrow = __shfl_sync(0xffffffffu, Ecur.xrow, j);
                const int q0 = __shfl_sync(0xffffffffu, Ecur.q0, j);
                const int len1 = __shfl_sync(0xffffffffu, Ecur.len1, j);
                const int len2 = __shfl_sync(0xffffffffu, Ecur.len2, j);
                const int len3 = __shfl_sync(0xffffffffu, Ecur.len3, j);
                int meta = 0;
                const void *msrc = nullptr;
                int moff = 0;
                if (pw == 1 % NPW) {
                    meta = 4 * __shfl_sync(0xffffffffu, Ecur.spw, j);
                    msrc = spglob + __shfl_sync(0xffffffffu, Ecur.spg, j);
                    moff = c.off_ps;
                } else if (pw == 2 % NPW) {
                    meta = 16 * __shfl_sync(0xffffffffu, Ecur.nsub, j);
                    msrc = hot + __shfl_sync(0xffffffffu, Ecur.u0, j);
                    moff = c.off_hot;
                } else if (pw == 3 % NPW) {
                    meta = (int)sizeof(Entry);
                    msrc = ents + e0 + i;
                    moff = c.off_ent;
                }
                if (rnd > 0) mbar_wait(&empty_bar[s], (rnd - 1) & 1);
                const int sofs = s * c.stage_bytes;
                if (lane == 0) mbar_expect_tx(&full_bar[s], (unsigned)(2 * nmine * (len1 + len2 + len3) + meta));
                __syncwarp();
                if (rvalid) {
                    const __half *src = chbase + (size_t)xrow * c.Wi;
                    __half *dst = rowdst + sofs / 2;
                    bulk_g2s(dst, src + q0, (unsigned)len1 * 2, &full_bar[s]);
                    if (len2 > 0) bulk_g2s(dst + len1, src, (unsigned)len2 * 2, &full_bar[s]);
                    if (len3 > 0) bulk_g2s(dst + len1 + len2, src, (unsigned)len3 * 2, &full_bar[s]);
                }
                if (lane == 31 && meta > 0) bulk_g2s(smem + sofs + moff, msrc, (unsigned)meta, &full_bar[s]);
                if (++s == nst) {
                    s = 0;
                    ++rnd;
                }
            }
            return;
        }

        // ---------------- consumers ----------------
        const int rs = warp / c.NWL, lw = warp % c.NWL;
        const int mu0 = lw * NMU;
        const int myring = gr * c.G + rs;
        const float inv_scale = gscale[1];
        const int g8 = lane >> 2, t4 = lane & 3;
        const int bsel = (P == 2) ? (g8 & 1) : 0, phl = (P == 2) ? (g8 >> 1) : g8;

        float acc[NMU][4];
#pragma unroll
        for (int j = 0; j < NMU; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[j][q] = 0.f;

        // lane row bases (words): row g8 and g8 + 8, + t4
        const int rw0 = c.rowch[g8] * c.CHS + c.rowsp[g8] * (c.Wi / c.S / 2) + t4;
        const int rw1 = c.rowch[g8 + 8] * c.CHS + c.rowsp[g8 + 8] * (c.Wi / c.S / 2) + t4;
        const int bofs = bsel * NSPLIT; // residue array (in units of nk words)
        const int blane = Ph - phl + 2 * t4;

        int s = 0, rnd = 0;
        for (int i = 0; i < n; ++i) {
            mbar_wait(&full_bar[s], rnd & 1);
            const uint32_t *sx = reinterpret_cast<const uint32_t *>(smem + s * c.stage_bytes);
            const uint32_t *sps = reinterpret_cast<const uint32_t *>(smem + s * c.stage_bytes + c.off_ps);
            const Entry *E = reinterpret_cast<const Entry *>(smem + s * c.stage_bytes + c.off_ent);
            const int nsub = E->nsub, WB = E->WB;
            const int4 *hs = reinterpret_cast<const int4 *>(smem + s * c.stage_bytes + c.off_hot);
            for (int k = 0; k < nsub; ++k) {
                const int4 h = hs[k];
                if (h.x != myring) continue;
                const int koff = h.y, kp = h.z;
                const int nk = nk_words(P, kp);
                const int wa = (Ph * mu0 + koff - WB) >> 1; // even
                arc_mma<P>(acc, sx + rw0 + wa, sx + rw1 + wa, sps + h.w + bofs * nk + blane, nk, kp >> 4, (kp & 8) != 0);
            }
            // order this warp's generic-proxy reads of the stage before the producer's next async-proxy (TMA) writes
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            __syncwarp();
            if (lane == 0) mbar_arrive(&empty_bar[s]);
            if (++s == nst) {
                s = 0;
                ++rnd;
            }
        }

        // output rows of this lane
        const int chA = c0 + c.rowch[g8], chB = c0 + c.rowch[g8 + 8];
        const int oA = c.rowsp[g8] * (c.Wo / c.S), oB = c.rowsp[g8 + 8] * (c.Wo / c.S);
        const int ospan = c.Wo / c.S;

        if (slot >= 0) {
            // split work item: write the partial sums to this split's scratch slice; the last split of the
            // (group, tile) sums all slices in fixed order and stores
            {
                float *base = scratch + (((size_t)(slot + kidx) * c.ntiles + tile) * c.G + rs) * 16 * (size_t)ospan;
#pragma unroll
                for (int mu = 0; mu < NMU; ++mu) {
                    const int o = 8 * (mu0 + mu) + 2 * t4;
                    __stcg(reinterpret_cast<float2 *>(base + (size_t)g8 * ospan + o),
                           make_float2(acc[mu][0], acc[mu][1]));
                    __stcg(reinterpret_cast<float2 *>(base + (size_t)(g8 + 8) * ospan + o),
                           make_float2(acc[mu][2], acc[mu][3]));
                }
            }
            __threadfence();
            asm volatile("bar.sync 1, %0;\n" ::"r"(nwc * 32) : "memory");
            if (tid == 0) s_last = (atomicAdd(&scnt[sid * c.ntiles + tile], 1) == nsplit - 1);
            asm volatile("bar.sync 1, %0;\n" ::"r"(nwc * 32) : "memory");
            if (!s_last) return;
            __threadfence();
#pragma unroll
            for (int mu = 0; mu < NMU; ++mu)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[mu][q] = 0.f;
            for (int k = 0; k < nsplit; ++k) {
                const float *base = scratch + (((size_t)(slot + k) * c.ntiles + tile) * c.G + rs) * 16 * (size_t)ospan;
#pragma unroll
                for (int mu = 0; mu < NMU; ++mu) {
                    const int o = 8 * (mu0 + mu) + 2 * t4;
                    const float2 v0 = __ldcg(reinterpret_cast<const float2 *>(base + (size_t)g8 * ospan + o));
                    const float2 v1 = __ldcg(reinterpret_cast<const float2 *>(base + (size_t)(g8 + 8) * ospan + o));
                    acc[mu][0] += v0.x;
                    acc[mu][1] += v0.y;
                    acc[mu][2] += v1.x;
                    acc[mu][3] += v1.y;
                }
            }
        }

        // Epilogue: scale back, round once to fp16, store.
        if (myring >= c.Ho) return;
        __half *rowA = (chA < c.BC) ? out + ((size_t)chA * c.Ho + myring) * c.Wo + oA : nullptr;
        __half *rowB = (chB < c.BC) ? out + ((size_t)chB * c.Ho + myring) * c.Wo + oB : nullptr;
#pragma unroll
        for (int mu = 0; mu < NMU; ++mu) {
            const int o = 8 * (mu0 + mu) + 2 * t4;
            if (rowA)
                *reinterpret_cast<__half2 *>(rowA + o)
                    = __floats2half2_rn(acc[mu][0] * inv_scale, acc[mu][1] * inv_scale);
            if (rowB)
                *reinterpret_cast<__half2 *>(rowB + o)
                    = __floats2half2_rn(acc[mu][2] * inv_scale, acc[mu][3] * inv_scale);
        }
    }

    // ----------------------------------------------------------------------------------------------------------
    // General path: fp32 atomic scatter (any pscale, any Wi, any arc length), then narrowing.
    // ----------------------------------------------------------------------------------------------------------
    __global__ void k_fb_prep(int nrows, const int64_t *__restrict__ seg_off, const int64_t *__restrict__ val_off,
                              const int *__restrict__ seg, long long *__restrict__ svoff, int *__restrict__ srow)
    {
        const int r = blockIdx.x * blockDim.x + threadIdx.x;
        if (r >= nrows) return;
        long long v = val_off[r];
        for (long long s = seg_off[r]; s < seg_off[r + 1]; ++s) {
            svoff[s] = v;
            srow[s] = r;
            const int len = seg[3 * s + 2];
            if (len > 0) v += len;
        }
    }

    __global__ void k_fb_scatter(const __half *__restrict__ inp, const int *__restrict__ row_ker,
                                 const int *__restrict__ row_lat, const int *__restrict__ seg,
                                 const float *__restrict__ vals, const long long *__restrict__ svoff,
                                 const int *__restrict__ srow, float *__restrict__ out32, long long nseg, int BC, int K,
                                 int Hi, int Wi, int Ho, int Wo)
    {
        const int P = Wo / Wi;
        for (long long s = blockIdx.x; s < nseg; s += gridDim.x) {
            const int r = srow[s];
            if (r < 0) continue;
            const int ring = seg[3 * s], start = seg[3 * s + 1];
            const int len = seg[3 * s + 2];
            if (len <= 0 || ring < 0 || ring >= Ho) continue;
            const long long voff = svoff[s];
            int st = start % Wo;
            if (st < 0) st += Wo;
            for (int bc = blockIdx.y; bc < BC; bc += gridDim.y) {
                const __half *x = inp + (((size_t)bc * K + row_ker[r]) * Hi + row_lat[r]) * Wi;
                float *o = out32 + ((size_t)bc * Ho + ring) * Wo;
                for (int q = threadIdx.x; q < Wi; q += blockDim.x) {
                    const float xv = __half2float(x[q]);
                    int w = (int)(((long long)st + (long long)q * P) % Wo);
                    for (int j = 0; j < len; ++j) {
                        atomicAdd(o + w, vals[voff + j] * xv);
                        if (++w == Wo) w = 0;
                    }
                }
            }
        }
    }

    __global__ void k_fb_narrow(const float *__restrict__ out32, __half *__restrict__ out, long long n)
    {
        for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += (long long)gridDim.x * blockDim.x)
            out[i] = __float2half_rn(out32[i]);
    }

    // ----------------------------------------------------------------------------------------------------------
    // Host
    // ----------------------------------------------------------------------------------------------------------
    static inline size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

    // Choose the tensor-core configuration; returns false if the fast path does not apply.
    static bool make_cfg(Cfg &c, int BC, int K, int Hi, int Wi, int Ho, int Wo, int NROWS)
    {
        if (Wi <= 0 || Wo % Wi != 0) return false;
        const int P = Wo / Wi;
        if (P != 1 && P != 2) return false;
        if (Wi % 8 != 0) return false;
        if ((long long)K * Hi * Wi >= (1LL << 31) || (long long)Ho * Wo >= (1LL << 31)) return false;
        const int Ph = 8 / P;
        int bestS = -1;
        double bestCost = 0;
        for (int S = 1; S <= 16; S <<= 1) {
            if (Wi % S) continue;
            const int sw = Wi / S;
            if (sw % 4 || sw % Ph) continue;
            const int MS = sw / Ph;
            if (MS % NMU) continue;
            const int NWL = MS / NMU;
            if (NWL > MAXMW) continue;
            const int CT = 16 / S;
            const int G = std::max(1, MAXMW / NWL);
            // padded channels x a ring re-read proxy (input rows are restaged once per ring group)
            const double cost = (double)((BC + CT - 1) / CT * CT) * (1.0 + 1.0 / G);
            if (bestS < 0 || cost < bestCost) {
                bestS = S;
                bestCost = cost;
            }
        }
        if (bestS < 0) return false;
        memset(&c, 0, sizeof(c));
        c.P = P;
        c.Ph = Ph;
        c.Wi = Wi;
        c.Wo = Wo;
        c.Ho = Ho;
        c.Hi = Hi;
        c.K = K;
        c.BC = BC;
        c.S = bestS;
        c.CT = 16 / bestS;
        c.MS = Wi / bestS / Ph;
        c.NWL = c.MS / NMU;
        c.G = std::max(1, MAXMW / c.NWL);
        c.NG = (Ho + c.G - 1) / c.G;
        c.NB = std::min(NBMAX, std::max(1, NROWS));
        c.RB = (NROWS + c.NB - 1) / c.NB;
        c.CLW = P * (KPMAX - Ph - 1);
        // MMA rows: (channel, split) pairs ordered so that rows 0..7 share one 16-byte phase and rows 8..15 the other
        const int sw = Wi / c.S;
        int nr = 0;
        for (int pass = 0; pass < 2; ++pass)
            for (int sp = 0; sp < c.S; ++sp) {
                const int sig = (sp * sw) % 8;
                if ((sig != 0) != (pass == 1)) continue;
                for (int ch = 0; ch < c.CT; ++ch) {
                    c.rowch[nr] = ch;
                    c.rowsp[nr] = sp;
                    ++nr;
                }
            }
        c.sig_lo = (c.rowsp[0] * sw) % 8;
        c.sig_hi = (c.rowsp[8] * sw) % 8;
        for (int i = 0; i < 8; ++i)
            if ((c.rowsp[i] * sw) % 8 != c.sig_lo || (c.rowsp[i + 8] * sw) % 8 != c.sig_hi) return false;
        // stage layout: one contiguous window row per channel; split s of the row starts s * sw / 2 words in
        const int xlmax = sw + KPMAX + 8; // max split window (halfs)
        c.XLMAX = xlmax;
        const int need = (xlmax + (c.S - 1) * sw + 7) / 8 * 4; // words
        c.CHS = -1;
        for (int cand = (need + 3) / 4 * 4; cand < need + 64; cand += 4) {
            bool ok = true;
            for (int h = 0; h < 2 && ok; ++h) {
                unsigned used = 0;
                for (int i = 8 * h; i < 8 * h + 8; ++i) {
                    const int base = (c.rowch[i] * cand + c.rowsp[i] * (sw / 2)) & 31;
                    for (int t = 0; t < 4; ++t) {
                        const unsigned bit = 1u << ((base + t) & 31);
                        if (used & bit) ok = false;
                        used |= bit;
                    }
                }
            }
            if (ok) {
                c.CHS = cand;
                break;
            }
        }
        if (c.CHS < 0) return false;
        c.SPMAX = P * NSPLIT * (KPMAX + 16 + 128);
        c.off_ps = (int)align_up((size_t)c.CT * c.CHS * 4, 128);
        c.off_hot = (int)align_up(c.off_ps + (size_t)c.SPMAX * 4, 128);
        c.off_ent = (int)align_up(c.off_hot + (size_t)CR_SEGMAX * 16, 16);
        c.stage_bytes = (int)align_up(c.off_ent + sizeof(Entry), 128);
        const int budget = 216 * 1024;
        c.nst = std::min(MAXST, budget / c.stage_bytes);
        if (c.nst < 2) return false;
        c.ntiles = (BC + c.CT - 1) / c.CT;
        return true;
    }

    bool RunCR(ffi::TensorView inp, ffi::TensorView row_ker, ffi::TensorView row_lat, ffi::TensorView seg_off,
               ffi::TensorView seg, ffi::TensorView val_off, ffi::TensorView vals, int64_t nlat_out, int64_t nlon_out,
               ffi::TensorView out)
    {
        const int B = (int)inp.size(0), C = (int)inp.size(1), K = (int)inp.size(2), Hi = (int)inp.size(3),
                  Wi = (int)inp.size(4);
        const long long BCl = (long long)B * C;
        const int NROWS = (int)row_ker.size(0);
        const long long NSEG = seg.size(0);
        const long long NNZ = vals.size(0);
        const int Ho = (int)nlat_out, Wo = (int)nlon_out;
        cudaStream_t stream
            = static_cast<cudaStream_t>(TVMFFIEnvGetStream(inp.device().device_type, inp.device().device_id));

        const __half *d_inp = static_cast<const __half *>(inp.data_ptr());
        const int *d_rk = static_cast<const int *>(row_ker.data_ptr());
        const int *d_rl = static_cast<const int *>(row_lat.data_ptr());
        const int64_t *d_so = static_cast<const int64_t *>(seg_off.data_ptr());
        const int *d_seg = static_cast<const int *>(seg.data_ptr());
        const int64_t *d_vo = static_cast<const int64_t *>(val_off.data_ptr());
        const float *d_vals = static_cast<const float *>(vals.data_ptr());
        __half *d_out = static_cast<__half *>(out.data_ptr());

        if (BCl <= 0 || Ho <= 0 || Wo <= 0) return true;
        const int BC = (int)BCl;

        Cfg c;
        bool tc = NROWS > 0 && NSEG > 0 && NSEG < (1LL << 30) && BCl < (1LL << 30)
            && make_cfg(c, BC, K, Hi, Wi, Ho, Wo, NROWS);
        if (tc && (long long)c.NG * c.NB > (1 << 22)) tc = false;

        if (tc) {
            const int NG = c.NG, NB = c.NB;
            int nsm = 132;
            {
                int dev = 0;
                cudaGetDevice(&dev);
                cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
            }
            const int nslots_cta = nsm;
            // sub-arc bound: every arc yields ceil(len / CLW) sub-arcs; len <= NNZ in total
            const long long nsub_max = NSEG + NNZ / c.CLW + 1;
            // per sub-arc: P * NSPLIT * nk words <= 2 * len + 66 * P
            const long long spw_max = 2 * NNZ + 66LL * c.P * nsub_max + 64;
            const bool use_split = c.ntiles <= 32;
            const size_t ospan = (size_t)Wo / c.S;
            const size_t scratch_floats = use_split ? (size_t)MAXSI * c.ntiles * c.G * 16 * ospan : 0;

            size_t o = 0;
            auto take = [&](size_t bytes) {
                size_t r = o;
                o = align_up(o + bytes, 256);
                return r;
            };
            const size_t o_bcount = take((size_t)NG * NB * 4);
            const size_t o_bcursor = take((size_t)NG * NB * 4);
            const size_t o_gwork = take((size_t)NG * 8);
            const size_t o_gmax = take(16);
            const size_t o_scnt = take((size_t)MAXSPLIT * 32 * c.ntiles * 4);
            const size_t zero_end = o;
            const size_t o_bstart = take(((size_t)NG * NB + 1) * 4);
            const size_t o_rnsub = take((size_t)NROWS * 4);
            const size_t o_rspw = take((size_t)NROWS * 4);
            const size_t o_rsuboff = take(((size_t)NROWS + 1) * 8);
            const size_t o_rspoff = take(((size_t)NROWS + 1) * 8);
            const size_t o_ginfo = take(64);
            const size_t o_gscale = take(16);
            const size_t o_items = take((size_t)(NG + MAXSI) * sizeof(Item));
            const size_t o_gnorm = take((size_t)NG * 4);
            const size_t o_ents = take((size_t)nsub_max * sizeof(Entry));
            const size_t o_hot = take((size_t)nsub_max * 16);
            const size_t o_brec = take((size_t)nsub_max * sizeof(BuildRec));
            const size_t o_scratch = take(scratch_floats * 4);
            const size_t o_sp = take((size_t)spw_max * 4);
            const size_t total = o;
            unsigned char *ws = nullptr;
            cudaMallocAsync((void **)&ws, total, stream);
            cudaMemsetAsync(ws, 0, zero_end, stream);
            int *d_bcount = reinterpret_cast<int *>(ws + o_bcount);
            int *d_bcursor = reinterpret_cast<int *>(ws + o_bcursor);
            unsigned long long *d_gwork = reinterpret_cast<unsigned long long *>(ws + o_gwork);
            int *d_gmax = reinterpret_cast<int *>(ws + o_gmax);
            int *d_scnt = reinterpret_cast<int *>(ws + o_scnt);
            int *d_bstart = reinterpret_cast<int *>(ws + o_bstart);
            int *d_rnsub = reinterpret_cast<int *>(ws + o_rnsub);
            int *d_rspw = reinterpret_cast<int *>(ws + o_rspw);
            long long *d_rsuboff = reinterpret_cast<long long *>(ws + o_rsuboff);
            long long *d_rspoff = reinterpret_cast<long long *>(ws + o_rspoff);
            int *d_ginfo = reinterpret_cast<int *>(ws + o_ginfo);
            float *d_gscale = reinterpret_cast<float *>(ws + o_gscale);
            Item *d_items = reinterpret_cast<Item *>(ws + o_items);
            int *d_gnorm = reinterpret_cast<int *>(ws + o_gnorm);
            Entry *d_ents = reinterpret_cast<Entry *>(ws + o_ents);
            int4 *d_hot = reinterpret_cast<int4 *>(ws + o_hot);
            BuildRec *d_brec = reinterpret_cast<BuildRec *>(ws + o_brec);
            float *d_scratch = reinterpret_cast<float *>(ws + o_scratch);
            uint32_t *d_sp = reinterpret_cast<uint32_t *>(ws + o_sp);

            const int pb = 64, pg = (NROWS + pb - 1) / pb;
            int mg = (int)std::min<long long>((NNZ + 255) / 256, 2 * nsm);
            if (mg < 1) mg = 1;
            k_absmax<<<mg, 256, 0, stream>>>(d_vals, NNZ, d_gmax);
            k_count<<<pg, pb, 0, stream>>>(NROWS, d_so, d_seg, d_vo, c, d_bcount, d_gwork, d_rnsub, d_rspw);
            k_scan<<<1, 1024, 0, stream>>>(c, d_bcount, d_gwork, d_bstart, d_rnsub, d_rsuboff, d_rspw, d_rspoff, NROWS,
                                           d_gmax, d_gscale, d_ginfo, d_items, d_gnorm, use_split ? nslots_cta : 1 << 30);
            k_place<<<pg, pb, 0, stream>>>(NROWS, d_so, d_seg, d_vo, d_rk, d_rl, c, d_bstart, d_bcursor, d_rsuboff,
                                           d_rspoff, d_ents, d_hot, d_brec);
            // the sub-arc count is device-side: launch for the bound, surplus warps exit
            const long long nsub_launch = nsub_max;
            const unsigned sb_blocks = (unsigned)((nsub_launch * 32 + 255) / 256);
            const int nctas = (NG + MAXSI) * c.ntiles;
            const size_t smem = (size_t)c.nst * c.stage_bytes;
            const int threads = 32 * (c.G * c.NWL + NPW);
            if (c.P == 1) {
                k_build_sp<1><<<sb_blocks, 256, 0, stream>>>(d_rsuboff + NROWS, d_brec, d_vals, d_gscale, d_sp);
                cudaFuncSetAttribute(k_disco_tc<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
                k_disco_tc<1><<<nctas, threads, smem, stream>>>(d_inp, d_sp, d_hot, d_ents, d_bstart, d_ginfo, d_items,
                                                                d_gnorm, d_gscale, d_out, d_scratch, d_scnt, c);
            } else {
                k_build_sp<2><<<sb_blocks, 256, 0, stream>>>(d_rsuboff + NROWS, d_brec, d_vals, d_gscale, d_sp);
                cudaFuncSetAttribute(k_disco_tc<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
                k_disco_tc<2><<<nctas, threads, smem, stream>>>(d_inp, d_sp, d_hot, d_ents, d_bstart, d_ginfo, d_items,
                                                                d_gnorm, d_gscale, d_out, d_scratch, d_scnt, c);
            }
            cudaFreeAsync(ws, stream);
            return true;
        }
        return false;
    }

    void RunFallback(ffi::TensorView inp, ffi::TensorView row_ker, ffi::TensorView row_lat, ffi::TensorView seg_off,
                     ffi::TensorView seg, ffi::TensorView val_off, ffi::TensorView vals, int64_t nlat_out,
                     int64_t nlon_out, ffi::TensorView out)
    {
        const int B = (int)inp.size(0), C = (int)inp.size(1), K = (int)inp.size(2), Hi = (int)inp.size(3),
                  Wi = (int)inp.size(4);
        const long long BCl = (long long)B * C;
        const int NROWS = (int)row_ker.size(0);
        const long long NSEG = seg.size(0);
        const long long NNZ = vals.size(0);
        const int Ho = (int)nlat_out, Wo = (int)nlon_out;
        cudaStream_t stream
            = static_cast<cudaStream_t>(TVMFFIEnvGetStream(inp.device().device_type, inp.device().device_id));

        const __half *d_inp = static_cast<const __half *>(inp.data_ptr());
        const int *d_rk = static_cast<const int *>(row_ker.data_ptr());
        const int *d_rl = static_cast<const int *>(row_lat.data_ptr());
        const int64_t *d_so = static_cast<const int64_t *>(seg_off.data_ptr());
        const int *d_seg = static_cast<const int *>(seg.data_ptr());
        const int64_t *d_vo = static_cast<const int64_t *>(val_off.data_ptr());
        const float *d_vals = static_cast<const float *>(vals.data_ptr());
        __half *d_out = static_cast<__half *>(out.data_ptr());

        if (BCl <= 0 || Ho <= 0 || Wo <= 0) return;
        const int BC = (int)BCl;

        // ---- general path ----
        const long long nout = BCl * Ho * Wo;
        size_t o_out32 = 0;
        size_t o_svoff = align_up((size_t)nout * 4, 256);
        size_t o_srow = align_up(o_svoff + (size_t)std::max<long long>(NSEG, 1) * 8, 256);
        size_t total = o_srow + (size_t)std::max<long long>(NSEG, 1) * 4;
        unsigned char *ws = nullptr;
        cudaMallocAsync((void **)&ws, total, stream);
        float *d_out32 = reinterpret_cast<float *>(ws + o_out32);
        long long *d_svoff = reinterpret_cast<long long *>(ws + o_svoff);
        int *d_srow = reinterpret_cast<int *>(ws + o_srow);
        cudaMemsetAsync(d_out32, 0, (size_t)nout * 4, stream);
        if (NSEG > 0 && NROWS > 0 && Wi > 0 && Wo % Wi == 0) {
            cudaMemsetAsync(d_srow, 0xff, (size_t)NSEG * 4, stream);
            k_fb_prep<<<(NROWS + 127) / 128, 128, 0, stream>>>(NROWS, d_so, d_vo, d_seg, d_svoff, d_srow);
            dim3 grid((unsigned)std::min<long long>(NSEG, 1 << 20), (unsigned)std::min<long long>(BCl, 65535));
            k_fb_scatter<<<grid, 256, 0, stream>>>(d_inp, d_rk, d_rl, d_seg, d_vals, d_svoff, d_srow, d_out32, NSEG, BC,
                                                   K, Hi, Wi, Ho, Wo);
        }
        k_fb_narrow<<<1024, 256, 0, stream>>>(d_out32, d_out, nout);
        cudaFreeAsync(ws, stream);
    }

} // namespace cr
