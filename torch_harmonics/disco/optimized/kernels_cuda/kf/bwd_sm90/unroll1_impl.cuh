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

// DISCO spherical convolution transpose (arc scatter), fp16 activations, fp32 psi / accumulation.
//
// Fast path (pscale 1 or 2, Wi % 8 == 0): gather by output ring on tensor cores.
//   For one (row r, output ring h) arc and residue b = w mod pscale, the arc is a run of taps a with values
//   v(a), and   out[h, pscale*m + b] += sum_a v(a) * x_r[(m - a) mod Wi].
//   Writing m = 8*mu + nu (nu = 0..7 "phases") turns this 1-D circular convolution into a GEMM
//     D[mu][nu] = sum_k A[mu][k] * B[k][nu],  A[mu][k] = x[8*mu + base + k] (Hankel, read with ldmatrix
//     straight from a shared copy of the input row),  B[k][nu] = v(nu - k - base) (Toeplitz, from psi).
//   psi (fp32) is split into two fp16 residual terms after a global power-of-two scale; both products are
//   accumulated in fp32 tensor-core accumulators and the output is rounded to fp16 once.
//   A CTA owns G consecutive output rings x CB channels x all longitudes and walks every input row that
//   touches those rings (the input row is staged in shared memory once and reused by all its arcs).
// General path (anything else): fp32 atomic scatter + fp16 narrowing.
//
// Every call recomputes the psi bookkeeping (per-arc tap ranges, ring-group lists, scale) on the GPU.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <algorithm>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/error.h>

namespace unroll1_impl
{
    namespace ffi = tvm::ffi;

#define NSPLIT 2 // scaled fp16 residual terms per fp32 psi value

    struct __align__(16) SegInfo {
        int ring, base, kp, st;
        int len, spoff, spg, row; // spoff: psi words before this arc in its entry; spg: ... in its row
        long long voff;
        long long pad;
    };

    struct __align__(16) SegHot {
        int ring, base, kp, spoff;
    };

    struct __align__(16) Entry {
        int xrow, s0, s1, EB8;  // xrow = row_ker * Hi + row_lat; EB8 = 8-aligned start of the staged input window
        int XL, spw, nseg, grp; // XL: staged window length (halfs); spw: psi operand words of the entry
        long long spg;          // global word offset of the entry's psi operand
        long long pad;
        SegHot hot[3];
    };

    struct PrepCfg {
        int P, Wi, Wo, Ho, Hi, G, XW, XLMAX, SPMAX, SEGMAX;
        int *laps;                 // max wraps of an arc around the circle (psi scale headroom for folded arcs)
        const unsigned char *skip; // rings owned by the frequency-domain path (nullptr: none)
    };

    // Programmatic dependent launch overlaps host launch latency with the predecessor grid draining.
    __device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
    __device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }

    __device__ __forceinline__ int floordiv_i(int a, int b)
    {
        int q = a / b;
        if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
        return q;
    }

    // Walk the arcs of row r, compute per-arc tap ranges and group consecutive arcs that hit the same ring group
    // into "entries" (one staged input window each).
    //   arc (start st, length lc) covers taps a in [floor(st/P), floor((st+lc-1)/P)];
    //   base = -2*ceil(a_hi/2) (window start, even), kp = K extent rounded to 8.
    // emit(g, s0, s1, EB8, XL, spw, nseg, sp_row_start, work)
    template <bool WRITE_SINFO, class Emit>
    __device__ __forceinline__ int walk_row(int r, const int64_t *__restrict__ seg_off, const int *__restrict__ seg,
                                            const int64_t *__restrict__ val_off, const PrepCfg c,
                                            SegInfo *__restrict__ sinfo, long long rsp0, Emit emit)
    {
        long long sb = seg_off[r], se = seg_off[r + 1];
        long long voff = val_off[r];
        int rsp = 0; // psi words of this row so far
        bool open = false;
        int e_g = 0, e_EB = 0, e_end = 0, e_spw = 0, e_n = 0, e_work = 0, e_rsp = 0;
        long long e_s0 = 0;
        auto xl_of = [&](int EB, int end) { return (end - (EB & ~7) + c.XW - 8 + 7) & ~7; };
        for (long long s8 = sb; s8 < se; s8 += 8) {
            // load the next (up to) 8 arcs of the row at once, then walk them
            int ring8[8], start8[8], len8[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const long long sk = min(s8 + k, se - 1);
                ring8[k] = seg[3 * sk];
                start8[k] = seg[3 * sk + 1];
                len8[k] = seg[3 * sk + 2];
            }
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const long long s = s8 + k;
                if (s >= se) break;
                int ring = ring8[k], start = start8[k], len = len8[k];
                long long myvoff = voff;
                if (len > 0) voff += len;
                // An arc longer than Wo wraps onto itself; k_build_sp folds the repeated values in fp32.
                if (!WRITE_SINFO && len > c.Wo && c.laps) atomicMax(c.laps, (int)(((long long)len + c.Wo - 1) / c.Wo));
                int lc = min(len, c.Wo);
                if (lc <= 0 || ring < 0 || ring >= c.Ho || (c.skip && c.skip[ring])) {
                    if (open) {
                        emit(e_g, e_s0, s, e_EB & ~7, xl_of(e_EB, e_end), e_spw, e_n, e_rsp, e_work);
                        open = false;
                    }
                    if (WRITE_SINFO) {
                        SegInfo si = {};
                        si.row = r;
                        sinfo[s] = si;
                    }
                    continue;
                }
                int st = start % c.Wo;
                if (st < 0) st += c.Wo;
                if (st > c.Wo / 2) st -= c.Wo;
                int a_lo = floordiv_i(st, c.P), a_hi = floordiv_i(st + lc - 1, c.P);
                int base = -2 * floordiv_i(a_hi + 1, 2);
                int kp = (8 - a_lo - base + 7) & ~7; // K extent, multiple of 8 (16-wide steps + optional 8-wide tail)
                int g = ring / c.G;
                int spw = c.P * NSPLIT * (kp + 8);
                if (open) {
                    int nEB = min(e_EB, base), nend = max(e_end, base + kp);
                    if (g != e_g || e_n >= c.SEGMAX || xl_of(nEB, nend) > c.XLMAX || e_spw + spw > c.SPMAX) {
                        emit(e_g, e_s0, s, e_EB & ~7, xl_of(e_EB, e_end), e_spw, e_n, e_rsp, e_work);
                        open = false;
                    } else {
                        e_EB = nEB;
                        e_end = nend;
                    }
                }
                if (!open) {
                    open = true;
                    e_g = g;
                    e_s0 = s;
                    e_EB = base;
                    e_end = base + kp;
                    e_spw = 0;
                    e_n = 0;
                    e_work = 0;
                    e_rsp = rsp;
                }
                if (WRITE_SINFO) {
                    SegInfo si;
                    si.ring = ring;
                    si.base = base;
                    si.kp = kp;
                    si.st = st;
                    si.len = lc;
                    si.spoff = e_spw;
                    si.spg = (int)0;
                    si.row = r;
                    si.voff = myvoff;
                    si.pad = rsp0 + rsp; // pad: absolute psi word offset of this arc
                    sinfo[s] = si;
                }
                e_spw += spw;
                rsp += spw;
                e_n++;
                e_work += kp + 16;
            }
        }
        if (open) emit(e_g, e_s0, se, e_EB & ~7, xl_of(e_EB, e_end), e_spw, e_n, e_rsp, e_work);
        return rsp;
    }

    __global__ void k_prep_count(int nrows, const int64_t *__restrict__ seg_off, const int *__restrict__ seg,
                                 const int64_t *__restrict__ val_off, PrepCfg c, int *__restrict__ gcount,
                                 unsigned long long *__restrict__ gwork, int *__restrict__ row_spw)
    {
        int r = blockIdx.x * blockDim.x + threadIdx.x;
        if (r >= nrows) return;
        row_spw[r] = walk_row<false>(r, seg_off, seg, val_off, c, nullptr, 0,
                                     [&](int g, long long, long long, int, int, int, int, int, int work) {
                                         atomicAdd(&gcount[g], 1);
                                         atomicAdd(&gwork[g], (unsigned long long)(work + 64));
                                     });
    }

    __global__ void k_prep_place(int nrows, const int64_t *__restrict__ seg_off, const int *__restrict__ seg,
                                 const int64_t *__restrict__ val_off, const int *__restrict__ row_ker,
                                 const int *__restrict__ row_lat, PrepCfg c, const int *__restrict__ gstart,
                                 const long long *__restrict__ row_spoff, int *__restrict__ gcursor,
                                 Entry *__restrict__ ents, SegInfo *__restrict__ sinfo)
    {
        pdl_wait();
        pdl_trigger();
        int r = blockIdx.x * blockDim.x + threadIdx.x;
        if (r >= nrows) return;
        const int xrow = row_ker[r] * c.Hi + row_lat[r];
        const long long rsp0 = row_spoff[r];
        // entries are buffered so that their slot atomics are in flight together
        Entry buf[8];
        int nb = 0;
        auto flush = [&]() {
            int pos[8];
#pragma unroll
            for (int k = 0; k < 8; ++k)
                if (k < nb) pos[k] = gstart[buf[k].grp] + atomicAdd(&gcursor[buf[k].grp], 1);
#pragma unroll
            for (int k = 0; k < 8; ++k)
                if (k < nb) ents[pos[k]] = buf[k];
            nb = 0;
        };
        walk_row<true>(r, seg_off, seg, val_off, c, sinfo, rsp0,
                       [&](int g, long long s0, long long s1, int EB8, int XL, int spw, int nseg, int rsp, int) {
                           Entry e;
                           int EBm = EB8 % c.Wi;
                           if (EBm < 0) EBm += c.Wi;
                           e.xrow = xrow;
                           e.s0 = (int)s0;
                           e.s1 = EBm;
                           e.EB8 = EB8;
                           e.XL = XL;
                           e.spw = spw;
                           e.nseg = nseg;
                           e.grp = g;
                           e.spg = rsp0 + rsp;
                           e.pad = (long long)xrow * c.Wi;
#pragma unroll
                           for (int k = 0; k < 3; ++k) {
                               SegHot h {};
                               if (k < nseg) {
                                   const SegInfo &si = sinfo[s0 + k];
                                   h = SegHot {si.ring, si.base, si.kp, si.spoff};
                               }
                               e.hot[k] = h;
                           }
                           buf[nb++] = e;
                           if (nb == 8) flush();
                       });
        flush();
    }

    // Psi operand of one arc, written once per call to global memory: for residue b and split term s,
    // word[t] = (R'[t], R'[t+1]) (fp16 pair), R'[t] = 2^sc * v(a = 8 - t - base) on residue b, 0 outside the arc.
    // The two scaled residual terms retain about 22 mantissa bits of each fp32 psi value.
    template <int P>
    __global__ void k_build_sp(long long nseg, const SegInfo *__restrict__ sinfo, const float *__restrict__ vals,
                               const float *__restrict__ gscale, uint32_t *__restrict__ spglob,
                               const int *__restrict__ seg, int Wo)
    {
        pdl_wait();
        const long long s = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        const int lane = threadIdx.x & 31;
        if (s >= nseg) return;
        const SegInfo sg = sinfo[s];
        if (sg.kp <= 0) return;
        const float scale = gscale[0];
        const int NK = sg.kp + 8;
        uint32_t *dst = spglob + sg.pad;
        const float *v = vals + sg.voff;
        const int flen = seg[3 * s + 2];
        auto emit = [&](int t, const float(&y0)[P][2]) {
#pragma unroll
            for (int b = 0; b < P; ++b) {
                float y[2] = {y0[b][0] * scale, y0[b][1] * scale};
#pragma unroll
                for (int k = 0; k < NSPLIT; ++k) {
                    __half h0 = __float2half_rn(y[0]), h1 = __float2half_rn(y[1]);
                    y[0] -= __half2float(h0);
                    y[1] -= __half2float(h1);
                    __half2 w = __halves2half2(h0, h1);
                    dst[(b * NSPLIT + k) * NK + t] = *reinterpret_cast<uint32_t *>(&w);
                }
            }
        };
        if (flen <= Wo) {
            // common case (the arc does not wrap onto itself): independent loads, several taps in flight per lane
#pragma unroll 4
            for (int t = lane; t < NK; t += 32) {
                float y[P][2];
#pragma unroll
                for (int b = 0; b < P; ++b)
#pragma unroll
                    for (int u = 0; u < 2; ++u) {
                        const int jj = P * (8 - (t + u) - sg.base) + b - sg.st;
                        y[b][u] = (jj >= 0 && jj < sg.len) ? __ldg(v + jj) : 0.f;
                    }
                emit(t, y);
            }
        } else {
            for (int t = lane; t < NK; t += 32) {
                float y[P][2];
#pragma unroll
                for (int b = 0; b < P; ++b)
#pragma unroll
                    for (int u = 0; u < 2; ++u) {
                        const int jj = P * (8 - (t + u) - sg.base) + b - sg.st;
                        float acc = 0.f;
                        if (jj >= 0 && jj < sg.len)
                            for (int j = jj; j < flen; j += Wo) acc += __ldg(v + j);
                        y[b][u] = acc;
                    }
                emit(t, y);
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

    __global__ void k_prep_count_abs(int nrows, const int64_t *__restrict__ seg_off, const int *__restrict__ seg,
                                     const int64_t *__restrict__ val_off, PrepCfg c, int *__restrict__ gcount,
                                     unsigned long long *__restrict__ gwork, int *__restrict__ row_spw,
                                     const float *__restrict__ vals, long long nnz, int *__restrict__ gmax)
    {
        pdl_trigger();
        const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
        if (warp == 0) {
            const int r = (int)blockIdx.x * 32 + lane;
            if (r < nrows)
                row_spw[r] = walk_row<false>(r, seg_off, seg, val_off, c, nullptr, 0,
                                             [&](int g, long long, long long, int, int, int, int, int, int work) {
                                                 atomicAdd(&gcount[g], 1);
                                                 atomicAdd(&gwork[g], (unsigned long long)(work + 64));
                                             });
        }
        float m = 0.f;
        if (warp > 0) {
            const long long i0 = (long long)blockIdx.x * 224 + (tid - 32);
            const long long stride = (long long)gridDim.x * 224;
            for (long long i = i0; i < nnz; i += stride) m = fmaxf(m, fabsf(__ldg(vals + i)));
            for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        }
        __shared__ float wm[7];
        if (warp > 0 && lane == 0) wm[warp - 1] = m;
        __syncthreads();
        if (warp == 0) {
            m = lane < 7 ? wm[lane] : 0.f;
            for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
            if (lane == 0) atomicMax(gmax, __float_as_int(m));
        }
    }

    // Block-wide exclusive scan (1024 threads) of n values; out[n] = total.
    template <class TIn, class TOut>
    __device__ void block_scan_excl(const TIn *__restrict__ in, TOut *__restrict__ out, int n, TOut *wsum, TOut *carry)
    {
        int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
        if (tid == 0) *carry = 0;
        __syncthreads();
        for (int base = 0; base < n; base += 1024) {
            int i = base + tid;
            TOut v = (i < n) ? (TOut)in[i] : (TOut)0;
            TOut x = v;
            for (int o = 1; o < 32; o <<= 1) {
                TOut y = __shfl_up_sync(0xffffffffu, x, o);
                if (lane >= o) x += y;
            }
            if (lane == 31) wsum[wid] = x;
            __syncthreads();
            if (wid == 0) {
                TOut w = wsum[lane];
                for (int o = 1; o < 32; o <<= 1) {
                    TOut y = __shfl_up_sync(0xffffffffu, w, o);
                    if (lane >= o) w += y;
                }
                wsum[lane] = w;
            }
            __syncthreads();
            TOut excl = x - v + (wid > 0 ? wsum[wid - 1] : (TOut)0) + *carry;
            if (i < n) out[i] = excl;
            __syncthreads();
            if (tid == 1023) *carry = excl + v;
            __syncthreads();
        }
        if (tid == 0) out[n] = *carry;
        __syncthreads();
    }

// Single CTA: scans (entries per group, psi words per row), heavy-first group order, psi scale.
#define SORT_MAX 4096
    __global__ void __launch_bounds__(1024)
        k_scan(const int *__restrict__ gcount, const unsigned long long *__restrict__ gwork, int *__restrict__ gstart,
               int *__restrict__ gorder, const int *__restrict__ row_spw, long long *__restrict__ row_spoff, int nrows,
               const int *__restrict__ gmax, float *__restrict__ gscale, int *__restrict__ gheavy,
               int4 *__restrict__ items, double split_scale, int max_extra, int max_slots, int NG)
    {
        pdl_wait();
        pdl_trigger();
        __shared__ int wsum[32];
        __shared__ long long wsum64[32];
        __shared__ int carry;
        __shared__ long long carry64;
        __shared__ unsigned long long keys[SORT_MAX];
        int tid = threadIdx.x;
        block_scan_excl<int, int>(gcount, gstart, NG, wsum, &carry);
        block_scan_excl<int, long long>(row_spw, row_spoff, nrows, wsum64, &carry64);
        if (NG <= SORT_MAX) {
            int n2 = 1;
            while (n2 < NG) n2 <<= 1;
            for (int i = tid; i < n2; i += 1024) {
                if (i < NG) {
                    unsigned long long w = gwork[i];
                    unsigned int w32 = w > 0xffffffffull ? 0xffffffffu : (unsigned int)w;
                    keys[i] = ((unsigned long long)(~w32) << 32) | (unsigned int)i;
                } else {
                    keys[i] = ~0ull;
                }
            }
            __syncthreads();
            for (int k = 2; k <= n2; k <<= 1) {
                for (int j = k >> 1; j > 0; j >>= 1) {
                    for (int i = tid; i < n2; i += 1024) {
                        int ixj = i ^ j;
                        if (ixj > i) {
                            unsigned long long a = keys[i], b = keys[ixj];
                            bool asc = ((i & k) == 0);
                            if ((a > b) == asc) {
                                keys[i] = b;
                                keys[ixj] = a;
                            }
                        }
                    }
                    __syncthreads();
                }
            }
            for (int i = tid; i < NG; i += 1024) gorder[i] = (int)(keys[i] & 0xffffffffu);
            if (max_extra == 0) {
                if (tid == 0) {
                    gheavy[0] = 0;
                    gheavy[1] = 0;
                }
            } else {
                // heavy groups (more than twice the mean work) are scheduled first for every channel block; when there
                // are few channel blocks, the heaviest of them are split by entry range (split_target = work per split)
                __shared__ unsigned long long s_tot;
                __shared__ int s_nh;
                if (tid == 0) {
                    s_tot = 0;
                    s_nh = 0;
                }
                __syncthreads();
                {
                    unsigned long long part = 0;
                    for (int i = tid; i < NG; i += 1024) part += (unsigned long long)(~(unsigned int)(keys[i] >> 32));
                    for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o);
                    if ((tid & 31) == 0 && part) atomicAdd(&s_tot, part);
                }
                __syncthreads();
                {
                    const unsigned long long tot = s_tot;
                    int cnt = 0;
                    for (int i = tid; i < NG; i += 1024)
                        cnt += ((unsigned long long)(~(unsigned int)(keys[i] >> 32)) * NG > 2 * tot) ? 1 : 0;
                    for (int o = 16; o > 0; o >>= 1) cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
                    if ((tid & 31) == 0 && cnt) atomicAdd(&s_nh, cnt);
                }
                __syncthreads();
                // Work items: the 32 heaviest groups may be split (decided by one warp with shuffle scans: split budget
                // max_extra, at most max_slots split groups); the other heavy groups are single items.
                __shared__ int s_n32;
                const int nh = s_nh;
                const int n32 = min(nh, 32);
                if (tid < 32) {
                    const unsigned long long tot = s_tot;
                    const double target = (double)tot * split_scale;
                    int g = 0, cnt = 0, ns = 1, e0g = 0;
                    if (tid < n32) {
                        g = (int)(keys[tid] & 0xffffffffu);
                        e0g = gstart[g];
                        cnt = gstart[g + 1] - e0g;
                        if (max_extra > 0 && target > 0.0) {
                            ns = (int)ceil((double)(~(unsigned int)(keys[tid] >> 32)) / target);
                            ns = max(1, min(min(ns, 8), cnt));
                        }
                    }
                    // greedy budget in heaviest-first order
                    int ex = ns - 1, inc = ex;
                    for (int o = 1; o < 32; o <<= 1) {
                        int y = __shfl_up_sync(0xffffffffu, inc, o);
                        if (tid >= o) inc += y;
                    }
                    if (inc > max_extra) ns = 1;
                    int sp = (ns > 1) ? 1 : 0, sinc = sp;
                    for (int o = 1; o < 32; o <<= 1) {
                        int y = __shfl_up_sync(0xffffffffu, sinc, o);
                        if (tid >= o) sinc += y;
                    }
                    if (sinc > max_slots) ns = 1;
                    const int slot = (ns > 1) ? sinc - 1 : 0xffff;
                    int pinc = (tid < n32) ? ns : 0;
                    for (int o = 1; o < 32; o <<= 1) {
                        int y = __shfl_up_sync(0xffffffffu, pinc, o);
                        if (tid >= o) pinc += y;
                    }
                    const int pos = pinc - ((tid < n32) ? ns : 0);
                    if (tid < n32)
                        for (int k = 0; k < ns; ++k)
                            items[pos + k] = make_int4(g, e0g + (int)((long long)k * cnt / ns),
                                                       e0g + (int)((long long)(k + 1) * cnt / ns), slot | (ns << 16));
                    if (tid == 31) s_n32 = pinc;
                }
                __syncthreads();
                for (int i = 32 + tid; i < nh; i += 1024) {
                    const int g = (int)(keys[i] & 0xffffffffu);
                    items[s_n32 + i - 32] = make_int4(g, gstart[g], gstart[g + 1], 0xffff | (1 << 16));
                }
                if (tid == 0) {
                    gheavy[0] = nh;
                    gheavy[1] = s_n32 + max(nh - 32, 0);
                }
            }
        } else {
            for (int i = tid; i < NG; i += 1024) gorder[i] = i;
            if (tid == 0) {
                gheavy[0] = 0;
                gheavy[1] = 0;
            }
        }
        if (tid == 0) {
            float m = __int_as_float(gmax[0]) * (float)max(1, gmax[1]);
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

    __device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
    {
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%0,%1,%2,%3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }

    __device__ __forceinline__ void mma1688(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0)
    {
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a1), "r"(b0));
    }

    // One arc. A[mu][k] = x_ext[off + 8*mu + k] (Hankel): lane (g, t) holds words at 8g + 2t (+64 for row g+8,
    // +8 for k+8). B[k][nu] from the split psi words: lane reads word 16ks + 8j + (2t + 8 - g).
    template <int P, int MTW, int G, int RL, bool PAIR>
    __device__ __forceinline__ void seg_mma(float (&acc)[G][MTW][P][4], const uint32_t *__restrict__ xw,
                                            const uint32_t *__restrict__ spb, int NK, int nks, bool tail8)
    {
#pragma unroll 1
        for (int ks = 0; ks < nks; ++ks) {
            uint32_t bf[P][NSPLIT][2];
#pragma unroll
            for (int b = 0; b < P; ++b)
#pragma unroll
                for (int s = 0; s < NSPLIT; ++s)
#pragma unroll
                    for (int j = 0; j < 2; ++j) bf[b][s][j] = spb[(b * NSPLIT + s) * NK + 16 * ks + 8 * j];
            if constexpr (P == 2 && PAIR) {
#pragma unroll
                for (int mg = 0; mg < MTW; mg += 3) {
                    uint32_t a[3][4];
#pragma unroll
                    for (int u = 0; u < 3; ++u) {
                        const uint32_t *p = xw + 64 * (mg + u) + 8 * ks;
                        a[u][0] = p[0];
                        a[u][1] = p[32];
                        a[u][2] = p[4];
                        a[u][3] = p[36];
                    }
#pragma unroll
                    for (int s = 0; s < NSPLIT; ++s)
#pragma unroll
                        for (int u = 0; u < 3; ++u)
#pragma unroll
                            for (int b = 0; b < P; ++b) mma16816(acc[RL][mg + u][b], a[u], bf[b][s][0], bf[b][s][1]);
                }
            } else {
                uint32_t a[MTW][4];
#pragma unroll
                for (int mt = 0; mt < MTW; ++mt) {
                    const uint32_t *p = xw + 64 * mt + 8 * ks;
                    a[mt][0] = p[0];
                    a[mt][1] = p[32];
                    a[mt][2] = p[4];
                    a[mt][3] = p[36];
                }
#pragma unroll
                for (int s = 0; s < NSPLIT; ++s)
#pragma unroll
                    for (int mt = 0; mt < MTW; ++mt)
#pragma unroll
                        for (int b = 0; b < P; ++b) mma16816(acc[RL][mt][b], a[mt], bf[b][s][0], bf[b][s][1]);
            }
        }
        if (tail8) { // last 8 k values: m16n8k8
            uint32_t bf[P][NSPLIT];
#pragma unroll
            for (int b = 0; b < P; ++b)
#pragma unroll
                for (int s = 0; s < NSPLIT; ++s) bf[b][s] = spb[(b * NSPLIT + s) * NK + 16 * nks];
            if constexpr (P == 2 && PAIR) {
#pragma unroll
                for (int mg = 0; mg < MTW; mg += 3) {
                    uint32_t a[3][2];
#pragma unroll
                    for (int u = 0; u < 3; ++u) {
                        const uint32_t *p = xw + 64 * (mg + u) + 8 * nks;
                        a[u][0] = p[0];
                        a[u][1] = p[32];
                    }
#pragma unroll
                    for (int s = 0; s < NSPLIT; ++s)
#pragma unroll
                        for (int u = 0; u < 3; ++u)
#pragma unroll
                            for (int b = 0; b < P; ++b) mma1688(acc[RL][mg + u][b], a[u][0], a[u][1], bf[b][s]);
                }
            } else {
                uint32_t a[MTW][2];
#pragma unroll
                for (int mt = 0; mt < MTW; ++mt) {
                    const uint32_t *p = xw + 64 * mt + 8 * nks;
                    a[mt][0] = p[0];
                    a[mt][1] = p[32];
                }
#pragma unroll
                for (int s = 0; s < NSPLIT; ++s)
#pragma unroll
                    for (int mt = 0; mt < MTW; ++mt)
#pragma unroll
                        for (int b = 0; b < P; ++b) mma1688(acc[RL][mt][b], a[mt][0], a[mt][1], bf[b][s]);
            }
        }
    }

    template <int P, int MTW, int G, int RL, bool PAIR>
    __device__ __forceinline__ void seg_dispatch(int rl, float (&acc)[G][MTW][P][4], const uint32_t *__restrict__ xw,
                                                 const uint32_t *__restrict__ spb, int NK, int nks, bool tail8)
    {
        if constexpr (RL < G) {
            if (rl == RL) {
                seg_mma<P, MTW, G, RL, PAIR>(acc, xw, spb, NK, nks, tail8);
                return;
            }
            seg_dispatch<P, MTW, G, RL + 1, PAIR>(rl, acc, xw, spb, NK, nks, tail8);
        }
    }

    struct MainCfg {
        int BC, K, Hi, Wi, Ho, Wo, NG, WPC, CB, ncb;
        int XLMAX, SPMAX, SEGMAX;
        int off_sp, off_ent, stage_bytes;
    };

    // Warp-specialized: warps [0, CB*WPC) compute, the last warp streams entries (staged input window, psi
    // operand, arc metadata) with TMA bulk copies into a STAGES-deep ring guarded by full/empty mbarriers.
    template <int P, int MTW, int G, int MINB, bool PAIR, bool DO_SPLIT, int MAXT = 160, int STAGES = 3>
    __global__ void __launch_bounds__(MAXT, MINB)
        k_disco_tc(const __half *__restrict__ inp, const uint32_t *__restrict__ spglob, const Entry *__restrict__ ents,
                   const int *__restrict__ gstart, const int *__restrict__ gorder, const int *__restrict__ gheavy,
                   const int4 *__restrict__ items, const float *__restrict__ gscale, __half *__restrict__ out,
                   float *__restrict__ scratch, int *__restrict__ scnt, MainCfg c)
    {
        pdl_wait();
        extern __shared__ __align__(128) unsigned char smem[];
        __shared__ __align__(8) uint64_t full_bar[STAGES];
        __shared__ __align__(8) uint64_t empty_bar[STAGES];
        const int tid = threadIdx.x;
        const int lane = tid & 31, warp = tid >> 5;
        const int nwc = c.CB * c.WPC;
        // CTA order: work items of the heavy ring groups (possibly split by entry range) for all channel blocks
        // first, then the remaining groups group-major to reuse psi and metadata across channel blocks.
        const int nh = gheavy[0], nhi = gheavy[1];
        int cbi, gr, e0, n, slot = -1, nsplit = 1;
        if constexpr (!DO_SPLIT) {
            const int oi = blockIdx.x / c.ncb;
            cbi = blockIdx.x - oi * c.ncb;
            gr = gorder[oi];
            e0 = gstart[gr];
            n = gstart[gr + 1] - e0;
        } else {
            if ((int)blockIdx.x < nhi * c.ncb) {
                const int4 it = items[blockIdx.x / c.ncb];
                cbi = blockIdx.x % c.ncb;
                gr = it.x;
                e0 = it.y;
                n = it.z - it.y;
                nsplit = it.w >> 16;
                if (nsplit > 1) slot = it.w & 0xffff;
            } else {
                const int r = blockIdx.x - nhi * c.ncb, nl = c.NG - nh;
                if (r >= nl * c.ncb) return; // spare CTA (grid is sized for the maximum number of splits)
                cbi = r % c.ncb;
                gr = gorder[nh + r / c.ncb];
                e0 = gstart[gr];
                n = gstart[gr + 1] - e0;
            }
        }

        if (tid == 0) {
            for (int s = 0; s < STAGES; ++s) {
                mbar_init(&full_bar[s], 1);
                mbar_init(&empty_bar[s], nwc);
            }
            asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        }
        __syncthreads();

        if (warp == nwc) {
            // ---------------- producer ----------------
            if (n > 0) {
                const size_t cstride = (size_t)c.K * c.Hi * c.Wi;
                const int nvalid = min(c.CB, c.BC - cbi * c.CB);
                Entry E = ents[e0];
                for (int i = 0; i < n; ++i) {
                    const Entry En = (i + 1 < n) ? ents[e0 + i + 1] : E;
                    const int s = i % STAGES;
                    if (lane == 0 && i >= STAGES) mbar_wait(&empty_bar[s], ((i / STAGES) - 1) & 1);
                    __syncwarp();
                    unsigned char *sb = smem + s * c.stage_bytes;
                    const unsigned bytes = (unsigned)(nvalid * E.XL * 2 + E.spw * 4 + (int)sizeof(Entry));
                    if (lane == 0) mbar_expect_tx(&full_bar[s], bytes);
                    __syncwarp();
                    if (lane < nvalid) {
                        const __half *src = inp + (size_t)(cbi * c.CB + lane) * cstride + (size_t)E.pad;
                        __half *dst = reinterpret_cast<__half *>(sb) + lane * c.XLMAX;
                        int EBm = E.s1;
                        int q = EBm, rem = E.XL;
                        while (rem > 0) {
                            const int len = min(rem, c.Wi - q);
                            bulk_g2s(dst, src + q, (unsigned)len * 2, &full_bar[s]);
                            dst += len;
                            rem -= len;
                            q = 0;
                        }
                    } else if (lane == nvalid) {
                        bulk_g2s(sb + c.off_sp, spglob + E.spg, (unsigned)E.spw * 4, &full_bar[s]);
                    } else if (lane == nvalid + 1) {
                        bulk_g2s(sb + c.off_ent, ents + e0 + i, (unsigned)sizeof(Entry), &full_bar[s]);
                    }
                    __syncwarp();
                    E = En;
                }
            }
            return;
        }

        // ---------------- consumers ----------------
        const int cl = warp / c.WPC, mt0 = (warp % c.WPC) * MTW;
        const int bc = cbi * c.CB + cl;
        const bool active = bc < c.BC;
        const float inv_scale = gscale[1];

        float acc[G][MTW][P][4];
#pragma unroll
        for (int i = 0; i < G; ++i)
#pragma unroll
            for (int j = 0; j < MTW; ++j)
#pragma unroll
                for (int b = 0; b < P; ++b)
#pragma unroll
                    for (int q = 0; q < 4; ++q) acc[i][j][b][q] = 0.f;

        const int g8 = lane >> 2, t4 = lane & 3;
        const int lw = 2 * t4 + 8 - g8; // B-fragment word offset of this lane

        for (int i = 0; i < n; ++i) {
            const int s = i % STAGES;
            mbar_wait(&full_bar[s], (i / STAGES) & 1);
            const unsigned char *sb = smem + s * c.stage_bytes;
            if (active) {
                const Entry *E = reinterpret_cast<const Entry *>(sb + c.off_ent);
                const SegHot *si = E->hot;
                const int nseg = E->nseg, EB8 = E->EB8;
                const __half *xs = reinterpret_cast<const __half *>(sb) + cl * c.XLMAX + 128 * mt0 + 8 * g8 + 2 * t4;
                const uint32_t *sp = reinterpret_cast<const uint32_t *>(sb + c.off_sp) + lw;
                for (int k = 0; k < nseg; ++k) {
                    const SegHot &sg = si[k];
                    const int rl = sg.ring - gr * G;
                    const uint32_t *xw = reinterpret_cast<const uint32_t *>(xs + (sg.base - EB8));
                    seg_dispatch<P, MTW, G, 0, PAIR>(rl, acc, xw, sp + sg.spoff, sg.kp + 8, sg.kp >> 4, (sg.kp & 8) != 0);
                }
            }
            // order this warp's generic-proxy reads of the stage before the producer's next async-proxy (TMA) writes
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            __syncwarp();
            if (lane == 0) mbar_arrive(&empty_bar[s]);
        }

        // Split work item: add the partial sums into the fp32 scratch; the last split of this (group, channel block)
        // rounds the totals once to fp16 below.
        const int nmu = c.Wi >> 3;
        if constexpr (DO_SPLIT)
            if (slot >= 0) {
                if (active) {
#pragma unroll
                    for (int rl = 0; rl < G; ++rl) {
                        float *srow = scratch + ((((size_t)slot * c.ncb + cbi) * G + rl) * c.CB + cl) * c.Wo;
#pragma unroll
                        for (int mt = 0; mt < MTW; ++mt)
#pragma unroll
                            for (int hh = 0; hh < 2; ++hh) {
                                const int mu = 16 * (mt0 + mt) + g8 + 8 * hh;
                                if (mu >= nmu) continue;
                                if (P == 1) {
                                    atomicAdd(reinterpret_cast<float2 *>(srow + 8 * mu + 2 * t4),
                                              make_float2(acc[rl][mt][0][2 * hh], acc[rl][mt][0][2 * hh + 1]));
                                } else {
                                    atomicAdd(reinterpret_cast<float4 *>(srow + 16 * mu + 4 * t4),
                                              make_float4(acc[rl][mt][0][2 * hh], acc[rl][mt][P - 1][2 * hh],
                                                          acc[rl][mt][0][2 * hh + 1], acc[rl][mt][P - 1][2 * hh + 1]));
                                }
                            }
                    }
                }
                __threadfence();
                __shared__ int s_last;
                asm volatile("bar.sync 1, %0;\n" ::"r"(nwc * 32) : "memory");
                if (tid == 0) s_last = (atomicAdd(&scnt[slot * c.ncb + cbi], 1) == nsplit - 1);
                asm volatile("bar.sync 1, %0;\n" ::"r"(nwc * 32) : "memory");
                if (!s_last) return;
                __threadfence();
                if (active) {
#pragma unroll
                    for (int rl = 0; rl < G; ++rl) {
                        const float *srow = scratch + ((((size_t)slot * c.ncb + cbi) * G + rl) * c.CB + cl) * c.Wo;
#pragma unroll
                        for (int mt = 0; mt < MTW; ++mt)
#pragma unroll
                            for (int hh = 0; hh < 2; ++hh) {
                                const int mu = 16 * (mt0 + mt) + g8 + 8 * hh;
                                if (mu >= nmu) continue;
                                if (P == 1) {
                                    const float2 v = __ldcg(reinterpret_cast<const float2 *>(srow + 8 * mu + 2 * t4));
                                    acc[rl][mt][0][2 * hh] = v.x;
                                    acc[rl][mt][0][2 * hh + 1] = v.y;
                                } else {
                                    const float4 v = __ldcg(reinterpret_cast<const float4 *>(srow + 16 * mu + 4 * t4));
                                    acc[rl][mt][0][2 * hh] = v.x;
                                    acc[rl][mt][P - 1][2 * hh] = v.y;
                                    acc[rl][mt][0][2 * hh + 1] = v.z;
                                    acc[rl][mt][P - 1][2 * hh + 1] = v.w;
                                }
                            }
                    }
                }
            }

        // Epilogue: scale back, round once to fp16, store.
        if (!active) return;
#pragma unroll
        for (int rl = 0; rl < G; ++rl) {
            const int ring = gr * G + rl;
            if (ring >= c.Ho) break;
            __half *orow = out + ((size_t)bc * c.Ho + ring) * c.Wo;
#pragma unroll
            for (int mt = 0; mt < MTW; ++mt) {
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int mu = 16 * (mt0 + mt) + g8 + 8 * hh;
                    if (mu >= nmu) continue;
                    if (P == 1) {
                        __half2 v = __floats2half2_rn(acc[rl][mt][0][2 * hh] * inv_scale,
                                                      acc[rl][mt][0][2 * hh + 1] * inv_scale);
                        *reinterpret_cast<__half2 *>(orow + 8 * mu + 2 * t4) = v;
                    } else {
                        __half2 v0 = __floats2half2_rn(acc[rl][mt][0][2 * hh] * inv_scale,
                                                       acc[rl][mt][P - 1][2 * hh] * inv_scale);
                        __half2 v1 = __floats2half2_rn(acc[rl][mt][0][2 * hh + 1] * inv_scale,
                                                       acc[rl][mt][P - 1][2 * hh + 1] * inv_scale);
                        uint2 pk;
                        pk.x = *reinterpret_cast<uint32_t *>(&v0);
                        pk.y = *reinterpret_cast<uint32_t *>(&v1);
                        *reinterpret_cast<uint2 *>(orow + 16 * mu + 4 * t4) = pk;
                    }
                }
            }
        }
    }

    // ----------------------------------------------------------------------------------------------------------
    // General path: fp32 atomic scatter (any pscale, any Wi), then narrowing.
    // ----------------------------------------------------------------------------------------------------------
    __global__ void k_fb_prep(int nrows, const int64_t *__restrict__ seg_off, const int64_t *__restrict__ val_off,
                              const int *__restrict__ seg, long long *__restrict__ svoff, int *__restrict__ srow)
    {
        int r = blockIdx.x * blockDim.x + threadIdx.x;
        if (r >= nrows) return;
        long long v = val_off[r];
        for (long long s = seg_off[r]; s < seg_off[r + 1]; ++s) {
            svoff[s] = v;
            srow[s] = r;
            int len = seg[3 * s + 2];
            if (len > 0) v += len;
        }
    }

    __global__ void k_fb_scatter(const __half *__restrict__ inp, const int *__restrict__ row_ker,
                                 const int *__restrict__ row_lat, const int *__restrict__ seg,
                                 const float *__restrict__ vals, const long long *__restrict__ svoff,
                                 const int *__restrict__ srow, float *__restrict__ out32, int BC, int K, int Hi, int Wi,
                                 int Ho, int Wo, long long nseg)
    {
        const long long task = (long long)blockIdx.x;
        const long long ntask = nseg * (long long)BC;
        if (task >= ntask) return;
        const int s = (int)(task % nseg);
        const int bc = (int)(task / nseg);
        const int r = srow[s];
        if (r < 0) return;
        const int ring = seg[3 * s], start = seg[3 * s + 1];
        const int len = min(seg[3 * s + 2], Wo);
        if (len <= 0 || ring < 0 || ring >= Ho) return;
        const int P = Wo / Wi;
        const long long voff = svoff[s];
        const __half *x = inp + (((size_t)bc * K + row_ker[r]) * Hi + row_lat[r]) * Wi;
        float *o = out32 + ((size_t)bc * Ho + ring) * Wo;
        int st = start % Wo;
        if (st < 0) st += Wo;
        for (int q = threadIdx.x; q < Wi; q += blockDim.x) {
            const float xv = __half2float(x[q]);
            for (int j = 0; j < len; ++j) {
                int w = st + j + q * P;
                w %= Wo;
                atomicAdd(o + w, vals[voff + j] * xv);
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

    template <typename... KArgs, typename... Args>
    static void launch_pdl(void (*k)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Args... args)
    {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid;
        cfg.blockDim = block;
        cfg.dynamicSmemBytes = smem;
        cfg.stream = st;
        cudaLaunchAttribute at[1];
        at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        at[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = at;
        cfg.numAttrs = 1;
        cudaLaunchKernelEx(&cfg, k, static_cast<KArgs>(args)...);
    }

    template <int P, int MTW, int G, int MINB, bool PAIR, bool DO_SPLIT, int MAXT = 160, int STAGES = 3>
    static void launch_tc(const MainCfg &mc, int nctas, cudaStream_t stream, const __half *inp, const uint32_t *spglob,
                          const Entry *ents, const int *gstart, const int *gorder, const int *gheavy, const int4 *items,
                          const float *gscale, __half *out, float *scratch, int *scnt)
    {
        size_t smem = STAGES * (size_t)mc.stage_bytes;
        static const int attrs_initialized = []() {
            cudaFuncSetAttribute(k_disco_tc<P, MTW, G, MINB, PAIR, DO_SPLIT, MAXT, STAGES>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
            if constexpr (P == 1 && G == 3)
                cudaFuncSetAttribute(k_disco_tc<P, MTW, G, MINB, PAIR, DO_SPLIT, MAXT, STAGES>,
                                     cudaFuncAttributePreferredSharedMemoryCarveout, 100);
            return 0;
        }();
        (void)attrs_initialized;
        dim3 grid((unsigned)nctas);
        dim3 block(32 * (mc.CB * mc.WPC + 1));
        launch_pdl(k_disco_tc<P, MTW, G, MINB, PAIR, DO_SPLIT, MAXT, STAGES>, grid, block, smem, stream, inp, spglob,
                   ents, gstart, gorder, gheavy, items, gscale, out, scratch, scnt, mc);
    }

    void Run(ffi::TensorView inp, ffi::TensorView row_ker, ffi::TensorView row_lat, ffi::TensorView seg_off,
             ffi::TensorView seg, ffi::TensorView val_off, ffi::TensorView vals, int64_t nlat_out, int64_t nlon_out,
             ffi::TensorView out, const unsigned char *skip = nullptr)
    {
        const int B = (int)inp.size(0), C = (int)inp.size(1), K = (int)inp.size(2), Hi = (int)inp.size(3),
                  Wi = (int)inp.size(4);
        const int BC = B * C;
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

        if (BC <= 0 || Ho <= 0 || Wo <= 0) return;
        const int P = (Wi > 0) ? Wo / Wi : 0;

        // ---- choose path ----
        bool tc = (Wi > 0) && (P == 1 || P == 2) && (P * Wi == Wo) && (Wi % 8 == 0) && NROWS > 0 && NSEG > 0
            && NSEG < (1LL << 30) && (long long)K * Hi * Wi < (1LL << 31);
        const int MTW = (P == 2 && BC > 16) ? 3 : 6;
        MainCfg mc {};
        PrepCfg pc {};
        if (tc) {
            int mt_tot = (Wi + 127) / 128;
            int WPC = (mt_tot + MTW - 1) / MTW;
            if (WPC > 4) tc = false;
            int CB = tc ? std::max(1, 4 / WPC) : 1;
            int G = (P == 2) ? 2 : 3;
            int stages = 3;
            int XW = 128 * MTW * WPC;
            int XLMAX = 2 * XW + 32;
            int SPMAX = (int)align_up((size_t)P * NSPLIT * (Wi + 40), 4);
            int SEGMAX = 3;
            int off_sp = (int)align_up((size_t)CB * XLMAX * 2, 128);
            int off_ent = (int)align_up(off_sp + (size_t)SPMAX * 4, 16);
            int stage_bytes = (int)align_up(off_ent + sizeof(Entry), 128);
            if (stages * (size_t)stage_bytes > 200 * 1024) tc = false;
            int NG = (Ho + G - 1) / G;
            pc = PrepCfg {P, Wi, Wo, Ho, Hi, G, XW, XLMAX, SPMAX, SEGMAX, nullptr, skip};
            mc = MainCfg {BC,    K,     Hi,     Wi,     Ho,      Wo,         NG, WPC, CB, (BC + CB - 1) / CB,
                          XLMAX, SPMAX, SEGMAX, off_sp, off_ent, stage_bytes};
        }

        if (tc) {
            const int NG = mc.NG;
            // workspace (stream-ordered; everything in it is recomputed every call)
            const size_t spw_bound = (size_t)P * NSPLIT * ((size_t)NNZ + 40 * (size_t)NSEG);
            size_t o_sinfo = 0;
            size_t o_ents = align_up(o_sinfo + (size_t)NSEG * sizeof(SegInfo), 256);
            size_t o_rspw = align_up(o_ents + (size_t)NSEG * sizeof(Entry), 256);
            size_t o_rspoff = align_up(o_rspw + (size_t)NROWS * 4, 256);
            size_t o_zero = align_up(o_rspoff + (size_t)(NROWS + 1) * 8, 256);
            size_t o_gwork = o_zero; // unsigned long long[NG]
            size_t o_gcount = align_up(o_gwork + (size_t)NG * 8, 16);
            size_t o_gcursor = align_up(o_gcount + (size_t)NG * 4, 16);
            size_t o_gmax = align_up(o_gcursor + (size_t)NG * 4, 16);
            size_t zero_end = o_gmax + 16;
            size_t o_gstart = align_up(zero_end, 256);
            size_t o_gorder = align_up(o_gstart + (size_t)(NG + 1) * 4, 16);
            size_t o_gscale = align_up(o_gorder + (size_t)NG * 4, 16);
            size_t o_gheavy = o_gscale + 8;
            // split-K for the heaviest ring groups only when there are few channel blocks
            const int ncb = (BC + mc.CB - 1) / mc.CB;
            const int max_slots = 16;
            const int max_extra = (ncb <= 4) ? std::min(NG, 20) : (ncb <= 8) ? std::min(NG, 64) : 0;
            constexpr int nsm = 132;
            const double split_scale = (double)ncb / (2.0 * 2.0 * nsm);
            size_t o_items = align_up(o_gscale + 16, 256);
            size_t o_scnt = align_up(o_items + (size_t)(NG + max_extra) * 16, 256);
            size_t o_scratch = align_up(o_scnt + (size_t)max_slots * ncb * 4, 256);
            const int Gg = (P == 2) ? 2 : 3;
            size_t scratch_bytes = max_extra > 0 ? (size_t)max_slots * ncb * Gg * mc.CB * Wo * 4 : 0;
            size_t o_sp = align_up(o_scratch + scratch_bytes, 256);
            size_t total = o_sp + spw_bound * 4;
            unsigned char *ws = nullptr;
            cudaMallocAsync((void **)&ws, total, stream);
            cudaMemsetAsync(ws + o_zero, 0, zero_end - o_zero, stream);
            if (max_extra > 0) cudaMemsetAsync(ws + o_scnt, 0, o_sp - o_scnt, stream);
            SegInfo *d_sinfo = reinterpret_cast<SegInfo *>(ws + o_sinfo);
            Entry *d_ents = reinterpret_cast<Entry *>(ws + o_ents);
            int *d_rspw = reinterpret_cast<int *>(ws + o_rspw);
            long long *d_rspoff = reinterpret_cast<long long *>(ws + o_rspoff);
            unsigned long long *d_gwork = reinterpret_cast<unsigned long long *>(ws + o_gwork);
            int *d_gcount = reinterpret_cast<int *>(ws + o_gcount);
            int *d_gcursor = reinterpret_cast<int *>(ws + o_gcursor);
            int *d_gmax = reinterpret_cast<int *>(ws + o_gmax);
            pc.laps = d_gmax + 1;
            int *d_gstart = reinterpret_cast<int *>(ws + o_gstart);
            int *d_gorder = reinterpret_cast<int *>(ws + o_gorder);
            float *d_gscale = reinterpret_cast<float *>(ws + o_gscale);
            int *d_gheavy = reinterpret_cast<int *>(ws + o_gheavy);
            int4 *d_items = reinterpret_cast<int4 *>(ws + o_items);
            int *d_scnt = reinterpret_cast<int *>(ws + o_scnt);
            float *d_scratch = reinterpret_cast<float *>(ws + o_scratch);
            uint32_t *d_sp = reinterpret_cast<uint32_t *>(ws + o_sp);

            int pb = 32, pg = (NROWS + pb - 1) / pb;
            int mg = (int)std::min<long long>((NNZ + 255) / 256, 2 * 132);
            if (mg < 1) mg = 1;
            const int fg = std::max(pg, mg);
            k_prep_count_abs<<<fg, 256, 0, stream>>>(NROWS, d_so, d_seg, d_vo, pc, d_gcount, d_gwork, d_rspw, d_vals,
                                                     NNZ, d_gmax);
            launch_pdl(k_scan, dim3(1), dim3(1024), 0, stream, d_gcount, d_gwork, d_gstart, d_gorder, d_rspw, d_rspoff,
                       NROWS, d_gmax, d_gscale, d_gheavy, d_items, split_scale, max_extra, max_slots, NG);
            launch_pdl(k_prep_place, dim3(pg), dim3(pb), 0, stream, NROWS, d_so, d_seg, d_vo, d_rk, d_rl, pc, d_gstart,
                       d_rspoff, d_gcursor, d_ents, d_sinfo);
            unsigned sg_blocks = (unsigned)((NSEG * 32 + 255) / 256);
            const int nctas = (NG + max_extra) * ncb;
            if (P == 1) {
                launch_pdl(k_build_sp<1>, dim3(sg_blocks), dim3(256), 0, stream, NSEG, d_sinfo, d_vals, d_gscale, d_sp,
                           d_seg, Wo);
                if (max_extra > 0)
                    launch_tc<1, 6, 3, 3, false, true>(mc, nctas, stream, d_inp, d_sp, d_ents, d_gstart, d_gorder,
                                                       d_gheavy, d_items, d_gscale, d_out, d_scratch, d_scnt);
                else
                    launch_tc<1, 6, 3, 3, false, false>(mc, nctas, stream, d_inp, d_sp, d_ents, d_gstart, d_gorder,
                                                        d_gheavy, d_items, d_gscale, d_out, d_scratch, d_scnt);
            } else {
                launch_pdl(k_build_sp<2>, dim3(sg_blocks), dim3(256), 0, stream, NSEG, d_sinfo, d_vals, d_gscale, d_sp,
                           d_seg, Wo);
                if (BC > 16)
                    launch_tc<2, 3, 2, 4, true, false>(mc, nctas, stream, d_inp, d_sp, d_ents, d_gstart, d_gorder,
                                                       d_gheavy, d_items, d_gscale, d_out, d_scratch, d_scnt);
                else if (max_extra == 0)
                    launch_tc<2, 6, 2, 3, true, false>(mc, nctas, stream, d_inp, d_sp, d_ents, d_gstart, d_gorder,
                                                       d_gheavy, d_items, d_gscale, d_out, d_scratch, d_scnt);
                else
                    launch_tc<2, 6, 2, 2, false, true>(mc, nctas, stream, d_inp, d_sp, d_ents, d_gstart, d_gorder,
                                                       d_gheavy, d_items, d_gscale, d_out, d_scratch, d_scnt);
            }
            cudaFreeAsync(ws, stream);
            return;
        }

        // ---- general path: shared fp32 atomic scatter (any pscale, Wi, B*C, arc length) ----
        cr::RunFallback(inp, row_ker, row_lat, seg_off, seg, val_off, vals, nlat_out, nlon_out, out);
    }

} // namespace unroll1_impl
