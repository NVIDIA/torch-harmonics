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

// FFT evaluation of heavy output rows (pole caps), fp32 throughout.
//
// For one output row ho and one input latitude hi the contribution is a circular cross-correlation along
// longitude:  y_hi[n] = sum_d h_hi[k][d] * X[c, hi, (d + n) mod Wi],  out[c,k,ho,wo] = sum_hi y_hi[p*wo].
// In the frequency domain this is a per-frequency product  Y[m] = sum_hi conj(H_hi[k][m]) * Xhat[c,hi,m],
// so a row costs O(nh * Wi) per (c, k) instead of O(support * Wo).  Rows near the poles carry whole
// longitude rings (thousands of points), which makes them the most expensive rows of the direct path and
// the cheapest ones here.  A run-time cost model on the psi row (support size, number of runs, latitude
// band) decides per row; the direct paths skip the rows taken here.
//
//   classify : per row cost model, FFT row list, slots for the row's filter spectra, needed input rows
//   xhat     : forward real FFT of the needed input rows (two channels per complex FFT)
//   filt     : dense filter per (row, hi), forward FFT (two basis functions per complex FFT), stored conj
//   prod     : per (FFT row, channel tile): per-frequency product, fold for pscale > 1, inverse FFT
//              (two outputs per complex FFT), scale 1/Wi, fp16 store
// FFT: Stockham autosort, radix 4/2/3/5 passes from a run-time plan of the length's factorisation.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <algorithm>

namespace fftp
{

#define FFT_MAXPASS 24
#define FFT_NHMAX 64

    struct Plan {
        int n, np;
        int rad[FFT_MAXPASS];
    };

    // radix set {16, 9, 10, 8, 6, 5, 4, 3, 2}; largest first so that typical lengths need 3 passes
    static bool make_plan(int n, Plan &pl)
    {
        pl.n = n;
        pl.np = 0;
        if (n < 2) return false;
        int m = n;
        for (int R : {16, 9, 10, 8, 6, 5, 4, 3, 2})
            while (m % R == 0 && pl.np < FFT_MAXPASS) {
                pl.rad[pl.np++] = R;
                m /= R;
            }
        return m == 1;
    }

    // e^{-2 pi i e / R} for the composite radices (constant-bank operands after unrolling)
    __constant__ float2 c_w16[16]
        = {{1.f, -0.f}, {0.923879533f, -0.382683432f},  {0.707106781f, -0.707106781f},  {0.382683432f, -0.923879533f},
           {0.f, -1.f}, {-0.382683432f, -0.923879533f}, {-0.707106781f, -0.707106781f}, {-0.923879533f, -0.382683432f},
           {-1.f, 0.f}, {-0.923879533f, 0.382683432f},  {-0.707106781f, 0.707106781f},  {-0.382683432f, 0.923879533f},
           {0.f, 1.f},  {0.382683432f, 0.923879533f},   {0.707106781f, 0.707106781f},   {0.923879533f, 0.382683432f}};
    __constant__ float2 c_w9[9] = {{1.f, -0.f},
                                   {0.766044443f, -0.64278761f},
                                   {0.173648178f, -0.984807753f},
                                   {-0.5f, -0.866025404f},
                                   {-0.939692621f, -0.342020143f},
                                   {-0.939692621f, 0.342020143f},
                                   {-0.5f, 0.866025404f},
                                   {0.173648178f, 0.984807753f},
                                   {0.766044443f, 0.64278761f}};
    __constant__ float2 c_w10[10] = {{1.f, -0.f},
                                     {0.809016994f, -0.587785252f},
                                     {0.309016994f, -0.951056516f},
                                     {-0.309016994f, -0.951056516f},
                                     {-0.809016994f, -0.587785252f},
                                     {-1.f, 0.f},
                                     {-0.809016994f, 0.587785252f},
                                     {-0.309016994f, 0.951056516f},
                                     {0.309016994f, 0.951056516f},
                                     {0.809016994f, 0.587785252f}};
    __constant__ float2 c_w6[6] = {{1.f, -0.f}, {0.5f, -0.866025404f}, {-0.5f, -0.866025404f},
                                   {-1.f, 0.f}, {-0.5f, 0.866025404f}, {0.5f, 0.866025404f}};

    template <int R> __device__ __forceinline__ float2 cw(int e)
    {
        if constexpr (R == 16)
            return c_w16[e];
        else if constexpr (R == 9)
            return c_w9[e];
        else if constexpr (R == 10)
            return c_w10[e];
        else
            return c_w6[e];
    }

    __device__ __forceinline__ float2 cmul(float2 a, float2 b)
    {
        return make_float2(fmaf(a.x, b.x, -a.y * b.y), fmaf(a.x, b.y, a.y * b.x));
    }
    __device__ __forceinline__ float2 cadd(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
    __device__ __forceinline__ float2 csub(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
    // multiply by -i (forward) or +i (inverse)
    template <bool INV> __device__ __forceinline__ float2 rot(float2 a)
    {
        return INV ? make_float2(-a.y, a.x) : make_float2(a.y, -a.x);
    }

    template <int R, bool INV> __device__ __forceinline__ void dft(float2 (&v)[R]);

    // R = R1 * R2 in registers: X[k1 + R1 k2] = sum_n2 W_R2^{n2 k2} W_R^{n2 k1} sum_n1 W_R1^{n1 k1} v[R2 n1 + n2]
    template <int R1, int R2, bool INV> __device__ __forceinline__ void dft_ct(float2 (&v)[R1 * R2])
    {
        constexpr int R = R1 * R2;
        float2 u[R2][R1];
#pragma unroll
        for (int n2 = 0; n2 < R2; ++n2) {
            float2 a[R1];
#pragma unroll
            for (int n1 = 0; n1 < R1; ++n1) a[n1] = v[R2 * n1 + n2];
            dft<R1, INV>(a);
#pragma unroll
            for (int k1 = 0; k1 < R1; ++k1) {
                if (n2 * k1 == 0) {
                    u[n2][k1] = a[k1];
                } else {
                    float2 w = cw<R>(n2 * k1);
                    if (INV) w.y = -w.y;
                    u[n2][k1] = cmul(a[k1], w);
                }
            }
        }
#pragma unroll
        for (int k1 = 0; k1 < R1; ++k1) {
            float2 b[R2];
#pragma unroll
            for (int n2 = 0; n2 < R2; ++n2) b[n2] = u[n2][k1];
            dft<R2, INV>(b);
#pragma unroll
            for (int k2 = 0; k2 < R2; ++k2) v[k1 + R1 * k2] = b[k2];
        }
    }

    template <int R, bool INV> __device__ __forceinline__ void dft(float2 (&v)[R])
    {
        if constexpr (R == 16) {
            dft_ct<4, 4, INV>(v);
        } else if constexpr (R == 9) {
            dft_ct<3, 3, INV>(v);
        } else if constexpr (R == 10) {
            dft_ct<2, 5, INV>(v);
        } else if constexpr (R == 6) {
            dft_ct<2, 3, INV>(v);
        } else if constexpr (R == 2) {
            const float2 a = v[0], b = v[1];
            v[0] = cadd(a, b);
            v[1] = csub(a, b);
        } else if constexpr (R == 4) {
            const float2 t0 = cadd(v[0], v[2]), t1 = csub(v[0], v[2]);
            const float2 t2 = cadd(v[1], v[3]), t3 = rot<INV>(csub(v[1], v[3]));
            v[0] = cadd(t0, t2);
            v[2] = csub(t0, t2);
            v[1] = cadd(t1, t3);
            v[3] = csub(t1, t3);
        } else if constexpr (R == 8) {
            // split into even/odd output halves: a_q = v_q + v_{q+4}, b_q = (v_q - v_{q+4}) w8^q
            const float h = 0.70710678118654752f;
            float2 a[4], b[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                a[q] = cadd(v[q], v[q + 4]);
                b[q] = csub(v[q], v[q + 4]);
            }
            b[1] = INV ? make_float2(h * (b[1].x - b[1].y), h * (b[1].x + b[1].y)) :
                         make_float2(h * (b[1].x + b[1].y), h * (b[1].y - b[1].x));
            b[2] = rot<INV>(b[2]);
            b[3] = INV ? make_float2(-h * (b[3].x + b[3].y), h * (b[3].x - b[3].y)) :
                         make_float2(h * (b[3].y - b[3].x), -h * (b[3].x + b[3].y));
            dft<4, INV>(a);
            dft<4, INV>(b);
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                v[2 * r] = a[r];
                v[2 * r + 1] = b[r];
            }
        } else if constexpr (R == 3) {
            const float c = -0.5f, s = 0.86602540378443865f;
            const float2 t = cadd(v[1], v[2]);
            const float2 d = rot<INV>(csub(v[1], v[2])); // -i (v1 - v2) forward
            const float2 b = make_float2(fmaf(c, t.x, v[0].x), fmaf(c, t.y, v[0].y));
            v[0] = cadd(v[0], t);
            v[1] = make_float2(fmaf(s, d.x, b.x), fmaf(s, d.y, b.y));
            v[2] = make_float2(fmaf(-s, d.x, b.x), fmaf(-s, d.y, b.y));
        } else if constexpr (R == 5) {
            const float c1 = 0.30901699437494742f, s1 = 0.95105651629515357f;
            const float c2 = -0.80901699437494742f, s2 = 0.58778525229247313f;
            const float2 a1 = cadd(v[1], v[4]), b1 = csub(v[1], v[4]);
            const float2 a2 = cadd(v[2], v[3]), b2 = csub(v[2], v[3]);
            const float2 x0 = v[0];
            const float2 p1 = make_float2(x0.x + c1 * a1.x + c2 * a2.x, x0.y + c1 * a1.y + c2 * a2.y);
            const float2 p2 = make_float2(x0.x + c2 * a1.x + c1 * a2.x, x0.y + c2 * a1.y + c1 * a2.y);
            const float2 q1 = rot<INV>(make_float2(s1 * b1.x + s2 * b2.x, s1 * b1.y + s2 * b2.y));
            const float2 q2 = rot<INV>(make_float2(s2 * b1.x - s1 * b2.x, s2 * b1.y - s1 * b2.y));
            v[0] = make_float2(x0.x + a1.x + a2.x, x0.y + a1.y + a2.y);
            v[1] = cadd(p1, q1);
            v[4] = csub(p1, q1);
            v[2] = cadd(p2, q2);
            v[3] = csub(p2, q2);
        }
    }

    // x / d for 0 <= x < 2^22 through a float reciprocal plus one correction step
    __device__ __forceinline__ int fdiv(int x, int d, float inv)
    {
        int q = __float2int_rz(__int2float_rn(x) * inv);
        const int r = x - q * d;
        q += (r >= d) - (r < 0);
        return q;
    }

    // In-place decimation-in-time stage (radix R, butterfly stride st) over NB buffers of length N with an odd
    // buffer stride ZS.  Consecutive threads take the same butterfly of consecutive buffers: twiddle and
    // butterfly-table reads are warp broadcasts and the odd stride keeps the buffer accesses conflict-free.
    // T: this stage's twiddles [(q-1)*st + i] (direction applied), Bt: butterfly r -> (offset | i << 16).
    template <int R, bool INV>
    __device__ __forceinline__ void dit_stage(float2 *z, int NB, int ZS, int nbf, int st, const float2 *__restrict__ T,
                                              const int *__restrict__ Bt, int tid, int nth)
    {
        const int total = NB * nbf;
        int r = tid / NB, b = tid - (tid / NB) * NB;
        const int dr = nth / NB, db = nth - dr * NB;
        for (int t = tid; t < total; t += nth) {
            const int e = Bt[r];
            const int i = e >> 16;
            float2 *x = z + b * ZS + (e & 0xffff);
            float2 v[R];
            v[0] = x[0];
            if (st > 1) {
#pragma unroll
                for (int q = 1; q < R; ++q) v[q] = cmul(x[q * st], T[(q - 1) * st + i]);
            } else {
#pragma unroll
                for (int q = 1; q < R; ++q) v[q] = x[q * st];
            }
            dft<R, INV>(v);
#pragma unroll
            for (int q = 0; q < R; ++q) x[q * st] = v[q];
            b += db;
            r += dr;
            if (b >= NB) {
                b -= NB;
                ++r;
            }
        }
    }

    // Full in-place transform of NB buffers by the whole CTA (tables from build_fft_tables).  Input element m
    // must have been stored at position perm[m]; the output is in natural order.  Caller synchronises before;
    // returns after a final __syncthreads.
    // barrier over a group of nth threads (id 0: the whole CTA)
    __device__ __forceinline__ void grp_sync(int id, int nth)
    {
        if (id == 0)
            __syncthreads();
        else
            asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(nth) : "memory");
    }

    template <bool INV>
    __device__ void fft_batch(float2 *z, int NB, int ZS, const Plan &pl, const float2 *T, const int *Bt, int tid,
                              int nth, int bar = 0)
    {
        const int N = pl.n;
        int L = 1, toff = 0, boff = 0;
        for (int s = pl.np - 1; s >= 0; --s) {
            const int R = pl.rad[s];
            L *= R;
            const int st = L / R, nbf = N / R;
            const float2 *Ts = T + toff;
            const int *Bs = Bt + boff;
            switch (R) {
            case 16: dit_stage<16, INV>(z, NB, ZS, nbf, st, Ts, Bs, tid, nth); break;
            case 10: dit_stage<10, INV>(z, NB, ZS, nbf, st, Ts, Bs, tid, nth); break;
            case 9: dit_stage<9, INV>(z, NB, ZS, nbf, st, Ts, Bs, tid, nth); break;
            case 8: dit_stage<8, INV>(z, NB, ZS, nbf, st, Ts, Bs, tid, nth); break;
            case 6: dit_stage<6, INV>(z, NB, ZS, nbf, st, Ts, Bs, tid, nth); break;
            case 5: dit_stage<5, INV>(z, NB, ZS, nbf, st, Ts, Bs, tid, nth); break;
            case 4: dit_stage<4, INV>(z, NB, ZS, nbf, st, Ts, Bs, tid, nth); break;
            case 3: dit_stage<3, INV>(z, NB, ZS, nbf, st, Ts, Bs, tid, nth); break;
            default: dit_stage<2, INV>(z, NB, ZS, nbf, st, Ts, Bs, tid, nth); break;
            }
            toff += (R - 1) * st;
            boff += nbf;
            grp_sync(bar, nth);
        }
    }

    // Per-stage twiddles (N-1 entries in total, direction applied), butterfly tables (< N entries) and the
    // input permutation (mixed-radix digit reversal) of a plan, in shared memory.
    __device__ void build_fft_tables(float2 *T, int *Bt, uint16_t *perm, const float2 *__restrict__ gtw, const Plan &pl,
                                     bool inv, int tid, int nth)
    {
        const int N = pl.n;
        int L = 1, toff = 0, boff = 0;
        for (int s = pl.np - 1; s >= 0; --s) {
            const int R = pl.rad[s];
            L *= R;
            const int st = L / R, nbf = N / R, tws = N / L;
            for (int idx = tid; idx < (R - 1) * st; idx += nth) {
                const int q = idx / st + 1, i = idx - (q - 1) * st;
                float2 w = gtw[i * q * tws];
                if (inv) w.y = -w.y;
                T[toff + idx] = w;
            }
            for (int r = tid; r < nbf; r += nth) {
                const int blk = r / st, i = r - blk * st;
                Bt[boff + r] = (blk * L + i) | (i << 16);
            }
            toff += (R - 1) * st;
            boff += nbf;
        }
        for (int t = tid; t < N; t += nth) {
            int k = t, pos = 0, prod = 1;
            for (int s = 0; s < pl.np; ++s) {
                const int R = pl.rad[s];
                const int q = k % R;
                k /= R;
                prod *= R;
                pos += q * (N / prod);
            }
            perm[t] = (uint16_t)pos;
        }
    }

    // ------------------------------------------------------------------------------------------------
    struct Params {
        const __half *x;
        const long long *idx;
        const __half *val;
        const long long *off;
        __half *out;
        int BC, Hi, Wi, Ho, Wo, K, KP, p;
        int M, Mp, Mo; // Wi/2+1 input half spectrum, padded row stride, Wo/2+1 output half spectrum
        int halo;      // direct-path halo per run, for the cost model
        float cdir, cfh, cfo, lam;
        int slotcap;
        int nct, cb;    // channel tiles / channels per tile of the product kernel
        int *counters;  // [0] FFT rows, [1] slots
        int *rowflag;   // [Ho] 1 = FFT row
        int4 *rows;     // [Ho] (ho, hmin, nh, slot)
        int2 *slotinfo; // [slotcap] (ho, hi)
        int *needed;    // [Hi]
        float2 *tw_in;  // [Wi]
        float2 *tw_out; // [Wo]
        float2 *xh;     // [Hi][BC][Mp]
        float2 *gf;     // [slotcap][K][Mp]  conj(filter spectrum)
        Plan pin, pout;
    };

    __global__ void twiddle_kernel(const __grid_constant__ Params P)
    {
        const int t = blockIdx.x * blockDim.x + threadIdx.x;
        if (t < P.Wi) {
            double s, c;
            sincospi(2.0 * (double)t / (double)P.Wi, &s, &c);
            P.tw_in[t] = make_float2((float)c, (float)-s);
        }
        if (t < P.Wo) {
            double s, c;
            sincospi(2.0 * (double)t / (double)P.Wo, &s, &c);
            P.tw_out[t] = make_float2((float)c, (float)-s);
        }
    }

    __global__ void __launch_bounds__(512) classify_kernel(const __grid_constant__ Params P)
    {
        const int ho = blockIdx.x, tid = threadIdx.x;
        const long long a = P.off[ho], b = P.off[ho + 1];
        __shared__ int s_min, s_max, s_runs, s_bad, s_flag, s_hmin, s_nh, s_slot, s_lim;
        if (tid == 0) {
            s_min = 0x7fffffff;
            s_max = -1;
            s_runs = 0;
            s_bad = 0;
        }
        __syncthreads();
        int lmin = 0x7fffffff, lmax = -1, lruns = 0, lbad = 0;
        // four consecutive points per thread, one 16-byte (hi, wi) load each
        const longlong2 *ip = reinterpret_cast<const longlong2 *>(P.idx);
        for (long long n0 = a + 4 * tid; n0 < b; n0 += 4 * blockDim.x) {
            longlong2 pv = make_longlong2(-1, -2);
            if (n0 > a) pv = __ldg(ip + n0 - 1);
            longlong2 v[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) v[u] = n0 + u < b ? __ldg(ip + n0 + u) : make_longlong2(-1, -1);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (n0 + u >= b) break;
                const long long h = v[u].x, w = v[u].y;
                if (h < 0 || h >= P.Hi || w < 0 || w >= P.Wi)
                    lbad = 1;
                else {
                    lmin = min(lmin, (int)h);
                    lmax = max(lmax, (int)h);
                }
                lruns += (n0 + u == a) || (pv.x != h) || (pv.y + 1 != w);
                pv = v[u];
            }
        }
        if (lbad) atomicOr(&s_bad, 1);
        atomicMin(&s_min, lmin);
        atomicMax(&s_max, lmax);
        atomicAdd(&s_runs, lruns);
        __syncthreads();
        if (tid == 0) {
            int flag = 0, slot = -1, nh = 0, lim = 0;
            const long long nnz = b - a;
            if (nnz > 0 && !s_bad && s_max >= s_min) {
                nh = s_max - s_min + 1;
                if (nh <= FFT_NHMAX) {
                    const float direct = ((float)nnz + (float)s_runs * (float)(P.halo + 4)) * P.cdir;
                    const float fft = (float)nh * P.cfh + P.cfo;
                    if (direct > P.lam * fft) {
                        slot = atomicAdd(&P.counters[1], nh);
                        if (slot + nh <= P.slotcap) {
                            const int f = atomicAdd(&P.counters[0], 1);
                            P.rows[f] = make_int4(ho, s_min, nh, slot);
                            flag = 1;
                            lim = nh;
                        } else if (slot < P.slotcap) {
                            lim = P.slotcap - slot; // partially reserved slots: mark them unused
                        }
                    }
                }
            }
            P.rowflag[ho] = flag;
            s_flag = flag;
            s_hmin = s_min;
            s_nh = nh;
            s_slot = slot;
            s_lim = lim;
        }
        __syncthreads();
        for (int j = tid; j < s_lim; j += blockDim.x) {
            if (s_flag) {
                P.slotinfo[s_slot + j] = make_int2(ho, s_hmin + j);
                P.needed[s_hmin + j] = 1;
            } else {
                P.slotinfo[s_slot + j] = make_int2(-1, -1);
            }
        }
    }

// forward FFT of the needed input rows: one CTA per (row, NBX channel pairs), all pairs in lockstep
#define XH_THREADS 512
    __global__ void __launch_bounds__(XH_THREADS) xhat_kernel(const __grid_constant__ Params P, int NBX)
    {
        extern __shared__ float2 fsm[];
        const int hi = blockIdx.x;
        if (!P.needed[hi]) return;
        const int N = P.Wi, ZS = N | 1, tid = threadIdx.x;
        float2 *z = fsm;                                       // [NBX][ZS]
        float2 *T = z + (size_t)NBX * ZS;                      // [N] stage twiddles
        int *Bt = reinterpret_cast<int *>(T + N);              // [N] butterfly tables
        uint16_t *perm = reinterpret_cast<uint16_t *>(Bt + N); // [N]
        build_fft_tables(T, Bt, perm, P.tw_in, P.pin, false, tid, XH_THREADS);
        __syncthreads();
        const int pr0 = blockIdx.y * NBX;
        const int nb = min(NBX, (P.BC + 1) / 2 - pr0);
        // four longitudes per thread (N % 4 == 0): 8-byte loads of both channels of the pair
        const int N4 = N >> 2;
        const float iN4 = 1.0f / (float)N4;
        for (int t = tid; t < nb * N4; t += XH_THREADS) {
            const int b = fdiv(t, N4, iN4), n = 4 * (t - b * N4);
            const int c0 = 2 * (pr0 + b), c1 = c0 + 1;
            const uint2 ua = __ldg(reinterpret_cast<const uint2 *>(P.x + ((size_t)c0 * P.Hi + hi) * N + n));
            uint2 ub = make_uint2(0u, 0u);
            if (c1 < P.BC) ub = __ldg(reinterpret_cast<const uint2 *>(P.x + ((size_t)c1 * P.Hi + hi) * N + n));
            const float2 a01 = __half22float2(*reinterpret_cast<const __half2 *>(&ua.x));
            const float2 a23 = __half22float2(*reinterpret_cast<const __half2 *>(&ua.y));
            const float2 b01 = __half22float2(*reinterpret_cast<const __half2 *>(&ub.x));
            const float2 b23 = __half22float2(*reinterpret_cast<const __half2 *>(&ub.y));
            float2 *zr = z + b * ZS;
            zr[perm[n]] = make_float2(a01.x, b01.x);
            zr[perm[n + 1]] = make_float2(a01.y, b01.y);
            zr[perm[n + 2]] = make_float2(a23.x, b23.x);
            zr[perm[n + 3]] = make_float2(a23.y, b23.y);
        }
        __syncthreads();
        fft_batch<false>(z, nb, ZS, P.pin, T, Bt, tid, XH_THREADS);
        const float iM = 1.0f / (float)P.M;
        for (int t = tid; t < nb * P.M; t += XH_THREADS) {
            const int b = fdiv(t, P.M, iM), m = t - b * P.M;
            const int c0 = 2 * (pr0 + b), c1 = c0 + 1;
            const float2 zm = z[b * ZS + m], zn = z[b * ZS + (m == 0 ? 0 : N - m)];
            P.xh[((size_t)hi * P.BC + c0) * P.Mp + m] = make_float2(0.5f * (zm.x + zn.x), 0.5f * (zm.y - zn.y));
            if (c1 < P.BC)
                P.xh[((size_t)hi * P.BC + c1) * P.Mp + m] = make_float2(0.5f * (zm.y + zn.y), 0.5f * (zn.x - zm.x));
        }
    }

// filter spectra: one CTA per slot (row, hi); basis functions 2q, 2q+1 share complex buffer q
#define FL_THREADS 512
    __global__ void __launch_bounds__(FL_THREADS) filt_kernel(const __grid_constant__ Params P)
    {
        extern __shared__ float2 fsm[];
        const int N = P.Wi, K = P.K, KP2 = (K + 1) >> 1, ZS = N | 1, tid = threadIdx.x;
        float2 *z = fsm;                                       // [KP2][ZS]
        float2 *T = z + (size_t)KP2 * ZS;                      // [N] stage twiddles
        int *Bt = reinterpret_cast<int *>(T + N);              // [N] butterfly tables
        uint16_t *perm = reinterpret_cast<uint16_t *>(Bt + N); // [N]
        build_fft_tables(T, Bt, perm, P.tw_in, P.pin, false, tid, FL_THREADS);
        const int nslot = min(*(volatile int *)&P.counters[1], P.slotcap);
        const float iM = 1.0f / (float)P.M;
        for (int s = blockIdx.x; s < nslot; s += gridDim.x) {
            const int2 si = P.slotinfo[s];
            if (si.x < 0) continue;
            __syncthreads();
            for (int t = tid; t < KP2 * ZS; t += FL_THREADS) z[t] = make_float2(0.f, 0.f);
            __syncthreads();
            const long long a = P.off[si.x], b = P.off[si.x + 1];
            // scan the row's points for this latitude: four independent index loads per thread in flight
            for (long long n0 = a + tid; n0 < b; n0 += 4 * FL_THREADS) {
                long long hv[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const long long n = n0 + (long long)u * FL_THREADS;
                    hv[u] = n < b ? __ldg(P.idx + 2 * n) : -1;
                }
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    if (hv[u] != si.y) continue;
                    const long long n = n0 + (long long)u * FL_THREADS;
                    const int d = perm[(int)__ldg(P.idx + 2 * n + 1)];
                    for (int k = 0; k < K; ++k) {
                        const float v = __half2float(P.val[n * P.KP + k]);
                        atomicAdd(reinterpret_cast<float *>(z + (k >> 1) * ZS + d) + (k & 1), v);
                    }
                }
            }
            __syncthreads();
            fft_batch<false>(z, KP2, ZS, P.pin, T, Bt, tid, FL_THREADS);
            for (int t = tid; t < KP2 * P.M; t += FL_THREADS) {
                const int q = fdiv(t, P.M, iM), m = t - q * P.M;
                const float2 zm = z[q * ZS + m], zn = z[q * ZS + (m == 0 ? 0 : N - m)];
                const int k0 = 2 * q, k1 = k0 + 1;
                // H = spectrum of the real filter; store conj(H)
                P.gf[((size_t)s * K + k0) * P.Mp + m] = make_float2(0.5f * (zm.x + zn.x), -0.5f * (zm.y - zn.y));
                if (k1 < K)
                    P.gf[((size_t)s * K + k1) * P.Mp + m] = make_float2(0.5f * (zm.y + zn.y), -0.5f * (zn.x - zm.x));
            }
        }
    }

    // per-frequency product of one channel pair (cl, cl + CBH) for all basis functions; CJ: the alias
    // frequency lies above Wi/2 and enters as the conjugate of its mirror
    template <int KK, bool CJ>
    __device__ __forceinline__ void prod_acc(float2 (&acc)[2][KK], const float2 *__restrict__ xp,
                                             const float2 *__restrict__ gp, size_t xstep, size_t xhi, size_t gstep,
                                             size_t gk, int nh, int K)
    {
        const float sg = CJ ? -1.f : 1.f;
#pragma unroll 4
        for (int j = 0; j < nh; ++j) {
            float2 xl, xu;
            xl = __ldg(xp);
            xu = __ldg(xp + xhi);
#pragma unroll
            for (int k = 0; k < KK; ++k) {
                if (k >= K) break;
                const float2 g = __ldg(gp + (size_t)k * gk);
                const float gy = sg * g.y; // folded into the FMA operands by the compiler (sg is a constant)
                acc[0][k].x = fmaf(g.x, xl.x, fmaf(-gy, sg * xl.y, acc[0][k].x));
                acc[0][k].y = fmaf(g.x, sg * xl.y, fmaf(gy, xl.x, acc[0][k].y));
                acc[1][k].x = fmaf(g.x, xu.x, fmaf(-gy, sg * xu.y, acc[1][k].x));
                acc[1][k].y = fmaf(g.x, sg * xu.y, fmaf(gy, xu.x, acc[1][k].y));
            }
            xp += xstep;
            gp += gstep;
        }
    }

// product + inverse FFT + store; persistent over items (FFT row, channel tile).  A tile holds CB channels
// (CB even); channels cp and cp + CB/2 share one complex buffer per basis function (real / imaginary part).
#define PROD_THREADS 512
    template <int KT> __global__ void __launch_bounds__(PROD_THREADS, 1) prod_kernel(const __grid_constant__ Params P)
    {
        extern __shared__ float2 fsm[];
        constexpr int KK = KT ? KT : 16;
        const int No = P.Wo, K = KT ? KT : P.K, ZS = No | 1;
        const int CB = P.cb, CBH = CB >> 1, NB = CBH * K;
        // two independent groups of GT threads, each with its own spectra buffers, so that one group's
        // product phase (L2 loads + FMA) overlaps the other group's transform phase (shared memory)
        constexpr int GT = PROD_THREADS / 2;
        float2 *T = fsm;                                        // [No] stage twiddles
        int *Bt = reinterpret_cast<int *>(T + No);              // [No] butterfly tables
        uint16_t *perm = reinterpret_cast<uint16_t *>(Bt + No); // [No]
        build_fft_tables(T, Bt, perm, P.tw_out, P.pout, true, threadIdx.x, PROD_THREADS);
        __syncthreads();
        const int grp = threadIdx.x / GT, bar = 1 + grp;
        float2 *zb
            = reinterpret_cast<float2 *>(reinterpret_cast<uint8_t *>(perm + No) + 16 - ((uintptr_t)(perm + No) & 15))
            + (size_t)grp * NB * ZS; // [NB][ZS] of this group
        const int tid = threadIdx.x - grp * GT, lane = tid & 31, warp = tid >> 5;
        const int nfft = *(volatile int *)&P.counters[0];
        const int items = nfft * P.nct;
        const int half = P.Wi >> 1;
        const float scale = 1.0f / (float)P.Wi;
        const int FPW = 32 / CBH; // frequencies per warp step (lanes of one frequency hold its channel pairs)
        const int nws = (P.Mo + FPW - 1) / FPW;
        const size_t xstep = (size_t)P.BC * P.Mp, gstep = (size_t)K * P.Mp;
        for (int it = 2 * blockIdx.x + grp; it < items; it += 2 * gridDim.x) {
            const int f = it / P.nct, ct = it - f * P.nct;
            const int4 rw = P.rows[f];
            const int ho = rw.x, hmin = rw.y, nh = rw.z, slot = rw.w;
            const int c0 = ct * CB;
            grp_sync(bar, GT); // previous item's buffers are free
            // ---- phase A: per-frequency product into the paired spectra (stored at the DIT input positions)
            for (int ws = warp; ws < nws; ws += GT / 32) {
                const int mp = ws * FPW + lane % FPW, cp = lane / FPW;
                if (mp >= P.Mo) continue;
                // channels past BC read a valid channel's data; their outputs are never stored
                const int cl = min(c0 + cp, P.BC - 1), chh = min(c0 + cp + CBH, P.BC - 1);
                const size_t xhi = (size_t)(chh - cl) * P.Mp;
                float2 acc[2][KK];
#pragma unroll
                for (int k = 0; k < KK; ++k) {
                    acc[0][k] = make_float2(0.f, 0.f);
                    acc[1][k] = make_float2(0.f, 0.f);
                }
                for (int r = 0; r < P.p; ++r) {
                    const int m = mp + r * No;
                    if (m <= half) {
                        prod_acc<KK, false>(acc, P.xh + ((size_t)hmin * P.BC + cl) * P.Mp + m,
                                            P.gf + (size_t)slot * gstep + m, xstep, xhi, gstep, P.Mp, nh, K);
                    } else {
                        const int m2 = P.Wi - m;
                        prod_acc<KK, true>(acc, P.xh + ((size_t)hmin * P.BC + cl) * P.Mp + m2,
                                           P.gf + (size_t)slot * gstep + m2, xstep, xhi, gstep, P.Mp, nh, K);
                    }
                }
                const bool edge = (mp == 0) || (2 * mp == No);
                const int p0 = perm[mp], p1 = edge ? 0 : perm[No - mp];
#pragma unroll
                for (int k = 0; k < KK; ++k) {
                    if (k >= K) break;
                    const float2 y1 = acc[0][k], y2 = acc[1][k];
                    float2 *z = zb + (cp * K + k) * ZS;
                    if (edge) {
                        z[p0] = make_float2(y1.x, y2.x);
                    } else {
                        z[p0] = make_float2(y1.x - y2.y, y1.y + y2.x);
                        z[p1] = make_float2(y1.x + y2.y, y2.x - y1.y);
                    }
                }
            }
            grp_sync(bar, GT);
            // ---- phase B: in-place inverse transform of all NB buffers (natural-order output)
            fft_batch<true>(zb, NB, ZS, P.pout, T, Bt, tid, GT, bar);
            // ---- output: buffer q = cp*K + k holds rows (c0+cp, k) and (c0+cp+CBH, k); four longitudes per thread
            {
                const int Nq = No >> 2, total = NB * Nq;
                const float inq = 1.0f / (float)Nq;
                for (int t = tid; t < total; t += GT) {
                    const int q = fdiv(t, Nq, inq);
                    const int n4 = t - q * Nq;
                    const int cp = q / K, k = q - cp * K;
                    const float2 *z = zb + q * ZS + 4 * n4;
                    const float2 a0 = z[0], a1 = z[1], a2 = z[2], a3 = z[3];
                    const int cl = c0 + cp, ch = cl + CBH;
                    if (cl < P.BC) {
                        const __half2 h0 = __floats2half2_rn(a0.x * scale, a1.x * scale);
                        const __half2 h1 = __floats2half2_rn(a2.x * scale, a3.x * scale);
                        uint2 u;
                        u.x = *reinterpret_cast<const uint32_t *>(&h0);
                        u.y = *reinterpret_cast<const uint32_t *>(&h1);
                        *(reinterpret_cast<uint2 *>(P.out + (((size_t)cl * K + k) * P.Ho + ho) * No) + n4) = u;
                    }
                    if (ch < P.BC) {
                        const __half2 h0 = __floats2half2_rn(a0.y * scale, a1.y * scale);
                        const __half2 h1 = __floats2half2_rn(a2.y * scale, a3.y * scale);
                        uint2 u;
                        u.x = *reinterpret_cast<const uint32_t *>(&h0);
                        u.y = *reinterpret_cast<const uint32_t *>(&h1);
                        *(reinterpret_cast<uint2 *>(P.out + (((size_t)ch * K + k) * P.Ho + ho) * No) + n4) = u;
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------------------------------------
    // host side
    // ------------------------------------------------------------------------------------------------
    static void *g_fws = nullptr;
    static size_t g_fws_size = 0;
    static void *fws_get(size_t bytes, cudaStream_t st)
    {
        if (bytes > g_fws_size) {
            if (g_fws) {
                cudaStreamSynchronize(st);
                cudaFree(g_fws);
            }
            if (cudaMalloc(&g_fws, bytes) != cudaSuccess) {
                g_fws = nullptr;
                g_fws_size = 0;
                return nullptr;
            }
            g_fws_size = bytes;
        }
        return g_fws;
    }

#define FFT_SMEM_MAX 232448

    // shared memory of the batched transform kernels: NB buffers of stride N|1, twiddles, permutation
    static size_t fft_smem(int N, int NB) { return ((size_t)NB * (N | 1) + N) * sizeof(float2) + (size_t)N * 6 + 32; }

    template <int KT> static void launch_prod(const Params &P, size_t smem, int nsm, cudaStream_t st)
    {
        auto kern = prod_kernel<KT>;
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        kern<<<nsm, PROD_THREADS, smem, st>>>(P);
    }

    // Sets up and launches the FFT path; returns the row flag array (nullptr: path not used).
    // halo / cdir describe the direct path that takes the remaining rows (cost model input).
    static const int *run(const __half *x, const long long *idx, const __half *val, const long long *off, __half *out,
                          int BC, int Hi, int Wi, int Ho, int Wo, int K, int KP, long long NNZ, int halo, float cdir,
                          int nsm, cudaStream_t st)
    {
        if (BC <= 0 || Ho <= 0 || Wo <= 0 || Wi % Wo != 0 || K <= 0 || K > 16 || K > KP || NNZ <= 0) return nullptr;
        if (Hi <= 0) return nullptr;
        Params P;
        if (!make_plan(Wi, P.pin) || !make_plan(Wo, P.pout)) return nullptr;
        const int p = Wi / Wo;
        const int KP2 = (K + 1) / 2;
        // shared memory of the batched transforms of the input rows and of the filters
        if (Wi > 65535 || fft_smem(Wi, KP2) > FFT_SMEM_MAX) return nullptr;
        // channel tile of the product kernel: as many channels as the spectra buffers allow
        // (even channel count; the in-place transform needs no scratch, so the spectra take the whole smem)
        if ((Wo & 3) || Wo > 65535) return nullptr;
        int cb = 0;
        for (int c : {16, 8, 4, 2}) {                            // c/2 channel pairs share a warp's lanes per frequency
            if (c > 2 && c / 2 >= BC) continue;                  // no wider than the channel count needs
            if (fft_smem(Wo, 2 * (c / 2) * K) <= FFT_SMEM_MAX) { // two groups of buffers
                cb = c;
                break;
            }
        }
        if (cb == 0) return nullptr;
        P.x = x;
        P.idx = idx;
        P.val = val;
        P.off = off;
        P.out = out;
        P.BC = BC;
        P.Hi = Hi;
        P.Wi = Wi;
        P.Ho = Ho;
        P.Wo = Wo;
        P.K = K;
        P.KP = KP;
        P.p = p;
        P.M = Wi / 2 + 1;
        P.Mp = (P.M + 3) & ~3;
        P.Mo = Wo / 2 + 1;
        P.halo = halo;
        P.cb = cb;
        P.nct = (BC + cb - 1) / cb;
        // cost model (per channel and output row, ~SM cycles): direct = support columns x cdir; FFT = product
        // (FMA + operand traffic) per latitude of the band + inverse transform
        {
            const float lg = log2f((float)Wo);
            P.cdir = cdir;
            // + filter spectra (independent of the channel count; calibrated at 256 channels)
            P.cfh = (float)K * (float)P.M * (0.0625f + 0.333f * (1.0f / cb + 1.0f / K))
                + std::max(0.f, 35000.f * (1.0f / (float)BC - 1.0f / 256.f));
            P.cfo = (float)K * (float)Wo * lg * 0.0195f;
            P.lam = 1.65f; // calibrated on measured per-row costs of both paths
        }
        const size_t gslot = (size_t)K * P.Mp * sizeof(float2);
        long long slotcap = (512ll << 20) / (long long)gslot;
        slotcap = std::min<long long>(slotcap, (long long)Ho * std::min(FFT_NHMAX, Hi));
        if (slotcap < 1) return nullptr;
        P.slotcap = (int)slotcap;
        const size_t xh_bytes = (size_t)Hi * BC * P.Mp * sizeof(float2);
        if (xh_bytes > (3ull << 30)) return nullptr;
        auto al = [](size_t v) { return (v + 255) & ~(size_t)255; };
        const size_t o_cnt = 0;
        const size_t o_need = o_cnt + 256;
        const size_t o_flag = o_need + al((size_t)Hi * 4);
        const size_t o_rows = o_flag + al((size_t)Ho * 4);
        const size_t o_si = o_rows + al((size_t)Ho * 16);
        const size_t o_twi = o_si + al((size_t)slotcap * 8);
        const size_t o_two = o_twi + al((size_t)Wi * 8);
        const size_t o_xh = o_two + al((size_t)Wo * 8);
        const size_t o_gf = o_xh + al(xh_bytes);
        const size_t total = o_gf + al((size_t)slotcap * gslot);
        uint8_t *ws = (uint8_t *)fws_get(total, st);
        if (!ws) return nullptr;
        P.counters = (int *)(ws + o_cnt);
        P.needed = (int *)(ws + o_need);
        P.rowflag = (int *)(ws + o_flag);
        P.rows = (int4 *)(ws + o_rows);
        P.slotinfo = (int2 *)(ws + o_si);
        P.tw_in = (float2 *)(ws + o_twi);
        P.tw_out = (float2 *)(ws + o_two);
        P.xh = (float2 *)(ws + o_xh);
        P.gf = (float2 *)(ws + o_gf);
        cudaMemsetAsync(ws, 0, o_flag, st); // counters + needed flags
        const int tmax = std::max(Wi, Wo);
        twiddle_kernel<<<(tmax + 255) / 256, 256, 0, st>>>(P);
        classify_kernel<<<Ho, 512, 0, st>>>(P);
        {
            int NBX = 16;
            while (NBX > 1 && fft_smem(Wi, NBX) > FFT_SMEM_MAX) --NBX;
            const size_t sm = fft_smem(Wi, NBX);
            cudaFuncSetAttribute(xhat_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
            const int npair = (BC + 1) / 2;
            dim3 g((unsigned)Hi, (unsigned)((npair + NBX - 1) / NBX));
            xhat_kernel<<<g, XH_THREADS, sm, st>>>(P, NBX);
        }
        {
            const size_t sm = fft_smem(Wi, KP2);
            cudaFuncSetAttribute(filt_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
            filt_kernel<<<3 * nsm, FL_THREADS, sm, st>>>(P);
        }
        {
            const size_t sm = fft_smem(Wo, 2 * (cb / 2) * K);
            if (K == 9)
                launch_prod<9>(P, sm, nsm, st);
            else
                launch_prod<0>(P, sm, nsm, st);
        }
        return P.rowflag;
    }

} // namespace fftp
