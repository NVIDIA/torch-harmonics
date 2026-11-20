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

// Frequency-domain path for the output rings whose direct (tensor-core Toeplitz) cost is high, i.e. rings
// reached by long arcs (polar caps). See the comment at the top of kernel.cu.
//
// For an output ring h, with x_r the length-Wi input row of psi row r and p_a the length-Wo psi row of arc a,
//   out_h = sum_a p_a (*) up_P(x_r)   (circular convolution of length Wo, up_P inserts P-1 zeros),
// so with real DFTs (K in [0, Wo/2]):  OUT_h[K] = sum_{(r,h) pairs} PSI_{r,h}[K] * X_r[K mod Wi].
// Arcs of one row on one ring form one pair (their psi rows are summed); arcs of any length and wrap are exact.
//   cost/select: per ring, a runtime cost model of the direct path vs this path; the smallest ratio threshold
//                whose flagged rings fit the buffers is chosen. The direct paths skip arcs on flagged rings.
//   place:       pairs (row, flagged ring) and visits (row, group of FFT_NSLOT flagged rings).
//   psi:         PSI spectra of the pairs (real FFT of length Wo of the pair's psi row).
//   fwd:         spectra X_r of the selected input rows for all channels (real FFT of length Wi), fp32 in global.
//   main:        work item (ring group, channel tile): OUT[slot][c][K] += PSI * X in registers over the visits,
//                then an inverse real FFT per (ring, channel) and one fp16 rounding.
// FFTs are mixed-radix Stockham (radices 8, 4, 2, 3, 5 and generic odd primes up to 31) on a runtime plan
// derived from the factorisation of Wi/2 and Wo/2. Everything is fp32 (data, twiddles, accumulation).
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <algorithm>

namespace fftp
{

    // Programmatic dependent launches overlap host dispatch with the predecessor grid draining.
    __device__ __forceinline__ void fx_pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
    __device__ __forceinline__ void fx_pdl_trigger()
    {
        asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
    }

#define FFT_MAXPASS 24
#define FFT_NSLOT 8      // flagged rings per ring group (contraction work item)
#define FFT_NRG_CAP 64   // ring groups (FFT_NSLOT rings each) the frequency-domain path may take
#define FFT_CON_MAXT 256 // contraction CTA threads
#define FFT_CV 4         // contraction: visits per (double-buffered) stage
#define FFT_PSI_NB 8     // pairs per PSI CTA
#define FFT_MAXGEN 31    // largest generic radix

    struct Plan {
        int M, np;
        int R[FFT_MAXPASS];
        int Ns[FFT_MAXPASS];    // product of the earlier radices
        float rMR[FFT_MAXPASS]; // 1 / (M / R)
        float rNs[FFT_MAXPASS]; // 1 / Ns
    };

    // shared-memory copy of a plan (avoids dynamically indexed kernel parameters)
    struct SPlan {
        int R[FFT_MAXPASS], Ns[FFT_MAXPASS];
        float rMR[FFT_MAXPASS], rNs[FFT_MAXPASS];
    };
    __device__ __forceinline__ void load_plan(SPlan &sp, const Plan &p, int tid)
    {
        if (tid < p.np) {
            sp.R[tid] = p.R[tid];
            sp.Ns[tid] = p.Ns[tid];
            sp.rMR[tid] = p.rMR[tid];
            sp.rNs[tid] = p.rNs[tid];
        }
    }

    struct FCfg {
        int Wi, Wo, P, Ho, Hi, K, BC, NROWS;
        int Mi, Mo, NKi, NKo, NKip, NKop; // NK*p: spectrum row strides (even, 16-byte rows for bulk copies)
        int Cb, ntile, nrg_max, cap_pairs, cap_rows, nbf, nbi, nst;
        int F, nfc; // contraction frequency chunk (output bins) and number of chunks
        float dunit, fcost_arc, fcost_ring, theta0;
        Plan pi, po;
    };

    struct __align__(16) Pair {
        int sb, se;     // arcs [sb, se) of the row, all on one flagged ring
        long long voff; // value offset of arc sb
    };

    struct __align__(16) Visit {
        int xrow, pad0, pad1, pad2;
        int pidx[FFT_NSLOT]; // pair of each ring slot, -1 if the row does not reach it
    };

    __device__ __forceinline__ float2 cmul(float2 a, float2 b)
    {
        return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }
    __device__ __forceinline__ float2 cadd(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
    __device__ __forceinline__ float2 csub(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
    __device__ __forceinline__ float2 cconj(float2 a) { return make_float2(a.x, -a.y); }
    // a * (DIR * i)
    template <int DIR> __device__ __forceinline__ float2 mul_si(float2 a)
    {
        return DIR > 0 ? make_float2(-a.y, a.x) : make_float2(a.y, -a.x);
    }

    // R-point DFTs, X[q] = sum_r v[r] exp(DIR * 2 pi i r q / R)  (DIR = -1 forward, +1 inverse)
    template <int DIR> __device__ __forceinline__ void dft2(float2 *v)
    {
        float2 a = v[0], b = v[1];
        v[0] = cadd(a, b);
        v[1] = csub(a, b);
    }
    template <int DIR> __device__ __forceinline__ void dft3(float2 *v)
    {
        const float h = 0.86602540378443864676f;
        float2 t = cadd(v[1], v[2]);
        float2 u = mul_si<DIR>(csub(v[1], v[2]));
        u.x *= h;
        u.y *= h;
        float2 m = make_float2(v[0].x - 0.5f * t.x, v[0].y - 0.5f * t.y);
        v[0] = cadd(v[0], t);
        v[1] = cadd(m, u);
        v[2] = csub(m, u);
    }
    template <int DIR> __device__ __forceinline__ void dft4(float2 *v)
    {
        float2 t0 = cadd(v[0], v[2]), t1 = csub(v[0], v[2]);
        float2 t2 = cadd(v[1], v[3]), t3 = mul_si<DIR>(csub(v[1], v[3]));
        v[0] = cadd(t0, t2);
        v[2] = csub(t0, t2);
        v[1] = cadd(t1, t3);
        v[3] = csub(t1, t3);
    }
    template <int DIR> __device__ __forceinline__ void dft5(float2 *v)
    {
        const float c1 = 0.30901699437494742410f, c2 = -0.80901699437494742410f;
        const float s1 = 0.95105651629515357212f, s2 = 0.58778525229247312917f;
        float2 t1 = cadd(v[1], v[4]), t2 = cadd(v[2], v[3]);
        float2 d1 = csub(v[1], v[4]), d2 = csub(v[2], v[3]);
        float2 a1 = make_float2(v[0].x + c1 * t1.x + c2 * t2.x, v[0].y + c1 * t1.y + c2 * t2.y);
        float2 a2 = make_float2(v[0].x + c2 * t1.x + c1 * t2.x, v[0].y + c2 * t1.y + c1 * t2.y);
        float2 b1 = mul_si<DIR>(make_float2(s1 * d1.x + s2 * d2.x, s1 * d1.y + s2 * d2.y));
        float2 b2 = mul_si<DIR>(make_float2(s2 * d1.x - s1 * d2.x, s2 * d1.y - s1 * d2.y));
        v[0] = cadd(v[0], cadd(t1, t2));
        v[1] = cadd(a1, b1);
        v[4] = csub(a1, b1);
        v[2] = cadd(a2, b2);
        v[3] = csub(a2, b2);
    }
    template <int DIR> __device__ __forceinline__ void dft8(float2 *v)
    {
        const float r = 0.70710678118654752440f;
        float2 e[4] = {v[0], v[2], v[4], v[6]};
        float2 o[4] = {v[1], v[3], v[5], v[7]};
        dft4<DIR>(e);
        dft4<DIR>(o);
        // W8^q = exp(DIR * i pi q / 4)
        float2 w1 = cmul(o[1], make_float2(r, DIR * r));
        float2 w2 = mul_si<DIR>(o[2]);
        float2 w3 = cmul(o[3], make_float2(-r, DIR * r));
        v[0] = cadd(e[0], o[0]);
        v[4] = csub(e[0], o[0]);
        v[1] = cadd(e[1], w1);
        v[5] = csub(e[1], w1);
        v[2] = cadd(e[2], w2);
        v[6] = csub(e[2], w2);
        v[3] = cadd(e[3], w3);
        v[7] = csub(e[3], w3);
    }

// FFT buffers in shared memory are padded by one element per 16 (conflict-free strided Stockham writes).
#define PD(i) ((i) + ((i) >> 4))
    __host__ __device__ __forceinline__ int pd_size(int n) { return n + (n >> 4) + 1; }

    // 9-point DFT as 3 x 3: inner DFT3 over n1 (n = 3 n1 + n2), twiddles W9^(n2 k1), outer DFT3 over n2
    template <int DIR> __device__ __forceinline__ void dft9(float2 *v)
    {
        float2 a[3][3];
#pragma unroll
        for (int n2 = 0; n2 < 3; ++n2) {
            float2 t[3] = {v[n2], v[n2 + 3], v[n2 + 6]};
            dft3<DIR>(t);
            a[n2][0] = t[0];
            a[n2][1] = t[1];
            a[n2][2] = t[2];
        }
        const float c1 = 0.76604444311897803520f, s1 = 0.64278760968653932632f;  // 2 pi / 9
        const float c2 = 0.17364817766693034885f, s2 = 0.98480775301220805936f;  // 4 pi / 9
        const float c4 = -0.93969262078590838405f, s4 = 0.34202014332566873304f; // 8 pi / 9
        a[1][1] = cmul(a[1][1], make_float2(c1, DIR * s1));
        a[1][2] = cmul(a[1][2], make_float2(c2, DIR * s2));
        a[2][1] = cmul(a[2][1], make_float2(c2, DIR * s2));
        a[2][2] = cmul(a[2][2], make_float2(c4, DIR * s4));
#pragma unroll
        for (int k1 = 0; k1 < 3; ++k1) {
            float2 t[3] = {a[0][k1], a[1][k1], a[2][k1]};
            dft3<DIR>(t);
            v[k1] = t[0];
            v[k1 + 3] = t[1];
            v[k1 + 6] = t[2];
        }
    }
    // 16-point DFT as 4 x 4: inner DFT4 over n1 (n = 4 n1 + n2), twiddles W16^(n2 k1), outer DFT4 over n2
    template <int DIR> __device__ __forceinline__ void dft16(float2 *v)
    {
        float2 a[4][4];
#pragma unroll
        for (int n2 = 0; n2 < 4; ++n2) {
            float2 t[4] = {v[n2], v[n2 + 4], v[n2 + 8], v[n2 + 12]};
            dft4<DIR>(t);
            a[n2][0] = t[0];
            a[n2][1] = t[1];
            a[n2][2] = t[2];
            a[n2][3] = t[3];
        }
        const float c1 = 0.92387953251128675613f, s1 = 0.38268343236508977173f; // pi / 8
        const float r2 = 0.70710678118654752440f;
#pragma unroll
        for (int n2 = 1; n2 < 4; ++n2)
#pragma unroll
            for (int k1 = 1; k1 < 4; ++k1) {
                const int e = n2 * k1; // W16^e, e in [1, 9]
                float cr, ci;
                if (e == 1) {
                    cr = c1;
                    ci = s1;
                } else if (e == 2) {
                    cr = r2;
                    ci = r2;
                } else if (e == 3) {
                    cr = s1;
                    ci = c1;
                } else if (e == 4) {
                    cr = 0.f;
                    ci = 1.f;
                } else if (e == 6) {
                    cr = -r2;
                    ci = r2;
                } else {
                    cr = -c1;
                    ci = -s1;
                } // e == 9: cos(9 pi / 8), sin(9 pi / 8)
                a[n2][k1] = cmul(a[n2][k1], make_float2(cr, DIR * ci));
            }
#pragma unroll
        for (int k1 = 0; k1 < 4; ++k1) {
            float2 t[4] = {a[0][k1], a[1][k1], a[2][k1], a[3][k1]};
            dft4<DIR>(t);
            v[k1] = t[0];
            v[k1 + 4] = t[1];
            v[k1 + 8] = t[2];
            v[k1 + 12] = t[3];
        }
    }

    // One Stockham pass of radix R over nb transforms of length M (in -> out), Ns = product of earlier radices.
    // tw[m] = exp(-2 pi i m / M). Element idx = b * M/R + j is walked incrementally (no per-element division);
    // k = j mod Ns is unchanged when j wraps since Ns divides M/R.
    // q = floor(x / d) for 0 <= x < 2^21 via the float reciprocal (exact: the fraction is >= 0.5 / d from an integer)
    __device__ __forceinline__ int fdiv(int x, float rd) { return (int)(((float)x + 0.5f) * rd); }

    template <int R, int DIR>
    __device__ __forceinline__ void pass_r(const float2 *__restrict__ in, float2 *__restrict__ out, int nb, int M,
                                           int Ns, float rMR, float rNs, const float2 *__restrict__ tw, int tid, int nt)
    {
        const int MR = M / R, step = M / (Ns * R);
        const int total = nb * MR;
        int b = fdiv(tid, rMR), j = tid - b * MR, k = j - Ns * fdiv(j, rNs);
        const int db = fdiv(nt, rMR), dj = nt - db * MR, dk = dj - Ns * fdiv(dj, rNs);
        for (int idx = tid; idx < total; idx += nt) {
            const int sb = b * M + j;
            float2 v[R];
#pragma unroll
            for (int r = 0; r < R; ++r) v[r] = in[PD(sb + r * MR)];
            if (k != 0) {
                const int ks = k * step;
#pragma unroll
                for (int r = 1; r < R; ++r) {
                    float2 w = tw[ks * r];
                    if (DIR > 0) w.y = -w.y;
                    v[r] = cmul(v[r], w);
                }
            }
            if (R == 2)
                dft2<DIR>(v);
            else if (R == 3)
                dft3<DIR>(v);
            else if (R == 4)
                dft4<DIR>(v);
            else if (R == 5)
                dft5<DIR>(v);
            else if (R == 8)
                dft8<DIR>(v);
            else if (R == 9)
                dft9<DIR>(v);
            else if (R == 16)
                dft16<DIR>(v);
            const int db0 = b * M + (j - k) * R + k;
#pragma unroll
            for (int r = 0; r < R; ++r) out[PD(db0 + r * Ns)] = v[r];
            b += db;
            j += dj;
            k += dk;
            if (k >= Ns) k -= Ns;
            if (j >= MR) {
                j -= MR;
                ++b;
            }
        }
    }

    // Generic odd radix (prime factors 7..FFT_MAXGEN): O(R^2) DFT with tabulated twiddles of the full length.
    template <int DIR>
    __device__ __noinline__ void pass_gen(const float2 *__restrict__ in, float2 *__restrict__ out, int nb, int M, int R,
                                          int Ns, const float2 *__restrict__ tw, int tid, int nt)
    {
        const int MR = M / R, step = M / (Ns * R);
        const int total = nb * MR;
        for (int idx = tid; idx < total; idx += nt) {
            const int b = idx / MR, j = idx - b * MR;
            const int k = j % Ns;
            const int sb = b * M + j;
            float2 v[FFT_MAXGEN];
            for (int r = 0; r < R; ++r) {
                float2 w = tw[k * step * r];
                if (DIR > 0) w.y = -w.y;
                v[r] = cmul(in[PD(sb + r * MR)], w);
            }
            const int db0 = b * M + (j - k) * R + k;
            for (int q = 0; q < R; ++q) {
                float2 s = make_float2(0.f, 0.f);
                for (int r = 0; r < R; ++r) {
                    float2 w = tw[((r * q) % R) * MR]; // exp(-2 pi i r q / R)
                    if (DIR > 0) w.y = -w.y;
                    s = cadd(s, cmul(v[r], w));
                }
                out[PD(db0 + q * Ns)] = s;
            }
        }
    }

    // All passes of a plan (radices rad[0..np) in shared memory); buffers a (input) and b ping-pong; returns the buffer
    // holding the result. Callers synchronise before (input written); every pass ends with a barrier.
    template <int DIR>
    __device__ float2 *run_fft(float2 *a, float2 *b, int nb, int M, int np, const SPlan &pl,
                               const float2 *__restrict__ tw, int tid, int nt)
    {
        for (int p = 0; p < np; ++p) {
            const int R = pl.R[p], Ns = pl.Ns[p];
            const float rMR = pl.rMR[p], rNs = pl.rNs[p];
            switch (R) {
            case 2: pass_r<2, DIR>(a, b, nb, M, Ns, rMR, rNs, tw, tid, nt); break;
            case 3: pass_r<3, DIR>(a, b, nb, M, Ns, rMR, rNs, tw, tid, nt); break;
            case 4: pass_r<4, DIR>(a, b, nb, M, Ns, rMR, rNs, tw, tid, nt); break;
            case 5: pass_r<5, DIR>(a, b, nb, M, Ns, rMR, rNs, tw, tid, nt); break;
            case 8: pass_r<8, DIR>(a, b, nb, M, Ns, rMR, rNs, tw, tid, nt); break;
            case 9: pass_r<9, DIR>(a, b, nb, M, Ns, rMR, rNs, tw, tid, nt); break;
            case 16: pass_r<16, DIR>(a, b, nb, M, Ns, rMR, rNs, tw, tid, nt); break;
            default: pass_gen<DIR>(a, b, nb, M, R, Ns, tw, tid, nt); break;
            }
            __syncthreads();
            float2 *t = a;
            a = b;
            b = t;
        }
        return a;
    }

    // exp(sgn * 2 pi i m / n) for m in [0, cnt)
    __device__ __forceinline__ void fill_tw(float2 *t, int cnt, int n, float sgn, int tid, int nt)
    {
        for (int m = tid; m < cnt; m += nt) {
            float s, c;
            sincospif(2.0f * (float)m / (float)n, &s, &c);
            t[m] = make_float2(c, sgn * s);
        }
    }

    // Forward real-FFT post-processing: Z = FFT_M(x[2n] + i x[2n+1]) -> X[k], k in [0, M], rf[k] = exp(-2 pi i k / 2M).
    __device__ __forceinline__ float2 rfft_post(const float2 *__restrict__ Z, int base, int M, int k,
                                                const float2 *__restrict__ rf)
    {
        const float2 zk = Z[PD(base + (k == M ? 0 : k))];
        const float2 zm = cconj(Z[PD(base + (k == 0 ? 0 : M - k))]);
        const float2 E = make_float2(0.5f * (zk.x + zm.x), 0.5f * (zk.y + zm.y));
        const float2 d = csub(zk, zm);
        const float2 O = make_float2(0.5f * d.y, -0.5f * d.x); // -i/2 * d
        return cadd(E, cmul(rf[k], O));
    }

// ------------------------------------------------------------------------------------------------------------
// selection and bookkeeping
// ------------------------------------------------------------------------------------------------------------
#define FFT_NBIN 64 // ratio thresholds theta0 * 1.25^t, t in [0, FFT_NBIN)

    // direct-path cost of every ring (one warp per row, lanes over its arcs)
    __global__ void k_fx_cost(FCfg c, const int64_t *__restrict__ seg_off, const int *__restrict__ seg,
                              float *__restrict__ dcost, int *__restrict__ acnt)
    {
        fx_pdl_trigger();
        const int r = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
        if (r >= c.NROWS) return;
        const long long sb = seg_off[r], se = seg_off[r + 1];
        for (long long s = sb + lane; s < se; s += 32) {
            const int ring = seg[3 * s], len = seg[3 * s + 2];
            if (len <= 0 || ring < 0 || ring >= c.Ho) continue;
            // direct path: per output 8-column tile, ceil(taps / 16) + 1 k-steps of 16 x 2 psi terms
            const int taps = (min(len, c.Wo) + c.P - 1) / c.P;
            atomicAdd(&dcost[ring], c.dunit * (float)c.Wo * (float)(((taps + 15) >> 4) + 1));
            atomicAdd(&acnt[ring], 1);
        }
    }

    __device__ __forceinline__ float ring_ratio(const FCfg &c, const float *__restrict__ dcost,
                                                const int *__restrict__ acnt, int h)
    {
        const int n = acnt[h];
        if (n <= 0) return 0.f;
        return dcost[h] / ((float)n * c.fcost_arc + c.fcost_ring);
    }
    __device__ __forceinline__ float fx_theta(const FCfg &c, int t) { return c.theta0 * powf(1.25f, (float)t); }
    // largest t with ratio >= theta(t), -1 if none (same comparisons as the final flagging)
    __device__ __forceinline__ int fx_bin(const FCfg &c, float x)
    {
        int lo = -1, hi = FFT_NBIN - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (x >= fx_theta(c, mid))
                lo = mid;
            else
                hi = mid - 1;
        }
        return lo;
    }

    // per row: largest ratio over the rings it reaches (the row is needed iff that ring is flagged); histograms of
    // rows, rings and arcs over the threshold bins
    __global__ void k_fx_rowmax(FCfg c, const int64_t *__restrict__ seg_off, const int *__restrict__ seg,
                                const float *__restrict__ dcost, const int *__restrict__ acnt,
                                float *__restrict__ rowmax, int *__restrict__ hist)
    {
        fx_pdl_wait();
        fx_pdl_trigger();
        const int gt = blockIdx.x * blockDim.x + threadIdx.x;
        if (gt < c.Ho) {
            const int t = fx_bin(c, ring_ratio(c, dcost, acnt, gt));
            if (t >= 0) {
                atomicAdd(&hist[FFT_NBIN + t], 1);
                atomicAdd(&hist[2 * FFT_NBIN + t], acnt[gt]);
            }
        }
        const int r = gt >> 5, lane = threadIdx.x & 31;
        if (r >= c.NROWS) return;
        const long long sb = seg_off[r], se = seg_off[r + 1];
        float m = 0.f;
        for (long long s = sb + lane; s < se; s += 32) {
            const int ring = seg[3 * s], len = seg[3 * s + 2];
            if (len <= 0 || ring < 0 || ring >= c.Ho) continue;
            m = fmaxf(m, ring_ratio(c, dcost, acnt, ring));
        }
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (lane == 0) {
            rowmax[r] = m;
            const int t = fx_bin(c, m);
            if (t >= 0) atomicAdd(&hist[t], 1);
        }
    }

    // Smallest threshold bin whose flagged arcs, rings and needed input rows fit the buffers; flags the rings and
    // lists them in ascending order (groups of FFT_NSLOT). cnt[3] = chosen bin (FFT_NBIN: nothing flagged).
    __global__ void __launch_bounds__(1024)
        k_fx_select(FCfg c, const float *__restrict__ dcost, const int *__restrict__ acnt, const int *__restrict__ hist,
                    unsigned char *__restrict__ flag, int *__restrict__ fidx, int *__restrict__ rlist,
                    int *__restrict__ cnt)
    {
        fx_pdl_wait();
        fx_pdl_trigger();
        __shared__ int s_t, s_w[32], s_carry;
        const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
        if (tid == 0) {
            long long rows = 0, rings = 0, arcs = 0;
            int t = FFT_NBIN;
            for (int b = FFT_NBIN - 1; b >= 0; --b) {
                rows += hist[b];
                rings += hist[FFT_NBIN + b];
                arcs += hist[2 * FFT_NBIN + b];
                if (rows > c.cap_rows || rings > (long long)c.nrg_max * FFT_NSLOT || arcs > c.cap_pairs) break;
                t = b;
            }
            s_t = t;
            s_carry = 0;
        }
        __syncthreads();
        const float th = s_t < FFT_NBIN ? fx_theta(c, s_t) : 3.0e38f;
        for (int base = 0; base < c.Ho; base += 1024) {
            const int h = base + tid;
            const bool f = h < c.Ho && ring_ratio(c, dcost, acnt, h) >= th;
            const unsigned bal = __ballot_sync(0xffffffffu, f);
            if (lane == 0) s_w[wid] = __popc(bal);
            __syncthreads();
            int pre = 0;
            for (int w = 0; w < wid; ++w) pre += s_w[w];
            const int pos = s_carry + pre + __popc(bal & ((1u << lane) - 1u));
            if (h < c.Ho) {
                flag[h] = f ? 1 : 0;
                fidx[h] = f ? pos : -1;
                if (f) rlist[pos] = h;
            }
            __syncthreads();
            if (tid == 0) {
                int tot = 0;
                for (int w = 0; w < 32; ++w) tot += s_w[w];
                s_carry += tot;
            }
            __syncthreads();
        }
        if (tid == 0) {
            const int nf = s_carry;
            const int ng = (nf + FFT_NSLOT - 1) / FFT_NSLOT;
            cnt[0] = ng; // ring groups
            cnt[3] = s_t;
            cnt[4] = nf; // flagged rings
            for (int i = nf; i < ng * FFT_NSLOT; ++i) rlist[i] = -1;
        }
    }

    // Walk the flagged pairs of row r: emit(pair arcs [sb, se), voff, ring)
    template <class F>
    __device__ __forceinline__ void walk_pairs(int r, const FCfg &c, const int64_t *__restrict__ seg_off,
                                               const int *__restrict__ seg, const int64_t *__restrict__ val_off,
                                               const unsigned char *__restrict__ flag, F emit)
    {
        const long long sb = seg_off[r], se = seg_off[r + 1];
        long long voff = val_off[r];
        long long ps = -1, pv = 0;
        int pring = -1;
        for (long long s = sb; s < se; ++s) {
            const int ring = seg[3 * s], len = seg[3 * s + 2];
            const long long myv = voff;
            if (len > 0) voff += len;
            const bool f = len > 0 && ring >= 0 && ring < c.Ho && flag[ring];
            if (ps >= 0 && (!f || ring != pring)) {
                emit((int)ps, (int)s, pv, pring);
                ps = -1;
            }
            if (f && ps < 0) {
                ps = s;
                pv = myv;
                pring = ring;
            }
        }
        if (ps >= 0) emit((int)ps, (int)se, pv, pring);
    }

    // Visits of row r: consecutive pairs of one ring group, a new visit whenever a slot would repeat.
    template <class FV, class FP>
    __device__ __forceinline__ void walk_visits(int r, const FCfg &c, const int64_t *__restrict__ seg_off,
                                                const int *__restrict__ seg, const int64_t *__restrict__ val_off,
                                                const unsigned char *__restrict__ flag, const int *__restrict__ fidx,
                                                FV new_visit, FP add_pair)
    {
        int lastg = -1;
        unsigned occ = 0;
        walk_pairs(r, c, seg_off, seg, val_off, flag, [&](int sb, int se, long long voff, int ring) {
            const int fi = fidx[ring];
            const int g = fi / FFT_NSLOT, slot = fi - g * FFT_NSLOT;
            if (g != lastg || ((occ >> slot) & 1u)) {
                new_visit(g);
                lastg = g;
                occ = 0;
            }
            occ |= 1u << slot;
            add_pair(sb, se, voff, slot);
        });
    }

    // needed rows get a spectrum slot; visits are counted per ring group
    __global__ void k_fx_count(FCfg c, const int64_t *__restrict__ seg_off, const int *__restrict__ seg,
                               const int64_t *__restrict__ val_off, const int *__restrict__ row_ker,
                               const int *__restrict__ row_lat, const float *__restrict__ rowmax,
                               const unsigned char *__restrict__ flag, const int *__restrict__ fidx,
                               int *__restrict__ cnt, int *__restrict__ rslot, int *__restrict__ rxrow,
                               int *__restrict__ vcount)
    {
        fx_pdl_wait();
        fx_pdl_trigger();
        const int r = blockIdx.x * blockDim.x + threadIdx.x;
        if (r >= c.NROWS) return;
        const int t = cnt[3];
        if (t >= FFT_NBIN || !(rowmax[r] >= fx_theta(c, t))) {
            rslot[r] = -1;
            return;
        }
        const int sl = atomicAdd(&cnt[2], 1);
        rslot[r] = sl;
        rxrow[sl] = row_ker[r] * c.Hi + row_lat[r];
        walk_visits(
            r, c, seg_off, seg, val_off, flag, fidx, [&](int g) { atomicAdd(&vcount[g], 1); },
            [&](int, int, long long, int) {});
    }

    __global__ void __launch_bounds__(1024)
        k_fx_scan(const int *__restrict__ cnt, const int *__restrict__ vcount, int *__restrict__ vstart)
    {
        fx_pdl_wait();
        fx_pdl_trigger();
        __shared__ int s_w[32], s_carry;
        const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
        const int n = cnt[0];
        if (tid == 0) s_carry = 0;
        __syncthreads();
        for (int base = 0; base < n; base += 1024) {
            const int i = base + tid;
            const int v = i < n ? vcount[i] : 0;
            int x = v;
            for (int o = 1; o < 32; o <<= 1) {
                const int y = __shfl_up_sync(0xffffffffu, x, o);
                if (lane >= o) x += y;
            }
            if (lane == 31) s_w[wid] = x;
            __syncthreads();
            int pre = 0;
            for (int w = 0; w < wid; ++w) pre += s_w[w];
            if (i < n) vstart[i] = s_carry + pre + x - v;
            __syncthreads();
            if (tid == 0) {
                int tot = 0;
                for (int w = 0; w < 32; ++w) tot += s_w[w];
                s_carry += tot;
            }
            __syncthreads();
        }
        if (tid == 0) vstart[n] = s_carry;
    }

    __global__ void k_fx_place(FCfg c, const int64_t *__restrict__ seg_off, const int *__restrict__ seg,
                               const int64_t *__restrict__ val_off, const unsigned char *__restrict__ flag,
                               const int *__restrict__ fidx, const int *__restrict__ rslot,
                               const int *__restrict__ vstart, int *__restrict__ vcursor, int *__restrict__ npairs,
                               Pair *__restrict__ pairs, Visit *__restrict__ visits)
    {
        fx_pdl_wait();
        fx_pdl_trigger();
        const int r = blockIdx.x * blockDim.x + threadIdx.x;
        if (r >= c.NROWS) return;
        const int slot_r = rslot[r];
        if (slot_r < 0) return;
        Visit v;
        int vg = -1;
        auto flush = [&]() {
            if (vg >= 0) visits[vstart[vg] + atomicAdd(&vcursor[vg], 1)] = v;
        };
        walk_visits(
            r, c, seg_off, seg, val_off, flag, fidx,
            [&](int g) {
                flush();
                vg = g;
                v.xrow = slot_r;
                v.pad0 = v.pad1 = v.pad2 = 0;
#pragma unroll
                for (int k = 0; k < FFT_NSLOT; ++k) v.pidx[k] = -1;
            },
            [&](int sb, int se, long long voff, int slot) {
                const int pi = atomicAdd(npairs, 1);
                Pair p;
                p.sb = sb;
                p.se = se;
                p.voff = voff;
                pairs[pi] = p;
                v.pidx[slot] = pi;
            });
        flush();
    }

    // PSI spectra: FFT_PSI_NB pairs per batch; PSI[p][K] = sum_n psi_p[n] exp(-2 pi i K n / Wo), K in [0, Wo/2].
    __global__ void __launch_bounds__(256)
        k_fx_psi(FCfg c, const int *__restrict__ npairs, const Pair *__restrict__ pairs, const int *__restrict__ seg,
                 const float *__restrict__ vals, float2 *__restrict__ psi)
    {
        fx_pdl_wait();
        fx_pdl_trigger();
        extern __shared__ __align__(16) unsigned char fx_smem[];
        __shared__ SPlan s_pl;
        const int tid = threadIdx.x, nt = blockDim.x;
        const int M = c.Mo;
        float2 *tw = reinterpret_cast<float2 *>(fx_smem);
        float2 *rf = tw + M;
        float2 *A = rf + c.NKo;
        float2 *B = A + pd_size(FFT_PSI_NB * M);
        const int np = *npairs;
        if (blockIdx.x * FFT_PSI_NB >= np) return;
        load_plan(s_pl, c.po, tid);
        fill_tw(tw, M, M, -1.f, tid, nt);
        fill_tw(rf, c.NKo, c.Wo, -1.f, tid, nt);
        float *Af = reinterpret_cast<float *>(A);
        for (int p0 = blockIdx.x * FFT_PSI_NB; p0 < np; p0 += gridDim.x * FFT_PSI_NB) {
            const int nb = min(FFT_PSI_NB, np - p0);
            for (int i = tid; i < 2 * pd_size(nb * M); i += nt) Af[i] = 0.f;
            __syncthreads();
            // one warp per pair: the lanes load up to 32 arcs at once, a scan gives their value offsets
            const int lane = tid & 31;
            for (int b = tid >> 5; b < nb; b += nt >> 5) {
                const Pair pr = pairs[p0 + b];
                long long voff = pr.voff;
                for (int s0 = pr.sb; s0 < pr.se; s0 += 32) {
                    const int s = s0 + lane;
                    int st = 0, len = 0;
                    if (s < pr.se) {
                        st = seg[3 * s + 1];
                        len = max(seg[3 * s + 2], 0);
                    }
                    int x = len;
                    for (int o = 1; o < 32; o <<= 1) {
                        const int y = __shfl_up_sync(0xffffffffu, x, o);
                        if (lane >= o) x += y;
                    }
                    const long long myoff = voff + (x - len);
                    const int na = min(32, pr.se - s0);
                    for (int a = 0; a < na; ++a) {
                        const int alen = __shfl_sync(0xffffffffu, len, a);
                        const long long aoff = __shfl_sync(0xffffffffu, myoff, a);
                        int st0 = __shfl_sync(0xffffffffu, st, a) % c.Wo;
                        if (st0 < 0) st0 += c.Wo;
                        for (int j0 = lane; j0 < alen; j0 += 128) {
                            float vv[4];
#pragma unroll
                            for (int q = 0; q < 4; ++q)
                                vv[q] = (j0 + 32 * q < alen) ? __ldg(vals + aoff + j0 + 32 * q) : 0.f;
#pragma unroll
                            for (int q = 0; q < 4; ++q) {
                                const int j = j0 + 32 * q;
                                if (j >= alen) break;
                                int w = st0 + (j < c.Wo ? j : j % c.Wo);
                                if (w >= c.Wo) w -= c.Wo;
                                atomicAdd(&Af[2 * PD(b * M + (w >> 1)) + (w & 1)], vv[q]);
                            }
                        }
                    }
                    voff += __shfl_sync(0xffffffffu, x, 31);
                }
            }
            __syncthreads();
            const float2 *Z = run_fft<-1>(A, B, nb, M, c.po.np, s_pl, tw, tid, nt);
            for (int b = 0; b < nb; ++b) {
                for (int k = tid; k < c.NKo; k += nt)
                    psi[(long long)(p0 + b) * c.NKop + k] = rfft_post(Z, b * M, M, k, rf);
                if (tid == 0 && c.NKop > c.NKo) psi[(long long)(p0 + b) * c.NKop + c.NKo] = make_float2(0.f, 0.f);
            }
            __syncthreads();
        }
    }

#define FFT_FWD_PV 4 // 16-byte input loads per thread and batch (prefetch registers)

    // Spectra of the selected input rows: transform t = slot * BC + channel, X[t][k], k in [0, Wi/2].
    __global__ void __launch_bounds__(256, 2) k_fx_fwd(FCfg c, const int *__restrict__ cnt, const int *__restrict__ rxrow,
                                                       const __half *__restrict__ inp, float2 *__restrict__ spec)
    {
        fx_pdl_wait();
        extern __shared__ __align__(16) unsigned char fx_smem[];
        __shared__ SPlan s_pl;
        __shared__ long long s_src[2][32];
        const int tid = threadIdx.x, nt = blockDim.x;
        const int M = c.Mi, NB = c.nbf;
        const int nsel = cnt[2];
        const long long ntr = (long long)nsel * c.BC;
        if ((long long)blockIdx.x * NB >= ntr) return;
        float2 *tw = reinterpret_cast<float2 *>(fx_smem);
        float2 *rf = tw + M;
        float2 *A = rf + c.NKi;
        float2 *B = A + pd_size(NB * M);
        load_plan(s_pl, c.pi, tid);
        fill_tw(tw, M, M, -1.f, tid, nt);
        fill_tw(rf, c.NKi, c.Wi, -1.f, tid, nt);
        const long long cstride = (long long)c.K * c.Hi * c.Wi;
        const int b_0 = tid / M, n_0 = tid - b_0 * M, db = nt / M, dn = nt - db * M;
        // rows start on 16-byte boundaries and a batch fits the prefetch registers
        const int M4 = M / 4;
        const bool vec = (c.Wi % 8) == 0 && NB * M4 <= FFT_FWD_PV * nt;
        // 16-byte loads (4 complex z[n] = x[2n] + i x[2n+1] each), prefetched one batch ahead in registers
        const float rM4 = 1.0f / (float)max(M4, 1);
        uint4 pre[FFT_FWD_PV];
        auto src_of = [&](long long t) { // t = slot * BC + channel
            const long long sl = t / c.BC;
            return (t - sl * c.BC) * cstride + (long long)rxrow[sl] * c.Wi;
        };
        auto prefetch = [&](long long t0n, int sbuf) {
            const int nbn = (int)min((long long)NB, ntr - t0n);
            if (tid < nbn) s_src[sbuf][tid] = src_of(t0n + tid);
            __syncthreads();
#pragma unroll
            for (int e = 0; e < FFT_FWD_PV; ++e) {
                const int i = tid + nt * e;
                if (i < nbn * M4) {
                    const int b = fdiv(i, rM4), q = i - b * M4;
                    pre[e] = *reinterpret_cast<const uint4 *>(inp + s_src[sbuf][b] + 8 * q);
                }
            }
        };
        int pbuf = 0;
        if (vec) prefetch((long long)blockIdx.x * NB, pbuf);
        for (long long t0 = (long long)blockIdx.x * NB; t0 < ntr; t0 += (long long)gridDim.x * NB) {
            const int nb = (int)min((long long)NB, ntr - t0);
            if (!vec) {
                if (tid < nb) s_src[0][tid] = src_of(t0 + tid);
                __syncthreads();
            }
            if (vec) {
#pragma unroll
                for (int e = 0; e < FFT_FWD_PV; ++e) {
                    const int i = tid + nt * e;
                    if (i < nb * M4) {
                        const int b = fdiv(i, rM4), q = i - b * M4;
                        const int i0 = b * M + 4 * q;
                        A[PD(i0)] = __half22float2(*reinterpret_cast<const __half2 *>(&pre[e].x));
                        A[PD(i0 + 1)] = __half22float2(*reinterpret_cast<const __half2 *>(&pre[e].y));
                        A[PD(i0 + 2)] = __half22float2(*reinterpret_cast<const __half2 *>(&pre[e].z));
                        A[PD(i0 + 3)] = __half22float2(*reinterpret_cast<const __half2 *>(&pre[e].w));
                    }
                }
                const long long t0n = t0 + (long long)gridDim.x * NB;
                if (t0n < ntr)
                    prefetch(t0n, pbuf ^ 1);
                else
                    __syncthreads();
            } else {
                for (int b = b_0, n = n_0; b < nb;) {
                    A[PD(b * M + n)] = __half22float2(*reinterpret_cast<const __half2 *>(inp + s_src[0][b] + 2 * n));
                    b += db;
                    n += dn;
                    if (n >= M) {
                        n -= M;
                        ++b;
                    }
                }
                __syncthreads();
            }
            const float2 *Z = run_fft<-1>(A, B, nb, M, c.pi.np, s_pl, tw, tid, nt);
            for (int b = 0; b < nb; ++b) {
                for (int k = tid; k < c.NKi; k += nt) spec[(t0 + b) * c.NKip + k] = rfft_post(Z, b * M, M, k, rf);
                if (tid == 0 && c.NKip > c.NKi) spec[(t0 + b) * c.NKip + c.NKi] = make_float2(0.f, 0.f);
            }
            if (vec) pbuf ^= 1;
            __syncthreads();
        }
    }

// ------------------------------------------------------------------------------------------------------------
// contraction: work item (ring group, channel tile), ring group fastest so that concurrently processed work items
// share their channel tile's row spectra in L2; persistent CTAs loop over the work items. A producer warp streams,
// per visit, the PSI rows of the visit's pairs and the tile's row spectra into a stage (bulk copies, full/empty
// mbarriers); consumer threads own KPT output frequencies x FFT_NSLOT rings x CB channels of accumulators and
// write the output spectra OUT[flagged ring][channel][K] (fp32).
// ------------------------------------------------------------------------------------------------------------
#define FFT_MAXST 12

    __device__ __forceinline__ unsigned fx_smem_u32(const void *p)
    {
        return static_cast<unsigned>(__cvta_generic_to_shared(p));
    }
    __device__ __forceinline__ void fx_mbar_init(uint64_t *bar, unsigned count)
    {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(fx_smem_u32(bar)), "r"(count));
    }
    __device__ __forceinline__ void fx_mbar_expect_tx(uint64_t *bar, unsigned bytes)
    {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(fx_smem_u32(bar)), "r"(bytes)
                     : "memory");
    }
    __device__ __forceinline__ void fx_mbar_arrive(uint64_t *bar)
    {
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(fx_smem_u32(bar)) : "memory");
    }
    __device__ __forceinline__ void fx_mbar_wait(uint64_t *bar, unsigned parity)
    {
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "FX_WAIT_%=:\n"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                     "@!p bra FX_WAIT_%=;\n"
                     "}\n" ::"r"(fx_smem_u32(bar)),
                     "r"(parity)
                     : "memory");
    }
    __device__ __forceinline__ void fx_bulk_g2s(void *dst, const void *src, unsigned bytes, uint64_t *bar)
    {
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
                         fx_smem_u32(dst)),
                     "l"(src), "r"(bytes), "r"(fx_smem_u32(bar))
                     : "memory");
    }

    // Stage (float2 units): the PSI chunks of the FFT_NSLOT ring slots (F bins each), then the row-spectrum chunks of
    // up to CBM channels (F + 2 bins each, starting at an even bin).
    __host__ __device__ __forceinline__ int fx_stage_f2(int F, int CBM) { return FFT_NSLOT * F + CBM * (F + 2); }

    // Input bin of output bin K (the real-DFT spectrum folded onto [0, Wi/2]) and whether it is conjugated.
    __host__ __device__ __forceinline__ int fx_kin(int K, int Wi, int Mi, bool &cj)
    {
        int ki = K % Wi;
        cj = ki > Mi;
        return cj ? Wi - ki : ki;
    }
    // Smallest input bin used by output bins [K0, K1) (1-Lipschitz triangle wave: an endpoint or a zero inside),
    // rounded down to an even bin (16-byte aligned bulk copies).
    __device__ __forceinline__ int fx_xlo(const FCfg &c, int K0, int K1)
    {
        bool cj;
        int lo = min(fx_kin(K0, c.Wi, c.Mi, cj), fx_kin(K1 - 1, c.Wi, c.Mi, cj));
        if ((K1 - 1) / c.Wi != K0 / c.Wi) lo = 0;
        return lo & ~1;
    }

    __device__ __forceinline__ void fx_cp16(void *dst, const void *src)
    {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(fx_smem_u32(dst)), "l"(src) : "memory");
    }
    __device__ __forceinline__ void fx_cp16u(unsigned dst, const void *src)
    {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
    }
    __device__ __forceinline__ void fx_cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
    __device__ __forceinline__ void fx_cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

    // Work item (ring group g, channel tile, frequency chunk of F output bins); persistent CTAs take the work items
    // with ring groups by decreasing visit count, snaking (b, 2N-1-b, 2N+b, ...) so that CTAs get similar visit counts.
    // The visits of a work item are staged FFT_CV at a time (double-buffered, 16-byte cp.async by all threads): per
    // visit the PSI chunks of its ring slots and the tile's row-spectrum chunks. Thread (k, cg) accumulates output
    // bin K0 + k of channels cg*CPT .. +CPT for all FFT_NSLOT rings of the group in registers.
    template <int CBM, int CPT>
    __global__ void __launch_bounds__(FFT_CON_MAXT, 2)
        k_fx_contract(FCfg c, const int *__restrict__ cnt, const int *__restrict__ vstart,
                      const Visit *__restrict__ visits, const float2 *__restrict__ psi, const float2 *__restrict__ spec,
                      float2 *__restrict__ ospec)
    {
        fx_pdl_trigger();
        extern __shared__ __align__(16) unsigned char fx_smem[];
        __shared__ int s_gord[FFT_NRG_CAP];
        __shared__ __align__(16) Visit s_rec[2][FFT_CV];
        constexpr int NCG = CBM / CPT;
        const int tid = threadIdx.x, NT = blockDim.x;
        const unsigned sbase = fx_smem_u32(fx_smem), s_rec_u = fx_smem_u32(&s_rec[0][0]);
        const int F = c.F, FX = F + 2, F2 = F >> 1, FX2 = F2 + 1;
        const int ng = cnt[0], nsel = cnt[2], nflag = cnt[4];
        const int nfc = c.nfc, per_g = c.ntile * nfc;
        const int nwork = ng * per_g;
        if ((int)blockIdx.x >= nwork) return;
        for (int t = tid; t < ng; t += NT) {
            const int nt_ = vstart[t + 1] - vstart[t];
            int r = 0;
            for (int u = 0; u < ng; ++u) {
                const int nu = vstart[u + 1] - vstart[u];
                r += (nu > nt_) || (nu == nt_ && u < t);
            }
            s_gord[r] = t;
        }
        __syncthreads();
        const int stf = fx_stage_f2(F, CBM); // one visit
        float2 *R0 = reinterpret_cast<float2 *>(fx_smem);
        // this thread's 16-byte pieces of a visit: one PSI piece (slot p_slot, piece p_j), up to 4 spectrum pieces
        const bool p_on = tid < FFT_NSLOT * F2;
        const int p_slot = p_on ? tid / F2 : 0, p_j = tid - (tid / F2) * F2;
        const int p_dst = p_slot * F + 2 * p_j;
        int x_ch[4], x_j[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int idx = tid + u * NT;
            x_ch[u] = idx < CBM * FX2 ? idx / FX2 : CBM;
            x_j[u] = idx - (idx / FX2) * FX2;
        }
        const int N = gridDim.x;
        const int k = tid % F, cg = tid / F;
        const long long NKip = c.NKip;
        const int NKop = c.NKop;
        for (int j = 0;; ++j) {
            const int rnd = j >> 1;
            const int w = (j & 1) ? (2 * rnd + 2) * N - 1 - (int)blockIdx.x : 2 * rnd * N + (int)blockIdx.x;
            if (w >= nwork) break;
            const int g = s_gord[w / per_g], rest = w % per_g;
            const int tile = rest / nfc, fc = rest % nfc;
            const int K0 = fc * F, K1 = min(K0 + F, c.NKo);
            const int xlo = fx_xlo(c, K0, K1);
            const int c0 = tile * c.Cb, cn = min(c.Cb, c.BC - c0);
            const int nP2 = min(F, NKop - K0) >> 1;
            const int nX2 = min(FX, c.NKip - xlo) >> 1;
            const int K = K0 + k;
            const bool kok = K < K1;
            bool cj;
            const int ki = fx_kin(kok ? K : K0, c.Wi, c.Mi, cj);
            const int xo = kok ? ki - xlo : 0;
            const int v0 = vstart[g], nv = vstart[g + 1] - v0;
            const long long xrs = (long long)c.BC * NKip; // row-spectrum stride of a selected row (slot-major)
            // per-item copy sources / destinations (shared byte offsets within one visit's stage)
            const float2 *psrc = psi + K0 + 2 * p_j;
            const bool pon = p_on && p_j < nP2;
            const unsigned pd = (unsigned)p_dst * 8u;
            const float2 *xsrc[4];
            unsigned xd[4];
            bool xon[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                xon[u] = x_ch[u] < cn && x_j[u] < nX2;
                xsrc[u] = spec + (long long)(c0 + (xon[u] ? x_ch[u] : 0)) * NKip + xlo + 2 * x_j[u];
                xd[u] = (unsigned)(FFT_NSLOT * F + x_ch[u] * FX + 2 * x_j[u]) * 8u;
            }
            // visit records of stage i (FFT_CV visits) -> s_rec[i & 1], 16-byte pieces by threads 0 .. 3*FFT_CV-1
            auto issue_rec = [&](int i0, int rb) {
                if (tid < 3 * FFT_CV) {
                    const int v = tid / 3, q = tid - (tid / 3) * 3;
                    if (i0 + v < nv)
                        fx_cp16u(s_rec_u + (unsigned)((rb * FFT_CV + v) * 48 + q * 16),
                                 reinterpret_cast<const int4 *>(visits + v0 + i0 + v) + q);
                }
            };
            auto issue = [&](int i0, int buf) {
                const unsigned sb = sbase + (unsigned)(buf * FFT_CV * stf) * 8u;
#pragma unroll
                for (int v = 0; v < FFT_CV; ++v) {
                    if (i0 + v >= nv) break;
                    const Visit &vr = s_rec[buf][v];
                    const unsigned so = sb + (unsigned)(v * stf) * 8u;
                    if (pon) { // a ring slot the row does not reach contributes zeros
                        const int pid = vr.pidx[p_slot];
                        if (pid >= 0)
                            fx_cp16u(so + pd, psrc + pid * NKop);
                        else
                            asm volatile("st.shared.v4.f32 [%0], {%1, %1, %1, %1};\n" ::"r"(so + pd), "f"(0.f)
                                         : "memory");
                    }
                    const long long xoff = (long long)vr.xrow * xrs;
#pragma unroll
                    for (int u = 0; u < 4; ++u)
                        if (xon[u]) fx_cp16u(so + xd[u], xsrc[u] + xoff);
                }
            };
            float2 acc[FFT_NSLOT][CPT];
#pragma unroll
            for (int s = 0; s < FFT_NSLOT; ++s)
#pragma unroll
                for (int ch = 0; ch < CPT; ++ch) acc[s][ch] = make_float2(0.f, 0.f);
            // prologue: records of stage 0, then data of stage 0 + records of stage 1
            issue_rec(0, 0);
            fx_cp_commit();
            fx_cp_wait_all();
            __syncthreads();
            issue(0, 0);
            issue_rec(FFT_CV, 1);
            fx_cp_commit();
            int buf = 0;
            for (int i0 = 0; i0 < nv; i0 += FFT_CV, buf ^= 1) {
                fx_cp_wait_all();
                __syncthreads(); // data of stage i0 and records of stage i0 + FFT_CV have landed
                if (i0 + FFT_CV < nv) {
                    issue(i0 + FFT_CV, buf ^ 1);
                    issue_rec(i0 + 2 * FFT_CV, buf);
                }
                fx_cp_commit();
                const float2 *sbp = R0 + buf * FFT_CV * stf;
                const int nvv = min(FFT_CV, nv - i0);
                for (int v = 0; v < nvv; ++v) {
                    const float2 *stg = sbp + v * stf;
                    const float2 *sx = stg + FFT_NSLOT * F + cg * CPT * FX + xo;
                    float2 xv[CPT];
#pragma unroll
                    for (int ch = 0; ch < CPT; ++ch) xv[ch] = sx[ch * FX];
                    if (cj) {
#pragma unroll
                        for (int ch = 0; ch < CPT; ++ch) xv[ch].y = -xv[ch].y;
                    }
                    float2 pv[FFT_NSLOT];
#pragma unroll
                    for (int s = 0; s < FFT_NSLOT; ++s) pv[s] = stg[s * F + k];
#pragma unroll
                    for (int s = 0; s < FFT_NSLOT; ++s) {
#pragma unroll
                        for (int ch = 0; ch < CPT; ++ch) {
                            acc[s][ch].x = fmaf(pv[s].x, xv[ch].x, fmaf(-pv[s].y, xv[ch].y, acc[s][ch].x));
                            acc[s][ch].y = fmaf(pv[s].x, xv[ch].y, fmaf(pv[s].y, xv[ch].x, acc[s][ch].y));
                        }
                    }
                }
            }
            fx_cp_wait_all();
            __syncthreads(); // the next item's first stage overwrites buffer 0
            if (kok) {
#pragma unroll
                for (int s = 0; s < FFT_NSLOT; ++s) {
                    const int fi = g * FFT_NSLOT + s;
                    if (fi >= nflag) continue;
#pragma unroll
                    for (int ch = 0; ch < CPT; ++ch) {
                        const int cc = cg * CPT + ch;
                        if (cc < cn) ospec[((long long)fi * c.BC + c0 + cc) * NKop + K] = acc[s][ch];
                    }
                }
            }
        }
    }

    // Inverse real FFTs: transform t = flagged ring * BC + channel, OUT[t][K] -> out[channel][ring][:] (fp16)
    __global__ void __launch_bounds__(256) k_fx_inv(FCfg c, const int *__restrict__ cnt, const int *__restrict__ rlist,
                                                    const float2 *__restrict__ ospec, __half *__restrict__ out)
    {
        fx_pdl_wait();
        extern __shared__ __align__(16) unsigned char fx_smem[];
        __shared__ SPlan s_pl;
        const int tid = threadIdx.x, nt = blockDim.x;
        const int M = c.Mo, NB = c.nbi;
        const long long ntr = (long long)cnt[4] * c.BC;
        if ((long long)blockIdx.x * NB >= ntr) return;
        float2 *tw = reinterpret_cast<float2 *>(fx_smem);
        float2 *rf = tw + M;
        float2 *A = rf + M;
        float2 *B = A + pd_size(NB * M);
        load_plan(s_pl, c.po, tid);
        fill_tw(tw, M, M, -1.f, tid, nt);
        fill_tw(rf, M, c.Wo, 1.f, tid, nt);
        __syncthreads();
        const float inv = 1.0f / (float)c.Wo;
        for (long long t0 = (long long)blockIdx.x * NB; t0 < ntr; t0 += (long long)gridDim.x * NB) {
            const int nb = (int)min((long long)NB, ntr - t0);
            // Z[k] = (X[k] + conj X[M-k]) + i (X[k] - conj X[M-k]) exp(2 pi i k / Wo), k in [0, M)
            {
                const float2 *X0 = ospec + t0 * c.NKop;
                const int tot = nb * M;
                int b = tid / M, k = tid - (tid / M) * M;
                const int db = nt / M, dk = nt - db * M;
#pragma unroll 4
                for (int i = tid; i < tot; i += nt) {
                    const float2 *X = X0 + (long long)b * c.NKop;
                    const float2 xk = X[k];
                    const float2 xm = cconj(X[M - k]);
                    const float2 e = cadd(xk, xm);
                    const float2 o = cmul(csub(xk, xm), rf[k]);
                    A[PD(b * M + k)] = make_float2(e.x - o.y, e.y + o.x);
                    b += db;
                    k += dk;
                    if (k >= M) {
                        k -= M;
                        ++b;
                    }
                }
            }
            __syncthreads();
            const float2 *zt = run_fft<1>(A, B, nb, M, c.po.np, s_pl, tw, tid, nt);
            for (int b = 0; b < nb; ++b) {
                const long long t = t0 + b;
                const int fi = (int)(t / c.BC), ch = (int)(t - (long long)fi * c.BC);
                __half *o = out + ((long long)ch * c.Ho + rlist[fi]) * c.Wo;
                for (int n = tid; n < M; n += nt) {
                    const float2 z = zt[PD(b * M + n)];
                    *reinterpret_cast<__half2 *>(o + 2 * n) = __floats2half2_rn(z.x * inv, z.y * inv);
                }
            }
            __syncthreads();
        }
    }

    // ------------------------------------------------------------------------------------------------------------
    // host
    // ------------------------------------------------------------------------------------------------------------
    static bool make_plan(Plan &p, int M)
    {
        p.M = M;
        p.np = 0;
        if (M < 1) return false;
        int m = M;
        // radices 16 / 8 / 4 / 2 for the power of two, 9, then odd primes
        int e2 = 0;
        while (m % 2 == 0) {
            m /= 2;
            ++e2;
        }
        const int np2 = (e2 + 3) / 4; // fewest passes of at most 16, bits spread evenly
        for (int i = 0; i < np2 && p.np < FFT_MAXPASS; ++i) p.R[p.np++] = 1 << (e2 / np2 + (i < e2 % np2 ? 1 : 0));
        while (m % 9 == 0 && p.np < FFT_MAXPASS) {
            p.R[p.np++] = 9;
            m /= 9;
        }
        for (int q = 3; q <= FFT_MAXGEN && m > 1; q += 2)
            while (m % q == 0 && p.np < FFT_MAXPASS) {
                p.R[p.np++] = q;
                m /= q;
            }
        int ns = 1;
        for (int i = 0; i < p.np; ++i) {
            p.Ns[i] = ns;
            p.rMR[i] = 1.0f / (float)(M / p.R[i]);
            p.rNs[i] = 1.0f / (float)ns;
            ns *= p.R[i];
        }
        return m == 1;
    }

    static inline size_t fx_align(size_t x, size_t a) { return (x + a - 1) / a * a; }

    template <typename... KArgs, typename... Args>
    static void fx_launch_pdl(void (*k)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Args... args)
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

    struct Ctx {
        bool on = false;
        int cbm = 16;
        FCfg c;
        unsigned char *ws = nullptr;
        unsigned char *flag = nullptr;
        int *cnt = nullptr, *vstart = nullptr, *rlist = nullptr;
        Visit *visits = nullptr;
        float2 *psi = nullptr, *spec = nullptr, *ospec = nullptr;
        size_t smem_con = 0, smem_inv = 0;
        int nt = 0;
    };

#define FFT_NSM 132
#define FFT_MIN_CHANNELS 32 // fewer channels do not amortise the per-call psi bookkeeping of this path

    // Selects the rings and builds the pair/visit lists, PSI spectra and row spectra; returns the ring skip flags for
    // the direct path (nullptr when the frequency-domain path does not apply).
    static const unsigned char *Begin(Ctx &x, int BC, int K, int Hi, int Wi, int Ho, int Wo, int NROWS, long long NSEG,
                                      const int64_t *d_so, const int *d_seg, const int64_t *d_vo, const float *d_vals,
                                      const int *d_rk, const int *d_rl, const __half *d_inp, cudaStream_t stream)
    {
        x.on = false;
        if (BC < FFT_MIN_CHANNELS || NROWS <= 0 || NSEG <= 0 || Wi <= 0 || Ho <= 0 || Wo % Wi != 0) return nullptr;
        if (Wi % 4 != 0 || NSEG >= (1LL << 30)) return nullptr;
        FCfg &c = x.c;
        c.Wi = Wi;
        c.Wo = Wo;
        c.P = Wo / Wi;
        c.Ho = Ho;
        c.Hi = Hi;
        c.K = K;
        c.BC = BC;
        c.NROWS = NROWS;
        c.Mi = Wi / 2;
        c.Mo = Wo / 2;
        c.NKi = c.Mi + 1;
        c.NKo = c.Mo + 1;
        c.NKip = (c.NKi + 1) & ~1;
        c.NKop = (c.NKo + 1) & ~1;
        if (!make_plan(c.pi, c.Mi) || !make_plan(c.po, c.Mo)) return nullptr;
        // Contraction tiling: CBM-channel tiles (balanced to Cb channels each) x F-bin chunks x FFT_NSLOT-ring groups;
        // F * CBM / 4 consumer threads. The chunk shrinks while there are too few work items to fill the GPU.
        x.cbm = BC >= 256 ? 32 : (BC > 8 ? 16 : 8);
        c.ntile = (BC + x.cbm - 1) / x.cbm;
        c.Cb = (BC + c.ntile - 1) / c.ntile;
        c.F = 256 * 4 / x.cbm;
        if (c.F > 64) c.F = 64;
        {
            const int ng_est = std::max(1, std::min((Ho + FFT_NSLOT - 1) / FFT_NSLOT, 8));
            while (c.F > 32 && (long long)ng_est * c.ntile * ((c.NKo + c.F - 1) / c.F) < 2 * FFT_NSM) c.F >>= 1;
        }
        c.nfc = (c.NKo + c.F - 1) / c.F;
        x.nt = c.F * (x.cbm / 4);
        x.smem_con = (size_t)2 * FFT_CV * fx_stage_f2(c.F, x.cbm) * 8;
        const size_t psi_smem = (size_t)(c.Mo + c.NKo + 2 * pd_size(FFT_PSI_NB * c.Mo)) * 8;
        c.nbf = std::max(1, std::min(32, 4096 / std::max(c.Mi, 1)));
        c.nbf = std::max(1, std::min(c.nbf, FFT_FWD_PV * 256 / std::max(c.Mi / 4, 1)));
        const size_t fwd_smem = (size_t)(c.Mi + c.NKi + 2 * pd_size(c.nbf * c.Mi)) * 8;
        c.nbi = std::max(1, std::min(32, 3072 / std::max(c.Mo, 1)));
        x.smem_inv = (size_t)(2 * c.Mo + 2 * pd_size(c.nbi * c.Mo)) * 8;
        if (psi_smem > 200 * 1024 || fwd_smem > 200 * 1024 || x.smem_inv > 200 * 1024) return nullptr;
        // cost model (seconds per channel): direct tensor-core k-steps vs FP32 contraction, plus per needed row
        // (about one per three arcs) its FFT and spectrum traffic, plus the inverse FFT and spectrum traffic per ring
        // (constants calibrated on H100 measurements of both paths)
        c.dunit = 7.5e-15f * 32.f;
        const float lg_i = log2f((float)std::max(c.Mi, 2)), lg_o = log2f((float)std::max(c.Mo, 2));
        const float fop = 1.0f / 11.9e12f, fby = 1.0f / 2.5e12f;
        const float rowcost = 4.f * (float)c.Mi * lg_i * fop + 16.f * (float)c.NKi * fby;
        c.fcost_arc = (float)c.NKo * 8.f * fop + rowcost / 3.f;
        c.fcost_ring = 4.f * (float)c.Mo * lg_o * fop + (16.f * (float)c.NKo + 2.f * (float)c.Wo) * fby;
        c.theta0 = BC >= 256 ? 0.75f : 0.90f;
        const size_t psi_budget = (size_t)192 << 20, spec_budget = (size_t)640 << 20, out_budget = (size_t)256 << 20;
        c.cap_pairs = (int)std::min<long long>(NSEG, (long long)(psi_budget / ((size_t)c.NKop * 8)));
        c.cap_rows = (int)std::min<long long>(NROWS, (long long)(spec_budget / ((size_t)BC * c.NKip * 8)));
        const long long cap_rings = std::min<long long>((long long)FFT_NRG_CAP * FFT_NSLOT,
                                                        (long long)(out_budget / ((size_t)BC * c.NKop * 8)));
        c.nrg_max = (int)std::min<long long>((Ho + FFT_NSLOT - 1) / FFT_NSLOT, cap_rings / FFT_NSLOT);
        if (c.cap_pairs < 1 || c.cap_rows < 1 || c.nrg_max < 1) return nullptr;

        size_t o = 0;
        auto take = [&](size_t b) {
            size_t r = o;
            o = fx_align(o + b, 256);
            return r;
        };
        const size_t o_dcost = take((size_t)Ho * 4);
        const size_t o_acnt = take((size_t)Ho * 4);
        const size_t o_vcount = take((size_t)c.nrg_max * 4);
        const size_t o_vcursor = take((size_t)c.nrg_max * 4);
        const size_t o_hist = take((size_t)3 * FFT_NBIN * 4);
        const size_t o_cnt = take(32);
        const size_t zero_end = o;
        const size_t o_rowmax = take((size_t)NROWS * 4);
        const size_t o_rslot = take((size_t)NROWS * 4);
        const size_t o_rxrow = take((size_t)c.cap_rows * 4);
        const size_t o_flag = take((size_t)Ho);
        const size_t o_fidx = take((size_t)Ho * 4);
        const size_t o_rlist = take((size_t)c.nrg_max * FFT_NSLOT * 4);
        const size_t o_vstart = take(((size_t)c.nrg_max + 1) * 4);
        const size_t o_pairs = take((size_t)c.cap_pairs * sizeof(Pair));
        const size_t o_visits = take((size_t)c.cap_pairs * sizeof(Visit));
        const size_t o_psi = take((size_t)c.cap_pairs * c.NKop * 8);
        const size_t o_spec = take((size_t)c.cap_rows * BC * c.NKip * 8);
        const size_t o_ospec = take((size_t)c.nrg_max * FFT_NSLOT * BC * c.NKop * 8);
        cudaMallocAsync((void **)&x.ws, o, stream);
        cudaMemsetAsync(x.ws, 0, zero_end, stream);
        float *dcost = reinterpret_cast<float *>(x.ws + o_dcost);
        int *acnt = reinterpret_cast<int *>(x.ws + o_acnt);
        int *vcount = reinterpret_cast<int *>(x.ws + o_vcount);
        int *vcursor = reinterpret_cast<int *>(x.ws + o_vcursor);
        int *hist = reinterpret_cast<int *>(x.ws + o_hist);
        // cnt: [0] ring groups, [1] pairs, [2] selected rows, [3] threshold bin, [4] flagged rings
        x.cnt = reinterpret_cast<int *>(x.ws + o_cnt);
        int *npairs = x.cnt + 1;
        float *rowmax = reinterpret_cast<float *>(x.ws + o_rowmax);
        int *rslot = reinterpret_cast<int *>(x.ws + o_rslot);
        int *rxrow = reinterpret_cast<int *>(x.ws + o_rxrow);
        x.flag = x.ws + o_flag;
        int *fidx = reinterpret_cast<int *>(x.ws + o_fidx);
        x.rlist = reinterpret_cast<int *>(x.ws + o_rlist);
        x.vstart = reinterpret_cast<int *>(x.ws + o_vstart);
        Pair *pairs = reinterpret_cast<Pair *>(x.ws + o_pairs);
        x.visits = reinterpret_cast<Visit *>(x.ws + o_visits);
        x.psi = reinterpret_cast<float2 *>(x.ws + o_psi);
        x.spec = reinterpret_cast<float2 *>(x.ws + o_spec);
        x.ospec = reinterpret_cast<float2 *>(x.ws + o_ospec);

        const int rb = 128, rg = (NROWS + rb - 1) / rb;
        const int wb = 256;
        const int wg = (int)((std::max<long long>((long long)NROWS * 32, Ho) + wb - 1) / wb);
        k_fx_cost<<<wg, wb, 0, stream>>>(c, d_so, d_seg, dcost, acnt);
        fx_launch_pdl(k_fx_rowmax, dim3(wg), dim3(wb), 0, stream, c, d_so, d_seg, dcost, acnt, rowmax, hist);
        fx_launch_pdl(k_fx_select, dim3(1), dim3(1024), 0, stream, c, dcost, acnt, hist, x.flag, fidx, x.rlist, x.cnt);
        fx_launch_pdl(k_fx_count, dim3(rg), dim3(rb), 0, stream, c, d_so, d_seg, d_vo, d_rk, d_rl, rowmax, x.flag, fidx,
                      x.cnt, rslot, rxrow, vcount);
        fx_launch_pdl(k_fx_scan, dim3(1), dim3(1024), 0, stream, x.cnt, vcount, x.vstart);
        fx_launch_pdl(k_fx_place, dim3(rg), dim3(rb), 0, stream, c, d_so, d_seg, d_vo, x.flag, fidx, rslot, x.vstart,
                      vcursor, npairs, pairs, x.visits);
        static bool attr_set = false;
        if (!attr_set) {
            cudaFuncSetAttribute(k_fx_psi, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
            cudaFuncSetAttribute(k_fx_fwd, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
            cudaFuncSetAttribute(k_fx_inv, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
            attr_set = true;
        }
        const int psi_blocks = std::min((c.cap_pairs + FFT_PSI_NB - 1) / FFT_PSI_NB, 2 * FFT_NSM);
        fx_launch_pdl(k_fx_psi, dim3(psi_blocks), dim3(256), psi_smem, stream, c, npairs, pairs, d_seg, d_vals, x.psi);
        const long long ntr_max = (long long)c.cap_rows * BC;
        const int fwd_blocks = (int)std::min<long long>((ntr_max + c.nbf - 1) / c.nbf, 4 * FFT_NSM);
        fx_launch_pdl(k_fx_fwd, dim3(fwd_blocks), dim3(256), fwd_smem, stream, c, x.cnt, rxrow, d_inp, x.spec);
        x.on = true;
        return x.flag;
    }

    template <int CBM> static void launch_contract(Ctx &x, cudaStream_t stream)
    {
        static bool attr_set = false;
        if (!attr_set) {
            cudaFuncSetAttribute(k_fx_contract<CBM, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
            attr_set = true;
        }
        const int grid = std::min(x.c.nrg_max * x.c.ntile * x.c.nfc, 2 * FFT_NSM);
        k_fx_contract<CBM, 4><<<grid, x.nt, x.smem_con, stream>>>(x.c, x.cnt, x.vstart, x.visits, x.psi, x.spec, x.ospec);
    }

    // Runs after the direct path (it overwrites the flagged rings, which the direct path skipped).
    static void Finish(Ctx &x, __half *d_out, cudaStream_t stream)
    {
        if (!x.on) return;
        if (x.cbm == 32)
            launch_contract<32>(x, stream);
        else if (x.cbm == 16)
            launch_contract<16>(x, stream);
        else
            launch_contract<8>(x, stream);
        const long long ntr_max = (long long)x.c.nrg_max * FFT_NSLOT * x.c.BC;
        const int inv_blocks = (int)std::min<long long>((ntr_max + x.c.nbi - 1) / x.c.nbi, 4 * FFT_NSM);
        fx_launch_pdl(k_fx_inv, dim3(inv_blocks), dim3(256), x.smem_inv, stream, x.c, x.cnt, x.rlist, x.ospec, d_out);
        cudaFreeAsync(x.ws, stream);
        x.on = false;
    }

} // namespace fftp
