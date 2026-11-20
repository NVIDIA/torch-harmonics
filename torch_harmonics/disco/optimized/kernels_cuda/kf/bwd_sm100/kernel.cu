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

// DISCO S2 transpose contraction (arc-form psi), fp16 in/out, fp32 psi + accumulation.
//
// Exact fast path (convolution theorem along longitude):
//   out[b,c,h,:] = sum_{arcs s on ring h} p_s (*) u_{row(s)}     (circular convolution, length Wo)
//   p_s = arc s placed on the Wo circle, u_r = input row r upsampled by pscale, so
//   DFT_Wo(u_r)[m] = DFT_Wi(x_r)[m mod Wi] and
//   OUT[b,c,h,m] = sum_s P[s,m] * X[b,c,row(s),m mod Wi],  m in [0, Wo/2];  out = irfft(OUT).
// Every call: prep (arc value offsets, per-band entry lists, twiddle tables) -> psi arc spectra P ->
//   input spectra X -> banded frequency-space contraction -> inverse real FFT rounded once to fp16.
// FFTs: real FFTs of even length 2n via complex n-point FFTs. n = N1 * N2 is factored at run time
//   (radices from a fixed set of small register DFTs) into a two-pass register FFT; other smooth n
//   use a multi-pass shared-memory Stockham FFT. Odd or non-smooth widths use a direct fp32 gather.
// Contraction: block = (band of HB rings, channel group, slot tile); one thread streams the X tile of
//   every row in the band plus the psi spectra of its arcs with bulk copies into a CSTG-stage ring of
//   transaction barriers; threads own (channel, MT slots, HB rings) fp32 accumulators. Arcs exactly centred
//   on longitude 0 with exactly (anti)symmetric values have real (imaginary) spectra, which halves the MACs.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>

namespace ffi = tvm::ffi;

// ---------------------------------------------------------------- packed fp32x2 (sm_100) and complex helpers
__device__ __forceinline__ float2 add2(float2 a, float2 b)
{
    float2 d;
    asm("{.reg .b64 ra, rb;\n mov.b64 ra, {%2, %3};\n mov.b64 rb, {%4, %5};\n add.rn.f32x2 ra, ra, rb;\n mov.b64 {%0, "
        "%1}, ra;\n}"
        : "=f"(d.x), "=f"(d.y)
        : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
    return d;
}
__device__ __forceinline__ float2 mul2(float2 a, float2 b)
{
    float2 d;
    asm("{.reg .b64 ra, rb;\n mov.b64 ra, {%2, %3};\n mov.b64 rb, {%4, %5};\n mul.rn.f32x2 ra, ra, rb;\n mov.b64 {%0, "
        "%1}, ra;\n}"
        : "=f"(d.x), "=f"(d.y)
        : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
    return d;
}
// a * b + c, elementwise
__device__ __forceinline__ float2 fma2(float2 a, float2 b, float2 c)
{
    float2 d;
    asm("{.reg .b64 ra, rb, rc;\n mov.b64 ra, {%2, %3};\n mov.b64 rb, {%4, %5};\n mov.b64 rc, {%6, %7};\n"
        " fma.rn.f32x2 rc, ra, rb, rc;\n mov.b64 {%0, %1}, rc;\n}"
        : "=f"(d.x), "=f"(d.y)
        : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y), "f"(c.x), "f"(c.y));
    return d;
}
__device__ __forceinline__ float2 cadd(float2 a, float2 b) { return add2(a, b); }
__device__ __forceinline__ float2 csub(float2 a, float2 b) { return add2(a, make_float2(-b.x, -b.y)); }
// a * b = a.x * b + a.y * (i b)
__device__ __forceinline__ float2 cmul(float2 a, float2 b)
{
    return fma2(make_float2(a.y, a.y), make_float2(-b.y, b.x), mul2(make_float2(a.x, a.x), b));
}
__device__ __forceinline__ float2 cscale(float2 a, float s) { return mul2(a, make_float2(s, s)); }
__device__ __forceinline__ float2 mul_negi(float2 a) { return make_float2(a.y, -a.x); } // -i * a

__device__ __forceinline__ uint32_t smem_u32(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }

// ---------------------------------------------------------------- small forward DFTs (w = e^{-2 pi i / R})
template <int R> __device__ __forceinline__ void dft(float2 *v);

template <> __device__ __forceinline__ void dft<2>(float2 *v)
{
    float2 t = v[0];
    v[0] = cadd(t, v[1]);
    v[1] = csub(t, v[1]);
}

template <> __device__ __forceinline__ void dft<3>(float2 *v)
{
    const float S3 = 0.86602540378443864676f;
    float2 t1 = cadd(v[1], v[2]);
    float2 t2 = csub(v[1], v[2]);
    float2 m1 = make_float2(v[0].x - 0.5f * t1.x, v[0].y - 0.5f * t1.y);
    float2 m2 = cscale(mul_negi(t2), S3);
    v[0] = cadd(v[0], t1);
    v[1] = cadd(m1, m2);
    v[2] = csub(m1, m2);
}

template <> __device__ __forceinline__ void dft<4>(float2 *v)
{
    float2 t0 = cadd(v[0], v[2]);
    float2 t1 = csub(v[0], v[2]);
    float2 t2 = cadd(v[1], v[3]);
    float2 t3 = mul_negi(csub(v[1], v[3]));
    v[0] = cadd(t0, t2);
    v[2] = csub(t0, t2);
    v[1] = cadd(t1, t3);
    v[3] = csub(t1, t3);
}

template <> __device__ __forceinline__ void dft<5>(float2 *v)
{
    const float C1 = 0.30901699437494742410f;  // cos(2pi/5)
    const float C2 = -0.80901699437494742410f; // cos(4pi/5)
    const float S1 = 0.95105651629515357212f;  // sin(2pi/5)
    const float S2 = 0.58778525229247312917f;  // sin(4pi/5)
    float2 a0 = v[0];
    float2 t1 = cadd(v[1], v[4]);
    float2 t2 = cadd(v[2], v[3]);
    float2 t3 = csub(v[1], v[4]);
    float2 t4 = csub(v[2], v[3]);
    float2 b1r = make_float2(a0.x + C1 * t1.x + C2 * t2.x, a0.y + C1 * t1.y + C2 * t2.y);
    float2 b2r = make_float2(a0.x + C2 * t1.x + C1 * t2.x, a0.y + C2 * t1.y + C1 * t2.y);
    float2 s1 = make_float2(S1 * t3.x + S2 * t4.x, S1 * t3.y + S2 * t4.y);
    float2 s2 = make_float2(S2 * t3.x - S1 * t4.x, S2 * t3.y - S1 * t4.y);
    float2 i1 = mul_negi(s1);
    float2 i2 = mul_negi(s2);
    v[0] = cadd(a0, cadd(t1, t2));
    v[1] = cadd(b1r, i1);
    v[4] = csub(b1r, i1);
    v[2] = cadd(b2r, i2);
    v[3] = csub(b2r, i2);
}

template <> __device__ __forceinline__ void dft<8>(float2 *v)
{
    const float H = 0.70710678118654752440f;
    float2 e[4] = {v[0], v[2], v[4], v[6]};
    float2 o[4] = {v[1], v[3], v[5], v[7]};
    dft<4>(e);
    dft<4>(o);
    o[1] = make_float2(H * (o[1].x + o[1].y), H * (o[1].y - o[1].x));
    o[2] = mul_negi(o[2]);
    o[3] = make_float2(H * (o[3].y - o[3].x), -H * (o[3].x + o[3].y));
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        v[k] = cadd(e[k], o[k]);
        v[k + 4] = csub(e[k], o[k]);
    }
}

#include "radix_twiddles.inc"

__host__ __device__ constexpr int ctf(int N) { return N % 4 == 0 ? 4 : (N % 3 == 0 ? 3 : (N % 2 == 0 ? 2 : 5)); }

template <int N> __device__ __forceinline__ void dftn(float2 *v);

// in-register Cooley-Tukey DFT for a composite radix N (compile-time indices, constant-bank twiddles)
template <int N> __device__ __forceinline__ void dft_ct(float2 *v)
{
    constexpr int F = ctf(N);
    constexpr int M = N / F;
    float2 sub[F][M];
#pragma unroll
    for (int f = 0; f < F; ++f)
#pragma unroll
        for (int j = 0; j < M; ++j) sub[f][j] = v[f + F * j];
#pragma unroll
    for (int f = 0; f < F; ++f) dftn<M>(sub[f]);
#pragma unroll
    for (int f = 1; f < F; ++f)
#pragma unroll
        for (int k = 1; k < M; ++k) sub[f][k] = cmul(sub[f][k], c_rtw[rtw_off(N) + f * k]);
#pragma unroll
    for (int k = 0; k < M; ++k) {
        float2 t[F];
#pragma unroll
        for (int f = 0; f < F; ++f) t[f] = sub[f][k];
        dftn<F>(t);
#pragma unroll
        for (int q = 0; q < F; ++q) v[k + M * q] = t[q];
    }
}

template <int N> __device__ __forceinline__ void dftn(float2 *v)
{
    if constexpr (N == 1) {
    } else if constexpr (N == 2 || N == 3 || N == 4 || N == 5 || N == 8) {
        dft<N>(v);
    } else {
        dft_ct<N>(v);
    }
}

// radix set of the two-pass register FFT
#define RADIX_SET(F)                                                                                                   \
    F(2) F(3) F(4) F(5) F(6) F(8) F(9) F(10) F(12) F(15) F(16) F(18) F(20) F(24) F(25) F(27) F(30) F(32)
static const int kRadix[] = {2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32};

template <int V> struct IC {
    static constexpr int value = V;
};

template <class Fn> __device__ __forceinline__ void with_radix(int R, Fn &&fn)
{
    switch (R) {
#define RADIX_CASE(V)                                                                                                  \
    case V: fn(IC<V> {}); break;
        RADIX_SET(RADIX_CASE)
#undef RADIX_CASE
    default: break;
    }
}

struct Plan2 {
    int n, N1, N2, LD, SROW, NT;
};

// pass A of the two-pass FFT (n = N1 * N2): thread column j2 gathers z[N2*j1 + j2] (j1 < N1), DFT_N1,
// twiddle W_n^{j2*k1}, store S[k1 * LD + j2]. T[t] = e^{-2 pi i t / (2n)}.
// sync: barrier between the gather and the store (the source aliases S).
// DFT_N1 of a gathered column, twiddle W_n^{j2*k1}, store S[k1 * LD + j2]. T[t] = e^{-2 pi i t / (2n)}.
template <int N1>
__device__ __forceinline__ void passA_dft_store(float2 *v, int j2, int n, const float2 *__restrict__ T,
                                                float2 *__restrict__ S, int LD)
{
    dftn<N1>(v);
    constexpr int NG = (N1 + 3) / 4;
    const int n2 = 2 * n;
    const float2 w1 = __ldg(T + 2 * j2);
    const float2 w4 = __ldg(T + (8 * j2) % n2);
    float2 wb[4];
    wb[0] = make_float2(1.f, 0.f);
    wb[1] = w1;
    wb[2] = cmul(w1, w1);
    wb[3] = cmul(wb[2], w1);
    float2 wg = make_float2(1.f, 0.f);
#pragma unroll
    for (int a = 0; a < NG; ++a) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int k = 4 * a + b;
            if (k < N1) {
                float2 t = v[k];
                if (k > 0) t = cmul(t, a == 0 ? wb[b] : (b == 0 ? wg : cmul(wg, wb[b])));
                S[k * LD + j2] = t;
            }
        }
        wg = (a == 0) ? w4 : cmul(wg, w4);
    }
}

// pass A of the two-pass FFT (n = N1 * N2): thread column j2 gathers z[N2*j1 + j2] (j1 < N1), DFT_N1,
// twiddle W_n^{j2*k1}, store S[k1 * LD + j2]. sync: barrier between the gather and the store (the source
// aliases S).
template <int N1, class Load>
__device__ __forceinline__ void fft_passA(bool act, Load &&load, int j2, int n, int N2, const float2 *__restrict__ T,
                                          float2 *__restrict__ S, int LD, bool sync)
{
    float2 v[N1];
    if (act) {
#pragma unroll
        for (int j1 = 0; j1 < N1; ++j1) v[j1] = load(N2 * j1 + j2);
    }
    if (sync) __syncthreads();
    if (act) passA_dft_store<N1>(v, j2, n, T, S, LD);
}

// pass A for an arc row. Dense rows gather from the row in S (aliased, so a barrier separates gather and store).
// A short arc (spanning <= N2 complex points c0 .. c0 + span - 1, mod n) has at most one nonzero point
// kpos = N2*j1 + j2 per column, so DFT_N1 + twiddle reduce to S[k1][j2] = z * W_n^{kpos*k1}.
template <int N1>
__device__ __forceinline__ void psi_passA(bool act, bool sparse, int j2, int n, int N2, const float2 *__restrict__ T,
                                          float2 *__restrict__ S, int LD, int c0, int span, int start, int len,
                                          const float *__restrict__ v)
{
    float2 vv[N1];
    if (act && !sparse) {
#pragma unroll
        for (int j1 = 0; j1 < N1; ++j1) vv[j1] = S[N2 * j1 + j2];
    }
    __syncthreads();
    if (!act) return;
    if (!sparse) {
        passA_dft_store<N1>(vv, j2, n, T, S, LD);
        return;
    }
    int d = j2 - c0 % N2;
    if (d < 0) d += N2;
    float2 p = make_float2(0.f, 0.f), w = make_float2(1.f, 0.f);
    if (d < span) {
        int kpos = c0 + d;
        if (kpos >= n) kpos -= n;
        const int Wo = 2 * n;
        int o0 = 2 * kpos - start;
        if (o0 < 0) o0 += Wo;
        int o1 = o0 + 1;
        if (o1 >= Wo) o1 -= Wo;
        p = make_float2(o0 < len ? __ldg(v + o0) : 0.f, o1 < len ? __ldg(v + o1) : 0.f);
        w = __ldg(T + 2 * kpos);
    }
#pragma unroll
    for (int k1 = 0; k1 < N1; ++k1) {
        S[k1 * LD + j2] = p;
        p = cmul(p, w);
    }
}

// pass B: thread k1 reads S[k1 * LD + j2] (j2 < N2), DFT_N2, then (after a barrier) store(k2, U[k2]),
// which is Z[k1 + N1 * k2].
template <int N2, class Store>
__device__ __forceinline__ void fft_passB(bool act, const float2 *__restrict__ Srow, Store &&store)
{
    float2 u[N2];
    if (act) {
#pragma unroll
        for (int j2 = 0; j2 < N2; ++j2) u[j2] = Srow[j2];
        dftn<N2>(u);
    }
    __syncthreads();
    if (act) {
#pragma unroll
        for (int k2 = 0; k2 < N2; ++k2) store(k2, u[k2]);
    }
}

// real-FFT post-processing: Z = FFT_n(x[2k] + i x[2k+1]) -> X[m], m in [0, n]; tw = e^{-2 pi i m / (2n)}
__device__ __forceinline__ float2 rpost(float2 zm, float2 zn, float2 tw)
{
    const float2 ze = make_float2(0.5f * (zm.x + zn.x), 0.5f * (zm.y - zn.y));
    const float2 d = make_float2(zm.x - zn.x, zm.y + zn.y);
    const float2 zo = make_float2(0.5f * d.y, -0.5f * d.x);
    return cadd(ze, cmul(tw, zo));
}

// both halves of a real-FFT pair: X[m] = ze + t and X[n - m] = conj(ze - t)
__device__ __forceinline__ void rpost2(float2 zm, float2 zn, float2 tw, float2 &xm, float2 &xnm)
{
    const float2 ze = make_float2(0.5f * (zm.x + zn.x), 0.5f * (zm.y - zn.y));
    const float2 d = make_float2(zm.x - zn.x, zm.y + zn.y);
    const float2 tz = cmul(tw, make_float2(0.5f * d.y, -0.5f * d.x));
    xm = cadd(ze, tz);
    xnm = make_float2(ze.x - tz.x, tz.y - ze.y);
}

// ---------------------------------------------------------------- input spectra (two-pass)
// X[((cg*KH + kh)*MPi + m)*LC + cl] = DFT_Wi(x[bc, kh, :])[m], bc = cg*LC + cl; zero for m >= Mi or bc >= BC.
// grid (octet of the channel group, kh, channel group); block (max(N1, N2), 8): row r = threadIdx.y
__global__ void __launch_bounds__(256, 3) fwdx2_kernel(const __half *__restrict__ x, float2 *__restrict__ X, int BC,
                                                       int KH, Plan2 pl, const float2 *__restrict__ T, int LC, int MPi)
{
    extern __shared__ float4 smem_f4[];
    float2 *S = reinterpret_cast<float2 *>(smem_f4);
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int NT = blockDim.x * 8, t = tx + blockDim.x * ty;
    const int kh = blockIdx.y, cg = blockIdx.z, cl0 = blockIdx.x * 8, bc0 = cg * LC + cl0;
    const int n = pl.n, N1 = pl.N1, N2 = pl.N2, LD = pl.LD, SROW = pl.SROW, Mi = n + 1;
    const int nlc = min(8, LC - cl0); // channels of this octet inside the group (LC is a multiple of 4)
    float2 *Xb = X + ((size_t)(cg * KH + kh) * MPi) * LC + cl0;
    if (bc0 >= BC) { // all channels are padding
        for (int i = t; i < 8 * MPi; i += NT)
            if ((i & 7) < nlc) Xb[(size_t)(i >> 3) * LC + (i & 7)] = make_float2(0.f, 0.f);
        return;
    }
    float2 *Sr = S + ty * SROW;
    const bool live = bc0 + ty < BC && ty < nlc;
    {
        const bool act = live && tx < N2;
        const __half2 *xr
            = reinterpret_cast<const __half2 *>(x + ((size_t)(live ? bc0 + ty : bc0) * KH + kh) * (size_t)(2 * n));
        with_radix(N1, [&](auto RC) {
            constexpr int R = decltype(RC)::value;
            fft_passA<R>(act, [&](int idx) { return __half22float2(xr[idx]); }, tx, n, N2, T, Sr, LD, false);
        });
    }
    __syncthreads();
    with_radix(N2, [&](auto RC) {
        constexpr int R = decltype(RC)::value;
        fft_passB<R>(live && tx < N1, Sr + tx * LD, [&](int k2, float2 u) { Sr[tx + N1 * k2] = u; });
    });
    __syncthreads();
    const int r = t & 7, m0 = t >> 3, ms = NT >> 3;
    if (r >= nlc) return;
    const bool ok = bc0 + r < BC;
    const float2 *Zr = S + r * SROW;
    const size_t step = (size_t)ms * LC;
    float2 *xa = Xb + r + (size_t)m0 * LC;
    float2 *xb = Xb + r + (size_t)(n - m0) * LC;
    for (int m = m0; 2 * m <= n; m += ms, xa += step, xb -= step) {
        float2 a = make_float2(0.f, 0.f), b = a;
        if (ok) rpost2(Zr[m], Zr[m == 0 ? 0 : n - m], __ldg(T + m), a, b);
        *xa = a;
        *xb = b;
    }
    if (Mi + m0 < MPi) {
        float2 *xz = Xb + r + (size_t)(Mi + m0) * LC;
        for (int m = Mi + m0; m < MPi; m += ms, xz += step) *xz = make_float2(0.f, 0.f);
    }
}

// ---------------------------------------------------------------- psi arc spectra (two-pass)
// P is tile-major: slot j < PLD of plane o for arc s lives at ((j / MBt) * nseg + s) * NO * MBt + o * MBt + j % MBt,
// so the arcs of one slot tile are contiguous. Plane 0 holds frequency j, plane 1 (pscale 2) frequency Wi - j;
// zero for j >= nslot. Arcs of any length (wrapping onto themselves) are summed. Block (max(N1, N2), 8).
__global__ void __launch_bounds__(256, 3)
    fwdpsi2_kernel(const int *__restrict__ seg, const int *__restrict__ segval, const float *__restrict__ vals,
                   float2 *__restrict__ P, int nseg, Plan2 pl, const float2 *__restrict__ T, int Wi, int NO, int nslot,
                   int PLD, int MBt, int mbt_sh)
{
    extern __shared__ float4 smem_f4[];
    float2 *S = reinterpret_cast<float2 *>(smem_f4);
    __shared__ int s_st[8], s_len[8], s_vo[8], s_short[8];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int NT = blockDim.x * 8, t = tx + blockDim.x * ty;
    const int n = pl.n, N1 = pl.N1, N2 = pl.N2, LD = pl.LD, SROW = pl.SROW, Wo = 2 * n;
    const int s0 = blockIdx.x * 8;
    if (t < 8) {
        const int s = s0 + t;
        int start = 0, len = 0, vo = 0;
        if (s < nseg) {
            start = seg[3 * s + 1] % Wo;
            if (start < 0) start += Wo;
            len = max(seg[3 * s + 2], 0);
            vo = segval[s];
        }
        s_st[t] = start;
        s_len[t] = len;
        s_vo[t] = vo;
        s_short[t] = len <= Wo && (((start & 1) + len + 1) >> 1) <= N2;
    }
    __syncthreads();
    for (int r = 0; r < 8; ++r) {
        if (s_short[r]) continue;
        float4 *z4 = reinterpret_cast<float4 *>(S + r * SROW);
        for (int i = t; i < SROW / 2; i += NT) z4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __syncthreads();
    bool wraps = false;
    for (int r = 0; r < 8; ++r) {
        if (s_short[r]) continue;
        const int len = s_len[r], start = s_st[r];
        wraps |= len > Wo;
        const float *vr = vals + s_vo[r];
        float *row = reinterpret_cast<float *>(S + r * SROW);
        for (int j = t; j < min(len, Wo); j += NT) {
            int pos = start + j;
            if (pos >= Wo) pos -= Wo;
            row[pos] = __ldg(vr + j);
        }
    }
    if (wraps) { // arcs longer than the circle wrap onto themselves
        __syncthreads();
        for (int r = 0; r < 8; ++r) {
            const int len = s_len[r], start = s_st[r];
            const float *vr = vals + s_vo[r];
            float *row = reinterpret_cast<float *>(S + r * SROW);
            for (int j = Wo + t; j < len; j += NT) atomicAdd(row + (start + j) % Wo, __ldg(vr + j));
        }
    }
    __syncthreads();
    float2 *Sr = S + ty * SROW;
    const bool live = s0 + ty < nseg;
    {
        const int start = s_st[ty], len = s_len[ty];
        const int span = ((start & 1) + len + 1) >> 1;
        const float *vr = vals + s_vo[ty];
        const bool sparse = s_short[ty] != 0;
        with_radix(N1, [&](auto RC) {
            constexpr int R = decltype(RC)::value;
            psi_passA<R>(live && tx < N2, sparse, tx, n, N2, T, Sr, LD, start >> 1, span, start, len, vr);
        });
    }
    __syncthreads();
    with_radix(N2, [&](auto RC) {
        constexpr int R = decltype(RC)::value;
        fft_passB<R>(live && tx < N1, Sr + tx * LD, [&](int k2, float2 u) { Sr[tx + N1 * k2] = u; });
    });
    __syncthreads();
    const int nr = min(8, nseg - s0);
    for (int r = 0; r < nr; ++r) {
        const float2 *Zr = S + r * SROW;
        float2 *Pr = P + (size_t)(s0 + r) * NO * MBt;
        auto put = [&](int o, int j, float2 v) {
            const int tj = mbt_sh >= 0 ? (j >> mbt_sh) : 0;
            Pr[(size_t)tj * nseg * NO * MBt + o * MBt + (j - tj * MBt)] = v;
        };
        for (int m = t; 2 * m <= n; m += NT) {
            float2 a, b;
            rpost2(Zr[m], Zr[m == 0 ? 0 : n - m], __ldg(T + m), a, b);
            // frequency f -> (plane, slot): pscale 2 keeps f <= Wi/2 in plane 0, f >= Wi/2 in plane 1 at Wi - f
            if (NO == 2) {
                put(0, m, a); // m <= n/2 = Wi/2
                put(1, m, b); // n - m >= Wi/2, slot Wi - (n - m) = m
            } else {
                put(0, m, a);
                if (n - m != m) put(0, n - m, b);
            }
        }
        for (int j = nslot + t; j < PLD; j += NT)
            for (int o = 0; o < NO; ++o) put(o, j, make_float2(0.f, 0.f));
    }
}

// ---------------------------------------------------------------- inverse real FFT -> fp16 (two-pass)
// out[bc, h, :] = irfft(OUT[h, :, bc]); grid (h, 8 channels); block (max(N1, N2), 8)
__global__ void __launch_bounds__(256, 3) inv2_kernel(const float2 *__restrict__ OUT, __half *__restrict__ out, int BC,
                                                      int BCp, int Ho, Plan2 pl, const float2 *__restrict__ T)
{
    extern __shared__ float4 smem_f4[];
    float2 *S = reinterpret_cast<float2 *>(smem_f4);
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int NT = blockDim.x * 8, t = tx + blockDim.x * ty;
    const int n = pl.n, N1 = pl.N1, N2 = pl.N2, LD = pl.LD, SROW = pl.SROW, Mf = n + 1;
    const int h = blockIdx.x, bc0 = blockIdx.y * 8;
    if (bc0 >= BC) return;
    {
        const int r = t & 7, ms = NT >> 3;
        const char *src = reinterpret_cast<const char *>(OUT + ((size_t)h * Mf + (t >> 3)) * BCp + bc0 + r);
        const size_t step = (size_t)ms * BCp * sizeof(float2);
        uint32_t dst = smem_u32(S + r * SROW + (t >> 3));
        if (bc0 + r < BC)
            for (int m = t >> 3; m < Mf; m += ms, src += step, dst += ms * (uint32_t)sizeof(float2))
                asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(dst), "l"(src) : "memory");
        asm volatile("cp.async.wait_all;\n" ::: "memory");
    }
    __syncthreads();
    float2 *Sr = S + ty * SROW;
    const bool live = bc0 + ty < BC;
    with_radix(N1, [&](auto RC) {
        constexpr int R = decltype(RC)::value;
        fft_passA<R>(
            live && tx < N2,
            [&](int m) {
                const float2 xm = Sr[m], xn = Sr[n - m];
                const float2 e = make_float2(xm.x + xn.x, xm.y - xn.y);
                const float2 d = make_float2(xm.x - xn.x, xm.y + xn.y);
                const float2 tw = __ldg(T + m);
                const float2 q = cmul(make_float2(tw.x, -tw.y), d);
                return make_float2(e.x - q.y, -(e.y + q.x));
            },
            tx, n, N2, T, Sr, LD, true);
    });
    __syncthreads();
    {
        const float sc = 1.0f / (float)(2 * n);
        __half2 *o = reinterpret_cast<__half2 *>(out + ((size_t)(live ? bc0 + ty : bc0) * Ho + h) * (size_t)(2 * n));
        with_radix(N2, [&](auto RC) {
            constexpr int R = decltype(RC)::value;
            fft_passB<R>(live && tx < N1, Sr + tx * LD,
                         [&](int k2, float2 u) { o[tx + N1 * k2] = __floats2half2_rn(u.x * sc, -u.y * sc); });
        });
    }
}

// ---------------------------------------------------------------- multi-pass Stockham FFT (general smooth n)
#define MAXST 24
struct PlanM {
    int n;
    int nst;
    int r[MAXST];
};

template <int R>
__device__ __forceinline__ void stage(const float2 *__restrict__ src, float2 *__restrict__ dst,
                                      const float2 *__restrict__ T, int n, int Ns, int j)
{
    const int nR = n / R;
    const int k = j % Ns;
    const int step = 2 * k * (n / (Ns * R)); // T has 2n entries
    float2 v[R];
#pragma unroll
    for (int r = 0; r < R; ++r) v[r] = src[j + r * nR];
#pragma unroll
    for (int r = 1; r < R; ++r) v[r] = cmul(v[r], __ldg(T + r * step));
    dftn<R>(v);
    const int idxD = (j - k) * R + k;
#pragma unroll
    for (int r = 0; r < R; ++r) dst[idxD + r * Ns] = v[r];
}

// forward complex FFT of nb rows of length p.n (row stride ld), ping-ponging A/B; returns the result buffer
__device__ float2 *block_fft(float2 *A, float2 *B, int ld, int nb, const float2 *T, const PlanM &p)
{
    const int n = p.n;
    int Ns = 1;
    for (int st = 0; st < p.nst; ++st) {
        const int R = p.r[st];
        const int nR = n / R;
        const int total = nb * nR;
        for (int w = threadIdx.x; w < total; w += blockDim.x) {
            const int a = w / nR;
            const int j = w - a * nR;
            const float2 *s = A + a * ld;
            float2 *d = B + a * ld;
            switch (R) {
            case 8: stage<8>(s, d, T, n, Ns, j); break;
            case 4: stage<4>(s, d, T, n, Ns, j); break;
            case 2: stage<2>(s, d, T, n, Ns, j); break;
            case 5: stage<5>(s, d, T, n, Ns, j); break;
            default: stage<3>(s, d, T, n, Ns, j); break;
            }
        }
        __syncthreads();
        float2 *t = A;
        A = B;
        B = t;
        Ns *= R;
    }
    return A;
}

// rows: (kh, bc) for bc in [0, BCp), row = bc * KH + kh ... flattened as blockIdx-strided groups of NB
__global__ void fwdxm_kernel(const __half *__restrict__ x, float2 *__restrict__ X, int BC, int BCp, int KH, PlanM p,
                             const float2 *__restrict__ T, int NB, int LC, int MPi)
{
    extern __shared__ float4 smem_f4[];
    float2 *A = reinterpret_cast<float2 *>(smem_f4);
    const int n = p.n;
    float2 *B = A + NB * n;
    const int tid = threadIdx.x, bd = blockDim.x;
    const long nrows = (long)BCp * KH;
    const long row0 = (long)blockIdx.x * NB;
    const int nb = (int)min((long)NB, nrows - row0);
    for (int w = tid; w < nb * n; w += bd) {
        const int a = w / n, k = w - a * n;
        const long row = row0 + a;
        const long bc = row / KH, kh = row - bc * KH;
        A[a * n + k] = bc < BC ?
            __half22float2(reinterpret_cast<const __half2 *>(x + (bc * KH + kh) * (long)(2 * n))[k]) :
            make_float2(0.f, 0.f);
    }
    __syncthreads();
    const float2 *Z = block_fft(A, B, n, nb, T, p);
    const int Mi = n + 1;
    for (int w = tid; w < nb * MPi; w += bd) {
        const int a = w / MPi, m = w - a * MPi;
        const long row = row0 + a;
        const long bc = row / KH, kh = row - bc * KH;
        const long cg = bc / LC, cl = bc - cg * LC;
        float2 val = make_float2(0.f, 0.f);
        if (m < Mi) val = rpost(Z[a * n + (m == n ? 0 : m)], Z[a * n + (m == 0 ? 0 : n - m)], __ldg(T + m));
        X[((cg * KH + kh) * MPi + m) * LC + cl] = val;
    }
}

__global__ void fwdpsim_kernel(const int *__restrict__ seg, const int *__restrict__ segval,
                               const float *__restrict__ vals, float2 *__restrict__ P, int nseg, PlanM p,
                               const float2 *__restrict__ T, int NB, int Wi, int NO, int nslot, int PLD, int MBt)
{
    extern __shared__ float4 smem_f4[];
    float2 *A = reinterpret_cast<float2 *>(smem_f4);
    const int n = p.n, Wo = 2 * n;
    float2 *B = A + NB * n;
    const int tid = threadIdx.x, bd = blockDim.x;
    const int s0 = blockIdx.x * NB;
    const int nb = min(NB, nseg - s0);
    for (int w = tid; w < nb * Wo; w += bd) {
        const int a = w / Wo, pos = w - a * Wo;
        const int s = s0 + a;
        int start = seg[3 * s + 1] % Wo;
        if (start < 0) start += Wo;
        const int len = seg[3 * s + 2];
        int j = pos - start;
        if (j < 0) j += Wo;
        float v = 0.f;
        for (; j < len; j += Wo) v += vals[segval[s] + j];
        reinterpret_cast<float *>(A + a * n)[pos] = v;
    }
    __syncthreads();
    const float2 *Z = block_fft(A, B, n, nb, T, p);
    const int RW = NO * PLD;
    for (int w = tid; w < nb * RW; w += bd) {
        const int a = w / RW, rem = w - a * RW;
        const int o = rem >= PLD ? 1 : 0, j = rem - o * PLD;
        float2 val = make_float2(0.f, 0.f);
        if (j < nslot) {
            const int m = o ? Wi - j : j;
            val = rpost(Z[a * n + (m == n ? 0 : m)], Z[a * n + (m == 0 ? 0 : n - m)], __ldg(T + m));
        }
        const int tj = j / MBt;
        P[((size_t)tj * nseg + s0 + a) * NO * MBt + o * MBt + (j - tj * MBt)] = val;
    }
}

__global__ void invm_kernel(const float2 *__restrict__ OUT, __half *__restrict__ out, int BC, int BCp, int Ho, PlanM p,
                            const float2 *__restrict__ T, int NB)
{
    extern __shared__ float4 smem_f4[];
    float2 *A = reinterpret_cast<float2 *>(smem_f4);
    const int n = p.n, Mf = n + 1;
    float2 *B = A + NB * n;
    const int tid = threadIdx.x, bd = blockDim.x;
    const long nrows = (long)BC * Ho;
    const long row0 = (long)blockIdx.x * NB;
    const int nb = (int)min((long)NB, nrows - row0);
    for (int w = tid; w < nb * n; w += bd) {
        const int a = w / n, m = w - a * n;
        const long row = row0 + a;
        const long bc = row / Ho, h = row - bc * Ho;
        const float2 *src = OUT + h * Mf * (long)BCp + bc;
        const float2 xm = src[(long)m * BCp], xn = src[(long)(n - m) * BCp];
        const float2 e = make_float2(xm.x + xn.x, xm.y - xn.y);
        const float2 d = make_float2(xm.x - xn.x, xm.y + xn.y);
        const float2 tw = __ldg(T + m);
        const float2 q = cmul(make_float2(tw.x, -tw.y), d);
        A[a * n + m] = make_float2(e.x - q.y, -(e.y + q.x));
    }
    __syncthreads();
    const float2 *Rz = block_fft(A, B, n, nb, T, p);
    const float sc = 1.0f / (float)(2 * n);
    for (int w = tid; w < nb * n; w += bd) {
        const int a = w / n, k = w - a * n;
        const float2 r = Rz[a * n + k];
        reinterpret_cast<__half2 *>(out + (row0 + a) * (long)(2 * n))[k] = __floats2half2_rn(r.x * sc, -r.y * sc);
    }
}

// ---------------------------------------------------------------- general direct path
// block per (ring h, bc): gather over the arcs of ring h (entries built with one ring per band)
__global__ void direct_kernel(const __half *__restrict__ x, const int4 *__restrict__ entries,
                              const int *__restrict__ boff, const int *__restrict__ seg, const int *__restrict__ segval,
                              const float *__restrict__ vals, __half *__restrict__ out, int Wi, int Ho, int Wo, int KH)
{
    extern __shared__ float dsm[];
    float *acc = dsm;
    float *xs = dsm + Wo;
    const int h = blockIdx.x;
    const long bc = blockIdx.y;
    const int tid = threadIdx.x, bd = blockDim.x;
    const int ps = Wo / Wi;
    for (int w = tid; w < Wo; w += bd) acc[w] = 0.f;
    const int e0 = boff[h], e1 = boff[h + 1];
    for (int e = e0; e < e1; ++e) {
        const int4 en = entries[e];
        const __half *xr = x + (bc * KH + en.w) * (long)Wi;
        __syncthreads();
        for (int q = tid; q < Wi; q += bd) xs[q] = __half2float(xr[q]);
        __syncthreads();
        for (int s = en.y; s < en.z; ++s) {
            int start = seg[3 * s + 1] % Wo;
            if (start < 0) start += Wo;
            const int len = seg[3 * s + 2];
            const float *v = vals + segval[s];
            for (int w = tid; w < Wo; w += bd) {
                const int d = w - start;
                int j0 = d % ps;
                if (j0 < 0) j0 += ps;
                int q = (int)((((long)d - j0) / ps % Wi + Wi) % Wi);
                float sum = 0.f;
                for (int j = j0; j < len; j += ps) {
                    sum = fmaf(v[j], xs[q], sum);
                    q = (q == 0) ? Wi - 1 : q - 1;
                }
                acc[w] += sum;
            }
        }
    }
    __syncthreads();
    __half *o = out + (bc * Ho + h) * (long)Wo;
    for (int w = tid; w < Wo; w += bd) o[w] = __float2half_rn(acc[w]);
}

// ---------------------------------------------------------------- prep
__global__ void prep_count_kernel(const int64_t *__restrict__ seg_off, const int *__restrict__ seg,
                                  const int64_t *__restrict__ val_off, int nrows, int Ho, int HB,
                                  int *__restrict__ segval, int *__restrict__ cnt, float2 *__restrict__ Ti, int Ni,
                                  float2 *__restrict__ To, int No)
{
    const int rb = (nrows + blockDim.x - 1) / blockDim.x;
    if ((int)blockIdx.x >= rb) {
        const int i = (blockIdx.x - rb) * blockDim.x + threadIdx.x;
        double s, c;
        if (i < Ni) {
            sincospi(-2.0 * (double)i / (double)Ni, &s, &c);
            Ti[i] = make_float2((float)c, (float)s);
        } else if (i - Ni < No) {
            sincospi(-2.0 * (double)(i - Ni) / (double)No, &s, &c);
            To[i - Ni] = make_float2((float)c, (float)s);
        }
        return;
    }
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrows) return;
    const int lo = (int)seg_off[r], hi = (int)seg_off[r + 1];
    int v = (int)val_off[r];
    int pband = -1, pring = 0;
    for (int s = lo; s < hi; ++s) {
        const int ring = seg[3 * s], len = seg[3 * s + 2];
        segval[s] = v;
        v += max(len, 0);
        if (ring < 0 || ring >= Ho || len <= 0) {
            pband = -1;
            continue;
        }
        const int band = ring / HB;
        if (band != pband || ring != pring + 1) atomicAdd(&cnt[band], 1);
        pband = band;
        pring = ring;
    }
}

__global__ void prep_scan_kernel(const int *__restrict__ cnt, int *__restrict__ boff, int *__restrict__ bcur, int nbands)
{
    __shared__ int ssum[1024];
    const int tid = threadIdx.x, bd = blockDim.x;
    const int per = (nbands + bd - 1) / bd;
    const int beg = min(tid * per, nbands), end = min(beg + per, nbands);
    int local = 0;
    for (int i = beg; i < end; ++i) local += cnt[i];
    ssum[tid] = local;
    __syncthreads();
    for (int off = 1; off < bd; off <<= 1) {
        int t = tid >= off ? ssum[tid - off] : 0;
        __syncthreads();
        ssum[tid] += t;
        __syncthreads();
    }
    int run = ssum[tid] - local;
    for (int i = beg; i < end; ++i) {
        boff[i] = run;
        bcur[i] = run;
        run += cnt[i];
    }
    if (tid == bd - 1) boff[nbands] = ssum[bd - 1];
}

// entry = (row, first arc, end arc, input row index kh): a maximal run of consecutive arcs of one row
// inside one band on consecutive rings
__global__ void prep_fill_kernel(const int64_t *__restrict__ seg_off, const int *__restrict__ seg,
                                 const int *__restrict__ row_ker, const int *__restrict__ row_lat, int nrows, int Hi,
                                 int Ho, int HB, int *__restrict__ bcur, int4 *__restrict__ entries)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrows) return;
    const int lo = (int)seg_off[r], hi = (int)seg_off[r + 1];
    int s = lo;
    while (s < hi) {
        const int ring = seg[3 * s], len = seg[3 * s + 2];
        if (ring < 0 || ring >= Ho || len <= 0) {
            ++s;
            continue;
        }
        const int band = ring / HB;
        int e = s + 1, pr = ring;
        while (e < hi) {
            const int rr = seg[3 * e], ll = seg[3 * e + 2];
            if (rr < 0 || rr >= Ho || ll <= 0 || rr / HB != band || rr != pr + 1) break;
            pr = rr;
            ++e;
        }
        const int pos = atomicAdd(&bcur[band], 1);
        entries[pos] = make_int4(r, s, e, row_ker[r] * Hi + row_lat[r]);
        s = e;
    }
}

#define SORTE 256
// sort each band's entries by row (deterministic order shared by all tiles of the band)
__global__ void prep_sort_kernel(int4 *__restrict__ entries, const int *__restrict__ boff, int nbands)
{
    const int band = blockIdx.x;
    if (band >= nbands) return;
    const int tid = threadIdx.x;
    __shared__ int4 s_ent[SORTE];
    const int e1 = boff[band + 1];
    for (int ec = boff[band]; ec < e1; ec += SORTE) {
        const int ne = min(SORTE, e1 - ec);
        s_ent[tid] = tid < ne ? entries[ec + tid] : make_int4(0x7fffffff, 0, 0, 0);
        __syncthreads();
        for (int kk = 2; kk <= SORTE; kk <<= 1) {
            for (int j = kk >> 1; j > 0; j >>= 1) {
                const int ixj = tid ^ j;
                if (ixj > tid) {
                    const int4 a = s_ent[tid], b = s_ent[ixj];
                    const bool up = (tid & kk) == 0;
                    if ((a.x > b.x) == up) {
                        s_ent[tid] = b;
                        s_ent[ixj] = a;
                    }
                }
                __syncthreads();
            }
        }
        if (tid < ne) entries[ec + tid] = s_ent[tid];
        __syncthreads();
    }
}

// arc spectrum type, from exact properties of the arc: an arc centred on longitude 0 (2*start + len - 1 = 0 mod Wo)
// whose values are exactly symmetric has a purely real spectrum (1), exactly antisymmetric a purely imaginary
// one (2); anything else is general (0). Eight lanes per arc.
__global__ void arc_type_kernel(const int *__restrict__ seg, const int *__restrict__ segval,
                                const float *__restrict__ vals, int *__restrict__ atype, int nseg, int Wo)
{
    const int s = blockIdx.x * (blockDim.x >> 3) + (threadIdx.x >> 3);
    const int lane = threadIdx.x & 7;
    const unsigned grp = 0xffu << (threadIdx.x & 24); // the 8 lanes of this arc
    if (s >= nseg) return;                            // whole 8-lane groups exit together
    const int start = seg[3 * s + 1], len = seg[3 * s + 2];
    long c2 = (2L * start + len - 1) % Wo;
    if (c2 < 0) c2 += Wo;
    bool sym = true, anti = true;
    if (len < 1 || len > Wo || c2 != 0) {
        sym = anti = false;
    } else {
        const float *v = vals + segval[s];
        for (int j = lane; j < (len + 1) / 2; j += 8) {
            const float a = __ldg(v + j), b = __ldg(v + len - 1 - j);
            sym &= (a == b);
            anti &= (a == -b);
        }
    }
    sym = __all_sync(grp, sym);
    anti = __all_sync(grp, anti);
    if (lane == 0) atype[s] = sym ? 1 : (anti ? 2 : 0);
}

__device__ __forceinline__ void mbar_init(uint32_t bar, int count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity)
{
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "WAIT_%=:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 "@!p bra WAIT_%=;\n"
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

// ---------------------------------------------------------------- banded contraction (pscale 1 / 2)
// grid (band, channel group, slot tile); blockDim = LC*MG. Every entry covers consecutive rings [klo, klo + cnt)
// of the band, so its arcs' spectra land at fixed ring slots of a CSTG-deep ring buffer filled by bulk copies.
// consumer (c, g): channel c of the group, slots mt*MB + g*MT + q (q < MT), rings band*HB + k (k < HB).
// PS == 1: slot j = output frequency j.  PS == 2: slot j feeds output j (P plane 0, X) and Wi - j
// (P plane 1, conj X).
#define MAXE 256
#define CSTG 10 // entry stages (even): five pairs
__device__ __forceinline__ void cp_async16(void *smem, const void *gmem)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(smem)), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

template <int PS, int HB, int MT, int MG>
__global__ void __launch_bounds__(288, 2)
    contract_kernel(const float2 *__restrict__ X, const float2 *__restrict__ P, const int4 *__restrict__ entries,
                    const int *__restrict__ boff, const int *__restrict__ seg, const int *__restrict__ atype,
                    float2 *__restrict__ OUT, int LC, int KH, int MPi, int nseg, int Ho, int Mf, int Wi, int BCp,
                    int nslot)
{
    constexpr int NO = (PS == 2) ? 2 : 1;
    constexpr int MB = MG * MT;
    constexpr int PA = NO * MB; // float2 per arc in a slot tile
    const int NC = LC * MG;     // threads; the X tile is exactly NC * MT / 2 16-byte chunks
    const int tid = threadIdx.x;
    const int band = blockIdx.x, cg = blockIdx.y, mt = blockIdx.z;
    const int h0 = band * HB;
    const int XT = MB * LC;
    const int STG = XT + HB * PA;
    extern __shared__ __align__(128) float2 smem_c[];
    int4 *s_meta = reinterpret_cast<int4 *>(smem_c + (size_t)CSTG * STG); // (row, first arc, klo | type << 8, cnt)
    uint64_t *s_full = reinterpret_cast<uint64_t *>(s_meta + MAXE);       // one transaction barrier per stage

    const int c = tid % LC, g = tid / LC;
    const int xo = g * MT * LC + c;
    float2 acc[HB][MT][NO];
#pragma unroll
    for (int k = 0; k < HB; ++k)
#pragma unroll
        for (int q = 0; q < MT; ++q)
#pragma unroll
            for (int o = 0; o < NO; ++o) acc[k][q][o] = make_float2(0.f, 0.f);

    const float2 *Xg = X + ((size_t)cg * KH * MPi + (size_t)mt * MB) * LC;
    const size_t xrow = (size_t)MPi * LC;
    const float2 *Pt = P + (size_t)mt * nseg * PA;
    const uint32_t sbase = smem_u32(smem_c), bar0 = smem_u32(s_full);
    const uint32_t sstg = (uint32_t)STG * sizeof(float2);
    if (tid == 0) {
        for (int d = 0; d < CSTG; ++d) mbar_init(bar0 + 8 * d, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    // thread 0 stages the X tile of entry i and the spectra of its arcs (at their ring slots) with two bulk
    // copies completing on the stage's transaction barrier
    auto issue = [&](int i, int sidx) {
        const int4 me = s_meta[i];
        const uint32_t sd = sbase + (uint32_t)sidx * sstg, bar = bar0 + 8 * sidx;
        const uint32_t pbytes = (uint32_t)(me.w * PA * sizeof(float2));
        mbar_expect_tx(bar, (uint32_t)(XT * sizeof(float2)) + pbytes);
        bulk_g2s(sd, Xg + (size_t)me.x * xrow, (uint32_t)(XT * sizeof(float2)), bar);
        bulk_g2s(sd + (uint32_t)(XT + (me.z & 255) * PA) * sizeof(float2), Pt + (size_t)me.y * PA, pbytes, bar);
    };
    uint32_t phase = 0; // parity bit per stage

    const int e0 = boff[band], e1 = boff[band + 1];
    for (int ec = e0; ec < e1; ec += MAXE) {
        const int ne = min(MAXE, e1 - ec);
        __syncthreads();
        for (int i = tid; i < ne; i += NC) {
            const int4 en = entries[ec + i];
            int ty = 3;
            for (int a = en.y; a < en.z; ++a) ty &= atype[a];
            s_meta[i] = make_int4(en.w, en.y, (seg[3 * en.y] - h0) | (ty << 8), en.z - en.y);
        }
        __syncthreads();
        // entries move in pairs: CSTG / 2 pair-stages, one block barrier per pair
        if (tid == 0) {
            for (int i = 0; i < min(ne, CSTG - 2); ++i) issue(i, i);
        }
        int sc = 0, si = CSTG - 2;
        for (int i = 0; i < ne; i += 2) {
            mbar_wait(bar0 + 8 * sc, (phase >> sc) & 1u);
            phase ^= 1u << sc;
            if (i + 1 < ne) {
                mbar_wait(bar0 + 8 * (sc + 1), (phase >> (sc + 1)) & 1u);
                phase ^= 1u << (sc + 1);
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncthreads(); // pair (i, i+1) is visible to all; every thread is done with the previous pair (stages si, si+1)
            if (tid == 0) {
                const int b = i + CSTG - 2;
                if (b < ne) issue(b, si);
                if (b + 1 < ne) issue(b + 1, si + 1);
            }
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                if (e == 1 && i + 1 >= ne) break;
                const int4 me = s_meta[i + e];
                const int sb = (sc + e) * STG;
                const int ty = me.z >> 8;
                const unsigned rings = ((1u << me.w) - 1u) << (me.z & 255);
                float2 xv[MT];
#pragma unroll
                for (int q = 0; q < MT; ++q) xv[q] = smem_c[sb + xo + q * LC];
                const int pb0 = sb + XT + g * MT;
                // acc0 += pa * x = pa.x * x + pa.y * (i x);  acc1 accumulates conj(pb * conj(x)) = conj(pb) * x.
                // Real (ty 1) / imaginary (ty 2) spectra need one of the two packed FMAs.
                auto ring_loop = [&](auto TY) {
                    constexpr int T = decltype(TY)::value;
#pragma unroll
                    for (int k = 0; k < HB; ++k) {
                        if (rings & (1u << k)) {
                            float2 pa[MT], pb[MT];
#pragma unroll
                            for (int q = 0; q < MT; q += 2) {
                                const float4 t4 = *reinterpret_cast<const float4 *>(&smem_c[pb0 + k * PA + q]);
                                pa[q] = make_float2(t4.x, t4.y);
                                pa[q + 1] = make_float2(t4.z, t4.w);
                                if constexpr (NO == 2) {
                                    const float4 u4 = *reinterpret_cast<const float4 *>(&smem_c[pb0 + k * PA + MB + q]);
                                    pb[q] = make_float2(u4.x, u4.y);
                                    pb[q + 1] = make_float2(u4.z, u4.w);
                                }
                            }
#pragma unroll
                            for (int q = 0; q < MT; ++q) {
                                const float2 ix = make_float2(-xv[q].y, xv[q].x);
                                if constexpr (T != 2)
                                    acc[k][q][0] = fma2(make_float2(pa[q].x, pa[q].x), xv[q], acc[k][q][0]);
                                if constexpr (T != 1)
                                    acc[k][q][0] = fma2(make_float2(pa[q].y, pa[q].y), ix, acc[k][q][0]);
                                if constexpr (NO == 2) {
                                    if constexpr (T != 2)
                                        acc[k][q][NO - 1] = fma2(make_float2(pb[q].x, pb[q].x), xv[q], acc[k][q][NO - 1]);
                                    if constexpr (T != 1)
                                        acc[k][q][NO - 1] = fma2(make_float2(-pb[q].y, -pb[q].y), ix, acc[k][q][NO - 1]);
                                }
                            }
                        }
                    }
                };
                if (ty == 1)
                    ring_loop(IC<1> {});
                else if (ty == 2)
                    ring_loop(IC<2> {});
                else
                    ring_loop(IC<0> {});
            }
            sc = (sc == CSTG - 2) ? 0 : sc + 2;
            si = (si == CSTG - 2) ? 0 : si + 2;
        }
    }
    const size_t cbase = (size_t)cg * LC + c;
#pragma unroll
    for (int k = 0; k < HB; ++k) {
        const int h = h0 + k;
        if (h < Ho) {
#pragma unroll
            for (int q = 0; q < MT; ++q) {
                const int j = mt * MB + g * MT + q;
                if (j < nslot) {
                    OUT[((size_t)h * Mf + j) * BCp + cbase] = acc[k][q][0];
                    if constexpr (NO == 2) {
                        if (Wi - j != j)
                            OUT[((size_t)h * Mf + (Wi - j)) * BCp + cbase]
                                = make_float2(acc[k][q][NO - 1].x, -acc[k][q][NO - 1].y);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------- generic contraction (any pscale)
// thread = (output frequency m, channel cl); X index m mod Wi (conjugated above Wi/2); P[s*PLD + m]
template <int HB>
__global__ void __launch_bounds__(256)
    contract_gen_kernel(const float2 *__restrict__ X, const float2 *__restrict__ P, const int4 *__restrict__ entries,
                        const int *__restrict__ boff, const int *__restrict__ seg, float2 *__restrict__ OUT, int LC,
                        int KH, int MPi, int PLD, int Ho, int Mf, int Wi, int BCp)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int band = blockIdx.y, cg = blockIdx.z;
    const int h0 = band * HB;
    if (idx >= Mf * LC) return;
    const int m = idx / LC, c = idx - m * LC;
    int xm = m % Wi;
    const bool cj = xm > (Wi >> 1);
    if (cj) xm = Wi - xm;
    float2 acc[HB];
#pragma unroll
    for (int k = 0; k < HB; ++k) acc[k] = make_float2(0.f, 0.f);
    const int e0 = boff[band], e1 = boff[band + 1];
    for (int e = e0; e < e1; ++e) {
        const int4 en = entries[e];
        float2 xv = __ldg(X + (((size_t)cg * KH + en.w) * MPi + xm) * LC + c);
        if (cj) xv.y = -xv.y;
        for (int s = en.y; s < en.z; ++s) {
            const int kk = seg[3 * s] - h0;
            const float2 pv = __ldg(P + (size_t)s * PLD + m);
            const float2 pr = cmul(pv, xv);
#pragma unroll
            for (int k = 0; k < HB; ++k)
                if (k == kk) acc[k] = cadd(acc[k], pr);
        }
    }
#pragma unroll
    for (int k = 0; k < HB; ++k) {
        const int h = h0 + k;
        if (h < Ho) OUT[((size_t)h * Mf + m) * BCp + (size_t)cg * LC + c] = acc[k];
    }
}

// ---------------------------------------------------------------- host
static bool make_plan2(int n, Plan2 &p)
{
    int best = 1 << 30;
    bool ok = false;
    for (int a : kRadix) {
        if (n % a) continue;
        const int b = n / a;
        bool inset = false;
        for (int r : kRadix) inset |= (r == b);
        if (!inset) continue;
        const int cost = max(a, b) * 64 + a; // balanced, then smaller first radix
        if (cost < best) {
            best = cost;
            p.N1 = a;
            p.N2 = b;
            ok = true;
        }
    }
    if (!ok) return false;
    p.n = n;
    p.LD = p.N2 | 1;
    int srow = max(p.N1 * p.LD, n + 1);
    while (srow % 4 != 2) ++srow;
    p.SROW = srow;
    p.NT = (8 * max(p.N1, p.N2) + 31) / 32 * 32;
    return true;
}

static bool make_planm(int n, PlanM &p)
{
    p.n = n;
    p.nst = 0;
    int m = n;
    const int rad[5] = {8, 4, 2, 5, 3};
    for (int ri = 0; ri < 5; ++ri) {
        while (m % rad[ri] == 0) {
            if (p.nst >= MAXST) return false;
            p.r[p.nst++] = rad[ri];
            m /= rad[ri];
        }
    }
    return m == 1;
}

static inline size_t align_up(size_t v) { return (v + 255) & ~(size_t)255; }

template <int PS, int HB, int MT, int MG>
static void launch_contract(dim3 grid, int threads, size_t smem, cudaStream_t st, const float2 *X, const float2 *P,
                            const int4 *ent, const int *boff, const int *seg, const int *aty, float2 *OUT, int LC,
                            int KH, int MPi, int nseg, int Ho, int Mf, int Wi, int BCp, int nslot)
{
    cudaFuncSetAttribute(contract_kernel<PS, HB, MT, MG>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    contract_kernel<PS, HB, MT, MG>
        <<<grid, threads, smem, st>>>(X, P, ent, boff, seg, aty, OUT, LC, KH, MPi, nseg, Ho, Mf, Wi, BCp, nslot);
}

template <int PS, int HB, int MT>
static void dispatch_contract(int MG, dim3 grid, int threads, size_t smem, cudaStream_t st, const float2 *X,
                              const float2 *P, const int4 *ent, const int *boff, const int *seg, const int *aty,
                              float2 *OUT, int LC, int KH, int MPi, int nseg, int Ho, int Mf, int Wi, int BCp, int nslot)
{
    switch (MG) {
    case 4:
        launch_contract<PS, HB, MT, 4>(grid, threads, smem, st, X, P, ent, boff, seg, aty, OUT, LC, KH, MPi, nseg, Ho,
                                       Mf, Wi, BCp, nslot);
        break;
    case 8:
        launch_contract<PS, HB, MT, 8>(grid, threads, smem, st, X, P, ent, boff, seg, aty, OUT, LC, KH, MPi, nseg, Ho,
                                       Mf, Wi, BCp, nslot);
        break;
    case 16:
        launch_contract<PS, HB, MT, 16>(grid, threads, smem, st, X, P, ent, boff, seg, aty, OUT, LC, KH, MPi, nseg, Ho,
                                        Mf, Wi, BCp, nslot);
        break;
    default:
        launch_contract<PS, HB, MT, 32>(grid, threads, smem, st, X, P, ent, boff, seg, aty, OUT, LC, KH, MPi, nseg, Ho,
                                        Mf, Wi, BCp, nslot);
        break;
    }
}

void Run(ffi::TensorView inp, ffi::TensorView row_ker, ffi::TensorView row_lat, ffi::TensorView seg_off,
         ffi::TensorView seg, ffi::TensorView val_off, ffi::TensorView vals, int64_t nlat_out, int64_t nlon_out,
         ffi::TensorView out)
{
    const int B = (int)inp.size(0), C = (int)inp.size(1), K = (int)inp.size(2), Hi = (int)inp.size(3),
              Wi = (int)inp.size(4);
    const int nrows = (int)row_ker.size(0);
    const int nseg = (int)seg.size(0);
    const int Ho = (int)nlat_out, Wo = (int)nlon_out;
    const int BC = B * C;
    const int KH = K * Hi;
    DLDevice dev = inp.device();
    cudaStream_t st = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));

    const __half *xp = static_cast<const __half *>(inp.data_ptr());
    const int *rk = static_cast<const int *>(row_ker.data_ptr());
    const int *rl = static_cast<const int *>(row_lat.data_ptr());
    const int64_t *so = static_cast<const int64_t *>(seg_off.data_ptr());
    const int *sg = static_cast<const int *>(seg.data_ptr());
    const int64_t *vo = static_cast<const int64_t *>(val_off.data_ptr());
    const float *vp = static_cast<const float *>(vals.data_ptr());
    __half *op = static_cast<__half *>(out.data_ptr());
    if (BC <= 0 || Ho <= 0 || Wo <= 0) return;

    const int ps = (Wi > 0) ? Wo / Wi : 0;
    const int ni = Wi / 2, no = Wo / 2;
    Plan2 pi2 {}, po2 {};
    PlanM pim {}, pom {};
    const bool even = Wi > 0 && (Wi % 2 == 0) && (Wo % 2 == 0) && ps >= 1 && ps * Wi == Wo;
    const bool i2 = even && make_plan2(ni, pi2), o2 = even && make_plan2(no, po2);
    const bool im = even && !i2 && make_planm(ni, pim), om = even && !o2 && make_planm(no, pom);
    const bool fft = even && (i2 || im) && (o2 || om) && no <= 16384;
    const int Mi = ni + 1, Mf = no + 1;

    // channel groups: LC channels per group (a multiple of 4 for a single group, of 8 otherwise), minimal padding
    const int BCp8 = (BC + 7) / 8 * 8;
    int LC = (BCp8 - BC) * 5 > BC ? (BC + 3) / 4 * 4 : BCp8, NCG = 1; // multiple of 4 only if 8 pads > 20%
    if (BCp8 > 72) {
        int bestp = 1 << 30;
        for (int g = (BCp8 + 63) / 64; g <= (BCp8 + 31) / 32; ++g) {
            const int lc = ((BC + g - 1) / g + 7) / 8 * 8;
            if (lc * g < bestp) {
                bestp = lc * g;
                LC = lc;
                NCG = g;
            }
        }
    }
    const int BCp = LC * NCG;
    const bool fast = fft && (ps == 1 || ps == 2);
    const int HB = fft ? 8 : 1;
    const int nbands = (Ho + HB - 1) / HB;
    const int MT = (ps == 1) ? 4 : 2;
    int MGf = 32; // slot groups per contraction block: largest of 32/16/8/4 keeping LC * MG <= 288
    while (MGf > 4 && LC * MGf > 288) MGf >>= 1;
    const int MB = (fast ? MGf : 1) * MT;
    const int NO = (fast && ps == 2) ? 2 : 1;
    const int nslot = fast ? (ps == 2 ? Wi / 2 + 1 : Mf) : Mf;
    const int NMT = fast ? (nslot + MB - 1) / MB : 1;
    const int MPi = fast ? NMT * MB : Mi;
    const int PLD = fast ? NMT * MB : (Mf + 1) / 2 * 2;

    size_t off = 0;
    const size_t o_segval = off;
    off = align_up(off + sizeof(int) * (size_t)max(nseg, 1));
    const size_t o_cnt = off;
    off = align_up(off + sizeof(int) * (size_t)(nbands + 1));
    const size_t o_boff = off;
    off = align_up(off + sizeof(int) * (size_t)(nbands + 1));
    const size_t o_bcur = off;
    off = align_up(off + sizeof(int) * (size_t)(nbands + 1));
    const size_t o_ent = off;
    off = align_up(off + sizeof(int4) * (size_t)max(nseg, 1));
    const size_t o_aty = off;
    off = align_up(off + sizeof(int) * (size_t)max(nseg, 1));
    size_t o_ti = 0, o_to = 0, o_P = 0, o_X = 0, o_OUT = 0;
    if (fft) {
        o_ti = off;
        off = align_up(off + sizeof(float2) * (size_t)Wi);
        o_to = off;
        off = align_up(off + sizeof(float2) * (size_t)Wo);
        o_P = off;
        off = align_up(off + sizeof(float2) * (size_t)max(nseg, 1) * NO * PLD);
        o_X = off;
        off = align_up(off + sizeof(float2) * (size_t)NCG * KH * MPi * LC);
        o_OUT = off;
        off = align_up(off + sizeof(float2) * (size_t)Ho * Mf * BCp);
    }
    ffi::Tensor ws = ffi::Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, ffi::ShapeView({(int64_t)off}),
                                               DLDataType {kDLUInt, 8, 1}, dev);
    char *base = static_cast<char *>(ws.data_ptr());
    int *segval = reinterpret_cast<int *>(base + o_segval);
    int *cnt = reinterpret_cast<int *>(base + o_cnt);
    int *boff = reinterpret_cast<int *>(base + o_boff);
    int *bcur = reinterpret_cast<int *>(base + o_bcur);
    int4 *ent = reinterpret_cast<int4 *>(base + o_ent);
    int *aty = reinterpret_cast<int *>(base + o_aty);
    float2 *Ti = fft ? reinterpret_cast<float2 *>(base + o_ti) : nullptr;
    float2 *To = fft ? reinterpret_cast<float2 *>(base + o_to) : nullptr;

    cudaMemsetAsync(cnt, 0, sizeof(int) * (size_t)(nbands + 1), st);
    {
        // one thread per psi row, in small blocks so the row loops spread over many SMs
        const int rb = (nrows + 63) / 64;
        const int tb = fft ? (Wi + Wo + 63) / 64 : 0;
        if (rb + tb > 0)
            prep_count_kernel<<<rb + tb, 64, 0, st>>>(so, sg, vo, nrows, Ho, HB, segval, cnt, Ti, fft ? Wi : 0, To,
                                                      fft ? Wo : 0);
        prep_scan_kernel<<<1, 1024, 0, st>>>(cnt, boff, bcur, nbands);
        if (rb > 0) prep_fill_kernel<<<rb, 64, 0, st>>>(so, sg, rk, rl, nrows, Hi, Ho, HB, bcur, ent);
        if (fast && nbands > 0) prep_sort_kernel<<<nbands, SORTE, 0, st>>>(ent, boff, nbands);
        if (fast && nseg > 0) arc_type_kernel<<<(nseg + 31) / 32, 256, 0, st>>>(sg, segval, vp, aty, nseg, Wo);
    }

    if (!fft) {
        const size_t smem = sizeof(float) * (size_t)(Wo + max(Wi, 1));
        if (smem > 48 * 1024)
            cudaFuncSetAttribute(direct_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        direct_kernel<<<dim3(Ho, BC), 256, smem, st>>>(xp, ent, boff, sg, segval, vp, op, Wi, Ho, Wo, KH);
        return;
    }

    float2 *P = reinterpret_cast<float2 *>(base + o_P);
    float2 *X = reinterpret_cast<float2 *>(base + o_X);
    float2 *OUT = reinterpret_cast<float2 *>(base + o_OUT);
    auto pick_nb = [](int n) { return n <= 256 ? 16 : (n <= 512 ? 8 : (n <= 1024 ? 4 : (n <= 2048 ? 2 : 1))); };

    // psi arc spectra
    if (nseg > 0) {
        if (o2) {
            const size_t smem = sizeof(float2) * 8 * (size_t)po2.SROW;
            cudaFuncSetAttribute(fwdpsi2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            fwdpsi2_kernel<<<(nseg + 7) / 8, dim3(max(po2.N1, po2.N2), 8), smem, st>>>(
                sg, segval, vp, P, nseg, po2, To, Wi, NO, nslot, PLD, fast ? MB : PLD,
                fast ? __builtin_ctz((unsigned)MB) : -1);
        } else {
            const int NB = pick_nb(no);
            const size_t smem = sizeof(float2) * 2 * (size_t)NB * no;
            cudaFuncSetAttribute(fwdpsim_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            fwdpsim_kernel<<<(nseg + NB - 1) / NB, 256, smem, st>>>(sg, segval, vp, P, nseg, pom, To, NB, Wi, NO, nslot,
                                                                    PLD, fast ? MB : PLD);
        }
    }
    // input spectra
    if (i2) {
        const size_t smem = sizeof(float2) * 8 * (size_t)pi2.SROW;
        cudaFuncSetAttribute(fwdx2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        fwdx2_kernel<<<dim3((LC + 7) / 8, KH, NCG), dim3(max(pi2.N1, pi2.N2), 8), smem, st>>>(xp, X, BC, KH, pi2, Ti,
                                                                                              LC, MPi);
    } else {
        const int NB = pick_nb(ni);
        const size_t smem = sizeof(float2) * 2 * (size_t)NB * ni;
        cudaFuncSetAttribute(fwdxm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        const long nxr = (long)BCp * KH;
        fwdxm_kernel<<<(unsigned)((nxr + NB - 1) / NB), 256, smem, st>>>(xp, X, BC, BCp, KH, pim, Ti, NB, LC, MPi);
    }
    // contraction
    if (fast) {
        const int threads = LC * MGf;
        const int XT = MB * LC, PT = HB * NO * MB;
        const size_t smem = sizeof(float2) * (size_t)CSTG * (XT + PT) + sizeof(int4) * (size_t)MAXE + 8 * CSTG;
        dim3 grid(nbands, NCG, NMT);
        if (ps == 1)
            dispatch_contract<1, 8, 4>(MGf, grid, threads, smem, st, X, P, ent, boff, sg, aty, OUT, LC, KH, MPi, nseg,
                                       Ho, Mf, Wi, BCp, nslot);
        else
            dispatch_contract<2, 8, 2>(MGf, grid, threads, smem, st, X, P, ent, boff, sg, aty, OUT, LC, KH, MPi, nseg,
                                       Ho, Mf, Wi, BCp, nslot);
    } else {
        dim3 grid((Mf * LC + 255) / 256, nbands, NCG);
        contract_gen_kernel<8><<<grid, 256, 0, st>>>(X, P, ent, boff, sg, OUT, LC, KH, MPi, PLD, Ho, Mf, Wi, BCp);
    }
    // inverse
    if (o2) {
        const size_t smem = sizeof(float2) * 8 * (size_t)po2.SROW;
        cudaFuncSetAttribute(inv2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        inv2_kernel<<<dim3(Ho, (BCp + 7) / 8), dim3(max(po2.N1, po2.N2), 8), smem, st>>>(OUT, op, BC, BCp, Ho, po2, To);
    } else {
        const int NB = pick_nb(no);
        const size_t smem = sizeof(float2) * 2 * (size_t)NB * no;
        cudaFuncSetAttribute(invm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        const long nout = (long)BC * Ho;
        invm_kernel<<<(unsigned)((nout + NB - 1) / NB), 256, smem, st>>>(OUT, op, BC, BCp, Ho, pom, To, NB);
    }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
