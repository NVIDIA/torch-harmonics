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
// The DISCO arc scatter in frequency space, for sm_100a (B200 / GB200), float16 activations, fp32 psi.
//
// Provenance: Kernel Factory campaign disco-bwd-b200-v2 (dpar3kk2t54g96gx2vt1qpdckg), delivered candidate 8e15cd9c
// (14.93x the shipped arc scatter, geometric mean over its four workloads). The device code below is the candidate's
// as delivered (disco_quartet_merge.cu, with radix_twiddles_a2.inc), up to formatting and one change in
// contract_kernel marked "torch-harmonics" (a generic lambda the CUDA 13.3 front end crashes on); the host side at the
// end of the file is torch-harmonics' own, see the comment there. The campaign's final review left two findings open:
// fp32 FFT round-off on outputs that should be exactly zero, and no check that the output longitudes are a multiple of
// the input longitudes (the host side checks it now).
//
// Needs Blackwell (packed fp32x2 arithmetic, bulk copies, programmatic dependent launch); part of the arch-specific
// extension torch_harmonics.disco._C_sm100 (see _DISCO_ARCH_EXTENSIONS in setup.py), and selected on compute
// capability 10.0 devices only, the one it was validated on.
//
// The candidate's description:
//
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
// Contraction: block = (band of HB rings, channel group, slot tile); a producer warp streams the X tile of
//   every row in the band plus the psi spectra of its arcs with bulk copies into a CSTG-stage ring (full /
//   empty mbarriers, no block barrier); consumer threads own (2 channels, 2 slots or planes, HB rings) fp32
//   accumulators. Arcs exactly centred on longitude 0 with exactly (anti)symmetric values have real
//   (imaginary) spectra, which halves the MACs.
// The kernels convert between __half and float implicitly, which torch's build flags disable for every source of
// an extension (-D__CUDA_NO_HALF_*); re-enabled for this file only, before cuda_fp16.h is read.
#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <limits.h>

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

__device__ __forceinline__ void bulk_g2s_l2_last(uint32_t dst, const void *src, uint32_t bytes, uint32_t bar,
                                                 uint64_t policy)
{
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
                 "[%0], [%1], %2, [%3], %4;" ::"r"(dst),
                 "l"(src), "r"(bytes), "r"(bar), "l"(policy)
                 : "memory");
}

__device__ __forceinline__ void bulk_s2g(void *dst, uint32_t src, uint32_t bytes)
{
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst), "r"(src), "r"(bytes)
                 : "memory");
}
__device__ __forceinline__ void bulk_commit_wait_read()
{
    asm volatile("cp.async.bulk.commit_group;\n cp.async.bulk.wait_group.read 0;" ::: "memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

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
    float2 m1 = fma2(make_float2(-0.5f, -0.5f), t1, v[0]);
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
    float2 b1r = fma2(make_float2(C2, C2), t2, fma2(make_float2(C1, C1), t1, a0));
    float2 b2r = fma2(make_float2(C1, C1), t2, fma2(make_float2(C2, C2), t1, a0));
    float2 s1 = fma2(make_float2(S2, S2), t4, mul2(make_float2(S1, S1), t3));
    float2 s2 = fma2(make_float2(-S1, -S1), t4, mul2(make_float2(S2, S2), t3));
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

#include "disco_cuda_bwd_fft_sm100_twiddles.inc"

__host__ __device__ constexpr int ctf(int N) { return N % 4 == 0 ? 4 : (N % 3 == 0 ? 3 : (N % 2 == 0 ? 2 : 5)); }

template <int N> __device__ __forceinline__ void dftn(float2 *v);

// Prime-factor (Good-Thomas) DFT for coprime A * B: the CRT input / output index maps make both factor passes
// twiddle-free. BINVA = B^-1 mod A, AINVB = A^-1 mod B.
template <int A, int B, int BINVA, int AINVB> __device__ __forceinline__ void dft_pfa(float2 *v)
{
    constexpr int N = A * B;
    float2 t[A][B];
#pragma unroll
    for (int n1 = 0; n1 < A; ++n1)
#pragma unroll
        for (int n2 = 0; n2 < B; ++n2) t[n1][n2] = v[(n1 * B + n2 * A) % N];
#pragma unroll
    for (int n2 = 0; n2 < B; ++n2) {
        float2 u[A];
#pragma unroll
        for (int n1 = 0; n1 < A; ++n1) u[n1] = t[n1][n2];
        dftn<A>(u);
#pragma unroll
        for (int k1 = 0; k1 < A; ++k1) t[k1][n2] = u[k1];
    }
#pragma unroll
    for (int k1 = 0; k1 < A; ++k1) dftn<B>(t[k1]);
#pragma unroll
    for (int k1 = 0; k1 < A; ++k1)
#pragma unroll
        for (int k2 = 0; k2 < B; ++k2) v[(k1 * B * BINVA + k2 * A * AINVB) % N] = t[k1][k2];
}

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
        for (int k = 1; k < M; ++k) {
            const int i = rtw_off(N) + f * k; // a * w = a.x * w + a.y * (i w)
            sub[f][k] = fma2(make_float2(sub[f][k].y, sub[f][k].y), c_rtwr[i],
                             mul2(make_float2(sub[f][k].x, sub[f][k].x), c_rtw[i]));
        }
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
    } else if constexpr (N == 6) {
        dft_pfa<2, 3, 1, 2>(v);
    } else if constexpr (N == 18) {
        dft_pfa<2, 9, 1, 5>(v);
    } else if constexpr (N == 20) {
        dft_pfa<4, 5, 1, 4>(v);
    } else if constexpr (N == 24) {
        dft_pfa<3, 8, 2, 3>(v);
    } else if constexpr (N == 30) {
        dft_pfa<5, 6, 1, 5>(v);
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

template <class Fn> __device__ __forceinline__ void with_radix20(int R, Fn &&fn)
{
    switch (R) {
#define RADIX20_CASE(V)                                                                                                \
    case V: fn(IC<V> {}); break;
        RADIX20_CASE(2)
        RADIX20_CASE(3)
        RADIX20_CASE(4)
        RADIX20_CASE(5)
        RADIX20_CASE(6)
        RADIX20_CASE(8)
        RADIX20_CASE(9)
        RADIX20_CASE(10)
        RADIX20_CASE(12)
        RADIX20_CASE(15)
        RADIX20_CASE(16)
        RADIX20_CASE(18)
        RADIX20_CASE(20)
#undef RADIX20_CASE
    default: break;
    }
}

template <class Fn> __device__ __forceinline__ void with_radix30(int R, Fn &&fn)
{
    switch (R) {
#define RADIX30_CASE(V)                                                                                                \
    case V: fn(IC<V> {}); break;
        RADIX30_CASE(2)
        RADIX30_CASE(3)
        RADIX30_CASE(4)
        RADIX30_CASE(5)
        RADIX30_CASE(6)
        RADIX30_CASE(8)
        RADIX30_CASE(9)
        RADIX30_CASE(10)
        RADIX30_CASE(12)
        RADIX30_CASE(15)
        RADIX30_CASE(16)
        RADIX30_CASE(18)
        RADIX30_CASE(20)
        RADIX30_CASE(24)
        RADIX30_CASE(25)
        RADIX30_CASE(27)
        RADIX30_CASE(30)
#undef RADIX30_CASE
    default: break;
    }
}

// Narrow runtime-plan tiers keep unrelated radix bodies out of hot FFT kernels.
#define TIER_S 120
#define TIER_L 130
#define TIER_M 110
#define TIER_N 125
__host__ __device__ constexpr bool in_tier_m(int r) { return r == 18 || r == 20; }
__host__ __device__ constexpr bool in_tier_n(int r) { return r == 24 || r == 30; }
__host__ __device__ constexpr bool in_tier_s(int r) { return r == 16 || r == 18 || r == 20; }
__host__ __device__ constexpr bool in_tier_l(int r) { return r == 24 || r == 25 || r == 27 || r == 30; }

template <int CAP, class Fn> __device__ __forceinline__ void with_radix_cap(int R, Fn &&fn)
{
    if constexpr (CAP == TIER_M) {
        switch (R) {
        case 18: fn(IC<18> {}); break;
        case 20: fn(IC<20> {}); break;
        default: break;
        }
    } else if constexpr (CAP == TIER_S) {
        switch (R) {
        case 16: fn(IC<16> {}); break;
        case 18: fn(IC<18> {}); break;
        case 20: fn(IC<20> {}); break;
        default: break;
        }
    } else if constexpr (CAP == TIER_N) {
        switch (R) {
        case 24: fn(IC<24> {}); break;
        case 30: fn(IC<30> {}); break;
        default: break;
        }
    } else if constexpr (CAP == TIER_L) {
        switch (R) {
        case 24: fn(IC<24> {}); break;
        case 25: fn(IC<25> {}); break;
        case 27: fn(IC<27> {}); break;
        case 30: fn(IC<30> {}); break;
        default: break;
        }
    } else if constexpr (CAP <= 20)
        with_radix20(R, static_cast<Fn &&>(fn));
    else if constexpr (CAP <= 30)
        with_radix30(R, static_cast<Fn &&>(fn));
    else
        with_radix(R, static_cast<Fn &&>(fn));
}

struct Plan2 {
    int n, N1, N2, LD, SROW, NT;
};

// pass A of the two-pass FFT (n = N1 * N2): thread column j2 gathers z[N2*j1 + j2] (j1 < N1), DFT_N1,
// twiddle W_n^{j2*k1}, store S[k1 * LD + j2]. T[t] = e^{-2 pi i t / (2n)}.
// sync: barrier between the gather and the store (the source aliases S).
// DFT_N1 of a gathered column, twiddle W_n^{j2*k1}, store S[k1 * LD + j2]. T[t] = e^{-2 pi i t / (2n)}.
// a * w given w and its rotation wr = i w (no operand shuffles in the packed products)
__device__ __forceinline__ float2 cmulr(float2 a, float2 w, float2 wr)
{
    return fma2(make_float2(a.y, a.y), wr, mul2(make_float2(a.x, a.x), w));
}
__device__ __forceinline__ float2 rot(float2 w) { return make_float2(-w.y, w.x); } // i w

template <int N1, class Store>
__device__ __forceinline__ void passA_dft_put(float2 *v, int j2, int n, const float2 *__restrict__ T, Store &&put)
{
    dftn<N1>(v);
    constexpr int NG = (N1 + 3) / 4;
    const float2 w1 = __ldg(T + 2 * j2);
    // W^(4a + b) applied as W^b then W^(4a), with every factor kept together with its rotation
    float2 wb[4], wbr[4];
    wb[1] = w1;
    wbr[1] = rot(w1);
    wb[2] = cmulr(w1, w1, wbr[1]);
    wbr[2] = rot(wb[2]);
    wb[3] = cmulr(wb[2], w1, wbr[1]);
    wbr[3] = rot(wb[3]);
    const float2 w4 = cmulr(wb[2], wb[2], wbr[2]);
    const float2 w4r = rot(w4);
    float2 wg = w4, wgr = w4r;
#pragma unroll
    for (int a = 0; a < NG; ++a) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int k = 4 * a + b;
            if (k < N1) {
                float2 t = v[k];
                if (b > 0) t = cmulr(t, wb[b], wbr[b]);
                if (a > 0) t = cmulr(t, wg, wgr);
                put(k, t);
            }
        }
        if (a > 0 && a + 1 < NG) {
            const float2 g = cmulr(wg, w4, w4r);
            wg = g;
            wgr = rot(g);
        }
    }
}

// Lower-register pass-A twiddle recurrence for the Tier-S input transform. Odd and even powers advance in
// independent W^2 chains, shortening the dependency path while keeping a single table load.
template <int N1, class Store>
__device__ __forceinline__ void passA_dft_put_twochain(float2 *v, int j2, const float2 *__restrict__ T, Store &&put)
{
    dftn<N1>(v);
    put(0, v[0]);
    if constexpr (N1 > 1) {
        float2 odd = __ldg(T + 2 * j2);     // W^1
        const float2 step = cmul(odd, odd); // W^2
        float2 even = step;                 // W^2
#pragma unroll
        for (int k = 1; k < N1; k += 2) {
            put(k, cmul(v[k], odd));
            if (k + 1 < N1) put(k + 1, cmul(v[k + 1], even));
            if (k + 2 < N1) {
                odd = cmul(odd, step);
                even = cmul(even, step);
            }
        }
    }
}

template <int N1>
__device__ __forceinline__ void passA_dft_store(float2 *v, int j2, int n, const float2 *__restrict__ T,
                                                float2 *__restrict__ S, int LD)
{
    passA_dft_put<N1>(v, j2, n, T, [&](int k, float2 t) { S[k * LD + j2] = t; });
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

// twice both halves (2 X[m], 2 X[n - m]) without the 1/2 scalings; input spectra are kept at this scale and the
// inverse transform compensates (exact power of two)
__device__ __forceinline__ void rpost2x2(float2 zm, float2 zn, float2 tw, float2 &xm, float2 &xnm)
{
    const float2 ze = make_float2(zm.x + zn.x, zm.y - zn.y);
    const float2 d = make_float2(zm.x - zn.x, zm.y + zn.y);
    const float2 tz = cmul(tw, make_float2(d.y, -d.x));
    xm = cadd(ze, tz);
    xnm = make_float2(ze.x - tz.x, tz.y - ze.y);
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

// ---------------------------------------------------------------- spectra layouts
// X (input spectra): per (channel group cg, input row kh) a block of MPi * LC float2, split into tiles of MB slots;
// inside a tile channels are octet-major (8 channels of one slot contiguous; the last octet may hold 4), and the slot
// index is XOR-swizzled by octet parity when MB is even, so one slot read across octets is bank-conflict free.
__host__ __device__ __forceinline__ size_t xoff(int m, int c, int MB, int LC)
{
    const int tl = m / MB, s = m - tl * MB, o = c >> 3, w = min(8, LC - 8 * o);
    const int sw = (MB & 1) ? 0 : (o & 1);
    return (size_t)tl * MB * LC + (size_t)8 * o * MB + (size_t)(s ^ sw) * w + (c & 7);
}
// P (arc spectra) in the tiled fast layout: per (slot tile, arc) MB * NO complex values in virtual-slot order
// u = slot * NO + plane; each pair of virtual slots is stored (re u, re u+1, im u, im u+1) so arcs with purely real
// (imaginary) spectra are read with one 8-byte load. pfoff = float offset of re(u); im(u) is 2 floats later.
__host__ __device__ __forceinline__ int pfoff(int u) { return 4 * (u >> 1) + (u & 1); }
// OUT (output spectra): per output ring h, channel octets of Mf * w float2 (frequency-major, channel fastest).
__host__ __device__ __forceinline__ size_t oidx(int h, int j, int bc, int Mf, int BCp)
{
    const int o8 = bc & ~7, w = min(8, BCp - o8);
    return ((size_t)h * BCp + o8) * Mf + (size_t)j * w + (bc & 7);
}
// staged fp16 row stride (32-bit words): >= n, a multiple of 4 with an odd quotient, so 8 rows x 4 consecutive words
// hit distinct banks and rows stay 16-byte aligned for bulk copies
static inline int stage_words(int n)
{
    int q = (n + 3) / 4;
    if (!(q & 1)) ++q;
    return 4 * q;
}

// ---------------------------------------------------------------- input spectra (two-pass)
// Block = one channel octet of group cg x input row kh. Thread t owns channel row r = t & 7 and FFT column t >> 3, and
// every shared array is row-interleaved (8 channels of one point contiguous), so warps touch 256 contiguous bytes.
// The fp16 rows arrive by bulk copy; spectra leave as contiguous octet slices of the X tiles.
template <int CAP, int MAXT, int MINB>
__global__ void __launch_bounds__(MAXT, MINB)
    fwdx3_kernel(const __half *__restrict__ x, float2 *__restrict__ X, int BC, int KH, Plan2 pl,
                 const float2 *__restrict__ T, int LC, int MPi, int mbs, int SW, int xoffw, int pair, bool pdl)
{
    extern __shared__ float4 smem_f4[];
    float2 *S = reinterpret_cast<float2 *>(smem_f4);
    uint32_t *xin = reinterpret_cast<uint32_t *>(smem_f4) + xoffw; // staged input rows (xoffw == 0: aliased by S)
    __shared__ __align__(8) uint64_t s_bar;
    const int t = threadIdx.x, r8 = t & 7, col8 = t >> 3, NT = blockDim.x, NC = NT >> 3;
    const int kh = blockIdx.x, oct = blockIdx.y, cg = blockIdx.z;
    const int c0 = 8 * oct, w = min(8, LC - c0), bc0 = cg * LC + c0;
    const int nlive = max(0, min(w, BC - bc0));
    const int n = pl.n, N1 = pl.N1, N2 = pl.N2, LD = pl.LD, Mi = n + 1;
    const int MB = 1 << mbs, sw = mbs ? (oct & 1) : 0, TL = MB * LC;
    float2 *Xb = X + (size_t)(cg * KH + kh) * MPi * LC + (size_t)c0 * MB;
    auto xo = [&](int m) { return (m >> mbs) * TL + ((m & (MB - 1)) ^ sw) * w; }; // see xoff (MB a power of two)
    if (nlive == 0) {                                                             // all channels are padding
        if (r8 < w)
            for (int m = col8; m < MPi; m += NC) Xb[r8 + xo(m)] = make_float2(0.f, 0.f);
        if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
        return;
    }
    // Compact a partial channel octet across the CTA. Shared rows retain stride eight, so the FFT layout and
    // synchronization remain unchanged while each live row receives more column threads.
    const bool compact = nlive < 8;
    const int row_count = compact ? nlive : 8;
    const bool row_pow2 = (row_count & (row_count - 1)) == 0;
    const int row_shift = __ffs(row_count) - 1;
    const int r = compact ? (row_pow2 ? (t & (row_count - 1)) : t % row_count) : r8;
    const int col = compact ? (row_pow2 ? (t >> row_shift) : t / row_count) : col8;
    const __half *xr0 = x + ((size_t)bc0 * KH + kh) * (size_t)(2 * n);
    const size_t rs = (size_t)KH * (size_t)(2 * n); // halves between channels
    if ((n & 3) == 0) {
        const uint32_t bar = smem_u32(&s_bar);
        if (t == 0) {
            mbar_init(bar, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbar_expect_tx(bar, (uint32_t)(nlive * 4 * n));
            for (int q = 0; q < nlive; ++q) bulk_g2s(smem_u32(xin + q * SW), xr0 + q * rs, (uint32_t)(4 * n), bar);
        }
        __syncthreads();
        mbar_wait(bar, 0);
    } else {
        for (int i = t; i < nlive * n; i += NT) {
            const int q = i / n, k = i - q * n;
            xin[q * SW + k] = reinterpret_cast<const uint32_t *>(xr0 + q * rs)[k];
        }
        __syncthreads();
    }
    if constexpr (CAP == TIER_M || CAP == TIER_S || CAP == TIER_N || CAP == TIER_L) {
        // both radices compile-time (nested dispatch over the tier's radix pairs): shared-memory strides fold into
        // immediate offsets. Padded channel rows are transformed too (they were zeroed), so no selects later.
        with_radix_cap<CAP>(N1, [&](auto R1C) {
            with_radix_cap<CAP>(N2, [&](auto R2C) {
                constexpr int R1 = decltype(R1C)::value, R2 = decltype(R2C)::value, LDC = R2 | 1;
                {
                    const bool act = col < R2;
                    float2 v[R1];
                    if (act) {
                        const uint32_t *xr = xin + r * SW + col;
#pragma unroll
                        for (int j1 = 0; j1 < R1; ++j1)
                            v[j1] = __half22float2(*reinterpret_cast<const __half2 *>(xr + R2 * j1));
                    }
                    if (xoffw == 0) __syncthreads(); // S aliases the staged rows
                    float2 *Sp = S + col * 8 + r;
                    if (act) {
                        if constexpr (CAP == TIER_M || CAP == TIER_S)
                            passA_dft_put_twochain<R1>(v, col, T, [&](int k, float2 u) { Sp[k * LDC * 8] = u; });
                        else
                            passA_dft_put<R1>(v, col, n, T, [&](int k, float2 u) { Sp[k * LDC * 8] = u; });
                    }
                }
                __syncthreads();
                {
                    const bool act = col < R1;
                    float2 u[R2];
                    if (act) {
                        const float2 *Sq = S + col * LDC * 8 + r;
#pragma unroll
                        for (int j2 = 0; j2 < R2; ++j2) u[j2] = Sq[j2 * 8];
                        dftn<R2>(u);
                    }
                    __syncthreads();
                    if (act) {
                        float2 *Sp = S + col * 8 + r;
#pragma unroll
                        for (int k2 = 0; k2 < R2; ++k2) Sp[R1 * k2 * 8] = u[k2];
                    }
                }
            });
        });
        __syncthreads();
    } else {
        {
            const bool act = col < N2;
            with_radix_cap<CAP>(N1, [&](auto RC) {
                constexpr int R = decltype(RC)::value;
                float2 v[R];
                if (act) {
#pragma unroll
                    for (int j1 = 0; j1 < R; ++j1)
                        v[j1] = __half22float2(*reinterpret_cast<const __half2 *>(xin + r * SW + N2 * j1 + col));
                }
                if (xoffw == 0) __syncthreads(); // S aliases the staged rows
                if (act) passA_dft_put<R>(v, col, n, T, [&](int k, float2 u) { S[(k * LD + col) * 8 + r] = u; });
            });
        }
        __syncthreads();
        {
            const bool act = col < N1;
            with_radix_cap<CAP>(N2, [&](auto RC) {
                constexpr int R = decltype(RC)::value;
                float2 u[R];
                if (act) {
#pragma unroll
                    for (int j2 = 0; j2 < R; ++j2) u[j2] = S[(col * LD + j2) * 8 + r];
                    dftn<R>(u);
                }
                __syncthreads();
                if (act) {
#pragma unroll
                    for (int k2 = 0; k2 < R; ++k2) S[(col + N1 * k2) * 8 + r] = u[k2];
                }
            });
        }
        __syncthreads();
    }
    // channel pairs: thread (rp = t & 3, frequency lane t >> 2) handles channels 2rp, 2rp + 1 with 16-byte accesses,
    // so one twiddle and one offset serve two transforms and a warp stores 8 consecutive slots (512 contiguous bytes)
    if (!compact) {
        const int rp = t & 3, MS = NT >> 2;
        float2 *Xp = Xb + 2 * rp;
        const float4 *Z4 = reinterpret_cast<const float4 *>(S) + rp;
        for (int m = t >> 2; 2 * m <= n; m += MS) {
            const float2 tw = __ldg(T + m);
            const float4 za = Z4[4 * m], zb = Z4[4 * (m == 0 ? 0 : n - m)];
            float2 a0, b0, a1, b1;
            rpost2x2(make_float2(za.x, za.y), make_float2(zb.x, zb.y), tw, a0, b0);
            rpost2x2(make_float2(za.z, za.w), make_float2(zb.z, zb.w), tw, a1, b1);
            // pair layout (pscale 1 packed inverse): slots 2m, 2m + 1 hold frequencies m, n - m
            *reinterpret_cast<float4 *>(Xp + xo(pair ? 2 * m : m)) = make_float4(a0.x, a0.y, a1.x, a1.y);
            *reinterpret_cast<float4 *>(Xp + xo(pair ? 2 * m + 1 : n - m)) = make_float4(b0.x, b0.y, b1.x, b1.y);
        }
        for (int m = (pair ? 2 * (n / 2 + 1) : Mi) + (t >> 2); m < MPi; m += MS)
            *reinterpret_cast<float4 *>(Xp + xo(m)) = make_float4(0.f, 0.f, 0.f, 0.f);
    } else {
        // Scalar channel stores avoid transforming padded rows while still initializing every padded spectrum slot.
        const int npair = n / 2 + 1;
        const bool w_pow2 = (w & (w - 1)) == 0;
        const int w_shift = __ffs(w) - 1;
        for (int z = t; z < npair * w; z += NT) {
            const int m = w_pow2 ? z >> w_shift : z / w;
            const int c = z - m * w;
            float2 a = make_float2(0.f, 0.f), b = make_float2(0.f, 0.f);
            if (c < nlive) rpost2x2(S[8 * m + c], S[8 * (m == 0 ? 0 : n - m) + c], __ldg(T + m), a, b);
            Xb[c + xo(pair ? 2 * m : m)] = a;
            Xb[c + xo(pair ? 2 * m + 1 : n - m)] = b;
        }
        const int first_pad = pair ? 2 * npair : Mi;
        for (int z = t; z < (MPi - first_pad) * w; z += NT) {
            const int dm = w_pow2 ? z >> w_shift : z / w;
            const int m = first_pad + dm, c = z - dm * w;
            Xb[c + xo(m)] = make_float2(0.f, 0.f);
        }
    }
    if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
}

// Quartet variant for a single channel group whose last physical octet has width four. Keeping this separate from
// fwdx3_kernel preserves the octet kernel's compact partial-row path and generated code. The quartet may trigger the
// unchanged full-octet grid so both grids can overlap and join through their final PDL waits.
template <int CAP, int MAXT, int MINB>
__global__ void __launch_bounds__(MAXT, MINB)
    fwdx4q_kernel(const __half *__restrict__ x, float2 *__restrict__ X, int BC, int KH, Plan2 pl,
                  const float2 *__restrict__ T, int LC, int MPi, int mbs, int SW, int xoffw, int pair, int octoff,
                  bool pdl, bool trig)
{
    extern __shared__ float4 smem_f4[];
    float2 *S = reinterpret_cast<float2 *>(smem_f4);
    uint32_t *xin = reinterpret_cast<uint32_t *>(smem_f4) + xoffw;
    __shared__ __align__(8) uint64_t s_bar;
    if (trig) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    constexpr int CH = 4;
    const int t = threadIdx.x, r = t & 3, col = t >> 2, NT = blockDim.x, NC = NT >> 2;
    const int kh = blockIdx.x, oct = octoff + blockIdx.y, cg = blockIdx.z;
    const int c0 = 8 * oct, w = min(CH, LC - c0), bc0 = cg * LC + c0;
    const int nlive = max(0, min(w, BC - bc0));
    const int n = pl.n, N1 = pl.N1, N2 = pl.N2, LD = pl.LD, Mi = n + 1;
    const int MB = 1 << mbs, sw = mbs ? (oct & 1) : 0, TL = MB * LC;
    float2 *Xb = X + (size_t)(cg * KH + kh) * MPi * LC + (size_t)c0 * MB + r;
    auto xo = [&](int m) { return (m >> mbs) * TL + ((m & (MB - 1)) ^ sw) * w; };
    if (nlive == 0) {
        if (r < w)
            for (int m = col; m < MPi; m += NC) Xb[xo(m)] = make_float2(0.f, 0.f);
        if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
        return;
    }
    const __half *xr0 = x + ((size_t)bc0 * KH + kh) * (size_t)(2 * n);
    const size_t rs = (size_t)KH * (size_t)(2 * n);
    if ((n & 3) == 0) {
        const uint32_t bar = smem_u32(&s_bar);
        if (t == 0) {
            mbar_init(bar, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbar_expect_tx(bar, (uint32_t)(nlive * 4 * n));
            for (int q = 0; q < nlive; ++q) bulk_g2s(smem_u32(xin + q * SW), xr0 + q * rs, (uint32_t)(4 * n), bar);
        }
        for (int i = t; i < (w - nlive) * n; i += NT) xin[(nlive + i / n) * SW + i % n] = 0u;
        __syncthreads();
        mbar_wait(bar, 0);
    } else {
        for (int i = t; i < nlive * n; i += NT) {
            const int q = i / n, k = i - q * n;
            xin[q * SW + k] = reinterpret_cast<const uint32_t *>(xr0 + q * rs)[k];
        }
        for (int i = t; i < (w - nlive) * n; i += NT) xin[(nlive + i / n) * SW + i % n] = 0u;
        __syncthreads();
    }
    if constexpr (CAP == TIER_M || CAP == TIER_S || CAP == TIER_N || CAP == TIER_L) {
        with_radix_cap<CAP>(N1, [&](auto R1C) {
            with_radix_cap<CAP>(N2, [&](auto R2C) {
                constexpr int R1 = decltype(R1C)::value, R2 = decltype(R2C)::value, LDC = R2 | 1;
                {
                    const bool act = r < w && col < R2;
                    float2 v[R1];
                    if (act) {
                        const uint32_t *xr = xin + r * SW + col;
#pragma unroll
                        for (int j1 = 0; j1 < R1; ++j1)
                            v[j1] = __half22float2(*reinterpret_cast<const __half2 *>(xr + R2 * j1));
                    }
                    if (xoffw == 0) __syncthreads();
                    float2 *Sp = S + col * CH + r;
                    if (act) {
                        if constexpr (CAP == TIER_M || CAP == TIER_S)
                            passA_dft_put_twochain<R1>(v, col, T, [&](int k, float2 u) { Sp[k * LDC * CH] = u; });
                        else
                            passA_dft_put<R1>(v, col, n, T, [&](int k, float2 u) { Sp[k * LDC * CH] = u; });
                    }
                }
                __syncthreads();
                {
                    const bool act = r < w && col < R1;
                    float2 u[R2];
                    if (act) {
                        const float2 *Sq = S + col * LDC * CH + r;
#pragma unroll
                        for (int j2 = 0; j2 < R2; ++j2) u[j2] = Sq[j2 * CH];
                        dftn<R2>(u);
                    }
                    __syncthreads();
                    if (act) {
                        float2 *Sp = S + col * CH + r;
#pragma unroll
                        for (int k2 = 0; k2 < R2; ++k2) Sp[R1 * k2 * CH] = u[k2];
                    }
                }
            });
        });
        __syncthreads();
    } else {
        {
            const bool act = r < w && col < N2;
            with_radix_cap<CAP>(N1, [&](auto RC) {
                constexpr int R = decltype(RC)::value;
                float2 v[R];
                if (act) {
#pragma unroll
                    for (int j1 = 0; j1 < R; ++j1)
                        v[j1] = __half22float2(*reinterpret_cast<const __half2 *>(xin + r * SW + N2 * j1 + col));
                }
                if (xoffw == 0) __syncthreads();
                if (act) passA_dft_put<R>(v, col, n, T, [&](int k, float2 u) { S[(k * LD + col) * CH + r] = u; });
            });
        }
        __syncthreads();
        {
            const bool act = r < w && col < N1;
            with_radix_cap<CAP>(N2, [&](auto RC) {
                constexpr int R = decltype(RC)::value;
                float2 u[R];
                if (act) {
#pragma unroll
                    for (int j2 = 0; j2 < R; ++j2) u[j2] = S[(col * LD + j2) * CH + r];
                    dftn<R>(u);
                }
                __syncthreads();
                if (act) {
#pragma unroll
                    for (int k2 = 0; k2 < R; ++k2) S[(col + N1 * k2) * CH + r] = u[k2];
                }
            });
        }
        __syncthreads();
    }
    const int rp = t & 1, MS = NT >> 1;
    const bool l0 = 2 * rp < nlive, l1 = 2 * rp + 1 < nlive;
    float2 *Xp = Xb - r + 2 * rp;
    const float4 *Z4 = reinterpret_cast<const float4 *>(S) + rp;
    for (int m = t >> 1; 2 * m <= n; m += MS) {
        const float2 tw = __ldg(T + m);
        const float4 za = Z4[2 * m], zb = Z4[2 * (m == 0 ? 0 : n - m)];
        float2 a0, b0, a1, b1;
        rpost2x2(make_float2(za.x, za.y), make_float2(zb.x, zb.y), tw, a0, b0);
        rpost2x2(make_float2(za.z, za.w), make_float2(zb.z, zb.w), tw, a1, b1);
        if constexpr (!(CAP == TIER_M || CAP == TIER_S || CAP == TIER_N || CAP == TIER_L)) {
            if (!l0) a0 = b0 = make_float2(0.f, 0.f);
            if (!l1) a1 = b1 = make_float2(0.f, 0.f);
        }
        *reinterpret_cast<float4 *>(Xp + xo(pair ? 2 * m : m)) = make_float4(a0.x, a0.y, a1.x, a1.y);
        *reinterpret_cast<float4 *>(Xp + xo(pair ? 2 * m + 1 : n - m)) = make_float4(b0.x, b0.y, b1.x, b1.y);
    }
    for (int m = (pair ? 2 * (n / 2 + 1) : Mi) + (t >> 1); m < MPi; m += MS)
        *reinterpret_cast<float4 *>(Xp + xo(m)) = make_float4(0.f, 0.f, 0.f, 0.f);
    if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
}

// ---------------------------------------------------------------- psi arc spectra (two-pass)
// P is tile-major: slot j < PLD of plane o for arc s lives at ((j / MBt) * nseg + s) * NO * MBt + o * MBt + j % MBt,
// so the arcs of one slot tile are contiguous. Plane 0 holds frequency j, plane 1 (pscale 2) frequency Wi - j;
// zero for j >= nslot. Arcs of any length (wrapping onto themselves) are summed. Block (max(N1, N2), 8).
#define PSI_PR 4 // arcs per psi block
template <int CAP, int MINB>
__global__ void __launch_bounds__(32 * PSI_PR, MINB * 8 / PSI_PR)
    fwdpsi2_kernel(const int *__restrict__ seg, const int64_t *__restrict__ segval, const float *__restrict__ vals,
                   float2 *__restrict__ P, int nseg, Plan2 pl, const float2 *__restrict__ T, int Wi, int NO, int nslot,
                   int PLD, int MBt, int mbt_sh, int *__restrict__ atype)
{
    extern __shared__ float4 smem_f4[];
    float2 *S = reinterpret_cast<float2 *>(smem_f4);
    __shared__ int s_st[PSI_PR], s_len[PSI_PR], s_short[PSI_PR], s_nsym[PSI_PR], s_nanti[PSI_PR];
    __shared__ int64_t s_vo[PSI_PR];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int NT = blockDim.x * PSI_PR, t = tx + blockDim.x * ty;
    const int n = pl.n, N1 = pl.N1, N2 = pl.N2, LD = pl.LD, SROW = pl.SROW, Wo = 2 * n;
    const int s0 = blockIdx.x * PSI_PR;
    if (t < PSI_PR) {
        const int s = s0 + t;
        int start = 0, len = 0;
        int64_t vo = 0;
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
        // spectrum type (see arc_type_kernel): only arcs centred on longitude 0 can be (anti)symmetric
        const bool cen = len >= 1 && len <= Wo && (2 * start + len - 1) % Wo == 0;
        s_nsym[t] = !cen;
        s_nanti[t] = !cen;
    }
    __syncthreads();
    for (int r = 0; r < PSI_PR; ++r) {
        if (s_short[r]) continue;
        float4 *z4 = reinterpret_cast<float4 *>(S + r * SROW);
        for (int i = t; i < SROW / 2; i += NT) z4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __syncthreads();
    bool wraps = false;
    for (int r = 0; r < PSI_PR; ++r) {
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
        for (int r = 0; r < PSI_PR; ++r) {
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
        with_radix_cap<CAP>(N1, [&](auto RC) {
            constexpr int R = decltype(RC)::value;
            psi_passA<R>(live && tx < N2, sparse, tx, n, N2, T, Sr, LD, start >> 1, span, start, len, vr);
        });
    }
    __syncthreads();
    if (atype) { // all arcs at once (thread -> arc t % PSI_PR); values are cache-hot; flags are read after two more barriers
        const int r = t % PSI_PR, len = s_len[r];
        if (len >= 1 && len <= Wo && (2 * s_st[r] + len - 1) % Wo == 0) { // only centred arcs can be (anti)symmetric
            const float *v = vals + s_vo[r];
            bool ns = false, na = false;
            for (int j = t / PSI_PR; j < (len + 1) / 2; j += NT / PSI_PR) {
                const float a = __ldg(v + j), b = __ldg(v + len - 1 - j);
                ns |= a != b;
                na |= a != -b;
            }
            if (ns) atomicOr(&s_nsym[r], 1);
            if (na) atomicOr(&s_nanti[r], 1);
        }
    }
    with_radix_cap<CAP>(N2, [&](auto RC) {
        constexpr int R = decltype(RC)::value;
        fft_passB<R>(live && tx < N1, Sr + tx * LD, [&](int k2, float2 u) { Sr[tx + N1 * k2] = u; });
    });
    __syncthreads();
    const int nr = min(PSI_PR, nseg - s0);
    // Traverse frequencies outside arcs so each twiddle and tile offset serves every arc in the CTA.
    // Tiled (fast) layout: virtual slot u = slot * NO + plane, pairs of virtual slots stored (re, re, im, im) (pfoff).
    const size_t tstride = (size_t)nseg * NO * MBt;
    float *Pf = reinterpret_cast<float *>(P + (size_t)s0 * NO * MBt);
    auto put = [&](int r, int j, int o, float2 v) {
        if (mbt_sh >= 0) {
            const int tj = j >> mbt_sh;
            float *q = Pf + 2 * ((size_t)tj * tstride + (size_t)r * NO * MBt) + pfoff((j - (tj << mbt_sh)) * NO + o);
            q[0] = v.x;
            q[2] = v.y;
        } else {
            reinterpret_cast<float2 *>(Pf)[(size_t)r * MBt + j] = v;
        }
    };
    for (int m = t; 2 * m <= n; m += NT) {
        const float2 tw = __ldg(T + m);
        const int mn = m == 0 ? 0 : n - m;
        const bool two = n - m != m;
        for (int r = 0; r < nr; ++r) {
            const float2 *Zr = S + r * SROW;
            float2 a, b;
            rpost2x2(Zr[m], Zr[mn], tw, a, b); // arc spectra kept doubled, like the input spectra
            if (NO == 2) {                     // virtual slots 2m', 2m' + 1 form one (re, re, im, im) quad
                const int tj = m >> mbt_sh;
                float *q = Pf + 2 * ((size_t)tj * tstride + (size_t)r * NO * MBt) + 4 * (m - (tj << mbt_sh));
                *reinterpret_cast<float4 *>(q) = make_float4(a.x, b.x, a.y, -b.y);
            } else {
                put(r, m, 0, a);
                if (two) put(r, n - m, 0, b);
            }
        }
    }
    const int ntail = PLD - nslot;
    for (int i = t; i < nr * ntail; i += NT) {
        const int r = i / ntail, j = nslot + i - r * ntail;
        for (int o = 0; o < NO; ++o) put(r, j, o, make_float2(0.f, 0.f));
    }
    if (atype && t < nr) atype[s0 + t] = !s_nsym[t] ? 1 : (!s_nanti[t] ? 2 : 0);
}

// Warp-per-arc variant: every warp transforms one arc in its own shared row with warp barriers only, so arcs of
// different lengths and paths never wait for each other. Same output layout and arc typing as fwdpsi2_kernel.
template <int N1>
__device__ __forceinline__ void psi_passA_w(bool act, bool sparse, int j2, int n, int N2, const float2 *__restrict__ T,
                                            float2 *__restrict__ S, int LD, int c0, int span, int start, int len,
                                            const float *__restrict__ v)
{
    if (!sparse) {
        float2 vv[N1];
        if (act) {
#pragma unroll
            for (int j1 = 0; j1 < N1; ++j1) vv[j1] = S[N2 * j1 + j2];
        }
        __syncwarp();
        if (act) passA_dft_store<N1>(vv, j2, n, T, S, LD);
        return;
    }
    if (!act) return;
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
    // powers p * w^k1 in two interleaved chains (w^2 steps) to halve the dependent-latency chain
    const float2 w2 = cmul(w, w);
    float2 pa = p, pb = cmul(p, w);
#pragma unroll
    for (int k1 = 0; k1 < N1; k1 += 2) {
        S[k1 * LD + j2] = pa;
        if (k1 + 1 < N1) S[(k1 + 1) * LD + j2] = pb;
        pa = cmul(pa, w2);
        pb = cmul(pb, w2);
    }
}

#define PSIW_WARPS 4
#define PSI_DIRECT_H 24 // (anti)symmetric arcs of half-length <= this use the direct series
template <int CAP, int MINB>
__global__ void __launch_bounds__(32 * PSIW_WARPS, MINB)
    fwdpsiw_kernel(const int *__restrict__ seg, const int64_t *__restrict__ segval, const float *__restrict__ vals,
                   float2 *__restrict__ P, int nseg, Plan2 pl, const float2 *__restrict__ T, int NO, int nslot, int PLD,
                   int MBt, int mbt_sh, int *__restrict__ atype, int pair, bool pdl)
{
    if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    extern __shared__ float4 smem_f4[];
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int n = pl.n, N1 = pl.N1, N2 = pl.N2, LD = pl.LD, SROW = pl.SROW, Wo = 2 * n;
    float2 *S = reinterpret_cast<float2 *>(smem_f4) + (size_t)wp * SROW;
    const size_t tstride2 = 2 * (size_t)nseg * NO * MBt; // floats between slot tiles
    const int s = blockIdx.x * PSIW_WARPS + wp;
    if (s < nseg) { // whole warp; no block barriers below
        const int st0 = seg[3 * s + 1];
        const int len = max(seg[3 * s + 2], 0);
        const float *vr = vals + segval[s];
        int start = st0;
        if ((unsigned)start >= (unsigned)Wo) {
            start %= Wo;
            if (start < 0) start += Wo;
        }
        int ty = 0;
        float vb0 = 0.f;
        const int64_t c2 = 2ll * start + len - 1;
        if (atype && len >= 1 && len <= Wo && (c2 == 0 || c2 == Wo || c2 == 2ll * Wo)) {
            bool ns = false, na = false; // only centred arcs can be exactly (anti)symmetric
            for (int j = lane; j < (len + 1) / 2; j += 32) {
                const float a = __ldg(vr + j), b = __ldg(vr + len - 1 - j);
                if (j == lane) vb0 = b;
                ns |= a != b;
                na |= a != -b;
            }
            ns = __any_sync(0xffffffffu, ns);
            na = __any_sync(0xffffffffu, na);
            ty = !ns ? 1 : (!na ? 2 : 0);
        }
        float *Pf = reinterpret_cast<float *>(P + (size_t)s * NO * MBt);
        auto put = [&](int j, int o, float2 v) {
            if (mbt_sh >= 0) {
                const int tj = j >> mbt_sh;
                float *q = Pf + (size_t)tj * tstride2 + pfoff((j - (tj << mbt_sh)) * NO + o);
                q[0] = v.x;
                q[2] = v.y;
            } else {
                reinterpret_cast<float2 *>(Pf)[j] = v;
            }
        };
        // spectra (doubled, like the input spectra) at frequencies m and n - m, stored for every m in [0, n/2]
        auto store = [&](int m, float2 a, float2 b) {
            if (NO == 2) { // virtual slots 2m, 2m + 1 form one (re, re, im, im) quad
                const int tj = m >> mbt_sh;
                *reinterpret_cast<float4 *>(Pf + (size_t)tj * tstride2 + 4 * (m - (tj << mbt_sh)))
                    = make_float4(a.x, b.x, a.y, -b.y);
            } else if (pair) { // slots 2m, 2m + 1 hold frequencies m, n - m (see fwdx3_kernel)
                const int j = 2 * m, tj = j >> mbt_sh;
                *reinterpret_cast<float4 *>(Pf + (size_t)tj * tstride2 + pfoff(j - (tj << mbt_sh)))
                    = make_float4(a.x, b.x, a.y, b.y);
            } else {
                put(m, 0, a);
                if (n - m != m) put(n - m, 0, b); // m = 0 stores the Nyquist value at slot n
            }
        };
        const int H = (len - 1) >> 1; // a centred arc has odd length 2H + 1, centre value index H
        // The Clenshaw formula below is centred at longitude zero. An otherwise symmetric arc centred at Wo/2 has an
        // additional (-1)^m phase, so leave that valid type classification intact but send it through the FFT path.
        const int64_t center = (int64_t)start + H;
        const bool direct_center = len >= 1 && (len <= Wo ? (center == 0 || center == Wo) : center % Wo == 0);
        if (ty != 0 && H <= PSI_DIRECT_H && direct_center) {
            // Short (anti)symmetric arc (fast layout only, atype is passed only then): with theta = 2 pi m / Wo,
            //   real:      P(m) = 2 v_H + 4 sum_{d=1..H} v_{H+d} cos(d theta)
            //   imaginary: P(m) = -4 i sum_{d=1..H} v_{H+d} sin(d theta)
            // by Clenshaw recurrences; frequencies m and n - m (cos -> -cos, sin unchanged) share one f32x2 lane.
            float *cf = reinterpret_cast<float *>(S);
            // v[H+d] is lane j=H-d's mirrored symmetry-test value. H<=PSI_DIRECT_H<32 here.
            if (lane <= H) cf[H - lane] = (lane == H ? 2.f : 4.f) * vb0;
            __syncwarp();
            const float a0 = ty == 1 ? cf[0] : 0.f;
            const int nh = n >> 1;
            // NO == 2: lane's quad for m = lane + 32 i is q0 + i * qstep (MBt <= 32 divides the lane stride)
            float *q0 = Pf + (size_t)(lane >> mbt_sh) * tstride2 + 4 * (lane & (MBt - 1));
            const size_t qstep = (size_t)(32 >> mbt_sh) * tstride2;
            for (int i0 = 0; 32 * i0 <= nh; i0 += 4) { // four frequency pairs per lane and pass
                float2 w[4], c[4], b1[4], b2[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    w[j] = __ldg(T + min(lane + 32 * (i0 + j), nh));
                    c[j] = make_float2(2.f * w[j].x, -2.f * w[j].x);
                    b1[j] = b2[j] = make_float2(0.f, 0.f);
                }
#pragma unroll 2
                for (int d = H; d >= 1; --d) {
                    const float ad = cf[d];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 t = fma2(c[j], b1[j], add2(make_float2(ad, ad), make_float2(-b2[j].x, -b2[j].y)));
                        b2[j] = b1[j];
                        b1[j] = t;
                    }
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int m = lane + 32 * (i0 + j);
                    if (m <= nh) {
                        float2 a, b;
                        if (ty == 1) {
                            a = make_float2(a0 + w[j].x * b1[j].x - b2[j].x, 0.f);
                            b = make_float2(a0 - w[j].x * b1[j].y - b2[j].y, 0.f);
                        } else {
                            a = make_float2(0.f, w[j].y * b1[j].x); // -sin(theta) = w.y
                            b = make_float2(0.f, w[j].y * b1[j].y);
                        }
                        if (NO == 2)
                            *reinterpret_cast<float4 *>(q0 + (size_t)(i0 + j) * qstep) = make_float4(a.x, b.x, a.y, -b.y);
                        else
                            store(m, a, b);
                    }
                }
            }
        } else {
            const int span = ((start & 1) + len + 1) >> 1;
            const bool sparse = len <= Wo && span <= N2;
            if (!sparse) {
                float4 *z4 = reinterpret_cast<float4 *>(S);
                for (int i = lane; i < SROW / 2; i += 32) z4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
                __syncwarp();
                float *row = reinterpret_cast<float *>(S);
                for (int j = lane; j < min(len, Wo); j += 32) {
                    int pos = start + j;
                    if (pos >= Wo) pos -= Wo;
                    row[pos] = __ldg(vr + j);
                }
                if (len > Wo) { // arcs longer than the circle wrap onto themselves
                    __syncwarp();
                    for (int j = Wo + lane; j < len; j += 32) atomicAdd(row + (start + j) % Wo, __ldg(vr + j));
                }
                __syncwarp();
            }
            if constexpr (CAP == TIER_M || CAP == TIER_S || CAP == TIER_N || CAP == TIER_L) {
                with_radix_cap<CAP>(N1, [&](auto R1C) {
                    with_radix_cap<CAP>(N2, [&](auto R2C) {
                        constexpr int R1 = decltype(R1C)::value, R2 = decltype(R2C)::value, LDC = R2 | 1;
                        psi_passA_w<R1>(lane < R2, sparse, lane, n, R2, T, S, LDC, start >> 1, span, start, len, vr);
                        __syncwarp();
                        const bool act = lane < R1;
                        float2 u[R2];
                        if (act) {
#pragma unroll
                            for (int j2 = 0; j2 < R2; ++j2) u[j2] = S[lane * LDC + j2];
                            dftn<R2>(u);
                        }
                        __syncwarp();
                        if (act) {
#pragma unroll
                            for (int k2 = 0; k2 < R2; ++k2) S[lane + R1 * k2] = u[k2];
                        }
                    });
                });
            } else {
                with_radix_cap<CAP>(N1, [&](auto RC) {
                    constexpr int R = decltype(RC)::value;
                    psi_passA_w<R>(lane < N2, sparse, lane, n, N2, T, S, LD, start >> 1, span, start, len, vr);
                });
                __syncwarp();
                with_radix_cap<CAP>(N2, [&](auto RC) {
                    constexpr int R = decltype(RC)::value;
                    const bool act = lane < N1;
                    float2 u[R];
                    if (act) {
#pragma unroll
                        for (int j2 = 0; j2 < R; ++j2) u[j2] = S[lane * LD + j2];
                        dftn<R>(u);
                    }
                    __syncwarp();
                    if (act) {
#pragma unroll
                        for (int k2 = 0; k2 < R; ++k2) S[lane + N1 * k2] = u[k2];
                    }
                });
            }
            __syncwarp();
            if (NO == 2) { // virtual slots 2m, 2m + 1 form one (re, re, im, im) quad; MBt <= 32 divides the lane stride
                float *q = Pf + (size_t)(lane >> mbt_sh) * tstride2 + 4 * (lane & (MBt - 1));
                const size_t qstep = (size_t)(32 >> mbt_sh) * tstride2;
#pragma unroll 4
                for (int m = lane; 2 * m <= n; m += 32, q += qstep) {
                    const float2 tw = __ldg(T + m);
                    float2 a, b;
                    rpost2x2(S[m], S[m == 0 ? 0 : n - m], tw, a, b); // arc spectra kept doubled, like the input spectra
                    *reinterpret_cast<float4 *>(q) = make_float4(a.x, b.x, a.y, -b.y);
                }
            } else {
#pragma unroll 4
                for (int m = lane; 2 * m <= n; m += 32) {
                    const float2 tw = __ldg(T + m);
                    float2 a, b;
                    rpost2x2(S[m], S[m == 0 ? 0 : n - m], tw, a, b);
                    store(m, a, b);
                }
            }
        }
        if (NO == 2) {
            for (int j = nslot + lane; j < PLD; j += 32) {
                const int tj = j >> mbt_sh;
                *reinterpret_cast<float4 *>(Pf + (size_t)tj * tstride2 + 4 * (j - (tj << mbt_sh)))
                    = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        } else if (pair) {
            for (int j = nslot + 2 * lane; j < PLD; j += 64) {
                const int tj = j >> mbt_sh;
                *reinterpret_cast<float4 *>(Pf + (size_t)tj * tstride2 + pfoff(j - (tj << mbt_sh)))
                    = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        } else {
            for (int j = nslot + lane; j < PLD; j += 32) put(j, 0, make_float2(0.f, 0.f));
        }
        if (atype && lane == 0) atype[s] = ty;
    }
}

// ---------------------------------------------------------------- inverse real FFT -> fp16 (two-pass)
// out[bc, h, :] = irfft(OUT[h, :, bc]). Block = channel octet x output ring h; thread t owns channel row
// r = t & 7 and FFT column t >> 3 (row-interleaved shared arrays). The octet's spectra arrive with one bulk copy; the
// fp16 rows are staged in shared memory and leave by bulk copy.
template <int CAP, int MAXT, int MINB, bool PACKED = false>
__global__ void __launch_bounds__(MAXT, MINB)
    inv3_kernel(const float2 *__restrict__ OUT, __half *__restrict__ out, int BC, int BCp, int Ho, Plan2 pl,
                const float2 *__restrict__ T, int SW)
{
    // Allow this dependent grid to be scheduled while the contraction drains, but do not consume OUT early.
    asm volatile("griddepcontrol.wait;" ::: "memory");
    extern __shared__ float4 smem_f4[];
    float2 *S = reinterpret_cast<float2 *>(smem_f4);
    uint32_t *stg = reinterpret_cast<uint32_t *>(smem_f4);
    __shared__ __align__(8) uint64_t s_bar;
    const int t = threadIdx.x, r8 = t & 7, col8 = t >> 3, NT = blockDim.x, NC = NT >> 3;
    const int n = pl.n, N1 = pl.N1, N2 = pl.N2, LD = pl.LD, Mf = PACKED ? n : n + 1;
    const int oct = blockIdx.x, h = blockIdx.y;
    const uint32_t bar = smem_u32(&s_bar);
    {
        const int c0 = 8 * oct, w = min(8, BCp - c0), bc0 = c0;
        const int nlive = max(0, min(w, BC - bc0));
        if (nlive == 0) return;
        // Concentrate a partial channel octet into whole warps while retaining the shared row stride of eight.
        const bool compact = nlive < 8;
        const bool row_pow2 = (nlive & (nlive - 1)) == 0;
        const int row_shift = __ffs(nlive) - 1;
        const int r = compact ? (row_pow2 ? (t & (nlive - 1)) : t % nlive) : r8;
        const int col = compact ? (row_pow2 ? (t >> row_shift) : t / nlive) : col8;
        const float2 *src = OUT + ((size_t)h * BCp + c0) * Mf;
        if (w == 8) {
            if (t == 0) {
                mbar_init(bar, 1);
                asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
                mbar_expect_tx(bar, (uint32_t)(Mf * 64));
                bulk_g2s(smem_u32(S), src, (uint32_t)(Mf * 64), bar);
            }
            __syncthreads();
            mbar_wait(bar, 0);
        } else {
            for (int i = t; i < Mf * 8; i += NT) {
                const int m = i >> 3, q = i & 7;
                S[i] = q < w ? src[(size_t)m * w + q] : make_float2(0.f, 0.f);
            }
            __syncthreads();
        }
        // inverse complex-FFT input, formed once per symmetric frequency pair, in place
        // (channel pairs with 16-byte accesses: thread (t & 3, frequency lane t >> 2)). PACKED spectra were formed by
        // the contraction epilogue (n points, already the complex-FFT input).
        auto preprocess_pair = [&](int rp, int m) {
            float4 *Z4 = reinterpret_cast<float4 *>(S) + rp;
            {
                const float4 xm = Z4[4 * m], xn = Z4[4 * (n - m)];
                const float2 tw = __ldg(T + m);
                float2 za[2], zb[2];
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const float2 xa = q ? make_float2(xm.z, xm.w) : make_float2(xm.x, xm.y);
                    const float2 xb = q ? make_float2(xn.z, xn.w) : make_float2(xn.x, xn.y);
                    const float2 e = make_float2(xa.x + xb.x, xa.y - xb.y);
                    const float2 d = make_float2(xa.x - xb.x, xa.y + xb.y);
                    const float2 z = cmul(make_float2(tw.x, -tw.y), d);
                    za[q] = make_float2(e.x - z.y, -e.y - z.x);
                    zb[q] = make_float2(e.x + z.y, e.y - z.x);
                }
                Z4[4 * m] = make_float4(za[0].x, za[0].y, za[1].x, za[1].y);
                if (m > 0 && n - m != m) Z4[4 * (n - m)] = make_float4(zb[0].x, zb[0].y, zb[1].x, zb[1].y);
            }
        };
        if constexpr (!PACKED) {
            if (!compact) {
                const int rp = t & 3;
                if (2 * rp < nlive)
                    for (int m = t >> 2; 2 * m <= n; m += NT >> 2) preprocess_pair(rp, m);
            } else {
                const int cpairs = (nlive + 1) >> 1;
                const bool cp_pow2 = (cpairs & (cpairs - 1)) == 0;
                const int cp_shift = __ffs(cpairs) - 1;
                for (int z = t; z < (n / 2 + 1) * cpairs; z += NT) {
                    const int m = cp_pow2 ? z >> cp_shift : z / cpairs;
                    const int rp = z - m * cpairs;
                    preprocess_pair(rp, m);
                }
            }
        }
        __syncthreads();
        if constexpr (CAP == TIER_M || CAP == TIER_S || CAP == TIER_N || CAP == TIER_L) {
            with_radix_cap<CAP>(N1, [&](auto R1C) {
                with_radix_cap<CAP>(N2, [&](auto R2C) {
                    constexpr int R1 = decltype(R1C)::value, R2 = decltype(R2C)::value, LDC = R2 | 1;
                    {
                        const bool act = col < R2;
                        float2 v[R1];
                        if (act) {
#pragma unroll
                            for (int j1 = 0; j1 < R1; ++j1) v[j1] = S[(R2 * j1 + col) * 8 + r];
                        }
                        __syncthreads();
                        if (act) {
                            if constexpr (CAP == TIER_M || CAP == TIER_S)
                                passA_dft_put_twochain<R1>(v, col, T,
                                                           [&](int k, float2 u) { S[(k * LDC + col) * 8 + r] = u; });
                            else
                                passA_dft_put<R1>(v, col, n, T, [&](int k, float2 u) { S[(k * LDC + col) * 8 + r] = u; });
                        }
                    }
                    __syncthreads();
                    {
                        const bool act = col < R1;
                        const float sc = 0.125f / (float)n;
                        float2 u[R2];
                        if (act) {
#pragma unroll
                            for (int j2 = 0; j2 < R2; ++j2) u[j2] = S[(col * LDC + j2) * 8 + r];
                            dftn<R2>(u);
                        }
                        __syncthreads();
                        if (act) {
#pragma unroll
                            for (int k2 = 0; k2 < R2; ++k2) {
                                const __half2 hv = __floats2half2_rn(u[k2].x * sc, -u[k2].y * sc);
                                stg[r * SW + col + R1 * k2] = *reinterpret_cast<const uint32_t *>(&hv);
                            }
                        }
                    }
                });
            });
        } else {
            {
                const bool act = col < N2;
                with_radix_cap<CAP>(N1, [&](auto RC) {
                    constexpr int R = decltype(RC)::value;
                    float2 v[R];
                    if (act) {
#pragma unroll
                        for (int j1 = 0; j1 < R; ++j1) v[j1] = S[(N2 * j1 + col) * 8 + r];
                    }
                    __syncthreads();
                    if (act) passA_dft_put<R>(v, col, n, T, [&](int k, float2 u) { S[(k * LD + col) * 8 + r] = u; });
                });
            }
            __syncthreads();
            {
                const bool act = col < N1;
                const float sc = 0.125f / (float)n; // irfft 1 / (2n); input and arc spectra are both kept doubled
                with_radix_cap<CAP>(N2, [&](auto RC) {
                    constexpr int R = decltype(RC)::value;
                    float2 u[R];
                    if (act) {
#pragma unroll
                        for (int j2 = 0; j2 < R; ++j2) u[j2] = S[(col * LD + j2) * 8 + r];
                        dftn<R>(u);
                    }
                    __syncthreads();
                    if (act) {
#pragma unroll
                        for (int k2 = 0; k2 < R; ++k2) {
                            const __half2 hv = __floats2half2_rn(u[k2].x * sc, -u[k2].y * sc);
                            stg[r * SW + col + N1 * k2] = *reinterpret_cast<const uint32_t *>(&hv);
                        }
                    }
                });
            }
        }
        __half *ob = out + ((size_t)bc0 * Ho + h) * (size_t)(2 * n);
        const size_t os = (size_t)Ho * (size_t)(2 * n); // halves between channels
        if ((n & 3) == 0) {
            fence_proxy_async();
            __syncthreads();
            if (t == 0) {
                for (int q = 0; q < nlive; ++q) bulk_s2g(ob + q * os, smem_u32(stg + q * SW), (uint32_t)(4 * n));
                bulk_commit_wait_read();
            }
        } else {
            __syncthreads();
            for (int i = t; i < nlive * n; i += NT) {
                const int q = i / n, k = i - q * n;
                reinterpret_cast<uint32_t *>(ob + q * os)[k] = stg[q * SW + k];
            }
        }
    }
}

// Packed inverse variant with four channels per block. It is used only for an underfilled Tier-N grid; halving the
// shared footprint and thread count permits twice as many resident blocks to overlap spectrum loads and transforms.
template <int CAP, int MAXT, int MINB>
__global__ void __launch_bounds__(MAXT, MINB)
    inv4q_kernel(const float2 *__restrict__ OUT, __half *__restrict__ out, int BC, int BCp, int Ho, Plan2 pl,
                 const float2 *__restrict__ T, int SW, float sc)
{
    asm volatile("griddepcontrol.wait;" ::: "memory");
    extern __shared__ float4 smem_f4[];
    float2 *S = reinterpret_cast<float2 *>(smem_f4);
    uint32_t *stg = reinterpret_cast<uint32_t *>(smem_f4);
    const int t = threadIdx.x, r = t & 3, col = t >> 2, NT = blockDim.x;
    const int n = pl.n, Mf = n;
    const int oct = blockIdx.x >> 1, hq = blockIdx.x & 1, h = blockIdx.y;
    const int w8 = min(8, BCp - 8 * oct);
    const int c0 = 8 * oct + 4 * hq, wq = min(4, w8 - 4 * hq);
    if (wq <= 0) return;
    const int nlive = min(wq, BC - c0);
    if (nlive <= 0) return;
    const float2 *src = OUT + ((size_t)h * BCp + 8 * oct) * Mf + 4 * hq;
    if (wq == 4 && (w8 & 1) == 0) {
        const float4 *s4 = reinterpret_cast<const float4 *>(src);
        float4 *S4 = reinterpret_cast<float4 *>(S);
        for (int i = t; i < Mf * 2; i += NT) S4[i] = __ldg(s4 + (((size_t)(i >> 1) * w8) >> 1) + (i & 1));
    } else {
        for (int i = t; i < Mf * 4; i += NT) {
            const int m = i >> 2, q = i & 3;
            S[i] = q < wq ? src[(size_t)m * w8 + q] : make_float2(0.f, 0.f);
        }
    }
    __syncthreads();
    with_radix_cap<CAP>(pl.N1, [&](auto R1C) {
        with_radix_cap<CAP>(pl.N2, [&](auto R2C) {
            constexpr int R1 = decltype(R1C)::value, R2 = decltype(R2C)::value, LDC = R2 | 1;
            {
                const bool act = r < nlive && col < R2;
                float2 v[R1];
                if (act) {
#pragma unroll
                    for (int j1 = 0; j1 < R1; ++j1) v[j1] = S[(R2 * j1 + col) * 4 + r];
                }
                __syncthreads();
                if (act) {
                    if constexpr (CAP == TIER_M || CAP == TIER_S)
                        passA_dft_put_twochain<R1>(v, col, T, [&](int k, float2 u) { S[(k * LDC + col) * 4 + r] = u; });
                    else
                        passA_dft_put<R1>(v, col, n, T, [&](int k, float2 u) { S[(k * LDC + col) * 4 + r] = u; });
                }
            }
            __syncthreads();
            {
                const bool act = r < nlive && col < R1;
                float2 u[R2];
                if (act) {
#pragma unroll
                    for (int j2 = 0; j2 < R2; ++j2) u[j2] = S[(col * LDC + j2) * 4 + r];
                    dftn<R2>(u);
                }
                __syncthreads();
                if (act) {
#pragma unroll
                    for (int k2 = 0; k2 < R2; ++k2) {
                        const __half2 hv = __floats2half2_rn(u[k2].x * sc, -u[k2].y * sc);
                        stg[r * SW + col + R1 * k2] = *reinterpret_cast<const uint32_t *>(&hv);
                    }
                }
            }
        });
    });
    __half *ob = out + ((size_t)c0 * Ho + h) * (size_t)(2 * n);
    const size_t os = (size_t)Ho * (size_t)(2 * n);
    if ((n & 3) == 0) {
        fence_proxy_async();
        __syncthreads();
        if (t == 0) {
            for (int q = 0; q < nlive; ++q) bulk_s2g(ob + q * os, smem_u32(stg + q * SW), (uint32_t)(4 * n));
            bulk_commit_wait_read();
        }
    } else {
        __syncthreads();
        for (int i = t; i < nlive * n; i += NT) {
            const int q = i / n, k = i - q * n;
            reinterpret_cast<uint32_t *>(ob + q * os)[k] = stg[q * SW + k];
        }
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
                             const float2 *__restrict__ T, int NB, int LC, int MPi, int MB)
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
        if (m < Mi)
            val = cscale(rpost(Z[a * n + (m == n ? 0 : m)], Z[a * n + (m == 0 ? 0 : n - m)], __ldg(T + m)), 2.f);
        X[(size_t)(cg * KH + kh) * MPi * LC + xoff(m, (int)cl, MB, LC)] = val;
    }
}

__global__ void fwdpsim_kernel(const int *__restrict__ seg, const int64_t *__restrict__ segval,
                               const float *__restrict__ vals, float2 *__restrict__ P, int nseg, PlanM p,
                               const float2 *__restrict__ T, int NB, int Wi, int NO, int nslot, int PLD, int MBt,
                               bool tiled)
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
            val = cscale(rpost(Z[a * n + (m == n ? 0 : m)], Z[a * n + (m == 0 ? 0 : n - m)], __ldg(T + m)), 2.f);
            if (o) val.y = -val.y;
        }
        const int tj = j / MBt;
        if (tiled) {
            float *q = reinterpret_cast<float *>(P + ((size_t)tj * nseg + s0 + a) * NO * MBt)
                + pfoff((j - tj * MBt) * NO + o);
            q[0] = val.x;
            q[2] = val.y;
        } else {
            P[((size_t)tj * nseg + s0 + a) * NO * MBt + (j - tj * MBt) * NO + o] = val;
        }
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
        const float2 xm = OUT[oidx((int)h, m, (int)bc, Mf, BCp)], xn = OUT[oidx((int)h, n - m, (int)bc, Mf, BCp)];
        const float2 e = make_float2(xm.x + xn.x, xm.y - xn.y);
        const float2 d = make_float2(xm.x - xn.x, xm.y + xn.y);
        const float2 tw = __ldg(T + m);
        const float2 q = cmul(make_float2(tw.x, -tw.y), d);
        A[a * n + m] = make_float2(e.x - q.y, -(e.y + q.x));
    }
    __syncthreads();
    const float2 *Rz = block_fft(A, B, n, nb, T, p);
    const float sc = 0.125f / (float)n; // irfft 1 / (2n); input and arc spectra are both kept doubled
    for (int w = tid; w < nb * n; w += bd) {
        const int a = w / n, k = w - a * n;
        const float2 r = Rz[a * n + k];
        reinterpret_cast<__half2 *>(out + (row0 + a) * (long)(2 * n))[k] = __floats2half2_rn(r.x * sc, -r.y * sc);
    }
}

// ---------------------------------------------------------------- general direct path
// Fully 64-bit output-owned fallback for shapes or offsets outside optimized 32-bit metadata.
__global__ void direct64_kernel(const __half *__restrict__ x, const int *__restrict__ row_ker,
                                const int *__restrict__ row_lat, const int64_t *__restrict__ seg_off,
                                const int *__restrict__ seg, const int64_t *__restrict__ val_off,
                                const float *__restrict__ vals, __half *__restrict__ out, int64_t nrows, int64_t Wi,
                                int64_t Hi, int64_t Ho, int64_t Wo, int64_t KH, int64_t nout_rows)
{
    const int64_t ps = Wo / Wi;
    const int64_t row_step = (int64_t)gridDim.x, w_step = (int64_t)blockDim.x;
    for (int64_t oh = (int64_t)blockIdx.x;;) {
        if (oh >= nout_rows) break;
        const int64_t bc = oh / Ho, h = oh - bc * Ho;
        for (int64_t w = (int64_t)threadIdx.x;;) {
            if (w >= Wo) break;
            float acc = 0.f;
            for (int64_t r = 0; r < nrows; ++r) {
                const int64_t kh = (int64_t)row_ker[r] * Hi + (int64_t)row_lat[r];
                const __half *xr = x + (bc * KH + kh) * Wi;
                int64_t v = val_off[r];
                for (int64_t s = seg_off[r]; s < seg_off[r + 1]; ++s) {
                    const int *a = seg + 3 * s;
                    const int64_t len = a[2] > 0 ? (int64_t)a[2] : 0;
                    if ((int64_t)a[0] == h && len > 0) {
                        int64_t start = (int64_t)a[1] % Wo;
                        if (start < 0) start += Wo;
                        const int64_t d = w - start;
                        int64_t quot = d / ps, j = d % ps;
                        if (j < 0) {
                            j += ps;
                            --quot;
                        }
                        int64_t q = quot % Wi;
                        if (q < 0) q += Wi;
                        for (;;) {
                            if (j >= len) break;
                            acc = fmaf(vals[v + j], __half2float(xr[q]), acc);
                            q = q == 0 ? Wi - 1 : q - 1;
                            if (ps >= len - j) break;
                            j += ps;
                        }
                    }
                    v += len;
                }
            }
            out[oh * Wo + w] = __float2half_rn(acc);
            if (w_step >= Wo - w) break;
            w += w_step;
        }
        if (row_step >= nout_rows - oh) break;
        oh += row_step;
    }
}

// Block per output row (bc, h), grid-stride. For widths that fit, the accumulator and input row live in shared
// memory; otherwise every thread owns output longitudes in registers and reads the input row from global memory.
template <bool SM>
__global__ void direct_kernel(const __half *__restrict__ x, const int4 *__restrict__ entries,
                              const int *__restrict__ boff, const int *__restrict__ seg,
                              const int64_t *__restrict__ segval, const float *__restrict__ vals,
                              __half *__restrict__ out, int Wi, int Ho, int Wo, int KH, long nrow)
{
    extern __shared__ float dsm[];
    float *acc = dsm;
    float *xs = dsm + Wo;
    const int tid = threadIdx.x, bd = blockDim.x;
    const int ps = Wo / Wi;
    for (long row = blockIdx.x; row < nrow; row += gridDim.x) {
        const long bc = row / Ho;
        const int h = (int)(row - bc * Ho);
        const int e0 = boff[h], e1 = boff[h + 1];
        __half *o = out + row * (long)Wo;
        if (SM) {
            __syncthreads();
            for (int w = tid; w < Wo; w += bd) acc[w] = 0.f;
            for (int e = e0; e < e1; ++e) {
                const int4 en = entries[e];
                const __half *xr = x + (bc * KH + en.x) * (long)Wi;
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
                        for (int64_t j = j0; j < (int64_t)len; j += ps) {
                            sum = fmaf(v[j], xs[q], sum);
                            q = (q == 0) ? Wi - 1 : q - 1;
                        }
                        acc[w] += sum;
                    }
                }
            }
            __syncthreads();
            for (int w = tid; w < Wo; w += bd) o[w] = __float2half_rn(acc[w]);
        } else {
            for (int w0 = 0; w0 < Wo; w0 += bd) {
                const int w = w0 + tid;
                if (w >= Wo) break;
                float a = 0.f;
                for (int e = e0; e < e1; ++e) {
                    const int4 en = entries[e];
                    const __half *xr = x + (bc * KH + en.x) * (long)Wi;
                    for (int s = en.y; s < en.z; ++s) {
                        int start = seg[3 * s + 1] % Wo;
                        if (start < 0) start += Wo;
                        const int len = seg[3 * s + 2];
                        const float *v = vals + segval[s];
                        const int d = w - start;
                        int j0 = d % ps;
                        if (j0 < 0) j0 += ps;
                        int q = (int)((((long)d - j0) / ps % Wi + Wi) % Wi);
                        float sum = 0.f;
                        for (int64_t j = j0; j < (int64_t)len; j += ps) {
                            sum = fmaf(v[j], __half2float(xr[q]), sum);
                            q = (q == 0) ? Wi - 1 : q - 1;
                        }
                        a += sum;
                    }
                }
                o[w] = __float2half_rn(a);
            }
        }
    }
}

// ---------------------------------------------------------------- prep
// One warp per psi row (arc value offsets by warp scan, entry counts per band from ballots) plus twiddle-table
// blocks; the last block to finish (ticket in cnt[nbands]) turns the counts into band offsets boff / fill cursors bcur.
#define PREP_T 256
template <int HBT>
__global__ void __launch_bounds__(PREP_T)
    prep_count_kernel(const int64_t *__restrict__ seg_off, const int *__restrict__ seg,
                      const int64_t *__restrict__ val_off, int nrows, int Ho, int64_t *__restrict__ segval,
                      int *__restrict__ cnt, float2 *__restrict__ Ti, int Ni, float2 *__restrict__ To, int No,
                      int nbands, int *__restrict__ boff, int *__restrict__ bcur)
{
    constexpr int WPB = PREP_T / 32;
    const int rb = (nrows + WPB - 1) / WPB;
    const int lane = threadIdx.x & 31;
    if ((int)blockIdx.x >= rb) {
        const int i = (blockIdx.x - rb) * blockDim.x + threadIdx.x;
        if (i < No) {
            float s, c;
            sincospif(-2.0f * (float)i / (float)No, &s, &c);
            const float2 z = make_float2(c, s);
            To[i] = z;
            const int ps = Ni > 0 ? No / Ni : 0;
            if (ps >= 1 && ps * Ni == No && i % ps == 0) Ti[i / ps] = z;
        }
    } else {
        const int r = blockIdx.x * WPB + (threadIdx.x >> 5);
        if (r < nrows) {
            const int lo = (int)seg_off[r], hi = (int)seg_off[r + 1];
            int64_t vbase = val_off[r];
            int pring = 0;
            bool pvalid = false;
            for (int b0 = lo; b0 < hi; b0 += 32) {
                const int s = b0 + lane;
                int ring = -1, len = 0;
                if (s < hi) {
                    ring = seg[3 * s];
                    len = seg[3 * s + 2];
                }
                const int l = max(len, 0);
                int inc = l;
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) {
                    const int t = __shfl_up_sync(0xffffffffu, inc, o);
                    if (lane >= o) inc += t;
                }
                if (s < hi) segval[s] = vbase + (inc - l);
                vbase += __shfl_sync(0xffffffffu, inc, 31);
                const bool valid = s < hi && ring >= 0 && ring < Ho && len > 0;
                int pr = __shfl_up_sync(0xffffffffu, ring, 1);
                bool pv = __shfl_up_sync(0xffffffffu, (int)valid, 1) != 0;
                if (lane == 0) {
                    pr = pring;
                    pv = pvalid;
                }
                if (valid && !(pv && pr / HBT == ring / HBT && ring == pr + 1)) atomicAdd(&cnt[ring / HBT], 1);
                pring = __shfl_sync(0xffffffffu, ring, 31);
                pvalid = __shfl_sync(0xffffffffu, (int)valid, 31) != 0;
            }
        }
    }
    __shared__ int s_last;
    __shared__ int ssum[PREP_T];
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        s_last = atomicAdd(&cnt[nbands], 1) == (int)gridDim.x - 1;
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    const int tid = threadIdx.x, bd = blockDim.x;
    const int per = (nbands + bd - 1) / bd;
    const int beg = min(tid * per, nbands), end = min(beg + per, nbands);
    int local = 0;
    for (int i = beg; i < end; ++i) local += __ldcg(cnt + i);
    ssum[tid] = local;
    __syncthreads();
    for (int off = 1; off < bd; off <<= 1) {
        const int t = tid >= off ? ssum[tid - off] : 0;
        __syncthreads();
        ssum[tid] += t;
        __syncthreads();
    }
    int run = ssum[tid] - local;
    for (int i = beg; i < end; ++i) {
        boff[i] = run;
        bcur[i] = run;
        run += __ldcg(cnt + i);
    }
    if (tid == bd - 1) boff[nbands] = ssum[bd - 1];
}

// entry = (input row index kh, first arc, end arc, ring mask in band | spectrum type << 16): a maximal run of
// consecutive arcs of one row inside one band on consecutive rings. type = AND of the arc types (0 if atype null).
template <int HBT>
__global__ void prep_fill_kernel(const int64_t *__restrict__ seg_off, const int *__restrict__ seg,
                                 const int *__restrict__ row_ker, const int *__restrict__ row_lat, int nrows, int Hi,
                                 int Ho, int *__restrict__ bcur, int4 *__restrict__ entries,
                                 const int *__restrict__ atype, bool pdl)
{
    // Permit the fast contraction grid to become resident while this metadata pass drains. Every CTA releases
    // dependents before its early exit, including the synthetic CTA used for an empty row set.
    if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    // one warp per row; rows of up to 32 arcs are split into runs with ballots, longer rows walk serially
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (r >= nrows) return;
    const int lo = (int)seg_off[r], hi = (int)seg_off[r + 1];
    const int kh = row_ker[r] * Hi + row_lat[r];
    auto emit = [&](int s, int e, int ring, int ty) {
        const int band = ring / HBT;
        const int pos = atomicAdd(&bcur[band], 1);
        const unsigned rmask = ((2u << (e - s - 1)) - 1u) << (ring - band * HBT); // HBT <= 16
        entries[pos] = make_int4(kh, s, e, (int)(rmask | ((unsigned)ty << 16)));
    };
    if (hi - lo <= 32) {
        const int s = lo + lane;
        int ring = -1, len = 0, ty = 0;
        if (s < hi) {
            ring = seg[3 * s];
            len = seg[3 * s + 2];
            ty = atype ? atype[s] : 0;
        }
        const bool valid = s < hi && ring >= 0 && ring < Ho && len > 0;
        const int pring = __shfl_up_sync(0xffffffffu, ring, 1);
        const bool pvalid = __shfl_up_sync(0xffffffffu, (int)valid, 1) != 0 && lane > 0;
        const bool start = valid && !(pvalid && pring / HBT == ring / HBT && ring == pring + 1);
        const unsigned brk = __ballot_sync(0xffffffffu, !valid || start);
        const unsigned t0 = __ballot_sync(0xffffffffu, (ty & 1) != 0), t1 = __ballot_sync(0xffffffffu, (ty & 2) != 0);
        if (start) {
            const unsigned above = lane == 31 ? 0u : (brk & ~((2u << lane) - 1u));
            const int el = above ? __ffs(above) - 1 : 32;
            const unsigned run = (el == 32 ? 0xffffffffu : ((1u << el) - 1u)) & ~((1u << lane) - 1u);
            const int rty = atype ? (((t0 & run) == run ? 1 : 0) | ((t1 & run) == run ? 2 : 0)) : 0;
            emit(s, lo + el, ring, rty);
        }
        return;
    }
    if (lane) return;
    int s = lo;
    while (s < hi) {
        const int ring = seg[3 * s], len = seg[3 * s + 2];
        if (ring < 0 || ring >= Ho || len <= 0) {
            ++s;
            continue;
        }
        const int band = ring / HBT;
        int e = s + 1, pr = ring;
        while (e < hi) {
            const int rr = seg[3 * e], ll = seg[3 * e + 2];
            if (rr < 0 || rr >= Ho || ll <= 0 || rr / HBT != band || rr != pr + 1) break;
            pr = rr;
            ++e;
        }
        int ty = atype ? 3 : 0;
        if (atype)
            for (int a = s; a < e; ++a) ty &= atype[a];
        emit(s, e, ring, ty);
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
        int np2 = 1;
        while (np2 < ne) np2 <<= 1;
        s_ent[tid] = tid < ne ? entries[ec + tid] : make_int4(0, 0x7fffffff, 0, 0);
        __syncthreads();
        for (int kk = 2; kk <= np2; kk <<= 1) {
            for (int j = kk >> 1; j > 0; j >>= 1) {
                const int ixj = tid ^ j;
                if (ixj > tid) {
                    const int4 a = s_ent[tid], b = s_ent[ixj];
                    const bool up = (tid & kk) == 0;
                    if ((a.y > b.y) == up) {
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
__global__ void arc_type_kernel(const int *__restrict__ seg, const int64_t *__restrict__ segval,
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

// ---------------------------------------------------------------- banded contraction (pscale 1 / 2)
// grid (band, channel group, slot tile). Every entry covers consecutive rings [klo, klo + cnt) of the band, so its
// arcs' spectra land at fixed ring slots of a stage. A producer warp streams entries into a runtime-sized stage ring
// (entry metadata + X tile with one bulk copy + arc spectra with another) completing on a full transaction barrier;
// every consumer thread releases a stage on its empty barrier once consumed, so the main loop has no block barrier.
// Consumer thread (cp, g) owns channels 2cp, 2cp + 1 and two virtual slots v (PS 1: slots 2g + v; PS 2: slot g,
// plane v; plane 0 = output frequency j, plane 1 = output frequency Wi - j conjugated), for all HB rings.
#define MAXSTG 16
// PACK: a thread's two virtual slots are output frequencies p and n - p (n = Wo / 2): pscale 2 planes (p = slot,
// n = Wi, plane 1 conjugated) or the pscale 1 pair layout (p = slot pair, n = Wi / 2). That is exactly the pair the
// inverse real FFT combines, so the epilogue writes the inverse complex-FFT input z[p], z[n - p] (n points per ring
// and channel).
template <int NO, int HB, int NSG, int CPT, bool EARLY_B, bool PACK = false, int MAXT = 320, int MINB = (HB <= 4 ? 3 : 2)>
__global__ void __launch_bounds__(MAXT, MINB)
    contract_kernel(const float2 *__restrict__ X, const float2 *__restrict__ P, const int4 *__restrict__ entries,
                    const int *__restrict__ boff, float2 *__restrict__ OUT, int LC, int KH, int MPi, int nseg, int Ho,
                    int Mf, int Wi, int BCp, int nslot, int CSTG, const float2 *__restrict__ T)
{
    constexpr int SPT = (NO == 1) ? 2 : 1; // slots per thread
    constexpr int MB = NSG * SPT;          // slots per tile
    constexpr int PA = MB * NO;            // float2 per arc
    const int NCP = LC / CPT;              // channel-group threads per slot group
    const int NT = NCP * NSG;              // consumer threads (whole warps)
    const int tid = threadIdx.x;
    const int band = blockIdx.x, cg = blockIdx.y, mt = blockIdx.z;
    const int XT = MB * LC;           // float2 of the X tile
    const int STG = 2 + XT + HB * PA; // float2 per stage: meta (int4), X tile, ring slots of arc spectra
    extern __shared__ __align__(128) float2 smem_c[];
    uint64_t *s_full = reinterpret_cast<uint64_t *>(smem_c + (size_t)CSTG * STG);
    uint64_t *s_empty = s_full + CSTG;
    const uint32_t full0 = smem_u32(s_full), empty0 = smem_u32(s_empty);
    if (tid == 0) {
        for (int d = 0; d < CSTG; ++d) {
            mbar_init(full0 + 8 * d, 1);
            mbar_init(empty0 + 8 * d, NT);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    // Barrier setup is independent of prep_fill. Wait only before reading its offsets and entries, then release the
    // inverse grid; this keeps the two PDL handoffs ordered instead of creating an over-eager three-grid chain.
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int e0 = boff[band], ne = boff[band + 1] - e0;

    if (tid >= NT) { // producer warp
        if (tid != NT) return;
        const float2 *Xg = X + ((size_t)cg * KH * MPi + (size_t)mt * MB) * LC;
        const size_t xrow = (size_t)MPi * LC;
        const float2 *Pt = P + (size_t)mt * nseg * PA;
        const uint32_t sbase = smem_u32(smem_c), sstg = (uint32_t)STG * sizeof(float2);
        const uint32_t xbytes = (uint32_t)(XT * sizeof(float2));
        uint64_t p_policy;
        asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p_policy));
        int sidx = 0;
        uint32_t par = 1; // the first pass over the stages needs no release
        int4 me = ne > 0 ? entries[e0] : make_int4(0, 0, 0, 0);
        for (int i = 0; i < ne; ++i) {
            const int4 nx = (i + 1 < ne) ? entries[e0 + i + 1] : me;
            if (i >= CSTG) { mbar_wait(empty0 + 8 * sidx, par); }
            const uint32_t sd = sbase + (uint32_t)sidx * sstg, bar = full0 + 8 * sidx;
            *reinterpret_cast<int4 *>(smem_c + (size_t)sidx * STG) = me;
            const uint32_t pbytes = (uint32_t)((me.z - me.y) * PA * sizeof(float2));
            mbar_expect_tx(bar, xbytes + pbytes);
            bulk_g2s(sd + 16, Xg + (size_t)me.x * xrow, xbytes, bar);
            const uint32_t pd = sd + 16 + xbytes + (uint32_t)((__ffs(me.w & 0xffff) - 1) * PA * sizeof(float2));
            if (gridDim.y > 1)
                bulk_g2s_l2_last(pd, Pt + (size_t)me.y * PA, pbytes, bar, p_policy);
            else
                bulk_g2s(pd, Pt + (size_t)me.y * PA, pbytes, bar);
            me = nx;
            if (++sidx == CSTG) {
                sidx = 0;
                par ^= 1u;
            }
        }
        return;
    }

    const int cp = tid % NCP, g = tid / NCP;
    float2 acc[HB][2][CPT];
#pragma unroll
    for (int k = 0; k < HB; ++k)
#pragma unroll
        for (int v = 0; v < 2; ++v)
#pragma unroll
            for (int c = 0; c < CPT; ++c) acc[k][v][c] = make_float2(0.f, 0.f);

    // this thread's X (channels CPT*cp .. CPT*cp + CPT-1; slots g * SPT + q) in stage 0: octet-major tile, see xoff
    const int c0t = CPT * cp, xoc = c0t >> 3, xw = min(8, LC - 8 * xoc), xsw = (MB & 1) ? 0 : (xoc & 1);
    const float2 *xp0 = smem_c + 2 + 8 * xoc * MB + ((g * SPT) ^ xsw) * xw + (c0t & 7);
    const float2 *xp1 = smem_c + 2 + 8 * xoc * MB + ((g * SPT + SPT - 1) ^ xsw) * xw + (c0t & 7);
    const float2 *pp = smem_c + 2 + XT + g * 2; // this thread's spectra (ring 0) in stage 0
    // Ring loop of one staged entry at stage offset so.
    auto load_x = [&](int so, float2(&xv)[SPT][CPT]) {
#pragma unroll
        for (int q = 0; q < SPT; ++q)
#pragma unroll
            for (int c = 0; c < CPT; c += 2) {
                const float4 t4 = *reinterpret_cast<const float4 *>((q ? xp1 : xp0) + so + c);
                xv[q][c] = make_float2(t4.x, t4.y);
                xv[q][c + 1] = make_float2(t4.z, t4.w);
            }
    };
    // torch-harmonics: T was a generic-lambda parameter (auto TY, T = decltype(TY)::value) with if constexpr on it,
    // which crashes the CUDA 13.3 front end (cudafe++). It is a plain int now; every call passes a constant and the
    // lambda is inlined, so the branches on it fold as before.
    auto entry = [&](const float2(&xv)[SPT][CPT], int so, unsigned rm, const int T) {
        const float2 *pr = pp + so;
        // rings in groups of 4: a group without arcs is skipped by a warp-uniform branch, inside a group the updates
        // are predicated
#pragma unroll
        for (int g4 = 0; g4 < HB; g4 += 4) {
            if (rm & (0xfu << g4)) {
#pragma unroll
                for (int k = g4; k < g4 + 4 && k < HB; ++k) {
                    if (rm & (1u << k)) {
                        // (re v0, re v1, im v0, im v1), see pfoff; real / imaginary arcs load only their half
                        float4 p4 = make_float4(0.f, 0.f, 0.f, 0.f);
                        if (T == 0) p4 = *reinterpret_cast<const float4 *>(pr + k * PA);
                        if (T == 1) {
                            const float2 q = pr[k * PA];
                            p4.x = q.x;
                            p4.y = q.y;
                        }
                        if (T == 2) {
                            const float2 q = pr[k * PA + 1];
                            p4.z = q.x;
                            p4.w = q.y;
                        }
#pragma unroll
                        for (int v = 0; v < 2; ++v) {
                            const float2 p = v ? make_float2(p4.y, p4.w) : make_float2(p4.x, p4.z);
#pragma unroll
                            for (int c = 0; c < CPT; ++c) {
                                const float2 x = xv[NO == 1 ? v : 0][c];
                                if (T != 2) acc[k][v][c] = fma2(make_float2(p.x, p.x), x, acc[k][v][c]);
                                if (T != 1)
                                    acc[k][v][c] = fma2(make_float2(p.y, p.y), make_float2(-x.y, x.x), acc[k][v][c]);
                            }
                        }
                    }
                }
            }
        }
    };
    auto run_entry = [&](const float2(&xv)[SPT][CPT], int so, int w) {
        const unsigned rm = (unsigned)w & 0xffffu;
        const int ty = w >> 16;
        if (ty == 1)
            entry(xv, so, rm, 1);
        else if (ty == 2)
            entry(xv, so, rm, 2);
        else
            entry(xv, so, rm, 0);
    };
    int sc = 0;
    uint32_t par = 0;
    if constexpr (CPT >= 4 && NO == 1) {
        // Four channels already expose twice the independent updates per staged entry. Consume one stage at a time so
        // the compiler need not keep two four-channel operand tiles live alongside the accumulators.
        for (int i = 0; i < ne; ++i) {
            mbar_wait(full0 + 8 * sc, par);
            const int wa = reinterpret_cast<const int4 *>(smem_c + sc * STG)->w;
            {
                float2 xa[SPT][CPT];
                load_x(sc * STG, xa);
                run_entry(xa, sc * STG, wa);
            }
            // Order every consumer's generic-proxy reads before the producer can overwrite this async-proxy stage.
            fence_proxy_async();
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(empty0 + 8 * sc) : "memory");
            if (++sc == CSTG) {
                sc = 0;
                par ^= 1u;
            }
        }
    } else {
        // Move entries in pairs to amortize stage bookkeeping and expose ILP between independent stages.
        for (int i = 0; i < ne; i += 2) {
            const bool two = i + 1 < ne;
            const int sc2 = (sc + 1 == CSTG) ? 0 : sc + 1;
            const uint32_t par2 = (sc2 == 0) ? (par ^ 1u) : par;
            mbar_wait(full0 + 8 * sc, par);
            const int wa = reinterpret_cast<const int4 *>(smem_c + sc * STG)->w;
            if constexpr (NO == 2 && CPT <= 2) {
                // both entries' X operands are requested before either is dispatched, so the second stage's shared-memory
                // latency hides behind the first entry's updates (pscale 2 has the register room: one slot per thread)
                float2 xa[SPT][CPT], xb[SPT][CPT];
                load_x(sc * STG, xa);
                int wb = 0;
                if (two) {
                    mbar_wait(full0 + 8 * sc2, par2);
                    wb = reinterpret_cast<const int4 *>(smem_c + sc2 * STG)->w;
                    load_x(sc2 * STG, xb);
                }
                run_entry(xa, sc * STG, wa);
                if (two) run_entry(xb, sc2 * STG, wb);
            } else {
                if constexpr (EARLY_B) {
                    {
                        float2 xa[SPT][CPT];
                        load_x(sc * STG, xa);
                        run_entry(xa, sc * STG, wa);
                    }
                    if (two) {
                        mbar_wait(full0 + 8 * sc2, par2);
                        const int wb = reinterpret_cast<const int4 *>(smem_c + sc2 * STG)->w;
                        float2 xb[SPT][CPT];
                        load_x(sc2 * STG, xb);
                        run_entry(xb, sc2 * STG, wb);
                    }
                } else {
                    int wb = 0;
                    if (two) {
                        mbar_wait(full0 + 8 * sc2, par2);
                        wb = reinterpret_cast<const int4 *>(smem_c + sc2 * STG)->w;
                    }
                    {
                        float2 xa[SPT][CPT];
                        load_x(sc * STG, xa);
                        run_entry(xa, sc * STG, wa);
                    }
                    if (two) {
                        float2 xb[SPT][CPT];
                        load_x(sc2 * STG, xb);
                        run_entry(xb, sc2 * STG, wb);
                    }
                }
            }
            // Consumer generic-proxy loads above must become visible to the producer's async proxy before either stage
            // is released for overwrite. One fence covers both independently consumed stages.
            fence_proxy_async();
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(empty0 + 8 * sc) : "memory");
            if (two) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(empty0 + 8 * sc2) : "memory");
            sc = (sc2 + 1 == CSTG) ? 0 : sc2 + 1;
            par = (sc2 + 1 == CSTG) ? (par2 ^ 1u) : par2;
        }
    }
    const int cbase = cg * LC + CPT * cp;
    if constexpr (PACK) {
        const int p = mt * NSG + g, n = NO == 2 ? Wi : Wi / 2;
        if (2 * p > n) return;
        const float2 tw = __ldg(T + p);
#pragma unroll
        for (int k = 0; k < HB; ++k) {
            const int h = band * HB + k;
            if (h < Ho) {
#pragma unroll
                for (int c = 0; c < CPT; c += 2) {
                    float2 za[2], zb[2];
#pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        const float2 xa = acc[k][0][c + q];
                        const float2 xb = (2 * p == n) ?
                            xa :
                            make_float2(acc[k][1][c + q].x, NO == 2 ? -acc[k][1][c + q].y : acc[k][1][c + q].y);
                        const float2 e = make_float2(xa.x + xb.x, xa.y - xb.y);
                        const float2 d = make_float2(xa.x - xb.x, xa.y + xb.y);
                        const float2 qq = cmul(make_float2(tw.x, -tw.y), d);
                        za[q] = make_float2(e.x - qq.y, -e.y - qq.x);
                        zb[q] = make_float2(e.x + qq.y, e.y - qq.x);
                    }
                    *reinterpret_cast<float4 *>(OUT + oidx(h, p, cbase + c, n, BCp))
                        = make_float4(za[0].x, za[0].y, za[1].x, za[1].y);
                    if (p > 0 && 2 * p < n)
                        *reinterpret_cast<float4 *>(OUT + oidx(h, n - p, cbase + c, n, BCp))
                            = make_float4(zb[0].x, zb[0].y, zb[1].x, zb[1].y);
                }
            }
        }
        return;
    }
#pragma unroll
    for (int k = 0; k < HB; ++k) {
        const int h = band * HB + k;
        if (h < Ho) {
#pragma unroll
            for (int v = 0; v < 2; ++v) {
                const int j = mt * MB + g * SPT + (NO == 1 ? v : 0);
                if (j >= nslot) continue;
                int jj = j;
                const float sg = (NO == 2 && v == 1) ? -1.f : 1.f;
                if (NO == 2 && v == 1) {
                    if (Wi - j == j) continue;
                    jj = Wi - j;
                }
                float2 *o = OUT + oidx(h, jj, cbase, Mf, BCp); // CPT channels contiguous (same octet)
#pragma unroll
                for (int c = 0; c < CPT; c += 2)
                    *reinterpret_cast<float4 *>(o + c)
                        = make_float4(acc[k][v][c].x, sg * acc[k][v][c].y, acc[k][v][c + 1].x, sg * acc[k][v][c + 1].y);
            }
        }
    }
}

#define CPT_FAST 2 // channels per consumer thread
#define HB_FAST 8  // rings per band of the fast contraction
template <int NO, int HBT, int NSG, bool EARLY_B, bool PACK, int MAXT = 320, int MINB = (HBT <= 4 ? 3 : 2), int CPT = CPT_FAST>
static void launch_contract(dim3 grid, int threads, size_t smem, cudaStream_t st, const float2 *X, const float2 *P,
                            const int4 *ent, const int *boff, float2 *OUT, int LC, int KH, int MPi, int nseg, int Ho,
                            int Mf, int Wi, int BCp, int nslot, int nstg, const float2 *T)
{
    auto kern = contract_kernel<NO, HBT, NSG, CPT, EARLY_B, PACK, MAXT, MINB>;
    // Small blocks (few channels): trade stages for residency. Take the largest block count per SM (up to 6) that
    // the registers allow with at least 10 stages each; otherwise keep the default ~100 KB ring (2 blocks per SM).
    const size_t probe_smem = smem;
    if (probe_smem > 48 * 1024)
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)probe_smem);
    const size_t sbytes = smem / (size_t)nstg; // bytes per stage (incl. its two barriers)
    for (int b = 6; b > 2; --b) {
        const int n = (int)min((size_t)MAXSTG, ((size_t)220 * 1024 / b) / sbytes);
        if (n < (HBT <= 5 ? 8 : 10)) continue; // HB4/5 trade a shallower ring for a third resident CTA
        int occ = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, threads, n * sbytes) == cudaSuccess && occ >= b) {
            nstg = n;
            smem = n * sbytes;
            break;
        }
    }
    if (smem != probe_smem) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(threads);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = st;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kern, X, P, ent, boff, OUT, LC, KH, MPi, nseg, Ho, Mf, Wi, BCp, nslot, nstg, T);
}

template <int NO, bool EARLY_B = false, bool PACK = false>
static void dispatch_contract(int NSG, dim3 grid, int threads, size_t smem, cudaStream_t st, const float2 *X,
                              const float2 *P, const int4 *ent, const int *boff, float2 *OUT, int LC, int KH, int MPi,
                              int nseg, int Ho, int Mf, int Wi, int BCp, int nslot, int nstg, const float2 *T = nullptr)
{
    switch (NSG) {
    case 1:
        launch_contract<NO, HB_FAST, 1, EARLY_B, PACK>(grid, threads, smem, st, X, P, ent, boff, OUT, LC, KH, MPi, nseg,
                                                       Ho, Mf, Wi, BCp, nslot, nstg, T);
        break;
    case 2:
        launch_contract<NO, HB_FAST, 2, EARLY_B, PACK>(grid, threads, smem, st, X, P, ent, boff, OUT, LC, KH, MPi, nseg,
                                                       Ho, Mf, Wi, BCp, nslot, nstg, T);
        break;
    case 4:
        launch_contract<NO, HB_FAST, 4, EARLY_B, PACK>(grid, threads, smem, st, X, P, ent, boff, OUT, LC, KH, MPi, nseg,
                                                       Ho, Mf, Wi, BCp, nslot, nstg, T);
        break;
    case 8:
        launch_contract<NO, HB_FAST, 8, EARLY_B, PACK>(grid, threads, smem, st, X, P, ent, boff, OUT, LC, KH, MPi, nseg,
                                                       Ho, Mf, Wi, BCp, nslot, nstg, T);
        break;
    case 16:
        launch_contract<NO, HB_FAST, 16, EARLY_B, PACK>(grid, threads, smem, st, X, P, ent, boff, OUT, LC, KH, MPi,
                                                        nseg, Ho, Mf, Wi, BCp, nslot, nstg, T);
        break;
    default:
        launch_contract<NO, HB_FAST, 32, EARLY_B, PACK>(grid, threads, smem, st, X, P, ent, boff, OUT, LC, KH, MPi,
                                                        nseg, Ho, Mf, Wi, BCp, nslot, nstg, T);
        break;
    }
}

// Four channels per consumer thread for pscale 2: twice the FMAs per staged entry and per psi-spectrum load.
template <bool PACK>
static void dispatch_contract4(int NSG, dim3 grid, int threads, size_t smem, cudaStream_t st, const float2 *X,
                               const float2 *P, const int4 *ent, const int *boff, float2 *OUT, int LC, int KH, int MPi,
                               int nseg, int Ho, int Mf, int Wi, int BCp, int nslot, int nstg, const float2 *T)
{
    switch (NSG) {
    case 8:
        launch_contract<2, 4, 8, true, PACK, 320, 2, 4>(grid, threads, smem, st, X, P, ent, boff, OUT, LC, KH, MPi,
                                                        nseg, Ho, Mf, Wi, BCp, nslot, nstg, T);
        break;
    case 16:
        launch_contract<2, 4, 16, true, PACK, 320, 2, 4>(grid, threads, smem, st, X, P, ent, boff, OUT, LC, KH, MPi,
                                                         nseg, Ho, Mf, Wi, BCp, nslot, nstg, T);
        break;
    default:
        launch_contract<2, 4, 32, true, PACK, 320, 2, 4>(grid, threads, smem, st, X, P, ent, boff, OUT, LC, KH, MPi,
                                                         nseg, Ho, Mf, Wi, BCp, nslot, nstg, T);
        break;
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
        float2 xv = __ldg(X + (((size_t)cg * KH + en.x) * MPi + xm) * LC + c);
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
        if (h < Ho) OUT[oidx(h, m, cg * LC + c, Mf, BCp)] = acc[k];
    }
}

// ---------------------------------------------------------------- host
// lf: among balanced factorizations prefer the larger first radix (else the smaller)
static bool make_plan2(int n, Plan2 &p, bool lf = false)
{
    int best = 1 << 30;
    bool ok = false;
    for (int a : kRadix) {
        if (n % a) continue;
        const int b = n / a;
        bool inset = false;
        for (int r : kRadix) inset |= (r == b);
        if (!inset) continue;
        const int cost = max(a, b) * 64 + (lf ? b : a); // balanced, then smaller (larger) first radix
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
    p.NT = 8 * max(p.N1, p.N2);
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

// Same-stream programmatic dependent launch: the successor may become resident while its predecessor drains and
// executes griddepcontrol.wait before reading the predecessor's results.
template <class... P, class... A>
static void launch_pdl(void (*kern)(P...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, A... args)
{
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = st;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kern, args...);
}

static inline bool mul_i64(int64_t a, int64_t b, int64_t &out)
{
    if (a < 0 || b < 0 || (a != 0 && b > INT64_MAX / a)) return false;
    out = a * b;
    return true;
}

static void launch_direct64(cudaStream_t st, const __half *x, const int *row_ker, const int *row_lat,
                            const int64_t *seg_off, const int *seg, const int64_t *val_off, const float *vals,
                            __half *out, int64_t nrows, int64_t Wi, int64_t Hi, int64_t Ho, int64_t Wo, int64_t KH,
                            int64_t nout_rows)
{
    const unsigned nblk = (unsigned)min(nout_rows, (int64_t)(1 << 20));
    direct64_kernel<<<nblk, 256, 0, st>>>(x, row_ker, row_lat, seg_off, seg, val_off, vals, out, nrows, Wi, Hi, Ho, Wo,
                                          KH, nout_rows);
}

// =====================================================================================================================
// Host side: the plan and run ops of torch-harmonics.
//
// What follows replaces the candidate's TVM FFI entry point. Its launch sequence is split in two:
//
//   plan  -- everything derived from psi: arc value offsets, band entry lists, twiddle tables, arc types and the psi
//            arc spectra (prep_count, fwdpsiw / fwdpsim + arc_type, prep_fill). Built once per (psi, B*C) into a
//            byte buffer the caller keeps, in the layout of ScatterFftConfig.
//   run   -- everything that reads the activations: the input spectra, the contraction and the inverse transform,
//            with per-call scratch from torch's allocator. It only reads the plan buffer.
//
// Both sides derive the launch configuration with make_config, a pure function of B*C, the geometry and the psi
// sizes, so a plan carries no host state of its own; run checks that the buffer it is given has the size its
// configuration expects. Kernels and their launch parameters are those of the candidate. The differences: the plan's
// kernels run without programmatic dependent launch (their dependents are in another call), the input spectra launch
// without it too (their predecessor is the caller's kernel), and the SM count comes from torch instead of a static
// cache that ignored the device.
// =====================================================================================================================

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

namespace
{

    // how the scatter runs: the 64-bit direct kernel on raw psi (no plan), the direct gather on the planned entry
    // lists, or the frequency-space path
    enum ScatterMode : int { MODE_DIRECT64 = 0, MODE_DIRECT = 1, MODE_FFT = 2 };

    struct ScatterFftConfig {
        int mode = MODE_DIRECT64;
        // sizes, in the kernel's naming: input (BC, K, Hi, Wi), output (BC, Ho, Wo)
        int64_t BC64 = 0, K64 = 0, Hi64 = 0, Wi64 = 0, Ho64 = 0, Wo64 = 0, nrows64 = 0, nseg64 = 0, nnz64 = 0;
        int64_t KH64 = 0, nout_rows64 = 0;
        int BC = 0, K = 0, Hi = 0, Wi = 0, Ho = 0, Wo = 0, KH = 0, nrows = 0, nseg = 0;
        int ps = 0, ni = 0, no = 0, Mi = 0, Mf = 0;
        Plan2 pi2 {}, po2 {}, pp2 {};
        PlanM pim {}, pom {};
        bool i2 = false, o2 = false, fft = false, fast = false, pair = false, pack = false;
        int LC = 0, NCG = 1, BCp = 0, SPT = 1, cpt = CPT_FAST, NCPT = 0, NSG = 1;
        int HB = 1, nbands = 0, MB = 1, NO = 1, nslot = 0, NMT = 1, MPi = 0, PLD = 0;
        int nsm = 148;
        // plan buffer layout (bytes) and per-call scratch sizes
        size_t o_segval = 0, o_cnt = 0, o_boff = 0, o_bcur = 0, o_ent = 0, o_aty = 0, o_ti = 0, o_to = 0, o_P = 0;
        size_t plan_bytes = 0, x_bytes = 0, out_bytes = 0;
    };

    ScatterFftConfig make_config(int64_t BC64, int64_t K64, int64_t Hi64, int64_t Wi64, int64_t Ho64, int64_t Wo64,
                                 int64_t nrows64, int64_t nseg64, int64_t nnz64, int nsm)
    {
        ScatterFftConfig c;
        c.BC64 = BC64;
        c.K64 = K64;
        c.Hi64 = Hi64;
        c.Wi64 = Wi64;
        c.Ho64 = Ho64;
        c.Wo64 = Wo64;
        c.nrows64 = nrows64;
        c.nseg64 = nseg64;
        c.nnz64 = nnz64;
        c.nsm = nsm > 0 ? nsm : 148;
        if (BC64 <= 0 || K64 <= 0 || Hi64 <= 0 || Wi64 <= 0 || Ho64 <= 0 || Wo64 <= 0) return c;

        int64_t in_rows64, in_elems64, out_elems64, seg_elems64;
        const bool products_ok = mul_i64(K64, Hi64, c.KH64) && mul_i64(BC64, c.KH64, in_rows64)
            && mul_i64(in_rows64, Wi64, in_elems64) && mul_i64(BC64, Ho64, c.nout_rows64)
            && mul_i64(c.nout_rows64, Wo64, out_elems64) && mul_i64(nseg64, 3, seg_elems64);
        TORCH_CHECK(products_ok, "DISCO tensor extent exceeds signed 64-bit addressing");

        const bool wide = BC64 > INT_MAX - 64 || c.KH64 > INT_MAX || Hi64 > INT_MAX || Wi64 > INT_MAX - 1024
            || Ho64 > INT_MAX - 1024 || Wo64 > INT_MAX - 1024 || nrows64 > INT_MAX - 64 || nseg64 > INT_MAX - 64
            || nnz64 > INT_MAX || Wi64 + Wo64 > INT_MAX - 64 || in_rows64 > INT_MAX || c.nout_rows64 > INT_MAX;
        if (wide) return c; // MODE_DIRECT64

        const int BC = c.BC = (int)BC64, K = c.K = (int)K64, Hi = c.Hi = (int)Hi64, Wi = c.Wi = (int)Wi64;
        const int Ho = c.Ho = (int)Ho64, Wo = c.Wo = (int)Wo64, KH = c.KH = (int)c.KH64;
        const int nrows = c.nrows = (int)nrows64, nseg = c.nseg = (int)nseg64;
        (void)K;
        (void)Hi;

        const int ps = c.ps = (Wi > 0) ? Wo / Wi : 0;
        const int ni = c.ni = Wi / 2, no = c.no = Wo / 2;
        const bool even = Wi > 0 && (Wi % 2 == 0) && (Wo % 2 == 0) && ps >= 1 && ps * Wi == Wo;
        const bool i2 = c.i2 = even && make_plan2(ni, c.pi2);
        const bool o2 = c.o2 = even && make_plan2(no, c.po2);
        c.pp2 = c.po2; // psi spectra: sparse first pass prefers the larger first radix
        if (o2) make_plan2(no, c.pp2, true);
        const bool im = even && !i2 && make_planm(ni, c.pim), om = even && !o2 && make_planm(no, c.pom);
        const bool fft = c.fft = even && (i2 || im) && (o2 || om) && no <= 8192;
        const int Mi = c.Mi = ni + 1, Mf = c.Mf = no + 1;

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
        c.LC = LC;
        c.NCG = NCG;
        const int BCp = c.BCp = LC * NCG;
        // contraction block: LC / CPT channel threads x NSG slot groups (largest power of two keeping the block
        // <= 288 threads and a whole number of warps), SPT slots per group
        const int SPT = c.SPT = (ps == 1) ? 2 : 1;
        const int nslot0 = ps == 2 ? Wi / 2 + 1 : Mf;
        auto pick_nsg = [](int ncpt) {
            int g = 32;
            while (g > 1 && (ncpt * g > 288 || (ncpt * g) % 32 != 0)) g >>= 1;
            if (ncpt * g > 128 && g > 1 && (ncpt * (g >> 1)) % 32 == 0) g >>= 1;
            return g;
        };
        // Four channels amortize staged-entry and psi-spectrum traffic once there are enough real channels to fill the
        // consumer groups. Keep the two-channel path for small tails, where CPT4's padding and register footprint
        // outweigh that reuse.
        int cpt = ps == 2 && BC >= 32 ? 4 : CPT_FAST;
        int NCPT = LC / 4;
        int NSG = pick_nsg(NCPT);
        if (cpt != 4 || LC % 4 != 0 || NSG < 8 || (NCPT * NSG) % 32 != 0) {
            cpt = CPT_FAST;
            NCPT = LC / CPT_FAST;
            NSG = pick_nsg(NCPT);
        }
        int hbf = cpt == 4 ? 4 : HB_FAST;
        // At high channel counts, choose slot grouping and band height from a runtime residency/work model. This remains
        // smooth across neighbouring channel counts and grid resolutions; it does not dispatch on a benchmark shape.
        if (cpt == CPT_FAST && BC >= 256) {
            const int SPTq = (ps == 1) ? 2 : 1;
            const int NOq = (ps == 2) ? 2 : 1;
            auto cres = [&](int nsg, int hb, int regs) {
                const int wpb = (NCPT * nsg + 32) / 32;
                const int mb = nsg * SPTq;
                const int stg = (2 + mb * LC + hb * mb * NOq) * (int)sizeof(float2) + 16;
                return max(
                    1, min(min(4 * (16384 / (32 * regs)) / wpb, 2048 / (32 * wpb)), (227 * 1024) / (4 * stg + 2048)));
            };
            int bestw = -1;
            for (int g = 32; g >= 1; g >>= 1) {
                if (NCPT * g > 288 || (NCPT * g) % 32 != 0) continue;
                const int w = cres(g, 8, 96) * (NCPT * g / 32);
                if (w > bestw) {
                    bestw = w;
                    NSG = g;
                }
            }
            // Keep the extra HB4/HB5 instantiations narrow: they are profitable when pscale 1 selects the 8-slot
            // grouping, where their smaller accumulator footprint raises residency from two to three CTAs per SM. HB5
            // additionally requires the full CTA (consumers plus producer) to fit its 288-thread launch bound.
            if (fft && ps == 1 && NSG == 8) {
                const int wpb = (NCPT * NSG + 32) / 32;
                const double apr = nrows > 0 ? (double)nseg / (double)nrows : 1.0;
                const double s8 = cres(NSG, 8, 96) * wpb / (1.0 + (apr - 1.0) / 8.0);
                const double s4 = cres(NSG, 4, 64) * wpb / (1.0 + (apr - 1.0) / 4.0);
                const double s5 = NCPT * NSG + 32 <= 288 ? cres(NSG, 5, 72) * wpb / (1.0 + (apr - 1.0) / 5.0) : -1.0;
                if (s5 > s8 && s5 > s4)
                    hbf = 5;
                else if (s4 > s8)
                    hbf = 4;
            }
        }
        c.cpt = cpt;
        c.NCPT = NCPT;
        c.NSG = NSG;
        const bool fast = c.fast = fft && (ps == 1 || ps == 2) && (NCPT * NSG) % 32 == 0;
        // pscale 1 with two-pass forward and inverse FFTs: pair slot layout (slots 2q, 2q + 1 = frequencies q, n - q)
        // so the contraction epilogue can emit the packed inverse input; pscale 2 planes already are such pairs. Used
        // only while the input rows do not dominate (KH <= 2 Ho).
        const bool pair = c.pair = fast && ps == 1 && i2 && o2 && c.KH64 <= 2 * Ho64;
        c.pack = fast && o2 && (ps == 2 || pair);
        const int HB = c.HB = fast ? hbf : (fft ? 8 : 1);
        const int nbands = c.nbands = (Ho + HB - 1) / HB;
        const int MB = c.MB = (fast ? NSG : 1) * SPT;
        const int NO = c.NO = (fast && ps == 2) ? 2 : 1;
        const int nslot = c.nslot = fast ? (pair ? 2 * (no / 2 + 1) : nslot0) : Mf;
        const int NMT = c.NMT = fast ? (nslot + MB - 1) / MB : 1;
        const int MPi = c.MPi = fast ? NMT * MB : Mi;
        const int PLD = c.PLD = fast ? NMT * MB : (Mf + 1) / 2 * 2;

        const int64_t fwd_cells = (int64_t)BCp * KH;
        const int64_t freq_cells = (int64_t)Mf * LC;
        bool fft_grid_ok = true;
        if (fft) {
            fft_grid_ok &= NCG <= 65535;
            fft_grid_ok &= i2 ? (((LC + 7) / 8) <= 65535) : (fwd_cells <= INT_MAX);
            fft_grid_ok &= o2 ? (Ho <= 65535) : (c.nout_rows64 <= INT_MAX);
            if (fast)
                fft_grid_ok &= NMT <= 65535;
            else
                fft_grid_ok &= nbands <= 65535 && freq_cells <= INT_MAX;
        }
        if (!fft_grid_ok) return c; // MODE_DIRECT64

        c.mode = fft ? MODE_FFT : MODE_DIRECT;
        size_t off = 0;
        c.o_segval = off;
        off = align_up(off + sizeof(int64_t) * (size_t)max(nseg, 1));
        c.o_cnt = off;
        off = align_up(off + sizeof(int) * (size_t)(nbands + 1));
        c.o_boff = off;
        off = align_up(off + sizeof(int) * (size_t)(nbands + 1));
        c.o_bcur = off;
        off = align_up(off + sizeof(int) * (size_t)(nbands + 1));
        c.o_ent = off;
        off = align_up(off + sizeof(int4) * (size_t)max(nseg, 1));
        c.o_aty = off;
        off = align_up(off + sizeof(int) * (size_t)max(nseg, 1));
        if (fft) {
            c.o_ti = off;
            off = align_up(off + sizeof(float2) * (size_t)Wi);
            c.o_to = off;
            off = align_up(off + sizeof(float2) * (size_t)Wo);
            c.o_P = off;
            off = align_up(off + sizeof(float2) * (size_t)max(nseg, 1) * NO * PLD);
            c.x_bytes = sizeof(float2) * (size_t)NCG * KH * MPi * LC;
            c.out_bytes = sizeof(float2) * (size_t)Ho * Mf * BCp;
        }
        c.plan_bytes = off;
        return c;
    }

    void check_arcs(const at::Tensor &row_ker, const at::Tensor &row_lat, const at::Tensor &seg_off,
                    const at::Tensor &seg, const at::Tensor &val_off, const at::Tensor &vals)
    {
        const auto dev = row_ker.device();
        TORCH_CHECK(dev.is_cuda(), "scatter_fft_sm100: psi must be on a CUDA device");
        auto check = [&](const at::Tensor &t, at::ScalarType dtype, const char *name) {
            TORCH_CHECK(t.device() == dev, "scatter_fft_sm100: ", name, " is on ", t.device(), ", expected ", dev);
            TORCH_CHECK(t.scalar_type() == dtype, "scatter_fft_sm100: ", name, " must be ", dtype, ", got ",
                        t.scalar_type());
            TORCH_CHECK(t.is_contiguous(), "scatter_fft_sm100: ", name, " must be contiguous");
        };
        check(row_ker, at::kInt, "row_ker");
        check(row_lat, at::kInt, "row_lat");
        check(seg_off, at::kLong, "seg_off");
        check(seg, at::kInt, "seg");
        check(val_off, at::kLong, "val_off");
        check(vals, at::kFloat, "vals");
        TORCH_CHECK(seg.dim() == 2 && seg.size(1) == 3, "scatter_fft_sm100: seg must be (nseg, 3), got ", seg.sizes());
        TORCH_CHECK(seg_off.numel() == row_ker.numel() + 1 && val_off.numel() == row_ker.numel() + 1,
                    "scatter_fft_sm100: seg_off and val_off must have one entry per row plus one");
        TORCH_CHECK(row_lat.numel() == row_ker.numel(), "scatter_fft_sm100: row_lat must have one entry per row");
    }

    int sm_count() { return at::cuda::getCurrentDeviceProperties()->multiProcessorCount; }

    ScatterFftConfig config_for(int64_t BC, int64_t K, int64_t Hi, int64_t Wi, int64_t Ho, int64_t Wo,
                                const at::Tensor &row_ker, const at::Tensor &seg, const at::Tensor &vals)
    {
        TORCH_CHECK(Wi > 0 && Wo % Wi == 0 && Wo / Wi >= 1, "scatter_fft_sm100: the output longitudes (", Wo,
                    ") must be a multiple of the input longitudes (", Wi, ")");
        return make_config(BC, K, Hi, Wi, Ho, Wo, row_ker.numel(), seg.size(0), vals.numel(), sm_count());
    }

} // namespace

namespace disco_kernels
{

    // The plan of the scatter of (BC, K, Hi, Wi) planes onto (BC, Ho, Wo) through psi in arc form.
    at::Tensor scatter_fft_sm100_plan(at::Tensor row_ker, at::Tensor row_lat, at::Tensor seg_off, at::Tensor seg,
                                      at::Tensor val_off, at::Tensor vals, int64_t BC, int64_t K, int64_t Hi,
                                      int64_t Wi, int64_t Ho, int64_t Wo)
    {
        check_arcs(row_ker, row_lat, seg_off, seg, val_off, vals);
        const c10::cuda::CUDAGuard guard(row_ker.device());
        const ScatterFftConfig c = config_for(BC, K, Hi, Wi, Ho, Wo, row_ker, seg, vals);
        auto plan = at::empty({(int64_t)c.plan_bytes}, row_ker.options().dtype(at::kByte));
        if (c.mode == MODE_DIRECT64 || c.plan_bytes == 0) return plan;

        cudaStream_t st = at::cuda::getCurrentCUDAStream();
        const int *rk = row_ker.data_ptr<int>();
        const int *rl = row_lat.data_ptr<int>();
        const int64_t *so = seg_off.data_ptr<int64_t>();
        const int *sg = seg.data_ptr<int>();
        const int64_t *vo = val_off.data_ptr<int64_t>();
        const float *vp = vals.data_ptr<float>();

        char *base = static_cast<char *>(plan.data_ptr());
        int64_t *segval = reinterpret_cast<int64_t *>(base + c.o_segval);
        int *cnt = reinterpret_cast<int *>(base + c.o_cnt);
        int *boff = reinterpret_cast<int *>(base + c.o_boff);
        int *bcur = reinterpret_cast<int *>(base + c.o_bcur);
        int4 *ent = reinterpret_cast<int4 *>(base + c.o_ent);
        int *aty = reinterpret_cast<int *>(base + c.o_aty);
        float2 *Ti = c.fft ? reinterpret_cast<float2 *>(base + c.o_ti) : nullptr;
        float2 *To = c.fft ? reinterpret_cast<float2 *>(base + c.o_to) : nullptr;
        float2 *P = c.fft ? reinterpret_cast<float2 *>(base + c.o_P) : nullptr;
        const int nrows = c.nrows, nseg = c.nseg, Hi_ = c.Hi, Ho_ = c.Ho, Wi_ = c.Wi, Wo_ = c.Wo, HB = c.HB;

        C10_CUDA_CHECK(cudaMemsetAsync(cnt, 0, sizeof(int) * (size_t)(c.nbands + 1), st));
        {
            // one warp per psi row, plus twiddle-table blocks
            const int rb = (nrows + PREP_T / 32 - 1) / (PREP_T / 32);
            const int tb = c.fft ? (Wo_ + PREP_T - 1) / PREP_T : 0;
            auto go = [&](auto HBC) {
                constexpr int HBT = decltype(HBC)::value;
                prep_count_kernel<HBT><<<max(rb + tb, 1), PREP_T, 0, st>>>(
                    so, sg, vo, nrows, Ho_, segval, cnt, Ti, c.fft ? Wi_ : 0, To, c.fft ? Wo_ : 0, c.nbands, boff, bcur);
            };
            if (HB == 1)
                go(IC<1> {});
            else if (HB == 4)
                go(IC<4> {});
            else if (HB == 5)
                go(IC<5> {});
            else
                go(IC<8> {});
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        // psi arc spectra (the two-pass kernel also classifies the arc spectra for the fast contraction)
        if (c.fft && nseg > 0) {
            const bool fast = c.fast, pair = c.pair;
            const int MB = c.MB;
            if (c.o2) {
                const Plan2 &pp2 = c.pp2;
                const size_t smem = sizeof(float2) * PSIW_WARPS * (size_t)pp2.SROW;
                const int nblk = (nseg + PSIW_WARPS - 1) / PSIW_WARPS;
                auto go = [&](auto kern) {
                    if (smem > 48 * 1024)
                        C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
                    kern<<<nblk, 32 * PSIW_WARPS, smem, st>>>(sg, segval, vp, P, nseg, pp2, To, c.NO, c.nslot, c.PLD,
                                                              fast ? MB : c.PLD, fast ? __builtin_ctz((unsigned)MB) : -1,
                                                              fast ? aty : nullptr, pair ? 1 : 0, false);
                };
                if (in_tier_m(pp2.N1) && in_tier_m(pp2.N2))
                    go(fwdpsiw_kernel<TIER_M, 6>);
                else if (in_tier_s(pp2.N1) && in_tier_s(pp2.N2))
                    go(fwdpsiw_kernel<TIER_S, 6>);
                else if (in_tier_n(pp2.N1) && in_tier_n(pp2.N2))
                    go(fwdpsiw_kernel<TIER_N, 8>);
                else if (in_tier_l(pp2.N1) && in_tier_l(pp2.N2))
                    go(fwdpsiw_kernel<TIER_L, 8>);
                else if (max(pp2.N1, pp2.N2) <= 20)
                    go(fwdpsiw_kernel<20, 6>);
                else if (max(pp2.N1, pp2.N2) <= 30)
                    go(fwdpsiw_kernel<30, 6>);
                else
                    go(fwdpsiw_kernel<32, 6>);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            } else {
                auto pick_nb
                    = [](int n) { return n <= 256 ? 16 : (n <= 512 ? 8 : (n <= 1024 ? 4 : (n <= 2048 ? 2 : 1))); };
                const int NB = pick_nb(c.no);
                const size_t smem = sizeof(float2) * 2 * (size_t)NB * c.no;
                if (smem > 48 * 1024)
                    C10_CUDA_CHECK(
                        cudaFuncSetAttribute(fwdpsim_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
                fwdpsim_kernel<<<(nseg + NB - 1) / NB, 256, smem, st>>>(sg, segval, vp, P, nseg, c.pom, To, NB, Wi_,
                                                                        c.NO, c.nslot, c.PLD, fast ? MB : c.PLD, fast);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                if (fast) {
                    arc_type_kernel<<<(nseg + 31) / 32, 256, 0, st>>>(sg, segval, vp, aty, nseg, Wo_);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                }
            }
        }

        // band entry lists; the fast contraction's carry the arc types
        if (nrows > 0) {
            const int *types = c.fast && nseg > 0 ? aty : nullptr;
            auto go = [&](auto HBC) {
                constexpr int HBT = decltype(HBC)::value;
                prep_fill_kernel<HBT>
                    <<<max((nrows + 7) / 8, 1), 256, 0, st>>>(so, sg, rk, rl, nrows, Hi_, Ho_, bcur, ent, types, false);
            };
            if (HB == 1)
                go(IC<1> {});
            else if (HB == 4)
                go(IC<4> {});
            else if (HB == 5)
                go(IC<5> {});
            else
                go(IC<8> {});
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return plan;
    }

    // The scatter of the contiguous fp16 (B, C, K, Hi, Wi) inp onto (B, C, Ho, Wo) through psi in arc form, with the
    // plan scatter_fft_sm100_plan built for B * C.
    at::Tensor scatter_fft_sm100_run(at::Tensor inp, at::Tensor plan, at::Tensor row_ker, at::Tensor row_lat,
                                     at::Tensor seg_off, at::Tensor seg, at::Tensor val_off, at::Tensor vals,
                                     int64_t Ho64, int64_t Wo64)
    {
        check_arcs(row_ker, row_lat, seg_off, seg, val_off, vals);
        TORCH_CHECK(inp.device() == row_ker.device(), "scatter_fft_sm100: inp is on ", inp.device(), ", psi on ",
                    row_ker.device());
        TORCH_CHECK(inp.scalar_type() == at::kHalf, "scatter_fft_sm100: inp must be float16, got ", inp.scalar_type());
        TORCH_CHECK(inp.dim() == 5 && inp.is_contiguous(),
                    "scatter_fft_sm100: inp must be a contiguous (B, C, K, H, W) tensor");
        TORCH_CHECK(plan.scalar_type() == at::kByte && plan.device() == inp.device() && plan.is_contiguous(),
                    "scatter_fft_sm100: plan must be a contiguous uint8 tensor on the device of inp");
        const c10::cuda::CUDAGuard guard(inp.device());

        const int64_t B64 = inp.size(0), C64 = inp.size(1);
        auto out = at::empty({B64, C64, Ho64, Wo64}, inp.options());
        if (inp.numel() == 0 || out.numel() == 0) return out.zero_();

        const ScatterFftConfig c
            = config_for(B64 * C64, inp.size(2), inp.size(3), inp.size(4), Ho64, Wo64, row_ker, seg, vals);
        TORCH_CHECK((size_t)plan.numel() == c.plan_bytes, "scatter_fft_sm100: the plan has ", plan.numel(),
                    " bytes, a plan for these shapes and B * C = ", B64 * C64, " has ", c.plan_bytes,
                    "; it was built for other shapes");

        cudaStream_t st = at::cuda::getCurrentCUDAStream();
        const __half *xp = reinterpret_cast<const __half *>(inp.data_ptr<at::Half>());
        __half *op = reinterpret_cast<__half *>(out.data_ptr<at::Half>());
        const int *rk = row_ker.data_ptr<int>();
        const int *rl = row_lat.data_ptr<int>();
        const int64_t *so = seg_off.data_ptr<int64_t>();
        const int *sg = seg.data_ptr<int>();
        const int64_t *vo = val_off.data_ptr<int64_t>();
        const float *vp = vals.data_ptr<float>();

        if (c.mode == MODE_DIRECT64) {
            launch_direct64(st, xp, rk, rl, so, sg, vo, vp, op, c.nrows64, c.Wi64, c.Hi64, c.Ho64, c.Wo64, c.KH64,
                            c.nout_rows64);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return out;
        }

        const char *base = static_cast<const char *>(plan.data_ptr());
        const int64_t *segval = reinterpret_cast<const int64_t *>(base + c.o_segval);
        const int *boff = reinterpret_cast<const int *>(base + c.o_boff);
        const int4 *ent = reinterpret_cast<const int4 *>(base + c.o_ent);
        const int BC = c.BC, Wi = c.Wi, Ho = c.Ho, Wo = c.Wo, KH = c.KH;

        if (c.mode == MODE_DIRECT) {
            const long nrow = (long)BC * Ho;
            const unsigned nblk = (unsigned)min(nrow, (long)1 << 20);
            const size_t smem = sizeof(float) * ((size_t)Wo + (size_t)max(Wi, 1));
            if (smem <= 160 * 1024) {
                if (smem > 48 * 1024)
                    C10_CUDA_CHECK(cudaFuncSetAttribute(direct_kernel<true>,
                                                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
                direct_kernel<true><<<nblk, 256, smem, st>>>(xp, ent, boff, sg, segval, vp, op, Wi, Ho, Wo, KH, nrow);
            } else {
                direct_kernel<false><<<nblk, 256, 0, st>>>(xp, ent, boff, sg, segval, vp, op, Wi, Ho, Wo, KH, nrow);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return out;
        }

        const float2 *Ti = reinterpret_cast<const float2 *>(base + c.o_ti);
        const float2 *To = reinterpret_cast<const float2 *>(base + c.o_to);
        const float2 *P = reinterpret_cast<const float2 *>(base + c.o_P);
        // per-call scratch: the input spectra and the contraction's result
        auto scratch = at::empty({(int64_t)(align_up(c.x_bytes) + c.out_bytes)}, inp.options().dtype(at::kByte));
        float2 *X = reinterpret_cast<float2 *>(scratch.data_ptr());
        float2 *OUT = reinterpret_cast<float2 *>(static_cast<char *>(scratch.data_ptr()) + align_up(c.x_bytes));

        const bool fast = c.fast, pair = c.pair, pack = c.pack;
        const int LC = c.LC, NCG = c.NCG, BCp = c.BCp, MB = c.MB, NO = c.NO, MPi = c.MPi, Mf = c.Mf, nslot = c.nslot;
        const int NSG = c.NSG, NCPT = c.NCPT, HB = c.HB, nseg = c.nseg, ni = c.ni, no = c.no, ps = c.ps;
        const Plan2 &pi2 = c.pi2, &po2 = c.po2;
        const PlanM &pim = c.pim, &pom = c.pom;
        auto pick_nb = [](int n) { return n <= 256 ? 16 : (n <= 512 ? 8 : (n <= 1024 ? 4 : (n <= 2048 ? 2 : 1))); };
        (void)NO;

        // input spectra
        const int MBX
            = fast ? MB : 1; // X tile height (slots); 1 = plain frequency-major rows for the generic contraction
        const int mbx_sh = __builtin_ctz((unsigned)MBX); // MB = NSG * SPT is a power of two
        if (c.i2) {
            const int SW = stage_words(ni);
            const size_t sb = max(sizeof(float2) * 8 * (size_t)pi2.N1 * pi2.LD, sizeof(float2) * 8 * (size_t)ni);
            const size_t xb = sizeof(uint32_t) * 8 * (size_t)SW;
            const dim3 grid(KH, (LC + 7) / 8, NCG);
            // staged rows get their own region when that keeps the residency of the instantiation (no extra barrier)
            auto go = [&](auto kern, int minb) {
                const bool sep = (sb + xb + 1024) * minb <= 227 * 1024;
                const size_t smem = sep ? sb + xb : max(sb, xb);
                if (smem > 48 * 1024)
                    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
                kern<<<grid, pi2.NT, smem, st>>>(xp, X, BC, KH, pi2, Ti, LC, MPi, mbx_sh, SW, sep ? (int)(sb / 4) : 0,
                                                 pair ? 1 : 0, false);
            };
            if (in_tier_m(pi2.N1) && in_tier_m(pi2.N2)) {
                const bool qsplit = BC <= 32 && NCG == 1 && LC % 8 == 4;
                if (BC >= 256)
                    go(fwdx3_kernel<TIER_M, 160, 7>, 7);
                else if (qsplit) {
                    const int nfull = LC / 8;
                    const size_t sb4 = max(sizeof(float2) * 4 * (size_t)pi2.N1 * pi2.LD, sizeof(float2) * 4 * (size_t)ni);
                    const size_t xb4 = sizeof(uint32_t) * 4 * (size_t)SW;
                    const bool sep4 = (sb4 + xb4 + 1024) * 12 <= 227 * 1024;
                    const size_t smem4 = sep4 ? sb4 + xb4 : max(sb4, xb4);
                    auto qkern = fwdx4q_kernel<TIER_M, 80, 12>;
                    if (smem4 > 48 * 1024)
                        C10_CUDA_CHECK(
                            cudaFuncSetAttribute(qkern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem4));
                    // the quarter-channel kernel releases the full-channel one, which overlaps it
                    cudaLaunchConfig_t cfg = {};
                    cfg.gridDim = dim3(KH, 1, 1);
                    cfg.blockDim = dim3(4 * max(pi2.N1, pi2.N2));
                    cfg.dynamicSmemBytes = smem4;
                    cfg.stream = st;
                    cfg.attrs = nullptr;
                    cfg.numAttrs = 0;
                    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, qkern, xp, X, BC, KH, pi2, Ti, LC, MPi, mbx_sh, SW,
                                                      sep4 ? (int)(sb4 / 4) : 0, pair ? 1 : 0, nfull, false, nfull > 0));
                    if (nfull > 0) {
                        const bool sep8 = (sb + xb + 1024) * 8 <= 227 * 1024;
                        const size_t smem8 = sep8 ? sb + xb : max(sb, xb);
                        auto fullkern = fwdx3_kernel<TIER_M, 160, 8>;
                        if (smem8 > 48 * 1024)
                            C10_CUDA_CHECK(cudaFuncSetAttribute(fullkern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                                (int)smem8));
                        cudaLaunchConfig_t cfg8 = {};
                        cudaLaunchAttribute attr[1];
                        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
                        attr[0].val.programmaticStreamSerializationAllowed = 1;
                        cfg8.gridDim = dim3(KH, nfull, 1);
                        cfg8.blockDim = dim3(pi2.NT);
                        cfg8.dynamicSmemBytes = smem8;
                        cfg8.stream = st;
                        cfg8.attrs = attr;
                        cfg8.numAttrs = 1;
                        C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg8, fullkern, xp, X, BC, KH, pi2, Ti, LC, MPi, mbx_sh, SW,
                                                          sep8 ? (int)(sb / 4) : 0, pair ? 1 : 0, true));
                    }
                } else
                    go(fwdx3_kernel<TIER_M, 160, 8>, 8);
            } else if (in_tier_s(pi2.N1) && in_tier_s(pi2.N2)) {
                if (BC >= 256)
                    go(fwdx3_kernel<TIER_S, 160, 7>, 7);
                else
                    go(fwdx3_kernel<TIER_S, 160, 8>, 8);
            } else if (in_tier_n(pi2.N1) && in_tier_n(pi2.N2))
                go(fwdx3_kernel<TIER_N, 240, 4>, 4);
            else if (in_tier_l(pi2.N1) && in_tier_l(pi2.N2))
                go(fwdx3_kernel<TIER_L, 240, 4>, 4);
            else if (max(pi2.N1, pi2.N2) <= 20)
                go(fwdx3_kernel<20, 160, 6>, 6);
            else if (max(pi2.N1, pi2.N2) <= 30)
                go(fwdx3_kernel<30, 240, 4>, 4);
            else
                go(fwdx3_kernel<32, 256, 3>, 3);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            const int NB = pick_nb(ni);
            const size_t smem = sizeof(float2) * 2 * (size_t)NB * ni;
            if (smem > 48 * 1024)
                C10_CUDA_CHECK(cudaFuncSetAttribute(fwdxm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
            const long nxr = (long)BCp * KH;
            fwdxm_kernel<<<(unsigned)((nxr + NB - 1) / NB), 256, smem, st>>>(xp, X, BC, BCp, KH, pim, Ti, NB, LC, MPi,
                                                                             MBX);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        // contraction (with pack the epilogue emits the inverse complex-FFT input, see PACK)
        if (fast) {
            const int threads = NCPT * NSG + 32;
            const int STG = 2 + MB * LC + HB * MB * c.NO;
            const int nstg = max(4, min(MAXSTG, (100 * 1024) / (int)(sizeof(float2) * STG + 16)));
            const size_t smem = (sizeof(float2) * (size_t)STG + 16) * nstg;
            dim3 grid(c.nbands, NCG, c.NMT);
            if (c.cpt == 4) {
                if (pack)
                    dispatch_contract4<true>(NSG, grid, threads, smem, st, X, P, ent, boff, OUT, LC, KH, MPi, nseg, Ho,
                                             Mf, Wi, BCp, nslot, nstg, To);
                else
                    dispatch_contract4<false>(NSG, grid, threads, smem, st, X, P, ent, boff, OUT, LC, KH, MPi, nseg, Ho,
                                              Mf, Wi, BCp, nslot, nstg, nullptr);
            } else if (HB == 5 && pack) {
                launch_contract<1, 5, 8, true, true, 288, 3>(grid, threads, smem, st, X, P, ent, boff, OUT, LC, KH, MPi,
                                                             nseg, Ho, Mf, Wi, BCp, nslot, nstg, To);
            } else if (HB == 5) {
                launch_contract<1, 5, 8, true, false, 288, 3>(grid, threads, smem, st, X, P, ent, boff, OUT, LC, KH,
                                                              MPi, nseg, Ho, Mf, Wi, BCp, nslot, nstg, nullptr);
            } else if (HB <= 4 && pack) {
                launch_contract<1, 4, 8, true, true>(grid, threads, smem, st, X, P, ent, boff, OUT, LC, KH, MPi, nseg,
                                                     Ho, Mf, Wi, BCp, nslot, nstg, To);
            } else if (HB <= 4) {
                launch_contract<1, 4, 8, true, false>(grid, threads, smem, st, X, P, ent, boff, OUT, LC, KH, MPi, nseg,
                                                      Ho, Mf, Wi, BCp, nslot, nstg, nullptr);
            } else if (ps == 1 && pack) {
                if (BC >= 256)
                    dispatch_contract<1, true, true>(NSG, grid, threads, smem, st, X, P, ent, boff, OUT, LC, KH, MPi,
                                                     nseg, Ho, Mf, Wi, BCp, nslot, nstg, To);
                else
                    dispatch_contract<1, false, true>(NSG, grid, threads, smem, st, X, P, ent, boff, OUT, LC, KH, MPi,
                                                      nseg, Ho, Mf, Wi, BCp, nslot, nstg, To);
            } else if (ps == 1) {
                if (BC >= 256)
                    dispatch_contract<1, true>(NSG, grid, threads, smem, st, X, P, ent, boff, OUT, LC, KH, MPi, nseg,
                                               Ho, Mf, Wi, BCp, nslot, nstg);
                else
                    dispatch_contract<1>(NSG, grid, threads, smem, st, X, P, ent, boff, OUT, LC, KH, MPi, nseg, Ho, Mf,
                                         Wi, BCp, nslot, nstg);
            } else if (pack)
                dispatch_contract<2, false, true>(NSG, grid, threads, smem, st, X, P, ent, boff, OUT, LC, KH, MPi, nseg,
                                                  Ho, Mf, Wi, BCp, nslot, nstg, To);
            else
                dispatch_contract<2>(NSG, grid, threads, smem, st, X, P, ent, boff, OUT, LC, KH, MPi, nseg, Ho, Mf, Wi,
                                     BCp, nslot, nstg);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            dim3 grid((Mf * LC + 255) / 256, c.nbands, NCG);
            contract_gen_kernel<8><<<grid, 256, 0, st>>>(X, P, ent, boff, sg, OUT, LC, KH, MPi, c.PLD, Ho, Mf, Wi, BCp);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        // inverse
        if (c.o2) {
            const int SW = stage_words(no);
            const size_t smem = max(max(sizeof(float2) * 8 * (size_t)po2.N1 * po2.LD, sizeof(float2) * 8 * (size_t)Mf),
                                    sizeof(uint32_t) * 8 * (size_t)SW);
            const dim3 grid((BCp + 7) / 8, Ho);
            auto go = [&](auto kern) {
                if (smem > 48 * 1024)
                    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
                launch_pdl(kern, grid, dim3(po2.NT), smem, st, (const float2 *)OUT, op, BC, BCp, Ho, po2,
                           (const float2 *)To, SW);
            };
            if (in_tier_m(po2.N1) && in_tier_m(po2.N2)) {
                if (pack)
                    go(inv3_kernel<TIER_M, 160, 7, true>);
                else
                    go(inv3_kernel<TIER_M, 160, 7>);
            } else if (in_tier_s(po2.N1) && in_tier_s(po2.N2)) {
                if (pack)
                    go(inv3_kernel<TIER_S, 160, 7, true>);
                else
                    go(inv3_kernel<TIER_S, 160, 7>);
            } else if (in_tier_n(po2.N1) && in_tier_n(po2.N2)) {
                const bool quartets = pack && (int64_t)((BCp + 7) / 8) * Ho < (int64_t)3 * 4 * c.nsm;
                if (pack && quartets) {
                    auto kern = inv4q_kernel<TIER_N, 120, 8>;
                    const size_t smem4
                        = max(max(sizeof(float2) * 4 * (size_t)po2.N1 * po2.LD, sizeof(float2) * 4 * (size_t)no),
                              sizeof(uint32_t) * 4 * (size_t)SW);
                    if (smem4 > 48 * 1024)
                        C10_CUDA_CHECK(
                            cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem4));
                    const float sc = 0.125f / (float)no;
                    launch_pdl(kern, dim3((BCp + 7) / 8 * 2, Ho), dim3(4 * max(po2.N1, po2.N2)), smem4, st,
                               (const float2 *)OUT, op, BC, BCp, Ho, po2, (const float2 *)To, SW, sc);
                } else if (pack) {
                    go(inv3_kernel<TIER_N, 240, 4, true>);
                } else {
                    go(inv3_kernel<TIER_N, 240, 4>);
                }
            } else if (in_tier_l(po2.N1) && in_tier_l(po2.N2)) {
                if (pack)
                    go(inv3_kernel<TIER_L, 240, 4, true>);
                else
                    go(inv3_kernel<TIER_L, 240, 4>);
            } else if (max(po2.N1, po2.N2) <= 20) {
                if (pack)
                    go(inv3_kernel<20, 160, 6, true>);
                else
                    go(inv3_kernel<20, 160, 6>);
            } else if (max(po2.N1, po2.N2) <= 30) {
                if (pack)
                    go(inv3_kernel<30, 240, 4, true>);
                else
                    go(inv3_kernel<30, 240, 4>);
            } else {
                if (pack)
                    go(inv3_kernel<32, 256, 3, true>);
                else
                    go(inv3_kernel<32, 256, 3>);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            const int NB = pick_nb(no);
            const size_t smem = sizeof(float2) * 2 * (size_t)NB * no;
            if (smem > 48 * 1024)
                C10_CUDA_CHECK(cudaFuncSetAttribute(invm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
            const long nout = (long)BC * Ho;
            invm_kernel<<<(unsigned)((nout + NB - 1) / NB), 256, smem, st>>>(OUT, op, BC, BCp, Ho, pom, To, NB);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return out;
    }

    TORCH_LIBRARY_FRAGMENT(disco_kernels, m)
    {
        // the plan of scatter_fft_sm100_run for B * C = bc planes of (K, nlat_in, nlon_in) onto (nlat_out, nlon_out)
        m.def("scatter_fft_sm100_plan(Tensor row_ker, Tensor row_lat, Tensor seg_off, Tensor seg, Tensor val_off, "
              "Tensor vals, int bc, int kernel_size, int nlat_in, int nlon_in, int nlat_out, int nlon_out) -> Tensor");
        // the arc scatter (B, C, K, nlat_in, nlon_in) -> (B, C, nlat_out, nlon_out), float16, through that plan
        m.def("scatter_fft_sm100_run(Tensor inp, Tensor plan, Tensor row_ker, Tensor row_lat, Tensor seg_off, Tensor "
              "seg, "
              "Tensor val_off, Tensor vals, int nlat_out, int nlon_out) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(disco_kernels, CUDA, m)
    {
        m.impl("scatter_fft_sm100_plan", &scatter_fft_sm100_plan);
        m.impl("scatter_fft_sm100_run", &scatter_fft_sm100_run);
    }

} // namespace disco_kernels
