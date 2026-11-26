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

#include "disco_cuda_fwd_hankel_common.cuh"

// ------------------------------------------------------------------------------------------
// wgmma helpers
// ------------------------------------------------------------------------------------------
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
// The configuration of a main-kernel family <DW, NWG, CPS, RG>: NWG consumer warpgroups, each owning TPW M tiles of
// 128 GEMM rows, so MT = NWG * TPW tiles per item; CPS CTAs per SM.  The accumulators are registers, TPW * MN per
// thread: narrow MMAs take several tiles per warpgroup, which issue against the same Toeplitz operand (the sm_100a
// kernel's items hold up to 10 tiles in tensor memory).  The widest families of the sm_100a kernel (row pairs at
// DW 8, MN 144) are not taken.
template <int DW, int NWG_, int CPS = 1, int RG = 1> struct HkCfg {
    static constexpr int MN1 = KB * DW;                 // logical columns per output row
    static constexpr int MN = ((RG * MN1 + 7) / 8) * 8; // exact MMA width across row groups
    static_assert(MN <= 80, "the accumulators of wider MMAs do not fit the register file");
    // M tiles per warpgroup: as many as about 80 accumulators per thread allow at one CTA per SM (170 registers),
    // 56 at two (128 registers), at most 4
    static constexpr int ACC_BUDGET = CPS == 1 ? 80 : 56;
    static constexpr int TPW = ACC_BUDGET / MN < 1 ? 1 : (ACC_BUDGET / MN > 4 ? 4 : ACC_BUDGET / MN);
    static constexpr int NWG = NWG_;     // consumer warpgroups
    static constexpr int MT = NWG * TPW; // M tiles per item
    // One-CTA kernels use a second producer so adjacent pieces can issue their TMA loads in
    // parallel; other configurations keep a single producer to preserve occupancy.
    static constexpr int NPW = CPS == 1 ? 2 : 1;
    static constexpr int THREADS = 128 * (1 + NWG);
};

// The stores of one accumulator row of an m64 half (sub 0: the thread's first row, 1: the row 8 below) to GEMM row
// (channel chan, block b): columns 8 j + 2 (lane % 4) + {0, 1}.  Loops of constant trip count only, so that they
// unroll and the accumulators keep their registers.
template <int DW, int RG, int MN, int R, int SUB>
__device__ __forceinline__ void store_rows(const float (&a)[R], int lane, uint16_t *chan, unsigned kstride,
                                           const HkParams &p, int ho, int b, int sh)
{
    constexpr int MN1 = KB * DW;
#pragma unroll
    for (int j = 0; j < MN / 8; ++j) {
        const unsigned pk = pack2(a[4 * j + 2 * SUB], a[4 * j + 2 * SUB + 1], p.bf16);
        const int n = 8 * j + 2 * (lane & 3); // the pair's first column
        if constexpr (DW >= 2) {
            // DW even: the pair lies in one (row r, basis k) block; columns past RG * MN1 are padding
            const int r = n / MN1;
            const int rn = n - r * MN1;
            const int k = rn / DW;
            const int hr = ho * RG + r;
            if (r < RG && k < p.K && hr < p.Ho) {
                int wo = DW * b + sh + (rn - k * DW);
                if (wo >= p.Wo) wo -= p.Wo;
                uint16_t *dst = chan + ((unsigned)k * kstride + (unsigned)hr * (unsigned)p.Wo);
                if (DW == 2 && sh != 0) {
                    // odd start: the pair straddles a 4-byte boundary, and the seam for the last block
                    dst[wo] = (uint16_t)(pk & 0xFFFFu);
                    dst[wo + 1 == p.Wo ? 0 : wo + 1] = (uint16_t)(pk >> 16);
                } else {
                    *reinterpret_cast<unsigned *>(dst + wo) = pk;
                }
            }
        } else {
            // DW 1: one column per (row, basis function)
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int r = (n + u) / MN1;
                const int k = n + u - r * MN1;
                const int hr = ho * RG + r;
                if (r < RG && k < p.K && hr < p.Ho)
                    chan[(unsigned)k * kstride + (unsigned)hr * (unsigned)p.Wo + (unsigned)b]
                        = (uint16_t)(u ? (pk >> 16) : (pk & 0xFFFFu));
            }
        }
    }
}

// One k-step of a consumer warpgroup: the MMAs of its tiles t (tile w + NWG t, first slot 128 (w + NWG t)) whose
// halves hold stored rows, A from slot a16 + the tile's first slot (K-direction stride alb), all against B descriptor
// bd.  A function rather than a lambda: it must inline, or the accumulators lose their registers.
template <int MN, int TPW, int NWG>
__device__ __forceinline__ void mma_tiles(float (&acc)[TPW][2][Wgmma<MN>::R], const int (&nh)[TPW], int w, unsigned a16,
                                          unsigned alb, uint64_t bd, int sc, bool bf)
{
    // tile by tile, with constant indices (spelled out rather than unrolled: see the epilogue)
#define HK_MMA_TILE(T)                                                                                                 \
    if constexpr (TPW > T) {                                                                                           \
        const unsigned at = a16 + (unsigned)((w + NWG * T) * 128);                                                     \
        if (nh[T] > 0) Wgmma<MN>::mma(acc[T][0], mkdesc(at, alb), bd, sc, bf);                                         \
        if (nh[T] > 1) Wgmma<MN>::mma(acc[T][1], mkdesc(at + 64u, alb), bd, sc, bf);                                   \
    }
    HK_MMA_TILE(0)
    HK_MMA_TILE(1)
    HK_MMA_TILE(2)
    HK_MMA_TILE(3)
#undef HK_MMA_TILE
}

template <int DW, bool TX, int NWG_, int CPS, int RG, int NV, bool Z>
__device__ __forceinline__ void hk_body(const TmapX<NV> &tmap_x, const HkParams &p)
{
    static_assert(NV == 8 || NV == NVMAX);
    using Cfg = HkCfg<DW, NWG_, CPS, RG>;
    constexpr int MN = Cfg::MN;
    constexpr int MN1 = Cfg::MN1;
    constexpr int NWG = Cfg::NWG;
    constexpr int NTH = Cfg::THREADS;
    constexpr int NPW = Cfg::NPW;
    extern __shared__ __align__(1024) unsigned char smem[];
    const int tid = threadIdx.x;
    // broadcast: warp-uniform to ptxas (the wgmma issue below must not sit in a path it deems divergent)
    const int warp = __shfl_sync(0xffffffffu, tid >> 5, 0);
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
    } else if (warp >= 4) {
        // ---------------- consumer warpgroup 1 + w: M tiles m = w + NWG t (t < TPW) ----------------
        // Every warp waits for each stage and releases it (one arrival per warp) once the wgmma group that read
        // it has completed.  One piece's group stays in flight behind the next one's issue where there are three
        // stages or more (deeper measured slower on H100: it holds the stages longer); with two, the wait after
        // each piece lets the producer fill the stage of piece q + 1 (a carried stage is freed with the next piece).
        constexpr int TPW = Cfg::TPW;
        constexpr int R = Wgmma<MN>::R; // accumulator registers per m64 half
        const int w = (warp >> 2) - 1;  // this warpgroup
        const int wq = warp & 3;        // its warp: rows 16 wq .. 16 wq + 15 of each m64 half
        const bool bf = p.bf16 != 0;
        const unsigned z16 = (smem_u32(smem) + (unsigned)p.z_off) >> 4;
        const bool keep = nst >= 3; // one group in flight
        // Output offsets: 64-bit to the channel, 32-bit within it (the host checks K * Ho * Wo < 2^32), so the
        // epilogue keeps 32-bit address arithmetic while the whole output may exceed 2^32 elements.
        const unsigned kstride = (unsigned)p.Ho * (unsigned)p.Wo;
        float acc[TPW][2][R];
        // the stages the in-flight group frees once it has completed (-1: none)
        int rel0 = -1, rel1 = -1;
        auto release = [&]() {
            if (lane == 0) {
                if (rel0 >= 0) mbar_arrive(EMPTY(rel0));
                if (rel1 >= 0) mbar_arrive(EMPTY(rel1));
            }
            rel0 = rel1 = -1;
        };
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
            // broadcast: the branches around the wgmma issue must be warp-uniform to ptxas, or it serializes them
            const int item = __shfl_sync(0xffffffffu, e.x, 0);
            if (item >= p.nitems) break;
            const int np = __shfl_sync(0xffffffffu, e.y, 0);
            int ho, cg;
            decode_item(p, item, ho, cg);
            const int c0 = cg * p.G;
            const int nch = min(p.G, p.BC - c0);
            // The last channel's halo rows are never stored, so exclude them from the tile count.
            const int nrow = (nch - 1) * p.RPC + p.nb;
            const int mtc = (nrow + 127) >> 7;
            // per tile: the m64 halves that hold stored rows
            int nh[TPW];
            bool any = false;
#pragma unroll
            for (int t = 0; t < TPW; ++t) {
                const int m = w + NWG * t;
                nh[t] = m < mtc ? (nrow - 128 * m > 64 ? 2 : 1) : 0;
                any |= nh[t] > 0;
            }

            bool init = false; // whether the accumulators hold this item's partial sums
            // Chunk-packed issue: a k-step covers two 8-longitude chunks.  A piece's odd last chunk is
            // paired with the next piece's first chunk (both operands via an LBO spanning the two
            // stages); a ring wrap reverses both K halves, which leaves A*B unchanged and keeps the LBO positive.
            int pend = -1;           // stage whose last chunk is still pending
            unsigned pa = 0, pb = 0; // its A / B start (16-byte units)
            for (int q = 0; q < np; ++q) {
                mbar_wait(FULL(s), ph); // TMA-written operands: the mbarrier completion orders them
                const int nc = __shfl_sync(0xffffffffu, sks[s], 0); // (broadcast: see the item)
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
                if (any) {
                    wgmma_fence();
                    if (comb) {
                        const bool fwd = ts16 > pb;
                        const uint64_t bd = fwd ? mkdesc(pb, ts16 - pb) : mkdesc(ts16, pb - ts16);
                        if (fwd)
                            mma_tiles<MN, TPW, NWG>(acc, nh, w, pa, xs16 - pa, bd, init ? 1 : 0, bf);
                        else
                            mma_tiles<MN, TPW, NWG>(acc, nh, w, xs16, pa - xs16, bd, init ? 1 : 0, bf);
                        init = true;
                    }
                    for (int j = 0; j < npair; ++j) {
                        const unsigned c = (unsigned)(cs + 2 * j);
                        mma_tiles<MN, TPW, NWG>(acc, nh, w, xs16 + c, 1u, bdesc(c), init ? 1 : 0, bf);
                        init = true;
                    }
                }
                wgmma_commit();
                // this group frees the pending stage (if it consumed its last chunk) and its own (unless carried)
                const int r0 = comb ? pend : -1;
                const int r1 = carry ? -1 : s;
                if (carry) {
                    pend = s;
                    pa = xs16 + (unsigned)(nc - 1);
                    pb = ts16 + (unsigned)((nc - 1) * MN);
                } else {
                    pend = -1;
                }
                if (keep) {
                    wgmma_wait<1>(); // the previous piece's group has completed
                    release();
                    rel0 = r0;
                    rel1 = r1;
                } else {
                    wgmma_wait<0>();
                    rel0 = r0;
                    rel1 = r1;
                    release();
                }
                if (++s == nst) {
                    s = 0;
                    ph ^= 1u;
                }
            }
            wgmma_wait<0>();
            release();
            // (per tile with constant indices, see the epilogue; no pieces: the outputs are zero)
#define HK_FINISH_TILE(T)                                                                                              \
    if constexpr (TPW > T) {                                                                                           \
        fence_acc(acc[T][0]);                                                                                          \
        fence_acc(acc[T][1]);                                                                                          \
        if (!init)                                                                                                     \
            for (int i = 0; i < R; ++i) acc[T][0][i] = acc[T][1][i] = 0.f;                                             \
    }
            HK_FINISH_TILE(0)
            HK_FINISH_TILE(1)
            HK_FINISH_TILE(2)
            HK_FINISH_TILE(3)
#undef HK_FINISH_TILE

            // ---------------- epilogue: registers -> fp16 / bf16 -> global ----------------
            // half-block shift: GEMM row b holds outputs DW b + DW / 2 + dwo (mod Wo)
            const int sh = (DW >= 2 && p.rowsig[ho] != 0) ? DW / 2 : 0;
            // tile by tile and half by half with constant indices into acc, spelled out: a loop the unroller leaves
            // rolled indexes acc at run time and sends it to local memory
            auto rows = [&](int t, int h, int sub, int &cl, int &b) {
                const int g = (w + NWG * t) * 128 + h * 64 + wq * 16 + (lane >> 2) + 8 * sub; // GEMM row
                cl = p.G == 2 ? (int)(g >= p.RPC) : g / p.RPC;
                b = g - cl * p.RPC;
                return cl < nch && b < p.nb;
            };
#define HK_STORE_ROW(T, H, SUB)                                                                                        \
    {                                                                                                                  \
        int cl, b;                                                                                                     \
        if (rows(T, H, SUB, cl, b))                                                                                    \
            store_rows<DW, RG, MN, R, SUB>(acc[T][H], lane, p.out + (size_t)(c0 + cl) * ((size_t)p.K * kstride),       \
                                           kstride, p, ho, b, sh);                                                     \
    }
#define HK_STORE_HALF(T, H)                                                                                            \
    if (nh[T] > H) {                                                                                                   \
        HK_STORE_ROW(T, H, 0)                                                                                          \
        HK_STORE_ROW(T, H, 1)                                                                                          \
    }
#define HK_STORE_TILE(T)                                                                                               \
    if constexpr (TPW > T) {                                                                                           \
        HK_STORE_HALF(T, 0)                                                                                            \
        HK_STORE_HALF(T, 1)                                                                                            \
    }
            HK_STORE_TILE(0)
            HK_STORE_TILE(1)
            HK_STORE_TILE(2)
            HK_STORE_TILE(3)
#undef HK_STORE_TILE
#undef HK_STORE_HALF
#undef HK_STORE_ROW
        }
    }
    // warps NPW .. 3 of warpgroup 0 have no role
}

template <int DW, bool TX, int NWG, int CPS, int RG, int NV, bool Z>
__global__ void __launch_bounds__(HkCfg<DW, NWG, CPS, RG>::THREADS, CPS)
    disco_hk_wg(const __grid_constant__ TmapX<NV> tmap_x, const HkParams p)
{
    hk_body<DW, TX, NWG, CPS, RG, NV, Z>(tmap_x, p);
}
// the main-kernel entry of a configuration
template <int DW, bool TX, int NWG, int CPS, int RG, int NV = NVMAX, bool Z = false>
static constexpr auto disco_hk_kernel()
{
    return &disco_hk_wg<DW, TX, NWG, CPS, RG, NV, Z>;
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

// the family fields of a configuration: NWG consumer warpgroups with MT M tiles per item
struct HkFamily {
    int NWG = 0, MT = 0;
    // no lane-parallel producer: the sm_100a kernel's variants measured slower on H100
    static constexpr bool LANE_PARALLEL = false;
    template <class Cfg, auto F> void set_family()
    {
        NWG = F;
        MT = Cfg::MT;
    }
};
#define HK_NAME "forward_hankel_sm90"

#include "disco_cuda_fwd_hankel_host.cuh"

namespace
{

    // The main kernel of a configuration (as the delivered launch_hk2 chose it).
    template <int DW, auto NWG, int CPS, int RG> const void *main_kernel(const HkConfig &c)
    {
        constexpr bool ZCFG = CPS != 3;
        switch (c.variant) {
        case MV_TX_NV: return (const void *)disco_hk_kernel<DW, true, NWG, CPS, RG, NVMAX, ZCFG>();
        case MV_TX_8: return (const void *)disco_hk_kernel<DW, true, NWG, CPS, RG, 8, ZCFG>();
        default: return (const void *)disco_hk_kernel<DW, false, NWG, CPS, RG, 8>();
        }
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
        if (!init_config(c, BC64, K64, kpad64, Hi64, Wi64, Ho64, Wo64, NNZ)) return c;
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
    template <class Fn> void with_family(const HkConfig &c, Fn &&f)
    {
#define DISCO_HK_FAMILY(DW_, NWG_, CPS_, RG_)                                                                          \
    if (c.DW == DW_ && c.NWG == NWG_ && c.CPS == CPS_ && c.RG == RG_)                                                  \
        return f.template operator()<DW_, NWG_, CPS_, RG_>();
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
        TORCH_CHECK(false, "forward_hankel_sm90: no main kernel for DW ", c.DW, ", NWG ", c.NWG, ", CPS ", c.CPS,
                    ", RG ", c.RG);
    }

    void check_device(int device)
    {
        const cudaDeviceProp *prop = at::cuda::getDeviceProperties(device);
        TORCH_CHECK(prop->major == 9 && prop->minor == 0,
                    "forward_hankel_sm90 requires a compute capability 9.0 device (Hopper); got ", prop->major, ".",
                    prop->minor);
    }

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
        return hankel_plan(pack_idx, pack_val, pack_offset, BC, K, Hi, Wi, Wo, dtype);
    }

    // The forward of the contiguous fp16 / bf16 (B, C, Hi, Wi) inp to (B, C, K, Ho, Wo) through the plan built for
    // B*C planes of its dtype.
    at::Tensor forward_hankel_sm90_run(at::Tensor inp, at::Tensor plan, at::Tensor pack_idx, at::Tensor pack_val,
                                       at::Tensor pack_offset, int64_t K, int64_t Ho, int64_t Wo)
    {
        return hankel_run(inp, plan, pack_idx, pack_val, pack_offset, K, Ho, Wo, &disco_cuda_fwd_kpacked_sm90);
    }

    // The configuration the forward takes for these sizes on the current device, for tests:
    // [ok, DW, MT, CPS, RG, variant, co, plan_bytes, scratch_bytes, nst]
    std::vector<int64_t> forward_hankel_sm90_config(int64_t BC, int64_t K, int64_t kpad, int64_t Hi, int64_t Wi,
                                                    int64_t Ho, int64_t Wo, int64_t nnz)
    {
        const cudaDeviceProp &prop = *at::cuda::getCurrentDeviceProperties();
        const HkConfig c = make_config(BC, K, kpad, Hi, Wi, Ho, Wo, nnz, prop);
        return {c.ok, c.DW, c.MT, c.CPS, c.RG, c.variant, c.co, (int64_t)c.plan_bytes, (int64_t)c.scratch_bytes, c.nst};
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
