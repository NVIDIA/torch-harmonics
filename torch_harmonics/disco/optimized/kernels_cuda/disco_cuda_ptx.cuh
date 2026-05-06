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

// =====================================================================================
// PTX inline-asm wrappers for the disco WGMMA path.
//
// All wrappers are __device__ __forceinline__. Each has a comment pointing to
// the relevant section of the PTX ISA reference; the PTX docs are the source
// of truth, this file is just a thin C++ veneer over them.
//
// Hardware gating:
//   cp.async / ldmatrix : SM_80+ (Ampere), but only used here from the WGMMA
//                         kernel which is itself Hopper-gated. Wrappers are
//                         compiled when __CUDA_ARCH__ >= 800.
//   wgmma.*             : SM_90+ (Hopper). Wrappers gated __CUDA_ARCH__ >= 900.
//
// Layout conventions for WGMMA:
//   wgmma.mma_async expects A and B in shared memory described by 64-bit
//   matrix descriptors (PTX ISA §9.7.16.5.1). For our K-packed dense kernel
//   we use M-major A (rows of M=BC*WO_TILE adjacent) and N-major B (cols of
//   N=K_pad adjacent), with no swizzle. trans-a / trans-b in the mma_async
//   calls below match those layouts.
// =====================================================================================

#pragma once

#include <cstdint>

namespace disco_kernels {

// =====================================================================================
// cp.async — asynchronous gmem → shmem copy (SM_80+).  PTX ISA §9.7.8.20.
// =====================================================================================
#if __CUDA_ARCH__ >= 800

// 16-byte cp.async with .cg cache hint (skip L1, hit L2).
__device__ __forceinline__ void cp_async_16B(void *smem_dst, const void *gmem_src) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_dst);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_src)
    );
}

// Commit all in-flight cp.async ops to a new pending group.
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most N groups remain pending.
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// Wait for all pending cp.async groups to complete.
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

// =====================================================================================
// ldmatrix — distribute 8×8 b16 matrices from shared memory across a warp.
// PTX ISA §9.7.13.4.15.
//
// .x4 (4 matrices) is used to load A's 16×16 fragment for HMMA / build A's
// register fragments for WGMMA when A lives in registers. For our WGMMA path
// A and B both live in shared memory (described by descriptors), so ldmatrix
// is used only when we need to move data between shmem layouts. Kept here for
// completeness and possible later use.
// =====================================================================================

__device__ __forceinline__ void ldmatrix_x4_b16(uint32_t (&d)[4], const void *src_smem) {
    unsigned smem_addr = __cvta_generic_to_shared(src_smem);
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(smem_addr)
    );
}

__device__ __forceinline__ void ldmatrix_x2_b16(uint32_t (&d)[2], const void *src_smem) {
    unsigned smem_addr = __cvta_generic_to_shared(src_smem);
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(d[0]), "=r"(d[1])
        : "r"(smem_addr)
    );
}

__device__ __forceinline__ void ldmatrix_x2_b16_trans(uint32_t (&d)[2], const void *src_smem) {
    unsigned smem_addr = __cvta_generic_to_shared(src_smem);
    asm volatile(
        "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(d[0]), "=r"(d[1])
        : "r"(smem_addr)
    );
}

#endif // __CUDA_ARCH__ >= 800

// =====================================================================================
// WGMMA — warp-group matrix-multiply-accumulate.
//
// WGMMA is gated to the *architecture-specific* Hopper target (sm_90a), not
// the forward-compatible sm_90. ptxas rejects WGMMA opcodes against plain
// .target sm_90 — they require .target sm_90a. NVCC defines
// `__CUDA_ARCH_FEAT_SM90_ALL` only when compiling for sm_90a, so we use that
// macro to gate the inline asm. Build with TORCH_CUDA_ARCH_LIST="9.0a+PTX"
// (lowercase `a`) to enable this path. Blackwell SM_100+ deprecated WGMMA
// entirely in favour of the tcgen05.mma family — handled by separate guards.
//
// PTX ISA §9.7.16.
// =====================================================================================
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)

// -------------------------------------------------------------------------------------
// Matrix descriptor (PTX ISA §9.7.16.5.1).
//
// The descriptor is a 64-bit value laid out as:
//
//   bit  0:13  start_address       (bits 4..17 of the shmem byte offset)
//   bit 14:15  unused (must be 0)
//   bit 16:29  leading_byte_offset (LBO, in 16-byte units)
//   bit 30:31  unused (must be 0)
//   bit 32:45  stride_byte_offset  (SBO, in 16-byte units)
//   bit 46:48  unused (must be 0)
//   bit 49:51  matrix_base_offset  (3-bit, only used with swizzle)
//   bit 52:61  unused (must be 0)
//   bit 62:63  swizzle             (0 = none, 1 = 128B, 2 = 64B, 3 = 32B)
//
// LBO: byte offset between adjacent "core matrices" along the leading dim.
// SBO: byte offset between adjacent core matrices along the stride dim.
// "Core matrix" for bf16/fp16 with K=16 is 8×16 elements (256 bytes); for our
// purposes both LBO and SBO are simply the byte stride between consecutive
// 8×16 core-matrix tiles in shmem along the M (or N) and K axes respectively.
//
// For the simple case (no swizzle, contiguous shmem region):
//   A 64×16 bf16 row-major (M outer, K inner) as 8 stacked 8×16 core matrices:
//     LBO = 256 bytes  (next 8-row block along M is 256 bytes after the prior)
//     SBO = 16  bytes  (within a row, K direction is 16 elements = 32 bytes; here
//                       there's only one 8×16 core in K so SBO is unused —
//                       but the descriptor still requires a value; use the
//                       byte-stride between consecutive K-segments if any.)
//
// IMPORTANT: this builder assumes no swizzle and base_offset=0. If we add
// swizzling later we'll need to thread those through.
// -------------------------------------------------------------------------------------
__device__ __forceinline__ uint64_t make_wgmma_desc(const void *smem_ptr,
                                                    uint32_t leading_byte_offset,
                                                    uint32_t stride_byte_offset,
                                                    uint32_t swizzle = 0) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
    uint64_t desc = 0;
    desc |= ((uint64_t)((smem_addr           >> 4) & 0x3fffu)) <<  0;  // start address
    desc |= ((uint64_t)((leading_byte_offset >> 4) & 0x3fffu)) << 16;  // LBO
    desc |= ((uint64_t)((stride_byte_offset  >> 4) & 0x3fffu)) << 32;  // SBO
    desc |= ((uint64_t)(swizzle & 0x3u))                       << 62;  // swizzle mode
    return desc;
}

// -------------------------------------------------------------------------------------
// WGMMA fence / commit / wait (PTX ISA §9.7.16.4.1 — §9.7.16.4.4).
// -------------------------------------------------------------------------------------

// Insert before mma_async to ensure register-residing operands and accumulator
// are visible to the WGMMA hardware.
__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::);
}

// Commit all preceding (uncommitted) wgmma.mma_async into a pending group.
__device__ __forceinline__ void wgmma_commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::);
}

// Wait until at most N pending wgmma groups remain.
template <int N>
__device__ __forceinline__ void wgmma_wait_group() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N));
}

// -------------------------------------------------------------------------------------
// wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16 (PTX ISA §9.7.16.5).
//
// Computes D = A * B + (scale_d ? D : 0) for a single 64×N×16 tile.
//
// Operands:
//   d_frag   : per-thread accumulator fragments. Each WGMMA m64nNk16 produces
//              N/2 fp32 cells per thread; we pass d_frag as a uint32_t* (128
//              threads × N/2 = 64*N total fp32 cells in the warpgroup, matching
//              the M=64, N=N tile).
//   desc_a   : 64-bit shmem descriptor for matrix A (M=64, K=16).
//   desc_b   : 64-bit shmem descriptor for matrix B (K=16, N=N).
//   scale_d  : 1 to accumulate (D += A*B), 0 to overwrite (D = A*B).
//   trans_a  : transpose flag for A's shmem layout (1 = transpose).
//   trans_b  : transpose flag for B's shmem layout (1 = transpose).
//   imm_scale_a / imm_scale_b are fixed at +1 here.
//
// One wrapper per N value (N ∈ {8, 16, 24, 32}) — the count of d_frag operands
// is N/2 fp32 = N/2 32-bit registers per thread, so the asm operand list size
// changes per N.
// -------------------------------------------------------------------------------------

// scale_a / scale_b / trans_a / trans_b must be PTX immediates; scale_d is a
// PTX predicate operand (not a literal immediate). The wrappers below mirror
// CUTLASS's MMA_64xNx16_F32BF16BF16_SS PTX patterns (see
// external/cutlass/include/cute/arch/mma_sm90_gmma.hpp), instantiated for our
// K-packed dense staging:
//
//   - A is M=64,K=16 with M fast in the 8x8 core matrix → tnspA = Major::MN = 1
//   - B is K=16,N   with N fast in the 8x8 core matrix → tnspB = Major::MN = 1
//   - scale_A = scale_B = ScaleIn::One   = 1
//   - scale_D = ScaleOut::One            = 1   (accumulating)
//
// Earlier hand-rolled versions of these wrappers passed scale_D as a literal
// immediate `1` (instead of a predicate) and used `tnspB = 0`. ptxas accepted
// the malformed input but silently produced wrong results.

// N = 8 → 4 fp32 accumulator regs per thread.
__device__ __forceinline__ void wgmma_m64n8k16_acc(
    float (&d)[4], uint64_t desc_a, uint64_t desc_b)
{
    int32_t scale_D = 1;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3},"
        " %4,"
        " %5,"
        " p,   %7,  %8,  %9,  %10;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(desc_a), "l"(desc_b),
          "r"(scale_D), "n"(1), "n"(1), "n"(1), "n"(1)
    );
}

// N = 16 → 8 fp32 accumulator regs per thread.
__device__ __forceinline__ void wgmma_m64n16k16_acc(
    float (&d)[8], uint64_t desc_a, uint64_t desc_b)
{
    int32_t scale_D = 1;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7},"
        " %8,"
        " %9,"
        " p,   %11, %12, %13, %14;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "l"(desc_a), "l"(desc_b),
          "r"(scale_D), "n"(1), "n"(1), "n"(1), "n"(1)
    );
}

// N = 24 → 12 fp32 accumulator regs per thread.
__device__ __forceinline__ void wgmma_m64n24k16_acc(
    float (&d)[12], uint64_t desc_a, uint64_t desc_b)
{
    int32_t scale_D = 1;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %14, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},"
        " %12,"
        " %13,"
        " p,   %15, %16, %17, %18;\n"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11])
        : "l"(desc_a), "l"(desc_b),
          "r"(scale_D), "n"(1), "n"(1), "n"(1), "n"(1)
    );
}

// N = 32 → 16 fp32 accumulator regs per thread.
__device__ __forceinline__ void wgmma_m64n32k16_acc(
    float (&d)[16], uint64_t desc_a, uint64_t desc_b)
{
    int32_t scale_D = 1;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %18, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},"
        " %16,"
        " %17,"
        " p,   %19, %20, %21, %22;\n"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
        : "l"(desc_a), "l"(desc_b),
          "r"(scale_D), "n"(1), "n"(1), "n"(1), "n"(1)
    );
}

#endif // __CUDA_ARCH_FEAT_SM90_ALL

}  // namespace disco_kernels
