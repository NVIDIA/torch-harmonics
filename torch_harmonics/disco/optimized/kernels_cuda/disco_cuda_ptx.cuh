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

namespace disco_kernels
{

// =====================================================================================
// cp.async — asynchronous gmem → shmem copy (SM_80+).  PTX ISA §9.7.8.20.
// =====================================================================================
#if __CUDA_ARCH__ >= 800

    // 16-byte cp.async with .cg cache hint (skip L1, hit L2).
    __device__ __forceinline__ void cp_async_16B(void *smem_dst, const void *gmem_src)
    {
        unsigned smem_addr = __cvta_generic_to_shared(smem_dst);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem_src));
    }

    // Commit all in-flight cp.async ops to a new pending group.
    __device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

    // Wait until at most N groups remain pending.
    template <int N> __device__ __forceinline__ void cp_async_wait_group()
    {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
    }

    // Wait for all pending cp.async groups to complete.
    __device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::); }

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

    __device__ __forceinline__ void ldmatrix_x4_b16(uint32_t (&d)[4], const void *src_smem)
    {
        unsigned smem_addr = __cvta_generic_to_shared(src_smem);
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                     : "r"(smem_addr));
    }

    __device__ __forceinline__ void ldmatrix_x2_b16(uint32_t (&d)[2], const void *src_smem)
    {
        unsigned smem_addr = __cvta_generic_to_shared(src_smem);
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(d[0]), "=r"(d[1])
                     : "r"(smem_addr));
    }

    __device__ __forceinline__ void ldmatrix_x2_b16_trans(uint32_t (&d)[2], const void *src_smem)
    {
        unsigned smem_addr = __cvta_generic_to_shared(src_smem);
        asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(d[0]), "=r"(d[1])
                     : "r"(smem_addr));
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
    // This builder is shared by SM_90a (WGMMA) and SM_100a (tcgen05.mma) —
    // the descriptor format is identical across both architectures.
    // IMPORTANT: this builder assumes no swizzle and base_offset=0.
    // -------------------------------------------------------------------------------------
    __device__ __forceinline__ uint64_t make_wgmma_desc(const void *smem_ptr, uint32_t leading_byte_offset,
                                                        uint32_t stride_byte_offset, uint32_t swizzle = 0)
    {
        unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
        uint64_t desc = 0;
        desc |= ((uint64_t)((smem_addr >> 4) & 0x3fffu)) << 0;            // start address
        desc |= ((uint64_t)((leading_byte_offset >> 4) & 0x3fffu)) << 16; // LBO
        desc |= ((uint64_t)((stride_byte_offset >> 4) & 0x3fffu)) << 32;  // SBO
        desc |= ((uint64_t)(swizzle & 0x3u)) << 62;                       // swizzle mode
        return desc;
    }

#if defined(__CUDA_ARCH_FEAT_SM90_ALL)

    // -------------------------------------------------------------------------------------
    // WGMMA fence / commit / wait (PTX ISA §9.7.16.4.1 — §9.7.16.4.4).
    // -------------------------------------------------------------------------------------

    // Insert before mma_async to ensure register-residing operands and accumulator
    // are visible to the WGMMA hardware.
    __device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::); }

    // Commit all preceding (uncommitted) wgmma.mma_async into a pending group.
    __device__ __forceinline__ void wgmma_commit_group() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::); }

    // Wait until at most N pending wgmma groups remain.
    template <int N> __device__ __forceinline__ void wgmma_wait_group()
    {
        asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N));
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
    // CUTLASS's MMA_64xNx16_F32{BF16BF16,F16F16}_SS PTX patterns from
    //   NVIDIA/cutlass v4.4.0 (cb37157d)
    //   include/cute/arch/mma_sm90_gmma.hpp
    //   https://github.com/NVIDIA/cutlass/blob/v4.4.0/include/cute/arch/mma_sm90_gmma.hpp
    // instantiated for our K-packed dense staging:
    //
    //   - A is M=64,K=16 with M fast in the 8x8 core matrix → tnspA = Major::MN = 1
    //   - B is K=16,N   with N fast in the 8x8 core matrix → tnspB = Major::MN = 1
    //   - scale_A = scale_B = ScaleIn::One   = 1
    //   - scale_D = ScaleOut::One            = 1   (accumulating)
    //
    // Earlier hand-rolled versions of these wrappers passed scale_D as a literal
    // immediate `1` (instead of a predicate) and used `tnspB = 0`. ptxas accepted
    // the malformed input but silently produced wrong results.
    //
    // _bf16 variants use .f32.bf16.bf16; _fp16 variants use .f32.f16.f16. The PTX
    // is otherwise identical.

    // ---- bf16 wrappers (.f32.bf16.bf16) ----

    __device__ __forceinline__ void wgmma_m64n8k16_acc_bf16(float (&d)[4], uint64_t desc_a, uint64_t desc_b)
    {
        int32_t scale_D = 1;
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %6, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
                     "{%0, %1, %2, %3},"
                     " %4,"
                     " %5,"
                     " p,   %7,  %8,  %9,  %10;\n"
                     "}\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "l"(desc_a), "l"(desc_b), "r"(scale_D), "n"(1), "n"(1), "n"(1), "n"(1));
    }

    __device__ __forceinline__ void wgmma_m64n16k16_acc_bf16(float (&d)[8], uint64_t desc_a, uint64_t desc_b)
    {
        int32_t scale_D = 1;
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %10, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7},"
                     " %8,"
                     " %9,"
                     " p,   %11, %12, %13, %14;\n"
                     "}\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                     : "l"(desc_a), "l"(desc_b), "r"(scale_D), "n"(1), "n"(1), "n"(1), "n"(1));
    }

    __device__ __forceinline__ void wgmma_m64n24k16_acc_bf16(float (&d)[12], uint64_t desc_a, uint64_t desc_b)
    {
        int32_t scale_D = 1;
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %14, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},"
                     " %12,"
                     " %13,"
                     " p,   %15, %16, %17, %18;\n"
                     "}\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                       "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
                     : "l"(desc_a), "l"(desc_b), "r"(scale_D), "n"(1), "n"(1), "n"(1), "n"(1));
    }

    __device__ __forceinline__ void wgmma_m64n32k16_acc_bf16(float (&d)[16], uint64_t desc_a, uint64_t desc_b)
    {
        int32_t scale_D = 1;
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %18, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},"
                     " %16,"
                     " %17,"
                     " p,   %19, %20, %21, %22;\n"
                     "}\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                       "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]),
                       "+f"(d[15])
                     : "l"(desc_a), "l"(desc_b), "r"(scale_D), "n"(1), "n"(1), "n"(1), "n"(1));
    }

    // ---- fp16 wrappers (.f32.f16.f16) ----

    __device__ __forceinline__ void wgmma_m64n8k16_acc_fp16(float (&d)[4], uint64_t desc_a, uint64_t desc_b)
    {
        int32_t scale_D = 1;
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %6, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
                     "{%0, %1, %2, %3},"
                     " %4,"
                     " %5,"
                     " p,   %7,  %8,  %9,  %10;\n"
                     "}\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "l"(desc_a), "l"(desc_b), "r"(scale_D), "n"(1), "n"(1), "n"(1), "n"(1));
    }

    __device__ __forceinline__ void wgmma_m64n16k16_acc_fp16(float (&d)[8], uint64_t desc_a, uint64_t desc_b)
    {
        int32_t scale_D = 1;
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %10, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7},"
                     " %8,"
                     " %9,"
                     " p,   %11, %12, %13, %14;\n"
                     "}\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                     : "l"(desc_a), "l"(desc_b), "r"(scale_D), "n"(1), "n"(1), "n"(1), "n"(1));
    }

    __device__ __forceinline__ void wgmma_m64n24k16_acc_fp16(float (&d)[12], uint64_t desc_a, uint64_t desc_b)
    {
        int32_t scale_D = 1;
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %14, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},"
                     " %12,"
                     " %13,"
                     " p,   %15, %16, %17, %18;\n"
                     "}\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                       "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
                     : "l"(desc_a), "l"(desc_b), "r"(scale_D), "n"(1), "n"(1), "n"(1), "n"(1));
    }

    __device__ __forceinline__ void wgmma_m64n32k16_acc_fp16(float (&d)[16], uint64_t desc_a, uint64_t desc_b)
    {
        int32_t scale_D = 1;
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %18, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},"
                     " %16,"
                     " %17,"
                     " p,   %19, %20, %21, %22;\n"
                     "}\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                       "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]),
                       "+f"(d[15])
                     : "l"(desc_a), "l"(desc_b), "r"(scale_D), "n"(1), "n"(1), "n"(1), "n"(1));
    }

#endif // __CUDA_ARCH_FEAT_SM90_ALL

// =====================================================================================
// tcgen05 — Blackwell (SM_100a) tensor-core instructions.
//
// SM_100a replaces wgmma.mma_async with tcgen05.mma.  The accumulator lives in
// TMEM (Tensor Memory) rather than the thread register file.  SMEM operand
// staging and the descriptor format (make_wgmma_desc) are shared with SM_90a.
//
// Key differences from the broken stubs that preceded these wrappers:
//
//  tcgen05.alloc  writes the TMEM base address to a SHARED MEMORY location
//                 (not a register).  The first warp (threadIdx.x < 32) must
//                 issue it; callers __syncthreads() afterward to broadcast.
//                 Minimum allocation: 32 columns (hardware constraint).
//
//  tcgen05.st/ld  use the 16x256b shape (not 32x32b).  One warpgroup-collective
//                 call covers the full M=64, N=8 (x1) or N=16 (x2) tile.
//
//  tcgen05.mma    is a CTA-collective — ALL threads must execute it.  It uses
//                 kind::f16 for BOTH
//                 fp16 and bf16 inputs (no kind::bf16 exists).  The fourth
//                 operand is the E-descriptor upper 32 bits (0 for dense A),
//                 followed by four mask registers (all 0 for full M), and a
//                 predicate that controls scale_C (0 = overwrite, 1 = add).
//                 No .sync / shape suffix in the PTX; shape is inferred from
//                 the SMEM descriptors.
//
//  Synchronisation: tcgen05.mma is asynchronous; results appear in TMEM only
//                 after tcgen05.commit signals an mbarrier.  Use
//                 tcgen05_mma_issue() (which also commits) then
//                 tcgen05_mma_wait() (which polls the mbarrier).
//
// Gate: __CUDA_ARCH_FEAT_SM100_ALL.  Build with TORCH_CUDA_ARCH_LIST="10.0a+PTX".
// =====================================================================================
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)

    // -------------------------------------------------------------------------------------
    // tcgen05.alloc — first warp writes TMEM base address to a shmem slot.
    //
    // Allocates ≥ 32 TMEM columns (hardware minimum; power-of-2).  The result is
    // broadcast via shared memory.  Callers must __syncthreads() before using the
    // returned address.
    // -------------------------------------------------------------------------------------
    __device__ __forceinline__ uint32_t tcgen05_alloc(uint32_t *smem_tmem_base)
    {
        if (threadIdx.x < 32) {
            uint32_t smem_ptr = __cvta_generic_to_shared(smem_tmem_base);
            uint32_t n_cols = 32u; // minimum; power-of-2
            asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(smem_ptr),
                         "r"(n_cols));
        }
        __syncthreads();
        return *smem_tmem_base;
    }

    // -------------------------------------------------------------------------------------
    // tcgen05.dealloc — first warp releases the TMEM allocation.
    // -------------------------------------------------------------------------------------
    __device__ __forceinline__ void tcgen05_dealloc(uint32_t tmem_addr)
    {
        if (threadIdx.x < 32) {
            uint32_t n_cols = 32u;
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(tmem_addr), "r"(n_cols));
        }
    }

    // -------------------------------------------------------------------------------------
    // tcgen05_zero — warpgroup-collective zero-init of the TMEM accumulator tile.
    //
    // Shape 16x256b.x1 covers M=64, N=8  (512 fp32 total, 4 per thread).
    // Shape 16x256b.x2 covers M=64, N=16 (1024 fp32 total, 8 per thread).
    // All 128 threads call this with the same base address; the hardware
    // routes zeros to the correct TMEM cells.
    // -------------------------------------------------------------------------------------
    template <int N_ACC> __device__ __forceinline__ void tcgen05_zero(uint32_t tmem_addr)
    {
        static_assert(N_ACC == 4 || N_ACC == 8, "tcgen05_zero: N_ACC must be 4 or 8");
        if constexpr (N_ACC == 4) {
            asm volatile("tcgen05.st.cta_group::1.sync.aligned.16x256b.x1.b32 [%0], {%1,%2,%3,%4};\n" ::"r"(tmem_addr),
                         "r"(0u), "r"(0u), "r"(0u), "r"(0u));
        } else {
            asm volatile("tcgen05.st.cta_group::1.sync.aligned.16x256b.x2.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};\n" ::"r"(
                             tmem_addr),
                         "r"(0u), "r"(0u), "r"(0u), "r"(0u), "r"(0u), "r"(0u), "r"(0u), "r"(0u));
        }
        asm volatile("tcgen05.wait::st.cta_group::1.sync.aligned;\n" ::: "memory");
    }

    // -------------------------------------------------------------------------------------
    // tcgen05_ld — warpgroup-collective read of the TMEM accumulator tile.
    //
    // Shape 16x256b.x1 → 4 fp32 per thread (N_ACC=4, N_PAD=8).
    // Shape 16x256b.x2 → 8 fp32 per thread (N_ACC=8, N_PAD=16).
    // All 128 threads call this; each receives its portion of the M×N tile.
    // Followed by tcgen05.wait::ld to ensure completion before register use.
    // -------------------------------------------------------------------------------------
    template <int N_ACC> __device__ __forceinline__ void tcgen05_ld(float (&regs)[N_ACC], uint32_t tmem_addr)
    {
        static_assert(N_ACC == 4 || N_ACC == 8, "tcgen05_ld: N_ACC must be 4 or 8");
        uint32_t(&r)[N_ACC] = reinterpret_cast<uint32_t(&)[N_ACC]>(regs);
        if constexpr (N_ACC == 4) {
            asm volatile("tcgen05.ld.cta_group::1.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                         : "r"(tmem_addr));
        } else {
            asm volatile("tcgen05.ld.cta_group::1.sync.aligned.16x256b.x2.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
                         : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                         : "r"(tmem_addr));
        }
        asm volatile("tcgen05.wait::ld.cta_group::1.sync.aligned;\n" ::: "memory");
    }

    // -------------------------------------------------------------------------------------
    // tcgen05_mma_issue — CTA-collective MMA + single-thread mbarrier commit.
    //
    // tcgen05.mma is a CTA-collective instruction: ALL 128 threads must execute it.
    // tcgen05.commit must be issued by exactly ONE thread (threadIdx.x == 0 here).
    //
    // kind::f16 is correct for BOTH fp16 and bf16 inputs (no kind::bf16 in PTX).
    // Operand 3 is idescE upper-32 (0 for dense A).  Masks are {0,0,0,0} for
    // full M=64 participation.  The predicate p controls scaleC:
    //   p = false (scale_c == 0) → D  = A*B      (overwrite)
    //   p = true  (scale_c != 0) → D += A*B      (accumulate)
    //
    // After the MMA, tcgen05.commit signals the mbarrier so tcgen05_mma_wait()
    // can unblock all threads.
    // -------------------------------------------------------------------------------------
    __device__ __forceinline__ void tcgen05_mma_issue(uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b,
                                                      uint32_t scale_c, uint32_t mbar_ptr)
    {
        // tcgen05.mma is CTA-collective: ALL threads must execute it.
        // tcgen05.commit is single-thread: only one thread signals the mbarrier.
        uint32_t mask0 = 0, mask1 = 0, mask2 = 0, mask3 = 0;
        asm volatile("{\n\t"
                     ".reg .pred p;\n\t"
                     "setp.ne.b32 p, %4, 0;\n\t"
                     "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5,%6,%7,%8}, p;\n\t"
                     "}\n" ::"r"(tmem_d),
                     "l"(desc_a), "l"(desc_b),
                     "r"(0u),      // idescE upper-32 (dense A → 0)
                     "r"(scale_c), // controls predicate p
                     "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3));
        if (threadIdx.x == 0) {
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(mbar_ptr)
                         : "memory");
        }
    }

    // -------------------------------------------------------------------------------------
    // tcgen05_mma_wait — all threads spin-wait for the MMA mbarrier to complete.
    //
    // Waits for the mbarrier at mbar_ptr to complete phase parity=0 (i.e., polls
    // until the mbarrier's parity transitions from 0 to 1 after the commit).
    // After returning, the TMEM contents are guaranteed to be the MMA result.
    // The caller must __syncthreads() before re-initialising the mbarrier for the
    // next chunk.
    // -------------------------------------------------------------------------------------
    __device__ __forceinline__ void tcgen05_mma_wait(uint32_t mbar_ptr)
    {
        uint32_t done = 0;
        do {
            asm volatile("{\n\t"
                         ".reg .pred p;\n\t"
                         "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
                         "selp.u32 %0, 1, 0, p;\n\t"
                         "}\n"
                         : "=r"(done)
                         : "r"(mbar_ptr), "r"(0u)
                         : "memory");
        } while (!done);
    }

#endif // __CUDA_ARCH_FEAT_SM100_ALL

} // namespace disco_kernels
