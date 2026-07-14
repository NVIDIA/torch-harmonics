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
// PTX inline-asm wrappers for the attention WGMMA path (Hopper sm_90a).
//
// These are extracted VERBATIM (semantics-preserving) from the validated DISCO
// tensor-core implementation (branch tkurth/disco-tc, disco_cuda_ptx.cuh), which
// was debugged on Hopper. Re-derive nothing here — the layout/encoding decisions
// below are the ones that produced correct DISCO results.
//
// Hardware gating:
//   cp.async   : SM_80+, used only from the Hopper kernel.
//   wgmma.*    : SM_90a. ptxas REJECTS WGMMA against plain .target sm_90 — it needs
//                .target sm_90a. NVCC defines __CUDA_ARCH_FEAT_SM90_ALL only for
//                sm_90a, so we gate on that. Build TORCH_CUDA_ARCH_LIST="9.0a+PTX".
//
// Layout conventions (Major::MN, no swizzle):
//   A [M=64, K=16]    M-fast in the 8x8 core:  byte(m,k) = (m/8)*256 + (m%8 + 8*k)*2
//   B [K=16, N]       N-fast in the 8x8 core:  byte(k,n) = (n/8)*256 + k*16 + (n%8)*2
//   descriptor LBO/SBO are byte strides between 8x8 core matrices in 16-byte units;
//   make_wgmma_desc lands `leading` in bits 16-29 and `stride` in bits 32-45.
//     A: leading(K-outer)=8 (128B), stride(M-outer)=16 (256B)
//     B: leading(K-outer)=8 (128B), stride(N-outer)= 0 (N=8) or 16 (256B, N>=16)
//
// Accumulator fragment (m64nNk16.f32, PTX ISA 9.7.16.5.4): N/2 fp32 cells/thread,
// in n-groups of 4 cells per 8 N-cols. warp w in [0,4), lane l in [0,32):
//   cell 4*ng+0: m = w*16 + l/4,     n = (l%4)*2     + 8*ng
//   cell 4*ng+1: m = w*16 + l/4,     n = (l%4)*2 + 1 + 8*ng
//   cell 4*ng+2: m = w*16 + l/4 + 8, n = (l%4)*2     + 8*ng
//   cell 4*ng+3: m = w*16 + l/4 + 8, n = (l%4)*2 + 1 + 8*ng
// =====================================================================================

#pragma once

#include <cstdint>

namespace attention_kernels
{

#if __CUDA_ARCH__ >= 800
    __device__ __forceinline__ void cp_async_16B(void *smem_dst, const void *gmem_src)
    {
        unsigned smem_addr = __cvta_generic_to_shared(smem_dst);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem_src));
    }
    __device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
    template <int N> __device__ __forceinline__ void cp_async_wait_group()
    {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
    }
    __device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::); }
#endif // __CUDA_ARCH__ >= 800

#if defined(__CUDA_ARCH_FEAT_SM90_ALL)

    // 64-bit shared-memory matrix descriptor (PTX ISA 9.7.16.5.1). No swizzle.
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

    __device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::); }

    // Make shared-memory writes issued through the GENERIC proxy (normal smem stores when
    // staging shQ/shK/shP/shV) visible to WGMMA's ASYNC proxy. Producer threads MUST
    // execute this after staging and before the __syncthreads that releases the
    // WGMMA-consuming warpgroup. wgmma.fence only orders the accumulator registers, NOT
    // the shared operands — omitting this is a nondeterministic race (garbage results).
    // Verbatim from the debugged DISCO path (disco_cuda_ptx.cuh).
    __device__ __forceinline__ void fence_proxy_async_shared_cta()
    {
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    }

    __device__ __forceinline__ void wgmma_commit_group() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::); }
    template <int N> __device__ __forceinline__ void wgmma_wait_group()
    {
        asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N));
    }

    // fp16 wrappers (.f32.f16.f16). scale_D is a PREDICATE (not a literal immediate);
    // scale_A=scale_B=1, tnspA=tnspB=1 (Major::MN). Accumulating (D += A*B).
    __device__ __forceinline__ void wgmma_m64n8k16_acc_fp16(float (&d)[4], uint64_t desc_a, uint64_t desc_b)
    {
        int32_t scale_D = 1;
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %6, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
                     "{%0, %1, %2, %3}, %4, %5, p, %7, %8, %9, %10;\n"
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
                     "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, p, %11, %12, %13, %14;\n"
                     "}\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                     : "l"(desc_a), "l"(desc_b), "r"(scale_D), "n"(1), "n"(1), "n"(1), "n"(1));
    }

    // bf16 wrapper (.f32.bf16.bf16) — identical encoding to fp16 except the input dtype.
    // Native on sm_90a (Hopper bf16 tensor cores). scale_D predicate, tnspA=tnspB=1.
    __device__ __forceinline__ void wgmma_m64n16k16_acc_bf16(float (&d)[8], uint64_t desc_a, uint64_t desc_b)
    {
        int32_t scale_D = 1;
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %10, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, p, %11, %12, %13, %14;\n"
                     "}\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                     : "l"(desc_a), "l"(desc_b), "r"(scale_D), "n"(1), "n"(1), "n"(1), "n"(1));
    }

    __device__ __forceinline__ void wgmma_m64n32k16_acc_fp16(float (&d)[16], uint64_t desc_a, uint64_t desc_b)
    {
        int32_t scale_D = 1;
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %18, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
                     "%16, %17, p, %19, %20, %21, %22;\n"
                     "}\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                       "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]),
                       "+f"(d[15])
                     : "l"(desc_a), "l"(desc_b), "r"(scale_D), "n"(1), "n"(1), "n"(1), "n"(1));
    }

#endif // __CUDA_ARCH_FEAT_SM90_ALL

} // namespace attention_kernels
