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
// Disco forward — K-packed dense psi (CUDA, WGMMA path, Hopper SM_90+)
// =====================================================================================
//
// First-pass WGMMA implementation. Restricted to:
//   - K_PAD = 8                  (one-core-matrix-along-N case; other K_PADs need
//                                 a multi-core staging path)
//   - pscale ≥ 1                  (any integer; gmem reads become strided when
//                                 pscale > 1, but correctness is preserved)
//   - bf16 inputs                 (matches our wgmma_m64n8k16_acc wrapper)
//
// The host wrapper checks these and falls back to the scalar K-packed kernel
// otherwise.
//
// CTA layout
// ----------
// Threads: 128 (1 warp-group, mandatory for WGMMA)
// Output tile: BC_TILE=8 channels × WO_TILE=8 wo positions × N=8 k_kerns = 512 cells
// Per-thread acc: 4 fp32 (m64n8k16 distributes 64×8 cells across 128 threads)
// Grid: (Ho × (Wo / WO_TILE), BC / BC_TILE)
//
// Shared memory layout (bf16, 16-byte aligned)
// --------------------------------------------
// A_tile [M=64, K=16] — M-major, 8 core matrices of 8×16 stacked along M:
//    byte(m, k) = (m / 8) * 256 + (m % 8 + 8 * k) * 2
//    total 8 × 256 = 2048 bytes
//    LBO = 256 bytes  (between core matrices along M)
//    SBO = 0          (only one core along K=16)
//
// B_tile [K=16, N=8] — N-major, 1 core matrix of 16×8:
//    byte(k, n) = (k * 8 + n) * 2
//    total 16 × 16 = 256 bytes
//    LBO = 0          (only one core along N=8)
//    SBO = 0          (only one core along K=16)
//
// Per nz_chunk inner loop
// -----------------------
// 1. Stage A_tile: for each (bc_local, nz_local) ∈ [0,8)×[0,16) — 128 cells —
//    copy 16 bytes (8 bf16 wo_local positions) from gmem inp into shmem A_tile.
//    Distributed 1:1 across the 128 threads.
// 2. Stage B_tile: copy 256 bytes contiguously from pack_val[ho, nz_chunk_offset..+15, 0..7].
//    Distributed across the first 16 threads (1 cp.async-16B each); rest idle.
// 3. __syncthreads.
// 4. wgmma.fence; wgmma.mma_async m64n8k16; wgmma.commit_group; wgmma.wait_group<0>.
//
// Output write
// ------------
// After all nz_chunks, each thread writes its 4 fp32 accumulator cells to gmem.
// WGMMA m64n8k16.f32 fragment-to-(m, n) mapping (PTX ISA §9.7.16.5.4.4):
//    warp w in [0,4), lane l in [0,32):
//      cell 0: m = w*16 + l/4,     n = (l%4)*2
//      cell 1: m = w*16 + l/4,     n = (l%4)*2 + 1
//      cell 2: m = w*16 + l/4 + 8, n = (l%4)*2
//      cell 3: m = w*16 + l/4 + 8, n = (l%4)*2 + 1
//    Our M dim is m = bc * WO_TILE + wo_local = bc * 8 + wo_local, so:
//      bc       = m / 8
//      wo_local = m % 8
//    Our N dim is n = k_kern.
// =====================================================================================

#include "../disco.h"
#include "disco_cuda.cuh"
#include "disco_cuda_ptx.cuh"

#include <ATen/Dispatch.h>
#include <cuda_bf16.h>

// =====================================================================================
// Debug toggle: replace the WGMMA call with a hand-written scalar matmul over
// the same A_tile / B_tile shmem buffers and the same per-thread output frag
// mapping. Used to bisect whether bugs live in the WGMMA descriptor / trans
// flag (manual gives correct output) or in the surrounding staging / write-
// back (manual gives the same wrong output as WGMMA).
//
// 0 = production WGMMA path
// 1 = manual scalar matmul (debug; correct but obviously slow)
// =====================================================================================
#ifndef DISCO_KPACKED_WGMMA_DEBUG_MANUAL
#define DISCO_KPACKED_WGMMA_DEBUG_MANUAL 0
#endif

namespace disco_kernels {

// Kernel symbol exists on all archs so the host launcher compiles cleanly;
// the body is empty for non-Hopper builds (and that path is never launched —
// see the runtime CC check in disco_cuda_fwd_dense_kpacked_wgmma_try).
template <int BC_TILE, int WO_TILE, int NZ_CHUNK, int N_PAD>
__global__ __launch_bounds__(128)
void disco_fwd_dense_kpacked_wgmma_blk_k(
    int Hi, int Wi, int K, int Ho, int Wo, int NBR_PAD, int pscale,
    const int64_t       *__restrict__ pack_idx,    // [Ho, NBR_PAD, 2]
    const __nv_bfloat16 *__restrict__ pack_val,    // [Ho, NBR_PAD, N_PAD]  (=K_PAD)
    const int64_t       *__restrict__ pack_count,  // [Ho]
    const __nv_bfloat16 *__restrict__ inp,         // [B, C, Hi, Wi]
    __nv_bfloat16       *__restrict__ out)         // [B, C, K, Ho, Wo]
{
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    static_assert(N_PAD == 8, "WGMMA path: only K_PAD=8 supported in this first pass");
    static_assert(BC_TILE == 8 && WO_TILE == 8, "WGMMA path: tile must be 8×8 for M=64");

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane    = tid - warp_id * 32;

    const int wo_per_ho = Wo / WO_TILE;
    const int ho        = blockIdx.x / wo_per_ho;
    const int wo_strip  = blockIdx.x - ho * wo_per_ho;
    const int wo_base   = wo_strip * WO_TILE;
    const int bc_start  = blockIdx.y * BC_TILE;

    const int64_t       *idx_ho = pack_idx + (int64_t)ho * NBR_PAD * 2;
    const __nv_bfloat16 *val_ho = pack_val + (int64_t)ho * NBR_PAD * N_PAD;
    const int            cnt    = (int)pack_count[ho];

    // Per-thread accumulator: 4 fp32 cells for m64n8k16.
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (cnt == 0) {
        // No-op fall through to writeback (which writes zeros).
    }

    // Shared memory: A_tile (2048 B) + B_tile (256 B) = 2304 B per CTA.
    extern __shared__ __align__(16) unsigned char shmem_raw[];
    __nv_bfloat16 *A_tile = reinterpret_cast<__nv_bfloat16*>(shmem_raw);
    __nv_bfloat16 *B_tile = A_tile + (BC_TILE * 8) * NZ_CHUNK;   // 64 * 16 = 1024 elements

    // ----------------------- nz_chunk loop -----------------------
    for (int nz_chunk_off = 0; nz_chunk_off < cnt; nz_chunk_off += NZ_CHUNK) {

        // -- Stage A_tile (128 cells = 1 per thread) --
        // Each thread copies 8 bf16 values (one wo_local strip at fixed bc, nz)
        // from gmem inp into A_tile shmem.
        const int bc_local = tid / NZ_CHUNK;     // [0, 8)
        const int nz_local = tid - bc_local * NZ_CHUNK;  // [0, 16)
        const int nz_global = nz_chunk_off + nz_local;

        const int bc = bc_start + bc_local;

        if (nz_global < cnt) {
            const int hi      = (int)idx_ho[nz_global * 2 + 0];
            const int wi_base = (int)idx_ho[nz_global * 2 + 1];
            const int64_t inp_row_base = (int64_t)bc * Hi * Wi + (int64_t)hi * Wi;
            __nv_bfloat16 *dst = A_tile
                + bc_local * (8 * NZ_CHUNK)        // bc_local * 128 elements
                + nz_local * 8;                    // nz_local * 8 elements (= 16 bytes)
            // wi_full wraps at most once: with wi_base ≤ Wi-1 and (wo_base+i)
            // ≤ Wo-1, wi_full ≤ (Wi-1) + (Wo-1)*pscale ≤ 2*Wi - 1 - pscale.
            // Relies on Wi == pscale*Wo (enforced by the host wrapper).
            // Per-element scalar reads here: with pscale > 1 the 8 wi
            // positions are no longer contiguous, so a single 16-byte vector
            // load isn't possible. The compiler issues 8 × 2-byte loads and
            // L1/L2 sectoring partially mitigates the strided access.
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int wi_full = wi_base + (wo_base + i) * pscale;
                if (wi_full >= Wi) wi_full -= Wi;
                dst[i] = inp[inp_row_base + wi_full];
            }
        } else {
            // Zero pad for nz_global >= cnt — shmem dst is 16-byte aligned, so
            // a vector store is safe here.
            __nv_bfloat16 *dst = A_tile
                + bc_local * (8 * NZ_CHUNK)
                + nz_local * 8;
            *reinterpret_cast<int4*>(dst) = make_int4(0, 0, 0, 0);
        }

        // -- Stage B_tile (256 bytes = 16 cells of 16 B each) --
        // First 16 threads copy 1 strip of 8 bf16 (one nz_local row of B).
        if (tid < NZ_CHUNK) {
            const int nz_local_b = tid;
            const int nz_global_b = nz_chunk_off + nz_local_b;
            const __nv_bfloat16 *src_b = (nz_global_b < cnt)
                ? val_ho + (int64_t)nz_global_b * N_PAD
                : nullptr;
            __nv_bfloat16 *dst_b = B_tile + nz_local_b * N_PAD;
            if (src_b != nullptr) {
                *reinterpret_cast<int4*>(dst_b) = *reinterpret_cast<const int4*>(src_b);
            } else {
                *reinterpret_cast<int4*>(dst_b) = make_int4(0, 0, 0, 0);
            }
        }

        __syncthreads();

#if DISCO_KPACKED_WGMMA_DEBUG_MANUAL
        // -- Manual scalar matmul (debug) --
        // Compute the 4 acc fragments using scalar FMAs over the staged
        // A_tile / B_tile. Same per-thread (m, n) layout as WGMMA m64n8k16.f32
        // (PTX ISA §9.7.16.5.4.4). Same output writeback. Lets us verify that
        // the staging + frag mapping are correct independently of the WGMMA
        // descriptor encoding.
        //
        // Shmem layout convention:
        //   A_tile[(m/8)*128 + k*8 + (m%8)]    — bc-major outer, k-major mid, M-fast inner
        //   B_tile[k*8 + n]                    — k-major outer, n-fast inner
        {
            const int m_top = warp_id * 16 + (lane >> 2);
            const int m_bot = m_top + 8;
            const int n0    = (lane & 3) * 2;
            const int n1    = n0 + 1;

            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            #pragma unroll
            for (int k = 0; k < NZ_CHUNK; k++) {
                // PyTorch passes -D__CUDA_NO_BFLOAT16_CONVERSIONS__ which
                // disables implicit bf16→float casts; use __bfloat162float
                // explicitly.
                const float a_top = __bfloat162float(A_tile[(m_top / 8) * 128 + k * 8 + (m_top % 8)]);
                const float a_bot = __bfloat162float(A_tile[(m_bot / 8) * 128 + k * 8 + (m_bot % 8)]);
                const float b_n0  = __bfloat162float(B_tile[k * N_PAD + n0]);
                const float b_n1  = __bfloat162float(B_tile[k * N_PAD + n1]);
                a0 += a_top * b_n0;
                a1 += a_top * b_n1;
                a2 += a_bot * b_n0;
                a3 += a_bot * b_n1;
            }
            acc[0] += a0;
            acc[1] += a1;
            acc[2] += a2;
            acc[3] += a3;
        }
#else
        // -- WGMMA m64n8k16, accumulating --
        // Descriptor LBO/SBO are byte strides between 8×8 core matrices in
        // shmem, stored in 16-byte (uint128_t) units. The bitfield mnemonics
        // are misleading; the source of truth is `cute::make_gmma_desc` in
        //   external/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp
        // which for Major::MN INTERLEAVE assigns:
        //   leading_byte_offset_ = K-outer stride (8-K-block to next 8-K-block)
        //   stride_byte_offset_  = M/N-outer stride (8-MN-block to next 8-MN-block)
        // `make_wgmma_desc(ptr, leading, stride)` lands `leading` in bits
        // 16-29 (the leading_byte_offset_ field) and `stride` in bits 32-45
        // (the stride_byte_offset_ field).
        //
        // Both A and B sit in our shmem as Major::MN-style tiles (M-fast in
        // core for A → trans-a=1; N-fast in core for B → trans-b=0 since
        // PTX trans-b semantics differ from trans-a). For bf16 K=16 the
        // core is 8×8 = 128 B, so:
        //
        //   A [M=64, K=16], byte(m,k)=(m/8)*256 + (m%8 + 8*k)*2:
        //     leading (K-outer block stride): 128 B → field  8
        //     stride  (M-outer block stride): 256 B → field 16
        //
        //   B [K=16, N=8],  byte(k,n)=(k*8 + n)*2:
        //     leading (K-outer block stride): 128 B → field  8
        //     stride  (N-outer block stride): unused (only one N-block)
        wgmma_fence();
        constexpr uint32_t A_LEADING_FIELD = 8;    // K-outer 8-block stride → 128 B
        constexpr uint32_t A_STRIDE_FIELD  = 16;   // M-outer 8-block stride → 256 B
        constexpr uint32_t B_LEADING_FIELD = 8;    // K-outer 8-block stride → 128 B
        constexpr uint32_t B_STRIDE_FIELD  = 0;    // N-outer unused (one N-block)
        uint64_t desc_a = make_wgmma_desc(A_tile,
                                          A_LEADING_FIELD * 16,
                                          A_STRIDE_FIELD  * 16);
        uint64_t desc_b = make_wgmma_desc(B_tile,
                                          B_LEADING_FIELD * 16,
                                          B_STRIDE_FIELD  * 16);
        wgmma_m64n8k16_acc(acc, desc_a, desc_b);
        wgmma_commit_group();
        wgmma_wait_group<0>();
#endif

        __syncthreads();
    }

    // ----------------------- writeback -----------------------
    // Map the 4 fragment cells to (bc, wo_local, k_kern):
    //   warp_id w in [0, 4), lane l in [0, 32).
    //   cell 0: m = w*16 + l/4,     n = (l%4)*2
    //   cell 1: m = w*16 + l/4,     n = (l%4)*2 + 1
    //   cell 2: m = w*16 + l/4 + 8, n = (l%4)*2
    //   cell 3: m = w*16 + l/4 + 8, n = (l%4)*2 + 1
    const int m01 = warp_id * 16 + (lane >> 2);          // m for cells 0, 1
    const int m23 = m01 + 8;                              // m for cells 2, 3
    const int n01 = (lane & 3) * 2;                       // n for cells 0, 2
    const int n13 = n01 + 1;                              // n for cells 1, 3

    const int bc01 = bc_start + (m01 >> 3);              // m / 8
    const int wo01 = wo_base  + (m01 & 7);               // m % 8
    const int bc23 = bc_start + (m23 >> 3);
    const int wo23 = wo_base  + (m23 & 7);

    auto write_cell = [&] (int bc_o, int wo_o, int k_o, float v) {
        if (k_o >= K) return;  // n in [K, K_PAD) is padding; skip.
        out[((int64_t)bc_o * K + k_o) * Ho * Wo + (int64_t)ho * Wo + wo_o] =
            __float2bfloat16(v);
    };
    write_cell(bc01, wo01, n01, acc[0]);
    write_cell(bc01, wo01, n13, acc[1]);
    write_cell(bc23, wo23, n01, acc[2]);
    write_cell(bc23, wo23, n13, acc[3]);
#else
    // Non-sm_90a target: WGMMA opcodes aren't available here. This includes
    // pre-Hopper, plain sm_90 (without the `a` suffix), and Blackwell+.
    // Empty body satisfies the linker; the host runtime check below ensures
    // the kernel is never launched on devices where its body is empty.
    (void)Hi; (void)Wi; (void)K; (void)Ho; (void)Wo; (void)NBR_PAD; (void)pscale;
    (void)pack_idx; (void)pack_val; (void)pack_count; (void)inp; (void)out;
#endif
}

// Host wrapper. Returns true iff the WGMMA path was actually invoked. The host
// runs this once per call, checks the device CC + alignment + dtype + K_PAD
// preconditions, and dispatches the kernel only when all hold. Anything else
// returns false and the caller falls back to the scalar K-packed kernel.
bool disco_cuda_fwd_dense_kpacked_wgmma_try(
    torch::Tensor inp,
    torch::Tensor pack_idx,
    torch::Tensor pack_val,
    torch::Tensor pack_count,
    torch::Tensor out,
    int64_t K, int64_t Ho, int64_t Wo)
{
    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    // WGMMA is Hopper-only (SM_90 / SM_90a). Skip pre-Hopper and Blackwell+.
    if (props.major != 9) return false;

    if (inp.scalar_type() != at::ScalarType::BFloat16) return false;

    const int64_t B  = inp.size(0);
    const int64_t C  = inp.size(1);
    const int64_t Hi = inp.size(2);
    const int64_t Wi = inp.size(3);

    if (Wi % Wo != 0)           return false;   // pscale must be integer
    if ((B * C) % 8 != 0)       return false;
    if (Wo % 8 != 0)            return false;
    if (pack_val.size(2) != 8)  return false;   // K_PAD==8 only

    constexpr int BC_TILE  = 8;
    constexpr int WO_TILE  = 8;
    constexpr int NZ_CHUNK = 16;
    constexpr int N_PAD    = 8;

    const int NBR_PAD = (int)pack_idx.size(1);

    // Shmem: 2048 B (A) + 256 B (B) = 2304 B per CTA.
    const size_t shmem_bytes = 2304;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Cast pack_val to bf16 if needed (psi values are stored fp32 by default).
    auto pack_val_bf16 = (pack_val.scalar_type() == at::ScalarType::BFloat16)
        ? pack_val
        : pack_val.to(at::ScalarType::BFloat16);

    auto fn = &disco_fwd_dense_kpacked_wgmma_blk_k<BC_TILE, WO_TILE, NZ_CHUNK, N_PAD>;
    cudaFuncSetAttribute(reinterpret_cast<const void*>(fn),
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)shmem_bytes);

    const int pscale = (int)(Wi / Wo);

    dim3 grid((unsigned)(Ho * (Wo / WO_TILE)), (unsigned)((B * C) / BC_TILE));
    fn<<<grid, 128, shmem_bytes, stream>>>(
        (int)Hi, (int)Wi, (int)K, (int)Ho, (int)Wo, NBR_PAD, pscale,
        pack_idx.data_ptr<int64_t>(),
        reinterpret_cast<const __nv_bfloat16*>(pack_val_bf16.data_ptr()),
        pack_count.data_ptr<int64_t>(),
        reinterpret_cast<const __nv_bfloat16*>(inp.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr())
    );
    return true;
}

}  // namespace disco_kernels
