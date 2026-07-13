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
// Disco forward — K-packed dense psi (CUDA, tcgen05 path, Blackwell SM_100a)
// =====================================================================================
//
// Structurally identical to the SM_90a WGMMA kernel in
// disco_cuda_fwd_dense_kpacked_sm90.cu. The SMEM layout, staging loop,
// descriptor construction, and writeback are unchanged. The only differences:
//
//   SM_90a (Hopper)         SM_100a (Blackwell)
//   ──────────────────────  ──────────────────────────────────────────────────
//   wgmma.mma_async         tcgen05.mma (issued by one elected CTA thread)
//   accumulator: registers  accumulator: TMEM (Tensor Memory, 32 cols minimum)
//   wgmma fence/commit/wait tcgen05.commit → mbarrier → mbarrier.try_wait
//   (no alloc/dealloc)      tcgen05.alloc / tcgen05.dealloc
//
// TMEM lifecycle (see disco_cuda_ptx.cuh for wrapper implementations):
//   tcgen05.alloc   — first warp allocates 32 TMEM columns; result written to shmem.
//   tcgen05.relinquish_alloc_permit — all 128 threads release the allocation permit.
//   tcgen05.st      — zero-initialise the tile (scaleC=1 is used throughout,
//                     so we need explicit zero-init for cnt==0 correctness).
//   tcgen05.mma     — one elected thread from the first warp; scaleC=1 (accumulate into zeros).
//   tcgen05.commit  — same elected thread signals the mbarrier.
//   mbarrier.try_wait — all 128 threads spin-poll; safe to read TMEM when done.
//   tcgen05.ld      — warpgroup-collective TMEM read back to registers.
//   tcgen05.dealloc — first warp releases allocation after writeback.
//
// Restrictions (same as SM_90a path):
//   K_PAD ∈ {8, 16}, BC_TILE = WO_TILE = 8, bf16 or fp16 input.
//
// Must be compiled with -arch=sm_100a (TORCH_CUDA_ARCH_LIST="10.0a+PTX").
// =====================================================================================

#include "../disco.h"
#include "disco_cuda.cuh"
#include "disco_cuda_ptx.cuh"

#include <ATen/Dispatch.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace disco_kernels
{

    template <int BC_TILE, int WO_TILE, int NZ_CHUNK, int N_PAD, typename T>
    __global__ __launch_bounds__(128) void disco_fwd_dense_kpacked_tcgen05_blk_k(
        int Hi, int Wi, int K, int Ho, int Wo, int NBR_PAD, int pscale, int BC_total,
        const int64_t *__restrict__ pack_idx,   // [Ho, NBR_PAD, 2]
        const T *__restrict__ pack_val,         // [Ho, NBR_PAD, N_PAD]
        const int64_t *__restrict__ pack_count, // [Ho]
        const T *__restrict__ inp,              // [B, C, Hi, Wi]
        T *__restrict__ out)                    // [B, C, K, Ho, Wo]
    {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
        static_assert(N_PAD == 8 || N_PAD == 16, "tcgen05 path: only K_PAD ∈ {8, 16} supported");
        static_assert(BC_TILE == 8 && WO_TILE == 8, "tcgen05 path: tile must be 8×8 for M=64");

        constexpr int N_ACC = N_PAD / 2; // fp32 accumulator cells per thread (4 or 8)

        const int tid = threadIdx.x;
        const int warp_id = tid / 32;
        const int lane = tid - warp_id * 32;

        const int wo_per_ho = Wo / WO_TILE;
        const int ho = blockIdx.x / wo_per_ho;
        const int wo_strip = blockIdx.x - ho * wo_per_ho;
        const int wo_base = wo_strip * WO_TILE;
        const int bc_start = blockIdx.y * BC_TILE;

        const int64_t *idx_ho = pack_idx + (int64_t)ho * NBR_PAD * 2;
        const T *val_ho = pack_val + (int64_t)ho * NBR_PAD * N_PAD;
        const int cnt = (int)pack_count[ho];

        // ─── Shared memory layout ──────────────────────────────────────────────
        // tcgen05.mma requires 128-byte aligned SMEM descriptors on SM100.
        // We guarantee this by keeping NO static __shared__ variables (so the
        // CUDA runtime places dynamic SMEM at SMEM offset 0, which is always
        // 128-byte aligned) and starting A_tile at shmem_raw+128.
        //
        // Layout within shmem_raw:
        //   [  0.. 3]  uint32_t smem_tmem_base
        //   [  8..15]  uint64_t smem_mma_mbar
        //   [ 16..127] padding (112 B)
        //   [128..128 + BC_TILE*NZ_CHUNK*8*sizeof(T) - 1]  A_tile  (128-B aligned)
        //   [128 + A_bytes .. + NZ_CHUNK*N_PAD*sizeof(T)]  B_tile  (128-B aligned)
        extern __shared__ unsigned char shmem_raw[];
        uint32_t *smem_tmem_base_ptr = reinterpret_cast<uint32_t *>(shmem_raw + 0);
        uint64_t *smem_mma_mbar_ptr = reinterpret_cast<uint64_t *>(shmem_raw + 8);
        T *A_tile = reinterpret_cast<T *>(shmem_raw + 128);
        T *B_tile = A_tile + (BC_TILE * 8) * NZ_CHUNK; // immediately after A

        // ─── Allocate TMEM ─────────────────────────────────────────────────────
        // tcgen05_alloc issues the PTX from the first warp and __syncthreads()
        // internally before returning the base address to all threads.
        uint32_t tmem_acc = tcgen05_alloc(smem_tmem_base_ptr);

        // Release the allocation permit once the TMEM base is known. Keeping it
        // until dealloc serializes allocation across CTAs and collapses achieved
        // occupancy to roughly one CTA per SM.
        tcgen05_relinquish_alloc_permit();

        // ─── Zero-init TMEM accumulator ────────────────────────────────────────
        // All threads participate; covers the full M=64, N=N_PAD tile.
        tcgen05_zero<N_ACC>(tmem_acc);

        // ─── Init MMA mbarrier ─────────────────────────────────────────────────
        uint32_t mbar_ptr = __cvta_generic_to_shared(smem_mma_mbar_ptr);
        if (tid == 0) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(mbar_ptr) : "memory"); }
        tcgen05_fence_mbarrier_init(); // makes init visible to the async tracking HW
        __syncthreads();

        // ─── nz_chunk loop (A/B staging identical to SM_90a) ───────────────────
        for (int nz_chunk_off = 0; nz_chunk_off < cnt; nz_chunk_off += NZ_CHUNK) {

            // -- Stage A_tile: one row per BC, one col per nz in this chunk --
            const int bc_local = tid / NZ_CHUNK;
            const int nz_local = tid - bc_local * NZ_CHUNK;
            const int nz_global = nz_chunk_off + nz_local;
            const int bc = bc_start + bc_local;

            if (nz_global < cnt && bc < BC_total) {
                const int hi = (int)idx_ho[nz_global * 2 + 0];
                const int wi_base = (int)idx_ho[nz_global * 2 + 1];
                const int64_t inp_row_base = (int64_t)bc * Hi * Wi + (int64_t)hi * Wi;
                T *dst = A_tile + bc_local * (8 * NZ_CHUNK) + nz_local * 8;
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    int wi_full = wi_base + (wo_base + i) * pscale;
                    if (wi_full >= Wi) wi_full -= Wi;
                    dst[i] = inp[inp_row_base + wi_full];
                }
            } else {
                T *dst = A_tile + bc_local * (8 * NZ_CHUNK) + nz_local * 8;
                *reinterpret_cast<int4 *>(dst) = make_int4(0, 0, 0, 0);
            }

            // -- Stage B_tile: nz × K_PAD values --
            constexpr int CHUNKS_PER_ROW = N_PAD / 8;
            constexpr int B_TOTAL_CHUNKS = NZ_CHUNK * CHUNKS_PER_ROW;
            if (tid < B_TOTAL_CHUNKS) {
                const int chunk_idx = tid / NZ_CHUNK;
                const int nz_local_b = tid - chunk_idx * NZ_CHUNK;
                const int nz_global_b = nz_chunk_off + nz_local_b;
                T *dst_b = B_tile + chunk_idx * (NZ_CHUNK * 8) + nz_local_b * 8;
                if (nz_global_b < cnt) {
                    const T *src_b = val_ho + (int64_t)nz_global_b * N_PAD + chunk_idx * 8;
                    *reinterpret_cast<int4 *>(dst_b) = *reinterpret_cast<const int4 *>(src_b);
                } else {
                    *reinterpret_cast<int4 *>(dst_b) = make_int4(0, 0, 0, 0);
                }
            }

            fence_proxy_async_shared_cta();
            __syncthreads(); // A/B tiles ready

            // -- Build SM100 UMMA SMEM descriptors --
            constexpr uint32_t A_LEADING_FIELD = 8;
            constexpr uint32_t A_STRIDE_FIELD = 16;
            constexpr uint32_t B_LEADING_FIELD = 8;
            constexpr uint32_t B_STRIDE_FIELD = 16;
            uint64_t desc_a = make_umma_desc(A_tile, A_LEADING_FIELD * 16, A_STRIDE_FIELD * 16);
            uint64_t desc_b = make_umma_desc(B_tile, B_LEADING_FIELD * 16, B_STRIDE_FIELD * 16);
            constexpr uint32_t mma_idesc = make_tcgen05_f16bf16_instr_desc<N_PAD, std::is_same_v<T, __nv_bfloat16>>();

            // -- Issue tcgen05.mma (thread 0) and commit to mbarrier --
            // scaleC=1: accumulate into the TMEM tile (which was zero-initialised).
            tcgen05_mma_issue(tmem_acc, desc_a, desc_b, mma_idesc, 1u, mbar_ptr);

            // -- All threads wait for MMA to complete --
            tcgen05_mma_wait(mbar_ptr);
            __syncthreads(); // ensure all threads exited wait before mbarrier reinit

            // Reinit mbarrier for the next chunk (if any).
            if (nz_chunk_off + NZ_CHUNK < cnt) {
                if (tid == 0) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(mbar_ptr) : "memory"); }
                tcgen05_fence_mbarrier_init();
                __syncthreads(); // reinit visible before next commit
            }
        }

        // ─── Read accumulator from TMEM ────────────────────────────────────────
        // 16x256b.x1 → 4 fp32 per thread (N_PAD=8)
        // 16x256b.x2 → 8 fp32 per thread (N_PAD=16)
        // mbarrier.try_wait already guarantees MMA completion; no additional fence
        // needed here. tcgen05_ld itself issues tcgen05.wait::ld internally.
        float acc[N_ACC];
        tcgen05_ld<N_ACC>(acc, tmem_acc);

        // ─── Release TMEM ──────────────────────────────────────────────────────
        tcgen05_dealloc(tmem_acc);

        // ─── Writeback (identical layout to SM_90a accumulator) ────────────────
        // Thread mapping for M=64, N=N_PAD, 128 threads:
        //   Each thread covers two M-rows (m01, m23 = m01+8) and two N-cols (n_a, n_b).
        const int m01 = warp_id * 16 + (lane >> 2);
        const int m23 = m01 + 8;
        const int n_a = (lane & 3) * 2;
        const int n_b = n_a + 1;

        const int bc01 = bc_start + (m01 >> 3);
        const int wo01 = wo_base + (m01 & 7);
        const int bc23 = bc_start + (m23 >> 3);
        const int wo23 = wo_base + (m23 & 7);

        auto write_cell = [&](int bc_o, int wo_o, int k_o, float v) {
            if (k_o >= K) return;
            if (bc_o >= BC_total) return;
            T narrow;
            if constexpr (std::is_same_v<T, __nv_bfloat16>)
                narrow = __float2bfloat16(v);
            else
                narrow = __float2half(v);
            out[((int64_t)bc_o * K + k_o) * Ho * Wo + (int64_t)ho * Wo + wo_o] = narrow;
        };

#pragma unroll
        for (int ng = 0; ng < N_PAD / 8; ng++) {
            const int n0 = n_a + 8 * ng;
            const int n1 = n_b + 8 * ng;
            write_cell(bc01, wo01, n0, acc[ng * 4 + 0]);
            write_cell(bc01, wo01, n1, acc[ng * 4 + 1]);
            write_cell(bc23, wo23, n0, acc[ng * 4 + 2]);
            write_cell(bc23, wo23, n1, acc[ng * 4 + 3]);
        }
#else
        // Non-sm_100a target: body intentionally empty.
        (void)Hi;
        (void)Wi;
        (void)K;
        (void)Ho;
        (void)Wo;
        (void)NBR_PAD;
        (void)pscale;
        (void)BC_total;
        (void)pack_idx;
        (void)pack_val;
        (void)pack_count;
        (void)inp;
        (void)out;
#endif
    }

    // Host launcher — called from the dispatcher in disco_interface.cpp.
    torch::Tensor disco_cuda_fwd_kpacked_sm100(torch::Tensor inp, torch::Tensor pack_idx, torch::Tensor pack_val,
                                               torch::Tensor pack_count, int64_t K, int64_t Ho, int64_t Wo)
    {
        const auto inp_dtype = inp.scalar_type();
        TORCH_CHECK(inp_dtype == at::ScalarType::BFloat16 || inp_dtype == at::ScalarType::Half,
                    "disco_kernels::forward_kpacked (SM_100a) requires bf16 or fp16 input");

        const int64_t B = inp.size(0);
        const int64_t C = inp.size(1);
        const int64_t Hi = inp.size(2);
        const int64_t Wi = inp.size(3);

        TORCH_CHECK(Wi % Wo == 0, "Wi (", Wi, ") must be divisible by Wo (", Wo, ")");
        TORCH_CHECK(Wo % 8 == 0, "Wo (", Wo, ") must be divisible by 8");

        const int64_t K_PAD = pack_val.size(2);
        TORCH_CHECK(K_PAD == 8 || K_PAD == 16, "K_PAD must be 8 or 16, got ", K_PAD);

        constexpr int BC_TILE = 8;
        constexpr int WO_TILE = 8;
        constexpr int NZ_CHUNK = 16;

        const int NBR_PAD = (int)pack_idx.size(1);
        int64_t out_dims[] = {B, C, K, Ho, Wo};
        auto out = torch::zeros(out_dims, torch::TensorOptions().device(inp.device()).dtype(inp_dtype));

        // Dynamic shmem layout: 128-byte header (tmem_base + mma_mbar + pad)
        //                     + A_tile (BC_TILE*NZ_CHUNK*8 elements)
        //                     + B_tile (NZ_CHUNK*K_PAD elements).
        // No static __shared__ variables — dynamic SMEM starts at SMEM offset 0,
        // guaranteeing 128-byte alignment for A_tile at shmem_raw+128.
        const size_t shmem_bytes = 128                        // header (tmem_base + mbar + pad)
            + (size_t)BC_TILE * NZ_CHUNK * 8 * sizeof(__half) // A_tile
            + (size_t)NZ_CHUNK * K_PAD * sizeof(__half);      // B_tile
        // = 128 + 2048 + 32*K_PAD bytes

        auto stream = at::cuda::getCurrentCUDAStream().stream();
        auto pack_val_cast = (pack_val.scalar_type() == inp_dtype) ? pack_val : pack_val.to(inp_dtype);

        const int pscale = (int)(Wi / Wo);
        const int BC_total = (int)(B * C);
        const int bc_blocks = (BC_total + BC_TILE - 1) / BC_TILE;
        const dim3 grid((unsigned)(Ho * (Wo / WO_TILE)), (unsigned)bc_blocks);

        auto launch = [&](auto fn, auto T_tag) {
            using T = decltype(T_tag);
            cudaFuncSetAttribute(reinterpret_cast<const void *>(fn), cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)shmem_bytes);
            fn<<<grid, 128, shmem_bytes, stream>>>(
                (int)Hi, (int)Wi, (int)K, (int)Ho, (int)Wo, NBR_PAD, pscale, BC_total, pack_idx.data_ptr<int64_t>(),
                reinterpret_cast<const T *>(pack_val_cast.data_ptr()), pack_count.data_ptr<int64_t>(),
                reinterpret_cast<const T *>(inp.data_ptr()), reinterpret_cast<T *>(out.data_ptr()));
        };

        if (inp_dtype == at::ScalarType::BFloat16) {
            if (K_PAD == 8)
                launch(&disco_fwd_dense_kpacked_tcgen05_blk_k<BC_TILE, WO_TILE, NZ_CHUNK, 8, __nv_bfloat16>,
                       __nv_bfloat16 {});
            else
                launch(&disco_fwd_dense_kpacked_tcgen05_blk_k<BC_TILE, WO_TILE, NZ_CHUNK, 16, __nv_bfloat16>,
                       __nv_bfloat16 {});
        } else {
            if (K_PAD == 8)
                launch(&disco_fwd_dense_kpacked_tcgen05_blk_k<BC_TILE, WO_TILE, NZ_CHUNK, 8, __half>, __half {});
            else
                launch(&disco_fwd_dense_kpacked_tcgen05_blk_k<BC_TILE, WO_TILE, NZ_CHUNK, 16, __half>, __half {});
        }
        return out;
    }

} // namespace disco_kernels
