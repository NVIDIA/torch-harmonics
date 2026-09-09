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
// WGMMA implementation. Restricted to:
//   - K_PAD ∈ {8, 16}              (n8 / n16 wgmma shapes; larger K_PADs need
//                                  more accumulator regs / wider B staging)
//   - pscale ≥ 1                   (any integer; gmem reads become strided when
//                                  pscale > 1, but correctness is preserved)
//   - bf16 OR fp16 inputs          (templated on T; selects the .f32.bf16.bf16
//                                   vs .f32.f16.f16 wgmma opcode)
//
// The host wrapper checks these and falls back to the scalar K-packed kernel
// otherwise.
//
// CTA layout
// ----------
// Threads: 128 (1 warp-group, mandatory for WGMMA)
// Output tile: BC_TILE=8 channels × WO_TILE=8 wo positions × N=K_PAD k_kerns
// Per-thread acc: N_PAD/2 fp32 (m64nNk16 distributes 64×N cells across 128 threads)
// Grid: (Ho × (Wo / WO_TILE), ceil(BC / BC_TILE))
// The last grid.y CTA may straddle the (B*C) boundary; rows with bc ≥ B*C
// are zero-padded in shmem and skipped at writeback.
//
// Shared memory layout (T = bf16 or fp16, both 2-byte; 16-byte aligned)
// ---------------------------------------------------------------------
// A_tile [M=64, K=16] — M-major:
//    byte(m, k) = (m / 8) * 256 + (m % 8 + 8 * k) * 2
//    total 2048 bytes
//
// B_tile [K=16, N=N_PAD] — N-group outer, then 8×8 cores with K-rows
// contiguous (one u128 per row). This matches the WGMMA Major::MN core-matrix
// layout, where each 8×8 core's K rows live in 8 consecutive 16-byte u128s:
//    byte(k, n) = (n / 8) * (16 * NZ_CHUNK) + k * 16 + (n % 8) * 2
//    total 32 * N_PAD bytes (256 for N=8, 512 for N=16)
// For N=8 this collapses to byte(k,n) = (k*8 + n)*2 (single n-group).
//
// Per nz_chunk inner loop
// -----------------------
// 1. Stage A_tile: for each (bc_local, nz_local) ∈ [0,8)×[0,16) — 128 cells —
//    each thread reads 8 narrow values (one wo_local strip at fixed bc, nz)
//    from gmem inp into shmem A_tile. For pscale ∈ {1, 2} those 8 values are
//    gathered from one aligned window of 32-bit words (see gather_window_rt);
//    pscale ≥ 3 and windows crossing the longitude seam use per-element reads.
// 2. Stage B_tile: 32 * N_PAD bytes total, in 16-byte chunks of 8 narrow values.
//    Each chunk is one (k_row, n-group) pair. Total chunks = NZ_CHUNK * (N_PAD/8):
//    16 for N=8, 32 for N=16. Distributed 1 chunk per active staging thread.
// 3. __syncthreads.
// 4. wgmma.fence; wgmma.mma_async m64n{8,16}k16; wgmma.commit_group; wgmma.wait_group<0>.
//
// Output write
// ------------
// After all nz_chunks, each thread writes its N_PAD/2 fp32 accumulator cells.
// WGMMA m64nNk16.f32 fragment-to-(m, n) mapping (PTX ISA §9.7.16.5.4): cells
// are organised into n-groups of 4 cells per 8 N-cols. Within an n-group ng:
//    warp w in [0,4), lane l in [0,32):
//      cell 4*ng+0: m = w*16 + l/4,     n = (l%4)*2     + 8*ng
//      cell 4*ng+1: m = w*16 + l/4,     n = (l%4)*2 + 1 + 8*ng
//      cell 4*ng+2: m = w*16 + l/4 + 8, n = (l%4)*2     + 8*ng
//      cell 4*ng+3: m = w*16 + l/4 + 8, n = (l%4)*2 + 1 + 8*ng
//    Our M dim is m = bc * WO_TILE + wo_local = bc * 8 + wo_local, so:
//      bc       = m / 8
//      wo_local = m % 8
//    Our N dim is n = k_kern.
// =====================================================================================

// TRIED AND REJECTED
// ------------------
// Measured on H100 (bf16, BC=64, 360x720 self-conv unless noted) through
// experimental kernels that used to sit alongside this file; all were verified
// bit-identical before timing. Recorded here so they are not re-derived.
//
//  - Warp-shuffle pack_idx sharing (16 lanes load, rest take it by shuffle):
//    saves no sectors at all. The memory system already deduplicates repeated
//    addresses across lanes within a warp, so 32 lanes hitting 16 distinct
//    entries touch the same lines as 16 lanes do. Pays for the shuffles and
//    gets nothing. 3.34 vs 3.13 ms.
//
//  - Shared-memory pack_idx staging: fewest instructions and fewest sectors of
//    any variant tried, and slower. The extra __syncthreads converted
//    long-scoreboard stall into barrier stall almost one for one (4.51 -> 2.46
//    against 2.54 -> 4.73). 3.24 vs 3.13 ms.
//
//  - Merging the A-tile stores into one STS.128: a no-op. Every counter
//    identical to the plain vector-load variant, because ptxas already
//    vectorizes those stores. The shared side was never a plausible ceiling
//    regardless -- 27.4M shared-store instructions against 102.3M global loads.
//
//  - Windowed gather at pscale 3: needs 12 words to deliver 8 halfwords, so
//    loads rise ~45% while instructions fall ~23%. That trades well only where
//    L1 has slack: -1% at 540x1080 -> 180x360 (L1 68.7%) but +13% at
//    1080x2160 -> 360x720 (L1 82.2%). Same code, opposite sign -- hence the
//    pscale <= 2 gate on the fast path below.
//
//  - Cross-thread run staging: load the covering run once per (bc, chunk) into
//    shared memory and have each thread gather its window from there, removing
//    the ~13x cross-thread redundancy. It works -- sectors -22% at pscale 1 and
//    -36% at pscale 2 -- but costs +24% instructions, leaving it neutral at
//    pscale 1 and worth 7% at pscale 2 only. Also structurally capped: just
//    50-60% of chunks have all 16 neighbours inside one arc, the rest fall back.
//
// Not worth attacking: K_PAD=16 against K=9 wastes 44% of every WGMMA, but the
// tensor pipe runs at 4.4% of peak, so eliminating all of it would save ~2% of
// cycles. It is structural anyway -- inp carries (bc, wo) and val carries k, so
// M must come from {bc, wo} and N from {k}, and there is no wgmma n shape
// between 8 and 16.
// =====================================================================================

#include "../disco.h"
#include "disco_cuda.cuh"
#include "disco_cuda_ptx.cuh"

#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace disco_kernels
{

    // gather_window_rt now lives in disco_cuda.cuh, shared with the SM_100a kernel.

    // Kernel symbol exists on all archs so the host launcher compiles cleanly;
    // the body is empty for non-Hopper builds (and that path is never launched —
    // see the runtime CC check in disco_cuda_fwd_dense_kpacked_sm90_try).
    // T ∈ {__nv_bfloat16, __half} — selects the .f32.bf16.bf16 vs .f32.f16.f16
    // WGMMA opcode and the matching narrow→float / float→narrow conversions.
    //
    // BC_total = B*C is the actual number of (batch×channel) rows in the input.
    // Grid.y = ceil(BC_total / BC_TILE), so the last CTA may straddle the BC
    // boundary; rows with bc ≥ BC_total are zero-padded in shmem and skipped at
    // writeback.
    template <int BC_TILE, int WO_TILE, int NZ_CHUNK, int N_PAD, typename T>
    __global__ __launch_bounds__(128) void disco_fwd_dense_kpacked_wgmma_blk_k(
        int Hi, int Wi, int K, int Ho, int Wo, int pscale, int BC_total,
        const int64_t *__restrict__ pack_idx,    // [nnz, 2]
        const T *__restrict__ pack_val,          // [nnz, N_PAD]  (=K_PAD)
        const int64_t *__restrict__ pack_offset, // [Ho + 1]
        const T *__restrict__ inp,               // [B, C, Hi, Wi]
        T *__restrict__ out)                     // [B, C, K, Ho, Wo]
    {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
        static_assert(N_PAD == 8 || N_PAD == 16, "WGMMA path: only K_PAD ∈ {8, 16} supported");
        static_assert(BC_TILE == 8 && WO_TILE == 8, "WGMMA path: tile must be 8×8 for M=64");

        constexpr int N_ACC = N_PAD / 2; // fp32 acc cells per thread (4 or 8)

        const int tid = threadIdx.x;
        const int warp_id = tid / 32;
        const int lane = tid - warp_id * 32;

        const int wo_per_ho = Wo / WO_TILE;
        const int ho = blockIdx.x / wo_per_ho;
        const int wo_strip = blockIdx.x - ho * wo_per_ho;
        const int wo_base = wo_strip * WO_TILE;
        const int bc_start = blockIdx.y * BC_TILE;

        // Blocked CSR: rows are packed back to back, so the base comes from the
        // offset array rather than a constant NBR_PAD stride, and cnt is the gap
        // between consecutive offsets. Both loads are CTA-uniform broadcasts.
        const int64_t off_beg = pack_offset[ho];
        const int cnt = (int)(pack_offset[ho + 1] - off_beg);
        const int64_t *idx_ho = pack_idx + off_beg * 2;
        const T *val_ho = pack_val + off_beg * N_PAD;

        // Per-thread accumulator: N_PAD/2 fp32 cells for m64nNk16.
        float acc[N_ACC];
#pragma unroll
        for (int i = 0; i < N_ACC; i++) acc[i] = 0.0f;

        if (cnt == 0) {
            // No-op fall through to writeback (which writes zeros).
        }

        // Shared memory: A_tile (2048 B) + B_tile (32 * N_PAD bytes) per CTA.
        // bf16 and fp16 are both 2-byte types; sizes don't depend on T.
        extern __shared__ __align__(16) unsigned char shmem_raw[];
        T *A_tile = reinterpret_cast<T *>(shmem_raw);
        T *B_tile = A_tile + (BC_TILE * 8) * NZ_CHUNK; // 1024 elements of A

        // ----------------------- nz_chunk loop -----------------------
        for (int nz_chunk_off = 0; nz_chunk_off < cnt; nz_chunk_off += NZ_CHUNK) {

            // -- Stage A_tile (128 cells = 1 per thread) --
            // Each thread copies 8 narrow values (one wo_local strip at fixed bc,
            // nz) from gmem inp into A_tile shmem.
            const int bc_local = tid / NZ_CHUNK;            // [0, 8)
            const int nz_local = tid - bc_local * NZ_CHUNK; // [0, 16)
            const int nz_global = nz_chunk_off + nz_local;

            const int bc = bc_start + bc_local;

            if (nz_global < cnt && bc < BC_total) {
                // (hi, wi_base) are adjacent in pack_idx, and idx_ho = pack_idx +
                // off_beg*2 int64 is 16-byte aligned for any offset, so the pair can
                // be fetched with one vector load rather than two scalar ones. Both
                // scalar loads span the same 256 B the warp touches (16 distinct nz
                // at stride 16 B), so merging them halves the sectors as well as the
                // instructions: measured 113.3M -> 102.3M global load instructions
                // and 567.8M -> 525.6M sectors on H100 at 360x720 self-conv, worth
                // ~3% (3.22 -> 3.13 ms). Verified bit-identical.
                const longlong2 idx_pair = reinterpret_cast<const longlong2 *>(idx_ho)[nz_global];
                const int hi = (int)idx_pair.x;
                const int wi_base = (int)idx_pair.y;
                const int64_t inp_row_base = (int64_t)bc * Hi * Wi + (int64_t)hi * Wi;
                T *dst = A_tile + bc_local * (8 * NZ_CHUNK) // bc_local * 128 elements
                    + nz_local * 8;                         // nz_local * 8 elements (= 16 bytes)
                // Fetch the eight elements as one aligned window instead of eight
                // scalar loads, each with its own IMAD / add / compare / select /
                // 64-bit-add chain. That address arithmetic dominated: the kernel is
                // instruction-issue bound (1.70G instructions at 71% of issue
                // capacity, only 6% of them loads). Measured on H100, bf16, BC=64,
                // bit-identical throughout:
                //   pscale 1   360x720 self          3.19 -> 2.65 ms   1.20x
                //   pscale 2   720x1440 -> 360x720  11.04 -> 10.61 ms  1.04x
                //
                // s is reduced into [0, Wi) FIRST. wi_base < Wi and wo_base*pscale <
                // Wi, so s < 2*Wi and one subtract suffices -- the same argument the
                // scalar loop below makes per element, hoisted. Skipping this leaves
                // s in [0, 2*Wi), where the guard rejects about half of all threads,
                // and a rejected lane makes its whole warp run the fallback *as well
                // as* the fast path.
                //
                // The guard needs the window to clear the longitude seam and the
                // trailing word read to stay in-row; s + 8P + 2 <= Wi covers both. It
                // rejects only genuine seam crossings, ~1.3% of threads, which fall
                // through to the scalar loop -- so correctness never depends on it.
                //
                // Word-aligned access needs the row base 4-byte aligned:
                // inp_row_base = Wi * (bc*Hi + hi), and Wi = pscale * Wo with Wo a
                // multiple of 8, so Wi is always even.
                int s = wi_base + wo_base * pscale;
                if (s >= Wi) s -= Wi;
                const uint32_t *p = reinterpret_cast<const uint32_t *>(inp + inp_row_base) + (s >> 1);
                int4 *dst4 = reinterpret_cast<int4 *>(dst);
                bool staged = true;

#define TH_KP_WINDOW(P_)                                                                                               \
    (pscale == (P_) && s + 8 * (P_) + 2 <= Wi) { *dst4 = gather_window_rt<(P_)>(p, s & 1); }

                if TH_KP_WINDOW (1) else if TH_KP_WINDOW (2) else
                    {
                        staged = false;
                    }

#undef TH_KP_WINDOW

                if (!staged) {
                    // pscale >= 3, or a window crossing the seam.
                    //
                    // wi_full wraps at most once: with wi_base <= Wi-1 and
                    // (wo_base+i) <= Wo-1, wi_full <= (Wi-1) + (Wo-1)*pscale
                    // <= 2*Wi - 1 - pscale. Relies on Wi == pscale*Wo, enforced by
                    // the host wrapper.
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        int wi_full = wi_base + (wo_base + i) * pscale;
                        if (wi_full >= Wi) wi_full -= Wi;
                        dst[i] = inp[inp_row_base + wi_full];
                    }
                }
            } else {
                // Zero pad for nz_global >= cnt — shmem dst is 16-byte aligned, so
                // a vector store is safe here.
                T *dst = A_tile + bc_local * (8 * NZ_CHUNK) + nz_local * 8;
                *reinterpret_cast<int4 *>(dst) = make_int4(0, 0, 0, 0);
            }

            // -- Stage B_tile (32 * N_PAD bytes total, in 16-byte chunks) --
            // Each K-row holds N_PAD bf16 = N_PAD/8 chunks of 8 bf16 (16 B). We
            // partition by N-group (chunk_idx = n / 8) as the OUTER stride so the
            // shmem layout matches WGMMA's Major::MN core-matrix layout, where the
            // K rows of one 8×8 core sit contiguously (one u128 per row):
            //
            //   byte(k, n) = (n / 8) * (16 * NZ_CHUNK)        // skip past prior n-groups
            //              +  k       *  16                   // K-row stride within group
            //              + (n % 8)  *   2;                  // N-fast within u128
            //
            // For N_PAD=8, n/8 == 0 always, so this collapses to k*16 + n*2 — same
            // as the prior staging. Total chunks = NZ_CHUNK * (N_PAD/8): 16 for
            // N=8, 32 for N=16. One chunk per active staging thread.
            constexpr int CHUNKS_PER_ROW = N_PAD / 8; // 1 for N=8, 2 for N=16
            constexpr int B_TOTAL_CHUNKS = NZ_CHUNK * CHUNKS_PER_ROW;
            if (tid < B_TOTAL_CHUNKS) {
                const int chunk_idx = tid / NZ_CHUNK; // n-group ∈ [0, CHUNKS_PER_ROW)
                const int nz_local_b = tid - chunk_idx * NZ_CHUNK;
                const int nz_global_b = nz_chunk_off + nz_local_b;
                T *dst_b = B_tile + chunk_idx * (NZ_CHUNK * 8) // n-group → outer stride
                    + nz_local_b * 8;                          // K-row stride within n-group
                if (nz_global_b < cnt) {
                    const T *src_b = val_ho + (int64_t)nz_global_b * N_PAD + chunk_idx * 8;
                    *reinterpret_cast<int4 *>(dst_b) = *reinterpret_cast<const int4 *>(src_b);
                } else {
                    *reinterpret_cast<int4 *>(dst_b) = make_int4(0, 0, 0, 0);
                }
            }

            fence_proxy_async_shared_cta();
            __syncthreads();

            // -- WGMMA m64nNk16, accumulating --
            // Descriptor LBO/SBO are byte strides between 8×8 core matrices in
            // shmem, stored in 16-byte (uint128_t) units. The CUTLASS reference
            // (NVIDIA/cutlass v4.4.0, include/cute/atom/mma_traits_sm90_gmma.hpp,
            // make_gmma_desc<Major::MN> INTERLEAVE) assigns:
            //   leading_byte_offset_ = K-outer stride (8-K-block to next 8-K-block)
            //   stride_byte_offset_  = M/N-outer stride (8-MN-block to next)
            // `make_wgmma_desc(ptr, leading, stride)` lands `leading` in bits
            // 16-29 and `stride` in bits 32-45.
            //
            // A is Major::MN (M-fast in core), B is Major::MN (N-fast in core).
            //
            //   A [M=64, K=16], byte(m,k) = (m/8)*256 + (m%8 + 8*k)*2:
            //     leading (K-outer): 128 B  → field  8
            //     stride  (M-outer): 256 B  → field 16
            //
            //   B [K=16, N=N_PAD], byte(k,n) = (n/8)*256 + k*16 + (n%8)*2:
            //     N=8 :  1 N-block, 2 K-blocks (K-stride 128 B within n-group)
            //              leading=8, stride=16 (canonical N-outer stride)
            //     N=16:  2 N-blocks, 2 K-blocks
            //              K-stride 128 B within n-group, N-stride 256 B between
            //              leading=8, stride=16
            wgmma_fence();
            constexpr uint32_t A_LEADING_FIELD = 8;
            constexpr uint32_t A_STRIDE_FIELD = 16;
            constexpr uint32_t B_LEADING_FIELD = 8; // K-outer 128 B
            constexpr uint32_t B_STRIDE_FIELD = 16; // N-outer 256 B
            uint64_t desc_a = make_wgmma_desc(A_tile, A_LEADING_FIELD * 16, A_STRIDE_FIELD * 16);
            uint64_t desc_b = make_wgmma_desc(B_tile, B_LEADING_FIELD * 16, B_STRIDE_FIELD * 16);
            const int32_t scale_D = (nz_chunk_off == 0) ? 0 : 1;
            if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                if constexpr (N_PAD == 8) {
                    wgmma_m64n8k16_acc_bf16(acc, desc_a, desc_b, scale_D);
                } else { // N_PAD == 16
                    wgmma_m64n16k16_acc_bf16(acc, desc_a, desc_b, scale_D);
                }
            } else { // T == __half
                if constexpr (N_PAD == 8) {
                    wgmma_m64n8k16_acc_fp16(acc, desc_a, desc_b, scale_D);
                } else { // N_PAD == 16
                    wgmma_m64n16k16_acc_fp16(acc, desc_a, desc_b, scale_D);
                }
            }
            wgmma_commit_group();
            wgmma_wait_group<0>();
            wgmma_fence();

            __syncthreads();
        }

        // ----------------------- writeback -----------------------
        // m64nNk16.f32 fragment-to-(m, n) mapping (PTX ISA §9.7.16.5.4): each
        // thread holds N/2 fp32 cells, organised into n-groups of 4 cells per
        // 8 N-cols. Within a group:
        //   cell 4*ng+0: m = w*16 + l/4,     n = (l%4)*2     + 8*ng
        //   cell 4*ng+1: m = w*16 + l/4,     n = (l%4)*2 + 1 + 8*ng
        //   cell 4*ng+2: m = w*16 + l/4 + 8, n = (l%4)*2     + 8*ng
        //   cell 4*ng+3: m = w*16 + l/4 + 8, n = (l%4)*2 + 1 + 8*ng
        const int m01 = warp_id * 16 + (lane >> 2); // m for cells 0, 1
        const int m23 = m01 + 8;                    // m for cells 2, 3
        const int n_a = (lane & 3) * 2;             // base n for "even" cells
        const int n_b = n_a + 1;                    // base n for "odd"  cells

        const int bc01 = bc_start + (m01 >> 3); // m / 8
        const int wo01 = wo_base + (m01 & 7);   // m % 8
        const int bc23 = bc_start + (m23 >> 3);
        const int wo23 = wo_base + (m23 & 7);

        auto write_cell = [&](int bc_o, int wo_o, int k_o, float v) {
            if (k_o >= K) return;         // n in [K, K_PAD) is padding; skip.
            if (bc_o >= BC_total) return; // last CTA may straddle the BC boundary.
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
        // Non-sm_90a target: WGMMA opcodes aren't available here. This includes
        // pre-Hopper, plain sm_90 (without the `a` suffix), and Blackwell+.
        // Empty body satisfies the linker; the host runtime check below ensures
        // the kernel is never launched on devices where its body is empty.
        (void)Hi;
        (void)Wi;
        (void)K;
        (void)Ho;
        (void)Wo;
        (void)pscale;
        (void)BC_total;
        (void)pack_idx;
        (void)pack_val;
        (void)pack_offset;
        (void)inp;
        (void)out;
#endif
    }

    // Forward declaration of the SM_100a launcher (defined in disco_cuda_fwd_dense_kpacked_sm100.cu).
    torch::Tensor disco_cuda_fwd_kpacked_sm100(torch::Tensor inp, torch::Tensor pack_idx, torch::Tensor pack_val,
                                               torch::Tensor pack_offset, int64_t K, int64_t Ho, int64_t Wo);

    // Torch op: runtime-dispatches to SM_90a (WGMMA) or SM_100a (tcgen05) based on
    // the current device's compute capability. Falls back with TORCH_CHECK on
    // unsupported architectures.
    torch::Tensor disco_cuda_fwd_kpacked(torch::Tensor inp,
                                         torch::Tensor pack_idx,    // [nnz, 2]             int64
                                         torch::Tensor pack_val,    // [nnz, K_PAD] fp16/bf16
                                         torch::Tensor pack_offset, // [Ho]                 int64
                                         int64_t K, int64_t Ho, int64_t Wo)
    {
        CHECK_CUDA_INPUT_TENSOR(inp);
        CHECK_CUDA_INPUT_TENSOR(pack_idx);
        CHECK_CUDA_INPUT_TENSOR(pack_val);
        CHECK_CUDA_INPUT_TENSOR(pack_offset);

        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, inp.get_device());
        TORCH_CHECK(props.major == 9 || props.major == 10,
                    "disco_kernels::forward_kpacked requires SM_90a (Hopper) or SM_100a (Blackwell); got SM_",
                    props.major, ".", props.minor);

        // Dispatch to SM_100a (tcgen05) path on Blackwell.
        if (props.major == 10) { return disco_cuda_fwd_kpacked_sm100(inp, pack_idx, pack_val, pack_offset, K, Ho, Wo); }

        const auto inp_dtype = inp.scalar_type();
        TORCH_CHECK(inp_dtype == at::ScalarType::BFloat16 || inp_dtype == at::ScalarType::Half,
                    "disco_kernels::forward_kpacked requires bf16 or fp16 input");

        const int64_t B = inp.size(0);
        const int64_t C = inp.size(1);
        const int64_t Hi = inp.size(2);
        const int64_t Wi = inp.size(3);

        TORCH_CHECK(Wi % Wo == 0, "Wi (", Wi, ") must be divisible by Wo (", Wo, ")");
        TORCH_CHECK(Wo % 8 == 0, "Wo (", Wo, ") must be divisible by 8");

        const int64_t K_PAD = pack_val.size(1); // pack_val is [nnz, K_PAD]
        TORCH_CHECK(K_PAD == 8 || K_PAD == 16, "K_PAD must be 8 or 16, got ", K_PAD);

        constexpr int BC_TILE = 8;
        constexpr int WO_TILE = 8;
        constexpr int NZ_CHUNK = 16;

        int64_t out_dims[] = {B, C, K, Ho, Wo};
        auto out = torch::zeros(out_dims, torch::TensorOptions().device(inp.device()).dtype(inp_dtype));

        const size_t shmem_bytes = 2048 + 32 * (size_t)K_PAD;
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
                (int)Hi, (int)Wi, (int)K, (int)Ho, (int)Wo, pscale, BC_total, pack_idx.data_ptr<int64_t>(),
                reinterpret_cast<const T *>(pack_val_cast.data_ptr()), pack_offset.data_ptr<int64_t>(),
                reinterpret_cast<const T *>(inp.data_ptr()), reinterpret_cast<T *>(out.data_ptr()));
        };

        if (inp_dtype == at::ScalarType::BFloat16) {
            if (K_PAD == 8)
                launch(&disco_fwd_dense_kpacked_wgmma_blk_k<BC_TILE, WO_TILE, NZ_CHUNK, 8, __nv_bfloat16>,
                       __nv_bfloat16 {});
            else
                launch(&disco_fwd_dense_kpacked_wgmma_blk_k<BC_TILE, WO_TILE, NZ_CHUNK, 16, __nv_bfloat16>,
                       __nv_bfloat16 {});
        } else {
            if (K_PAD == 8)
                launch(&disco_fwd_dense_kpacked_wgmma_blk_k<BC_TILE, WO_TILE, NZ_CHUNK, 8, __half>, __half {});
            else
                launch(&disco_fwd_dense_kpacked_wgmma_blk_k<BC_TILE, WO_TILE, NZ_CHUNK, 16, __half>, __half {});
        }

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return out;
    }

    TORCH_LIBRARY_IMPL(disco_kernels, CUDA, m) { m.impl("forward_kpacked", &disco_cuda_fwd_kpacked); }

} // namespace disco_kernels
