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
// EXPERIMENTAL — Disco forward, K-packed dense psi, pack_idx staging variants
// =====================================================================================
//
// Not wired into the module forward. Registered as a bare op
// (disco_kernels::forward_kpacked_exp) with no PT2 compliance tag, no meta/fake
// kernel, no autograd and no autocast, so it can be driven and profiled directly
// from performance/disco/ncu_disco.py. If a variant wins, it gets folded back
// into disco_cuda_fwd_dense_kpacked_sm90.cu and this file goes away.
//
// SM_90a (WGMMA) only. The SM_100a tcgen05 path has the identical staging block
// and would need the same treatment; there is no point porting it before the
// Hopper measurement says which variant is worth keeping.
//
// Everything outside the pack_idx staging block — A/B tile layout, descriptors,
// the WGMMA itself, writeback — is byte-for-byte the SM_90a kernel. Variant 0 is
// literally the production staging code, so it serves as an in-harness control:
// any difference between variant 0 here and the production kernel is measurement
// noise, not a code difference.
//
// Why pack_idx staging
// --------------------
// ncu on the production kernel (H100, hdeg_self_tc003, bf16, BC=64):
//
//   l1tex__throughput                    79.9 %   <- the ceiling
//   gpu__dram_throughput                  3.05%
//   sm__warps_active                     71.5 %
//   duration                              3.22 ms
//   smsp__inst_executed_op_global_ld  113,271,840
//   l1tex__t_sectors...global_op_ld   650,773,440  (5.75 sectors/request)
//
// The kernel is L1-throughput bound: not DRAM, not latency, not occupancy. The
// instruction count decomposes exactly (2,737,440 CTA-chunks):
//
//   A staging   32 warp-inst/CTA-chunk   87,598,080   77.3 %
//   pack_idx     8 warp-inst/CTA-chunk   21,899,520   19.3 %
//   B staging    1 warp-inst/CTA-chunk    2,737,440    2.4 %
//   pack_count   4 warp-inst/CTA           1,036,800    0.9 %
//                                        -----------
//                                        113,271,840   (measured: identical)
//
// pack_idx is ~27% of L1 *sectors*, which is worse than its instruction share:
// each `idx_ho[nz*2+i]` warp-instruction reads 8 B at stride 16 B across 16
// distinct nz, spanning 256 B = 8 sectors, and every one of the 4 warps issues
// both halves. So 64 sectors (2 KB of L1 traffic) per CTA-chunk to deliver the
// 256 B the CTA needs exactly once.
//
// This file only changes that. The A-staging redundancy (66% of sectors, and the
// larger prize) is a separate, bigger change that needs host-side run
// descriptors; doing pack_idx first isolates one variable and tests whether the
// sector model actually predicts the speedup.
//
// Variants (IDX_MODE)
// -------------------
//   0 DIRECT      production code: 2 scalar loads per thread.
//                 8 warp-inst, 64 sectors per CTA-chunk.
//   1 VEC         one vector load of the adjacent (hi, wi) pair per thread.
//                 Halves instructions, same sectors: isolates instruction count
//                 from sector count.        4 warp-inst, 64 sectors (int64).
//   2 SHFL        lanes 0..15 of each warp load the chunk's 16 pairs; the other
//                 16 lanes take them by shuffle. No extra barrier.
//                                           4 warp-inst, 32 sectors (int64).
//   3 SMEM        threads 0..15 of the CTA load all 16 pairs into shmem, shared
//                 by all 4 warps. Costs one extra __syncthreads per chunk.
//                                           1 warp-inst,  8 sectors (int64).
//
// Orthogonal axis: pack_idx may be passed as int64 (as the module builds it) or
// int32, which halves the sector count of every variant. hi < nlat_in and
// wi < nlon_in both fit int32 with room to spare.
//
// The interesting question is 2 vs 3: SMEM moves 8x less than SHFL but adds a
// barrier per chunk, and the barrier stall is already 2.44. That trade is what
// the measurement is for.
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

    // pack_idx staging modes; see the header comment.
    enum IdxMode : int {
        IDX_DIRECT = 0,
        IDX_VEC = 1,
        IDX_SHFL = 2,
        IDX_SMEM = 3,
    };

    // Load the (hi, wi_base) pair for neighbour nzg as one vector access. The two
    // components are adjacent in pack_idx, and pack_idx rows are 16-byte aligned
    // (row stride NBR_PAD*2 elements), so the wide load is always legal.
    template <typename IDX_T>
    __device__ __forceinline__ void load_idx_pair(const IDX_T *idx_ho, int nzg, int &hi, int &wb)
    {
        if constexpr (sizeof(IDX_T) == 8) {
            const longlong2 v = reinterpret_cast<const longlong2 *>(idx_ho)[nzg];
            hi = (int)v.x;
            wb = (int)v.y;
        } else {
            const int2 v = reinterpret_cast<const int2 *>(idx_ho)[nzg];
            hi = v.x;
            wb = v.y;
        }
    }

    template <int BC_TILE, int WO_TILE, int NZ_CHUNK, int N_PAD, int IDX_MODE, typename IDX_T, typename T>
    __global__ __launch_bounds__(128) void disco_fwd_dense_kpacked_wgmma_exp_blk_k(
        int Hi, int Wi, int K, int Ho, int Wo, int NBR_PAD, int pscale, int BC_total,
        const IDX_T *__restrict__ pack_idx,     // [Ho, NBR_PAD, 2]
        const T *__restrict__ pack_val,         // [Ho, NBR_PAD, N_PAD]  (=K_PAD)
        const int64_t *__restrict__ pack_count, // [Ho]
        const T *__restrict__ inp,              // [B, C, Hi, Wi]
        T *__restrict__ out)                    // [B, C, K, Ho, Wo]
    {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
        static_assert(N_PAD == 8 || N_PAD == 16, "WGMMA path: only K_PAD in {8, 16} supported");
        static_assert(BC_TILE == 8 && WO_TILE == 8, "WGMMA path: tile must be 8x8 for M=64");
        static_assert(NZ_CHUNK == 16, "the SHFL variant assumes nz_local == (lane & 15)");

        constexpr int N_ACC = N_PAD / 2;

        const int tid = threadIdx.x;
        const int warp_id = tid / 32;
        const int lane = tid - warp_id * 32;

        const int wo_per_ho = Wo / WO_TILE;
        const int ho = blockIdx.x / wo_per_ho;
        const int wo_strip = blockIdx.x - ho * wo_per_ho;
        const int wo_base = wo_strip * WO_TILE;
        const int bc_start = blockIdx.y * BC_TILE;

        const IDX_T *idx_ho = pack_idx + (int64_t)ho * NBR_PAD * 2;
        const T *val_ho = pack_val + (int64_t)ho * NBR_PAD * N_PAD;
        const int cnt = (int)pack_count[ho];

        float acc[N_ACC];
#pragma unroll
        for (int i = 0; i < N_ACC; i++) acc[i] = 0.0f;

        // Shared memory: A_tile (2048 B) + B_tile (32 * N_PAD B) + sh_idx (8 * NZ_CHUNK B).
        // sh_idx is only touched by IDX_SMEM but is always allocated, so the shmem
        // footprint (and therefore occupancy) is identical across variants and the
        // comparison isn't confounded by an occupancy change.
        extern __shared__ __align__(16) unsigned char shmem_raw[];
        T *A_tile = reinterpret_cast<T *>(shmem_raw);
        T *B_tile = A_tile + (BC_TILE * 8) * NZ_CHUNK;
        int2 *sh_idx = reinterpret_cast<int2 *>(B_tile + NZ_CHUNK * N_PAD);

        // ----------------------- nz_chunk loop -----------------------
        for (int nz_chunk_off = 0; nz_chunk_off < cnt; nz_chunk_off += NZ_CHUNK) {

            const int bc_local = tid / NZ_CHUNK;            // [0, 8)
            const int nz_local = tid - bc_local * NZ_CHUNK; // [0, 16)
            const int nz_global = nz_chunk_off + nz_local;
            const int bc = bc_start + bc_local;

            // -- Stage pack_idx: this is the whole point of the file --
            int hi = 0;
            int wi_base = 0;
            if constexpr (IDX_MODE == IDX_DIRECT) {
                if (nz_global < cnt) {
                    hi = (int)idx_ho[nz_global * 2 + 0];
                    wi_base = (int)idx_ho[nz_global * 2 + 1];
                }
            } else if constexpr (IDX_MODE == IDX_VEC) {
                if (nz_global < cnt) { load_idx_pair<IDX_T>(idx_ho, nz_global, hi, wi_base); }
            } else if constexpr (IDX_MODE == IDX_SHFL) {
                // nz_local == (lane & 15), so lanes l and l+16 of a warp want the same
                // pair. Lanes 0..15 fetch the warp's 16 pairs; everyone else shuffles.
                int hi_r = 0, wb_r = 0;
                if (lane < NZ_CHUNK) {
                    const int nzg = nz_chunk_off + lane;
                    if (nzg < cnt) { load_idx_pair<IDX_T>(idx_ho, nzg, hi_r, wb_r); }
                }
                hi = __shfl_sync(0xffffffffu, hi_r, lane & (NZ_CHUNK - 1));
                wi_base = __shfl_sync(0xffffffffu, wb_r, lane & (NZ_CHUNK - 1));
            } else { // IDX_SMEM
                if (tid < NZ_CHUNK) {
                    const int nzg = nz_chunk_off + tid;
                    int hi_r = 0, wb_r = 0;
                    if (nzg < cnt) { load_idx_pair<IDX_T>(idx_ho, nzg, hi_r, wb_r); }
                    sh_idx[tid] = make_int2(hi_r, wb_r);
                }
                // The extra barrier this variant pays for. It cannot be folded into
                // the existing post-staging barrier: A staging reads what this writes.
                __syncthreads();
                const int2 v = sh_idx[nz_local];
                hi = v.x;
                wi_base = v.y;
            }

            // -- Stage A_tile (128 cells = 1 per thread) -- unchanged from production
            T *dst = A_tile + bc_local * (8 * NZ_CHUNK) + nz_local * 8;
            if (nz_global < cnt && bc < BC_total) {
                const int64_t inp_row_base = (int64_t)bc * Hi * Wi + (int64_t)hi * Wi;
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    int wi_full = wi_base + (wo_base + i) * pscale;
                    if (wi_full >= Wi) wi_full -= Wi;
                    dst[i] = inp[inp_row_base + wi_full];
                }
            } else {
                *reinterpret_cast<int4 *>(dst) = make_int4(0, 0, 0, 0);
            }

            // -- Stage B_tile -- unchanged from production
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
            __syncthreads();

            // -- WGMMA m64nNk16, accumulating -- unchanged from production
            wgmma_fence();
            constexpr uint32_t A_LEADING_FIELD = 8;
            constexpr uint32_t A_STRIDE_FIELD = 16;
            constexpr uint32_t B_LEADING_FIELD = 8;
            constexpr uint32_t B_STRIDE_FIELD = 16;
            uint64_t desc_a = make_wgmma_desc(A_tile, A_LEADING_FIELD * 16, A_STRIDE_FIELD * 16);
            uint64_t desc_b = make_wgmma_desc(B_tile, B_LEADING_FIELD * 16, B_STRIDE_FIELD * 16);
            const int32_t scale_D = (nz_chunk_off == 0) ? 0 : 1;
            if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                if constexpr (N_PAD == 8) {
                    wgmma_m64n8k16_acc_bf16(acc, desc_a, desc_b, scale_D);
                } else {
                    wgmma_m64n16k16_acc_bf16(acc, desc_a, desc_b, scale_D);
                }
            } else {
                if constexpr (N_PAD == 8) {
                    wgmma_m64n8k16_acc_fp16(acc, desc_a, desc_b, scale_D);
                } else {
                    wgmma_m64n16k16_acc_fp16(acc, desc_a, desc_b, scale_D);
                }
            }
            wgmma_commit_group();
            wgmma_wait_group<0>();
            wgmma_fence();

            __syncthreads();
        }

        // ----------------------- writeback ----------------------- unchanged
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

    // -------------------------------------------------------------------------
    // Host launcher. Deliberately minimal: no pt2_compliant_tag, no fake kernel,
    // no autograd, no autocast. This op exists to be profiled, not to be used
    // from a module.
    // -------------------------------------------------------------------------
    torch::Tensor disco_cuda_fwd_kpacked_exp(torch::Tensor inp,
                                             torch::Tensor pack_idx,   // [Ho, NBR_PAD, 2] int64 or int32
                                             torch::Tensor pack_val,   // [Ho, NBR_PAD, K_PAD] fp16/bf16
                                             torch::Tensor pack_count, // [Ho] int64
                                             int64_t K, int64_t Ho, int64_t Wo, int64_t variant)
    {
        CHECK_CUDA_INPUT_TENSOR(inp);
        CHECK_CUDA_INPUT_TENSOR(pack_idx);
        CHECK_CUDA_INPUT_TENSOR(pack_val);
        CHECK_CUDA_INPUT_TENSOR(pack_count);

        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, inp.get_device());
        TORCH_CHECK(props.major == 9, "forward_kpacked_exp is SM_90a (Hopper) only; got SM_", props.major, ".",
                    props.minor);

        const auto inp_dtype = inp.scalar_type();
        TORCH_CHECK(inp_dtype == at::ScalarType::BFloat16 || inp_dtype == at::ScalarType::Half,
                    "forward_kpacked_exp requires bf16 or fp16 input");
        TORCH_CHECK(pack_val.scalar_type() == inp_dtype,
                    "pack_val must already be in the input dtype (cast it once on the host; casting here would put a "
                    "conversion kernel in every profiled iteration)");
        const auto idx_dtype = pack_idx.scalar_type();
        TORCH_CHECK(idx_dtype == at::ScalarType::Long || idx_dtype == at::ScalarType::Int,
                    "pack_idx must be int64 or int32");
        TORCH_CHECK(variant >= 0 && variant <= 3, "variant must be 0 (direct), 1 (vec), 2 (shfl) or 3 (smem); got ",
                    variant);

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

        // A_tile + B_tile + sh_idx. sh_idx is allocated for every variant so the
        // occupancy is variant-independent.
        const size_t shmem_bytes = 2048 + 32 * (size_t)K_PAD + 8 * (size_t)NZ_CHUNK;
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        const int pscale = (int)(Wi / Wo);
        const int BC_total = (int)(B * C);
        const int bc_blocks = (BC_total + BC_TILE - 1) / BC_TILE;
        const dim3 grid((unsigned)(Ho * (Wo / WO_TILE)), (unsigned)bc_blocks);

        // Explicit instantiation over (T, IDX_T, N_PAD, IDX_MODE). Written out
        // rather than generated so the build stays readable; 2*2*2*4 = 32 kernels.
#define TH_LAUNCH_EXP(T_, IDX_T_, NPAD_, MODE_)                                                                        \
    do {                                                                                                               \
        auto fn = &disco_fwd_dense_kpacked_wgmma_exp_blk_k<BC_TILE, WO_TILE, NZ_CHUNK, NPAD_, MODE_, IDX_T_, T_>;      \
        cudaFuncSetAttribute(reinterpret_cast<const void *>(fn), cudaFuncAttributeMaxDynamicSharedMemorySize,          \
                             (int)shmem_bytes);                                                                        \
        fn<<<grid, 128, shmem_bytes, stream>>>(                                                                        \
            (int)Hi, (int)Wi, (int)K, (int)Ho, (int)Wo, NBR_PAD, pscale, BC_total,                                     \
            reinterpret_cast<const IDX_T_ *>(pack_idx.data_ptr()), reinterpret_cast<const T_ *>(pack_val.data_ptr()),  \
            pack_count.data_ptr<int64_t>(), reinterpret_cast<const T_ *>(inp.data_ptr()),                              \
            reinterpret_cast<T_ *>(out.data_ptr()));                                                                   \
    } while (0)

#define TH_DISPATCH_MODE(T_, IDX_T_, NPAD_)                                                                            \
    do {                                                                                                               \
        switch ((int)variant) {                                                                                        \
        case IDX_DIRECT: TH_LAUNCH_EXP(T_, IDX_T_, NPAD_, IDX_DIRECT); break;                                          \
        case IDX_VEC: TH_LAUNCH_EXP(T_, IDX_T_, NPAD_, IDX_VEC); break;                                                \
        case IDX_SHFL: TH_LAUNCH_EXP(T_, IDX_T_, NPAD_, IDX_SHFL); break;                                              \
        default: TH_LAUNCH_EXP(T_, IDX_T_, NPAD_, IDX_SMEM); break;                                                    \
        }                                                                                                              \
    } while (0)

#define TH_DISPATCH_NPAD(T_, IDX_T_)                                                                                   \
    do {                                                                                                               \
        if (K_PAD == 8) {                                                                                              \
            TH_DISPATCH_MODE(T_, IDX_T_, 8);                                                                           \
        } else {                                                                                                       \
            TH_DISPATCH_MODE(T_, IDX_T_, 16);                                                                          \
        }                                                                                                              \
    } while (0)

#define TH_DISPATCH_IDX(T_)                                                                                            \
    do {                                                                                                               \
        if (idx_dtype == at::ScalarType::Long) {                                                                       \
            TH_DISPATCH_NPAD(T_, int64_t);                                                                             \
        } else {                                                                                                       \
            TH_DISPATCH_NPAD(T_, int32_t);                                                                             \
        }                                                                                                              \
    } while (0)

        if (inp_dtype == at::ScalarType::BFloat16) {
            TH_DISPATCH_IDX(__nv_bfloat16);
        } else {
            TH_DISPATCH_IDX(__half);
        }

#undef TH_DISPATCH_IDX
#undef TH_DISPATCH_NPAD
#undef TH_DISPATCH_MODE
#undef TH_LAUNCH_EXP

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return out;
    }

    // Bare registration: adds one op to the existing disco_kernels library. No
    // pt2_compliant_tag — this op is not meant to survive torch.compile or
    // autograd, only to be called directly.
    TORCH_LIBRARY_FRAGMENT(disco_kernels, m)
    {
        m.def("forward_kpacked_exp(Tensor inp, Tensor pack_idx, Tensor pack_val, Tensor pack_count, "
              "int kernel_size, int nlat_out, int nlon_out, int variant) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(disco_kernels, CUDA, m) { m.impl("forward_kpacked_exp", &disco_cuda_fwd_kpacked_exp); }

} // namespace disco_kernels
