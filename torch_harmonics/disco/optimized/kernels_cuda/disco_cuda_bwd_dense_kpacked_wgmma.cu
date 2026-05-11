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
// Disco backward — K-packed dense psi (CUDA, WGMMA path, Hopper SM_90+)
// =====================================================================================
//
// Computes:
//   grad_inp[bc, hi, wi] +=
//     sum_{ho, wo, k_kern, nz : hi(ho,nz)==hi, (wi_base(ho,nz) + wo*pscale)%Wi == wi}
//       grad_out[bc, k_kern, ho, wo] * pack_val[ho, nz, k_kern]
//
// Approach (Option C — atomic-add scatter):
//   - Parallelise over (ho, wo_strip, bc-tile) like the forward kernel.
//   - WGMMA m64n16k16 multiplies grad_out by pack_val per nz_chunk:
//       D[m=(bc, wo_local), n=nz_local] = sum_{k_kern} A[m, k] * B[k, n]
//     where K=k_kern (padded to 16 when K_PAD=8) and N=nz_local (=NZ_CHUNK=16).
//   - Each thread atomicAdds its acc fragment cells into a fp32 grad_inp_scratch
//     buffer (target indexed by (bc, hi(nz_global), wi_full)). The host wrapper
//     casts the scratch back to inp.dtype after the kernel completes.
//
// Restricted to (host-enforced):
//   - K_PAD ∈ {8, 16}
//   - bf16 OR fp16 inputs (templated on T)
//   - pscale ≥ 1 integer; Wi % Wo == 0
//   - Wo % 8 == 0
//   - B*C boundary handled by predicated A-staging + atomicAdd skip
//
// CTA layout: 128 threads (1 warp-group). Output tile M=64 (=BC_TILE×WO_TILE),
// inner loop over nz_chunks of 16 nz each. WGMMA shape m64n16k16 always.
//
// Shared memory layout (T = bf16 or fp16, both 2-byte; 16-byte aligned)
// ---------------------------------------------------------------------
// A_tile [M=64, K=16] (= grad_out tile, M-major in core, identical layout to
// the forward kernel's A_tile):
//    byte(m, k) = (m / 8) * 256 + (m % 8 + 8 * k) * 2
//    total 2048 bytes
// Staged ONCE per CTA — A is constant across nz_chunks.
//
// B_tile [K=16, N=16] (= pack_val tile, N-group outer with K-rows contiguous,
// identical layout to the forward kernel's B_tile for N=16):
//    byte(k, n) = (n / 8) * 256 + k * 16 + (n % 8) * 2
//    total 512 bytes
// Restaged per nz_chunk. NOTE: in this layout the K-axis is k_kern and the
// N-axis is nz_local; the source pack_val is k_kern-fast in gmem, so the
// staging gathers strided.
//
// Per nz_chunk inner loop
// -----------------------
// 1. Stage B_tile (32 active threads, each writes 8 narrow elements at fixed
//    (k_local, n_group), gathering 8 strided nz from pack_val).
// 2. __syncthreads.
// 3. zero acc[]; wgmma.fence; wgmma.mma_async m64n16k16; wgmma.commit_group;
//    wgmma.wait_group<0>.
// 4. Per-thread: for each frag cell, atomicAdd into grad_inp_scratch[bc, hi(nz),
//    wi_full] (fp32). Threads with bc ≥ BC_total or nz_global ≥ cnt skip.
// =====================================================================================

#include "../disco.h"
#include "disco_cuda.cuh"
#include "disco_cuda_ptx.cuh"

#include <ATen/Dispatch.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace disco_kernels {

// T ∈ {__nv_bfloat16, __half}. Forward declaration in host wrapper picks the
// matching .f32.bf16.bf16 vs .f32.f16.f16 wgmma opcode.
template <int BC_TILE, int WO_TILE, int NZ_CHUNK, int K_PAD, typename T>
__global__ __launch_bounds__(128)
void disco_bwd_dense_kpacked_wgmma_blk_k(
    int Hi, int Wi, int K, int Ho, int Wo, int NBR_PAD, int pscale, int BC_total,
    const int64_t *__restrict__ pack_idx,    // [Ho, NBR_PAD, 2]
    const T       *__restrict__ pack_val,    // [Ho, NBR_PAD, K_PAD]
    const int64_t *__restrict__ pack_count,  // [Ho]
    const T       *__restrict__ grad_out,    // [B, C, K, Ho, Wo]
    float         *__restrict__ grad_inp)    // [B, C, Hi, Wi]  (fp32 scratch)
{
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    static_assert(K_PAD == 8 || K_PAD == 16,
                  "Backward WGMMA path: only K_PAD ∈ {8, 16} supported");
    static_assert(BC_TILE == 8 && WO_TILE == 8, "Backward WGMMA: M=64 tile only");
    static_assert(NZ_CHUNK == 16, "Backward WGMMA: NZ_CHUNK == 16 (= N of m64n16k16)");

    constexpr int N_ACC = 8;   // m64n16k16: 8 fp32 cells per thread

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane    = tid - warp_id * 32;

    const int wo_per_ho = Wo / WO_TILE;
    const int ho        = blockIdx.x / wo_per_ho;
    const int wo_strip  = blockIdx.x - ho * wo_per_ho;
    const int wo_base   = wo_strip * WO_TILE;
    const int bc_start  = blockIdx.y * BC_TILE;

    const int64_t *idx_ho = pack_idx + (int64_t)ho * NBR_PAD * 2;
    const T       *val_ho = pack_val + (int64_t)ho * NBR_PAD * K_PAD;
    const int      cnt    = (int)pack_count[ho];

    if (cnt == 0) return;   // nothing to scatter for this ho

    // Shared memory: A_tile (2048 B) + B_tile (512 B) per CTA.
    extern __shared__ __align__(16) unsigned char shmem_raw[];
    T *A_tile = reinterpret_cast<T*>(shmem_raw);
    T *B_tile = A_tile + (BC_TILE * 8) * NZ_CHUNK;   // 1024 elements after A_tile

    // ----------------------- Stage A_tile (grad_out, once) -----------------------
    // Each thread populates one cell of the [M=64, K=16] tile = 8 narrow values.
    // Mapping: tid = bc_local * 16 + k_local; thread reads 8 wo positions for
    // its (bc_local, k_local) at fixed (ho).
    {
        const int bc_local = tid / 16;
        const int k_local  = tid - bc_local * 16;
        const int bc       = bc_start + bc_local;
        T *a_dst = A_tile + bc_local * (8 * NZ_CHUNK) + k_local * 8;

        if (bc < BC_total && k_local < K_PAD) {
            const int64_t go_row_base =
                ((int64_t)bc * K + (int64_t)k_local) * Ho * Wo + (int64_t)ho * Wo;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                a_dst[i] = grad_out[go_row_base + (int64_t)(wo_base + i)];
            }
        } else {
            // K-pad zero (k_local >= K_PAD) or BC out-of-range.
            *reinterpret_cast<int4*>(a_dst) = make_int4(0, 0, 0, 0);
        }
    }

    // Per-thread WGMMA accumulator. Zeroed before each chunk's WGMMA — D is
    // per-chunk (NOT accumulated across chunks).
    float acc[N_ACC];

    // WGMMA descriptor field constants (fixed across chunks).
    constexpr uint32_t A_LEADING_FIELD = 8;    // K-outer 128 B
    constexpr uint32_t A_STRIDE_FIELD  = 16;   // M-outer 256 B
    constexpr uint32_t B_LEADING_FIELD = 8;    // K-outer 128 B (within n-group)
    constexpr uint32_t B_STRIDE_FIELD  = 16;   // N-outer 256 B (between n-groups)

    // ----------------------- nz_chunk loop -----------------------
    for (int nz_chunk_off = 0; nz_chunk_off < cnt; nz_chunk_off += NZ_CHUNK) {

        // -- Stage B_tile (pack_val, restaged per chunk) --
        // Layout target: byte(k=k_kern, n=nz_local) = (n/8)*256 + k*16 + (n%8)*2.
        // 32 active threads (16 K-rows × 2 N-groups). Each writes 8 contiguous
        // narrow elements (16 bytes) at a fixed (k_local, n_group), reading 8
        // strided nz values from pack_val (k_kern-fast in gmem → stride K_PAD).
        constexpr int B_TOTAL_CHUNKS = 32;   // 16 × 2 (NZ_CHUNK / 8 N-inner × CHUNKS_PER_ROW)
        if (tid < B_TOTAL_CHUNKS) {
            const int n_group   = tid / 16;        // ∈ [0, 2)
            const int k_local_b = tid - n_group * 16;
            T *dst_b = B_tile + n_group * (NZ_CHUNK * 8) + k_local_b * 8;

            if (k_local_b < K_PAD) {
                // Typed zero — narrow types (__half / __nv_bfloat16) have no
                // int constructor. Build via __float2*, branched with
                // `if constexpr` so each instantiation only sees its own type
                // (avoids default-constructing T and stays type-clean).
                T zero_v;
                if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                    zero_v = __float2bfloat16(0.f);
                } else {
                    zero_v = __float2half(0.f);
                }

                const int nz_global_base = nz_chunk_off + n_group * 8;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    const int nz_global = nz_global_base + i;
                    dst_b[i] = (nz_global < cnt)
                        ? val_ho[(int64_t)nz_global * K_PAD + (int64_t)k_local_b]
                        : zero_v;
                }
            } else {
                // K-pad zero (k_local_b >= K_PAD).
                *reinterpret_cast<int4*>(dst_b) = make_int4(0, 0, 0, 0);
            }
        }

        __syncthreads();

        // -- WGMMA m64n16k16, NON-accumulating (fresh per chunk) --
        #pragma unroll
        for (int i = 0; i < N_ACC; i++) acc[i] = 0.0f;

        wgmma_fence();
        uint64_t desc_a = make_wgmma_desc(A_tile,
                                          A_LEADING_FIELD * 16,
                                          A_STRIDE_FIELD  * 16);
        uint64_t desc_b = make_wgmma_desc(B_tile,
                                          B_LEADING_FIELD * 16,
                                          B_STRIDE_FIELD  * 16);
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            wgmma_m64n16k16_acc_bf16(acc, desc_a, desc_b);
        } else { // T == __half
            wgmma_m64n16k16_acc_fp16(acc, desc_a, desc_b);
        }
        wgmma_commit_group();
        wgmma_wait_group<0>();

        // -- Per-thread atomicAdd of 8 frag cells to grad_inp_scratch --
        // m64n16k16.f32 fragment-to-(m, n) mapping (PTX ISA §9.7.16.5.4):
        //   ng ∈ [0, 2):
        //     cell 4*ng+0: m = w*16 + l/4,     n = (l%4)*2     + 8*ng
        //     cell 4*ng+1: m = w*16 + l/4,     n = (l%4)*2 + 1 + 8*ng
        //     cell 4*ng+2: m = w*16 + l/4 + 8, n = (l%4)*2     + 8*ng
        //     cell 4*ng+3: m = w*16 + l/4 + 8, n = (l%4)*2 + 1 + 8*ng
        const int m01 = warp_id * 16 + (lane >> 2);
        const int m23 = m01 + 8;
        const int n_a = (lane & 3) * 2;
        const int n_b = n_a + 1;

        const int bc01     = bc_start + (m01 >> 3);
        const int wo01_loc = m01 & 7;            // wo_local for cells 0,1,4,5
        const int bc23     = bc_start + (m23 >> 3);
        const int wo23_loc = m23 & 7;            // wo_local for cells 2,3,6,7

        // Resolve target (bc, hi, wi_full) per (n) and atomicAdd if in bounds.
        // wi_full wraps at most once: with wi_base ≤ Wi-1 and wo_full ≤ Wo-1,
        // wi_full ≤ (Wi-1) + (Wo-1)*pscale ≤ 2*Wi - 1 - pscale. Single subtract
        // suffices when Wi == pscale*Wo (host-enforced).
        auto scatter = [&] (int bc_o, int wo_local, int n, float v) {
            if (bc_o >= BC_total) return;
            const int nz_global = nz_chunk_off + n;
            if (nz_global >= cnt) return;        // nz padding past end
            const int hi      = (int)idx_ho[nz_global * 2 + 0];
            const int wi_base = (int)idx_ho[nz_global * 2 + 1];
            int wi_full = wi_base + (wo_base + wo_local) * pscale;
            if (wi_full >= Wi) wi_full -= Wi;
            atomicAdd(&grad_inp[(int64_t)bc_o * Hi * Wi
                              + (int64_t)hi * Wi
                              + (int64_t)wi_full], v);
        };

        #pragma unroll
        for (int ng = 0; ng < 2; ng++) {
            const int n0 = n_a + 8 * ng;
            const int n1 = n_b + 8 * ng;
            scatter(bc01, wo01_loc, n0, acc[ng * 4 + 0]);
            scatter(bc01, wo01_loc, n1, acc[ng * 4 + 1]);
            scatter(bc23, wo23_loc, n0, acc[ng * 4 + 2]);
            scatter(bc23, wo23_loc, n1, acc[ng * 4 + 3]);
        }

        // No __syncthreads needed: atomicAdd doesn't touch shmem; the next
        // chunk's B-stage will overwrite B_tile, but we only sync before the
        // WGMMA reads it.
    }
#else
    // Non-sm_90a target: empty body. Host runtime check ensures we never
    // dispatch to this kernel on devices without WGMMA.
    (void)Hi; (void)Wi; (void)K; (void)Ho; (void)Wo; (void)NBR_PAD; (void)pscale; (void)BC_total;
    (void)pack_idx; (void)pack_val; (void)pack_count; (void)grad_out; (void)grad_inp;
#endif
}

// Host wrapper. Returns true iff the WGMMA backward path was actually invoked.
// Same precondition gate as the forward kpacked WGMMA path; falls through to
// the caller's preferred fallback (CSR backward typically) otherwise.
bool disco_cuda_bwd_dense_kpacked_wgmma_try(
    torch::Tensor grad_out,
    torch::Tensor pack_idx,
    torch::Tensor pack_val,
    torch::Tensor pack_count,
    torch::Tensor grad_inp_scratch,    // pre-allocated fp32 [B, C, Hi, Wi]
    int64_t K, int64_t Hi, int64_t Wi)
{
    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    if (props.major != 9) return false;

    const auto inp_dtype = grad_out.scalar_type();
    if (inp_dtype != at::ScalarType::BFloat16 &&
        inp_dtype != at::ScalarType::Half) return false;

    const int64_t B  = grad_out.size(0);
    const int64_t C  = grad_out.size(1);
    const int64_t Ho = grad_out.size(3);
    const int64_t Wo = grad_out.size(4);

    if (Wi % Wo != 0)            return false;   // pscale must be integer
    if (Wo % 8 != 0)             return false;

    const int64_t K_PAD = pack_val.size(2);
    if (K_PAD != 8 && K_PAD != 16) return false;

    constexpr int BC_TILE  = 8;
    constexpr int WO_TILE  = 8;
    constexpr int NZ_CHUNK = 16;

    const int NBR_PAD = (int)pack_idx.size(1);

    // Shmem per CTA: A_tile (2048 B) + B_tile (32 * NZ_CHUNK = 512 B).
    const size_t shmem_bytes = 2048 + 32 * (size_t)NZ_CHUNK;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Cast pack_val to match grad_out's dtype (psi values default to fp32).
    auto pack_val_cast = (pack_val.scalar_type() == inp_dtype)
        ? pack_val
        : pack_val.to(inp_dtype);

    const int pscale   = (int)(Wi / Wo);
    const int BC_total = (int)(B * C);
    const int bc_blocks = (BC_total + BC_TILE - 1) / BC_TILE;
    const dim3 grid((unsigned)(Ho * (Wo / WO_TILE)), (unsigned)bc_blocks);

    auto fire = [&] (auto fn, auto T_tag) {
        using T = decltype(T_tag);
        cudaFuncSetAttribute(reinterpret_cast<const void*>(fn),
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)shmem_bytes);
        fn<<<grid, 128, shmem_bytes, stream>>>(
            (int)Hi, (int)Wi, (int)K, (int)Ho, (int)Wo, NBR_PAD, pscale, BC_total,
            pack_idx.data_ptr<int64_t>(),
            reinterpret_cast<const T*>(pack_val_cast.data_ptr()),
            pack_count.data_ptr<int64_t>(),
            reinterpret_cast<const T*>(grad_out.data_ptr()),
            grad_inp_scratch.data_ptr<float>()
        );
    };

    if (inp_dtype == at::ScalarType::BFloat16) {
        if (K_PAD == 8)
            fire(&disco_bwd_dense_kpacked_wgmma_blk_k<BC_TILE, WO_TILE, NZ_CHUNK,  8, __nv_bfloat16>,
                 __nv_bfloat16{});
        else
            fire(&disco_bwd_dense_kpacked_wgmma_blk_k<BC_TILE, WO_TILE, NZ_CHUNK, 16, __nv_bfloat16>,
                 __nv_bfloat16{});
    } else { // Half
        if (K_PAD == 8)
            fire(&disco_bwd_dense_kpacked_wgmma_blk_k<BC_TILE, WO_TILE, NZ_CHUNK,  8, __half>,
                 __half{});
        else
            fire(&disco_bwd_dense_kpacked_wgmma_blk_k<BC_TILE, WO_TILE, NZ_CHUNK, 16, __half>,
                 __half{});
    }
    return true;
}

}  // namespace disco_kernels
