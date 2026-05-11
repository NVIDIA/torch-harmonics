// coding=utf-8
//
// SPDX-FileCopyrightText: Copyright (c) 2024 The torch-harmonics Authors. All rights reserved.
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
// DISCO backward (gather) — CUDA implementation
// =====================================================================================
//
// Each output cell out[b, c, ho, wo] is computed by exactly one thread:
// no atomics, no race. The kernel consumes psi_T (CSR over the bigger grid),
// built by `_transpose_convolution_tensor_s2` in convolution.py:
//
//   row_T   = ho * pscale + (wo % pscale)             // row index, length Ho*pscale
//   col_idx = hi * Wo + wi_offset, wi_offset in [0,Wo)
//   ker_idx is per-entry (rows mix kernel indices since the backward
//           contracts k_kern in addition to the psi-neighbor axis).
//
// Bucketing by `wi_offset % pscale` guarantees every entry in the selected
// row contributes — no per-entry validity check beyond the row lookup.
//
// Op shape (matches `disco_kernels::backward`):
//   inp [B, C, K, Hi, Wi]   — smaller grid (function-local "input")
//   out [B, C, Ho, Wo]      — bigger grid  (function-local "output")
//   pscale = Wo / Wi
// =====================================================================================

#include "../disco.h"
#include "disco_cuda.cuh"

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <type_traits>

namespace disco_kernels {

// =====================================================================================
// Gather backward — shmem-accumulator design
// =====================================================================================
//
// Parallelism: one CTA per (bc, ho) cell.
// Inner loop:  outer over psi_T entries, inner over wo (vectorized via threads).
// Shmem:       per-CTA `acc[Wo]` accumulator, no shmem atomics.
//
// Why: the naive "one thread per output cell" version reduces over psi_T's row
// (~nnz_T / nrows_T entries) per cell, and each entry triggers a non-contiguous
// inp load. That layout is bandwidth-bound *and* cache-thrashing for low-pscale
// workloads with very long psi_T rows (e.g. pscale=1, ~14k entries / row).
//
// Restructuring so each CTA owns a full output row out[bc, ho, :] turns the
// hot inner loop into a coalesced read of inp[bc, ker, hi, :] per entry plus a
// linear shmem accumulate. This mirrors the access pattern the old scatter
// kernel used to win, but with no global atomics — each CTA writes distinct
// out[bc, ho, *] cells exactly once at the end.
//
// Per-entry write safety: for a single entry, `wo = (wi_in - col_offset) % Wo`
// (or for fixed wi_in, wo = (wi_in - wi_offset)/pscale within bucket-r) is
// unique per thread. Across entries, different warps may run ahead, so a
// `__syncthreads()` is required between entries to avoid racing on acc[].
// =====================================================================================
//
// IMPORTANT: this kernel reuses the same op contract as the naive version
// (input grad_out [B,C,K,Hi,Wi]; output grad_inp [B,C,Ho,Wo]; pscale = Wo/Wi).
// In the kernel-local naming, "Hi"/"Wi" is the smaller grid (input), "Ho"/"Wo"
// is the bigger grid (output). The shmem accumulator is sized Wo (= bigger).
// =====================================================================================

// Symmetric to the forward kernel:
//   * regs hold per-thread accumulators for OUTPUT cells (one per ELXTH stripe),
//   * shmem caches the INPUT row inp[bc, ker_cur, hi_cur, :], sized 2*Wi so the
//     no-mod address `wi_idx + pscale*pp` (or with one wrap) lands inside.
//   * __syncthreads() only on (ker, hi) change — not per entry.
//
// PSCALE templated: nonzero values fold the multiply, runtime fallback at 0.
template <int BLOCK_X, int ELXTH, int PSCALE, typename STORAGE_T, typename COMPUTE_T>
__launch_bounds__(BLOCK_X)
__global__ void disco_bwd_gather_shmem_blk(
    const int Hi, const int Wi, const int K, const int Ho, const int Wo, const int pscale_runtime,
    const int64_t *__restrict__ roff_T,
    const int64_t *__restrict__ ker_T,
    const int64_t *__restrict__ col_T,
    const COMPUTE_T *__restrict__ vals_T,
    const STORAGE_T *__restrict__ inp,   // grad_out  [BC, K, Hi, Wi]
    COMPUTE_T *__restrict__ out)         // grad_inp  [BC, Ho, Wo]
{
    const int pscale = (PSCALE != 0) ? PSCALE : pscale_runtime;

    const int bc  = blockIdx.y;
    const int ho  = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ __align__(sizeof(double)) unsigned char __sh_raw[];
    STORAGE_T *sh = reinterpret_cast<STORAGE_T *>(__sh_raw);   // size = 2*Wi (storage dtype)

    // Per-thread output-cell accumulators. Thread `tid` owns output cells
    // pp = tid, tid + BLOCK_X, ..., tid + (ELXTH-1)*BLOCK_X (those that fall in [0, Wo)).
    COMPUTE_T regs[ELXTH];
    #pragma unroll
    for (int i = 0; i < ELXTH; i++) regs[i] = COMPUTE_T(0);

    int prev_ker = -1;
    int prev_hi  = -1;

    for (int r = 0; r < pscale; r++) {
        const int row_T = ho * pscale + r;
        const int64_t s = roff_T[row_T];
        const int64_t e = roff_T[row_T + 1];

        for (int64_t z = s; z < e; z++) {
            const int       ker       = (int)ker_T[z];
            const int64_t   col       = col_T[z];
            const int       hi        = (int)(col / Wo);
            const int       wi_offset = (int)(col % Wo);
            const COMPUTE_T val       = vals_T[z];

            // Refill shmem with inp[bc, ker, hi, :] only on (ker, hi) change.
            // Replicated 2*Wi-deep to avoid an inner-loop modulo.
            if (ker != prev_ker || hi != prev_hi) {
                __syncthreads();   // ensure prior reads of sh[] are done
                const int64_t inp_base = (((int64_t)bc * K + ker) * Hi + hi) * Wi;
                for (int i = tid; i < Wi; i += BLOCK_X) {
                    const STORAGE_T v = inp[inp_base + i];
                    sh[i]      = v;
                    sh[Wi + i] = v;
                }
                __syncthreads();
                prev_ker = ker;
                prev_hi  = hi;
            }

            // Inner loop: each thread updates its ELXTH output-cell regs.
            // For output cell pp the matching shmem index is:
            //   wi_idx = (pp - wi_offset) / pscale,  if (pp - wi_offset) >= 0 (else +Wo first)
            // The bucketing invariant (wi_offset % pscale == r and we only land
            // here when pp % pscale == r) guarantees the division is exact.
            //
            // For PSCALE=1 the residue check is vacuous and the divide is free —
            // the inner loop reduces to a single shmem read + reg fma.
            #pragma unroll
            for (int i = 0; i < ELXTH; i++) {
                const int pp = i * BLOCK_X + tid;
                if (pp >= Wo) break;
                if constexpr (PSCALE > 1) {
                    if ((pp % PSCALE) != r) continue;
                } else if (PSCALE == 0) {
                    if ((pp % pscale) != r) continue;
                }
                int delta = pp - wi_offset;
                if (delta < 0) delta += Wo;          // single wrap (wi_offset, pp both in [0, Wo))
                const int wi_idx = (PSCALE == 1) ? delta : (delta / pscale);
                regs[i] += val * static_cast<COMPUTE_T>(sh[wi_idx]);
            }
            // No __syncthreads here — regs are per-thread; sh[] is read-only this iter.
        }
    }

    // Flush regs to global output. One coalesced write stripe per thread.
    const int64_t out_base = ((int64_t)bc * Ho + ho) * Wo;
    #pragma unroll
    for (int i = 0; i < ELXTH; i++) {
        const int pp = i * BLOCK_X + tid;
        if (pp < Wo) out[out_base + pp] = regs[i];
    }
}

torch::Tensor disco_cuda_bwd(torch::Tensor inp, torch::Tensor roff_idx, torch::Tensor ker_idx,
                             torch::Tensor col_idx, torch::Tensor val, int64_t K, int64_t Ho, int64_t Wo)
{
    CHECK_CUDA_INPUT_TENSOR(inp);
    CHECK_CUDA_INPUT_TENSOR(roff_idx);
    CHECK_CUDA_INPUT_TENSOR(ker_idx);
    CHECK_CUDA_INPUT_TENSOR(col_idx);
    CHECK_CUDA_INPUT_TENSOR(val);

    const int64_t B  = inp.size(0);
    const int64_t C  = inp.size(1);
    const int64_t BC = B * C;
    const int64_t Hi = inp.size(3);
    const int64_t Wi = inp.size(4);

    TORCH_CHECK(inp.size(2) == K, "inp.size(2) (", inp.size(2), ") != K (", K, ")");
    TORCH_CHECK(Wo % Wi == 0, "Wo (", Wo, ") must be an integer multiple of Wi (", Wi, ")");
    const int64_t pscale = Wo / Wi;
    TORCH_CHECK(roff_idx.size(0) - 1 == Ho * pscale,
                "psi_T roff_idx length (", roff_idx.size(0), ") inconsistent with Ho*pscale+1 (",
                Ho * pscale + 1, ")");

    auto options = torch::TensorOptions().device(inp.device()).dtype(val.dtype());
    torch::Tensor out = torch::zeros({B, C, Ho, Wo}, options);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Block / grid: one CTA per (bc, ho) output row; threads stride through
    // wi (smaller grid) inside the entry loop, writing into a per-CTA shmem
    // accumulator of size Wo (bigger grid).
    //
    // ELXTH = ceil(Wi / BDIM_X) — number of wi values each thread caches in
    // registers per (ker, hi) tuple, set at compile time so registers
    // actually allocate (templated dispatch below).
    constexpr int BDIM_X = 128;
    dim3 block(BDIM_X);
    dim3 grid((unsigned)Ho, (unsigned)BC);

    // Pick ELXTH for the largest Wi we expect to encounter. Cover up to
    // BDIM_X * ELXTH_MAX = 128 * 32 = 4096; pick the smallest ELXTH that
    // fits the actual Wi to keep register pressure / spills low.
    constexpr int ELXTH_CAP = ELXTH_MAX;  // 32

    AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_backward_cuda", ([&] {
        using storage_t = scalar_t;
        using compute_t = typename at::opmath_type<storage_t>;
        // Shmem holds inp[bc, ker, hi, :] replicated to 2*Wi (storage dtype) so
        // the inner loop can index without a modulo.
        const size_t shmem_bytes = sizeof(storage_t) * 2 * Wi;

        auto launch = [&](auto elxth, auto pscale_tag) {
            constexpr int ELXTH  = decltype(elxth)::value;
            constexpr int PSCALE = decltype(pscale_tag)::value;
            disco_bwd_gather_shmem_blk<BDIM_X, ELXTH, PSCALE, storage_t, compute_t>
                <<<grid, block, shmem_bytes, stream>>>(
                (int)Hi, (int)Wi, (int)K, (int)Ho, (int)Wo, (int)pscale,
                roff_idx.data_ptr<int64_t>(),
                ker_idx.data_ptr<int64_t>(),
                col_idx.data_ptr<int64_t>(),
                val.data_ptr<compute_t>(),
                inp.data_ptr<storage_t>(),
                out.data_ptr<compute_t>());
        };

        // Pick a compile-time PSCALE specialization for the common values
        // (1, 2, 3, 4); everything else uses PSCALE=0 (runtime path).
        auto launch_with_pscale = [&](auto elxth) {
            switch ((int)pscale) {
                case 1: launch(elxth, std::integral_constant<int, 1>{}); break;
                case 2: launch(elxth, std::integral_constant<int, 2>{}); break;
                case 3: launch(elxth, std::integral_constant<int, 3>{}); break;
                case 4: launch(elxth, std::integral_constant<int, 4>{}); break;
                default: launch(elxth, std::integral_constant<int, 0>{}); break;
            }
        };

        // ELXTH must cover the output cells (Wo, the bigger grid). Since
        // Wo = pscale * Wi, this also covers the shmem read range.
        const int elxth_needed = (int)((Wo + BDIM_X - 1) / BDIM_X);
        TORCH_CHECK(elxth_needed <= ELXTH_CAP,
                    "Wo (", Wo, ") exceeds supported maximum (BDIM_X*ELXTH_CAP = ",
                    BDIM_X * ELXTH_CAP, ")");

        // Compile-time switch over ELXTH (1..ELXTH_CAP). Branches with
        // ELXTH > elxth_needed are also valid but waste registers, so we
        // pick the smallest value that fits.
        switch (elxth_needed) {
            case  1: launch_with_pscale(std::integral_constant<int,  1>{}); break;
            case  2: launch_with_pscale(std::integral_constant<int,  2>{}); break;
            case  3: launch_with_pscale(std::integral_constant<int,  3>{}); break;
            case  4: launch_with_pscale(std::integral_constant<int,  4>{}); break;
            case  5: launch_with_pscale(std::integral_constant<int,  5>{}); break;
            case  6: launch_with_pscale(std::integral_constant<int,  6>{}); break;
            case  7: launch_with_pscale(std::integral_constant<int,  7>{}); break;
            case  8: launch_with_pscale(std::integral_constant<int,  8>{}); break;
            case  9: launch_with_pscale(std::integral_constant<int,  9>{}); break;
            case 10: launch_with_pscale(std::integral_constant<int, 10>{}); break;
            case 11: launch_with_pscale(std::integral_constant<int, 11>{}); break;
            case 12: launch_with_pscale(std::integral_constant<int, 12>{}); break;
            case 13: launch_with_pscale(std::integral_constant<int, 13>{}); break;
            case 14: launch_with_pscale(std::integral_constant<int, 14>{}); break;
            case 15: launch_with_pscale(std::integral_constant<int, 15>{}); break;
            case 16: launch_with_pscale(std::integral_constant<int, 16>{}); break;
            default: launch_with_pscale(std::integral_constant<int, ELXTH_CAP>{}); break;
        }
    }));

    out = out.to(inp.dtype());
    return out;
}

TORCH_LIBRARY_IMPL(disco_kernels, CUDA, m)
{
    m.impl("backward", &disco_cuda_bwd);
}

}  // namespace disco_kernels
