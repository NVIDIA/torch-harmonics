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

// =====================================================================================
// DISCO backward — K-packed gather (CUDA, split-by-K)
// =====================================================================================
//
// Same math as `disco_cuda_bwd.cu` (`backward`), but with K-packed psi_T:
//
//   psi_T_kpacked_idx   : [Hi*pscale, NBR_PAD_T, 2]      int64  (ho, wi_offset)
//   psi_T_kpacked_vals  : [Hi*pscale, NBR_PAD_T, K_pad]  storage_t
//   psi_T_kpacked_count : [Hi*pscale]                    int64
//
// Parallelism: one CTA per (bc, ho, k) — K× more CTAs than the plain gather.
// Each CTA pulls one COLUMN of the K-packed val vector per entry and
// accumulates into per-thread output-cell registers (regs hold partial
// contributions from this k slice). At CTA end, atomicAdds into a fp32
// scratch buffer. K-way contention per output cell (8 or 16 contributors).
//
// Inner loop is the forward-style shmem-input + reg-output pattern: shmem
// caches `inp[bc, k, hi, :]` (Wi values, doubled to 2*Wi for no-mod indexing),
// regs accumulate output cells, `__syncthreads()` only on hi change.
//
// PSCALE templated for compile-time bucket-residue arithmetic.
// =====================================================================================

#include "../disco.h"
#include "disco_cuda.cuh"

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <type_traits>

namespace disco_kernels {

template <int BLOCK_X, int ELXTH, int PSCALE, typename STORAGE_T, typename COMPUTE_T>
__launch_bounds__(BLOCK_X)
__global__ void disco_bwd_gather_kpacked_blk(
    const int Hi, const int Wi, const int K, const int K_pad,
    const int Ho, const int Wo, const int pscale_runtime,
    const int NBR_PAD_T,
    const int64_t *__restrict__ idx_T,        // [Hi*pscale, NBR_PAD_T, 2]   (ho, wi_offset)
    const STORAGE_T *__restrict__ vals_T,     // [Hi*pscale, NBR_PAD_T, K_pad]
    const int64_t *__restrict__ count_T,      // [Hi*pscale]
    const STORAGE_T *__restrict__ inp,        // grad_out [B, C, K, Hi, Wi]
    COMPUTE_T *__restrict__ out_scratch)      // grad_inp scratch (fp32) [B, C, Ho, Wo]
{
    const int pscale = (PSCALE != 0) ? PSCALE : pscale_runtime;

    const int bc  = blockIdx.z;
    const int k   = blockIdx.y;       // this CTA's slice of the K-packed val
    const int ho  = blockIdx.x;       // bigger-grid lat (= kernel-local "ho")
    const int tid = threadIdx.x;

    // Shmem caches inp[bc, k, hi, :], doubled to 2*Wi for no-mod inner index.
    extern __shared__ __align__(sizeof(double)) unsigned char __sh_raw[];
    STORAGE_T *sh = reinterpret_cast<STORAGE_T *>(__sh_raw);

    // Per-thread output-cell accumulators. Thread `tid` owns output cells
    // pp = tid, tid + BLOCK_X, ..., tid + (ELXTH-1)*BLOCK_X (those that fall in [0, Wo)).
    COMPUTE_T regs[ELXTH];
    #pragma unroll
    for (int i = 0; i < ELXTH; i++) regs[i] = COMPUTE_T(0);

    int prev_hi = -1;

    // Iterate pscale buckets for this ho. Within each bucket the K-packed
    // psi_T row is a single CSR-style list of (ho_orig, wi_offset) tuples,
    // each with K_pad stacked vals — we use the k-th.
    for (int r = 0; r < pscale; r++) {
        const int     row_T = ho * pscale + r;
        const int64_t cnt   = count_T[row_T];

        const int64_t   *idx_row  = idx_T  + (int64_t)row_T * NBR_PAD_T * 2;
        const STORAGE_T *vals_row = vals_T + (int64_t)row_T * NBR_PAD_T * K_pad;

        for (int64_t nz = 0; nz < cnt; nz++) {
            const int       ho_orig   = (int)idx_row[nz * 2 + 0];   // kernel-local "hi" (smaller-grid lat)
            const int       wi_offset = (int)idx_row[nz * 2 + 1];
            const COMPUTE_T val       = static_cast<COMPUTE_T>(vals_row[nz * K_pad + k]);

            // Refill shmem cache when (hi) changes. (ker is fixed across the
            // CTA — that's the split-by-K parallelism.)
            if (ho_orig != prev_hi) {
                __syncthreads();
                const int64_t inp_base = (((int64_t)bc * K + k) * Hi + ho_orig) * Wi;
                for (int i = tid; i < Wi; i += BLOCK_X) {
                    const STORAGE_T v = inp[inp_base + i];
                    sh[i]      = v;
                    sh[Wi + i] = v;
                }
                __syncthreads();
                prev_hi = ho_orig;
            }

            // Inner loop: each thread accumulates into its ELXTH output-cell regs.
            // For output cell pp the matching shmem index is
            //   wi_idx = (pp - wi_offset) / pscale, with wrap.
            // Bucketing invariant (wi_offset%pscale == r, pp%pscale == r) makes
            // the division exact when PSCALE > 1.
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
                if (delta < 0) delta += Wo;
                const int wi_idx = (PSCALE == 1) ? delta : (delta / pscale);
                regs[i] += val * static_cast<COMPUTE_T>(sh[wi_idx]);
            }
        }
    }

    // AtomicAdd regs to global scratch (fp32). K-way contention per cell —
    // moderate, well within HW throughput. Host casts scratch back to
    // grad_inp's dtype after the kernel completes.
    const int64_t out_base = ((int64_t)bc * Ho + ho) * Wo;
    #pragma unroll
    for (int i = 0; i < ELXTH; i++) {
        const int pp = i * BLOCK_X + tid;
        if (pp < Wo) {
            atomicAdd(&out_scratch[out_base + pp], regs[i]);
        }
    }
}


// Host wrapper. Allocates fp32 scratch, dispatches the templated kernel,
// casts back to grad_out's dtype.
torch::Tensor disco_cuda_bwd_kpacked(
    torch::Tensor grad_out,    // [B, C, K, Hi, Wi]
    torch::Tensor idx_T,       // [Hi*pscale, NBR_PAD_T, 2]
    torch::Tensor vals_T,      // [Hi*pscale, NBR_PAD_T, K_pad]
    torch::Tensor count_T,     // [Hi*pscale]
    int64_t K, int64_t Ho, int64_t Wo)
{
    CHECK_CUDA_INPUT_TENSOR(grad_out);
    CHECK_CUDA_INPUT_TENSOR(idx_T);
    CHECK_CUDA_INPUT_TENSOR(vals_T);
    CHECK_CUDA_INPUT_TENSOR(count_T);

    const int64_t B  = grad_out.size(0);
    const int64_t C  = grad_out.size(1);
    const int64_t BC = B * C;
    const int64_t Hi = grad_out.size(3);
    const int64_t Wi = grad_out.size(4);

    TORCH_CHECK(grad_out.size(2) == K,
                "grad_out.size(2) (", grad_out.size(2), ") != K (", K, ")");
    TORCH_CHECK(Wo % Wi == 0,
                "Wo (", Wo, ") must be an integer multiple of Wi (", Wi, ")");
    const int64_t pscale = Wo / Wi;
    TORCH_CHECK(count_T.size(0) == Ho * pscale,
                "count_T length (", count_T.size(0), ") inconsistent with Ho*pscale (",
                Ho * pscale, ")");
    TORCH_CHECK(idx_T.dim()  == 3 && idx_T.size(0)  == Ho * pscale && idx_T.size(2) == 2,
                "idx_T must be [Ho*pscale, NBR_PAD_T, 2]");
    TORCH_CHECK(vals_T.dim() == 3 && vals_T.size(0) == Ho * pscale,
                "vals_T must be [Ho*pscale, NBR_PAD_T, K_pad]");

    const int64_t NBR_PAD_T = idx_T.size(1);
    const int64_t K_pad     = vals_T.size(2);
    TORCH_CHECK(vals_T.size(1) == NBR_PAD_T,
                "vals_T NBR_PAD_T (", vals_T.size(1), ") != idx_T NBR_PAD_T (", NBR_PAD_T, ")");
    TORCH_CHECK(K_pad >= K, "K_pad (", K_pad, ") must be >= K (", K, ")");

    // Scratch for atomicAdd accumulation. Matches grad_out's dtype, which —
    // for the Float/Double types this dispatch supports — equals compute_t.
    // (When narrow-dtype dispatch is added later, switch this to fp32 scratch
    // and cast back at the end.)
    auto scratch_opts = torch::TensorOptions().device(grad_out.device()).dtype(grad_out.dtype());
    torch::Tensor out_scratch = torch::zeros({B, C, Ho, Wo}, scratch_opts);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Grid: (Ho, K, BC). BC is the slowest axis — keeps each (ho, k) plane
    // contiguous in CTA scheduling, which tends to help L2 reuse on idx_T /
    // vals_T (both are independent of BC).
    constexpr int BDIM_X = 128;
    dim3 block(BDIM_X);
    dim3 grid((unsigned)Ho, (unsigned)K, (unsigned)BC);

    constexpr int ELXTH_CAP = ELXTH_MAX;  // 32

    AT_DISPATCH_FLOATING_TYPES(grad_out.scalar_type(), "disco_backward_kpacked_cuda", ([&] {
        using storage_t = scalar_t;
        using compute_t = typename at::opmath_type<storage_t>;
        const size_t shmem_bytes = sizeof(storage_t) * 2 * Wi;

        auto launch = [&](auto elxth, auto pscale_tag) {
            constexpr int ELXTH  = decltype(elxth)::value;
            constexpr int PSCALE = decltype(pscale_tag)::value;
            disco_bwd_gather_kpacked_blk<BDIM_X, ELXTH, PSCALE, storage_t, compute_t>
                <<<grid, block, shmem_bytes, stream>>>(
                (int)Hi, (int)Wi, (int)K, (int)K_pad,
                (int)Ho, (int)Wo, (int)pscale, (int)NBR_PAD_T,
                idx_T.data_ptr<int64_t>(),
                reinterpret_cast<const storage_t*>(vals_T.data_ptr()),
                count_T.data_ptr<int64_t>(),
                grad_out.data_ptr<storage_t>(),
                out_scratch.data_ptr<compute_t>());
        };

        auto launch_with_pscale = [&](auto elxth) {
            switch ((int)pscale) {
                case 1: launch(elxth, std::integral_constant<int, 1>{}); break;
                case 2: launch(elxth, std::integral_constant<int, 2>{}); break;
                case 3: launch(elxth, std::integral_constant<int, 3>{}); break;
                case 4: launch(elxth, std::integral_constant<int, 4>{}); break;
                default: launch(elxth, std::integral_constant<int, 0>{}); break;
            }
        };

        const int elxth_needed = (int)((Wo + BDIM_X - 1) / BDIM_X);
        TORCH_CHECK(elxth_needed <= ELXTH_CAP,
                    "Wo (", Wo, ") exceeds supported maximum (BDIM_X*ELXTH_CAP = ",
                    BDIM_X * ELXTH_CAP, ")");

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

    // Scratch was allocated in grad_out.dtype() — already satisfies the API
    // contract (grad_inp.dtype == grad_out.dtype). Cast becomes a no-op when
    // dtypes already match; kept for clarity and forward compatibility with
    // a future narrow-dtype scratch path.
    return out_scratch.to(grad_out.dtype());
}


TORCH_LIBRARY_IMPL(disco_kernels, CUDA, m)
{
    m.impl("backward_kpacked", &disco_cuda_bwd_kpacked);
}

}  // namespace disco_kernels
