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
// DISCO backward — K-packed gather with grid-split over the pscale residue (r)
// =====================================================================================
//
// Same math + buffer layout as `disco_cuda_bwd_kpacked.cu`, but with each CTA
// owning ONE (k, ho, r) tuple instead of (k, ho) iterating r internally.
// Motivation: the inner-loop bucket-residue check in the vanilla kpacked
// kernel masks out (pscale-1)/pscale of the lanes when pscale > 1, paying
// the warp-divergence + masked-skip cost without doing the corresponding
// useful work. This kernel makes r a grid axis so:
//
//   * Inner loop iterates only the pp values that match the bucket residue —
//     no `continue` masking, no divergence.
//   * Per-thread tile shrinks from ELXTH = ceil(Wo / BLOCK_X) to
//     ELXTH_R = ceil(Wo / (pscale * BLOCK_X)). For pscale=3, Wo=1440,
//     BLOCK_X=128: ELXTH_R = 4 (vs 12 in the vanilla kpacked).
//   * Grid size grows by `pscale` — extra parallelism on top of split-by-K.
//
// Parallelism for K=16, pscale=3 on this shape: K * Ho * pscale * BC =
// 16 * 360 * 3 * 256 = 4.4M CTAs (vs 736k for vanilla kpacked = 6× more,
// vs 245k for scatter = 18× more). Per-CTA work is correspondingly smaller.
//
// Trade-off: atomic-add contention per output cell goes from K-way (vanilla
// kpacked) to K-way still (each of the pscale r-CTAs writes a disjoint subset
// of out[bc, ho, *] cells, so r doesn't add contention). Memory traffic on
// vals_T / idx_T is identical to the vanilla kpacked kernel.
// =====================================================================================

#include "../disco.h"
#include "disco_cuda.cuh"

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <type_traits>

namespace disco_kernels {

template <int BLOCK_X, int ELXTH_R, int PSCALE, typename STORAGE_T, typename COMPUTE_T>
__launch_bounds__(BLOCK_X)
__global__ void disco_bwd_gather_kpacked_rsplit_blk(
    const int Hi, const int Wi, const int K, const int K_pad,
    const int Ho, const int Wo, const int pscale_runtime,
    const int NBR_PAD_T,
    const int64_t *__restrict__ idx_T,        // [Hi*pscale, NBR_PAD_T, 2]   (ho, wi_offset)
    const STORAGE_T *__restrict__ vals_T,     // [Hi*pscale, NBR_PAD_T, K_pad]
    const int64_t *__restrict__ count_T,      // [Hi*pscale]
    const STORAGE_T *__restrict__ inp,        // grad_out [B, C, K, Hi, Wi]
    COMPUTE_T *__restrict__ out_scratch)      // grad_inp scratch [B, C, Ho, Wo]
{
    const int pscale = (PSCALE != 0) ? PSCALE : pscale_runtime;

    const int bc      = blockIdx.z;
    const int k       = blockIdx.x;
    const int row_T   = blockIdx.y;            // [0, Ho*pscale)
    const int ho      = row_T / pscale;
    const int r       = row_T - ho * pscale;
    const int tid     = threadIdx.x;

    // Shmem caches inp[bc, k, hi, :], doubled to 2*Wi for no-mod inner index.
    extern __shared__ __align__(sizeof(double)) unsigned char __sh_raw[];
    STORAGE_T *sh = reinterpret_cast<STORAGE_T *>(__sh_raw);

    // Per-thread regs hold ELXTH_R output cells, all with residue r:
    //   pp(j) = (j * BLOCK_X + tid) * pscale + r,   j in [0, ELXTH_R).
    COMPUTE_T regs[ELXTH_R];
    #pragma unroll
    for (int j = 0; j < ELXTH_R; j++) regs[j] = COMPUTE_T(0);

    int prev_hi = -1;

    const int64_t cnt      = count_T[row_T];
    const int64_t  *idx_row  = idx_T  + (int64_t)row_T * NBR_PAD_T * 2;
    const STORAGE_T *vals_row = vals_T + (int64_t)row_T * NBR_PAD_T * K_pad;

    for (int64_t nz = 0; nz < cnt; nz++) {
        const int       ho_orig   = (int)idx_row[nz * 2 + 0];
        const int       wi_offset = (int)idx_row[nz * 2 + 1];
        const COMPUTE_T val       = static_cast<COMPUTE_T>(vals_row[nz * K_pad + k]);

        // (Re)load inp[bc, k, ho_orig, :] into shmem on hi change.
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

        // Inner loop: only pp values with residue r — no per-iter mask.
        #pragma unroll
        for (int j = 0; j < ELXTH_R; j++) {
            const int pp_idx = j * BLOCK_X + tid;
            const int pp     = pp_idx * pscale + r;
            if (pp >= Wo) break;
            int delta = pp - wi_offset;
            if (delta < 0) delta += Wo;
            const int wi_idx = (PSCALE == 1) ? delta : (delta / pscale);
            regs[j] += val * static_cast<COMPUTE_T>(sh[wi_idx]);
        }
    }

    // AtomicAdd regs to global scratch. K-way contention per output cell
    // (the pscale r-CTAs write disjoint cells, so r does not add contention).
    const int64_t out_base = ((int64_t)bc * Ho + ho) * Wo;
    #pragma unroll
    for (int j = 0; j < ELXTH_R; j++) {
        const int pp_idx = j * BLOCK_X + tid;
        const int pp     = pp_idx * pscale + r;
        if (pp < Wo) {
            atomicAdd(&out_scratch[out_base + pp], regs[j]);
        }
    }
}


// Host wrapper.
torch::Tensor disco_cuda_bwd_kpacked_rsplit(
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

    const int64_t NBR_PAD_T = idx_T.size(1);
    const int64_t K_pad     = vals_T.size(2);
    TORCH_CHECK(K_pad >= K, "K_pad (", K_pad, ") must be >= K (", K, ")");

    auto scratch_opts = torch::TensorOptions().device(grad_out.device()).dtype(grad_out.dtype());
    torch::Tensor out_scratch = torch::zeros({B, C, Ho, Wo}, scratch_opts);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Grid: (K, Ho*pscale, BC). K is fastest so sibling k-CTAs at the same
    // row_T are close in scheduling — good for L2 reuse on idx_T / vals_T.
    constexpr int BDIM_X = 128;
    dim3 block(BDIM_X);
    dim3 grid((unsigned)K, (unsigned)(Ho * pscale), (unsigned)BC);

    // ELXTH_R = ceil(Wo / (pscale * BDIM_X)) — number of output cells per
    // thread within one r bucket. Lookup table for known pscale values.
    constexpr int ELXTH_R_CAP = ELXTH_MAX;   // 32

    AT_DISPATCH_FLOATING_TYPES(grad_out.scalar_type(), "disco_backward_kpacked_rsplit_cuda", ([&] {
        using storage_t = scalar_t;
        using compute_t = typename at::opmath_type<storage_t>;
        const size_t shmem_bytes = sizeof(storage_t) * 2 * Wi;

        auto launch = [&](auto elxth_r, auto pscale_tag) {
            constexpr int ELXTH_R = decltype(elxth_r)::value;
            constexpr int PSCALE  = decltype(pscale_tag)::value;
            disco_bwd_gather_kpacked_rsplit_blk<BDIM_X, ELXTH_R, PSCALE, storage_t, compute_t>
                <<<grid, block, shmem_bytes, stream>>>(
                (int)Hi, (int)Wi, (int)K, (int)K_pad,
                (int)Ho, (int)Wo, (int)pscale, (int)NBR_PAD_T,
                idx_T.data_ptr<int64_t>(),
                reinterpret_cast<const storage_t*>(vals_T.data_ptr()),
                count_T.data_ptr<int64_t>(),
                grad_out.data_ptr<storage_t>(),
                out_scratch.data_ptr<compute_t>());
        };

        auto launch_with_pscale = [&](auto elxth_r) {
            switch ((int)pscale) {
                case 1: launch(elxth_r, std::integral_constant<int, 1>{}); break;
                case 2: launch(elxth_r, std::integral_constant<int, 2>{}); break;
                case 3: launch(elxth_r, std::integral_constant<int, 3>{}); break;
                case 4: launch(elxth_r, std::integral_constant<int, 4>{}); break;
                default: launch(elxth_r, std::integral_constant<int, 0>{}); break;
            }
        };

        // ELXTH_R is set per (Wo, pscale, BDIM_X). For Wo=1440, BDIM_X=128:
        //   pscale=1 → 12, pscale=2 → 6, pscale=3 → 4, pscale=4 → 3.
        const int wo_per_r        = (int)((Wo + pscale - 1) / pscale);   // ceil(Wo/pscale)
        const int elxth_r_needed  = (wo_per_r + BDIM_X - 1) / BDIM_X;
        TORCH_CHECK(elxth_r_needed <= ELXTH_R_CAP,
                    "Wo/pscale (", wo_per_r, ") exceeds supported maximum "
                    "(BDIM_X*ELXTH_R_CAP = ", BDIM_X * ELXTH_R_CAP, ")");

        switch (elxth_r_needed) {
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
            default: launch_with_pscale(std::integral_constant<int, ELXTH_R_CAP>{}); break;
        }
    }));

    return out_scratch;
}


TORCH_LIBRARY_IMPL(disco_kernels, CUDA, m)
{
    m.impl("backward_kpacked_rsplit", &disco_cuda_bwd_kpacked_rsplit);
}

}  // namespace disco_kernels
