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
// Disco forward — K-packed dense psi (CUDA, scalar checkpoint)
// =====================================================================================
//
// Scalar K-packed disco fwd kernel. This is the *correctness checkpoint* for the
// upcoming WGMMA path: same data layout (K-packed psi, where every k_kern shares
// the (hi, wi_base) support per output latitude — see
// `_maybe_kpack_psi` in convolution.py), same per-CTA tile shape (M = BC_TILE *
// WO_TILE outputs × N = K_PAD k_kerns), but the inner contraction is scalar
// FMAs rather than tensor-core MMAs. Matching the WGMMA tile shape here means
// the WGMMA upgrade replaces only the inner FMA loop, not the surrounding
// kernel structure or dispatch.
//
// Inputs:
//   inp        [B, C, Hi, Wi]               STORAGE_T (bf16/fp16/fp32/fp64)
//   pack_idx   [Ho, NBR_PAD, 2]              int64    (hi, wi_base) shared across k_kern
//   pack_val   [Ho, NBR_PAD, K_PAD]          COMPUTE_T psi values; K_PAD = ceil(K/8)*8,
//                                                     padded with zeros for k >= K
//   pack_count [Ho]                          int64    nz count per output ho
//
// Output:
//   out        [B, C, K, Ho, Wo]             STORAGE_T (only k < K columns written)
//
// Per CTA:
//   threads = BC_TILE * WO_TILE = 64
//   covers  (BC_TILE channels) × (WO_TILE wo positions) × (all K k_kerns)
//
// Each thread:
//   owns one (bc, wo_local) output cell, holds K_PAD fp32 accumulators,
//   iterates all nz, writes K outputs at the end.
// =====================================================================================

#include "../disco.h"
#include "disco_cuda.cuh"

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>

namespace disco_kernels {

template <int BC_TILE, int WO_TILE, int K_PAD, typename STORAGE_T, typename COMPUTE_T>
__global__ __launch_bounds__(BC_TILE * WO_TILE)
void disco_fwd_dense_kpacked_blk_k(
    int Hi, int Wi, int K, int Ho, int Wo, int NBR_PAD, int pscale,
    const int64_t   *__restrict__ pack_idx,    // [Ho, NBR_PAD, 2]
    const COMPUTE_T *__restrict__ pack_val,    // [Ho, NBR_PAD, K_PAD]
    const int64_t   *__restrict__ pack_count,  // [Ho]
    const STORAGE_T *__restrict__ inp,         // [B, C, Hi, Wi]
    STORAGE_T       *__restrict__ out)          // [B, C, K, Ho, Wo]
{
    const int tid       = threadIdx.x;
    const int bc_local  = tid / WO_TILE;
    const int wo_local  = tid - bc_local * WO_TILE;

    const int wo_per_ho = Wo / WO_TILE;          // host enforces Wo % WO_TILE == 0
    const int ho        = blockIdx.x / wo_per_ho;
    const int wo_strip  = blockIdx.x - ho * wo_per_ho;
    const int wo_base   = wo_strip * WO_TILE;
    const int wo        = wo_base + wo_local;

    const int bc        = blockIdx.y * BC_TILE + bc_local;

    const int64_t   *idx_ho = pack_idx + (int64_t)ho * NBR_PAD * 2;
    const COMPUTE_T *val_ho = pack_val + (int64_t)ho * NBR_PAD * K_PAD;
    const int        cnt    = (int)pack_count[ho];

    // Output base address. out[bc, k, ho, wo] = out + ((bc*K + k)*Ho + ho)*Wo + wo
    const int64_t out_base   = ((int64_t)bc * K) * (int64_t)Ho * Wo + (int64_t)ho * Wo + wo;
    const int64_t out_stride = (int64_t)Ho * Wo;

    if (cnt == 0) {
        // Zero this CTA's output cells (output tensor is zero-initialized in
        // host wrapper but make this explicit for clarity).
        return;
    }

    // Per-thread accumulator: K_PAD fp32 (fits in registers; K_PAD ≤ ~32).
    COMPUTE_T acc[K_PAD];
    #pragma unroll
    for (int k = 0; k < K_PAD; k++) acc[k] = static_cast<COMPUTE_T>(0);

    // Loop over nz. Shared support → all k_kern share (hi, wi_base) per nz, so
    // we read inp once and multiply against the K_PAD values v_packed[..., k].
    for (int nz = 0; nz < cnt; nz++) {
        const int hi      = (int)idx_ho[nz * 2 + 0];
        const int wi_base = (int)idx_ho[nz * 2 + 1];
        const int wi_full = (wi_base + pscale * wo) % Wi;

        const COMPUTE_T inp_val = static_cast<COMPUTE_T>(
            inp[(int64_t)bc * Hi * Wi + (int64_t)hi * Wi + wi_full]);

        const COMPUTE_T *v_nz = val_ho + (int64_t)nz * K_PAD;
        #pragma unroll
        for (int k = 0; k < K_PAD; k++) {
            acc[k] += v_nz[k] * inp_val;
        }
    }

    // Write outputs. Only k < K is written; k in [K, K_PAD) is padded with zero
    // psi values, so the corresponding acc[k] == 0 — but those cells are not
    // part of the output tensor anyway.
    #pragma unroll
    for (int k = 0; k < K_PAD; k++) {
        if (k < K) {
            out[out_base + (int64_t)k * out_stride] = static_cast<STORAGE_T>(acc[k]);
        }
    }
}

template <int K_PAD, typename STORAGE_T, typename COMPUTE_T>
static void launch_dense_fwd_kpacked(
    int B, int C, int K, int Hi, int Wi, int Ho, int Wo, int NBR_PAD, int pscale,
    const int64_t *pack_idx, const COMPUTE_T *pack_val, const int64_t *pack_count,
    const STORAGE_T *inp, STORAGE_T *out, cudaStream_t stream)
{
    constexpr int BC_TILE = 8;
    constexpr int WO_TILE = 8;
    constexpr int NUM_THREADS = BC_TILE * WO_TILE;

    const int BC         = B * C;
    const int wo_per_ho  = Wo / WO_TILE;

    dim3 grid((unsigned)(Ho * wo_per_ho), (unsigned)(BC / BC_TILE));
    disco_fwd_dense_kpacked_blk_k<BC_TILE, WO_TILE, K_PAD, STORAGE_T, COMPUTE_T>
        <<<grid, NUM_THREADS, 0, stream>>>(
            Hi, Wi, K, Ho, Wo, NBR_PAD, pscale,
            pack_idx, pack_val, pack_count, inp, out);
}

torch::Tensor disco_cuda_fwd_dense_kpacked(torch::Tensor inp,
                                           torch::Tensor pack_idx,
                                           torch::Tensor pack_val,
                                           torch::Tensor pack_count,
                                           int64_t K, int64_t Ho, int64_t Wo)
{
    CHECK_CUDA_INPUT_TENSOR(inp);
    CHECK_CUDA_INPUT_TENSOR(pack_idx);
    CHECK_CUDA_INPUT_TENSOR(pack_val);
    CHECK_CUDA_INPUT_TENSOR(pack_count);

    const int64_t B  = inp.size(0);
    const int64_t C  = inp.size(1);
    const int64_t Hi = inp.size(2);
    const int64_t Wi = inp.size(3);

    TORCH_CHECK(Wi % Wo == 0, "Wi (", Wi, ") must be an integer multiple of Wo (", Wo, ")");

    TORCH_CHECK(pack_idx.dim() == 3 && pack_idx.size(0) == Ho && pack_idx.size(2) == 2,
                "pack_idx (kpacked) must have shape [Ho, NBR_PAD, 2]");
    TORCH_CHECK(pack_val.dim() == 3 && pack_val.size(0) == Ho,
                "pack_val (kpacked) must have shape [Ho, NBR_PAD, K_PAD]");
    TORCH_CHECK(pack_count.dim() == 1 && pack_count.size(0) == Ho,
                "pack_count (kpacked) must have shape [Ho]");
    TORCH_CHECK(pack_idx.size(1) == pack_val.size(1),
                "pack_idx and pack_val NBR_PAD dimensions must match");

    const int64_t NBR_PAD = pack_idx.size(1);
    const int64_t K_PAD   = pack_val.size(2);

    constexpr int BC_TILE = 8;
    constexpr int WO_TILE = 8;
    TORCH_CHECK(Wo % WO_TILE == 0,
                "Wo (", Wo, ") must be a multiple of WO_TILE (", WO_TILE, ")");
    TORCH_CHECK((B * C) % BC_TILE == 0,
                "B*C (", B * C, ") must be a multiple of BC_TILE (", BC_TILE, ")");
    TORCH_CHECK(K_PAD == 8 || K_PAD == 16 || K_PAD == 24 || K_PAD == 32,
                "K_PAD (", K_PAD, ") must be one of {8, 16, 24, 32} — got ", K_PAD);
    TORCH_CHECK(K <= K_PAD,
                "K (", K, ") must be <= K_PAD (", K_PAD, ")");

    const int pscale = (int)(Wi / Wo);

    auto options = torch::TensorOptions().device(inp.device()).dtype(inp.dtype());
    auto out = torch::zeros({B, C, K, Ho, Wo}, options);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // WGMMA fast path: Hopper SM_90+, bf16, K_PAD=8, pscale=1. The helper does
    // its own preconditions + runtime CC check and returns false if any fail,
    // in which case we drop into the scalar K-packed kernel below.
    if (disco_cuda_fwd_dense_kpacked_wgmma_try(inp, pack_idx, pack_val, pack_count, out, K, Ho, Wo)) {
        return out;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        inp.scalar_type(), "disco_fwd_dense_kpacked_cuda", ([&] {
        using storage_t = scalar_t;
        using compute_t = at::opmath_type<storage_t>;
        const auto compute_dtype = c10::CppTypeToScalarType<compute_t>::value;
        auto pack_val_c = (pack_val.scalar_type() == compute_dtype)
                              ? pack_val
                              : pack_val.to(compute_dtype);

        switch (K_PAD) {
        case 8:
            launch_dense_fwd_kpacked<8, storage_t, compute_t>(
                (int)B, (int)C, (int)K, (int)Hi, (int)Wi, (int)Ho, (int)Wo,
                (int)NBR_PAD, pscale,
                pack_idx.data_ptr<int64_t>(),
                pack_val_c.data_ptr<compute_t>(),
                pack_count.data_ptr<int64_t>(),
                inp.data_ptr<storage_t>(),
                out.data_ptr<storage_t>(), stream);
            break;
        case 16:
            launch_dense_fwd_kpacked<16, storage_t, compute_t>(
                (int)B, (int)C, (int)K, (int)Hi, (int)Wi, (int)Ho, (int)Wo,
                (int)NBR_PAD, pscale,
                pack_idx.data_ptr<int64_t>(),
                pack_val_c.data_ptr<compute_t>(),
                pack_count.data_ptr<int64_t>(),
                inp.data_ptr<storage_t>(),
                out.data_ptr<storage_t>(), stream);
            break;
        case 24:
            launch_dense_fwd_kpacked<24, storage_t, compute_t>(
                (int)B, (int)C, (int)K, (int)Hi, (int)Wi, (int)Ho, (int)Wo,
                (int)NBR_PAD, pscale,
                pack_idx.data_ptr<int64_t>(),
                pack_val_c.data_ptr<compute_t>(),
                pack_count.data_ptr<int64_t>(),
                inp.data_ptr<storage_t>(),
                out.data_ptr<storage_t>(), stream);
            break;
        case 32:
            launch_dense_fwd_kpacked<32, storage_t, compute_t>(
                (int)B, (int)C, (int)K, (int)Hi, (int)Wi, (int)Ho, (int)Wo,
                (int)NBR_PAD, pscale,
                pack_idx.data_ptr<int64_t>(),
                pack_val_c.data_ptr<compute_t>(),
                pack_count.data_ptr<int64_t>(),
                inp.data_ptr<storage_t>(),
                out.data_ptr<storage_t>(), stream);
            break;
        }
    }));

    return out;
}

TORCH_LIBRARY_IMPL(disco_kernels, CUDA, m)
{
    m.impl("forward_dense_kpacked", &disco_cuda_fwd_dense_kpacked);
}

}
