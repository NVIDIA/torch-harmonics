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
// Disco backward — K-packed dense psi (host wrapper / op registration)
// =====================================================================================
//
// Allocates an fp32 grad_inp scratch buffer, dispatches to the WGMMA backward
// kernel when its preconditions hold (Hopper + bf16/fp16 + K_PAD ∈ {8,16} +
// shape constraints), and casts the scratch back to grad_out's dtype.
//
// First-pass scope: WGMMA path only. On non-Hopper devices or for unsupported
// configs, the op errors out — callers should route to backward_csr instead.
// (A scalar K-packed backward fallback can be added later if we decide to
// retire backward_csr in favor of a unified kpacked path.)
// =====================================================================================

#include "../disco.h"
#include "disco_cuda.cuh"

#include <ATen/Dispatch.h>

namespace disco_kernels {

// Forward declaration of the WGMMA host wrapper (defined in
// disco_cuda_bwd_dense_kpacked_wgmma.cu).
bool disco_cuda_bwd_dense_kpacked_wgmma_try(
    torch::Tensor grad_out,
    torch::Tensor pack_idx,
    torch::Tensor pack_val,
    torch::Tensor pack_count,
    torch::Tensor grad_inp_scratch,
    int64_t K, int64_t Hi, int64_t Wi);

// Op signature matches forward_dense_kpacked: takes grad_out (where forward
// produced out) plus the same kpacked psi buffers, plus K / Hi / Wi (the
// input spatial shape we're producing the gradient for).
torch::Tensor disco_cuda_bwd_dense_kpacked(
    torch::Tensor grad_out,             // [B, C, K, Ho, Wo]
    torch::Tensor pack_idx,
    torch::Tensor pack_val,
    torch::Tensor pack_count,
    int64_t K, int64_t Hi, int64_t Wi)
{
    CHECK_CUDA_INPUT_TENSOR(grad_out);
    CHECK_CUDA_INPUT_TENSOR(pack_idx);
    CHECK_CUDA_INPUT_TENSOR(pack_val);
    CHECK_CUDA_INPUT_TENSOR(pack_count);

    const int64_t B = grad_out.size(0);
    const int64_t C = grad_out.size(1);

    TORCH_CHECK(grad_out.dim() == 5, "grad_out must be 5D [B, C, K, Ho, Wo]");
    TORCH_CHECK(grad_out.size(2) == K,
                "grad_out.size(2) (", grad_out.size(2), ") != K (", K, ")");

    // Allocate fp32 scratch — atomicAdd accumulates here, casts to inp.dtype below.
    auto scratch_options = torch::TensorOptions().device(grad_out.device()).dtype(at::ScalarType::Float);
    torch::Tensor grad_inp_scratch = torch::zeros({B, C, Hi, Wi}, scratch_options);

    const bool ok = disco_cuda_bwd_dense_kpacked_wgmma_try(
        grad_out, pack_idx, pack_val, pack_count, grad_inp_scratch, K, Hi, Wi);

    TORCH_CHECK(ok,
        "backward_dense_kpacked currently requires the WGMMA fast path: "
        "Hopper SM_90 + bf16/fp16 + K_PAD ∈ {8,16} + Wi%Wo==0 + Wo%8==0. "
        "Use backward_csr for unsupported configurations.");

    // Cast scratch to grad_out's dtype (matches the API contract: grad_inp.dtype == inp.dtype).
    return grad_inp_scratch.to(grad_out.dtype());
}

TORCH_LIBRARY_IMPL(disco_kernels, CUDA, m)
{
    m.impl("backward_dense_kpacked", &disco_cuda_bwd_dense_kpacked);
}

}  // namespace disco_kernels
