// coding=utf-8
//
// SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
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

#include "attention_cpu_ragged.h"

namespace attention_kernels
{

    std::tuple<at::Tensor, at::Tensor, at::Tensor>
    s2_attention_bwd_ragged_cpu(at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor dy, at::Tensor y,
                                at::Tensor y_hi, at::Tensor alpha_sum, at::Tensor qdotk_max, at::Tensor ring_weights,
                                at::Tensor psi_seg, at::Tensor psi_seg_off, at::Tensor ring_base, at::Tensor ring_size,
                                int64_t num_heads, int64_t npoints_out)
    {
        CHECK_CPU_INPUT_TENSOR(kx);
        CHECK_CPU_INPUT_TENSOR(vx);
        CHECK_CPU_INPUT_TENSOR(qy);
        CHECK_CPU_INPUT_TENSOR(dy);
        CHECK_CPU_INPUT_TENSOR(alpha_sum);
        CHECK_CPU_INPUT_TENSOR(qdotk_max);
        CHECK_CPU_INPUT_TENSOR(ring_weights);
        CHECK_CPU_INPUT_TENSOR(psi_seg);
        CHECK_CPU_INPUT_TENSOR(psi_seg_off);
        CHECK_CPU_INPUT_TENSOR(ring_base);
        CHECK_CPU_INPUT_TENSOR(ring_size);

        TORCH_CHECK(num_heads >= 1, "num_heads must be positive, got ", num_heads);
        TORCH_CHECK(qy.size(1) == npoints_out, "qy has ", qy.size(1), " points but npoints_out is ", npoints_out);
        TORCH_CHECK(psi_seg_off.size(0) == npoints_out + 1, "seg_off must have npoints_out + 1 = ", npoints_out + 1,
                    " entries, got ", psi_seg_off.size(0));

        const auto inp_dtype = qy.scalar_type();

        // integral_p = dy_p . y_p, the term the softmax derivative subtracts. Formed
        // before the head fold, on the packed layout the caller handed in, then
        // folded with everything else so the kernel sees one indexing.
        //
        // y_hi rather than y where it is present: it is the fp32 output, so on a path
        // whose activations are narrower this is the only copy that has not already
        // been rounded. Here the two coincide, but reading the same one the CUDA
        // launcher reads keeps the two implementations arithmetically comparable.
        auto y_for_integral = y_hi.defined() && y_hi.numel() == y.numel() ? y_hi : y;

        auto dy_f = fold_heads_ragged(dy.to(torch::kFloat32), num_heads).contiguous();
        auto y_f = fold_heads_ragged(y_for_integral.to(torch::kFloat32), num_heads).contiguous();
        auto integral = (dy_f * y_f).sum(-1).contiguous();

        auto kx_f = fold_heads_ragged(kx.to(torch::kFloat32), num_heads).contiguous();
        auto vx_f = fold_heads_ragged(vx.to(torch::kFloat32), num_heads).contiguous();
        auto qy_f = fold_heads_ragged(qy.to(torch::kFloat32), num_heads).contiguous();

        const int64_t nbatch_heads = kx_f.size(0);
        const int64_t npoints_in = kx_f.size(1);
        const int64_t nchan_in = qy_f.size(2);
        const int64_t nchan_out = vx_f.size(2);

        auto dkx = torch::zeros_like(kx_f);
        auto dvx = torch::zeros_like(vx_f);
        auto dqy = torch::zeros_like(qy_f);

        s2_attn_bwd_ragged_cpu_kernel(
            kx_f.data_ptr<float>(), vx_f.data_ptr<float>(), qy_f.data_ptr<float>(), dy_f.data_ptr<float>(),
            integral.data_ptr<float>(), alpha_sum.data_ptr<float>(), qdotk_max.data_ptr<float>(),
            ring_weights.data_ptr<float>(), psi_seg.data_ptr<int32_t>(), psi_seg_off.data_ptr<int32_t>(),
            ring_base.data_ptr<int64_t>(), ring_size.data_ptr<int64_t>(), dkx.data_ptr<float>(), dvx.data_ptr<float>(),
            dqy.data_ptr<float>(), nbatch_heads, npoints_in, npoints_out, nchan_in, nchan_out);

        return {unfold_heads_ragged(dkx, num_heads).contiguous().to(inp_dtype),
                unfold_heads_ragged(dvx, num_heads).contiguous().to(inp_dtype),
                unfold_heads_ragged(dqy, num_heads).contiguous().to(inp_dtype)};
    }

    TORCH_LIBRARY_IMPL(attention_kernels, CPU, m) { m.impl("backward_ragged", &s2_attention_bwd_ragged_cpu); }

} // namespace attention_kernels
