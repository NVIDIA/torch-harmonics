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

    // Ragged NHC ABI, identical to the CUDA ragged kernels: kx, vx, qy are physical
    // (B, npoints, num_heads * C) and contiguous. Returns (y, y_hi, alpha_sum,
    // qdotk_max) -- the trailing three are the backward's bookkeeping, and are fp32
    // whatever the activations are, because that is what the schema commits to and
    // what the backward reads unconditionally.
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    s2_attention_fwd_ragged_cpu(at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor ring_weights,
                                at::Tensor psi_seg, at::Tensor psi_seg_off, at::Tensor ring_base, at::Tensor ring_size,
                                int64_t num_heads, int64_t npoints_out)
    {
        CHECK_CPU_INPUT_TENSOR(kx);
        CHECK_CPU_INPUT_TENSOR(vx);
        CHECK_CPU_INPUT_TENSOR(qy);
        CHECK_CPU_INPUT_TENSOR(ring_weights);
        CHECK_CPU_INPUT_TENSOR(psi_seg);
        CHECK_CPU_INPUT_TENSOR(psi_seg_off);
        CHECK_CPU_INPUT_TENSOR(ring_base);
        CHECK_CPU_INPUT_TENSOR(ring_size);

        TORCH_CHECK(kx.dim() == 3, "kx must be (B, npoints_in, num_heads * C_k), got ", kx.dim(), " dims");
        TORCH_CHECK(vx.dim() == 3, "vx must be (B, npoints_in, num_heads * C_v), got ", vx.dim(), " dims");
        TORCH_CHECK(qy.dim() == 3, "qy must be (B, npoints_out, num_heads * C_k), got ", qy.dim(), " dims");

        TORCH_CHECK(num_heads >= 1, "num_heads must be positive, got ", num_heads);
        TORCH_CHECK(qy.size(2) % num_heads == 0, "q/k channel count (", qy.size(2),
                    ") must be divisible by num_heads (", num_heads, ")");
        TORCH_CHECK(vx.size(2) % num_heads == 0, "v channel count (", vx.size(2), ") must be divisible by num_heads (",
                    num_heads, ")");
        TORCH_CHECK(qy.size(1) == npoints_out, "qy has ", qy.size(1), " points but npoints_out is ", npoints_out);
        TORCH_CHECK(kx.size(1) == vx.size(1), "kx has ", kx.size(1), " points but vx has ", vx.size(1));
        TORCH_CHECK(psi_seg_off.size(0) == npoints_out + 1, "seg_off must have npoints_out + 1 = ", npoints_out + 1,
                    " entries, got ", psi_seg_off.size(0));
        TORCH_CHECK(psi_seg.dim() == 2 && psi_seg.size(1) == 3, "seg must be (nsegs, 3)");

        // One dtype for every activation: the kernel reads them through a single
        // float pointer each, so a mismatch would be reinterpreted, not converted.
        TORCH_CHECK(kx.scalar_type() == qy.scalar_type(), "k dtype (", kx.scalar_type(), ") must match q dtype (",
                    qy.scalar_type(), ")");
        TORCH_CHECK(vx.scalar_type() == qy.scalar_type(), "v dtype (", vx.scalar_type(), ") must match q dtype (",
                    qy.scalar_type(), ")");

        // fp32 only, as on the product-grid CPU path: reduced-precision arithmetic is
        // emulated here, so upcasting costs nothing and keeps the op device-agnostic.
        const auto inp_dtype = qy.scalar_type();
        kx = fold_heads_ragged(kx.to(torch::kFloat32), num_heads).contiguous();
        vx = fold_heads_ragged(vx.to(torch::kFloat32), num_heads).contiguous();
        qy = fold_heads_ragged(qy.to(torch::kFloat32), num_heads).contiguous();

        const auto f32 = qy.options().dtype(torch::kFloat32);
        const int64_t nbatch_heads = kx.size(0);
        const int64_t npoints_in = kx.size(1);
        const int64_t nchan_in = qy.size(2);
        const int64_t nchan_out = vx.size(2);

        auto y = torch::zeros({nbatch_heads, npoints_out, nchan_out}, f32);
        auto alpha_sum = torch::zeros({nbatch_heads, npoints_out}, f32);
        auto qdotk_max = torch::zeros({nbatch_heads, npoints_out}, f32);

        s2_attn_fwd_ragged_cpu_kernel(
            kx.data_ptr<float>(), vx.data_ptr<float>(), qy.data_ptr<float>(), ring_weights.data_ptr<float>(),
            psi_seg.data_ptr<int32_t>(), psi_seg_off.data_ptr<int32_t>(), ring_base.data_ptr<int64_t>(),
            ring_size.data_ptr<int64_t>(), y.data_ptr<float>(), alpha_sum.data_ptr<float>(),
            qdotk_max.data_ptr<float>(), nbatch_heads, npoints_in, npoints_out, nchan_in, nchan_out);

        // y_hi is the fp32 output. On CUDA it exists because the activations may be
        // narrower than the accumulation; here the two coincide, so it carries the same
        // values -- kept rather than elided so the schema holds on both devices.
        auto y_hi = unfold_heads_ragged(y, num_heads).contiguous();

        // Distinct storage, always. A custom operator may not return two aliases of one
        // tensor, and `.to(dtype)` is a no-op returning its argument when the dtype
        // already matches -- which is the fp32 case, i.e. the common one here.
        auto y_out = y_hi.to(inp_dtype);
        if (y_out.is_same(y_hi)) { y_out = y_hi.clone(); }

        // statistics stay (B * heads, npoints), which is how the backward indexes them
        return {y_out, y_hi, alpha_sum, qdotk_max};
    }

    TORCH_LIBRARY_IMPL(attention_kernels, CPU, m) { m.impl("forward_ragged", &s2_attention_fwd_ragged_cpu); }

} // namespace attention_kernels
