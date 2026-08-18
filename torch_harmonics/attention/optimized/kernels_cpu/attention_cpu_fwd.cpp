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

#include "attention_cpu_fwd.h"
#include "attention_cpu_fwd_upsample.h"

using namespace torch::indexing;

namespace attention_kernels
{

    // Fold the packed head axis into the batch dimension.
    //
    // Input is physical (B, nlat, nlon, num_heads * C); the compute kernels below
    // are head-agnostic and want (B * num_heads, nlat, nlon, C). Unlike the CUDA
    // path -- which addresses heads in place via a leading dimension -- this costs
    // a materialization, because in a channel-innermost layout the head axis is
    // interior and cannot be moved to the front by a view. That trade is
    // deliberate: the CPU path is the fallback/reference implementation, and one
    // copy here keeps four sets of loop kernels free of head bookkeeping.
    at::Tensor fold_heads(const at::Tensor &t, int64_t num_heads)
    {
        if (num_heads == 1) { return t; }
        const int64_t B = t.size(0), H = t.size(1), W = t.size(2), C = t.size(3) / num_heads;
        return t.reshape({B, H, W, num_heads, C}).permute({0, 3, 1, 2, 4}).reshape({B * num_heads, H, W, C});
    }

    // inverse of fold_heads
    at::Tensor unfold_heads(const at::Tensor &t, int64_t num_heads)
    {
        if (num_heads == 1) { return t; }
        const int64_t BH = t.size(0), H = t.size(1), W = t.size(2), C = t.size(3);
        const int64_t B = BH / num_heads;
        return t.reshape({B, num_heads, H, W, C}).permute({0, 2, 3, 1, 4}).reshape({B, H, W, num_heads * C});
    }

    // NHWC ABI: kx, vx, qy are physical (B, nlat, nlon, num_heads * C) and
    // contiguous; the result is returned in the same layout. Layout is never
    // inferred from strides -- the caller states it by construction.
    torch::Tensor s2_attention_fwd_cpu(at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor quad_weights,
                                       at::Tensor col_idx, at::Tensor row_off, int64_t num_heads, int64_t nlon_in,
                                       int64_t nlat_out, int64_t nlon_out)
    {
        CHECK_CPU_INPUT_TENSOR(kx);
        CHECK_CPU_INPUT_TENSOR(vx);
        CHECK_CPU_INPUT_TENSOR(qy);
        CHECK_CPU_INPUT_TENSOR(quad_weights);
        CHECK_CPU_INPUT_TENSOR(col_idx);
        CHECK_CPU_INPUT_TENSOR(row_off);

        // downsample/self-attention iff nlon_in is a multiple of nlon_out;
        // upsample iff nlon_out is a multiple of nlon_in. Equal (self) hits both
        // and routes through the gather kernel (pscale == 1).
        const bool downsample = (nlon_in % nlon_out == 0);
        const bool upsample = (nlon_out % nlon_in == 0);
        TORCH_CHECK(downsample || upsample, "either nlon_in (", nlon_in, ") must be an integer multiple of nlon_out (",
                    nlon_out, "), or vice versa");

        TORCH_CHECK(num_heads >= 1, "num_heads must be positive, got ", num_heads);
        TORCH_CHECK(qy.size(3) % num_heads == 0, "q/k channel count (", qy.size(3),
                    ") must be divisible by num_heads (", num_heads, ")");
        TORCH_CHECK(vx.size(3) % num_heads == 0, "v channel count (", vx.size(3), ") must be divisible by num_heads (",
                    num_heads, ")");

        // Every activation must share one dtype: the dispatch below selects a single
        // scalar_t from qy and the launchers reinterpret_cast k/v/q (and dy) to it, so
        // a mismatched input would be reinterpreted rather than converted.
        TORCH_CHECK(kx.scalar_type() == qy.scalar_type(), "k dtype (", kx.scalar_type(), ") must match q dtype (",
                    qy.scalar_type(), ")");
        TORCH_CHECK(vx.scalar_type() == qy.scalar_type(), "v dtype (", vx.scalar_type(), ") must match q dtype (",
                    qy.scalar_type(), ")");

        // The CPU kernels are fp32-only (the storage/compute split is a CUDA-only
        // Tier B optimization). Upcast fp16/bf16 inputs to fp32 here and cast the
        // result back at the end; on CPU reduced-precision compute is emulated, so
        // this loses no performance and keeps the Python op device-agnostic.
        const auto inp_dtype = qy.scalar_type();
        kx = kx.to(torch::kFloat32);
        vx = vx.to(torch::kFloat32);
        qy = qy.to(torch::kFloat32);

        // already NHWC by contract; only the head fold is needed
        kx = fold_heads(kx, num_heads).contiguous();
        vx = fold_heads(vx, num_heads).contiguous();
        qy = fold_heads(qy, num_heads).contiguous();

        const int64_t batch_size = kx.size(0);
        const int64_t nlat_in = kx.size(1);
        const int64_t nchannels_in = qy.size(3);
        const int64_t nchannels_out = vx.size(3);

        // y allocated as physical (B, H, W, C).
        auto y = torch::zeros({batch_size, nlat_out, nlon_out, nchannels_out}, qy.options());

        auto kx_arr = kx.packed_accessor64<float, 4>();
        auto vx_arr = vx.packed_accessor64<float, 4>();
        auto qy_arr = qy.packed_accessor64<float, 4>();
        auto y_arr = y.packed_accessor64<float, 4>();
        auto quad_weights_arr = quad_weights.packed_accessor64<float, 1>();
        auto col_idx_arr = col_idx.packed_accessor64<int64_t, 1>();
        auto roff_arr = row_off.packed_accessor64<int64_t, 1>();

        if (downsample) {
            s2_attn_fwd_kernel<float>(kx_arr, vx_arr, qy_arr, quad_weights_arr, col_idx_arr, roff_arr, y_arr, nlon_in,
                                      nlat_out, nlon_out, batch_size, nchannels_in, nchannels_out);
        } else {
            s2_attn_fwd_upsample_dispatch(kx_arr, vx_arr, qy_arr, quad_weights_arr, col_idx_arr, roff_arr, y_arr,
                                          nlon_in, nlat_in, nlat_out, nlon_out, batch_size, nchannels_in, nchannels_out);
        }

        // back to the packed (B, nlat_out, nlon_out, num_heads * C_v) NHWC form
        y = unfold_heads(y, num_heads).contiguous();
        return y.to(inp_dtype);
    }

    // Implement the operators: CPU
    TORCH_LIBRARY_IMPL(attention_kernels, CPU, m) { m.impl("forward", &s2_attention_fwd_cpu); }

} // namespace attention_kernels
