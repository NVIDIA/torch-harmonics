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

#include "attention_cpu_bwd.h"
#include "attention_cpu_bwd_upsample.h"

using namespace torch::indexing;

namespace attention_kernels
{

    // NHWC ABI with heads packed along channels -- see s2_attention_fwd_cpu.
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    // seg / seg_off are accepted but unused here: the CPU backward still walks the column
    // list. They are part of the shared schema because the CUDA backward consumes them,
    // and keeping this path on col_idx is what keeps it independent of that derivation.
    s2_attention_bwd_cpu(torch::Tensor kx, torch::Tensor vx, torch::Tensor qy, torch::Tensor dy,
                         torch::Tensor quad_weights, torch::Tensor col_idx, torch::Tensor row_off, torch::Tensor seg,
                         torch::Tensor seg_off, int64_t num_heads, int64_t nlon_in, int64_t nlat_out, int64_t nlon_out)
    {

        // Caller-visible shapes (NHWC, heads packed along channels):
        //   kx, vx          : (B, Hi, Wi, num_heads * C)
        //   qy, dy          : (B, Ho, Wo, num_heads * C)
        //   quad_weights    : (Hi,)
        //   dkx, dvx (out)  : same as kx, vx
        //   dqy (out)       : same as qy
        // The loop kernels are head-agnostic, so the wrapper folds heads into the
        // batch dimension on the way in and unfolds on the way out.

        CHECK_CPU_INPUT_TENSOR(kx);
        CHECK_CPU_INPUT_TENSOR(vx);
        CHECK_CPU_INPUT_TENSOR(qy);
        CHECK_CPU_INPUT_TENSOR(dy);
        CHECK_CPU_INPUT_TENSOR(quad_weights);
        CHECK_CPU_INPUT_TENSOR(col_idx);
        CHECK_CPU_INPUT_TENSOR(row_off);

        // direction selection: same as fwd. Self (nlon_in == nlon_out) hits both
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
        TORCH_CHECK(dy.scalar_type() == qy.scalar_type(), "dy dtype (", dy.scalar_type(), ") must match q dtype (",
                    qy.scalar_type(), ")");

        // The CPU kernels are fp32-only (storage/compute split is CUDA-only Tier B).
        // Upcast fp16/bf16 inputs to fp32 and cast the grads back at the end; CPU
        // reduced-precision compute is emulated, so this loses no performance and
        // keeps the Python op device-agnostic.
        const auto inp_dtype = qy.scalar_type();
        kx = kx.to(torch::kFloat32);
        vx = vx.to(torch::kFloat32);
        qy = qy.to(torch::kFloat32);
        dy = dy.to(torch::kFloat32);

        // already NHWC by contract; only the head fold is needed
        kx = fold_heads(kx, num_heads).contiguous();
        vx = fold_heads(vx, num_heads).contiguous();
        qy = fold_heads(qy, num_heads).contiguous();
        dy = fold_heads(dy, num_heads).contiguous();

        const int64_t batch_size = kx.size(0);
        const int64_t nlat_in = kx.size(1);
        const int64_t nchannels_in = qy.size(3);
        const int64_t nchannels_out = vx.size(3);

        // grads allocated as physical (B, H, W, C) zeros — matches input layout.
        auto dkx = torch::zeros({batch_size, nlat_in, nlon_in, nchannels_in}, kx.options());
        auto dvx = torch::zeros({batch_size, nlat_in, nlon_in, nchannels_out}, vx.options());
        auto dqy = torch::zeros({batch_size, nlat_out, nlon_out, nchannels_in}, qy.options());

        auto kx_arr = kx.packed_accessor64<float, 4>();
        auto vx_arr = vx.packed_accessor64<float, 4>();
        auto qy_arr = qy.packed_accessor64<float, 4>();
        auto dy_arr = dy.packed_accessor64<float, 4>();
        auto quad_weights_arr = quad_weights.packed_accessor64<float, 1>();
        auto col_idx_arr = col_idx.packed_accessor64<int64_t, 1>();
        auto roff_arr = row_off.packed_accessor64<int64_t, 1>();
        auto dqy_arr = dqy.packed_accessor64<float, 4>();
        auto dvx_arr = dvx.packed_accessor64<float, 4>();
        auto dkx_arr = dkx.packed_accessor64<float, 4>();

        if (downsample) {
            s2_attn_bwd_kernel<float>(kx_arr, vx_arr, qy_arr, dy_arr, quad_weights_arr, col_idx_arr, roff_arr, dqy_arr,
                                      dvx_arr, dkx_arr, nlon_in, nlat_out, nlon_out, batch_size, nchannels_in,
                                      nchannels_out);
        } else {
            s2_attn_bwd_upsample_dispatch(kx_arr, vx_arr, qy_arr, dy_arr, quad_weights_arr, col_idx_arr, roff_arr,
                                          dqy_arr, dvx_arr, dkx_arr, nlon_in, nlat_in, nlat_out, nlon_out, batch_size,
                                          nchannels_in, nchannels_out);
        }

        // back to the packed NHWC form the caller supplied
        dkx = unfold_heads(dkx, num_heads).contiguous();
        dvx = unfold_heads(dvx, num_heads).contiguous();
        dqy = unfold_heads(dqy, num_heads).contiguous();

        return std::make_tuple(dkx.to(inp_dtype), dvx.to(inp_dtype), dqy.to(inp_dtype));
    }

    TORCH_LIBRARY_IMPL(attention_kernels, CPU, m) { m.impl("backward", &s2_attention_bwd_cpu); }

} // namespace attention_kernels
