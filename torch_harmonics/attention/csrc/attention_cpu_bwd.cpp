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

#include "attention_cpu.h"

using namespace torch::indexing;

namespace attention_kernels {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> s2_attention_bwd_cpu(torch::Tensor kx, torch::Tensor vx, torch::Tensor qy, torch::Tensor dy,
                                                      torch::Tensor quad_weights, torch::Tensor col_idx, torch::Tensor row_off,
                                                      int64_t nlon_in, int64_t nlat_out, int64_t nlon_out) {

    // shapes: all channels LAST!
    // input
    // kx: B, Hi, Wi, C
    // vx: B, Hi, Wi, C
    // qy: B, Ho, Wo, C
    // quad_weights: Hi
    // output
    // dkx: B, Hi, Wi, C
    // dvx: B, Hi, Wi, C
    // dqy: B, Ho, Wo, C

    // sanity checks
    CHECK_CPU_INPUT_TENSOR(kx);
    CHECK_CPU_INPUT_TENSOR(vx);
    CHECK_CPU_INPUT_TENSOR(qy);
    CHECK_CPU_INPUT_TENSOR(dy);
    CHECK_CPU_INPUT_TENSOR(quad_weights);
    CHECK_CPU_INPUT_TENSOR(col_idx);
    CHECK_CPU_INPUT_TENSOR(row_off);

    auto dkx = torch::zeros_like(kx);
    auto dvx = torch::zeros_like(vx);
    auto dqy = torch::zeros_like(qy);

    // some parameters
    const int64_t batch_size = kx.size(0);
    const int64_t nchannels_out = vx.size(3);
    const int64_t nchannels_in = qy.size(3);

    // extract accessors
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

    s2_attn_bwd_kernel<float>(kx_arr, vx_arr, qy_arr, dy_arr,
        quad_weights_arr, col_idx_arr, roff_arr, dqy_arr, dvx_arr, dkx_arr,
        nlon_in, nlat_out, nlon_out,
        batch_size, nchannels_in, nchannels_out);

    return std::make_tuple(dkx, dvx, dqy);
}

TORCH_LIBRARY_IMPL(attention_kernels, CPU, m)
{
    m.impl("backward",  &s2_attention_bwd_cpu);
}

}