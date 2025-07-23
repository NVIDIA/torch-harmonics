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

    torch::Tensor s2_attention_fwd_cpu(at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor quad_weights,
                                       at::Tensor psi_col_idx, at::Tensor psi_row_off, 
                                       int64_t nlon_in, int64_t nlat_out, int64_t nlon_out) {
        // sanity checks
        CHECK_CPU_INPUT_TENSOR(kx);
        CHECK_CPU_INPUT_TENSOR(vx);
        CHECK_CPU_INPUT_TENSOR(qy);
        CHECK_CPU_INPUT_TENSOR(quad_weights);
        CHECK_CPU_INPUT_TENSOR(psi_col_idx);
        CHECK_CPU_INPUT_TENSOR(psi_row_off);

        // change to channels first:
        bool kx_is_channels_last = kx.strides()[1] == 1;
        bool vx_is_channels_last = vx.strides()[1] == 1;
        bool qy_is_channels_last = qy.strides()[1] == 1;

        if (!kx_is_channels_last) { kx = kx.contiguous(at::MemoryFormat::ChannelsLast); }
        if (!vx_is_channels_last) { vx = vx.contiguous(at::MemoryFormat::ChannelsLast); }
        if (!qy_is_channels_last) { qy = qy.contiguous(at::MemoryFormat::ChannelsLast); }

        // prepare result tensor
        auto y = torch::zeros_like(qy);

        // some parameters
        const int64_t batch_size = kx.size(0);
        const int64_t nchannels_out = vx.size(1);
        const int64_t nchannels_in = qy.size(1);

        // extract accessors
        auto roff_arr = psi_row_off.packed_accessor64<int64_t, 1>();
        auto col_idx_arr = psi_col_idx.packed_accessor64<int64_t, 1>();
        auto quad_weights_arr = quad_weights.packed_accessor64<float, 1>();
        auto vx_arr = vx.packed_accessor64<float, 4>();
        auto qy_arr = qy.packed_accessor64<float, 4>();
        auto kx_arr = kx.packed_accessor64<float, 4>();
        auto y_arr = y.packed_accessor64<float, 4>();

        s2_attn_fwd_kernel<float>(kx_arr, vx_arr, qy_arr, quad_weights_arr, col_idx_arr, roff_arr, y_arr, 
            nlon_in, nlat_out, nlon_out, batch_size, nchannels_in, nchannels_out);

        // permute back
        if (!qy_is_channels_last) { y = y.contiguous(at::MemoryFormat::Contiguous); }

        return y;
    }

    // Implement the operators: CPU
    TORCH_LIBRARY_IMPL(attention_kernels, CPU, m)
    {
        m.impl("forward",  &s2_attention_fwd_cpu);
    }

}