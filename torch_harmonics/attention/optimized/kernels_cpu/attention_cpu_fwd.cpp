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

namespace attention_kernels {

    torch::Tensor s2_attention_fwd_cpu(at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor quad_weights,
                                       at::Tensor col_idx, at::Tensor row_off,
                                       int64_t nlon_in, int64_t nlat_out, int64_t nlon_out) {
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
        const bool upsample   = (nlon_out % nlon_in == 0);
        TORCH_CHECK(downsample || upsample,
                    "either nlon_in (", nlon_in, ") must be an integer multiple of nlon_out (", nlon_out,
                    "), or vice versa");

        // stride(1) == 1 ⇒ caller is already in BCHW-logical / BHWC-physical
        // (channels-last) layout; we match that on the output side.
        const bool qy_is_channels_last = qy.strides()[1] == 1;

        // Force inputs to physical (B, H, W, C) via explicit permute + contiguous.
        // Mirrors CUDA permute_4D_to0231; avoids relying on
        // MemoryFormat::ChannelsLast which is silently ignored on some PyTorch
        // builds (notably the C == 1 degenerate case). For already-channels-last
        // inputs the permute reinterprets strides and .contiguous() is a no-op.
        kx = kx.permute({0, 2, 3, 1}).contiguous();
        vx = vx.permute({0, 2, 3, 1}).contiguous();
        qy = qy.permute({0, 2, 3, 1}).contiguous();

        const int64_t batch_size    = kx.size(0);
        const int64_t nlat_in       = kx.size(1);
        const int64_t nchannels_in  = qy.size(3);
        const int64_t nchannels_out = vx.size(3);

        // y allocated as physical (B, H, W, C).
        auto y = torch::zeros({batch_size, nlat_out, nlon_out, nchannels_out}, qy.options());

        auto kx_arr = kx.packed_accessor64<float, 4>();
        auto vx_arr = vx.packed_accessor64<float, 4>();
        auto qy_arr = qy.packed_accessor64<float, 4>();
        auto y_arr  = y.packed_accessor64<float, 4>();
        auto quad_weights_arr = quad_weights.packed_accessor64<float, 1>();
        auto col_idx_arr      = col_idx.packed_accessor64<int64_t, 1>();
        auto roff_arr         = row_off.packed_accessor64<int64_t, 1>();

        if (downsample) {
            s2_attn_fwd_kernel<float>(kx_arr, vx_arr, qy_arr, quad_weights_arr, col_idx_arr, roff_arr, y_arr,
                nlon_in, nlat_out, nlon_out, batch_size, nchannels_in, nchannels_out);
        } else {
            s2_attn_fwd_upsample_dispatch(kx_arr, vx_arr, qy_arr, quad_weights_arr, col_idx_arr, roff_arr, y_arr,
                nlon_in, nlat_in, nlat_out, nlon_out, batch_size, nchannels_in, nchannels_out);
        }

        // Return logical (B, C, H, W). Permute view is the desired layout when
        // caller was channels-last; otherwise .contiguous() materializes default
        // BCHW.
        y = y.permute({0, 3, 1, 2});
        if (!qy_is_channels_last) { y = y.contiguous(); }
        return y;
    }

    // Implement the operators: CPU
    TORCH_LIBRARY_IMPL(attention_kernels, CPU, m)
    {
        m.impl("forward",  &s2_attention_fwd_cpu);
    }

}
