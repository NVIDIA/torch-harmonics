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

#pragma once

#include "../attention.h"

namespace attention_kernels {

    // Upsample backward dispatch helper. Defined in
    // attention_cpu_bwd_upsample.cpp; called by s2_attention_bwd_cpu when
    // nlon_out is an integer multiple of nlon_in. Takes accessors over tensors
    // already in physical (B, H, W, C) layout (set up by the wrapper); the
    // template kernel stays file-local.
    void s2_attn_bwd_upsample_dispatch(
        const torch::PackedTensorAccessor64<float, 4> kx_arr,
        const torch::PackedTensorAccessor64<float, 4> vx_arr,
        const torch::PackedTensorAccessor64<float, 4> qy_arr,
        const torch::PackedTensorAccessor64<float, 4> dy_arr,
        const torch::PackedTensorAccessor64<float, 1> quad_weights_arr,
        const torch::PackedTensorAccessor64<int64_t, 1> col_idx_arr,
        const torch::PackedTensorAccessor64<int64_t, 1> roff_arr,
        torch::PackedTensorAccessor64<float, 4> dqy_arr,
        torch::PackedTensorAccessor64<float, 4> dvx_arr,
        torch::PackedTensorAccessor64<float, 4> dkx_arr,
        int64_t nlon_in, int64_t nlat_in,
        int64_t nlat_out, int64_t nlon_out,
        int64_t batch_size, int64_t nchannels_in, int64_t nchannels_out);

}
