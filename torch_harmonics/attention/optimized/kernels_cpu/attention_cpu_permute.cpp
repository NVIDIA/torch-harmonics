// coding=utf-8

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
//

#include "../attention.h"

namespace attention_kernels
{

    // CPU side of the permute_to_nhwc / permute_to_nchw ops. There is no
    // hand-written CPU transpose: ATen's copy is what the CPU attention kernels
    // used before this was factored out, and the CPU path is not the one under
    // performance pressure. What matters is that the op exists on CPU too, so
    // the module, the CPU kernels and the pure-torch reference all convert
    // layout through the same operator instead of open-coding permutes.
    //
    // dtype is handled by ATen (fp32 / fp16 / bf16 alike).

    at::Tensor permute_to_nhwc_cpu(at::Tensor x)
    {
        CHECK_CPU_TENSOR(x);
        TORCH_CHECK(x.dim() == 4, "permute_to_nhwc expects a 4D (B, C, H, W) tensor, got ", x.dim(), "D");
        TORCH_CHECK(x.is_contiguous(), "permute_to_nhwc expects a contiguous (B, C, H, W) tensor");
        return x.permute({0, 2, 3, 1}).contiguous();
    }

    at::Tensor permute_to_nchw_cpu(at::Tensor x)
    {
        CHECK_CPU_TENSOR(x);
        TORCH_CHECK(x.dim() == 4, "permute_to_nchw expects a 4D (B, H, W, C) tensor, got ", x.dim(), "D");
        TORCH_CHECK(x.is_contiguous(), "permute_to_nchw expects a contiguous (B, H, W, C) tensor");
        return x.permute({0, 3, 1, 2}).contiguous();
    }

    TORCH_LIBRARY_IMPL(attention_kernels, CPU, m)
    {
        m.impl("permute_to_nhwc", &permute_to_nhwc_cpu);
        m.impl("permute_to_nchw", &permute_to_nchw_cpu);
    }

} // namespace attention_kernels
