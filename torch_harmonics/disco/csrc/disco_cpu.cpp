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

#include "disco_cpu.h"

namespace disco_kernels {

    // cpu ops
    torch::Tensor disco_cpu_fwd(torch::Tensor inp, torch::Tensor roff_idx, torch::Tensor ker_idx, torch::Tensor row_idx,
        torch::Tensor col_idx, torch::Tensor vals, int64_t K, int64_t Ho, int64_t Wo) {
        
        // sanity checks
        CHECK_CPU_INPUT_TENSOR(inp);
        CHECK_CPU_INPUT_TENSOR(roff_idx);
        CHECK_CPU_INPUT_TENSOR(ker_idx);
        CHECK_CPU_INPUT_TENSOR(row_idx);
        CHECK_CPU_INPUT_TENSOR(col_idx);
        CHECK_CPU_INPUT_TENSOR(vals);

        // initialize output tensor
        auto out = torch::zeros({inp.size(0), inp.size(1), K, Ho, Wo}, inp.options());

        AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_forward_cpu", ([&] {
            disco_fwd_cpu<scalar_t>(
                inp.size(0), inp.size(1), K, inp.size(2), inp.size(3), 
                Ho, Wo, vals.size(0), roff_idx.size(0) - 1,
                inp.packed_accessor64<scalar_t, 4>(), 
                roff_idx.packed_accessor64<int64_t, 1>(), 
                ker_idx.packed_accessor64<int64_t, 1>(), 
                row_idx.packed_accessor64<int64_t, 1>(), 
                col_idx.packed_accessor64<int64_t, 1>(), 
                vals.packed_accessor64<scalar_t, 1>(), 
                out.packed_accessor64<scalar_t, 5>());
        }));

        return out;
    }

    torch::Tensor disco_cpu_bwd(torch::Tensor inp, torch::Tensor roff_idx, torch::Tensor ker_idx, torch::Tensor row_idx,
        torch::Tensor col_idx, torch::Tensor vals, int64_t K, int64_t Ho, int64_t Wo) {
        
        // sanity checks
        CHECK_CPU_INPUT_TENSOR(inp);
        CHECK_CPU_INPUT_TENSOR(roff_idx);
        CHECK_CPU_INPUT_TENSOR(ker_idx);
        CHECK_CPU_INPUT_TENSOR(row_idx);
        CHECK_CPU_INPUT_TENSOR(col_idx);
        CHECK_CPU_INPUT_TENSOR(vals);

        // initialize output tensor
        auto out = torch::zeros({inp.size(0), inp.size(1), Ho, Wo}, inp.options());

        AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_backward_cpu", ([&] {
            disco_bwd_cpu<scalar_t>(
                inp.size(0), inp.size(1), K, inp.size(3), 
                inp.size(4), Ho, Wo, vals.size(0), roff_idx.size(0) - 1,
                inp.packed_accessor64<scalar_t, 5>(), 
                roff_idx.packed_accessor64<int64_t, 1>(), 
                ker_idx.packed_accessor64<int64_t, 1>(), 
                row_idx.packed_accessor64<int64_t, 1>(), 
                col_idx.packed_accessor64<int64_t, 1>(), 
                vals.packed_accessor64<scalar_t, 1>(), 
                out.packed_accessor64<scalar_t, 4>());
        }));

        return out;
    }

    // Implement the operators: CPU
    TORCH_LIBRARY_IMPL(disco_kernels, CPU, m)
    {
        m.impl("forward",  &disco_cpu_fwd);
        m.impl("backward",  &disco_cpu_bwd);
    }

}
