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

        // prepare result tensor
        auto y = torch::zeros_like(qy);

        for (int64_t ho = 0; ho < nlat_out; ho++) {
        
	        // get number of nonzeros
            int64_t zstart = psi_row_off.index({ho}).item<int64_t>();
            int64_t zend = psi_row_off.index({ho+1}).item<int64_t>();

            for (int64_t wo = 0; wo < nlon_out; wo++) {

                auto alpha_sum = torch::zeros({y.size(0)}, y.options());
                auto qdotk_max = torch::zeros({y.size(0)}, y.options());

                for (int64_t idz = zstart; idz < zend; idz++) {
                    int64_t nz_col_idx = psi_col_idx.index({idz}).item<int64_t>();

                    // compute input indices from psi datastructure
                    int64_t hi = static_cast<int64_t>(nz_col_idx / nlon_in);
                    // account for output shift and ensure positive index due to circular condition
                    int64_t wi = nz_col_idx % nlon_in;
                    int64_t wip = (wi + wo) % nlon_in;

                    // compute correlation & softmax numerator
                    auto q_ho_wo = qy.index({Slice(), Slice(), ho, wo});
                    auto k_hi_wip = kx.index({Slice(), Slice(), hi, wip});
                    auto qdotk = torch::sum(q_ho_wo * k_hi_wip, 1);

                    // tmp max
                    auto qdotk_max_tmp = torch::maximum(qdotk_max, qdotk);

                    // alpha sum update
                    auto alpha = torch::exp(qdotk - qdotk_max_tmp) * quad_weights.index({hi});
                    alpha_sum = alpha + alpha_sum * torch::exp(qdotk_max - qdotk_max_tmp);

                    // update output
                    y.index({Slice(), Slice(), ho, wo}) = y.index({Slice(), Slice(), ho, wo}) * torch::exp(qdotk_max - qdotk_max_tmp).unsqueeze(1) + alpha.unsqueeze(1) * vx.index({Slice(), Slice(), hi, wip});

                    // define new max
                    qdotk_max = qdotk_max_tmp;
                }
                y.index({Slice(), Slice(), ho, wo}) = y.index({Slice(), Slice(), ho, wo}) / alpha_sum.unsqueeze(1);
            }
        }
        return y;
    }

    // Implement the operators: CPU
    TORCH_LIBRARY_IMPL(attention_kernels, CPU, m)
    {
        m.impl("forward",  &s2_attention_fwd_cpu);
    }

}