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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> s2_attention_bwd_dkvq_cpu(torch::Tensor kx, torch::Tensor vx, torch::Tensor qy, torch::Tensor dy,
                                                      torch::Tensor quad_weights, torch::Tensor col_idx, torch::Tensor row_off,
                                                      int64_t nlon_in, int64_t nlat_out, int64_t nlon_out) {

    // shapes:
    // input
    // kx: B, C, Hi, Wi
    // vx: B, C, Hi, Wi
    // qy: B, C, Ho, Wo
    // quad_weights: Hi
    // output
    // dkx: B, C, Hi, Wi
    // dvx: B, C, Hi, Wi
    // dqy: B, C, Ho, Wo

    auto dkx = torch::zeros_like(kx);
    auto dvx = torch::zeros_like(vx);
    auto dqy = torch::zeros_like(qy);

    for (int64_t ho = 0; ho < nlat_out; ho++) {

        // get number of nonzeros
        int64_t zstart = row_off.index({ho}).item<int64_t>();
        int64_t zend = row_off.index({ho+1}).item<int64_t>();

        for (int64_t wo = 0; wo < nlon_out; wo++) {

            auto alpha_nz = torch::zeros({dy.size(0), zend-zstart}, dy.options());
            auto qdotk_nz = torch::zeros({dy.size(0), zend-zstart}, dy.options());
            auto alpha_k = torch::zeros({dy.size(0), dy.size(1)}, dy.options());
            auto alpha_gdotv = torch::zeros({dy.size(0)}, dy.options());
            auto alpha_k_gdotv = torch::zeros({dy.size(0), dy.size(1)}, dy.options());
            auto alpha_sum = torch::zeros({dy.size(0)}, dy.options());

            for (int64_t idz = zstart; idz < zend; idz++) {
                int64_t nz_col_idx = col_idx.index({idz}).item<int64_t>();

                // compute input indices from psi datastructure
                int64_t hi = static_cast<int64_t>(nz_col_idx / nlon_in);
                // account for output shift and ensure positive index due to circular condition
                int64_t wi = nz_col_idx % nlon_in;
                int64_t wip = (wi+wo) % nlon_in;

                // compute correlation & softmax numerator
                auto q_ho_wo = qy.index({Slice(0, -1), Slice(0, -1), ho, wo});
                auto k_hi_wi = kx.index({Slice(0, -1), Slice(0, -1), hi, wip});
                qdotk_nz.index({Slice(0, -1), idz-zstart}) = torch::sum(q_ho_wo * k_hi_wi, 1);
            }

            auto qdotk_max = std::get<0>(torch::max(qdotk_nz, 1));

            for (int64_t idz = zstart; idz < zend; idz++) {
                int64_t nz_col_idx = col_idx.index({idz}).item<int64_t>();

                // compute input indices from psi datastructure
                int64_t hi = static_cast<int64_t>(nz_col_idx / nlon_in);
                // account for output shift and ensure positive index due to circular condition
                int64_t wi = nz_col_idx % nlon_in;
                int64_t wip = (wi+wo) % nlon_in;
            
                // alpha update
                alpha_nz.index({Slice(0, -1), idz-zstart}) = torch::exp(qdotk_nz.index({Slice(0, -1), idz-zstart}) - qdotk_max) * quad_weights.index({hi});
                alpha_sum = alpha_nz + alpha_sum;

                // input dot
                auto gdotv = torch::sum(dy.index({Slice(0, -1), Slice(0, -1), ho, wo}) * vx.index({Slice(0, -1), Slice(0, -1), hi, wip}), 1);
                alpha_gdotv += alpha_nz.index({Slice(0, -1), idz-zstart}) * gdotv;

                // alpha_k
                auto k_hi_wi = kx.index({Slice(0, -1), Slice(0, -1), hi, wip});
                alpha_k += alpha_nz.unsqueeze(1).index({Slice(0, -1), Slice(0, -1), idz-zstart}) * k_hi_wi;
                // alpha_k_gdotv
                alpha_k_gdotv += alpha_gdotv.unsqueeze(1).index({Slice(0, -1), Slice(0, -1), idz-zstart}) * k_hi_wi;
            }

            alpha_gdotv = alpha_gdotv / alpha_sum;
            alpha_k = alpha_k / alpha_sum.unsqueeze(1);
            alpha_k_gdotv = alpha_k_gdotv / alpha_sum.unsqueeze(1);

            // dqy update
            dqy.index({Slice(0, -1), Slice(0, -1), ho, wo}) = (alpha_k_gdotv - alpha_gdotv.unsqueeze(1) * alpha_k);

            for (int64_t idz = zstart; idz < zend; idz++) {
                int64_t nz_col_idx = col_idx.index({idz}).item<int64_t>();

                // compute input indices from psi datastructure
                int64_t hi = static_cast<int64_t>(nz_col_idx / nlon_in);
                // account for output shift and ensure positive index due to circular condition
                int64_t wi = nz_col_idx % nlon_in;
                int64_t wip = (wi+wo) % nlon_in;

                // dvx update
                dvx.index({Slice(0, -1), Slice(0, -1), hi, wip}) += (alpha_nz.unsqueeze(1).index({Slice(0, -1), Slice(0, -1), idz-zstart}) / alpha_sum.unsqueeze(1)) * dy.index({Slice(0, -1), Slice(0, -1), ho, wo});

                // dkx update
                auto gdotv = torch::sum(dy.index({Slice(0, -1), Slice(0, -1), ho, wo}) * vx.index({Slice(0, -1), Slice(0, -1), hi, wip}), 1);
                dkx.index({Slice(0, -1), Slice(0, -1), hi, wip}) += qy.index({Slice(0, -1), Slice(0, -1), ho, wo}) * (alpha_nz.unsqueeze(1).index({Slice(0, -1), Slice(0, -1), idz-zstart}) / alpha_sum.unsqueeze(1)) * (gdotv - alpha_gdotv).unsqueeze(1);
            }
        }
    }
    return std::make_tuple(dkx, dvx, dqy);
}

TORCH_LIBRARY_IMPL(attention_kernels, CPU, m)
{
    m.impl("backward",  &s2_attention_bwd_dkvq_cpu);
}

}