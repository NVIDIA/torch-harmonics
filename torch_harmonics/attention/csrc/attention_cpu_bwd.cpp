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

    // change to channels first:
    bool kx_is_channels_last = kx.strides()[1] == 1;
    bool vx_is_channels_last = vx.strides()[1] == 1;
    bool qy_is_channels_last = qy.strides()[1] == 1;
    bool dy_is_channels_last = dy.strides()[1] == 1;

    if (!kx_is_channels_last) { kx = kx.contiguous(at::MemoryFormat::ChannelsLast); }
    if (!vx_is_channels_last) { vx = vx.contiguous(at::MemoryFormat::ChannelsLast); }
    if (!qy_is_channels_last) { qy = qy.contiguous(at::MemoryFormat::ChannelsLast); }
    if (!dy_is_channels_last) { dy = dy.contiguous(at::MemoryFormat::ChannelsLast); }

    auto dkx = torch::zeros_like(kx);
    auto dvx = torch::zeros_like(vx);
    auto dqy = torch::zeros_like(qy);

    // some parameters
    const int64_t batch_size = kx.size(0);
    const int64_t nchannels_out = vx.size(1);
    const int64_t nchannels_in = qy.size(1);

    for (int64_t b = 0; b < batch_size; b++) {
        for (int64_t co = 0; co < nchannels_out; co++) {
            for (int64_t ho = 0; ho < nlat_out; ho++) {

                // get number of nonzeros
                int64_t zstart = row_off.index({ho}).item<int64_t>();
                int64_t zend = row_off.index({ho+1}).item<int64_t>();

                for (int64_t wo = 0; wo < nlon_out; wo++) {

                    // required for all grads
                    auto qdotk_nz = torch::zeros({zend-zstart}, dy.options());
                    float qdotk_max = -std::numeric_limits<float>::max();
                    auto alpha_nz = torch::zeros({zend-zstart}, dy.options());
                    float alpha_sum = 0.0;

                    // required for dkx
                    float alpha_gdotv = 0.0;
                    //auto qy_alpha_gdotv = torch::zeros({dy.size(0), dy.size(1)}, dy.options());

                    // required for dqy
                    float alpha_k = 0.0;
                    float alpha_k_gdotv = 0.0;

                    for (int64_t idz = zstart; idz < zend; idz++) {
                        int64_t nz_col_idx = col_idx.index({idz}).item<int64_t>();

                        // compute input indices from psi datastructure
                        int64_t hi = static_cast<int64_t>(nz_col_idx / nlon_in);
                        // account for output shift and ensure positive index due to circular condition
                        int64_t wi = nz_col_idx % nlon_in;
                        int64_t wip = (wi+wo) % nlon_in;

                        // compute correlation & softmax numerator
                        auto q_ho_wo = qy.index({b, Slice(), ho, wo});
                        auto k_hi_wi = kx.index({b, Slice(), hi, wip});
                        qdotk_nz.index({idz-zstart}) = torch::sum(q_ho_wo * k_hi_wi, 0);

                        // tmp max and discount
                        float qdotk_max_tmp = std::max(qdotk_max, qdotk_nz.index({idz-zstart}).item<float>());
                        float discount = std::exp(qdotk_max - qdotk_max_tmp);

                        // alpha update
                        alpha_nz.index({idz-zstart}) = std::exp(qdotk_nz.index({idz-zstart}).item<float>() - qdotk_max_tmp) * quad_weights.index({hi}).item<float>();
                        alpha_sum = alpha_nz.index({idz-zstart}).item<float>() + alpha_sum * discount;

                        // dkx: input dot
                        float gdotv = torch::sum(dy.index({b, Slice(), ho, wo}) * vx.index({b, Slice(), hi, wip}), 0).item<float>();
                        float alpha_gdotv_tmp = alpha_nz.index({idz-zstart}).item<float>() * gdotv;
                        alpha_gdotv = alpha_gdotv_tmp + alpha_gdotv * discount;

                        // dqy: alpha_k
                        alpha_k = alpha_nz.index({idz-zstart}).item<float>() * k_hi_wi.index({co}).item<float>() + alpha_k * discount;

                        // dqy: alpha_k_gdotv
                        alpha_k_gdotv = alpha_gdotv_tmp * k_hi_wi.index({co}).item<float>() + alpha_k_gdotv * discount;

                        // define new max
                        qdotk_max = qdotk_max_tmp;
                    }

                    // normalization
                    alpha_gdotv = alpha_gdotv / alpha_sum;
                    alpha_k = alpha_k / alpha_sum;
                    alpha_k_gdotv = alpha_k_gdotv / alpha_sum;

                    // dqy: update
                    dqy.index({b, co, ho, wo}) = (alpha_k_gdotv - alpha_gdotv * alpha_k);

                    for (int64_t idz = zstart; idz < zend; idz++) {
                        int64_t nz_col_idx = col_idx.index({idz}).item<int64_t>();

                        // compute input indices from psi datastructure
                        int64_t hi = static_cast<int64_t>(nz_col_idx / nlon_in);
                        // account for output shift and ensure positive index due to circular condition
                        int64_t wi = nz_col_idx % nlon_in;
                        int64_t wip = (wi+wo) % nlon_in;

                        // recompute alpha
                        float alpha_norm = std::exp(qdotk_nz.index({idz-zstart}).item<float>() - qdotk_max) * quad_weights.index({hi}).item<float>() / alpha_sum;
                        dvx.index({b, co, hi, wip}) += alpha_norm * dy.index({b, co, ho, wo});

                        // dkx: update
                        float gdotv = torch::sum(dy.index({b, Slice(), ho, wo}) * vx.index({b, Slice(), hi, wip}), 0).item<float>();
                        dkx.index({b, co, hi, wip}) += qy.index({b, co, ho, wo}) * alpha_norm * (gdotv - alpha_gdotv);
                    }
                }
            }
        }
    }

    // permute back
    if (!qy_is_channels_last) { dqy = dqy.contiguous(at::MemoryFormat::Contiguous); }
    if (!vx_is_channels_last) { dvx = dvx.contiguous(at::MemoryFormat::Contiguous); }
    if (!kx_is_channels_last) { dkx = dkx.contiguous(at::MemoryFormat::Contiguous); }

    return std::make_tuple(dkx, dvx, dqy);
}

TORCH_LIBRARY_IMPL(attention_kernels, CPU, m)
{
    m.impl("backward",  &s2_attention_bwd_dkvq_cpu);
}

}