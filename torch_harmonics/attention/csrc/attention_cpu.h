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

#pragma once

#include "attention.h"
#include <array>
#include <vector>

#define CACHE_BLOCK_SIZE (64)

namespace attention_kernels {

    template <typename scalar_t>
    void s2_attn_fwd_kernel(
        const torch::PackedTensorAccessor64<scalar_t, 4> kx_arr, 
        const torch::PackedTensorAccessor64<scalar_t, 4> vx_arr, 
        const torch::PackedTensorAccessor64<scalar_t, 4> qy_arr, 
        const torch::PackedTensorAccessor64<scalar_t, 1> quad_weights_arr,
        const torch::PackedTensorAccessor64<int64_t, 1> col_idx_arr, 
        const torch::PackedTensorAccessor64<int64_t, 1> roff_arr, 
        torch::PackedTensorAccessor64<scalar_t, 4> y_arr,
        const int64_t nlon_in, const int64_t nlat_out, const int64_t nlon_out,
        const int64_t batch_size, const int64_t nchannels_in, const int64_t nchannels_out) {

        // some parameters
        const int64_t block_wo = CACHE_BLOCK_SIZE;
        const int64_t nblock_wo = static_cast<int64_t>((nlon_out + block_wo - 1) / block_wo);

        #pragma omp parallel for collapse(4)
        for (int64_t b = 0; b < batch_size; b++) {
            for (int64_t co = 0; co < nchannels_out; co++) {
                for (int64_t ho = 0; ho < nlat_out; ho++) {
                    for (int64_t bwo = 0; bwo < nblock_wo; bwo++) {
    
                        // compute block start and end
                        int64_t wo_start = bwo * block_wo;
                        int64_t wo_end = std::min(nlon_out, wo_start + block_wo);
    
                        // get number of nonzeros
                        int64_t zstart = roff_arr[ho];
                        int64_t zend = roff_arr[ho+1];
    
                        // init temp aray to zero
                        std::array<float, block_wo> alpha_sum;
                        std::array<float, block_wo> qdotk_max;
                        std::array<float, block_wo> y_tmp;
                        for (int64_t wob = 0; wob < block_wo; wob++) {
                            alpha_sum[wob] = 0.0;
                            qdotk_max[wob] = -std::numeric_limits<float>::max();
                            y_tmp[wob] = 0.0;
                        }
    
                        // loop over nonzeros
                        for (int64_t idz = zstart; idz < zend; idz++) {
                            // get column index
                            int64_t nz_col_idx = col_idx_arr[idz];
    
                            // compute input indices from psi datastructure
                            int64_t hi = static_cast<int64_t>(nz_col_idx / nlon_in);
                            // account for output shift and ensure positive index due to circular condition
                            int64_t wi = nz_col_idx % nlon_in;
                                
                            // loop over wo block
                            for (int64_t wo = wo_start; wo < wo_end; wo++) {
                                int64_t wip = (wi + wo) % nlon_in;
    
                                float qdotk = 0.0;
                                //#pragma omp simd reduction(+:qdotk)
                                for (int64_t ci = 0; ci < nchannels_in; ci++) {
                                    qdotk += static_cast<float>(qy_arr[b][ci][ho][wo] * kx_arr[b][ci][hi][wip]);
                                }
    
                                // update tmp max
                                float qdotk_max_tmp = std::max(qdotk_max[wo-wo_start], qdotk);
    
                                // alpha sum update
                                float alpha = std::exp(qdotk - qdotk_max_tmp) * static_cast<float>(quad_weights_arr[hi]);
                                alpha_sum[wo-wo_start] = alpha + alpha_sum[wo-wo_start] * std::exp(qdotk_max[wo-wo_start] - qdotk_max_tmp);
    
                                // update output
                                y_tmp[wo-wo_start] = y_tmp[wo-wo_start] * std::exp(qdotk_max[wo-wo_start] - qdotk_max_tmp) + alpha * static_cast<float>(vx_arr[b][co][hi][wip]);
    
                                // define new max
                                qdotk_max[wo-wo_start] = qdotk_max_tmp;
                            }
                        }
    
                        // update output
                        for (int64_t wo = wo_start; wo < wo_end; wo++) {
                            y_arr[b][co][ho][wo] = static_cast<scalar_t>(y_tmp[wo-wo_start] / alpha_sum[wo-wo_start]);
                        }
                    }
                }
            }
        }
    }

    template <typename scalar_t>
    void s2_attn_bwd_kernel(
        const torch::PackedTensorAccessor64<scalar_t, 4> kx_arr, 
        const torch::PackedTensorAccessor64<scalar_t, 4> vx_arr, 
        const torch::PackedTensorAccessor64<scalar_t, 4> qy_arr, 
        const torch::PackedTensorAccessor64<scalar_t, 4> dy_arr,
        const torch::PackedTensorAccessor64<scalar_t, 1> quad_weights_arr,
        const torch::PackedTensorAccessor64<int64_t, 1> col_idx_arr, 
        const torch::PackedTensorAccessor64<int64_t, 1> roff_arr, 
        torch::PackedTensorAccessor64<scalar_t, 4> dqy_arr,
        torch::PackedTensorAccessor64<scalar_t, 4> dvx_arr,
        torch::PackedTensorAccessor64<scalar_t, 4> dkx_arr,
        const int64_t nlon_in, const int64_t nlat_out, const int64_t nlon_out,
        const int64_t batch_size, const int64_t nchannels_in, const int64_t nchannels_out) {

        // compute dqy and dkx
        #pragma omp parallel for collapse(2)
        for (int64_t b = 0; b < batch_size; b++) {
            for (int64_t ci = 0; ci < nchannels_in; ci++) {

                for (int64_t ho = 0; ho < nlat_out; ho++) {
        
                    // get number of nonzeros
                    int64_t zstart = roff_arr[ho];
                    int64_t zend = roff_arr[ho+1];
        
                    for (int64_t wo = 0; wo < nlon_out; wo++) {
        
                        // required for all grads
                        std::vector<float> qdotk_nz(zend-zstart);
                        float qdotk_max = -std::numeric_limits<float>::max();
                        std::vector<float> alpha_nz(zend-zstart);
                        float alpha_sum = 0.0;
        
                        // required for dkx
                        float alpha_gdotv = 0.0;
        
                        // required for dqy
                        float alpha_k = 0.0;
                        float alpha_k_gdotv = 0.0;
        
                        for (int64_t idz = zstart; idz < zend; idz++) {
                            int64_t nz_col_idx = col_idx_arr[idz];
        
                            // compute input indices from psi datastructure
                            int64_t hi = static_cast<int64_t>(nz_col_idx / nlon_in);
                            // account for output shift and ensure positive index due to circular condition
                            int64_t wi = nz_col_idx % nlon_in;
                            int64_t wip = (wi+wo) % nlon_in;
        
                            // compute correlation & softmax numerator
                            qdotk_nz[idz-zstart] = 0.0;
                            for (int64_t cit = 0; cit < nchannels_in; cit++) {
                                qdotk_nz[idz-zstart] += qy_arr[b][cit][ho][wo] * kx_arr[b][cit][hi][wip];
                            }
        
                            // tmp max and discount
                            float qdotk_max_tmp = std::max(qdotk_max, qdotk_nz[idz-zstart]);
                            float discount = std::exp(qdotk_max - qdotk_max_tmp);
        
                            // alpha update
                            alpha_nz[idz-zstart] = std::exp(qdotk_nz[idz-zstart] - qdotk_max_tmp) * quad_weights_arr[hi];
                            alpha_sum = alpha_nz[idz-zstart] + alpha_sum * discount;
        
                            // dkx: input dot
                            float gdotv = 0.0;
                            for (int64_t cot = 0; cot < nchannels_out; cot++) {
                                gdotv += dy_arr[b][cot][ho][wo] * vx_arr[b][cot][hi][wip];
                            }
                            float alpha_gdotv_tmp = alpha_nz[idz-zstart] * gdotv;
                            alpha_gdotv = alpha_gdotv_tmp + alpha_gdotv * discount;
        
                            // dqy: alpha_k
                            alpha_k = alpha_nz[idz-zstart] * kx_arr[b][ci][hi][wip] + alpha_k * discount;
        
                            // dqy: alpha_k_gdotv
                            alpha_k_gdotv = alpha_gdotv_tmp * kx_arr[b][ci][hi][wip] + alpha_k_gdotv * discount;
        
                            // define new max
                            qdotk_max = qdotk_max_tmp;
                        }
        
                        // normalization
                        alpha_gdotv = alpha_gdotv / alpha_sum;
                        alpha_k = alpha_k / alpha_sum;
                        alpha_k_gdotv = alpha_k_gdotv / alpha_sum;
        
                        // dqy: update
                        dqy_arr[b][ci][ho][wo] = (alpha_k_gdotv - alpha_gdotv * alpha_k);
        
                        for (int64_t idz = zstart; idz < zend; idz++) {
                            int64_t nz_col_idx = col_idx_arr[idz];
        
                            // compute input indices from psi datastructure
                            int64_t hi = static_cast<int64_t>(nz_col_idx / nlon_in);
                            // account for output shift and ensure positive index due to circular condition
                            int64_t wi = nz_col_idx % nlon_in;
                            int64_t wip = (wi+wo) % nlon_in;
        
                            // dkx: alpha normalization
                            float alpha_norm = std::exp(qdotk_nz[idz-zstart] - qdotk_max) * quad_weights_arr[hi] / alpha_sum;

                            // dkx: input dot
                            float gdotv = 0.0;
                            for (int64_t cot = 0; cot < nchannels_out; cot++) {
                                gdotv += dy_arr[b][cot][ho][wo] * vx_arr[b][cot][hi][wip];
                            }

                            // dkx: update
                            dkx_arr[b][ci][hi][wip] += qy_arr[b][ci][ho][wo] * alpha_norm * (gdotv - alpha_gdotv);
                        }
                    }
                }
            }
        }

        // compute dvx
        #pragma omp parallel for collapse(2)
        for (int64_t b = 0; b < batch_size; b++) {
            for (int64_t co = 0; co < nchannels_out; co++) {

                for (int64_t ho = 0; ho < nlat_out; ho++) {
        
                    // get number of nonzeros
                    int64_t zstart = roff_arr[ho];
                    int64_t zend = roff_arr[ho+1];
        
                    for (int64_t wo = 0; wo < nlon_out; wo++) {
        
                        // required for all grads
                        std::vector<float> qdotk_nz(zend-zstart);
                        float qdotk_max = -std::numeric_limits<float>::max();
                        std::vector<float> alpha_nz(zend-zstart);
                        float alpha_sum = 0.0;
        
                        for (int64_t idz = zstart; idz < zend; idz++) {
                            int64_t nz_col_idx = col_idx_arr[idz];
        
                            // compute input indices from psi datastructure
                            int64_t hi = static_cast<int64_t>(nz_col_idx / nlon_in);
                            // account for output shift and ensure positive index due to circular condition
                            int64_t wi = nz_col_idx % nlon_in;
                            int64_t wip = (wi+wo) % nlon_in;
        
                            // compute correlation & softmax numerator
                            qdotk_nz[idz-zstart] = 0.0;
                            for (int64_t ci = 0; ci < nchannels_in; ci++) {
                                qdotk_nz[idz-zstart] += qy_arr[b][ci][ho][wo] * kx_arr[b][ci][hi][wip];
                            }
        
                            // tmp max and discount
                            float qdotk_max_tmp = std::max(qdotk_max, qdotk_nz[idz-zstart]);
                            float discount = std::exp(qdotk_max - qdotk_max_tmp);
        
                            // alpha update
                            alpha_nz[idz-zstart] = std::exp(qdotk_nz[idz-zstart] - qdotk_max_tmp) * quad_weights_arr[hi];
                            alpha_sum = alpha_nz[idz-zstart] + alpha_sum * discount;
        
                            // define new max
                            qdotk_max = qdotk_max_tmp;
                        }
        
                        for (int64_t idz = zstart; idz < zend; idz++) {
                            int64_t nz_col_idx = col_idx_arr[idz];
        
                            // compute input indices from psi datastructure
                            int64_t hi = static_cast<int64_t>(nz_col_idx / nlon_in);
                            // account for output shift and ensure positive index due to circular condition
                            int64_t wi = nz_col_idx % nlon_in;
                            int64_t wip = (wi+wo) % nlon_in;
        
                            // recompute alpha
                            float alpha_norm = std::exp(qdotk_nz[idz-zstart] - qdotk_max) * quad_weights_arr[hi] / alpha_sum;
                            dvx_arr[b][co][hi][wip] += alpha_norm * dy_arr[b][co][ho][wo];
                        }
                    }
                }
            }
        }
    }


    torch::Tensor s2_attention_fwd_cpu(at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor quad_weights,
        at::Tensor col_idx, at::Tensor row_off, int64_t nlon_in, int64_t nlat_out, int64_t nlon_out);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> s2_attention_bwd_cpu(torch::Tensor kx, torch::Tensor vx, torch::Tensor qy, torch::Tensor dy, 
        torch::Tensor quad_weights, torch::Tensor col_idx, torch::Tensor row_off,
        int64_t nlon_in, int64_t nlat_out, int64_t nlon_out);

}