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

#include "disco.h"

#define CACHE_BLOCK_SIZE (64)

namespace disco_kernels {

    template <typename scalar_t>
    static void disco_fused_fwd_cpu(
        int64_t B, int64_t G, int64_t Cin, int64_t Cout, int64_t K, int64_t Hi, int64_t Wi, 
        int64_t Ho, int64_t Wo, int64_t nnz, int64_t nnr,
        const torch::PackedTensorAccessor64<scalar_t, 5> inp,
        const torch::PackedTensorAccessor64<scalar_t, 4> weight,
        const torch::PackedTensorAccessor64<int64_t, 1> roff_idx,
        const torch::PackedTensorAccessor64<int64_t, 1> ker_idx,
        const torch::PackedTensorAccessor64<int64_t, 1> row_idx,
        const torch::PackedTensorAccessor64<int64_t, 1> col_idx,
        const torch::PackedTensorAccessor64<scalar_t, 1> vals,
        torch::PackedTensorAccessor64<scalar_t, 5> out) {

        const int64_t pscale = static_cast<int64_t>(Wi / Wo);
        
        // some parameters
        const int64_t block_wo = CACHE_BLOCK_SIZE;
        const int64_t nblock_wo = static_cast<int64_t>((Wo + block_wo - 1) / block_wo);

        // loop over matrix entries
        #pragma omp parallel for collapse(3)
        for (int64_t b = 0; b < B; b++) {
            for (int64_t g = 0; g < G; g++) {
                for (int64_t co = 0; co < Cout; co++) {

                    // we cannot easily fuse that loop, since we sum over ker
                    // we want to avoid atomic reductions:
                    for (int64_t row = 0; row < nnr; row++) {

                        // since the rows are ordered accordingly, we can compute ho and ker in here
                        int64_t ho = row_idx[roff_idx[row]];
                        int64_t ker = ker_idx[roff_idx[row]];

                        for (int64_t bwo = 0; bwo < nblock_wo; bwo++) {

                            // compute block start and end
                            int64_t wo_start = bwo * block_wo;
                            int64_t wo_end = std::min(Wo, wo_start + block_wo);

                            std::array<scalar_t, block_wo> out_tmp;
                            for (int64_t wob = 0; wob < block_wo; wob++) {
                                out_tmp[wob] = scalar_t(0);
                            }
                        
                            // loop over input rows
                            for (int64_t z = roff_idx[row]; z < roff_idx[row + 1]; z++) {
                    
                                // COO format, we can optimize later
                                int64_t col = col_idx[z];
                                scalar_t val = vals[z];

                                int64_t wi = static_cast<int64_t>(col % Wi);
                                int64_t hi = static_cast<int64_t>(col / Wi);

                                // sum wo
                                for (int64_t wo = wo_start; wo < wo_end; wo++) {
                                    // compute shifted w
                                    int64_t wipp = static_cast<int64_t>((wi + pscale * wo) % Wi);

                                    // sum over ci
                                    for (int64_t ci = 0; ci < Cin; ci++) {
                                        out_tmp[wo-wo_start] += val * inp[b][g][ci][hi][wipp] * weight[g][co][ci][ker];
                                    }
                                }
                            }
                        }
                        // write out: we need to use += since we sum over ker
                        for (int64_t wo = wo_start; wo < wo_end; wo++) {
                            out[b][g][co][ho][wo] += out_tmp[wo-wo_start];
                        }
                    }
                }
            }
        }
    }

    template <typename scalar_t>
    static void disco_fused_bwd_cpu(
        int64_t B, int64_t G, int64_t Cin, int64_t Cout, int64_t K, int64_t Hi, int64_t Wi, 
        int64_t Ho, int64_t Wo, int64_t nnz, int64_t nnr,
        const torch::PackedTensorAccessor64<scalar_t, 5> inp,
        const torch::PackedTensorAccessor64<scalar_t, 5> ograd,
        const torch::PackedTensorAccessor64<scalar_t, 4> weight,
        const torch::PackedTensorAccessor64<int64_t, 1> roff_idx,
        const torch::PackedTensorAccessor64<int64_t, 1> ker_idx,
        const torch::PackedTensorAccessor64<int64_t, 1> row_idx,
        const torch::PackedTensorAccessor64<int64_t, 1> col_idx,
        const torch::PackedTensorAccessor64<scalar_t, 1> vals,
        torch::PackedTensorAccessor64<scalar_t, 5> igrad,
        torch::PackedTensorAccessor64<scalar_t, 5> wgrad) {

        const int64_t pscale = static_cast<int64_t>(Wo / Wi);

        // loop over matrix entries
        #pragma omp parallel for collapse(2)
        for (int64_t b = 0; b < B; b++) {
            for (int64_t g = 0; g < G; g++) {
                for (int64_t co = 0; co < Cout; co++) {

                    // we cannot simply collapse on this loop
                    // because we sum over ker, and row defines ker
                    for (int64_t row = 0; row < nnr; row++) {

                        // since the rows are ordered accordingly, we can compute ho and ker in here
                        int64_t hi = row_idx[roff_idx[row]];
                        int64_t ker = ker_idx[roff_idx[row]];
                        
                        // loop over input rows
                        for (int64_t z = roff_idx[row]; z < roff_idx[row + 1]; z++) {

                            // COO format, we can optimize later
                            int64_t col = col_idx[z];
                            scalar_t val = vals[z];

                            int64_t wi = static_cast<int64_t>(col % Wi);
                            int64_t ho = static_cast<int64_t>(col / Wi);

                            // compute shifted w
                            int64_t wopp = static_cast<int64_t>((wo + pscale * wi) % Wo);

                            // sum over ci
                            for (int64_t ci = 0; ci < Cin; ci++) {
                                igrad[b][g][co][ho][wopp] += val * ograd[b][g][ci][hi][wi] * weight[g][ci][co][ker];
                                // sum over b later:
                                wgrad[b][g][ci][co][ker] += val * inp[b][g][co][ho][wopp] * ograd[b][g][ci][hi][wi];
                            }
                        }
                    }
                }
            }
        }
    }

}
