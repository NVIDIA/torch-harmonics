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

#include "../disco.h"

#include <algorithm>
#include <vector>

namespace disco_kernels {

    template <typename scalar_t>
    static void disco_fwd_cpu(
        int64_t B, int64_t C, int64_t K, int64_t Hi, int64_t Wi,
        int64_t Ho, int64_t Wo, int64_t nnz, int64_t nnr,
        const torch::PackedTensorAccessor64<scalar_t, 4> inp,
        const torch::PackedTensorAccessor64<int64_t, 1> roff_idx,
        const torch::PackedTensorAccessor64<int64_t, 1> ker_idx,
        const torch::PackedTensorAccessor64<int64_t, 1> row_idx,
        const torch::PackedTensorAccessor64<int64_t, 1> col_idx,
        const torch::PackedTensorAccessor64<scalar_t, 1> vals,
        torch::PackedTensorAccessor64<scalar_t, 5> out) {

        const int64_t pscale = static_cast<int64_t>(Wi / Wo);

        // hoist base pointers + strides; raw pointer arithmetic in the hot path
        const scalar_t* __restrict__ inp_base = inp.data();
        scalar_t* __restrict__ out_base = out.data();
        const int64_t inp_sB = inp.stride(0);
        const int64_t inp_sC = inp.stride(1);
        const int64_t inp_sH = inp.stride(2);
        const int64_t out_sB = out.stride(0);
        const int64_t out_sC = out.stride(1);
        const int64_t out_sK = out.stride(2);
        const int64_t out_sH = out.stride(3);

        const int64_t* __restrict__ roff_p = roff_idx.data();
        const int64_t* __restrict__ ker_p  = ker_idx.data();
        const int64_t* __restrict__ row_p  = row_idx.data();
        const int64_t* __restrict__ col_p  = col_idx.data();
        const scalar_t* __restrict__ val_p = vals.data();

        #pragma omp parallel
        {
            // per-thread accumulator for one full output row, allocated once
            std::vector<scalar_t> out_tmp(Wo);
            // per-thread doubled row buffer; on hi-change we copy the input
            // row twice so reads at index (wi + pscale*wo) need no modulo.
            // Bound: wi < Wi and pscale*wo < pscale*Wo = Wi, so the access
            // index is always < 2*Wi.
            std::vector<scalar_t> sh(2 * Wi);
            scalar_t* __restrict__ sh_ptr = sh.data();

            #pragma omp for collapse(3)
            for (int64_t b = 0; b < B; b++) {
                for (int64_t c = 0; c < C; c++) {
                    for (int64_t row = 0; row < nnr; row++) {

                        const int64_t soff = roff_p[row];
                        const int64_t eoff = roff_p[row + 1];
                        const int64_t ho   = row_p[soff];
                        const int64_t ker  = ker_p[soff];

                        // per-(b,c) input plane base
                        const scalar_t* __restrict__ inp_bc =
                            inp_base + b * inp_sB + c * inp_sC;

                        std::fill(out_tmp.begin(), out_tmp.end(), scalar_t(0));

                        int64_t hi_prev = -1;

                        for (int64_t z = soff; z < eoff; z++) {

                            const int64_t col = col_p[z];
                            const scalar_t val = val_p[z];

                            const int64_t wi = col % Wi;
                            const int64_t hi = col / Wi;

                            // only refresh the doubled buffer when hi changes
                            if (hi != hi_prev) {
                                hi_prev = hi;
                                const scalar_t* inp_row = inp_bc + hi * inp_sH;
                                std::copy(inp_row, inp_row + Wi, sh_ptr);
                                std::copy(inp_row, inp_row + Wi, sh_ptr + Wi);
                            }

                            for (int64_t wo = 0; wo < Wo; wo++) {
                                out_tmp[wo] += val * sh_ptr[wi + pscale * wo];
                            }
                        }

                        // write out via raw output row pointer
                        scalar_t* __restrict__ out_row =
                            out_base + b * out_sB + c * out_sC + ker * out_sK + ho * out_sH;
                        for (int64_t wo = 0; wo < Wo; wo++) {
                            out_row[wo] = out_tmp[wo];
                        }
                    }
                }
            }
        }
    }

    template <typename scalar_t>
    static void disco_bwd_cpu(
        int64_t B, int64_t C, int64_t K, int64_t Hi, int64_t Wi, 
        int64_t Ho, int64_t Wo, int64_t nnz, int64_t nnr,
        const torch::PackedTensorAccessor64<scalar_t, 5> inp,
        const torch::PackedTensorAccessor64<int64_t, 1> roff_idx,
        const torch::PackedTensorAccessor64<int64_t, 1> ker_idx,
        const torch::PackedTensorAccessor64<int64_t, 1> row_idx,
        const torch::PackedTensorAccessor64<int64_t, 1> col_idx,
        const torch::PackedTensorAccessor64<scalar_t, 1> vals,
        torch::PackedTensorAccessor64<scalar_t, 4> out) {

        const int64_t pscale = static_cast<int64_t>(Wo / Wi);

        // loop over matrix entries
        #pragma omp parallel for collapse(2)
        for (int64_t b = 0; b < B; b++) {
            for (int64_t c = 0; c < C; c++) {

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

                        int64_t wo = static_cast<int64_t>(col % Wo);
                        int64_t ho = static_cast<int64_t>(col / Wo);

                        for (int64_t wi = 0; wi < Wi; wi++) {
                            // compute shifted w
                            int64_t wopp = static_cast<int64_t>((wo + pscale * wi) % Wo);
                            out[b][c][ho][wopp] += val * inp[b][c][ker][hi][wi];
                        }
                    }
                }
            }
        }
    }

}
