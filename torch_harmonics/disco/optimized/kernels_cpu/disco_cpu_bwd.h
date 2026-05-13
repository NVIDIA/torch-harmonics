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

        // hoist base pointers + strides; raw pointer arithmetic in the hot path
        const scalar_t* __restrict__ inp_base = inp.data();
        scalar_t* __restrict__ out_base = out.data();
        const int64_t inp_sB = inp.stride(0);
        const int64_t inp_sC = inp.stride(1);
        const int64_t inp_sK = inp.stride(2);
        const int64_t inp_sH = inp.stride(3);
        const int64_t out_sB = out.stride(0);
        const int64_t out_sC = out.stride(1);
        const int64_t out_sH = out.stride(2);

        const int64_t* __restrict__ roff_p = roff_idx.data();
        const int64_t* __restrict__ ker_p  = ker_idx.data();
        const int64_t* __restrict__ row_p  = row_idx.data();
        const int64_t* __restrict__ col_p  = col_idx.data();
        const scalar_t* __restrict__ val_p = vals.data();

        // parallel over (b, c) only — rows cannot be collapsed because
        // different rows write to the same (ho, wopp) positions (race).
        #pragma omp parallel
        {
            // per-thread doubled output buffer; sh[wo + pscale*wi] accumulates
            // without modulo. Bound: wo < Wo and pscale*wi < pscale*Wi = Wo, so
            // the access index is always < 2*Wo. Flushed into out as
            // out[ho][i] += sh[i] + sh[Wo + i] when ho changes.
            std::vector<scalar_t> sh(2 * Wo, scalar_t(0));
            scalar_t* __restrict__ sh_ptr = sh.data();

            #pragma omp for collapse(2)
            for (int64_t b = 0; b < B; b++) {
                for (int64_t c = 0; c < C; c++) {

                    // (b, c, :, :) output plane base
                    scalar_t* __restrict__ out_bc = out_base + b * out_sB + c * out_sC;

                    // ho_prev persists across rows within this (b, c): when a
                    // new row starts with the same ho, we keep accumulating;
                    // otherwise we flush the previous ho's buffer.
                    int64_t ho_prev = -1;

                    for (int64_t row = 0; row < nnr; row++) {

                        const int64_t soff = roff_p[row];
                        const int64_t eoff = roff_p[row + 1];
                        const int64_t hi   = row_p[soff];
                        const int64_t ker  = ker_p[soff];

                        // input row pointer (b, c, ker, hi, :) — fixed for this CSR row
                        const scalar_t* __restrict__ inp_row =
                            inp_base + b * inp_sB + c * inp_sC + ker * inp_sK + hi * inp_sH;

                        for (int64_t z = soff; z < eoff; z++) {

                            const int64_t col = col_p[z];
                            const scalar_t val = val_p[z];

                            const int64_t wo = col % Wo;
                            const int64_t ho = col / Wo;

                            if (ho != ho_prev) {
                                if (ho_prev != -1) {
                                    // flush sh into out[ho_prev] and zero sh
                                    scalar_t* __restrict__ flush_row = out_bc + ho_prev * out_sH;
                                    for (int64_t i = 0; i < Wo; i++) {
                                        flush_row[i] += sh_ptr[i] + sh_ptr[Wo + i];
                                        sh_ptr[i]      = scalar_t(0);
                                        sh_ptr[Wo + i] = scalar_t(0);
                                    }
                                }
                                ho_prev = ho;
                            }

                            // accumulate without modulo: position wo + pscale*wi
                            // wraps into [Wo, 2*Wo) automatically; flush combines.
                            for (int64_t wi = 0; wi < Wi; wi++) {
                                sh_ptr[wo + pscale * wi] += val * inp_row[wi];
                            }
                        }
                    }

                    // final flush for this (b, c) — handles trailing ho_prev.
                    if (ho_prev != -1) {
                        scalar_t* __restrict__ flush_row = out_bc + ho_prev * out_sH;
                        for (int64_t i = 0; i < Wo; i++) {
                            flush_row[i] += sh_ptr[i] + sh_ptr[Wo + i];
                            sh_ptr[i]      = scalar_t(0);
                            sh_ptr[Wo + i] = scalar_t(0);
                        }
                    }
                }
            }
        }
    }

}
