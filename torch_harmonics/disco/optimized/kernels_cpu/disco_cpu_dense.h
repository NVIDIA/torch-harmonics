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

#include "../disco.h"

namespace disco_kernels {

    // Dense-packed-psi forward (CPU baseline). One iteration per (b, c, k, ho, wo);
    // the inner contraction loops over the per-(k, ho) padded neighbor list with
    //   (hi, wi_base) = pack_idx[k, ho, nz]
    //   v            = pack_val[k, ho, nz]
    //   wi           = (wi_base + pscale * wo) mod Wi
    // and accumulates v * inp[b, c, hi, wi]. cnt = pack_count[k, ho] bounds the
    // real-entry portion of the padded slot list; padding entries have val == 0
    // and are simply skipped via the cnt bound.
    template <typename scalar_t>
    static void disco_fwd_dense(
        int64_t B, int64_t C, int64_t K, int64_t Hi, int64_t Wi,
        int64_t Ho, int64_t Wo, int64_t NBR_PAD,
        const torch::PackedTensorAccessor64<scalar_t, 4> inp,
        const torch::PackedTensorAccessor64<int64_t, 4> pack_idx,
        const torch::PackedTensorAccessor64<scalar_t, 3> pack_val,
        const torch::PackedTensorAccessor64<int64_t, 2> pack_count,
        torch::PackedTensorAccessor64<scalar_t, 5> out) {

        const int64_t pscale = Wi / Wo;

        #pragma omp parallel for collapse(4)
        for (int64_t b = 0; b < B; b++) {
            for (int64_t c = 0; c < C; c++) {
                for (int64_t k = 0; k < K; k++) {
                    for (int64_t ho = 0; ho < Ho; ho++) {
                        const int64_t cnt = pack_count[k][ho];
                        for (int64_t wo = 0; wo < Wo; wo++) {
                            scalar_t acc = static_cast<scalar_t>(0);
                            for (int64_t nz = 0; nz < cnt; nz++) {
                                const int64_t hi      = pack_idx[k][ho][nz][0];
                                const int64_t wi_base = pack_idx[k][ho][nz][1];
                                const scalar_t v      = pack_val[k][ho][nz];
                                int64_t wi = wi_base + pscale * wo;
                                if (wi >= Wi) wi -= Wi;
                                acc += v * inp[b][c][hi][wi];
                            }
                            out[b][c][k][ho][wo] = acc;
                        }
                    }
                }
            }
        }
    }

}
