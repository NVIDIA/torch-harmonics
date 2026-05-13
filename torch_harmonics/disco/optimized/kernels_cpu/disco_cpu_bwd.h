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

    // Templated implementation. PSCALE == 0 means "use the runtime pscale
    // arg"; PSCALE > 0 selects a compile-time constant so the inner-loop
    // stride into sh[wo + pscale*wi] is known to the compiler (especially
    // PSCALE == 1 → contiguous SIMD store).
    template <typename scalar_t, int PSCALE>
    static void disco_bwd_cpu_impl(
        int64_t B, int64_t C, int64_t K, int64_t Hi, int64_t Wi,
        int64_t Ho, int64_t Wo, int64_t nnz, int64_t nnr,
        int64_t pscale_runtime,
        const torch::PackedTensorAccessor64<scalar_t, 5> inp,
        const torch::PackedTensorAccessor64<int64_t, 1> roff_idx,
        const torch::PackedTensorAccessor64<int64_t, 1> ker_idx,
        const torch::PackedTensorAccessor64<int64_t, 1> row_idx,
        const torch::PackedTensorAccessor64<int64_t, 1> col_idx,
        const torch::PackedTensorAccessor64<scalar_t, 1> vals,
        torch::PackedTensorAccessor64<scalar_t, 4> out) {

        // When PSCALE != 0 the compiler sees `pscale` as a compile-time
        // constant and folds the multiply / picks a contiguous store.
        const int64_t pscale = (PSCALE != 0) ? static_cast<int64_t>(PSCALE) : pscale_runtime;

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

        // 2D-lane layout: sh is conceptually sh[pscale][2*Wi]. The 1D buffer
        // is split into `pscale` contiguous lanes of length 2*Wi each.
        // Mapping: 1D position `p` (where 0 <= p < 2*Wo) maps to
        //   (lane, idx) = (p % pscale, p / pscale).
        // Within a single nz, all writes go to a SINGLE lane:
        //   wo + pscale*wi has (wo + pscale*wi) % pscale = wo % pscale (fixed)
        //   and (wo + pscale*wi) / pscale = wo/pscale + wi  (stride 1 in wi).
        // So the inner-loop write becomes stride-1 within the chosen lane,
        // regardless of pscale. Mirrors disco_cuda_bwd.cu's __sh[w_mod_ps][...].
        const int64_t lane_size = 2 * Wi;

        // parallel over (b, c) only — rows cannot be collapsed because
        // different rows write to the same (ho, wopp) positions (race).
        #pragma omp parallel
        {
            // per-thread sh: pscale lanes of 2*Wi each (total 2*Wo).
            std::vector<scalar_t> sh(2 * Wo, scalar_t(0));
            scalar_t* __restrict__ sh_ptr = sh.data();

            #pragma omp for collapse(2)
            for (int64_t b = 0; b < B; b++) {
                for (int64_t c = 0; c < C; c++) {

                    scalar_t* __restrict__ out_bc = out_base + b * out_sB + c * out_sC;

                    // ho_prev persists across rows within this (b, c).
                    int64_t ho_prev = -1;

                    for (int64_t row = 0; row < nnr; row++) {

                        const int64_t soff = roff_p[row];
                        const int64_t eoff = roff_p[row + 1];
                        const int64_t hi   = row_p[soff];
                        const int64_t ker  = ker_p[soff];

                        const scalar_t* __restrict__ inp_row =
                            inp_base + b * inp_sB + c * inp_sC + ker * inp_sK + hi * inp_sH;

                        for (int64_t z = soff; z < eoff; z++) {

                            const int64_t col = col_p[z];
                            const scalar_t val = val_p[z];

                            const int64_t wo = col % Wo;
                            const int64_t ho = col / Wo;

                            if (ho != ho_prev) {
                                if (ho_prev != -1) {
                                    // flush sh into out[ho_prev] and zero sh.
                                    // For each idx in [0, Wi) and lane in [0, pscale),
                                    //   out[idx*pscale + lane] += sh[lane][idx] + sh[lane][Wi+idx]
                                    // Inner lane loop unrolls when PSCALE is a
                                    // compile-time literal (1/2/3), giving
                                    // `pscale` contiguous out writes per idx.
                                    scalar_t* __restrict__ flush_row = out_bc + ho_prev * out_sH;
                                    #pragma omp simd
                                    for (int64_t idx = 0; idx < Wi; idx++) {
                                        for (int64_t lane = 0; lane < pscale; lane++) {
                                            const int64_t off = lane * lane_size;
                                            flush_row[idx * pscale + lane] +=
                                                sh_ptr[off + idx] + sh_ptr[off + Wi + idx];
                                            sh_ptr[off + idx]      = scalar_t(0);
                                            sh_ptr[off + Wi + idx] = scalar_t(0);
                                        }
                                    }
                                }
                                ho_prev = ho;
                            }

                            // stride-1 write into the (wo % pscale) lane,
                            // starting at index (wo / pscale).
                            const int64_t wo_mod_ps = wo % pscale;
                            const int64_t wo_div_ps = wo / pscale;
                            scalar_t* __restrict__ sh_lane =
                                sh_ptr + wo_mod_ps * lane_size + wo_div_ps;
                            #pragma omp simd
                            for (int64_t wi = 0; wi < Wi; wi++) {
                                sh_lane[wi] += val * inp_row[wi];
                            }
                        }
                    }

                    // final flush for this (b, c).
                    if (ho_prev != -1) {
                        scalar_t* __restrict__ flush_row = out_bc + ho_prev * out_sH;
                        #pragma omp simd
                        for (int64_t idx = 0; idx < Wi; idx++) {
                            for (int64_t lane = 0; lane < pscale; lane++) {
                                const int64_t off = lane * lane_size;
                                flush_row[idx * pscale + lane] +=
                                    sh_ptr[off + idx] + sh_ptr[off + Wi + idx];
                                sh_ptr[off + idx]      = scalar_t(0);
                                sh_ptr[off + Wi + idx] = scalar_t(0);
                            }
                        }
                    }
                }
            }
        }
    }

    // Dispatcher: pick a PSCALE-specialized instantiation for the common
    // values (1, 2, 3) and fall back to a runtime-pscale generic for everything
    // else. Mirrors disco_cuda_bwd.cu:179-195.
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

        switch (pscale) {
        case 1:
            disco_bwd_cpu_impl<scalar_t, 1>(B, C, K, Hi, Wi, Ho, Wo, nnz, nnr,
                                            pscale, inp, roff_idx, ker_idx, row_idx, col_idx, vals, out);
            break;
        case 2:
            disco_bwd_cpu_impl<scalar_t, 2>(B, C, K, Hi, Wi, Ho, Wo, nnz, nnr,
                                            pscale, inp, roff_idx, ker_idx, row_idx, col_idx, vals, out);
            break;
        case 3:
            disco_bwd_cpu_impl<scalar_t, 3>(B, C, K, Hi, Wi, Ho, Wo, nnz, nnr,
                                            pscale, inp, roff_idx, ker_idx, row_idx, col_idx, vals, out);
            break;
        default:
            disco_bwd_cpu_impl<scalar_t, 0>(B, C, K, Hi, Wi, Ho, Wo, nnz, nnr,
                                            pscale, inp, roff_idx, ker_idx, row_idx, col_idx, vals, out);
            break;
        }
    }

}
