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

#include "../attention.h"
#include <array>
#include <vector>

#define CACHE_BLOCK_SIZE (64)

namespace attention_kernels {

    // -----------------------------------------------------------------------
    // Self-attention / downsample (output-centric gather, canonical psi).
    //
    // K, V live on the input grid (nlat_in, nlon_in); Q lives on the output
    // grid (nlat_out, nlon_out). The neighborhood is encoded as a single
    // canonical psi at output longitude wo=0:
    //     row_off : indexed by ho,    length nlat_out + 1
    //     col_idx : hi * nlon_in + wi_canonical
    //               (input-lon offset for wo=0; the kernel applies the integer
    //                p-shift  wip = (wi + pscale * wo) mod nlon_in  internally,
    //                where pscale = nlon_in / nlon_out).
    // Requires nlon_in % nlon_out == 0.
    //
    // Each output (b, co, ho, wo) is independent → the kernel parallelizes
    // collapse(4) over (batch, out-channel, ho, wo-block). Within a block we
    // use online softmax with a small stack-allocated state of size block_wo.
    // No atomics needed since outputs are strictly partitioned across threads.
    // -----------------------------------------------------------------------

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

        // one output lon step corresponds to pscale input lon steps (requires nlon_in % nlon_out == 0)
        const int64_t pscale = nlon_in / nlon_out;

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
                                int64_t wip = (wi + pscale * wo) % nlon_in;

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


    // -----------------------------------------------------------------------
    // Upsample (scatter-style by intent, transposed psi).
    //
    // K, V live on the input (smaller) grid; Q lives on the output (larger)
    // grid. psi is built with rows indexed by hi and cols encoding
    // (ho, wo_canonical) on the output grid as ho * nlon_out + wo_canonical
    // (canonical at input longitude wi=0):
    //     row_off : indexed by hi,    length nlat_in + 1
    //     col_idx : ho * nlon_out + wo_canonical
    // For wi > 0 the actual output column is
    //     wo = (wo_canonical + pscale_out * wi) mod nlon_out,
    //     pscale_out = nlon_out / nlon_in.
    // Requires nlon_out % nlon_in == 0.
    //
    // Algorithm: classical 2-pass softmax (matches the torch reference,
    // _neighborhood_s2_attention_upsample_fwd_torch):
    //   pass 1: for every (input, output-neighbor) pair, find qdotk_max[ho, wo]
    //   pass 2: with max fixed, accumulate alpha_sum[ho, wo] and y_acc[ho, wo]
    //   finalize: y = y_acc / alpha_sum
    //
    // To keep parallelization safe without atomics we invert the iteration so
    // each thread owns a disjoint output block (collapse(4) over
    // (b, co, ho, wo-block)). For each output cell we scan all psi[hi] rows
    // and skip entries whose stored ho_neigh doesn't match — a stored
    // (hi, wo_canonical) entry contributes to output wo iff
    //   wo ≡ wo_canonical (mod pscale_out),
    // and the contributing wi is wi = (wo - wo_canonical) / pscale_out within
    // [0, nlon_in). The redundant scan over non-matching hi rows is the price
    // we pay to avoid scatter atomics; this is the correctness reference path,
    // not the perf path.
    // -----------------------------------------------------------------------

    template <typename scalar_t>
    void s2_attn_fwd_upsample_kernel(
        const torch::PackedTensorAccessor64<scalar_t, 4> kx_arr,
        const torch::PackedTensorAccessor64<scalar_t, 4> vx_arr,
        const torch::PackedTensorAccessor64<scalar_t, 4> qy_arr,
        const torch::PackedTensorAccessor64<scalar_t, 1> quad_weights_arr,
        const torch::PackedTensorAccessor64<int64_t, 1> col_idx_arr,
        const torch::PackedTensorAccessor64<int64_t, 1> roff_arr,
        torch::PackedTensorAccessor64<scalar_t, 4> y_arr,
        const int64_t nlon_in, const int64_t nlat_in,
        const int64_t nlat_out, const int64_t nlon_out,
        const int64_t batch_size, const int64_t nchannels_in, const int64_t nchannels_out) {

        // one input lon step corresponds to pscale_out output lon steps (requires nlon_out % nlon_in == 0)
        const int64_t pscale_out = nlon_out / nlon_in;

        const int64_t block_wo = CACHE_BLOCK_SIZE;
        const int64_t nblock_wo = static_cast<int64_t>((nlon_out + block_wo - 1) / block_wo);

        #pragma omp parallel for collapse(4)
        for (int64_t b = 0; b < batch_size; b++) {
            for (int64_t co = 0; co < nchannels_out; co++) {
                for (int64_t ho = 0; ho < nlat_out; ho++) {
                    for (int64_t bwo = 0; bwo < nblock_wo; bwo++) {

                        int64_t wo_start = bwo * block_wo;
                        int64_t wo_end = std::min(nlon_out, wo_start + block_wo);

                        // per-block softmax state (classical, not online)
                        std::array<float, block_wo> qdotk_max;
                        std::array<float, block_wo> alpha_sum;
                        std::array<float, block_wo> y_tmp;
                        for (int64_t wob = 0; wob < block_wo; wob++) {
                            qdotk_max[wob] = -std::numeric_limits<float>::max();
                            alpha_sum[wob] = 0.0;
                            y_tmp[wob] = 0.0;
                        }

                        // -------------------------------------------------------------
                        // pass 1: find qdotk_max[wo] over all (input, neighbor) pairs
                        // that contribute to this (ho, wo_block)
                        // -------------------------------------------------------------
                        for (int64_t hi = 0; hi < nlat_in; hi++) {
                            int64_t zstart = roff_arr[hi];
                            int64_t zend = roff_arr[hi + 1];

                            for (int64_t idz = zstart; idz < zend; idz++) {
                                int64_t col = col_idx_arr[idz];
                                int64_t ho_neigh = col / nlon_out;
                                if (ho_neigh != ho) continue;
                                int64_t wo_canonical = col % nlon_out;

                                for (int64_t wo = wo_start; wo < wo_end; wo++) {
                                    // The map wi -> (wo_canonical + pscale_out * wi) mod nlon_out hits
                                    // exactly the wo values congruent to wo_canonical mod pscale_out;
                                    // for those, wi is uniquely determined within [0, nlon_in).
                                    int64_t wo_diff = (wo - wo_canonical + nlon_out) % nlon_out;
                                    if (wo_diff % pscale_out != 0) continue;
                                    int64_t wi = wo_diff / pscale_out;

                                    float qdotk = 0.0;
                                    //#pragma omp simd reduction(+:qdotk)
                                    for (int64_t ci = 0; ci < nchannels_in; ci++) {
                                        qdotk += static_cast<float>(qy_arr[b][ci][ho][wo] * kx_arr[b][ci][hi][wi]);
                                    }

                                    qdotk_max[wo - wo_start] = std::max(qdotk_max[wo - wo_start], qdotk);
                                }
                            }
                        }

                        // -------------------------------------------------------------
                        // pass 2: with max fixed, accumulate alpha_sum[wo] and y_tmp[wo]
                        // -------------------------------------------------------------
                        for (int64_t hi = 0; hi < nlat_in; hi++) {
                            int64_t zstart = roff_arr[hi];
                            int64_t zend = roff_arr[hi + 1];

                            for (int64_t idz = zstart; idz < zend; idz++) {
                                int64_t col = col_idx_arr[idz];
                                int64_t ho_neigh = col / nlon_out;
                                if (ho_neigh != ho) continue;
                                int64_t wo_canonical = col % nlon_out;

                                for (int64_t wo = wo_start; wo < wo_end; wo++) {
                                    int64_t wo_diff = (wo - wo_canonical + nlon_out) % nlon_out;
                                    if (wo_diff % pscale_out != 0) continue;
                                    int64_t wi = wo_diff / pscale_out;

                                    float qdotk = 0.0;
                                    //#pragma omp simd reduction(+:qdotk)
                                    for (int64_t ci = 0; ci < nchannels_in; ci++) {
                                        qdotk += static_cast<float>(qy_arr[b][ci][ho][wo] * kx_arr[b][ci][hi][wi]);
                                    }

                                    int64_t wob = wo - wo_start;
                                    float alpha = std::exp(qdotk - qdotk_max[wob]) * static_cast<float>(quad_weights_arr[hi]);
                                    alpha_sum[wob] += alpha;
                                    y_tmp[wob] += alpha * static_cast<float>(vx_arr[b][co][hi][wi]);
                                }
                            }
                        }

                        // finalize
                        for (int64_t wo = wo_start; wo < wo_end; wo++) {
                            y_arr[b][co][ho][wo] = static_cast<scalar_t>(y_tmp[wo - wo_start] / alpha_sum[wo - wo_start]);
                        }
                    }
                }
            }
        }
    }


    torch::Tensor s2_attention_fwd_cpu(at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor quad_weights,
        at::Tensor col_idx, at::Tensor row_off, int64_t nlon_in, int64_t nlat_out, int64_t nlon_out);

}
