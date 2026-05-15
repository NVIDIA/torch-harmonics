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

#include "attention_cpu_fwd_upsample.h"

#include <array>
#include <limits>
#include <vector>

#define CACHE_BLOCK_SIZE (64)

namespace attention_kernels {

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
    // Tensor layout: K/V/Q/Y all enter as physical (B, H, W, C) — accessor
    // indexing is [b][h][w][c], with c (channels) the innermost stride-1 dim.
    //
    // Parallelism: collapse(3) over (b, ho, wo-block). The softmax state
    // (qdotk_max, alpha_sum) is per-wo and shared across the co loop, so co
    // stays inside the pass-2 body — that avoids recomputing the softmax /
    // qdotk reductions for every output channel (the redundancy the legacy
    // kernel still pays). Per-thread y_tmp[block_wo × nchannels_out] buffers
    // the running v-accumulation; the finalize loop divides by alpha_sum and
    // writes out. No atomics needed (outputs strictly partitioned).
    //
    // Algorithm: classical 2-pass softmax (matches the torch reference):
    //   pass 1: find qdotk_max[wo] over all contributing (hi, idz) pairs
    //   pass 2: with max fixed, accumulate alpha_sum[wo] and y_tmp[wo, co]
    //   finalize: y[b, ho, wo, co] = y_tmp[wo, co] / alpha_sum[wo]
    //
    // For each output cell we scan all psi[hi] rows and skip entries whose
    // stored ho_neigh doesn't match — a stored (hi, wo_canonical) entry
    // contributes to output wo iff wo ≡ wo_canonical (mod pscale_out), and
    // the contributing wi is wi = (wo - wo_canonical) / pscale_out within
    // [0, nlon_in). The redundant scan over non-matching hi rows is the price
    // we pay to avoid scatter atomics.
    // -----------------------------------------------------------------------

    template <typename scalar_t>
    static void s2_attn_fwd_upsample_kernel(
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

        const int64_t block_wo  = CACHE_BLOCK_SIZE;
        const int64_t nblock_wo = static_cast<int64_t>((nlon_out + block_wo - 1) / block_wo);

        // K/V/Q/Y all enter as physical (B, H, W, C); strides:
        //   stride(0) = H*W*C   stride(1) = W*C   stride(2) = C   stride(3) = 1
        // The ci/co inner loops are stride-1 in the (innermost) C dim.
        const scalar_t* __restrict__ kx_base = kx_arr.data();
        const scalar_t* __restrict__ vx_base = vx_arr.data();
        const scalar_t* __restrict__ qy_base = qy_arr.data();
        scalar_t* __restrict__ y_base        = y_arr.data();
        const scalar_t* __restrict__ quad_p  = quad_weights_arr.data();
        const int64_t* __restrict__ col_p    = col_idx_arr.data();
        const int64_t* __restrict__ roff_p   = roff_arr.data();

        const int64_t kx_sB = kx_arr.stride(0);
        const int64_t kx_sH = kx_arr.stride(1);
        const int64_t kx_sW = kx_arr.stride(2);
        const int64_t vx_sB = vx_arr.stride(0);
        const int64_t vx_sH = vx_arr.stride(1);
        const int64_t vx_sW = vx_arr.stride(2);
        const int64_t qy_sB = qy_arr.stride(0);
        const int64_t qy_sH = qy_arr.stride(1);
        const int64_t qy_sW = qy_arr.stride(2);
        const int64_t y_sB  = y_arr.stride(0);
        const int64_t y_sH  = y_arr.stride(1);
        const int64_t y_sW  = y_arr.stride(2);

        #pragma omp parallel
        {
            // per-thread running v-accumulation for one wo-block:
            //   y_tmp[wob * nchannels_out + co]. Inner-co is stride-1.
            std::vector<float> y_tmp(block_wo * nchannels_out);

            #pragma omp for collapse(3)
            for (int64_t b = 0; b < batch_size; b++) {
                for (int64_t ho = 0; ho < nlat_out; ho++) {
                    for (int64_t bwo = 0; bwo < nblock_wo; bwo++) {

                        const int64_t wo_start = bwo * block_wo;
                        const int64_t wo_end   = std::min(nlon_out, wo_start + block_wo);
                        const int64_t this_block = wo_end - wo_start;

                        // zero this block's y_tmp slice
                        std::fill(y_tmp.begin(), y_tmp.begin() + this_block * nchannels_out, 0.0f);

                        std::array<float, block_wo> alpha_sum{};
                        std::array<float, block_wo> qdotk_max;
                        qdotk_max.fill(-std::numeric_limits<float>::max());

                        // hoist (b, ho) pointer bases that survive the whole block
                        const scalar_t* __restrict__ qy_b_ho = qy_base + b * qy_sB + ho * qy_sH;
                        scalar_t* __restrict__       y_b_ho  = y_base  + b * y_sB  + ho * y_sH;
                        const scalar_t* __restrict__ kx_b    = kx_base + b * kx_sB;
                        const scalar_t* __restrict__ vx_b    = vx_base + b * vx_sB;

                        // -------------------------------------------------------------
                        // pass 1: find qdotk_max[wo] over all (input, neighbor) pairs
                        // that contribute to this (ho, wo_block)
                        // -------------------------------------------------------------
                        for (int64_t hi = 0; hi < nlat_in; hi++) {
                            const int64_t zstart = roff_p[hi];
                            const int64_t zend   = roff_p[hi + 1];

                            // hoist (b, hi) pointer bases for this row
                            const scalar_t* __restrict__ kx_b_hi = kx_b + hi * kx_sH;

                            for (int64_t idz = zstart; idz < zend; idz++) {
                                const int64_t col = col_p[idz];
                                const int64_t ho_neigh = col / nlon_out;
                                if (ho_neigh != ho) continue;
                                const int64_t wo_canonical = col % nlon_out;

                                for (int64_t wo = wo_start; wo < wo_end; wo++) {
                                    // wi -> (wo_canonical + pscale_out*wi) mod nlon_out hits
                                    // exactly wo ≡ wo_canonical (mod pscale_out); for those
                                    // wi is uniquely determined within [0, nlon_in).
                                    const int64_t wo_diff = (wo - wo_canonical + nlon_out) % nlon_out;
                                    if (wo_diff % pscale_out != 0) continue;
                                    const int64_t wi = wo_diff / pscale_out;

                                    // per-wo channel-vector pointers (ci stride-1)
                                    const scalar_t* __restrict__ qy_bow   = qy_b_ho + wo * qy_sW;
                                    const scalar_t* __restrict__ kx_biwi  = kx_b_hi + wi * kx_sW;

                                    float qdotk = 0.0f;
                                    for (int64_t ci = 0; ci < nchannels_in; ci++) {
                                        qdotk += static_cast<float>(qy_bow[ci] * kx_biwi[ci]);
                                    }

                                    qdotk_max[wo - wo_start] = std::max(qdotk_max[wo - wo_start], qdotk);
                                }
                            }
                        }

                        // -------------------------------------------------------------
                        // pass 2: accumulate alpha_sum[wob] and y_tmp[wob, co] across
                        // all output channels in a single inner SAXPY
                        // -------------------------------------------------------------
                        for (int64_t hi = 0; hi < nlat_in; hi++) {
                            const int64_t zstart = roff_p[hi];
                            const int64_t zend   = roff_p[hi + 1];
                            const float qw_hi = static_cast<float>(quad_p[hi]);

                            // hoist (b, hi) pointer bases for this row
                            const scalar_t* __restrict__ kx_b_hi = kx_b + hi * kx_sH;
                            const scalar_t* __restrict__ vx_b_hi = vx_b + hi * vx_sH;

                            for (int64_t idz = zstart; idz < zend; idz++) {
                                const int64_t col = col_p[idz];
                                const int64_t ho_neigh = col / nlon_out;
                                if (ho_neigh != ho) continue;
                                const int64_t wo_canonical = col % nlon_out;

                                for (int64_t wo = wo_start; wo < wo_end; wo++) {
                                    const int64_t wo_diff = (wo - wo_canonical + nlon_out) % nlon_out;
                                    if (wo_diff % pscale_out != 0) continue;
                                    const int64_t wi  = wo_diff / pscale_out;
                                    const int64_t wob = wo - wo_start;

                                    // per-wo / per-wi channel-vector pointers
                                    const scalar_t* __restrict__ qy_bow  = qy_b_ho + wo * qy_sW;
                                    const scalar_t* __restrict__ kx_biwi = kx_b_hi + wi * kx_sW;
                                    const scalar_t* __restrict__ vx_biwi = vx_b_hi + wi * vx_sW;

                                    // recompute qdotk (matches pass 1)
                                    float qdotk = 0.0f;
                                    for (int64_t ci = 0; ci < nchannels_in; ci++) {
                                        qdotk += static_cast<float>(qy_bow[ci] * kx_biwi[ci]);
                                    }

                                    const float alpha = std::exp(qdotk - qdotk_max[wob]) * qw_hi;
                                    alpha_sum[wob] += alpha;

                                    // SAXPY: y_tmp[wob][co] += alpha * vx[b, hi, wi, co]
                                    float* __restrict__ y_tmp_wob = y_tmp.data() + wob * nchannels_out;
                                    for (int64_t co = 0; co < nchannels_out; co++) {
                                        y_tmp_wob[co] += alpha * static_cast<float>(vx_biwi[co]);
                                    }
                                }
                            }
                        }

                        // finalize: divide by alpha_sum and write to y (stride-1 in co)
                        for (int64_t wo = wo_start; wo < wo_end; wo++) {
                            const int64_t wob = wo - wo_start;
                            const float inv_sum = 1.0f / alpha_sum[wob];
                            const float* __restrict__ y_tmp_wob = y_tmp.data() + wob * nchannels_out;
                            scalar_t* __restrict__ y_bow = y_b_ho + wo * y_sW;
                            for (int64_t co = 0; co < nchannels_out; co++) {
                                y_bow[co] = static_cast<scalar_t>(y_tmp_wob[co] * inv_sum);
                            }
                        }
                    }
                }
            }
        }
    }


    void s2_attn_fwd_upsample_dispatch(
        const torch::PackedTensorAccessor64<float, 4> kx_arr,
        const torch::PackedTensorAccessor64<float, 4> vx_arr,
        const torch::PackedTensorAccessor64<float, 4> qy_arr,
        const torch::PackedTensorAccessor64<float, 1> quad_weights_arr,
        const torch::PackedTensorAccessor64<int64_t, 1> col_idx_arr,
        const torch::PackedTensorAccessor64<int64_t, 1> roff_arr,
        torch::PackedTensorAccessor64<float, 4> y_arr,
        int64_t nlon_in, int64_t nlat_in,
        int64_t nlat_out, int64_t nlon_out,
        int64_t batch_size, int64_t nchannels_in, int64_t nchannels_out) {

        s2_attn_fwd_upsample_kernel<float>(
            kx_arr, vx_arr, qy_arr, quad_weights_arr, col_idx_arr, roff_arr, y_arr,
            nlon_in, nlat_in, nlat_out, nlon_out,
            batch_size, nchannels_in, nchannels_out);
    }

}
