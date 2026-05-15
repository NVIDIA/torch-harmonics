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

#include "attention_cpu_bwd_upsample.h"

#include <array>
#include <limits>

#define CACHE_BLOCK_SIZE_BWD (64)

namespace attention_kernels {

    // -----------------------------------------------------------------------
    // Upsample backward (matches the torch reference, which is itself written
    // as classical 2/3-pass scatter softmax). psi rows are indexed by hi and
    // cols are ho * nlon_out + wo_canonical (canonical at wi=0); the actual
    // output column for wi > 0 is (wo_canonical + pscale_out * wi) mod nlon_out
    // with pscale_out = nlon_out / nlon_in.
    //
    // Tensor layout: all tensors enter as physical (B, H, W, C) — accessor
    // indexing is [b][h][w][c], channel stride = 1; wrapper handles BCHW↔BHWC.
    //
    // Two OpenMP regions, same parallelization trick as the downsample bwd:
    //   1) dqy and dkx — parallelize collapse(2) over (batch, in-channel). Each
    //      thread owns its (b, ci) slice of dkx, so the input-side accumulation
    //      is race-free without atomics.
    //   2) dvx — parallelize collapse(2) over (batch, out-channel). Each thread
    //      owns its (b, co) slice of dvx.
    // Within each region we iterate (ho, wo-block) sequentially per thread and
    // run 3 passes per block:
    //   pass 1: scan psi for entries with ho_neigh == ho, find qdotk_max[wo].
    //   pass 2: with max fixed, accumulate alpha_sum / alpha_vw / alpha_k /
    //           alpha_kvw (region 1) or just alpha_sum (region 2). Region 1
    //           also computes integral = alpha_vw / alpha_sum and writes
    //           dqy[b,ci,ho,wo] = (alpha_kvw - integral * alpha_k) / alpha_sum.
    //   pass 3: scatter dkx[b,ci,hi,wi] (region 1) or dvx[b,co,hi,wi]
    //           (region 2) using the finalized softmax stats.
    // The redundant scan over non-matching hi rows is the price of avoiding
    // scatter atomics — reference correctness path, not perf path.
    // -----------------------------------------------------------------------
    template <typename scalar_t>
    static void s2_attn_bwd_upsample_kernel(
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
        const int64_t nlon_in, const int64_t nlat_in,
        const int64_t nlat_out, const int64_t nlon_out,
        const int64_t batch_size, const int64_t nchannels_in, const int64_t nchannels_out) {

        const int64_t pscale_out = nlon_out / nlon_in;

        const int64_t block_wo = CACHE_BLOCK_SIZE_BWD;
        const int64_t nblock_wo = static_cast<int64_t>((nlon_out + block_wo - 1) / block_wo);

        // ====================================================================
        // Region 1: dqy and dkx
        // ====================================================================
        #pragma omp parallel for collapse(2)
        for (int64_t b = 0; b < batch_size; b++) {
            for (int64_t ci = 0; ci < nchannels_in; ci++) {

                for (int64_t ho = 0; ho < nlat_out; ho++) {
                    for (int64_t bwo = 0; bwo < nblock_wo; bwo++) {

                        int64_t wo_start = bwo * block_wo;
                        int64_t wo_end = std::min(nlon_out, wo_start + block_wo);

                        std::array<float, block_wo> alpha_sum{};
                        std::array<float, block_wo> alpha_vw{};
                        std::array<float, block_wo> alpha_k{};
                        std::array<float, block_wo> alpha_kvw{};
                        std::array<float, block_wo> integral{};
                        std::array<float, block_wo> qdotk_max;
                        qdotk_max.fill(-std::numeric_limits<float>::max());

                        // ---- pass 1: qdotk_max[wo] ----
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
                                    for (int64_t cit = 0; cit < nchannels_in; cit++) {
                                        qdotk += static_cast<float>(qy_arr[b][ho][wo][cit] * kx_arr[b][hi][wi][cit]);
                                    }
                                    qdotk_max[wo - wo_start] = std::max(qdotk_max[wo - wo_start], qdotk);
                                }
                            }
                        }

                        // ---- pass 2: alpha_sum, alpha_vw, alpha_k, alpha_kvw ----
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
                                    int64_t wob = wo - wo_start;

                                    float qdotk = 0.0;
                                    for (int64_t cit = 0; cit < nchannels_in; cit++) {
                                        qdotk += static_cast<float>(qy_arr[b][ho][wo][cit] * kx_arr[b][hi][wi][cit]);
                                    }

                                    float alpha = std::exp(qdotk - qdotk_max[wob]) * static_cast<float>(quad_weights_arr[hi]);

                                    float gdotv = 0.0;
                                    for (int64_t cot = 0; cot < nchannels_out; cot++) {
                                        gdotv += static_cast<float>(dy_arr[b][ho][wo][cot] * vx_arr[b][hi][wi][cot]);
                                    }

                                    alpha_sum[wob]  += alpha;
                                    alpha_vw[wob]   += alpha * gdotv;
                                    alpha_k[wob]    += alpha * static_cast<float>(kx_arr[b][hi][wi][ci]);
                                    alpha_kvw[wob]  += alpha * gdotv * static_cast<float>(kx_arr[b][hi][wi][ci]);
                                }
                            }
                        }

                        // ---- finalize dqy and integrals per wo in block ----
                        for (int64_t wo = wo_start; wo < wo_end; wo++) {
                            int64_t wob = wo - wo_start;
                            integral[wob] = alpha_vw[wob] / alpha_sum[wob];
                            float ak_norm   = alpha_k[wob]   / alpha_sum[wob];
                            float akvw_norm = alpha_kvw[wob] / alpha_sum[wob];
                            dqy_arr[b][ho][wo][ci] = static_cast<scalar_t>(akvw_norm - integral[wob] * ak_norm);
                        }

                        // ---- pass 3: scatter dkx ----
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
                                    int64_t wob = wo - wo_start;

                                    float qdotk = 0.0;
                                    for (int64_t cit = 0; cit < nchannels_in; cit++) {
                                        qdotk += static_cast<float>(qy_arr[b][ho][wo][cit] * kx_arr[b][hi][wi][cit]);
                                    }

                                    float alpha_norm = std::exp(qdotk - qdotk_max[wob]) * static_cast<float>(quad_weights_arr[hi]) / alpha_sum[wob];

                                    float gdotv = 0.0;
                                    for (int64_t cot = 0; cot < nchannels_out; cot++) {
                                        gdotv += static_cast<float>(dy_arr[b][ho][wo][cot] * vx_arr[b][hi][wi][cot]);
                                    }

                                    dkx_arr[b][hi][wi][ci] += static_cast<scalar_t>(static_cast<float>(qy_arr[b][ho][wo][ci]) * alpha_norm * (gdotv - integral[wob]));
                                }
                            }
                        }
                    }
                }
            }
        }

        // ====================================================================
        // Region 2: dvx
        // ====================================================================
        #pragma omp parallel for collapse(2)
        for (int64_t b = 0; b < batch_size; b++) {
            for (int64_t co = 0; co < nchannels_out; co++) {

                for (int64_t ho = 0; ho < nlat_out; ho++) {
                    for (int64_t bwo = 0; bwo < nblock_wo; bwo++) {

                        int64_t wo_start = bwo * block_wo;
                        int64_t wo_end = std::min(nlon_out, wo_start + block_wo);

                        std::array<float, block_wo> alpha_sum{};
                        std::array<float, block_wo> qdotk_max;
                        qdotk_max.fill(-std::numeric_limits<float>::max());

                        // ---- pass 1: qdotk_max[wo] ----
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
                                    int64_t wob = wo - wo_start;

                                    float qdotk = 0.0;
                                    for (int64_t cit = 0; cit < nchannels_in; cit++) {
                                        qdotk += static_cast<float>(qy_arr[b][ho][wo][cit] * kx_arr[b][hi][wi][cit]);
                                    }
                                    qdotk_max[wob] = std::max(qdotk_max[wob], qdotk);
                                }
                            }
                        }

                        // ---- pass 2: alpha_sum ----
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
                                    int64_t wob = wo - wo_start;

                                    float qdotk = 0.0;
                                    for (int64_t cit = 0; cit < nchannels_in; cit++) {
                                        qdotk += static_cast<float>(qy_arr[b][ho][wo][cit] * kx_arr[b][hi][wi][cit]);
                                    }
                                    alpha_sum[wob] += std::exp(qdotk - qdotk_max[wob]) * static_cast<float>(quad_weights_arr[hi]);
                                }
                            }
                        }

                        // ---- pass 3: scatter dvx ----
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
                                    int64_t wob = wo - wo_start;

                                    float qdotk = 0.0;
                                    for (int64_t cit = 0; cit < nchannels_in; cit++) {
                                        qdotk += static_cast<float>(qy_arr[b][ho][wo][cit] * kx_arr[b][hi][wi][cit]);
                                    }
                                    float alpha_norm = std::exp(qdotk - qdotk_max[wob]) * static_cast<float>(quad_weights_arr[hi]) / alpha_sum[wob];
                                    dvx_arr[b][hi][wi][co] += static_cast<scalar_t>(alpha_norm * static_cast<float>(dy_arr[b][ho][wo][co]));
                                }
                            }
                        }
                    }
                }
            }
        }
    }


    void s2_attn_bwd_upsample_dispatch(
        const torch::PackedTensorAccessor64<float, 4> kx_arr,
        const torch::PackedTensorAccessor64<float, 4> vx_arr,
        const torch::PackedTensorAccessor64<float, 4> qy_arr,
        const torch::PackedTensorAccessor64<float, 4> dy_arr,
        const torch::PackedTensorAccessor64<float, 1> quad_weights_arr,
        const torch::PackedTensorAccessor64<int64_t, 1> col_idx_arr,
        const torch::PackedTensorAccessor64<int64_t, 1> roff_arr,
        torch::PackedTensorAccessor64<float, 4> dqy_arr,
        torch::PackedTensorAccessor64<float, 4> dvx_arr,
        torch::PackedTensorAccessor64<float, 4> dkx_arr,
        int64_t nlon_in, int64_t nlat_in,
        int64_t nlat_out, int64_t nlon_out,
        int64_t batch_size, int64_t nchannels_in, int64_t nchannels_out) {

        s2_attn_bwd_upsample_kernel<float>(
            kx_arr, vx_arr, qy_arr, dy_arr,
            quad_weights_arr, col_idx_arr, roff_arr,
            dqy_arr, dvx_arr, dkx_arr,
            nlon_in, nlat_in, nlat_out, nlon_out,
            batch_size, nchannels_in, nchannels_out);
    }

}
