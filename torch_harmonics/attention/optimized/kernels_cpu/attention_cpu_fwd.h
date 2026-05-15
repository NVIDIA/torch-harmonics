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
    // Parallelism: collapse(3) over (batch, ho, wo-block). The softmax state
    // (qdotk_max, alpha_sum) is per-wo and shared across the co loop, so co
    // stays *inside* the per-nz body — that avoids recomputing the softmax
    // for every output channel. Per-thread y_tmp[block_wo × nchannels_out]
    // buffers the running v-accumulation; the finalize loop divides by
    // alpha_sum and writes out. No atomics needed (outputs strictly partitioned).
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

        // one output lon step corresponds to pscale input lon steps
        const int64_t pscale = nlon_in / nlon_out;

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

                        const int64_t zstart = roff_p[ho];
                        const int64_t zend   = roff_p[ho + 1];

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

                        for (int64_t idz = zstart; idz < zend; idz++) {
                            const int64_t nz_col_idx = col_p[idz];
                            const int64_t hi = nz_col_idx / nlon_in;
                            const int64_t wi = nz_col_idx % nlon_in;

                            // hoist (b, hi) pointer bases + quad weight for this row
                            const scalar_t* __restrict__ kx_b_hi = kx_b + hi * kx_sH;
                            const scalar_t* __restrict__ vx_b_hi = vx_b + hi * vx_sH;
                            const float qw_hi = static_cast<float>(quad_p[hi]);

                            // Incremental wip = (wi + pscale * wo) mod nlon_in.
                            // One real modulo per (idz, bwo); then wip advances by
                            // pscale per wo step and wraps with a single
                            // conditional subtract (pscale <= nlon_in so the
                            // post-increment value is always < 2*nlon_in).
                            int64_t wip = (wi + pscale * wo_start) % nlon_in;

                            for (int64_t wo = wo_start; wo < wo_end; wo++) {
                                const int64_t wob = wo - wo_start;

                                // per-wo channel-vector pointers (ci/co stride-1)
                                const scalar_t* __restrict__ qy_bow   = qy_b_ho + wo  * qy_sW;
                                const scalar_t* __restrict__ kx_biwip = kx_b_hi + wip * kx_sW;
                                const scalar_t* __restrict__ vx_biwip = vx_b_hi + wip * vx_sW;

                                // qdotk: pure ci reduction, stride-1 dot product
                                float qdotk = 0.0f;
                                #pragma omp simd reduction(+:qdotk)
                                for (int64_t ci = 0; ci < nchannels_in; ci++) {
                                    qdotk += static_cast<float>(qy_bow[ci] * kx_biwip[ci]);
                                }

                                // online softmax update
                                const float qdotk_max_tmp = std::max(qdotk_max[wob], qdotk);
                                const float discount      = std::exp(qdotk_max[wob] - qdotk_max_tmp);
                                const float alpha         = std::exp(qdotk - qdotk_max_tmp) * qw_hi;
                                alpha_sum[wob] = alpha + alpha_sum[wob] * discount;

                                // v-accumulation: stride-1 SAXPY in co
                                float* __restrict__ y_tmp_wob = y_tmp.data() + wob * nchannels_out;
                                #pragma omp simd
                                for (int64_t co = 0; co < nchannels_out; co++) {
                                    y_tmp_wob[co] = y_tmp_wob[co] * discount +
                                                    alpha * static_cast<float>(vx_biwip[co]);
                                }

                                qdotk_max[wob] = qdotk_max_tmp;

                                // advance wip for the next wo; single branchless wrap
                                wip += pscale;
                                if (wip >= nlon_in) wip -= nlon_in;
                            }
                        }

                        // finalize: divide by alpha_sum and write to y (stride-1 in co)
                        for (int64_t wo = wo_start; wo < wo_end; wo++) {
                            const int64_t wob = wo - wo_start;
                            const float inv_sum = 1.0f / alpha_sum[wob];
                            const float* __restrict__ y_tmp_wob = y_tmp.data() + wob * nchannels_out;
                            scalar_t* __restrict__ y_bow = y_b_ho + wo * y_sW;
                            #pragma omp simd
                            for (int64_t co = 0; co < nchannels_out; co++) {
                                y_bow[co] = static_cast<scalar_t>(y_tmp_wob[co] * inv_sum);
                            }
                        }
                    }
                }
            }
        }
    }


    torch::Tensor s2_attention_fwd_cpu(at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor quad_weights,
        at::Tensor col_idx, at::Tensor row_off, int64_t nlon_in, int64_t nlat_out, int64_t nlon_out);

}
