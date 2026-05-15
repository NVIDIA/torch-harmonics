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

#define CACHE_BLOCK_SIZE_BWD (64)

namespace attention_kernels {

    // -----------------------------------------------------------------------
    // Self-attention / downsample backward.
    //
    // Canonical psi convention matches the forward (col_idx = hi*nlon_in + wi;
    // kernel applies the integer p-shift wip = (wi + pscale*wo) mod nlon_in).
    // All tensors enter as physical (B, H, W, C) — accessor indexing is
    // [b][h][w][c], channel stride = 1.
    //
    // Three OpenMP phases:
    //   Phase A (parallel collapse(3) over (b, ho, wo)):
    //     2-pass softmax per output cell. Computes & caches
    //       alpha_nz[idz]   = exp(qdotk[idz] - qdotk_max) * quad[hi]
    //       gdotv_nz[idz]   = <dy[b,ho,wo,:], vx[b,hi,wip,:]>
    //       alpha_sum       = sum alpha_nz
    //       alpha_gdotv     = sum alpha_nz * gdotv_nz
    //     into shared per-(b, ho, wo, idz) buffers.
    //
    //   Phase B (parallel collapse(2) over (b, ci)) — dqy + dkx:
    //     Iterates by hi OUTER (the dkx-scatter dim), scanning psi for
    //     entries matching this hi (mirrors the fwd-upsample-style redundant
    //     scan, avoids psi inversion). Per-hi scratch_dkx_row[Wi] ~720 B
    //     fits in L1; persisted scratch_dqy[Ho*Wo] ~64 KB lives in L2.
    //     Crucially, kx[hi, :, ci] reads localize to a single hi-row per
    //     iteration (~720 B, all-L1) instead of spanning the entire input
    //     plane as in earlier iterate-by-(ho,wo) designs.
    //
    //   Phase C (parallel collapse(2) over (b, co)) — dvx:
    //     Same iterate-by-hi pattern, simpler inner body (no dqy term).
    //
    // Memory cost: 2 × (B × Ho × Wo × max_nnz_per_row) floats for alpha_nz
    // and gdotv_nz, plus 2 × (B × Ho × Wo) for alpha_sum, alpha_gdotv. A few
    // MB at typical attention shapes.
    // -----------------------------------------------------------------------
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

        // one output lon step corresponds to pscale input lon steps
        const int64_t pscale = nlon_in / nlon_out;

        // input grid dim (not in the signature; derived from the K accessor)
        const int64_t nlat_in = kx_arr.size(1);

        // K/V/Q/dY/dK/dV/dQ all enter as physical (B, H, W, C); strides:
        //   stride(0) = H*W*C   stride(1) = W*C   stride(2) = C   stride(3) = 1
        // The inner reductions / scatters all walk the (innermost) C dim
        // stride-1 via raw pointer arithmetic.
        const scalar_t* __restrict__ kx_base = kx_arr.data();
        const scalar_t* __restrict__ vx_base = vx_arr.data();
        const scalar_t* __restrict__ qy_base = qy_arr.data();
        const scalar_t* __restrict__ dy_base = dy_arr.data();
        scalar_t* __restrict__       dkx_base = dkx_arr.data();
        scalar_t* __restrict__       dvx_base = dvx_arr.data();
        scalar_t* __restrict__       dqy_base = dqy_arr.data();
        const scalar_t* __restrict__ quad_p  = quad_weights_arr.data();
        const int64_t* __restrict__  col_p   = col_idx_arr.data();
        const int64_t* __restrict__  roff_p  = roff_arr.data();

        const int64_t kx_sB = kx_arr.stride(0), kx_sH = kx_arr.stride(1), kx_sW = kx_arr.stride(2);
        const int64_t vx_sB = vx_arr.stride(0), vx_sH = vx_arr.stride(1), vx_sW = vx_arr.stride(2);
        const int64_t qy_sB = qy_arr.stride(0), qy_sH = qy_arr.stride(1), qy_sW = qy_arr.stride(2);
        const int64_t dy_sB = dy_arr.stride(0), dy_sH = dy_arr.stride(1), dy_sW = dy_arr.stride(2);
        const int64_t dkx_sB = dkx_arr.stride(0), dkx_sH = dkx_arr.stride(1), dkx_sW = dkx_arr.stride(2);
        const int64_t dvx_sB = dvx_arr.stride(0), dvx_sH = dvx_arr.stride(1), dvx_sW = dvx_arr.stride(2);
        const int64_t dqy_sB = dqy_arr.stride(0), dqy_sH = dqy_arr.stride(1), dqy_sW = dqy_arr.stride(2);

        // max nnz across rows; controls the per-(b, ho, wo) buffer stride.
        int64_t max_nnz = 0;
        for (int64_t ho = 0; ho < nlat_out; ho++) {
            max_nnz = std::max(max_nnz, roff_p[ho + 1] - roff_p[ho]);
        }

        // Shared softmax-stats buffers. Layout:
        //   alpha_nz_buf[((b * nlat_out + ho) * nlon_out + wo) * max_nnz + idz_local]
        //   alpha_sum_buf[(b * nlat_out + ho) * nlon_out + wo]
        // The (idz_local) entries beyond (zend - zstart) are unused for any given ho.
        const int64_t nz_buf_per_cell = max_nnz;
        const int64_t nz_buf_total    = batch_size * nlat_out * nlon_out * nz_buf_per_cell;
        const int64_t sc_buf_total    = batch_size * nlat_out * nlon_out;

        std::vector<float> alpha_nz_buf  (nz_buf_total);
        std::vector<float> gdotv_nz_buf  (nz_buf_total);
        std::vector<float> alpha_sum_buf (sc_buf_total);
        std::vector<float> alpha_gdotv_buf(sc_buf_total);

        auto nz_off = [&](int64_t b, int64_t ho, int64_t wo) -> int64_t {
            return ((b * nlat_out + ho) * nlon_out + wo) * nz_buf_per_cell;
        };
        auto sc_off = [&](int64_t b, int64_t ho, int64_t wo) -> int64_t {
            return (b * nlat_out + ho) * nlon_out + wo;
        };

        // ----- Phase A: compute and cache softmax stats per (b, ho, wo) -----
        #pragma omp parallel
        {
            // per-thread reusable qdotk scratch (sized to max_nnz; reused across
            // all (b, ho, wo) cells assigned to this thread). Avoids the per-cell
            // std::vector allocation the legacy did.
            std::vector<float> qdotk_local(max_nnz);

            #pragma omp for collapse(3)
            for (int64_t b = 0; b < batch_size; b++) {
                for (int64_t ho = 0; ho < nlat_out; ho++) {
                    for (int64_t wo = 0; wo < nlon_out; wo++) {

                        const int64_t zstart = roff_p[ho];
                        const int64_t zend   = roff_p[ho + 1];

                        float* __restrict__ a_nz = &alpha_nz_buf[nz_off(b, ho, wo)];
                        float* __restrict__ g_nz = &gdotv_nz_buf[nz_off(b, ho, wo)];

                        // per-(b, ho, wo) base pointers (qy and dy)
                        const scalar_t* __restrict__ qy_bow =
                            qy_base + b * qy_sB + ho * qy_sH + wo * qy_sW;
                        const scalar_t* __restrict__ dy_bow =
                            dy_base + b * dy_sB + ho * dy_sH + wo * dy_sW;

                        // pass 1: compute qdotk[idz] and find qdotk_max
                        float qdotk_max = -std::numeric_limits<float>::max();
                        for (int64_t idz = zstart; idz < zend; idz++) {
                            const int64_t nz_col_idx = col_p[idz];
                            const int64_t hi  = nz_col_idx / nlon_in;
                            const int64_t wi  = nz_col_idx % nlon_in;
                            int64_t wip = wi + pscale * wo;
                            if (wip >= nlon_in) wip -= nlon_in;

                            const scalar_t* __restrict__ kx_biwip =
                                kx_base + b * kx_sB + hi * kx_sH + wip * kx_sW;

                            float qdotk = 0.0f;
                            #pragma omp simd reduction(+:qdotk)
                            for (int64_t cit = 0; cit < nchannels_in; cit++) {
                                qdotk += static_cast<float>(qy_bow[cit] * kx_biwip[cit]);
                            }
                            qdotk_local[idz - zstart] = qdotk;
                            qdotk_max = std::max(qdotk_max, qdotk);
                        }

                        // pass 2: compute alpha_nz, gdotv_nz, alpha_sum, alpha_gdotv
                        float alpha_sum   = 0.0f;
                        float alpha_gdotv = 0.0f;
                        for (int64_t idz = zstart; idz < zend; idz++) {
                            const int64_t nz_col_idx = col_p[idz];
                            const int64_t hi  = nz_col_idx / nlon_in;
                            const int64_t wi  = nz_col_idx % nlon_in;
                            int64_t wip = wi + pscale * wo;
                            if (wip >= nlon_in) wip -= nlon_in;

                            const scalar_t* __restrict__ vx_biwip =
                                vx_base + b * vx_sB + hi * vx_sH + wip * vx_sW;

                            const float alpha = std::exp(qdotk_local[idz - zstart] - qdotk_max)
                                              * static_cast<float>(quad_p[hi]);

                            float gdotv = 0.0f;
                            #pragma omp simd reduction(+:gdotv)
                            for (int64_t cot = 0; cot < nchannels_out; cot++) {
                                gdotv += static_cast<float>(dy_bow[cot] * vx_biwip[cot]);
                            }

                            a_nz[idz - zstart] = alpha;
                            g_nz[idz - zstart] = gdotv;
                            alpha_sum   += alpha;
                            alpha_gdotv += alpha * gdotv;
                        }

                        alpha_sum_buf  [sc_off(b, ho, wo)] = alpha_sum;
                        alpha_gdotv_buf[sc_off(b, ho, wo)] = alpha_gdotv;
                    }
                }
            }
        }

        // ----- Phase B: dqy and dkx, parallel over (b, ci) -----
        // Iterate-by-hi (mirrors fwd-upsample's redundant-scan-with-filter):
        // outer hi loop, scan psi for matches, accumulate into per-hi
        // scratch_dkx_row (~720 B, L1) and persistent scratch_dqy (~64 KB,
        // L2). kx[hi, :, ci] reads localize to one hi-row per iter and are
        // L1-resident after the first match.
        #pragma omp parallel
        {
            std::vector<float> scratch_dqy(static_cast<size_t>(nlat_out) * static_cast<size_t>(nlon_out));
            std::vector<float> scratch_dkx_row(static_cast<size_t>(nlon_in));

            #pragma omp for collapse(2)
            for (int64_t b = 0; b < batch_size; b++) {
                for (int64_t ci = 0; ci < nchannels_in; ci++) {

                    const int64_t qy_b_off  = b * qy_sB;
                    const int64_t kx_b_off  = b * kx_sB;
                    const int64_t dkx_b_off = b * dkx_sB;
                    const int64_t dqy_b_off = b * dqy_sB;

                    std::fill(scratch_dqy.begin(), scratch_dqy.end(), 0.0f);

                    for (int64_t hi = 0; hi < nlat_in; hi++) {
                        std::fill(scratch_dkx_row.begin(), scratch_dkx_row.end(), 0.0f);

                        for (int64_t ho = 0; ho < nlat_out; ho++) {
                            const int64_t zstart = roff_p[ho];
                            const int64_t zend   = roff_p[ho + 1];

                            for (int64_t idz = zstart; idz < zend; idz++) {
                                const int64_t nz_col_idx = col_p[idz];
                                if (nz_col_idx / nlon_in != hi) continue;  // filter to this hi
                                const int64_t wi        = nz_col_idx % nlon_in;
                                const int64_t idz_local = idz - zstart;

                                for (int64_t wo = 0; wo < nlon_out; wo++) {
                                    int64_t wip = wi + pscale * wo;
                                    if (wip >= nlon_in) wip -= nlon_in;

                                    const int64_t nz_off_wo = nz_off(b, ho, wo) + idz_local;
                                    const int64_t sc_off_wo = sc_off(b, ho, wo);

                                    const float alpha_nz    = alpha_nz_buf[nz_off_wo];
                                    const float gdotv       = gdotv_nz_buf[nz_off_wo];
                                    const float alpha_sum   = alpha_sum_buf  [sc_off_wo];
                                    const float alpha_gdotv = alpha_gdotv_buf[sc_off_wo];
                                    const float inv_sum     = 1.0f / alpha_sum;
                                    const float ag_over_as  = alpha_gdotv * inv_sum;

                                    const float kx_v = static_cast<float>(
                                        kx_base[kx_b_off + hi * kx_sH + wip * kx_sW + ci]);
                                    const float qy_v = static_cast<float>(
                                        qy_base[qy_b_off + ho * qy_sH + wo * qy_sW + ci]);

                                    const float weight = alpha_nz * inv_sum * (gdotv - ag_over_as);
                                    scratch_dkx_row[wip]            += qy_v * weight;
                                    scratch_dqy[ho * nlon_out + wo] += kx_v * weight;
                                }
                            }
                        }

                        // flush per-hi scratch into dkx[b, hi, :, ci]
                        for (int64_t wip = 0; wip < nlon_in; wip++) {
                            dkx_base[dkx_b_off + hi * dkx_sH + wip * dkx_sW + ci] +=
                                static_cast<scalar_t>(scratch_dkx_row[wip]);
                        }
                    }

                    // flush persisted dqy scratch into dqy[b, :, :, ci]
                    for (int64_t ho = 0; ho < nlat_out; ho++) {
                        for (int64_t wo = 0; wo < nlon_out; wo++) {
                            dqy_base[dqy_b_off + ho * dqy_sH + wo * dqy_sW + ci] +=
                                static_cast<scalar_t>(scratch_dqy[ho * nlon_out + wo]);
                        }
                    }
                }
            }
        }

        // ----- Phase C: dvx, parallel over (b, co) -----
        // Same iterate-by-hi pattern (no dqy-equivalent here, simpler body).
        #pragma omp parallel
        {
            std::vector<float> scratch_dvx_row(static_cast<size_t>(nlon_in));

            #pragma omp for collapse(2)
            for (int64_t b = 0; b < batch_size; b++) {
                for (int64_t co = 0; co < nchannels_out; co++) {

                    const int64_t dy_b_off  = b * dy_sB;
                    const int64_t dvx_b_off = b * dvx_sB;

                    for (int64_t hi = 0; hi < nlat_in; hi++) {
                        std::fill(scratch_dvx_row.begin(), scratch_dvx_row.end(), 0.0f);

                        for (int64_t ho = 0; ho < nlat_out; ho++) {
                            const int64_t zstart = roff_p[ho];
                            const int64_t zend   = roff_p[ho + 1];

                            for (int64_t idz = zstart; idz < zend; idz++) {
                                const int64_t nz_col_idx = col_p[idz];
                                if (nz_col_idx / nlon_in != hi) continue;  // filter to this hi
                                const int64_t wi        = nz_col_idx % nlon_in;
                                const int64_t idz_local = idz - zstart;

                                for (int64_t wo = 0; wo < nlon_out; wo++) {
                                    int64_t wip = wi + pscale * wo;
                                    if (wip >= nlon_in) wip -= nlon_in;

                                    const float alpha_nz   = alpha_nz_buf[nz_off(b, ho, wo) + idz_local];
                                    const float inv_sum    = 1.0f / alpha_sum_buf[sc_off(b, ho, wo)];
                                    const float alpha_norm = alpha_nz * inv_sum;

                                    const float dy_v = static_cast<float>(
                                        dy_base[dy_b_off + ho * dy_sH + wo * dy_sW + co]);
                                    scratch_dvx_row[wip] += alpha_norm * dy_v;
                                }
                            }
                        }

                        // flush per-hi scratch into dvx[b, hi, :, co]
                        for (int64_t wip = 0; wip < nlon_in; wip++) {
                            dvx_base[dvx_b_off + hi * dvx_sH + wip * dvx_sW + co] +=
                                static_cast<scalar_t>(scratch_dvx_row[wip]);
                        }
                    }
                }
            }
        }
    }


    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> s2_attention_bwd_cpu(torch::Tensor kx, torch::Tensor vx, torch::Tensor qy, torch::Tensor dy,
        torch::Tensor quad_weights, torch::Tensor col_idx, torch::Tensor row_off,
        int64_t nlon_in, int64_t nlat_out, int64_t nlon_out);

}
