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
#include <vector>

namespace attention_kernels {

    // -----------------------------------------------------------------------
    // Upsample backward.
    //
    // psi convention (matches the upsample forward): rows indexed by hi and
    // cols encoding (ho_neigh, wo_canonical) on the output grid as
    // ho_neigh * nlon_out + wo_canonical. For wi > 0 the actual output column
    // is wo = (wo_canonical + pscale_out * wi) mod nlon_out, with
    // pscale_out = nlon_out / nlon_in. Requires nlon_out % nlon_in == 0.
    //
    // Tensor layout: all tensors enter as physical (B, H, W, C) — accessor
    // indexing is [b][h][w][c], channel stride = 1; wrapper handles BCHW↔BHWC.
    //
    // Three OpenMP phases (mirrors bwd downsample's structure, but adapted to
    // the upsample-side psi keying):
    //
    //   Phase A (parallel collapse(3) over (b, ho, wo)):
    //     Per-output-cell 2-pass softmax. Walks all hi rows of psi to find
    //     contributing entries (the unavoidable redundant scan, same as the
    //     fwd upsample). Computes and caches:
    //       alpha_nz[hi, idz_local, wi]   = exp(qdotk - qdotk_max[ho, wo]) * quad[hi]
    //       gdotv_nz[hi, idz_local, wi]   = <dy[b, ho, wo, :], vx[b, hi, wi, :]>
    //       alpha_sum[ho, wo]             = sum of alpha_nz across contributors
    //       alpha_gdotv[ho, wo]           = sum of alpha_nz * gdotv_nz
    //     The per-entry buffers are indexed by (b, hi, idz_local, wi) so
    //     phases B/C can walk psi[hi] directly without a reverse lookup.
    //
    //   Phase B (parallel collapse(2) over (b, ci)) — dqy + dkx:
    //     Inside, iterate hi OUTER, then walk psi[hi] directly (no filter,
    //     since psi is keyed by hi here). For each entry (idz → ho_neigh,
    //     wo_canonical), iterate wi → derive wo = wo_canonical + pscale_out*wi
    //     (mod nlon_out via conditional subtract). Per-hi scratch_dkx_row
    //     (Wi floats, L1) for dkx accumulation; persistent scratch_dqy
    //     (Ho × Wo floats, L2) for dqy. Mirrors fwd downsample's natural
    //     walk-psi-directly pattern.
    //
    //   Phase C (parallel collapse(2) over (b, co)) — dvx:
    //     Same iterate-by-hi pattern, simpler body (only dvx).
    //
    // Memory cost: 2 × (B × nlat_in × max_nnz_per_psi_row × nlon_in) floats
    // for alpha_nz / gdotv_nz, plus 2 × (B × Ho × Wo) for the scalar caches.
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

        // Hoist base pointers + strides. Channel stride is 1 throughout.
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

        // max nnz per psi row (keyed by hi); buffer stride.
        int64_t max_nnz = 0;
        for (int64_t hi = 0; hi < nlat_in; hi++) {
            max_nnz = std::max(max_nnz, roff_p[hi + 1] - roff_p[hi]);
        }

        // Shared softmax-stats buffers.
        //   alpha_nz_buf, gdotv_nz_buf: per (b, hi, idz_local, wi). Indexed so
        //     that for fixed (b, hi, idz_local), wi varies contiguously.
        //   alpha_sum_buf, alpha_gdotv_buf: per (b, ho, wo). Stride-1 in wo.
        const int64_t nz_buf_per_hi = max_nnz * nlon_in;
        const int64_t nz_buf_total  = batch_size * nlat_in  * nz_buf_per_hi;
        const int64_t sc_buf_total  = batch_size * nlat_out * nlon_out;

        std::vector<float> alpha_nz_buf  (nz_buf_total);
        std::vector<float> gdotv_nz_buf  (nz_buf_total);
        std::vector<float> alpha_sum_buf (sc_buf_total);
        std::vector<float> alpha_gdotv_buf(sc_buf_total);

        auto nz_off = [&](int64_t b, int64_t hi, int64_t idz_local, int64_t wi) -> int64_t {
            return ((b * nlat_in + hi) * max_nnz + idz_local) * nlon_in + wi;
        };
        auto sc_off = [&](int64_t b, int64_t ho, int64_t wo) -> int64_t {
            return (b * nlat_out + ho) * nlon_out + wo;
        };

        // ----- Phase A: precompute softmax stats per (b, ho, wo) -----
        // Redundant scan over psi[hi] for matches (mirrors fwd upsample).
        #pragma omp parallel
        {
            // per-thread qdotk scratch sized for max possible matches per cell.
            // Conservative bound: total psi nnz (all entries could match in
            // worst case). In practice much smaller; we resize the active prefix.
            std::vector<float> qdotk_local(roff_p[nlat_in]);

            #pragma omp for collapse(3)
            for (int64_t b = 0; b < batch_size; b++) {
                for (int64_t ho = 0; ho < nlat_out; ho++) {
                    for (int64_t wo = 0; wo < nlon_out; wo++) {

                        // per-(b, ho, wo) base pointers
                        const scalar_t* __restrict__ qy_bow =
                            qy_base + b * qy_sB + ho * qy_sH + wo * qy_sW;
                        const scalar_t* __restrict__ dy_bow =
                            dy_base + b * dy_sB + ho * dy_sH + wo * dy_sW;

                        // pass 1: find qdotk_max over contributing entries
                        int64_t n_match = 0;
                        float qdotk_max = -std::numeric_limits<float>::max();
                        for (int64_t hi = 0; hi < nlat_in; hi++) {
                            const int64_t zstart = roff_p[hi];
                            const int64_t zend   = roff_p[hi + 1];
                            for (int64_t idz = zstart; idz < zend; idz++) {
                                const int64_t col = col_p[idz];
                                const int64_t ho_neigh = col / nlon_out;
                                if (ho_neigh != ho) continue;
                                const int64_t wo_canonical = col % nlon_out;
                                int64_t wo_diff = wo - wo_canonical;
                                if (wo_diff < 0) wo_diff += nlon_out;
                                if (wo_diff % pscale_out != 0) continue;
                                const int64_t wi = wo_diff / pscale_out;

                                const scalar_t* __restrict__ kx_biwi =
                                    kx_base + b * kx_sB + hi * kx_sH + wi * kx_sW;

                                float qdotk = 0.0f;
                                #pragma omp simd reduction(+:qdotk)
                                for (int64_t cit = 0; cit < nchannels_in; cit++) {
                                    qdotk += static_cast<float>(qy_bow[cit] * kx_biwi[cit]);
                                }
                                qdotk_local[n_match++] = qdotk;
                                qdotk_max = std::max(qdotk_max, qdotk);
                            }
                        }

                        // pass 2: compute alpha_nz, gdotv_nz, sums (same iteration order)
                        float alpha_sum   = 0.0f;
                        float alpha_gdotv = 0.0f;
                        int64_t match_idx = 0;
                        for (int64_t hi = 0; hi < nlat_in; hi++) {
                            const int64_t zstart = roff_p[hi];
                            const int64_t zend   = roff_p[hi + 1];
                            const float qw_hi = static_cast<float>(quad_p[hi]);
                            for (int64_t idz = zstart; idz < zend; idz++) {
                                const int64_t col = col_p[idz];
                                const int64_t ho_neigh = col / nlon_out;
                                if (ho_neigh != ho) continue;
                                const int64_t wo_canonical = col % nlon_out;
                                int64_t wo_diff = wo - wo_canonical;
                                if (wo_diff < 0) wo_diff += nlon_out;
                                if (wo_diff % pscale_out != 0) continue;
                                const int64_t wi        = wo_diff / pscale_out;
                                const int64_t idz_local = idz - zstart;

                                const scalar_t* __restrict__ vx_biwi =
                                    vx_base + b * vx_sB + hi * vx_sH + wi * vx_sW;

                                const float alpha = std::exp(qdotk_local[match_idx++] - qdotk_max) * qw_hi;

                                float gdotv = 0.0f;
                                #pragma omp simd reduction(+:gdotv)
                                for (int64_t cot = 0; cot < nchannels_out; cot++) {
                                    gdotv += static_cast<float>(dy_bow[cot] * vx_biwi[cot]);
                                }

                                alpha_nz_buf[nz_off(b, hi, idz_local, wi)] = alpha;
                                gdotv_nz_buf[nz_off(b, hi, idz_local, wi)] = gdotv;
                                alpha_sum   += alpha;
                                alpha_gdotv += alpha * gdotv;
                            }
                        }

                        alpha_sum_buf  [sc_off(b, ho, wo)] = alpha_sum;
                        alpha_gdotv_buf[sc_off(b, ho, wo)] = alpha_gdotv;
                    }
                }
            }
        }

        // ----- Phase B: dqy and dkx, parallel over (b, ci) -----
        // Walks psi[hi] directly (no filter / no redundant scan) — psi is
        // natively keyed by hi here. Mirrors fwd downsample's iteration.
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

                        const int64_t zstart = roff_p[hi];
                        const int64_t zend   = roff_p[hi + 1];

                        for (int64_t idz = zstart; idz < zend; idz++) {
                            const int64_t col          = col_p[idz];
                            const int64_t ho_neigh     = col / nlon_out;
                            const int64_t wo_canonical = col % nlon_out;
                            const int64_t idz_local    = idz - zstart;

                            for (int64_t wi = 0; wi < nlon_in; wi++) {
                                int64_t wo = wo_canonical + pscale_out * wi;
                                if (wo >= nlon_out) wo -= nlon_out;

                                const int64_t nz_off_wi = nz_off(b, hi, idz_local, wi);
                                const int64_t sc_off_wo = sc_off(b, ho_neigh, wo);

                                const float alpha_nz    = alpha_nz_buf  [nz_off_wi];
                                const float gdotv       = gdotv_nz_buf  [nz_off_wi];
                                const float alpha_sum   = alpha_sum_buf [sc_off_wo];
                                const float alpha_gdotv = alpha_gdotv_buf[sc_off_wo];
                                const float inv_sum     = 1.0f / alpha_sum;
                                const float integral    = alpha_gdotv * inv_sum;
                                const float alpha_norm  = alpha_nz * inv_sum;

                                const float kx_v = static_cast<float>(
                                    kx_base[kx_b_off + hi * kx_sH + wi * kx_sW + ci]);
                                const float qy_v = static_cast<float>(
                                    qy_base[qy_b_off + ho_neigh * qy_sH + wo * qy_sW + ci]);

                                const float weight = alpha_norm * (gdotv - integral);
                                scratch_dkx_row[wi]                  += qy_v * weight;
                                scratch_dqy[ho_neigh * nlon_out + wo] += kx_v * weight;
                            }
                        }

                        // flush per-hi dkx scratch
                        for (int64_t wi = 0; wi < nlon_in; wi++) {
                            dkx_base[dkx_b_off + hi * dkx_sH + wi * dkx_sW + ci] +=
                                static_cast<scalar_t>(scratch_dkx_row[wi]);
                        }
                    }

                    // flush persisted dqy scratch
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

                        const int64_t zstart = roff_p[hi];
                        const int64_t zend   = roff_p[hi + 1];

                        for (int64_t idz = zstart; idz < zend; idz++) {
                            const int64_t col          = col_p[idz];
                            const int64_t ho_neigh     = col / nlon_out;
                            const int64_t wo_canonical = col % nlon_out;
                            const int64_t idz_local    = idz - zstart;

                            for (int64_t wi = 0; wi < nlon_in; wi++) {
                                int64_t wo = wo_canonical + pscale_out * wi;
                                if (wo >= nlon_out) wo -= nlon_out;

                                const float alpha_nz   = alpha_nz_buf[nz_off(b, hi, idz_local, wi)];
                                const float inv_sum    = 1.0f / alpha_sum_buf[sc_off(b, ho_neigh, wo)];
                                const float alpha_norm = alpha_nz * inv_sum;

                                const float dy_v = static_cast<float>(
                                    dy_base[dy_b_off + ho_neigh * dy_sH + wo * dy_sW + co]);
                                scratch_dvx_row[wi] += alpha_norm * dy_v;
                            }
                        }

                        // flush per-hi dvx scratch
                        for (int64_t wi = 0; wi < nlon_in; wi++) {
                            dvx_base[dvx_b_off + hi * dvx_sH + wi * dvx_sW + co] +=
                                static_cast<scalar_t>(scratch_dvx_row[wi]);
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
