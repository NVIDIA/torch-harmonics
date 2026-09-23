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

#include "../../attention.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace attention_kernels
{

    // -----------------------------------------------------------------------
    // Neighborhood attention on a ragged grid, for CPU.
    //
    // This mirrors kernels_cuda/ragged/ rather than kernels_cpu/regular/. The
    // product-grid CPU kernels take the CSR column list and ignore the arcs the
    // CUDA kernels consume, so on that path the two devices read different
    // descriptions of the same neighborhood and each has to be maintained against
    // its own. There is no reason to repeat that here: the arcs are what the
    // precompute produces natively, they are the smaller of the two forms, and
    // sharing them means one backend, one set of buffers, and a CPU/CUDA
    // disagreement can only ever be arithmetic rather than structural.
    //
    // Decoding an arc is the same three lines as on the GPU. A segment is
    // (iring, lo, len); a ring is contiguous in ring-major order, so the flat
    // column is ring_base[iring] + lo, counting up and wrapping once at the ring's
    // end -- which is what the arc encoding exists to make possible without a
    // modulo. Every point of a ring carries the same quadrature weight, so
    // ring_weights[iring] hoists out of the inner loop.
    //
    // What is deliberately *not* mirrored is the GPU's execution shape. There are
    // no warps to reduce across and no shared memory to stage through, so the
    // channel loops are plain and the softmax state lives in registers. The
    // parallel axis is (batch * head, output point), scheduled dynamically: on a
    // ragged grid the neighbor count varies with the ring -- on HEALPix by a
    // factor of 4 * nside between the poles and the equator -- so a static split
    // would leave threads that drew polar points idle. This is the one place the
    // ragged CPU kernel is better balanced than the product-grid one, whose
    // collapse(3) assumes every row costs the same.
    //
    // fp32 only, like the rest of kernels_cpu: reduced-precision compute is
    // emulated on CPU, so the launcher upcasts and casts the result back.
    // -----------------------------------------------------------------------

    // Fold the packed head axis into the batch dimension, for a flat point grid.
    //
    // Inputs are physical (B, npoints, num_heads * C); the kernels below are
    // head-agnostic and want (B * num_heads, npoints, C). As on the product-grid
    // CPU path this costs a materialization -- in a channel-innermost layout the
    // head axis is interior and no view can bring it to the front -- and is worth
    // one copy to keep the loop bodies free of head bookkeeping.
    inline at::Tensor fold_heads_ragged(const at::Tensor &t, int64_t num_heads)
    {
        if (num_heads == 1) { return t; }
        const int64_t B = t.size(0), N = t.size(1), C = t.size(2) / num_heads;
        return t.reshape({B, N, num_heads, C}).permute({0, 2, 1, 3}).reshape({B * num_heads, N, C});
    }

    inline at::Tensor unfold_heads_ragged(const at::Tensor &t, int64_t num_heads)
    {
        if (num_heads == 1) { return t; }
        const int64_t BH = t.size(0), N = t.size(1), C = t.size(2);
        const int64_t B = BH / num_heads;
        return t.reshape({B, num_heads, N, C}).permute({0, 2, 1, 3}).reshape({B, N, num_heads * C});
    }

    // Forward. Produces y together with the softmax bookkeeping the backward
    // consumes: alpha_sum as accumulated (before the reciprocal) and qdotk_max as
    // the walk left it, which is the same contract the CUDA kernel documents.
    inline void s2_attn_fwd_ragged_cpu_kernel(const float *__restrict__ kx, const float *__restrict__ vx,
                                              const float *__restrict__ qy, const float *__restrict__ ring_weights,
                                              const int32_t *__restrict__ seg, const int32_t *__restrict__ seg_off,
                                              const int64_t *__restrict__ ring_base,
                                              const int64_t *__restrict__ ring_size, float *__restrict__ y,
                                              float *__restrict__ alpha_sum_out, float *__restrict__ qdotk_max_out,
                                              const int64_t nbatch_heads, const int64_t npoints_in,
                                              const int64_t npoints_out, const int64_t nchan_in, const int64_t nchan_out)
    {
#pragma omp parallel
        {
            std::vector<float> acc(nchan_out);

#pragma omp for collapse(2) schedule(dynamic, 8)
            for (int64_t bh = 0; bh < nbatch_heads; bh++) {
                for (int64_t ipoint = 0; ipoint < npoints_out; ipoint++) {

                    const float *__restrict__ qy_p = qy + (bh * npoints_out + ipoint) * nchan_in;
                    const float *__restrict__ kx_b = kx + bh * npoints_in * nchan_in;
                    const float *__restrict__ vx_b = vx + bh * npoints_in * nchan_out;

                    std::fill(acc.begin(), acc.end(), 0.0f);
                    float alpha_sum = 0.0f;
                    float qdotk_max = -std::numeric_limits<float>::max();

                    for (int32_t sg = seg_off[ipoint]; sg < seg_off[ipoint + 1]; sg++) {

                        const int64_t iring = seg[3 * sg + 0];
                        const int64_t lo = seg[3 * sg + 1];
                        const int64_t len = seg[3 * sg + 2];

                        const float qw = ring_weights[iring];
                        const int64_t ring_lo = ring_base[iring];
                        const int64_t ring_hi = ring_lo + ring_size[iring];

                        int64_t col = ring_lo + lo;

                        for (int64_t j = 0; j < len; j++) {

                            const float *__restrict__ kx_p = kx_b + col * nchan_in;
                            const float *__restrict__ vx_p = vx_b + col * nchan_out;

                            float qdotk = 0.0f;
                            for (int64_t c = 0; c < nchan_in; c++) { qdotk += qy_p[c] * kx_p[c]; }

                            // online softmax: rescale the running numerator and
                            // denominator whenever a neighbour raises the maximum
                            const float qdotk_max_tmp = std::max(qdotk_max, qdotk);
                            const float alpha = std::exp(qdotk - qdotk_max_tmp) * qw;
                            const float exp_save = std::exp(qdotk_max - qdotk_max_tmp);

                            alpha_sum = alpha + alpha_sum * exp_save;
                            for (int64_t c = 0; c < nchan_out; c++) { acc[c] = acc[c] * exp_save + alpha * vx_p[c]; }
                            qdotk_max = qdotk_max_tmp;

                            // wraps at most once, so compare-and-subtract stands in
                            // for the modulo the arc encoding avoids
                            if (++col == ring_hi) { col = ring_lo; }
                        }
                    }

                    const int64_t istat = bh * npoints_out + ipoint;
                    alpha_sum_out[istat] = alpha_sum;
                    qdotk_max_out[istat] = qdotk_max;

                    const float inv = 1.0f / alpha_sum;
                    float *__restrict__ y_p = y + istat * nchan_out;
                    for (int64_t c = 0; c < nchan_out; c++) { y_p[c] = acc[c] * inv; }
                }
            }
        }
    }

    // Backward.
    //
    // Replays the forward's weights from the saved statistics rather than storing
    // one alpha per neighbour, exactly as the CUDA kernel does: with qdotk_max and
    // alpha_sum in hand a second walk reproduces every alpha_pj without a second
    // pass over the maximum.
    //
    //   dv_j  = sum_p (alpha_pj / A_p) dy_p
    //   dqk_pj = (alpha_pj / A_p) (dy_p . v_j - integral_p),  integral_p = dy_p . y_p
    //   dq_p  = sum_j dqk_pj k_j
    //   dk_j  = sum_p dqk_pj q_p
    //
    // dq is keyed by the output point, and the loop below owns one output point per
    // iteration, so those writes need no synchronization at all. dk and dv are keyed
    // by the *input* point and neighborhoods overlap, so they are accumulated
    // atomically -- the same scatter the CUDA kernel does with atomicAdd.
    //
    // Per-thread buffers summed at the end would avoid the atomics, but they cost
    // nbatch_heads * npoints_in * nchan floats *per thread*: at nside=64 with 64
    // threads that is hundreds of megabytes allocated invisibly inside a backward
    // pass. Being slower is a better failure than being unable to run, and this path
    // exists for CI and small grids.
    inline void s2_attn_bwd_ragged_cpu_kernel(
        const float *__restrict__ kx, const float *__restrict__ vx, const float *__restrict__ qy,
        const float *__restrict__ dy, const float *__restrict__ integral, const float *__restrict__ alpha_sum_in,
        const float *__restrict__ qdotk_max_in, const float *__restrict__ ring_weights, const int32_t *__restrict__ seg,
        const int32_t *__restrict__ seg_off, const int64_t *__restrict__ ring_base, const int64_t *__restrict__ ring_size,
        // dkx and dvx are deliberately not __restrict__: threads scatter into overlapping
        // regions of them, which is exactly the aliasing that qualifier promises does not
        // happen. dqy is, because each iteration owns its slice outright.
        float *dkx, float *dvx, float *__restrict__ dqy, const int64_t nbatch_heads, const int64_t npoints_in,
        const int64_t npoints_out, const int64_t nchan_in, const int64_t nchan_out)
    {
#pragma omp parallel
        {
#pragma omp for collapse(2) schedule(dynamic, 8)
            for (int64_t bh = 0; bh < nbatch_heads; bh++) {
                for (int64_t ipoint = 0; ipoint < npoints_out; ipoint++) {

                    const int64_t istat = bh * npoints_out + ipoint;
                    const float *__restrict__ qy_p = qy + istat * nchan_in;
                    const float *__restrict__ dy_p = dy + istat * nchan_out;
                    float *__restrict__ dqy_p = dqy + istat * nchan_in;

                    const float *__restrict__ kx_b = kx + bh * npoints_in * nchan_in;
                    const float *__restrict__ vx_b = vx + bh * npoints_in * nchan_out;
                    float *dk_b = dkx + bh * npoints_in * nchan_in;
                    float *dv_b = dvx + bh * npoints_in * nchan_out;

                    const float inv_sum = 1.0f / alpha_sum_in[istat];
                    const float qdotk_max = qdotk_max_in[istat];
                    const float integ = integral[istat];

                    for (int32_t sg = seg_off[ipoint]; sg < seg_off[ipoint + 1]; sg++) {

                        const int64_t iring = seg[3 * sg + 0];
                        const int64_t lo = seg[3 * sg + 1];
                        const int64_t len = seg[3 * sg + 2];

                        const float qw = ring_weights[iring];
                        const int64_t ring_lo = ring_base[iring];
                        const int64_t ring_hi = ring_lo + ring_size[iring];

                        int64_t col = ring_lo + lo;

                        for (int64_t j = 0; j < len; j++) {

                            const float *__restrict__ kx_p = kx_b + col * nchan_in;
                            const float *__restrict__ vx_p = vx_b + col * nchan_out;

                            float qdotk = 0.0f;
                            for (int64_t c = 0; c < nchan_in; c++) { qdotk += qy_p[c] * kx_p[c]; }

                            // the forward's alpha, reconstructed from the saved maximum
                            const float alpha_norm = std::exp(qdotk - qdotk_max) * qw * inv_sum;

                            float dy_dot_v = 0.0f;
                            for (int64_t c = 0; c < nchan_out; c++) { dy_dot_v += dy_p[c] * vx_p[c]; }

                            const float dqk = alpha_norm * (dy_dot_v - integ);

                            float *dk_p = dk_b + col * nchan_in;
                            float *dv_p = dv_b + col * nchan_out;
                            for (int64_t c = 0; c < nchan_in; c++) {
                                // shared across output points, hence atomic
#pragma omp atomic
                                dk_p[c] += dqk * qy_p[c];
                                // this output point's own gradient, hence not
                                dqy_p[c] += dqk * kx_p[c];
                            }
                            for (int64_t c = 0; c < nchan_out; c++) {
#pragma omp atomic
                                dv_p[c] += alpha_norm * dy_p[c];
                            }

                            if (++col == ring_hi) { col = ring_lo; }
                        }
                    }
                }
            }
        }
    }

} // namespace attention_kernels
