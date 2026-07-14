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

// =====================================================================================
// Gather / self attention forward — cross-wo COOPERATIVE TILING (sm_80+, no WGMMA/TMA).
//
// Motivation (H100 ncu, 180x360 C=256 tc=0.03): the baseline s2_attn_fwd_special_vec_k
// is latency-bound on the col_idx-driven K/V gather (long_scoreboard #1, L1 hit ~52%).
// Adjacent output wo attend nearly the SAME input K/V window (shifted by pscale*wo), so
// each (ho,wo) warp re-gathers redundantly. This kernel tiles WO_TILE consecutive wo of
// one output row into a block; the block stages the UNION K/V longitude arc once into
// shared and every wo reuses it -> ~WO_TILE fewer global loads on the polar rows that
// dominate the work.
//
// Occupancy-conscious (Mauro's TMA attempt tanked occupancy -> much worse): accumulators
// (locy) + query (locq) stay in REGISTERS; only the K/V arc is shared, streamed in
// CHUNK-wide longitude tiles so the shared footprint is small.
//
// Geometry: col_idx is sorted by input lat hi; each (ho,hi) neighbor set is a contiguous
// signed longitude arc [-r, +r] centered on the output longitude (a geodesic disc, no
// holes). For output wo the arc shifts by pscale*wo. Per hi we reduce r over the row's
// neighbors, stage the tile-union arc, and each wo tests membership by de-shifting a
// staged cell back to canonical signed wi and checking |swi| <= r.
//
// Uses accurate expf (NOT __expf) so an A/B vs baseline isolates the TILING effect.
// Assumes nchan_in == nchan_out (self / square-channel gather). fp16/bf16/fp32 storage,
// fp32 compute. Registered as the ``forward_coop`` op.
// =====================================================================================

#include "attention_cuda.cuh"
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>
#include <cfloat>
#include <climits>

#include "cudamacro.h"
#include "attention_cuda_utils.cuh"

namespace attention_kernels
{

    // compacted-row -> ho map, defined in attention_cuda_utils.cu.
    // (declared in attention_cuda_utils.cuh: sortRows)

    namespace coop
    {

        static constexpr int WARP = 32;
        static constexpr int WO_TILE = 4; // output wo positions (warps) per block
        static constexpr int CHUNK = 16;  // input longitudes staged per shared tile

        __device__ __forceinline__ int cmod(int a, int n) { return ((a % n) + n) % n; }
        // map a longitude in [0,n) to signed [-n/2, n/2) so a pole-centred window is contiguous
        __device__ __forceinline__ int csigned(int wi, int n) { return (wi >= (n >> 1)) ? (wi - n) : wi; }

        // block-wide max over ints (blockDim.x = WO_TILE*WARP, WO_TILE <= 32).
        __device__ __forceinline__ int block_max_int(int v, int *sh)
        {
            const int lane = threadIdx.x & (WARP - 1);
            const int wid = threadIdx.x >> 5;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) { v = max(v, __shfl_down_sync(0xffffffffu, v, o)); }
            if (lane == 0) { sh[wid] = v; }
            __syncthreads();
            v = (threadIdx.x < (blockDim.x >> 5)) ? sh[lane] : INT_MIN;
            if (wid == 0) {
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) { v = max(v, __shfl_down_sync(0xffffffffu, v, o)); }
                if (lane == 0) { sh[0] = v; }
            }
            __syncthreads();
            return sh[0];
        }

        // NLOC = ceil(nchan / WARP): per-lane register slots for query / output accumulator.
        template <int NLOC, typename STORAGE_T>
        __global__ __launch_bounds__(WO_TILE *WARP) void s2_attn_fwd_coop_k(
            int nchan, int nlat_in, int nlon_in, int nlat_out, int nlon_out, int tiles_per_row,
            const STORAGE_T *__restrict__ kx, const STORAGE_T *__restrict__ vx, const STORAGE_T *__restrict__ qy,
            const int32_t *__restrict__ row_idx, const int64_t *__restrict__ row_off,
            const int64_t *__restrict__ col_idx, const float *__restrict__ quad_weights, STORAGE_T *__restrict__ y)
        {
            using COMPUTE_T = typename vec_traits<STORAGE_T>::compute_t; // float for the scalar path

            const int lane = threadIdx.x & (WARP - 1);
            const int warp = threadIdx.x >> 5;

            const int batch = blockIdx.y;
            const int brow = blockIdx.x / tiles_per_row;
            const int wtile = blockIdx.x - brow * tiles_per_row;
            if (brow >= nlat_out) { return; }

            const int ho = row_idx[brow];
            const int pscale = nlon_in / nlon_out;
            const int wo = wtile * WO_TILE + warp;
            const bool valid = (wo < nlon_out);

            extern __shared__ __align__(16) char smem[];
            STORAGE_T *shK = reinterpret_cast<STORAGE_T *>(smem);              // [CHUNK * nchan]
            STORAGE_T *shV = shK + size_t(CHUNK) * nchan;                      // [CHUNK * nchan]
            int *shred = reinterpret_cast<int *>(shV + size_t(CHUNK) * nchan); // [WO_TILE] reduction scratch

            // query for this wo, resident in registers (widened once).
            COMPUTE_T locq[NLOC];
            COMPUTE_T locy[NLOC];
#pragma unroll
            for (int i = 0; i < NLOC; i++) { locy[i] = 0.f; }
            if (valid) {
                const STORAGE_T *_qy = qy + (int64_t(batch) * nlat_out * nlon_out + int64_t(ho) * nlon_out + wo) * nchan;
#pragma unroll
                for (int i = 0; i < NLOC; i++) {
                    const int ch = lane + i * WARP;
                    locq[i] = (ch < nchan) ? vload(_qy, ch) : COMPUTE_T(0.f);
                }
            }

            float qmax = -FLT_MAX;
            float asum = 0.f;

            const int64_t rbeg = row_off[ho];
            const int64_t rend = row_off[ho + 1];
            const int rlen = int(rend - rbeg);
            if (rlen <= 0) { return; }

            const int hi_min = int(col_idx[rbeg] / nlon_in);
            const int hi_max = int(col_idx[rend - 1] / nlon_in);

            for (int hi = hi_min; hi <= hi_max; hi++) {

                // radius r for this hi = max |signed wi| over the row's neighbors at this hi.
                int loc = -1;
                for (int off = threadIdx.x; off < rlen; off += blockDim.x) {
                    const int64_t c = col_idx[rbeg + off];
                    if (int(c / nlon_in) == hi) { loc = max(loc, abs(csigned(int(c % nlon_in), nlon_in))); }
                }
                const int r = block_max_int(loc, shred);
                if (r < 0) { continue; } // no neighbors at this hi (shouldn't happen for a contiguous band)

                const int arc_lo = -r + pscale * (wtile * WO_TILE); // signed abs longitude of arc start
                const int arc_w = min(nlon_in, 2 * r + pscale * (WO_TILE - 1) + 1);
                const float qw = quad_weights[hi];
                const STORAGE_T *kx_hi = kx + (int64_t(batch) * nlat_in + hi) * nlon_in * nchan;
                const STORAGE_T *vx_hi = vx + (int64_t(batch) * nlat_in + hi) * nlon_in * nchan;

                for (int c0 = 0; c0 < arc_w; c0 += CHUNK) {
                    const int cw = min(CHUNK, arc_w - c0);

                    // stage cw input longitudes (K and V) into shared, coalesced along channel.
                    for (int idx = threadIdx.x; idx < cw * nchan; idx += blockDim.x) {
                        const int j = idx / nchan;
                        const int ch = idx - j * nchan;
                        const int lon = cmod(arc_lo + c0 + j, nlon_in);
                        const int64_t g = int64_t(lon) * nchan + ch;
                        shK[j * nchan + ch] = kx_hi[g];
                        shV[j * nchan + ch] = vx_hi[g];
                    }
                    __syncthreads();

                    if (valid) {
                        for (int j = 0; j < cw; j++) {
                            const int lon = cmod(arc_lo + c0 + j, nlon_in);
                            const int swi = csigned(cmod(lon - pscale * wo, nlon_in), nlon_in);
                            if (abs(swi) > r) { continue; } // not a neighbor of this wo (disc membership)

                            const STORAGE_T *shk = shK + j * nchan;
                            const STORAGE_T *shv = shV + j * nchan;

                            COMPUTE_T acc = 0.f;
#pragma unroll
                            for (int i = 0; i < NLOC; i++) {
                                const int ch = lane + i * WARP;
                                if (ch < nchan) { acc = __vfma(locq[i], vload(shk, ch), acc); }
                            }
                            float qdotk = __warp_sum(acc);

                            const float qmax2 = max(qmax, qdotk);
                            const float alpha = expf(qdotk - qmax2) * qw;
                            const float corr = expf(qmax - qmax2);
                            asum = alpha + asum * corr;
#pragma unroll
                            for (int i = 0; i < NLOC; i++) {
                                const int ch = lane + i * WARP;
                                if (ch < nchan) {
                                    locy[i] = __vfma_scale(alpha, vload(shv, ch), __vscale(corr, locy[i]));
                                }
                            }
                            qmax = qmax2;
                        }
                    }
                    __syncthreads();
                }
            }

            if (valid) {
                const float inv = (asum > 0.f) ? 1.f / asum : 0.f;
                STORAGE_T *_y = y + (int64_t(batch) * nlat_out * nlon_out + int64_t(ho) * nlon_out + wo) * nchan;
#pragma unroll
                for (int i = 0; i < NLOC; i++) {
                    const int ch = lane + i * WARP;
                    if (ch < nchan) { vstore(_y, ch, __vscale(inv, locy[i])); }
                }
            }
        }

        // ---- launcher: pick NLOC = ceil(nchan/WARP) via a compile-time recursion ----
        template <int NLOC, typename STORAGE_T>
        void launch_coop(int nloc, int nchan, int nlat_in, int nlon_in, int nlat_out, int nlon_out, int tiles_per_row,
                         int batch_size, const STORAGE_T *kx, const STORAGE_T *vx, const STORAGE_T *qy,
                         const int32_t *row_idx, const int64_t *row_off, const int64_t *col_idx, const float *qw,
                         STORAGE_T *y, cudaStream_t stream)
        {
            if (NLOC == nloc) {
                dim3 block(WO_TILE * WARP);
                dim3 grid(nlat_out * tiles_per_row, batch_size);
                size_t shbytes = size_t(2) * CHUNK * nchan * sizeof(STORAGE_T) + WO_TILE * sizeof(int);
                s2_attn_fwd_coop_k<NLOC, STORAGE_T>
                    <<<grid, block, shbytes, stream>>>(nchan, nlat_in, nlon_in, nlat_out, nlon_out, tiles_per_row, kx,
                                                       vx, qy, row_idx, row_off, col_idx, qw, y);
                CHECK_ERROR("s2_attn_fwd_coop_k");
                return;
            }
            if constexpr (NLOC < 32) {
                launch_coop<NLOC + 1, STORAGE_T>(nloc, nchan, nlat_in, nlon_in, nlat_out, nlon_out, tiles_per_row,
                                                 batch_size, kx, vx, qy, row_idx, row_off, col_idx, qw, y, stream);
            }
        }

        torch::Tensor s2_attention_fwd_coop_cuda(at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor quad_weights,
                                                 at::Tensor col_idx, at::Tensor row_off, int64_t nlon_in,
                                                 int64_t nlat_out, int64_t nlon_out)
        {
            CHECK_CUDA_INPUT_TENSOR(kx);
            CHECK_CUDA_INPUT_TENSOR(vx);
            CHECK_CUDA_INPUT_TENSOR(qy);
            CHECK_CUDA_TENSOR(quad_weights);
            CHECK_CUDA_TENSOR(col_idx);
            CHECK_CUDA_TENSOR(row_off);

            const int64_t nchans_in = qy.size(1);
            const int64_t nchans_out = vx.size(1);
            TORCH_CHECK(nchans_in == nchans_out, "forward_coop: requires nchan_in == nchan_out (self/square-channel)");
            TORCH_CHECK(nlon_in % nlon_out == 0, "forward_coop: gather path only (nlon_in % nlon_out == 0)");
            const int nchan = int(nchans_in);
            TORCH_CHECK(nchan <= 32 * 32, "forward_coop: nchan too large for register accumulator");

            const int batch_size = kx.size(0);
            const int64_t nlat_in = kx.size(2);
            auto qy_type = qy.dtype();
            auto stream = at::cuda::getCurrentCUDAStream().stream();

            at::Tensor row_idx = sortRows(int(nlat_out), row_off, stream);

            // channels-last (NHWC) as the kernels index it.
            torch::Tensor kxP = kx, vxP = vx, qyP = qy;
            const bool qy_cl = (qyP.strides()[1] == 1);
            if (kxP.strides()[1] != 1) { kxP = permute_4D_to0231(kxP); }
            if (vxP.strides()[1] != 1) { vxP = permute_4D_to0231(vxP); }
            if (!qy_cl) { qyP = permute_4D_to0231(qyP); }

            const int64_t out_dims[] = {batch_size, nlat_out, nlon_out, nchans_out};
            torch::Tensor yP = torch::empty(out_dims, kxP.options());

            const int tiles_per_row = int((nlon_out + WO_TILE - 1) / WO_TILE);
            const int nloc = (nchan + WARP - 1) / WARP;

            const int32_t *_row_idx = reinterpret_cast<const int32_t *>(row_idx.data_ptr());
            const int64_t *_row_off = reinterpret_cast<const int64_t *>(row_off.data_ptr());
            const int64_t *_col_idx = reinterpret_cast<const int64_t *>(col_idx.data_ptr());
            const float *_qw = reinterpret_cast<const float *>(quad_weights.data_ptr());

            AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, qyP.scalar_type(), "s2_attention_fwd_coop_cuda", [&] {
                launch_coop<1, scalar_t>(nloc, nchan, int(nlat_in), int(nlon_in), int(nlat_out), int(nlon_out),
                                         tiles_per_row, batch_size, reinterpret_cast<const scalar_t *>(kxP.data_ptr()),
                                         reinterpret_cast<const scalar_t *>(vxP.data_ptr()),
                                         reinterpret_cast<const scalar_t *>(qyP.data_ptr()), _row_idx, _row_off,
                                         _col_idx, _qw, reinterpret_cast<scalar_t *>(yP.data_ptr()), stream);
            });

            C10_CUDA_KERNEL_LAUNCH_CHECK();

            torch::Tensor yout = yP;
            if (!qy_cl) { yout = permute_4D_to0312(yout); }
            return yout.to(qy_type);
        }

        TORCH_LIBRARY_IMPL(attention_kernels, CUDA, m) { m.impl("forward_coop", &s2_attention_fwd_coop_cuda); }

    } // namespace coop

} // namespace attention_kernels
