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
// Upsample (scatter-style) attention backward, RING-STEP variants — CUDA
// =====================================================================================
//
// Ring counterpart of attention_cuda_bwd_upsample.cu, used by
// DistributedNeighborhoodAttentionS2 in the upsample direction. See
// attention_cuda_fwd_ring_upsample.cu for the psi / index conventions (local
// halo-keyed rows, local-output-keyed cols, wo pre-shifted by -lon_lo_out).
//
// Unlike the serial upsample backward, no max-recomputation pass is needed:
// the forward saves the FINAL (globally reduced) alpha_sum and qdotk_max
// buffers, which are passed back in here. That leaves two ring sweeps:
//
//   pass 1 (stats): per coarse chunk cell, scatter with the fixed forward max
//                   alpha       -> (implicitly known: forward alpha_sum)
//                   alpha*g     -> atomicAdd integral_buf[b,ho,wo]     (g = dy.v)
//                   alpha*k     -> atomicAdd alpha_k_buf[b,ho,wo,:]
//                   alpha*g*k   -> atomicAdd alpha_kvw_buf[b,ho,wo,:]
//                   dqy is finalized in Python from these buffers:
//                   dqy = (alpha_sum*alpha_kvw - integral*alpha_k) / alpha_sum^2
//
//   pass 2 (dkdv):  per coarse chunk cell, accumulate locally over its row
//                   dvx += (alpha/S)*dy and dkx += q*(g - integral_norm)*(alpha/S)
//                   with integral_norm = integral/S precomputed in Python.
//                   dkx/dvx are chunk-shaped, direct write (no atomics); Python
//                   accumulates chunks into a full-width buffer and allreduces
//                   over the azimuth group.
//
// Generic-only (scalar loads): correctness path.
// =====================================================================================

#include "attention_cuda.cuh"
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>
#include <cfloat>

#include "cudamacro.h"
#include "attention_cuda_utils.cuh"

#define THREADS (64)

namespace attention_kernels
{

    // map (wo_shifted, wi_global) -> local output longitude; returns -1 if the
    // target cell is not owned by this rank.
    __device__ __forceinline__ int bwd_ring_up_local_wo(int wo_shifted, int wi_global, int pscale_out,
                                                        int nlon_out_global, int nlon_out_local)
    {
        int w = wo_shifted + pscale_out * wi_global; // < 2*nlon_out_global
        if (w >= nlon_out_global) { w -= nlon_out_global; }
        return (w < nlon_out_local) ? w : -1;
    }

    // pass 1: per coarse chunk cell, scatter the softmax stats (integral, alpha_k,
    // alpha_kvw) to the local fine output cells, using the saved forward max.
    template <int THREADS_PER_BLOCK, typename STORAGE_T>
    __global__ __launch_bounds__(THREADS_PER_BLOCK) void s2_attn_bwd_ring_upsample_stats_k(
        int nchan_in, int nchan_out, int nlat_halo, int nlon_kx, int nlon_out_global, int pscale_out, int lon_lo_kx,
        int lat_halo_start, int nlat_out, int nlon_out, const STORAGE_T *__restrict__ kx,
        const STORAGE_T *__restrict__ vx, const STORAGE_T *__restrict__ qy, const STORAGE_T *__restrict__ dy,
        const int64_t *__restrict__ row_off, const int64_t *__restrict__ col_idx,
        const float *__restrict__ quad_weights, const float *__restrict__ qdotk_max_buf,
        float *__restrict__ integral_buf, float *__restrict__ alpha_k_buf, float *__restrict__ alpha_kvw_buf)
    {
        extern __shared__ float shext[];
        float *sh_k = shext + threadIdx.y * (nchan_in + nchan_out);
        float *sh_v = sh_k + nchan_in;

        const int batch = blockIdx.y;
        const int wid = blockIdx.x * blockDim.y + threadIdx.y;
        if (wid >= nlat_halo * nlon_kx) { return; }

        const int tidx = threadIdx.x;
        const int hi = wid / nlon_kx; // LOCAL halo row
        const int wi_local = wid - hi * nlon_kx;
        const int wi_global = lon_lo_kx + wi_local;

        const int64_t rbeg = row_off[hi];
        const int rlen = static_cast<int>(row_off[hi + 1] - rbeg);
        if (rlen == 0) { return; } // empty row (e.g. pole padding)
        const int64_t *col_hi = col_idx + rbeg;

        kx += int64_t(batch) * nlat_halo * nlon_kx * nchan_in + (int64_t(hi) * nlon_kx + wi_local) * nchan_in;
        vx += int64_t(batch) * nlat_halo * nlon_kx * nchan_out + (int64_t(hi) * nlon_kx + wi_local) * nchan_out;
        qy += int64_t(batch) * nlat_out * nlon_out * nchan_in;
        dy += int64_t(batch) * nlat_out * nlon_out * nchan_out;
        qdotk_max_buf += int64_t(batch) * nlat_out * nlon_out;
        integral_buf += int64_t(batch) * nlat_out * nlon_out;
        alpha_k_buf += int64_t(batch) * nlat_out * nlon_out * nchan_in;
        alpha_kvw_buf += int64_t(batch) * nlat_out * nlon_out * nchan_in;

        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) { sh_k[chan] = vload(kx, chan); }
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { sh_v[chan] = vload(vx, chan); }

        // quad_weights is indexed by the GLOBAL input latitude
        const float qw = quad_weights[lat_halo_start + hi];

        for (int off = 0; off < rlen; off++) {
            const int64_t col = col_hi[off];
            const int ho = static_cast<int>(col / nlon_out_global); // LOCAL output row
            const int wo = bwd_ring_up_local_wo(static_cast<int>(col - int64_t(ho) * nlon_out_global), wi_global,
                                                pscale_out, nlon_out_global, nlon_out);
            if (wo < 0) { continue; } // target cell not owned by this rank

            const int64_t cell = int64_t(ho) * nlon_out + wo;
            const STORAGE_T *_qy = qy + cell * nchan_in;
            const STORAGE_T *_dy = dy + cell * nchan_out;

            float qd = 0.f, gd = 0.f;
            for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) { qd += sh_k[chan] * vload(_qy, chan); }
            for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { gd += sh_v[chan] * vload(_dy, chan); }
            qd = __warp_sum(qd);
            gd = __warp_sum(gd);

            const float alpha = expf(qd - qdotk_max_buf[cell]) * qw;
            const float ag = alpha * gd;
            if (tidx == 0) { atomicAdd(&integral_buf[cell], ag); }
            float *_alpha_k = alpha_k_buf + cell * nchan_in;
            float *_alpha_kvw = alpha_kvw_buf + cell * nchan_in;
            for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
                atomicAdd(&_alpha_k[chan], alpha * sh_k[chan]);
                atomicAdd(&_alpha_kvw[chan], ag * sh_k[chan]);
            }
        }
    }

    // pass 2: per coarse chunk cell, accumulate dkx/dvx locally over its row and
    // write into the chunk-shaped gradient buffers (no atomics).
    template <int THREADS_PER_BLOCK, typename STORAGE_T>
    __global__ __launch_bounds__(THREADS_PER_BLOCK) void s2_attn_bwd_ring_upsample_dkv_k(
        int nchan_in, int nchan_out, int nlat_halo, int nlon_kx, int nlon_out_global, int pscale_out, int lon_lo_kx,
        int lat_halo_start, int nlat_out, int nlon_out, const STORAGE_T *__restrict__ kx,
        const STORAGE_T *__restrict__ vx, const STORAGE_T *__restrict__ qy, const STORAGE_T *__restrict__ dy,
        const int64_t *__restrict__ row_off, const int64_t *__restrict__ col_idx, const float *__restrict__ quad_weights,
        const float *__restrict__ alpha_sum_buf, const float *__restrict__ qdotk_max_buf,
        const float *__restrict__ integral_norm_buf, float *__restrict__ dkx, float *__restrict__ dvx)
    {
        extern __shared__ float shext[];
        float *sh_k = shext + threadIdx.y * (2 * nchan_in + 2 * nchan_out);
        float *sh_v = sh_k + nchan_in;
        float *sh_dk = sh_v + nchan_out;
        float *sh_dv = sh_dk + nchan_in;

        const int batch = blockIdx.y;
        const int wid = blockIdx.x * blockDim.y + threadIdx.y;
        if (wid >= nlat_halo * nlon_kx) { return; }

        const int tidx = threadIdx.x;
        const int hi = wid / nlon_kx; // LOCAL halo row
        const int wi_local = wid - hi * nlon_kx;
        const int wi_global = lon_lo_kx + wi_local;

        const int64_t rbeg = row_off[hi];
        const int rlen = static_cast<int>(row_off[hi + 1] - rbeg);
        if (rlen == 0) { return; } // empty row: dkx/dvx stay zero (buffers pre-zeroed)
        const int64_t *col_hi = col_idx + rbeg;

        kx += int64_t(batch) * nlat_halo * nlon_kx * nchan_in + (int64_t(hi) * nlon_kx + wi_local) * nchan_in;
        vx += int64_t(batch) * nlat_halo * nlon_kx * nchan_out + (int64_t(hi) * nlon_kx + wi_local) * nchan_out;
        qy += int64_t(batch) * nlat_out * nlon_out * nchan_in;
        dy += int64_t(batch) * nlat_out * nlon_out * nchan_out;
        alpha_sum_buf += int64_t(batch) * nlat_out * nlon_out;
        qdotk_max_buf += int64_t(batch) * nlat_out * nlon_out;
        integral_norm_buf += int64_t(batch) * nlat_out * nlon_out;
        dkx += int64_t(batch) * nlat_halo * nlon_kx * nchan_in + (int64_t(hi) * nlon_kx + wi_local) * nchan_in;
        dvx += int64_t(batch) * nlat_halo * nlon_kx * nchan_out + (int64_t(hi) * nlon_kx + wi_local) * nchan_out;

        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
            sh_k[chan] = vload(kx, chan);
            sh_dk[chan] = 0.f;
        }
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
            sh_v[chan] = vload(vx, chan);
            sh_dv[chan] = 0.f;
        }

        // quad_weights is indexed by the GLOBAL input latitude
        const float qw = quad_weights[lat_halo_start + hi];

        for (int off = 0; off < rlen; off++) {
            const int64_t col = col_hi[off];
            const int ho = static_cast<int>(col / nlon_out_global); // LOCAL output row
            const int wo = bwd_ring_up_local_wo(static_cast<int>(col - int64_t(ho) * nlon_out_global), wi_global,
                                                pscale_out, nlon_out_global, nlon_out);
            if (wo < 0) { continue; } // target cell not owned by this rank

            const int64_t cell = int64_t(ho) * nlon_out + wo;
            const STORAGE_T *_qy = qy + cell * nchan_in;
            const STORAGE_T *_dy = dy + cell * nchan_out;

            float qd = 0.f, gd = 0.f;
            for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) { qd += sh_k[chan] * vload(_qy, chan); }
            for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { gd += sh_v[chan] * vload(_dy, chan); }
            qd = __warp_sum(qd);
            gd = __warp_sum(gd);

            const float alpha_norm = expf(qd - qdotk_max_buf[cell]) * qw / alpha_sum_buf[cell];
            const float scale_dk = (gd - integral_norm_buf[cell]) * alpha_norm;

            for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { sh_dv[chan] += alpha_norm * vload(_dy, chan); }
            for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) { sh_dk[chan] += scale_dk * vload(_qy, chan); }
        }

        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) { dkx[chan] = sh_dk[chan]; }
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { dvx[chan] = sh_dv[chan]; }
    }

    void s2_attention_bwd_ring_step_upsample_pass1_cuda(at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor dy,
                                                        at::Tensor qdotk_max_buf, at::Tensor integral_buf,
                                                        at::Tensor alpha_k_buf, at::Tensor alpha_kvw_buf,
                                                        at::Tensor quad_weights, at::Tensor psi_col_idx,
                                                        at::Tensor psi_row_off, int64_t nlon_in,
                                                        int64_t nlon_out_global, int64_t pscale_out, int64_t lon_lo_kx,
                                                        int64_t lat_halo_start, int64_t nlat_out, int64_t nlon_out)
    {
        CHECK_CUDA_INPUT_TENSOR(kx);
        CHECK_CUDA_INPUT_TENSOR(vx);
        CHECK_CUDA_INPUT_TENSOR(qy);
        CHECK_CUDA_INPUT_TENSOR(dy);
        CHECK_CUDA_TENSOR(qdotk_max_buf);
        CHECK_CUDA_TENSOR(integral_buf);
        CHECK_CUDA_TENSOR(alpha_k_buf);
        CHECK_CUDA_TENSOR(alpha_kvw_buf);
        CHECK_CUDA_TENSOR(quad_weights);
        CHECK_CUDA_TENSOR(psi_col_idx);
        CHECK_CUDA_TENSOR(psi_row_off);

        // NHWC by contract: kx/vx/qy (and dy) are physical (B, H, W, C), which is
        // the layout these kernels address. The conversion happens once at the ring
        // boundary rather than per step -- see distributed_attention.py.
        const int batch_size = kx.size(0);
        const int nlat_halo = kx.size(1);
        const int nlon_kx = kx.size(2);
        const size_t nchans_in = qy.size(3);
        const size_t nchans_out = vx.size(3);

        auto stream = at::cuda::getCurrentCUDAStream().stream();

        int64_t *_row_off = reinterpret_cast<int64_t *>(psi_row_off.data_ptr());
        int64_t *_col_idx = reinterpret_cast<int64_t *>(psi_col_idx.data_ptr());
        float *_quad_weights = reinterpret_cast<float *>(quad_weights.data_ptr());
        float *_qdotk_max = reinterpret_cast<float *>(qdotk_max_buf.data_ptr());
        float *_integral = reinterpret_cast<float *>(integral_buf.data_ptr());
        float *_alpha_k = reinterpret_cast<float *>(alpha_k_buf.data_ptr());
        float *_alpha_kvw = reinterpret_cast<float *>(alpha_kvw_buf.data_ptr());

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, qy.scalar_type(), "s2_attention_bwd_ring_step_upsample_pass1_cuda", [&] {
                using storage_t = scalar_t;

                torch::Tensor kxP = kx;
                torch::Tensor vxP = vx;
                torch::Tensor qyP = qy;
                torch::Tensor dyP = dy;

                storage_t *_kxp = reinterpret_cast<storage_t *>(kxP.data_ptr());
                storage_t *_vxp = reinterpret_cast<storage_t *>(vxP.data_ptr());
                storage_t *_qyp = reinterpret_cast<storage_t *>(qyP.data_ptr());
                storage_t *_dyp = reinterpret_cast<storage_t *>(dyP.data_ptr());

                dim3 block(WARP_SIZE, THREADS / WARP_SIZE);
                dim3 grid_in(DIV_UP(nlat_halo * nlon_kx, block.y), batch_size);

                const size_t sh_stats = sizeof(float) * (nchans_in + nchans_out) * block.y;

                s2_attn_bwd_ring_upsample_stats_k<THREADS><<<grid_in, block, sh_stats, stream>>>(
                    static_cast<int>(nchans_in), static_cast<int>(nchans_out), nlat_halo, nlon_kx,
                    static_cast<int>(nlon_out_global), static_cast<int>(pscale_out), static_cast<int>(lon_lo_kx),
                    static_cast<int>(lat_halo_start), static_cast<int>(nlat_out), static_cast<int>(nlon_out), _kxp,
                    _vxp, _qyp, _dyp, _row_off, _col_idx, _quad_weights, _qdotk_max, _integral, _alpha_k, _alpha_kvw);
                CHECK_ERROR("s2_attn_bwd_ring_upsample_stats_k");
            });

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    void s2_attention_bwd_ring_step_upsample_pass2_cuda(at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor dy,
                                                        at::Tensor alpha_sum_buf, at::Tensor qdotk_max_buf,
                                                        at::Tensor integral_norm_buf, at::Tensor dkx, at::Tensor dvx,
                                                        at::Tensor quad_weights, at::Tensor psi_col_idx,
                                                        at::Tensor psi_row_off, int64_t nlon_in,
                                                        int64_t nlon_out_global, int64_t pscale_out, int64_t lon_lo_kx,
                                                        int64_t lat_halo_start, int64_t nlat_out, int64_t nlon_out)
    {
        CHECK_CUDA_INPUT_TENSOR(kx);
        CHECK_CUDA_INPUT_TENSOR(vx);
        CHECK_CUDA_INPUT_TENSOR(qy);
        CHECK_CUDA_INPUT_TENSOR(dy);
        CHECK_CUDA_TENSOR(alpha_sum_buf);
        CHECK_CUDA_TENSOR(qdotk_max_buf);
        CHECK_CUDA_TENSOR(integral_norm_buf);
        CHECK_CUDA_TENSOR(dkx);
        CHECK_CUDA_TENSOR(dvx);
        CHECK_CUDA_TENSOR(quad_weights);
        CHECK_CUDA_TENSOR(psi_col_idx);
        CHECK_CUDA_TENSOR(psi_row_off);

        // NHWC by contract: kx/vx/qy (and dy) are physical (B, H, W, C), which is
        // the layout these kernels address. The conversion happens once at the ring
        // boundary rather than per step -- see distributed_attention.py.
        const int batch_size = kx.size(0);
        const int nlat_halo = kx.size(1);
        const int nlon_kx = kx.size(2);
        const size_t nchans_in = qy.size(3);
        const size_t nchans_out = vx.size(3);

        auto stream = at::cuda::getCurrentCUDAStream().stream();

        int64_t *_row_off = reinterpret_cast<int64_t *>(psi_row_off.data_ptr());
        int64_t *_col_idx = reinterpret_cast<int64_t *>(psi_col_idx.data_ptr());
        float *_quad_weights = reinterpret_cast<float *>(quad_weights.data_ptr());
        float *_alpha_sum = reinterpret_cast<float *>(alpha_sum_buf.data_ptr());
        float *_qdotk_max = reinterpret_cast<float *>(qdotk_max_buf.data_ptr());
        float *_integral_norm = reinterpret_cast<float *>(integral_norm_buf.data_ptr());
        // gradient chunk buffers are always fp32, channels-last [B, nlat_halo, nlon_kx, C]
        float *_dkx = reinterpret_cast<float *>(dkx.data_ptr());
        float *_dvx = reinterpret_cast<float *>(dvx.data_ptr());

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, qy.scalar_type(), "s2_attention_bwd_ring_step_upsample_pass2_cuda", [&] {
                using storage_t = scalar_t;

                torch::Tensor kxP = kx;
                torch::Tensor vxP = vx;
                torch::Tensor qyP = qy;
                torch::Tensor dyP = dy;

                storage_t *_kxp = reinterpret_cast<storage_t *>(kxP.data_ptr());
                storage_t *_vxp = reinterpret_cast<storage_t *>(vxP.data_ptr());
                storage_t *_qyp = reinterpret_cast<storage_t *>(qyP.data_ptr());
                storage_t *_dyp = reinterpret_cast<storage_t *>(dyP.data_ptr());

                dim3 block(WARP_SIZE, THREADS / WARP_SIZE);
                dim3 grid_in(DIV_UP(nlat_halo * nlon_kx, block.y), batch_size);

                const size_t sh_dkv = sizeof(float) * (2 * nchans_in + 2 * nchans_out) * block.y;

                s2_attn_bwd_ring_upsample_dkv_k<THREADS><<<grid_in, block, sh_dkv, stream>>>(
                    static_cast<int>(nchans_in), static_cast<int>(nchans_out), nlat_halo, nlon_kx,
                    static_cast<int>(nlon_out_global), static_cast<int>(pscale_out), static_cast<int>(lon_lo_kx),
                    static_cast<int>(lat_halo_start), static_cast<int>(nlat_out), static_cast<int>(nlon_out), _kxp, _vxp,
                    _qyp, _dyp, _row_off, _col_idx, _quad_weights, _alpha_sum, _qdotk_max, _integral_norm, _dkx, _dvx);
                CHECK_ERROR("s2_attn_bwd_ring_upsample_dkv_k");
            });

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    TORCH_LIBRARY_IMPL(attention_kernels, CUDA, m)
    {
        m.impl("backward_ring_step_upsample_pass1", &s2_attention_bwd_ring_step_upsample_pass1_cuda);
        m.impl("backward_ring_step_upsample_pass2", &s2_attention_bwd_ring_step_upsample_pass2_cuda);
    }

} // namespace attention_kernels
