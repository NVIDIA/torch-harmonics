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
// Upsample (scatter-style) attention forward — CUDA
// =====================================================================================
//
// K, V live on the input (smaller) grid; Q lives on the output (larger) grid.
// psi is built with rows indexed by hi and cols encoding (ho, wo_canonical) on
// the output grid as ho * nlon_out + wo_canonical (canonical at wi=0). For
// wi > 0 the actual output column is (wo_canonical + pscale_out * wi) mod
// nlon_out, with pscale_out = nlon_out / nlon_in. Requires nlon_out % nlon_in
// == 0.
//
// Algorithm: INPUT-keyed scatter. Each warp owns a coarse input cell (b, hi, wi)
// and walks ONLY its real psi row row_off[hi]..row_off[hi+1]. For each entry the
// mapping wi -> wo = (wo_canonical + pscale_out * wi) mod nlon_out (a bijection
// within each pscale_out residue class) gives the fine output cell it feeds, and
// the warp scatters its contribution there via atomics. Because the softmax cell
// (fine output) is not the row key (coarse input), the per-output-cell reduction
// spans multiple warps, so the softmax is done in two passes with cross-warp
// atomics (max, then normalize + accumulate) plus a finalize. This replaced an
// output-centric scan kernel that rescanned every psi row per output cell — an
// O(out_cells * nnz) cost that made it 15-76x slower than this O(nnz) form.
// =====================================================================================

#include "attention_cuda.cuh"
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAUtils.h>

#include <cuda_runtime.h>
#include <cfloat>

#include "cudamacro.h"
#include "attention_cuda_utils.cuh"

#define THREADS (64)

namespace attention_kernels
{

    // =====================================================================================
    // SCATTER (input-keyed) forward. One warp per COARSE INPUT cell (hi, wi) walks ONLY
    // its real psi row (row_off[hi]..row_off[hi+1]) and SCATTERS its contribution into
    // the fine output cells via atomics, mirroring the downsample-backward structure.
    // This replaces the old output-centric scan kernel, whose O(out_cells * nnz)
    // redundant scan made it 15-76x slower (the scan is now O(nnz), gather-class).
    //
    // Because the softmax-normalization cell (fine output) is NOT the row key (coarse
    // input), the per-output-cell reduction spans multiple coarse warps, so the softmax
    // is done in the standard two-pass form with cross-warp atomics:
    //   pass 1 (max):  scatter q.k -> atomicMax  into maxbuf[b, ho, wo]
    //   pass 2 (acc):  scatter exp(q.k - max)*w -> atomicAdd into denom[b, ho, wo]
    //                  and exp(...)*w*v          -> atomicAdd into numer[b, ho, wo, :]
    //   finalize:      y[b, ho, wo, :] = numer / denom   (narrowed to STORAGE_T)
    // numer/denom/maxbuf are fp32 regardless of activation dtype. Prototype is scalar
    // (no float4 atomic fast path); activations widen at load via vload.
    // =====================================================================================

    // float atomicMax (no native float overload); CAS loop, correct for any sign.
    __device__ __forceinline__ float atomicMaxf(float *addr, float val)
    {
        int *ai = reinterpret_cast<int *>(addr);
        int old = *ai;
        while (val > __int_as_float(old)) {
            const int assumed = old;
            old = atomicCAS(ai, assumed, __float_as_int(val));
            if (old == assumed) { break; }
        }
        return __int_as_float(old);
    }

    // map a coarse longitude wi + canonical fine longitude to the actual fine longitude
    __device__ __forceinline__ int scatter_wo(int wo_canonical, int wi, int pscale_out, int nlon_out)
    {
        int wo = wo_canonical + pscale_out * wi; // < 2*nlon_out since both terms < nlon_out
        if (wo >= nlon_out) { wo -= nlon_out; }
        return wo;
    }

    // pass 1: per coarse cell, scatter q.k into the per-output-cell running max.
    template <int THREADS_PER_BLOCK, typename STORAGE_T>
    __global__ __launch_bounds__(THREADS_PER_BLOCK) void s2_attn_fwd_upsample_scatter_max_k(
        int nchan_in, int nlat_in, int nlon_in, int nlat_out, int nlon_out, const STORAGE_T *__restrict__ kx,
        const STORAGE_T *__restrict__ qy, const int64_t *__restrict__ row_off, const int64_t *__restrict__ col_idx,
        float *__restrict__ maxbuf)
    {
        extern __shared__ float shext[];
        float *sh_k = shext + threadIdx.y * nchan_in;

        const int batch = blockIdx.y;
        const int wid = blockIdx.x * blockDim.y + threadIdx.y;
        if (wid >= nlat_in * nlon_in) { return; }

        const int tidx = threadIdx.x;
        const int hi = wid / nlon_in;
        const int wi = wid - hi * nlon_in;
        const int pscale_out = nlon_out / nlon_in;

        kx += int64_t(batch) * nlat_in * nlon_in * nchan_in + (int64_t(hi) * nlon_in + wi) * nchan_in;
        qy += int64_t(batch) * nlat_out * nlon_out * nchan_in;
        maxbuf += int64_t(batch) * nlat_out * nlon_out;

        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) { sh_k[chan] = vload(kx, chan); }

        const int64_t rbeg = row_off[hi];
        const int rlen = static_cast<int>(row_off[hi + 1] - rbeg);
        const int64_t *col_hi = col_idx + rbeg;

        for (int off = 0; off < rlen; off++) {
            const int64_t col = col_hi[off];
            const int ho = static_cast<int>(col / nlon_out);
            const int wo = scatter_wo(static_cast<int>(col - int64_t(ho) * nlon_out), wi, pscale_out, nlon_out);
            const STORAGE_T *_qy = qy + (int64_t(ho) * nlon_out + wo) * nchan_in;

            float qd = 0.f;
            for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) { qd += sh_k[chan] * vload(_qy, chan); }
            qd = __warp_sum(qd);
            if (tidx == 0) { atomicMaxf(&maxbuf[int64_t(ho) * nlon_out + wo], qd); }
        }
    }

    // pass 2: per coarse cell, scatter exp(q.k - max)*w into denom and exp(...)*w*v into numer.
    template <int THREADS_PER_BLOCK, typename STORAGE_T>
    __global__ __launch_bounds__(THREADS_PER_BLOCK) void s2_attn_fwd_upsample_scatter_acc_k(
        int nchan_in, int nchan_out, int nlat_in, int nlon_in, int nlat_out, int nlon_out,
        const STORAGE_T *__restrict__ kx, const STORAGE_T *__restrict__ vx, const STORAGE_T *__restrict__ qy,
        const int64_t *__restrict__ row_off, const int64_t *__restrict__ col_idx, const float *__restrict__ quad_weights,
        const float *__restrict__ maxbuf, float *__restrict__ numer, float *__restrict__ denom)
    {
        extern __shared__ float shext[];
        float *sh_k = shext + threadIdx.y * (nchan_in + nchan_out);
        float *sh_v = sh_k + nchan_in;

        const int batch = blockIdx.y;
        const int wid = blockIdx.x * blockDim.y + threadIdx.y;
        if (wid >= nlat_in * nlon_in) { return; }

        const int tidx = threadIdx.x;
        const int hi = wid / nlon_in;
        const int wi = wid - hi * nlon_in;
        const int pscale_out = nlon_out / nlon_in;

        kx += int64_t(batch) * nlat_in * nlon_in * nchan_in + (int64_t(hi) * nlon_in + wi) * nchan_in;
        vx += int64_t(batch) * nlat_in * nlon_in * nchan_out + (int64_t(hi) * nlon_in + wi) * nchan_out;
        qy += int64_t(batch) * nlat_out * nlon_out * nchan_in;
        numer += int64_t(batch) * nlat_out * nlon_out * nchan_out;
        denom += int64_t(batch) * nlat_out * nlon_out;
        maxbuf += int64_t(batch) * nlat_out * nlon_out;

        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) { sh_k[chan] = vload(kx, chan); }
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { sh_v[chan] = vload(vx, chan); }

        const float qw = quad_weights[hi];
        const int64_t rbeg = row_off[hi];
        const int rlen = static_cast<int>(row_off[hi + 1] - rbeg);
        const int64_t *col_hi = col_idx + rbeg;

        for (int off = 0; off < rlen; off++) {
            const int64_t col = col_hi[off];
            const int ho = static_cast<int>(col / nlon_out);
            const int wo = scatter_wo(static_cast<int>(col - int64_t(ho) * nlon_out), wi, pscale_out, nlon_out);
            const int64_t cell = int64_t(ho) * nlon_out + wo;
            const STORAGE_T *_qy = qy + cell * nchan_in;

            float qd = 0.f;
            for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) { qd += sh_k[chan] * vload(_qy, chan); }
            qd = __warp_sum(qd);

            const float alpha = expf(qd - maxbuf[cell]) * qw;
            if (tidx == 0) { atomicAdd(&denom[cell], alpha); }
            float *_numer = numer + cell * nchan_out;
            for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { atomicAdd(&_numer[chan], alpha * sh_v[chan]); }
        }
    }

    // finalize: y[b, ho, wo, :] = numer / denom (one warp per fine output cell).
    template <int THREADS_PER_BLOCK, typename STORAGE_T>
    __global__ __launch_bounds__(THREADS_PER_BLOCK) void s2_attn_fwd_upsample_scatter_final_k(
        int nchan_out, int nlat_out, int nlon_out, const float *__restrict__ numer, const float *__restrict__ denom,
        STORAGE_T *__restrict__ y)
    {
        const int batch = blockIdx.y;
        const int wid = blockIdx.x * blockDim.y + threadIdx.y;
        if (wid >= nlat_out * nlon_out) { return; }
        const int tidx = threadIdx.x;

        const int64_t cell = int64_t(batch) * nlat_out * nlon_out + wid;
        const float inv = 1.0f / denom[cell];
        const float *_numer = numer + cell * nchan_out;
        STORAGE_T *_y = y + cell * nchan_out;
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { vstore(_y, chan, _numer[chan] * inv); }
    }

    // host launcher for the scatter forward (allocates the fp32 reduction buffers,
    // runs the three passes). STORAGE_T deduces from the activation pointers.
    template <typename STORAGE_T>
    static void launch_attn_fwd_upsample_scatter(int batch_size, int nchans_in, int nchans_out, int nlat_in,
                                                 int nlon_in, int nlat_out, int nlon_out, STORAGE_T *_kxp,
                                                 STORAGE_T *_vxp, STORAGE_T *_qyp, int64_t *_row_off, int64_t *_col_idx,
                                                 float *_quad_weights, STORAGE_T *_yp, cudaStream_t stream)
    {
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        torch::Tensor numer = torch::zeros({batch_size, nlat_out, nlon_out, nchans_out}, opts);
        torch::Tensor denom = torch::zeros({batch_size, nlat_out, nlon_out}, opts);
        torch::Tensor maxbuf = torch::full({batch_size, nlat_out, nlon_out}, -FLT_MAX, opts);

        float *_numer = reinterpret_cast<float *>(numer.data_ptr());
        float *_denom = reinterpret_cast<float *>(denom.data_ptr());
        float *_maxbuf = reinterpret_cast<float *>(maxbuf.data_ptr());

        dim3 block(WARP_SIZE, THREADS / WARP_SIZE);
        dim3 grid_in(DIV_UP(nlat_in * nlon_in, block.y), batch_size);
        dim3 grid_out(DIV_UP(nlat_out * nlon_out, block.y), batch_size);

        const size_t sh1 = sizeof(float) * nchans_in * block.y;
        const size_t sh2 = sizeof(float) * (nchans_in + nchans_out) * block.y;

        s2_attn_fwd_upsample_scatter_max_k<THREADS><<<grid_in, block, sh1, stream>>>(
            nchans_in, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _qyp, _row_off, _col_idx, _maxbuf);
        CHECK_ERROR("s2_attn_fwd_upsample_scatter_max_k");

        s2_attn_fwd_upsample_scatter_acc_k<THREADS>
            <<<grid_in, block, sh2, stream>>>(nchans_in, nchans_out, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp,
                                              _qyp, _row_off, _col_idx, _quad_weights, _maxbuf, _numer, _denom);
        CHECK_ERROR("s2_attn_fwd_upsample_scatter_acc_k");

        s2_attn_fwd_upsample_scatter_final_k<THREADS>
            <<<grid_out, block, 0, stream>>>(nchans_out, nlat_out, nlon_out, _numer, _denom, _yp);
        CHECK_ERROR("s2_attn_fwd_upsample_scatter_final_k");
    }

    // -----------------------------------------------------------------------------
    // host dispatcher — called from s2_attention_fwd_cuda when the direction is
    // upsample (nlon_out % nlon_in == 0). Native-storage (Tier B): kept NON-templated
    // (it is called from a different TU, attention_cuda_fwd.cu) and does its own
    // AT_DISPATCH over the input dtype, then routes to the input-keyed scatter forward
    // (scalar path for every dtype; activations widen at load, fp32 compute, output
    // narrowed at store). The fp32 reduction buffers are allocated inside the launcher.
    // -----------------------------------------------------------------------------
    void s2_attn_fwd_upsample_dispatch(int batch_size, size_t nchans_in, size_t nchans_out, int64_t nlon_in,
                                       int64_t nlat_in, int64_t nlat_out, int64_t nlon_out, torch::Tensor kxP,
                                       torch::Tensor vxP, torch::Tensor qyP, torch::Tensor psi_row_off,
                                       torch::Tensor psi_col_idx, torch::Tensor quad_weights, torch::Tensor yP)
    {

        auto stream = at::cuda::getCurrentCUDAStream().stream();

        int64_t *_row_off = reinterpret_cast<int64_t *>(psi_row_off.data_ptr());
        int64_t *_col_idx = reinterpret_cast<int64_t *>(psi_col_idx.data_ptr());
        float *_quad_weights = reinterpret_cast<float *>(quad_weights.data_ptr());

        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, qyP.scalar_type(), "s2_attn_fwd_upsample", [&] {
            scalar_t *_kxp = reinterpret_cast<scalar_t *>(kxP.data_ptr());
            scalar_t *_vxp = reinterpret_cast<scalar_t *>(vxP.data_ptr());
            scalar_t *_qyp = reinterpret_cast<scalar_t *>(qyP.data_ptr());
            scalar_t *_yp = reinterpret_cast<scalar_t *>(yP.data_ptr());

            launch_attn_fwd_upsample_scatter(batch_size, static_cast<int>(nchans_in), static_cast<int>(nchans_out),
                                             static_cast<int>(nlat_in), static_cast<int>(nlon_in),
                                             static_cast<int>(nlat_out), static_cast<int>(nlon_out), _kxp, _vxp, _qyp,
                                             _row_off, _col_idx, _quad_weights, _yp, stream);
        });
    }

} // namespace attention_kernels
