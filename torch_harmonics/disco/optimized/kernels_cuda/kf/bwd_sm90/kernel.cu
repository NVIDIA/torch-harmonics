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

// DISCO spherical convolution transpose (arc scatter), fp16 activations, fp32 psi / accumulation.
//
// Regime dispatch between three tensor-core polyphase Toeplitz gather implementations. All split each fp32 psi
// value into two power-of-two-scaled fp16 terms, accumulate both products in fp32 mma.sync accumulators, round
// the output to fp16 once, and recompute all psi bookkeeping on the GPU on every call.
//   cr_impl   (B*C >= 256): channel-row MMA tiles (16 rows = channel x longitude-split pairs, a warp slides 15
//             mu-steps along one ring so consecutive MMAs reuse A registers), 4 ring slots x 3 longitude warps,
//             4 TMA producer warps, ring-ordered tile-major scheduling for L2 reuse of shared input rows.
//   ring_impl (small-channel pscale 1): longitude-row Hankel kernel fed through a variable-size byte ring.
//   unroll1_impl (everything else): longitude-row Hankel kernel with a three-stage TMA pipeline, inline
//             hot-arc entries, serial k-step loop bodies, heavy-first scheduling and split-K.
// Each implementation falls back to an fp32 atomic-scatter path for inputs its fast path does not cover
// (any pscale, Wi, B*C, arc length, empty rows).
#include "cr_impl.cuh"
#include "unroll1_impl.cuh"
#include "ring_impl.cuh"
#include "fft_impl.cuh"

void Run(ffi::TensorView inp, ffi::TensorView row_ker, ffi::TensorView row_lat, ffi::TensorView seg_off,
         ffi::TensorView seg, ffi::TensorView val_off, ffi::TensorView vals, int64_t nlat_out, int64_t nlon_out,
         ffi::TensorView out)
{
    const long long BC = (long long)inp.size(0) * inp.size(1);
    const int K = (int)inp.size(2), Hi = (int)inp.size(3), Wi = (int)inp.size(4);
    const int Ho = (int)nlat_out, Wo = (int)nlon_out;
    const int P = Wi > 0 ? Wo / Wi : 0;
    if (BC <= 0 || Ho <= 0 || Wo <= 0) return;
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(inp.device().device_type, inp.device().device_id));
    // Retain backing pages in the default async pool. Workspaces are still allocated, zeroed, fully recomputed,
    // and freed on every call; this only avoids unmapping and remapping their storage between calls.
    static const int pool_cfg = []() {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaMemPool_t pool;
        if (cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess) {
            uint64_t threshold = ~0ull;
            cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
        }
        return 0;
    }();
    (void)pool_cfg;
    fftp::Ctx fx;
    const unsigned char *skip = nullptr;
    if (BC < (1LL << 30))
        skip = fftp::Begin(fx, (int)BC, K, Hi, Wi, Ho, Wo, (int)row_ker.size(0), seg.size(0),
                           static_cast<const int64_t *>(seg_off.data_ptr()), static_cast<const int *>(seg.data_ptr()),
                           static_cast<const int64_t *>(val_off.data_ptr()), static_cast<const float *>(vals.data_ptr()),
                           static_cast<const int *>(row_ker.data_ptr()), static_cast<const int *>(row_lat.data_ptr()),
                           static_cast<const __half *>(inp.data_ptr()), stream);
    if (BC >= (1LL << 30)) {
        // the tensor-core paths index B*C with 32-bit ints: the fp32 atomic-scatter path covers huge batches
        cr::RunFallback(inp, row_ker, row_lat, seg_off, seg, val_off, vals, nlat_out, nlon_out, out);
    } else if (BC >= 256 && cr::RunCR(inp, row_ker, row_lat, seg_off, seg, val_off, vals, nlat_out, nlon_out, out, skip)) {
    } else if (P == 1) {
        leader_impl::Run(inp, row_ker, row_lat, seg_off, seg, val_off, vals, nlat_out, nlon_out, out, skip);
    } else {
        unroll1_impl::Run(inp, row_ker, row_lat, seg_off, seg, val_off, vals, nlat_out, nlon_out, out, skip);
    }
    fftp::Finish(fx, static_cast<__half *>(out.data_ptr()), stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
