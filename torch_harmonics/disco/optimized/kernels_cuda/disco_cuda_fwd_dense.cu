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
// Disco forward — dense-packed psi (CUDA baseline)
// =====================================================================================
//
// Consumes the (K, Ho, NBR_PAD)-packed psi produced by pack_psi_dense and computes
//
//   out[b, c, k, ho, wo] = sum_{nz=0..count[k,ho]} val[k,ho,nz]
//                                     * inp[b, c, hi[k,ho,nz],
//                                              (wi_base[k,ho,nz] + pscale * wo) mod Wi]
//
// with pscale = Wi / Wo. Padded slots (nz >= count[k,ho]) are skipped via the cnt
// bound and also have val == 0 by construction.
//
// Parallelization (baseline): one CTA per (BC, k*Ho); threads parallelize wo.
// Plain CUDA cores, no TMA, no Tensor Cores, no shmem staging — this is the
// correctness floor. Performance variants (channel/K tiling, TMA slabs,
// WGMMA) will live in additional translation units alongside this one.
// =====================================================================================

#include "../disco.h"
#include "disco_cuda.cuh"

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>

namespace disco_kernels {

template <int BDIM_X, typename T>
__global__ __launch_bounds__(BDIM_X)
void disco_fwd_dense_blk_k(int Hi, int Wi, int K, int Ho, int Wo, int NBR_PAD, int pscale,
                           const int64_t *__restrict__ pack_idx,
                           const T       *__restrict__ pack_val,
                           const int64_t *__restrict__ pack_count,
                           const T       *__restrict__ inp,
                           T             *__restrict__ out)
{
    const int bc = blockIdx.y;          // b * C + c
    const int kh = blockIdx.x;          // k * Ho + ho
    const int k  = kh / Ho;
    const int ho = kh - k * Ho;

    const int tid = threadIdx.x;

    const int64_t kh_off = (int64_t)k * Ho + ho;
    const int64_t *idx_kh = pack_idx + kh_off * NBR_PAD * 2;
    const T       *val_kh = pack_val + kh_off * NBR_PAD;
    const int      cnt    = (int)pack_count[kh_off];

    const T *inp_bc = inp + (int64_t)bc * Hi * Wi;
    T       *out_kh = out + ((int64_t)bc * K + k) * Ho * Wo + (int64_t)ho * Wo;

    for (int wo = tid; wo < Wo; wo += BDIM_X) {
        T acc = static_cast<T>(0);

        for (int nz = 0; nz < cnt; nz++) {
            const int hi      = (int)idx_kh[nz * 2 + 0];
            const int wi_base = (int)idx_kh[nz * 2 + 1];
            const T   v       = val_kh[nz];

            // wi_base + pscale * wo < 2 * Wi, so a single subtract suffices.
            int wi = wi_base + pscale * wo;
            if (wi >= Wi) wi -= Wi;

            acc += v * inp_bc[(int64_t)hi * Wi + wi];
        }

        out_kh[wo] = acc;
    }
}

torch::Tensor disco_cuda_fwd_dense(torch::Tensor inp,
                                   torch::Tensor pack_idx,
                                   torch::Tensor pack_val,
                                   torch::Tensor pack_count,
                                   int64_t K, int64_t Ho, int64_t Wo)
{
    CHECK_CUDA_INPUT_TENSOR(inp);
    CHECK_CUDA_INPUT_TENSOR(pack_idx);
    CHECK_CUDA_INPUT_TENSOR(pack_val);
    CHECK_CUDA_INPUT_TENSOR(pack_count);

    const int64_t B  = inp.size(0);
    const int64_t C  = inp.size(1);
    const int64_t Hi = inp.size(2);
    const int64_t Wi = inp.size(3);
    const int64_t BC = B * C;

    TORCH_CHECK(Wi % Wo == 0,
                "Wi (", Wi, ") must be an integer multiple of Wo (", Wo, ")");

    TORCH_CHECK(pack_idx.dim()   == 4 && pack_idx.size(0)   == K && pack_idx.size(1)   == Ho && pack_idx.size(3) == 2,
                "pack_idx must have shape [K, Ho, NBR_PAD, 2]");
    TORCH_CHECK(pack_val.dim()   == 3 && pack_val.size(0)   == K && pack_val.size(1)   == Ho,
                "pack_val must have shape [K, Ho, NBR_PAD]");
    TORCH_CHECK(pack_count.dim() == 2 && pack_count.size(0) == K && pack_count.size(1) == Ho,
                "pack_count must have shape [K, Ho]");

    const int64_t NBR_PAD = pack_idx.size(2);
    const int     pscale  = (int)(Wi / Wo);

    auto options = torch::TensorOptions().device(inp.device()).dtype(inp.dtype());
    auto out = torch::zeros({B, C, K, Ho, Wo}, options);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    constexpr int BDIM_X = 256;
    dim3 grid((unsigned)(K * Ho), (unsigned)BC);
    dim3 block(BDIM_X);

    AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_fwd_dense_cuda", ([&] {
        disco_fwd_dense_blk_k<BDIM_X, scalar_t><<<grid, block, 0, stream>>>(
            (int)Hi, (int)Wi, (int)K, (int)Ho, (int)Wo, (int)NBR_PAD, pscale,
            pack_idx.data_ptr<int64_t>(),
            pack_val.data_ptr<scalar_t>(),
            pack_count.data_ptr<int64_t>(),
            inp.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>());
    }));

    return out;
}

TORCH_LIBRARY_IMPL(disco_kernels, CUDA, m)
{
    m.impl("forward_dense", &disco_cuda_fwd_dense);
}

}
