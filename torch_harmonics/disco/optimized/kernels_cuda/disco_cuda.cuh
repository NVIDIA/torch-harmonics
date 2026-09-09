// coding=utf-8
//
// SPDX-FileCopyrightText: Copyright (c) 2024 The torch-harmonics Authors. All rights reserved.
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

#include "../disco.h"

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#define CHECK_CUDA_TENSOR(x) TORCH_INTERNAL_ASSERT(x.device().type() == torch::kCUDA)
#define CHECK_CUDA_INPUT_TENSOR(x)                                                                                     \
    CHECK_CUDA_TENSOR(x);                                                                                              \
    CHECK_CONTIGUOUS_TENSOR(x)

#define DIV_UP(a, b) (((a) + ((b) - 1)) / (b))

#define MIN_THREADS (64)
#define ELXTH_MAX (32)

namespace disco_kernels
{

    // Gather the eight input elements one thread needs for its A-tile cell, as one
    // 16-byte value, using aligned 32-bit word loads.
    //
    // Element i lives at halfword s + i*P of the input row. Writing s = 2q + R, it
    // sits at word offset (R + i*P) >> 1 from word q, in the low half when
    // (R + i*P) is even and the high half otherwise. With P compile-time every
    // offset folds to a constant and repeated offsets are CSE'd, so P=1 needs 5
    // words and P=2 needs 8 -- against 8 scalar 2-byte loads plus eightfold address
    // arithmetic in the form this replaces.
    //
    // R is deliberately a *runtime* value. Lanes in a warp hold consecutive wi_base,
    // so s alternates parity every lane and every warp contains both parities;
    // selecting between two compile-time-R bodies makes the warp execute both. That
    // cost 11.2 loads/warp instead of 8 at pscale=2 and made the whole change a 16%
    // regression there. Both parities are served from one set of loads instead:
    //   P=1: the union is p[0..4] and parity is a 16-bit shift through it.
    //   P=2: the word offset (R + 2i) >> 1 == i for both parities -- R only picks
    //        which half of each word.
    //
    // Working at 4-byte rather than 16-byte granularity keeps the word offsets
    // constant-foldable; a 16-byte scheme would need a runtime-indexed source word,
    // and a runtime-indexed local array spills.
    //
    // Gated to P in {1, 2}. A P=3 form was written and verified bit-identical, but
    // it needs 12 words to deliver 8 halfwords, so loads rise ~45% while
    // instructions fall ~23% -- which trades well only where L1 has slack:
    //   540x1080  -> 180x360   L1 68.7% -> 80.1%   1.91 -> 1.89 ms   (-1%)
    //   1080x2160 -> 360x720   L1 82.2% -> 89.2%   24.55 -> 27.71 ms (+13%)
    // No workload uses pscale 3, so it takes the scalar path with pscale >= 4.
    //
    // Shared between the SM_90a and SM_100a kernels: one definition, because two
    // differing definitions of the same template in namespace disco_kernels across
    // translation units would be an ODR violation.
    template <int P> __device__ __forceinline__ int4 gather_window_rt(const uint32_t *p, int R)
    {
        int4 v;
        if constexpr (P == 1) {
            const uint32_t w0 = p[0], w1 = p[1], w2 = p[2], w3 = p[3], w4 = p[4];
            const int sh = 16 * R;
            v.x = (int)__funnelshift_r(w0, w1, sh);
            v.y = (int)__funnelshift_r(w1, w2, sh);
            v.z = (int)__funnelshift_r(w2, w3, sh);
            v.w = (int)__funnelshift_r(w3, w4, sh);
        } else {
            static_assert(P == 2, "windowed gather is gated to pscale 1 and 2");
            // low halves of each word pair for R=0, high halves for R=1
            const uint32_t sel = R ? 0x7632u : 0x5410u;
            v.x = (int)__byte_perm(p[0], p[1], sel);
            v.y = (int)__byte_perm(p[2], p[3], sel);
            v.z = (int)__byte_perm(p[4], p[5], sel);
            v.w = (int)__byte_perm(p[6], p[7], sel);
        }
        return v;
    }

    // forward kernel (CSR)
    torch::Tensor disco_cuda_fwd(torch::Tensor inp, torch::Tensor roff_idx, torch::Tensor ker_idx, torch::Tensor row_idx,
                                 torch::Tensor col_idx, torch::Tensor val, int64_t K, int64_t Ho, int64_t Wo);

    // backward kernel (CSR)
    torch::Tensor disco_cuda_bwd(torch::Tensor inp, torch::Tensor roff_idx, torch::Tensor ker_idx, torch::Tensor row_idx,
                                 torch::Tensor col_idx, torch::Tensor val, int64_t K, int64_t Ho, int64_t Wo);

    // K-packed forward (WGMMA, Hopper SM_90a + bf16/fp16 only)
    torch::Tensor disco_cuda_fwd_kpacked(torch::Tensor inp, torch::Tensor pack_idx, torch::Tensor pack_val,
                                         torch::Tensor pack_offset, int64_t K, int64_t Ho, int64_t Wo);

} // namespace disco_kernels
