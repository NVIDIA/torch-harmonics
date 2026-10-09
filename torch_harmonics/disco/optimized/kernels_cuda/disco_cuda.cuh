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
#include "../disco_checks.h"

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <type_traits>

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
    // Gated to P in {1, 2}. A P=3 form is bit-identical but needs 12 words to deliver
    // 8 halfwords, trading more loads for fewer instructions. That only pays where L1
    // has slack, so it helps at coarser output and hurts once L1 is near saturation --
    // the same code with opposite sign, which is why it is not enabled. No workload
    // uses pscale 3, so it takes the scalar path with pscale >= 4.
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

    // psi in arc form as the kernels take it: raw pointers to the arrays of _psi_layouts.py,
    // one set per launch
    struct ArcPsi {
        int64_t nrows;
        const int32_t *row_ker;
        const int32_t *row_lat;
        const int64_t *seg_off;
        const int32_t *seg;
        const int64_t *val_off;
    };

    inline ArcPsi arc_psi(const torch::Tensor &row_ker, const torch::Tensor &row_lat, const torch::Tensor &seg_off,
                          const torch::Tensor &seg, const torch::Tensor &val_off)
    {
        return ArcPsi {row_ker.size(0),
                       row_ker.data_ptr<int32_t>(),
                       row_lat.data_ptr<int32_t>(),
                       seg_off.data_ptr<int64_t>(),
                       seg.data_ptr<int32_t>(),
                       val_off.data_ptr<int64_t>()};
    }

    // The starting block shape for a row of W elements: 64 lanes up to 64*ELXTH_MAX, then
    // the smallest wider block starting from (ELXTH_MAX / 2) + 1 elements per lane.
    // Calls launch(integral_constant<NTH>, integral_constant<ELXTH>).
    template <typename LAUNCH> inline void with_block_shape(int64_t W, const char *what, LAUNCH &&launch)
    {
        // the wide configs split the element count as (ELXTH_MAX / 2) + 1, which is exact
        // only for an even ELXTH_MAX
        static_assert(0 == (ELXTH_MAX % 2));
        constexpr int E = (ELXTH_MAX / 2) + 1;
        if (W <= 64 * ELXTH_MAX) {
            launch(std::integral_constant<int, 64> {}, std::integral_constant<int, 1> {});
        } else if (W <= 128 * ELXTH_MAX) {
            launch(std::integral_constant<int, 128> {}, std::integral_constant<int, E> {});
        } else if (W <= 256 * ELXTH_MAX) {
            launch(std::integral_constant<int, 256> {}, std::integral_constant<int, E> {});
        } else if (W <= 512 * ELXTH_MAX) {
            launch(std::integral_constant<int, 512> {}, std::integral_constant<int, E> {});
        } else if (W <= 1024 * ELXTH_MAX) {
            launch(std::integral_constant<int, 1024> {}, std::integral_constant<int, E> {});
        } else {
            TORCH_CHECK(false, what, " (", W, ") exceeds the largest supported value (", 1024 * ELXTH_MAX, ")");
        }
    }

    // K-packed forward (WGMMA, Hopper SM_90a + bf16/fp16 only)
    torch::Tensor disco_cuda_fwd_kpacked(torch::Tensor inp, torch::Tensor pack_idx, torch::Tensor pack_val,
                                         torch::Tensor pack_offset, int64_t K, int64_t Ho, int64_t Wo);

} // namespace disco_kernels
