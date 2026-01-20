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

#include "disco.h"

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#include "cudamacro.h"

#define MIN_THREADS (64)
#define ELXTH_MAX (32)

namespace disco_kernels {

    // fast base 2 logarithm for integer types
    inline int64_t countl_zero_u64(std::uint64_t v) {
#if defined(__GNUC__) || defined(__clang__)
        return v ? static_cast<int64_t>(__builtin_clzll(v)) : 64;
#elif defined(_MSC_VER)
        unsigned long idx;
        if (_BitScanReverse64(&idx, v)) return 63 - static_cast<int64_t>(idx);
        return 64;
#else
        int64_t n = 0;
        while (v && (v >> 63) == 0) { v <<= 1; ++n; }
        return v ? n : 64;
#endif
    }

    // fast base 2 logarithm for integer types
    inline int64_t ilog2(int64_t n) {
        return static_cast<int64_t>(
            std::numeric_limits<std::uint64_t>::digits - 1 -
            countl_zero_u64(static_cast<std::uint64_t>(n)));
    }

    // fast power of 2 for integer types
    inline int64_t pow2(int64_t n) {
        if (n < 0 || n >= 64) {
            // Handle error: overflow or invalid input
            return 0;
        }
        return (1ULL << n);  // 1 shifted left x times
    }

    // forward kernel
    torch::Tensor disco_cuda_fwd(torch::Tensor inp, torch::Tensor roff_idx, torch::Tensor ker_idx, torch::Tensor row_idx, 
                                 torch::Tensor col_idx, torch::Tensor val, int64_t kernel_size, int64_t Ho, int64_t Wo);

    // backward kernel
    torch::Tensor disco_cuda_bwd(torch::Tensor ograd, torch::Tensor roff_idx, torch::Tensor ker_idx, torch::Tensor row_idx,
                                torch::Tensor col_idx, torch::Tensor val, int64_t kernel_size, int64_t Ho, int64_t Wo);

}
