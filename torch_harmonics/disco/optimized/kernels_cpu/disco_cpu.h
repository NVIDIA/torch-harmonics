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

// Shared by the CPU DISCO kernels, disco_cpu_fwd.cpp and disco_cpu_bwd.cpp: psi in arc form
// as the kernels take it, and the dispatch on the longitude stride.
//
// The CPU kernels compute in fp32/fp64 only: the storage/compute split is a CUDA
// optimization, and CPU fp16/bf16 arithmetic is emulated anyway, so the hosts upcast
// reduced-precision activations and cast the result back.

#pragma once

#include "../disco.h"
#include "../disco_checks.h"

#include <algorithm>
#include <vector>

// Call IMPL<scalar_t, PSCALE> with a compile-time stride for pscale 1, 2 and 3, and the
// runtime stride (PSCALE = 0) otherwise. Expects `pscale` and `scalar_t` in scope.
#define DISCO_PSCALE_DISPATCH(IMPL, ...)                                                                               \
    switch (pscale) {                                                                                                  \
    case 1: IMPL<scalar_t, 1>(__VA_ARGS__); break;                                                                     \
    case 2: IMPL<scalar_t, 2>(__VA_ARGS__); break;                                                                     \
    case 3: IMPL<scalar_t, 3>(__VA_ARGS__); break;                                                                     \
    default: IMPL<scalar_t, 0>(__VA_ARGS__); break;                                                                    \
    }

namespace disco_kernels
{

    // the psi of one call, as raw pointers
    struct ArcPsiCpu {
        int64_t nrows;
        const int32_t *row_ker;
        const int32_t *row_lat;
        const int64_t *seg_off;
        const int32_t *seg;
        const int64_t *val_off;
    };

    inline ArcPsiCpu arc_psi_cpu(const torch::Tensor &row_ker, const torch::Tensor &row_lat,
                                 const torch::Tensor &seg_off, const torch::Tensor &seg, const torch::Tensor &val_off)
    {
        return ArcPsiCpu {row_ker.size(0),
                          row_ker.data_ptr<int32_t>(),
                          row_lat.data_ptr<int32_t>(),
                          seg_off.data_ptr<int64_t>(),
                          seg.data_ptr<int32_t>(),
                          val_off.data_ptr<int64_t>()};
    }

} // namespace disco_kernels
