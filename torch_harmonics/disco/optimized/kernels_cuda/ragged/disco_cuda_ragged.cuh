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

// Shared by the ragged CUDA DISCO kernels, disco_cuda_fwd_ragged.cu and
// disco_cuda_bwd_ragged.cu: psi in arc form keyed per point, with the ring tables of the
// grid the arcs walk.
//
// The regular kernels stage a ring in shared memory and serve every longitude of it from
// one block through the p-shift. A ragged grid has no p-shift, so a row is a (basis
// function, point) and does one point's worth of work. The ragged kernels therefore give
// each row a thread rather than a block, and keep the channels-first layout of the regular
// ops: rows are sorted by basis function and then point, and the neighbourhoods of
// adjacent points on a ring are offset by about one input point, so adjacent threads read
// (forward) or update (backward) adjacent addresses of the same (batch, channel) plane.
// Batch and channel run along the grid's y dimension.

#pragma once

#include "../common/disco_cuda.cuh"

#include <algorithm>

// threads per block of the ragged kernels, one row each
#define DISCO_RAGGED_THREADS (128)

// the grid's y dimension is capped; larger B*C strides over it
#define DISCO_RAGGED_MAX_GRID_Y (65535)

namespace disco_kernels
{

    // the ring tables of one launch, as raw pointers
    struct Rings {
        const int64_t *base;
        const int64_t *size;
    };

    inline Rings rings(const torch::Tensor &ring_base, const torch::Tensor &ring_size)
    {
        return Rings {ring_base.data_ptr<int64_t>(), ring_size.data_ptr<int64_t>()};
    }

    inline dim3 ragged_grid(int64_t nrows, int64_t BC)
    {
        return dim3(DIV_UP(nrows, DISCO_RAGGED_THREADS), std::min<int64_t>(BC, DISCO_RAGGED_MAX_GRID_Y));
    }

} // namespace disco_kernels
