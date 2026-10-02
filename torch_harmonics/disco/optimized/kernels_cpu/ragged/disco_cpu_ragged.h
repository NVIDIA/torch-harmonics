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

// Shared by the ragged CPU DISCO kernels, disco_cpu_fwd_ragged.cpp and
// disco_cpu_bwd_ragged.cpp: psi in arc form keyed per point, with the ring tables of the
// grid the arcs walk. See kernels_cuda/ragged/ for the CUDA counterparts.
//
// The regular kernels stage an input (or output) ring and serve every longitude of it
// through the p-shift. A ragged grid has no p-shift, so a row is a (basis function,
// point) and does one point's worth of work: these kernels walk the arcs of a row and
// address the field by flat index, ring_base[ring] + offset, wrapping at
// ring_base[ring] + ring_size[ring] with a compare-and-subtract.

#pragma once

#include "../common/disco_cpu.h"

namespace disco_kernels
{

    // the ring tables of one call, as raw pointers
    struct RingsCpu {
        const int64_t *base;
        const int64_t *size;
    };

    inline RingsCpu rings_cpu(const torch::Tensor &ring_base, const torch::Tensor &ring_size)
    {
        return RingsCpu {ring_base.data_ptr<int64_t>(), ring_size.data_ptr<int64_t>()};
    }

} // namespace disco_kernels
