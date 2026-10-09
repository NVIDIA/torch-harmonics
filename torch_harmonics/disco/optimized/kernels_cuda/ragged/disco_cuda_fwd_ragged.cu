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

// The DISCO contraction on a ragged grid, on CUDA: forward_ragged, a gather, with psi in arc
// form keyed per output point -- see torch_harmonics/disco/_psi_layouts.py and
// disco_cuda_ragged.cuh for the thread layout. The scatter is in disco_cuda_bwd_ragged.cu.

#include "../../disco.h"
#include "disco_cuda_ragged.cuh"

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <c10/cuda/CUDAException.h>

namespace disco_kernels
{

    // One thread per row. Each row writes its own output point, so no two threads write the
    // same address and the result is stored once, in the storage dtype.
    template <typename STORAGE_T, typename COMPUTE_T>
    __global__ __launch_bounds__(DISCO_RAGGED_THREADS) void disco_fwd_ragged_k(
        const int64_t BC, const int64_t K, const int64_t Ni, const int64_t No, const ArcPsi psi, const Rings rings,
        const COMPUTE_T *__restrict__ vals, const STORAGE_T *__restrict__ inp, STORAGE_T *__restrict__ out)
    {
        const int64_t r = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (r >= psi.nrows) return;

        const int64_t sbeg = psi.seg_off[r];
        const int64_t send = psi.seg_off[r + 1];
        const int64_t vbeg = psi.val_off[r];
        // row_lat holds the row's output point on a ragged grid
        const int64_t opt = static_cast<int64_t>(psi.row_ker[r]) * No + psi.row_lat[r];

        for (int64_t bc = blockIdx.y; bc < BC; bc += gridDim.y) {
            const STORAGE_T *__restrict__ inp_bc = inp + bc * Ni;

            COMPUTE_T acc = static_cast<COMPUTE_T>(0);
            int64_t v = vbeg;
            for (int64_t s = sbeg; s < send; s++) {
                const int64_t ring = psi.seg[3 * s + 0];
                const int64_t start = psi.seg[3 * s + 1];
                const int64_t len = psi.seg[3 * s + 2];

                const int64_t lo = rings.base[ring];
                const int64_t hi = lo + rings.size[ring];

                int64_t col = lo + start;
                for (int64_t j = 0; j < len; j++) {
                    acc += vals[v++] * static_cast<COMPUTE_T>(inp_bc[col]);
                    if (++col == hi) col = lo;
                }
            }

            out[bc * K * No + opt] = static_cast<STORAGE_T>(acc);
        }
    }

    torch::Tensor disco_cuda_fwd_ragged(torch::Tensor inp, torch::Tensor row_ker, torch::Tensor row_pt,
                                        torch::Tensor seg_off, torch::Tensor seg, torch::Tensor val_off,
                                        torch::Tensor vals, torch::Tensor ring_base, torch::Tensor ring_size, int64_t K,
                                        int64_t No)
    {
        TORCH_CHECK(inp.device().is_cuda(), "inp must be a CUDA tensor, got ", inp.device());
        check_ragged_forward_inputs(inp, row_ker, row_pt, seg_off, seg, val_off, vals, ring_base, ring_size, K, No);

        // launch on the inputs' device, not whichever one is current
        const at::cuda::OptionalCUDAGuard device_guard(inp.device());

        const int64_t BC = inp.size(0) * inp.size(1);
        const int64_t Ni = inp.size(2);

        // rows without entries are never visited, so their outputs must start at zero
        auto out = torch::zeros({inp.size(0), inp.size(1), K, No}, inp.options());
        if (row_ker.size(0) == 0 || BC == 0) return out;

        const ArcPsi psi = arc_psi(row_ker, row_pt, seg_off, seg, val_off);
        const Rings rg = rings(ring_base, ring_size);
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, inp.scalar_type(), "disco_forward_ragged_cuda", ([&] {
                                            using storage_t = scalar_t;
                                            using compute_t = typename at::opmath_type<storage_t>;
                                            disco_fwd_ragged_k<storage_t, compute_t>
                                                <<<ragged_grid(psi.nrows, BC), DISCO_RAGGED_THREADS, 0, stream>>>(
                                                    BC, K, Ni, No, psi, rg, vals.data_ptr<compute_t>(),
                                                    inp.data_ptr<storage_t>(), out.data_ptr<storage_t>());
                                        }));

        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }

    TORCH_LIBRARY_IMPL(disco_kernels, CUDA, m) { m.impl("forward_ragged", &disco_cuda_fwd_ragged); }

} // namespace disco_kernels
