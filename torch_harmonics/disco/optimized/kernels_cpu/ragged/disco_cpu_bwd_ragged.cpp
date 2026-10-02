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

// The transpose of the DISCO contraction on a ragged grid, on the CPU: backward_ragged, a
// scatter, with psi in arc form keyed per input point -- see torch_harmonics/disco/_psi.py
// and, for the CUDA counterpart, kernels_cuda/ragged/disco_cuda_bwd_ragged.cu.

#include "disco_cpu_ragged.h"

namespace disco_kernels
{

    // Scatter. Parallel over (batch, channel) only: different rows scatter into the same
    // output points, so collapsing the rows into the parallel loop would race.
    template <typename scalar_t>
    static void disco_bwd_ragged_cpu_impl(int64_t B, int64_t C, int64_t K, int64_t Ni, int64_t No, const ArcPsiCpu psi,
                                          const RingsCpu rings, const scalar_t *__restrict__ vals,
                                          const scalar_t *__restrict__ inp, scalar_t *__restrict__ out)
    {
#pragma omp parallel for collapse(2)
        for (int64_t b = 0; b < B; b++) {
            for (int64_t c = 0; c < C; c++) {

                scalar_t *__restrict__ out_bc = out + (b * C + c) * No;

                for (int64_t r = 0; r < psi.nrows; r++) {

                    // row_lat holds the row's input point on a ragged grid
                    const scalar_t x = inp[((b * C + c) * K + psi.row_ker[r]) * Ni + psi.row_lat[r]];

                    int64_t v = psi.val_off[r];
                    for (int64_t s = psi.seg_off[r]; s < psi.seg_off[r + 1]; s++) {

                        const int64_t ring = psi.seg[3 * s + 0];
                        const int64_t start = psi.seg[3 * s + 1];
                        const int64_t len = psi.seg[3 * s + 2];

                        const int64_t lo = rings.base[ring];
                        const int64_t hi = lo + rings.size[ring];

                        int64_t col = lo + start;
                        for (int64_t j = 0; j < len; j++) {
                            out_bc[col] += vals[v++] * x;
                            if (++col == hi) col = lo;
                        }
                    }
                }
            }
        }
    }

    torch::Tensor disco_cpu_bwd_ragged(torch::Tensor inp, torch::Tensor row_ker, torch::Tensor row_pt,
                                       torch::Tensor seg_off, torch::Tensor seg, torch::Tensor val_off,
                                       torch::Tensor vals, torch::Tensor ring_base, torch::Tensor ring_size, int64_t K,
                                       int64_t No)
    {
        TORCH_CHECK(inp.device().is_cpu(), "inp must be a CPU tensor, got ", inp.device());
        check_ragged_backward_inputs(inp, row_ker, row_pt, seg_off, seg, val_off, vals, ring_base, ring_size, K, No);

        const auto inp_dtype = inp.scalar_type();
        inp = inp.to(compute_dtype(inp_dtype)).contiguous();

        const int64_t B = inp.size(0), C = inp.size(1), Ni = inp.size(3);
        auto out = torch::zeros({B, C, No}, inp.options());
        const ArcPsiCpu psi = arc_psi_cpu(row_ker, row_pt, seg_off, seg, val_off);
        const RingsCpu rings = rings_cpu(ring_base, ring_size);

        AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_backward_ragged_cpu", ([&] {
                                       disco_bwd_ragged_cpu_impl<scalar_t>(
                                           B, C, K, Ni, No, psi, rings, vals.data_ptr<scalar_t>(),
                                           inp.data_ptr<scalar_t>(), out.data_ptr<scalar_t>());
                                   }));

        return out.to(inp_dtype);
    }

    TORCH_LIBRARY_IMPL(disco_kernels, CPU, m) { m.impl("backward_ragged", &disco_cpu_bwd_ragged); }

} // namespace disco_kernels
