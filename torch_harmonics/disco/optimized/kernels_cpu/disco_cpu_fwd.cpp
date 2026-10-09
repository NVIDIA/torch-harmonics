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

// The DISCO contraction on the CPU: forward_regular, a gather, with psi in arc form -- see
// torch_harmonics/disco/_psi_layouts.py and, for the CUDA counterpart,
// kernels_cuda/disco_cuda_fwd.cu.

#include "disco_cpu.h"

namespace disco_kernels
{

    // Gather, pscale = Wi / Wo. Parallel over (batch, channel, row): every row writes its
    // own output row. PSCALE > 0 makes the stride a compile-time constant, so the multiply
    // folds and PSCALE == 1 becomes a contiguous SIMD load.
    template <typename scalar_t, int PSCALE>
    static void disco_fwd_cpu_impl(int64_t B, int64_t C, int64_t K, int64_t Hi, int64_t Wi, int64_t Ho, int64_t Wo,
                                   int64_t pscale_runtime, const ArcPsiCpu psi, const scalar_t *__restrict__ vals,
                                   const scalar_t *__restrict__ inp, scalar_t *__restrict__ out)
    {
        const int64_t pscale = (PSCALE != 0) ? static_cast<int64_t>(PSCALE) : pscale_runtime;

#pragma omp parallel
        {
            std::vector<scalar_t> out_tmp(Wo);
            // the input ring twice, so reads at w + pscale*wo < 2*Wi need no modulo
            std::vector<scalar_t> sh(2 * Wi);
            scalar_t *__restrict__ sh_ptr = sh.data();

#pragma omp for collapse(3)
            for (int64_t b = 0; b < B; b++) {
                for (int64_t c = 0; c < C; c++) {
                    for (int64_t r = 0; r < psi.nrows; r++) {

                        const scalar_t *__restrict__ inp_bc = inp + (b * C + c) * Hi * Wi;
                        std::fill(out_tmp.begin(), out_tmp.end(), scalar_t(0));

                        int64_t v = psi.val_off[r];
                        int64_t h_prev = -1;
                        for (int64_t s = psi.seg_off[r]; s < psi.seg_off[r + 1]; s++) {

                            const int64_t ring = psi.seg[3 * s + 0];
                            const int64_t start = psi.seg[3 * s + 1];
                            const int64_t len = psi.seg[3 * s + 2];

                            if (ring != h_prev) {
                                h_prev = ring;
                                const scalar_t *inp_row = inp_bc + ring * Wi;
                                std::copy(inp_row, inp_row + Wi, sh_ptr);
                                std::copy(inp_row, inp_row + Wi, sh_ptr + Wi);
                            }

                            int64_t w = start;
                            for (int64_t j = 0; j < len; j++) {
                                const scalar_t val = vals[v++];
#pragma omp simd
                                for (int64_t wo = 0; wo < Wo; wo++) { out_tmp[wo] += val * sh_ptr[w + pscale * wo]; }
                                if (++w == Wi) w = 0;
                            }
                        }

                        scalar_t *__restrict__ out_row
                            = out + (((b * C + c) * K + psi.row_ker[r]) * Ho + psi.row_lat[r]) * Wo;
#pragma omp simd
                        for (int64_t wo = 0; wo < Wo; wo++) { out_row[wo] = out_tmp[wo]; }
                    }
                }
            }
        }
    }

    torch::Tensor disco_cpu_fwd(torch::Tensor inp, torch::Tensor row_ker, torch::Tensor row_lat, torch::Tensor seg_off,
                                torch::Tensor seg, torch::Tensor val_off, torch::Tensor vals, int64_t K, int64_t Ho,
                                int64_t Wo)
    {
        TORCH_CHECK(inp.device().is_cpu(), "inp must be a CPU tensor, got ", inp.device());
        check_forward_inputs(inp, row_ker, row_lat, seg_off, seg, val_off, vals, K, Ho, Wo);

        const auto inp_dtype = inp.scalar_type();
        inp = inp.to(compute_dtype(inp_dtype)).contiguous();

        const int64_t B = inp.size(0), C = inp.size(1), Hi = inp.size(2), Wi = inp.size(3);
        auto out = torch::zeros({B, C, K, Ho, Wo}, inp.options());
        const ArcPsiCpu psi = arc_psi_cpu(row_ker, row_lat, seg_off, seg, val_off);
        const int64_t pscale = Wi / Wo;

        AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_forward_cpu", ([&] {
                                       DISCO_PSCALE_DISPATCH(disco_fwd_cpu_impl, B, C, K, Hi, Wi, Ho, Wo, pscale, psi,
                                                             vals.data_ptr<scalar_t>(), inp.data_ptr<scalar_t>(),
                                                             out.data_ptr<scalar_t>());
                                   }));

        return out.to(inp_dtype);
    }

    TORCH_LIBRARY_IMPL(disco_kernels, CPU, m) { m.impl("forward_regular", &disco_cpu_fwd); }

} // namespace disco_kernels
