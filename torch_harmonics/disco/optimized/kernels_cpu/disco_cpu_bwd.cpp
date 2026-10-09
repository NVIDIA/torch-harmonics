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

// The transpose of the DISCO contraction on the CPU: backward_regular, a scatter, with psi
// in arc form -- see torch_harmonics/disco/_psi_layouts.py and, for the CUDA counterpart,
// kernels_cuda/disco_cuda_bwd.cu.

#include "disco_cpu.h"

namespace disco_kernels
{

    // Scatter, pscale = Wo / Wi. Parallel over (batch, channel) only: different rows scatter
    // into the same output positions, so collapsing the rows into the parallel loop would
    // race. The output ring is accumulated in pscale lanes of a doubled buffer and flushed
    // only when it changes -- across rows too, since consecutive rows often land on the
    // same output ring.
    template <typename scalar_t, int PSCALE>
    static void disco_bwd_cpu_impl(int64_t B, int64_t C, int64_t K, int64_t Hi, int64_t Wi, int64_t Ho, int64_t Wo,
                                   int64_t pscale_runtime, const ArcPsiCpu psi, const scalar_t *__restrict__ vals,
                                   const scalar_t *__restrict__ inp, scalar_t *__restrict__ out)
    {
        const int64_t pscale = (PSCALE != 0) ? static_cast<int64_t>(PSCALE) : pscale_runtime;
        const int64_t lane_size = 2 * Wi;

#pragma omp parallel
        {
            std::vector<scalar_t> sh(2 * Wo, scalar_t(0));
            scalar_t *__restrict__ sh_ptr = sh.data();

            auto flush = [&](scalar_t *__restrict__ flush_row) {
#pragma omp simd
                for (int64_t idx = 0; idx < Wi; idx++) {
                    for (int64_t lane = 0; lane < pscale; lane++) {
                        const int64_t off = lane * lane_size;
                        flush_row[idx * pscale + lane] += sh_ptr[off + idx] + sh_ptr[off + Wi + idx];
                        sh_ptr[off + idx] = scalar_t(0);
                        sh_ptr[off + Wi + idx] = scalar_t(0);
                    }
                }
            };

#pragma omp for collapse(2)
            for (int64_t b = 0; b < B; b++) {
                for (int64_t c = 0; c < C; c++) {

                    scalar_t *__restrict__ out_bc = out + (b * C + c) * Ho * Wo;
                    int64_t ho_prev = -1;

                    for (int64_t r = 0; r < psi.nrows; r++) {

                        const scalar_t *__restrict__ inp_row
                            = inp + (((b * C + c) * K + psi.row_ker[r]) * Hi + psi.row_lat[r]) * Wi;

                        int64_t v = psi.val_off[r];
                        for (int64_t s = psi.seg_off[r]; s < psi.seg_off[r + 1]; s++) {

                            const int64_t ring = psi.seg[3 * s + 0];
                            const int64_t start = psi.seg[3 * s + 1];
                            const int64_t len = psi.seg[3 * s + 2];

                            if (ring != ho_prev) {
                                if (ho_prev != -1) { flush(out_bc + ho_prev * Wo); }
                                ho_prev = ring;
                            }

                            int64_t wo = start;
                            for (int64_t j = 0; j < len; j++) {
                                const scalar_t val = vals[v++];
                                scalar_t *__restrict__ sh_lane = sh_ptr + (wo % pscale) * lane_size + wo / pscale;
#pragma omp simd
                                for (int64_t wi = 0; wi < Wi; wi++) { sh_lane[wi] += val * inp_row[wi]; }
                                if (++wo == Wo) wo = 0;
                            }
                        }
                    }

                    if (ho_prev != -1) { flush(out_bc + ho_prev * Wo); }
                }
            }
        }
    }

    torch::Tensor disco_cpu_bwd(torch::Tensor inp, torch::Tensor row_ker, torch::Tensor row_lat, torch::Tensor seg_off,
                                torch::Tensor seg, torch::Tensor val_off, torch::Tensor vals, int64_t K, int64_t Ho,
                                int64_t Wo)
    {
        TORCH_CHECK(inp.device().is_cpu(), "inp must be a CPU tensor, got ", inp.device());
        check_backward_inputs(inp, row_ker, row_lat, seg_off, seg, val_off, vals, K, Ho, Wo);

        const auto inp_dtype = inp.scalar_type();
        inp = inp.to(compute_dtype(inp_dtype)).contiguous();

        const int64_t B = inp.size(0), C = inp.size(1), Hi = inp.size(3), Wi = inp.size(4);
        auto out = torch::zeros({B, C, Ho, Wo}, inp.options());
        const ArcPsiCpu psi = arc_psi_cpu(row_ker, row_lat, seg_off, seg, val_off);
        const int64_t pscale = Wo / Wi;

        AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_backward_cpu", ([&] {
                                       DISCO_PSCALE_DISPATCH(disco_bwd_cpu_impl, B, C, K, Hi, Wi, Ho, Wo, pscale, psi,
                                                             vals.data_ptr<scalar_t>(), inp.data_ptr<scalar_t>(),
                                                             out.data_ptr<scalar_t>());
                                   }));

        return out.to(inp_dtype);
    }

    TORCH_LIBRARY_IMPL(disco_kernels, CPU, m) { m.impl("backward_regular", &disco_cpu_bwd); }

} // namespace disco_kernels
