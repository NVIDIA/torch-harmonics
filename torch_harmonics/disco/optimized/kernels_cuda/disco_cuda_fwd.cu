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

// The DISCO contraction on CUDA: forward_regular, a gather, with psi in arc form -- see
// torch_harmonics/disco/_psi.py. Its transpose, the scatter, is in disco_cuda_bwd.cu.
//
// One block per (psi row, batch*channel). A row carries its basis function and latitude
// once and walks arcs (ring, start, length) whose values lie consecutively; the longitude
// within an arc is a counter with a compare-subtract wrap. The p-shift of the output
// longitude is a stride through a row held in shared memory or registers, so the per-arc
// work is independent of the longitude it serves.
//
// These replaced kernels that walked psi in CSR form, reading a flat column per nonzero
// and dividing it into ring and longitude, with the same block shapes, shared memory and
// pscale specializations. Only the decode changed, and it measured 1.05-1.67x faster on
// H100 and GB200 (fp32 and bf16, forward and backward, 1 degree to the production
// encoder), with psi 5-6.5x smaller.

#include "../disco.h"
#include "disco_cuda.cuh"
#include "../../../csrc/cuda_launch.cuh"

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <c10/cuda/CUDAException.h>

namespace disco_kernels
{

    // =================================================================================
    // forward: gather along the longitude with stride pscale = Wi / Wo
    // =================================================================================
    //
    // Each thread owns ELXTH output longitudes pp = i*BDIM_X + tid and accumulates in
    // registers. The input ring an arc lies on is staged in shared memory twice, so the read
    // at w + pscale*pp needs no modulo: w < Wi and pscale*pp < pscale*BDIM_X*ELXTH, which
    // bounds the index by 2*Wi + pscale*(BDIM_X*ELXTH - Wo) -- the shared allocation. The
    // lanes past Wo read that tail unconditionally, which keeps the inner loop branch-free,
    // and are dropped at the store.
    template <int BDIM_X, int ELXTH, typename STORAGE_T, typename COMPUTE_T>
    __device__ void disco_fwd_d(const int Hi, const int Wi, const int K, const int Ho, const int Wo, const int pscale,
                                const int32_t *__restrict__ row_ker, const int32_t *__restrict__ row_lat,
                                const int64_t *__restrict__ seg_off, const int32_t *__restrict__ seg,
                                const int64_t *__restrict__ val_off, const COMPUTE_T *__restrict__ vals,
                                const STORAGE_T *__restrict__ inp, STORAGE_T *__restrict__ out)
    {
        const int tid = threadIdx.x;
        const int64_t bidx = blockIdx.x; // psi row
        const int64_t bidy = blockIdx.y; // batch * channel

        const int64_t sbeg = seg_off[bidx];
        const int64_t send = seg_off[bidx + 1];
        int64_t v = val_off[bidx];

        const int64_t ker = row_ker[bidx];
        const int64_t lat = row_lat[bidx];

        inp += bidy * Hi * Wi;
        out += bidy * K * Ho * Wo + ker * Ho * Wo + lat * Wo;

        COMPUTE_T __reg[ELXTH] = {0};

        // STORAGE_T __sh[2*Wi + pscale*(BDIM_X*ELXTH - Wo)], aligned for the widest type
        extern __shared__ __align__(sizeof(double)) unsigned char __sh_ptr[];
        STORAGE_T *__sh = reinterpret_cast<STORAGE_T *>(__sh_ptr);

        int h_prev = -1;
        for (int64_t s = sbeg; s < send; s++) {

            const int ring = seg[3 * s + 0];
            const int start = seg[3 * s + 1];
            const int len = seg[3 * s + 2];

            // arcs are sorted by ring, so the ring is staged once per ring, not per arc
            if (ring != h_prev) {
                h_prev = ring;
                __syncthreads();
                for (int i = tid; i < Wi; i += BDIM_X) {
                    const STORAGE_T x = inp[ring * Wi + i];
                    __sh[i] = x;
                    __sh[Wi + i] = x;
                }
                __syncthreads();
            }

            int w = start;
            for (int j = 0; j < len; j++) {
                const COMPUTE_T val = vals[v++];
#pragma unroll
                for (int i = 0; i < ELXTH; i++) {
                    const int pp = i * BDIM_X + tid;
                    __reg[i] += val * static_cast<COMPUTE_T>(__sh[w + pscale * pp]);
                }
                if (++w == Wi) w = 0;
            }
        }

#pragma unroll
        for (int i = 0; i < ELXTH; i++) {
            const int pp = i * BDIM_X + tid;
            if (pp >= Wo) break;
            out[pp] = static_cast<STORAGE_T>(__reg[i]);
        }
    }

    template <int BDIM_X, int ELXTH, int PSCALE, typename STORAGE_T, typename COMPUTE_T>
    __global__ __launch_bounds__(BDIM_X) void disco_fwd_blk_k(
        const int Hi, const int Wi, const int K, const int Ho, const int Wo, const int pscale,
        const int32_t *__restrict__ row_ker, const int32_t *__restrict__ row_lat, const int64_t *__restrict__ seg_off,
        const int32_t *__restrict__ seg, const int64_t *__restrict__ val_off, const COMPUTE_T *__restrict__ vals,
        const STORAGE_T *__restrict__ inp, STORAGE_T *__restrict__ out)
    {
        // PSCALE > 0 makes the stride a compile-time constant; 0 falls back to the runtime value
        disco_fwd_d<BDIM_X, ELXTH, STORAGE_T, COMPUTE_T>(Hi, Wi, K, Ho, Wo, (PSCALE != 0) ? PSCALE : pscale, row_ker,
                                                         row_lat, seg_off, seg, val_off, vals, inp, out);
    }

    // =================================================================================
    // launch
    // =================================================================================

    // Grow ELXTH until NTH*ELXTH covers the row the block holds, then dispatch on pscale:
    // 1, 2 and 3 get compile-time instantiations, anything else the runtime stride. The
    // shared memory grows with the row, so every launch opts in beyond the default 48 KiB
    // (th_cuda::launch_dyn_shmem), and says so if even the opt-in cannot serve it.
    template <int NTH, int ELXTH, typename STORAGE_T, typename COMPUTE_T>
    static void launch_fwd(int BC, int Hi, int Wi, int K, int Ho, int Wo, const ArcPsi &psi, const COMPUTE_T *vals,
                           const STORAGE_T *inp, STORAGE_T *out, cudaStream_t stream)
    {
        if constexpr (ELXTH <= ELXTH_MAX) {
            if (NTH * ELXTH >= Wo) {
                const dim3 grid(psi.nrows, BC);
                const int pscale = Wi / Wo;
                const size_t shmem = sizeof(STORAGE_T) * (Wi * 2 + pscale * (NTH * ELXTH - Wo));
#define DISCO_FWD_LAUNCH(PS)                                                                                           \
    th_cuda::launch_dyn_shmem(&disco_fwd_blk_k<NTH, ELXTH, PS, STORAGE_T, COMPUTE_T>, grid, dim3(NTH), shmem, stream,  \
                              "disco forward", "the request grows with nlon_in", Hi, Wi, K, Ho, Wo, pscale,            \
                              psi.row_ker, psi.row_lat, psi.seg_off, psi.seg, psi.val_off, vals, inp, out)
                switch (pscale) {
                case 1: DISCO_FWD_LAUNCH(1); break;
                case 2: DISCO_FWD_LAUNCH(2); break;
                case 3: DISCO_FWD_LAUNCH(3); break;
                default: DISCO_FWD_LAUNCH(0); break;
                }
#undef DISCO_FWD_LAUNCH
            } else {
                launch_fwd<NTH, ELXTH + 1, STORAGE_T, COMPUTE_T>(BC, Hi, Wi, K, Ho, Wo, psi, vals, inp, out, stream);
            }
        }
    }

    torch::Tensor disco_cuda_fwd(torch::Tensor inp, torch::Tensor row_ker, torch::Tensor row_lat, torch::Tensor seg_off,
                                 torch::Tensor seg, torch::Tensor val_off, torch::Tensor vals, int64_t K, int64_t Ho,
                                 int64_t Wo)
    {
        TORCH_CHECK(inp.device().is_cuda(), "inp must be a CUDA tensor, got ", inp.device());
        check_forward_inputs(inp, row_ker, row_lat, seg_off, seg, val_off, vals, K, Ho, Wo);

        // launch on the inputs' device, not whichever one is current
        const at::cuda::OptionalCUDAGuard device_guard(inp.device());

        const int64_t BC = inp.size(0) * inp.size(1);
        const int64_t Hi = inp.size(2);
        const int64_t Wi = inp.size(3);

        // the output is written in the storage dtype, one row per block and no overlap
        auto out = torch::zeros({inp.size(0), inp.size(1), K, Ho, Wo}, inp.options());
        if (row_ker.size(0) == 0 || BC == 0) return out;

        const ArcPsi psi = arc_psi(row_ker, row_lat, seg_off, seg, val_off);
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        // the block holds the output row, so its shape follows Wo
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, inp.scalar_type(), "disco_forward_cuda", ([&] {
                using storage_t = scalar_t;
                using compute_t = typename at::opmath_type<storage_t>;
                with_block_shape(Wo, "disco forward: nlon_out", [&](auto nth, auto elxth) {
                    launch_fwd<decltype(nth)::value, decltype(elxth)::value, storage_t, compute_t>(
                        BC, Hi, Wi, K, Ho, Wo, psi, vals.data_ptr<compute_t>(), inp.data_ptr<storage_t>(),
                        out.data_ptr<storage_t>(), stream);
                });
            }));

        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }

    TORCH_LIBRARY_IMPL(disco_kernels, CUDA, m) { m.impl("forward_regular", &disco_cuda_fwd); }

} // namespace disco_kernels
