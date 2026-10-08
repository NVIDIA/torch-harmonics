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

// The transpose of the DISCO contraction on CUDA: backward_regular, a scatter, with psi in
// arc form -- see torch_harmonics/disco/_psi.py. The gather is in disco_cuda_fwd.cu.
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
    // backward: scatter along the longitude with stride pscale = Wo / Wi
    // =================================================================================

    // Shared-accumulator body, for pscale >= 3. Each thread holds ELXTH input longitudes
    // in registers and adds into a pscale-way shared copy of the output ring,
    //   __sh[w % pscale][w / pscale + q] += val * inp[q],
    // doubled like the forward's staging so the offset needs no modulo. Every entry is a
    // shared read-modify-write followed by a barrier, so consecutive entries cannot race;
    // the ring is flushed to global memory with atomics when the arcs move to the next
    // output ring, since other blocks scatter into it too.
    template <int BDIM_X, int ELXTH, typename STORAGE_T, typename COMPUTE_T>
    __device__ void disco_bwd_d(const int Hi, const int Wi, const int K, const int Ho, const int Wo, const int pscale,
                                const int32_t *__restrict__ row_ker, const int32_t *__restrict__ row_lat,
                                const int64_t *__restrict__ seg_off, const int32_t *__restrict__ seg,
                                const int64_t *__restrict__ val_off, const COMPUTE_T *__restrict__ vals,
                                const STORAGE_T *__restrict__ inp, COMPUTE_T *__restrict__ out)
    {
        const int tid = threadIdx.x;
        const int64_t bidx = blockIdx.x;
        const int64_t bidy = blockIdx.y;

        const int64_t sbeg = seg_off[bidx];
        const int64_t send = seg_off[bidx + 1];
        if (sbeg == send) return; // uniform across the block
        int64_t v = val_off[bidx];

        const int64_t ker = row_ker[bidx];
        const int64_t lat = row_lat[bidx];

        inp += bidy * K * Hi * Wi + ker * Hi * Wi + lat * Wi;
        out += bidy * Ho * Wo;

        // COMPUTE_T __sh[pscale][2*BDIM_X*ELXTH]
        extern __shared__ __align__(sizeof(double)) unsigned char __sh_ptr[];
        COMPUTE_T(*__sh)[BDIM_X * ELXTH * 2] = reinterpret_cast<COMPUTE_T(*)[BDIM_X * ELXTH * 2]>(__sh_ptr);

        COMPUTE_T __reg[ELXTH];
#pragma unroll
        for (int i = 0; i < ELXTH; i++) {
            __reg[i]
                = (i * BDIM_X + tid < Wi) ? static_cast<COMPUTE_T>(inp[i * BDIM_X + tid]) : static_cast<COMPUTE_T>(0);
        }

        // the entries past 2*Wi are written but never flushed
        for (int i = 0; i < pscale; i++) {
#pragma unroll
            for (int j = 0; j < 2 * BDIM_X * ELXTH; j += BDIM_X) { __sh[i][j + tid] = static_cast<COMPUTE_T>(0); }
        }
        __syncthreads();

        int h_prev = seg[3 * sbeg];
        for (int64_t s = sbeg; s < send; s++) {

            const int ring = seg[3 * s + 0];
            const int start = seg[3 * s + 1];
            const int len = seg[3 * s + 2];

            if (ring != h_prev) {
                __syncthreads();
                for (int i = 0; i < pscale; i++) {
                    for (int j = tid; j < Wi; j += BDIM_X) {
                        const COMPUTE_T x = __sh[i][j] + __sh[i][Wi + j];
                        atomicAdd(&out[h_prev * Wo + j * pscale + i], x);
                        __sh[i][j] = static_cast<COMPUTE_T>(0);
                        __sh[i][Wi + j] = static_cast<COMPUTE_T>(0);
                    }
                }
                __syncthreads();
                h_prev = ring;
            }

            int w = start;
            for (int j = 0; j < len; j++) {
                const COMPUTE_T val = vals[v++];
                const int w_mod_ps = w % pscale;
                const int w_div_ps = w / pscale;
#pragma unroll
                for (int i = 0; i < ELXTH; i++) {
                    const int pp = i * BDIM_X + tid;
                    __sh[w_mod_ps][w_div_ps + pp] += val * __reg[i];
                }
                // consecutive entries may hit the same shared locations
                __syncthreads();
                if (++w == Wo) w = 0;
            }
        }
        __syncthreads();

        for (int i = 0; i < pscale; i++) {
            for (int j = tid; j < Wi; j += BDIM_X) {
                const COMPUTE_T x = __sh[i][j] + __sh[i][Wi + j];
                atomicAdd(&out[h_prev * Wo + j * pscale + i], x);
            }
        }
    }

    // Transposed-ownership body ("the swap"), for pscale 1 and 2.
    //
    // disco_bwd_d gives each thread a slice of the INPUT row in registers and accumulates
    // into a shared OUTPUT row, so every entry is a shared read-modify-write plus a
    // barrier. This swaps the two: shared holds the input row, written once and read-only
    // thereafter, and registers hold the accumulator. Same sum, transposed ownership:
    //
    //   disco_bwd_d : __sh[w_mod][w_div + q] += val * inp[q]         q owned by the thread
    //   here        : acc[w_mod][t]          += val * inp[t - w_div]  t owned by the thread
    //
    // The inner loop becomes LDS + FFMA instead of LDS + FADD + STS, and the only barrier
    // left is the one after the shared fill.
    //
    // Gated to PSCALE <= 2. The accumulator costs PSCALE*ELXTH registers, and at pscale 3
    // (36 registers for Wi=720) occupancy drops far enough that the kernel loses despite
    // executing ~10% fewer instructions.
    template <int BDIM_X, int ELXTH, int PSCALE, typename STORAGE_T, typename COMPUTE_T>
    __device__ void disco_bwd_swap_d(const int Hi, const int Wi, const int K, const int Ho, const int Wo,
                                     const int32_t *__restrict__ row_ker, const int32_t *__restrict__ row_lat,
                                     const int64_t *__restrict__ seg_off, const int32_t *__restrict__ seg,
                                     const int64_t *__restrict__ val_off, const COMPUTE_T *__restrict__ vals,
                                     const STORAGE_T *__restrict__ inp, COMPUTE_T *__restrict__ out)
    {
        const int tid = threadIdx.x;
        const int64_t bidx = blockIdx.x;
        const int64_t bidy = blockIdx.y;

        const int64_t sbeg = seg_off[bidx];
        const int64_t send = seg_off[bidx + 1];
        if (sbeg == send) return; // uniform across the block
        int64_t v = val_off[bidx];

        const int64_t ker = row_ker[bidx];
        const int64_t lat = row_lat[bidx];

        inp += bidy * K * Hi * Wi + ker * Hi * Wi + lat * Wi;
        out += bidy * Ho * Wo;

        // the input row, doubled so the wraparound is an offset rather than a modulo
        extern __shared__ __align__(sizeof(double)) unsigned char __sh_ptr[];
        COMPUTE_T *sh_inp = reinterpret_cast<COMPUTE_T *>(__sh_ptr);

        constexpr int SH_LEN = 2 * BDIM_X * ELXTH;
        for (int j = tid; j < Wi; j += BDIM_X) {
            const COMPUTE_T x = static_cast<COMPUTE_T>(inp[j]);
            sh_inp[j] = x;
            sh_inp[Wi + j] = x;
        }
        // Lanes with t >= Wi still issue their read, at Wi + t - w_div, which can run past
        // 2*Wi. BDIM_X*ELXTH >= Wi so SH_LEN covers it, but the tail would otherwise be
        // uninitialised. Zeroing it once keeps the inner loop branch-free; those lanes'
        // accumulators are discarded at flush anyway.
        for (int j = 2 * Wi + tid; j < SH_LEN; j += BDIM_X) { sh_inp[j] = static_cast<COMPUTE_T>(0); }
        __syncthreads(); // the only barrier in the kernel

        COMPUTE_T acc[PSCALE][ELXTH];
#pragma unroll
        for (int m = 0; m < PSCALE; m++) {
#pragma unroll
            for (int i = 0; i < ELXTH; i++) acc[m][i] = static_cast<COMPUTE_T>(0);
        }

        int h_prev = seg[3 * sbeg];
        for (int64_t s = sbeg; s < send; s++) {

            const int ring = seg[3 * s + 0];
            const int start = seg[3 * s + 1];
            const int len = seg[3 * s + 2];

            // a new output ring: flush straight from registers. No barrier needed -- acc is
            // private and sh_inp is read-only.
            if (ring != h_prev) {
#pragma unroll
                for (int m = 0; m < PSCALE; m++) {
#pragma unroll
                    for (int i = 0; i < ELXTH; i++) {
                        const int t = i * BDIM_X + tid;
                        if (t < Wi) { atomicAdd(&out[h_prev * Wo + t * PSCALE + m], acc[m][i]); }
                        acc[m][i] = static_cast<COMPUTE_T>(0);
                    }
                }
                h_prev = ring;
            }

            int w = start;
            for (int j = 0; j < len; j++) {
                const COMPUTE_T val = vals[v++];
                const int w_mod_ps = w % PSCALE;
                const int w_div_ps = w / PSCALE;

                // The bank select is the OUTER loop, with the whole element loop inside it.
                // w_mod_ps derives only from the arc -- never from tid -- so it is
                // CTA-uniform and this compiles to a branch, not predication. With the
                // element loop outside and a one-instruction body inside, ptxas predicates
                // instead and the kernel issues PSCALE adds per element instead of one,
                // which turns a memory-bound kernel into an issue-bound one.
#pragma unroll
                for (int m = 0; m < PSCALE; m++) {
                    if (m == w_mod_ps) {
#pragma unroll
                        for (int i = 0; i < ELXTH; i++) {
                            acc[m][i] += val * sh_inp[Wi + (i * BDIM_X + tid) - w_div_ps];
                        }
                    }
                }
                if (++w == Wo) w = 0;
            }
        }

#pragma unroll
        for (int m = 0; m < PSCALE; m++) {
#pragma unroll
            for (int i = 0; i < ELXTH; i++) {
                const int t = i * BDIM_X + tid;
                if (t < Wi) { atomicAdd(&out[h_prev * Wo + t * PSCALE + m], acc[m][i]); }
            }
        }
    }

    template <int BDIM_X, int ELXTH, int PSCALE, typename STORAGE_T, typename COMPUTE_T>
    __global__ __launch_bounds__(BDIM_X) void disco_bwd_blk_k(
        const int Hi, const int Wi, const int K, const int Ho, const int Wo, const int pscale,
        const int32_t *__restrict__ row_ker, const int32_t *__restrict__ row_lat, const int64_t *__restrict__ seg_off,
        const int32_t *__restrict__ seg, const int64_t *__restrict__ val_off, const COMPUTE_T *__restrict__ vals,
        const STORAGE_T *__restrict__ inp, COMPUTE_T *__restrict__ out)
    {
        if constexpr (PSCALE != 0 && PSCALE <= 2) {
            disco_bwd_swap_d<BDIM_X, ELXTH, PSCALE, STORAGE_T, COMPUTE_T>(Hi, Wi, K, Ho, Wo, row_ker, row_lat, seg_off,
                                                                          seg, val_off, vals, inp, out);
        } else {
            disco_bwd_d<BDIM_X, ELXTH, STORAGE_T, COMPUTE_T>(Hi, Wi, K, Ho, Wo, (PSCALE != 0) ? PSCALE : pscale,
                                                             row_ker, row_lat, seg_off, seg, val_off, vals, inp, out);
        }
    }

    // =================================================================================
    // launch
    // =================================================================================

    // Grow ELXTH until NTH*ELXTH covers the row the block holds, then dispatch on pscale:
    // 1, 2 and 3 get compile-time instantiations, anything else the runtime stride. The
    // shared memory grows with the row, so every launch opts in beyond the default 48 KiB
    // (th_cuda::launch_dyn_shmem), and says so if even the opt-in cannot serve it.
    template <int NTH, int ELXTH, typename STORAGE_T, typename COMPUTE_T>
    static void launch_bwd(int BC, int Hi, int Wi, int K, int Ho, int Wo, const ArcPsi &psi, const COMPUTE_T *vals,
                           const STORAGE_T *inp, COMPUTE_T *out, cudaStream_t stream)
    {
        if constexpr (ELXTH <= ELXTH_MAX) {
            if (NTH * ELXTH >= Wi) {
                const dim3 grid(psi.nrows, BC);
                const int pscale = Wo / Wi;
                // the swap (pscale <= 2) stores the input row: 2*NTH*ELXTH whatever the
                // pscale; the shared-accumulator body stores pscale copies of the output ring
                const int sh_banks = (pscale <= 2) ? 1 : pscale;
                const size_t shmem = sizeof(COMPUTE_T) * (2 * (NTH * ELXTH) * sh_banks);
#define DISCO_BWD_LAUNCH(PS)                                                                                           \
    th_cuda::launch_dyn_shmem(&disco_bwd_blk_k<NTH, ELXTH, PS, STORAGE_T, COMPUTE_T>, grid, dim3(NTH), shmem, stream,  \
                              "disco backward", "the request grows with nlon_in, and with pscale beyond 2", Hi, Wi, K, \
                              Ho, Wo, pscale, psi.row_ker, psi.row_lat, psi.seg_off, psi.seg, psi.val_off, vals, inp,  \
                              out)
                switch (pscale) {
                case 1: DISCO_BWD_LAUNCH(1); break;
                case 2: DISCO_BWD_LAUNCH(2); break;
                case 3: DISCO_BWD_LAUNCH(3); break;
                default: DISCO_BWD_LAUNCH(0); break;
                }
#undef DISCO_BWD_LAUNCH
            } else {
                launch_bwd<NTH, ELXTH + 1, STORAGE_T, COMPUTE_T>(BC, Hi, Wi, K, Ho, Wo, psi, vals, inp, out, stream);
            }
        }
    }

    torch::Tensor disco_cuda_bwd(torch::Tensor inp, torch::Tensor row_ker, torch::Tensor row_lat, torch::Tensor seg_off,
                                 torch::Tensor seg, torch::Tensor val_off, torch::Tensor vals, int64_t K, int64_t Ho,
                                 int64_t Wo)
    {
        TORCH_CHECK(inp.device().is_cuda(), "inp must be a CUDA tensor, got ", inp.device());
        check_backward_inputs(inp, row_ker, row_lat, seg_off, seg, val_off, vals, K, Ho, Wo);

        // launch on the inputs' device, not whichever one is current
        const at::cuda::OptionalCUDAGuard device_guard(inp.device());

        const int64_t BC = inp.size(0) * inp.size(1);
        const int64_t Hi = inp.size(3);
        const int64_t Wi = inp.size(4);

        // The scatter accumulates into `out` from many blocks, so partial sums must not
        // round in between: the buffer is in the compute dtype (vals's, e.g. fp32 for
        // fp16/bf16 activations) and narrowed to the activations' dtype on return, which is
        // what every host returns and what the registered fake promises.
        auto out = torch::zeros({inp.size(0), inp.size(1), Ho, Wo}, inp.options().dtype(vals.dtype()));
        if (row_ker.size(0) > 0 && BC > 0) {
            const ArcPsi psi = arc_psi(row_ker, row_lat, seg_off, seg, val_off);
            auto stream = at::cuda::getCurrentCUDAStream().stream();

            // The block holds the INPUT row -- in registers, or in shared memory for the
            // swap -- so its shape follows Wi, not Wo. Sizing it from Wo overshot whenever
            // Wo > 64*ELXTH_MAX >= Wi: at Wi=720 the 128-lane config held 2176 elements, 3x
            // what it needs, which times pscale overran the 48 KB static shared limit for
            // Wo > 2048 and pscale >= 3 (e.g. 1080x2160 -> 360x720).
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kHalf, at::kBFloat16, inp.scalar_type(), "disco_backward_cuda", ([&] {
                    using storage_t = scalar_t;
                    using compute_t = typename at::opmath_type<storage_t>;
                    with_block_shape(Wi, "disco backward: nlon_in", [&](auto nth, auto elxth) {
                        launch_bwd<decltype(nth)::value, decltype(elxth)::value, storage_t, compute_t>(
                            BC, Hi, Wi, K, Ho, Wo, psi, vals.data_ptr<compute_t>(), inp.data_ptr<storage_t>(),
                            out.data_ptr<compute_t>(), stream);
                    });
                }));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return out.to(inp.scalar_type());
    }

    TORCH_LIBRARY_IMPL(disco_kernels, CUDA, m) { m.impl("backward_regular", &disco_cuda_bwd); }

} // namespace disco_kernels
