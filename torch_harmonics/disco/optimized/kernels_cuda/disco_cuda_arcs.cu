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

// The CSR kernels of disco_cuda_fwd.cu and disco_cuda_bwd.cu with psi in arc form.
//
// Deliberately the same kernels: one block per (psi row, batch*channel), the same block
// shapes, shared-memory layouts and pscale specializations, and the same two backward
// bodies (the transposed-ownership "swap" for pscale <= 2, the shared accumulator
// otherwise). Only the decode of psi differs. The CSR form stores a basis function,
// latitude and flat column per nonzero and derives (ring, longitude) from the column; here
// a row stores its basis function and latitude once and walks arcs (ring, start, length),
// whose values lie consecutively, so the longitude is a counter with a compare-subtract
// wrap. That keeps an A/B against the CSR kernels a measurement of the representation
// alone. Once the arc form is adopted the CSR kernels go, and so does the duplication.

#include "../disco.h"
#include "disco_cuda.cuh"

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <c10/cuda/CUDAException.h>

#include <type_traits>

namespace disco_kernels
{

    // ---------------------------------------------------------------------------------
    // forward: gather along the longitude with stride pscale = Wi / Wo
    // ---------------------------------------------------------------------------------

    template <int BDIM_X, int ELXTH, typename STORAGE_T, typename COMPUTE_T>
    __device__ void disco_fwd_arcs_d(const int Hi, const int Wi, const int K, const int Ho, const int Wo,
                                     const int pscale, const int32_t *__restrict__ row_ker,
                                     const int32_t *__restrict__ row_lat, const int64_t *__restrict__ seg_off,
                                     const int32_t *__restrict__ seg, const int64_t *__restrict__ val_off,
                                     const COMPUTE_T *__restrict__ vals, const STORAGE_T *__restrict__ inp,
                                     STORAGE_T *__restrict__ out)
    {
        const int tid = threadIdx.x;
        const int64_t bidx = blockIdx.x; // psi row
        const int64_t bidy = blockIdx.y; // bc

        const int64_t sbeg = seg_off[bidx];
        const int64_t send = seg_off[bidx + 1];
        int64_t v = val_off[bidx];

        const int64_t ker = row_ker[bidx];
        const int64_t lat = row_lat[bidx];

        inp += bidy * Hi * Wi;
        out += bidy * K * Ho * Wo + ker * Ho * Wo + lat * Wo;

        COMPUTE_T __reg[ELXTH] = {0};

        // STORAGE_T __sh[2*Wi + pscale*(BDIM_X*ELXTH - Wo)], as in disco_fwd_d
        extern __shared__ __align__(sizeof(double)) unsigned char __sh_ptr[];
        STORAGE_T *__sh = reinterpret_cast<STORAGE_T *>(__sh_ptr);

        int h_prev = -1;
        for (int64_t s = sbeg; s < send; s++) {

            const int ring = seg[3 * s + 0];
            const int start = seg[3 * s + 1];
            const int len = seg[3 * s + 2];

            // a new input ring: stage it twice, so reads at w + pscale*pp need no modulo
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

            // the arc's longitudes: w stays in [0, Wi), so w + pscale*pp < 2*Wi + pscale*(BDIM_X*ELXTH - Wo)
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
    __global__ __launch_bounds__(BDIM_X) void disco_fwd_arcs_blk_k(
        const int Hi, const int Wi, const int K, const int Ho, const int Wo, const int pscale,
        const int32_t *__restrict__ row_ker, const int32_t *__restrict__ row_lat, const int64_t *__restrict__ seg_off,
        const int32_t *__restrict__ seg, const int64_t *__restrict__ val_off, const COMPUTE_T *__restrict__ vals,
        const STORAGE_T *__restrict__ inp, STORAGE_T *__restrict__ out)
    {
        disco_fwd_arcs_d<BDIM_X, ELXTH, STORAGE_T, COMPUTE_T>(Hi, Wi, K, Ho, Wo, (PSCALE != 0) ? PSCALE : pscale,
                                                              row_ker, row_lat, seg_off, seg, val_off, vals, inp, out);
    }

    // ---------------------------------------------------------------------------------
    // backward: scatter along the longitude with stride pscale = Wo / Wi
    // ---------------------------------------------------------------------------------

    // shared-memory accumulator body, see disco_bwd_d
    template <int BDIM_X, int ELXTH, typename STORAGE_T, typename COMPUTE_T>
    __device__ void disco_bwd_arcs_d(const int Hi, const int Wi, const int K, const int Ho, const int Wo,
                                     const int pscale, const int32_t *__restrict__ row_ker,
                                     const int32_t *__restrict__ row_lat, const int64_t *__restrict__ seg_off,
                                     const int32_t *__restrict__ seg, const int64_t *__restrict__ val_off,
                                     const COMPUTE_T *__restrict__ vals, const STORAGE_T *__restrict__ inp,
                                     COMPUTE_T *__restrict__ out)
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

        extern __shared__ __align__(sizeof(double)) unsigned char __sh_ptr[]; // COMPUTE_T __sh[pscale][2*BDIM_X*ELXTH]
        COMPUTE_T(*__sh)[BDIM_X * ELXTH * 2] = reinterpret_cast<COMPUTE_T(*)[BDIM_X * ELXTH * 2]>(__sh_ptr);

        COMPUTE_T __reg[ELXTH];
#pragma unroll
        for (int i = 0; i < ELXTH; i++) {
            __reg[i]
                = (i * BDIM_X + tid < Wi) ? static_cast<COMPUTE_T>(inp[i * BDIM_X + tid]) : static_cast<COMPUTE_T>(0);
        }

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

            // a new output ring: flush the accumulated one
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
                // consecutive nonzeros may hit the same shared entries
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

    // transposed-ownership body for PSCALE 1 and 2, see disco_bwd_swap_d
    template <int BDIM_X, int ELXTH, int PSCALE, typename STORAGE_T, typename COMPUTE_T>
    __device__ void disco_bwd_arcs_swap_d(const int Hi, const int Wi, const int K, const int Ho, const int Wo,
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

        extern __shared__ __align__(sizeof(double)) unsigned char __sh_ptr[];
        COMPUTE_T *sh_inp = reinterpret_cast<COMPUTE_T *>(__sh_ptr);

        constexpr int SH_LEN = 2 * BDIM_X * ELXTH;
        for (int j = tid; j < Wi; j += BDIM_X) {
            const COMPUTE_T x = static_cast<COMPUTE_T>(inp[j]);
            sh_inp[j] = x;
            sh_inp[Wi + j] = x;
        }
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
                // bank select outside the element loop, CTA-uniform: see disco_bwd_swap_d
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
    __global__ __launch_bounds__(BDIM_X) void disco_bwd_arcs_blk_k(
        const int Hi, const int Wi, const int K, const int Ho, const int Wo, const int pscale,
        const int32_t *__restrict__ row_ker, const int32_t *__restrict__ row_lat, const int64_t *__restrict__ seg_off,
        const int32_t *__restrict__ seg, const int64_t *__restrict__ val_off, const COMPUTE_T *__restrict__ vals,
        const STORAGE_T *__restrict__ inp, COMPUTE_T *__restrict__ out)
    {
        if constexpr (PSCALE != 0 && PSCALE <= 2) {
            disco_bwd_arcs_swap_d<BDIM_X, ELXTH, PSCALE, STORAGE_T, COMPUTE_T>(Hi, Wi, K, Ho, Wo, row_ker, row_lat,
                                                                               seg_off, seg, val_off, vals, inp, out);
        } else {
            disco_bwd_arcs_d<BDIM_X, ELXTH, STORAGE_T, COMPUTE_T>(Hi, Wi, K, Ho, Wo, (PSCALE != 0) ? PSCALE : pscale,
                                                                  row_ker, row_lat, seg_off, seg, val_off, vals, inp,
                                                                  out);
        }
    }

    // ---------------------------------------------------------------------------------
    // launch: the block-shape search and shared-memory sizing of the CSR hosts
    // ---------------------------------------------------------------------------------

    struct ArcPsi {
        int64_t nrows;
        const int32_t *row_ker;
        const int32_t *row_lat;
        const int64_t *seg_off;
        const int32_t *seg;
        const int64_t *val_off;
    };

    // grow ELXTH until NTH*ELXTH covers W, as the CSR launchers do
    template <int NTH, int ELXTH, typename STORAGE_T, typename COMPUTE_T>
    static void launch_fwd_arcs(int BC, int Hi, int Wi, int K, int Ho, int Wo, const ArcPsi &psi, const COMPUTE_T *vals,
                                const STORAGE_T *inp, STORAGE_T *out, cudaStream_t stream)
    {
        if constexpr (ELXTH <= ELXTH_MAX) {
            if (NTH * ELXTH >= Wo) {
                const dim3 grid(psi.nrows, BC);
                const int pscale = Wi / Wo;
                const size_t shmem = sizeof(STORAGE_T) * (Wi * 2 + pscale * (NTH * ELXTH - Wo));
#define DISCO_FWD_ARCS_LAUNCH(PS)                                                                                      \
    disco_fwd_arcs_blk_k<NTH, ELXTH, PS, STORAGE_T, COMPUTE_T><<<grid, NTH, shmem, stream>>>(                          \
        Hi, Wi, K, Ho, Wo, pscale, psi.row_ker, psi.row_lat, psi.seg_off, psi.seg, psi.val_off, vals, inp, out)
                switch (pscale) {
                case 1: DISCO_FWD_ARCS_LAUNCH(1); break;
                case 2: DISCO_FWD_ARCS_LAUNCH(2); break;
                case 3: DISCO_FWD_ARCS_LAUNCH(3); break;
                default: DISCO_FWD_ARCS_LAUNCH(0); break;
                }
#undef DISCO_FWD_ARCS_LAUNCH
            } else {
                launch_fwd_arcs<NTH, ELXTH + 1, STORAGE_T, COMPUTE_T>(BC, Hi, Wi, K, Ho, Wo, psi, vals, inp, out, stream);
            }
        }
    }

    template <int NTH, int ELXTH, typename STORAGE_T, typename COMPUTE_T>
    static void launch_bwd_arcs(int BC, int Hi, int Wi, int K, int Ho, int Wo, const ArcPsi &psi, const COMPUTE_T *vals,
                                const STORAGE_T *inp, COMPUTE_T *out, cudaStream_t stream)
    {
        if constexpr (ELXTH <= ELXTH_MAX) {
            if (NTH * ELXTH >= Wi) {
                const dim3 grid(psi.nrows, BC);
                const int pscale = Wo / Wi;
                const int sh_banks = (pscale <= 2) ? 1 : pscale;
                const size_t shmem = sizeof(COMPUTE_T) * (2 * (NTH * ELXTH) * sh_banks);

                int shmem_max = 0;
                int dev = 0;
                cudaGetDevice(&dev);
                cudaDeviceGetAttribute(&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, dev);
                TORCH_CHECK(shmem <= static_cast<size_t>(shmem_max), "disco backward (arcs): shared memory request (",
                            shmem, " B) for Wi=", Wi, " Wo=", Wo, " pscale=", pscale, " exceeds the per-block limit (",
                            shmem_max, " B)");
#define DISCO_BWD_ARCS_LAUNCH(PS)                                                                                      \
    disco_bwd_arcs_blk_k<NTH, ELXTH, PS, STORAGE_T, COMPUTE_T><<<grid, NTH, shmem, stream>>>(                          \
        Hi, Wi, K, Ho, Wo, pscale, psi.row_ker, psi.row_lat, psi.seg_off, psi.seg, psi.val_off, vals, inp, out)
                switch (pscale) {
                case 1: DISCO_BWD_ARCS_LAUNCH(1); break;
                case 2: DISCO_BWD_ARCS_LAUNCH(2); break;
                case 3: DISCO_BWD_ARCS_LAUNCH(3); break;
                default: DISCO_BWD_ARCS_LAUNCH(0); break;
                }
#undef DISCO_BWD_ARCS_LAUNCH
            } else {
                launch_bwd_arcs<NTH, ELXTH + 1, STORAGE_T, COMPUTE_T>(BC, Hi, Wi, K, Ho, Wo, psi, vals, inp, out, stream);
            }
        }
    }

    // The starting block shape for a row of W elements, as in the CSR hosts: 64 lanes up to
    // 64*ELXTH_MAX, then the smallest wider block starting from (ELXTH_MAX / 2) + 1 elements
    // per lane. Calls launch(integral_constant<NTH>, integral_constant<ELXTH>).
    template <typename LAUNCH> static void with_block_shape(int64_t W, const char *what, LAUNCH &&launch)
    {
        static_assert(0 == (ELXTH_MAX % 2));
        constexpr int E = (ELXTH_MAX / 2) + 1;
        if (W <= 64 * ELXTH_MAX) {
            launch(std::integral_constant<int, 64> {}, std::integral_constant<int, 1> {});
        } else if (W <= 128 * ELXTH_MAX) {
            launch(std::integral_constant<int, 128> {}, std::integral_constant<int, E> {});
        } else if (W <= 256 * ELXTH_MAX) {
            launch(std::integral_constant<int, 256> {}, std::integral_constant<int, E> {});
        } else if (W <= 512 * ELXTH_MAX) {
            launch(std::integral_constant<int, 512> {}, std::integral_constant<int, E> {});
        } else if (W <= 1024 * ELXTH_MAX) {
            launch(std::integral_constant<int, 1024> {}, std::integral_constant<int, E> {});
        } else {
            TORCH_CHECK(false, what, " (", W, ") exceeds the largest supported value (", 1024 * ELXTH_MAX, ")");
        }
    }

    static ArcPsi arc_psi(const torch::Tensor &row_ker, const torch::Tensor &row_lat, const torch::Tensor &seg_off,
                          const torch::Tensor &seg, const torch::Tensor &val_off)
    {
        return ArcPsi {row_ker.size(0),
                       row_ker.data_ptr<int32_t>(),
                       row_lat.data_ptr<int32_t>(),
                       seg_off.data_ptr<int64_t>(),
                       seg.data_ptr<int32_t>(),
                       val_off.data_ptr<int64_t>()};
    }

    torch::Tensor disco_cuda_fwd_arcs(torch::Tensor inp, torch::Tensor row_ker, torch::Tensor row_lat,
                                      torch::Tensor seg_off, torch::Tensor seg, torch::Tensor val_off,
                                      torch::Tensor vals, int64_t K, int64_t Ho, int64_t Wo)
    {
        TORCH_CHECK(inp.device().is_cuda(), "inp must be a CUDA tensor, got ", inp.device());
        check_forward_arcs_inputs(inp, row_ker, row_lat, seg_off, seg, val_off, vals, K, Ho, Wo);
        const at::cuda::OptionalCUDAGuard device_guard(inp.device());

        const int64_t BC = inp.size(0) * inp.size(1);
        const int64_t Hi = inp.size(2);
        const int64_t Wi = inp.size(3);

        auto out = torch::zeros({inp.size(0), inp.size(1), K, Ho, Wo}, inp.options());
        if (row_ker.size(0) == 0 || BC == 0) return out;

        const ArcPsi psi = arc_psi(row_ker, row_lat, seg_off, seg, val_off);
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, inp.scalar_type(), "disco_forward_arcs_cuda", ([&] {
                using storage_t = scalar_t;
                using compute_t = typename at::opmath_type<storage_t>;
                with_block_shape(Wo, "disco forward (arcs): nlon_out", [&](auto nth, auto elxth) {
                    launch_fwd_arcs<decltype(nth)::value, decltype(elxth)::value, storage_t, compute_t>(
                        BC, Hi, Wi, K, Ho, Wo, psi, vals.data_ptr<compute_t>(), inp.data_ptr<storage_t>(),
                        out.data_ptr<storage_t>(), stream);
                });
            }));

        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }

    torch::Tensor disco_cuda_bwd_arcs(torch::Tensor inp, torch::Tensor row_ker, torch::Tensor row_lat,
                                      torch::Tensor seg_off, torch::Tensor seg, torch::Tensor val_off,
                                      torch::Tensor vals, int64_t K, int64_t Ho, int64_t Wo)
    {
        TORCH_CHECK(inp.device().is_cuda(), "inp must be a CUDA tensor, got ", inp.device());
        check_backward_arcs_inputs(inp, row_ker, row_lat, seg_off, seg, val_off, vals, K, Ho, Wo);
        const at::cuda::OptionalCUDAGuard device_guard(inp.device());

        const int64_t BC = inp.size(0) * inp.size(1);
        const int64_t Hi = inp.size(3);
        const int64_t Wi = inp.size(4);

        // accumulated in the compute dtype and narrowed on return, as in disco_cuda_bwd
        auto out = torch::zeros({inp.size(0), inp.size(1), Ho, Wo}, inp.options().dtype(vals.dtype()));
        if (row_ker.size(0) > 0 && BC > 0) {
            const ArcPsi psi = arc_psi(row_ker, row_lat, seg_off, seg, val_off);
            auto stream = at::cuda::getCurrentCUDAStream().stream();

            // block shape from Wi, not Wo: see disco_cuda_bwd
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kHalf, at::kBFloat16, inp.scalar_type(), "disco_backward_arcs_cuda", ([&] {
                    using storage_t = scalar_t;
                    using compute_t = typename at::opmath_type<storage_t>;
                    with_block_shape(Wi, "disco backward (arcs): nlon_in", [&](auto nth, auto elxth) {
                        launch_bwd_arcs<decltype(nth)::value, decltype(elxth)::value, storage_t, compute_t>(
                            BC, Hi, Wi, K, Ho, Wo, psi, vals.data_ptr<compute_t>(), inp.data_ptr<storage_t>(),
                            out.data_ptr<compute_t>(), stream);
                    });
                }));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return out.to(inp.scalar_type());
    }

    TORCH_LIBRARY_IMPL(disco_kernels, CUDA, m)
    {
        m.impl("forward_arcs", &disco_cuda_fwd_arcs);
        m.impl("backward_arcs", &disco_cuda_bwd_arcs);
    }

} // namespace disco_kernels
