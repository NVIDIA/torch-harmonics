// coding=utf-8
//
// SPDX-FileCopyrightText: Copyright (c) 2024 The torch-harmonics Authors. All rights reserved.
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

#include "../disco.h"
#include "disco_cuda.cuh"

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>

namespace disco_kernels
{

    template <int BDIM_X, int ELXTH, typename STORAGE_T, typename COMPUTE_T>
    __device__ void disco_bwd_d(const int Hi, const int Wi, const int K, const int Ho, const int Wo, const int pscale,
                                const int64_t *__restrict__ roff, const int64_t *__restrict__ kers,
                                const int64_t *__restrict__ rows, const int64_t *__restrict__ cols,
                                const COMPUTE_T *__restrict__ vals, const STORAGE_T *__restrict__ inp,
                                COMPUTE_T *__restrict__ out)
    {

        const int tid = threadIdx.x;

        const int64_t bidx = blockIdx.x; // gloabl row
        const int64_t bidy = blockIdx.y; // bc

        int64_t soff = roff[bidx];
        int64_t eoff = roff[bidx + 1];

        const int64_t ker = kers[soff];
        const int64_t row = rows[soff];

        inp += bidy * K * Hi * Wi + ker * Hi * Wi + row * Wi;
        out += bidy * Ho * Wo;

        // align to larger supported fp type
        extern __shared__ __align__(sizeof(double)) unsigned char __sh_ptr[]; // COMPUTE_T __sh[2*(BDIM_X*ELXTH)*pscale]

        COMPUTE_T(*__sh)[BDIM_X * ELXTH * 2] = reinterpret_cast<COMPUTE_T(*)[BDIM_X * ELXTH * 2]>(__sh_ptr);

        // copy current inp row in regs
        COMPUTE_T __reg[ELXTH];

#pragma unroll
        for (int i = 0; i < ELXTH; i++) {
            __reg[i]
                = (i * BDIM_X + tid < Wi) ? static_cast<COMPUTE_T>(inp[i * BDIM_X + tid]) : static_cast<COMPUTE_T>(0);
        }

        // reset shared row up to Wo+2, remaining
        // ppscale*(BDIM_X*ELXTH - Wo) locations
        // will be written to but never copied to
        // global mem
        for (int i = 0; i < pscale; i++) {
#pragma unroll
            for (int j = 0; j < 2 * BDIM_X * ELXTH; j += BDIM_X) { __sh[i][j + tid] = static_cast<COMPUTE_T>(0); }
        }
        __syncthreads();

        int col_prev = cols[soff];

        int h_prev = col_prev / Wo;
        int w_prev = col_prev % Wo;

        // loops along the colums of CTA's row
        for (int64_t nz = soff; nz < eoff; nz++) {

            const int col = cols[nz];
            const COMPUTE_T val = vals[nz];

            // if we are processing a nz with a col value
            // leading to a new row of inp then copy it
            // to shmem;
            // we read a col that points to a new output
            // row if (col / Wo) > (col_prev / Wo)
            if (col >= col_prev - w_prev + Wo) {
                __syncthreads();
                for (int i = 0; i < pscale; i++) {
                    for (int j = tid; j < Wi; j += BDIM_X) {

                        const COMPUTE_T v = __sh[i][j] + __sh[i][Wi + j];

                        atomicAdd(&out[h_prev * Wo + j * pscale + i], v);

                        __sh[i][j] = static_cast<COMPUTE_T>(0);
                        __sh[i][Wi + j] = static_cast<COMPUTE_T>(0);
                    }
                }
                __syncthreads();

                col_prev = col;
                h_prev = col / Wo;
                w_prev = col % Wo;
            }

            const int w = w_prev + (col - col_prev);
            const int w_mod_ps = w % pscale;
            const int w_div_ps = w / pscale;

#pragma unroll
            for (int i = 0; i < ELXTH; i++) {

                const int pp = i * BDIM_X + tid;
                __sh[w_mod_ps][w_div_ps + pp] += val * __reg[i];
            }

            // to avoid race conditions on __sh[]
            // among consecutive iterations along nz
            __syncthreads();
        }
        __syncthreads();

        // write last row
        for (int i = 0; i < pscale; i++) {

            for (int j = tid; j < Wi; j += BDIM_X) {

                const COMPUTE_T v = __sh[i][j] + __sh[i][Wi + j];
                atomicAdd(&out[h_prev * Wo + j * pscale + i], v);
            }
        }
        return;
    }

    template <int BDIM_X, int ELXTH, int PSCALE, typename STORAGE_T, typename COMPUTE_T>
    __global__
    __launch_bounds__(BDIM_X) void disco_bwd_blk_k(const int Hi, const int Wi, const int K, const int Ho, const int Wo,
                                                   const int pscale, const int64_t *__restrict__ roff,
                                                   const int64_t *__restrict__ kers, const int64_t *__restrict__ rows,
                                                   const int64_t *__restrict__ cols, const COMPUTE_T *__restrict__ vals,
                                                   const STORAGE_T *__restrict__ inp, COMPUTE_T *__restrict__ out)
    {

        if constexpr (PSCALE != 0) {
            disco_bwd_d<BDIM_X, ELXTH, STORAGE_T, COMPUTE_T>(Hi, Wi, K, Ho, Wo, PSCALE, roff, kers, rows, cols, vals,
                                                             inp, out);
        } else {
            disco_bwd_d<BDIM_X, ELXTH, STORAGE_T, COMPUTE_T>(Hi, Wi, K, Ho, Wo, pscale, roff, kers, rows, cols, vals,
                                                             inp, out);
        }

        return;
    }

    // ---------------------------------------------------------------------------
    // BC-tiled backward device function.
    // Each CTA processes BC_TILE channels simultaneously, amortising index loads
    // (roff/kers/rows/cols/vals) across BC_TILE accumulations. The index arrays
    // are loaded once per CTA; the input rows and shmem accumulators are
    // replicated BC_TILE times. Shared memory layout:
    //   __sh[BC_TILE * pscale][BDIM_X * ELXTH * 2]   (COMPUTE_T)
    // ---------------------------------------------------------------------------
    template <int BDIM_X, int ELXTH, int BC_TILE, typename STORAGE_T, typename COMPUTE_T>
    __device__ void disco_bwd_d_bctile(const int Hi, const int Wi, const int K, const int Ho, const int Wo,
                                       const int pscale, const int BC_total, const int64_t *__restrict__ roff,
                                       const int64_t *__restrict__ kers, const int64_t *__restrict__ rows,
                                       const int64_t *__restrict__ cols, const COMPUTE_T *__restrict__ vals,
                                       const STORAGE_T *__restrict__ inp, COMPUTE_T *__restrict__ out)
    {
        const int tid = threadIdx.x;
        const int64_t bidx = blockIdx.x; // CSR row
        const int bc_start = (int)blockIdx.y * BC_TILE;

        int64_t soff = roff[bidx];
        int64_t eoff = roff[bidx + 1];

        const int64_t ker = kers[soff];
        const int64_t row = rows[soff];

        extern __shared__ __align__(sizeof(double)) unsigned char __sh_ptr[];
        // __sh[BC_TILE * pscale][BDIM_X * ELXTH * 2]
        COMPUTE_T(*__sh)[BDIM_X * ELXTH * 2] = reinterpret_cast<COMPUTE_T(*)[BDIM_X * ELXTH * 2]>(__sh_ptr);

        // Load BC_TILE input rows into registers; zero-fill out-of-range bc slots.
        COMPUTE_T __reg[BC_TILE][ELXTH];
#pragma unroll
        for (int b = 0; b < BC_TILE; b++) {
            const int bc = bc_start + b;
            if (bc < BC_total) {
                const STORAGE_T *inp_bc = inp + (int64_t)bc * K * Hi * Wi + ker * Hi * Wi + row * Wi;
#pragma unroll
                for (int i = 0; i < ELXTH; i++) {
                    __reg[b][i] = (i * BDIM_X + tid < Wi) ? static_cast<COMPUTE_T>(inp_bc[i * BDIM_X + tid]) :
                                                            static_cast<COMPUTE_T>(0);
                }
            } else {
#pragma unroll
                for (int i = 0; i < ELXTH; i++) __reg[b][i] = static_cast<COMPUTE_T>(0);
            }
        }

        // Reset shared accumulators.
        for (int b = 0; b < BC_TILE; b++) {
            for (int i = 0; i < pscale; i++) {
#pragma unroll
                for (int j = 0; j < 2 * BDIM_X * ELXTH; j += BDIM_X)
                    __sh[b * pscale + i][j + tid] = static_cast<COMPUTE_T>(0);
            }
        }
        __syncthreads();

        int col_prev = cols[soff];
        int h_prev = col_prev / Wo;
        int w_prev = col_prev % Wo;

        for (int64_t nz = soff; nz < eoff; nz++) {

            const int col = cols[nz];
            const COMPUTE_T val_nz = vals[nz];

            if (col >= col_prev - w_prev + Wo) {
                __syncthreads();
                for (int b = 0; b < BC_TILE; b++) {
                    const int bc = bc_start + b;
                    if (bc >= BC_total) continue;
                    COMPUTE_T *out_bc = out + (int64_t)bc * Ho * Wo;
                    for (int i = 0; i < pscale; i++) {
                        for (int j = tid; j < Wi; j += BDIM_X) {
                            const COMPUTE_T v = __sh[b * pscale + i][j] + __sh[b * pscale + i][Wi + j];
                            atomicAdd(&out_bc[h_prev * Wo + j * pscale + i], v);
                            __sh[b * pscale + i][j] = static_cast<COMPUTE_T>(0);
                            __sh[b * pscale + i][Wi + j] = static_cast<COMPUTE_T>(0);
                        }
                    }
                }
                __syncthreads();

                col_prev = col;
                h_prev = col / Wo;
                w_prev = col % Wo;
            }

            const int w = w_prev + (col - col_prev);
            const int w_mod_ps = w % pscale;
            const int w_div_ps = w / pscale;

#pragma unroll
            for (int b = 0; b < BC_TILE; b++) {
#pragma unroll
                for (int i = 0; i < ELXTH; i++) {
                    const int pp = i * BDIM_X + tid;
                    __sh[b * pscale + w_mod_ps][w_div_ps + pp] += val_nz * __reg[b][i];
                }
            }
            __syncthreads();
        }
        __syncthreads();

        // Flush last row.
        for (int b = 0; b < BC_TILE; b++) {
            const int bc = bc_start + b;
            if (bc >= BC_total) continue;
            COMPUTE_T *out_bc = out + (int64_t)bc * Ho * Wo;
            for (int i = 0; i < pscale; i++) {
                for (int j = tid; j < Wi; j += BDIM_X) {
                    const COMPUTE_T v = __sh[b * pscale + i][j] + __sh[b * pscale + i][Wi + j];
                    atomicAdd(&out_bc[h_prev * Wo + j * pscale + i], v);
                }
            }
        }
    }

    template <int BDIM_X, int ELXTH, int PSCALE, int BC_TILE, typename STORAGE_T, typename COMPUTE_T>
    __global__ __launch_bounds__(BDIM_X) void disco_bwd_blk_k_bctile(
        const int Hi, const int Wi, const int K, const int Ho, const int Wo, const int pscale, const int BC_total,
        const int64_t *__restrict__ roff, const int64_t *__restrict__ kers, const int64_t *__restrict__ rows,
        const int64_t *__restrict__ cols, const COMPUTE_T *__restrict__ vals, const STORAGE_T *__restrict__ inp,
        COMPUTE_T *__restrict__ out)
    {
        if constexpr (PSCALE != 0) {
            disco_bwd_d_bctile<BDIM_X, ELXTH, BC_TILE, STORAGE_T, COMPUTE_T>(Hi, Wi, K, Ho, Wo, PSCALE, BC_total, roff,
                                                                             kers, rows, cols, vals, inp, out);
        } else {
            disco_bwd_d_bctile<BDIM_X, ELXTH, BC_TILE, STORAGE_T, COMPUTE_T>(Hi, Wi, K, Ho, Wo, pscale, BC_total, roff,
                                                                             kers, rows, cols, vals, inp, out);
        }
    }

    // BC_TILE=1 falls through to the original kernel; BC_TILE>1 uses the tiled variant.
    template <int NTH, int ELXTH, int BC_TILE, typename STORAGE_T, typename COMPUTE_T>
    static void launch_kernel(int BC, int Hi, int Wi, int K, int Ho, int Wo, int64_t nrows, int64_t *roff_d,
                              int64_t *ker_d, int64_t *row_d, int64_t *col_d, COMPUTE_T *val_d, STORAGE_T *inp_d,
                              COMPUTE_T *out_d, cudaStream_t stream)
    {

        static_assert(sizeof(STORAGE_T) == 2 || sizeof(STORAGE_T) == 4 || sizeof(STORAGE_T) == 8);

        if constexpr (ELXTH <= ELXTH_MAX) {
            if (NTH * ELXTH >= Wi) {
                const int pscale = Wo / Wi;
                const int bc_blocks = (BC + BC_TILE - 1) / BC_TILE;
                dim3 grid(nrows, bc_blocks);
                size_t shmem = sizeof(*out_d) * (2 * (NTH * ELXTH) * pscale * BC_TILE);

                if constexpr (BC_TILE == 1) {
                    switch (pscale) {
                    case 1:
                        disco_bwd_blk_k<NTH, ELXTH, 1, STORAGE_T, COMPUTE_T><<<grid, NTH, shmem, stream>>>(
                            Hi, Wi, K, Ho, Wo, pscale, roff_d, ker_d, row_d, col_d, val_d, inp_d, out_d);
                        break;
                    case 2:
                        disco_bwd_blk_k<NTH, ELXTH, 2, STORAGE_T, COMPUTE_T><<<grid, NTH, shmem, stream>>>(
                            Hi, Wi, K, Ho, Wo, pscale, roff_d, ker_d, row_d, col_d, val_d, inp_d, out_d);
                        break;
                    case 3:
                        disco_bwd_blk_k<NTH, ELXTH, 3, STORAGE_T, COMPUTE_T><<<grid, NTH, shmem, stream>>>(
                            Hi, Wi, K, Ho, Wo, pscale, roff_d, ker_d, row_d, col_d, val_d, inp_d, out_d);
                        break;
                    default:
                        disco_bwd_blk_k<NTH, ELXTH, 0, STORAGE_T, COMPUTE_T><<<grid, NTH, shmem, stream>>>(
                            Hi, Wi, K, Ho, Wo, pscale, roff_d, ker_d, row_d, col_d, val_d, inp_d, out_d);
                    }
                } else {
                    // Kernels with large BC_TILE may exceed the default 48 KB shmem limit.
                    auto set_shmem = [&](const void *fn) {
                        if (shmem > 49152)
                            cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
                    };
                    switch (pscale) {
                    case 1: {
                        auto *fn = &disco_bwd_blk_k_bctile<NTH, ELXTH, 1, BC_TILE, STORAGE_T, COMPUTE_T>;
                        set_shmem(reinterpret_cast<const void *>(fn));
                        fn<<<grid, NTH, shmem, stream>>>(Hi, Wi, K, Ho, Wo, pscale, BC, roff_d, ker_d, row_d, col_d,
                                                         val_d, inp_d, out_d);
                        break;
                    }
                    case 2: {
                        auto *fn = &disco_bwd_blk_k_bctile<NTH, ELXTH, 2, BC_TILE, STORAGE_T, COMPUTE_T>;
                        set_shmem(reinterpret_cast<const void *>(fn));
                        fn<<<grid, NTH, shmem, stream>>>(Hi, Wi, K, Ho, Wo, pscale, BC, roff_d, ker_d, row_d, col_d,
                                                         val_d, inp_d, out_d);
                        break;
                    }
                    case 3: {
                        auto *fn = &disco_bwd_blk_k_bctile<NTH, ELXTH, 3, BC_TILE, STORAGE_T, COMPUTE_T>;
                        set_shmem(reinterpret_cast<const void *>(fn));
                        fn<<<grid, NTH, shmem, stream>>>(Hi, Wi, K, Ho, Wo, pscale, BC, roff_d, ker_d, row_d, col_d,
                                                         val_d, inp_d, out_d);
                        break;
                    }
                    default: {
                        auto *fn = &disco_bwd_blk_k_bctile<NTH, ELXTH, 0, BC_TILE, STORAGE_T, COMPUTE_T>;
                        set_shmem(reinterpret_cast<const void *>(fn));
                        fn<<<grid, NTH, shmem, stream>>>(Hi, Wi, K, Ho, Wo, pscale, BC, roff_d, ker_d, row_d, col_d,
                                                         val_d, inp_d, out_d);
                        break;
                    }
                    }
                }
            } else {
                launch_kernel<NTH, ELXTH + 1, BC_TILE, STORAGE_T, COMPUTE_T>(BC, Hi, Wi, K, Ho, Wo, nrows, roff_d, ker_d,
                                                                             row_d, col_d, val_d, inp_d, out_d, stream);
            }
        }
        return;
    }

    torch::Tensor disco_cuda_bwd(torch::Tensor inp, torch::Tensor roff_idx, torch::Tensor ker_idx, torch::Tensor row_idx,
                                 torch::Tensor col_idx, torch::Tensor val, int64_t K, int64_t Ho, int64_t Wo)
    {

        // some sanity checks
        CHECK_CUDA_INPUT_TENSOR(inp);
        CHECK_CUDA_INPUT_TENSOR(roff_idx);
        CHECK_CUDA_INPUT_TENSOR(ker_idx);
        CHECK_CUDA_INPUT_TENSOR(row_idx);
        CHECK_CUDA_INPUT_TENSOR(col_idx);
        CHECK_CUDA_INPUT_TENSOR(val);

        // extract some shapes
        int64_t B = inp.size(0);
        int64_t C = inp.size(1);
        int64_t BC = B * C;
        int64_t Hi = inp.size(3);
        int64_t Wi = inp.size(4);
        int64_t nrows = roff_idx.size(0) - 1;

        // the kernel uses pscale = Wo / Wi; require an integer ratio so the p-shift is exact
        TORCH_CHECK(Wo % Wi == 0, "Wo (", Wo, ") must be an integer multiple of Wi (", Wi, ")");

        // allocate output. NOTE: unlike the forward kernel (which writes storage_t),
        // the backward kernel writes its result in COMPUTE type (out.data_ptr<compute_t>(),
        // i.e. fp32 for fp16/bf16 inp). vals is already compute type, so we key the
        // output dtype off vals; the Python op narrows the fp32 grad back to the input
        // dtype. Keying this off inp.dtype() would mismatch the kernel under fp16/bf16.
        int64_t out_dims[] = {B, C, Ho, Wo};
        auto options = torch::TensorOptions().device(inp.device()).dtype(val.dtype());
        torch::Tensor out = torch::zeros(out_dims, options);

        // get stream
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        // the wide launch configs (128/256/512/1024 lanes) split the per-thread element count
        // as (ELXTH_MAX / 2) + 1, so ELXTH_MAX must be even for the partition to be exact
        static_assert(0 == (ELXTH_MAX % 2));

        // Select BC_TILE: amortises index loads across BC_TILE channels per CTA.
        // BC_TILE=8 → shmem grows 8× but index load traffic shrinks 8×, improving
        // FMA utilisation from ~13% to near-peak on L1-bound workloads (C≥8).
        const int bc_tile = (BC >= 8) ? 8 : (BC >= 4) ? 4 : 1;

#define LAUNCH(NTH, ELXTH_START)                                                                                       \
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, inp.scalar_type(), "disco_backward_cuda", ([&] {         \
                                        using storage_t = scalar_t;                                                    \
                                        using compute_t = typename at::opmath_type<storage_t>;                         \
                                        auto *roff = roff_idx.data_ptr<int64_t>();                                     \
                                        auto *ker = ker_idx.data_ptr<int64_t>();                                       \
                                        auto *row = row_idx.data_ptr<int64_t>();                                       \
                                        auto *col = col_idx.data_ptr<int64_t>();                                       \
                                        auto *v = val.data_ptr<compute_t>();                                           \
                                        auto *i_ = inp.data_ptr<storage_t>();                                          \
                                        auto *o_ = out.data_ptr<compute_t>();                                          \
                                        if (bc_tile == 8)                                                              \
                                            launch_kernel<NTH, ELXTH_START, 8, storage_t, compute_t>(                  \
                                                BC, Hi, Wi, K, Ho, Wo, nrows, roff, ker, row, col, v, i_, o_, stream); \
                                        else if (bc_tile == 4)                                                         \
                                            launch_kernel<NTH, ELXTH_START, 4, storage_t, compute_t>(                  \
                                                BC, Hi, Wi, K, Ho, Wo, nrows, roff, ker, row, col, v, i_, o_, stream); \
                                        else                                                                           \
                                            launch_kernel<NTH, ELXTH_START, 1, storage_t, compute_t>(                  \
                                                BC, Hi, Wi, K, Ho, Wo, nrows, roff, ker, row, col, v, i_, o_, stream); \
                                    }))

        if (Wo <= 64 * ELXTH_MAX) {
            LAUNCH(64, 1);
        } else if (Wo <= 128 * ELXTH_MAX) {
            LAUNCH(128, (ELXTH_MAX / 2) + 1);
        } else if (Wo <= 256 * ELXTH_MAX) {
            LAUNCH(256, (ELXTH_MAX / 2) + 1);
        } else if (Wo <= 512 * ELXTH_MAX) {
            LAUNCH(512, (ELXTH_MAX / 2) + 1);
        } else if (Wo <= 1024 * ELXTH_MAX) {
            LAUNCH(1024, (ELXTH_MAX / 2) + 1);
        } else {
            fprintf(stderr, "%s:%d: error, unsupported Wo value (%ld), max supported is %d\n", __FILE__, __LINE__, Wo,
                    1024 * ELXTH_MAX);
            exit(EXIT_FAILURE);
        }
#undef LAUNCH

        // convert type if requested
        out = out.to(inp.dtype());

        return out;
    }

    TORCH_LIBRARY_IMPL(disco_kernels, CUDA, m) { m.impl("backward", &disco_cuda_bwd); }

} // namespace disco_kernels
