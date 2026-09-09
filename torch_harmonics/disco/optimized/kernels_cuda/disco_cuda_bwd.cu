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
#include <c10/cuda/CUDAException.h>
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

    // =================================================================================
    // Transposed-ownership backward ("the swap"), for pscale 1 and 2.
    //
    // disco_bwd_d above gives each thread a slice of the INPUT row in registers and
    // accumulates into a shared-memory output row. Every nonzero is therefore a
    // shared read-modify-write plus a __syncthreads to keep consecutive nonzeros
    // from racing.
    //
    // This swaps the two. Shared holds the input row, written once and read-only
    // thereafter; registers hold the accumulator. Same sum, transposed ownership:
    //
    //   disco_bwd_d : __sh[w_mod][w_div + q] += val * inp[q]     q owned by thread
    //   here        : acc[w_mod][t]          += val * inp[t - w_div]  t owned by thread
    //
    // The inner loop becomes LDS + FFMA instead of LDS + FADD + STS, and the only
    // barrier left is the one after the shared fill. Measured (bf16, BC=64):
    //
    //             prod_decoder (ps 1)      prod_encoder (ps 2)
    //   H100      121.6 -> 59.19 ms 2.05x   40.20 -> 21.97 ms 1.83x
    //   GB200      83.21 -> 39.94 ms 2.08x   27.38 -> 14.20 ms 1.93x
    //
    // Gated to PSCALE <= 2. The accumulator costs PSCALE*ELXTH registers, and at
    // pscale 3 (36 registers for Wi=720) occupancy drops far enough that the kernel
    // loses despite executing ~10% fewer instructions. pscale >= 3 is not a shape
    // the models use, so it keeps the original body.
    // =================================================================================
    template <int BDIM_X, int ELXTH, int PSCALE, typename STORAGE_T, typename COMPUTE_T>
    __device__ void disco_bwd_swap_d(const int Hi, const int Wi, const int K, const int Ho, const int Wo,
                                     const int64_t *__restrict__ roff, const int64_t *__restrict__ kers,
                                     const int64_t *__restrict__ rows, const int64_t *__restrict__ cols,
                                     const COMPUTE_T *__restrict__ vals, const STORAGE_T *__restrict__ inp,
                                     COMPUTE_T *__restrict__ out)
    {
        const int tid = threadIdx.x;
        const int64_t bidx = blockIdx.x;
        const int64_t bidy = blockIdx.y;

        int64_t soff = roff[bidx];
        int64_t eoff = roff[bidx + 1];

        const int64_t ker = kers[soff];
        const int64_t row = rows[soff];

        inp += bidy * K * Hi * Wi + ker * Hi * Wi + row * Wi;
        out += bidy * Ho * Wo;

        // Shared holds the input row, duplicated so the wraparound is an offset
        // rather than a modulo -- the same trick disco_cuda_fwd.cu uses.
        extern __shared__ __align__(sizeof(double)) unsigned char __sh_ptr[];
        COMPUTE_T *sh_inp = reinterpret_cast<COMPUTE_T *>(__sh_ptr);

        constexpr int SH_LEN = 2 * BDIM_X * ELXTH;
        for (int j = tid; j < Wi; j += BDIM_X) {
            const COMPUTE_T v = static_cast<COMPUTE_T>(inp[j]);
            sh_inp[j] = v;
            sh_inp[Wi + j] = v;
        }
        // Lanes with t >= Wi still issue their read, at Wi + t - w_div, which can run
        // past 2*Wi. BDIM_X*ELXTH >= Wi so SH_LEN covers it, but the tail would
        // otherwise be uninitialised. Zeroing once keeps the inner loop branch-free;
        // those lanes' accumulators are discarded at flush anyway.
        for (int j = 2 * Wi + tid; j < SH_LEN; j += BDIM_X) { sh_inp[j] = static_cast<COMPUTE_T>(0); }
        __syncthreads(); // the only barrier in the kernel

        COMPUTE_T acc[PSCALE][ELXTH];
#pragma unroll
        for (int m = 0; m < PSCALE; m++) {
#pragma unroll
            for (int i = 0; i < ELXTH; i++) acc[m][i] = static_cast<COMPUTE_T>(0);
        }

        int col_prev = cols[soff];
        int h_prev = col_prev / Wo;
        int w_prev = col_prev % Wo;

        for (int64_t nz = soff; nz < eoff; nz++) {

            const int col = cols[nz];
            const COMPUTE_T val = vals[nz];

            if (col >= col_prev - w_prev + Wo) {
                // Row change: flush straight from registers. No barrier needed --
                // acc is private and sh_inp is read-only.
#pragma unroll
                for (int m = 0; m < PSCALE; m++) {
#pragma unroll
                    for (int i = 0; i < ELXTH; i++) {
                        const int t = i * BDIM_X + tid;
                        if (t < Wi) { atomicAdd(&out[h_prev * Wo + t * PSCALE + m], acc[m][i]); }
                        acc[m][i] = static_cast<COMPUTE_T>(0);
                    }
                }

                col_prev = col;
                h_prev = col / Wo;
                w_prev = col % Wo;
            }

            const int w = w_prev + (col - col_prev);
            const int w_mod_ps = w % PSCALE;
            const int w_div_ps = w / PSCALE;

            // The bank select is the OUTER loop, with the whole element loop inside
            // it. w_mod_ps derives only from cols[nz] and w_prev -- never from tid --
            // so it is CTA-uniform and this compiles to a branch, not predication.
            //
            // The nesting matters. With the element loop outside and a
            // one-instruction body (acc[m][i] += x) inside, ptxas predicates rather
            // than branches -- correctly, for a body that small -- and the kernel
            // then issues PSCALE adds per element instead of one. That measured +28%
            // instructions at pscale 2 and turned a memory-bound kernel into an
            // issue-bound one (42.25 ms, against 24.45 ms for this form).
#pragma unroll
            for (int m = 0; m < PSCALE; m++) {
                if (m == w_mod_ps) {
#pragma unroll
                    for (int i = 0; i < ELXTH; i++) { acc[m][i] += val * sh_inp[Wi + (i * BDIM_X + tid) - w_div_ps]; }
                }
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
    __global__
    __launch_bounds__(BDIM_X) void disco_bwd_blk_k(const int Hi, const int Wi, const int K, const int Ho, const int Wo,
                                                   const int pscale, const int64_t *__restrict__ roff,
                                                   const int64_t *__restrict__ kers, const int64_t *__restrict__ rows,
                                                   const int64_t *__restrict__ cols, const COMPUTE_T *__restrict__ vals,
                                                   const STORAGE_T *__restrict__ inp, COMPUTE_T *__restrict__ out)
    {

        if constexpr (PSCALE != 0 && PSCALE <= 2) {
            disco_bwd_swap_d<BDIM_X, ELXTH, PSCALE, STORAGE_T, COMPUTE_T>(Hi, Wi, K, Ho, Wo, roff, kers, rows, cols,
                                                                          vals, inp, out);
        } else if constexpr (PSCALE != 0) {
            disco_bwd_d<BDIM_X, ELXTH, STORAGE_T, COMPUTE_T>(Hi, Wi, K, Ho, Wo, PSCALE, roff, kers, rows, cols, vals,
                                                             inp, out);
        } else {
            disco_bwd_d<BDIM_X, ELXTH, STORAGE_T, COMPUTE_T>(Hi, Wi, K, Ho, Wo, pscale, roff, kers, rows, cols, vals,
                                                             inp, out);
        }

        return;
    }

    template <int NTH, int ELXTH, typename STORAGE_T, typename COMPUTE_T>
    static void launch_kernel(int BC, int Hi, int Wi, int K, int Ho, int Wo, int64_t nrows, int64_t *roff_d,
                              int64_t *ker_d, int64_t *row_d, int64_t *col_d, COMPUTE_T *val_d, STORAGE_T *inp_d,
                              COMPUTE_T *out_d, cudaStream_t stream)
    {

        static_assert(sizeof(STORAGE_T) == 2 || sizeof(STORAGE_T) == 4 || sizeof(STORAGE_T) == 8);

        if constexpr (ELXTH <= ELXTH_MAX) {
            if (NTH * ELXTH >= Wi) {
                dim3 grid(nrows, BC);

                const int pscale = Wo / Wi;
                // The swap (pscale <= 2) stores the input row, so it needs 2*NTH*ELXTH
                // regardless of pscale; the original body stores the pscale-way output
                // accumulator and needs the full 2*NTH*ELXTH*pscale.
                const int sh_banks = (pscale <= 2) ? 1 : pscale;
                size_t shmem = sizeof(*out_d) * (2 * (NTH * ELXTH) * sh_banks);

                // A bare over-limit launch surfaces only as cudaErrorInvalidValue
                // from the next API call, with nothing pointing at shared memory.
                int shmem_max = 0;
                int dev = 0;
                cudaGetDevice(&dev);
                cudaDeviceGetAttribute(&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, dev);
                if (shmem > static_cast<size_t>(shmem_max)) {
                    fprintf(stderr,
                            "%s:%d: error, shared memory request (%zu B) for Wi=%d Wo=%d pscale=%d "
                            "NTH=%d ELXTH=%d exceeds the per-block limit (%d B)\n",
                            __FILE__, __LINE__, shmem, Wi, Wo, pscale, NTH, ELXTH, shmem_max);
                    exit(EXIT_FAILURE);
                }

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
                launch_kernel<NTH, ELXTH + 1, STORAGE_T, COMPUTE_T>(BC, Hi, Wi, K, Ho, Wo, nrows, roff_d, ker_d, row_d,
                                                                    col_d, val_d, inp_d, out_d, stream);
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

        // NOTE: the block shape is chosen from Wi, not Wo. The only geometric
        // requirement the kernel has is NTH*ELXTH >= Wi: __reg[ELXTH] holds the
        // input row, the flush loops run to Wi, and Wo enters only through the
        // global index h_prev*Wo + j*pscale + i, which is block-shape agnostic.
        //
        // Keying this off Wo (inherited from the forward, where the *output* row
        // does live in shared) overshoots whenever Wo > 64*ELXTH_MAX >= Wi: the
        // NTH=128 branch starts ELXTH at (ELXTH_MAX/2)+1 = 17, so a shape with
        // Wi=720 got NTH*ELXTH = 2176 -- 3x more than it needs. Multiplied by
        // pscale that overran the 48 KB static shared limit and the launch failed
        // with cudaErrorInvalidValue for Wo > 2048 && pscale >= 3 (e.g.
        // 1080x2160 -> 360x720). Sizing from Wi both fixes that and cuts shared
        // usage for every Wo > 2048 shape; for Wo <= 64*ELXTH_MAX it is a no-op,
        // since Wi <= Wo puts both on the NTH=64 branch with identical growth.

        if (Wi <= 64 * ELXTH_MAX) {
            AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, inp.scalar_type(), "disco_backward_cuda", ([&] {
                                                using storage_t = scalar_t;
                                                using compute_t = typename at::opmath_type<storage_t>;
                                                launch_kernel<64, 1, storage_t, compute_t>(
                                                    BC, Hi, Wi, K, Ho, Wo, nrows, roff_idx.data_ptr<int64_t>(),
                                                    ker_idx.data_ptr<int64_t>(), row_idx.data_ptr<int64_t>(),
                                                    col_idx.data_ptr<int64_t>(), val.data_ptr<compute_t>(),
                                                    inp.data_ptr<storage_t>(), out.data_ptr<compute_t>(), stream);
                                            }));
        } else if (Wi <= 128 * ELXTH_MAX) {
            AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, inp.scalar_type(), "disco_backward_cuda", ([&] {
                                                using storage_t = scalar_t;
                                                using compute_t = typename at::opmath_type<storage_t>;
                                                launch_kernel<128, (ELXTH_MAX / 2) + 1, storage_t, compute_t>(
                                                    BC, Hi, Wi, K, Ho, Wo, nrows, roff_idx.data_ptr<int64_t>(),
                                                    ker_idx.data_ptr<int64_t>(), row_idx.data_ptr<int64_t>(),
                                                    col_idx.data_ptr<int64_t>(), val.data_ptr<compute_t>(),
                                                    inp.data_ptr<storage_t>(), out.data_ptr<compute_t>(), stream);
                                            }));
        } else if (Wi <= 256 * ELXTH_MAX) {
            AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, inp.scalar_type(), "disco_backward_cuda", ([&] {
                                                using storage_t = scalar_t;
                                                using compute_t = typename at::opmath_type<storage_t>;
                                                launch_kernel<256, (ELXTH_MAX / 2) + 1, storage_t, compute_t>(
                                                    BC, Hi, Wi, K, Ho, Wo, nrows, roff_idx.data_ptr<int64_t>(),
                                                    ker_idx.data_ptr<int64_t>(), row_idx.data_ptr<int64_t>(),
                                                    col_idx.data_ptr<int64_t>(), val.data_ptr<compute_t>(),
                                                    inp.data_ptr<storage_t>(), out.data_ptr<compute_t>(), stream);
                                            }));
        } else if (Wi <= 512 * ELXTH_MAX) {
            AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, inp.scalar_type(), "disco_backward_cuda", ([&] {
                                                using storage_t = scalar_t;
                                                using compute_t = typename at::opmath_type<storage_t>;
                                                launch_kernel<512, (ELXTH_MAX / 2) + 1, storage_t, compute_t>(
                                                    BC, Hi, Wi, K, Ho, Wo, nrows, roff_idx.data_ptr<int64_t>(),
                                                    ker_idx.data_ptr<int64_t>(), row_idx.data_ptr<int64_t>(),
                                                    col_idx.data_ptr<int64_t>(), val.data_ptr<compute_t>(),
                                                    inp.data_ptr<storage_t>(), out.data_ptr<compute_t>(), stream);
                                            }));
        } else if (Wi <= 1024 * ELXTH_MAX) {
            AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, inp.scalar_type(), "disco_backward_cuda", ([&] {
                                                using storage_t = scalar_t;
                                                using compute_t = typename at::opmath_type<storage_t>;
                                                launch_kernel<1024, (ELXTH_MAX / 2) + 1, storage_t, compute_t>(
                                                    BC, Hi, Wi, K, Ho, Wo, nrows, roff_idx.data_ptr<int64_t>(),
                                                    ker_idx.data_ptr<int64_t>(), row_idx.data_ptr<int64_t>(),
                                                    col_idx.data_ptr<int64_t>(), val.data_ptr<compute_t>(),
                                                    inp.data_ptr<storage_t>(), out.data_ptr<compute_t>(), stream);
                                            }));
        } else {
            fprintf(stderr, "%s:%d: error, unsupported Wi value (%ld), max supported is %d\n", __FILE__, __LINE__, Wi,
                    1024 * ELXTH_MAX);
            exit(EXIT_FAILURE);
        }

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // convert type if requested
        out = out.to(inp.dtype());

        return out;
    }

    TORCH_LIBRARY_IMPL(disco_kernels, CUDA, m) { m.impl("backward", &disco_cuda_bwd); }

} // namespace disco_kernels
