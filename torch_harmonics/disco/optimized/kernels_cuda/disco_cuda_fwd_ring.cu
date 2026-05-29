// coding=utf-8
//
// SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
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

// Ring-step DISCO forward kernel. Counterpart of disco_cuda_fwd.cu for the
// distributed ring-exchange algorithm (DistributedDiscreteContinuousConvS2Ring).
//
// Key differences from disco_cuda_fwd:
//   - The input tensor holds a single rank's azimuth chunk: x_chunk has
//     shape (B, C, Hi_local, Wi_local_src) where Wi_local_src is the src
//     rank's chunk width along longitude.
//   - psi.col_idx encodes  h_in_local * Wi_global + wi_shifted  (the
//     wi_shift is baked in by _build_local_psi on the Python side; see
//     distributed_convolution_ring.py). Wi_global is passed as a separate
//     parameter so the kernel can do modular arithmetic on the global
//     longitude grid for pshift resolution.
//   - lon_lo_src tells the kernel where the held x_chunk lives in the
//     global longitude. Each (entry, output column) pair resolves a global
//     w_in; only those falling in [lon_lo_src, lon_lo_src + Wi_local_src)
//     contribute this step. Entries that fall outside this src's chunk are
//     skipped and will be picked up at a later ring step.
//   - The output (y_acc) is ACCUMULATED into (`+=`), not overwritten — the
//     same y_acc buffer receives partial contributions from every ring
//     step's kernel call.

#include "../disco.h"
#include "disco_cuda.cuh"

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>

#include <cstdlib>

namespace disco_kernels
{

    // Wi_global threshold below which the doubled-shmem fast path is used.
    // Above it, shmem (~2 * Wi_global * sizeof(STORAGE_T) per block) starts
    // to limit occupancy and we fall back to the masked kernel.
    // Tunable via TORCH_HARMONICS_RING_FWD_FAST_MAX; default 5000 is tuned
    // for fp32 storage on H100/B200 (~40 KB shmem per block). Setting it to
    // 0 forces the masked path. Re-read every call so it can be flipped
    // within one process.
    static int _ring_fwd_fast_path_wi_max()
    {
        constexpr int default_max = 5000;
        const char *e = std::getenv("TORCH_HARMONICS_RING_FWD_FAST_MAX");
        if (e == nullptr || *e == '\0') return default_max;
        char *end = nullptr;
        long v = std::strtol(e, &end, 10);
        if (end == e || *end != '\0' || v < 0) return default_max;
        return static_cast<int>(v);
    }

    template <int BDIM_X, int ELXTH, typename STORAGE_T, typename COMPUTE_T>
    __device__ void disco_fwd_ring_d(const int Hi, const int Wi_local_src, const int Wi_global, const int K,
                                     const int Ho, const int Wo_local_self, const int pscale, const int lon_lo_src,
                                     const int64_t *__restrict__ roff, const int64_t *__restrict__ kers,
                                     const int64_t *__restrict__ rows, const int64_t *__restrict__ cols,
                                     const COMPUTE_T *__restrict__ vals, const STORAGE_T *__restrict__ inp,
                                     STORAGE_T *__restrict__ out)
    {
        const int tid = threadIdx.x;

        const int64_t bidx = blockIdx.x; // global psi row (sorted)
        const int64_t bidy = blockIdx.y; // batch * channel

        int64_t soff = roff[bidx];
        int64_t eoff = roff[bidx + 1];

        // skip empty rows
        if (soff == eoff) return;

        const int64_t ker = kers[soff];
        const int64_t row = rows[soff];

        inp += bidy * Hi * Wi_local_src;
        out += bidy * K * Ho * Wo_local_self + ker * Ho * Wo_local_self + row * Wo_local_self;

        COMPUTE_T __reg[ELXTH] = {0};

        extern __shared__ __align__(sizeof(double)) unsigned char __sh_ptr[];
        STORAGE_T *__sh = reinterpret_cast<STORAGE_T *>(__sh_ptr);

        // col_idx layout (after _build_local_psi): h_in_local * Wi_global + wi_shifted
        int col_prev = cols[soff];
        int h_prev = col_prev / Wi_global;          // h_in_local
        int w_prev = col_prev - h_prev * Wi_global; // wi_shifted

        // Stage the active input row (size Wi_local_src) into shmem. Unlike
        // disco_cuda_fwd we cannot use a doubled-shmem mod-elision trick here
        // because the pp-stride uses the GLOBAL Wi for the modular roll while
        // the chunk we hold is only Wi_local_src wide.
        for (int i = tid; i < Wi_local_src; i += BDIM_X) { __sh[i] = inp[h_prev * Wi_local_src + i]; }
        __syncthreads();

        for (int64_t nz = soff; nz < eoff; nz++) {

            const int col = cols[nz];
            const COMPUTE_T val = vals[nz];

            // row-change detection: same logic as disco_cuda_fwd, but on Wi_global.
            // col >= (col_prev - w_prev) + Wi_global  iff  h_in_local advanced.
            if (col >= col_prev - w_prev + Wi_global) {
                col_prev = col;
                h_prev = col / Wi_global;
                w_prev = col - h_prev * Wi_global;

                __syncthreads();
                for (int i = tid; i < Wi_local_src; i += BDIM_X) { __sh[i] = inp[h_prev * Wi_local_src + i]; }
                __syncthreads();
            }

            const int wi_shifted = col - h_prev * Wi_global; // = col % Wi_global

#pragma unroll
            for (int i = 0; i < ELXTH; i++) {

                const int pp = i * BDIM_X + tid;
                if (pp >= Wo_local_self) break;

                // Resolve the global input column for output column pp.
                // w_in_global = (wi_shifted + pscale * pp) mod Wi_global
                //
                // wi_shifted lies in [0, Wi_global) and pscale*pp lies in
                // [0, pscale * Wo_local_self) <= [0, Wi_global), so the sum is
                // in [0, 2*Wi_global) and the mod becomes a single subtract.
                int w_in_global = wi_shifted + pscale * pp;
                if (w_in_global >= Wi_global) w_in_global -= Wi_global;

                // Range check against this src's chunk. Out-of-chunk entries
                // are picked up at a different ring step. Cast to unsigned so
                // the (w_in_local < 0) and (w_in_local >= Wi_local_src) checks
                // collapse into one compare; the FMA stays branch-free by
                // reading __sh[0] for the masked-out lanes and selecting zero.
                const int w_in_local = w_in_global - lon_lo_src;
                const bool in_range = ((unsigned)w_in_local < (unsigned)Wi_local_src);
                const int safe_idx = in_range ? w_in_local : 0;
                const COMPUTE_T s = static_cast<COMPUTE_T>(__sh[safe_idx]);
                __reg[i] += in_range ? (val * s) : COMPUTE_T(0);
            }
        }

        // Accumulate into the output buffer. Across ring steps the same y_acc
        // receives contributions from every src; the Python driver clears it
        // once at the start of the autograd forward.
#pragma unroll
        for (int i = 0; i < ELXTH; i++) {
            const int pp = i * BDIM_X + tid;
            if (pp >= Wo_local_self) break;

            out[pp] = static_cast<STORAGE_T>(static_cast<COMPUTE_T>(out[pp]) + __reg[i]);
        }
    }

    // Fast doubled-shmem variant: ports disco_cuda_fwd's mod-elision trick to
    // the ring case. shmem is sized to (2 * Wi_global + pscale * slack) and
    // zero-filled; the local chunk is then placed at [lon_lo_src, lon_lo_src
    // + Wi_local_src) AND duplicated at [Wi_global + lon_lo_src, ...). All
    // out-of-chunk shmem positions stay zero, so out-of-chunk reads in the
    // inner loop contribute nothing — matching the masked variant's
    // semantics without the per-iter branches. The implicit wrap at
    // Wi_global is absorbed by the duplicate. Selected at launch time when
    // Wi_global <= _ring_fwd_fast_path_wi_max() (default 5000, env-tunable).
    template <int BDIM_X, int ELXTH, typename STORAGE_T, typename COMPUTE_T>
    __device__ void disco_fwd_ring_d_fast(const int Hi, const int Wi_local_src, const int Wi_global, const int K,
                                          const int Ho, const int Wo_local_self, const int pscale, const int lon_lo_src,
                                          const int64_t *__restrict__ roff, const int64_t *__restrict__ kers,
                                          const int64_t *__restrict__ rows, const int64_t *__restrict__ cols,
                                          const COMPUTE_T *__restrict__ vals, const STORAGE_T *__restrict__ inp,
                                          STORAGE_T *__restrict__ out)
    {
        const int tid = threadIdx.x;

        const int64_t bidx = blockIdx.x;
        const int64_t bidy = blockIdx.y;

        int64_t soff = roff[bidx];
        int64_t eoff = roff[bidx + 1];

        if (soff == eoff) return;

        const int64_t ker = kers[soff];
        const int64_t row = rows[soff];

        inp += bidy * Hi * Wi_local_src;
        out += bidy * K * Ho * Wo_local_self + ker * Ho * Wo_local_self + row * Wo_local_self;

        COMPUTE_T __reg[ELXTH] = {0};

        extern __shared__ __align__(sizeof(double)) unsigned char __sh_ptr[];
        STORAGE_T *__sh = reinterpret_cast<STORAGE_T *>(__sh_ptr);

        // Zero-fill the full shmem extent. Positions outside the chunk stay
        // zero for the whole kernel lifetime; only the two chunk windows are
        // refreshed on row changes.
        const int sh_total = Wi_global * 2 + pscale * (BDIM_X * ELXTH - Wo_local_self);
        for (int i = tid; i < sh_total; i += BDIM_X) { __sh[i] = STORAGE_T(0); }
        __syncthreads();

        int col_prev = cols[soff];
        int h_prev = col_prev / Wi_global;
        int w_prev = col_prev - h_prev * Wi_global;

        for (int i = tid; i < Wi_local_src; i += BDIM_X) {
            const STORAGE_T v = inp[h_prev * Wi_local_src + i];
            __sh[lon_lo_src + i] = v;
            __sh[Wi_global + lon_lo_src + i] = v;
        }
        __syncthreads();

        for (int64_t nz = soff; nz < eoff; nz++) {

            const int col = cols[nz];
            const COMPUTE_T val = vals[nz];

            if (col >= col_prev - w_prev + Wi_global) {
                col_prev = col;
                h_prev = col / Wi_global;
                w_prev = col - h_prev * Wi_global;

                __syncthreads();
                // Only refresh the chunk windows; the outside region stays
                // zero across row changes.
                for (int i = tid; i < Wi_local_src; i += BDIM_X) {
                    const STORAGE_T v = inp[h_prev * Wi_local_src + i];
                    __sh[lon_lo_src + i] = v;
                    __sh[Wi_global + lon_lo_src + i] = v;
                }
                __syncthreads();
            }

            const int wi_shifted = col - h_prev * Wi_global;

#pragma unroll
            for (int i = 0; i < ELXTH; i++) {
                const int pp = i * BDIM_X + tid;
                const int wpp = wi_shifted + pscale * pp;
                __reg[i] += val * static_cast<COMPUTE_T>(__sh[wpp]);
            }
        }

#pragma unroll
        for (int i = 0; i < ELXTH; i++) {
            const int pp = i * BDIM_X + tid;
            if (pp >= Wo_local_self) break;

            out[pp] = static_cast<STORAGE_T>(static_cast<COMPUTE_T>(out[pp]) + __reg[i]);
        }
    }

    template <int BDIM_X, int ELXTH, int PSCALE, bool FAST, typename STORAGE_T, typename COMPUTE_T>
    __global__ __launch_bounds__(BDIM_X) void disco_fwd_ring_blk_k(
        const int Hi, const int Wi_local_src, const int Wi_global, const int K, const int Ho, const int Wo_local_self,
        const int pscale, const int lon_lo_src, const int64_t *__restrict__ roff, const int64_t *__restrict__ kers,
        const int64_t *__restrict__ rows, const int64_t *__restrict__ cols, const COMPUTE_T *__restrict__ vals,
        const STORAGE_T *__restrict__ inp, STORAGE_T *__restrict__ out)
    {
        if constexpr (FAST) {
            if constexpr (PSCALE != 0) {
                disco_fwd_ring_d_fast<BDIM_X, ELXTH, STORAGE_T, COMPUTE_T>(Hi, Wi_local_src, Wi_global, K, Ho,
                                                                           Wo_local_self, PSCALE, lon_lo_src, roff,
                                                                           kers, rows, cols, vals, inp, out);
            } else {
                disco_fwd_ring_d_fast<BDIM_X, ELXTH, STORAGE_T, COMPUTE_T>(Hi, Wi_local_src, Wi_global, K, Ho,
                                                                           Wo_local_self, pscale, lon_lo_src, roff,
                                                                           kers, rows, cols, vals, inp, out);
            }
        } else {
            if constexpr (PSCALE != 0) {
                disco_fwd_ring_d<BDIM_X, ELXTH, STORAGE_T, COMPUTE_T>(Hi, Wi_local_src, Wi_global, K, Ho, Wo_local_self,
                                                                      PSCALE, lon_lo_src, roff, kers, rows, cols, vals,
                                                                      inp, out);
            } else {
                disco_fwd_ring_d<BDIM_X, ELXTH, STORAGE_T, COMPUTE_T>(Hi, Wi_local_src, Wi_global, K, Ho, Wo_local_self,
                                                                      pscale, lon_lo_src, roff, kers, rows, cols, vals,
                                                                      inp, out);
            }
        }
    }

    template <int NTH, int ELXTH, typename STORAGE_T, typename COMPUTE_T>
    static void launch_kernel_ring(int BC, int Hi, int Wi_local_src, int Wi_global, int K, int Ho, int Wo_local_self,
                                   int pscale, int lon_lo_src, int64_t nrows, int64_t *roff_d, int64_t *ker_d,
                                   int64_t *row_d, int64_t *col_d, COMPUTE_T *val_d, STORAGE_T *inp_d, STORAGE_T *out_d,
                                   cudaStream_t stream)
    {
        static_assert(sizeof(STORAGE_T) == 2 || sizeof(STORAGE_T) == 4 || sizeof(STORAGE_T) == 8);

        if constexpr (ELXTH <= ELXTH_MAX) {
            if (NTH * ELXTH >= Wo_local_self) {
                dim3 grid(nrows, BC);

                // Fast path: doubled-shmem mod-elision (cheap inner loop,
                // shmem ~ 2 * Wi_global). Fallback: masked kernel that only
                // stages the local chunk (Wi_local_src). Switch on Wi_global
                // to keep shmem bounded.
                const bool use_fast = (Wi_global <= _ring_fwd_fast_path_wi_max());
                const size_t shmem = use_fast ?
                    sizeof(*out_d) * (Wi_global * 2 + pscale * (NTH * ELXTH - Wo_local_self)) :
                    sizeof(*out_d) * Wi_local_src;

                if (use_fast) {
                    switch (pscale) {
                    case 1:
                        disco_fwd_ring_blk_k<NTH, ELXTH, 1, true, STORAGE_T, COMPUTE_T>
                            <<<grid, NTH, shmem, stream>>>(Hi, Wi_local_src, Wi_global, K, Ho, Wo_local_self, pscale,
                                                           lon_lo_src, roff_d, ker_d, row_d, col_d, val_d, inp_d, out_d);
                        break;
                    case 2:
                        disco_fwd_ring_blk_k<NTH, ELXTH, 2, true, STORAGE_T, COMPUTE_T>
                            <<<grid, NTH, shmem, stream>>>(Hi, Wi_local_src, Wi_global, K, Ho, Wo_local_self, pscale,
                                                           lon_lo_src, roff_d, ker_d, row_d, col_d, val_d, inp_d, out_d);
                        break;
                    case 3:
                        disco_fwd_ring_blk_k<NTH, ELXTH, 3, true, STORAGE_T, COMPUTE_T>
                            <<<grid, NTH, shmem, stream>>>(Hi, Wi_local_src, Wi_global, K, Ho, Wo_local_self, pscale,
                                                           lon_lo_src, roff_d, ker_d, row_d, col_d, val_d, inp_d, out_d);
                        break;
                    default:
                        disco_fwd_ring_blk_k<NTH, ELXTH, 0, true, STORAGE_T, COMPUTE_T>
                            <<<grid, NTH, shmem, stream>>>(Hi, Wi_local_src, Wi_global, K, Ho, Wo_local_self, pscale,
                                                           lon_lo_src, roff_d, ker_d, row_d, col_d, val_d, inp_d, out_d);
                        break;
                    }
                } else {
                    switch (pscale) {
                    case 1:
                        disco_fwd_ring_blk_k<NTH, ELXTH, 1, false, STORAGE_T, COMPUTE_T>
                            <<<grid, NTH, shmem, stream>>>(Hi, Wi_local_src, Wi_global, K, Ho, Wo_local_self, pscale,
                                                           lon_lo_src, roff_d, ker_d, row_d, col_d, val_d, inp_d, out_d);
                        break;
                    case 2:
                        disco_fwd_ring_blk_k<NTH, ELXTH, 2, false, STORAGE_T, COMPUTE_T>
                            <<<grid, NTH, shmem, stream>>>(Hi, Wi_local_src, Wi_global, K, Ho, Wo_local_self, pscale,
                                                           lon_lo_src, roff_d, ker_d, row_d, col_d, val_d, inp_d, out_d);
                        break;
                    case 3:
                        disco_fwd_ring_blk_k<NTH, ELXTH, 3, false, STORAGE_T, COMPUTE_T>
                            <<<grid, NTH, shmem, stream>>>(Hi, Wi_local_src, Wi_global, K, Ho, Wo_local_self, pscale,
                                                           lon_lo_src, roff_d, ker_d, row_d, col_d, val_d, inp_d, out_d);
                        break;
                    default:
                        disco_fwd_ring_blk_k<NTH, ELXTH, 0, false, STORAGE_T, COMPUTE_T>
                            <<<grid, NTH, shmem, stream>>>(Hi, Wi_local_src, Wi_global, K, Ho, Wo_local_self, pscale,
                                                           lon_lo_src, roff_d, ker_d, row_d, col_d, val_d, inp_d, out_d);
                        break;
                    }
                }
            } else {
                launch_kernel_ring<NTH, ELXTH + 1, STORAGE_T, COMPUTE_T>(
                    BC, Hi, Wi_local_src, Wi_global, K, Ho, Wo_local_self, pscale, lon_lo_src, nrows, roff_d, ker_d,
                    row_d, col_d, val_d, inp_d, out_d, stream);
            }
        }
    }

    void disco_cuda_fwd_ring_step(torch::Tensor inp, torch::Tensor out, torch::Tensor roff_idx, torch::Tensor ker_idx,
                                  torch::Tensor row_idx, torch::Tensor col_idx, torch::Tensor val, int64_t K,
                                  int64_t Ho, int64_t Wo_local_self, int64_t Wi_global, int64_t pscale,
                                  int64_t lon_lo_src, int64_t nlon_in_local_src)
    {
        CHECK_CUDA_INPUT_TENSOR(inp);
        CHECK_CUDA_INPUT_TENSOR(out);
        CHECK_CUDA_INPUT_TENSOR(roff_idx);
        CHECK_CUDA_INPUT_TENSOR(ker_idx);
        CHECK_CUDA_INPUT_TENSOR(row_idx);
        CHECK_CUDA_INPUT_TENSOR(col_idx);
        CHECK_CUDA_INPUT_TENSOR(val);

        const int64_t B = inp.size(0);
        const int64_t C = inp.size(1);
        const int64_t BC = B * C;
        const int64_t Hi = inp.size(2);
        const int64_t Wi_local_src_actual = inp.size(3);
        const int64_t nrows = roff_idx.size(0) - 1;

        TORCH_CHECK(Wi_local_src_actual == nlon_in_local_src, "inp.size(3) (", Wi_local_src_actual,
                    ") must match nlon_in_local_src (", nlon_in_local_src, ")");
        TORCH_CHECK(Wi_global % (Wo_local_self == 0 ? 1 : Wo_local_self) >= 0, "Wi_global / pscale arithmetic check");
        TORCH_CHECK(pscale >= 1, "pscale (", pscale, ") must be positive");
        TORCH_CHECK(lon_lo_src >= 0 && lon_lo_src + nlon_in_local_src <= Wi_global, "src chunk [", lon_lo_src, ", ",
                    lon_lo_src + nlon_in_local_src, ") must lie within [0, Wi_global=", Wi_global, ")");
        TORCH_CHECK(out.size(0) == B && out.size(1) == C && out.size(2) == K && out.size(3) == Ho
                        && out.size(4) == Wo_local_self,
                    "out shape must be (B, C, K, Ho, Wo_local_self)");

        auto stream = at::cuda::getCurrentCUDAStream().stream();
        static_assert(0 == (ELXTH_MAX % 2));

        if (Wo_local_self <= 64 * ELXTH_MAX) {
            AT_DISPATCH_FLOATING_TYPES(
                inp.scalar_type(), "disco_forward_ring_cuda", ([&] {
                    using storage_t = scalar_t;
                    using compute_t = typename at::opmath_type<storage_t>;
                    launch_kernel_ring<64, 1, storage_t, compute_t>(
                        BC, Hi, (int)Wi_local_src_actual, (int)Wi_global, (int)K, (int)Ho, (int)Wo_local_self,
                        (int)pscale, (int)lon_lo_src, nrows, roff_idx.data_ptr<int64_t>(), ker_idx.data_ptr<int64_t>(),
                        row_idx.data_ptr<int64_t>(), col_idx.data_ptr<int64_t>(), val.data_ptr<compute_t>(),
                        inp.data_ptr<storage_t>(), out.data_ptr<storage_t>(), stream);
                }));
        } else if (Wo_local_self <= 128 * ELXTH_MAX) {
            AT_DISPATCH_FLOATING_TYPES(
                inp.scalar_type(), "disco_forward_ring_cuda", ([&] {
                    using storage_t = scalar_t;
                    using compute_t = typename at::opmath_type<storage_t>;
                    launch_kernel_ring<128, (ELXTH_MAX / 2) + 1, storage_t, compute_t>(
                        BC, Hi, (int)Wi_local_src_actual, (int)Wi_global, (int)K, (int)Ho, (int)Wo_local_self,
                        (int)pscale, (int)lon_lo_src, nrows, roff_idx.data_ptr<int64_t>(), ker_idx.data_ptr<int64_t>(),
                        row_idx.data_ptr<int64_t>(), col_idx.data_ptr<int64_t>(), val.data_ptr<compute_t>(),
                        inp.data_ptr<storage_t>(), out.data_ptr<storage_t>(), stream);
                }));
        } else if (Wo_local_self <= 256 * ELXTH_MAX) {
            AT_DISPATCH_FLOATING_TYPES(
                inp.scalar_type(), "disco_forward_ring_cuda", ([&] {
                    using storage_t = scalar_t;
                    using compute_t = typename at::opmath_type<storage_t>;
                    launch_kernel_ring<256, (ELXTH_MAX / 2) + 1, storage_t, compute_t>(
                        BC, Hi, (int)Wi_local_src_actual, (int)Wi_global, (int)K, (int)Ho, (int)Wo_local_self,
                        (int)pscale, (int)lon_lo_src, nrows, roff_idx.data_ptr<int64_t>(), ker_idx.data_ptr<int64_t>(),
                        row_idx.data_ptr<int64_t>(), col_idx.data_ptr<int64_t>(), val.data_ptr<compute_t>(),
                        inp.data_ptr<storage_t>(), out.data_ptr<storage_t>(), stream);
                }));
        } else if (Wo_local_self <= 512 * ELXTH_MAX) {
            AT_DISPATCH_FLOATING_TYPES(
                inp.scalar_type(), "disco_forward_ring_cuda", ([&] {
                    using storage_t = scalar_t;
                    using compute_t = typename at::opmath_type<storage_t>;
                    launch_kernel_ring<512, (ELXTH_MAX / 2) + 1, storage_t, compute_t>(
                        BC, Hi, (int)Wi_local_src_actual, (int)Wi_global, (int)K, (int)Ho, (int)Wo_local_self,
                        (int)pscale, (int)lon_lo_src, nrows, roff_idx.data_ptr<int64_t>(), ker_idx.data_ptr<int64_t>(),
                        row_idx.data_ptr<int64_t>(), col_idx.data_ptr<int64_t>(), val.data_ptr<compute_t>(),
                        inp.data_ptr<storage_t>(), out.data_ptr<storage_t>(), stream);
                }));
        } else if (Wo_local_self <= 1024 * ELXTH_MAX) {
            AT_DISPATCH_FLOATING_TYPES(
                inp.scalar_type(), "disco_forward_ring_cuda", ([&] {
                    using storage_t = scalar_t;
                    using compute_t = typename at::opmath_type<storage_t>;
                    launch_kernel_ring<1024, (ELXTH_MAX / 2) + 1, storage_t, compute_t>(
                        BC, Hi, (int)Wi_local_src_actual, (int)Wi_global, (int)K, (int)Ho, (int)Wo_local_self,
                        (int)pscale, (int)lon_lo_src, nrows, roff_idx.data_ptr<int64_t>(), ker_idx.data_ptr<int64_t>(),
                        row_idx.data_ptr<int64_t>(), col_idx.data_ptr<int64_t>(), val.data_ptr<compute_t>(),
                        inp.data_ptr<storage_t>(), out.data_ptr<storage_t>(), stream);
                }));
        } else {
            fprintf(stderr, "%s:%d: error, unsupported Wo_local_self value (%ld), max supported is %d\n", __FILE__,
                    __LINE__, Wo_local_self, 1024 * ELXTH_MAX);
            exit(EXIT_FAILURE);
        }
    }

    TORCH_LIBRARY_IMPL(disco_kernels, CUDA, m) { m.impl("forward_ring_step", &disco_cuda_fwd_ring_step); }

} // namespace disco_kernels
