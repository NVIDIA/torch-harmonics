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

// Ring-step DISCO transpose (backward) kernel. Counterpart of
// disco_cuda_bwd.cu for the distributed ring algorithm.
//
// Key differences from disco_cuda_bwd:
//   - inp (grad_y_chunk) holds a single src rank's azimuth chunk along the
//     output longitude: shape (B, C, K, Hi, Wi_local_src) where
//     Wi_local_src = nlon_out_local_src.
//   - out (grad_x_acc) holds THIS rank's grad_x chunk along the input
//     longitude: shape (B, C, Ho, Wo_local_self) where
//     Wo_local_self = nlon_in_local_self. The kernel ATOMICALLY accumulates
//     into out; the Python autograd Function zero-inits it once before the
//     ring loop, then calls this kernel P_az times.
//   - psi.col_idx encodes  h_in_local * Wo_global + wi_shifted, where the
//     wi_shifted bakes in this rank's lon_lo_out_self (see _build_local_psi
//     in distributed_convolution_ring.py). Wo_global = nlon_in_global.
//   - Because the wi-shift was applied for THIS rank's wo offset but we
//     iterate through SRC's wo positions, we pass an offset
//     pscale_wo_offset = pscale * (lon_lo_src_out - lon_lo_out_self) and
//     fold it into the mod arithmetic.
//   - lon_lo_in_self tells us where this rank's grad_x window lives in the
//     global input longitude; entries that resolve to a wi_global outside
//     [lon_lo_in_self, lon_lo_in_self + Wo_local_self) are skipped this
//     step and will be picked up at a later ring step.
//
// Structure mirrors disco_cuda_bwd.cu closely: same pscale-lane shmem,
// same per-entry thread loop with __reg[i] holding the grad_y row, same
// row-change detection + flush mechanism, same __syncthreads cadence.
// The differences are local and minimal:
//   - row decomposition uses Wo_global (psi's column denominator after
//     the wi-shift) instead of Wo (= nlon_in in serial);
//   - the wi resolution inside the inner loop does an explicit mod on
//     Wo_global plus a range check against the local window — entries
//     outside the window skip the shmem write (which replaces the
//     serial bwd's doubled-shmem wraparound mechanism, since wraparound
//     in ring lands at Wo_global, not at Wo_local_self / pscale);
//   - per-lane shmem is single-sized (Wo_local_div_ps slots) rather than
//     doubled — the second half would always stay zero given the range
//     check, so it's elided.
//
// Strict-alignment assumption: lon_lo_in_self is a multiple of pscale,
// which holds for all standard split configs (compute_split_shapes with
// pscale | nlon_in_local boundaries). With that, wi_local mod pscale ==
// wi_shifted mod pscale, so w_mod_ps stays constant per entry exactly
// like the serial bwd.

#include "../disco.h"
#include "disco_cuda.cuh"

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>

namespace disco_kernels {

template <int BDIM_X, int ELXTH, typename STORAGE_T, typename COMPUTE_T>
__device__ void disco_bwd_ring_d(const int Hi,
                                 const int Wi_local_src,
                                 const int Wo_global,
                                 const int K,
                                 const int Ho,
                                 const int Wo_local_self,
                                 const int Wo_local_div_ps,   // = Wo_local_self / pscale
                                 const int pscale,
                                 const int pscale_wo_offset,
                                 const int lon_lo_in_self,
                                 const int64_t *__restrict__ roff,
                                 const int64_t *__restrict__ kers,
                                 const int64_t *__restrict__ rows,
                                 const int64_t *__restrict__ cols,
                                 const COMPUTE_T *__restrict__ vals,
                                 const STORAGE_T *__restrict__ inp,
                                 COMPUTE_T *__restrict__ out)
{
    const int tid = threadIdx.x;

    const int64_t bidx = blockIdx.x; // psi row (h_out)
    const int64_t bidy = blockIdx.y; // batch * channel

    int64_t soff = roff[bidx];
    int64_t eoff = roff[bidx + 1];

    if (soff == eoff) return;

    const int64_t ker = kers[soff];
    const int64_t row = rows[soff];

    inp += bidy * K * Hi * Wi_local_src + ker * Hi * Wi_local_src + row * Wi_local_src;
    out += bidy * Ho * Wo_local_self;

    // Shmem: pscale lanes × Wo_local_div_ps slots — same layout as serial
    // bwd modulo the per-lane size (single-half here instead of doubled,
    // see file header for rationale). Access pattern __sh[w_mod_ps][wi_local_div_ps]
    // is identical to the serial kernel's __sh[w_mod_ps][w_div_ps + pp].
    extern __shared__ __align__(sizeof(double)) unsigned char __sh_ptr[];
    COMPUTE_T *__sh_flat = reinterpret_cast<COMPUTE_T *>(__sh_ptr);
    auto __sh_at = [&](int lane, int slot) -> COMPUTE_T & {
        return __sh_flat[lane * Wo_local_div_ps + slot];
    };

    // Stage this row of grad_y_chunk into per-thread registers (verbatim
    // from disco_cuda_bwd: each thread holds ELXTH grad_y columns indexed
    // by pp = i*BDIM_X + tid).
    COMPUTE_T __reg[ELXTH];
#pragma unroll
    for (int i = 0; i < ELXTH; i++) {
        __reg[i] = (i * BDIM_X + tid < Wi_local_src) ?
                       static_cast<COMPUTE_T>(inp[i * BDIM_X + tid]) :
                       static_cast<COMPUTE_T>(0);
    }

    // Zero the shmem accumulator.
    for (int i = 0; i < pscale; i++) {
        for (int j = tid; j < Wo_local_div_ps; j += BDIM_X) {
            __sh_at(i, j) = static_cast<COMPUTE_T>(0);
        }
    }
    __syncthreads();

    int col_prev = cols[soff];
    int h_prev   = col_prev / Wo_global;
    int w_prev   = col_prev - h_prev * Wo_global; // wi_shifted for first entry

    // Loop along psi columns of this CTA's row — mirrors serial bwd.
    for (int64_t nz = soff; nz < eoff; nz++) {

        const int col       = cols[nz];
        const COMPUTE_T val = vals[nz];

        // Row-change detection: identical pattern to serial bwd (col jumped
        // past the current h_in_local row). Flush shmem → grad_x_acc.
        if (col >= col_prev - w_prev + Wo_global) {
            __syncthreads();
            for (int i = 0; i < pscale; i++) {
                for (int j = tid; j < Wo_local_div_ps; j += BDIM_X) {
                    const COMPUTE_T v = __sh_at(i, j);
                    atomicAdd(&out[h_prev * Wo_local_self + j * pscale + i], v);
                    __sh_at(i, j) = static_cast<COMPUTE_T>(0);
                }
            }
            __syncthreads();

            col_prev = col;
            h_prev   = col / Wo_global;
            w_prev   = col - h_prev * Wo_global;
        }

        // wi_shifted for this entry; constant w_mod_ps follows from the
        // strict-alignment assumption on lon_lo_in_self (see file header).
        const int wi_shifted = col - h_prev * Wo_global;
        const int w_mod_ps   = wi_shifted - (wi_shifted / pscale) * pscale; // = wi_shifted % pscale

#pragma unroll
        for (int i = 0; i < ELXTH; i++) {

            const int pp = i * BDIM_X + tid;
            if (pp >= Wi_local_src) break;

            // Resolve wi_global with the explicit mod (see fwd-ring header
            // for the bound argument: sum lies in (-Wo_global, 3·Wo_global),
            // so three conditional adjusts suffice).
            int t = wi_shifted + pscale * pp + pscale_wo_offset;
            if (t < 0)          t += Wo_global;
            if (t >= Wo_global) t -= Wo_global;
            if (t >= Wo_global) t -= Wo_global;

            const int wi_local = t - lon_lo_in_self;
            // Range check replaces the serial bwd's doubled-shmem
            // wraparound mechanism. Out-of-window entries are skipped here
            // and will be picked up by a later ring step on the rank that
            // owns the corresponding grad_x slice.
            if (wi_local >= 0 && wi_local < Wo_local_self) {
                const int wi_local_div_ps = wi_local / pscale;
                __sh_at(w_mod_ps, wi_local_div_ps) += val * __reg[i];
            }
        }

        // Sync so that this entry's shmem writes are visible before the
        // next entry potentially overwrites a slot — same role as the
        // sync at the end of the serial bwd's per-entry body.
        __syncthreads();
    }
    __syncthreads();

    // Final flush of the last h_in_local's accumulator — verbatim from
    // serial bwd modulo the per-lane size.
    for (int i = 0; i < pscale; i++) {
        for (int j = tid; j < Wo_local_div_ps; j += BDIM_X) {
            const COMPUTE_T v = __sh_at(i, j);
            atomicAdd(&out[h_prev * Wo_local_self + j * pscale + i], v);
        }
    }
}

template <int BDIM_X, int ELXTH, int PSCALE, typename STORAGE_T, typename COMPUTE_T>
__global__
    __launch_bounds__(BDIM_X) void disco_bwd_ring_blk_k(const int Hi,
                                                        const int Wi_local_src,
                                                        const int Wo_global,
                                                        const int K,
                                                        const int Ho,
                                                        const int Wo_local_self,
                                                        const int Wo_local_div_ps,
                                                        const int pscale,
                                                        const int pscale_wo_offset,
                                                        const int lon_lo_in_self,
                                                        const int64_t *__restrict__ roff,
                                                        const int64_t *__restrict__ kers,
                                                        const int64_t *__restrict__ rows,
                                                        const int64_t *__restrict__ cols,
                                                        const COMPUTE_T *__restrict__ vals,
                                                        const STORAGE_T *__restrict__ inp,
                                                        COMPUTE_T *__restrict__ out)
{
    if constexpr (PSCALE != 0) {
        disco_bwd_ring_d<BDIM_X, ELXTH, STORAGE_T, COMPUTE_T>(
            Hi, Wi_local_src, Wo_global, K, Ho, Wo_local_self, Wo_local_div_ps,
            PSCALE, pscale_wo_offset, lon_lo_in_self,
            roff, kers, rows, cols, vals, inp, out);
    } else {
        disco_bwd_ring_d<BDIM_X, ELXTH, STORAGE_T, COMPUTE_T>(
            Hi, Wi_local_src, Wo_global, K, Ho, Wo_local_self, Wo_local_div_ps,
            pscale, pscale_wo_offset, lon_lo_in_self,
            roff, kers, rows, cols, vals, inp, out);
    }
}

template <int NTH, int ELXTH, typename STORAGE_T, typename COMPUTE_T>
static void launch_kernel_ring_bwd(int BC,
                                   int Hi,
                                   int Wi_local_src,
                                   int Wo_global,
                                   int K,
                                   int Ho,
                                   int Wo_local_self,
                                   int pscale,
                                   int pscale_wo_offset,
                                   int lon_lo_in_self,
                                   int64_t nrows,
                                   int64_t *roff_d,
                                   int64_t *ker_d,
                                   int64_t *row_d,
                                   int64_t *col_d,
                                   COMPUTE_T *val_d,
                                   STORAGE_T *inp_d,
                                   COMPUTE_T *out_d,
                                   cudaStream_t stream)
{
    static_assert(sizeof(STORAGE_T) == 2 || sizeof(STORAGE_T) == 4 || sizeof(STORAGE_T) == 8);

    if constexpr (ELXTH <= ELXTH_MAX) {
        if (NTH * ELXTH >= Wi_local_src) {
            dim3 grid(nrows, BC);

            // pscale-lane accumulator: one lane per residue class (matches
            // disco_cuda_bwd's structure). Per-lane size is Wo_local_div_ps
            // — single-half because the range check in the inner loop
            // catches the writes that the serial bwd's doubled half absorbs.
            const int Wo_local_div_ps = Wo_local_self / pscale;
            size_t shmem = sizeof(*out_d) * static_cast<size_t>(pscale) * static_cast<size_t>(Wo_local_div_ps);

            switch (pscale) {
            case 1:
                disco_bwd_ring_blk_k<NTH, ELXTH, 1, STORAGE_T, COMPUTE_T>
                    <<<grid, NTH, shmem, stream>>>(Hi, Wi_local_src, Wo_global, K, Ho, Wo_local_self,
                                                   Wo_local_div_ps, pscale, pscale_wo_offset, lon_lo_in_self,
                                                   roff_d, ker_d, row_d, col_d, val_d, inp_d, out_d);
                break;
            case 2:
                disco_bwd_ring_blk_k<NTH, ELXTH, 2, STORAGE_T, COMPUTE_T>
                    <<<grid, NTH, shmem, stream>>>(Hi, Wi_local_src, Wo_global, K, Ho, Wo_local_self,
                                                   Wo_local_div_ps, pscale, pscale_wo_offset, lon_lo_in_self,
                                                   roff_d, ker_d, row_d, col_d, val_d, inp_d, out_d);
                break;
            case 3:
                disco_bwd_ring_blk_k<NTH, ELXTH, 3, STORAGE_T, COMPUTE_T>
                    <<<grid, NTH, shmem, stream>>>(Hi, Wi_local_src, Wo_global, K, Ho, Wo_local_self,
                                                   Wo_local_div_ps, pscale, pscale_wo_offset, lon_lo_in_self,
                                                   roff_d, ker_d, row_d, col_d, val_d, inp_d, out_d);
                break;
            default:
                disco_bwd_ring_blk_k<NTH, ELXTH, 0, STORAGE_T, COMPUTE_T>
                    <<<grid, NTH, shmem, stream>>>(Hi, Wi_local_src, Wo_global, K, Ho, Wo_local_self,
                                                   Wo_local_div_ps, pscale, pscale_wo_offset, lon_lo_in_self,
                                                   roff_d, ker_d, row_d, col_d, val_d, inp_d, out_d);
                break;
            }
        } else {
            launch_kernel_ring_bwd<NTH, ELXTH + 1, STORAGE_T, COMPUTE_T>(
                BC, Hi, Wi_local_src, Wo_global, K, Ho, Wo_local_self, pscale, pscale_wo_offset, lon_lo_in_self,
                nrows, roff_d, ker_d, row_d, col_d, val_d, inp_d, out_d, stream);
        }
    }
}

void disco_cuda_bwd_ring_step(torch::Tensor inp,
                              torch::Tensor out,
                              torch::Tensor roff_idx,
                              torch::Tensor ker_idx,
                              torch::Tensor row_idx,
                              torch::Tensor col_idx,
                              torch::Tensor val,
                              int64_t K,
                              int64_t Ho,
                              int64_t Wo_local_self,
                              int64_t Wo_global,
                              int64_t pscale,
                              int64_t pscale_wo_offset,
                              int64_t lon_lo_in_self,
                              int64_t nlon_out_local_src)
{
    CHECK_CUDA_INPUT_TENSOR(inp);
    CHECK_CUDA_INPUT_TENSOR(out);
    CHECK_CUDA_INPUT_TENSOR(roff_idx);
    CHECK_CUDA_INPUT_TENSOR(ker_idx);
    CHECK_CUDA_INPUT_TENSOR(row_idx);
    CHECK_CUDA_INPUT_TENSOR(col_idx);
    CHECK_CUDA_INPUT_TENSOR(val);

    const int64_t B  = inp.size(0);
    const int64_t C  = inp.size(1);
    const int64_t BC = B * C;
    // inp shape: (B, C, K, Hi, Wi_local_src)
    const int64_t Hi = inp.size(3);
    const int64_t Wi_local_src_actual = inp.size(4);
    const int64_t nrows = roff_idx.size(0) - 1;

    TORCH_CHECK(Wi_local_src_actual == nlon_out_local_src,
                "inp.size(4) (", Wi_local_src_actual,
                ") must match nlon_out_local_src (", nlon_out_local_src, ")");
    TORCH_CHECK(pscale >= 1, "pscale (", pscale, ") must be positive");
    TORCH_CHECK(out.size(0) == B && out.size(1) == C && out.size(2) == Ho &&
                    out.size(3) == Wo_local_self,
                "out shape must be (B, C, Ho, Wo_local_self)");
    // Strict-alignment assumption (mirrors what _build_local_psi guarantees):
    // the local input window's offset and width must each be a multiple of
    // pscale. With this, wi_local mod pscale == wi_shifted mod pscale is
    // constant across threads in one psi entry — same property the serial
    // bwd's shmem-accumulation trick relies on.
    TORCH_CHECK(Wo_local_self % pscale == 0,
                "Wo_local_self (", Wo_local_self, ") must be a multiple of pscale (", pscale, ")");
    TORCH_CHECK(lon_lo_in_self % pscale == 0,
                "lon_lo_in_self (", lon_lo_in_self, ") must be a multiple of pscale (", pscale, ")");
    TORCH_CHECK(pscale_wo_offset % pscale == 0,
                "pscale_wo_offset (", pscale_wo_offset, ") must be a multiple of pscale (", pscale, ")");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    static_assert(0 == (ELXTH_MAX % 2));

    if (Wi_local_src_actual <= 64 * ELXTH_MAX) {
        AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_backward_ring_cuda", ([&] {
            using storage_t = scalar_t;
            using compute_t = typename at::opmath_type<storage_t>;
            launch_kernel_ring_bwd<64, 1, storage_t, compute_t>(
                BC, (int)Hi, (int)Wi_local_src_actual, (int)Wo_global, (int)K, (int)Ho,
                (int)Wo_local_self, (int)pscale, (int)pscale_wo_offset, (int)lon_lo_in_self, nrows,
                roff_idx.data_ptr<int64_t>(), ker_idx.data_ptr<int64_t>(),
                row_idx.data_ptr<int64_t>(), col_idx.data_ptr<int64_t>(),
                val.data_ptr<compute_t>(), inp.data_ptr<storage_t>(),
                out.data_ptr<compute_t>(), stream);
        }));
    } else if (Wi_local_src_actual <= 128 * ELXTH_MAX) {
        AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_backward_ring_cuda", ([&] {
            using storage_t = scalar_t;
            using compute_t = typename at::opmath_type<storage_t>;
            launch_kernel_ring_bwd<128, (ELXTH_MAX / 2) + 1, storage_t, compute_t>(
                BC, (int)Hi, (int)Wi_local_src_actual, (int)Wo_global, (int)K, (int)Ho,
                (int)Wo_local_self, (int)pscale, (int)pscale_wo_offset, (int)lon_lo_in_self, nrows,
                roff_idx.data_ptr<int64_t>(), ker_idx.data_ptr<int64_t>(),
                row_idx.data_ptr<int64_t>(), col_idx.data_ptr<int64_t>(),
                val.data_ptr<compute_t>(), inp.data_ptr<storage_t>(),
                out.data_ptr<compute_t>(), stream);
        }));
    } else if (Wi_local_src_actual <= 256 * ELXTH_MAX) {
        AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_backward_ring_cuda", ([&] {
            using storage_t = scalar_t;
            using compute_t = typename at::opmath_type<storage_t>;
            launch_kernel_ring_bwd<256, (ELXTH_MAX / 2) + 1, storage_t, compute_t>(
                BC, (int)Hi, (int)Wi_local_src_actual, (int)Wo_global, (int)K, (int)Ho,
                (int)Wo_local_self, (int)pscale, (int)pscale_wo_offset, (int)lon_lo_in_self, nrows,
                roff_idx.data_ptr<int64_t>(), ker_idx.data_ptr<int64_t>(),
                row_idx.data_ptr<int64_t>(), col_idx.data_ptr<int64_t>(),
                val.data_ptr<compute_t>(), inp.data_ptr<storage_t>(),
                out.data_ptr<compute_t>(), stream);
        }));
    } else if (Wi_local_src_actual <= 512 * ELXTH_MAX) {
        AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_backward_ring_cuda", ([&] {
            using storage_t = scalar_t;
            using compute_t = typename at::opmath_type<storage_t>;
            launch_kernel_ring_bwd<512, (ELXTH_MAX / 2) + 1, storage_t, compute_t>(
                BC, (int)Hi, (int)Wi_local_src_actual, (int)Wo_global, (int)K, (int)Ho,
                (int)Wo_local_self, (int)pscale, (int)pscale_wo_offset, (int)lon_lo_in_self, nrows,
                roff_idx.data_ptr<int64_t>(), ker_idx.data_ptr<int64_t>(),
                row_idx.data_ptr<int64_t>(), col_idx.data_ptr<int64_t>(),
                val.data_ptr<compute_t>(), inp.data_ptr<storage_t>(),
                out.data_ptr<compute_t>(), stream);
        }));
    } else if (Wi_local_src_actual <= 1024 * ELXTH_MAX) {
        AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_backward_ring_cuda", ([&] {
            using storage_t = scalar_t;
            using compute_t = typename at::opmath_type<storage_t>;
            launch_kernel_ring_bwd<1024, (ELXTH_MAX / 2) + 1, storage_t, compute_t>(
                BC, (int)Hi, (int)Wi_local_src_actual, (int)Wo_global, (int)K, (int)Ho,
                (int)Wo_local_self, (int)pscale, (int)pscale_wo_offset, (int)lon_lo_in_self, nrows,
                roff_idx.data_ptr<int64_t>(), ker_idx.data_ptr<int64_t>(),
                row_idx.data_ptr<int64_t>(), col_idx.data_ptr<int64_t>(),
                val.data_ptr<compute_t>(), inp.data_ptr<storage_t>(),
                out.data_ptr<compute_t>(), stream);
        }));
    } else {
        fprintf(stderr, "%s:%d: error, unsupported Wi_local_src value (%ld), max supported is %d\n",
                __FILE__, __LINE__, Wi_local_src_actual, 1024 * ELXTH_MAX);
        exit(EXIT_FAILURE);
    }
}

TORCH_LIBRARY_IMPL(disco_kernels, CUDA, m)
{
    m.impl("backward_ring_step", &disco_cuda_bwd_ring_step);
}

}
