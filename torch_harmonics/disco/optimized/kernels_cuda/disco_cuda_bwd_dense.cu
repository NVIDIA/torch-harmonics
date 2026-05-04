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

// =====================================================================================
// Disco backward — dense-packed psi (CUDA, shmem-staged accumulator)
// =====================================================================================
//
// Same packed psi as the fwd kernel. For each (b, c, k, hi) row of dout, threads
// parallelize wi (BDIM_X * ELXTH >= Wi). The owned dout values live in registers.
//
// Shmem holds a per-CTA accumulator for the *output* row currently being written
// to. Layout: [pscale][2 * BDIM_X * ELXTH] — one shmem row per residue class
// (mod pscale) of the bwd output column wo. The "2 *" duplication is for
// wrap-around handling: a write at logical position (wo_base + pscale*wi) that
// falls past Wi can land in the second half and be summed back during flush.
//
// For each nz with (ho, wo_base, val):
//   - On ho transition: flush shmem to dinp[b, c, ho_prev, *] via gmem
//     atomicAdd, then zero shmem.
//   - Compute residue / quotient of wo_base, accumulate val * dout[wi] into
//     __sh[residue][quotient + pp] for each thread-owned wi position pp.
//
// The packed psi inherits col-sorted ordering from preprocess_psi (cols are
// non-decreasing within each (k, hi) row), so same-ho entries are contiguous
// and each distinct ho triggers exactly one flush. This keeps the gmem
// atomicAdds proportional to (#distinct ho) × Wi rather than cnt × Wi.
// =====================================================================================

#include "../disco.h"
#include "disco_cuda.cuh"

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>

namespace disco_kernels {

template <int BDIM_X, int ELXTH, typename T>
__device__ void disco_bwd_dense_d(int Hi, int Wi, int K, int Ho, int Wo, int NBR_PAD, int pscale,
                                  const int64_t *__restrict__ pack_idx,
                                  const T       *__restrict__ pack_val,
                                  const int64_t *__restrict__ pack_count,
                                  const T       *__restrict__ inp,
                                  T             *__restrict__ out)
{
    const int tid = threadIdx.x;
    const int bc  = blockIdx.y;          // b * C + c
    const int kh  = blockIdx.x;          // k * Hi + hi
    const int k   = kh / Hi;
    const int hi  = kh - k * Hi;

    const int64_t kh_off = (int64_t)k * Hi + hi;
    const int64_t *idx_kh = pack_idx + kh_off * NBR_PAD * 2;
    const T       *val_kh = pack_val + kh_off * NBR_PAD;
    const int      cnt    = (int)pack_count[kh_off];

    // dout[b, c, k, hi, :]
    const T *inp_kh = inp + (((int64_t)bc * K + k) * Hi + hi) * Wi;
    // dinp[b, c, :, :]
    T       *out_b  = out + (int64_t)bc * Ho * Wo;

    // Empty (k, hi) row (can occur in the distributed case after the lat split).
    // dinp is allocated zeroed, so leaving these CTAs idle is correct.
    if (cnt == 0) return;

    // Shmem accumulator: T __sh[pscale][2 * BDIM_X * ELXTH], aligned for double.
    extern __shared__ __align__(sizeof(double)) unsigned char __sh_raw[];
    T(*__sh)[BDIM_X * ELXTH * 2] = reinterpret_cast<T(*)[BDIM_X * ELXTH * 2]>(__sh_raw);

    // Cache the dout row in registers for this CTA's wi positions.
    T __reg[ELXTH];
    #pragma unroll
    for (int i = 0; i < ELXTH; i++) {
        __reg[i] = (i * BDIM_X + tid < Wi) ? inp_kh[i * BDIM_X + tid] : static_cast<T>(0);
    }

    // Reset shmem (all residue rows, full 2*BDIM_X*ELXTH width).
    for (int r = 0; r < pscale; r++) {
        #pragma unroll
        for (int j = 0; j < 2 * BDIM_X * ELXTH; j += BDIM_X) {
            __sh[r][j + tid] = static_cast<T>(0);
        }
    }
    __syncthreads();

    int ho_prev = (int)idx_kh[0];

    for (int nz = 0; nz < cnt; nz++) {
        const int ho      = (int)idx_kh[nz * 2 + 0];
        const int wo_base = (int)idx_kh[nz * 2 + 1];
        const T   v       = val_kh[nz];

        // On ho transition: flush shmem to dinp[b, c, ho_prev, *] and zero.
        if (ho != ho_prev) {
            __syncthreads();
            for (int r = 0; r < pscale; r++) {
                for (int j = tid; j < Wi; j += BDIM_X) {
                    const T s = __sh[r][j] + __sh[r][Wi + j];
                    atomicAdd(&out_b[(int64_t)ho_prev * Wo + j * pscale + r], s);
                    __sh[r][j]      = static_cast<T>(0);
                    __sh[r][Wi + j] = static_cast<T>(0);
                }
            }
            __syncthreads();
            ho_prev = ho;
        }

        // wo_full = wo_base + pscale * wi, decomposed as residue r = wo_full mod pscale,
        // quotient q = wo_full / pscale (mod Wi via the duplicated 2*Wi shmem layout).
        // r is invariant in pscale (all pp share the same residue as wo_base);
        // q depends on pp = i*BDIM_X + tid.
        const int w_mod_ps = wo_base % pscale;
        const int w_div_ps = wo_base / pscale;

        #pragma unroll
        for (int i = 0; i < ELXTH; i++) {
            const int pp = i * BDIM_X + tid;
            __sh[w_mod_ps][w_div_ps + pp] += v * __reg[i];
        }

        // Sync between nz iterations to avoid races on overlapping shmem cells
        // when the next nz's (residue, quotient + pp) collides with the current
        // one across different threads.
        __syncthreads();
    }
    __syncthreads();

    // Final flush for the last ho_prev row.
    for (int r = 0; r < pscale; r++) {
        for (int j = tid; j < Wi; j += BDIM_X) {
            const T s = __sh[r][j] + __sh[r][Wi + j];
            atomicAdd(&out_b[(int64_t)ho_prev * Wo + j * pscale + r], s);
        }
    }
}

template <int BDIM_X, int ELXTH, int PSCALE, typename T>
__global__ __launch_bounds__(BDIM_X)
void disco_bwd_dense_blk_k(int Hi, int Wi, int K, int Ho, int Wo, int NBR_PAD, int pscale,
                           const int64_t *__restrict__ pack_idx,
                           const T       *__restrict__ pack_val,
                           const int64_t *__restrict__ pack_count,
                           const T       *__restrict__ inp,
                           T             *__restrict__ out)
{
    if constexpr (PSCALE != 0) {
        disco_bwd_dense_d<BDIM_X, ELXTH, T>(Hi, Wi, K, Ho, Wo, NBR_PAD, PSCALE,
                                            pack_idx, pack_val, pack_count, inp, out);
    } else {
        disco_bwd_dense_d<BDIM_X, ELXTH, T>(Hi, Wi, K, Ho, Wo, NBR_PAD, pscale,
                                            pack_idx, pack_val, pack_count, inp, out);
    }
}

template <int NTH, int ELXTH, typename T>
static void launch_dense_bwd(int B, int C, int K, int Hi, int Wi, int Ho, int Wo,
                             int NBR_PAD,
                             const int64_t *pack_idx, const T *pack_val,
                             const int64_t *pack_count, const T *inp, T *out,
                             cudaStream_t stream)
{
    if constexpr (ELXTH <= ELXTH_MAX) {
        if (NTH * ELXTH >= Wi) {
            dim3 grid((unsigned)(K * Hi), (unsigned)(B * C));
            const int pscale = Wo / Wi;
            const size_t shmem = sizeof(T) * (size_t)(2 * (NTH * ELXTH) * pscale);

            switch (pscale) {
            case 1:
                disco_bwd_dense_blk_k<NTH, ELXTH, 1, T><<<grid, NTH, shmem, stream>>>(
                    Hi, Wi, K, Ho, Wo, NBR_PAD, pscale,
                    pack_idx, pack_val, pack_count, inp, out);
                break;
            case 2:
                disco_bwd_dense_blk_k<NTH, ELXTH, 2, T><<<grid, NTH, shmem, stream>>>(
                    Hi, Wi, K, Ho, Wo, NBR_PAD, pscale,
                    pack_idx, pack_val, pack_count, inp, out);
                break;
            case 3:
                disco_bwd_dense_blk_k<NTH, ELXTH, 3, T><<<grid, NTH, shmem, stream>>>(
                    Hi, Wi, K, Ho, Wo, NBR_PAD, pscale,
                    pack_idx, pack_val, pack_count, inp, out);
                break;
            default:
                disco_bwd_dense_blk_k<NTH, ELXTH, 0, T><<<grid, NTH, shmem, stream>>>(
                    Hi, Wi, K, Ho, Wo, NBR_PAD, pscale,
                    pack_idx, pack_val, pack_count, inp, out);
            }
        } else {
            launch_dense_bwd<NTH, ELXTH + 1, T>(B, C, K, Hi, Wi, Ho, Wo, NBR_PAD,
                                                pack_idx, pack_val, pack_count, inp, out, stream);
        }
    }
    return;
}

torch::Tensor disco_cuda_bwd_dense(torch::Tensor inp,
                                   torch::Tensor pack_idx,
                                   torch::Tensor pack_val,
                                   torch::Tensor pack_count,
                                   int64_t K, int64_t Ho, int64_t Wo)
{
    CHECK_CUDA_INPUT_TENSOR(inp);
    CHECK_CUDA_INPUT_TENSOR(pack_idx);
    CHECK_CUDA_INPUT_TENSOR(pack_val);
    CHECK_CUDA_INPUT_TENSOR(pack_count);

    const int64_t B  = inp.size(0);
    const int64_t C  = inp.size(1);
    // bwd input shape: [B, C, K, Hi=nlat_out_fwd, Wi=nlon_out_fwd]
    const int64_t Hi = inp.size(3);
    const int64_t Wi = inp.size(4);

    TORCH_CHECK(inp.size(2) == K, "inp.size(2) must equal K (got ", inp.size(2), " vs ", K, ")");
    TORCH_CHECK(Wo % Wi == 0,
                "Wo (", Wo, ") must be an integer multiple of Wi (", Wi, ")");

    TORCH_CHECK(pack_idx.dim()   == 4 && pack_idx.size(0)   == K && pack_idx.size(1)   == Hi && pack_idx.size(3) == 2,
                "pack_idx must have shape [K, Hi, NBR_PAD, 2] with Hi == inp.size(3)");
    TORCH_CHECK(pack_val.dim()   == 3 && pack_val.size(0)   == K && pack_val.size(1)   == Hi,
                "pack_val must have shape [K, Hi, NBR_PAD]");
    TORCH_CHECK(pack_count.dim() == 2 && pack_count.size(0) == K && pack_count.size(1) == Hi,
                "pack_count must have shape [K, Hi]");

    const int64_t NBR_PAD = pack_idx.size(2);

    // dinp is zero-initialized; empty (k, hi) rows from the distributed lat split
    // are then represented correctly without the kernel having to touch them.
    auto options = torch::TensorOptions().device(inp.device()).dtype(inp.dtype());
    auto out = torch::zeros({B, C, Ho, Wo}, options);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    static_assert(0 == (ELXTH_MAX % 2));

    // Dispatch table mirrors disco_cuda_bwd_csr: keyed on Wo, the inner check is
    // NTH*ELXTH >= Wi (the kernel's parallelism axis).
    if (Wo <= 64 * ELXTH_MAX) {
        AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_bwd_dense_cuda", ([&] {
            launch_dense_bwd<64, 1, scalar_t>(
                (int)B, (int)C, (int)K, (int)Hi, (int)Wi, (int)Ho, (int)Wo, (int)NBR_PAD,
                pack_idx.data_ptr<int64_t>(),
                pack_val.data_ptr<scalar_t>(),
                pack_count.data_ptr<int64_t>(),
                inp.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(), stream);
        }));
    } else if (Wo <= 128 * ELXTH_MAX) {
        AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_bwd_dense_cuda", ([&] {
            launch_dense_bwd<128, (ELXTH_MAX / 2) + 1, scalar_t>(
                (int)B, (int)C, (int)K, (int)Hi, (int)Wi, (int)Ho, (int)Wo, (int)NBR_PAD,
                pack_idx.data_ptr<int64_t>(),
                pack_val.data_ptr<scalar_t>(),
                pack_count.data_ptr<int64_t>(),
                inp.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(), stream);
        }));
    } else if (Wo <= 256 * ELXTH_MAX) {
        AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_bwd_dense_cuda", ([&] {
            launch_dense_bwd<256, (ELXTH_MAX / 2) + 1, scalar_t>(
                (int)B, (int)C, (int)K, (int)Hi, (int)Wi, (int)Ho, (int)Wo, (int)NBR_PAD,
                pack_idx.data_ptr<int64_t>(),
                pack_val.data_ptr<scalar_t>(),
                pack_count.data_ptr<int64_t>(),
                inp.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(), stream);
        }));
    } else if (Wo <= 512 * ELXTH_MAX) {
        AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_bwd_dense_cuda", ([&] {
            launch_dense_bwd<512, (ELXTH_MAX / 2) + 1, scalar_t>(
                (int)B, (int)C, (int)K, (int)Hi, (int)Wi, (int)Ho, (int)Wo, (int)NBR_PAD,
                pack_idx.data_ptr<int64_t>(),
                pack_val.data_ptr<scalar_t>(),
                pack_count.data_ptr<int64_t>(),
                inp.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(), stream);
        }));
    } else if (Wo <= 1024 * ELXTH_MAX) {
        AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_bwd_dense_cuda", ([&] {
            launch_dense_bwd<1024, (ELXTH_MAX / 2) + 1, scalar_t>(
                (int)B, (int)C, (int)K, (int)Hi, (int)Wi, (int)Ho, (int)Wo, (int)NBR_PAD,
                pack_idx.data_ptr<int64_t>(),
                pack_val.data_ptr<scalar_t>(),
                pack_count.data_ptr<int64_t>(),
                inp.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(), stream);
        }));
    } else {
        fprintf(stderr, "%s:%d: error, unsupported Wo value (%ld), max supported is %d\n",
                __FILE__, __LINE__, Wo, 1024 * ELXTH_MAX);
        exit(EXIT_FAILURE);
    }

    return out;
}

TORCH_LIBRARY_IMPL(disco_kernels, CUDA, m)
{
    m.impl("backward_dense", &disco_cuda_bwd_dense);
}

}
