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
// Disco forward — dense-packed psi (CUDA, shmem-staged)
// =====================================================================================
//
// Consumes the (K, Ho, NBR_PAD)-packed psi produced by pack_psi_dense and computes
//
//   out[b, c, k, ho, wo] = sum_{nz=0..count[k,ho]} val[k,ho,nz]
//                                     * inp[b, c, hi[nz], (wi_base[nz] + pscale*wo) mod Wi]
//
// with pscale = Wi / Wo. Padded slots (nz >= count[k,ho]) are skipped via the cnt
// bound and also have val == 0 by construction.
//
// Parallelization: one CTA per (BC, k*Ho); each thread owns ELXTH wo positions held
// in registers, with BDIM_X * ELXTH >= Wo. The input lat row is staged in shmem
// (duplicated 2x for wrap-free indexing, mirroring the CSR fwd kernel) so each nz
// reads from shmem rather than gmem. The shmem is reloaded only when the nz's hi
// changes — same-hi entries are contiguous in the packed psi (it inherits the
// col-sorted ordering from preprocess_psi).
// =====================================================================================

#include "../disco.h"
#include "disco_cuda.cuh"

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>

namespace disco_kernels {

template <int BDIM_X, int ELXTH, typename T>
__device__ void disco_fwd_dense_d(int Hi, int Wi, int K, int Ho, int Wo, int NBR_PAD, int pscale,
                                  const int64_t *__restrict__ pack_idx,
                                  const T       *__restrict__ pack_val,
                                  const int64_t *__restrict__ pack_count,
                                  const T       *__restrict__ inp,
                                  T             *__restrict__ out)
{
    const int tid = threadIdx.x;
    const int bc  = blockIdx.y;          // b * C + c
    const int kh  = blockIdx.x;          // k * Ho + ho
    const int k   = kh / Ho;
    const int ho  = kh - k * Ho;

    const int64_t kh_off = (int64_t)k * Ho + ho;
    const int64_t *idx_kh = pack_idx + kh_off * NBR_PAD * 2;
    const T       *val_kh = pack_val + kh_off * NBR_PAD;
    const int      cnt    = (int)pack_count[kh_off];

    // out[b, c, k, ho, :]
    T *out_kh = out + ((int64_t)bc * K + k) * Ho * Wo + (int64_t)ho * Wo;

    // Empty (k, ho) row (can occur in the distributed case after the lat split).
    // The output is allocated with torch::zeros, so leaving these cells untouched
    // is correct. Early return saves the shmem prelude.
    if (cnt == 0) return;

    const T *inp_bc = inp + (int64_t)bc * Hi * Wi;

    // Shmem holds one input lat row, duplicated 2x:
    //   inp_sh[0..Wi-1]      = inp_bc[hi, 0..Wi-1]
    //   inp_sh[Wi..2*Wi-1]   = inp_bc[hi, 0..Wi-1]
    // Then wi_full = wi_base + pscale*pp ∈ [0, 2*Wi) is a direct shmem index without
    // any modulo (same trick as the CSR fwd kernel).
    extern __shared__ __align__(sizeof(double)) unsigned char inp_sh_raw[];
    T *inp_sh = reinterpret_cast<T *>(inp_sh_raw);

    // Per-thread accumulators (registers); pp = i * BDIM_X + tid covers wo positions.
    T acc[ELXTH];
    #pragma unroll
    for (int i = 0; i < ELXTH; i++) { acc[i] = static_cast<T>(0); }

    // Initial load: row of the first nz.
    int hi_prev = (int)idx_kh[0];
    for (int i = tid; i < Wi; i += BDIM_X) {
        const T v = inp_bc[(int64_t)hi_prev * Wi + i];
        inp_sh[i]      = v;
        inp_sh[Wi + i] = v;
    }
    __syncthreads();

    for (int nz = 0; nz < cnt; nz++) {
        const int hi      = (int)idx_kh[nz * 2 + 0];
        const int wi_base = (int)idx_kh[nz * 2 + 1];
        const T   v       = val_kh[nz];

        // Reload shmem on hi transition. Packed psi inherits col-sorted ordering
        // from preprocess_psi (cols are non-decreasing within a (k, ho) row), so
        // same-hi entries are contiguous and a transition fires at most once per
        // distinct hi.
        if (hi != hi_prev) {
            __syncthreads();
            hi_prev = hi;
            for (int i = tid; i < Wi; i += BDIM_X) {
                const T vv = inp_bc[(int64_t)hi * Wi + i];
                inp_sh[i]      = vv;
                inp_sh[Wi + i] = vv;
            }
            __syncthreads();
        }

        #pragma unroll
        for (int i = 0; i < ELXTH; i++) {
            const int pp = i * BDIM_X + tid;
            // Bound: wi_base < Wi and pscale*pp = (Wi/Wo)*pp.
            // For pp < Wo we have pscale*pp < Wi, so wi_full < 2*Wi → safe shmem index.
            // For pp >= Wo the index is out of range, but we'll discard these accs
            // at write-time. We still need to keep the read in-bounds: NTH*ELXTH may
            // exceed Wo so pp can be > Wo, hence pscale*pp can be >= Wi. The
            // duplicated 2*Wi window covers up to pp = 2*Wo - 1 = 2*Wi/pscale - 1,
            // i.e. pscale*pp < 2*Wi. NTH*ELXTH is constrained by the launch dispatch
            // so this holds.
            const int wi_full = wi_base + pscale * pp;
            acc[i] += v * inp_sh[wi_full];
        }
    }

    // Write results. Drop the wo positions outside [0, Wo).
    #pragma unroll
    for (int i = 0; i < ELXTH; i++) {
        const int pp = i * BDIM_X + tid;
        if (pp < Wo) { out_kh[pp] = acc[i]; }
    }
}

template <int BDIM_X, int ELXTH, typename T>
__global__ __launch_bounds__(BDIM_X)
void disco_fwd_dense_blk_k(int Hi, int Wi, int K, int Ho, int Wo, int NBR_PAD, int pscale,
                           const int64_t *__restrict__ pack_idx,
                           const T       *__restrict__ pack_val,
                           const int64_t *__restrict__ pack_count,
                           const T       *__restrict__ inp,
                           T             *__restrict__ out)
{
    disco_fwd_dense_d<BDIM_X, ELXTH, T>(Hi, Wi, K, Ho, Wo, NBR_PAD, pscale,
                                        pack_idx, pack_val, pack_count, inp, out);
}

template <int NTH, int ELXTH, typename T>
static void launch_dense_fwd(int B, int C, int K, int Hi, int Wi, int Ho, int Wo,
                             int NBR_PAD, int pscale,
                             const int64_t *pack_idx, const T *pack_val,
                             const int64_t *pack_count, const T *inp, T *out,
                             cudaStream_t stream)
{
    if constexpr (ELXTH <= ELXTH_MAX) {
        if (NTH * ELXTH >= Wo) {
            dim3 grid((unsigned)(K * Ho), (unsigned)(B * C));
            // The kernel reads inp_sh[wi_full] for all pp in [0, NTH*ELXTH) without
            // gating on pp < Wo (we discard the corresponding acc[i] at write-time
            // instead). Max wi_full = (Wi-1) + pscale*(NTH*ELXTH-1), so the shmem
            // window must extend past 2*Wi by pscale*(NTH*ELXTH-Wo) bytes of slack.
            // The slack stays uninitialized — its contents feed acc[i] for pp>=Wo
            // and are discarded. Same trick the CSR fwd kernel uses.
            const size_t shmem = sizeof(T) * (size_t)(2 * Wi + pscale * (NTH * ELXTH - Wo));
            disco_fwd_dense_blk_k<NTH, ELXTH, T><<<grid, NTH, shmem, stream>>>(
                Hi, Wi, K, Ho, Wo, NBR_PAD, pscale,
                pack_idx, pack_val, pack_count, inp, out);
        } else {
            launch_dense_fwd<NTH, ELXTH + 1, T>(
                B, C, K, Hi, Wi, Ho, Wo, NBR_PAD, pscale,
                pack_idx, pack_val, pack_count, inp, out, stream);
        }
    }
    return;
}

torch::Tensor disco_cuda_fwd_dense(torch::Tensor inp,
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
    const int64_t Hi = inp.size(2);
    const int64_t Wi = inp.size(3);

    TORCH_CHECK(Wi % Wo == 0,
                "Wi (", Wi, ") must be an integer multiple of Wo (", Wo, ")");

    TORCH_CHECK(pack_idx.dim()   == 4 && pack_idx.size(0)   == K && pack_idx.size(1)   == Ho && pack_idx.size(3) == 2,
                "pack_idx must have shape [K, Ho, NBR_PAD, 2]");
    TORCH_CHECK(pack_val.dim()   == 3 && pack_val.size(0)   == K && pack_val.size(1)   == Ho,
                "pack_val must have shape [K, Ho, NBR_PAD]");
    TORCH_CHECK(pack_count.dim() == 2 && pack_count.size(0) == K && pack_count.size(1) == Ho,
                "pack_count must have shape [K, Ho]");

    const int64_t NBR_PAD = pack_idx.size(2);
    const int     pscale  = (int)(Wi / Wo);

    // Output is zero-initialized: empty (k, ho) rows from the distributed lat split
    // are then represented correctly without the kernel having to touch them.
    auto options = torch::TensorOptions().device(inp.device()).dtype(inp.dtype());
    auto out = torch::zeros({B, C, K, Ho, Wo}, options);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    static_assert(0 == (ELXTH_MAX % 2));

    if (Wo <= 64 * ELXTH_MAX) {
        AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_fwd_dense_cuda", ([&] {
            launch_dense_fwd<64, 1, scalar_t>(
                (int)B, (int)C, (int)K, (int)Hi, (int)Wi, (int)Ho, (int)Wo,
                (int)NBR_PAD, pscale,
                pack_idx.data_ptr<int64_t>(),
                pack_val.data_ptr<scalar_t>(),
                pack_count.data_ptr<int64_t>(),
                inp.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(), stream);
        }));
    } else if (Wo <= 128 * ELXTH_MAX) {
        AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_fwd_dense_cuda", ([&] {
            launch_dense_fwd<128, (ELXTH_MAX / 2) + 1, scalar_t>(
                (int)B, (int)C, (int)K, (int)Hi, (int)Wi, (int)Ho, (int)Wo,
                (int)NBR_PAD, pscale,
                pack_idx.data_ptr<int64_t>(),
                pack_val.data_ptr<scalar_t>(),
                pack_count.data_ptr<int64_t>(),
                inp.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(), stream);
        }));
    } else if (Wo <= 256 * ELXTH_MAX) {
        AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_fwd_dense_cuda", ([&] {
            launch_dense_fwd<256, (ELXTH_MAX / 2) + 1, scalar_t>(
                (int)B, (int)C, (int)K, (int)Hi, (int)Wi, (int)Ho, (int)Wo,
                (int)NBR_PAD, pscale,
                pack_idx.data_ptr<int64_t>(),
                pack_val.data_ptr<scalar_t>(),
                pack_count.data_ptr<int64_t>(),
                inp.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(), stream);
        }));
    } else if (Wo <= 512 * ELXTH_MAX) {
        AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_fwd_dense_cuda", ([&] {
            launch_dense_fwd<512, (ELXTH_MAX / 2) + 1, scalar_t>(
                (int)B, (int)C, (int)K, (int)Hi, (int)Wi, (int)Ho, (int)Wo,
                (int)NBR_PAD, pscale,
                pack_idx.data_ptr<int64_t>(),
                pack_val.data_ptr<scalar_t>(),
                pack_count.data_ptr<int64_t>(),
                inp.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(), stream);
        }));
    } else if (Wo <= 1024 * ELXTH_MAX) {
        AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_fwd_dense_cuda", ([&] {
            launch_dense_fwd<1024, (ELXTH_MAX / 2) + 1, scalar_t>(
                (int)B, (int)C, (int)K, (int)Hi, (int)Wi, (int)Ho, (int)Wo,
                (int)NBR_PAD, pscale,
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
    m.impl("forward_dense", &disco_cuda_fwd_dense);
}

}
