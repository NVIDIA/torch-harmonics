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
// Disco forward — dense-packed psi (CUDA, shmem-staged, channel-tiled)
// =====================================================================================
//
// Each CTA owns a tile of BC_TILE channels for fixed (k, ho). The shmem holds
// BC_TILE input lat rows so each nz iteration fans out BC_TILE accumulator
// updates from a single val read.
//
// STORAGE_T is the on-disk type of inp/out (fp32/fp16/bf16/fp64). COMPUTE_T is
// the promoted op-math type (fp32 for fp16/bf16, otherwise same as STORAGE_T)
// — used for the accumulators and the val reads. The kernel reads inp from
// shmem in STORAGE_T and casts to COMPUTE_T at the FMA, and casts back to
// STORAGE_T on the final write.
//
// BC_TILE=1 is the current default. BC_TILE>1 is still compiled.
// =====================================================================================

#include "../disco.h"
#include "disco_cuda.cuh"

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>

namespace disco_kernels {

template <int BDIM_X, int ELXTH, int BC_TILE, typename STORAGE_T, typename COMPUTE_T>
__device__ void disco_fwd_dense_d(int Hi, int Wi, int K, int Ho, int Wo, int NBR_PAD, int pscale,
                                  int sh_stride,
                                  const int64_t   *__restrict__ pack_idx,
                                  const COMPUTE_T *__restrict__ pack_val,
                                  const int64_t   *__restrict__ pack_count,
                                  const STORAGE_T *__restrict__ inp,
                                  STORAGE_T       *__restrict__ out)
{
    const int tid = threadIdx.x;
    const int bc_tile_idx = blockIdx.y;        // 0 .. (BC / BC_TILE) - 1
    const int bc_start    = bc_tile_idx * BC_TILE;
    const int kh  = blockIdx.x;                // k * Ho + ho
    const int k   = kh / Ho;
    const int ho  = kh - k * Ho;

    const int64_t kh_off = (int64_t)k * Ho + ho;
    const int64_t   *idx_kh = pack_idx + kh_off * NBR_PAD * 2;
    const COMPUTE_T *val_kh = pack_val + kh_off * NBR_PAD;
    const int        cnt    = (int)pack_count[kh_off];

    // Empty (k, ho) row — output is zero-initialized so leave untouched.
    if (cnt == 0) return;

    // Shmem layout: BC_TILE rows of length sh_stride = 2*Wi + slack. The slack
    // covers the over-shoot when (BDIM_X*ELXTH) > Wo (the inner loop reads
    // wi_full = wi_base + pscale*pp where pp can run past Wo, the result is
    // discarded at write-time but the shmem read still happens).
    extern __shared__ __align__(sizeof(double)) unsigned char inp_sh_raw[];
    STORAGE_T *inp_sh = reinterpret_cast<STORAGE_T *>(inp_sh_raw);

    // Per-thread accumulators in COMPUTE_T: BC_TILE × ELXTH.
    COMPUTE_T acc[BC_TILE][ELXTH];
    #pragma unroll
    for (int bc_off = 0; bc_off < BC_TILE; bc_off++) {
        #pragma unroll
        for (int i = 0; i < ELXTH; i++) {
            acc[bc_off][i] = static_cast<COMPUTE_T>(0);
        }
    }

    // Initial load: BC_TILE rows of inp into shmem (duplicated 2x for wrap).
    int hi_prev = (int)idx_kh[0];
    #pragma unroll
    for (int bc_off = 0; bc_off < BC_TILE; bc_off++) {
        const STORAGE_T *inp_row = inp + (int64_t)(bc_start + bc_off) * Hi * Wi
                                       + (int64_t)hi_prev * Wi;
        STORAGE_T *sh_row = inp_sh + bc_off * sh_stride;
        for (int i = tid; i < Wi; i += BDIM_X) {
            const STORAGE_T v = inp_row[i];
            sh_row[i]      = v;
            sh_row[Wi + i] = v;
        }
    }
    __syncthreads();

    for (int nz = 0; nz < cnt; nz++) {
        const int       hi      = (int)idx_kh[nz * 2 + 0];
        const int       wi_base = (int)idx_kh[nz * 2 + 1];
        const COMPUTE_T v       = val_kh[nz];

        if (hi != hi_prev) {
            __syncthreads();
            hi_prev = hi;
            #pragma unroll
            for (int bc_off = 0; bc_off < BC_TILE; bc_off++) {
                const STORAGE_T *inp_row = inp + (int64_t)(bc_start + bc_off) * Hi * Wi
                                               + (int64_t)hi * Wi;
                STORAGE_T *sh_row = inp_sh + bc_off * sh_stride;
                for (int i = tid; i < Wi; i += BDIM_X) {
                    const STORAGE_T vv = inp_row[i];
                    sh_row[i]      = vv;
                    sh_row[Wi + i] = vv;
                }
            }
            __syncthreads();
        }

        #pragma unroll
        for (int i = 0; i < ELXTH; i++) {
            const int pp = i * BDIM_X + tid;
            const int wi_full = wi_base + pscale * pp;
            #pragma unroll
            for (int bc_off = 0; bc_off < BC_TILE; bc_off++) {
                const STORAGE_T *sh_row = inp_sh + bc_off * sh_stride;
                const COMPUTE_T sh_val = static_cast<COMPUTE_T>(sh_row[wi_full]);
                acc[bc_off][i] += v * sh_val;
            }
        }
    }

    // Write results: BC_TILE channels × ELXTH wo positions per thread.
    #pragma unroll
    for (int bc_off = 0; bc_off < BC_TILE; bc_off++) {
        STORAGE_T *out_kh = out + ((int64_t)(bc_start + bc_off) * K + k) * Ho * Wo
                                + (int64_t)ho * Wo;
        #pragma unroll
        for (int i = 0; i < ELXTH; i++) {
            const int pp = i * BDIM_X + tid;
            if (pp < Wo) { out_kh[pp] = static_cast<STORAGE_T>(acc[bc_off][i]); }
        }
    }
}

template <int BDIM_X, int ELXTH, int BC_TILE, typename STORAGE_T, typename COMPUTE_T>
__global__ __launch_bounds__(BDIM_X)
void disco_fwd_dense_blk_k(int Hi, int Wi, int K, int Ho, int Wo, int NBR_PAD, int pscale,
                           int sh_stride,
                           const int64_t   *__restrict__ pack_idx,
                           const COMPUTE_T *__restrict__ pack_val,
                           const int64_t   *__restrict__ pack_count,
                           const STORAGE_T *__restrict__ inp,
                           STORAGE_T       *__restrict__ out)
{
    disco_fwd_dense_d<BDIM_X, ELXTH, BC_TILE, STORAGE_T, COMPUTE_T>(
        Hi, Wi, K, Ho, Wo, NBR_PAD, pscale, sh_stride,
        pack_idx, pack_val, pack_count, inp, out);
}

template <int NTH, int ELXTH, int BC_TILE, typename STORAGE_T, typename COMPUTE_T>
static void launch_dense_fwd(int B, int C, int K, int Hi, int Wi, int Ho, int Wo,
                             int NBR_PAD, int pscale,
                             const int64_t *pack_idx, const COMPUTE_T *pack_val,
                             const int64_t *pack_count, const STORAGE_T *inp, STORAGE_T *out,
                             cudaStream_t stream)
{
    if constexpr (ELXTH <= ELXTH_MAX) {
        if (NTH * ELXTH >= Wo) {
            const int BC = B * C;
            // sh_stride per channel row: 2*Wi covers the duplicated row, plus
            // pscale * (NTH*ELXTH - Wo) of slack for the out-of-range pp values.
            const int sh_stride = 2 * Wi + pscale * (NTH * ELXTH - Wo);
            const size_t shmem  = sizeof(STORAGE_T) * (size_t)(BC_TILE * sh_stride);

            // For BC_TILE > 1 (or large Wi) the shmem exceeds V100's default
            // 48KB carveout; opt in to the device max once.
            using Kernel = void(*)(int, int, int, int, int, int, int, int,
                                   const int64_t *, const COMPUTE_T *, const int64_t *,
                                   const STORAGE_T *, STORAGE_T *);
            Kernel fn = &disco_fwd_dense_blk_k<NTH, ELXTH, BC_TILE, STORAGE_T, COMPUTE_T>;
            cudaFuncSetAttribute(reinterpret_cast<const void *>(fn),
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)shmem);

            dim3 grid((unsigned)(K * Ho), (unsigned)(BC / BC_TILE));
            disco_fwd_dense_blk_k<NTH, ELXTH, BC_TILE, STORAGE_T, COMPUTE_T><<<grid, NTH, shmem, stream>>>(
                Hi, Wi, K, Ho, Wo, NBR_PAD, pscale, sh_stride,
                pack_idx, pack_val, pack_count, inp, out);
        } else {
            launch_dense_fwd<NTH, ELXTH + 1, BC_TILE, STORAGE_T, COMPUTE_T>(
                B, C, K, Hi, Wi, Ho, Wo, NBR_PAD, pscale,
                pack_idx, pack_val, pack_count, inp, out, stream);
        }
    }
    return;
}

template <int BC_TILE, typename STORAGE_T, typename COMPUTE_T>
static void dispatch_dense_fwd_by_wo(int B, int C, int K, int Hi, int Wi, int Ho, int Wo,
                                     int NBR_PAD, int pscale,
                                     const int64_t *pack_idx, const COMPUTE_T *pack_val,
                                     const int64_t *pack_count, const STORAGE_T *inp, STORAGE_T *out,
                                     cudaStream_t stream)
{
    // Dispatch by Wo. Threshold at NTH * (ELXTH_MAX/2) instead of NTH * ELXTH_MAX
    // routes moderate Wo (e.g. 1440) to the next-larger BDIM_X with a smaller
    // per-thread ELXTH — that lowers per-thread accumulator register pressure
    // (acc[BC_TILE][ELXTH] fp32) and avoids the local-memory spill that ncu
    // flagged at ELXTH=23 (49% est. speedup). All branches start ELXTH=1 so
    // the recursion picks the smallest valid ELXTH.
    constexpr int ELXTH_TARGET = ELXTH_MAX / 2;   // target max ELXTH per branch
    if (Wo <= 64 * ELXTH_TARGET) {
        launch_dense_fwd<64, 1, BC_TILE, STORAGE_T, COMPUTE_T>(
            B, C, K, Hi, Wi, Ho, Wo, NBR_PAD, pscale,
            pack_idx, pack_val, pack_count, inp, out, stream);
    } else if (Wo <= 128 * ELXTH_TARGET) {
        launch_dense_fwd<128, 1, BC_TILE, STORAGE_T, COMPUTE_T>(
            B, C, K, Hi, Wi, Ho, Wo, NBR_PAD, pscale,
            pack_idx, pack_val, pack_count, inp, out, stream);
    } else if (Wo <= 256 * ELXTH_TARGET) {
        launch_dense_fwd<256, 1, BC_TILE, STORAGE_T, COMPUTE_T>(
            B, C, K, Hi, Wi, Ho, Wo, NBR_PAD, pscale,
            pack_idx, pack_val, pack_count, inp, out, stream);
    } else if (Wo <= 512 * ELXTH_TARGET) {
        launch_dense_fwd<512, 1, BC_TILE, STORAGE_T, COMPUTE_T>(
            B, C, K, Hi, Wi, Ho, Wo, NBR_PAD, pscale,
            pack_idx, pack_val, pack_count, inp, out, stream);
    } else if (Wo <= 1024 * ELXTH_MAX) {
        // High end: keep the wider ELXTH range (start=1, max=ELXTH_MAX) to
        // preserve the previous max Wo ceiling of 1024 * ELXTH_MAX.
        launch_dense_fwd<1024, 1, BC_TILE, STORAGE_T, COMPUTE_T>(
            B, C, K, Hi, Wi, Ho, Wo, NBR_PAD, pscale,
            pack_idx, pack_val, pack_count, inp, out, stream);
    } else {
        fprintf(stderr, "%s:%d: error, unsupported Wo value (%d), max supported is %d\n",
                __FILE__, __LINE__, Wo, 1024 * ELXTH_MAX);
        exit(EXIT_FAILURE);
    }
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

    auto options = torch::TensorOptions().device(inp.device()).dtype(inp.dtype());
    auto out = torch::zeros({B, C, K, Ho, Wo}, options);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    static_assert(0 == (ELXTH_MAX % 2));

    // Channel-tile dispatch: hard-coded to BC_TILE=1 for now. BC_TILE=4 amortizes
    // each val read across 4 channels in principle, but on V100/H100 it dropped
    // CTAs/SM via shmem pressure (V100 96KB cap → 1 CTA/SM with BC_TILE=4 vs 6
    // with BC_TILE=1). The BC_TILE=4 instantiations are kept compiled so flipping
    // back is a one-liner.
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        inp.scalar_type(), "disco_fwd_dense_cuda", ([&] {
        using storage_t = scalar_t;
        using compute_t = at::opmath_type<storage_t>;
        const auto compute_dtype = c10::CppTypeToScalarType<compute_t>::value;
        // pack_val lives in compute_t — upcast on the fly if the buffer was
        // registered in a different dtype (e.g. fp32 buffer with fp16 storage_t).
        auto pack_val_c = (pack_val.scalar_type() == compute_dtype)
                              ? pack_val
                              : pack_val.to(compute_dtype);
        dispatch_dense_fwd_by_wo<1, storage_t, compute_t>(
            (int)B, (int)C, (int)K, (int)Hi, (int)Wi, (int)Ho, (int)Wo,
            (int)NBR_PAD, pscale,
            pack_idx.data_ptr<int64_t>(),
            pack_val_c.data_ptr<compute_t>(),
            pack_count.data_ptr<int64_t>(),
            inp.data_ptr<storage_t>(),
            out.data_ptr<storage_t>(), stream);
    }));

    return out;
}

TORCH_LIBRARY_IMPL(disco_kernels, CUDA, m)
{
    m.impl("forward_dense", &disco_cuda_fwd_dense);
}

}
