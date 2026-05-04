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
// Disco backward — dense-packed psi (CUDA, shmem-staged accumulator, channel-tiled)
// =====================================================================================
//
// Same structure as the previous shmem-staged bwd, but each CTA now owns BC_TILE
// channels for fixed (k, hi). The shmem accumulator is replicated BC_TILE-fold
// (one residue-quotient grid per channel), per-thread dout registers grow to
// BC_TILE × ELXTH, and on each ho transition the kernel atomic-flushes BC_TILE
// rows back to dinp.
//
// Reuse: each nz's val read is shared across BC_TILE channels — BC_TILE FMAs
// per val. Same factor of compute density that the fwd kernel got.
//
// STORAGE_T is the on-disk type of inp (dout); the kernel reads inp in STORAGE_T
// and casts to COMPUTE_T at the FMA. The output (dinp) is in COMPUTE_T because
// fp32 atomicAdd is the only one available pre-Hopper for fp16/bf16 storage —
// the host wrapper casts back to STORAGE_T before returning.
//
// Shmem layout (logical 3D, flattened to 2D):
//   __sh[BC_TILE * pscale][2 * BDIM_X * ELXTH]
// indexed as __sh[bc_off * pscale + r][slot]. Shmem is in COMPUTE_T to match
// the accumulator dtype, so the per-nz add into shmem is plain fp32 += fp32.
//
// BC_TILE=1 reduces to the previous single-channel kernel.
// =====================================================================================

#include "../disco.h"
#include "disco_cuda.cuh"

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>

namespace disco_kernels {

template <int BDIM_X, int ELXTH, int BC_TILE, typename STORAGE_T, typename COMPUTE_T>
__device__ void disco_bwd_dense_d(int Hi, int Wi, int K, int Ho, int Wo, int NBR_PAD, int pscale,
                                  const int64_t   *__restrict__ pack_idx,
                                  const COMPUTE_T *__restrict__ pack_val,
                                  const int64_t   *__restrict__ pack_count,
                                  const STORAGE_T *__restrict__ inp,
                                  COMPUTE_T       *__restrict__ out)
{
    const int tid = threadIdx.x;
    const int bc_tile_idx = blockIdx.y;
    const int bc_start    = bc_tile_idx * BC_TILE;
    const int kh  = blockIdx.x;
    const int k   = kh / Hi;
    const int hi  = kh - k * Hi;

    const int64_t kh_off = (int64_t)k * Hi + hi;
    const int64_t   *idx_kh = pack_idx + kh_off * NBR_PAD * 2;
    const COMPUTE_T *val_kh = pack_val + kh_off * NBR_PAD;
    const int        cnt    = (int)pack_count[kh_off];

    // Empty (k, hi) row — dinp is zero-initialized so leave untouched.
    if (cnt == 0) return;

    // Shmem: BC_TILE * pscale rows of (2 * BDIM_X * ELXTH) entries each, in COMPUTE_T.
    extern __shared__ __align__(sizeof(double)) unsigned char __sh_raw[];
    COMPUTE_T (*__sh)[2 * BDIM_X * ELXTH] = reinterpret_cast<COMPUTE_T(*)[2 * BDIM_X * ELXTH]>(__sh_raw);

    // Per-thread dout cache in COMPUTE_T: BC_TILE × ELXTH register slots covering
    // this CTA's wi positions (cast from STORAGE_T at load).
    COMPUTE_T __reg[BC_TILE][ELXTH];
    #pragma unroll
    for (int bc_off = 0; bc_off < BC_TILE; bc_off++) {
        const STORAGE_T *inp_kh = inp + (((int64_t)(bc_start + bc_off) * K + k) * Hi + hi) * Wi;
        #pragma unroll
        for (int i = 0; i < ELXTH; i++) {
            const int p = i * BDIM_X + tid;
            __reg[bc_off][i] = (p < Wi) ? static_cast<COMPUTE_T>(inp_kh[p]) : static_cast<COMPUTE_T>(0);
        }
    }

    // Reset all BC_TILE * pscale shmem rows.
    for (int row = 0; row < BC_TILE * pscale; row++) {
        #pragma unroll
        for (int j = 0; j < 2 * BDIM_X * ELXTH; j += BDIM_X) {
            __sh[row][j + tid] = static_cast<COMPUTE_T>(0);
        }
    }
    __syncthreads();

    int ho_prev = (int)idx_kh[0];

    for (int nz = 0; nz < cnt; nz++) {
        const int       ho      = (int)idx_kh[nz * 2 + 0];
        const int       wo_base = (int)idx_kh[nz * 2 + 1];
        const COMPUTE_T v       = val_kh[nz];

        // ho transition: flush all BC_TILE channels' shmem to dinp[bc, ho_prev, *]
        // and reset to zero.
        if (ho != ho_prev) {
            __syncthreads();
            for (int bc_off = 0; bc_off < BC_TILE; bc_off++) {
                COMPUTE_T *out_b = out + (int64_t)(bc_start + bc_off) * Ho * Wo;
                for (int r = 0; r < pscale; r++) {
                    const int row = bc_off * pscale + r;
                    for (int j = tid; j < Wi; j += BDIM_X) {
                        const COMPUTE_T s = __sh[row][j] + __sh[row][Wi + j];
                        atomicAdd(&out_b[(int64_t)ho_prev * Wo + j * pscale + r], s);
                        __sh[row][j]      = static_cast<COMPUTE_T>(0);
                        __sh[row][Wi + j] = static_cast<COMPUTE_T>(0);
                    }
                }
            }
            __syncthreads();
            ho_prev = ho;
        }

        const int w_mod_ps = wo_base % pscale;
        const int w_div_ps = wo_base / pscale;

        #pragma unroll
        for (int i = 0; i < ELXTH; i++) {
            const int pp = i * BDIM_X + tid;
            #pragma unroll
            for (int bc_off = 0; bc_off < BC_TILE; bc_off++) {
                const int row = bc_off * pscale + w_mod_ps;
                __sh[row][w_div_ps + pp] += v * __reg[bc_off][i];
            }
        }

        __syncthreads();
    }
    __syncthreads();

    // Final flush for the last ho_prev row.
    for (int bc_off = 0; bc_off < BC_TILE; bc_off++) {
        COMPUTE_T *out_b = out + (int64_t)(bc_start + bc_off) * Ho * Wo;
        for (int r = 0; r < pscale; r++) {
            const int row = bc_off * pscale + r;
            for (int j = tid; j < Wi; j += BDIM_X) {
                const COMPUTE_T s = __sh[row][j] + __sh[row][Wi + j];
                atomicAdd(&out_b[(int64_t)ho_prev * Wo + j * pscale + r], s);
            }
        }
    }
}

template <int BDIM_X, int ELXTH, int BC_TILE, int PSCALE, typename STORAGE_T, typename COMPUTE_T>
__global__ __launch_bounds__(BDIM_X)
void disco_bwd_dense_blk_k(int Hi, int Wi, int K, int Ho, int Wo, int NBR_PAD, int pscale,
                           const int64_t   *__restrict__ pack_idx,
                           const COMPUTE_T *__restrict__ pack_val,
                           const int64_t   *__restrict__ pack_count,
                           const STORAGE_T *__restrict__ inp,
                           COMPUTE_T       *__restrict__ out)
{
    if constexpr (PSCALE != 0) {
        disco_bwd_dense_d<BDIM_X, ELXTH, BC_TILE, STORAGE_T, COMPUTE_T>(Hi, Wi, K, Ho, Wo, NBR_PAD, PSCALE,
                                                                        pack_idx, pack_val, pack_count, inp, out);
    } else {
        disco_bwd_dense_d<BDIM_X, ELXTH, BC_TILE, STORAGE_T, COMPUTE_T>(Hi, Wi, K, Ho, Wo, NBR_PAD, pscale,
                                                                        pack_idx, pack_val, pack_count, inp, out);
    }
}

template <int NTH, int ELXTH, int BC_TILE, typename STORAGE_T, typename COMPUTE_T>
static void launch_dense_bwd(int B, int C, int K, int Hi, int Wi, int Ho, int Wo,
                             int NBR_PAD,
                             const int64_t *pack_idx, const COMPUTE_T *pack_val,
                             const int64_t *pack_count, const STORAGE_T *inp, COMPUTE_T *out,
                             cudaStream_t stream)
{
    if constexpr (ELXTH <= ELXTH_MAX) {
        if (NTH * ELXTH >= Wi) {
            const int BC = B * C;
            const int pscale = Wo / Wi;
            const size_t shmem = sizeof(COMPUTE_T) * (size_t)(BC_TILE * 2 * (NTH * ELXTH) * pscale);

            // Opt in to larger dynamic shmem when BC_TILE > 1 — V100 default
            // carveout is 48KB but BC_TILE=4 with pscale=1 needs up to ~64KB.
            #define SET_ATTR(KFN)                                                          \
                cudaFuncSetAttribute(reinterpret_cast<const void *>(KFN),                  \
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,          \
                                     (int)shmem)

            dim3 grid((unsigned)(K * Hi), (unsigned)(BC / BC_TILE));

            switch (pscale) {
            case 1: {
                using KFn = void(*)(int, int, int, int, int, int, int,
                                    const int64_t *, const COMPUTE_T *, const int64_t *, const STORAGE_T *, COMPUTE_T *);
                KFn fn = &disco_bwd_dense_blk_k<NTH, ELXTH, BC_TILE, 1, STORAGE_T, COMPUTE_T>;
                SET_ATTR(fn);
                disco_bwd_dense_blk_k<NTH, ELXTH, BC_TILE, 1, STORAGE_T, COMPUTE_T><<<grid, NTH, shmem, stream>>>(
                    Hi, Wi, K, Ho, Wo, NBR_PAD, pscale,
                    pack_idx, pack_val, pack_count, inp, out);
                break;
            }
            case 2: {
                using KFn = void(*)(int, int, int, int, int, int, int,
                                    const int64_t *, const COMPUTE_T *, const int64_t *, const STORAGE_T *, COMPUTE_T *);
                KFn fn = &disco_bwd_dense_blk_k<NTH, ELXTH, BC_TILE, 2, STORAGE_T, COMPUTE_T>;
                SET_ATTR(fn);
                disco_bwd_dense_blk_k<NTH, ELXTH, BC_TILE, 2, STORAGE_T, COMPUTE_T><<<grid, NTH, shmem, stream>>>(
                    Hi, Wi, K, Ho, Wo, NBR_PAD, pscale,
                    pack_idx, pack_val, pack_count, inp, out);
                break;
            }
            case 3: {
                using KFn = void(*)(int, int, int, int, int, int, int,
                                    const int64_t *, const COMPUTE_T *, const int64_t *, const STORAGE_T *, COMPUTE_T *);
                KFn fn = &disco_bwd_dense_blk_k<NTH, ELXTH, BC_TILE, 3, STORAGE_T, COMPUTE_T>;
                SET_ATTR(fn);
                disco_bwd_dense_blk_k<NTH, ELXTH, BC_TILE, 3, STORAGE_T, COMPUTE_T><<<grid, NTH, shmem, stream>>>(
                    Hi, Wi, K, Ho, Wo, NBR_PAD, pscale,
                    pack_idx, pack_val, pack_count, inp, out);
                break;
            }
            default: {
                using KFn = void(*)(int, int, int, int, int, int, int,
                                    const int64_t *, const COMPUTE_T *, const int64_t *, const STORAGE_T *, COMPUTE_T *);
                KFn fn = &disco_bwd_dense_blk_k<NTH, ELXTH, BC_TILE, 0, STORAGE_T, COMPUTE_T>;
                SET_ATTR(fn);
                disco_bwd_dense_blk_k<NTH, ELXTH, BC_TILE, 0, STORAGE_T, COMPUTE_T><<<grid, NTH, shmem, stream>>>(
                    Hi, Wi, K, Ho, Wo, NBR_PAD, pscale,
                    pack_idx, pack_val, pack_count, inp, out);
                break;
            }
            }
            #undef SET_ATTR
        } else {
            launch_dense_bwd<NTH, ELXTH + 1, BC_TILE, STORAGE_T, COMPUTE_T>(
                B, C, K, Hi, Wi, Ho, Wo, NBR_PAD,
                pack_idx, pack_val, pack_count, inp, out, stream);
        }
    }
    return;
}

template <int BC_TILE, typename STORAGE_T, typename COMPUTE_T>
static void dispatch_dense_bwd_by_wo(int B, int C, int K, int Hi, int Wi, int Ho, int Wo,
                                     int NBR_PAD,
                                     const int64_t *pack_idx, const COMPUTE_T *pack_val,
                                     const int64_t *pack_count, const STORAGE_T *inp, COMPUTE_T *out,
                                     cudaStream_t stream)
{
    if (Wo <= 64 * ELXTH_MAX) {
        launch_dense_bwd<64, 1, BC_TILE, STORAGE_T, COMPUTE_T>(
            B, C, K, Hi, Wi, Ho, Wo, NBR_PAD,
            pack_idx, pack_val, pack_count, inp, out, stream);
    } else if (Wo <= 128 * ELXTH_MAX) {
        launch_dense_bwd<128, (ELXTH_MAX / 2) + 1, BC_TILE, STORAGE_T, COMPUTE_T>(
            B, C, K, Hi, Wi, Ho, Wo, NBR_PAD,
            pack_idx, pack_val, pack_count, inp, out, stream);
    } else if (Wo <= 256 * ELXTH_MAX) {
        launch_dense_bwd<256, (ELXTH_MAX / 2) + 1, BC_TILE, STORAGE_T, COMPUTE_T>(
            B, C, K, Hi, Wi, Ho, Wo, NBR_PAD,
            pack_idx, pack_val, pack_count, inp, out, stream);
    } else if (Wo <= 512 * ELXTH_MAX) {
        launch_dense_bwd<512, (ELXTH_MAX / 2) + 1, BC_TILE, STORAGE_T, COMPUTE_T>(
            B, C, K, Hi, Wi, Ho, Wo, NBR_PAD,
            pack_idx, pack_val, pack_count, inp, out, stream);
    } else if (Wo <= 1024 * ELXTH_MAX) {
        launch_dense_bwd<1024, (ELXTH_MAX / 2) + 1, BC_TILE, STORAGE_T, COMPUTE_T>(
            B, C, K, Hi, Wi, Ho, Wo, NBR_PAD,
            pack_idx, pack_val, pack_count, inp, out, stream);
    } else {
        fprintf(stderr, "%s:%d: error, unsupported Wo value (%d), max supported is %d\n",
                __FILE__, __LINE__, Wo, 1024 * ELXTH_MAX);
        exit(EXIT_FAILURE);
    }
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

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    static_assert(0 == (ELXTH_MAX % 2));

    // Channel tiling: BC_TILE=4 when (B*C) divisible by 4 AND pscale=1 (the
    // shmem footprint scales with pscale, so BC_TILE=4 with pscale>=2 can
    // overshoot V100's 96KB max for high BDIM_X*ELXTH cases — fall back to
    // BC_TILE=1 there).
    const int64_t BC = B * C;
    const int     pscale = (int)(Wo / Wi);
    const bool    use_bc4 = (BC % 4 == 0) && (pscale == 1);

    torch::Tensor out_storage; // returned dinp in storage dtype
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        inp.scalar_type(), "disco_bwd_dense_cuda", ([&] {
        using storage_t = scalar_t;
        using compute_t = at::opmath_type<storage_t>;
        const auto compute_dtype = c10::CppTypeToScalarType<compute_t>::value;

        // Allocate dinp accumulator in COMPUTE_T (fp32 atomicAdd is the
        // generally-supported one). Cast back to STORAGE_T at the end.
        auto out_compute = torch::zeros(
            {B, C, Ho, Wo},
            torch::TensorOptions().device(inp.device()).dtype(compute_dtype));

        // pack_val lives in compute_t — upcast on the fly if needed.
        auto pack_val_c = (pack_val.scalar_type() == compute_dtype)
                              ? pack_val
                              : pack_val.to(compute_dtype);

        if (use_bc4) {
            dispatch_dense_bwd_by_wo<4, storage_t, compute_t>(
                (int)B, (int)C, (int)K, (int)Hi, (int)Wi, (int)Ho, (int)Wo, (int)NBR_PAD,
                pack_idx.data_ptr<int64_t>(),
                pack_val_c.data_ptr<compute_t>(),
                pack_count.data_ptr<int64_t>(),
                inp.data_ptr<storage_t>(),
                out_compute.data_ptr<compute_t>(), stream);
        } else {
            dispatch_dense_bwd_by_wo<1, storage_t, compute_t>(
                (int)B, (int)C, (int)K, (int)Hi, (int)Wi, (int)Ho, (int)Wo, (int)NBR_PAD,
                pack_idx.data_ptr<int64_t>(),
                pack_val_c.data_ptr<compute_t>(),
                pack_count.data_ptr<int64_t>(),
                inp.data_ptr<storage_t>(),
                out_compute.data_ptr<compute_t>(), stream);
        }

        // Cast dinp back to storage dtype (no-op when storage_t == compute_t).
        if (compute_dtype == inp.scalar_type()) {
            out_storage = out_compute;
        } else {
            out_storage = out_compute.to(inp.scalar_type());
        }
    }));

    return out_storage;
}

TORCH_LIBRARY_IMPL(disco_kernels, CUDA, m)
{
    m.impl("backward_dense", &disco_cuda_bwd_dense);
}

}
