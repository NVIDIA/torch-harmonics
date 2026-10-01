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

#include "attention_cuda_utils.cuh"

#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <atomic>
#include <cub/cub.cuh>
#include <limits>
#include <utility>

#include "cudamacro.h"
#include "attention_cuda.cuh"

#define THREADS (64)

#define TRANSP_WARPS_X_TILE_GENERIC (32)
#define TRANSP_WARPS_X_TILE_SM100 (4)

namespace attention_kernels
{

    // BEGIN - CSR rows sorting kernels and functions
    __global__ void set_rlen_rids_k(const int n, const int64_t *__restrict__ offs, int *__restrict__ rids,
                                    int *__restrict__ rlen)
    {

        const int nth = gridDim.x * blockDim.x;
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;

        for (int i = tid; i < n; i += nth) {
            rids[i] = i;
            rlen[i] = offs[i + 1] - offs[i];
        }

        return;
    }

    at::Tensor sortRows(int nlat_out, at::Tensor row_off, cudaStream_t stream)
    {

        int64_t *_row_off_d = reinterpret_cast<int64_t *>(row_off.data_ptr());

        auto options = torch::TensorOptions().dtype(torch::kInt32).device(row_off.device());

        torch::Tensor rids_d = torch::empty({nlat_out}, options);
        torch::Tensor rlen_d = torch::empty({nlat_out}, options);

        int *_rids_d = reinterpret_cast<int *>(rids_d.data_ptr());
        int *_rlen_d = reinterpret_cast<int *>(rlen_d.data_ptr());

        const int grid = DIV_UP(nlat_out, THREADS);
        const int block = THREADS;

        set_rlen_rids_k<<<grid, block, 0, stream>>>(nlat_out, _row_off_d, _rids_d, _rlen_d);

        torch::Tensor rids_sort_d = torch::empty({nlat_out}, options);
        torch::Tensor rlen_sort_d = torch::empty({nlat_out}, options);

        int *_rids_sort_d = reinterpret_cast<int *>(rids_sort_d.data_ptr());
        int *_rlen_sort_d = reinterpret_cast<int *>(rlen_sort_d.data_ptr());

        size_t temp_storage_bytes = 0;
        CHECK_CUDA(cub::DeviceRadixSort::SortPairsDescending(NULL, temp_storage_bytes, _rlen_d, _rlen_sort_d, _rids_d,
                                                             _rids_sort_d, nlat_out, 0, sizeof(*_rlen_d) * 8, stream));

        options = torch::TensorOptions().dtype(torch::kByte).device(row_off.device());
        torch::Tensor temp_storage_d = torch::empty({int64_t(temp_storage_bytes)}, options);

        void *_temp_storage_d = reinterpret_cast<void *>(temp_storage_d.data_ptr());

        CHECK_CUDA(cub::DeviceRadixSort::SortPairsDescending(_temp_storage_d, temp_storage_bytes, _rlen_d, _rlen_sort_d,
                                                             _rids_d, _rids_sort_d, nlat_out, 0, sizeof(*_rlen_d) * 8,
                                                             stream));
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return rids_sort_d;
    }
    // END - CSR rows sorting kernels and functions

    // BEGIN - 4D tensor permutation kernels and functions
    __global__ void empty_k() { }

    int getPtxver()
    {
        // Cached: this is called on every transpose to pick the tile width, but
        // the answer is fixed for the lifetime of the process.
        //
        // Cached *per device* rather than process-wide: ptxVersion describes the
        // cubin actually loaded for empty_k, and a fat binary can load a
        // different one on a device of a different arch. Homogeneous nodes make
        // this moot, but the check costs a TLS read.
        //
        // The race is benign -- concurrent callers on the same device compute
        // the same value -- but the entries are atomic so the write is not a
        // data race. 0 means "not yet queried"; no real PTX version is 0.
        constexpr int MAX_DEVICES = 64;
        static std::atomic<int> cache[MAX_DEVICES];

        int dev = 0;
        CHECK_CUDA(cudaGetDevice(&dev));

        const bool cacheable = (dev >= 0) && (dev < MAX_DEVICES);
        if (cacheable) {
            const int cached = cache[dev].load(std::memory_order_relaxed);
            if (cached != 0) { return cached; }
        }

        cudaFuncAttributes attrs;
        CHECK_CUDA(cudaFuncGetAttributes(&attrs, empty_k));

        if (cacheable) { cache[dev].store(attrs.ptxVersion, std::memory_order_relaxed); }

        return attrs.ptxVersion;
    }

    at::Tensor permute_4D_to0231(at::Tensor src)
    {

        auto options = torch::TensorOptions().dtype(src.dtype()).device(src.device());
        torch::Tensor dst = torch::empty({src.size(0), src.size(2), src.size(3), src.size(1)}, options);

        const int ptxv = getPtxver();

        // to be further specialized for additional archs, if necessary
        if (ptxv < 100) {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kHalf, at::kBFloat16, src.scalar_type(), "permute_to0231_k_tile_generic",
                ([&] { launch_permute_to0231<TRANSP_WARPS_X_TILE_GENERIC, scalar_t>(src, dst); }));
            CHECK_ERROR("permute_to0231_k_tile_generic");
        } else {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kHalf, at::kBFloat16, src.scalar_type(), "permute_to0231_k_tile_sm100",
                ([&] { launch_permute_to0231<TRANSP_WARPS_X_TILE_SM100, scalar_t>(src, dst); }));
            CHECK_ERROR("permute_to0231_k_tile_sm100");
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return dst;
    }

    at::Tensor permute_4D_to0312(at::Tensor src)
    {

        auto options = torch::TensorOptions().dtype(src.dtype()).device(src.device());
        torch::Tensor dst = torch::empty({src.size(0), src.size(3), src.size(1), src.size(2)}, options);

        const int ptxv = getPtxver();

        // to be further specialized for additional archs, if necessary
        if (ptxv < 100) {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kHalf, at::kBFloat16, src.scalar_type(), "permute_to0312_k_tile_generic",
                ([&] { launch_permute_to0312<TRANSP_WARPS_X_TILE_GENERIC, scalar_t>(src, dst); }));
            CHECK_ERROR("permute_to0312_k_tile_generic");
        } else {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kHalf, at::kBFloat16, src.scalar_type(), "permute_to0312_k_tile_sm100",
                ([&] { launch_permute_to0312<TRANSP_WARPS_X_TILE_SM100, scalar_t>(src, dst); }));
            CHECK_ERROR("permute_to0312_k_tile_sm100");
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return dst;
    }

    // Registered op wrappers. The dtype dispatch lives in permute_4D_to*
    // (AT_DISPATCH_FLOATING_TYPES_AND2 over kHalf/kBFloat16): the tiled kernel
    // is templated on the element type and only moves elements, so fp32, fp16
    // and bf16 all share one code path.
    at::Tensor permute_to_nhwc_cuda(at::Tensor x)
    {
        CHECK_CUDA_TENSOR(x);

        // run on the inputs' device: without this, the current stream, the scratch
        // allocations and the per-device queries (ensure_dyn_shmem, getPtxver) would all
        // resolve to whichever CUDA device happens to be current
        const at::cuda::OptionalCUDAGuard device_guard(x.device());
        TORCH_CHECK(x.dim() == 4, "permute_to_nhwc expects a 4D (B, C, H, W) tensor, got ", x.dim(), "D");
        TORCH_CHECK(x.is_contiguous(), "permute_to_nhwc expects a contiguous (B, C, H, W) tensor");
        return permute_4D_to0231(x);
    }

    at::Tensor permute_to_nchw_cuda(at::Tensor x)
    {
        CHECK_CUDA_TENSOR(x);

        // run on the inputs' device: without this, the current stream, the scratch
        // allocations and the per-device queries (ensure_dyn_shmem, getPtxver) would all
        // resolve to whichever CUDA device happens to be current
        const at::cuda::OptionalCUDAGuard device_guard(x.device());
        TORCH_CHECK(x.dim() == 4, "permute_to_nchw expects a 4D (B, H, W, C) tensor, got ", x.dim(), "D");
        TORCH_CHECK(x.is_contiguous(), "permute_to_nchw expects a contiguous (B, H, W, C) tensor");
        return permute_4D_to0312(x);
    }

    TORCH_LIBRARY_IMPL(attention_kernels, CUDA, m)
    {
        m.impl("permute_to_nhwc", &permute_to_nhwc_cuda);
        m.impl("permute_to_nchw", &permute_to_nchw_cuda);
    }
    // END - tensor permutation kernels and functions

    // BEGIN - general host-side functions
    unsigned int next_pow2(unsigned int x)
    {

        x -= 1;

#pragma unroll
        for (int i = 1; i <= sizeof(x) * 8 / 2; i *= 2) { x |= x >> i; }
        return x + 1;
    }

    // END - general host-side functions

} // namespace attention_kernels
