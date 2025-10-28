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

#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

//#include "cudamacro.h"
#include "attention_cuda.cuh"

#define THREADS (64)

#define TRANSP_WARPS_X_TILE_GENERIC (32)
#define TRANSP_WARPS_X_TILE_SM100    (4)

namespace attention_kernels {

// BEGIN - CSR rows sorting kernels and functions
__global__ void set_rlen_rids_k(const int n,
                                const int64_t *__restrict__ offs,
                                      int *__restrict__ rids,
                                      int *__restrict__ rlen) {

    const int nth = gridDim.x*blockDim.x;
    const int tid = blockIdx.x*blockDim.x + threadIdx.x;

    for(int i = tid; i < n; i += nth) {
        rids[i] = i;
        rlen[i] = offs[i+1]-offs[i];
    }

    return;
}   

at::Tensor sortRows(int nlat_out, at::Tensor row_off, cudaStream_t stream) {

    int64_t *_row_off_d = reinterpret_cast<int64_t *>(row_off.data_ptr());

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(row_off.device());

    torch::Tensor rids_d = torch::empty({nlat_out}, options);
    torch::Tensor rlen_d = torch::empty({nlat_out}, options);

    int *_rids_d = reinterpret_cast<int *>(rids_d.data_ptr());
    int *_rlen_d = reinterpret_cast<int *>(rlen_d.data_ptr());

    const int grid = DIV_UP(nlat_out, THREADS);
    const int block = THREADS;

    set_rlen_rids_k<<<grid, block, 0, stream>>>(nlat_out,
                                                _row_off_d,
                                                _rids_d,
                                                _rlen_d);

    torch::Tensor rids_sort_d = torch::empty({nlat_out}, options);
    torch::Tensor rlen_sort_d = torch::empty({nlat_out}, options);

    int *_rids_sort_d = reinterpret_cast<int *>(rids_sort_d.data_ptr());
    int *_rlen_sort_d = reinterpret_cast<int *>(rlen_sort_d.data_ptr());

    size_t temp_storage_bytes = 0;
    CHECK_CUDA(cub::DeviceRadixSort::SortPairsDescending(NULL, temp_storage_bytes,
                                                         _rlen_d, _rlen_sort_d, 
                                                         _rids_d, _rids_sort_d,
                                                         nlat_out, 0, sizeof(*_rlen_d)*8, stream));

    options = torch::TensorOptions().dtype(torch::kByte).device(row_off.device());
    torch::Tensor temp_storage_d = torch::empty({int64_t(temp_storage_bytes)}, options);

    void *_temp_storage_d = reinterpret_cast<void *>(temp_storage_d.data_ptr());

    CHECK_CUDA(cub::DeviceRadixSort::SortPairsDescending(_temp_storage_d, temp_storage_bytes,
                                                         _rlen_d, _rlen_sort_d, 
                                                         _rids_d, _rids_sort_d,
                                                         nlat_out, 0, sizeof(*_rlen_d)*8, stream));
    return rids_sort_d;
}
// END - CSR rows sorting kernels and functions

// BEGIN - general host-side functions
unsigned int next_pow2(unsigned int x) { 

    x -= 1;

    #pragma unroll
    for(int i = 1; i <= sizeof(x)*8 / 2; i *= 2) {
        x |= x >> i;    
    }
    return x+1;
}
// END - general host-side functions

}
