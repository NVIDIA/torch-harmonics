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

#include "attention.cuh"
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/CUDAUtils.h>

#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

#include "cudamacro.h"
#include "attention_utils.cuh"

#define THREADS (64)

#define TRANSP_WARPS_X_TILE_GENERIC (32)
#define TRANSP_WARPS_X_TILE_SM100    (4)

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


// BEGIN - 4D tensor permutation kernels and functions
template<int BDIM_X,
         int BDIM_Y,
         typename VAL_T>
__global__
__launch_bounds__(BDIM_X*BDIM_Y)
void  permute_to0231_k(const int nchn,
                       const int nlat,
                       const int nlon,
                       const torch::PackedTensorAccessor32<VAL_T, 4, torch::RestrictPtrTraits> src,
                             torch::PackedTensorAccessor32<VAL_T, 4, torch::RestrictPtrTraits> dst) {

    static_assert(!(BDIM_X & (BDIM_X-1)));
    static_assert(!(BDIM_Y & (BDIM_Y-1)));
    static_assert(BDIM_X >= BDIM_Y);

    __shared__ VAL_T sh[BDIM_X][BDIM_X+1];

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int coff  = blockIdx.x*BDIM_X;           // channel offset
    const int woff  = blockIdx.y*BDIM_X;           // width offset
    const int batch = blockIdx.z / nlat;           // batch (same for all block)
    const int h     = blockIdx.z - (batch * nlat); // height (same for all block)

    const int nchn_full = (nchn-coff) >= BDIM_X;
    const int nlon_full = (nlon-woff) >= BDIM_X;

    if (nchn_full && nlon_full) {
        #pragma unroll
        for(int j = 0; j < BDIM_X; j += BDIM_Y) {
            sh[j+tidy][tidx] = src[batch][coff + j+tidy][h][woff+tidx];
        }
        __syncthreads();

        #pragma unroll
        for(int j = 0; j < BDIM_X; j += BDIM_Y) {
            dst[batch][h][woff + j+tidy][coff+tidx] = sh[tidx][j+tidy];
        }
    } else {
        if (woff+tidx < nlon) {
            #pragma unroll
            for(int j = 0; j < BDIM_X; j += BDIM_Y) {
                sh[j+tidy][tidx] = (coff + j+tidy < nchn) ? src[batch][coff + j+tidy][h][woff+tidx] : 0.f;
            }
        }
        __syncthreads();

        if (coff+tidx < nchn) {
            #pragma unroll
            for(int j = 0; j < BDIM_X; j += BDIM_Y) {
                if (woff + j+tidy < nlon) {
                    dst[batch][h][woff + j+tidy][coff+tidx] = sh[tidx][j+tidy];
                }
            }
        }
    }
    return;
}

__global__ void empty_k() {}

static int getPtxver() {
    cudaFuncAttributes attrs;
    CHECK_CUDA(cudaFuncGetAttributes(&attrs, empty_k));
    return attrs.ptxVersion*10;
}

at::Tensor permute_4D_floatT_to0231(at::Tensor src, cudaStream_t stream) {

    dim3 block;
    dim3 grid;

    block.x = WARP_SIZE;
    grid.x = DIV_UP(src.size(1), block.x);
    grid.y = DIV_UP(src.size(3), block.x);
    grid.z = src.size(2)*src.size(0);

    assert(grid.y < 65536);
    assert(grid.z < 65536);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(src.device());
    torch::Tensor dst = torch::empty({src.size(0), src.size(2), src.size(3), src.size(1)}, options);

    const int ptxv = getPtxver();

    // to be further specialized for additional archs, if necessary
    if (ptxv < 100) {
        block.y = TRANSP_WARPS_X_TILE_GENERIC;
        permute_to0231_k<WARP_SIZE, TRANSP_WARPS_X_TILE_GENERIC>
                        <<<grid, block, 0, stream>>>(src.size(1),
                                                     src.size(2),
                                                     src.size(3),
                                                     src.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                     dst.packed_accessor32<float, 4, torch::RestrictPtrTraits>());
        CHECK_ERROR("permute_to0231_k_tile_generic");
    } else {
        block.y = TRANSP_WARPS_X_TILE_SM100;
        permute_to0231_k<WARP_SIZE, TRANSP_WARPS_X_TILE_SM100>
                        <<<grid, block, 0, stream>>>(src.size(1),
                                                     src.size(2),
                                                     src.size(3),
                                                     src.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                     dst.packed_accessor32<float, 4, torch::RestrictPtrTraits>());
        CHECK_ERROR("permute_to0231_k_tile_sm100");
    }

    return dst;
}

template<int BDIM_X,
         int BDIM_Y,
         typename VAL_T>
__global__
__launch_bounds__(BDIM_X*BDIM_Y)
void  permute_to0312_k(const int nchn,
                       const int nlat,
                       const int nlon,
                       const torch::PackedTensorAccessor32<VAL_T, 4, torch::RestrictPtrTraits> src,
                             torch::PackedTensorAccessor32<VAL_T, 4, torch::RestrictPtrTraits> dst) {

    static_assert(!(BDIM_X & (BDIM_X-1)));
    static_assert(!(BDIM_Y & (BDIM_Y-1)));
    static_assert(BDIM_X >= BDIM_Y);

    __shared__ VAL_T sh[BDIM_X][BDIM_X+1];

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int woff  = blockIdx.x*BDIM_X;           // width offset
    const int coff  = blockIdx.y*BDIM_X;           // channel offset
    const int batch = blockIdx.z / nlat;           // batch (same for all block)
    const int h     = blockIdx.z - (batch * nlat); // height (same for all block)

    const int nchn_full = (nchn-coff) >= BDIM_X;
    const int nlon_full = (nlon-woff) >= BDIM_X;

    if (nchn_full && nlon_full) {
        #pragma unroll
        for(int j = 0; j < BDIM_X; j += BDIM_Y) {
            sh[j+tidy][tidx] = src[batch][h][woff + j+tidy][coff+tidx];
        }
        __syncthreads();

        #pragma unroll
        for(int j = 0; j < BDIM_X; j += BDIM_Y) {
            dst[batch][coff + j+tidy][h][woff+tidx] = sh[tidx][j+tidy];
        }
    } else {
        if (coff+tidx < nchn) {
            #pragma unroll
            for(int j = 0; j < BDIM_X; j += BDIM_Y) {
                sh[j+tidy][tidx] = (woff + j+tidy < nlon) ? src[batch][h][woff + j+tidy][coff+tidx] : 0.f;
            }
        }
        __syncthreads();

        if (woff+tidx < nlon) {
            #pragma unroll
            for(int j = 0; j < BDIM_X; j += BDIM_Y) {
                if (coff + j+tidy < nchn) {
                    dst[batch][coff + j+tidy][h][woff+tidx] = sh[tidx][j+tidy];;
                }
            }
        }
    }
    return;
}

at::Tensor permute_4D_floatT_to0312(at::Tensor src, cudaStream_t stream) {

    dim3 block;
    dim3 grid;

    block.x = WARP_SIZE;
    grid.x = DIV_UP(src.size(2), block.x);
    grid.y = DIV_UP(src.size(3), block.x);
    grid.z = src.size(1)*src.size(0);

    assert(grid.y < 65536);
    assert(grid.z < 65536);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(src.device());
    torch::Tensor dst = torch::empty({src.size(0), src.size(3), src.size(1), src.size(2)}, options);

    const int ptxv = getPtxver();

    // to be further specialized for additional archs, if necessary
    if (ptxv < 100) {
        block.y = TRANSP_WARPS_X_TILE_GENERIC;
        permute_to0312_k<WARP_SIZE, TRANSP_WARPS_X_TILE_GENERIC>
                        <<<grid, block, 0, stream>>>(src.size(3),
                                                     src.size(1),
                                                     src.size(2),
                                                     src.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                     dst.packed_accessor32<float, 4, torch::RestrictPtrTraits>());
        CHECK_ERROR("permute_to0312_k_tile_generic");
    } else {
        block.y = TRANSP_WARPS_X_TILE_SM100;
        permute_to0312_k<WARP_SIZE, TRANSP_WARPS_X_TILE_SM100>
                        <<<grid, block, 0, stream>>>(src.size(3),
                                                     src.size(1),
                                                     src.size(2),
                                                     src.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                     dst.packed_accessor32<float, 4, torch::RestrictPtrTraits>());
        CHECK_ERROR("permute_to0312_k_tile_sm100");
    }

    return dst;
}
// END - tensor permutation kernels and functions

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
