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

#include "cudamacro.h"
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


// BEGIN - 4D tensor permutation kernels and functions
__global__ void empty_k() {}

int getPtxver() {
    cudaFuncAttributes attrs;
    CHECK_CUDA(cudaFuncGetAttributes(&attrs, empty_k));
    return attrs.ptxVersion;
}

at::Tensor permute_4D_to0231(at::Tensor src) {

    auto options = torch::TensorOptions().dtype(src.dtype()).device(src.device());
    torch::Tensor dst = torch::empty({src.size(0), src.size(2), src.size(3), src.size(1)}, options);


    const int ptxv = getPtxver();

    // to be further specialized for additional archs, if necessary
    if (ptxv < 100) {
        AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "permute_to0231_k_tile_generic", ([&] {
            launch_permute_to0231<TRANSP_WARPS_X_TILE_GENERIC, scalar_t>(src, dst);
        }));
        CHECK_ERROR("permute_to0231_k_tile_generic");
    } else {
        AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "permute_to0231_k_tile_sm100", ([&] {
            launch_permute_to0231<TRANSP_WARPS_X_TILE_SM100, scalar_t>(src, dst);
        }));
        CHECK_ERROR("permute_to0231_k_tile_sm100");
    }

    return dst;
}

at::Tensor permute_4D_to0312(at::Tensor src) {

    auto options = torch::TensorOptions().dtype(src.dtype()).device(src.device());
    torch::Tensor dst = torch::empty({src.size(0), src.size(3), src.size(1), src.size(2)}, options);

    const int ptxv = getPtxver();

    // to be further specialized for additional archs, if necessary
    if (ptxv < 100) {
        AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "permute_to0312_k_tile_generic", ([&] {
            launch_permute_to0312<TRANSP_WARPS_X_TILE_GENERIC, scalar_t>(src, dst);
        }));
        CHECK_ERROR("permute_to0312_k_tile_generic");
    } else {
        AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "permute_to0312_k_tile_sm100", ([&] {
            launch_permute_to0312<TRANSP_WARPS_X_TILE_SM100, scalar_t>(src, dst);
        }));
        CHECK_ERROR("permute_to0312_k_tile_sm100");
    }

    return dst;
}


__global__ void get_rlen_boundary_k(const float thres,
                                    const int64_t split_len,
                                    const int64_t n,
                                    const int32_t *idx,
                                    const int64_t *off,
                                          int64_t *num_lrow_ptr,
                                          int64_t *max_rlen_ptr0,
                                          int64_t *max_rlen_ptr1) {
    const int tid = threadIdx.x;

    int64_t max_rlen = off[idx[0]+1]-off[idx[0]];

    int64_t min_longr_len = max(split_len, int64_t(max_rlen*thres));

    int64_t tot_long = 0;
    for(int64_t i = 0; i < n; i += blockDim.x) {

        int64_t rlen = 0;

        if (i+tid < n) {
            int32_t row = idx[i+tid];
            rlen = off[row+1]-off[row];
        }

        int n_long = __syncthreads_count(rlen >= min_longr_len);
        if (n_long == 0) { break; }
  
        tot_long += n_long;
    }

    if (!tid) {
        *num_lrow_ptr = tot_long;
        *max_rlen_ptr0 = tot_long ? max_rlen : 0;

        if (tot_long < n) {
            int32_t first_short_row = idx[tot_long];
            *max_rlen_ptr1 = off[first_short_row+1] - off[first_short_row];
        } else {
            *max_rlen_ptr1 = 0;
        }
    }

    return;
}   

// ASSUMES row_idx sorted by decreasing length.
//
// Splits the rows int two sections:
// 1) "long  rows": first "n_long_rows" with (length >= split_len && length >= thres*max_row_length);
// 2) "short rows": remaining rows with      (                       length <  thres*max_row_length);
//
// Note than split_len is only used to determine whether a row is long or not. If there are
// long rows, then the short ones are selected based on the condition that their length is 
// less than one tenth the longest long row, regardless of the value of split_len (i.e.
// short rows can have length >= split_len, if thres*max_row_length > split_len).
//
// Returns:
//  n_long_rows: size of section 1;
//  max_row_len0: max row length of section 1, or 0 if section 1 is empty (i.e., n_long_rows == 0).
//  max_row_len1: max row length of section 2, or 0 if section 2 is empty (i.e., n_long_rows == nrows).

void split_csr_rows(float thres,
                    int64_t split_len, // minimum length for long rows
                    int64_t nrows,
                    int32_t *row_idx,
                    int64_t *row_off,
                    int64_t *n_long_rows,
                    int64_t *max_row_len0,
                    int64_t *max_row_len1) {

    if (!nrows) {
        *n_long_rows = 0;
        *max_row_len0 = 0;
        *max_row_len1 = 0;
        return;
    }

    torch::Tensor tmp_d = torch::empty({3}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    int64_t *tmp_ptr_d = reinterpret_cast<int64_t *>(tmp_d.data_ptr());

    int64_t *num_lr  = tmp_ptr_d;
    int64_t *max_rl0 = tmp_ptr_d+1;
    int64_t *max_rl1 = tmp_ptr_d+2;
    
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    get_rlen_boundary_k<<<1, 1024, 0, stream>>>(thres, split_len, nrows, row_idx, row_off, num_lr, max_rl0, max_rl1);

    at::Tensor tmp_h = tmp_d.cpu();
    int64_t *tmp_ptr_h = tmp_h.data_ptr<int64_t>();

    *n_long_rows  = tmp_ptr_h[0];
    *max_row_len0 = tmp_ptr_h[1];
    *max_row_len1 = tmp_ptr_h[2];
#if 0
    int64_t len_thresh = *max_row_len0*thres;
    int64_t max_thres = max(len_thresh, split_len);
    printf("Rows split in two sections:\n");
    if (n_long_rows[0]) {
        printf("\t%ld long rows, range: [0 - %ld], max length: %ld, (all lengths >= max(%ld, %ld)=%ld)\n",
               n_long_rows[0], n_long_rows[0]-1, *max_row_len0, split_len, len_thresh, max_thres);
    } else {
        printf("\tno long rows found!\n");
    }
    if (n_long_rows[0] < nrows) {
        printf("\t%ld short rows, range: [%ld - %ld], max length: %ld, (all lengths <  max(%ld, %ld)=%ld)\n",
               nrows-n_long_rows[0], n_long_rows[0], nrows-1, *max_row_len1, split_len, len_thresh, max_thres);
    } else {
        printf("\tno short rows found!\n");
    }
#endif
    return;
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

void ensure_dyn_shmem(const void* kern, size_t shsize) {

    if (shsize <= 48u*1024u) {
        return;
    }

    static std::unordered_set<const void*> done;

    if (done.insert(kern).second) {
        // this value only impacts whether a lauch specifying
        // a dyn shmem amount fails or not; it has no impact
        // on occupancy/L1
        CHECK_CUDA(cudaFuncSetAttribute(
                   kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   static_cast<int>(shsize)));
    }
}
// END - general host-side functions

}
