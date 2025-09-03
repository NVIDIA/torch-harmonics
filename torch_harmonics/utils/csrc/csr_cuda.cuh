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

#pragma once

//
//#include <ATen/cuda/CUDAContext.h>
//#include <ATen/cuda/CUDAUtils.h>

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>

#define WARP_SIZE (32)
#define FULL_MASK (0xFFFFFFFF)

#ifndef DIV_UP
#define DIV_UP(a,b) (((a)+((b)-1))/(b))
#endif

namespace utility_kernels {

// CSR rows sorting kernels and functions
torch::Tensor sortRows(int nlat_out, torch::Tensor row_off, cudaStream_t stream);


// Host tensor dump and CSR manipulation functions
void dump_tensor(const char *fname, torch::Tensor t);
void dump_csr(const char *fname, torch::Tensor roff, torch::Tensor cols);

int part_csr_rows(int *row_perm,
                  const torch::Tensor roff,
                  const torch::Tensor cols,
                  int **part_off,
                  int **part_val);

int verify_part(const int npart,
                const int *part_off,
                const int *part_val,
                const torch::Tensor roff,
                const torch::Tensor cols);

void verify_part_new(const int nlon_out,
                const int nlat_in,
                const int nlon_in,
                const int npart,      // partitioning data
                const int *part_off,
                const int *part_val,
                const torch::Tensor roff, 
                const torch::Tensor cols);

unsigned int next_pow2(unsigned int x);


// utility host functions and templates

template<unsigned int ALIGN>
int is_aligned(const void *ptr) {

    static_assert(0 == (ALIGN & (ALIGN-1)));
    return (0 == (uintptr_t(ptr) & (ALIGN-1)));
}


// utility device functions and templates

template<typename FLOATV_T>
__device__ FLOATV_T __vset(float x) {
    static_assert(sizeof(FLOATV_T) == 0, "Unsupported type for __vset");
    return FLOATV_T{};
}

template<>
__device__ float __forceinline__ __vset<float>(float x) {
    return x;
}

__device__ float __forceinline__ __vmul(float a, float b) {
    return a*b;
}

__device__ float __forceinline__ __vadd(float a, float b) {
    return a+b;
}

__device__ float __forceinline__ __vsub(float a, float b) {
    return a-b;
}

__device__ float __forceinline__ __vred(float a) {
    return a;
}

__device__ float __forceinline__ __vscale(float s, float v) {
    return v*s;
}

__device__ float __forceinline__ __vdiv(float s, float v) {
    return v/s;
}

template<>
__device__ float4 __forceinline__ __vset<float4>(float x) {
    return make_float4(x, x, x, x);
}

__device__ float4 __forceinline__ __vmul(float4 a, float4 b) {
    return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}

__device__ float4 __forceinline__ __vadd(float4 a, float4 b) {
    return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

__device__ float4 __forceinline__ __vsub(float4 a, float4 b) {
    return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}

__device__ float __forceinline__ __vred(float4 a) {
    return a.x + a.y + a.z + a.w;
}

__device__ float4 __forceinline__ __vscale(float s, float4 v) {
    return make_float4(s*v.x, s*v.y, s*v.z, s*v.w);
}

__device__ float4 __forceinline__ __vdiv(float s, float4 v) {
    return make_float4(s/v.x, s/v.y, s/v.z, s/v.w);;
}

template<int BDIM_X>
static __device__ void __sync() {

    unsigned int subwarp_mask = FULL_MASK;

    if constexpr(BDIM_X <= WARP_SIZE) {
        const int tidy = threadIdx.y;
        constexpr unsigned int MASK = (1ull << BDIM_X)-1;
        subwarp_mask = MASK << (tidy*BDIM_X);
    }

    if constexpr(BDIM_X <= WARP_SIZE) { __syncwarp(subwarp_mask); }
    else                              {          __syncthreads(); }

    return;
}

template<typename VAL_T>
__device__ VAL_T __warp_sum(VAL_T val) {

    #pragma unroll
    for(int i = WARP_SIZE/2; i; i /= 2) {
        val += __shfl_xor_sync(FULL_MASK, val, i, WARP_SIZE);
    }
    return val;
}

template<int BDIM_X,
         int BDIM_Y=1,
         int BDIM_Z=1,
         typename VAL_T>
__device__ VAL_T __block_sum(VAL_T val) {

    const int NWARP = (BDIM_X*BDIM_Y*BDIM_Z) / WARP_SIZE;

    val = __warp_sum(val);

    if constexpr(NWARP > 1) {

        int tid = threadIdx.x;
        if constexpr(BDIM_Y > 1) { tid += threadIdx.y*BDIM_X;        }
        if constexpr(BDIM_Z > 1) { tid += threadIdx.z*BDIM_X*BDIM_Y; }

        const int lid = tid%WARP_SIZE;
        const int wid = tid/WARP_SIZE;

        __shared__ VAL_T sh[NWARP];

        if (lid == 0) {
            sh[wid] = val;
        }
        __syncthreads();

        if (wid == 0) {
            val = (lid < NWARP) ? sh[lid] : 0;

            val = __warp_sum(val);
            __syncwarp();

            if (!lid) {
                sh[0] = val;
            }
        }
        __syncthreads();

        val = sh[0];
        __syncthreads();
    }
    return val;
}

}
