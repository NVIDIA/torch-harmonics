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

#include <torch/torch.h>
#include <ATen/ATen.h>

// include cuda utils
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>


#define WARP_SIZE (32)


namespace utility_kernels {

// transpose utils
template<int BDIM_X,
int BDIM_Y,
typename VAL_T>
__global__
__launch_bounds__(BDIM_X*BDIM_Y)
void  permute_to0231_k(const int nchn,
              const int nlat,
              const int nlon,
              const at::PackedTensorAccessor32<VAL_T, 4, at::RestrictPtrTraits> src,
                    at::PackedTensorAccessor32<VAL_T, 4, at::RestrictPtrTraits> dst) {

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
       sh[j+tidy][tidx] = (coff + j+tidy < nchn) ? src[batch][coff + j+tidy][h][woff+tidx] : VAL_T(0);
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

template<int WARPS_X_TILE, typename VAL_T>
void launch_permute_to0231(torch::Tensor src, torch::Tensor dst){
dim3 block;
dim3 grid;

block.x = WARP_SIZE;
block.y = WARPS_X_TILE;
grid.x = DIV_UP(src.size(1), block.x);
grid.y = DIV_UP(src.size(3), block.x);
grid.z = src.size(2)*src.size(0);

assert(grid.y < 65536);
assert(grid.z < 65536);

// get stream
auto stream = at::cuda::getCurrentCUDAStream().stream();

permute_to0231_k<WARP_SIZE, WARPS_X_TILE>
               <<<grid, block, 0, stream>>>(src.size(1),
                                            src.size(2),
                                            src.size(3),
                                            src.packed_accessor32<VAL_T, 4, at::RestrictPtrTraits>(),
                                            dst.packed_accessor32<VAL_T, 4, at::RestrictPtrTraits>());
}

template<int BDIM_X,
int BDIM_Y,
typename VAL_T>
__global__
__launch_bounds__(BDIM_X*BDIM_Y)
void  permute_to0312_k(const int nchn,
              const int nlat,
              const int nlon,
              const at::PackedTensorAccessor32<VAL_T, 4, at::RestrictPtrTraits> src,
                    at::PackedTensorAccessor32<VAL_T, 4, at::RestrictPtrTraits> dst) {

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
       sh[j+tidy][tidx] = (woff + j+tidy < nlon) ? src[batch][h][woff + j+tidy][coff+tidx] : VAL_T(0);
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

template<int WARPS_X_TILE, typename VAL_T>
void launch_permute_to0312(torch::Tensor src, torch::Tensor dst){
dim3 block;
dim3 grid;

block.x = WARP_SIZE;
block.y = WARPS_X_TILE;
grid.x = DIV_UP(src.size(2), block.x);
grid.y = DIV_UP(src.size(3), block.x);
grid.z = src.size(1)*src.size(0);

assert(grid.y < 65536);
assert(grid.z < 65536);

// get stream
auto stream = at::cuda::getCurrentCUDAStream().stream();

permute_to0312_k<WARP_SIZE, WARPS_X_TILE>
               <<<grid, block, 0, stream>>>(src.size(3),
                                            src.size(1),
                                            src.size(2),
                                            src.packed_accessor32<VAL_T, 4, at::RestrictPtrTraits>(),
                                            dst.packed_accessor32<VAL_T, 4, at::RestrictPtrTraits>());
}

torch::Tensor permute_4D_to0312(torch::Tensor src);
torch::Tensor permute_4D_to0231(torch::Tensor src);

}