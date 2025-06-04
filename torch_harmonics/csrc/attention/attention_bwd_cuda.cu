// coding=utf-8
//
// SPDX-FileCopyrightText: Copyright (c) 2024 The torch-harmonics Authors. All rights reserved.
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
#include "c10/core/MemoryFormat.h"

#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/CUDAUtils.h>

#include <ctime>
#include <cub/cub.cuh>
#include <limits>

#ifndef WARP_SIZE
#define WARP_SIZE (32)
#endif
#ifndef FULL_MASK
#define FULL_MASK (0xFFFFFFFF)
#endif
#ifndef THREADS
#define THREADS (64)
#endif
#ifndef DIV_UP
#define DIV_UP(a,b) (((a)+((b)-1))/(b))
#endif
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) {                                            \
    cudaError_t err = call;                                           \
    if( cudaSuccess != err) {                                         \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\\n",  \
              __FILE__, __LINE__, cudaGetErrorString( err) );         \
      exit(EXIT_FAILURE);                                             \
    }}
#endif

#include <iostream>
#include <chrono>
#include <string>

class ScopeTimer {
public:
  explicit ScopeTimer(const std::string& label = "")
    : label_(label), start_(std::chrono::high_resolution_clock::now()) {}

  ~ScopeTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
    std::cout << label_ << "Elapsed time: " << elapsed.count() << " ms" << std::endl;
  }

private:
  std::string label_;
  std::chrono::high_resolution_clock::time_point start_;
};

__device__ static float atomicMax(float* address, float val)
{
  int* address_as_i = (int*) address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(address_as_i, assumed,
                      __float_as_int(::fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

static __device__ float __warp_sum(float val) {
#pragma unroll
  for(int i = WARP_SIZE/2; i; i /= 2) {
    val += __shfl_xor_sync(FULL_MASK, val, i);
  }
  return val;

}

// easier to understand version of manual shfl_xor_sync, performance appears similar
__device__ float __warp_sum_cub(float val) {
  // use cub to reduce within a warp
  __shared__ typename cub::WarpReduce<float>::TempStorage temp_storage;
  
  // 1. Compute sum (initially only in lane 0)
  float sum = cub::WarpReduce<float>(temp_storage).Sum(val);
  // 2. Broadcast sum to all threads
  sum = __shfl_sync(0xFFFFFFFF, sum, 0);
  return sum;
}

__global__ void
s2_attention_bwd_dv_kernel(int num_channels, int nlon_in, int nlat_out, int nlon_out,
                           const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> kx,
                           const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> vx,
                           const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> qy,
                           const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> dy,
                           torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> dydv,
                           const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> psi_col_idx,
                           const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> psi_row_offset,
                           const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> quad_weights)
{
  // shared memory
  extern __shared__ float sharedMem[];

  float* sh_alpha_sum = (float*)&sharedMem; // 1
  float* sh_qdotk_max = (float*)&sharedMem[1]; // 1
  float* sh_qy_ho_wo = (float*)&sharedMem[2]; // num_channels

  if (threadIdx.x == 0) {
    sh_qdotk_max[0] = std::numeric_limits<float>::lowest();
    sh_alpha_sum[0] = 0.0;
  }
  __syncthreads();

  int ho = blockIdx.x;
  int wo = blockIdx.y;
  int batch_b = blockIdx.z;

  // load qy channels into shared memory
  for(int channel_block_i = 0; channel_block_i<(num_channels/blockDim.x)+1; channel_block_i++) {
    int channel_idx = channel_block_i*blockDim.x + threadIdx.x;
    if(channel_idx >= num_channels) break;
    sh_qy_ho_wo[channel_idx] = qy[batch_b][channel_idx][ho][wo];
  }
  __syncthreads();

  int psi_offset = psi_row_offset[ho];
  int psi_nnz_ho = psi_row_offset[ho + 1] - psi_offset;
  float qdotk_max = std::numeric_limits<float>::lowest();
  for(int psi_block=0; psi_block<(psi_nnz_ho/blockDim.x)+1; psi_block++) {
    int idz = psi_block*blockDim.x + threadIdx.x;

    // skip if index >= length of psi_idx because last loop iteration will have extra threads
    if(idz >= psi_nnz_ho) break;

    int nz_col_idx = psi_col_idx[psi_offset+idz];

    // compute input indices from psi datastructure
    int hi = nz_col_idx / nlon_in;
    // account for output shift and ensure positive index due to circular condition
    // int wi = (nz_col_idx % nlon_in - wo) % nlon_in;
    int wi = nz_col_idx % nlon_in;
    int wip = (wi + wo) % nlon_in;

    // correlation Q&K (dot-product Q.K)
    float qdotk = 0.0;
    for(int channel_idx = 0; channel_idx<num_channels; channel_idx++) {
      qdotk += sh_qy_ho_wo[channel_idx]*kx[batch_b][channel_idx][hi][wip];
    }
    qdotk_max = std::max(qdotk, qdotk_max);
  }

  // collect thread-local qdotk max
  atomicMax(&sh_qdotk_max[0], qdotk_max);
  __syncthreads();

  // "broadcast" qdotk_max back into all thread-local registers
  qdotk_max = sh_qdotk_max[0];

  // form alpha & sum alpha
  float alpha_sum = 0.0;
  for(int psi_block=0; psi_block<(psi_nnz_ho/blockDim.x)+1; psi_block++) {
    int idz = psi_block*blockDim.x + threadIdx.x;

    // skip if index >= length of psi_idx because last loop iteration will have extra threads
    if(idz >= psi_nnz_ho) break;

    int nz_col_idx = psi_col_idx[psi_offset+idz];

    // compute input indices from psi datastructure
    int hi = nz_col_idx / nlon_in;
    // account for output shift and ensure positive index due to circular condition
    // int wi = (nz_col_idx % nlon_in - wo) % nlon_in;
    int wi = nz_col_idx % nlon_in;
    int wip = (wi + wo) % nlon_in;

    // softmax numerator
    float qdotk = 0.0;
    for(int channel_idx = 0; channel_idx<num_channels; channel_idx++) {
      qdotk += sh_qy_ho_wo[channel_idx]*kx[batch_b][channel_idx][hi][wip];
    }
    float alpha_inz = expf(qdotk - qdotk_max) * quad_weights[hi];
    // sum alpha
    alpha_sum += alpha_inz;
  }
  // collect thread-local alpha_sum
  atomicAdd(&sh_alpha_sum[0], alpha_sum);
  __syncthreads();

  // "broadcast" alpha sum back to thread-local registers
  alpha_sum = sh_alpha_sum[0];

  // alpha * dy * omega / alpha_sum
  for(int psi_block=0; psi_block<(psi_nnz_ho/blockDim.x)+1; psi_block++) {
    int idz = psi_block*blockDim.x + threadIdx.x;

    // skip if index >= length of psi_idx because last loop iteration will have extra threads
    if(idz >= psi_nnz_ho) break;

    int nz_col_idx = psi_col_idx[psi_offset+idz];

    // compute input indices from psi datastructure
    int hi = nz_col_idx / nlon_in;
    // account for output shift and ensure positive index due to circular condition
    int wi = nz_col_idx % nlon_in;
    int wip = (wi + wo) % nlon_in;

    float qdotk = 0.0;
    for(int channel_idx = 0; channel_idx<num_channels; channel_idx++) {
      qdotk += sh_qy_ho_wo[channel_idx]*kx[batch_b][channel_idx][hi][wip];
    }
    float alpha_inz = expf(qdotk - qdotk_max) * quad_weights[hi];

    // multiply alpha/sum_alpha, dy, and quadrature weights
    for(int channel_idx = 0; channel_idx<num_channels; channel_idx++) {
      atomicAdd(&dydv[batch_b][channel_idx][hi][wip], (alpha_inz/alpha_sum) * dy[batch_b][channel_idx][ho][wo]);
    }

  }
}

at::Tensor s2_attention_bwd_dv_cuda(at::Tensor kx,
                                    at::Tensor vx,
                                    at::Tensor qy,
                                    at::Tensor dy,
                                    at::Tensor quad_weights,
                                    at::Tensor psi_col_idx,
                                    at::Tensor psi_row_off,
                                    int nlon_in, int nlat_out, int nlon_out) {

  CHECK_CUDA_TENSOR(kx);
  CHECK_CUDA_TENSOR(vx);
  CHECK_CUDA_TENSOR(qy);
  CHECK_CUDA_TENSOR(quad_weights);
  CHECK_CUDA_TENSOR(psi_col_idx);
  CHECK_CUDA_TENSOR(psi_row_off);
  CHECK_CUDA_TENSOR(dy);

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  torch::Tensor dydv = torch::zeros_like(vx);

  size_t uo_num_channels = kx.size(1);

  size_t sharedMemSize = (uo_num_channels+2)*sizeof(float);

  const int batch_size = kx.size(0);

  // cuda grid y,z size limitations
  assert(nlon_out < 65535);
  assert(batch_size < 65535);

  // block-parallel over output points and batches
  dim3 gridDim(nlat_out,nlon_out,batch_size);

  // threads compute "blocks" of neighborhood and also "blocks" of channels
  dim3 blockDim(256, 1, 1);

  s2_attention_bwd_dv_kernel <<<gridDim, blockDim, sharedMemSize, stream>>>(
                                                                            uo_num_channels, nlon_in, nlat_out, nlon_out,
                                                                            kx.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                                            vx.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                                            qy.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                                            dy.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                                            dydv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                                            psi_col_idx.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                                                                            psi_row_off.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                                                                            quad_weights.packed_accessor32<float, 1, torch::RestrictPtrTraits>()
                                                                            );


  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return dydv;
}

__global__ void
s2_attention_bwd_dk_kernel(int num_channels, int nlon_in, int nlat_out, int nlon_out,
                           const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> kx,
                           const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> vx,
                           const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> qy,
                           const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> dy,
                           torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> dydk,
                           const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> psi_col_idx,
                           const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> psi_row_offset,
                           const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> quad_weights)
{
  // shared memory
  extern __shared__ float sharedMem[];

  float* sh_alpha_sum = (float*)&sharedMem;
  float *sh_qy_ho_wo = (float *)&sharedMem[1];
  float *sh_integral = (float *)&sharedMem[1 + num_channels];
  float *sh_dy_ho_wo = (float *)&sharedMem[2 + num_channels];
  float *sh_qdotk_max = (float *)&sharedMem[2 + 2 * num_channels];

  if (threadIdx.x == 0) {
    sh_alpha_sum[0] = 0.0;
    sh_integral[0] = 0.0;
    sh_qdotk_max[0] = std::numeric_limits<float>::lowest();
  }
  __syncthreads();

  int ho = blockIdx.x;
  int wo = blockIdx.y;
  int batch_b = blockIdx.z;

  // load qy channels into shared memory
  for(int channel_block_i = 0; channel_block_i<(num_channels/blockDim.x)+1; channel_block_i++) {
    int channel_idx = channel_block_i*blockDim.x + threadIdx.x;
    if(channel_idx >= num_channels) break;
    sh_qy_ho_wo[channel_idx] = qy[batch_b][channel_idx][ho][wo];
    sh_dy_ho_wo[channel_idx] = dy[batch_b][channel_idx][ho][wo];
  }
  __syncthreads();

  int psi_offset = psi_row_offset[ho];
  int psi_nnz_ho = psi_row_offset[ho + 1] - psi_offset;
  float qdotk_max = std::numeric_limits<float>::lowest();
  for(int psi_block=0; psi_block<(psi_nnz_ho/blockDim.x)+1; psi_block++) {
    int idz = psi_block*blockDim.x + threadIdx.x;
    // skip if index >= length of psi_idx because last loop iteration will have extra threads
    if(idz >= psi_nnz_ho) break;

    int nz_col_idx = psi_col_idx[psi_offset+idz];

    // compute input indices from psi datastructure
    int hi = nz_col_idx / nlon_in;
    // account for output shift and ensure positive index due to circular condition
    // int wi = (nz_col_idx % nlon_in - wo) % nlon_in;
    int wi = nz_col_idx % nlon_in;
    int wip = (wi + wo) % nlon_in;

    // correlation Q&K (dot-product Q.K)
    float qdotk = 0.0;
    for(int channel_idx = 0; channel_idx<num_channels; channel_idx++) {
      qdotk += sh_qy_ho_wo[channel_idx]*kx[batch_b][channel_idx][hi][wip];
    }
    qdotk_max = max(qdotk_max, qdotk);
  }

  // compute max over all threads
  atomicMax(&sh_qdotk_max[0], qdotk_max);
  __syncthreads();
  // "broadcast" qdotk_max back into all thread-local registers
  qdotk_max = sh_qdotk_max[0];

  float alpha_sum = 0.0;
  float integral = 0.0;
  for(int psi_block=0; psi_block<(psi_nnz_ho/blockDim.x)+1; psi_block++) {
    int idz = psi_block*blockDim.x + threadIdx.x;

    // skip if index >= length of psi_idx because last loop iteration will have extra threads
    if(idz >= psi_nnz_ho) break;

    int nz_col_idx = psi_col_idx[psi_offset+idz];

    // compute input indices from psi datastructure
    int hi = nz_col_idx / nlon_in;
    // account for output shift and ensure positive index due to circular condition
    // int wi = (nz_col_idx % nlon_in - wo) % nlon_in;
    int wi = nz_col_idx % nlon_in;
    int wip = (wi + wo) % nlon_in;

    // correlation Q&K (dot-product Q.K)
    float gdotv = 0.0;
    float qdotk = 0.0;
    for(int channel_idx = 0; channel_idx<num_channels; channel_idx++) {
      gdotv += sh_dy_ho_wo[channel_idx] * vx[batch_b][channel_idx][hi][wip];
      qdotk += sh_qy_ho_wo[channel_idx]*kx[batch_b][channel_idx][hi][wip];
    }
    // softmax numerator
    float alpha_inz = expf(qdotk - qdotk_max) * quad_weights[hi];

    // sum alpha & integral
    alpha_sum += alpha_inz;
    integral += alpha_inz * gdotv;
  }

  // block sum thread-local alpha_sum and integral
  atomicAdd(&sh_alpha_sum[0], alpha_sum);
  atomicAdd(&sh_integral[0], integral);
  __syncthreads();
  // finish integral computation
  if(threadIdx.x==0) sh_integral[0] /= sh_alpha_sum[0];
  __syncthreads();
  // broadcast sum and integral back to thread-local registers
  integral = sh_integral[0];
  alpha_sum = sh_alpha_sum[0];

  // divide output by alpha_sum
  for(int psi_block=0; psi_block<(psi_nnz_ho/blockDim.x)+1; psi_block++) {
    int idz = psi_block*blockDim.x + threadIdx.x;
    // skip if index >= length of psi_idx because last loop iteration will have extra threads
    if(idz >= psi_nnz_ho) break;

    int nz_col_idx = psi_col_idx[psi_offset+idz];

    // compute input indices from psi datastructure
    int hi = nz_col_idx / nlon_in;
    // account for output shift and ensure positive index due to circular condition
    // int wi = (nz_col_idx % nlon_in - wo) % nlon_in;
    int wi = nz_col_idx % nlon_in;
    int wip = (wi + wo) % nlon_in;
    float gdotv = 0.0;
    float qdotk = 0.0;
    for(int channel_idx = 0; channel_idx<num_channels; channel_idx++) {
      gdotv += sh_dy_ho_wo[channel_idx] * vx[batch_b][channel_idx][hi][wip];
      qdotk += sh_qy_ho_wo[channel_idx]*kx[batch_b][channel_idx][hi][wip];
    }
    float alpha_inz = expf(qdotk - qdotk_max) * quad_weights[hi];
    // multiply alpha/sum_alpha, vx, and quadrature weights
    for(int channel_idx = 0; channel_idx<num_channels; channel_idx++) {
      atomicAdd(&dydk[batch_b][channel_idx][hi][wip],
                sh_qy_ho_wo[channel_idx] * (alpha_inz/alpha_sum) * (gdotv - integral));
    }
  }
  __syncthreads();
}

__global__ void
s2_attention_bwd_dq_kernel(int num_channels, int nlon_in, int nlat_out, int nlon_out,
                           const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> kx,
                           const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> vx,
                           const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> qy,
                           const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> dy,
                           torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> dydq,
                           const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> psi_col_idx,
                           const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> psi_row_offset,
                           const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> quad_weights)
{
  // shared memory
  extern __shared__ float sharedMem[];

  float* sh_alpha_sum = (float*)&sharedMem;
  float *sh_qy_ho_wo = (float *)&sharedMem[1];
  float *sh_alpha_k = (float *)&sharedMem[1 + num_channels];
  float *sh_alpha_vw = (float *)&sharedMem[1 + 2*num_channels];
  float *sh_alpha_kvw = (float *)&sharedMem[1 + 3*num_channels];
  float *sh_dy_ho_wo = (float *)&sharedMem[1 + 4 * num_channels];
  float *sh_qdotk_max = (float *)&sharedMem[1 +  5 * num_channels];

  if (threadIdx.x == 0) {
    sh_alpha_sum[0] = 0.0;
    sh_qdotk_max[0] = std::numeric_limits<float>::lowest();
  }
  __syncthreads();

  int ho = blockIdx.x;
  int wo = blockIdx.y;
  int batch_b = blockIdx.z;

  // load qy channels into shared memory and zero temporary variables
  for(int channel_block_i = 0; channel_block_i<(num_channels/blockDim.x)+1; channel_block_i++) {
    int channel_idx = channel_block_i*blockDim.x + threadIdx.x;
    if(channel_idx >= num_channels) break;
    sh_qy_ho_wo[channel_idx] = qy[batch_b][channel_idx][ho][wo];
    sh_dy_ho_wo[channel_idx] = dy[batch_b][channel_idx][ho][wo];
    sh_alpha_k[channel_idx] = 0.0f;
    sh_alpha_vw[channel_idx] = 0.0f;
    sh_alpha_kvw[channel_idx] = 0.0f;
  }
  __syncthreads();

  int psi_offset = psi_row_offset[ho];
  int psi_nnz_ho = psi_row_offset[ho + 1] - psi_offset;
  float qdotk_max = std::numeric_limits<float>::lowest();
  for(int psi_block=0; psi_block<(psi_nnz_ho/blockDim.x)+1; psi_block++) {
    int idz = psi_block*blockDim.x + threadIdx.x;

    // skip if index >= length of psi_idx because last loop iteration will have extra threads
    if(idz >= psi_nnz_ho) break;

    int nz_col_idx = psi_col_idx[psi_offset+idz];

    // compute input indices from psi datastructure
    int hi = nz_col_idx / nlon_in;
    // account for output shift and ensure positive index due to circular condition
    // int wi = (nz_col_idx % nlon_in - wo) % nlon_in;
    int wi = nz_col_idx % nlon_in;
    int wip = (wi + wo) % nlon_in;
    // correlation Q&K (dot-product Q.K)
    float qdotk = 0.0f;
    for(int channel_idx = 0; channel_idx<num_channels; channel_idx++) {
      qdotk += sh_qy_ho_wo[channel_idx] * kx[batch_b][channel_idx][hi][wip];
    }
    qdotk_max = std::max(qdotk, qdotk_max);
  }
  atomicMax(&sh_qdotk_max[0], qdotk_max);
  __syncthreads();

  // "broadcast" qdotk_max back into all thread-local registers
  qdotk_max = sh_qdotk_max[0];
  float alpha_sum = 0.0;
  for(int psi_block=0; psi_block<(psi_nnz_ho/blockDim.x)+1; psi_block++) {
    int idz = psi_block*blockDim.x + threadIdx.x;
    // skip if index >= length of psi_idx because last loop iteration will have extra threads
    if(idz >= psi_nnz_ho) break;

    int nz_col_idx = psi_col_idx[psi_offset+idz];

    // compute input indices from psi datastructure
    int hi = nz_col_idx / nlon_in;
    // account for output shift and ensure positive index due to circular condition
    // int wi = (nz_col_idx % nlon_in - wo) % nlon_in;
    int wi = nz_col_idx % nlon_in;
    int wip = (wi + wo) % nlon_in;
    // correlation Q&K (dot-product Q.K)
    float qdotk = 0.0f;
    float gdotv = 0.0f;
    for(int channel_idx = 0; channel_idx<num_channels; channel_idx++) {
      gdotv += sh_dy_ho_wo[channel_idx] * vx[batch_b][channel_idx][hi][wip];
      qdotk += sh_qy_ho_wo[channel_idx] * kx[batch_b][channel_idx][hi][wip];
    }
    // softmax numerator
    float alpha_inz = expf(qdotk - qdotk_max) * quad_weights[hi];
    // sum alpha
    alpha_sum += alpha_inz;
    for(int channel_idx = 0; channel_idx<num_channels; channel_idx++) {
      atomicAdd(&sh_alpha_k[channel_idx],
                alpha_inz * kx[batch_b][channel_idx][hi][wip]);
      atomicAdd(&sh_alpha_vw[channel_idx],
                alpha_inz * gdotv);
      atomicAdd(&sh_alpha_kvw[channel_idx],
                alpha_inz * kx[batch_b][channel_idx][hi][wip] * gdotv);
    }
  }
  // sum thread-local alpha_sums across block
  atomicAdd(&sh_alpha_sum[0], alpha_sum);
  __syncthreads();
  // "broadcast" alpha sum back to thread-local registers
  alpha_sum = sh_alpha_sum[0];
  for(int channel_block_i = 0; channel_block_i<(num_channels/blockDim.x)+1; channel_block_i++) {
    int channel_idx = channel_block_i*blockDim.x + threadIdx.x;
    if (channel_idx >= num_channels)
      break;

    dydq[batch_b][channel_idx][ho][wo] = (sh_alpha_kvw[channel_idx]*sh_alpha_sum[0] - sh_alpha_vw[channel_idx]*sh_alpha_k[channel_idx])/(alpha_sum*alpha_sum);
  }

}

__global__ void s2_attention_bwd_dkvq_kernel(int num_channels, int nlon_in, int nlat_out, int nlon_out,
                                             const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> kx,
                                             const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> vx,
                                             const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> qy,
                                             const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits>
                                             dy,
                                             torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> dydk,
                                             torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> dydv,
                                             torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> dydq,
                                             const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> psi_col_idx,
                                             const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> psi_row_offset,
                                             const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> quad_weights)
{
  // shared memory
  extern __shared__ float sharedMem[];

  float *sh_alpha_sum = (float *)&sharedMem;
  float* sh_integral = (float*)&sharedMem[1];
  float *sh_qy_ho_wo = (float *)&sharedMem[2];
  float *sh_alpha_k = (float *)&sharedMem[2 + num_channels];
  float *sh_alpha_vw = (float *)&sharedMem[2 + 2*num_channels];
  float *sh_alpha_kvw = (float *)&sharedMem[2 + 3*num_channels];
  float *sh_dy_ho_wo = (float *)&sharedMem[2 + 4 * num_channels];
  float *sh_qdotk_max = (float *)&sharedMem[2 +  5 * num_channels];

  if (threadIdx.x == 0) {
    sh_alpha_sum[0] = 0.0;
    sh_integral[0] = 0.0;
    sh_qdotk_max[0] = std::numeric_limits<float>::lowest();
  }
  __syncthreads();

  int ho = blockIdx.x;
  int wo = blockIdx.y;
  int batch_b = blockIdx.z;

  // load qy channels into shared memory and zero temporary variables
  for(int channel_block_i = 0; channel_block_i<(num_channels/blockDim.x)+1; channel_block_i++) {
    int channel_idx = channel_block_i*blockDim.x + threadIdx.x;
    if(channel_idx >= num_channels) break;
    sh_qy_ho_wo[channel_idx] = qy[batch_b][channel_idx][ho][wo];
    sh_dy_ho_wo[channel_idx] = dy[batch_b][channel_idx][ho][wo];
    sh_alpha_k[channel_idx] = 0.0f;
    sh_alpha_vw[channel_idx] = 0.0f;
    sh_alpha_kvw[channel_idx] = 0.0f;
  }
  __syncthreads();

  int psi_offset = psi_row_offset[ho];
  int psi_nnz_ho = psi_row_offset[ho + 1] - psi_offset;
  float qdotk_max = std::numeric_limits<float>::lowest();
  for(int psi_block=0; psi_block<(psi_nnz_ho/blockDim.x)+1; psi_block++) {
    int idz = psi_block*blockDim.x + threadIdx.x;

    // skip if index >= length of psi_idx because last loop iteration will have extra threads
    if(idz >= psi_nnz_ho) break;

    int nz_col_idx = psi_col_idx[psi_offset+idz];

    // compute input indices from psi datastructure
    int hi = nz_col_idx / nlon_in;
    // account for output shift and ensure positive index due to circular condition
    // int wi = (nz_col_idx % nlon_in - wo) % nlon_in;
    int wi = nz_col_idx % nlon_in;
    int wip = (wi + wo) % nlon_in;
    // correlation Q&K (dot-product Q.K)
    float qdotk = 0.0f;
    for(int channel_idx = 0; channel_idx<num_channels; channel_idx++) {
      qdotk += sh_qy_ho_wo[channel_idx] * kx[batch_b][channel_idx][hi][wip];
    }
    qdotk_max = std::max(qdotk, qdotk_max);
  }
  atomicMax(&sh_qdotk_max[0], qdotk_max);
  __syncthreads();

  // "broadcast" qdotk_max back into all thread-local registers
  qdotk_max = sh_qdotk_max[0];
  float alpha_sum = 0.0;
  float integral = 0.0;
  for(int psi_block=0; psi_block<(psi_nnz_ho/blockDim.x)+1; psi_block++) {
    int idz = psi_block*blockDim.x + threadIdx.x;
    // skip if index >= length of psi_idx because last loop iteration will have extra threads
    if(idz >= psi_nnz_ho) break;

    int nz_col_idx = psi_col_idx[psi_offset+idz];

    // compute input indices from psi datastructure
    int hi = nz_col_idx / nlon_in;
    // account for output shift and ensure positive index due to circular condition
    // int wi = (nz_col_idx % nlon_in - wo) % nlon_in;
    int wi = nz_col_idx % nlon_in;
    int wip = (wi + wo) % nlon_in;
    // correlation Q&K (dot-product Q.K)
    float qdotk = 0.0f;
    float gdotv = 0.0f;
    for(int channel_idx = 0; channel_idx<num_channels; channel_idx++) {
      gdotv += sh_dy_ho_wo[channel_idx] * vx[batch_b][channel_idx][hi][wip];
      qdotk += sh_qy_ho_wo[channel_idx] * kx[batch_b][channel_idx][hi][wip];
    }
    // softmax numerator
    float alpha_inz = expf(qdotk - qdotk_max) * quad_weights[hi];
    // sum alpha
    alpha_sum += alpha_inz;
    for(int channel_idx = 0; channel_idx<num_channels; channel_idx++) {
      atomicAdd(&sh_alpha_k[channel_idx],
                alpha_inz * kx[batch_b][channel_idx][hi][wip]);
      atomicAdd(&sh_alpha_vw[channel_idx],
                alpha_inz * gdotv);
      atomicAdd(&sh_alpha_kvw[channel_idx],
                alpha_inz * kx[batch_b][channel_idx][hi][wip] * gdotv);
    }

    integral += alpha_inz * gdotv;
  }
  // sum thread-local alpha_sums & integral across block
  atomicAdd(&sh_alpha_sum[0], alpha_sum);
  atomicAdd(&sh_integral[0], integral);
  __syncthreads();

  // finalize integral
  if(threadIdx.x==0) sh_integral[0] /= sh_alpha_sum[0];
  __syncthreads();
  // "broadcast" alpha sum & integral back to thread-local registers
  alpha_sum = sh_alpha_sum[0];
  integral = sh_integral[0];

  // dq
  for(int channel_block_i = 0; channel_block_i<(num_channels/blockDim.x)+1; channel_block_i++) {
    int channel_idx = channel_block_i*blockDim.x + threadIdx.x;
    if (channel_idx >= num_channels)
      break;

    dydq[batch_b][channel_idx][ho][wo] = (sh_alpha_kvw[channel_idx]*sh_alpha_sum[0] - sh_alpha_vw[channel_idx]*sh_alpha_k[channel_idx])/(alpha_sum*alpha_sum);
  }
  __syncthreads();
  // dk & dv
  for(int psi_block=0; psi_block<(psi_nnz_ho/blockDim.x)+1; psi_block++) {
    int idz = psi_block*blockDim.x + threadIdx.x;
    // skip if index >= length of psi_idx because last loop iteration will have extra threads
    if(idz >= psi_nnz_ho) break;

    int nz_col_idx = psi_col_idx[psi_offset+idz];

    // compute input indices from psi datastructure
    int hi = nz_col_idx / nlon_in;
    // account for output shift and ensure positive index due to circular condition
    // int wi = (nz_col_idx % nlon_in - wo) % nlon_in;
    int wi = nz_col_idx % nlon_in;
    int wip = (wi + wo) % nlon_in;
    float gdotv = 0.0;
    float qdotk = 0.0;
    for(int channel_idx = 0; channel_idx<num_channels; channel_idx++) {
      gdotv += sh_dy_ho_wo[channel_idx] * vx[batch_b][channel_idx][hi][wip];
      qdotk += sh_qy_ho_wo[channel_idx]*kx[batch_b][channel_idx][hi][wip];
    }
    float alpha_inz = expf(qdotk - qdotk_max) * quad_weights[hi];
    // multiply alpha/sum_alpha, vx, and quadrature weights
    for(int channel_idx = 0; channel_idx<num_channels; channel_idx++) {
      atomicAdd(&dydk[batch_b][channel_idx][hi][wip],
                sh_qy_ho_wo[channel_idx] * (alpha_inz / alpha_sum) *
                (gdotv - integral));
      atomicAdd(&dydv[batch_b][channel_idx][hi][wip],
                (alpha_inz / alpha_sum) * sh_dy_ho_wo[channel_idx]);

    }
  }
  __syncthreads();

}

// New kernel: s2_attention_bwd_dkvq_kernel_mbT
// This kernel assumes kx, vx, qy, dy, dydk, dydv, dydq are all [batch, ho, wo, channel] (transposed)
template<int BDIM_X>
__global__
__launch_bounds__(BDIM_X)
  void s2_attention_bwd_dkvq_kernel_mbT(
                                        int num_channels,
                                        int nlon_in,
                                        int nlat_out,
                                        int nlon_out,
                                        const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> kx,
                                        const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> vx,
                                        const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> qy,
                                        const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> dy,
                                        torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> dydk,
                                        torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> dydv,
                                        torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> dydq,
                                        const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> psi_col_idx,
                                        const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> psi_row_offset,
                                        const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> quad_weights) {

  extern __shared__ float sh[];
  float* sh_alpha_k = sh + threadIdx.y * num_channels * 5;
  float* sh_alpha_vw = sh_alpha_k + num_channels;
  float* sh_alpha_kvw = sh_alpha_vw + num_channels;
  float *sh_dy = sh_alpha_kvw + num_channels;
  float* sh_qy = sh_dy + num_channels;
  // (optionally, could use more shared memory for other intermediates)

  const uint64_t batchId = blockIdx.y;
  const uint64_t wid = uint64_t(blockIdx.x) * blockDim.y + threadIdx.y;
  if (wid >= uint64_t(nlat_out) * nlon_in) return;
  const int tidx = threadIdx.x;
  const int ho = wid / nlon_out;
  const int wo = wid - (ho * nlon_out);

  // Zero shared memory
  for (int chan = tidx; chan < num_channels; chan += WARP_SIZE) {
    sh_alpha_k[chan] = 0.0f;
    sh_alpha_vw[chan] = 0.0f;
    sh_alpha_kvw[chan] = 0.0f;
    sh_dy[chan] = dy[batchId][chan][ho][wo];
    sh_qy[chan] = qy[batchId][chan][ho][wo];
  }
  float alpha_sum = 0.0f;
  float qdotk_max = -FLT_MAX;
  float integral = 0.0f;
  __syncthreads();

  const int64_t rbeg = psi_row_offset[ho];
  const int64_t rend = psi_row_offset[ho+1];
  const int rlen = rend - rbeg;

  // First pass: find qdotk_max
  for (int off = 0; off < rlen; off++) {
    const int64_t col = psi_col_idx[rbeg + off];
    const int hi = col / nlon_in;
    const int wi = col - (hi * nlon_in);
    const int wip = (wi + wo) - ((wi + wo) / nlon_in) * nlon_in;
    float qdotk = 0.0f;
    for (int chan = tidx; chan < num_channels; chan += WARP_SIZE) {
      qdotk += sh_qy[chan] * kx[batchId][chan][hi][wip];
    }
    qdotk = __warp_sum_cub(qdotk);
    qdotk_max = max(qdotk_max, qdotk);
  }

  // Second pass: accumulate alpha_sum, integral, and shared stats
  for (int off = 0; off < rlen; off++) {
    const int64_t col = psi_col_idx[rbeg + off];
    const int hi = col / nlon_in;
    const int wi = col - (hi * nlon_in);
    const int wip = (wi + wo) - ((wi + wo) / nlon_in) * nlon_in;
    float qdotk = 0.0f, gdotv = 0.0f;
    for (int chan = tidx; chan < num_channels; chan += WARP_SIZE) {
      qdotk += sh_qy[chan] * kx[batchId][chan][hi][wip];
      gdotv += sh_dy[chan] * vx[batchId][chan][hi][wip];
    }
    qdotk = __warp_sum_cub(qdotk);
    gdotv = __warp_sum_cub(gdotv);
    float alpha_inz = expf(qdotk - qdotk_max) * quad_weights[hi];
    alpha_sum += alpha_inz;
    integral += alpha_inz * gdotv;
    for (int chan = tidx; chan < num_channels; chan += WARP_SIZE) {
      float kxval = kx[batchId][chan][hi][wip];
      sh_alpha_k[chan] += alpha_inz * kxval;
      sh_alpha_vw[chan] += alpha_inz * gdotv;
      sh_alpha_kvw[chan] += alpha_inz * kxval * gdotv;
    }
  }

  integral /= alpha_sum;

  // Write dydq
  for (int chan = tidx; chan < num_channels; chan += WARP_SIZE) {
    dydq[batchId][chan][ho][wo] = (sh_alpha_kvw[chan] * alpha_sum - sh_alpha_vw[chan] * sh_alpha_k[chan]) / (alpha_sum * alpha_sum);
  }

  // Third pass: accumulate gradients for k and v
  for (int off = 0; off < rlen; off++) {
    const int64_t col = psi_col_idx[rbeg + off];
    const int hi = col / nlon_in;
    const int wi = col - (hi * nlon_in);
    const int wip = (wi + wo) - ((wi + wo) / nlon_in) * nlon_in;
    float qdotk = 0.0f, gdotv = 0.0f;
    for (int chan = tidx; chan < num_channels; chan += WARP_SIZE) {
      qdotk += qy[batchId][chan][ho][wo] * kx[batchId][chan][hi][wip];
      gdotv += sh_dy[chan] * vx[batchId][chan][hi][wip];
    }
    qdotk = __warp_sum_cub(qdotk);
    gdotv = __warp_sum_cub(gdotv);
    float alpha_inz = expf(qdotk - qdotk_max) * quad_weights[hi];
    for (int chan = tidx; chan < num_channels; chan += WARP_SIZE) {
      float qyval = qy[batchId][chan][ho][wo];
      float dyval = sh_dy[chan];
      atomicAdd(&dydk[batchId][chan][hi][wip], qyval * (alpha_inz / alpha_sum) * (gdotv - integral));
      atomicAdd(&dydv[batchId][chan][hi][wip], (alpha_inz / alpha_sum) * dyval);
    }
  }
}

at::Tensor s2_attention_bwd_dk_cuda(at::Tensor kx, 
                                    at::Tensor vx,
                                    at::Tensor qy,
                                    at::Tensor dy,
                                    at::Tensor quad_weights,
                                    at::Tensor psi_col_idx,
                                    at::Tensor psi_row_off,
                                    int nlon_in, int nlat_out, int nlon_out) {

  CHECK_CUDA_TENSOR(kx);
  CHECK_CUDA_TENSOR(vx);
  CHECK_CUDA_TENSOR(qy);
  CHECK_CUDA_TENSOR(quad_weights);
  CHECK_CUDA_TENSOR(psi_col_idx);
  CHECK_CUDA_TENSOR(psi_row_off);
  CHECK_CUDA_TENSOR(dy);

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  torch::Tensor dydk = torch::zeros_like(kx);

  size_t uo_num_channels = kx.size(1);

  size_t sharedMemSize = (2*uo_num_channels+3)*sizeof(float);

  const int batch_size = kx.size(0);

  // cuda grid y,z size limitations
  assert(nlon_out < 65535);
  assert(batch_size < 65535);

  // block-parallel over output points and batches
  dim3 gridDim(nlat_out,nlon_out,batch_size);

  // threads compute "blocks" of neighborhood and also "blocks" of channels
  dim3 blockDim(256, 1, 1);

  s2_attention_bwd_dk_kernel <<<gridDim, blockDim, sharedMemSize, stream>>>(
                                                                            uo_num_channels, nlon_in, nlat_out, nlon_out,
                                                                            kx.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                                            vx.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                                            qy.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                                            dy.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                                            dydk.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                                            psi_col_idx.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                                                                            psi_row_off.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                                                                            quad_weights.packed_accessor32<float, 1, torch::RestrictPtrTraits>()
                                                                            );


  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return dydk;
}


at::Tensor s2_attention_bwd_dq_cuda(at::Tensor kx, 
                                    at::Tensor vx,
                                    at::Tensor qy,
                                    at::Tensor dy,
                                    at::Tensor quad_weights,
                                    at::Tensor psi_col_idx,
                                    at::Tensor psi_row_off,
                                    int nlon_in, int nlat_out, int nlon_out) {

  CHECK_CUDA_TENSOR(kx);
  CHECK_CUDA_TENSOR(vx);
  CHECK_CUDA_TENSOR(qy);
  CHECK_CUDA_TENSOR(quad_weights);
  CHECK_CUDA_TENSOR(psi_col_idx);
  CHECK_CUDA_TENSOR(psi_row_off);
  CHECK_CUDA_TENSOR(dy);

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  torch::Tensor dydq = torch::zeros_like(qy);

  size_t uo_num_channels = kx.size(1);

  size_t sharedMemSize = (5*uo_num_channels+2)*sizeof(float);

  const int batch_size = kx.size(0);

  // cuda grid y,z size limitations
  assert(nlon_out < 65535);
  assert(batch_size < 65535);

  // block-parallel over output points and batches
  dim3 gridDim(nlat_out,nlon_out,batch_size);

  // threads compute "blocks" of neighborhood and also "blocks" of channels
  dim3 blockDim(256, 1, 1);

  s2_attention_bwd_dq_kernel <<<gridDim, blockDim, sharedMemSize, stream>>>(
                                                                            uo_num_channels, nlon_in, nlat_out, nlon_out,
                                                                            kx.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                                            vx.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                                            qy.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                                            dy.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                                            dydq.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                                            psi_col_idx.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                                                                            psi_row_off.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                                                                            quad_weights.packed_accessor32<float, 1, torch::RestrictPtrTraits>()
                                                                            );


  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return dydq;
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> s2_attention_bwd_dkvq_cuda(at::Tensor kx, at::Tensor vx,
                                                                        at::Tensor qy,
                                                                        at::Tensor dy,
                                                                        at::Tensor quad_weights,
                                                                        at::Tensor psi_col_idx,
                                                                        at::Tensor psi_row_off,
                                                                        int nlon_in, int nlat_out, int nlon_out) {

  CHECK_CUDA_TENSOR(kx);
  CHECK_CUDA_TENSOR(vx);
  CHECK_CUDA_TENSOR(qy);
  CHECK_CUDA_TENSOR(quad_weights);
  CHECK_CUDA_TENSOR(psi_col_idx);
  CHECK_CUDA_TENSOR(psi_row_off);
  CHECK_CUDA_TENSOR(dy);

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  size_t uo_num_channels = kx.size(1);

  const int batch_size = kx.size(0);

  // enum for which kernel version
  enum KERNEL_VERSION {
    OLD_VERSION = 0,
    HOWO_WARP_VERSION = 2,
  };
  auto version = HOWO_WARP_VERSION;
  // auto version = OLD_VERSION;
  if (version == OLD_VERSION) {
    printf("old version\n");
    torch::Tensor dydk = torch::zeros_like(qy);
    torch::Tensor dydv = torch::zeros_like(qy);
    torch::Tensor dydq = torch::zeros_like(qy);

    size_t sharedMemSize = (6*uo_num_channels+3)*sizeof(float);

    // cuda grid y,z size limitations
    assert(nlon_out < 65535);
    assert(batch_size < 65535);

    // block-parallel over output points and batches
    dim3 gridDim(nlat_out,nlon_out,batch_size);

    // threads compute "blocks" of neighborhood and also "blocks" of channels
    dim3 blockDim(256, 1, 1);

    // Define CUDA event variables for timing
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    // Record the start event
    cudaEventRecord(start_event, stream);

    s2_attention_bwd_dkvq_kernel<<<
      gridDim, blockDim, sharedMemSize, stream>>>(
                                                  uo_num_channels, nlon_in, nlat_out, nlon_out,
                                                  kx.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                  vx.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                  qy.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                  dy.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                  dydk.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                  dydv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                  dydq.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                  psi_col_idx.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                                                  psi_row_off.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                                                  quad_weights.packed_accessor32<float, 1, torch::RestrictPtrTraits>()
                                                  );

    // Record the stop event
    cudaEventRecord(stop_event, stream);
    cudaEventSynchronize(stop_event);

    // Calculate elapsed time
    float kernel_time_ms;
    cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);

    // Output the result

    // [1, 256, 1, (721, 1440), (721, 1440), "equiangular", "equiangular", 1e-5, 1e-5],
    // Old bwd kernel execution time: 803.477 ms
    // std::cout << "Old bwd kernel execution time: " << kernel_time_ms << " ms" << std::endl;

    // Cleanup events
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(dydk, dydv, dydq);

  } else if (version == HOWO_WARP_VERSION) {
    ScopeTimer timer("Full s2_attention_bwd_dkvq_kernel_mbT");
    // Time this function via C++
    time_t start_time, end_time;
    start_time = clock();

    // Transpose to [batch, ho, wo, channel]
    // nvtxRangePush("s2_attention_bwd_dkvq_kernel_mbT permute inputs");
    // auto* permute_timer = new ScopeTimer("permute inputs");
    // auto kxP = kx.permute({0,2,3,1}).contiguous().permute({0,3,1,2});
    // auto vxP = vx.permute({0,2,3,1}).contiguous().permute({0,3,1,2});
    // auto qyP = qy.permute({0,2,3,1}).contiguous().permute({0,3,1,2});
    // auto dyP = dy.permute({0, 2, 3, 1}).contiguous().permute({0, 3, 1, 2});

    // cudaDeviceSynchronize();
    // delete permute_timer;
    // nvtxRangePop();

    nvtxRangePush("s2_attention_bwd_dkvq_kernel_mbT output allocation & zero");
    auto dydkP = torch::zeros_like(qy);
    auto dydvP = torch::zeros_like(qy);
    auto dydqP = torch::zeros_like(qy);
    // print strdie of dydkP, dydvP, dydqP
    printf("dydkP strides: ");
    for(auto& stride_i :dydkP.strides()) {
      printf("%ld ", stride_i);
    }
    printf("\n");
    cudaDeviceSynchronize();
    nvtxRangePop();

    size_t uo_num_channels = kx.size(1);
    const int batch_size = kx.size(0);

    dim3 block(WARP_SIZE, THREADS/WARP_SIZE);
    dim3 grid(DIV_UP(nlat_out*nlon_out, block.y), batch_size);
    size_t shared_size = sizeof(float) * uo_num_channels * 5 * block.y; // 4 arrays per warp

    cudaEvent_t start, stop;
    float milliseconds = 0;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, stream));

    s2_attention_bwd_dkvq_kernel_mbT<THREADS><<<
      grid, block, shared_size, stream>>>(
                                          uo_num_channels, nlon_in, nlat_out, nlon_out,
                                          kx.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                          vx.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                          qy.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                          dy.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                          dydkP.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                          dydvP.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                          dydqP.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                          psi_col_idx.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                                          psi_row_off.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                                          quad_weights.packed_accessor32<float, 1, torch::RestrictPtrTraits>());
  
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // [1, 256, 1, (721, 1440), (721, 1440), "equiangular", "equiangular", 1e-5, 1e-5],
    // s2_attention_bwd_kernel_mbT execution time: 63.280128 ms
    printf("s2_attention_bwd_kernel_mbT execution time: %f ms\n", milliseconds);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
  
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Permute outputs back to [batch, channel, ho, wo]
    // nvtxRangePush("s2_attention_bwd_dkvq_kernel_mbT output permutation");
    // auto* permute_output_timer = new ScopeTimer("permute outputs");
    // auto dydk = dydkP.permute({0,3,1,2}).contiguous().permute({0,3,1,2});
    // auto dydv = dydvP.permute({0,3,1,2}).contiguous();
    // auto dydq = dydqP.permute({0, 3, 1, 2}).contiguous();
    // cudaDeviceSynchronize();
    // delete permute_output_timer;
    // nvtxRangePop();
    return std::make_tuple(dydkP, dydvP, dydqP);
  } else {
    throw std::runtime_error("Invalid kernel version specified");
  }
}

