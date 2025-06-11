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
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/CUDAUtils.h>

#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

using BlockReduceFloat256 = cub::BlockReduce<float, 256>;
using BlockReduceFloat512 = cub::BlockReduce<float, 512>;

#define WARP_SIZE (32)
#define FULL_MASK (0xFFFFFFFF)
#define THREADS (64)
#define DIV_UP(a,b) (((a)+((b)-1))/(b))

#define NNZ_TRESH (32)

#define CHECK_CUDA(call) {                                          \
    cudaError_t err = call;                                         \
    if( cudaSuccess != err) {                                       \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
              __FILE__, __LINE__, cudaGetErrorString( err) );       \
      exit(EXIT_FAILURE);                                           \
    }}

#define CHECK_ERROR(errorMessage) {                                     \
    cudaError_t err = cudaGetLastError();                               \
    if( cudaSuccess != err) {                                           \
      fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n", \
              errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) ); \
      exit(EXIT_FAILURE);                                               \
    }}

static __device__ float __warp_sum(float val) {
#pragma unroll
  for(int i = WARP_SIZE/2; i; i /= 2) {
    val += __shfl_xor_sync(FULL_MASK, val, i);
  }
  return val;
}

// easier to understand version of manual shfl_xor_sync, performance appears similar
static __device__ float __warp_sum_cub(float val) {
  // use cub to reduce within a warp
  __shared__ typename cub::WarpReduce<float>::TempStorage temp_storage;
  
  // 1. Compute sum (initially only in lane 0)
  float sum = cub::WarpReduce<float>(temp_storage).Sum(val);
  // 2. Broadcast sum to all threads
  sum = __shfl_sync(0xFFFFFFFF, sum, 0);
  return sum;
}


// one warp per (ho,wo)
template<int BDIM_X> 
__global__ 
__launch_bounds__(BDIM_X)
  void s2_attention_kernel(int num_channels,
                           int nlon_in,
                           int nlat_out,
                           int nlon_out,
                           const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> kx,
                           const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> vx,
                           const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> qy,
                           torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> y,
                           const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> psi_col_idx,
                           const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> psi_row_offset,
                           const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> quad_weights) {


  extern __shared__ float sh[];
  float *shy = sh + threadIdx.y*num_channels;

  const uint64_t batchId = blockIdx.y;
  const uint64_t wid = uint64_t(blockIdx.x)*blockDim.y + threadIdx.y;

  if (wid >= uint64_t(nlat_out)*nlon_in) {
    return;
  }
  
  const int tidx = threadIdx.x;

  const int ho = wid / nlon_out;
  const int wo = wid - (ho*nlon_out);
  
  for(int chan = tidx; chan < num_channels; chan += WARP_SIZE) {
#if 0
    // useless read, y is always zeroed before kernel is called
    shy[chan] = y[batchId][chan][ho][wo];
#else
    shy[chan] = 0;
#endif
  }
  float alpha_sum = 0.0f;
  float qdotk_max = -FLT_MAX;

  const int64_t rbeg = psi_row_offset[ho];
  const int64_t rend = psi_row_offset[ho+1];

  const int rlen = rend-rbeg;

  for(int off = 0; off < rlen; off++) {

    const int64_t col = psi_col_idx[rbeg+off];

    const int hi = col / nlon_in;
    const int wi = col - (hi*nlon_in);
    const int wip = (wi+wo) - ((wi+wo) / nlon_in) * nlon_in;

    float qdotk = 0.0f;

    for(int chan = tidx; chan < num_channels; chan += WARP_SIZE) {
      qdotk += qy[batchId][chan][ho][ wo]*
        kx[batchId][chan][hi][wip];
    }
    qdotk = __warp_sum_cub(qdotk);

    float qdotk_max_tmp;
    float alpha;
    float exp_save;

    qdotk_max_tmp = max(qdotk_max, qdotk);
    alpha = expf(qdotk - qdotk_max_tmp) * quad_weights[hi];
    exp_save = expf(qdotk_max - qdotk_max_tmp);

    alpha_sum = alpha + alpha_sum*exp_save;

    for(int chan = tidx; chan < num_channels; chan += WARP_SIZE) {
      shy[chan] = shy[chan]*exp_save + vx[batchId][chan][hi][wip]*alpha;
    }
    qdotk_max = qdotk_max_tmp;
  }

  for(int chan = tidx; chan < num_channels; chan += WARP_SIZE) {
    y[batchId][chan][ho][wo] = shy[chan] / alpha_sum;
  }

  return;
}


torch::Tensor s2_attention_fwd_cuda(at::Tensor kx, 
                                    at::Tensor vx,
                                    at::Tensor qy, 
                                    at::Tensor quad_weights,
                                    at::Tensor psi_col_idx,
                                    at::Tensor psi_row_off,
                                    int nlon_in,
                                    int nlat_out,
                                    int nlon_out) {

  CHECK_CUDA_TENSOR(kx);
  CHECK_CUDA_TENSOR(vx);
  CHECK_CUDA_TENSOR(qy);
  CHECK_CUDA_TENSOR(quad_weights);
  CHECK_CUDA_TENSOR(psi_col_idx);
  CHECK_CUDA_TENSOR(psi_row_off);

  // TODO: check sizes

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  size_t uo_num_channels = kx.size(1);

  const int batch_size = kx.size(0);

  auto k_channel_first = kx.strides()[1] == 1;
  auto v_channel_first = vx.strides()[1] == 1;
  auto q_channel_first = qy.strides()[1] == 1;

  // transpose inputs so that channels are in the last dimension, allowing for
  // coalesced memory access
  nvtxRangePush("s2_attention_fwd_kernel_mbT permute inputs");
  //Permute kx,vx,qy,dy to [batch, ho, wo, channel] in memory layout, but keep the original shape [batch, channel, ho, wo]
  auto kxP = at::Tensor();
  if (!k_channel_first) {
    // printf("Permuting kx from [batch, channel, ho, wo] to [batch, ho, wo, channel]\n");
    kxP = kx.permute({0, 2, 3, 1}).contiguous().permute({0, 3, 1, 2});
  } else {
    kxP = kx;
  }
  auto vxP = at::Tensor();
  if (!v_channel_first) {
    // printf("Permuting vx from [batch, channel, ho, wo] to [batch, ho, wo, channel]\n");
    vxP = vx.permute({0, 2, 3, 1}).contiguous().permute({0, 3, 1, 2});
  } else {
    vxP = vx;
  }
  auto qyP = at::Tensor();
  if (!q_channel_first) {
    // printf("Permuting qy from [batch, channel, ho, wo] to [batch, ho, wo, channel]\n");
    qyP = qy.permute({0, 2, 3, 1}).contiguous().permute({0, 3, 1, 2});
  } else {
    qyP = qy;
  }
  cudaDeviceSynchronize();
  nvtxRangePop();
  torch::Tensor y = torch::empty_like(qy);

  dim3 block(WARP_SIZE, THREADS/WARP_SIZE);
  dim3 grid(DIV_UP(nlat_out*nlon_out, block.y), batch_size);

  size_t shared_size = sizeof(float)*uo_num_channels * block.y;

  cudaEvent_t start, stop;
  float milliseconds = 0;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start, stream));

  s2_attention_kernel<THREADS>
    <<<grid, block, shared_size, stream>>>(uo_num_channels, nlon_in, nlat_out, nlon_out,
                                           kxP.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                           vxP.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                           qyP.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                           y.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                           psi_col_idx.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                                           psi_row_off.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                                           quad_weights.packed_accessor32<float, 1, torch::RestrictPtrTraits>());

  CHECK_CUDA(cudaEventRecord(stop, stream));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
  // printf("s2_attention_kernel_mbT execution time: %f ms\n", milliseconds);
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  // match output layout to input layout
  if (!q_channel_first) y = y.contiguous();

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return y;
}

