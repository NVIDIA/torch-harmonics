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

#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/CUDAUtils.h>

#include <cub/cub.cuh>
#include <limits>

using BlockReduceFloat256 = cub::BlockReduce<float, 256>;
using BlockReduceFloat512 = cub::BlockReduce<float, 512>;

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

__global__ void
s2_attention_bwd_dv_kernel(int num_channels, int nlon_in, int nlat_out,
                           int nlon_out,
                           torch::PackedTensorAccessor32<float, 4> kx,
                           torch::PackedTensorAccessor32<float, 4> vx,
                           torch::PackedTensorAccessor32<float, 4> qy,
                           torch::PackedTensorAccessor32<float, 4> dy,
                           torch::PackedTensorAccessor32<float, 4> dydv,
                           torch::PackedTensorAccessor64<int64_t, 1> psi_col_idx,
                           torch::PackedTensorAccessor64<int64_t, 1> psi_row_offset,
                           torch::PackedTensorAccessor32<float, 1> quad_weights)
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
    float alpha_inz = expf(qdotk - qdotk_max);
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
    float alpha_inz = expf(qdotk - qdotk_max);

    // multiply alpha/sum_alpha, dy, and quadrature weights
    for(int channel_idx = 0; channel_idx<num_channels; channel_idx++) {
      atomicAdd(&dydv[batch_b][channel_idx][hi][wip], (alpha_inz/alpha_sum) * dy[batch_b][channel_idx][ho][wo] * quad_weights[hi]);
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
                       kx.packed_accessor32<float, 4>(), vx.packed_accessor32<float, 4>(),
                       qy.packed_accessor32<float, 4>(),
                       dy.packed_accessor32<float, 4>(),
                       dydv.packed_accessor32<float, 4>(),
                       psi_col_idx.packed_accessor64<int64_t, 1>(),
                       psi_row_off.packed_accessor64<int64_t, 1>(),
                       quad_weights.packed_accessor32<float, 1>()
                       );


  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return dydv;
}

__global__ void
s2_attention_bwd_dk_kernel(int num_channels, int nlon_in, int nlat_out,
                           int nlon_out, int max_nnz,
                           torch::PackedTensorAccessor32<float, 4> kx,
                           torch::PackedTensorAccessor32<float, 4> vx,
                           torch::PackedTensorAccessor32<float, 4> qy,
                           torch::PackedTensorAccessor32<float, 4> dy,
                           torch::PackedTensorAccessor32<float, 4> dydk,
                           torch::PackedTensorAccessor64<int64_t, 1> psi_col_idx,
                           torch::PackedTensorAccessor64<int64_t, 1> psi_row_offset,
                           torch::PackedTensorAccessor32<float, 1> quad_weights)
{
    // shared memory
  extern __shared__ float sharedMem[];

  float* sh_alpha_sum = (float*)&sharedMem;
  float* sh_alpha_inz = (float*)&sharedMem[1];
  float *sh_qy_ho_wo = (float *)&sharedMem[1 + max_nnz];
  float *sh_integral = (float *)&sharedMem[1 + max_nnz + num_channels];
  float *sh_dy_ho_wo = (float *)&sharedMem[2 + max_nnz + num_channels];
  float *sh_qdotk_inz = (float *)&sharedMem[2 + max_nnz + 2 * num_channels];
  float *sh_qdotk_max = (float *)&sharedMem[2 + 2 * max_nnz + 2 * num_channels];
  typename BlockReduceFloat256::TempStorage* max_reduce_temp_storage = (typename BlockReduceFloat256::TempStorage*)&sharedMem[3 + 2 * max_nnz + 2 * num_channels];

  if(threadIdx.x == 0) {sh_alpha_sum[0] = 0.0; sh_integral[0]=0.0;}
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
  int psi_nnz_ho = psi_row_offset[ho+1] - psi_offset;
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
    sh_qdotk_inz[idz] = qdotk;
  }
  __syncthreads();
  // max reduction qdotk
  float qdotk_max = std::numeric_limits<float>::lowest();
  for(int psi_block=0; psi_block<(psi_nnz_ho/blockDim.x)+1; psi_block++) {
    int idz = psi_block * blockDim.x + threadIdx.x;

    // handle case when thread > number of elements
    float qdotk = (idz<psi_nnz_ho) ? sh_qdotk_inz[idz] : std::numeric_limits<float>::lowest();
    // reduce
    float block_max = BlockReduceFloat256(*max_reduce_temp_storage)
      .Reduce(qdotk, cub::Max());

    // note: 'block_max' is only valid for thread 0
    if(threadIdx.x == 0) qdotk_max = std::max(block_max, qdotk_max);
  }
  __syncthreads();
  // store full max-reduction into shared memory for all threads
  if(threadIdx.x == 0) sh_qdotk_max[0] = qdotk_max;
  __syncthreads();

  // "broadcast" qdotk_max back into all thread-local registers
  qdotk_max = sh_qdotk_max[0];
  for(int psi_block=0; psi_block<(psi_nnz_ho/blockDim.x)+1; psi_block++) {
    int idz = psi_block*blockDim.x + threadIdx.x;
    float alpha_inz = 0.0;
    float gdotv = 0.0;
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
    for(int channel_idx = 0; channel_idx<num_channels; channel_idx++) {
      gdotv += sh_dy_ho_wo[channel_idx]*vx[batch_b][channel_idx][hi][wip];
    }
    // softmax numerator
    alpha_inz = expf(sh_qdotk_inz[idz] - qdotk_max);
    sh_alpha_inz[idz] = alpha_inz;
    // sum alpha
    atomicAdd(&sh_alpha_sum[0], alpha_inz);
    atomicAdd(&sh_integral[0], alpha_inz * gdotv * quad_weights[hi]);

  }
  __syncthreads();
  if(threadIdx.x==0) sh_integral[0] /= sh_alpha_sum[0];
  __syncthreads();

  // divide output by alpha_sum
  for(int psi_block=0; psi_block<(psi_nnz_ho/blockDim.x)+1; psi_block++) {
    float gdotv = 0.0;

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

    for(int channel_idx = 0; channel_idx<num_channels; channel_idx++) {
      gdotv += sh_dy_ho_wo[channel_idx]*vx[batch_b][channel_idx][hi][wip];
    }

    // multiply alpha/sum_alpha, vx, and quadrature weights
    for(int channel_idx = 0; channel_idx<num_channels; channel_idx++) {
      atomicAdd(&dydk[batch_b][channel_idx][hi][wip],
                sh_qy_ho_wo[channel_idx] * (sh_alpha_inz[idz]/sh_alpha_sum[0]) * (quad_weights[hi] * gdotv - sh_integral[0]));
    }
  }
  __syncthreads();
}

__global__ void
s2_attention_bwd_dq_kernel(int num_channels, int nlon_in, int nlat_out,
                           int nlon_out, int max_nnz,
                           torch::PackedTensorAccessor32<float, 4> kx,
                           torch::PackedTensorAccessor32<float, 4> vx,
                           torch::PackedTensorAccessor32<float, 4> qy,
                           torch::PackedTensorAccessor32<float, 4> dy,
                           torch::PackedTensorAccessor32<float, 4> dydq,
                           torch::PackedTensorAccessor64<int64_t, 1> psi_col_idx,
                           torch::PackedTensorAccessor64<int64_t, 1> psi_row_offset,
                           torch::PackedTensorAccessor32<float, 1> quad_weights)
{
    // shared memory
  extern __shared__ float sharedMem[];

  float* sh_alpha_sum = (float*)&sharedMem;
  float *sh_qy_ho_wo = (float *)&sharedMem[1];
  float *sh_alpha_k = (float *)&sharedMem[1 + num_channels];
  float *sh_alpha_vw = (float *)&sharedMem[1 + 2*num_channels];
  float *sh_alpha_kvw = (float *)&sharedMem[1 + 3*num_channels];
  float *sh_dy_ho_wo = (float *)&sharedMem[1 + 4 * num_channels];
  float *sh_qdotk_inz = (float *)&sharedMem[1 + 5 * num_channels];
  float *sh_qdotk_max = (float *)&sharedMem[1 +  max_nnz + 5 * num_channels];
  typename BlockReduceFloat256::TempStorage* max_reduce_temp_storage = (typename BlockReduceFloat256::TempStorage*)&sharedMem[2 +  max_nnz + 5 * num_channels];

  if(threadIdx.x == 0) {sh_alpha_sum[0] = 0.0;}
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
  int psi_nnz_ho = psi_row_offset[ho+1] - psi_offset;
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
    sh_qdotk_inz[idz] = qdotk;
  }
  __syncthreads();
  // max reduction qdotk
  float qdotk_max = std::numeric_limits<float>::lowest();
  for(int psi_block=0; psi_block<(psi_nnz_ho/blockDim.x)+1; psi_block++) {
    int idz = psi_block * blockDim.x + threadIdx.x;

    // handle case when thread > number of elements
    float qdotk = (idz<psi_nnz_ho) ? sh_qdotk_inz[idz] : std::numeric_limits<float>::lowest();
    // reduce
    float block_max = BlockReduceFloat256(*max_reduce_temp_storage)
      .Reduce(qdotk, cub::Max());

    // note: 'block_max' is only valid for thread 0
    if(threadIdx.x == 0) qdotk_max = std::max(block_max, qdotk_max);
  }
  __syncthreads();
  // store full max-reduction into shared memory for all threads
  if(threadIdx.x == 0) sh_qdotk_max[0] = qdotk_max;
  __syncthreads();

  // "broadcast" qdotk_max back into all thread-local registers
  qdotk_max = sh_qdotk_max[0];
  for(int psi_block=0; psi_block<(psi_nnz_ho/blockDim.x)+1; psi_block++) {
    int idz = psi_block*blockDim.x + threadIdx.x;
    float gdotv = 0.0f;
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
    for(int channel_idx = 0; channel_idx<num_channels; channel_idx++) {
      gdotv += sh_dy_ho_wo[channel_idx]*vx[batch_b][channel_idx][hi][wip];
    }
    // softmax numerator
    float alpha_inz = expf(sh_qdotk_inz[idz] - qdotk_max);
    // sum alpha
    atomicAdd(&sh_alpha_sum[0], alpha_inz);
    for(int channel_idx = 0; channel_idx<num_channels; channel_idx++) {
      atomicAdd(&sh_alpha_k[channel_idx],
                alpha_inz * kx[batch_b][channel_idx][hi][wip]);
      atomicAdd(&sh_alpha_vw[channel_idx],
                alpha_inz * gdotv * quad_weights[hi]);
      atomicAdd(&sh_alpha_kvw[channel_idx],
                alpha_inz * kx[batch_b][channel_idx][hi][wip] * gdotv * quad_weights[hi]);
    }
  }
  __syncthreads();
  for(int channel_block_i = 0; channel_block_i<(num_channels/blockDim.x)+1; channel_block_i++) {
    int channel_idx = channel_block_i*blockDim.x + threadIdx.x;
    if (channel_idx >= num_channels)
      break;

    dydq[batch_b][channel_idx][ho][wo] = (sh_alpha_kvw[channel_idx]*sh_alpha_sum[0] - sh_alpha_vw[channel_idx]*sh_alpha_k[channel_idx])/(sh_alpha_sum[0]*sh_alpha_sum[0]);
  }

}


at::Tensor s2_attention_bwd_dk_cuda(at::Tensor kx, 
                                    at::Tensor vx,
                                    at::Tensor qy,
                                    at::Tensor dy,
                                    at::Tensor quad_weights,
                                    at::Tensor psi_col_idx,
                                    at::Tensor psi_row_off,
                                    const int max_psi_nnz,
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

  size_t sharedMemSize = (2*uo_num_channels+2*max_psi_nnz+3)*sizeof(float) + sizeof(typename BlockReduceFloat256::TempStorage);

  const int batch_size = kx.size(0);

  // cuda grid y,z size limitations
  assert(nlon_out < 65535);
  assert(batch_size < 65535);

  // block-parallel over output points and batches
  dim3 gridDim(nlat_out,nlon_out,batch_size);

  // threads compute "blocks" of neighborhood and also "blocks" of channels
  dim3 blockDim(256, 1, 1);

  s2_attention_bwd_dk_kernel <<<gridDim, blockDim, sharedMemSize, stream>>>(
                       uo_num_channels, nlon_in, nlat_out, nlon_out, max_psi_nnz,
                       kx.packed_accessor32<float, 4>(), vx.packed_accessor32<float, 4>(),
                       qy.packed_accessor32<float, 4>(),
                       dy.packed_accessor32<float, 4>(),
                       dydk.packed_accessor32<float, 4>(),
                       psi_col_idx.packed_accessor64<int64_t, 1>(),
                       psi_row_off.packed_accessor64<int64_t, 1>(),
                       quad_weights.packed_accessor32<float, 1>()
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
                                    const int max_psi_nnz,
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

  size_t sharedMemSize = (5*uo_num_channels+max_psi_nnz+2)*sizeof(float) + sizeof(typename BlockReduceFloat256::TempStorage);

  const int batch_size = kx.size(0);

  // cuda grid y,z size limitations
  assert(nlon_out < 65535);
  assert(batch_size < 65535);

  // block-parallel over output points and batches
  dim3 gridDim(nlat_out,nlon_out,batch_size);

  // threads compute "blocks" of neighborhood and also "blocks" of channels
  dim3 blockDim(256, 1, 1);

  s2_attention_bwd_dq_kernel <<<gridDim, blockDim, sharedMemSize, stream>>>(
                       uo_num_channels, nlon_in, nlat_out, nlon_out, max_psi_nnz,
                       kx.packed_accessor32<float, 4>(), vx.packed_accessor32<float, 4>(),
                       qy.packed_accessor32<float, 4>(),
                       dy.packed_accessor32<float, 4>(),
                       dydq.packed_accessor32<float, 4>(),
                       psi_col_idx.packed_accessor64<int64_t, 1>(),
                       psi_row_off.packed_accessor64<int64_t, 1>(),
                       quad_weights.packed_accessor32<float, 1>()
                       );


  C10_CUDA_KERNEL_LAUNCH_CHECK();
  
  return dydq;
}
