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
      atomicAdd(&dydv[batch_b][channel_idx][hi][wip], (alpha_inz/alpha_sum) * sh_dy_ho_wo[channel_idx]);
    }
  }
  __syncthreads();

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

  torch::Tensor dydk = torch::zeros_like(qy);
  torch::Tensor dydv = torch::zeros_like(qy);
  torch::Tensor dydq = torch::zeros_like(qy);

  size_t uo_num_channels = kx.size(1);

  size_t sharedMemSize = (6*uo_num_channels+3)*sizeof(float);

  const int batch_size = kx.size(0);

  // cuda grid y,z size limitations
  assert(nlon_out < 65535);
  assert(batch_size < 65535);

  // block-parallel over output points and batches
  dim3 gridDim(nlat_out,nlon_out,batch_size);

  // threads compute "blocks" of neighborhood and also "blocks" of channels
  dim3 blockDim(256, 1, 1);

  s2_attention_bwd_dkvq_kernel<<<gridDim, blockDim, sharedMemSize, stream>>>(
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


  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_tuple(dydk, dydv, dydq);
}

