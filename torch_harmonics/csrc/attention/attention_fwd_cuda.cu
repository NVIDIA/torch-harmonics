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

__global__ void s2_attention_kernel(int num_channels, int nlon_in, int nlat_out,
                                    int nlon_out,
                                    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> kx,
                                    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> vx,
                                    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> qy,
                                    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> y,
                                    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> psi_col_idx,
                                    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> psi_row_offset,
                                    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> quad_weights)
{

  // shared memory
  extern __shared__ float sharedMem[];
  float *sh_alpha_sum = (float *)&sharedMem;
  float* sh_qdotk_max = (float*)&sharedMem[1];
  float* sh_qy_ho_wo = (float *)&sharedMem[2];

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
    y[batch_b][channel_idx][ho][wo] = 0.0;
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
    qdotk_max = std::max(qdotk_max, qdotk);
  }

  // collect thread-local qdotk max
  atomicMax(&sh_qdotk_max[0], qdotk_max);
  __syncthreads();
  // "broadcast" qdotk_max back into all thread-local registers
  qdotk_max = sh_qdotk_max[0];

  float alpha_sum = 0.0f;
  for(int psi_block=0; psi_block<(psi_nnz_ho/blockDim.x)+1; psi_block++) {
    int idz = psi_block*blockDim.x + threadIdx.x;
    float alpha_inz = 0.0;
    // skip if index >= length of psi_idx because last loop iteration will have extra threads
    if(idz < psi_nnz_ho) {

      int nz_col_idx = psi_col_idx[psi_offset+idz];

      // compute input indices from psi datastructure
      int hi = nz_col_idx / nlon_in;
      // account for output shift and ensure positive index due to circular condition
      // int wi = (nz_col_idx % nlon_in - wo) % nlon_in;
      int wi = nz_col_idx % nlon_in;
      int wip = (wi + wo) % nlon_in;

      // softmax numerator with minus qdotk_max to avoid numerical overflow.
      // Because qdotk_max is in both numerator and denominator (due to
      // alpha_sum), it doesn't effect the solution other than removing overflow
      // correlation Q&K (dot-product Q.K)
      float qdotk = 0.0;
      for(int channel_idx = 0; channel_idx<num_channels; channel_idx++) {
        qdotk += sh_qy_ho_wo[channel_idx]*kx[batch_b][channel_idx][hi][wip];
      }
      alpha_inz = expf(qdotk - qdotk_max) * quad_weights[hi];

      // thread-local sum alpha
      alpha_sum += alpha_inz;

      // multiply alpha, vx, and quadrature weights
      for(int channel_idx = 0; channel_idx<num_channels; channel_idx++) {
        atomicAdd(&y[batch_b][channel_idx][ho][wo], alpha_inz * vx[batch_b][channel_idx][hi][wip]);
      }
    }
  }

  // collect all alpha_sum across threads
  atomicAdd(&sh_alpha_sum[0], alpha_sum);
  __syncthreads();
  // rebroadcast sum to all threads
  alpha_sum = sh_alpha_sum[0];
  // divide output by alpha_sum
  for(int channel_block_i = 0; channel_block_i<(num_channels/blockDim.x)+1; channel_block_i++) {
    int channel_idx = channel_block_i*blockDim.x + threadIdx.x;
    if(channel_idx >= num_channels) break;
    y[batch_b][channel_idx][ho][wo] /= alpha_sum;
  }

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

  // allocate output
  torch::Tensor y = torch::zeros_like(qy);

  size_t uo_num_channels = kx.size(1);

  size_t sharedMemSize = (uo_num_channels+2)*sizeof(float);

  const int batch_size = kx.size(0);

  // cuda grid y,z size limitations
  assert(nlon_out < 65535);
  assert(batch_size < 65535);

  // block-parallel over output points and batches
  dim3 gridDim(nlat_out,nlon_out,batch_size);

  // threads compute "blocks" of neighborhood and also "blocks" of channels
  // note: blocksize of 512 runs into resource limits
  dim3 blockDim(256,1,1);

  s2_attention_kernel<<<gridDim, blockDim, sharedMemSize,stream>>>(
                       uo_num_channels, nlon_in, nlat_out, nlon_out,
                       kx.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
		       vx.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                       qy.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
		       y.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                       psi_col_idx.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                       psi_row_off.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                       quad_weights.packed_accessor32<float, 1, torch::RestrictPtrTraits>()
                       );

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return y;
}

