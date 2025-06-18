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
#define DIV_UP(a, b) (((a) + ((b)-1)) / (b))
#endif
#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                                                                 \
    {                                                                                                                    \
        cudaError_t err = call;                                                                                          \
        if (cudaSuccess != err) {                                                                                        \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                                          \
        }                                                                                                                \
    }
#endif

#include <iostream>
#include <chrono>
#include <string>

class ScopeTimer
{
  public:
    explicit ScopeTimer(const std::string &label = "") :
        label_(label), start_(std::chrono::high_resolution_clock::now())
    {
    }

    ~ScopeTimer()
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
        std::cout << label_ << "Elapsed time: " << elapsed.count() << " ms" << std::endl;
    }

  private:
    std::string label_;
    std::chrono::high_resolution_clock::time_point start_;
};

static __device__ float __warp_sum(float val)
{
#pragma unroll
    for (int i = WARP_SIZE / 2; i; i /= 2) { val += __shfl_xor_sync(FULL_MASK, val, i); }
    return val;
}

// easier to understand version of manual shfl_xor_sync, performance appears similar
static __device__ float __warp_sum_cub(float val)
{
    // use cub to reduce within a warp
    __shared__ typename cub::WarpReduce<float>::TempStorage temp_storage;

    // 1. Compute sum (initially only in lane 0)
    float sum = cub::WarpReduce<float>(temp_storage).Sum(val);
    // 2. Broadcast sum to all threads
    sum = __shfl_sync(0xFFFFFFFF, sum, 0);
    return sum;
}

// This kernel computes the backward pass for the S2 attention mechanism, using
// shared memory as a cache and one warp per output point, warp-parallel over
// channels, which should be layed out in the fastest dimension for coalesced
// memory access.
template <int BDIM_X>
__global__ __launch_bounds__(BDIM_X) void s2_attention_bwd_dkvq_kernel(
    int num_channels, int nlon_in, int nlat_out, int nlon_out,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> kx,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> vx,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> qy,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> dy,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> dydk,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> dydv,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> dydq,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> psi_col_idx,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> psi_row_offset,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> quad_weights)
{

    extern __shared__ float sh[];
    float *sh_alpha_k = sh + threadIdx.y * num_channels * 5;
    float *sh_alpha_vw = sh_alpha_k + num_channels;
    float *sh_alpha_kvw = sh_alpha_vw + num_channels;
    float *sh_dy = sh_alpha_kvw + num_channels;
    float *sh_qy = sh_dy + num_channels;
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
    const int64_t rend = psi_row_offset[ho + 1];
    const int rlen = rend - rbeg;

    // 1st pass: accumulate alpha_sum, integral, and shared stats, along with a progressively computed qdotk_max.
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
        float qdotk_max_tmp = max(qdotk_max, qdotk);
        float alpha_inz = expf(qdotk - qdotk_max_tmp) * quad_weights[hi];
        float max_correction = expf(qdotk_max - qdotk_max_tmp);
        alpha_sum = alpha_sum * max_correction + alpha_inz;
        integral = integral * max_correction + alpha_inz * gdotv;
        for (int chan = tidx; chan < num_channels; chan += WARP_SIZE) {
            float kxval = kx[batchId][chan][hi][wip];
            sh_alpha_k[chan] = sh_alpha_k[chan] * max_correction + alpha_inz * kxval;
            sh_alpha_vw[chan] = sh_alpha_vw[chan] * max_correction + alpha_inz * gdotv;
            sh_alpha_kvw[chan] = sh_alpha_kvw[chan] * max_correction + alpha_inz * kxval * gdotv;
        }
        qdotk_max = qdotk_max_tmp;
    }

    integral /= alpha_sum;

    // Write dydq
    for (int chan = tidx; chan < num_channels; chan += WARP_SIZE) {
        dydq[batchId][chan][ho][wo]
            = (sh_alpha_kvw[chan] * alpha_sum - sh_alpha_vw[chan] * sh_alpha_k[chan]) / (alpha_sum * alpha_sum);
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

std::tuple<at::Tensor, at::Tensor, at::Tensor> s2_attention_bwd_dkvq_cuda(at::Tensor kx, at::Tensor vx, at::Tensor qy,
                                                                          at::Tensor dy, at::Tensor quad_weights,
                                                                          at::Tensor psi_col_idx, at::Tensor psi_row_off,
                                                                          int nlon_in, int nlat_out, int nlon_out)
{

    CHECK_CUDA_TENSOR(kx);
    CHECK_CUDA_TENSOR(vx);
    CHECK_CUDA_TENSOR(qy);
    CHECK_CUDA_TENSOR(quad_weights);
    CHECK_CUDA_TENSOR(psi_col_idx);
    CHECK_CUDA_TENSOR(psi_row_off);
    CHECK_CUDA_TENSOR(dy);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto k_channel_first = kx.strides()[1] == 1;
    auto v_channel_first = vx.strides()[1] == 1;
    auto q_channel_first = qy.strides()[1] == 1;
    auto dy_channel_first = dy.strides()[1] == 1;

    // Transpose to [batch, ho, wo, channel]
    nvtxRangePush("s2_attention_bwd_dkvq_kernel_mbT permute inputs");
    // auto* permute_timer = new ScopeTimer("permute inputs");

    // Permute kx,vx,qy,dy to [batch, ho, wo, channel] in memory layout, but keep the original shape [batch, channel, ho, wo]
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
    auto dyP = at::Tensor();
    if (!dy_channel_first) {
        // printf("Permuting dy from [batch, channel, ho, wo] to [batch, ho, wo, channel]\n");
        dyP = dy.permute({0, 2, 3, 1}).contiguous().permute({0, 3, 1, 2});
    } else {
        dyP = dy;
    }
    // cudaDeviceSynchronize();
    // delete permute_timer;
    nvtxRangePop();

    nvtxRangePush("s2_attention_bwd_dkvq_kernel_mbT output allocation & zero");
    auto dydk = torch::zeros_like(qyP);
    auto dydv = torch::zeros_like(qyP);
    auto dydq = torch::zeros_like(qyP);
    // print strdie of dydkP, dydvP, dydqP
    nvtxRangePop();

    size_t uo_num_channels = kx.size(1);
    const int batch_size = kx.size(0);

    dim3 block(WARP_SIZE, THREADS / WARP_SIZE);
    dim3 grid(DIV_UP(nlat_out * nlon_out, block.y), batch_size);
    size_t shared_size = sizeof(float) * uo_num_channels * 5 * block.y; // 4 arrays per warp

    cudaEvent_t start, stop;
    float milliseconds = 0;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, stream));

    s2_attention_bwd_dkvq_kernel<THREADS><<<grid, block, shared_size, stream>>>(
        uo_num_channels, nlon_in, nlat_out, nlon_out, kxP.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        vxP.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        qyP.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        dyP.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        dydk.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        dydv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        dydq.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        psi_col_idx.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
        psi_row_off.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
        quad_weights.packed_accessor32<float, 1, torch::RestrictPtrTraits>());

    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // [1, 256, 1, (721, 1440), (721, 1440), "equiangular", "equiangular", 1e-5, 1e-5],
    // s2_attention_bwd_kernel execution time: 50.724865 ms
    // [1, 256, 1, (361, 720), (361, 720), "equiangular", "equiangular", 1e-5, 1e-5],
    // s2_attention_bwd_kernel execution time: 11.679744 ms
    // printf("s2_attention_bwd_kernel execution time: %f ms\n", milliseconds);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Permute outputs back to memory layout given by input. if input had channels
    // first, leave it in that layout, otherwise permute layout back to [batch,
    // channel, ho, wo]
    if (!k_channel_first) dydk = dydk.contiguous();
    if (!v_channel_first) dydv = dydv.contiguous();
    if (!q_channel_first) dydq = dydq.contiguous();

    // printf("dydk strides:[");
    // for(auto& stride : dydk.strides()) {
    //   printf("%ld,", stride);
    // }
    // printf("]\n");
    // cudaDeviceSynchronize();
    // delete permute_output_timer;
    // nvtxRangePop();
    return std::make_tuple(dydk, dydv, dydq);
}
