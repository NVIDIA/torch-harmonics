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


//#include <ATen/cuda/detail/TensorInfo.cuh>
//#include <ATen/cuda/detail/KernelUtils.h>
//#include <ATen/cuda/detail/IndexUtils.cuh>
#include <torch/all.h>

#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

#include "cudamacro.h"
#include "permute_cuda.cuh"

// Define the missing macros
#define TRANSP_WARPS_X_TILE_GENERIC (32)
#define TRANSP_WARPS_X_TILE_SM100    (4)

namespace utility_kernels {

    // BEGIN - 4D tensor permutation kernels and functions
__global__ void empty_k() {}

static int getPtxver() {
    cudaFuncAttributes attrs;
    CHECK_CUDA(cudaFuncGetAttributes(&attrs, empty_k));
    return attrs.ptxVersion*10;
}

torch::Tensor permute_4D_to0231(torch::Tensor src) {

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

torch::Tensor permute_4D_to0312(torch::Tensor src) {

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

TORCH_LIBRARY_IMPL(utility_kernels, CUDA, m)
{
    m.impl("permute_to_0231",  &permute_4D_to0231);
    m.impl("permute_to_0312",  &permute_4D_to0312);
}

// END - tensor permutation kernels and functions

}