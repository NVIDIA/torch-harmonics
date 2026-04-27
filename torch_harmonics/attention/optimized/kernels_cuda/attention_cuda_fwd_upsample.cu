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

// =====================================================================================
// Upsample (scatter-style) attention forward — CUDA
// =====================================================================================
//
// K, V live on the input (smaller) grid; Q lives on the output (larger) grid.
// psi is built with rows indexed by hi and cols encoding (ho, wo_canonical) on
// the output grid as ho * nlon_out + wo_canonical (canonical at wi=0). For
// wi > 0 the actual output column is (wo_canonical + pscale_out * wi) mod
// nlon_out, with pscale_out = nlon_out / nlon_in. Requires nlon_out % nlon_in
// == 0.
//
// Algorithm: classical 2-pass scatter softmax (matches the torch and CPU
// references). Each input neighbor "fires" into one or more output cells; the
// natural CUDA implementation parallelizes over (b, hi, wi, idz) and uses
// atomicMax / atomicAdd on per-output state buffers (qdotk_max, alpha_sum,
// y_acc).
//
// This file currently only ships the host-side launcher; the device kernel(s)
// will be filled in later.
// =====================================================================================

#include "attention_cuda.cuh"
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/CUDAUtils.h>

#include <cuda_runtime.h>

#include "cudamacro.h"
#include "attention_cuda_utils.cuh"

namespace attention_kernels {

// -----------------------------------------------------------------------------
// host dispatcher — called from s2_attention_fwd_cuda when the direction is
// upsample (nlon_out % nlon_in == 0). Stub for now; replace TORCH_CHECK with
// the actual scatter launcher selection once the device kernel(s) land.
// -----------------------------------------------------------------------------
void s2_attn_fwd_upsample_dispatch(int batch_size,
                              size_t nchans_in,
                              size_t nchans_out,
                              int64_t nlon_in,
                              int64_t nlat_in,
                              int64_t nlat_out,
                              int64_t nlon_out,
                              torch::Tensor kxP,
                              torch::Tensor vxP,
                              torch::Tensor qyP,
                              torch::Tensor psi_row_off,
                              torch::Tensor psi_col_idx,
                              torch::Tensor quad_weights,
                              torch::Tensor yP) {
    TORCH_CHECK(false,
                "CUDA upsample forward kernel is not yet implemented. "
                "For now, run the upsample direction on CPU or via the torch reference.");
}

}
