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

#pragma once

#include "../disco.h"

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#define CHECK_CUDA_TENSOR(x) TORCH_INTERNAL_ASSERT(x.device().type() == torch::kCUDA)
#define CHECK_CUDA_INPUT_TENSOR(x)                                                                                     \
    CHECK_CUDA_TENSOR(x);                                                                                              \
    CHECK_CONTIGUOUS_TENSOR(x)

#define DIV_UP(a, b) (((a) + ((b)-1)) / (b))

#define MIN_THREADS (64)
#define ELXTH_MAX (32)

namespace disco_kernels {

    // forward kernel
    torch::Tensor disco_cuda_fwd(torch::Tensor inp, torch::Tensor roff_idx, torch::Tensor ker_idx, torch::Tensor row_idx,
                                 torch::Tensor col_idx, torch::Tensor val, int64_t K, int64_t Ho, int64_t Wo);

    // backward kernel
    torch::Tensor disco_cuda_bwd(torch::Tensor inp, torch::Tensor roff_idx, torch::Tensor ker_idx, torch::Tensor row_idx,
                                torch::Tensor col_idx, torch::Tensor val, int64_t K, int64_t Ho, int64_t Wo);

    // ring-step forward kernel (used by the distributed ring-exchange conv).
    // Accumulates partial contributions into ``out`` in place; the caller is
    // responsible for zero-initializing ``out`` once before the first ring step.
    void disco_cuda_fwd_ring_step(torch::Tensor inp, torch::Tensor out,
                                  torch::Tensor roff_idx, torch::Tensor ker_idx,
                                  torch::Tensor row_idx, torch::Tensor col_idx,
                                  torch::Tensor val,
                                  int64_t K, int64_t Ho, int64_t Wo_local_self,
                                  int64_t Wi_global, int64_t pscale,
                                  int64_t lon_lo_src, int64_t nlon_in_local_src);

    // ring-step backward (transpose) kernel — same usage as the forward
    // ring-step but produces gradient-input contributions from gradient-output
    // chunks. Atomically accumulates into ``out`` (the grad_x buffer is
    // compute_t / fp32 on the host side for fp16/bf16 inputs).
    void disco_cuda_bwd_ring_step(torch::Tensor inp, torch::Tensor out,
                                  torch::Tensor roff_idx, torch::Tensor ker_idx,
                                  torch::Tensor row_idx, torch::Tensor col_idx,
                                  torch::Tensor val,
                                  int64_t K, int64_t Ho, int64_t Wo_local_self,
                                  int64_t Wo_global, int64_t pscale,
                                  int64_t pscale_wo_offset, int64_t lon_lo_in_self,
                                  int64_t nlon_out_local_src);

}
