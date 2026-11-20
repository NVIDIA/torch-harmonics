// coding=utf-8
//
// SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
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

// Python binding of a Kernel Factory DISCO forward kernel (one extension per architecture;
// kf_entry.cu of the architecture's directory provides kf_disco_fwd). Experimental: used by
// the regular-kf backend, see torch_harmonics/disco/optimized/disco_kf.py.

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

void kf_disco_fwd(const at::Tensor &inp, const at::Tensor &pack_idx, const at::Tensor &pack_val,
                  const at::Tensor &pack_offset, int64_t kernel_size, int64_t nlon_out, const at::Tensor &out);

// inp (B, C, H_in, W_in) fp16 -> out (B, C, K, H_out, W_out) fp16, written in place
static void run(at::Tensor inp, at::Tensor pack_idx, at::Tensor pack_val, at::Tensor pack_offset, int64_t kernel_size,
                int64_t nlon_out, at::Tensor out)
{
    TORCH_CHECK(inp.is_cuda() && inp.scalar_type() == at::kHalf && inp.is_contiguous(),
                "kf forward: inp must be contiguous fp16 on CUDA");
    TORCH_CHECK(out.scalar_type() == at::kHalf && out.is_contiguous(), "kf forward: out must be contiguous fp16");
    const c10::cuda::CUDAGuard guard(inp.device());
    kf_disco_fwd(inp, pack_idx.contiguous(), pack_val.contiguous(), pack_offset.contiguous(), kernel_size, nlon_out, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("run", &run, "Kernel Factory DISCO forward contraction (kpacked psi)");
}
