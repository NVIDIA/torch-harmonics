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

// Python binding of a Kernel Factory DISCO backward (arc scatter) kernel (one extension per
// architecture; kf_entry.cu of the architecture's directory provides kf_disco_bwd).
// Experimental: used by the regular-kf backend, see torch_harmonics/disco/optimized/disco_kf.py.

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

void kf_disco_bwd(const at::Tensor &inp, const at::Tensor &row_ker, const at::Tensor &row_lat,
                  const at::Tensor &seg_off, const at::Tensor &seg, const at::Tensor &val_off, const at::Tensor &vals,
                  int64_t nlat_out, int64_t nlon_out, const at::Tensor &out);

// inp (B, C, K, H_in, W_in) fp16, vals fp32 -> out (B, C, H_out, W_out) fp16, written in place
static void run(at::Tensor inp, at::Tensor row_ker, at::Tensor row_lat, at::Tensor seg_off, at::Tensor seg,
                at::Tensor val_off, at::Tensor vals, int64_t nlat_out, int64_t nlon_out, at::Tensor out)
{
    TORCH_CHECK(inp.is_cuda() && inp.scalar_type() == at::kHalf && inp.is_contiguous(),
                "kf backward: inp must be contiguous fp16 on CUDA");
    TORCH_CHECK(vals.scalar_type() == at::kFloat, "kf backward: vals must be fp32");
    TORCH_CHECK(out.scalar_type() == at::kHalf && out.is_contiguous(), "kf backward: out must be contiguous fp16");
    const c10::cuda::CUDAGuard guard(inp.device());
    kf_disco_bwd(inp, row_ker.contiguous(), row_lat.contiguous(), seg_off.contiguous(), seg.contiguous(),
                 val_off.contiguous(), vals.contiguous(), nlat_out, nlon_out, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("run", &run, "Kernel Factory DISCO backward contraction (arc scatter)");
}
