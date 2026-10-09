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

#pragma once

#include <ATen/ATen.h>

// Input validation shared by the host entry points of every compiled extension in the
// package, CPU and CUDA alike. Each check reads tensor metadata only -- device, dtype,
// sizes, strides -- so it costs no synchronization and is invisible to torch.compile,
// which traces an op's fake implementation rather than its host. TORCH_CHECK rather
// than TORCH_INTERNAL_ASSERT: these are caller errors, and the message says which
// tensor is wrong and how.
//
// Header-only and namespaced on its own, so each extension pulls the helpers into its
// namespace with a using-declaration and its call sites stay unqualified.

namespace th_checks
{

    // A tensor the kernels address as dense and row-major over its logical shape: the
    // innermost dimension has stride 1, and each outer one the product of the extents
    // inside it.
    //
    // Decided from the strides directly, not from a memory-format predicate, and the stride
    // of a dimension of extent 1 is never inspected: it addresses nothing, and PyTorch
    // leaves it arbitrary, so a singleton channel (or batch, or anything else) cannot be
    // rejected on it. An empty tensor has nothing to misread and passes.
    inline void check_dense(const at::Tensor &t, const char *name)
    {
        if (t.numel() == 0) { return; }
        int64_t expected = 1;
        for (int64_t d = t.dim() - 1; d >= 0; --d) {
            if (t.size(d) != 1) {
                TORCH_CHECK(t.stride(d) == expected, name,
                            " must be dense and row-major over its logical shape (innermost dimension "
                            "contiguous); got sizes ",
                            t.sizes(), " and strides ", t.strides());
            }
            expected *= t.size(d);
        }
    }

    // The kernels take one device's pointers; a tensor elsewhere would be read through a
    // pointer that means nothing there.
    inline void check_same_device(const at::Tensor &t, const at::Tensor &ref, const char *name)
    {
        TORCH_CHECK(t.device() == ref.device(), name, " must be on the same device as the activations (", ref.device(),
                    "), got ", t.device());
    }

    // An index table the kernels read through a raw pointer of a fixed integer type.
    inline void check_index_vector(const at::Tensor &t, at::ScalarType dtype, const char *name)
    {
        TORCH_CHECK(t.scalar_type() == dtype && t.dim() == 1, name, " must be a 1-D ", dtype, " tensor, got ",
                    t.scalar_type(), " of shape ", t.sizes());
        check_dense(t, name);
    }

    // The dtype a kernel accumulates in for activations stored as `t`: fp32 for the
    // 16-bit floating types, the type itself otherwise.
    inline at::ScalarType compute_dtype(at::ScalarType t)
    {
        return (t == at::kHalf || t == at::kBFloat16) ? at::kFloat : t;
    }

} // namespace th_checks
