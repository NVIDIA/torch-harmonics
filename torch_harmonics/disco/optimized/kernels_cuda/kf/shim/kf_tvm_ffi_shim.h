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

// The part of the TVM FFI API the Kernel Factory kernels in this directory use, on top of
// ATen, so their sources build verbatim as torch extensions. Not TVM FFI: only what these
// kernels call, with the semantics they rely on.
//
//   ffi::TensorView    a non-owning view of an at::Tensor: size(i), data_ptr(), device()
//   ffi::Tensor        an owning allocation (scratch workspaces) from FromEnvAlloc
//   TVMFFIEnvGetStream the current torch CUDA stream of the device
//   TVM_FFI_DLL_EXPORT_TYPED_FUNC   nothing: the extension's own entry point calls Run

#pragma once

#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/dlpack.h>

#include <cstdint>
#include <initializer_list>
#include <vector>

namespace tvm
{
    namespace ffi
    {

        struct ShapeView {
            std::vector<int64_t> dims;
            ShapeView(std::initializer_list<int64_t> d) : dims(d) { }
        };
        using Shape = ShapeView;

        class Tensor;

        class TensorView
        {
          public:
            TensorView(const at::Tensor &t) : t_(t) { }
            TensorView(const Tensor &t);

            int64_t size(int64_t i) const { return t_.size(i); }
            int64_t ndim() const { return t_.dim(); }
            int64_t numel() const { return t_.numel(); }
            void *data_ptr() const { return t_.data_ptr(); }
            DLDevice device() const
            {
                DLDevice d;
                d.device_type = t_.is_cuda() ? kDLCUDA : kDLCPU;
                d.device_id = t_.is_cuda() ? t_.get_device() : 0;
                return d;
            }

          private:
            at::Tensor t_;
        };

        class Tensor
        {
          public:
            Tensor() = default;
            explicit Tensor(at::Tensor t) : t_(std::move(t)) { }

            // TVM passes the environment's allocator; torch's caching allocator stands in for it
            template <typename Alloc>
            static Tensor FromEnvAlloc(Alloc, const ShapeView &shape, DLDataType dtype, DLDevice dev)
            {
                auto opts
                    = at::TensorOptions().dtype(at::toScalarType(dtype)).device(at::Device(at::kCUDA, dev.device_id));
                return Tensor(at::empty(shape.dims, opts));
            }

            void *data_ptr() const { return t_.data_ptr(); }
            int64_t size(int64_t i) const { return t_.size(i); }
            const at::Tensor &tensor() const { return t_; }

          private:
            at::Tensor t_;
        };

        inline TensorView::TensorView(const Tensor &t) : t_(t.tensor()) { }

    } // namespace ffi
} // namespace tvm

// the environment allocator argument of FromEnvAlloc; unused
constexpr int TVMFFIEnvTensorAlloc = 0;

inline void *TVMFFIEnvGetStream(int32_t /*device_type*/, int32_t device_id)
{
    return static_cast<void *>(at::cuda::getCurrentCUDAStream(device_id).stream());
}

#define TVM_FFI_DLL_EXPORT_TYPED_FUNC(name, fn)
