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

// What this build of the DISCO extension contains, queryable before the extension itself
// is loaded. psi is prepared in Python (torch_harmonics/disco/_psi.py), so the module holds
// nothing else.

#include <torch/extension.h>

// set default values for BUILD_CPP and BUILD_CUDA
#ifndef BUILD_CPP
#define BUILD_CPP 0
#endif

#ifndef BUILD_CUDA
#define BUILD_CUDA 0
#endif

#ifndef BUILD_KPACKED_SM90
#define BUILD_KPACKED_SM90 0
#endif

#ifndef BUILD_KPACKED_SM100
#define BUILD_KPACKED_SM100 0
#endif

bool cpp_kernels_is_available() { return static_cast<bool>(BUILD_CPP); }

bool cuda_kernels_is_available() { return static_cast<bool>(BUILD_CUDA); }

bool optimized_kernels_is_available() { return cuda_kernels_is_available() || cpp_kernels_is_available(); }

bool kpacked_sm90_kernels_is_available() { return static_cast<bool>(BUILD_KPACKED_SM90); }

bool kpacked_sm100_kernels_is_available() { return static_cast<bool>(BUILD_KPACKED_SM100); }

PYBIND11_MODULE(disco_helpers, m)
{
    m.def("cuda_kernels_is_available", &cuda_kernels_is_available, "Check if CUDA kernels are available.");
    m.def("optimized_kernels_is_available", &optimized_kernels_is_available,
          "Check if optimized kernels (CUDA or C++) are available.");
    m.def("kpacked_sm90_kernels_is_available", &kpacked_sm90_kernels_is_available,
          "Check if Hopper SM_90a kpacked kernels were requested at build time.");
    m.def("kpacked_sm100_kernels_is_available", &kpacked_sm100_kernels_is_available,
          "Check if Blackwell SM_100a kpacked kernels were requested at build time.");
}
