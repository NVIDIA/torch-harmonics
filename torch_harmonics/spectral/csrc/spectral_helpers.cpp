// coding=utf-8
//
// SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>

// set default values for BUILD_CPP and BUILD_CUDA
#ifndef BUILD_CPP
#define BUILD_CPP 0
#endif

#ifndef BUILD_CUDA
#define BUILD_CUDA 0
#endif

bool cpp_kernels_is_available()
{
    return static_cast<bool>(BUILD_CPP);
}

bool cuda_kernels_is_available()
{
    return static_cast<bool>(BUILD_CUDA);
}

bool optimized_kernels_is_available()
{
    return cuda_kernels_is_available() || cpp_kernels_is_available();
}

PYBIND11_MODULE(spectral_helpers, m)
{
    m.def("cuda_kernels_is_available", &cuda_kernels_is_available, "Check if CUDA kernels are available.");
    m.def("optimized_kernels_is_available", &optimized_kernels_is_available, "Check if optimized kernels (CUDA or C++) are available.");
}
