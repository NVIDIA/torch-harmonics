// coding=utf-8
//
// SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <Python.h>
#include "spectral.h"

extern "C" {
PyMODINIT_FUNC PyInit__C(void)
{
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_C",
        NULL,
        -1,
        NULL,
    };
    return PyModule_Create(&module_def);
}
}

namespace spectral_kernels {

TORCH_LIBRARY(spectral_kernels, m)
{
    m.def(
        "forward(Tensor x, Tensor w, int num_groups, int gemm_dtype_code, bool accum_fp32) -> Tensor",
        {at::Tag::pt2_compliant_tag});
    m.def(
        "forward_prepacked(Tensor x, Tensor w_re, Tensor w_im, int num_groups, bool accum_fp32) -> Tensor",
        {at::Tag::pt2_compliant_tag});
}

}  // namespace spectral_kernels
