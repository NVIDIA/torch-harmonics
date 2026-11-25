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

// The extension torch_harmonics.disco._C_sm90: the DISCO kernels that need the arch-specific target(s)
// sm_90a, compiled for nothing else (see _DISCO_ARCH_EXTENSIONS in setup.py). Today the WGMMA kpacked forward (Hopper),
// the Hankel/Toeplitz tensor-core forward (disco_cuda_fwd_hankel_sm90.cu, ops forward_hankel_sm90_*) and the
// frequency-space arc scatter (disco_cuda_bwd_fft.cu, ops scatter_fft_sm90_*).
//
// Importing the module registers its operators in the disco_kernels namespace. Its TARGETS attribute lists the
// targets the build holds images for, which torch_harmonics/disco/optimized/disco_arch.py matches against the device:
// an arch-specific image loads on its own compute capability only.

#include <Python.h>

#include "disco.h"

namespace
{
    // the DISCO_TARGET_* macros are set by setup.py for each target the extension is compiled for
#ifndef DISCO_TARGET_SM90A
#define DISCO_TARGET_SM90A 0
#endif
#ifndef DISCO_TARGET_SM100A
#define DISCO_TARGET_SM100A 0
#endif
#ifndef DISCO_TARGET_SM103A
#define DISCO_TARGET_SM103A 0
#endif

    [[maybe_unused]] void add_target(PyObject *targets, const char *target)
    {
        PyObject *s = PyUnicode_FromString(target);
        if (s != NULL) {
            PyList_Append(targets, s);
            Py_DECREF(s);
        }
    }
} // namespace

extern "C" {
PyMODINIT_FUNC PyInit__C_sm90(void)
{
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_C_sm90", /* name of module */
        NULL,      /* module documentation, may be NULL */
        -1,        /* size of per-interpreter state of the module */
        NULL,      /* methods */
    };
    PyObject *module = PyModule_Create(&module_def);
    if (module == NULL) return NULL;

    PyObject *targets = PyList_New(0);
    if (targets == NULL) {
        Py_DECREF(module);
        return NULL;
    }
#if DISCO_TARGET_SM90A
    add_target(targets, "9.0a");
#endif
    PyObject *tuple = PyList_AsTuple(targets);
    Py_DECREF(targets);
    if (tuple == NULL || PyModule_AddObject(module, "TARGETS", tuple) < 0) {
        Py_XDECREF(tuple);
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
}

namespace disco_kernels
{

    TORCH_LIBRARY_FRAGMENT(disco_kernels, m)
    {
        // the tensor-core forward for fp16/bf16 on psi in the blocked layout of _psi_layouts.build_kpacked:
        // inp (B, C, H_in, W_in) -> (B, C, K, H_out, W_out)
        m.def("forward_kpacked_sm90(Tensor inp, Tensor pack_idx, Tensor pack_val, Tensor pack_offset, "
              "int kernel_size, int nlat_out, int nlon_out) -> Tensor",
              {at::Tag::pt2_compliant_tag});
    }

} // namespace disco_kernels
