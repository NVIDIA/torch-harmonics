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

#include <Python.h>
#include "disco.h"

extern "C" {
    /* Creates a dummy empty _C module that can be imported from Python.
       The import from Python will load the .so consisting of this file
       in this extension, so that the TORCH_LIBRARY static initializers
       below are run. */
    PyMODINIT_FUNC PyInit__C(void)
    {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_C",   /* name of module */
            NULL,   /* module documentation, may be NULL */
            -1,     /* size of per-interpreter state of the module,
                       or -1 if the module keeps state in global variables. */
            NULL,   /* methods */
        };
        return PyModule_Create(&module_def);
    }
}

namespace disco_kernels {

    // Declare the operators.
    //
    // `forward`  (gather): inp[B,C,Hi,Wi] + psi  -> out[B,C,K,Ho,Wo].   K is a free axis.
    // `backward` (gather): inp[B,C,K,Hi,Wi] + psi_T -> out[B,C,Ho,Wo].  K is contracted.
    //
    // Both ops are gather-style and take CSR-like buffers. The backward op no
    // longer carries a row_idx tensor: psi_T's row is derivable from the output
    // cell coordinate via `row_T = ho * pscale + (wo % pscale)` where
    // pscale = nlon_out / nlon_in. (Hi, Wi) is the smaller grid (input of this
    // op); (Ho, Wo) is the bigger grid (output of this op).
    TORCH_LIBRARY(disco_kernels, m) {
        m.def("forward(Tensor inp, Tensor roff_idx, Tensor ker_idx, Tensor row_idx, Tensor col_idx, Tensor vals, int kernel_size, int nlat_out, int nlon_out) -> Tensor", {at::Tag::pt2_compliant_tag});
        m.def("backward(Tensor inp, Tensor roff_idx, Tensor ker_idx, Tensor col_idx, Tensor vals, int kernel_size, int nlat_out, int nlon_out) -> Tensor", {at::Tag::pt2_compliant_tag});
        // Side op kept ONLY for A/B benchmarking against the new gather backward.
        // Same signature as the pre-refactor `backward` (with row_idx). Not wired
        // into the conv module.
        m.def("backward_scatter(Tensor inp, Tensor roff_idx, Tensor ker_idx, Tensor row_idx, Tensor col_idx, Tensor vals, int kernel_size, int nlat_out, int nlon_out) -> Tensor", {at::Tag::pt2_compliant_tag});
    }

}

