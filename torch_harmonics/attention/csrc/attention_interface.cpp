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
#include "attention.h"

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

namespace attention_kernels {

    // Declare the operators
    TORCH_LIBRARY(attention_kernels, m) {
        m.def("forward(Tensor kx, Tensor vx, Tensor qy, Tensor quad_weights, Tensor col_idx, Tensor row_off, int nlon_in, int nlat_out, int nlon_out) -> Tensor", {at::Tag::pt2_compliant_tag});
        m.def("backward(Tensor kx, Tensor vx, Tensor qy, Tensor dy, Tensor quad_weights, Tensor col_idx, Tensor row_off, int nlon_in, int nlat_out, int nlon_out) -> (Tensor, Tensor, Tensor)", {at::Tag::pt2_compliant_tag});
        m.def("forward_ring_step(Tensor kx, Tensor vx, Tensor qy, Tensor(a!) y_acc, Tensor(b!) alpha_sum_buf, Tensor(c!) qdotk_max_buf, Tensor quad_weights, Tensor col_idx, Tensor row_off, Tensor row_idx, int nlon_in, int lon_lo_kx, int lat_halo_start, int nlat_out, int nlon_out) -> ()");
        m.def("backward_ring_step_pass1(Tensor kx, Tensor vx, Tensor qy, Tensor dy, Tensor(a!) alpha_sum_buf, Tensor(b!) qdotk_max_buf, Tensor(c!) integral_buf, Tensor(d!) alpha_k_buf, Tensor(e!) alpha_kvw_buf, Tensor quad_weights, Tensor col_idx, Tensor row_off, Tensor row_idx, int nlon_in, int lon_lo_kx, int lat_halo_start, int nlat_out, int nlon_out) -> ()");
        m.def("backward_ring_step_pass2(Tensor kx, Tensor vx, Tensor qy, Tensor dy, Tensor alpha_sum_buf, Tensor qdotk_max_buf, Tensor integral_norm_buf, Tensor(a!) dkx, Tensor(b!) dvx, Tensor quad_weights, Tensor col_idx, Tensor row_off, Tensor row_idx, int nlon_in, int lon_lo_kx, int lat_halo_start, int nlat_out, int nlon_out) -> ()");
    }

}