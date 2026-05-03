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

#include "disco_helpers.h"
#include <torch/extension.h>

torch::Tensor preprocess_psi(const int64_t K, const int64_t Ho, torch::Tensor ker_idx, torch::Tensor row_idx,
                             torch::Tensor col_idx, torch::Tensor val)
{

    CHECK_CONTIGUOUS_TENSOR(ker_idx);
    CHECK_CONTIGUOUS_TENSOR(row_idx);
    CHECK_CONTIGUOUS_TENSOR(col_idx);
    CHECK_CONTIGUOUS_TENSOR(val);

    // get the input device and make sure all tensors are on the same device
    auto device = ker_idx.device();
    TORCH_INTERNAL_ASSERT(device.type() == row_idx.device().type() && (device.type() == col_idx.device().type()) && (device.type() == val.device().type()));

    // move to cpu
    ker_idx = ker_idx.to(torch::kCPU);
    row_idx = row_idx.to(torch::kCPU);
    col_idx = col_idx.to(torch::kCPU);
    val = val.to(torch::kCPU);

    int64_t nnz = val.size(0);
    int64_t *ker_h = ker_idx.data_ptr<int64_t>();
    int64_t *row_h = row_idx.data_ptr<int64_t>();
    int64_t *col_h = col_idx.data_ptr<int64_t>();
    int64_t *roff_h = new int64_t[Ho * K + 1];
    int64_t nrows;

    AT_DISPATCH_FLOATING_TYPES(val.scalar_type(), "preprocess_psi", ([&] {
                                   preprocess_psi_kernel<scalar_t>(nnz, K, Ho, ker_h, row_h, col_h, roff_h,
                                                                   val.data_ptr<scalar_t>(), nrows);
                               }));

    // create output tensor
    auto roff_idx = torch::empty({nrows + 1}, row_idx.options());
    int64_t *roff_out_h = roff_idx.data_ptr<int64_t>();

    for (int64_t i = 0; i < (nrows + 1); i++) { roff_out_h[i] = roff_h[i]; }
    delete[] roff_h;

    // move to original device
    ker_idx = ker_idx.to(device);
    row_idx = row_idx.to(device);
    col_idx = col_idx.to(device);
    val = val.to(device);
    roff_idx = roff_idx.to(device);

    return roff_idx;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
pack_psi_dense(const int64_t K, const int64_t Ho, const int64_t Wi, const int64_t nbr_pad,
               torch::Tensor ker_idx, torch::Tensor row_idx, torch::Tensor col_idx,
               torch::Tensor val, torch::Tensor roff_idx)
{

    CHECK_CONTIGUOUS_TENSOR(ker_idx);
    CHECK_CONTIGUOUS_TENSOR(row_idx);
    CHECK_CONTIGUOUS_TENSOR(col_idx);
    CHECK_CONTIGUOUS_TENSOR(val);
    CHECK_CONTIGUOUS_TENSOR(roff_idx);

    TORCH_CHECK(ker_idx.dtype() == torch::kInt64,  "ker_idx must be int64");
    TORCH_CHECK(row_idx.dtype() == torch::kInt64,  "row_idx must be int64");
    TORCH_CHECK(col_idx.dtype() == torch::kInt64,  "col_idx must be int64");
    TORCH_CHECK(roff_idx.dtype() == torch::kInt64, "roff_idx must be int64");
    TORCH_CHECK(K > 0  && Ho > 0 && Wi > 0, "K, Ho, Wi must be positive");

    auto device = ker_idx.device();
    TORCH_INTERNAL_ASSERT(device.type() == row_idx.device().type() &&
                          device.type() == col_idx.device().type() &&
                          device.type() == val.device().type()     &&
                          device.type() == roff_idx.device().type());

    // move to cpu (mirrors preprocess_psi)
    auto ker_cpu  = ker_idx.to(torch::kCPU);
    auto row_cpu  = row_idx.to(torch::kCPU);
    auto col_cpu  = col_idx.to(torch::kCPU);
    auto val_cpu  = val.to(torch::kCPU);
    auto roff_cpu = roff_idx.to(torch::kCPU);

    const int64_t nrows = roff_cpu.size(0) - 1;
    TORCH_CHECK(nrows == K * Ho,
                "pack_psi_dense expects roff_idx.size(0)-1 == K*Ho (got nrows=", nrows,
                ", K=", K, ", Ho=", Ho, "); make sure preprocess_psi was run first");

    // resolve nbr_pad: <= 0 means auto-pad to max row length
    int64_t resolved_nbr_pad = nbr_pad;
    {
        const int64_t *roff_h = roff_cpu.data_ptr<int64_t>();
        int64_t max_nbr = 0;
        for (int64_t i = 0; i < nrows; i++) {
            max_nbr = std::max(max_nbr, roff_h[i + 1] - roff_h[i]);
        }
        if (resolved_nbr_pad <= 0) {
            resolved_nbr_pad = max_nbr;
        } else {
            TORCH_CHECK(resolved_nbr_pad >= max_nbr,
                        "nbr_pad (", resolved_nbr_pad,
                        ") is smaller than the maximum number of entries in any row (",
                        max_nbr, ")");
        }
    }

    // allocate outputs (zero-initialized; padding stays as (hi=0, wi=0, val=0))
    auto idx_out   = torch::zeros({K, Ho, resolved_nbr_pad, 2}, ker_idx.options().dtype(torch::kInt64).device(torch::kCPU));
    auto val_out   = torch::zeros({K, Ho, resolved_nbr_pad},    val.options().device(torch::kCPU));
    auto count_out = torch::zeros({K, Ho},                       ker_idx.options().dtype(torch::kInt64).device(torch::kCPU));

    AT_DISPATCH_FLOATING_TYPES(val.scalar_type(), "pack_psi_dense", ([&] {
                                   pack_psi_dense_kernel<scalar_t>(
                                       K, Ho, Wi, resolved_nbr_pad, nrows,
                                       ker_cpu.data_ptr<int64_t>(),
                                       row_cpu.data_ptr<int64_t>(),
                                       col_cpu.data_ptr<int64_t>(),
                                       val_cpu.data_ptr<scalar_t>(),
                                       roff_cpu.data_ptr<int64_t>(),
                                       idx_out.data_ptr<int64_t>(),
                                       val_out.data_ptr<scalar_t>(),
                                       count_out.data_ptr<int64_t>());
                               }));

    // move outputs to original device
    idx_out   = idx_out.to(device);
    val_out   = val_out.to(device);
    count_out = count_out.to(device);

    return std::make_tuple(idx_out, val_out, count_out);
}

// set default values for BUILD_CPP and BUILD_CUDA
#ifndef BUILD_CPP
#define BUILD_CPP 0
#endif

#ifndef BUILD_CUDA
#define BUILD_CUDA 0
#endif

bool cpp_kernels_is_available() {
    return static_cast<bool>(BUILD_CPP);
}

bool cuda_kernels_is_available() {
    return static_cast<bool>(BUILD_CUDA);
}

bool optimized_kernels_is_available() {
    return cuda_kernels_is_available() || cpp_kernels_is_available();
}

PYBIND11_MODULE(disco_helpers, m)
{
    m.def("preprocess_psi", &preprocess_psi, "Sort psi matrix, required for using CUDA kernels.",
          pybind11::arg("K"), pybind11::arg("Ho"),
          pybind11::arg("ker_idx"), pybind11::arg("row_idx"), pybind11::arg("col_idx"),
          pybind11::arg("val"));
    m.def("pack_psi_dense", &pack_psi_dense,
          "Repack a CSR-preprocessed psi into a dense (K, Ho, NBR_PAD, ...) layout. "
          "Returns (idx [K,Ho,NBR_PAD,2] of (hi, wi_base), val [K,Ho,NBR_PAD], count [K,Ho]). "
          "Pass nbr_pad <= 0 to auto-pad to the row maximum.",
          pybind11::arg("K"), pybind11::arg("Ho"), pybind11::arg("Wi"), pybind11::arg("nbr_pad"),
          pybind11::arg("ker_idx"), pybind11::arg("row_idx"), pybind11::arg("col_idx"),
          pybind11::arg("val"), pybind11::arg("roff_idx"));
    m.def("cuda_kernels_is_available", &cuda_kernels_is_available, "Check if CUDA kernels are available.");
    m.def("optimized_kernels_is_available", &optimized_kernels_is_available, "Check if optimized kernels (CUDA or C++) are available.");
}

