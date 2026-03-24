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

#include "disco.h"
#include <torch/extension.h>
#include <algorithm>
#include <numeric>
#include <vector>

template <typename REAL_T>
void preprocess_psi_kernel(int64_t nnz, int64_t K, int64_t Ho, int64_t *ker_h, int64_t *row_h, int64_t *col_h,
                           int64_t *roff_h, REAL_T *val_h, int64_t &nrows)
{

    int64_t *Koff = new int64_t[K];
    for (int i = 0; i < K; i++) { Koff[i] = 0; }

    for (int64_t i = 0; i < nnz; i++) { Koff[ker_h[i]]++; }

    int64_t prev = Koff[0];
    Koff[0] = 0;
    for (int i = 1; i < K; i++) {
        int64_t save = Koff[i];
        Koff[i] = prev + Koff[i - 1];
        prev = save;
    }

    int64_t *ker_sort = new int64_t[nnz];
    int64_t *row_sort = new int64_t[nnz];
    int64_t *col_sort = new int64_t[nnz];
    float *val_sort = new float[nnz];

    for (int64_t i = 0; i < nnz; i++) {

        const int64_t ker = ker_h[i];
        const int64_t off = Koff[ker]++;

        ker_sort[off] = ker;
        row_sort[off] = row_h[i];
        col_sort[off] = col_h[i];
        val_sort[off] = val_h[i];
    }
    for (int64_t i = 0; i < nnz; i++) {
        ker_h[i] = ker_sort[i];
        row_h[i] = row_sort[i];
        col_h[i] = col_sort[i];
        val_h[i] = val_sort[i];
    }

    delete[] Koff;
    delete[] ker_sort;
    delete[] row_sort;
    delete[] col_sort;
    delete[] val_sort;

    // compute rows offsets
    nrows = 1;
    roff_h[0] = 0;
    for (int64_t i = 1; i < nnz; i++) {

        if (row_h[i - 1] == row_h[i]) continue;
        roff_h[nrows++] = i;

        if (nrows > Ho * K) {
            fprintf(stderr, "%s:%d: error, found more rows in the K COOs than Ho*K (%ld)\n", __FILE__, __LINE__,
                    int64_t(Ho) * K);
            exit(EXIT_FAILURE);
        }
    }
    roff_h[nrows] = nnz;

    return;
}


// ---------------------------------------------------------------------------
// Transpose preprocessing: sort NZ entries by (h_in, w_phase) so the backward
// kernel can gather without atomic writes to global memory.
//
// Sort key = h_in * pscale + w_phase  where:
//   h_in    = col_val / nlon_in
//   w_phase = (col_val % nlon_in) % pscale
//   pscale  = nlon_in / nlon_out
//
// Within a group all NZs share the same (h_in, w_phase) and therefore write
// to the same exclusive set of output positions {w_phase + pscale*pp}, so no
// global atomics are needed in the backward kernel.
// ---------------------------------------------------------------------------
template <typename REAL_T>
void preprocess_psi_transpose_kernel(int64_t nnz, int64_t nlat_in, int64_t nlon_in, int64_t nlon_out,
                                     int64_t *ker_h, int64_t *row_h, int64_t *col_h,
                                     int64_t *roff_h, REAL_T *val_h, int64_t &nrows)
{
    const int64_t Wo     = nlon_in;           // backward output longitude count
    const int64_t pscale = nlon_in / nlon_out; // Wo / Wi

    // Compute sort key for a NZ entry given its col value.
    auto sort_key = [&](int64_t col) -> int64_t {
        return (col / Wo) * pscale + (col % Wo) % pscale;
    };

    // Argsort by sort_key (stable to keep original relative order within groups).
    std::vector<int64_t> order(nnz);
    std::iota(order.begin(), order.end(), 0LL);
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        return sort_key(col_h[a]) < sort_key(col_h[b]);
    });

    // Apply permutation into temporary buffers, then copy back.
    int64_t *ker_s = new int64_t[nnz];
    int64_t *row_s = new int64_t[nnz];
    int64_t *col_s = new int64_t[nnz];
    REAL_T  *val_s = new REAL_T[nnz];

    for (int64_t i = 0; i < nnz; i++) {
        ker_s[i] = ker_h[order[i]];
        row_s[i] = row_h[order[i]];
        col_s[i] = col_h[order[i]];
        val_s[i] = val_h[order[i]];
    }
    for (int64_t i = 0; i < nnz; i++) {
        ker_h[i] = ker_s[i]; row_h[i] = row_s[i];
        col_h[i] = col_s[i]; val_h[i] = val_s[i];
    }
    delete[] ker_s; delete[] row_s; delete[] col_s; delete[] val_s;

    // Build row offsets: new row whenever (h_in, w_phase) changes.
    nrows = 1;
    roff_h[0] = 0;
    for (int64_t i = 1; i < nnz; i++) {
        if (sort_key(col_h[i - 1]) == sort_key(col_h[i])) continue;
        roff_h[nrows++] = i;
        if (nrows > nlat_in * pscale) {
            fprintf(stderr, "%s:%d: error, exceeded max transposed rows (%ld)\n",
                    __FILE__, __LINE__, (long)(nlat_in * pscale));
            exit(EXIT_FAILURE);
        }
    }
    roff_h[nrows] = nnz;
}

// Unified wrapper for both forward and transpose preprocessing.
//
// transpose=false  Sort by (ker, h_out); uses K and Ho.
//                  max rows = K * Ho
//
// transpose=true   Sort by (h_in, w_phase); uses Ho (as nlat_in), nlon_in, nlon_out.
//                  max rows = nlat_in * pscale  where pscale = nlon_in / nlon_out
//                  K is unused in this mode.
torch::Tensor preprocess_psi(const int64_t K, const int64_t Ho,
                              const int64_t nlon_in, const int64_t nlon_out,
                              torch::Tensor ker_idx, torch::Tensor row_idx,
                              torch::Tensor col_idx, torch::Tensor val,
                              bool transpose)
{
    CHECK_CONTIGUOUS_TENSOR(ker_idx);
    CHECK_CONTIGUOUS_TENSOR(row_idx);
    CHECK_CONTIGUOUS_TENSOR(col_idx);
    CHECK_CONTIGUOUS_TENSOR(val);

    auto device = ker_idx.device();
    TORCH_INTERNAL_ASSERT(device.type() == row_idx.device().type() &&
                          device.type() == col_idx.device().type() &&
                          device.type() == val.device().type());

    ker_idx = ker_idx.to(torch::kCPU);
    row_idx = row_idx.to(torch::kCPU);
    col_idx = col_idx.to(torch::kCPU);
    val     = val.to(torch::kCPU);

    int64_t nnz      = val.size(0);
    int64_t max_rows = transpose ? Ho * (nlon_in / nlon_out) : K * Ho;
    int64_t *roff_h  = new int64_t[max_rows + 1];
    int64_t nrows;

    if (transpose) {
        AT_DISPATCH_FLOATING_TYPES(val.scalar_type(), "preprocess_psi_transpose", ([&] {
            preprocess_psi_transpose_kernel<scalar_t>(
                nnz, Ho, nlon_in, nlon_out,
                ker_idx.data_ptr<int64_t>(), row_idx.data_ptr<int64_t>(),
                col_idx.data_ptr<int64_t>(), roff_h,
                val.data_ptr<scalar_t>(), nrows);
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES(val.scalar_type(), "preprocess_psi", ([&] {
            preprocess_psi_kernel<scalar_t>(
                nnz, K, Ho,
                ker_idx.data_ptr<int64_t>(), row_idx.data_ptr<int64_t>(),
                col_idx.data_ptr<int64_t>(), roff_h,
                val.data_ptr<scalar_t>(), nrows);
        }));
    }

    auto roff_idx = torch::empty({nrows + 1}, ker_idx.options());
    int64_t *roff_out = roff_idx.data_ptr<int64_t>();
    for (int64_t i = 0; i <= nrows; i++) roff_out[i] = roff_h[i];
    delete[] roff_h;

    ker_idx  = ker_idx.to(device);  row_idx = row_idx.to(device);
    col_idx  = col_idx.to(device);  val     = val.to(device);
    roff_idx = roff_idx.to(device);

    return roff_idx;
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
    m.def("preprocess_psi", &preprocess_psi,
          "Sort psi matrix for CUDA kernels. "
          "transpose=False: sort by (ker, h_out) for the forward kernel. "
          "transpose=True:  sort by (h_in, w_phase) for the atomic-free backward kernel.",
          py::arg("K"), py::arg("Ho"), py::arg("nlon_in"), py::arg("nlon_out"),
          py::arg("ker_idx"), py::arg("row_idx"), py::arg("col_idx"), py::arg("val"),
          py::arg("transpose") = false);
    m.def("cuda_kernels_is_available", &cuda_kernels_is_available, "Check if CUDA kernels are available.");
    m.def("optimized_kernels_is_available", &optimized_kernels_is_available, "Check if optimized kernels (CUDA or C++) are available.");
}

