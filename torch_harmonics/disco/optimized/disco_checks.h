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

#include "../../csrc/tensor_checks.h"

// Input validation for the DISCO ops' host entry points, shared by the CPU and CUDA
// sides so the two cannot drift apart: one set of conditions, one set of messages. The
// device-specific part of a host is then only the device check and the launch.

namespace disco_kernels
{

    using th_checks::check_dense;
    using th_checks::check_index_vector;
    using th_checks::check_same_device;
    using th_checks::compute_dtype;

    // psi in CSR form: row offsets over the (basis function, output latitude) rows, and
    // one basis function, output latitude, flat input column and value per entry. The
    // kernels read all five through raw pointers -- the indices as int64, the values in
    // the compute dtype of the activations -- so the dtypes are part of the contract.
    inline void check_csr_psi(const at::Tensor &inp, const at::Tensor &roff_idx, const at::Tensor &ker_idx,
                              const at::Tensor &row_idx, const at::Tensor &col_idx, const at::Tensor &vals, int64_t K)
    {
        TORCH_CHECK(K > 0, "kernel_size must be positive, got ", K);

        const struct {
            const at::Tensor &t;
            const char *name;
        } indices[] = {{roff_idx, "roff_idx"}, {ker_idx, "ker_idx"}, {row_idx, "row_idx"}, {col_idx, "col_idx"}};
        for (const auto &index : indices) {
            check_same_device(index.t, inp, index.name);
            check_index_vector(index.t, at::kLong, index.name);
        }
        TORCH_CHECK(roff_idx.size(0) >= 1, "roff_idx must hold at least the leading 0 offset");

        check_same_device(vals, inp, "vals");
        TORCH_CHECK(vals.dim() == 1, "vals must be 1-D, got shape ", vals.sizes());
        check_dense(vals, "vals");
        TORCH_CHECK(vals.scalar_type() == compute_dtype(inp.scalar_type()),
                    "vals must be in the compute dtype of the input (", compute_dtype(inp.scalar_type()), " for ",
                    inp.scalar_type(), " activations), got ", vals.scalar_type());

        const int64_t nnz = vals.size(0);
        TORCH_CHECK(ker_idx.size(0) == nnz && row_idx.size(0) == nnz && col_idx.size(0) == nnz,
                    "ker_idx, row_idx and col_idx must have one entry per value (", nnz, "), got ", ker_idx.size(0),
                    ", ", row_idx.size(0), " and ", col_idx.size(0));
    }

    // forward_regular: inp (B, C, Hi, Wi) -> (B, C, K, Ho, Wo), gathering along the
    // longitude with stride pscale = Wi / Wo.
    inline void check_forward_inputs(const at::Tensor &inp, const at::Tensor &roff_idx, const at::Tensor &ker_idx,
                                     const at::Tensor &row_idx, const at::Tensor &col_idx, const at::Tensor &vals,
                                     int64_t K, int64_t Ho, int64_t Wo)
    {
        TORCH_CHECK(inp.dim() == 4, "inp must be (B, C, Hi, Wi), got shape ", inp.sizes());
        TORCH_CHECK(Ho > 0 && Wo > 0, "nlat_out and nlon_out must be positive, got ", Ho, " and ", Wo);
        TORCH_CHECK(inp.size(3) % Wo == 0, "Wi (", inp.size(3), ") must be an integer multiple of Wo (", Wo,
                    ") for the p-shift to be exact");
        check_dense(inp, "inp");
        check_csr_psi(inp, roff_idx, ker_idx, row_idx, col_idx, vals, K);
    }

    // backward_regular, the scatter direction: inp (B, C, K, Hi, Wi) -> (B, C, Ho, Wo),
    // with pscale = Wo / Wi.
    inline void check_backward_inputs(const at::Tensor &inp, const at::Tensor &roff_idx, const at::Tensor &ker_idx,
                                      const at::Tensor &row_idx, const at::Tensor &col_idx, const at::Tensor &vals,
                                      int64_t K, int64_t Ho, int64_t Wo)
    {
        TORCH_CHECK(inp.dim() == 5, "inp must be (B, C, K, Hi, Wi), got shape ", inp.sizes());
        TORCH_CHECK(inp.size(2) == K, "inp must hold kernel_size (", K, ") basis-function planes, got ", inp.size(2));
        TORCH_CHECK(Ho > 0 && Wo > 0, "nlat_out and nlon_out must be positive, got ", Ho, " and ", Wo);
        TORCH_CHECK(inp.size(4) > 0 && Wo % inp.size(4) == 0, "Wo (", Wo, ") must be an integer multiple of Wi (",
                    inp.size(4), ") for the p-shift to be exact");
        check_dense(inp, "inp");
        check_csr_psi(inp, roff_idx, ker_idx, row_idx, col_idx, vals, K);
    }

    // forward_kpacked: the blocked-CSR layout of the tensor-core kernels. Every
    // neighbour carries (hi, wi) in pack_idx and all K_pad filter values in pack_val, and
    // pack_offset holds one row offset per output latitude plus the total.
    inline void check_kpacked_inputs(const at::Tensor &inp, const at::Tensor &pack_idx, const at::Tensor &pack_val,
                                     const at::Tensor &pack_offset, int64_t K, int64_t Ho, int64_t Wo)
    {
        TORCH_CHECK(inp.dim() == 4, "inp must be (B, C, Hi, Wi), got shape ", inp.sizes());
        TORCH_CHECK(inp.scalar_type() == at::kHalf || inp.scalar_type() == at::kBFloat16,
                    "forward_kpacked requires fp16 or bf16 activations, got ", inp.scalar_type());
        TORCH_CHECK(Ho > 0 && Wo > 0 && Wo % 8 == 0,
                    "nlat_out must be positive and nlon_out a positive multiple of 8, got ", Ho, " and ", Wo);
        TORCH_CHECK(inp.size(3) % Wo == 0, "Wi (", inp.size(3), ") must be an integer multiple of Wo (", Wo, ")");
        check_dense(inp, "inp");

        check_same_device(pack_idx, inp, "pack_idx");
        check_same_device(pack_val, inp, "pack_val");
        check_same_device(pack_offset, inp, "pack_offset");
        TORCH_CHECK(pack_idx.scalar_type() == at::kLong && pack_idx.dim() == 2 && pack_idx.size(1) == 2,
                    "pack_idx must be an int64 (nnz, 2) tensor, got ", pack_idx.scalar_type(), " of shape ",
                    pack_idx.sizes());
        TORCH_CHECK(pack_val.dim() == 2 && pack_val.size(0) == pack_idx.size(0)
                        && at::isFloatingType(pack_val.scalar_type()),
                    "pack_val must be a floating (nnz, K_pad) tensor with one row per pack_idx entry, got ",
                    pack_val.scalar_type(), " of shape ", pack_val.sizes());
        TORCH_CHECK((pack_val.size(1) == 8 || pack_val.size(1) == 16) && K <= pack_val.size(1),
                    "pack_val must be padded to K_pad 8 or 16 covering kernel_size (", K, "), got ", pack_val.size(1));
        check_index_vector(pack_offset, at::kLong, "pack_offset");
        TORCH_CHECK(pack_offset.size(0) == Ho + 1, "pack_offset must hold nlat_out + 1 (", Ho + 1, ") offsets, got ",
                    pack_offset.size(0));
        check_dense(pack_idx, "pack_idx");
        check_dense(pack_val, "pack_val");
    }

} // namespace disco_kernels
