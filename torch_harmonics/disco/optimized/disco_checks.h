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

    // psi in arc form (torch_harmonics/disco/_psi.py): per row its basis function and
    // latitude (int32), the row's range of arcs (seg_off, int64) and of values (val_off,
    // int64); per arc (ring, start, length) (int32), with start in [0, ring length) and the
    // arc wrapping at the ring's end. The values are in arc order and in the compute dtype of
    // the activations, which the kernels read them as. Only metadata is checked: validating
    // the index contents would read them, and sync.
    inline void check_arc_psi(const at::Tensor &inp, const at::Tensor &row_ker, const at::Tensor &row_lat,
                              const at::Tensor &seg_off, const at::Tensor &seg, const at::Tensor &val_off,
                              const at::Tensor &vals, int64_t K)
    {
        TORCH_CHECK(K > 0, "kernel_size must be positive, got ", K);
        TORCH_CHECK(at::isFloatingType(inp.scalar_type()), "inp must be a floating-point tensor, got ",
                    inp.scalar_type());

        check_same_device(row_ker, inp, "row_ker");
        check_same_device(row_lat, inp, "row_lat");
        check_same_device(seg_off, inp, "seg_off");
        check_same_device(seg, inp, "seg");
        check_same_device(val_off, inp, "val_off");
        check_same_device(vals, inp, "vals");

        check_index_vector(row_ker, at::kInt, "row_ker");
        check_index_vector(row_lat, at::kInt, "row_lat");
        check_index_vector(seg_off, at::kLong, "seg_off");
        check_index_vector(val_off, at::kLong, "val_off");
        const int64_t nrows = row_ker.size(0);
        TORCH_CHECK(row_lat.size(0) == nrows && seg_off.size(0) == nrows + 1 && val_off.size(0) == nrows + 1,
                    "row_ker and row_lat must hold one entry per row and seg_off, val_off one more (", nrows,
                    " rows), got ", row_lat.size(0), ", ", seg_off.size(0), " and ", val_off.size(0));

        TORCH_CHECK(seg.scalar_type() == at::kInt && seg.dim() == 2 && seg.size(1) == 3,
                    "seg must be an int32 (nsegs, 3) tensor, got ", seg.scalar_type(), " of shape ", seg.sizes());
        check_dense(seg, "seg");

        TORCH_CHECK(vals.dim() == 1, "vals must be 1-D, got shape ", vals.sizes());
        check_dense(vals, "vals");
        TORCH_CHECK(vals.scalar_type() == compute_dtype(inp.scalar_type()),
                    "vals must be in the compute dtype of the input (", compute_dtype(inp.scalar_type()), " for ",
                    inp.scalar_type(), " activations), got ", vals.scalar_type());
    }

    inline void check_forward_inputs(const at::Tensor &inp, const at::Tensor &row_ker, const at::Tensor &row_lat,
                                     const at::Tensor &seg_off, const at::Tensor &seg, const at::Tensor &val_off,
                                     const at::Tensor &vals, int64_t K, int64_t Ho, int64_t Wo)
    {
        TORCH_CHECK(inp.dim() == 4, "inp must be (B, C, Hi, Wi), got shape ", inp.sizes());
        TORCH_CHECK(Ho > 0 && Wo > 0, "nlat_out and nlon_out must be positive, got ", Ho, " and ", Wo);
        TORCH_CHECK(inp.size(3) % Wo == 0, "Wi (", inp.size(3), ") must be an integer multiple of Wo (", Wo,
                    ") for the p-shift to be exact");
        check_dense(inp, "inp");
        check_arc_psi(inp, row_ker, row_lat, seg_off, seg, val_off, vals, K);
    }

    inline void check_backward_inputs(const at::Tensor &inp, const at::Tensor &row_ker, const at::Tensor &row_lat,
                                      const at::Tensor &seg_off, const at::Tensor &seg, const at::Tensor &val_off,
                                      const at::Tensor &vals, int64_t K, int64_t Ho, int64_t Wo)
    {
        TORCH_CHECK(inp.dim() == 5, "inp must be (B, C, K, Hi, Wi), got shape ", inp.sizes());
        TORCH_CHECK(inp.size(2) == K, "inp must hold kernel_size (", K, ") basis-function planes, got ", inp.size(2));
        TORCH_CHECK(Ho > 0 && Wo > 0, "nlat_out and nlon_out must be positive, got ", Ho, " and ", Wo);
        TORCH_CHECK(inp.size(4) > 0 && Wo % inp.size(4) == 0, "Wo (", Wo, ") must be an integer multiple of Wi (",
                    inp.size(4), ") for the p-shift to be exact");
        check_dense(inp, "inp");
        check_arc_psi(inp, row_ker, row_lat, seg_off, seg, val_off, vals, K);
    }

    // The ring tables of the ragged ops: one int64 entry per ring of the grid the arcs walk,
    // the flat index of its first point and its length.
    inline void check_ring_tables(const at::Tensor &inp, const at::Tensor &ring_base, const at::Tensor &ring_size)
    {
        check_same_device(ring_base, inp, "ring_base");
        check_same_device(ring_size, inp, "ring_size");
        check_index_vector(ring_base, at::kLong, "ring_base");
        check_index_vector(ring_size, at::kLong, "ring_size");
        TORCH_CHECK(ring_base.size(0) == ring_size.size(0), "ring_base and ring_size must hold one entry per ring, got ",
                    ring_base.size(0), " and ", ring_size.size(0));
    }

    inline void check_ragged_forward_inputs(const at::Tensor &inp, const at::Tensor &row_ker, const at::Tensor &row_pt,
                                            const at::Tensor &seg_off, const at::Tensor &seg, const at::Tensor &val_off,
                                            const at::Tensor &vals, const at::Tensor &ring_base,
                                            const at::Tensor &ring_size, int64_t K, int64_t No)
    {
        TORCH_CHECK(inp.dim() == 3, "inp must be (B, C, npoints_in), got shape ", inp.sizes());
        TORCH_CHECK(No > 0, "npoints_out must be positive, got ", No);
        check_dense(inp, "inp");
        check_arc_psi(inp, row_ker, row_pt, seg_off, seg, val_off, vals, K);
        check_ring_tables(inp, ring_base, ring_size);
    }

    inline void check_ragged_backward_inputs(const at::Tensor &inp, const at::Tensor &row_ker, const at::Tensor &row_pt,
                                             const at::Tensor &seg_off, const at::Tensor &seg, const at::Tensor &val_off,
                                             const at::Tensor &vals, const at::Tensor &ring_base,
                                             const at::Tensor &ring_size, int64_t K, int64_t No)
    {
        TORCH_CHECK(inp.dim() == 4, "inp must be (B, C, K, npoints_in), got shape ", inp.sizes());
        TORCH_CHECK(inp.size(2) == K, "inp must hold kernel_size (", K, ") basis-function planes, got ", inp.size(2));
        TORCH_CHECK(No > 0, "npoints_out must be positive, got ", No);
        check_dense(inp, "inp");
        check_arc_psi(inp, row_ker, row_pt, seg_off, seg, val_off, vals, K);
        check_ring_tables(inp, ring_base, ring_size);
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
