// coding=utf-8
//
// SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
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

#include "disco_cpu_dense.h"

namespace disco_kernels {

    // Compute dtype for the CPU dense kernels: fp16/bf16 → fp32, otherwise the
    // input dtype. Mirrors the policy in _disco_utils._compute_dtype.
    static inline at::ScalarType _cpu_compute_dtype(at::ScalarType st) {
        if (st == at::ScalarType::Half || st == at::ScalarType::BFloat16) {
            return at::ScalarType::Float;
        }
        return st;
    }

    torch::Tensor disco_cpu_fwd_dense(torch::Tensor inp, torch::Tensor pack_idx, torch::Tensor pack_val,
        torch::Tensor pack_count, int64_t K, int64_t Ho, int64_t Wo) {

        // sanity checks
        CHECK_CPU_INPUT_TENSOR(inp);
        CHECK_CPU_INPUT_TENSOR(pack_idx);
        CHECK_CPU_INPUT_TENSOR(pack_val);
        CHECK_CPU_INPUT_TENSOR(pack_count);

        // pscale = Wi / Wo; require an integer ratio so the p-shift is exact
        TORCH_CHECK(inp.size(3) % Wo == 0,
                    "Wi (", inp.size(3), ") must be an integer multiple of Wo (", Wo, ")");

        // shape sanity
        TORCH_CHECK(pack_idx.dim()   == 4 && pack_idx.size(0)   == K && pack_idx.size(1)   == Ho && pack_idx.size(3) == 2,
                    "pack_idx must have shape [K, Ho, NBR_PAD, 2]");
        TORCH_CHECK(pack_val.dim()   == 3 && pack_val.size(0)   == K && pack_val.size(1)   == Ho,
                    "pack_val must have shape [K, Ho, NBR_PAD]");
        TORCH_CHECK(pack_count.dim() == 2 && pack_count.size(0) == K && pack_count.size(1) == Ho,
                    "pack_count must have shape [K, Ho]");

        const int64_t NBR_PAD = pack_idx.size(2);

        // CPU has no efficient bf16/fp16 path — upcast on host before the inner
        // dispatch and cast back at the end.
        const at::ScalarType in_dtype = inp.scalar_type();
        const at::ScalarType cdtype   = _cpu_compute_dtype(in_dtype);
        auto inp_c      = (in_dtype == cdtype) ? inp      : inp.to(cdtype);
        auto pack_val_c = (pack_val.scalar_type() == cdtype) ? pack_val : pack_val.to(cdtype);

        auto out_c = torch::zeros({inp_c.size(0), inp_c.size(1), K, Ho, Wo}, inp_c.options());

        AT_DISPATCH_FLOATING_TYPES(inp_c.scalar_type(), "disco_forward_dense_cpu", ([&] {
            disco_fwd_dense<scalar_t>(
                inp_c.size(0), inp_c.size(1), K, inp_c.size(2), inp_c.size(3),
                Ho, Wo, NBR_PAD,
                inp_c.packed_accessor64<scalar_t, 4>(),
                pack_idx.packed_accessor64<int64_t, 4>(),
                pack_val_c.packed_accessor64<scalar_t, 3>(),
                pack_count.packed_accessor64<int64_t, 2>(),
                out_c.packed_accessor64<scalar_t, 5>());
        }));

        return (in_dtype == cdtype) ? out_c : out_c.to(in_dtype);
    }

    torch::Tensor disco_cpu_bwd_dense(torch::Tensor inp, torch::Tensor pack_idx, torch::Tensor pack_val,
        torch::Tensor pack_count, int64_t K, int64_t Ho, int64_t Wo) {

        // sanity checks
        CHECK_CPU_INPUT_TENSOR(inp);
        CHECK_CPU_INPUT_TENSOR(pack_idx);
        CHECK_CPU_INPUT_TENSOR(pack_val);
        CHECK_CPU_INPUT_TENSOR(pack_count);

        // pscale = Wo / Wi; require an integer ratio so the p-shift is exact.
        // bwd input shape is [B, C, K, Hi, Wi]; the 4th dim is Wi.
        TORCH_CHECK(Wo % inp.size(4) == 0,
                    "Wo (", Wo, ") must be an integer multiple of Wi (", inp.size(4), ")");

        // pack is keyed by (K, Hi=inp.size(3))
        TORCH_CHECK(pack_idx.dim()   == 4 && pack_idx.size(0)   == K && pack_idx.size(1)   == inp.size(3) && pack_idx.size(3) == 2,
                    "pack_idx must have shape [K, Hi, NBR_PAD, 2] with Hi == inp.size(3)");
        TORCH_CHECK(pack_val.dim()   == 3 && pack_val.size(0)   == K && pack_val.size(1)   == inp.size(3),
                    "pack_val must have shape [K, Hi, NBR_PAD]");
        TORCH_CHECK(pack_count.dim() == 2 && pack_count.size(0) == K && pack_count.size(1) == inp.size(3),
                    "pack_count must have shape [K, Hi]");

        const int64_t NBR_PAD = pack_idx.size(2);

        // CPU has no efficient bf16/fp16 path — upcast on host before the inner
        // dispatch and cast back at the end.
        const at::ScalarType in_dtype = inp.scalar_type();
        const at::ScalarType cdtype   = _cpu_compute_dtype(in_dtype);
        auto inp_c      = (in_dtype == cdtype) ? inp      : inp.to(cdtype);
        auto pack_val_c = (pack_val.scalar_type() == cdtype) ? pack_val : pack_val.to(cdtype);

        auto out_c = torch::zeros({inp_c.size(0), inp_c.size(1), Ho, Wo}, inp_c.options());

        AT_DISPATCH_FLOATING_TYPES(inp_c.scalar_type(), "disco_backward_dense_cpu", ([&] {
            disco_bwd_dense<scalar_t>(
                inp_c.size(0), inp_c.size(1), K, inp_c.size(3), inp_c.size(4),
                Ho, Wo, NBR_PAD,
                inp_c.packed_accessor64<scalar_t, 5>(),
                pack_idx.packed_accessor64<int64_t, 4>(),
                pack_val_c.packed_accessor64<scalar_t, 3>(),
                pack_count.packed_accessor64<int64_t, 2>(),
                out_c.packed_accessor64<scalar_t, 4>());
        }));

        return (in_dtype == cdtype) ? out_c : out_c.to(in_dtype);
    }

    TORCH_LIBRARY_IMPL(disco_kernels, CPU, m)
    {
        m.impl("forward_dense",  &disco_cpu_fwd_dense);
        m.impl("backward_dense", &disco_cpu_bwd_dense);
    }

}
