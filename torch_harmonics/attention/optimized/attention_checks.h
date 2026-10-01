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

#pragma once

#include <ATen/ATen.h>

#include <vector>

// Input validation for the attention ops' host entry points, shared by the CPU and CUDA
// sides. Every check reads tensor metadata only -- device, dtype, sizes, strides -- so it
// costs no synchronization and is invisible to torch.compile, which traces the ops' fake
// implementations rather than this. TORCH_CHECK rather than TORCH_INTERNAL_ASSERT: these
// are caller errors, and the message says which tensor is wrong and how.

namespace attention_kernels
{

    // A tensor the kernels address as dense and row-major over its logical shape: the
    // innermost dimension has stride 1, and each outer one the product of the extents
    // inside it. For the activations the logical shape is (B, H, W, C), heads packed along
    // C, so this is "channels-last" in the only sense the kernels rely on.
    //
    // Decided from the strides directly, not from a memory-format predicate, and the stride
    // of a dimension of extent 1 is never inspected: it addresses nothing, and PyTorch
    // leaves it arbitrary, so a singleton C (or batch, or anything else) cannot be rejected
    // on it. An empty tensor has nothing to misread and passes.
    inline void check_dense(const at::Tensor &t, const char *name)
    {
        if (t.numel() == 0) { return; }
        int64_t expected = 1;
        for (int64_t d = t.dim() - 1; d >= 0; --d) {
            if (t.size(d) != 1) {
                TORCH_CHECK(t.stride(d) == expected, name,
                            " must be dense and row-major over its logical shape (innermost dimension "
                            "contiguous); got sizes ",
                            t.sizes(), " and strides ", t.strides());
            }
            expected *= t.size(d);
        }
    }

    // The kernels take one device's pointers; a tensor elsewhere would be read through a
    // pointer that means nothing there.
    inline void check_same_device(const at::Tensor &t, const at::Tensor &ref, const char *name)
    {
        TORCH_CHECK(t.device() == ref.device(), name, " must be on the same device as the activations (", ref.device(),
                    "), got ", t.device());
    }

    // The activations share one dtype: each host dispatches on qy's scalar type and
    // reinterpret_casts the others to it, so a mismatch would be reinterpreted rather than
    // converted.
    inline void check_same_dtype(const at::Tensor &t, const at::Tensor &ref, const char *name, const char *ref_name)
    {
        TORCH_CHECK(t.scalar_type() == ref.scalar_type(), name, " dtype (", t.scalar_type(), ") must match ", ref_name,
                    " dtype (", ref.scalar_type(), ")");
    }

    // The arc form of a neighbourhood as the kernels read it: seg is (nsegs, 3) int32 and
    // seg_off (nrows + 1,) int32, both dense, since the kernels reinterpret_cast them
    // without further checks.
    inline void check_arc_pattern(const at::Tensor &seg, const at::Tensor &seg_off, int64_t nrows)
    {
        TORCH_CHECK(seg.scalar_type() == at::kInt && seg.dim() == 2 && seg.size(1) == 3,
                    "psi_seg must be an int32 (nsegs, 3) tensor, got ", seg.scalar_type(), " of shape ", seg.sizes());
        TORCH_CHECK(seg_off.scalar_type() == at::kInt && seg_off.dim() == 1 && seg_off.size(0) == nrows + 1,
                    "psi_seg_off must be an int32 tensor of ", nrows + 1, " row offsets, got ", seg_off.scalar_type(),
                    " of shape ", seg_off.sizes());
        check_dense(seg, "psi_seg");
        check_dense(seg_off, "psi_seg_off");
    }

    // The operands of the serial product-grid ops, forward_regular and backward_regular:
    // kx/vx (B, nlat_in, nlon_in, nh*C) on the input grid, qy (B, nlat_out, nlon_out,
    // nh*C_k) on the output grid, ring_weights one float32 per input ring, and the arcs,
    // whose rows are output rings on the gather path and input rings on the scatter path.
    inline void check_regular_attention_inputs(const at::Tensor &kx, const at::Tensor &vx, const at::Tensor &qy,
                                               const at::Tensor &ring_weights, const at::Tensor &psi_seg,
                                               const at::Tensor &psi_seg_off, int64_t num_heads, int64_t nlon_in,
                                               int64_t nlat_out, int64_t nlon_out)
    {
        TORCH_CHECK(kx.dim() == 4 && vx.dim() == 4 && qy.dim() == 4,
                    "kx, vx and qy must be 4-D (B, nlat, nlon, channels), got ", kx.dim(), ", ", vx.dim(), " and ",
                    qy.dim(), " dims");
        TORCH_CHECK(nlon_in > 0 && nlon_out > 0 && nlat_out > 0,
                    "nlon_in, nlat_out and nlon_out must be positive, got ", nlon_in, ", ", nlat_out, " and ", nlon_out);
        TORCH_CHECK(num_heads >= 1, "num_heads must be positive, got ", num_heads);
        // gather (self / downsample) iff nlon_in is a multiple of nlon_out, scatter (upsample)
        // iff the reverse; checked before the arc rows, which depend on it
        TORCH_CHECK(nlon_in % nlon_out == 0 || nlon_out % nlon_in == 0, "either nlon_in (", nlon_in,
                    ") must be an integer multiple of nlon_out (", nlon_out, "), or vice versa");

        check_same_device(vx, kx, "vx");
        check_same_device(qy, kx, "qy");
        check_same_device(ring_weights, kx, "ring_weights");
        check_same_device(psi_seg, kx, "psi_seg");
        check_same_device(psi_seg_off, kx, "psi_seg_off");
        check_same_dtype(kx, qy, "k", "q");
        check_same_dtype(vx, qy, "v", "q");

        // K and V are sampled on the input grid, Q on the output grid
        TORCH_CHECK(kx.size(2) == nlon_in, "kx has ", kx.size(2), " longitudes but nlon_in is ", nlon_in);
        TORCH_CHECK(vx.size(0) == kx.size(0) && vx.size(1) == kx.size(1) && vx.size(2) == kx.size(2),
                    "vx must share kx's (B, nlat_in, nlon_in), got ", vx.sizes(), " against ", kx.sizes());
        TORCH_CHECK(qy.size(0) == kx.size(0) && qy.size(1) == nlat_out && qy.size(2) == nlon_out, "qy must be (",
                    kx.size(0), ", ", nlat_out, ", ", nlon_out, ", channels), got ", qy.sizes());
        TORCH_CHECK(qy.size(3) == kx.size(3), "q and k must have the same channel count, got ", qy.size(3), " and ",
                    kx.size(3));
        TORCH_CHECK(qy.size(3) % num_heads == 0 && vx.size(3) % num_heads == 0, "channel counts (q/k ", qy.size(3),
                    ", v ", vx.size(3), ") must be divisible by num_heads (", num_heads, ")");

        TORCH_CHECK(ring_weights.scalar_type() == at::kFloat && ring_weights.dim() == 1
                        && ring_weights.size(0) == kx.size(1),
                    "ring_weights must be a float32 tensor of one weight per input ring (", kx.size(1), "), got ",
                    ring_weights.scalar_type(), " of shape ", ring_weights.sizes());

        // gather (self / downsample) keys the arcs by output ring; scatter (upsample) builds
        // the pattern with the grids swapped and keys them by input ring
        const bool gather = (nlon_in % nlon_out == 0);
        check_arc_pattern(psi_seg, psi_seg_off, gather ? nlat_out : kx.size(1));

        check_dense(kx, "kx");
        check_dense(vx, "vx");
        check_dense(qy, "qy");
        check_dense(ring_weights, "ring_weights");
    }

    // The operands of one ring step: kx/vx are the K/V chunk this step holds, halo-padded in
    // latitude and narrowed to one rank's longitudes, and qy is this rank's output block,
    // (B, nlat_out, nlon_out, nh*C_k). ring_weights is only required to be a dense float32
    // vector: the kernels index it by global input ring, which the chunk shape does not
    // determine. seg_rows is the number of arc rows -- the local output rings on the
    // gather path, the chunk's halo-padded input rings on the scatter path.
    inline void check_ring_step_inputs(const at::Tensor &kx, const at::Tensor &vx, const at::Tensor &qy,
                                       const at::Tensor &ring_weights, const at::Tensor &psi_seg,
                                       const at::Tensor &psi_seg_off, int64_t num_heads, int64_t nlat_out,
                                       int64_t nlon_out, int64_t seg_rows)
    {
        TORCH_CHECK(kx.dim() == 4 && vx.dim() == 4 && qy.dim() == 4,
                    "kx, vx and qy must be 4-D (B, nlat, nlon, channels), got ", kx.dim(), ", ", vx.dim(), " and ",
                    qy.dim(), " dims");
        TORCH_CHECK(nlat_out > 0 && nlon_out > 0, "nlat_out and nlon_out must be positive, got ", nlat_out, " and ",
                    nlon_out);
        TORCH_CHECK(num_heads >= 1, "num_heads must be positive, got ", num_heads);

        check_same_device(vx, kx, "vx");
        check_same_device(qy, kx, "qy");
        check_same_device(ring_weights, kx, "ring_weights");
        check_same_device(psi_seg, kx, "psi_seg");
        check_same_device(psi_seg_off, kx, "psi_seg_off");
        check_same_dtype(kx, qy, "k", "q");
        check_same_dtype(vx, qy, "v", "q");

        TORCH_CHECK(vx.size(0) == kx.size(0) && vx.size(1) == kx.size(1) && vx.size(2) == kx.size(2),
                    "vx must share kx's (B, nlat_halo, nlon_chunk), got ", vx.sizes(), " against ", kx.sizes());
        TORCH_CHECK(qy.size(0) == kx.size(0) && qy.size(1) == nlat_out && qy.size(2) == nlon_out, "qy must be (",
                    kx.size(0), ", ", nlat_out, ", ", nlon_out, ", channels), got ", qy.sizes());
        TORCH_CHECK(qy.size(3) == kx.size(3), "q and k must have the same channel count, got ", qy.size(3), " and ",
                    kx.size(3));
        TORCH_CHECK(qy.size(3) % num_heads == 0 && vx.size(3) % num_heads == 0, "channel counts (q/k ", qy.size(3),
                    ", v ", vx.size(3), ") must be divisible by num_heads (", num_heads, ")");
        TORCH_CHECK(ring_weights.scalar_type() == at::kFloat && ring_weights.dim() == 1,
                    "ring_weights must be a float32 vector, got ", ring_weights.scalar_type(), " of shape ",
                    ring_weights.sizes());

        check_arc_pattern(psi_seg, psi_seg_off, seg_rows);

        check_dense(kx, "kx");
        check_dense(vx, "vx");
        check_dense(qy, "qy");
        check_dense(ring_weights, "ring_weights");
    }

    // dy, the gradient of the output: shaped like qy, with v's channel count, whatever the
    // rank -- (B, nlat, nlon, C_v) on a product grid, (B, npoints, C_v) on a ragged one.
    inline void check_output_grad(const at::Tensor &dy, const at::Tensor &kx, const at::Tensor &vx, const at::Tensor &qy)
    {
        check_same_device(dy, kx, "dy");
        check_same_dtype(dy, qy, "dy", "q");
        std::vector<int64_t> expected(qy.sizes().begin(), qy.sizes().end());
        expected.back() = vx.size(-1);
        TORCH_CHECK(dy.sizes() == at::IntArrayRef(expected), "dy must have shape ", at::IntArrayRef(expected), ", got ",
                    dy.sizes());
        check_dense(dy, "dy");
    }

    // A float32 buffer the ring kernels read or write in place across steps: the softmax
    // statistics, (B, nh, nlat_out, nlon_out), the per-output vectors, (B, nlat_out,
    // nlon_out, nh*C), and the chunk gradients, shaped like their chunk. All are allocated
    // by the Python driver and cast to float* by the host.
    inline void check_state_buffer(const at::Tensor &t, const at::Tensor &kx, const char *name, at::IntArrayRef shape)
    {
        check_same_device(t, kx, name);
        TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32, got ", t.scalar_type());
        TORCH_CHECK(t.sizes() == shape, name, " must have shape ", shape, ", got ", t.sizes());
        check_dense(t, name);
    }

    // The operands of the ragged ops, forward_ragged and backward_ragged, on any GridS2:
    // fields are flat, kx/vx (B, npoints_in, nh*C) and qy (B, npoints_out, nh*C_k); the
    // arcs are keyed per output point; ring_base/ring_size are the input ring tables and
    // ring_weights one float32 weight per input ring. That the ring sizes sum to
    // npoints_in is not checked: it would read ring_size's data, a sync on a GPU.
    inline void check_ragged_attention_inputs(const at::Tensor &kx, const at::Tensor &vx, const at::Tensor &qy,
                                              const at::Tensor &ring_weights, const at::Tensor &psi_seg,
                                              const at::Tensor &psi_seg_off, const at::Tensor &ring_base,
                                              const at::Tensor &ring_size, int64_t num_heads, int64_t npoints_out)
    {
        TORCH_CHECK(kx.dim() == 3 && vx.dim() == 3 && qy.dim() == 3,
                    "kx, vx and qy must be 3-D (B, npoints, channels), got ", kx.dim(), ", ", vx.dim(), " and ",
                    qy.dim(), " dims");
        TORCH_CHECK(num_heads >= 1, "num_heads must be positive, got ", num_heads);

        check_same_device(vx, kx, "vx");
        check_same_device(qy, kx, "qy");
        check_same_device(ring_weights, kx, "ring_weights");
        check_same_device(psi_seg, kx, "psi_seg");
        check_same_device(psi_seg_off, kx, "psi_seg_off");
        check_same_device(ring_base, kx, "ring_base");
        check_same_device(ring_size, kx, "ring_size");
        check_same_dtype(kx, qy, "k", "q");
        check_same_dtype(vx, qy, "v", "q");

        TORCH_CHECK(vx.size(0) == kx.size(0) && vx.size(1) == kx.size(1), "vx must share kx's (B, npoints_in), got ",
                    vx.sizes(), " against ", kx.sizes());
        TORCH_CHECK(qy.size(0) == kx.size(0) && qy.size(1) == npoints_out, "qy must be (", kx.size(0), ", ",
                    npoints_out, ", channels), got ", qy.sizes());
        TORCH_CHECK(qy.size(2) == kx.size(2), "q and k must have the same channel count, got ", qy.size(2), " and ",
                    kx.size(2));
        TORCH_CHECK(qy.size(2) % num_heads == 0 && vx.size(2) % num_heads == 0, "channel counts (q/k ", qy.size(2),
                    ", v ", vx.size(2), ") must be divisible by num_heads (", num_heads, ")");

        TORCH_CHECK(ring_base.scalar_type() == at::kLong && ring_base.dim() == 1 && ring_size.scalar_type() == at::kLong
                        && ring_size.dim() == 1 && ring_base.size(0) == ring_size.size(0),
                    "ring_base and ring_size must be int64 vectors of equal length, got ", ring_base.scalar_type(), " ",
                    ring_base.sizes(), " and ", ring_size.scalar_type(), " ", ring_size.sizes());
        // ring_weights and the softmax statistics are read as float32 whatever the activations
        // are; casting a whole module to bf16 would sweep them along, and reinterpreting those
        // bytes as float would compute silent garbage instead of failing
        TORCH_CHECK(ring_weights.scalar_type() == at::kFloat && ring_weights.dim() == 1
                        && ring_weights.size(0) == ring_base.size(0),
                    "ring_weights must be a float32 tensor of one weight per input ring (", ring_base.size(0),
                    "), got ", ring_weights.scalar_type(), " of shape ", ring_weights.sizes());

        check_arc_pattern(psi_seg, psi_seg_off, npoints_out);

        check_dense(kx, "kx");
        check_dense(vx, "vx");
        check_dense(qy, "qy");
        check_dense(ring_weights, "ring_weights");
        check_dense(ring_base, "ring_base");
        check_dense(ring_size, "ring_size");
    }

    // What backward_ragged reads beyond the forward's operands: dy and y, shaped like the
    // output (B, npoints_out, nh*C_v); y_hi, the fp32 output, which may be empty; and the
    // forward's softmax statistics, float32 (B, nh, npoints_out).
    inline void check_ragged_backward_state(const at::Tensor &dy, const at::Tensor &y, const at::Tensor &y_hi,
                                            const at::Tensor &alpha_sum, const at::Tensor &qdotk_max,
                                            const at::Tensor &kx, const at::Tensor &vx, const at::Tensor &qy,
                                            int64_t num_heads)
    {
        check_output_grad(dy, kx, vx, qy);
        check_same_device(y, kx, "y");
        check_same_dtype(y, qy, "y", "q");
        TORCH_CHECK(y.sizes() == dy.sizes(), "y must have dy's shape, got ", y.sizes(), " against ", dy.sizes());
        check_dense(y, "y");
        if (y_hi.defined() && y_hi.numel() > 0) {
            check_same_device(y_hi, kx, "y_hi");
            TORCH_CHECK(y_hi.scalar_type() == at::kFloat && y_hi.sizes() == y.sizes(),
                        "y_hi must be empty or a float32 tensor of y's shape, got ", y_hi.scalar_type(), " of shape ",
                        y_hi.sizes());
            check_dense(y_hi, "y_hi");
        }
        const std::vector<int64_t> stat_shape {qy.size(0), num_heads, qy.size(1)};
        check_state_buffer(alpha_sum, kx, "alpha_sum", stat_shape);
        check_state_buffer(qdotk_max, kx, "qdotk_max", stat_shape);
    }

} // namespace attention_kernels
