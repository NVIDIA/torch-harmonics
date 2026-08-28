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
        "_C", /* name of module */
        NULL, /* module documentation, may be NULL */
        -1,   /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
        NULL, /* methods */
    };
    return PyModule_Create(&module_def);
}
}

namespace attention_kernels
{

    // Declare the operators
    //
    // Convention used across all of these ops:
    //   B          batch size
    //   C_k, C_v   channel counts for K/Q (= C_k) and V/output (= C_v)
    //   nlat_in    K/V latitude  count   (input-side grid)
    //   nlon_in    K/V longitude count
    //   nlat_out   Q   latitude  count   (output-side grid)
    //   nlon_out   Q   longitude count
    //   psi (col_idx + row_off) encodes the spherical neighborhood pattern. The
    //   exact indexing convention depends on the op family — see each block below.
    //
    TORCH_LIBRARY(attention_kernels, m)
    {
        // ---- Layout conversion ----
        // Single point of truth for NCHW <-> NHWC conversion in the attention
        // stack. Every attention kernel operates on physical NHWC (channel
        // innermost) data, so layout is converted explicitly at the module
        // boundary rather than inferred from strides inside each launcher --
        // stride inspection cannot distinguish the two layouts when a dimension
        // is degenerate (a contiguous NCHW tensor with H*W == 1 has stride(1)
        // == 1 and is indistinguishable from NHWC).
        //
        // Both directions are pure element permutations, hence exact inverses
        // of each other and dtype-agnostic: fp32, fp16 and bf16 all dispatch to
        // the same tiled transpose. The autograd rule (registered in Python,
        // see attention/_layout.py) is therefore just the opposite direction.
        //   permute_to_nhwc : (B, C, H, W) contiguous -> (B, H, W, C) contiguous
        //   permute_to_nchw : (B, H, W, C) contiguous -> (B, C, H, W) contiguous
        m.def("permute_to_nhwc(Tensor x) -> Tensor", {at::Tag::pt2_compliant_tag});
        m.def("permute_to_nchw(Tensor x) -> Tensor", {at::Tag::pt2_compliant_tag});

        // ---- Self-attention / downsample (output-centric gather) ----
        // Standard direction: each Q point at (ho, wo) gathers from a neighborhood
        // of K/V points. K/V are at the higher resolution (or equal).
        //
        // LAYOUT: all activation tensors are physical NHWC (channel innermost) and
        // contiguous, with the attention heads packed along the channel dimension.
        // C_k / C_v below denote the PER-HEAD channel counts, so the extent of the
        // last dimension is num_heads * C. Layout is part of the contract and is
        // never inferred from strides -- see attention/_layout.py, which owns the
        // conversion, and note that stride inspection cannot distinguish the two
        // layouts when a dimension is degenerate.
        //
        // Heads are packed rather than folded into the batch dimension because in
        // a channel-innermost layout the head axis is interior: folding it to the
        // front would require materializing a copy, whereas the kernels can address
        // a head in place via a leading dimension (num_heads * C) and an offset.
        //   kx, vx : [B, nlat_in,  nlon_in,  num_heads * C_k / C_v]
        //   qy     : [B, nlat_out, nlon_out, num_heads * C_k]
        //   y      : [B, nlat_out, nlon_out, num_heads * C_v]
        // psi convention (canonical at wo=0):
        //   row_off : indexed by ho in [0, nlat_out],  length nlat_out + 1
        //   col_idx : hi * nlon_in + wi_canonical
        //             (input-lon offset for the canonical wo=0; the kernel
        //              applies the integer p-shift  wip = wi + pscale*wo  internally,
        //              where pscale = nlon_in / nlon_out).
        // Requires nlon_in % nlon_out == 0.
        // seg / seg_off are the contiguous-arc form of psi (see _build_psi_segments):
        // seg is (nsegs, 3) int32 holding (input_lat, lon_start, arc_len), seg_off maps
        // an output row to its segment range. The CUDA kernels use them instead of
        // col_idx; col_idx and row_off stay because the CPU and torch reference paths
        // still consume them -- which is what keeps the reference independent.
        m.def("forward(Tensor kx, Tensor vx, Tensor qy, Tensor quad_weights, Tensor col_idx, Tensor row_off, "
              "Tensor seg, Tensor seg_off, int num_heads, int nlon_in, int nlat_out, int nlon_out) -> Tensor",
              {at::Tag::pt2_compliant_tag});
        m.def("backward(Tensor kx, Tensor vx, Tensor qy, Tensor dy, Tensor quad_weights, Tensor col_idx, Tensor "
              "row_off, Tensor seg, Tensor seg_off, int num_heads, int nlon_in, int nlat_out, int nlon_out) -> "
              "(Tensor, Tensor, Tensor)",
              {at::Tag::pt2_compliant_tag});

        // ---- Ring-step variants for DistributedNeighborhoodAttentionS2 ----
        // K/V are sharded along longitude across an azimuth process group; each
        // step processes one rotating chunk and accumulates state buffers for
        // online softmax. lon_lo_kx is the global longitude offset of the
        // currently-held kx/vx chunk; pscale is the GLOBAL nlon_in / nlon_out
        // (must be passed explicitly, since the kernel's own nlon_out arg is the
        // LOCAL output width and would give the wrong ratio when az_size > 1).
        // col_idx must have wi pre-shifted by pscale * lon_lo_out — see
        // _build_local_psi in distributed_attention.py.
        // split_csr_rows precomputes the long/short row split for a fixed psi
        // (returns n_long_rows, max_row_len, mid_row_len). Called once from the
        // module constructor; the result is passed into the ring-step ops below
        // as the trailing (n_long_rows, max_row_len, mid_row_len) ints, hoisting
        // it out of the per-step hot path (where it otherwise cost a 24-byte D2H
        // sync/step). NOT pt2-compliant and intentionally has no fake/meta impl:
        // it is a setup-time host-side scalar computation that is never traced.
        m.def("split_csr_rows(Tensor row_idx, Tensor row_off, int nlat_out) -> (int, int, int)");
        m.def("forward_ring_step(Tensor kx, Tensor vx, Tensor qy, Tensor(a!) y_acc, Tensor(b!) alpha_sum_buf, "
              "Tensor(c!) qdotk_max_buf, Tensor quad_weights, Tensor col_idx, Tensor row_off, Tensor row_idx, int "
              "nlon_in, int pscale, int lon_lo_kx, int lat_halo_start, int nlat_out, int nlon_out, int n_long_rows, "
              "int max_row_len, int mid_row_len) -> ()",
              {at::Tag::pt2_compliant_tag});
        // Backward is split into two passes: pass1 finalizes per-output softmax
        // statistics (alpha_sum, qdotk_max, integral, alpha_k, alpha_kvw), pass2
        // scatters dkx/dvx into the current chunk using those finalized stats.
        m.def("backward_ring_step_pass1(Tensor kx, Tensor vx, Tensor qy, Tensor dy, Tensor(a!) alpha_sum_buf, "
              "Tensor(b!) qdotk_max_buf, Tensor(c!) integral_buf, Tensor(d!) alpha_k_buf, Tensor(e!) alpha_kvw_buf, "
              "Tensor quad_weights, Tensor col_idx, Tensor row_off, Tensor row_idx, int nlon_in, int pscale, int "
              "lon_lo_kx, int lat_halo_start, int nlat_out, int nlon_out, int n_long_rows, int max_row_len, int "
              "mid_row_len) -> ()",
              {at::Tag::pt2_compliant_tag});
        m.def("backward_ring_step_pass2(Tensor kx, Tensor vx, Tensor qy, Tensor dy, Tensor alpha_sum_buf, Tensor "
              "qdotk_max_buf, Tensor integral_norm_buf, Tensor(a!) dkx, Tensor(b!) dvx, Tensor quad_weights, Tensor "
              "col_idx, Tensor row_off, Tensor row_idx, int nlon_in, int pscale, int lon_lo_kx, int lat_halo_start, "
              "int nlat_out, int nlon_out, int n_long_rows, int max_row_len, int mid_row_len) -> ()",
              {at::Tag::pt2_compliant_tag});

        // ---- Ring-step variants for the UPSAMPLE (input-keyed scatter) direction ----
        // Used by DistributedNeighborhoodAttentionS2 when nlon_out % nlon_in == 0.
        // K/V live on the coarse input grid and rotate along the azimuth ring; Q and
        // the softmax state buffers live on the fine output grid and stay local.
        // psi convention (see _build_local_psi_upsample in distributed_attention.py):
        //   row_off : indexed by hi_local in [0, nlat_halo] (halo-padded local input
        //             rows; hi_global = lat_halo_start + hi_local)
        //   col_idx : ho_local * nlon_out_global + wo_shifted, with
        //             wo_shifted = (wo_canonical - lon_lo_out) mod nlon_out_global.
        // The kernel maps wo_shifted -> (wo_shifted + pscale_out * (lon_lo_kx + wi_local))
        // mod nlon_out_global and treats the cell as local iff the result < nlon_out
        // (the LOCAL output width). pscale_out is the GLOBAL nlon_out / nlon_in.
        // A single forward step runs the 3-phase max/rescale/accumulate scheme so the
        // online softmax stays consistent across ring steps despite the scatter form.
        m.def("forward_ring_step_upsample(Tensor kx, Tensor vx, Tensor qy, Tensor(a!) y_acc, Tensor(b!) "
              "alpha_sum_buf, Tensor(c!) qdotk_max_buf, Tensor quad_weights, Tensor col_idx, Tensor row_off, int "
              "nlon_in, int nlon_out_global, int pscale_out, int lon_lo_kx, int lat_halo_start, int nlat_out, int "
              "nlon_out) -> ()",
              {at::Tag::pt2_compliant_tag});
        // Backward reuses the forward-final alpha_sum / qdotk_max (no max recompute):
        // pass1 scatters the per-output stats (integral, alpha_k, alpha_kvw) needed for
        // dqy; pass2 accumulates chunk-local dkx/dvx (allreduced in Python).
        m.def("backward_ring_step_upsample_pass1(Tensor kx, Tensor vx, Tensor qy, Tensor dy, Tensor qdotk_max_buf, "
              "Tensor(a!) integral_buf, Tensor(b!) alpha_k_buf, Tensor(c!) alpha_kvw_buf, Tensor quad_weights, Tensor "
              "col_idx, Tensor row_off, int nlon_in, int nlon_out_global, int pscale_out, int lon_lo_kx, int "
              "lat_halo_start, int nlat_out, int nlon_out) -> ()",
              {at::Tag::pt2_compliant_tag});
        m.def("backward_ring_step_upsample_pass2(Tensor kx, Tensor vx, Tensor qy, Tensor dy, Tensor alpha_sum_buf, "
              "Tensor qdotk_max_buf, Tensor integral_norm_buf, Tensor(a!) dkx, Tensor(b!) dvx, Tensor quad_weights, "
              "Tensor col_idx, Tensor row_off, int nlon_in, int nlon_out_global, int pscale_out, int lon_lo_kx, int "
              "lat_halo_start, int nlat_out, int nlon_out) -> ()",
              {at::Tag::pt2_compliant_tag});
    }

} // namespace attention_kernels
