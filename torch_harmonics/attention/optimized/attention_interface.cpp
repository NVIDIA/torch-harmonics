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
    //   psi (seg + seg_off) encodes the spherical neighborhood pattern as contiguous
    //   longitude arcs. The exact indexing convention depends on the op family — see
    //   each block below.
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
        //   seg_off : indexed by ho in [0, nlat_out],  length nlat_out + 1
        //   seg     : (hi, wi_canonical, len) arcs
        //             (input-lon start for the canonical wo=0; the kernel
        //              applies the integer p-shift  wip = wi + pscale*wo  internally,
        //              where pscale = nlon_in / nlon_out).
        // Requires nlon_in % nlon_out == 0.
        // seg / seg_off are the contiguous-arc form of psi: seg is (nsegs, 3) int32
        // holding (input_lat, lon_start, arc_len), and seg_off maps an output row to its
        // segment range. Both the CUDA and the CPU kernels read this form, so it is the
        // only one declared here. The torch reference takes the column list instead, and
        // has its own operator -- which is what keeps it independent of this derivation
        // rather than merely documented as being so.
        m.def("forward_regular(Tensor kx, Tensor vx, Tensor qy, Tensor ring_weights, "
              "Tensor seg, Tensor seg_off, int num_heads, int nlon_in, int nlat_out, int nlon_out) -> Tensor",
              {at::Tag::pt2_compliant_tag});
        m.def("backward_regular(Tensor kx, Tensor vx, Tensor qy, Tensor dy, Tensor ring_weights, "
              "Tensor seg, Tensor seg_off, int num_heads, int nlon_in, int nlat_out, int nlon_out) -> "
              "(Tensor, Tensor, Tensor)",
              {at::Tag::pt2_compliant_tag});

        // Ragged counterparts, for a grid whose rings differ in length (HEALPix, reduced
        // Gaussian). Two differences from the regular schemas above, both forced by the
        // absence of longitudinal translation invariance:
        //
        //  - there is no (nlat, nlon) to address by, so the extent is one npoints_out and
        //    the ring tables carry what nlon used to give arithmetically;
        //  - the forward returns its softmax bookkeeping (y_hi, alpha_sum, qdotk_max)
        //    rather than recomputing it, because the backward walks a neighbour list it
        //    cannot cheaply re-derive. These are fp32 whatever the activations are.
        m.def("forward_ragged(Tensor kx, Tensor vx, Tensor qy, Tensor ring_weights, Tensor psi_seg, "
              "Tensor psi_seg_off, Tensor ring_base, Tensor ring_size, int num_heads, int npoints_out) -> "
              "(Tensor, Tensor, Tensor, Tensor)",
              {at::Tag::pt2_compliant_tag});
        m.def("backward_ragged(Tensor kx, Tensor vx, Tensor qy, Tensor dy, Tensor y, Tensor y_hi, Tensor alpha_sum, "
              "Tensor qdotk_max, Tensor ring_weights, Tensor psi_seg, Tensor psi_seg_off, Tensor ring_base, "
              "Tensor ring_size, int num_heads, int npoints_out) -> (Tensor, Tensor, Tensor)",
              {at::Tag::pt2_compliant_tag});

        // ---- Ring-step variants for DistributedNeighborhoodAttentionS2 ----
        // The serial ops above, one K/V chunk at a time. K/V are sharded along
        // longitude across an azimuth process group and rotate around it; each step
        // folds the chunk it holds into online-softmax state that persists across
        // steps. Everything the serial ops take is taken here in the same form -- the
        // NHWC ABI with heads packed along channels, the arc form of psi -- plus:
        //   - the state buffers, (B, H, W, num_heads * C) for the per-channel ones and
        //     (B, num_heads, H, W) for the softmax statistics;
        //   - where the chunk sits in the global grid: lon_lo_kx for its first
        //     longitude, lat_halo_start for its first (halo) latitude;
        //   - nlon_out_global, because nlat_out / nlon_out are this rank's extents and
        //     the serial ops' p-shift ratio needs the global one when az_size > 1.
        // All six take the same integer arguments.
        // seg / seg_off are this rank's output rows of the serial arcs, with the arc
        // starts pre-shifted by pscale * lon_lo_out; see RingGatherBackend in
        // distributed_attention.py.
        m.def("forward_ring_step(Tensor kx, Tensor vx, Tensor qy, Tensor(a!) y_acc, Tensor(b!) alpha_sum_buf, "
              "Tensor(c!) qdotk_max_buf, Tensor ring_weights, Tensor seg, Tensor seg_off, int num_heads, int nlon_in, "
              "int nlon_out_global, int lon_lo_kx, int lat_halo_start, int nlat_out, int nlon_out) -> ()",
              {at::Tag::pt2_compliant_tag});
        // The serial backward's two neighbour loops, one pass each: pass1 accumulates
        // the softmax statistics (alpha_sum, qdotk_max, integral, alpha_k, alpha_kvw)
        // across the ring, pass2 scatters dkx/dvx into the current chunk using them
        // once final.
        m.def("backward_ring_step_pass1(Tensor kx, Tensor vx, Tensor qy, Tensor dy, Tensor(a!) alpha_sum_buf, "
              "Tensor(b!) qdotk_max_buf, Tensor(c!) integral_buf, Tensor(d!) alpha_k_buf, Tensor(e!) alpha_kvw_buf, "
              "Tensor ring_weights, Tensor seg, Tensor seg_off, int num_heads, int nlon_in, int nlon_out_global, int "
              "lon_lo_kx, int lat_halo_start, int nlat_out, int nlon_out) -> ()",
              {at::Tag::pt2_compliant_tag});
        m.def("backward_ring_step_pass2(Tensor kx, Tensor vx, Tensor qy, Tensor dy, Tensor alpha_sum_buf, Tensor "
              "qdotk_max_buf, Tensor integral_norm_buf, Tensor(a!) dkx, Tensor(b!) dvx, Tensor ring_weights, Tensor "
              "seg, Tensor seg_off, int num_heads, int nlon_in, int nlon_out_global, int lon_lo_kx, int "
              "lat_halo_start, int nlat_out, int nlon_out) -> ()",
              {at::Tag::pt2_compliant_tag});

        // ---- Ring-step variants for the UPSAMPLE (input-keyed scatter) direction ----
        // Used by DistributedNeighborhoodAttentionS2 when nlon_out % nlon_in == 0.
        // K/V live on the coarse input grid and rotate along the azimuth ring; Q and
        // the softmax state buffers live on the fine output grid and stay local.
        // psi is the serial scatter arc form, sliced to this rank (see
        // RingUpsampleBackend in distributed_attention.py):
        //   seg_off : indexed by hi_local in [0, nlat_halo] (halo-padded local input
        //             rows; hi_global = lat_halo_start + hi_local)
        //   seg     : (ho_local, lo, len) arcs, only those on local output rows, with
        //             lo pre-shifted by -lon_lo_out (mod nlon_out_global).
        // The serial shift of an arc start by pscale_out * wi then lands relative to
        // lon_lo_out, and the arc is clipped to the LOCAL output width nlon_out.
        // A single forward step runs the 3-phase max/rescale/accumulate scheme so the
        // online softmax stays consistent across ring steps despite the scatter form.
        m.def("forward_ring_step_upsample(Tensor kx, Tensor vx, Tensor qy, Tensor(a!) y_acc, Tensor(b!) alpha_sum_buf, "
              "Tensor(c!) qdotk_max_buf, Tensor ring_weights, Tensor seg, Tensor seg_off, int num_heads, int nlon_in, "
              "int nlon_out_global, int lon_lo_kx, int lat_halo_start, int nlat_out, int nlon_out) -> ()",
              {at::Tag::pt2_compliant_tag});
        // Backward reuses the forward-final alpha_sum / qdotk_max (no max recompute):
        // pass1 scatters the per-output stats (integral, alpha_k, alpha_kvw) needed for
        // dqy; pass2 writes the chunk-local dkx/dvx, which Python accumulates.
        m.def("backward_ring_step_upsample_pass1(Tensor kx, Tensor vx, Tensor qy, Tensor dy, Tensor qdotk_max_buf, "
              "Tensor(a!) integral_buf, Tensor(b!) alpha_k_buf, Tensor(c!) alpha_kvw_buf, Tensor ring_weights, Tensor "
              "seg, Tensor seg_off, int num_heads, int nlon_in, int nlon_out_global, int lon_lo_kx, int "
              "lat_halo_start, int nlat_out, int nlon_out) -> ()",
              {at::Tag::pt2_compliant_tag});
        m.def("backward_ring_step_upsample_pass2(Tensor kx, Tensor vx, Tensor qy, Tensor dy, Tensor alpha_sum_buf, "
              "Tensor qdotk_max_buf, Tensor integral_norm_buf, Tensor(a!) dkx, Tensor(b!) dvx, Tensor ring_weights, "
              "Tensor seg, Tensor seg_off, int num_heads, int nlon_in, int nlon_out_global, int lon_lo_kx, int "
              "lat_halo_start, int nlat_out, int nlon_out) -> ()",
              {at::Tag::pt2_compliant_tag});
    }

} // namespace attention_kernels
