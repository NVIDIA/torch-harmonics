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

// Backward neighborhood attention on a RAGGED isolatitude grid (HEALPix).
//
// Relation to the product-grid kernel in attention_cuda_bwd.cu
// ------------------------------------------------------------
// The mathematics and the scatter strategy are taken over verbatim; only the
// addressing changes, in exactly the same way as in attention_cuda_fwd_ragged.cu.
// The pattern is keyed per output POINT rather than per output LATITUDE, so there
// is no p-shift (wip = wi + pscale*wo) and no (ho, wo) decomposition -- a neighbour
// is located as ring_base[iring] + offset, counting along the arc.
//
// Why the scatter strategy needs no rethinking
// --------------------------------------------
// dk/dv are scatter-accumulated: neighbourhoods of distinct output points overlap,
// so several output points contribute to the same input point. The product-grid
// kernel resolves that with atomicAdd into fp32 buffers, and nothing about that
// choice depends on the pattern being a product grid -- it depends only on the
// pattern being many-to-many, which is equally true here. Raggedness changes which
// input point an output point lands on, not how many of them collide. So this is a
// port, not a redesign: no transposed (input-keyed) pattern is built, and no second
// precompute is needed.
//
// One pass over the arcs
// ----------------------
// This kernel used to walk each neighbourhood twice. Pass 1 accumulated the running-
// softmax statistics (alpha_sum, qdotk_max, the two per-channel reductions and the
// scalar alpha_vw_) and wrote dqy; pass 2 replayed the arcs against the final
// qdotk_max to scatter dk/dv. The replay is what avoided materialising the per-
// neighbour alphas, which on a HEALPix neighbourhood would be a far larger array than
// on a product grid, since the pattern here is npoints_out/nlat_out ~ 3*nside times
// bigger -- recomputing q.k is cheaper than storing it.
//
// Both walks are now one, on three identities. Writing p_i = alpha_i / alpha_sum for
// the softmax weight of neighbour i, gdotv_i = dy . v_i, and out for the forward
// output at this point:
//
//   (1)  integral = sum_i p_i gdotv_i = dy . out,
//
//        since out = sum_i p_i v_i. So the quantity pass 1 spent a traversal on costs
//        O(nchan) given the forward output, which is FlashAttention's D = rowsum(dO*O)
//        and is formed in torch before the launch.
//
//   (2)  dqy = (alpha_sum * loc_kvw - alpha_vw_ * loc_k__) / alpha_sum^2
//            = sum_i p_i (gdotv_i - integral) k_i,
//
//        by substituting loc_kvw = sum_i alpha_i gdotv_i k_i, loc_k__ = sum_i alpha_i
//        k_i and alpha_vw_ = integral * alpha_sum. So dqy is a plain accumulation over
//        neighbours once integral is known up front, rather than three reductions
//        combined after the fact.
//
//   (3)  the scatter's own multipliers are (gdotv_i - integral) * p_i for dk and p_i
//        for dv -- the same two numbers (2) needs.
//
// So given qdotk_max, alpha_sum and integral before the walk begins, dqy and the
// dk/dv scatter are functions of identical per-neighbour quantities and belong in one
// traversal. The first two come from the forward, which held both and threw them away
// (see attention_cuda_fwd_ragged.cu); the third comes from (1).
//
// The identities are exact, so the only thing that changes is which fp32 sums happen
// in which order, and gradients differ in their last bits. TORCH_HARMONICS_RAGGED_BWD_
// TWO_PASS=1 restores the two-pass formulation for exactly that reason.
//
// Both variants of the product-grid file are now provided, for the reason the
// forward gave: the generic kernel keeps one neighbour in flight at a time and
// re-reads qy and dy from shared memory on every one of them, and it measures like
// it -- the ragged backward is 24 ms of the 33 ms fwd+bwd at HEALPix level 5, at 1.6
// TFLOP/s, so bound by the latency of a dependency chain 102 neighbours long rather
// than by arithmetic or bandwidth. s2_attn_bwd_ragged_special_vec_k is the port of
// s2_attn_bwd_special_vec_k, which moves the four per-channel reductions and the
// staged qy/dy into registers; shared memory survives only for the pre-9.0 atomic
// epilogue. It applies when the per-head channel count fits NLOC registers per lane
// and q/k and v agree on it, and the generic kernel remains for everything else and
// as an escape hatch, since only the last of NLOC registers is bounds-checked and
// that argument depends on NLOC being exactly DIV_UP(nchan, BDIM_X).

#include "../common/attention_cuda.cuh"
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include "c10/core/MemoryFormat.h"

#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>
#include <cfloat>
#include <cstdlib>

#include "../common/cudamacro.h"
#include "../common/attention_cuda_utils.cuh"

// Threads per block, and so warps per block, since BDIM_X is a warp. Overridable for
// the reason given in attention_cuda_fwd_ragged.cu: warps in a block take consecutive
// output points and adjacent points share 74% of their neighbours by measurement, so a
// wider block turns L1 into a shared cache for that overlap. The backward used to have
// twice as much to gain from it, since it walked each neighbourhood twice; that half of
// the argument is gone. Kept separate from the forward's knob because the two kernels
// have different register pressure.
#ifndef TH_ATTENTION_RAGGED_BWD_THREADS
#define TH_ATTENTION_RAGGED_BWD_THREADS (64)
#endif
#define THREADS (TH_ATTENTION_RAGGED_BWD_THREADS)

// Neighbours per group in the special kernel's arc walk, the backward's counterpart of
// TH_ATTENTION_RAGGED_NB.
//
// Neither the product-grid backward nor the first ragged port grouped at all, so
// both walked one neighbour at a time: address, load k and v, two warp reductions,
// softmax update, next. That is the same fully serial chain the forward was
// measured at 1.6 TFLOP/s with, and after the forward was fixed the backward is
// what remains -- 4.05 ms of the 5.36 ms fwd+bwd at nside 64, i.e. 76%.
//
// Setting this to 1 recovers the ungrouped kernel exactly, which is the escape
// hatch if the grouped arithmetic ever looks suspect; TORCH_HARMONICS_RAGGED_BWD_
// TWO_PASS=1 and TORCH_HARMONICS_RAGGED_BWD_GENERIC=1 are the coarser ones, and
// unlike this constant they need no rebuild.
//
// Unlike the forward, the choice here is a real trade rather than a free lunch, and
// collapsing the two walks into one made it a sharper one: the single-pass body holds
// the reductions, dqy's accumulator and the scatter's addresses at the same time, where
// the two-pass kernel spread them over two loops whose register peaks did not overlap.
// For nchan 96 (NLOC 3), fp32, sm_100a:
//
//   NB   single pass                two pass
//        registers spill warps/SM   registers spill warps/SM
//    1       56       0      36        56        0      36
//    2       72       0      28        64        0      32
//    4       96       0      21        72        0      28
//
// So a doubling costs the single-pass kernel about 20 registers against the two-pass
// kernel's 8, and NB 4 runs at 33% occupancy rather than 44%. Nothing spills at the
// real workload either way, which is the thing that would have made grouping
// pointless; the instantiations that do spill are all double above NLOC 14.
//
// 4 stays the default on the usual argument that memory-level parallelism per thread
// beats occupancy for a latency-bound kernel, and the single pass halves the traversals
// that latency is spent on before NB is considered at all. But that argument is now
// carrying more weight than it was asked to, and it remains a prediction: NB is the one
// number here a GPU still has to settle, more so than before. Sweeping needs no source
// edit:
//
//   NVCC_APPEND_FLAGS="-DTH_ATTENTION_RAGGED_BWD_NB=2" python setup.py build_ext --inplace
//
// Both formulations measuring 56 registers at NB 1, which is what the kernel used
// before grouping, is the check that the escape hatch really is the old code.
#ifndef TH_ATTENTION_RAGGED_BWD_NB
#define TH_ATTENTION_RAGGED_BWD_NB (4)
#endif

// Largest number of COMPUTE_T registers per lane the special kernel will hold for
// one accumulator. 16 matches MAX_LOCAL_ARR_LEN in attention_cuda_bwd.cu, so with a
// 32-lane BDIM_X it covers up to 512 channels per head; HEALDA's dit-5B runs 96.
//
// Single pass holds three accumulators of that length, two-pass four, so this is the
// register-hungry half of the pair either way and occupancy here is register-limited.
// See benchmarks/ptxas_register_report.py for the table, and the commentary on
// TH_ATTENTION_RAGGED_BWD_NB above for what it says.
#define MAX_LOCAL_ARR_LEN_RAGGED_BWD (16)

namespace attention_kernels
{

    // One warp per output point, as in the forward. STORAGE_T is the global-memory
    // element type of the inputs (kx/vx/qy/dy); COMPUTE_T is the arithmetic type and
    // the type of the gradient OUTPUTS dkx/dvx/dqy. The gradients stay fp32 because
    // dkx/dvx are atomically scatter-accumulated and reduced-precision atomics would
    // lose precision; the wrapper narrows them back at the end.
    //
    // TWO_PASS selects the formulation, not a different result: false walks each arc
    // once on the identities at the top of the file, true restores the two walks it
    // replaced. It has to be a template parameter rather than a flag because it decides
    // which accumulators exist and how much shared memory a warp needs, so both are
    // compiled and the launcher picks one -- see ragged_bwd_two_pass().
    template <int BDIM_X, bool TWO_PASS, typename STORAGE_T>
    __global__ __launch_bounds__(BDIM_X) void s2_attn_bwd_ragged_generic_vec_k(
        int nheads,     // no. of attention heads packed along the channel dim
        int nchans_in,  // no. of STORAGE_T elements along channel dim, per head
        int nchans_out, // no. of STORAGE_T elements along channel dim, per head
        int64_t npoints_in, int64_t npoints_out,
        const STORAGE_T *__restrict__ kx, // [batch][npoints_in][nheads * nchan_in]
        const STORAGE_T *__restrict__ vx, // [batch][npoints_in][nheads * nchan_out]
        const STORAGE_T *__restrict__ qy, // [batch][npoints_out][nheads * nchan_in]
        const STORAGE_T *__restrict__ dy, // [batch][npoints_out][nheads * nchan_out]
        // The forward's softmax statistics and integral = dy . out, all per
        // (batch, head, point) and fp32. Read only when TWO_PASS is false, which is
        // the point of them: they are what the first walk used to produce.
        const float *__restrict__ alpha_sum_fwd, // [batch][nheads][npoints_out]
        const float *__restrict__ qdotk_max_fwd, // [batch][nheads][npoints_out]
        const float *__restrict__ integral_fwd,  // [batch][nheads][npoints_out]
        const int32_t *__restrict__ seg, const int32_t *__restrict__ seg_off, const int64_t *__restrict__ ring_base,
        const int64_t *__restrict__ ring_size, const float *__restrict__ ring_weights,
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dkx, // [batch][npoints_in][nheads * nchan_in]
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dvx, // [batch][npoints_in][nheads * nchan_out]
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dqy) // [batch][npoints_out][nheads * nchan_in]
    {
        using COMPUTE_T = typename vec_traits<STORAGE_T>::compute_t;

        extern __shared__ __align__(sizeof(float4)) float shext[];

        // Per-warp arrays: NACC accumulators for dqy of nchans_in entries each, then dy
        // and qy. So (NACC + 1) * nchans_in + nchans_out.
        //
        // The two-pass formulation needs two accumulators, sum_i alpha_i k_i and
        // sum_i alpha_i gdotv_i k_i, and combines them once the statistics are final.
        // Identity (2) at the top of the file says that combination is
        // sum_i p_i (gdotv_i - integral) k_i, which a single pass accumulates directly,
        // so it needs one.
        //
        // The product-grid kernels carry a further array, alpha_vw_, one entry per
        // channel. Its recurrence is alpha_vw_ = alpha_vw_ * max_correction +
        // ainz_gdotv and both of those are warp-uniform scalars, so every entry holds
        // the same number and always has: it is a scalar accumulator stored nchans_in
        // times, and 6253cae made it a register. Identity (2) removes it outright --
        // alpha_vw_ is integral * alpha_sum, so with integral known up front there is
        // nothing left for it to carry. (attention_cuda_bwd.cu replicates it in both
        // its generic and its register-blocked kernel; the redundancy is not ragged's.)
        constexpr int NACC = TWO_PASS ? 2 : 1;

        COMPUTE_T *sh_acc = reinterpret_cast<COMPUTE_T *>(shext) + threadIdx.y * (nchans_in * (NACC + 1) + nchans_out);

        COMPUTE_T *sh_dy = sh_acc + nchans_in * NACC;
        COMPUTE_T *sh_qy = sh_dy + nchans_out;

        const int bh = blockIdx.y;
        const int batch = bh / nheads;
        const int head = bh - (batch * nheads);

        // leading dimensions: elements between adjacent spatial points
        const int64_t ldi = int64_t(nheads) * nchans_in;
        const int64_t ldo = int64_t(nheads) * nchans_out;

        const int64_t ipoint = int64_t(blockIdx.x) * blockDim.y + threadIdx.y;

        if (ipoint >= npoints_out) { return; }

        const int tidx = threadIdx.x;

        // No row_idx indirection, for the reason given in the forward: the geodesic
        // neighbourhood of an equal-area grid has near-uniform neighbour counts, so
        // sorting output rows by length would cost a device-side sort per call to
        // balance nothing.

        // offset input tensors
        kx += int64_t(batch) * npoints_in * ldi + int64_t(head) * nchans_in;
        vx += int64_t(batch) * npoints_in * ldo + int64_t(head) * nchans_out;

        qy += int64_t(batch) * npoints_out * ldi + int64_t(head) * nchans_in + ipoint * ldi;
        dy += int64_t(batch) * npoints_out * ldo + int64_t(head) * nchans_out + ipoint * ldo;

        // offset output tensors (same packed layout as their inputs)
        dkx += int64_t(batch) * npoints_in * ldi + int64_t(head) * nchans_in;
        dvx += int64_t(batch) * npoints_in * ldo + int64_t(head) * nchans_out;
        dqy += int64_t(batch) * npoints_out * ldi + int64_t(head) * nchans_in + ipoint * ldi;

        // zero/init shared memory
        for (int chan = tidx; chan < nchans_in * NACC; chan += WARP_SIZE) { sh_acc[chan] = __vset<COMPUTE_T>(0.0f); }
        for (int chan = tidx; chan < nchans_in; chan += WARP_SIZE) { sh_qy[chan] = vload(qy, chan); }
        for (int chan = tidx; chan < nchans_out; chan += WARP_SIZE) { sh_dy[chan] = vload(dy, chan); }

#if __CUDA_ARCH__ < 900
        // for architectures < 9.0, sh_dy and sh_qy will be read as individual floats
        // at the end of the kernel, which breaks the assumption that each COMPUTE_T
        // location is written to and read by the same thread throughout the kernel,
        // in the case COMPUTE_T==float4
        if constexpr (std::is_same<COMPUTE_T, float4>::value) { __syncwarp(); }
#endif

        // The three warp-uniform scalars the walk below needs, however they are come by:
        // the final running maximum, the softmax normaliser and integral =
        // sum_i p_i gdotv_i.
        float qdotk_max = -FLT_MAX;
        float alpha_sum_inv;
        float integral = 0.0f;

        const int seg_beg = seg_off[ipoint];
        const int seg_end = seg_off[ipoint + 1];

        if constexpr (TWO_PASS) {

            COMPUTE_T *sh_alpha_k__ = sh_acc;
            COMPUTE_T *sh_alpha_kvw = sh_acc + nchans_in;

            float alpha_sum = 0.0f;

            // the scalar that replaces the replicated sh_alpha_vw_ array
            float alpha_vw_ = 0.0f;

            // Pass 1: accumulate alpha_sum, integral and the shared reductions, along
            // with a progressively computed qdotk_max.
            for (int sg = seg_beg; sg < seg_end; sg++) {

                const int iring = seg[3 * sg + 0];
                const int seg_lo = seg[3 * sg + 1];
                const int seg_len = seg[3 * sg + 2];

                // constant along the arc: every point of a ring carries the same
                // quadrature weight
                const float qw_seg = ring_weights[iring];

                // a ring is contiguous in RING order, so the flat column is the ring's
                // base plus an offset that counts up and wraps at the ring's end
                const int64_t ring_lo = ring_base[iring];
                const int64_t ring_hi = ring_lo + ring_size[iring];

                int64_t col = ring_lo + seg_lo;

                for (int j = 0; j < seg_len; j++) {

                    const STORAGE_T *_kx = kx + col * ldi;
                    const STORAGE_T *_vx = vx + col * ldo;

                    COMPUTE_T qdotk_v = __vset<COMPUTE_T>(0.0f);
                    COMPUTE_T gdotv_v = __vset<COMPUTE_T>(0.0f);

                    for (int chan = tidx; chan < nchans_in; chan += WARP_SIZE) {
                        qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[chan], vload(_kx, chan)));
                    }
                    for (int chan = tidx; chan < nchans_out; chan += WARP_SIZE) {
                        gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[chan], vload(_vx, chan)));
                    }

                    const float qdotk = __warp_sum(__vred(qdotk_v));
                    const float gdotv = __warp_sum(__vred(gdotv_v));

                    const float qdotk_max_tmp = max(qdotk_max, qdotk);
                    const float alpha_inz = expf(qdotk - qdotk_max_tmp) * qw_seg;
                    const float max_correction = expf(qdotk_max - qdotk_max_tmp);
                    alpha_sum = alpha_sum * max_correction + alpha_inz;

                    integral = integral * max_correction + alpha_inz * gdotv;

                    const float ainz_gdotv = alpha_inz * gdotv;

                    // same recurrence the per-channel array used to run, once
                    alpha_vw_ = alpha_vw_ * max_correction + ainz_gdotv;

                    for (int chan = tidx; chan < nchans_in; chan += WARP_SIZE) {

                        const COMPUTE_T kxval = vload(_kx, chan);

                        sh_alpha_k__[chan]
                            = __vadd(__vscale(max_correction, sh_alpha_k__[chan]), __vscale(alpha_inz, kxval));
                        sh_alpha_kvw[chan]
                            = __vadd(__vscale(max_correction, sh_alpha_kvw[chan]), __vscale(ainz_gdotv, kxval));
                    }
                    qdotk_max = qdotk_max_tmp;

                    // next point in the arc; wraps at most once
                    if (++col == ring_hi) { col = ring_lo; }
                }
            }

            alpha_sum_inv = 1.0f / alpha_sum;

            integral *= alpha_sum_inv;

            // Write dqy (fp32 output)
            for (int chan = tidx; chan < nchans_in; chan += WARP_SIZE) {

                // __vscale by the scalar, where this used to __vmul by a vector every
                // entry of which held that scalar
                dqy[chan]
                    = __vscale(alpha_sum_inv * alpha_sum_inv,
                               __vsub(__vscale(alpha_sum, sh_alpha_kvw[chan]), __vscale(alpha_vw_, sh_alpha_k__[chan])));
            }

        } else {

            const int64_t istat = int64_t(bh) * npoints_out + ipoint;

            qdotk_max = qdotk_max_fwd[istat];
            integral = integral_fwd[istat];
            alpha_sum_inv = 1.0f / alpha_sum_fwd[istat];
        }

        // The walk. Under TWO_PASS it is the replay of arcs pass 1 has already been
        // over, and dqy is already written, so it only scatters. Otherwise it is the
        // whole backward: identity (3) says the scatter's multipliers are
        // (gdotv_i - integral) * p_i and p_i, and identity (2) says dqy accumulates the
        // first of those times k_i, so one traversal serves all three gradients.
        for (int sg = seg_beg; sg < seg_end; sg++) {

            const int iring = seg[3 * sg + 0];
            const int seg_lo = seg[3 * sg + 1];
            const int seg_len = seg[3 * sg + 2];
            const float qw_seg = ring_weights[iring];

            const int64_t ring_lo = ring_base[iring];
            const int64_t ring_hi = ring_lo + ring_size[iring];

            int64_t col = ring_lo + seg_lo;

            for (int j = 0; j < seg_len; j++) {

                const STORAGE_T *_kx = kx + col * ldi;
                const STORAGE_T *_vx = vx + col * ldo;

                COMPUTE_T qdotk_v = __vset<COMPUTE_T>(0.0f);
                COMPUTE_T gdotv_v = __vset<COMPUTE_T>(0.0f);

                for (int chan = tidx; chan < nchans_in; chan += WARP_SIZE) {
                    qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[chan], vload(_kx, chan)));
                }
                for (int chan = tidx; chan < nchans_out; chan += WARP_SIZE) {
                    gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[chan], vload(_vx, chan)));
                }

                const float qdotk = __warp_sum(__vred(qdotk_v));
                const float gdotv = __warp_sum(__vred(gdotv_v));

                const float alpha_inz = expf(qdotk - qdotk_max) * qw_seg;

                // _dkx / _dvx are COMPUTE_T (fp32) gradient buffers, accumulated
                // atomically: neighbourhoods overlap, so several output points hit
                // the same input point.
                COMPUTE_T *_dkx = dkx + col * ldi;
                COMPUTE_T *_dvx = dvx + col * ldo;

                const float alpha_mul = alpha_inz * alpha_sum_inv;

                const float scale_fact_qy = (gdotv - integral) * alpha_mul;
                const float scale_fact_dy = alpha_mul;

                // dqy by identity (2), in the same traversal: scale_fact_qy is already
                // p_i (gdotv_i - integral), which is exactly the weight k_i carries.
                if constexpr (!TWO_PASS) {
                    for (int chan = tidx; chan < nchans_in; chan += WARP_SIZE) {
                        sh_acc[chan] = __vadd(sh_acc[chan], __vscale(scale_fact_qy, vload(_kx, chan)));
                    }
                }

                // float4, 128-bit atomics are only supported by devices of compute
                // capability 9.x+, so on older devices we resort to 32-bit atomics

#if __CUDA_ARCH__ < 900
                // to use 32-bit operations on consecutive addresses
                float *sh_qy_scl = reinterpret_cast<float *>(sh_qy);
                float *sh_dy_scl = reinterpret_cast<float *>(sh_dy);

                float *_dkx_scl = reinterpret_cast<float *>(_dkx);
                float *_dvx_scl = reinterpret_cast<float *>(_dvx);

                constexpr int VEC_SIZE = sizeof(COMPUTE_T) / sizeof(float);

                // 32-bit, consecutive atomics to glmem;
                // strided atomics results in a severe slowdown
                for (int chan = tidx; chan < nchans_in * VEC_SIZE; chan += WARP_SIZE) {
                    atomicAdd(_dkx_scl + chan, scale_fact_qy * sh_qy_scl[chan]);
                }
                for (int chan = tidx; chan < nchans_out * VEC_SIZE; chan += WARP_SIZE) {
                    atomicAdd(_dvx_scl + chan, scale_fact_dy * sh_dy_scl[chan]);
                }
#else
                // 128-bit, consecutive atomics to glmem
                for (int chan = tidx; chan < nchans_in; chan += WARP_SIZE) {
                    atomicAdd(_dkx + chan, __vscale(scale_fact_qy, sh_qy[chan]));
                }
                for (int chan = tidx; chan < nchans_out; chan += WARP_SIZE) {
                    atomicAdd(_dvx + chan, __vscale(scale_fact_dy, sh_dy[chan]));
                }
#endif
                if (++col == ring_hi) { col = ring_lo; }
            }
        }

        // Write dqy (fp32 output). No reciprocal and no combination: alpha_sum_inv went
        // into every term as it was accumulated.
        if constexpr (!TWO_PASS) {
            for (int chan = tidx; chan < nchans_in; chan += WARP_SIZE) { dqy[chan] = sh_acc[chan]; }
        }

        return;
    }

    // Register-blocked counterpart of the kernel above, and the port of
    // s2_attn_bwd_special_vec_k in attention_cuda_bwd.cu. Everything that differs is
    // addressing, and it gets simpler: the neighbour list is keyed per output point,
    // and an arc is a contiguous run of one ring, so there is no (ho, wo)
    // decomposition, no pscale, no wrap_lon and no row_idx indirection -- just a
    // column that counts and wraps at the ring end.
    //
    // NLOC must be exactly DIV_UP(nchan, BDIM_X). The unrolled loops below leave every
    // register but the last unguarded, which is sound only under that equality: for
    // i <= NLOC-2, i*BDIM_X + tidx <= (NLOC-1)*BDIM_X - 1 < nchan. The launcher
    // enforces it, and requires nchan_in == nchan_out so one NLOC serves both. That
    // equality is also why there is no CHOUT_AS_IN template parameter here: it selects
    // between taking the dy side through the same unrolled loops as the qy side or
    // through a runtime-bounded loop over shared memory, and only the former is
    // reachable once the channel counts are required to agree.
    //
    // TWO_PASS carries the same meaning as in the generic kernel above.
    template <int BDIM_X, int BDIM_Y, int NLOC, bool TWO_PASS, typename STORAGE_T>
    __global__ __launch_bounds__(BDIM_X *BDIM_Y) void s2_attn_bwd_ragged_special_vec_k(
        int nheads,    // no. of attention heads packed along the channel dim
        int nchan_in,  // no. of STORAGE_T elements along channel dim, per head
        int nchan_out, // no. of STORAGE_T elements along channel dim, per head
        int64_t npoints_in, int64_t npoints_out,
        const STORAGE_T *__restrict__ kx,        // [batch][npoints_in][nheads * nchan_in]
        const STORAGE_T *__restrict__ vx,        // [batch][npoints_in][nheads * nchan_out]
        const STORAGE_T *__restrict__ qy,        // [batch][npoints_out][nheads * nchan_in]
        const STORAGE_T *__restrict__ dy,        // [batch][npoints_out][nheads * nchan_out]
        const float *__restrict__ alpha_sum_fwd, // [batch][nheads][npoints_out]
        const float *__restrict__ qdotk_max_fwd, // [batch][nheads][npoints_out]
        const float *__restrict__ integral_fwd,  // [batch][nheads][npoints_out]
        const int32_t *__restrict__ seg, const int32_t *__restrict__ seg_off, const int64_t *__restrict__ ring_base,
        const int64_t *__restrict__ ring_size, const float *__restrict__ ring_weights,
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dkx, // [batch][npoints_in][nheads * nchan_in]
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dvx, // [batch][npoints_in][nheads * nchan_out]
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dqy) // [batch][npoints_out][nheads * nchan_in]
    {
        using COMPUTE_T = typename vec_traits<STORAGE_T>::compute_t;

        static_assert(BDIM_X == WARP_SIZE, "the ragged special kernel reduces with __warp_sum");
        static_assert(NLOC >= 1);

        constexpr int NLOC_M1 = NLOC - 1;
        constexpr int NB = TH_ATTENTION_RAGGED_BWD_NB;
        static_assert(NB >= 1);

        const int tidx = threadIdx.x;

        const int bh = blockIdx.y;
        const int batch = bh / nheads;
        const int head = bh - (batch * nheads);

        const int64_t ldi = int64_t(nheads) * nchan_in;
        const int64_t ldo = int64_t(nheads) * nchan_out;

        const int64_t ipoint = int64_t(blockIdx.x) * blockDim.y + threadIdx.y;

        if (ipoint >= npoints_out) { return; }

        extern __shared__ __align__(sizeof(float4)) float shext[];

        // sh_dy[nchan_out], sh_qy[nchan_in]. Two arrays per warp where the generic
        // kernel needs three or four: the per-channel reductions are in registers now,
        // and these two survive only because the pre-9.0 epilogue undoes the tidx offset
        // and reads them as individual floats, which registers cannot serve.
        COMPUTE_T *sh_dy = reinterpret_cast<COMPUTE_T *>(shext) + threadIdx.y * (nchan_in + nchan_out) + tidx;
        COMPUTE_T *sh_qy = sh_dy + nchan_out;

        // dqy, accumulated in place by identity (2): sum_i p_i (gdotv_i - integral) k_i,
        // which is what the two-pass formulation assembles at the end of its first walk
        // out of two accumulators and a scalar. Dead under TWO_PASS, which declares its
        // own pair below.
        COMPUTE_T loc_dq[NLOC];
#pragma unroll
        for (int i = 0; i < NLOC; i++) { loc_dq[i] = __vset<COMPUTE_T>(0.0f); }

        // Register copies of this thread's slice of qy / dy. Both are loop-invariant
        // across neighbours, and each thread only ever touches its own
        // (tidx + i*BDIM_X) slots, so re-reading them from shared once per neighbour
        // was pure overhead.
        COMPUTE_T loc_qy[NLOC];
        COMPUTE_T loc_dy[NLOC];
#pragma unroll
        for (int i = 0; i < NLOC; i++) {
            loc_qy[i] = __vset<COMPUTE_T>(0.0f);
            loc_dy[i] = __vset<COMPUTE_T>(0.0f);
        }

        // the lane's channel offset folded into the base pointers, as on the product
        // grid, so the inner loops index by register rather than by channel
        kx += int64_t(batch) * npoints_in * ldi + int64_t(head) * nchan_in + tidx;
        vx += int64_t(batch) * npoints_in * ldo + int64_t(head) * nchan_out + tidx;

        qy += int64_t(batch) * npoints_out * ldi + int64_t(head) * nchan_in + ipoint * ldi + tidx;
        dy += int64_t(batch) * npoints_out * ldo + int64_t(head) * nchan_out + ipoint * ldo + tidx;

        dkx += int64_t(batch) * npoints_in * ldi + int64_t(head) * nchan_in + tidx;
        dvx += int64_t(batch) * npoints_in * ldo + int64_t(head) * nchan_out + tidx;
        dqy += int64_t(batch) * npoints_out * ldi + int64_t(head) * nchan_in + ipoint * ldi + tidx;

#pragma unroll
        for (int i = 0; i < NLOC_M1; i++) {
            loc_qy[i] = vload(qy, i * BDIM_X);
            sh_qy[i * BDIM_X] = loc_qy[i];
        }
        if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
            loc_qy[NLOC_M1] = vload(qy, NLOC_M1 * BDIM_X);
            sh_qy[NLOC_M1 * BDIM_X] = loc_qy[NLOC_M1];
        }

#pragma unroll
        for (int i = 0; i < NLOC_M1; i++) {
            loc_dy[i] = vload(dy, i * BDIM_X);
            sh_dy[i * BDIM_X] = loc_dy[i];
        }
        if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
            loc_dy[NLOC_M1] = vload(dy, NLOC_M1 * BDIM_X);
            sh_dy[NLOC_M1 * BDIM_X] = loc_dy[NLOC_M1];
        }

#if __CUDA_ARCH__ < 900
        // for architectures < 9.0, sh_dy and sh_qy will be read as individual floats
        // at the end of the kernel, which breaks the assumption that each COMPUTE_T
        // location is written to and read by the same thread throughout the kernel,
        // in the case COMPUTE_T==float4
        if constexpr (std::is_same<COMPUTE_T, float4>::value) { __syncwarp(); }
#endif

        // The three warp-uniform scalars the walk below needs; see the generic kernel.
        float qdotk_max = -FLT_MAX;
        float alpha_sum_inv;
        float integral = 0.0f;

        const int seg_beg = seg_off[ipoint];
        const int seg_end = seg_off[ipoint + 1];

        if constexpr (TWO_PASS) {

            // dqy's two accumulators, live only in this formulation. The product grid
            // carries a third array, loc_vw_[NLOC], every entry of which always holds the
            // same number; it is the scalar alpha_vw_ here, for the reason spelled out in
            // the generic kernel above.
            COMPUTE_T loc_k__[NLOC];
            COMPUTE_T loc_kvw[NLOC];
#pragma unroll
            for (int i = 0; i < NLOC; i++) {
                loc_k__[i] = __vset<COMPUTE_T>(0.0f);
                loc_kvw[i] = __vset<COMPUTE_T>(0.0f);
            }

            float alpha_sum = 0.0f;

            // the scalar that replaces the replicated loc_vw_ array
            float alpha_vw_ = 0.0f;

            // Pass 1: accumulate alpha_sum, integral and the register reductions, along
            // with a progressively computed qdotk_max.
            for (int sg = seg_beg; sg < seg_end; sg++) {

                const int iring = seg[3 * sg + 0];
                const int seg_lo = seg[3 * sg + 1];
                const int seg_len = seg[3 * sg + 2];

                // constant along the arc: every point of a ring carries the same
                // quadrature weight
                const float qw_seg = ring_weights[iring];

                // a ring is contiguous in RING order, so the flat column is the ring's
                // base plus an offset that counts up and wraps at the ring's end
                const int64_t ring_lo = ring_base[iring];
                const int64_t ring_hi = ring_lo + ring_size[iring];

                int64_t col = ring_lo + seg_lo;

                // Grouped body: NB neighbours' addresses are formed first so their k and
                // v loads are all outstanding before any is consumed, and the online
                // softmax then rescales once per group instead of once per neighbour.
                //
                // This is the one place in the two-pass formulation where grouping changes
                // the arithmetic. Rescaling per group rather than per neighbour is the same
                // reduction in exact arithmetic but a different order of fp32 sums, so
                // the gradient differs in its last bits. That is why grouping landed with
                // e9092eb and not with the original port: the oracle gradient checks at 96
                // channels per head exist and pass, so the change was finally testable.
                int j = 0;
                for (; j + NB <= seg_len; j += NB) {

                    const STORAGE_T *kp[NB];
                    const STORAGE_T *vp[NB];
#pragma unroll
                    for (int u = 0; u < NB; u++) {
                        kp[u] = kx + col * ldi;
                        vp[u] = vx + col * ldo;
                        if (++col == ring_hi) { col = ring_lo; }
                    }

                    COMPUTE_T qdotk_g[NB];
                    COMPUTE_T gdotv_g[NB];
#pragma unroll
                    for (int u = 0; u < NB; u++) {
                        qdotk_g[u] = __vset<COMPUTE_T>(0.0f);
                        gdotv_g[u] = __vset<COMPUTE_T>(0.0f);
                    }

                    // one channel step feeds 2*NB accumulators, so that many loads are
                    // outstanding per step rather than two
#pragma unroll
                    for (int i = 0; i < NLOC_M1; i++) {
                        const COMPUTE_T q = loc_qy[i];
                        const COMPUTE_T d = loc_dy[i];
#pragma unroll
                        for (int u = 0; u < NB; u++) {
                            qdotk_g[u] = __vadd(qdotk_g[u], __vmul(q, vload(kp[u], i * BDIM_X)));
                            gdotv_g[u] = __vadd(gdotv_g[u], __vmul(d, vload(vp[u], i * BDIM_X)));
                        }
                    }
                    if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                        const COMPUTE_T q = loc_qy[NLOC_M1];
#pragma unroll
                        for (int u = 0; u < NB; u++) {
                            qdotk_g[u] = __vadd(qdotk_g[u], __vmul(q, vload(kp[u], NLOC_M1 * BDIM_X)));
                        }
                    }
                    if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
                        const COMPUTE_T d = loc_dy[NLOC_M1];
#pragma unroll
                        for (int u = 0; u < NB; u++) {
                            gdotv_g[u] = __vadd(gdotv_g[u], __vmul(d, vload(vp[u], NLOC_M1 * BDIM_X)));
                        }
                    }

                    float qdotk_b[NB];
                    float gdotv_b[NB];
#pragma unroll
                    for (int u = 0; u < NB; u++) {
                        qdotk_b[u] = __warp_sum(__vred(qdotk_g[u]));
                        gdotv_b[u] = __warp_sum(__vred(gdotv_g[u]));
                    }

                    float qdotk_max_grp = qdotk_max;
#pragma unroll
                    for (int u = 0; u < NB; u++) { qdotk_max_grp = max(qdotk_max_grp, qdotk_b[u]); }
                    const float mc_grp = expf(qdotk_max - qdotk_max_grp);

                    // ag[u] is alpha_inz * gdotv, wanted once for the scalar recurrences
                    // and again for each of the NLOC register accumulators
                    float alpha_b[NB];
                    float ag_b[NB];
                    float alpha_grp = 0.0f;
                    float ag_grp = 0.0f;
#pragma unroll
                    for (int u = 0; u < NB; u++) {
                        alpha_b[u] = expf(qdotk_b[u] - qdotk_max_grp) * qw_seg;
                        ag_b[u] = alpha_b[u] * gdotv_b[u];
                        alpha_grp += alpha_b[u];
                        ag_grp += ag_b[u];
                    }

                    alpha_sum = alpha_sum * mc_grp + alpha_grp;
                    integral = integral * mc_grp + ag_grp;
                    alpha_vw_ = alpha_vw_ * mc_grp + ag_grp;

#pragma unroll
                    for (int i = 0; i < NLOC_M1; i++) {
                        COMPUTE_T k_acc = __vscale(mc_grp, loc_k__[i]);
                        COMPUTE_T kvw_acc = __vscale(mc_grp, loc_kvw[i]);
#pragma unroll
                        for (int u = 0; u < NB; u++) {
                            const COMPUTE_T kxval = vload(kp[u], i * BDIM_X);
                            k_acc = __vadd(k_acc, __vscale(alpha_b[u], kxval));
                            kvw_acc = __vadd(kvw_acc, __vscale(ag_b[u], kxval));
                        }
                        loc_k__[i] = k_acc;
                        loc_kvw[i] = kvw_acc;
                    }
                    if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                        COMPUTE_T k_acc = __vscale(mc_grp, loc_k__[NLOC_M1]);
                        COMPUTE_T kvw_acc = __vscale(mc_grp, loc_kvw[NLOC_M1]);
#pragma unroll
                        for (int u = 0; u < NB; u++) {
                            const COMPUTE_T kxval = vload(kp[u], NLOC_M1 * BDIM_X);
                            k_acc = __vadd(k_acc, __vscale(alpha_b[u], kxval));
                            kvw_acc = __vadd(kvw_acc, __vscale(ag_b[u], kxval));
                        }
                        loc_k__[NLOC_M1] = k_acc;
                        loc_kvw[NLOC_M1] = kvw_acc;
                    }

                    qdotk_max = qdotk_max_grp;
                }

                // remainder: fewer than NB neighbours left in this arc
                for (; j < seg_len; j++) {

                    const STORAGE_T *_kx = kx + col * ldi;
                    const STORAGE_T *_vx = vx + col * ldo;

                    COMPUTE_T qdotk_v = __vset<COMPUTE_T>(0.0f);
                    COMPUTE_T gdotv_v = __vset<COMPUTE_T>(0.0f);

#pragma unroll
                    for (int i = 0; i < NLOC_M1; i++) {
                        qdotk_v = __vadd(qdotk_v, __vmul(loc_qy[i], vload(_kx, i * BDIM_X)));
                    }
                    if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                        qdotk_v = __vadd(qdotk_v, __vmul(loc_qy[NLOC_M1], vload(_kx, NLOC_M1 * BDIM_X)));
                    }
#pragma unroll
                    for (int i = 0; i < NLOC_M1; i++) {
                        gdotv_v = __vadd(gdotv_v, __vmul(loc_dy[i], vload(_vx, i * BDIM_X)));
                    }
                    if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
                        gdotv_v = __vadd(gdotv_v, __vmul(loc_dy[NLOC_M1], vload(_vx, NLOC_M1 * BDIM_X)));
                    }

                    const float qdotk = __warp_sum(__vred(qdotk_v));
                    const float gdotv = __warp_sum(__vred(gdotv_v));

                    const float qdotk_max_tmp = max(qdotk_max, qdotk);
                    const float alpha_inz = expf(qdotk - qdotk_max_tmp) * qw_seg;
                    const float max_correction = expf(qdotk_max - qdotk_max_tmp);

                    alpha_sum = alpha_sum * max_correction + alpha_inz;
                    integral = integral * max_correction + alpha_inz * gdotv;

                    const float ainz_gdotv = alpha_inz * gdotv;

                    // same recurrence the per-channel array used to run, once
                    alpha_vw_ = alpha_vw_ * max_correction + ainz_gdotv;

#pragma unroll
                    for (int i = 0; i < NLOC_M1; i++) {
                        const COMPUTE_T kxval = vload(_kx, i * BDIM_X);
                        loc_k__[i] = __vadd(__vscale(max_correction, loc_k__[i]), __vscale(alpha_inz, kxval));
                        loc_kvw[i] = __vadd(__vscale(max_correction, loc_kvw[i]), __vscale(ainz_gdotv, kxval));
                    }
                    if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                        const COMPUTE_T kxval = vload(_kx, NLOC_M1 * BDIM_X);
                        loc_k__[NLOC_M1] = __vadd(__vscale(max_correction, loc_k__[NLOC_M1]), __vscale(alpha_inz, kxval));
                        loc_kvw[NLOC_M1]
                            = __vadd(__vscale(max_correction, loc_kvw[NLOC_M1]), __vscale(ainz_gdotv, kxval));
                    }

                    qdotk_max = qdotk_max_tmp;

                    // next point in the arc; wraps at most once
                    if (++col == ring_hi) { col = ring_lo; }
                }
            }

            alpha_sum_inv = 1.0f / alpha_sum;

            integral *= alpha_sum_inv;

            // Write dqy (fp32 output)
            const float alpha_sum_inv_sq = alpha_sum_inv * alpha_sum_inv;

#pragma unroll
            for (int i = 0; i < NLOC_M1; i++) {
                // __vscale by the scalar, where the product grid __vmul-s by a vector every
                // entry of which holds that scalar
                dqy[i * BDIM_X] = __vscale(alpha_sum_inv_sq,
                                           __vsub(__vscale(alpha_sum, loc_kvw[i]), __vscale(alpha_vw_, loc_k__[i])));
            }
            if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                dqy[NLOC_M1 * BDIM_X]
                    = __vscale(alpha_sum_inv_sq,
                               __vsub(__vscale(alpha_sum, loc_kvw[NLOC_M1]), __vscale(alpha_vw_, loc_k__[NLOC_M1])));
            }

        } else {

            const int64_t istat = int64_t(bh) * npoints_out + ipoint;

            qdotk_max = qdotk_max_fwd[istat];
            integral = integral_fwd[istat];
            alpha_sum_inv = 1.0f / alpha_sum_fwd[istat];
        }

        // The walk; see the generic kernel for which gradients it carries in which
        // formulation.
        for (int sg = seg_beg; sg < seg_end; sg++) {

            const int iring = seg[3 * sg + 0];
            const int seg_lo = seg[3 * sg + 1];
            const int seg_len = seg[3 * sg + 2];
            const float qw_seg = ring_weights[iring];

            const int64_t ring_lo = ring_base[iring];
            const int64_t ring_hi = ring_lo + ring_size[iring];

            int64_t col = ring_lo + seg_lo;

            // Grouped, and without the caveat the two-pass first walk carries: there is
            // no running maximum to update here, so each neighbour's contribution is
            // computed independently of the others. Under TWO_PASS the only sums it
            // feeds are the atomics, which were never ordered, so grouping is
            // bit-identical there; in the single-pass formulation it also feeds loc_dq,
            // where NB fixes the order of an fp32 sum as it does in pass 1.
            int j = 0;
            for (; j + NB <= seg_len; j += NB) {

                int64_t cols[NB];
#pragma unroll
                for (int u = 0; u < NB; u++) {
                    cols[u] = col;
                    if (++col == ring_hi) { col = ring_lo; }
                }

                const STORAGE_T *kp[NB];
                const STORAGE_T *vp[NB];
#pragma unroll
                for (int u = 0; u < NB; u++) {
                    kp[u] = kx + cols[u] * ldi;
                    vp[u] = vx + cols[u] * ldo;
                }

                COMPUTE_T qdotk_g[NB];
                COMPUTE_T gdotv_g[NB];
#pragma unroll
                for (int u = 0; u < NB; u++) {
                    qdotk_g[u] = __vset<COMPUTE_T>(0.0f);
                    gdotv_g[u] = __vset<COMPUTE_T>(0.0f);
                }

#pragma unroll
                for (int i = 0; i < NLOC_M1; i++) {
                    const COMPUTE_T q = loc_qy[i];
                    const COMPUTE_T d = loc_dy[i];
#pragma unroll
                    for (int u = 0; u < NB; u++) {
                        qdotk_g[u] = __vadd(qdotk_g[u], __vmul(q, vload(kp[u], i * BDIM_X)));
                        gdotv_g[u] = __vadd(gdotv_g[u], __vmul(d, vload(vp[u], i * BDIM_X)));
                    }
                }
                if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                    const COMPUTE_T q = loc_qy[NLOC_M1];
#pragma unroll
                    for (int u = 0; u < NB; u++) {
                        qdotk_g[u] = __vadd(qdotk_g[u], __vmul(q, vload(kp[u], NLOC_M1 * BDIM_X)));
                    }
                }
                if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
                    const COMPUTE_T d = loc_dy[NLOC_M1];
#pragma unroll
                    for (int u = 0; u < NB; u++) {
                        gdotv_g[u] = __vadd(gdotv_g[u], __vmul(d, vload(vp[u], NLOC_M1 * BDIM_X)));
                    }
                }

                // all 2*NB reductions before any is consumed, so their shuffles
                // pipeline instead of alternating with the scatter
                float qdotk_b[NB];
                float gdotv_b[NB];
#pragma unroll
                for (int u = 0; u < NB; u++) {
                    qdotk_b[u] = __warp_sum(__vred(qdotk_g[u]));
                    gdotv_b[u] = __warp_sum(__vred(gdotv_g[u]));
                }

                // Both multipliers for the whole group up front. Identity (3): these are
                // (gdotv_i - integral) * p_i and p_i, and identity (2) says the first is
                // also the weight k_i carries into dqy -- which wants them
                // channel-outer, so they cannot stay inside the scatter's loop.
                float scale_fact_qy[NB];
                float scale_fact_dy[NB];
#pragma unroll
                for (int u = 0; u < NB; u++) {
                    const float alpha_inz_u = expf(qdotk_b[u] - qdotk_max) * qw_seg;
                    const float alpha_mul_u = alpha_inz_u * alpha_sum_inv;

                    scale_fact_qy[u] = (gdotv_b[u] - integral) * alpha_mul_u;
                    scale_fact_dy[u] = alpha_mul_u;
                }

                // one channel step feeds NB terms of dqy, matching the loads already in
                // flight for the reductions above
                if constexpr (!TWO_PASS) {
#pragma unroll
                    for (int i = 0; i < NLOC_M1; i++) {
                        COMPUTE_T dq_acc = loc_dq[i];
#pragma unroll
                        for (int u = 0; u < NB; u++) {
                            dq_acc = __vadd(dq_acc, __vscale(scale_fact_qy[u], vload(kp[u], i * BDIM_X)));
                        }
                        loc_dq[i] = dq_acc;
                    }
                    if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                        COMPUTE_T dq_acc = loc_dq[NLOC_M1];
#pragma unroll
                        for (int u = 0; u < NB; u++) {
                            dq_acc = __vadd(dq_acc, __vscale(scale_fact_qy[u], vload(kp[u], NLOC_M1 * BDIM_X)));
                        }
                        loc_dq[NLOC_M1] = dq_acc;
                    }
                }

#pragma unroll
                for (int u = 0; u < NB; u++) {

                    COMPUTE_T *_dkx = dkx + cols[u] * ldi;
                    COMPUTE_T *_dvx = dvx + cols[u] * ldo;

#if __CUDA_ARCH__ < 900
                    constexpr int VEC_SIZE = sizeof(COMPUTE_T) / sizeof(float);

                    float *sh_qy_scl = reinterpret_cast<float *>(sh_qy) - tidx * VEC_SIZE;
                    float *sh_dy_scl = reinterpret_cast<float *>(sh_dy) - tidx * VEC_SIZE;
                    float *_dkx_scl = reinterpret_cast<float *>(_dkx) - tidx * VEC_SIZE;
                    float *_dvx_scl = reinterpret_cast<float *>(_dvx) - tidx * VEC_SIZE;

                    for (int chan = tidx; chan < nchan_in * VEC_SIZE; chan += BDIM_X) {
                        atomicAdd(_dkx_scl + chan, scale_fact_qy[u] * sh_qy_scl[chan]);
                    }
                    for (int chan = tidx; chan < nchan_out * VEC_SIZE; chan += BDIM_X) {
                        atomicAdd(_dvx_scl + chan, scale_fact_dy[u] * sh_dy_scl[chan]);
                    }
#else
#pragma unroll
                    for (int i = 0; i < NLOC_M1; i++) {
                        atomicAdd(_dkx + i * BDIM_X, __vscale(scale_fact_qy[u], loc_qy[i]));
                    }
                    if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                        atomicAdd(_dkx + NLOC_M1 * BDIM_X, __vscale(scale_fact_qy[u], loc_qy[NLOC_M1]));
                    }
#pragma unroll
                    for (int i = 0; i < NLOC_M1; i++) {
                        atomicAdd(_dvx + i * BDIM_X, __vscale(scale_fact_dy[u], loc_dy[i]));
                    }
                    if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
                        atomicAdd(_dvx + NLOC_M1 * BDIM_X, __vscale(scale_fact_dy[u], loc_dy[NLOC_M1]));
                    }
#endif
                }
            }

            // remainder: fewer than NB neighbours left in this arc
            for (; j < seg_len; j++) {

                const STORAGE_T *_kx = kx + col * ldi;
                const STORAGE_T *_vx = vx + col * ldo;

                COMPUTE_T qdotk_v = __vset<COMPUTE_T>(0.0f);
                COMPUTE_T gdotv_v = __vset<COMPUTE_T>(0.0f);

#pragma unroll
                for (int i = 0; i < NLOC_M1; i++) {
                    qdotk_v = __vadd(qdotk_v, __vmul(loc_qy[i], vload(_kx, i * BDIM_X)));
                }
                if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                    qdotk_v = __vadd(qdotk_v, __vmul(loc_qy[NLOC_M1], vload(_kx, NLOC_M1 * BDIM_X)));
                }
#pragma unroll
                for (int i = 0; i < NLOC_M1; i++) {
                    gdotv_v = __vadd(gdotv_v, __vmul(loc_dy[i], vload(_vx, i * BDIM_X)));
                }
                if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
                    gdotv_v = __vadd(gdotv_v, __vmul(loc_dy[NLOC_M1], vload(_vx, NLOC_M1 * BDIM_X)));
                }

                const float qdotk = __warp_sum(__vred(qdotk_v));
                const float gdotv = __warp_sum(__vred(gdotv_v));

                const float alpha_inz = expf(qdotk - qdotk_max) * qw_seg;

                // _dkx / _dvx are COMPUTE_T (fp32) gradient buffers, accumulated
                // atomically: neighbourhoods overlap, so several output points hit
                // the same input point.
                COMPUTE_T *_dkx = dkx + col * ldi;
                COMPUTE_T *_dvx = dvx + col * ldo;

                const float alpha_mul = alpha_inz * alpha_sum_inv;

                const float scale_fact_qy = (gdotv - integral) * alpha_mul;
                const float scale_fact_dy = alpha_mul;

                // dqy by identity (2)
                if constexpr (!TWO_PASS) {
#pragma unroll
                    for (int i = 0; i < NLOC_M1; i++) {
                        loc_dq[i] = __vadd(loc_dq[i], __vscale(scale_fact_qy, vload(_kx, i * BDIM_X)));
                    }
                    if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                        loc_dq[NLOC_M1] = __vadd(loc_dq[NLOC_M1], __vscale(scale_fact_qy, vload(_kx, NLOC_M1 * BDIM_X)));
                    }
                }

                // float4, 128-bit atomics are only supported by devices of compute
                // capability 9.x+, so on older devices we resort to 32-bit atomics

#if __CUDA_ARCH__ < 900
                constexpr int VEC_SIZE = sizeof(COMPUTE_T) / sizeof(float);

                // to use 32-bit operations on consecutive addresses; the registers
                // cannot serve this, which is why sh_qy / sh_dy exist
                float *sh_qy_scl = reinterpret_cast<float *>(sh_qy);
                float *sh_dy_scl = reinterpret_cast<float *>(sh_dy);

                float *_dkx_scl = reinterpret_cast<float *>(_dkx);
                float *_dvx_scl = reinterpret_cast<float *>(_dvx);

                sh_qy_scl -= tidx * VEC_SIZE;
                sh_dy_scl -= tidx * VEC_SIZE;
                _dkx_scl -= tidx * VEC_SIZE;
                _dvx_scl -= tidx * VEC_SIZE;

                // 32-bit, consecutive atomics to glmem;
                // strided atomics results in a severe slowdown
                for (int chan = tidx; chan < nchan_in * VEC_SIZE; chan += BDIM_X) {
                    atomicAdd(_dkx_scl + chan, scale_fact_qy * sh_qy_scl[chan]);
                }
                for (int chan = tidx; chan < nchan_out * VEC_SIZE; chan += BDIM_X) {
                    atomicAdd(_dvx_scl + chan, scale_fact_dy * sh_dy_scl[chan]);
                }
#else
                // 128-bit, consecutive atomics to glmem, straight out of the registers
#pragma unroll
                for (int i = 0; i < NLOC_M1; i++) { atomicAdd(_dkx + i * BDIM_X, __vscale(scale_fact_qy, loc_qy[i])); }
                if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                    atomicAdd(_dkx + NLOC_M1 * BDIM_X, __vscale(scale_fact_qy, loc_qy[NLOC_M1]));
                }
#pragma unroll
                for (int i = 0; i < NLOC_M1; i++) { atomicAdd(_dvx + i * BDIM_X, __vscale(scale_fact_dy, loc_dy[i])); }
                if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
                    atomicAdd(_dvx + NLOC_M1 * BDIM_X, __vscale(scale_fact_dy, loc_dy[NLOC_M1]));
                }
#endif
                if (++col == ring_hi) { col = ring_lo; }
            }
        }

        // Write dqy (fp32 output). No reciprocal and no combination: alpha_sum_inv went
        // into every term as it was accumulated.
        if constexpr (!TWO_PASS) {
#pragma unroll
            for (int i = 0; i < NLOC_M1; i++) { dqy[i * BDIM_X] = loc_dq[i]; }
            if (NLOC_M1 * BDIM_X + tidx < nchan_in) { dqy[NLOC_M1 * BDIM_X] = loc_dq[NLOC_M1]; }
        }

        return;
    }

    // Resolve NLOC, which has to be a compile-time constant, from the runtime channel
    // dk and dv by gather instead of scatter, for grid_in == grid_out.
    // ------------------------------------------------------------------
    // The scatter form is already minimal in its own orientation: loc_qy and loc_dy
    // are loop-invariant, so every one of a query's neighbours receives a scaled copy
    // of one register-resident vector, coalesced across lanes. What it costs is the
    // atomics -- one add per channel per neighbour, 1.24e11 of them per layer at
    // nside 64, and after the single-pass change they are what binds: the backward
    // moves 23.2e9 L2 reduction sectors, about 496 GB against 248 GB of reads, and
    // runs at 85.7% memory throughput with L2 at 71%.
    //
    // Turning it round removes them entirely. Fix an input point j and accumulate
    //
    //   dk[j] = sum_{i in N(j)} p_i(j) (gdotv_i(j) - integral_i) q_i
    //   dv[j] = sum_{i in N(j)} p_i(j) dy_i
    //
    // in registers, then store once. The arithmetic is the same -- both forms
    // recompute q.k and dy.v per pair -- so this trades 496 GB of atomic writes for
    // 248 GB of q/dy reads and a plain store.
    //
    // It works because the neighbourhood is symmetric, so N(j) is j's own arc list
    // read as queries rather than as neighbours. That is a property of the computed
    // pattern and not of the continuous geometry, since the arc endpoints come from a
    // ceil and a floor: measured, zero one-directional entries over 20,181,696 pairs
    // at nside 64 and over the smaller grids too. It also requires the two grids to be
    // the same one; for a mixed pair j in N(i) does not imply i in N(j), and the
    // launcher keeps the scatter for that case.
    //
    // Two things fall out of the orientation. The quadrature weight is keyed by the
    // input ring, which is j's own here, so it leaves the arc loop and becomes a
    // constant -- hence point_ring, which the wrapper builds once. And dqy still wants
    // the query orientation, so it stays in the scatter kernel and this runs as a
    // second launch.
    //
    // Neighbour grouping is deliberately absent for now: it is what hides the
    // dependency chain through the two warp reductions, and adding it at the same time
    // as a new formulation would make a correctness failure and a performance failure
    // look alike.
    template <int BDIM_X, int BDIM_Y, int NLOC, typename STORAGE_T>
    __global__ __launch_bounds__(BDIM_X *BDIM_Y) void s2_attn_bwd_ragged_gather_kv(
        int nheads, int nchan_in, int nchan_out, int64_t npoints, const STORAGE_T *__restrict__ kx,
        const STORAGE_T *__restrict__ vx, const STORAGE_T *__restrict__ qy, const STORAGE_T *__restrict__ dy,
        const float *__restrict__ alpha_sum_in, const float *__restrict__ qdotk_max_in,
        const float *__restrict__ integral_in, const int32_t *__restrict__ seg, const int32_t *__restrict__ seg_off,
        const int64_t *__restrict__ ring_base, const int64_t *__restrict__ ring_size,
        const float *__restrict__ ring_weights, const int32_t *__restrict__ point_ring,
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dkx,
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dvx)
    {
        using COMPUTE_T = typename vec_traits<STORAGE_T>::compute_t;

        static_assert(BDIM_X == WARP_SIZE, "the gather kernel reduces with __warp_sum");
        static_assert(NLOC >= 1);

        constexpr int NLOC_M1 = NLOC - 1;

        const int tidx = threadIdx.x;

        const int bh = blockIdx.y;
        const int batch = bh / nheads;
        const int head = bh - (batch * nheads);

        const int64_t ldi = int64_t(nheads) * nchan_in;
        const int64_t ldo = int64_t(nheads) * nchan_out;

        const int64_t jpoint = int64_t(blockIdx.x) * blockDim.y + threadIdx.y;

        if (jpoint >= npoints) { return; }

        // k_j and v_j are what stays put here, where qy and dy do in the scatter form
        COMPUTE_T loc_k[NLOC];
        COMPUTE_T loc_v[NLOC];
        COMPUTE_T acc_dk[NLOC];
        COMPUTE_T acc_dv[NLOC];
#pragma unroll
        for (int i = 0; i < NLOC; i++) {
            loc_k[i] = __vset<COMPUTE_T>(0.0f);
            loc_v[i] = __vset<COMPUTE_T>(0.0f);
            acc_dk[i] = __vset<COMPUTE_T>(0.0f);
            acc_dv[i] = __vset<COMPUTE_T>(0.0f);
        }

        // the lane's channel offset folded in, as elsewhere. qy and dy keep their
        // point stride because they are indexed per neighbour.
        const STORAGE_T *kxj = kx + int64_t(batch) * npoints * ldi + int64_t(head) * nchan_in + jpoint * ldi + tidx;
        const STORAGE_T *vxj = vx + int64_t(batch) * npoints * ldo + int64_t(head) * nchan_out + jpoint * ldo + tidx;
        qy += int64_t(batch) * npoints * ldi + int64_t(head) * nchan_in + tidx;
        dy += int64_t(batch) * npoints * ldo + int64_t(head) * nchan_out + tidx;
        dkx += int64_t(batch) * npoints * ldi + int64_t(head) * nchan_in + jpoint * ldi + tidx;
        dvx += int64_t(batch) * npoints * ldo + int64_t(head) * nchan_out + jpoint * ldo + tidx;

#pragma unroll
        for (int i = 0; i < NLOC_M1; i++) { loc_k[i] = vload(kxj, i * BDIM_X); }
        if (NLOC_M1 * BDIM_X + tidx < nchan_in) { loc_k[NLOC_M1] = vload(kxj, NLOC_M1 * BDIM_X); }
#pragma unroll
        for (int i = 0; i < NLOC_M1; i++) { loc_v[i] = vload(vxj, i * BDIM_X); }
        if (NLOC_M1 * BDIM_X + tidx < nchan_out) { loc_v[NLOC_M1] = vload(vxj, NLOC_M1 * BDIM_X); }

        // constant for the whole walk, which it is not in the other orientation
        const float qw = ring_weights[point_ring[jpoint]];

        const int64_t stat_off = int64_t(bh) * npoints;

        const int seg_beg = seg_off[jpoint];
        const int seg_end = seg_off[jpoint + 1];

        for (int sg = seg_beg; sg < seg_end; sg++) {

            const int iring = seg[3 * sg + 0];
            const int lo = seg[3 * sg + 1];
            const int len = seg[3 * sg + 2];

            const int64_t ring_lo = ring_base[iring];
            const int64_t ring_hi = ring_lo + ring_size[iring];

            int64_t col = ring_lo + lo;

            for (int t = 0; t < len; t++) {

                // col is a query that attends to j
                const STORAGE_T *qp = qy + col * ldi;
                const STORAGE_T *dp = dy + col * ldo;

                COMPUTE_T qk = __vset<COMPUTE_T>(0.0f);
#pragma unroll
                for (int i = 0; i < NLOC_M1; i++) { qk = __vadd(qk, __vmul(loc_k[i], vload(qp, i * BDIM_X))); }
                if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                    qk = __vadd(qk, __vmul(loc_k[NLOC_M1], vload(qp, NLOC_M1 * BDIM_X)));
                }
                const float qdotk = __warp_sum(__vred(qk));

                COMPUTE_T gv = __vset<COMPUTE_T>(0.0f);
#pragma unroll
                for (int i = 0; i < NLOC_M1; i++) { gv = __vadd(gv, __vmul(loc_v[i], vload(dp, i * BDIM_X))); }
                if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
                    gv = __vadd(gv, __vmul(loc_v[NLOC_M1], vload(dp, NLOC_M1 * BDIM_X)));
                }
                const float gdotv = __warp_sum(__vred(gv));

                // the query's own softmax state, which is why the forward returns it
                const int64_t ist = stat_off + col;
                const float alpha = expf(qdotk - qdotk_max_in[ist]) * qw;
                const float p = alpha / alpha_sum_in[ist];

                const float s_k = p * (gdotv - integral_in[ist]);
                const float s_v = p;

                // re-read rather than stage: the values are in L1 from the reductions
                // above, and registers are what this kernel family runs short of
#pragma unroll
                for (int i = 0; i < NLOC_M1; i++) {
                    acc_dk[i] = __vadd(acc_dk[i], __vscale(s_k, vload(qp, i * BDIM_X)));
                }
                if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                    acc_dk[NLOC_M1] = __vadd(acc_dk[NLOC_M1], __vscale(s_k, vload(qp, NLOC_M1 * BDIM_X)));
                }
#pragma unroll
                for (int i = 0; i < NLOC_M1; i++) {
                    acc_dv[i] = __vadd(acc_dv[i], __vscale(s_v, vload(dp, i * BDIM_X)));
                }
                if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
                    acc_dv[NLOC_M1] = __vadd(acc_dv[NLOC_M1], __vscale(s_v, vload(dp, NLOC_M1 * BDIM_X)));
                }

                if (++col == ring_hi) { col = ring_lo; }
            }
        }

        // one store per output, where the scatter form issued one atomic per
        // contributing query. This is the whole point of the kernel.
#pragma unroll
        for (int i = 0; i < NLOC_M1; i++) { dkx[i * BDIM_X] = acc_dk[i]; }
        if (NLOC_M1 * BDIM_X + tidx < nchan_in) { dkx[NLOC_M1 * BDIM_X] = acc_dk[NLOC_M1]; }
#pragma unroll
        for (int i = 0; i < NLOC_M1; i++) { dvx[i * BDIM_X] = acc_dv[i]; }
        if (NLOC_M1 * BDIM_X + tidx < nchan_out) { dvx[NLOC_M1 * BDIM_X] = acc_dv[NLOC_M1]; }
    }

    // count by walking the supported range. Mirrors launch_spc_attn_fwd_ragged, and so
    // launch_spc_attn_bwd, minus its CHOUT_AS_IN branch (see the kernel).
    template <int BDIM_X, int BDIM_Y, int CUR_LOC, int MAX_LOC, bool TWO_PASS, typename STORAGE_T>
    static void launch_spc_attn_bwd_ragged(int nloc, int batch_size, int nheads, int nchans_in, int nchans_out,
                                           int64_t npoints_in, int64_t npoints_out, const STORAGE_T *_kxp,
                                           const STORAGE_T *_vxp, const STORAGE_T *_qyp, const STORAGE_T *_dyp,
                                           const float *_alpha_sum, const float *_qdotk_max, const float *_integral,
                                           const int32_t *_seg, const int32_t *_seg_off, const int64_t *_ring_base,
                                           const int64_t *_ring_size, const float *_ring_weights,
                                           typename vec_traits<STORAGE_T>::compute_t *_dkxp,
                                           typename vec_traits<STORAGE_T>::compute_t *_dvxp,
                                           typename vec_traits<STORAGE_T>::compute_t *_dqyp, cudaStream_t stream)
    {
        if constexpr (CUR_LOC > MAX_LOC) {
            TORCH_CHECK(false, "ragged special attention kernel reached nloc ", nloc, " above its bound ", MAX_LOC);
            return;
        } else {
            if (CUR_LOC == nloc) {
                dim3 block(BDIM_X, BDIM_Y);
                dim3 grid(DIV_UP(npoints_out, block.y), batch_size * nheads);

                // 2 arrays per warp, against the generic kernel's 3 or 4: only qy and dy
                // are staged, and only for the pre-9.0 epilogue
                size_t shsize = sizeof(typename vec_traits<STORAGE_T>::compute_t) * (nchans_in + nchans_out) * block.y;

                s2_attn_bwd_ragged_special_vec_k<BDIM_X, BDIM_Y, CUR_LOC, TWO_PASS><<<grid, block, shsize, stream>>>(
                    nheads, nchans_in, nchans_out, npoints_in, npoints_out, _kxp, _vxp, _qyp, _dyp, _alpha_sum,
                    _qdotk_max, _integral, _seg, _seg_off, _ring_base, _ring_size, _ring_weights, _dkxp, _dvxp, _dqyp);
                CHECK_ERROR("s2_attn_bwd_ragged_special_vec_k");
                return;
            }
            launch_spc_attn_bwd_ragged<BDIM_X, BDIM_Y, CUR_LOC + 1, MAX_LOC, TWO_PASS, STORAGE_T>(
                nloc, batch_size, nheads, nchans_in, nchans_out, npoints_in, npoints_out, _kxp, _vxp, _qyp, _dyp,
                _alpha_sum, _qdotk_max, _integral, _seg, _seg_off, _ring_base, _ring_size, _ring_weights, _dkxp, _dvxp,
                _dqyp, stream);
        }
    }

    // Set TORCH_HARMONICS_RAGGED_BWD_GENERIC=1 to force the original kernel. Separate
    // from the forward's TORCH_HARMONICS_RAGGED_GENERIC on purpose: the two ports are
    // independent, so a wrong gradient can be attributed to one or the other at
    // runtime without a rebuild.
    static bool ragged_bwd_force_generic()
    {
        static const bool forced = []() {
            const char *env = std::getenv("TORCH_HARMONICS_RAGGED_BWD_GENERIC");
            return env != nullptr && env[0] == '1';
        }();
        return forced;
    }

    // Set TORCH_HARMONICS_RAGGED_BWD_TWO_PASS=1 to force the two-pass formulation, in
    // whichever of the two kernels is selected. Separate again from _BWD_GENERIC, and
    // for a sharper reason than that switch has: collapsing the two walks is exact in
    // exact arithmetic but regroups fp32 sums, so it is the change here that can move a
    // gradient's last bits. Telling that apart from a genuine error has to be possible
    // at runtime, on the build a training job is already holding.
    //
    // Defaults by dtype, because the collapse is exact in exact arithmetic but not in
    // bfloat16. integral = dy . out is read from the *stored* output, and it is then
    // subtracted: dqy and dk both carry (gdotv_i - integral). In fp32 and fp16 the
    // rounding in that difference stays inside the suite's 3e-2 tolerance; in bf16,
    // with 8 mantissa bits in out, it does not -- job 3835420 failed dk and dq there
    // while dv, the one gradient that does not involve integral, passed, and fp32 and
    // fp16 passed outright. The two-pass form never had the problem because it built
    // gdotv_i and integral in the same fp32 accumulation from the same data, so the
    // cancellation was between consistent quantities.
    //
    // So bf16 takes the two walks and everything else takes one. That is the
    // conservative way round, and it is the unhelpful way round as well: bf16 is what
    // training runs under, so the collapse currently buys nothing there. Recovering it
    // needs integral built from something better than a bf16 output -- an fp32 copy of
    // y is the obvious candidate and costs as much memory as the output itself, so it
    // wants measuring against the 2x it would buy back rather than assuming.
    static bool ragged_bwd_two_pass(at::ScalarType dtype, bool have_precise_y)
    {
        static const int forced = []() {
            const char *env = std::getenv("TORCH_HARMONICS_RAGGED_BWD_TWO_PASS");
            if (env == nullptr) { return -1; }
            return env[0] == '1' ? 1 : 0;
        }();
        if (forced >= 0) { return forced == 1; }
        // bf16 only needs the two walks when integral would have to come from the
        // bf16 output. Given the fp32 copy the forward now emits for exactly this,
        // it takes the single walk like every other dtype.
        return dtype == at::kBFloat16 && !have_precise_y;
    }

    template <bool TWO_PASS, typename STORAGE_T>
    static void launch_gen_attn_bwd_ragged(int batch_size, int nheads, int nchans_in, int nchans_out, int64_t npoints_in,
                                           int64_t npoints_out, const STORAGE_T *_kxp, const STORAGE_T *_vxp,
                                           const STORAGE_T *_qyp, const STORAGE_T *_dyp, const float *_alpha_sum,
                                           const float *_qdotk_max, const float *_integral, const int32_t *_seg,
                                           const int32_t *_seg_off, const int64_t *_ring_base, const int64_t *_ring_size,
                                           const float *_ring_weights, typename vec_traits<STORAGE_T>::compute_t *_dkxp,
                                           typename vec_traits<STORAGE_T>::compute_t *_dvxp,
                                           typename vec_traits<STORAGE_T>::compute_t *_dqyp, cudaStream_t stream)
    {
        // The register-blocked kernel needs NLOC == DIV_UP(nchan, WARP_SIZE) to hold for
        // both channel counts at once, which is why the equality is required rather
        // than taking the larger: NLOC also decides which registers go unguarded.
        const int nloc = DIV_UP(nchans_in, WARP_SIZE);
        if (!ragged_bwd_force_generic() && nchans_in == nchans_out && nloc <= MAX_LOCAL_ARR_LEN_RAGGED_BWD) {
            launch_spc_attn_bwd_ragged<WARP_SIZE, THREADS / WARP_SIZE, 1, MAX_LOCAL_ARR_LEN_RAGGED_BWD, TWO_PASS, STORAGE_T>(
                nloc, batch_size, nheads, nchans_in, nchans_out, npoints_in, npoints_out, _kxp, _vxp, _qyp, _dyp,
                _alpha_sum, _qdotk_max, _integral, _seg, _seg_off, _ring_base, _ring_size, _ring_weights, _dkxp, _dvxp,
                _dqyp, stream);
            return;
        }

        dim3 block(WARP_SIZE, THREADS / WARP_SIZE);
        // one block row per (batch, head) pair
        dim3 grid(DIV_UP(npoints_out, block.y), batch_size * nheads);

        // shared memory holds compute-type (COMPUTE_T) data, not STORAGE_T. One
        // accumulator per warp in the single-pass formulation and two in the two-pass
        // one, plus dy and qy; alpha_vw_ was never an array here (see the kernel).
        constexpr int NACC = TWO_PASS ? 2 : 1;
        size_t shsize
            = sizeof(typename vec_traits<STORAGE_T>::compute_t) * (nchans_in * (NACC + 1) + nchans_out) * block.y;

        s2_attn_bwd_ragged_generic_vec_k<THREADS, TWO_PASS><<<grid, block, shsize, stream>>>(
            nheads, nchans_in, nchans_out, npoints_in, npoints_out, _kxp, _vxp, _qyp, _dyp, _alpha_sum, _qdotk_max,
            _integral, _seg, _seg_off, _ring_base, _ring_size, _ring_weights, _dkxp, _dvxp, _dqyp);
        CHECK_ERROR("s2_attn_bwd_ragged_generic_vec_k");

        return;
    }

    // Instantiate both formulations and pick one at launch. TWO_PASS cannot be a runtime
    // flag -- it decides which accumulators exist, and so the register and shared-memory
    // footprint -- but it has to be selectable without a rebuild, which is the whole
    // point of keeping the two-pass code.
    template <typename STORAGE_T>
    static void launch_attn_bwd_ragged(int batch_size, int nheads, int nchans_in, int nchans_out, int64_t npoints_in,
                                       int64_t npoints_out, const STORAGE_T *_kxp, const STORAGE_T *_vxp,
                                       const STORAGE_T *_qyp, const STORAGE_T *_dyp, const float *_alpha_sum,
                                       const float *_qdotk_max, const float *_integral, const int32_t *_seg,
                                       const int32_t *_seg_off, const int64_t *_ring_base, const int64_t *_ring_size,
                                       const float *_ring_weights, typename vec_traits<STORAGE_T>::compute_t *_dkxp,
                                       typename vec_traits<STORAGE_T>::compute_t *_dvxp,
                                       typename vec_traits<STORAGE_T>::compute_t *_dqyp, bool _have_precise_y,
                                       cudaStream_t stream)
    {
        if (ragged_bwd_two_pass(c10::CppTypeToScalarType<STORAGE_T>::value, _have_precise_y)) {
            launch_gen_attn_bwd_ragged<true, STORAGE_T>(batch_size, nheads, nchans_in, nchans_out, npoints_in,
                                                        npoints_out, _kxp, _vxp, _qyp, _dyp, _alpha_sum, _qdotk_max,
                                                        _integral, _seg, _seg_off, _ring_base, _ring_size,
                                                        _ring_weights, _dkxp, _dvxp, _dqyp, stream);
            return;
        }
        launch_gen_attn_bwd_ragged<false, STORAGE_T>(
            batch_size, nheads, nchans_in, nchans_out, npoints_in, npoints_out, _kxp, _vxp, _qyp, _dyp, _alpha_sum,
            _qdotk_max, _integral, _seg, _seg_off, _ring_base, _ring_size, _ring_weights, _dkxp, _dvxp, _dqyp, stream);
    }

    // NHWC ABI, flattened: see s2_attention_fwd_ragged_cuda. Argument order mirrors
    // `forward_ragged` with dy and then the forward's three returns inserted after qy,
    // extending how `backward` mirrors `forward` on the product grids.
    //
    // y, alpha_sum and qdotk_max are what make one traversal possible: the first gives
    // integral by identity (1), the other two are the statistics the first walk used to
    // rebuild. The two-pass formulation ignores all three, so nothing about the caller
    // changes when TORCH_HARMONICS_RAGGED_BWD_TWO_PASS selects it.
    std::tuple<at::Tensor, at::Tensor, at::Tensor>
    s2_attention_bwd_ragged_cuda(at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor dy, at::Tensor y,
                                 at::Tensor y_hi, at::Tensor alpha_sum, at::Tensor qdotk_max, at::Tensor ring_weights,
                                 at::Tensor psi_seg, at::Tensor psi_seg_off, at::Tensor ring_base, at::Tensor ring_size,
                                 int64_t num_heads, int64_t npoints_out)
    {
        CHECK_CUDA_INPUT_TENSOR(kx);
        CHECK_CUDA_INPUT_TENSOR(vx);
        CHECK_CUDA_INPUT_TENSOR(qy);
        CHECK_CUDA_INPUT_TENSOR(dy);
        CHECK_CUDA_INPUT_TENSOR(y);
        CHECK_CUDA_INPUT_TENSOR(alpha_sum);
        CHECK_CUDA_INPUT_TENSOR(qdotk_max);
        CHECK_CUDA_TENSOR(ring_weights);
        CHECK_CUDA_TENSOR(psi_seg);
        CHECK_CUDA_TENSOR(psi_seg_off);
        CHECK_CUDA_TENSOR(ring_base);
        CHECK_CUDA_TENSOR(ring_size);

        TORCH_CHECK(kx.dim() == 3, "kx must be (B, npoints_in, num_heads * C_k), got ", kx.dim(), " dims");
        TORCH_CHECK(vx.dim() == 3, "vx must be (B, npoints_in, num_heads * C_v), got ", vx.dim(), " dims");
        TORCH_CHECK(qy.dim() == 3, "qy must be (B, npoints_out, num_heads * C_k), got ", qy.dim(), " dims");
        TORCH_CHECK(dy.dim() == 3, "dy must be (B, npoints_out, num_heads * C_v), got ", dy.dim(), " dims");
        TORCH_CHECK(y.dim() == 3, "y must be (B, npoints_out, num_heads * C_v), got ", y.dim(), " dims");

        TORCH_CHECK(num_heads >= 1, "num_heads must be positive, got ", num_heads);
        TORCH_CHECK(qy.size(2) % num_heads == 0, "q/k channel count (", qy.size(2),
                    ") must be divisible by num_heads (", num_heads, ")");
        TORCH_CHECK(vx.size(2) % num_heads == 0, "v channel count (", vx.size(2), ") must be divisible by num_heads (",
                    num_heads, ")");

        TORCH_CHECK(qy.size(1) == npoints_out, "qy has ", qy.size(1), " points but npoints_out is ", npoints_out);
        TORCH_CHECK(dy.size(1) == npoints_out, "dy has ", dy.size(1), " points but npoints_out is ", npoints_out);
        TORCH_CHECK(dy.size(2) == vx.size(2), "dy has ", dy.size(2), " channels but vx has ", vx.size(2));
        TORCH_CHECK(kx.size(1) == vx.size(1), "kx has ", kx.size(1), " points but vx has ", vx.size(1));
        TORCH_CHECK(psi_seg_off.size(0) == npoints_out + 1, "seg_off must have npoints_out + 1 = ", npoints_out + 1,
                    " entries, got ", psi_seg_off.size(0));
        TORCH_CHECK(psi_seg.dim() == 2 && psi_seg.size(1) == 3, "seg must be (nsegs, 3)");
        TORCH_CHECK(ring_base.size(0) == ring_size.size(0), "ring_base and ring_size must agree in length, got ",
                    ring_base.size(0), " and ", ring_size.size(0));
        TORCH_CHECK(ring_weights.size(0) == ring_base.size(0), "ring_weights must have one entry per input ring, got ",
                    ring_weights.size(0), " for ", ring_base.size(0), " rings");

        TORCH_CHECK(y.sizes() == dy.sizes(), "y must have dy's shape, got ", y.sizes(), " against ", dy.sizes());
        for (const auto &stat : {std::make_pair("alpha_sum", alpha_sum), std::make_pair("qdotk_max", qdotk_max)}) {
            TORCH_CHECK(stat.second.dim() == 3 && stat.second.size(0) == kx.size(0) && stat.second.size(1) == num_heads
                            && stat.second.size(2) == npoints_out,
                        stat.first, " must be (B, num_heads, npoints_out) = (", kx.size(0), ", ", num_heads, ", ",
                        npoints_out, "), got ", stat.second.sizes());
        }

        // Every activation must share one dtype: the dispatch below selects a single
        // scalar_t from qy and the launcher reinterpret_casts k/v/q/dy to it, so a
        // mismatched input would be reinterpreted rather than converted.
        TORCH_CHECK(kx.scalar_type() == qy.scalar_type(), "k dtype (", kx.scalar_type(), ") must match q dtype (",
                    qy.scalar_type(), ")");
        TORCH_CHECK(vx.scalar_type() == qy.scalar_type(), "v dtype (", vx.scalar_type(), ") must match q dtype (",
                    qy.scalar_type(), ")");
        TORCH_CHECK(dy.scalar_type() == qy.scalar_type(), "dy dtype (", dy.scalar_type(), ") must match q dtype (",
                    qy.scalar_type(), ")");
        TORCH_CHECK(y.scalar_type() == qy.scalar_type(), "y dtype (", y.scalar_type(), ") must match q dtype (",
                    qy.scalar_type(), ")");

        // ring_weights is read as float32 whatever scalar_t the dispatch picks, so it
        // is the one tensor that must not follow the activations. Casting a whole
        // module to bf16 would sweep it along, and reinterpreting those bytes as float
        // computes silent garbage instead of failing.
        TORCH_CHECK(ring_weights.scalar_type() == at::kFloat, "ring_weights must be float32, got ",
                    ring_weights.scalar_type());

        // The softmax statistics are fp32 whatever the activations are, for the same
        // reason: the kernel reads them as float unconditionally.
        TORCH_CHECK(alpha_sum.scalar_type() == at::kFloat, "alpha_sum must be float32, got ", alpha_sum.scalar_type());
        TORCH_CHECK(qdotk_max.scalar_type() == at::kFloat, "qdotk_max must be float32, got ", qdotk_max.scalar_type());

        // per-head channel counts; the packed extent is num_heads times these
        const int nchans_in = qy.size(2) / num_heads; // or kx.size(2) / num_heads
        const int nchans_out = vx.size(2) / num_heads;

        const int batch_size = kx.size(0);
        const int64_t npoints_in = kx.size(1);

        auto kx_type = kx.dtype();
        auto vx_type = vx.dtype();
        auto qy_type = qy.dtype();

        torch::Tensor dkx, dvx, dqy;

        // integral = dy . out per (batch, head, point), by identity (1). Done here
        // rather than in the kernel because at O(nchan) per point it is a rounding
        // error against the neighbourhood walk it replaces -- a reduction over 96
        // channels against 102 of them -- and because getting it wrong in torch is
        // visible, where getting it wrong in a warp reduction is not.
        //
        // fp32, and in fp32 arithmetic, because that is what the walk it replaces
        // accumulated in: vload widens every activation at the load site. Laid out to
        // match alpha_sum and qdotk_max, which the transpose is for -- the product is
        // natural in (batch, point, head) and the kernel indexes by batch * nheads +
        // head.
        //
        // From y_hi when the forward produced one. Upcasting y here would not help:
        // the precision was lost when it was stored, and it is precisely the term
        // that gets subtracted from quantities close to it.
        const bool have_precise_y = y_hi.defined() && y_hi.numel() > 0;
        const at::Tensor &y_for_integral = have_precise_y ? y_hi : y;

        const int64_t per_head[] = {batch_size, npoints_out, num_heads, nchans_out};
        const int64_t chan_dim = 3;
        torch::Tensor integral = (dy.reshape(per_head).to(at::kFloat) * y_for_integral.reshape(per_head).to(at::kFloat))
                                     .sum(chan_dim)
                                     .transpose(1, 2)
                                     .contiguous();

        // Activations stay in their native dtype and are widened to fp32 at load.
        // Gradient buffers are allocated fp32 because dkx/dvx are atomically
        // scatter-accumulated; they are narrowed back to the input dtype at the end.
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, qy.scalar_type(), "s2_attention_bwd_ragged_cuda", [&] {
            using storage_t = scalar_t;

            auto stream = at::cuda::getCurrentCUDAStream().stream();

            // zeros, not empty: dkx/dvx are accumulated into with atomicAdd, so the
            // buffers must start at zero. dqy is written outright, but is allocated
            // the same way to keep the three identical.
            const auto f32_like
                = [](const torch::Tensor &t) { return torch::zeros_like(t, t.options().dtype(torch::kFloat32)); };

            torch::Tensor dkxP = f32_like(kx);
            torch::Tensor dvxP = f32_like(vx);
            torch::Tensor dqyP = f32_like(qy);

            using compute_t = typename vec_traits<storage_t>::compute_t;

            launch_attn_bwd_ragged<storage_t>(
                batch_size, num_heads, nchans_in, nchans_out, npoints_in, npoints_out,
                reinterpret_cast<const storage_t *>(kx.data_ptr()), reinterpret_cast<const storage_t *>(vx.data_ptr()),
                reinterpret_cast<const storage_t *>(qy.data_ptr()), reinterpret_cast<const storage_t *>(dy.data_ptr()),
                reinterpret_cast<const float *>(alpha_sum.data_ptr()),
                reinterpret_cast<const float *>(qdotk_max.data_ptr()),
                reinterpret_cast<const float *>(integral.data_ptr()),
                reinterpret_cast<const int32_t *>(psi_seg.data_ptr()),
                reinterpret_cast<const int32_t *>(psi_seg_off.data_ptr()),
                reinterpret_cast<const int64_t *>(ring_base.data_ptr()),
                reinterpret_cast<const int64_t *>(ring_size.data_ptr()),
                reinterpret_cast<const float *>(ring_weights.data_ptr()),
                reinterpret_cast<compute_t *>(dkxP.data_ptr()), reinterpret_cast<compute_t *>(dvxP.data_ptr()),
                reinterpret_cast<compute_t *>(dqyP.data_ptr()), have_precise_y, stream);

            dkx = dkxP;
            dvx = dvxP;
            dqy = dqyP;
        });

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // convert precision back to starting dtype (no-op for fp32; narrows for fp16/bf16)
        dkx = dkx.to(kx_type);
        dvx = dvx.to(vx_type);
        dqy = dqy.to(qy_type);

        return std::make_tuple(dkx, dvx, dqy);
    }

    TORCH_LIBRARY_IMPL(attention_kernels, CUDA, m) { m.impl("backward_ragged", &s2_attention_bwd_ragged_cuda); }

} // namespace attention_kernels
