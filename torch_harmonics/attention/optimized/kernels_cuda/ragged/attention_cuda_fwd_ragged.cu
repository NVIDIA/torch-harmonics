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

// Forward neighborhood attention on a RAGGED isolatitude grid (HEALPix).
//
// Relation to the product-grid kernels in attention_cuda_fwd.cu
// -------------------------------------------------------------
// Those kernels store one neighbour list per output LATITUDE and slide it to the
// current longitude with an integer p-shift, wip = wi + pscale*wo. That works
// because rotating a product grid about the polar axis maps it onto itself. On a
// ragged grid it does not: rotating a ring by one of its own points maps that ring
// onto itself but not the rings above and below, whose point counts differ. So the
// pattern here is keyed per output POINT and the p-shift is gone.
//
// The trade is index reuse for footprint. The product-grid kernel reads one column
// list per latitude and reuses it across every longitude in the ring; this one reads
// a distinct list per point, so the pattern is npoints_out/nlat_out times larger --
// about 3*nside on HEALPix. In exchange the addressing gets strictly simpler: no
// pscale, no (ho, wo) decomposition, no wrap_lon on a shifted index.
//
// What carries over unchanged is the part that matters for speed. An arc is a run of
// consecutive points of one input ring, and RING ordering lays a ring out
// contiguously, so k/v reads along an arc are still stride-1 and the ring's
// quadrature weight is still an per-arc constant. The column advances by counting
// and wraps with a compare-and-subtract, so there is no integer division per
// neighbour -- which is the optimization the arc form exists for (see the commentary
// in attention_cuda_fwd.cu: emulated 64-bit division dominated the original
// col_idx-based inner loop).
//
// The benchmark has now said it is worth the code, so both variants are provided.
//
// The generic kernel below keeps one neighbour in flight at a time: address, load k,
// warp reduce, softmax, load v, with every step depending on the one before. That is
// the same structure attention_cuda_fwd.cu describes having replaced on the product
// grids, and it measures the same way -- 1.6 TFLOP/s at HEALPix level 5 on GB300,
// 0.02% of tensor-core peak and under 1% of HBM, so bound by neither arithmetic nor
// bandwidth but by the latency of a dependency chain 102 neighbours long. The
// accumulator sitting in shared memory puts a read-modify-write inside that chain as
// well, and q is re-read from global on every neighbour because the channel loop has
// a runtime bound and cannot be hoisted.
//
// s2_attn_fwd_ragged_special_vec_k is the port of the product-grid fix: the output
// accumulator moves to registers, q is staged once, and neighbours are processed in
// groups of NB so that NB independent k loads are outstanding before any is consumed.
// The online softmax then runs once per group rather than once per neighbour, which
// is the same reduction with one rescale per group instead of NB of them.
//
// It applies when the per-head channel count fits NLOC registers per lane and q/k and
// v agree on it; the generic kernel remains for everything else and as an escape
// hatch, since only the last of NLOC registers is bounds-checked and that argument
// depends on NLOC being exactly DIV_UP(nchan, BDIM_X).
//
// Why the softmax statistics are an output
// ----------------------------------------
// Both kernels finish holding alpha_sum and the final qdotk_max for their output
// point, and both used to throw them away. The backward then had to rebuild them,
// which took a whole traversal of the neighbourhood -- and a neighbourhood is ~102
// points, against the one float each statistic costs to store. Saving them is what
// lets attention_cuda_bwd_ragged.cu walk the arcs once instead of twice; see the
// identities at the top of that file. Nothing about the forward's own arithmetic
// changes: the two values are written before alpha_sum is inverted.

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

// Threads per block, and so warps per block, since BDIM_X is a warp. Overridable
// because the ragged kernel wants it swept independently of the product-grid one:
// warps in a block take consecutive output points, adjacent points share 74% of
// their neighbours by measurement, and so a wider block turns L1 into a shared cache
// for that overlap. The product grid has no equivalent gain and settled on 64.
#ifndef TH_ATTENTION_RAGGED_THREADS
#define TH_ATTENTION_RAGGED_THREADS (64)
#endif
#define THREADS (TH_ATTENTION_RAGGED_THREADS)

// Neighbours per group in the special kernel. Buys memory-level parallelism at the
// cost of NB accumulator sets, and this kernel's occupancy is register-limited.
//
// 4 is where ptxas puts the boundary, for nchan 96 (NLOC 3) at sm_100a:
//
//   NB    registers   spill   warps/SM
//    2       48         0        42
//    4       48         0        42
//    8       64        12        32
//
// so 4 is free relative to 2 and doubles the loads in flight, while 8 spills the
// accumulator to local memory, which defeats the purpose of holding it in registers.
// For reference the generic kernel needs 56 registers and gets 36 warps/SM, so the
// register-blocked kernel is cheaper despite keeping the accumulator in registers:
// the shared-memory addressing and the per-neighbour q reload it removes cost
// registers of their own. See benchmarks/ptxas_register_report.py.
#ifndef TH_ATTENTION_RAGGED_NB
#define TH_ATTENTION_RAGGED_NB (4)
#endif

// Largest number of COMPUTE_T registers per lane the special kernel will hold for
// the accumulator. 16 matches MAX_LOCAL_ARR_LEN in attention_cuda_fwd.cu, so with a
// 32-lane BDIM_X it covers up to 512 channels per head; HEALDA's dit-5B runs 96.
#define MAX_LOCAL_ARR_LEN_RAGGED (16)

namespace attention_kernels
{

    // One warp per output point: warp lanes span the channel dimension, so the
    // per-neighbour q.k reduction is a warp reduction and the running softmax state
    // lives in registers replicated across the warp.
    template <int BDIM_X, typename STORAGE_T>
    __global__ __launch_bounds__(BDIM_X) void s2_attn_fwd_ragged_generic_vec_k(
        int nheads,    // no. of attention heads packed along the channel dim
        int nchan_in,  // no. of STORAGE_T elements along channel dim, per head
        int nchan_out, // no. of STORAGE_T elements along channel dim, per head
        int64_t npoints_in, int64_t npoints_out, const STORAGE_T *__restrict__ kx, const STORAGE_T *__restrict__ vx,
        const STORAGE_T *__restrict__ qy, const int32_t *__restrict__ seg, const int32_t *__restrict__ seg_off,
        const int64_t *__restrict__ ring_base, const int64_t *__restrict__ ring_size,
        const float *__restrict__ ring_weights, STORAGE_T *__restrict__ y,
        // see the special kernel: the unnarrowed copy the bf16 one-pass backward needs,
        // or nullptr when the caller has not asked for it
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ y_hi,
        float *__restrict__ alpha_sum_out, // [batch][nheads][npoints_out], fp32
        float *__restrict__ qdotk_max_out) // [batch][nheads][npoints_out], fp32
    {
        using COMPUTE_T = typename vec_traits<STORAGE_T>::compute_t;

        extern __shared__ __align__(sizeof(float4)) float shext[];
        COMPUTE_T *shy = reinterpret_cast<COMPUTE_T *>(shext) + threadIdx.y * nchan_out;

        const int bh = blockIdx.y;
        const int batch = bh / nheads;
        const int head = bh - (batch * nheads);

        // leading dimensions: elements between adjacent spatial points
        const int64_t ldi = int64_t(nheads) * nchan_in;
        const int64_t ldo = int64_t(nheads) * nchan_out;

        const int64_t ipoint = int64_t(blockIdx.x) * blockDim.y + threadIdx.y;

        if (ipoint >= npoints_out) { return; }

        const int tidx = threadIdx.x;

        // No row_idx indirection. The product-grid kernel sorts output rows by
        // neighbour count to keep long rows off the tail of the grid; on a geodesic
        // neighbourhood of a near-uniform grid the counts barely vary (HEALPix pixels
        // are equal-area by construction), so the sort would cost a device-side sort
        // per call to balance nothing.
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { shy[chan] = __vset<COMPUTE_T>(0.f); }

        kx += int64_t(batch) * npoints_in * ldi + int64_t(head) * nchan_in;
        vx += int64_t(batch) * npoints_in * ldo + int64_t(head) * nchan_out;

        qy += int64_t(batch) * npoints_out * ldi + int64_t(head) * nchan_in + ipoint * ldi;
        y += int64_t(batch) * npoints_out * ldo + int64_t(head) * nchan_out + ipoint * ldo;
        // same offset, different element width -- see the special kernel
        if (y_hi != nullptr) { y_hi += int64_t(batch) * npoints_out * ldo + int64_t(head) * nchan_out + ipoint * ldo; }

        float alpha_sum = 0.0f;
        float qdotk_max = -FLT_MAX;

        const int seg_beg = seg_off[ipoint];
        const int seg_end = seg_off[ipoint + 1];

        for (int sg = seg_beg; sg < seg_end; sg++) {

            const int iring = seg[3 * sg + 0];
            const int lo = seg[3 * sg + 1];
            const int len = seg[3 * sg + 2];

            // constant along the arc: every point of a ring carries the same
            // quadrature weight, so this is hoisted exactly as ring_weights[hi] is
            // in the product-grid kernel
            const float qw = ring_weights[iring];

            // An arc is a run of consecutive points of one ring, and a ring is
            // contiguous in RING order, so the flat column is just the ring's base
            // plus an offset that counts up and wraps at the ring's end.
            const int64_t ring_lo = ring_base[iring];
            const int64_t ring_hi = ring_lo + ring_size[iring];

            int64_t col = ring_lo + lo;

            for (int j = 0; j < len; j++) {

                const STORAGE_T *_kx = kx + col * ldi;
                const STORAGE_T *_vx = vx + col * ldo;

                COMPUTE_T qdotkv = __vset<COMPUTE_T>(0.f);

                for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
                    qdotkv = __vadd(qdotkv, __vmul(vload(qy, chan), vload(_kx, chan)));
                }

                float qdotk = __warp_sum(__vred(qdotkv));

                // online (streaming) softmax, identical to the product-grid kernel:
                // rescale the running numerator and denominator whenever a new
                // neighbour raises the running maximum
                const float qdotk_max_tmp = max(qdotk_max, qdotk);
                const float alpha = expf(qdotk - qdotk_max_tmp) * qw;
                const float exp_save = expf(qdotk_max - qdotk_max_tmp);

                alpha_sum = alpha + alpha_sum * exp_save;

                for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
                    shy[chan] = __vadd(__vscale(exp_save, shy[chan]), __vscale(alpha, vload(_vx, chan)));
                }
                qdotk_max = qdotk_max_tmp;

                // next point in the arc; wraps at most once, so a compare-and-subtract
                // replaces the modulo the arc encoding is designed to avoid
                if (++col == ring_hi) { col = ring_lo; }
            }
        }

        // Both statistics are warp-uniform, so one lane stores them. They go out before
        // the reciprocal below, which is the whole of this kernel's contract with the
        // backward: alpha_sum as accumulated, qdotk_max as the walk left it.
        if (tidx == 0) {
            const int64_t istat = int64_t(bh) * npoints_out + ipoint;
            alpha_sum_out[istat] = alpha_sum;
            qdotk_max_out[istat] = qdotk_max;
        }

        alpha_sum = 1.0f / alpha_sum;
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
            const COMPUTE_T out = __vscale(alpha_sum, shy[chan]);
            vstore(y, chan, out);
            if (y_hi != nullptr) { vstore(y_hi, chan, out); }
        }

        return;
    }

    // Register-blocked, group-scheduled counterpart of the kernel above. The port of
    // s2_attn_fwd_special_vec_k in attention_cuda_fwd.cu; everything that differs is
    // addressing, because a ragged arc walks a contiguous run of one ring rather than
    // a longitude run of a product row, so there is no (ho, wo) decomposition, no
    // pscale and no wrap_lon -- just a column that counts and wraps at the ring end.
    //
    // NLOC must be exactly DIV_UP(nchan, BDIM_X). The unrolled loops below leave every
    // register but the last unguarded, which is sound only under that equality: for
    // i <= NLOC-2, i*BDIM_X + tidx <= (NLOC-1)*BDIM_X - 1 < nchan. The launcher
    // enforces it, and requires nchan_in == nchan_out so one NLOC serves both.
    template <int BDIM_X, int BDIM_Y, int NLOC, typename STORAGE_T>
    __global__ __launch_bounds__(BDIM_X *BDIM_Y) void s2_attn_fwd_ragged_special_vec_k(
        int nheads,    // no. of attention heads packed along the channel dim
        int nchan_in,  // no. of STORAGE_T elements along channel dim, per head
        int nchan_out, // no. of STORAGE_T elements along channel dim, per head
        int64_t npoints_in, int64_t npoints_out, const STORAGE_T *__restrict__ kx, const STORAGE_T *__restrict__ vx,
        const STORAGE_T *__restrict__ qy, const int32_t *__restrict__ seg, const int32_t *__restrict__ seg_off,
        const int64_t *__restrict__ ring_base, const int64_t *__restrict__ ring_size,
        const float *__restrict__ ring_weights, STORAGE_T *__restrict__ y,
        // The same output again, unnarrowed, or nullptr. The backward's one-pass form
        // gets integral = dy . out from the stored output, and in bf16 that output's 8
        // mantissa bits are not enough: integral is subtracted from quantities close to
        // it, so the cancellation amplifies the rounding past the suite's tolerance.
        // Writing a second, full-precision copy is what lets bf16 take the one-pass
        // path. Allocated by the caller only when it is needed, hence the null check.
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ y_hi,
        float *__restrict__ alpha_sum_out, // [batch][nheads][npoints_out], fp32
        float *__restrict__ qdotk_max_out) // [batch][nheads][npoints_out], fp32
    {
        using COMPUTE_T = typename vec_traits<STORAGE_T>::compute_t;

        static_assert(BDIM_X == WARP_SIZE, "the ragged special kernel reduces with __warp_sum");
        static_assert(NLOC >= 1);

        constexpr int NLOC_M1 = NLOC - 1;
        constexpr int NB = TH_ATTENTION_RAGGED_NB;

        // q staged per warp, so the neighbour loop reads it from shared instead of
        // re-reading global on every one of the ~102 iterations.
        extern __shared__ __align__(sizeof(float4)) float shext[];
        COMPUTE_T *shq = reinterpret_cast<COMPUTE_T *>(shext) + threadIdx.y * nchan_in;

        const int tidx = threadIdx.x;

        const int bh = blockIdx.y;
        const int batch = bh / nheads;
        const int head = bh - (batch * nheads);

        const int64_t ldi = int64_t(nheads) * nchan_in;
        const int64_t ldo = int64_t(nheads) * nchan_out;

        const int64_t ipoint = int64_t(blockIdx.x) * blockDim.y + threadIdx.y;

        if (ipoint >= npoints_out) { return; }

        qy += int64_t(batch) * npoints_out * ldi + int64_t(head) * nchan_in + ipoint * ldi;

        for (int chan = tidx; chan < nchan_in; chan += BDIM_X) { shq[chan] = vload(qy, chan); }
        // Lanes read each other's entries below, and a warp is not implicitly in step
        // on every architecture.
        __syncwarp();

        // the lane's channel offset folded into the base pointers, as on the product
        // grid, so the inner loops index by register rather than by channel
        kx += int64_t(batch) * npoints_in * ldi + int64_t(head) * nchan_in + tidx;
        vx += int64_t(batch) * npoints_in * ldo + int64_t(head) * nchan_out + tidx;
        y += int64_t(batch) * npoints_out * ldo + int64_t(head) * nchan_out + ipoint * ldo + tidx;
        // COMPUTE_T and STORAGE_T are one vector element each in this indexing -- the
        // bf16 path pairs bf164 with float4, four channels either way -- so the offset
        // is the same expression and only the element width differs.
        if (y_hi != nullptr) {
            y_hi += int64_t(batch) * npoints_out * ldo + int64_t(head) * nchan_out + ipoint * ldo + tidx;
        }

        COMPUTE_T locy[NLOC];
#pragma unroll
        for (int i = 0; i < NLOC; i++) { locy[i] = __vset<COMPUTE_T>(0.f); }

        float alpha_sum = 0.0f;
        float qdotk_max = -FLT_MAX;

        const int seg_beg = seg_off[ipoint];
        const int seg_end = seg_off[ipoint + 1];

        for (int sg = seg_beg; sg < seg_end; sg++) {

            const int iring = seg[3 * sg + 0];
            const int lo = seg[3 * sg + 1];
            const int len = seg[3 * sg + 2];

            const float qw_seg = ring_weights[iring];

            const int64_t ring_lo = ring_base[iring];
            const int64_t ring_hi = ring_lo + ring_size[iring];

            int64_t col = ring_lo + lo;

            int j = 0;
            for (; j + NB <= len; j += NB) {

                // addresses first, so the NB loads that follow have nothing to wait on
                //
                // The two 64-bit multiplies per neighbour look like an obvious target:
                // col advances by one and wraps at most once per arc, which is what
                // the arc encoding is for, so they could be one add each with the
                // multiply hoisted to the arc. Tried, and it loses. Carrying the
                // running pointers costs registers -- NLOC=3 BFloat16 goes from 56 to
                // 72 and occupancy from 56% to 44%, or 64 and 50% if the wrap targets
                // are held instead, which makes fp32 spill. Measured at nside 64 the
                // forward went 1.363 -> 1.522 ms in bf16 and 1.266 -> 1.430 in fp32,
                // so the lost occupancy outweighs the saved integer work by about 12%.
                //
                // This is the same trade e9092eb found cancelling out, and it says the
                // kernel is not short of issue slots so much as short of warps. The
                // version that might still pay is splitting each arc at its seam into
                // two runs, so the inner loop is a pure increment with no wrap test and
                // no extra live pointers -- that needs the NB grouping to respect the
                // run boundary, which is why it was not the first thing tried.
                const STORAGE_T *kp[NB];
                const STORAGE_T *vp[NB];
#pragma unroll
                for (int u = 0; u < NB; u++) {
                    kp[u] = kx + col * ldi;
                    vp[u] = vx + col * ldo;
                    if (++col == ring_hi) { col = ring_lo; }
                }

                COMPUTE_T acc[NB];
#pragma unroll
                for (int u = 0; u < NB; u++) { acc[u] = __vset<COMPUTE_T>(0.f); }

                // one channel step feeds NB accumulators, so NB k loads are in flight
#pragma unroll
                for (int i = 0; i < NLOC_M1; i++) {
                    const COMPUTE_T q = shq[i * BDIM_X + tidx];
#pragma unroll
                    for (int u = 0; u < NB; u++) { acc[u] = __vadd(acc[u], __vmul(q, vload(kp[u], i * BDIM_X))); }
                }
                if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                    const COMPUTE_T q = shq[NLOC_M1 * BDIM_X + tidx];
#pragma unroll
                    for (int u = 0; u < NB; u++) { acc[u] = __vadd(acc[u], __vmul(q, vload(kp[u], NLOC_M1 * BDIM_X))); }
                }

                float qdotk[NB];
#pragma unroll
                for (int u = 0; u < NB; u++) { qdotk[u] = __warp_sum(__vred(acc[u])); }

                // group-wise online softmax: one running max and one rescale for all NB
                float qdotk_max_tmp = qdotk_max;
#pragma unroll
                for (int u = 0; u < NB; u++) { qdotk_max_tmp = max(qdotk_max_tmp, qdotk[u]); }
                const float exp_save = expf(qdotk_max - qdotk_max_tmp);

                float alpha[NB];
                float alpha_grp = 0.0f;
#pragma unroll
                for (int u = 0; u < NB; u++) {
                    alpha[u] = expf(qdotk[u] - qdotk_max_tmp) * qw_seg;
                    alpha_grp += alpha[u];
                }
                alpha_sum = alpha_grp + alpha_sum * exp_save;

#pragma unroll
                for (int i = 0; i < NLOC_M1; i++) {
                    COMPUTE_T t = __vscale(exp_save, locy[i]);
#pragma unroll
                    for (int u = 0; u < NB; u++) { t = __vadd(t, __vscale(alpha[u], vload(vp[u], i * BDIM_X))); }
                    locy[i] = t;
                }
                if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
                    COMPUTE_T t = __vscale(exp_save, locy[NLOC_M1]);
#pragma unroll
                    for (int u = 0; u < NB; u++) { t = __vadd(t, __vscale(alpha[u], vload(vp[u], NLOC_M1 * BDIM_X))); }
                    locy[NLOC_M1] = t;
                }

                qdotk_max = qdotk_max_tmp;
            }

            // remainder: fewer than NB neighbours left in this arc
            for (; j < len; j++) {

                const STORAGE_T *_kx = kx + col * ldi;
                const STORAGE_T *_vx = vx + col * ldo;

                COMPUTE_T qdotkv = __vset<COMPUTE_T>(0.f);
#pragma unroll
                for (int i = 0; i < NLOC_M1; i++) {
                    qdotkv = __vadd(qdotkv, __vmul(shq[i * BDIM_X + tidx], vload(_kx, i * BDIM_X)));
                }
                if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                    qdotkv = __vadd(qdotkv, __vmul(shq[NLOC_M1 * BDIM_X + tidx], vload(_kx, NLOC_M1 * BDIM_X)));
                }

                const float qdotk = __warp_sum(__vred(qdotkv));

                const float qdotk_max_tmp = max(qdotk_max, qdotk);
                const float alpha = expf(qdotk - qdotk_max_tmp) * qw_seg;
                const float exp_save = expf(qdotk_max - qdotk_max_tmp);

                alpha_sum = alpha + alpha_sum * exp_save;

#pragma unroll
                for (int i = 0; i < NLOC_M1; i++) {
                    locy[i] = __vadd(__vscale(exp_save, locy[i]), __vscale(alpha, vload(_vx, i * BDIM_X)));
                }
                if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
                    locy[NLOC_M1]
                        = __vadd(__vscale(exp_save, locy[NLOC_M1]), __vscale(alpha, vload(_vx, NLOC_M1 * BDIM_X)));
                }

                qdotk_max = qdotk_max_tmp;

                if (++col == ring_hi) { col = ring_lo; }
            }
        }

        // see the generic kernel: written as accumulated, before the reciprocal
        if (tidx == 0) {
            const int64_t istat = int64_t(bh) * npoints_out + ipoint;
            alpha_sum_out[istat] = alpha_sum;
            qdotk_max_out[istat] = qdotk_max;
        }

        const float alpha_inv = 1.0f / alpha_sum;
#pragma unroll
        for (int i = 0; i < NLOC_M1; i++) {
            const COMPUTE_T out = __vscale(alpha_inv, locy[i]);
            vstore(y, i * BDIM_X, out);
            if (y_hi != nullptr) { vstore(y_hi, i * BDIM_X, out); }
        }
        if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
            const COMPUTE_T out = __vscale(alpha_inv, locy[NLOC_M1]);
            vstore(y, NLOC_M1 * BDIM_X, out);
            if (y_hi != nullptr) { vstore(y_hi, NLOC_M1 * BDIM_X, out); }
        }

        return;
    }

    // Resolve NLOC, which has to be a compile-time constant, from the runtime channel
    // count by walking the supported range. Mirrors launch_spc_attn_fwd.
    template <int BDIM_X, int BDIM_Y, int CUR_LOC, int MAX_LOC, typename STORAGE_T>
    static void launch_spc_attn_fwd_ragged(int nloc, int batch_size, int nheads, int nchans_in, int nchans_out,
                                           int64_t npoints_in, int64_t npoints_out, const STORAGE_T *__restrict__ _kxp,
                                           const STORAGE_T *__restrict__ _vxp, const STORAGE_T *__restrict__ _qyp,
                                           const int32_t *_seg, const int32_t *_seg_off, const int64_t *_ring_base,
                                           const int64_t *_ring_size, const float *_ring_weights,
                                           STORAGE_T *__restrict__ _yp,
                                           typename vec_traits<STORAGE_T>::compute_t *__restrict__ _y_hi,
                                           float *_alpha_sum, float *_qdotk_max, cudaStream_t stream)
    {
        if constexpr (CUR_LOC > MAX_LOC) {
            TORCH_CHECK(false, "ragged special attention kernel reached nloc ", nloc, " above its bound ", MAX_LOC);
            return;
        } else {
            if (CUR_LOC == nloc) {
                dim3 block(BDIM_X, BDIM_Y);
                dim3 grid(DIV_UP(npoints_out, block.y), batch_size * nheads);

                // only q is staged; the accumulator is in registers, which is the point
                size_t shsize = sizeof(typename vec_traits<STORAGE_T>::compute_t) * nchans_in * block.y;

                s2_attn_fwd_ragged_special_vec_k<BDIM_X, BDIM_Y, CUR_LOC><<<grid, block, shsize, stream>>>(
                    nheads, nchans_in, nchans_out, npoints_in, npoints_out, _kxp, _vxp, _qyp, _seg, _seg_off,
                    _ring_base, _ring_size, _ring_weights, _yp, _y_hi, _alpha_sum, _qdotk_max);
                CHECK_ERROR("s2_attn_fwd_ragged_special_vec_k");
                return;
            }
            launch_spc_attn_fwd_ragged<BDIM_X, BDIM_Y, CUR_LOC + 1, MAX_LOC, STORAGE_T>(
                nloc, batch_size, nheads, nchans_in, nchans_out, npoints_in, npoints_out, _kxp, _vxp, _qyp, _seg,
                _seg_off, _ring_base, _ring_size, _ring_weights, _yp, _y_hi, _alpha_sum, _qdotk_max, stream);
        }
    }

    // Set TORCH_HARMONICS_RAGGED_GENERIC=1 to force the original kernel. The two
    // compute the same function, so this is an A/B switch for the benchmark and a way
    // to fall back without rebuilding if the register-blocked path misbehaves.
    static bool ragged_force_generic()
    {
        static const bool forced = []() {
            const char *env = std::getenv("TORCH_HARMONICS_RAGGED_GENERIC");
            return env != nullptr && env[0] == '1';
        }();
        return forced;
    }

    template <typename STORAGE_T>
    static void launch_gen_attn_fwd_ragged(int batch_size, int nheads, int nchans_in, int nchans_out,
                                           int64_t npoints_in, int64_t npoints_out, const STORAGE_T *__restrict__ _kxp,
                                           const STORAGE_T *__restrict__ _vxp, const STORAGE_T *__restrict__ _qyp,
                                           const int32_t *_seg, const int32_t *_seg_off, const int64_t *_ring_base,
                                           const int64_t *_ring_size, const float *_ring_weights,
                                           STORAGE_T *__restrict__ _yp,
                                           typename vec_traits<STORAGE_T>::compute_t *__restrict__ _y_hi,
                                           float *_alpha_sum, float *_qdotk_max, cudaStream_t stream)
    {
        // The register-blocked kernel needs NLOC == DIV_UP(nchan, WARP_SIZE) to hold for
        // both channel counts at once, which is why the equality is required rather
        // than taking the larger: NLOC also decides which registers go unguarded.
        const int nloc = DIV_UP(nchans_out, WARP_SIZE);
        if (!ragged_force_generic() && nchans_in == nchans_out && nloc <= MAX_LOCAL_ARR_LEN_RAGGED) {
            launch_spc_attn_fwd_ragged<WARP_SIZE, THREADS / WARP_SIZE, 1, MAX_LOCAL_ARR_LEN_RAGGED, STORAGE_T>(
                nloc, batch_size, nheads, nchans_in, nchans_out, npoints_in, npoints_out, _kxp, _vxp, _qyp, _seg,
                _seg_off, _ring_base, _ring_size, _ring_weights, _yp, _y_hi, _alpha_sum, _qdotk_max, stream);
            return;
        }

        dim3 block(WARP_SIZE, THREADS / WARP_SIZE);
        // one block row per (batch, head) pair
        dim3 grid(DIV_UP(npoints_out, block.y), batch_size * nheads);

        // shared memory holds compute-type (COMPUTE_T) data, not STORAGE_T.
        // sized from the per-head channel count, so it does not scale with nheads
        size_t shsize = sizeof(typename vec_traits<STORAGE_T>::compute_t) * nchans_out * block.y;

        s2_attn_fwd_ragged_generic_vec_k<THREADS><<<grid, block, shsize, stream>>>(
            nheads, nchans_in, nchans_out, npoints_in, npoints_out, _kxp, _vxp, _qyp, _seg, _seg_off, _ring_base,
            _ring_size, _ring_weights, _yp, _y_hi, _alpha_sum, _qdotk_max);
        CHECK_ERROR("s2_attn_fwd_ragged_generic_vec_k");

        return;
    }

    // NHWC ABI, flattened: kx, vx, qy are physically (B, npoints, num_heads * nchan)
    // and contiguous, with the spatial axes of the product-grid ABI collapsed into
    // one. Layout is never inferred from strides -- the caller states it by
    // construction (see attention/_layout.py). Heads stay packed along the channel
    // dimension for the same reason as on the product grids: folding them into the
    // batch dimension is not free in a channel-innermost layout.
    //
    // Returns (y, alpha_sum, qdotk_max). The two statistics are per (batch, head,
    // point) and fp32 whatever the activations are, for the same reason ring_weights
    // is: they are softmax bookkeeping, not activations, and the backward reads them
    // as float unconditionally.
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    s2_attention_fwd_ragged_cuda(at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor ring_weights,
                                 at::Tensor psi_seg, at::Tensor psi_seg_off, at::Tensor ring_base, at::Tensor ring_size,
                                 int64_t num_heads, int64_t npoints_out)
    {
        CHECK_CUDA_INPUT_TENSOR(kx);
        CHECK_CUDA_INPUT_TENSOR(vx);
        CHECK_CUDA_INPUT_TENSOR(qy);
        CHECK_CUDA_TENSOR(ring_weights);
        CHECK_CUDA_TENSOR(psi_seg);
        CHECK_CUDA_TENSOR(psi_seg_off);
        CHECK_CUDA_TENSOR(ring_base);
        CHECK_CUDA_TENSOR(ring_size);

        TORCH_CHECK(kx.dim() == 3, "kx must be (B, npoints_in, num_heads * C_k), got ", kx.dim(), " dims");
        TORCH_CHECK(vx.dim() == 3, "vx must be (B, npoints_in, num_heads * C_v), got ", vx.dim(), " dims");
        TORCH_CHECK(qy.dim() == 3, "qy must be (B, npoints_out, num_heads * C_k), got ", qy.dim(), " dims");

        TORCH_CHECK(num_heads >= 1, "num_heads must be positive, got ", num_heads);
        TORCH_CHECK(qy.size(2) % num_heads == 0, "q/k channel count (", qy.size(2),
                    ") must be divisible by num_heads (", num_heads, ")");
        TORCH_CHECK(vx.size(2) % num_heads == 0, "v channel count (", vx.size(2), ") must be divisible by num_heads (",
                    num_heads, ")");

        TORCH_CHECK(qy.size(1) == npoints_out, "qy has ", qy.size(1), " points but npoints_out is ", npoints_out);
        TORCH_CHECK(kx.size(1) == vx.size(1), "kx has ", kx.size(1), " points but vx has ", vx.size(1));
        TORCH_CHECK(psi_seg_off.size(0) == npoints_out + 1, "seg_off must have npoints_out + 1 = ", npoints_out + 1,
                    " entries, got ", psi_seg_off.size(0));
        TORCH_CHECK(psi_seg.dim() == 2 && psi_seg.size(1) == 3, "seg must be (nsegs, 3)");
        TORCH_CHECK(ring_base.size(0) == ring_size.size(0), "ring_base and ring_size must agree in length, got ",
                    ring_base.size(0), " and ", ring_size.size(0));
        TORCH_CHECK(ring_weights.size(0) == ring_base.size(0), "ring_weights must have one entry per input ring, got ",
                    ring_weights.size(0), " for ", ring_base.size(0), " rings");

        // Every activation must share one dtype: the dispatch below selects a single
        // scalar_t from qy and the launcher reinterpret_casts k/v/q to it, so a
        // mismatched input would be reinterpreted rather than converted.
        TORCH_CHECK(kx.scalar_type() == qy.scalar_type(), "k dtype (", kx.scalar_type(), ") must match q dtype (",
                    qy.scalar_type(), ")");
        TORCH_CHECK(vx.scalar_type() == qy.scalar_type(), "v dtype (", vx.scalar_type(), ") must match q dtype (",
                    qy.scalar_type(), ")");

        // ring_weights is read as float32 whatever scalar_t the dispatch picks, so it
        // is the one tensor that must not follow the activations. Casting a whole
        // module to bf16 would sweep it along, and reinterpreting those bytes as float
        // computes silent garbage instead of failing.
        TORCH_CHECK(ring_weights.scalar_type() == at::kFloat, "ring_weights must be float32, got ",
                    ring_weights.scalar_type());

        // per-head channel counts; the packed extent is num_heads times these
        const int nchans_in = qy.size(2) / num_heads; // or kx.size(2) / num_heads
        const int nchans_out = vx.size(2) / num_heads;

        const int batch_size = kx.size(0);
        const int64_t npoints_in = kx.size(1);

        auto qy_type = qy.dtype();
        const int64_t out_dims[] = {batch_size, npoints_out, int64_t(nchans_out) * num_heads};
        torch::Tensor y;

        // One entry per (batch, head, point), which is the kernel's own indexing: a warp
        // owns one output point and blockIdx.y is batch * nheads + head. empty(), not
        // zeros(): every entry is written by the warp that owns it, and the grid covers
        // every (batch, head, point) triple.
        const int64_t stat_dims[] = {batch_size, num_heads, npoints_out};
        torch::Tensor alpha_sum = torch::empty(stat_dims, kx.options().dtype(torch::kFloat32));
        torch::Tensor qdotk_max = torch::empty(stat_dims, kx.options().dtype(torch::kFloat32));

        // An fp32 copy of the output, for bf16 only, so the backward can take its
        // one-pass form there. It costs a second output-sized tensor, which is why it
        // is not allocated for the dtypes whose stored output is already precise
        // enough: fp32 trivially, and fp16 because its 11 mantissa bits keep the
        // cancellation in (gdotv_i - integral) inside tolerance where bf16's 8 do not.
        // Empty otherwise, and the kernel takes nullptr and skips the store.
        const bool want_y_hi = (qy.scalar_type() == at::kBFloat16);
        torch::Tensor y_hi = want_y_hi ? torch::empty(out_dims, kx.options().dtype(torch::kFloat32)) :
                                         torch::empty({0}, kx.options().dtype(torch::kFloat32));

        // Activations stay in their native dtype and y is allocated in it, so there is
        // no whole-tensor fp32 copy and the read bandwidth for fp16/bf16 is halved.
        // The kernel widens to fp32 at load and narrows back at store; compute and
        // softmax accumulation are fp32 in-kernel either way.
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, qy.scalar_type(), "s2_attention_fwd_ragged_cuda", [&] {
            using storage_t = scalar_t;

            auto stream = at::cuda::getCurrentCUDAStream().stream();

            torch::Tensor y_nhwc = torch::empty(out_dims, kx.options()); // native dtype

            launch_gen_attn_fwd_ragged<storage_t>(
                batch_size, num_heads, nchans_in, nchans_out, npoints_in, npoints_out,
                reinterpret_cast<const storage_t *>(kx.data_ptr()), reinterpret_cast<const storage_t *>(vx.data_ptr()),
                reinterpret_cast<const storage_t *>(qy.data_ptr()), reinterpret_cast<const int32_t *>(psi_seg.data_ptr()),
                reinterpret_cast<const int32_t *>(psi_seg_off.data_ptr()),
                reinterpret_cast<const int64_t *>(ring_base.data_ptr()),
                reinterpret_cast<const int64_t *>(ring_size.data_ptr()),
                reinterpret_cast<const float *>(ring_weights.data_ptr()),
                reinterpret_cast<storage_t *>(y_nhwc.data_ptr()),
                want_y_hi ? reinterpret_cast<typename vec_traits<storage_t>::compute_t *>(y_hi.data_ptr()) : nullptr,
                reinterpret_cast<float *>(alpha_sum.data_ptr()), reinterpret_cast<float *>(qdotk_max.data_ptr()), stream);

            y = y_nhwc;
        });

        y = y.to(qy_type);

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return std::make_tuple(y, y_hi, alpha_sum, qdotk_max);
    }

    TORCH_LIBRARY_IMPL(attention_kernels, CUDA, m) { m.impl("forward_ragged", &s2_attention_fwd_ragged_cuda); }

} // namespace attention_kernels
