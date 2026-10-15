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

// =====================================================================================
// Upsample (scatter-style) attention forward, RING-STEP variant — CUDA
// =====================================================================================
//
// Used by DistributedNeighborhoodAttentionS2 in the upsample direction
// (nlon_out % nlon_in == 0). K/V live on the coarse INPUT grid and are sharded
// along longitude across an azimuth process group; Q and the output live on the
// fine OUTPUT grid and stay local. Each call processes one rotating K/V chunk.
//
// psi is the serial scatter arc form (see attention_cuda_fwd_upsample.cu), sliced to
// this rank (built by RingUpsampleBackend in distributed_attention.py):
//   seg_off : indexed by hi_local in [0, nlat_halo], the halo-padded LOCAL input
//             latitude rows; hi_global = lat_halo_start + hi_local. Pole-padding
//             rows (hi_global outside the global grid) are empty.
//   seg     : (ho_local, lo, len) arcs on the fine output grid, only those whose
//             output row is owned by this polar rank, with ho_local indexing the
//             LOCAL output rows and lo pre-shifted by -lon_lo_out (mod
//             nlon_out_global). The serial shift
//                w = (lo + pscale_out * wi_global) mod nlon_out_global
//             then gives the arc start relative to lon_lo_out, and clip_arc keeps
//             the part inside [0, nlon_out), the LOCAL output width.
//
// Because the softmax-normalization cell (fine output) is NOT the row key
// (coarse input), the per-ring-step online softmax cannot be done warp-locally
// like in the gather ring kernel. Instead each step runs the serial scatter
// kernels' max-then-accumulate scheme, with a rescale in between that folds the
// step into the state carried over from the previous ones:
//   phase 1 (max):     scatter q.k -> atomicMax into qdotk_max_curr (a copy of
//                      the running qdotk_max_buf)
//   phase 2 (rescale): per local output cell, y_acc / alpha_sum are rescaled by
//                      exp(old_max - new_max) and qdotk_max_buf := qdotk_max_curr
//   phase 3 (acc):     scatter alpha = exp(q.k - new_max)*w_quad -> atomicAdd
//                      into alpha_sum_buf, and alpha * v -> atomicAdd into y_acc
// Finalization (y = y_acc / alpha_sum) happens once in Python after the last step.
//
// Generic-only (scalar loads), mirroring the serial upsample scatter kernels in
// attention_cuda_fwd_upsample.cu.
//
// Heads are packed along the channel dim as in the serial kernels: kx/vx/qy and
// y_acc are (B, H, W, nheads * nchan), gridDim.y spans batch * nheads, and
// nchan_in / nchan_out are per-head counts. The softmax stats (alpha_sum,
// qdotk_max) are (B, nheads, H, W).
// =====================================================================================

#include "../common/attention_cuda.cuh"
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>
#include <cfloat>

#include "../common/cudamacro.h"
#include "../common/attention_cuda_utils.cuh"

#define THREADS (64)

namespace attention_kernels
{

    // float atomicMax (no native float overload); CAS loop, correct for any sign.
    __device__ __forceinline__ float ring_up_atomicMaxf(float *addr, float val)
    {
        int *ai = reinterpret_cast<int *>(addr);
        int old = *ai;
        while (val > __int_as_float(old)) {
            const int assumed = old;
            old = atomicCAS(ai, assumed, __float_as_int(val));
            if (old == assumed) { break; }
        }
        return __int_as_float(old);
    }

    // map a coarse longitude wi + (shifted) canonical fine longitude to the fine
    // longitude relative to lon_lo_out; see scatter_wo in attention_cuda_fwd_upsample.cu
    __device__ __forceinline__ int ring_up_scatter_wo(int wo_canonical, int wi, int pscale_out, int nlon_out)
    {
        int wo = wo_canonical + pscale_out * wi; // < 2*nlon_out since both terms < nlon_out
        if (wo >= nlon_out) { wo -= nlon_out; }
        return wo;
    }

    // phase 1: per coarse chunk cell, scatter q.k into the per-output-cell running max.
    template <int THREADS_PER_BLOCK, typename STORAGE_T>
    __global__ __launch_bounds__(THREADS_PER_BLOCK) void s2_attn_fwd_ring_upsample_max_k(
        int nheads, int nchan_in, int nlat_halo, int nlon_kx, int nlon_out_global, int pscale_out, int lon_lo_kx,
        int nlat_out, int nlon_out, const STORAGE_T *__restrict__ kx, const STORAGE_T *__restrict__ qy,
        const int32_t *__restrict__ seg, const int32_t *__restrict__ seg_off, float *__restrict__ qdotk_max_curr)
    {
        extern __shared__ float shext[];
        float *sh_k = shext + threadIdx.y * nchan_in;

        // one block row per (batch, head) pair
        const int bh = blockIdx.y;
        const int batch = bh / nheads;
        const int head = bh - (batch * nheads);
        const int64_t ldi = int64_t(nheads) * nchan_in;

        const int wid = blockIdx.x * blockDim.y + threadIdx.y;
        if (wid >= nlat_halo * nlon_kx) { return; }

        const int tidx = threadIdx.x;
        const int hi = wid / nlon_kx; // LOCAL halo row
        const int wi_local = wid - hi * nlon_kx;
        const int wi_global = lon_lo_kx + wi_local;

        const int seg_beg = seg_off[hi];
        const int seg_end = seg_off[hi + 1];
        if (seg_beg == seg_end) { return; } // empty row (e.g. pole padding)

        kx += int64_t(batch) * nlat_halo * nlon_kx * ldi + (int64_t(hi) * nlon_kx + wi_local) * ldi
            + int64_t(head) * nchan_in;
        qy += int64_t(batch) * nlat_out * nlon_out * ldi + int64_t(head) * nchan_in;
        qdotk_max_curr += int64_t(bh) * nlat_out * nlon_out;

        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) { sh_k[chan] = vload(kx, chan); }

        for (int sg = seg_beg; sg < seg_end; sg++) {

            const int ho = seg[3 * sg + 0]; // LOCAL output row
            const int seg_lo = seg[3 * sg + 1];
            const int seg_len = seg[3 * sg + 2];

            // the part of the arc on this rank's output longitudes
            int2 piece[2];
            const int npiece = clip_arc(ring_up_scatter_wo(seg_lo, wi_global, pscale_out, nlon_out_global), seg_len,
                                        nlon_out_global, 0, nlon_out, piece);

            for (int pc = 0; pc < npiece; pc++) {

                int wo = piece[pc].x;

                for (int j = 0; j < piece[pc].y; j++) {
                    const STORAGE_T *_qy = qy + (int64_t(ho) * nlon_out + wo) * ldi;

                    float qd = 0.f;
                    for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) { qd += sh_k[chan] * vload(_qy, chan); }
                    qd = __warp_sum(qd);
                    if (tidx == 0) { ring_up_atomicMaxf(&qdotk_max_curr[int64_t(ho) * nlon_out + wo], qd); }

                    // pieces do not wrap: clip_arc split the arc at the seam
                    wo++;
                }
            }
        }
    }

    // phase 2: per LOCAL output cell, rescale the running state by exp(old_max - new_max)
    // and commit the new max into the persistent buffer.
    template <int THREADS_PER_BLOCK>
    __global__ __launch_bounds__(THREADS_PER_BLOCK) void s2_attn_fwd_ring_upsample_rescale_k(
        int nheads, int nchan_out, int nlat_out, int nlon_out, const float *__restrict__ qdotk_max_curr,
        float *__restrict__ qdotk_max_buf, float *__restrict__ alpha_sum_buf, float *__restrict__ y_acc)
    {
        // one block row per (batch, head) pair
        const int bh = blockIdx.y;
        const int batch = bh / nheads;
        const int head = bh - (batch * nheads);
        const int64_t ldo = int64_t(nheads) * nchan_out;

        const int wid = blockIdx.x * blockDim.y + threadIdx.y;
        if (wid >= nlat_out * nlon_out) { return; }
        const int tidx = threadIdx.x;

        const int64_t cell = int64_t(bh) * nlat_out * nlon_out + wid;
        const float qdotk_max_old = qdotk_max_buf[cell];
        const float qdotk_max_new = qdotk_max_curr[cell];

        // covers the untouched case, including -inf == -inf (exp would yield NaN)
        if (qdotk_max_old == qdotk_max_new) { return; }

        const float corr = expf(qdotk_max_old - qdotk_max_new);

        float *_y_acc = y_acc + (int64_t(batch) * nlat_out * nlon_out + wid) * ldo + int64_t(head) * nchan_out;
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { _y_acc[chan] *= corr; }

        if (tidx == 0) {
            alpha_sum_buf[cell] *= corr;
            qdotk_max_buf[cell] = qdotk_max_new;
        }
    }

    // phase 3: per coarse chunk cell, scatter exp(q.k - max)*w into alpha_sum and
    // exp(...)*w*v into y_acc.
    template <int THREADS_PER_BLOCK, typename STORAGE_T>
    __global__ __launch_bounds__(THREADS_PER_BLOCK) void s2_attn_fwd_ring_upsample_acc_k(
        int nheads, int nchan_in, int nchan_out, int nlat_halo, int nlon_kx, int nlon_out_global, int pscale_out,
        int lon_lo_kx, int lat_halo_start, int nlat_out, int nlon_out, const STORAGE_T *__restrict__ kx,
        const STORAGE_T *__restrict__ vx, const STORAGE_T *__restrict__ qy, const int32_t *__restrict__ seg,
        const int32_t *__restrict__ seg_off, const float *__restrict__ ring_weights,
        const float *__restrict__ qdotk_max_buf, float *__restrict__ alpha_sum_buf, float *__restrict__ y_acc)
    {
        extern __shared__ float shext[];
        float *sh_k = shext + threadIdx.y * (nchan_in + nchan_out);
        float *sh_v = sh_k + nchan_in;

        // one block row per (batch, head) pair
        const int bh = blockIdx.y;
        const int batch = bh / nheads;
        const int head = bh - (batch * nheads);
        const int64_t ldi = int64_t(nheads) * nchan_in;
        const int64_t ldo = int64_t(nheads) * nchan_out;

        const int wid = blockIdx.x * blockDim.y + threadIdx.y;
        if (wid >= nlat_halo * nlon_kx) { return; }

        const int tidx = threadIdx.x;
        const int hi = wid / nlon_kx; // LOCAL halo row
        const int wi_local = wid - hi * nlon_kx;
        const int wi_global = lon_lo_kx + wi_local;

        const int seg_beg = seg_off[hi];
        const int seg_end = seg_off[hi + 1];
        if (seg_beg == seg_end) { return; } // empty row (e.g. pole padding)

        kx += int64_t(batch) * nlat_halo * nlon_kx * ldi + (int64_t(hi) * nlon_kx + wi_local) * ldi
            + int64_t(head) * nchan_in;
        vx += int64_t(batch) * nlat_halo * nlon_kx * ldo + (int64_t(hi) * nlon_kx + wi_local) * ldo
            + int64_t(head) * nchan_out;
        qy += int64_t(batch) * nlat_out * nlon_out * ldi + int64_t(head) * nchan_in;
        qdotk_max_buf += int64_t(bh) * nlat_out * nlon_out;
        alpha_sum_buf += int64_t(bh) * nlat_out * nlon_out;
        y_acc += int64_t(batch) * nlat_out * nlon_out * ldo + int64_t(head) * nchan_out;

        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) { sh_k[chan] = vload(kx, chan); }
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { sh_v[chan] = vload(vx, chan); }

        // ring_weights is indexed by the GLOBAL input latitude
        const float qw = ring_weights[lat_halo_start + hi];

        for (int sg = seg_beg; sg < seg_end; sg++) {

            const int ho = seg[3 * sg + 0]; // LOCAL output row
            const int seg_lo = seg[3 * sg + 1];
            const int seg_len = seg[3 * sg + 2];

            // the part of the arc on this rank's output longitudes
            int2 piece[2];
            const int npiece = clip_arc(ring_up_scatter_wo(seg_lo, wi_global, pscale_out, nlon_out_global), seg_len,
                                        nlon_out_global, 0, nlon_out, piece);

            for (int pc = 0; pc < npiece; pc++) {

                int wo = piece[pc].x;

                for (int j = 0; j < piece[pc].y; j++) {
                    const int64_t cell = int64_t(ho) * nlon_out + wo;
                    const STORAGE_T *_qy = qy + cell * ldi;

                    float qd = 0.f;
                    for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) { qd += sh_k[chan] * vload(_qy, chan); }
                    qd = __warp_sum(qd);

                    const float alpha = expf(qd - qdotk_max_buf[cell]) * qw;
                    if (tidx == 0) { atomicAdd(&alpha_sum_buf[cell], alpha); }
                    float *_y_acc = y_acc + cell * ldo;
                    for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
                        atomicAdd(&_y_acc[chan], alpha * sh_v[chan]);
                    }

                    // pieces do not wrap: clip_arc split the arc at the seam
                    wo++;
                }
            }
        }
    }

    // host launcher for one ring step (allocates the temporary max buffer, runs
    // the three phases). STORAGE_T deduces from the activation pointers.
    template <typename STORAGE_T>
    static void launch_attn_fwd_ring_upsample_step(int batch_size, int nheads, int nchans_in, int nchans_out,
                                                   int nlat_halo, int nlon_kx, int nlon_out_global, int pscale_out,
                                                   int lon_lo_kx, int lat_halo_start, int nlat_out, int nlon_out,
                                                   STORAGE_T *_kxp, STORAGE_T *_vxp, STORAGE_T *_qyp, int32_t *_seg,
                                                   int32_t *_seg_off, float *_quad_weights, float *_y_acc,
                                                   float *_alpha_sum, float *_qdotk_max, cudaStream_t stream)
    {
        // temporary buffer holding the updated max; starts as a copy of the
        // running max so untouched cells stay unchanged (rescale is a no-op there)
        torch::Tensor qdotk_max_curr
            = torch::from_blob(_qdotk_max, {int64_t(batch_size) * nheads * nlat_out * nlon_out},
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                  .clone();
        float *_qdotk_max_curr = reinterpret_cast<float *>(qdotk_max_curr.data_ptr());

        dim3 block(WARP_SIZE, THREADS / WARP_SIZE);
        // one block row per (batch, head) pair
        dim3 grid_in(DIV_UP(nlat_halo * nlon_kx, block.y), batch_size * nheads);
        dim3 grid_out(DIV_UP(nlat_out * nlon_out, block.y), batch_size * nheads);

        const size_t sh_max = sizeof(float) * nchans_in * block.y;
        const size_t sh_acc = sizeof(float) * (nchans_in + nchans_out) * block.y;

        launch_dyn_shmem(&s2_attn_fwd_ring_upsample_max_k<THREADS, STORAGE_T>, grid_in, block, sh_max, stream, nheads,
                         nchans_in, nlat_halo, nlon_kx, nlon_out_global, pscale_out, lon_lo_kx, nlat_out, nlon_out,
                         _kxp, _qyp, _seg, _seg_off, _qdotk_max_curr);
        CHECK_ERROR("s2_attn_fwd_ring_upsample_max_k");

        // also commits qdotk_max_curr into the persistent _qdotk_max buffer
        s2_attn_fwd_ring_upsample_rescale_k<THREADS><<<grid_out, block, 0, stream>>>(
            nheads, nchans_out, nlat_out, nlon_out, _qdotk_max_curr, _qdotk_max, _alpha_sum, _y_acc);
        CHECK_ERROR("s2_attn_fwd_ring_upsample_rescale_k");

        launch_dyn_shmem(&s2_attn_fwd_ring_upsample_acc_k<THREADS, STORAGE_T>, grid_in, block, sh_acc, stream, nheads,
                         nchans_in, nchans_out, nlat_halo, nlon_kx, nlon_out_global, pscale_out, lon_lo_kx,
                         lat_halo_start, nlat_out, nlon_out, _kxp, _vxp, _qyp, _seg, _seg_off, _quad_weights,
                         _qdotk_max, _alpha_sum, _y_acc);
        CHECK_ERROR("s2_attn_fwd_ring_upsample_acc_k");
    }

    void s2_attention_fwd_ring_step_upsample_cuda(at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor y_acc,
                                                  at::Tensor alpha_sum_buf, at::Tensor qdotk_max_buf,
                                                  at::Tensor ring_weights, at::Tensor psi_seg, at::Tensor psi_seg_off,
                                                  int64_t num_heads, int64_t nlon_in, int64_t nlon_out_global,
                                                  int64_t lon_lo_kx, int64_t lat_halo_start, int64_t nlat_out,
                                                  int64_t nlon_out)
    {
        CHECK_CUDA_INPUT_TENSOR(kx);
        CHECK_CUDA_INPUT_TENSOR(vx);
        CHECK_CUDA_INPUT_TENSOR(qy);
        CHECK_CUDA_TENSOR(y_acc);
        CHECK_CUDA_TENSOR(alpha_sum_buf);
        CHECK_CUDA_TENSOR(qdotk_max_buf);
        CHECK_CUDA_TENSOR(ring_weights);

        // run on the inputs' device: without this, the current stream, the scratch
        // allocations and the per-device queries (ensure_dyn_shmem, getPtxver) would all
        // resolve to whichever CUDA device happens to be current
        const at::cuda::OptionalCUDAGuard device_guard(kx.device());
        // devices, shapes, index and weight dtypes, dense layouts; the arc rows are the
        // chunk's halo-padded input latitudes
        TORCH_CHECK(kx.dim() == 4, "kx must be 4-D (B, nlat_halo, nlon_chunk, channels), got ", kx.dim(), " dims");
        check_ring_step_inputs(kx, vx, qy, ring_weights, psi_seg, psi_seg_off, num_heads, nlat_out, nlon_out, kx.size(1));
        check_state_buffer(y_acc, kx, "y_acc", {kx.size(0), nlat_out, nlon_out, vx.size(3)});
        check_state_buffer(alpha_sum_buf, kx, "alpha_sum_buf", {kx.size(0), num_heads, nlat_out, nlon_out});
        check_state_buffer(qdotk_max_buf, kx, "qdotk_max_buf", {kx.size(0), num_heads, nlat_out, nlon_out});

        // NHWC ABI, as in the serial kernels: kx, vx, qy are physically
        // (B, H, W, num_heads * nchan) and contiguous, with the heads packed along the
        // channel dim; y_acc is packed the same way and alpha_sum / qdotk_max are
        // (B, num_heads, H, W).
        const int batch_size = kx.size(0);
        const int nlat_halo = kx.size(1);
        const int nlon_kx = kx.size(2);
        // per-head channel counts; the packed extent is num_heads times these
        const size_t nchans_in = qy.size(3) / num_heads;
        const size_t nchans_out = vx.size(3) / num_heads;

        // the p-shift ratio, derived as the serial ops derive it but from the GLOBAL
        // output width: nlon_out is this rank's
        TORCH_CHECK(nlon_in > 0 && nlon_out_global % nlon_in == 0, "nlon_out_global (", nlon_out_global,
                    ") must be an integer multiple of nlon_in (", nlon_in, ")");
        const int64_t pscale_out = nlon_out_global / nlon_in;

        auto stream = at::cuda::getCurrentCUDAStream().stream();

        int32_t *_seg = reinterpret_cast<int32_t *>(psi_seg.data_ptr());
        int32_t *_seg_off = reinterpret_cast<int32_t *>(psi_seg_off.data_ptr());
        float *_quad_weights = reinterpret_cast<float *>(ring_weights.data_ptr());
        float *_y_acc = reinterpret_cast<float *>(y_acc.data_ptr());
        float *_alpha_sum = reinterpret_cast<float *>(alpha_sum_buf.data_ptr());
        float *_qdotk_max = reinterpret_cast<float *>(qdotk_max_buf.data_ptr());

        // ATen dispatch over the input dtype. Tier B: native storage — kernels
        // widen to fp32 at load; the state buffers stay fp32.
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, qy.scalar_type(), "s2_attention_fwd_ring_step_upsample_cuda", [&] {
                using storage_t = scalar_t;

                torch::Tensor kxP = kx;
                torch::Tensor vxP = vx;
                torch::Tensor qyP = qy;

                storage_t *_kxp = reinterpret_cast<storage_t *>(kxP.data_ptr());
                storage_t *_vxp = reinterpret_cast<storage_t *>(vxP.data_ptr());
                storage_t *_qyp = reinterpret_cast<storage_t *>(qyP.data_ptr());

                launch_attn_fwd_ring_upsample_step(
                    batch_size, static_cast<int>(num_heads), static_cast<int>(nchans_in), static_cast<int>(nchans_out),
                    nlat_halo, nlon_kx, static_cast<int>(nlon_out_global), static_cast<int>(pscale_out),
                    static_cast<int>(lon_lo_kx), static_cast<int>(lat_halo_start), static_cast<int>(nlat_out),
                    static_cast<int>(nlon_out), _kxp, _vxp, _qyp, _seg, _seg_off, _quad_weights, _y_acc, _alpha_sum,
                    _qdotk_max, stream);
            });

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    TORCH_LIBRARY_IMPL(attention_kernels, CUDA, m)
    {
        m.impl("forward_ring_step_upsample", &s2_attention_fwd_ring_step_upsample_cuda);
    }

} // namespace attention_kernels
