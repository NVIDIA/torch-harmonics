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

#include "attention_cuda.cuh"
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/CUDAUtils.h>

#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

#include <cuda/barrier>

#include "cudamacro.h"
#include "attention_cuda_utils.cuh"

#define THREADS (64)

#define MAX_LOCAL_ARR_LEN (16)

// Ring-step variant of the forward attention kernel. Used by
// DistributedNeighborhoodAttentionS2: K/V are sharded along longitude across
// an azimuth process group; each call processes one rotating chunk and
// accumulates softmax state into externally-allocated buffers (y_acc,
// alpha_sum_buf, qdotk_max_buf). col_idx must have wi pre-shifted by
// pscale * lon_lo_out (see _build_local_psi in distributed_attention.py).

namespace attention_kernels {

template<int BDIM_X,
         typename FLOATV_T>
__global__
__launch_bounds__(BDIM_X)
void s2_attn_fwd_ring_generic_k(
    const __grid_constant__ attn_params_t p,
    const FLOATV_T *__restrict__ kx,           // [batch][nlat_halo][nlon_kx][nchan_in]
    const FLOATV_T *__restrict__ vx,           // [batch][nlat_halo][nlon_kx][nchan_out]
    const FLOATV_T *__restrict__ qy,           // [batch][nlat_out][nlon_out][nchan_in]
    const int32_t  *__restrict__ row_idx,
    const int64_t  *__restrict__ row_off,
    const int64_t  *__restrict__ col_idx,      // wi already shifted by pscale * lon_lo_out
    const float    *__restrict__ quad_weights, // [nlat_in_global]
    FLOATV_T *__restrict__ y_acc,              // [batch][nlat_out][nlon_out][nchan_out] (in/out)
    float    *__restrict__ alpha_sum_buf,      // [batch][nlat_out][nlon_out] (in/out)
    float    *__restrict__ qdotk_max_buf       // [batch][nlat_out][nlon_out] (in/out)
) {
    const int &nchan_in       = p.nchan_in;
    const int &nchan_out      = p.nchan_out;
    const int &nlat_halo      = p.nlat_halo;
    const int &nlon_kx        = p.nlon_kx;
    const int &nlon_in        = p.nlon_in;
    const int &pscale         = p.pscale;
    const int &lon_lo_kx      = p.lon_lo_kx;
    const int &lat_halo_start = p.lat_halo_start;
    const int &nlat_out       = p.nlat_out;
    const int &nlon_out       = p.nlon_out;

    extern __shared__ __align__(sizeof(float4)) float shext[];
    FLOATV_T *shy = reinterpret_cast<FLOATV_T *>(shext) + threadIdx.y * nchan_out;

    const int batch = blockIdx.y;
    const int wid   = blockIdx.x * blockDim.y + threadIdx.y;
    if (wid >= nlat_out * nlon_out) return;

    const int tidx  = threadIdx.x;
    const int h     = wid / nlon_out;
    const int wo    = wid - (h * nlon_out);   // LOCAL wo
    const int ho    = row_idx[h];

    kx += int64_t(batch) * nlat_halo * nlon_kx * nchan_in;
    vx += int64_t(batch) * nlat_halo * nlon_kx * nchan_out;
    qy += int64_t(batch) * nlat_out  * nlon_out * nchan_in
        + int64_t(ho)    * nlon_out  * nchan_in
        + int64_t(wo)    * nchan_in;

    const int64_t out_flat = int64_t(batch) * nlat_out * nlon_out
                           + int64_t(ho)    * nlon_out + wo;
    y_acc         += out_flat * nchan_out;
    alpha_sum_buf += out_flat;
    qdotk_max_buf += out_flat;

    // Load current state from buffers
    float alpha_sum = alpha_sum_buf[0];
    float qdotk_max = qdotk_max_buf[0];
    for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
        shy[chan] = y_acc[chan];
    }

    // `pscale` is the GLOBAL ratio nlon_in / nlon_out_global, passed by the caller.
    // Computing it here as `nlon_in / nlon_out` would be wrong because the kernel's
    // `nlon_out` is the LOCAL output width (nlon_out_local), not the global one.

    const int64_t rbeg = row_off[ho];
    const int64_t rend = row_off[ho + 1];
    col_idx += rbeg;
    const int rlen = rend - rbeg;

    for (int off = 0; off < rlen; off++) {
        const int64_t col = col_idx[off];

        // col_idx stores hi_global * nlon_in + wi_shifted
        // where wi_shifted = (wi_canonical + pscale * lon_lo_out) % nlon_in (baked in at Python __init__)
        const int hi_global = col / nlon_in;
        const int wi        = col - (hi_global * nlon_in);
        // wip = (wi + pscale * wo_local) % nlon_in
        //     = (wi_canonical + pscale * (lon_lo_out + wo_local)) % nlon_in
        const int wi_wo     = wi + pscale * wo;
        const int wip       = wi_wo - (wi_wo / nlon_in) * nlon_in;

        // Skip neighbors not in current kx chunk
        if (wip < lon_lo_kx || wip >= lon_lo_kx + nlon_kx) continue;

        const int hi_local  = hi_global - lat_halo_start;
        // Skip neighbors outside the halo-padded lat range (needed for distributed case)
        if (hi_local < 0 || hi_local >= nlat_halo) continue;
        const int wip_local = wip - lon_lo_kx;

        const FLOATV_T *_kx = kx + int64_t(hi_local) * nlon_kx * nchan_in  + int64_t(wip_local) * nchan_in;
        const FLOATV_T *_vx = vx + int64_t(hi_local) * nlon_kx * nchan_out + int64_t(wip_local) * nchan_out;

        FLOATV_T qdotkv = __vset<FLOATV_T>(0.f);
        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
            qdotkv = __vadd(qdotkv, __vmul(qy[chan], _kx[chan]));
        }
        float qdotk = __warp_sum(__vred(qdotkv));

        const float qdotk_max_tmp = max(qdotk_max, qdotk);
        const float alpha          = expf(qdotk - qdotk_max_tmp) * quad_weights[hi_global];
        const float exp_save       = expf(qdotk_max - qdotk_max_tmp);

        alpha_sum = alpha + alpha_sum * exp_save;
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
            shy[chan] = __vadd(__vscale(exp_save, shy[chan]),
                               __vscale(alpha,    _vx[chan]));
        }
        qdotk_max = qdotk_max_tmp;
    }

    // Store updated state back to buffers
    if (!tidx) {
        alpha_sum_buf[0] = alpha_sum;
        qdotk_max_buf[0] = qdotk_max;
    }
    for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
        y_acc[chan] = shy[chan];
    }
}

template<typename FLOATV_T>
void launch_gen_attn_ring_fwd(attn_params_t params,
                              int64_t batch_size,
                              FLOATV_T *_kxp,
                              FLOATV_T *_vxp,
                              FLOATV_T *_qyp,
                              int32_t *_row_idx,
                              int64_t *_row_off,
                              int64_t *_col_idx,
                              float *_quad_weights,
                              FLOATV_T *_y_acc,
                              float *_alpha_sum,
                              float *_qdotk_max,
                              cudaStream_t stream) {

    const int nlat_out = params.nlat_out;
    const int nlon_out = params.nlon_out;
    const int nchans_out = params.nchan_out;

    dim3 block(WARP_SIZE, THREADS/WARP_SIZE);
    dim3 grid(DIV_UP(nlat_out*nlon_out, block.y), batch_size);

    size_t shsize = sizeof(FLOATV_T)*nchans_out * block.y;

    auto kern = &s2_attn_fwd_ring_generic_k<THREADS, FLOATV_T>;
    ensure_dyn_shmem(reinterpret_cast<const void*>(kern), shsize); 

    kern<<<grid, block, shsize, stream>>>(params, _kxp, _vxp, _qyp,
                                          _row_idx, _row_off, _col_idx, 
                                          _quad_weights, _y_acc,
                                          _alpha_sum, _qdotk_max);

    CHECK_ERROR("s2_attn_fwd_ring_generic_k");

    return;
}

template<int BDIM_X,
         int BDIM_Y,
         int CHIN_AS_OUT, // 1 iif "BDIM_X*(NLOC-1) <= nchan_in <= BDIM_X*NLOC" else 0
         int NLOC,        // smallest int such that BDIM_X*NLOC >= nchan_out
         typename FLOATV_T>
__global__
__launch_bounds__(BDIM_X*BDIM_Y)
void s2_attn_fwd_ring_special_k(const __grid_constant__ attn_params_t p,
                                const int shcol_len_max,
                                const int nlat_max,
                                const FLOATV_T *__restrict__ kx,           // [batch][nlat_halo][nlon_kx][nchan_in]
                                const FLOATV_T *__restrict__ vx,           // [batch][nlat_halo][nlon_kx][nchan_out]
                                const FLOATV_T *__restrict__ qy,           // [batch][nlat_out][nlon_out][nchan_in]
                                const int32_t  *__restrict__ row_idx,
                                const int64_t  *__restrict__ row_off,
                                const int64_t  *__restrict__ col_idx,      // wi already shifted by pscale * lon_lo_out
                                const float    *__restrict__ quad_weights, // [nlat_in_global]
                                      FLOATV_T *__restrict__ y_acc,              // [batch][nlat_out][nlon_out][nchan_out] (in/out)
                                      float    *__restrict__ alpha_sum_buf,      // [batch][nlat_out][nlon_out] (in/out)
                                      float    *__restrict__ qdotk_max_buf) {    // [batch][nlat_out][nlon_out] (in/out)

    static_assert(0 == (BDIM_X & (BDIM_X-1)));
    static_assert(0 == (BDIM_Y & (BDIM_Y-1)));
    static_assert((BDIM_X == WARP_SIZE && BDIM_Y  > 1) ||
                  (BDIM_X  > WARP_SIZE && BDIM_Y == 1)) ;

    const int tidx  = threadIdx.x;
    const int tidy  = threadIdx.y;

    const int ctaid = blockIdx.x*blockDim.y + threadIdx.y;
    const int batch = blockIdx.y;

    const int &nchan_in       = p.nchan_in;
    const int &nchan_out      = p.nchan_out;
    const int &nlat_halo      = p.nlat_halo;
    const int &nlon_kx        = p.nlon_kx;
    const int &nlon_in        = p.nlon_in;
    const int &pscale         = p.pscale;
    const int &lon_lo_kx      = p.lon_lo_kx;
    const int &lat_halo_start = p.lat_halo_start;
    const int &nlat_out       = p.nlat_out;
    const int &nlon_out       = p.nlon_out;

    if (ctaid >= nlat_max*nlon_out) {
        return;
    }

    extern __shared__ __align__(sizeof(float4)) float shext[];

    using FLOATV_PTR_T = const FLOATV_T *;

    // chunked into 4 arrays: FLOATV_T shq[BDIM_Y][nchan_in]
    //                        FLOATV_T *shkx_ptr[BDIM_Y][shcol_len_max]
    //                        FLOATV_T *shvx_ptr[BDIM_Y][shcol_len_max]
    //                        float     shweight[BDIM_Y][shcol_len_max]
    FLOATV_T     *base_fltv     = NULL;
    FLOATV_PTR_T *base_fltv_ptr = NULL;
    float        *base_flt      = NULL;

    if constexpr(sizeof(FLOATV_T) > sizeof(FLOATV_PTR_T)) {
        base_fltv     = reinterpret_cast<FLOATV_T     *>(shext);
        base_fltv_ptr = reinterpret_cast<FLOATV_PTR_T *>(base_fltv + BDIM_Y*nchan_in);
        base_flt      = reinterpret_cast<float        *>(base_fltv_ptr + BDIM_Y*2*shcol_len_max);
    } else {
        base_fltv_ptr = reinterpret_cast<FLOATV_PTR_T *>(shext);
        base_fltv     = reinterpret_cast<FLOATV_T     *>(base_fltv_ptr  + BDIM_Y*2*shcol_len_max);
        base_flt      = reinterpret_cast<float        *>(base_fltv + BDIM_Y*nchan_in);
    }

    FLOATV_T          *shq = base_fltv                            + tidy*nchan_in; 
    FLOATV_PTR_T *shkx_ptr = base_fltv_ptr                        + tidy*shcol_len_max;
    FLOATV_PTR_T *shvx_ptr = base_fltv_ptr + BDIM_Y*shcol_len_max + tidy*shcol_len_max;
    float        *shweight = base_flt                             + tidy*shcol_len_max;

    shq += tidx;

    const int h  = ctaid / nlon_out;
    const int wo = ctaid - (h*nlon_out);   // LOCAL wo
    const int ho = row_idx[h];

    kx += int64_t(batch)*nlat_halo*nlon_kx*nchan_in;
    vx += int64_t(batch)*nlat_halo*nlon_kx*nchan_out;

    qy += int64_t(batch)*nlat_out*nlon_out*nchan_in + int64_t(ho)*nlon_out*nchan_in + int64_t(wo)*nchan_in + tidx;

    const int64_t out_flat = int64_t(batch)*nlat_out*nlon_out + int64_t(ho)*nlon_out + wo;

    y_acc         += out_flat*nchan_out + tidx;
    alpha_sum_buf += out_flat;
    qdotk_max_buf += out_flat;

    FLOATV_T locy[NLOC];

    // Load current state from buffers
    float alpha_sum = alpha_sum_buf[0];
    float qdotk_max = qdotk_max_buf[0];

    strided_op<BDIM_X,               NLOC    >(nchan_out, [&](int i) {       locy[i] = y_acc[i*BDIM_X]; });
    strided_op<BDIM_X, CHIN_AS_OUT ? NLOC : 0>(nchan_in,  [&](int i) { shq[i*BDIM_X] =    qy[i*BDIM_X]; });

    // `pscale` is the GLOBAL ratio nlon_in / nlon_out_global, passed by the caller.
    // Computing it here as `nlon_in / nlon_out` would be wrong because the kernel's
    // `nlon_out` is the LOCAL output width (nlon_out_local), not the global one.

    const int64_t rbeg = row_off[ho];
    const int64_t rend = row_off[ho + 1];
    const int     rlen = rend - rbeg;
    col_idx += rbeg;

    int n = rlen;
    int n_active = 0;

    for (int i = 0; i < n; i += BDIM_X) {

        const FLOATV_T *kx_ptr = NULL;
        const FLOATV_T *vx_ptr = NULL;
        float weight = 0;

        if (i+tidx < n) {
            const int64_t col = col_idx[i+tidx];

            const int hi_global = col / nlon_in;
            const int wi        = col - (hi_global * nlon_in);
            const int wi_wo     = wi + pscale * wo;

            const int wip = wi_wo - (wi_wo / nlon_in) * nlon_in;

            if (wip >= lon_lo_kx && wip < lon_lo_kx + nlon_kx) {

                const int hi_local  = hi_global - lat_halo_start;

                if (hi_local >= 0 && hi_local < nlat_halo) {

                    const int wip_local = wip - lon_lo_kx;
                    kx_ptr = kx + int64_t(hi_local)*nlon_kx*nchan_in  + int64_t(wip_local)*nchan_in;
                    vx_ptr = vx + int64_t(hi_local)*nlon_kx*nchan_out + int64_t(wip_local)*nchan_out;
                    weight = quad_weights[hi_global];
                }
            }
        }

        int toff;
        int ntot = __compact<BDIM_X, BDIM_Y>(kx_ptr != NULL, &toff);
        if (kx_ptr != NULL) {
            shkx_ptr[n_active + toff] = kx_ptr;
            shvx_ptr[n_active + toff] = vx_ptr;
            shweight[n_active + toff] = weight;
        }
        n_active += ntot;
    }
    __gsync<BDIM_X>();

    n = n_active;

    if (n == 0) {
        return;
    }
    
    for (int i = 0; i < n; i++) {

        const FLOATV_T *_kx = shkx_ptr[i] + tidx;
        const FLOATV_T *_vx = shvx_ptr[i] + tidx;

        FLOATV_T qdotkv = __vset<FLOATV_T>(0.f);
        strided_op<BDIM_X, CHIN_AS_OUT ? NLOC : 0>(nchan_in,  [&](int i) { qdotkv = __vadd(qdotkv, __vmul(shq[i*BDIM_X], _kx[i*BDIM_X])); });

        float qdotk = __vred(qdotkv);

        __group_sum<BDIM_X, BDIM_Y>(qdotk);

        const float qdotk_max_tmp = max(qdotk_max, qdotk);
        const float alpha         = expf(qdotk - qdotk_max_tmp) * shweight[i];
        const float exp_save      = expf(qdotk_max - qdotk_max_tmp);

        alpha_sum = alpha + alpha_sum * exp_save;

        strided_op<BDIM_X, NLOC>(nchan_out, [&](int i) { locy[i] = __vadd(__vscale(exp_save, locy[i]), __vscale(   alpha, _vx[i*BDIM_X])); });
        qdotk_max = qdotk_max_tmp;
    }

    // Store updated state back to buffers, no need for BDIM_X benign race conditions...
    if (!tidx) {
        alpha_sum_buf[0] = alpha_sum;
        qdotk_max_buf[0] = qdotk_max;
    }

    strided_op<BDIM_X, NLOC>(nchan_out, [&](int i) { y_acc[i*BDIM_X] = locy[i]; });

    return;
}

#include "attention_cuda_fwd_ring_lr.cuh"

template<int BDIM_X,
         int CUR_LOC_SIZE,
         int MAX_LOC_SIZE, // max size of FLOATV_T[] local array
         typename FLOATV_T>
void launch_spc_attn_ring_fwd(attn_params_t params,
                              int nloc,
                              int64_t batch_size,
                              FLOATV_T *_kxp,
                              FLOATV_T *_vxp,
                              FLOATV_T *_qyp,
                              int32_t *_row_idx,
                              int64_t *_row_off,
                              int64_t *_col_idx,
                              float *_quad_weights,
                              FLOATV_T *_y_acc,
                              float *_alpha_sum,
                              float *_qdotk_max,
                              cudaStream_t stream) {

    static_assert(0 == (BDIM_X & (BDIM_X-1)));
    static_assert(std::is_same<FLOATV_T, float>::value ||
                  std::is_same<FLOATV_T, float4>::value);

    if (CUR_LOC_SIZE == nloc) {

        const int nlat_out = params.nlat_out;
        const int nlon_out = params.nlon_out;
        const int nchans_in  = params.nchan_in;

        int64_t n_long_rows;
        int64_t max_row_len;
        int64_t mid_row_len;

        // splits the rows in "long" and "short" rows; long rows have
        // a length >= max(SPLIT_LONG_ROW_MIN_LEN(1024), len(row_0))
        // (since the rows are sorted in decreasing order, row_0 is the
        // longest row); short rows are the remaining rows.
        // If there are long rows, they are processed with a separate 
        // kernels using multiple blocks per row, in order to mitigate 
        // the imbalance causing long temporal tails.
        split_csr_rows(SPLIT_ROW_LENGTH_THRES, SPLIT_LONG_ROW_MIN_LEN,
                       nlat_out, _row_idx, _row_off, &n_long_rows, &max_row_len, &mid_row_len);

        if (n_long_rows > 0) {
            // processes the "long rows", from _row_idx[0] to _row_idx[n_long_rows-1]
            spc_attn_ring_fwd_long_rows<BDIM_X, CUR_LOC_SIZE>(params, n_long_rows, max_row_len,
                                               batch_size, _kxp, _vxp, _qyp, _row_idx,
                                               _row_off, _col_idx, _quad_weights, _y_acc, _alpha_sum,
                                               _qdotk_max, stream);
        }

        // nloc determines the size of local arrays used to store
        // y vectors, of length nchans_out;
        // if nchans_in is >= BDIM_X*(nloc-1) and <= BDIM_X*nloc
        // then we can use the same compile-time known loops used
        // for output channels, with the execpetion of testing 
        // whether to execute the last iteration based on "nchans_in"
        // rather than on "nchans_out"; in this way as long as the
        // difference between the number of input and output channels
        // is <= BDIM_X we can use the faster path 
        const bool chin_as_out = (nchans_in >= BDIM_X*(CUR_LOC_SIZE-1) && 
                                  nchans_in <= BDIM_X* CUR_LOC_SIZE  );

        constexpr int BDIM_Y = (BDIM_X <= WARP_SIZE) ? THREADS/BDIM_X : 1;

        dim3 block(BDIM_X, BDIM_Y);
        dim3 grid(DIV_UP((nlat_out-n_long_rows)*nlon_out, block.y), batch_size);

        size_t shsize = (sizeof(FLOATV_T)*nchans_in + sizeof(FLOATV_T *)*mid_row_len*2 + sizeof(float)*mid_row_len) * block.y;

        if (chin_as_out) {
            auto kern = &s2_attn_fwd_ring_special_k<BDIM_X, BDIM_Y, 1, CUR_LOC_SIZE, FLOATV_T>;
            ensure_dyn_shmem(reinterpret_cast<const void*>(kern), shsize);

            kern<<<grid, block, shsize, stream>>>(params, mid_row_len, nlat_out-n_long_rows,
                                                  _kxp, _vxp, _qyp, _row_idx + n_long_rows,
                                                  _row_off, _col_idx, _quad_weights, _y_acc,
                                                  _alpha_sum, _qdotk_max);
        } else {
            auto kern = &s2_attn_fwd_ring_special_k<BDIM_X, BDIM_Y, 0, CUR_LOC_SIZE, FLOATV_T>;
            ensure_dyn_shmem(reinterpret_cast<const void*>(kern), shsize);

            kern<<<grid, block, shsize, stream>>>(params, mid_row_len, nlat_out-n_long_rows,
                                                  _kxp, _vxp, _qyp, _row_idx + n_long_rows, 
                                                  _row_off, _col_idx, _quad_weights, _y_acc,
                                                  _alpha_sum, _qdotk_max);
        }
        CHECK_ERROR("s2_attn_fwd_ring_special_k");

        return;
    }
    if constexpr(CUR_LOC_SIZE < MAX_LOC_SIZE) {
        launch_spc_attn_ring_fwd<BDIM_X,
                                 CUR_LOC_SIZE+1,
                                 MAX_LOC_SIZE>(params, nloc, batch_size,
                                               _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx,
                                               _quad_weights, _y_acc, _alpha_sum, _qdotk_max,
                                               stream);
    }
}

static void s2_attn_fwd_ring_step_dispatch(int64_t batch_size,
                                           int64_t nchans_in,
                                           int64_t nchans_out,
                                           int64_t nlon_in,
                                           int64_t pscale,
                                           int64_t nlat_halo,
                                           int64_t nlon_kx,
                                           int64_t lon_lo_kx,
                                           int64_t lat_halo_start,
                                           int64_t nlat_out,
                                           int64_t nlon_out,
                                           at::Tensor kxP,
                                           at::Tensor vxP,
                                           at::Tensor qyP,
                                           at::Tensor row_idx,
                                           at::Tensor row_off,
                                           at::Tensor col_idx,
                                           at::Tensor quad_weights,
                                           at::Tensor y_acc,
                                           at::Tensor alpha_sum_buf,
                                           at::Tensor qdotk_max_buf) {

    static_assert(0 == (MAX_LOCAL_ARR_LEN & (MAX_LOCAL_ARR_LEN-1)));

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // smallest power of two "bdimx" (>=32) s.t. bdimx*MAX_LOCAL_ARR_LEN >= nchans_out
    int bdimx;
    bdimx = DIV_UP(nchans_out, MAX_LOCAL_ARR_LEN);
    bdimx = max(bdimx, WARP_SIZE);
    bdimx = next_pow2(bdimx);

    float *_kxp          = reinterpret_cast<float *>(kxP.data_ptr());
    float *_vxp          = reinterpret_cast<float *>(vxP.data_ptr());
    float *_qyp          = reinterpret_cast<float *>(qyP.data_ptr());
    int32_t *_row_idx    = reinterpret_cast<int32_t *>(row_idx.data_ptr());
    int64_t *_row_off    = reinterpret_cast<int64_t *>(row_off.data_ptr());
    int64_t *_col_idx    = reinterpret_cast<int64_t *>(col_idx.data_ptr());
    float *_quad_weights = reinterpret_cast<float *>(quad_weights.data_ptr());
    float *_y_acc        = reinterpret_cast<float *>(y_acc.data_ptr());
    float *_alpha_sum    = reinterpret_cast<float *>(alpha_sum_buf.data_ptr());
    float *_qdotk_max    = reinterpret_cast<float *>(qdotk_max_buf.data_ptr());

    constexpr int VEC_SIZE = sizeof(float4) / sizeof(float);

    attn_params_t params;

    params.nchan_in       = nchans_in;
    params.nchan_out      = nchans_out;
    params.nlat_halo      = nlat_halo;
    params.nlon_kx        = nlon_kx;
    params.nlon_in        = nlon_in;
    params.pscale         = pscale;
    params.lon_lo_kx      = lon_lo_kx;
    params.lat_halo_start = lat_halo_start;
    params.nlat_out       = nlat_out;
    params.nlon_out       = nlon_out;

    if (!is_aligned<sizeof(float4)>(_kxp) ||
        !is_aligned<sizeof(float4)>(_vxp) ||
        !is_aligned<sizeof(float4)>(_qyp) ||
        !is_aligned<sizeof(float4)>(_y_acc) ||
        (nchans_in  % VEC_SIZE) != 0 ||
        (nchans_out % VEC_SIZE) != 0) {

        const int nloc = DIV_UP(nchans_out, bdimx);

        // to avoid the compilation of unused template instances;
        // we use a block size BDIM_X that is the smallest power of 2
        // such that BDIM_X*MAX_LOCAL_ARR_LEN >= nchans_out, so
        // BDIM_X > 32 are used only for:
        //
        //  (BDIM_X-1)*MAX_LOCAL_ARR_LEN < nchans_out <= BDIM_X*MAX_LOCAL_ARR_LEN
        constexpr int MIN_LOC_ARR_LEN = MAX_LOCAL_ARR_LEN/2+1;

        switch(bdimx) {
            case   32: launch_spc_attn_ring_fwd<  32,               1, MAX_LOCAL_ARR_LEN>(params, nloc, batch_size, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _y_acc, _alpha_sum, _qdotk_max, stream); break;
            case   64: launch_spc_attn_ring_fwd<  64, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(params, nloc, batch_size, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _y_acc, _alpha_sum, _qdotk_max, stream); break;
            case  128: launch_spc_attn_ring_fwd< 128, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(params, nloc, batch_size, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _y_acc, _alpha_sum, _qdotk_max, stream); break;
            case  256: launch_spc_attn_ring_fwd< 256, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(params, nloc, batch_size, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _y_acc, _alpha_sum, _qdotk_max, stream); break;
            case  512: launch_spc_attn_ring_fwd< 512, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(params, nloc, batch_size, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _y_acc, _alpha_sum, _qdotk_max, stream); break;
            case 1024: launch_spc_attn_ring_fwd<1024, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(params, nloc, batch_size, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _y_acc, _alpha_sum, _qdotk_max, stream); break;
            default:   launch_gen_attn_ring_fwd                                          (params,       batch_size, _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _y_acc, _alpha_sum, _qdotk_max, stream); break;
        }

    } else {

        float4 *_kxp4  = reinterpret_cast<float4 *>(_kxp);
        float4 *_vxp4  = reinterpret_cast<float4 *>(_vxp);
        float4 *_qyp4  = reinterpret_cast<float4 *>(_qyp);
        float4 *_y_acc4 = reinterpret_cast<float4 *>(_y_acc);

        nchans_in  /= VEC_SIZE;
        nchans_out /= VEC_SIZE;
        
        params.nchan_in       = nchans_in;
        params.nchan_out      = nchans_out;

        const int nloc = DIV_UP(nchans_out, bdimx);

        constexpr int MAX_LOCAL_VEC_LEN = MAX_LOCAL_ARR_LEN / VEC_SIZE;
        constexpr int MIN_LOC_VEC_LEN = MAX_LOCAL_VEC_LEN/2+1;

        switch(bdimx) {
            case   32: launch_spc_attn_ring_fwd<  32,               1, MAX_LOCAL_VEC_LEN>(params, nloc, batch_size, _kxp4, _vxp4, _qyp4, _row_idx, _row_off, _col_idx, _quad_weights, _y_acc4, _alpha_sum, _qdotk_max, stream); break;
            case   64: launch_spc_attn_ring_fwd<  64, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(params, nloc, batch_size, _kxp4, _vxp4, _qyp4, _row_idx, _row_off, _col_idx, _quad_weights, _y_acc4, _alpha_sum, _qdotk_max, stream); break;
            case  128: launch_spc_attn_ring_fwd< 128, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(params, nloc, batch_size, _kxp4, _vxp4, _qyp4, _row_idx, _row_off, _col_idx, _quad_weights, _y_acc4, _alpha_sum, _qdotk_max, stream); break;
            case  256: launch_spc_attn_ring_fwd< 256, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(params, nloc, batch_size, _kxp4, _vxp4, _qyp4, _row_idx, _row_off, _col_idx, _quad_weights, _y_acc4, _alpha_sum, _qdotk_max, stream); break;
            case  512: launch_spc_attn_ring_fwd< 512, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(params, nloc, batch_size, _kxp4, _vxp4, _qyp4, _row_idx, _row_off, _col_idx, _quad_weights, _y_acc4, _alpha_sum, _qdotk_max, stream); break;
            case 1024: launch_spc_attn_ring_fwd<1024, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(params, nloc, batch_size, _kxp4, _vxp4, _qyp4, _row_idx, _row_off, _col_idx, _quad_weights, _y_acc4, _alpha_sum, _qdotk_max, stream); break;
            default:   launch_gen_attn_ring_fwd                                          (params,       batch_size, _kxp4, _vxp4, _qyp4, _row_idx, _row_off, _col_idx, _quad_weights, _y_acc4, _alpha_sum, _qdotk_max, stream); break;
        }
    }
}

void s2_attention_fwd_ring_step_cuda(
    at::Tensor kx,
    at::Tensor vx,
    at::Tensor qy,
    at::Tensor y_acc,
    at::Tensor alpha_sum_buf,
    at::Tensor qdotk_max_buf,
    at::Tensor quad_weights,
    at::Tensor psi_col_idx,
    at::Tensor psi_row_off,
    at::Tensor psi_row_idx,
    int64_t nlon_in,
    int64_t pscale,
    int64_t lon_lo_kx,
    int64_t lat_halo_start,
    int64_t nlat_out,
    int64_t nlon_out)
{
    CHECK_CUDA_INPUT_TENSOR(kx);
    CHECK_CUDA_INPUT_TENSOR(vx);
    CHECK_CUDA_INPUT_TENSOR(qy);
    CHECK_CUDA_TENSOR(y_acc);
    CHECK_CUDA_TENSOR(alpha_sum_buf);
    CHECK_CUDA_TENSOR(qdotk_max_buf);
    CHECK_CUDA_TENSOR(quad_weights);
    CHECK_CUDA_TENSOR(psi_col_idx);
    CHECK_CUDA_TENSOR(psi_row_off);
    CHECK_CUDA_TENSOR(psi_row_idx);

    const int batch_size = kx.size(0);
    const int nlat_halo  = kx.size(2);   // kx is [B,C,H,W], H is dim 2
    const int nlon_kx    = kx.size(3);   // W is dim 3
    const size_t nchans_in  = qy.size(1);
    const size_t nchans_out = vx.size(1);

    torch::Tensor kxP = kx.to(torch::kFloat32);
    torch::Tensor vxP = vx.to(torch::kFloat32);
    torch::Tensor qyP = qy.to(torch::kFloat32);

    bool kx_is_channels_last = kxP.strides()[1] == 1;
    bool vx_is_channels_last = vxP.strides()[1] == 1;
    bool qy_is_channels_last = qyP.strides()[1] == 1;

    if (!kx_is_channels_last) { kxP = permute_4D_to0231(kxP); }
    if (!vx_is_channels_last) { vxP = permute_4D_to0231(vxP); }
    if (!qy_is_channels_last) { qyP = permute_4D_to0231(qyP); }

    s2_attn_fwd_ring_step_dispatch(
        batch_size, nchans_in, nchans_out,
        nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start,
        nlat_out, nlon_out,
        kxP, vxP, qyP,
        psi_row_idx, psi_row_off, psi_col_idx,
        quad_weights,
        y_acc, alpha_sum_buf, qdotk_max_buf);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY_IMPL(attention_kernels, CUDA, m)
{
    m.impl("forward_ring_step", &s2_attention_fwd_ring_step_cuda);
}

}
