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

#include "cudamacro.h"
#include "attention_cuda_utils.cuh"

#define THREADS (64)

#define MAX_LOCAL_ARR_LEN (16)

#define PASS2_MIN_WORK_PER_BLOCK (32)
#define PASS2_ROW_LENGTH_THRES   (0.1f)

// Ring-step backward variant for DistributedNeighborhoodAttentionS2. Two
// passes per ring step:
//   pass1: accumulate softmax statistics (alpha_sum, qdotk_max, integral)
//          plus the per-output buffers alpha_k, alpha_kvw needed to finalize
//          dqy after all ring steps complete.
//   pass2: scatter dkx/dvx contributions for the current KV chunk, using the
//          finalized state from pass1 (alpha_sum, qdotk_max, integral_norm).
// col_idx must have wi pre-shifted by pscale * lon_lo_out (see
// _build_local_psi in distributed_attention.py).

namespace attention_kernels {

    void dump_csr_linear(const char *fname, int64_t pscale, int64_t nlon_in, int64_t lon_lo_kx, int nlon_kx, int64_t lat_halo_start, int nlat_halo, int64_t nrows, at::Tensor row_idx, at::Tensor row_off, at::Tensor col_idx);

// Pass 1: accumulate softmax statistics across ring steps.
// After all ring steps, finalize dqy in Python using the accumulated state.
template<int BDIM_X, typename FLOATV_T>
__global__
__launch_bounds__(BDIM_X)
void s2_attn_bwd_ring_step_pass1_generic_vec_k(
    int nchan_in,
    int nchan_out,
    int nlat_halo,
    int nlon_kx,
    int nlon_in,
    int pscale,           // GLOBAL pscale = nlon_in / nlon_out_global
    int lon_lo_kx,
    int lat_halo_start,
    int nlat_out,
    int nlon_out,
    const FLOATV_T *__restrict__ kx,           // [batch][nlat_halo][nlon_kx][nchan_in]
    const FLOATV_T *__restrict__ vx,           // [batch][nlat_halo][nlon_kx][nchan_out]
    const FLOATV_T *__restrict__ qy,           // [batch][nlat_out][nlon_out][nchan_in]
    const FLOATV_T *__restrict__ dy,           // [batch][nlat_out][nlon_out][nchan_out]
    const int32_t  *__restrict__ row_idx,
    const int64_t  *__restrict__ row_off,
    const int64_t  *__restrict__ col_idx,
    const float    *__restrict__ quad_weights,
    float    *__restrict__ alpha_sum_buf,      // [batch][nlat_out][nlon_out] (in/out)
    float    *__restrict__ qdotk_max_buf,      // [batch][nlat_out][nlon_out] (in/out)
    float    *__restrict__ integral_buf,       // [batch][nlat_out][nlon_out] unnormalized (in/out)
    FLOATV_T *__restrict__ alpha_k_buf,        // [batch][nlat_out][nlon_out][nchan_in] (in/out)
    FLOATV_T *__restrict__ alpha_kvw_buf       // [batch][nlat_out][nlon_out][nchan_in] (in/out)
) {
    extern __shared__ __align__(sizeof(float4)) float shext[];
    // sh_alpha_k[nchan_in], sh_alpha_kvw[nchan_in], sh_dy[nchan_out]
    FLOATV_T *sh_alpha_k   = reinterpret_cast<FLOATV_T *>(shext) + threadIdx.y * (2*nchan_in + nchan_out);
    FLOATV_T *sh_alpha_kvw = sh_alpha_k   + nchan_in;
    FLOATV_T *sh_dy        = sh_alpha_kvw + nchan_in;

    const int batch = blockIdx.y;
    const int wid   = blockIdx.x * blockDim.y + threadIdx.y;
    if (wid >= nlat_out * nlon_out) return;

    const int tidx  = threadIdx.x;
    const int h     = wid / nlon_out;
    const int wo    = wid - (h * nlon_out);
    const int ho    = row_idx[h];

    kx += int64_t(batch) * nlat_halo * nlon_kx * nchan_in;
    vx += int64_t(batch) * nlat_halo * nlon_kx * nchan_out;
    qy += int64_t(batch) * nlat_out * nlon_out * nchan_in
        + int64_t(ho) * nlon_out * nchan_in + int64_t(wo) * nchan_in;
    dy += int64_t(batch) * nlat_out * nlon_out * nchan_out
        + int64_t(ho) * nlon_out * nchan_out + int64_t(wo) * nchan_out;

    const int64_t out_flat = int64_t(batch) * nlat_out * nlon_out
                           + int64_t(ho) * nlon_out + wo;
    alpha_sum_buf  += out_flat;
    qdotk_max_buf  += out_flat;
    integral_buf   += out_flat;
    alpha_k_buf    += out_flat * nchan_in;
    alpha_kvw_buf  += out_flat * nchan_in;

    // Load current state
    float alpha_sum = alpha_sum_buf[0];
    float qdotk_max = qdotk_max_buf[0];
    float integral  = integral_buf[0];

    for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
        sh_alpha_k[chan]   = alpha_k_buf[chan];
        sh_alpha_kvw[chan] = alpha_kvw_buf[chan];
    }
    for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
        sh_dy[chan] = dy[chan];
    }

#if __CUDA_ARCH__ < 900
    if constexpr(std::is_same<FLOATV_T, float4>::value) { __syncwarp(); }
#endif

    // `pscale` is the GLOBAL ratio nlon_in / nlon_out_global, passed by the caller.
    // The kernel's `nlon_out` is the LOCAL output width (nlon_out_local), not the global one.

    const int64_t rbeg = row_off[ho];
    const int64_t rend = row_off[ho + 1];
    col_idx += rbeg;
    const int rlen = rend - rbeg;

    for (int off = 0; off < rlen; off++) {
        const int64_t col   = col_idx[off];
        const int hi_global = col / nlon_in;
        const int wi        = col - (hi_global * nlon_in);
        // wip = (wi + pscale * wo_local) % nlon_in
        //     = (wi_canonical + pscale * (lon_lo_out + wo_local)) % nlon_in
        const int wi_wo     = wi + pscale * wo;
        const int wip       = wi_wo - (wi_wo / nlon_in) * nlon_in;

        if (wip < lon_lo_kx || wip >= lon_lo_kx + nlon_kx) continue;

        const int hi_local  = hi_global - lat_halo_start;
        if (hi_local < 0 || hi_local >= nlat_halo) continue;
        const int wip_local = wip - lon_lo_kx;

        const FLOATV_T *_kx = kx + int64_t(hi_local)*nlon_kx*nchan_in  + int64_t(wip_local)*nchan_in;
        const FLOATV_T *_vx = vx + int64_t(hi_local)*nlon_kx*nchan_out + int64_t(wip_local)*nchan_out;

        FLOATV_T qdotk_v = __vset<FLOATV_T>(0.0f);
        FLOATV_T gdotv_v = __vset<FLOATV_T>(0.0f);
        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
            qdotk_v = __vadd(qdotk_v, __vmul(qy[chan], _kx[chan]));
        }
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
            gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[chan], _vx[chan]));
        }
        const float qdotk = __warp_sum(__vred(qdotk_v));
        const float gdotv = __warp_sum(__vred(gdotv_v));

        const float qdotk_max_tmp = max(qdotk_max, qdotk);
        const float alpha_inz     = expf(qdotk - qdotk_max_tmp) * quad_weights[hi_global];
        const float max_correction = expf(qdotk_max - qdotk_max_tmp);

        alpha_sum = alpha_sum * max_correction + alpha_inz;
        integral  = integral  * max_correction + alpha_inz * gdotv;

        const float ainz_gdotv = alpha_inz * gdotv;
        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
            const FLOATV_T kxval = _kx[chan];
            sh_alpha_k[chan]   = __vadd(__vscale(max_correction, sh_alpha_k[chan]),
                                        __vscale(alpha_inz,  kxval));
            sh_alpha_kvw[chan] = __vadd(__vscale(max_correction, sh_alpha_kvw[chan]),
                                        __vscale(ainz_gdotv, kxval));
        }
        qdotk_max = qdotk_max_tmp;
    }

    // Store updated state
    alpha_sum_buf[0] = alpha_sum;
    qdotk_max_buf[0] = qdotk_max;
    integral_buf[0]  = integral;
    for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
        alpha_k_buf[chan]   = sh_alpha_k[chan];
        alpha_kvw_buf[chan] = sh_alpha_kvw[chan];
    }
}

// Pass 1: accumulate softmax statistics across ring steps.
// After all ring steps, finalize dqy in Python using the accumulated state.
//
// called with either (BDIM_X=32 and BDIM_Y>1) || (2^K=BDIM_X > 32 and BDIM_Y=1)
template<int BDIM_X,
         int BDIM_Y,
         int CHOUT_AS_IN,   // 1 iif "BDIM_X*(NLOC-1) <= nchan_out <= BDIM_X*NLOC" else 0
         int NLOC,          // smallest int such that BDIM_X*NLOC >= nchan_in
         typename FLOATV_T> // either float or float4
__global__
__launch_bounds__(BDIM_X*BDIM_Y)
void s2_attn_bwd_ring_step_pass1_special_vec_k(int nchan_in,
                                               int nchan_out,
                                               int nlat_halo,
                                               int nlon_kx,
                                               int nlon_in,
                                               int pscale,           // GLOBAL pscale = nlon_in / nlon_out_global
                                               int lon_lo_kx,
                                               int lat_halo_start,
                                               int nlat_out,
                                               int nlon_out,
                                               const FLOATV_T *__restrict__ kx,           // [batch][nlat_halo][nlon_kx][nchan_in]
                                               const FLOATV_T *__restrict__ vx,           // [batch][nlat_halo][nlon_kx][nchan_out]
                                               const FLOATV_T *__restrict__ qy,           // [batch][nlat_out][nlon_out][nchan_in]
                                               const FLOATV_T *__restrict__ dy,           // [batch][nlat_out][nlon_out][nchan_out]
                                               const int32_t  *__restrict__ row_idx,
                                               const int64_t  *__restrict__ row_off,
                                               const int64_t  *__restrict__ col_idx,
                                               const float    *__restrict__ quad_weights,
                                                     float    *__restrict__ alpha_sum_buf,      // [batch][nlat_out][nlon_out] (in/out)
                                                     float    *__restrict__ qdotk_max_buf,      // [batch][nlat_out][nlon_out] (in/out)
                                                     float    *__restrict__ integral_buf,       // [batch][nlat_out][nlon_out] unnormalized (in/out)
                                                     FLOATV_T *__restrict__ alpha_k_buf,        // [batch][nlat_out][nlon_out][nchan_in] (in/out)
                                                     FLOATV_T *__restrict__ alpha_kvw_buf) {    // [batch][nlat_out][nlon_out][nchan_in] (in/out)

    static_assert(0 == (BDIM_X & (BDIM_X-1)));
    static_assert(0 == (BDIM_Y & (BDIM_Y-1)));
    static_assert((BDIM_X == 32 && BDIM_Y  > 1) ||
                  (BDIM_X  > 32 && BDIM_Y == 1)) ;

    const int tidx = threadIdx.x;
    const int batch = blockIdx.y;
    const uint64_t ctaid = uint64_t(blockIdx.x) * blockDim.y + threadIdx.y;
    
    if (ctaid >= uint64_t(nlat_out)*nlon_out) {
        return;
    }

    extern __shared__ __align__(sizeof(float4)) float shext[];

    // sh_alpha_k[nchan_in], sh_alpha_kvw[nchan_in], sh_dy[nchan_out]
    FLOATV_T loc_k__[NLOC];
    FLOATV_T loc_kvw[NLOC];

    FLOATV_T *sh_dy = reinterpret_cast<FLOATV_T *>(shext) + threadIdx.y*(nchan_in+nchan_out) + tidx;
    FLOATV_T *sh_qy = sh_dy + nchan_out; // [nchan_in], so always offest by tidx

    const int h     = ctaid / nlon_out;
    const int wo    = ctaid - (h * nlon_out);
    const int ho    = row_idx[h];

    kx += int64_t(batch)*nlat_halo*nlon_kx*nchan_in + tidx;
    qy += int64_t(batch)*nlat_out*nlon_out*nchan_in  + int64_t(ho)*nlon_out*nchan_in  + int64_t(wo)*nchan_in + tidx;

    vx += int64_t(batch)*nlat_halo*nlon_kx*nchan_out + tidx;
    dy += int64_t(batch)*nlat_out*nlon_out*nchan_out + int64_t(ho)*nlon_out*nchan_out + int64_t(wo)*nchan_out + tidx;

    const int64_t out_flat = int64_t(batch)*nlat_out*nlon_out + int64_t(ho)*nlon_out + wo;

    alpha_sum_buf  += out_flat;
    qdotk_max_buf  += out_flat;
    integral_buf   += out_flat;
    alpha_k_buf    += out_flat*nchan_in + tidx;
    alpha_kvw_buf  += out_flat*nchan_in + tidx;

    // Load current state
    float alpha_sum = alpha_sum_buf[0];
    float qdotk_max = qdotk_max_buf[0];
    float integral  = integral_buf[0];

    strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { loc_k__[i] =   alpha_k_buf[i*BDIM_X]; });
    strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { loc_kvw[i] = alpha_kvw_buf[i*BDIM_X]; });

    strided_op<BDIM_X,               NLOC    >(nchan_in,  [&](int i) { sh_qy[i*BDIM_X] = qy[i*BDIM_X]; });
    strided_op<BDIM_X, CHOUT_AS_IN ? NLOC : 0>(nchan_out, [&](int i) { sh_dy[i*BDIM_X] = dy[i*BDIM_X]; });

    // `pscale` is the GLOBAL ratio nlon_in / nlon_out_global, passed by the caller.
    // The kernel's `nlon_out` is the LOCAL output width (nlon_out_local), not the global one.

    const int64_t rbeg = row_off[ho];
    const int64_t rend = row_off[ho + 1];
    col_idx += rbeg;
    const int rlen = rend - rbeg;

    for (int off = 0; off < rlen; off++) {
        const int64_t col   = col_idx[off];
        const int hi_global = col / nlon_in;
        const int wi        = col - (hi_global * nlon_in);
        // wip = (wi + pscale * wo_local) % nlon_in
        //     = (wi_canonical + pscale * (lon_lo_out + wo_local)) % nlon_in
        const int wi_wo     = wi + pscale * wo;
        const int wip       = wi_wo - (wi_wo / nlon_in) * nlon_in;

        if (wip < lon_lo_kx || wip >= lon_lo_kx + nlon_kx) continue;

        const int hi_local  = hi_global - lat_halo_start;
        if (hi_local < 0 || hi_local >= nlat_halo) continue;
        const int wip_local = wip - lon_lo_kx;

        const FLOATV_T *_kx = kx + int64_t(hi_local)*nlon_kx*nchan_in  + int64_t(wip_local)*nchan_in;
        const FLOATV_T *_vx = vx + int64_t(hi_local)*nlon_kx*nchan_out + int64_t(wip_local)*nchan_out;

        FLOATV_T qdotk_v = __vset<FLOATV_T>(0.0f);
        FLOATV_T gdotv_v = __vset<FLOATV_T>(0.0f);

        strided_op<BDIM_X,               NLOC>    (nchan_in,  [&](int i) { qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[i*BDIM_X], _kx[i*BDIM_X])); });
        strided_op<BDIM_X, CHOUT_AS_IN ? NLOC : 0>(nchan_out, [&](int i) { gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[i*BDIM_X], _vx[i*BDIM_X])); });

        float qdotk = __vred(qdotk_v);
        float gdotv = __vred(gdotv_v);

        if constexpr(BDIM_X == 32) { 
            qdotk = __warp_sum(qdotk); 
            gdotv = __warp_sum(gdotv); 
        } else {
            qdotk = __block_sum<BDIM_X>(qdotk);
            gdotv = __block_sum<BDIM_X>(gdotv);
        }

        const float qdotk_max_tmp = max(qdotk_max, qdotk);
        const float alpha_inz     = expf(qdotk - qdotk_max_tmp) * quad_weights[hi_global];
        const float max_correction = expf(qdotk_max - qdotk_max_tmp);

        alpha_sum = alpha_sum * max_correction + alpha_inz;
        integral  = integral  * max_correction + alpha_inz * gdotv;

        const float ainz_gdotv = alpha_inz * gdotv;

        strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { loc_k__[i] = __vadd(__vscale(max_correction, loc_k__[i]), __vscale(alpha_inz,  _kx[i*BDIM_X])); });
        strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { loc_kvw[i] = __vadd(__vscale(max_correction, loc_kvw[i]), __vscale(ainz_gdotv, _kx[i*BDIM_X])); });

        qdotk_max = qdotk_max_tmp;
    }

    // Store updated state
    if (!tidx) {
        alpha_sum_buf[0] = alpha_sum;
        qdotk_max_buf[0] = qdotk_max;
        integral_buf[0]  = integral;
    }

    strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) {   alpha_k_buf[i*BDIM_X] = loc_k__[i]; });
    strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { alpha_kvw_buf[i*BDIM_X] = loc_kvw[i]; });

    return;
}

template<typename FLOATV_T>
void launch_gen_attn_ring_pass1_bwd(int64_t batch_size, 
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
                                    FLOATV_T *_kxp,
                                    FLOATV_T *_vxp,
                                    FLOATV_T *_qyp,
                                    FLOATV_T *_dyp,
                                    int32_t *_row_idx,
                                    int64_t *_row_off,
                                    int64_t *_col_idx,
                                    float *_quad_weights,
                                    float *_alpha_sum,
                                    float *_qdotk_max,
                                    float *_integral,
                                    FLOATV_T *_alpha_k,
                                    FLOATV_T *_alpha_kvw,
                                    cudaStream_t stream) {

    dim3 block(WARP_SIZE, THREADS / WARP_SIZE);
    dim3 grid(DIV_UP(nlat_out*nlon_out, block.y), batch_size);

    //size_t shsize = sizeof(FLOATV_T)*(nchans_in*4+nchans_out) * block.y; // 5 arrays per warp
    size_t shsize = sizeof(FLOATV_T) * (2*nchans_in + nchans_out) * block.y;

    s2_attn_bwd_ring_step_pass1_generic_vec_k<THREADS>
                                             <<<grid, block, shsize, stream>>>(nchans_in, nchans_out, nlat_halo, nlon_kx,
                                                                               nlon_in, pscale, lon_lo_kx, lat_halo_start, nlat_out, nlon_out,
                                                                               _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights,
                                                                               _alpha_sum, _qdotk_max, _integral, _alpha_k, _alpha_kvw);
    CHECK_ERROR("s2_attn_bwd_ring_step_pass1_generic_vec_k");

    return;
}

template<int BDIM_X,
         int CUR_LOC_SIZE,
         int MAX_LOC_SIZE, // max size of FLOATV_T[] local array
         typename FLOATV_T>
void launch_spc_attn_ring_pass1_bwd(int nloc,
                                    int64_t batch_size,
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
                                    FLOATV_T *_kxp,
                                    FLOATV_T *_vxp,
                                    FLOATV_T *_qyp,
                                    FLOATV_T *_dyp,
                                    int32_t *_row_idx,
                                    int64_t *_row_off,
                                    int64_t *_col_idx,
                                    float *_quad_weights,
                                    float *_alpha_sum,
                                    float *_qdotk_max,
                                    float *_integral,
                                    FLOATV_T *_alpha_k,
                                    FLOATV_T *_alpha_kvw,
                                    cudaStream_t stream) {


    static_assert(0 == (BDIM_X & (BDIM_X-1)));
    static_assert(std::is_same<FLOATV_T, float>::value ||
                  std::is_same<FLOATV_T, float4>::value);

    if (CUR_LOC_SIZE == nloc) {

        // nloc determines the size of local arrays used to store
        // temporary buffers loc_k__[], loc_vw_[] and loc_kvw[],
        // of size nchans_in each;
        // if nchans_out is >= BDIM_X*(nloc-1) and <= BDIM_X*nloc
        // then we can use the same compile-time known loops used
        // for input channels, with the exception of testing 
        // whether to execute the last iteration based on "nchans_out"
        // instead of "nchans_in"; in this way as long as the
        // difference between the number of input and output channels
        // is <= BDIM_X we can use the faster path
        const bool chout_as_in = (nchans_out >= BDIM_X*(CUR_LOC_SIZE-1) && 
                                  nchans_out <= BDIM_X* CUR_LOC_SIZE  );

        constexpr int BDIM_Y = (BDIM_X <= 32) ? THREADS/BDIM_X : 1;

        dim3 block(BDIM_X, BDIM_Y);
        dim3 grid(DIV_UP(nlat_out*nlon_out, block.y), batch_size);

        size_t shsize = sizeof(FLOATV_T)*(nchans_in+nchans_out) * block.y;

        if (chout_as_in) {
            s2_attn_bwd_ring_step_pass1_special_vec_k<BDIM_X, BDIM_Y, 1, CUR_LOC_SIZE>
                                                     <<<grid, block, shsize, stream>>>(nchans_in, nchans_out, nlat_halo, nlon_kx,
                                                                                       nlon_in, pscale, lon_lo_kx, lat_halo_start, nlat_out, nlon_out,
                                                                                       _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights,
                                                                                       _alpha_sum, _qdotk_max, _integral, _alpha_k, _alpha_kvw);
        } else {
            s2_attn_bwd_ring_step_pass1_special_vec_k<BDIM_X, BDIM_Y, 0, CUR_LOC_SIZE>
                                                     <<<grid, block, shsize, stream>>>(nchans_in, nchans_out, nlat_halo, nlon_kx,
                                                                                       nlon_in, pscale, lon_lo_kx, lat_halo_start, nlat_out, nlon_out,
                                                                                       _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights,
                                                                                       _alpha_sum, _qdotk_max, _integral, _alpha_k, _alpha_kvw);
        }
        CHECK_ERROR("s2_attn_bwd_special_vec_k");

        return;
    }
    if constexpr(CUR_LOC_SIZE < MAX_LOC_SIZE) {
        launch_spc_attn_ring_pass1_bwd<BDIM_X,
                                       CUR_LOC_SIZE+1,
                                       MAX_LOC_SIZE>(nloc, batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo,
                                                     nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out, _kxp,
                                                     _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights,
                                                     _alpha_sum, _qdotk_max, _integral, _alpha_k, _alpha_kvw, stream);
    }
    return;
}

static void s2_attn_bwd_ring_step_pass1_dispatch(int64_t batch_size, 
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
                                                 at::Tensor dyP,
                                                 at::Tensor row_idx,
                                                 at::Tensor row_off,
                                                 at::Tensor col_idx,
                                                 at::Tensor quad_weights,
                                                 at::Tensor alpha_sum_buf,
                                                 at::Tensor qdotk_max_buf,
                                                 at::Tensor integral_buf,
                                                 at::Tensor alpha_k_buf,
                                                 at::Tensor alpha_kvw_buf) {

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    
    // smallest power of two "bdimx" (>=32) s.t. bdimx*MAX_LOCAL_ARR_LEN >= nchans_in
    int bdimx;
    bdimx = DIV_UP(nchans_in, MAX_LOCAL_ARR_LEN);
    bdimx = max(bdimx, WARP_SIZE);
    bdimx = next_pow2(bdimx);

    float *_kxp          = reinterpret_cast<float *>(kxP.data_ptr());
    float *_vxp          = reinterpret_cast<float *>(vxP.data_ptr());
    float *_qyp          = reinterpret_cast<float *>(qyP.data_ptr());
    float *_dyp          = reinterpret_cast<float *>(dyP.data_ptr());
    float *_alpha_sum    = reinterpret_cast<float *>(alpha_sum_buf.data_ptr());
    float *_qdotk_max    = reinterpret_cast<float *>(qdotk_max_buf.data_ptr());
    float *_integral     = reinterpret_cast<float *>(integral_buf.data_ptr());
    int32_t *_row_idx    = reinterpret_cast<int32_t *>(row_idx.data_ptr());
    int64_t *_row_off    = reinterpret_cast<int64_t *>(row_off.data_ptr());
    int64_t *_col_idx    = reinterpret_cast<int64_t *>(col_idx.data_ptr());
    float *_quad_weights = reinterpret_cast<float *>(quad_weights.data_ptr());
    float *_alpha_k      = reinterpret_cast<float *>(alpha_k_buf.data_ptr());
    float *_alpha_kvw    = reinterpret_cast<float *>(alpha_kvw_buf.data_ptr());

    constexpr int VEC_SIZE = sizeof(float4) / sizeof(float);

    if (!is_aligned<sizeof(float4)>(_kxp) ||
        !is_aligned<sizeof(float4)>(_vxp) ||
        !is_aligned<sizeof(float4)>(_qyp) ||
        !is_aligned<sizeof(float4)>(_dyp) ||
        (nchans_in  % VEC_SIZE) != 0 ||
        (nchans_out % VEC_SIZE) != 0) {

        const int nloc = DIV_UP(nchans_in, bdimx);

        constexpr int MIN_LOC_ARR_LEN = MAX_LOCAL_ARR_LEN/2+1;

        switch(bdimx) {
            case   32: launch_spc_attn_ring_pass1_bwd< 32,               1, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _alpha_sum, _qdotk_max, _integral, _alpha_k, _alpha_kvw, stream); break;
            case   64: launch_spc_attn_ring_pass1_bwd< 64, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _alpha_sum, _qdotk_max, _integral, _alpha_k, _alpha_kvw, stream); break;
            case  128: launch_spc_attn_ring_pass1_bwd<128, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _alpha_sum, _qdotk_max, _integral, _alpha_k, _alpha_kvw, stream); break;
            case  256: launch_spc_attn_ring_pass1_bwd<256, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _alpha_sum, _qdotk_max, _integral, _alpha_k, _alpha_kvw, stream); break;
            case  512: launch_spc_attn_ring_pass1_bwd<512, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _alpha_sum, _qdotk_max, _integral, _alpha_k, _alpha_kvw, stream); break;
            default:   launch_gen_attn_ring_pass1_bwd                                         (      batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _alpha_sum, _qdotk_max, _integral, _alpha_k, _alpha_kvw, stream); break;
        }

    } else {

        float4 *_kxp4    = reinterpret_cast<float4 *>(_kxp);
        float4 *_vxp4    = reinterpret_cast<float4 *>(_vxp);
        float4 *_qyp4    = reinterpret_cast<float4 *>(_qyp);
        float4 *_dyp4    = reinterpret_cast<float4 *>(_dyp);
        float4 *_alpha_k4   = reinterpret_cast<float4 *>(alpha_k_buf.data_ptr());
        float4 *_alpha_kvw4 = reinterpret_cast<float4 *>(alpha_kvw_buf.data_ptr());

        nchans_in  /= VEC_SIZE;
        nchans_out /= VEC_SIZE;

        const int nloc = DIV_UP(nchans_in, bdimx);

        constexpr int MAX_LOCAL_VEC_LEN = MAX_LOCAL_ARR_LEN / VEC_SIZE;
        constexpr int MIN_LOC_VEC_LEN = MAX_LOCAL_VEC_LEN/2+1;

        switch(bdimx) {
            case   32: launch_spc_attn_ring_pass1_bwd< 32,               1, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, _dyp4, _row_idx, _row_off, _col_idx, _quad_weights, _alpha_sum, _qdotk_max, _integral, _alpha_k4, _alpha_kvw4, stream); break;
            case   64: launch_spc_attn_ring_pass1_bwd< 64, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, _dyp4, _row_idx, _row_off, _col_idx, _quad_weights, _alpha_sum, _qdotk_max, _integral, _alpha_k4, _alpha_kvw4, stream); break;
            case  128: launch_spc_attn_ring_pass1_bwd<128, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, _dyp4, _row_idx, _row_off, _col_idx, _quad_weights, _alpha_sum, _qdotk_max, _integral, _alpha_k4, _alpha_kvw4, stream); break;
            case  256: launch_spc_attn_ring_pass1_bwd<256, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, _dyp4, _row_idx, _row_off, _col_idx, _quad_weights, _alpha_sum, _qdotk_max, _integral, _alpha_k4, _alpha_kvw4, stream); break;
            case  512: launch_spc_attn_ring_pass1_bwd<512, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, _dyp4, _row_idx, _row_off, _col_idx, _quad_weights, _alpha_sum, _qdotk_max, _integral, _alpha_k4, _alpha_kvw4, stream); break;
            default:   launch_gen_attn_ring_pass1_bwd                                         (      batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, _dyp4, _row_idx, _row_off, _col_idx, _quad_weights, _alpha_sum, _qdotk_max, _integral, _alpha_k4, _alpha_kvw4, stream);
        }
    }
}


// Pass 2: scatter dkx/dvx contributions for the current KV chunk.
// Requires FINALIZED state from pass 1: alpha_sum, qdotk_max, integral_norm (= integral/alpha_sum).
template<int BDIM_X, typename FLOATV_T>
__global__
__launch_bounds__(BDIM_X)
void s2_attn_bwd_ring_step_pass2_generic_vec_k(
    int nchan_in,
    int nchan_out,
    int nlat_halo,
    int nlon_kx,
    int nlon_in,
    int pscale,           // GLOBAL pscale = nlon_in / nlon_out_global
    int lon_lo_kx,
    int lat_halo_start,
    int nlat_out,
    int nlon_out,
    const FLOATV_T *__restrict__ kx,
    const FLOATV_T *__restrict__ vx,
    const FLOATV_T *__restrict__ qy,
    const FLOATV_T *__restrict__ dy,
    const int32_t  *__restrict__ row_idx,
    const int64_t  *__restrict__ row_off,
    const int64_t  *__restrict__ col_idx,
    const float    *__restrict__ quad_weights,
    const float    *__restrict__ alpha_sum_buf,      // finalized [batch][nlat_out][nlon_out]
    const float    *__restrict__ qdotk_max_buf,      // finalized [batch][nlat_out][nlon_out]
    const float    *__restrict__ integral_norm_buf,  // finalized, normalized [batch][nlat_out][nlon_out]
          FLOATV_T *__restrict__ dkx,                // [batch][nlat_halo][nlon_kx][nchan_in] (atomically updated)
          FLOATV_T *__restrict__ dvx                 // [batch][nlat_halo][nlon_kx][nchan_out] (atomically updated)
) {
    extern __shared__ __align__(sizeof(float4)) float shext[];
    FLOATV_T *sh_qy = reinterpret_cast<FLOATV_T *>(shext) + threadIdx.y * (nchan_in + nchan_out);
    FLOATV_T *sh_dy = sh_qy + nchan_in;

    const int batch = blockIdx.y;
    const int wid   = blockIdx.x * blockDim.y + threadIdx.y;
    if (wid >= nlat_out * nlon_out) return;

    const int tidx  = threadIdx.x;
    const int h     = wid / nlon_out;
    const int wo    = wid - (h * nlon_out);
    const int ho    = row_idx[h];

    kx  += int64_t(batch) * nlat_halo * nlon_kx * nchan_in;
    vx  += int64_t(batch) * nlat_halo * nlon_kx * nchan_out;
    dkx += int64_t(batch) * nlat_halo * nlon_kx * nchan_in;
    dvx += int64_t(batch) * nlat_halo * nlon_kx * nchan_out;

    qy  += int64_t(batch) * nlat_out * nlon_out * nchan_in
         + int64_t(ho) * nlon_out * nchan_in + int64_t(wo) * nchan_in;
    dy  += int64_t(batch) * nlat_out * nlon_out * nchan_out
         + int64_t(ho) * nlon_out * nchan_out + int64_t(wo) * nchan_out;

    const int64_t out_flat     = int64_t(batch) * nlat_out * nlon_out + int64_t(ho) * nlon_out + wo;
    const float alpha_sum      = alpha_sum_buf[out_flat];
    const float qdotk_max      = qdotk_max_buf[out_flat];
    const float integral_norm  = integral_norm_buf[out_flat];
    const float alpha_sum_inv  = 1.0f / alpha_sum;

    for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
        sh_qy[chan] = qy[chan];
    }
    for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
        sh_dy[chan] = dy[chan];
    }

#if __CUDA_ARCH__ < 900
    if constexpr(std::is_same<FLOATV_T, float4>::value) { __syncwarp(); }
#endif

    // `pscale` is the GLOBAL ratio nlon_in / nlon_out_global, passed by the caller.
    // The kernel's `nlon_out` is the LOCAL output width (nlon_out_local), not the global one.

    const int64_t rbeg = row_off[ho];
    const int64_t rend = row_off[ho + 1];
    col_idx += rbeg;
    const int rlen = rend - rbeg;

    for (int off = 0; off < rlen; off++) {
        const int64_t col   = col_idx[off];
        const int hi_global = col / nlon_in;
        const int wi        = col - (hi_global * nlon_in);
        // wip = (wi + pscale * wo_local) % nlon_in
        //     = (wi_canonical + pscale * (lon_lo_out + wo_local)) % nlon_in
        const int wi_wo     = wi + pscale * wo;
        const int wip       = wi_wo - (wi_wo / nlon_in) * nlon_in;

        if (wip < lon_lo_kx || wip >= lon_lo_kx + nlon_kx) continue;

        const int hi_local  = hi_global - lat_halo_start;
        if (hi_local < 0 || hi_local >= nlat_halo) continue;
        const int wip_local = wip - lon_lo_kx;

        const FLOATV_T *_kx = kx + int64_t(hi_local)*nlon_kx*nchan_in  + int64_t(wip_local)*nchan_in;
        const FLOATV_T *_vx = vx + int64_t(hi_local)*nlon_kx*nchan_out + int64_t(wip_local)*nchan_out;
        FLOATV_T *_dkx = dkx + int64_t(hi_local)*nlon_kx*nchan_in  + int64_t(wip_local)*nchan_in;
        FLOATV_T *_dvx = dvx + int64_t(hi_local)*nlon_kx*nchan_out + int64_t(wip_local)*nchan_out;

        FLOATV_T qdotk_v = __vset<FLOATV_T>(0.0f);
        FLOATV_T gdotv_v = __vset<FLOATV_T>(0.0f);
        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
            qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[chan], _kx[chan]));
        }
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
            gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[chan], _vx[chan]));
        }
        const float qdotk = __warp_sum(__vred(qdotk_v));
        const float gdotv = __warp_sum(__vred(gdotv_v));

        const float alpha_inz  = expf(qdotk - qdotk_max) * quad_weights[hi_global];
        const float alpha_mul  = alpha_inz * alpha_sum_inv;
        const float scale_dkx = (gdotv - integral_norm) * alpha_mul;
        const float scale_dvx = alpha_mul;

#if __CUDA_ARCH__ < 900
        float *sh_qy_scl = reinterpret_cast<float *>(sh_qy);
        float *sh_dy_scl = reinterpret_cast<float *>(sh_dy);
        float *_dkx_scl  = reinterpret_cast<float *>(_dkx);
        float *_dvx_scl  = reinterpret_cast<float *>(_dvx);
        constexpr int VEC_SIZE = sizeof(FLOATV_T)/sizeof(float);
        for (int chan = tidx; chan < nchan_in*VEC_SIZE; chan += WARP_SIZE) {
            atomicAdd(_dkx_scl + chan, scale_dkx * sh_qy_scl[chan]);
        }
        for (int chan = tidx; chan < nchan_out*VEC_SIZE; chan += WARP_SIZE) {
            atomicAdd(_dvx_scl + chan, scale_dvx * sh_dy_scl[chan]);
        }
#else
        for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
            atomicAdd(_dkx + chan, __vscale(scale_dkx, sh_qy[chan]));
        }
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
            atomicAdd(_dvx + chan, __vscale(scale_dvx, sh_dy[chan]));
        }
#endif
    }
}

template<int BDIM_X,
         int BDIM_Y,
         int CHOUT_AS_IN,   // 1 iif "BDIM_X*(NLOC-1) <= nchan_out <= BDIM_X*NLOC" else 0
         int NLOC,          // smallest int such that BDIM_X*NLOC >= nchan_in
         typename FLOATV_T> // either float or float4
__global__
__launch_bounds__(BDIM_X*BDIM_Y)
void s2_attn_bwd_ring_step_pass2_special_vec_k(int nchan_in,
                                               int nchan_out,
                                               int nlat_halo,
                                               int nlon_kx,
                                               int nlon_in,
                                               int pscale,           // GLOBAL pscale = nlon_in / nlon_out_global
                                               int lon_lo_kx,
                                               int lat_halo_start,
                                               int nlat_out,
                                               int nlon_out,
                                               const FLOATV_T *__restrict__ kx,
                                               const FLOATV_T *__restrict__ vx,
                                               const FLOATV_T *__restrict__ qy,
                                               const FLOATV_T *__restrict__ dy,
                                               const int32_t  *__restrict__ row_idx,
                                               const int64_t  *__restrict__ row_off,
                                               const int64_t  *__restrict__ col_idx,
                                               const float    *__restrict__ quad_weights,
                                               const float    *__restrict__ alpha_sum_buf,      // finalized [batch][nlat_out][nlon_out]
                                               const float    *__restrict__ qdotk_max_buf,      // finalized [batch][nlat_out][nlon_out]
                                               const float    *__restrict__ integral_norm_buf,  // finalized, normalized [batch][nlat_out][nlon_out]
                                                     FLOATV_T *__restrict__ dkx,                // [batch][nlat_halo][nlon_kx][nchan_in] (atomically updated)
                                                     FLOATV_T *__restrict__ dvx) {              // [batch][nlat_halo][nlon_kx][nchan_out] (atomically updated)

    static_assert(0 == (BDIM_X & (BDIM_X-1)));
    static_assert(0 == (BDIM_Y & (BDIM_Y-1)));
    static_assert((BDIM_X == 32 && BDIM_Y  > 1) ||
                  (BDIM_X  > 32 && BDIM_Y == 1)) ;

    const int tidx = threadIdx.x;
    
    const int blk_per_row = gridDim.y; // blocks along Y process the same (ho,wo) 
                                       // point by iteration over the (same) CSR 
                                       // row in an interleaved fashion
    const int blk_split_id = blockIdx.y;
    
    //const int batch = blockIdx.y;
    const int batch = blockIdx.z;

    const uint64_t ctaid = uint64_t(blockIdx.x) * blockDim.y + threadIdx.y;
    
    if (ctaid >= uint64_t(nlat_out)*nlon_out) {
        return;
    }

    FLOATV_T loc_qy[NLOC];

    extern __shared__ __align__(sizeof(float4)) float shext[];

    FLOATV_T *sh_dy = reinterpret_cast<FLOATV_T *>(shext) + threadIdx.y*(nchan_in + nchan_out) + tidx; // [nchan_out]

    // used only with __CUDA_ARCH__ < 900
    FLOATV_T *sh_qy = sh_dy + nchan_out; // [nchan_in]

    const int h  = ctaid / nlon_out;
    const int wo = ctaid - (h * nlon_out);
    const int ho = row_idx[h];

    kx  += int64_t(batch)*nlat_halo*nlon_kx*nchan_in  + tidx;
    vx  += int64_t(batch)*nlat_halo*nlon_kx*nchan_out + tidx;
    dkx += int64_t(batch)*nlat_halo*nlon_kx*nchan_in  + tidx;
    dvx += int64_t(batch)*nlat_halo*nlon_kx*nchan_out + tidx;

    qy += int64_t(batch)*nlat_out*nlon_out*nchan_in  + int64_t(ho)*nlon_out*nchan_in  + int64_t(wo)*nchan_in  + tidx;
    dy += int64_t(batch)*nlat_out*nlon_out*nchan_out + int64_t(ho)*nlon_out*nchan_out + int64_t(wo)*nchan_out + tidx;

    const int64_t out_flat     = int64_t(batch)*nlat_out*nlon_out + int64_t(ho)*nlon_out + wo;

    const float alpha_sum      = alpha_sum_buf[out_flat];
    const float qdotk_max      = qdotk_max_buf[out_flat];
    const float integral_norm  = integral_norm_buf[out_flat];
    const float alpha_sum_inv  = 1.0f / alpha_sum;

    strided_op<BDIM_X,               NLOC    >(nchan_in,  [&](int i) {       loc_qy[i] = qy[i*BDIM_X]; });
    strided_op<BDIM_X, CHOUT_AS_IN ? NLOC : 0>(nchan_out, [&](int i) { sh_dy[i*BDIM_X] = dy[i*BDIM_X]; });

#if __CUDA_ARCH__ < 900
    strided_op<BDIM_X,               NLOC    >(nchan_in,  [&](int i) { sh_qy[i*BDIM_X] = loc_qy[i]; });

    if constexpr(std::is_same<FLOATV_T, float4>::value) {
        if constexpr(BDIM_X == 32) {    __syncwarp(); }
        else                       { __syncthreads(); }
    }
#endif

    // `pscale` is the GLOBAL ratio nlon_in / nlon_out_global, passed by the caller.
    // The kernel's `nlon_out` is the LOCAL output width (nlon_out_local), not the global one.
#if 0
    int64_t rbeg = row_off[ho];
    int64_t rend = row_off[ho + 1];
    int rlen = rend - rbeg;

    int ncol_div = rlen / blk_per_row;
    int ncol_mod = rlen - ncol_div*blk_per_row;

    rbeg += blk_split_id*ncol_div + min(blk_split_id, ncol_mod);
    rlen  = ncol_div + (blk_split_id < ncol_mod);

    col_idx += rbeg;

    for (int off = 0; off < rlen; off++) {
#else
    const int64_t rbeg = row_off[ho];
    const int64_t rend = row_off[ho + 1];
    col_idx += rbeg;
    const int rlen = rend - rbeg;

    for (int off = blk_split_id; off < rlen; off += blk_per_row) {
#endif
        const int64_t col   = col_idx[off];
        const int hi_global = col / nlon_in;
        const int wi        = col - (hi_global * nlon_in);
        // wip = (wi + pscale * wo_local) % nlon_in
        //     = (wi_canonical + pscale * (lon_lo_out + wo_local)) % nlon_in
        const int wi_wo     = wi + pscale * wo;
        const int wip       = wi_wo - (wi_wo / nlon_in) * nlon_in;

        if (wip < lon_lo_kx || wip >= lon_lo_kx + nlon_kx) continue;

        const int hi_local  = hi_global - lat_halo_start;
        if (hi_local < 0 || hi_local >= nlat_halo) continue;
        const int wip_local = wip - lon_lo_kx;

        const FLOATV_T *_kx  = kx  + int64_t(hi_local)*nlon_kx*nchan_in  + int64_t(wip_local)*nchan_in;
        const FLOATV_T *_vx  = vx  + int64_t(hi_local)*nlon_kx*nchan_out + int64_t(wip_local)*nchan_out;

        FLOATV_T qdotk_v = __vset<FLOATV_T>(0.0f);
        FLOATV_T gdotv_v = __vset<FLOATV_T>(0.0f);

        strided_op<BDIM_X,               NLOC>    (nchan_in,  [&](int i) { qdotk_v = __vadd(qdotk_v, __vmul(      loc_qy[i], _kx[i*BDIM_X])); });
        strided_op<BDIM_X, CHOUT_AS_IN ? NLOC : 0>(nchan_out, [&](int i) { gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[i*BDIM_X], _vx[i*BDIM_X])); });

        float qdotk = __vred(qdotk_v);
        float gdotv = __vred(gdotv_v);

        if constexpr(BDIM_X == 32) { 
            qdotk = __warp_sum(qdotk); 
            gdotv = __warp_sum(gdotv); 
        } else {
            qdotk = __block_sum<BDIM_X>(qdotk);
            gdotv = __block_sum<BDIM_X>(gdotv);
        }

        const float alpha_inz  = expf(qdotk - qdotk_max) * quad_weights[hi_global];
        const float alpha_mul  = alpha_inz * alpha_sum_inv;
        const float scale_dkx = (gdotv - integral_norm) * alpha_mul;
        const float scale_dvx = alpha_mul;

        FLOATV_T *_dkx = dkx + int64_t(hi_local)*nlon_kx*nchan_in  + int64_t(wip_local)*nchan_in;
        FLOATV_T *_dvx = dvx + int64_t(hi_local)*nlon_kx*nchan_out + int64_t(wip_local)*nchan_out;

#if __CUDA_ARCH__ < 900
        constexpr int VEC_SIZE = sizeof(FLOATV_T)/sizeof(float);

        float *sh_qy_scl = reinterpret_cast<float *>(sh_qy - tidx);
        float *sh_dy_scl = reinterpret_cast<float *>(sh_dy - tidx);

        float *_dkx_scl  = reinterpret_cast<float *>(_dkx - tidx); 
        float *_dvx_scl  = reinterpret_cast<float *>(_dvx - tidx);
        
        for (int chan = tidx; chan < nchan_in*VEC_SIZE; chan += WARP_SIZE) {
            atomicAdd(_dkx_scl + chan, scale_dkx*sh_qy_scl[chan]);
        }
        for (int chan = tidx; chan < nchan_out*VEC_SIZE; chan += BDIM_X) {
            atomicAdd(_dvx_scl + chan, scale_dvx*sh_dy_scl[chan]);
        }
#else
        strided_op<BDIM_X,               NLOC>    (nchan_in,  [&](int i) { atomicAdd(_dkx + i*BDIM_X, __vscale(scale_dkx,       loc_qy[i])); });
        strided_op<BDIM_X, CHOUT_AS_IN ? NLOC : 0>(nchan_out, [&](int i) { atomicAdd(_dvx + i*BDIM_X, __vscale(scale_dvx, sh_dy[i*BDIM_X])); });
#endif
    }
    return;
}

template<typename FLOATV_T>
void launch_gen_attn_ring_pass2_bwd(int64_t batch_size,
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
                                    FLOATV_T *_kxp,
                                    FLOATV_T *_vxp,
                                    FLOATV_T *_qyp,
                                    FLOATV_T *_dyp,
                                    int32_t *_row_idx,
                                    int64_t *_row_off,
                                    int64_t *_col_idx,
                                    float *_quad_weights,
                                    float *_alpha_sum,
                                    float *_qdotk_max,
                                    float *_integral_n,
                                    FLOATV_T *_dkxp,
                                    FLOATV_T *_dvxp,
                                    cudaStream_t stream) {

    dim3 block(WARP_SIZE, THREADS / WARP_SIZE);
    dim3 grid(DIV_UP(nlat_out*nlon_out, block.y), batch_size);

    size_t shsize = sizeof(FLOATV_T)*(nchans_in + nchans_out)*block.y;

    s2_attn_bwd_ring_step_pass2_generic_vec_k<THREADS>
                                             <<<grid, block, shsize, stream>>>(nchans_in, nchans_out, nlat_halo, nlon_kx, nlon_in, 
                                                                               pscale, lon_lo_kx, lat_halo_start, nlat_out, nlon_out,
                                                                               _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, 
                                                                               _col_idx, _quad_weights, _alpha_sum, 
                                                                               _qdotk_max, _integral_n, _dkxp, _dvxp);
    CHECK_ERROR("s2_attn_bwd_ring_step_pass2_generic_vec_k<float>");

    return;
}

__global__ void get_rlen_boundary_k(const int64_t n,
                                    const int32_t *idx,
                                    const int64_t *off,
                                    const float thres,
                                          int64_t *num_lrow_ptr,
                                          int64_t *max_rlen_ptr) {
    const int tid = threadIdx.x;

    int64_t max_rlen = off[idx[0]+1]-off[idx[0]];
    if (!tid) {
        *max_rlen_ptr = max_rlen;
    }
    
    int64_t thres_len = max_rlen * thres;

    for(int64_t i = 0; i < n; i += blockDim.x) {

        int64_t rlen = thres_len;

        if (i+tid < n) {
            int32_t row = idx[i+tid];
            rlen = off[row+1]-off[row];
        }
        
        if (__syncthreads_or(rlen < thres_len)) {
            if (rlen < thres_len) {
                atomicMin((unsigned long long *)num_lrow_ptr, (unsigned long long )i+tid);
            }
            break;
        }
    }
    return;
}   

static void get_crs_rlen_thresh(int64_t nrows,
                                int32_t *row_idx,
                                int64_t *row_off,
                                int64_t *n_long_rows,
                                int64_t *max_row_len) {

    torch::Tensor num_lr = torch::tensor(nrows, torch::kCUDA);
    torch::Tensor max_rl = torch::tensor(int64_t(0), torch::kCUDA);

    int64_t *_num_lr = reinterpret_cast<int64_t *>(num_lr.data_ptr());
    int64_t *_max_rl = reinterpret_cast<int64_t *>(max_rl.data_ptr());

    float thres = PASS2_ROW_LENGTH_THRES;
    
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    get_rlen_boundary_k<<<1, 1024, 0, stream>>>(nrows, row_idx, row_off, thres, _num_lr, _max_rl);

    *n_long_rows = num_lr.item<int64_t>();
    *max_row_len = max_rl.item<int64_t>();

    //printf("Found %ld rows with length less than %f*%ld\n", *n_long_rows, thres, *max_row_len);

    return;
}

template<int BDIM_X,
         int CUR_LOC_SIZE,
         int MAX_LOC_SIZE, // max size of FLOATV_T[] local array
         typename FLOATV_T>
void launch_spc_attn_ring_pass2_bwd(int nloc,
                                    int64_t batch_size,
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
                                    FLOATV_T *_kxp,
                                    FLOATV_T *_vxp,
                                    FLOATV_T *_qyp,
                                    FLOATV_T *_dyp,
                                    int32_t  *_row_idx,
                                    int64_t  *_row_off,
                                    int64_t  *_col_idx,
                                    float    *_quad_weights,
                                    float    *_alpha_sum,
                                    float    *_qdotk_max,
                                    float    *_integral_n,
                                    FLOATV_T *_dkxp,
                                    FLOATV_T *_dvxp,
                                    cudaStream_t stream) {

    static_assert(0 == (BDIM_X & (BDIM_X-1)));
    static_assert(std::is_same<FLOATV_T, float>::value ||
                  std::is_same<FLOATV_T, float4>::value);

    if (CUR_LOC_SIZE == nloc) {

        int64_t n_long_rows;
        int64_t max_row_len;

        // finds the (initial) number of rows with length >= than 0.1
        // of the longest row, i.e. the first; if there are such rows,
        // they are processed with a separate kernel invocation, using 
        // multiple blocks per row, in order to mitigate the imbalance
        // causing long temporal tails in kernel execution.
        get_crs_rlen_thresh(nlat_out, _row_idx, _row_off, &n_long_rows, &max_row_len);

        // nloc determines the size of local arrays used to store
        // temporary buffers loc_k__[], loc_vw_[] and loc_kvw[],
        // of size nchans_in each;
        // if nchans_out is >= BDIM_X*(nloc-1) and <= BDIM_X*nloc
        // then we can use the same compile-time known loops used
        // for input channels, with the exception of testing 
        // whether to execute the last iteration based on "nchans_out"
        // instead of "nchans_in"; in this way as long as the
        // difference between the number of input and output channels
        // is <= BDIM_X we can use the faster path
        const bool chout_as_in = (nchans_out >= BDIM_X*(CUR_LOC_SIZE-1) && 
                                  nchans_out <= BDIM_X* CUR_LOC_SIZE  );

        constexpr int BDIM_Y = (BDIM_X <= 32) ? THREADS/BDIM_X : 1;
        dim3 block(BDIM_X, BDIM_Y);

        // if there are "long rows" use at most
        // 32 blocks for each one (empirically determined) 
        int cta_per_row = min(int64_t(32), DIV_UP(max_row_len, PASS2_MIN_WORK_PER_BLOCK));
#if 0
        const char *val = getenv("NUM_LONG_ROWS");
        if (val) {
            n_long_rows = strtol(val, NULL, 10);
        }
        val = getenv("CTA_PER_ROW");
        if (val) {
            cta_per_row = strtol(val, NULL, 10);
        }
#endif
        dim3 grid_lr(DIV_UP(          n_long_rows *nlon_out, block.y), cta_per_row, batch_size);
        dim3 grid   (DIV_UP((nlat_out-n_long_rows)*nlon_out, block.y),           1, batch_size);

        size_t shsize = sizeof(FLOATV_T)*(nchans_out + nchans_in)*block.y; // 2 arrays per cta, block.y > 1 iif block.x==32
                                                                           // nchans_in arrays used only for __CUDA_ARCH__ < 900

        if (chout_as_in) {
            if (n_long_rows > 0) {
                s2_attn_bwd_ring_step_pass2_special_vec_k<BDIM_X, BDIM_Y, 1, CUR_LOC_SIZE>
                                                      <<<grid_lr, block, shsize, stream>>>(nchans_in, nchans_out, nlat_halo, nlon_kx,
                                                                                        nlon_in, pscale, lon_lo_kx, lat_halo_start,
                                                                                        nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp,
                                                                                        _row_idx, _row_off, _col_idx, _quad_weights,
                                                                                        _alpha_sum, _qdotk_max, _integral_n,
                                                                                        _dkxp, _dvxp);
            }
            s2_attn_bwd_ring_step_pass2_special_vec_k<BDIM_X, BDIM_Y, 1, CUR_LOC_SIZE>
                                                  <<<grid, block, shsize, stream>>>(nchans_in, nchans_out, nlat_halo, nlon_kx,
                                                                                    nlon_in, pscale, lon_lo_kx, lat_halo_start,
                                                                                    nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp,
                                                                                    _row_idx+n_long_rows/**/, _row_off, _col_idx, _quad_weights,
                                                                                    _alpha_sum, _qdotk_max, _integral_n,
                                                                                    _dkxp, _dvxp);
      } else {
            if (n_long_rows > 0) {
                s2_attn_bwd_ring_step_pass2_special_vec_k<BDIM_X, BDIM_Y, 0, CUR_LOC_SIZE>
                                                      <<<grid_lr, block, shsize, stream>>>(nchans_in, nchans_out, nlat_halo, nlon_kx,
                                                                                        nlon_in, pscale, lon_lo_kx, lat_halo_start,
                                                                                        nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp,
                                                                                        _row_idx, _row_off, _col_idx, _quad_weights,
                                                                                        _alpha_sum, _qdotk_max, _integral_n,
                                                                                        _dkxp, _dvxp);
            }
            s2_attn_bwd_ring_step_pass2_special_vec_k<BDIM_X, BDIM_Y, 0, CUR_LOC_SIZE>
                                                  <<<grid, block, shsize, stream>>>(nchans_in, nchans_out, nlat_halo, nlon_kx,
                                                                                    nlon_in, pscale, lon_lo_kx, lat_halo_start,
                                                                                    nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp,
                                                                                    _row_idx+n_long_rows/**/, _row_off, _col_idx, _quad_weights,
                                                                                    _alpha_sum, _qdotk_max, _integral_n,
                                                                                    _dkxp, _dvxp);
        }
        CHECK_ERROR("s2_attn_bwd_special_vec_k");

        return;
    }
    if constexpr(CUR_LOC_SIZE < MAX_LOC_SIZE) {
        launch_spc_attn_ring_pass2_bwd<BDIM_X,
                                       CUR_LOC_SIZE+1,
                                       MAX_LOC_SIZE>(nloc, batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo,
                                                     nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out,
                                                     _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights,
                                                     _alpha_sum, _qdotk_max, _integral_n,
                                                     _dkxp, _dvxp, stream);
    }
    return;
}

static void s2_attn_bwd_ring_step_pass2_dispatch(int64_t batch_size,
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
                                                 at::Tensor dyP,
                                                 at::Tensor row_idx,
                                                 at::Tensor row_off,
                                                 at::Tensor col_idx,
                                                 at::Tensor quad_weights,
                                                 at::Tensor alpha_sum_buf,
                                                 at::Tensor qdotk_max_buf,
                                                 at::Tensor integral_norm_buf,
                                                 at::Tensor dkxP,
                                                 at::Tensor dvxP) {

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // smallest power of two "bdimx" (>=32) s.t. bdimx*MAX_LOCAL_ARR_LEN >= nchans_in
    int bdimx;
    bdimx = DIV_UP(nchans_in, MAX_LOCAL_ARR_LEN);
    bdimx = max(bdimx, WARP_SIZE);
    bdimx = next_pow2(bdimx);

    float *_kxp          = reinterpret_cast<float *>(kxP.data_ptr());
    float *_vxp          = reinterpret_cast<float *>(vxP.data_ptr());
    float *_qyp          = reinterpret_cast<float *>(qyP.data_ptr());
    float *_dyp          = reinterpret_cast<float *>(dyP.data_ptr());
    int32_t *_row_idx    = reinterpret_cast<int32_t *>(row_idx.data_ptr());
    int64_t *_row_off    = reinterpret_cast<int64_t *>(row_off.data_ptr());
    int64_t *_col_idx    = reinterpret_cast<int64_t *>(col_idx.data_ptr());
    float *_quad_weights = reinterpret_cast<float *>(quad_weights.data_ptr());
    float *_alpha_sum    = reinterpret_cast<float *>(alpha_sum_buf.data_ptr());
    float *_qdotk_max    = reinterpret_cast<float *>(qdotk_max_buf.data_ptr());
    float *_integral_n   = reinterpret_cast<float *>(integral_norm_buf.data_ptr());
    float *_dkxp         = reinterpret_cast<float *>(dkxP.data_ptr());
    float *_dvxp         = reinterpret_cast<float *>(dvxP.data_ptr());

    constexpr int VEC_SIZE = sizeof(float4) / sizeof(float);

    if (!is_aligned<sizeof(float4)>(_kxp) ||
        !is_aligned<sizeof(float4)>(_vxp) ||
        !is_aligned<sizeof(float4)>(_qyp) ||
        !is_aligned<sizeof(float4)>(_dyp) ||
        (nchans_in  % VEC_SIZE) != 0 ||
        (nchans_out % VEC_SIZE) != 0) {

        const int nloc = DIV_UP(nchans_in, bdimx);

        constexpr int MIN_LOC_ARR_LEN = MAX_LOCAL_ARR_LEN/2+1;

        switch(bdimx) {
            case   32: launch_spc_attn_ring_pass2_bwd< 32,               1, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _alpha_sum, _qdotk_max, _integral_n, _dkxp, _dvxp, stream); break;
            case   64: launch_spc_attn_ring_pass2_bwd< 64, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _alpha_sum, _qdotk_max, _integral_n, _dkxp, _dvxp, stream); break;
            case  128: launch_spc_attn_ring_pass2_bwd<128, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _alpha_sum, _qdotk_max, _integral_n, _dkxp, _dvxp, stream); break;
            case  256: launch_spc_attn_ring_pass2_bwd<256, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _alpha_sum, _qdotk_max, _integral_n, _dkxp, _dvxp, stream); break;
            case  512: launch_spc_attn_ring_pass2_bwd<512, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _alpha_sum, _qdotk_max, _integral_n, _dkxp, _dvxp, stream); break;
            default:   launch_gen_attn_ring_pass2_bwd                                         (      batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out, _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights, _alpha_sum, _qdotk_max, _integral_n, _dkxp, _dvxp, stream); break;
        }

    } else {

        float4 *_kxp4  = reinterpret_cast<float4 *>(_kxp);
        float4 *_vxp4  = reinterpret_cast<float4 *>(_vxp);
        float4 *_qyp4  = reinterpret_cast<float4 *>(_qyp);
        float4 *_dyp4  = reinterpret_cast<float4 *>(_dyp);
        float4 *_dkxp4 = reinterpret_cast<float4 *>(dkxP.data_ptr());
        float4 *_dvxp4 = reinterpret_cast<float4 *>(dvxP.data_ptr());

        nchans_in  /= VEC_SIZE;
        nchans_out /= VEC_SIZE;

        const int nloc = DIV_UP(nchans_in, bdimx);

        constexpr int MAX_LOCAL_VEC_LEN = MAX_LOCAL_ARR_LEN / VEC_SIZE;
        constexpr int MIN_LOC_VEC_LEN = MAX_LOCAL_VEC_LEN/2+1;

        switch(bdimx) {
            case   32: launch_spc_attn_ring_pass2_bwd< 32,               1, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, _dyp4, _row_idx, _row_off, _col_idx, _quad_weights, _alpha_sum, _qdotk_max, _integral_n, _dkxp4, _dvxp4, stream); break;
            case   64: launch_spc_attn_ring_pass2_bwd< 64, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, _dyp4, _row_idx, _row_off, _col_idx, _quad_weights, _alpha_sum, _qdotk_max, _integral_n, _dkxp4, _dvxp4, stream); break;
            case  128: launch_spc_attn_ring_pass2_bwd<128, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, _dyp4, _row_idx, _row_off, _col_idx, _quad_weights, _alpha_sum, _qdotk_max, _integral_n, _dkxp4, _dvxp4, stream); break;
            case  256: launch_spc_attn_ring_pass2_bwd<256, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, _dyp4, _row_idx, _row_off, _col_idx, _quad_weights, _alpha_sum, _qdotk_max, _integral_n, _dkxp4, _dvxp4, stream); break;
            case  512: launch_spc_attn_ring_pass2_bwd<512, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, _dyp4, _row_idx, _row_off, _col_idx, _quad_weights, _alpha_sum, _qdotk_max, _integral_n, _dkxp4, _dvxp4, stream); break;
            default:   launch_gen_attn_ring_pass2_bwd                                         (      batch_size, nchans_in, nchans_out, nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, _dyp4, _row_idx, _row_off, _col_idx, _quad_weights, _alpha_sum, _qdotk_max, _integral_n, _dkxp4, _dvxp4, stream); break;
        }
    }
}

void s2_attention_bwd_ring_step_pass1_cuda(
    at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor dy,
    at::Tensor alpha_sum_buf, at::Tensor qdotk_max_buf, at::Tensor integral_buf,
    at::Tensor alpha_k_buf, at::Tensor alpha_kvw_buf,
    at::Tensor quad_weights, at::Tensor psi_col_idx, at::Tensor psi_row_off, at::Tensor psi_row_idx,
    int64_t nlon_in, int64_t pscale, int64_t lon_lo_kx, int64_t lat_halo_start,
    int64_t nlat_out, int64_t nlon_out)
{
    CHECK_CUDA_INPUT_TENSOR(kx); CHECK_CUDA_INPUT_TENSOR(vx);
    CHECK_CUDA_INPUT_TENSOR(qy); CHECK_CUDA_INPUT_TENSOR(dy);
    CHECK_CUDA_TENSOR(alpha_sum_buf); CHECK_CUDA_TENSOR(qdotk_max_buf);
    CHECK_CUDA_TENSOR(integral_buf);  CHECK_CUDA_TENSOR(alpha_k_buf);
    CHECK_CUDA_TENSOR(alpha_kvw_buf); CHECK_CUDA_TENSOR(quad_weights);
    CHECK_CUDA_TENSOR(psi_col_idx);   CHECK_CUDA_TENSOR(psi_row_off); CHECK_CUDA_TENSOR(psi_row_idx);

    const int batch_size = kx.size(0);
    const int nlat_halo  = kx.size(2);   // kx is [B,C,H,W], H is dim 2
    const int nlon_kx    = kx.size(3);   // W is dim 3
    const size_t nchans_in  = qy.size(1);
    const size_t nchans_out = vx.size(1);

    torch::Tensor kxP = kx.to(torch::kFloat32);
    torch::Tensor vxP = vx.to(torch::kFloat32);
    torch::Tensor qyP = qy.to(torch::kFloat32);
    torch::Tensor dyP = dy.to(torch::kFloat32);

    if (kxP.strides()[1] != 1) { kxP = permute_4D_to0231(kxP); }
    if (vxP.strides()[1] != 1) { vxP = permute_4D_to0231(vxP); }
    if (qyP.strides()[1] != 1) { qyP = permute_4D_to0231(qyP); }
    if (dyP.strides()[1] != 1) { dyP = permute_4D_to0231(dyP); }

    s2_attn_bwd_ring_step_pass1_dispatch(
        batch_size, nchans_in, nchans_out,
        nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start,
        nlat_out, nlon_out,
        kxP, vxP, qyP, dyP,
        psi_row_idx, psi_row_off, psi_col_idx, quad_weights,
        alpha_sum_buf, qdotk_max_buf, integral_buf, alpha_k_buf, alpha_kvw_buf);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void s2_attention_bwd_ring_step_pass2_cuda(
    at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor dy,
    at::Tensor alpha_sum_buf, at::Tensor qdotk_max_buf, at::Tensor integral_norm_buf,
    at::Tensor dkx, at::Tensor dvx,
    at::Tensor quad_weights, at::Tensor psi_col_idx, at::Tensor psi_row_off, at::Tensor psi_row_idx,
    int64_t nlon_in, int64_t pscale, int64_t lon_lo_kx, int64_t lat_halo_start,
    int64_t nlat_out, int64_t nlon_out)
{
    CHECK_CUDA_INPUT_TENSOR(kx); CHECK_CUDA_INPUT_TENSOR(vx);
    CHECK_CUDA_INPUT_TENSOR(qy); CHECK_CUDA_INPUT_TENSOR(dy);
    CHECK_CUDA_TENSOR(alpha_sum_buf); CHECK_CUDA_TENSOR(qdotk_max_buf);
    CHECK_CUDA_TENSOR(integral_norm_buf);
    CHECK_CUDA_TENSOR(dkx); CHECK_CUDA_TENSOR(dvx);
    CHECK_CUDA_TENSOR(quad_weights);
    CHECK_CUDA_TENSOR(psi_col_idx); CHECK_CUDA_TENSOR(psi_row_off); CHECK_CUDA_TENSOR(psi_row_idx);

    const int batch_size = kx.size(0);
    const int nlat_halo  = kx.size(2);   // kx is [B,C,H,W], H is dim 2
    const int nlon_kx    = kx.size(3);   // W is dim 3
    const size_t nchans_in  = qy.size(1);
    const size_t nchans_out = vx.size(1);

    torch::Tensor kxP = kx.to(torch::kFloat32);
    torch::Tensor vxP = vx.to(torch::kFloat32);
    torch::Tensor qyP = qy.to(torch::kFloat32);
    torch::Tensor dyP = dy.to(torch::kFloat32);

    if (kxP.strides()[1] != 1) { kxP = permute_4D_to0231(kxP); }
    if (vxP.strides()[1] != 1) { vxP = permute_4D_to0231(vxP); }
    if (qyP.strides()[1] != 1) { qyP = permute_4D_to0231(qyP); }
    if (dyP.strides()[1] != 1) { dyP = permute_4D_to0231(dyP); }
#if 0
    dump_csr_linear("csr_attn_distr", pscale, nlon_in, lon_lo_kx, nlon_kx, lat_halo_start, nlat_halo, nlat_out, psi_row_idx, psi_row_off, psi_col_idx);
#endif
    // dkx/dvx are already in channels-last format (allocated that way in Python)
    s2_attn_bwd_ring_step_pass2_dispatch(
        batch_size, nchans_in, nchans_out,
        nlon_in, pscale, nlat_halo, nlon_kx, lon_lo_kx, lat_halo_start,
        nlat_out, nlon_out,
        kxP, vxP, qyP, dyP,
        psi_row_idx, psi_row_off, psi_col_idx, quad_weights,
        alpha_sum_buf, qdotk_max_buf, integral_norm_buf,
        dkx, dvx);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY_IMPL(attention_kernels, CUDA, m)
{
    m.impl("backward_ring_step_pass1", &s2_attention_bwd_ring_step_pass1_cuda);
    m.impl("backward_ring_step_pass2", &s2_attention_bwd_ring_step_pass2_cuda);
}

int is_in_lat_range(int64_t col,  
                    //int64_t pscale,
                    int64_t nlon_in,
                    //int64_t lon_lo_kx,
                    //int nlon_kx,
                    int64_t lat_halo_start,
                    int nlat_halo) {

    const int hi_global = col / nlon_in;

    const int hi_local  = hi_global - lat_halo_start;

    if (hi_local < 0 || hi_local >= nlat_halo) return 0;

    return 1;

}

void dump_csr_linear(const char *fname,
                     int64_t pscale,
                     int64_t nlon_in,
                     int64_t lon_lo_kx,
                     int nlon_kx,
                     int64_t lat_halo_start,
                     int nlat_halo,
                     int64_t nrows,
                     at::Tensor row_idx,
                     at::Tensor row_off,
                     at::Tensor col_idx) {

        int64_t nnz = col_idx.size(0);

        int32_t *row_idx_h = new int32_t[nrows];
        int64_t *row_off_h = new int64_t[nrows+1];
        int64_t *col_idx_h = new int64_t[nnz];

        int32_t *row_idx_d = row_idx.data_ptr<int32_t>();
        int64_t *row_off_d = row_off.data_ptr<int64_t>();
        int64_t *col_idx_d = col_idx.data_ptr<int64_t>();

        CHECK_CUDA(cudaMemcpy(row_idx_h, row_idx_d, sizeof(*row_idx_h)*nrows    , cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(row_off_h, row_off_d, sizeof(*row_off_h)*(nrows+1), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(col_idx_h, col_idx_d, sizeof(*col_idx_h)*nnz      , cudaMemcpyDeviceToHost));

        printf("Writing data to file...");

        static int count = 0;

        char file_name[256];
        snprintf(file_name, sizeof(file_name), "%s_%d.txt", fname, count);
        count++;

        FILE *fp = fopen(file_name, "w");
        if (!fp) {
                fprintf(stderr, "Cannot open file %s for writing!\n", fname);
                exit(EXIT_FAILURE);
        }
        
        fprintf(fp, "nrows: %ld, row_idx.size(0): %ld, row_off.size(0): %ld, col_idx.size(0): %ld\n",
                nrows, row_idx.size(0), row_off.size(0), col_idx.size(0));

        fprintf(fp, "pscale: %ld, nlon_in: %ld, lon_lo_kx: %ld, nlon_kx: %d, lat_halo_start: %ld, nlat_halo: %d\n",
                pscale, nlon_in, lon_lo_kx, nlon_kx, lat_halo_start, nlat_halo);

        fprintf(fp, "CSR:\n");

        for(int64_t i = 0; i < nrows; i++) {

                int32_t r = row_idx_h[i];
                
                int n_internal = 0;
                for(int64_t o = row_off_h[r]; o < row_off_h[r+1]; o++) {
                        n_internal += is_in_lat_range(col_idx_h[o], nlon_in, lat_halo_start, nlat_halo);
                }

                fprintf(fp, "%6ld, row: %6d, len: %6ld, in H rng: %d - ", i, r, row_off_h[r+1]-row_off_h[r], n_internal);

                for(int64_t o = row_off_h[r]; o < row_off_h[r+1]; o++) {
                        fprintf(fp, "%10ld", col_idx_h[o]);
                }
                fprintf(fp, "\n");
        }
        fclose(fp);
        printf("done\n");

        delete [] row_idx_h;
        delete [] row_off_h;
        delete [] col_idx_h;
}
}
