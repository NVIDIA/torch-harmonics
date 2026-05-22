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

template<int BDIM_X,
         int BDIM_Y,
         int NLOC,          // smallest int such that BDIM_X*NLOC >= nchan_in
         typename FLOATV_T> // either float or float4
__global__
__launch_bounds__(BDIM_X*BDIM_Y)
void s2_attn_fwd_ring_softmax_k(const __grid_constant__ attn_params_t p,
                                const int shcol_len_max,
                                const int nlat_max,
                                const FLOATV_T *__restrict__ kx,           // [batch][nlat_halo][nlon_kx][nchan_in]
                                const FLOATV_T *__restrict__ qy,           // [batch][nlat_out][nlon_out][nchan_in]
                                const int32_t  *__restrict__ row_idx,
                                const int64_t  *__restrict__ row_off,
                                const int64_t  *__restrict__ col_idx,
                                const float    *__restrict__ qdotk_max_prev,
                                      float    *__restrict__ qdotk_max_curr) {  // NEW, caller inits to -inf each ring step

    static_assert(0 == (BDIM_X & (BDIM_X-1)));
    static_assert(0 == (BDIM_Y & (BDIM_Y-1)));
    static_assert((BDIM_X == WARP_SIZE && BDIM_Y  > 1) ||
                  (BDIM_X  > WARP_SIZE && BDIM_Y == 1)) ;

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int blk_per_row = gridDim.y; // blocks along Y process the same (ho,wo)
                                       // point by iteration over the (same) CSR
                                       // row in an interleaved fashion
    const int blk_split_id = blockIdx.y;

    const int batch = blockIdx.z;
    const uint64_t ctaid = uint64_t(blockIdx.x) * blockDim.y + threadIdx.y;

    const int &nchan_in       = p.nchan_in;
    const int &nlat_halo      = p.nlat_halo;
    const int &nlon_kx        = p.nlon_kx;
    const int &nlon_in        = p.nlon_in;
    const int &pscale         = p.pscale;
    const int &lon_lo_kx      = p.lon_lo_kx;
    const int &lat_halo_start = p.lat_halo_start;
    const int &nlat_out       = p.nlat_out;
    const int &nlon_out       = p.nlon_out;

    if (ctaid >= uint64_t(nlat_max)*nlon_out) {
        return;
    }

    const int h  = ctaid / nlon_out;
    const int wo = ctaid - (h * nlon_out);
    const int ho = row_idx[h];

    kx += int64_t(batch)*nlat_halo*nlon_kx*nchan_in;
    qy += int64_t(batch)*nlat_out*nlon_out*nchan_in + int64_t(ho)*nlon_out*nchan_in + int64_t(wo)*nchan_in + tidx;

    extern __shared__ __align__(sizeof(float4)) float shext[];

    // just to simplify the seatup of the shared memory layout
    using FLOATV_PTR_T = const FLOATV_T *;

    FLOATV_T     *shqy     = NULL;
    FLOATV_PTR_T *shkx_ptr = NULL;
#if 1
    if constexpr(sizeof(FLOATV_T) > sizeof(FLOATV_PTR_T)) {
        FLOATV_T *base = reinterpret_cast<FLOATV_T *>(shext);
        shqy  = base + tidy*nchan_in;
        shkx_ptr = reinterpret_cast<FLOATV_PTR_T *>(base + BDIM_Y*nchan_in) + tidy*shcol_len_max;
    } else {
      FLOATV_PTR_T *base = reinterpret_cast<FLOATV_PTR_T *>(shext);
      shkx_ptr = base + tidy*shcol_len_max;
        shqy  = reinterpret_cast<FLOATV_T *>(base + BDIM_Y*shcol_len_max) + tidy*nchan_in;
    }
    shqy += tidx;
#else
    FLOATV_PTR_T *base = reinterpret_cast<FLOATV_PTR_T *>(shext);
    shkx_ptr = base + tidy*shcol_len_max;
#endif

#if 1
    strided_op<BDIM_X, 0>(nchan_in, [&](int i) { shqy[i*BDIM_X] = qy[i*BDIM_X]; });
#else
    FLOATV_T loc_qy[NLOC];
    strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { loc_qy[i] = qy[i*BDIM_X]; });
#endif
    const int64_t out_flat = int64_t(batch)*nlat_out*nlon_out + int64_t(ho)*nlon_out + wo;

    qdotk_max_prev += out_flat;
    qdotk_max_curr += out_flat;

    float qdotk_max = qdotk_max_prev[0];

    const int64_t rbeg = row_off[ho];
    const int64_t rend = row_off[ho + 1];
    const int     rlen = rend - rbeg;

    col_idx += rbeg + blk_split_id;

    const int rlen_div = rlen / blk_per_row;
    const int rlen_mod = rlen % blk_per_row;

    int n = rlen_div + (blk_split_id < rlen_mod);
    int n_active = 0;

    for (int i = 0; i < n; i += BDIM_X) {

        const FLOATV_T *kx_ptr = NULL;

        if (i+tidx < n) {
            const int64_t col = col_idx[(i+tidx)*blk_per_row];

            const int hi_global = col / nlon_in;
            const int wi        = col - (hi_global * nlon_in);
            const int wi_wo     = wi + pscale * wo;

            const int wip = wi_wo - (wi_wo / nlon_in) * nlon_in;

            if (wip >= lon_lo_kx && wip < lon_lo_kx + nlon_kx) {

                const int hi_local  = hi_global - lat_halo_start;

                if (hi_local >= 0 && hi_local < nlat_halo) {

                    const int wip_local = wip - lon_lo_kx;
                    kx_ptr = kx + int64_t(hi_local)*nlon_kx*nchan_in + int64_t(wip_local)*nchan_in;
                }
            }
        }

        int toff;
        int ntot = __compact<BDIM_X, BDIM_Y>(kx_ptr != NULL, &toff);
        if (kx_ptr != NULL) {
            shkx_ptr[n_active + toff] = kx_ptr;
        }
        n_active += ntot;
    }
    n = n_active;

    __gsync<BDIM_X>();

    for (int i = 0; i < n; i++) {

        const FLOATV_T *__restrict__ _kx = shkx_ptr[i] + tidx;

        FLOATV_T qdotk_v = __vset<FLOATV_T>(0.0f);
#if 1
        strided_op<BDIM_X, 0>(nchan_in, [&](int i) { qdotk_v = __vadd(qdotk_v, __vmul(shqy[i*BDIM_X], _kx[i*BDIM_X])); });
#else
        strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { qdotk_v = __vadd(qdotk_v, __vmul(loc_qy[i], _kx[i*BDIM_X])); });
#endif
        float qdotk = __vred(qdotk_v);

        if constexpr(BDIM_X == WARP_SIZE) { qdotk = __warp_sum(qdotk);          }
        else                              { qdotk = __block_sum<BDIM_X>(qdotk); }

        qdotk_max = max(qdotk_max, qdotk);
    }

    if (!tidx) {
        atomicMax(qdotk_max_curr, qdotk_max);
    }
    return;
}

template<int BDIM_X,
         int BDIM_Y,
         int NLOC,          // smallest int such that BDIM_X*NLOC >= nchan_in
         typename FLOATV_T> // either float or float4
__global__
__launch_bounds__(BDIM_X*BDIM_Y)
void s2_attn_fwd_ring_rescale_k(const __grid_constant__ attn_params_t p,
                                const int nlat_max,
                                const int32_t  *__restrict__ row_idx,
                                      float    *__restrict__ alpha_sum_buf,  // [batch][nlat_out][nlon_out] (in/out)
                                      float    *__restrict__ qdotk_max_prev, // [batch][nlat_out][nlon_out] (in/out)
                                      float    *__restrict__ qdotk_max_curr, // [batch][nlat_out][nlon_out] (in/out)
                                      FLOATV_T *__restrict__ y_acc) {        // [batch][nlat_out][nlon_out][nchan_in] (in/out)

    static_assert(0 == (BDIM_X & (BDIM_X-1)));
    static_assert(0 == (BDIM_Y & (BDIM_Y-1)));
    static_assert((BDIM_X == WARP_SIZE && BDIM_Y  > 1) ||
                  (BDIM_X  > WARP_SIZE && BDIM_Y == 1)) ;

    const int tidx = threadIdx.x;
    const int batch = blockIdx.y;
    const uint64_t ctaid = uint64_t(blockIdx.x) * blockDim.y + threadIdx.y;

    const int &nchan_out = p.nchan_out;
    const int &nlat_out = p.nlat_out;
    const int &nlon_out = p.nlon_out;

    if (ctaid >= uint64_t(nlat_max)*nlon_out) {
        return;
    }

    const int h  = ctaid / nlon_out;
    const int wo = ctaid - (h * nlon_out);
    const int ho = row_idx[h];

    const int64_t out_flat = int64_t(batch)*nlat_out*nlon_out + int64_t(ho)*nlon_out + wo;

    qdotk_max_prev += out_flat;
    qdotk_max_curr += out_flat;
    alpha_sum_buf  += out_flat;

    y_acc += out_flat*nchan_out + tidx;

    const float qdotk_max_old = qdotk_max_prev[0];
    const float qdotk_max_new = qdotk_max_curr[0];

    if (qdotk_max_old == qdotk_max_new) {
        return;
    }

    const float max_correction = expf(qdotk_max_old - qdotk_max_new);

    const float alpha_sum = max_correction*alpha_sum_buf[0];

    FLOATV_T locy[NLOC];

    strided_op<BDIM_X, NLOC>(nchan_out, [&](int i) { locy[i] = y_acc[i*BDIM_X]; });
    strided_op<BDIM_X, NLOC>(nchan_out, [&](int i) { locy[i] = __vscale(max_correction, locy[i]); });
    strided_op<BDIM_X, NLOC>(nchan_out, [&](int i) { y_acc[i*BDIM_X] = locy[i]; });

    if (!tidx) {
        qdotk_max_prev[0] = qdotk_max_new; // used for next ring call
        alpha_sum_buf[0] = alpha_sum;
    }

    return;
}

template<int BDIM_X,
         int BDIM_Y,
         int CHIN_AS_OUT,   // 1 iif "BDIM_X*(NLOC-1) <= nchan_out <= BDIM_X*NLOC" else 0
         int NLOC,          // smallest int such that BDIM_X*NLOC >= nchan_in
         typename FLOATV_T> // either float or float4
__global__
__launch_bounds__(BDIM_X*BDIM_Y)
void s2_attn_fwd_ring_finalize_k(const __grid_constant__ attn_params_t p,
                                 const int shcol_len_max,
                                 const int nlat_max,
                                 const FLOATV_T *__restrict__ kx,           // [batch][nlat_halo][nlon_kx][nchan_in]
                                 const FLOATV_T *__restrict__ vx,           // [batch][nlat_halo][nlon_kx][nchan_out]
                                 const FLOATV_T *__restrict__ qy,           // [batch][nlat_out][nlon_out][nchan_in]
                                 const int32_t  *__restrict__ row_idx,
                                 const int64_t  *__restrict__ row_off,
                                 const int64_t  *__restrict__ col_idx,
                                 const float    *__restrict__ quad_weights,
                                       float    *__restrict__ alpha_sum_buf,      // [batch][nlat_out][nlon_out] (in/out)
                                       float    *__restrict__ qdotk_max_curr,     // [batch][nlat_out][nlon_out] (in/out)
                                       FLOATV_T *__restrict__ y_acc) {            // [batch][nlat_out][nlon_out][nchan_out] (in/out)

    static_assert(0 == (BDIM_X & (BDIM_X-1)));
    static_assert(0 == (BDIM_Y & (BDIM_Y-1)));
    static_assert((BDIM_X == WARP_SIZE && BDIM_Y  > 1) ||
                  (BDIM_X  > WARP_SIZE && BDIM_Y == 1)) ;

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int blk_per_row = gridDim.y; // blocks along Y process the same (ho,wo)
                                       // point by iteration over the (same) CSR
                                       // row in an interleaved fashion
    const int blk_split_id = blockIdx.y;

    const int batch = blockIdx.z;
    const uint64_t ctaid = uint64_t(blockIdx.x)*blockDim.y + threadIdx.y;

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

    if (ctaid >= uint64_t(nlat_max)*nlon_out) {
        return;
    }

    extern __shared__ __align__(sizeof(float4)) float shext[];

    //FLOATV_T *sh_dy = reinterpret_cast<FLOATV_T *>(shext) + threadIdx.y*(nchan_in+nchan_out) + tidx;
    //FLOATV_T *sh_qy = sh_dy + nchan_out; // [nchan_in], so always offest by tidx

    // just to simplify the seatup of the shared memory layout
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

    FLOATV_T     *shq      = base_fltv                            + tidy*nchan_in;       // [nchan_in]
    FLOATV_PTR_T *shkx_ptr = base_fltv_ptr                        + tidy*shcol_len_max;  // [shcol_len_max]
    FLOATV_PTR_T *shvx_ptr = base_fltv_ptr + BDIM_Y*shcol_len_max + tidy*shcol_len_max;  // [shcol_len_max]
    float        *shweight = base_flt                             + tidy*shcol_len_max;  // [shcol_len_max]

    shq += tidx;


    const int h  = ctaid / nlon_out;
    const int wo = ctaid - (h * nlon_out);
    const int ho = row_idx[h];

    kx += int64_t(batch)*nlat_halo*nlon_kx*nchan_in; //  + tidx;
    vx += int64_t(batch)*nlat_halo*nlon_kx*nchan_out;//  + tidx;

    qy += int64_t(batch)*nlat_out*nlon_out*nchan_in  + int64_t(ho)*nlon_out*nchan_in  + int64_t(wo)*nchan_in  + tidx;

    const int64_t out_flat = int64_t(batch)*nlat_out*nlon_out + int64_t(ho)*nlon_out + wo;

    y_acc          += out_flat*nchan_out + tidx;
    alpha_sum_buf  += out_flat;
    qdotk_max_curr += out_flat;

    // Load current state
    float alpha_sum = 0;
    float qdotk_max_new = qdotk_max_curr[0];
    
    FLOATV_T locy[NLOC];
    strided_op<BDIM_X, NLOC>(nchan_out, [&](int i) { locy[i] = __vset<FLOATV_T>(0); });
    
    strided_op<BDIM_X, CHIN_AS_OUT ? NLOC : 0>(nchan_in,  [&](int i) { shq[i*BDIM_X] =    qy[i*BDIM_X]; });
    
    // `pscale` is the GLOBAL ratio nlon_in / nlon_out_global, passed by the caller.
    // Computing it here as `nlon_in / nlon_out` would be wrong because the kernel's
    // `nlon_out` is the LOCAL output width (nlon_out_local), not the global one.

    const int64_t rbeg = row_off[ho];
    const int64_t rend = row_off[ho + 1];
    const int     rlen = rend - rbeg;

    col_idx += rbeg + blk_split_id;

    const int rlen_div = rlen / blk_per_row;
    const int rlen_mod = rlen % blk_per_row;

    int n = rlen_div + (blk_split_id < rlen_mod);
    int n_active = 0;

    for (int i = 0; i < n; i += BDIM_X) {

        const FLOATV_T *kx_ptr = NULL;
        const FLOATV_T *vx_ptr = NULL;
        float weight = 0;

        if (i+tidx < n) {
            const int64_t col = col_idx[(i+tidx)*blk_per_row];

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
    n = n_active;

    __gsync<BDIM_X>();

    for (int i = 0; i < n; i++) {

        const FLOATV_T *_kx = shkx_ptr[i] + tidx;
        const FLOATV_T *_vx = shvx_ptr[i] + tidx;

        FLOATV_T qdotkv = __vset<FLOATV_T>(0.f);
        strided_op<BDIM_X, CHIN_AS_OUT ? NLOC : 0>(nchan_in, [&](int i) { qdotkv = __vadd(qdotkv, __vmul(shq[i*BDIM_X], _kx[i*BDIM_X])); });

        float qdotk = __vred(qdotkv);
        if constexpr(BDIM_X == WARP_SIZE) { qdotk =          __warp_sum(qdotk); }
        else                              { qdotk = __block_sum<BDIM_X>(qdotk); }

        const float alpha  = expf(qdotk - qdotk_max_new) * shweight[i]; //quad_weights[hi_global];

        alpha_sum += alpha;

        strided_op<BDIM_X, NLOC>(nchan_out, [&](int i) { locy[i] = __vadd(locy[i], __vscale(alpha, _vx[i*BDIM_X])); });
    }

    // Store updated state
    if (!tidx) {
        atomicAdd(alpha_sum_buf, alpha_sum);
        //qdotk_max_buf[0] = qdotk_max; // no need to store, after kernel qdotk_max_curr will be copied into qdotk_max_prev
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
    constexpr bool DO_SPLIT_VEC = sizeof(FLOATV_T) / sizeof(float) == 4;
#else
    constexpr bool DO_SPLIT_VEC = false;
#endif
    if constexpr(DO_SPLIT_VEC) {
        strided_op<BDIM_X, NLOC>(nchan_out, [&](int i) { atomicAdd(&y_acc[i*BDIM_X].x, locy[i].x);
                                                         atomicAdd(&y_acc[i*BDIM_X].y, locy[i].y);
                                                         atomicAdd(&y_acc[i*BDIM_X].z, locy[i].z);
                                                         atomicAdd(&y_acc[i*BDIM_X].w, locy[i].w); });
    } else {
        strided_op<BDIM_X, NLOC>(nchan_out, [&](int i) { atomicAdd(y_acc + i*BDIM_X, locy[i]); });
    }
    return;
}

template<int BDIM_X,
         int LOC_SIZE,
         typename FLOATV_T>
void spc_attn_ring_fwd_long_rows(attn_params_t params,
                                 int64_t n_long_rows,
                                 int64_t max_row_len,
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

    if (!n_long_rows) {
        return;
    }

    const int nlat_out = params.nlat_out;
    const int nlon_out = params.nlon_out;
    const int nchans_in  = params.nchan_in;
    const int nchans_out = params.nchan_out;

    const bool chin_as_out = (nchans_in >= BDIM_X*(LOC_SIZE-1) && 
                              nchans_in <= BDIM_X* LOC_SIZE  );

    constexpr int BDIM_Y = (BDIM_X <= WARP_SIZE) ? THREADS/BDIM_X : 1;

    dim3 block(BDIM_X, BDIM_Y);

    // temporary, for correctness only
    torch::Tensor t_qdotk_max_new = torch::from_blob(_qdotk_max,
                                                     {batch_size*nlat_out*nlon_out},
                                                     torch::TensorOptions().dtype(torch::kFloat32)
                                                                           .device(torch::kCUDA)).clone();

    float *_qdotk_max_new = reinterpret_cast<float *>(t_qdotk_max_new.data_ptr());

    const int cta_per_row = min(int64_t(32), DIV_UP(max_row_len, PASS2_MIN_WORK_PER_BLOCK));

    dim3 grid_lr  (DIV_UP(n_long_rows*nlon_out, block.y), cta_per_row, batch_size);
    dim3 grid_resc(DIV_UP(n_long_rows*nlon_out, block.y),              batch_size);
#if 0
    printf("getPtxver(): %d\n", getPtxver());
    printf("n_long_rows: %ld, max_row_len: %ld\n", n_long_rows, max_row_len);
    printf("Launching s2_attn_fwd_ring_softmax_k<%d, %d, %d><<<(%u, %u, %u), (%u, %u), ...>>>\n",
            BDIM_X, BDIM_Y, LOC_SIZE, grid_lr.x, grid_lr.y, grid_lr.z, block.x, block.y);
#endif
    const int max_niter_cta = DIV_UP(max_row_len, cta_per_row);

    size_t shsize = (sizeof(FLOATV_T)*nchans_in + sizeof(FLOATV_T *)*max_niter_cta)* block.y;

    auto kern = &s2_attn_fwd_ring_softmax_k<BDIM_X, BDIM_Y, LOC_SIZE, FLOATV_T>;
    ensure_dyn_shmem(reinterpret_cast<const void*>(kern), shsize);

    kern<<<grid_lr, block, shsize, stream>>>(params, max_niter_cta, n_long_rows,
                                             _kxp, _qyp, _row_idx, _row_off, _col_idx,
                                             _qdotk_max, _qdotk_max_new);
    CHECK_ERROR("s2_attn_fwd_ring_softmax_k");

    // also copies _qdotk_max_new values for long rows back into caller-provided _qdotk_max buffer
    s2_attn_fwd_ring_rescale_k<BDIM_X, BDIM_Y, LOC_SIZE>
                              <<<grid_resc, block, 0, stream>>>(params, n_long_rows, _row_idx,
                                                                _alpha_sum, _qdotk_max, _qdotk_max_new, _y_acc);
    CHECK_ERROR("s2_attn_fwd_ring_rescale_k");

    shsize = (sizeof(FLOATV_T)*(nchans_in + nchans_out) + sizeof(FLOATV_T *)*max_niter_cta*2 + sizeof(float)*max_niter_cta) * block.y;
    if (chin_as_out) {
        auto kern = &s2_attn_fwd_ring_finalize_k<BDIM_X, BDIM_Y, 1, LOC_SIZE, FLOATV_T>;
        ensure_dyn_shmem(reinterpret_cast<const void*>(kern), shsize);

        kern<<<grid_lr, block, shsize, stream>>>(params, max_niter_cta, n_long_rows,
                                                 _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx,
                                                 _quad_weights, _alpha_sum, _qdotk_max_new, _y_acc);
    } else {
        auto kern = &s2_attn_fwd_ring_finalize_k<BDIM_X, BDIM_Y, 0, LOC_SIZE, FLOATV_T>;
        ensure_dyn_shmem(reinterpret_cast<const void*>(kern), shsize);

        kern<<<grid_lr, block, shsize, stream>>>(params, max_niter_cta, n_long_rows,
                                                 _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx,
                                                 _quad_weights, _alpha_sum, _qdotk_max_new, _y_acc);
    }
    CHECK_ERROR("s2_attn_fwd_ring_finalize_k");

    return;
}


#if 0
// REFERENCE SINGLE-KERNEL TWO-PASS SOFTMAX VERSION
template<int BDIM_X,
         int BDIM_Y,
         int CHIN_AS_OUT, // 1 iif "BDIM_X*(NLOC-1) <= nchan_in <= BDIM_X*NLOC" else 0
         int NLOC,        // smallest int such that BDIM_X*NLOC >= nchan_out
         typename FLOATV_T>
__global__
__launch_bounds__(BDIM_X*BDIM_Y)
void s2_attn_fwd_ring_special_ldg_2pass_k(int nchan_in,
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
                                          const int32_t  *__restrict__ row_idx,
                                          const int64_t  *__restrict__ row_off,
                                          const int64_t  *__restrict__ col_idx,
                                          const float    *__restrict__ quad_weights,
                                                FLOATV_T *__restrict__ y_acc,
                                                float    *__restrict__ alpha_sum_buf,
                                                float    *__restrict__ qdotk_max_buf) {
    static_assert(0 == (BDIM_X & (BDIM_X-1)));
    static_assert(0 == (BDIM_Y & (BDIM_Y-1)));
    static_assert((BDIM_X == 32 && BDIM_Y  > 1) ||
                  (BDIM_X  > 32 && BDIM_Y == 1)) ;

    const int tidx  = threadIdx.x;
    const int ctaid = blockIdx.x*blockDim.y + threadIdx.y;
    const int batch = blockIdx.y;

    if (ctaid >= nlat_out*nlon_out) {
        return;
    }

    extern __shared__ __align__(sizeof(float4)) float shext[];

    FLOATV_T *shq = reinterpret_cast<FLOATV_T *>(shext) + threadIdx.y*nchan_in + tidx;

    const int h  = ctaid / nlon_out;
    const int wo = ctaid - (h*nlon_out);   // LOCAL wo
    const int ho = row_idx[h];

    kx += int64_t(batch)*nlat_halo*nlon_kx*nchan_in + tidx;
    qy += int64_t(batch)*nlat_out*nlon_out*nchan_in + int64_t(ho)*nlon_out*nchan_in + int64_t(wo)*nchan_in + tidx;
    vx += int64_t(batch)*nlat_halo*nlon_kx*nchan_out + tidx;

    const int64_t out_flat = int64_t(batch)*nlat_out*nlon_out + int64_t(ho)*nlon_out + wo;

    y_acc         += out_flat*nchan_out + tidx;
    alpha_sum_buf += out_flat;
    qdotk_max_buf += out_flat;

    FLOATV_T locy[NLOC];

    // Load current state from buffers (running sums across prior ring steps)
    float       alpha_sum     = alpha_sum_buf[0];
    const float qdotk_max_old = qdotk_max_buf[0];

    strided_op<BDIM_X,               NLOC    >(nchan_out, [&](int i) {       locy[i] = y_acc[i*BDIM_X]; });
    strided_op<BDIM_X, CHIN_AS_OUT ? NLOC : 0>(nchan_in,  [&](int i) { shq[i*BDIM_X] =    qy[i*BDIM_X]; });

    const int64_t rbeg = row_off[ho];
    const int64_t rend = row_off[ho + 1];
    col_idx += rbeg;
    const int rlen = rend - rbeg;

    // ------------------------------------------------------------------
    // Pass A: scan all valid columns in this ring step and combine
    // with the running max carried in qdotk_max_old.
    // ------------------------------------------------------------------
    float qdotk_max = qdotk_max_old;
    for (int off = 0; off < rlen; off++) {
        const int64_t col   = col_idx[off];
        const int hi_global = col / nlon_in;
        const int wi        = col - (hi_global * nlon_in);
        const int wi_wo     = wi + pscale * wo;
        const int wip       = wi_wo - (wi_wo / nlon_in) * nlon_in;

        if (wip < lon_lo_kx || wip >= lon_lo_kx + nlon_kx) continue;

        const int hi_local  = hi_global - lat_halo_start;
        if (hi_local < 0 || hi_local >= nlat_halo) continue;

        const int wip_local = wip - lon_lo_kx;
        const FLOATV_T *_kx = kx + int64_t(hi_local)*nlon_kx*nchan_in + int64_t(wip_local)*nchan_in;

        FLOATV_T qdotkv = __vset<FLOATV_T>(0.f);
        strided_op<BDIM_X, CHIN_AS_OUT ? NLOC : 0>(nchan_in, [&](int i) { qdotkv = __vadd(qdotkv, __vmul(shq[i*BDIM_X], _kx[i*BDIM_X])); });

        float qdotk = __vred(qdotkv);
        if constexpr(BDIM_X == 32) { qdotk =          __warp_sum(qdotk); }
        else                       { qdotk = __block_sum<BDIM_X>(qdotk); }

        qdotk_max = max(qdotk_max, qdotk);
    }

    // Rescale the carry (alpha_sum, locy) to the new max.
    // Skip when equal: no new entry advanced the max, and on the very first
    // ring step qdotk_max_old may be -inf with no valid neighbors, in which
    // case expf(-inf - (-inf)) would be NaN.
    if (qdotk_max != qdotk_max_old) {
        const float exp_save = expf(qdotk_max_old - qdotk_max);
        alpha_sum *= exp_save;
        strided_op<BDIM_X, NLOC>(nchan_out, [&](int i) { locy[i] = __vscale(exp_save, locy[i]); });
    }

    // ------------------------------------------------------------------
    // Pass B: accumulate alpha_sum and locy against the now-final
    // qdotk_max, no per-iteration exp_save correction.
    // ------------------------------------------------------------------
    for (int off = 0; off < rlen; off++) {
        const int64_t col   = col_idx[off];
        const int hi_global = col / nlon_in;
        const int wi        = col - (hi_global * nlon_in);
        const int wi_wo     = wi + pscale * wo;
        const int wip       = wi_wo - (wi_wo / nlon_in) * nlon_in;

        if (wip < lon_lo_kx || wip >= lon_lo_kx + nlon_kx) continue;

        const int hi_local  = hi_global - lat_halo_start;
        if (hi_local < 0 || hi_local >= nlat_halo) continue;

        const int wip_local = wip - lon_lo_kx;
        const FLOATV_T *_kx = kx + int64_t(hi_local)*nlon_kx*nchan_in  + int64_t(wip_local)*nchan_in;
        const FLOATV_T *_vx = vx + int64_t(hi_local)*nlon_kx*nchan_out + int64_t(wip_local)*nchan_out;

        FLOATV_T qdotkv = __vset<FLOATV_T>(0.f);
        strided_op<BDIM_X, CHIN_AS_OUT ? NLOC : 0>(nchan_in, [&](int i) { qdotkv = __vadd(qdotkv, __vmul(shq[i*BDIM_X], _kx[i*BDIM_X])); });

        float qdotk = __vred(qdotkv);
        if constexpr(BDIM_X == 32) { qdotk =          __warp_sum(qdotk); }
        else                       { qdotk = __block_sum<BDIM_X>(qdotk); }

        const float alpha = expf(qdotk - qdotk_max) * quad_weights[hi_global];

        alpha_sum += alpha;

        strided_op<BDIM_X, NLOC>(nchan_out, [&](int i) { locy[i] = __vadd(locy[i], __vscale(alpha, _vx[i*BDIM_X])); });
    }

    // Store updated state back to buffers
    if (!tidx) {
        alpha_sum_buf[0] = alpha_sum;
        qdotk_max_buf[0] = qdotk_max;
    }

    strided_op<BDIM_X, NLOC>(nchan_out, [&](int i) { y_acc[i*BDIM_X] = locy[i]; });

    return;
}
#endif


