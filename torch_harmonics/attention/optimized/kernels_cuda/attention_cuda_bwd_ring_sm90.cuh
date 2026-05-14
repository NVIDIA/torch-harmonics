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

__device__ __forceinline__ void atomicMax(float *ptr, float val) {

    int *int_ptr = (int*)ptr;
    
    // Read the current value at the ptr
    int old = *int_ptr, assumed;
    
    do {
        assumed = old;
        if (__int_as_float(assumed) >= val) {
            break;
        }
        old = atomicCAS(int_ptr, assumed, __float_as_int(val));
        
    } while (assumed != old);
    return;
}

template<int BDIM_X,
         int BDIM_Y,
         int NLOC,          // smallest int such that BDIM_X*NLOC >= nchan_in
         typename FLOATV_T> // either float or float4
__global__
__launch_bounds__(BDIM_X*BDIM_Y)
void s2_attn_bwd_ring_pass1_softmax_k(const int nlat_max,
                                      const int nchan_in,
                                      const int nlat_halo,
                                      const int nlon_kx,
                                      const int nlon_in,
                                      const int pscale,           // GLOBAL pscale = nlon_in / nlon_out_global
                                      const int lon_lo_kx,
                                      const int lat_halo_start,
                                      const int nlat_out,
                                      const int nlon_out,
                                      const FLOATV_T *__restrict__ kx,           // [batch][nlat_halo][nlon_kx][nchan_in]
                                      const FLOATV_T *__restrict__ qy,           // [batch][nlat_out][nlon_out][nchan_in]
                                      const int32_t  *__restrict__ row_idx,
                                      const int64_t  *__restrict__ row_off,
                                      const int64_t  *__restrict__ col_idx,
                                      const float    *__restrict__ qdotk_max_prev,   
                                            float    *__restrict__ qdotk_max_curr) {  // NEW, caller inits to -inf each ring step

    static_assert(0 == (BDIM_X & (BDIM_X-1)));
    static_assert(0 == (BDIM_Y & (BDIM_Y-1)));
    static_assert((BDIM_X == 32 && BDIM_Y  > 1) ||
                  (BDIM_X  > 32 && BDIM_Y == 1)) ;

    const int tidx = threadIdx.x;

    const int blk_per_row = gridDim.y; // blocks along Y process the same (ho,wo) 
                                       // point by iteration over the (same) CSR 
                                       // row in an interleaved fashion
    const int blk_split_id = blockIdx.y;

    const int batch = blockIdx.z;
    const uint64_t ctaid = uint64_t(blockIdx.x) * blockDim.y + threadIdx.y;

    if (ctaid >= uint64_t(nlat_max)*nlon_out) {
        return;
    }

    const int h     = ctaid / nlon_out;
    const int wo    = ctaid - (h * nlon_out);
    const int ho    = row_idx[h];

    kx += int64_t(batch)*nlat_halo*nlon_kx*nchan_in + tidx;
    qy += int64_t(batch)*nlat_out*nlon_out*nchan_in  + int64_t(ho)*nlon_out*nchan_in  + int64_t(wo)*nchan_in + tidx;

    FLOATV_T loc_qy[NLOC];
    strided_op<BDIM_X, NLOC>(nchan_in,  [&](int i) { loc_qy[i] = qy[i*BDIM_X]; });

    const int64_t out_flat = int64_t(batch)*nlat_out*nlon_out + int64_t(ho)*nlon_out + wo;

    qdotk_max_prev  += out_flat;
    qdotk_max_curr  += out_flat;

    // Load current state
    float qdotk_max = qdotk_max_prev[0];

    // `pscale` is the GLOBAL ratio nlon_in / nlon_out_global, passed by the caller.
    // The kernel's `nlon_out` is the LOCAL output width (nlon_out_local), not the global one.

    const int64_t rbeg = row_off[ho];
    const int64_t rend = row_off[ho + 1];
    col_idx += rbeg;
    const int rlen = rend - rbeg;

    for (int off = blk_split_id; off < rlen; off += blk_per_row) {

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

        FLOATV_T qdotk_v = __vset<FLOATV_T>(0.0f);
        strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { qdotk_v = __vadd(qdotk_v, __vmul(loc_qy[i], _kx[i*BDIM_X])); });

        float qdotk = __vred(qdotk_v);
        if constexpr(BDIM_X == 32) { qdotk = __warp_sum(qdotk);          } 
        else                       { qdotk = __block_sum<BDIM_X>(qdotk); }

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
void s2_attn_bwd_ring_pass1_rescale_k(const int nlat_max,
                                      const int nchan_in,
                                      const int nlat_out,
                                      const int nlon_out,
                                      const int32_t  *__restrict__ row_idx,
                                            float    *__restrict__ alpha_sum_buf,      // [batch][nlat_out][nlon_out] (in/out)
                                            float    *__restrict__ qdotk_max_prev,      // [batch][nlat_out][nlon_out] (in/out)
                                            float    *__restrict__ qdotk_max_curr,      // [batch][nlat_out][nlon_out] (in/out)          ////////////// NEW
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

    if (ctaid >= uint64_t(nlat_max)*nlon_out) {
        return;
    }

    const int h     = ctaid / nlon_out;
    const int wo    = ctaid - (h * nlon_out);
    const int ho    = row_idx[h];

    const int64_t out_flat = int64_t(batch)*nlat_out*nlon_out + int64_t(ho)*nlon_out + wo;

    qdotk_max_prev += out_flat;
    qdotk_max_curr += out_flat;
    alpha_sum_buf  += out_flat;
    integral_buf   += out_flat;
    alpha_k_buf    += out_flat*nchan_in + tidx;
    alpha_kvw_buf  += out_flat*nchan_in + tidx;

    const float qdotk_max_old = qdotk_max_prev[0];
    const float qdotk_max_new = qdotk_max_curr[0];

    if (qdotk_max_old == qdotk_max_new) {
        return;
    }

    const float max_correction = expf(qdotk_max_old - qdotk_max_new);

    const float alpha_sum = max_correction*alpha_sum_buf[0];
    const float integral  = max_correction*integral_buf[0];

    FLOATV_T loc_k__[NLOC];
    FLOATV_T loc_kvw[NLOC];

    strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { loc_k__[i] =   alpha_k_buf[i*BDIM_X]; });
    strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { loc_kvw[i] = alpha_kvw_buf[i*BDIM_X]; });

    strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { loc_k__[i] = __vscale(max_correction, loc_k__[i]); });
    strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { loc_kvw[i] = __vscale(max_correction, loc_kvw[i]); });
    
    strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) {   alpha_k_buf[i*BDIM_X] = loc_k__[i]; });
    strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { alpha_kvw_buf[i*BDIM_X] = loc_kvw[i]; });
    
    if (!tidx) {
        qdotk_max_prev[0] = qdotk_max_new; // used for next ring call
        alpha_sum_buf[0] = alpha_sum;
        integral_buf[0] = integral;
    }

    return;
}

template<int BDIM_X,
         int BDIM_Y,
         int CHOUT_AS_IN,   // 1 iif "BDIM_X*(NLOC-1) <= nchan_out <= BDIM_X*NLOC" else 0
         int NLOC,          // smallest int such that BDIM_X*NLOC >= nchan_in
         typename FLOATV_T> // either float or float4
__global__
__launch_bounds__(BDIM_X*BDIM_Y)
void s2_attn_bwd_ring_pass1_finalize_k(const int nlat_max,
                                       const int nchan_in,
                                       const int nchan_out,
                                       const int nlat_halo,
                                       const int nlon_kx,
                                       const int nlon_in,
                                       const int pscale,           // GLOBAL pscale = nlon_in / nlon_out_global
                                       const int lon_lo_kx,
                                       const int lat_halo_start,
                                       const int nlat_out,
                                       const int nlon_out,
                                       const FLOATV_T *__restrict__ kx,           // [batch][nlat_halo][nlon_kx][nchan_in]
                                       const FLOATV_T *__restrict__ vx,           // [batch][nlat_halo][nlon_kx][nchan_out]
                                       const FLOATV_T *__restrict__ qy,           // [batch][nlat_out][nlon_out][nchan_in]
                                       const FLOATV_T *__restrict__ dy,           // [batch][nlat_out][nlon_out][nchan_out]
                                       const int32_t  *__restrict__ row_idx,
                                       const int64_t  *__restrict__ row_off,
                                       const int64_t  *__restrict__ col_idx,
                                       const float    *__restrict__ quad_weights,
                                             float    *__restrict__ alpha_sum_buf,      // [batch][nlat_out][nlon_out] (in/out)
                                             float    *__restrict__ qdotk_max_curr,     // [batch][nlat_out][nlon_out] (in/out)
                                             float    *__restrict__ integral_buf,       // [batch][nlat_out][nlon_out] unnormalized (in/out)
                                             FLOATV_T *__restrict__ alpha_k_buf,        // [batch][nlat_out][nlon_out][nchan_in] (in/out)
                                             FLOATV_T *__restrict__ alpha_kvw_buf) {    // [batch][nlat_out][nlon_out][nchan_in] (in/out)

    static_assert(0 == (BDIM_X & (BDIM_X-1)));
    static_assert(0 == (BDIM_Y & (BDIM_Y-1)));
    static_assert((BDIM_X == 32 && BDIM_Y  > 1) ||
                  (BDIM_X  > 32 && BDIM_Y == 1)) ;

    const int tidx = threadIdx.x;
    
    const int blk_per_row = gridDim.y; // blocks along Y process the same (ho,wo) 
                                       // point by iteration over the (same) CSR 
                                       // row in an interleaved fashion
    const int blk_split_id = blockIdx.y;

    const int batch = blockIdx.z;
    const uint64_t ctaid = uint64_t(blockIdx.x) * blockDim.y + threadIdx.y;

    if (ctaid >= uint64_t(nlat_max)*nlon_out) {
        return;
    }

    extern __shared__ __align__(sizeof(float4)) float shext[];

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
    qdotk_max_curr += out_flat;
    integral_buf   += out_flat;
    alpha_k_buf    += out_flat*nchan_in + tidx;
    alpha_kvw_buf  += out_flat*nchan_in + tidx;

    //strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { loc_k__[i] =   alpha_k_buf[i*BDIM_X]; });
    //strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { loc_kvw[i] = alpha_kvw_buf[i*BDIM_X]; });

    strided_op<BDIM_X,               NLOC    >(nchan_in,  [&](int i) { sh_qy[i*BDIM_X] = qy[i*BDIM_X]; });
    strided_op<BDIM_X, CHOUT_AS_IN ? NLOC : 0>(nchan_out, [&](int i) { sh_dy[i*BDIM_X] = dy[i*BDIM_X]; });

    // `pscale` is the GLOBAL ratio nlon_in / nlon_out_global, passed by the caller.
    // The kernel's `nlon_out` is the LOCAL output width (nlon_out_local), not the global one.

    // sh_alpha_k[nchan_in], sh_alpha_kvw[nchan_in], sh_dy[nchan_out]
    FLOATV_T loc_k__[NLOC];
    FLOATV_T loc_kvw[NLOC];
        
    strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { loc_k__[i] = __vset<FLOATV_T>(0); });
    strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { loc_kvw[i] = __vset<FLOATV_T>(0); });

    // Load current state
    float alpha_sum = 0;
    float integral = 0;
    float qdotk_max_new = qdotk_max_curr[0];

    const int64_t rbeg = row_off[ho];
    const int64_t rend = row_off[ho + 1];
    col_idx += rbeg;
    const int rlen = rend - rbeg;

    // Pass B: accumulate alpha_sum, integral, alpha_k, alpha_kvw against
    // the now-final qdotk_max, no per-iteration max_correction needed.
    for (int off = blk_split_id; off < rlen; off += blk_per_row) {

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

        const float alpha_inz  = expf(qdotk - qdotk_max_new) * quad_weights[hi_global];
        const float ainz_gdotv = alpha_inz * gdotv;

        alpha_sum += alpha_inz;
        integral  += ainz_gdotv;

        strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { loc_k__[i] = __vadd(loc_k__[i], __vscale(alpha_inz,  _kx[i*BDIM_X])); });
        strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { loc_kvw[i] = __vadd(loc_kvw[i], __vscale(ainz_gdotv, _kx[i*BDIM_X])); });
    }

    // Store updated state
    if (!tidx) {
        atomicAdd(alpha_sum_buf, alpha_sum);
        //qdotk_max_buf[0] = qdotk_max; // no need to store, after kernel qdotk_max_curr will be copied into qdotk_max_prev
        atomicAdd(integral_buf, integral);
    }

    strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { atomicAdd(alpha_k_buf   + i*BDIM_X, loc_k__[i]); });
    strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { atomicAdd(alpha_kvw_buf + i*BDIM_X, loc_kvw[i]); });

    return;
}

template<int BDIM_X,
         int LOC_SIZE,
         typename FLOATV_T>
void spc_attn_ring_pass1_bwd_long_rows(int64_t n_long_rows,
                                       int64_t max_row_len,
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

    // TMA prerequisites:                                                                                                                                                              
    //   - FLOATV_T == float4 (TMA bulk-async copy needs 16-B alignment)
    //   - loaded device code compiled for compute_90+ (ptxVersion is                                                                                                                  
    //     resolved per-current-device, so this alone gates correctly                                                                                                                  
    //     across SASS-only and PTX-JIT builds)
    constexpr bool VEC4    = std::is_same_v<FLOATV_T, float4>;                                                                                                                         
    const bool     use_tma = VEC4 && (getPtxver() >= 90); 

    const bool chout_as_in = (nchans_out >= BDIM_X*(LOC_SIZE-1) && 
                              nchans_out <= BDIM_X* LOC_SIZE  );

    constexpr int BDIM_Y = (BDIM_X <= 32) ? THREADS/BDIM_X : 1;

    int cta_per_row = min(int64_t(32), DIV_UP(max_row_len, PASS2_MIN_WORK_PER_BLOCK));

    dim3 block(BDIM_X, BDIM_Y);

    dim3 grid_lr  (DIV_UP(n_long_rows*nlon_out, block.y), cta_per_row, batch_size);
    dim3 grid_resc(DIV_UP(n_long_rows*nlon_out, block.y),              batch_size);


    // temporary for correctness only
    torch::Tensor t_qdotk_max_new = torch::from_blob(_qdotk_max,
                                                     {batch_size*nlat_out*nlon_out},
                                                     torch::TensorOptions().dtype(torch::kFloat32)
                                                                           .device(torch::kCUDA)).clone();

    float *_qdotk_max_new = reinterpret_cast<float *>(t_qdotk_max_new.data_ptr());

    if (0 && use_tma) {
#if 0
        //                                       qy           dy         2x kx          2x vx
        size_t shsize = sizeof(FLOATV_T)*(/*nchans_in +*/ nchans_out + nchans_in*2 + nchans_out*2) * block.y;
#if 0
        printf("getPtxver(): %d\n", getPtxver());
        printf("Launching s2_attn_bwd_ring_pass1_special_tma_k<%d, %d, %d, %d><<<(%u, %u), (%u, %u), %zu, ...>>>\n",
                BDIM_X, BDIM_Y, chout_as_in, CUR_LOC_SIZE, grid.x, grid.y, block.x, block.y, shsize);
#endif
        if (chout_as_in) {
            auto kern = &s2_attn_bwd_ring_pass1_special_tma_k<BDIM_X, BDIM_Y, 1, CUR_LOC_SIZE, FLOATV_T>;

            ensure_dyn_shmem(reinterpret_cast<const void*>(kern), shsize);

            kern<<<grid, block, shsize, stream>>>(nchans_in, nchans_out, nlat_halo, nlon_kx,
                                                  nlon_in, pscale, lon_lo_kx, lat_halo_start, nlat_out, nlon_out,
                                                  _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights,
                                                  _alpha_sum, _qdotk_max, _integral, _alpha_k, _alpha_kvw);
        } else {
            auto kern = &s2_attn_bwd_ring_pass1_special_tma_k<BDIM_X, BDIM_Y, 0, CUR_LOC_SIZE, FLOATV_T>;

            ensure_dyn_shmem(reinterpret_cast<const void*>(kern), shsize);

            kern<<<grid, block, shsize, stream>>>(nchans_in, nchans_out, nlat_halo, nlon_kx,
                                                  nlon_in, pscale, lon_lo_kx, lat_halo_start, nlat_out, nlon_out,
                                                  _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights,
                                                  _alpha_sum, _qdotk_max, _integral, _alpha_k, _alpha_kvw);
        }
        CHECK_ERROR("s2_attn_bwd_ring_pass1_special_tma_k");
#endif
    } else {
#if 0
        printf("getPtxver(): %d\n", getPtxver());
        printf("n_long_rows: %ld, max_row_len: %ld\n", n_long_rows, max_row_len);
        printf("Launching s2_attn_bwd_ring_pass1_softmax_k<%d, %d, %d><<<(%u, %u, %u), (%u, %u), ...>>>\n",
                BDIM_X, BDIM_Y, LOC_SIZE, grid_lr.x, grid_lr.y, grid_lr.z, block.x, block.y);
#endif
        s2_attn_bwd_ring_pass1_softmax_k<BDIM_X, BDIM_Y, LOC_SIZE>
                                        <<<grid_lr, block, 0, stream>>>(n_long_rows,
                                                                       nchans_in, nlat_halo, nlon_kx,
                                                                       nlon_in, pscale, lon_lo_kx, lat_halo_start, nlat_out, nlon_out,
                                                                       _kxp, _qyp, _row_idx, _row_off, _col_idx, _qdotk_max, _qdotk_max_new);
        CHECK_ERROR("s2_attn_bwd_ring_pass1_softmax_k");
        
        // also copies _qdotk_max_new values for long rows back into caller-provided _qdotk_max buffer
        s2_attn_bwd_ring_pass1_rescale_k<BDIM_X, BDIM_Y, LOC_SIZE>
                                        <<<grid_resc, block, 0, stream>>>(n_long_rows,
                                                                          nchans_in, nlat_out, nlon_out, _row_idx,
                                                                          _alpha_sum, _qdotk_max, _qdotk_max_new, _integral, _alpha_k, _alpha_kvw);
        CHECK_ERROR("s2_attn_bwd_ring_pass1_rescale_k");

        size_t shsize = sizeof(FLOATV_T)*(nchans_in + nchans_out) * block.y;
        if (chout_as_in) {
            s2_attn_bwd_ring_pass1_finalize_k<BDIM_X, BDIM_Y, 1, LOC_SIZE>
                                            <<<grid_lr, block, shsize, stream>>>(n_long_rows,
                                                                                 nchans_in, nchans_out, nlat_halo, nlon_kx,
                                                                                 nlon_in, pscale, lon_lo_kx, lat_halo_start, nlat_out, nlon_out,
                                                                                 _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights,
                                                                                 _alpha_sum, _qdotk_max_new, _integral, _alpha_k, _alpha_kvw);
        } else {
            s2_attn_bwd_ring_pass1_finalize_k<BDIM_X, BDIM_Y, 0, LOC_SIZE>
                                            <<<grid_lr, block, shsize, stream>>>(n_long_rows,
                                                                                 nchans_in, nchans_out, nlat_halo, nlon_kx,
                                                                                 nlon_in, pscale, lon_lo_kx, lat_halo_start, nlat_out, nlon_out,
                                                                                 _kxp, _vxp, _qyp, _dyp, _row_idx, _row_off, _col_idx, _quad_weights,
                                                                                 _alpha_sum, _qdotk_max_new, _integral, _alpha_k, _alpha_kvw);
        }
        CHECK_ERROR("s2_attn_bwd_ring_pass1_finalize_k");
    }
    return;
}


#if 0
// REFERENCE SINGLE-KERNEL TWO-PASS SOFTMAX VERSION

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
void s2_attn_bwd_ring_step_pass1_special_vec_k(const int nchan_in,
                                               const int nchan_out,
                                               const int nlat_halo,
                                               const int nlon_kx,
                                               const int nlon_in,
                                               const int pscale,           // GLOBAL pscale = nlon_in / nlon_out_global
                                               const int lon_lo_kx,
                                               const int lat_halo_start,
                                               const int nlat_out,
                                               const int nlon_out,
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
    float alpha_sum     = alpha_sum_buf[0];
    float qdotk_max_old = qdotk_max_buf[0];
    float integral      = integral_buf[0];

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

    // Pass A: find qdotk_max over all valid columns in this ring step,
    // combined with the running max carried in qdotk_max_old.
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

        FLOATV_T qdotk_v = __vset<FLOATV_T>(0.0f);
        strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[i*BDIM_X], _kx[i*BDIM_X])); });

        float qdotk = __vred(qdotk_v);
        if constexpr(BDIM_X == 32) {
            qdotk = __warp_sum(qdotk);
        } else {
            qdotk = __block_sum<BDIM_X>(qdotk);
        }

        qdotk_max = max(qdotk_max, qdotk);
    }

    // Rescale the carry from previous ring steps to the new qdotk_max.
    // qdotk_max >= qdotk_max_old by construction; skip when they are equal
    // (no new entries advanced the max). This also avoids NaN in the first
    // ring step when qdotk_max_old is -inf and no valid entries were seen,
    // since expf(-inf - (-inf)) is NaN.
    if (qdotk_max != qdotk_max_old) {
        const float max_correction = expf(qdotk_max_old - qdotk_max);
        alpha_sum *= max_correction;
        integral  *= max_correction;
        strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { loc_k__[i] = __vscale(max_correction, loc_k__[i]); });
        strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { loc_kvw[i] = __vscale(max_correction, loc_kvw[i]); });
    }

    // Pass B: accumulate alpha_sum, integral, alpha_k, alpha_kvw against
    // the now-final qdotk_max, no per-iteration max_correction needed.
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

        const float alpha_inz  = expf(qdotk - qdotk_max) * quad_weights[hi_global];
        const float ainz_gdotv = alpha_inz * gdotv;

        alpha_sum += alpha_inz;
        integral  += ainz_gdotv;

        strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { loc_k__[i] = __vadd(loc_k__[i], __vscale(alpha_inz,  _kx[i*BDIM_X])); });
        strided_op<BDIM_X, NLOC>(nchan_in, [&](int i) { loc_kvw[i] = __vadd(loc_kvw[i], __vscale(ainz_gdotv, _kx[i*BDIM_X])); });
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
#endif


