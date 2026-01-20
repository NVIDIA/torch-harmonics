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

//#include "cudamacro.h"
#include "disco.h"
#include "disco_cuda.cuh"
#include "csr_cuda.cuh"

#define THREADS (64)

#define MAX_LOCAL_ARR_LEN (20)

namespace disco_kernels {

using namespace utility_kernels;

template <int BDIM_X, int ELXTH, typename REAL_T>
__device__ void disco_fwd_d(const int Hi, const int Wi, const int K, const int Ho, const int Wo, const int pscale,
                            const int64_t *__restrict__ roff, const int64_t *__restrict__ kers,
                            const int64_t *__restrict__ rows, const int64_t *__restrict__ cols,
                            const REAL_T *__restrict__ vals, const REAL_T *__restrict__ inp, REAL_T *__restrict__ out)
{

    const int tid = threadIdx.x;

    const int64_t bidx = blockIdx.x; // global row
    const int64_t bidy = blockIdx.y; // bc

    int64_t soff = roff[bidx];
    int64_t eoff = roff[bidx + 1];

    const int64_t ker = kers[soff];
    const int64_t row = rows[soff];

    inp += bidy*Hi*Wi;
    out += bidy*K*Ho*Wo + ker*Ho*Wo + row*Wo;

    REAL_T __reg[ELXTH] = {0};

    // align to larger supported fp type
    extern __shared__ __align__(sizeof(double)) unsigned char __sh_ptr[]; // REAL_T __sh[2*Wi + ppscale*(BDIM_X*ELXTH - Wo)]
    REAL_T *__sh = reinterpret_cast<REAL_T *>(__sh_ptr);

    int col_prev = cols[soff];

    int h_prev = col_prev / Wi;
    int w_prev = col_prev % Wi;

    // copy current inp row in shmem
    for (int i = tid; i < Wi; i += BDIM_X) {
        const REAL_T v = inp[h_prev * Wi + i];
        __sh[i] = v;
        __sh[Wi + i] = v;
    }
    // locations __sh[2*Wi : ppscale*(BDIM_X*ELXTH-Wo)] are not used
    __syncthreads();

    // loops along the colums of CTA's row
    for (int64_t nz = soff; nz < eoff; nz++) {

        const int col = cols[nz];
        const REAL_T val = vals[nz];

        // if we are processing a nz with a col value
        // leading to a new row of inp then copy it
        // to shmem;
        // checks whether (h_prev < h) with:
        //  (col >= col_prev - (col_prev % Wi) + Wi)
        if (col >= col_prev - w_prev + Wi) {

            col_prev = col;
            h_prev = col / Wi;
            w_prev = col % Wi;

            __syncthreads();
            for (int i = tid; i < Wi; i += BDIM_X) {
                const REAL_T v = inp[h_prev * Wi + i];
                __sh[i] = v;
                __sh[Wi + i] = v;
            }
            __syncthreads();
        }

        const int w = w_prev + (col - col_prev);

#pragma unroll
        for (int i = 0; i < ELXTH; i++) {

            const int pp = i * BDIM_X + tid;

            // original lines:
            //
            //   if (pp >= Wo) break;
            //   const int wpp = (w + pscale*pp) % Wi;
            //
            // value of (w + pscale*pp) < (Wi + (Wi/Wo)*Wo) = 2*Wi
            // so we can allocate twice the amount of shmem,
            // replicate the current inp row and avoid the costly mod
            //
            // also, to avoid the conditional, sh can be extended to
            // cover the maximum location accessed during this loop
            //
            // REAL_T __sh[2*Wi + ppscale*NUM_REM]
            //
            //   Wi + (Wi/Wo)*BDIM_X*ELXTH = (since BDIM_X*ELXTH >= Wo) =
            // = Wi + (Wi/Wo)*(Wo + (BDIM_X*ELXTH - Wo)) =
            // = 2*Wi + ppscale*NUM_REM
            //
            // with NUM_REM = BDIM_X*ELXTH - Wo

            const int wpp = w + pscale * pp;

            __reg[i] += val * __sh[wpp];
        }
    }

#pragma unroll
    for (int i = 0; i < ELXTH; i++) {

        const int pp = i * BDIM_X + tid;
        if (pp >= Wo) break;

        out[pp] = __reg[i];
    }

    return;
}

template <int BDIM_X, int ELXTH, typename REAL_T>
__global__
    __launch_bounds__(BDIM_X) void disco_fwd_blk_k(const int Hi, const int Wi, const int K, const int Ho, const int Wo,
                                                   const int pscale, const int64_t *__restrict__ roff,
                                                   const int64_t *__restrict__ kers, const int64_t *__restrict__ rows,
                                                   const int64_t *__restrict__ cols, const REAL_T *__restrict__ vals,
                                                   const REAL_T *__restrict__ inp, REAL_T *__restrict__ out)
{

    disco_fwd_d<BDIM_X, ELXTH>(Hi, Wi, K, Ho, Wo, pscale, roff, kers, rows, cols, vals, inp, out);

    return;
}

template <int NTH, int ELXTH, typename REAL_T>
static void launch_kernel(int BC, int Hi, int Wi, int K, int Ho, int Wo, int64_t nrows, int64_t *roff_d, int64_t *ker_d,
                          int64_t *row_d, int64_t *col_d, REAL_T *val_d, REAL_T *inp_d, REAL_T *out_d,
                          cudaStream_t stream)
{

    static_assert(sizeof(REAL_T) == 2 || sizeof(REAL_T) == 4 || sizeof(REAL_T) == 8);

    if constexpr (ELXTH <= ELXTH_MAX) {
        if (NTH * ELXTH >= Wo) {
            dim3 grid(nrows, BC);

            const int pscale = Wi / Wo;
            size_t shmem = sizeof(*out_d) * (Wi * 2 + pscale * (NTH * ELXTH - Wo));

            disco_fwd_blk_k<NTH, ELXTH><<<grid, NTH, shmem, stream>>>(Hi, Wi, K, Ho, Wo, pscale, roff_d, ker_d, row_d,
                                                                      col_d, val_d, inp_d, out_d);
        } else {
            launch_kernel<NTH, ELXTH + 1>(BC, Hi, Wi, K, Ho, Wo, nrows, roff_d, ker_d, row_d, col_d, val_d, inp_d,
                                          out_d, stream);
        }
    }
    return;
}

template<typename VAL_T>
static __global__ void pack_vals_k(const int64_t K,
                                   const int64_t nrows,
                                   const int64_t *__restrict__ row_off,
                                   const VAL_T   *__restrict__ val_dat,
                                         VAL_T   *__restrict__ val_pck) {

    const int tidx = threadIdx.x;
    const int wid = blockIdx.x*blockDim.y + threadIdx.y;
    if (wid >= nrows) {
        return;
    }

    const int64_t rbeg = row_off[wid];
    const int64_t rend = row_off[wid+1];

    const int rlen = rend-rbeg;

    val_pck += rbeg*K;

    for(int off = tidx; off < rlen; off += blockDim.x) {
        for(int ker = 0; ker < K; ker++) {

            val_pck[off*K + ker] = val_dat[ row_off[ker*nrows + wid]  + off];
        }
    }

    return;
}


// BEGIN VERSION WITH CHANNEL-LAST WITH 2D BLOCKS, 2ND DIM IDENTIFYING CHANNLES, NO EINSUM
template<int BDIM_X,
         typename FLOATV_T>
__device__ void processCSR_Kpow2_shm_d(const int wo,
                                       const int rlen,
                                       const int nchan_in, // no. of input floats (not FLOATV_T!) elements along channel dim
                                       const int nlon_in,
                                       const int pscale,
                                       const int K,
                                       const float    *__restrict__ x,
                                       const int64_t  *__restrict__ cols,
                                       const FLOATV_T *__restrict__ vals,
                                             FLOATV_T *__restrict__ shy) {
    const int tidx = threadIdx.x;

    // only used in K_POWER_2==1 branch
    const int log2_K = __ffs(K)-1;

    x    += tidx >> log2_K;
    vals += tidx & (K-1);

    const int BDIM_XdivK = BDIM_X >> log2_K;

    for(int off = 0; off < rlen; off++) {

        const int64_t  col = cols[off];

        const int hi = col / nlon_in;
        const int wi = col - (hi*nlon_in);

        //const int wip = (wi + pscale*wo) % nlon_in;
        // value of (wi + pscale*wo) < (Wi + (Wi/Wo)*Wo) = 2*Wi
        // so we can replace the modulo with:
        int wip = wi + pscale*wo;
        if (wip >= nlon_in) wip -= nlon_in;

        const float *_x = x + int64_t(hi)*nlon_in*nchan_in + int64_t(wip)*nchan_in;

        // if BDIM_X is a multiple of K then "i*(j*BDIM_X) % K = const",
        // so thread "i" only needs to read vals[off*K + (i % K)] to update the
        // whole channel array

        const FLOATV_T myval = vals[0]; //vals[off*K + tidxModK];

        for(int chan = tidx; chan < nchan_in*K; chan += BDIM_X) { // no. of vectors in nchan_in*K dim on intermediate out

            shy[chan] = __vadd(shy[chan],
                               __vmul(myval,
                                      __vset<FLOATV_T>(_x[0])));
            _x += BDIM_XdivK;
        }

        vals += K;
    }
    return;
}

template<int BDIM_X,
         typename FLOATV_T>
__device__ void processCSR_Kanyv_shm_d(const int wo,
                                       const int rlen,
                                       const int nchan_in, // no. of input floats (not FLOATV_T!) elements along channel dim
                                       const int nlon_in,
                                       const int pscale,
                                       const int K,
                                       const float    *__restrict__ x,
                                       const int64_t  *__restrict__ cols,
                                       const FLOATV_T *__restrict__ vals,
                                             FLOATV_T *__restrict__ shy) {
    const int tidx = threadIdx.x;

    for(int off = 0; off < rlen; off++) {

        const int64_t  col = cols[off];

        const int hi = col / nlon_in;
        const int wi = col - (hi*nlon_in);

        //const int wip = (wi + pscale*wo) % nlon_in;
        // value of (wi + pscale*wo) < (Wi + (Wi/Wo)*Wo) = 2*Wi
        // so we can replace the modulo with:
        int wip = wi + pscale*wo;
        if (wip >= nlon_in) wip -= nlon_in;

        const float *_x = x + int64_t(hi)*nlon_in*nchan_in + int64_t(wip)*nchan_in;

        // if BDIM_X is not a multiple of K then "i*(j*BDIM_X) % K = f(i,j)",
        // so the mod need to be recomputed at each iteration of update the update loop
        for(int chan = tidx; chan < nchan_in*K; chan += BDIM_X) { // no. of vectors in nchan_in*K dim on intermediate out

            const int iDivK = chan / K;
            const int iModK = chan - (iDivK*K);

            shy[chan] = __vadd(shy[chan],
                               __vmul(vals[iModK],
                                      __vset<FLOATV_T>(_x[iDivK])));
        }

        vals += K;
    }
    return;
}

template<int BDIM, // change to BDIM_X(<=WARP_SIZE), BDIM_Y TO HANDLE SMALL CHXGR_IN
         typename FLOATV_T> // either float or float4
__global__
__launch_bounds__(BDIM)
void s2_disco_fwd_generic_vec_k(int nchan_in,   // no. of input  float (not FLOATV_T!) elements along channel dim
                                int nlat_in,
                                int nlon_in,
                                int nlat_out,
                                int nlon_out,
                                int pscale,
                                int K,          // no. of output FLOATV_T elem along K dim (kernel size)
                                const float   *__restrict__ x,
                                const int64_t csr_nrow,
                                const int32_t *__restrict__ row_sort,
                                const int64_t *__restrict__ row_off,
                                const int64_t *__restrict__ row_idx,
                                const int64_t *__restrict__ col_idx,
                                const FLOATV_T *__restrict__ val_pck,
                                      FLOATV_T *__restrict__ y) {

    const int tidx  = threadIdx.x;

    const int batch = blockIdx.y;
    const int ctaid = blockIdx.x*blockDim.y + threadIdx.y;

    if (ctaid >= csr_nrow*nlon_out) {
        return;
    }

    const int h  = ctaid / nlon_out;
    const int wo = ctaid - (h*nlon_out);

    // set csr_row to "h" to bypass the row sorting
    const int csr_row = row_sort[h]; // h

    const int64_t rbeg = row_off[csr_row  ];
    const int64_t rend = row_off[csr_row+1];

    const int ho = row_idx[rbeg]; // reads only the first "nrow" rows of row_idx and only the first element of each row

    const int nchan_out = nchan_in*K;

    extern __shared__ __align__(sizeof(float4)) float shext[];
    FLOATV_T *shy = reinterpret_cast<FLOATV_T *>(shext) + threadIdx.y*nchan_out;

    for(int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
        shy[chan] = __vset<FLOATV_T>(0.f);
    }

    x += int64_t(batch)*nlat_in*nlon_in*nchan_in;
    y += int64_t(batch)*nlat_out*nlon_out*nchan_out + int64_t(ho)*nlon_out*nchan_out + int64_t(wo)*nchan_out;

    col_idx += rbeg;
    val_pck += rbeg*K; // val_pck CSR contains K values per element

    const int rlen = rend-rbeg;

    // check if BDIM_X is a multiple of K; since BDIM_X is a power of 2, check if K is also a power of two
    if (!(K & K-1)) { processCSR_Kpow2_shm_d<WARP_SIZE>(wo, rlen, nchan_in, nlon_in, pscale, K, x, col_idx, val_pck, shy); }
    else            { processCSR_Kanyv_shm_d<WARP_SIZE>(wo, rlen, nchan_in, nlon_in, pscale, K, x, col_idx, val_pck, shy); }

    for(int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
        y[chan] = shy[chan];
    }

    return;
}

template<int BDIM_X,
         int SHPAD,
         int NLOC,
         typename FLOATV_T>
__device__ void processCSR_Kpow2_reg_d(const int wo,
                                       const int rlen,
                                       const int nchan_in,    // no. of input floats (not FLOATV_T!) elements along channel dim
                                       const int nlon_in,
                                       const int pscale,
                                       const int K,           // kernel size
                                       const float    *__restrict__ x,
                                       const int64_t  *__restrict__ cols,
                                       const FLOATV_T *__restrict__ vals,
                                       const float *(&shXOff)[BDIM_X+SHPAD],
                                             FLOATV_T (&locy)[NLOC]) {

    constexpr int NLOC_M1 = NLOC-1;

    const int tidx = threadIdx.x;

    const int log2_K = __ffs(K)-1;

    const int tidxDivK = tidx >> log2_K;
    const int tidxModK = tidx  & (K-1);

    cols += tidx;
    vals += tidxModK;

    const int BDIM_XdivK = BDIM_X >> log2_K;

    for(int off = 0; off < rlen; off++) {

        if ((off % BDIM_X) == 0) {
            __sync<BDIM_X>();

            const int64_t  col = (off+tidx < rlen) ? cols[0] : 0;

            const int hi = col / nlon_in;
            const int wi = col - (hi*nlon_in);

            //const int wip = (wi + pscale*wo) % nlon_in;
            // value of (wi + pscale*wo) < (Wi + (Wi/Wo)*Wo) = 2*Wi
            // so we can replace the modulo with:
            int wip = wi + pscale*wo;
            if (wip >= nlon_in) wip -= nlon_in;

            shXOff[tidx] = x + int64_t(hi)*nlon_in*nchan_in + int64_t(wip)*nchan_in;
            cols += BDIM_X;

            __sync<BDIM_X>();
        }

        const float *_x = shXOff[off % BDIM_X] + tidxDivK;

        // if BDIM_X is a multiple of K then "i*(j*BDIM_X) % K = const",
        // so thread "i" only needs to read vals[off*K + (i % K)] to update the
        // whole channel array

        #pragma unroll
        for(int i = 0; i < NLOC_M1; i++) {
            locy[i] = __vadd(locy[i],
                             __vmul(vals[0],
                                    __vset<FLOATV_T>(_x[i*BDIM_XdivK])));
        }
        if (NLOC_M1*BDIM_X+tidx < nchan_in*K) {
            locy[NLOC_M1] = __vadd(locy[NLOC_M1],
                                   __vmul(vals[0],
                                          __vset<FLOATV_T>(_x[NLOC_M1*BDIM_XdivK])));
        }

        vals += K;
    }
    return;
}

template<int BDIM_X,
         int SHPAD,
         int NLOC,
         typename FLOATV_T>
__device__ void processCSR_Kanyv_reg_d(const int wo,
                                       const int rlen,
                                       const int nchan_in,    // no. of input floats (not FLOATV_T!) elements along channel dim
                                       const int nlon_in,
                                       const int pscale,
                                       const int K,           // kernel size
                                       const float    *__restrict__ x,
                                       const int64_t  *__restrict__ cols,
                                       const FLOATV_T *__restrict__ vals,
                                       const float *(&shXOff)[BDIM_X+SHPAD],
                                             FLOATV_T (&locy)[NLOC]) {

    constexpr int NLOC_M1 = NLOC-1;

    const int tidx = threadIdx.x;

    cols += tidx;

    for(int off = 0; off < rlen; off++) {

        if ((off % BDIM_X) == 0) {
            __sync<BDIM_X>();

            const int64_t  col = (off+tidx < rlen) ? cols[0] : 0;

            const int hi = col / nlon_in;
            const int wi = col - (hi*nlon_in);

            //const int wip = (wi + pscale*wo) % nlon_in;
            // value of (wi + pscale*wo) < (Wi + (Wi/Wo)*Wo) = 2*Wi
            // so we can replace the modulo with:
            int wip = wi + pscale*wo;
            if (wip >= nlon_in) wip -= nlon_in;

            shXOff[tidx] = x + int64_t(hi)*nlon_in*nchan_in + int64_t(wip)*nchan_in;
            cols += BDIM_X;

            __sync<BDIM_X>();
        }

        const float *_x = shXOff[off % BDIM_X];

        // if BDIM_X is not a multiple of K then "i*(j*BDIM_X) % K = f(i,j)",
        // so the mod need to be recomputed at each iteration of update the update loop

        #pragma unroll
        for(int i = 0; i < NLOC_M1; i++) {

            const int chan = i*BDIM_X+tidx;
            const int iDivK = chan / K;
            const int iModK = chan - (iDivK*K);

            const FLOATV_T vval = vals[iModK]; //vals[off*K + iModK];
            const FLOATV_T xval = __vset<FLOATV_T>(_x[iDivK]);

            locy[i] = __vadd(locy[i], __vmul(vval, xval));
        }
        if (NLOC_M1*BDIM_X+tidx < nchan_in*K) {

            const int chan = NLOC_M1*BDIM_X+tidx;
            const int iDivK = chan / K;
            const int iModK = chan - (iDivK*K);

            const FLOATV_T vval = vals[iModK]; //vals[off*K + iModK];
            const FLOATV_T xval = __vset<FLOATV_T>(_x[iDivK]);

            locy[NLOC_M1] = __vadd(locy[NLOC_M1], __vmul(vval, xval));
        }

        vals += K;
    }
    return;
}

template<int BDIM_X,
         int BDIM_Y,
         int NLOC,
         typename FLOATV_T> // either float or float4
__global__
__launch_bounds__(BDIM_X*BDIM_Y)
void s2_disco_fwd_special_vec_k(const int nchan_in,   // no. of input  float (not FLOATV_T!) elements along channel dim
                                const int nlat_in,
                                const int nlon_in,
                                const int nlat_out,
                                const int nlon_out,
                                const int pscale,
                                const int K,          // no. of output FLOATV_T elem along K dim (kernel size)
                                const float   *__restrict__ x,
                                const int64_t csr_nrow,
                                const int32_t *__restrict__ row_sort,
                                const int64_t *__restrict__ row_off,
                                const int64_t *__restrict__ row_idx,
                                const int64_t *__restrict__ col_idx,
                                const FLOATV_T *__restrict__ val_pck,
                                      FLOATV_T *__restrict__ y) {

    static_assert(0 == (BDIM_X & (BDIM_X-1)));
    static_assert(0 == (BDIM_Y & (BDIM_Y-1)));
    static_assert((BDIM_X <= 32 && BDIM_Y  > 1) ||
                  (BDIM_X  > 32 && BDIM_Y == 1)) ;

    constexpr int NLOC_M1 = NLOC-1;


    const int tidx  = threadIdx.x;
    const int tidy  = threadIdx.y;

    const int batch = blockIdx.y;
    const int ctaid = blockIdx.x*blockDim.y + threadIdx.y;

    if (ctaid >= csr_nrow*nlon_out) {
        return;
    }

    const int h  = ctaid / nlon_out;
    const int wo = ctaid - (h*nlon_out);

    // set csr_row to "h" to bypass the row sorting
    const int csr_row = row_sort[h]; // h

    const int64_t rbeg = row_off[csr_row  ];
    const int64_t rend = row_off[csr_row+1];

    const int ho = row_idx[rbeg]; // reads only the first "nrow" rows of row_idx and only the first element of each row

    const int nchan_out = nchan_in*K;

    FLOATV_T locy[NLOC];

    x += int64_t(batch)*nlat_in*nlon_in*nchan_in;
    y += int64_t(batch)*nlat_out*nlon_out*nchan_out + int64_t(ho)*nlon_out*nchan_out + int64_t(wo)*nchan_out + tidx;

    #pragma unroll
    for(int i = 0; i < NLOC; i++) {
        locy[i] = __vset<FLOATV_T>(0.f);
    }

    col_idx += rbeg;
    val_pck += rbeg*K; // val_pck CSR contains K values per element

    const int rlen = rend-rbeg;

    constexpr int PAD = (BDIM_X < WARP_SIZE) ? 1 : 0;
    __shared__ const float *shXOffAll[BDIM_Y][BDIM_X+PAD];

    // check if BDIM_X is a multiple of K; since BDIM_X is a power of 2, check if K is also a power of two
    const int isKpow2 = !(K & (K-1));
    if (isKpow2) { processCSR_Kpow2_reg_d<BDIM_X, PAD, NLOC>(wo, rlen, nchan_in, nlon_in, pscale, K, x, col_idx, val_pck, shXOffAll[tidy], locy); }
    else         { processCSR_Kanyv_reg_d<BDIM_X, PAD, NLOC>(wo, rlen, nchan_in, nlon_in, pscale, K, x, col_idx, val_pck, shXOffAll[tidy], locy); }


    #pragma unroll
    for(int i = 0; i < NLOC_M1; i++) {
        y[i*BDIM_X] = locy[i];
    }
    if (NLOC_M1*BDIM_X+tidx < nchan_out) {
        y[NLOC_M1*BDIM_X] = locy[NLOC_M1];
    }

    return;
}

template<typename FLOATV_T>
void launch_gen_disco_fwd(int64_t batch_size,
                          int64_t nchan_in,
                          int64_t nlat_in,
                          int64_t nlon_in,
                          int64_t nlat_out,
                          int64_t nlon_out,
                          int64_t K,
                          float *__restrict__ _xp,
                          int64_t nrow,
                          int32_t *_row_sort,
                          int64_t *_row_off,
                          int64_t *_row_idx,
                          int64_t *_col_idx,
                          FLOATV_T *_val_pck,
                          FLOATV_T *__restrict__ _yp,
                          cudaStream_t stream) {

    dim3 block(WARP_SIZE, THREADS/WARP_SIZE);
    dim3 grid(DIV_UP(nrow*nlon_out, block.y), batch_size);

    size_t shsize = sizeof(FLOATV_T)*(nchan_in*K)*block.y;

    const int pscale = nlon_in / nlon_out;
#if 0
    printf("Launching s2_disco_fwd_generic_vec_k<%d, float%s><<<..., ..., %zu, ...>>> with:\n"
           "\tnchan_in: %ld\n"
           "\tK: %ld\n"
           "\tpscale: %d\n\n",
           THREADS, sizeof(FLOATV_T)==16?"4":"", shsize, nchan_in, K, pscale);
#endif
    // will use only the first 1/K-th of the CSR, i.e. only the first nlat_out rows
    s2_disco_fwd_generic_vec_k<THREADS>
                              <<<grid, block, shsize, stream>>>(nchan_in, nlat_in, nlon_in, nlat_out, nlon_out, pscale, K,
                                                                _xp, nrow, _row_sort, _row_off, _row_idx, _col_idx, _val_pck, _yp);
    CHECK_ERROR("s2_disco_fwd_generic_vec_k");

    return;
}

template<int BDIM_X,
         int CUR_LOC_SIZE,
         int MAX_LOC_SIZE, // max size of FLOATV_T[] local array
         typename FLOATV_T>
void launch_spc_disco_fwd(int nloc,      // "BDIM_X*nloc" >= nchans
                          int64_t batch_size,
                          int64_t nchan_in,
                          int64_t nlat_in,
                          int64_t nlon_in,
                          int64_t nlat_out,
                          int64_t nlon_out,
                          int64_t K,
                          float *__restrict__ _xp,
                          int64_t nrow,
                          int32_t *_row_sort,
                          int64_t *_row_off,
                          int64_t *_row_idx,
                          int64_t *_col_idx,
                          FLOATV_T *_val_pck,
                          FLOATV_T *__restrict__ _yp,
                          cudaStream_t stream) {

    if (CUR_LOC_SIZE == nloc) {

        constexpr int BDIM_Y = (BDIM_X <= WARP_SIZE) ? THREADS / BDIM_X : 1;

        dim3 block(BDIM_X, BDIM_Y);
        dim3 grid(DIV_UP(nrow*nlon_out, block.y), batch_size);

        size_t shsize = 0; //sizeof(float)*chxgrp_out * block.y;

        const int pscale = nlon_in / nlon_out;
#if 0
        printf("Launching s2_disco_fwd_special_vec_k<%d, %d, %d, float%s><<<(%d, %d), (%d, %d), ..., %zu, ...>>> with:\n"
               "\tnchan_in: %ld\n"
               "\tK: %ld\n"
               "\tpscale: %d\n\n",
               BDIM_X, BDIM_Y, CUR_LOC_SIZE, sizeof(FLOATV_T)==16?"4":"", grid.x, grid.y, block.x, block.y, shsize, nchan_in, K, pscale);
#endif
        s2_disco_fwd_special_vec_k<BDIM_X, BDIM_Y, CUR_LOC_SIZE>
                                  <<<grid, block, shsize, stream>>>(nchan_in, nlat_in, nlon_in, nlat_out, nlon_out, pscale, K,
                                                                    _xp, nrow, _row_sort, _row_off, _row_idx, _col_idx, _val_pck, _yp);

        CHECK_ERROR("s2_disco_fwd_special_vec_k");

        return;
    }
    if constexpr(CUR_LOC_SIZE < MAX_LOC_SIZE) {
         launch_spc_disco_fwd<BDIM_X,
                              CUR_LOC_SIZE+1,
                              MAX_LOC_SIZE>(nloc, batch_size, nchan_in, nlat_in, nlon_in, nlat_out, nlon_out,
                                            K, _xp, nrow, _row_sort, _row_off, _row_idx, _col_idx, _val_pck, _yp, stream);
    }
    return;
}

static void s2_disco_fwd_dispatch(int64_t batch_size,
                                  int64_t nchan_in,
                                  int64_t nlat_in,
                                  int64_t nlon_in,
                                  int64_t nlat_out,
                                  int64_t nlon_out,
                                  int64_t K,
                                  at::Tensor xP,
                                  at::Tensor row_off, // CSR non-empty row offsets
                                  at::Tensor row_idx, // CSR non-empty row indices
                                  at::Tensor col_idx, // CSR non-empty col indices
                                  at::Tensor val_dat, // CSR non-empty value data
                                  at::Tensor yP) {

    if (batch_size <= 0 ||
        nchan_in   <= 0 ||
        nlon_in    <= 0 ||
        nlat_out   <= 0 ||
        nlon_out   <= 0 ||
        K          <= 0) {

            fprintf(stderr,
                    ":%s:%d: invalid value of one or more input parameters!\n",
                    __FILE__, __LINE__);
            exit(EXIT_FAILURE);
    }

    // get stream
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // replace the K sequential CRSs in "val_dat":
    //
    //  val_dat[    0:  nnz/K) for ker = 0
    //  val_dat[nnz/K:2*nnz/K) for ker = 1
    //  ...
    //  val_dat[nnz/K:2*nnz/K) for ker = K-1
    //
    // with a packed CSR:
    //
    //  val_dat[nnz/K][K], i.e. with a CSR where elements of the original K CSRs are packed in consecutive elements
    assert(0 == (val_idx.size(0) % K));

    int64_t nrow_csr = row_off.size(0)-1;
    assert(0 == (nrow_csr % K));

    int64_t nrow = nrow_csr / K;

    // sort row indices (ho-s) in descending order
    // based on (row_off[ho+1]-row_off[ho])
    at::Tensor row_sort = sortRows(nrow, row_off, stream);

    // move into "disco_cuda_utils.cu" IF val_dat format won't be changed upstream in the call chain
    int64_t val_dims[] = {val_dat.size(0)};
    auto options = torch::TensorOptions().device(val_dat.device()).dtype(val_dat.dtype());
    torch::Tensor val_pck = torch::zeros(val_dims, options);
    {
        dim3 block(WARP_SIZE, THREADS/WARP_SIZE);
        dim3 grid(DIV_UP(nrow, block.y));

        pack_vals_k<<<grid, block, 0, stream>>>(K, nrow,
                                                row_off.data_ptr<int64_t>(),
                                                val_dat.data_ptr<float>(),
                                                val_pck.data_ptr<float>());
    }
    // if K is a multiple of VEC_SIZE it will be read with vector lds

    // smallest power of two "bdimx" (>=4) s.t. bdimx*MAX_LOCAL_ARR_LEN >= nchan_in*K
    int bdimx;
    bdimx = DIV_UP(nchan_in*K, MAX_LOCAL_ARR_LEN);
    bdimx = max(bdimx, WARP_SIZE/8); // min 4 threads per group
    bdimx = next_pow2(bdimx);

    float *_xp = reinterpret_cast<float *>(xP.data_ptr());
    float *_yp = reinterpret_cast<float *>(yP.data_ptr());

    int32_t *_row_sort = reinterpret_cast<int32_t *>(row_sort.data_ptr());
    int64_t *_row_off = reinterpret_cast<int64_t *>(row_off.data_ptr());
    int64_t *_row_idx = reinterpret_cast<int64_t *>(row_idx.data_ptr());
    int64_t *_col_idx = reinterpret_cast<int64_t *>(col_idx.data_ptr());
    float   *_val_pck = reinterpret_cast<float   *>(val_pck.data_ptr());

    constexpr int VEC_SIZE = sizeof(float4) / sizeof(float);

    if (!is_aligned<sizeof(float4)>(_yp)      ||
        !is_aligned<sizeof(float4)>(_val_pck) ||
        (K % VEC_SIZE) != 0) {

        const int nloc = DIV_UP(nchan_in*K, bdimx);

        // to avoid the compilation of unused template instances;
        // we use a block size BDIM_X that is the smallest power of 2
        // such that BDIM_X*MAX_LOCAL_ARR_LEN >= nchan_in*K, so
        // BDIM_X > 32 are used only for:
        //
        //  (BDIM_X-1)*MAX_LOCAL_ARR_LEN < nchan <= BDIM_X*MAX_LOCAL_ARR_LEN
        constexpr int MIN_LOCAL_ARR_LEN = MAX_LOCAL_ARR_LEN/2+1;

        // use 2D blocks only if 32 threads are enough
        switch(bdimx) {
            case    8: launch_spc_disco_fwd<   8,                 1, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchan_in, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, nrow, _row_sort, _row_off, _row_idx, _col_idx, _val_pck, _yp, stream); break;
            case   16: launch_spc_disco_fwd<  16, MIN_LOCAL_ARR_LEN, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchan_in, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, nrow, _row_sort, _row_off, _row_idx, _col_idx, _val_pck, _yp, stream); break;
            case   32: launch_spc_disco_fwd<  32, MIN_LOCAL_ARR_LEN, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchan_in, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, nrow, _row_sort, _row_off, _row_idx, _col_idx, _val_pck, _yp, stream); break;
            case   64: launch_spc_disco_fwd<  64, MIN_LOCAL_ARR_LEN, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchan_in, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, nrow, _row_sort, _row_off, _row_idx, _col_idx, _val_pck, _yp, stream); break;
            case  128: launch_spc_disco_fwd< 128, MIN_LOCAL_ARR_LEN, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchan_in, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, nrow, _row_sort, _row_off, _row_idx, _col_idx, _val_pck, _yp, stream); break;
            case  256: launch_spc_disco_fwd< 256, MIN_LOCAL_ARR_LEN, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchan_in, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, nrow, _row_sort, _row_off, _row_idx, _col_idx, _val_pck, _yp, stream); break;
            case  512: launch_spc_disco_fwd< 512, MIN_LOCAL_ARR_LEN, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchan_in, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, nrow, _row_sort, _row_off, _row_idx, _col_idx, _val_pck, _yp, stream); break;
            case 1024: launch_spc_disco_fwd<1024, MIN_LOCAL_ARR_LEN, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchan_in, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, nrow, _row_sort, _row_off, _row_idx, _col_idx, _val_pck, _yp, stream); break;
            default:   launch_gen_disco_fwd                                            (      batch_size, nchan_in, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, nrow, _row_sort, _row_off, _row_idx, _col_idx, _val_pck, _yp, stream); break;
        }

    } else {

        //float4 *_xp4 = reinterpret_cast<float4 *>(_xp);
        float4 *_yp4 = reinterpret_cast<float4 *>(_yp);

        float4 *_val_pck4 = reinterpret_cast<float4 *>(_val_pck);

        K /= VEC_SIZE;
        const int nloc = DIV_UP(nchan_in*K, bdimx);

        constexpr int MAX_LOCAL_VEC_LEN = MAX_LOCAL_ARR_LEN / VEC_SIZE;
        constexpr int MIN_LOCAL_VEC_LEN = MAX_LOCAL_VEC_LEN/2+1;

        // use 2D blocks only if 32 threads are enough
        switch(bdimx) {
            case    8: launch_spc_disco_fwd<   8,                 1, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchan_in, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, nrow, _row_sort, _row_off, _row_idx, _col_idx, _val_pck4, _yp4, stream); break;
            case   16: launch_spc_disco_fwd<  16, MIN_LOCAL_VEC_LEN, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchan_in, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, nrow, _row_sort, _row_off, _row_idx, _col_idx, _val_pck4, _yp4, stream); break;
            case   32: launch_spc_disco_fwd<  32, MIN_LOCAL_VEC_LEN, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchan_in, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, nrow, _row_sort, _row_off, _row_idx, _col_idx, _val_pck4, _yp4, stream); break;
            case   64: launch_spc_disco_fwd<  64, MIN_LOCAL_VEC_LEN, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchan_in, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, nrow, _row_sort, _row_off, _row_idx, _col_idx, _val_pck4, _yp4, stream); break;
            case  128: launch_spc_disco_fwd< 128, MIN_LOCAL_VEC_LEN, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchan_in, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, nrow, _row_sort, _row_off, _row_idx, _col_idx, _val_pck4, _yp4, stream); break;
            case  256: launch_spc_disco_fwd< 256, MIN_LOCAL_VEC_LEN, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchan_in, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, nrow, _row_sort, _row_off, _row_idx, _col_idx, _val_pck4, _yp4, stream); break;
            case  512: launch_spc_disco_fwd< 512, MIN_LOCAL_VEC_LEN, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchan_in, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, nrow, _row_sort, _row_off, _row_idx, _col_idx, _val_pck4, _yp4, stream); break;
            case 1024: launch_spc_disco_fwd<1024, MIN_LOCAL_VEC_LEN, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchan_in, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, nrow, _row_sort, _row_off, _row_idx, _col_idx, _val_pck4, _yp4, stream); break;
            default:   launch_gen_disco_fwd                                            (      batch_size, nchan_in, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, nrow, _row_sort, _row_off, _row_idx, _col_idx, _val_pck4, _yp4, stream); break;
        }
    }
    return;
}
// END VERSION WITH CHANNEL-LAST WITH 2D BLOCKS, 2ND DIM IDENTIFYING CHANNLES, NO EINSUM


    torch::Tensor disco_cuda_fwd(torch::Tensor inp, torch::Tensor roff_idx, torch::Tensor ker_idx, torch::Tensor row_idx,
                                 torch::Tensor col_idx, torch::Tensor val, int64_t K, int64_t Ho, int64_t Wo)
    {

        // some sanity checks
        CHECK_CUDA_INPUT_TENSOR(inp);
        CHECK_CUDA_INPUT_TENSOR(roff_idx);
        CHECK_CUDA_INPUT_TENSOR(ker_idx);
        CHECK_CUDA_INPUT_TENSOR(row_idx);
        CHECK_CUDA_INPUT_TENSOR(col_idx);
        CHECK_CUDA_INPUT_TENSOR(val);

        // assume input is B, H, W, C
        int64_t B = inp.size(0);
        int64_t Hi = inp.size(1);
        int64_t Wi = inp.size(2);
        int64_t C = inp.size(3);
        //int64_t nrows = roff_idx.size(0) - 1;

        // rename dimensions consistent with attention
        int64_t batch_size = B;
        int64_t nchan = C;
        int64_t nlat_in = Hi;
        int64_t nlon_in = Wi;
        int64_t nlat_out = Ho;
        int64_t nlon_out = Wo;

        // get stream
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        // create output tensor
        auto x_type = inp.dtype();
        auto xP = inp.to(torch::kFloat32).contiguous();

        // first, we split the input tensors along kernel dims, so that we work on K_1 = 2^k
        // kernels first and then compute the K - K_1 kernels later:
        int64_t log_K = ilog2(K);
        int64_t K_1 = pow2(log_K);
        int64_t K_2 = K - K_1;

        // to test before fusion
        torch::Tensor out;
        if (K == 1 || K_2 == 0) {
        //if (true) {
            int64_t out_dims[] = {batch_size, nlat_out, nlon_out, nchan*K};
            //auto options = torch::TensorOptions().device(inp.device()).dtype(inp.dtype());
            torch::Tensor yP = torch::zeros(out_dims, xP.options());

            // call channel-last kernel implementation
            s2_disco_fwd_dispatch(batch_size,
                                  nchan,
                                  nlat_in,
                                  nlon_in,
                                  nlat_out,
                                  nlon_out,
                                  K,
                                  xP,
                                  roff_idx,
                                  row_idx,
                                  col_idx,
                                  val,
                                  yP);

            out = yP.reshape({batch_size, nlat_out, nlon_out, nchan, K}).to(x_type);
        } else {
            // we need to split the psi tensors here:
            ker_idx = torch::reshape(ker_idx, {K, -1});
            row_idx = torch::reshape(row_idx, {K, -1});
            col_idx = torch::reshape(col_idx, {K, -1});
            val = torch::reshape(val, {K, -1});
            int64_t nrows = (roff_idx.size(0) - 1) / K;

            // now, perform computation on the first K_1 kernels:
            auto ker_idx_1 = ker_idx.narrow(0, 0, K_1).reshape({-1}).contiguous();
            auto row_idx_1 = row_idx.narrow(0, 0, K_1).reshape({-1}).contiguous();
            auto col_idx_1 = col_idx.narrow(0, 0, K_1).reshape({-1}).contiguous();
            auto val_1 = val.narrow(0, 0, K_1).reshape({-1}).contiguous();
            auto roff_idx_1 = roff_idx.narrow(0, 0, nrows * K_1 + 1).contiguous();
            
            // prepare output 1
            int64_t out_dims_1[] = {batch_size, nlat_out, nlon_out, nchan*K_1};
            torch::Tensor yP_1 = torch::zeros(out_dims_1, xP.options());

            // call channel-last kernel implementation
            s2_disco_fwd_dispatch(batch_size,
                nchan,
                nlat_in,
                nlon_in,
                nlat_out,
                nlon_out,
                K_1,
                xP,
                roff_idx_1,
                row_idx_1,
                col_idx_1,
                val_1,
                yP_1);

            yP_1 = yP_1.reshape({batch_size, nlat_out, nlon_out, nchan, K_1}).to(x_type);

            // now, perform computation on the remaining K_2 kernels:
            auto ker_idx_2 = ker_idx.narrow(0, K_1, K_2).reshape({-1}).contiguous() - K_1;
            auto row_idx_2 = row_idx.narrow(0, K_1, K_2).reshape({-1}).contiguous();
            auto col_idx_2 = col_idx.narrow(0, K_1, K_2).reshape({-1}).contiguous();
            auto val_2 = val.narrow(0, K_1, K_2).reshape({-1}).contiguous();
            auto roff_idx_2 = roff_idx.narrow(0, nrows * K_1, nrows * K_2 + 1).contiguous() - roff_idx_1.index({roff_idx_1.size(0) - 1});

            // prepare output
            int64_t out_dims_2[] = {batch_size, nlat_out, nlon_out, nchan*K_2};
            torch::Tensor yP_2 = torch::zeros(out_dims_2, xP.options());

            // call channel-last kernel implementation
            s2_disco_fwd_dispatch(batch_size,
                nchan,
                nlat_in,
                nlon_in,
                nlat_out,
                nlon_out,
                K_2,
                xP,
                roff_idx_2,
                row_idx_2,
                col_idx_2,
                val_2,
                yP_2);

            yP_2 = yP_2.reshape({batch_size, nlat_out, nlon_out, nchan, K_2}).to(x_type);

            // merge results
            out = torch::cat({yP_1, yP_2}, -1).contiguous();
        }

        return out;
    }

TORCH_LIBRARY_IMPL(disco_kernels, CUDA, m)
{
    m.impl("forward",  &disco_cuda_fwd);
}
}

