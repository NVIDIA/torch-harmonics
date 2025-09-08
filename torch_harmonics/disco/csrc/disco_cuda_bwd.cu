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

#include "disco.h"
#include "disco_cuda.cuh"
#include "csr_cuda.cuh"
#include "cudamacro.h"

#define THREADS (64)

#define MAX_LOCAL_ARR_LEN (16)

namespace disco_kernels {

using namespace utility_kernels;

template <int BDIM_X, int ELXTH, typename REAL_T>
__device__ void disco_bwd_d(const int Hi, const int Wi, const int K, const int Ho, const int Wo, const int pscale,
                            const int64_t *__restrict__ roff, const int64_t *__restrict__ kers,
                            const int64_t *__restrict__ rows, const int64_t *__restrict__ cols,
                            const REAL_T *__restrict__ vals, const REAL_T *__restrict__ inp, REAL_T *__restrict__ out)
{

    const int tid = threadIdx.x;

    const int64_t bidx = blockIdx.x; // gloabl row
    const int64_t bidy = blockIdx.y; // bc

    int64_t soff = roff[bidx];
    int64_t eoff = roff[bidx + 1];

    const int64_t ker = kers[soff];
    const int64_t row = rows[soff];

    inp += bidy * K * Hi * Wi + ker * Hi * Wi + row * Wi;
    out += bidy * Ho * Wo;

    // align to larger supported fp type
    extern __shared__ __align__(sizeof(double)) unsigned char __sh_ptr[]; // REAL_T __sh[2*(BDIM_X*ELXTH)*pscale]

    REAL_T(*__sh)[BDIM_X * ELXTH * 2] = reinterpret_cast<REAL_T(*)[BDIM_X * ELXTH * 2]>(__sh_ptr);

    // copy current inp row in regs
    REAL_T __reg[ELXTH];

#pragma unroll
    for (int i = 0; i < ELXTH; i++) { __reg[i] = (i * BDIM_X + tid < Wi) ? inp[i * BDIM_X + tid] : REAL_T(0); }

    // reset shared row up to Wo+2, remaining
    // ppscale*(BDIM_X*ELXTH - Wo) locations
    // will be written to but never copied to
    // global mem
    for (int i = 0; i < pscale; i++) {
#pragma unroll
        for (int j = 0; j < 2 * BDIM_X * ELXTH; j += BDIM_X) { __sh[i][j + tid] = 0; }
    }
    __syncthreads();

    int col_prev = cols[soff];

    int h_prev = col_prev / Wo;
    int w_prev = col_prev % Wo;

    // loops along the colums of CTA's row
    for (int64_t nz = soff; nz < eoff; nz++) {

        const int col = cols[nz];
        const REAL_T val = vals[nz];

        // if we are processing a nz with a col value
        // leading to a new row of inp then copy it
        // to shmem;
        // we read a col that points to a new output
        // row if (col / Wo) > (col_prev / Wo)
        if (col >= col_prev - w_prev + Wo) {
            __syncthreads();
            for (int i = 0; i < pscale; i++) {
                for (int j = tid; j < Wi; j += BDIM_X) {

                    const REAL_T v = __sh[i][j] + __sh[i][Wi + j];

                    atomicAdd(&out[h_prev * Wo + j * pscale + i], v);

                    __sh[i][j] = 0;
                    __sh[i][Wi + j] = 0;
                }
            }
            __syncthreads();

            col_prev = col;
            h_prev = col / Wo;
            w_prev = col % Wo;
        }

        const int w = w_prev + (col - col_prev);
        const int w_mod_ps = w % pscale;
        const int w_div_ps = w / pscale;

#pragma unroll
        for (int i = 0; i < ELXTH; i++) {

            const int pp = i * BDIM_X + tid;
            __sh[w_mod_ps][w_div_ps + pp] += val * __reg[i];
        }

        // to avoid race conditions on __sh[]
        // among consecutive iterations along nz
        __syncthreads();
    }
    __syncthreads();

    // write last row
    for (int i = 0; i < pscale; i++) {

        for (int j = tid; j < Wi; j += BDIM_X) {

            const REAL_T v = __sh[i][j] + __sh[i][Wi + j];
            atomicAdd(&out[h_prev * Wo + j * pscale + i], v);
        }
    }
    return;
}

template <int BDIM_X, int ELXTH, int PSCALE, typename REAL_T>
__global__
    __launch_bounds__(BDIM_X) void disco_bwd_blk_k(const int Hi, const int Wi, const int K, const int Ho, const int Wo,
                                                   const int pscale, const int64_t *__restrict__ roff,
                                                   const int64_t *__restrict__ kers, const int64_t *__restrict__ rows,
                                                   const int64_t *__restrict__ cols, const REAL_T *__restrict__ vals,
                                                   const REAL_T *__restrict__ inp, REAL_T *__restrict__ out)
{

    if constexpr (PSCALE != 0) {
        disco_bwd_d<BDIM_X, ELXTH>(Hi, Wi, K, Ho, Wo, PSCALE, roff, kers, rows, cols, vals, inp, out);
    } else {
        disco_bwd_d<BDIM_X, ELXTH>(Hi, Wi, K, Ho, Wo, pscale, roff, kers, rows, cols, vals, inp, out);
    }

    return;
}

template <int NTH, int ELXTH, typename REAL_T>
static void launch_kernel(int BC, int Hi, int Wi, int K, int Ho, int Wo, int64_t nrows, int64_t *roff_d, int64_t *ker_d,
                          int64_t *row_d, int64_t *col_d, REAL_T *val_d, REAL_T *inp_d, REAL_T *out_d,
                          cudaStream_t stream)
{

    static_assert(sizeof(REAL_T) == 2 || sizeof(REAL_T) == 4 || sizeof(REAL_T) == 8);

    if constexpr (ELXTH <= ELXTH_MAX) {
        if (NTH * ELXTH >= Wi) {
            dim3 grid(nrows, BC);

            const int pscale = Wo / Wi;
            size_t shmem = sizeof(*out_d) * (2 * (NTH * ELXTH) * pscale);

            switch (pscale) {
            case 1:
                disco_bwd_blk_k<NTH, ELXTH, 1><<<grid, NTH, shmem, stream>>>(Hi, Wi, K, Ho, Wo, pscale, roff_d, ker_d,
                                                                             row_d, col_d, val_d, inp_d, out_d);
                break;
            case 2:
                disco_bwd_blk_k<NTH, ELXTH, 2><<<grid, NTH, shmem, stream>>>(Hi, Wi, K, Ho, Wo, pscale, roff_d, ker_d,
                                                                             row_d, col_d, val_d, inp_d, out_d);
                break;
            case 3:
                disco_bwd_blk_k<NTH, ELXTH, 3><<<grid, NTH, shmem, stream>>>(Hi, Wi, K, Ho, Wo, pscale, roff_d, ker_d,
                                                                             row_d, col_d, val_d, inp_d, out_d);
                break;
            default:
                disco_bwd_blk_k<NTH, ELXTH, 0><<<grid, NTH, shmem, stream>>>(Hi, Wi, K, Ho, Wo, pscale, roff_d, ker_d,
                                                                             row_d, col_d, val_d, inp_d, out_d);
            }
        } else {
            launch_kernel<NTH, ELXTH + 1>(BC, Hi, Wi, K, Ho, Wo, nrows, roff_d, ker_d, row_d, col_d, val_d, inp_d,
                                          out_d, stream);
        }
    }
    return;
}

// BEGIN NEW CHANNEL-LAST VERSION

template<typename VAL_T>
static __global__ void pack_vals_k(const int64_t K,
                                   const int64_t nlat_out,
                                   const int64_t *__restrict__ row_off,
                                   const VAL_T *__restrict__ val_dat,
                                         VAL_T *__restrict__ val_pck) {

    const int tidx = threadIdx.x;
    const int wid = blockIdx.x*blockDim.y + threadIdx.y;
    if (wid >= nlat_out) {
        return;
    }

    const int64_t rbeg = row_off[wid];
    const int64_t rend = row_off[wid+1];

    const int rlen = rend-rbeg;

    val_pck += rbeg*K;

    for(int off = tidx; off < rlen; off += blockDim.x) {
        for(int ker = 0; ker < K; ker++) {

            val_pck[off*K + ker] = val_dat[ row_off[ker*nlat_out + wid]  + off];
        }
    }

    return;
}

template<int BDIM_X,
         typename FLOATV_T>
static __device__ void processCSR_Kpow2_shm_d(const int wi,
                                              const int rlen,
                                              const int nchans,  // no. of input FLOATV_T elements along channel dim
                                              const int nlon_out,
                                              const int pscale,
                                              const int K,
                                              const FLOATV_T *__restrict__ shx,
                                              const int64_t  *__restrict__ cols,
                                              const FLOATV_T *__restrict__ vals,
                                                    float *__restrict__ shy,
                                                    float *__restrict__ y) {
    const int tidx = threadIdx.x;

    const int log2_K = __ffs(K)-1;

    const int tidxDivK = tidx >> log2_K;
    const int tidxModK = tidx  & (K-1);
    
    vals += tidxModK;
    
    const int BDIMX_div_K = BDIM_X >> log2_K;

    for(int chan = tidx; chan < nchans; chan += WARP_SIZE) {
       shy[chan] = 0;
    }
    __syncwarp();

    for(int off = 0; off < rlen; off++) {

        const int64_t  col = cols[off];

        const int ho = col / nlon_out;
        const int wo = col - (ho*nlon_out);

        int wop = wo + pscale*wi;
        wop -= (wop / nlon_out)*nlon_out;

        float *_y = y + int64_t(ho)*nlon_out*nchans + int64_t(wop)*nchans;

        float *_shy = shy + tidxDivK;

        const FLOATV_T myval = vals[0];

        for(int i = 0; i < nchans*K; i += WARP_SIZE) {
    
            float sum = (i+tidx < nchans*K) ? __vred(__vmul(myval, shx[i+tidx])) : 0;

            for(int j = 1; j < K; j *= 2) {
                sum += __shfl_xor_sync(FULL_MASK, sum, j);
            }

            if (i+tidx < nchans*K && !tidxModK) {
                _shy[0] += sum;
            }
            _shy += BDIMX_div_K;
        }
        __syncwarp();

        for(int chan = tidx; chan < nchans; chan += WARP_SIZE) {
            atomicAdd(_y+chan, shy[chan]);
            shy[chan] = 0;
        }
        __syncwarp();

        vals += K;
    }

    return;
}

template<int BDIM_X,
         typename FLOATV_T>
static __device__ void processCSR_Kanyv_shm_d(const int wi,
                                              const int rlen,
                                              const int nchans,  // no. of input FLOATV_T elements along channel dim
                                              const int nlon_out,
                                              const int pscale,
                                              const int K,
                                              const FLOATV_T *__restrict__ shx,
                                              const int64_t  *__restrict__ cols,
                                              const FLOATV_T *__restrict__ vals,
                                                    float *__restrict__ shy,
                                                    float *__restrict__ y) {
    const int tidx = threadIdx.x;

    for(int chan = tidx; chan < nchans; chan += WARP_SIZE) {
       shy[chan] = 0;
    }
    __syncwarp();

    for(int off = 0; off < rlen; off++) {

        const int64_t  col = cols[off];

        const int ho = col / nlon_out;
        const int wo = col - (ho*nlon_out);

        int wop = wo + pscale*wi;
        wop -= (wop / nlon_out)*nlon_out;

        float *_y = y + int64_t(ho)*nlon_out*nchans + int64_t(wop)*nchans;

        for(int chan = tidx; chan < nchans*K; chan += WARP_SIZE) {

            const int cDivK = chan / K;
            const int cModK = chan - (cDivK*K);

            float sum = __vred(__vmul(vals[cModK], shx[chan]));

            atomicAdd(shy+cDivK, sum);
        }
        __syncwarp();

        for(int chan = tidx; chan < nchans; chan += WARP_SIZE) {
            atomicAdd(_y+chan, shy[chan]);
            shy[chan] = 0;
        }
        __syncwarp();

        vals += K;
    }

    return;
}

template<int BDIM, // change to BDIM_X(<=WARP_SIZE), BDIM_Y TO HANDLE SMALL CHXGR_IN
         typename FLOATV_T> // either float or float4
__global__
__launch_bounds__(BDIM)
void s2_disco_bwd_generic_vec_k(int nchans,   // no. of input  float (not FLOATV_T!) elements along channel dim
                                int nlat_in,
                                int nlon_in,
                                int nlat_out,
                                int nlon_out,
                                int pscale,
                                int K,          // no. of output FLOATV_T elem along K dim (kernel size)
                                const FLOATV_T *__restrict__ x,
                                const int32_t  *__restrict__ row_idx,
                                const int64_t  *__restrict__ row_off,
                                const int64_t  *__restrict__ col_idx,
                                const FLOATV_T *__restrict__ val_pck,
                                      float    *__restrict__ y) {

    constexpr int VEC_SIZE = sizeof(FLOATV_T) / sizeof(float);

    const int tidx  = threadIdx.x;

    const int batch = blockIdx.y;
    const int ctaid = blockIdx.x*blockDim.y + threadIdx.y;

    if (ctaid >= nlat_in*nlon_in) {
        return;
    }

#if 1
    const int h = ctaid / nlon_in;
    const int wi = ctaid - (h*nlon_in);
    const int hi = row_idx[h];
#else
    // for now don't use row_idx
    const int hi = ctaid / nlon_in;
    const int wi = ctaid - (hi*nlon_in);
#endif
    
    x += int64_t(batch)*nlat_in*nlon_in*nchans*K + int64_t(hi)*nlon_in*nchans*K + int64_t(wi)*nchans*K;
    y += int64_t(batch)*nlat_out*nlon_out*nchans;

    extern __shared__ __align__(sizeof(float4)) float shext[];
    FLOATV_T *shx = reinterpret_cast<FLOATV_T *>(shext + (VEC_SIZE*nchans*K + nchans)*threadIdx.y);
    float    *shy = reinterpret_cast<float *>(shx + nchans*K);

    for(int chan = tidx; chan < nchans*K; chan += WARP_SIZE) {
        shx[chan] = x[chan];
    }

    const int64_t rbeg = row_off[hi];
    const int64_t rend = row_off[hi+1];

    col_idx += rbeg;
    val_pck += rbeg*K; // val_pck CSR contains K values per element

    const int rlen = rend-rbeg;

    // check if BDIM_X is a multiple of K; since BDIM_X is a power of 2, check if K is also a power of two
    if (!(K & K-1) && K <= WARP_SIZE) { processCSR_Kpow2_shm_d<WARP_SIZE>(wi, rlen, nchans, nlon_out, pscale, K, shx, col_idx, val_pck, shy, y); }
    else                              { processCSR_Kanyv_shm_d<WARP_SIZE>(wi, rlen, nchans, nlon_out, pscale, K, shx, col_idx, val_pck, shy, y); }

    return;
}

template<int BDIM_X,
         int SHPAD,
         int NLOC,
         typename FLOATV_T>
static __device__ void processCSR_Kpow2_reg_d(const int wi,
                                              const int rlen,
                                              const int nchans,  // no. of input FLOATV_T elements along channel dim
                                              const int nlon_out,
                                              const int pscale,
                                              const int K,
                                              const FLOATV_T (&locx)[NLOC],
                                              const int64_t  *__restrict__ cols,
                                              const FLOATV_T *__restrict__ vals,
                                                    float *(&shYOff)[BDIM_X+SHPAD],
                                                    float *__restrict__ shy,       // NO LONGER USED
                                                    float *__restrict__ y) {
    constexpr int NLOC_M1 = NLOC-1;

    const int tidx = threadIdx.x;

    unsigned int subwarp_mask = FULL_MASK;

    if constexpr(BDIM_X <= WARP_SIZE) {
        const int tidy = threadIdx.y;
        constexpr unsigned int MASK = (1ull << BDIM_X)-1;
        subwarp_mask = MASK << (tidy*BDIM_X);
    }

    // K is a power of two <= BDIM_X
    const int log2_K = __popc(K-1);

    const int tidxDivK = tidx >> log2_K;
    const int tidxModK = tidx  & (K-1);
    
    cols += tidx;
    vals += tidxModK;

    const int BDIMX_div_K = BDIM_X >> log2_K;

    for(int off = 0; off < rlen; off++) {
        if ((off % BDIM_X) == 0) {
            __sync<BDIM_X>();

            const int64_t  col = (off+tidx < rlen) ? cols[0] : 0;

            const int ho = col / nlon_out;
            const int wo = col - (ho*nlon_out);

            int wop = wo + pscale*wi;
            wop -= (wop / nlon_out)*nlon_out;

            shYOff[tidx] = y + int64_t(ho)*nlon_out*nchans + int64_t(wop)*nchans;
            cols += BDIM_X;

            __sync<BDIM_X>();
        }

        float *_y = shYOff[off % BDIM_X] + tidxDivK;

        const FLOATV_T myval = vals[0];

        float locy[NLOC];
        
        #pragma unroll
        for(int i = 0; i < NLOC; i++) {
            locy[i] = __vred(__vmul(myval, locx[i]));
        }

        // K is a power of two <= 32
        #pragma unroll
        for(int j = 1; j < BDIM_X; j *= 2) {

            if (j >= K) break;

            #pragma unroll
            for(int i = 0; i < NLOC; i++) {
                locy[i] += __shfl_xor_sync(subwarp_mask, locy[i], j);
            }
        }
                
        if (!tidxModK) {
            // NLOC*BDIM_X >= nchans*K
            // NLOC_M1*BDIM_X < nchans*K => NLOC_M1*BDIM_X/K < nchans

            #pragma unroll
            for(int i = 0; i < NLOC_M1; i++) {
                atomicAdd(_y + i*BDIMX_div_K, locy[i]);
            }
            if (NLOC_M1*BDIM_X+tidx < nchans*K) {
                atomicAdd(_y + NLOC_M1*BDIMX_div_K, locy[NLOC_M1]);
            }
        }
        vals += K;
    }

    return;
}

#if 0
template<int BDIM_X,
         int SHPAD,
         int NLOC,
         typename FLOATV_T>
static __device__ void processCSR_Kpow2_reg_d2(const int wi,
                                              const int rlen,
                                              const int nchans,  // no. of input FLOATV_T elements along channel dim
                                              const int nlon_out,
                                              const int pscale,
                                              const int K,
                                              const FLOATV_T (&locx)[NLOC],
                                              const int64_t  *__restrict__ cols,
                                              const FLOATV_T *__restrict__ vals,
                                                    float *(&shYOff)[BDIM_X+SHPAD],
                                                    float *__restrict__ shy,       // NO LONGER USED
                                                    float *__restrict__ y) {
    constexpr int NLOC_M1 = NLOC-1;

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int log2_K = __ffs(K)-1;

    const int tidxDivK = tidx >> log2_K;
    const int tidxModK = tidx  & (K-1);
    
    vals += tidxModK;
    
    const int BDIMX_div_K = BDIM_X >> log2_K;
#if 1
    for(int chan = tidx; chan < nchans; chan += BDIM_X) {
       shy[chan] = 0;
    }
    __sync<BDIM_X>();
#endif

    cols += tidx;

    for(int off = 0; off < rlen; off++) {
        if ((off % BDIM_X) == 0) {
            __sync<BDIM_X>();

            const int64_t  col = (off+tidx < rlen) ? cols[0] : 0;

            const int ho = col / nlon_out;
            const int wo = col - (ho*nlon_out);

            int wop = wo + pscale*wi;
            wop -= (wop / nlon_out)*nlon_out;

            shYOff[tidx] = y + int64_t(ho)*nlon_out*nchans + int64_t(wop)*nchans;
            cols += BDIM_X;

            __sync<BDIM_X>();
        }

        float *_y = shYOff[off % BDIM_X];// + tidxDivK;
        float *_shy = shy + tidxDivK;

        const FLOATV_T myval = vals[0];

        float locy[NLOC];
        
        #pragma unroll
        for(int i = 0; i < NLOC; i++) {
            locy[i] = __vred(__vmul(myval, locx[i]));
        }

        #pragma unroll
        for(int i = 0; i < NLOC; i++) {

            // K is a power of two <= 32
            for(int j = 1; j < K; j *= 2) {
                locy[i] += __shfl_xor_sync(FULL_MASK, locy[i], j);
            }
        }
                
        if (!tidxModK) {
            // NLOC*BDIM_X >= nchans*K
            // NLOC_M1*BDIM_X < nchans*K => NLOC_M1*BDIM_X/K < nchans
            #pragma unroll
            for(int i = 0; i < NLOC; i++) {
                _shy[i*BDIMX_div_K] += locy[i];
            }
           // if (NLOC_M1*BDIM_X+tidx < nchans*K) {
           //     _shy[NLOC_M1*BDIMX_div_K] += locy[NLOC_M1];
           // }
        }
        __sync<BDIM_X>();

        for(int chan = tidx; chan < nchans; chan += BDIM_X) {
            atomicAdd(_y+chan, shy[chan]);
            shy[chan] = 0;
        }
        __sync<BDIM_X>();

        vals += K;
    }

    return;
}
#endif

template<int BDIM_X,
         int SHPAD,
         int NLOC,
         typename FLOATV_T>
static __device__ void processCSR_Kanyv_reg_d(const int wi,
                                              const int rlen,
                                              const int nchans,  // no. of input FLOATV_T elements along channel dim
                                              const int nlon_out,
                                              const int pscale,
                                              const int K,
                                              const FLOATV_T (&locx)[NLOC],
                                              const int64_t  *__restrict__ cols,
                                              const FLOATV_T *__restrict__ vals,
                                                    float *(&shYOff)[BDIM_X+SHPAD],
                                                    float *__restrict__ shy,
                                                    float *__restrict__ y) {
    const int tidx = threadIdx.x;

    for(int chan = tidx; chan < nchans; chan += BDIM_X) {
       shy[chan] = 0;
    }
    __sync<BDIM_X>();

    cols += tidx;

    for(int off = 0; off < rlen; off++) {

        if ((off % BDIM_X) == 0) {
            __sync<BDIM_X>();

            const int64_t  col = (off+tidx < rlen) ? cols[0] : 0;

            const int ho = col / nlon_out;
            const int wo = col - (ho*nlon_out);

            int wop = wo + pscale*wi;
            wop -= (wop / nlon_out)*nlon_out;

            shYOff[tidx] = y + int64_t(ho)*nlon_out*nchans + int64_t(wop)*nchans;
            cols += BDIM_X;

            __sync<BDIM_X>();
        }

        float *_y = shYOff[off % BDIM_X];

        // shy is allocated as ceil(nchans / (BDIM_X/K))*(BDIM_X/K)
        // so we can just loop NLOC timss
        #pragma unroll
        for(int i = 0; i < NLOC; i++) {
            
            const int chan = i*BDIM_X+tidx;
            const int cDivK = chan / K;
            const int cModK = chan - (cDivK*K);

            float sum = __vred(__vmul(vals[cModK], locx[i]));

            atomicAdd(shy+cDivK, sum);
        }
        __sync<BDIM_X>();

        for(int chan = tidx; chan < nchans; chan += BDIM_X) {
            atomicAdd(_y+chan, shy[chan]);
            shy[chan] = 0;
        }
        __sync<BDIM_X>();

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
void s2_disco_bwd_special_vec_k(int nchans,   // no. of input  float (not FLOATV_T!) elements along channel dim
                                int nlat_in,
                                int nlon_in,
                                int nlat_out,
                                int nlon_out,
                                int pscale,
                                int K,          // no. of output FLOATV_T elem along K dim (kernel size)
                                const FLOATV_T *__restrict__ x,
                                const int32_t  *__restrict__ row_idx,
                                const int64_t  *__restrict__ row_off,
                                const int64_t  *__restrict__ col_idx,
                                const FLOATV_T *__restrict__ val_pck,
                                      float    *__restrict__ y) {

    static_assert(0 == (BDIM_X & (BDIM_X-1)));
    static_assert(0 == (BDIM_Y & (BDIM_Y-1)));
    static_assert((BDIM_X <= 32 && BDIM_Y  > 1) ||
                  (BDIM_X  > 32 && BDIM_Y == 1)) ;

    constexpr int NLOC_M1 = NLOC-1;

    const int tidx  = threadIdx.x;
    const int tidy  = threadIdx.y;

    const int batch = blockIdx.y;
    const int ctaid = blockIdx.x*blockDim.y + threadIdx.y;

    if (ctaid >= nlat_in*nlon_in) {
        return;
    }

#if 1
    const int h = ctaid / nlon_out;
    const int wi = ctaid - (h*nlon_out);
    const int hi = row_idx[h];
#else
    // for now don't use row_idx
    const int hi = ctaid / nlon_out;
    const int wi = ctaid - (hi*nlon_out);
#endif

    x += int64_t(batch)*nlat_in*nlon_in*nchans*K + int64_t(hi)*nlon_in*nchans*K + int64_t(wi)*nchans*K + tidx;
    y += int64_t(batch)*nlat_out*nlon_out*nchans;

    FLOATV_T locx[NLOC];
    
    #pragma unroll
    for(int i = 0; i < NLOC_M1; i++) {
        locx[i] = x[i*BDIM_X];
    }
    locx[NLOC_M1] = __vset<FLOATV_T>(0.0f);
    if (NLOC_M1*BDIM_X+tidx < nchans*K) {
        locx[NLOC_M1] = x[NLOC_M1*BDIM_X];
    }

    // only used if K is not a multiple of 2
    extern __shared__ __align__(sizeof(float4)) float shext[];
    float *shy = shext + nchans*threadIdx.y;

    const int64_t rbeg = row_off[hi];
    const int64_t rend = row_off[hi+1];

    col_idx += rbeg;
    val_pck += rbeg*K; // val_pck CSR contains K values per element

    const int rlen = rend-rbeg;
    
    constexpr int PAD = (BDIM_X < WARP_SIZE) ? 1 : 0;
    __shared__ float *shYOffAll[BDIM_Y][BDIM_X+PAD];

    // check if BDIM_X is a multiple of K; since BDIM_X is a power of 2, check if K is also a power of two
    if (!(K & K-1) && K <= BDIM_X) { processCSR_Kpow2_reg_d<BDIM_X, PAD, NLOC>(wi, rlen, nchans, nlon_out, pscale, K, locx, col_idx, val_pck, shYOffAll[tidy], NULL, y); }
    else                           { processCSR_Kanyv_reg_d<BDIM_X, PAD, NLOC>(wi, rlen, nchans, nlon_out, pscale, K, locx, col_idx, val_pck, shYOffAll[tidy],  shy, y); }

    return;

}

template<typename FLOATV_T>
void launch_gen_disco_bwd(int64_t batch_size,
                          int64_t nchans,
                          int64_t nlat_in,
                          int64_t nlon_in,
                          int64_t nlat_out,
                          int64_t nlon_out,
                          int64_t K,
                          FLOATV_T *__restrict__ _xp,
                          int32_t *_row_idx,
                          int64_t *_row_off,
                          int64_t *_col_idx,
                          FLOATV_T *_val_pck,
                          float *__restrict__ _yp,
                          cudaStream_t stream) {

    dim3 block(WARP_SIZE, THREADS/WARP_SIZE);
    dim3 grid(DIV_UP(nlat_in*nlon_in, block.y), batch_size);

    size_t shsize = (sizeof(FLOATV_T)*(nchans*K) + sizeof(float)*nchans)*block.y;

    const int pscale = nlon_out / nlon_in;
#if 1
    printf("Launching s2_disco_bwd_generic_vec_k<%d, float%s><<<..., ..., %zu, ...>>> with:\n"
           "\tnchan_out: %ld\n"
           "\tK: %ld\n\n",
           THREADS, sizeof(FLOATV_T)==16?"4":"", shsize, nchans, K);
#endif
    // will use only the first 1/K-th of the CSR, i.e. only the first nlat_out rows
    s2_disco_bwd_generic_vec_k<THREADS>
                              <<<grid, block, shsize, stream>>>(nchans, nlat_in, nlon_in, nlat_out, nlon_out, pscale, K,
                                                                _xp, _row_idx, _row_off, _col_idx, _val_pck, _yp);
    CHECK_ERROR("s2_disco_bwd_generic_vec_k");

    return;
}

template<int BDIM_X,
         int CUR_LOC_SIZE,
         int MAX_LOC_SIZE, // max size of FLOATV_T[] local array
         typename FLOATV_T>
void launch_spc_disco_bwd(int nloc,      // "BDIM_X*nloc" >= nchans
                          int64_t batch_size,
                          int64_t nchans,
                          int64_t nlat_in,
                          int64_t nlon_in,
                          int64_t nlat_out,
                          int64_t nlon_out,
                          int64_t K,
                          FLOATV_T *__restrict__ _xp,
                          int32_t *_row_idx,
                          int64_t *_row_off,
                          int64_t *_col_idx,
                          FLOATV_T *_val_pck,
                          float *__restrict__ _yp,
                          cudaStream_t stream) {

    if (CUR_LOC_SIZE == nloc) {

        // block size set to 64 threads
        constexpr int BDIM_Y = (BDIM_X <= WARP_SIZE) ? THREADS / BDIM_X : 1;

        // groups in gridDim.y
        dim3 block(BDIM_X, BDIM_Y);
        dim3 grid(DIV_UP(nlat_out*nlon_out, block.y), batch_size);

        // could be (BDIM_X/K) instead of BDIM_X but let's keep it simple
        size_t shsize = (K & (K-1)) ? sizeof(float)*DIV_UP(nchans, BDIM_X)*BDIM_X*block.y : 0;
        //size_t shsize = sizeof(float)*DIV_UP(nchans, BDIM_X)*BDIM_X*block.y;

        const int pscale = nlon_out / nlon_in;
#if 1
        printf("Launching s2_disco_bwd_special_vec_k<%d, %d, %d, float%s><<<(%d, %d), (%d, %d), ..., %zu, ...>>> with:\n"
               "\tnchans: %ld\n"
               "\tK: %ld\n\n",
               BDIM_X, BDIM_Y, CUR_LOC_SIZE, sizeof(FLOATV_T)==16?"4":"", grid.x, grid.y, block.x, block.y, shsize, nchans, K);
#endif
        s2_disco_bwd_special_vec_k<BDIM_X, BDIM_Y, CUR_LOC_SIZE>
                                  <<<grid, block, shsize, stream>>>(nchans, nlat_in, nlon_in, nlat_out, nlon_out, pscale, K,
                                                                    _xp, _row_idx, _row_off, _col_idx, _val_pck, _yp);

        CHECK_ERROR("s2_disco_bwd_special_vec_k");

        return;
    }
    if constexpr(CUR_LOC_SIZE < MAX_LOC_SIZE) {
         launch_spc_disco_bwd<BDIM_X,
                              CUR_LOC_SIZE+1,
                              MAX_LOC_SIZE>(nloc, batch_size, nchans, nlat_in, nlon_in, nlat_out, nlon_out,
                                            K, _xp, _row_idx, _row_off, _col_idx, _val_pck, _yp, stream);
    }
    return;
}

static void s2_disco_bwd_dispatch(int64_t batch_size,
                                  int64_t nchans,
                                  int64_t nlat_in,
                                  int64_t nlon_in,
                                  int64_t nlat_out,
                                  int64_t nlon_out,
                                  int64_t K,
                                  at::Tensor xP,
                                  at::Tensor row_off, // CSR row offsets
                                  at::Tensor col_idx, // CSR column indices
                                  at::Tensor val_dat, // CSR value data
                                  at::Tensor yP) {

    static_assert(0 == (MAX_LOCAL_ARR_LEN & (MAX_LOCAL_ARR_LEN-1)));

    if (batch_size <=         0 ||
        nchans     <=         0 ||
        nlon_in    <=         0 ||
        nlat_out   <=         0 ||
        nlon_out   <=         0 ||
        K          <=         0 ||
        K           > WARP_SIZE) {

            fprintf(stderr,
                    ":%s:%d: invalid value of one or more input parameters!\n", 
                    __FILE__, __LINE__);
            exit(EXIT_FAILURE);
    }

    // get stream
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // sort row indices (ho-s) in descending order
    // based on (row_off[ho+1]-row_off[ho])
    at::Tensor row_idx = sortRows(nlat_in, row_off, stream);


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

    // move into "disco_cuda_utils.cu" IF val_dat format won't be changed upstream in the call chain
    int64_t val_dims[] = {val_dat.size(0)};
    auto options = torch::TensorOptions().device(val_dat.device()).dtype(val_dat.dtype());
    torch::Tensor val_pck = torch::zeros(val_dims, options);
    {
        dim3 block(WARP_SIZE, THREADS/WARP_SIZE);
        dim3 grid(DIV_UP(nlat_out, block.y));

        pack_vals_k<<<grid, block, 0, stream>>>(K, nlat_out,
                                                row_off.data_ptr<int64_t>(),
                                                val_dat.data_ptr<float>(),
                                                val_pck.data_ptr<float>());
    }
    // if K is a multiple of VEC_SIZE it will be read with vector lds

    // smallest power of two "bdimx" (>=4) s.t. bdimx*MAX_LOCAL_ARR_LEN >= nchans*K
    int bdimx;
    bdimx = DIV_UP(nchans*K, MAX_LOCAL_ARR_LEN);
    bdimx = max(bdimx, WARP_SIZE/8); // min 4 threads per group
    bdimx = next_pow2(bdimx);

    float *_xp = reinterpret_cast<float *>(xP.data_ptr());
    float *_yp = reinterpret_cast<float *>(yP.data_ptr());

    int32_t *_row_idx = reinterpret_cast<int32_t *>(row_idx.data_ptr());
    int64_t *_row_off = reinterpret_cast<int64_t *>(row_off.data_ptr());
    int64_t *_col_idx = reinterpret_cast<int64_t *>(col_idx.data_ptr());
    float   *_val_pck = reinterpret_cast<float   *>(val_pck.data_ptr());

    constexpr int VEC_SIZE = sizeof(float4) / sizeof(float);

    if (!is_aligned<sizeof(float4)>(_yp)      ||
        !is_aligned<sizeof(float4)>(_xp)      ||
        !is_aligned<sizeof(float4)>(_val_pck) ||
        (K % VEC_SIZE) != 0) {

        //printf("%s:%d: VEC_SIZE: %d, nchans: %d, K: %d, _xp: %p, _yp: %p\n", __func__, __LINE__, VEC_SIZE, nchans, K, _xp, _yp);

        const int nloc = DIV_UP(nchans*K, bdimx);
        
        // to avoid the compilation of unused template instances;
        // we use a block size BDIM_X that is the smallest power of 2
        // such that BDIM_X*MAX_LOCAL_ARR_LEN >= nchans*K, so
        // BDIM_X > 32 are used only for:
        //
        //  (BDIM_X-1)*MAX_LOCAL_ARR_LEN < nchan <= BDIM_X*MAX_LOCAL_ARR_LEN
        constexpr int MIN_LOC_ARR_LEN = MAX_LOCAL_ARR_LEN/2+1;

        // use 2D blocks only if 32 threads are enough
        switch(bdimx) {
            case    8: launch_spc_disco_bwd<   8,               1, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchans, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, _row_idx, _row_off, _col_idx, _val_pck, _yp, stream); break;
            case   16: launch_spc_disco_bwd<  16, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchans, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, _row_idx, _row_off, _col_idx, _val_pck, _yp, stream); break;
            case   32: launch_spc_disco_bwd<  32, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchans, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, _row_idx, _row_off, _col_idx, _val_pck, _yp, stream); break;
            case   64: launch_spc_disco_bwd<  64, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchans, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, _row_idx, _row_off, _col_idx, _val_pck, _yp, stream); break;
            case  128: launch_spc_disco_bwd< 128, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchans, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, _row_idx, _row_off, _col_idx, _val_pck, _yp, stream); break;
            case  256: launch_spc_disco_bwd< 256, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchans, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, _row_idx, _row_off, _col_idx, _val_pck, _yp, stream); break;
            case  512: launch_spc_disco_bwd< 512, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchans, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, _row_idx, _row_off, _col_idx, _val_pck, _yp, stream); break;
            case 1024: launch_spc_disco_bwd<1024, MIN_LOC_ARR_LEN, MAX_LOCAL_ARR_LEN>(nloc, batch_size, nchans, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, _row_idx, _row_off, _col_idx, _val_pck, _yp, stream); break;
            default:   launch_gen_disco_bwd                                          (      batch_size, nchans, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp, _row_idx, _row_off, _col_idx, _val_pck, _yp, stream); break;
        }

    } else {

        //printf("%s:%d: VEC_SIZE: %d, nchans: %d, K: %d, _xp: %p, _yp: %p\n", __func__, __LINE__, VEC_SIZE, nchans, K, _xp, _yp);

        float4 *_xp4 = reinterpret_cast<float4 *>(_xp);
        //float4 *_yp4 = reinterpret_cast<float4 *>(_yp);

        float4 *_val_pck4 = reinterpret_cast<float4 *>(_val_pck);

        K /= VEC_SIZE;
        const int nloc = DIV_UP(nchans*K, bdimx);

        constexpr int MAX_LOCAL_VEC_LEN = MAX_LOCAL_ARR_LEN / VEC_SIZE;
        constexpr int MIN_LOC_VEC_LEN = MAX_LOCAL_VEC_LEN/2+1;

        // use 2D blocks only if 32 threads are enough
        switch(bdimx) {
            case    8: launch_spc_disco_bwd<   8,               1, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchans, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp4, _row_idx, _row_off, _col_idx, _val_pck4, _yp, stream); break;
            case   16: launch_spc_disco_bwd<  16, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchans, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp4, _row_idx, _row_off, _col_idx, _val_pck4, _yp, stream); break;
            case   32: launch_spc_disco_bwd<  32, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchans, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp4, _row_idx, _row_off, _col_idx, _val_pck4, _yp, stream); break;
            case   64: launch_spc_disco_bwd<  64, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchans, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp4, _row_idx, _row_off, _col_idx, _val_pck4, _yp, stream); break;
            case  128: launch_spc_disco_bwd< 128, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchans, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp4, _row_idx, _row_off, _col_idx, _val_pck4, _yp, stream); break;
            case  256: launch_spc_disco_bwd< 256, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchans, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp4, _row_idx, _row_off, _col_idx, _val_pck4, _yp, stream); break;
            case  512: launch_spc_disco_bwd< 512, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchans, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp4, _row_idx, _row_off, _col_idx, _val_pck4, _yp, stream); break;
            case 1024: launch_spc_disco_bwd<1024, MIN_LOC_VEC_LEN, MAX_LOCAL_VEC_LEN>(nloc, batch_size, nchans, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp4, _row_idx, _row_off, _col_idx, _val_pck4, _yp, stream); break;
            default:   launch_gen_disco_bwd                                          (      batch_size, nchans, nlat_in, nlon_in, nlat_out, nlon_out, K, _xp4, _row_idx, _row_off, _col_idx, _val_pck4, _yp, stream); break;
        }
    }
    return;
}

// END NEW CHANNEL-LAST VERSION
    torch::Tensor disco_cuda_bwd(torch::Tensor ograd, torch::Tensor roff_idx, torch::Tensor ker_idx, torch::Tensor row_idx,
                                 torch::Tensor col_idx, torch::Tensor val, int64_t K, int64_t Ho, int64_t Wo)
    {

        // some sanity checks
        CHECK_CUDA_INPUT_TENSOR(ograd);
        CHECK_CUDA_INPUT_TENSOR(roff_idx);
        CHECK_CUDA_INPUT_TENSOR(ker_idx);
        CHECK_CUDA_INPUT_TENSOR(row_idx);
        CHECK_CUDA_INPUT_TENSOR(col_idx);
        CHECK_CUDA_INPUT_TENSOR(val);
#if 0
        // extract some shapes
        int64_t B = ograd.size(0);
        int64_t Hi = ograd.size(1);
        int64_t Wi = ograd.size(2);
        int64_t C = ograd.size(3);
        int64_t BC = B * C;
        int64_t nrows = roff_idx.size(0) - 1;

        // allocate output
        int64_t out_dims[] = {B, Ho, Wo, C};

        // get stream
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        // extract dtype
        auto x_type = ograd.dtype();
        torch::Tensor xP = ograd.to(torch::kFloat32);

        torch::Tensor igrad = torch::zeros(out_dims, xP.options());

        // assert
        static_assert(0 == (ELXTH_MAX % 2));

        if (Wo <= 64 * ELXTH_MAX) {
            AT_DISPATCH_FLOATING_TYPES(xP.scalar_type(), "disco_backward_cuda", ([&] {
                                           launch_kernel<64, 1, scalar_t>(
                                               BC, Hi, Wi, K, Ho, Wo, nrows, roff_idx.data_ptr<int64_t>(),
                                               ker_idx.data_ptr<int64_t>(), row_idx.data_ptr<int64_t>(),
                                               col_idx.data_ptr<int64_t>(), val.data_ptr<scalar_t>(),
                                               xP.data_ptr<scalar_t>(), igrad.data_ptr<scalar_t>(), stream);
                                       }));
        } else if (Wo <= 128 * ELXTH_MAX) {
            AT_DISPATCH_FLOATING_TYPES(xP.scalar_type(), "disco_backward_cuda", ([&] {
                                           launch_kernel<128, (ELXTH_MAX / 2) + 1, scalar_t>(
                                               BC, Hi, Wi, K, Ho, Wo, nrows, roff_idx.data_ptr<int64_t>(),
                                               ker_idx.data_ptr<int64_t>(), row_idx.data_ptr<int64_t>(),
                                               col_idx.data_ptr<int64_t>(), val.data_ptr<scalar_t>(),
                                               xP.data_ptr<scalar_t>(), igrad.data_ptr<scalar_t>(), stream);
                                        }));
        } else if (Wo <= 256 * ELXTH_MAX) {
            AT_DISPATCH_FLOATING_TYPES(xP.scalar_type(), "disco_backward_cuda", ([&] {
                                           launch_kernel<256, (ELXTH_MAX / 2) + 1, scalar_t>(
                                               BC, Hi, Wi, K, Ho, Wo, nrows, roff_idx.data_ptr<int64_t>(),
                                               ker_idx.data_ptr<int64_t>(), row_idx.data_ptr<int64_t>(),
                                               col_idx.data_ptr<int64_t>(), val.data_ptr<scalar_t>(),
                                               xP.data_ptr<scalar_t>(), igrad.data_ptr<scalar_t>(), stream);
                                       }));
        } else if (Wo <= 512 * ELXTH_MAX) {
            AT_DISPATCH_FLOATING_TYPES(xP.scalar_type(), "disco_backward_cuda", ([&] {
                                           launch_kernel<512, (ELXTH_MAX / 2) + 1, scalar_t>(
                                               BC, Hi, Wi, K, Ho, Wo, nrows, roff_idx.data_ptr<int64_t>(),
                                               ker_idx.data_ptr<int64_t>(), row_idx.data_ptr<int64_t>(),
                                               col_idx.data_ptr<int64_t>(), val.data_ptr<scalar_t>(),
                                               xP.data_ptr<scalar_t>(), igrad.data_ptr<scalar_t>(), stream);
                                       }));
        } else if (Wo <= 1024 * ELXTH_MAX) {
            AT_DISPATCH_FLOATING_TYPES(xP.scalar_type(), "disco_backward_cuda", ([&] {
                                         launch_kernel<1024, (ELXTH_MAX / 2) + 1, scalar_t>(
                                               BC, Hi, Wi, K, Ho, Wo, nrows, roff_idx.data_ptr<int64_t>(),
                                               ker_idx.data_ptr<int64_t>(), row_idx.data_ptr<int64_t>(),
                                               col_idx.data_ptr<int64_t>(), val.data_ptr<scalar_t>(),
                                               xP.data_ptr<scalar_t>(), igrad.data_ptr<scalar_t>(), stream);
                                       }));
        } else {
            fprintf(stderr, "%s:%d: error, unsupported Wo value (%ld), max supported is %d\n", __FILE__, __LINE__, Wo,
                    1024 * ELXTH_MAX);
            exit(EXIT_FAILURE);
        }
#else
        // extract some shapes
        int64_t batch_size = ograd.size(0);
        int64_t nlat_in = ograd.size(1);
        int64_t nlon_in = ograd.size(2);
        int64_t Co = ograd.size(3);
        int64_t nchan = Co * K;
        int64_t nrows = roff_idx.size(0) - 1;

        printf("%s:%d: batch_size: %ld, nlat_in: %ld, nlon_in: %ld, nchan: %ld, K: %ld\n", 
               __func__, __LINE__, batch_size, nlat_in, nlon_in, nchan, K);
        
        int64_t nlat_out = Ho;
        int64_t nlon_out = Wo;

        // allocate output
        int64_t out_dims[] = {batch_size, Ho, Wo, Co};

        // get stream
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        // extract dtype
        auto x_type = ograd.dtype();
        torch::Tensor xP = ograd.reshape({batch_size, nlat_in, nlon_in, nchan}).to(torch::kFloat32).contiguous();

        torch::Tensor igrad = torch::zeros(out_dims, xP.options());
        
#if 0
        printf("%s:%d: tensors info:\n", __func__, __LINE__);
        printf("\tbatch_size: %ld\n", batch_size);
        printf("\t   nlat_in: %ld\n", nlat_in);
        printf("\t   nlon_in: %ld\n", nlon_in);
        printf("\t         C: %ld\n", nchan);
        printf("\t         K: %ld\n\n", K);
        printf("\troff_idx.size(0)-1 == nlat_in*K: %d\n", roff_idx.size(0)-1 == nlat_in*K);
        //printf("\tinp channle-last: %d\n", x_is_channels_last);
        printf("\treshaped inp to: {%ld, %ld, %ld, %ld}\n", xP.size(0), xP.size(1), xP.size(2), xP.size(3));
        fflush(stdout);

        //exit(1);
#endif
        
        // call channel-last kernel implementation
        s2_disco_bwd_dispatch(batch_size,
                              Co, //nchan,
                              nlat_in,
                              nlon_in,
                              nlat_out,
                              nlon_out,
                              K,
                              xP,
                              roff_idx,
                              col_idx,
                              val,
                              igrad);
/*
        // switch back to original layout;
        torch::Tensor out = yP;
        if (!x_is_channels_last) { 
            out = permute_4D_to0312(yP);
            // make y {batch_size, nchan, nlat_out, nlon_out}
        }
*/
#endif
        // convert back to original dtype
        igrad = igrad.to(x_type);

        return igrad;
    }

    TORCH_LIBRARY_IMPL(disco_kernels, CUDA, m)
    {
        m.impl("backward",  &disco_cuda_bwd);
    }

}
