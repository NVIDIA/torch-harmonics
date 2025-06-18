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

#include "attention.cuh"
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/CUDAUtils.h>

#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

#define WARP_SIZE (32)
#define FULL_MASK (0xFFFFFFFF)
#define THREADS (64)
#define DIV_UP(a,b) (((a)+((b)-1))/(b))

#define TRANSP_WARPS_X_TILE_GENERIC (32)
#define TRANSP_WARPS_X_TILE_SM100    (4)

#define MAX_LOCAL_ARR_LEN (16)

#define CHECK_CUDA(call) {                                          \
    cudaError_t err = call;                                         \
    if( cudaSuccess != err) {                                       \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
              __FILE__, __LINE__, cudaGetErrorString( err) );       \
      exit(EXIT_FAILURE);                                           \
    }}

#define CHECK_ERROR(errorMessage) {                                     \
    cudaError_t err = cudaGetLastError();                               \
    if( cudaSuccess != err) {                                           \
      fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n", \
              errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) ); \
      exit(EXIT_FAILURE);                                               \
    }}


// BEGIN - forward kernels and functions

template<typename VAL_T>
__device__ VAL_T __warp_sum(VAL_T val) {

    #pragma unroll
    for(int i = WARP_SIZE/2; i; i /= 2) {
        val += __shfl_xor_sync(FULL_MASK, val, i);
    }
    return val;
}

template<int BDIM_X,
         int BDIM_Y=1,
         int BDIM_Z=1,
         typename VAL_T>
__device__ VAL_T __block_sum(VAL_T val) {

    const int NWARP = (BDIM_X*BDIM_Y*BDIM_Z) / WARP_SIZE;

    val = __warp_sum(val);

    if constexpr(NWARP > 1) {

        int tid = threadIdx.x;
        if constexpr(BDIM_Y > 1) { tid += threadIdx.y*BDIM_X;        }
        if constexpr(BDIM_Z > 1) { tid += threadIdx.z*BDIM_X*BDIM_Y; }

        const int lid = tid%WARP_SIZE;
        const int wid = tid/WARP_SIZE;

        __shared__ VAL_T sh[NWARP];

        if (lid == 0) {
            sh[wid] = val;
        }
        __syncthreads();

        if (wid == 0) {
            val = (lid < NWARP) ? sh[lid] : 0;

            val = __warp_sum(val);
            __syncwarp();

            if (!lid) {
                sh[0] = val;
            }
        }
        __syncthreads();

        val = sh[0];
        __syncthreads();
    }
    return val;
}

template<typename FLOATV_T>
__device__ FLOATV_T __vset(float x) {}

template<>
__device__ float __forceinline__ __vset<float>(float x) {
    return x;
}

__device__ float __forceinline__ __vmul(float a, float b) {
    return a*b;
}

__device__ float __forceinline__ __vadd(float a, float b) {
    return a+b;
}

__device__ float __forceinline__ __vred(float a) {
    return a;
}

__device__ float __forceinline__ __vscale(float s, float v) {
    return v*s;
}

__device__ float __forceinline__ __vdiv(float s, float v) {
    return v/s;
}

template<>
__device__ float4 __forceinline__ __vset<float4>(float x) {
    return make_float4(x, x, x, x);
}

__device__ float4 __forceinline__ __vmul(float4 a, float4 b) {
    return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}

__device__ float4 __forceinline__ __vadd(float4 a, float4 b) {
    return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

__device__ float __forceinline__ __vred(float4 a) {
    return a.x + a.y + a.z + a.w;
}

__device__ float4 __forceinline__ __vscale(float s, float4 v) {
    return make_float4(s*v.x, s*v.y, s*v.z, s*v.w);
}

__device__ float4 __forceinline__ __vdiv(float s, float4 v) {
    return make_float4(s/v.x, s/v.y, s/v.z, s/v.w);;
}

// called with (blockDim.x=32 and blockDim.y>1, BDIM=blockDim.x*blockDim.y)
template<int BDIM,
         typename FLOATV_T> // either float or float4
__global__
__launch_bounds__(BDIM)
void s2_attn_fwd_generic_vec_k(int nchan,  // no. of FLOATV_T elements along channel dim
                               int nlat_in,
                               int nlon_in,
                               int nlat_out,
                               int nlon_out,
                               const FLOATV_T *__restrict__ kx,
                               const FLOATV_T *__restrict__ vx,
                               const FLOATV_T *__restrict__ qy,
                               const torch::PackedTensorAccessor32<    int, 1, torch::RestrictPtrTraits> row_idx,
                               const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> row_off,
                               const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> col_idx,
                               const torch::PackedTensorAccessor32<  float, 1, torch::RestrictPtrTraits> quad_weights,
                                     FLOATV_T *__restrict__ y) {

    extern __shared__ __align__(sizeof(float4)) float shext[];
    FLOATV_T *shy = reinterpret_cast<FLOATV_T *>(shext) + threadIdx.y*nchan;

    const int batch = blockIdx.y;
    const int wid = blockIdx.x*blockDim.y + threadIdx.y;

    if (wid >= nlat_out*nlon_out) {
        return;
    }

    const int tidx = threadIdx.x;

    const int h = wid / nlon_out;
    const int wo = wid - (h*nlon_out);
    const int ho = row_idx[h];

    for(int chan = tidx; chan < nchan; chan += WARP_SIZE) {
        shy[chan] = __vset<FLOATV_T>(0.f);
    }

    kx += batch*nlat_in*nlon_in*nchan;
    vx += batch*nlat_in*nlon_in*nchan;
    qy += batch*nlat_out*nlon_out*nchan + ho*nchan*nlon_out + wo*nchan;
    y  += batch*nlat_out*nlon_out*nchan + ho*nchan*nlon_out + wo*nchan;

    float alpha_sum = 0.0f;
    float qdotk_max = -FLT_MAX;

    const int64_t rbeg = row_off[ho];
    const int64_t rend = row_off[ho+1];

    const int rlen = rend-rbeg;

    for(int off = 0; off < rlen; off++) {

        const int64_t col = col_idx[rbeg+off];

        const int hi = col / nlon_in;
        const int wi = col - (hi*nlon_in);
        const int wip = (wi+wo) - ((wi+wo) / nlon_in) * nlon_in;

        const FLOATV_T *_kx = kx + hi*nlon_in*nchan + wip*nchan;
        const FLOATV_T *_vx = vx + hi*nlon_in*nchan + wip*nchan;

        FLOATV_T qdotkv = __vset<FLOATV_T>(0.f);

        for(int chan = tidx; chan < nchan; chan += WARP_SIZE) {
            qdotkv = __vadd(qdotkv,
                            __vmul( qy[chan],
                                   _kx[chan]));
        }

        float qdotk = __warp_sum(__vred(qdotkv));

        float qdotk_max_tmp;
        float alpha;
        float exp_save;

        qdotk_max_tmp = max(qdotk_max, qdotk);
        alpha = expf(qdotk - qdotk_max_tmp) * quad_weights[hi];
        exp_save = expf(qdotk_max - qdotk_max_tmp);

        alpha_sum = alpha + alpha_sum*exp_save;

        for(int chan = tidx; chan < nchan; chan += WARP_SIZE) {
            shy[chan] = __vadd(__vscale(exp_save, shy[chan]),
                               __vscale(   alpha, _vx[chan]));
        }
        qdotk_max = qdotk_max_tmp;
    }

    alpha_sum = 1.0f / alpha_sum;
    for(int chan = tidx; chan < nchan; chan += WARP_SIZE) {
        y[chan] = __vscale(alpha_sum, shy[chan]);
    }

    return;
}

template<typename FLOATV_T>
void launch_gen_attn_kernel(int batch_size,
                            int nloc,
                            int nchans,
                            int nlat_in,
                            int nlon_in,
                            int nlat_out,
                            int nlon_out,
                            FLOATV_T *__restrict__ _kxp,
                            FLOATV_T *__restrict__ _vxp,
                            FLOATV_T *__restrict__ _qyp,
                            at::Tensor row_idx,
                            at::Tensor row_off,
                            at::Tensor col_idx,
                            at::Tensor quad_weights,
                            FLOATV_T *__restrict__ _yp,
                            cudaStream_t stream) {

    dim3 block(WARP_SIZE, THREADS/WARP_SIZE);
    dim3 grid(DIV_UP(nlat_out*nlon_out, block.y), batch_size);

    size_t shsize = sizeof(FLOATV_T)*nchans * block.y;

    auto _row_idx  = col_idx.packed_accessor32<    int, 1, torch::RestrictPtrTraits>();
    auto _row_off  = row_off.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>();
    auto _col_idx  = col_idx.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>();
    auto _quad_weights = quad_weights.packed_accessor32< float, 1, torch::RestrictPtrTraits>();

    s2_attn_fwd_generic_vec_k<THREADS>
                             <<<grid, block, shsize, stream>>>(nchans, nlat_in, nlon_in, nlat_out, nlon_out,
                                                               _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp);
    return;
}

// called with either (BDIM_X=32 and BDIM_Y>1) || (2^K=BDIM_X > 32 and BDIM_Y=1)
template<int BDIM_X,
         int BDIM_Y,
         int NLOC,
         typename FLOATV_T> // either float or float4
__global__
__launch_bounds__(BDIM_X*BDIM_Y)
void s2_attn_fwd_special_vec_k(int nchan, // no. of FLOATV_T elements along channel dim
                               int nlat_in,
                               int nlon_in,
                               int nlat_out,
                               int nlon_out,
                               const FLOATV_T *__restrict__ kx,
                               const FLOATV_T *__restrict__ vx,
                               const FLOATV_T *__restrict__ qy,
                               const torch::PackedTensorAccessor32<    int, 1, torch::RestrictPtrTraits> row_idx,
                               const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> row_off,
                               const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> col_idx,
                               const torch::PackedTensorAccessor32<  float, 1, torch::RestrictPtrTraits> quad_weights,
                                     FLOATV_T *__restrict__ y) {

    static_assert(0 == (BDIM_X & (BDIM_X-1)));
    static_assert(0 == (BDIM_Y & (BDIM_Y-1)));
    static_assert((BDIM_X == 32 && BDIM_Y  > 1) ||
                  (BDIM_X  > 32 && BDIM_Y == 1)) ;

    constexpr int NLOC_M1 = NLOC-1;

    const int tidx = threadIdx.x;
    const int batch = blockIdx.y;
    const int ctaid = blockIdx.x*blockDim.y + threadIdx.y;

    if (ctaid >= nlat_out*nlon_out) {
        return;
    }

    FLOATV_T locy[NLOC];

    extern __shared__ __align__(sizeof(float4)) float shext[];
    FLOATV_T *shq = reinterpret_cast<FLOATV_T *>(shext) + threadIdx.y*nchan + tidx;

    const int h = ctaid / nlon_out;
    const int wo = ctaid - (h*nlon_out);
    const int ho = row_idx[h];

    kx += batch*nlat_in*nlon_in*nchan + tidx;
    vx += batch*nlat_in*nlon_in*nchan + tidx;
    qy += batch*nlat_out*nlon_out*nchan + ho*nlon_out*nchan + wo*nchan + tidx;
    y  += batch*nlat_out*nlon_out*nchan + ho*nlon_out*nchan + wo*nchan + tidx;

    #pragma unroll
    for(int i = 0; i < NLOC; i++) {
        locy[i] = __vset<FLOATV_T>(0.f);
    }

    #pragma unroll
    for(int i = 0; i < NLOC_M1; i++) {
        shq[i*BDIM_X] = qy[i*BDIM_X];
    }
    if (NLOC_M1*BDIM_X+tidx < nchan) {
        shq[NLOC_M1*BDIM_X] = qy[NLOC_M1*BDIM_X];
    }

    float alpha_sum = 0.0f;
    float qdotk_max = -FLT_MAX;

    const int64_t rbeg = row_off[ho];
    const int64_t rend = row_off[ho+1];

    const int rlen = rend-rbeg;

    for(int off = 0; off < rlen; off++) {

        const int64_t col = col_idx[rbeg+off];

        const int hi = col / nlon_in;
        const int wi = col - (hi*nlon_in);
        const int wip = (wi+wo) - ((wi+wo) / nlon_in) * nlon_in;

        const FLOATV_T *_kx = kx + hi*nlon_in*nchan + wip*nchan;
        const FLOATV_T *_vx = vx + hi*nlon_in*nchan + wip*nchan;

        FLOATV_T qdotkv = __vset<FLOATV_T>(0.f);

        #pragma unroll
        for(int i = 0; i < NLOC_M1; i++) {
            qdotkv = __vadd(qdotkv,
                            __vmul(shq[i*BDIM_X],
                                   _kx[i*BDIM_X]));
        }
        if (NLOC_M1*BDIM_X+tidx < nchan) {
            qdotkv = __vadd(qdotkv,
                            __vmul(shq[NLOC_M1*BDIM_X],
                                   _kx[NLOC_M1*BDIM_X]));
        }

        float qdotk = __vred(qdotkv);
        if constexpr(BDIM_X == 32) { qdotk =          __warp_sum(qdotk); }
        else                       { qdotk = __block_sum<BDIM_X>(qdotk); }

        float qdotk_max_tmp;
        float alpha;
        float exp_save;

        qdotk_max_tmp = max(qdotk_max, qdotk);
        alpha = expf(qdotk - qdotk_max_tmp) * quad_weights[hi];
        exp_save = expf(qdotk_max - qdotk_max_tmp);

        alpha_sum = alpha + alpha_sum*exp_save;

        #pragma unroll
        for(int i = 0; i < NLOC_M1; i++) {
            locy[i] = __vadd(__vscale(exp_save, locy[i]),
                             __vscale(alpha, _vx[i*BDIM_X]));
        }
        if (NLOC_M1*BDIM_X+tidx < nchan) {
            locy[NLOC_M1] = __vadd(__vscale(exp_save, locy[NLOC_M1]),
                                   __vscale(alpha, _vx[NLOC_M1*BDIM_X]));
        }

        qdotk_max = qdotk_max_tmp;
    }

    alpha_sum = 1.0f / alpha_sum;

    #pragma unroll
    for(int i = 0; i < NLOC_M1; i++) {
        y[i*BDIM_X] = __vscale(alpha_sum, locy[i]);
    }
    if (NLOC_M1*BDIM_X+tidx < nchan) {
        y[NLOC_M1*BDIM_X] = __vscale(alpha_sum, locy[NLOC_M1]);
    }

    return;
}

template<int BDIM_X,
         int BDIM_Y,
         int CUR_LOC_SIZE,
         int MAX_LOC_SIZE, // max size of FLOATV_T[] local array
         typename FLOATV_T>
void launch_spc_attn_kernel(int batch_size,
                            int nloc,      // "BDIM_X*nloc" >= nchans
                            int nchans,
                            int nlat_in,
                            int nlon_in,
                            int nlat_out,
                            int nlon_out,
                            FLOATV_T *__restrict__ _kxp,
                            FLOATV_T *__restrict__ _vxp,
                            FLOATV_T *__restrict__ _qyp,
                            at::Tensor row_idx,
                            at::Tensor row_off,
                            at::Tensor col_idx,
                            at::Tensor quad_weights,
                            FLOATV_T *__restrict__ _yp,
                            cudaStream_t stream) {

    if (CUR_LOC_SIZE == nloc) {

        auto _row_idx  = row_idx.packed_accessor32<    int, 1, torch::RestrictPtrTraits>();
        auto _row_off  = row_off.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>();
        auto _col_idx  = col_idx.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>();
        auto _quad_weights = quad_weights.packed_accessor32<float, 1, torch::RestrictPtrTraits>();

        dim3 block(BDIM_X, BDIM_Y);
        dim3 grid(DIV_UP(nlat_out*nlon_out, block.y), batch_size);

        //printf("block: (%d, %d)\n", block.x, block.y);
        //printf("grid:  (%d, %d)\n", grid.x, grid.y);

        size_t shsize = sizeof(FLOATV_T)*nchans * block.y; // block.y > 1 iif block.x==32

        s2_attn_fwd_special_vec_k<BDIM_X, BDIM_Y, CUR_LOC_SIZE>
                                 <<<grid, block, shsize, stream>>>(nchans, nlat_in, nlon_in, nlat_out, nlon_out,
                                                                   _kxp, _vxp, _qyp, _row_idx, _row_off, _col_idx, _quad_weights, _yp);
        return;
    }
    if constexpr(CUR_LOC_SIZE < MAX_LOC_SIZE) {
        launch_spc_attn_kernel<BDIM_X,
                               BDIM_Y,
                               CUR_LOC_SIZE+1,
                               MAX_LOC_SIZE>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out,
                                             _kxp, _vxp, _qyp, row_idx, row_off, col_idx, quad_weights, _yp,
                                             stream);
    }
    return;
}

__global__ void set_rlen_rids_k(const int n,
                                const int64_t *__restrict__ offs,
                                      int *__restrict__ rids,
                                      int *__restrict__ rlen) {

    const int nth = gridDim.x*blockDim.x;
    const int tid = blockIdx.x*blockDim.x + threadIdx.x;

    for(int i = tid; i < n; i += nth) {
        rids[i] = i;
        rlen[i] = offs[i+1]-offs[i];
    }

    return;
}   

at::Tensor sortRows(int nlat_out, at::Tensor row_off, cudaStream_t stream) {

    int64_t *_row_off_d = reinterpret_cast<int64_t *>(row_off.data_ptr());

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(row_off.device());

    torch::Tensor rids_d = torch::empty({nlat_out}, options);
    torch::Tensor rlen_d = torch::empty({nlat_out}, options);

    int *_rids_d = reinterpret_cast<int *>(rids_d.data_ptr());
    int *_rlen_d = reinterpret_cast<int *>(rlen_d.data_ptr());

    const int grid = DIV_UP(nlat_out, THREADS);
    const int block = THREADS;

    set_rlen_rids_k<<<grid, block, 0, stream>>>(nlat_out,
                                                _row_off_d,
                                                _rids_d,
                                                _rlen_d);

    torch::Tensor rids_sort_d = torch::empty({nlat_out}, options);
    torch::Tensor rlen_sort_d = torch::empty({nlat_out}, options);

    int *_rids_sort_d = reinterpret_cast<int *>(rids_sort_d.data_ptr());
    int *_rlen_sort_d = reinterpret_cast<int *>(rlen_sort_d.data_ptr());

    size_t temp_storage_bytes = 0;
    CHECK_CUDA(cub::DeviceRadixSort::SortPairsDescending(NULL, temp_storage_bytes,
                                                         _rlen_d, _rlen_sort_d, 
                                                         _rids_d, _rids_sort_d,
                                                         nlat_out, 0, sizeof(*_rlen_d)*8, stream));

    options = torch::TensorOptions().dtype(torch::kByte).device(row_off.device());
    torch::Tensor temp_storage_d = torch::empty({int64_t(temp_storage_bytes)}, options);

    void *_temp_storage_d = reinterpret_cast<void *>(temp_storage_d.data_ptr());

    CHECK_CUDA(cub::DeviceRadixSort::SortPairsDescending(_temp_storage_d, temp_storage_bytes,
                                                         _rlen_d, _rlen_sort_d, 
                                                         _rids_d, _rids_sort_d,
                                                         nlat_out, 0, sizeof(*_rlen_d)*8, stream));

    return rids_sort_d;
}

template<unsigned int ALIGN>
int is_aligned(const void *ptr) {

    static_assert(0 == (ALIGN & (ALIGN-1)));
    return (0 == (uintptr_t(ptr) & (ALIGN-1)));
}

static unsigned int next_pow2(unsigned int x) { 

    x -= 1;

    #pragma unroll
    for(int i = 1; i <= sizeof(x)*8 / 2; i *= 2) {
        x |= x >> i;    
    }
    return x+1;
}

static void s2_attention_dipatch(int batch_size,
                                 int nchans,
                                 int nlon_in,
                                 int nlat_out,
                                 int nlon_out,
                                 at::Tensor kxP,
                                 at::Tensor vxP,
                                 at::Tensor qyP,
                                 at::Tensor row_off,
                                 at::Tensor col_idx,
                                 at::Tensor quad_weights,
                                 at::Tensor yP,
                                 cudaStream_t stream) {

    static_assert(0 == (MAX_LOCAL_ARR_LEN & (MAX_LOCAL_ARR_LEN-1)));

    // sort row indices (ho-s) in descending order
    // based on (row_off[ho+1]-row_off[ho])
    at::Tensor row_idx = sortRows(nlat_out, row_off, stream);

    const int nlat_in = kxP.size(1);

    // smallest power of two "bdimx" (>=32) s.t. bdimx*MAX_LOCAL_ARR_LEN >= nchans
    int bdimx;
    bdimx = DIV_UP(nchans, MAX_LOCAL_ARR_LEN);
    bdimx = max(bdimx, WARP_SIZE);
    bdimx = next_pow2(bdimx);

    float *_kxp = reinterpret_cast<float *>(kxP.data_ptr());
    float *_vxp = reinterpret_cast<float *>(vxP.data_ptr());
    float *_qyp = reinterpret_cast<float *>(qyP.data_ptr());
    float *_yp  = reinterpret_cast<float *>(yP.data_ptr());

    constexpr int VEC_SIZE = sizeof(float4) / sizeof(float);

    if (!is_aligned<sizeof(float4)>(_kxp) ||
        !is_aligned<sizeof(float4)>(_vxp) ||
        !is_aligned<sizeof(float4)>(_qyp) ||
        !is_aligned<sizeof(float4)>(_yp)  ||
        (nchans % VEC_SIZE) != 0) {

        const int nloc = DIV_UP(nchans, bdimx);

        // use 2D blocks only if 32 threads are enough
        switch(bdimx) {
            case   32: launch_spc_attn_kernel<  32, 2, 1, MAX_LOCAL_ARR_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, row_idx, row_off, col_idx, quad_weights, _yp, stream); break;
            case   64: launch_spc_attn_kernel<  64, 1, 1, MAX_LOCAL_ARR_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, row_idx, row_off, col_idx, quad_weights, _yp, stream); break;
            case  128: launch_spc_attn_kernel< 128, 1, 1, MAX_LOCAL_ARR_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, row_idx, row_off, col_idx, quad_weights, _yp, stream); break;
            case  256: launch_spc_attn_kernel< 256, 1, 1, MAX_LOCAL_ARR_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, row_idx, row_off, col_idx, quad_weights, _yp, stream); break;
            case  512: launch_spc_attn_kernel< 512, 1, 1, MAX_LOCAL_ARR_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, row_idx, row_off, col_idx, quad_weights, _yp, stream); break;
            case 1024: launch_spc_attn_kernel<1024, 1, 1, MAX_LOCAL_ARR_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, row_idx, row_off, col_idx, quad_weights, _yp, stream); break;
            default:   launch_gen_attn_kernel                               (batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp, _vxp, _qyp, row_idx, row_off, col_idx, quad_weights, _yp, stream); break;
        }

    } else {

        float4 *_kxp4 = reinterpret_cast<float4 *>(_kxp);
        float4 *_vxp4 = reinterpret_cast<float4 *>(_vxp);
        float4 *_qyp4 = reinterpret_cast<float4 *>(_qyp);
        float4 *_yp4  = reinterpret_cast<float4 *>(_yp);

        nchans /= VEC_SIZE;
        const int nloc = DIV_UP(nchans, bdimx);

        static constexpr int MAX_LOCAL_VEC_LEN = MAX_LOCAL_ARR_LEN / VEC_SIZE;

        // use 2D blocks only if 32 threads are enough
        switch(bdimx) {
            case   32: launch_spc_attn_kernel<  32, 2, 1, MAX_LOCAL_VEC_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, row_idx, row_off, col_idx, quad_weights, _yp4, stream); break;
            case   64: launch_spc_attn_kernel<  64, 1, 1, MAX_LOCAL_VEC_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, row_idx, row_off, col_idx, quad_weights, _yp4, stream); break;
            case  128: launch_spc_attn_kernel< 128, 1, 1, MAX_LOCAL_VEC_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, row_idx, row_off, col_idx, quad_weights, _yp4, stream); break;
            case  256: launch_spc_attn_kernel< 256, 1, 1, MAX_LOCAL_VEC_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, row_idx, row_off, col_idx, quad_weights, _yp4, stream); break;
            case  512: launch_spc_attn_kernel< 512, 1, 1, MAX_LOCAL_VEC_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, row_idx, row_off, col_idx, quad_weights, _yp4, stream); break;
            case 1024: launch_spc_attn_kernel<1024, 1, 1, MAX_LOCAL_VEC_LEN>(batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, row_idx, row_off, col_idx, quad_weights, _yp4, stream); break;
            default:   launch_gen_attn_kernel                               (batch_size, nloc, nchans, nlat_in, nlon_in, nlat_out, nlon_out, _kxp4, _vxp4, _qyp4, row_idx, row_off, col_idx, quad_weights, _yp4, stream); break;
        }
    }

    return;
}
// END - forward kernels and functions

// BEGIN - tensor permutation kernels and functions
template<int BDIM_X,
         int BDIM_Y,
         typename VAL_T>
__global__
__launch_bounds__(BDIM_X*BDIM_Y)
void  permute_to0231_k(const int nchn,
                       const int nlat,
                       const int nlon,
                       const torch::PackedTensorAccessor32<VAL_T, 4, torch::RestrictPtrTraits> src,
                             torch::PackedTensorAccessor32<VAL_T, 4, torch::RestrictPtrTraits> dst) {

    static_assert(!(BDIM_X & (BDIM_X-1)));
    static_assert(!(BDIM_Y & (BDIM_Y-1)));
    static_assert(BDIM_X >= BDIM_Y);

    __shared__ VAL_T sh[BDIM_X][BDIM_X+1];

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int coff  = blockIdx.x*BDIM_X;           // channel offset
    const int woff  = blockIdx.y*BDIM_X;           // width offset
    const int batch = blockIdx.z / nlat;           // batch (same for all block)
    const int h     = blockIdx.z - (batch * nlat); // height (same for all block)

    const int nchn_full = (nchn-coff) >= BDIM_X;
    const int nlon_full = (nlon-woff) >= BDIM_X;

    if (nchn_full && nlon_full) {
        #pragma unroll
        for(int j = 0; j < BDIM_X; j += BDIM_Y) {
            sh[j+tidy][tidx] = src[batch][coff + j+tidy][h][woff+tidx];
        }
        __syncthreads();

        #pragma unroll
        for(int j = 0; j < BDIM_X; j += BDIM_Y) {
            dst[batch][h][woff + j+tidy][coff+tidx] = sh[tidx][j+tidy];
        }
    } else {
        if (woff+tidx < nlon) {
            #pragma unroll
            for(int j = 0; j < BDIM_X; j += BDIM_Y) {
                sh[j+tidy][tidx] = (coff + j+tidy < nchn) ? src[batch][coff + j+tidy][h][woff+tidx] : 0.f;
            }
        }
        __syncthreads();

        if (coff+tidx < nchn) {
            #pragma unroll
            for(int j = 0; j < BDIM_X; j += BDIM_Y) {
                if (woff + j+tidy < nlon) {
                    dst[batch][h][woff + j+tidy][coff+tidx] = sh[tidx][j+tidy];
                }
            }
        }
    }
    return;
}

__global__ void empty_k() {}

static int getPtxver() {
    cudaFuncAttributes attrs;
    CHECK_CUDA(cudaFuncGetAttributes(&attrs, empty_k));
    return attrs.ptxVersion*10;
}

static at::Tensor permute_4D_floatT_to0231(at::Tensor src, cudaStream_t stream) {

    dim3 block;
    dim3 grid;

    block.x = WARP_SIZE;
    grid.x = DIV_UP(src.size(1), block.x);
    grid.y = DIV_UP(src.size(3), block.x);
    grid.z = src.size(2)*src.size(0);

    assert(grid.y < 65536);
    assert(grid.z < 65536);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(src.device());
    torch::Tensor dst = torch::empty({src.size(0), src.size(2), src.size(3), src.size(1)}, options);

    const int ptxv = getPtxver();

    // to be further specialized for additional archs, if necessary
    if (ptxv < 100) {
        block.y = TRANSP_WARPS_X_TILE_GENERIC;
        permute_to0231_k<WARP_SIZE, TRANSP_WARPS_X_TILE_GENERIC>
                        <<<grid, block, 0, stream>>>(src.size(1),
                                                     src.size(2),
                                                     src.size(3),
                                                     src.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                     dst.packed_accessor32<float, 4, torch::RestrictPtrTraits>());
    } else {
        block.y = TRANSP_WARPS_X_TILE_SM100;
        permute_to0231_k<WARP_SIZE, TRANSP_WARPS_X_TILE_SM100>
                        <<<grid, block, 0, stream>>>(src.size(1),
                                                     src.size(2),
                                                     src.size(3),
                                                     src.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                     dst.packed_accessor32<float, 4, torch::RestrictPtrTraits>());
    }

    return dst;
}

template<int BDIM_X,
         int BDIM_Y,
         typename VAL_T>
__global__
__launch_bounds__(BDIM_X*BDIM_Y)
void  permute_to0312_k(const int nchn,
                       const int nlat,
                       const int nlon,
                       const torch::PackedTensorAccessor32<VAL_T, 4, torch::RestrictPtrTraits> src,
                             torch::PackedTensorAccessor32<VAL_T, 4, torch::RestrictPtrTraits> dst) {

    static_assert(!(BDIM_X & (BDIM_X-1)));
    static_assert(!(BDIM_Y & (BDIM_Y-1)));
    static_assert(BDIM_X >= BDIM_Y);

    __shared__ VAL_T sh[BDIM_X][BDIM_X+1];

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int woff  = blockIdx.x*BDIM_X;           // width offset
    const int coff  = blockIdx.y*BDIM_X;           // channel offset
    const int batch = blockIdx.z / nlat;           // batch (same for all block)
    const int h     = blockIdx.z - (batch * nlat); // height (same for all block)

    const int nchn_full = (nchn-coff) >= BDIM_X;
    const int nlon_full = (nlon-woff) >= BDIM_X;

    if (nchn_full && nlon_full) {
        #pragma unroll
        for(int j = 0; j < BDIM_X; j += BDIM_Y) {
            sh[j+tidy][tidx] = src[batch][h][woff + j+tidy][coff+tidx];
        }
        __syncthreads();

        #pragma unroll
        for(int j = 0; j < BDIM_X; j += BDIM_Y) {
            dst[batch][coff + j+tidy][h][woff+tidx] = sh[tidx][j+tidy];
        }
    } else {
        if (coff+tidx < nchn) {
            #pragma unroll
            for(int j = 0; j < BDIM_X; j += BDIM_Y) {
                sh[j+tidy][tidx] = (woff + j+tidy < nlon) ? src[batch][h][woff + j+tidy][coff+tidx] : 0.f;
            }
        }
        __syncthreads();

        if (woff+tidx < nlon) {
            #pragma unroll
            for(int j = 0; j < BDIM_X; j += BDIM_Y) {
                if (coff + j+tidy < nchn) {
                    dst[batch][coff + j+tidy][h][woff+tidx] = sh[tidx][j+tidy];;
                }
            }
        }
    }
    return;
}

static at::Tensor permute_4D_floatT_to0312(at::Tensor src, cudaStream_t stream) {

    dim3 block;
    dim3 grid;

    block.x = WARP_SIZE;
    grid.x = DIV_UP(src.size(2), block.x);
    grid.y = DIV_UP(src.size(3), block.x);
    grid.z = src.size(1)*src.size(0);

    assert(grid.y < 65536);
    assert(grid.z < 65536);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(src.device());
    torch::Tensor dst = torch::empty({src.size(0), src.size(3), src.size(1), src.size(2)}, options);

    const int ptxv = getPtxver();

    // to be further specialized for additional archs, if necessary
    if (ptxv < 100) {
        block.y = TRANSP_WARPS_X_TILE_GENERIC;
        permute_to0312_k<WARP_SIZE, TRANSP_WARPS_X_TILE_GENERIC>
                        <<<grid, block, 0, stream>>>(src.size(3),
                                                     src.size(1),
                                                     src.size(2),
                                                     src.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                     dst.packed_accessor32<float, 4, torch::RestrictPtrTraits>());
    } else {
        block.y = TRANSP_WARPS_X_TILE_SM100;
        permute_to0312_k<WARP_SIZE, TRANSP_WARPS_X_TILE_SM100>
                        <<<grid, block, 0, stream>>>(src.size(3),
                                                     src.size(1),
                                                     src.size(2),
                                                     src.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                     dst.packed_accessor32<float, 4, torch::RestrictPtrTraits>());
    }

    return dst;
}
// END - tensor permutation kernels and functions

torch::Tensor s2_attention_fwd_cuda(at::Tensor kx,
                                    at::Tensor vx,
                                    at::Tensor qy,
                                    at::Tensor quad_weights,
                                    at::Tensor psi_col_idx,
                                    at::Tensor psi_row_off,
                                    int nlon_in,
                                    int nlat_out,
                                    int nlon_out) {
    CHECK_CUDA_TENSOR(kx);
    CHECK_CUDA_TENSOR(vx);
    CHECK_CUDA_TENSOR(qy);
    CHECK_CUDA_TENSOR(quad_weights);
    CHECK_CUDA_TENSOR(psi_col_idx);
    CHECK_CUDA_TENSOR(psi_row_off);

    // TODO: check sizes

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    size_t uo_num_channels = kx.size(1);

    const int batch_size = kx.size(0);

    torch::Tensor kxP = kx;
    torch::Tensor vxP = vx;
    torch::Tensor qyP = qy;

    auto k_channel_first = kx.strides()[1] == 1;
    auto v_channel_first = vx.strides()[1] == 1;
    auto q_channel_first = qy.strides()[1] == 1;

    if (!k_channel_first) { kxP = permute_4D_floatT_to0231(kx, stream); }
    if (!v_channel_first) { vxP = permute_4D_floatT_to0231(vx, stream); }
    if (!q_channel_first) { qyP = permute_4D_floatT_to0231(qy, stream); }

    torch::Tensor yP = torch::empty_like(qyP);

    s2_attention_dipatch(batch_size,
                         uo_num_channels,
                         nlon_in,
                         nlat_out,
                         nlon_out,
                         kxP, vxP, qyP,
                         psi_row_off,
                         psi_col_idx,
                         quad_weights,
                         yP, // out tensor
                         stream);

    torch::Tensor y = yP;
    if (!q_channel_first) { y = permute_4D_floatT_to0312(yP, stream); }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}
