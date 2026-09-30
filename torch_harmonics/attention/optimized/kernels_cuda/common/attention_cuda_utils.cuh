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

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <type_traits>
#include <utility>

#include "../../attention_checks.h"

#define WARP_SIZE (32)
#define FULL_MASK (0xFFFFFFFF)
#define DIV_UP(a, b) (((a) + ((b) - 1)) / (b))

namespace attention_kernels
{

    // Extents of one ring step: the serial kernels' (nheads, nchan_in, nchan_out,
    // nlon_in, nlat_out, nlon_out), plus where the K/V chunk currently held sits in the
    // global input grid.
    struct attn_params_t {
        // Heads are packed along the channel dim, as in the serial kernels: tensors
        // are physically (B, H, W, nheads * nchan), gridDim spans batch * nheads, and
        // nchan_in / nchan_out are per-head counts. The stride between two spatial
        // points is therefore nheads * nchan, not nchan.
        int nheads;
        int nchan_in;
        int nchan_out;
        int nlat_halo;      // latitudes in the K/V chunk, halo included
        int nlon_kx;        // longitudes in the K/V chunk
        int nlon_in;        // GLOBAL input longitudes
        int pscale;         // GLOBAL nlon_in / nlon_out; not derivable from the local nlon_out
        int lon_lo_kx;      // global longitude of the chunk's first column
        int lat_halo_start; // global latitude of the chunk's first (halo) row
        int nlat_out;       // LOCAL output latitudes
        int nlon_out;       // LOCAL output longitudes
    };

    // CSR rows sorting kernels and functions
    at::Tensor sortRows(int nlat_out, at::Tensor row_off, cudaStream_t stream);

    // 4D tensor permutation kernels and functions
    at::Tensor permute_4D_to0231(at::Tensor src);
    at::Tensor permute_4D_to0312(at::Tensor src);

    // Registered op entry points wrapping the two transposes above. These are
    // the only layout conversion the attention stack should use: they are
    // torch.compile visible (fake impl + autograd registered in Python, see
    // attention/_layout.py) and validate their input rather than inferring
    // layout from strides.
    at::Tensor permute_to_nhwc_cuda(at::Tensor x);
    at::Tensor permute_to_nchw_cuda(at::Tensor x);

    unsigned int next_pow2(unsigned int x);

    void ensure_dyn_shmem(const void *kern, size_t shsize);

    // Launch a kernel that takes dynamic shared memory, opting in first when the request
    // exceeds the default 48 KiB -- without the opt-in such a launch fails with
    // cudaErrorInvalidValue. The request grows with the channel count, so every launch
    // with a channel-dependent shsize goes through here. kern must be the fully
    // specified instantiation; the launch arguments then have to convert to its
    // parameters, so naming the wrong storage type fails to compile instead of opting
    // in a kernel that is never launched.
    template <typename... KArgs, typename... Args>
    inline void launch_dyn_shmem(void (*kern)(KArgs...), dim3 grid, dim3 block, size_t shsize, cudaStream_t stream,
                                 Args &&...args)
    {
        ensure_dyn_shmem(reinterpret_cast<const void *>(kern), shsize);
        kern<<<grid, block, shsize, stream>>>(std::forward<Args>(args)...);
    }

    int getPtxver();

    // utility host functions and templates

    template <unsigned int ALIGN> int is_aligned(const void *ptr)
    {

        static_assert(0 == (ALIGN & (ALIGN - 1)));
        return (0 == (uintptr_t(ptr) & (ALIGN - 1)));
    }

    // utility device functions and templates

    template <typename FLOATV_T> __device__ FLOATV_T __vset(float x)
    {
        static_assert(sizeof(FLOATV_T) == 0, "Unsupported type for __vset");
        return FLOATV_T {};
    }

    template <> __device__ float __forceinline__ __vset<float>(float x) { return x; }

    __device__ float __forceinline__ __vmul(float a, float b) { return a * b; }

    __device__ float __forceinline__ __vadd(float a, float b) { return a + b; }

    __device__ float __forceinline__ __vsub(float a, float b) { return a - b; }

    __device__ float __forceinline__ __vred(float a) { return a; }

    __device__ float __forceinline__ __vscale(float s, float v) { return v * s; }

    template <> __device__ float4 __forceinline__ __vset<float4>(float x) { return make_float4(x, x, x, x); }

    __device__ float4 __forceinline__ __vmul(float4 a, float4 b)
    {
        return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
    }

    __device__ float4 __forceinline__ __vadd(float4 a, float4 b)
    {
        return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }

    __device__ float4 __forceinline__ __vsub(float4 a, float4 b)
    {
        return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
    }

    __device__ float __forceinline__ __vred(float4 a) { return a.x + a.y + a.z + a.w; }

    __device__ float4 __forceinline__ __vscale(float s, float4 v)
    {
        return make_float4(s * v.x, s * v.y, s * v.z, s * v.w);
    }

    // ---- storage <-> compute helpers for native fp16/bf16 storage ----
    //
    // Kernels are templated on STORAGE_T (the element type as laid out in global
    // memory). The COMPUTE_T it maps to is the type used for all arithmetic /
    // accumulation: float4 stays float4 (the fp32 vectorized fast path), every
    // scalar storage type (float, c10::Half, c10::BFloat16) computes in float.
    // vload widens STORAGE_T -> COMPUTE_T at the load site; vstore narrows back
    // at the store site. c10::Half / c10::BFloat16 provide device-side
    // conversions to/from float, so no half intrinsics are needed here.
    template <typename STORAGE_T> struct vec_traits {
        using compute_t = float;
    };
    template <> struct vec_traits<float4> {
        using compute_t = float4;
    };

    // 4-wide 16-bit storage: eight bytes, one LDG.64 instead of four LDG.U16.
    //
    // The point is instruction count, not bandwidth. The forward kernel is issue-slot
    // limited (82% compute throughput, DRAM at 1%, ~9.4 warp cycles per issued
    // instruction), so four scalar 16-bit loads cost four issue slots where one
    // vector load costs one. The measured evidence that this trade is worth it: the
    // fp32 float4 path beats the fp16 scalar path by 29% while running at 31%
    // occupancy against 50% and moving twice the bytes.
    //
    // compute_t is float4, so every existing __vadd/__vmul/__vred/__vscale overload
    // applies unchanged and accumulation stays fp32 -- no __hfma2, no precision
    // change. Critically, the packed registers do not outlive the conversion in
    // vload: nvcc only keeps a __half2 in a single register if every use of it is a
    // half2 operation, and one scalar touch anywhere unpacks the whole chain. Here
    // the packed value is consumed immediately and only the float4 accumulator
    // survives, so there is no long-lived packed value to mis-schedule.
    struct alignas(8) half4 {
        __half2 lo, hi;
    };
    struct alignas(8) bf164 {
        c10::BFloat16 x, y, z, w;
    };

    template <> struct vec_traits<half4> {
        using compute_t = float4;
    };
    template <> struct vec_traits<bf164> {
        using compute_t = float4;
    };

    // scalar load/store: STORAGE_T in {float, c10::Half, c10::BFloat16}; compute_t == float.
    // (The vectorised fp16 paths below deliberately DO use a half intrinsic:
    // __half22float2 converts a pair in one instruction where two c10::Half
    // conversions would take two, which is the point of vectorising on an
    // issue-limited kernel. bf16 has no such packed conversion below sm_80, so it
    // stays on c10's operators, which already handle the arch split.)
    // The return type is spelled via vec_traits so the float4 specialization below
    // (which returns float4) matches the primary template's signature.
    template <typename STORAGE_T>
    __device__ __forceinline__ typename vec_traits<STORAGE_T>::compute_t vload(const STORAGE_T *p, int idx)
    {
        return static_cast<float>(p[idx]);
    }
    template <typename STORAGE_T> __device__ __forceinline__ void vstore(STORAGE_T *p, int idx, float v)
    {
        p[idx] = static_cast<STORAGE_T>(v);
    }

    // float4 vectorized load/store (fp32 fast path): identity, no conversion
    template <> __device__ __forceinline__ float4 vload<float4>(const float4 *p, int idx) { return p[idx]; }
    __device__ __forceinline__ void vstore(float4 *p, int idx, float4 v) { p[idx] = v; }

    // 16-bit vectorized load/store. One 8-byte access, then widen to fp32 immediately
    // so nothing packed stays live (see the note on vec_traits<half4> above).
    template <> __device__ __forceinline__ float4 vload<half4>(const half4 *p, int idx)
    {
        const half4 v = p[idx];
        const float2 a = __half22float2(v.lo);
        const float2 b = __half22float2(v.hi);
        return make_float4(a.x, a.y, b.x, b.y);
    }
    __device__ __forceinline__ void vstore(half4 *p, int idx, float4 v)
    {
        half4 out;
        out.lo = __floats2half2_rn(v.x, v.y);
        out.hi = __floats2half2_rn(v.z, v.w);
        p[idx] = out;
    }

    template <> __device__ __forceinline__ float4 vload<bf164>(const bf164 *p, int idx)
    {
        // c10::BFloat16's conversions already select __float2bfloat16 on sm_80+ and
        // software round-to-nearest-even below it, so this needs no arch guard and no
        // hand-rolled bit manipulation. fp16 below keeps __half22float2 because that
        // converts a whole pair in one instruction and is available from sm_53.
        const bf164 v = p[idx];
        return make_float4(float(v.x), float(v.y), float(v.z), float(v.w));
    }
    __device__ __forceinline__ void vstore(bf164 *p, int idx, float4 v)
    {
        bf164 out;
        out.x = c10::BFloat16(v.x);
        out.y = c10::BFloat16(v.y);
        out.z = c10::BFloat16(v.z);
        out.w = c10::BFloat16(v.w);
        p[idx] = out;
    }

    __device__ __forceinline__ void atomicMax(float *ptr, float val)
    {
        int *int_ptr = (int *)ptr;
        int old = *int_ptr, assumed;

        do {
            assumed = old;
            if (__int_as_float(assumed) >= val) { break; }
            old = atomicCAS(int_ptr, assumed, __float_as_int(val));

        } while (assumed != old);
        return;
    }

    template <typename VAL_T> __device__ VAL_T __warp_sum(VAL_T val)
    {

#pragma unroll
        for (int i = WARP_SIZE / 2; i; i /= 2) { val += __shfl_xor_sync(FULL_MASK, val, i, WARP_SIZE); }
        return val;
    }

    template <int BDIM_X, int BDIM_Y = 1, int BDIM_Z = 1, typename VAL_T> __device__ VAL_T __block_sum(VAL_T val)
    {

        const int NWARP = (BDIM_X * BDIM_Y * BDIM_Z) / WARP_SIZE;

        val = __warp_sum(val);

        if constexpr (NWARP > 1) {

            int tid = threadIdx.x;
            if constexpr (BDIM_Y > 1) { tid += threadIdx.y * BDIM_X; }
            if constexpr (BDIM_Z > 1) { tid += threadIdx.z * BDIM_X * BDIM_Y; }

            const int lid = tid % WARP_SIZE;
            const int wid = tid / WARP_SIZE;

            __shared__ VAL_T sh[NWARP];

            if (lid == 0) { sh[wid] = val; }
            __syncthreads();

            if (wid == 0) {
                val = (lid < NWARP) ? sh[lid] : 0;

                val = __warp_sum(val);
                __syncwarp();

                if (!lid) { sh[0] = val; }
            }
            __syncthreads();

            val = sh[0];
            __syncthreads();
        }
        return val;
    }

    // transpose utils
    template <int BDIM_X, int BDIM_Y, typename VAL_T>
    __global__ __launch_bounds__(BDIM_X *BDIM_Y) void permute_to0231_k(
        const int nchn, const int nlat, const int nlon,
        const at::PackedTensorAccessor32<VAL_T, 4, at::RestrictPtrTraits> src,
        at::PackedTensorAccessor32<VAL_T, 4, at::RestrictPtrTraits> dst)
    {

        static_assert(!(BDIM_X & (BDIM_X - 1)));
        static_assert(!(BDIM_Y & (BDIM_Y - 1)));
        static_assert(BDIM_X >= BDIM_Y);

        __shared__ VAL_T sh[BDIM_X][BDIM_X + 1];

        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;

        const int coff = blockIdx.x * BDIM_X;      // channel offset
        const int woff = blockIdx.y * BDIM_X;      // width offset
        const int batch = blockIdx.z / nlat;       // batch (same for all block)
        const int h = blockIdx.z - (batch * nlat); // height (same for all block)

        const int nchn_full = (nchn - coff) >= BDIM_X;
        const int nlon_full = (nlon - woff) >= BDIM_X;

        if (nchn_full && nlon_full) {
#pragma unroll
            for (int j = 0; j < BDIM_X; j += BDIM_Y) {
                sh[j + tidy][tidx] = src[batch][coff + j + tidy][h][woff + tidx];
            }
            __syncthreads();

#pragma unroll
            for (int j = 0; j < BDIM_X; j += BDIM_Y) {
                dst[batch][h][woff + j + tidy][coff + tidx] = sh[tidx][j + tidy];
            }
        } else {
            if (woff + tidx < nlon) {
#pragma unroll
                for (int j = 0; j < BDIM_X; j += BDIM_Y) {
                    sh[j + tidy][tidx]
                        = (coff + j + tidy < nchn) ? src[batch][coff + j + tidy][h][woff + tidx] : VAL_T(0);
                }
            }
            __syncthreads();

            if (coff + tidx < nchn) {
#pragma unroll
                for (int j = 0; j < BDIM_X; j += BDIM_Y) {
                    if (woff + j + tidy < nlon) { dst[batch][h][woff + j + tidy][coff + tidx] = sh[tidx][j + tidy]; }
                }
            }
        }
        return;
    }

    template <int WARPS_X_TILE, typename VAL_T> void launch_permute_to0231(at::Tensor src, at::Tensor dst)
    {
        dim3 block;
        dim3 grid;

        block.x = WARP_SIZE;
        block.y = WARPS_X_TILE;
        grid.x = DIV_UP(src.size(1), block.x);
        grid.y = DIV_UP(src.size(3), block.x);
        grid.z = src.size(2) * src.size(0);

        TORCH_CHECK(grid.y < 65536, "permute_to0231: grid.y (", grid.y,
                    ") exceeds CUDA gridDim.y limit of 65535; input nlon dimension is too large");
        TORCH_CHECK(grid.z < 65536, "permute_to0231: grid.z (", grid.z,
                    ") exceeds CUDA gridDim.z limit of 65535; batch * nlat is too large");

        // get stream
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        permute_to0231_k<WARP_SIZE, WARPS_X_TILE><<<grid, block, 0, stream>>>(
            src.size(1), src.size(2), src.size(3), src.packed_accessor32<VAL_T, 4, at::RestrictPtrTraits>(),
            dst.packed_accessor32<VAL_T, 4, at::RestrictPtrTraits>());
    }

    template <int BDIM_X, int BDIM_Y, typename VAL_T>
    __global__ __launch_bounds__(BDIM_X *BDIM_Y) void permute_to0312_k(
        const int nchn, const int nlat, const int nlon,
        const at::PackedTensorAccessor32<VAL_T, 4, at::RestrictPtrTraits> src,
        at::PackedTensorAccessor32<VAL_T, 4, at::RestrictPtrTraits> dst)
    {

        static_assert(!(BDIM_X & (BDIM_X - 1)));
        static_assert(!(BDIM_Y & (BDIM_Y - 1)));
        static_assert(BDIM_X >= BDIM_Y);

        __shared__ VAL_T sh[BDIM_X][BDIM_X + 1];

        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;

        const int woff = blockIdx.x * BDIM_X;      // width offset
        const int coff = blockIdx.y * BDIM_X;      // channel offset
        const int batch = blockIdx.z / nlat;       // batch (same for all block)
        const int h = blockIdx.z - (batch * nlat); // height (same for all block)

        const int nchn_full = (nchn - coff) >= BDIM_X;
        const int nlon_full = (nlon - woff) >= BDIM_X;

        if (nchn_full && nlon_full) {
#pragma unroll
            for (int j = 0; j < BDIM_X; j += BDIM_Y) {
                sh[j + tidy][tidx] = src[batch][h][woff + j + tidy][coff + tidx];
            }
            __syncthreads();

#pragma unroll
            for (int j = 0; j < BDIM_X; j += BDIM_Y) {
                dst[batch][coff + j + tidy][h][woff + tidx] = sh[tidx][j + tidy];
            }
        } else {
            if (coff + tidx < nchn) {
#pragma unroll
                for (int j = 0; j < BDIM_X; j += BDIM_Y) {
                    sh[j + tidy][tidx]
                        = (woff + j + tidy < nlon) ? src[batch][h][woff + j + tidy][coff + tidx] : VAL_T(0);
                }
            }
            __syncthreads();

            if (woff + tidx < nlon) {
#pragma unroll
                for (int j = 0; j < BDIM_X; j += BDIM_Y) {
                    if (coff + j + tidy < nchn) {
                        dst[batch][coff + j + tidy][h][woff + tidx] = sh[tidx][j + tidy];
                        ;
                    }
                }
            }
        }
        return;
    }

    template <int WARPS_X_TILE, typename VAL_T> void launch_permute_to0312(at::Tensor src, at::Tensor dst)
    {
        dim3 block;
        dim3 grid;

        block.x = WARP_SIZE;
        block.y = WARPS_X_TILE;
        grid.x = DIV_UP(src.size(2), block.x);
        grid.y = DIV_UP(src.size(3), block.x);
        grid.z = src.size(1) * src.size(0);

        TORCH_CHECK(grid.y < 65536, "permute_to0312: grid.y (", grid.y,
                    ") exceeds CUDA gridDim.y limit of 65535; input nlon dimension is too large");
        TORCH_CHECK(grid.z < 65536, "permute_to0312: grid.z (", grid.z,
                    ") exceeds CUDA gridDim.z limit of 65535; batch * nchn is too large");

        // get stream
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        permute_to0312_k<WARP_SIZE, WARPS_X_TILE><<<grid, block, 0, stream>>>(
            src.size(3), src.size(1), src.size(2), src.packed_accessor32<VAL_T, 4, at::RestrictPtrTraits>(),
            dst.packed_accessor32<VAL_T, 4, at::RestrictPtrTraits>());
    }

    // Reduce wi + pscale*wo into [0, nlon_in).
    //
    // Deliberately not `%`. nlon_in is a runtime value and the GPU has no integer
    // divide instruction, so `x % nlon_in` compiles to ~25 instructions of software
    // emulation. Profiling the forward kernel on H100 showed exactly this dominating:
    // 78% compute throughput while only ~2.4% of peak FLOPs were the actual dot
    // product, with DRAM at 0.5% -- the kernel was spending its time on address
    // arithmetic, not on math or memory.
    //
    // The reduction is exact with one conditional subtract because both terms are
    // already bounded: wi is a canonical column so wi < nlon_in, and
    // pscale*wo <= pscale*(nlon_out - 1) < pscale*nlon_out == nlon_in. Hence
    // wi_wo < 2*nlon_in and at most one wrap can occur. This also holds in the ring
    // kernels, where wi arrives pre-shifted modulo nlon_in and wo is rank-local.
    __device__ __forceinline__ int wrap_lon(int wi_wo, int nlon_in)
    {
        return (wi_wo >= nlon_in) ? wi_wo - nlon_in : wi_wo;
    }

    // Clip a longitude arc to a window, for the ring kernels.
    //
    // The arc is the one a serial kernel walks: `len` longitudes from `start` on a ring
    // of `nlon`, wrapping at most once (start < nlon, len <= nlon). The window is the
    // range [w_lo, w_lo + w_len) of the same ring that one rank holds, and does not
    // wrap. Split at the seam, the arc is at most two linear pieces, and each meets the
    // window in at most one interval, so the result is at most two pieces, returned as
    // (first longitude relative to w_lo, count) in the order the serial walk visits
    // them. With the whole ring as the window this reproduces the serial walk exactly.
    //
    // This is what the column form could not do: it had to decode every neighbour of a
    // row on every ring step and discard the (P-1)/P of them outside the chunk. Here the
    // cost per step is a few integer ops per arc plus the neighbours actually present.
    __device__ __forceinline__ int clip_arc(int start, int len, int nlon, int w_lo, int w_len, int2 (&piece)[2])
    {
        const int w_hi = w_lo + w_len;
        int n = 0;

        // [start, min(start + len, nlon))
        int a = max(start, w_lo);
        int b = min(min(start + len, nlon), w_hi);
        if (a < b) { piece[n++] = make_int2(a - w_lo, b - a); }

        // the part past the seam, [0, start + len - nlon)
        a = w_lo;
        b = min(start + len - nlon, w_hi);
        if (a < b) { piece[n++] = make_int2(a - w_lo, b - a); }

        return n;
    }

} // namespace attention_kernels
