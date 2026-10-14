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

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>

#include <map>
#include <mutex>
#include <utility>

// Kernel launch helpers shared by the CUDA extensions of the package. Header-only, so each
// extension carries its own copy of the opt-in cache, which is all it needs: a cache only
// ever holds that extension's kernels.

namespace th_cuda
{

    // Opt a kernel in to shsize bytes of dynamic shared memory when that exceeds the
    // default 48 KiB -- without the opt-in such a launch fails with cudaErrorInvalidValue,
    // and nothing points at shared memory. `what` names the kernel and `hint` says what
    // the request grows with, for the message when even the opt-in cannot serve it.
    inline void ensure_dyn_shmem(const void *kern, size_t shsize, const char *what, const char *hint)
    {
        if (shsize <= 48u * 1024u) { return; }

        // The opt-in is per (device, kernel), and the granted size has to be at least the
        // largest ever requested for that pair:
        //
        //  - shsize depends on the problem size, so one kernel instantiation is launched
        //    at different sizes by different module instances in the same process.
        //    Caching on the kernel alone drops every request after the first, and a later,
        //    larger launch then fails with cudaErrorInvalidValue.
        //
        //  - cudaFuncSetAttribute applies to the current device, so a process-wide cache
        //    would let device 0 suppress the opt-in that device 1 never received.
        //
        // The mutex is what makes the cache safe to touch from the launch path, which torch
        // may drive from several threads. It is uncontended, and only reached on the
        // >48KB path.
        int dev = 0;
        C10_CUDA_CHECK(cudaGetDevice(&dev));

        static std::mutex mtx;
        static std::map<std::pair<int, const void *>, size_t> granted;

        std::lock_guard<std::mutex> lock(mtx);

        // inserts a 0 entry when this (device, kernel) pair is new
        size_t &granted_size = granted[std::make_pair(dev, kern)];
        if (granted_size >= shsize) { return; }

        // The opt-in cannot exceed what the device offers a block, less the kernel's static
        // shared memory. Past that point the launch is impossible rather than merely
        // un-opted-in; say so instead of letting cudaFuncSetAttribute fail with a bare
        // invalid argument.
        int optin_max = 0;
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        cudaFuncAttributes attr;
        C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, kern));
        const size_t avail = static_cast<size_t>(optin_max) - attr.sharedSizeBytes;
        TORCH_CHECK(shsize <= avail, what, " needs ", shsize, " bytes of dynamic shared memory, but device ", dev,
                    " offers at most ", avail, " per block; ", hint);

        C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shsize)));
        granted_size = shsize;
    }

    // Launch a kernel that takes dynamic shared memory, opting in first when needed. kern
    // must be the fully specified instantiation; the launch arguments then have to convert
    // to its parameters, so naming the wrong instantiation fails to compile instead of
    // opting in a kernel that is never launched.
    template <typename... KArgs, typename... Args>
    inline void launch_dyn_shmem(void (*kern)(KArgs...), dim3 grid, dim3 block, size_t shsize, cudaStream_t stream,
                                 const char *what, const char *hint, Args &&...args)
    {
        ensure_dyn_shmem(reinterpret_cast<const void *>(kern), shsize, what, hint);
        kern<<<grid, block, shsize, stream>>>(std::forward<Args>(args)...);
    }

} // namespace th_cuda
