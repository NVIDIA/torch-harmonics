# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

r"""
The Hankel/Toeplitz tensor-core forward of the regular DISCO contraction, on sm_100a (B200 / GB200).

A Kernel Factory candidate (``kernels_cuda/regular/disco_cuda_fwd_hankel_sm100.cu``), built into
``torch_harmonics.disco._C_sm100``. For every output row it cuts psi's support into pieces of consecutive
longitudes on one input latitude and contracts each piece as a GEMM of the input longitudes (a Hankel
matrix, straight from shared memory) and the filter values (a Toeplitz matrix) on the tensor cores, with
channels and longitude blocks as the GEMM rows. fp16 and bf16 activations, psi in the kpacked layout,
fp32 accumulation; the result is rounded once to the activations' dtype.

Its ops, ``forward_hankel_sm100_``:

``plan``
    Everything the forward derives from psi for a number of planes B * C, a dtype and an input shape: the
    pieces, the Toeplitz tiles, the filter values in that dtype, the latitude ranges of the extended input.
    The gather's call state of the kpacked plan (``KpackedPlanS2``), built once per key and kept. Zero
    bytes where the kernel does not apply.
``run``
    The forward through such a plan. Sizes the kernel does not take -- pscale 3 or above 8, kernel_size
    above 9, ``nlon_out`` not a multiple of 8 -- run the shipped kpacked forward inside it.
``config``
    The launch configuration for given sizes on the current device, for tests.

The backends of the kpacked forward (``regular-kpacked``, ``regular-fft``) run it where
``TORCH_HARMONICS_DISCO_ENABLE_HANKEL_KERNEL=1`` allows it, on compute capability 10.0 (B200 / GB200) devices
whose build holds the 10.0a image. Per call it takes scratch of about twice the input's size, the input
extended along longitude.
"""

import torch

from . import disco_arch


def _register_fakes(prefix: str) -> None:
    @torch.library.register_fake(f"disco_kernels::{prefix}_plan")
    def _(pack_idx, pack_val, pack_offset, bc, kernel_size, nlat_in, nlon_in, nlon_out, dtype):
        # a byte buffer whose size the configuration decides
        return pack_idx.new_empty((torch.library.get_ctx().new_dynamic_size(),), dtype=torch.uint8)

    @torch.library.register_fake(f"disco_kernels::{prefix}_run")
    def _(inp, plan, pack_idx, pack_val, pack_offset, kernel_size, nlat_out, nlon_out):
        return inp.new_empty((inp.shape[0], inp.shape[1], kernel_size, nlat_out, nlon_out))


for _family, _prefix in disco_arch.HANKEL_FORWARD_OPS.items():
    if disco_arch.hankel_forward_registered(_family):
        _register_fakes(_prefix)


def hankel_forward_available(device: torch.device) -> bool:
    """Whether the Hankel/Toeplitz forward can run on ``device``: a compute capability 10.0 GPU and a build holding its image."""
    if device.type != "cuda":
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    return disco_arch.hankel_forward_supported_on_device(index)
