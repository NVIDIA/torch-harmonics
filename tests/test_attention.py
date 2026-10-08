# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
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
#

import math
import unittest
import warnings
from dataclasses import dataclass
from typing import ClassVar
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from parameterized import parameterized, parameterized_class

# from torch.autograd import gradcheck
from test_neighborhood import _brute_force_neighborhood
from testutils import build_psi_segments, compare_tensors, disable_tf32, expand_psi_segments, maybe_autocast, requires_torch_compile, set_seed
from torch.library import opcheck

from torch_harmonics import AttentionS2, GridS2, HealpixGrid, NeighborhoodAttentionS2, as_grid
from torch_harmonics.attention import backends as attention_backends
from torch_harmonics.attention import cuda_kernels_is_available, optimized_kernels_is_available
from torch_harmonics.attention._layout import to_nhwc
from torch_harmonics.attention.backends import RaggedOptimizedBackend, RaggedReferenceBackend, RegularOptimizedBackend, RegularReferenceBackend, _point_weights
from torch_harmonics.attention.kernels_torch.attention_ragged_torch import _neighborhood_s2_attention_ragged_torch
from torch_harmonics.attention.kernels_torch.attention_regular_torch import (
    _neighborhood_s2_attention_regular_bwd_dk_torch,
    _neighborhood_s2_attention_regular_bwd_dq_torch,
    _neighborhood_s2_attention_regular_bwd_dv_torch,
    _neighborhood_s2_attention_regular_fwd_torch,
    _neighborhood_s2_attention_upsample_bwd_dk_torch,
    _neighborhood_s2_attention_upsample_bwd_dq_torch,
    _neighborhood_s2_attention_upsample_bwd_dv_torch,
    _neighborhood_s2_attention_upsample_fwd_torch,
)

# None on a build without the kernels, where the tests that use it are skipped
from torch_harmonics.attention.optimized.attention_optimized import _neighborhood_s2_attention_ragged_optimized
from torch_harmonics.disco.convolution import _precompute_convolution_tensor_s2
from torch_harmonics.filter_basis import get_filter_basis
from torch_harmonics.grid import _GRID_REGISTRY
from torch_harmonics.neighborhood import precompute_neighborhood_csr_s2
from torch_harmonics.quadrature import precompute_latitudes

if not optimized_kernels_is_available():
    print("Warning: Couldn't import optimized disco convolution kernels")

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


@parameterized_class(("device"), _devices)
class TestNeighborhoodAttentionRegularS2(unittest.TestCase):
    """Test the neighborhood attention module (CPU/CUDA if available)."""

    def setUp(self):
        disable_tf32()
        torch.manual_seed(333)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(333)

    @parameterized.expand(
        [
            # Format: [batch_size, channels, channels_out, heads, in_shape, out_shape, grid_in, grid_out, use_qknorm, atol, rtol]
            [4, 4, 4, 1, (6, 12), (6, 12), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],
            [4, 4, 4, 2, (6, 12), (6, 12), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],
            [4, 4, 4, 4, (6, 12), (6, 12), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],
            [4, 4, 8, 4, (6, 12), (6, 12), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],
            [4, 8, 4, 4, (6, 12), (6, 12), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],
            [4, 1, 1, 1, (2, 4), (2, 4), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],
            [4, 1, 4, 1, (2, 4), (2, 4), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],
            [4, 4, 4, 4, (6, 12), (6, 12), "legendre-gauss", "legendre-gauss", False, torch.float32, 1e-5, 1e-3],
            [4, 4, 4, 1, (6, 12), (6, 12), "lobatto", "lobatto", False, torch.float32, 1e-5, 1e-3],
            # downsampling: nlon_in must be an integer multiple of nlon_out (pscale = nlon_in / nlon_out)
            [4, 8, 4, 4, (12, 24), (6, 12), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],  # lat 2x, lon 2x (pscale=2)
            [4, 4, 8, 4, (12, 24), (6, 12), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],  # C_in<C_out asym, pscale=2
            [4, 4, 4, 1, (6, 12), (6, 6), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],  # lon-only, pscale=2
            [4, 4, 4, 1, (12, 24), (6, 8), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],  # pscale=3
            [4, 4, 4, 1, (12, 24), (3, 6), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],  # pscale=4
            [4, 4, 4, 1, (12, 12), (6, 12), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],  # lat-only, pscale=1
            [4, 4, 4, 1, (12, 24), (6, 12), "legendre-gauss", "legendre-gauss", False, torch.float32, 1e-5, 1e-3],  # LG grid, pscale=2
            # odd latitude sizes
            [4, 4, 4, 1, (7, 12), (5, 6), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],  # odd-odd lat, pscale=2
            [4, 4, 4, 1, (9, 12), (5, 4), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],  # odd-odd lat, pscale=3
            [4, 4, 4, 1, (11, 24), (7, 12), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],  # odd-odd lat, pscale=2
            [4, 4, 4, 1, (12, 24), (11, 24), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],  # odd nlat_out only, pscale=1
            # upsampling: mirror of the downsampling rows above (in_shape ↔ out_shape, grid_in ↔ grid_out)
            [4, 8, 4, 4, (6, 12), (12, 24), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],  # pscale_out=2
            [4, 4, 8, 4, (6, 12), (12, 24), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],  # C_in<C_out asym, pscale_out=2
            [4, 4, 4, 1, (6, 6), (6, 12), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],  # lon-only, pscale_out=2
            [4, 4, 4, 1, (6, 8), (12, 24), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],  # pscale_out=3
            [4, 4, 4, 1, (3, 6), (12, 24), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],  # pscale_out=4
            [4, 4, 4, 1, (6, 12), (12, 12), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],  # lat-only, pscale_out=1
            [4, 4, 4, 1, (6, 12), (12, 24), "legendre-gauss", "legendre-gauss", False, torch.float32, 1e-5, 1e-3],  # LG grid, pscale_out=2
            [4, 4, 4, 1, (5, 6), (7, 12), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],  # odd-odd lat, pscale_out=2
            [4, 4, 4, 1, (5, 4), (9, 12), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],  # odd-odd lat, pscale_out=3
            [4, 4, 4, 1, (7, 12), (11, 24), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],  # odd-odd lat, pscale_out=2
            [4, 4, 4, 1, (11, 24), (12, 24), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],  # odd nlat_in only, pscale_out=1
            # same cases with QK norm enabled
            [4, 4, 4, 1, (6, 12), (6, 12), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],
            [4, 4, 4, 2, (6, 12), (6, 12), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],
            [4, 4, 4, 4, (6, 12), (6, 12), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],
            [4, 4, 8, 4, (6, 12), (6, 12), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],
            [4, 8, 4, 4, (6, 12), (6, 12), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],
            [4, 8, 4, 4, (12, 24), (6, 12), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],
            [4, 1, 1, 1, (2, 4), (2, 4), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],
            [4, 1, 4, 1, (2, 4), (2, 4), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],
            [4, 4, 4, 4, (6, 12), (6, 12), "legendre-gauss", "legendre-gauss", True, torch.float32, 1e-5, 1e-3],
            [4, 4, 4, 1, (6, 12), (6, 12), "lobatto", "lobatto", True, torch.float32, 1e-5, 1e-3],
            # downsampling: nlon_in must be an integer multiple of nlon_out (pscale = nlon_in / nlon_out)
            [4, 8, 4, 4, (12, 24), (6, 12), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],  # lat 2x, lon 2x (pscale=2)
            [4, 4, 8, 4, (12, 24), (6, 12), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],  # C_in<C_out asym, pscale=2
            [4, 4, 4, 1, (6, 12), (6, 6), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],  # lon-only, pscale=2
            [4, 4, 4, 1, (12, 24), (6, 8), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],  # pscale=3
            [4, 4, 4, 1, (12, 24), (3, 6), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],  # pscale=4
            [4, 4, 4, 1, (12, 12), (6, 12), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],  # lat-only, pscale=1
            [4, 4, 4, 1, (12, 24), (6, 12), "legendre-gauss", "legendre-gauss", True, torch.float32, 1e-5, 1e-3],  # LG grid, pscale=2
            # odd latitude sizes
            [4, 4, 4, 1, (7, 12), (5, 6), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],  # odd-odd lat, pscale=2
            [4, 4, 4, 1, (9, 12), (5, 4), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],  # odd-odd lat, pscale=3
            [4, 4, 4, 1, (11, 24), (7, 12), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],  # odd-odd lat, pscale=2
            [4, 4, 4, 1, (12, 24), (11, 24), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],  # odd nlat_out only, pscale=1
            # upsampling: mirror of the downsampling rows above (in_shape ↔ out_shape, grid_in ↔ grid_out)
            [4, 8, 4, 4, (6, 12), (12, 24), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],  # pscale_out=2
            [4, 4, 8, 4, (6, 12), (12, 24), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],  # C_in<C_out asym, pscale_out=2
            [4, 4, 4, 1, (6, 6), (6, 12), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],  # lon-only, pscale_out=2
            [4, 4, 4, 1, (6, 8), (12, 24), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],  # pscale_out=3
            [4, 4, 4, 1, (3, 6), (12, 24), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],  # pscale_out=4
            [4, 4, 4, 1, (6, 12), (12, 12), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],  # lat-only, pscale_out=1
            [4, 4, 4, 1, (6, 12), (12, 24), "legendre-gauss", "legendre-gauss", True, torch.float32, 1e-5, 1e-3],  # LG grid, pscale_out=2
            [4, 4, 4, 1, (5, 6), (7, 12), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],  # odd-odd lat, pscale_out=2
            [4, 4, 4, 1, (5, 4), (9, 12), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],  # odd-odd lat, pscale_out=3
            [4, 4, 4, 1, (7, 12), (11, 24), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],  # odd-odd lat, pscale_out=2
            [4, 4, 4, 1, (11, 24), (12, 24), "equiangular", "equiangular", True, torch.float32, 1e-5, 1e-3],  # odd nlat_in only, pscale_out=1
            # AMP coverage — one row per code path × dtype. Tolerances are
            # looser because cuBLAS bf16/fp16 rounding at the einsum output
            # dominates the abs error; see feedback_amp_fp16_cancellation.
            # downsampling
            [4, 4, 4, 1, (12, 24), (6, 12), "equiangular", "equiangular", False, torch.float16, 2e-2, 1e-2],
            [4, 4, 4, 1, (12, 24), (6, 12), "equiangular", "equiangular", False, torch.bfloat16, 5e-2, 5e-2],
            # upsampling
            [4, 4, 4, 1, (6, 12), (12, 24), "equiangular", "equiangular", False, torch.float16, 2e-2, 1e-2],
            [4, 4, 4, 1, (6, 12), (12, 24), "equiangular", "equiangular", False, torch.bfloat16, 5e-2, 5e-2],
            # same-resolution gather (self-attention) — the path the down/upsample rows skip
            [4, 4, 4, 1, (6, 12), (6, 12), "equiangular", "equiangular", False, torch.float16, 2e-2, 1e-2],
            [4, 4, 4, 1, (6, 12), (6, 12), "equiangular", "equiangular", False, torch.bfloat16, 5e-2, 5e-2],
            # gather, multi-head + asymmetric channels (C_in > C_out)
            [4, 8, 4, 4, (6, 12), (6, 12), "equiangular", "equiangular", False, torch.float16, 2e-2, 1e-2],
            [4, 8, 4, 4, (6, 12), (6, 12), "equiangular", "equiangular", False, torch.bfloat16, 5e-2, 5e-2],
            # resampling paths, multi-head + both channel asymmetries. Heads are packed
            # along the channel dimension, so the per-head base offset is h*C and the
            # channel extent the kernel sees is num_heads*C -- which is exactly what the
            # vectorized (float4) branch gates on. The fp32 rows above cover these shapes;
            # these cover them on the AMP paths, where the vector and scalar branches
            # diverge most.
            # downsampling (gather)
            [4, 8, 4, 4, (12, 24), (6, 12), "equiangular", "equiangular", False, torch.float16, 2e-2, 1e-2],
            [4, 8, 4, 4, (12, 24), (6, 12), "equiangular", "equiangular", False, torch.bfloat16, 5e-2, 5e-2],
            [4, 4, 8, 4, (12, 24), (6, 12), "equiangular", "equiangular", False, torch.float16, 2e-2, 1e-2],
            [4, 4, 8, 4, (12, 24), (6, 12), "equiangular", "equiangular", False, torch.bfloat16, 5e-2, 5e-2],
            # upsampling (scatter)
            [4, 8, 4, 4, (6, 12), (12, 24), "equiangular", "equiangular", False, torch.float16, 2e-2, 1e-2],
            [4, 8, 4, 4, (6, 12), (12, 24), "equiangular", "equiangular", False, torch.bfloat16, 5e-2, 5e-2],
            [4, 4, 8, 4, (6, 12), (12, 24), "equiangular", "equiangular", False, torch.float16, 2e-2, 1e-2],
            [4, 4, 8, 4, (6, 12), (12, 24), "equiangular", "equiangular", False, torch.bfloat16, 5e-2, 5e-2],
            # multi-head on the scalar branch: 6 channels over 2 heads is not a multiple
            # of the float4 vector width, so the vectorized branch is gated off and the
            # scalar path handles the packed-head offsets instead. Every other heads>1
            # row uses C in {4, 8} and therefore only exercises the vector branch.
            [4, 6, 6, 2, (6, 12), (12, 24), "equiangular", "equiangular", False, torch.float16, 2e-2, 1e-2],
            [4, 6, 6, 2, (6, 12), (12, 24), "equiangular", "equiangular", False, torch.bfloat16, 5e-2, 5e-2],
            # Vectorized-load coverage. The CUDA dispatch only takes the vector path when
            # the vectorized channel count still fills the block -- nchans_per_head / 4 >=
            # bdimx, i.e. >= 128 channels per head. Every other row in this grid has <= 8,
            # so without these the float4 / half4 / bf164 loads are compiled but never
            # executed. 256 channels over 2 heads also gives 128 per head, which checks
            # that the gate reads the per-head count rather than the packed extent.
            [1, 128, 128, 1, (6, 12), (6, 12), "equiangular", "equiangular", False, torch.float32, 1e-5, 1e-3],
            [1, 128, 128, 1, (6, 12), (6, 12), "equiangular", "equiangular", False, torch.float16, 2e-2, 1e-2],
            [1, 128, 128, 1, (6, 12), (6, 12), "equiangular", "equiangular", False, torch.bfloat16, 5e-2, 5e-2],
            [1, 256, 256, 2, (6, 12), (6, 12), "equiangular", "equiangular", False, torch.float16, 2e-2, 1e-2],
            [1, 128, 128, 1, (12, 24), (6, 12), "equiangular", "equiangular", False, torch.float16, 2e-2, 1e-2],
            [1, 128, 128, 1, (6, 12), (12, 24), "equiangular", "equiangular", False, torch.float16, 2e-2, 1e-2],
            # Dynamic shared memory above the default 48 KiB. The backward kernels stage
            # per-channel rows in shared memory, so the request grows with the channel
            # count, and past 48 KiB the launch fails unless the kernel is opted in first.
            # gather backward: 8192 is the widest the register-blocked kernel serves
            # (BDIM_X=512), (8192 + 8192) * 4 B = 64 KiB. Wider falls to the generic kernel,
            # whose ~40 B per channel exceeds every device's per-block maximum.
            [1, 8192, 8192, 1, (6, 12), (6, 12), "equiangular", "equiangular", False, torch.float32, 1e-4, 1e-3],
            # upsample backward dk/dv: 4 * 2048 * 4 B * 2 warps = 64 KiB
            [1, 2048, 2048, 1, (6, 12), (12, 24), "equiangular", "equiangular", False, torch.float32, 1e-4, 1e-3],
            # gather with QK norm enabled
            [4, 4, 4, 1, (6, 12), (6, 12), "equiangular", "equiangular", True, torch.float16, 2e-2, 1e-2],
            [4, 4, 4, 1, (6, 12), (6, 12), "equiangular", "equiangular", True, torch.bfloat16, 5e-2, 5e-2],
            # resampling, multi-head, with QK norm enabled
            [4, 8, 4, 4, (12, 24), (6, 12), "equiangular", "equiangular", True, torch.float16, 2e-2, 1e-2],
            [4, 8, 4, 4, (12, 24), (6, 12), "equiangular", "equiangular", True, torch.bfloat16, 5e-2, 5e-2],
            [4, 8, 4, 4, (6, 12), (12, 24), "equiangular", "equiangular", True, torch.float16, 2e-2, 1e-2],
            [4, 8, 4, 4, (6, 12), (12, 24), "equiangular", "equiangular", True, torch.bfloat16, 5e-2, 5e-2],
        ],
        skip_on_empty=True,
    )
    @unittest.skipUnless(optimized_kernels_is_available(), "skipping test because optimized kernels are not available")
    def test_custom_implementation(self, batch_size, channels, channels_out, heads, in_shape, out_shape, grid_in, grid_out, use_qknorm, dtype, atol, rtol, verbose=False):
        """Tests numerical equivalence between the custom (CUDA) implementation and the reference torch implementation"""

        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

        # set seed
        set_seed(333)

        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        # Helper: create inputs
        inputs_ref = {
            "k": torch.randn(batch_size, channels, nlat_in, nlon_in, requires_grad=True, device=self.device, dtype=torch.float32),
            "v": torch.randn(batch_size, channels, nlat_in, nlon_in, requires_grad=True, device=self.device, dtype=torch.float32),
            "q": torch.randn(batch_size, channels, nlat_out, nlon_out, requires_grad=True, device=self.device, dtype=torch.float32),
        }
        inputs_opt = {k: v.detach().clone().to(self.device).requires_grad_() for k, v in inputs_ref.items()}

        # reference input and model
        model_ref = NeighborhoodAttentionS2(
            grid_in=as_grid(grid_in, nlat=in_shape[0], nlon=in_shape[1]),
            grid_out=as_grid(grid_out, nlat=out_shape[0], nlon=out_shape[1]),
            in_channels=channels,
            num_heads=heads,
            use_qknorm=use_qknorm,
            bias=True,
            out_channels=channels_out,
            optimized_kernel=False,
        ).to(self.device)

        # Device model and inputs
        model_opt = NeighborhoodAttentionS2(
            grid_in=as_grid(grid_in, nlat=in_shape[0], nlon=in_shape[1]),
            grid_out=as_grid(grid_out, nlat=out_shape[0], nlon=out_shape[1]),
            in_channels=channels,
            num_heads=heads,
            use_qknorm=use_qknorm,
            bias=True,
            out_channels=channels_out,
            optimized_kernel=True,
        ).to(self.device)

        # Synchronize parameters of model
        model_opt.load_state_dict(model_ref.state_dict())
        for (name_ref, p_ref), (name_opt, p_opt) in zip(model_ref.named_parameters(), model_opt.named_parameters()):
            self.assertTrue(torch.allclose(p_ref.cpu(), p_opt.cpu()))

        # Forward passes — modules + inputs stay fp32; maybe_autocast wraps
        # fwd in autocast(dtype) for fp16/bf16 and is a no-op for fp32, so
        # the same body covers fp32 + AMP rows uniformly.
        with maybe_autocast(self.device.type, dtype):
            out_ref = model_ref(inputs_ref["q"], inputs_ref["k"], inputs_ref["v"])
            out_opt = model_opt(inputs_opt["q"], inputs_opt["k"], inputs_opt["v"])

        # Check forward equivalence
        self.assertTrue(torch.allclose(out_opt, out_ref, atol=atol, rtol=rtol), "Forward outputs differ between torch reference and custom implementation")

        # Backward passes
        grad = torch.randn_like(out_ref)
        out_ref.backward(grad)
        out_opt.backward(grad)

        # Check input gradient equivalence
        for inp in ["q", "v", "k"]:
            grad_ref = inputs_ref[inp].grad.cpu()
            grad_opt = inputs_opt[inp].grad.cpu()
            self.assertTrue(compare_tensors(f"input grad {inp}", grad_opt, grad_ref, atol=atol, rtol=rtol, verbose=verbose))

        # Check parameter gradient equivalence. A bias gradient is a sum over the whole batch and
        # every grid point, the largest reduction in the layer, so its absolute rounding error
        # grows with the size of the terms it sums rather than with its own value, and varies
        # with the CPU the reference's matmuls dispatch to. Bias gradients therefore get an
        # absolute tolerance of a few float32 ulps of the largest parameter gradient; weight
        # gradients keep atol.
        pgrad_scale = max(p_ref.grad.abs().max().item() for _, p_ref in model_ref.named_parameters())
        bias_atol = max(atol, 8 * torch.finfo(torch.float32).eps * pgrad_scale)
        for (name_ref, p_ref), (name_opt, p_opt) in zip(model_ref.named_parameters(), model_opt.named_parameters()):
            pgrad_opt = p_opt.grad.cpu()
            pgrad_ref = p_ref.grad.cpu()
            patol = bias_atol if name_ref.endswith("bias") else atol
            self.assertTrue(compare_tensors(f"parameter grad {name_ref}", pgrad_opt, pgrad_ref, atol=patol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            # [in_shape, out_shape, autocast_dtype]
            # downsample (nlon_in % nlon_out == 0)
            [(12, 24), (6, 12), torch.float16],
            [(12, 24), (6, 12), torch.bfloat16],
            # upsample (nlon_out % nlon_in == 0)
            [(6, 12), (12, 24), torch.float16],
            [(6, 12), (12, 24), torch.bfloat16],
        ],
        skip_on_empty=True,
    )
    @unittest.skipUnless(optimized_kernels_is_available(), "skipping test because optimized kernels are not available")
    def test_optimized_autocast_dtype(self, in_shape, out_shape, autocast_dtype):
        """Direct check that the autocast registration on the optimized attention
        custom_op produces output in the active autocast dtype.

        Uses bias=False so the final op of the attention module is the output
        projection (Linear, autocast-eligible), which preserves the autocast dtype.
        A bias add of bf16+fp32 would dtype-promote to fp32 and mask the autocast
        contract we're testing.
        """
        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

        set_seed(333)

        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        model = NeighborhoodAttentionS2(
            grid_in=as_grid("equiangular", nlat=in_shape[0], nlon=in_shape[1]),
            grid_out=as_grid("equiangular", nlat=out_shape[0], nlon=out_shape[1]),
            in_channels=4,
            num_heads=1,
            use_qknorm=False,
            bias=False,
            out_channels=4,
            optimized_kernel=True,
        ).to(self.device)

        # Inputs in fp32; autocast handles the cast inside fwd.
        k = torch.randn(2, 4, nlat_in, nlon_in, device=self.device, dtype=torch.float32)
        v = torch.randn(2, 4, nlat_in, nlon_in, device=self.device, dtype=torch.float32)
        q = torch.randn(2, 4, nlat_out, nlon_out, device=self.device, dtype=torch.float32)

        with torch.autocast(self.device.type, dtype=autocast_dtype):
            out = model(q, k, v)

        self.assertEqual(
            out.dtype,
            autocast_dtype,
            f"Attention output dtype {out.dtype} != autocast dtype {autocast_dtype}",
        )

    @parameterized.expand([[torch.float16], [torch.bfloat16]], skip_on_empty=True)
    @unittest.skipUnless(optimized_kernels_is_available(), "skipping test because optimized kernels are not available")
    def test_autocast_normalizes_mixed_input_dtypes(self, autocast_dtype):
        """Autocast must reconcile k/v/q dtypes before they reach the kernel.

        The kernels dispatch once on q's scalar type and then reinterpret every
        activation pointer as that type, so they require k, v and q to share a dtype
        and check it explicitly. Autocast does not guarantee that by itself: it casts
        some ops and not others, so a module that mixes projections with normalization
        can hand the op an fp32 q next to an fp16 v. The Autocast{CUDA,CPU} registrations
        are what make the requirement hold.

        This is a direct op-level test rather than a module-level one because at module
        level the mismatch only appears on some torch versions -- it was found on
        torch 2.6 (CPU) while newer builds happened to produce consistent dtypes and
        passed. Constructing the mismatch here makes the regression detectable
        everywhere. The negative control is implicit: without autocast active the same
        inputs raise "must match q dtype".
        """
        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

        set_seed(333)
        nlat, nlon, channels = 6, 12, 4

        model = NeighborhoodAttentionS2(
            grid_in=as_grid("equiangular", nlat=nlat, nlon=nlon),
            grid_out=as_grid("equiangular", nlat=nlat, nlon=nlon),
            in_channels=channels,
            num_heads=1,
            bias=False,
            optimized_kernel=True,
        ).to(self.device)

        # NHWC, as the op expects. q is deliberately left fp32 while k/v are reduced
        # precision -- the exact shape of the failure observed on torch 2.6.
        kw = torch.randn(2, nlat, nlon, channels, device=self.device, dtype=autocast_dtype)
        vw = torch.randn(2, nlat, nlon, channels, device=self.device, dtype=autocast_dtype)
        qw = torch.randn(2, nlat, nlon, channels, device=self.device, dtype=torch.float32)

        with torch.autocast(self.device.type, dtype=autocast_dtype):
            out = torch.ops.attention_kernels._neighborhood_s2_attention_regular_optimized(kw, vw, qw, model.ring_weights, model.psi_seg, model.psi_seg_off, 1, nlon, nlat, nlon)

        self.assertEqual(out.dtype, autocast_dtype, f"autocast output dtype {out.dtype} != {autocast_dtype}")

    @parameterized.expand(
        [
            # Format: [in_shape, out_shape, frozen]  -- which of {k, v, q} has no requires_grad
            # one downsample (pscale=2) and one upsample (pscale_out=2) row, each
            # exercised three times to freeze each input branch in turn.
            [(12, 24), (6, 12), "k"],  # downsample, frozen k
            [(12, 24), (6, 12), "v"],  # downsample, frozen v
            [(12, 24), (6, 12), "q"],  # downsample, frozen q
            [(6, 12), (12, 24), "k"],  # upsample,   frozen k
            [(6, 12), (12, 24), "v"],  # upsample,   frozen v
            [(6, 12), (12, 24), "q"],  # upsample,   frozen q
        ],
        skip_on_empty=True,
    )
    @unittest.skipUnless(optimized_kernels_is_available(), "skipping test because optimized kernels are not available")
    def test_selective_requires_grad(self, in_shape, out_shape, frozen, verbose=False):
        """Verifies the autograd contract when exactly one of {k, v, q} doesn't require gradients.

        Freezing the raw input AND its projection weight+bias makes the op's input tensor a
        non-requires_grad leaf, so ctx.needs_input_grad reflects the intended frozen branch.
        We then confirm:
          - forward outputs still match between torch ref and optimized,
          - the frozen input + its projection params have .grad == None,
          - the remaining input/parameter grads match between ref and optimized.
        """
        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

        set_seed(333)

        batch_size, channels, heads = 4, 4, 1
        atol, rtol = 1e-5, 1e-3
        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        def make_inputs(device):
            ins = {
                "k": torch.randn(batch_size, channels, nlat_in, nlon_in, device=device, dtype=torch.float32),
                "v": torch.randn(batch_size, channels, nlat_in, nlon_in, device=device, dtype=torch.float32),
                "q": torch.randn(batch_size, channels, nlat_out, nlon_out, device=device, dtype=torch.float32),
            }
            for name in ("k", "v", "q"):
                ins[name].requires_grad_(name != frozen)
            return ins

        inputs_ref = make_inputs(self.device)
        inputs_opt = {n: t.detach().clone().requires_grad_(t.requires_grad) for n, t in inputs_ref.items()}

        model_ref = NeighborhoodAttentionS2(
            grid_in=as_grid("equiangular", nlat=in_shape[0], nlon=in_shape[1]),
            grid_out=as_grid("equiangular", nlat=out_shape[0], nlon=out_shape[1]),
            in_channels=channels,
            num_heads=heads,
            bias=True,
            optimized_kernel=False,
        ).to(self.device)
        model_opt = NeighborhoodAttentionS2(
            grid_in=as_grid("equiangular", nlat=in_shape[0], nlon=in_shape[1]),
            grid_out=as_grid("equiangular", nlat=out_shape[0], nlon=out_shape[1]),
            in_channels=channels,
            num_heads=heads,
            bias=True,
            optimized_kernel=True,
        ).to(self.device)
        model_opt.load_state_dict(model_ref.state_dict())

        # freeze the chosen branch's projection so kw/vw/qw (the op inputs) are non-requires_grad
        for model in (model_ref, model_opt):
            getattr(model, f"{frozen}_weights").requires_grad_(False)
            bias = getattr(model, f"{frozen}_bias")
            if bias is not None:
                bias.requires_grad_(False)

        # forward
        out_ref = model_ref(inputs_ref["q"], inputs_ref["k"], inputs_ref["v"])
        out_opt = model_opt(inputs_opt["q"], inputs_opt["k"], inputs_opt["v"])
        self.assertTrue(torch.allclose(out_opt, out_ref, atol=atol, rtol=rtol), f"Forward outputs differ between torch ref and optimized (frozen={frozen})")

        # backward
        grad = torch.randn_like(out_ref)
        out_ref.backward(grad)
        out_opt.backward(grad)

        # input grads: frozen one must be None, the other two must match
        for name in ("k", "v", "q"):
            g_ref = inputs_ref[name].grad
            g_opt = inputs_opt[name].grad
            if name == frozen:
                self.assertIsNone(g_ref, f"ref: expected None grad for frozen input {name}")
                self.assertIsNone(g_opt, f"opt: expected None grad for frozen input {name}")
            else:
                self.assertIsNotNone(g_ref, f"ref: missing grad for input {name}")
                self.assertIsNotNone(g_opt, f"opt: missing grad for input {name}")
                self.assertTrue(
                    compare_tensors(
                        f"input grad {name} (frozen={frozen})",
                        g_opt.cpu(),
                        g_ref.cpu(),
                        atol=atol,
                        rtol=rtol,
                        verbose=verbose,
                    )
                )

        # parameter grads: frozen-branch projection (weights + bias) must be None, others must match
        for (n_ref, p_ref), (n_opt, p_opt) in zip(model_ref.named_parameters(), model_opt.named_parameters()):
            if n_ref.startswith(f"{frozen}_"):
                self.assertIsNone(p_ref.grad, f"ref: expected None grad for frozen param {n_ref}")
                self.assertIsNone(p_opt.grad, f"opt: expected None grad for frozen param {n_opt}")
            else:
                self.assertIsNotNone(p_ref.grad, f"ref: missing grad for param {n_ref}")
                self.assertIsNotNone(p_opt.grad, f"opt: missing grad for param {n_opt}")
                self.assertTrue(
                    compare_tensors(
                        f"parameter grad {n_ref} (frozen={frozen})",
                        p_opt.grad.cpu(),
                        p_ref.grad.cpu(),
                        atol=atol,
                        rtol=rtol,
                        verbose=verbose,
                    )
                )

    # caution: multihead-implementation between full and neighborhood attention still seem to differ. tests are only done for single head
    @parameterized.expand(
        [
            # Format: [batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol]
            # same shape
            [2, 64, 1, (25, 48), (25, 48), "equiangular", "equiangular", 5e-2, 1e-4],
            # downsampling: nlon_in must be an integer multiple of nlon_out (pscale = nlon_in / nlon_out)
            [2, 16, 1, (24, 48), (12, 24), "equiangular", "equiangular", 5e-2, 1e-4],  # lat 2x, lon 2x (pscale=2)
            [2, 16, 1, (12, 24), (12, 12), "equiangular", "equiangular", 5e-2, 1e-4],  # lon-only, pscale=2
            [2, 16, 1, (12, 24), (6, 8), "equiangular", "equiangular", 5e-2, 1e-4],  # pscale=3
            [2, 16, 1, (24, 48), (6, 12), "equiangular", "equiangular", 5e-2, 1e-4],  # pscale=4
            [2, 16, 1, (24, 48), (12, 24), "legendre-gauss", "legendre-gauss", 5e-2, 1e-4],  # LG grid, pscale=2
            # odd latitude sizes
            [2, 16, 1, (11, 24), (7, 12), "equiangular", "equiangular", 5e-2, 1e-4],  # odd-odd lat, pscale=2
            [2, 16, 1, (13, 24), (9, 8), "equiangular", "equiangular", 5e-2, 1e-4],  # odd-odd lat, pscale=3
            # upsampling: mirror of the downsampling rows above (in_shape ↔ out_shape, grid_in ↔ grid_out)
            [2, 16, 1, (12, 24), (24, 48), "equiangular", "equiangular", 5e-2, 1e-4],  # pscale_out=2
            [2, 16, 1, (12, 12), (12, 24), "equiangular", "equiangular", 5e-2, 1e-4],  # lon-only, pscale_out=2
            [2, 16, 1, (6, 8), (12, 24), "equiangular", "equiangular", 5e-2, 1e-4],  # pscale_out=3
            [2, 16, 1, (6, 12), (24, 48), "equiangular", "equiangular", 5e-2, 1e-4],  # pscale_out=4
            [2, 16, 1, (12, 24), (24, 48), "legendre-gauss", "legendre-gauss", 5e-2, 1e-4],  # LG grid, pscale_out=2
            [2, 16, 1, (7, 12), (11, 24), "equiangular", "equiangular", 5e-2, 1e-4],  # odd-odd lat, pscale_out=2
            [2, 16, 1, (9, 8), (13, 24), "equiangular", "equiangular", 5e-2, 1e-4],  # odd-odd lat, pscale_out=3
        ],
        skip_on_empty=True,
    )
    @unittest.skipUnless(cuda_kernels_is_available(), "skipping test because CUDA kernels are not available")
    def test_device_vs_cpu(self, batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol, verbose=False):
        """Tests numerical equivalence between optimized CUDA and CPU implementations"""

        if self.device.type == "cpu":
            # comparing CPU with itself does not make sense
            return

        # set seed
        set_seed(333)

        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        # Helper: create inputs
        inputs_host = {
            "k": torch.randn(batch_size, channels, nlat_in, nlon_in, requires_grad=True, dtype=torch.float32),
            "v": torch.randn(batch_size, channels, nlat_in, nlon_in, requires_grad=True, dtype=torch.float32),
            "q": torch.randn(batch_size, channels, nlat_out, nlon_out, requires_grad=True, dtype=torch.float32),
        }
        inputs_device = {k: v.detach().clone().to(self.device).requires_grad_() for k, v in inputs_host.items()}

        # reference input and model (use default local theta_cutoff so the test is sensitive
        # to the (wi + pscale*wo) % nlon_in shift; a global cutoff makes every input a neighbor
        # of every output and collapses the shift to a permutation the result is invariant to)
        att_host = NeighborhoodAttentionS2(
            grid_in=as_grid(grid_in, nlat=in_shape[0], nlon=in_shape[1]),
            grid_out=as_grid(grid_out, nlat=out_shape[0], nlon=out_shape[1]),
            in_channels=channels,
            num_heads=heads,
            bias=True,
        )

        # Device model and inputs
        att_device = NeighborhoodAttentionS2(
            grid_in=as_grid(grid_in, nlat=in_shape[0], nlon=in_shape[1]),
            grid_out=as_grid(grid_out, nlat=out_shape[0], nlon=out_shape[1]),
            in_channels=channels,
            num_heads=heads,
            bias=True,
        ).to(self.device)

        # Synchronize parameters of model
        att_device.load_state_dict(att_host.state_dict())
        for (name_host, p_host), (name_device, p_device) in zip(att_host.named_parameters(), att_device.named_parameters()):
            p_host_copy = p_host.detach().clone().cpu()
            p_device_copy = p_device.detach().clone().cpu()
            self.assertTrue(compare_tensors(f"weight {name_host}", p_device_copy, p_host_copy, atol=atol, rtol=rtol, verbose=verbose))

        # reference forward passes
        out_host = att_host(inputs_host["q"], inputs_host["k"], inputs_host["v"])
        out_device = att_device(inputs_device["q"], inputs_device["k"], inputs_device["v"])
        self.assertTrue(compare_tensors("output", out_device.cpu(), out_host.cpu(), atol=atol, rtol=rtol, verbose=verbose))

        # Backward passes
        grad = torch.randn_like(out_host)
        out_host.backward(grad)
        out_device.backward(grad.to(self.device))

        for inp in ["q", "k", "v"]:
            igrad_host = inputs_host[inp].grad.cpu()
            igrad_device = inputs_device[inp].grad.cpu()
            self.assertTrue(compare_tensors(f"input grad {inp}", igrad_device, igrad_host, atol=atol, rtol=rtol, verbose=verbose))

        # Check parameter gradient equivalence - check only q,k, v weights
        for (name_host, p_host), (name_device, p_device) in zip(att_host.named_parameters(), att_device.named_parameters()):
            grad_host = p_host.grad.cpu()
            grad_device = p_device.grad.cpu()
            self.assertTrue(compare_tensors(f"parameter grad {name_host}", grad_device, grad_host, atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            # Format: [batch_size, channels, channels_out, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol]
            [4, 4, 4, 1, (6, 12), (6, 12), "equiangular", "equiangular", 1e-2, 0],
            [4, 4, 8, 1, (6, 12), (6, 12), "equiangular", "equiangular", 1e-2, 0],
            [4, 8, 4, 1, (6, 12), (6, 12), "equiangular", "equiangular", 1e-2, 0],
            [4, 4, 4, 1, (6, 12), (6, 12), "legendre-gauss", "legendre-gauss", 1e-2, 0],
            [4, 4, 4, 1, (6, 12), (6, 12), "lobatto", "lobatto", 1e-2, 0],
        ],
        skip_on_empty=True,
    )
    def test_neighborhood_global_equivalence(self, batch_size, channels, channels_out, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol, verbose=False):
        """Tests that NeighborhoodAttentionS2 reduces to the global AttentionS2 when its neighborhood covers the whole sphere,
        and that both agree with the dense reference.

        Passing ``theta_cutoff = 2 * pi`` forces every input point into the support of every output point,
        so the sparse psi mechanism in NeighborhoodAttentionS2 becomes mathematically identical to the dense
        softmax(Q Kt) V computation in AttentionS2 (with the same quadrature weights applied).

        Cases are restricted to ``in_shape == out_shape`` because the neighborhood kernel advances the input
        column index by ``pscale * wo`` where ``pscale = nlon_in / nlon_out``; only when the two grids match
        does that shift reproduce the translation-invariant behavior of the global attention. With
        downsampling the shift maps multiple outputs to the same input column set, and the two modules are
        not expected to agree numerically.

        The two layers share their projection and layout code, so an error there would appear on both
        sides of the first comparison and cancel. Each is therefore also checked against
        ``_dense_masked_attention``, which computes the same thing with none of that code."""

        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

        # set seed
        set_seed(333)

        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        # Helper: create inputs
        inputs_ref = {
            "k": torch.randn(batch_size, channels, nlat_in, nlon_in, requires_grad=True, device=self.device, dtype=torch.float32),
            "v": torch.randn(batch_size, channels, nlat_in, nlon_in, requires_grad=True, device=self.device, dtype=torch.float32),
            "q": torch.randn(batch_size, channels, nlat_out, nlon_out, requires_grad=True, device=self.device, dtype=torch.float32),
        }
        inputs = {k: v.detach().clone().to(self.device).requires_grad_() for k, v in inputs_ref.items()}

        # reference input and model
        model_ref = AttentionS2(
            grid_in=as_grid(grid_in, nlat=in_shape[0], nlon=in_shape[1]),
            grid_out=as_grid(grid_out, nlat=out_shape[0], nlon=out_shape[1]),
            in_channels=channels,
            num_heads=heads,
            bias=False,
            out_channels=channels_out,
        ).to(self.device)

        # Device model and inputs
        model = NeighborhoodAttentionS2(
            grid_in=as_grid(grid_in, nlat=in_shape[0], nlon=in_shape[1]),
            grid_out=as_grid(grid_out, nlat=out_shape[0], nlon=out_shape[1]),
            in_channels=channels,
            num_heads=heads,
            bias=False,
            theta_cutoff=2 * torch.pi,
            out_channels=channels_out,
        )

        # Synchronize parameters of model
        model.load_state_dict(model_ref.state_dict())
        model = model.to(self.device)
        for (name_ref, p_ref), (name, p) in zip(model_ref.named_parameters(), model.named_parameters()):
            self.assertTrue(compare_tensors(f"weight {name_ref}", p, p_ref, atol=atol, rtol=rtol, verbose=verbose))

        # reference forward passes
        out_ref = model_ref(inputs_ref["q"], inputs_ref["k"], inputs_ref["v"])
        out = model(inputs["q"], inputs["k"], inputs["v"])

        # Check forward equivalence
        self.assertTrue(compare_tensors("output", out, out_ref, atol=atol, rtol=rtol, verbose=verbose))

        # Backward passes
        ograd = torch.randn_like(out_ref)
        out_ref.backward(ograd)
        out.backward(ograd.to(self.device))

        # Check input gradient equivalence
        for inp in ["q", "k", "v"]:
            grad_ref = inputs_ref[inp].grad
            grad = inputs[inp].grad
            self.assertTrue(compare_tensors(f"input grad {inp}", grad, grad_ref, atol=atol, rtol=rtol, verbose=verbose))

        # Check parameter gradient equivalence - check only q,k, v weights
        for key in ["q_weights", "k_weights", "v_weights"]:
            grad_ref = getattr(model_ref, key).grad
            grad = getattr(model, key).grad
            self.assertTrue(compare_tensors(f"parameter grad {key}", grad, grad_ref, atol=atol, rtol=rtol, verbose=verbose))

        # Both layers against the dense reference, on the flattened fields. Last, since its
        # backward accumulates into the parameter gradients compared above.
        mask = torch.ones(nlat_out * nlon_out, nlat_in * nlon_in, dtype=torch.bool, device=self.device)
        for name, layer, out_layer, inputs_layer in (("AttentionS2", model_ref, out_ref, inputs_ref), ("NeighborhoodAttentionS2", model, out, inputs)):
            dense_inputs = {k: v.detach().flatten(-2).requires_grad_() for k, v in inputs_layer.items()}
            out_dense = _dense_masked_attention(layer, dense_inputs["q"], dense_inputs["k"], dense_inputs["v"], mask)
            self.assertTrue(compare_tensors(f"{name} output vs dense", out_layer.flatten(-2), out_dense, atol=atol, rtol=rtol, verbose=verbose))

            out_dense.backward(ograd.flatten(-2))
            for inp in ["q", "k", "v"]:
                grad_layer = inputs_layer[inp].grad.flatten(-2)
                self.assertTrue(compare_tensors(f"{name} input grad {inp} vs dense", grad_layer, dense_inputs[inp].grad, atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            [None],
            ["tensor"],
        ],
        skip_on_empty=True,
    )
    def test_attention_eval_disables_sdpa_dropout(self, scale_mode, verbose=False):
        set_seed(333)

        scale = None
        if scale_mode == "tensor":
            scale = torch.tensor(0.5, device=self.device, dtype=torch.float32)

        model = AttentionS2(
            grid_in=as_grid("equiangular", nlat=4, nlon=8),
            grid_out=as_grid("equiangular", nlat=4, nlon=8),
            in_channels=4,
            num_heads=1,
            scale=scale,
            bias=False,
            out_channels=4,
            drop_rate=0.9,
        ).to(self.device)
        model.eval()

        inputs = torch.randn(2, 4, 4, 8, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            torch.manual_seed(101)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(101)
            out_a = model(inputs)
            torch.manual_seed(202)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(202)
            out_b = model(inputs)

        self.assertTrue(compare_tensors("eval disables SDPA dropout", out_a, out_b, atol=0.0, rtol=0.0, verbose=verbose))

    @parameterized.expand(
        [
            # Format: [batch_size, channels_in, channels_out, shape, grid, atol, rtol]
            [2, 4, 4, (4, 8), "equiangular", 1e-5, 1e-4],
            [2, 4, 8, (4, 8), "equiangular", 1e-5, 1e-4],
            [2, 8, 4, (4, 8), "equiangular", 1e-5, 1e-4],
            [2, 4, 4, (6, 12), "legendre-gauss", 1e-5, 1e-4],
            [2, 4, 4, (6, 12), "lobatto", 1e-5, 1e-4],
        ],
        skip_on_empty=True,
    )
    def test_self_attention_kernel_equivalence(self, batch_size, channels_in, channels_out, shape, grid, atol, rtol, verbose=False):
        """For nlat_in == nlat_out and nlon_in == nlon_out the gather (downsample) and
        scatter (upsample) torch reference math kernels must produce numerically identical
        forward and backward results: the p-shift collapses to pscale = pscale_out = 1, so
        both formulations describe the same self-attention computation, just with the psi
        sparsity pattern stored as either rows-by-output (gather) or rows-by-input (scatter).

        This is a pure-Python equivalence check on the math fns themselves; the C++/CUDA
        dispatcher always routes self-attention through the gather kernel, so this test
        exercises the upsample math that the dispatcher would otherwise hide.

        Runs same-device-vs-same-device on each parameterized device (CPU or CUDA). No
        cross-device comparison.
        """

        set_seed(333)

        nlat, nlon = shape
        # head-folded shapes; pass directly to the math fns (which expect [B, C, H, W]).
        kw = torch.randn(batch_size, channels_in, nlat, nlon, dtype=torch.float32, device=self.device)
        vw = torch.randn(batch_size, channels_out, nlat, nlon, dtype=torch.float32, device=self.device)
        qw = torch.randn(batch_size, channels_in, nlat, nlon, dtype=torch.float32, device=self.device)

        # quadrature weights on the (input == output) grid
        _, wgl = precompute_latitudes(nlat, grid=grid)
        ring_weights = (2.0 * torch.pi * wgl.to(torch.float32) / nlon).to(self.device)

        # neighborhood pattern; theta_cutoff heuristics for self-attention agree (nlat_in == nlat_out)
        fb = get_filter_basis(kernel_shape=1, basis_type="zernike")
        theta_cutoff = math.pi / float(nlat - 1)

        # gather psi (rows by ho, cols by hi*nlon + wi_canonical)
        idx_g, _, roff_g = _precompute_convolution_tensor_s2(
            as_grid(grid, nlat=shape[0], nlon=shape[1]),
            as_grid(grid, nlat=shape[0], nlon=shape[1]),
            fb,
            theta_cutoff=theta_cutoff,
            transpose_normalization=False,
            basis_norm_mode="none",
            merge_quadrature=True,
        )
        col_g = idx_g[2].contiguous().to(self.device)
        roff_g = roff_g.contiguous().to(self.device)

        # scatter psi (rows by hi, cols by ho*nlon + wo_canonical) — shapes swapped + transpose_normalization=True
        idx_s, _, roff_s = _precompute_convolution_tensor_s2(
            as_grid(grid, nlat=shape[0], nlon=shape[1]),
            as_grid(grid, nlat=shape[0], nlon=shape[1]),
            fb,
            theta_cutoff=theta_cutoff,
            transpose_normalization=True,
            basis_norm_mode="none",
            merge_quadrature=True,
        )
        col_s = idx_s[2].contiguous().to(self.device)
        roff_s = roff_s.contiguous().to(self.device)

        # ---- forward ----
        y_g = _neighborhood_s2_attention_regular_fwd_torch(kw, vw, qw, ring_weights, col_g, roff_g, nlon, nlat, nlon)
        y_s = _neighborhood_s2_attention_upsample_fwd_torch(kw, vw, qw, ring_weights, col_s, roff_s, nlon, nlat, nlon)

        self.assertTrue(compare_tensors("fwd output (gather vs scatter)", y_s, y_g, atol=atol, rtol=rtol, verbose=verbose))

        # ---- backward (dvx, dkx, dqy individually) ----
        dy = torch.randn_like(y_g)

        dvx_g = _neighborhood_s2_attention_regular_bwd_dv_torch(kw, vw, qw, dy, ring_weights, col_g, roff_g, nlon, nlat, nlon)
        dkx_g = _neighborhood_s2_attention_regular_bwd_dk_torch(kw, vw, qw, dy, ring_weights, col_g, roff_g, nlon, nlat, nlon)
        dqy_g = _neighborhood_s2_attention_regular_bwd_dq_torch(kw, vw, qw, dy, ring_weights, col_g, roff_g, nlon, nlat, nlon)

        dvx_s = _neighborhood_s2_attention_upsample_bwd_dv_torch(kw, vw, qw, dy, ring_weights, col_s, roff_s, nlon, nlat, nlon)
        dkx_s = _neighborhood_s2_attention_upsample_bwd_dk_torch(kw, vw, qw, dy, ring_weights, col_s, roff_s, nlon, nlat, nlon)
        dqy_s = _neighborhood_s2_attention_upsample_bwd_dq_torch(kw, vw, qw, dy, ring_weights, col_s, roff_s, nlon, nlat, nlon)

        self.assertTrue(compare_tensors("bwd dv (gather vs scatter)", dvx_s, dvx_g, atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors("bwd dk (gather vs scatter)", dkx_s, dkx_g, atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors("bwd dq (gather vs scatter)", dqy_s, dqy_g, atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            # Format: [batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol]
            # one row per dispatcher code path (gather / scatter); pscale=2 covers
            # the wip = (wi + pscale*wo) % nlon_in shift, which the trivial self-attention
            # case (pscale=1) would not.
            [4, 4, 1, (12, 24), (6, 12), "equiangular", "equiangular", 1e-2, 0],  # downsample, pscale=2
            [4, 4, 1, (6, 12), (12, 24), "equiangular", "equiangular", 1e-2, 0],  # upsample, pscale_out=2
            # Same two paths with heads packed along the channel dimension. The op
            # contract differs between these and the heads=1 rows above: the registered
            # fake derives the output extent from vw.shape[3], which is num_heads*C
            # rather than C, so a packing mistake there is invisible until num_heads>1.
            # test_faketensor and test_aot_dispatch_dynamic are what catch it.
            [4, 8, 4, (12, 24), (6, 12), "equiangular", "equiangular", 1e-2, 0],  # downsample, packed heads
            [4, 8, 4, (6, 12), (12, 24), "equiangular", "equiangular", 1e-2, 0],  # upsample, packed heads
        ],
        skip_on_empty=True,
    )
    def test_optimized_pt2_compatibility(self, batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol, verbose=False):
        """Tests whether the optimized kernels are PyTorch 2 compatible"""

        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping GPU test because CUDA kernels are not available")

        set_seed(333)

        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        att = NeighborhoodAttentionS2(
            grid_in=as_grid(grid_in, nlat=in_shape[0], nlon=in_shape[1]),
            grid_out=as_grid(grid_out, nlat=out_shape[0], nlon=out_shape[1]),
            in_channels=channels,
            num_heads=heads,
            bias=False,
            optimized_kernel=True,
        ).to(self.device)

        inputs = {
            "k": torch.randn(batch_size, channels, nlat_in, nlon_in, requires_grad=True, device=self.device, dtype=torch.float32),
            "v": torch.randn(batch_size, channels, nlat_in, nlon_in, requires_grad=True, device=self.device, dtype=torch.float32),
            "q": torch.randn(batch_size, channels, nlat_out, nlon_out, requires_grad=True, device=self.device, dtype=torch.float32),
        }

        kw = F.conv2d(inputs["k"], att.k_weights, att.k_bias)
        vw = F.conv2d(inputs["v"], att.v_weights, att.v_bias)
        qw = F.conv2d(inputs["q"], att.q_weights, att.q_bias) * att.scale

        # The op takes physical NHWC with heads packed along the channel dimension;
        # the projections above are channels-first, so convert exactly as the module
        # does at its boundary. Heads are NOT folded into the batch dimension -- the
        # op receives num_heads and addresses a head in place.
        test_inputs = (
            to_nhwc(kw),
            to_nhwc(vw),
            to_nhwc(qw),
            att.ring_weights,
            att.psi_seg,
            att.psi_seg_off,
            att.num_heads,
            nlon_in,
            nlat_out,
            nlon_out,
        )

        opcheck(torch.ops.attention_kernels._neighborhood_s2_attention_regular_optimized, test_inputs)

        # The raw kernels the custom op wraps have fakes of their own, which tracing reads
        # whenever it sees through the wrapper, and checking the wrapper never runs them.
        # They have no autograd registration (that is the wrapper's), so their inputs are
        # detached, which keeps test_autograd_registration from demanding one.
        raw_inputs = tuple(t.detach() if isinstance(t, torch.Tensor) else t for t in test_inputs)
        opcheck(torch.ops.attention_kernels.forward_regular, raw_inputs)
        dy = torch.randn_like(torch.ops.attention_kernels.forward_regular(*raw_inputs))
        opcheck(torch.ops.attention_kernels.backward_regular, (*raw_inputs[:3], dy, *raw_inputs[3:]))

    @parameterized.expand(
        [
            # Format: [batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out]
            # same-shape (pscale=1) and downsampling (pscale=2) cases
            [4, 4, 1, (6, 12), (6, 12), "equiangular", "equiangular"],
            [4, 8, 4, (6, 12), (6, 12), "equiangular", "equiangular"],
            [4, 4, 1, (12, 24), (6, 12), "equiangular", "equiangular"],
        ],
        skip_on_empty=True,
    )
    def test_ring_kernels_pt2_compatibility(self, batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, verbose=False):
        """Tests whether the ring-step CUDA kernels (used by DistributedNeighborhoodAttentionS2)
        are PyTorch 2 compatible.

        Only the local CUDA kernels are exercised — the ring exchange itself (NCCL P2P) is not
        tested here. With az_size = 1 a single ring step covers the full longitude, so we can call
        the kernels in single-rank mode (lon_lo_kx=0, lat_halo_start=0, no halo padding).

        opcheck only verifies the op contract (schema, fake tensors, AOT dispatch); the input
        tensor values do not need to be numerically meaningful, so kw/vw/qw and the gradient /
        state buffers are allocated directly with the right shapes."""

        if self.device.type != "cuda":
            raise unittest.SkipTest("ring kernels are only registered for CUDA")
        if not cuda_kernels_is_available():
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

        set_seed(333)

        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        # Build the module just to get a consistent (ring_weights, neighbourhood) for the
        # chosen grid; we do not exercise its forward. On a single rank the local arcs are
        # the serial ones: every output row is local and the arc starts need no shift.
        att = NeighborhoodAttentionS2(
            grid_in=as_grid(grid_in, nlat=in_shape[0], nlon=in_shape[1]),
            grid_out=as_grid(grid_out, nlat=out_shape[0], nlon=out_shape[1]),
            in_channels=channels,
            num_heads=heads,
            bias=False,
            optimized_kernel=True,
        ).to(self.device)
        arcs = att._neighborhood_arcs()
        psi_seg, psi_seg_off = arcs.segments.contiguous().to(self.device), arcs.offsets.contiguous().to(self.device)

        # Packed channel counts, heads along the channel dim as in the serial kernels.
        C_k = channels
        C_v = channels

        # Synthetic projected k/v/q with the correct kernel-side shapes.
        kw = torch.randn(batch_size, nlat_in, nlon_in, C_k, device=self.device, dtype=torch.float32)
        vw = torch.randn(batch_size, nlat_in, nlon_in, C_v, device=self.device, dtype=torch.float32)
        qw = torch.randn(batch_size, nlat_out, nlon_out, C_k, device=self.device, dtype=torch.float32)

        # ---- forward ring step ----
        # State buffers in the kernels' ABI: per-point vectors packed like the activations,
        # softmax statistics per (batch, head, point).
        y_acc = torch.zeros(batch_size, nlat_out, nlon_out, C_v, device=self.device, dtype=torch.float32)
        alpha_sum = torch.zeros(batch_size, heads, nlat_out, nlon_out, device=self.device, dtype=torch.float32)
        qdotk_max = torch.full((batch_size, heads, nlat_out, nlon_out), float("-inf"), device=self.device, dtype=torch.float32)

        fwd_inputs = (
            kw,
            vw,
            qw,
            y_acc,
            alpha_sum,
            qdotk_max,
            att.ring_weights,
            psi_seg,
            psi_seg_off,
            heads,
            nlon_in,
            nlon_out,
            0,
            0,
            nlat_out,
            nlon_out,
        )

        opcheck(torch.ops.attention_kernels.forward_ring_step, fwd_inputs)

        # ---- backward pass 1: re-accumulate softmax stats + alpha_k / alpha_kvw ----
        dy = torch.randn(batch_size, nlat_out, nlon_out, C_v, device=self.device, dtype=torch.float32)

        bwd_alpha_sum = torch.zeros(batch_size, heads, nlat_out, nlon_out, device=self.device, dtype=torch.float32)
        bwd_qdotk_max = torch.full((batch_size, heads, nlat_out, nlon_out), float("-inf"), device=self.device, dtype=torch.float32)
        integral_buf = torch.zeros(batch_size, heads, nlat_out, nlon_out, device=self.device, dtype=torch.float32)
        alpha_k_buf = torch.zeros(batch_size, nlat_out, nlon_out, C_k, device=self.device, dtype=torch.float32)
        alpha_kvw_buf = torch.zeros(batch_size, nlat_out, nlon_out, C_k, device=self.device, dtype=torch.float32)

        bwd1_inputs = (
            kw,
            vw,
            qw,
            dy,
            bwd_alpha_sum,
            bwd_qdotk_max,
            integral_buf,
            alpha_k_buf,
            alpha_kvw_buf,
            att.ring_weights,
            psi_seg,
            psi_seg_off,
            heads,
            nlon_in,
            nlon_out,
            0,
            0,
            nlat_out,
            nlon_out,
        )

        opcheck(torch.ops.attention_kernels.backward_ring_step_pass1, bwd1_inputs)

        # ---- backward pass 2: scatter dkx/dvx using finalized stats ----
        # Synthetic but well-formed stats from a previous pass1: avoid -inf in qdotk_max and 0 in
        # alpha_sum so the kernel doesn't produce NaNs (opcheck doesn't check numerics, but NaNs
        # can interact badly with AOT dispatch comparisons).
        fwd_alpha_sum = torch.ones(batch_size, heads, nlat_out, nlon_out, device=self.device, dtype=torch.float32)
        fwd_qdotk_max = torch.zeros(batch_size, heads, nlat_out, nlon_out, device=self.device, dtype=torch.float32)
        integral_norm = torch.zeros(batch_size, heads, nlat_out, nlon_out, device=self.device, dtype=torch.float32)

        dkw = torch.zeros(batch_size, nlat_in, nlon_in, C_k, device=self.device, dtype=torch.float32)
        dvw = torch.zeros(batch_size, nlat_in, nlon_in, C_v, device=self.device, dtype=torch.float32)

        bwd2_inputs = (
            kw,
            vw,
            qw,
            dy,
            fwd_alpha_sum,
            fwd_qdotk_max,
            integral_norm,
            dkw,
            dvw,
            att.ring_weights,
            psi_seg,
            psi_seg_off,
            heads,
            nlon_in,
            nlon_out,
            0,
            0,
            nlat_out,
            nlon_out,
        )

        opcheck(torch.ops.attention_kernels.backward_ring_step_pass2, bwd2_inputs)

    @parameterized.expand(
        [
            # Format: [batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out]
            # upsampling (scatter) cases, pscale_out=2
            [4, 4, 1, (6, 12), (12, 24), "equiangular", "equiangular"],
            [4, 8, 4, (6, 12), (12, 24), "equiangular", "equiangular"],
        ],
        skip_on_empty=True,
    )
    def test_ring_upsample_kernels_pt2_compatibility(self, batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, verbose=False):
        """Tests whether the upsample (scatter) ring-step CUDA kernels (used by
        DistributedNeighborhoodAttentionS2 in the upsample direction) are PyTorch 2 compatible.

        Only the local CUDA kernels are exercised — the ring exchange itself (NCCL P2P) is not
        tested here. With az_size = polar_size = 1 a single ring step covers the full longitude
        and the serial scatter psi coincides with the local psi built by
        RingUpsampleBackend.prepare (lat_lo_out = lon_lo_out = 0, no halo padding), so we can call
        the kernels in single-rank mode (lon_lo_kx=0, lat_halo_start=0).

        opcheck only verifies the op contract (schema, fake tensors, AOT dispatch); the input
        tensor values do not need to be numerically meaningful, so kw/vw/qw and the gradient /
        state buffers are allocated directly with the right shapes."""

        if self.device.type != "cuda":
            raise unittest.SkipTest("ring kernels are only registered for CUDA")
        if not cuda_kernels_is_available():
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

        set_seed(333)

        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        # Build the module just to get a consistent (ring_weights, neighbourhood) for the
        # chosen grid; we do not exercise its forward. For upsample shapes the neighbourhood
        # is the scatter psi (rows keyed by hi, arcs on the output grid).
        att = NeighborhoodAttentionS2(
            grid_in=as_grid(grid_in, nlat=in_shape[0], nlon=in_shape[1]),
            grid_out=as_grid(grid_out, nlat=out_shape[0], nlon=out_shape[1]),
            in_channels=channels,
            num_heads=heads,
            bias=False,
            optimized_kernel=True,
        ).to(self.device)
        self.assertTrue(att.upsample)
        arcs = att._neighborhood_arcs()
        psi_seg, psi_seg_off = arcs.segments.contiguous().to(self.device), arcs.offsets.contiguous().to(self.device)

        # Packed channel counts, heads along the channel dim as in the serial kernels.
        C_k = channels
        C_v = channels

        # Synthetic projected k/v/q with the correct kernel-side shapes.
        kw = torch.randn(batch_size, nlat_in, nlon_in, C_k, device=self.device, dtype=torch.float32)
        vw = torch.randn(batch_size, nlat_in, nlon_in, C_v, device=self.device, dtype=torch.float32)
        qw = torch.randn(batch_size, nlat_out, nlon_out, C_k, device=self.device, dtype=torch.float32)

        # ---- forward ring step ----
        # State buffers in the kernels' ABI: per-point vectors packed like the activations,
        # softmax statistics per (batch, head, point).
        y_acc = torch.zeros(batch_size, nlat_out, nlon_out, C_v, device=self.device, dtype=torch.float32)
        alpha_sum = torch.zeros(batch_size, heads, nlat_out, nlon_out, device=self.device, dtype=torch.float32)
        qdotk_max = torch.full((batch_size, heads, nlat_out, nlon_out), float("-inf"), device=self.device, dtype=torch.float32)

        fwd_inputs = (
            kw,
            vw,
            qw,
            y_acc,
            alpha_sum,
            qdotk_max,
            att.ring_weights,
            psi_seg,
            psi_seg_off,
            heads,
            nlon_in,
            nlon_out,
            0,
            0,
            nlat_out,
            nlon_out,
        )

        opcheck(torch.ops.attention_kernels.forward_ring_step_upsample, fwd_inputs)

        # ---- backward pass 1: scatter integral / alpha_k / alpha_kvw stats ----
        # Uses the forward-final qdotk_max; synthetic but well-formed values (no -inf) so the
        # kernel doesn't produce NaNs (opcheck doesn't check numerics, but NaNs can interact
        # badly with AOT dispatch comparisons).
        dy = torch.randn(batch_size, nlat_out, nlon_out, C_v, device=self.device, dtype=torch.float32)

        fwd_qdotk_max = torch.zeros(batch_size, heads, nlat_out, nlon_out, device=self.device, dtype=torch.float32)
        integral_buf = torch.zeros(batch_size, heads, nlat_out, nlon_out, device=self.device, dtype=torch.float32)
        alpha_k_buf = torch.zeros(batch_size, nlat_out, nlon_out, C_k, device=self.device, dtype=torch.float32)
        alpha_kvw_buf = torch.zeros(batch_size, nlat_out, nlon_out, C_k, device=self.device, dtype=torch.float32)

        bwd1_inputs = (
            kw,
            vw,
            qw,
            dy,
            fwd_qdotk_max,
            integral_buf,
            alpha_k_buf,
            alpha_kvw_buf,
            att.ring_weights,
            psi_seg,
            psi_seg_off,
            heads,
            nlon_in,
            nlon_out,
            0,
            0,
            nlat_out,
            nlon_out,
        )

        opcheck(torch.ops.attention_kernels.backward_ring_step_upsample_pass1, bwd1_inputs)

        # ---- backward pass 2: accumulate chunk-local dkx/dvx using finalized stats ----
        fwd_alpha_sum = torch.ones(batch_size, heads, nlat_out, nlon_out, device=self.device, dtype=torch.float32)
        integral_norm = torch.zeros(batch_size, heads, nlat_out, nlon_out, device=self.device, dtype=torch.float32)

        dkw = torch.zeros(batch_size, nlat_in, nlon_in, C_k, device=self.device, dtype=torch.float32)
        dvw = torch.zeros(batch_size, nlat_in, nlon_in, C_v, device=self.device, dtype=torch.float32)

        bwd2_inputs = (
            kw,
            vw,
            qw,
            dy,
            fwd_alpha_sum,
            fwd_qdotk_max,
            integral_norm,
            dkw,
            dvw,
            att.ring_weights,
            psi_seg,
            psi_seg_off,
            heads,
            nlon_in,
            nlon_out,
            0,
            0,
            nlat_out,
            nlon_out,
        )

        opcheck(torch.ops.attention_kernels.backward_ring_step_upsample_pass2, bwd2_inputs)

    def test_wrong_shape_assertions(self):
        """Verify that forward raises RuntimeError on spatial-shape mismatches."""
        B, C = 2, 16
        in_shape = (12, 24)
        out_shape = (6, 12)
        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        model = NeighborhoodAttentionS2(
            grid_in=as_grid("equiangular", nlat=in_shape[0], nlon=in_shape[1]),
            grid_out=as_grid("equiangular", nlat=out_shape[0], nlon=out_shape[1]),
            in_channels=C,
            num_heads=1,
            bias=False,
        ).to(self.device)

        q = torch.randn(B, C, nlat_out, nlon_out, device=self.device)
        kv = torch.randn(B, C, nlat_in, nlon_in, device=self.device)

        # 1. Self-attention on an up/downsampling module: a single tensor cannot
        #    simultaneously satisfy in_shape (for k/v) and out_shape (for q).
        with self.assertRaises(RuntimeError):
            model(q)  # key defaults to query, but key must have in_shape

        # 2. q_shape == k_shape != v_shape: key carries out_shape instead of in_shape.
        with self.assertRaises(RuntimeError):
            model(q, q, kv)

        # 3. q_shape == v_shape != k_shape: value carries out_shape instead of in_shape.
        with self.assertRaises(RuntimeError):
            model(q, kv, q)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.device_count() >= 2 and cuda_kernels_is_available(),
    "needs two CUDA devices and the CUDA kernels",
)
class TestCrossDeviceExecution(unittest.TestCase):
    """
    A module on a GPU other than the current one must run there.

    The ops launch on the current stream and allocate scratch on the current device, so
    without a device guard a module on cuda:1 in a process whose current device is cuda:0
    would enqueue cuda:1 pointers on cuda:0 -- an illegal access or wrong results. The
    reference is the same module, with the same weights, on the current device.
    """

    @parameterized.expand(
        [
            ["gather", as_grid("equiangular", nlat=16, nlon=32), as_grid("equiangular", nlat=16, nlon=32)],
            ["downsample", as_grid("equiangular", nlat=16, nlon=32), as_grid("equiangular", nlat=8, nlon=16)],
            ["upsample", as_grid("equiangular", nlat=8, nlon=16), as_grid("equiangular", nlat=16, nlon=32)],
            ["ragged", HealpixGrid(nside=4), HealpixGrid(nside=4)],
        ]
    )
    def test_a_module_on_another_gpu_runs_on_it(self, name, grid_in, grid_out):
        set_seed(333)
        cur, other = torch.device("cuda", 0), torch.device("cuda", 1)

        ref = NeighborhoodAttentionS2(grid_in=grid_in, grid_out=grid_out, in_channels=8, num_heads=2).to(cur)
        mod = NeighborhoodAttentionS2(grid_in=grid_in, grid_out=grid_out, in_channels=8, num_heads=2).to(other)
        mod.load_state_dict(ref.state_dict())
        self.assertTrue(mod.optimized_kernel)

        q = torch.randn(2, 8, *grid_out.shape)
        kv = torch.randn(2, 8, *grid_in.shape)

        def run(module, device):
            qd, kd = (t.to(device).requires_grad_(True) for t in (q, kv))
            out = module(qd, kd, kd)
            out.square().sum().backward()
            return out.detach().cpu(), qd.grad.cpu(), kd.grad.cpu()

        # other suites may have moved the current device, so pin it here
        with torch.cuda.device(cur):
            expected = run(ref, cur)
            got = run(mod, other)
            torch.cuda.synchronize(other)
            # the guard restores the current device when the op returns
            self.assertEqual(torch.cuda.current_device(), cur.index)
        for label, g, e in zip(("output", "grad q", "grad kv"), got, expected):
            with self.subTest(what=label):
                self.assertTrue(compare_tensors(f"{name} {label}", g, e, atol=1e-5, rtol=1e-4))


@parameterized_class(("device"), _devices)
class TestKeyBias(unittest.TestCase):
    """
    Without qk-norm a key bias adds ``q_i . b`` to every score of query ``i``, a shift the
    softmax removes, so the layers carry one only under qk-norm. Checkpoints from before
    that change still hold one, and must load and compute the same.
    """

    def setUp(self):
        disable_tf32()
        set_seed(333)

    _cases = [
        # Format: [name, layer, grid_in, grid_out]
        ["neighborhood regular", NeighborhoodAttentionS2, as_grid("equiangular", nlat=8, nlon=16), as_grid("equiangular", nlat=8, nlon=16)],
        ["neighborhood ragged", NeighborhoodAttentionS2, HealpixGrid(nside=2), as_grid("equiangular", nlat=8, nlon=16)],
        ["global regular", AttentionS2, as_grid("legendre-gauss", nlat=8, nlon=16), as_grid("equiangular", nlat=6, nlon=12)],
        ["global ragged", AttentionS2, HealpixGrid(nside=2), HealpixGrid(nside=2)],
    ]

    @parameterized.expand(_cases, skip_on_empty=True)
    def test_it_exists_only_under_qknorm(self, name, layer, grid_in, grid_out):
        for use_qknorm in (False, True):
            model = layer(grid_in=grid_in, grid_out=grid_out, in_channels=4, num_heads=2, use_qknorm=use_qknorm, bias=True)
            self.assertEqual(model.k_bias is not None, use_qknorm, f"{name}, use_qknorm={use_qknorm}")
            self.assertEqual("k_bias" in model.state_dict(), use_qknorm, f"{name}, use_qknorm={use_qknorm}")
            # the other biases are unaffected
            for other in ("q_bias", "v_bias", "proj_bias"):
                self.assertIsNotNone(getattr(model, other), f"{name}: {other}")

    @parameterized.expand(_cases, skip_on_empty=True)
    def test_an_old_checkpoint_loads_and_its_key_bias_was_inert(self, name, layer, grid_in, grid_out):
        model = layer(grid_in=grid_in, grid_out=grid_out, in_channels=4, num_heads=2, bias=True).to(self.device)

        def make(grid):
            return torch.randn(2, 4, *grid.shape, device=self.device)

        q, kv = make(grid_out), make(grid_in)
        with torch.no_grad():
            out = model(q, kv, kv)

        # an old checkpoint, with a key bias far from its zero initialization
        old = model.state_dict()
        old["k_bias"] = 10 * torch.randn(model.k_channels, device=self.device)

        # inert: the same layer with that bias in its projection computes the same output
        with_bias = layer(grid_in=grid_in, grid_out=grid_out, in_channels=4, num_heads=2, bias=True).to(self.device)
        with_bias.k_bias = torch.nn.Parameter(old["k_bias"].clone())
        with_bias.load_state_dict(old)
        with torch.no_grad():
            self.assertTrue(compare_tensors(f"{name} output with a key bias", with_bias(q, kv, kv), out, atol=1e-5, rtol=1e-4))

        # and loadable: strict loading drops it rather than reporting an unexpected key
        loaded = layer(grid_in=grid_in, grid_out=grid_out, in_channels=4, num_heads=2, bias=True).to(self.device)
        loaded.load_state_dict(old, strict=True)
        self.assertIsNone(loaded.k_bias)
        with torch.no_grad():
            self.assertTrue(torch.equal(loaded(q, kv, kv), out), f"{name}: output after loading an old checkpoint")


@parameterized_class(("device"), _devices)
class TestScale(unittest.TestCase):
    """
    The logit scale: ``1/sqrt(channels per head)`` unless given, a positive number, or a
    0-dimensional tensor, which the layer trains when it is an ``nn.Parameter``.
    """

    def setUp(self):
        disable_tf32()
        set_seed(333)

    _grids = [
        # Format: [name, layer, grid_in, grid_out]
        ["neighborhood regular", NeighborhoodAttentionS2, as_grid("equiangular", nlat=8, nlon=16), as_grid("equiangular", nlat=8, nlon=16)],
        ["neighborhood ragged", NeighborhoodAttentionS2, HealpixGrid(nside=2), as_grid("equiangular", nlat=8, nlon=16)],
        ["global regular", AttentionS2, as_grid("legendre-gauss", nlat=8, nlon=16), as_grid("equiangular", nlat=6, nlon=12)],
        ["global ragged", AttentionS2, HealpixGrid(nside=2), HealpixGrid(nside=2)],
    ]

    @staticmethod
    def _build(layer, grid_in, grid_out, **kwargs):
        return layer(grid_in=grid_in, grid_out=grid_out, in_channels=8, num_heads=2, **kwargs)

    @parameterized.expand(_grids, skip_on_empty=True)
    def test_the_default_is_one_over_root_head_channels(self, name, layer, grid_in, grid_out):
        model = self._build(layer, grid_in, grid_out)
        self.assertIsInstance(model.scale, float, name)
        self.assertEqual(model.scale, 1.0 / math.sqrt(4), name)

    @parameterized.expand(
        [[f"{name}, {kind}", layer, grid_in, grid_out, kind] for name, layer, grid_in, grid_out in _grids for kind in ("number", "tensor", "parameter")],
        skip_on_empty=True,
    )
    def test_a_custom_scale_matches_the_oracle(self, name, layer, grid_in, grid_out, kind, atol=1e-5, rtol=1e-4):
        scale = {"number": 0.3, "tensor": torch.tensor(0.3), "parameter": torch.nn.Parameter(torch.tensor(0.3))}[kind]
        model = self._build(layer, grid_in, grid_out, scale=scale).to(self.device)
        # a parameter is registered, so the optimizer sees it and .to() moves it
        self.assertEqual("scale" in dict(model.named_parameters()), kind == "parameter", name)

        if layer is NeighborhoodAttentionS2:
            mask = _brute_force_neighborhood(grid_in, grid_out, model.theta_cutoff).to(self.device)
        else:
            mask = torch.ones(grid_out.npoints, grid_in.npoints, dtype=torch.bool, device=self.device)

        def as_grid_shape(tensor, grid):
            return tensor if not grid.is_regular else tensor.unflatten(-1, grid.shape)

        flat = {
            "q": torch.randn(2, 8, grid_out.npoints, device=self.device, requires_grad=True),
            "k": torch.randn(2, 8, grid_in.npoints, device=self.device, requires_grad=True),
            "v": torch.randn(2, 8, grid_in.npoints, device=self.device, requires_grad=True),
        }
        flat_ref = {key: t.detach().clone().requires_grad_() for key, t in flat.items()}

        out = model(as_grid_shape(flat["q"], grid_out), as_grid_shape(flat["k"], grid_in), as_grid_shape(flat["v"], grid_in)).flatten(2)
        out_ref = _dense_masked_attention(model, flat_ref["q"], flat_ref["k"], flat_ref["v"], mask)
        self.assertTrue(compare_tensors(f"{name} output", out, out_ref, atol=atol, rtol=rtol))

        # one backward per graph, so the scale's gradients from the two do not accumulate
        grad = torch.randn_like(out_ref)
        trained = [model.scale] if kind == "parameter" else []
        grads = torch.autograd.grad(out, list(flat.values()) + trained, grad_outputs=grad)
        grads_ref = torch.autograd.grad(out_ref, list(flat_ref.values()) + trained, grad_outputs=grad)
        for key, got, expected in zip(list(flat) + ["scale"], grads, grads_ref):
            self.assertTrue(compare_tensors(f"{name} grad {key}", got, expected, atol=atol, rtol=rtol))
        if trained:
            self.assertGreater(float(grads[-1].abs()), 0.0, f"{name}: the scale received no gradient")

        # the scale is used at all: the same weights at the default scale compute something else
        default = self._build(layer, grid_in, grid_out).to(self.device)
        default.load_state_dict(model.state_dict(), strict=kind != "parameter")
        with torch.no_grad():
            out_default = default(as_grid_shape(flat["q"], grid_out), as_grid_shape(flat["k"], grid_in), as_grid_shape(flat["v"], grid_in)).flatten(2)
        self.assertGreater(float((out_default - out.detach()).abs().max()), 1e-2, f"{name}: the scale did not change the output")

    @parameterized.expand(
        [
            # Format: [name, scale, error]
            ["a vector", torch.tensor([0.5, 0.5]), ValueError],
            ["a bool", True, TypeError],
            ["a string", "0.5", TypeError],
            ["zero", 0.0, ValueError],
            ["a negative number", -0.5, ValueError],
            ["infinity", math.inf, ValueError],
            ["nan", math.nan, ValueError],
            # a tensor is held to the same values as a number
            ["a nan tensor", torch.tensor(math.nan), ValueError],
            ["an infinite tensor", torch.tensor(math.inf), ValueError],
            ["a zero tensor", torch.tensor(0.0), ValueError],
            ["a negative parameter", torch.nn.Parameter(torch.tensor(-0.5)), ValueError],
            ["a nan parameter", torch.nn.Parameter(torch.tensor(math.nan)), ValueError],
        ],
        skip_on_empty=True,
    )
    def test_it_rejects(self, name, scale, error):
        for layer in (NeighborhoodAttentionS2, AttentionS2):
            with self.subTest(layer=layer.__name__), self.assertRaises(error):
                self._build(layer, as_grid("equiangular", nlat=8, nlon=16), as_grid("equiangular", nlat=8, nlon=16), scale=scale)

    def test_a_meta_tensor_has_no_value_to_check(self):
        # building a layer on the meta device must not fail on the scale
        for layer in (NeighborhoodAttentionS2, AttentionS2):
            with self.subTest(layer=layer.__name__):
                model = self._build(layer, as_grid("equiangular", nlat=8, nlon=16), as_grid("equiangular", nlat=8, nlon=16), scale=torch.empty((), device="meta"))
                self.assertTrue(model.scale.is_meta)

    def test_it_accepts_any_real_number(self):
        for scale in (1, 0.5, np.float32(0.25), np.float64(0.125)):
            model = self._build(AttentionS2, as_grid("equiangular", nlat=8, nlon=16), as_grid("equiangular", nlat=8, nlon=16), scale=scale)
            self.assertIsInstance(model.scale, float)
            self.assertEqual(model.scale, float(scale))


@parameterized_class(("device"), _devices)
class TestEmptyNeighborhood(unittest.TestCase):
    """
    A cutoff below the input spacing leaves output points between input points with no
    neighbour at all. Their softmax has an empty sum, and every path -- the references,
    the CPU and CUDA kernels, regular and ragged -- returns zero output and zero gradient
    for them rather than dividing by that zero and returning NaN.
    """

    def setUp(self):
        disable_tf32()
        set_seed(333)

    @parameterized.expand(
        [
            ["ragged", HealpixGrid(nside=2), HealpixGrid(nside=8)],
            ["regular upsample", as_grid("equiangular", nlat=4, nlon=8), as_grid("equiangular", nlat=16, nlon=32)],
            ["regular downsample", as_grid("legendre-gauss", nlat=16, nlon=32), as_grid("equiangular", nlat=8, nlon=16)],
        ]
    )
    def test_the_optimized_path_matches_the_reference(self, name, grid_in, grid_out, atol=1e-5, rtol=1e-3):
        cutoff = 0.25 * grid_in.max_node_spacing
        model, ref = (
            NeighborhoodAttentionS2(grid_in=grid_in, grid_out=grid_out, in_channels=4, num_heads=2, theta_cutoff=cutoff, optimized_kernel=optimized).to(self.device)
            for optimized in (True, False)
        )
        if not model.optimized_kernel:
            raise unittest.SkipTest("needs the compiled kernels")
        ref.load_state_dict(model.state_dict())

        # the case under test exists: some rows empty, but not all of them
        _, roff_idx = precompute_neighborhood_csr_s2(grid_in, grid_out, model.theta_cutoff)
        empty = roff_idx[1:] == roff_idx[:-1]
        self.assertTrue(bool(empty.any()) and not bool(empty.all()), f"{name}: {int(empty.sum())} of {empty.numel()} rows empty")

        make = lambda grid: torch.randn(2, 4, *grid.shape, device=self.device, requires_grad=True)
        inputs = {"q": make(grid_out), "k": make(grid_in), "v": make(grid_in)}
        inputs_ref = {key: tensor.detach().clone().requires_grad_() for key, tensor in inputs.items()}
        out = model(inputs["q"], inputs["k"], inputs["v"])
        out_ref = ref(inputs_ref["q"], inputs_ref["k"], inputs_ref["v"])
        self.assertTrue(torch.isfinite(out_ref).all(), "reference output")
        self.assertTrue(torch.isfinite(out).all(), "output")
        self.assertTrue(compare_tensors(f"{name} output", out, out_ref, atol=atol, rtol=rtol))

        grad = torch.randn_like(out)
        grads = torch.autograd.grad(out, list(inputs.values()) + list(model.parameters()), grad_outputs=grad)
        grads_ref = torch.autograd.grad(out_ref, list(inputs_ref.values()) + list(ref.parameters()), grad_outputs=grad)
        names = list(inputs.keys()) + [key for key, _ in model.named_parameters()]
        for key, got, expected in zip(names, grads, grads_ref):
            self.assertTrue(torch.isfinite(expected).all(), f"reference grad {key}")
            self.assertTrue(torch.isfinite(got).all(), f"grad {key}")
            self.assertTrue(compare_tensors(f"{name} grad {key}", got, expected, atol=atol, rtol=rtol))


@parameterized_class(("device"), _devices)
class TestBackendState(unittest.TestCase):
    """
    A layer holds exactly the state of the backend it selected, and nothing else.

    Every tensor derived from the grid and the neighbourhood belongs to a backend
    (see backends.py), so the layer's buffers are those one backend's prepare()
    returned -- the quadrature weights included, which come in two forms of which each
    backend reads one. A union of what several implementations might want is what the
    backend protocol replaced; this pins that it stays replaced.
    """

    # what each backend reads, and so all a layer using it may register
    EXPECTED = {
        "ragged-optimized": {"ring_weights", "psi_seg", "psi_seg_off", "psi_ring_base", "psi_ring_size"},
        "ragged-reference": {"point_weights", "psi_col_idx", "psi_roff_idx"},
        "regular-optimized": {"ring_weights", "psi_seg", "psi_seg_off"},
        "regular-reference": {"ring_weights", "psi_col_idx", "psi_roff_idx"},
    }

    @parameterized.expand(
        [
            ["regular", True],
            ["regular", False],
            ["ragged", True],
            ["ragged", False],
        ],
        skip_on_empty=True,
    )
    def test_buffers_are_the_backends_state(self, family, optimized_kernel):
        grid = HealpixGrid(nside=4) if family == "ragged" else as_grid("equiangular", nlat=8, nlon=16)
        model = NeighborhoodAttentionS2(grid_in=grid, grid_out=grid, in_channels=4, num_heads=2, optimized_kernel=optimized_kernel)

        def check_state():
            self.assertEqual({name for name, _ in model.named_buffers()}, self.EXPECTED[model.backend.name])
            self.assertEqual(set(model._backend_state), self.EXPECTED[model.backend.name])

        # constructed on CPU and moved, as a model normally is; .to() moves the module
        # in place and reselects the backend on the new device
        check_state()
        model.to(self.device)
        check_state()

    @parameterized.expand([["regular"], ["ragged"]], skip_on_empty=True)
    def test_optimized_backends_need_a_kernel_for_the_device(self, family):
        """
        Being built is not enough: the optimized backends serve only the device types the
        build registered kernels for. MPS never has one, and CUDA has one only in a CUDA
        build, so a layer moved there selects the reference rather than failing in the
        dispatcher on its first forward. Asked of available() directly, so it needs no
        such device to run.
        """
        grid = HealpixGrid(nside=4) if family == "ragged" else as_grid("equiangular", nlat=8, nlon=16)
        model = NeighborhoodAttentionS2(grid_in=grid, grid_out=grid, in_channels=4, num_heads=2)
        optimized, reference = (RaggedOptimizedBackend, RaggedReferenceBackend) if family == "ragged" else (RegularOptimizedBackend, RegularReferenceBackend)

        self.assertFalse(optimized.available(model, torch.device("mps")))
        self.assertEqual(optimized.available(model, torch.device("cuda")), optimized_kernels_is_available() and cuda_kernels_is_available())
        self.assertEqual(optimized.available(model, torch.device("cpu")), optimized_kernels_is_available())
        self.assertTrue(reference.available(model, torch.device("mps")))

    @parameterized.expand([["regular"], ["ragged"]], skip_on_empty=True)
    def test_falling_back_to_the_reference_warns(self, family):
        """
        Asking for the compiled kernels and getting the reference is reported, whether the
        extension is missing or has no kernel for the device; asking for the reference is
        not. A build without a kernel for this device is simulated by emptying the
        backend's device set, so the test runs on any machine.
        """
        grid = HealpixGrid(nside=4) if family == "ragged" else as_grid("equiangular", nlat=8, nlon=16)
        devices = "_RAGGED_DEVICES" if family == "ragged" else "_REGULAR_DEVICES"
        make = lambda optimized_kernel: NeighborhoodAttentionS2(grid_in=grid, grid_out=grid, in_channels=4, num_heads=2, optimized_kernel=optimized_kernel).to(self.device)

        with mock.patch.object(attention_backends, devices, frozenset()):
            with self.assertWarnsRegex(UserWarning, "falls back to the torch reference"):
                model = make(True)
            self.assertTrue(model.backend.reference)

            with warnings.catch_warnings():
                warnings.filterwarnings("error", message=".*falls back to the torch reference")
                make(False)

    @parameterized.expand([["regular"], ["ragged"]], skip_on_empty=True)
    def test_float64_takes_the_reference(self, family):
        """
        The compiled kernels compute in float32 whatever the storage type, so a float64
        layer is handed to the torch reference, the one path that computes in float64, and
        says so. Casting back to float32 returns it to the kernels: the layer's dtype, not
        only its state's, decides the backend.
        """
        grid = HealpixGrid(nside=4) if family == "ragged" else as_grid("equiangular", nlat=8, nlon=16)
        model = NeighborhoodAttentionS2(grid_in=grid, grid_out=grid, in_channels=4, num_heads=2).to(self.device)
        reference = NeighborhoodAttentionS2(grid_in=grid, grid_out=grid, in_channels=4, num_heads=2, optimized_kernel=False).to(self.device)
        reference.load_state_dict(model.state_dict())
        optimized = model.backend.name

        with self.assertWarnsRegex(UserWarning, "falls back to the torch reference.*float64"):
            model.double()
        self.assertTrue(model.backend.reference)

        x = torch.randn(2, 4, *grid.shape, device=self.device, dtype=torch.float64, requires_grad=True)
        out = model(x)
        self.assertEqual(out.dtype, torch.float64)
        self.assertTrue(torch.equal(out, reference.double()(x)))

        # computed in float64, not float32 stored as float64: against the dense oracle in
        # float64 it agrees to float64 roundoff, where float32 compute misses by ~1e-7
        mask = _brute_force_neighborhood(grid, grid, model.theta_cutoff).to(self.device)
        flat = x.detach().reshape(2, 4, -1)
        expected = _dense_masked_attention(model, flat, flat, flat, mask).reshape(out.shape)
        self.assertTrue(compare_tensors("float64 output", out, expected, atol=1e-12, rtol=1e-12))
        out.sum().backward()
        self.assertEqual(x.grad.dtype, torch.float64)

        model.float()
        self.assertEqual(model.backend.name, optimized)

    @parameterized.expand(
        [
            ["regular", True],
            ["regular", False],
            ["ragged", True],
            ["ragged", False],
        ],
        skip_on_empty=True,
    )
    def test_a_dtype_cast_keeps_the_state_float32(self, family, optimized_kernel):
        """
        .half() / .double() cast every floating buffer, but the backend fixes the dtype of
        its quadrature weights: float32, which the kernels read as float, and float64 only
        for a float64 layer, which takes the reference. A cast must not decide it, or a
        16-bit buffer reaches a kernel that reinterprets it as 32-bit.
        """
        grid = HealpixGrid(nside=4) if family == "ragged" else as_grid("equiangular", nlat=8, nlon=16)
        model = NeighborhoodAttentionS2(grid_in=grid, grid_out=grid, in_channels=4, num_heads=2, optimized_kernel=optimized_kernel).to(self.device)

        with warnings.catch_warnings():
            # .double() falls back to the reference and says so; that is tested elsewhere
            warnings.filterwarnings("ignore", message=".*falls back to the torch reference")
            for cast in (torch.nn.Module.half, torch.nn.Module.double, torch.nn.Module.float):
                cast(model)
                self.assertEqual({name for name, _ in model.named_buffers()}, TestBackendState.EXPECTED[model.backend.name])
                expected = torch.float64 if cast is torch.nn.Module.double else torch.float32
                for name in ("ring_weights", "point_weights"):
                    if hasattr(model, name):
                        self.assertEqual(getattr(model, name).dtype, expected, f"{name} after {cast.__name__}")

    def test_a_half_model_computes_what_the_float_model_does(self):
        """
        The numerical side of the cast: a .half() layer on the regular CUDA path agrees with
        its float32 original. Before the state was restored after a cast, the regular CUDA
        kernels read the fp16 weights as float and returned garbage without an error.
        """
        if self.device.type != "cuda" or not cuda_kernels_is_available():
            raise unittest.SkipTest("the regular CUDA kernels are what read the weights as float")

        grid = as_grid("equiangular", nlat=16, nlon=32)
        model = NeighborhoodAttentionS2(grid_in=grid, grid_out=grid, in_channels=8, num_heads=2).to(self.device)
        x = torch.randn(2, 8, *grid.shape, device=self.device)

        with torch.no_grad():
            expected = model(x)
            model.half()
            got = model(x.half()).float()

        self.assertTrue(compare_tensors("half vs float", got, expected, atol=1e-2, rtol=1e-2))


class TestPsiArcStructure(unittest.TestCase):
    """psi's sparsity is a union of contiguous longitude arcs, one per (row, input lat).

    This is a geometric consequence of the neighborhood being a geodesic ball: a ball
    intersects a latitude circle in a single arc, never in disjoint pieces. It is not
    an accident of one grid or cutoff, and the kernels are entitled to rely on it.

    Why pin it down: it means the neighbor longitudes for a row can be described by
    (start, width) instead of an explicit column list, so a kernel can compute a
    neighbor's address arithmetically -- wip = (lo + j + pscale*wo) % nlon_in -- rather
    than loading it from psi_col_idx. That removes a dependent load from the inner loop
    and makes the k/v accesses stride-1 in longitude. If a future change to how psi is
    built ever introduces holes, every such kernel silently reads the wrong elements,
    and this test is what catches it.

    Device-independent: psi is built on CPU at construction time.
    """

    @staticmethod
    def _is_single_arc(wi, nlon):
        """True if the longitudes form one contiguous arc on the circle (wrap allowed)."""

        w = torch.unique(wi).sort().values
        if w.numel() <= 1:
            return True
        # A single arc has exactly one gap larger than 1 -- the complement. Count the
        # seam between last and first as a gap too, which is what makes a wrapping arc
        # (e.g. {718, 719, 0, 1}) register as contiguous rather than as two pieces.
        interior = int((torch.diff(w) > 1).sum())
        seam = 1 if (w[0] + nlon) - w[-1] > 1 else 0
        return interior + seam <= 1

    @parameterized.expand(
        [
            # name, in_shape, out_shape, grid_in, grid_out, theta_cutoff
            ["self_equiangular", (64, 128), (64, 128), "equiangular", "equiangular", 0.05],
            ["self_legendre_gauss", (64, 128), (64, 128), "legendre-gauss", "legendre-gauss", 0.05],
            ["self_lobatto", (64, 128), (64, 128), "lobatto", "lobatto", 0.05],
            ["self_wide_cutoff", (64, 128), (64, 128), "equiangular", "equiangular", 0.25],
            ["downsample", (64, 128), (32, 64), "equiangular", "equiangular", 0.05],
            ["upsample", (32, 64), (64, 128), "equiangular", "equiangular", 0.05],
            ["odd_lat_down", (65, 128), (33, 64), "equiangular", "equiangular", 0.05],
            ["odd_lat_up", (33, 64), (65, 128), "equiangular", "equiangular", 0.05],
            ["lon_only_down", (64, 128), (64, 64), "equiangular", "equiangular", 0.05],
        ]
    )
    def test_rows_are_contiguous_arcs(self, name, in_shape, out_shape, grid_in, grid_out, theta_cutoff):
        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        att = NeighborhoodAttentionS2(
            grid_in=as_grid(grid_in, nlat=in_shape[0], nlon=in_shape[1]),
            grid_out=as_grid(grid_out, nlat=out_shape[0], nlon=out_shape[1]),
            in_channels=8,
            num_heads=1,
            bias=False,
            theta_cutoff=theta_cutoff,
            optimized_kernel=False,
        )

        col = att.psi_col_idx.cpu()
        roff = att.psi_roff_idx.cpu()

        # The scatter (upsample) psi is keyed by input latitude and its columns index
        # the output grid; the gather psi is the other way round.
        upsample = (nlat_out > nlat_in) or (nlon_out > nlon_in)
        nlon_decode = nlon_out if upsample else nlon_in

        nrows = roff.numel() - 1
        checked = 0
        for row in range(nrows):
            beg, end = int(roff[row]), int(roff[row + 1])
            if end <= beg:
                continue
            cols = col[beg:end]
            lat = cols // nlon_decode
            lon = cols - lat * nlon_decode
            for h in torch.unique(lat):
                checked += 1
                self.assertTrue(
                    self._is_single_arc(lon[lat == h], nlon_decode),
                    msg=f"{name}: row {row}, input lat {int(h)} has non-contiguous longitudes; " f"kernels that derive addresses arithmetically would read the wrong cells",
                )

        self.assertGreater(checked, 0, msg=f"{name}: psi was empty, nothing verified")

    @parameterized.expand(
        [
            ["self_equiangular", (64, 128), (64, 128), "equiangular", "equiangular", 0.05],
            ["self_legendre_gauss", (64, 128), (64, 128), "legendre-gauss", "legendre-gauss", 0.05],
            ["self_lobatto", (64, 128), (64, 128), "lobatto", "lobatto", 0.05],
            ["self_wide_cutoff", (64, 128), (64, 128), "equiangular", "equiangular", 0.25],
            ["downsample", (64, 128), (32, 64), "equiangular", "equiangular", 0.05],
            ["upsample", (32, 64), (64, 128), "equiangular", "equiangular", 0.05],
            ["odd_lat_down", (65, 128), (33, 64), "equiangular", "equiangular", 0.05],
            ["odd_lat_up", (33, 64), (65, 128), "equiangular", "equiangular", 0.05],
            ["lon_only_down", (64, 128), (64, 64), "equiangular", "equiangular", 0.05],
        ]
    )
    def test_segments_reproduce_col_idx(self, name, in_shape, out_shape, grid_in, grid_out, theta_cutoff):
        """(hi, lo, len) segments expand back to exactly psi_col_idx.

        Stronger than the arc property above and closer to what a kernel relies on: it
        is not enough that the arcs are contiguous, the segment table has to describe
        the *same* sparsity. A segment whose start is off by one -- which is what a
        mishandled wrapping arc produces -- still passes the contiguity check while
        silently attending to the wrong cells.
        """

        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        att = NeighborhoodAttentionS2(
            grid_in=as_grid(grid_in, nlat=in_shape[0], nlon=in_shape[1]),
            grid_out=as_grid(grid_out, nlat=out_shape[0], nlon=out_shape[1]),
            in_channels=8,
            num_heads=1,
            bias=False,
            theta_cutoff=theta_cutoff,
            optimized_kernel=False,
        )

        upsample = (nlat_out > nlat_in) or (nlon_out > nlon_in)
        nlon_decode = nlon_out if upsample else nlon_in

        col = att.psi_col_idx.cpu()
        roff = att.psi_roff_idx.cpu()
        seg, seg_off = build_psi_segments(col, roff, nlon_decode)
        expanded = expand_psi_segments(seg, seg_off, nlon_decode)

        self.assertEqual(seg_off.numel() - 1, roff.numel() - 1, msg=f"{name}: row count mismatch")
        for row in range(roff.numel() - 1):
            want = sorted(col[int(roff[row]) : int(roff[row + 1])].tolist())
            self.assertEqual(
                want,
                expanded[row],
                msg=f"{name}: row {row} segments do not reproduce psi_col_idx",
            )


# ---------------------------------------------------------------------------
# Ragged grids (HEALPix and other non-product grids)
#
# Same layer, same contract, a different neighbourhood representation underneath.
# These live here rather than in a file of their own so that the two grid families
# are read and maintained side by side: when the regular suite grows a case, the
# question of whether the ragged path needs the same one is in front of whoever is
# adding it.
# ---------------------------------------------------------------------------


def _dense_masked_attention(model, query, key, value, mask):
    r"""
    The oracle: the mathematics of :class:`NeighborhoodAttentionS2` and of
    :class:`AttentionS2` as a dense masked softmax, using ``model``'s parameters but none
    of its machinery -- channels-first projections, an explicit softmax, no layout
    conversion and no kernel. With an all-``True`` mask it is global attention.

    Computes, for each head,

    .. math::
        y_p = \sum_{j \in D(p)} \frac{e^{s\, q_p \cdot k_j} w_j}{\sum_{j' \in D(p)} e^{s\, q_p \cdot k_{j'}} w_{j'}} v_j

    with the neighbourhood :math:`D(p)` and the quadrature weights :math:`w_j` both
    folded into a single additive pre-softmax mask -- the weights as
    :math:`\log w_j`, which the softmax exponential turns back into factors, and the
    neighbourhood as :math:`-\infty` outside the disk.

    Note the weights are kept in the formula even though HEALPix is equal-area and
    they therefore cancel: the point is to check the module against the general
    expression, not against a simplification that happens to hold on this grid.

    Parameters
    ----------
    model : NeighborhoodAttentionS2 or AttentionS2
        Supplies the projection weights, biases, head count and scale.
    query : torch.Tensor
        ``(batch, in_channels, npoints_out)``.
    key, value : torch.Tensor
        ``(batch, in_channels, npoints_in)``.
    mask : torch.Tensor
        Boolean ``(npoints_out, npoints_in)``; ``True`` where the input point is in
        the output point's neighbourhood.

    Returns
    -------
    torch.Tensor
        ``(batch, out_channels, npoints_out)``.
    """
    heads = model.num_heads

    def project(signal, weights, bias):
        # the stored weights are (C_out, C_in, 1, 1) convolution kernels
        return F.conv1d(signal, weights.reshape(*weights.shape[:2], 1), bias=bias)

    q = project(query, model.q_weights, model.q_bias)
    k = project(key, model.k_weights, model.k_bias)
    v = project(value, model.v_weights, model.v_bias)

    # (batch, channels, npoints) -> (batch, heads, npoints, channels per head)
    def split_heads(signal):
        batch, channels, npoints = signal.shape
        return signal.reshape(batch, heads, channels // heads, npoints).transpose(-1, -2)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)

    if model.q_norm_weights is not None:
        q = F.rms_norm(q, normalized_shape=model.q_norm_weights.shape, weight=1 + model.q_norm_weights)
    if model.k_norm_weights is not None:
        k = F.rms_norm(k, normalized_shape=model.k_norm_weights.shape, weight=1 + model.k_norm_weights)

    logits = model.scale * (q @ k.transpose(-1, -2))

    # computed in the logits' dtype, not rounded through float32, so a float64 oracle is float64 throughout
    log_weights = torch.log(_point_weights(model, logits.device, logits.dtype))
    logits = logits + log_weights.reshape(1, 1, 1, -1)
    logits = logits.masked_fill(~mask.reshape(1, 1, *mask.shape), float("-inf"))

    out = torch.softmax(logits, dim=-1) @ v

    # (batch, heads, npoints, channels per head) -> (batch, out_channels, npoints)
    batch, _, npoints, _ = out.shape
    out = out.transpose(-1, -2).reshape(batch, model.out_channels, npoints)

    return project(out, model.proj_weights, model.proj_bias)


@dataclass(frozen=True, eq=False)
class _ProductGridAsRagged(GridS2):
    """
    An equiangular grid presented through the ragged interface.

    Every ring carries the same number of longitudes, so this *is* a product grid --
    it simply declines to say so, which routes it down the ragged path. That makes the
    two implementations comparable on identical geometry: any difference between them
    is the implementation, since the points, the weights and the neighbourhood are the
    same tensors either way.

    It is a test fixture rather than a library grid because nothing in the library
    would want it: a real product grid should be a RegularGridS2 and take the faster
    path. Its whole purpose is to be the control in that comparison.
    """

    nlat: int
    nlon: int
    grid_type: ClassVar[str] = "test-product-as-ragged"

    @property
    def is_regular(self):
        # Deliberately false. GridS2 computes this from the geometry -- every ring the
        # same length means regular -- so a uniform grid cannot be ragged by accident,
        # and saying so here is the only way to route identical geometry down the other
        # path. That is the whole point of the fixture: the lie is the experiment.
        return False

    @property
    def nrings(self):
        return self.nlat

    @property
    def nlon_per_lat(self):
        return torch.full((self.nlat,), self.nlon, dtype=torch.int64)

    @property
    def colats(self):
        return as_grid("equiangular", nlat=self.nlat, nlon=self.nlon).colats

    @property
    def colat_weights(self):
        return as_grid("equiangular", nlat=self.nlat, nlon=self.nlon).colat_weights

    def lons(self, ilat=None):
        return torch.arange(self.nlon, dtype=torch.float64) * (2.0 * math.pi / self.nlon)


# Defining a GridS2 subclass with a grid_type registers it, and the registry is global:
# left in place this fixture would be swept up by every test elsewhere that parameterizes
# over grid_types() and constructs with nlat/nlon. It is only ever built directly, by name
# it has no business being discoverable, so it is withdrawn immediately -- the class object
# keeps working, only as_grid("test-product-as-ragged") stops resolving.
_GRID_REGISTRY.pop(_ProductGridAsRagged.grid_type, None)


@parameterized_class(("device"), _devices)
class TestNeighborhoodAttentionRaggedS2(unittest.TestCase):
    """NeighborhoodAttentionS2 on HEALPix, against a dense masked-softmax oracle."""

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _build(self, nside_in, nside_out, channels, out_channels, heads, use_qknorm, bias, theta_cutoff=None):
        grid_in = HealpixGrid(nside=nside_in)
        grid_out = HealpixGrid(nside=nside_out)
        model = NeighborhoodAttentionS2(
            grid_in=grid_in,
            grid_out=grid_out,
            in_channels=channels,
            out_channels=out_channels,
            num_heads=heads,
            use_qknorm=use_qknorm,
            bias=bias,
            theta_cutoff=theta_cutoff,
        ).to(self.device)
        return grid_in, grid_out, model

    def _csr(self, model):
        """
        The neighbourhood as a column list, built rather than read off the layer.

        A layer holds whatever its backend needs, and the compiled ragged kernels take
        the arc form: the columns are several MB they never look at. So a test that
        wants the neighbourhood as columns -- to compare against a mask, or to drive the
        reference -- has to ask the precompute for it, not the module. It is the same
        pattern either way, and the precompute is cached, so this costs nothing.
        """
        col_idx, roff_idx = precompute_neighborhood_csr_s2(model.grid_in, model.grid_out, model.theta_cutoff)
        return col_idx.to(self.device), roff_idx.to(self.device)

    def _inputs(self, batch, channels, npoints_in, npoints_out):
        make = lambda npoints: torch.randn(batch, channels, npoints, device=self.device, dtype=torch.float32, requires_grad=True)
        return {"q": make(npoints_out), "k": make(npoints_in), "v": make(npoints_in)}

    @parameterized.expand([[torch.float16], [torch.bfloat16]], skip_on_empty=True)
    def test_autocast_normalizes_mixed_input_dtypes(self, autocast_dtype):
        """
        Autocast must reconcile k/v/q dtypes before they reach the ragged kernel, on CPU as on CUDA.

        The ragged counterpart of the regular test of the same name: the kernels check that k, v
        and q share a dtype, autocast alone does not guarantee it, and the Autocast{CUDA,CPU}
        registrations are what make it hold. The CPU key was once missing because there was no
        CPU ragged kernel; there is now, and the backend selects it on CPU. Without autocast the
        same inputs raise "must match q dtype".
        """
        grid, _, model = self._build(2, 2, 4, 4, 1, use_qknorm=False, bias=False)
        if model.backend.name != "ragged-optimized":
            raise unittest.SkipTest(f"needs the compiled ragged kernels, got {model.backend.name}")

        # channels-last (batch, npoints, channels), as the op expects. q is left fp32 while
        # k/v are reduced precision.
        kw = torch.randn(2, grid.npoints, 4, device=self.device, dtype=autocast_dtype)
        vw = torch.randn(2, grid.npoints, 4, device=self.device, dtype=autocast_dtype)
        qw = torch.randn(2, grid.npoints, 4, device=self.device, dtype=torch.float32)
        args = (model.ring_weights, model.psi_seg, model.psi_seg_off, model.psi_ring_base, model.psi_ring_size, 1, grid.npoints)

        with torch.autocast(self.device.type, dtype=autocast_dtype):
            out = torch.ops.attention_kernels._neighborhood_s2_attention_ragged_optimized(kw, vw, qw, *args)[0]

        self.assertEqual(out.dtype, autocast_dtype, f"autocast output dtype {out.dtype} != {autocast_dtype}")

        # the negative control, made explicit: the same inputs without autocast are rejected
        with self.assertRaises(RuntimeError):
            torch.ops.attention_kernels._neighborhood_s2_attention_ragged_optimized(kw, vw, qw, *args)

    @parameterized.expand(
        [
            # Format: [nside_in, nside_out, batch, channels, out_channels, heads, use_qknorm, bias]
            [4, 4, 2, 4, 4, 1, False, True],
            [4, 4, 2, 4, 4, 2, False, True],
            [4, 4, 2, 4, 8, 4, False, True],
            [4, 4, 2, 8, 4, 4, False, True],
            [4, 4, 2, 4, 4, 2, True, True],
            [4, 4, 2, 4, 4, 1, False, False],
            [8, 8, 1, 4, 4, 2, False, True],
            # resampling in both directions. The ragged path has no p-shift, so unlike
            # the regular one it does not need the point counts to divide each other
            [8, 4, 1, 4, 4, 2, False, True],
            [4, 8, 1, 4, 4, 2, False, True],
            [2, 4, 2, 4, 4, 1, False, True],
            # Channel counts that reach the register-blocked CUDA kernel.
            #
            # Every case above has at most 32 channels per head, so NLOC, which is
            # DIV_UP(channels per head, 32), is 1 for all of them. At NLOC == 1 the
            # kernel's unrolled loops run from 0 to NLOC-1 == 0, i.e. not at all, and
            # only the bounds-guarded tail iteration executes. So none of those cases
            # touch the unguarded register indexing that the whole kernel rests on --
            # it is sound only because i <= NLOC-2 implies i*32 + tidx < nchan, and an
            # off-by-one there reads a neighbouring channel block and still returns a
            # smooth, plausible field.
            #
            # 96 per head is an exact multiple of the warp, so it never exercises the
            # guard on the final register; 80 and 40
            # leave that one partial (only 16 and 8 lanes of the last register are in
            # range) and are here for the boundary. 192 over 2 heads puts NLOC > 1 and
            # a head offset together, since the head stride is what the unrolled
            # indexing is added to.
            [4, 4, 1, 96, 96, 1, False, True],  # NLOC 3, exact
            [4, 4, 1, 80, 80, 1, False, True],  # NLOC 3, last register partial
            [4, 4, 1, 40, 40, 1, False, True],  # NLOC 2, last register partial
            [4, 4, 1, 192, 192, 2, False, True],  # NLOC 3, two heads
            # Unequal counts at a large channel width, which the register-blocked
            # kernel refuses (it needs one NLOC to serve both), so this pins the
            # generic fallback on the sizes that now bypass it.
            [4, 4, 1, 96, 64, 1, False, True],
        ],
        skip_on_empty=True,
    )
    def test_it_matches_a_dense_masked_softmax(self, nside_in, nside_out, batch, channels, out_channels, heads, use_qknorm, bias, atol=1e-5, rtol=1e-3):
        """Forward and every gradient, against the dense oracle."""
        grid_in, grid_out, model = self._build(nside_in, nside_out, channels, out_channels, heads, use_qknorm, bias)

        mask = _brute_force_neighborhood(grid_in, grid_out, model.theta_cutoff).to(self.device)

        inputs = self._inputs(batch, channels, grid_in.npoints, grid_out.npoints)
        inputs_ref = {name: tensor.detach().clone().requires_grad_() for name, tensor in inputs.items()}

        out = model(inputs["q"], inputs["k"], inputs["v"])
        out_ref = _dense_masked_attention(model, inputs_ref["q"], inputs_ref["k"], inputs_ref["v"], mask)

        self.assertTrue(compare_tensors("output", out, out_ref, atol=atol, rtol=rtol))

        # one backward per graph, from the same upstream gradient
        grad = torch.randn_like(out)
        grads_ref = torch.autograd.grad(out_ref, list(inputs_ref.values()) + list(model.parameters()), grad_outputs=grad, retain_graph=True)
        grads = torch.autograd.grad(out, list(inputs.values()) + list(model.parameters()), grad_outputs=grad)

        names = list(inputs.keys()) + [name for name, _ in model.named_parameters()]
        for name, got, expected in zip(names, grads, grads_ref):
            self.assertTrue(compare_tensors(f"grad {name}", got, expected, atol=atol, rtol=rtol))

    @parameterized.expand([[2], [4]], skip_on_empty=True)
    def test_a_global_cutoff_attends_to_the_whole_sphere(self, nside, atol=1e-5, rtol=1e-3):
        r"""
        With :math:`\theta_\mathrm{cutoff} = 2\pi` every input point must be in every
        neighbourhood, and the layer must then agree with an *unmasked* dense softmax.

        The degenerate end of the cutoff range, where the local operator becomes the
        global one. It is worth its own case because it is the one radius at which
        every arc wraps a full ring, so an off-by-one in the arc encoding that the
        interior cases tolerate has nowhere to hide.
        """
        grid = HealpixGrid(nside=nside)
        channels, batch = 4, 2

        model = NeighborhoodAttentionS2(grid_in=grid, grid_out=grid, in_channels=channels, num_heads=2, bias=False, theta_cutoff=2 * math.pi).to(self.device)

        # the pattern is complete: every output point sees every input point, exactly once
        col_idx, _ = self._csr(model)
        self.assertEqual(col_idx.numel(), grid.npoints * grid.npoints)

        inputs = self._inputs(batch, channels, grid.npoints, grid.npoints)
        inputs_ref = {name: tensor.detach().clone().requires_grad_() for name, tensor in inputs.items()}

        unmasked = torch.ones(grid.npoints, grid.npoints, dtype=torch.bool, device=self.device)

        out = model(inputs["q"], inputs["k"], inputs["v"])
        out_ref = _dense_masked_attention(model, inputs_ref["q"], inputs_ref["k"], inputs_ref["v"], unmasked)

        self.assertTrue(compare_tensors("output", out, out_ref, atol=atol, rtol=rtol))

    def test_the_quadrature_weights_integrate_to_the_sphere(self):
        """A constant field must integrate to the sphere's area, at every resolution."""
        for nside in (1, 2, 4, 8):
            with self.subTest(nside=nside):
                grid = HealpixGrid(nside=nside)
                model = NeighborhoodAttentionS2(grid_in=grid, grid_out=grid, in_channels=1).to(self.device)
                self.assertAlmostEqual(float(_point_weights(model, self.device).sum()), 4.0 * math.pi, places=4)

    def test_a_constant_field_is_reproduced(self):
        """
        Attention over a constant field returns that constant, whatever the weights.

        The normalized attention weights of each output point sum to one by
        construction, so this is really a check that the neighbour list, the weight
        gather and the normalization all agree about which points a row contains --
        a row that gathered a weight it did not gather a value for would break it.
        """
        grid_in, grid_out, model = self._build(4, 4, 4, 4, 1, False, False)

        # with the projections set to the identity and no bias, attention over a
        # constant input is the only thing left that could change the value
        with torch.no_grad():
            for weights in (model.q_weights, model.k_weights, model.v_weights, model.proj_weights):
                weights.zero_()
                weights[:, :, 0, 0] = torch.eye(weights.shape[0], weights.shape[1])

        constant = torch.full((2, 4, grid_in.npoints), 0.375, device=self.device)
        out = model(constant)

        self.assertTrue(compare_tensors("output", out, torch.full_like(out, 0.375), atol=1e-5, rtol=1e-5))

    def test_the_op_satisfies_its_schema(self):
        """
        opcheck: schema, fake tensor and autograd registration of the ragged op.

        ``test_aot_dispatch_dynamic`` is deliberately excluded. It traces the backward,
        which -- like the regular reference's backward -- walks the neighbour list in
        Python and so reads ``row_off`` with ``int()``. Under AOT autograd those reads
        are guards on unbacked symints and raise, because the values of an index tensor
        are not known at trace time. The forward does not hit this only because a
        ``custom_op`` is opaque to tracing; a backward registered through
        ``register_autograd`` is not.

        This is a property of every neighbour-list-walking reference in the library,
        which is why the optimized ops are the ones checked under AOT dispatch and the
        references are not (see the opcheck calls in test_attention.py). The remaining
        three utilities are the ones that mean something for an op whose whole purpose
        is to be an opaque, readable specification.
        """
        grid = HealpixGrid(nside=2)
        model = NeighborhoodAttentionS2(grid_in=grid, grid_out=grid, in_channels=4, num_heads=2).to(self.device)

        batch, npoints = 2, grid.npoints
        make = lambda channels: torch.randn(batch, npoints, channels, device=self.device, dtype=torch.float32, requires_grad=True)
        args = (
            make(model.k_channels),
            make(model.out_channels),
            make(model.k_channels),
            _point_weights(model, self.device),
            *self._csr(model),
            model.num_heads,
            npoints,
        )
        opcheck(
            torch.ops.attention_kernels._neighborhood_s2_attention_ragged_torch,
            args,
            test_utils=("test_schema", "test_autograd_registration", "test_faketensor"),
        )

    def test_a_mixed_pair_takes_the_ragged_path_and_keeps_each_layout(self):
        """
        A mixed pair is computed by the ragged path, because keying the neighbourhood
        by output point is the general choice and a regular grid admits it too. What
        each side must not lose is its own layout: the ragged side stays flat and the
        regular side keeps its two spatial axes.
        """
        hpx, eqa = HealpixGrid(nside=2), as_grid("equiangular", nlat=6, nlon=12)

        decode = NeighborhoodAttentionS2(grid_in=hpx, grid_out=eqa, in_channels=4)
        self.assertTrue(decode.ragged)
        self.assertTrue(decode.ragged_in)
        self.assertFalse(decode.ragged_out)

        encode = NeighborhoodAttentionS2(grid_in=eqa, grid_out=hpx, in_channels=4)
        self.assertTrue(encode.ragged)
        self.assertFalse(encode.ragged_in)
        self.assertTrue(encode.ragged_out)

    def test_it_rejects_inputs_of_the_wrong_rank_or_extent(self):
        """A ragged field is flat, so the module wants 3 dims and the right point count."""
        grid = HealpixGrid(nside=2)
        model = NeighborhoodAttentionS2(grid_in=grid, grid_out=grid, in_channels=4).to(self.device)

        # the rectangular shape a caller might reach for, (rings, widest ring), which is
        # not how a ragged field is laid out and does not even hold npoints values
        with self.assertRaises(RuntimeError):
            model(torch.randn(2, 4, grid.nrings, 4 * grid.nside, device=self.device))

        with self.assertRaises(RuntimeError):
            model(torch.randn(2, 4, grid.npoints + 1, device=self.device))

    def test_every_output_point_attends_to_itself(self):
        """
        Self-attention on the same grid must include the diagonal, or an output point
        would be built without reference to its own value.
        """
        grid = HealpixGrid(nside=4)
        model = NeighborhoodAttentionS2(grid_in=grid, grid_out=grid, in_channels=1)

        col_idx, row_off = self._csr(model)
        for ipoint in range(grid.npoints):
            neighbors = col_idx[row_off[ipoint] : row_off[ipoint + 1]]
            self.assertIn(ipoint, neighbors.tolist(), f"output point {ipoint} does not attend to itself")

    def test_the_neighbourhood_matches_the_brute_force_reference(self):
        """
        The module's expanded neighbour list is the brute-force one.

        The dense test above builds its mask from the reference, so on its own it
        would not notice the module and the reference disagreeing about the
        neighbourhood in a way that happened to be self-consistent. This compares the
        two directly.
        """
        for nside_in, nside_out in ((4, 4), (8, 4), (4, 8)):
            with self.subTest(nside_in=nside_in, nside_out=nside_out):
                grid_in, grid_out, model = self._build(nside_in, nside_out, 1, 1, 1, False, False)
                expected = _brute_force_neighborhood(grid_in, grid_out, model.theta_cutoff)
                col_idx, roff_idx = self._csr(model)

                got = torch.zeros_like(expected)
                for ipoint in range(grid_out.npoints):
                    got[ipoint, col_idx[roff_idx[ipoint] : roff_idx[ipoint + 1]]] = True

                self.assertTrue(torch.equal(got, expected))

    @parameterized.expand(
        [
            # Format: [in_shape, out_shape, batch, channels, heads, use_qknorm, dtype, atol, rtol]
            #
            # Precisions match the regular cases above: float32 throughout, with fp16/bf16
            # through autocast. float64 is absent because a float64 layer never reaches the
            # kernels: it takes the torch reference (test_float64_takes_the_reference).
            [(6, 12), (6, 12), 2, 4, 1, False, torch.float32, 1e-5, 1e-3],
            [(6, 12), (6, 12), 2, 4, 2, False, torch.float32, 1e-5, 1e-3],
            [(6, 12), (6, 12), 2, 8, 4, True, torch.float32, 1e-5, 1e-3],
            [(12, 24), (12, 24), 1, 4, 2, False, torch.float32, 1e-5, 1e-3],
            # odd ring counts, where the poles are a single ring rather than a pair
            [(7, 12), (7, 12), 1, 4, 1, False, torch.float32, 1e-5, 1e-3],
            # nlon not twice nlat, so the neighbourhood is wider in longitude than latitude
            [(8, 8), (8, 8), 1, 4, 1, False, torch.float32, 1e-5, 1e-3],
            # Resampling, which is where the two paths stop resembling each other. The
            # regular side keys psi by output latitude and recovers the other longitudes
            # by the integer p-shift, swapping the grids entirely when upsampling; the
            # ragged side has no p-shift and always runs input to output. So these rows
            # compare two genuinely different constructions of the same operator, which
            # the equal-shape rows above (pscale == 1) do not.
            [(12, 24), (6, 12), 1, 4, 2, False, torch.float32, 1e-5, 1e-3],  # pscale 2
            [(12, 24), (6, 8), 1, 4, 1, False, torch.float32, 1e-5, 1e-3],  # pscale 3
            [(12, 12), (6, 12), 1, 4, 1, False, torch.float32, 1e-5, 1e-3],  # lat only, pscale 1
            # lat-only upsample: the regular path takes the gather direction (nlon_in divides
            # nlon_out), the ragged path the scatter one; the cutoff must still agree
            [(6, 12), (12, 12), 1, 4, 1, False, torch.float32, 1e-5, 1e-3],
            [(6, 12), (12, 24), 1, 4, 2, False, torch.float32, 1e-5, 1e-3],  # upsample
            [(6, 8), (12, 24), 1, 4, 1, False, torch.float32, 1e-5, 1e-3],  # upsample, pscale 3
            [(12, 24), (6, 12), 1, 8, 2, True, torch.float32, 1e-5, 1e-3],  # resampling with qk-norm
            # reduced precision, where the two paths accumulate differently
            [(6, 12), (6, 12), 2, 4, 1, False, torch.float16, 2e-2, 1e-2],
            [(6, 12), (6, 12), 2, 4, 1, False, torch.bfloat16, 5e-2, 5e-2],
            [(12, 24), (6, 12), 1, 4, 2, False, torch.float16, 2e-2, 1e-2],
        ],
        skip_on_empty=True,
    )
    def test_it_matches_the_regular_path_on_a_product_grid(self, in_shape, out_shape, batch, channels, heads, use_qknorm, dtype, atol, rtol):
        """
        The ragged path against the regular one, on geometry where both are defined.

        A product grid is the special case of a ragged grid in which every ring happens
        to carry the same number of longitudes, so the two must agree on it exactly.
        That makes the established regular path an oracle for the ragged one over the
        parts no HEALPix case can reach on its own -- it is the same softmax, the same
        quadrature and the same gradient algebra, checked without writing a second
        implementation of any of them.

        The two layers are given identical parameters, so any difference is the path and
        not the initialization. The ragged side sees the field flattened to one point
        axis, which is what its grid descriptor reports as the shape.

        Both sides are also required to agree on the cutoff, which is what defines the
        operator: a resampling case that disagreed there would be comparing two different
        operators and passing or failing for the wrong reason. They need not agree on the
        direction. The regular path reads it off the ratio of longitude counts and the
        ragged path off the point counts, so a latitude-only upsample is computed as a
        gather on one and a scatter on the other -- two organizations of the same sum, which
        is exactly what the output comparison below then checks.
        """
        set_seed(333)
        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape
        regular_in = as_grid("equiangular", nlat=nlat_in, nlon=nlon_in)
        regular_out = as_grid("equiangular", nlat=nlat_out, nlon=nlon_out)
        ragged_in = _ProductGridAsRagged(nlat=nlat_in, nlon=nlon_in)
        ragged_out = _ProductGridAsRagged(nlat=nlat_out, nlon=nlon_out)

        kwargs = dict(in_channels=channels, out_channels=channels, num_heads=heads, use_qknorm=use_qknorm)
        regular = NeighborhoodAttentionS2(grid_in=regular_in, grid_out=regular_out, **kwargs).to(self.device)
        ragged = NeighborhoodAttentionS2(grid_in=ragged_in, grid_out=ragged_out, **kwargs).to(self.device)

        self.assertFalse(regular.ragged, "the control took the ragged path")
        self.assertTrue(ragged.ragged, "the stand-in did not take the ragged path")
        self.assertAlmostEqual(regular.theta_cutoff, ragged.theta_cutoff, places=12, msg="the two paths chose different cutoffs")

        # identical weights, so the comparison is of the paths and nothing else
        ragged.load_state_dict(regular.state_dict())

        # Modules and inputs stay fp32; reduced precision is exercised through autocast,
        # which is what the ops' autocast registrations are for and how the regular cases
        # above do it. Casting either side by hand would test a configuration the layer
        # does not support -- the projections require matching dtypes.
        q = torch.randn(batch, channels, nlat_out, nlon_out, device=self.device, dtype=torch.float32, requires_grad=True)
        x = torch.randn(batch, channels, nlat_in, nlon_in, device=self.device, dtype=torch.float32, requires_grad=True)
        q_flat = q.detach().flatten(-2, -1).clone().requires_grad_(True)
        x_flat = x.detach().flatten(-2, -1).clone().requires_grad_(True)

        with maybe_autocast(self.device.type, dtype):
            out_reg = regular(q, x, x)
            out_rag = ragged(q_flat, x_flat, x_flat)

        self.assertTrue(
            compare_tensors("forward", out_rag.float(), out_reg.flatten(-2, -1).float(), atol=atol, rtol=rtol),
            "ragged and regular forwards differ on a product grid",
        )

        grad = torch.randn_like(out_reg)
        out_reg.backward(grad)
        out_rag.backward(grad.flatten(-2, -1))

        for name, got, expected in (("query", q_flat.grad, q.grad), ("key/value", x_flat.grad, x.grad)):
            with self.subTest(gradient=name):
                self.assertTrue(compare_tensors(f"{name} grad", got.float(), expected.flatten(-2, -1).float(), atol=atol, rtol=rtol))
        for (name, p_reg), (_, p_rag) in zip(regular.named_parameters(), ragged.named_parameters()):
            with self.subTest(parameter=name):
                self.assertTrue(compare_tensors(f"grad {name}", p_rag.grad.float(), p_reg.grad.float(), atol=atol, rtol=rtol))

    @parameterized.expand(
        [
            # Format: [nside, batch, channels, heads]
            [2, 1, 8, 1],
            [4, 2, 32, 4],
        ],
        skip_on_empty=True,
    )
    @unittest.skipUnless(
        optimized_kernels_is_available(),
        "skipping test because the ragged kernels are not available",
    )
    def test_optimized_pt2_compatibility(self, nside, batch, channels, heads):
        """
        Schema, fake and autograd registration of the ragged optimized op.

        Both directions in one test, because a backward's fake is only read when a
        backward is traced: ``test_aot_dispatch_dynamic`` is what exercises it, and it
        is the utility that caught ``backward_ragged`` being traced with one argument
        more than its fake accepted. The forward-only utilities would not have.
        """
        set_seed(333)
        grid = HealpixGrid(nside=nside)
        layer = NeighborhoodAttentionS2(in_channels=channels, num_heads=heads, grid_in=grid, grid_out=grid).to(self.device)

        npoints = grid.npoints
        make = lambda: torch.randn(batch, npoints, channels, device=self.device, dtype=torch.float32, requires_grad=True)
        args = (
            make(),
            make(),
            make(),
            layer.ring_weights,
            layer.psi_seg,
            layer.psi_seg_off,
            layer.psi_ring_base,
            layer.psi_ring_size,
            heads,
            npoints,
        )
        opcheck(torch.ops.attention_kernels._neighborhood_s2_attention_ragged_optimized, args)

        # The raw kernels under the custom op, whose fakes checking the wrapper never runs;
        # see test_optimized_pt2_compatibility of the regular layer. Both dtypes, because
        # y_hi is the fp32 output for bfloat16 and an empty placeholder otherwise, and the
        # fakes branch on that.
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                kx, vx, qy = (t.detach().to(dtype) for t in args[:3])
                fwd_args = (kx, vx, qy, *args[3:])
                opcheck(torch.ops.attention_kernels.forward_ragged, fwd_args)
                y, y_hi, alpha_sum, qdotk_max = torch.ops.attention_kernels.forward_ragged(*fwd_args)
                bwd_args = (kx, vx, qy, torch.randn_like(y), y, y_hi, alpha_sum, qdotk_max, *args[3:])
                opcheck(torch.ops.attention_kernels.backward_ragged, bwd_args)


@parameterized_class(("device"), _devices)
class TestAttentionS2RingGrids(unittest.TestCase):
    """
    Global attention on ragged grids and on mixed pairs, against the dense reference.

    AttentionS2 reads nothing of a grid but its points and weights, so a HEALPix side
    differs from a regular one only in layout: a single flat spatial axis instead of
    (nlat, nlon). The dense reference is addressed in flat indices throughout and shares
    none of the layer's projection or layout code, so it pins both the mathematics and
    the ordering a regular side is flattened in.
    """

    def setUp(self):
        disable_tf32()
        set_seed(333)

    @parameterized.expand(
        [
            # Format: [name, grid_in, grid_out, heads, use_qknorm]
            ["healpix", HealpixGrid(nside=4), HealpixGrid(nside=4), 1, False],
            ["healpix_heads_qknorm", HealpixGrid(nside=4), HealpixGrid(nside=4), 2, True],
            ["healpix_to_coarser", HealpixGrid(nside=4), HealpixGrid(nside=2), 2, False],
            ["decode", HealpixGrid(nside=4), as_grid("equiangular", nlat=8, nlon=16), 2, True],
            ["encode", as_grid("legendre-gauss", nlat=8, nlon=16), HealpixGrid(nside=2), 2, False],
        ],
        skip_on_empty=True,
    )
    def test_matches_the_dense_reference(self, name, grid_in, grid_out, heads, use_qknorm, atol=1e-5, rtol=1e-3, verbose=False):
        batch, channels = 2, 8

        model = AttentionS2(grid_in=grid_in, grid_out=grid_out, in_channels=channels, num_heads=heads, use_qknorm=use_qknorm, bias=True).to(self.device)
        if use_qknorm:
            # zero-initialized, i.e. unit gain; perturb them so the norm is exercised
            with torch.no_grad():
                model.q_norm_weights.normal_()
                model.k_norm_weights.normal_()

        q = torch.randn(batch, channels, *grid_out.shape, device=self.device, requires_grad=True)
        k = torch.randn(batch, channels, *grid_in.shape, device=self.device, requires_grad=True)
        v = torch.randn(batch, channels, *grid_in.shape, device=self.device, requires_grad=True)

        out = model(q, k, v)
        self.assertEqual(out.shape, (batch, channels, *grid_out.shape))

        ograd = torch.randn_like(out)
        out.backward(ograd)
        param_grads = {n: p.grad.clone() for n, p in model.named_parameters()}
        model.zero_grad()

        # the reference, on flat fields; flatten(2) is a no-op on a ragged side
        dense = {n: t.detach().flatten(2).requires_grad_() for n, t in (("q", q), ("k", k), ("v", v))}
        mask = torch.ones(grid_out.npoints, grid_in.npoints, dtype=torch.bool, device=self.device)
        out_dense = _dense_masked_attention(model, dense["q"], dense["k"], dense["v"], mask)
        out_dense.backward(ograd.flatten(2))

        self.assertTrue(compare_tensors(f"{name} output", out.flatten(2), out_dense, atol=atol, rtol=rtol, verbose=verbose))
        for n, t in (("q", q), ("k", k), ("v", v)):
            self.assertTrue(compare_tensors(f"{name} input grad {n}", t.grad.flatten(2), dense[n].grad, atol=atol, rtol=rtol, verbose=verbose))
        # A parameter gradient is a sum over batch and points, so channels that nearly cancel
        # carry an absolute error set by the large terms they were summed from; the absolute
        # tolerance scales with the tensor's range accordingly
        for n, p in model.named_parameters():
            atol_n = atol * max(1.0, float(p.grad.abs().max()))
            self.assertTrue(compare_tensors(f"{name} parameter grad {n}", param_grads[n], p.grad, atol=atol_n, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            # Format: [name, grid_in, grid_out]
            ["healpix", HealpixGrid(nside=4), HealpixGrid(nside=4)],
            ["healpix_to_coarser", HealpixGrid(nside=4), HealpixGrid(nside=2)],
            ["healpix_to_finer", HealpixGrid(nside=2), HealpixGrid(nside=4)],
            ["decode", HealpixGrid(nside=4), as_grid("equiangular", nlat=8, nlon=16)],
            ["encode", as_grid("legendre-gauss", nlat=8, nlon=16), HealpixGrid(nside=2)],
        ],
        skip_on_empty=True,
    )
    def test_neighborhood_global_equivalence(self, name, grid_in, grid_out, atol=1e-5, rtol=1e-3, verbose=False):
        """
        NeighborhoodAttentionS2 with a whole-sphere cutoff reduces to AttentionS2 on ragged grids.

        The HEALPix counterpart of the regular-grid test of the same name, with two things
        that test cannot cover. First, the grids may differ: the regular path is restricted to
        identical grids because its p-shift only reproduces global attention there, but the
        ragged path has no p-shift, so resampling pairs must agree too. Second, on an
        equal-area input grid AttentionS2 passes no weight mask while the neighborhood
        kernels still apply the weights, so agreement checks that dropping the mask is exact
        through both real code paths.

        The two layers share their projection and layout code, so both are also held against
        the dense reference, which shares none of it.
        """
        batch, channels, heads = 2, 8, 2

        model_ref = AttentionS2(grid_in=grid_in, grid_out=grid_out, in_channels=channels, num_heads=heads, use_qknorm=True, bias=True)
        model = NeighborhoodAttentionS2(grid_in=grid_in, grid_out=grid_out, in_channels=channels, num_heads=heads, use_qknorm=True, bias=True, theta_cutoff=2 * math.pi)

        # zero-initialized qk-norm weights are a unit gain; perturb them so the norm is exercised,
        # then give both layers the same parameters
        with torch.no_grad():
            model_ref.q_norm_weights.normal_()
            model_ref.k_norm_weights.normal_()
        model.load_state_dict(model_ref.state_dict())
        model_ref, model = model_ref.to(self.device), model.to(self.device)

        # every output point must see every input point
        self.assertEqual(int(model._neighborhood_arcs().to_csr()[0].numel()), grid_out.npoints * grid_in.npoints)

        def make(grid):
            return torch.randn(batch, channels, *grid.shape, device=self.device)

        base = {"q": make(grid_out), "k": make(grid_in), "v": make(grid_in)}
        inputs_ref = {n: t.clone().requires_grad_() for n, t in base.items()}
        inputs = {n: t.clone().requires_grad_() for n, t in base.items()}

        out_ref = model_ref(inputs_ref["q"], inputs_ref["k"], inputs_ref["v"])
        out = model(inputs["q"], inputs["k"], inputs["v"])
        self.assertTrue(compare_tensors(f"{name} output", out, out_ref, atol=atol, rtol=rtol, verbose=verbose))

        ograd = torch.randn_like(out_ref)
        out_ref.backward(ograd)
        out.backward(ograd)

        for n in ("q", "k", "v"):
            self.assertTrue(compare_tensors(f"{name} input grad {n}", inputs[n].grad, inputs_ref[n].grad, atol=atol, rtol=rtol, verbose=verbose))
        # scaled absolute tolerance for the parameter gradients; see test_matches_the_dense_reference
        for (n, p_ref), (_, p) in zip(model_ref.named_parameters(), model.named_parameters()):
            atol_n = atol * max(1.0, float(p_ref.grad.abs().max()))
            self.assertTrue(compare_tensors(f"{name} parameter grad {n}", p.grad, p_ref.grad, atol=atol_n, rtol=rtol, verbose=verbose))

        # both layers against the dense reference, on flat fields (flatten(2) is a no-op on a
        # ragged side); last, since its backward is not compared and would accumulate into
        # the parameter gradients above
        mask = torch.ones(grid_out.npoints, grid_in.npoints, dtype=torch.bool, device=self.device)
        for layer_name, layer, out_layer, inputs_layer in (("AttentionS2", model_ref, out_ref, inputs_ref), ("NeighborhoodAttentionS2", model, out, inputs)):
            dense = {n: t.detach().flatten(2).requires_grad_() for n, t in inputs_layer.items()}
            out_dense = _dense_masked_attention(layer, dense["q"], dense["k"], dense["v"], mask)
            self.assertTrue(compare_tensors(f"{name} {layer_name} output vs dense", out_layer.flatten(2), out_dense, atol=atol, rtol=rtol, verbose=verbose))

            out_dense.backward(ograd.flatten(2))
            for n in ("q", "k", "v"):
                grad_layer = inputs_layer[n].grad.flatten(2)
                self.assertTrue(compare_tensors(f"{name} {layer_name} input grad {n} vs dense", grad_layer, dense[n].grad, atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            # Format: [name, grid_in, grid_out, masked]. Only the input grid matters: the
            # weights are over the keys.
            ["healpix", HealpixGrid(nside=2), HealpixGrid(nside=2), False],
            ["decode", HealpixGrid(nside=2), as_grid("equiangular", nlat=8, nlon=16), False],
            ["encode", as_grid("equiangular", nlat=8, nlon=16), HealpixGrid(nside=2), True],
            ["regular", as_grid("legendre-gauss", nlat=8, nlon=16), as_grid("legendre-gauss", nlat=8, nlon=16), True],
        ],
        skip_on_empty=True,
    )
    def test_mask_only_where_the_weights_differ(self, name, grid_in, grid_out, masked):
        """The log-weight mask is dropped exactly when the input grid is equal-area, where it would be a constant."""
        model = AttentionS2(grid_in=grid_in, grid_out=grid_out, in_channels=4, num_heads=1).to(self.device)
        self.assertEqual(model.log_point_weights is not None, masked, name)

    def test_equal_area_runs_on_flash_attention(self):
        """
        With no mask, SDPA can take its fused FlashAttention kernel, which accepts none.

        Restricting SDPA to that kernel alone turns "the mask is gone" into a hard
        requirement: with a mask there is no kernel to fall back to and the call raises,
        which the regular grid below confirms, so the check cannot pass vacuously.
        """
        if self.device.type != "cuda":
            raise unittest.SkipTest("FlashAttention is a CUDA kernel")
        if torch.cuda.get_device_capability(self.device) < (8, 0):
            raise unittest.SkipTest("FlashAttention needs sm_80 or newer")
        from torch.nn.attention import SDPBackend, sdpa_kernel

        def run(grid):
            # fp16 and a head dim of 8: what the flash kernel accepts
            model = AttentionS2(grid_in=grid, grid_out=grid, in_channels=16, num_heads=2).to(self.device).half()
            x = torch.randn(2, 16, *grid.shape, device=self.device, dtype=torch.float16)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return model(x)

        out = run(HealpixGrid(nside=4))
        self.assertTrue(torch.isfinite(out).all())

        with self.assertRaises(RuntimeError):
            run(as_grid("equiangular", nlat=8, nlon=16))

    def test_self_attention_on_healpix(self):
        """One tensor bound to all three inputs takes the aliased path; it must agree with passing it three times."""
        grid = HealpixGrid(nside=4)
        model = AttentionS2(grid_in=grid, grid_out=grid, in_channels=8, num_heads=2).to(self.device)
        x = torch.randn(2, 8, grid.npoints, device=self.device)
        self.assertTrue(compare_tensors("self attention", model(x), model(x, x.clone(), x.clone()), atol=1e-6, rtol=1e-5))

    def test_ragged_shape_errors(self):
        """A ragged side takes (batch, channels, npoints); a regular-shaped tensor there is rejected, and vice versa."""
        hp = HealpixGrid(nside=2)
        eq = as_grid("equiangular", nlat=8, nlon=16)
        model = AttentionS2(grid_in=hp, grid_out=eq, in_channels=4, num_heads=1).to(self.device)
        q = torch.randn(1, 4, *eq.shape, device=self.device)
        k = torch.randn(1, 4, hp.npoints, device=self.device)
        model(q, k, k)
        with self.assertRaises(RuntimeError):
            model(q, q, q)  # key/value carry the output grid's shape
        with self.assertRaises(RuntimeError):
            model(k, k, k)  # query carries the input grid's shape


@parameterized_class(("device"), _devices)
class TestMixedGridNeighborhoodAttentionS2(unittest.TestCase):
    """
    Attention between a ragged grid and a product grid: the encoder and decoder case.

    The neighbourhood is keyed by output point whichever way the resampling goes, so
    the ragged path serves both and there is no new mathematics here. What is new is
    that the two sides carry different layouts, and the thing that can go wrong is the
    flattening: a regular grid enters the ragged path as ``ilat * nlon + ilon``, and if
    that disagreed with the order the neighbourhood indexes, the result would be a
    plausible-looking field built from the wrong neighbours. The oracle below is the
    same dense masked softmax the pure-ragged tests use, addressed in flat indices
    throughout, so it pins the ordering rather than assuming it.
    """

    def setUp(self):
        disable_tf32()
        set_seed(333)

    @parameterized.expand(
        [
            # Format: [name, grid_in, grid_out]. Both directions, and both families
            # in the output role, since only the output side gets unflattened again.
            ["decode", HealpixGrid(nside=4), as_grid("equiangular", nlat=16, nlon=32)],
            ["encode", as_grid("equiangular", nlat=16, nlon=32), HealpixGrid(nside=4)],
            ["decode_to_lobatto", HealpixGrid(nside=4), as_grid("lobatto", nlat=15, nlon=30)],
            ["decode_coarser", HealpixGrid(nside=8), as_grid("equiangular", nlat=12, nlon=24)],
        ],
        skip_on_empty=True,
    )
    def test_it_matches_a_dense_masked_softmax(self, name, grid_in, grid_out, atol=1e-5, rtol=1e-3):
        """Forward and every gradient, against the oracle, in flat index space."""
        batch, channels, heads = 2, 8, 2
        model = NeighborhoodAttentionS2(grid_in=grid_in, grid_out=grid_out, in_channels=channels, num_heads=heads).to(self.device)

        mask = _brute_force_neighborhood(grid_in, grid_out, model.theta_cutoff).to(self.device)
        # an output point with an empty neighbourhood would make the oracle's softmax
        # divide by zero, which would look like a module bug rather than a test one
        self.assertTrue(bool(mask.any(dim=-1).all()), f"{name}: some output point has no neighbours")

        # The leaves are flat for both the module and the oracle, and the module's
        # view is a reshape of the same leaf. That way the gradients come back in one
        # layout and comparing them needs no reinterpretation of its own.
        def leaf(npoints):
            return torch.randn(batch, channels, npoints, device=self.device, requires_grad=True)

        flat = {"q": leaf(grid_out.npoints), "k": leaf(grid_in.npoints), "v": leaf(grid_in.npoints)}
        flat_ref = {name_: t.detach().clone().requires_grad_() for name_, t in flat.items()}

        def as_grid_shape(tensor, grid):
            return tensor if not grid.is_regular else tensor.unflatten(-1, grid.shape)

        out = model(as_grid_shape(flat["q"], grid_out), as_grid_shape(flat["k"], grid_in), as_grid_shape(flat["v"], grid_in))

        expected_shape = (batch, model.out_channels, *grid_out.shape)
        self.assertEqual(tuple(out.shape), expected_shape, f"{name}: output must be laid out on grid_out")

        out_ref = _dense_masked_attention(model, flat_ref["q"], flat_ref["k"], flat_ref["v"], mask)

        out_flat = out.reshape(batch, model.out_channels, grid_out.npoints)
        self.assertTrue(compare_tensors("output", out_flat, out_ref, atol=atol, rtol=rtol))

        # one backward per graph, from the same upstream gradient
        grad = torch.randn_like(out_ref)
        grads_ref = torch.autograd.grad(out_ref, list(flat_ref.values()) + list(model.parameters()), grad_outputs=grad, retain_graph=True)
        grads = torch.autograd.grad(out_flat, list(flat.values()) + list(model.parameters()), grad_outputs=grad)

        names = list(flat.keys()) + [pname for pname, _ in model.named_parameters()]
        for pname, got, expected in zip(names, grads, grads_ref):
            # A parameter gradient is a sum over every point and batch entry, and can nearly
            # cancel: a small absolute error on a tensor whose entries reach O(10) then fails a
            # fixed atol, and whether it does varies with the CPU kernel's OpenMP summation
            # order. Scale atol with the tensor, as the other dense-reference tests do.
            # the input gradients keep the fixed tolerance
            atol_n = atol if pname in flat else atol * max(1.0, float(expected.abs().max()))
            self.assertTrue(compare_tensors(f"grad {pname}", got, expected, atol=atol_n, rtol=rtol))

    def test_the_flattening_is_ring_major(self):
        """
        The ordering assumption, on its own so that a violation of it is not diagnosed
        as an attention bug.

        ``GridS2.lon_offsets`` places ``(ilat, ilon)`` at ``lon_offsets[ilat] + ilon``,
        which on a regular grid is ``ilat * nlon + ilon``. That is what the layer
        relies on when it reshapes a regular field into the flat axis the
        neighbourhood indexes.
        """
        grid = as_grid("equiangular", nlat=6, nlon=12)
        offsets = grid.lon_offsets
        for ilat in range(grid.nlat):
            for ilon in (0, 1, grid.nlon - 1):
                self.assertEqual(int(offsets[ilat]) + ilon, ilat * grid.nlon + ilon)

        field = torch.arange(grid.npoints, dtype=torch.float32).reshape(grid.nlat, grid.nlon)
        self.assertTrue(torch.equal(field.flatten(-2, -1), torch.arange(grid.npoints, dtype=torch.float32)))

    def test_a_constant_field_decodes_to_a_constant_field(self):
        """
        The decoder's sanity check. Attention weights are a partition of unity, so a
        constant input must come back constant on the output grid whatever the
        neighbourhoods look like -- including at the poles, where an equiangular grid
        stacks many output points onto nearly the same place.
        """
        grid_in, grid_out = HealpixGrid(nside=4), as_grid("equiangular", nlat=16, nlon=32)
        model = NeighborhoodAttentionS2(grid_in=grid_in, grid_out=grid_out, in_channels=4, num_heads=1, bias=False).to(self.device)

        # with no biases and a constant input, every value vector is the same, so the
        # output is that vector projected -- independent of the softmax entirely
        const = torch.full((1, 4, grid_in.npoints), 0.75, device=self.device)
        query = torch.full((1, 4, *grid_out.shape), 0.75, device=self.device)
        out = model(query, const, const)

        self.assertEqual(tuple(out.shape), (1, 4, *grid_out.shape))
        spread = (out - out.amin(dim=(-2, -1), keepdim=True)).abs().max().detach()
        self.assertLess(float(spread), 1e-5, "a constant field did not decode to a constant field")


@unittest.skipUnless(
    torch.cuda.is_available() and optimized_kernels_is_available(),
    "requires a CUDA device and an extension built with the ragged attention kernels",
)
class TestRaggedForwardCudaKernel(unittest.TestCase):
    """
    The ragged CUDA forward kernel, against the torch reference.

    The reference is the thing already checked against a dense masked softmax and
    against the regular kernels, so pinning the CUDA kernel to it is what carries
    that validation across. The kernel is a rewrite of the addressing only -- the
    online softmax and the quadrature weighting are copied unchanged from the
    product-grid kernel -- so a disagreement points at the arc walk, which is the
    part that is new.
    """

    def setUp(self):
        set_seed(333)
        disable_tf32()
        self.device = torch.device("cuda")

    def _run_both(self, nside, batch, num_heads, channels, dtype, q_scale=1.0):
        grid = HealpixGrid(nside=nside)
        layer = NeighborhoodAttentionS2(
            in_channels=channels,
            num_heads=num_heads,
            grid_in=grid,
            grid_out=grid,
        ).to(self.device)

        npix = grid.npoints
        packed = num_heads * (channels // num_heads)
        shape = (batch, npix, packed)
        kx = torch.randn(shape, device=self.device, dtype=dtype)
        vx = torch.randn(shape, device=self.device, dtype=dtype)
        qy = torch.randn(shape, device=self.device, dtype=dtype) * q_scale

        # the op returns its softmax statistics alongside the output, plus an fp32 copy
        # of the output in bf16 only; they exist for the backward's benefit and are
        # checked for shape and dtype below
        got, y_hi, alpha_sum, qdotk_max = torch.ops.attention_kernels.forward_ragged(
            kx.contiguous(),
            vx.contiguous(),
            qy.contiguous(),
            layer.ring_weights,
            layer.psi_seg,
            layer.psi_seg_off,
            layer.psi_ring_base,
            layer.psi_ring_size,
            num_heads,
            npix,
        )

        for name, stat in (("alpha_sum", alpha_sum), ("qdotk_max", qdotk_max)):
            self.assertEqual(stat.shape, (batch, num_heads, npix), name)
            self.assertEqual(stat.dtype, torch.float32, name)
        # alpha_sum is a sum of positive terms, and every output point has neighbours
        self.assertTrue((alpha_sum > 0).all())
        self.assertTrue(torch.isfinite(qdotk_max).all())

        # y_hi carries the output again at full precision, and only where the backward
        # needs it: in bf16, whose 8 mantissa bits cannot form integral = dy . out
        # accurately enough for the single-pass form. Empty for every other dtype, so
        # they pay nothing. Asserted because "silently absent" and "silently empty"
        # would both leave bf16 quietly back on two passes.
        self.assertEqual(y_hi.dtype, torch.float32, "y_hi")
        if dtype == torch.bfloat16:
            self.assertEqual(y_hi.shape, got.shape, "y_hi")
            self.assertTrue(torch.allclose(y_hi.to(dtype), got, atol=0, rtol=0), "y_hi must equal y once narrowed")
        else:
            self.assertEqual(y_hi.numel(), 0, "y_hi should be empty except in bf16")

        # the reference shares this op's channels-last ABI, so it takes the same
        # tensors; it differs only in consuming the CSR expansion instead of the arcs
        expected = _neighborhood_s2_attention_ragged_torch(
            kx.contiguous(),
            vx.contiguous(),
            qy.contiguous(),
            _point_weights(layer, self.device),
            # from the precompute, not the layer: with the kernels present the layer
            # holds the arcs and not the column list, and a test comparing the two
            # implementations must not depend on it carrying state for the one it did
            # not select
            *[t.to(self.device) for t in precompute_neighborhood_csr_s2(grid, grid, layer.theta_cutoff)],
            num_heads,
            npix,
        )

        return got, expected

    @parameterized.expand([(2, 1, 1, 8), (4, 2, 4, 32), (8, 1, 8, 64)])
    def test_it_matches_the_torch_reference(self, nside, batch, num_heads, channels):
        got, expected = self._run_both(nside, batch, num_heads, channels, torch.float32)
        self.assertTrue(compare_tensors("forward", got, expected, rtol=1e-5, atol=1e-5))

    def test_it_launches_above_the_default_shared_memory_limit(self):
        # Past 512 vector channels the forward takes the generic kernel, which stages one
        # row of output channels per warp in dynamic shared memory: 4 B * 6400 * 2 warps
        # = 50 KiB, above the 48 KiB a launch gets without opting in. q is scaled as the
        # layer would scale it, so the softmax stays a softmax at this width rather than
        # collapsing onto a single neighbour.
        channels = 6400
        got, expected = self._run_both(2, 1, 1, channels, torch.float32, q_scale=channels**-0.5)
        self.assertTrue(compare_tensors("forward", got, expected, rtol=1e-4, atol=1e-4))

    @parameterized.expand([(torch.float16,), (torch.bfloat16,)])
    def test_it_matches_the_torch_reference_in_reduced_precision(self, dtype):
        got, expected = self._run_both(4, 2, 4, 32, dtype)
        self.assertEqual(got.dtype, dtype)
        self.assertTrue(compare_tensors(f"forward {dtype}", got.float(), expected.float(), rtol=2e-2, atol=2e-2))

    def test_it_rejects_shapes_it_cannot_serve(self):
        grid = HealpixGrid(nside=2)
        layer = NeighborhoodAttentionS2(in_channels=8, num_heads=1, grid_in=grid, grid_out=grid).to(self.device)
        npix = grid.npoints
        good = torch.randn(1, npix, 8, device=self.device)

        args = (layer.ring_weights, layer.psi_seg, layer.psi_seg_off, layer.psi_ring_base, layer.psi_ring_size)

        # a 4-D activation is the product-grid ABI, which this op does not accept
        with self.assertRaises(RuntimeError):
            torch.ops.attention_kernels.forward_ragged(good.unsqueeze(1), good, good, *args, 1, npix)

        # npoints_out that disagrees with qy
        with self.assertRaises(RuntimeError):
            torch.ops.attention_kernels.forward_ragged(good, good, good, *args, 1, npix + 1)

        # a channel count that does not divide by num_heads
        with self.assertRaises(RuntimeError):
            torch.ops.attention_kernels.forward_ragged(good, good, good, *args, 3, npix)


@unittest.skipUnless(
    torch.cuda.is_available() and optimized_kernels_is_available(),
    "requires a CUDA device and an extension built with the ragged attention kernels",
)
class TestRaggedBackwardCudaKernel(unittest.TestCase):
    """
    The ragged CUDA backward kernel, against autograd through the torch reference.

    Differentiating the reference is a stronger check than differentiating a
    hand-written formula: the reference's own forward is already pinned to a dense
    masked softmax, so autograd through it is a gradient of something independently
    known to be right. The kernel recomputes q.k in a second pass rather than
    storing the per-neighbour alphas, so a disagreement concentrated in dk or dv
    points at that replay, and one in dq points at the three shared reductions.
    """

    def setUp(self):
        set_seed(444)
        disable_tf32()
        self.device = torch.device("cuda")

    def _grads_from_both(self, nside, batch, num_heads, channels, dtype, q_scale=1.0):
        grid = HealpixGrid(nside=nside)
        layer = NeighborhoodAttentionS2(
            in_channels=channels,
            num_heads=num_heads,
            grid_in=grid,
            grid_out=grid,
        ).to(self.device)

        npix = grid.npoints
        packed = num_heads * (channels // num_heads)
        shape = (batch, npix, packed)

        base = [torch.randn(shape, device=self.device, dtype=dtype) for _ in range(3)]
        base[2] = base[2] * q_scale
        dy = torch.randn(shape, device=self.device, dtype=dtype)

        def run(fn, weights, *pattern):
            # fresh leaves per side so the two backward passes cannot accumulate
            # into each other's .grad
            kx, vx, qy = (t.clone().detach().requires_grad_(True) for t in base)
            out = fn(kx, vx, qy, weights, *pattern, num_heads, npix)
            # the optimized op also returns the softmax statistics its backward needs;
            # the reference returns the output alone
            if isinstance(out, tuple):
                out = out[0]
            out.backward(dy)
            return kx.grad, vx.grad, qy.grad

        got = run(
            _neighborhood_s2_attention_ragged_optimized,
            layer.ring_weights,
            layer.psi_seg,
            layer.psi_seg_off,
            layer.psi_ring_base,
            layer.psi_ring_size,
        )
        # The reference's column list and per-point weights come from the precompute,
        # not from the layer. A layer on CUDA selects the optimized backend and so
        # registers the arcs and ring weights, not the columns or the point weights --
        # and a test comparing two implementations should not depend on the module
        # happening to carry state for the one it did not choose.
        col_idx, roff_idx = precompute_neighborhood_csr_s2(grid, grid, layer.theta_cutoff)
        expected = run(
            _neighborhood_s2_attention_ragged_torch,
            _point_weights(layer, self.device),
            col_idx.to(self.device),
            roff_idx.to(self.device),
        )

        return got, expected

    @parameterized.expand([(2, 1, 1, 8), (4, 2, 4, 32), (8, 1, 8, 64)])
    def test_it_matches_autograd_through_the_reference(self, nside, batch, num_heads, channels):
        got, expected = self._grads_from_both(nside, batch, num_heads, channels, torch.float32)
        for name, g, e in zip(("dk", "dv", "dq"), got, expected):
            with self.subTest(grad=name):
                self.assertTrue(compare_tensors(f"grad {name}", g, e, rtol=1e-4, atol=1e-4))

    @parameterized.expand([(torch.float16,), (torch.bfloat16,)])
    def test_it_matches_autograd_through_the_reference_in_reduced_precision(self, dtype):
        got, expected = self._grads_from_both(4, 2, 4, 32, dtype)
        for name, g, e in zip(("dk", "dv", "dq"), got, expected):
            with self.subTest(grad=name):
                self.assertEqual(g.dtype, dtype)
                self.assertTrue(compare_tensors(f"grad {name}", g.float(), e.float(), rtol=3e-2, atol=3e-2))

    def test_it_launches_above_the_default_shared_memory_limit(self):
        # Past 512 vector channels the backward takes the generic kernel, which stages
        # dy, qy and one accumulator per warp in dynamic shared memory:
        # 4 B * (2 * 2560 + 2560) * 2 warps = 60 KiB in the single-pass form, above the
        # 48 KiB a launch gets without opting in. q is scaled as in the layer.
        channels = 2560
        got, expected = self._grads_from_both(2, 1, 1, channels, torch.float32, q_scale=channels**-0.5)
        for name, g, e in zip(("dk", "dv", "dq"), got, expected):
            with self.subTest(grad=name):
                self.assertTrue(compare_tensors(f"grad {name}", g, e, rtol=1e-4, atol=1e-4))

    def test_the_layer_selects_the_optimized_path_and_stays_differentiable(self):
        # the point of the pair: with both halves present the module should pick the
        # kernel rather than the reference, and still produce a gradient
        grid = HealpixGrid(nside=4)
        layer = NeighborhoodAttentionS2(in_channels=32, num_heads=4, grid_in=grid, grid_out=grid).to(self.device)

        self.assertTrue(layer.optimized_kernel)

        x = torch.randn(2, 32, grid.npoints, device=self.device, requires_grad=True)
        layer(x).sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_a_cpu_built_module_runs_on_cpu_and_after_moving(self):
        # Building on CPU and moving with .to() is normal, so the backend cannot be
        # fixed at construction: the module must run where it was built and, once
        # moved, on the device it was moved to, and agree between the two.
        grid = HealpixGrid(nside=2)
        layer = NeighborhoodAttentionS2(in_channels=8, num_heads=1, grid_in=grid, grid_out=grid)

        x = torch.randn(1, 8, grid.npoints, requires_grad=True)
        out = layer(x)
        out.sum().backward()

        self.assertTrue(torch.isfinite(out).all())
        self.assertIsNotNone(x.grad)

        # and the same module, once moved, must take the kernel
        layer = layer.to(self.device)
        xc = x.detach().to(self.device).requires_grad_(True)
        out_cuda = layer(xc)
        out_cuda.sum().backward()

        self.assertTrue(torch.isfinite(out_cuda).all())
        self.assertTrue(compare_tensors("cpu vs cuda forward", out_cuda.cpu(), out, atol=1e-4, rtol=1e-4))

    @requires_torch_compile
    def test_the_op_survives_torch_compile(self):
        """
        A fake whose signature has drifted from its schema is invisible until
        something traces the graph.

        Eager execution calls the CUDA implementation directly and never consults a
        fake, so a stale one passes every other test here. Only tracing reads them,
        and only tracing a *backward* reads the backward's -- which is how
        `backward_ragged` came to be traced with one argument more than its fake
        accepted, after y_hi was added to its schema. Every gate in
        rebuild_and_validate_ragged.sh passed on that build; the first thing to
        notice was a benchmark whose compiled column had quietly gone empty.

        So this compiles a forward and a backward, which is the cheapest thing that
        reads both fakes, and asserts the gradients arrive. It is a registration
        test, not a numerics test -- the accuracy of the compiled path is the same
        kernel the other tests already check.
        """
        grid = HealpixGrid(nside=4)
        layer = NeighborhoodAttentionS2(in_channels=8, num_heads=2, grid_in=grid, grid_out=grid).to(self.device)

        # channels-first, which is the layer's ABI. The raw op takes the transpose --
        # (batch, npoints, packed channels) -- and mixing the two up gets caught by a
        # shape check naming the point count, which reads like a channel error.
        x = torch.randn(1, 8, grid.npoints, device=self.device, requires_grad=True)

        compiled = torch.compile(layer, dynamic=False)
        compiled(x, x, x).sum().backward()

        self.assertIsNotNone(x.grad, "no gradient came back through the compiled op")
        self.assertTrue(torch.isfinite(x.grad).all(), "compiled backward produced non-finite gradients")

    def test_dk_and_dv_accumulate_over_overlapping_neighborhoods(self):
        # dk/dv are scatter-accumulated with atomicAdd because neighbourhoods overlap.
        # If the buffers were allocated with empty() instead of zeros(), or a
        # contribution were dropped, the gradient of a point that many neighbourhoods
        # touch would be wrong -- so require every input point to receive one.
        grid = HealpixGrid(nside=4)
        layer = NeighborhoodAttentionS2(in_channels=8, num_heads=1, grid_in=grid, grid_out=grid).to(self.device)

        npix = grid.npoints
        kx, vx, qy = (torch.randn(1, npix, 8, device=self.device, requires_grad=True) for _ in range(3))

        out, y_hi, alpha_sum, qdotk_max = _neighborhood_s2_attention_ragged_optimized(
            kx,
            vx,
            qy,
            layer.ring_weights,
            layer.psi_seg,
            layer.psi_seg_off,
            layer.psi_ring_base,
            layer.psi_ring_size,
            1,
            npix,
        )

        # the statistics are the backward's own bookkeeping, so nothing must be able to
        # route a gradient through them -- see _setup_context_attention_ragged_backward.
        # y_hi is the same for the same reason and one more: it is the output again, so
        # a gradient through it would be counted twice.
        self.assertFalse(alpha_sum.requires_grad)
        self.assertFalse(qdotk_max.requires_grad)
        self.assertFalse(y_hi.requires_grad)

        out.backward(torch.ones_like(out))

        for name, g in (("dk", kx.grad), ("dv", vx.grad)):
            with self.subTest(grad=name):
                self.assertTrue(torch.isfinite(g).all())
                touched = (g.abs().sum(dim=-1) > 0).sum().item()
                self.assertEqual(touched, npix, f"{name}: only {touched} of {npix} input points received a gradient")


if __name__ == "__main__":
    unittest.main()
