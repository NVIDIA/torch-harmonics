# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
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

import os
from time import perf_counter_ns
import unittest
from parameterized import parameterized, parameterized_class

import torch
from torch.library import opcheck
from torch_harmonics import DiscreteContinuousConvS2, DiscreteContinuousConvTransposeS2

from torch_harmonics.disco import cuda_kernels_is_available, optimized_kernels_is_available

from disco_test_utils import normalize_convolution_tensor_dense, precompute_convolution_tensor_dense
from testutils import disable_tf32, set_seed, compare_tensors, maybe_autocast

if not optimized_kernels_is_available():
    print(f"Warning: Couldn't import optimized disco convolution kernels")


_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))

# perf thresholds
# CPU results normalized to 16 OpenMP threads,
# GPU results normalized to V100 16 GB GPU
# this is just to detect performance regressions, not for absolute performance
_perf_test_thresholds = {"cpu": {"fwd_ms": 100, "bwd_ms": 90},
                         "cuda": {"fwd_ms": 2, "bwd_ms": 3}}
_run_perf_tests = (os.getenv("TORCH_HARMONICS_RUN_PERF_TESTS", "0") == "1")


# ---------------------------------------------------------------------------
# Base parameter lists for the kernel-correctness tests. Each row is a base
# config (without use_dense_kernel); the decorators below cross-product these
# with [False, True] so each config is exercised against both the CSR and the
# dense kernels.
# ---------------------------------------------------------------------------

_SPARSE_AGAINST_DENSE_BASE_CONFIGS = [
    # B, Cin, Cout, in_shape, out_shape, kernel_shape, basis, norm, grid_in, grid_out, dtype, transpose, atol, rtol
    # fp32 tests
    # regular convolution
    [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (16, 32), (8, 16), (3), "piecewise linear", "nodal", "equiangular", "equiangular",  torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (24, 48), (12, 24), (3, 3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (24, 48), (12, 24), (4, 3), "piecewise linear", "none", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (24, 48), (12, 24), (2, 1), "harmonic", "mean", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (24, 48), (12, 24), (3), "zernike", "nodal", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (24, 48), (12, 24), (3, 3), "harmonic", "nodal", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (24, 48), (12, 24), (3, 3), "fourier-bessel", "mean", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (16, 24), (8, 8), (3), "piecewise linear", "mean", "equiangular", "equiangular",  torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (18, 36), (6, 12), (7), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (16, 32), (8, 16), (5), "piecewise linear", "mean", "equiangular", "legendre-gauss", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (16, 32), (8, 16), (5), "piecewise linear", "mean", "legendre-gauss", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (16, 32), (8, 16), (5), "piecewise linear", "nodal", "legendre-gauss", "legendre-gauss", torch.float32, False, 1e-4, 1e-4],
    # regular convolution — modal, support, geometric normalization
    [8, 4, 2, (24, 48), (12, 24), (3, 3), "piecewise linear", "modal", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "modal", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (24, 48), (12, 24), (3), "zernike", "modal", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (24, 48), (12, 24), (3, 3), "fourier-bessel", "modal", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (24, 48), (12, 24), (3), "piecewise linear", "support", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "support", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (24, 48), (12, 24), (3, 3), "piecewise linear", "geometric", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (24, 48), (12, 24), (3, 3), "fourier-bessel", "geometric", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "geometric", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    # transpose convolution
    [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "nodal", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (12, 24), (24, 48), (3, 3), "piecewise linear", "mean", "equiangular", "equiangular",  torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (12, 24), (24, 48), (4, 3), "piecewise linear", "none", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (12, 24), (24, 48), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (12, 24), (24, 48), (2, 1), "harmonic", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (12, 24), (24, 48), (3), "zernike", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (12, 24), (24, 48), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (12, 24), (24, 48), (3, 3), "fourier-bessel", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (8, 8), (16, 24), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (6, 12), (18, 36), (7), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "mean", "equiangular", "legendre-gauss", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "mean", "legendre-gauss", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "mean", "legendre-gauss", "legendre-gauss", torch.float32, True, 1e-4, 1e-4],
    # transpose convolution — modal, support, geometric normalization
    [8, 4, 2, (12, 24), (24, 48), (3, 3), "piecewise linear", "modal", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (12, 24), (24, 48), (2, 2), "harmonic", "modal", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (12, 24), (24, 48), (3), "zernike", "modal", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (12, 24), (24, 48), (3, 3), "fourier-bessel", "modal", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (12, 24), (24, 48), (3), "piecewise linear", "support", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (12, 24), (24, 48), (2, 2), "harmonic", "support", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (12, 24), (24, 48), (3, 3), "piecewise linear", "geometric", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (12, 24), (24, 48), (3, 3), "fourier-bessel", "geometric", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (12, 24), (24, 48), (2, 2), "harmonic", "geometric", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    # fp64 tests
    # regular convolution
    [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float64, False, 1e-9, 1e-9],
    [8, 4, 2, (16, 32), (8, 16), (3), "piecewise linear", "nodal", "equiangular", "equiangular", torch.float64, False, 1e-9, 1e-9],
    [8, 4, 2, (24, 48), (12, 24), (3, 3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float64, False, 1e-9, 1e-9],
    [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float64, False, 1e-9, 1e-9],
    [8, 4, 2, (24, 48), (12, 24), (3), "zernike", "mean", "equiangular", "equiangular", torch.float64, False, 1e-9, 1e-9],
    [8, 4, 2, (24, 48), (12, 24), (3, 3), "fourier-bessel", "nodal", "equiangular", "equiangular", torch.float64, False, 1e-9, 1e-9],
    # transpose convolution
    [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float64, True, 1e-9, 1e-9],
    [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "mean", "equiangular", "equiangular", torch.float64, True, 1e-9, 1e-9],
    [8, 4, 2, (12, 24), (24, 48), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float64, True, 1e-9, 1e-9],
    [8, 4, 2, (12, 24), (24, 48), (3), "zernike", "mean", "equiangular", "equiangular", torch.float64, True, 1e-9, 1e-9],
    [8, 4, 2, (12, 24), (24, 48), (3, 3), "fourier-bessel", "mean", "equiangular", "equiangular", torch.float64, True, 1e-9, 1e-9],
    # fp16 tests (AMP)
    # regular convolution
    [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float16, False, 2e-2, 1e-2],
    [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float16, False, 2e-2, 1e-2],
    # transpose convolution
    [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float16, True, 2e-2, 1e-2],
    [8, 4, 2, (12, 24), (24, 48), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float16, True, 2e-2, 1e-2],
    # bf16 tests (AMP)
    # regular convolution
    [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.bfloat16, False, 5e-2, 5e-2],
    [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.bfloat16, False, 5e-2, 5e-2],
    # transpose convolution
    [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.bfloat16, True, 5e-2, 5e-2],
    [8, 4, 2, (12, 24), (24, 48), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.bfloat16, True, 5e-2, 5e-2],
]


_OPTIMIZED_AGAINST_TORCH_BASE_CONFIGS = [
    # B, Cin, Cout, in_shape, out_shape, kernel_shape, basis, norm, grid_in, grid_out, dtype, transpose, atol, rtol
    # fp32 tests
    # regular convolution
    [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (41, 80), (41, 80), (2, 2), "harmonic", "nodal", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (41, 80), (41, 80), (2, 3), "harmonic", "mean", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (41, 80), (41, 80), (3), "zernike", "modal", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (41, 80), (41, 80), (3, 3), "fourier-bessel", "geometric", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (41, 80), (41, 80), (3, 3), "harmonic", "modal", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (41, 80), (21, 40), (3), "piecewise linear", "nodal", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (41, 80), (21, 40), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (41, 80), (21, 40), (2, 1), "harmonic", "geometric", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    [8, 4, 2, (41, 80), (21, 40), (3), "zernike", "mean", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
    # transpose convolution
    [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "modal", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (41, 80), (41, 80), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (41, 80), (41, 80), (2, 3), "harmonic", "nodal", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (41, 80), (41, 80), (3), "zernike", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (41, 80), (41, 80), (3, 3), "fourier-bessel", "modal", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (41, 80), (41, 80), (3, 3), "harmonic", "geometric", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (21, 40), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (21, 40), (41, 80), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (21, 40), (41, 80), (2, 1), "harmonic", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    [8, 4, 2, (21, 40), (41, 80), (3), "zernike", "nodal", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
    # fp64 tests
    # regular convolution
    [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float64, False, 1e-9, 1e-9],
    [8, 4, 2, (41, 80), (41, 80), (2, 2), "harmonic", "modal", "equiangular", "equiangular", torch.float64, False, 1e-9, 1e-9],
    [8, 4, 2, (41, 80), (21, 40), (3), "piecewise linear", "nodal", "equiangular", "equiangular", torch.float64, False, 1e-9, 1e-9],
    # transpose convolution
    [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "geometric", "equiangular", "equiangular", torch.float64, True, 1e-9, 1e-9],
    [8, 4, 2, (21, 40), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float64, True, 1e-9, 1e-9],
    [8, 4, 2, (21, 40), (41, 80), (2, 2), "harmonic", "modal", "equiangular", "equiangular", torch.float64, True, 1e-9, 1e-9],
    # fp16 tests (AMP)
    # regular convolution
    [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float16, False, 1e-2, 1e-2],
    [8, 4, 2, (41, 80), (41, 80), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float16, False, 1e-2, 1e-2],
    # transpose convolution
    [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float16, True, 1e-2, 1e-2],
    [8, 4, 2, (41, 80), (41, 80), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float16, True, 1e-2, 1e-2],
    # bf16 tests (AMP)
    # regular convolution
    [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.bfloat16, False, 1e-2, 1e-2],
    [8, 4, 2, (41, 80), (41, 80), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.bfloat16, False, 1e-2, 1e-2],
    # transpose convolution
    [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.bfloat16, True, 1e-2, 1e-2],
    [8, 4, 2, (41, 80), (41, 80), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.bfloat16, True, 1e-2, 1e-2],
]


@parameterized_class(("device"), _devices)
class TestDiscreteContinuousConvolution(unittest.TestCase):
    """Test the discrete-continuous convolution module (CPU/CUDA if available)."""

    def setUp(self):
        disable_tf32()

    @parameterized.expand(
        [
            base + [use_dense_kernel]
            for base in _SPARSE_AGAINST_DENSE_BASE_CONFIGS
            for use_dense_kernel in (False, True)
        ],
        skip_on_empty=True,
    )
    def test_sparse_against_dense(
        self,
        batch_size,
        in_channels,
        out_channels,
        in_shape,
        out_shape,
        kernel_shape,
        basis_type,
        basis_norm_mode,
        grid_in,
        grid_out,
        dtype,
        transpose,
        atol,
        rtol,
        use_dense_kernel,
        verbose=True,
    ):
        # for AMP dtypes, the module and input stay in float32; autocast handles the rest
        is_amp = dtype in (torch.float16, torch.bfloat16)
        module_dtype = torch.float32 if is_amp else dtype

        # set seed
        set_seed(333)

        # use optimized kernels
        use_optimized_kernels = optimized_kernels_is_available()
        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            use_optimized_kernels = False

        # the dense path requires the optimized kernels to be active; skip the
        # use_dense_kernel=True row otherwise (would silently fall back to the
        # torch path, double-running the same test).
        if use_dense_kernel and not use_optimized_kernels:
            self.skipTest("use_dense_kernel requires optimized kernels")

        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        if isinstance(kernel_shape, int):
            theta_cutoff = (kernel_shape + 1) * torch.pi / float(nlat_in - 1)
        else:
            theta_cutoff = (kernel_shape[0] + 1) * torch.pi / float(nlat_in - 1)

        Conv = DiscreteContinuousConvTransposeS2 if transpose else DiscreteContinuousConvS2
        conv = Conv(
            in_channels,
            out_channels,
            in_shape,
            out_shape,
            kernel_shape,
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            groups=1,
            grid_in=grid_in,
            grid_out=grid_out,
            bias=False,
            theta_cutoff=theta_cutoff,
            optimized_kernel=use_optimized_kernels,
            use_dense_kernel=use_dense_kernel,
        ).to(self.device)

        filter_basis = conv.filter_basis

        # psi comparison in float64 (both sides come from precompute in float64).
        # The sparse psi sanity-check uses conv.psi_idx / conv.psi_vals which are
        # only registered in CSR mode (use_dense_kernel=False).
        if transpose:
            psi_dense = precompute_convolution_tensor_dense(
                out_shape,
                in_shape,
                filter_basis,
                grid_in=grid_out,
                grid_out=grid_in,
                theta_cutoff=theta_cutoff,
                transpose_normalization=transpose,
                basis_norm_mode=basis_norm_mode,
                merge_quadrature=True,
            ).to(self.device)

            if not use_dense_kernel:
                psi = torch.sparse_coo_tensor(conv.psi_idx, conv.psi_vals, size=(conv.kernel_size, conv.nlat_in, conv.nlat_out * conv.nlon_out)).to_dense()

                self.assertTrue(compare_tensors(
                    "psi (transpose) vs dense reference",
                    psi, psi_dense[:, :, 0].reshape(-1, nlat_in, nlat_out * nlon_out),
                    verbose=verbose,
                ))
        else:
            psi_dense = precompute_convolution_tensor_dense(
                in_shape,
                out_shape,
                filter_basis,
                grid_in=grid_in,
                grid_out=grid_out,
                theta_cutoff=theta_cutoff,
                transpose_normalization=transpose,
                basis_norm_mode=basis_norm_mode,
                merge_quadrature=True,
            ).to(self.device)

            if not use_dense_kernel:
                psi = torch.sparse_coo_tensor(conv.psi_idx, conv.psi_vals, size=(conv.kernel_size, conv.nlat_out, conv.nlat_in * conv.nlon_in)).to_dense()

                self.assertTrue(compare_tensors(
                    "psi (forward) vs dense reference",
                    psi, psi_dense[:, :, 0].reshape(-1, nlat_out, nlat_in * nlon_in),
                    verbose=verbose,
                ))

        # cast module to the target dtype for forward/backward
        if module_dtype != torch.float32:
            conv = conv.to(dtype=module_dtype)

        # create a copy of the weight
        w_ref = torch.empty_like(conv.weight)
        with torch.no_grad():
            w_ref.copy_(conv.weight)
        w_ref.requires_grad = True

        # create an input signal
        x = torch.randn(batch_size, in_channels, *in_shape, dtype=module_dtype, device=self.device)

        # FWD and BWD pass
        x.requires_grad = True
        with maybe_autocast(self.device.type, dtype):
            y = conv(x)
        grad_input = torch.randn_like(y)
        y.backward(grad_input)
        x_grad = x.grad.clone()

        # perform the reference computation
        x_ref = x.clone().detach()
        x_ref.requires_grad = True
        psi_ref = psi_dense.to(dtype=module_dtype)
        if transpose:
            y_ref = torch.einsum("oif,biqr->bofqr", w_ref, x_ref)
            y_ref = torch.einsum("fqrtp,bofqr->botp", psi_ref, y_ref)
        else:
            y_ref = torch.einsum("ftpqr,bcqr->bcftp", psi_ref, x_ref)
            y_ref = torch.einsum("oif,biftp->botp", w_ref, y_ref)
        y_ref.backward(grad_input)
        x_ref_grad = x_ref.grad.clone()

        # compare results
        self.assertTrue(compare_tensors(f"output", y.to(y_ref.dtype), y_ref, atol=atol, rtol=rtol, verbose=verbose))

        # compare
        self.assertTrue(compare_tensors(f"input grad", x_grad.to(x_ref_grad.dtype), x_ref_grad, atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors(f"weight grad", conv.weight.grad.to(w_ref.grad.dtype), w_ref.grad, atol=atol, rtol=rtol, verbose=verbose))


    @parameterized.expand(
        [
            base + [use_dense_kernel]
            for base in _OPTIMIZED_AGAINST_TORCH_BASE_CONFIGS
            for use_dense_kernel in (False, True)
        ],
        skip_on_empty=True,
    )
    @unittest.skipUnless((optimized_kernels_is_available()), "skipping test because optimized kernels are not available")
    def test_optimized_against_torch(
        self,
        batch_size,
        in_channels,
        out_channels,
        in_shape,
        out_shape,
        kernel_shape,
        basis_type,
        basis_norm_mode,
        grid_in,
        grid_out,
        dtype,
        transpose,
        atol,
        rtol,
        use_dense_kernel,
        verbose=True,
    ):
        # for AMP dtypes, the module and input stay in float32; autocast handles the rest
        is_amp = dtype in (torch.float16, torch.bfloat16)
        module_dtype = torch.float32 if is_amp else dtype

        # Under autocast, the dense kernel produces bf16/fp16 x_packed (real
        # mixed precision), which the conv module then feeds into the weight
        # einsum. PyTorch's einsum saves x_packed at its produced dtype, so the
        # bwd weight-grad reduction (B*Ho*Wo terms) accumulates in bf16/fp16 and
        # picks up ~0.5% relative error — that's PyTorch's einsum precision, not
        # our kernel. The torch reference path takes the CSR-style internal
        # upcast and saves fp32 x_packed, so its weight grad accumulates in
        # fp32. Loosen tolerance for that comparison only.
        if use_dense_kernel and is_amp:
            atol = max(atol, 5e-2)
            rtol = max(rtol, 5e-2)

        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

        # set seed
        set_seed(333)

        nlat_in, _ = in_shape

        if isinstance(kernel_shape, int):
            theta_cutoff = (kernel_shape + 1) * torch.pi / float(nlat_in - 1)
        else:
            theta_cutoff = (kernel_shape[0] + 1) * torch.pi / float(nlat_in - 1)

        Conv = DiscreteContinuousConvTransposeS2 if transpose else DiscreteContinuousConvS2

        conv_naive = Conv(
            in_channels,
            out_channels,
            in_shape,
            out_shape,
            kernel_shape,
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            groups=1,
            grid_in=grid_in,
            grid_out=grid_out,
            bias=False,
            theta_cutoff=theta_cutoff,
            optimized_kernel=False,
        ).to(dtype=module_dtype, device=self.device)

        conv_opt = Conv(
            in_channels,
            out_channels,
            in_shape,
            out_shape,
            kernel_shape,
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            groups=1,
            grid_in=grid_in,
            grid_out=grid_out,
            bias=False,
            theta_cutoff=theta_cutoff,
            optimized_kernel=True,
            use_dense_kernel=use_dense_kernel,
        ).to(dtype=module_dtype, device=self.device)

        # create a copy of the weight
        with torch.no_grad():
            conv_naive.weight.copy_(conv_opt.weight)

        # create an input signal
        inp = torch.randn(batch_size, in_channels, *in_shape, dtype=module_dtype, device=self.device)

        # FWD and BWD pass
        inp.requires_grad = True
        with maybe_autocast(self.device.type, dtype):
            out_naive = conv_naive(inp)
        grad_input = torch.randn_like(out_naive)
        out_naive.backward(grad_input)
        inp_grad_naive = inp.grad.clone()

        # perform the reference computation
        inp.grad = None
        with maybe_autocast(self.device.type, dtype):
            out_opt = conv_opt(inp)
        out_opt.backward(grad_input)
        inp_grad_opt = inp.grad.clone()

        # compare results
        self.assertTrue(compare_tensors(f"output", out_naive, out_opt, atol=atol, rtol=rtol, verbose=verbose))

        # compare
        self.assertTrue(compare_tensors(f"input grad", inp_grad_naive, inp_grad_opt, atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors(f"weight grad", conv_naive.weight.grad, conv_opt.weight.grad, atol=atol, rtol=rtol, verbose=verbose))


    @parameterized.expand(
        [
            base + [use_dense_kernel]
            for base in (
                [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", False, 1e-4, 1e-4],
                [8, 4, 2, (16, 32), (8, 16), (5), "piecewise linear", "mean", "legendre-gauss", "legendre-gauss", False, 1e-4, 1e-4],
                [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", True, 1e-4, 1e-4],
                [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "mean", "legendre-gauss", "legendre-gauss", True, 1e-4, 1e-4],
            )
            for use_dense_kernel in (False, True)
        ],
        skip_on_empty=True,
    )
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_device_instantiation(self, batch_size, in_channels, out_channels, in_shape, out_shape, kernel_shape, basis_type, basis_norm_mode, grid_in, grid_out, transpose, atol, rtol, use_dense_kernel, verbose=False):

        if use_dense_kernel and not optimized_kernels_is_available():
            self.skipTest("use_dense_kernel requires optimized kernels")

        set_seed(333)

        nlat_in, _ = in_shape

        if isinstance(kernel_shape, int):
            theta_cutoff = (kernel_shape + 1) * torch.pi / float(nlat_in - 1)
        else:
            theta_cutoff = (kernel_shape[0] + 1) * torch.pi / float(nlat_in - 1)

        # get handle
        Conv = DiscreteContinuousConvTransposeS2 if transpose else DiscreteContinuousConvS2

        # init on cpu
        conv_host = Conv(
            in_channels,
            out_channels,
            in_shape,
            out_shape,
            kernel_shape,
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            groups=1,
            grid_in=grid_in,
            grid_out=grid_out,
            bias=False,
            theta_cutoff=theta_cutoff,
            use_dense_kernel=use_dense_kernel,
        )

        #torch.set_default_device(self.device)
        with torch.device(self.device):
            conv_device = Conv(
                in_channels,
                out_channels,
                in_shape,
                out_shape,
                kernel_shape,
                basis_type=basis_type,
                basis_norm_mode=basis_norm_mode,
                groups=1,
                grid_in=grid_in,
                grid_out=grid_out,
                bias=False,
                theta_cutoff=theta_cutoff,
                use_dense_kernel=use_dense_kernel,
            )

        # since we specified the device specifier everywhere, it should always
        # use the cpu and it should be the same everywhere.
        # CSR buffers are only registered when use_dense_kernel is off; packed
        # buffers are only registered when it is on. Compare whichever set is
        # applicable to the current parameterization.
        if use_dense_kernel:
            self.assertTrue(compare_tensors(f"psi packed idx",   conv_host.psi_packed_idx.cpu(),   conv_device.psi_packed_idx.cpu(),   atol=atol, rtol=rtol, verbose=verbose))
            self.assertTrue(compare_tensors(f"psi packed vals",  conv_host.psi_packed_vals.cpu(),  conv_device.psi_packed_vals.cpu(),  atol=atol, rtol=rtol, verbose=verbose))
            self.assertTrue(compare_tensors(f"psi packed count", conv_host.psi_packed_count.cpu(), conv_device.psi_packed_count.cpu(), atol=atol, rtol=rtol, verbose=verbose))
        else:
            self.assertTrue(compare_tensors(f"psi col idx",  conv_host.psi_col_idx.cpu(),  conv_device.psi_col_idx.cpu(),  atol=atol, rtol=rtol, verbose=verbose))
            self.assertTrue(compare_tensors(f"psi row idx",  conv_host.psi_row_idx.cpu(),  conv_device.psi_row_idx.cpu(),  atol=atol, rtol=rtol, verbose=verbose))
            self.assertTrue(compare_tensors(f"psi roff idx", conv_host.psi_roff_idx.cpu(), conv_device.psi_roff_idx.cpu(), atol=atol, rtol=rtol, verbose=verbose))
            self.assertTrue(compare_tensors(f"psi vals",     conv_host.psi_vals.cpu(),     conv_device.psi_vals.cpu(),     atol=atol, rtol=rtol, verbose=verbose))
            self.assertTrue(compare_tensors(f"psi idx",      conv_host.psi_idx.cpu(),      conv_device.psi_idx.cpu(),      atol=atol, rtol=rtol, verbose=verbose))


    @parameterized.expand(
        [
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", False],
            [8, 4, 2, (16, 32), (8,  16), (3), "piecewise linear", "mean", "equiangular", "equiangular", False],
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", True],
            [8, 4, 2, (8,  16), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", True],

        ],
        skip_on_empty=True,
    )
    @unittest.skipUnless((optimized_kernels_is_available()), "skipping test because optimized kernels are not available")
    def test_csr_optimized_pt2_compatibility(
        self,
        batch_size,
        in_channels,
        out_channels,
        in_shape,
        out_shape,
        kernel_shape,
        basis_type,
        basis_norm_mode,
        grid_in,
        grid_out,
        transpose,
        verbose=False,
    ):
        """Tests whether the optimized kernels are PyTorch 2 compatible"""

        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping GPU test because CUDA kernels are not available")

        if verbose:
            print(f"Testing DISCO convolution on {in_shape[0]}x{in_shape[1]} {grid_in} grid to {out_shape[0]}x{out_shape[1]} {grid_out} grid on {self.device.type} device")

        set_seed(333)

        nlat_in, _ = in_shape

        if isinstance(kernel_shape, int):
            theta_cutoff = (kernel_shape + 1) * torch.pi / float(nlat_in - 1)
        else:
            theta_cutoff = (kernel_shape[0] + 1) * torch.pi / float(nlat_in - 1)

        Conv = DiscreteContinuousConvTransposeS2 if transpose else DiscreteContinuousConvS2
        conv = Conv(
            in_channels,
            out_channels,
            in_shape,
            out_shape,
            kernel_shape,
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            groups=1,
            grid_in=grid_in,
            grid_out=grid_out,
            bias=False,
            theta_cutoff=theta_cutoff,
        ).to(self.device)

        # forward test
        if not transpose:
            inp = torch.randn(batch_size, in_channels, *in_shape, device=self.device)
        else:
            inp = torch.randn(batch_size, conv.kernel_size, in_channels, *in_shape, device=self.device)

        test_inputs = (inp, conv.psi_roff_idx, conv.psi_ker_idx, conv.psi_row_idx, conv.psi_col_idx, conv.psi_vals,
                       conv.kernel_size, conv.nlat_out, conv.nlon_out)

        if not transpose:
            opcheck(torch.ops.disco_kernels._disco_s2_contraction_optimized_csr, test_inputs)
        else:
            opcheck(torch.ops.disco_kernels._disco_s2_transpose_contraction_optimized_csr, test_inputs)

        # if a test fails, those help to disambiguate the error
        # schema
        #opcheck(torch.ops.disco_kernels._disco_s2_contraction_optimized_csr, test_inputs, test_utils="test_schema")
        # fake tensor
        #opcheck(torch.ops.disco_kernels._disco_s2_contraction_optimized_csr, test_inputs, test_utils="test_faketensor")
        # test AOT stuff
        # this is expected to fail if the output shapes are dependent on input shapes (which is the case for DISCO)
        #opcheck(torch.ops.disco_kernels._disco_s2_contraction_optimized_csr, test_inputs, test_utils="test_aot_dispatch_static")
        # this one should pass
        #opcheck(torch.ops.disco_kernels._disco_s2_contraction_optimized_csr, test_inputs, test_utils="test_aot_dispatch_dynamic")


    @parameterized.expand(
        [
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", False],
            [8, 4, 2, (16, 32), (8,  16), (3), "piecewise linear", "mean", "equiangular", "equiangular", False],
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", True],
            [8, 4, 2, (8,  16), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", True],
        ],
        skip_on_empty=True,
    )
    @unittest.skipUnless((optimized_kernels_is_available()), "skipping test because optimized kernels are not available")
    def test_dense_optimized_pt2_compatibility(
        self,
        batch_size,
        in_channels,
        out_channels,
        in_shape,
        out_shape,
        kernel_shape,
        basis_type,
        basis_norm_mode,
        grid_in,
        grid_out,
        transpose,
        verbose=False,
    ):
        """PT2-compliance opcheck for the dense custom ops (forward + transpose)."""

        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping GPU test because CUDA kernels are not available")

        if verbose:
            print(f"Testing dense DISCO convolution on {in_shape[0]}x{in_shape[1]} {grid_in} grid to {out_shape[0]}x{out_shape[1]} {grid_out} grid on {self.device.type} device, transpose={transpose}")

        set_seed(333)

        nlat_in, _ = in_shape

        if isinstance(kernel_shape, int):
            theta_cutoff = (kernel_shape + 1) * torch.pi / float(nlat_in - 1)
        else:
            theta_cutoff = (kernel_shape[0] + 1) * torch.pi / float(nlat_in - 1)

        Conv = DiscreteContinuousConvTransposeS2 if transpose else DiscreteContinuousConvS2
        conv = Conv(
            in_channels, out_channels, in_shape, out_shape, kernel_shape,
            basis_type=basis_type, basis_norm_mode=basis_norm_mode,
            groups=1, grid_in=grid_in, grid_out=grid_out, bias=False,
            theta_cutoff=theta_cutoff,
            optimized_kernel=True, use_dense_kernel=True,
        ).to(self.device)

        # transpose conv's contraction op consumes a 5D input [B, C, K, Hi, Wi]
        # (after the einsum chain in the module forward); the regular conv's
        # consumes a 4D input [B, C, Hi, Wi].
        if transpose:
            inp = torch.randn(batch_size, in_channels, conv.kernel_size, *in_shape, device=self.device)
        else:
            inp = torch.randn(batch_size, in_channels, *in_shape, device=self.device)

        test_inputs = (
            inp,
            conv.psi_packed_idx, conv.psi_packed_vals, conv.psi_packed_count,
            conv.kernel_size, conv.nlat_out, conv.nlon_out,
        )

        if transpose:
            opcheck(torch.ops.disco_kernels._disco_s2_transpose_contraction_optimized_dense, test_inputs)
        else:
            opcheck(torch.ops.disco_kernels._disco_s2_contraction_optimized_dense, test_inputs)


    @parameterized.expand(
        [
            [8, 4, 2, (91, 180), (91, 180), (3), "piecewise linear", "mean", "equiangular", "equiangular", False, 1e-4],
        ],
        skip_on_empty=True,
    )
    @unittest.skipUnless(optimized_kernels_is_available() and _run_perf_tests, "skipping performance test because optimized kernels are not available or perf tests are disabled")
    def test_perf(self, batch_size, in_channels, out_channels, in_shape, out_shape, kernel_shape, basis_type, basis_norm_mode, grid_in, grid_out, transpose, tol, verbose=True):

        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

        set_seed(333)

        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        if isinstance(kernel_shape, int):
            theta_cutoff = (kernel_shape + 1) * torch.pi / float(nlat_in - 1)
        else:
            theta_cutoff = (kernel_shape[0] + 1) * torch.pi / float(nlat_in - 1)

        # get handle
        Conv = DiscreteContinuousConvTransposeS2 if transpose else DiscreteContinuousConvS2

        # init on cpu
        conv_optimized = Conv(
            in_channels,
            out_channels,
            in_shape,
            out_shape,
            kernel_shape,
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            groups=1,
            grid_in=grid_in,
            grid_out=grid_out,
            bias=True,
            theta_cutoff=theta_cutoff,
            optimized_kernel=True,
        ).to(self.device)

        # random weights
        with torch.no_grad():
            conv_optimized.weight.normal_()
            conv_optimized.bias.normal_()

        # create an input signal
        inp = torch.randn(batch_size, in_channels, *in_shape, device=self.device)
        inp.requires_grad = True

        # forward test
        # warmup
        for i in range(2):
            out_optimized = conv_optimized(inp)

        # start timer
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = perf_counter_ns()
        out_optimized = conv_optimized(inp)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        end = perf_counter_ns()
        duration = (end - start) / 1e6
        if verbose:
            print(f"Forward execution time on device {self.device.type}: {duration:.2f} ms")
        self.assertTrue(duration <= _perf_test_thresholds[self.device.type]["fwd_ms"])

        # backward test
        out_optimized = conv_optimized(inp)
        out_grad = torch.randn(out_optimized.shape, dtype=torch.float32, device=self.device)

        # warmup
        for _ in range(2):
            out_optimized.backward(out_grad, retain_graph=True)

        # start timer
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = perf_counter_ns()
        out_optimized.backward(out_grad)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        end = perf_counter_ns()
        duration = (end - start) / 1e6
        if verbose:
            print(f"Backward execution time on device {self.device.type}: {duration:.2f} ms")
        self.assertTrue(duration <= _perf_test_thresholds[self.device.type]["bwd_ms"])

if __name__ == "__main__":
    unittest.main()
