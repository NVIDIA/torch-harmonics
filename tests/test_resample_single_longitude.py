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
#

import unittest

import torch
from parameterized import parameterized
from testutils import compare_tensors

from torch_harmonics import ResampleS2
from torch_harmonics.distributed import DistributedResampleS2


class TestSingleLongitudeResampling(unittest.TestCase):
    """A single periodic sample defines a longitude-independent field."""

    @parameterized.expand(
        [
            (layer, mode, dtype, nlon_out)
            for layer in (ResampleS2, DistributedResampleS2)
            for mode in ("bilinear", "bilinear-spherical")
            for dtype in (torch.float32, torch.float64)
            for nlon_out in (1, 7)
        ]
    )
    def test_values_and_gradients(self, layer, mode, dtype, nlon_out, verbose=False):
        # Three equiangular latitudes interpolate to five with exact half weights.
        # DistributedResampleS2 exercises its single-rank path without collectives.
        resample = layer(3, 1, 5, nlon_out, mode=mode).to(dtype=dtype)
        data = (torch.arange(12, dtype=dtype).reshape(2, 2, 3, 1) / 20).requires_grad_()
        reference_data = data.detach().clone().requires_grad_()
        latitude_map = torch.tensor([[1, 0, 0], [0.5, 0.5, 0], [0, 1, 0], [0, 0.5, 0.5], [0, 0, 1]], dtype=dtype)
        expected = torch.einsum("oi,bcik->bcok", latitude_map, reference_data).expand(2, 2, 5, nlon_out)
        actual = resample(data)
        cotangent = torch.linspace(0.1, 1.0, actual.numel(), dtype=dtype).reshape(actual.shape)
        actual.backward(cotangent)
        expected.backward(cotangent)
        self.assertTrue(compare_tensors("axisymmetric field", expected, actual, atol=1e-6, rtol=1e-5, verbose=verbose))
        self.assertTrue(compare_tensors("input gradient", reference_data.grad, data.grad, atol=1e-6, rtol=1e-5, verbose=verbose))
        self.assertTrue(torch.isfinite(resample.lon_weights).all())

    @parameterized.expand([(mode,) for mode in ("bilinear", "bilinear-spherical")])
    def test_compiled_longitude_broadcast(self, mode, verbose=False):
        resample = ResampleS2(4, 1, 4, 8, mode=mode).double()
        compiled = torch.compile(resample, backend="aot_eager", fullgraph=True)
        data = torch.linspace(-0.5, 0.5, 4, dtype=torch.float64).reshape(1, 1, 4, 1).requires_grad_()
        output = compiled(data)
        expected = data.detach().expand(1, 1, 4, 8)
        output.sum().backward()
        self.assertTrue(compare_tensors("compiled output", expected, output, atol=1e-12, rtol=1e-12, verbose=verbose))
        self.assertTrue(compare_tensors("compiled gradient", torch.full_like(data, 8), data.grad, atol=1e-12, rtol=1e-12, verbose=verbose))


if __name__ == "__main__":
    unittest.main()
