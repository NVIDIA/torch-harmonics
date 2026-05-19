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
from parameterized import parameterized, parameterized_class
from testutils import compare_tensors, set_seed

from torch_harmonics import ResampleS2
from torch_harmonics.quadrature import precompute_latitudes, precompute_longitudes

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


@parameterized_class(("device"), _devices)
class TestResampleS2(unittest.TestCase):
    """Tests for the ResampleS2 spherical resampling module."""

    @parameterized.expand(
        [
            [32, 64, "equiangular"],
            [32, 64, "legendre-gauss"],
            [32, 64, "lobatto"],
        ]
    )
    def test_identity(self, nlat, nlon, grid, verbose=False):
        """Identical input/output grid → skip_resampling=True and output is the same object."""
        set_seed(333)

        resample = ResampleS2(nlat, nlon, nlat, nlon, grid_in=grid, grid_out=grid).to(self.device)

        self.assertTrue(resample.skip_resampling, "skip_resampling should be True for identical grids")

        data = torch.randn(2, 3, nlat, nlon, dtype=torch.float32, device=self.device)
        out = resample(data)

        # forward() does `return x` so the same tensor object must come back
        self.assertTrue(out is data, "identity resample must return the exact same tensor object")

    @parameterized.expand(
        [
            [32, 64, 16, 32, "equiangular", "equiangular", "bilinear", 1e-5, 1e-5],
            [32, 64, 16, 32, "equiangular", "legendre-gauss", "bilinear", 1e-5, 1e-5],
            [32, 64, 16, 32, "equiangular", "lobatto", "bilinear", 1e-5, 1e-5],
            [32, 64, 16, 32, "legendre-gauss", "equiangular", "bilinear", 1e-5, 1e-5],
            [32, 64, 16, 32, "legendre-gauss", "legendre-gauss", "bilinear", 1e-5, 1e-5],
            [32, 64, 16, 32, "legendre-gauss", "lobatto", "bilinear", 1e-5, 1e-5],
            [32, 64, 16, 32, "lobatto", "equiangular", "bilinear", 1e-5, 1e-5],
            [32, 64, 16, 32, "lobatto", "legendre-gauss", "bilinear", 1e-5, 1e-5],
            [32, 64, 16, 32, "lobatto", "lobatto", "bilinear", 1e-5, 1e-5],
            [32, 64, 16, 32, "equiangular", "equiangular", "bilinear-spherical", 1e-5, 1e-5],
        ]
    )
    def test_constant_field(self, nlat_in, nlon_in, nlat_out, nlon_out, grid_in, grid_out, mode, atol, rtol, verbose=False):
        """A constant field f=1 must be reproduced exactly under any resampling."""
        set_seed(333)

        resample = ResampleS2(nlat_in, nlon_in, nlat_out, nlon_out, grid_in=grid_in, grid_out=grid_out, mode=mode).to(self.device)

        data = torch.ones(2, 3, nlat_in, nlon_in, dtype=torch.float32, device=self.device)
        out = resample(data)

        expected = torch.ones(2, 3, nlat_out, nlon_out, dtype=torch.float32, device=self.device)

        self.assertTrue(compare_tensors("constant field", out, expected, atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            # Only grid pairs where output latitudes lie strictly within the input latitude range
            # (expand_poles=False), so that bilinear interpolation is exact for linear-in-θ functions.
            [32, 64, 16, 32, "equiangular", "equiangular", 1e-5, 1e-5],
            [32, 64, 16, 32, "equiangular", "legendre-gauss", 1e-5, 1e-5],
            [32, 64, 16, 32, "legendre-gauss", "equiangular", 1e-5, 1e-5],
            [32, 64, 16, 32, "legendre-gauss", "legendre-gauss", 1e-5, 1e-5],
            [32, 64, 16, 32, "lobatto", "equiangular", 1e-5, 1e-5],
            [32, 64, 16, 32, "lobatto", "legendre-gauss", 1e-5, 1e-5],
        ]
    )
    def test_linear_latitude_exactness(self, nlat_in, nlon_in, nlat_out, nlon_out, grid_in, grid_out, atol, rtol, verbose=False):
        """Bilinear interpolation is exact for f(θ,φ)=θ (linear in latitude)."""
        set_seed(333)

        resample = ResampleS2(nlat_in, nlon_in, nlat_out, nlon_out, grid_in=grid_in, grid_out=grid_out).to(self.device)

        # self.assertFalse(resample.expand_poles,
        #                 f"expand_poles must be False for this test ({grid_in}→{grid_out}), "
        #                  f"otherwise pole extrapolation breaks linear exactness")

        lats_in, _ = precompute_latitudes(nlat_in, grid=grid_in)
        lats_out, _ = precompute_latitudes(nlat_out, grid=grid_out)

        # f(θ, φ) = θ — constant across longitude, linear in latitude
        data = lats_in.float().to(self.device).unsqueeze(-1).expand(nlat_in, nlon_in).contiguous()
        data = data.unsqueeze(0).unsqueeze(0)  # (1, 1, nlat_in, nlon_in)

        out = resample(data)

        expected = lats_out.float().to(self.device).unsqueeze(-1).expand(nlat_out, nlon_out).contiguous()
        expected = expected.unsqueeze(0).unsqueeze(0)  # (1, 1, nlat_out, nlon_out)

        # the pole value would differ if this is true
        if resample.expand_poles:
            out = out[..., 1:-1, :]
            expected = expected[..., 1:-1, :]

        self.assertTrue(compare_tensors("linear-in-theta", out, expected, atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            # Upsample in longitude (nlon_out > nlon_in) so that output nodes near φ=2π
            # require the wrap-around lon_idx_right=0 branch in _upscale_longitudes.
            # Tolerance reflects the O((Δφ)²) bilinear error for sin(φ) with Δφ=2π/32.
            [16, 32, 16, 64, "equiangular", "equiangular", 1e-2, 1e-2],
            [16, 32, 16, 64, "legendre-gauss", "legendre-gauss", 1e-2, 1e-2],
        ]
    )
    def test_longitude_periodicity(self, nlat_in, nlon_in, nlat_out, nlon_out, grid_in, grid_out, atol, rtol, verbose=True):
        """Upsampling in longitude handles the 2π→0 periodic wrap-around correctly."""
        set_seed(333)

        resample = ResampleS2(nlat_in, nlon_in, nlat_out, nlon_out, grid_in=grid_in, grid_out=grid_out).to(self.device)

        lons_in = precompute_longitudes(nlon_in)
        lons_out = precompute_longitudes(nlon_out)

        # f(θ, φ) = sin(φ) — smooth, periodic; tests that wrap-around is handled correctly
        data = torch.sin(lons_in).float().to(self.device).unsqueeze(0).expand(nlat_in, nlon_in).contiguous()
        data = data.unsqueeze(0).unsqueeze(0)  # (1, 1, nlat_in, nlon_in)

        out = resample(data)

        expected = torch.sin(lons_out).float().to(self.device).unsqueeze(0).expand(nlat_out, nlon_out).contiguous()
        expected = expected.unsqueeze(0).unsqueeze(0)  # (1, 1, nlat_out, nlon_out)

        self.assertTrue(compare_tensors("sin(phi) periodic", out, expected, atol=atol, rtol=rtol, verbose=verbose))


if __name__ == "__main__":
    unittest.main()
