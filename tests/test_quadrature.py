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
from parameterized import parameterized, parameterized_class
import math

import torch
import torch_harmonics as th
from torch_harmonics.quadrature import precompute_latitudes, precompute_longitudes

from testutils import set_seed, compare_tensors

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


@parameterized_class(("device"), _devices)
class TestQuadrature(unittest.TestCase):
    """Serial QuadratureS2 integration tests."""

    @parameterized.expand(
        [
            [64, 128, 2, 3, "equiangular", False, 1e-6, 1e-6],
            [64, 128, 2, 3, "equiangular", True, 1e-6, 1e-6],
            [65, 128, 1, 1, "legendre-gauss", False, 1e-6, 1e-6],
            [65, 128, 1, 1, "legendre-gauss", True, 1e-6, 1e-6],
            [65, 128, 2, 2, "lobatto", False, 1e-6, 1e-6],
            [65, 128, 2, 2, "lobatto", True, 1e-6, 1e-6],
            [64, 128, 2, 3, "equidistant", False, 1e-4, 1e-4],
            [64, 128, 2, 3, "equidistant", True, 1e-4, 1e-4],
        ]
    )
    def test_constant_integral(self, nlat, nlon, batch_size, num_chan, grid, normalize, atol, rtol, verbose=False):

        set_seed(333)

        quad = th.QuadratureS2(img_shape=(nlat, nlon), grid=grid, normalize=normalize).to(self.device)

        data = torch.ones((batch_size, num_chan, nlat, nlon), dtype=torch.float32, device=self.device)
        out = quad(data)

        expected_value = 1.0 if normalize else 4.0 * torch.pi
        expected = torch.full((batch_size, num_chan), expected_value, dtype=torch.float32, device=self.device)

        self.assertTrue(compare_tensors(f"output", out, expected, atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            # nlat, nlon, grid
            [64, 128, "equiangular"],
            [65, 128, "legendre-gauss"],
            [65, 128, "lobatto"],
            [64, 128, "equidistant"],
        ]
    )
    def test_odd_latitude_integral(self, nlat, nlon, grid, verbose=False):
        """cos(theta) is odd in cos-theta, so its integral over S^2 must be zero.

        Analytically: 2*pi * integral_{-1}^{1} t dt = 0.
        This exercises the sign and symmetry of the latitude quadrature weights.
        """
        set_seed(333)

        quad = th.QuadratureS2(img_shape=(nlat, nlon), grid=grid, normalize=False).to(self.device)

        # cos(theta) on the grid: precompute_latitudes returns colatitude angles
        # theta in [0, pi], so cos(theta) in [-1, 1]
        lats, _ = precompute_latitudes(nlat, grid=grid)
        cos_theta = torch.cos(lats).to(torch.float32).to(self.device)  # shape [nlat]
        # broadcast over batch=1, channel=1, and all longitudes
        f = cos_theta.view(1, 1, nlat, 1).expand(1, 1, nlat, nlon)

        out = quad(f)
        expected = torch.zeros(1, 1, device=self.device)

        self.assertTrue(compare_tensors("odd latitude integral", out, expected, atol=1e-5, rtol=0.0, verbose=verbose))

    @parameterized.expand(
        [
            # nlat, nlon, grid
            [64, 128, "equiangular"],
            [65, 128, "legendre-gauss"],
            [65, 128, "lobatto"],
            [64, 128, "equidistant"],
        ]
    )
    def test_polynomial_latitude_integral(self, nlat, nlon, grid, verbose=False):
        """cos^2(theta) integrates to 4*pi/3 (unnormalized) or 1/3 (normalized).

        Analytically: 2*pi * integral_{-1}^{1} t^2 dt = 2*pi * 2/3 = 4*pi/3.
        All three quadrature rules integrate quadratic polynomials in cos-theta
        exactly, so the error should be at floating-point precision.
        """
        set_seed(333)

        lats, _ = precompute_latitudes(nlat, grid=grid)
        cos2_theta = torch.cos(lats).pow(2).to(dtype=torch.float32, device=self.device)  # shape [nlat]
        f = cos2_theta.view(1, 1, nlat, 1).expand(1, 1, nlat, nlon)

        for normalize, expected_val in [(False, 4.0 * math.pi / 3.0), (True, 1.0 / 3.0)]:
            quad = th.QuadratureS2(img_shape=(nlat, nlon), grid=grid, normalize=normalize).to(self.device)
            out = quad(f)
            expected = torch.full((1, 1), expected_val, device=self.device)
            self.assertTrue(
                compare_tensors(
                    f"cos^2 integral (normalize={normalize})", out, expected,
                    atol=1e-5, rtol=1e-5,
                    verbose=verbose,
                )
            )

    @parameterized.expand(
        [
            # nlat, nlon, grid
            [64, 128, "equiangular"],
            [65, 128, "legendre-gauss"],
            [65, 128, "lobatto"],
            [64, 128, "equidistant"],
        ]
    )
    def test_zero_longitude_mean(self, nlat, nlon, grid, verbose=False):
        """cos(phi) integrates to zero over S^2.

        Analytically: integral_{0}^{2*pi} cos(phi) dphi = 0 (over a full period),
        independent of the latitude weights.  This tests the uniform longitude
        discretization and the dlambda = 2*pi/nlon prefactor.
        """
        set_seed(333)

        quad = th.QuadratureS2(img_shape=(nlat, nlon), grid=grid, normalize=False).to(self.device)

        lons = precompute_longitudes(nlon).to(self.device)  # shape [nlon], in [0, 2*pi)
        cos_phi = torch.cos(lons).to(torch.float32)
        f = cos_phi.view(1, 1, 1, nlon).expand(1, 1, nlat, nlon)

        out = quad(f)
        expected = torch.zeros(1, 1, device=self.device)

        self.assertTrue(compare_tensors("zero longitude mean", out, expected, atol=1e-5, rtol=0.0, verbose=verbose))

    @parameterized.expand(
        [
            # nlat, nlon, batch, channels, grid
            [64, 128, 2, 3, "equiangular"],
            [65, 128, 1, 1, "legendre-gauss"],
        ]
    )
    def test_normalization_consistency(self, nlat, nlon, batch_size, num_chan, grid, verbose=False):
        """normalize=True must equal normalize=False divided by 4*pi.

        Tests the 4*pi divisor branch independently of which function is integrated.
        """
        set_seed(333)

        quad_raw  = th.QuadratureS2(img_shape=(nlat, nlon), grid=grid, normalize=False).to(self.device)
        quad_norm = th.QuadratureS2(img_shape=(nlat, nlon), grid=grid, normalize=True).to(self.device)

        data = torch.randn(batch_size, num_chan, nlat, nlon, device=self.device)

        out_raw  = quad_raw(data)
        out_norm = quad_norm(data)

        self.assertTrue(
            compare_tensors(
                "normalization consistency",
                out_norm, out_raw / (4.0 * math.pi),
                atol=1e-6, rtol=1e-5,
                verbose=verbose,
            )
        )


if __name__ == "__main__":
    unittest.main()

