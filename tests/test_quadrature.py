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

import math
import unittest

import torch
from parameterized import parameterized, parameterized_class
from testutils import compare_tensors, set_seed

import torch_harmonics as th
from torch_harmonics.quadrature import geometric_weights, precompute_latitudes, precompute_longitudes, trapezoidal_weights


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
            [64, 128, 2, 3, "equiangular-trapezoidal", False, 1e-6, 1e-6],
            [64, 128, 2, 3, "equiangular-trapezoidal", True, 1e-6, 1e-6],
        ]
    )
    def test_constant_integral(self, nlat, nlon, batch_size, num_chan, grid, normalize, atol, rtol, verbose=False):

        set_seed(333)

        quad = th.QuadratureS2(img_shape=(nlat, nlon), grid=grid, normalize=normalize).to(self.device)

        data = torch.ones((batch_size, num_chan, nlat, nlon), dtype=torch.float32, device=self.device)
        out = quad(data)

        expected_value = 1.0 if normalize else 4.0 * torch.pi
        expected = torch.full((batch_size, num_chan), expected_value, dtype=torch.float32, device=self.device)

        self.assertTrue(compare_tensors("output", out, expected, atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            # nlat, nlon, grid
            [64, 128, "equiangular"],
            [65, 128, "legendre-gauss"],
            [65, 128, "lobatto"],
            [64, 128, "equiangular-trapezoidal"],
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
            # nlat, nlon, grid, atol, rtol
            [64, 128, "equiangular", 1e-5, 1e-5],
            [65, 128, "legendre-gauss", 1e-5, 1e-5],
            [65, 128, "lobatto", 1e-5, 1e-5],
            [64, 128, "equiangular-trapezoidal", 1e-2, 1e-2],
        ]
    )
    def test_polynomial_latitude_integral(self, nlat, nlon, grid, atol, rtol, verbose=False):
        """cos^2(theta) integrates to 4*pi/3 (unnormalized) or 1/3 (normalized).

        Analytically: 2*pi * integral_{-1}^{1} t^2 dt = 2*pi * 2/3 = 4*pi/3.
        Gauss-type quadrature rules integrate quadratic polynomials in cos-theta
        exactly; the trapezoidal rule (equiangular-trapezoidal) has O(h^2) error.
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
                    f"cos^2 integral (normalize={normalize})",
                    out,
                    expected,
                    atol=atol,
                    rtol=rtol,
                    verbose=verbose,
                )
            )

    @parameterized.expand(
        [
            # nlat, nlon, grid
            [64, 128, "equiangular"],
            [65, 128, "legendre-gauss"],
            [65, 128, "lobatto"],
            [64, 128, "equiangular-trapezoidal"],
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

        quad_raw = th.QuadratureS2(img_shape=(nlat, nlon), grid=grid, normalize=False).to(self.device)
        quad_norm = th.QuadratureS2(img_shape=(nlat, nlon), grid=grid, normalize=True).to(self.device)

        data = torch.randn(batch_size, num_chan, nlat, nlon, device=self.device)

        out_raw = quad_raw(data)
        out_norm = quad_norm(data)

        self.assertTrue(
            compare_tensors(
                "normalization consistency",
                out_norm,
                out_raw / (4.0 * math.pi),
                atol=1e-6,
                rtol=1e-5,
                verbose=verbose,
            )
        )


class TestGeometricWeights(unittest.TestCase):
    """Geometrically spaced quadrature nodes and weights on a positive interval."""

    @parameterized.expand(
        [
            # n, a, b
            [8, 1e-3, 1e3],
            [64, 1e-3, 1e3],
            [512, 1e-3, 1e3],
            [33, 1.0, 2.0],
            [65, 1e-6, 1.0],
        ]
    )
    def test_inverse_integral(self, n, a, b, verbose=False):
        """The rule is exact for f(x) = 1/x, whose integral over [a, b] is log(b / a).

        The nodes are equispaced in t = log(x), so f dx/dt = 1 is constant and the
        trapezoidal rule integrates it without discretization error at any n. This
        pins down both the node placement and the dx/dt Jacobian carried by the
        weights: dropping it turns the result into something n-dependent.
        """

        x, w = geometric_weights(n, a, b)

        integral = (w / x).sum()
        expected = torch.as_tensor(math.log(b / a), dtype=integral.dtype)

        self.assertTrue(compare_tensors("inverse integral", integral, expected, atol=1e-6, rtol=1e-6, verbose=verbose))

    @parameterized.expand(
        [
            # n, a, b
            [16, 1e-3, 1e3],
            [65, 1e-2, 1.0],
        ]
    )
    def test_node_placement(self, n, a, b, verbose=False):
        """Nodes span [a, b] and are geometrically spaced, i.e. a constant ratio apart."""

        x, w = geometric_weights(n, a, b)

        self.assertEqual(x.shape, (n,))
        self.assertEqual(w.shape, (n,))
        self.assertTrue(compare_tensors("endpoints", x[[0, -1]], torch.as_tensor([a, b], dtype=x.dtype), atol=1e-12, rtol=1e-12, verbose=verbose))

        ratio = x[1:] / x[:-1]
        expected = torch.full_like(ratio, (b / a) ** (1.0 / (n - 1)))
        self.assertTrue(compare_tensors("node ratio", ratio, expected, atol=1e-12, rtol=1e-12, verbose=verbose))
        self.assertTrue(torch.all(w > 0.0))

    def test_convergence(self, verbose=False):
        """For f(x) = 1, which is not exact, the error decays at second order in h = log(b / a) / (n - 1)."""

        a, b = 1.0, 10.0
        errors = [abs(geometric_weights(n, a, b)[1].sum().item() - (b - a)) for n in (64, 128, 256)]

        for coarse, fine in zip(errors[:-1], errors[1:]):
            self.assertGreater(coarse / fine, 3.5)

    def test_invalid_bounds(self):
        """A geometric grid is undefined for a non-positive lower bound.

        Matched on the message rather than the type: math.log raises ValueError for
        these inputs by itself, so a bare assertRaises would also pass if the explicit
        bound check were removed.
        """

        for a in (0.0, -1.0):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                geometric_weights(8, a, 10.0)
class TestQuadratureWeightPrecision(unittest.TestCase):
    """Every quadrature rule must carry its weights in the same precision as its nodes."""

    @parameterized.expand(
        [
            ["equiangular"],
            ["legendre-gauss"],
            ["lobatto"],
            ["equiangular-trapezoidal"],
        ]
    )
    def test_latitude_weight_dtype(self, grid):
        """Nodes and weights come back in float64, whichever rule produced them."""

        lats, weights = precompute_latitudes(64, grid=grid)

        self.assertEqual(lats.dtype, torch.float64)
        self.assertEqual(weights.dtype, torch.float64)

    @parameterized.expand(
        [
            # n, a, b, periodic
            [64, -1.0, 1.0, False],
            [64, 0.0, 2.0, True],
            [33, -1.0, 1.0, False],
        ]
    )
    def test_trapezoidal_weight_values(self, n, a, b, periodic, verbose=False):
        """Weights match the analytic rule to double precision.

        The uniform weight (b - a) / (n - 1) is not exactly representable in binary, so
        computing it in float32 leaves a relative error around 1e-8. Comparing against a
        float64 reference at 1e-15 is what distinguishes the two.
        """

        x, w = trapezoidal_weights(n, a, b, periodic=periodic)

        h = (b - a) / (n - 1 + periodic * 1)
        expected = torch.full((n,), h, dtype=torch.float64)
        if not periodic:
            expected[0] *= 0.5
            expected[-1] *= 0.5

        self.assertEqual(w.dtype, x.dtype)
        self.assertTrue(compare_tensors(f"trapezoidal weights (periodic={periodic})", w, expected, atol=1e-15, rtol=1e-15, verbose=verbose))


if __name__ == "__main__":
    unittest.main()
