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

import torch

from torch_harmonics.random_fields import GaussianRandomFieldS2

from testutils import set_seed, compare_tensors

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


@parameterized_class(("device"), _devices)
class TestGaussianRandomFieldS2(unittest.TestCase):
    """Tests for GaussianRandomFieldS2."""

    @parameterized.expand(
        [
            [16, 2.0, 3.0, None, "equiangular"],
            [16, 3.0, 2.0, None, "legendre-gauss"],
            [16, 2.0, 3.0, None, "lobatto"],
            [16, 2.0, 3.0, 1.0,  "equiangular"],
        ],
        skip_on_empty=True,
    )
    def test_reproducibility(self, nlat, alpha, tau, sigma, grid, verbose=False):
        """Fixed seed produces identical output across two calls."""
        field = GaussianRandomFieldS2(nlat, alpha=alpha, tau=tau, sigma=sigma, grid=grid).to(self.device)

        set_seed(333)
        u1 = field(4)
        set_seed(333)
        u2 = field(4)

        self.assertTrue(compare_tensors("reproducibility", u1, u2, atol=0.0, rtol=0.0, verbose=verbose))

    @parameterized.expand(
        [
            [16, 2.0, 3.0, None, "equiangular"],
            [16, 3.0, 2.0, None, "legendre-gauss"],
            [16, 2.0, 3.0, None, "lobatto"],
            [16, 2.0, 3.0, 1.0,  "equiangular"],
        ],
        skip_on_empty=True,
    )
    def test_custom_xi(self, nlat, alpha, tau, sigma, grid, verbose=False):
        """Providing xi explicitly bypasses sampling and gives deterministic output."""
        field = GaussianRandomFieldS2(nlat, alpha=alpha, tau=tau, sigma=sigma, grid=grid).to(self.device)

        lmax = field.isht.lmax
        mmax = field.isht.mmax
        set_seed(333)
        xi = torch.view_as_complex(
            torch.randn(4, lmax, mmax, 2, dtype=torch.float32, device=self.device)
        )

        u1 = field(4, xi=xi)
        u2 = field(4, xi=xi)

        self.assertTrue(compare_tensors("custom xi determinism", u1, u2, atol=0.0, rtol=0.0, verbose=verbose))

    @parameterized.expand(
        [
            [16, 2.0, 3.0, None, "equiangular"],
            [16, 3.0, 2.0, None, "legendre-gauss"],
        ],
        skip_on_empty=True,
    )
    def test_zero_xi(self, nlat, alpha, tau, sigma, grid, verbose=False):
        """xi=0 → output is zero everywhere (ISHT is linear)."""
        field = GaussianRandomFieldS2(nlat, alpha=alpha, tau=tau, sigma=sigma, grid=grid).to(self.device)

        lmax = field.isht.lmax
        mmax = field.isht.mmax
        xi = torch.zeros(4, lmax, mmax, dtype=torch.complex64, device=self.device)
        u = field(4, xi=xi)

        expected = torch.zeros_like(u)
        self.assertTrue(compare_tensors("zero xi", u, expected, atol=1e-5, rtol=1e-5, verbose=verbose))

    @parameterized.expand(
        [
            [16, 0.5],
            [16, 1.0],
        ],
        skip_on_empty=True,
    )
    def test_alpha_assertion(self, nlat, alpha, verbose=False):
        """alpha <= 1 with sigma=None must raise AssertionError."""
        with self.assertRaises(AssertionError):
            GaussianRandomFieldS2(nlat, alpha=alpha, sigma=None)

    @parameterized.expand(
        [
            [16, 2.0, 3.0, 0.5, "equiangular"],
            [16, 0.5, 3.0, 2.0, "equiangular"],  # alpha <= 1 is allowed when sigma is explicit
        ],
        skip_on_empty=True,
    )
    def test_explicit_sigma(self, nlat, alpha, tau, sigma, grid, verbose=False):
        """Providing sigma directly bypasses the alpha > 1 check and tau-based formula."""
        field = GaussianRandomFieldS2(nlat, alpha=alpha, tau=tau, sigma=sigma, grid=grid).to(self.device)

        set_seed(333)
        u = field(4)

        self.assertEqual(u.shape, (4, nlat, 2 * nlat))
        self.assertTrue(torch.isfinite(u).all(), "output contains non-finite values")


@parameterized_class(("device"), _devices)
class TestGaussianRandomFieldS2Probabilistic(unittest.TestCase):
    """Statistical tests for GaussianRandomFieldS2. All use fixed seeds for determinism."""

    @parameterized.expand(
        [
            [16, 2.0, 3.0, "equiangular",    500, 0.2],
            [16, 2.0, 3.0, "legendre-gauss", 500, 0.2],
        ]
    )
    def test_zero_mean(self, nlat, alpha, tau, grid, num_samples, atol, verbose=False):
        """Sample mean over many realizations is near zero (DC spectral coefficient is forced to 0)."""
        field = GaussianRandomFieldS2(nlat, alpha=alpha, tau=tau, grid=grid).to(self.device)

        set_seed(333)
        u = field(num_samples)  # (num_samples, nlat, 2*nlat)

        batch_mean = u.mean(dim=0)
        expected = torch.zeros_like(batch_mean)
        self.assertTrue(compare_tensors("zero mean", batch_mean, expected, atol=atol, rtol=0.0, verbose=verbose))

    @parameterized.expand(
        [
            [16, 2.0, 4.0, 3.0, "equiangular",    200],
            [16, 2.0, 4.0, 3.0, "legendre-gauss", 200],
        ]
    )
    def test_power_spectrum_ordering(self, nlat, alpha_rough, alpha_smooth, tau, grid, num_samples, verbose=False):
        """Larger alpha suppresses high-degree modes → lower total variance (smoother field)."""
        field_rough  = GaussianRandomFieldS2(nlat, alpha=alpha_rough,  tau=tau, grid=grid).to(self.device)
        field_smooth = GaussianRandomFieldS2(nlat, alpha=alpha_smooth, tau=tau, grid=grid).to(self.device)

        set_seed(333)
        u_rough  = field_rough(num_samples)
        set_seed(333)
        u_smooth = field_smooth(num_samples)

        self.assertLess(
            u_smooth.var().item(), u_rough.var().item(),
            "Larger alpha should produce a smoother (lower-variance) field",
        )


if __name__ == "__main__":
    unittest.main()
