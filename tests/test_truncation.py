# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import torch

import torch_harmonics as th


class TestSHTTruncation(unittest.TestCase):
    def test_defaults_are_triangular(self):
        self.assertEqual(th.truncate_sht(16, 16, grid="legendre-gauss"), (9, 9))

    def test_single_explicit_limit_is_triangular(self):
        self.assertEqual(th.truncate_sht(16, 16, lmax=12, grid="legendre-gauss"), (9, 9))
        self.assertEqual(th.truncate_sht(16, 32, mmax=5, grid="legendre-gauss"), (5, 5))

    def test_explicit_limits_are_independent(self):
        self.assertEqual(th.truncate_sht(16, 32, lmax=12, mmax=5, grid="legendre-gauss"), (12, 5))

    def test_mmax_is_capped_at_lmax(self):
        self.assertEqual(th.truncate_sht(16, 32, lmax=5, mmax=12, grid="legendre-gauss"), (5, 5))


class TestRectangularSHT(unittest.TestCase):
    def test_scalar_round_trip(self):
        nlat, nlon = 16, 32
        lmax, mmax = 12, 5

        sht = th.RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss")
        isht = th.InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss")

        self.assertEqual((sht.lmax, sht.mmax), (lmax, mmax))
        self.assertEqual((isht.lmax, isht.mmax), (lmax, mmax))

        coeffs = torch.zeros(1, lmax, mmax, dtype=torch.complex128)
        coeffs[0, 10, 4] = 1.0 + 0.25j

        with torch.no_grad():
            actual = sht(isht(coeffs))

        torch.testing.assert_close(actual, coeffs, atol=1e-9, rtol=1e-9)

    def test_vector_round_trip(self):
        nlat, nlon = 16, 32
        lmax, mmax = 12, 5

        sht = th.RealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss")
        isht = th.InverseRealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss")

        self.assertEqual((sht.lmax, sht.mmax), (lmax, mmax))
        self.assertEqual((isht.lmax, isht.mmax), (lmax, mmax))

        coeffs = torch.zeros(1, 2, lmax, mmax, dtype=torch.complex128)
        coeffs[0, 0, 10, 4] = 1.0 + 0.25j
        coeffs[0, 1, 9, 3] = -0.5 + 0.75j

        with torch.no_grad():
            actual = sht(isht(coeffs))

        torch.testing.assert_close(actual, coeffs, atol=1e-9, rtol=1e-9)


if __name__ == "__main__":
    unittest.main()
