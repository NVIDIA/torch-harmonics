# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import torch
from testutils import compare_tensors

import torch_harmonics as th


class TestSHTTruncation(unittest.TestCase):
    def test_default_is_triangular(self):
        self.assertEqual(th.truncate_sht(16, 16, grid="legendre-gauss"), (9, 9))
        self.assertEqual(th.truncate_sht(32, 16, grid="legendre-gauss"), (9, 9))

    def test_explicit_triangular(self):
        self.assertEqual(th.truncate_sht(16, 32, lmax=12, mmax=5, grid="legendre-gauss"), (5, 5))
        self.assertEqual(th.truncate_sht(32, 16, lmax=24, grid="legendre-gauss"), (9, 9))
        self.assertEqual(th.truncate_sht(32, 16, mmax=12, grid="legendre-gauss"), (12, 12))
        self.assertEqual(th.truncate_sht(16, 32, lmax=12, mmax=5, grid="legendre-gauss", truncation="triangular"), (5, 5))

        for cls in (th.RealSHT, th.InverseRealSHT, th.RealVectorSHT, th.InverseRealVectorSHT):
            default = cls(16, 32, lmax=12, mmax=5, grid="legendre-gauss")
            explicit = cls(16, 32, lmax=12, mmax=5, grid="legendre-gauss", truncation="triangular")
            self.assertEqual((default.lmax, default.mmax), (5, 5))
            self.assertEqual((explicit.lmax, explicit.mmax), (5, 5))

    def test_trapezoidal(self):
        cases = (
            ((32, 16), {}, (32, 9)),
            ((32, 16), {"lmax": 24}, (24, 9)),
            ((32, 16), {"mmax": 12}, (32, 12)),
            ((16, 32), {"lmax": 12, "mmax": 5}, (12, 5)),
            ((16, 32), {"lmax": 5, "mmax": 12}, (5, 5)),
        )
        for (nlat, nlon), limits, expected in cases:
            with self.subTest(nlat=nlat, nlon=nlon, limits=limits):
                self.assertEqual(
                    th.truncate_sht(nlat, nlon, grid="legendre-gauss", truncation="trapezoidal", **limits),
                    expected,
                )

    def test_invalid_truncation(self):
        with self.assertRaisesRegex(ValueError, "triangular.*trapezoidal"):
            th.truncate_sht(16, 32, truncation="invalid")

    def test_scalar_trapezoidal_round_trip(self, verbose=False):
        nlat, nlon = 16, 32
        lmax, mmax = 12, 5

        sht = th.RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss", truncation="trapezoidal")
        isht = th.InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss", truncation="trapezoidal")

        self.assertEqual((sht.lmax, sht.mmax), (lmax, mmax))
        self.assertEqual((isht.lmax, isht.mmax), (lmax, mmax))

        coeffs = torch.zeros(1, lmax, mmax, dtype=torch.complex128)
        coeffs[0, 11, 4] = 1.0 + 0.25j

        with torch.no_grad():
            actual = sht(isht(coeffs))

        self.assertTrue(compare_tensors("scalar round trip", coeffs, actual, atol=1e-9, rtol=1e-9, verbose=verbose))

    def test_vector_trapezoidal_round_trip(self, verbose=False):
        nlat, nlon = 16, 32
        lmax, mmax = 12, 5

        sht = th.RealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss", truncation="trapezoidal")
        isht = th.InverseRealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss", truncation="trapezoidal")

        self.assertEqual((sht.lmax, sht.mmax), (lmax, mmax))
        self.assertEqual((isht.lmax, isht.mmax), (lmax, mmax))

        coeffs = torch.zeros(1, 2, lmax, mmax, dtype=torch.complex128)
        coeffs[0, 0, 11, 4] = 1.0 + 0.25j
        coeffs[0, 1, 10, 4] = -0.5 + 0.75j

        with torch.no_grad():
            actual = sht(isht(coeffs))

        self.assertTrue(compare_tensors("vector round trip", coeffs, actual, atol=1e-7, rtol=1e-7, verbose=verbose))


if __name__ == "__main__":
    unittest.main()
