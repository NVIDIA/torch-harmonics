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

import unittest
from parameterized import parameterized, parameterized_class
import math
import torch
from torch.autograd import gradcheck
import torch_harmonics as th

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


class TestLegendrePolynomials(unittest.TestCase):

    def setUp(self):
        self.cml = lambda m, l: math.sqrt((2 * l + 1) / 4 / math.pi) * math.sqrt(math.factorial(l - m) / math.factorial(l + m))
        self.pml = dict()

        # preparing associated Legendre Polynomials (These include the Condon-Shortley phase)
        # for reference see e.g. https://en.wikipedia.org/wiki/Associated_Legendre_polynomials
        self.pml[(0, 0)] = lambda x: torch.ones_like(x)
        self.pml[(0, 1)] = lambda x: x
        self.pml[(1, 1)] = lambda x: -torch.sqrt(1.0 - x**2)
        self.pml[(0, 2)] = lambda x: 0.5 * (3 * x**2 - 1)
        self.pml[(1, 2)] = lambda x: -3 * x * torch.sqrt(1.0 - x**2)
        self.pml[(2, 2)] = lambda x: 3 * (1 - x**2)
        self.pml[(0, 3)] = lambda x: 0.5 * (5 * x**3 - 3 * x)
        self.pml[(1, 3)] = lambda x: 1.5 * (1 - 5 * x**2) * torch.sqrt(1.0 - x**2)
        self.pml[(2, 3)] = lambda x: 15 * x * (1 - x**2)
        self.pml[(3, 3)] = lambda x: -15 * torch.sqrt(1.0 - x**2) ** 3

        self.lmax = self.mmax = 4

        self.tol = 1e-9

    def test_legendre(self, verbose=False):
        if verbose:
            print("Testing computation of associated Legendre polynomials")

        t = torch.linspace(0, 1, 100, dtype=torch.float64)
        vdm = th.legendre.legpoly(self.mmax, self.lmax, t)

        for l in range(self.lmax):
            for m in range(l + 1):
                diff = vdm[m, l] / self.cml(m, l) - self.pml[(m, l)](t)
                self.assertTrue(diff.max() <= self.tol)


@parameterized_class(("device"), _devices)
class TestSphericalHarmonicTransform(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(333)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(333)

    @parameterized.expand(
        [
            # even-even
            [32, 64, 32, "ortho", "equiangular", 1e-9, False],
            [32, 64, 32, "ortho", "legendre-gauss", 1e-9, False],
            [32, 64, 32, "ortho", "lobatto", 1e-9, False],
            [32, 64, 32, "four-pi", "equiangular", 1e-9, False],
            [32, 64, 32, "four-pi", "legendre-gauss", 1e-9, False],
            [32, 64, 32, "four-pi", "lobatto", 1e-9, False],
            [32, 64, 32, "schmidt", "equiangular", 1e-9, False],
            [32, 64, 32, "schmidt", "legendre-gauss", 1e-9, False],
            [32, 64, 32, "schmidt", "lobatto", 1e-9, False],
            # odd-even
            [33, 64, 32, "ortho", "equiangular", 1e-9, False],
            [33, 64, 32, "ortho", "legendre-gauss", 1e-9, False],
            [33, 64, 32, "ortho", "lobatto", 1e-9, False],
	    [33, 64, 32, "four-pi", "equiangular", 1e-9, False],
            [33, 64, 32, "four-pi", "legendre-gauss", 1e-9, False],
            [33, 64, 32, "four-pi", "lobatto", 1e-9, False],
            [33, 64, 32, "schmidt", "equiangular", 1e-9, False],
            [33, 64, 32, "schmidt", "legendre-gauss", 1e-9, False],
            [33, 64, 32, "schmidt", "lobatto", 1e-9, False],
        ]
    )
    def test_sht(self, nlat, nlon, batch_size, norm, grid, tol, verbose):
        if verbose:
            print(f"Testing real-valued SHT on {nlat}x{nlon} {grid} grid with {norm} normalization on {self.device.type} device")

        testiters = [1, 2, 4, 8, 16]
        if grid == "equiangular":
            mmax = nlat // 2
        elif grid == "lobatto":
            mmax = nlat - 1
        else:
            mmax = nlat
        lmax = mmax

        sht = th.RealSHT(nlat, nlon, mmax=mmax, lmax=lmax, grid=grid, norm=norm).to(self.device)
        isht = th.InverseRealSHT(nlat, nlon, mmax=mmax, lmax=lmax, grid=grid, norm=norm).to(self.device)

        with torch.no_grad():
            coeffs = torch.zeros(batch_size, lmax, mmax, device=self.device, dtype=torch.complex128)
            coeffs[:, :lmax, :mmax] = torch.randn(batch_size, lmax, mmax, device=self.device, dtype=torch.complex128)
            signal = isht(coeffs)

        # testing error accumulation
        for iter in testiters:
            with self.subTest(i=iter):
                if verbose:
                    print(f"{iter} iterations of batchsize {batch_size}:")

                base = signal

                for _ in range(iter):
                    base = isht(sht(base))

                err = torch.mean(torch.norm(base - signal, p="fro", dim=(-1, -2)) / torch.norm(signal, p="fro", dim=(-1, -2)))
                if verbose:
                    print(f"final relative error: {err.item()}")
                self.assertTrue(err.item() <= tol)

    @parameterized.expand(
        [
            # even-even
            [12, 24, 2, "ortho", "equiangular", 1e-5, False],
            [12, 24, 2, "ortho", "legendre-gauss", 1e-5, False],
            [12, 24, 2, "ortho", "lobatto", 1e-5, False],
            [12, 24, 2, "four-pi", "equiangular", 1e-5, False],
            [12, 24, 2, "four-pi", "legendre-gauss", 1e-5, False],
            [12, 24, 2, "four-pi", "lobatto", 1e-5, False],
            [12, 24, 2, "schmidt", "equiangular", 1e-5, False],
            [12, 24, 2, "schmidt", "legendre-gauss", 1e-5, False],
            [12, 24, 2, "schmidt", "lobatto", 1e-5, False],
            # odd-even
            [15, 30, 2, "ortho", "equiangular", 1e-5, False],
            [15, 30, 2, "ortho", "legendre-gauss", 1e-5, False],
            [15, 30, 2, "ortho", "lobatto", 1e-5, False],
            [15, 30, 2, "four-pi", "equiangular", 1e-5, False],
            [15, 30, 2, "four-pi", "legendre-gauss", 1e-5, False],
            [15, 30, 2, "four-pi", "lobatto", 1e-5, False],
            [15, 30, 2, "schmidt", "equiangular", 1e-5, False],
            [15, 30, 2, "schmidt", "legendre-gauss", 1e-5, False],
            [15, 30, 2, "schmidt", "lobatto", 1e-5, False],
        ]
    )
    def test_sht_grads(self, nlat, nlon, batch_size, norm, grid, tol, verbose):
        if verbose:
            print(f"Testing gradients of real-valued SHT on {nlat}x{nlon} {grid} grid with {norm} normalization")

        if grid == "equiangular":
            mmax = nlat // 2
        elif grid == "lobatto":
            mmax = nlat - 1
        else:
            mmax = nlat
        lmax = mmax

        sht = th.RealSHT(nlat, nlon, mmax=mmax, lmax=lmax, grid=grid, norm=norm).to(self.device)
        isht = th.InverseRealSHT(nlat, nlon, mmax=mmax, lmax=lmax, grid=grid, norm=norm).to(self.device)

        with torch.no_grad():
            coeffs = torch.zeros(batch_size, lmax, mmax, device=self.device, dtype=torch.complex128)
            coeffs[:, :lmax, :mmax] = torch.randn(batch_size, lmax, mmax, device=self.device, dtype=torch.complex128)
            signal = isht(coeffs)

        # test the sht
        grad_input = torch.randn_like(signal, requires_grad=True)
        err_handle = lambda x: torch.mean(torch.norm(sht(x) - coeffs, p="fro", dim=(-1, -2)) / torch.norm(coeffs, p="fro", dim=(-1, -2)))
        test_result = gradcheck(err_handle, grad_input, eps=1e-6, atol=tol)
        self.assertTrue(test_result)

        # test the isht
        grad_input = torch.randn_like(coeffs, requires_grad=True)
        err_handle = lambda x: torch.mean(torch.norm(isht(x) - signal, p="fro", dim=(-1, -2)) / torch.norm(signal, p="fro", dim=(-1, -2)))
        test_result = gradcheck(err_handle, grad_input, eps=1e-6, atol=tol)
        self.assertTrue(test_result)


if __name__ == "__main__":
    unittest.main()
