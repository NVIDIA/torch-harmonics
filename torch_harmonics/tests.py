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
import numpy as np
import torch
from torch.autograd import gradcheck
from torch_harmonics import *

# try:
#     from tqdm import tqdm
# except:
#     tqdm = lambda x : x

tqdm = lambda x : x

class TestLegendrePolynomials(unittest.TestCase):

    def __init__(self, testname, tol=1e-9):
        super(TestLegendrePolynomials, self).__init__(testname)
        self.tol = tol

    def setUp(self):
        self.cml = lambda m, l : np.sqrt((2*l + 1) / 4 / np.pi) * np.sqrt(np.math.factorial(l-m) / np.math.factorial(l+m))
        self.pml = dict()

        # preparing associated Legendre Polynomials (These include the Condon-Shortley phase)
        # for reference see e.g. https://en.wikipedia.org/wiki/Associated_Legendre_polynomials
        self.pml[(0, 0)] = lambda x : np.ones_like(x)
        self.pml[(0, 1)] = lambda x : x
        self.pml[(1, 1)] = lambda x : - np.sqrt(1. - x**2)
        self.pml[(0, 2)] = lambda x : 0.5 * (3*x**2 - 1)
        self.pml[(1, 2)] = lambda x : - 3 * x * np.sqrt(1. - x**2)
        self.pml[(2, 2)] = lambda x : 3 * (1 - x**2)
        self.pml[(0, 3)] = lambda x : 0.5 * (5*x**3 - 3*x)
        self.pml[(1, 3)] = lambda x : 1.5 * (1 - 5*x**2) * np.sqrt(1. - x**2)
        self.pml[(2, 3)] = lambda x : 15 * x * (1 - x**2)
        self.pml[(3, 3)] = lambda x : -15 * np.sqrt(1. - x**2)**3

        self.lmax = self.mmax = 4

    def test_legendre(self):
        print("Testing computation of associated Legendre polynomials")
        from torch_harmonics.legendre import precompute_legpoly

        t = np.linspace(0, np.pi, 100)
        pct = precompute_legpoly(self.mmax, self.lmax, t)

        for l in range(self.lmax):
            for m in range(l+1):
                diff = pct[m, l].numpy() / self.cml(m,l) - self.pml[(m,l)](np.cos(t))
                self.assertTrue(diff.max() <= self.tol)
        print("done.")


class TestSphericalHarmonicTransform(unittest.TestCase):

    def __init__(self, testname, nlat=256, nlon=512, batch_size=32, norm="ortho", grid="legendre-gauss", tol=1e-9):
        super(TestSphericalHarmonicTransform, self).__init__(testname)  # calling the super class init varies for different python versions.  This works for 2.7
        self.norm = norm
        self.grid = grid
        self.tol = tol
        self.batch_size = batch_size
        self.nlat = nlat
        self.nlon = nlon
    
    def setUp(self):

        if torch.cuda.is_available():
            print("Running test on GPU")
            self.device = torch.device('cuda')
        else:
            print("Running test on CPU")
            self.device = torch.device('cpu')

    def test_sht(self):
        print(f"Testing real-valued SHT on {self.nlat}x{self.nlon} {self.grid} grid with {self.norm} normalization")

        testiters = [1, 2, 4, 8, 16]
        if self.grid == "equiangular":
            mmax = self.nlat // 2
        else:
            mmax = self.nlat
        lmax = mmax

        sht = RealSHT(self.nlat, self.nlon, mmax=mmax, lmax=lmax, grid=self.grid, norm=self.norm).to(self.device)
        isht = InverseRealSHT(self.nlat, self.nlon, mmax=mmax, lmax=lmax, grid=self.grid, norm=self.norm).to(self.device)

        with torch.no_grad():
            coeffs = torch.zeros(self.batch_size, lmax, mmax, device=self.device, dtype=torch.complex128)
            coeffs[:, :lmax, :mmax] = torch.randn(self.batch_size, lmax, mmax, device=self.device, dtype=torch.complex128)
            signal = isht(coeffs)
        
        # testing error accumulation
        for iter in testiters:
            with self.subTest(i = iter):
                print(f"{iter} iterations of batchsize {self.batch_size}:")

                base = signal

                for _ in tqdm(range(iter)):
                    base = isht(sht(base))
            
                err = torch.mean(torch.norm(base-signal, p='fro', dim=(-1,-2)) / torch.norm(signal, p='fro', dim=(-1,-2)) )
                print(f"final relative error: {err.item()}")
                self.assertTrue(err.item() <= self.tol)

    def test_sht_grad(self):
        print(f"Testing gradients of real-valued SHT on {self.nlat}x{self.nlon} {self.grid} grid with {self.norm} normalization")

        if self.grid == "equiangular":
            mmax = self.nlat // 2
        else:
            mmax = self.nlat
        lmax = mmax

        sht = RealSHT(self.nlat, self.nlon, mmax=mmax, lmax=lmax, grid=self.grid, norm=self.norm).to(self.device)
        isht = InverseRealSHT(self.nlat, self.nlon, mmax=mmax, lmax=lmax, grid=self.grid, norm=self.norm).to(self.device)

        with torch.no_grad():
            coeffs = torch.zeros(self.batch_size, lmax, mmax, device=self.device, dtype=torch.complex128)
            coeffs[:, :lmax, :mmax] = torch.randn(self.batch_size, lmax, mmax, device=self.device, dtype=torch.complex128)
            signal = isht(coeffs)
        
        input = torch.randn_like(signal, requires_grad=True)
        err_handle = lambda x : torch.mean(torch.norm( isht(sht(x)) - signal , p='fro', dim=(-1,-2)) / torch.norm(signal, p='fro', dim=(-1,-2)) )
        test_result = gradcheck(err_handle, input, eps=1e-6, atol=self.tol)
        self.assertTrue(test_result)


if __name__ == '__main__':
    sht_test_suite = unittest.TestSuite()

    # test computation of Legendre polynomials
    sht_test_suite.addTest(TestLegendrePolynomials('test_legendre', tol=1e-9))

    # test error growth when computing repeatedly isht(sht(x))
    sht_test_suite.addTest(TestSphericalHarmonicTransform('test_sht', norm="ortho",   grid="equiangular",    tol=1e-1))
    sht_test_suite.addTest(TestSphericalHarmonicTransform('test_sht', norm="ortho",   grid="legendre-gauss", tol=1e-9))
    sht_test_suite.addTest(TestSphericalHarmonicTransform('test_sht', norm="four-pi", grid="equiangular",    tol=1e-1))
    sht_test_suite.addTest(TestSphericalHarmonicTransform('test_sht', norm="four-pi", grid="legendre-gauss", tol=1e-9))
    sht_test_suite.addTest(TestSphericalHarmonicTransform('test_sht', norm="schmidt", grid="equiangular",    tol=1e-1))
    sht_test_suite.addTest(TestSphericalHarmonicTransform('test_sht', norm="schmidt", grid="legendre-gauss", tol=1e-9))

    # test error growth when computing repeatedly isht(sht(x))
    sht_test_suite.addTest(TestSphericalHarmonicTransform('test_sht_grad', norm="ortho",   grid="equiangular",    tol=1e-4, nlat=12, nlon=24, batch_size=2))
    sht_test_suite.addTest(TestSphericalHarmonicTransform('test_sht_grad', norm="ortho",   grid="legendre-gauss", tol=1e-4, nlat=12, nlon=24, batch_size=2))
    sht_test_suite.addTest(TestSphericalHarmonicTransform('test_sht_grad', norm="four-pi", grid="equiangular",    tol=1e-4, nlat=12, nlon=24, batch_size=2))
    sht_test_suite.addTest(TestSphericalHarmonicTransform('test_sht_grad', norm="four-pi", grid="legendre-gauss", tol=1e-4, nlat=12, nlon=24, batch_size=2))
    sht_test_suite.addTest(TestSphericalHarmonicTransform('test_sht_grad', norm="schmidt", grid="equiangular",    tol=1e-4, nlat=12, nlon=24, batch_size=2))
    sht_test_suite.addTest(TestSphericalHarmonicTransform('test_sht_grad', norm="schmidt", grid="legendre-gauss", tol=1e-4, nlat=12, nlon=24, batch_size=2))

    # run the test suite
    unittest.TextTestRunner(verbosity=2).run(sht_test_suite)