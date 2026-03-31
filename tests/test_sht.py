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
from torch_harmonics.quadrature import precompute_latitudes

from testutils import disable_tf32, set_seed, compare_tensors

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


def random_sht_coeffs(batch_size, lmax, mmax, device, zero_l0=False):
    """Random scalar SHT coefficients with proper structure:
    m=0 column real (real-valued field), triangular support (m <= l),
    and optionally l=0 row zeroed (needed when testing gradient/curl)."""
    c = torch.randn(batch_size, lmax, mmax, dtype=torch.complex128, device=device)
    c[:, :, 0] = c[:, :, 0].real
    for l in range(lmax):
        if l + 1 < mmax:
            c[:, l, l + 1:] = 0.0
    if zero_l0:
        c[:, 0, :] = 0.0
    return c


class TestLegendrePolynomials(unittest.TestCase):
    """Test the associated Legendre polynomials (CPU/CUDA if available)."""
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
    """Test the spherical harmonic transform (CPU/CUDA if available)."""

    @parameterized.expand(
        [
            # even-even
            [32, 64, 32, "ortho", "equiangular", 1e-9, 1e-9],
            [32, 64, 32, "ortho", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, 32, "ortho", "lobatto", 1e-9, 1e-9],
            [32, 64, 32, "four-pi", "equiangular", 1e-9, 1e-9],
            [32, 64, 32, "four-pi", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, 32, "four-pi", "lobatto", 1e-9, 1e-9],
            [32, 64, 32, "schmidt", "equiangular", 1e-9, 1e-9],
            [32, 64, 32, "schmidt", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, 32, "schmidt", "lobatto", 1e-9, 1e-9],
            # odd-even
            [33, 64, 32, "ortho", "equiangular", 1e-9, 1e-9],
            [33, 64, 32, "ortho", "legendre-gauss", 1e-9, 1e-9],
            [33, 64, 32, "ortho", "lobatto", 1e-9, 1e-9],
            [33, 64, 32, "four-pi", "equiangular", 1e-9, 1e-9],
            [33, 64, 32, "four-pi", "legendre-gauss", 1e-9, 1e-9],
            [33, 64, 32, "four-pi", "lobatto", 1e-9, 1e-9],
            [33, 64, 32, "schmidt", "equiangular", 1e-9, 1e-9],
            [33, 64, 32, "schmidt", "legendre-gauss", 1e-9, 1e-9],
            [33, 64, 32, "schmidt", "lobatto", 1e-9, 1e-9],
        ],
        skip_on_empty=True,
    )
    def test_forward_inverse(self, nlat, nlon, batch_size, norm, grid, atol, rtol, verbose=False):
        if verbose:
            print(f"Testing real-valued SHT on {nlat}x{nlon} {grid} grid with {norm} normalization on {self.device.type} device")

        # disable tf32
        disable_tf32()

        # set seed
        set_seed(333)

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

                self.assertTrue(compare_tensors(f"output iteration {iter}", base, signal, atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            # even-even
            [12, 24, 2, "ortho", "equiangular", 1e-5, 1e-5],
            [12, 24, 2, "ortho", "legendre-gauss", 1e-5, 1e-5],
            [12, 24, 2, "ortho", "lobatto", 1e-5, 1e-5],
            [12, 24, 2, "four-pi", "equiangular", 1e-5, 1e-5],
            [12, 24, 2, "four-pi", "legendre-gauss", 1e-5, 1e-5],
            [12, 24, 2, "four-pi", "lobatto", 1e-5, 1e-5],
            [12, 24, 2, "schmidt", "equiangular", 1e-5, 1e-5],
            [12, 24, 2, "schmidt", "legendre-gauss", 1e-5, 1e-5],
            [12, 24, 2, "schmidt", "lobatto", 1e-5, 1e-5],
            # odd-even
            [15, 30, 2, "ortho", "equiangular", 1e-5, 1e-5],
            [15, 30, 2, "ortho", "legendre-gauss", 1e-5, 1e-5],
            [15, 30, 2, "ortho", "lobatto", 1e-5, 1e-5],
            [15, 30, 2, "four-pi", "equiangular", 1e-5, 1e-5],
            [15, 30, 2, "four-pi", "legendre-gauss", 1e-5, 1e-5],
            [15, 30, 2, "four-pi", "lobatto", 1e-5, 1e-5],
            [15, 30, 2, "schmidt", "equiangular", 1e-5, 1e-5],
            [15, 30, 2, "schmidt", "legendre-gauss", 1e-5, 1e-5],
            [15, 30, 2, "schmidt", "lobatto", 1e-5, 1e-5],
        ],
        skip_on_empty=True,
    )
    def test_grads(self, nlat, nlon, batch_size, norm, grid, atol, rtol, verbose=False):
        if verbose:
            print(f"Testing gradients of real-valued SHT on {nlat}x{nlon} {grid} grid with {norm} normalization")

        # disable tf32
        disable_tf32()

        # set seed
        set_seed(333)

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
        test_result = gradcheck(err_handle, grad_input, eps=1e-6, atol=atol, rtol=rtol)
        self.assertTrue(test_result)

        # test the isht
        grad_input = torch.randn_like(coeffs, requires_grad=True)
        err_handle = lambda x: torch.mean(torch.norm(isht(x) - signal, p="fro", dim=(-1, -2)) / torch.norm(signal, p="fro", dim=(-1, -2)))
        test_result = gradcheck(err_handle, grad_input, eps=1e-6, atol=atol, rtol=rtol)
        self.assertTrue(test_result)

    @parameterized.expand(
        [
            [32, 64, 32, "ortho",   "equiangular",   1e-9, 1e-9],
            [32, 64, 32, "ortho",   "legendre-gauss", 1e-9, 1e-9],
            [32, 64, 32, "ortho",   "lobatto",        1e-9, 1e-9],
            [32, 64, 32, "four-pi", "equiangular",   1e-9, 1e-9],
            [32, 64, 32, "four-pi", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, 32, "four-pi", "lobatto",        1e-9, 1e-9],
            [32, 64, 32, "schmidt", "equiangular",   1e-9, 1e-9],
            [32, 64, 32, "schmidt", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, 32, "schmidt", "lobatto",        1e-9, 1e-9],
        ],
        skip_on_empty=True,
    )
    def test_parseval(self, nlat, nlon, batch_size, norm, grid, atol, rtol, verbose=False):
        """Parseval's theorem: the spatial L2 norm of isht(c) equals a weighted spectral norm
        of c.  The spectral weights W_{l,m} depend on the normalization convention:

          ortho:   W_{l,m} = w_m                        (w_m = 1 for m=0, 2 for m>0)
          four-pi: W_{l,m} = w_m / (4*pi)               (coefficients are sqrt(4*pi) larger)
          schmidt: W_{l,m} = w_m * (2*l+1) / (4*pi)     (coefficients are sqrt(4*pi/(2*l+1)) larger)

        In all cases: ||f||^2_{S^2} = sum_{l,m} W_{l,m} * |c_{l,m}|^2
        """
        if verbose:
            print(f"Testing Parseval's theorem on {nlat}x{nlon} {grid} grid with {norm} normalization on {self.device.type}")

        # disable tf32
        disable_tf32()

        # set seed
        set_seed(333)

        isht = th.InverseRealSHT(nlat, nlon, grid=grid, norm=norm).to(self.device)
        lmax = isht.lmax
        mmax = isht.mmax

        with torch.no_grad():
            c = random_sht_coeffs(batch_size, lmax, mmax, self.device)
            f = isht(c)  # (batch, nlat, nlon)

        # Spatial L2 norm via spherical quadrature: integral of f^2 over S^2
        _, w_lat = precompute_latitudes(nlat, grid=grid)
        w_lat = w_lat.to(device=self.device, dtype=torch.float64)
        dlon = 2.0 * math.pi / nlon
        spatial_norm_sq = torch.einsum("bnl,n->b", f ** 2, w_lat) * dlon  # (batch,)

        # Build the (lmax, mmax) spectral weight matrix W_{l,m}.
        # w_m accounts for the ±m folding in the real irfft (m=0: weight 1, m>0: weight 2).
        w_m = torch.ones(mmax, dtype=torch.float64, device=self.device)
        w_m[1:] = 2.0

        if norm == "ortho":
            # c_lm^ortho are the orthonormal coefficients; W_{l,m} = w_m
            W = w_m.unsqueeze(0).expand(lmax, mmax)
        elif norm == "four-pi":
            # c_lm^{four-pi} = sqrt(4*pi) * c_lm^ortho  =>  W_{l,m} = w_m / (4*pi)
            W = w_m.unsqueeze(0).expand(lmax, mmax) / (4.0 * math.pi)
        elif norm == "schmidt":
            # c_lm^{schmidt} = sqrt(4*pi / (2*l+1)) * c_lm^ortho  =>  W_{l,m} = w_m * (2*l+1) / (4*pi)
            l_vals = torch.arange(lmax, dtype=torch.float64, device=self.device)
            W = torch.outer(2.0 * l_vals + 1.0, w_m) / (4.0 * math.pi)

        spectral_norm_sq = torch.einsum("blm,lm->b", c.abs() ** 2, W)  # (batch,)

        self.assertTrue(compare_tensors("Parseval's theorem", spatial_norm_sq, spectral_norm_sq, atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            # even-even
            [12, 24, "ortho", "equiangular", 1e-5, 1e-5],
            [12, 24, "ortho", "legendre-gauss", 1e-5, 1e-5],
            [12, 24, "ortho", "lobatto", 1e-5, 1e-5],
        ],
        skip_on_empty=True,
    )
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_device_instantiation(self, nlat, nlon, norm, grid, atol, rtol, verbose=False):
        if verbose:
            print(f"Testing device instantiation of real-valued SHT on {nlat}x{nlon} {grid} grid with {norm} normalization")

        # disable tf32
        disable_tf32()

        # set seed
        set_seed(333)

        # init on cpu
        sht_host = th.RealSHT(nlat, nlon, grid=grid, norm=norm)
        isht_host = th.InverseRealSHT(nlat, nlon, grid=grid, norm=norm)

        # init on device
        with torch.device(self.device):
            sht_device = th.RealSHT(nlat, nlon, grid=grid, norm=norm)
            isht_device = th.InverseRealSHT(nlat, nlon, grid=grid, norm=norm)

        self.assertTrue(compare_tensors(f"sht weights", sht_host.weights.cpu(), sht_device.weights.cpu(), atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors(f"isht weights", isht_host.pct.cpu(), isht_device.pct.cpu(), atol=atol, rtol=rtol, verbose=verbose))


@parameterized_class(("device"), _devices)
class TestSphericalHarmonicsY(unittest.TestCase):
    """Test fundamental properties of the real spherical harmonic basis functions.

    InverseRealSHT with norm="ortho" synthesizes orthonormal basis functions on
    the sphere.  Setting a single complex coefficient c_{l,m} = 1 synthesizes

      f_{l,0}      = Y_l^0(theta, phi)                   for m = 0
      f_{l,m,cos}  ~ P_l^m(cos theta) * cos(m*phi)       for m > 0  (c_{l,m} = 1+0j)
      f_{l,m,sin}  ~ P_l^m(cos theta) * sin(m*phi)       for m > 0  (c_{l,m} = 0+1j)

    With ortho normalization the sphere inner products satisfy:
      <f_{l,0},     f_{l',0}    > = delta_{ll'}
      <f_{l,m,cos}, f_{l',m',*}> = 2 * delta_{ll'} * delta_{mm'}   for m, m' > 0
      <f_{l,m,sin}, f_{l',m',*}> = 2 * delta_{ll'} * delta_{mm'}   for m, m' > 0

    The factor of 2 for m > 0 arises because the real irfft folds the +m and -m
    modes together, doubling the amplitude of each mode.
    """

    @parameterized.expand(
        [
            [12, 24, "legendre-gauss", 1e-9, 1e-9],
            [12, 24, "equiangular",    1e-9, 1e-9],
            [12, 24, "lobatto",        1e-9, 1e-9],
        ],
        skip_on_empty=True,
    )
    def test_orthogonality(self, nlat, nlon, grid, atol, rtol, verbose=False):
        """Verify that isht(norm="ortho") synthesizes mutually orthogonal basis
        functions and that the self inner-products equal 1 (m=0) or 2 (m>0)."""
        if verbose:
            print(f"Testing Y_lm orthogonality on {nlat}x{nlon} {grid} grid on {self.device.type}")

        # disable tf32
        disable_tf32()

        # set seed
        set_seed(333)

        if grid == "equiangular":
            lmax = mmax = nlat // 2
        elif grid == "lobatto":
            lmax = mmax = nlat - 1
        else:
            lmax = mmax = nlat

        isht = th.InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, norm="ortho").to(self.device)

        # Build one coefficient tensor per real basis function.
        # For m = 0: one tensor with c[l, 0] = 1+0j (real mode only).
        # For m > 0: two tensors — c[l, m] = 1+0j (cos) and c[l, m] = 0+1j (sin).
        basis_list = []
        expected_diag = []
        for l in range(lmax):
            for m in range(min(l + 1, mmax)):
                c = torch.zeros(lmax, mmax, dtype=torch.complex128)
                c[l, m] = 1.0 + 0.0j
                basis_list.append(c)
                expected_diag.append(1.0 if m == 0 else 2.0)
                if m > 0:
                    c = torch.zeros(lmax, mmax, dtype=torch.complex128)
                    c[l, m] = 0.0 + 1.0j
                    basis_list.append(c)
                    expected_diag.append(2.0)

        coeffs = torch.stack(basis_list).to(self.device)  # (N, lmax, mmax)

        with torch.no_grad():
            funcs = isht(coeffs)  # (N, nlat, nlon), real-valued

        # Gram matrix via spherical quadrature: G[i,j] = integral of f_i * f_j over S^2
        # Weights from precompute_latitudes are in the cos(theta) domain and integrate
        # over [-1, 1], so the full measure is w_lat[k] * dlon.
        _, w_lat = precompute_latitudes(nlat, grid=grid)
        w_lat = w_lat.to(device=self.device, dtype=torch.float64)
        dlon = 2.0 * math.pi / nlon

        weighted = funcs * (dlon * w_lat).unsqueeze(-1)  # (N, nlat, nlon)
        gram = torch.einsum("inl,jnl->ij", weighted, funcs)  # (N, N)

        expected = torch.diag(
            torch.tensor(expected_diag, dtype=torch.float64, device=self.device)
        )
        self.assertTrue(compare_tensors("Gram matrix", gram, expected, atol=atol, rtol=rtol, verbose=verbose))


@parameterized_class(("device"), _devices)
class TestVectorSphericalHarmonicTransform(unittest.TestCase):
    """Tests for the consistency between the scalar SHT and the vector SHT.

    RealVectorSHT includes a 1/(l*(l+1)) normalization in its quadrature weights
    so that the spheroidal/toroidal spectral coefficients relate directly to the
    scalar SHT coefficients of the generating potential:

      Gradient:  vsht(ivsht([c, 0]))[spheroidal] = c,  [toroidal] = 0  (l > 0)
      Curl:      vsht(ivsht([0, c]))[spheroidal] = 0,  [toroidal] = c  (l > 0)

    The l = 0 mode is zero in both vsht and ivsht because the gradient and curl
    of a constant field (Y_0^0) vanish identically on the sphere.

    These tests catch swapped spheroidal/toroidal channels, wrong signs in the
    dP/dtheta or P/sin(theta) terms, and incorrect l*(l+1) normalization.
    """

    @parameterized.expand(
        [
            [32, 64, 16, "ortho",   "legendre-gauss", 1e-7, 1e-7],
            [32, 64, 16, "ortho",   "equiangular",    1e-7, 1e-7],
            [32, 64, 16, "ortho",   "lobatto",        1e-7, 1e-7],
            [32, 64, 16, "four-pi", "legendre-gauss", 1e-7, 1e-7],
            [32, 64, 16, "four-pi", "equiangular",    1e-7, 1e-7],
            [32, 64, 16, "four-pi", "lobatto",        1e-7, 1e-7],
            # [32, 64, 16, "schmidt", "legendre-gauss", 1e-7, 1e-7],
            # [32, 64, 16, "schmidt", "equiangular",    1e-7, 1e-7],
            # [32, 64, 16, "schmidt", "lobatto",        1e-7, 1e-7],
        ],
        skip_on_empty=True,
    )
    def test_gradient_consistency(self, nlat, nlon, batch_size, norm, grid, atol, rtol, verbose=True):
        """ivsht([c, 0]) synthesizes the surface gradient ∇_S f of a scalar field
        f = isht(c).  Applying vsht to this gradient field must recover c in the
        spheroidal channel and zero in the toroidal channel, because a gradient
        field is curl-free (purely spheroidal).
        """
        if verbose:
            print(f"Testing gradient consistency on {nlat}x{nlon} {grid} grid with {norm} norm on {self.device.type}")

        # disable tf32
        disable_tf32()

        # set seed
        set_seed(333)

        vsht  = th.RealVectorSHT        (nlat, nlon, grid=grid, norm=norm).to(self.device)
        ivsht = th.InverseRealVectorSHT (nlat, nlon, grid=grid, norm=norm).to(self.device)
        lmax, mmax = vsht.lmax, vsht.mmax

        with torch.no_grad():
            c     = random_sht_coeffs(batch_size, lmax, mmax, self.device, zero_l0=True)
            zeros = torch.zeros_like(c)

            # synthesize gradient field: ivsht([c, 0]) = ∇_S f
            grad_f = ivsht(torch.stack([c, zeros], dim=-3))  # (batch, 2, nlat, nlon)

            # analyse: vsht(∇_S f) must give [c, 0]
            st = vsht(grad_f)   # (batch, 2, lmax, mmax)
            s  = st[..., 0, :, :]  # spheroidal
            t  = st[..., 1, :, :]  # toroidal

        self.assertTrue(compare_tensors("spheroidal coefficients", s, c,     atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors("toroidal coefficients",   t, zeros, atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            [32, 64, 16, "ortho",   "legendre-gauss", 1e-7, 1e-7],
            [32, 64, 16, "ortho",   "equiangular",    1e-7, 1e-7],
            [32, 64, 16, "ortho",   "lobatto",        1e-7, 1e-7],
            [32, 64, 16, "four-pi", "legendre-gauss", 1e-7, 1e-7],
            [32, 64, 16, "four-pi", "equiangular",    1e-7, 1e-7],
            [32, 64, 16, "four-pi", "lobatto",        1e-7, 1e-7],
            # [32, 64, 16, "schmidt", "legendre-gauss", 1e-9, 1e-9],
            # [32, 64, 16, "schmidt", "equiangular",    1e-9, 1e-9],
            # [32, 64, 16, "schmidt", "lobatto",        1e-9, 1e-9],
        ],
        skip_on_empty=True,
    )
    def test_curl_consistency(self, nlat, nlon, batch_size, norm, grid, atol, rtol, verbose=False):
        """ivsht([0, c]) synthesizes the surface curl ê_r × ∇_S f of a scalar field
        f = isht(c).  Applying vsht to this curl field must recover c in the
        toroidal channel and zero in the spheroidal channel, because a surface
        curl field is divergence-free (purely toroidal).
        """
        if verbose:
            print(f"Testing curl consistency on {nlat}x{nlon} {grid} grid with {norm} norm on {self.device.type}")

        # disable tf32
        disable_tf32()

        # set seed
        set_seed(333)

        vsht  = th.RealVectorSHT        (nlat, nlon, grid=grid, norm=norm).to(self.device)
        ivsht = th.InverseRealVectorSHT (nlat, nlon, grid=grid, norm=norm).to(self.device)
        lmax, mmax = vsht.lmax, vsht.mmax

        with torch.no_grad():
            c     = random_sht_coeffs(batch_size, lmax, mmax, self.device, zero_l0=True)
            zeros = torch.zeros_like(c)

            # synthesize curl field: ivsht([0, c]) = ê_r × ∇_S f
            curl_f = ivsht(torch.stack([zeros, c], dim=-3))  # (batch, 2, nlat, nlon)

            # analyse: vsht(ê_r × ∇_S f) must give [0, c]
            st = vsht(curl_f)   # (batch, 2, lmax, mmax)
            s  = st[..., 0, :, :]  # spheroidal
            t  = st[..., 1, :, :]  # toroidal

        self.assertTrue(compare_tensors("spheroidal coefficients", s, zeros, atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors("toroidal coefficients",   t, c,     atol=atol, rtol=rtol, verbose=verbose))


if __name__ == "__main__":
    unittest.main()
