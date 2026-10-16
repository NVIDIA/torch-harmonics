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

"""CC and Fejer-1 latitude resampling, plus their shared folded SHT analysis."""

import unittest

import numpy as np
import torch
from parameterized import parameterized, parameterized_class
from scipy.special import gammaln, lpmv
from testutils import compare_tensors, disable_tf32, requires_torch_compile, set_seed
from torch.autograd import gradcheck

import torch_harmonics as th
from torch_harmonics.legendre import _precompute_dlegpoly, _precompute_legpoly
from torch_harmonics.sht import (
    _fold_resampled_latitude,
    _fourier_shift_latitude,
    _periodic_latitude_extension,
    _periodic_latitude_extension_adjoint,
    _precompute_cc_resampling,
    _precompute_f1_resampling,
    _precompute_folded_projection,
)

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


def _transforms(grid, lmax, mmax, vector, **kwargs):
    analysis = th.RealVectorSHT if vector else th.RealSHT
    synthesis = th.InverseRealVectorSHT if vector else th.InverseRealSHT
    return analysis(grid, lmax=lmax, mmax=mmax, **kwargs), synthesis(grid, lmax=lmax, mmax=mmax, **kwargs)


def _coefficients(lmax, mmax, vector, lmmax, dtype, device):
    shape = (2, 2, lmax, mmax) if vector else (2, lmax, mmax)
    c = torch.randn(shape, dtype=torch.complex128 if dtype == torch.float64 else torch.complex64, device=device)
    l = torch.arange(lmax, device=device)[:, None]
    m = torch.arange(mmax, device=device)[None, :]
    mask = l >= m
    if lmmax is not None:
        mask = mask & (l - m < lmmax)
    c *= mask
    c[..., 0] = c[..., 0].real
    if vector:
        c[..., 0, :] = 0
    return c


def _scipy_mode_field(grid, l, m, coefficient, vector=False, component=0):
    """Evaluate one normalized positive-m harmonic (or vector derivative) with SciPy."""
    theta = grid.colats.numpy()[:, None]
    phi = grid.lons().numpy()[None, :]
    normalization = np.sqrt((2 * l + 1) / (4 * np.pi)) * np.exp(0.5 * (gammaln(l - m + 1) - gammaln(l + m + 1)))
    p = normalization * lpmv(m, l, np.cos(theta))
    harmonic = p * np.exp(1j * m * phi)
    if not vector:
        basis = harmonic
    else:
        derivative = ((l * np.cos(theta) * p - (l + m) * normalization * lpmv(m, l - 1, np.cos(theta))) / np.sin(theta)) * np.exp(1j * m * phi)
        longitude = 1j * m * harmonic / np.sin(theta)
        basis = np.stack((derivative, longitude) if component == 0 else (longitude, -derivative))
    return (1 if m == 0 else 2) * (coefficient * basis).real


@parameterized_class(("device",), _devices)
class TestCCLatitudeResampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        disable_tf32()

    @parameterized.expand([(nlat, vector) for nlat in (8, 9) for vector in (False, True)], skip_on_empty=True)
    def test_extension_shift_and_adjoint(self, nlat, vector, verbose=False):
        set_seed(333)
        buffers = {name: value.to(self.device) for name, value in _precompute_cc_resampling(nlat, 5, vector).items()}
        signs = buffers["_cc_parity"]
        phase = torch.complex(*buffers["_cc_phase"])
        x = torch.randn(2, 5, nlat, dtype=torch.complex128, device=self.device)
        y = torch.randn(2, 5, 2 * (nlat - 1), dtype=torch.complex128, device=self.device)
        extended = _periodic_latitude_extension(x, signs)
        # Pole samples are not doubled, and vector continuation has the opposite sign.
        expected_signs = (1 - 2 * torch.arange(5, device=self.device).remainder(2))[:, None] * (-1 if vector else 1)
        expected = torch.cat((x, expected_signs * x[..., 1:-1].flip(-1)), dim=-1)
        self.assertTrue(compare_tensors("extension", extended, expected, atol=0, rtol=0, verbose=verbose))
        lhs = (extended.conj() * y).sum()
        rhs = (x.conj() * _periodic_latitude_extension_adjoint(y, signs)).sum()
        self.assertTrue(compare_tensors("extension adjoint", lhs, rhs, atol=1e-12, rtol=1e-12, verbose=verbose))
        shifted = _fourier_shift_latitude(extended, phase)
        lhs = (shifted.conj() * y).sum()
        rhs = (extended.conj() * _fourier_shift_latitude(y, phase.conj())).sum()
        self.assertTrue(compare_tensors("shift adjoint", lhs, rhs, atol=1e-12, rtol=1e-12, verbose=verbose))
        # All paired frequencies shift exactly; the unpaired Nyquist cosine vanishes at midpoints.
        length = y.shape[-1]
        frequencies = torch.cat((torch.arange(length // 2), torch.arange(-length // 2, 0))).to(self.device)
        theta = 2 * torch.pi * torch.arange(length, dtype=torch.float64, device=self.device) / length
        modes = torch.exp(1j * frequencies[:, None] * theta)
        expected = torch.exp(1j * frequencies[:, None] * (theta + torch.pi / length))
        expected[length // 2] = 0
        self.assertTrue(compare_tensors("half-grid shift", _fourier_shift_latitude(modes, phase), expected, atol=1e-13, rtol=1e-13, verbose=verbose))
        shifted = _fourier_shift_latitude(y, phase)
        self.assertTrue(compare_tensors("shift conjugation", shifted.conj(), _fourier_shift_latitude(y.conj(), phase), atol=1e-12, rtol=1e-12, verbose=verbose))
        alternating = torch.where(torch.arange(length, device=self.device).remainder(2) == 0, 1.0, -1.0)
        expected = y - (y * alternating).mean(dim=-1, keepdim=True) * alternating
        recovered = _fourier_shift_latitude(shifted, phase.conj())
        self.assertTrue(compare_tensors("shift normal operator", recovered, expected, atol=1e-12, rtol=1e-12, verbose=verbose))

    @parameterized.expand([(nlat, vector) for nlat in (8, 9) for vector in (False, True)], skip_on_empty=True)
    def test_folded_quadrature(self, nlat, vector, verbose=False):
        set_seed(333)
        b = {name: value.to(self.device) for name, value in _precompute_cc_resampling(nlat, 4, vector).items()}
        phase = torch.complex(*b["_cc_phase"])
        x = torch.randn(2, 4, nlat, dtype=torch.complex128, device=self.device, requires_grad=True)
        folded = _fold_resampled_latitude(x, b["_cc_parity"], b["_cc_weights"], b["_cc_midpoint_weights"], phase)
        # Independent dense Fourier interpolation matrix, then A^H Q_o A.
        length = 2 * (nlat - 1)
        # Real cardinal interpolation with paired +/- frequencies. The Nyquist cosine
        # cos(pi * delta) is zero at these half-grid positions, for complex inputs too.
        freq = np.arange(1, length // 2)
        delta = np.arange(nlat - 1)[:, None] + 0.5 - np.arange(length)[None, :]
        shift = torch.tensor((1 + 2 * np.cos(2 * np.pi * delta[..., None] * freq / length).sum(axis=-1)) / length, device=self.device, dtype=torch.complex128)
        extension = _periodic_latitude_extension(torch.eye(nlat, dtype=torch.complex128, device=self.device).expand(4, -1, -1), b["_cc_parity"][:, None, :])
        a = torch.einsum("jk,mik->mji", shift, extension)
        dense_u = torch.diag_embed(b["_cc_weights"].expand(4, -1)).to(a.dtype) + torch.einsum("mji,j,mjk->mik", a.conj(), b["_cc_midpoint_weights"][: nlat - 1].to(a.dtype), a)
        torch.testing.assert_close(dense_u.imag, torch.zeros_like(dense_u.imag), atol=1e-13, rtol=0)
        torch.testing.assert_close(dense_u, dense_u.transpose(-1, -2), atol=1e-13, rtol=0)
        midpoint = torch.einsum("mji,...mi->...mj", a, x)
        expected = b["_cc_weights"] * x + torch.einsum("mji,...mj->...mi", a.conj(), b["_cc_midpoint_weights"][: nlat - 1] * midpoint)
        self.assertTrue(compare_tensors("folded quadrature", folded, expected, atol=1e-12, rtol=1e-12, verbose=verbose))
        gradient = torch.randn_like(folded)
        actual_grad = torch.autograd.grad(folded, x, gradient)[0]
        expected_grad = torch.autograd.grad(expected, x, gradient)[0]
        self.assertTrue(compare_tensors("fold gradient", actual_grad, expected_grad, atol=1e-12, rtol=1e-12, verbose=verbose))
        self.assertTrue(
            compare_tensors(
                "fold self-adjoint",
                actual_grad,
                _fold_resampled_latitude(gradient, b["_cc_parity"], b["_cc_weights"], b["_cc_midpoint_weights"], phase),
                atol=1e-12,
                rtol=1e-12,
                verbose=verbose,
            )
        )


@parameterized_class(("device",), _devices)
class TestFejer1LatitudeResampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        disable_tf32()

    @parameterized.expand([(nlat, vector) for nlat in (4, 5, 8, 9) for vector in (False, True)], skip_on_empty=True)
    def test_extension_shift_and_adjoint(self, nlat, vector, verbose=False):
        set_seed(334)
        buffers = {name: value.to(self.device) for name, value in _precompute_f1_resampling(nlat, 5, vector).items()}
        signs = buffers["_f1_parity"]
        phase = torch.complex(*buffers["_f1_phase"])
        x = torch.randn(2, 5, nlat, dtype=torch.complex128, device=self.device)
        y = torch.randn(2, 5, 2 * nlat, dtype=torch.complex128, device=self.device)
        extended = _periodic_latitude_extension(x, signs, include_poles=False)
        expected_signs = (1 - 2 * torch.arange(5, device=self.device).remainder(2))[:, None] * (-1 if vector else 1)
        expected = torch.cat((x, expected_signs * x.flip(-1)), dim=-1)
        self.assertTrue(compare_tensors("F1 extension", extended, expected, atol=0, rtol=0, verbose=verbose))
        lhs = (extended.conj() * y).sum()
        rhs = (x.conj() * _periodic_latitude_extension_adjoint(y, signs, include_poles=False)).sum()
        self.assertTrue(compare_tensors("F1 extension adjoint", lhs, rhs, atol=1e-12, rtol=1e-12, verbose=verbose))

        shifted = _fourier_shift_latitude(extended, phase)
        lhs = (shifted.conj() * y).sum()
        rhs = (extended.conj() * _fourier_shift_latitude(y, phase.conj())).sum()
        self.assertTrue(compare_tensors("F1 shift adjoint", lhs, rhs, atol=1e-12, rtol=1e-12, verbose=verbose))

        length = 2 * nlat
        frequencies = torch.cat((torch.arange(length // 2), torch.arange(-length // 2, 0))).to(self.device)
        theta = 2 * torch.pi * torch.arange(length, dtype=torch.float64, device=self.device) / length
        modes = torch.exp(1j * frequencies[:, None] * theta)
        expected_shift = torch.exp(1j * frequencies[:, None] * (theta - torch.pi / length))
        expected_shift[length // 2] = 0
        self.assertTrue(compare_tensors("F1 half-grid shift", _fourier_shift_latitude(modes, phase), expected_shift, atol=1e-13, rtol=1e-13, verbose=verbose))
        self.assertTrue(compare_tensors("F1 shift conjugation", shifted.conj(), _fourier_shift_latitude(extended.conj(), phase), atol=1e-12, rtol=1e-12, verbose=verbose))

    @parameterized.expand([(nlat, vector) for nlat in (4, 5, 8, 9) for vector in (False, True)], skip_on_empty=True)
    def test_folded_quadrature_matches_dense_operator(self, nlat, vector, verbose=False):
        set_seed(335)
        b = {name: value.to(self.device) for name, value in _precompute_f1_resampling(nlat, 4, vector).items()}
        phase = torch.complex(*b["_f1_phase"])
        x = torch.randn(2, 4, nlat, dtype=torch.complex128, device=self.device, requires_grad=True)
        folded = _fold_resampled_latitude(
            x,
            b["_f1_parity"],
            b["_f1_native_weights"],
            b["_f1_complementary_weights"],
            phase,
            include_poles=False,
        )

        # Form A explicitly from the periodic extension and Fourier shift, then
        # compare U = Q_native + A.T Q_complement A with the FFT folded operator.
        eye = torch.eye(nlat, dtype=torch.complex128, device=self.device).expand(4, -1, -1)
        extension = _periodic_latitude_extension(eye, b["_f1_parity"][:, None, :], include_poles=False)
        interpolation = _fourier_shift_latitude(extension, phase)[..., : nlat + 1]
        # The eye rows represent input samples; transpose them into the usual
        # target-by-source interpolation matrix convention.
        a = interpolation.transpose(-1, -2)
        complement = b["_f1_complementary_weights"][: nlat + 1]
        native = b["_f1_native_weights"]
        dense_u = torch.diag_embed(native.expand(4, -1)).to(a.dtype) + torch.einsum("mji,j,mjk->mik", a.conj(), complement, a)
        self.assertTrue(compare_tensors("F1 real projection", dense_u.imag, torch.zeros_like(dense_u.imag), atol=2e-13, rtol=0))
        self.assertTrue(compare_tensors("F1 symmetric projection", dense_u, dense_u.transpose(-1, -2), atol=2e-13, rtol=0))
        expected = torch.einsum("mji,...mi->...mj", a, x)
        expected = native * x + torch.einsum("mji,j,...mj->...mi", a.conj(), complement, expected)
        self.assertTrue(compare_tensors("F1 folded quadrature", folded, expected, atol=2e-12, rtol=2e-12, verbose=verbose))

        gradient = torch.randn_like(folded)
        actual_grad = torch.autograd.grad(folded, x, gradient)[0]
        expected_grad = torch.autograd.grad(expected, x, gradient)[0]
        self.assertTrue(compare_tensors("F1 fold gradient", actual_grad, expected_grad, atol=2e-12, rtol=2e-12, verbose=verbose))
        self.assertTrue(
            compare_tensors(
                "F1 fold self-adjoint",
                actual_grad,
                _fold_resampled_latitude(
                    gradient,
                    b["_f1_parity"],
                    native,
                    b["_f1_complementary_weights"],
                    phase,
                    include_poles=False,
                ),
                atol=2e-12,
                rtol=2e-12,
                verbose=verbose,
            )
        )


@parameterized_class(("device",), _devices)
class TestSharedFoldedSHTAnalysis(unittest.TestCase):
    """Shared extended-analysis coverage for Clenshaw-Curtis and Fejer-1 grids."""

    @classmethod
    def setUpClass(cls):
        disable_tf32()

    @parameterized.expand(
        [(vector, "equiangular", nlat, 2 * nlat - 1, nlat - 1, nlat - 1, lmmax) for vector in (False, True) for nlat in (8, 9, 34) for lmmax in (None, 3)]
        + [(vector, "fejer1", nlat, 2 * nlat - 1, nlat, nlat, lmmax) for vector in (False, True) for nlat in (8, 9) for lmmax in (None, 3)]
        + [(vector, "equiangular", 257, 33, 256, 16, None) for vector in (False, True)],
        skip_on_empty=True,
    )
    def test_effective_projection(self, vector, grid_type, nlat, nlon, lmax, mmax, lmmax):
        """Compare every row, complex contractions and gradients with runtime folding."""
        set_seed(417)
        grid = th.as_grid(grid_type, nlat=nlat, nlon=nlon)
        analysis, _ = _transforms(grid, lmax, mmax, vector, lmmax=lmmax)
        if grid_type == "fejer1":
            b = {name: value.to(self.device) for name, value in _precompute_f1_resampling(nlat, analysis.mmax, vector).items()}
            native_weights = b["_f1_native_weights"]
            complementary_weights = b["_f1_complementary_weights"]
            phase = torch.complex(*b["_f1_phase"])
            parity = b["_f1_parity"]
            include_poles = False
        else:
            b = {name: value.to(self.device) for name, value in _precompute_cc_resampling(nlat, analysis.mmax, vector).items()}
            native_weights = b["_cc_weights"]
            complementary_weights = b["_cc_midpoint_weights"]
            phase = torch.complex(*b["_cc_phase"])
            parity = b["_cc_parity"]
            include_poles = True
        if vector:
            basis = _precompute_dlegpoly(analysis.mmax, analysis.lmax, grid, truncation=analysis.grid_out)
            degrees = torch.arange(analysis.lmax, dtype=torch.float64)
            basis *= (1 / (degrees * (degrees + 1)).clamp(min=1))[None, None, :, None]
            basis[1].neg_()
        else:
            basis = _precompute_legpoly(analysis.mmax, analysis.lmax, grid, truncation=analysis.grid_out)
        basis = basis.to(self.device)
        folded = _fold_resampled_latitude(basis.transpose(-3, -2), parity, native_weights, complementary_weights, phase, include_poles)
        self.assertLess(folded.imag.abs().max().item(), 1e-13)
        effective = analysis.weights.to(self.device)
        torch.testing.assert_close(effective, folded.real.transpose(-3, -2), atol=1e-13, rtol=1e-13)
        # Local order offsets must use global parity, including odd starting orders.
        for mmin in (1, 2, analysis.mmax):
            local = _precompute_folded_projection(grid, analysis.grid_out, "ortho", True, vector, mmin=mmin, kmin=2, kmax=nlat - 1)
            torch.testing.assert_close(local.to(self.device), effective[..., mmin:, :, 2:-1], atol=1e-13, rtol=1e-13)
        x = torch.randn(2, analysis.mmax, nlat, device=self.device, dtype=torch.complex128, requires_grad=True)
        folded_x = _fold_resampled_latitude(x, parity, native_weights, complementary_weights, phase, include_poles)
        equation = "...mk,dmlk->...dlm" if vector else "...mk,mlk->...lm"
        expected = torch.einsum(equation, folded_x, basis.to(x.dtype))
        actual = torch.einsum(equation, x, effective.to(x.dtype))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        gradient = torch.randn_like(actual)
        expected_grad = torch.autograd.grad(expected, x, gradient, retain_graph=True)[0]
        for _ in range(2):
            actual_grad = torch.autograd.grad(actual, x, gradient, retain_graph=True)[0]
            torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-12)
        self.assertEqual(list(dict(analysis.named_buffers())), ["weights"])
        # A warm constructor must reuse numerical work without sharing mutable buffers.
        another, _ = _transforms(grid, analysis.lmax, analysis.mmax, vector, lmmax=lmmax)
        torch.testing.assert_close(analysis.weights, another.weights, atol=0, rtol=0)
        self.assertNotEqual(analysis.weights.data_ptr(), another.weights.data_ptr())
        another.weights.zero_()
        self.assertGreater(analysis.weights.abs().max().item(), 0)
        self.assertEqual(_precompute_folded_projection.cache_parameters()["maxsize"], 2)

    @parameterized.expand(
        [
            (vector, dtype, grid_type, *case)
            for vector in (False, True)
            for dtype in (torch.float32, torch.float64)
            for grid_type, cases in [
                (
                    "equiangular",
                    [
                        (73, 144, 71, 71, None, "ortho", True),
                        (73, 144, 72, 72, None, "ortho", True),
                        (73, 144, 72, 19, None, "schmidt", False),
                        (73, 145, 72, 72, 36, "four-pi", False),
                        (74, 145, 73, 37, 37, "ortho", True),
                        (16, 29, 15, 15, 5, "schmidt", True),
                    ],
                ),
                (
                    "fejer1",
                    [
                        (8, 15, 8, 8, None, "ortho", True),
                        (9, 18, 9, 9, None, "ortho", True),
                        (17, 35, 17, 6, 5, "schmidt", False),
                        (72, 144, 72, 72, None, "ortho", True),
                    ],
                ),
            ]
            for case in cases
        ],
        skip_on_empty=True,
    )
    def test_bandlimited_spectra(self, vector, dtype, grid_type, nlat, nlon, lmax, mmax, lmmax, norm, csphase, verbose=False):
        set_seed(333)
        grid = th.as_grid(grid_type, nlat=nlat, nlon=nlon)
        analysis, synthesis = _transforms(grid, lmax, mmax, vector, lmmax=lmmax, norm=norm, csphase=csphase)
        self.assertTrue(analysis._extended_folded)
        analysis = analysis.to(device=self.device, dtype=dtype)
        synthesis = synthesis.to(device=self.device, dtype=dtype)
        expected = _coefficients(lmax, mmax, vector, lmmax, dtype, self.device)
        actual = analysis(synthesis(expected))
        # Vector synthesis scales degree l by its derivative. In Schmidt normalization,
        # rounding that field and analyzing low degrees amplifies cancellation noise.
        atol = (5e-5 if vector and norm == "schmidt" else 2e-5) if dtype == torch.float32 else 2e-12
        rtol = 2e-6 if dtype == torch.float32 else 2e-13
        self.assertTrue(compare_tensors("bandlimited spectrum", expected, actual, atol=atol, rtol=rtol, verbose=verbose))
        self.assertLess((actual - expected).abs().max().item(), atol)
        self.assertLess((actual - expected).norm().item() / expected.norm().item(), rtol)
        self.assertTrue(compare_tensors("highest degree", expected[..., -1, :], actual[..., -1, :], atol=atol, rtol=rtol, verbose=verbose))
        self.assertEqual(analysis.grid_out, th.SpectralGrid(lmax, mmax, lmmax))
        self.assertIs(analysis.grid_in, grid)
        self.assertEqual(analysis.state_dict(), {})

    @parameterized.expand(
        [
            (vector, l, m, component, norm, csphase)
            for vector in (False, True)
            for l, m in [(1, 0), (35, 0), (35, 7), (70, 69), (71, 1), (71, 35), (71, 71)]
            for component in (range(2) if vector else range(1))
            for norm, csphase in ([("ortho", True)] if (l, m) != (71, 35) else [(norm, csphase) for norm in ("ortho", "four-pi", "schmidt", "unnorm") for csphase in (True, False)])
        ],
        skip_on_empty=True,
    )
    def test_f1_scipy_harmonics_through_maximum_degree(self, vector, l, m, component, norm, csphase, verbose=False):
        """Analyze independent SciPy harmonics, including F1 sectoral T71 modes."""
        grid = th.as_grid("fejer1", nlat=72, nlon=144)
        coefficient = 0.7 + (0.3j if m else 0j)
        spatial = torch.tensor(_scipy_mode_field(grid, l, m, coefficient, vector, component), dtype=torch.float64, device=self.device)
        analysis = (th.RealVectorSHT if vector else th.RealSHT)(grid, lmax=72, mmax=72, norm=norm, csphase=csphase).to(self.device, torch.float64)
        actual = analysis(spatial)
        expected = torch.zeros_like(actual)
        factor = 1.0 if norm == "ortho" else np.sqrt(4 * np.pi)
        if norm == "schmidt":
            factor /= np.sqrt(2 * l + 1)
        if not csphase:
            factor *= (-1) ** m
        if vector:
            expected[component, l, m] = factor * coefficient
        else:
            expected[l, m] = factor * coefficient
        self.assertTrue(compare_tensors("SciPy F1 harmonic", expected, actual, atol=3e-11, rtol=3e-11, verbose=verbose))
        self.assertLess((actual - expected).abs().max().item(), 3e-11)
        synthesis = (th.InverseRealVectorSHT if vector else th.InverseRealSHT)(grid, lmax=72, mmax=72, norm=norm, csphase=csphase).to(self.device, torch.float64)
        synthesized = synthesis(expected)
        self.assertTrue(compare_tensors("SciPy F1 synthesis", spatial, synthesized, atol=3e-11, rtol=3e-11, verbose=verbose))
        self.assertLess((synthesized - spatial).abs().max().item(), 3e-11)

    @parameterized.expand(
        [
            (vector, l, m, component, norm, csphase)
            for vector in (False, True)
            for l, m in [(70, 0), (70, 69), (71, 0), (71, 1), (71, 35), (71, 70), (71, 71)]
            for component in (range(2) if vector else range(1))
            for norm, csphase in [("ortho", True), ("four-pi", False), ("schmidt", False), ("unnorm", True)]
        ],
        skip_on_empty=True,
    )
    def test_independent_harmonics(self, vector, l, m, component, norm, csphase, verbose=False):
        """Analyze SciPy harmonics and derivatives without using our inverse transform."""
        grid = th.as_grid("equiangular", nlat=73, nlon=144)
        theta = grid.colats.numpy()[:, None]
        phi = grid.lons().numpy()[None, :]
        # SciPy's associated Legendre function uses its own implementation.
        normalization = np.sqrt((2 * l + 1) / (4 * np.pi)) * np.exp(0.5 * (gammaln(l - m + 1) - gammaln(l + m + 1)))
        p = normalization * lpmv(m, l, np.cos(theta))
        y = p * np.exp(1j * m * phi)
        derivative = np.zeros_like(y)
        derivative[1:-1] = ((l * np.cos(theta[1:-1]) * p[1:-1] - (l + m) * normalization * lpmv(m, l - 1, np.cos(theta[1:-1]))) / np.sin(theta[1:-1])) * np.exp(1j * m * phi)
        if m == 1:
            derivative[0] = -0.5 * l * (l + 1) * normalization * np.exp(1j * phi)
            derivative[-1] = (-1) ** (l + 1) * 0.5 * l * (l + 1) * normalization * np.exp(1j * phi)
        c = 0.7 + (0.3j if m else 0j)
        if vector:
            longitude = np.zeros_like(y)
            longitude[1:-1] = 1j * m * y[1:-1] / np.sin(theta[1:-1])
            if m == 1:
                longitude[[0, -1]] = 1j * derivative[[0, -1]] / np.cos(theta[[0, -1]])
            basis = np.stack((derivative, longitude) if component == 0 else (longitude, -derivative))
        else:
            basis = y
        field = torch.tensor((1 if m == 0 else 2) * (c * basis).real, device=self.device)
        analysis = (th.RealVectorSHT if vector else th.RealSHT)(grid, lmax=72, mmax=72, norm=norm, csphase=csphase).to(self.device)
        actual = analysis(field)
        expected = torch.zeros_like(actual)
        factor = 1.0 if norm == "ortho" else np.sqrt(4 * np.pi)
        if norm == "schmidt":
            factor /= np.sqrt(2 * l + 1)
        if not csphase:
            factor *= (-1) ** m
        if vector:
            expected[component, l, m] = factor * c
        else:
            expected[l, m] = factor * c
        self.assertTrue(compare_tensors("SciPy harmonic analysis", expected, actual, atol=3e-12, rtol=3e-12, verbose=verbose))
        self.assertLess((actual - expected).abs().max().item(), 3e-12)

    @parameterized.expand(
        [
            (vector, grid_type, lmax)
            for vector in (False, True)
            for grid_type, lmax in [
                ("equiangular", None),
                ("equiangular", 5),
                ("equiangular", 9),
                ("fejer1", None),
                ("fejer1", 9),
                ("legendre-gauss", 17),
                ("lobatto", 16),
            ]
        ],
        skip_on_empty=True,
    )
    def test_direct_path(self, vector, grid_type, lmax, verbose=False):
        """CC buffers retain the original precision; F1 vector factors use float64."""
        grid = th.as_grid(grid_type, nlat=17, nlon=34)
        analysis = (th.RealVectorSHT if vector else th.RealSHT)(grid, lmax=lmax)
        self.assertFalse(analysis._extended_folded)
        weights = 2.0 * torch.pi * grid.colat_weights
        if vector:
            p = _precompute_dlegpoly(analysis.mmax, analysis.lmax, grid, truncation=analysis.grid_out)
            if grid_type == "fejer1":
                l = torch.arange(analysis.lmax, dtype=torch.float64)
            else:
                l = torch.arange(analysis.lmax)
            factor = 1.0 / l / (l + 1)
            factor[0] = 1
            expected = torch.einsum("dmlk,k,l->dmlk", p, weights, factor).contiguous()
            expected[1] *= -1
        else:
            p = _precompute_legpoly(analysis.mmax, analysis.lmax, grid, truncation=analysis.grid_out)
            expected = torch.einsum("mlk,k->mlk", p, weights).contiguous()
        self.assertTrue(compare_tensors("native projection", expected, analysis.weights, atol=0, rtol=0, verbose=verbose))
        self.assertEqual(list(analysis.named_buffers())[0][0], "weights")
        self.assertEqual(len(list(analysis.buffers())), 1)

    @parameterized.expand([(vector, nlat, nlon) for vector in (False, True) for nlat, nlon in [(8, 16), (9, 19)]], skip_on_empty=True)
    def test_f1_native_direct_bandwidth(self, vector, nlat, nlon, verbose=False):
        """The ordinary Fejér rule recovers its documented native triangular band."""
        set_seed(337)
        grid = th.as_grid("fejer1", nlat=nlat, nlon=nlon)
        lmax = (nlat + 1) // 2
        analysis, synthesis = _transforms(grid, lmax, lmax, vector)
        analysis = analysis.to(device=self.device, dtype=torch.float64)
        synthesis = synthesis.to(device=self.device, dtype=torch.float64)
        self.assertFalse(analysis._extended_folded)
        self.assertEqual(grid.max_exact_degree, lmax)
        expected = _coefficients(lmax, lmax, vector, None, torch.float64, self.device)[0]
        actual = analysis(synthesis(expected))
        self.assertTrue(compare_tensors("F1 native-band spectrum", expected, actual, atol=2e-12, rtol=2e-12, verbose=verbose))
        if nlat == 8:
            default = (th.RealVectorSHT if vector else th.RealSHT)(grid)
            self.assertEqual((default.lmax, default.mmax), (4, 4))
            self.assertFalse(default._extended_folded)

    def test_f1_default_is_t89_on_180_by_360(self):
        grid = th.as_grid("fejer1", nlat=180, nlon=360)
        self.assertEqual(grid.max_exact_degree, 90)
        self.assertEqual(th.truncate_sht(grid), th.SpectralGrid(90, 90))
        self.assertEqual(th.RealSHT(grid).grid_out, th.SpectralGrid(90, 90))

    def test_f1_extended_selection_and_oversized_bounds(self):
        for nlat, nlon in [(8, 16), (9, 19)]:
            grid = th.as_grid("fejer1", nlat=nlat, nlon=nlon)
            supported = th.RealSHT(grid, lmax=nlat, mmax=nlat)
            self.assertTrue(supported._extended_folded)
            self.assertEqual(supported.grid_out, th.SpectralGrid(nlat, nlat))
            oversized = th.RealSHT(grid, lmax=nlat + 1, mmax=nlat + 1)
            self.assertFalse(oversized._extended_folded)
            self.assertEqual(oversized.grid_out, th.SpectralGrid(nlat + 1, nlat + 1))
        # With even nlon=2*nlat-2, lmax=nlat is latitude-supported but mmax
        # would include the unrecoverable complex longitude Nyquist order.
        nyquist_limited = th.RealSHT(th.as_grid("fejer1", nlat=8, nlon=14), lmax=8, mmax=8)
        self.assertFalse(nyquist_limited._extended_folded)

    @parameterized.expand(
        [
            (vector, dtype, *case)
            for vector in (False, True)
            for dtype in (torch.float32, torch.float64)
            for case in [
                (32, 64, 64, 64, None),
                (33, 64, 65, 65, None),
                (32, 64, 64, None, 11),
                (33, 64, 65, 65, 20),
                (73, 144, 73, 72, 18),
                (25, 32, 24, None, None),
                (25, 32, 24, 17, 6),
                (25, 32, 24, 18, None),
                (73, 144, 74, None, None),
            ]
        ],
        skip_on_empty=True,
    )
    def test_oversized_dimensions(self, vector, dtype, nlat, nlon, lmax, mmax, lmmax, verbose=False):
        set_seed(333)
        grid = th.as_grid("equiangular", nlat=nlat, nlon=nlon)
        analysis = (th.RealVectorSHT if vector else th.RealSHT)(grid, lmax=lmax, mmax=mmax, lmmax=lmmax)
        self.assertFalse(analysis._extended_folded)
        self.assertEqual(analysis.grid_out, th.SpectralGrid(lmax, min(lmax, nlon // 2 + 1) if mmax is None else mmax, lmmax))
        # Native projection formula, retaining the original degree-factor precision.
        q = 2 * torch.pi * grid.colat_weights
        if vector:
            p = _precompute_dlegpoly(analysis.mmax, analysis.lmax, grid, truncation=analysis.grid_out)
            l = torch.arange(analysis.lmax)
            factor = 1.0 / l / (l + 1)
            factor[0] = 1
            weights = torch.einsum("dmlk,k,l->dmlk", p, q, factor).contiguous()
            weights[1] *= -1
        else:
            p = _precompute_legpoly(analysis.mmax, analysis.lmax, grid, truncation=analysis.grid_out)
            weights = torch.einsum("mlk,k->mlk", p, q).contiguous()
        analysis = analysis.to(device=self.device, dtype=dtype)
        self.assertTrue(compare_tensors("oversized native weights", weights.to(device=self.device, dtype=dtype), analysis.weights, atol=0, rtol=0, verbose=verbose))
        x = torch.randn((2,) + grid.shape if vector else grid.shape, device=self.device, dtype=dtype, requires_grad=True)
        modes = torch.fft.rfft(x, dim=-1, norm="forward")
        if analysis.mmax > modes.shape[-1]:
            modes = torch.nn.functional.pad(modes, (0, analysis.mmax - modes.shape[-1]))
        modes = modes[..., : analysis.mmax].transpose(-1, -2)
        # Evaluate the discrete quadrature in complex arithmetic independently of
        # the implementation's separate real/imaginary contractions.
        w = weights.to(device=self.device, dtype=modes.dtype)
        if vector:
            s0 = torch.einsum("mk,mlk->lm", modes[0], w[0]) + 1j * torch.einsum("mk,mlk->lm", modes[1], w[1])
            t0 = 1j * torch.einsum("mk,mlk->lm", modes[0], w[1]) - torch.einsum("mk,mlk->lm", modes[1], w[0])
            expected = torch.stack((s0, t0))
        else:
            expected = torch.einsum("mk,mlk->lm", modes, w)
        actual = analysis(x)
        if mmax is None:
            explicit = (th.RealVectorSHT if vector else th.RealSHT)(grid, lmax=lmax, mmax=analysis.mmax, lmmax=lmmax).to(device=self.device, dtype=dtype)
            self.assertFalse(explicit._extended_folded)
            self.assertTrue(compare_tensors("inferred and explicit orders", actual, explicit(x), atol=0, rtol=0, verbose=verbose))
        self.assertEqual(actual.shape, ((2,) if vector else ()) + analysis.grid_out.shape)
        tol = 1e-5 if dtype == torch.float32 else 1e-12
        self.assertTrue(compare_tensors("oversized direct quadrature", expected, actual, atol=tol, rtol=tol, verbose=verbose))
        gradient = torch.randn_like(actual)
        actual_grad = torch.autograd.grad(actual, x, gradient)[0]
        expected_grad = torch.autograd.grad(expected, x, gradient)[0]
        self.assertTrue(compare_tensors("oversized input gradient", expected_grad, actual_grad, atol=tol, rtol=tol, verbose=verbose))
        l = torch.arange(analysis.lmax, device=self.device)[:, None]
        m = torch.arange(analysis.mmax, device=self.device)[None, :]
        mask = m <= l
        if lmmax is not None:
            mask &= l - m < lmmax
        self.assertTrue(compare_tensors("oversized mask", actual[..., ~mask], torch.zeros_like(actual[..., ~mask]), atol=0, rtol=0, verbose=verbose))

    @parameterized.expand(
        [
            (vector, lmax, mmax, lmmax)
            for vector in (False, True)
            for lmax, mmax, lmmax in [(72, 73, None), (72, 100, None), (-1, 0, None), (3, -1, None), (True, 1, None), (3, 3, 0)]
        ],
        skip_on_empty=True,
    )
    def test_invalid_spectral_bounds(self, vector, lmax, mmax, lmmax):
        grid = th.as_grid("equiangular", nlat=73, nlon=144)
        with self.assertRaises(ValueError):
            (th.RealVectorSHT if vector else th.RealSHT)(grid, lmax=lmax, mmax=mmax, lmmax=lmmax)
        self.assertEqual(grid.max_exact_degree, 37)

    @parameterized.expand(
        [
            ("equiangular", vector, dtype, nlat, profile)
            for vector in (False, True)
            for dtype in (torch.float32, torch.float64)
            for nlat in (8, 9, 73, 74)
            for profile in ("spatial", "latitude", "localized", "alternating")
        ]
        + [
            ("fejer1", vector, dtype, nlat, profile)
            for vector in (False, True)
            for dtype in (torch.float32, torch.float64)
            for nlat in (8, 9)
            for profile in ("spatial", "latitude", "localized", "alternating")
        ],
        skip_on_empty=True,
    )
    def test_arbitrary_input_zero_order_reality(self, geometry, vector, dtype, nlat, profile, verbose=False):
        set_seed(333)
        nlon = 2 * (nlat - 1) if geometry == "equiangular" else 2 * nlat
        grid = th.as_grid(geometry, nlat=nlat, nlon=nlon)
        lmax = nlat - 1 if geometry == "equiangular" else nlat
        analysis = (th.RealVectorSHT if vector else th.RealSHT)(grid, lmax=lmax, mmax=lmax).to(device=self.device, dtype=dtype)
        self.assertTrue(analysis._extended_folded)
        leading = (2,) if vector else ()
        if profile == "spatial":
            x = torch.randn(leading + grid.shape, device=self.device, dtype=dtype)
        else:
            if profile == "latitude":
                latitude = torch.randn(leading + (nlat,), device=self.device, dtype=dtype)
            elif profile == "localized":
                theta = grid.colats.to(device=self.device, dtype=dtype)
                latitude = torch.exp(-(((theta - 0.27 * torch.pi) / 0.2) ** 2))
            else:
                latitude = (1 - 2 * torch.arange(nlat, device=self.device).remainder(2)).to(dtype)
            x = latitude[..., None].expand(leading + grid.shape).contiguous()
        actual = analysis(x)
        tol = 64 * torch.finfo(dtype).eps * max(1.0, actual.abs().max().item())
        self.assertTrue(compare_tensors("arbitrary-input zero-order reality", actual[..., 0].imag, torch.zeros_like(actual[..., 0].imag), atol=tol, rtol=0, verbose=verbose))

    @parameterized.expand(
        [(geometry, vector, lmax) for geometry in ("equiangular", "fejer1") for vector in (False, True) for lmax in (0, 8)],
        skip_on_empty=True,
    )
    def test_empty_analysis_synthesis(self, geometry, vector, lmax):
        grid = th.as_grid(geometry, nlat=9, nlon=16)
        analysis, synthesis = _transforms(grid, lmax, 0, vector)
        analysis, synthesis = analysis.to(self.device), synthesis.to(self.device)
        x = torch.randn(((2,) if vector else ()) + grid.shape, device=self.device, dtype=torch.float64, requires_grad=True)
        coefficients = analysis(x)
        actual = synthesis(coefficients)
        torch.testing.assert_close(actual, torch.zeros_like(x), atol=0, rtol=0)
        actual.sum().backward()
        torch.testing.assert_close(x.grad, torch.zeros_like(x), atol=0, rtol=0)

    @parameterized.expand([(geometry, vector) for geometry in ("equiangular", "fejer1") for vector in (False, True)], skip_on_empty=True)
    def test_empty_orders(self, geometry, vector, verbose=False):
        set_seed(333)
        grid = th.as_grid(geometry, nlat=9, nlon=16)
        analysis, _ = _transforms(grid, 8, 0, vector)
        analysis = analysis.to(self.device)
        x = torch.randn((2,) + grid.shape if vector else grid.shape, device=self.device, requires_grad=True)
        result = analysis(x)
        expected = torch.zeros((2, 8, 0) if vector else (8, 0), device=self.device, dtype=torch.complex64)
        self.assertTrue(compare_tensors("empty spectrum", expected, result, atol=0, rtol=0, verbose=verbose))
        result.real.sum().backward()
        self.assertTrue(compare_tensors("empty gradient", torch.zeros_like(x), x.grad, atol=0, rtol=0, verbose=verbose))

    @parameterized.expand(
        [(geometry, vector, nlat, nlon, lmax) for geometry, nlat, nlon, lmax in [("equiangular", 5, 7, 4), ("fejer1", 5, 9, 5)] for vector in (False, True)],
        skip_on_empty=True,
    )
    def test_gradcheck(self, geometry, vector, nlat, nlon, lmax):
        set_seed(333)
        grid = th.as_grid(geometry, nlat=nlat, nlon=nlon)
        analysis, _ = _transforms(grid, lmax, lmax, vector)
        analysis = analysis.to(self.device)
        x = torch.randn((2,) + grid.shape if vector else grid.shape, device=self.device, dtype=torch.float64, requires_grad=True)
        self.assertTrue(gradcheck(analysis, (x,), eps=1e-6, atol=1e-7, rtol=1e-5))
        with self.assertRaises(RuntimeError):
            analysis(x[..., :-1])

    @parameterized.expand(
        [
            (vector, dtype, grid_type, nlat, nlon, lmax)
            for vector in (False, True)
            for dtype in (torch.float32, torch.float64)
            for grid_type, nlat, nlon, lmax in [
                ("equiangular", 9, 16, 8),
                ("equiangular", 73, 144, 72),
                ("fejer1", 9, 18, 9),
                ("fejer1", 72, 144, 72),
            ]
        ],
        skip_on_empty=True,
    )
    @requires_torch_compile
    def test_compile(self, vector, dtype, grid_type, nlat, nlon, lmax, verbose=False):
        set_seed(333)
        grid = th.as_grid(grid_type, nlat=nlat, nlon=nlon)
        analysis, synthesis = _transforms(grid, lmax, lmax, vector)
        analysis = analysis.to(device=self.device, dtype=dtype)
        synthesis = synthesis.to(device=self.device, dtype=dtype)

        def fn(t):
            return synthesis(analysis(t))

        x = torch.randn((2, 2) + grid.shape if vector else (2,) + grid.shape, device=self.device, dtype=dtype, requires_grad=True)
        gradient = torch.randn_like(x)
        expected = fn(x)
        expected_grad = torch.autograd.grad(expected, x, gradient)[0]
        actual = torch.compile(fn, fullgraph=True, dynamic=False)(x)
        actual_grad = torch.autograd.grad(actual, x, gradient)[0]
        tolerance = 1e-5 if dtype == torch.float32 else 1e-12
        self.assertTrue(compare_tensors("compiled forward", expected, actual, atol=tolerance, rtol=tolerance, verbose=verbose))
        self.assertTrue(compare_tensors("compiled backward", expected_grad, actual_grad, atol=tolerance, rtol=tolerance, verbose=verbose))
