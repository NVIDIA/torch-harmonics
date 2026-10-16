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

"""The CC and Fejer-1 latitude folding operators behind the extended SHT analysis.

The analysis built on them is tested in test_sht.py."""

import unittest

import numpy as np
import torch
from parameterized import parameterized, parameterized_class
from testutils import compare_tensors, disable_tf32, set_seed

from torch_harmonics.sht import (
    _fold_resampled_latitude,
    _fourier_shift_latitude,
    _periodic_latitude_extension,
    _periodic_latitude_extension_adjoint,
    _precompute_cc_resampling,
    _precompute_f1_resampling,
)

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


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
