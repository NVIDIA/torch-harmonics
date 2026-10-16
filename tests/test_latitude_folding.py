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

"""The latitude folding behind the extended SHT analysis on equiangular and Fejer-1 grids.

The analysis integrates on a denser grid: it extends each order's latitude samples
periodically, shifts them half a sample onto complementary nodes and folds the
complementary quadrature back onto the native samples. The transform built on it is
tested in test_sht.py."""

import unittest

import numpy as np
import torch
from parameterized import parameterized, parameterized_class
from testutils import compare_tensors, disable_tf32, set_seed

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

# How each grid folds: its buffers, whether the native samples include the poles, and where
# the complementary nodes sit, in native sample spacings from each native node.
_RULES = {
    "equiangular": dict(
        precompute=_precompute_cc_resampling,
        keys=("_cc_parity", "_cc_weights", "_cc_midpoint_weights", "_cc_phase"),
        include_poles=True,
        offset=0.5,
    ),
    "fejer1": dict(
        precompute=_precompute_f1_resampling,
        keys=("_f1_parity", "_f1_native_weights", "_f1_complementary_weights", "_f1_phase"),
        include_poles=False,
        offset=-0.5,
    ),
}


def _folding(grid, nlat, mmax, vector, device):
    """(parity, native weights, complementary weights, phase, include_poles) of a grid."""
    rule = _RULES[grid]
    buffers = rule["precompute"](nlat, mmax, vector)
    parity, native, complement, phase = (buffers[key].to(device) for key in rule["keys"])
    return parity, native, complement, torch.complex(*phase), rule["include_poles"]


def _period(nlat, include_poles):
    """Length of the periodic extension: the reflected half repeats no pole sample."""
    return 2 * (nlat - 1) if include_poles else 2 * nlat


# grid, nlat, vector
_OPERATOR_CASES = [[grid, nlat, vector] for grid, nlats in (("equiangular", (8, 9)), ("fejer1", (4, 5, 8, 9))) for nlat in nlats for vector in (False, True)]


@parameterized_class(("device",), _devices)
class TestLatitudeFolding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        disable_tf32()

    @parameterized.expand(_OPERATOR_CASES, skip_on_empty=True)
    def test_extension_and_shift(self, grid, nlat, vector, verbose=False):
        """The periodic extension and the half-sample Fourier shift, and their adjoints."""
        set_seed(333)
        parity, _, _, phase, include_poles = _folding(grid, nlat, 5, vector, self.device)
        length = _period(nlat, include_poles)
        x = torch.randn(2, 5, nlat, dtype=torch.complex128, device=self.device)
        y = torch.randn(2, 5, length, dtype=torch.complex128, device=self.device)

        extended = _periodic_latitude_extension(x, parity, include_poles=include_poles)
        # The reflection alternates sign with the order, and vector continuation has the opposite sign.
        expected_signs = (1 - 2 * torch.arange(5, device=self.device).remainder(2))[:, None] * (-1 if vector else 1)
        reflected = x[..., 1:-1] if include_poles else x
        expected = torch.cat((x, expected_signs * reflected.flip(-1)), dim=-1)
        self.assertTrue(compare_tensors("extension", extended, expected, atol=0, rtol=0, verbose=verbose))
        lhs = (extended.conj() * y).sum()
        rhs = (x.conj() * _periodic_latitude_extension_adjoint(y, parity, include_poles=include_poles)).sum()
        self.assertTrue(compare_tensors("extension adjoint", lhs, rhs, atol=1e-12, rtol=1e-12, verbose=verbose))

        shifted = _fourier_shift_latitude(extended, phase)
        lhs = (shifted.conj() * y).sum()
        rhs = (extended.conj() * _fourier_shift_latitude(y, phase.conj())).sum()
        self.assertTrue(compare_tensors("shift adjoint", lhs, rhs, atol=1e-12, rtol=1e-12, verbose=verbose))

        # All paired frequencies move onto the complementary nodes; the unpaired Nyquist cosine vanishes there.
        frequencies = torch.cat((torch.arange(length // 2), torch.arange(-length // 2, 0))).to(self.device)
        theta = 2 * torch.pi * torch.arange(length, dtype=torch.float64, device=self.device) / length
        modes = torch.exp(1j * frequencies[:, None] * theta)
        expected = torch.exp(1j * frequencies[:, None] * (theta + 2 * torch.pi * _RULES[grid]["offset"] / length))
        expected[length // 2] = 0
        self.assertTrue(compare_tensors("half-sample shift", _fourier_shift_latitude(modes, phase), expected, atol=1e-13, rtol=1e-13, verbose=verbose))

        shifted = _fourier_shift_latitude(y, phase)
        self.assertTrue(compare_tensors("shift conjugation", shifted.conj(), _fourier_shift_latitude(y.conj(), phase), atol=1e-12, rtol=1e-12, verbose=verbose))
        alternating = torch.where(torch.arange(length, device=self.device).remainder(2) == 0, 1.0, -1.0)
        expected = y - (y * alternating).mean(dim=-1, keepdim=True) * alternating
        recovered = _fourier_shift_latitude(shifted, phase.conj())
        self.assertTrue(compare_tensors("shift normal operator", recovered, expected, atol=1e-12, rtol=1e-12, verbose=verbose))

    @parameterized.expand(_OPERATOR_CASES, skip_on_empty=True)
    def test_folded_quadrature(self, grid, nlat, vector, verbose=False):
        """Folding equals Q_native + A^H Q_complement A for an independently built interpolation A."""
        set_seed(333)
        parity, native, complement, phase, include_poles = _folding(grid, nlat, 4, vector, self.device)
        x = torch.randn(2, 4, nlat, dtype=torch.complex128, device=self.device, requires_grad=True)
        folded = _fold_resampled_latitude(x, parity, native, complement, phase, include_poles)

        # Real cardinal interpolation with paired +/- frequencies, from the periodic extension
        # onto the complementary nodes. The Nyquist cosine is zero at these half-sample positions.
        length = _period(nlat, include_poles)
        ncomplement = nlat - 1 if include_poles else nlat + 1
        freq = np.arange(1, length // 2)
        delta = np.arange(ncomplement)[:, None] + _RULES[grid]["offset"] - np.arange(length)[None, :]
        cardinal = (1 + 2 * np.cos(2 * np.pi * delta[..., None] * freq / length).sum(axis=-1)) / length
        shift = torch.tensor(cardinal, device=self.device, dtype=torch.complex128)
        eye = torch.eye(nlat, dtype=torch.complex128, device=self.device).expand(4, -1, -1)
        extension = _periodic_latitude_extension(eye, parity[:, None, :], include_poles=include_poles)
        a = torch.einsum("jk,mik->mji", shift, extension)
        q = complement[:ncomplement].to(a.dtype)

        dense_u = torch.diag_embed(native.expand(4, -1)).to(a.dtype) + torch.einsum("mji,j,mjk->mik", a.conj(), q, a)
        self.assertTrue(compare_tensors("real projection", dense_u.imag, torch.zeros_like(dense_u.imag), atol=2e-13, rtol=0, verbose=verbose))
        self.assertTrue(compare_tensors("symmetric projection", dense_u, dense_u.transpose(-1, -2), atol=2e-13, rtol=0, verbose=verbose))
        expected = native * x + torch.einsum("mji,j,...mj->...mi", a.conj(), q, torch.einsum("mji,...mi->...mj", a, x))
        self.assertTrue(compare_tensors("folded quadrature", folded, expected, atol=2e-12, rtol=2e-12, verbose=verbose))

        gradient = torch.randn_like(folded)
        actual_grad = torch.autograd.grad(folded, x, gradient)[0]
        expected_grad = torch.autograd.grad(expected, x, gradient)[0]
        self.assertTrue(compare_tensors("fold gradient", actual_grad, expected_grad, atol=2e-12, rtol=2e-12, verbose=verbose))
        adjoint = _fold_resampled_latitude(gradient, parity, native, complement, phase, include_poles)
        self.assertTrue(compare_tensors("fold self-adjoint", actual_grad, adjoint, atol=2e-12, rtol=2e-12, verbose=verbose))

    @parameterized.expand(
        [(vector, "equiangular", nlat, 2 * nlat - 1, nlat - 1, nlat - 1, lmmax) for vector in (False, True) for nlat in (8, 9, 34) for lmmax in (None, 3)]
        + [(vector, "fejer1", nlat, 2 * nlat - 1, nlat, nlat, lmmax) for vector in (False, True) for nlat in (8, 9) for lmmax in (None, 3)]
        + [(vector, "equiangular", 257, 33, 256, 16, None) for vector in (False, True)],
        skip_on_empty=True,
    )
    def test_effective_projection(self, vector, grid_type, nlat, nlon, lmax, mmax, lmmax):
        """The SHT weights are the Legendre basis folded at construction.

        Compares every row, the complex contraction and its gradient against folding at
        runtime, local order offsets against the global parity, and the cache against
        shared mutable buffers."""
        set_seed(417)
        grid = th.as_grid(grid_type, nlat=nlat, nlon=nlon)
        analysis = (th.RealVectorSHT if vector else th.RealSHT)(grid, lmax=lmax, mmax=mmax, lmmax=lmmax)
        parity, native_weights, complementary_weights, phase, include_poles = _folding(grid_type, nlat, analysis.mmax, vector, self.device)
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
        another = (th.RealVectorSHT if vector else th.RealSHT)(grid, lmax=analysis.lmax, mmax=analysis.mmax, lmmax=lmmax)
        torch.testing.assert_close(analysis.weights, another.weights, atol=0, rtol=0)
        self.assertNotEqual(analysis.weights.data_ptr(), another.weights.data_ptr())
        another.weights.zero_()
        self.assertGreater(analysis.weights.abs().max().item(), 0)
        self.assertEqual(_precompute_folded_projection.cache_parameters()["maxsize"], 2)


if __name__ == "__main__":
    unittest.main()
