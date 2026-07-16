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
from testutils import compare_tensors

from torch_harmonics.wigner import wigner_D, wigner_d

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


def wigner_d_reference(l: int, beta: float, device: torch.device) -> torch.Tensor:
    r"""Independent reference for the Wigner d-matrix d^l(beta).

    Built straight from the definition d^l(beta) = exp(-i beta J_y), where J_y is
    the (2l+1) angular-momentum operator assembled from the ladder operators

        J_+ |l, m> = sqrt((l - m)(l + m + 1)) |l, m + 1>
        J_- |l, m> = sqrt((l + m)(l - m + 1)) |l, m - 1>
        J_y = (J_+ - J_-) / (2i).

    This shares no code with the recurrence in :func:`wigner_d`, so agreement is
    a genuine correctness check.  Rows/cols are ordered m = -l, ..., +l.
    """
    m = torch.arange(-l, l + 1, dtype=torch.float64, device=device)
    # super-diagonal coupling for the raising operator: <m+1| J_+ |m>
    off = torch.sqrt((l - m[:-1]) * (l + m[:-1] + 1.0))
    jp = torch.diag(off, diagonal=1).to(torch.complex128)
    jm = torch.diag(off, diagonal=-1).to(torch.complex128)
    jy = (jp - jm) / (2.0j)
    return torch.matrix_exp(-1.0j * beta * jy).real


@parameterized_class(("device",), _devices)
class TestWignerD(unittest.TestCase):
    """Correctness of the Wigner d-/D-matrix routines."""

    def _dval(self, d: torch.Tensor, lmax: int, l: int, mp: int, m: int) -> float:
        """Extract d^l_{m'm} from the band-packed tensor."""
        return d[l, mp + lmax, m + lmax].item()

    # -----------------------------------------------------------------------
    # Test 1: agreement with the exp(-i beta J_y) oracle
    #
    # For a set of fixed angles, every block d^l(beta) up to lmax must match the
    # matrix-exponential reference to double-precision tolerance.
    # -----------------------------------------------------------------------
    def test_oracle_agreement(self, verbose=False):
        lmax = 8
        betas = [0.0, math.pi / 6, math.pi / 3, math.pi / 2, 2.0 * math.pi / 3, math.pi, 1.3, 2.7]

        for beta in betas:
            d = wigner_d(lmax, torch.tensor(beta, device=self.device))
            for l in range(lmax + 1):
                ref = wigner_d_reference(l, beta, self.device)
                block = d[l, lmax - l : lmax + l + 1, lmax - l : lmax + l + 1]
                self.assertTrue(
                    compare_tensors(f"oracle l={l} beta={beta:.4f}", block, ref, atol=1e-12, rtol=1e-10, verbose=verbose),
                )

    # -----------------------------------------------------------------------
    # Test 2: targeted low-lying coefficients against analytic closed forms
    #
    # Hand-picked (l, m', m) with textbook expressions, evaluated at a couple of
    # concrete angles.  Covers the trivial l=0 entry, the m=0 column (which is
    # the Legendre polynomial P_l(cos beta)), diagonal / anti-diagonal entries,
    # and the sign- and normalization-carrying off-diagonal entries.
    # -----------------------------------------------------------------------
    @parameterized.expand([[math.pi / 3], [math.pi / 5], [0.7]])
    def test_targeted_values(self, beta, verbose=False):
        lmax = 4
        d = wigner_d(lmax, torch.tensor(beta, device=self.device))
        c = math.cos(beta)
        s = math.sin(beta)
        r2 = math.sqrt(2.0)

        # (l, m', m): analytic value
        expected = {
            # l = 0 : the trivial coefficient
            (0, 0, 0): 1.0,
            # l = 1 : m=0 column is P_1(cos b) = cos b
            (1, 0, 0): c,
            (1, 1, 1): (1.0 + c) / 2.0,  # diagonal
            (1, 1, -1): (1.0 - c) / 2.0,  # anti-diagonal
            (1, 1, 0): -s / r2,  # off-diagonal, sign + 1/sqrt2
            (1, 0, 1): s / r2,  # transpose partner, opposite sign
            (1, -1, -1): (1.0 + c) / 2.0,  # negation symmetry of (1,1,1)
            # l = 2 : m=0 column is P_2(cos b) = (3 cos^2 b - 1)/2
            (2, 0, 0): (3.0 * c * c - 1.0) / 2.0,
            (2, 2, 2): ((1.0 + c) / 2.0) ** 2,  # cos^4(b/2)
            (2, 2, -2): ((1.0 - c) / 2.0) ** 2,  # sin^4(b/2)
            (2, 2, 0): math.sqrt(6.0) / 4.0 * s * s,  # normalization factor sqrt(6)/4
            (2, 1, 0): -math.sqrt(3.0 / 8.0) * math.sin(2.0 * beta),
            # l = 3 : m=0 column is P_3(cos b) = (5 cos^3 b - 3 cos b)/2
            (3, 0, 0): (5.0 * c**3 - 3.0 * c) / 2.0,
        }

        for (l, mp, m), ref in expected.items():
            val = self._dval(d, lmax, l, mp, m)
            self.assertTrue(
                compare_tensors(
                    f"d^{l}_({mp},{m})",
                    torch.tensor(val),
                    torch.tensor(ref),
                    atol=1e-12,
                    rtol=1e-10,
                    verbose=verbose,
                )
            )

    # -----------------------------------------------------------------------
    # Test 3: d^l(0) = I
    # -----------------------------------------------------------------------
    def test_identity_at_zero(self, verbose=False):
        lmax = 6
        d = wigner_d(lmax, torch.tensor(0.0, device=self.device))
        for l in range(lmax + 1):
            block = d[l, lmax - l : lmax + l + 1, lmax - l : lmax + l + 1]
            eye = torch.eye(2 * l + 1, dtype=block.dtype, device=self.device)
            self.assertTrue(compare_tensors(f"d^{l}(0)=I", block, eye, atol=1e-12, rtol=0.0, verbose=verbose))

    # -----------------------------------------------------------------------
    # Test 4: d^l(pi)_{m'm} = (-1)^{l-m} delta_{m', -m}
    #
    # The only non-zeros sit on the anti-diagonal with a known sign.
    # -----------------------------------------------------------------------
    def test_pi_structure(self, verbose=False):
        lmax = 6
        d = wigner_d(lmax, torch.tensor(math.pi, device=self.device))
        for l in range(lmax + 1):
            block = d[l, lmax - l : lmax + l + 1, lmax - l : lmax + l + 1]
            ref = torch.zeros(2 * l + 1, 2 * l + 1, dtype=block.dtype, device=self.device)
            for m in range(-l, l + 1):
                ref[-m + l, m + l] = (-1.0) ** (l - m)
            self.assertTrue(compare_tensors(f"d^{l}(pi)", block, ref, atol=1e-12, rtol=0.0, verbose=verbose))

    # -----------------------------------------------------------------------
    # Test 5: orthogonality  d^l(beta) d^l(beta)^T = I
    # -----------------------------------------------------------------------
    @parameterized.expand([[0.4], [math.pi / 2], [2.1]])
    def test_orthogonality(self, beta, verbose=False):
        lmax = 8
        d = wigner_d(lmax, torch.tensor(beta, device=self.device))
        for l in range(lmax + 1):
            block = d[l, lmax - l : lmax + l + 1, lmax - l : lmax + l + 1]
            prod = block @ block.transpose(-1, -2)
            eye = torch.eye(2 * l + 1, dtype=block.dtype, device=self.device)
            self.assertTrue(compare_tensors(f"orthogonality l={l}", prod, eye, atol=1e-11, rtol=0.0, verbose=verbose))

    # -----------------------------------------------------------------------
    # Test 6: group law  d^l(b1) d^l(b2) = d^l(b1 + b2)
    #
    # Rotations about the same axis commute and compose additively in angle.
    # -----------------------------------------------------------------------
    def test_group_law(self, verbose=False):
        lmax = 8
        b1, b2 = 0.6, 1.15
        d1 = wigner_d(lmax, torch.tensor(b1, device=self.device))
        d2 = wigner_d(lmax, torch.tensor(b2, device=self.device))
        dsum = wigner_d(lmax, torch.tensor(b1 + b2, device=self.device))
        for l in range(lmax + 1):
            sl = slice(lmax - l, lmax + l + 1)
            prod = d1[l, sl, sl] @ d2[l, sl, sl]
            self.assertTrue(compare_tensors(f"group law l={l}", prod, dsum[l, sl, sl], atol=1e-11, rtol=1e-9, verbose=verbose))

    # -----------------------------------------------------------------------
    # Test 7: symmetry relations
    #   d^l_{m'm} = (-1)^{m'-m} d^l_{m m'}      (transpose)
    #   d^l_{m'm} = (-1)^{m'-m} d^l_{-m'-m}     (joint negation)
    # -----------------------------------------------------------------------
    def test_symmetry_relations(self, verbose=False):
        lmax = 7
        beta = 0.9
        d = wigner_d(lmax, torch.tensor(beta, device=self.device))
        for l in range(lmax + 1):
            for mp in range(-l, l + 1):
                for m in range(-l, l + 1):
                    val = self._dval(d, lmax, l, mp, m)
                    sign = (-1.0) ** (mp - m)
                    self.assertAlmostEqual(val, sign * self._dval(d, lmax, l, m, mp), places=10)
                    self.assertAlmostEqual(val, sign * self._dval(d, lmax, l, -mp, -m), places=10)

    # -----------------------------------------------------------------------
    # Test 8: batched beta matches per-angle evaluation
    # -----------------------------------------------------------------------
    def test_batched_beta(self, verbose=False):
        lmax = 5
        betas = torch.tensor([0.2, 1.1, 2.5, 3.0], device=self.device)
        d_batched = wigner_d(lmax, betas)
        for i, beta in enumerate(betas.tolist()):
            d_single = wigner_d(lmax, torch.tensor(beta, device=self.device))
            self.assertTrue(compare_tensors(f"batched[{i}]", d_batched[i], d_single, atol=1e-12, rtol=0.0, verbose=verbose))

    # -----------------------------------------------------------------------
    # Test 9: Wigner D reduces to d at alpha = gamma = 0, and is unitary
    # -----------------------------------------------------------------------
    def test_wigner_D_reduction_and_unitarity(self, verbose=False):
        lmax = 6
        alpha, beta, gamma = 0.5, 1.3, 2.2

        # alpha = gamma = 0  =>  D = d (real)
        zero = torch.tensor(0.0, device=self.device)
        D0 = wigner_D(lmax, zero, torch.tensor(beta, device=self.device), zero)
        d = wigner_d(lmax, torch.tensor(beta, device=self.device)).to(D0.dtype)
        self.assertTrue(compare_tensors("D(0,b,0)=d", D0, d, atol=1e-12, rtol=0.0, verbose=verbose))

        # unitarity: D D^H = I on each block
        D = wigner_D(
            lmax,
            torch.tensor(alpha, device=self.device),
            torch.tensor(beta, device=self.device),
            torch.tensor(gamma, device=self.device),
        )
        for l in range(lmax + 1):
            sl = slice(lmax - l, lmax + l + 1)
            block = D[l, sl, sl]
            prod = block @ block.conj().transpose(-1, -2)
            eye = torch.eye(2 * l + 1, dtype=block.dtype, device=self.device)
            self.assertTrue(compare_tensors(f"D unitary l={l}", prod, eye, atol=1e-11, rtol=0.0, verbose=verbose))


if __name__ == "__main__":
    unittest.main()
