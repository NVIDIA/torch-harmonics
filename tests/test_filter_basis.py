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
from parameterized import parameterized
import math
import torch

from torch_harmonics.filter_basis import get_filter_basis


# quadrature grid on the disk for integration tests
def _disk_quadrature(r_cutoff, nr=100, nphi=400):
    """Midpoint-rule quadrature on the disk: returns (r, phi, dr, dphi)."""
    dr = r_cutoff / nr
    dphi = 2 * math.pi / nphi
    r = (torch.arange(nr, dtype=torch.float64) + 0.5) * dr
    phi = torch.arange(nphi, dtype=torch.float64) * dphi
    r, phi = torch.meshgrid(r, phi, indexing="ij")
    return r, phi, dr, dphi


def _eval_dense(basis, r, phi, r_cutoff):
    """Evaluate all basis functions on a dense grid, returning (kernel_size, nr, nphi)."""
    iidx, vals = basis.compute_support_vals(r, phi, r_cutoff=r_cutoff)
    nr, nphi = r.shape
    psi = torch.sparse_coo_tensor(iidx.t(), vals, size=(basis.kernel_size, nr, nphi)).to_dense()
    return psi


class TestFilterBasis(unittest.TestCase):
    """Tests for filter basis analytical properties."""

    # ------------------------------------------------------------------
    # L2 normalization: ||psi_i||_L2 ~ 1 for normalized bases
    # ------------------------------------------------------------------
    # The piecewise-linear basis is intentionally excluded: it is a partition-of-unity
    # interpolation basis (its defining property is that the hat functions sum to 1 in
    # the interior support), not an L2-orthonormal expansion. Imposing unit L2 norm on
    # its hats would break the partition-of-unity property that test_piecewise_linear_partition_of_unity
    # checks.
    @parameterized.expand(
        [
            # basis_type, kernel_shape, r_cutoff
            ["harmonic", (2, 2), 0.5],
            ["harmonic", (3, 3), 0.3],
            ["zernike", (3,), 0.5],
            ["zernike", (4,), 0.3],
            ["fourier-bessel", (3, 3), 0.5],
            ["fourier-bessel", (2, 2), 0.3],
        ],
        skip_on_empty=True,
    )
    def test_l2_normalization(self, basis_type, kernel_shape, r_cutoff, tol=1e-2):
        """L2-normalized bases should have ||psi_i||_L2 ~ 1 for all i."""
        basis = get_filter_basis(kernel_shape=kernel_shape, basis_type=basis_type)
        r, phi, dr, dphi = _disk_quadrature(r_cutoff, nr=150, nphi=600)
        psi = _eval_dense(basis, r, phi, r_cutoff)

        for i in range(basis.kernel_size):
            norm_sq = (psi[i] ** 2 * r * dr * dphi).sum()
            norm = norm_sq.sqrt().item()
            self.assertAlmostEqual(norm, 1.0, delta=tol,
                                   msg=f"basis {basis_type} k={i}: ||psi||={norm:.6f}, expected ~1.0")

    # ------------------------------------------------------------------
    # Orthogonality: <psi_i, psi_j> ~ 0 for i != j
    # ------------------------------------------------------------------
    @parameterized.expand(
        [
            ["harmonic", (2, 2), 0.5],
            ["harmonic", (3, 3), 0.3],
            ["zernike", (3,), 0.5],
            ["zernike", (4,), 0.3],
            ["fourier-bessel", (3, 3), 0.5],
            ["fourier-bessel", (2, 2), 0.3],
        ],
        skip_on_empty=True,
    )
    def test_orthogonality(self, basis_type, kernel_shape, r_cutoff, tol=5e-2):
        """Inner products <psi_i, psi_j> should be ~0 for i != j on orthogonal bases."""
        basis = get_filter_basis(kernel_shape=kernel_shape, basis_type=basis_type)
        r, phi, dr, dphi = _disk_quadrature(r_cutoff, nr=150, nphi=600)
        psi = _eval_dense(basis, r, phi, r_cutoff)

        K = basis.kernel_size
        # compute Gram matrix G_ij = <psi_i, psi_j> with disk measure r dr dphi
        weight = r * dr * dphi
        gram = torch.einsum("inm,jnm,nm->ij", psi, psi, weight)

        # off-diagonal should be near zero
        off_diag = gram - torch.diag(gram.diag())
        max_off = off_diag.abs().max().item()
        self.assertLess(max_off, tol,
                        msg=f"basis {basis_type}: max off-diagonal inner product = {max_off:.6f}")

    # ------------------------------------------------------------------
    # kernel_size consistency: number of returned basis functions matches kernel_size
    # ------------------------------------------------------------------
    @parameterized.expand(
        [
            ["piecewise linear", (3,)],
            ["piecewise linear", (3, 3)],
            ["piecewise linear", (4, 2)],
            ["harmonic", (2, 2)],
            ["harmonic", (3, 3)],
            ["zernike", (3,)],
            ["zernike", (5,)],
            ["fourier-bessel", (3, 3)],
        ],
        skip_on_empty=True,
    )
    def test_kernel_size_consistency(self, basis_type, kernel_shape):
        """compute_support_vals should return exactly kernel_size distinct basis functions."""
        basis = get_filter_basis(kernel_shape=kernel_shape, basis_type=basis_type)
        r_cutoff = 0.5
        r, phi, _, _ = _disk_quadrature(r_cutoff, nr=50, nphi=200)
        iidx, vals = basis.compute_support_vals(r, phi, r_cutoff=r_cutoff)

        # kernel indices returned should span [0, kernel_size)
        kernel_ids = iidx[:, 0].unique()
        self.assertEqual(len(kernel_ids), basis.kernel_size,
                         msg=f"{basis_type}: expected {basis.kernel_size} basis functions, got {len(kernel_ids)}")
        self.assertEqual(kernel_ids.min().item(), 0)
        self.assertEqual(kernel_ids.max().item(), basis.kernel_size - 1)

    # ------------------------------------------------------------------
    # Support: no values outside the cutoff radius
    # ------------------------------------------------------------------
    @parameterized.expand(
        [
            ["piecewise linear", (3,), 0.3],
            ["piecewise linear", (3, 3), 0.5],
            ["harmonic", (3, 3), 0.3],
            ["zernike", (3,), 0.5],
            ["fourier-bessel", (3, 3), 0.5],
        ],
        skip_on_empty=True,
    )
    def test_support_within_cutoff(self, basis_type, kernel_shape, r_cutoff):
        """All returned values should come from r <= r_cutoff."""
        basis = get_filter_basis(kernel_shape=kernel_shape, basis_type=basis_type)
        # use a grid that extends beyond r_cutoff
        r_max = r_cutoff * 2.0
        nr, nphi = 100, 200
        dr = r_max / nr
        dphi = 2 * math.pi / nphi
        r = (torch.arange(nr, dtype=torch.float64) + 0.5) * dr
        phi = torch.arange(nphi, dtype=torch.float64) * dphi
        r, phi = torch.meshgrid(r, phi, indexing="ij")

        iidx, vals = basis.compute_support_vals(r, phi, r_cutoff=r_cutoff)
        r_vals = r[iidx[:, 1], iidx[:, 2]]
        self.assertTrue((r_vals <= r_cutoff).all(),
                        msg=f"{basis_type}: found values at r > r_cutoff")

    # ------------------------------------------------------------------
    # Isotropy: isotropic basis functions should be rotationally invariant
    # ------------------------------------------------------------------
    @parameterized.expand(
        [
            ["piecewise linear", (3,), 0.5],
            ["piecewise linear", (3, 3), 0.3],
            ["harmonic", (3, 3), 0.3],
            ["zernike", (3,), 0.5],
            ["fourier-bessel", (3, 3), 0.5],
        ],
        skip_on_empty=True,
    )
    def test_isotropic_invariance(self, basis_type, kernel_shape, r_cutoff, tol=1e-10):
        """Basis functions marked isotropic should not depend on phi."""
        basis = get_filter_basis(kernel_shape=kernel_shape, basis_type=basis_type)

        # evaluate on two phi slices at the same radii
        nr = 50
        dr = r_cutoff / nr
        r1d = (torch.arange(nr, dtype=torch.float64) + 0.5) * dr
        phi_a = torch.full_like(r1d, 0.0)
        phi_b = torch.full_like(r1d, math.pi / 3)

        r_a = r1d.unsqueeze(1)
        r_b = r1d.unsqueeze(1)
        phi_a = phi_a.unsqueeze(1)
        phi_b = phi_b.unsqueeze(1)

        psi_a = _eval_dense(basis, r_a, phi_a, r_cutoff)
        psi_b = _eval_dense(basis, r_b, phi_b, r_cutoff)

        for k, is_iso in enumerate(basis.isotropic_mask):
            if is_iso:
                diff = (psi_a[k] - psi_b[k]).abs().max().item()
                self.assertLess(diff, tol,
                                msg=f"{basis_type} k={k}: isotropic basis varies with phi, max diff={diff:.2e}")

    # ------------------------------------------------------------------
    # Fourier-Bessel: boundary condition psi(r_cutoff, phi) = 0
    # ------------------------------------------------------------------
    @parameterized.expand(
        [
            [(2, 2), 0.3],
            [(3, 3), 0.5],
        ],
        skip_on_empty=True,
    )
    def test_fourier_bessel_boundary(self, kernel_shape, r_cutoff, tol=1e-10):
        """Fourier-Bessel basis should vanish at r = r_cutoff (Dirichlet BC)."""
        basis = get_filter_basis(kernel_shape=kernel_shape, basis_type="fourier-bessel")

        nphi = 100
        dphi = 2 * math.pi / nphi
        r = torch.full((1, nphi), r_cutoff, dtype=torch.float64)
        phi = (torch.arange(nphi, dtype=torch.float64) * dphi).unsqueeze(0)

        psi = _eval_dense(basis, r, phi, r_cutoff * 1.01)  # slightly larger cutoff to include boundary

        for k in range(basis.kernel_size):
            max_val = psi[k].abs().max().item()
            self.assertLess(max_val, tol,
                            msg=f"fourier-bessel k={k}: |psi(r_cutoff)|={max_val:.2e}, expected ~0")

    # ------------------------------------------------------------------
    # Zernike: R_n^m(1) = 1 for all (n, m)
    # ------------------------------------------------------------------
    @parameterized.expand(
        [
            [(3,)],
            [(4,)],
            [(5,)],
        ],
        skip_on_empty=True,
    )
    def test_zernike_radial_at_boundary(self, kernel_shape, tol=1e-10):
        """Zernike radial polynomial R_n^|m|(1) should equal 1."""
        basis = get_filter_basis(kernel_shape=kernel_shape, basis_type="zernike")

        # evaluate R_n^m at r=1 by setting r=r_cutoff and phi=0
        r_cutoff = 1.0
        r = torch.ones(1, 1, dtype=torch.float64)
        phi = torch.zeros(1, 1, dtype=torch.float64)

        iidx, vals = basis.compute_support_vals(r, phi, r_cutoff=r_cutoff)

        # group by kernel index to check each basis function
        nmax = kernel_shape[0] if isinstance(kernel_shape, tuple) else kernel_shape
        idx = 0
        for n in range(nmax):
            for l in range(n + 1):
                m = 2 * l - n
                # at r=1, phi=0: Z_n^m = R_n^|m|(1) * cos(m*0) = R_n^|m|(1) * 1 for m >= 0
                # for m < 0: Z_n^m = R_n^|m|(1) * sin(m*0) = 0
                mask = iidx[:, 0] == idx
                if mask.any():
                    v = vals[mask].item()
                    # L2 normalization factor
                    epsilon_m = 2.0 if m == 0 else 1.0
                    norm = math.sqrt(math.pi * epsilon_m / (2.0 * (n + 1)))
                    if m >= 0:
                        # R_n^m(1) = 1, angular = cos(m*0) = 1, so val = 1/norm
                        expected = 1.0 / norm
                        self.assertAlmostEqual(v, expected, delta=tol,
                                               msg=f"zernike (n={n},m={m}): R(1)={v*norm:.6f}, expected 1.0")
                    else:
                        # angular = sin(m*0) = 0
                        self.assertAlmostEqual(v, 0.0, delta=tol,
                                               msg=f"zernike (n={n},m={m}): expected 0 at phi=0")
                idx += 1

    # ------------------------------------------------------------------
    # Piecewise linear: partition of unity for isotropic case
    # ------------------------------------------------------------------
    @parameterized.expand(
        [
            [(3,), 0.3],
            [(5,), 0.5],
            [(7,), 0.4],
        ],
        skip_on_empty=True,
    )
    def test_piecewise_linear_partition_of_unity(self, kernel_shape, r_cutoff, tol=1e-10):
        """Isotropic piecewise-linear basis should sum to ~1 in the interior of the support."""
        basis = get_filter_basis(kernel_shape=kernel_shape, basis_type="piecewise linear")
        nr = 200
        dr = r_cutoff / nr
        r = (torch.arange(nr, dtype=torch.float64) + 0.5) * dr
        phi = torch.zeros_like(r)  # isotropic, phi doesn't matter

        r = r.unsqueeze(1)
        phi = phi.unsqueeze(1)

        psi = _eval_dense(basis, r, phi, r_cutoff)
        total = psi.sum(dim=0).squeeze()

        # in the interior (away from edges where partial support is expected),
        # the sum should be close to 1
        interior = r.squeeze() < r_cutoff * 0.9
        interior_sum = total[interior]
        max_err = (interior_sum - 1.0).abs().max().item()
        self.assertLess(max_err, tol,
                        msg=f"piecewise linear {kernel_shape}: partition-of-unity error = {max_err:.2e}")

    # ------------------------------------------------------------------
    # isotropic_mask length matches kernel_size
    # ------------------------------------------------------------------
    @parameterized.expand(
        [
            ["piecewise linear", (3,)],
            ["piecewise linear", (3, 3)],
            ["harmonic", (3, 3)],
            ["zernike", (4,)],
            ["fourier-bessel", (3, 3)],
        ],
        skip_on_empty=True,
    )
    def test_isotropic_mask_length(self, basis_type, kernel_shape):
        """isotropic_mask should have exactly kernel_size entries."""
        basis = get_filter_basis(kernel_shape=kernel_shape, basis_type=basis_type)
        self.assertEqual(len(basis.isotropic_mask), basis.kernel_size)


if __name__ == "__main__":
    unittest.main()
