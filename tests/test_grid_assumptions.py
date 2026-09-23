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

"""
Contract tests for grid-dependent quantities that are currently derived from
``nlat`` plus a grid *string*, in several places independently.

These pin down the invariants that a future grid descriptor is meant to own as
properties, so that the refactor can be validated against them:

* the default angular cutoff, which must follow the grid's actual latitudinal node
  spacing rather than ``nlat`` alone (:func:`compute_theta_cutoff`),
* whether the nodes of a grid are equispaced in :math:`\\theta` or in
  :math:`\\cos\\theta`,
* the pole symmetry of nodes and weights, which is what currently makes the
  differing flip conventions of ``precompute_latitudes`` and ``QuadratureS2``
  agree by accident.
"""

import math
import unittest
import warnings

import torch
from parameterized import parameterized
from testutils import compare_tensors

from torch_harmonics.disco.convolution import _precompute_convolution_tensor_s2
from torch_harmonics.filter_basis import get_filter_basis
from torch_harmonics.quadrature import compute_latitude_spacing, compute_theta_cutoff, precompute_latitudes

_ALL_GRIDS = ["equiangular", "legendre-gauss", "lobatto", "equiangular-trapezoidal"]

# grids on which the superseded pi / (nlat - 1) heuristic was too narrow near the poles
_IRREGULAR_THETA_GRIDS = ["lobatto", "equiangular-trapezoidal"]

_NLATS = [33, 65, 129]


def _legacy_theta_cutoff(nlat: int) -> float:
    """The nlat-only heuristic that compute_theta_cutoff replaced."""
    return math.pi / float(nlat - 1)


def _default_theta_cutoff(nlat: int, grid: str) -> float:
    """The cutoff actually used by DiscreteContinuousConvS2 and NeighborhoodAttentionS2."""
    return compute_theta_cutoff(nlat, grid=grid)


def _min_latitude_rings_in_cutoff(nlat: int, grid: str) -> int:
    """
    Minimum number of input latitude rings that fall within the default cutoff of
    any output latitude, for a same-in/same-out grid.

    The great-circle distance between ``(theta_out, 0)`` and ``(theta_in, phi)`` is
    bounded below by ``|theta_out - theta_in|``, so a latitude ring outside the
    cutoff in ``theta`` cannot contribute to the psi row for any ``phi``. A value of
    1 therefore means the neighborhood of that output point degenerates to the
    single ring it sits on.
    """
    lats, _ = precompute_latitudes(nlat, grid=grid)
    cutoff = _default_theta_cutoff(nlat, grid)
    # count ties: an input latitude exactly at the cutoff is a boundary artifact of
    # the strict comparison in _precompute_convolution_tensor_s2, not a real gap
    within = (lats.unsqueeze(0) - lats.unsqueeze(1)).abs() <= cutoff * (1.0 + 1e-9)
    return int(within.sum(dim=1).min().item())


class TestThetaCutoffContract(unittest.TestCase):
    """
    ``theta_cutoff`` defaults to :func:`compute_theta_cutoff`, which takes one
    latitudinal grid spacing from the grid's actual node distribution. It replaced
    a hardcoded ``pi / (nlat - 1)``, which is the exact node spacing of an
    *equiangular* (Clenshaw-Curtis) grid but a significant underestimate near the
    poles for ``lobatto`` and ``equiangular-trapezoidal``.
    """

    @parameterized.expand([[nlat, grid] for nlat in _NLATS for grid in _ALL_GRIDS])
    def test_default_cutoff_covers_latitude_spacing(self, nlat, grid):
        cutoff = _default_theta_cutoff(nlat, grid)
        dlat_max = compute_latitude_spacing(nlat, grid=grid)
        self.assertGreaterEqual(
            cutoff * (1.0 + 1e-9),
            dlat_max,
            msg=f"grid={grid} nlat={nlat}: default theta_cutoff {cutoff:.6f} < max latitude spacing {dlat_max:.6f}",
        )

    @parameterized.expand([[nlat, grid] for nlat in _NLATS for grid in _ALL_GRIDS])
    def test_default_cutoff_gives_multi_ring_neighborhood(self, nlat, grid):
        rings = _min_latitude_rings_in_cutoff(nlat, grid)
        self.assertGreaterEqual(
            rings,
            2,
            msg=f"grid={grid} nlat={nlat}: some output latitude sees only its own ring within the default cutoff",
        )

    @parameterized.expand([[nlat] for nlat in _NLATS])
    def test_equiangular_cutoff_is_unchanged_by_the_fix(self, nlat):
        """
        Regression guard: the equiangular grid is the default and by far the most
        used one, so switching to the node-distribution-based cutoff must not
        perturb it. The two agree to ~1e-13 relative, not bit-identically, since the
        new value comes back through ``arccos`` of the Clenshaw-Curtis nodes.

        See :meth:`TestThetaCutoffPsiRegression.test_equiangular_psi_is_unchanged`
        for the consequence that actually matters.
        """
        self.assertAlmostEqual(_default_theta_cutoff(nlat, "equiangular") / _legacy_theta_cutoff(nlat), 1.0, places=12)

    @parameterized.expand([[nlat, grid] for nlat in _NLATS for grid in _IRREGULAR_THETA_GRIDS])
    def test_irregular_grids_get_a_wider_cutoff_than_the_legacy_heuristic(self, nlat, grid):
        """
        The actual fix: on these two grids the legacy heuristic was too narrow, so
        the new cutoff must be strictly wider. Lobatto by ~21%, and
        equiangular-trapezoidal by ~5x since its nodes are equispaced in cos(theta).
        """
        self.assertGreater(_default_theta_cutoff(nlat, grid), _legacy_theta_cutoff(nlat))

    @parameterized.expand([[nlat, grid] for nlat in _NLATS for grid in _ALL_GRIDS])
    def test_scale_is_applied_linearly(self, nlat, grid):
        self.assertAlmostEqual(compute_theta_cutoff(nlat, grid=grid, scale=2.5), 2.5 * compute_theta_cutoff(nlat, grid=grid), places=15)

    @parameterized.expand([[nlat, grid] for nlat in _NLATS for grid in _IRREGULAR_THETA_GRIDS + ["legendre-gauss"]])
    def test_changed_default_warns(self, nlat, grid):
        """
        Mirrors the ``truncate_sht`` precedent: grids whose default moved must say
        so, since a silently different cutoff would silently change existing models.
        """
        with self.assertWarns(UserWarning):
            compute_theta_cutoff(nlat, grid=grid)

    @parameterized.expand([[nlat] for nlat in _NLATS])
    def test_unchanged_default_does_not_warn(self, nlat):
        """The equiangular default is unchanged, so warning there would be noise."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            compute_theta_cutoff(nlat, grid="equiangular")


class TestThetaCutoffPsiRegression(unittest.TestCase):
    """
    The cutoff only matters through the convolution tensor it produces, so check
    that directly: on the equiangular grid, swapping the legacy ``pi / (nlat - 1)``
    for :func:`compute_theta_cutoff` must leave the sparsity pattern of psi
    untouched and its values equal to well within float32 resolution. Existing
    equiangular models are therefore unaffected by the fix.
    """

    @parameterized.expand([[nlat, nlon] for (nlat, nlon) in [(33, 64), (65, 128), (129, 256)]])
    def test_equiangular_psi_is_unchanged(self, nlat, nlon, verbose=False):
        filter_basis = get_filter_basis(kernel_shape=(3, 3), basis_type="piecewise linear")

        def _psi(theta_cutoff):
            idx, vals, _ = _precompute_convolution_tensor_s2(
                (nlat, nlon),
                (nlat, nlon),
                filter_basis,
                grid_in="equiangular",
                grid_out="equiangular",
                theta_cutoff=theta_cutoff,
                transpose_normalization=False,
                basis_norm_mode="mean",
                merge_quadrature=True,
            )
            return idx, vals

        idx_legacy, vals_legacy = _psi(_legacy_theta_cutoff(nlat))
        idx_new, vals_new = _psi(_default_theta_cutoff(nlat, "equiangular"))

        self.assertEqual(idx_legacy.shape, idx_new.shape, msg=f"nlat={nlat}: psi nnz changed, {idx_legacy.shape[1]} -> {idx_new.shape[1]}")
        self.assertTrue(compare_tensors(f"psi sparsity pattern (nlat={nlat})", idx_legacy, idx_new, verbose=verbose))
        # float32 eps is ~1.2e-7, so 1e-10 leaves several orders of headroom
        self.assertTrue(compare_tensors(f"psi values (nlat={nlat})", vals_legacy, vals_new, atol=1e-10, rtol=1e-10, verbose=verbose))


class TestGridNodeDistribution(unittest.TestCase):
    """
    Characterization tests for *where* the nodes of each grid actually sit. These
    encode facts that are currently implicit in the grid string and that a
    descriptor should expose explicitly.
    """

    @parameterized.expand([[nlat] for nlat in _NLATS])
    def test_equiangular_nodes_are_equispaced_in_theta(self, nlat, verbose=False):
        lats, _ = precompute_latitudes(nlat, grid="equiangular")
        dlat = lats[1:] - lats[:-1]
        self.assertTrue(compare_tensors(f"equiangular dtheta (nlat={nlat})", dlat, dlat.mean().expand_as(dlat), atol=1e-12, rtol=0.0, verbose=verbose))

    @parameterized.expand([[nlat] for nlat in _NLATS])
    def test_equiangular_trapezoidal_nodes_are_equispaced_in_cos_theta(self, nlat, verbose=False):
        """
        Despite its name, ``equiangular-trapezoidal`` is *not* equiangular in theta:
        ``precompute_latitudes`` builds it via ``trapezoidal_weights`` on the
        cos(theta) interval [-1, 1], so the nodes are equispaced in cos(theta)
        instead. This is the root cause of the polar under-coverage above, and it is
        exactly the kind of fact a grid descriptor should carry rather than a name.
        """
        lats, _ = precompute_latitudes(nlat, grid="equiangular-trapezoidal")
        cost = torch.cos(lats)
        dcos = cost[1:] - cost[:-1]
        self.assertTrue(compare_tensors(f"equiangular-trapezoidal dcos(theta) (nlat={nlat})", dcos, dcos.mean().expand_as(dcos), atol=1e-12, rtol=0.0, verbose=verbose))

        # ... and correspondingly it is strongly non-uniform in theta
        dlat = lats[1:] - lats[:-1]
        self.assertGreater(dlat.max().item() / dlat.min().item(), 4.0)

    @parameterized.expand([[nlat, grid] for nlat in _NLATS for grid in _ALL_GRIDS])
    def test_latitudes_are_ascending_colatitudes(self, nlat, grid):
        """All grids must return colatitudes in [0, pi], strictly ascending (north pole first)."""
        lats, _ = precompute_latitudes(nlat, grid=grid)
        self.assertGreaterEqual(lats[0].item(), 0.0)
        self.assertLessEqual(lats[-1].item(), math.pi)
        self.assertTrue(bool(((lats[1:] - lats[:-1]) > 0).all()), msg=f"grid={grid} nlat={nlat}: latitudes are not strictly ascending")


class TestQuadratureOrderingContract(unittest.TestCase):
    """
    ``precompute_latitudes`` flips both nodes and weights out of the cos(theta)
    domain, whereas ``QuadratureS2`` (``quadrature.py:403``) uses the *unflipped*
    weights directly against data indexed by the flipped latitudes. Both are
    correct today only because every currently supported quadrature rule has nodes
    and weights that are symmetric about the equator, which makes the discrepancy
    invisible.

    A grid descriptor hands out ``lats`` and ``quad_weights`` as a pair, so this
    coincidence has to become an explicit, checked property: any newly added rule
    that is not pole-symmetric would silently break ``QuadratureS2`` today.
    """

    @parameterized.expand([[nlat, grid] for nlat in [64, 65] for grid in _ALL_GRIDS])
    def test_quadrature_weights_are_pole_symmetric(self, nlat, grid, verbose=False):
        _, w = precompute_latitudes(nlat, grid=grid)
        self.assertTrue(compare_tensors(f"weight pole symmetry (grid={grid}, nlat={nlat})", w, torch.flip(w, dims=(0,)), atol=1e-12, rtol=0.0, verbose=verbose))

    @parameterized.expand([[nlat, grid] for nlat in [64, 65] for grid in _ALL_GRIDS])
    def test_latitudes_are_pole_symmetric(self, nlat, grid, verbose=False):
        """Companion to the above: ``theta_k + theta_{n-1-k} == pi``."""
        lats, _ = precompute_latitudes(nlat, grid=grid)
        self.assertTrue(
            compare_tensors(
                f"latitude pole symmetry (grid={grid}, nlat={nlat})",
                lats,
                math.pi - torch.flip(lats, dims=(0,)),
                atol=1e-12,
                rtol=0.0,
                verbose=verbose,
            )
        )

    @parameterized.expand([[nlat, grid] for nlat in [65, 129] for grid in _ALL_GRIDS])
    def test_weights_integrate_asymmetric_field(self, nlat, grid, verbose=False):
        r"""
        :math:`\int_{S^2} e^{\cos\theta}\,dA = 2\pi \int_{-1}^{1} e^x dx = 2\pi(e - e^{-1})`.

        A pole-asymmetric, monotone integrand, so this exercises the node/weight
        pairing rather than just the total mass. The tolerance is loose because
        equiangular-trapezoidal is only second-order accurate here.
        """
        lats, w = precompute_latitudes(nlat, grid=grid)
        integral = 2.0 * math.pi * torch.sum(w * torch.exp(torch.cos(lats)))
        expected = torch.full_like(integral, 2.0 * math.pi * (math.e - 1.0 / math.e))
        self.assertTrue(compare_tensors(f"exp(cos theta) integral (grid={grid}, nlat={nlat})", integral, expected, atol=0.0, rtol=1e-3, verbose=verbose))


if __name__ == "__main__":
    unittest.main()
