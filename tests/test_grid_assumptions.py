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

import functools
import inspect
import math
import unittest
import warnings

import torch
from parameterized import parameterized
from testutils import compare_tensors

from torch_harmonics.disco.convolution import _precompute_convolution_tensor_s2
from torch_harmonics.distributed.primitives import split_tensor_along_dim
from torch_harmonics.filter_basis import get_filter_basis
from torch_harmonics.grid import (
    _GRID_REGISTRY,
    EquiangularGrid,
    GridS2,
    GridShardS2,
    LegendreGaussGrid,
    LobattoGrid,
    PointSetS2,
    RegularGridS2,
    TrapezoidalGrid,
    as_grid,
    grid_params,
    grid_types,
    require_grid,
    require_point_set,
    require_regular_grid,
)
from torch_harmonics.integration import QuadratureS2
from torch_harmonics.partition import compute_split_shapes
from torch_harmonics.quadrature import compute_latitude_spacing, compute_theta_cutoff, precompute_latitudes, precompute_longitudes
from torch_harmonics.truncation import truncate_support

_ALL_GRIDS = ["equiangular", "legendre-gauss", "lobatto", "trapezoidal"]

# grids on which the superseded pi / (nlat - 1) heuristic was too narrow near the poles
_IRREGULAR_THETA_GRIDS = ["lobatto", "trapezoidal"]

# grids whose node set includes both poles, so colat runs the full [0, pi] and the
# endpoints pin the lat/colat convention exactly. Gauss-Legendre nodes are interior.
_POLE_INCLUSIVE_GRIDS = ["equiangular", "lobatto", "trapezoidal"]

# the class a caller would instantiate directly, against the name as_grid resolves
_DIRECT_CLASSES = {
    "equiangular": EquiangularGrid,
    "legendre-gauss": LegendreGaussGrid,
    "lobatto": LobattoGrid,
    "trapezoidal": TrapezoidalGrid,
}


_NLATS = [33, 65, 129]

# shapes used to compare the two construction routes; includes the smallest legal grid
_PAIR_SHAPES = [(32, 64), (33, 64), (2, 1)]


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
    poles for ``lobatto`` and ``trapezoidal``.
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
        trapezoidal by ~5x since its nodes are equispaced in cos(theta).
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
                as_grid("equiangular", nlat=nlat, nlon=nlon),
                as_grid("equiangular", nlat=nlat, nlon=nlon),
                filter_basis,
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
        Despite its name, ``trapezoidal`` is *not* equiangular in theta:
        ``precompute_latitudes`` builds it via ``trapezoidal_weights`` on the
        cos(theta) interval [-1, 1], so the nodes are equispaced in cos(theta)
        instead. This is the root cause of the polar under-coverage above, and it is
        exactly the kind of fact a grid descriptor should carry rather than a name.
        """
        lats, _ = precompute_latitudes(nlat, grid="trapezoidal")
        cost = torch.cos(lats)
        dcos = cost[1:] - cost[:-1]
        self.assertTrue(compare_tensors(f"trapezoidal dcos(theta) (nlat={nlat})", dcos, dcos.mean().expand_as(dcos), atol=1e-12, rtol=0.0, verbose=verbose))

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

    @parameterized.expand([[nlat, grid] for nlat in _NLATS for grid in _ALL_GRIDS])
    def test_lats_is_latitude_and_colats_is_colatitude(self, nlat, grid):
        r"""
        ``colats`` is :math:`\theta \in [0, \pi]` from the north pole; ``lats`` is
        geographic latitude :math:`\pi/2 - \theta \in [-\pi/2, \pi/2]`.

        Pinned per grid rather than left to the docstring because the two have the
        same shape and dtype and differ only in offset, so confusing them produces
        plausible numbers rather than an error. The specific trap is
        :math:`\pi - \theta`, which also lands in a familiar range: it is checked
        explicitly below so that writing it would fail here rather than in a
        downstream plot.
        """
        g = as_grid(grid, nlat=nlat, nlon=2 * nlat)
        colats, lats = g.colats, g.lats

        self.assertTrue(torch.allclose(lats, math.pi / 2 - colats))
        self.assertGreaterEqual(colats[0].item(), 0.0)
        self.assertLessEqual(colats[-1].item(), math.pi)
        self.assertGreaterEqual(lats.min().item(), -math.pi / 2 - 1e-12)
        self.assertLessEqual(lats.max().item(), math.pi / 2 + 1e-12)

        # colatitude ascends from the north pole, so latitude descends
        self.assertTrue(bool((colats.diff() > 0).all()))
        self.assertTrue(bool((lats.diff() < 0).all()))

        # the near-miss conversion, rejected explicitly
        self.assertFalse(torch.allclose(lats, math.pi - colats))

    @parameterized.expand([[grid] for grid in _POLE_INCLUSIVE_GRIDS])
    def test_the_poles_map_to_plus_and_minus_ninety_degrees(self, grid):
        """On the grids that carry the poles, the endpoints pin the convention exactly."""
        g = as_grid(grid, nlat=7, nlon=8)
        self.assertAlmostEqual(g.colats[0].item(), 0.0, places=12)
        self.assertAlmostEqual(g.colats[-1].item(), math.pi, places=12)
        self.assertAlmostEqual(g.lats[0].item(), math.pi / 2, places=12)
        self.assertAlmostEqual(g.lats[-1].item(), -math.pi / 2, places=12)


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
        trapezoidal is only second-order accurate here.
        """
        lats, w = precompute_latitudes(nlat, grid=grid)
        integral = 2.0 * math.pi * torch.sum(w * torch.exp(torch.cos(lats)))
        expected = torch.full_like(integral, 2.0 * math.pi * (math.e - 1.0 / math.e))
        self.assertTrue(compare_tensors(f"exp(cos theta) integral (grid={grid}, nlat={nlat})", integral, expected, atol=0.0, rtol=1e-3, verbose=verbose))


class TestGridDescriptor(unittest.TestCase):
    """
    Contract for :class:`torch_harmonics.grid.GridS2`.

    The descriptor is meant to become the single argument layers take in place of
    ``(nlat, nlon, grid)``. These tests fix the properties the rest of the codebase
    will rely on once that migration happens.
    """

    @parameterized.expand([[grid] for grid in _ALL_GRIDS])
    def test_as_grid_builds_from_a_name_and_parameters(self, grid):
        g = as_grid(grid, nlat=64, nlon=128)
        self.assertEqual(g.grid_type, grid)
        self.assertEqual(g.shape, (64, 128))
        self.assertEqual((g.nlat, g.nlon), (64, 128))

    @parameterized.expand([[grid] for grid in _ALL_GRIDS])
    def test_as_grid_is_idempotent(self, grid):
        g = as_grid(grid, nlat=64, nlon=128)
        self.assertIs(as_grid(g), g)
        self.assertIs(as_grid(g, nlat=64, nlon=128), g)

    @parameterized.expand([[grid] for grid in _ALL_GRIDS])
    def test_as_grid_accepts_the_class_as_a_spec(self, grid):
        self.assertEqual(as_grid(_DIRECT_CLASSES[grid], nlat=64, nlon=128), as_grid(grid, nlat=64, nlon=128))

    def test_as_grid_rejects_bad_specs(self):
        with self.assertRaises(ValueError):
            as_grid("not-a-grid", nlat=64, nlon=128)
        with self.assertRaises(ValueError):
            as_grid("equiangular")  # parameters are required for a string spec
        with self.assertRaises(ValueError):
            as_grid("equiangular", nlat=64)  # nlon missing
        with self.assertRaises(ValueError):
            as_grid(as_grid("equiangular", nlat=64, nlon=128), nlat=32)  # contradicts the descriptor

    def test_as_grid_rejects_a_parameter_the_grid_does_not_take(self):
        """
        The point of the keyword factory: a parameter that is meaningless for a
        grid family must be refused, not silently ignored. A future HEALPix or
        icosahedral grid takes a refinement level rather than ``(nlat, nlon)``, and
        accepting ``nlon`` there would build a grid that is not the one asked for.
        """
        with self.assertRaises(ValueError) as ctx:
            as_grid("equiangular", nlat=64, nlon=128, nside=16)
        self.assertIn("nside", str(ctx.exception))
        self.assertIn("nlat", str(ctx.exception))  # names what the grid does take

    def test_as_grid_suggests_a_near_miss(self):
        for spec, params, expected in [("equiangulr", dict(nlat=64, nlon=128), "equiangular"), ("equiangular", dict(nlat=64, nlong=128), "nlon")]:
            with self.subTest(spec=spec):
                with self.assertRaises(ValueError) as ctx:
                    as_grid(spec, **params)
                self.assertIn(expected, str(ctx.exception))

    def test_the_superseded_trapezoidal_name_is_rejected(self):
        """
        ``"equiangular-trapezoidal"`` named the rule after nodes it does not have --
        they are equispaced in cos(theta), up to 19 degrees from the equiangular
        grid's -- so it was renamed ``"trapezoidal"`` and the old string dropped
        outright rather than aliased. The error should still point at the new name.
        """
        with self.assertRaises(ValueError) as ctx:
            as_grid("equiangular-trapezoidal", nlat=64, nlon=128)
        self.assertIn("trapezoidal", str(ctx.exception))
        self.assertNotIn("equiangular-trapezoidal", grid_types())

    def test_trapezoidal_nodes_are_not_equiangular(self):
        """
        The fact the rename records: same nlat, materially different grids. If these
        ever coincided, one class would do for both and the old name would have been
        accurate.
        """
        eq = as_grid("equiangular", nlat=17, nlon=32)
        tr = as_grid("trapezoidal", nlat=17, nlon=32)
        self.assertIs(type(tr), TrapezoidalGrid)
        self.assertTrue(eq.is_uniform_in_theta)
        self.assertFalse(tr.is_uniform_in_theta)
        self.assertGreater((eq.colats - tr.colats).abs().max().item(), 0.3)
        # trapezoidal nodes are equispaced in cos(theta), not theta
        self.assertTrue(torch.allclose(torch.cos(tr.colats).diff(), torch.cos(tr.colats).diff()[0].expand(16), atol=1e-12))

    @parameterized.expand([[grid] for grid in _ALL_GRIDS])
    def test_grid_params_reports_the_parameterization(self, grid):
        """
        The parameterization is derived from the dataclass fields rather than
        declared, so it cannot drift from the constructor.
        """
        self.assertEqual(grid_params(grid), ("nlat", "nlon"))
        self.assertEqual(grid_params(_DIRECT_CLASSES[grid]), ("nlat", "nlon"))
        self.assertEqual(grid_params(as_grid(grid, nlat=64, nlon=128)), ("nlat", "nlon"))

    def test_abstract_bases_are_not_instantiable(self):
        """
        Neither level of the hierarchy is a grid. ``GridS2`` fixes no
        parameterization at all, and ``RegularGridS2`` fixes ``(nlat, nlon)`` but
        no latitudinal quadrature rule; only the leaves are complete.
        """
        with self.assertRaises(TypeError):
            GridS2()
        with self.assertRaises(TypeError):
            GridS2(nlat=64, nlon=128)  # GridS2 has no such parameters
        with self.assertRaises(TypeError):
            RegularGridS2(nlat=64, nlon=128)
        with self.assertRaises(TypeError):
            GridShardS2(grid=as_grid("equiangular", nlat=64, nlon=128))

    @parameterized.expand([[grid] for grid in _ALL_GRIDS])
    def test_concrete_grids_are_regular_grids(self, grid):
        """
        Every grid implemented today is regular, and the parameterization by
        ``(nlat, nlon)`` belongs to that level rather than to ``GridS2``.
        """
        g = as_grid(grid, nlat=64, nlon=128)
        self.assertIsInstance(g, RegularGridS2)
        self.assertEqual(RegularGridS2.params(), ("nlat", "nlon"))
        self.assertEqual(GridS2.params(), ())

    @parameterized.expand([[grid] for grid in _ALL_GRIDS])
    def test_regular_grid_satisfies_the_ragged_contract(self, grid):
        """
        The generic implementations on ``GridS2`` -- which are what a ragged grid
        will use -- must agree with the fast overrides on ``RegularGridS2``. If
        they drift, a ragged grid and a regular one would flatten differently.
        """
        g = as_grid(grid, nlat=16, nlon=32)
        self.assertEqual(GridS2.npoints.fget(g), g.npoints)
        self.assertEqual(GridS2.is_regular.fget(g), g.is_regular)
        self.assertTrue(torch.equal(GridS2.lon_offsets.fget(g), g.lon_offsets))
        self.assertEqual(g.nrings, g.nlat)

    @parameterized.expand([[grid] for grid in _ALL_GRIDS])
    def test_the_regular_coords_fast_path_matches_the_ragged_one(self, grid):
        """
        ``coords`` tiles the longitudes when every ring is the same length and walks the
        rings otherwise. Only regular grids reach the first branch and only ragged ones
        reach the second, so nothing else compares them -- and if they drift, a field
        would flatten one way on one grid family and another way on the other.
        """
        g = as_grid(grid, nlat=16, nlon=32)
        fast = g.coords
        walked_lons = torch.cat([g.lons(k) for k in range(g.nrings)])
        walked_colats = torch.repeat_interleave(g.colats, g.nlon_per_lat)
        slow = torch.stack([walked_colats, walked_lons.to(walked_colats.dtype)], dim=-1)
        self.assertTrue(torch.equal(fast, slow))

    def test_numpy_integers_are_accepted_and_normalized(self):
        """
        ``isinstance(np.int64(9), int)`` is False, which rejected the most ordinary way
        to write a resolution sweep::

            for nlat in 2 ** np.arange(3, 8) + 1: ...

        A numpy integer is an integer and is accepted, but stored as a plain ``int``.
        These fields are the descriptor's key, so they reach ``repr`` and ``to_dict``,
        and a numpy scalar there serializes badly even though it hashes and compares
        equal to the int.
        """
        import json

        import numpy as np

        for nlat in 2 ** np.arange(3, 8) + 1:
            with self.subTest(nlat=int(nlat)):
                g = as_grid("equiangular", nlat=nlat, nlon=2 * (nlat - 1))
                self.assertIs(type(g.nlat), int)
                self.assertIs(type(g.nlon), int)

        wide = as_grid("equiangular", nlat=np.int64(32), nlon=np.int32(64))
        plain = as_grid("equiangular", nlat=32, nlon=64)
        self.assertEqual(wide, plain)
        self.assertEqual(hash(wide), hash(plain))
        self.assertEqual(repr(wide), repr(plain))
        self.assertEqual(json.dumps(wide.to_dict()), json.dumps(plain.to_dict()))

        # the shard's ranks are part of its key for the same reason
        shard = plain.shard(polar=(np.int64(1), np.int64(2)))
        self.assertIs(type(shard.polar_rank), int)
        self.assertEqual(shard, plain.shard(polar=(1, 2)))

    def test_a_bool_is_not_a_resolution(self):
        """``bool`` is an ``Integral``, but ``nlat=True`` is a mistake, not nlat=1."""
        for bad in (True, False):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as ctx:
                    as_grid("equiangular", nlat=bad, nlon=64)
                self.assertIn("must be an integer", str(ctx.exception))

    def test_invalid_resolutions_raise(self):
        for nlat, nlon in [(1, 128), (0, 128), (64, 0), (-4, 128)]:
            with self.subTest(nlat=nlat, nlon=nlon):
                with self.assertRaises(ValueError):
                    as_grid("equiangular", nlat=nlat, nlon=nlon)

    @parameterized.expand([[nlat, grid] for nlat in [64, 65] for grid in _ALL_GRIDS])
    def test_nodes_and_weights_match_the_quadrature_helpers(self, nlat, grid, verbose=False):
        """The descriptor must be a view onto the existing routines, not a reimplementation."""
        g = as_grid(grid, nlat=nlat, nlon=2 * nlat)
        lats, w = precompute_latitudes(nlat, grid=grid)
        self.assertTrue(compare_tensors(f"lats (grid={grid}, nlat={nlat})", g.colats, lats, atol=0.0, rtol=0.0, verbose=verbose))
        self.assertTrue(compare_tensors(f"weights (grid={grid}, nlat={nlat})", g.colat_weights, w, atol=0.0, rtol=0.0, verbose=verbose))
        self.assertTrue(compare_tensors(f"lons (grid={grid}, nlat={nlat})", g.lons(), precompute_longitudes(2 * nlat), atol=0.0, rtol=0.0, verbose=verbose))

    @parameterized.expand([[nlat, grid] for nlat in _NLATS for grid in _ALL_GRIDS])
    def test_theta_cutoff_matches_the_free_function(self, nlat, grid):
        """Descriptor-based and legacy call sites must not be able to drift apart."""
        g = as_grid(grid, nlat=nlat, nlon=2 * nlat)
        self.assertEqual(g.max_latitude_spacing, compute_latitude_spacing(nlat, grid=grid))
        # compute_theta_cutoff reports the latitudinal spacing alone, which is what the
        # descriptor's max_latitude_spacing is; max_node_spacing may exceed it
        self.assertEqual(g.max_latitude_spacing, compute_theta_cutoff(nlat, grid=grid))
        self.assertGreaterEqual(g.max_node_spacing, g.max_latitude_spacing)
        self.assertEqual(g.max_node_spacing, max(g.max_latitude_spacing, g.max_longitude_spacing))

    @parameterized.expand([[grid] for grid in _ALL_GRIDS])
    def test_longitude_spacing_is_the_arc_not_the_coordinate_gap(self, grid):
        """
        Adjacent points on a ring are ``2*pi/n`` apart *in longitude*, but their
        great-circle distance is ``2 asin(sin(theta) sin(pi/n))`` -- shorter everywhere
        but the equator, and tending to 0 at the poles where the coordinate gap does
        not. Reporting the gap would make the polar rings look like the widest part of
        the grid when their points are nearly coincident.
        """
        g = as_grid(grid, nlat=64, nlon=128)
        self.assertLess(g.max_longitude_spacing, 2 * math.pi / 128)

        colats, counts = g.colats, g.nlon_per_lat
        arc = 2 * torch.asin(torch.sin(colats) * torch.sin(math.pi / counts.to(colats.dtype)))
        self.assertAlmostEqual(g.max_longitude_spacing, arc.max().item(), places=14)

        # the widest ring is the one nearest the equator, not a polar one
        self.assertLess(abs(colats[int(arc.argmax())].item() - math.pi / 2), 0.1)

    @parameterized.expand([[grid] for grid in _ALL_GRIDS])
    def test_node_spacing_reaches_the_coarser_neighbour(self, grid):
        """
        ``max_node_spacing`` is the coarser of the two directions, because a ring
        grid's neighbours run both ways and an operator's support has to reach the
        further one. Which wins is not a formality: it is longitudinal on a Gauss grid
        even at nlon = 2 nlat, and on anything with nlon < 2 nlat.
        """
        for nlat, nlon in [(64, 128), (64, 64), (64, 32)]:
            with self.subTest(nlat=nlat, nlon=nlon):
                g = as_grid(grid, nlat=nlat, nlon=nlon)
                self.assertEqual(g.max_node_spacing, max(g.max_latitude_spacing, g.max_longitude_spacing))
                self.assertGreaterEqual(g.max_node_spacing, g.max_latitude_spacing)

        # A grid coarse enough in longitude is bounded by longitude rather than latitude.
        # nlon=16 rather than something milder because trapezoidal needs it: its nodes are
        # equispaced in cos(theta), so its polar spacing is ~5x the other grids' and still
        # wins at nlon=32. That is a fact about that grid, not a property of the rule.
        narrow = as_grid(grid, nlat=64, nlon=16)
        self.assertEqual(narrow.max_node_spacing, narrow.max_longitude_spacing)
        self.assertGreater(narrow.max_node_spacing, narrow.max_latitude_spacing)

        # and the case the release notes promise is unaffected: an equiangular grid at
        # nlon = 2 nlat is still bounded by latitude, so its default cutoff does not move
        if grid == "equiangular":
            square = as_grid(grid, nlat=64, nlon=128)
            self.assertEqual(square.max_node_spacing, square.max_latitude_spacing)

    @parameterized.expand([[grid] for grid in _ALL_GRIDS])
    def test_is_uniform_in_theta_agrees_with_the_actual_nodes(self, grid):
        """The advertised flag has to match what the node distribution really does."""
        g = as_grid(grid, nlat=65, nlon=128)
        dlat = g.latitude_spacing
        actually_uniform = bool(((dlat - dlat.mean()).abs().max() < 1e-12).item())
        self.assertEqual(g.is_uniform_in_theta, actually_uniform, msg=f"grid={grid}: is_uniform_in_theta={g.is_uniform_in_theta} but measured uniformity={actually_uniform}")

    # -- identity / caching --------------------------------------------------

    @parameterized.expand([[grid] for grid in _ALL_GRIDS])
    def test_equal_descriptors_hash_equal(self, grid):
        """
        Load-bearing: `torch_harmonics/cache.py` keys its `lru_cache` on the grid.
        A descriptor that hashed by object identity would turn every lookup into a
        miss, silently regressing psi and Legendre precompute.
        """
        a = as_grid(grid, nlat=64, nlon=128)
        b = as_grid(grid, nlat=64, nlon=128)
        self.assertIsNot(a, b)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(len({a, b}), 1)

    def test_differing_descriptors_are_distinct(self):
        base = as_grid("equiangular", nlat=64, nlon=128)
        for other in [as_grid("equiangular", nlat=65, nlon=128), as_grid("equiangular", nlat=64, nlon=256), as_grid("lobatto", nlat=64, nlon=128)]:
            with self.subTest(other=repr(other)):
                self.assertNotEqual(base, other)

    def test_key_contains_only_scalars(self):
        """Anything unhashable or identity-hashed in `key` would break the cache contract."""
        for grid in _ALL_GRIDS:
            with self.subTest(grid=grid):
                key = as_grid(grid, nlat=64, nlon=128).key
                self.assertIsInstance(key, tuple)
                for field in key:
                    self.assertIsInstance(field, (str, int, float, bool, tuple))

    @parameterized.expand([[grid] for grid in _ALL_GRIDS])
    def test_hash_is_stable_across_tensor_access(self, grid):
        """Guards against node/weight tensors ever becoming dataclass fields."""
        g = as_grid(grid, nlat=64, nlon=128)
        before = hash(g)
        _ = g.colats, g.colat_weights, g.lons(), g.latitude_spacing
        self.assertEqual(hash(g), before)

    def test_descriptor_works_as_an_lru_cache_key(self):
        calls = []

        @functools.lru_cache(maxsize=None)
        def _expensive(g):
            calls.append(g)
            return g.nlat * g.nlon

        first = _expensive(as_grid("equiangular", nlat=64, nlon=128))
        second = _expensive(as_grid("equiangular", nlat=64, nlon=128))
        third = _expensive(as_grid("lobatto", nlat=64, nlon=128))

        self.assertEqual(first, second)
        self.assertEqual(len(calls), 2, msg="an equal-but-distinct descriptor missed the cache")
        self.assertEqual(third, 64 * 128)

    # -- raggedness ----------------------------------------------------------

    @parameterized.expand([[grid] for grid in _ALL_GRIDS])
    def test_regular_grid_ragged_fields_are_trivial(self, grid, verbose=False):
        """
        The ragged accessors exist on regular grids too, so consumers can flatten via
        `lon_offsets` instead of assuming a uniform `nlon` stride.
        """
        g = as_grid(grid, nlat=16, nlon=32)
        self.assertTrue(g.is_regular)
        self.assertEqual(g.npoints, 16 * 32)
        self.assertTrue(compare_tensors(f"nlon_per_lat (grid={grid})", g.nlon_per_lat, torch.full((16,), 32, dtype=torch.int64), verbose=verbose))
        self.assertTrue(compare_tensors(f"lon_offsets (grid={grid})", g.lon_offsets, torch.arange(17, dtype=torch.int64) * 32, verbose=verbose))
        self.assertEqual(int(g.lon_offsets[-1].item()), g.npoints)

    # -- serialization -------------------------------------------------------

    @parameterized.expand([[grid] for grid in _ALL_GRIDS])
    def test_to_dict_roundtrip(self, grid):
        """Checkpoints and configs carry the grid as plain data, so this must be lossless."""
        g = as_grid(grid, nlat=64, nlon=128)
        restored = GridS2.from_dict(g.to_dict())
        self.assertEqual(g, restored)
        self.assertEqual(hash(g), hash(restored))
        self.assertIs(type(g), type(restored))

    def test_from_dict_rejects_incomplete_data(self):
        with self.assertRaises(ValueError):
            GridS2.from_dict({"grid": "equiangular", "nlat": 64})

    def test_registry_covers_every_supported_grid_string(self):
        """A grid string accepted by precompute_latitudes must have a descriptor."""
        self.assertEqual(set(grid_types()), set(_ALL_GRIDS))


class TestDirectConstructionMatchesFactory(unittest.TestCase):
    """
    A directly constructed grid and one built by :func:`as_grid` must be the same thing.

    ``as_grid`` is a convenience for callers holding a grid *name*, not a separate
    construction path, so ``EquiangularGrid(nlat=32, nlon=64)`` has to be
    indistinguishable from ``as_grid("equiangular", nlat=32, nlon=64)``. Everything else in
    the suite reaches for the factory, so without this the direct constructors are
    effectively untested -- and they are what a user writes once they know which
    grid they want.

    The property comparison is driven by introspection rather than a hand-written
    list, so a property added to :class:`GridS2` later is covered here the moment it
    exists.
    """

    def _pair(self, name, shape):
        nlat, nlon = shape
        return _DIRECT_CLASSES[name](nlat=nlat, nlon=nlon), as_grid(name, nlat=shape[0], nlon=shape[1])

    @parameterized.expand([[name, shape] for name in grid_types() for shape in _PAIR_SHAPES])
    def test_identity_matches(self, name, shape):
        direct, factory = self._pair(name, shape)
        self.assertIs(type(direct), type(factory))
        self.assertEqual(direct, factory)
        self.assertEqual(hash(direct), hash(factory))
        self.assertEqual(direct.key, factory.key)
        self.assertEqual(repr(direct), repr(factory))
        self.assertEqual(len({direct, factory}), 1)

    @parameterized.expand([[name] for name in grid_types()])
    def test_factory_resolves_to_the_class_you_would_write(self, name):
        """The registry must not drift from the concrete classes."""
        self.assertIs(type(as_grid(name, nlat=32, nlon=64)), _DIRECT_CLASSES[name])

    @parameterized.expand([[name, shape] for name in grid_types() for shape in _PAIR_SHAPES])
    def test_every_property_matches(self, name, shape, verbose=False):
        """
        Compare every public property on the class, discovered by introspection.

        This is the part that keeps working as GridS2 grows: a property whose value
        depended on how the grid was built would be caught without anyone
        remembering to extend a list here.
        """
        direct, factory = self._pair(name, shape)
        names = sorted(n for n, _ in inspect.getmembers(type(direct), lambda m: isinstance(m, property)) if not n.startswith("_"))
        self.assertGreater(len(names), 8, msg=f"introspection found only {names}, which suggests it stopped working")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # non-equiangular grids announce the changed theta_cutoff default
            for prop in names:
                with self.subTest(prop=prop):
                    a, b = getattr(direct, prop), getattr(factory, prop)
                    self.assertIs(type(a), type(b))
                    if isinstance(a, torch.Tensor):
                        self.assertTrue(compare_tensors(f"{name}{shape}.{prop}", a, b, atol=0.0, rtol=0.0, verbose=verbose))
                    else:
                        self.assertEqual(a, b)

    @parameterized.expand([[name, shape] for name in grid_types() for shape in _PAIR_SHAPES])
    def test_methods_match(self, name, shape, verbose=False):
        """Properties are not the whole surface: lons() is a method."""
        direct, factory = self._pair(name, shape)
        self.assertTrue(compare_tensors(f"{name}{shape}.lons()", direct.lons(), factory.lons(), atol=0.0, rtol=0.0, verbose=verbose))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.assertEqual(direct.max_node_spacing, factory.max_node_spacing)
        self.assertEqual(direct.to_dict(), factory.to_dict())

    @parameterized.expand([[name] for name in grid_types()])
    def test_interchangeable_as_a_cache_key(self, name):
        """The two routes must not produce two cache entries for one grid."""
        calls = []

        @functools.lru_cache(maxsize=None)
        def _expensive(g):
            calls.append(g)
            return g.npoints

        direct, factory = self._pair(name, (32, 64))
        self.assertEqual(_expensive(direct), _expensive(factory))
        self.assertEqual(len(calls), 1, msg=f"{name}: direct construction and as_grid missed each other's cache entry")

    @parameterized.expand([[name] for name in grid_types()])
    def test_round_trip_lands_on_the_same_grid_either_way(self, name):
        direct, factory = self._pair(name, (32, 64))
        self.assertEqual(GridS2.from_dict(direct.to_dict()), GridS2.from_dict(factory.to_dict()))
        self.assertIs(type(GridS2.from_dict(direct.to_dict())), _DIRECT_CLASSES[name])

    @parameterized.expand([[name] for name in grid_types()])
    def test_positional_and_keyword_construction_agree(self, name):
        cls = _DIRECT_CLASSES[name]
        self.assertEqual(cls(32, 64), cls(nlat=32, nlon=64))


# (polar_size, azimuth_size) decompositions exercised against a global grid. This is
# the outer product of {1, 2, 4}, matching the grid sizes the distributed suites are
# actually run at, plus (3, 2) so that an uneven split -- where the per-rank shapes
# differ by one -- is covered too.
_DECOMPOSITIONS = [(p, a) for p in (1, 2, 4) for a in (1, 2, 4)] + [(3, 2)]

# global grids the decompositions are applied to; (13, 7) is deliberately awkward
_GLOBAL_SHAPES = [(13, 7), (32, 64), (10, 8)]


class TestGridShard(unittest.TestCase):
    """
    Decomposition of a grid into per-rank pieces.

    The grid owns this because *how* a grid decomposes depends on the grid: a
    regular latitude--longitude grid splits as a product of a latitude range and a
    longitude range, a reduced Gaussian grid has no single ``nlon`` to split, and an
    unstructured grid has no axes at all. The distributed layers currently derive
    these ranges themselves, in twenty-two places, all assuming the product form.

    A shard is a separate type from :class:`GridS2` on purpose: a band of latitudes
    does not cover the sphere, so its weights are a partial sum and the quantities
    describing the quadrature *rule* remain global.
    """

    @parameterized.expand([[name, shape, dec] for name in grid_types() for shape in _GLOBAL_SHAPES for dec in _DECOMPOSITIONS])
    def test_shards_tile_the_global_grid_exactly(self, name, shape, dec, verbose=False):
        """Concatenating the pieces in rank order must reproduce the global arrays."""
        psize, asize = dec
        grid = as_grid(name, nlat=shape[0], nlon=shape[1])
        if grid.nlat < psize or grid.nlon < asize:
            self.skipTest(f"{shape} cannot be split {psize}x{asize} with every chunk non-empty")

        colats = torch.cat([grid.shard(polar=(r, psize)).colats for r in range(psize)])
        weights = torch.cat([grid.shard(polar=(r, psize)).colat_weights for r in range(psize)])
        lons = torch.cat([grid.shard(azimuth=(r, asize)).lons() for r in range(asize)])

        self.assertTrue(compare_tensors(f"{name}{shape} colats tiled {psize}x", colats, grid.colats, atol=0.0, rtol=0.0, verbose=verbose))
        self.assertTrue(compare_tensors(f"{name}{shape} weights tiled {psize}x", weights, grid.colat_weights, atol=0.0, rtol=0.0, verbose=verbose))
        self.assertTrue(compare_tensors(f"{name}{shape} lons tiled {asize}x", lons, grid.lons(), atol=0.0, rtol=0.0, verbose=verbose))

    @parameterized.expand([[name, dec] for name in grid_types() for dec in _DECOMPOSITIONS])
    def test_partial_weights_sum_to_the_global_total(self, name, dec):
        """
        The local weights are a partial contribution completed by a reduction, so
        they must add up across ranks and not, individually, to 2.
        """
        psize, _ = dec
        grid = as_grid(name, nlat=32, nlon=64)
        total = sum(grid.shard(polar=(r, psize)).colat_weights.sum().item() for r in range(psize))
        self.assertAlmostEqual(total, grid.colat_weights.sum().item(), places=14)

    @parameterized.expand([[name, shape] for name in grid_types() for shape in _GLOBAL_SHAPES])
    def test_trivial_shard_is_the_whole_grid(self, name, shape, verbose=False):
        grid = as_grid(name, nlat=shape[0], nlon=shape[1])
        shard = grid.shard()
        self.assertEqual(shard.shape, grid.shape)
        self.assertEqual((shard.lat_offset, shard.lon_offset), (0, 0))
        self.assertTrue(compare_tensors("trivial shard lats", shard.colats, grid.colats, atol=0.0, rtol=0.0, verbose=verbose))

    @parameterized.expand([[dec] for dec in _DECOMPOSITIONS])
    def test_agrees_with_the_tensor_splitter_the_collectives_use(self, dec, verbose=False):
        """
        The load-bearing interop property. The distributed layers move tensors with
        ``split_tensor_along_dim``; if a shard's idea of its own range differed from
        that, the descriptor and the data would silently disagree about which
        latitudes a rank owns.
        """
        psize, asize = dec
        grid = as_grid("lobatto", nlat=13, nlon=7)
        if grid.nlat < psize or grid.nlon < asize:
            self.skipTest("decomposition too fine for this grid")
        for r in range(psize):
            with self.subTest(polar=r):
                expected = split_tensor_along_dim(grid.colats, dim=0, num_chunks=psize)[r]
                self.assertTrue(compare_tensors(f"polar {r}/{psize}", grid.shard(polar=(r, psize)).colats, expected, atol=0.0, rtol=0.0, verbose=verbose))
        for r in range(asize):
            with self.subTest(azimuth=r):
                expected = split_tensor_along_dim(grid.lons(), dim=0, num_chunks=asize)[r]
                self.assertTrue(compare_tensors(f"azimuth {r}/{asize}", grid.shard(azimuth=(r, asize)).lons(), expected, atol=0.0, rtol=0.0, verbose=verbose))

    @parameterized.expand([[dec] for dec in _DECOMPOSITIONS])
    def test_shapes_come_from_the_shared_partitioner(self, dec):
        """One implementation of the split arithmetic, not two that must agree."""
        psize, asize = dec
        grid = as_grid("equiangular", nlat=32, nlon=64)
        self.assertEqual(list(grid.lat_shapes(psize)), compute_split_shapes(32, psize))
        self.assertEqual(list(grid.lon_shapes(asize)), compute_split_shapes(64, asize))
        shard = grid.shard(polar=(0, psize), azimuth=(0, asize))
        self.assertEqual(list(shard.lat_shapes), compute_split_shapes(32, psize))
        self.assertEqual(list(shard.lon_shapes), compute_split_shapes(64, asize))

    @parameterized.expand([[dec] for dec in _DECOMPOSITIONS])
    def test_offsets_follow_the_shapes(self, dec):
        psize, _ = dec
        grid = as_grid("equiangular", nlat=13, nlon=8)
        if grid.nlat < psize:
            self.skipTest("decomposition too fine")
        offset = 0
        for r in range(psize):
            shard = grid.shard(polar=(r, psize))
            self.assertEqual(shard.lat_offset, offset)
            offset += shard.nlat
        self.assertEqual(offset, grid.nlat)

    def test_a_shard_is_not_a_grid(self):
        """
        The distinction that makes the separate type worth having: a shard must not
        be usable where a global grid is required, because its weights are partial
        and its spectral bounds would be meaningless.
        """
        shard = as_grid("equiangular", nlat=32, nlon=64).shard(polar=(1, 2))
        self.assertNotIsInstance(shard, GridS2)
        self.assertFalse(shard.is_global)
        self.assertEqual(shard.global_grid, as_grid("equiangular", nlat=32, nlon=64))
        with self.assertRaises(TypeError) as ctx:
            require_grid(shard)
        self.assertIn("global_grid", str(ctx.exception))

    def test_identity_and_round_trip(self):
        grid = as_grid("legendre-gauss", nlat=32, nlon=64)
        a, b = grid.shard(polar=(1, 2)), grid.shard(polar=(1, 2))
        self.assertIsNot(a, b)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(len({a, b}), 1)
        self.assertNotEqual(a, grid.shard(polar=(0, 2)))
        self.assertEqual(GridShardS2.from_dict(a.to_dict()), a)

    def test_rejects_a_nonsensical_decomposition(self):
        grid = as_grid("equiangular", nlat=32, nlon=64)
        for kwargs in [dict(polar=(2, 2)), dict(polar=(-1, 2)), dict(polar=(0, 0)), dict(azimuth=(5, 3))]:
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    grid.shard(**kwargs)

    def test_shard_weights_reassemble_the_global_ones(self):
        """
        The shard's per-point weights are a partition of the grid's, which is what makes
        the collective in DistributedQuadratureS2 a plain sum.

        The trap this pins: the longitudinal factor takes the *global* ring length while
        the tiling takes the *local* count. Using the local length in the factor makes
        every rank's block sum to the global total on its own, and the reduction then
        overcounts by the azimuth group size -- a wrong answer, not a crash.
        """
        grid = as_grid("legendre-gauss", nlat=16, nlon=32)
        for psize, asize in [(1, 1), (2, 1), (1, 2), (2, 4), (4, 2)]:
            with self.subTest(polar=psize, azimuth=asize):
                blocks = {
                    (pr, ar): grid.shard(polar=(pr, psize), azimuth=(ar, asize)).quad_weights.reshape(*grid.shard(polar=(pr, psize), azimuth=(ar, asize)).shape)
                    for pr in range(psize)
                    for ar in range(asize)
                }
                rows = [torch.cat([blocks[(pr, ar)] for ar in range(asize)], dim=1) for pr in range(psize)]
                self.assertTrue(torch.equal(torch.cat(rows, dim=0).flatten(), grid.quad_weights))
                self.assertAlmostEqual(sum(b.sum().item() for b in blocks.values()), 4.0 * math.pi, places=12)


class TestRaggedGridContract(unittest.TestCase):
    """
    The point of splitting :class:`RegularGridS2` out of :class:`GridS2`: a grid
    family parameterized by something other than ``(nlat, nlon)`` must be
    expressible, and every routine that cannot yet handle one must say so rather
    than silently mis-index.

    ``_RaggedGrid`` below stands in for a HEALPix or reduced Gaussian grid. It is
    defined here rather than shipped, so these tests fail the day the base class
    grows a regular-grid assumption back.
    """

    # These tests exercise as_grid() and grid_params() by name, so the stand-in has
    # to be registered. It is entered and removed around this class alone: other
    # suites build every registered type with (nlat, nlon), which a ragged grid does
    # not take, and one test asserts the registry holds exactly the shipped grids.
    # Leaving it in would make those pass or fail on test ordering.
    _NAME = "test-ragged"

    @classmethod
    def setUpClass(cls):
        from dataclasses import dataclass
        from typing import ClassVar

        @dataclass(frozen=True, eq=False)
        class _RaggedGrid(GridS2):
            """Three rings carrying 4, 8 and 4 longitudes."""

            level: int
            grid_type: ClassVar[str] = cls._NAME

            @property
            def nrings(self):
                return 3

            @property
            def nlon_per_lat(self):
                return torch.tensor([4, 8, 4], dtype=torch.int64) * self.level

            @property
            def colats(self):
                return torch.linspace(0.25, math.pi - 0.25, 3, dtype=torch.float64)

            @property
            def colat_weights(self):
                # any latitudinal rule sums to 2. Chosen so that w_k / nlon_k differs
                # between rings: [0.5, 1.0, 0.5] against counts [4, 8, 4] would make
                # every per-point weight exactly pi/4, and a test on a fixture that
                # uniform cannot tell a per-ring factor from a global one.
                return torch.tensor([0.4, 1.2, 0.4], dtype=torch.float64)

            def lons(self, ilat=None):
                if ilat is None:
                    raise ValueError("a ragged grid has no single set of longitudes")
                n = int(self.nlon_per_lat[ilat])
                return torch.arange(n, dtype=torch.float64) * (2.0 * math.pi / n)

        cls.ragged_cls = _RaggedGrid

    @classmethod
    def tearDownClass(cls):
        _GRID_REGISTRY.pop(cls._NAME, None)

    def setUp(self):
        self.grid = self.ragged_cls(level=1)

    def test_it_is_parameterized_by_its_own_fields(self):
        self.assertEqual(grid_params("test-ragged"), ("level",))
        self.assertEqual(self.grid.key, ("test-ragged", 1))
        self.assertEqual(self.grid.to_dict(), {"grid": "test-ragged", "level": 1})
        self.assertEqual(GridS2.from_dict(self.grid.to_dict()), self.grid)
        self.assertNotEqual(self.grid, as_grid("equiangular", nlat=64, nlon=128))

    def test_the_factory_refuses_regular_grid_parameters(self):
        with self.assertRaises(ValueError) as ctx:
            as_grid("test-ragged", nlat=64, nlon=128)
        self.assertIn("level", str(ctx.exception))

    def test_the_base_class_serves_it_without_nlat_or_nlon(self):
        self.assertFalse(self.grid.is_regular)
        self.assertEqual(self.grid.npoints, 16)
        self.assertEqual(self.grid.shape, (16,))
        self.assertEqual(self.grid.lon_offsets.tolist(), [0, 4, 12, 16])
        self.assertGreater(self.grid.max_node_spacing, 0.0)  # derived from the nodes, not nlat
        self.assertFalse(hasattr(self.grid, "nlat"))
        self.assertFalse(hasattr(self.grid, "max_azimuthal_order"))

    def test_it_does_not_offer_a_two_dimensional_decomposition(self):
        """``shard(polar=, azimuth=)`` is a regular-grid signature, not a universal one."""
        with self.assertRaises(NotImplementedError):
            self.grid.shard(polar=(0, 2), azimuth=(0, 2))

    def test_routines_that_assume_regularity_reject_it(self):
        """
        ``require_regular_grid`` is the single guard each such routine calls, so it
        can be relaxed to ``require_grid`` one routine at a time as backends gain
        support. ``require_grid`` itself must keep accepting the grid.
        """
        self.assertIs(require_grid(self.grid), self.grid)
        with self.assertRaises(TypeError) as ctx:
            require_regular_grid(self.grid, "grid_in")
        self.assertIn("grid_in", str(ctx.exception))
        self.assertIn("RegularGridS2", str(ctx.exception))

    @parameterized.expand([[grid] for grid in _ALL_GRIDS])
    def test_regular_grids_pass_the_guard(self, grid):
        g = as_grid(grid, nlat=32, nlon=64)
        self.assertIs(require_regular_grid(g), g)

    def test_the_per_point_contract_works_on_a_ragged_grid(self):
        """
        ``coords`` and ``quad_weights`` are the PointSetS2 contract, and this is the
        only grid in the suite that actually exercises the ragged path through them.
        """
        grid = self.grid
        coords, weights = grid.coords, grid.quad_weights

        self.assertEqual(tuple(coords.shape), (16, 2))
        self.assertEqual(tuple(weights.shape), (16,))

        # row i of coords describes element i of a field flattened to (npoints,), in
        # ring-major order -- the correspondence the whole contract rests on
        for k in range(grid.nrings):
            base = int(grid.lon_offsets[k])
            for j in range(int(grid.nlon_per_lat[k])):
                self.assertAlmostEqual(coords[base + j, 0].item(), grid.colats[k].item(), places=14)
                self.assertAlmostEqual(coords[base + j, 1].item(), grid.lons(k)[j].item(), places=14)

        # the longitudinal factor is applied per ring, not once globally
        self.assertAlmostEqual(weights.sum().item(), 4.0 * math.pi, places=12)
        for k in range(grid.nrings):
            expected = grid.colat_weights[k].item() * 2.0 * math.pi / int(grid.nlon_per_lat[k])
            lo, hi = int(grid.lon_offsets[k]), int(grid.lon_offsets[k + 1])
            for i in range(lo, hi):
                self.assertAlmostEqual(weights[i].item(), expected, places=14)

        # and the rings really do disagree, so the check above has teeth
        self.assertNotAlmostEqual(weights[0].item(), weights[int(grid.lon_offsets[1])].item(), places=6)

    def test_a_single_dlambda_would_get_the_ragged_integral_wrong(self):
        """
        Regression for the assumption QuadratureS2 used to make.

        The old construction took one ``2 * pi / nlon``. Standing in the widest ring's
        count for every ring is what that amounts to on a ragged grid, and it does not
        integrate a constant correctly -- which is the cheapest possible check that the
        per-ring factor is really being applied.
        """
        grid = self.grid
        widest = int(grid.nlon_per_lat.max())
        naive = torch.repeat_interleave(grid.colat_weights * (2.0 * math.pi / widest), grid.nlon_per_lat)

        self.assertAlmostEqual(grid.quad_weights.sum().item(), 4.0 * math.pi, places=12)
        self.assertNotAlmostEqual(naive.sum().item(), 4.0 * math.pi, places=2)

    def test_quadrature_integrates_on_a_ragged_grid(self):
        """QuadratureS2 takes the weakest guard, so a ragged grid goes straight through."""
        quad = QuadratureS2(self.grid)
        self.assertEqual(quad.spatial_dims, (-1,))
        ones = torch.ones(1, 1, self.grid.npoints, dtype=torch.float32)
        self.assertAlmostEqual(quad(ones).item(), 4.0 * math.pi, places=4)

        mean = QuadratureS2(self.grid, normalize=True)
        self.assertAlmostEqual(mean(ones).item(), 1.0, places=5)


class TestPointSetContract(unittest.TestCase):
    """
    The point of :class:`PointSetS2`: a sampling with no ring structure at all must be
    expressible, and the routines that need only points and weights must accept it.

    ``_FibonacciPointSet`` stands in for an ICON-style unstructured mesh. It is defined
    here rather than shipped: with no rings there is no fast SHT, and its node spacing
    costs a brute-force O(N^2) sweep, which is the right implementation at 200 points
    and unusable past a few thousand. What it buys is coverage no ``GridS2`` stand-in
    can give -- ``_RaggedGrid`` still has rings, so it cannot catch a ring assumption
    leaking down into the base class.
    """

    _NAME = "test-fibonacci"

    @classmethod
    def setUpClass(cls):
        from dataclasses import dataclass
        from typing import ClassVar

        @dataclass(frozen=True, eq=False)
        class _FibonacciPointSet(PointSetS2):
            """Points on a Fibonacci spiral: equal areas, and no two sharing a colatitude."""

            num_points: int
            grid_type: ClassVar[str] = cls._NAME

            @property
            def npoints(self):
                return self.num_points

            @property
            def coords(self):
                i = torch.arange(self.num_points, dtype=torch.float64)
                z = 1.0 - (2.0 * i + 1.0) / self.num_points
                lon = torch.remainder(math.pi * (3.0 - math.sqrt(5.0)) * i, 2.0 * math.pi)
                return torch.stack([torch.arccos(z), lon], dim=-1)

            @property
            def quad_weights(self):
                # equal-area construction, so every point carries the same solid angle
                return torch.full((self.num_points,), 4.0 * math.pi / self.num_points, dtype=torch.float64)

            @property
            def max_node_spacing(self):
                # no rings to binary-search, so the honest answer is a nearest-neighbour sweep
                c = self.coords
                xyz = torch.stack([torch.sin(c[:, 0]) * torch.cos(c[:, 1]), torch.sin(c[:, 0]) * torch.sin(c[:, 1]), torch.cos(c[:, 0])], dim=-1)
                arc = torch.arccos(torch.clamp(xyz @ xyz.T, -1.0, 1.0))
                arc.fill_diagonal_(float("inf"))
                return float(arc.min(dim=1).values.max())

        cls.point_set_cls = _FibonacciPointSet

    @classmethod
    def tearDownClass(cls):
        _GRID_REGISTRY.pop(cls._NAME, None)

    def setUp(self):
        self.ps = self.point_set_cls(num_points=200)

    def test_the_base_contract_is_coords_and_weights(self):
        self.assertEqual(self.ps.npoints, 200)
        self.assertEqual(self.ps.shape, (200,))
        self.assertEqual(tuple(self.ps.coords.shape), (200, 2))
        self.assertEqual(tuple(self.ps.quad_weights.shape), (200,))
        self.assertAlmostEqual(self.ps.quad_weights.sum().item(), 4.0 * math.pi, places=12)
        self.assertGreaterEqual(self.ps.coords[:, 0].min().item(), 0.0)
        self.assertLessEqual(self.ps.coords[:, 0].max().item(), math.pi)

    def test_it_has_no_ring_structure(self):
        """
        Not merely ragged -- absent. A consumer reaching for rings must fail here, which
        is what keeps the ring-specific members on ``GridS2`` where they belong.
        """
        for name in ("nrings", "colats", "lats", "colat_weights", "lons", "nlon_per_lat", "lon_offsets", "is_regular", "max_latitude_spacing"):
            with self.subTest(member=name):
                self.assertFalse(hasattr(self.ps, name), msg=f"PointSetS2 should not expose {name}")

        # and no two points share a colatitude, so there is nothing ring-like to infer
        self.assertEqual(len(torch.unique(self.ps.coords[:, 0])), self.ps.npoints)

    def test_quadrature_accepts_it(self):
        """The payoff: a shipped layer runs on a sampling with no rings."""
        quad = QuadratureS2(self.ps)
        self.assertEqual(quad.spatial_dims, (-1,))
        ones = torch.ones(1, 1, self.ps.npoints, dtype=torch.float32)
        self.assertAlmostEqual(quad(ones).item(), 4.0 * math.pi, places=4)
        self.assertAlmostEqual(QuadratureS2(self.ps, normalize=True)(ones).item(), 1.0, places=5)

    def test_the_guards_separate_the_three_levels(self):
        self.assertIs(require_point_set(self.ps), self.ps)
        with self.assertRaises(TypeError) as ctx:
            require_grid(self.ps, "grid_in")
        self.assertIn("grid_in", str(ctx.exception))
        self.assertIn("isolatitude rings", str(ctx.exception))
        with self.assertRaises(TypeError):
            require_regular_grid(self.ps)

    def test_the_support_radius_comes_from_its_own_nodes(self):
        spacing = self.ps.max_node_spacing
        self.assertGreater(spacing, 0.0)
        self.assertEqual(truncate_support(self.ps), spacing)
        self.assertAlmostEqual(truncate_support(self.ps, scale=2.0), 2.0 * spacing, places=15)
        # a 200-point equal-area sampling has nearest neighbours of order 3.5 / sqrt(N)
        self.assertLess(spacing, 0.5)

    def test_identity_follows_its_own_parameterization(self):
        self.assertEqual(self.point_set_cls.params(), ("num_points",))
        self.assertEqual(self.ps.key, (self._NAME, 200))
        self.assertEqual(self.ps.to_dict(), {"grid": self._NAME, "num_points": 200})
        self.assertEqual(PointSetS2.from_dict(self.ps.to_dict()), self.ps)
        self.assertNotEqual(self.ps, self.point_set_cls(num_points=201))
        self.assertNotEqual(self.ps, as_grid("equiangular", nlat=64, nlon=128))

    def test_it_claims_neither_spectral_accuracy_nor_a_decomposition(self):
        """Both defaults are the conservative ones, so a new family opts in deliberately."""
        self.assertFalse(self.ps.is_spectrally_accurate)
        with self.assertRaises(NotImplementedError):
            self.ps.max_exact_degree
        with self.assertRaises(NotImplementedError):
            self.ps.shard()


if __name__ == "__main__":
    unittest.main()
