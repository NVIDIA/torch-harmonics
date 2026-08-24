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
from torch_harmonics.grid import EquiangularGrid, EquiangularTrapezoidalGrid, GridS2, GridShardS2, LegendreGaussGrid, LobattoGrid, as_grid, grid_types, require_grid
from torch_harmonics.partition import compute_split_shapes
from torch_harmonics.quadrature import compute_latitude_spacing, compute_theta_cutoff, precompute_latitudes, precompute_longitudes

_ALL_GRIDS = ["equiangular", "legendre-gauss", "lobatto", "equiangular-trapezoidal"]

# grids on which the superseded pi / (nlat - 1) heuristic was too narrow near the poles
_IRREGULAR_THETA_GRIDS = ["lobatto", "equiangular-trapezoidal"]

# the class a caller would instantiate directly, against the name as_grid resolves
_DIRECT_CLASSES = {
    "equiangular": EquiangularGrid,
    "legendre-gauss": LegendreGaussGrid,
    "lobatto": LobattoGrid,
    "equiangular-trapezoidal": EquiangularTrapezoidalGrid,
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
                as_grid("equiangular", (nlat, nlon)),
                as_grid("equiangular", (nlat, nlon)),
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


class TestGridDescriptor(unittest.TestCase):
    """
    Contract for :class:`torch_harmonics.grid.GridS2`.

    The descriptor is meant to become the single argument layers take in place of
    ``(nlat, nlon, grid)``. These tests fix the properties the rest of the codebase
    will rely on once that migration happens.
    """

    @parameterized.expand([[grid] for grid in _ALL_GRIDS])
    def test_as_grid_coerces_a_legacy_spec(self, grid):
        g = as_grid(grid, (64, 128))
        self.assertEqual(g.grid_type, grid)
        self.assertEqual(g.shape, (64, 128))
        self.assertEqual((g.nlat, g.nlon), (64, 128))

    @parameterized.expand([[grid] for grid in _ALL_GRIDS])
    def test_as_grid_is_idempotent(self, grid):
        g = as_grid(grid, (64, 128))
        self.assertIs(as_grid(g), g)
        self.assertIs(as_grid(g, (64, 128)), g)

    def test_as_grid_rejects_bad_specs(self):
        with self.assertRaises(ValueError):
            as_grid("not-a-grid", (64, 128))
        with self.assertRaises(ValueError):
            as_grid("equiangular")  # shape required for a string spec
        with self.assertRaises(ValueError):
            as_grid("equiangular", (64,))
        with self.assertRaises(ValueError):
            as_grid(as_grid("equiangular", (64, 128)), (32, 64))  # contradicts the descriptor

    def test_abstract_base_is_not_instantiable(self):
        with self.assertRaises(TypeError):
            GridS2(nlat=64, nlon=128)

    def test_invalid_resolutions_raise(self):
        for nlat, nlon in [(1, 128), (0, 128), (64, 0), (-4, 128)]:
            with self.subTest(nlat=nlat, nlon=nlon):
                with self.assertRaises(ValueError):
                    as_grid("equiangular", (nlat, nlon))

    @parameterized.expand([[nlat, grid] for nlat in [64, 65] for grid in _ALL_GRIDS])
    def test_nodes_and_weights_match_the_quadrature_helpers(self, nlat, grid, verbose=False):
        """The descriptor must be a view onto the existing routines, not a reimplementation."""
        g = as_grid(grid, (nlat, 2 * nlat))
        lats, w = precompute_latitudes(nlat, grid=grid)
        self.assertTrue(compare_tensors(f"lats (grid={grid}, nlat={nlat})", g.lats, lats, atol=0.0, rtol=0.0, verbose=verbose))
        self.assertTrue(compare_tensors(f"weights (grid={grid}, nlat={nlat})", g.quad_weights, w, atol=0.0, rtol=0.0, verbose=verbose))
        self.assertTrue(compare_tensors(f"lons (grid={grid}, nlat={nlat})", g.lons(), precompute_longitudes(2 * nlat), atol=0.0, rtol=0.0, verbose=verbose))

    @parameterized.expand([[nlat, grid] for nlat in _NLATS for grid in _ALL_GRIDS])
    def test_theta_cutoff_matches_the_free_function(self, nlat, grid):
        """Descriptor-based and legacy call sites must not be able to drift apart."""
        g = as_grid(grid, (nlat, 2 * nlat))
        self.assertEqual(g.theta_cutoff(), compute_theta_cutoff(nlat, grid=grid))
        self.assertEqual(g.theta_cutoff(scale=2.5), compute_theta_cutoff(nlat, grid=grid, scale=2.5))
        self.assertEqual(g.max_latitude_spacing, compute_latitude_spacing(nlat, grid=grid))

    @parameterized.expand([[grid] for grid in _ALL_GRIDS])
    def test_is_uniform_in_theta_agrees_with_the_actual_nodes(self, grid):
        """The advertised flag has to match what the node distribution really does."""
        g = as_grid(grid, (65, 128))
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
        a = as_grid(grid, (64, 128))
        b = as_grid(grid, (64, 128))
        self.assertIsNot(a, b)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(len({a, b}), 1)

    def test_differing_descriptors_are_distinct(self):
        base = as_grid("equiangular", (64, 128))
        for other in [as_grid("equiangular", (65, 128)), as_grid("equiangular", (64, 256)), as_grid("lobatto", (64, 128))]:
            with self.subTest(other=repr(other)):
                self.assertNotEqual(base, other)

    def test_key_contains_only_scalars(self):
        """Anything unhashable or identity-hashed in `key` would break the cache contract."""
        for grid in _ALL_GRIDS:
            with self.subTest(grid=grid):
                key = as_grid(grid, (64, 128)).key
                self.assertIsInstance(key, tuple)
                for field in key:
                    self.assertIsInstance(field, (str, int, float, bool, tuple))

    @parameterized.expand([[grid] for grid in _ALL_GRIDS])
    def test_hash_is_stable_across_tensor_access(self, grid):
        """Guards against node/weight tensors ever becoming dataclass fields."""
        g = as_grid(grid, (64, 128))
        before = hash(g)
        _ = g.lats, g.quad_weights, g.lons(), g.latitude_spacing
        self.assertEqual(hash(g), before)

    def test_descriptor_works_as_an_lru_cache_key(self):
        calls = []

        @functools.lru_cache(maxsize=None)
        def _expensive(g):
            calls.append(g)
            return g.nlat * g.nlon

        first = _expensive(as_grid("equiangular", (64, 128)))
        second = _expensive(as_grid("equiangular", (64, 128)))
        third = _expensive(as_grid("lobatto", (64, 128)))

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
        g = as_grid(grid, (16, 32))
        self.assertTrue(g.is_regular)
        self.assertEqual(g.npoints, 16 * 32)
        self.assertTrue(compare_tensors(f"nlon_per_lat (grid={grid})", g.nlon_per_lat, torch.full((16,), 32, dtype=torch.int64), verbose=verbose))
        self.assertTrue(compare_tensors(f"lon_offsets (grid={grid})", g.lon_offsets, torch.arange(17, dtype=torch.int64) * 32, verbose=verbose))
        self.assertEqual(int(g.lon_offsets[-1].item()), g.npoints)

    # -- serialization -------------------------------------------------------

    @parameterized.expand([[grid] for grid in _ALL_GRIDS])
    def test_to_dict_roundtrip(self, grid):
        """Checkpoints and configs carry the grid as plain data, so this must be lossless."""
        g = as_grid(grid, (64, 128))
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
    indistinguishable from ``as_grid("equiangular", (32, 64))``. Everything else in
    the suite reaches for the factory, so without this the direct constructors are
    effectively untested -- and they are what a user writes once they know which
    grid they want.

    The property comparison is driven by introspection rather than a hand-written
    list, so a property added to :class:`GridS2` later is covered here the moment it
    exists.
    """

    def _pair(self, name, shape):
        nlat, nlon = shape
        return _DIRECT_CLASSES[name](nlat=nlat, nlon=nlon), as_grid(name, shape)

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
        self.assertIs(type(as_grid(name, (32, 64))), _DIRECT_CLASSES[name])

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
        """Properties are not the whole surface: lons() and theta_cutoff() are methods."""
        direct, factory = self._pair(name, shape)
        self.assertTrue(compare_tensors(f"{name}{shape}.lons()", direct.lons(), factory.lons(), atol=0.0, rtol=0.0, verbose=verbose))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.assertEqual(direct.theta_cutoff(), factory.theta_cutoff())
            self.assertEqual(direct.theta_cutoff(scale=2.5), factory.theta_cutoff(scale=2.5))
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
        grid = as_grid(name, shape)
        if grid.nlat < psize or grid.nlon < asize:
            self.skipTest(f"{shape} cannot be split {psize}x{asize} with every chunk non-empty")

        lats = torch.cat([grid.shard(polar=(r, psize)).lats for r in range(psize)])
        weights = torch.cat([grid.shard(polar=(r, psize)).quad_weights for r in range(psize)])
        lons = torch.cat([grid.shard(azimuth=(r, asize)).lons() for r in range(asize)])

        self.assertTrue(compare_tensors(f"{name}{shape} lats tiled {psize}x", lats, grid.lats, atol=0.0, rtol=0.0, verbose=verbose))
        self.assertTrue(compare_tensors(f"{name}{shape} weights tiled {psize}x", weights, grid.quad_weights, atol=0.0, rtol=0.0, verbose=verbose))
        self.assertTrue(compare_tensors(f"{name}{shape} lons tiled {asize}x", lons, grid.lons(), atol=0.0, rtol=0.0, verbose=verbose))

    @parameterized.expand([[name, dec] for name in grid_types() for dec in _DECOMPOSITIONS])
    def test_partial_weights_sum_to_the_global_total(self, name, dec):
        """
        The local weights are a partial contribution completed by a reduction, so
        they must add up across ranks and not, individually, to 2.
        """
        psize, _ = dec
        grid = as_grid(name, (32, 64))
        total = sum(grid.shard(polar=(r, psize)).quad_weights.sum().item() for r in range(psize))
        self.assertAlmostEqual(total, grid.quad_weights.sum().item(), places=14)

    @parameterized.expand([[name, shape] for name in grid_types() for shape in _GLOBAL_SHAPES])
    def test_trivial_shard_is_the_whole_grid(self, name, shape, verbose=False):
        grid = as_grid(name, shape)
        shard = grid.shard()
        self.assertEqual(shard.shape, grid.shape)
        self.assertEqual((shard.lat_offset, shard.lon_offset), (0, 0))
        self.assertTrue(compare_tensors("trivial shard lats", shard.lats, grid.lats, atol=0.0, rtol=0.0, verbose=verbose))

    @parameterized.expand([[dec] for dec in _DECOMPOSITIONS])
    def test_agrees_with_the_tensor_splitter_the_collectives_use(self, dec, verbose=False):
        """
        The load-bearing interop property. The distributed layers move tensors with
        ``split_tensor_along_dim``; if a shard's idea of its own range differed from
        that, the descriptor and the data would silently disagree about which
        latitudes a rank owns.
        """
        psize, asize = dec
        grid = as_grid("lobatto", (13, 7))
        if grid.nlat < psize or grid.nlon < asize:
            self.skipTest("decomposition too fine for this grid")
        for r in range(psize):
            with self.subTest(polar=r):
                expected = split_tensor_along_dim(grid.lats, dim=0, num_chunks=psize)[r]
                self.assertTrue(compare_tensors(f"polar {r}/{psize}", grid.shard(polar=(r, psize)).lats, expected, atol=0.0, rtol=0.0, verbose=verbose))
        for r in range(asize):
            with self.subTest(azimuth=r):
                expected = split_tensor_along_dim(grid.lons(), dim=0, num_chunks=asize)[r]
                self.assertTrue(compare_tensors(f"azimuth {r}/{asize}", grid.shard(azimuth=(r, asize)).lons(), expected, atol=0.0, rtol=0.0, verbose=verbose))

    @parameterized.expand([[dec] for dec in _DECOMPOSITIONS])
    def test_shapes_come_from_the_shared_partitioner(self, dec):
        """One implementation of the split arithmetic, not two that must agree."""
        psize, asize = dec
        grid = as_grid("equiangular", (32, 64))
        self.assertEqual(list(grid.lat_shapes(psize)), compute_split_shapes(32, psize))
        self.assertEqual(list(grid.lon_shapes(asize)), compute_split_shapes(64, asize))
        shard = grid.shard(polar=(0, psize), azimuth=(0, asize))
        self.assertEqual(list(shard.lat_shapes), compute_split_shapes(32, psize))
        self.assertEqual(list(shard.lon_shapes), compute_split_shapes(64, asize))

    @parameterized.expand([[dec] for dec in _DECOMPOSITIONS])
    def test_offsets_follow_the_shapes(self, dec):
        psize, _ = dec
        grid = as_grid("equiangular", (13, 8))
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
        shard = as_grid("equiangular", (32, 64)).shard(polar=(1, 2))
        self.assertNotIsInstance(shard, GridS2)
        self.assertFalse(shard.is_global)
        self.assertEqual(shard.global_grid, as_grid("equiangular", (32, 64)))
        with self.assertRaises(TypeError) as ctx:
            require_grid(shard)
        self.assertIn("global_grid", str(ctx.exception))

    def test_identity_and_round_trip(self):
        grid = as_grid("legendre-gauss", (32, 64))
        a, b = grid.shard(polar=(1, 2)), grid.shard(polar=(1, 2))
        self.assertIsNot(a, b)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(len({a, b}), 1)
        self.assertNotEqual(a, grid.shard(polar=(0, 2)))
        self.assertEqual(GridShardS2.from_dict(a.to_dict()), a)

    def test_rejects_a_nonsensical_decomposition(self):
        grid = as_grid("equiangular", (32, 64))
        for kwargs in [dict(polar=(2, 2)), dict(polar=(-1, 2)), dict(polar=(0, 0)), dict(azimuth=(5, 3))]:
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    grid.shard(**kwargs)


if __name__ == "__main__":
    unittest.main()
