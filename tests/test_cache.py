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

import gc
import unittest
import weakref
from copy import deepcopy

import torch
from parameterized import parameterized, parameterized_class
from testutils import compare_tensors, regular_grid_types

import torch_harmonics as th
from torch_harmonics.cache import is_shared, lru_cache, move_shared, shared_table, store_shared
from torch_harmonics.grid import as_grid

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


class TestCacheConsistency(unittest.TestCase):
    def test_consistency(self, verbose=False):
        """Equal arguments share one table while it is held: a second build would be a second copy."""
        from torch_harmonics.legendre import _precompute_legpoly

        with torch.no_grad():
            leg1 = _precompute_legpoly(10, 10, as_grid("legendre-gauss", nlat=10, nlon=20))
            leg2 = _precompute_legpoly(10, 10, as_grid("legendre-gauss", nlat=10, nlon=20))
            self.assertIs(leg1, leg2)
            # the keyword spelling of the same call is the same entry
            self.assertIs(_precompute_legpoly(mmax=10, lmax=10, grid=as_grid("legendre-gauss", nlat=10, nlon=20)), leg1)

    def test_cache_tensor(self, verbose=False):
        from torch_harmonics.legendre import _precompute_legpoly

        with torch.no_grad():
            # the grid is part of the key, so two grids of equal size must not collide
            leg1 = _precompute_legpoly(10, 10, as_grid("legendre-gauss", nlat=10, nlon=20))
            leg2 = _precompute_legpoly(10, 10, as_grid("equiangular", nlat=10, nlon=20))
            self.assertFalse(torch.allclose(leg1, leg2))

    def test_cache_hits(self, verbose=False):
        """Repeated setup at one resolution must reuse the table, not rebuild it.

        Stacked models construct many layers at the same resolution, and each rebuild of the
        Legendre tables is the most expensive part of that setup. Nothing else in the suite
        would notice if caching silently stopped -- the other tests here only check that a
        cached value cannot be wrong, not that it is ever reused.

        The historical failure this guards against: these tables were once keyed on a tensor
        of colatitudes, and tensors hash by identity, so every layer built a fresh node tensor
        and therefore missed. The key is now the grid descriptor, which hashes on its type
        and resolution rather than on identity.
        """

        import torch_harmonics.legendre as legendre

        def count_builds(inner_name, call, repeats=5):
            """
            Invocations of the uncached core while ``call`` runs ``repeats`` times.

            The results are held throughout, as the layers of a model hold their tables: a
            shared table lives only as long as someone uses it.
            """
            calls = []
            held = []
            original = getattr(legendre, inner_name)

            def counting(*args, **kwargs):
                calls.append(1)
                return original(*args, **kwargs)

            setattr(legendre, inner_name, counting)
            try:
                for _ in range(repeats):
                    held.append(call())
            finally:
                setattr(legendre, inner_name, original)
            return len(calls)

        with torch.no_grad():
            # distinct grids per assertion so a warm cache from another test cannot
            # mask a miss, and so the two assertions cannot warm each other
            n = count_builds("legpoly", lambda: legendre._precompute_legpoly(8, 8, as_grid("legendre-gauss", nlat=14, nlon=28)))
            self.assertEqual(n, 1, msg=f"_precompute_legpoly rebuilt the table {n} times instead of caching it")

            n = count_builds("dlegpoly", lambda: legendre._precompute_dlegpoly(8, 8, as_grid("legendre-gauss", nlat=18, nlon=36)))
            self.assertEqual(n, 1, msg=f"_precompute_dlegpoly rebuilt the table {n} times instead of caching it")

            # ...and the key must still discriminate: a different resolution is a real miss,
            # so a cache that returned everything unconditionally would fail here
            calls = iter((22, 22, 26, 26, 22))
            n = count_builds("legpoly", lambda: legendre._precompute_legpoly(8, 8, as_grid("lobatto", nlat=next(calls), nlon=44)), repeats=5)
            self.assertEqual(n, 2, msg=f"expected one build per distinct grid, got {n}")


class TestLruCacheWrapper(unittest.TestCase):
    """
    The copying ``lru_cache`` wrapper must not hide the function it decorates.

    ``functools.lru_cache`` attaches ``cache_info`` / ``cache_clear`` /
    ``cache_parameters`` to the object it returns, and carries the wrapped
    function's metadata. The copying wrapper sits in front of all of that, so it
    has to forward both explicitly.
    """

    def test_cache_info_is_forwarded(self):
        @lru_cache(maxsize=20, typed=True)
        def _square(x):
            return x * x

        _square.cache_clear()
        self.assertEqual(_square.cache_info().currsize, 0)

        self.assertEqual(_square(4), 16)
        self.assertEqual(_square.cache_info().misses, 1)
        self.assertEqual(_square.cache_info().hits, 0)

        self.assertEqual(_square(4), 16)
        self.assertEqual(_square.cache_info().hits, 1)
        self.assertEqual(_square.cache_info().currsize, 1)

    def test_cache_clear_is_forwarded(self):
        calls = []

        @lru_cache(maxsize=20)
        def _tracked(x):
            calls.append(x)
            return x

        _tracked(1)
        _tracked(1)
        self.assertEqual(len(calls), 1)

        _tracked.cache_clear()
        self.assertEqual(_tracked.cache_info().currsize, 0)

        _tracked(1)
        self.assertEqual(len(calls), 2, msg="cache_clear did not drop the cached entry")

    def test_cache_parameters_are_forwarded(self):
        @lru_cache(maxsize=7, typed=True)
        def _identity(x):
            return x

        params = _identity.cache_parameters()
        self.assertEqual(params["maxsize"], 7)
        self.assertTrue(params["typed"])

    def test_wrapped_function_metadata_is_preserved(self):
        """
        Without functools.wraps every cached routine documents as
        ``wrapper(*args, **kwargs)`` with no docstring, which silently dropped the
        API documentation of precompute_latitudes and friends.
        """

        @lru_cache(maxsize=20)
        def _documented(x):
            """A very specific docstring."""
            return x

        self.assertEqual(_documented.__name__, "_documented")
        self.assertEqual(_documented.__doc__, "A very specific docstring.")
        self.assertIs(_documented.__wrapped__.__doc__, _documented.__doc__)

    def test_public_cached_routines_keep_their_docstrings(self):
        """The regression that motivated the fix, checked on the real functions."""
        from torch_harmonics.quadrature import compute_latitude_spacing, precompute_latitudes, precompute_longitudes

        for func, name in [
            (precompute_latitudes, "precompute_latitudes"),
            (precompute_longitudes, "precompute_longitudes"),
            (compute_latitude_spacing, "compute_latitude_spacing"),
        ]:
            with self.subTest(func=name):
                self.assertEqual(func.__name__, name)
                self.assertIsNotNone(func.__doc__, msg=f"{name} lost its docstring to the cache decorator")
                self.assertIn("Parameters", func.__doc__)

    def test_copying_wrapper_still_copies(self):
        """Forwarding must not disturb the deep-copy behaviour the decorator exists for."""

        @lru_cache(maxsize=20, copy=True)
        def _make_list(n):
            return [0] * n

        first = _make_list(3)
        first[0] = 99
        self.assertEqual(_make_list(3), [0, 0, 0])
        self.assertEqual(_make_list.cache_info().hits, 1)


class TestGridDescriptorCaching(unittest.TestCase):
    """
    Interaction between :class:`torch_harmonics.grid.GridS2` and this module's
    ``lru_cache``.

    A grid descriptor is intended to become the cache key for the expensive
    precomputes (psi, Legendre coefficients, quadrature nodes) that are currently
    keyed on ``(nlat, grid_string)``. That only works if equal descriptors are
    interchangeable as keys; if the descriptor ever fell back to identity hashing,
    every lookup would become a miss and the failure would show up as a startup
    slowdown rather than as a test failure. These tests make it a test failure.
    """

    def test_equal_descriptors_share_a_cache_entry(self, verbose=False):
        calls = []

        @lru_cache(maxsize=20, typed=True)
        def _expensive(grid):
            calls.append(grid)
            return grid.nlat * grid.nlon

        for grid_type in regular_grid_types():
            with self.subTest(grid=grid_type):
                calls.clear()
                first = _expensive(as_grid(grid_type, nlat=64, nlon=128))
                second = _expensive(as_grid(grid_type, nlat=64, nlon=128))
                self.assertEqual(first, second)
                self.assertEqual(len(calls), 1, msg=f"grid={grid_type}: an equal-but-distinct descriptor missed the cache")

    def test_cache_distinguishes_grids_that_differ(self):
        calls = []

        @lru_cache(maxsize=20, typed=True)
        def _expensive(grid):
            calls.append(grid)
            return grid.max_latitude_spacing

        distinct = [
            as_grid("equiangular", nlat=64, nlon=128),
            as_grid("equiangular", nlat=65, nlon=128),
            as_grid("equiangular", nlat=64, nlon=256),
            as_grid("lobatto", nlat=64, nlon=128),
        ]
        for grid in distinct:
            _expensive(grid)
        # a second sweep must be served entirely from the cache
        for grid in distinct:
            _expensive(grid)

        self.assertEqual(len(calls), len(distinct), msg="descriptors differing in grid type or resolution collided or missed")

    def test_descriptor_survives_deepcopy(self):
        """``lru_cache(copy=True)`` deep-copies its return value, so a cached descriptor must remain a valid key."""
        for grid_type in regular_grid_types():
            with self.subTest(grid=grid_type):
                grid = as_grid(grid_type, nlat=64, nlon=128)
                clone = deepcopy(grid)
                self.assertIsNot(clone, grid)
                self.assertEqual(clone, grid)
                self.assertEqual(hash(clone), hash(grid))
                self.assertEqual(len({grid, clone}), 1)

    def test_descriptor_tensors_are_not_externally_mutable(self, verbose=False):
        """
        The node and weight tensors are handed out from the ``copy=True`` cache in
        ``torch_harmonics.quadrature``. Mutating what a descriptor returns must not
        corrupt the cached entry for every later consumer, in the same way
        :meth:`TestCacheConsistency.test_consistency` guards the Legendre cache.
        """
        with torch.no_grad():
            for grid_type in regular_grid_types():
                with self.subTest(grid=grid_type):
                    grid = as_grid(grid_type, nlat=32, nlon=64)
                    pristine_colats, pristine_weights = grid.colats.clone(), grid.colat_weights.clone()

                    grid.colats.mul_(-1.0)
                    grid.colat_weights.mul_(-1.0)
                    grid.lons().mul_(-1.0)

                    self.assertTrue(compare_tensors(f"lats after external mutation (grid={grid_type})", grid.colats, pristine_colats, atol=0.0, rtol=0.0, verbose=verbose))
                    self.assertTrue(
                        compare_tensors(f"weights after external mutation (grid={grid_type})", grid.colat_weights, pristine_weights, atol=0.0, rtol=0.0, verbose=verbose)
                    )
                    self.assertGreaterEqual(grid.lons().min().item(), 0.0, msg=f"grid={grid_type}: longitudes were corrupted by an external in-place write")

    def test_per_point_tensors_are_cached_and_independent(self, verbose=False):
        """
        ``coords`` and ``quad_weights`` are O(npoints) -- ~66 MB at 1440x2880 -- so they
        are cached on the descriptor rather than rebuilt per access. They must still obey
        the same no-aliasing contract as the per-ring tensors, since a cache that hands
        out shared storage turns one consumer's in-place op into everyone's bug.
        """
        with torch.no_grad():
            for grid_type in regular_grid_types():
                with self.subTest(grid=grid_type):
                    grid = as_grid(grid_type, nlat=32, nlon=64)
                    for name in ("coords", "quad_weights"):
                        first, second = getattr(grid, name), getattr(grid, name)
                        self.assertIsNot(first, second, msg=f"{grid_type}.{name} returned the same object twice")
                        self.assertNotEqual(first.data_ptr(), second.data_ptr(), msg=f"{grid_type}.{name} aliased storage")

                        pristine = second.clone()
                        first.mul_(-1.0)
                        self.assertTrue(compare_tensors(f"{name} after external mutation (grid={grid_type})", getattr(grid, name), pristine, atol=0.0, rtol=0.0, verbose=verbose))

    def test_descriptor_returns_independent_tensors(self):
        """Two accesses must not alias, otherwise one consumer's in-place op leaks into another's."""
        for grid_type in regular_grid_types():
            with self.subTest(grid=grid_type):
                grid = as_grid(grid_type, nlat=32, nlon=64)
                first, second = grid.colats, grid.colats
                self.assertIsNot(first, second)
                self.assertNotEqual(first.data_ptr(), second.data_ptr(), msg=f"grid={grid_type}: repeated access to lats returned aliased storage")

    def test_derived_scalars_are_consistent_across_accesses(self):
        """``max_latitude_spacing`` is itself cached; repeated access must be stable and match the nodes."""
        for grid_type in regular_grid_types():
            with self.subTest(grid=grid_type):
                grid = as_grid(grid_type, nlat=65, nlon=128)
                first = grid.max_latitude_spacing
                _ = grid.colats.mul_(-1.0)  # try to poison the underlying node cache
                self.assertEqual(grid.max_latitude_spacing, first)
                self.assertEqual(grid.max_latitude_spacing, grid.latitude_spacing.max().item())


class TestSharedTable(unittest.TestCase):
    """The registry behind shared_table: one table per distinct arguments, alive while held."""

    def setUp(self):
        self.builds = []

        @shared_table
        def table(n, scale=1.0):
            self.builds.append((n, scale))
            return scale * torch.arange(n, dtype=torch.float64)

        @shared_table
        def pair(n):
            self.builds.append(("pair", n))
            return torch.zeros(n), torch.ones(n), "metadata"

        self.table = table
        self.pair = pair

    def test_equal_arguments_share_one_table(self):
        a = self.table(4)
        self.assertIs(self.table(4), a)
        self.assertIs(self.table(n=4, scale=1.0), a)
        self.assertEqual(len(self.builds), 1)
        self.assertTrue(is_shared(a))
        self.assertFalse(is_shared(a.clone()))

    def test_typed_arguments_do_not_collide(self):
        """As with lru_cache(typed=True), 1 and 1.0 are different keys."""
        a = self.table(4, 1.0)
        b = self.table(4, 1)
        self.assertIsNot(a, b)
        self.assertEqual(len(self.builds), 2)

    def test_a_released_table_is_freed_and_rebuilt(self):
        a = self.table(5)
        ref = weakref.ref(a)
        del a
        gc.collect()
        self.assertIsNone(ref(), "the registry kept a table nobody holds")
        self.table(5)
        self.assertEqual(len(self.builds), 2)

    def test_a_value_is_returned_whole_or_rebuilt(self):
        """A nested value whose tensors have partly died is a miss, never half a value."""
        zeros, ones, meta = self.pair(3)
        self.assertEqual(meta, "metadata")
        self.assertIs(self.pair(3)[0], zeros)
        del ones
        gc.collect()
        rebuilt = self.pair(3)
        self.assertIsNot(rebuilt[0], zeros)
        self.assertEqual(len(self.builds), 2)

    def test_unhashable_arguments_raise(self):
        with self.assertRaises(TypeError):
            self.table({})

    def test_builders_of_the_same_name_never_share(self):
        """Closures from one factory share a qualname but not their tables."""

        def make(scale):
            @shared_table
            def scaled(n):
                self.builds.append(("scaled", scale))
                return scale * torch.ones(n, dtype=torch.float64)

            return scaled

        double, triple = make(2.0), make(3.0)
        self.assertEqual(double.__qualname__, triple.__qualname__)
        a, b = double(4), triple(4)
        self.assertIsNot(a, b)
        torch.testing.assert_close(b, torch.full((4,), 3.0, dtype=torch.float64), atol=0, rtol=0)
        self.assertIs(double(4), a)
        self.assertEqual(len(self.builds), 2)

    def test_a_rounded_copy_is_never_served_as_exact(self):
        """The float32 copy of a table that went through half is not the float32 copy of the table."""
        exact = self.table(7, 0.1)
        rounded = move_shared(move_shared(exact, lambda t: t.half()), lambda t: t.float())
        direct = move_shared(exact, lambda t: t.float())
        self.assertIsNot(direct, rounded)
        self.assertTrue(torch.equal(direct, exact.float()))
        self.assertTrue(torch.equal(rounded, exact.half().float()))
        # the same conversions share, and widening adds nothing to the history
        self.assertIs(move_shared(move_shared(exact, lambda t: t.half()), lambda t: t.float()), rounded)
        self.assertIs(move_shared(direct, lambda t: t.double()), move_shared(direct, lambda t: t.double()))
        self.assertIsNot(move_shared(direct, lambda t: t.double()), exact)

    def test_an_exact_copy_spares_the_rebuild(self):
        """While copies with the exact values live, the value is put back together from them."""
        zeros, ones, _ = self.pair(3)
        wider = (move_shared(zeros, lambda t: t.double()), move_shared(ones, lambda t: t.double()))
        del zeros, ones
        gc.collect()
        again = self.pair(3)
        self.assertEqual(len(self.builds), 1)
        self.assertEqual(again[2], "metadata")
        self.assertEqual(again[0].dtype, torch.float32)
        self.assertTrue(torch.equal(again[1], torch.ones(3)))
        self.assertIs(move_shared(again[1], lambda t: t.double()), wider[1])

    def test_a_rounded_copy_does_not_spare_the_rebuild(self):
        source = self.table(6, 0.1)
        rounded = move_shared(source, lambda t: t.float())
        del source
        gc.collect()
        again = self.table(6, 0.1)
        self.assertEqual(len(self.builds), 2)
        self.assertTrue(torch.equal(again, 0.1 * torch.arange(6, dtype=torch.float64)))
        self.assertIs(move_shared(again, lambda t: t.float()), rounded)

    def test_values_come_back_from_the_meta_device(self):
        """to_empty off the meta device rebuilds the table and replays its history."""
        source = self.table(6, 0.1)
        on_meta = move_shared(move_shared(source, lambda t: t.half()), lambda t: t.to("meta"))
        del source
        gc.collect()
        back = move_shared(on_meta, lambda t: torch.empty_like(t, device="cpu"))
        self.assertEqual(back.dtype, torch.float16)
        self.assertTrue(torch.equal(back, (0.1 * torch.arange(6, dtype=torch.float64)).half()))
        self.assertEqual(len(self.builds), 2)

    def test_unbuildable_state_falls_back_to_the_conversion(self):
        """State stored without a builder leaves the meta device as an unshared buffer would."""
        state = store_shared(("test-state", id(self)), torch.ones(3))
        on_meta = move_shared(state, lambda t: t.to("meta"))
        del state
        gc.collect()
        back = move_shared(on_meta, lambda t: torch.empty_like(t, device="cpu"))
        self.assertEqual((back.device.type, back.shape), ("cpu", (3,)))
        self.assertFalse(is_shared(back))


def _sht_pairs(device):
    """Two identical instances of each SHT, and one that differs only in lmax."""
    grid = as_grid("legendre-gauss", nlat=12, nlon=24)
    for cls, name in ((th.RealSHT, "weights"), (th.InverseRealSHT, "pct"), (th.RealVectorSHT, "weights"), (th.InverseRealVectorSHT, "dpct")):
        yield cls, name, cls(grid).to(device), cls(grid).to(device), cls(grid, lmax=8, mmax=8).to(device)


@parameterized_class(("device"), _devices)
class TestSharedSHTBuffers(unittest.TestCase):
    """Identical SHTs hold one copy of their table, on every device and in every dtype."""

    def test_identical_transforms_share_their_table(self):
        for cls, name, a, b, other in _sht_pairs(self.device):
            with self.subTest(cls=cls.__name__):
                self.assertIs(getattr(a, name), getattr(b, name))
                self.assertEqual(getattr(a, name).device.type, self.device.type)
                self.assertIsNot(getattr(other, name), getattr(a, name))
                self.assertNotIn(name, a.state_dict())

    @parameterized.expand([["ortho"], ["schmidt"], ["unnorm"]])
    def test_forward_and_inverse_never_share(self, norm):
        """
        The forward table folds in the quadrature weights and the inverse one does not, so
        on the same grid and bounds they must stay distinct entries: the builders key on
        their own names, and the Legendre tables on ``inverse``.
        """
        from torch_harmonics.legendre import _precompute_legpoly

        grid = as_grid("legendre-gauss", nlat=10, nlon=20)
        for forward_cls, inverse_cls in ((th.RealSHT, th.InverseRealSHT), (th.RealVectorSHT, th.InverseRealVectorSHT)):
            with self.subTest(cls=forward_cls.__name__):
                forward = forward_cls(grid, norm=norm).to(self.device)
                inverse = inverse_cls(grid, norm=norm).to(self.device)
                table = inverse.pct if inverse_cls is th.InverseRealSHT else inverse.dpct
                # the inverse table, transposed back to (..., m, l, k), is not the forward one
                self.assertFalse(torch.equal(forward.weights, table.transpose(-1, -2)))
        # and the raw tables are kept apart by inverse=
        self.assertIsNot(_precompute_legpoly(10, 10, grid, norm=norm, inverse=True), _precompute_legpoly(10, 10, grid, norm=norm, inverse=False))
        # the forward table is the inverse-free table times the quadrature weights: exactly what
        # an unshared computation gives
        forward = th.RealSHT(grid, norm=norm)
        expected = torch.einsum("mlk,k->mlk", _precompute_legpoly(10, 10, grid, norm=norm), 2.0 * torch.pi * grid.colat_weights)
        self.assertTrue(torch.equal(forward.weights, expected))

    def test_sharing_survives_conversions(self):
        for cls, name, a, b, _ in _sht_pairs("cpu"):
            with self.subTest(cls=cls.__name__):
                for convert in (lambda m: m.to(self.device), lambda m: m.float(), lambda m: m.double(), lambda m: m.cpu()):
                    convert(a)
                    convert(b)
                    self.assertIs(getattr(a, name), getattr(b, name))
                self.assertEqual(getattr(a, name).dtype, torch.float64)

    def test_a_lower_precision_copy_stays_with_its_layer(self):
        """A layer rounded through half never hands its rounded table to a layer converted directly."""
        for cls, name, a, _, _ in _sht_pairs(self.device):
            with self.subTest(cls=cls.__name__):
                rounded = deepcopy(a).half().float()
                direct = deepcopy(a).float()
                self.assertIsNot(getattr(direct, name), getattr(rounded, name))
                self.assertTrue(torch.equal(getattr(direct, name), getattr(a, name).float()))
                self.assertTrue(torch.equal(getattr(rounded, name), getattr(a, name).half().float()))

    def test_a_transform_returns_from_the_meta_device(self):
        """to_empty restores the table, which as a non-persistent buffer no state dict would."""
        grid = as_grid("lobatto", nlat=9, nlon=18)
        for cls, name in ((th.RealSHT, "weights"), (th.InverseRealSHT, "pct"), (th.RealVectorSHT, "weights"), (th.InverseRealVectorSHT, "dpct")):
            with self.subTest(cls=cls.__name__):
                layer = cls(grid).to("meta")
                gc.collect()
                layer.to_empty(device=self.device)
                self.assertEqual(getattr(layer, name).device.type, torch.device(self.device).type)
                self.assertTrue(torch.equal(getattr(layer, name).cpu(), getattr(cls(grid), name)))

    def test_a_new_transform_reuses_a_live_device_copy(self):
        """Constructed while an identical transform lives on the device, a transform shares its table there."""
        for cls, name, a, _, _ in _sht_pairs(self.device):
            with self.subTest(cls=cls.__name__):
                gc.collect()
                c = cls(a.grid)
                self.assertTrue(torch.equal(getattr(c, name), getattr(a, name).cpu()))
                self.assertIs(getattr(c.to(self.device), name), getattr(a, name))

    def test_a_deep_copy_shares_the_table(self):
        for cls, name, a, _, _ in _sht_pairs(self.device):
            with self.subTest(cls=cls.__name__):
                self.assertIs(getattr(deepcopy(a), name), getattr(a, name))

    def test_the_table_is_freed_with_the_last_transform(self):
        grid = as_grid("lobatto", nlat=11, nlon=22)
        layers = [th.RealSHT(grid).to(self.device) for _ in range(3)]
        ref = weakref.ref(layers[0].weights)
        layers.pop()
        layers.pop()
        gc.collect()
        self.assertIsNotNone(ref())
        layers.clear()
        gc.collect()
        self.assertIsNone(ref(), "the table outlived every transform using it")

    def test_results_are_unchanged(self):
        """Sharing is storage only: a transform computes exactly what an unshared one would."""
        grid = as_grid("equiangular", nlat=13, nlon=26)
        a = th.RealSHT(grid).to(self.device)
        b = th.RealSHT(grid).to(self.device)
        unshared = deepcopy(a)
        unshared.weights = a.weights.clone()
        x = torch.randn(2, 13, 26, device=self.device)
        self.assertTrue(torch.equal(a(x), unshared(x)))
        self.assertTrue(torch.equal(a(x), b(x)))


def _disco(kind, **kwargs):
    if kind == "regular":
        g_in, g_out = as_grid("equiangular", nlat=16, nlon=32), as_grid("equiangular", nlat=8, nlon=16)
    else:
        g_in, g_out = th.HealpixGrid(nside=4), th.HealpixGrid(nside=2)
    return th.DiscreteContinuousConvS2(g_in, g_out, 4, 6, kernel_shape=3, **kwargs)


def _disco_transpose(kind, **kwargs):
    if kind == "regular":
        g_in, g_out = as_grid("equiangular", nlat=8, nlon=16), as_grid("equiangular", nlat=16, nlon=32)
    else:
        g_in, g_out = th.HealpixGrid(nside=2), th.HealpixGrid(nside=4)
    return th.DiscreteContinuousConvTransposeS2(g_in, g_out, 4, 6, kernel_shape=3, **kwargs)


def _attention(kind, **kwargs):
    if kind == "regular":
        g_in, g_out = as_grid("equiangular", nlat=16, nlon=32), as_grid("equiangular", nlat=8, nlon=16)
    else:
        g_in, g_out = th.HealpixGrid(nside=4), th.HealpixGrid(nside=2)
    return th.NeighborhoodAttentionS2(g_in, g_out, 8, num_heads=2, **kwargs)


_BACKEND_LAYERS = [[name, build, kind] for name, build in (("disco", _disco), ("disco-transpose", _disco_transpose), ("attention", _attention)) for kind in ("regular", "ragged")]


@parameterized_class(("device"), _devices)
class TestSharedBackendState(unittest.TestCase):
    """Identical DISCO and attention layers share the state their backend prepared."""

    def _assert_shared(self, a, b):
        self.assertIs(a.backend, b.backend)
        self.assertEqual(a._backend_state, b._backend_state)
        for name in a._backend_state:
            self.assertIs(getattr(a, name), getattr(b, name), msg=name)

    @parameterized.expand(_BACKEND_LAYERS)
    def test_identical_layers_share_backend_state(self, _, build, kind):
        a = build(kind).to(self.device)
        b = build(kind).to(self.device)
        self._assert_shared(a, b)
        # the parameters are the layer's own
        self.assertIsNot(next(a.parameters()), next(b.parameters()))

    @parameterized.expand(_BACKEND_LAYERS)
    def test_shared_state_is_what_the_backend_prepares(self, _, build, kind):
        """
        The guard against an incomplete key: state found in the registry must equal what the
        backend prepares for this very layer from scratch.
        """
        a = build(kind).to(self.device)
        b = build(kind).to(self.device)
        fresh = type(b.backend)().prepare(b, b.device)
        self.assertEqual(set(fresh), set(b._backend_state))
        for name, tensor in fresh.items():
            self.assertTrue(torch.equal(getattr(b, name), tensor), msg=name)
        self._assert_shared(a, b)

    @parameterized.expand(_BACKEND_LAYERS)
    def test_a_different_support_is_not_shared(self, _, build, kind):
        a = build(kind).to(self.device)
        b = build(kind, theta_cutoff=1.5 * a.theta_cutoff).to(self.device)
        self.assertTrue(any(getattr(a, name) is not getattr(b, name) for name in a._backend_state))

    @parameterized.expand(_BACKEND_LAYERS)
    def test_layers_moved_together_share_state_on_the_target(self, _, build, kind):
        model = torch.nn.ModuleList([build(kind) for _ in range(3)]).to(self.device)
        for layer in model[1:]:
            self._assert_shared(model[0], layer)
        self.assertEqual(model[0].device.type, self.device.type)

    @parameterized.expand(_BACKEND_LAYERS)
    def test_a_deep_copy_shares_state_but_not_parameters(self, _, build, kind):
        a = build(kind).to(self.device)
        b = deepcopy(a)
        self._assert_shared(a, b)
        self.assertIsNot(next(a.parameters()), next(b.parameters()))

    def test_attention_layers_of_different_dtype_do_not_share(self):
        """The reference prepares float64 weights for a float64 layer; the dtype is part of the key."""
        a = _attention("regular").to(self.device)
        b = _attention("regular").to(self.device).double()
        self.assertEqual(b.dtype, torch.float64)
        self.assertTrue(any(getattr(a, name) is not getattr(b, name) for name in a._backend_state))

    def test_forward_and_transpose_do_not_share(self):
        """Mirrored grids give psi of the same extents; the class tells them apart."""
        grid = as_grid("equiangular", nlat=8, nlon=16)
        a = th.DiscreteContinuousConvS2(grid, grid, 4, 6, kernel_shape=3).to(self.device)
        b = th.DiscreteContinuousConvTransposeS2(grid, grid, 4, 6, kernel_shape=3).to(self.device)
        self.assertTrue(any(getattr(a, name) is not getattr(b, name) for name in a._backend_state))

    def test_the_attention_point_weights_are_shared(self):
        grid = as_grid("legendre-gauss", nlat=8, nlon=16)
        a = th.AttentionS2(grid, grid, 8, num_heads=2).to(self.device)
        b = th.AttentionS2(grid, grid, 8, num_heads=2).to(self.device)
        self.assertIs(a.log_point_weights, b.log_point_weights)
