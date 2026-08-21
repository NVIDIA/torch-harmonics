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

import math
import unittest
from copy import deepcopy

import torch
from testutils import compare_tensors

from torch_harmonics.cache import lru_cache
from torch_harmonics.grid import as_grid, grid_types
from torch_harmonics.quadrature import clenshaw_curtiss_weights, legendre_gauss_weights


class TestCacheConsistency(unittest.TestCase):
    def test_consistency(self, verbose=False):
        from torch_harmonics.legendre import _precompute_legpoly

        with torch.no_grad():
            cost = torch.cos(torch.linspace(0.0, 2.0 * math.pi, 10, dtype=torch.float64))
            leg1 = _precompute_legpoly(10, 10, cost)
            # perform in-place modification of leg1
            leg1 *= -1.0
            leg2 = _precompute_legpoly(10, 10, cost)
            self.assertFalse(torch.allclose(leg1, leg2))

    def test_cache_tensor(self, verbose=False):
        from torch_harmonics.legendre import _precompute_legpoly

        with torch.no_grad():
            # compute legpoly with given cost
            cost, _ = legendre_gauss_weights(10, -1, 1)
            tq = torch.flip(torch.arccos(cost), dims=(0,))
            leg1 = _precompute_legpoly(10, 10, tq)
            # compute legpoly with different cost
            cost, _ = clenshaw_curtiss_weights(10, -1, 1)
            tq = torch.flip(torch.arccos(cost), dims=(0,))
            leg2 = _precompute_legpoly(10, 10, tq)
            self.assertFalse(torch.allclose(leg1, leg2))


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

        for grid_type in grid_types():
            with self.subTest(grid=grid_type):
                calls.clear()
                first = _expensive(as_grid(grid_type, (64, 128)))
                second = _expensive(as_grid(grid_type, (64, 128)))
                self.assertEqual(first, second)
                self.assertEqual(len(calls), 1, msg=f"grid={grid_type}: an equal-but-distinct descriptor missed the cache")

    def test_cache_distinguishes_grids_that_differ(self):
        calls = []

        @lru_cache(maxsize=20, typed=True)
        def _expensive(grid):
            calls.append(grid)
            return grid.max_latitude_spacing

        distinct = [
            as_grid("equiangular", (64, 128)),
            as_grid("equiangular", (65, 128)),
            as_grid("equiangular", (64, 256)),
            as_grid("lobatto", (64, 128)),
        ]
        for grid in distinct:
            _expensive(grid)
        # a second sweep must be served entirely from the cache
        for grid in distinct:
            _expensive(grid)

        self.assertEqual(len(calls), len(distinct), msg="descriptors differing in grid type or resolution collided or missed")

    def test_descriptor_survives_deepcopy(self):
        """``lru_cache(copy=True)`` deep-copies its return value, so a cached descriptor must remain a valid key."""
        for grid_type in grid_types():
            with self.subTest(grid=grid_type):
                grid = as_grid(grid_type, (64, 128))
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
            for grid_type in grid_types():
                with self.subTest(grid=grid_type):
                    grid = as_grid(grid_type, (32, 64))
                    pristine_lats, pristine_weights = grid.lats.clone(), grid.quad_weights.clone()

                    grid.lats.mul_(-1.0)
                    grid.quad_weights.mul_(-1.0)
                    grid.lons().mul_(-1.0)

                    self.assertTrue(compare_tensors(f"lats after external mutation (grid={grid_type})", grid.lats, pristine_lats, atol=0.0, rtol=0.0, verbose=verbose))
                    self.assertTrue(
                        compare_tensors(f"weights after external mutation (grid={grid_type})", grid.quad_weights, pristine_weights, atol=0.0, rtol=0.0, verbose=verbose)
                    )
                    self.assertGreaterEqual(grid.lons().min().item(), 0.0, msg=f"grid={grid_type}: longitudes were corrupted by an external in-place write")

    def test_descriptor_returns_independent_tensors(self):
        """Two accesses must not alias, otherwise one consumer's in-place op leaks into another's."""
        for grid_type in grid_types():
            with self.subTest(grid=grid_type):
                grid = as_grid(grid_type, (32, 64))
                first, second = grid.lats, grid.lats
                self.assertIsNot(first, second)
                self.assertNotEqual(first.data_ptr(), second.data_ptr(), msg=f"grid={grid_type}: repeated access to lats returned aliased storage")

    def test_derived_scalars_are_consistent_across_accesses(self):
        """``max_latitude_spacing`` is itself cached; repeated access must be stable and match the nodes."""
        for grid_type in grid_types():
            with self.subTest(grid=grid_type):
                grid = as_grid(grid_type, (65, 128))
                first = grid.max_latitude_spacing
                _ = grid.lats.mul_(-1.0)  # try to poison the underlying node cache
                self.assertEqual(grid.max_latitude_spacing, first)
                self.assertEqual(grid.max_latitude_spacing, grid.latitude_spacing.max().item())


if __name__ == "__main__":
    unittest.main()
