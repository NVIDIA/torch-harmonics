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


import unittest
from dataclasses import dataclass

import numpy as np
import torch
from parameterized import parameterized

from torch_harmonics import GridS2, as_grid, grid_types

try:
    import matplotlib

    matplotlib.use("Agg")
    import cartopy  # noqa: F401
    import matplotlib.pyplot as plt

    from torch_harmonics.plotting import plot_sphere

    _PLOTTING_AVAILABLE = True
except ImportError:
    _PLOTTING_AVAILABLE = False

_GRIDS = ["equiangular", "legendre-gauss", "lobatto", "trapezoidal"]


@unittest.skipUnless(_PLOTTING_AVAILABLE, "matplotlib and cartopy are required for the plotting tests")
class TestPlotSphereGrid(unittest.TestCase):
    """
    The grid descriptor is what places the samples. Only the equiangular grid is
    equispaced in latitude, so the fallback placement is wrong for every other
    grid -- and wrong silently, since the picture still renders.
    """

    def setUp(self):
        self.nlat, self.nlon = 32, 64
        self.data = torch.randn(self.nlat, self.nlon)

    def tearDown(self):
        plt.close("all")

    def _latitudes(self, **kwargs):
        """Latitudes, in degrees, that the quadmesh actually placed the data at."""
        im = plot_sphere(self.data, fig=plt.figure(), **kwargs)
        return np.asarray(im.get_coordinates())[..., 1]

    @parameterized.expand(_GRIDS)
    def test_grid_places_samples_at_the_grid_latitudes(self, grid_type):
        grid = as_grid(grid_type, nlat=self.nlat, nlon=self.nlon)
        expected = self._latitudes(lat=(np.pi / 2.0 - grid.lats).numpy())
        self.assertTrue(np.allclose(self._latitudes(grid=grid), expected))

    @parameterized.expand(_GRIDS)
    def test_grid_accepts_a_string(self, grid_type):
        grid = as_grid(grid_type, nlat=self.nlat, nlon=self.nlon)
        self.assertTrue(np.allclose(self._latitudes(grid=grid_type), self._latitudes(grid=grid)))

    @parameterized.expand(_GRIDS)
    def test_fallback_placement_only_agrees_for_equiangular(self, grid_type):
        """Pins the reason the parameter exists: the fallback is not a harmless default."""
        offset = np.abs(self._latitudes(grid=grid_type) - self._latitudes()).max()
        if grid_type == "equiangular":
            self.assertLess(offset, 1e-9, msg="the equiangular grid is equispaced in latitude, so it must match the fallback")
        else:
            self.assertGreater(offset, 1.0, msg=f"grid={grid_type}: the fallback placement should be visibly wrong, by degrees")

    def test_rows_run_north_to_south_without_flipping(self):
        """Transform output can be handed over directly; the contract is ascending co-latitude."""
        for grid_type in _GRIDS:
            with self.subTest(grid=grid_type):
                lats = self._latitudes(grid=grid_type)[:, 0]
                self.assertGreater(lats[0], lats[-1], msg=f"grid={grid_type}: row 0 must be the northernmost")

    def test_grid_and_explicit_coordinates_are_mutually_exclusive(self):
        grid = as_grid("legendre-gauss", nlat=self.nlat, nlon=self.nlon)
        for coords in ({"lat": np.zeros(self.nlat)}, {"lon": np.zeros(self.nlon)}):
            with self.subTest(**coords), self.assertRaises(ValueError):
                plot_sphere(self.data, fig=plt.figure(), grid=grid, **coords)

    def test_grid_must_match_the_data_shape(self):
        grid = as_grid("legendre-gauss", nlat=self.nlat // 2, nlon=self.nlon // 2)
        with self.assertRaises(ValueError):
            plot_sphere(self.data, fig=plt.figure(), grid=grid)

    def test_non_regular_grid_is_rejected(self):
        """
        A ragged grid has no single longitude vector, so pcolormesh cannot draw it.

        Regularity is a property of the *type* rather than a runtime flag: every
        grid that carries one longitude vector per ring is a :class:`RegularGridS2`,
        and ``plot_sphere`` demands one through :func:`require_regular_grid`. So this
        builds an actual ragged grid rather than patching ``is_regular`` on a regular
        one, which would no longer reach the guard.
        """

        @dataclass(frozen=True, eq=False)
        class _RaggedGrid(GridS2):
            """Rings of unequal length; stands in for a reduced Gaussian or HEALPix grid."""

            @property
            def nrings(self):
                return 3

            @property
            def nlon_per_lat(self):
                return torch.tensor([4, 8, 4], dtype=torch.int64)

            @property
            def shape(self):
                return (self.npoints,)

            @property
            def lats(self):
                return torch.linspace(0.25, np.pi - 0.25, 3)

        # Assigned after the class body rather than in it: __init_subclass__ enters
        # a class into the grid registry only when the class body declares its own
        # grid_type, so this makes the stand-in instantiable without publishing it
        # to as_grid(). Tests elsewhere build every registered type with
        # (nlat, nlon), which a ragged grid does not take.
        _RaggedGrid.grid_type = "test-ragged-plotting"

        grid = _RaggedGrid()
        self.assertFalse(grid.is_regular)
        self.assertNotIn(_RaggedGrid.grid_type, grid_types())
        with self.assertRaises(TypeError) as ctx:
            plot_sphere(self.data, fig=plt.figure(), grid=grid)
        self.assertIn("RegularGridS2", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
