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
        expected = self._latitudes(lat=grid.lats.numpy())
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

    @staticmethod
    def _ragged_grid():
        """Rings of 4, 8 and 4 longitudes; stands in for a reduced Gaussian or HEALPix grid."""

        @dataclass(frozen=True, eq=False)
        class _RaggedGrid(GridS2):
            @property
            def nrings(self):
                return 3

            @property
            def nlon_per_lat(self):
                return torch.tensor([4, 8, 4], dtype=torch.int64)

            @property
            def colats(self):
                return torch.linspace(0.25, np.pi - 0.25, 3, dtype=torch.float64)

            @property
            def colat_weights(self):
                return torch.tensor([0.4, 1.2, 0.4], dtype=torch.float64)

            def lons(self, ilat=None):
                n = int(self.nlon_per_lat[ilat])
                return torch.arange(n, dtype=torch.float64) * (2.0 * np.pi / n)

        # Assigned after the class body rather than in it: __init_subclass__ enters
        # a class into the grid registry only when the class body declares its own
        # grid_type, so this makes the stand-in instantiable without publishing it
        # to as_grid(). Tests elsewhere build every registered type with
        # (nlat, nlon), which a ragged grid does not take.
        _RaggedGrid.grid_type = "test-ragged-plotting"
        grid = _RaggedGrid()
        assert grid.npoints == 16 and _RaggedGrid.grid_type not in grid_types()
        return grid

    def test_a_ragged_grid_is_drawn_by_nearest_point(self):
        """
        A ragged field is flat and has no rectangular mesh, so it is resampled onto an
        equiangular image, each cell taking the value of its nearest grid point. With
        the point index as the field, the image says exactly which point each cell drew.
        """
        grid = self._ragged_grid()
        im = plot_sphere(torch.arange(grid.npoints, dtype=torch.float64), fig=plt.figure(), grid=grid)
        image = np.asarray(im.get_array()).reshape(8, 16)

        # rows go to the ring nearest in colatitude: two polar rows each, four equatorial
        for rows, ring in ((slice(0, 2), range(0, 4)), (slice(2, 6), range(4, 12)), (slice(6, 8), range(12, 16))):
            self.assertTrue(set(np.unique(image[rows]).astype(int)) <= set(ring))

        # along a ring, the nearest point in longitude, wrapping at the seam
        cells = np.arange(16)
        self.assertTrue(np.array_equal(image[0], np.floor((cells + 0.5) / 4 + 0.5) % 4))

        # and no point is lost
        self.assertEqual(set(np.unique(image).astype(int)), set(range(grid.npoints)))

    def test_a_ragged_grid_needs_flat_data_of_its_length(self):
        grid = self._ragged_grid()
        for data in (self.data, torch.randn(grid.npoints + 1)):
            with self.subTest(shape=tuple(data.shape)), self.assertRaises(ValueError):
                plot_sphere(data, fig=plt.figure(), grid=grid)

    def test_a_healpix_field_plots_directly(self):
        """The guard runs before the data is read as (nlat, nlon), so a flat field gets through."""
        from torch_harmonics import HealpixGrid

        grid = HealpixGrid(nside=4)
        im = plot_sphere(torch.randn(grid.npoints), fig=plt.figure(), grid=grid, colorbar=True)
        # 192 points at about four cells each: a 20 x 40 equiangular image
        self.assertEqual(np.asarray(im.get_array()).size, 20 * 40)

    def test_a_point_set_without_rings_is_rejected(self):
        from torch_harmonics.grid import PointSetS2

        @dataclass(frozen=True, eq=False)
        class _Points(PointSetS2):
            @property
            def npoints(self):
                return 5

            @property
            def coords(self):
                return torch.zeros(5, 2, dtype=torch.float64)

        _Points.grid_type = "test-points-plotting"
        with self.assertRaises(TypeError) as ctx:
            plot_sphere(torch.randn(5), fig=plt.figure(), grid=_Points())
        self.assertIn("GridS2", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
