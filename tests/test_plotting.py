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
from unittest import mock

import numpy as np
import torch
from parameterized import parameterized

from torch_harmonics import as_grid

try:
    import matplotlib

    matplotlib.use("Agg")
    import cartopy  # noqa: F401
    import matplotlib.pyplot as plt

    from torch_harmonics.plotting import plot_sphere

    _PLOTTING_AVAILABLE = True
except ImportError:
    _PLOTTING_AVAILABLE = False

_GRIDS = ["equiangular", "legendre-gauss", "lobatto", "equiangular-trapezoidal"]


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
        grid = as_grid(grid_type, (self.nlat, self.nlon))
        expected = self._latitudes(lat=(np.pi / 2.0 - grid.lats).numpy())
        self.assertTrue(np.allclose(self._latitudes(grid=grid), expected))

    @parameterized.expand(_GRIDS)
    def test_grid_accepts_a_string(self, grid_type):
        grid = as_grid(grid_type, (self.nlat, self.nlon))
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
        grid = as_grid("legendre-gauss", (self.nlat, self.nlon))
        for coords in ({"lat": np.zeros(self.nlat)}, {"lon": np.zeros(self.nlon)}):
            with self.subTest(**coords), self.assertRaises(ValueError):
                plot_sphere(self.data, fig=plt.figure(), grid=grid, **coords)

    def test_grid_must_match_the_data_shape(self):
        grid = as_grid("legendre-gauss", (self.nlat // 2, self.nlon // 2))
        with self.assertRaises(ValueError):
            plot_sphere(self.data, fig=plt.figure(), grid=grid)

    def test_non_regular_grid_is_rejected(self):
        """A reduced grid has no single longitude vector, so pcolormesh cannot draw it."""
        grid = as_grid("legendre-gauss", (self.nlat, self.nlon))
        with mock.patch.object(type(grid), "is_regular", False):
            with self.assertRaises(NotImplementedError):
                plot_sphere(self.data, fig=plt.figure(), grid=grid)


if __name__ == "__main__":
    unittest.main()
