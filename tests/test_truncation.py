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
Tests for SHT mode truncation.

The split of responsibility being pinned here is that a
:class:`~torch_harmonics.grid.GridS2` reports what it can *represent* --
``max_exact_degree`` from the exactness of its quadrature rule, and
``max_azimuthal_order`` from the Nyquist limit of its longitude sampling -- while
:func:`~torch_harmonics.truncation.truncate_sht` decides what an SHT actually
*keeps*: it applies user overrides, enforces the triangular truncation, and warns
where the default changed. The grid properties stay silent so that reading them
is never a side effect.
"""

import unittest
import warnings

import torch
from parameterized import parameterized

import torch_harmonics as th
from torch_harmonics.grid import as_grid
from torch_harmonics.truncation import truncate_sht, truncate_support

# grid type -> expected max_exact_degree as a function of nlat, per the table in
# the truncate_sht docstring
_EXACT_DEGREE = {
    "legendre-gauss": lambda nlat: nlat,
    "lobatto": lambda nlat: nlat - 1,
    "equiangular": lambda nlat: (nlat + 1) // 2,
    "trapezoidal": lambda nlat: (nlat + 1) // 2,
}

# grids whose default lmax changed in v0.9.0 and therefore still announce it
_WARNING_GRIDS = ["equiangular", "trapezoidal"]
_QUIET_GRIDS = ["legendre-gauss", "lobatto"]

_NLATS = [32, 33, 128, 129]


class TestGridSpectralBounds(unittest.TestCase):
    """The facts side: what the grid reports it can represent."""

    @parameterized.expand([[nlat, grid] for nlat in _NLATS for grid in _EXACT_DEGREE])
    def test_max_exact_degree_matches_the_quadrature_rule(self, nlat, grid):
        self.assertEqual(as_grid(grid, nlat=nlat, nlon=2 * nlat).max_exact_degree, _EXACT_DEGREE[grid](nlat))

    @parameterized.expand([[nlon, grid] for nlon in [63, 64, 256] for grid in _EXACT_DEGREE])
    def test_max_azimuthal_order_is_the_nyquist_limit(self, nlon, grid):
        self.assertEqual(as_grid(grid, nlat=32, nlon=nlon).max_azimuthal_order, nlon // 2 + 1)

    @parameterized.expand([[grid] for grid in _EXACT_DEGREE])
    def test_reading_the_bounds_is_silent(self, grid):
        """
        The bounds are facts, not decisions, so they must not warn. The v0.9.0
        notice belongs to truncate_sht, which is where a default is actually chosen.
        """
        g = as_grid(grid, nlat=128, nlon=256)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _ = g.max_exact_degree, g.max_azimuthal_order


class TestTruncateSht(unittest.TestCase):
    """The policy side: what the transform keeps."""

    def setUp(self):
        # These classes exercise the default truncation, so they necessarily take the
        # path that announces the v0.9.0 change, once per equiangular parameterization.
        # It is noise here; test_changed_default_warns and
        # test_unchanged_default_does_not_warn cover the announcement itself, and both
        # override this filter through assertWarns / catch_warnings.
        warnings.filterwarnings("ignore", message="Default SHT truncation changed", category=UserWarning)
        self.addCleanup(warnings.resetwarnings)

    @parameterized.expand(
        [
            # grid, (nlat, nlon), expected (lmax, mmax)
            ["legendre-gauss", (128, 256), (128, 128)],
            ["lobatto", (128, 256), (127, 127)],
            ["equiangular", (128, 256), (64, 64)],
            ["trapezoidal", (128, 256), (64, 64)],
        ]
    )
    def test_documented_defaults(self, grid, shape, expected):
        self.assertEqual(truncate_sht(as_grid(grid, nlat=shape[0], nlon=shape[1])), expected)

    @parameterized.expand([[nlat, grid] for nlat in _NLATS for grid in _EXACT_DEGREE])
    def test_triangular_truncation_is_enforced(self, nlat, grid):
        """``lmax == mmax`` always, so every retained degree has a full set of orders."""
        lmax, mmax = truncate_sht(as_grid(grid, nlat=nlat, nlon=2 * nlat))
        self.assertEqual(lmax, mmax)

    def test_a_narrow_longitude_grid_limits_lmax(self):
        """mmax is the binding constraint when the longitude sampling is coarse."""
        lmax, mmax = truncate_sht(as_grid("legendre-gauss", nlat=128, nlon=16))
        self.assertEqual((lmax, mmax), (9, 9))  # nlon // 2 + 1 = 9, below max_exact_degree = 128

    @parameterized.expand([[grid] for grid in _EXACT_DEGREE])
    def test_user_truncation_overrides_the_grid_default(self, grid):
        """A user must be able to ask for a different truncation than the grid's."""
        g = as_grid(grid, nlat=128, nlon=256)
        self.assertEqual(truncate_sht(g, lmax=20), (20, 20))
        self.assertEqual(truncate_sht(g, mmax=20), (20, 20))
        self.assertEqual(truncate_sht(g, lmax=20, mmax=50), (20, 20))
        self.assertEqual(truncate_sht(g, lmax=50, mmax=20), (20, 20))

    def test_zero_is_treated_as_a_request_not_as_unset(self):
        """
        Regression: the previous implementation used ``lmax or default``, so an
        explicit lmax=0 fell through to the grid default instead of being honoured.
        """
        self.assertEqual(truncate_sht(as_grid("legendre-gauss", nlat=128, nlon=256), lmax=0), (0, 0))
        self.assertEqual(truncate_sht(as_grid("legendre-gauss", nlat=128, nlon=256), mmax=0), (0, 0))

    @parameterized.expand([[grid] for grid in _WARNING_GRIDS])
    def test_changed_default_warns(self, grid):
        with self.assertWarns(UserWarning):
            truncate_sht(as_grid(grid, nlat=128, nlon=256))

    @parameterized.expand([[grid] for grid in _QUIET_GRIDS])
    def test_unchanged_default_does_not_warn(self, grid):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            truncate_sht(as_grid(grid, nlat=128, nlon=256))

    @parameterized.expand([[grid] for grid in _WARNING_GRIDS])
    def test_explicit_lmax_does_not_warn(self, grid):
        """The notice is about the default being chosen for you; it is noise if you did not."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            truncate_sht(as_grid(grid, nlat=128, nlon=256), lmax=32)


class TestShtLayerTruncationAgrees(unittest.TestCase):
    """The SHT layers must not derive a truncation of their own."""

    def setUp(self):
        # These classes exercise the default truncation, so they necessarily take the
        # path that announces the v0.9.0 change, once per equiangular parameterization.
        # It is noise here; test_changed_default_warns and
        # test_unchanged_default_does_not_warn cover the announcement itself, and both
        # override this filter through assertWarns / catch_warnings.
        warnings.filterwarnings("ignore", message="Default SHT truncation changed", category=UserWarning)
        self.addCleanup(warnings.resetwarnings)

    @parameterized.expand([[nlat, grid] for nlat in [32, 33] for grid in ["equiangular", "legendre-gauss", "lobatto"]])
    def test_layers_match_truncate_sht(self, nlat, grid, verbose=False):
        nlon = 2 * nlat
        expected = truncate_sht(as_grid(grid, nlat=nlat, nlon=nlon))
        for cls in [th.RealSHT, th.InverseRealSHT, th.RealVectorSHT, th.InverseRealVectorSHT]:
            with self.subTest(layer=cls.__name__):
                layer = cls(as_grid(grid, nlat=nlat, nlon=nlon))
                self.assertEqual((layer.lmax, layer.mmax), expected)

    @parameterized.expand([[grid] for grid in ["equiangular", "legendre-gauss", "lobatto"]])
    def test_layers_honour_an_explicit_truncation(self, grid):
        layer = th.RealSHT(th.as_grid(grid, nlat=64, nlon=128), lmax=17)
        self.assertEqual((layer.lmax, layer.mmax), (17, 17))

    def test_roundtrip_still_works_at_a_user_truncation(self):
        """A non-default truncation has to remain a usable transform, not just a pair of ints."""
        sht = th.RealSHT(th.as_grid("legendre-gauss", nlat=64, nlon=128), lmax=20)
        isht = th.InverseRealSHT(th.as_grid("legendre-gauss", nlat=64, nlon=128), lmax=20)
        coeffs = torch.zeros(2, 20, 20, dtype=torch.complex128)
        coeffs[:, :10, :10] = torch.randn(2, 10, 10, dtype=torch.complex128)
        signal = isht(coeffs)
        self.assertEqual(signal.shape, (2, 64, 128))
        self.assertEqual(sht(signal).shape, coeffs.shape)


class TestTruncateSupport(unittest.TestCase):
    """
    The spatial counterpart of truncate_sht: the default comes from the grid, an
    explicit value wins, and the descriptor itself stays out of the decision.
    """

    # this class exercises the changed default on purpose
    warnings.filterwarnings("ignore", message="Default theta_cutoff changed", category=UserWarning)

    @parameterized.expand([(g, n) for g in th.grid_types() for n in (16, 33, 64)])
    def test_default_is_one_grid_spacing(self, grid, nlat):
        g = as_grid(grid, nlat=nlat, nlon=2 * nlat)
        self.assertEqual(truncate_support(g), g.max_latitude_spacing)

    @parameterized.expand([(g,) for g in th.grid_types()])
    def test_default_matches_the_descriptor_property(self, grid):
        g = as_grid(grid, nlat=32, nlon=64)
        self.assertEqual(truncate_support(g), g.theta_cutoff())

    @parameterized.expand([(g,) for g in th.grid_types()])
    def test_scale_multiplies_the_default(self, grid):
        g = as_grid(grid, nlat=32, nlon=64)
        self.assertAlmostEqual(truncate_support(g, scale=2.5), 2.5 * truncate_support(g), places=15)

    def test_explicit_value_is_returned_verbatim(self):
        g = as_grid("equiangular", nlat=32, nlon=64)
        self.assertEqual(truncate_support(g, theta_cutoff=0.3), 0.3)

    def test_explicit_value_ignores_scale(self):
        """scale tunes the default; an explicit cutoff is already final."""
        g = as_grid("equiangular", nlat=32, nlon=64)
        self.assertEqual(truncate_support(g, theta_cutoff=0.3, scale=2.0), 0.3)

    @parameterized.expand([(0.0,), (-1.0,)])
    def test_non_positive_is_rejected(self, bad):
        g = as_grid("equiangular", nlat=32, nlon=64)
        with self.assertRaises(ValueError):
            truncate_support(g, theta_cutoff=bad)

    @parameterized.expand([(g, s) for g in th.grid_types() for s in (0.0, -1.0)])
    def test_non_positive_scale_is_rejected(self, grid, bad):
        """The guard is on the radius returned, not on the argument it came from."""
        g = as_grid(grid, nlat=32, nlon=64)
        with self.assertRaises(ValueError):
            truncate_support(g, scale=bad)

    def test_a_shard_is_rejected(self):
        """A radius from a shard's own spacing would differ between ranks."""
        g = as_grid("equiangular", nlat=64, nlon=128)
        with self.assertRaises(TypeError):
            truncate_support(g.shard(polar=(0, 2)))

    def test_the_descriptor_does_not_warn(self):
        """The fact is silent; only the policy announces a changed default."""
        g = as_grid("lobatto", nlat=32, nlon=64)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            g.theta_cutoff()
            g.max_latitude_spacing

    def test_an_explicit_value_does_not_warn(self):
        """Nothing defaulted, so there is no changed default to announce."""
        g = as_grid("lobatto", nlat=32, nlon=64)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            truncate_support(g, theta_cutoff=0.3)

    @parameterized.expand([("lobatto",), ("trapezoidal",), ("legendre-gauss",)])
    def test_the_default_warns_on_non_uniform_grids(self, grid):
        with self.assertWarns(UserWarning):
            truncate_support(as_grid(grid, nlat=32, nlon=64))

    def test_the_default_is_silent_on_equiangular(self):
        """Equiangular spacing is pi/(nlat-1), so nothing changed there."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            truncate_support(as_grid("equiangular", nlat=32, nlon=64))


class TestLayersUseTruncateSupport(unittest.TestCase):
    """The layers must not reimplement the policy they delegate."""

    warnings.filterwarnings("ignore", message="Default theta_cutoff changed", category=UserWarning)

    @parameterized.expand([(g,) for g in th.grid_types()])
    def test_conv_default_comes_from_the_output_grid(self, grid):
        gi, go = as_grid(grid, nlat=32, nlon=64), as_grid(grid, nlat=16, nlon=32)
        conv = th.DiscreteContinuousConvS2(gi, go, 2, 2, kernel_shape=3)
        self.assertEqual(conv.theta_cutoff, truncate_support(go))

    @parameterized.expand([(g,) for g in th.grid_types()])
    def test_transpose_conv_default_comes_from_the_input_grid(self, grid):
        gi, go = as_grid(grid, nlat=16, nlon=32), as_grid(grid, nlat=32, nlon=64)
        conv = th.DiscreteContinuousConvTransposeS2(gi, go, 2, 2, kernel_shape=3)
        self.assertEqual(conv.theta_cutoff, truncate_support(gi))

    def test_attention_default_comes_from_the_coarser_grid(self):
        fine, coarse = as_grid("equiangular", nlat=32, nlon=64), as_grid("equiangular", nlat=16, nlon=32)
        down = th.NeighborhoodAttentionS2(fine, coarse, 2, num_heads=1, optimized_kernel=False)
        up = th.NeighborhoodAttentionS2(coarse, fine, 2, num_heads=1, optimized_kernel=False)
        self.assertEqual(down.theta_cutoff, truncate_support(coarse))
        self.assertEqual(up.theta_cutoff, truncate_support(coarse))

    def test_layers_reject_a_non_positive_cutoff(self):
        g = as_grid("equiangular", nlat=32, nlon=64)
        with self.assertRaises(ValueError):
            th.DiscreteContinuousConvS2(g, g, 2, 2, kernel_shape=3, theta_cutoff=-1.0)


if __name__ == "__main__":
    unittest.main()
