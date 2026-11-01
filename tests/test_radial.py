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


import math
import unittest

import torch
from parameterized import parameterized, parameterized_class
from testutils import compare_tensors, set_seed

import torch_harmonics as th
from torch_harmonics import GeometricRadialGrid, HealpixGrid, LegendreGaussRadialGrid, SphericalProductGrid, as_grid, as_radial_grid
from torch_harmonics.radial import RadialGrid, fold_radius, split_radial_pair, unfold_radius

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


class TestRadialGrid(unittest.TestCase):
    """The radial rules: their nodes, their weights and their identity."""

    @parameterized.expand([[2], [5], [33]])
    def test_geometric_integrates_inverse_distance_exactly(self, nr):
        """
        The nodes are uniform in t = log(r - r_inner), so (r - r_inner) dt/dr = 1: the
        trapezoidal rule integrates 1 / (r - r_inner) exactly at any nr, which pins down both
        the node placement and the Jacobian the weights carry.
        """
        for r_inner in (0.0, 2.0):
            grid = GeometricRadialGrid(nr=nr, r_min=r_inner + 0.01, r_max=r_inner + 100.0, r_inner=r_inner)
            integral = (grid.quad_weights / (grid.r - r_inner)).sum()
            self.assertAlmostEqual(integral.item(), math.log(1e4), places=12)

    def test_geometric_nodes(self):
        grid = GeometricRadialGrid(nr=9, r_min=1.5, r_max=3.0, r_inner=1.0)
        d = grid.r - 1.0
        self.assertAlmostEqual(grid.r[0].item(), 1.5, places=14)
        self.assertAlmostEqual(grid.r[-1].item(), 3.0, places=14)
        ratio = d[1:] / d[:-1]
        self.assertTrue(torch.allclose(ratio, ratio[0].expand_as(ratio), rtol=1e-13, atol=0))
        self.assertEqual(grid.r.dtype, torch.float64)

    def test_geometric_converges_at_second_order(self):
        """On a non-exact integrand the error falls as the square of the step in log r."""
        errors = []
        for nr in (17, 33, 65):
            grid = GeometricRadialGrid(nr=nr, r_min=0.5, r_max=4.0)
            errors.append(abs(grid.quad_weights.sum().item() - 3.5))
        self.assertAlmostEqual(errors[0] / errors[1], 4.0, delta=0.1)
        self.assertAlmostEqual(errors[1] / errors[2], 4.0, delta=0.1)

    @parameterized.expand([[1], [4], [9]])
    def test_legendre_gauss_is_exact_to_degree_2nr_minus_1(self, nr):
        grid = LegendreGaussRadialGrid(nr=nr, r_inner=0.5, r_outer=2.0)
        degree = 2 * nr - 1
        exact = (2.0 ** (degree + 1) - 0.5 ** (degree + 1)) / (degree + 1)
        self.assertAlmostEqual((grid.quad_weights * grid.r**degree).sum().item() / exact, 1.0, places=12)

    def test_volume_weights_carry_r_squared(self):
        grid = LegendreGaussRadialGrid(nr=4, r_inner=1.0, r_outer=2.0)
        self.assertAlmostEqual(grid.volume_weights.sum().item(), (8.0 - 1.0) / 3.0, places=13)

    @parameterized.expand(
        [
            ["ball", LegendreGaussRadialGrid(nr=4, r_inner=0.0, r_outer=1.0)],
            ["shell", LegendreGaussRadialGrid(nr=4, r_inner=1.0, r_outer=2.0)],
            ["half-line", GeometricRadialGrid(nr=4, r_min=0.1, r_max=10.0)],
            ["exterior", GeometricRadialGrid(nr=4, r_min=1.1, r_max=10.0, r_inner=1.0)],
        ]
    )
    def test_domain_follows_from_the_bounds(self, domain, grid):
        self.assertEqual(grid.domain, domain)

    def test_node_range_is_not_the_domain(self):
        """On the exterior domain the boundary is not a node; the grid keeps both."""
        grid = GeometricRadialGrid(nr=4, r_min=1.1, r_max=10.0, r_inner=1.0)
        self.assertEqual(grid.r_inner, 1.0)
        self.assertGreater(grid.r[0].item(), grid.r_inner)
        self.assertTrue(math.isinf(grid.r_outer))

    @parameterized.expand(
        [
            ["r_min_below_boundary", lambda: GeometricRadialGrid(nr=4, r_min=1.0, r_max=2.0, r_inner=1.0)],
            ["r_max_below_r_min", lambda: GeometricRadialGrid(nr=4, r_min=2.0, r_max=1.0)],
            ["r_max_outside_domain", lambda: GeometricRadialGrid(nr=4, r_min=1.0, r_max=3.0, r_outer=2.0)],
            ["one_geometric_node", lambda: GeometricRadialGrid(nr=1, r_min=1.0, r_max=2.0)],
            ["infinite_r_max", lambda: GeometricRadialGrid(nr=4, r_min=1.0, r_max=math.inf)],
            ["negative_bound", lambda: LegendreGaussRadialGrid(nr=4, r_inner=-1.0, r_outer=1.0)],
            ["unbounded_gauss", lambda: LegendreGaussRadialGrid(nr=4, r_inner=0.0, r_outer=math.inf)],
            ["no_nodes", lambda: LegendreGaussRadialGrid(nr=0, r_inner=0.0, r_outer=1.0)],
            ["float_nr", lambda: LegendreGaussRadialGrid(nr=4.0, r_inner=0.0, r_outer=1.0)],
        ]
    )
    def test_rejects_invalid_parameters(self, _, build):
        with self.assertRaises(ValueError):
            build()

    @parameterized.expand(
        [
            ["geometric_nodes", lambda: GeometricRadialGrid(nr=4, r_min=2.0, r_max=2.0)],
            ["gauss_domain", lambda: LegendreGaussRadialGrid(nr=4, r_inner=1.0, r_outer=1.0)],
            ["geometric_domain", lambda: GeometricRadialGrid(nr=4, r_min=1.5, r_max=2.0, r_inner=1.0, r_outer=1.0)],
        ]
    )
    def test_single_sphere_points_at_the_angular_grid(self, _, build):
        """A radial grid collapsed onto one radius is rejected with a pointer to the angular grid, not just an ordering error."""
        with self.assertRaisesRegex(ValueError, "single sphere.*angular grid alone"):
            build()

    def test_identity(self):
        a = GeometricRadialGrid(nr=8, r_min=0.1, r_max=10)
        b = as_radial_grid("geometric", nr=8, r_min=0.1, r_max=10.0)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertIsInstance(a.r_max, float)
        self.assertNotEqual(a, GeometricRadialGrid(nr=8, r_min=0.1, r_max=10.0, r_inner=0.05))
        self.assertEqual(RadialGrid.from_dict(a.to_dict()), a)

    def test_registry_is_separate_from_the_angular_one(self):
        self.assertIsInstance(as_radial_grid("legendre-gauss", nr=4, r_inner=0.0, r_outer=1.0), LegendreGaussRadialGrid)
        self.assertIsInstance(as_grid("legendre-gauss", nlat=4, nlon=8), th.LegendreGaussGrid)
        self.assertIn("geometric", th.radial_grid_types())
        self.assertNotIn("geometric", th.grid_types())
        with self.assertRaisesRegex(ValueError, "Did you mean 'geometric'"):
            as_radial_grid("geometrik", nr=4)

    def test_returns_independent_tensors(self):
        grid = LegendreGaussRadialGrid(nr=4, r_inner=0.0, r_outer=1.0)
        grid.r.zero_()
        self.assertGreater(grid.r.min().item(), 0.0)


class TestSphericalProductGrid(unittest.TestCase):

    def setUp(self):
        self.angular = as_grid("legendre-gauss", nlat=8, nlon=16)
        self.radial = LegendreGaussRadialGrid(nr=3, r_inner=1.0, r_outer=2.0)
        self.grid = SphericalProductGrid(self.angular, self.radial)

    def test_shape(self):
        self.assertEqual(self.grid.shape, (3, 8, 16))
        self.assertEqual(self.grid.npoints, 3 * 128)
        self.assertEqual(SphericalProductGrid(HealpixGrid(nside=2), self.radial).shape, (3, 48))

    def test_quad_weights_integrate_the_volume(self):
        weights = self.grid.quad_weights
        self.assertEqual(tuple(weights.shape), (3, 128))
        self.assertAlmostEqual(weights.sum().item(), 4.0 / 3.0 * math.pi * (8.0 - 1.0), places=10)

    def test_identity(self):
        same = SphericalProductGrid(as_grid("legendre-gauss", nlat=8, nlon=16), LegendreGaussRadialGrid(nr=3, r_inner=1.0, r_outer=2.0))
        self.assertEqual(self.grid, same)
        self.assertEqual(hash(self.grid), hash(same))
        self.assertNotEqual(self.grid, SphericalProductGrid(self.angular, LegendreGaussRadialGrid(nr=4, r_inner=1.0, r_outer=2.0)))
        self.assertEqual(SphericalProductGrid.from_dict(self.grid.to_dict()), self.grid)

    def test_is_not_a_point_set(self):
        """S2-only consumers must not mistake the volumetric shape and weights for angular ones."""
        self.assertNotIsInstance(self.grid, th.PointSetS2)
        with self.assertRaises(TypeError):
            th.require_point_set(self.grid)

    def test_rejects_invalid_factors(self):
        with self.assertRaises(TypeError):
            SphericalProductGrid(self.angular.shard(polar=(0, 2)), self.radial)
        with self.assertRaises(TypeError):
            SphericalProductGrid(self.angular, self.angular)

    def test_pair_must_share_the_radial_grid(self):
        self.assertEqual(split_radial_pair(self.grid, self.grid), (self.angular, self.angular, self.radial))
        self.assertEqual(split_radial_pair(self.angular, self.angular), (self.angular, self.angular, None))
        other = SphericalProductGrid(self.angular, LegendreGaussRadialGrid(nr=4, r_inner=1.0, r_outer=2.0))
        for grid_in, grid_out in [(self.grid, other), (self.grid, self.angular), (self.angular, self.grid)]:
            with self.assertRaisesRegex(ValueError, "same radial grid"):
                th.ResampleS2(grid_in, grid_out)

    @parameterized.expand([["regular", as_grid("legendre-gauss", nlat=8, nlon=16)], ["healpix", HealpixGrid(nside=2)]])
    def test_volume_quadrature(self, _, angular):
        grid = SphericalProductGrid(angular, self.radial)
        quad = th.QuadratureVolume(grid)
        # the shell volume, and the integral of r, which the radial rule integrates exactly
        self.assertAlmostEqual(quad(torch.ones(grid.shape)).item(), 4.0 / 3.0 * math.pi * 7.0, places=4)
        r = self.radial.r.to(torch.float32).reshape(-1, *([1] * len(angular.shape)))
        self.assertAlmostEqual(quad(r.expand(grid.shape)).item(), math.pi * 15.0, places=4)
        self.assertAlmostEqual(th.QuadratureVolume(grid, normalize=True)(torch.ones(2, *grid.shape))[1].item(), 1.0, places=6)

    @parameterized.expand([["tiny", 1e-16], ["huge", 1e20]])
    def test_volume_quadrature_keeps_the_weights_range(self, _, scale):
        """r^2 w scales like r^3, outside float32's range at both ends; a float64 field integrates in float64."""
        grid = SphericalProductGrid(self.angular, LegendreGaussRadialGrid(nr=3, r_inner=0.0, r_outer=scale))
        volume = th.QuadratureVolume(grid)(torch.ones(grid.shape, dtype=torch.float64))
        self.assertEqual(volume.dtype, torch.float64)
        self.assertAlmostEqual(volume.item() / (4.0 / 3.0 * math.pi * scale**3), 1.0, places=12)

    @parameterized.expand(
        [
            [f"{domain}_{name}", angular, r_inner, r_outer]
            for domain, r_inner, r_outer in [("ball", 0.0, 1.0), ("shell", 1.0, 2.0), ("thin_shell", 10.0, 10.5)]
            for name, angular in [
                ("legendre_gauss", as_grid("legendre-gauss", nlat=8, nlon=16)),
                ("equiangular", as_grid("equiangular", nlat=9, nlon=16)),
                ("healpix", HealpixGrid(nside=2)),
            ]
        ]
    )
    def test_volume_of_the_domain(self, _, angular, r_inner, r_outer):
        """The volume of ones is 4/3 pi (r_outer^3 - r_inner^3); Gauss with nr >= 2 integrates the r^2 exactly."""
        grid = SphericalProductGrid(angular, LegendreGaussRadialGrid(nr=2, r_inner=r_inner, r_outer=r_outer))
        volume = th.QuadratureVolume(grid)(torch.ones(grid.shape, dtype=torch.float64))
        expected = 4.0 / 3.0 * math.pi * (r_outer**3 - r_inner**3)
        self.assertAlmostEqual(volume.item() / expected, 1.0, places=12)
        # and the descriptor's volume weights, without the layer
        self.assertAlmostEqual(grid.quad_weights.sum().item() / expected, 1.0, places=12)

    def test_volume_quadrature_sums_the_shells(self):
        x = torch.randn(2, 5, *self.grid.shape)
        shells = th.QuadratureS2(self.grid)(x)
        expected = (shells * self.radial.volume_weights.to(torch.float32)).sum(-1)
        self.assertTrue(torch.allclose(th.QuadratureVolume(self.grid)(x), expected, rtol=1e-5, atol=1e-5))

    def test_volume_quadrature_needs_a_product(self):
        with self.assertRaisesRegex(TypeError, "SphericalProductGrid"):
            th.QuadratureVolume(self.angular)

    def test_fold_round_trip_copies_once(self):
        x = torch.randn(2, 4, 3, 8, 16)
        folded = fold_radius(x)
        self.assertEqual(tuple(folded.shape), (6, 4, 8, 16))
        unfolded = unfold_radius(folded, 3)
        self.assertTrue(torch.equal(unfolded, x))
        # a chain of layers on the product folds for free after the first one
        self.assertEqual(fold_radius(unfolded).data_ptr(), folded.data_ptr())


_RADIAL = LegendreGaussRadialGrid(nr=3, r_inner=1.0, r_outer=2.0)
_REGULAR = as_grid("legendre-gauss", nlat=8, nlon=16)
_HEALPIX = HealpixGrid(nside=2)
_C = 4
_C_OUT = 6


# Each case builds a layer from a pair of grids, and states where the radial axis sits in
# its input and in its output, the input's shape without it, and its dtype.

_LAYERS = [
    ["sht", lambda gi, go: th.RealSHT(gi), 1, 1, lambda g, l: (2, *g.shape), torch.float32],
    ["inverse_sht", lambda gi, go: th.InverseRealSHT(gi), 1, 1, lambda g, l: (2, l.lmax, l.mmax), torch.complex64],
    ["vector_sht", lambda gi, go: th.RealVectorSHT(gi), 2, 2, lambda g, l: (2, 2, *g.shape), torch.float32],
    ["inverse_vector_sht", lambda gi, go: th.InverseRealVectorSHT(gi), 2, 2, lambda g, l: (2, 2, l.lmax, l.mmax), torch.complex64],
    ["quadrature", lambda gi, go: th.QuadratureS2(gi), 1, 1, lambda g, l: (2, *g.shape), torch.float32],
    ["quadrature_healpix", lambda gi, go: th.QuadratureS2(gi), 1, 1, lambda g, l: (2, *g.shape), torch.float32],
    ["resample", lambda gi, go: th.ResampleS2(gi, go), 1, 1, lambda g, l: (2, *g.shape), torch.float32],
    ["spectral_conv", lambda gi, go: th.SpectralConvS2(gi, go, _C, _C_OUT, bias=True), 2, 2, lambda g, l: (2, _C, *g.shape), torch.float32],
    ["disco", lambda gi, go: th.DiscreteContinuousConvS2(gi, go, _C, _C_OUT, kernel_shape=3), 2, 2, lambda g, l: (2, _C, *g.shape), torch.float32],
    ["disco_healpix", lambda gi, go: th.DiscreteContinuousConvS2(gi, go, _C, _C_OUT, kernel_shape=3), 2, 2, lambda g, l: (2, _C, *g.shape), torch.float32],
    ["disco_transpose", lambda gi, go: th.DiscreteContinuousConvTransposeS2(gi, go, _C, _C_OUT, kernel_shape=3), 2, 2, lambda g, l: (2, _C, *g.shape), torch.float32],
    ["attention", lambda gi, go: th.AttentionS2(gi, go, _C, out_channels=_C_OUT, num_heads=2), 2, 2, lambda g, l: (2, _C, *g.shape), torch.float32],
    ["neighborhood_attention", lambda gi, go: th.NeighborhoodAttentionS2(gi, go, _C, out_channels=_C_OUT, num_heads=2), 2, 2, lambda g, l: (2, _C, *g.shape), torch.float32],
    [
        "neighborhood_attention_healpix",
        lambda gi, go: th.NeighborhoodAttentionS2(gi, go, _C, out_channels=_C_OUT, num_heads=2),
        2,
        2,
        lambda g, l: (2, _C, *g.shape),
        torch.float32,
    ],
]


def _close(msg, actual, expected):
    # the batched and the per-shell contractions sum in different orders, so an entry that
    # cancels to well below the field's scale differs by roundoff of that scale, not of its own
    return compare_tensors(msg, actual, expected, atol=1e-5 * expected.abs().max().item(), rtol=1e-5)


def _loss(y):
    return (torch.view_as_real(y) if y.is_complex() else y).square().sum()


def _grids(name):
    if name.endswith("healpix"):
        return _HEALPIX, _HEALPIX
    if name == "resample":
        return _REGULAR, as_grid("equiangular", nlat=10, nlon=16)
    return _REGULAR, _REGULAR


@parameterized_class(("device",), _devices)
class TestLayersOnProductGrids(unittest.TestCase):
    """A layer on a product grid computes, on every shell, what it computes on the angular grid alone."""

    def setUp(self):
        set_seed(333)

    @parameterized.expand(_LAYERS)
    def test_acts_on_each_shell_alike(self, name, build, rdim_in, rdim_out, input_shape, dtype):
        angular_in, angular_out = _grids(name)
        layer = build(SphericalProductGrid(angular_in, _RADIAL), SphericalProductGrid(angular_out, _RADIAL)).to(self.device)
        reference = build(angular_in, angular_out).to(self.device)

        # randomize every parameter, so a bias or a weight that the product path misapplies
        # cannot hide behind its initialization; the reference shares them
        with torch.no_grad():
            for p in layer.parameters():
                p.add_(0.1 * torch.randn_like(p))
        reference.load_state_dict(layer.state_dict())

        shape = list(input_shape(angular_in, reference))
        shape.insert(rdim_in, _RADIAL.nr)
        x = torch.randn(shape, device=self.device, dtype=dtype)

        out = layer(x)
        for k in range(_RADIAL.nr):
            expected = reference(x.select(rdim_in, k))
            self.assertTrue(_close(f"{name} output on shell {k}", out.select(rdim_out, k), expected))

    @parameterized.expand([[name, build, rdim_in, rdim_out, input_shape, dtype] for name, build, rdim_in, rdim_out, input_shape, dtype in _LAYERS if dtype == torch.float32])
    def test_gradient_matches_per_shell(self, name, build, rdim_in, rdim_out, input_shape, dtype):
        """The fold into the batch and back is differentiated like any other view."""
        angular_in, angular_out = _grids(name)
        layer = build(SphericalProductGrid(angular_in, _RADIAL), SphericalProductGrid(angular_out, _RADIAL)).to(self.device)
        reference = build(angular_in, angular_out).to(self.device)
        reference.load_state_dict(layer.state_dict())

        shape = list(input_shape(angular_in, reference))
        shape.insert(rdim_in, _RADIAL.nr)
        x = torch.randn(shape, device=self.device, dtype=dtype, requires_grad=True)
        _loss(layer(x)).backward()

        for k in range(_RADIAL.nr):
            xk = x.detach().select(rdim_in, k).clone().requires_grad_(True)
            _loss(reference(xk)).backward()
            self.assertTrue(_close(f"{name} input gradient on shell {k}", x.grad.select(rdim_in, k), xk.grad))

    def test_folded_fejer_analysis_acts_on_each_shell(self):
        angular = as_grid("fejer1", nlat=10, nlon=20)
        product = SphericalProductGrid(angular, _RADIAL)
        cases = [
            ("scalar", lambda grid: th.RealSHT(grid, lmax=10, mmax=10), 1, (2, _RADIAL.nr, *angular.shape)),
            ("vector", lambda grid: th.RealVectorSHT(grid, lmax=10, mmax=10), 2, (2, 2, _RADIAL.nr, *angular.shape)),
        ]

        for name, build, radial_dim, shape in cases:
            with self.subTest(transform=name):
                layer = build(product).to(self.device)
                reference = build(angular).to(self.device)
                self.assertTrue(layer._extended_folded)
                x = torch.randn(shape, device=self.device, dtype=torch.float32, requires_grad=True)

                out = layer(x)
                _loss(out).backward()
                for k in range(_RADIAL.nr):
                    xk = x.detach().select(radial_dim, k).clone().requires_grad_(True)
                    expected = reference(xk)
                    _loss(expected).backward()
                    self.assertTrue(_close(f"{name} output on shell {k}", out.detach().select(radial_dim, k), expected.detach()))
                    self.assertTrue(_close(f"{name} gradient on shell {k}", x.grad.select(radial_dim, k), xk.grad))

    def test_repr_names_the_radial_grid(self):
        layer = th.RealSHT(SphericalProductGrid(_REGULAR, _RADIAL))
        self.assertIn(repr(_RADIAL), repr(layer))


if __name__ == "__main__":
    unittest.main()
