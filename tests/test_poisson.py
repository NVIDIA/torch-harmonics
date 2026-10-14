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
from parameterized import parameterized
from testutils import compare_tensors

from torch_harmonics import GeometricRadialGrid, LegendreGaussRadialGrid, SphericalProductGrid, as_grid
from torch_harmonics.examples.poisson_equation import RadialPoissonSolver


def make_solver(nr, rmin=1e-2, rmax=1e3, grid="legendre-gauss", inner_radius=None):
    """A solver on a 16 x 32 angular grid times a geometric radial grid.

    On the half-line rmin and rmax bound r itself. On the exterior domain they bound the
    reduced coordinate (r - R) / R, so the nodes run from R (1 + rmin) to R (1 + rmax).
    """
    if inner_radius is None:
        radial = GeometricRadialGrid(nr, rmin, rmax)
    else:
        radial = GeometricRadialGrid(nr, inner_radius * (1 + rmin), inner_radius * (1 + rmax), r_inner=inner_radius)
    return RadialPoissonSolver(SphericalProductGrid(as_grid(grid, nlat=16, nlon=32), radial))


def gaussian_shell(r, r0=1.0, width=0.2, l=0):
    """A smooth radial profile and the source that produces it on degree l.

    Manufactured solution: pick u, apply the Laplacian analytically, and feed the result
    back in. For u(r) times a spherical harmonic of degree l,

        lap u = (1/r^2) d/dr (r^2 du/dr) - l (l + 1) u / r^2

    so u = exp(-(r - r0)^2 / 2 s^2) gives the expression below. The profile is negligible
    at both ends of the radial grid, which keeps the truncation of the domain out of the
    comparison.
    """
    u = torch.exp(-((r - r0) ** 2) / (2 * width**2))
    f = (-2 * (r - r0) / (r * width**2) - 1 / width**2 + (r - r0) ** 2 / width**4 - l * (l + 1) / r**2) * u
    return u, f


def solve_mode(solver, u_radial, f_radial, l, m):
    """Put the radial profiles on the spherical harmonic (l, m), solve, and return the expected and the actual solution."""
    spec = torch.zeros(2, solver.nr, solver.lmax, solver.mmax, dtype=torch.complex128)
    spec[:, :, l, m] = torch.stack([u_radial, f_radial])
    expected, source = solver.spec2grid(spec)
    return expected, solver.solve(source.unsqueeze(0)).squeeze(0)


def solve_radial(solver, f_radial):
    """Broadcast a radial profile over the sphere, solve, and read one column back."""
    field = f_radial.reshape(-1, 1, 1).expand(solver.nr, solver.nlat, solver.nlon).contiguous()
    return solver.solve(field.unsqueeze(0)).squeeze(0)[:, 0, 0]


class TestRadialPoissonSolver(unittest.TestCase):
    """Correctness of the radial Green's operator against closed-form solutions."""

    @parameterized.expand([[256, 4.8e-3], [512, 1.2e-3], [1024, 3.0e-4]])
    def test_manufactured_solution(self, nr, tol, verbose=False):
        """The solve reproduces a known smooth solution to within the quadrature error."""
        solver = make_solver(nr).double()
        expected, source = gaussian_shell(solver.r)
        actual = solve_radial(solver, source)

        # compare where the profile carries signal; the tails are dominated by round-off
        window = (solver.r > 0.2) & (solver.r < 5.0)
        error = (actual[window] - expected[window]).abs().max() / expected[window].abs().max()
        self.assertLess(error.item(), tol, f"relative error {error.item():.3e} exceeds {tol:.1e} at nr={nr}")

    def test_second_order_convergence(self, verbose=False):
        """Trapezoidal weights in log(r) converge at second order on a smooth source.

        This is the property that pins the Green's kernel itself: a wrong power of r, a
        missing r^2 dr measure or a misplaced 1/(2l+1) still produces a plausible-looking
        field, but not one that converges at the rate of the underlying quadrature.
        """
        errors = []
        for nr in (256, 512, 1024, 2048):
            solver = make_solver(nr).double()
            expected, source = gaussian_shell(solver.r)
            actual = solve_radial(solver, source)
            window = (solver.r > 0.2) & (solver.r < 5.0)
            errors.append(((actual[window] - expected[window]).abs().max() / expected[window].abs().max()).item())

        orders = [math.log2(a / b) for a, b in zip(errors, errors[1:])]
        for order in orders:
            self.assertGreater(order, 1.8, f"observed orders {orders} fall short of second order")

    # domain, inner radius, shell centre and width, and degree. On the exterior domain the
    # shell sits far enough out that the profile vanishes on the inner sphere, as u must,
    # and R is not 1, where every power of it is 1 and the image term's exponent goes unseen
    @parameterized.expand(
        [[domain, inner_radius, r0, width, l] for domain, inner_radius, r0, width in (("half-line", None, 1.0, 0.2), ("exterior", 2.0, 4.0, 0.3)) for l in (1, 2, 5)]
    )
    def test_higher_degree_converges(self, domain, inner_radius, r0, width, l, verbose=False):
        """
        Off degree zero the solve is accurate and converges at second order, on both domains.

        A source constant over the sphere only reaches l = 0, where 1/(2l+1) is one and the
        powers r_<^l and the image term's R^(2l+1) drop out, so a mistake in any of them
        passes every other accuracy test.
        """
        errors = []
        for nr in (256, 512, 1024):
            solver = make_solver(nr, rmin=1e-2 if inner_radius is None else 1e-3, inner_radius=inner_radius).double()
            u, f = gaussian_shell(solver.r, r0=r0, width=width, l=l)
            expected, actual = solve_mode(solver, u, f, l, m=1)
            # compare where the profile carries signal; the tails are dominated by round-off
            window = (solver.r - r0).abs() < 4 * width
            errors.append(((actual[window] - expected[window]).abs().max() / expected[window].abs().max()).item())

        orders = [math.log2(a / b) for a, b in zip(errors, errors[1:])]
        if verbose:
            print(domain, l, errors, orders)
        self.assertLess(errors[-1], 2e-3, f"relative error {errors[-1]:.3e} at nr=1024")
        for order in orders:
            self.assertGreater(order, 1.8, f"observed orders {orders} fall short of second order")

    def test_bump_source_needs_a_node_in_every_bump(self):
        """
        A grid whose widest gap could hold a whole bump is refused, rather than sampling
        a source that vanishes on every node and normalizing it into NaNs.
        """
        with self.assertRaisesRegex(ValueError, "too coarse"):
            make_solver(2, rmin=1e-1, rmax=10.0).random_bump_source()
        # the narrowest bump spans 6% of the range, wider than the gaps of 17 intervals
        make_solver(18, rmin=1e-1, rmax=10.0).random_bump_source()

    @parameterized.expand([["legendre-gauss"], ["equiangular"], ["lobatto"]])
    def test_uniform_ball_source(self, grid, verbose=False):
        """A uniform ball reproduces the textbook potential.

        For lap u = 1 inside radius a and 0 outside, with u -> 0 at infinity,
        u = r^2/6 - a^2/2 inside and -a^3/(3 r) outside. The source is discontinuous, so
        the tolerance is loose -- this checks the closed form, not the convergence rate.
        """
        radius = 1.0
        solver = make_solver(512, grid=grid).double()
        actual = solver.solve(solver.ball_source(radius=radius).unsqueeze(0)).squeeze(0)[:, 0, 0]

        r = solver.r
        expected = torch.where(r <= radius, r**2 / 6 - radius**2 / 2, -(radius**3) / (3 * r))
        window = (r > 2e-2) & (r < 1e2)
        self.assertTrue(compare_tensors(f"ball source on {grid}", actual[window], expected[window], atol=2e-2, rtol=1e-2, verbose=verbose))

    @parameterized.expand([[0], [1], [2], [5]])
    def test_exterior_domain_vanishes_on_the_inner_sphere(self, l, verbose=False):
        """
        The image term imposes u = 0 at the inner radius, on every degree.

        Only the boundary value shows the image term: a wrong one differs from it by a
        function harmonic in the source radius, which a manufactured solution vanishing near
        R integrates to zero. R is not 1, where every power of it is 1.
        """
        inner_radius = 2.0
        solver = make_solver(512, rmin=1e-3, rmax=1e3, inner_radius=inner_radius).double()

        # a source near the inner sphere, whose free-space solution is far from zero on it
        source, _ = gaussian_shell(solver.r, r0=2 * inner_radius, width=0.5 * inner_radius)
        _, u = solve_mode(solver, torch.zeros_like(source), source, l, m=min(l, 1))

        # the first node sits just outside the inner sphere, so the solution there is
        # small rather than exactly zero; measure it against the scale of the solution
        ratio = (u[0].abs().max() / u.abs().max()).item()
        self.assertLess(ratio, 1e-2, f"u on the innermost node is {ratio:.2e} of its maximum on degree {l}")

    def test_rejects_bounded_domains(self):
        """The Green's operator assumes u -> 0 at infinity, so a radial grid with a finite outer bound is refused."""
        angular = as_grid("legendre-gauss", nlat=16, nlon=32)
        for radial in (LegendreGaussRadialGrid(16, 0.0, 1.0), LegendreGaussRadialGrid(16, 1.0, 2.0), GeometricRadialGrid(16, 1e-2, 1.0, r_outer=1.0)):
            with self.assertRaisesRegex(NotImplementedError, radial.domain):
                RadialPoissonSolver(SphericalProductGrid(angular, radial))

    def test_requires_a_product_grid(self):
        """The angular grid alone has no radial nodes to solve on."""
        with self.assertRaisesRegex(TypeError, "SphericalProductGrid"):
            RadialPoissonSolver(as_grid("legendre-gauss", nlat=16, nlon=32))

    @parameterized.expand([[torch.float32], [torch.float64]])
    def test_solve_honours_the_input_precision(self, dtype, verbose=False):
        """The Green's kernel is assembled in float64 whatever the caller uses.

        RadialPoissonDataset hands back float32 samples, and einsum rejects a float32 operand
        against a float64 kernel outright, so the solve has to meet the input where it is.
        """
        solver = make_solver(64, rmin=1e-1)
        out = solver.solve(torch.randn(1, 64, 16, 32, dtype=dtype))
        self.assertEqual(out.dtype, dtype)
        self.assertTrue(torch.isfinite(out).all())

    @parameterized.expand([[torch.float32], [torch.float64]])
    def test_converted_solver_honours_the_input_precision(self, dtype, verbose=False):
        """Converting the module must not pin the solve to one dtype either."""
        solver = make_solver(64, rmin=1e-1).float()
        out = solver.solve(torch.randn(1, 64, 16, 32, dtype=dtype))
        self.assertEqual(out.dtype, dtype)
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
