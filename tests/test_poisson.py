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
from unittest import mock

import torch
from parameterized import parameterized, parameterized_class
from testutils import compare_tensors, set_seed

from torch_harmonics import GeometricRadialGrid, LegendreGaussRadialGrid, SphericalProductGrid, as_grid
from torch_harmonics.examples import RadialPoissonDataset
from torch_harmonics.examples.poisson_equation import RadialPoissonSolver

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


def make_solver(nr, rmin=1e-2, rmax=1e3, grid="legendre-gauss", inner_radius=None, dtype=torch.float64, device=torch.device("cpu")):
    """A solver on a 16 x 32 angular grid times a geometric radial grid.

    On the half-line rmin and rmax bound r itself. On the exterior domain they bound the
    reduced coordinate (r - R) / R, so the nodes run from R (1 + rmin) to R (1 + rmax).
    """
    if inner_radius is None:
        radial = GeometricRadialGrid(nr, rmin, rmax)
    else:
        radial = GeometricRadialGrid(nr, inner_radius * (1 + rmin), inner_radius * (1 + rmax), r_inner=inner_radius)
    solver = RadialPoissonSolver(SphericalProductGrid(as_grid(grid, nlat=16, nlon=32), radial))
    # .double() and .float() rather than .to(dtype), which would also cast complex state
    solver = solver.double() if dtype == torch.float64 else solver.float()
    return solver.to(device)


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
    ctype = torch.complex128 if solver.r.dtype == torch.float64 else torch.complex64
    spec = torch.zeros(2, solver.nr, solver.lmax, solver.mmax, dtype=ctype, device=solver.r.device)
    spec[:, :, l, m] = torch.stack([u_radial, f_radial])
    expected, source = solver.spec2grid(spec)
    return expected, solver.solve(source.unsqueeze(0)).squeeze(0)


def solve_radial(solver, f_radial):
    """Broadcast a radial profile over the sphere, solve, and read one column back."""
    field = f_radial.reshape(-1, 1, 1).expand(solver.nr, solver.nlat, solver.nlon).contiguous()
    return solver.solve(field.unsqueeze(0)).squeeze(0)[:, 0, 0]


def relative_error(actual, expected):
    """Largest deviation relative to the largest expected value."""
    return ((actual - expected).abs().max() / expected.abs().max()).item()


@parameterized_class(("device",), _devices)
class TestRadialPoissonSolver(unittest.TestCase):
    """Correctness of the radial Green's operator against closed-form solutions."""

    def setUp(self):
        set_seed(333)

    @parameterized.expand([[256, 4.8e-3], [512, 1.2e-3], [1024, 3.0e-4]])
    def test_manufactured_solution(self, nr, tol, verbose=False):
        """The solve reproduces a known smooth solution to within the quadrature error."""
        solver = make_solver(nr, device=self.device)
        expected, source = gaussian_shell(solver.r)
        actual = solve_radial(solver, source)

        # compare where the profile carries signal; the tails are dominated by round-off
        window = (solver.r > 0.2) & (solver.r < 5.0)
        error = relative_error(actual[window], expected[window])
        self.assertLess(error, tol, f"relative error {error:.3e} exceeds {tol:.1e} at nr={nr}")

    def test_second_order_convergence(self, verbose=False):
        """Trapezoidal weights in log(r) converge at second order on a smooth source.

        This is the property that pins the Green's kernel itself: a wrong power of r, a
        missing r^2 dr measure or a misplaced 1/(2l+1) still produces a plausible-looking
        field, but not one that converges at the rate of the underlying quadrature.
        """
        errors = []
        for nr in (256, 512, 1024, 2048):
            solver = make_solver(nr, device=self.device)
            expected, source = gaussian_shell(solver.r)
            actual = solve_radial(solver, source)
            window = (solver.r > 0.2) & (solver.r < 5.0)
            errors.append(relative_error(actual[window], expected[window]))

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
            solver = make_solver(nr, rmin=1e-2 if inner_radius is None else 1e-3, inner_radius=inner_radius, device=self.device)
            u, f = gaussian_shell(solver.r, r0=r0, width=width, l=l)
            expected, actual = solve_mode(solver, u, f, l, m=1)
            # compare where the profile carries signal; the tails are dominated by round-off
            window = (solver.r - r0).abs() < 4 * width
            errors.append(relative_error(actual[window], expected[window]))

        orders = [math.log2(a / b) for a, b in zip(errors, errors[1:])]
        if verbose:
            print(domain, l, errors, orders)
        self.assertLess(errors[-1], 2e-3, f"relative error {errors[-1]:.3e} at nr=1024")
        for order in orders:
            self.assertGreater(order, 1.8, f"observed orders {orders} fall short of second order")

    @parameterized.expand([["legendre-gauss"], ["equiangular"], ["lobatto"]])
    def test_uniform_ball_source(self, grid, verbose=False):
        """A uniform ball reproduces the textbook potential.

        For lap u = 1 inside radius a and 0 outside, with u -> 0 at infinity,
        u = r^2/6 - a^2/2 inside and -a^3/(3 r) outside. The source is discontinuous, so
        the tolerance is loose -- this checks the closed form, not the convergence rate.
        """
        radius = 1.0
        solver = make_solver(512, grid=grid, device=self.device)
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
        solver = make_solver(512, rmin=1e-3, rmax=1e3, inner_radius=inner_radius, device=self.device)

        # a source near the inner sphere, whose free-space solution is far from zero on it
        source, _ = gaussian_shell(solver.r, r0=2 * inner_radius, width=0.5 * inner_radius)
        _, u = solve_mode(solver, torch.zeros_like(source), source, l, m=min(l, 1))

        # the first node sits just outside the inner sphere, so the solution there is
        # small rather than exactly zero; measure it against the scale of the solution
        ratio = (u[0].abs().max() / u.abs().max()).item()
        self.assertLess(ratio, 1e-2, f"u on the innermost node is {ratio:.2e} of its maximum on degree {l}")

    # solver dtype, input dtype, and the tolerance relative to the solution's maximum. A
    # float32 solve agrees with the float64 one to a few 1e-7; 1e-5 leaves room for the
    # accumulation order of other devices
    @parameterized.expand(
        [
            [torch.float64, torch.float64, 1e-12],
            [torch.float64, torch.float32, 1e-5],
            [torch.float32, torch.float32, 1e-5],
            [torch.float32, torch.float64, 1e-5],
        ]
    )
    def test_solve_in_any_precision(self, solver_dtype, input_dtype, tol, verbose=False):
        """
        The solve runs in the precision of its input, and agrees with a float64 solve.

        The Green's kernel is assembled in float64 and cast at use, so a float32 input is
        solved in float32 even by a float64 solver, and a solver cast to float32 still meets
        a float64 input; in either case the result is the float64 one, rounded.
        """
        reference = make_solver(64, rmin=1e-1, device=self.device)
        solver = make_solver(64, rmin=1e-1, dtype=solver_dtype, device=self.device)
        source = solver.random_bump_source(nblobs=4).to(input_dtype)

        out = solver.solve(source.unsqueeze(0)).squeeze(0)
        expected = reference.solve(source.double().unsqueeze(0)).squeeze(0)

        self.assertEqual(out.dtype, input_dtype)
        error = relative_error(out.double(), expected)
        self.assertLess(error, tol, f"relative error {error:.3e} of a {input_dtype} solve by a {solver_dtype} solver")

    @parameterized.expand([[17, True], [18, False]])
    def test_bump_source_needs_a_node_in_every_bump(self, nr, refused):
        """
        A grid whose widest gap could hold a whole bump is refused, rather than sampling
        a source that vanishes on every node and normalizing it into NaNs.

        The narrowest bump spans 6% of the log-radial range, so the guard is exact between
        16 intervals, 6.25% each, and 17, 5.9% each.
        """
        solver = make_solver(nr, rmin=1e-1, rmax=10.0, device=self.device)
        if refused:
            with self.assertRaisesRegex(ValueError, "too coarse"):
                solver.random_bump_source()
        else:
            self.assertGreater(solver.random_bump_source().abs().max().item(), 0.0)

    @parameterized.expand([[1], [3]])
    def test_bump_source_is_separable_and_band_limited(self, nblobs, verbose=False):
        """
        A bump source is a sum of nblobs radial profiles, each times an angular field of degree at most l_src.

        As a matrix over (radius, angle) it therefore has rank nblobs, which pins
        how each blob's radial profile is combined with its angular coefficients, and its
        spherical harmonic coefficients vanish above l_src.
        """
        l_src = 4
        solver = make_solver(64, rmin=1e-1, device=self.device)
        source = solver.random_bump_source(nblobs=nblobs, l_src=l_src)

        # exactly nblobs: the blobs' profiles and angular fields are drawn independently, so
        # combining them in any other way shows as a lower or a higher rank
        singular = torch.linalg.svdvals(source.reshape(solver.nr, -1).cpu())
        self.assertGreater((singular[nblobs - 1] / singular[0]).item(), 1e-6, f"singular values {singular[: nblobs + 2].tolist()}")
        self.assertLess((singular[nblobs] / singular[0]).item(), 1e-12, f"singular values {singular[: nblobs + 2].tolist()}")

        coeffs = solver.grid2spec(source)
        above = coeffs[:, l_src + 1 :].abs().max() / coeffs.abs().max()
        self.assertLess(above.item(), 1e-12, f"degrees above l_src={l_src} carry {above.item():.2e} of the largest coefficient")

    def test_bump_source_vanishes_outside_its_margins(self):
        """
        No bump reaches into the margins at either end of the log-radial range.

        Enough bumps that, were they placed without regard to the margins, some would very
        likely reach into them: each of the 32 does so with a probability of about a quarter.
        """
        margin = 0.2
        solver = make_solver(64, rmin=1e-1, device=self.device)
        source = solver.random_bump_source(nblobs=32, margin=margin)

        x = solver.x
        span = (x[-1] - x[0]).item()
        outside = (x < x[0] + margin * span) | (x > x[-1] - margin * span)
        self.assertTrue(outside.any())
        self.assertEqual(source[outside].abs().max().item(), 0.0)
        self.assertGreater(source[~outside].abs().max().item(), 0.0)

    @parameterized.expand([["bump"], ["ball"]])
    def test_float32_solver_generates_float32_sources(self, source):
        """A solver cast to float32 builds its sources in float32 too, so nothing on its device needs float64."""
        solver = make_solver(64, rmin=1e-1, dtype=torch.float32, device=self.device)
        f = solver.random_bump_source(nblobs=2) if source == "bump" else solver.ball_source(radius=1.0)
        self.assertEqual(f.dtype, torch.float32)

    @parameterized.expand([["bump"], ["ball"]])
    def test_dataset_samples_solve_the_equation(self, source, verbose=False):
        """
        Each training pair is a source times r**2 and its solution, scaled alike.

        The problem is linear, so whatever the random draw and the scale, the target is the
        float64 solve of the input divided by r**2, to the float32 the dataset solves in.
        """
        grid = make_solver(64, rmin=1e-1).grid
        inp, tar = RadialPoissonDataset(grid, source=source, num_examples=1, device=self.device)[0]
        self.assertEqual((inp.dtype, tar.dtype), (torch.float32, torch.float32))

        reference = make_solver(64, rmin=1e-1, device=self.device)
        expected = reference.solve((inp.double() / reference.r.reshape(-1, 1, 1) ** 2).unsqueeze(0)).squeeze(0)
        error = relative_error(tar.double(), expected)
        self.assertLess(error, 1e-5, f"relative error {error:.3e} of the {source} target")

    @parameterized.expand([["bump", 1e-1, 1e3], ["ball", 1e10, 1e11]])
    def test_dataset_normalizes_to_unit_rms(self, source, rmin, rmax):
        """
        The input has unit RMS over the sphere and the radial nodes, as the scale promises.

        On radii of 1e10, r**2 f is about 1e20 and squares to beyond float32, so a scale
        taken by squaring the samples outright would be infinite and the pair zeros.
        """
        grid = SphericalProductGrid(as_grid("legendre-gauss", nlat=16, nlon=32), GeometricRadialGrid(64, rmin, rmax))
        dataset = RadialPoissonDataset(grid, source=source, num_examples=1, device=self.device)
        inp, tar = dataset[0]
        self.assertTrue(torch.isfinite(inp).all() and torch.isfinite(tar).all())

        # in float64, where the normalized samples square safely
        rms = dataset.sphere_mean(inp.double() ** 2).mean().sqrt().item()
        self.assertAlmostEqual(rms, 1.0, delta=1e-5)

    def test_dataset_casts_its_solver_before_moving_it(self):
        """The solver holds no float64 state when it is moved, so the dataset can be built on a device without float64 such as MPS."""
        moved_with = []
        original = RadialPoissonSolver.to

        def to(solver, *args, **kwargs):
            moved_with.append([name for name, b in solver.named_buffers() if b.dtype in (torch.float64, torch.complex128)])
            return original(solver, *args, **kwargs)

        grid = make_solver(64, rmin=1e-1).grid
        with mock.patch.object(RadialPoissonSolver, "to", to):
            RadialPoissonDataset(grid, num_examples=1, device=self.device)
        self.assertEqual(moved_with, [[]])

    @parameterized.expand(
        [
            ["ball", LegendreGaussRadialGrid(16, 0.0, 1.0)],
            ["shell", LegendreGaussRadialGrid(16, 1.0, 2.0)],
            ["geometric_ball", GeometricRadialGrid(16, 1e-2, 1.0, r_outer=1.0)],
        ]
    )
    def test_rejects_bounded_domains(self, _, radial):
        """The Green's operator assumes u -> 0 at infinity, so a radial grid with a finite outer bound is refused."""
        with self.assertRaisesRegex(NotImplementedError, radial.domain):
            RadialPoissonSolver(SphericalProductGrid(as_grid("legendre-gauss", nlat=16, nlon=32), radial))

    def test_requires_a_product_grid(self):
        """The angular grid alone has no radial nodes to solve on."""
        with self.assertRaisesRegex(TypeError, "SphericalProductGrid"):
            RadialPoissonSolver(as_grid("legendre-gauss", nlat=16, nlon=32))


if __name__ == "__main__":
    unittest.main()
