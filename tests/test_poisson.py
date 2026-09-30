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

from torch_harmonics.examples.poisson_equation import RadialPoissonSolver


def gaussian_shell(r, r0=1.0, width=0.2):
    """A smooth radial profile and the source that produces it.

    Manufactured solution: pick u, apply the radial Laplacian analytically, and feed the
    result back in. For a function of r alone,

        lap u = (1/r^2) d/dr (r^2 du/dr)

    so u = exp(-(r - r0)^2 / 2 s^2) gives the expression below. The profile is negligible
    at both ends of the radial grid, which keeps the truncation of the domain out of the
    comparison.
    """
    u = torch.exp(-((r - r0) ** 2) / (2 * width**2))
    f = (-2 * (r - r0) / (r * width**2) - 1 / width**2 + (r - r0) ** 2 / width**4) * u
    return u, f


def solve_radial(solver, f_radial):
    """Broadcast a radial profile over the sphere, solve, and read one column back."""
    field = f_radial.reshape(-1, 1, 1).expand(solver.nr, solver.nlat, solver.nlon).contiguous()
    return solver.solve(field.unsqueeze(0)).squeeze(0)[:, 0, 0]


class TestRadialPoissonSolver(unittest.TestCase):
    """Correctness of the radial Green's operator against closed-form solutions."""

    @parameterized.expand([[256, 4.8e-3], [512, 1.2e-3], [1024, 3.0e-4]])
    def test_manufactured_solution(self, nr, tol, verbose=False):
        """The solve reproduces a known smooth solution to within the quadrature error."""
        solver = RadialPoissonSolver(nlat=16, nlon=32, nr=nr, rmin=1e-2, rmax=1e3, grid="legendre-gauss").double()
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
            solver = RadialPoissonSolver(nlat=16, nlon=32, nr=nr, rmin=1e-2, rmax=1e3, grid="legendre-gauss").double()
            expected, source = gaussian_shell(solver.r)
            actual = solve_radial(solver, source)
            window = (solver.r > 0.2) & (solver.r < 5.0)
            errors.append(((actual[window] - expected[window]).abs().max() / expected[window].abs().max()).item())

        orders = [math.log2(a / b) for a, b in zip(errors, errors[1:])]
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
        solver = RadialPoissonSolver(nlat=16, nlon=32, nr=512, rmin=1e-2, rmax=1e3, grid=grid).double()
        actual = solver.solve(solver.ball_source(radius=radius).unsqueeze(0)).squeeze(0)[:, 0, 0]

        r = solver.r
        expected = torch.where(r <= radius, r**2 / 6 - radius**2 / 2, -(radius**3) / (3 * r))
        window = (r > 2e-2) & (r < 1e2)
        self.assertTrue(compare_tensors(f"ball source on {grid}", actual[window], expected[window], atol=2e-2, rtol=1e-2, verbose=verbose))

    def test_exterior_domain_vanishes_on_the_inner_sphere(self, verbose=False):
        """The image term imposes u = 0 at the inner radius."""
        inner_radius = 1.0
        solver = RadialPoissonSolver(nlat=16, nlon=32, nr=512, rmin=1e-3, rmax=1e3, grid="legendre-gauss", domain="exterior", inner_radius=inner_radius).double()

        _, source = gaussian_shell(solver.r, r0=2.0, width=0.5)
        field = source.reshape(-1, 1, 1).expand(solver.nr, 16, 32).contiguous()
        u = solver.solve(field.unsqueeze(0))

        # the first node sits just outside the inner sphere, so the solution there is
        # small rather than exactly zero; measure it against the scale of the solution
        self.assertLess((u[:, 0].abs().max() / u.abs().max()).item(), 1e-2)

    @parameterized.expand([[torch.float32], [torch.float64]])
    def test_solve_honours_the_input_precision(self, dtype, verbose=False):
        """The Green's kernel is assembled in float64 whatever the caller uses.

        PoissonDataset hands back float32 samples, and einsum rejects a float32 operand
        against a float64 kernel outright, so the solve has to meet the input where it is.
        """
        solver = RadialPoissonSolver(nlat=16, nlon=32, nr=64, grid="legendre-gauss")
        out = solver.solve(torch.randn(1, 64, 16, 32, dtype=dtype))
        self.assertEqual(out.dtype, dtype)
        self.assertTrue(torch.isfinite(out).all())

    @parameterized.expand([[torch.float32], [torch.float64]])
    def test_converted_solver_honours_the_input_precision(self, dtype, verbose=False):
        """Converting the module must not pin the solve to one dtype either."""
        solver = RadialPoissonSolver(nlat=16, nlon=32, nr=64, grid="legendre-gauss").float()
        out = solver.solve(torch.randn(1, 64, 16, 32, dtype=dtype))
        self.assertEqual(out.dtype, dtype)
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
