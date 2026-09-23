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
from unittest.mock import patch

import numpy as np
import torch
from parameterized import parameterized
from testutils import compare_tensors, set_seed

from torch_harmonics.examples.shallow_water_equations import ShallowWaterSolver


def _reference_timestep(solver, initial, nsteps):
    # Integrate the interpolation polynomial through the available RHS samples.
    # The moment equations derive the weights independently of the implementation.
    state = initial.clone()
    history = []
    for _ in range(nsteps):
        history.insert(0, solver.dudtspec(state))
        history = history[:3]
        order = len(history)
        nodes = -np.arange(order, dtype=np.float64)
        moments = np.vander(nodes, N=order, increasing=True).T
        weights = np.linalg.solve(moments, 1.0 / np.arange(1, order + 1))
        increment = sum(float(weight) * rhs for weight, rhs in zip(weights, history))
        state = state + solver.dt * increment
        state = torch.cat((state[:1], state[1:] * solver.hyperdiff), dim=0)
    return state


class TestShallowWaterStartup(unittest.TestCase):
    @parameterized.expand([(steps, dtype, damping) for steps in (0, 1, 2, 3, 6) for dtype in (torch.float32, torch.float64) for damping in (1.0, 0.8)])
    def test_linear_tendency(self, steps, dtype, damping, verbose=False):
        solver = ShallowWaterSolver(8, 16, 0.1, lmax=3, mmax=3).to(dtype=dtype)
        solver.hyperdiff.fill_(damping)
        ctype = torch.complex64 if dtype == torch.float32 else torch.complex128
        initial = torch.arange(1, 28, dtype=dtype).reshape(3, 3, 3).to(ctype) / 10
        original = initial.clone()
        # Use a manufactured diagonal ODE to isolate time-history bookkeeping.
        rates = torch.tensor([1.0, -0.5, 0.25], dtype=dtype).reshape(3, 1, 1)
        with patch.object(solver, "dudtspec", side_effect=lambda state: rates * state):
            expected = _reference_timestep(solver, initial, steps)
            actual = solver.timestep(initial, steps)
        self.assertTrue(compare_tensors("linear evolution", expected, actual, atol=1e-6, rtol=1e-6, verbose=verbose))
        self.assertTrue(compare_tensors("input unchanged", original, initial, atol=0, rtol=0, verbose=verbose))

    @parameterized.expand([(grid, steps) for grid in ("equiangular", "legendre-gauss", "lobatto") for steps in (2, 5)])
    def test_nonlinear_shallow_water_tendency(self, grid, steps, verbose=False):
        solver = ShallowWaterSolver(12, 24, 120.0, lmax=4, mmax=4, grid=grid)
        set_seed(333)
        initial = solver.random_initial_condition(mach=0.2)
        expected = _reference_timestep(solver, initial, steps)
        actual = solver.timestep(initial, steps)
        self.assertTrue(compare_tensors("shallow-water evolution", expected, actual, atol=1e-12, rtol=1e-10, verbose=verbose))

    def test_second_step_has_adams_bashforth_two_weights(self, verbose=False):
        solver = ShallowWaterSolver(8, 16, 0.1, lmax=3, mmax=3)
        solver.hyperdiff.fill_(1)
        initial = torch.ones(3, 3, 3, dtype=torch.complex128)
        with patch.object(solver, "dudtspec", side_effect=lambda state: state):
            actual = solver.timestep(initial, 2)
        # Euler gives y1=1+h; AB2 then gives y2=1+2h+3h^2/2.
        expected = torch.full_like(initial, 1.215)
        self.assertTrue(compare_tensors("Euler then AB2", expected, actual, atol=1e-14, rtol=1e-14, verbose=verbose))


if __name__ == "__main__":
    unittest.main()
