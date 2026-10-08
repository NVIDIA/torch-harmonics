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
Smoke tests for the example models and solvers in :mod:`torch_harmonics.examples`.

They pin the descriptor interface -- every model takes the grid of its data and the
grid(s) it works on internally, rather than a shape, a grid name and a scale factor --
and that each model maps a field on ``grid`` to a field on ``grid``. They are not
accuracy tests; the grids are as small as the layers allow.
"""

import unittest

import torch
from parameterized import parameterized

from torch_harmonics import as_grid
from torch_harmonics.examples import PdeDataset, ShallowWaterSolver, SphereSolver
from torch_harmonics.examples.models import (
    LocalSphericalNeuralOperator,
    SphericalFourierNeuralOperator,
    SphericalSegformer,
    SphericalTransformer,
    SphericalUNet,
)

_GRID = as_grid("equiangular", nlat=17, nlon=32)
_LATENT = as_grid("legendre-gauss", nlat=9, nlon=16)
_PYRAMID = [as_grid("equiangular", nlat=8, nlon=16), as_grid("equiangular", nlat=4, nlon=8)]


def _models():
    return [
        ("sfno", lambda: SphericalFourierNeuralOperator(_GRID, _LATENT, in_chans=2, out_chans=3, embed_dim=8, num_layers=2, pos_embed="spectral")),
        ("lsno", lambda: LocalSphericalNeuralOperator(_GRID, _LATENT, in_chans=2, out_chans=3, embed_dim=8, num_layers=2)),
        ("s2transformer", lambda: SphericalTransformer(_GRID, _LATENT, in_chans=2, out_chans=3, embed_dim=8, num_layers=1, attention_mode="global")),
        ("s2ntransformer", lambda: SphericalTransformer(_GRID, _LATENT, in_chans=2, out_chans=3, embed_dim=8, num_layers=1, attention_mode="neighborhood")),
        ("s2unet", lambda: SphericalUNet(_GRID, _PYRAMID, in_chans=2, out_chans=3, embed_dims=[4, 8], depths=[1, 1])),
        ("s2segformer", lambda: SphericalSegformer(_GRID, _PYRAMID, in_chans=2, out_chans=3, embed_dims=[4, 8], heads=[1, 2], depths=[1, 1])),
    ]


class TestExampleModels(unittest.TestCase):

    @parameterized.expand(_models())
    def test_maps_a_field_on_the_grid_to_a_field_on_the_grid(self, name, build):
        torch.manual_seed(333)
        model = build().eval()
        with torch.no_grad():
            out = model(torch.randn(2, 2, *_GRID.shape))
        self.assertEqual(tuple(out.shape), (2, 3, *_GRID.shape))
        self.assertTrue(torch.isfinite(out).all())

    def test_the_old_shape_and_scale_factor_arguments_are_rejected(self):
        with self.assertRaises(TypeError):
            SphericalFourierNeuralOperator(img_size=(17, 32), grid="equiangular", scale_factor=2)
        # a grid name where a descriptor belongs is refused by the grid guard
        with self.assertRaises(TypeError):
            SphericalFourierNeuralOperator("equiangular", _LATENT)

    def test_a_pyramid_needs_one_grid_per_stage(self):
        with self.assertRaises(ValueError):
            SphericalUNet(_GRID, _PYRAMID[:1], embed_dims=[4, 8], depths=[1, 1])


class TestExampleSolvers(unittest.TestCase):

    def test_shallow_water_solver_steps_on_its_grid(self):
        grid = as_grid("equiangular", nlat=16, nlon=32)
        solver = ShallowWaterSolver(grid, dt=60.0, lmax=8, mmax=8)
        self.assertEqual((solver.nlat, solver.nlon), grid.shape)
        uspec = solver.timestep(solver.random_initial_condition(mach=0.1), 2)
        self.assertTrue(torch.isfinite(solver.spec2grid(uspec)).all())

    def test_sphere_solver_takes_a_descriptor(self):
        grid = as_grid("legendre-gauss", nlat=16, nlon=32)
        solver = SphereSolver(grid, dt=1e-3, lmax=8, mmax=8)
        self.assertEqual(tuple(solver.lats.shape[-1:]), (16,))

    def test_pde_dataset_generates_samples_on_its_grid(self):
        grid = as_grid("equiangular", nlat=16, nlon=32)
        dataset = PdeDataset(dt=60.0, nsteps=1, grid=grid, num_examples=1, normalize=False)
        inp, tar = dataset[0]
        self.assertEqual(tuple(inp.shape[-2:]), grid.shape)
        self.assertEqual(tuple(tar.shape[-2:]), grid.shape)


if __name__ == "__main__":
    unittest.main()
