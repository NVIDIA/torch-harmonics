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
import torch.distributed as dist
from parameterized import parameterized
from testutils import compare_tensors, disable_tf32, set_seed, setup_class_from_context, setup_module, teardown_module

import torch_harmonics.distributed as thd
from torch_harmonics import LegendreGaussRadialGrid, SphericalProductGrid, as_grid

# shared state
_DIST_CTX = {}


def setUpModule():
    setup_module(_DIST_CTX)


def tearDownModule():
    teardown_module(_DIST_CTX)


_C = 8
_C_OUT = 12
_RADIAL = LegendreGaussRadialGrid(nr=3, r_inner=1.0, r_outer=2.0)
# splits unevenly over 2 and 4 ranks, as in test_distributed_field_contract
_GRID = as_grid("legendre-gauss", nlat=33, nlon=66)
_GRID_OUT = as_grid("equiangular", nlat=35, nlon=66)


def _close(msg, actual, expected):
    # the batched and the per-shell contractions sum in different orders, so an entry that
    # cancels to well below the field's scale differs by roundoff of that scale, not of its own.
    # Diagnostics from rank 0 only, as in test_distributed_convolution
    verbose = not dist.is_initialized() or dist.get_rank() == 0
    return compare_tensors(msg, actual, expected, atol=1e-5 * expected.abs().max().item(), rtol=1e-5, verbose=verbose)


def _loss(y):
    return (torch.view_as_real(y) if y.is_complex() else y).square().sum()


# name, layer from (grid_in, grid_out), radial axis of the input and the output, this
# rank's input shape without the radial axis, input dtype, whether it needs CUDA
_LAYERS = [
    ["sht", lambda gi, go: thd.DistributedRealSHT(gi), 1, 1, lambda l: (_C, *l.shard.shape), torch.float32, False],
    ["inverse_sht", lambda gi, go: thd.DistributedInverseRealSHT(gi), 1, 1, lambda l: (_C, l.lmax_local, l.mmax_local), torch.complex64, False],
    ["vector_sht", lambda gi, go: thd.DistributedRealVectorSHT(gi), 2, 2, lambda l: (_C, 2, *l.shard.shape), torch.float32, False],
    ["inverse_vector_sht", lambda gi, go: thd.DistributedInverseRealVectorSHT(gi), 2, 2, lambda l: (_C, 2, l.lmax_local, l.mmax_local), torch.complex64, False],
    ["quadrature", lambda gi, go: thd.DistributedQuadratureS2(gi), 1, 1, lambda l: (2, *l.shard.shape), torch.float32, False],
    ["resample", lambda gi, go: thd.DistributedResampleS2(gi, go), 1, 1, lambda l: (_C, *l.shard_in.shape), torch.float32, False],
    ["spectral_conv", lambda gi, go: thd.DistributedSpectralConvS2(gi, go, _C, _C_OUT, bias=True), 2, 2, lambda l: (2, _C, *l.sht.shard.shape), torch.float32, False],
    ["disco", lambda gi, go: thd.DistributedDiscreteContinuousConvS2(gi, go, _C, _C_OUT, kernel_shape=3), 2, 2, lambda l: (2, _C, *l.shard_in.shape), torch.float32, False],
    [
        "disco_transpose",
        lambda gi, go: thd.DistributedDiscreteContinuousConvTransposeS2(gi, go, _C, _C_OUT, kernel_shape=3),
        2,
        2,
        lambda l: (2, _C, *l.shard_in.shape),
        torch.float32,
        False,
    ],
    # the ring kernels exist for CUDA only
    [
        "neighborhood_attention",
        lambda gi, go: thd.DistributedNeighborhoodAttentionS2(gi, go, _C, out_channels=_C_OUT, num_heads=2),
        2,
        2,
        lambda l: (2, _C, *l.shard_in.shape),
        torch.float32,
        True,
    ],
]


def _grids(name):
    return (_GRID, _GRID_OUT) if name == "resample" else (_GRID, _GRID)


class TestDistributedLayersOnProductGrids(unittest.TestCase):
    """A distributed layer on a product grid computes, on every shell, what it computes on the angular grid alone."""

    @classmethod
    def setUpClass(cls):
        setup_class_from_context(cls, _DIST_CTX)
        # the distributed DISCO weight contraction is a cuBLAS einsum; with TF32 the batched
        # and the per-shell call can land on different kernels, see test_distributed_convolution
        disable_tf32()

    def setUp(self):
        # start from a drained device and matched ranks, as test_distributed_convolution does:
        # module-owned receive buffers of a previous test may still be written by NCCL
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        set_seed(333)

    def tearDown(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @parameterized.expand(_LAYERS)
    def test_acts_on_each_shell_alike(self, name, build, rdim_in, rdim_out, input_shape, dtype, needs_cuda):
        if needs_cuda and self.device.type != "cuda":
            self.skipTest(f"{name} runs on CUDA only")
        angular_in, angular_out = _grids(name)
        layer = build(SphericalProductGrid(angular_in, _RADIAL), SphericalProductGrid(angular_out, _RADIAL)).to(self.device)
        reference = build(angular_in, angular_out).to(self.device)

        # randomize every parameter, so a bias or a weight that the product path misapplies
        # cannot hide behind its initialization; the reference shares them
        with torch.no_grad():
            for p in layer.parameters():
                p.add_(0.1 * torch.randn_like(p))
        reference.load_state_dict(layer.state_dict())

        # this rank's block; every rank runs the same sequence of calls, so the collectives match
        shape = list(input_shape(reference))
        shape.insert(rdim_in, _RADIAL.nr)
        x = torch.randn(shape, device=self.device, dtype=dtype, requires_grad=True)

        out = layer(x)
        _loss(out).backward()

        for k in range(_RADIAL.nr):
            xk = x.detach().select(rdim_in, k).clone().requires_grad_(True)
            expected = reference(xk)
            _loss(expected).backward()
            self.assertTrue(_close(f"{name} output on shell {k}", out.detach().select(rdim_out, k), expected.detach()))
            self.assertTrue(_close(f"{name} input gradient on shell {k}", x.grad.select(rdim_in, k), xk.grad))

    def test_volume_quadrature_sums_the_shells(self):
        grid = SphericalProductGrid(_GRID, _RADIAL)
        quad = thd.DistributedQuadratureVolume(grid).to(self.device)
        shells = thd.DistributedQuadratureS2(grid).to(self.device)
        x = torch.randn(2, _RADIAL.nr, *quad.shard.shape, device=self.device)
        expected = (shells(x) * _RADIAL.volume_weights.to(device=self.device, dtype=torch.float32)).sum(-1)
        self.assertTrue(_close("volume integral", quad(x), expected))

    def test_volume_quadrature_of_one_is_the_volume(self):
        grid = SphericalProductGrid(_GRID, _RADIAL)
        quad = thd.DistributedQuadratureVolume(grid).to(self.device)
        x = torch.ones(_RADIAL.nr, *quad.shard.shape, device=self.device)
        self.assertAlmostEqual(quad(x).item(), 4.0 / 3.0 * math.pi * 7.0, places=3)
        self.assertAlmostEqual(thd.DistributedQuadratureVolume(grid, normalize=True).to(self.device)(x).item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
