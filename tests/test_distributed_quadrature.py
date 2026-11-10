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

import torch
from parameterized import parameterized
from testutils import compare_tensors, gather_tensor_hw, range_atol, reduce_success, set_seed, setup_class_from_context, setup_module, split_tensor_hw, teardown_module

import torch_harmonics as th
import torch_harmonics.distributed as thd

# shared state
_DIST_CTX = {}

# Error budget in ULPs of the integral of |f| (see testutils.range_atol): the distributed
# integral reassociates the float32 grid sum across ranks.
# Measured: 0.06 ULPs of int |f|.
_QUAD_ULPS = 8


def setUpModule():
    setup_module(_DIST_CTX)


def tearDownModule():
    teardown_module(_DIST_CTX)


class TestDistributedQuadrature(unittest.TestCase):
    """Compare QuadratureS2 with DistributedQuadratureS2 (CPU/CUDA if available)."""

    @classmethod
    def setUpClass(cls):
        setup_class_from_context(cls, _DIST_CTX)

    def _split_helper(self, tensor):
        return split_tensor_hw(tensor, hdim=-2, wdim=-1, hsize=self.grid_size_h, wsize=self.grid_size_w, hrank=self.hrank, wrank=self.wrank)

    def _gather_helper(self, tensor, quadrature_dist):

        tensor_gather = gather_tensor_hw(
            tensor,
            hdim=-2,
            wdim=-1,
            hshapes=quadrature_dist.lat_shapes,
            wshapes=quadrature_dist.lon_shapes,
            hsize=self.grid_size_h,
            wsize=self.grid_size_w,
            hrank=self.hrank,
            wrank=self.wrank,
            hgroup=self.h_group,
            wgroup=self.w_group,
        )

        return tensor_gather

    @parameterized.expand(
        [
            [64, 128, 4, 3, "equiangular", False],
            [65, 128, 2, 2, "equiangular", True],
            [64, 128, 4, 3, "legendre-gauss", False],
            [65, 128, 2, 2, "lobatto", False],
        ]
    )
    def test_distributed_quadrature(self, nlat, nlon, batch_size, num_chan, grid, normalize, verbose=False):

        set_seed(333)

        B, C, H, W = batch_size, num_chan, nlat, nlon

        quad_local = th.QuadratureS2(th.as_grid(grid, nlat=H, nlon=W), normalize=normalize).to(self.device)
        quad_dist = thd.DistributedQuadratureS2(th.as_grid(grid, nlat=H, nlon=W), normalize=normalize).to(self.device)

        # create tensors
        inp_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device)

        # local quadrature
        inp_full.requires_grad = True
        out_full = quad_local(inp_full)

        with torch.no_grad():
            ograd_full = torch.randn_like(out_full)

        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        # distributed quadrature
        inp_local = self._split_helper(inp_full.detach().clone())
        inp_local.requires_grad = True
        out_local = quad_dist(inp_local)
        ograd_local = ograd_full.clone()

        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        # only the spatial axes are reduced: a single field integrates to a scalar
        self.assertEqual(quad_dist(inp_local[0, 0].detach()).shape, torch.Size([]))

        # Print diagnostics from rank 0 only; assert the all-reduced verdict on every
        # rank so a failure on any rank fails the test consistently (see reduce_success).
        verbose = verbose and self.world_rank == 0

        # evaluate FWD pass
        # the integral of random data cancels, so the floor comes from the summed terms, int |f|
        with torch.no_grad():
            atol = range_atol(quad_local(inp_full.detach().abs()), 0.0, _QUAD_ULPS)
        ok = compare_tensors("output", out_local, out_full, atol=atol, rtol=0.0, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "output")

        # evaluate BWD pass
        igrad_gather_full = self._gather_helper(igrad_local, quad_dist)
        # the gradient is the upstream gradient times the weights, elementwise: exact
        ok = compare_tensors("gradients", igrad_gather_full, igrad_full, atol=0.0, rtol=0.0, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "gradients")

    @parameterized.expand(
        [
            [8, 4, 3, False],
            [8, 2, 2, True],
            [16, 2, 3, False],
        ]
    )
    def test_distributed_quadrature_healpix(self, nside, batch_size, num_chan, normalize, verbose=False):
        """A HEALPix field is a flat pixel axis split into bands of rings along the polar group."""
        if self.grid_size_w > 1:
            self.skipTest("HEALPix decomposes along latitude only")

        set_seed(333)

        B, C = batch_size, num_chan
        grid = th.HealpixGrid(nside=nside)
        quad_local = th.QuadratureS2(grid, normalize=normalize).to(self.device)
        quad_dist = thd.DistributedQuadratureS2(grid, normalize=normalize).to(self.device)

        inp_full = torch.randn((B, C, grid.npoints), dtype=torch.float32, device=self.device)

        # serial quadrature
        inp_full.requires_grad = True
        out_full = quad_local(inp_full)
        with torch.no_grad():
            ograd_full = torch.randn_like(out_full)
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        # distributed quadrature: the shard knows how many pixels every rank holds
        pixel_shapes = quad_dist.pixel_shapes
        self.assertEqual(sum(pixel_shapes), grid.npoints)
        inp_local = torch.split(inp_full.detach().clone(), pixel_shapes, dim=-1)[self.hrank].contiguous()
        self.assertEqual(inp_local.shape[-1], quad_dist.shard.npoints)
        inp_local.requires_grad = True
        out_local = quad_dist(inp_local)
        out_local.backward(ograd_full.clone())
        igrad_local = inp_local.grad.clone()

        # only the spatial axis is reduced: a single field integrates to a scalar
        self.assertEqual(quad_dist(inp_local[0, 0].detach()).shape, torch.Size([]))

        verbose = verbose and self.world_rank == 0

        with torch.no_grad():
            atol = range_atol(quad_local(inp_full.detach().abs()), 0.0, _QUAD_ULPS)
        ok = compare_tensors("output", out_local, out_full, atol=atol, rtol=0.0, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "output")

        igrad_gather_full = gather_tensor_hw(
            igrad_local,
            hdim=-1,
            wdim=-1,
            hshapes=pixel_shapes,
            wshapes=[],
            hsize=self.grid_size_h,
            wsize=1,
            hrank=self.hrank,
            wrank=self.wrank,
            hgroup=self.h_group,
            wgroup=self.w_group,
        )
        ok = compare_tensors("gradients", igrad_gather_full, igrad_full, atol=0.0, rtol=0.0, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "gradients")

    def test_a_healpix_grid_refuses_an_azimuthal_decomposition(self):
        """With azimuth ranks the rings would be cut mid-ring; the layer says so at construction."""
        if self.grid_size_w == 1:
            self.skipTest("needs an azimuth group of more than one rank")
        with self.assertRaises(NotImplementedError):
            thd.DistributedQuadratureS2(th.HealpixGrid(nside=8))
