# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
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
from testutils import (
    compare_tensors,
    gather_tensor_hw,
    reduce_success,
    set_seed,
    setup_class_from_context,
    setup_module,
    split_tensor_hw,
    teardown_module,
)

import torch_harmonics as th
import torch_harmonics.distributed as thd

# shared state
_DIST_CTX = {}


def setUpModule():
    setup_module(_DIST_CTX)


def tearDownModule():
    teardown_module(_DIST_CTX)


def _make_grid(name, shape):
    """A grid descriptor from a row's grid name and shape: ``(nside,)`` for HEALPix, ``(nlat, nlon)`` otherwise."""
    if name == "healpix":
        grid = th.HealpixGrid(nside=shape[0])
    else:
        grid = th.as_grid(name, nlat=shape[0], nlon=shape[1])
    return grid


class TestDistributedResampling(unittest.TestCase):
    """Test the distributed resampling module (CPU/CUDA if available)."""

    @classmethod
    def setUpClass(cls):
        setup_class_from_context(cls, _DIST_CTX)

    def _skip_unless_decomposable(self, *grids):
        """HEALPix decomposes along latitude only; a layer on it refuses an azimuth group, see test_healpix_refuses_an_azimuthal_decomposition."""
        if self.grid_size_w > 1 and any(isinstance(g, th.HealpixGrid) for g in grids):
            self.skipTest("HEALPix decomposes along latitude only")

    def _split_helper(self, tensor, shard):
        """This rank's piece of a global field, as ``shard`` describes it."""
        if isinstance(shard.grid, th.HealpixGrid):
            tensor_local = torch.split(tensor, list(shard.pixel_shapes), dim=-1)[self.hrank].contiguous()
        else:
            tensor_local = split_tensor_hw(tensor, hdim=-2, wdim=-1, hsize=self.grid_size_h, wsize=self.grid_size_w, hrank=self.hrank, wrank=self.wrank)
        return tensor_local

    def _gather_helper(self, tensor, shard):
        """The global field from every rank's piece, as ``shard`` describes them."""
        if isinstance(shard.grid, th.HealpixGrid):
            hdim, hshapes, wshapes, wsize = -1, list(shard.pixel_shapes), [], 1
        else:
            hdim, hshapes, wshapes, wsize = -2, list(shard.lat_shapes), list(shard.lon_shapes), self.grid_size_w
        return gather_tensor_hw(
            tensor,
            hdim=hdim,
            wdim=-1,
            hshapes=hshapes,
            wshapes=wshapes,
            hsize=self.grid_size_h,
            wsize=wsize,
            hrank=self.hrank,
            wrank=self.wrank,
            hgroup=self.h_group,
            wgroup=self.w_group,
        )

    @parameterized.expand(
        [[mode, nlon_out] for mode in ("bilinear", "bilinear-spherical") for nlon_out in (1, 7)],
        skip_on_empty=True,
    )
    def test_single_longitude_resampling(self, mode, nlon_out, verbose=False):
        """A single longitude matches serial resampling and keeps finite gradients."""
        if self.grid_size_w != 1:
            self.skipTest("a single longitude cannot be partitioned across multiple azimuth ranks")
        if self.grid_size_h > 16:
            self.skipTest("test grid has fewer input latitudes than polar ranks")

        set_seed(334)
        res_args = dict(
            grid_in=th.as_grid("equiangular", nlat=16, nlon=1),
            grid_out=th.as_grid("equiangular", nlat=24, nlon=nlon_out),
            mode=mode,
        )
        res_local = th.ResampleS2(**res_args).to(self.device)
        res_dist = thd.DistributedResampleS2(**res_args).to(self.device)

        inp_full = torch.randn((2, 8, 16, 1), dtype=torch.float32, device=self.device, requires_grad=True)
        out_full = res_local(inp_full)
        ograd_full = torch.randn_like(out_full)
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        inp_local = self._split_helper(inp_full.detach().clone(), res_dist.shard_in)
        inp_local.requires_grad = True
        out_local = res_dist(inp_local)
        ograd_local = self._split_helper(ograd_full, res_dist.shard_out)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        verbose = verbose and self.world_rank == 0
        out_gather_full = self._gather_helper(out_local, res_dist.shard_out)
        ok = compare_tensors("single-longitude output", out_gather_full, out_full, atol=1e-6, rtol=1e-5, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "single-longitude output")

        igrad_gather_full = self._gather_helper(igrad_local, res_dist.shard_in)
        ok = compare_tensors("single-longitude gradients", igrad_gather_full, igrad_full, atol=1e-6, rtol=1e-5, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "single-longitude gradients")
        self.assertTrue(reduce_success(bool(torch.isfinite(igrad_local).all()), self.device), "finite gradients")
        self.assertTrue(
            reduce_success(bool(torch.isfinite(res_dist.lon_weights).all()), self.device),
            "finite longitude weights",
        )

    @parameterized.expand(
        [
            [(64, 128), (128, 256), 32, 8, "equiangular", "equiangular", "bilinear", 1e-6, 1e-7],
            [(128, 256), (64, 128), 32, 8, "equiangular", "equiangular", "bilinear", 1e-6, 1e-7],
            [(64, 128), (128, 256), 32, 8, "equiangular", "equiangular", "bilinear-spherical", 1e-6, 1e-7],
            [(128, 256), (64, 128), 32, 8, "equiangular", "equiangular", "bilinear-spherical", 1e-6, 1e-7],
            [(129, 256), (65, 128), 32, 8, "equiangular", "equiangular", "bilinear", 1e-6, 1e-7],
            [(65, 128), (129, 256), 32, 8, "equiangular", "equiangular", "bilinear", 1e-6, 1e-7],
            [(129, 256), (65, 128), 32, 8, "equiangular", "legendre-gauss", "bilinear", 1e-6, 1e-7],
            [(65, 128), (129, 256), 32, 8, "legendre-gauss", "equiangular", "bilinear", 1e-6, 1e-7],
            # HEALPix, on one side or both; shapes are (nside,). Interpolated point by point
            [(8,), (16,), 4, 8, "healpix", "healpix", "bilinear", 1e-6, 1e-7],
            [(16,), (8,), 4, 8, "healpix", "healpix", "bilinear-spherical", 1e-6, 1e-7],
            [(16,), (33, 64), 4, 8, "healpix", "equiangular", "bilinear", 1e-6, 1e-7],  # reaches the poles
            [(16,), (33, 64), 4, 8, "healpix", "equiangular", "bilinear-spherical", 1e-6, 1e-7],
            [(32, 64), (16,), 4, 8, "legendre-gauss", "healpix", "bilinear", 1e-6, 1e-7],
            [(33, 64), (8,), 2, 4, "equiangular", "healpix", "bilinear-spherical", 1e-6, 1e-7],
        ],
        skip_on_empty=True,
    )
    def test_distributed_resampling(self, in_shape, out_shape, batch_size, num_chan, grid_in, grid_out, mode, atol, rtol, verbose=False):

        set_seed(333)

        B, C = batch_size, num_chan
        gin, gout = _make_grid(grid_in, in_shape), _make_grid(grid_out, out_shape)
        self._skip_unless_decomposable(gin, gout)

        res_args = dict(grid_in=gin, grid_out=gout, mode=mode)

        # set up handlesD
        res_local = th.ResampleS2(**res_args).to(self.device)
        res_dist = thd.DistributedResampleS2(**res_args).to(self.device)

        # create tensors
        inp_full = torch.randn((B, C, *gin.shape), dtype=torch.float32, device=self.device)

        # local conv
        # FWD pass
        inp_full.requires_grad = True
        out_full = res_local(inp_full)

        # create grad for backward
        with torch.no_grad():
            # create full grad
            ograd_full = torch.randn_like(out_full)

        # BWD pass
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        # distributed conv
        # FWD pass
        inp_local = self._split_helper(inp_full.detach().clone(), res_dist.shard_in)
        inp_local.requires_grad = True
        out_local = res_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full, res_dist.shard_out)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        # Print diagnostics from rank 0 only; assert the all-reduced verdict on every
        # rank so a failure on any rank fails the test consistently (see reduce_success).
        verbose = verbose and self.world_rank == 0

        # evaluate FWD pass
        out_gather_full = self._gather_helper(out_local, res_dist.shard_out)
        ok = compare_tensors("output", out_gather_full, out_full, atol=atol, rtol=rtol, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "output")

        # evaluate BWD pass
        igrad_gather_full = self._gather_helper(igrad_local, res_dist.shard_in)
        ok = compare_tensors("gradients", igrad_gather_full, igrad_full, atol=atol, rtol=rtol, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "gradients")

    @parameterized.expand(
        [
            ["healpix", "healpix"],
            ["healpix", "equiangular"],
            ["equiangular", "healpix"],
        ]
    )
    def test_healpix_refuses_an_azimuthal_decomposition(self, grid_in, grid_out):
        """A HEALPix grid on either side cannot be cut along the azimuth; the layer says so at construction."""
        if self.grid_size_w == 1:
            self.skipTest("needs an azimuth group of more than one rank")
        gin = _make_grid(grid_in, (8,) if grid_in == "healpix" else (17, 32))
        gout = _make_grid(grid_out, (8,) if grid_out == "healpix" else (17, 32))
        with self.assertRaises(NotImplementedError):
            thd.DistributedResampleS2(gin, gout)
