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
from parameterized import parameterized
from testutils import (
    compare_tensors,
    gather_tensor_hw,
    reduce_success,
    set_seed,
    setup_class_from_context,
    setup_module,
    split_tensor_dim,
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


class TestDistributedSpectralConvolution(unittest.TestCase):
    """Compare SpectralConvS2 with DistributedSpectralConvS2 (CPU/CUDA if available)."""

    @classmethod
    def setUpClass(cls):
        setup_class_from_context(cls, _DIST_CTX)

    def _split_helper(self, tensor):
        return split_tensor_hw(tensor, hdim=-2, wdim=-1, hsize=self.grid_size_h, wsize=self.grid_size_w, hrank=self.hrank, wrank=self.wrank)

    def _gather_helper(self, tensor, lat_shapes, lon_shapes):
        tensor_gather = gather_tensor_hw(
            tensor,
            hdim=-2,
            wdim=-1,
            hshapes=lat_shapes,
            wshapes=lon_shapes,
            hsize=self.grid_size_h,
            wsize=self.grid_size_w,
            hrank=self.hrank,
            wrank=self.wrank,
            hgroup=self.h_group,
            wgroup=self.w_group,
        )

        return tensor_gather

    def test_weight_initialization_matches_serial(self):
        """Degree 0 is initialized differently from the others, and only the first polar rank holds it."""
        grid = th.as_grid("legendre-gauss", nlat=32, nlon=64)
        channels = 16
        # with ones for the random numbers the weights are the scale factors themselves
        with mock.patch("torch.randn", side_effect=lambda *args, **kwargs: torch.ones(*args, **kwargs)):
            conv_local = th.SpectralConvS2(grid, grid, in_channels=channels, out_channels=channels)
            conv_dist = thd.DistributedSpectralConvS2(grid, grid, in_channels=channels, out_channels=channels)
        expected = split_tensor_dim(conv_local.weight.detach(), dim=-1, dimsize=self.grid_size_h, dimrank=self.hrank)
        self.assertEqual(conv_dist.weight.shape, expected.shape)
        self.assertTrue(torch.equal(conv_dist.weight.detach(), expected))
        self.assertAlmostEqual(conv_local.weight[0, 0, 0, 0].abs().item(), math.sqrt(2.0 / channels), places=6)

    @parameterized.expand(
        [
            # no bias
            [(64, 128), (64, 128), 4, 8, 1, "equiangular", "equiangular", False, 1e-6, 1e-6],
            [(65, 128), (65, 128), 4, 8, 1, "equiangular", "equiangular", False, 1e-6, 1e-6],
            [(64, 128), (32, 64), 4, 8, 1, "equiangular", "equiangular", False, 1e-6, 1e-6],
            [(65, 128), (33, 64), 4, 8, 1, "equiangular", "equiangular", False, 1e-6, 1e-6],
            [(33, 64), (65, 128), 4, 8, 1, "equiangular", "equiangular", False, 1e-6, 1e-6],
            [(64, 128), (64, 128), 4, 8, 1, "equiangular", "legendre-gauss", False, 1e-6, 1e-6],
            [(65, 128), (65, 128), 4, 8, 1, "equiangular", "legendre-gauss", False, 1e-6, 1e-6],
            # fewer channels than the polar group size (gh #207): B*C is padded inside the
            # internal (i)SHT. batch_size=1 so B*C is just num_chan. Looser tolerance because
            # padding pushes the channel GEMM into the batch>=2 regime the serial path does
            # not use (see test_distributed_sht), a benign ~1e-6 fp32 rounding shift.
            [(64, 128), (64, 128), 1, 1, 1, "equiangular", "equiangular", False, 1e-5, 1e-5],
            [(64, 128), (64, 128), 1, 2, 1, "equiangular", "equiangular", False, 1e-5, 1e-5],
            [(64, 128), (64, 128), 1, 1, 1, "equiangular", "legendre-gauss", False, 1e-5, 1e-5],
            # with bias
            [(64, 128), (64, 128), 4, 8, 1, "equiangular", "equiangular", True, 1e-6, 1e-6],
            [(65, 128), (65, 128), 4, 8, 1, "equiangular", "equiangular", True, 1e-6, 1e-6],
            [(64, 128), (32, 64), 4, 8, 1, "equiangular", "equiangular", True, 1e-6, 1e-6],
            [(65, 128), (33, 64), 4, 8, 1, "equiangular", "equiangular", True, 1e-6, 1e-6],
            [(33, 64), (65, 128), 4, 8, 1, "equiangular", "equiangular", True, 1e-6, 1e-6],
            [(64, 128), (64, 128), 4, 8, 1, "equiangular", "legendre-gauss", True, 1e-6, 1e-6],
            [(65, 128), (65, 128), 4, 8, 1, "equiangular", "legendre-gauss", True, 1e-6, 1e-6],
            # fewer channels than the polar group size (gh #207), with bias
            [(64, 128), (64, 128), 1, 1, 1, "equiangular", "equiangular", True, 1e-5, 1e-5],
            [(64, 128), (64, 128), 1, 2, 1, "equiangular", "equiangular", True, 1e-5, 1e-5],
        ],
        skip_on_empty=True,
    )
    def test_distributed_spectral_conv(self, in_shape, out_shape, batch_size, num_chan, num_groups, grid_in, grid_out, bias, atol, rtol, verbose=True):

        set_seed(333)

        B, C = batch_size, num_chan

        conv_local = th.SpectralConvS2(
            th.as_grid(grid_in, nlat=in_shape[0], nlon=in_shape[1]),
            th.as_grid(grid_out, nlat=out_shape[0], nlon=out_shape[1]),
            in_channels=C,
            out_channels=C,
            num_groups=num_groups,
            bias=bias,
        ).to(self.device)

        conv_dist = thd.DistributedSpectralConvS2(
            th.as_grid(grid_in, nlat=in_shape[0], nlon=in_shape[1]),
            th.as_grid(grid_out, nlat=out_shape[0], nlon=out_shape[1]),
            in_channels=C,
            out_channels=C,
            num_groups=num_groups,
            bias=bias,
        ).to(self.device)

        # copy weights: split local weight along l dimension
        with torch.no_grad():
            weight_split = split_tensor_dim(conv_local.weight.clone(), dim=-1, dimsize=self.grid_size_h, dimrank=self.hrank)
            conv_dist.weight.copy_(weight_split)
            if bias:
                bias_split = split_tensor_hw(conv_local.spectral_bias.clone(), hdim=-2, wdim=-1, hsize=self.grid_size_h, wsize=self.grid_size_w, hrank=self.hrank, wrank=self.wrank)
                conv_dist.spectral_bias.copy_(bias_split)

        # generate input tensor. Reseed first: the distributed weight holds only the local l
        # slice, so with an uneven l split every rank drew a different number of random
        # numbers above, and the ranks would otherwise each generate a different input
        set_seed(333)
        inp_full = torch.randn((B, C, in_shape[0], in_shape[1]), dtype=torch.float32, device=self.device)

        # local forward/backward
        inp_full.requires_grad = True
        out_full = conv_local(inp_full)
        ograd_full = torch.randn_like(out_full)
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        # distributed forward/backward
        inp_local = self._split_helper(inp_full.detach().clone())
        inp_local.requires_grad = True
        out_local = conv_dist(inp_local)
        ograd_local = self._split_helper(ograd_full)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        # Print diagnostics from rank 0 only; assert the all-reduced verdict on every
        # rank so a failure on any rank fails the test consistently (see reduce_success).
        verbose = verbose and self.world_rank == 0

        out_gather_full = self._gather_helper(
            out_local,
            conv_dist.isht.lat_shapes,
            conv_dist.isht.lon_shapes,
        )
        ok = compare_tensors("output", out_gather_full, out_full, atol=atol, rtol=rtol, verbose=verbose, ulps=8)
        self.assertTrue(reduce_success(ok, self.device), "output")

        igrad_gather_full = self._gather_helper(
            igrad_local,
            conv_dist.sht.lat_shapes,
            conv_dist.sht.lon_shapes,
        )
        ok = compare_tensors("gradients", igrad_gather_full, igrad_full, atol=atol, rtol=rtol, verbose=verbose, ulps=8)
        self.assertTrue(reduce_success(ok, self.device), "gradients")

    @parameterized.expand(
        [
            # in_shape, out_shape, grid_in, grid_out, lmax, mmax, lmmax, bias
            [(64, 128), (64, 128), "legendre-gauss", "legendre-gauss", 40, None, None, False],  # triangular, only lmax
            [(64, 128), (64, 128), "legendre-gauss", "legendre-gauss", 48, 20, None, True],  # rectangular
            [(64, 128), (64, 128), "legendre-gauss", "legendre-gauss", 48, 20, 20, True],  # rhomboidal
            [(65, 128), (33, 64), "equiangular", "equiangular", 17, 9, 5, True],  # downsampling, odd bounds
            [(33, 64), (64, 128), "equiangular", "legendre-gauss", None, None, 7, False],  # default bounds + lmmax
        ],
        skip_on_empty=True,
    )
    def test_distributed_spectral_conv_truncation(self, in_shape, out_shape, grid_in, grid_out, lmax, mmax, lmmax, bias, verbose=True):
        """Serial and distributed layers built with the same (lmax, mmax, lmmax) agree."""

        set_seed(333)

        B, C, num_groups = 2, 4, 2
        gin = th.as_grid(grid_in, nlat=in_shape[0], nlon=in_shape[1])
        gout = th.as_grid(grid_out, nlat=out_shape[0], nlon=out_shape[1])
        trunc = dict(lmax=lmax, mmax=mmax, lmmax=lmmax)

        conv_local = th.SpectralConvS2(gin, gout, in_channels=C, out_channels=C, num_groups=num_groups, bias=bias, **trunc).to(self.device)
        conv_dist = thd.DistributedSpectralConvS2(gin, gout, in_channels=C, out_channels=C, num_groups=num_groups, bias=bias, **trunc).to(self.device)

        self.assertEqual((conv_dist.lmax, conv_dist.mmax, conv_dist.lmmax), (conv_local.lmax, conv_local.mmax, conv_local.lmmax))
        self.assertEqual(conv_dist.sht.grid_out, conv_local.sht.grid_out)

        # copy weights: split along l; a nonzero bias so its support is exercised too
        with torch.no_grad():
            if bias:
                conv_local.spectral_bias.copy_(torch.randn_like(conv_local.spectral_bias))
            weight_split = split_tensor_dim(conv_local.weight.clone(), dim=-1, dimsize=self.grid_size_h, dimrank=self.hrank)
            conv_dist.weight.copy_(weight_split)
            if bias:
                bias_split = split_tensor_hw(conv_local.spectral_bias.clone(), hdim=-2, wdim=-1, hsize=self.grid_size_h, wsize=self.grid_size_w, hrank=self.hrank, wrank=self.wrank)
                conv_dist.spectral_bias.copy_(bias_split)

        # reseed: the ranks drew different amounts of random numbers above, see test_distributed_spectral_conv
        set_seed(333)
        inp_full = torch.randn((B, C, in_shape[0], in_shape[1]), dtype=torch.float32, device=self.device)

        inp_full.requires_grad = True
        out_full = conv_local(inp_full)
        ograd_full = torch.randn_like(out_full)
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        inp_local = self._split_helper(inp_full.detach().clone())
        inp_local.requires_grad = True
        out_local = conv_dist(inp_local)
        out_local.backward(self._split_helper(ograd_full))
        igrad_local = inp_local.grad.clone()

        verbose = verbose and self.world_rank == 0

        out_gather_full = self._gather_helper(out_local, conv_dist.isht.lat_shapes, conv_dist.isht.lon_shapes)
        ok = compare_tensors("output", out_gather_full, out_full, atol=1e-6, rtol=1e-6, verbose=verbose, ulps=8)
        self.assertTrue(reduce_success(ok, self.device), "output")

        igrad_gather_full = self._gather_helper(igrad_local, conv_dist.sht.lat_shapes, conv_dist.sht.lon_shapes)
        ok = compare_tensors("input gradients", igrad_gather_full, igrad_full, atol=1e-6, rtol=1e-6, verbose=verbose, ulps=8)
        self.assertTrue(reduce_success(ok, self.device), "input gradients")
