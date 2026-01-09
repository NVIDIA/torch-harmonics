# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
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

import os
import unittest
from parameterized import parameterized

import torch
import torch_harmonics as th
import torch_harmonics.distributed as thd

from testutils import (
    set_seed,
    setup_module, 
    teardown_module, 
    setup_class_from_context,
    split_tensor_hw, 
    gather_tensor_hw,
    compare_tensors,
)

# shared state
_DIST_CTX = {}

def setUpModule():
    setup_module(_DIST_CTX)

def tearDownModule():
    teardown_module(_DIST_CTX)

class TestDistributedSphericalHarmonicTransform(unittest.TestCase):
    """Test the distributed spherical harmonic transform module (CPU/CUDA if available)."""

    @classmethod
    def setUpClass(cls):
        setup_class_from_context(cls, _DIST_CTX)

    def _split_helper(self, tensor):
        return split_tensor_hw(
            tensor, 
            hdim=-2,
            wdim=-1, 
            hsize=self.grid_size_h, 
            wsize=self.grid_size_w, 
            hrank=self.hrank, 
            wrank=self.wrank
        )

    def _gather_helper_fwd(self, tensor, transform_dist):
        tensor_gather = gather_tensor_hw(
            tensor, 
            hdim=-2, 
            wdim=-1, 
            hshapes=transform_dist.l_shapes, 
            wshapes=transform_dist.m_shapes, 
            hsize=self.grid_size_h, 
            wsize=self.grid_size_w, 
            hrank=self.hrank, 
            wrank=self.wrank, 
            hgroup=self.h_group, 
            wgroup=self.w_group
        )

        return tensor_gather

    def _gather_helper_bwd(self, tensor, transform_dist):

        tensor_gather = gather_tensor_hw(
            tensor, 
            hdim=-2, 
            wdim=-1, 
            hshapes=transform_dist.lat_shapes, 
            wshapes=transform_dist.lon_shapes, 
            hsize=self.grid_size_h, 
            wsize=self.grid_size_w, 
            hrank=self.hrank, 
            wrank=self.wrank, 
            hgroup=self.h_group, 
            wgroup=self.w_group
        )

        return tensor_gather

    @parameterized.expand(
        [
            [64, 128, 32, 8, "equiangular", False, 1e-7,1e-9],
            [64, 128, 32, 8, "legendre-gauss", False, 1e-7, 1e-9],
            [64, 128, 32, 8, "equiangular", False, 1e-7, 1e-9],
            [64, 128, 32, 8, "legendre-gauss", False, 1e-7, 1e-9],
            [64, 128, 32, 8, "equiangular", False, 1e-7, 1e-9],
            [64, 128, 32, 8, "legendre-gauss", False, 1e-7, 1e-9],
            [65, 128, 1, 10, "equiangular", False, 1e-6, 1e-6],
            [65, 128, 1, 10, "legendre-gauss", False, 1e-6, 1e-6],
            [4, 8, 1, 10, "equiangular", False, 1e-6, 1e-6],
            [64, 128, 32, 8, "equiangular", True, 1e-7, 1e-9],
            [64, 128, 32, 8, "legendre-gauss", True, 1e-7, 1e-9],
            [64, 128, 32, 8, "equiangular", True, 1e-7, 1e-9],
            [64, 128, 32, 8, "legendre-gauss", True, 1e-7, 1e-9],
            [64, 128, 32, 8, "equiangular", True, 1e-7, 1e-9],
            [64, 128, 32, 8, "legendre-gauss", True, 1e-7, 1e-9],
            [64, 128, 1, 10, "equiangular", True, 1e-6, 1e-6],
            [65, 128, 1, 10, "legendre-gauss", True, 1e-6, 1e-6],
        ], skip_on_empty=True
    )
    def test_distributed_sht(self, nlat, nlon, batch_size, num_chan, grid, vector, atol, rtol, verbose=False):

        set_seed(333)

        B, C, H, W = batch_size, num_chan, nlat, nlon

        # set up handles
        if vector:
            forward_transform_local = th.RealVectorSHT(nlat=H, nlon=W, grid=grid).to(self.device)
            forward_transform_dist = thd.DistributedRealVectorSHT(nlat=H, nlon=W, grid=grid).to(self.device)
        else:
            forward_transform_local = th.RealSHT(nlat=H, nlon=W, grid=grid).to(self.device)
            forward_transform_dist = thd.DistributedRealSHT(nlat=H, nlon=W, grid=grid).to(self.device)

        # create tensors
        if vector:
            inp_full = torch.randn((B, C, 2, H, W), dtype=torch.float32, device=self.device)
        else:
            inp_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device)

        # local transform
        # FWD pass
        inp_full.requires_grad = True
        out_full = forward_transform_local(inp_full)

        # create grad for backward
        with torch.no_grad():
            # create full grad
            ograd_full = torch.randn_like(out_full)

        # BWD pass
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        # distributed transform
        # FWD pass
        inp_local = self._split_helper(inp_full.detach().clone())
        inp_local.requires_grad = True
        out_local = forward_transform_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full)
        out_local = forward_transform_dist(inp_local)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        # evaluate FWD pass
        out_gather_full = self._gather_helper_fwd(out_local, forward_transform_dist)
        self.assertTrue(compare_tensors("output", out_full, out_gather_full, atol=atol, rtol=rtol, verbose=verbose))

        # evaluate BWD pass
        igrad_gather_full = self._gather_helper_bwd(igrad_local, forward_transform_dist)
        self.assertTrue(compare_tensors("gradients", igrad_full, igrad_gather_full, atol=atol, rtol=rtol, verbose=verbose))


    @parameterized.expand(
        [
            [64, 128, 32, 8, "equiangular", False, 1e-7, 1e-9],
            [64, 128, 32, 8, "legendre-gauss", False, 1e-7, 1e-9],
            [64, 128, 32, 8, "equiangular", False, 1e-7, 1e-9],
            [64, 128, 32, 8, "legendre-gauss", False, 1e-7, 1e-9],
            [64, 128, 32, 8, "equiangular", False, 1e-7, 1e-9],
            [64, 128, 32, 8, "legendre-gauss", False, 1e-7, 1e-9],
            [65, 128, 1, 10, "equiangular", False, 1e-6, 1e-6],
            [65, 128, 1, 10, "legendre-gauss", False, 1e-6, 1e-6],
            [64, 128, 32, 8, "equiangular", True, 1e-7, 1e-9],
            [64, 128, 32, 8, "legendre-gauss", True, 1e-7, 1e-9],
            [64, 128, 32, 8, "equiangular", True, 1e-7, 1e-9],
            [64, 128, 32, 8, "legendre-gauss", True, 1e-7, 1e-9],
            [64, 128, 32, 8, "equiangular", True, 1e-7, 1e-9],
            [64, 128, 32, 8, "legendre-gauss", True, 1e-7, 1e-9],
            [65, 128, 1, 10, "equiangular", True, 1e-6, 1e-6],
            [65, 128, 1, 10, "legendre-gauss", True, 1e-6, 1e-6],
        ], skip_on_empty=True
    )
    def test_distributed_isht(self, nlat, nlon, batch_size, num_chan, grid, vector, atol, rtol, verbose=True):

        set_seed(333)

        B, C, H, W = batch_size, num_chan, nlat, nlon

        if vector:
            forward_transform_local = th.RealVectorSHT(nlat=H, nlon=W, grid=grid).to(self.device)
            backward_transform_local = th.InverseRealVectorSHT(nlat=H, nlon=W, grid=grid).to(self.device)
            backward_transform_dist = thd.DistributedInverseRealVectorSHT(nlat=H, nlon=W, grid=grid).to(self.device)
        else:
            forward_transform_local = th.RealSHT(nlat=H, nlon=W, grid=grid).to(self.device)
            backward_transform_local = th.InverseRealSHT(nlat=H, nlon=W, grid=grid).to(self.device)
            backward_transform_dist = thd.DistributedInverseRealSHT(nlat=H, nlon=W, grid=grid).to(self.device)

        # create tensors
        if vector:
            dummy_full = torch.randn((B, C, 2, H, W), dtype=torch.float32, device=self.device)
        else:
            dummy_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device)
        inp_full = forward_transform_local(dummy_full)

        #############################################################
        # local transform
        #############################################################
        # FWD pass
        inp_full.requires_grad = True
        out_full = backward_transform_local(inp_full)

        # create grad for backward
        with torch.no_grad():
            # create full grad
            ograd_full = torch.randn_like(out_full)

        # BWD pass
        out_full.backward(ograd_full)

        # repeat once due to known irfft bug
        inp_full.grad = None
        out_full = backward_transform_local(inp_full)
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        # distributed transform
        # FWD pass
        inp_local = self._split_helper(inp_full.detach().clone())
        inp_local.requires_grad = True
        out_local = backward_transform_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full)
        out_local = backward_transform_dist(inp_local)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        # evaluate FWD pass
        out_gather_full = self._gather_helper_bwd(out_local, backward_transform_dist)
        self.assertTrue(compare_tensors("output", out_full, out_gather_full, atol=atol, rtol=rtol, verbose=verbose))

        # evaluate BWD pass
        igrad_gather_full = self._gather_helper_fwd(igrad_local, backward_transform_dist)
        self.assertTrue(compare_tensors("gradients", igrad_full, igrad_gather_full, atol=atol, rtol=rtol, verbose=verbose))


if __name__ == "__main__":
    unittest.main()
