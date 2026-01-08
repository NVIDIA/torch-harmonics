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

import os
import unittest
from parameterized import parameterized

import torch
import torch_harmonics as th
import torch_harmonics.distributed as thd

from testutils import (
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

class TestDistributedResampling(unittest.TestCase):
    """Test the distributed resampling module (CPU/CUDA if available)."""

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

    def _gather_helper_fwd(self, tensor, resampling_dist):

        tensor_gather = gather_tensor_hw(
            tensor, 
            hdim=-2, 
            wdim=-1, 
            hshapes=resampling_dist.lat_out_shapes, 
            wshapes=resampling_dist.lon_out_shapes, 
            hsize=self.grid_size_h, 
            wsize=self.grid_size_w, 
            hrank=self.hrank, 
            wrank=self.wrank, 
            hgroup=self.h_group, 
            wgroup=self.w_group
        )

        return tensor_gather

    def _gather_helper_bwd(self, tensor, resampling_dist):

        tensor_gather = gather_tensor_hw(
            tensor, 
            hdim=-2, 
            wdim=-1, 
            hshapes=resampling_dist.lat_in_shapes, 
            wshapes=resampling_dist.lon_in_shapes, 
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
            [64, 128, 128, 256, 32, 8, "equiangular", "equiangular", "bilinear", 1e-6, 1e-7],
            [128, 256, 64, 128, 32, 8, "equiangular", "equiangular", "bilinear", 1e-6, 1e-7],
            [64, 128, 128, 256, 32, 8, "equiangular", "equiangular", "bilinear-spherical", 1e-6, 1e-7],
            [128, 256, 64, 128, 32, 8, "equiangular", "equiangular", "bilinear-spherical", 1e-6, 1e-7],
            [129, 256, 65, 128, 32, 8, "equiangular", "equiangular", "bilinear", 1e-6, 1e-7],
            [65, 128, 129, 256, 32, 8, "equiangular", "equiangular", "bilinear", 1e-6, 1e-7],
            [129, 256, 65, 128, 32, 8, "equiangular", "legendre-gauss", "bilinear", 1e-6, 1e-7],
            [65, 128, 129, 256, 32, 8, "legendre-gauss", "equiangular", "bilinear", 1e-6, 1e-7],
        ], skip_on_empty=True
    )
    def test_distributed_resampling(
            self, nlat_in, nlon_in, nlat_out, nlon_out, batch_size, num_chan, grid_in, grid_out, mode, atol, rtol, verbose=False
    ):
        
        B, C, H, W = batch_size, num_chan, nlat_in, nlon_in

        res_args = dict(
            nlat_in=nlat_in,
            nlon_in=nlon_in,
            nlat_out=nlat_out,
            nlon_out=nlon_out,
            grid_in=grid_in,
            grid_out=grid_out,
            mode=mode,
        )

        # set up handlesD
        res_local = th.ResampleS2(**res_args).to(self.device)
        res_dist = thd.DistributedResampleS2(**res_args).to(self.device)

        # create tensors
        inp_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device)

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
        inp_local = self._split_helper(inp_full)
        inp_local.requires_grad = True
        out_local = res_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full)
        out_local = res_dist(inp_local)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        # evaluate FWD pass
        out_gather_full = self._gather_helper_fwd(out_local, res_dist)
        self.assertTrue(compare_tensors("output", out_full, out_gather_full, atol=atol, rtol=rtol, verbose=verbose))

        # evaluate BWD pass
        igrad_gather_full = self._gather_helper_bwd(igrad_local, res_dist)
        self.assertTrue(compare_tensors("gradients", igrad_full, igrad_gather_full, atol=atol, rtol=rtol, verbose=verbose))

if __name__ == "__main__":
    unittest.main()
