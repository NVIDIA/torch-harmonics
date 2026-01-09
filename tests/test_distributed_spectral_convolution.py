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
    split_tensor_dim,
    gather_tensor_hw,
    compare_tensors,
)

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
        return split_tensor_hw(
            tensor, 
            hdim=-2,
            wdim=-1, 
            hsize=self.grid_size_h, 
            wsize=self.grid_size_w, 
            hrank=self.hrank, 
            wrank=self.wrank
        )

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
            wgroup=self.w_group
        )

        return tensor_gather

    @parameterized.expand(
        [
            # no bias
            [(64, 128), (64, 128), 4, 8, 1, "equiangular", "equiangular", False, 1e-6, 1e-6],
            [(65, 128), (65, 128), 4, 8, 1, "equiangular", "equiangular", False, 1e-6, 1e-6],
            [(64, 128), (32,  64), 4, 8, 1, "equiangular", "equiangular", False, 1e-6, 1e-6],
            [(65, 128), (33,  64), 4, 8, 1, "equiangular", "equiangular", False, 1e-6, 1e-6],
            [(33,  64), (65, 128), 4, 8, 1, "equiangular", "equiangular", False, 1e-6, 1e-6],
            [(64, 128), (64, 128), 4, 8, 1, "equiangular", "legendre-gauss", False, 1e-6, 1e-6],
            [(65, 128), (65, 128), 4, 8, 1, "equiangular", "legendre-gauss", False, 1e-6, 1e-6],
            # with bias
            [(64, 128), (64, 128), 4, 8, 1, "equiangular", "equiangular", True, 1e-6, 1e-6],
            [(65, 128), (65, 128), 4, 8, 1, "equiangular", "equiangular", True, 1e-6, 1e-6],
            [(64, 128), (32,  64), 4, 8, 1, "equiangular", "equiangular", True, 1e-6, 1e-6],
            [(65, 128), (33,  64), 4, 8, 1, "equiangular", "equiangular", True, 1e-6, 1e-6],
            [(33,  64), (65, 128), 4, 8, 1, "equiangular", "equiangular", True, 1e-6, 1e-6],
            [(64, 128), (64, 128), 4, 8, 1, "equiangular", "legendre-gauss", True, 1e-6, 1e-6],
            [(65, 128), (65, 128), 4, 8, 1, "equiangular", "legendre-gauss", True, 1e-6, 1e-6],
        ], skip_on_empty=True
    )
    def test_distributed_spectral_conv(self, in_shape, out_shape, batch_size, num_chan, num_groups, grid_in, grid_out, bias, atol, rtol, verbose=True):

        set_seed(333)

        B, C = batch_size, num_chan

        conv_local = th.SpectralConvS2(
            in_shape=in_shape,
            out_shape=out_shape,
            in_channels=C,
            out_channels=C,
            num_groups=num_groups,
            grid_in=grid_in,
            grid_out=grid_out,
            bias=bias,
        ).to(self.device)

        conv_dist = thd.DistributedSpectralConvS2(
            in_shape=in_shape,
            out_shape=out_shape,
            in_channels=C,
            out_channels=C,
            num_groups=num_groups,
            grid_in=grid_in,
            grid_out=grid_out,
            bias=bias,
        ).to(self.device)

        # copy weights: split local weight along l dimension
        with torch.no_grad():
            weight_split = split_tensor_dim(conv_local.weight.clone(), dim=-1, dimsize=self.grid_size_h, dimrank=self.hrank)
            conv_dist.weight.copy_(weight_split)
            if bias:
                bias_split = split_tensor_hw(conv_local.spectral_bias.clone(), hdim=-2, wdim=-1, hsize=self.grid_size_h, wsize=self.grid_size_w, hrank=self.hrank, wrank=self.wrank)
                conv_dist.spectral_bias.copy_(bias_split)

        # generate input tensor
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

        out_gather_full = self._gather_helper(
            out_local,
            conv_dist.isht.lat_shapes,
            conv_dist.isht.lon_shapes,
        )
        self.assertTrue(compare_tensors("output", out_full, out_gather_full, atol=atol, rtol=rtol, verbose=verbose))

        igrad_gather_full = self._gather_helper(
            igrad_local,
            conv_dist.sht.lat_shapes,
            conv_dist.sht.lon_shapes,
        )
        self.assertTrue(compare_tensors("gradients", igrad_full, igrad_gather_full, atol=atol, rtol=rtol, verbose=verbose))

if __name__ == "__main__":
    unittest.main()

