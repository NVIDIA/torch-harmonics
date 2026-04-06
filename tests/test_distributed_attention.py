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


class TestDistributedNeighborhoodAttention(unittest.TestCase):
    """
    Compare serial NeighborhoodAttentionS2 against DistributedNeighborhoodAttentionS2.

    CPU-only runs are skipped: distributed attention requires CUDA (NCCL + custom kernels).
    """

    @classmethod
    def setUpClass(cls):
        setup_class_from_context(cls, _DIST_CTX)
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Distributed neighborhood attention requires CUDA")

    def _split_helper(self, tensor):
        return split_tensor_hw(
            tensor,
            hdim=-2,
            wdim=-1,
            hsize=self.grid_size_h,
            wsize=self.grid_size_w,
            hrank=self.hrank,
            wrank=self.wrank,
        )

    def _gather_helper_fwd(self, tensor, attn_dist):
        return gather_tensor_hw(
            tensor,
            hdim=-2,
            wdim=-1,
            hshapes=attn_dist.lat_out_shapes,
            wshapes=attn_dist.lon_out_shapes,
            hsize=self.grid_size_h,
            wsize=self.grid_size_w,
            hrank=self.hrank,
            wrank=self.wrank,
            hgroup=self.h_group,
            wgroup=self.w_group,
        )

    def _gather_helper_bwd(self, tensor, attn_dist, use_out_shapes=False):
        hshapes = attn_dist.lat_out_shapes if use_out_shapes else attn_dist.lat_in_shapes
        wshapes = attn_dist.lon_out_shapes if use_out_shapes else attn_dist.lon_in_shapes
        return gather_tensor_hw(
            tensor,
            hdim=-2,
            wdim=-1,
            hshapes=hshapes,
            wshapes=wshapes,
            hsize=self.grid_size_h,
            wsize=self.grid_size_w,
            hrank=self.hrank,
            wrank=self.wrank,
            hgroup=self.h_group,
            wgroup=self.w_group,
        )

    @parameterized.expand(
        [
            # nlat_in, nlon_in, nlat_out, nlon_out, batch_size, in_channels, num_heads, k_channels, out_channels, grid_in, grid_out, atol, rtol
            [64, 128, 64, 128, 2, 16, 1, None, None, "equiangular", "equiangular", 1e-5, 1e-4],
            [64, 128, 64, 128, 2, 16, 2, None, None, "equiangular", "equiangular", 1e-5, 1e-4],
            [64, 128, 64, 128, 2, 16, 1,   8,    8, "equiangular", "equiangular", 1e-5, 1e-4],
            [64, 128, 32,  64, 2, 16, 1, None, None, "equiangular", "equiangular", 1e-5, 1e-4],
            [65, 128, 65, 128, 2, 16, 1, None, None, "equiangular", "equiangular", 1e-5, 1e-4],
        ],
        skip_on_empty=True,
    )
    def test_distributed_neighborhood_attention(
        self,
        nlat_in, nlon_in, nlat_out, nlon_out,
        batch_size, in_channels, num_heads, k_channels, out_channels,
        grid_in, grid_out,
        atol, rtol,
        verbose=True,
    ):
        set_seed(333)

        B, C, Hi, Wi, Ho, Wo = batch_size, in_channels, nlat_in, nlon_in, nlat_out, nlon_out

        attn_args = dict(
            in_channels=C,
            in_shape=(nlat_in, nlon_in),
            out_shape=(nlat_out, nlon_out),
            grid_in=grid_in,
            grid_out=grid_out,
            num_heads=num_heads,
            bias=True,
            k_channels=k_channels,
            out_channels=out_channels,
        )

        # build serial and distributed modules with identical weights
        attn_serial = th.NeighborhoodAttentionS2(**attn_args).to(self.device)
        attn_dist   = thd.DistributedNeighborhoodAttentionS2(**attn_args).to(self.device)

        with torch.no_grad():
            attn_dist.k_weights.copy_(attn_serial.k_weights)
            attn_dist.v_weights.copy_(attn_serial.v_weights)
            attn_dist.q_weights.copy_(attn_serial.q_weights)
            attn_dist.proj_weights.copy_(attn_serial.proj_weights)
            if attn_args["bias"]:
                attn_dist.k_bias.copy_(attn_serial.k_bias)
                attn_dist.v_bias.copy_(attn_serial.v_bias)
                attn_dist.q_bias.copy_(attn_serial.q_bias)
                attn_dist.proj_bias.copy_(attn_serial.proj_bias)

        # Helper: create inputs
        inp_full = {
            "k": torch.randn(B, C, Hi, Wi, requires_grad=True, device=self.device, dtype=torch.float32),
            "v": torch.randn(B, C, Hi, Wi, requires_grad=True, device=self.device, dtype=torch.float32),
            "q": torch.randn(B, C, Ho, Wo, requires_grad=True, device=self.device, dtype=torch.float32),
        }

        # ---- serial forward ----
        out_full = attn_serial(inp_full["q"], inp_full["k"], inp_full["v"])
        
        torch.cuda.synchronize()

        # ---- serial backward ----
        with torch.no_grad():
            ograd_full = torch.randn_like(out_full)
        out_full.backward(ograd_full)

        igrad_full = {}
        for inp in ["q", "k", "v"]:
            igrad_full[inp] = inp_full[inp].grad.clone()

        torch.cuda.synchronize()

        # ---- distributed forward ----
        inp_local = {}
        for inp in ["q", "k", "v"]:
            inp_local[inp] = self._split_helper(inp_full[inp].detach())
            inp_local[inp].requires_grad_(True)
        out_local = attn_dist(inp_local["q"], inp_local["k"], inp_local["v"])

        torch.cuda.synchronize()

        # ---- distributed backward ----
        ograd_local = self._split_helper(ograd_full)
        out_local.backward(ograd_local)

        igrad_local = {}
        for inp in ["q", "k", "v"]:
            igrad_local[inp] = inp_local[inp].grad.clone()

        torch.cuda.synchronize()

        # ---- compare forward ----
        out_gather = self._gather_helper_fwd(out_local, attn_dist)
        self.assertTrue(compare_tensors("forward output", out_full, out_gather, atol=atol, rtol=rtol, verbose=verbose))

        # ---- compare backward ----
        for inp in ["q", "k", "v"]:
            use_out = (inp == "q")
            igrad_gather = self._gather_helper_bwd(igrad_local[inp], attn_dist, use_out_shapes=use_out)
            self.assertTrue(compare_tensors(f"input gradient {inp}", igrad_full[inp], igrad_gather, atol=atol, rtol=rtol, verbose=verbose))


if __name__ == "__main__":
    unittest.main()
