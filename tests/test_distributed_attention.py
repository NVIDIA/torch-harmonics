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
import torch.nn.functional as F
import torch.distributed as dist
import torch_harmonics as th
import torch_harmonics.distributed as thd


class TestDistributedAttention(unittest.TestCase):
    """Test the distributed attention module."""

    @classmethod
    def setUpClass(cls):
        # set up distributed
        cls.world_rank = int(os.getenv("WORLD_RANK", 0))
        cls.grid_size_h = int(os.getenv("GRID_H", 1))
        cls.grid_size_w = int(os.getenv("GRID_W", 1))
        port = int(os.getenv("MASTER_PORT", "29501"))
        master_address = os.getenv("MASTER_ADDR", "localhost")
        cls.world_size = cls.grid_size_h * cls.grid_size_w

        if torch.cuda.is_available():
            if cls.world_rank == 0:
                print("Running test on GPU")
            local_rank = cls.world_rank % torch.cuda.device_count()
            cls.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(local_rank)
            torch.cuda.manual_seed(333)
            proc_backend = "nccl"
        else:
            if cls.world_rank == 0:
                print("Running test on CPU")
            cls.device = torch.device("cpu")
            proc_backend = "gloo"
        torch.manual_seed(333)

        dist.init_process_group(backend=proc_backend, init_method=f"tcp://{master_address}:{port}", rank=cls.world_rank, world_size=cls.world_size)

        cls.wrank = cls.world_rank % cls.grid_size_w
        cls.hrank = cls.world_rank // cls.grid_size_w

        # now set up the comm groups:
        # set default
        cls.w_group = None
        cls.h_group = None

        # do the init
        wgroups = []
        for w in range(0, cls.world_size, cls.grid_size_w):
            start = w
            end = w + cls.grid_size_w
            wgroups.append(list(range(start, end)))

        if cls.world_rank == 0:
            print("w-groups:", wgroups)
        for grp in wgroups:
            if len(grp) == 1:
                continue
            tmp_group = dist.new_group(ranks=grp)
            if cls.world_rank in grp:
                cls.w_group = tmp_group

        # transpose:
        hgroups = [sorted(list(i)) for i in zip(*wgroups)]

        if cls.world_rank == 0:
            print("h-groups:", hgroups)
        for grp in hgroups:
            if len(grp) == 1:
                continue
            tmp_group = dist.new_group(ranks=grp)
            if cls.world_rank in grp:
                cls.h_group = tmp_group

        if cls.world_rank == 0:
            print(f"Running distributed tests on grid H x W = {cls.grid_size_h} x {cls.grid_size_w}")

        # initializing distributed
        thd.init(cls.h_group, cls.w_group)

    @classmethod
    def tearDownClass(cls):
        thd.finalize()
        dist.destroy_process_group(None)

    def _split_helper(self, tensor):

        with torch.no_grad():
            # split in W
            tensor_list_local = thd.split_tensor_along_dim(tensor, dim=-1, num_chunks=self.grid_size_w)
            tensor_local = tensor_list_local[self.wrank]

            # split in H
            tensor_list_local = thd.split_tensor_along_dim(tensor_local, dim=-2, num_chunks=self.grid_size_h)
            tensor_local = tensor_list_local[self.hrank]

        return tensor_local

    def _gather_helper(self, tensor, B, C, attention_dist, is_output=True):

        # we need the shapes
        if is_output:
            lat_shapes = attention_dist.lat_out_shapes
            lon_shapes = attention_dist.lon_out_shapes
            nlat = attention_dist.nlat_out
            nlon = attention_dist.nlon_out
        else:
            lat_shapes = attention_dist.lat_in_shapes
            lon_shapes = attention_dist.lon_in_shapes
            nlat = attention_dist.nlat_in
            nlon = attention_dist.nlon_in

        # gather in W
        tensor = tensor.contiguous()
        if self.grid_size_w > 1:
            gather_shapes = [(B, C, lat_shapes[self.hrank], w) for w in lon_shapes]
            olist = [torch.empty(shape, dtype=tensor.dtype, device=tensor.device) for shape in gather_shapes]
            olist[self.wrank] = tensor
            dist.all_gather(olist, tensor, group=self.w_group)
            tensor_gather = torch.cat(olist, dim=-1)
        else:
            tensor_gather = tensor

        # gather in H
        tensor_gather = tensor_gather.contiguous()
        if self.grid_size_h > 1:
            gather_shapes = [(B, C, h, nlon) for h in lat_shapes]
            olist = [torch.empty(shape, dtype=tensor_gather.dtype, device=tensor_gather.device) for shape in gather_shapes]
            olist[self.hrank] = tensor_gather
            dist.all_gather(olist, tensor_gather, group=self.h_group)
            tensor_gather = torch.cat(olist, dim=-2)

        return tensor_gather

    @parameterized.expand(
        [
            [16, 32, 16, 32, 2, 16, 16, 4, "equiangular", "equiangular", False, 1e-5],
            [16, 32, 16, 32, 2,  8, 16, 4, "equiangular", "equiangular", False, 1e-5],
        ]
    )
    def test_distributed_attention_s2(
        self, nlat_in, nlon_in, nlat_out, nlon_out, batch_size, in_channels, out_channels, num_heads, grid_in, grid_out, bias, tol
    ):
        
        B, Cin, Cout, H, W = batch_size, in_channels, out_channels, nlat_in, nlon_in

        attention_args = dict(
            in_channels=Cin,
            out_channels=Cout,
            num_heads=num_heads,
            in_shape=(nlat_in, nlon_in),
            out_shape=(nlat_out, nlon_out),
            grid_in=grid_in,
            grid_out=grid_out,
            bias=bias,
        )

        # set up handles
        attn_local = th.AttentionS2(**attention_args).to(self.device)
        attn_dist = thd.DistributedAttentionS2(**attention_args).to(self.device)

        # copy the weights from the local attention into the dist attention
        with torch.no_grad():
            attn_dist.q_weights.copy_(attn_local.q_weights)
            attn_dist.k_weights.copy_(attn_local.k_weights)
            attn_dist.v_weights.copy_(attn_local.v_weights)
            attn_dist.proj_weights.copy_(attn_local.proj_weights)
            if bias:
                attn_dist.q_bias.copy_(attn_local.q_bias)
                attn_dist.k_bias.copy_(attn_local.k_bias)
                attn_dist.v_bias.copy_(attn_local.v_bias)
                attn_dist.proj_bias.copy_(attn_local.proj_bias)

        # create tensors
        query_full = torch.randn((B, Cin, nlat_out, nlon_out), dtype=torch.float32, device=self.device)
        key_full = torch.randn((B, Cin, H, W), dtype=torch.float32, device=self.device)
        value_full = torch.randn((B, Cin, H, W), dtype=torch.float32, device=self.device)

        # local attention
        # FWD pass
        query_full.requires_grad = True
        key_full.requires_grad = True
        value_full.requires_grad = True
        out_full = attn_local(query_full, key_full, value_full)

        # create grad for backward
        with torch.no_grad():
            # create full grad
            ograd_full = torch.randn_like(out_full)

        # BWD pass
        out_full.backward(ograd_full)
        query_grad_full = query_full.grad.clone()
        key_grad_full = key_full.grad.clone()
        value_grad_full = value_full.grad.clone()

        # distributed attention
        # FWD pass
        query_local = self._split_helper(query_full)
        key_local = self._split_helper(key_full)
        value_local = self._split_helper(value_full)
        query_local.requires_grad = True
        key_local.requires_grad = True
        value_local.requires_grad = True
        out_local = attn_dist(query_local, key_local, value_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full)
        out_local.backward(ograd_local)
        query_grad_local = query_local.grad.clone()
        key_grad_local = key_local.grad.clone()
        value_grad_local = value_local.grad.clone()

        # evaluate FWD pass
        with torch.no_grad():
            out_gather_full = self._gather_helper(out_local, B, Cout, attn_dist, is_output=True)
            err = torch.mean(torch.norm(out_full - out_gather_full, p="fro", dim=(-1, -2)) / torch.norm(out_full, p="fro", dim=(-1, -2)))
            if self.world_rank == 0:
                print(f"final relative error of output: {err.item()}")
        self.assertTrue(err.item() <= tol)

        # evaluate BWD pass - query gradient
        with torch.no_grad():
            query_grad_gather_full = self._gather_helper(query_grad_local, B, Cin, attn_dist, is_output=True)
            err = torch.mean(torch.norm(query_grad_full - query_grad_gather_full, p="fro", dim=(-1, -2)) / torch.norm(query_grad_full, p="fro", dim=(-1, -2)))
            if self.world_rank == 0:
                print(f"final relative error of query gradients: {err.item()}")
        self.assertTrue(err.item() <= tol)

        # evaluate BWD pass - key gradient
        with torch.no_grad():
            key_grad_gather_full = self._gather_helper(key_grad_local, B, Cin, attn_dist, is_output=False)
            err = torch.mean(torch.norm(key_grad_full - key_grad_gather_full, p="fro", dim=(-1, -2)) / torch.norm(key_grad_full, p="fro", dim=(-1, -2)))
            if self.world_rank == 0:
                print(f"final relative error of key gradients: {err.item()}")
        self.assertTrue(err.item() <= tol)

        # evaluate BWD pass - value gradient
        with torch.no_grad():
            value_grad_gather_full = self._gather_helper(value_grad_local, B, Cin, attn_dist, is_output=False)
            err = torch.mean(torch.norm(value_grad_full - value_grad_gather_full, p="fro", dim=(-1, -2)) / torch.norm(value_grad_full, p="fro", dim=(-1, -2)))
            if self.world_rank == 0:
                print(f"final relative error of value gradients: {err.item()}")
        self.assertTrue(err.item() <= tol)


if __name__ == "__main__":
    unittest.main()
