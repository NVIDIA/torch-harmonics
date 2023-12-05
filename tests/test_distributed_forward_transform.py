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
#import sys
#sys.path.append("..")
#sys.path.append(".")

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch_harmonics as harmonics
import torch_harmonics.distributed as thd


class TestDistributedSphericalHarmonicTransform(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # set up distributed
        cls.world_rank = int(os.getenv('WORLD_RANK', 0))
        cls.grid_size_h = int(os.getenv('GRID_H', 1))
        cls.grid_size_w = int(os.getenv('GRID_W', 1))
        port = int(os.getenv('MASTER_PORT', '29501'))
        master_address = os.getenv('MASTER_ADDR', 'localhost')
        cls.world_size = cls.grid_size_h * cls.grid_size_w

        if torch.cuda.is_available():
            if cls.world_rank == 0:
                print("Running test on GPU")
            local_rank = cls.world_rank % torch.cuda.device_count()
            cls.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.manual_seed(333)
            proc_backend = 'nccl'
        else:
            if cls.world_rank == 0:
                print("Running test on CPU")
            cls.device = torch.device('cpu')
            proc_backend = 'gloo'
        torch.manual_seed(333)

        dist.init_process_group(backend = proc_backend,
                                init_method = f"tcp://{master_address}:{port}",
                                rank = cls.world_rank,
				world_size = cls.world_size)
            
        cls.wrank = cls.world_rank % cls.grid_size_w
        cls.hrank = cls.world_rank // cls.grid_size_w

        # now set up the comm groups:
        #set default
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

        # set seed
        torch.manual_seed(333)
        

        if cls.world_rank == 0:
            print(f"Running distributed tests on grid H x W = {cls.grid_size_h} x {cls.grid_size_w}")

        # initializing sht
        thd.init(cls.h_group, cls.w_group)

        
    def _split_helper(self, tensor):
        with torch.no_grad():
            # split in W
            tensor_list_local = thd.split_tensor_along_dim(tensor, dim=-1, num_chunks=self.grid_size_w)
            tensor_local = tensor_list_local[self.wrank]

            # split in H
            tensor_list_local = thd.split_tensor_along_dim(tensor_local, dim=-2, num_chunks=self.grid_size_h)
            tensor_local = tensor_list_local[self.hrank]

        return tensor_local
        
        
    def _gather_helper_fwd(self, tensor, B, C, transform_dist, vector):
        # we need the shapes
        l_shapes = transform_dist.l_shapes
        m_shapes = transform_dist.m_shapes

        # gather in W
        if self.grid_size_w > 1:
            if vector:
                gather_shapes = [(B, C, 2, l_shapes[self.hrank], m) for m in m_shapes]
            else:
                gather_shapes = [(B, C, l_shapes[self.hrank], m) for m in m_shapes]
            olist = [torch.empty(shape, dtype=tensor.dtype, device=tensor.device) for shape in gather_shapes]
            olist[self.wrank] = tensor
            dist.all_gather(olist, tensor, group=self.w_group)
            tensor_gather = torch.cat(olist, dim=-1)
        else:
            tensor_gather = tensor

        # gather in H
        if self.grid_size_h > 1:
            if vector:
                gather_shapes = [(B, C, 2, l, transform_dist.mmax) for l in l_shapes]
            else:
                gather_shapes = [(B, C, l, transform_dist.mmax) for l in l_shapes]
            olist = [torch.empty(shape, dtype=tensor_gather.dtype, device=tensor_gather.device) for shape in gather_shapes]
            olist[self.hrank] = tensor_gather
            dist.all_gather(olist, tensor_gather, group=self.h_group)
            tensor_gather = torch.cat(olist, dim=-2)

        return tensor_gather

    def _gather_helper_bwd(self, tensor, B, C, transform_dist, vector):
        # we need the shapes
        lat_shapes = transform_dist.lat_shapes
        lon_shapes = transform_dist.lon_shapes

        # gather in W
        if self.grid_size_w > 1:
            if vector:
                gather_shapes = [(B, C, 2, lat_shapes[self.hrank], w) for w in lon_shapes]
            else:
                gather_shapes = [(B, C, lat_shapes[self.hrank], w) for w in lon_shapes]
            olist = [torch.empty(shape, dtype=tensor.dtype, device=tensor.device) for shape in gather_shapes]
            olist[self.wrank] = tensor
            dist.all_gather(olist, tensor, group=self.w_group)
            tensor_gather = torch.cat(olist, dim=-1)
        else:
            tensor_gather = tensor

        # gather in H
        if self.grid_size_h > 1:
            if vector:
                gather_shapes = [(B, C, 2, h, transform_dist.nlon) for h in lat_shapes]
            else:
                gather_shapes = [(B, C, h, transform_dist.nlon) for h in lat_shapes]
            olist = [torch.empty(shape, dtype=tensor_gather.dtype, device=tensor_gather.device) for shape in gather_shapes]
            olist[self.hrank] = tensor_gather
            dist.all_gather(olist, tensor_gather, group=self.h_group)
            tensor_gather = torch.cat(olist, dim=-2)

        return tensor_gather

    @parameterized.expand([
        [256, 512, 32,  8, "equiangular",    False, 1e-9],
        [256, 512, 32,  8, "legendre-gauss", False, 1e-9],
        [256, 512, 32,  8, "equiangular",    False, 1e-9],
        [256, 512, 32,  8, "legendre-gauss", False, 1e-9],
        [256, 512, 32,  8, "equiangular",    False, 1e-9],
        [256, 512, 32,  8, "legendre-gauss", False, 1e-9],
        [361, 720,  1, 10, "equiangular",    False, 1e-6],
        [361, 720,  1, 10, "legendre-gauss", False, 1e-6],
        [256, 512, 32,  8, "equiangular",    True,  1e-9],
        [256, 512, 32,  8, "legendre-gauss", True,  1e-9],
        [256, 512, 32,  8, "equiangular",    True,  1e-9],
	[256, 512, 32,  8, "legendre-gauss", True,  1e-9],
        [256, 512, 32,  8, "equiangular",    True,  1e-9],
        [256, 512, 32,  8, "legendre-gauss", True,  1e-9],
	[361, 720,  1, 10, "equiangular",    True,  1e-6],
        [361, 720,  1, 10, "legendre-gauss", True,  1e-6],
    ])
    def test_distributed_sht(self, nlat, nlon, batch_size, num_chan, grid, vector, tol):
        B, C, H, W = batch_size, num_chan, nlat, nlon

        # set up handles
        if vector:
            forward_transform_local = harmonics.RealVectorSHT(nlat=H, nlon=W, grid=grid).to(self.device)
            forward_transform_dist = thd.DistributedRealVectorSHT(nlat=H, nlon=W, grid=grid).to(self.device)
        else:
            forward_transform_local = harmonics.RealSHT(nlat=H, nlon=W, grid=grid).to(self.device)
            forward_transform_dist = thd.DistributedRealSHT(nlat=H, nlon=W, grid=grid).to(self.device)

        # create tensors
        if vector:
            inp_full = torch.randn((B, C, 2, H, W), dtype=torch.float32, device=self.device)
        else:
            inp_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device)

        #############################################################
        # local transform
        #############################################################
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

        #############################################################
        # distributed transform
        #############################################################
        # FWD pass
        inp_local = self._split_helper(inp_full)
        inp_local.requires_grad = True
        out_local = forward_transform_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full)
        out_local = forward_transform_dist(inp_local)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        #############################################################
        # evaluate FWD pass
        #############################################################
        with torch.no_grad():
            out_gather_full = self._gather_helper_fwd(out_local, B, C, forward_transform_dist, vector)
            err = torch.mean(torch.norm(out_full-out_gather_full, p='fro', dim=(-1,-2)) / torch.norm(out_full, p='fro', dim=(-1,-2)) )
            if self.world_rank == 0:
                print(f"final relative error of output: {err.item()}")
        self.assertTrue(err.item() <= tol)

        #############################################################
        # evaluate BWD pass
        #############################################################
        with torch.no_grad():
            igrad_gather_full = self._gather_helper_bwd(igrad_local, B, C, forward_transform_dist, vector)
            err = torch.mean(torch.norm(igrad_full-igrad_gather_full, p='fro', dim=(-1,-2)) / torch.norm(igrad_full, p='fro', dim=(-1,-2)) )
            if self.world_rank == 0:
                print(f"final relative error of gradients: {err.item()}")
        self.assertTrue(err.item() <= tol)

if __name__ == '__main__':
    unittest.main()
        
#if vector:
#    forward_transform_local = harmonics.RealVectorSHT(nlat=H, nlon=W, grid=grid_type).to(device)
#    forward_transform_dist = thd.DistributedRealVectorSHT(nlat=H, nlon=W, grid=grid_type).to(device)
#else:
#    forward_transform_local = harmonics.RealSHT(nlat=H, nlon=W, grid=grid_type).to(device)
#    forward_transform_dist = thd.DistributedRealSHT(nlat=H, nlon=W, grid=grid_type).to(device)

# create tensors
#if vector:
#    inp_full = torch.randn((B, C, 2, H, W), dtype=torch.float32, device=device)
#else:
#    inp_full = torch.randn((B, C, H, W), dtype=torch.float32, device=device)

#############################################################
# local transform
#############################################################
# FWD pass
#inp_full.requires_grad = True
#out_full = forward_transform_local(inp_full)

# create grad for backward
#with torch.no_grad():
#    # create full grad
#    ograd_full = torch.randn_like(out_full)

# BWD pass
#out_full.backward(ograd_full)
#igrad_full = inp_full.grad.clone()

#############################################################
# distributed transform
#############################################################
# split input
#with torch.no_grad():
#    # split in W
#    inp_list_local = thd.split_tensor_along_dim(inp_full, dim=-1, num_chunks=grid_size_w)
#    shapes_w = [x.shape[-1] for x in inp_list_local]
#    inp_local = inp_list_local[wrank]
#
#    # split in H
#    inp_list_local = thd.split_tensor_along_dim(inp_local, dim=-2, num_chunks=grid_size_h)
#    shapes_h = [x.shape[-2] for x in inp_list_local]
#    inp_local = inp_list_local[hrank]
    
# FWD pass
#inp_local.requires_grad = True
#out_local = forward_transform_dist(inp_local)

# split grad
# create split input grad
#with torch.no_grad():
#    # split in M
#    ograd_list_local = thd.split_tensor_along_dim(ograd_full, dim=-1, num_chunks=grid_size_w)
#    shapes_m = [x.shape[-1] for x in ograd_list_local]
#    ograd_local = ograd_list_local[wrank]

#    # split in L
#    ograd_list_local = thd.split_tensor_along_dim(ograd_local, dim=-2, num_chunks=grid_size_h)
#    shapes_l = [x.shape[-2] for x in ograd_list_local]
#    ograd_local = ograd_list_local[hrank]

# BWD pass
#out_local = forward_transform_dist(inp_local)
#out_local.backward(ograd_local)
#igrad_local = inp_local.grad.clone()


#############################################################
# evaluate FWD pass
#############################################################
# gather the local data
# we need the shapes
#l_shapes = forward_transform_dist.l_shapes
#m_shapes = forward_transform_dist.m_shapes

# gather in W
#if grid_size_w > 1:
#    if vector:
#        gather_shapes = [(B, C, 2, l_shapes[hrank], m) for m in m_shapes]
#    else:
#        gather_shapes = [(B, C, l_shapes[hrank], m) for m in m_shapes]
#    olist = [torch.empty(shape, dtype=out_local.dtype, device=out_local.device) for shape in gather_shapes]
#    olist[wrank] = out_local
#    dist.all_gather(olist, out_local, group=w_group)
#    out_full_gather = torch.cat(olist, dim=-1)
#else:
#    out_full_gather = out_local
    
# gather in H
#if grid_size_h > 1:
#    if vector:
#        gather_shapes = [(B, C, 2, l, forward_transform_dist.mmax) for l in l_shapes]
#    else:
#        gather_shapes = [(B, C, l, forward_transform_dist.mmax) for l in l_shapes]
#    olist = [torch.empty(shape, dtype=out_full_gather.dtype, device=out_full_gather.device) for shape in gather_shapes]
#    olist[hrank] = out_full_gather
#    dist.all_gather(olist, out_full_gather, group=h_group)
#    out_full_gather = torch.cat(olist, dim=-2)


#if world_rank == 0:
#    with torch.no_grad():
#        print(f"Local Out: sum={out_full.abs().sum().item()}, max={out_full.abs().max().item()}, min={out_full.abs().min().item()}")
#        print(f"Dist Out: sum={out_full_gather.abs().sum().item()}, max={out_full_gather.abs().max().item()}, min={out_full_gather.abs().min().item()}")
#        diff = (out_full-out_full_gather).abs()
#        print(f"Out Difference: abs={diff.sum().item()}, rel={diff.sum().item() / (0.5*(out_full.abs().sum() + out_full_gather.abs().sum()))}, max={diff.abs().max().item()}")
#        print("")

#############################################################
# evaluate BWD pass
#############################################################
# gather
# we need the shapes
#lat_shapes = forward_transform_dist.lat_shapes
#lon_shapes = forward_transform_dist.lon_shapes

# gather in W
#if grid_size_w > 1:
#    if vector:
#        gather_shapes = [(B, C, 2, lat_shapes[hrank], w) for w in lon_shapes]
#    else:
#        gather_shapes = [(B, C, lat_shapes[hrank], w) for w in lon_shapes]
#    olist = [torch.empty(shape, dtype=igrad_local.dtype, device=igrad_local.device) for shape in gather_shapes]
#    olist[wrank] = igrad_local
#    dist.all_gather(olist, igrad_local, group=w_group)
#    igrad_full_gather = torch.cat(olist, dim=-1)
#else:
#    igrad_full_gather = igrad_local

# gather in h
#if grid_size_h > 1:
#    if vector:
#        gather_shapes = [(B, C, 2, h, forward_transform_dist.nlon) for h in lat_shapes]
#    else:
#        gather_shapes = [(B, C, h, forward_transform_dist.nlon) for h in lat_shapes]
#    olist = [torch.empty(shape, dtype=igrad_full_gather.dtype, device=igrad_full_gather.device) for shape in gather_shapes]
#    olist[hrank] = igrad_full_gather
#    dist.all_gather(olist, igrad_full_gather, group=h_group)
#    igrad_full_gather = torch.cat(olist, dim=-2)

#if world_rank == 0:
#    with torch.no_grad():
#        print(f"Local Grad: sum={igrad_full.abs().sum().item()}, max={igrad_full.abs().max().item()}, min={igrad_full.abs().min().item()}")
#        print(f"Dist Grad: sum={igrad_full_gather.abs().sum().item()}, max={igrad_full_gather.abs().max().item()}, min={igrad_full_gather.abs().min().item()}")
#        diff = (igrad_full-igrad_full_gather).abs()
#        print(f"Grad Difference: abs={diff.sum().item()}, rel={diff.sum().item() / (0.5*(igrad_full.abs().sum() + igrad_full_gather.abs().sum()))}, max={diff.abs().max().item()}")
