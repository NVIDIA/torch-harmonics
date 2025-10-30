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
import torch.nn.functional as F
import torch.distributed as dist
import torch_harmonics as th
import torch_harmonics.distributed as thd
from torch_harmonics.disco import optimized_kernels_is_available

from disco_helpers import preprocess_psi
from torch_harmonics.filter_basis import get_filter_basis
from torch_harmonics.disco.convolution import _precompute_convolution_tensor_s2
from torch_harmonics.distributed.distributed_convolution import _split_distributed_convolution_tensor_s2

from testutils import compare_tensors

if not optimized_kernels_is_available():
    print(f"Warning: Couldn't import optimized disco convolution kernels")


class TestDistributedDiscreteContinuousConvolution(unittest.TestCase):
    """Test the distributed discrete-continuous convolution module."""

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

        # initializing sht
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

        # make contiguous
        tensor_local = tensor_local.contiguous()

        return tensor_local

    def _gather_helper_fwd(self, tensor, B, C, convolution_dist):

        # we need the shapes
        lat_shapes = convolution_dist.lat_out_shapes
        lon_shapes = convolution_dist.lon_out_shapes

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
            gather_shapes = [(B, C, h, convolution_dist.nlon_out) for h in lat_shapes]
            olist = [torch.empty(shape, dtype=tensor_gather.dtype, device=tensor_gather.device) for shape in gather_shapes]
            olist[self.hrank] = tensor_gather
            dist.all_gather(olist, tensor_gather, group=self.h_group)
            tensor_gather = torch.cat(olist, dim=-2)

        # make contiguous
        tensor_gather = tensor_gather.contiguous()

        return tensor_gather

    def _gather_helper_bwd(self, tensor, B, C, convolution_dist):

        # we need the shapes
        lat_shapes = convolution_dist.lat_in_shapes
        lon_shapes = convolution_dist.lon_in_shapes

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
            gather_shapes = [(B, C, h, convolution_dist.nlon_in) for h in lat_shapes]
            olist = [torch.empty(shape, dtype=tensor_gather.dtype, device=tensor_gather.device) for shape in gather_shapes]
            olist[self.hrank] = tensor_gather
            dist.all_gather(olist, tensor_gather, group=self.h_group)
            tensor_gather = torch.cat(olist, dim=-2)

        # make contiguous
        tensor_gather = tensor_gather.contiguous()

        return tensor_gather

    @parameterized.expand(
        [
            # piecewise linear
            # normal isotropic
            [(16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular"],
            [(17, 32), (17, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular"],
            # normal anisotropic
            [(16, 32), (16, 32), (3, 4), "piecewise linear", "mean", "equiangular", "equiangular"],
            [(16, 32), (16, 32), (3, 2), "piecewise linear", "mean", "equiangular", "equiangular"],
            # downsampling isotropic
            [(16, 32), (8, 16), (3), "piecewise linear", "mean", "equiangular", "equiangular"],
            [(17, 32), (9, 16), (3), "piecewise linear", "mean", "equiangular", "equiangular"],
            # downsampling anisotropic
            [(16, 32), (8, 16), (3, 4), "piecewise linear", "mean", "equiangular", "equiangular"],
            [(16, 32), (8, 16), (3, 2), "piecewise linear", "mean", "equiangular", "equiangular"],
            # morlet
            # normal isotropic
            [(16, 32), (16, 32), (1), "morlet", "mean", "equiangular", "equiangular"], # important for attention
            [(16, 32), (16, 32), (3), "morlet", "mean", "equiangular", "equiangular"],
            [(17, 32), (17, 32), (3), "morlet", "mean", "equiangular", "equiangular"],
            # normal anisotropic
            [(16, 32), (16, 32), (3, 4), "morlet", "mean", "equiangular", "equiangular"],
            [(16, 32), (16, 32), (3, 2), "morlet", "mean", "equiangular", "equiangular"],
            # downsampling isotropic
            [(16, 32), (8, 16), (1), "morlet", "mean", "equiangular", "equiangular"], # important for attention
            [(16, 32), (8, 16), (3), "morlet", "mean", "equiangular", "equiangular"],
            [(17, 32), (9, 16), (3), "morlet", "mean", "equiangular", "equiangular"],
            # downsampling anisotropic
            [(16, 32), (8, 16), (3, 4), "morlet", "mean", "equiangular", "equiangular"],
            [(16, 32), (8, 16), (3, 2), "morlet", "mean", "equiangular", "equiangular"],
            # zernike
            # normal 
            [(16, 32), (16, 32), (1), "zernike", "mean", "equiangular", "equiangular"],
            [(16, 32), (16, 32), (3, 3), "zernike", "mean", "equiangular", "equiangular"],
            [(17, 32), (17, 32), (3, 3), "zernike", "mean", "equiangular", "equiangular"],
            # downsampling
            [(16, 32), (8, 16), (1), "zernike", "mean", "equiangular", "equiangular"],
            [(16, 32), (8, 16), (3, 3), "zernike", "mean", "equiangular", "equiangular"],
            [(17, 32), (9, 16), (3, 3), "zernike", "mean", "equiangular", "equiangular"],
        ],
        skip_on_empty=True,
    )
    def test_distributed_convolution_tensor_integrity(self, in_shape, out_shape, kernel_shape, basis_type, basis_norm_mode, grid_in, grid_out, verbose=False):

        nlat_in, _ = in_shape
        nlat_out, _ = out_shape

        # get filter basis
        filter_basis = get_filter_basis(kernel_shape=kernel_shape, basis_type=basis_type)

        # use default value for cutoff
        theta_cutoff = torch.pi / float(nlat_out - 1)

        idx, vals, _ = _precompute_convolution_tensor_s2(
            in_shape=in_shape,
            out_shape=out_shape,
            filter_basis=filter_basis,
            grid_in=grid_in,
            grid_out=grid_out,
            theta_cutoff=theta_cutoff,
            transpose_normalization=False,
            basis_norm_mode=basis_norm_mode,
            merge_quadrature=True,
        )

        # split tensor along latitude
        idx, vals = _split_distributed_convolution_tensor_s2(idx, vals, in_shape)

        # define contiguous tensors
        ker_idx = idx[0, ...].contiguous()
        row_idx = idx[1, ...].contiguous()
        col_idx = idx[2, ...].contiguous()
        vals = vals.contiguous()

        # sort values
        roff_idx = preprocess_psi(filter_basis.kernel_size, nlat_out, ker_idx, row_idx, col_idx, vals).contiguous()

        print(f"{self.hrank} after splitting sorted \n ker = {ker_idx}\n row = {row_idx}\n col = {col_idx}", flush=True)
        print(f"{self.hrank} roff_idx = {roff_idx}", flush=True)
        print(f"{self.hrank} ker_shape = {ker_idx.shape}, row_shape = {row_idx.shape}, col_shape = {col_idx.shape}, vals_shape = {vals.shape}", flush=True)

        # check shapes
        self.assertTrue(ker_idx.shape[0] == row_idx.shape[0], f"ker_idx and row_idx have to have the same shape: found {ker_idx.shape[0]} and {row_idx.shape[0]}")
        self.assertTrue(ker_idx.shape[0] == col_idx.shape[0], f"ker_idx and col_idx have to have the same shape: found {ker_idx.shape[0]} and {col_idx.shape[0]}")
        self.assertTrue(ker_idx.shape[0] == vals.shape[0], f"ker_idx and vals have to have the same shape: found {ker_idx.shape[0]} and {vals.shape[0]}")

        # the multiplicitiy in ker_idx has to be the same for all kernel indices
        unique, counts = torch.unique(ker_idx, return_counts=True)
        self.assertTrue(torch.all(counts.max() == counts), f"The multiplicity in ker_idx has to be the same for all kernel indices: found {counts} for entries {unique}")

        if verbose:
            print(f"\n ker_idx = {ker_idx},\n row_idx = {row_idx},\n col_idx = {col_idx}")

        # the following has to be true: the row_idx and col_idx have to be the same for all kernel indices
        row_idx_ref = row_idx[ker_idx == 0]
        col_idx_ref = col_idx[ker_idx == 0]
        for k in range(1, filter_basis.kernel_size):
            self.assertTrue(torch.all(row_idx_ref == row_idx[ker_idx == k]), f"The row_idx has to be the same for all kernel indices: found {row_idx_ref} for entries {ker_idx == k}")
            self.assertTrue(torch.all(col_idx_ref == col_idx[ker_idx == k]), f"The row_idx has to be the same for all kernel indices: found {col_idx_ref} for entries {ker_idx == k}")


    @parameterized.expand(
        [
            # Forward
            [128, 256, 128, 256, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", False, 1e-5],
            [129, 256, 128, 256, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", False, 1e-5],
            [128, 256, 128, 256, 32, 8, (3, 2), "piecewise linear", "mean", 1, "equiangular", "equiangular", False, 1e-5],
            [128, 256, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", False, 1e-5],
            [128, 256, 128, 256, 32, 8, (3), "piecewise linear", "mean", 2, "equiangular", "equiangular", False, 1e-5],
            [128, 256, 128, 256, 32, 6, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", False, 1e-5],
            [129, 256, 129, 256, 32, 8, (3, 4), "morlet", "mean", 1, "equiangular", "equiangular", False, 1e-5],
            [129, 256, 65, 128, 32, 8, (3, 4), "morlet", "mean", 1, "equiangular", "equiangular", False, 1e-5],
            # Transpose
            [128, 256, 128, 256, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", True, 1e-5],
            [129, 256, 129, 256, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", True, 1e-5],
            [128, 256, 128, 256, 32, 8, (3, 2), "piecewise linear", "mean", 1, "equiangular", "equiangular", True, 1e-5],
            [64, 128, 128, 256, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", True, 1e-5],
            [128, 256, 128, 256, 32, 8, (3), "piecewise linear", "mean", 2, "equiangular", "equiangular", True, 1e-5],
            [128, 256, 128, 256, 32, 6, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", True, 1e-5],
            [129, 256, 129, 256, 32, 8, (3, 4), "morlet", "mean", 1, "equiangular", "equiangular", True, 1e-5],
            [65, 128, 129, 256, 32, 8, (3, 4), "morlet", "mean", 1, "equiangular", "equiangular", True, 1e-5],
        ],
        skip_on_empty=True,
    )
    def test_distributed_disco_conv(
        self, nlat_in, nlon_in, nlat_out, nlon_out, batch_size, num_chan, kernel_shape, basis_type, basis_norm_mode, groups, grid_in, grid_out, transpose, tol, verbose=True
    ):
        verbose = verbose and (self.world_rank == 0)
        
        B, C, H, W = batch_size, num_chan, nlat_in, nlon_in

        disco_args = dict(
            in_channels=C,
            out_channels=C,
            in_shape=(nlat_in, nlon_in),
            out_shape=(nlat_out, nlon_out),
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            kernel_shape=kernel_shape,
            groups=groups,
            grid_in=grid_in,
            grid_out=grid_out,
            bias=True,
            optimized_kernel=True,
        )

        # set up handles
        if transpose:
            conv_full = th.DiscreteContinuousConvTransposeS2(**disco_args).to(self.device)
            conv_dist = thd.DistributedDiscreteContinuousConvTransposeS2(**disco_args).to(self.device)
        else:
            conv_full = th.DiscreteContinuousConvS2(**disco_args).to(self.device)
            conv_dist = thd.DistributedDiscreteContinuousConvS2(**disco_args).to(self.device)

        # copy the weights from the local conv into the dist conv
        with torch.no_grad():
            conv_dist.weight.copy_(conv_full.weight)
            if disco_args["bias"]:
                conv_dist.bias.copy_(conv_full.bias)

        # create tensors
        inp_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device)

        # local conv
        # FWD pass
        inp_full.requires_grad = True
        out_full = conv_full(inp_full)

        # create grad for backward
        with torch.no_grad():
            # create full grad
            ograd_full = torch.randn_like(out_full)

        # BWD pass
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        # distributed conv
        # FWD pass
        with torch.no_grad():
            inp_local = self._split_helper(inp_full)
        inp_local.requires_grad = True
        out_local = conv_dist(inp_local)

        # BWD pass
        with torch.no_grad():
            ograd_local = self._split_helper(ograd_full)
        out_local = conv_dist(inp_local)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        # evaluate FWD pass
        with torch.no_grad():
            out_gather_full = self._gather_helper_fwd(out_local, B, C, conv_dist)
        self.assertTrue(compare_tensors("output", out_gather_full, out_full, rtol=tol, atol=tol, verbose=verbose))

        # evaluate BWD pass
        with torch.no_grad():
            igrad_gather_full = self._gather_helper_bwd(igrad_local, B, C, conv_dist)
        self.assertTrue(compare_tensors("input gradient", igrad_gather_full, igrad_full, rtol=tol, atol=tol, verbose=verbose))


if __name__ == "__main__":
    unittest.main()
