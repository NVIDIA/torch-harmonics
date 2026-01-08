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
import torch
import torch.distributed as dist
import torch_harmonics.distributed as thd


def setup_distributed(cls):
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

    dist.init_process_group(
        backend=proc_backend,
        init_method=f"tcp://{master_address}:{port}",
        rank=cls.world_rank,
        world_size=cls.world_size,
    )

    cls.wrank = cls.world_rank % cls.grid_size_w
    cls.hrank = cls.world_rank // cls.grid_size_w

    cls.w_group = None
    cls.h_group = None

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

    thd.init(cls.h_group, cls.w_group)

    return


def teardown_distributed(cls):
    thd.finalize()
    if cls.h_group is not None:
        dist.destroy_process_group(cls.h_group)
    if cls.w_group is not None:
        dist.destroy_process_group(cls.w_group)
    dist.destroy_process_group(None)

    return


def split_tensor_hw(tensor, hdim=-2, wdim=-1, hsize=1, wsize=1, hrank=0, wrank=0):
    """Split tensor along height/width according to process grid ranks."""
    with torch.no_grad():
        tensor_list_local = thd.split_tensor_along_dim(tensor, dim=wdim, num_chunks=wsize)
        tensor_local = tensor_list_local[wrank]

        tensor_list_local = thd.split_tensor_along_dim(tensor_local, dim=hdim, num_chunks=hsize)
        tensor_local = tensor_list_local[hrank]

    return tensor_local


def gather_tensor_hw(tensor, hdim=-2, wdim=-1, hshapes=[], wshapes=[], hsize=1, wsize=1, hrank=0, wrank=0, hgroup=None, wgroup=None):
    """Gather tensor along height/width according to process grid ranks and shapes."""
    with torch.no_grad():
        tensor = tensor.contiguous()
        if wsize > 1:
            local_shape = list(tensor.shape)
            gather_shapes = []
            for w in wshapes:
                local_shape[wdim] = w
                gather_shapes.append(tuple(local_shape))
            olist = [torch.empty(shape, dtype=tensor.dtype, device=tensor.device) for shape in gather_shapes]
            olist[wrank] = tensor
            dist.all_gather(olist, tensor, group=wgroup)
            tensor = torch.cat(olist, dim=wdim)

        if hsize > 1:
            local_shape = list(tensor.shape)
            gather_shapes = []
            for h in hshapes:
                local_shape[hdim] = h
                gather_shapes.append(tuple(local_shape))
            olist = [torch.empty(shape, dtype=tensor.dtype, device=tensor.device) for shape in gather_shapes]
            olist[hrank] = tensor
            dist.all_gather(olist, tensor, group=hgroup)
            tensor = torch.cat(olist, dim=hdim)

    return tensor


def compare_tensors(msg, tensor1, tensor2, atol=1e-8, rtol=1e-5, verbose=False):

    # some None checks
    if tensor1 is None and tensor2 is None:
        allclose = True
    elif tensor1 is None and tensor2 is not None:
        allclose = False
        if verbose:
            print(f"tensor1 is None and tensor2 is not None")
    elif tensor1 is not None and tensor2 is None:
        allclose = False
        if verbose:
            print(f"tensor1 is not None and tensor2 is None")
    else:
        diff = torch.abs(tensor1 - tensor2)
        abs_diff = torch.mean(diff, dim=0)
        rel_diff = torch.mean(diff / torch.clamp(torch.abs(tensor2), min=1e-6), dim=0)
        allclose = torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol)
        if not allclose and verbose:
            print(f"Absolute difference on {msg}: min = {abs_diff.min()}, mean = {abs_diff.mean()}, max = {abs_diff.max()}")
            print(f"Relative difference on {msg}: min = {rel_diff.min()}, mean = {rel_diff.mean()}, max = {rel_diff.max()}")
            print(f"Element values with max difference on {msg}: {tensor1.flatten()[diff.argmax()]} and {tensor2.flatten()[diff.argmax()]}")
            # find violating entry
            worst_diff = torch.argmax(diff - (atol + rtol * torch.abs(tensor2)))
            diff_bad = diff.flatten()[worst_diff].item()
            tensor2_abs_bad = torch.abs(tensor2).flatten()[worst_diff].item()
            print(f"Worst allclose condition violation: {diff_bad} <= {atol} + {rtol} * {tensor2_abs_bad} = {atol + rtol * tensor2_abs_bad}")

    return allclose