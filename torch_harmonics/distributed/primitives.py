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
from typing import List

import torch
import torch.distributed as dist

from .utils import polar_group, azimuth_group, is_initialized

# helper routine to compute uneven splitting in balanced way:
def compute_split_shapes(size: int, num_chunks: int) -> List[int]:
    
    # treat trivial case first
    if num_chunks == 1:
        return [size]
    
    # first, check if we can split using div-up to balance the load: 
    chunk_size = (size + num_chunks - 1) // num_chunks
    last_chunk_size = max(0, size - chunk_size * (num_chunks - 1))
    if last_chunk_size == 0:
        # in this case, the last shard would be empty, split with floor instead:
        chunk_size = size // num_chunks
        last_chunk_size = size - chunk_size * (num_chunks-1)

    # generate sections list
    sections = [chunk_size for _ in range(num_chunks - 1)] + [last_chunk_size]

    return sections


# general helpers
def get_memory_format(tensor):
    if tensor.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    else:
        return torch.contiguous_format

    
def split_tensor_along_dim(tensor, dim, num_chunks):
    assert dim < tensor.dim(), f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}"
    assert (tensor.shape[dim] >= num_chunks), f"Error, cannot split dim {dim} of size {tensor.shape[dim]} into \
                                              {num_chunks} chunks. Empty slices are currently not supported."
    
    # get split
    sections = compute_split_shapes(tensor.shape[dim], num_chunks)
    tensor_list = torch.split(tensor, sections, dim=dim)
    
    return tensor_list


def _transpose(tensor, dim0, dim1, dim1_split_sizes, group=None, async_op=False):
    # get input format
    input_format = get_memory_format(tensor)
    
    # get comm params
    comm_size = dist.get_world_size(group=group)
    comm_rank = dist.get_rank(group=group)

    # split and local transposition
    tsplit = split_tensor_along_dim(tensor, num_chunks=comm_size, dim=dim0)
    x_send = [y.contiguous(memory_format=input_format) for y in tsplit]
    x_send_shapes = [x.shape for x in x_send]
    x_recv = []
    x_shape = list(x_send_shapes[comm_rank])
    for dim1_len in dim1_split_sizes:
        x_shape[dim1] = dim1_len
        x_recv.append(torch.empty(x_shape, dtype=tensor.dtype, device=tensor.device, memory_format=input_format))
    
    # global transposition
    req = dist.all_to_all(x_recv, x_send, group=group, async_op=async_op)

    # get dim0 split sizes
    dim0_split_sizes = [x[dim0] for x in x_send_shapes]
    
    return x_recv, dim0_split_sizes, req


class distributed_transpose_azimuth(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, dims, dim1_split_sizes):
        input_format = get_memory_format(x)
        # WAR for a potential contig check torch bug for channels last contig tensors
        x = x.contiguous()
        xlist, dim0_split_sizes, _ = _transpose(x, dims[0], dims[1], dim1_split_sizes, group=azimuth_group())
        x = torch.cat(xlist, dim=dims[1]).contiguous(memory_format=input_format)
        ctx.dims = dims
        ctx.dim0_split_sizes = dim0_split_sizes
        return x

    @staticmethod
    def backward(ctx, go):
        input_format = get_memory_format(go)
        dims = ctx.dims
        dim0_split_sizes = ctx.dim0_split_sizes
        # WAR for a potential contig check torch bug for channels last contig tensors 
        go = go.contiguous()
        gilist, _, _ = _transpose(go, dims[1], dims[0], dim0_split_sizes, group=azimuth_group())
        gi = torch.cat(gilist, dim=dims[0]).contiguous(memory_format=input_format)
        return gi, None, None

    
class distributed_transpose_polar(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, dim, dim1_split_sizes):
        input_format = get_memory_format(x)
        # WAR for a potential contig check torch bug for channels last contig tensors 
        x = x.contiguous()
        xlist, dim0_split_sizes, _ = _transpose(x, dim[0], dim[1], dim1_split_sizes, group=polar_group())
        x = torch.cat(xlist, dim=dim[1]).contiguous(memory_format=input_format)
        ctx.dim = dim
        ctx.dim0_split_sizes = dim0_split_sizes
        return x

    @staticmethod
    def backward(ctx, go):
        input_format = get_memory_format(go)
        dim = ctx.dim
        dim0_split_sizes = ctx.dim0_split_sizes
        # WAR for a potential contig check torch bug for channels last contig tensors 
        go = go.contiguous()
        gilist, _, _ = _transpose(go, dim[1], dim[0], dim0_split_sizes, group=polar_group())
        gi = torch.cat(gilist, dim=dim[0]).contiguous(memory_format=input_format)
        return gi, None, None

