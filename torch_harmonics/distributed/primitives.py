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

from .utils import polar_group, azimuth_group, polar_group_size
from .utils import is_initialized, is_distributed_polar

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

    
# we need those additional primitives for distributed matrix multiplications
def _reduce(input_, use_fp32=True, group=None):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if dist.get_world_size(group=group) == 1:
        return input_

    # make input contiguous
    input_ = input_.contiguous()
    
    # All-reduce.
    if use_fp32:
        dtype = input_.dtype
        inputf_ = input_.float()
        dist.all_reduce(inputf_, group=group)
        input_ = inputf_.to(dtype)
    else:
        dist.all_reduce(input_, group=group)
        
    return input_


def _split(input_, dim_, group=None):
    """Split the tensor along its last dimension and keep the corresponding slice."""
    # Bypass the function if we are using only 1 GPU.
    comm_size = dist.get_world_size(group=group)
    if comm_size == 1:
        return input_
    
    # Split along last dimension.
    input_list = split_tensor_along_dim(input_, dim_, comm_size)
    
    # Note: torch.split does not create contiguous tensors by default.
    rank = dist.get_rank(group=group)
    output = input_list[rank].contiguous()
    
    return output


def _gather(input_, dim_, shapes_, group=None):
    """Gather unevenly split tensors across ranks"""
    
    comm_size = dist.get_world_size(group=group)

    if (shapes_ is not None) and (len(shapes_) != comm_size):
        raise ValueError()
    if dim_ >= input_.dim():
        raise ValueError()

    if comm_size == 1:
        return input_

    # make contiguous:
    input_ = input_.contiguous()
    input_shape = list(input_.shape)

    if shapes_ is not None:
        input_list = [None] * comm_size

        for src in range(comm_size):
            input_shape[dim_] = shapes_[src]
            input_list[src] = torch.empty(
                input_shape,
                dtype=input_.dtype,
                device=input_.device,
            )
    else:
        # assume equal shape on all ranks
        input_list = [torch.empty_like(input_) for _ in range(comm_size)]

    dist.all_gather(input_list, input_, group=group)

    output = torch.cat(input_list, dim=dim_).contiguous()

    return output
    
    
class _ScatterToPolarRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chunk to the rank."""

    @staticmethod
    def symbolic(graph, input_, dim_):
        return _split(input_, dim_, group=polar_group())

    @staticmethod
    def forward(ctx, input_, dim_):
        if is_distributed_polar():
            ctx.dim = dim_
            ctx.split_shapes = compute_split_shapes(
                input_.shape[dim_], polar_group_size()
            )
            return _split(input_, dim_, group=polar_group())
        else:
            return input_

    @staticmethod
    def backward(ctx, grad_output):
        if is_distributed_polar():
            return _gather(grad_output, ctx.dim, ctx.split_shapes, polar_group()), None
        else:
            return grad_output, None

    
class _ReduceFromPolarRegion(torch.autograd.Function):
    """All-reduce the input from the polar region."""
    
    @staticmethod
    def symbolic(graph, input_):
        if is_distributed_polar():
            return _reduce(input_, group=polar_group())
        else:
            return input_

    @staticmethod
    def forward(ctx, input_):
        if is_distributed_polar():
            return _reduce(input_, group=polar_group())
        else:
            return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

    
def reduce_from_polar_region(input_):
    return _ReduceFromPolarRegion.apply(input_)


def scatter_to_polar_region(input_, dim_):
    return _ScatterToPolarRegion.apply(input_, dim_)
