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

import torch
import torch.distributed as dist

from .utils import get_model_parallel_group, is_initialized

# general helpers
def get_memory_format(tensor):
    if tensor.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    else:
        return torch.contiguous_format

def split_tensor_along_dim(tensor, dim, num_chunks):
    assert dim < tensor.dim(), f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}"
    assert (tensor.shape[dim] % num_chunks == 0), f"Error, cannot split dim {dim} evenly. Dim size is \
                                                   {tensor.shape[dim]} and requested numnber of splits is {num_chunks}"
    chunk_size = tensor.shape[dim] // num_chunks
    tensor_list = torch.split(tensor, chunk_size, dim=dim)
    
    return tensor_list

# split
def _split(input_, dim_, group=None):
    """Split the tensor along its last dimension and keep the corresponding slice."""
    # get input format
    input_format = get_memory_format(input_)
    
    # Bypass the function if we are using only 1 GPU.
    comm_size = dist.get_world_size(group=group)
    if comm_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_dim(input_, dim_, comm_size)
    
    # Note: torch.split does not create contiguous tensors by default.
    rank = dist.get_rank(group=group)
    output = input_list[rank].contiguous(memory_format=input_format)
    
    return output


# those are used by the various helper functions
def _reduce(input_, use_fp32=True, group=None):
    """All-reduce the input tensor across model parallel group."""
    
    # Bypass the function if we are using only 1 GPU.
    if dist.get_world_size(group=group) == 1:
        return input_

    # All-reduce.
    if use_fp32:
        dtype = input_.dtype
        inputf_ = input_.float()
        dist.all_reduce(inputf_, group=group)
        input_ = inputf_.to(dtype)
    else:
        dist.all_reduce(input_, group=group)
        
    return input_

class _CopyToParallelRegion(torch.autograd.Function):
    """Pass the input to the parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, group=get_model_parallel_group())

# write a convenient functional wrapper
def copy_to_parallel_region(input_):
    if not is_initialized():
        return input_
    else:
        return _CopyToParallelRegion.apply(input_)

# reduce
class _ReduceFromParallelRegion(torch.autograd.Function):
    """All-reduce the input from the parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_, group=get_model_parallel_group())
    
    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_, group=get_model_parallel_group())

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
                                                        

def reduce_from_parallel_region(input_):
    if not is_initialized():
        return input_
    else:
        return _ReduceFromParallelRegion.apply(input_)

# gather
def _gather(input_, dim_, group=None):
    """Gather tensors and concatinate along the last dimension."""
    # get input format
    input_format = get_memory_format(input_)

    print(input_format)

    comm_size = dist.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if comm_size==1:
        return input_

    # sanity checks
    assert(dim_ < input_.dim()), f"Error, cannot gather along {dim_} for tensor with {input_.dim()} dimensions."
    
    # Size and dimension.
    comm_rank = dist.get_rank(group=group)

    # input needs to be contiguous
    input_ = input_.contiguous(memory_format=input_format)
    tensor_list = [torch.empty_like(input_) for _ in range(comm_size)]
    tensor_list[comm_rank] = input_
    dist.all_gather(tensor_list, input_, group=group)
    output = torch.cat(tensor_list, dim=dim_).contiguous(memory_format=input_format)
    
    return output
                                                                            

class _GatherFromParallelRegion(torch.autograd.Function):
    """Gather the input from parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, dim_):
        return _gather(input_, dim_, group=get_model_parallel_group())
    
    @staticmethod
    def forward(ctx, input_, dim_):
        ctx.dim = dim_
        return _gather(input_, dim_, group=get_model_parallel_group())

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.dim, group=get_model_parallel_group()), None

    
def gather_from_parallel_region(input_, dim):
    if not is_initialized():
        return input_
    else:
        return _GatherFromParallelRegion.apply(input_, dim)

# scatter
class _ScatterToParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""
    
    @staticmethod
    def symbolic(graph, input_, dim_):
        return _split(input_, dim_, group=get_model_parallel_group())

    @staticmethod
    def forward(ctx, input_, dim_):
        ctx.dim = dim_
        return _split(input_, dim_, group=get_model_parallel_group())

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output, ctx.dim, group=get_model_parallel_group()), None

def scatter_to_parallel_region(input_, dim):
    if not is_initialized():
        return input_
    else:
        return _ScatterToParallelRegion.apply(input_, dim)
    
