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

from .utils import polar_group, azimuth_group, is_initialized

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

def _transpose(tensor, dim0, dim1, group=None, async_op=False):
    # get input format
    input_format = get_memory_format(tensor)
    
    # get comm params
    comm_size = dist.get_world_size(group=group)

    # split and local transposition
    split_size = tensor.shape[dim0] // comm_size
    x_send = [y.contiguous(memory_format=input_format) for y in torch.split(tensor, split_size, dim=dim0)]
    x_recv = [torch.empty_like(x_send[0]).contiguous(memory_format=input_format) for _ in range(comm_size)]
    
    # global transposition
    req = dist.all_to_all(x_recv, x_send, group=group, async_op=async_op)
    
    return x_recv, req 


class distributed_transpose_azimuth(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, dim):
        input_format = get_memory_format(x)
        # WAR for a potential contig check torch bug for channels last contig tensors
        x = x.contiguous()
        xlist, _ = _transpose(x, dim[0], dim[1], group=azimuth_group())
        x = torch.cat(xlist, dim=dim[1]).contiguous(memory_format=input_format)
        ctx.dim = dim
        return x

    @staticmethod
    def backward(ctx, go):
        input_format = get_memory_format(go)
        dim = ctx.dim
        # WAR for a potential contig check torch bug for channels last contig tensors 
        go = go.contiguous()
        gilist, _ = _transpose(go, dim[1], dim[0], group=azimuth_group())
        gi = torch.cat(gilist, dim=dim[0]).contiguous(memory_format=input_format)
        return gi, None

    
class distributed_transpose_polar(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, dim):
        input_format = get_memory_format(x)
        # WAR for a potential contig check torch bug for channels last contig tensors 
        x = x.contiguous()
        xlist, _ = _transpose(x, dim[0], dim[1], group=polar_group())
        x = torch.cat(xlist, dim=dim[1]).contiguous(memory_format=input_format)
        ctx.dim = dim
        return x

    @staticmethod
    def backward(ctx, go):
        input_format = get_memory_format(go)
        dim = ctx.dim
        # WAR for a potential contig check torch bug for channels last contig tensors 
        go = go.contiguous()
        gilist, _ = _transpose(go, dim[1], dim[0], group=polar_group())
        gi = torch.cat(gilist, dim=dim[0]).contiguous(memory_format=input_format)
        return gi, None

