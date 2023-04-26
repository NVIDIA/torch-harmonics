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

# ignore this (just for development without installation)
import sys
import os
sys.path.append("..")
sys.path.append(".")

import torch
import torch.distributed as dist
import torch_harmonics as harmonics

try:
    from tqdm import tqdm
except:
    tqdm = lambda x : x

# set up distributed
world_size = int(os.getenv('WORLD_SIZE', 1))
world_rank = int(os.getenv('WORLD_RANK', 0))
port = int(os.getenv('MASTER_PORT', 0))
master_address = os.getenv('MASTER_ADDR', 'localhost')
dist.init_process_group(backend = 'nccl',
                        init_method = f"tcp://{master_address}:{port}",
                        rank = world_rank,
                        world_size = world_size)
local_rank = world_rank % torch.cuda.device_count()
mp_group = dist.new_group(ranks=list(range(world_size)))
my_rank = dist.get_rank(mp_group)
group_size = 1 if not dist.is_initialized() else dist.get_world_size(mp_group)

if my_rank == 0:
    print(f"Running distributed test on {group_size} ranks.")

# init distributed SHT:
harmonics.distributed.init(mp_group)
    
# everything is awesome on GPUs
device = torch.device(f"cuda:{local_rank}")

# create a batch with one sample and 21 channels
b, c, n_theta, n_lambda = 1, 21, 360, 720

# your layers to play with
forward_transform = harmonics.RealSHT(n_theta, n_lambda).to(device)
inverse_transform = harmonics.InverseRealSHT(n_theta, n_lambda).to(device)
forward_transform_equi = harmonics.RealSHT(n_theta, n_lambda, grid="equiangular").to(device)
inverse_transform_equi = harmonics.InverseRealSHT(n_theta, n_lambda, grid="equiangular").to(device)

signal_leggauss = inverse_transform(torch.randn(b, c, n_theta // group_size, n_theta+1, device=device, dtype=torch.complex128))
signal_equi = inverse_transform(torch.randn(b, c, n_theta // group_size, n_theta+1, device=device, dtype=torch.complex128))

# let's check the layers
for num_iters in [1, 8, 64, 512]:
    base = signal_leggauss
    for iteration in tqdm(range(num_iters), disable=(my_rank!=0)):
        base = inverse_transform(forward_transform(base))

    # compute error:
    numerator = torch.sum(torch.square(torch.abs(base-signal_leggauss)), dim=(-1,-2))
    denominator = torch.sum(torch.square(torch.abs(signal_leggauss)), dim=(-1,-2))
    if dist.is_initialized():
        dist.all_reduce(numerator, group=mp_group)
        dist.all_reduce(denominator, group=mp_group)
    if my_rank == 0:
        print("relative l2 error accumulation on the legendre-gauss grid: ",
              torch.mean(torch.sqrt(numerator / denominator)).item(),
              "after", num_iters, "iterations")

# let's check the equiangular layers
for num_iters in [1, 8, 64, 512]:
    base = signal_equi
    for iteration in tqdm(range(num_iters), disable=(my_rank!=0)):
        base = inverse_transform_equi(forward_transform_equi(base))

    # compute error
    numerator = torch.sum(torch.square(torch.abs(base-signal_equi)), dim=(-1,-2))
    denominator = torch.sum(torch.square(torch.abs(signal_equi)), dim=(-1,-2))
    if dist.is_initialized():
        dist.all_reduce(numerator, group=mp_group)
        dist.all_reduce(denominator, group=mp_group)
    if my_rank == 0:
        print("relative l2 error accumulation with interpolation onto equiangular grid: ",
              torch.mean(torch.sqrt(numerator / denominator)).item(),
              "after", num_iters, "iterations")
