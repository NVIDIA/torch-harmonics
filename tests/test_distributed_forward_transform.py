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
from torch_harmonics.distributed.primitives import gather_from_parallel_region

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
device = torch.device(f"cuda:{local_rank}")

# set seed
torch.manual_seed(333)
torch.cuda.manual_seed(333)

if my_rank == 0:
    print(f"Running distributed test on {group_size} ranks.")

# common parameters
b, c, n_theta, n_lambda = 1, 21, 361, 720

# do serial tests first:
forward_transform = harmonics.RealSHT(n_theta, n_lambda).to(device)
inverse_transform = harmonics.InverseRealSHT(n_theta, n_lambda).to(device) 

# set up signal
with torch.no_grad():
    signal_leggauss = inverse_transform(torch.randn(b, c, forward_transform.lmax, forward_transform.mmax, device=device, dtype=torch.complex128))
    signal_leggauss_dist = signal_leggauss.clone()
signal_leggauss.requires_grad = True
signal_leggauss_dist.requires_grad = True

# do a fwd and bwd pass:
x_local = forward_transform(signal_leggauss)
loss = torch.sum(torch.view_as_real(x_local))
loss.backward()
x_local = torch.view_as_real(x_local)
local_grad = signal_leggauss.grad.clone()

# now the distributed test
harmonics.distributed.init(mp_group)
forward_transform_dist = harmonics.RealSHT(n_theta, n_lambda).to(device)
inverse_transform_dist = harmonics.InverseRealSHT(n_theta, n_lambda).to(device)

# do distributed sht
x_dist = forward_transform_dist(signal_leggauss_dist)
loss = torch.sum(torch.view_as_real(x_dist))
loss.backward()
x_dist = torch.view_as_real(x_dist)
dist_grad = signal_leggauss_dist.grad.clone()

# gather the output
x_dist = gather_from_parallel_region(x_dist, dim=2)

if my_rank == 0:
    print(f"Local Out: sum={x_local.abs().sum().item()}, max={x_local.max().item()}, min={x_local.min().item()}")
    print(f"Dist Out: sum={x_dist.abs().sum().item()}, max={x_dist.max().item()}, min={x_dist.min().item()}")
    diff = (x_local-x_dist).abs()
    print(f"Out Difference: abs={diff.sum().item()}, rel={diff.sum().item() / (0.5*(x_local.abs().sum() + x_dist.abs().sum()))}, max={diff.max().item()}")
    print("")
    print(f"Local Grad: sum={local_grad.abs().sum().item()}, max={local_grad.max().item()}, min={local_grad.min().item()}")
    print(f"Dist Grad: sum={dist_grad.abs().sum().item()}, max={dist_grad.max().item()}, min={dist_grad.min().item()}")
    diff = (local_grad-dist_grad).abs()
    print(f"Grad Difference: abs={diff.sum().item()}, rel={diff.sum().item() / (0.5*(local_grad.abs().sum() + dist_grad.abs().sum()))}, max={diff.max().item()}")
