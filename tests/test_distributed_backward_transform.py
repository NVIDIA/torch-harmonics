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
import torch.nn.functional as F
import torch.distributed as dist
import torch_harmonics as harmonics
import torch_harmonics.distributed as thd

try:
    from tqdm import tqdm
except:
    tqdm = lambda x : x

# set up distributed
world_rank = int(os.getenv('WORLD_RANK', 0))
grid_size_h = int(os.getenv('GRID_H', 1))
grid_size_w = int(os.getenv('GRID_W', 1))
port = int(os.getenv('MASTER_PORT', 0))
master_address = os.getenv('MASTER_ADDR', 'localhost')
world_size = grid_size_h * grid_size_w
dist.init_process_group(backend = 'nccl',
                        init_method = f"tcp://{master_address}:{port}",
                        rank = world_rank,
                        world_size = world_size)
local_rank = world_rank % torch.cuda.device_count()
device = torch.device(f"cuda:{local_rank}")
# compute local ranks in h and w:
# rank = wrank + grid_size_w * hrank
wrank = world_rank % grid_size_w
hrank = world_rank // grid_size_w
w_group = None
h_group = None

# now set up the comm grid:
wgroups = []
for h in range(grid_size_h):
    start = h
    end = h + grid_size_w
    wgroups.append(list(range(start, end)))

print(wgroups)
for grp in wgroups:
    if len(grp) == 1:
        continue
    tmp_group = dist.new_group(ranks=grp)
    if wrank in grp:
        w_group = tmp_group

# transpose:
hgroups = [sorted(list(i)) for i in zip(*wgroups)]
print(hgroups)
for grp in hgroups:
    if len(grp) == 1:
        continue
    tmp_group = dist.new_group(ranks=grp)
    if hrank in	grp:
        h_group = tmp_group
        
# set device
torch.cuda.set_device(device.index)

# set seed
torch.manual_seed(333)
torch.cuda.manual_seed(333)

if world_rank == 0:
    print(f"Running distributed test on grid H x W = {grid_size_h} x {grid_size_w}")

# initializing sht
thd.init(h_group, w_group)

# common parameters
B, C, H, W = 1, 8, 721, 1440
Hloc = (H + grid_size_h - 1) // grid_size_h
Wloc = (W + grid_size_w - 1) // grid_size_w
Hpad = grid_size_h * Hloc - H
Wpad = grid_size_w * Wloc - W

# do serial tests first:
forward_transform_local = harmonics.RealSHT(nlat=H, nlon=W).to(device)
backward_transform_local = harmonics.InverseRealSHT(nlat=H, nlon=W).to(device)
backward_transform_dist = thd.DistributedInverseRealSHT(nlat=H, nlon=W).to(device)
Lpad = backward_transform_dist.lpad
Mpad = backward_transform_dist.mpad
Lloc = (Lpad + backward_transform_dist.lmax) // grid_size_h
Mloc = (Mpad + backward_transform_dist.mmax) // grid_size_w

# create tensors
dummy_full = torch.randn((B, C, H, W), dtype=torch.float32, device=device)
inp_full = forward_transform_local(dummy_full)

# pad
with torch.no_grad():
    inp_pad = F.pad(inp_full, (0, Mpad, 0, Lpad))

    # split in W
    inp_local = torch.split(inp_pad, split_size_or_sections=Mloc, dim=-1)[wrank]

    # split in H
    inp_local = torch.split(inp_local, split_size_or_sections=Lloc, dim=-2)[hrank]

# do FWD transform
out_full = backward_transform_local(inp_full)
out_local = backward_transform_dist(inp_local)

# gather the local data
# gather in W
if grid_size_w > 1:
    olist = [torch.empty_like(out_local) for _ in range(grid_size_w)]
    olist[wrank] = out_local
    dist.all_gather(olist, out_local, group=w_group)
    out_full_gather = torch.cat(olist, dim=-1)
    out_full_gather = out_full_gather[..., :W]
else:
    out_full_gather = out_local

# gather in h
if grid_size_h > 1:
    olist = [torch.empty_like(out_full_gather) for _ in range(grid_size_h)]
    olist[hrank] = out_full_gather
    dist.all_gather(olist, out_full_gather, group=h_group)
    out_full_gather = torch.cat(olist, dim=-2)
    out_full_gather = out_full_gather[..., :H, :]


if world_rank == 0:
    print(f"Local Out: sum={out_full.abs().sum().item()}, max={out_full.abs().max().item()}, min={out_full.abs().min().item()}")
    print(f"Dist Out: sum={out_full_gather.abs().sum().item()}, max={out_full_gather.abs().max().item()}, min={out_full_gather.abs().min().item()}")
    diff = (out_full-out_full_gather).abs()
    print(f"Out Difference: abs={diff.sum().item()}, rel={diff.sum().item() / (0.5*(out_full.abs().sum() + out_full_gather.abs().sum()))}, max={diff.abs().max().item()}")
    print("")

    
# create split input grad
with torch.no_grad():
    # create full grad
    ograd_full = torch.randn_like(out_full)

    # pad
    ograd_pad = F.pad(ograd_full, [0, Wpad, 0, Hpad])

    # split in W
    ograd_local = torch.split(ograd_pad, split_size_or_sections=Wloc, dim=-1)[wrank]

    # split in H
    ograd_local = torch.split(ograd_local, split_size_or_sections=Hloc, dim=-2)[hrank]


# backward pass:
# local
inp_full.requires_grad = True
out_full = backward_transform_local(inp_full)
out_full.backward(ograd_full)
igrad_full = inp_full.grad.clone()

# distributed
inp_local.requires_grad = True
out_local = backward_transform_dist(inp_local)
out_local.backward(ograd_local)
igrad_local = inp_local.grad.clone()

# gather
# gather in W
if grid_size_w > 1:
    olist = [torch.empty_like(igrad_local) for _ in range(grid_size_w)]
    olist[wrank] = igrad_local
    dist.all_gather(olist, igrad_local, group=w_group)
    igrad_full_gather = torch.cat(olist, dim=-1)
    igrad_full_gather = igrad_full_gather[..., :backward_transform_dist.mmax]
else:
    igrad_full_gather = igrad_local

# gather in h
if grid_size_h > 1:
    olist = [torch.empty_like(igrad_full_gather) for _ in range(grid_size_h)]
    olist[hrank] = igrad_full_gather
    dist.all_gather(olist, igrad_full_gather, group=h_group)
    igrad_full_gather = torch.cat(olist, dim=-2)
    igrad_full_gather = igrad_full_gather[..., :backward_transform_dist.lmax, :]

if world_rank == 0:
    print(f"Local Grad: sum={igrad_full.abs().sum().item()}, max={igrad_full.abs().max().item()}, min={igrad_full.abs().min().item()}")
    print(f"Dist Grad: sum={igrad_full_gather.abs().sum().item()}, max={igrad_full_gather.abs().max().item()}, min={igrad_full_gather.abs().min().item()}")
    diff = (igrad_full-igrad_full_gather).abs()
    print(f"Grad Difference: abs={diff.sum().item()}, rel={diff.sum().item() / (0.5*(igrad_full.abs().sum() + igrad_full_gather.abs().sum()))}, max={diff.abs().max().item()}")
