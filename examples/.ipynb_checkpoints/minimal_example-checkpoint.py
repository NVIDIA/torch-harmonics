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
sys.path.append("..")
sys.path.append(".")

import torch
import torch_harmonics as harmonics

try:
    from tqdm import tqdm
except:
    tqdm = lambda x : x

# everything is awesome on GPUs
device = torch.device("cuda")

# create a batch with one sample and 21 channels
b, c, n_theta, n_lambda = 1, 21, 360, 720

# your layers to play with
forward_transform = harmonics.RealSHT(n_theta, n_lambda).to(device)
inverse_transform = harmonics.InverseRealSHT(n_theta, n_lambda).to(device)
forward_transform_equi = harmonics.RealSHT(n_theta, n_lambda, grid="equiangular").to(device)
inverse_transform_equi = harmonics.InverseRealSHT(n_theta, n_lambda, grid="equiangular").to(device)

signal_leggauss = inverse_transform(torch.randn(b, c, n_theta, n_theta+1, device=device, dtype=torch.complex128))
signal_equi = inverse_transform(torch.randn(b, c, n_theta, n_theta+1, device=device, dtype=torch.complex128))

# let's check the layers
for num_iters in [1, 8, 64, 512]:
    base = signal_leggauss
    for iteration in tqdm(range(num_iters)):
        base = inverse_transform(forward_transform(base))
    print("relative l2 error accumulation on the legendre-gauss grid: ",
          torch.mean(torch.norm(base-signal_leggauss, p='fro', dim=(-1,-2)) / torch.norm(signal_leggauss, p='fro', dim=(-1,-2)) ).item(),
          "after", num_iters, "iterations")

# let's check the equiangular layers
for num_iters in [1, 8, 64, 512]:
    base = signal_equi
    for iteration in tqdm(range(num_iters)):
        base = inverse_transform_equi(forward_transform_equi(base))
    print("relative l2 error accumulation with interpolation onto equiangular grid: ",
          torch.mean(torch.norm(base-signal_equi, p='fro', dim=(-1,-2)) / torch.norm(signal_equi, p='fro', dim=(-1,-2)) ).item(),
          "after", num_iters, "iterations")
