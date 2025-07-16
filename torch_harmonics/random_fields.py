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
from .sht import InverseRealSHT

class GaussianRandomFieldS2(torch.nn.Module):
    def __init__(self, nlat, alpha=2.0, tau=3.0, sigma=None, radius=1.0, grid="equiangular", dtype=torch.float32):
        super().__init__()

        #Number of latitudinal modes.
        self.nlat = nlat

        #Default value of sigma if None is given.
        if sigma is None:
            assert alpha > 1.0, f"Alpha must be greater than one, got {alpha}."
            sigma = tau**(0.5*(2*alpha - 2.0))

        # Inverse SHT
        self.isht = InverseRealSHT(self.nlat, 2*self.nlat, grid=grid, norm='backward').to(dtype=dtype)

        #Square root of the eigenvalues of C.
        sqrt_eig = torch.tensor([j*(j+1) for j in range(self.nlat)]).view(self.nlat,1).repeat(1, self.nlat+1)
        sqrt_eig = torch.tril(sigma*(((sqrt_eig/radius**2) + tau**2)**(-alpha/2.0)))
        sqrt_eig[0,0] = 0.0
        sqrt_eig = sqrt_eig.unsqueeze(0)
        self.register_buffer('sqrt_eig', sqrt_eig)

        #Save mean and var of the standard Gaussian.
        #Need these to re-initialize distribution on a new device.
        mean = torch.tensor([0.0]).to(dtype=dtype)
        var = torch.tensor([1.0]).to(dtype=dtype)
        self.register_buffer('mean', mean)
        self.register_buffer('var', var)

        #Standard normal noise sampler.
        self.gaussian_noise = torch.distributions.normal.Normal(self.mean, self.var)

    def forward(self, N, xi=None):

        #Sample Gaussian noise.
        if xi is None:
            xi = self.gaussian_noise.sample(torch.Size((N, self.nlat, self.nlat + 1, 2))).squeeze()
            xi = torch.view_as_complex(xi)
        
        #Karhunen-Loeve expansion.
        u = self.isht(xi*self.sqrt_eig)
        
        return u
    
    #Override cuda and to methods so sampler gets initialized with mean
    #and variance on the correct device.
    def cuda(self, *args, **kwargs):
        super().cuda(*args, **kwargs)
        self.gaussian_noise = torch.distributions.normal.Normal(self.mean, self.var)

        return self
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.gaussian_noise = torch.distributions.normal.Normal(self.mean, self.var)

        return self
