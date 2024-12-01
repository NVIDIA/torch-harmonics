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
import torch.nn as nn
import torch.nn.functional as F

from scipy.special import lpmv
import numpy as np

class SHT_DSE():
    r"""
    Defines a module for computing the forward/backward SHT on arbitrary points.
    Requires the collocation points (locations of measurements on the surface of the sphere),
    as defined by the polar (theta) and azimuthal (phi) angles.
    The SHT is applied to the last two dimensions of the input. 

    [1] L. Lingsch, M. Michelis, E. de Bezenac, S. M. Perera, R. K. Katzschmann, S. Mishra; 
    Beyond Regular Grids: Fourier-Based Neural Operators on Arbitrary Domains; ICML 2024
    """
    
    def __init__(self, phi, theta, degree):        
        r"""
        Initializes the matrices to compute the forward/backward SHT on arbitrary points.

        Parameters:
        phi: input point locations as a azimuthal angle
        theta: input grid locations as a polar angle
        degree: degree of the spherical harmonics, total number of modes equal to degree^2
        """
        self.theta = theta  # between 0 and pi
        self.phi = phi      # between 0 and 2 pi

        self.degree = degree

        self.num_points = theta.shape[0]

        self.V_fwd, self.V_inv = self.make_matrix()

    def make_matrix(self):       
        r"""
        Constructs the matrices to compute spherical harmonics transforms

        Inputs: 
        class variables
        Outputs:
        V_fwd computes the forward transform via matrix multiplication
        V_inv computes the inverse transform via matrix multiplication
        """
        V_forward = torch.zeros((self.num_points, self.degree ** 2), dtype=torch.float)
        index = 0
        for l in range(self.degree):
             for m in range(-l, l+1):
                if index > 0:
                    c = np.sqrt(2)
                else:
                    c = 1
                if m < 0:
                    V_forward[:, index] = (lpmv(m, l, torch.cos(self.theta)) * torch.sin(m*self.phi))
                    V_forward[:,index] =  c * V_forward[:,index] / torch.max( V_forward[:,index])
                else:
                    V_forward[:, index] = (lpmv(m, l, torch.cos(self.theta)) * torch.cos(m*self.phi))
                    V_forward[:,index] =  c * V_forward[:,index] / torch.max( V_forward[:,index])
                index += 1
        
        return V_forward.cuda(), torch.transpose(V_forward, 0, 1).cuda()

    def forward(self, data):
        r"""
        Computes the spherical harmonics from the data

        Inputs: 
        class variables
        data; vector of inputs in spatial domain
        Outputs:
        data_fwd; data in spherical harmonic domain up to set degree
        """
        data_fwd = torch.matmul(data, self.V_fwd)

        return data_fwd
    
    def inverse(self, data):
        r"""
        Computes the modified data from the spherical harmonics
        Note: This is not technically an inverse, as orthogonality is not preserved.
            Nonetheless, we refer to it as such.

        Inputs: 
        class variables
        data; vector of inputs in spherical harmonics domain
        Outputs:
        data_inv; data in spatial domain
        """
        data_inv = torch.matmul(data, self.V_inv) / self.num_points
        
        return data_inv


##################################################
# Classes for data with fixed collocation points
##################################################

class SHT_DSE_Conv_fp(nn.Module):
    r"""
    Defines a module for computing the convolution in the Spherical Harmonic Domain. 
    This is computed as a vector-vector multiplication between a vector of spherical harmonic
    coefficients and a vector of learnable weights.

    """
    def __init__(self, in_channels, out_channels, degree, sht_transform):
        r"""
        Initializes the SHT Convolution Layer, 

        Parameters:
        in_channels: number of channels to be taken as input for the SH convolution
        out_channels: number of channels produced by the spherical harmonics convolution
        degree: degree of the spherical harmonics, total number of modes equal to degree^2
        sht_transform: computes the SH transform itself
        """
        super(SHT_DSE_Conv_fp, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degree = degree 
        
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.degree**2, dtype=torch.float))

        self.sht_transform = sht_transform

    def compl_mul1d(self, input, weights):
        r"""
        Computes the SH via multiplication in SH domain

        Inputs:
        input; SH data
        weights; trainable weights for convolution in SH domain
        Outputs:
        data convolved via multiplication in SH domain
        """
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        r"""
        Computes the forward pass of the SH Convolution layer

        Inputs:
        x; data in spatial domain
        Outputs:
        x; data convolved in SH domain with learnable weights
        """
        # Calculate SH for given data
        x_ft = self.sht_transform.forward(x)

        # Multiply relevant SH modes
        out_ft = self.compl_mul1d(x_ft, self.weights1)

        # Return to physical space
        x = self.sht_transform.inverse(out_ft)

        return x


class SFNO_dse_fp(nn.Module):    
    r"""
    Defines a module for training a SFNO on FIXED arbitrary collocation (measurement) points.

    [1] L. Lingsch, M. Michelis, E. de Bezenac, S. M. Perera, R. K. Katzschmann, S. Mishra; 
    Beyond Regular Grids: Fourier-Based Neural Operators on Arbitrary Domains; ICML 2024
    """
    def __init__(self, in_channels, out_channels, degree, width, sht_transform, num_layers):
        r"""
        Initializes the class to learn the SFNO. 

        Parameters:
        in_channels: number of channels of the input data
        out_channels: number of channels for the output data (to be learned)
        degree: degree of the spherical harmonics, total number of modes equal to degree^2
        sht_transform: computes the SH transform itself
        num_layers: number of trainable SFNO layers (SH convolution + Pointwise convolution)
        """
        super(SFNO_dse_fp, self).__init__()
        
        self.degree = degree
        self.width = width
        self.num_layers = num_layers
        
        # Input layer
        self.fc0 = nn.Linear(in_channels, self.width)
        
        # Dynamically create convolutional and pointwise linear layers
        self.conv_layers = nn.ModuleList([
            SHT_DSE_Conv_fp(self.width, self.width, self.degree, sht_transform)
            for _ in range(num_layers)
        ])
        
        self.pointwise_layers = nn.ModuleList([
            nn.Conv1d(self.width, self.width, 1)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        r"""
        Learns to predict an output as a function of the input data via SH

        Inputs: 
        class variables
        x; vector of inputs in spatial domain with dimensions [batchsize, in_channels, N] 
        Outputs:
        x; vector of outputs in spatial domain with dimensions [batchsize, out_channels, N] 
        """
        x = x.permute(0, 2, 1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        # Apply dynamically created layers
        for i in range(self.num_layers):
            x1 = self.conv_layers[i](x)
            x2 = self.pointwise_layers[i](x)
            x = x1 + x2
            x = F.gelu(x)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        return x


##################################################
# Classes for data with varying collocation points
##################################################

class SHT_DSE_Conv_vp(nn.Module):
    r"""
    Defines a module for computing the convolution in the Spherical Harmonic Domain. 
    This is computed as a vector-vector multiplication between a vector of spherical harmonic
    coefficients and a vector of learnable weights.

    """
    def __init__(self, in_channels, out_channels, degree):
        r"""
        Initializes the SHT Convolution Layer, 

        Parameters:
        in_channels: number of channels to be taken as input for the SH convolution
        out_channels: number of channels produced by the spherical harmonics convolution
        degree: degree of the spherical harmonics, total number of modes equal to degree^2
        sht_transform: computes the SH transform itself
        """
        super(SHT_DSE_Conv_vp, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degree = degree 
        
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.degree**2, dtype=torch.float))


    def compl_mul1d(self, input, weights):
        r"""
        Computes the SH via multiplication in SH domain

        Inputs:
        input; SH data
        weights; trainable weights for convolution in SH domain
        Outputs:
        data convolved via multiplication in SH domain
        """
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x, sht_transform):
        r"""
        Computes the forward pass of the SH Convolution layer

        Inputs:
        x; data in spatial domain
        Outputs:
        x; data convolved in SH domain with learnable weights
        """
        # Calculate SH for given data
        x_ft = sht_transform.forward(x)

        # Multiply relevant SH modes
        out_ft = self.compl_mul1d(x_ft, self.weights1)

        # Return to physical space
        x = sht_transform.inverse(out_ft)

        return x
        
class SFNO_dse_vp(nn.Module):    
    r"""
    Defines a module for training a SFNO on VARIABLE arbitrary collocation (measurement) points.

    [1] L. Lingsch, M. Michelis, E. de Bezenac, S. M. Perera, R. K. Katzschmann, S. Mishra; 
    Beyond Regular Grids: Fourier-Based Neural Operators on Arbitrary Domains; ICML 2024
    """
    def __init__(self, in_channels, out_channels, degree, width, num_layers):
        r"""
        Initializes the class to learn the SFNO. 

        Parameters:
        in_channels: number of channels of the input data
        out_channels: number of channels for the output data (to be learned)
        degree: degree of the spherical harmonics, total number of modes equal to degree^2
        sht_transform: computes the SH transform itself
        num_layers: number of trainable SFNO layers (SH convolution + Pointwise convolution)
        """
        super(SFNO_dse_vp, self).__init__()
        
        self.degree = degree
        self.width = width
        self.num_layers = num_layers
        
        # Input layer
        self.fc0 = nn.Linear(in_channels, self.width)
        
        # Dynamically create convolutional and pointwise linear layers
        self.conv_layers = nn.ModuleList([
            SHT_DSE_Conv_vp(self.width, self.width, self.degree)
            for _ in range(num_layers)
        ])
        
        self.pointwise_layers = nn.ModuleList([
            nn.Conv1d(self.width, self.width, 1)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x, sht_transform):
        r"""
        Learns to predict an output as a function of the input data via SH

        Inputs: 
        class variables
        x; vector of inputs in spatial domain with dimensions [batchsize, in_channels, N] 
        Outputs:
        x; vector of outputs in spatial domain with dimensions [batchsize, out_channels, N] 
        """
        x = x.permute(0, 2, 1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        # Apply dynamically created layers
        for i in range(self.num_layers):
            x1 = self.conv_layers[i](x, sft_transform)
            x2 = self.pointwise_layers[i](x)
            x = x1 + x2
            x = F.gelu(x)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        return x