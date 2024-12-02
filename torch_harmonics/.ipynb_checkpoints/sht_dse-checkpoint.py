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

class RealSHTDSE():
    r"""
    Defines a module for computing the forward/backward SHT on arbitrary points.
    Requires the collocation points (locations of measurements on the surface of the sphere),
    as defined by the polar (theta) and azimuthal (phi) angles.
    The SHT is applied to the last two dimensions of the input. 

    [1] L. Lingsch, M. Michelis, E. de Bezenac, S. M. Perera, R. K. Katzschmann, S. Mishra; 
    Beyond Regular Grids: Fourier-Based Neural Operators on Arbitrary Domains; ICML 2024
    """
    
    def __init__(self, phi, theta, degree):        
        """
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
        """
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
        """
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
        """
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


class BatchedRealSHTDSE():
    r"""
    Defines a module for computing the forward/backward SHT on arbitrary points.
    Requires the collocation points (locations of measurements on the surface of the sphere),
    as defined by the polar (theta) and azimuthal (phi) angles.
    The SHT is applied to the last two dimensions of the input. 

    [1] L. Lingsch, M. Michelis, E. de Bezenac, S. M. Perera, R. K. Katzschmann, S. Mishra; 
    Beyond Regular Grids: Fourier-Based Neural Operators on Arbitrary Domains; ICML 2024
    """
    
    def __init__(self, phi, theta, degree):        
        """
        Initializes the matrices to compute the forward/backward SHT on arbitrary points.

        Parameters:
        phi: input point locations as a azimuthal angle
        theta: input grid locations as a polar angle
        degree: degree of the spherical harmonics, total number of modes equal to degree^2
        """
        self.theta = theta  # between 0 and pi
        self.phi = phi      # between 0 and 2 pi

        self.degree = degree
        
        self.batch_size = theta.shape[0]
        self.num_points = theta.shape[1]

        self.V_fwd, self.V_inv = self.make_matrix()

    
    def compute_legendre_matrix(self, l):
        """
        Compute all associated Legendre polynomials for degree `l` across the batch.
        Uses scipy.special.lpmv to generate values in a vectorized way.
        """
        theta_cos = torch.cos(self.theta)
        P_l_m = []
        for m in range(-l, l + 1):
            P_lm = lpmv(m, l, theta_cos.cpu().numpy())  # lpmv operates on numpy arrays
            P_l_m.append(torch.tensor(P_lm, dtype=torch.float, device=self.theta.device))
        return torch.stack(P_l_m, dim=0)  # Shape: (2l+1, num_points)

    def make_matrix(self):
        V_fwd = torch.zeros((self.batch_size, self.num_points, self.degree ** 2), dtype=torch.float, device=self.theta.device)

        index = 0

        for l in range(self.degree):
            P_l_m = self.compute_legendre_matrix(l)  # Shape: (2l+1, num_points)
            
            for m in range(-l, l + 1):
                trig_term = torch.sin(m * self.phi) if m < 0 else torch.cos(m * self.phi)
                scale_factor = np.sqrt(2) if m != 0 else 1.0
                
                V_fwd[:, :, index] = scale_factor * P_l_m[m + l, :] * trig_term
                V_fwd[:, :, index] /= torch.max(V_fwd[:, :, index]).clamp(min=1e-6)  # Avoid division by zero
                index += 1
                    


        return V_fwd.cuda(), V_fwd.permute(0,2,1).cuda()

    def forward(self, data):
        """
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
        """
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

