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

import os
import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

from torch_harmonics.quadrature import legendre_gauss_weights, lobatto_weights, clenshaw_curtiss_weights
from torch_harmonics.legendre import _precompute_legpoly, _precompute_dlegpoly
from torch_harmonics.distributed import polar_group_size, azimuth_group_size, distributed_transpose_azimuth, distributed_transpose_polar
from torch_harmonics.distributed import polar_group_rank, azimuth_group_rank
from torch_harmonics.distributed import compute_split_shapes, split_tensor_along_dim


class DistributedRealSHT(nn.Module):
    """
    Defines a module for computing the forward (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last two dimensions of the input

    Parameters
    ----------
    nlat: int
        Number of latitude points
    nlon: int
        Number of longitude points
    lmax: int
        Maximum spherical harmonic degree
    mmax: int
        Maximum spherical harmonic order
    grid: str
        Grid type ("equiangular", "legendre-gauss", "lobatto", "equidistant"), by default "equiangular"
    norm: str
        Normalization type ("ortho", "schmidt", "unnorm"), by default "ortho"
    csphase: bool
        Whether to apply the Condon-Shortley phase factor, by default True

    Returns
    -------
    x: torch.Tensor
        Tensor of shape (..., lmax, mmax)

    References
    ----------
    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="equiangular", norm="ortho", csphase=True):
        
        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # TODO: include assertions regarding the dimensions

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, weights = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, weights = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, weights = clenshaw_curtiss_weights(nlat, -1, 1)
            # cost, w = fejer2_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise(ValueError("Unknown quadrature mode"))

        # get the comms grid:
        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # apply cosine transform and flip them
        tq = torch.flip(torch.arccos(cost), dims=(0,))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1
        self.mmax_local = compute_split_shapes(self.mmax, self.comm_size_azimuth)[self.comm_rank_azimuth]

        # compute splits
        self.lat_shapes = compute_split_shapes(self.nlat, self.comm_size_polar)
        self.lon_shapes = compute_split_shapes(self.nlon, self.comm_size_azimuth)
        self.l_shapes = compute_split_shapes(self.lmax, self.comm_size_polar)
        self.m_shapes = compute_split_shapes(self.mmax, self.comm_size_azimuth)

        # combine quadrature weights with the legendre weights
        pct = _precompute_legpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)
        weights = torch.einsum('mlk,k->mlk', pct, weights)

        # split weights
        weights = split_tensor_along_dim(weights, dim=0, num_chunks=self.comm_size_azimuth)[self.comm_rank_azimuth].contiguous()

        # remember quadrature weights
        self.register_buffer('weights', weights, persistent=False)

    def extra_repr(self):
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        if x.dim() < 3:
            raise ValueError(f"Expected tensor with at least 3 dimensions but got {x.dim()} instead")

        # we need to ensure that we can split the channels evenly
        num_chans = x.shape[-3]

        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_azimuth > 1:
            x = distributed_transpose_azimuth.apply(x, (-3, -1), self.lon_shapes)

        # apply real fft in the longitudinal direction: make sure to truncate to nlon
        x = 2.0 * torch.pi * torch.fft.rfft(x, n=self.nlon, dim=-1, norm="forward")

        # truncate
        x = x[..., :self.mmax]

        # transpose: after this, m is split and c is local
        if self.comm_size_azimuth > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = distributed_transpose_azimuth.apply(x, (-1, -3), chan_shapes)

        # transpose: after this, c is split and h is local
        if self.comm_size_polar > 1:
            x = distributed_transpose_polar.apply(x, (-3, -2), self.lat_shapes)

        # do the Legendre-Gauss quadrature
        x = torch.view_as_real(x)

        # contraction
        out_shape = list(x.size())
        out_shape[-3] = self.lmax
        out_shape[-2] = self.mmax_local
        xs = torch.zeros(out_shape, dtype=x.dtype, device=x.device)
        xs[..., 0] = torch.einsum("...km,mlk->...lm", x[..., : self.mmax_local, 0], self.weights.to(x.dtype))
        xs[..., 1] = torch.einsum("...km,mlk->...lm", x[..., : self.mmax_local, 1], self.weights.to(x.dtype))

        # cast to complex
        x = torch.view_as_complex(xs)

        # transpose: after this, l is split and c is local
        if self.comm_size_polar	> 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_polar)
            x = distributed_transpose_polar.apply(x, (-2, -3), chan_shapes)

        return x


class DistributedInverseRealSHT(nn.Module):
    """
    Defines a module for computing the inverse (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.

    Parameters
    ----------
    nlat: int
        Number of latitude points
    nlon: int
        Number of longitude points
    lmax: int
        Maximum spherical harmonic degree
    mmax: int
        Maximum spherical harmonic order
    grid: str
        Grid type ("equiangular", "legendre-gauss", "lobatto", "equidistant"), by default "equiangular"
    norm: str
        Normalization type ("ortho", "schmidt", "unnorm"), by default "ortho"
    csphase: bool
        Whether to apply the Condon-Shortley phase factor, by default True

    Returns
    -------
    x: torch.Tensor
        Tensor of shape (..., lmax, mmax)

    References
    ----------
    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="equiangular", norm="ortho", csphase=True):

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, _ = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, _ = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, _ = clenshaw_curtiss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise(ValueError("Unknown quadrature mode"))

        # get the comms grid:
        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # apply cosine transform and flip them
        t = torch.flip(torch.arccos(cost), dims=(0,))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        # compute splits
        self.lat_shapes = compute_split_shapes(self.nlat, self.comm_size_polar)
        self.lon_shapes = compute_split_shapes(self.nlon, self.comm_size_azimuth)
        self.l_shapes = compute_split_shapes(self.lmax, self.comm_size_polar)
        self.m_shapes = compute_split_shapes(self.mmax, self.comm_size_azimuth)

        # compute legende polynomials
        pct = _precompute_legpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase)

        # split in m
        pct = split_tensor_along_dim(pct, dim=0, num_chunks=self.comm_size_azimuth)[self.comm_rank_azimuth].contiguous()

        # register
        self.register_buffer('pct', pct, persistent=False)

    def extra_repr(self):
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        if x.dim() < 3:
            raise ValueError(f"Expected tensor with at least 3 dimensions but got {x.dim()} instead")

        # we need to ensure that we can split the channels evenly
        num_chans = x.shape[-3]

        # transpose: after that, channels are split, l is local:
        if self.comm_size_polar > 1:
            x = distributed_transpose_polar.apply(x, (-3, -2), self.l_shapes)

        # Evaluate associated Legendre functions on the output nodes
        x = torch.view_as_real(x)

        # einsum
        xs = torch.einsum('...lmr, mlk->...kmr', x, self.pct.to(x.dtype)).contiguous()

        # inverse FFT
        x = torch.view_as_complex(xs)

        if self.comm_size_polar > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_polar)
            x = distributed_transpose_polar.apply(x, (-2, -3), chan_shapes)

        # transpose: after this, channels are split and m is local
        if self.comm_size_azimuth > 1:
            x = distributed_transpose_azimuth.apply(x, (-3, -1), self.m_shapes)

        # set DCT and nyquist frequencies to 0:
        x[..., 0].imag = 0.0
        if (self.nlon % 2 == 0) and (self.nlon // 2 < x.shape[-1]):
            x[..., self.nlon // 2].imag = 0.0
            
        # apply the inverse (real) FFT
        x = torch.fft.irfft(x, n=self.nlon, dim=-1, norm="forward")

        # transpose: after this, m is split and channels are local
        if self.comm_size_azimuth > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = distributed_transpose_azimuth.apply(x, (-1, -3), chan_shapes)

        return x


class DistributedRealVectorSHT(nn.Module):
    """
    Defines a module for computing the forward (real) vector SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last three dimensions of the input.

    Parameters
    ----------
    nlat: int
        Number of latitude points
    nlon: int
        Number of longitude points  
    lmax: int
        Maximum spherical harmonic degree
    mmax: int
        Maximum spherical harmonic order
    grid: str
        Grid type ("equiangular", "legendre-gauss", "lobatto", "equidistant"), by default "equiangular"
    norm: str
        Normalization type ("ortho", "schmidt", "unnorm"), by default "ortho"
    csphase: bool
        Whether to apply the Condon-Shortley phase factor, by default True

    Returns
    -------
    x: torch.Tensor
        Tensor of shape (..., lmax, mmax)

    References
    ----------
    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="equiangular", norm="ortho", csphase=True):

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, weights = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, weights = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, weights = clenshaw_curtiss_weights(nlat, -1, 1)
            # cost, w = fejer2_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise(ValueError("Unknown quadrature mode"))

        # get the comms grid:
        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # apply cosine transform and flip them
        tq = torch.flip(torch.arccos(cost), dims=(0,))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        # compute splits
        self.lat_shapes = compute_split_shapes(self.nlat, self.comm_size_polar)
        self.lon_shapes = compute_split_shapes(self.nlon, self.comm_size_azimuth)
        self.l_shapes = compute_split_shapes(self.lmax, self.comm_size_polar)
        self.m_shapes = compute_split_shapes(self.mmax, self.comm_size_azimuth)

        # compute weights
        dpct = _precompute_dlegpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)

        # combine integration weights, normalization factor in to one:
        l = torch.arange(0, self.lmax)
        norm_factor = 1. / l / (l+1)
        norm_factor[0] = 1.
        weights = torch.einsum('dmlk,k,l->dmlk', dpct, weights, norm_factor)
        # since the second component is imaginary, we need to take complex conjugation into account
        weights[1] = -1 * weights[1]

        # we need to split in m, pad before:
        weights = split_tensor_along_dim(weights, dim=1, num_chunks=self.comm_size_azimuth)[self.comm_rank_azimuth].contiguous()

        # remember quadrature weights
        self.register_buffer('weights', weights, persistent=False)


    def extra_repr(self):
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        if x.dim() < 4:
            raise ValueError(f"Expected tensor with at least 4 dimensions but got {x.dim()} instead")

        # we need to ensure that we can split the channels evenly
        num_chans = x.shape[-4]

        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_azimuth > 1:
            x = distributed_transpose_azimuth.apply(x, (-4, -1), self.lon_shapes)

        # apply real fft in the longitudinal direction: make sure to truncate to nlon
        x = 2.0 * torch.pi * torch.fft.rfft(x, n=self.nlon, dim=-1, norm="forward")

        # truncate
        x = x[..., :self.mmax]

        # transpose: after this, m is split and c is local
        if self.comm_size_azimuth > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = distributed_transpose_azimuth.apply(x, (-1, -4), chan_shapes)

        # transpose: after this, c is split and h is local
        if self.comm_size_polar > 1:
            x = distributed_transpose_polar.apply(x, (-4, -2), self.lat_shapes)

        # do the Legendre-Gauss quadrature
        x = torch.view_as_real(x)

        # create output array
        xs = torch.zeros_like(x, dtype=x.dtype, device=x.device)

        # contraction - spheroidal component
        # real component
        xs[..., 0, :, :, 0] =   torch.einsum('...km,mlk->...lm', x[..., 0, :, :, 0], self.weights[0].to(xs.dtype)) \
                              - torch.einsum('...km,mlk->...lm', x[..., 1, :, :, 1], self.weights[1].to(xs.dtype))
        # imag component
        xs[..., 0, :, :, 1] =   torch.einsum('...km,mlk->...lm', x[..., 0, :, :, 1], self.weights[0].to(xs.dtype)) \
                              + torch.einsum('...km,mlk->...lm', x[..., 1, :, :, 0], self.weights[1].to(xs.dtype))

        # contraction - toroidal component
        # real component
        xs[..., 1, :, :, 0] = - torch.einsum('...km,mlk->...lm', x[..., 0, :, :, 1], self.weights[1].to(xs.dtype)) \
                              - torch.einsum('...km,mlk->...lm', x[..., 1, :, :, 0], self.weights[0].to(xs.dtype))
        # imag component
        xs[..., 1, :, :, 1] =   torch.einsum('...km,mlk->...lm', x[..., 0, :, :, 0], self.weights[1].to(xs.dtype)) \
                              - torch.einsum('...km,mlk->...lm', x[..., 1, :, :, 1], self.weights[0].to(xs.dtype))

        # pad if required
        x = torch.view_as_complex(xs)

        # transpose: after this, l is split and c is local
        if self.comm_size_polar > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_polar)
            x = distributed_transpose_polar.apply(x, (-2, -4), chan_shapes)

        return x


class DistributedInverseRealVectorSHT(nn.Module):
    """
    Defines a module for computing the inverse (real-valued) vector SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.

    Parameters
    ----------
    nlat: int
        Number of latitude points
    nlon: int
        Number of longitude points
    lmax: int
        Maximum spherical harmonic degree
    mmax: int
        Maximum spherical harmonic order
    grid: str
        Grid type ("equiangular", "legendre-gauss", "lobatto", "equidistant"), by default "equiangular"
    norm: str
        Normalization type ("ortho", "schmidt", "unnorm"), by default "ortho"
    csphase: bool
        Whether to apply the Condon-Shortley phase factor, by default True

    Returns
    -------
    x: torch.Tensor
        Tensor of shape (..., lmax, mmax)

    References
    ----------
    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """
    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="equiangular", norm="ortho", csphase=True):

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, _ = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, _ = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, _ = clenshaw_curtiss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise(ValueError("Unknown quadrature mode"))

        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # apply cosine transform and flip them
        t = torch.flip(torch.arccos(cost), dims=(0,))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        # compute splits
        self.lat_shapes = compute_split_shapes(self.nlat, self.comm_size_polar)
        self.lon_shapes = compute_split_shapes(self.nlon, self.comm_size_azimuth)
        self.l_shapes = compute_split_shapes(self.lmax, self.comm_size_polar)
        self.m_shapes = compute_split_shapes(self.mmax, self.comm_size_azimuth)

        # compute legende polynomials
        dpct = _precompute_dlegpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase)

        # split in m
        dpct = split_tensor_along_dim(dpct, dim=1, num_chunks=self.comm_size_azimuth)[self.comm_rank_azimuth].contiguous()

        # register buffer
        self.register_buffer('dpct', dpct, persistent=False)

    def extra_repr(self):
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        if x.dim() < 4:
            raise ValueError(f"Expected tensor with at least 4 dimensions but got {x.dim()} instead")

        # store num channels
        num_chans = x.shape[-4]

        # transpose: after that, channels are split, l is local:
        if self.comm_size_polar > 1:
            x = distributed_transpose_polar.apply(x, (-4, -2), self.l_shapes)

        # Evaluate associated Legendre functions on the output nodes
        x = torch.view_as_real(x)

        # contraction - spheroidal component
        # real component
        srl =   torch.einsum('...lm,mlk->...km', x[..., 0, :, :, 0], self.dpct[0].to(x.dtype)) \
              - torch.einsum('...lm,mlk->...km', x[..., 1, :, :, 1], self.dpct[1].to(x.dtype))
        # imag component
        sim =   torch.einsum('...lm,mlk->...km', x[..., 0, :, :, 1], self.dpct[0].to(x.dtype)) \
              + torch.einsum('...lm,mlk->...km', x[..., 1, :, :, 0], self.dpct[1].to(x.dtype))

        # contraction - toroidal component
        # real component
        trl = - torch.einsum('...lm,mlk->...km', x[..., 0, :, :, 1], self.dpct[1].to(x.dtype)) \
              - torch.einsum('...lm,mlk->...km', x[..., 1, :, :, 0], self.dpct[0].to(x.dtype))
        # imag component
        tim =   torch.einsum('...lm,mlk->...km', x[..., 0, :, :, 0], self.dpct[1].to(x.dtype)) \
              - torch.einsum('...lm,mlk->...km', x[..., 1, :, :, 1], self.dpct[0].to(x.dtype))

        # reassemble
        s = torch.stack((srl, sim), -1)
        t = torch.stack((trl, tim), -1)
        xs = torch.stack((s, t), -4)

        # convert to complex
        x = torch.view_as_complex(xs)

        if self.comm_size_polar > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_polar)
            x = distributed_transpose_polar.apply(x, (-2, -4), chan_shapes)

        # transpose: after this, channels are split and m is local
        if self.comm_size_azimuth > 1:
            x = distributed_transpose_azimuth.apply(x, (-4, -1), self.m_shapes)

        # set DCT and nyquist frequencies to zero
        x[..., 0].imag = 0.0
        if (self.nlon % 2 == 0) and (self.nlon // 2 < x.shape[-1]):
            x[..., self.nlon // 2].imag = 0.0

        # apply the inverse (real) FFT
        x = torch.fft.irfft(x, n=self.nlon, dim=-1, norm="forward")

        # transpose: after this, m is split and channels are local
        if self.comm_size_azimuth > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = distributed_transpose_azimuth.apply(x, (-1, -4), chan_shapes)

        return x
