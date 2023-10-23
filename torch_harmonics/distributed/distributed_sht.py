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
import numpy as np
import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

from torch_harmonics.quadrature import legendre_gauss_weights, lobatto_weights, clenshaw_curtiss_weights
from torch_harmonics.legendre import _precompute_legpoly, _precompute_dlegpoly
from torch_harmonics.distributed import polar_group_size, azimuth_group_size, distributed_transpose_azimuth, distributed_transpose_polar
from torch_harmonics.distributed import polar_group_rank, azimuth_group_rank


class DistributedRealSHT(nn.Module):
    """
    Defines a module for computing the forward (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last two dimensions of the input

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True):
        """
        Initializes the SHT Layer, precomputing the necessary quadrature weights

        Parameters:
        nlat: input grid resolution in the latitudinal direction
        nlon: input grid resolution in the longitudinal direction
        grid: grid in the latitude direction (for now only tensor product grids are supported)
        """

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # TODO: include assertions regarding the dimensions

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, w = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, w = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, w = clenshaw_curtiss_weights(nlat, -1, 1)
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
        tq = np.flip(np.arccos(cost))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        # spatial paddings
        latdist = (self.nlat + self.comm_size_polar - 1) // self.comm_size_polar
        self.nlatpad = latdist * self.comm_size_polar - self.nlat
        londist = (self.nlon + self.comm_size_azimuth - 1) // self.comm_size_azimuth
        self.nlonpad = londist * self.comm_size_azimuth - self.nlon

        # frequency paddings
        ldist = (self.lmax + self.comm_size_polar - 1) // self.comm_size_polar
        self.lpad = ldist * self.comm_size_polar - self.lmax
        mdist = (self.mmax + self.comm_size_azimuth - 1) // self.comm_size_azimuth
        self.mpad = mdist * self.comm_size_azimuth - self.mmax

        # combine quadrature weights with the legendre weights
        weights = torch.from_numpy(w)
        pct = _precompute_legpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)
        pct = torch.from_numpy(pct)
        weights = torch.einsum('mlk,k->mlk', pct, weights)

        # we need to split in m, pad before:
        weights = F.pad(weights, [0, 0, 0, 0, 0, self.mpad], mode="constant")
        weights = torch.split(weights, (self.mmax+self.mpad) // self.comm_size_azimuth, dim=0)[self.comm_rank_azimuth]

        # compute the local pad and size
        # spatial
        self.nlat_local = min(latdist, self.nlat - self.comm_rank_polar * latdist)
        self.nlatpad_local = latdist - self.nlat_local
        self.nlon_local = min(londist, self.nlon - self.comm_rank_azimuth * londist)
        self.nlonpad_local = londist - self.nlon_local

        # frequency
        self.lmax_local = min(ldist, self.lmax - self.comm_rank_polar * ldist)
        self.lpad_local = ldist - self.lmax_local
        self.mmax_local	= min(mdist, self.mmax - self.comm_rank_azimuth * mdist)
        self.mpad_local	= mdist	- self.mmax_local

        # remember quadrature weights
        self.register_buffer('weights', weights, persistent=False)

    def extra_repr(self):
        """
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        # we need to ensure that we can split the channels evenly
        assert(x.shape[1] % self.comm_size_polar == 0)
        assert(x.shape[1] % self.comm_size_azimuth == 0)

        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_azimuth > 1:
            xt = distributed_transpose_azimuth.apply(x, (1, -1))
        else:
            xt = x

        # apply real fft in the longitudinal direction: make sure to truncate to nlon
        xtf = 2.0 * torch.pi * torch.fft.rfft(xt, n=self.nlon, dim=-1, norm="forward")

        # truncate
        xtft = xtf[..., :self.mmax]

        # pad the dim to allow for splitting
        xtfp = F.pad(xtft, [0, self.mpad], mode="constant")

        # transpose: after this, m is split and c is local
        if self.comm_size_azimuth > 1:
            y = distributed_transpose_azimuth.apply(xtfp, (-1, 1))
        else:
            y = xtfp

        # transpose: after this, c is split and h is local
        if self.comm_size_polar > 1:
            yt = distributed_transpose_polar.apply(y, (1, -2))
        else:
            yt = y

        # the input data might be padded, make sure to truncate to nlat:
        ytt = yt[..., :self.nlat, :]

        # do the Legendre-Gauss quadrature
        yttr = torch.view_as_real(ytt)

        # contraction
        yor = torch.einsum('...kmr,mlk->...lmr', yttr, self.weights.to(yttr.dtype)).contiguous()

        # pad if required, truncation is implicit
        yopr = F.pad(yor, [0, 0, 0, 0, 0, self.lpad], mode="constant")
        yop = torch.view_as_complex(yopr)

        # transpose: after this, l is split and c is local
        if self.comm_size_polar	> 1:
            y = distributed_transpose_polar.apply(yop, (-2, 1))
        else:
            y = yop

        return y


class DistributedInverseRealSHT(nn.Module):
    """
    Defines a module for computing the inverse (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    nlat, nlon: Output dimensions
    lmax, mmax: Input dimensions (spherical coefficients). For convenience, these are inferred from the output dimensions

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True):

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
        t = np.flip(np.arccos(cost))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        # spatial paddings
        latdist = (self.nlat + self.comm_size_polar - 1) // self.comm_size_polar
        self.nlatpad = latdist * self.comm_size_polar - self.nlat
        londist = (self.nlon + self.comm_size_azimuth - 1) // self.comm_size_azimuth
        self.nlonpad = londist * self.comm_size_azimuth - self.nlon

        # frequency paddings
        ldist = (self.lmax + self.comm_size_polar - 1) // self.comm_size_polar
        self.lpad = ldist * self.comm_size_polar - self.lmax
        mdist = (self.mmax + self.comm_size_azimuth - 1) // self.comm_size_azimuth
        self.mpad = mdist * self.comm_size_azimuth - self.mmax

        # compute legende polynomials
        pct = _precompute_legpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase)
        pct = torch.from_numpy(pct)

        # split in m
        pct = F.pad(pct, [0, 0, 0, 0, 0, self.mpad], mode="constant")
        pct = torch.split(pct, (self.mmax+self.mpad) // self.comm_size_azimuth, dim=0)[self.comm_rank_azimuth]

        # compute the local pads and sizes
        # spatial
        self.nlat_local = min(latdist, self.nlat - self.comm_rank_polar * latdist)
        self.nlatpad_local = latdist - self.nlat_local
        self.nlon_local = min(londist, self.nlon - self.comm_rank_azimuth * londist)
        self.nlonpad_local = londist - self.nlon_local

        # frequency
        self.lmax_local = min(ldist, self.lmax - self.comm_rank_polar * ldist)
        self.lpad_local = ldist - self.lmax_local
        self.mmax_local = min(mdist, self.mmax - self.comm_rank_azimuth * mdist)
        self.mpad_local = mdist - self.mmax_local

        # register
        self.register_buffer('pct', pct, persistent=False)

    def extra_repr(self):
        """
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        # we need to ensure that we can split the channels evenly
        assert(x.shape[1] % self.comm_size_polar == 0)
        assert(x.shape[1] % self.comm_size_azimuth == 0)

        # transpose: after that, channels are split, l is local:
        if self.comm_size_polar > 1:
            xt = distributed_transpose_polar.apply(x, (1, -2))
        else:
            xt = x

        # remove padding in l:
        xtt = xt[..., :self.lmax, :]

        # Evaluate associated Legendre functions on the output nodes
        xttr = torch.view_as_real(xtt)

        # einsum
        xs = torch.einsum('...lmr, mlk->...kmr', xttr, self.pct.to(xttr.dtype)).contiguous()
        x = torch.view_as_complex(xs)

        # transpose: after this, l is split and channels are local
        xp = F.pad(x, [0, 0, 0, self.nlatpad])

        if self.comm_size_polar > 1:
            y = distributed_transpose_polar.apply(xp, (-2, 1))
        else:
            y = xp

        # transpose: after this, channels are split and m is local
        if self.comm_size_azimuth > 1:
            yt = distributed_transpose_azimuth.apply(y, (1, -1))
        else:
            yt = y

        # truncate
        ytt = yt[..., :self.mmax]

        # apply the inverse (real) FFT
        x = torch.fft.irfft(ytt, n=self.nlon, dim=-1, norm="forward")

        # pad before we transpose back
        xp = F.pad(x, [0, self.nlonpad])

        # transpose: after this, m is split and channels are local
        if self.comm_size_azimuth > 1:
            out = distributed_transpose_azimuth.apply(xp, (-1, 1))
        else:
            out = xp

        return out


class DistributedRealVectorSHT(nn.Module):
    """
    Defines a module for computing the forward (real) vector SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last three dimensions of the input.

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True):
        """
        Initializes the vector SHT Layer, precomputing the necessary quadrature weights

        Parameters:
        nlat: input grid resolution in the latitudinal direction
        nlon: input grid resolution in the longitudinal direction
        grid: type of grid the data lives on
        """

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, w = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, w = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, w = clenshaw_curtiss_weights(nlat, -1, 1)
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
        tq = np.flip(np.arccos(cost))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        # spatial paddings
        latdist = (self.nlat + self.comm_size_polar - 1) // self.comm_size_polar
        self.nlatpad = latdist * self.comm_size_polar - self.nlat
        londist = (self.nlon + self.comm_size_azimuth - 1) // self.comm_size_azimuth
        self.nlonpad = londist * self.comm_size_azimuth - self.nlon

        # frequency paddings
        ldist = (self.lmax + self.comm_size_polar - 1) // self.comm_size_polar
        self.lpad = ldist * self.comm_size_polar - self.lmax
        mdist = (self.mmax + self.comm_size_azimuth - 1) // self.comm_size_azimuth
        self.mpad = mdist * self.comm_size_azimuth - self.mmax

        weights = torch.from_numpy(w)
        dpct = _precompute_dlegpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)
        dpct = torch.from_numpy(dpct)

        # combine integration weights, normalization factor in to one:
        l = torch.arange(0, self.lmax)
        norm_factor = 1. / l / (l+1)
        norm_factor[0] = 1.
        weights = torch.einsum('dmlk,k,l->dmlk', dpct, weights, norm_factor)
        # since the second component is imaginary, we need to take complex conjugation into account
        weights[1] = -1 * weights[1]

        # we need to split in m, pad before:
        weights = F.pad(weights, [0, 0, 0, 0, 0, self.mpad], mode="constant")
        weights = torch.split(weights, (self.mmax+self.mpad) // self.comm_size_azimuth, dim=1)[self.comm_rank_azimuth]

        # remember quadrature weights
        self.register_buffer('weights', weights, persistent=False)

        # compute the local pad and size
        # spatial
        self.nlat_local = min(latdist, self.nlat - self.comm_rank_polar * latdist)
        self.nlatpad_local = latdist - self.nlat_local
        self.nlon_local = min(londist, self.nlon - self.comm_rank_azimuth * londist)
        self.nlonpad_local = londist - self.nlon_local

        # frequency
        self.lmax_local = min(ldist, self.lmax - self.comm_rank_polar * ldist)
        self.lpad_local = ldist - self.lmax_local
        self.mmax_local = min(mdist, self.mmax - self.comm_rank_azimuth * mdist)
        self.mpad_local = mdist - self.mmax_local

    def extra_repr(self):
        """
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        assert(len(x.shape) >= 3)
        assert(x.shape[1] % self.comm_size_polar == 0)
        assert(x.shape[1] % self.comm_size_azimuth == 0)

        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_azimuth > 1:
            xt = distributed_transpose_azimuth.apply(x, (1, -1))
        else:
            xt = x

        # apply real fft in the longitudinal direction: make sure to truncate to nlon
        xtf = 2.0 * torch.pi * torch.fft.rfft(xt, n=self.nlon, dim=-1, norm="forward")

        # truncate
        xtft = xtf[..., :self.mmax]

        # pad the dim to allow for splitting
        xtfp = F.pad(xtft, [0, self.mpad], mode="constant")

        # transpose: after this, m is split and c is local
        if self.comm_size_azimuth > 1:
            y = distributed_transpose_azimuth.apply(xtfp, (-1, 1))
        else:
            y = xtfp

        # transpose: after this, c is split and h is local
        if self.comm_size_polar > 1:
            yt = distributed_transpose_polar.apply(y, (1, -2))
        else:
            yt = y

        # the input data might be padded, make sure to truncate to nlat:
        ytt = yt[..., :self.nlat, :]

        # do the Legendre-Gauss quadrature
        yttr = torch.view_as_real(ytt)

        # create output array
        yor = torch.zeros_like(yttr, dtype=yttr.dtype, device=yttr.device)

        # contraction - spheroidal component
        # real component
        yor[..., 0, :, :, 0] =   torch.einsum('...km,mlk->...lm', yttr[..., 0, :, :, 0], self.weights[0].to(yttr.dtype)) \
                               - torch.einsum('...km,mlk->...lm', yttr[..., 1, :, :, 1], self.weights[1].to(yttr.dtype))
        # iamg component
        yor[..., 0, :, :, 1] =   torch.einsum('...km,mlk->...lm', yttr[..., 0, :, :, 1], self.weights[0].to(yttr.dtype)) \
                               + torch.einsum('...km,mlk->...lm', yttr[..., 1, :, :, 0], self.weights[1].to(yttr.dtype))

        # contraction - toroidal component
        # real component
        yor[..., 1, :, :, 0] = - torch.einsum('...km,mlk->...lm', yttr[..., 0, :, :, 1], self.weights[1].to(yttr.dtype)) \
                               - torch.einsum('...km,mlk->...lm', yttr[..., 1, :, :, 0], self.weights[0].to(yttr.dtype))
        # imag component
        yor[..., 1, :, :, 1] =   torch.einsum('...km,mlk->...lm', yttr[..., 0, :, :, 0], self.weights[1].to(yttr.dtype)) \
                               - torch.einsum('...km,mlk->...lm', yttr[..., 1, :, :, 1], self.weights[0].to(yttr.dtype))

        # pad if required
        yopr = F.pad(yor, [0, 0, 0, 0, 0, self.lpad], mode="constant")
        yop = torch.view_as_complex(yopr)

        # transpose: after this, l is split and c is local
        if self.comm_size_polar > 1:
            y = distributed_transpose_polar.apply(yop, (-2, 1))
        else:
            y = yop

        return y


class DistributedInverseRealVectorSHT(nn.Module):
    """
    Defines a module for computing the inverse (real-valued) vector SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """
    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True):

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
        t = np.flip(np.arccos(cost))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        # spatial paddings
        latdist = (self.nlat + self.comm_size_polar - 1) // self.comm_size_polar
        self.nlatpad = latdist * self.comm_size_polar - self.nlat
        londist = (self.nlon + self.comm_size_azimuth - 1) // self.comm_size_azimuth
        self.nlonpad = londist * self.comm_size_azimuth - self.nlon

        # frequency paddings
        ldist = (self.lmax + self.comm_size_polar - 1) // self.comm_size_polar
        self.lpad = ldist * self.comm_size_polar - self.lmax
        mdist = (self.mmax + self.comm_size_azimuth - 1) // self.comm_size_azimuth
        self.mpad = mdist * self.comm_size_azimuth - self.mmax

        # compute legende polynomials
        dpct = _precompute_dlegpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase)
        dpct = torch.from_numpy(dpct)

        # split in m
        dpct = F.pad(dpct, [0, 0, 0, 0, 0, self.mpad], mode="constant")
        dpct = torch.split(dpct, (self.mmax+self.mpad) // self.comm_size_azimuth, dim=0)[self.comm_rank_azimuth]

        # register buffer
        self.register_buffer('dpct', dpct, persistent=False)

        # compute the local pad and size
        # spatial
        self.nlat_local = min(latdist, self.nlat - self.comm_rank_polar * latdist)
        self.nlatpad_local = latdist - self.nlat_local
        self.nlon_local = min(londist, self.nlon - self.comm_rank_azimuth * londist)
        self.nlonpad_local = londist - self.nlon_local

        # frequency
        self.lmax_local = min(ldist, self.lmax - self.comm_rank_polar * ldist)
        self.lpad_local = ldist - self.lmax_local
        self.mmax_local = min(mdist, self.mmax - self.comm_rank_azimuth * mdist)
        self.mpad_local = mdist - self.mmax_local

    def extra_repr(self):
        """
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        assert(x.shape[1] % self.comm_size_polar == 0)
        assert(x.shape[1] % self.comm_size_azimuth == 0)

        # transpose: after that, channels are split, l is local:
        if self.comm_size_polar > 1:
            xt = distributed_transpose_polar.apply(x, (1, -2))
        else:
            xt = x

        # remove padding in l:
        xtt = xt[..., :self.lmax, :]

        # Evaluate associated Legendre functions on the output nodes
        xttr = torch.view_as_real(xtt)

        # contraction - spheroidal component
        # real component
        srl =   torch.einsum('...lm,mlk->...km', xttr[..., 0, :, :, 0], self.dpct[0].to(xttr.dtype)) \
              - torch.einsum('...lm,mlk->...km', xttr[..., 1, :, :, 1], self.dpct[1].to(xttr.dtype))
        # imag component
        sim =   torch.einsum('...lm,mlk->...km', xttr[..., 0, :, :, 1], self.dpct[0].to(xttr.dtype)) \
              + torch.einsum('...lm,mlk->...km', xttr[..., 1, :, :, 0], self.dpct[1].to(xttr.dtype))

        # contraction - toroidal component
        # real component
        trl = - torch.einsum('...lm,mlk->...km', xttr[..., 0, :, :, 1], self.dpct[1].to(xttr.dtype)) \
              - torch.einsum('...lm,mlk->...km', xttr[..., 1, :, :, 0], self.dpct[0].to(xttr.dtype))
        # imag component
        tim =   torch.einsum('...lm,mlk->...km', xttr[..., 0, :, :, 0], self.dpct[1].to(xttr.dtype)) \
              - torch.einsum('...lm,mlk->...km', xttr[..., 1, :, :, 1], self.dpct[0].to(xttr.dtype))

        # reassemble
        s = torch.stack((srl, sim), -1)
        t = torch.stack((trl, tim), -1)
        xs = torch.stack((s, t), -4)

        # convert to complex
        x = torch.view_as_complex(xs)

        # transpose: after this, l is split and channels are local
        xp = F.pad(x, [0, 0, 0, self.nlatpad])

        if self.comm_size_polar > 1:
            y = distributed_transpose_polar.apply(xp, (-2, 1))
        else:
            y = xp

        # transpose: after this, channels are split and m is local
        if self.comm_size_azimuth > 1:
            yt = distributed_transpose_azimuth.apply(y, (1, -1))
        else:
            yt = y

        # truncate
        ytt = yt[..., :self.mmax]

        # apply the inverse (real) FFT
        x = torch.fft.irfft(x, n=self.nlon, dim=-1, norm="forward")

        # pad before we transpose back
        xp = F.pad(x, [0, self.nlonpad])

        # transpose: after this, m is split and channels are local
        if self.comm_size_azimuth > 1:
            out = distributed_transpose_azimuth.apply(xp, (-1, 1))
        else:
            out = xp

        return out