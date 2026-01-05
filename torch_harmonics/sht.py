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
import torch.fft

from torch_harmonics.quadrature import legendre_gauss_weights, lobatto_weights, clenshaw_curtiss_weights
from torch_harmonics.legendre import _precompute_legpoly, _precompute_dlegpoly


class RealSHT(nn.Module):
    """
    Defines a module for computing the forward (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last two dimensions of the input

    Parameters
    -----------
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

        # compute quadrature points and lmax based on the exactness of the quadrature
        if self.grid == "legendre-gauss":
            cost, weights = legendre_gauss_weights(nlat, -1, 1)
            # maximum polynomial degree for Gauss Legendre is 2 * nlat - 1 >= 2 * lmax
            # and therefore lmax = nlat - 1 (inclusive)
            self.lmax = lmax or (self.nlat // 2)
        elif self.grid == "lobatto":
            cost, weights = lobatto_weights(nlat, -1, 1)
            # maximum polynomial degree for Gauss Legendre is 2 * nlat - 3 >= 2 * lmax
            # and therefore lmax = nlat - 2 (inclusive)
            self.lmax = lmax or (self.nlat - 1) // 2
        elif self.grid == "equiangular":
            cost, weights = clenshaw_curtiss_weights(nlat, -1, 1)
            # in principle, Clenshaw-Curtiss quadrature is only exact up to polynomial degrees of nlat
            # however, we observe that the quadrature is remarkably accurate for higher degress. This is why we do not
            # choose a lower lmax for now.
            self.lmax = lmax or (self.nlat // 2)
        else:
            raise (ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        tq = torch.flip(torch.arccos(cost), dims=(0,))

        # determine the dimensions
        self.mmax = mmax or (self.nlon // 2 + 1)

        # use the minimum of mmax and lmax
        self.lmax = min(self.lmax, self.mmax)
        self.mmax = self.lmax


        # combine quadrature weights with the legendre weights
        pct = _precompute_legpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)
        weights = torch.einsum("mlk,k->mlk", pct, weights).contiguous()

        # remember quadrature weights
        self.register_buffer("weights", weights, persistent=False)

    def extra_repr(self):
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    def forward(self, x: torch.Tensor):

        if x.dim() < 2:
            raise ValueError(f"Expected tensor with at least 2 dimensions but got {x.dim()} instead")

        assert x.shape[-2] == self.nlat
        assert x.shape[-1] == self.nlon

        # apply real fft in the longitudinal direction
        x = 2.0 * torch.pi * torch.fft.rfft(x, dim=-1, norm="forward")

        # do the Legendre-Gauss quadrature
        x = torch.view_as_real(x)

        # distributed contraction: fork
        out_shape = list(x.size())
        out_shape[-3] = self.lmax
        out_shape[-2] = self.mmax
        xout = torch.zeros(out_shape, dtype=x.dtype, device=x.device)

        # contraction
        xout[..., 0] = torch.einsum("...km,mlk->...lm", x[..., : self.mmax, 0], self.weights.to(x.dtype))
        xout[..., 1] = torch.einsum("...km,mlk->...lm", x[..., : self.mmax, 1], self.weights.to(x.dtype))
        x = torch.view_as_complex(xout)

        return x


class InverseRealSHT(nn.Module):
    """
    Defines a module for computing the inverse (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.

    Parameters
    -----------
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

    Raises
    ------
    ValueError: If the grid type is unknown

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
            self.lmax = lmax or (self.nlat // 2)
        elif self.grid == "lobatto":
            cost, _ = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or (self.nlat - 1) // 2
        elif self.grid == "equiangular":
            cost, _ = clenshaw_curtiss_weights(nlat, -1, 1)
            self.lmax = lmax or (self.nlat // 2)
        else:
            raise (ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        t = torch.flip(torch.arccos(cost), dims=(0,))

        # determine the dimensions
        self.mmax = mmax or (self.nlon // 2 + 1)

        # use the minimum of mmax and lmax
        self.lmax = min(self.lmax, self.mmax)
        self.mmax = self.lmax

        pct = _precompute_legpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase)

        # register buffer
        self.register_buffer("pct", pct, persistent=False)

    def extra_repr(self):
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    def forward(self, x: torch.Tensor):

        if len(x.shape) < 2:
            raise ValueError(f"Expected tensor with at least 2 dimensions but got {len(x.shape)} instead")

        assert x.shape[-2] == self.lmax
        assert x.shape[-1] == self.mmax

        # Evaluate associated Legendre functions on the output nodes
        x = torch.view_as_real(x)

        # prepare output
        out_shape = list(x.size())
        out_shape[-3] = self.nlat
        out_shape[-2] = self.mmax
        xs = torch.zeros(out_shape, dtype=x.dtype, device=x.device)

        # legendre transformation
        xs[..., 0] = torch.einsum("...lm,mlk->...km", x[..., 0], self.pct.to(x.dtype))
        xs[..., 1] = torch.einsum("...lm,mlk->...km", x[..., 1], self.pct.to(x.dtype))

        # apply the inverse (real) FFT
        x = torch.view_as_complex(xs)

        # ensure that imaginary part of 0 and nyquist components are zero
        # this is important because not all backend algorithms provided through the
        # irfft interface ensure that
        x[..., 0].imag = 0.0
        if (self.nlon % 2 == 0) and (self.nlon // 2 < self.mmax):
            x[..., self.nlon // 2].imag = 0.0
        
        x = torch.fft.irfft(x, n=self.nlon, dim=-1, norm="forward")

        return x


class RealVectorSHT(nn.Module):
    """
    Defines a module for computing the forward (real) vector SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last three dimensions of the input.

    Parameters
    -----------
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
            self.lmax = lmax or self.nlat - 1
        elif self.grid == "equiangular":
            cost, weights = clenshaw_curtiss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise (ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        tq = torch.flip(torch.arccos(cost), dims=(0,))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1
        dpct = _precompute_dlegpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)

        # combine integration weights, normalization factor in to one:
        l = torch.arange(0, self.lmax)
        norm_factor = 1.0 / l / (l + 1)
        norm_factor[0] = 1.0
        weights = torch.einsum("dmlk,k,l->dmlk", dpct, weights, norm_factor).contiguous()
        # since the second component is imaginary, we need to take complex conjugation into account
        weights[1] = -1 * weights[1]

        # remember quadrature weights
        self.register_buffer("weights", weights, persistent=False)

    def extra_repr(self):
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    def forward(self, x: torch.Tensor):

        if x.dim() < 3:
            raise ValueError(f"Expected tensor with at least 3 dimensions but got {x.dim()} instead")

        assert x.shape[-2] == self.nlat
        assert x.shape[-1] == self.nlon

        # apply real fft in the longitudinal direction
        x = 2.0 * torch.pi * torch.fft.rfft(x, dim=-1, norm="forward")

        # do the Legendre-Gauss quadrature
        x = torch.view_as_real(x)

        # distributed contraction: fork
        out_shape = list(x.size())
        out_shape[-3] = self.lmax
        out_shape[-2] = self.mmax
        xout = torch.zeros(out_shape, dtype=x.dtype, device=x.device)

        # contraction - spheroidal component
        # real component
        xout[..., 0, :, :, 0] = torch.einsum("...km,mlk->...lm", x[..., 0, :, : self.mmax, 0], self.weights[0].to(x.dtype)) - torch.einsum(
            "...km,mlk->...lm", x[..., 1, :, : self.mmax, 1], self.weights[1].to(x.dtype)
        )

        # iamg component
        xout[..., 0, :, :, 1] = torch.einsum("...km,mlk->...lm", x[..., 0, :, : self.mmax, 1], self.weights[0].to(x.dtype)) + torch.einsum(
            "...km,mlk->...lm", x[..., 1, :, : self.mmax, 0], self.weights[1].to(x.dtype)
        )

        # contraction - toroidal component
        # real component
        xout[..., 1, :, :, 0] = -torch.einsum("...km,mlk->...lm", x[..., 0, :, : self.mmax, 1], self.weights[1].to(x.dtype)) - torch.einsum(
            "...km,mlk->...lm", x[..., 1, :, : self.mmax, 0], self.weights[0].to(x.dtype)
        )
        # imag component
        xout[..., 1, :, :, 1] = torch.einsum("...km,mlk->...lm", x[..., 0, :, : self.mmax, 0], self.weights[1].to(x.dtype)) - torch.einsum(
            "...km,mlk->...lm", x[..., 1, :, : self.mmax, 1], self.weights[0].to(x.dtype)
        )

        return torch.view_as_complex(xout)


class InverseRealVectorSHT(nn.Module):
    """
    Defines a module for computing the inverse (real-valued) vector SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.

    Parameters
    -----------
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
            self.lmax = lmax or self.nlat - 1
        elif self.grid == "equiangular":
            cost, _ = clenshaw_curtiss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise (ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        t = torch.flip(torch.arccos(cost), dims=(0,))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        dpct = _precompute_dlegpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase)

        # register weights
        self.register_buffer("dpct", dpct, persistent=False)

    def extra_repr(self):
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    def forward(self, x: torch.Tensor):

        if x.dim() < 3:
            raise ValueError(f"Expected tensor with at least 3 dimensions but got {x.dim()} instead")

        assert x.shape[-2] == self.lmax
        assert x.shape[-1] == self.mmax

        # Evaluate associated Legendre functions on the output nodes
        x = torch.view_as_real(x)

        # contraction - spheroidal component
        # real component
        srl = torch.einsum("...lm,mlk->...km", x[..., 0, :, :, 0], self.dpct[0].to(x.dtype)) - torch.einsum("...lm,mlk->...km", x[..., 1, :, :, 1], self.dpct[1].to(x.dtype))
        # iamg component
        sim = torch.einsum("...lm,mlk->...km", x[..., 0, :, :, 1], self.dpct[0].to(x.dtype)) + torch.einsum("...lm,mlk->...km", x[..., 1, :, :, 0], self.dpct[1].to(x.dtype))

        # contraction - toroidal component
        # real component
        trl = -torch.einsum("...lm,mlk->...km", x[..., 0, :, :, 1], self.dpct[1].to(x.dtype)) - torch.einsum("...lm,mlk->...km", x[..., 1, :, :, 0], self.dpct[0].to(x.dtype))
        # imag component
        tim = torch.einsum("...lm,mlk->...km", x[..., 0, :, :, 0], self.dpct[1].to(x.dtype)) - torch.einsum("...lm,mlk->...km", x[..., 1, :, :, 1], self.dpct[0].to(x.dtype))

        # reassemble
        s = torch.stack((srl, sim), -1)
        t = torch.stack((trl, tim), -1)
        xs = torch.stack((s, t), -4)

        # apply the inverse (real) FFT
        x = torch.view_as_complex(xs)

        # ensure that imaginary part of 0 and nyquist components are zero
        # this is important because not all backend algorithms provided through the
        # irfft interface ensure that
        x[..., 0].imag = 0.0
        if (self.nlon % 2 == 0) and (self.nlon // 2 < self.mmax):
            x[..., self.nlon // 2].imag = 0.0
        
        x = torch.fft.irfft(x, n=self.nlon, dim=-1, norm="forward")

        return x
