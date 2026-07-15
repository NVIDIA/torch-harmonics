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

from torch_harmonics.fft import irfft, rfft
from torch_harmonics.legendre import _precompute_dlegpoly, _precompute_legpoly
from torch_harmonics.quadrature import clenshaw_curtiss_weights, legendre_gauss_weights, lobatto_weights
from torch_harmonics.truncation import truncate_sht

from .primitives import (
    compute_split_shapes,
    distributed_transpose_azimuth,
    distributed_transpose_polar,
    flatten_and_pad_leading_dims,
    split_tensor_along_dim,
    unpad_and_unflatten_leading_dims,
)
from .utils import azimuth_group_rank, azimuth_group_size, polar_group_rank, polar_group_size


class DistributedRealSHT(nn.Module):
    """
    Distributed version of the forward (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last two dimensions of the input.

    **Distribution scheme.**
    The input tensor has shape ``(B, C, nlat_local, nlon_local)`` where latitudes
    and longitudes are split across the polar and azimuth process groups
    respectively.  All leading dimensions are flattened into a single axis
    ``N = B * C`` which is used as the redistribution currency during the
    all-to-all transposes.  The forward pass proceeds as follows:

    1. **Azimuth transpose** (``nlon`` ↔ ``N``) — each rank trades its local
       longitude chunk for a slice of the channel axis, making ``nlon`` fully
       local so the real FFT can be applied.
    2. **Real FFT** along the (now local) longitude dimension.
    3. **Azimuth transpose** (``N`` ↔ ``mmax``) — redistribute so that spectral
       orders ``m`` are split across azimuth ranks and channels are local again.
    4. **Polar transpose** (``N`` ↔ ``nlat``) — trade channel slices for the
       full latitude axis, making ``nlat`` local for the Legendre contraction.
    5. **Legendre contraction** — local matrix multiply with the quadrature
       weights, producing spectral degrees ``l``.
    6. **Polar transpose** (``l`` ↔ ``N``) — redistribute so that degrees ``l``
       are split across polar ranks.

    The output has shape ``(B, C, lmax_local, mmax_local)`` with spectral modes
    partitioned in the same way as the spatial grid.

    If ``N < max(polar_group_size, azimuth_group_size)``, the leading axis is
    zero-padded before the transposes and the padding is removed afterwards;
    since the transform is linear this is exact.

    .. seealso::
        :class:`torch_harmonics.RealSHT`
            Serial counterpart with full mathematical description and parameter
            documentation.

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
        Grid type (``"equiangular"``, ``"legendre-gauss"``, ``"lobatto"``, ``"equiangular-trapezoidal"``), by default ``"equiangular"``
    norm: str
        Normalization type (``"ortho"``, ``"schmidt"``, ``"unnorm"``), by default ``"ortho"``
    csphase: bool
        Whether to apply the Condon-Shortley phase factor, by default True

    Returns
    -------
    x: torch.Tensor
        Tensor of shape (..., lmax, mmax)

    References
    ----------
    :cite:`Schaeffer2013`, :cite:`Wang2018`
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
        elif self.grid == "lobatto":
            cost, weights = lobatto_weights(nlat, -1, 1)
        elif self.grid == "equiangular":
            cost, weights = clenshaw_curtiss_weights(nlat, -1, 1)
        else:
            raise (ValueError("Unknown quadrature mode"))

        # get the comms grid:
        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # apply cosine transform and flip them
        tq = torch.flip(torch.arccos(cost), dims=(0,))

        # determine maximum degrees based on triangular truncation
        self.lmax, self.mmax = truncate_sht(self.nlat, self.nlon, lmax, mmax, self.grid)

        # compute splits
        self.lat_shapes = compute_split_shapes(self.nlat, self.comm_size_polar)
        self.nlat_local = self.lat_shapes[self.comm_rank_polar]
        self.lon_shapes = compute_split_shapes(self.nlon, self.comm_size_azimuth)
        self.nlon_local = self.lon_shapes[self.comm_rank_azimuth]
        self.l_shapes = compute_split_shapes(self.lmax, self.comm_size_polar)
        self.m_shapes = compute_split_shapes(self.mmax, self.comm_size_azimuth)
        self.mmax_local = self.m_shapes[self.comm_rank_azimuth]

        # combine quadrature weights with the legendre weights
        pct = _precompute_legpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)
        weights = torch.einsum("mlk,k->mlk", pct, weights)

        # split weights
        weights = split_tensor_along_dim(weights, dim=0, num_chunks=self.comm_size_azimuth)[self.comm_rank_azimuth].contiguous()

        # remember quadrature weights
        self.register_buffer("weights", weights, persistent=False)

    def extra_repr(self):
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    def forward(self, x: torch.Tensor):

        torch._check(x.dim() >= 3, lambda: f"Expected tensor with at least 3 dimensions but got {x.dim()} instead")
        torch._check(x.shape[-2] == self.nlat_local, lambda: f"Expected latitudes shape[-2]=={self.nlat_local}, got {x.shape[-2]}")
        torch._check(x.shape[-1] == self.nlon_local, lambda: f"Expected longitudes shape[-1]=={self.nlon_local}, got {x.shape[-1]}")

        # the transposes below redistribute the leading (channel/batch) axis across the
        # process grid, so it must be at least as large as the larger comm group. Flatten
        # all leading dims into that axis and zero-pad it if needed (linear transform, so
        # padding stays zero); restore the original layout before returning.
        x, lead_shape, lead_size = flatten_and_pad_leading_dims(x, max(self.comm_size_polar, self.comm_size_azimuth))
        num_chans = x.shape[-3]

        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_azimuth > 1:
            x = distributed_transpose_azimuth(x, (-3, -1), self.lon_shapes)

        # apply real fft in the longitudinal direction: make sure to truncate to nlon
        x = 2.0 * torch.pi * rfft(x, nmodes=self.mmax, dim=-1, norm="forward")

        # transpose: after this, m is split and c is local
        if self.comm_size_azimuth > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = distributed_transpose_azimuth(x, (-1, -3), chan_shapes)

        # transpose: after this, c is split and h is local
        if self.comm_size_polar > 1:
            x = distributed_transpose_polar(x, (-3, -2), self.lat_shapes)

        # transpose to put the contraction dim (nlat) on the fast axis
        x = x.transpose(-1, -2)
        x_re = x.real.contiguous()
        x_im = x.imag.contiguous()

        # Legendre-Gauss quadrature: contract over k=nlat (stride-1 in both operands)
        w = self.weights.to(x_re.dtype)
        out_re = torch.einsum("...mk,mlk->...lm", x_re, w)
        out_im = torch.einsum("...mk,mlk->...lm", x_im, w)
        # force contiguous: the ...lm einsum output is non-contiguous and inductor's aten.complex
        # meta predicts a contiguous layout, tripping assert_size_stride under torch.compile.
        x = torch.complex(out_re.contiguous(), out_im.contiguous())

        # transpose: after this, l is split and c is local
        if self.comm_size_polar > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_polar)
            x = distributed_transpose_polar(x, (-2, -3), chan_shapes)

        # drop padding and restore the original leading dims
        x = unpad_and_unflatten_leading_dims(x, lead_shape, lead_size)

        return x


class DistributedInverseRealSHT(nn.Module):
    """
    Distributed version of the inverse (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.

    **Distribution scheme.**
    The input tensor has shape ``(B, C, lmax_local, mmax_local)`` where spectral
    degrees and orders are split across the polar and azimuth process groups.
    All leading dimensions are flattened into ``N = B * C`` for redistribution.
    The forward pass proceeds as follows:

    1. **Polar transpose** (``N`` ↔ ``lmax``) — trade channel slices for the
       full degree axis, making ``l`` local for the Legendre synthesis.
    2. **Legendre synthesis** — local matrix multiply with the associated
       Legendre polynomials, producing latitude points.
    3. **Polar transpose** (``nlat`` ↔ ``N``) — redistribute so that latitudes
       are split across polar ranks and channels are local.
    4. **Azimuth transpose** (``N`` ↔ ``mmax``) — make spectral orders ``m``
       fully local for the inverse FFT.
    5. **Inverse real FFT** along the (now local) ``m`` / longitude dimension.
    6. **Azimuth transpose** (``nlon`` ↔ ``N``) — redistribute so that
       longitudes are split across azimuth ranks.

    The output has shape ``(B, C, nlat_local, nlon_local)`` with the spatial
    grid partitioned in the same way as the input spectral modes.

    If ``N < max(polar_group_size, azimuth_group_size)``, the leading axis is
    zero-padded before the transposes and the padding is removed afterwards;
    since the transform is linear this is exact.

    .. seealso::
        :class:`torch_harmonics.InverseRealSHT`
            Serial counterpart with full mathematical description and parameter
            documentation.

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
        Grid type (``"equiangular"``, ``"legendre-gauss"``, ``"lobatto"``, ``"equiangular-trapezoidal"``), by default ``"equiangular"``
    norm: str
        Normalization type (``"ortho"``, ``"schmidt"``, ``"unnorm"``), by default ``"ortho"``
    csphase: bool
        Whether to apply the Condon-Shortley phase factor, by default True

    Returns
    -------
    x: torch.Tensor
        Tensor of shape (..., lmax, mmax)

    References
    ----------
    :cite:`Schaeffer2013`, :cite:`Wang2018`
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
        elif self.grid == "lobatto":
            cost, _ = lobatto_weights(nlat, -1, 1)
        elif self.grid == "equiangular":
            cost, _ = clenshaw_curtiss_weights(nlat, -1, 1)
        else:
            raise (ValueError("Unknown quadrature mode"))

        # get the comms grid:
        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # apply cosine transform and flip them
        t = torch.flip(torch.arccos(cost), dims=(0,))

        # determine maximum degrees based on triangular truncation
        self.lmax, self.mmax = truncate_sht(self.nlat, self.nlon, lmax, mmax, self.grid)

        # compute splits
        self.lat_shapes = compute_split_shapes(self.nlat, self.comm_size_polar)
        self.lon_shapes = compute_split_shapes(self.nlon, self.comm_size_azimuth)
        self.l_shapes = compute_split_shapes(self.lmax, self.comm_size_polar)
        self.lmax_local = self.l_shapes[self.comm_rank_polar]
        self.m_shapes = compute_split_shapes(self.mmax, self.comm_size_azimuth)
        self.mmax_local = self.m_shapes[self.comm_rank_azimuth]

        # compute legendre polynomials
        # store as (mmax, nlat, lmax) so the contraction dim l is stride-1
        pct = _precompute_legpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase)
        pct = pct.permute(0, 2, 1).contiguous()

        # split in m
        pct = split_tensor_along_dim(pct, dim=0, num_chunks=self.comm_size_azimuth)[self.comm_rank_azimuth].contiguous()

        # register
        self.register_buffer("pct", pct, persistent=False)

    def extra_repr(self):
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    def forward(self, x: torch.Tensor):

        torch._check(x.dim() >= 3, lambda: f"Expected tensor with at least 3 dimensions but got {x.dim()} instead")
        torch._check(x.shape[-2] == self.lmax_local, lambda: f"Expected spherical harmonic degrees (lmax) shape[-2]=={self.lmax_local}, got {x.shape[-2]}")
        torch._check(x.shape[-1] == self.mmax_local, lambda: f"Expected spherical harmonic orders (mmax) shape[-1]=={self.mmax_local}, got {x.shape[-1]}")

        # the transposes below redistribute the leading (channel/batch) axis across the
        # process grid, so it must be at least as large as the larger comm group. Flatten
        # all leading dims into that axis and zero-pad it if needed (linear transform, so
        # padding stays zero); restore the original layout before returning.
        x, lead_shape, lead_size = flatten_and_pad_leading_dims(x, max(self.comm_size_polar, self.comm_size_azimuth))
        num_chans = x.shape[-3]

        # transpose: after that, channels are split, l is local:
        if self.comm_size_polar > 1:
            x = distributed_transpose_polar(x, (-3, -2), self.l_shapes)

        # transpose to put the contraction dim (lmax) on the fast axis
        x = x.transpose(-1, -2)
        x_re = x.real.contiguous()
        x_im = x.imag.contiguous()

        # legendre transformation: contract over l=lmax (stride-1 in both operands)
        # pct layout: (mmax_local, nlat, lmax)
        w = self.pct.to(x_re.dtype)
        out_re = torch.einsum("...ml,mkl->...km", x_re, w)
        out_im = torch.einsum("...ml,mkl->...km", x_im, w)
        # force contiguous: the einsum output is non-contiguous and inductor's aten.complex meta
        # predicts a contiguous layout, tripping assert_size_stride under torch.compile.
        x = torch.complex(out_re.contiguous(), out_im.contiguous())

        if self.comm_size_polar > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_polar)
            x = distributed_transpose_polar(x, (-2, -3), chan_shapes)

        # transpose: after this, channels are split and m is local
        if self.comm_size_azimuth > 1:
            x = distributed_transpose_azimuth(x, (-3, -1), self.m_shapes)

        # apply the inverse (real) FFT
        x = irfft(x, n=self.nlon, dim=-1, norm="forward")

        # transpose: after this, m is split and channels are local
        if self.comm_size_azimuth > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = distributed_transpose_azimuth(x, (-1, -3), chan_shapes)

        # drop padding and restore the original leading dims
        x = unpad_and_unflatten_leading_dims(x, lead_shape, lead_size)

        return x


class DistributedRealVectorSHT(nn.Module):
    """
    Distributed version of the forward (real) vector SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last three dimensions of the input.

    The distribution scheme is the same as for
    :class:`DistributedRealSHT` (see its docstring for a step-by-step
    description of the all-to-all transposes over the ``N = B * C`` axis).
    The additional size-2 vector component dimension is preserved throughout.

    .. seealso::
        :class:`torch_harmonics.RealVectorSHT`
            Serial counterpart with full mathematical description and parameter
            documentation.

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
        Grid type (``"equiangular"``, ``"legendre-gauss"``, ``"lobatto"``, ``"equiangular-trapezoidal"``), by default ``"equiangular"``
    norm: str
        Normalization type (``"ortho"``, ``"schmidt"``, ``"unnorm"``), by default ``"ortho"``
    csphase: bool
        Whether to apply the Condon-Shortley phase factor, by default True

    Returns
    -------
    x: torch.Tensor
        Tensor of shape (..., lmax, mmax)

    References
    ----------
    :cite:`Schaeffer2013`, :cite:`Wang2018`
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
        elif self.grid == "lobatto":
            cost, weights = lobatto_weights(nlat, -1, 1)
        elif self.grid == "equiangular":
            cost, weights = clenshaw_curtiss_weights(nlat, -1, 1)
        else:
            raise (ValueError("Unknown quadrature mode"))

        # get the comms grid:
        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # apply cosine transform and flip them
        tq = torch.flip(torch.arccos(cost), dims=(0,))

        # determine maximum degrees based on triangular truncation
        self.lmax, self.mmax = truncate_sht(self.nlat, self.nlon, lmax, mmax, self.grid)

        # compute splits
        self.lat_shapes = compute_split_shapes(self.nlat, self.comm_size_polar)
        self.nlat_local = self.lat_shapes[self.comm_rank_polar]
        self.lon_shapes = compute_split_shapes(self.nlon, self.comm_size_azimuth)
        self.nlon_local = self.lon_shapes[self.comm_rank_azimuth]
        self.l_shapes = compute_split_shapes(self.lmax, self.comm_size_polar)
        self.m_shapes = compute_split_shapes(self.mmax, self.comm_size_azimuth)
        self.mmax_local = self.m_shapes[self.comm_rank_azimuth]

        # compute weights
        dpct = _precompute_dlegpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)

        # combine integration weights, normalization factor in to one:
        l = torch.arange(0, self.lmax)
        norm_factor = 1.0 / l / (l + 1)
        norm_factor[0] = 1.0
        weights = torch.einsum("dmlk,k,l->dmlk", dpct, weights, norm_factor)
        # since the second component is imaginary, we need to take complex conjugation into account
        weights[1] = -1 * weights[1]

        # we need to split in m, pad before:
        weights = split_tensor_along_dim(weights, dim=1, num_chunks=self.comm_size_azimuth)[self.comm_rank_azimuth].contiguous()

        # remember quadrature weights
        self.register_buffer("weights", weights, persistent=False)

    def extra_repr(self):
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    def forward(self, x: torch.Tensor):

        torch._check(x.dim() >= 4, lambda: f"Expected tensor with at least 4 dimensions but got {x.dim()} instead")
        torch._check(x.shape[-3] == 2, lambda: f"Expected vector field shape[-3]==2, got {x.shape[-3]}")
        torch._check(x.shape[-2] == self.nlat_local, lambda: f"Expected latitudes shape[-2]=={self.nlat_local}, got {x.shape[-2]}")
        torch._check(x.shape[-1] == self.nlon_local, lambda: f"Expected longitudes shape[-1]=={self.nlon_local}, got {x.shape[-1]}")

        # the transposes below redistribute the leading (channel/batch) axis across the
        # process grid, so it must be at least as large as the larger comm group. Flatten
        # all leading dims into that axis -- keeping the trailing (2, nlat, nlon) intact --
        # and zero-pad it if needed (linear transform, so padding stays zero); restore the
        # original layout before returning.
        x, lead_shape, lead_size = flatten_and_pad_leading_dims(x, max(self.comm_size_polar, self.comm_size_azimuth), num_trailing_dims=3)
        num_chans = x.shape[-4]

        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_azimuth > 1:
            x = distributed_transpose_azimuth(x, (-4, -1), self.lon_shapes)

        # apply real fft in the longitudinal direction: make sure to truncate to nlon
        x = 2.0 * torch.pi * rfft(x, nmodes=self.mmax, dim=-1, norm="forward")

        # transpose: after this, m is split and c is local
        if self.comm_size_azimuth > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = distributed_transpose_azimuth(x, (-1, -4), chan_shapes)

        # transpose: after this, c is split and h is local
        if self.comm_size_polar > 1:
            x = distributed_transpose_polar(x, (-4, -2), self.lat_shapes)

        # transpose to put the contraction dim (nlat) on the fast axis
        x = x.transpose(-1, -2)
        x_re = x.real.contiguous()
        x_im = x.imag.contiguous()

        w0 = self.weights[0].to(x_re.dtype)
        w1 = self.weights[1].to(x_re.dtype)

        # contraction - spheroidal component
        s_re = torch.einsum("...mk,mlk->...lm", x_re[..., 0, :, :], w0) - torch.einsum("...mk,mlk->...lm", x_im[..., 1, :, :], w1)
        s_im = torch.einsum("...mk,mlk->...lm", x_im[..., 0, :, :], w0) + torch.einsum("...mk,mlk->...lm", x_re[..., 1, :, :], w1)

        # contraction - toroidal component
        t_re = -torch.einsum("...mk,mlk->...lm", x_im[..., 0, :, :], w1) - torch.einsum("...mk,mlk->...lm", x_re[..., 1, :, :], w0)
        t_im = torch.einsum("...mk,mlk->...lm", x_re[..., 0, :, :], w1) - torch.einsum("...mk,mlk->...lm", x_im[..., 1, :, :], w0)

        x = torch.stack((torch.complex(s_re, s_im), torch.complex(t_re, t_im)), dim=-3).contiguous()

        # transpose: after this, l is split and c is local
        if self.comm_size_polar > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_polar)
            x = distributed_transpose_polar(x, (-2, -4), chan_shapes)

        # drop padding and restore the original leading dims
        x = unpad_and_unflatten_leading_dims(x, lead_shape, lead_size, num_trailing_dims=3)

        return x


class DistributedInverseRealVectorSHT(nn.Module):
    """
    Distributed version of the inverse (real-valued) vector SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.

    The distribution scheme is the same as for
    :class:`DistributedInverseRealSHT` (see its docstring for a step-by-step
    description of the all-to-all transposes over the ``N = B * C`` axis).
    The additional size-2 vector component dimension is preserved throughout.

    .. seealso::
        :class:`torch_harmonics.InverseRealVectorSHT`
            Serial counterpart with full mathematical description and parameter
            documentation.

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
        Grid type (``"equiangular"``, ``"legendre-gauss"``, ``"lobatto"``, ``"equiangular-trapezoidal"``), by default ``"equiangular"``
    norm: str
        Normalization type (``"ortho"``, ``"schmidt"``, ``"unnorm"``), by default ``"ortho"``
    csphase: bool
        Whether to apply the Condon-Shortley phase factor, by default True

    Returns
    -------
    x: torch.Tensor
        Tensor of shape (..., lmax, mmax)

    References
    ----------
    :cite:`Schaeffer2013`, :cite:`Wang2018`
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
        elif self.grid == "lobatto":
            cost, _ = lobatto_weights(nlat, -1, 1)
        elif self.grid == "equiangular":
            cost, _ = clenshaw_curtiss_weights(nlat, -1, 1)
        else:
            raise (ValueError("Unknown quadrature mode"))

        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # apply cosine transform and flip them
        t = torch.flip(torch.arccos(cost), dims=(0,))

        # determine maximum degrees based on triangular truncation
        self.lmax, self.mmax = truncate_sht(self.nlat, self.nlon, lmax, mmax, self.grid)

        # compute splits
        self.lat_shapes = compute_split_shapes(self.nlat, self.comm_size_polar)
        self.lon_shapes = compute_split_shapes(self.nlon, self.comm_size_azimuth)
        self.l_shapes = compute_split_shapes(self.lmax, self.comm_size_polar)
        self.lmax_local = self.l_shapes[self.comm_rank_polar]
        self.m_shapes = compute_split_shapes(self.mmax, self.comm_size_azimuth)
        self.mmax_local = self.m_shapes[self.comm_rank_azimuth]

        # compute legendre polynomials
        # store as (2, mmax, nlat, lmax) so the contraction dim l is stride-1
        dpct = _precompute_dlegpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase)
        dpct = dpct.permute(0, 1, 3, 2).contiguous()

        # split in m
        dpct = split_tensor_along_dim(dpct, dim=1, num_chunks=self.comm_size_azimuth)[self.comm_rank_azimuth].contiguous()

        # register buffer
        self.register_buffer("dpct", dpct, persistent=False)

    def extra_repr(self):
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    def forward(self, x: torch.Tensor):

        torch._check(x.dim() >= 4, lambda: f"Expected tensor with at least 4 dimensions but got {x.dim()} instead")
        torch._check(x.shape[-3] == 2, lambda: f"Expected vector field shape[-3]==2, got {x.shape[-3]}")
        torch._check(x.shape[-2] == self.lmax_local, lambda: f"Expected spherical harmonic degrees (lmax) shape[-2]=={self.lmax_local}, got {x.shape[-2]}")
        torch._check(x.shape[-1] == self.mmax_local, lambda: f"Expected spherical harmonic orders (mmax) shape[-1]=={self.mmax_local}, got {x.shape[-1]}")

        # the transposes below redistribute the leading (channel/batch) axis across the
        # process grid, so it must be at least as large as the larger comm group. Flatten
        # all leading dims into that axis -- keeping the trailing (2, lmax, mmax) intact --
        # and zero-pad it if needed (linear transform, so padding stays zero); restore the
        # original layout before returning.
        x, lead_shape, lead_size = flatten_and_pad_leading_dims(x, max(self.comm_size_polar, self.comm_size_azimuth), num_trailing_dims=3)
        num_chans = x.shape[-4]

        # transpose: after that, channels are split, l is local:
        if self.comm_size_polar > 1:
            x = distributed_transpose_polar(x, (-4, -2), self.l_shapes)

        # transpose to put the contraction dim (lmax) on the fast axis
        x = x.transpose(-1, -2)
        x_re = x.real.contiguous()
        x_im = x.imag.contiguous()

        # dpct layout: (2, mmax_local, nlat, lmax) — contract over l (stride-1 in both operands)
        d0 = self.dpct[0].to(x_re.dtype)
        d1 = self.dpct[1].to(x_re.dtype)

        # contraction - spheroidal component
        srl = torch.einsum("...ml,mkl->...km", x_re[..., 0, :, :], d0) - torch.einsum("...ml,mkl->...km", x_im[..., 1, :, :], d1)
        sim = torch.einsum("...ml,mkl->...km", x_im[..., 0, :, :], d0) + torch.einsum("...ml,mkl->...km", x_re[..., 1, :, :], d1)

        # contraction - toroidal component
        trl = -torch.einsum("...ml,mkl->...km", x_im[..., 0, :, :], d1) - torch.einsum("...ml,mkl->...km", x_re[..., 1, :, :], d0)
        tim = torch.einsum("...ml,mkl->...km", x_re[..., 0, :, :], d1) - torch.einsum("...ml,mkl->...km", x_im[..., 1, :, :], d0)

        # reassemble
        x = torch.stack((torch.complex(srl, sim), torch.complex(trl, tim)), dim=-3).contiguous()

        if self.comm_size_polar > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_polar)
            x = distributed_transpose_polar(x, (-2, -4), chan_shapes)

        # transpose: after this, channels are split and m is local
        if self.comm_size_azimuth > 1:
            x = distributed_transpose_azimuth(x, (-4, -1), self.m_shapes)

        # apply the inverse (real) FFT
        x = irfft(x, n=self.nlon, dim=-1, norm="forward")

        # transpose: after this, m is split and channels are local
        if self.comm_size_azimuth > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = distributed_transpose_azimuth(x, (-1, -4), chan_shapes)

        # drop padding and restore the original leading dims
        x = unpad_and_unflatten_leading_dims(x, lead_shape, lead_size, num_trailing_dims=3)

        return x
