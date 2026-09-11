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
from torch_harmonics.utils import check

from .primitives import (
    compute_split_shapes,
    distributed_transpose_azimuth,
    flatten_and_pad_leading_dims,
    reduce_from_scatter_to_polar_region,
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
    azimuth all-to-all transposes.  The forward pass proceeds as follows:

    1. **Azimuth transpose** (``nlon`` ↔ ``N``) — each rank trades its local
       longitude chunk for a slice of the channel axis, making ``nlon`` fully
       local so the real FFT can be applied.
    2. **Real FFT** along the (now local) longitude dimension.
    3. **Azimuth transpose** (``N`` ↔ ``mmax``) — redistribute so that spectral
       orders ``m`` are split across azimuth ranks and channels are local again.
    4. **Legendre contraction** — a *distributed* matrix multiply: latitudes stay
       split across the polar group and each rank contracts only the latitudes it
       owns, producing a partial sum over all degrees ``l``.
    5. **Reduce-scatter** over the polar group along ``l`` — completes the
       quadrature sum and leaves the degrees partitioned in one collective.

    The output has shape ``(B, C, lmax_local, mmax_local)`` with spectral modes
    partitioned in the same way as the spatial grid.

    Keeping ``nlat`` distributed is what makes the precomputed Legendre weights
    scale: they are partitioned as ``(mmax_local, lmax, nlat_local)``, so the
    tensor is split across the full process grid rather than replicated over the
    polar group.  The cost is that the quadrature sum is now accumulated across
    ranks, so results are not bitwise identical to the serial transform.

    If ``N < azimuth_group_size``, the leading axis is zero-padded before the
    transposes and the padding is removed afterwards; since the transform is
    linear this is exact.

    .. seealso::
        :class:`torch_harmonics.RealSHT`
            Serial counterpart with full mathematical description and parameter
            documentation.

    Parameters
    ----------
    nlat : int
        Number of latitude points
    nlon : int
        Number of longitude points
    lmax : int
        Maximum spherical harmonic degree
    mmax : int
        Maximum spherical harmonic order
    grid : str
        Grid type (``"equiangular"``, ``"legendre-gauss"``, ``"lobatto"``, ``"equiangular-trapezoidal"``), by default ``"equiangular"``
    norm : str
        Normalization type (``"ortho"``, ``"schmidt"``, ``"unnorm"``), by default ``"ortho"``
    csphase : bool
        Whether to apply the Condon-Shortley phase factor, by default True

    Returns
    -------
    torch.Tensor
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
        self.lmax_local = self.l_shapes[self.comm_rank_polar]
        self.m_shapes = compute_split_shapes(self.mmax, self.comm_size_azimuth)
        self.mmax_local = self.m_shapes[self.comm_rank_azimuth]
        self.lat_offset = sum(self.lat_shapes[: self.comm_rank_polar])
        self.mmax_offset = sum(self.m_shapes[: self.comm_rank_azimuth])

        # fold the 2*pi longitudinal scale factor of the forward-normalized FFT into the
        # quadrature weights. It is a constant prefactor of a linear transform, so folding it
        # here is exact and saves a pointwise multiply on a complex tensor in every forward.
        weights = 2.0 * torch.pi * weights

        # build only the block this rank keeps, rather than the whole table. The contraction
        # over k is a distributed matmul completed by a reduce-scatter in the forward, so only
        # the local latitudes are needed; l is contracted in full. Latitudes restrict by simply
        # passing fewer evaluation points -- they are independent of each other -- whereas the
        # order range needs mmin, since reaching P^m_m means walking the seed up from m=0.
        tq_local = tq[self.lat_offset : self.lat_offset + self.nlat_local]
        weights = weights[self.lat_offset : self.lat_offset + self.nlat_local]

        # combine quadrature weights with the legendre weights
        pct = _precompute_legpoly(self.mmax_offset + self.mmax_local, self.lmax, tq_local, norm=self.norm, csphase=self.csphase, mmin=self.mmax_offset)
        weights = torch.einsum("mlk,k->mlk", pct, weights).contiguous()

        # remember quadrature weights
        self.register_buffer("weights", weights, persistent=False)

    def extra_repr(self):
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    # This transform cannot be captured in a single graph: the redistribution collectives it
    # calls are themselves torch.compiler.disable()d, so at comm_size > 1 dynamo breaks at
    # every one of them and inductor only ever sees the slivers in between. Those slivers hold
    # complex intermediates, which triton cannot type (KeyError: 'complex64' in codegen), and
    # compiling them buys nothing next to the all-to-alls surrounding them. Disabling the whole
    # forward costs no fusion that the breaks had not already cost, and keeps the complex
    # spectral data out of inductor entirely. The serial transforms are compiled as usual.
    @torch.compiler.disable()
    def forward(self, x: torch.Tensor):

        check(x.dim() >= 3, lambda: f"Expected tensor with at least 3 dimensions but got {x.dim()} instead")
        check(x.shape[-2] == self.nlat_local, lambda: f"Expected latitudes shape[-2]=={self.nlat_local}, got {x.shape[-2]}")
        check(x.shape[-1] == self.nlon_local, lambda: f"Expected longitudes shape[-1]=={self.nlon_local}, got {x.shape[-1]}")

        # the azimuth transposes below redistribute the leading (channel/batch) axis, so it
        # must be at least as large as the azimuth group. Flatten all leading dims into that
        # axis and zero-pad it if needed (linear transform, so padding stays zero); restore
        # the original layout before returning. The polar group never touches this axis --
        # the latitude contraction is a distributed matmul, not a transpose.
        x, lead_shape, lead_size = flatten_and_pad_leading_dims(x, self.comm_size_azimuth)
        num_chans = x.shape[-3]

        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_azimuth > 1:
            x = distributed_transpose_azimuth(x, (-3, -1), self.lon_shapes)

        # apply real fft in the longitudinal direction: make sure to truncate to nlon. The 2*pi
        # scale factor is folded into the quadrature weights, so no scaling is needed here.
        x = rfft(x, nmodes=self.mmax, dim=-1, norm="forward")

        # transpose: after this, m is split and c is local
        if self.comm_size_azimuth > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = distributed_transpose_azimuth(x, (-1, -3), chan_shapes)

        # transpose to put the contraction dim (nlat) on the fast axis. nlat stays split across
        # the polar group: each rank contracts the latitudes it owns and the reduce-scatter
        # below completes the sum.
        x = x.transpose(-1, -2)
        x_re = x.real.contiguous()
        x_im = x.imag.contiguous()

        # Legendre-Gauss quadrature: partial contraction over the local k=nlat chunk
        w = self.weights.to(x_re.dtype)
        out_re = torch.einsum("...mk,mlk->...lm", x_re, w)
        out_im = torch.einsum("...mk,mlk->...lm", x_im, w)

        # complete the quadrature sum and split l in a single collective. This runs on the real
        # view: the reduce-scatter reduces in fp32, which would discard the imaginary part of a
        # complex tensor, and NCCL has no complex reduction to begin with.
        out = torch.stack((out_re, out_im), dim=-1)
        if self.comm_size_polar > 1:
            out = reduce_from_scatter_to_polar_region(out, -3)
        x = torch.view_as_complex(out.contiguous())

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

    1. **Legendre synthesis** — a *distributed* matrix multiply: degrees stay
       split across the polar group and each rank synthesizes from the degrees it
       owns, producing a partial sum over all latitudes.
    2. **Reduce-scatter** over the polar group along ``nlat`` — completes the
       synthesis sum and leaves the latitudes partitioned in one collective.
    3. **Azimuth transpose** (``N`` ↔ ``mmax``) — make spectral orders ``m``
       fully local for the inverse FFT.
    4. **Inverse real FFT** along the (now local) ``m`` / longitude dimension.
    5. **Azimuth transpose** (``nlon`` ↔ ``N``) — redistribute so that
       longitudes are split across azimuth ranks.

    The output has shape ``(B, C, nlat_local, nlon_local)`` with the spatial
    grid partitioned in the same way as the input spectral modes.

    Keeping ``l`` distributed is what makes the precomputed Legendre polynomials
    scale: they are partitioned as ``(mmax_local, nlat, lmax_local)``, so the
    tensor is split across the full process grid rather than replicated over the
    polar group.  The cost is that the synthesis sum is now accumulated across
    ranks, so results are not bitwise identical to the serial transform.

    If ``N < azimuth_group_size``, the leading axis is zero-padded before the
    transposes and the padding is removed afterwards; since the transform is
    linear this is exact.

    .. seealso::
        :class:`torch_harmonics.InverseRealSHT`
            Serial counterpart with full mathematical description and parameter
            documentation.

    Parameters
    ----------
    nlat : int
        Number of latitude points
    nlon : int
        Number of longitude points
    lmax : int
        Maximum spherical harmonic degree
    mmax : int
        Maximum spherical harmonic order
    grid : str
        Grid type (``"equiangular"``, ``"legendre-gauss"``, ``"lobatto"``, ``"equiangular-trapezoidal"``), by default ``"equiangular"``
    norm : str
        Normalization type (``"ortho"``, ``"schmidt"``, ``"unnorm"``), by default ``"ortho"``
    csphase : bool
        Whether to apply the Condon-Shortley phase factor, by default True

    Returns
    -------
    torch.Tensor
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
        self.lmax_offset = sum(self.l_shapes[: self.comm_rank_polar])
        self.mmax_offset = sum(self.m_shapes[: self.comm_rank_azimuth])

        # build only the block this rank keeps. The synthesis over l is a distributed matmul
        # completed by a reduce-scatter in the forward, so only the local degrees are needed,
        # while all latitudes are produced. Both ranges need an explicit offset: the sectoral
        # seed couples orders and the three-term recurrence couples degrees, so each has to be
        # walked from the start even though only the local window is stored.
        # store as (mmax_local, nlat, lmax_local) so the contraction dim l is stride-1
        pct = _precompute_legpoly(
            self.mmax_offset + self.mmax_local,
            self.lmax_offset + self.lmax_local,
            t,
            norm=self.norm,
            inverse=True,
            csphase=self.csphase,
            mmin=self.mmax_offset,
            lmin=self.lmax_offset,
        )
        pct = pct.permute(0, 2, 1).contiguous()

        # register
        self.register_buffer("pct", pct, persistent=False)

    def extra_repr(self):
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    # This transform cannot be captured in a single graph: the redistribution collectives it
    # calls are themselves torch.compiler.disable()d, so at comm_size > 1 dynamo breaks at
    # every one of them and inductor only ever sees the slivers in between. Those slivers hold
    # complex intermediates, which triton cannot type (KeyError: 'complex64' in codegen), and
    # compiling them buys nothing next to the all-to-alls surrounding them. Disabling the whole
    # forward costs no fusion that the breaks had not already cost, and keeps the complex
    # spectral data out of inductor entirely. The serial transforms are compiled as usual.
    @torch.compiler.disable()
    def forward(self, x: torch.Tensor):

        check(x.dim() >= 3, lambda: f"Expected tensor with at least 3 dimensions but got {x.dim()} instead")
        check(x.shape[-2] == self.lmax_local, lambda: f"Expected spherical harmonic degrees (lmax) shape[-2]=={self.lmax_local}, got {x.shape[-2]}")
        check(x.shape[-1] == self.mmax_local, lambda: f"Expected spherical harmonic orders (mmax) shape[-1]=={self.mmax_local}, got {x.shape[-1]}")

        # the azimuth transposes below redistribute the leading (channel/batch) axis, so it
        # must be at least as large as the azimuth group. Flatten all leading dims into that
        # axis and zero-pad it if needed (linear transform, so padding stays zero); restore
        # the original layout before returning. The polar group never touches this axis --
        # the degree contraction is a distributed matmul, not a transpose.
        x, lead_shape, lead_size = flatten_and_pad_leading_dims(x, self.comm_size_azimuth)
        num_chans = x.shape[-3]

        # transpose to put the contraction dim (lmax) on the fast axis. l stays split across
        # the polar group: each rank synthesizes from the degrees it owns and the
        # reduce-scatter below completes the sum.
        x = x.transpose(-1, -2)
        x_re = x.real.contiguous()
        x_im = x.imag.contiguous()

        # legendre transformation: partial contraction over the local l chunk
        # pct layout: (mmax_local, nlat, lmax_local)
        w = self.pct.to(x_re.dtype)
        out_re = torch.einsum("...ml,mkl->...km", x_re, w)
        out_im = torch.einsum("...ml,mkl->...km", x_im, w)

        # complete the synthesis sum and split nlat in a single collective, on the real view
        # (see DistributedRealSHT.forward for why the collective cannot take complex input).
        out = torch.stack((out_re, out_im), dim=-1)
        if self.comm_size_polar > 1:
            out = reduce_from_scatter_to_polar_region(out, -3)
        x = torch.view_as_complex(out.contiguous())

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
    nlat : int
        Number of latitude points
    nlon : int
        Number of longitude points
    lmax : int
        Maximum spherical harmonic degree
    mmax : int
        Maximum spherical harmonic order
    grid : str
        Grid type (``"equiangular"``, ``"legendre-gauss"``, ``"lobatto"``, ``"equiangular-trapezoidal"``), by default ``"equiangular"``
    norm : str
        Normalization type (``"ortho"``, ``"schmidt"``, ``"unnorm"``), by default ``"ortho"``
    csphase : bool
        Whether to apply the Condon-Shortley phase factor, by default True

    Returns
    -------
    torch.Tensor
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
        self.lat_offset = sum(self.lat_shapes[: self.comm_rank_polar])
        self.mmax_offset = sum(self.m_shapes[: self.comm_rank_azimuth])

        # build only the block this rank keeps: local latitudes, local orders, all degrees,
        # see DistributedRealSHT.__init__
        tq_local = tq[self.lat_offset : self.lat_offset + self.nlat_local]
        weights = weights[self.lat_offset : self.lat_offset + self.nlat_local]

        # compute weights
        dpct = _precompute_dlegpoly(self.mmax_offset + self.mmax_local, self.lmax, tq_local, norm=self.norm, csphase=self.csphase, mmin=self.mmax_offset)

        # fold the 2*pi longitudinal scale factor of the forward-normalized FFT into the
        # quadrature weights (see DistributedRealSHT.__init__)
        weights = 2.0 * torch.pi * weights

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

    # This transform cannot be captured in a single graph: the redistribution collectives it
    # calls are themselves torch.compiler.disable()d, so at comm_size > 1 dynamo breaks at
    # every one of them and inductor only ever sees the slivers in between. Those slivers hold
    # complex intermediates, which triton cannot type (KeyError: 'complex64' in codegen), and
    # compiling them buys nothing next to the all-to-alls surrounding them. Disabling the whole
    # forward costs no fusion that the breaks had not already cost, and keeps the complex
    # spectral data out of inductor entirely. The serial transforms are compiled as usual.
    @torch.compiler.disable()
    def forward(self, x: torch.Tensor):

        check(x.dim() >= 4, lambda: f"Expected tensor with at least 4 dimensions but got {x.dim()} instead")
        check(x.shape[-3] == 2, lambda: f"Expected vector field shape[-3]==2, got {x.shape[-3]}")
        check(x.shape[-2] == self.nlat_local, lambda: f"Expected latitudes shape[-2]=={self.nlat_local}, got {x.shape[-2]}")
        check(x.shape[-1] == self.nlon_local, lambda: f"Expected longitudes shape[-1]=={self.nlon_local}, got {x.shape[-1]}")

        # the transposes below redistribute the leading (channel/batch) axis across the
        # process grid, so it must be at least as large as the larger comm group. Flatten
        # all leading dims into that axis -- keeping the trailing (2, nlat, nlon) intact --
        # and zero-pad it if needed (linear transform, so padding stays zero); restore the
        # original layout before returning.
        x, lead_shape, lead_size = flatten_and_pad_leading_dims(x, self.comm_size_azimuth, num_trailing_dims=3)
        num_chans = x.shape[-4]

        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_azimuth > 1:
            x = distributed_transpose_azimuth(x, (-4, -1), self.lon_shapes)

        # apply real fft in the longitudinal direction: make sure to truncate to nlon. The 2*pi
        # scale factor is folded into the quadrature weights, so no scaling is needed here.
        x = rfft(x, nmodes=self.mmax, dim=-1, norm="forward")

        # transpose: after this, m is split and c is local
        if self.comm_size_azimuth > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = distributed_transpose_azimuth(x, (-1, -4), chan_shapes)

        # transpose to put the contraction dim (nlat) on the fast axis. nlat stays split across
        # the polar group, see DistributedRealSHT.forward.
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

        # stack the components in real space, see RealVectorSHT.forward
        out_re = torch.stack((s_re, t_re), dim=-3)
        out_im = torch.stack((s_im, t_im), dim=-3)

        # complete the quadrature sum and split l in a single collective, on the real view
        # (see DistributedRealSHT.forward for why the collective cannot take complex input).
        out = torch.stack((out_re, out_im), dim=-1)
        if self.comm_size_polar > 1:
            out = reduce_from_scatter_to_polar_region(out, -3)
        x = torch.view_as_complex(out.contiguous())

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
    nlat : int
        Number of latitude points
    nlon : int
        Number of longitude points
    lmax : int
        Maximum spherical harmonic degree
    mmax : int
        Maximum spherical harmonic order
    grid : str
        Grid type (``"equiangular"``, ``"legendre-gauss"``, ``"lobatto"``, ``"equiangular-trapezoidal"``), by default ``"equiangular"``
    norm : str
        Normalization type (``"ortho"``, ``"schmidt"``, ``"unnorm"``), by default ``"ortho"``
    csphase : bool
        Whether to apply the Condon-Shortley phase factor, by default True

    Returns
    -------
    torch.Tensor
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

        self.lmax_offset = sum(self.l_shapes[: self.comm_rank_polar])
        self.mmax_offset = sum(self.m_shapes[: self.comm_rank_azimuth])

        # build only the block this rank keeps: local orders, local degrees, all latitudes,
        # see DistributedInverseRealSHT.__init__
        # store as (2, mmax_local, nlat, lmax_local) so the contraction dim l is stride-1
        dpct = _precompute_dlegpoly(
            self.mmax_offset + self.mmax_local,
            self.lmax_offset + self.lmax_local,
            t,
            norm=self.norm,
            inverse=True,
            csphase=self.csphase,
            mmin=self.mmax_offset,
            lmin=self.lmax_offset,
        )
        dpct = dpct.permute(0, 1, 3, 2).contiguous()

        # register buffer
        self.register_buffer("dpct", dpct, persistent=False)

    def extra_repr(self):
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    # This transform cannot be captured in a single graph: the redistribution collectives it
    # calls are themselves torch.compiler.disable()d, so at comm_size > 1 dynamo breaks at
    # every one of them and inductor only ever sees the slivers in between. Those slivers hold
    # complex intermediates, which triton cannot type (KeyError: 'complex64' in codegen), and
    # compiling them buys nothing next to the all-to-alls surrounding them. Disabling the whole
    # forward costs no fusion that the breaks had not already cost, and keeps the complex
    # spectral data out of inductor entirely. The serial transforms are compiled as usual.
    @torch.compiler.disable()
    def forward(self, x: torch.Tensor):

        check(x.dim() >= 4, lambda: f"Expected tensor with at least 4 dimensions but got {x.dim()} instead")
        check(x.shape[-3] == 2, lambda: f"Expected vector field shape[-3]==2, got {x.shape[-3]}")
        check(x.shape[-2] == self.lmax_local, lambda: f"Expected spherical harmonic degrees (lmax) shape[-2]=={self.lmax_local}, got {x.shape[-2]}")
        check(x.shape[-1] == self.mmax_local, lambda: f"Expected spherical harmonic orders (mmax) shape[-1]=={self.mmax_local}, got {x.shape[-1]}")

        # the transposes below redistribute the leading (channel/batch) axis across the
        # process grid, so it must be at least as large as the larger comm group. Flatten
        # all leading dims into that axis -- keeping the trailing (2, lmax, mmax) intact --
        # and zero-pad it if needed (linear transform, so padding stays zero); restore the
        # original layout before returning.
        x, lead_shape, lead_size = flatten_and_pad_leading_dims(x, self.comm_size_azimuth, num_trailing_dims=3)
        num_chans = x.shape[-4]

        # transpose to put the contraction dim (lmax) on the fast axis. l stays split across
        # the polar group, see DistributedInverseRealSHT.forward.
        x = x.transpose(-1, -2)
        x_re = x.real.contiguous()
        x_im = x.imag.contiguous()

        # dpct layout: (2, mmax_local, nlat, lmax_local) — contract over l (stride-1 in both operands)
        d0 = self.dpct[0].to(x_re.dtype)
        d1 = self.dpct[1].to(x_re.dtype)

        # contraction - spheroidal component
        srl = torch.einsum("...ml,mkl->...km", x_re[..., 0, :, :], d0) - torch.einsum("...ml,mkl->...km", x_im[..., 1, :, :], d1)
        sim = torch.einsum("...ml,mkl->...km", x_im[..., 0, :, :], d0) + torch.einsum("...ml,mkl->...km", x_re[..., 1, :, :], d1)

        # contraction - toroidal component
        trl = -torch.einsum("...ml,mkl->...km", x_im[..., 0, :, :], d1) - torch.einsum("...ml,mkl->...km", x_re[..., 1, :, :], d0)
        tim = torch.einsum("...ml,mkl->...km", x_re[..., 0, :, :], d1) - torch.einsum("...ml,mkl->...km", x_im[..., 1, :, :], d0)

        # reassemble in real space, see RealVectorSHT.forward
        out_re = torch.stack((srl, trl), dim=-3)
        out_im = torch.stack((sim, tim), dim=-3)

        # complete the synthesis sum and split nlat in a single collective, on the real view
        # (see DistributedRealSHT.forward for why the collective cannot take complex input).
        out = torch.stack((out_re, out_im), dim=-1)
        if self.comm_size_polar > 1:
            out = reduce_from_scatter_to_polar_region(out, -3)
        x = torch.view_as_complex(out.contiguous())

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
