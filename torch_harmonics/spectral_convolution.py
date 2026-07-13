# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
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

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from torch_harmonics import InverseRealSHT, RealSHT
from torch_harmonics.quadrature import QuadratureS2
from torch_harmonics.truncation import truncate_sht


class SpectralConvS2(nn.Module):
    r"""
    Spectral convolution layer on :math:`S^2` implemented via real SHT
    (Driscoll--Healy formulation, see https://api.semanticscholar.org/CorpusID:122817218).

    Given a multi-channel input signal :math:`u^{c_i}(\theta, \lambda)` on the
    sphere, the layer computes the output channels :math:`v^{c_o}(\theta, \lambda)`
    in three steps:

    1. **Forward SHT** (cf. :class:`~torch_harmonics.RealSHT`) -- transform
       each input channel to spectral space:

    .. math::

        \hat{u}_l^{m,\,c_i} = \text{SHT}\!\left[\, u^{c_i}(\theta, \lambda) \,\right]

    2. **Spectral contraction** -- mix channels with learnable weights
       :math:`K_l^{c_o,\,c_i}` that are diagonal in :math:`(l, m)` (i.e.\ the
       same weight is applied to every order :math:`m` at a given degree
       :math:`l`):

    .. math::

        \hat{v}_l^{m,\,c_o}
            = \sum_{c_i} K_l^{c_o,\,c_i}\; \hat{u}_l^{m,\,c_i}

    3. **Inverse SHT** (cf. :class:`~torch_harmonics.InverseRealSHT`) --
       transform back to the spatial domain:

    .. math::

        v^{c_o}(\theta, \lambda)
            = \text{ISHT}\!\left[\, \hat{v}_l^{m,\,c_o} \,\right]

    Because the spectral weights depend only on degree :math:`l` and not on order
    :math:`m`, this corresponds to an **isotropic** (azimuthally symmetric)
    convolution kernel on the sphere.  When ``num_groups > 1``, the channel
    contraction is performed independently within each group (grouped
    convolution).

    **Spectral bias.**
    When ``bias=True``, a learnable spectral bias :math:`b_l^{m,\,c_i}` is
    added to the SHT coefficients before the channel contraction.  The bias is
    modulated by the spatial integral (zeroth moment) of each input channel:

    .. math::

        I^{c_i} = \int_0^{2\pi}\!\int_0^{\pi}
            u^{c_i}(\theta,\lambda)\,\sin\theta\;d\theta\;d\lambda

    .. math::

        \hat{u}_l^{m,\,c_i} \;\leftarrow\;
            \hat{u}_l^{m,\,c_i} + I^{c_i}\, b_l^{m,\,c_i}

    This allows the layer to learn a spectral response that depends on the
    global mean of each input channel, effectively coupling the zero-frequency
    content into all spectral modes.

    Parameters
    -----------
    in_shape: Tuple[int]
        Spatial input grid shape ``(nlat, nlon)``.
    out_shape: Tuple[int]
        Spatial output grid shape ``(nlat, nlon)``.
    in_channels: int
        Number of input channels.
    out_channels: int
        Number of output channels.
    num_groups: int, optional
        Number of channel groups for grouped spectral weights, by default 1.
    grid_in: str, optional
        Grid used for the forward SHT (``"equiangular"``, ``"legendre-gauss"``,
        ``"lobatto"``, ``"equiangular-trapezoidal"``), by default ``"equiangular"``.
    grid_out: str, optional
        Grid used for the inverse SHT, same options as ``grid_in``.
    bias: bool, optional
        If ``True``, adds a learnable spectral bias computed from the spatial
        integral, by default ``False``.

    Examples
    --------
    >>> import torch
    >>> import torch_harmonics as th
    >>> conv = th.SpectralConvS2(
    ...     in_shape=(128, 256), out_shape=(128, 256),
    ...     in_channels=16, out_channels=32,
    ... ).cuda()
    >>> x = torch.randn(4, 16, 128, 256, device="cuda")
    >>> y = conv(x)
    >>> y.shape
    torch.Size([4, 32, 128, 256])

    Raises
    ------
    AssertionError
        If ``in_channels`` or ``out_channels`` is not divisible by
        ``num_groups``.

    Notes
    -----
    The SHT truncation ``lmax``/``mmax`` is the minimum of the input and output
    truncations.
    """

    def __init__(
        self,
        in_shape: Tuple[int],
        out_shape: Tuple[int],
        in_channels: int,
        out_channels: int,
        num_groups: Optional[int] = 1,
        grid_in: Optional[str] = "equiangular",
        grid_out: Optional[str] = "equiangular",
        bias: Optional[bool] = False,
    ):
        super().__init__()

        if in_channels % num_groups != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by num_groups ({num_groups})")
        if out_channels % num_groups != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by num_groups ({num_groups})")

        # copy inputs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups

        # compute truncation
        lmax_in, mmax_in = truncate_sht(in_shape[0], in_shape[1], grid=grid_in)
        lmax_out, mmax_out = truncate_sht(out_shape[0], out_shape[1], grid=grid_out)

        # compute lmax and lmin
        lmax = min(lmax_in, lmax_out)
        mmax = min(mmax_in, mmax_out)
        self.lmax = min(lmax, mmax)
        self.mmax = self.lmax

        # set up sht layers
        self.sht = RealSHT(*in_shape, grid=grid_in, lmax=self.lmax, mmax=self.mmax)
        self.isht = InverseRealSHT(*out_shape, grid=grid_out, lmax=self.lmax, mmax=self.mmax)

        # weight shape
        weight_shape = [num_groups, in_channels // num_groups, out_channels // num_groups, self.lmax]

        # Compute scaling factor for correct initialization
        scale = math.sqrt(1.0 / (in_channels // num_groups)) * torch.ones(self.lmax, dtype=torch.complex64)
        # seemingly the first weight is not really complex, so we need to account for that
        scale[0] *= math.sqrt(2.0)
        self.weight = nn.Parameter(scale * torch.randn(*weight_shape, dtype=torch.complex64))

        if bias:
            self.spectral_bias = nn.Parameter(torch.zeros(1, self.in_channels, self.lmax, self.mmax, dtype=torch.complex64))
            self.quadrature = QuadratureS2(img_shape=in_shape, grid=grid_in, normalize=False)

    @torch.compile
    def _contract_lwise(self, ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
        resc = torch.einsum("bgixy,giox->bgoxy", ac, bc)
        return resc

    def forward(self, x):
        """
        Apply the spectral convolution.

        Parameters
        ----------
        x: torch.Tensor
            Input signal of shape ``(batch, in_channels, nlat_in, nlon_in)``.

        Returns
        -------
        torch.Tensor
            Convolved signal of shape ``(batch, out_channels, nlat_out, nlon_out)``.
        """
        dtype = x.dtype

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = x.to(torch.float32)

            # compute integral in case if bias is used
            if hasattr(self, "spectral_bias"):
                integral = self.quadrature(x)

            # perform SHT
            x = self.sht(x).contiguous()

        # store the shapes
        B, C, H, W = x.shape

        # deal with bias
        if hasattr(self, "spectral_bias"):
            x = x + integral.reshape(B, C, 1, 1) * self.spectral_bias

        # perform contraction
        x = x.reshape(B, self.num_groups, C // self.num_groups, H, W)
        xp = self._contract_lwise(x, self.weight)
        x = xp.reshape(B, self.out_channels, H, W).contiguous()

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = self.isht(x)

        # convert datatype
        x = x.to(dtype=dtype)

        return x
