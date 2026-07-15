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


class RealSHT(nn.Module):
    r"""
    Defines a module for computing the forward (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last two dimensions of the input.

    Given a real-valued signal :math:`f(\theta, \lambda)` sampled on the sphere,
    the forward scalar SHT computes the spherical harmonic coefficients via a
    longitudinal FFT followed by Legendre quadrature:

    .. math::

        \hat{f}_l^m = 2\pi \sum_{k=0}^{N_\theta - 1}
            \tilde{f}_m(\theta_k)\, P_l^m(\cos\theta_k)\, q_k

    where :math:`\tilde{f}_m` are the Fourier modes and :math:`q_k` are the
    quadrature weights.

    .. seealso::
        :doc:`/guide/spherical_harmonic_transforms`
            User guide with the full mathematical derivation, normalization
            conventions, grid types, and worked examples.

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
        Grid type (``"equiangular"``, ``"legendre-gauss"``, ``"lobatto"``,
        ``"equiangular-trapezoidal"``), by default ``"equiangular"``
    norm: str
        Normalization convention (``"ortho"``, ``"schmidt"``, ``"unnorm"``),
        by default ``"ortho"``.
    csphase: bool
        Whether to include the Condon--Shortley phase factor :math:`(-1)^m`,
        by default ``True``.

    Examples
    --------
    >>> import torch
    >>> import torch_harmonics as th
    >>> nlat, nlon = 128, 256
    >>> sht = th.RealSHT(nlat, nlon).cuda()
    >>> signal = torch.randn(1, nlat, nlon, device="cuda")
    >>> coeffs = sht(signal)   # shape (1, lmax, mmax), complex
    >>> coeffs.shape
    torch.Size([1, 128, 129])

    .. note::
        This module uses **cuFFT** (via :func:`torch.fft.rfft`) to compute the
        longitudinal Fourier transform efficiently.  When running in **float16** or
        **bfloat16** precision, cuFFT requires the transformed dimension (``nlon``)
        to be a **power of two**.  If your grid does not satisfy this constraint and
        the module is called inside a :class:`torch.autocast` context, guard it with
        ``torch.autocast(device_type="cuda", enabled=False)``::

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # ... other half-precision work ...
                with torch.autocast(device_type="cuda", enabled=False):
                    coeffs = sht(signal.float())

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

        # compute quadrature points and lmax based on the exactness of the quadrature
        if self.grid == "legendre-gauss":
            cost, weights = legendre_gauss_weights(nlat, -1, 1)
        elif self.grid == "lobatto":
            cost, weights = lobatto_weights(nlat, -1, 1)
        elif self.grid == "equiangular":
            cost, weights = clenshaw_curtiss_weights(nlat, -1, 1)
        else:
            raise (ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        tq = torch.flip(torch.arccos(cost), dims=(0,))

        # determine maximum degrees based on triangular truncation
        self.lmax, self.mmax = truncate_sht(self.nlat, self.nlon, lmax, mmax, self.grid)

        # combine quadrature weights with the legendre weights
        pct = _precompute_legpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)
        weights = torch.einsum("mlk,k->mlk", pct, weights).contiguous()

        # remember quadrature weights
        self.register_buffer("weights", weights, persistent=False)

    def extra_repr(self):
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    def forward(self, x: torch.Tensor):
        """
        Compute the forward (real) spherical harmonic transform.

        Parameters
        ----------
        x: torch.Tensor
            Real-valued signal on the sphere of shape ``(..., nlat, nlon)``.

        Returns
        -------
        torch.Tensor
            Complex spherical harmonic coefficients of shape ``(..., lmax, mmax)``.
        """

        torch._check(x.dim() >= 2, lambda: f"Expected tensor with at least 2 dimensions but got {x.dim()} instead")
        torch._check(x.shape[-2] == self.nlat, lambda: f"Expected latitudes shape[-2]=={self.nlat}, got {x.shape[-2]}")
        torch._check(x.shape[-1] == self.nlon, lambda: f"Expected longitudes shape[-1]=={self.nlon}, got {x.shape[-1]}")

        # apply real fft in the longitudinal direction
        x = 2.0 * torch.pi * rfft(x, nmodes=self.mmax, dim=-1, norm="forward")

        # transpose to put the contraction dim (nlat) on the fast axis
        x = x.transpose(-1, -2)
        x_re = x.real.contiguous()
        x_im = x.imag.contiguous()

        # Legendre-Gauss quadrature: contract over k=nlat (stride-1 in both operands)
        w = self.weights.to(x_re.dtype)
        out_re = torch.einsum("...mk,mlk->...lm", x_re, w)
        out_im = torch.einsum("...mk,mlk->...lm", x_im, w)

        # the ...lm einsum output is non-contiguous (l ends up stride-1, m slow); torch.complex
        # preserves those strides, but inductor's meta kernel for aten.complex predicts a contiguous
        # layout, tripping assert_size_stride under torch.compile(dynamic=False). Force contiguous.
        return torch.complex(out_re.contiguous(), out_im.contiguous())


class InverseRealSHT(nn.Module):
    r"""
    Defines a module for computing the inverse (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.

    Given complex spherical harmonic coefficients :math:`\hat{f}_l^m`, the inverse
    scalar SHT reconstructs the real-valued signal on the sphere via Legendre
    synthesis followed by an inverse FFT:

    .. math::

        f(\theta, \lambda) = \sum_{l=0}^{l_{\max}-1} \sum_{m=0}^{m_{\max}-1}
            \hat{f}_l^m\, Y_l^m(\theta, \lambda)

    .. seealso::
        :doc:`/guide/spherical_harmonic_transforms`
            User guide with the full mathematical derivation, normalization
            conventions, grid types, and worked examples.

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
        Grid type (``"equiangular"``, ``"legendre-gauss"``, ``"lobatto"``,
        ``"equiangular-trapezoidal"``), by default ``"equiangular"``
    norm: str
        Normalization convention (``"ortho"``, ``"schmidt"``, ``"unnorm"``),
        by default ``"ortho"``.
    csphase: bool
        Whether to include the Condon--Shortley phase factor :math:`(-1)^m`,
        by default ``True``.

    Examples
    --------
    >>> import torch
    >>> import torch_harmonics as th
    >>> nlat, nlon = 128, 256
    >>> isht = th.InverseRealSHT(nlat, nlon).cuda()
    >>> coeffs = torch.randn(1, 128, 129, dtype=torch.cfloat, device="cuda")
    >>> signal = isht(coeffs)   # shape (1, 128, 256), real
    >>> signal.shape
    torch.Size([1, 128, 256])

    .. note::
        This module uses **cuFFT** (via :func:`torch.fft.irfft`) to compute the
        longitudinal inverse Fourier transform efficiently.  When running in
        **float16** or **bfloat16** precision, cuFFT requires the transformed
        dimension (``nlon``) to be a **power of two**.  If your grid does not
        satisfy this constraint and the module is called inside a
        :class:`torch.autocast` context, guard it with
        ``torch.autocast(device_type="cuda", enabled=False)``::

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # ... other half-precision work ...
                with torch.autocast(device_type="cuda", enabled=False):
                    signal = isht(coeffs.to(torch.cfloat))

    .. note::
        The inverse real FFT (C2R transform) expects the DC component (:math:`m = 0`)
        and, when ``nlon`` is even, the Nyquist component (:math:`m = N_\lambda / 2`)
        to be purely real.  This routine zeros out the imaginary parts of these
        components before calling the transform.

    Raises
    ------
    ValueError: If the grid type is unknown

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

        # apply cosine transform and flip them
        t = torch.flip(torch.arccos(cost), dims=(0,))

        # determine maximum degrees based on triangular truncation
        self.lmax, self.mmax = truncate_sht(self.nlat, self.nlon, lmax, mmax, self.grid)

        # precompute associated Legendre polynomials
        # store as (mmax, nlat, lmax) so the contraction dim l is stride-1
        pct = _precompute_legpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase)
        pct = pct.permute(0, 2, 1).contiguous()

        # register buffer
        self.register_buffer("pct", pct, persistent=False)

    def extra_repr(self):
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    def forward(self, x: torch.Tensor):
        """
        Compute the inverse (real) spherical harmonic transform.

        Parameters
        ----------
        x: torch.Tensor
            Complex spherical harmonic coefficients of shape ``(..., lmax, mmax)``.

        Returns
        -------
        torch.Tensor
            Real-valued signal on the sphere of shape ``(..., nlat, nlon)``.
        """

        torch._check(x.dim() >= 2, lambda: f"Expected tensor with at least 2 dimensions but got {x.dim()} instead")
        torch._check(x.shape[-2] == self.lmax, lambda: f"Expected spherical harmonic degrees (lmax) shape[-2]=={self.lmax}, got {x.shape[-2]}")
        torch._check(x.shape[-1] == self.mmax, lambda: f"Expected spherical harmonic orders (mmax) shape[-1]=={self.mmax}, got {x.shape[-1]}")

        # transpose to put the contraction dim (lmax) on the fast axis
        x = x.transpose(-1, -2)
        x_re = x.real.contiguous()
        x_im = x.imag.contiguous()

        # legendre transformation: contract over l=lmax (stride-1 in both operands)
        # pct layout: (mmax, nlat, lmax)
        w = self.pct.to(x_re.dtype)
        out_re = torch.einsum("...ml,mkl->...km", x_re, w)
        out_im = torch.einsum("...ml,mkl->...km", x_im, w)
        # force contiguous: the einsum output is non-contiguous and inductor's aten.complex meta
        # predicts a contiguous layout, tripping assert_size_stride under torch.compile (see fwd SHT).
        x = torch.complex(out_re.contiguous(), out_im.contiguous())

        # apply the inverse (real) FFT
        x = irfft(x, n=self.nlon, dim=-1, norm="forward")

        return x


class RealVectorSHT(nn.Module):
    r"""
    Defines a module for computing the forward (real) vector SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last three dimensions of the input.

    Decomposes a tangential vector field
    :math:`\mathbf{v} = v_\theta\,\hat{e}_\theta + v_\lambda\,\hat{e}_\lambda`
    into **spheroidal** and **toroidal** spectral coefficients
    :math:`\hat{s}_l^m` and :math:`\hat{t}_l^m` using the derivatives of the
    associated Legendre polynomials.

    .. seealso::
        :doc:`/guide/spherical_harmonic_transforms`
            User guide with the full mathematical derivation of the vector SHT
            formulas, normalization conventions, and worked examples.

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
        Grid type (``"equiangular"``, ``"legendre-gauss"``, ``"lobatto"``,
        ``"equiangular-trapezoidal"``), by default ``"equiangular"``
    norm: str
        Normalization convention (``"ortho"``, ``"schmidt"``, ``"unnorm"``),
        by default ``"ortho"``.
    csphase: bool
        Whether to include the Condon--Shortley phase factor :math:`(-1)^m`,
        by default ``True``.

    Examples
    --------
    >>> import torch
    >>> import torch_harmonics as th
    >>> nlat, nlon = 128, 256
    >>> vsht = th.RealVectorSHT(nlat, nlon).cuda()
    >>> vector_field = torch.randn(1, 2, nlat, nlon, device="cuda")
    >>> coeffs = vsht(vector_field)   # shape (1, 2, lmax, mmax), complex
    >>> coeffs.shape
    torch.Size([1, 2, 128, 129])

    .. note::
        This module uses **cuFFT** (via :func:`torch.fft.rfft`) to compute the
        longitudinal Fourier transform efficiently.  When running in **float16** or
        **bfloat16** precision, cuFFT requires the transformed dimension (``nlon``)
        to be a **power of two**.  If your grid does not satisfy this constraint and
        the module is called inside a :class:`torch.autocast` context, guard it with
        ``torch.autocast(device_type="cuda", enabled=False)``::

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # ... other half-precision work ...
                with torch.autocast(device_type="cuda", enabled=False):
                    coeffs = vsht(vector_field.float())

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

        # apply cosine transform and flip them
        tq = torch.flip(torch.arccos(cost), dims=(0,))

        # determine maximum degrees based on triangular truncation
        self.lmax, self.mmax = truncate_sht(self.nlat, self.nlon, lmax, mmax, self.grid)

        # precompute associated Legendre polynomials
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
        """
        Compute the forward (real) vector spherical harmonic transform.

        Parameters
        ----------
        x: torch.Tensor
            Real-valued tangential vector field of shape ``(..., 2, nlat, nlon)``, where the
            size-2 dimension holds the two tangential (colatitude, longitude) components.

        Returns
        -------
        torch.Tensor
            Complex vector harmonic coefficients of shape ``(..., 2, lmax, mmax)``, where the
            size-2 dimension holds the spheroidal and toroidal components.
        """

        torch._check(x.dim() >= 3, lambda: f"Expected tensor with at least 3 dimensions but got {x.dim()} instead")
        torch._check(x.shape[-3] == 2, lambda: f"Expected vector field shape[-3]==2, got {x.shape[-3]}")
        torch._check(x.shape[-2] == self.nlat, lambda: f"Expected latitudes shape[-2]=={self.nlat}, got {x.shape[-2]}")
        torch._check(x.shape[-1] == self.nlon, lambda: f"Expected longitudes shape[-1]=={self.nlon}, got {x.shape[-1]}")

        # apply real fft in the longitudinal direction
        x = 2.0 * torch.pi * rfft(x, nmodes=self.mmax, dim=-1, norm="forward")

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

        return torch.stack((torch.complex(s_re, s_im), torch.complex(t_re, t_im)), dim=-3)


class InverseRealVectorSHT(nn.Module):
    r"""
    Defines a module for computing the inverse (real-valued) vector SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.

    Given spheroidal and toroidal spectral coefficients :math:`\hat{s}_l^m` and
    :math:`\hat{t}_l^m`, reconstructs the tangential vector field on the sphere
    via Legendre synthesis with the derivatives of the associated Legendre
    polynomials, followed by an inverse real FFT.

    .. seealso::
        :doc:`/guide/spherical_harmonic_transforms`
            User guide with the full mathematical derivation of the inverse
            vector SHT formulas, normalization conventions, and worked examples.

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
        Grid type (``"equiangular"``, ``"legendre-gauss"``, ``"lobatto"``,
        ``"equiangular-trapezoidal"``), by default ``"equiangular"``
    norm: str
        Normalization convention (``"ortho"``, ``"schmidt"``, ``"unnorm"``),
        by default ``"ortho"``.
    csphase: bool
        Whether to include the Condon--Shortley phase factor :math:`(-1)^m`,
        by default ``True``.

    Examples
    --------
    >>> import torch
    >>> import torch_harmonics as th
    >>> nlat, nlon = 128, 256
    >>> ivsht = th.InverseRealVectorSHT(nlat, nlon).cuda()
    >>> coeffs = torch.randn(1, 2, 128, 129, dtype=torch.cfloat, device="cuda")
    >>> vector_field = ivsht(coeffs)   # shape (1, 2, 128, 256), real
    >>> vector_field.shape
    torch.Size([1, 2, 128, 256])

    .. note::
        This module uses **cuFFT** (via :func:`torch.fft.irfft`) to compute the
        longitudinal inverse Fourier transform efficiently.  When running in
        **float16** or **bfloat16** precision, cuFFT requires the transformed
        dimension (``nlon``) to be a **power of two**.  If your grid does not
        satisfy this constraint and the module is called inside a
        :class:`torch.autocast` context, guard it with
        ``torch.autocast(device_type="cuda", enabled=False)``::

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # ... other half-precision work ...
                with torch.autocast(device_type="cuda", enabled=False):
                    vector_field = ivsht(coeffs.to(torch.cfloat))

    .. note::
        The inverse real FFT (C2R transform) expects the DC component (:math:`m = 0`)
        and, when ``nlon`` is even, the Nyquist component (:math:`m = N_\lambda / 2`)
        to be purely real.  This routine zeros out the imaginary parts of these
        components before calling the transform.

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

        # apply cosine transform and flip them
        t = torch.flip(torch.arccos(cost), dims=(0,))

        # determine maximum degrees based on triangular truncation
        self.lmax, self.mmax = truncate_sht(self.nlat, self.nlon, lmax, mmax, self.grid)

        # precompute associated Legendre polynomials
        # store as (2, mmax, nlat, lmax) so the contraction dim l is stride-1
        dpct = _precompute_dlegpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase)
        dpct = dpct.permute(0, 1, 3, 2).contiguous()

        # register weights
        self.register_buffer("dpct", dpct, persistent=False)

    def extra_repr(self):
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    def forward(self, x: torch.Tensor):
        """
        Compute the inverse (real) vector spherical harmonic transform.

        Parameters
        ----------
        x: torch.Tensor
            Complex vector harmonic coefficients of shape ``(..., 2, lmax, mmax)``, where the
            size-2 dimension holds the spheroidal and toroidal components.

        Returns
        -------
        torch.Tensor
            Real-valued tangential vector field of shape ``(..., 2, nlat, nlon)``, where the
            size-2 dimension holds the two tangential (colatitude, longitude) components.
        """

        torch._check(x.dim() >= 3, lambda: f"Expected tensor with at least 3 dimensions but got {x.dim()} instead")
        torch._check(x.shape[-3] == 2, lambda: f"Expected vector field shape[-3]==2, got {x.shape[-3]}")
        torch._check(x.shape[-2] == self.lmax, lambda: f"Expected spherical harmonic degrees (lmax) shape[-2]=={self.lmax}, got {x.shape[-2]}")
        torch._check(x.shape[-1] == self.mmax, lambda: f"Expected spherical harmonic orders (mmax) shape[-1]=={self.mmax}, got {x.shape[-1]}")

        # transpose to put the contraction dim (lmax) on the fast axis
        x = x.transpose(-1, -2)
        x_re = x.real.contiguous()
        x_im = x.imag.contiguous()

        # dpct layout: (2, mmax, nlat, lmax) — contract over l (stride-1 in both operands)
        d0 = self.dpct[0].to(x_re.dtype)
        d1 = self.dpct[1].to(x_re.dtype)

        # contraction - spheroidal component
        srl = torch.einsum("...ml,mkl->...km", x_re[..., 0, :, :], d0) - torch.einsum("...ml,mkl->...km", x_im[..., 1, :, :], d1)
        sim = torch.einsum("...ml,mkl->...km", x_im[..., 0, :, :], d0) + torch.einsum("...ml,mkl->...km", x_re[..., 1, :, :], d1)

        # contraction - toroidal component
        trl = -torch.einsum("...ml,mkl->...km", x_im[..., 0, :, :], d1) - torch.einsum("...ml,mkl->...km", x_re[..., 1, :, :], d0)
        tim = torch.einsum("...ml,mkl->...km", x_re[..., 0, :, :], d1) - torch.einsum("...ml,mkl->...km", x_im[..., 1, :, :], d0)

        # reassemble and apply inverse FFT
        xs = torch.stack((torch.complex(srl, sim), torch.complex(trl, tim)), dim=-3)
        x = irfft(xs, n=self.nlon, dim=-1, norm="forward")

        return x
