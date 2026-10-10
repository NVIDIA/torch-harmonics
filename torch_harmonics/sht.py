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

from typing import Optional

import torch
import torch.nn as nn

from torch_harmonics.cache import lru_cache
from torch_harmonics.fft import irfft, rfft
from torch_harmonics.grid import EquiangularGrid, RegularGridS2, SpectralGrid, _rejects_legacy_signature, require_regular_grid
from torch_harmonics.legendre import _mask_spectral_block, _precompute_dlegpoly, _precompute_legpoly, dlegpoly, legpoly
from torch_harmonics.truncation import _warn_if_not_spectrally_accurate, truncate_sht
from torch_harmonics.utils import check


def _extended_cc_analysis(grid: RegularGridS2, spectral_grid: SpectralGrid) -> bool:
    """Select extended analysis after the usual spectral support resolution."""
    # Oversized dimensions retain legacy direct quadrature for spectral upsampling.
    # Accurate folded recovery requires both latitude and longitude bandwidth limits.
    return isinstance(grid, EquiangularGrid) and grid.max_exact_degree < spectral_grid.lmax <= grid.nlat - 1 and spectral_grid.mmax <= (grid.nlon + 1) // 2


def _precompute_cc_resampling(nlat: int, mmax: int, vector: bool = False) -> dict[str, torch.Tensor]:
    """Small real buffers for the runtime fold, including the longitude scale."""
    dense_weights = 2.0 * torch.pi * EquiangularGrid(nlat=2 * nlat - 1, nlon=1).colat_weights
    midpoint_weights = dense_weights[1::2]
    frequencies = torch.fft.fftfreq(2 * (nlat - 1), dtype=torch.float64)
    angle = torch.pi * frequencies
    # Store real components so Module.to(dtype=...) preserves the phase.
    phase = torch.stack((angle.cos(), angle.sin()), dim=0)
    # The unpaired Nyquist sample represents a cosine, with equal +/- frequencies.
    # At half-grid positions their phases cancel. Suppressing this bin preserves
    # conjugation for arbitrary inputs; the supported harmonic band has no such mode.
    phase[:, nlat - 1] = 0.0
    signs = torch.ones(mmax, 1, dtype=torch.int8)
    signs[1::2] = -1
    if vector:
        signs = -signs
    return {
        "_cc_weights": dense_weights[::2].contiguous(),
        "_cc_midpoint_weights": torch.cat((midpoint_weights, torch.zeros_like(midpoint_weights))),
        "_cc_phase": phase,
        "_cc_parity": signs,
    }


def _periodic_latitude_extension(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    r"""Extend ``(..., m, nlat)`` modes, reversing only interior rings.

    ``signs`` has shape ``(m, 1)``: scalar parity is :math:`(-1)^m`,
    tangential vector parity is :math:`(-1)^{m+1}`. Each pole occurs once.
    The caller supplies signs for the actual spectral orders being processed.
    """
    return torch.cat((x, signs * x[..., 1:-1].flip(-1)), dim=-1)


def _periodic_latitude_extension_adjoint(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Fold a periodic meridian, adding mirrored interiors without doubling poles."""
    nlat = x.shape[-1] // 2 + 1
    interior = x[..., 1 : nlat - 1] + signs * x[..., nlat:].flip(-1)
    return torch.cat((x[..., :1], interior, x[..., nlat - 1 : nlat]), dim=-1)


def _fourier_shift_latitude(x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Half-grid interpolation; its Hilbert adjoint uses the conjugate phase."""
    return torch.fft.ifft(torch.fft.fft(x, dim=-1) * phase, dim=-1)


def _fold_resampled_latitude(
    x: torch.Tensor,
    signs: torch.Tensor,
    quadrature_weights: torch.Tensor,
    midpoint_weights: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    r"""Apply :math:`U = Q_e + A^* Q_o A` to complex longitude modes.

    ``A`` periodically extends the meridian, shifts it by half a latitude step,
    and selects the first ``nlat - 1`` midpoint rings. Zero-padded midpoint
    weights implement this selection and its adjoint. Both quadratures come
    from the doubled CC grid; neither is the native CC quadrature.
    """
    # FFT backends reject empty batches; an empty order axis has no midpoint contribution.
    if x.shape[-2] == 0:
        return quadrature_weights * x
    shifted = _fourier_shift_latitude(_periodic_latitude_extension(x, signs), phase)
    folded = _fourier_shift_latitude(shifted * midpoint_weights, phase.conj())
    return quadrature_weights * x + _periodic_latitude_extension_adjoint(folded, signs)


@lru_cache(maxsize=2, typed=True, copy=True)
@torch.no_grad()
def _precompute_cc_projection(
    grid: EquiangularGrid,
    truncation: SpectralGrid,
    norm: str,
    csphase: bool,
    vector: bool = False,
    *,
    mmin: int = 0,
    mmax: Optional[int] = None,
    kmin: int = 0,
    kmax: Optional[int] = None,
) -> torch.Tensor:
    r"""Fold doubled-grid quadrature onto real projection rows in bounded blocks.

    Half-grid interpolation with the unpaired Nyquist bin suppressed is real.
    Thus ``U = Q_e + A* Q_o A`` is both real and symmetric, and the bilinear
    coefficient contraction ``P.T @ U @ x`` uses ``W = U.T @ P = U @ P``.
    This also applies to complex longitude modes: no conjugation of ``x`` is
    involved. Both vector derivative rows have parity ``(-1)**(m+1)``.

    Only local orders are generated, on the full meridian; only the requested
    latitude shard is retained. Recurrence and FFT temporaries are bounded by
    order/degree chunks, without caching unfolded tables or dense operators.
    A two-entry cache reuses completed projections; copying keeps module buffers
    independent. Computation and storage use float64 until explicitly cast.
    """
    mmax = truncation.mmax if mmax is None else mmax
    kmax = grid.nlat if kmax is None else kmax
    shape = (mmax - mmin, truncation.lmax, kmax - kmin)
    weights = torch.empty((2, *shape) if vector else shape, dtype=torch.float64)
    if mmax == mmin or truncation.lmax == 0 or kmax == kmin:
        return weights

    buffers = _precompute_cc_resampling(grid.nlat, mmax, vector)
    phase = torch.complex(*buffers["_cc_phase"])
    # At most ~8 MiB of real basis data per block (FFT workspaces are larger).
    order_chunk = min(16, mmax - mmin)
    degree_limit = max(1, 2**20 // (grid.nlat * order_chunk * (2 if vector else 1)))
    # Balance the chunks: a tiny final block would still walk the entire degree
    # recurrence. Equal-sized blocks minimize those repeated recurrence steps.
    degree_chunks = (truncation.lmax + degree_limit - 1) // degree_limit
    degree_chunk = (truncation.lmax + degree_chunks - 1) // degree_chunks
    colats = grid.colats
    for ms in range(mmin, mmax, order_chunk):
        me = min(ms + order_chunk, mmax)
        for ls in range(0, truncation.lmax, degree_chunk):
            le = min(ls + degree_chunk, truncation.lmax)
            if ms >= le:
                # This entire block lies outside the triangular harmonic support.
                weights[..., ms - mmin : me - mmin, ls:le, :].zero_()
                continue
            if vector:
                basis = dlegpoly(me, le, colats, norm=norm, csphase=csphase, mmin=ms, lmin=ls)
                degrees = torch.arange(ls, le, dtype=torch.float64)
                factor = 1.0 / (degrees * (degrees + 1)).clamp(min=1)
                basis *= factor[None, None, :, None]
                basis[1].neg_()
            else:
                basis = legpoly(me, le, colats.cos(), norm=norm, csphase=csphase, mmin=ms, lmin=ls)
            _mask_spectral_block(basis, truncation, ms, ls)
            # The fold expects (..., m, latitude); degree is a batch dimension.
            folded = _fold_resampled_latitude(
                basis.transpose(-3, -2),
                buffers["_cc_parity"][ms:me],
                buffers["_cc_weights"],
                buffers["_cc_midpoint_weights"],
                phase,
            )
            weights[..., ms - mmin : me - mmin, ls:le, :] = folded.real.transpose(-3, -2)[..., kmin:kmax]
    return weights


class RealSHT(nn.Module):
    r"""
    Defines a module for computing the forward (real-valued) SHT.
    Precomputes the associated Legendre polynomials and quadrature weights of the given grid.
    The SHT is applied to the last two dimensions of the input.

    The input and output domains are available as ``grid_in`` and ``grid_out``;
    ``grid`` remains the spatial descriptor.

    Given a real-valued signal :math:`f(\theta, \lambda)` sampled on the sphere,
    the forward scalar SHT computes the spherical harmonic coefficients via a
    longitudinal FFT followed by Legendre quadrature:

    .. math::

        \hat{f}_l^m = 2\pi \sum_{k=0}^{N_\theta - 1}
            \tilde{f}_m(\theta_k)\, P_l^m(\cos\theta_k)\, q_k

    where :math:`\tilde{f}_m` are the Fourier modes and :math:`q_k` are the
    quadrature weights.

    On equiangular grids, ``lmax > grid.max_exact_degree`` selects precomputed
    folded Clenshaw--Curtis analysis following :cite:`Reinecke2023`, Appendix A,
    when the resolved exclusive bounds satisfy ``lmax <= nlat - 1`` and
    ``mmax <= (nlon + 1) // 2``. These are the accurate band-limited recovery
    limits. Oversized dimensions retain legacy direct quadrature without
    implying accurate recovery. The default truncation is unchanged. Extended analysis
    is intended for float32 and float64; its precomputation latitude FFT has length
    ``2 * (nlat - 1)``.

    .. seealso::
        :doc:`/guide/spherical_harmonic_transforms`
            User guide with the full mathematical derivation, normalization
            conventions, grid types, and worked examples.

    Parameters
    ----------
    grid : RegularGridS2
        Descriptor of the spatial grid the transform operates on. It carries the
        resolution as well as the quadrature rule, so no separate ``nlat``/``nlon``
        is needed. Build one with :func:`torch_harmonics.grid.as_grid`.
    lmax : int, optional
        Non-inclusive maximum spherical harmonic degree.
    mmax : int, optional
        Non-inclusive maximum spherical harmonic order.
    norm : str
        Normalization convention (``"ortho"``, ``"schmidt"``, ``"unnorm"``),
        by default ``"ortho"``.
    csphase : bool
        Whether to include the Condon--Shortley phase factor :math:`(-1)^m`,
        by default ``True``.
    lmmax : int, optional
        Non-inclusive upper bound on degree minus order: retain only ``l - m < lmmax``.
        ``None`` leaves this bandwidth unrestricted.

    Examples
    --------
    >>> import torch
    >>> import torch_harmonics as th
    >>> grid = th.as_grid("equiangular", nlat=128, nlon=256)
    >>> sht = th.RealSHT(grid)
    >>> signal = torch.randn(1, grid.nlat, grid.nlon)
    >>> coeffs = sht(signal)   # shape (1, lmax, mmax), complex
    >>> coeffs.shape
    torch.Size([1, 64, 64])

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
    :cite:`Schaeffer2013`, :cite:`Wang2018`, :cite:`Reinecke2023`
    """

    @_rejects_legacy_signature(
        'nlat, nlon, lmax=None, mmax=None, grid="equiangular", norm="ortho", csphase=True',
        grid=("nlat", "nlon"),
    )
    def __init__(
        self,
        grid: RegularGridS2,
        lmax: Optional[int] = None,
        mmax: Optional[int] = None,
        norm: Optional[str] = "ortho",
        csphase: Optional[bool] = True,
        lmmax: Optional[int] = None,
    ):

        super().__init__()

        self.grid = require_regular_grid(grid)
        self.nlat, self.nlon = self.grid.shape
        _warn_if_not_spectrally_accurate(self.grid)
        self.norm = norm
        self.csphase = csphase

        # TODO: include assertions regarding the dimensions

        # quadrature weights come from the grid descriptor, which supports every grid
        # precompute_latitudes does -- the switch this replaced silently rejected
        # "trapezoidal". The nodes are not needed here: _precompute_legpoly takes the
        # descriptor and reads them itself, which is also what keys its cache.
        weights = self.grid.colat_weights

        self._trunc = truncate_sht(self.grid, lmax, mmax, lmmax)
        self._extended_cc = _extended_cc_analysis(self.grid, self._trunc)
        if self._extended_cc:
            weights = _precompute_cc_projection(self.grid, self._trunc, self.norm, self.csphase)
        else:
            # Include the longitude scale of the forward-normalized FFT.
            pct = _precompute_legpoly(self.mmax, self.lmax, self.grid, norm=self.norm, csphase=self.csphase, truncation=self._trunc)
            weights = torch.einsum("mlk,k->mlk", pct, 2.0 * torch.pi * weights).contiguous()

        # remember quadrature weights
        self.register_buffer("weights", weights, persistent=False)

    @property
    def lmax(self) -> int:
        return self._trunc.lmax

    @property
    def mmax(self) -> int:
        return self._trunc.mmax

    @property
    def lmmax(self) -> Optional[int]:
        return self._trunc.lmmax

    @property
    def grid_in(self) -> RegularGridS2:
        """Spatial domain of the input field."""
        return self.grid

    @property
    def grid_out(self) -> SpectralGrid:
        """Spectral coefficient support of the output."""
        return self._trunc

    def extra_repr(self):
        return f"grid={self.grid!r},\nlmax={self.lmax}, mmax={self.mmax}, lmmax={self.lmmax}, csphase={self.csphase}"

    def forward(self, x: torch.Tensor):
        """
        Compute the forward (real) spherical harmonic transform.

        Parameters
        ----------
        x : torch.Tensor
            Real-valued signal on the sphere of shape ``(..., nlat, nlon)``.

        Returns
        -------
        torch.Tensor
            Complex spherical harmonic coefficients of shape ``(..., lmax, mmax)``.
        """

        check(x.dim() >= 2, lambda: f"Expected tensor with at least 2 dimensions but got {x.dim()} instead")
        check(x.shape[-2] == self.nlat, lambda: f"Expected latitudes shape[-2]=={self.nlat}, got {x.shape[-2]}")
        check(x.shape[-1] == self.nlon, lambda: f"Expected longitudes shape[-1]=={self.nlon}, got {x.shape[-1]}")

        # apply real fft in the longitudinal direction. The 2*pi scale factor is folded into
        # the quadrature weights, so no scaling of the complex output is needed here.
        x = rfft(x, nmodes=self.mmax, dim=-1, norm="forward")

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
    Precomputes the associated Legendre polynomials on the nodes of the given grid.

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
    ----------
    grid : RegularGridS2
        Descriptor of the spatial grid the transform operates on. It carries the
        resolution as well as the quadrature rule, so no separate ``nlat``/``nlon``
        is needed. Build one with :func:`torch_harmonics.grid.as_grid`.
    lmax : int, optional
        Non-inclusive maximum spherical harmonic degree.
    mmax : int, optional
        Non-inclusive maximum spherical harmonic order.
    norm : str
        Normalization convention (``"ortho"``, ``"schmidt"``, ``"unnorm"``),
        by default ``"ortho"``.
    csphase : bool
        Whether to include the Condon--Shortley phase factor :math:`(-1)^m`,
        by default ``True``.
    lmmax : int, optional
        Non-inclusive upper bound on degree minus order: retain only ``l - m < lmmax``.
        ``None`` leaves this bandwidth unrestricted.

    Examples
    --------
    >>> import torch
    >>> import torch_harmonics as th
    >>> grid = th.as_grid("equiangular", nlat=128, nlon=256)
    >>> isht = th.InverseRealSHT(grid)
    >>> coeffs = torch.randn(1, isht.lmax, isht.mmax, dtype=torch.cfloat)
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
    TypeError
        If ``grid`` is not a :class:`~torch_harmonics.grid.RegularGridS2`.

    References
    ----------
    :cite:`Schaeffer2013`, :cite:`Wang2018`
    """

    @_rejects_legacy_signature(
        'nlat, nlon, lmax=None, mmax=None, grid="equiangular", norm="ortho", csphase=True',
        grid=("nlat", "nlon"),
    )
    def __init__(
        self,
        grid: RegularGridS2,
        lmax: Optional[int] = None,
        mmax: Optional[int] = None,
        norm: Optional[str] = "ortho",
        csphase: Optional[bool] = True,
        lmmax: Optional[int] = None,
    ):

        super().__init__()

        self.grid = require_regular_grid(grid)
        self.nlat, self.nlon = self.grid.shape
        _warn_if_not_spectrally_accurate(self.grid)
        self.norm = norm
        self.csphase = csphase

        self._trunc = truncate_sht(self.grid, lmax, mmax, lmmax)

        # precompute associated Legendre polynomials
        # store as (mmax, nlat, lmax) so the contraction dim l is stride-1
        pct = _precompute_legpoly(self.mmax, self.lmax, self.grid, norm=self.norm, inverse=True, csphase=self.csphase, truncation=self._trunc)
        pct = pct.permute(0, 2, 1).contiguous()

        # register buffer
        self.register_buffer("pct", pct, persistent=False)

    @property
    def lmax(self) -> int:
        return self._trunc.lmax

    @property
    def mmax(self) -> int:
        return self._trunc.mmax

    @property
    def lmmax(self) -> Optional[int]:
        return self._trunc.lmmax

    @property
    def grid_in(self) -> SpectralGrid:
        """Spectral coefficient support of the input field."""
        return self._trunc

    @property
    def grid_out(self) -> RegularGridS2:
        """Spatial domain of the output field."""
        return self.grid

    def extra_repr(self):
        return f"grid={self.grid!r},\nlmax={self.lmax}, mmax={self.mmax}, lmmax={self.lmmax}, csphase={self.csphase}"

    def forward(self, x: torch.Tensor):
        """
        Compute the inverse (real) spherical harmonic transform.

        Parameters
        ----------
        x : torch.Tensor
            Complex spherical harmonic coefficients of shape ``(..., lmax, mmax)``.

        Returns
        -------
        torch.Tensor
            Real-valued signal on the sphere of shape ``(..., nlat, nlon)``.
        """

        check(x.dim() >= 2, lambda: f"Expected tensor with at least 2 dimensions but got {x.dim()} instead")
        check(x.shape[-2] == self.lmax, lambda: f"Expected spherical harmonic degrees (lmax) shape[-2]=={self.lmax}, got {x.shape[-2]}")
        check(x.shape[-1] == self.mmax, lambda: f"Expected spherical harmonic orders (mmax) shape[-1]=={self.mmax}, got {x.shape[-1]}")

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
    Precomputes the associated Legendre polynomials and quadrature weights of the given grid.
    The SHT is applied to the last three dimensions of the input.

    Decomposes a tangential vector field
    :math:`\mathbf{v} = v_\theta\,\hat{e}_\theta + v_\lambda\,\hat{e}_\lambda`
    into **spheroidal** and **toroidal** spectral coefficients
    :math:`\hat{s}_l^m` and :math:`\hat{t}_l^m` using the derivatives of the
    associated Legendre polynomials.

    On equiangular grids, ``lmax > grid.max_exact_degree`` selects precomputed
    folded Clenshaw--Curtis analysis following :cite:`Reinecke2023`, Appendix A,
    when the resolved exclusive bounds satisfy ``lmax <= nlat - 1`` and
    ``mmax <= (nlon + 1) // 2``. These are the accurate band-limited recovery
    limits. Oversized dimensions retain legacy direct quadrature without
    implying accurate recovery. The default truncation is unchanged. Extended analysis
    is intended for float32 and float64; its precomputation latitude FFT has length
    ``2 * (nlat - 1)``.

    .. seealso::
        :doc:`/guide/spherical_harmonic_transforms`
            User guide with the full mathematical derivation of the vector SHT
            formulas, normalization conventions, and worked examples.

    Parameters
    ----------
    grid : RegularGridS2
        Descriptor of the spatial grid the transform operates on. It carries the
        resolution as well as the quadrature rule, so no separate ``nlat``/``nlon``
        is needed. Build one with :func:`torch_harmonics.grid.as_grid`.
    lmax : int, optional
        Non-inclusive maximum spherical harmonic degree.
    mmax : int, optional
        Non-inclusive maximum spherical harmonic order.
    norm : str
        Normalization convention (``"ortho"``, ``"schmidt"``, ``"unnorm"``),
        by default ``"ortho"``.
    csphase : bool
        Whether to include the Condon--Shortley phase factor :math:`(-1)^m`,
        by default ``True``.
    lmmax : int, optional
        Non-inclusive upper bound on degree minus order: retain only ``l - m < lmmax``.
        ``None`` leaves this bandwidth unrestricted.

    Examples
    --------
    >>> import torch
    >>> import torch_harmonics as th
    >>> grid = th.as_grid("equiangular", nlat=128, nlon=256)
    >>> vsht = th.RealVectorSHT(grid)
    >>> vector_field = torch.randn(1, 2, grid.nlat, grid.nlon)
    >>> coeffs = vsht(vector_field)   # shape (1, 2, lmax, mmax), complex
    >>> coeffs.shape
    torch.Size([1, 2, 64, 64])

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
    :cite:`Schaeffer2013`, :cite:`Wang2018`, :cite:`Reinecke2023`
    """

    @_rejects_legacy_signature(
        'nlat, nlon, lmax=None, mmax=None, grid="equiangular", norm="ortho", csphase=True',
        grid=("nlat", "nlon"),
    )
    def __init__(
        self,
        grid: RegularGridS2,
        lmax: Optional[int] = None,
        mmax: Optional[int] = None,
        norm: Optional[str] = "ortho",
        csphase: Optional[bool] = True,
        lmmax: Optional[int] = None,
    ):

        super().__init__()

        self.grid = require_regular_grid(grid)
        self.nlat, self.nlon = self.grid.shape
        _warn_if_not_spectrally_accurate(self.grid)
        self.norm = norm
        self.csphase = csphase

        # quadrature weights come from the grid descriptor; see the note in RealSHT
        weights = self.grid.colat_weights

        self._trunc = truncate_sht(self.grid, lmax, mmax, lmmax)
        self._extended_cc = _extended_cc_analysis(self.grid, self._trunc)
        if self._extended_cc:
            weights = _precompute_cc_projection(self.grid, self._trunc, self.norm, self.csphase, vector=True)
        else:
            dpct = _precompute_dlegpoly(self.mmax, self.lmax, self.grid, norm=self.norm, csphase=self.csphase, truncation=self._trunc)
            l = torch.arange(0, self.lmax)
            norm_factor = 1.0 / l / (l + 1)
            if self.lmax:
                norm_factor[0] = 1.0
            weights = torch.einsum("dmlk,k,l->dmlk", dpct, 2.0 * torch.pi * weights, norm_factor).contiguous()
            # Conjugate the imaginary derivative component.
            weights[1] = -1 * weights[1]

        # remember quadrature weights
        self.register_buffer("weights", weights, persistent=False)

    @property
    def lmax(self) -> int:
        return self._trunc.lmax

    @property
    def mmax(self) -> int:
        return self._trunc.mmax

    @property
    def lmmax(self) -> Optional[int]:
        return self._trunc.lmmax

    @property
    def grid_in(self) -> RegularGridS2:
        """Spatial domain of the input field."""
        return self.grid

    @property
    def grid_out(self) -> SpectralGrid:
        """Spectral coefficient support of the output."""
        return self._trunc

    def extra_repr(self):
        return f"grid={self.grid!r},\nlmax={self.lmax}, mmax={self.mmax}, lmmax={self.lmmax}, csphase={self.csphase}"

    def forward(self, x: torch.Tensor):
        """
        Compute the forward (real) vector spherical harmonic transform.

        Parameters
        ----------
        x : torch.Tensor
            Real-valued tangential vector field of shape ``(..., 2, nlat, nlon)``, where the
            size-2 dimension holds the two tangential (colatitude, longitude) components.

        Returns
        -------
        torch.Tensor
            Complex vector harmonic coefficients of shape ``(..., 2, lmax, mmax)``, where the
            size-2 dimension holds the spheroidal and toroidal components.
        """

        check(x.dim() >= 3, lambda: f"Expected tensor with at least 3 dimensions but got {x.dim()} instead")
        check(x.shape[-3] == 2, lambda: f"Expected vector field shape[-3]==2, got {x.shape[-3]}")
        check(x.shape[-2] == self.nlat, lambda: f"Expected latitudes shape[-2]=={self.nlat}, got {x.shape[-2]}")
        check(x.shape[-1] == self.nlon, lambda: f"Expected longitudes shape[-1]=={self.nlon}, got {x.shape[-1]}")

        # apply real fft in the longitudinal direction. The 2*pi scale factor is folded into
        # the quadrature weights, so no scaling of the complex output is needed here.
        x = rfft(x, nmodes=self.mmax, dim=-1, norm="forward")

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

        # stack the spheroidal and toroidal components in real space, so the only complex-typed
        # op is a single aten.complex over contiguous operands. Stacking complex tensors instead
        # would leave a complex cat, which inductor cannot codegen (triton has no complex type),
        # and feeding aten.complex the non-contiguous ...lm einsum outputs directly trips
        # assert_size_stride, as its meta predicts a contiguous layout. torch.stack allocates a
        # fresh contiguous buffer, which is exactly what that meta expects.
        out_re = torch.stack((s_re, t_re), dim=-3)
        out_im = torch.stack((s_im, t_im), dim=-3)

        return torch.complex(out_re, out_im)


class InverseRealVectorSHT(nn.Module):
    r"""
    Defines a module for computing the inverse (real-valued) vector SHT.
    Precomputes the associated Legendre polynomials on the nodes of the given grid.

    Given spheroidal and toroidal spectral coefficients :math:`\hat{s}_l^m` and
    :math:`\hat{t}_l^m`, reconstructs the tangential vector field on the sphere
    via Legendre synthesis with the derivatives of the associated Legendre
    polynomials, followed by an inverse real FFT.

    .. seealso::
        :doc:`/guide/spherical_harmonic_transforms`
            User guide with the full mathematical derivation of the inverse
            vector SHT formulas, normalization conventions, and worked examples.

    Parameters
    ----------
    grid : RegularGridS2
        Descriptor of the spatial grid the transform operates on. It carries the
        resolution as well as the quadrature rule, so no separate ``nlat``/``nlon``
        is needed. Build one with :func:`torch_harmonics.grid.as_grid`.
    lmax : int, optional
        Non-inclusive maximum spherical harmonic degree.
    mmax : int, optional
        Non-inclusive maximum spherical harmonic order.
    norm : str
        Normalization convention (``"ortho"``, ``"schmidt"``, ``"unnorm"``),
        by default ``"ortho"``.
    csphase : bool
        Whether to include the Condon--Shortley phase factor :math:`(-1)^m`,
        by default ``True``.
    lmmax : int, optional
        Non-inclusive upper bound on degree minus order: retain only ``l - m < lmmax``.
        ``None`` leaves this bandwidth unrestricted.

    Examples
    --------
    >>> import torch
    >>> import torch_harmonics as th
    >>> grid = th.as_grid("equiangular", nlat=128, nlon=256)
    >>> ivsht = th.InverseRealVectorSHT(grid)
    >>> coeffs = torch.randn(1, 2, ivsht.lmax, ivsht.mmax, dtype=torch.cfloat)
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

    @_rejects_legacy_signature(
        'nlat, nlon, lmax=None, mmax=None, grid="equiangular", norm="ortho", csphase=True',
        grid=("nlat", "nlon"),
    )
    def __init__(
        self,
        grid: RegularGridS2,
        lmax: Optional[int] = None,
        mmax: Optional[int] = None,
        norm: Optional[str] = "ortho",
        csphase: Optional[bool] = True,
        lmmax: Optional[int] = None,
    ):

        super().__init__()

        self.grid = require_regular_grid(grid)
        self.nlat, self.nlon = self.grid.shape
        _warn_if_not_spectrally_accurate(self.grid)
        self.norm = norm
        self.csphase = csphase

        self._trunc = truncate_sht(self.grid, lmax, mmax, lmmax)

        # precompute associated Legendre polynomials
        # store as (2, mmax, nlat, lmax) so the contraction dim l is stride-1
        dpct = _precompute_dlegpoly(self.mmax, self.lmax, self.grid, norm=self.norm, inverse=True, csphase=self.csphase, truncation=self._trunc)
        dpct = dpct.permute(0, 1, 3, 2).contiguous()

        # register weights
        self.register_buffer("dpct", dpct, persistent=False)

    @property
    def lmax(self) -> int:
        return self._trunc.lmax

    @property
    def mmax(self) -> int:
        return self._trunc.mmax

    @property
    def lmmax(self) -> Optional[int]:
        return self._trunc.lmmax

    @property
    def grid_in(self) -> SpectralGrid:
        """Spectral coefficient support of the input field."""
        return self._trunc

    @property
    def grid_out(self) -> RegularGridS2:
        """Spatial domain of the output field."""
        return self.grid

    def extra_repr(self):
        return f"grid={self.grid!r},\nlmax={self.lmax}, mmax={self.mmax}, lmmax={self.lmmax}, csphase={self.csphase}"

    def forward(self, x: torch.Tensor):
        """
        Compute the inverse (real) vector spherical harmonic transform.

        Parameters
        ----------
        x : torch.Tensor
            Complex vector harmonic coefficients of shape ``(..., 2, lmax, mmax)``, where the
            size-2 dimension holds the spheroidal and toroidal components.

        Returns
        -------
        torch.Tensor
            Real-valued tangential vector field of shape ``(..., 2, nlat, nlon)``, where the
            size-2 dimension holds the two tangential (colatitude, longitude) components.
        """

        check(x.dim() >= 3, lambda: f"Expected tensor with at least 3 dimensions but got {x.dim()} instead")
        check(x.shape[-3] == 2, lambda: f"Expected vector field shape[-3]==2, got {x.shape[-3]}")
        check(x.shape[-2] == self.lmax, lambda: f"Expected spherical harmonic degrees (lmax) shape[-2]=={self.lmax}, got {x.shape[-2]}")
        check(x.shape[-1] == self.mmax, lambda: f"Expected spherical harmonic orders (mmax) shape[-1]=={self.mmax}, got {x.shape[-1]}")

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

        # reassemble in real space and apply inverse FFT, see RealVectorSHT.forward
        out_re = torch.stack((srl, trl), dim=-3)
        out_im = torch.stack((sim, tim), dim=-3)
        xs = torch.complex(out_re, out_im)
        x = irfft(xs, n=self.nlon, dim=-1, norm="forward")

        return x
