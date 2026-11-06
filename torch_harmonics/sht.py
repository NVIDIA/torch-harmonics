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

import numbers
import os
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn

from torch_harmonics._field_contract import check_coeffs, check_field
from torch_harmonics.cache import SharedBufferMixin, shared_table
from torch_harmonics.fft import RingFFT, irfft, rfft
from torch_harmonics.grid import EquiangularGrid, Fejer1Grid, GridS2, RegularGridS2, SpectralGrid, _rejects_legacy_signature, require_grid
from torch_harmonics.healpix import HealpixGrid, healpix_ring_weights
from torch_harmonics.legendre import _mask_spectral_block, _precompute_dlegpoly, _precompute_legpoly, dlegpoly, legpoly
from torch_harmonics.radial import split_radial
from torch_harmonics.truncation import _validate_forward_sht_degree, _warn_if_not_spectrally_accurate, truncate_sht


def _require_sht_grid(grid) -> GridS2:
    """Validate the grid of an SHT: a :class:`RegularGridS2` or a :class:`HealpixGrid`."""
    grid = require_grid(grid)
    if not isinstance(grid, (RegularGridS2, HealpixGrid)):
        raise TypeError(f"grid must be a RegularGridS2 or a HealpixGrid; the SHT is not implemented for {type(grid).__name__}. Got {grid!r}.")
    return grid


def _extended_folded_analysis(grid: RegularGridS2, spectral_grid: SpectralGrid) -> bool:
    """Select extended analysis after the usual spectral support resolution."""
    # Oversized dimensions retain legacy direct quadrature for spectral upsampling.
    # Accurate folded recovery requires both latitude and longitude bandwidth limits.
    # Folding exists for regular grids only; HEALPix has no exact degree to extend.
    if not isinstance(grid, RegularGridS2):
        return False
    return grid.max_exact_degree < spectral_grid.lmax <= grid.max_recoverable_degree and spectral_grid.mmax <= grid.max_azimuthal_order


def _latitude_folding(grid: RegularGridS2, mmax: int, vector: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """The folding buffers of a grid as ``(parity, native_weights, complementary_weights, phase, include_poles)``.

    Both rules interleave the native rings with complementary ones on a fine
    Clenshaw--Curtis grid. Equiangular grids fold onto the midpoints between their rings
    and sample the poles; Fejer-1 grids fold onto the Clenshaw--Curtis nodes around their
    rings and do not. The weights include the longitude scale ``2 pi``, and the phase of
    the half-step shift is complex.
    """
    if isinstance(grid, EquiangularGrid):
        # native rings are the even fine rings, the shift runs forward to the midpoints
        dense_weights = 2.0 * torch.pi * EquiangularGrid(nlat=2 * grid.nlat - 1, nlon=1).colat_weights
        native_weights, complement, include_poles, direction = dense_weights[::2], dense_weights[1::2], True, 1.0
    elif isinstance(grid, Fejer1Grid):
        # native rings are the odd fine rings, the shift runs backward to the even ones
        dense_weights = 2.0 * torch.pi * EquiangularGrid(nlat=2 * grid.nlat + 1, nlon=1).colat_weights
        native_weights, complement, include_poles, direction = dense_weights[1::2], dense_weights[::2], False, -1.0
    else:
        raise ValueError(f"folded projection is not implemented for {type(grid).__name__}")

    # period of the meridian extension; the zero padding selects the complementary rings
    period = 2 * (grid.nlat - 1) if include_poles else 2 * grid.nlat
    complementary_weights = torch.zeros(period, dtype=torch.float64)
    complementary_weights[: complement.numel()] = complement

    angle = direction * torch.pi * torch.fft.fftfreq(period, dtype=torch.float64)
    phase = torch.complex(angle.cos(), angle.sin())
    # The unpaired meridional Nyquist cosine vanishes at the half-step targets.
    # Supported harmonics do not contain this frequency; suppressing it retains
    # conjugate symmetry for arbitrary complex longitude modes.
    phase[period // 2] = 0

    # scalar parity (-1)^m, tangential vector parity (-1)^(m+1)
    parity = torch.ones(mmax, 1, dtype=torch.int8)
    parity[1::2] = -1
    if vector:
        parity = -parity

    return parity, native_weights.contiguous(), complementary_weights, phase, include_poles


def _periodic_latitude_extension(x: torch.Tensor, signs: torch.Tensor, include_poles: bool = True) -> torch.Tensor:
    r"""Extend ``(..., m, nlat)`` modes to a periodic meridian.

    ``signs`` has shape ``(m, 1)``: scalar parity is :math:`(-1)^m`,
    tangential vector parity is :math:`(-1)^{m+1}`. A pole-inclusive grid
    reflects only interior rings so each pole occurs once; a pole-excluding grid
    reflects every sampled ring. The caller supplies signs for the actual orders.
    """
    mirrored = x[..., 1:-1] if include_poles else x
    return torch.cat((x, signs * mirrored.flip(-1)), dim=-1)


def _periodic_latitude_extension_adjoint(x: torch.Tensor, signs: torch.Tensor, include_poles: bool = True) -> torch.Tensor:
    """Adjoint of the periodic meridian extension for both endpoint conventions."""
    if include_poles:
        nlat = x.shape[-1] // 2 + 1
        interior = x[..., 1 : nlat - 1] + signs * x[..., nlat:].flip(-1)
        return torch.cat((x[..., :1], interior, x[..., nlat - 1 : nlat]), dim=-1)
    nlat = x.shape[-1] // 2
    return x[..., :nlat] + signs * x[..., nlat:].flip(-1)


def _fourier_shift_latitude(x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Half-grid interpolation; its Hilbert adjoint uses the conjugate phase."""
    return torch.fft.ifft(torch.fft.fft(x, dim=-1) * phase, dim=-1)


def _fold_resampled_latitude(
    x: torch.Tensor,
    signs: torch.Tensor,
    native_weights: torch.Tensor,
    complementary_weights: torch.Tensor,
    phase: torch.Tensor,
    include_poles: bool = True,
) -> torch.Tensor:
    r"""Fold native and complementary quadrature onto the sampled meridian.

    ``A`` periodically extends the meridian, shifts it by half a sample spacing,
    and selects the complementary rings on the northern-to-southern half. The
    zero-padded complementary weights encode both that selection and its
    adjoint. For CC the original nodes are the even fine-grid rings; for F1 they
    are the odd ones.
    """
    # FFT backends reject empty batches; an empty order axis has no complement term.
    if x.shape[-2] == 0:
        return native_weights * x
    shifted = _fourier_shift_latitude(_periodic_latitude_extension(x, signs, include_poles), phase)
    folded = _fourier_shift_latitude(shifted * complementary_weights, phase.conj())
    return native_weights * x + _periodic_latitude_extension_adjoint(folded, signs, include_poles)


@shared_table
@torch.no_grad()
def _precompute_folded_projection(
    grid: RegularGridS2,
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
    r"""Fold a fine Clenshaw--Curtis rule onto real projection rows in bounded blocks.

    CC and F1 interleave their native rings with complementary rings on fine
    Clenshaw--Curtis nodes. The half-grid shift suppresses the unpaired Nyquist
    bin, making ``U = Q_native + A* Q_complement A`` real and symmetric. The
    bilinear coefficient contraction ``P.T @ U @ x`` therefore uses
    ``W = U.T @ P = U @ P`` without conjugating complex longitude modes.
    Both vector derivative rows have parity ``(-1)**(m+1)``.

    This is the latitude resampling of DUCC's ``analysis_2d`` for its CC and
    F1 geometries (``resample_to_prepared_CC``, after Reinecke, Belkner and
    Carron 2023, Appendix A), applied to the Legendre basis once rather than
    to the data in every call.

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

    parity, native_weights, complementary_weights, phase, include_poles = _latitude_folding(grid, mmax, vector)
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
                parity[ms:me],
                native_weights,
                complementary_weights,
                phase,
                include_poles,
            )
            weights[..., ms - mmin : me - mmin, ls:le, :] = folded.real.transpose(-3, -2)[..., kmin:kmax]
    return weights


def _quadrature_weights(grid: GridS2) -> torch.Tensor:
    r"""
    The per-ring weights of the analysis, in the :math:`\cos\theta` domain.

    A regular grid's quadrature weights come from its descriptor, which supports every
    grid precompute_latitudes does. HEALPix has no exact quadrature; its analysis uses the
    ring weights of :func:`~torch_harmonics.healpix.healpix_ring_weights`.
    """
    if isinstance(grid, HealpixGrid):
        return healpix_ring_weights(grid.nside)
    return grid.colat_weights


SHT_BLOCK_SIZE_ENV = "TORCH_HARMONICS_SHT_BLOCK_SIZE"
"""Environment variable that sets the default ``block_size`` of the SHTs: an integer, or ``none`` for dense tables."""

DEFAULT_SHT_BLOCK_SIZE = 32
"""The default ``block_size`` of the SHTs when :data:`SHT_BLOCK_SIZE_ENV` is not set."""


class _BlockSizeFromEnvironment:
    """The default of every SHT's ``block_size``: read :data:`SHT_BLOCK_SIZE_ENV` at construction."""

    def __repr__(self) -> str:
        return f"${SHT_BLOCK_SIZE_ENV} or {DEFAULT_SHT_BLOCK_SIZE}"


BLOCK_SIZE_FROM_ENVIRONMENT = _BlockSizeFromEnvironment()


def _resolve_block_size(block_size, mmax: int) -> Optional[int]:
    """
    The order block size a transform stores its tables in, ``None`` for a single dense block.

    ``block_size`` is a positive integer, ``None`` for dense tables, or
    :data:`BLOCK_SIZE_FROM_ENVIRONMENT`, which reads :data:`SHT_BLOCK_SIZE_ENV` and falls back to
    :data:`DEFAULT_SHT_BLOCK_SIZE`. A block size of at least ``mmax`` is a single block, the
    dense table, and resolves to ``None``, so those transforms share their tables.
    """
    if block_size is BLOCK_SIZE_FROM_ENVIRONMENT:
        value = os.environ.get(SHT_BLOCK_SIZE_ENV, "").strip()
        if not value:
            block_size = DEFAULT_SHT_BLOCK_SIZE
        elif value.lower() in ("none", "dense"):
            block_size = None
        else:
            try:
                block_size = int(value)
            except ValueError:
                raise ValueError(f"{SHT_BLOCK_SIZE_ENV}={value!r} is neither a positive integer nor 'none'") from None
    if block_size is not None and (isinstance(block_size, bool) or not isinstance(block_size, numbers.Integral) or block_size < 1):
        raise ValueError(f"block_size must be a positive integer or None, got {block_size!r}")
    return None if block_size is None or block_size >= mmax else int(block_size)


def _sht_blocks(truncation: SpectralGrid, block_size: Optional[int], mmin: int, mmax: int, lmin: int, lmax: int) -> Tuple[Tuple[int, int, int, int], ...]:
    r"""
    The order blocks of a table holding orders ``[mmin, mmax)`` and degrees ``[lmin, lmax)``.

    Each block is ``(ms, me, ls, le)`` in global indices: orders ``[ms, me)`` and the degrees
    ``[ls, le)`` any of them has, :math:`m \le l < m + lm_{\max}`, clipped to ``[lmin, lmax)``.
    The orders are cut at the global multiples of ``block_size``, so the blocks of a distributed
    rank are a subset of the serial ones; ``None`` keeps them in one block.
    """
    if mmax <= mmin:
        # one empty block, which still contracts: the output keeps its degrees and its graph
        return ((mmin, mmin, lmin, lmax),)
    if block_size is None:
        cuts = [mmin, mmax]
    else:
        cuts = [mmin, *range((mmin // block_size + 1) * block_size, mmax, block_size), mmax]
    blocks = []
    for ms, me in zip(cuts[:-1], cuts[1:]):
        top = lmax if truncation.lmmax is None else min(lmax, me - 1 + truncation.lmmax)
        ls = min(max(ms, lmin), lmax)
        blocks.append((ms, me, ls, max(top, ls)))
    return tuple(blocks)


def _analysis_block(grid: GridS2, truncation: SpectralGrid, norm: str, csphase: bool, vector: bool, ms: int, me: int, ls: int, le: int, kmin: int, kmax: int) -> torch.Tensor:
    """The analysis table of orders ``[ms, me)`` and degrees ``[ls, le)`` on rings ``[kmin, kmax)``, see :func:`_analysis_weights`."""
    # Only this block's rings are needed: the contraction over rings is completed by a
    # reduce-scatter in a distributed forward, and rings are independent of one another, so
    # restricting them costs nothing
    quad_weights = (2.0 * torch.pi * _quadrature_weights(grid))[kmin:kmax]

    if _extended_folded_analysis(grid, truncation):
        weights = _precompute_folded_projection(grid, truncation, norm, csphase, vector=vector, mmin=ms, mmax=me, kmin=kmin, kmax=kmax)
        # always copy: a single-order slice counts as contiguous, so .contiguous() would return
        # the view and keep the projection of every degree alive
        weights = weights[..., ls:le, :].clone(memory_format=torch.contiguous_format)
    elif not vector:
        pct = _precompute_legpoly(me, le, grid, norm=norm, csphase=csphase, truncation=truncation, mmin=ms, lmin=ls, kmin=kmin, kmax=kmax)
        weights = torch.einsum("mlk,k->mlk", pct, quad_weights).contiguous()
    else:
        dpct = _precompute_dlegpoly(me, le, grid, norm=norm, csphase=csphase, truncation=truncation, mmin=ms, lmin=ls, kmin=kmin, kmax=kmax)
        # combine integration weights, normalization factor in to one. The degrees are
        # float64: an integer arange divided by a float gives the default dtype, float32,
        # and that rounding of 1/(l(l+1)) would survive a later .double()
        l = torch.arange(ls, le, dtype=torch.float64)
        norm_factor = torch.where(l > 0, 1.0 / l / (l + 1), 1.0)
        weights = torch.einsum("dmlk,k,l->dmlk", dpct, quad_weights, norm_factor).contiguous()
        # conjugate the imaginary derivative component; weights is fresh, so writing to it is safe
        weights[1] = -1 * weights[1]

    return weights


@shared_table
@torch.no_grad()
def _analysis_weights(
    grid: GridS2,
    truncation: SpectralGrid,
    norm: str,
    csphase: bool,
    vector: bool,
    mmin: int,
    mmax: int,
    kmin: int,
    kmax: int,
    block_size: Optional[int],
) -> Tuple[torch.Tensor, ...]:
    r"""
    The analysis buffers of a forward SHT: quadrature-weighted projection rows, in order blocks.

    One table per block of :func:`_sht_blocks` over orders ``[mmin, mmax)`` and every degree,
    shaped ``(me - ms, le - ls, kmax - kmin)``, or ``(2, ...)`` for a vector transform: the
    block's orders, the degrees they have, and rings ``[kmin, kmax)``. A serial transform asks
    for every order and ring, a distributed one for its block. The degrees below an order
    vanish, so a block stores only the lower triangle of the dense ``(mmax, lmax)`` table it
    covers and its diagonal block; ``block_size=None`` is one block, the dense table.

    On equiangular and Fejer-1 grids beyond the native bound these are the folded projection
    rows. Otherwise the Legendre table times the quadrature weights (the ring weights on
    HEALPix), which include the longitude scale :math:`2\pi` of the forward-normalized FFT, and for a vector transform
    the degree factor :math:`1 / (l (l + 1))` and the conjugation of the imaginary derivative
    component.

    Shared between all transforms with the same arguments, the block size included, see
    :func:`~torch_harmonics.cache.shared_table`.
    """
    blocks = _sht_blocks(truncation, block_size, mmin, mmax, 0, truncation.lmax)
    return tuple(_analysis_block(grid, truncation, norm, csphase, vector, ms, me, ls, le, kmin, kmax) for ms, me, ls, le in blocks)


@shared_table
@torch.no_grad()
def _synthesis_table(
    grid: GridS2,
    truncation: SpectralGrid,
    norm: str,
    csphase: bool,
    vector: bool,
    mmin: int,
    mmax: int,
    lmin: int,
    lmax: int,
    block_size: Optional[int],
) -> Tuple[torch.Tensor, ...]:
    r"""
    The synthesis buffers of an inverse SHT: the Legendre table on the grid's rings, in order blocks.

    One table per block of :func:`_sht_blocks` over orders ``[mmin, mmax)`` and degrees
    ``[lmin, lmax)``, shaped ``(me - ms, nlat, le - ls)``, or ``(2, ...)`` with the derivative
    tables for a vector transform, stored with the contracted degree axis stride-1. A serial
    transform asks for every order and degree, a distributed one for its block; both ranges
    are walked from the start, since the sectoral seed couples orders and the three-term
    recurrence couples degrees. ``block_size=None`` is one block, the dense table.

    Shared between all transforms with the same arguments, the block size included, see
    :func:`~torch_harmonics.cache.shared_table`.
    """
    tables = []
    for ms, me, ls, le in _sht_blocks(truncation, block_size, mmin, mmax, lmin, lmax):
        if not vector:
            pct = _precompute_legpoly(me, le, grid, norm=norm, inverse=True, csphase=csphase, truncation=truncation, mmin=ms, lmin=ls)
            tables.append(pct.permute(0, 2, 1).contiguous())
        else:
            dpct = _precompute_dlegpoly(me, le, grid, norm=norm, inverse=True, csphase=csphase, truncation=truncation, mmin=ms, lmin=ls)
            tables.append(dpct.permute(0, 1, 3, 2).contiguous())
    return tuple(tables)


class _BlockLayout:
    """
    Where the order blocks of a table sit in a transform's local spectral block.

    ``sizes`` are the blocks' order counts, to split the order axis by; ``degrees`` their
    ``(offset, count)`` within the local degree range ``[lmin, ...)``. Plain ints, so
    ``torch.compile`` specializes on them.
    """

    __slots__ = ("sizes", "degrees")

    def __init__(self, blocks: Tuple[Tuple[int, int, int, int], ...], lmin: int):
        self.sizes = [me - ms for ms, me, _, _ in blocks]
        self.degrees = [(ls - lmin, le - ls) for _, _, ls, le in blocks]


def _register_blocks(module: nn.Module, name: str, tables: Tuple[torch.Tensor, ...], blocks, lmin: int) -> None:
    """Register the block tables of a transform as the shared buffers ``{name}_{i}``."""
    module._layout = _BlockLayout(blocks, lmin)
    module._block_names = tuple(f"{name}_{i}" for i in range(len(tables)))
    for buffer, table in zip(module._block_names, tables):
        module.register_shared_buffer(buffer, table)


def _blocks(module: nn.Module) -> list:
    """The block tables of a transform, as registered by :func:`_register_blocks`."""
    return [getattr(module, name) for name in module._block_names]


def _dense_table(module: nn.Module, degree_dim: int, ndegrees: int) -> torch.Tensor:
    """The dense table of a transform, assembled from its blocks: a fresh tensor, for inspection."""
    tables = _blocks(module)
    padded = []
    for table, (offset, count) in zip(tables, module._layout.degrees):
        pad = [0, 0] * (table.dim() - 1 - (degree_dim % table.dim())) + [offset, ndegrees - offset - count]
        padded.append(nn.functional.pad(table, pad))
    return torch.cat(padded, dim=-3)


def _analysis_contraction(x: torch.Tensor, tables: list, layout: _BlockLayout, lmax: int) -> torch.Tensor:
    """
    Contract ``(..., m, k)`` with the analysis blocks into ``(..., lmax, m)``.

    One einsum per block over its degrees, each padded to every degree. The input is cut with
    :func:`torch.split`, whose backward is a single concatenation; slicing it per block instead
    would allocate a zero gradient of the full input for every block.
    """
    outs = []
    for xb, w, (offset, count) in zip(torch.split(x, layout.sizes, dim=-2), tables, layout.degrees):
        y = torch.einsum("...mk,mlk->...lm", xb, w)
        if offset or offset + count != lmax:
            y = nn.functional.pad(y, (0, 0, offset, lmax - offset - count))
        outs.append(y)
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)


def _synthesis_contraction(x: torch.Tensor, tables: list, layout: _BlockLayout) -> torch.Tensor:
    """Contract ``(..., m, l)`` with the synthesis blocks into ``(..., nlat, m)``: one einsum per block over its degrees."""
    outs = []
    for xb, p, (offset, count) in zip(torch.split(x, layout.sizes, dim=-2), tables, layout.degrees):
        outs.append(torch.einsum("...ml,mkl->...km", xb.narrow(-1, offset, count), p))
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)


def _components(y: torch.Tensor, cdim: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """The real and imaginary parts of both vector components of a contraction of stacked parts, ``((re0, im0), (re1, im1))``."""
    return tuple((y[0].select(cdim, c), y[1].select(cdim, c)) for c in (0, 1))


def _real_and_imaginary(x: torch.Tensor) -> torch.Tensor:
    """A complex tensor as a real one, with its real and imaginary parts stacked on a new leading axis."""
    return torch.stack((x.real, x.imag))


class RealSHT(SharedBufferMixin, nn.Module):
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

    On equiangular and Fejér-1 grids, explicit ``lmax > grid.max_exact_degree``
    selects precomputed folded Clenshaw--Curtis analysis following
    :cite:`Reinecke2023`, Appendix A, and the latitude resampling of DUCC
    :cite:`DUCC`, folded into the weights at construction instead of applied
    in every call, when the resolved bounds fit the
    geometry-specific latitude limits (``lmax <= nlat - 1`` for equiangular,
    ``lmax <= nlat`` for Fejér-1) and ``mmax <= (nlon + 1) // 2``. These are
    the accurate band-limited recovery limits; the latter excludes the complex
    longitude Nyquist order when ``nlon`` is even. Oversized dimensions retain
    legacy direct quadrature without implying accurate recovery. The default
    truncation is unchanged. Extended analysis is intended for float32 and
    float64; its latitude FFT has length ``2 * (nlat - 1)`` for equiangular
    and ``2 * nlat`` for Fejér-1 grids.

    On a :class:`~torch_harmonics.healpix.HealpixGrid` the field is flat, ``(...,
    npix)`` in RING order, and the coefficients have the same ``(..., lmax, mmax)``
    layout as on every other grid. Each ring is Fourier transformed at its own length
    and phase, and orders beyond a ring's Nyquist limit are aliased, not dropped, as
    the short polar rings require. No quadrature on HEALPix is exact, so the analysis
    is approximate: it uses the ring weights of
    :func:`~torch_harmonics.healpix.healpix_ring_weights`, which recover a field
    band-limited to the default ``lmax = 2 * nside`` to a relative error of about
    :math:`10^{-2}`. Pass ``iter=3`` for about :math:`10^{-6}`, as with healpy's
    ``map2alm(..., iter=3)``.

    .. seealso::
        :doc:`/guide/spherical_harmonic_transforms`
            User guide with the full mathematical derivation, normalization
            conventions, grid types, and worked examples.

    Parameters
    ----------
    grid : RegularGridS2 or HealpixGrid
        Descriptor of the spatial grid the transform operates on. It carries the
        resolution as well as the quadrature rule, so no separate ``nlat``/``nlon``
        is needed. Build one with :func:`torch_harmonics.grid.as_grid`. A
        :class:`~torch_harmonics.radial.SphericalProductGrid` over one transforms every shell
        alike, with the radial axis in front of the spatial (or spectral) ones.
    lmax : int, optional
        Non-inclusive maximum spherical harmonic degree.
    mmax : int, optional
        Non-inclusive maximum spherical harmonic order.
    lmmax : int, optional
        Non-inclusive upper bound on degree minus order: retain only ``l - m < lmmax``.
        ``None`` leaves this bandwidth unrestricted.
    norm : str
        Normalization convention (``"ortho"``, ``"schmidt"``, ``"unnorm"``),
        by default ``"ortho"``.
    csphase : bool
        Whether to include the Condon--Shortley phase factor :math:`(-1)^m`,
        by default ``True``.
    block_size : int or None
        Number of orders per block of the stored Legendre tables, or ``None`` for one dense
        table. A block keeps only the degrees its orders have, :math:`l \ge m`, which roughly
        halves the tables and the contraction of a triangular truncation. By default
        ``$TORCH_HARMONICS_SHT_BLOCK_SIZE`` (an integer, or ``none``) if set, else 32. Transforms
        share their tables only with transforms of the same block size.
    iter : int
        Number of Jacobi iterations refining the analysis, by default 0. Each one
        computes :math:`a \leftarrow a + A(f - S a)` with the analysis :math:`A` and
        the synthesis :math:`S` on the same support, and costs one more synthesis and
        analysis. Meant for grids without an exact quadrature such as HEALPix, where
        ``iter=3`` is recommended when precision matters. On a grid with exact
        quadrature the corrections are round-off.

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

    On HEALPix, with three Jacobi iterations:

    >>> grid = th.HealpixGrid(nside=16)
    >>> sht = th.RealSHT(grid, iter=3)
    >>> sht(torch.randn(1, grid.npoints)).shape   # lmax = mmax = 2 * nside
    torch.Size([1, 32, 32])

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
    :cite:`Schaeffer2013`, :cite:`Wang2018`, :cite:`Reinecke2023`, :cite:`DUCC`
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
        lmmax: Optional[int] = None,
        *,
        norm: Optional[str] = "ortho",
        csphase: Optional[bool] = True,
        block_size: Optional[int] = BLOCK_SIZE_FROM_ENVIRONMENT,
        iter: int = 0,
    ):

        super().__init__()

        # on a SphericalProductGrid every shell is transformed alike; the radial grid only
        # adds its axis to the field shape
        grid, self.radial = split_radial(grid)
        self.grid = _require_sht_grid(grid)
        self.norm = norm
        self.csphase = csphase
        if isinstance(iter, bool) or not isinstance(iter, int) or iter < 0:
            raise ValueError(f"iter must be a non-negative integer, got {iter!r}")
        self.iter = iter

        if isinstance(self.grid, RegularGridS2):
            _warn_if_not_spectrally_accurate(self.grid)

        self._trunc = truncate_sht(self.grid, lmax, mmax, lmmax)
        _validate_forward_sht_degree(self.grid, self.lmax)
        self._extended_folded = _extended_folded_analysis(self.grid, self._trunc)

        if isinstance(self.grid, RegularGridS2):
            self.nlat, self.nlon = self.grid.shape
            self.ring_fft = None
        else:
            # HEALPix: ragged rings, an exact ring FFT with aliasing, and approximate
            # analysis with the ring weights (see _quadrature_weights)
            self.nlat, self.nlon = self.grid.nrings, None
            self.ring_fft = RingFFT(self.grid, self.mmax)

        # quadrature-weighted Legendre rows in order blocks, shared with every transform of the same kind
        self.block_size = _resolve_block_size(block_size, self.mmax)
        blocks = _sht_blocks(self._trunc, self.block_size, 0, self.mmax, 0, self.lmax)
        tables = _analysis_weights(self.grid, self._trunc, self.norm, self.csphase, False, 0, self.mmax, 0, self.nlat, self.block_size)
        _register_blocks(self, "weights", tables, blocks, 0)

        # Jacobi iteration a <- a + A (f - S a) needs the synthesis S on the same support.
        # Its bounds are already resolved, so the inverse cannot warn about a changed
        # default; a non-accurate grid has warned once above already.
        self._synthesis = None
        if self.iter > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self._synthesis = InverseRealSHT(self.grid, self.lmax, self.mmax, self.lmmax, norm=self.norm, csphase=self.csphase, block_size=self.block_size)

    @property
    def weights(self) -> torch.Tensor:
        """The dense analysis table ``(mmax, lmax, nlat)``, assembled from the blocks: a fresh tensor, for inspection."""
        return _dense_table(self, -2, self.lmax)

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
        radial = "" if self.radial is None else f",\nradial={self.radial!r}"
        return f"grid={self.grid!r}{radial},\nlmax={self.lmax}, mmax={self.mmax}, lmmax={self.lmmax}, csphase={self.csphase}" + (f", iter={self.iter}" if self.iter else "")

    def forward(self, x: torch.Tensor):
        """
        Compute the forward (real) spherical harmonic transform.

        Parameters
        ----------
        x : torch.Tensor
            Real-valued signal on the sphere of shape ``(..., *grid.shape)``: ``(..., nlat,
            nlon)`` on a regular grid and ``(..., npix)`` on HEALPix.

        Returns
        -------
        torch.Tensor
            Complex spherical harmonic coefficients of shape ``(..., lmax, mmax)``.
        """

        check_field(x, self.grid, radial=self.radial)

        # Jacobi iteration a <- a + A (f - S a). The coefficients are carried as their real
        # and imaginary parts, (..., lmax, mmax) each: they are used twice per step, and
        # autograd then sums two complex gradients, one of them transposed, which inductor
        # cannot lower (view.dtype needs a unit inner stride). Real gradients sum fine.
        re, im = self._analyze(x)
        for _ in range(self.iter):
            residual = x - self._synthesis._synthesize_parts(re.transpose(-1, -2).contiguous(), im.transpose(-1, -2).contiguous())
            dre, dim = self._analyze(residual)
            re, im = re + dre, im + dim

        # the ...lm einsum output is non-contiguous (l ends up stride-1, m slow); torch.complex
        # preserves those strides, but inductor's meta kernel for aten.complex predicts a contiguous
        # layout, tripping assert_size_stride under torch.compile(dynamic=False). Force contiguous.
        return torch.complex(re.contiguous(), im.contiguous())

    def _analyze(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Fourier transform in longitude. The 2*pi scale factor is folded into the
        # quadrature weights, so no scaling of the complex output is needed here.
        if self.ring_fft is None:
            x = rfft(x, nmodes=self.mmax, dim=-1, norm="forward")
        else:
            x = self.ring_fft(x)

        # transpose to put the contraction dim (nlat) on the fast axis, real and imaginary
        # parts stacked so that each block is one contraction
        x = _real_and_imaginary(x.transpose(-1, -2))

        # Legendre-Gauss quadrature: contract over k=nlat (stride-1 in both operands)
        out = _analysis_contraction(x, [w.to(x.dtype) for w in _blocks(self)], self._layout, self.lmax)
        return out[0], out[1]


class InverseRealSHT(SharedBufferMixin, nn.Module):
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
    grid : RegularGridS2 or HealpixGrid
        Descriptor of the spatial grid the transform operates on. It carries the
        resolution as well as the quadrature rule, so no separate ``nlat``/``nlon``
        is needed. Build one with :func:`torch_harmonics.grid.as_grid`. A
        :class:`~torch_harmonics.radial.SphericalProductGrid` over one transforms every shell
        alike, with the radial axis in front of the spatial (or spectral) ones.
    lmax : int, optional
        Non-inclusive maximum spherical harmonic degree.
    mmax : int, optional
        Non-inclusive maximum spherical harmonic order.
    lmmax : int, optional
        Non-inclusive upper bound on degree minus order: retain only ``l - m < lmmax``.
        ``None`` leaves this bandwidth unrestricted.
    norm : str
        Normalization convention (``"ortho"``, ``"schmidt"``, ``"unnorm"``),
        by default ``"ortho"``.
    csphase : bool
        Whether to include the Condon--Shortley phase factor :math:`(-1)^m`,
        by default ``True``.
    block_size : int or None
        Number of orders per block of the stored Legendre tables, or ``None`` for one dense
        table. A block keeps only the degrees its orders have, :math:`l \ge m`, which roughly
        halves the tables and the contraction of a triangular truncation. By default
        ``$TORCH_HARMONICS_SHT_BLOCK_SIZE`` (an integer, or ``none``) if set, else 32. Transforms
        share their tables only with transforms of the same block size.

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

    On a :class:`~torch_harmonics.healpix.HealpixGrid` the output is flat, ``(...,
    npix)`` in RING order, and the synthesis is exact: every order is folded into the
    Fourier bins it aliases to on each ring before that ring's inverse FFT.

    .. note::
        The inverse real FFT (C2R transform) expects the DC component (:math:`m = 0`)
        and, when ``nlon`` is even, the Nyquist component (:math:`m = N_\lambda / 2`)
        to be purely real.  This routine zeros out the imaginary parts of these
        components before calling the transform.

    Raises
    ------
    TypeError
        If ``grid`` is neither a :class:`~torch_harmonics.grid.RegularGridS2` nor a
        :class:`~torch_harmonics.healpix.HealpixGrid`.

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
        lmmax: Optional[int] = None,
        *,
        norm: Optional[str] = "ortho",
        csphase: Optional[bool] = True,
        block_size: Optional[int] = BLOCK_SIZE_FROM_ENVIRONMENT,
    ):

        super().__init__()

        # on a SphericalProductGrid every shell is transformed alike; the radial grid only
        # adds its axis to the field shape
        grid, self.radial = split_radial(grid)
        self.grid = _require_sht_grid(grid)
        self.norm = norm
        self.csphase = csphase

        if isinstance(self.grid, RegularGridS2):
            _warn_if_not_spectrally_accurate(self.grid)

        self._trunc = truncate_sht(self.grid, lmax, mmax, lmmax)

        if isinstance(self.grid, RegularGridS2):
            self.nlat, self.nlon = self.grid.shape
            self.ring_fft = None
        else:
            # synthesis on HEALPix is exact: it only evaluates the expansion at the pixels
            self.nlat, self.nlon = self.grid.nrings, None
            self.ring_fft = RingFFT(self.grid, self.mmax)

        # associated Legendre polynomials in order blocks of (orders, nlat, degrees), so the
        # contraction dim l is stride-1; shared with every transform of the same kind
        self.block_size = _resolve_block_size(block_size, self.mmax)
        blocks = _sht_blocks(self._trunc, self.block_size, 0, self.mmax, 0, self.lmax)
        tables = _synthesis_table(self.grid, self._trunc, self.norm, self.csphase, False, 0, self.mmax, 0, self.lmax, self.block_size)
        _register_blocks(self, "pct", tables, blocks, 0)

    @property
    def pct(self) -> torch.Tensor:
        """The dense synthesis table ``(mmax, nlat, lmax)``, assembled from the blocks: a fresh tensor, for inspection."""
        return _dense_table(self, -1, self.lmax)

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
        radial = "" if self.radial is None else f",\nradial={self.radial!r}"
        return f"grid={self.grid!r}{radial},\nlmax={self.lmax}, mmax={self.mmax}, lmmax={self.lmmax}, csphase={self.csphase}"

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
            Real-valued signal on the sphere of shape ``(..., *grid.shape)``: ``(..., nlat,
            nlon)`` on a regular grid and ``(..., npix)`` on HEALPix.
        """

        check_coeffs(x, self.lmax, self.mmax, radial=self.radial)
        return self._synthesize(x)

    def _synthesize(self, x: torch.Tensor) -> torch.Tensor:
        # transpose to put the contraction dim (lmax) on the fast axis
        x = x.transpose(-1, -2)
        return self._synthesize_parts(x.real, x.imag)

    def _synthesize_parts(self, x_re: torch.Tensor, x_im: torch.Tensor) -> torch.Tensor:
        # the real and imaginary parts of the coefficients, (..., mmax, lmax) each, stacked so that each
        # block is one contraction
        x = torch.stack((x_re, x_im))

        # legendre transformation: contract over l (stride-1 in both operands), block by block
        out = _synthesis_contraction(x, [p.to(x.dtype) for p in _blocks(self)], self._layout)
        # force contiguous: the einsum output is non-contiguous and inductor's aten.complex meta
        # predicts a contiguous layout, tripping assert_size_stride under torch.compile (see fwd SHT).
        x = torch.complex(out[0].contiguous(), out[1].contiguous())

        # apply the inverse (real) FFT
        if self.ring_fft is None:
            return irfft(x, n=self.nlon, dim=-1, norm="forward")
        return self.ring_fft.inverse(x)


class RealVectorSHT(SharedBufferMixin, nn.Module):
    r"""
    Defines a module for computing the forward (real) vector SHT.
    Precomputes the associated Legendre polynomials and quadrature weights of the given grid.
    The SHT is applied to the last three dimensions of the input.

    Decomposes a tangential vector field
    :math:`\mathbf{v} = v_\theta\,\hat{e}_\theta + v_\lambda\,\hat{e}_\lambda`
    into **spheroidal** and **toroidal** spectral coefficients
    :math:`\hat{s}_l^m` and :math:`\hat{t}_l^m` using the derivatives of the
    associated Legendre polynomials.

    On equiangular and Fejér-1 grids, explicit ``lmax > grid.max_exact_degree``
    selects the same precomputed folded analysis as :class:`~torch_harmonics.RealSHT`,
    under the same recovery limits, with the tangential parity of the derivative
    components.

    .. seealso::
        :doc:`/guide/spherical_harmonic_transforms`
            User guide with the full mathematical derivation of the vector SHT
            formulas, normalization conventions, and worked examples.

    On a :class:`~torch_harmonics.healpix.HealpixGrid` the field is ``(..., 2, npix)``
    and the transform works as the scalar :class:`RealSHT` does there: an exact ring
    Fourier transform with aliasing, and analysis with the ring weights, approximate
    to about :math:`10^{-2}` at the default ``lmax = 2 * nside`` and refined by
    ``iter``. The coefficients relate to healpy's spin-1 gradient and curl ones by
    :math:`a^E_{lm} = \sqrt{l(l+1)}\, \hat{s}_l^m` and
    :math:`a^B_{lm} = -\sqrt{l(l+1)}\, \hat{t}_l^m`.

    Parameters
    ----------
    grid : RegularGridS2 or HealpixGrid
        Descriptor of the spatial grid the transform operates on. It carries the
        resolution as well as the quadrature rule, so no separate ``nlat``/``nlon``
        is needed. Build one with :func:`torch_harmonics.grid.as_grid`. A
        :class:`~torch_harmonics.radial.SphericalProductGrid` over one transforms every shell
        alike, with the radial axis in front of the spatial (or spectral) ones.
    lmax : int, optional
        Non-inclusive maximum spherical harmonic degree.
    mmax : int, optional
        Non-inclusive maximum spherical harmonic order.
    lmmax : int, optional
        Non-inclusive upper bound on degree minus order: retain only ``l - m < lmmax``.
        ``None`` leaves this bandwidth unrestricted.
    norm : str
        Normalization convention (``"ortho"``, ``"schmidt"``, ``"unnorm"``),
        by default ``"ortho"``.
    csphase : bool
        Whether to include the Condon--Shortley phase factor :math:`(-1)^m`,
        by default ``True``.
    block_size : int or None
        Number of orders per block of the stored Legendre tables, or ``None`` for one dense
        table. A block keeps only the degrees its orders have, :math:`l \ge m`, which roughly
        halves the tables and the contraction of a triangular truncation. By default
        ``$TORCH_HARMONICS_SHT_BLOCK_SIZE`` (an integer, or ``none``) if set, else 32. Transforms
        share their tables only with transforms of the same block size.
    iter : int
        Number of Jacobi iterations refining the analysis, by default 0; see
        :class:`RealSHT`. ``iter=3`` is recommended on HEALPix when precision matters.

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
    :cite:`Schaeffer2013`, :cite:`Wang2018`, :cite:`Reinecke2023`, :cite:`DUCC`
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
        lmmax: Optional[int] = None,
        *,
        norm: Optional[str] = "ortho",
        csphase: Optional[bool] = True,
        block_size: Optional[int] = BLOCK_SIZE_FROM_ENVIRONMENT,
        iter: int = 0,
    ):

        super().__init__()

        # on a SphericalProductGrid every shell is transformed alike; the radial grid only
        # adds its axis to the field shape
        full_grid = grid
        grid, self.radial = split_radial(grid)
        self.grid = _require_sht_grid(grid)
        self.norm = norm
        self.csphase = csphase
        if isinstance(iter, bool) or not isinstance(iter, int) or iter < 0:
            raise ValueError(f"iter must be a non-negative integer, got {iter!r}")
        self.iter = iter

        if isinstance(self.grid, RegularGridS2):
            _warn_if_not_spectrally_accurate(self.grid)

        self._trunc = truncate_sht(self.grid, lmax, mmax, lmmax)
        _validate_forward_sht_degree(self.grid, self.lmax)
        self._extended_folded = _extended_folded_analysis(self.grid, self._trunc)

        if isinstance(self.grid, RegularGridS2):
            self.nlat, self.nlon = self.grid.shape
            self.ring_fft = None
        else:
            # HEALPix: ragged rings, an exact ring FFT with aliasing, and approximate
            # analysis with the ring weights (see _quadrature_weights)
            self.nlat, self.nlon = self.grid.nrings, None
            self.ring_fft = RingFFT(self.grid, self.mmax)

        # quadrature-weighted derivative rows in order blocks, shared with every transform of the same kind
        self.block_size = _resolve_block_size(block_size, self.mmax)
        blocks = _sht_blocks(self._trunc, self.block_size, 0, self.mmax, 0, self.lmax)
        tables = _analysis_weights(self.grid, self._trunc, self.norm, self.csphase, True, 0, self.mmax, 0, self.nlat, self.block_size)
        _register_blocks(self, "weights", tables, blocks, 0)

        # Jacobi iteration, see RealSHT. Built on the full grid, radial axis included: the
        # vector contraction finds the component axis by its position, which the radial
        # axis shifts.
        self._synthesis = None
        if self.iter > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self._synthesis = InverseRealVectorSHT(full_grid, self.lmax, self.mmax, self.lmmax, norm=self.norm, csphase=self.csphase, block_size=self.block_size)

    @property
    def weights(self) -> torch.Tensor:
        """The dense analysis tables ``(2, mmax, lmax, nlat)``, assembled from the blocks: a fresh tensor, for inspection."""
        return _dense_table(self, -2, self.lmax)

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
        radial = "" if self.radial is None else f",\nradial={self.radial!r}"
        return f"grid={self.grid!r}{radial},\nlmax={self.lmax}, mmax={self.mmax}, lmmax={self.lmmax}, csphase={self.csphase}" + (f", iter={self.iter}" if self.iter else "")

    def forward(self, x: torch.Tensor):
        """
        Compute the forward (real) vector spherical harmonic transform.

        Parameters
        ----------
        x : torch.Tensor
            Real-valued tangential vector field of shape ``(..., 2, *grid.shape)``, where the
            size-2 dimension holds the two tangential (colatitude, longitude) components:
            ``(..., 2, nlat, nlon)`` on a regular grid and ``(..., 2, npix)`` on HEALPix.

        Returns
        -------
        torch.Tensor
            Complex vector harmonic coefficients of shape ``(..., 2, lmax, mmax)``, where the
            size-2 dimension holds the spheroidal and toroidal components.
        """

        check_field(x, self.grid, leading=(..., 2), radial=self.radial)

        # Jacobi iteration in real and imaginary parts, see RealSHT.forward
        re, im = self._analyze(x)
        for _ in range(self.iter):
            residual = x - self._synthesis._synthesize_parts(re.transpose(-1, -2).contiguous(), im.transpose(-1, -2).contiguous())
            dre, dim = self._analyze(residual)
            re, im = re + dre, im + dim
        return torch.complex(re, im)

    def _analyze(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Fourier transform in longitude, per component. The 2*pi scale factor is folded
        # into the quadrature weights, so no scaling of the complex output is needed here.
        if self.ring_fft is None:
            x = rfft(x, nmodes=self.mmax, dim=-1, norm="forward")
        else:
            x = self.ring_fft(x)

        # transpose to put the contraction dim (nlat) on the fast axis, real and imaginary
        # parts stacked so that each block is one contraction per derivative table
        x = _real_and_imaginary(x.transpose(-1, -2))

        # the component axis, in front of the radial one on a product grid
        cdim = -3 if self.radial is None else -4

        tables = [w.to(x.dtype) for w in _blocks(self)]
        y0 = _analysis_contraction(x, [w[0] for w in tables], self._layout, self.lmax)
        y1 = _analysis_contraction(x, [w[1] for w in tables], self._layout, self.lmax)
        (x0_re, x0_im), (x1_re, x1_im) = _components(y0, cdim)
        (z0_re, z0_im), (z1_re, z1_im) = _components(y1, cdim)

        # spheroidal and toroidal components; xc_* holds component c contracted with the
        # first derivative table, zc_* with the second
        s_re = x0_re - z1_im
        s_im = x0_im + z1_re
        t_re = -z0_im - x1_re
        t_im = z0_re - x1_im

        # stack the spheroidal and toroidal components in real space, so the only complex-typed
        # op is a single aten.complex over contiguous operands. Stacking complex tensors instead
        # would leave a complex cat, which inductor cannot codegen (triton has no complex type),
        # and feeding aten.complex the non-contiguous ...lm einsum outputs directly trips
        # assert_size_stride, as its meta predicts a contiguous layout. torch.stack allocates a
        # fresh contiguous buffer, which is exactly what that meta expects.
        out_re = torch.stack((s_re, t_re), dim=cdim)
        out_im = torch.stack((s_im, t_im), dim=cdim)
        return out_re, out_im


class InverseRealVectorSHT(SharedBufferMixin, nn.Module):
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
    grid : RegularGridS2 or HealpixGrid
        Descriptor of the spatial grid the transform operates on. It carries the
        resolution as well as the quadrature rule, so no separate ``nlat``/``nlon``
        is needed. Build one with :func:`torch_harmonics.grid.as_grid`. A
        :class:`~torch_harmonics.radial.SphericalProductGrid` over one transforms every shell
        alike, with the radial axis in front of the spatial (or spectral) ones.
    lmax : int, optional
        Non-inclusive maximum spherical harmonic degree.
    mmax : int, optional
        Non-inclusive maximum spherical harmonic order.
    lmmax : int, optional
        Non-inclusive upper bound on degree minus order: retain only ``l - m < lmmax``.
        ``None`` leaves this bandwidth unrestricted.
    norm : str
        Normalization convention (``"ortho"``, ``"schmidt"``, ``"unnorm"``),
        by default ``"ortho"``.
    csphase : bool
        Whether to include the Condon--Shortley phase factor :math:`(-1)^m`,
        by default ``True``.
    block_size : int or None
        Number of orders per block of the stored Legendre tables, or ``None`` for one dense
        table. A block keeps only the degrees its orders have, :math:`l \ge m`, which roughly
        halves the tables and the contraction of a triangular truncation. By default
        ``$TORCH_HARMONICS_SHT_BLOCK_SIZE`` (an integer, or ``none``) if set, else 32. Transforms
        share their tables only with transforms of the same block size.

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

    On a :class:`~torch_harmonics.healpix.HealpixGrid` the output is ``(..., 2, npix)``
    and the synthesis is exact, as for :class:`InverseRealSHT`.

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
        lmmax: Optional[int] = None,
        *,
        norm: Optional[str] = "ortho",
        csphase: Optional[bool] = True,
        block_size: Optional[int] = BLOCK_SIZE_FROM_ENVIRONMENT,
    ):

        super().__init__()

        # on a SphericalProductGrid every shell is transformed alike; the radial grid only
        # adds its axis to the field shape
        grid, self.radial = split_radial(grid)
        self.grid = _require_sht_grid(grid)
        self.norm = norm
        self.csphase = csphase

        if isinstance(self.grid, RegularGridS2):
            _warn_if_not_spectrally_accurate(self.grid)

        self._trunc = truncate_sht(self.grid, lmax, mmax, lmmax)

        if isinstance(self.grid, RegularGridS2):
            self.nlat, self.nlon = self.grid.shape
            self.ring_fft = None
        else:
            # synthesis on HEALPix is exact: it only evaluates the expansion at the pixels
            self.nlat, self.nlon = self.grid.nrings, None
            self.ring_fft = RingFFT(self.grid, self.mmax)

        # derivative Legendre tables in order blocks of (2, orders, nlat, degrees), so the
        # contraction dim l is stride-1; shared with every transform of the same kind
        self.block_size = _resolve_block_size(block_size, self.mmax)
        blocks = _sht_blocks(self._trunc, self.block_size, 0, self.mmax, 0, self.lmax)
        tables = _synthesis_table(self.grid, self._trunc, self.norm, self.csphase, True, 0, self.mmax, 0, self.lmax, self.block_size)
        _register_blocks(self, "dpct", tables, blocks, 0)

    @property
    def dpct(self) -> torch.Tensor:
        """The dense synthesis tables ``(2, mmax, nlat, lmax)``, assembled from the blocks: a fresh tensor, for inspection."""
        return _dense_table(self, -1, self.lmax)

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
        radial = "" if self.radial is None else f",\nradial={self.radial!r}"
        return f"grid={self.grid!r}{radial},\nlmax={self.lmax}, mmax={self.mmax}, lmmax={self.lmmax}, csphase={self.csphase}"

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
            Real-valued tangential vector field of shape ``(..., 2, *grid.shape)``, where the
            size-2 dimension holds the two tangential (colatitude, longitude) components:
            ``(..., 2, nlat, nlon)`` on a regular grid and ``(..., 2, npix)`` on HEALPix.
        """

        check_coeffs(x, self.lmax, self.mmax, leading=(..., 2), radial=self.radial)
        return self._synthesize(x)

    def _synthesize(self, x: torch.Tensor) -> torch.Tensor:
        # transpose to put the contraction dim (lmax) on the fast axis
        x = x.transpose(-1, -2)
        return self._synthesize_parts(x.real, x.imag)

    def _synthesize_parts(self, x_re: torch.Tensor, x_im: torch.Tensor) -> torch.Tensor:
        # the real and imaginary parts of the coefficients, (..., 2, mmax, lmax) each, stacked so that each
        # block is one contraction
        x = torch.stack((x_re, x_im))

        # dpct blocks: (2, orders, nlat, degrees) -- contract over l (stride-1 in both operands)
        # the component axis, in front of the radial one on a product grid
        cdim = -3 if self.radial is None else -4

        tables = [d.to(x.dtype) for d in _blocks(self)]
        y0 = _synthesis_contraction(x, [d[0] for d in tables], self._layout)
        y1 = _synthesis_contraction(x, [d[1] for d in tables], self._layout)
        (x0_re, x0_im), (x1_re, x1_im) = _components(y0, cdim)
        (z0_re, z0_im), (z1_re, z1_im) = _components(y1, cdim)

        # spheroidal and toroidal components; xc_* holds component c contracted with the
        # first derivative table, zc_* with the second
        srl = x0_re - z1_im
        sim = x0_im + z1_re
        trl = -z0_im - x1_re
        tim = z0_re - x1_im

        # reassemble in real space and apply inverse FFT, see RealVectorSHT.forward
        out_re = torch.stack((srl, trl), dim=cdim)
        out_im = torch.stack((sim, tim), dim=cdim)
        xs = torch.complex(out_re, out_im)
        if self.ring_fft is None:
            return irfft(xs, n=self.nlon, dim=-1, norm="forward")
        return self.ring_fft.inverse(xs)
