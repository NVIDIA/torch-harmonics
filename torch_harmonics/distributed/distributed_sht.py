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

import warnings
from typing import Optional

import torch
import torch.nn as nn

from torch_harmonics._field_contract import check_coeffs, check_field
from torch_harmonics.cache import SharedBufferMixin
from torch_harmonics.fft import RingFFT, irfft, rfft
from torch_harmonics.grid import GridS2, RegularGridS2, SpectralGrid, _rejects_legacy_signature
from torch_harmonics.radial import split_radial
from torch_harmonics.sht import (
    BLOCK_SIZE_FROM_ENVIRONMENT,
    _analysis_contraction,
    _analysis_weights,
    _blocks,
    _components,
    _dense_table,
    _extended_folded_analysis,
    _real_and_imaginary,
    _register_blocks,
    _require_sht_grid,
    _resolve_block_size,
    _sht_blocks,
    _synthesis_contraction,
    _synthesis_table,
)
from torch_harmonics.truncation import _validate_forward_sht_degree, _warn_if_not_spectrally_accurate, truncate_sht

from .primitives import (
    compute_split_shapes,
    distributed_transpose_azimuth,
    flatten_and_pad_leading_dims,
    reduce_from_scatter_to_polar_region,
    unpad_and_unflatten_leading_dims,
)
from .utils import azimuth_group_rank, azimuth_group_size, polar_group_rank, polar_group_size


def _setup_grid(module: nn.Module, grid: GridS2) -> None:
    """
    The global grid of a distributed SHT and the communicator layout it is split over.

    A :class:`~torch_harmonics.grid.RegularGridS2` or a :class:`~torch_harmonics.healpix.HealpixGrid`.
    On HEALPix a ring stands in for a latitude: ``nlat`` counts the rings and there is no
    ``nlon``, since the rings differ in length.
    """
    module.grid = _require_sht_grid(grid)
    if isinstance(module.grid, RegularGridS2):
        module.nlat, module.nlon = module.grid.shape
        _warn_if_not_spectrally_accurate(module.grid)
    else:
        module.nlat, module.nlon = module.grid.nrings, None

    module.comm_size_polar = polar_group_size()
    module.comm_rank_polar = polar_group_rank()
    module.comm_size_azimuth = azimuth_group_size()
    module.comm_rank_azimuth = azimuth_group_rank()


def _setup_shard(module: nn.Module) -> None:
    """
    This rank's spatial shard, the extents the collectives split by, and the longitudinal transform.

    The grid decomposes itself in space. A regular grid splits its latitudes over the polar
    group and its longitudes over the azimuth group, and the longitudes are made local by the
    azimuth transposes before the FFT. A HEALPix grid splits into bands of whole rings over the
    polar group only, cut by pixel count rather than evenly, and refuses an azimuth split: each
    band is a contiguous slice of the flat pixel axis, so the ring FFT of its own rings is
    local and no transpose is needed. ``lat_shapes`` holds the rings of every rank then, which
    is what the synthesis reduce-scatter splits by.
    """
    module.shard = module.grid.shard(
        polar=(module.comm_rank_polar, module.comm_size_polar),
        azimuth=(module.comm_rank_azimuth, module.comm_size_azimuth),
    )
    if isinstance(module.grid, RegularGridS2):
        module.lat_shapes = list(module.shard.lat_shapes)
        module.lon_shapes = list(module.shard.lon_shapes)
        module.nlat_local = module.shard.nlat
        module.nlon_local = module.shard.nlon
    else:
        module.lat_shapes = list(module.shard.ring_shapes)
        module.lon_shapes = None
        module.pixel_shapes = list(module.shard.pixel_shapes)
        module.nlat_local = module.shard.nrings
        module.nlon_local = None
    module.lat_offset = sum(module.lat_shapes[: module.comm_rank_polar])

    if isinstance(module.grid, RegularGridS2):
        module.ring_fft = None
    else:
        module.ring_fft = RingFFT(module.grid, module.mmax, rings=(module.lat_offset, module.lat_offset + module.nlat_local))


def _validate_iter(iter: int) -> int:
    if isinstance(iter, bool) or not isinstance(iter, int) or iter < 0:
        raise ValueError(f"iter must be a non-negative integer, got {iter!r}")
    return iter


def _setup_jacobi(module: nn.Module, grid: GridS2, inverse_cls: type) -> None:
    """
    The distributed synthesis a forward transform with ``iter > 0`` refines its analysis with.

    ``grid`` is the grid the forward was given, the radial factor included, so that the
    synthesis takes and returns the forward's own layouts. Its bounds are already resolved,
    so the inverse cannot warn about a changed default.
    """
    module._synthesis = None
    if module.iter > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            module._synthesis = inverse_cls(grid, module.lmax, module.mmax, module.lmmax, norm=module.norm, csphase=module.csphase, block_size=module.block_size)


def _vector_radial_dims(module: nn.Module, spatial: bool) -> tuple:
    """
    The ``(component, radial)`` axes of a vector field on a product grid, counted from the end.

    A product field is ``(..., 2, nr, *spatial)``; the transforms move the radial axis in
    front of the component one, where it is just another leading axis, and back at the end.
    The spatial shape is ``(nlat, nlon)`` on a regular grid and ``(npix,)`` on HEALPix, the
    spectral one is always ``(lmax, mmax)``.
    """
    ndims = len(module.shard.shape) if spatial else 2
    return -(ndims + 2), -(ndims + 1)


class DistributedRealSHT(SharedBufferMixin, nn.Module):
    """
    Distributed version of the forward (real-valued) SHT.
    Precomputes the associated Legendre polynomials and quadrature weights of the given grid.
    The SHT is applied to the last two dimensions of the input.

    The endpoints ``grid_in`` and ``grid_out`` describe global domains; ``grid``
    remains the global spatial descriptor.

    Folded equiangular CC and Fejér-1 analysis use the same precomputed real
    effective projection as :class:`~torch_harmonics.RealSHT`. Each rank builds
    only its global order range on the full meridian and retains its latitude
    shard. Forward and backward use the existing collectives without latitude
    gathers.

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

    The precomputed Legendre weights are split across the full process grid rather
    than replicated. Because the quadrature sum is accumulated across ranks, results
    are not bitwise identical to the serial transform. A leading axis ``N`` smaller
    than the group size is supported; it is zero-padded internally, which is exact.

    **HEALPix.** On a :class:`~torch_harmonics.healpix.HealpixGrid` the input is flat,
    ``(B, C, npix_local)``: each polar rank holds a band of whole rings, cut so that every
    rank holds about the same number of pixels (see
    :class:`~torch_harmonics.healpix.HealpixGridShardS2`), and there is no azimuth split.
    Steps 1 to 3 collapse into the ring FFT of the local rings, which needs no
    communication; steps 4 and 5 are unchanged. The output is split as on a regular grid.

    .. seealso::
        :class:`torch_harmonics.RealSHT`
            Serial counterpart with full mathematical description and parameter
            documentation.

    Parameters
    ----------
    grid : RegularGridS2 or HealpixGrid
        Descriptor of the spatial grid the transform operates on. It carries the
        resolution as well as the quadrature rule, so no separate ``nlat``/``nlon``
        is needed. Build one with :func:`torch_harmonics.grid.as_grid`.
        The grid is the *global* one; the local shard is derived from it.
    lmax : int, optional
        Non-inclusive maximum spherical harmonic degree.
    mmax : int, optional
        Non-inclusive maximum spherical harmonic order.
    lmmax : int, optional
        Non-inclusive upper bound on degree minus order: retain only ``l - m < lmmax``.
        ``None`` leaves this bandwidth unrestricted.
    norm : str
        Normalization type (``"ortho"``, ``"schmidt"``, ``"unnorm"``), by default ``"ortho"``
    csphase : bool
        Whether to apply the Condon-Shortley phase factor, by default True
    block_size : int or None
        Number of orders per block of the stored Legendre tables, or ``None`` for one dense
        table, see :class:`~torch_harmonics.RealSHT`. The blocks are cut at the global
        multiples of ``block_size``, so a rank holds the serial blocks of its orders.
    iter : int
        Number of Jacobi iterations refining the analysis, by default 0, see
        :class:`~torch_harmonics.RealSHT`. Each one adds a distributed synthesis and analysis,
        and with them their collectives.

    Returns
    -------
    torch.Tensor
        Local spectral coefficients, shape ``(..., lmax_local, mmax_local)``. On HEALPix
        ``mmax_local`` is ``mmax``, since there is no azimuth split.

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
        iter: int = 0,
    ):

        super().__init__()

        # on a SphericalProductGrid every shell is transformed alike; the radial axis is one
        # more leading axis, flattened into the one the transposes split
        input_grid = grid
        grid, self.radial = split_radial(grid)
        _setup_grid(self, grid)
        self.norm = norm
        self.csphase = csphase
        self.iter = _validate_iter(iter)

        self._trunc = truncate_sht(self.grid, lmax, mmax, lmmax)
        _validate_forward_sht_degree(self.grid, self.lmax)

        # compute splits: the grid decomposes itself in space; the spectral split stays manual,
        # since lmax/mmax are not grid dimensions
        _setup_shard(self)
        self.l_shapes = compute_split_shapes(self.lmax, self.comm_size_polar)
        self.lmax_local = self.l_shapes[self.comm_rank_polar]
        self.m_shapes = compute_split_shapes(self.mmax, self.comm_size_azimuth)
        self.mmax_local = self.m_shapes[self.comm_rank_azimuth]
        self.mmax_offset = sum(self.m_shapes[: self.comm_rank_azimuth])

        # build only the block this rank keeps: local orders, all degrees, local latitudes. The
        # contraction over k is a distributed matmul completed by a reduce-scatter in the forward,
        # so only the local latitudes are needed; l is contracted in full. The block is shared
        # with every transform of the same kind on this rank's block, see _analysis_weights.
        self._extended_folded = _extended_folded_analysis(self.grid, self._trunc)
        self.block_size = _resolve_block_size(block_size, self.mmax)
        mmin, mmax = self.mmax_offset, self.mmax_offset + self.mmax_local
        blocks = _sht_blocks(self._trunc, self.block_size, mmin, mmax, 0, self.lmax)
        tables = _analysis_weights(self.grid, self._trunc, self.norm, self.csphase, False, mmin, mmax, self.lat_offset, self.lat_offset + self.nlat_local, self.block_size)
        _register_blocks(self, "weights", tables, blocks, 0)

        # the Jacobi iteration refines the analysis with the distributed synthesis, see RealSHT
        _setup_jacobi(self, input_grid, DistributedInverseRealSHT)

    @property
    def weights(self) -> torch.Tensor:
        """This rank's dense analysis table ``(mmax_local, lmax, nlat_local)``, assembled from the blocks: a fresh tensor, for inspection."""
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
    def grid_in(self) -> GridS2:
        """Global spatial domain of the input field."""
        return self.grid

    @property
    def grid_out(self) -> SpectralGrid:
        """Global spectral coefficient support of the output."""
        return self._trunc

    def extra_repr(self):
        radial = "" if self.radial is None else f",\nradial={self.radial!r}"
        return f"grid={self.grid!r}{radial},\nlmax={self.lmax}, mmax={self.mmax}, lmmax={self.lmmax}, csphase={self.csphase}" + (f", iter={self.iter}" if self.iter else "")

    # This transform cannot be captured in a single graph: the redistribution collectives it
    # calls are themselves torch.compiler.disable()d, so at comm_size > 1 dynamo breaks at
    # every one of them and inductor only ever sees the slivers in between. Those slivers hold
    # complex intermediates, which triton cannot type (KeyError: 'complex64' in codegen), and
    # compiling them buys nothing next to the all-to-alls surrounding them. Disabling the whole
    # forward costs no fusion that the breaks had not already cost, and keeps the complex
    # spectral data out of inductor entirely. The serial transforms are compiled as usual.
    @torch.compiler.disable()
    def forward(self, x: torch.Tensor):

        # a channel axis is required: the transposes split it across the ranks
        check_field(x, self.shard, leading=(..., None), radial=self.radial)

        # Jacobi iteration a <- a + A (f - S a), see RealSHT.forward. Every step is one more
        # distributed synthesis and analysis, each with its own collectives.
        coeffs = self._analyze(x)
        for _ in range(self.iter):
            coeffs = coeffs + self._analyze(x - self._synthesis(coeffs))
        return coeffs

    def _analyze(self, x: torch.Tensor) -> torch.Tensor:

        # the azimuth transposes below redistribute the leading (channel/batch) axis, so it
        # must be at least as large as the azimuth group. Flatten all leading dims into that
        # axis and zero-pad it if needed (linear transform, so padding stays zero); restore
        # the original layout before returning. The polar group never touches this axis --
        # the latitude contraction is a distributed matmul, not a transpose.
        x, lead_shape, lead_size = flatten_and_pad_leading_dims(x, self.comm_size_azimuth, num_trailing_dims=len(self.shard.shape))
        num_chans = x.shape[-1 - len(self.shard.shape)]

        if self.ring_fft is None:
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
        else:
            # HEALPix: this rank holds whole rings, so their FFT is local, (..., npix_local) ->
            # (..., nrings_local, mmax) with every order on every rank
            x = self.ring_fft(x)

        # transpose to put the contraction dim (nlat) on the fast axis. nlat stays split across
        # the polar group: each rank contracts the latitudes it owns and the reduce-scatter
        # below completes the sum.
        x = _real_and_imaginary(x.transpose(-1, -2))

        # Legendre-Gauss quadrature: partial contraction over the local k=nlat chunk, block by block
        out = _analysis_contraction(x, [w.to(x.dtype) for w in _blocks(self)], self._layout, self.lmax)
        out_re, out_im = out[0], out[1]

        # complete the quadrature sum and split l in a single collective. This runs on the real
        # view: the reduce-scatter reduces in fp32, which would discard the imaginary part of a
        # complex tensor, and NCCL has no complex reduction to begin with.
        out = torch.stack((out_re, out_im), dim=-1)
        if self.comm_size_polar > 1:
            out = reduce_from_scatter_to_polar_region(out, -3, self.l_shapes)
        x = torch.view_as_complex(out.contiguous())

        # drop padding and restore the original leading dims
        x = unpad_and_unflatten_leading_dims(x, lead_shape, lead_size)

        return x


class DistributedInverseRealSHT(SharedBufferMixin, nn.Module):
    """
    Distributed version of the inverse (real-valued) SHT.
    Precomputes the associated Legendre polynomials on the nodes of the given grid.

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

    The precomputed Legendre polynomials are split across the full process grid
    rather than replicated. Because the synthesis sum is accumulated across ranks,
    results are not bitwise identical to the serial transform. A leading axis ``N``
    smaller than the group size is supported; it is zero-padded internally, which is
    exact.

    **HEALPix.** On a :class:`~torch_harmonics.healpix.HealpixGrid` the reduce-scatter of
    step 2 leaves each polar rank its band of rings, split by pixel count rather than evenly,
    and steps 3 to 5 collapse into the inverse ring FFT of those rings. The output is flat,
    ``(B, C, npix_local)``.

    .. seealso::
        :class:`torch_harmonics.InverseRealSHT`
            Serial counterpart with full mathematical description and parameter
            documentation.

    Parameters
    ----------
    grid : RegularGridS2 or HealpixGrid
        Descriptor of the spatial grid the transform operates on. It carries the
        resolution as well as the quadrature rule, so no separate ``nlat``/``nlon``
        is needed. Build one with :func:`torch_harmonics.grid.as_grid`.
        The grid is the *global* one; the local shard is derived from it.
    lmax : int, optional
        Non-inclusive maximum spherical harmonic degree.
    mmax : int, optional
        Non-inclusive maximum spherical harmonic order.
    lmmax : int, optional
        Non-inclusive upper bound on degree minus order: retain only ``l - m < lmmax``.
        ``None`` leaves this bandwidth unrestricted.
    norm : str
        Normalization type (``"ortho"``, ``"schmidt"``, ``"unnorm"``), by default ``"ortho"``
    csphase : bool
        Whether to apply the Condon-Shortley phase factor, by default True
    block_size : int or None
        Number of orders per block of the stored Legendre tables, or ``None`` for one dense
        table, see :class:`~torch_harmonics.RealSHT`. The blocks are cut at the global
        multiples of ``block_size``, so a rank holds the serial blocks of its orders.

    Returns
    -------
    torch.Tensor
        Local spatial signal, shape ``(..., nlat_local, nlon_local)``, or ``(..., npix_local)``
        on HEALPix.

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

        # on a SphericalProductGrid every shell is transformed alike; the radial axis is one
        # more leading axis, flattened into the one the transposes split
        grid, self.radial = split_radial(grid)
        _setup_grid(self, grid)
        self.norm = norm
        self.csphase = csphase

        self._trunc = truncate_sht(self.grid, lmax, mmax, lmmax)

        # compute splits: the grid decomposes itself in space; the spectral split stays manual,
        # since lmax/mmax are not grid dimensions
        _setup_shard(self)
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
        # stored as (mmax_local, nlat, lmax_local) so the contraction dim l is stride-1, and
        # shared with every transform of the same kind on this rank's block
        self.block_size = _resolve_block_size(block_size, self.mmax)
        mmin, mmax = self.mmax_offset, self.mmax_offset + self.mmax_local
        lmin, lmax = self.lmax_offset, self.lmax_offset + self.lmax_local
        blocks = _sht_blocks(self._trunc, self.block_size, mmin, mmax, lmin, lmax)
        tables = _synthesis_table(self.grid, self._trunc, self.norm, self.csphase, False, mmin, mmax, lmin, lmax, self.block_size)
        _register_blocks(self, "pct", tables, blocks, lmin)

    @property
    def pct(self) -> torch.Tensor:
        """This rank's dense synthesis table ``(mmax_local, nlat, lmax_local)``, assembled from the blocks: a fresh tensor, for inspection."""
        return _dense_table(self, -1, self.lmax_local)

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
        """Global spectral coefficient support of the input."""
        return self._trunc

    @property
    def grid_out(self) -> GridS2:
        """Global spatial grid sampled by the output field."""
        return self.grid

    def extra_repr(self):
        radial = "" if self.radial is None else f",\nradial={self.radial!r}"
        return f"grid={self.grid!r}{radial},\nlmax={self.lmax}, mmax={self.mmax}, lmmax={self.lmmax}, csphase={self.csphase}"

    # This transform cannot be captured in a single graph: the redistribution collectives it
    # calls are themselves torch.compiler.disable()d, so at comm_size > 1 dynamo breaks at
    # every one of them and inductor only ever sees the slivers in between. Those slivers hold
    # complex intermediates, which triton cannot type (KeyError: 'complex64' in codegen), and
    # compiling them buys nothing next to the all-to-alls surrounding them. Disabling the whole
    # forward costs no fusion that the breaks had not already cost, and keeps the complex
    # spectral data out of inductor entirely. The serial transforms are compiled as usual.
    @torch.compiler.disable()
    def forward(self, x: torch.Tensor):

        check_coeffs(x, self.lmax_local, self.mmax_local, leading=(..., None), radial=self.radial)

        # the azimuth transposes below redistribute the leading (channel/batch) axis, so it
        # must be at least as large as the azimuth group. Flatten all leading dims into that
        # axis and zero-pad it if needed (linear transform, so padding stays zero); restore
        # the original layout before returning. The polar group never touches this axis --
        # the degree contraction is a distributed matmul, not a transpose.
        x, lead_shape, lead_size = flatten_and_pad_leading_dims(x, self.comm_size_azimuth)

        # transpose to put the contraction dim (lmax) on the fast axis. l stays split across
        # the polar group: each rank synthesizes from the degrees it owns and the
        # reduce-scatter below completes the sum.
        x = _real_and_imaginary(x.transpose(-1, -2))

        # legendre transformation: partial contraction over the local l chunk, block by block
        # pct blocks: (orders, nlat, local degrees)
        out = _synthesis_contraction(x, [p.to(x.dtype) for p in _blocks(self)], self._layout)
        out_re, out_im = out[0], out[1]

        # complete the synthesis sum and split nlat in a single collective, on the real view
        # (see DistributedRealSHT.forward for why the collective cannot take complex input).
        out = torch.stack((out_re, out_im), dim=-1)
        if self.comm_size_polar > 1:
            out = reduce_from_scatter_to_polar_region(out, -3, self.lat_shapes)
        x = torch.view_as_complex(out.contiguous())

        if self.ring_fft is None:
            num_chans = x.shape[-3]

            # transpose: after this, channels are split and m is local
            if self.comm_size_azimuth > 1:
                x = distributed_transpose_azimuth(x, (-3, -1), self.m_shapes)

            # apply the inverse (real) FFT
            x = irfft(x, n=self.nlon, dim=-1, norm="forward")

            # transpose: after this, m is split and channels are local
            if self.comm_size_azimuth > 1:
                chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
                x = distributed_transpose_azimuth(x, (-1, -3), chan_shapes)
        else:
            # HEALPix: every order of this rank's rings is local, (..., nrings_local, mmax) ->
            # (..., npix_local)
            x = self.ring_fft.inverse(x)

        # drop padding and restore the original leading dims
        x = unpad_and_unflatten_leading_dims(x, lead_shape, lead_size, num_trailing_dims=len(self.shard.shape))

        return x


class DistributedRealVectorSHT(SharedBufferMixin, nn.Module):
    """
    Distributed version of the forward (real) vector SHT.
    Precomputes the associated Legendre polynomials and quadrature weights of the given grid.
    The SHT is applied to the last three dimensions of the input.

    The distribution scheme is the same as for
    :class:`DistributedRealSHT` (see its docstring for a step-by-step
    description of the all-to-all transposes over the ``N = B * C`` axis).
    The additional size-2 vector component dimension is preserved throughout, and
    HEALPix is handled as there, with the field ``(B, C, 2, npix_local)``.

    .. seealso::
        :class:`torch_harmonics.RealVectorSHT`
            Serial counterpart with full mathematical description and parameter
            documentation.

    Folded equiangular CC and Fejér-1 analysis use the same precomputed real
    effective projection as :class:`~torch_harmonics.RealVectorSHT`, with
    global-order tangential parity. It adds no forward or backward collectives.

    Parameters
    ----------
    grid : RegularGridS2 or HealpixGrid
        Descriptor of the spatial grid the transform operates on. It carries the
        resolution as well as the quadrature rule, so no separate ``nlat``/``nlon``
        is needed. Build one with :func:`torch_harmonics.grid.as_grid`.
        The grid is the *global* one; the local shard is derived from it.
    lmax : int, optional
        Non-inclusive maximum spherical harmonic degree.
    mmax : int, optional
        Non-inclusive maximum spherical harmonic order.
    lmmax : int, optional
        Non-inclusive upper bound on degree minus order: retain only ``l - m < lmmax``.
        ``None`` leaves this bandwidth unrestricted.
    norm : str
        Normalization type (``"ortho"``, ``"schmidt"``, ``"unnorm"``), by default ``"ortho"``
    csphase : bool
        Whether to apply the Condon-Shortley phase factor, by default True
    block_size : int or None
        Number of orders per block of the stored Legendre tables, or ``None`` for one dense
        table, see :class:`~torch_harmonics.RealSHT`. The blocks are cut at the global
        multiples of ``block_size``, so a rank holds the serial blocks of its orders.
    iter : int
        Number of Jacobi iterations refining the analysis, by default 0, see
        :class:`~torch_harmonics.RealSHT`. Each one adds a distributed synthesis and analysis,
        and with them their collectives.

    Returns
    -------
    torch.Tensor
        Local spectral coefficients, shape ``(..., 2, lmax_local, mmax_local)``.

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
        iter: int = 0,
    ):

        super().__init__()

        # on a SphericalProductGrid every shell is transformed alike; the radial axis is one
        # more leading axis, flattened into the one the transposes split
        input_grid = grid
        grid, self.radial = split_radial(grid)
        _setup_grid(self, grid)
        self.norm = norm
        self.csphase = csphase
        self.iter = _validate_iter(iter)

        self._trunc = truncate_sht(self.grid, lmax, mmax, lmmax)
        _validate_forward_sht_degree(self.grid, self.lmax)

        # compute splits: the grid decomposes itself in space; the spectral split stays manual,
        # since lmax/mmax are not grid dimensions
        _setup_shard(self)
        self.l_shapes = compute_split_shapes(self.lmax, self.comm_size_polar)
        self.m_shapes = compute_split_shapes(self.mmax, self.comm_size_azimuth)
        self.mmax_local = self.m_shapes[self.comm_rank_azimuth]
        self.mmax_offset = sum(self.m_shapes[: self.comm_rank_azimuth])

        # build only the block this rank keeps: local orders, all degrees, local latitudes, see
        # DistributedRealSHT.__init__
        self._extended_folded = _extended_folded_analysis(self.grid, self._trunc)
        self.block_size = _resolve_block_size(block_size, self.mmax)
        mmin, mmax = self.mmax_offset, self.mmax_offset + self.mmax_local
        blocks = _sht_blocks(self._trunc, self.block_size, mmin, mmax, 0, self.lmax)
        tables = _analysis_weights(self.grid, self._trunc, self.norm, self.csphase, True, mmin, mmax, self.lat_offset, self.lat_offset + self.nlat_local, self.block_size)
        _register_blocks(self, "weights", tables, blocks, 0)

        # the Jacobi iteration refines the analysis with the distributed synthesis, see RealSHT
        _setup_jacobi(self, input_grid, DistributedInverseRealVectorSHT)

    @property
    def weights(self) -> torch.Tensor:
        """This rank's dense analysis tables ``(2, mmax_local, lmax, nlat_local)``, assembled from the blocks: a fresh tensor, for inspection."""
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
    def grid_in(self) -> GridS2:
        """Global spatial domain of the input field."""
        return self.grid

    @property
    def grid_out(self) -> SpectralGrid:
        """Global spectral coefficient support of the output."""
        return self._trunc

    def extra_repr(self):
        radial = "" if self.radial is None else f",\nradial={self.radial!r}"
        return f"grid={self.grid!r}{radial},\nlmax={self.lmax}, mmax={self.mmax}, lmmax={self.lmmax}, csphase={self.csphase}" + (f", iter={self.iter}" if self.iter else "")

    # This transform cannot be captured in a single graph: the redistribution collectives it
    # calls are themselves torch.compiler.disable()d, so at comm_size > 1 dynamo breaks at
    # every one of them and inductor only ever sees the slivers in between. Those slivers hold
    # complex intermediates, which triton cannot type (KeyError: 'complex64' in codegen), and
    # compiling them buys nothing next to the all-to-alls surrounding them. Disabling the whole
    # forward costs no fusion that the breaks had not already cost, and keeps the complex
    # spectral data out of inductor entirely. The serial transforms are compiled as usual.
    @torch.compiler.disable()
    def forward(self, x: torch.Tensor):

        check_field(x, self.shard, leading=(..., None, 2), radial=self.radial)

        # Jacobi iteration a <- a + A (f - S a), see DistributedRealSHT.forward
        coeffs = self._analyze(x)
        for _ in range(self.iter):
            coeffs = coeffs + self._analyze(x - self._synthesis(coeffs))
        return coeffs

    def _analyze(self, x: torch.Tensor) -> torch.Tensor:

        # a product field is (..., 2, nr, *spatial); the radial axis moves in front of the
        # component one, where it is just another leading axis, and back at the end
        if self.radial is not None:
            component, radial = _vector_radial_dims(self, spatial=True)
            x = x.movedim(radial, component)

        # the transposes below redistribute the leading (channel/batch) axis across the
        # process grid, so it must be at least as large as the larger comm group. Flatten
        # all leading dims into that axis -- keeping the trailing (2, *spatial) intact --
        # and zero-pad it if needed (linear transform, so padding stays zero); restore the
        # original layout before returning.
        x, lead_shape, lead_size = flatten_and_pad_leading_dims(x, self.comm_size_azimuth, num_trailing_dims=1 + len(self.shard.shape))
        num_chans = x.shape[-2 - len(self.shard.shape)]

        if self.ring_fft is None:
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
        else:
            # HEALPix: the ring FFT of this rank's rings is local, see DistributedRealSHT
            x = self.ring_fft(x)

        # transpose to put the contraction dim (nlat) on the fast axis. nlat stays split across
        # the polar group, see DistributedRealSHT.forward.
        x = _real_and_imaginary(x.transpose(-1, -2))

        # one contraction per block and derivative table; xc_* holds component c contracted
        # with the first derivative table, zc_* with the second
        tables = [w.to(x.dtype) for w in _blocks(self)]
        y0 = _analysis_contraction(x, [w[0] for w in tables], self._layout, self.lmax)
        y1 = _analysis_contraction(x, [w[1] for w in tables], self._layout, self.lmax)
        (x0_re, x0_im), (x1_re, x1_im) = _components(y0, -3)
        (z0_re, z0_im), (z1_re, z1_im) = _components(y1, -3)

        # spheroidal and toroidal components
        s_re = x0_re - z1_im
        s_im = x0_im + z1_re
        t_re = -z0_im - x1_re
        t_im = z0_re - x1_im

        # stack the components in real space, see RealVectorSHT.forward
        out_re = torch.stack((s_re, t_re), dim=-3)
        out_im = torch.stack((s_im, t_im), dim=-3)

        # complete the quadrature sum and split l in a single collective, on the real view
        # (see DistributedRealSHT.forward for why the collective cannot take complex input).
        out = torch.stack((out_re, out_im), dim=-1)
        if self.comm_size_polar > 1:
            out = reduce_from_scatter_to_polar_region(out, -3, self.l_shapes)
        x = torch.view_as_complex(out.contiguous())

        # drop padding and restore the original leading dims
        x = unpad_and_unflatten_leading_dims(x, lead_shape, lead_size, num_trailing_dims=3)

        if self.radial is not None:
            component, radial = _vector_radial_dims(self, spatial=False)
            x = x.movedim(component, radial).contiguous()

        return x


class DistributedInverseRealVectorSHT(SharedBufferMixin, nn.Module):
    """
    Distributed version of the inverse (real-valued) vector SHT.
    Precomputes the associated Legendre polynomials on the nodes of the given grid.

    The distribution scheme is the same as for
    :class:`DistributedInverseRealSHT` (see its docstring for a step-by-step
    description of the all-to-all transposes over the ``N = B * C`` axis).
    The additional size-2 vector component dimension is preserved throughout, and
    HEALPix is handled as there, with the output ``(B, C, 2, npix_local)``.

    .. seealso::
        :class:`torch_harmonics.InverseRealVectorSHT`
            Serial counterpart with full mathematical description and parameter
            documentation.

    Parameters
    ----------
    grid : RegularGridS2 or HealpixGrid
        Descriptor of the spatial grid the transform operates on. It carries the
        resolution as well as the quadrature rule, so no separate ``nlat``/``nlon``
        is needed. Build one with :func:`torch_harmonics.grid.as_grid`.
        The grid is the *global* one; the local shard is derived from it.
    lmax : int, optional
        Non-inclusive maximum spherical harmonic degree.
    mmax : int, optional
        Non-inclusive maximum spherical harmonic order.
    lmmax : int, optional
        Non-inclusive upper bound on degree minus order: retain only ``l - m < lmmax``.
        ``None`` leaves this bandwidth unrestricted.
    norm : str
        Normalization type (``"ortho"``, ``"schmidt"``, ``"unnorm"``), by default ``"ortho"``
    csphase : bool
        Whether to apply the Condon-Shortley phase factor, by default True
    block_size : int or None
        Number of orders per block of the stored Legendre tables, or ``None`` for one dense
        table, see :class:`~torch_harmonics.RealSHT`. The blocks are cut at the global
        multiples of ``block_size``, so a rank holds the serial blocks of its orders.

    Returns
    -------
    torch.Tensor
        Local spatial vector field, shape ``(..., 2, nlat_local, nlon_local)``, or
        ``(..., 2, npix_local)`` on HEALPix.

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

        # on a SphericalProductGrid every shell is transformed alike; the radial axis is one
        # more leading axis, flattened into the one the transposes split
        grid, self.radial = split_radial(grid)
        _setup_grid(self, grid)
        self.norm = norm
        self.csphase = csphase

        self._trunc = truncate_sht(self.grid, lmax, mmax, lmmax)

        # compute splits: the grid decomposes itself in space; the spectral split stays manual,
        # since lmax/mmax are not grid dimensions
        _setup_shard(self)
        self.l_shapes = compute_split_shapes(self.lmax, self.comm_size_polar)
        self.lmax_local = self.l_shapes[self.comm_rank_polar]
        self.m_shapes = compute_split_shapes(self.mmax, self.comm_size_azimuth)
        self.mmax_local = self.m_shapes[self.comm_rank_azimuth]
        self.lmax_offset = sum(self.l_shapes[: self.comm_rank_polar])
        self.mmax_offset = sum(self.m_shapes[: self.comm_rank_azimuth])

        # build only the block this rank keeps: local orders, local degrees, all latitudes,
        # see DistributedInverseRealSHT.__init__
        # stored as (2, mmax_local, nlat, lmax_local) so the contraction dim l is stride-1,
        # and shared with every transform of the same kind on this rank's block
        self.block_size = _resolve_block_size(block_size, self.mmax)
        mmin, mmax = self.mmax_offset, self.mmax_offset + self.mmax_local
        lmin, lmax = self.lmax_offset, self.lmax_offset + self.lmax_local
        blocks = _sht_blocks(self._trunc, self.block_size, mmin, mmax, lmin, lmax)
        tables = _synthesis_table(self.grid, self._trunc, self.norm, self.csphase, True, mmin, mmax, lmin, lmax, self.block_size)
        _register_blocks(self, "dpct", tables, blocks, lmin)

    @property
    def dpct(self) -> torch.Tensor:
        """This rank's dense synthesis tables ``(2, mmax_local, nlat, lmax_local)``, assembled from the blocks: a fresh tensor, for inspection."""
        return _dense_table(self, -1, self.lmax_local)

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
        """Global spectral coefficient support of the input field."""
        return self._trunc

    @property
    def grid_out(self) -> GridS2:
        """Global spatial domain of the output field."""
        return self.grid

    def extra_repr(self):
        radial = "" if self.radial is None else f",\nradial={self.radial!r}"
        return f"grid={self.grid!r}{radial},\nlmax={self.lmax}, mmax={self.mmax}, lmmax={self.lmmax}, csphase={self.csphase}"

    # This transform cannot be captured in a single graph: the redistribution collectives it
    # calls are themselves torch.compiler.disable()d, so at comm_size > 1 dynamo breaks at
    # every one of them and inductor only ever sees the slivers in between. Those slivers hold
    # complex intermediates, which triton cannot type (KeyError: 'complex64' in codegen), and
    # compiling them buys nothing next to the all-to-alls surrounding them. Disabling the whole
    # forward costs no fusion that the breaks had not already cost, and keeps the complex
    # spectral data out of inductor entirely. The serial transforms are compiled as usual.
    @torch.compiler.disable()
    def forward(self, x: torch.Tensor):

        check_coeffs(x, self.lmax_local, self.mmax_local, leading=(..., None, 2), radial=self.radial)

        # a product field is (..., 2, nr, *spatial); the radial axis moves in front of the
        # component one, where it is just another leading axis, and back at the end
        if self.radial is not None:
            component, radial = _vector_radial_dims(self, spatial=False)
            x = x.movedim(radial, component)

        # the transposes below redistribute the leading (channel/batch) axis across the
        # process grid, so it must be at least as large as the larger comm group. Flatten
        # all leading dims into that axis -- keeping the trailing (2, lmax, mmax) intact --
        # and zero-pad it if needed (linear transform, so padding stays zero); restore the
        # original layout before returning.
        x, lead_shape, lead_size = flatten_and_pad_leading_dims(x, self.comm_size_azimuth, num_trailing_dims=3)

        # transpose to put the contraction dim (lmax) on the fast axis. l stays split across
        # the polar group, see DistributedInverseRealSHT.forward.
        x = _real_and_imaginary(x.transpose(-1, -2))

        # dpct blocks: (2, orders, nlat, local degrees) -- contract over l (stride-1 in both
        # operands), one contraction per block and derivative table; xc_* holds component c
        # contracted with the first derivative table, zc_* with the second
        tables = [d.to(x.dtype) for d in _blocks(self)]
        y0 = _synthesis_contraction(x, [d[0] for d in tables], self._layout)
        y1 = _synthesis_contraction(x, [d[1] for d in tables], self._layout)
        (x0_re, x0_im), (x1_re, x1_im) = _components(y0, -3)
        (z0_re, z0_im), (z1_re, z1_im) = _components(y1, -3)

        # spheroidal and toroidal components
        srl = x0_re - z1_im
        sim = x0_im + z1_re
        trl = -z0_im - x1_re
        tim = z0_re - x1_im

        # reassemble in real space, see RealVectorSHT.forward
        out_re = torch.stack((srl, trl), dim=-3)
        out_im = torch.stack((sim, tim), dim=-3)

        # complete the synthesis sum and split nlat in a single collective, on the real view
        # (see DistributedRealSHT.forward for why the collective cannot take complex input).
        out = torch.stack((out_re, out_im), dim=-1)
        if self.comm_size_polar > 1:
            out = reduce_from_scatter_to_polar_region(out, -3, self.lat_shapes)
        x = torch.view_as_complex(out.contiguous())

        if self.ring_fft is None:
            num_chans = x.shape[-4]

            # transpose: after this, channels are split and m is local
            if self.comm_size_azimuth > 1:
                x = distributed_transpose_azimuth(x, (-4, -1), self.m_shapes)

            # apply the inverse (real) FFT
            x = irfft(x, n=self.nlon, dim=-1, norm="forward")

            # transpose: after this, m is split and channels are local
            if self.comm_size_azimuth > 1:
                chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
                x = distributed_transpose_azimuth(x, (-1, -4), chan_shapes)
        else:
            # HEALPix: every order of this rank's rings is local, see DistributedInverseRealSHT
            x = self.ring_fft.inverse(x)

        # drop padding and restore the original leading dims
        x = unpad_and_unflatten_leading_dims(x, lead_shape, lead_size, num_trailing_dims=1 + len(self.shard.shape))

        if self.radial is not None:
            component, radial = _vector_radial_dims(self, spatial=True)
            x = x.movedim(component, radial).contiguous()

        return x
