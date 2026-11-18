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

from itertools import accumulate
from typing import Optional, Tuple, Union

import torch

from torch_harmonics._field_contract import check_field
from torch_harmonics.disco._disco_utils import _weight_expansion
from torch_harmonics.disco.convolution import (
    DiscreteContinuousConv,
    _from_flat,
    _setup_grids,
    _to_flat,
)
from torch_harmonics.disco.optimized.disco_optimized import _use_spatial_first_dgrad
from torch_harmonics.grid import GridS2, _rejects_legacy_signature
from torch_harmonics.radial import fold_radius, unfold_radius
from torch_harmonics.truncation import truncate_support

# a2a forward orchestration: standard (fused=False) and reordered (fused=True).
from .kernels import (
    _distributed_disco_fwd_a2a,
    _distributed_disco_fwd_a2a_reordered,
)

# distributed stuff — relative imports to avoid the circular dependency
# through torch_harmonics.distributed.__init__.
from .primitives import (
    _plan_ring_band_halo,
    compute_split_shapes,
    distributed_transpose_azimuth,
    polar_halo_exchange,
    reduce_from_scatter_to_polar_region,
)
from .utils import (
    _point_shapes,
    _ring_shapes,
    _shard_grid_pair,
)

#: Strategies for combining the polar ranks' contributions, see DistributedDiscreteContinuousConvS2.
_POLAR_MODES = frozenset({"halo-exchange", "reduce-scatter"})


def _check_channel_split(channels: int, name: str, comm_size_azimuth: int) -> None:
    """
    Reject a channel count the azimuth all-to-all cannot split.

    The a2a hands each azimuth rank a share of the channels in exchange for the full
    longitude range, and a share may be uneven but not empty. The distributed SHT pads
    its leading axis to get around this; the convolutions do not (yet), so a layer with
    fewer channels than azimuth ranks fails here rather than inside the collective.
    """
    if channels < comm_size_azimuth:
        raise ValueError(
            f"{name}={channels} cannot be split across {comm_size_azimuth} azimuth ranks: the distributed convolution "
            f"exchanges longitudes for channels, so it needs at least one channel per azimuth rank."
        )


def _setup_distributed_grids(layer: DiscreteContinuousConv, grid_in: GridS2, grid_out: GridS2, polar_mode: str, split_channels: Tuple[str, int]) -> None:
    """
    The polar mode and grids of a distributed DISCO layer, and this rank's shards of them.

    The global grids are set up as in the serial layer: the radius of a pair of product grids
    folded into the batch, and the pair ragged if either side is. Each grid then decomposes
    itself over the process grid, and the fields are checked and shaped against the shards. A
    HEALPix shard refuses an azimuth split, which is how a ragged pair refuses one. The block
    of psi stays the serial layer's whole one until the layer narrows it to its rank's.

    On a regular pair, the azimuth a2a trades the longitude split for a split of the
    channels named by ``split_channels``, ``(name, count)``: the input channels of the
    forward, the output channels of the transpose, whose a2a runs after the weight
    contraction. Every azimuth rank needs one of them. The pair also keeps the shards'
    latitude and longitude extents per rank, which the collectives split by.
    """
    if polar_mode not in _POLAR_MODES:
        raise ValueError(f"Unknown polar_mode '{polar_mode}', expected one of {sorted(_POLAR_MODES)}")
    layer.polar_mode = polar_mode
    layer.use_halo = polar_mode == "halo-exchange"

    _setup_grids(layer, grid_in, grid_out)
    _shard_grid_pair(layer)
    layer._field_in, layer._field_out = layer.shard_in, layer.shard_out

    if not layer.ragged:
        name, channels = split_channels
        _check_channel_split(channels, name, layer.comm_size_azimuth)
        layer.lat_in_shapes = list(layer.shard_in.lat_shapes)
        layer.lon_in_shapes = list(layer.shard_in.lon_shapes)
        layer.lat_out_shapes = list(layer.shard_out.lat_shapes)
        layer.lon_out_shapes = list(layer.shard_out.lon_shapes)


def _ring_reach(grid: GridS2, points: torch.Tensor) -> Optional[Tuple[int, int]]:
    """The half-open range of ``grid``'s rings that the flat ``points`` lie on, or ``None`` if there are none."""
    reach = None
    if points.numel() > 0:
        bounds = grid.lon_offsets.to(torch.int64)
        lo = int(torch.searchsorted(bounds, points.min(), right=True)) - 1
        hi = int(torch.searchsorted(bounds, points.max(), right=True))
        reach = (lo, hi)
    return reach


# ---- the rank's block of psi from its halo ----
#
# Every polar split of psi keeps one contiguous range of its rows and one of its columns,
# which the layer sets as _psi_rows and _psi_cols and DiscreteContinuousConv._psi_coo cuts
# psi to. On a ragged pair the serial ragged backends then do the work, as for neighborhood
# attention, and the layer also sets _psi_col_ring_range, the rings of _psi_col_grid the
# columns cover, whose ring tables the arcs walk, and npoints_out_local.

#: appended to the refusal of a halo too wide for the polar decomposition
_REDUCE_SCATTER_ADVICE = " Alternatively pass polar_mode='reduce-scatter', which places no bound on the angular reach but computes every output {} on every polar rank."


def _plan_psi_halo(layer, owned: torch.Tensor, reached: torch.Tensor, owned_shard, reached_shard, advice: str = ""):
    """
    Plan the halo on ``reached_shard`` from the psi entries along ``owned``.

    Every rank keeps the psi entries whose ``owned`` point -- a row of the forward, a column
    of the transpose -- lies in its block of ``owned_shard``, and reads the rings of
    ``reached_shard`` that their ``reached`` points lie on. Both are flat point indices of
    the global grids. Returns this rank's halo plan, its band of ``reached_shard`` as rings
    and as flat points, and its block of ``owned_shard`` as flat points. The halo is planned in rings
    on a regular layer, whose fields keep their latitude axis, and in points on a ragged
    one, whose fields are flat. ``advice`` is appended to the error a band too wide for the
    decomposition raises.
    """
    owned_starts = list(accumulate([0] + _point_shapes(owned_shard)))
    reach = []
    for r in range(layer.comm_size_polar):
        sel = (owned >= owned_starts[r]) & (owned < owned_starts[r + 1])
        reach.append(_ring_reach(reached_shard.grid, reached[sel]))
    try:
        halo, ring_range, point_range = _plan_ring_band_halo(reached_shard, reach, layer.comm_rank_polar, per_ring=not layer.ragged)
    except ValueError as err:
        raise ValueError(f"{type(layer).__name__}: theta_cutoff={layer.theta_cutoff} is too wide for this polar decomposition. {err}{advice}") from err
    return halo, ring_range, point_range, (owned_starts[layer.comm_rank_polar], owned_starts[layer.comm_rank_polar + 1])


class DistributedDiscreteContinuousConvS2(DiscreteContinuousConv):
    """
    Distributed version of Discrete-continuous convolutions (DISCO) on the 2-Sphere as described in :cite:`Ocampo2023`.
    We assume the data can be split in polar and azimuthal directions.

    .. seealso::
        :class:`torch_harmonics.DiscreteContinuousConvS2`
            Serial counterpart with full mathematical description and parameter
            documentation.

    Longitudes are handled by all-to-all transposes; latitudes are handled as
    selected by ``polar_mode``. Of the options, ``polar_mode`` usually matters most
    for memory, because it sets how the per-basis-function intermediate scales
    with the number of polar ranks.

    Either grid may be HEALPix, which decomposes along latitude only: each polar rank
    holds a band of whole rings, fields on a HEALPix side are ``(batch, channels,
    npix_local)``, and more than one azimuth rank raises ``NotImplementedError``. The
    serial ragged backends then run on the rank's block of psi, see ``polar_mode``.

    Parameters
    ----------
    grid_in : GridS2
        Descriptor of the input grid; it carries the resolution as well as the
        quadrature rule. A latitude-longitude grid or HEALPix.
    grid_out : GridS2
        Descriptor of the output grid.
        Both are the **global** grids; each rank derives its own slice.
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_shape : Union[int, Tuple[int], Tuple[int, int]]
        Shape of the kernel
    basis_type : Optional[str]
        Type of basis to use
    basis_norm_mode : Optional[str]
        Normalization mode for the filter basis
    groups : Optional[int]
        Number of groups
    bias : Optional[bool]
        Whether to use bias
    theta_cutoff : Optional[float]
        Theta cutoff for the filter basis
    optimized_kernel : Optional[bool]
        Whether to use the optimized kernels (if available)
    fused : bool
        As in the serial layer: if ``True``, the per-basis-function intermediate is
        recomputed in the backward pass instead of stored, reducing activation memory
        and communication volume by a factor of the kernel size at the cost of extra
        compute. The recompute needs the optimized kernels; the torch reference keeps
        the reordering but saves the intermediate. By default ``False``.
    polar_mode : Optional[str]
        How the polar ranks obtain their output latitudes.

        ``"halo-exchange"`` (default): each rank borrows the neighbouring input rows
        it needs and computes only its own output latitudes, so the per-basis-function
        intermediate is ``(B, C, K, nlat_out / polar_size, nlon_out)`` and shrinks as
        polar ranks are added. The filter support must not extend past the neighbouring
        rank; otherwise construction raises.

        ``"reduce-scatter"``: every rank computes partial sums over all output
        latitudes, which a reduce-scatter completes. Works for any cutoff, but memory
        does not shrink with the number of polar ranks.

        On HEALPix the same two choices hold per point: the halo brings in the whole
        input rings this rank's output points reach, unevenly where the bands differ,
        and the reduce-scatter splits the output into the polar ranks' bands of pixels.

    Returns
    -------
    torch.Tensor
        Local output of shape ``(batch, out_channels, nlat_out_local, nlon_out_local)``,
        or ``(batch, out_channels, npix_out_local)`` on a HEALPix output grid.

    References
    ----------
    :cite:`Ocampo2023`
    """

    @_rejects_legacy_signature(
        'in_channels, out_channels, in_shape, out_shape, kernel_shape, basis_type="piecewise linear", basis_norm_mode="nodal", '
        'groups=1, grid_in="equiangular", grid_out="equiangular", bias=True, theta_cutoff=None, optimized_kernel=True, '
        'fused=False, polar_mode="halo-exchange"',
        grid_in="in_shape",
        grid_out="out_shape",
    )
    def __init__(
        self,
        grid_in: GridS2,
        grid_out: GridS2,
        in_channels: int,
        out_channels: int,
        kernel_shape: Union[int, Tuple[int], Tuple[int, int]],
        basis_type: Optional[str] = "piecewise linear",
        basis_norm_mode: Optional[str] = "nodal",
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
        theta_cutoff: Optional[float] = None,
        optimized_kernel: Optional[bool] = True,
        fused: bool = False,
        polar_mode: Optional[str] = "halo-exchange",
    ):
        super().__init__(in_channels, out_channels, kernel_shape, basis_type, groups, bias, optimized_kernel)

        # fused=True uses the reordered a2a, see _distributed_disco_fwd_a2a_reordered
        self.fused = bool(fused)
        self.basis_norm_mode = basis_norm_mode

        _setup_distributed_grids(self, grid_in, grid_out, polar_mode, ("in_channels", in_channels))

        # compute theta cutoff based on the bandlimit of the input field
        self.theta_cutoff = truncate_support(self.grid_out, theta_cutoff)

        self._setup_polar()
        self._select_backend()

    def _setup_polar(self) -> None:
        """
        Cut psi to this rank's block, on the side the polar mode picks, and plan the halo of input rings.

        psi's rows are output rings -- output points on a ragged pair -- and its columns flat
        indices of the input grid. The contraction along the longitude is a convolution,
        which the a2a makes local, so psi is split along latitude only.

        The operator is local: an output latitude only reads input latitudes within
        theta_cutoff of it. Under ``halo-exchange`` the rank keeps the rows of its own output
        rings and reads the input rings their entries reach, which a halo brings in from the
        neighbouring ranks, so its rows are complete -- no polar reduction, and the
        K-expanded intermediate shrinks with the polar group. Under ``reduce-scatter`` it
        keeps the columns of its own input rings over every output ring, and a
        reduce-scatter completes the partial sums; no halo moves. The two kinds of grid
        differ only in what a row is, and in that the regular layer exchanges whole rows of
        its ``(nlat, nlon)`` field where the ragged one exchanges points of its flat one.
        """
        rank, size = self.comm_rank_polar, self.comm_size_polar
        if self.use_halo:
            idx = self._global_psi()[0]
            # the output points the rows key: a regular row is a latitude, whose first point is lat * nlon
            owned = idx[1] if self.ragged else idx[1] * self.nlon_out
            unit = "point" if self.ragged else "latitude"
            self.halo, ring_range, point_range, _ = _plan_psi_halo(self, owned, idx[2], self.shard_out, self.shard_in, advice=_REDUCE_SCATTER_ADVICE.format(unit))
            out_starts = list(accumulate([0] + (_point_shapes(self.shard_out) if self.ragged else _ring_shapes(self.shard_out))))
            self._psi_rows = (out_starts[rank], out_starts[rank + 1])
        else:
            self.halo, ring_range, point_range = _plan_ring_band_halo(self.shard_in, [None] * size, rank, per_ring=not self.ragged)
            self._psi_rows = (0, self.npoints_out if self.ragged else self.nlat_out)
        self._psi_cols = point_range
        self._psi_col_ring_range = ring_range

        # the contraction's result: the output points or latitudes psi's rows cover
        nrows = self._psi_rows[1] - self._psi_rows[0]
        if self.ragged:
            self.npoints_out_local = nrows
            # the ragged layer always contracts through the fused node, as the serial one does
            self._needs_split = _use_spatial_first_dgrad(self.out_per_group, self.groupsize, self.kernel_size)
        else:
            self.nlat_out_local = nrows
            self._psi_nlon = self.nlon_in
            self._contract_shape = (self.nlat_out_local, self.nlon_out)
            # only the reordered path contracts through the fused node
            self._needs_split = self.fused and _use_spatial_first_dgrad(self.out_per_group, self.groupsize, self.kernel_size)

    def _shard_key(self):
        # this rank's block: its place in the process grid, and the polar mode with its halo
        return (self.comm_rank_polar, self.comm_size_polar, self.comm_rank_azimuth, self.comm_size_azimuth, self.use_halo, self.halo)

    def extra_repr(self):
        radial = "" if self.radial is None else f",\nradial={self.radial!r}"
        return (
            f"grid_in={self.grid_in!r},\n"
            f"grid_out={self.grid_out!r}{radial},\n"
            f"in_channels={self.groupsize * self.groups}, "
            f"out_channels={self.weight.shape[0]}, "
            f"filter_basis={self.filter_basis}, "
            f"kernel_shape={self.kernel_shape}, "
            f"theta_cutoff={self.theta_cutoff}, "
            f"groups={self.groups}, fused={self.fused}, polar_mode={self.polar_mode}"
        )

    def _forward_ragged(self, x: torch.Tensor) -> torch.Tensor:
        """
        The serial ragged convolution on this rank's block, between the polar collectives.

        There is no azimuth split to undo, so the K-expanded intermediate never travels:
        the halo moves the input, the reduce-scatter the K-less output.
        """
        x = _to_flat(self, x)
        if self.use_halo and self.comm_size_polar > 1:
            x = polar_halo_exchange(x, self.halo, lat_dim=-1)

        out = self.backend.conv(self, x, self._weight_r(), self.groups, self.groupsize, recompute=self.fused)

        if not self.use_halo and self.comm_size_polar > 1:
            out = reduce_from_scatter_to_polar_region(out, -1, _point_shapes(self.shard_out))
        return _from_flat(self, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # this rank's block, with every channel: the a2a splits the channels itself
        check_field(x, self.shard_in, leading=(None, self.in_channels), radial=self.radial)

        if self.radial is not None:
            x = fold_radius(x)

        # reordered a2a: weights first on the local channel shard, K-expanded recomputed in
        # backward; standard a2a: contraction, collectives, then the weights
        if self.ragged:
            out = self._forward_ragged(x)
        elif self.fused:
            out = _distributed_disco_fwd_a2a_reordered(self, x)
        else:
            out = _distributed_disco_fwd_a2a(self, x)

        if self.bias is not None:
            out = out + self.bias.reshape(1, self.bias.shape[0], *([1] * (out.dim() - 2)))

        if self.radial is not None:
            out = unfold_radius(out, self.radial.nr)

        return out


class DistributedDiscreteContinuousConvTransposeS2(DiscreteContinuousConv):
    """
    Distributed version of discrete-continuous transpose convolutions (DISCO) on the 2-Sphere as described in :cite:`Ocampo2023`.

    .. seealso::
        :class:`torch_harmonics.DiscreteContinuousConvTransposeS2`
            Serial counterpart with full mathematical description and parameter
            documentation.

    Longitudes are handled by all-to-all transposes; latitudes are handled as selected
    by ``polar_mode``. The polar exchange moves the input, before the weight
    contraction, rather than the per-basis-function intermediate after it: the
    contraction is pointwise, so the two commute, and the input is smaller by a
    factor of ``out_channels * K / in_channels``.

    Either grid may be HEALPix, which decomposes along latitude only: each polar rank
    holds a band of whole rings, fields on a HEALPix side are ``(batch, channels,
    npix_local)``, and more than one azimuth rank raises ``NotImplementedError``. The
    serial ragged backends then run on the rank's block of psi, see ``polar_mode``.

    Parameters
    ----------
    grid_in : GridS2
        Descriptor of the input grid; it carries the resolution as well as the
        quadrature rule. A latitude-longitude grid or HEALPix.
    grid_out : GridS2
        Descriptor of the output grid.
        Both are the **global** grids; each rank derives its own slice.
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_shape : Union[int, Tuple[int], Tuple[int, int]]
        Shape of the kernel
    basis_type : Optional[str]
        Type of basis to use
    basis_norm_mode : Optional[str]
        Normalization mode for the filter basis
    groups : Optional[int]
        Number of groups
    bias : Optional[bool]
        Whether to use bias
    theta_cutoff : Optional[float]
        Theta cutoff for the filter basis
    optimized_kernel : Optional[bool]
        Whether to use the optimized kernel (if available)
    polar_mode : Optional[str]
        How the polar ranks obtain their output latitudes.

        ``"halo-exchange"`` (default): each rank borrows the neighbouring input rows
        that scatter onto its own output latitudes and computes only those, so the
        per-basis-function intermediate covers the rank's input band and shrinks as
        polar ranks are added. The input rows have to come from the neighbouring ranks;
        a wider filter support raises at construction.

        ``"reduce-scatter"``: every rank scatters its own input rows onto all output
        latitudes, and a reduce-scatter of the output completes the sums. Works for any
        cutoff, but the output buffer does not shrink with the number of polar ranks.

        On HEALPix the same two choices hold per point: the halo brings in the whole
        input rings that scatter onto this rank's output points, and the reduce-scatter
        splits the output into the polar ranks' bands of pixels.

    Returns
    -------
    torch.Tensor
        Local output of shape ``(batch, out_channels, nlat_out_local, nlon_out_local)``,
        or ``(batch, out_channels, npix_out_local)`` on a HEALPix output grid.

    References
    ----------
    :cite:`Ocampo2023`
    """

    #: psi is applied in the scatter direction, as in the serial transpose convolution
    transpose = True

    @_rejects_legacy_signature(
        'in_channels, out_channels, in_shape, out_shape, kernel_shape, basis_type="piecewise linear", basis_norm_mode="nodal", '
        'groups=1, grid_in="equiangular", grid_out="equiangular", bias=True, theta_cutoff=None, optimized_kernel=True',
        grid_in="in_shape",
        grid_out="out_shape",
    )
    def __init__(
        self,
        grid_in: GridS2,
        grid_out: GridS2,
        in_channels: int,
        out_channels: int,
        kernel_shape: Union[int, Tuple[int], Tuple[int, int]],
        basis_type: Optional[str] = "piecewise linear",
        basis_norm_mode: Optional[str] = "nodal",
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
        theta_cutoff: Optional[float] = None,
        optimized_kernel: Optional[bool] = True,
        polar_mode: Optional[str] = "halo-exchange",
    ):
        super().__init__(in_channels, out_channels, kernel_shape, basis_type, groups, bias, optimized_kernel)

        self.basis_norm_mode = basis_norm_mode
        _setup_distributed_grids(self, grid_in, grid_out, polar_mode, ("out_channels", out_channels))

        # bandlimit
        self.theta_cutoff = truncate_support(self.grid_in, theta_cutoff)

        self._setup_polar()
        self._select_backend()

    def _setup_polar(self) -> None:
        """
        Cut psi to this rank's block, and plan the halo of input rings that scatter onto it.

        psi's rows are input rings -- input points on a ragged pair -- and its columns flat
        indices of the output grid they scatter onto. Under ``halo-exchange`` the rank keeps
        the columns of its own output rings, so its result is complete and no reduction
        follows, and reads the input rings those columns are reached from, which a halo
        brings in. Under ``reduce-scatter`` it keeps the rows of its own input rings over
        every output ring, and a reduce-scatter completes the partial sums; no halo moves.
        The two kinds of grid differ only in what a row is, and in that the regular layer
        exchanges whole rows of its ``(nlat, nlon)`` field where the ragged one exchanges
        points of its flat one.
        """
        rank, size = self.comm_rank_polar, self.comm_size_polar
        if self.use_halo:
            idx = self._global_psi()[0]
            # the input points the entries read: a regular row is a latitude, whose first point is lat * nlon
            reached = idx[1] if self.ragged else idx[1] * self.nlon_in
            unit = "point" if self.ragged else "latitude"
            self.halo, ring_range, point_range, self._psi_cols = _plan_psi_halo(self, idx[2], reached, self.shard_out, self.shard_in, advice=_REDUCE_SCATTER_ADVICE.format(unit))
            out_rings = list(accumulate([0] + _ring_shapes(self.shard_out)))
            self._psi_col_ring_range = (out_rings[rank], out_rings[rank + 1])
        else:
            self.halo, ring_range, point_range = _plan_ring_band_halo(self.shard_in, [None] * size, rank, per_ring=not self.ragged)
            self._psi_cols = (0, self.grid_out.npoints)
            self._psi_col_ring_range = (0, self.grid_out.nrings)
        self._psi_rows = point_range if self.ragged else ring_range

        # the scatter's result: the output points or latitudes psi's columns cover
        ncols = self._psi_cols[1] - self._psi_cols[0]
        if self.ragged:
            self.npoints_out_local = ncols
        else:
            self.nlat_out_local = ncols // self.nlon_out
            self._psi_nlon = self.nlon_out
            self._contract_shape = (self.nlat_out_local, self.nlon_out)

    def _shard_key(self):
        # this rank's block: its place in the process grid, and the polar mode with its halo
        return (self.comm_rank_polar, self.comm_size_polar, self.comm_rank_azimuth, self.comm_size_azimuth, self.use_halo, self.halo)

    def extra_repr(self):
        radial = "" if self.radial is None else f",\nradial={self.radial!r}"
        return (
            f"grid_in={self.grid_in!r},\ngrid_out={self.grid_out!r}{radial},\nin_channels={self.groupsize * self.groups}, out_channels={self.weight.shape[0]}, "
            f"filter_basis={self.filter_basis}, kernel_shape={self.kernel_shape}, theta_cutoff={self.theta_cutoff}, groups={self.groups}, polar_mode={self.polar_mode}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_field(x, self.shard_in, leading=(None, self.in_channels), radial=self.radial)

        if self.radial is not None:
            x = fold_radius(x)

        x = _to_flat(self, x)

        # The collectives are guarded at size 1 although they are identities there: skipping
        # keeps their torch.compiler.disable()d wrappers out of the graph, so the layer stays
        # fullgraph-compilable without a split.
        if self.use_halo and self.comm_size_polar > 1:
            x = polar_halo_exchange(x, self.halo, lat_dim=-1 if self.ragged else -2)

        # do weight multiplication, (B, in_channels, *spatial) -> (B, out_channels, K, *spatial)
        x = _weight_expansion(x, self._weight_r().transpose(1, 2))
        num_chans = x.shape[1]

        # transpose such that lon is local, channels are split
        if self.comm_size_azimuth > 1:
            x = distributed_transpose_azimuth(x, (1, -1), self.lon_in_shapes)

        out = self.backend.transpose(self, x)

        # the K-less partial sums, before the a2a so that it moves only this rank's rows
        if not self.use_halo and self.comm_size_polar > 1:
            out = reduce_from_scatter_to_polar_region(out, -1 if self.ragged else -2, _point_shapes(self.shard_out) if self.ragged else self.lat_out_shapes)

        # now we can transpose back the result, so that lon is split and channels are local
        if self.comm_size_azimuth > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            out = distributed_transpose_azimuth(out, (-1, 1), chan_shapes)

        out = _from_flat(self, out)

        if self.bias is not None:
            out = out + self.bias.reshape(1, self.bias.shape[0], *([1] * (out.dim() - 2)))

        if self.radial is not None:
            out = unfold_radius(out, self.radial.nr)

        return out
