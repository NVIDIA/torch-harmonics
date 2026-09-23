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

r"""
Geodesic neighbourhood structure of an isolatitude grid, as contiguous arcs.

The sparsity pattern the localized operators need: for each output point, which
input points lie within an angular radius of it. On an isolatitude grid that
pattern is not an arbitrary index set. A geodesic ball meets a latitude circle in
a single arc, so the neighbourhood of an output point is a handful of *runs* of
consecutive points -- one per input ring the ball reaches -- and can be stored as
``(ring, start, length)`` triples rather than as an explicit column list. That is
what the CUDA attention kernels consume, because deriving a neighbour's column by
incrementing a counter avoids the per-neighbour 64-bit integer division that
recovering it from a flat column index would cost on a GPU.

The existing DISCO precompute in :mod:`torch_harmonics.disco.convolution` finds
this pattern by evaluating a filter basis at every rotated input point and keeping
what falls inside the cutoff, then recovering the runs afterwards in
:func:`~torch_harmonics.attention._attention_utils._build_psi_segments`. That works
on a product grid, and it leans on the product structure twice: it computes one
output *longitude* per output latitude and lets the kernels reach the rest by
rotating the pattern, and it flattens columns with a uniform ``nlon`` stride.

Neither holds on a ragged grid. Rotating a HEALPix output ring of :math:`n_k`
pixels by one of its own pixels maps that ring onto itself but not the *other*
rings, whose pixel counts differ, so there is no shift that carries one output
pixel's neighbourhood onto the next one's -- except within the equatorial belt,
where every ring has :math:`4 N` pixels. The pattern therefore has to be keyed by
output point rather than by output ring.

This module computes it directly from the ring geometry instead, keyed per output
point, which removes both assumptions at once and costs less than the basis
evaluation it replaces: the arc on each ring is the solution of one inequality, so
the work is proportional to the number of arcs rather than to the number of
candidate input points.
"""

import math
from typing import NamedTuple, Optional, Tuple

import torch

from torch_harmonics.cache import lru_cache
from torch_harmonics.grid import GridS2, require_grid

__all__ = ["NeighborhoodArcsS2", "precompute_neighborhood_arcs_s2", "precompute_neighborhood_csr_s2"]

# ceiling on the (output point, ring) pairs held in flight at once, which bounds the
# precompute's peak memory independently of the grid size
_PAIR_BUDGET = 1 << 22

# below this, a latitude circle is a point rather than a circle and the arc
# construction has nothing to solve; the pole rings of an equiangular grid hit it
# exactly, so it must be a tolerance and not an equality test
_POLE_EPS = 1e-14

# ceiling on the neighbour entries expanded at once by to_csr. The expansion holds a
# handful of int64 temporaries per entry, so this trades a bounded peak against a
# Python iteration count of nnz / budget -- one, for any cutoff of practical size.
_CSR_ENTRY_BUDGET = 1 << 20


def _row_chunks(row_off: torch.Tensor, budget: int):
    """
    Split rows into runs whose combined entry count stays within ``budget``.

    Chunking on rows rather than on entries is what lets the caller sort each chunk
    independently: a row never straddles two chunks, so concatenating the chunks in
    order reproduces the global row order. A row wider than the budget is emitted
    alone rather than split, so the loop always advances.
    """
    npoints = row_off.numel() - 1
    p0 = 0
    while p0 < npoints:
        limit = row_off[p0] + budget
        p1 = int(torch.searchsorted(row_off, limit, right=True)) - 1
        p1 = min(max(p1, p0 + 1), npoints)
        yield p0, p1
        p0 = p1


class NeighborhoodArcsS2(NamedTuple):
    r"""
    A geodesic neighbourhood pattern in contiguous-arc form.

    Attributes
    ----------
    segments : torch.Tensor
        ``int32``, shape ``(nsegs, 3)``, columns ``(ring, start, length)``. One arc of
        ``length`` consecutive points of input ring ``ring``, beginning at index
        ``start`` within that ring and wrapping at the ring's end. ``start`` is always
        a valid index into the ring and ``length`` never exceeds the ring size, so a
        consumer can advance with a single compare-and-subtract and never needs a
        modulo.
    offsets : torch.Tensor
        ``int32``, shape ``(npoints_out + 1,)``. Segments of output point ``p`` are
        ``segments[offsets[p]:offsets[p + 1]]``. Output points are in the flat order of
        the output grid.
    ring_base : torch.Tensor
        ``int64``, shape ``(nlat_in,)``. Flat index of the first point of each input
        ring, i.e. the input grid's ``lon_offsets[:-1]``. Carried alongside so a
        consumer can turn ``(ring, offset within ring)`` into a flat column without
        holding the grid.
    ring_size : torch.Tensor
        ``int64``, shape ``(nlat_in,)``. Points on each input ring, i.e. the input
        grid's ``nlon_per_lat``. Needed to wrap an arc.
    theta_cutoff : float
        The radius the pattern was built for, in radians.
    """

    segments: torch.Tensor
    offsets: torch.Tensor
    ring_base: torch.Tensor
    ring_size: torch.Tensor
    theta_cutoff: float

    @property
    def nnz(self) -> int:
        """Number of (output point, input point) pairs the pattern contains."""
        return int(self.segments[:, 2].to(torch.int64).sum().item()) if self.segments.numel() else 0

    def columns(self, ipoint: int) -> torch.Tensor:
        r"""
        Expand one output point's arcs into the flat input indices they stand for.

        The inverse of the arc encoding, for the reference paths and for tests: the
        arcs are the representation the kernels want, and this is what they mean.

        Parameters
        ----------
        ipoint : int
            Index of the output point, in the flat order of the output grid.

        Returns
        -------
        torch.Tensor
            ``int64`` flat input indices, ascending.
        """
        cols = []
        for iseg in range(int(self.offsets[ipoint]), int(self.offsets[ipoint + 1])):
            ring, start, length = (int(x) for x in self.segments[iseg])
            n = int(self.ring_size[ring])
            base = int(self.ring_base[ring])
            offsets_in_ring = (start + torch.arange(length, dtype=torch.int64)) % n
            cols.append(base + offsets_in_ring)
        if not cols:
            return torch.empty(0, dtype=torch.int64)
        return torch.cat(cols).sort().values

    def to_csr(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Expand the whole pattern into a CSR-style column list keyed by output point.

        The arcs are the form the kernels want, because an arc can be walked with a
        compare-and-subtract while a column list costs an integer division per
        neighbour. This is the other form: what the reference paths and the sparse
        linear algebra in the tests consume. Materializing it is what makes those
        consumers independent of the arc derivation, so a mistake in the encoding
        cannot hide by being made identically on both sides.

        Returns
        -------
        col_idx : torch.Tensor
            ``int64``, shape ``(nnz,)``. Flat input indices, ascending within each
            output point.
        row_off : torch.Tensor
            ``int64``, shape ``(npoints_out + 1,)``. Neighbours of output point ``p``
            are ``col_idx[row_off[p]:row_off[p + 1]]``.

        Notes
        -----
        The expansion is vectorized over arcs rather than looping over output points,
        because the loop is over ``npoints_out`` and so grows with the grid: at HEALPix
        ``nside`` 32 it is twelve thousand iterations of small allocations, and every
        attention layer in a model pays it again. :func:`precompute_neighborhood_csr_s2`
        is the cached entry point that keeps layers sharing one expansion.
        """
        npoints_out = self.offsets.numel() - 1
        seg_off = self.offsets.to(torch.int64)
        ring = self.segments[:, 0].to(torch.int64)
        start = self.segments[:, 1].to(torch.int64)
        lengths = self.segments[:, 2].to(torch.int64)

        # an output point's neighbour count is the total length of its arcs, so the row
        # offsets are a segment sum over arcs grouped by the point that owns them
        point_of_seg = torch.repeat_interleave(torch.arange(npoints_out, dtype=torch.int64), seg_off[1:] - seg_off[:-1])
        counts = torch.zeros(npoints_out, dtype=torch.int64)
        counts.index_add_(0, point_of_seg, lengths)
        row_off = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(dim=0)])

        # a composite sort key needs a stride wider than any column it has to separate
        npoints_in = int(self.ring_base[-1] + self.ring_size[-1])

        chunks = []
        for p0, p1 in _row_chunks(row_off, _CSR_ENTRY_BUDGET):
            s0, s1 = int(seg_off[p0]), int(seg_off[p1])
            length_c = lengths[s0:s1]
            nnz_c = int(length_c.sum())

            # walk the arcs of this chunk as one flat run of entries: which arc each
            # entry belongs to, and how far into that arc it sits
            seg_of_entry = torch.repeat_interleave(torch.arange(s1 - s0, dtype=torch.int64), length_c)
            arc_begin = length_c.cumsum(dim=0) - length_c
            within = torch.arange(nnz_c, dtype=torch.int64) - arc_begin[seg_of_entry]

            iring = ring[s0:s1][seg_of_entry]
            size = self.ring_size[iring]
            col = self.ring_base[iring] + (start[s0:s1][seg_of_entry] + within) % size

            # arcs arrive in ring order and a wrapping arc is not itself ascending, so
            # the columns of a point need sorting; ordering by (point, column) sorts
            # every row of the chunk at once without disturbing the rows themselves
            local_point = point_of_seg[s0:s1][seg_of_entry] - p0
            chunks.append(col[torch.argsort(local_point * npoints_in + col)])

        col_idx = torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.int64)

        return col_idx, row_off


@lru_cache(maxsize=8, typed=True, copy=True)
def precompute_neighborhood_arcs_s2(grid_in: GridS2, grid_out: GridS2, theta_cutoff: float, theta_eps: Optional[float] = 1e-3) -> NeighborhoodArcsS2:
    r"""
    Geodesic neighbourhood of every output point, as contiguous arcs of input points.

    Works on any isolatitude grid, ragged or not, because it uses only the ring
    geometry the descriptor exposes: ring colatitudes, ring sizes and the fractional
    longitude offset of each ring.

    Method
    ------
    An output point sits at :math:`(\theta_o, \phi_o)`. The points of input ring
    :math:`m`, at colatitude :math:`\theta_m`, that lie within a geodesic radius
    :math:`r` of it are exactly those whose longitude satisfies

    .. math::
        \cos r \le \cos\theta_o \cos\theta_m + \sin\theta_o \sin\theta_m \cos(\phi - \phi_o),

    which is one interval in :math:`\phi` centred on :math:`\phi_o`:

    .. math::
        |\phi - \phi_o| \le \Delta_m, \qquad
        \cos \Delta_m = \frac{\cos r - \cos\theta_o \cos\theta_m}{\sin\theta_o \sin\theta_m}.

    A right-hand side at or below :math:`-1` means the whole ring is inside the ball
    and the arc is the entire ring; at or above :math:`+1` the ring is missed
    entirely. Otherwise the ring's points, at
    :math:`\phi_j = \frac{2\pi}{n_m}(j + \delta_m)`, give the index range

    .. math::
        j \in \left[\left\lceil \frac{n_m}{2\pi}(\phi_o - \Delta_m) - \delta_m \right\rceil,
                    \left\lfloor \frac{n_m}{2\pi}(\phi_o + \Delta_m) - \delta_m \right\rfloor\right],

    contiguous by construction and wrapping at the seam. Only rings with
    :math:`|\theta_m - \theta_o| \le r` can contribute, which bounds the search to a
    band found by binary search on the sorted ring colatitudes.

    Parameters
    ----------
    grid_in : GridS2
        Grid the neighbours are drawn from.
    grid_out : GridS2
        Grid whose points the neighbourhoods are centred on.
    theta_cutoff : float
        Angular radius of the neighbourhood, in radians. Positive. Usually obtained
        from :func:`torch_harmonics.truncate_support`.
    theta_eps : float, optional
        Relative widening of the radius, by default ``1e-3``. Matches
        :func:`~torch_harmonics.disco.convolution._precompute_convolution_tensor_s2`,
        where it keeps a point that lands exactly on the cutoff -- which happens on
        symmetric grids far more often than a random-geometry intuition suggests --
        from falling in or out on a floating-point tie.

    Returns
    -------
    NeighborhoodArcsS2
        The pattern. Cached on the descriptors, which is why they are required to be
        hashable and to exclude tensors.

    Raises
    ------
    ValueError
        If ``theta_cutoff`` is not positive.

    Notes
    -----
    A ring that sits exactly on a pole -- which every equiangular grid has, at both
    ends -- degenerates to a single location repeated ``nlon`` times, so its distance
    to an output point does not depend on longitude and the arc is either the whole
    ring or nothing. The same is true of an output point on a pole. Both are handled
    as the whole-ring case rather than rejected, which is what makes the pattern agree
    with the DISCO precompute on an equiangular grid.
    """
    grid_in = require_grid(grid_in, "grid_in")
    grid_out = require_grid(grid_out, "grid_out")

    if not theta_cutoff > 0.0:
        raise ValueError(f"theta_cutoff must be positive, got {theta_cutoff}")

    # capped at pi, beyond which the ball is the whole sphere. Without the cap the
    # widened radius would wrap past the antipode, where cos turns back upwards, and
    # the comparison in cos(theta) would start *excluding* the farthest points -- so a
    # cutoff of pi would return everything except each point's antipode.
    radius = min((1.0 + theta_eps) * theta_cutoff, math.pi)
    cos_radius = math.cos(radius)

    colats_in = grid_in.lats.to(torch.float64)
    ring_size = grid_in.nlon_per_lat.to(torch.int64)
    ring_base = grid_in.lon_offsets[:-1].to(torch.int64)
    shifts_in = _lon_shifts(grid_in)
    sin_in, cos_in = torch.sin(colats_in), torch.cos(colats_in)

    colats_out = _flat_colats(grid_out)
    lons_out = _flat_lons(grid_out)
    npoints_out = colats_out.numel()

    # rings are sorted by colatitude, so the band that can reach a given output point
    # is a contiguous slice, found by binary search rather than by a scan over rings
    first_ring = torch.searchsorted(colats_in, colats_out - radius, right=False)
    last_ring = torch.searchsorted(colats_in, colats_out + radius, right=True)
    band = (last_ring - first_ring).clamp(min=0)
    max_band = int(band.max()) if npoints_out else 0

    if max_band == 0:
        return NeighborhoodArcsS2(
            segments=torch.empty((0, 3), dtype=torch.int32),
            offsets=torch.zeros(npoints_out + 1, dtype=torch.int32),
            ring_base=ring_base,
            ring_size=ring_size,
            theta_cutoff=theta_cutoff,
        )

    # Vectorized over (output point, ring in band) pairs, in chunks of output points.
    # The band is only a few rings wide, so padding every point out to the widest band
    # wastes little; chunking keeps that padded block bounded independently of the grid
    # size, which at nside 256 is the difference between 30 MB and 4 GB.
    chunk = max(1, _PAIR_BUDGET // max_band)
    seg_chunks = []
    counts = torch.empty(npoints_out, dtype=torch.int64)

    for begin in range(0, npoints_out, chunk):
        end = min(begin + chunk, npoints_out)

        rings = first_ring[begin:end, None] + torch.arange(max_band, dtype=torch.int64)
        in_band = rings < last_ring[begin:end, None]
        rings = rings.clamp(max=grid_in.nlat - 1)  # keep the padded lanes indexable

        theta_o = colats_out[begin:end, None]
        phi_o = lons_out[begin:end, None]
        n = ring_size[rings]
        delta = shifts_in[rings]

        sin_o, cos_o = torch.sin(theta_o), torch.cos(theta_o)
        denominator = sin_o * sin_in[rings]

        # a degenerate ring, or an output point on a pole, has no longitude dependence
        degenerate = denominator.abs() < _POLE_EPS
        cos_half_width = torch.where(degenerate, torch.full_like(denominator, -1.0), (cos_radius - cos_o * cos_in[rings]) / denominator.masked_fill(degenerate, 1.0))

        whole_ring = cos_half_width <= -1.0
        half_width = torch.arccos(cos_half_width.clamp(-1.0, 1.0))

        # the ring's points sit at phi = (2 pi / n) (j + delta), so the arc in phi
        # becomes an inclusive index range in j
        scale = n.to(torch.float64) / (2.0 * math.pi)
        j_lo = torch.ceil(scale * (phi_o - half_width) - delta)
        j_hi = torch.floor(scale * (phi_o + half_width) - delta)

        length = (j_hi - j_lo + 1.0).to(torch.int64).clamp(min=0)
        length = torch.where(whole_ring, n, length)
        # an arc wider than its ring would list points twice
        length = torch.minimum(length, n)

        start = torch.where(whole_ring, torch.zeros_like(n), j_lo.to(torch.int64).remainder(n))

        keep = in_band & (length > 0)
        counts[begin:end] = keep.sum(dim=-1)
        seg_chunks.append(torch.stack([rings[keep], start[keep], length[keep]], dim=-1).to(torch.int32))

    segments = torch.cat(seg_chunks, dim=0) if seg_chunks else torch.empty((0, 3), dtype=torch.int32)

    offsets = torch.zeros(npoints_out + 1, dtype=torch.int32)
    offsets[1:] = counts.cumsum(0).to(torch.int32)

    return NeighborhoodArcsS2(segments=segments, offsets=offsets, ring_base=ring_base, ring_size=ring_size, theta_cutoff=theta_cutoff)


@lru_cache(maxsize=8, typed=True, copy=True)
def precompute_neighborhood_csr_s2(grid_in: GridS2, grid_out: GridS2, theta_cutoff: float, theta_eps: Optional[float] = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Cached CSR form of the neighbourhood, for consumers that want columns not arcs.

    :func:`precompute_neighborhood_arcs_s2` is cached, so layers built on the same
    grid pair and radius already share the arcs. Expanding those arcs is the other
    half of the cost, and without a cache of its own every layer repeats it: a DiT
    backbone stacks tens of attention blocks on one grid, which is tens of identical
    expansions of a pattern that is a pure function of its arguments. Caching here
    rather than inside :meth:`NeighborhoodArcsS2.to_csr` keeps the arcs a plain
    ``NamedTuple``, with no per-instance state to invalidate.

    Parameters
    ----------
    grid_in, grid_out : GridS2
        Input and output grid descriptors.
    theta_cutoff : float
        Geodesic neighbourhood radius, in radians.
    theta_eps : float, optional
        Relative widening of the radius, guarding the boundary against round-off.

    Returns
    -------
    col_idx : torch.Tensor
        ``int64``, flat input indices, ascending within each output point.
    row_off : torch.Tensor
        ``int64``, shape ``(npoints_out + 1,)``, start of each output point's columns.

    See Also
    --------
    precompute_neighborhood_arcs_s2 : the arc form, which the kernels consume.
    """
    return precompute_neighborhood_arcs_s2(grid_in, grid_out, theta_cutoff, theta_eps).to_csr()


def _lon_shifts(grid: GridS2) -> torch.Tensor:
    r"""
    Fractional longitude offset of each ring, in units of one point of that ring.

    Zero on the product grids, where every ring starts at :math:`\lambda = 0`;
    HEALPix staggers successive rings by half a point and reports it as
    ``lon_shifts``.
    """
    shifts = getattr(grid, "lon_shifts", None)
    if shifts is None:
        return torch.zeros(grid.nlat, dtype=torch.float64)
    return shifts.to(torch.float64)


def _flat_colats(grid: GridS2) -> torch.Tensor:
    """Colatitude of every point of a grid, in its flat order."""
    if grid.is_regular:
        return grid.lats.to(torch.float64).repeat_interleave(grid.nlon)
    return torch.repeat_interleave(grid.lats.to(torch.float64), grid.nlon_per_lat)


def _flat_lons(grid: GridS2) -> torch.Tensor:
    """Longitude of every point of a grid, in its flat order."""
    if grid.is_regular:
        return grid.lons().to(torch.float64).tile(grid.nlat)
    all_lons = getattr(grid, "all_lons", None)
    if all_lons is not None:
        return all_lons().to(torch.float64)
    return torch.cat([grid.lons(ilat).to(torch.float64) for ilat in range(grid.nlat)])
