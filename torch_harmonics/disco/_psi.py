# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
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
The layouts the DISCO kernels read psi in, built from its COO entries.

The precompute produces psi as entries ``(basis function k, row, column, value)``, a
row being a latitude of the grid the op is keyed by and a column a flat index
``ring * nlon + lon`` into the other grid. The kernels want it in two forms:

**Arcs**, read by the gather and scatter kernels on CPU and CUDA. A row's entries on one
ring are almost always a run of consecutive longitudes -- the support is a geodesic disk,
or part of one -- so a row stores its basis function and latitude once, and each run once
as ``(ring, start, length)``, with the run's values consecutive. A run crossing the seam is
one arc that wraps. This is the arc form of neighborhood attention
(:class:`~torch_harmonics.neighborhood.NeighborhoodArcsS2`) with a value per entry added:
DISCO's support is the same disk, but each basis function weights it differently.

**Kpacked**, read by the tensor-core forward. Where every basis function has the same
support, each entry of a latitude stores ``(ring, lon)`` once and all K values together,
so the MMA contracts over entries with k as its N dimension.

Both are pure functions of the entries, computed with torch ops on whichever device the
entries are on, when a backend is selected.
"""

from typing import NamedTuple, Optional, Tuple

import torch


class DiscoArcsS2(NamedTuple):
    r"""
    psi in arc form.

    For ``nrows`` rows, ``nsegs`` arcs and ``nnz`` entries:

    Attributes
    ----------
    row_ker : torch.Tensor
        int32 ``(nrows,)``, the basis function of each row. Rows are sorted by it, so the
        rows of one basis function are contiguous.
    row_lat : torch.Tensor
        int32 ``(nrows,)``, the latitude of each row: the output latitude of the gather,
        the input latitude of the scatter.
    seg_off : torch.Tensor
        int64 ``(nrows + 1,)``; row ``r``'s arcs are ``seg[seg_off[r]:seg_off[r + 1]]``.
    seg : torch.Tensor
        int32 ``(nsegs, 3)``, ``(ring, start, length)`` per arc, rings ascending within a
        row. ``start`` lies in ``[0, nlon)`` and ``length`` is at most ``nlon``, ``nlon``
        being the longitude count of the grid the columns index; an arc wraps at the end
        of its ring.
    val_off : torch.Tensor
        int64 ``(nrows + 1,)``; row ``r``'s values are ``vals[val_off[r]:val_off[r + 1]]``.
    vals : torch.Tensor
        ``(nnz,)``, the values in the order the arcs walk them.
    """

    row_ker: torch.Tensor
    row_lat: torch.Tensor
    seg_off: torch.Tensor
    seg: torch.Tensor
    val_off: torch.Tensor
    vals: torch.Tensor


def _sort_entries(ker_idx, row_idx, col_idx, vals, *major):
    """Sort the entries by the given index arrays, most significant first."""
    key = torch.zeros_like(ker_idx, dtype=torch.int64)
    for idx in major:
        idx = idx.to(torch.int64)
        key = key * (int(idx.max()) + 1 if idx.numel() else 1) + idx
    order = torch.argsort(key, stable=True)
    return ker_idx[order], row_idx[order], col_idx[order], vals[order]


def build_arcs(ker_idx: torch.Tensor, row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor, nlon: int) -> DiscoArcsS2:
    """
    Encode psi's entries as arcs.

    Rows are ordered by (basis function, latitude) and entries within a row by column, so
    rings ascend and longitudes ascend within a ring; consecutive longitudes on one ring
    then form a run. A row's first run on a ring that starts at longitude 0 and its last
    that ends at ``nlon - 1`` are one run across the seam: they become a single arc,
    placed where the later run was, walking its longitudes and then the earlier run's.

    Lossless -- the arcs hold exactly the entries given -- and the order of summation in
    the kernels is the column order except across such a seam.

    Parameters
    ----------
    ker_idx, row_idx, col_idx, vals : torch.Tensor
        psi's entries, in any order.
    nlon : int
        Longitudes of the grid the columns index: ``nlon_in`` for the gather, ``nlon_out``
        for the scatter.
    """
    ker_idx, row_idx, col_idx, vals = _sort_entries(ker_idx, row_idx, col_idx, vals, ker_idx, row_idx, col_idx)
    nnz = col_idx.numel()
    device = col_idx.device

    # rows: maximal groups of equal (basis function, latitude)
    new_row = torch.ones(nnz, dtype=torch.bool, device=device)
    new_row[1:] = (ker_idx[1:] != ker_idx[:-1]) | (row_idx[1:] != row_idx[:-1])
    row_first = torch.nonzero(new_row).squeeze(-1)
    nrows = row_first.numel()
    row_of = torch.cumsum(new_row.to(torch.int64), 0) - 1

    ring = col_idx // nlon
    lon = col_idx % nlon

    # runs: a new one at each row start, ring change, or gap in longitude
    new_run = new_row.clone()
    new_run[1:] |= (ring[1:] != ring[:-1]) | (lon[1:] != lon[:-1] + 1)
    run_first = torch.nonzero(new_run).squeeze(-1)
    run_len = torch.diff(run_first, append=run_first.new_tensor([nnz]))
    run_row = row_of[run_first]
    run_ring = ring[run_first]
    run_start = lon[run_first]
    nruns = run_first.numel()

    # the runs of one (row, ring) are consecutive: merge first and last across the seam
    group_first = torch.ones(nruns, dtype=torch.bool, device=device)
    group_first[1:] = (run_row[1:] != run_row[:-1]) | (run_ring[1:] != run_ring[:-1])
    first_idx = torch.nonzero(group_first).squeeze(-1)
    last_idx = torch.diff(first_idx, append=first_idx.new_tensor([nruns])) + first_idx - 1
    wraps = (last_idx > first_idx) & (run_start[first_idx] == 0) & (run_start[last_idx] + run_len[last_idx] == nlon)
    merged_first, merged_last = first_idx[wraps], last_idx[wraps]

    # move each merged first run's values right behind its partner's, then drop it as an
    # arc of its own; the per-row value ranges are unchanged by this
    # (integer keys: 2*position, and one past the partner's for a moved run -- MPS has no float64)
    key = 2 * torch.arange(nruns, device=device)
    key[merged_first] = 2 * merged_last + 1
    order = torch.argsort(key)
    piece_first, piece_len = run_first[order], run_len[order]
    gather = torch.repeat_interleave(piece_first - (torch.cumsum(piece_len, 0) - piece_len), piece_len) + torch.arange(nnz, device=device)
    vals = vals[gather].contiguous()

    is_arc = torch.ones(nruns, dtype=torch.bool, device=device)
    is_arc[merged_first] = False
    arc_len = run_len.clone()
    arc_len[merged_last] += run_len[merged_first]
    arc_idx = torch.nonzero(is_arc).squeeze(-1)
    seg = torch.stack([run_ring[arc_idx], run_start[arc_idx], arc_len[arc_idx]], dim=1).to(torch.int32).contiguous()

    seg_off = torch.zeros(nrows + 1, dtype=torch.int64, device=device)
    seg_off[1:] = torch.cumsum(torch.bincount(run_row[arc_idx], minlength=nrows), 0)
    val_off = torch.zeros(nrows + 1, dtype=torch.int64, device=device)
    val_off[:-1] = row_first
    val_off[-1] = nnz

    row_ker = ker_idx[row_first].to(torch.int32).contiguous()
    row_lat = row_idx[row_first].to(torch.int32).contiguous()
    return DiscoArcsS2(row_ker, row_lat, seg_off, seg, val_off, vals)


def arcs_to_coo(arcs: DiscoArcsS2, nlon: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand arcs back into entries ``(ker_idx, row_idx, col_idx, vals)``, in arc order; the inverse of :func:`build_arcs`."""
    seg = arcs.seg.to(torch.int64)
    device = seg.device
    ring, start, length = seg[:, 0], seg[:, 1], seg[:, 2]
    nnz = int(length.sum())
    arc_of = torch.repeat_interleave(torch.arange(seg.shape[0], device=device), length)
    within = torch.arange(nnz, device=device) - torch.repeat_interleave(torch.cumsum(length, 0) - length, length)
    col = ring[arc_of] * nlon + (start[arc_of] + within) % nlon
    arcs_per_row = arcs.seg_off[1:] - arcs.seg_off[:-1]
    row_of = torch.repeat_interleave(torch.arange(arcs_per_row.numel(), device=device), arcs_per_row)[arc_of]
    return arcs.row_ker.to(torch.int64)[row_of], arcs.row_lat.to(torch.int64)[row_of], col, arcs.vals


def build_kpacked(
    ker_idx: torch.Tensor, row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor, kernel_size: int, k_pad: int, nrows: int, nlon: int
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Encode psi's entries in the blocked layout of the tensor-core forward, if it applies.

    The layout stores each (latitude, column) once with all K values beside it, so it
    needs every basis function to have exactly the same support. That holds for the
    harmonic basis but not, say, the piecewise-linear one, whose functions cover
    different annuli; then this returns ``None``.

    Returns
    -------
    pack_idx : torch.Tensor
        int64 ``(npoints, 2)``, ``(ring, lon)`` per point, latitudes ascending and columns
        ascending within one.
    pack_val : torch.Tensor
        ``(npoints, k_pad)``, the K values of each point, zero-padded to ``k_pad``.
    pack_offset : torch.Tensor
        int64 ``(nrows + 1,)``; latitude ``r``'s points are
        ``pack_idx[pack_offset[r]:pack_offset[r + 1]]``.
    """
    ker_idx, row_idx, col_idx, vals = _sort_entries(ker_idx, row_idx, col_idx, vals, row_idx, col_idx, ker_idx)
    device = col_idx.device
    nnz = col_idx.numel()
    if nnz % kernel_size != 0:
        return None

    # after the sort each point's entries are adjacent and ordered by k, so the support is
    # shared exactly when the entries fall into consecutive blocks of K, one per k
    npoints = nnz // kernel_size
    blk_row = row_idx.view(npoints, kernel_size)
    blk_col = col_idx.view(npoints, kernel_size)
    blk_ker = ker_idx.view(npoints, kernel_size)
    shared = (
        bool((blk_row == blk_row[:, :1]).all())
        and bool((blk_col == blk_col[:, :1]).all())
        and bool((blk_ker == torch.arange(kernel_size, dtype=blk_ker.dtype, device=device)).all())
    )
    if not shared:
        return None

    # the layout is indexed by the latitudes of the forward's output; rows beyond them mean
    # a psi of another shape -- a transpose's, keyed by input latitude -- was passed
    if npoints and int(blk_row[:, 0].max()) >= nrows:
        raise ValueError(f"psi rows reach latitude {int(blk_row[:, 0].max())}, but the kpacked layout is sized for {nrows}; it serves the forward direction only")

    col = blk_col[:, 0]
    pack_idx = torch.stack([col // nlon, col % nlon], dim=1).to(torch.int64).contiguous()
    pack_val = torch.zeros(npoints, k_pad, dtype=vals.dtype, device=device)
    pack_val[:, :kernel_size] = vals.view(npoints, kernel_size)
    pack_offset = torch.zeros(nrows + 1, dtype=torch.int64, device=device)
    pack_offset[1:] = torch.cumsum(torch.bincount(blk_row[:, 0].to(torch.int64), minlength=nrows), 0)
    return pack_idx, pack_val.contiguous(), pack_offset


def build_split(arcs: DiscoArcsS2, kernel_size: int) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Index the arc rows per basis function, for the spatial-first input gradient.

    That gradient runs the scatter once per basis function with K = 1. The rows of one
    basis function are contiguous and their arc and value offsets absolute, so each call
    reads a slice of the same arrays; all it needs besides is a row_ker of zeros, one
    shared vector as long as the largest block, and the block boundaries -- as Python ints,
    since they slice tensors inside the backward, where a tensor bound would sync.

    Returns
    -------
    split_ker : torch.Tensor
        int32 zeros, as long as the largest per-basis-function block of rows.
    row_offsets : Tuple[int, ...]
        ``kernel_size + 1`` boundaries: basis function ``k`` owns rows
        ``row_offsets[k]:row_offsets[k + 1]``.
    """
    rows_per_k = torch.bincount(arcs.row_ker.to(torch.int64), minlength=kernel_size)
    row_offsets = (0, *torch.cumsum(rows_per_k, 0).tolist())
    split_ker = torch.zeros(int(rows_per_k.max()) if rows_per_k.numel() else 0, dtype=torch.int32, device=rows_per_k.device)
    return split_ker, tuple(int(r) for r in row_offsets)
