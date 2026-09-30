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
psi in arc form, and the raw ``forward_arcs`` / ``backward_arcs`` operators.

Experimental: nothing in the layers uses this yet. It exists so the arc kernels can be
compared against the CSR ones on identical inputs before anything is wired in -- see
``performance/disco/bench_arcs_ab.py``.

The CSR psi stores, per nonzero, a basis function, a latitude and a flat column
``ring * nlon + lon``, and the kernels recover the ring and longitude from the column. But
a row's nonzeros on one ring are almost always a run of consecutive longitudes -- the
support is a geodesic disk, or part of one -- so the arc form stores a row's basis
function and latitude once, and each run once as ``(ring, start, length)``, with the
values of each run consecutive. A run that crosses the seam is one arc that wraps.

Arrays, for ``nrows`` psi rows and ``nsegs`` arcs:

``row_ker``, ``row_lat`` : int32 ``(nrows,)``
    The basis function and the latitude (output latitude of the gather, input latitude of
    the scatter) of each row.
``seg_off`` : int64 ``(nrows + 1,)``
    Row ``r``'s arcs are ``seg[seg_off[r]:seg_off[r + 1]]``.
``seg`` : int32 ``(nsegs, 3)``
    ``(ring, start, length)`` per arc, ``start`` in ``[0, nlon)`` and ``length`` at most
    ``nlon``, where ``nlon`` is the longitude count of the grid the columns index.
``val_off`` : int64 ``(nrows + 1,)``
    Row ``r``'s values are ``vals[val_off[r]:val_off[r + 1]]``, in arc order.
``vals`` : ``(nnz,)``
    The values, in the order the arcs walk them.
"""

from typing import NamedTuple

import torch
from disco_helpers import optimized_kernels_is_available

from .. import disco_kernels
from .._disco_utils import _compute_dtype


class DiscoArcsS2(NamedTuple):
    """psi in arc form; see the module docstring for the layout."""

    row_ker: torch.Tensor
    row_lat: torch.Tensor
    seg_off: torch.Tensor
    seg: torch.Tensor
    val_off: torch.Tensor
    vals: torch.Tensor

    def to_device(self, device) -> "DiscoArcsS2":
        """Move every array; the values keep their dtype, the ops cast them to the compute dtype."""
        return DiscoArcsS2(*(t.to(device) for t in self))

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self)


def csr_to_arcs(roff_idx: torch.Tensor, ker_idx: torch.Tensor, row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor, nlon: int) -> DiscoArcsS2:
    """
    Re-encode a CSR psi, as :func:`preprocess_psi` leaves it, in arc form.

    Lossless: the arcs hold exactly the CSR's nonzeros. Within a row the order is the
    CSR's -- rings ascending, longitudes ascending -- except that a run ending at the last
    longitude of a ring and a run starting at longitude 0 of the same ring become one arc
    that wraps, placed where the later run was. The kernels' sums therefore differ from the
    CSR kernels' by reassociation at most.

    Parameters
    ----------
    roff_idx, ker_idx, row_idx, col_idx, vals : torch.Tensor
        The CSR psi, rows grouped by (basis function, latitude).
    nlon : int
        Longitudes of the grid the columns index: ``nlon_in`` for the gather, ``nlon_out``
        for the scatter.
    """
    roff_idx, ker_idx, row_idx, col_idx, vals = (t.cpu() for t in (roff_idx, ker_idx, row_idx, col_idx, vals))
    nrows = roff_idx.numel() - 1
    nnz = col_idx.numel()

    counts = roff_idx[1:] - roff_idx[:-1]
    row_of = torch.repeat_interleave(torch.arange(nrows), counts)
    ring = col_idx // nlon
    lon = col_idx % nlon

    # a run starts at a row's first nonzero, on a new ring, or after a gap in longitude
    new_run = torch.ones(nnz, dtype=torch.bool)
    if nnz > 1:
        same_row = row_of[1:] == row_of[:-1]
        new_run[1:] = ~(same_row & (ring[1:] == ring[:-1]) & (lon[1:] == lon[:-1] + 1))
    run_first = torch.nonzero(new_run).squeeze(-1)
    run_len = torch.diff(run_first, append=torch.tensor([nnz]))
    run_row = row_of[run_first]
    run_ring = ring[run_first]
    run_start = lon[run_first]
    nruns = run_first.numel()

    # the runs of one (row, ring) are consecutive; where the first starts at longitude 0
    # and the last ends at the last longitude, and they are different runs, the pair is one
    # arc across the seam
    group = run_row * (int(ring.max()) + 1 if nnz else 1) + run_ring
    group_first = torch.ones(nruns, dtype=torch.bool)
    group_first[1:] = group[1:] != group[:-1]
    first_idx = torch.nonzero(group_first).squeeze(-1)
    last_idx = torch.diff(first_idx, append=torch.tensor([nruns])) + first_idx - 1
    wraps = (last_idx > first_idx) & (run_start[first_idx] == 0) & (run_start[last_idx] + run_len[last_idx] == nlon)
    merged_first = first_idx[wraps]
    merged_last = last_idx[wraps]

    # order the runs so that each merged first run sits right after its partner, then drop
    # it as an arc of its own while keeping its values there
    key = torch.arange(nruns, dtype=torch.float64)
    key[merged_first] = merged_last.to(torch.float64) + 0.5
    order = torch.argsort(key)
    piece_first = run_first[order]
    piece_len = run_len[order]
    starts = torch.repeat_interleave(piece_first - (torch.cumsum(piece_len, 0) - piece_len), piece_len)
    val_perm = starts + torch.arange(nnz)
    vals_out = vals[val_perm].contiguous()

    is_arc = torch.ones(nruns, dtype=torch.bool)
    is_arc[merged_first] = False
    arc_len = run_len.clone()
    arc_len[merged_last] += run_len[merged_first]
    arc_idx = torch.nonzero(is_arc).squeeze(-1)
    seg = torch.stack([run_ring[arc_idx], run_start[arc_idx], arc_len[arc_idx]], dim=1).to(torch.int32).contiguous()

    arcs_per_row = torch.bincount(run_row[arc_idx], minlength=nrows)
    seg_off = torch.zeros(nrows + 1, dtype=torch.int64)
    seg_off[1:] = torch.cumsum(arcs_per_row, 0)
    val_off = roff_idx.to(torch.int64).clone()

    row_start = roff_idx[:-1]
    row_ker = ker_idx[row_start].to(torch.int32).contiguous()
    row_lat = row_idx[row_start].to(torch.int32).contiguous()
    return DiscoArcsS2(row_ker, row_lat, seg_off, seg, val_off, vals_out)


def arcs_to_coo(arcs: DiscoArcsS2, nlon: int):
    """Expand arcs back to ``(ker_idx, row_idx, col_idx, vals)``, in arc order: the inverse of :func:`csr_to_arcs`, for checking it."""
    seg = arcs.seg.cpu().to(torch.int64)
    ring, start, length = seg[:, 0], seg[:, 1], seg[:, 2]
    nnz = int(length.sum())
    arc_of = torch.repeat_interleave(torch.arange(seg.shape[0]), length)
    within = torch.arange(nnz) - torch.repeat_interleave(torch.cumsum(length, 0) - length, length)
    col = ring[arc_of] * nlon + (start[arc_of] + within) % nlon
    arcs_per_row = (arcs.seg_off[1:] - arcs.seg_off[:-1]).cpu()
    row_of_arc = torch.repeat_interleave(torch.arange(arcs_per_row.numel()), arcs_per_row)
    row_of = row_of_arc[arc_of]
    return arcs.row_ker.cpu().to(torch.int64)[row_of], arcs.row_lat.cpu().to(torch.int64)[row_of], col, arcs.vals.cpu()


if optimized_kernels_is_available():

    @torch.library.register_fake("disco_kernels::forward_arcs")
    def _(inp, row_ker, row_lat, seg_off, seg, val_off, vals, kernel_size: int, nlat_out: int, nlon_out: int):
        return inp.new_empty((inp.shape[0], inp.shape[1], kernel_size, nlat_out, nlon_out))

    @torch.library.register_fake("disco_kernels::backward_arcs")
    def _(inp, row_ker, row_lat, seg_off, seg, val_off, vals, kernel_size: int, nlat_out: int, nlon_out: int):
        return inp.new_empty((inp.shape[0], inp.shape[1], nlat_out, nlon_out))


def forward_arcs(inp: torch.Tensor, arcs: DiscoArcsS2, kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
    """The gather with arc psi, ``(B, C, Hi, Wi) -> (B, C, K, Ho, Wo)``; the raw op, no autograd."""
    vals = arcs.vals.to(_compute_dtype(inp.dtype))
    return disco_kernels.forward_arcs.default(inp.contiguous(), arcs.row_ker, arcs.row_lat, arcs.seg_off, arcs.seg, arcs.val_off, vals, kernel_size, nlat_out, nlon_out)


def backward_arcs(inp: torch.Tensor, arcs: DiscoArcsS2, kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
    """The scatter with arc psi, ``(B, C, K, Hi, Wi) -> (B, C, Ho, Wo)``; the raw op, no autograd."""
    vals = arcs.vals.to(_compute_dtype(inp.dtype))
    return disco_kernels.backward_arcs.default(inp.contiguous(), arcs.row_ker, arcs.row_lat, arcs.seg_off, arcs.seg, arcs.val_off, vals, kernel_size, nlat_out, nlon_out)
