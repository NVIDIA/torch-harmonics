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

"""
Contract for the arc-segment neighbourhood precompute.

The precompute replaces a search with a formula: instead of testing candidate input
points against the cutoff, it solves for the arc each input ring contributes. That
is what makes it work on a ragged grid, where there is no shift carrying one output
point's neighbourhood onto the next one's, and it is also what makes it worth
testing hard -- an error in the arc arithmetic produces a plausible-looking pattern
that is quietly wrong at the seam, at the poles, or on the short rings near them.

Everything here is checked against pairwise great-circle distances, computed
directly from the pixel coordinates. That reference is O(n^2) and knows nothing
about rings, arcs or orderings, so it cannot share a bug with the thing it checks.
"""

import math
import unittest
import unittest.mock

import torch
from parameterized import parameterized

from torch_harmonics.attention._attention_utils import _build_psi_segments
from torch_harmonics.disco.convolution import _precompute_convolution_tensor_s2
from torch_harmonics.filter_basis import get_filter_basis
from torch_harmonics.grid import as_grid
from torch_harmonics.healpix import HealpixGrid
from torch_harmonics.neighborhood import precompute_neighborhood_arcs_s2, precompute_neighborhood_csr_s2

# the same relative widening the precompute applies, so a reference built here
# includes exactly the points that land on the cutoff
_THETA_EPS = 1e-3


def _flat_coords(grid):
    """Colatitude and longitude of every point of a grid, in its flat order."""
    if grid.is_regular:
        colats = grid.lats.to(torch.float64).repeat_interleave(grid.nlon)
        lons = grid.lons().to(torch.float64).tile(grid.nlat)
    else:
        colats = grid.all_lats().to(torch.float64)
        lons = grid.all_lons().to(torch.float64)
    return colats, lons


def _unit_vectors(colats, lons):
    return torch.stack([torch.sin(colats) * torch.cos(lons), torch.sin(colats) * torch.sin(lons), torch.cos(colats)], dim=-1)


def _brute_force_neighborhood(grid_in, grid_out, theta_cutoff):
    """
    Reference pattern, as a boolean ``(npoints_out, npoints_in)`` mask.

    Uses the cosine of the angle rather than the angle itself, so that a point on the
    cutoff is decided by one comparison and not by the round-off of an ``arccos``.
    """
    x_in = _unit_vectors(*_flat_coords(grid_in))
    x_out = _unit_vectors(*_flat_coords(grid_out))
    radius = min((1.0 + _THETA_EPS) * theta_cutoff, math.pi)
    return (x_out @ x_in.T) >= math.cos(radius)


def _arcs_to_mask(arcs, npoints_in):
    """Expand an arc pattern into the same boolean mask, via the documented decode."""
    npoints_out = arcs.offsets.numel() - 1
    mask = torch.zeros(npoints_out, npoints_in, dtype=torch.bool)
    for ipoint in range(npoints_out):
        mask[ipoint, arcs.columns(ipoint)] = True
    return mask


class TestNeighborhoodArcs(unittest.TestCase):
    """The pattern must be the geodesic ball, on ragged and product grids alike."""

    @parameterized.expand([[2], [3], [4], [8]])
    def test_healpix_self_neighborhood_matches_brute_force(self, nside):
        """
        The case the module exists for. Every output pixel is checked, so a mistake
        confined to the short rings at the poles or to one side of the cap/belt join
        cannot hide behind the equatorial majority.
        """
        grid = HealpixGrid(nside=nside)
        cutoff = grid.theta_cutoff()
        arcs = precompute_neighborhood_arcs_s2(grid, grid, cutoff)

        expected = _brute_force_neighborhood(grid, grid, cutoff)
        actual = _arcs_to_mask(arcs, grid.npoints)

        self.assertTrue(torch.equal(actual, expected), msg=f"nside={nside}: {int((actual ^ expected).sum())} of {expected.numel()} entries disagree")
        self.assertEqual(arcs.nnz, int(expected.sum()))

    @parameterized.expand([[4], [8]])
    def test_healpix_resampling_neighborhood_matches_brute_force(self, nside):
        """Cross-resolution, where output and input rings share no structure at all."""
        fine, coarse = HealpixGrid(nside=nside), HealpixGrid(nside=nside // 2)
        cutoff = coarse.theta_cutoff()

        for grid_in, grid_out in [(fine, coarse), (coarse, fine)]:
            with self.subTest(nside_in=grid_in.nside, nside_out=grid_out.nside):
                arcs = precompute_neighborhood_arcs_s2(grid_in, grid_out, cutoff)
                expected = _brute_force_neighborhood(grid_in, grid_out, cutoff)
                self.assertTrue(torch.equal(_arcs_to_mask(arcs, grid_in.npoints), expected))

    @parameterized.expand([["equiangular", 16], ["legendre-gauss", 16], ["lobatto", 17], ["equiangular-trapezoidal", 16]])
    def test_product_grids_match_brute_force(self, name, nlat):
        """
        The generalization must not have cost the regular case. Equiangular and Lobatto
        grids also put points exactly on both poles, where a latitude circle is a single
        location repeated nlon times -- the degenerate case the arc solve has to
        special-case rather than divide by.
        """
        grid = as_grid(name, (nlat, 2 * nlat))
        cutoff = grid.max_latitude_spacing
        arcs = precompute_neighborhood_arcs_s2(grid, grid, cutoff)

        expected = _brute_force_neighborhood(grid, grid, cutoff)
        self.assertTrue(torch.equal(_arcs_to_mask(arcs, grid.npoints), expected))

    def test_it_agrees_with_the_disco_precompute_on_a_product_grid(self):
        """
        The interop property that makes this a drop-in replacement: on a grid where
        both routes are defined they must find the same sparsity, even though one
        evaluates a filter basis at rotated points and the other solves for arcs.

        The DISCO precompute keys its rows by output *latitude* and relies on the
        kernels to reach the other output longitudes by rotating the pattern, so its
        rows are compared against the arcs of the output points at longitude zero.
        """
        grid = as_grid("equiangular", (16, 32))
        cutoff = grid.max_latitude_spacing

        idx, _, roff = _precompute_convolution_tensor_s2(
            grid,
            grid,
            get_filter_basis(kernel_shape=1, basis_type="zernike"),
            theta_cutoff=cutoff,
            transpose_normalization=False,
            basis_norm_mode="none",
            merge_quadrature=True,
        )
        seg, seg_off = _build_psi_segments(idx[2].contiguous(), roff.contiguous(), grid.nlon)

        arcs = precompute_neighborhood_arcs_s2(grid, grid, cutoff)

        for ilat in range(grid.nlat):
            with self.subTest(ilat=ilat):
                legacy = sorted(
                    hi * grid.nlon + (lo + j) % grid.nlon
                    for s in range(int(seg_off[ilat]), int(seg_off[ilat + 1]))
                    for hi, lo, length in [tuple(int(x) for x in seg[s])]
                    for j in range(length)
                )
                self.assertEqual(arcs.columns(ilat * grid.nlon).tolist(), legacy)

    # -- the arc encoding itself ---------------------------------------------

    @parameterized.expand([[2], [4], [8]])
    def test_arcs_are_well_formed(self, nside):
        """
        The invariants the CUDA kernels rely on. ``start`` inside the ring and
        ``length`` no larger than it are what let a kernel wrap with a single
        compare-and-subtract instead of a modulo, which is the whole reason for this
        representation.
        """
        grid = HealpixGrid(nside=nside)
        arcs = precompute_neighborhood_arcs_s2(grid, grid, grid.theta_cutoff())

        ring, start, length = arcs.segments[:, 0].to(torch.int64), arcs.segments[:, 1].to(torch.int64), arcs.segments[:, 2].to(torch.int64)
        sizes = arcs.ring_size[ring]

        self.assertTrue(bool(((ring >= 0) & (ring < grid.nlat)).all()))
        self.assertTrue(bool(((start >= 0) & (start < sizes)).all()), msg="start must be a valid index into its own ring")
        self.assertTrue(bool(((length > 0) & (length <= sizes)).all()), msg="an arc must be non-empty and no longer than its ring")
        self.assertTrue(torch.equal(arcs.ring_base, grid.lon_offsets[:-1]))
        self.assertTrue(torch.equal(arcs.ring_size, grid.nlon_per_lat))

    @parameterized.expand([[2], [4], [8]])
    def test_each_output_point_gets_at_most_one_arc_per_ring(self, nside):
        """
        A geodesic ball meets a latitude circle in *one* arc. Two arcs on the same ring
        would mean the wrapping split a single interval in half, which would double
        some points and, in the kernels, double-count them in the softmax.
        """
        grid = HealpixGrid(nside=nside)
        arcs = precompute_neighborhood_arcs_s2(grid, grid, grid.theta_cutoff())

        for ipoint in range(grid.npoints):
            begin, end = int(arcs.offsets[ipoint]), int(arcs.offsets[ipoint + 1])
            rings = arcs.segments[begin:end, 0]
            self.assertEqual(rings.numel(), rings.unique().numel(), msg=f"pixel {ipoint} has two arcs on one ring")

    @parameterized.expand([[2], [4], [8]])
    def test_offsets_partition_the_segments(self, nside):
        grid = HealpixGrid(nside=nside)
        arcs = precompute_neighborhood_arcs_s2(grid, grid, grid.theta_cutoff())

        self.assertEqual(arcs.offsets.numel(), grid.npoints + 1)
        self.assertEqual(int(arcs.offsets[0]), 0)
        self.assertEqual(int(arcs.offsets[-1]), arcs.segments.shape[0])
        self.assertTrue(bool((arcs.offsets[1:] >= arcs.offsets[:-1]).all()))

    @parameterized.expand([[2], [4], [8]])
    def test_every_output_point_sees_itself(self, nside):
        """A self-neighbourhood that dropped its own centre would break the softmax's
        one guaranteed entry."""
        grid = HealpixGrid(nside=nside)
        arcs = precompute_neighborhood_arcs_s2(grid, grid, grid.theta_cutoff())

        for ipoint in range(grid.npoints):
            self.assertIn(ipoint, arcs.columns(ipoint).tolist())

    def test_arcs_wrap_the_seam(self):
        """
        The case a naive index range gets wrong. A pixel near longitude zero has
        neighbours at both ends of its ring, which must come back as one wrapping arc
        rather than as two, or as the complementary arc across the far side.
        """
        grid = HealpixGrid(nside=8)
        arcs = precompute_neighborhood_arcs_s2(grid, grid, grid.theta_cutoff())

        # first pixel of the equatorial ring, which sits at the seam
        equator = 2 * grid.nside - 1
        ipoint = int(grid.lon_offsets[equator])
        n = int(grid.nlon_per_lat[equator])

        own_ring = [(int(s[1]), int(s[2])) for s in arcs.segments[int(arcs.offsets[ipoint]) : int(arcs.offsets[ipoint + 1])] if int(s[0]) == equator]
        self.assertEqual(len(own_ring), 1)
        start, length = own_ring[0]
        self.assertGreater(start + length, n, msg="the arc at the seam must wrap rather than be clipped")

        expected = _brute_force_neighborhood(grid, grid, grid.theta_cutoff())[ipoint].nonzero().flatten()
        self.assertEqual(arcs.columns(ipoint).tolist(), expected.tolist())

    # -- radius behaviour ----------------------------------------------------

    def test_a_wider_radius_only_adds_neighbours(self):
        grid = HealpixGrid(nside=4)
        narrow = precompute_neighborhood_arcs_s2(grid, grid, grid.theta_cutoff())
        wide = precompute_neighborhood_arcs_s2(grid, grid, 2.0 * grid.theta_cutoff())

        self.assertGreater(wide.nnz, narrow.nnz)
        for ipoint in range(grid.npoints):
            self.assertTrue(set(narrow.columns(ipoint).tolist()) <= set(wide.columns(ipoint).tolist()))

    def test_a_radius_covering_the_sphere_is_global_attention(self):
        """
        A cutoff of pi must mean the whole sphere. The widening would otherwise push
        the radius past the antipode, where the comparison in cos(theta) reverses and
        every point's antipode -- of which HEALPix, being pole-symmetric, gives each
        pixel exactly one -- would drop out.
        """
        grid = HealpixGrid(nside=2)
        arcs = precompute_neighborhood_arcs_s2(grid, grid, math.pi)

        self.assertEqual(arcs.nnz, grid.npoints**2)
        for ipoint in range(grid.npoints):
            self.assertEqual(arcs.columns(ipoint).tolist(), list(range(grid.npoints)))

    def test_a_non_positive_radius_is_rejected(self):
        grid = HealpixGrid(nside=2)
        for bad in [0.0, -1.0]:
            with self.subTest(theta_cutoff=bad):
                with self.assertRaises(ValueError):
                    precompute_neighborhood_arcs_s2(grid, grid, bad)

    def test_the_pattern_is_cached_on_the_descriptors(self):
        """
        The layers call this in their constructor, and a model stacks many of them on
        the same pair of grids. Recomputing per layer would dominate model setup.
        """
        # an nside no other test in this file uses, so the first call below is
        # guaranteed to be a miss rather than a hit on an already-warm entry
        grid = HealpixGrid(nside=5)
        cutoff = grid.theta_cutoff()

        before = precompute_neighborhood_arcs_s2.cache_info()
        precompute_neighborhood_arcs_s2(grid, grid, cutoff)
        after_miss = precompute_neighborhood_arcs_s2.cache_info()
        self.assertEqual(after_miss.misses, before.misses + 1)
        self.assertEqual(after_miss.hits, before.hits)

        # a distinct but equal pair of descriptors must land on that same entry
        precompute_neighborhood_arcs_s2(HealpixGrid(nside=5), HealpixGrid(nside=5), cutoff)
        after_hit = precompute_neighborhood_arcs_s2.cache_info()
        self.assertEqual(after_hit.hits, after_miss.hits + 1)
        self.assertEqual(after_hit.misses, after_miss.misses)

    def test_cached_results_are_independent_copies(self):
        grid = HealpixGrid(nside=2)
        first = precompute_neighborhood_arcs_s2(grid, grid, grid.theta_cutoff())
        first.segments[0, 2] = 0
        second = precompute_neighborhood_arcs_s2(grid, grid, grid.theta_cutoff())
        self.assertGreater(int(second.segments[0, 2]), 0)


class TestCsrExpansion(unittest.TestCase):
    """
    Contract for turning the arcs into a column list.

    ``to_csr`` is vectorized over arcs, which is a good deal less obvious than the
    loop over output points it replaced. ``columns`` is still that loop, one point at
    a time, so it is the reference: the two must agree everywhere, and the cheap way
    to be wrong is at a wrapping arc, where the entries of a row are not ascending in
    the order the arcs produce them.
    """

    @parameterized.expand([(2,), (3,), (4,), (8,)])
    def test_it_agrees_with_the_per_point_expansion(self, nside):
        grid = HealpixGrid(nside=nside)
        arcs = precompute_neighborhood_arcs_s2(grid, grid, grid.theta_cutoff())
        col_idx, row_off = arcs.to_csr()

        self.assertEqual(row_off.numel(), grid.npoints + 1)
        self.assertEqual(int(row_off[0]), 0)
        self.assertEqual(int(row_off[-1]), col_idx.numel())
        self.assertEqual(col_idx.numel(), arcs.nnz)

        for ipoint in range(grid.npoints):
            expected = arcs.columns(ipoint)
            got = col_idx[row_off[ipoint] : row_off[ipoint + 1]]
            self.assertEqual(got.tolist(), expected.tolist(), msg=f"point {ipoint}")

    @parameterized.expand([("equiangular", 16), ("legendre-gauss", 12)])
    def test_it_agrees_with_the_per_point_expansion_on_product_grids(self, name, nlat):
        grid = as_grid(name, (nlat, 2 * nlat))
        arcs = precompute_neighborhood_arcs_s2(grid, grid, grid.theta_cutoff())
        col_idx, row_off = arcs.to_csr()

        for ipoint in range(grid.npoints):
            got = col_idx[row_off[ipoint] : row_off[ipoint + 1]]
            self.assertEqual(got.tolist(), arcs.columns(ipoint).tolist(), msg=f"point {ipoint}")

    def test_the_rows_are_ascending_even_where_the_arcs_wrap(self):
        """
        A row is built from arcs in ring order, and an arc that crosses the seam starts
        high and continues from zero. Sorting is what reconciles that; without it the
        rows would still hold the right columns in the wrong order.
        """
        grid = HealpixGrid(nside=4)
        arcs = precompute_neighborhood_arcs_s2(grid, grid, grid.theta_cutoff())
        col_idx, row_off = arcs.to_csr()

        wrapping = 0
        for ipoint in range(grid.npoints):
            row = col_idx[row_off[ipoint] : row_off[ipoint + 1]]
            self.assertTrue(bool((row[1:] > row[:-1]).all()), msg=f"point {ipoint} not ascending")
            for iseg in range(int(arcs.offsets[ipoint]), int(arcs.offsets[ipoint + 1])):
                ring, start, length = (int(x) for x in arcs.segments[iseg])
                if start + length > int(arcs.ring_size[ring]):
                    wrapping += 1
        # otherwise the assertion above never saw the case it exists for
        self.assertGreater(wrapping, 0)

    def test_it_expands_the_same_pattern_when_it_has_to_chunk(self):
        """
        The entry budget only bites on grids far larger than the suite runs, so drive
        the chunked path directly: chunking must not change the answer.
        """
        grid = HealpixGrid(nside=4)
        arcs = precompute_neighborhood_arcs_s2(grid, grid, grid.theta_cutoff())
        reference = arcs.to_csr()

        for budget in [1, 7, 64, 1 << 20]:
            with self.subTest(budget=budget), unittest.mock.patch("torch_harmonics.neighborhood._CSR_ENTRY_BUDGET", budget):
                col_idx, row_off = arcs.to_csr()
                self.assertEqual(col_idx.tolist(), reference[0].tolist())
                self.assertEqual(row_off.tolist(), reference[1].tolist())

    def test_a_global_radius_expands_to_every_column(self):
        grid = HealpixGrid(nside=2)
        arcs = precompute_neighborhood_arcs_s2(grid, grid, 2.0 * math.pi)
        col_idx, row_off = arcs.to_csr()

        self.assertEqual(col_idx.numel(), grid.npoints**2)
        for ipoint in range(grid.npoints):
            row = col_idx[row_off[ipoint] : row_off[ipoint + 1]]
            self.assertEqual(row.tolist(), list(range(grid.npoints)))

    def test_the_expansion_is_cached_across_layers(self):
        """
        A DiT backbone builds tens of attention blocks on one grid. The arcs were
        already shared; the expansion has to be too, or each block repeats it.
        """
        # an nside no other test in this file uses, so the first call is a miss
        grid = HealpixGrid(nside=7)
        cutoff = grid.theta_cutoff()

        before = precompute_neighborhood_csr_s2.cache_info()
        precompute_neighborhood_csr_s2(grid, grid, cutoff)
        after_miss = precompute_neighborhood_csr_s2.cache_info()
        self.assertEqual(after_miss.misses, before.misses + 1)

        for _ in range(4):
            precompute_neighborhood_csr_s2(HealpixGrid(nside=7), HealpixGrid(nside=7), cutoff)
        after_hits = precompute_neighborhood_csr_s2.cache_info()
        self.assertEqual(after_hits.hits, after_miss.hits + 4)
        self.assertEqual(after_hits.misses, after_miss.misses)

    def test_cached_expansions_are_independent_copies(self):
        grid = HealpixGrid(nside=2)
        cutoff = grid.theta_cutoff()
        first, _ = precompute_neighborhood_csr_s2(grid, grid, cutoff)
        sentinel = int(first[0])
        first[0] = -1
        second, _ = precompute_neighborhood_csr_s2(grid, grid, cutoff)
        self.assertEqual(int(second[0]), sentinel)

    def test_it_matches_the_cached_entry_point(self):
        grid = HealpixGrid(nside=3)
        cutoff = grid.theta_cutoff()
        direct = precompute_neighborhood_arcs_s2(grid, grid, cutoff).to_csr()
        cached = precompute_neighborhood_csr_s2(grid, grid, cutoff)
        self.assertEqual(cached[0].tolist(), direct[0].tolist())
        self.assertEqual(cached[1].tolist(), direct[1].tolist())


if __name__ == "__main__":
    unittest.main()
