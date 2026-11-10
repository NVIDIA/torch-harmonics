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
The HEALPix grid as a :class:`~torch_harmonics.grid.GridS2`.

HEALPix (Hierarchical Equal Area isoLatitude Pixelization, Gorski et al. 2005)
tessellates the sphere into ``12 * nside**2`` pixels of exactly equal area whose
centres lie on ``4 * nside - 1`` rings of constant latitude. The equal-area
property is what makes it the grid of choice for a generative model on the sphere:
no pixel carries more of the sphere than any other, so a per-pixel loss needs no
area weighting and the poles are not oversampled the way they are on an
equiangular grid.

The isolatitude property is what makes it expressible here at all. The DISCO
convolutions, neighborhood attention, the quadrature and the spherical harmonic
transforms need only that the points be organized into rings of constant
colatitude, which is exactly what
:attr:`~torch_harmonics.grid.GridS2.nlon_per_lat` and
:attr:`~torch_harmonics.grid.GridS2.lon_offsets` describe. What HEALPix does not
give is a *rectangular* ring structure: ring sizes grow from 4 at the pole to
``4 * nside`` at the equator, so a field cannot be stored as ``(nlat, nlon)``.
That is the raggedness the descriptor protocol exists to carry.

Pixel order
-----------
Fields on this grid are flat ``(..., npix)`` tensors in **RING** order: pixels are
numbered ring by ring from the north pole, and within a ring by increasing
longitude. This is the ordering in which a ring is contiguous in memory, which is
what lets the localized operators keep describing an output point's stencil as a
handful of longitude *arcs* -- one interval of pixel indices per ring it touches --
rather than as an unstructured index list. NEST order, HEALPix's other convention,
interleaves the rings and destroys that contiguity; it is not supported, and a field
in NEST order has to be reordered to RING before it is handed to a layer.

Notes
-----
Only RING geometry is implemented, and the ring formulas below are valid for any
``nside >= 1``. NEST order and the hierarchical refinement HEALPix is named for
additionally require ``nside`` to be a power of two, so interoperating with
``healpy`` or ``earth2grid``, which index by ``level = log2(nside)``, is restricted
to those; :attr:`HealpixGrid.level` enforces it.

References
----------
.. [1] Gorski K. M., Hivon E., Banday A. J., et al.; HEALPix: A Framework for
    High-Resolution Discretization and Fast Analysis of Data Distributed on the
    Sphere; The Astrophysical Journal, 622:759-771, 2005.
"""

import numbers
from bisect import bisect_left
from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

import torch

from torch_harmonics.cache import lru_cache
from torch_harmonics.grid import GridS2, GridShardS2, _as_int, _grid_quad_weights

__all__ = ["HealpixGrid", "HealpixGridShardS2", "healpix_ring_structure", "healpix_ring_weights"]


@lru_cache(typed=True, copy=True)
def healpix_ring_structure(nside: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Ring geometry of a HEALPix grid, in closed form.

    Evaluates the standard RING-order pixel-centre formulas [1]_ for every ring at
    once. Rings are indexed :math:`i = 1 \dots 4 N - 1` from the north pole, and
    split into three regimes:

    ==================  ======================  ==============================================  ===========
    regime              rings                   :math:`\cos\theta_i`                            ring size
    ==================  ======================  ==============================================  ===========
    north polar cap     :math:`i < N`           :math:`1 - i^2 / (3 N^2)`                        :math:`4 i`
    equatorial belt     :math:`N \le i \le 3N`   :math:`4/3 - 2 i / (3 N)`                       :math:`4 N`
    south polar cap     :math:`i > 3N`          :math:`-(1 - (4N - i)^2 / (3 N^2))`              :math:`4 (4N - i)`
    ==================  ======================  ==============================================  ===========

    The two cap formulas agree with the belt one at the joins :math:`i = N` and
    :math:`i = 3N`, where :math:`\cos\theta = \pm 2/3`, so the rings are strictly
    ordered from pole to pole.

    Longitudes on ring :math:`i` are equispaced, :math:`\lambda_j = \frac{2\pi}{n_i}
    (j + \delta_i)`, with a fractional offset :math:`\delta_i` of half a pixel on
    every cap ring and alternating between :math:`1/2` and :math:`0` across the belt,
    starting at :math:`1/2` on ring :math:`i = N`. Those offsets are what stagger
    successive rings and give the pixels their rhombic shape; a consumer that
    ignored them would place every ring's points on the same meridians.

    Parameters
    ----------
    nside : int
        Resolution parameter :math:`N`, at least 1.

    Returns
    -------
    nlon_per_lat : torch.Tensor
        Number of pixels on each ring, ``int64``, shape ``(4 * nside - 1,)``.
    lon_shifts : torch.Tensor
        Fractional longitude offset :math:`\delta_i` of each ring, ``float64``,
        shape ``(4 * nside - 1,)``. Either ``0.5`` or ``0.0``.
    colats : torch.Tensor
        Colatitude :math:`\theta_i` of each ring in radians, ``float64``, ascending,
        shape ``(4 * nside - 1,)``.

    References
    ----------
    .. [1] Gorski K. M., et al.; HEALPix: A Framework for High-Resolution
        Discretization and Fast Analysis of Data Distributed on the Sphere;
        ApJ 622:759, 2005; eqs. 4-9.
    """
    if nside < 1:
        raise ValueError(f"nside must be at least 1, got {nside}")

    i = torch.arange(1, 4 * nside, dtype=torch.int64)
    in_north_cap = i < nside
    in_south_cap = i > 3 * nside

    # mirrored ring index, so both caps share one formula
    i_cap = torch.where(in_south_cap, 4 * nside - i, i)

    nlon_per_lat = torch.where(in_north_cap | in_south_cap, 4 * i_cap, torch.full_like(i, 4 * nside))

    x = i.to(torch.float64)
    x_cap = i_cap.to(torch.float64)
    n = float(nside)

    cap_z = 1.0 - x_cap * x_cap / (3.0 * n * n)
    z = torch.where(in_north_cap, cap_z, torch.where(in_south_cap, -cap_z, 4.0 / 3.0 - 2.0 * x / (3.0 * n)))
    colats = torch.arccos(z.clamp(-1.0, 1.0))

    # half a pixel on the caps; on the belt the stagger alternates, in phase with
    # the cap rings on both sides
    belt_shift = 0.5 * ((i - nside + 1) % 2).to(torch.float64)
    lon_shifts = torch.where(in_north_cap | in_south_cap, torch.full_like(belt_shift, 0.5), belt_shift)

    return nlon_per_lat, lon_shifts, colats


@lru_cache(typed=True, copy=True)
def healpix_ring_weights(nside: int) -> torch.Tensor:
    r"""
    Latitudinal analysis weights of the HEALPix SHT, one per ring.

    No quadrature on HEALPix is exact for spherical harmonics, so its analysis is
    approximate whichever weights it uses. These *ring weights* are the smallest
    correction to the equal-area weights :math:`4\pi / N_{pix}` per pixel that integrates
    every zonal harmonic :math:`P_l(\cos\theta)` of degree :math:`l < 3 N` exactly:

    .. math::

        \min_v \lVert v - u \rVert_2
        \quad \text{subject to} \quad
        \sum_k n_k\, v_k\, P_l(z_k) = 4\pi\, \delta_{l0}, \quad l < 3N,

    with :math:`v_k` the weight of a pixel on ring :math:`k`, :math:`n_k` its ring size,
    :math:`z_k = \cos\theta_k` and :math:`u` the equal-area weight. The weights are
    symmetric about the equator, so odd degrees hold automatically and only the
    :math:`2N` northern rings are unknowns. The degree :math:`3N` is chosen for
    conditioning: the constraint matrix then has a condition number of about
    :math:`2N`, and the weights stay within a few percent of :math:`u` at every
    resolution (between :math:`-3\%` and :math:`+14\%`). Requiring exactness up to
    :math:`4N`, which the ring count would allow, is ill-conditioned beyond
    :math:`N \approx 16`.

    The resulting analysis error matches that of the precomputed ring weights HEALPix
    and healpy ship for ``use_weights=True`` [1]_. They improve the zonal coefficients
    by orders of magnitude but the others only modestly: most of the analysis error of a
    band-limited field comes from orders :math:`m > 0`, which no choice of ring weights
    corrects. Jacobi iteration, ``RealSHT(..., iter=k)``, removes it.

    Parameters
    ----------
    nside : int
        Resolution parameter :math:`N`, at least 1.

    Returns
    -------
    torch.Tensor
        Weights in the :math:`\cos\theta` domain, like
        :attr:`~torch_harmonics.grid.GridS2.colat_weights`: ``float64``, shape
        ``(4 * nside - 1,)``, summing to 2. Ring :math:`k` carries
        :math:`w_k = n_k v_k / (2\pi)`.

    References
    ----------
    .. [1] Gorski K. M., et al.; HEALPix: A Framework for High-Resolution
        Discretization and Fast Analysis of Data Distributed on the Sphere;
        ApJ 622:759, 2005.
    """
    if nside < 1:
        raise ValueError(f"nside must be at least 1, got {nside}")

    nlon_per_lat, _, colats = healpix_ring_structure(nside)
    npix = 12 * nside * nside

    # the 2N northern rings, the last of them the equator; every other one has a mirror
    nnorth = 2 * nside
    z = torch.cos(colats[:nnorth])
    multiplicity = nlon_per_lat[:nnorth].to(torch.float64) * 2.0
    multiplicity[-1] /= 2.0

    # P_l(z) for l < 3N by the three-term recurrence, keeping the even degrees
    ldegree = 3 * nside
    rows = []
    p_prev, p = torch.ones_like(z), z.clone()
    rows.append(p_prev)
    for l in range(1, ldegree - 1):
        p_prev, p = p, ((2 * l + 1) * z * p - l * p_prev) / (l + 1)
        if (l + 1) % 2 == 0:
            rows.append(p)
    A = torch.stack(rows) * multiplicity

    b = torch.zeros(A.shape[0], dtype=torch.float64)
    b[0] = 4.0 * torch.pi
    u = torch.full((nnorth,), 4.0 * torch.pi / npix, dtype=torch.float64)

    # minimum-norm correction, v = u + A^T (A A^T)^{-1} (b - A u)
    gram = A @ A.T
    v = u + A.T @ torch.linalg.solve(gram, b - A @ u)

    pixel_weights = torch.cat([v, v[:-1].flip(0)])
    return nlon_per_lat.to(torch.float64) * pixel_weights / (2.0 * torch.pi)


@dataclass(frozen=True, eq=False)
class HealpixGrid(GridS2):
    r"""
    HEALPix grid in RING pixel order.

    A ragged :class:`~torch_harmonics.grid.GridS2`: ``12 * nside**2`` equal-area
    pixels on ``4 * nside - 1`` isolatitude rings of unequal length. A field on it is
    a flat ``(..., npix)`` tensor.

    Parameters
    ----------
    nside : int
        Resolution parameter, at least 1. Rings carry between 4 and ``4 * nside``
        pixels and the sphere is covered by ``12 * nside**2`` of them.

    Notes
    -----
    :class:`~torch_harmonics.RealSHT`, :class:`~torch_harmonics.InverseRealSHT`,
    :class:`~torch_harmonics.RealVectorSHT` and :class:`~torch_harmonics.InverseRealVectorSHT`
    accept this grid. Synthesis is exact; analysis is approximate, since no quadrature on
    HEALPix makes the spherical harmonics discretely orthogonal. It uses
    :func:`healpix_ring_weights` and can be refined by Jacobi iteration, see
    :class:`~torch_harmonics.RealSHT`. The default truncation is ``lmax = mmax = 2 *
    nside``, and a forward transform accepts up to ``3 * nside``
    (:attr:`max_recoverable_degree`).

    Examples
    --------
    >>> from torch_harmonics import HealpixGrid
    >>> grid = HealpixGrid(nside=2)
    >>> grid.npoints, grid.nrings
    (48, 7)
    >>> grid.shape
    (48,)
    >>> grid.nlon_per_lat.tolist()
    [4, 8, 8, 8, 8, 8, 4]
    >>> round(float(grid.colat_weights.sum()), 6)  # a latitudinal rule sums to 2
    2.0
    >>> round(float(grid.quad_weights.sum()), 6)  # 4 pi, one entry per pixel
    12.566371
    """

    grid_type: ClassVar[str] = "healpix"

    nside: int

    def __post_init__(self):
        super().__post_init__()
        # accepts any integral type and normalizes to int, so that a numpy nside does
        # not end up in the descriptor's key and therefore in repr and to_dict
        _as_int(self, "nside")
        if self.nside < 1:
            raise ValueError(f"nside must be at least 1, got {self.nside}")

    # -- resolution ----------------------------------------------------------

    @property
    def nrings(self) -> int:
        r"""Number of rings, :math:`4 N - 1`."""
        return 4 * self.nside - 1

    @property
    def npoints(self) -> int:
        r"""Number of pixels, :math:`12 N^2`."""
        return 12 * self.nside * self.nside

    @property
    def level(self) -> int:
        r"""
        Refinement level :math:`\log_2 N`.

        Raises
        ------
        ValueError
            If ``nside`` is not a power of two.
        """
        if self.nside & (self.nside - 1):
            raise ValueError(f"nside={self.nside} is not a power of two, so it has no HEALPix refinement level")
        return self.nside.bit_length() - 1

    # -- weights -------------------------------------------------------------

    @property
    def is_equal_area(self) -> bool:
        """``True``: every pixel covers :math:`4\\pi / (12 N^2)`."""
        return True

    # -- raggedness ----------------------------------------------------------

    @property
    def is_regular(self) -> bool:
        return False

    @property
    def nlon_per_lat(self) -> torch.Tensor:
        nlon_per_lat, _, _ = healpix_ring_structure(self.nside)
        return nlon_per_lat

    @property
    def lon_offsets(self) -> torch.Tensor:
        return torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(self.nlon_per_lat, dim=0)])

    @property
    def lon_shifts(self) -> torch.Tensor:
        r"""
        Fractional longitude offset of each ring, shape ``(nrings,)``, in units of one
        pixel of that ring: either 0 or 1/2.
        """
        _, lon_shifts, _ = healpix_ring_structure(self.nside)
        return lon_shifts

    # -- geometry ------------------------------------------------------------

    @property
    def colats(self) -> torch.Tensor:
        r"""Ring colatitudes, shape ``(nrings,)``. :attr:`~torch_harmonics.grid.GridS2.lats` is the geographic latitude."""
        _, _, colats = healpix_ring_structure(self.nside)
        return colats

    @property
    def colat_weights(self) -> torch.Tensor:
        r"""
        Latitudinal weights of the equal-area rule, :math:`w_k = 2 n_k / N_{pix}`.

        Proportional to the ring size :math:`n_k` and summing to 2, so that every pixel
        carries :math:`4\pi / N_{pix}`.
        """
        return 2.0 * self.nlon_per_lat.to(torch.float64) / self.npoints

    def lons(self, ilat: Optional[int] = None) -> torch.Tensor:
        r"""
        Longitudes of one ring, :math:`\lambda_j = \frac{2\pi}{n_k}(j + \delta_k)`.

        Parameters
        ----------
        ilat : int
            Ring index, ``0 <= ilat < nrings``. Required, since rings differ in length
            and offset; use :meth:`all_lons` for every pixel at once.

        Returns
        -------
        torch.Tensor
            Longitudes in radians, ``float64``, shape ``(nlon_per_lat[ilat],)``.

        Raises
        ------
        ValueError
            If ``ilat`` is omitted or out of range.
        """
        if ilat is None:
            raise ValueError(
                f"{type(self).__name__} is ragged, so lons() needs a ring index: ring lengths run from 4 to "
                f"{4 * self.nside} and successive rings are staggered by half a pixel. Use all_lons() for the "
                "longitude of every pixel at once."
            )
        if not -self.nrings <= ilat < self.nrings:
            raise ValueError(f"ilat must be in [0, {self.nrings}), got {ilat}")

        ilat = ilat % self.nrings
        nlon_per_lat, lon_shifts, _ = healpix_ring_structure(self.nside)
        n = int(nlon_per_lat[ilat].item())
        return (2.0 * torch.pi / n) * (torch.arange(n, dtype=torch.float64) + lon_shifts[ilat])

    def all_lons(self) -> torch.Tensor:
        r"""Longitude of every pixel, shape ``(npoints,)``, in RING order."""
        nlon_per_lat, lon_shifts, _ = healpix_ring_structure(self.nside)
        offsets = self.lon_offsets

        # index of each pixel within its own ring
        ring_of = torch.repeat_interleave(torch.arange(self.nrings, dtype=torch.int64), nlon_per_lat)
        j = torch.arange(self.npoints, dtype=torch.int64) - offsets[ring_of]

        n = nlon_per_lat[ring_of].to(torch.float64)
        return (2.0 * torch.pi / n) * (j.to(torch.float64) + lon_shifts[ring_of])

    def all_colats(self) -> torch.Tensor:
        r"""Colatitude of every pixel, shape ``(npoints,)``, in RING order; equal to ``coords[:, 0]``."""
        return torch.repeat_interleave(self.colats, self.nlon_per_lat)

    # -- derived quantities --------------------------------------------------

    @property
    def is_uniform_in_theta(self) -> bool:
        return False

    # max_node_spacing is deliberately not overridden. The latitudinal spacing alone
    # falls short of a pixel's own nearest neighbours on HEALPix -- its in-ring spacing
    # reaches pi/(2N) at the equator against a ring spacing peaking near 0.89/N, a ratio
    # of about 1.8 -- and a cutoff derived from it would collapse every stencil onto the
    # pixel underneath it. GridS2.max_node_spacing takes the larger of the two
    # directions, so the anisotropy is handled generically and this grid needs no
    # special case.

    # -- spectral bounds -----------------------------------------------------

    @property
    def max_exact_degree(self) -> int:
        raise NotImplementedError(
            "HEALPix has no exact quadrature degree: its equal-area rule integrates a constant exactly and "
            "nothing beyond, so the associated Legendre functions are not discretely orthogonal under it. "
            "Its SHT analysis is approximate; see default_sht_degree and max_recoverable_degree for its bounds."
        )

    @property
    def default_sht_degree(self) -> int:
        r"""
        Default ``lmax`` of an SHT on this grid, :math:`2 N`.

        Non-inclusive. Below it, the analysis is well conditioned: with
        :func:`healpix_ring_weights` a band-limited field is recovered to a relative
        error of about :math:`10^{-2}` without iteration and :math:`10^{-6}` with three
        Jacobi iterations.
        """
        return 2 * self.nside

    @property
    def max_azimuthal_order(self) -> int:
        r"""
        Largest order an SHT on this grid accepts, :math:`3 N`, non-inclusive.

        Orders at or above half a ring's length alias on that ring; the belt rings resolve
        :math:`m < 2N` directly and the polar rings fewer. Orders up to
        :math:`3N` are still determined by the rings jointly, the same bound as
        :attr:`max_recoverable_degree`.
        """
        return 3 * self.nside

    @property
    def max_recoverable_degree(self) -> int:
        r"""
        Largest ``lmax`` a forward SHT accepts on this grid, :math:`3 N`, non-inclusive.

        The usual HEALPix bound :math:`l \le 3N - 1`. Beyond it the
        :math:`12 N^2` pixels no longer determine the coefficients. Accuracy degrades well
        before it: see :attr:`default_sht_degree`.
        """
        return 3 * self.nside

    # -- decomposition -------------------------------------------------------

    @classmethod
    def shard_class(cls) -> type:
        return HealpixGridShardS2

    def shard(self, polar: Tuple[int, int] = (0, 1), azimuth: Tuple[int, int] = (0, 1)) -> "HealpixGridShardS2":
        """
        Return the piece of this grid held by one rank of a latitude decomposition.

        The rings are split into contiguous bands of about equal pixel count, see
        :class:`HealpixGridShardS2`. There is no azimuthal split: rings differ in length
        and carry different phases, so there is no global ``nlon`` to divide.

        Parameters
        ----------
        polar : tuple of int, optional
            ``(rank, size)`` along latitude, by default ``(0, 1)``.
        azimuth : tuple of int, optional
            ``(rank, size)`` along longitude. Only ``size == 1`` is supported; it is accepted so
            that code written for a :class:`~torch_harmonics.grid.RegularGridS2` decomposition,
            which passes both, runs unchanged on a one-dimensional communicator layout.

        Returns
        -------
        HealpixGridShardS2
            The local piece, which knows the global grid it came from.

        Raises
        ------
        NotImplementedError
            If ``azimuth`` splits the rings.
        """
        if tuple(azimuth)[1] != 1:
            raise NotImplementedError(
                f"HEALPix decomposes along latitude only; got an azimuthal split of size {azimuth[1]}. Rings differ in length, so there is no global nlon to divide."
            )
        return HealpixGridShardS2(grid=self, polar_rank=polar[0], polar_size=polar[1])

    # -- construction --------------------------------------------------------

    @classmethod
    def from_level(cls, level: int) -> "HealpixGrid":
        r"""
        Build the grid at a HEALPix refinement level, :math:`N = 2^{\ell}`.

        Parameters
        ----------
        level : int
            Refinement level, at least 0.
        """
        # the same rule as _as_int applies to descriptor fields: any Integral, but not bool
        if isinstance(level, bool) or not isinstance(level, numbers.Integral):
            raise ValueError(f"level must be an integer, got {type(level).__name__}")
        level = int(level)
        if level < 0:
            raise ValueError(f"level must be non-negative, got {level}")
        return cls(nside=1 << level)


@lru_cache(typed=True)
def _ring_cuts(nside: int, size: int) -> Tuple[int, ...]:
    """
    Ring indices at which a ``size``-way latitude split of a HEALPix grid cuts, ``size + 1`` of them.

    Rank ``r`` owns the rings ``cuts[r]:cuts[r + 1]``. Each cut is the ring boundary nearest the
    point where the cumulative pixel count reaches ``r / size`` of the total, so every rank holds
    the same number of pixels up to half a ring, at most ``2 * nside``. Every rank holds at least one ring.
    """
    offsets = HealpixGrid(nside=nside).lon_offsets.tolist()
    nrings = len(offsets) - 1
    cuts = [0]
    for rank in range(1, size):
        target = offsets[-1] * rank / size
        above = bisect_left(offsets, target)
        below = max(above - 1, 0)
        cut = below if target - offsets[below] <= offsets[above] - target else above
        # at least one ring for this rank, and for each of the ranks after it
        cuts.append(min(max(cut, cuts[-1] + 1), nrings - (size - rank)))
    cuts.append(nrings)
    return tuple(cuts)


@dataclass(frozen=True, eq=False)
class HealpixGridShardS2(GridShardS2):
    r"""
    One rank's piece of a :class:`HealpixGrid` under a latitude decomposition.

    The local piece is a contiguous band of rings, which in RING order is a contiguous slice
    of the flat pixel axis, so a local field is ``(..., npoints)``. The bands are cut so that
    every rank holds about the same number of **pixels**, not rings: rings grow from 4 pixels
    at the poles to :math:`4 N` at the equator, and equal ring counts would leave a polar rank
    with a fraction of an equatorial rank's pixels. Polar ranks therefore hold more, shorter rings.

    Parameters
    ----------
    grid : HealpixGrid
        The global grid this is a piece of.
    polar_rank, polar_size : int
        Position and extent of the decomposition along latitude. ``polar_size`` cannot exceed
        the number of rings.
    """

    polar_rank: int = 0
    polar_size: int = 1

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.grid, HealpixGrid):
            raise ValueError(f"grid must be a HealpixGrid, got {type(self.grid).__name__}")
        for field in ("polar_rank", "polar_size"):
            _as_int(self, field)
        if self.polar_size < 1:
            raise ValueError(f"polar_size must be at least 1, got {self.polar_size}")
        if not 0 <= self.polar_rank < self.polar_size:
            raise ValueError(f"polar_rank must lie in [0, {self.polar_size}), got {self.polar_rank}")
        if self.polar_size > self.grid.nrings:
            raise ValueError(f"cannot split {self.grid.nrings} rings over {self.polar_size} ranks")

    def __repr__(self) -> str:
        return f"HealpixGridShardS2({self.grid!r}, polar={self.polar_rank}/{self.polar_size})"

    # -- local extent --------------------------------------------------------

    @property
    def ring_cuts(self) -> Tuple[int, ...]:
        """Ring indices bounding every rank's band, ``polar_size + 1`` of them."""
        return _ring_cuts(self.grid.nside, self.polar_size)

    @property
    def ring_shapes(self) -> Tuple[int, ...]:
        """Ring counts held by every polar rank, ordered by rank."""
        cuts = self.ring_cuts
        return tuple(b - a for a, b in zip(cuts, cuts[1:]))

    @property
    def pixel_shapes(self) -> Tuple[int, ...]:
        """Pixel counts held by every polar rank, ordered by rank; the sizes to split a flat field by."""
        offsets = self.grid.lon_offsets.tolist()
        cuts = self.ring_cuts
        return tuple(offsets[b] - offsets[a] for a, b in zip(cuts, cuts[1:]))

    @property
    def nrings(self) -> int:
        """Number of rings on this rank."""
        return self.ring_shapes[self.polar_rank]

    @property
    def ring_offset(self) -> int:
        """Index of this rank's first ring within the global grid."""
        return self.ring_cuts[self.polar_rank]

    @property
    def npoints(self) -> int:
        """Number of pixels on this rank."""
        return self.pixel_shapes[self.polar_rank]

    @property
    def pixel_offset(self) -> int:
        """Index of this rank's first pixel within the global flat field."""
        return int(self.grid.lon_offsets[self.ring_offset])

    @property
    def shape(self) -> Tuple[int]:
        """Local field shape ``(npoints,)``."""
        return (self.npoints,)

    # -- local geometry ------------------------------------------------------

    @property
    def colats(self) -> torch.Tensor:
        """This rank's slice of the global ring colatitudes, shape ``(nrings,)``."""
        return self.grid.colats[self.ring_offset : self.ring_offset + self.nrings]

    @property
    def lats(self) -> torch.Tensor:
        r"""This rank's slice of the global ring latitudes :math:`\pi/2 - \theta`, shape ``(nrings,)``."""
        return torch.pi / 2 - self.colats

    @property
    def nlon_per_lat(self) -> torch.Tensor:
        """Pixels in each of this rank's rings, shape ``(nrings,)``."""
        return self.grid.nlon_per_lat[self.ring_offset : self.ring_offset + self.nrings]

    @property
    def quad_weights(self) -> torch.Tensor:
        r"""
        This rank's per-pixel solid-angle weights, shape ``(npoints,)``.

        Sums to :math:`4\pi` only across all ranks.
        """
        return _grid_quad_weights(self.grid)[self.pixel_offset : self.pixel_offset + self.npoints]

    def lons(self, ilat: int) -> torch.Tensor:
        """Longitudes of the pixels of this rank's ``ilat``-th ring."""
        return self.grid.lons(self.ring_offset + ilat)
