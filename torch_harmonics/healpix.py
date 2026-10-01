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

The isolatitude property is what makes it expressible here at all. Everything in
this library that is not a spherical harmonic transform -- the DISCO convolutions,
neighborhood attention, the quadrature -- needs only that the points be organized
into rings of constant colatitude, which is exactly what
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
from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

import torch

from torch_harmonics.cache import lru_cache
from torch_harmonics.grid import GridS2, _as_int

__all__ = ["HealpixGrid", "healpix_ring_structure"]


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


@dataclass(frozen=True, eq=False)
class HealpixGrid(GridS2):
    r"""
    HEALPix grid in RING pixel order.

    A ragged :class:`~torch_harmonics.grid.GridS2`: its ``12 * nside**2`` points are
    organized into ``4 * nside - 1`` isolatitude rings of unequal length, so a field
    on it is a flat ``(..., npix)`` tensor rather than a ``(..., nlat, nlon)`` one.
    :attr:`~torch_harmonics.grid.PointSetS2.shape` reports ``(npix,)`` and
    :attr:`~torch_harmonics.grid.GridS2.is_regular` is ``False``. Every pixel has the
    same area, so :attr:`~torch_harmonics.grid.PointSetS2.is_equal_area` is ``True``.

    Parameters
    ----------
    nside : int
        Resolution parameter, at least 1. Rings carry between 4 and ``4 * nside``
        pixels and the sphere is covered by ``12 * nside**2`` of them.

    Notes
    -----
    Not suitable for a spherical harmonic transform. HEALPix quadrature gives every
    pixel the same weight, which integrates a constant exactly and nothing else: the
    rings are neither at the nodes of a latitudinal quadrature rule nor equispaced in
    :math:`\theta`, so the associated Legendre functions are not discretely
    orthogonal under it and ``max_exact_degree`` is undefined. Accordingly
    :attr:`~torch_harmonics.grid.PointSetS2.is_spectrally_accurate` is ``False`` and
    :func:`~torch_harmonics.truncate_sht` will refuse this grid. A transform on
    HEALPix needs iterative or least-squares machinery that this library does not
    have; the localized operators and plain quadrature are what this descriptor is
    for. Today that is :class:`~torch_harmonics.NeighborhoodAttentionS2`,
    :class:`~torch_harmonics.AttentionS2` and :class:`~torch_harmonics.QuadratureS2`;
    the DISCO convolutions do not accept it yet.

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
        r"""Number of rings, :math:`4 N - 1`. The name :class:`~torch_harmonics.grid.GridS2` addresses rings by."""
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
            If ``nside`` is not a power of two, in which case the grid has no
            hierarchical refinement level and cannot be expressed in NEST order.
        """
        if self.nside & (self.nside - 1):
            raise ValueError(f"nside={self.nside} is not a power of two, so it has no HEALPix refinement level")
        return self.nside.bit_length() - 1

    # -- weights -------------------------------------------------------------

    @property
    def is_equal_area(self) -> bool:
        """Every pixel covers :math:`4\\pi / (12 N^2)` -- the defining property of HEALPix."""
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
        Fractional longitude offset of each ring, shape ``(nlat,)``, in units of one
        pixel of that ring.

        Successive HEALPix rings are staggered by half a pixel, which is what makes
        the pixels rhombic and equal-area. A consumer that assumed every ring starts
        at :math:`\lambda = 0` would misplace half the grid.
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

        HEALPix quadrature is the mean: every pixel has solid angle
        :math:`4\pi / N_{pix}`. Expressed in this library's per-ring
        :math:`\cos\theta` convention, where a single point on ring :math:`k` carries
        :math:`2 \pi w_k / n_k`, that makes :math:`w_k` proportional to the ring size,
        and :math:`\sum_k w_k = 2` as on every other grid. The per-point weights, which
        sum to :math:`4\pi`, come from :attr:`~torch_harmonics.grid.PointSetS2.quad_weights`
        and are uniform here because HEALPix is equal-area.
        """
        return 2.0 * self.nlon_per_lat.to(torch.float64) / self.npoints

    def lons(self, ilat: Optional[int] = None) -> torch.Tensor:
        r"""
        Longitudes of one ring, :math:`\lambda_j = \frac{2\pi}{n_k}(j + \delta_k)`.

        Parameters
        ----------
        ilat : int
            Ring index, ``0 <= ilat < nlat``. Required: rings differ in both length
            and phase here, so there is no ring-independent answer to return.

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
        r"""
        Longitude of every pixel, shape ``(npoints,)``, in RING order.

        The flat counterpart of :meth:`lons`, so a consumer can get the full geometry
        without a Python loop over ``nlat`` rings.
        """
        nlon_per_lat, lon_shifts, _ = healpix_ring_structure(self.nside)
        offsets = self.lon_offsets

        # index of each pixel within its own ring
        ring_of = torch.repeat_interleave(torch.arange(self.nrings, dtype=torch.int64), nlon_per_lat)
        j = torch.arange(self.npoints, dtype=torch.int64) - offsets[ring_of]

        n = nlon_per_lat[ring_of].to(torch.float64)
        return (2.0 * torch.pi / n) * (j.to(torch.float64) + lon_shifts[ring_of])

    def all_colats(self) -> torch.Tensor:
        r"""
        Colatitude of every pixel, shape ``(npoints,)``, in RING order.

        Equal to ``coords[:, 0]``; :attr:`~torch_harmonics.grid.PointSetS2.coords`
        carries both columns and is the form the base class contract is written in.
        """
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
            "Use HEALPix for the localized operators and quadrature, and a Gauss or equiangular grid for an SHT."
        )

    # -- decomposition -------------------------------------------------------

    def shard(self, polar: Optional[Tuple[int, int]] = (0, 1), azimuth: Optional[Tuple[int, int]] = (0, 1)) -> "GridS2":
        raise NotImplementedError(
            "HEALPix does not decompose as a product of a latitude range and a longitude range: rings differ "
            "in length, so there is no global nlon to split azimuthally. A ragged decomposition should split "
            "the flat pixel range, which GridShardS2 does not yet describe."
        )

    # -- construction --------------------------------------------------------

    @classmethod
    def from_level(cls, level: int) -> "HealpixGrid":
        r"""
        Build the grid at a HEALPix refinement level, :math:`N = 2^{\ell}`.

        Parameters
        ----------
        level : int
            Refinement level, at least 0. Any integral type is accepted, e.g.
            ``numpy.int64``, as for the other descriptors' integer parameters.
        """
        # the same rule as _as_int applies to descriptor fields: any Integral, but not bool
        if isinstance(level, bool) or not isinstance(level, numbers.Integral):
            raise ValueError(f"level must be an integer, got {type(level).__name__}")
        level = int(level)
        if level < 0:
            raise ValueError(f"level must be non-negative, got {level}")
        return cls(nside=1 << level)
