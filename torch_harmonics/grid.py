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

import ast
import difflib
import functools
import inspect
import numbers
from dataclasses import MISSING, dataclass, fields
from typing import Any, ClassVar, Dict, Optional, Tuple, Type, Union

import torch

from torch_harmonics.cache import lru_cache
from torch_harmonics.partition import compute_split_shapes
from torch_harmonics.quadrature import precompute_latitudes, precompute_longitudes

__all__ = [
    "PointSetS2",
    "GridS2",
    "RegularGridS2",
    "GridShardS2",
    "RegularGridShardS2",
    "EquiangularGrid",
    "LegendreGaussGrid",
    "LobattoGrid",
    "TrapezoidalGrid",
    "as_grid",
    "grid_params",
    "grid_types",
    "require_point_set",
    "require_grid",
    "require_regular_grid",
]

# populated by __init_subclass__; maps the historical grid string to its class.
# Only concrete classes -- those that define their own `grid_type` -- are entered;
# abstract intermediates such as RegularGridS2 are not.
_GRID_REGISTRY: Dict[str, Type["PointSetS2"]] = {}


def _as_int(owner: Any, name: str) -> None:
    r"""
    Validate a descriptor's integer field in place, accepting any integral type.

    ``isinstance(x, int)`` is False for ``numpy.int64``, which rejects the most ordinary
    way to write a resolution sweep::

        for nlat in 2 ** np.arange(3, 8) + 1:      # numpy int64, not int
            grid = as_grid("equiangular", nlat=nlat, nlon=2 * (nlat - 1))

    A numpy integer *is* an integer, so it is accepted -- but normalized to ``int``
    rather than stored as-is. That matters more here than it would elsewhere: these
    fields are the descriptor's :attr:`~PointSetS2.key`, so they end up in ``repr`` and
    in :meth:`~PointSetS2.to_dict`, and a numpy scalar there serializes badly even
    though it hashes and compares equal to the plain int.

    ``bool`` is excluded deliberately: it is an ``Integral``, but ``nlat=True`` is a
    mistake rather than a resolution of 1.
    """
    value = getattr(owner, name)
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    if type(value) is not int:
        # frozen dataclass, so normalize through object.__setattr__
        object.__setattr__(owner, name, int(value))


# The per-point tensors are built once per descriptor and cached on it. Caching matters
# here in a way it does not for the per-ring tensors: these are O(npoints), so a
# 1440x2880 grid rebuilds ~66 MB of coordinates on every access, and building them walks
# the rings in Python on a ragged grid.
#
# `copy=True` matches precompute_latitudes: each caller gets an independent tensor, so an
# in-place write by one consumer cannot poison the entry for the next. That costs a copy
# per access, which is the right trade because these are read at layer construction
# rather than per forward -- and `test_descriptor_returns_independent_tensors` pins the
# no-aliasing half of it.
#
# Free functions rather than methods because the cache must be keyed on the descriptor,
# which is exactly what its `key`/`__hash__` were designed for.


@lru_cache(typed=True, copy=True)
def _grid_coords(grid: "GridS2") -> torch.Tensor:
    """Per-point ``(colat, lon)`` of a ring-structured grid, in ring-major order."""
    counts = grid.nlon_per_lat
    colats = torch.repeat_interleave(grid.colats, counts)
    if isinstance(grid, RegularGridS2):
        # every ring carries the same longitudes, so tile rather than walking the rings.
        # Decided by type: lons() without a ring index is a RegularGridS2 promise, and a
        # GridS2 with equal ring lengths may still stagger them
        lons = grid.lons().repeat(grid.nrings)
    else:
        # ragged, and possibly with a per-ring phase offset (HEALPix staggers by half a
        # pixel), so each ring has to be asked for its own longitudes
        lons = torch.cat([grid.lons(k) for k in range(grid.nrings)])
    return torch.stack([colats, lons.to(colats.dtype)], dim=-1)


@lru_cache(typed=True, copy=True)
def _shard_quad_weights(shard: "RegularGridShardS2") -> torch.Tensor:
    r"""
    Per-point solid-angle weights of one rank's block, in its local ``(nlat, nlon)`` order.

    Note which extent each factor takes. The longitudinal factor is
    :math:`2\pi / N_\lambda` with the **global** ring length, because splitting a ring
    across azimuth ranks divides the points up but does not change how much solid angle
    each one covers. The tiling then uses the **local** count, because that is how many
    of them this rank holds. Using the local length in the factor instead would make every
    rank's weights sum to the global total, and the reduction would then overcount by the
    azimuth group size.

    These sum to :math:`4\pi` only across all ranks; locally they are a partial sum.
    """
    grid = shard.grid
    lo = shard.lat_offset
    counts_global = grid.nlon_per_lat[lo : lo + shard.nlat]
    weights = shard.colat_weights
    per_point = weights * (2.0 * torch.pi / counts_global.to(weights.dtype))
    return torch.repeat_interleave(per_point, shard.nlon)


@lru_cache(typed=True, copy=True)
def _grid_quad_weights(grid: "GridS2") -> torch.Tensor:
    r"""Per-point solid-angle weights of a ring-structured grid, summing to :math:`4\pi`."""
    counts = grid.nlon_per_lat
    weights = grid.colat_weights
    per_point = weights * (2.0 * torch.pi / counts.to(weights.dtype))
    return torch.repeat_interleave(per_point, counts)


@lru_cache(typed=True, copy=True)
def _grid_ring_weights(grid: "GridS2", dtype: torch.dtype) -> torch.Tensor:
    r"""Quadrature weight carried by a single point of each ring, shape ``(nrings,)``."""
    # every operand cast before the arithmetic, not after: a consumer working in fp32
    # that let this be computed in fp64 and rounded at the end would land about one ulp
    # away, which is a visible shift in the weights of an already-trained model
    weights = grid.colat_weights.to(dtype)
    counts = grid.nlon_per_lat.to(dtype)
    return 2.0 * torch.pi * weights / counts


@lru_cache(typed=True, copy=True)
def _grid_point_weights(grid: "GridS2", dtype: torch.dtype) -> torch.Tensor:
    r"""The same quadrature, one entry per point, shape ``(npoints,)``."""
    return torch.repeat_interleave(_grid_ring_weights(grid, dtype), grid.nlon_per_lat)


@dataclass(frozen=True, eq=False)
class PointSetS2:
    r"""
    Descriptor for a set of sample points on :math:`S^2`.

    The base of the grid hierarchy: :attr:`npoints` locations on the sphere, each
    with an angular position and a quadrature weight, with no assumption about how
    they are arranged. Each subclass adds a constraint:

    * :class:`PointSetS2` -- points and weights; enough to integrate.
    * :class:`GridS2` -- adds isolatitude rings with equispaced longitudes; enables
      an FFT in longitude (hence a fast SHT) and a contiguous polar decomposition.
    * :class:`RegularGridS2` -- adds the same longitude count on every ring; a field
      is a dense ``(nlat, nlon)`` array.

    ======================  ================================================================  =================================================
    level                   arrangement                                                       examples
    ======================  ================================================================  =================================================
    :class:`PointSetS2`     any points with weights                                           ICON (icosahedral), cubed-sphere
    :class:`GridS2`         latitude rings, equispaced in longitude; ring lengths may differ  HEALPix, reduced (octahedral) Gaussian
    :class:`RegularGridS2`  latitude rings of equal length                                    equiangular (ERA5's 721 x 1440), regular Gaussian
    ======================  ================================================================  =================================================

    The regular latitude-longitude grids and HEALPix are implemented; the other
    examples only mark where the boundaries lie.

    This class is abstract, as are :class:`GridS2` and :class:`RegularGridS2`.
    Instantiate a concrete grid, or build one by name with :func:`as_grid`. Each
    concrete grid has a class-level ``grid_type`` string, e.g. ``"equiangular"``,
    used by :func:`as_grid` and for serialization. A subclass that does not declare
    its own ``grid_type`` is treated as abstract: it is neither registered nor
    instantiable.

    Descriptors are immutable and hashable; equality and hashing are defined by
    the grid type and its constructor parameters.
    """

    # Maintainer notes: the dataclass fields *are* the parameterization -- params, key,
    # to_dict and __repr__ derive from them, so a subclass cannot forget to extend its
    # identity (which would silently collide in descriptor-keyed caches). Node and weight
    # tensors are deliberately not fields: tensors would fall back to identity hashing.
    # A family whose nodes are data (e.g. read from an ICON file) needs a non-data
    # identity such as a registered name or a path plus content hash.

    #: historical grid string; set by each concrete subclass, absent on abstract ones
    grid_type: ClassVar[str]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # only a class that declares its own grid_type is concrete; an abstract
        # intermediate such as GridS2 or RegularGridS2 merely inherits the annotation
        grid_type = cls.__dict__.get("grid_type")
        if grid_type is None:
            # A subclass that declares nothing inherits its parent's grid_type, and with
            # it the parent's key, hash and equality -- so it would collide with the
            # parent in every descriptor-keyed cache and be handed the parent's
            # precomputed geometry. Refuse rather than let a changed grid silently reuse
            # tables built for a different one. Deriving from an abstract intermediate is
            # unaffected; those carry no grid_type to inherit.
            inherited = next((base.__dict__["grid_type"] for base in cls.__mro__[1:] if "grid_type" in base.__dict__), None)
            if inherited is not None:
                raise TypeError(
                    f"{cls.__name__} subclasses a concrete grid without declaring its own grid_type, so it would "
                    f"inherit '{inherited}' and share that grid's identity. Declare a distinct grid_type, or derive "
                    f"from an abstract base such as RegularGridS2 instead."
                )
            return
        if grid_type in _GRID_REGISTRY:
            raise ValueError(f"grid_type '{grid_type}' is already registered to {_GRID_REGISTRY[grid_type].__name__}")
        _GRID_REGISTRY[grid_type] = cls

    def __post_init__(self):
        if not hasattr(type(self), "grid_type"):
            raise TypeError(f"{type(self).__name__} is abstract; instantiate a concrete grid or use as_grid()")

    # -- identity ------------------------------------------------------------

    @classmethod
    def params(cls) -> Tuple[str, ...]:
        """Names of this descriptor's constructor parameters, in declaration order."""
        return tuple(f.name for f in fields(cls))

    @property
    def key(self) -> Tuple[Any, ...]:
        """
        Canonical, hashable identity of this descriptor.

        A tuple of scalars, the grid type followed by the constructor parameters;
        it defines both equality and hashing.
        """
        return (self.grid_type,) + tuple(getattr(self, name) for name in self.params())

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PointSetS2):
            return NotImplemented
        return self.key == other.key

    def __repr__(self) -> str:
        args = ", ".join(f"{name}={getattr(self, name)!r}" for name in self.params())
        return f"{type(self).__name__}({args})"

    # -- extent --------------------------------------------------------------

    @property
    def npoints(self) -> int:
        """Total number of sample points."""
        raise NotImplementedError(f"{type(self).__name__} does not define npoints")

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Trailing shape of a tensor holding a field sampled on this point set.

        ``(npoints,)`` in general; ``(nlat, nlon)`` on a :class:`RegularGridS2`. To
        stay grid-agnostic, use ``x.reshape(*batch, *grid.shape)`` or
        ``len(grid.shape)`` rather than unpacking it as ``nlat, nlon = grid.shape``.
        """
        return (self.npoints,)

    # -- geometry ------------------------------------------------------------

    @property
    def coords(self) -> torch.Tensor:
        r"""
        Angular position of every point, shape ``(npoints, 2)``.

        Column 0 is colatitude :math:`\theta \in [0, \pi]` measured from the north
        pole, column 1 is longitude :math:`\lambda \in [0, 2\pi)`.

        Row ``i`` describes element ``i`` of a field flattened to ``(npoints,)``.
        For a :class:`GridS2` the order is ring-major: point ``(ilat, ilon)`` sits
        at ``lon_offsets[ilat] + ilon``. Pole-inclusive grids repeat the pole once
        per longitude.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define coords")

    @property
    def quad_weights(self) -> torch.Tensor:
        r"""
        Quadrature weight of every point, shape ``(npoints,)``, summing to :math:`4\pi`.

        The full solid-angle weight, so that ``(f * grid.quad_weights).sum(-1)``
        approximates :math:`\int_{S^2} f \, dA` for a field flattened to
        ``(npoints,)``.

        .. warning::
            Not to be confused with :attr:`GridS2.colat_weights`, the latitudinal
            factor alone (shape ``(nrings,)``, summing to 2). Using one in place of
            the other silently scales an integral by :math:`2\pi`.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define quad_weights")

    @property
    def is_equal_area(self) -> bool:
        """
        Whether every point carries the same quadrature weight, :math:`4\\pi / N`.

        Declared by the grid family (``True`` for HEALPix, ``False`` for every
        latitude-longitude grid). Consumers may use it to drop weights that only enter
        through a normalization, e.g. in the attention softmax.
        """
        return False

    # -- spectral bounds -----------------------------------------------------
    #
    # These are facts about what the sampling can represent, not decisions about
    # what an SHT should keep. The policy -- applying user overrides, enforcing
    # triangular truncation, warning about changed defaults -- lives in
    # :mod:`torch_harmonics.truncation`, so these properties stay silent.

    @property
    def max_exact_degree(self) -> int:
        r"""
        Highest spherical harmonic degree the quadrature rule integrates exactly.

        Non-inclusive, i.e. degrees :math:`0 \le l < l_{\max}`. Raises on grids
        without such a bound, e.g. HEALPix.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define max_exact_degree")

    @property
    def is_spectrally_accurate(self) -> bool:
        r"""
        Whether the quadrature rule converges spectrally.

        An accurate SHT needs the associated Legendre polynomials to be discretely
        orthogonal under the rule, which interpolatory rules (Gauss--Legendre,
        Gauss--Lobatto, Clenshaw--Curtis) provide. ``False`` unless a grid family
        declares otherwise.
        """
        return False

    @property
    def max_node_spacing(self) -> float:
        r"""
        Largest great-circle distance between neighbouring nodes, in radians.

        The grid's resolution expressed as an angle.
        :func:`torch_harmonics.truncate_support` derives the default support radius
        of localized operators from it.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define max_node_spacing")

    # -- decomposition -------------------------------------------------------

    @classmethod
    def shard_class(cls) -> Type["GridShardS2"]:
        """The :class:`GridShardS2` subclass that :meth:`shard` produces."""
        raise NotImplementedError(f"{cls.__name__} does not define shard_class")

    def shard(self, **decomposition: Any) -> "GridShardS2":
        """
        Return the piece of this descriptor held by one rank of a decomposition.

        The decomposition parameters depend on the grid family and are plain
        integers, not process groups; see :meth:`RegularGridS2.shard`.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define shard")

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Plain-data representation, suitable for a config file or a checkpoint.

        The grid type under ``"grid"`` plus the constructor parameters, e.g.
        ``{"grid": "healpix", "nside": 64}``.
        """
        data = {"grid": self.grid_type}
        data.update({name: getattr(self, name) for name in self.params()})
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PointSetS2":
        """Inverse of :meth:`to_dict`."""
        if "grid" not in data:
            raise ValueError("grid dict is missing 'grid'")
        return as_grid(data["grid"], **{k: v for k, v in data.items() if k != "grid"})


@dataclass(frozen=True, eq=False)
class GridS2(PointSetS2):
    r"""
    A point set organized into isolatitude rings.

    Each ring sits at a colatitude :math:`\theta_k` and carries longitudes
    equispaced around the circle. The number of longitudes may differ from ring to
    ring, as on HEALPix or a reduced Gaussian grid. A field on such a grid is stored
    flat, ring after ring (point ``(ilat, ilon)`` at flat index
    ``lon_offsets[ilat] + ilon``), with shape ``(npoints,)``:

    >>> from torch_harmonics import HealpixGrid
    >>> grid = HealpixGrid(nside=4)
    >>> grid.nrings, grid.npoints, grid.shape
    (15, 192, (192,))
    >>> grid.nlon_per_lat[:4].tolist()  # ring lengths grow away from the pole
    [4, 8, 12, 16]

    The quadrature weight of a point on ring :math:`k` factorizes as
    :math:`w_k \cdot 2\pi / N_{\lambda,k}`, with :math:`w_k` given by
    :attr:`colat_weights`. Abstract; the node distribution is chosen by the concrete
    subclasses.
    """

    # -- extent --------------------------------------------------------------

    @property
    def nrings(self) -> int:
        """Number of latitude rings."""
        raise NotImplementedError(f"{type(self).__name__} does not define nrings")

    @property
    def npoints(self) -> int:
        return int(self.nlon_per_lat.sum())

    # -- geometry, per ring --------------------------------------------------

    @property
    def colats(self) -> torch.Tensor:
        r"""
        Colatitudes :math:`\theta_k \in [0, \pi]`, ascending (north pole first), shape ``(nrings,)``.

        One value per ring; :attr:`coords` is the per-point form and :attr:`lats`
        the geographic latitude.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define colats")

    @property
    def lats(self) -> torch.Tensor:
        r"""
        Geographic latitudes :math:`\phi_k = \pi/2 - \theta_k \in [-\pi/2, \pi/2]`, shape ``(nrings,)``.

        Descending, north pole first, in the same order as :attr:`colats`.
        """
        return torch.pi / 2 - self.colats

    @property
    def colat_weights(self) -> torch.Tensor:
        r"""
        Latitudinal quadrature weights, shape ``(nrings,)``, paired with :attr:`colats`.

        Formulated in the :math:`\cos\theta` domain, so they absorb the
        :math:`\sin\theta` Jacobian and **sum to 2**. The longitudinal factor
        :math:`2\pi / N_{\lambda,k}` is not included; see
        :attr:`~PointSetS2.quad_weights` for the per-point weights.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define colat_weights")

    def lons(self, ilat: Optional[int] = None) -> torch.Tensor:
        r"""
        Longitudes :math:`\lambda_j \in [0, 2\pi)` of a latitude ring.

        Parameters
        ----------
        ilat : int, optional
            Index of the latitude ring. Ignored on regular grids, where every ring
            carries the same longitudes.

        Returns
        -------
        torch.Tensor
            Longitudes in radians, shape ``(nlon_per_lat[ilat],)``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define lons")

    # -- geometry, per point -------------------------------------------------

    @property
    def coords(self) -> torch.Tensor:
        """Per-point positions, built from the ring structure in ring-major order."""
        return _grid_coords(self)

    @property
    def quad_weights(self) -> torch.Tensor:
        r"""
        Per-point solid-angle weights, summing to :math:`4\pi`.

        A point on ring :math:`k` carries :math:`w_k \cdot 2\pi / N_{\lambda,k}`.
        """
        return _grid_quad_weights(self)

    def ring_weights(self, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        r"""
        Quadrature weight carried by a single point of each ring, shape ``(nrings,)``.

        Ring :math:`k` carries :math:`w_k \cdot 2\pi / N_{\lambda,k}` at each of its
        points; this is the compact, per-ring form of :attr:`quad_weights`.

        Parameters
        ----------
        dtype : torch.dtype, optional
            Dtype to compute in, by default ``torch.float64``. The arithmetic is done
            in this dtype, which can differ by an ulp from casting a float64 result.

        Returns
        -------
        torch.Tensor
            Per-ring weights of shape ``(nrings,)``.
        """
        return _grid_ring_weights(self, dtype)

    def point_weights(self, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        r"""
        The same quadrature as :meth:`ring_weights`, one entry per point.

        Equal to :attr:`quad_weights` up to rounding, and computed in ``dtype`` so
        that it is exactly consistent with :meth:`ring_weights`.

        Parameters
        ----------
        dtype : torch.dtype, optional
            Dtype to compute in, by default ``torch.float64``.

        Returns
        -------
        torch.Tensor
            Per-point weights of shape ``(npoints,)``, summing to :math:`4\pi`.
        """
        return _grid_point_weights(self, dtype)

    # -- raggedness ----------------------------------------------------------

    @property
    def nlon_per_lat(self) -> torch.Tensor:
        """Number of longitudes on each latitude ring, shape ``(nrings,)``."""
        raise NotImplementedError(f"{type(self).__name__} does not define nlon_per_lat")

    @property
    def lon_shifts(self) -> torch.Tensor:
        r"""
        Fractional longitude offset of each ring, shape ``(nrings,)``, in units of one
        point of that ring.

        Zero on the latitude-longitude grids. On HEALPix it is either 0 or 1/2 (see
        :attr:`~torch_harmonics.healpix.HealpixGrid.lon_shifts`). :meth:`lons` already
        includes the shift.
        """
        return torch.zeros(self.nrings, dtype=torch.float64)

    @property
    def lon_offsets(self) -> torch.Tensor:
        """
        Exclusive prefix sum of :attr:`nlon_per_lat`, shape ``(nrings + 1,)``.

        A point ``(ilat, ilon)`` sits at flat index ``lon_offsets[ilat] + ilon``.
        """
        counts = self.nlon_per_lat
        return torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])

    @property
    def is_regular(self) -> bool:
        """
        Whether every latitude ring carries the same number of longitudes.

        Code that needs a dense ``(nlat, nlon)`` layout should call
        :func:`require_regular_grid` rather than test this or assume a uniform stride.
        """
        counts = self.nlon_per_lat
        return bool((counts == counts[0]).all())

    # -- derived quantities --------------------------------------------------

    @property
    def latitude_spacing(self) -> torch.Tensor:
        r"""
        Gaps :math:`\theta_{k+1} - \theta_k` between adjacent latitudes, shape ``(nrings - 1,)``.
        """
        colats = self.colats
        return colats[1:] - colats[:-1]

    @property
    def max_latitude_spacing(self) -> float:
        r"""
        Largest gap between adjacent latitudes, :math:`\max_k (\theta_{k+1} - \theta_k)`.

        On :class:`EquiangularGrid` this is :math:`\pi / (N_\theta - 1)`.
        """
        return self.latitude_spacing.max().item()

    @property
    def is_uniform_in_theta(self) -> bool:
        r"""Whether the latitude nodes are equispaced in :math:`\theta`."""
        return False

    @property
    def max_longitude_spacing(self) -> float:
        r"""
        Largest great-circle distance between adjacent nodes *within* a ring, in radians.

        The great-circle arc rather than the coordinate gap :math:`2\pi / N_{\lambda,k}`,
        so polar rings count as narrow:

        .. math::

            d_k = 2 \arcsin\!\left( \sin\theta_k \, \sin\frac{\pi}{N_{\lambda,k}} \right)
        """
        colats = self.colats
        dlambda = 2.0 * torch.pi / self.nlon_per_lat.to(colats.dtype)
        return float((2.0 * torch.asin(torch.sin(colats) * torch.sin(0.5 * dlambda))).max())

    @property
    def max_node_spacing(self) -> float:
        r"""
        Largest great-circle distance between neighbouring nodes, in radians.

        The larger of :attr:`max_latitude_spacing` and :attr:`max_longitude_spacing`.
        The latitudinal spacing dominates on an equiangular grid with
        ``nlon = 2 * nlat``; the longitudinal one dominates on a Gauss grid at the same
        resolution, on any grid with ``nlon < 2 * nlat``, and on HEALPix.
        """
        return max(self.max_latitude_spacing, self.max_longitude_spacing)

    # :meth:`lat_shapes` and :meth:`lon_shapes` are deliberately *not* defined here,
    # even though splitting `nrings` into contiguous chunks is well defined for any
    # ring-structured grid. Balancing rings only balances work where the rings are of
    # equal length, which is what makes it right on a RegularGridS2 and wrong on a
    # ragged one: HEALPix ring lengths vary by a factor of 4N between the poles and
    # the equator, so an even split of rings hands ranks badly uneven point counts.
    # A ragged grid has to balance the flat point range instead. Inheriting a
    # plausible-but-unbalanced default would not raise, it would just run slowly and
    # asymmetrically, so the method lives on the class where it is correct.


@dataclass(frozen=True, eq=False)
class RegularGridS2(GridS2):
    r"""
    A grid whose latitude rings all carry the same number of longitudes.

    *Regular* in the sense of numerical weather prediction and GRIB
    (``regular_ll``, ``regular_gg``), as opposed to a *reduced* grid whose rings
    shrink toward the poles. The sampling is a tensor product of a latitudinal rule
    and equispaced longitudes, so a field on it is a dense ``(nlat, nlon)`` array.

    >>> from torch_harmonics import as_grid
    >>> grid = as_grid("legendre-gauss", nlat=64, nlon=128)  # a regular Gaussian grid
    >>> grid.shape, grid.npoints
    ((64, 128), 8192)
    >>> bool((grid.nlon_per_lat == 128).all())
    True

    Abstract: the latitudinal rule is chosen by the concrete subclasses
    (:class:`EquiangularGrid`, :class:`LegendreGaussGrid`, :class:`LobattoGrid`,
    :class:`TrapezoidalGrid`).

    Parameters
    ----------
    nlat : int
        Number of latitudinal nodes. Must be at least 2.
    nlon : int
        Number of longitudinal nodes. Must be at least 1.
    """

    nlat: int
    nlon: int

    def __post_init__(self):
        super().__post_init__()
        _as_int(self, "nlat")
        _as_int(self, "nlon")
        if self.nlat < 2:
            raise ValueError(f"nlat must be at least 2, got {self.nlat}")
        if self.nlon < 1:
            raise ValueError(f"nlon must be at least 1, got {self.nlon}")

    # -- extent --------------------------------------------------------------

    @property
    def nrings(self) -> int:
        return self.nlat

    @property
    def shape(self) -> Tuple[int, int]:
        """Spatial shape ``(nlat, nlon)`` of a field sampled on this grid."""
        return (self.nlat, self.nlon)

    @property
    def npoints(self) -> int:
        return self.nlat * self.nlon

    # -- geometry ------------------------------------------------------------

    @property
    def colats(self) -> torch.Tensor:
        colats, _ = precompute_latitudes(self.nlat, grid=self.grid_type)
        return colats

    @property
    def colat_weights(self) -> torch.Tensor:
        _, w = precompute_latitudes(self.nlat, grid=self.grid_type)
        return w

    def lons(self, ilat: Optional[int] = None) -> torch.Tensor:
        return precompute_longitudes(self.nlon)

    # -- raggedness ----------------------------------------------------------

    @property
    def is_regular(self) -> bool:
        return True

    @property
    def nlon_per_lat(self) -> torch.Tensor:
        return torch.full((self.nlat,), self.nlon, dtype=torch.int64)

    @property
    def lon_offsets(self) -> torch.Tensor:
        return torch.arange(self.nlat + 1, dtype=torch.int64) * self.nlon

    # -- spectral bounds -----------------------------------------------------

    @property
    def is_spectrally_accurate(self) -> bool:
        """``True``: the latitudinal rules of this family are interpolatory."""
        return True

    @property
    def max_azimuthal_order(self) -> int:
        r"""
        Nyquist limit of the longitudinal sampling, :math:`\lfloor N_\lambda / 2 \rfloor + 1`.

        Non-inclusive.
        """
        return self.nlon // 2 + 1

    # -- decomposition -------------------------------------------------------

    @classmethod
    def shard_class(cls) -> Type["GridShardS2"]:
        return RegularGridShardS2

    def shard(self, polar: Optional[Tuple[int, int]] = (0, 1), azimuth: Optional[Tuple[int, int]] = (0, 1)) -> "RegularGridShardS2":
        """
        Return the piece of this grid held by one rank of a 2D decomposition.

        Parameters
        ----------
        polar : tuple of int, optional
            ``(rank, size)`` along the polar (latitude) direction, by default ``(0, 1)``.
        azimuth : tuple of int, optional
            ``(rank, size)`` along the azimuthal (longitude) direction, by default ``(0, 1)``.

        Returns
        -------
        RegularGridShardS2
            The local piece, which knows the global grid it came from.
        """
        return RegularGridShardS2(grid=self, polar_rank=polar[0], polar_size=polar[1], azimuth_rank=azimuth[0], azimuth_size=azimuth[1])

    def lat_shapes(self, num_chunks: int) -> Tuple[int, ...]:
        """Latitude counts held by each rank of a ``num_chunks``-way polar split."""
        return tuple(compute_split_shapes(self.nlat, num_chunks))

    def lon_shapes(self, num_chunks: int) -> Tuple[int, ...]:
        """Longitude counts held by each rank of a ``num_chunks``-way azimuthal split."""
        return tuple(compute_split_shapes(self.nlon, num_chunks))


@dataclass(frozen=True, eq=False)
class GridShardS2:
    r"""
    One rank's piece of a decomposed :class:`GridS2`.

    A shard is not a :class:`GridS2`, since it does not cover the sphere: its
    quadrature weights are partial sums that a collective completes, and global
    quantities such as the spectral bounds or the support radius are not defined on
    it -- use :attr:`global_grid` for those. Abstract; see
    :class:`RegularGridShardS2`.

    Parameters
    ----------
    grid : GridS2
        The global grid this is a piece of.
    """

    grid: GridS2

    def __post_init__(self):
        if type(self) is GridShardS2:
            raise TypeError("GridShardS2 is abstract; obtain one from GridS2.shard()")
        if not isinstance(self.grid, GridS2):
            raise ValueError(f"grid must be a GridS2, got {type(self.grid).__name__}")

    # -- identity ------------------------------------------------------------

    @classmethod
    def params(cls) -> Tuple[str, ...]:
        """Names of this shard's constructor parameters, in declaration order."""
        return tuple(f.name for f in fields(cls))

    @property
    def key(self) -> Tuple[Any, ...]:
        """Canonical identity, including the global grid's own key."""
        return tuple(self.grid.key if name == "grid" else getattr(self, name) for name in self.params())

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GridShardS2):
            return NotImplemented
        return type(self) is type(other) and self.key == other.key

    def __repr__(self) -> str:
        args = ", ".join(f"{name}={getattr(self, name)!r}" for name in self.params() if name != "grid")
        return f"{type(self).__name__}({self.grid!r}, {args})"

    # -- the global grid this came from --------------------------------------

    @property
    def global_grid(self) -> GridS2:
        """The undecomposed grid. Pass this wherever a global quantity is needed."""
        return self.grid

    @property
    def is_global(self) -> bool:
        """``False``; see :attr:`global_grid`."""
        return False

    @property
    def is_regular(self) -> bool:
        """Whether every local latitude ring carries the same number of longitudes."""
        return self.grid.is_regular

    @property
    def quad_weights(self) -> torch.Tensor:
        r"""
        Per-point solid-angle weights of this rank's block.

        These sum to :math:`4\pi` only across all ranks.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define quad_weights")

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Plain-data representation, carrying the global grid with it."""
        return {name: self.grid.to_dict() if name == "grid" else getattr(self, name) for name in self.params()}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "GridShardS2":
        """Inverse of :meth:`to_dict`."""
        if "grid" not in data:
            raise ValueError("grid shard dict is missing 'grid'")
        grid = GridS2.from_dict(data["grid"])
        cls = type(grid).shard_class()
        missing = set(cls.params()) - set(data)
        if missing:
            raise ValueError(f"grid shard dict is missing {sorted(missing)}")
        return cls(grid=grid, **{k: v for k, v in data.items() if k != "grid"})


@dataclass(frozen=True, eq=False)
class RegularGridShardS2(GridShardS2):
    r"""
    One rank's piece of a :class:`RegularGridS2` under a 2D decomposition.

    The local piece is a contiguous latitude range times a contiguous longitude
    range.

    Parameters
    ----------
    grid : RegularGridS2
        The global grid this is a piece of.
    polar_rank, polar_size : int
        Position and extent of the decomposition along latitude.
    azimuth_rank, azimuth_size : int
        Position and extent of the decomposition along longitude.
    """

    polar_rank: int = 0
    polar_size: int = 1
    azimuth_rank: int = 0
    azimuth_size: int = 1

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.grid, RegularGridS2):
            raise ValueError(f"grid must be a RegularGridS2, got {type(self.grid).__name__}")
        for field in ("polar_rank", "polar_size", "azimuth_rank", "azimuth_size"):
            _as_int(self, field)
        for rank, size, name in [(self.polar_rank, self.polar_size, "polar"), (self.azimuth_rank, self.azimuth_size, "azimuth")]:
            if size < 1:
                raise ValueError(f"{name}_size must be at least 1, got {size}")
            if not 0 <= rank < size:
                raise ValueError(f"{name}_rank must lie in [0, {size}), got {rank}")

    def __repr__(self) -> str:
        return f"RegularGridShardS2({self.grid!r}, polar={self.polar_rank}/{self.polar_size}, azimuth={self.azimuth_rank}/{self.azimuth_size})"

    # -- local extent --------------------------------------------------------

    @property
    def lat_shapes(self) -> Tuple[int, ...]:
        """Latitude counts held by every polar rank, ordered by rank."""
        return self.grid.lat_shapes(self.polar_size)

    @property
    def lon_shapes(self) -> Tuple[int, ...]:
        """Longitude counts held by every azimuthal rank, ordered by rank."""
        return self.grid.lon_shapes(self.azimuth_size)

    @property
    def nlat(self) -> int:
        """Number of latitudes on this rank."""
        return self.lat_shapes[self.polar_rank]

    @property
    def nlon(self) -> int:
        """Number of longitudes on this rank."""
        return self.lon_shapes[self.azimuth_rank]

    @property
    def lat_offset(self) -> int:
        """Index of this rank's first latitude within the global grid."""
        return sum(self.lat_shapes[: self.polar_rank])

    @property
    def lon_offset(self) -> int:
        """Index of this rank's first longitude within the global grid."""
        return sum(self.lon_shapes[: self.azimuth_rank])

    @property
    def shape(self) -> Tuple[int, int]:
        """Local spatial shape ``(nlat, nlon)``."""
        return (self.nlat, self.nlon)

    @property
    def npoints(self) -> int:
        """Number of grid points on this rank."""
        return self.nlat * self.nlon

    # -- local geometry ------------------------------------------------------

    @property
    def colats(self) -> torch.Tensor:
        """This rank's slice of the global colatitudes, shape ``(nlat,)``."""
        return self.grid.colats[self.lat_offset : self.lat_offset + self.nlat]

    @property
    def lats(self) -> torch.Tensor:
        r"""This rank's slice of the global latitudes :math:`\pi/2 - \theta`, shape ``(nlat,)``."""
        return torch.pi / 2 - self.colats

    @property
    def colat_weights(self) -> torch.Tensor:
        """
        This rank's slice of the global latitudinal weights, shape ``(nlat,)``.

        These sum to 2 only across all polar ranks; locally they are a partial sum.
        """
        return self.grid.colat_weights[self.lat_offset : self.lat_offset + self.nlat]

    @property
    def quad_weights(self) -> torch.Tensor:
        r"""
        This rank's per-point solid-angle weights, shape ``(nlat * nlon,)``.

        Sums to :math:`4\pi` only across all ranks.
        """
        return _shard_quad_weights(self)

    def lons(self, ilat: Optional[int] = None) -> torch.Tensor:
        """This rank's slice of the longitudes of a latitude ring."""
        return self.grid.lons(ilat)[self.lon_offset : self.lon_offset + self.nlon]


@dataclass(frozen=True, eq=False)
class EquiangularGrid(RegularGridS2):
    r"""
    Equiangular grid with Clenshaw--Curtis quadrature.

    Nodes are equally spaced in :math:`\theta` and include both poles, with spacing
    :math:`\pi / (N_\theta - 1)`.
    """

    grid_type: ClassVar[str] = "equiangular"

    @property
    def is_uniform_in_theta(self) -> bool:
        return True

    @property
    def max_exact_degree(self) -> int:
        r"""Clenshaw--Curtis is exact to roughly degree :math:`N_\theta - 1`, giving :math:`\lfloor (N_\theta + 1) / 2 \rfloor`."""
        return (self.nlat + 1) // 2


@dataclass(frozen=True, eq=False)
class LegendreGaussGrid(RegularGridS2):
    r"""
    Gauss--Legendre grid; nodes are the roots of :math:`P_N(\cos\theta)`.

    Exact for polynomials up to degree :math:`2N - 1`; the nodes exclude the poles
    and are not uniform in :math:`\theta`.
    """

    grid_type: ClassVar[str] = "legendre-gauss"

    @property
    def max_exact_degree(self) -> int:
        r"""Gauss--Legendre is exact to degree :math:`2N_\theta - 1`, giving :math:`N_\theta`."""
        return self.nlat


@dataclass(frozen=True, eq=False)
class LobattoGrid(RegularGridS2):
    r"""
    Gauss--Lobatto grid; nodes are the roots of :math:`P'_{N-1}(\cos\theta)` plus both poles.

    Exact for polynomials up to degree :math:`2N - 3`. Nodes cluster towards the
    equator, so the polar spacing is coarser than :math:`\pi / (N_\theta - 1)`.
    """

    grid_type: ClassVar[str] = "lobatto"

    @property
    def max_exact_degree(self) -> int:
        r"""Gauss--Lobatto is exact to degree :math:`2N_\theta - 3`, giving :math:`N_\theta - 1`."""
        return self.nlat - 1


@dataclass(frozen=True, eq=False)
class TrapezoidalGrid(RegularGridS2):
    r"""
    Trapezoidal rule applied on the :math:`\cos\theta` interval :math:`[-1, 1]`.

    The nodes are equispaced in :math:`\cos\theta`, **not** in :math:`\theta`, so
    the polar spacing is about :math:`\sqrt{N_\theta - 1}` times coarser than the
    equatorial one. Formerly named ``"equiangular-trapezoidal"``; that name is no
    longer accepted.
    """

    grid_type: ClassVar[str] = "trapezoidal"

    @property
    def max_exact_degree(self) -> int:
        r"""
        Matches the equiangular grid, :math:`\lfloor (N_\theta + 1) / 2 \rfloor`.

        Kept for backwards compatibility; the value is optimistic, see
        :attr:`is_spectrally_accurate`.
        """
        return (self.nlat + 1) // 2

    @property
    def is_spectrally_accurate(self) -> bool:
        r"""
        ``False``. The trapezoidal rule converges only algebraically, as :math:`O(h^2)`.

        An SHT on this grid is therefore accurate only at very low truncation, far
        below the default ``lmax`` that :func:`~torch_harmonics.truncate_sht` assigns;
        pass a small ``lmax`` explicitly. The grid remains suitable for quadrature and
        for the localized operators.
        """
        return False


def require_point_set(grid: Any, name: Optional[str] = "grid") -> PointSetS2:
    """
    Validate that a routine received a descriptor of *some* sampling of the sphere.

    For routines that need only points and weights, such as integration.

    Parameters
    ----------
    grid : Any
        The value supplied by the caller.
    name : str, optional
        Name of the parameter, used in the error message, by default ``"grid"``.

    Returns
    -------
    PointSetS2
        ``grid`` unchanged, once validated.

    Raises
    ------
    TypeError
        If ``grid`` is not a :class:`PointSetS2`.
    """
    if isinstance(grid, PointSetS2):
        return grid
    _raise_not_a_descriptor(grid, name)


def require_grid(grid: Any, name: Optional[str] = "grid") -> GridS2:
    """
    Validate that a routine received a ring-structured grid.

    For routines that need the points organized into isolatitude rings. A grid
    name or ``(nlat, nlon)`` shape is rejected with a message showing how to build
    the descriptor.

    Parameters
    ----------
    grid : Any
        The value supplied by the caller.
    name : str, optional
        Name of the parameter, used in the error message, by default ``"grid"``.

    Returns
    -------
    GridS2
        ``grid`` unchanged, once validated.

    Raises
    ------
    TypeError
        If ``grid`` is not a :class:`GridS2`.
    """
    if isinstance(grid, GridS2):
        return grid
    if isinstance(grid, PointSetS2):
        raise TypeError(f"{name} must be a GridS2; this routine needs the points organized into isolatitude rings, which " f"{type(grid).__name__} does not provide. Got {grid!r}.")
    _raise_not_a_descriptor(grid, name)


def _raise_not_a_descriptor(grid: Any, name: str) -> None:
    """Shared migration-friendly rejection, so the three guards word it identically."""
    if isinstance(grid, GridShardS2):
        raise TypeError(
            f"{name} must be the global descriptor, not a shard of one. Quantities such as the spectral bounds and the angular cutoff are global; " f"pass {name}.global_grid."
        )
    if isinstance(grid, str):
        # name the parameters this grid type actually takes: HEALPix takes nside, not (nlat, nlon)
        try:
            params = ", ".join(f"{p}=..." for p in grid_params(grid))
        except ValueError:
            params = "..."
        raise TypeError(f"{name} must be a grid descriptor, not the grid name {grid!r}. The descriptor carries the resolution too, so build one with as_grid({grid!r}, {params}).")
    if isinstance(grid, (tuple, list)) and len(grid) == 2:
        nlat, nlon = grid
        raise TypeError(
            f"{name} must be a grid descriptor, not a shape {tuple(grid)!r}. The descriptor carries the shape, so pass "
            f"as_grid(<grid name>, nlat={nlat!r}, nlon={nlon!r}) instead."
        )
    raise TypeError(f"{name} must be a grid descriptor, got {type(grid).__name__}. Build one with as_grid(<grid name>, nlat=..., nlon=...).")


def require_regular_grid(grid: Any, name: Optional[str] = "grid") -> RegularGridS2:
    """
    Validate that a routine received a :class:`RegularGridS2`.

    For routines that address a field as a dense ``(nlat, nlon)`` array, which a
    ragged grid such as HEALPix is not.

    Parameters
    ----------
    grid : Any
        The value supplied by the caller.
    name : str, optional
        Name of the parameter, used in the error message, by default ``"grid"``.

    Returns
    -------
    RegularGridS2
        ``grid`` unchanged, once validated.

    Raises
    ------
    TypeError
        If ``grid`` is not a :class:`GridS2` at all, or is one that is not regular.
    """
    grid = require_grid(grid, name)
    if not isinstance(grid, RegularGridS2):
        raise TypeError(
            f"{name} must be a RegularGridS2; this routine is not yet implemented for {type(grid).__name__}, whose latitude rings do not all "
            f"carry the same number of longitudes. Got {grid!r}."
        )
    return grid


def grid_types() -> Tuple[str, ...]:
    """Names of all registered grid types, in registration order."""
    return tuple(_GRID_REGISTRY)


def _resolve_grid_class(spec: Union[str, Type[PointSetS2]]) -> Type[PointSetS2]:
    """Look up the class behind a grid type name, suggesting a near miss if there is one."""
    if isinstance(spec, type) and issubclass(spec, PointSetS2):
        return spec
    if not isinstance(spec, str):
        raise ValueError(f"expected a PointSetS2, a PointSetS2 subclass or a grid type name, got {type(spec).__name__}")
    if spec not in _GRID_REGISTRY:
        message = f"Unknown grid type '{spec}', expected one of {list(_GRID_REGISTRY)}"
        close = difflib.get_close_matches(spec, _GRID_REGISTRY, n=1)
        if close:
            message += f". Did you mean '{close[0]}'?"
        raise ValueError(message)
    return _GRID_REGISTRY[spec]


def grid_params(spec: Union[PointSetS2, str, Type[PointSetS2]]) -> Tuple[str, ...]:
    """
    Names of the parameters a grid type is constructed from, in order.

    A latitude--longitude grid takes ``(nlat, nlon)``; HEALPix takes ``(nside,)``.

    Examples
    --------
    >>> from torch_harmonics import grid_params
    >>> grid_params("equiangular")
    ('nlat', 'nlon')
    """
    if isinstance(spec, PointSetS2):
        return spec.params()
    return _resolve_grid_class(spec).params()


def as_grid(spec: Union[PointSetS2, str, Type[PointSetS2]], **params: Any) -> PointSetS2:
    """
    Construct a grid descriptor from a grid type name and its parameters.

    Parameters are passed by keyword and validated against the requested grid
    type; one that does not apply to it is rejected rather than ignored.

    Parameters
    ----------
    spec : PointSetS2 or str or type
        A descriptor, which is returned unchanged, a grid type name such as
        ``"equiangular"``, or a :class:`PointSetS2` subclass.
    **params
        Parameters of the requested grid, by keyword. Which ones apply depends on
        the grid type; :func:`grid_params` reports them.

    Returns
    -------
    PointSetS2
        The corresponding descriptor.

    Raises
    ------
    ValueError
        If the grid type is unknown, if a parameter does not apply to it, if a
        required parameter is missing, or if the parameters contradict a
        descriptor passed as ``spec``.

    Examples
    --------
    >>> from torch_harmonics import as_grid
    >>> as_grid("equiangular", nlat=128, nlon=256)
    EquiangularGrid(nlat=128, nlon=256)

    Passing a descriptor through is a no-op, so a layer can accept either:

    >>> grid = as_grid("legendre-gauss", nlat=64, nlon=128)
    >>> as_grid(grid) is grid
    True
    """
    if isinstance(spec, PointSetS2):
        # only the type's own parameters can be restated; anything else is a typo or a
        # derived attribute, and comparing it would either always fail or silently pass
        accepted = type(spec).params()
        unknown = [key for key in params if key not in accepted]
        if unknown:
            message = f"{sorted(unknown)} " + ("is not a parameter" if len(unknown) == 1 else "are not parameters")
            raise ValueError(f"{message} of the grid descriptor {spec!r}, which takes {list(accepted)}")
        contradictions = {name: value for name, value in params.items() if getattr(spec, name) != value}
        if contradictions:
            raise ValueError(f"{contradictions} contradicts the grid descriptor {spec!r}")
        return spec

    cls = _resolve_grid_class(spec)
    name = getattr(cls, "grid_type", cls.__name__)
    accepted = cls.params()

    unknown = [key for key in params if key not in accepted]
    if unknown:
        message = f"{sorted(unknown)} " + ("is not a parameter" if len(unknown) == 1 else "are not parameters")
        message += f" of grid '{name}' ({cls.__name__}), which takes {list(accepted)}"
        close = difflib.get_close_matches(unknown[0], accepted, n=1)
        if close:
            message += f". Did you mean '{close[0]}'?"
        raise ValueError(message)

    required = [f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING]
    missing = [key for key in required if key not in params]
    if missing:
        message = f"grid '{name}' ({cls.__name__}) requires {missing}"
        if list(accepted) != missing:
            message += f"; it takes {list(accepted)}"
        raise ValueError(message)

    return cls(**params)


#: how a descriptor parameter is recovered from a legacy call; see _rejects_legacy_signature
_GridSpec = Union[str, Tuple[str, Optional[str]]]


def _rejects_legacy_signature(legacy: str, **grids: _GridSpec):
    r"""
    Turn a pre-v1.0.0 constructor call into an actionable error.

    Layers used to take the resolution and the grid name as separate arguments; they now
    take a descriptor that carries both. The old call either fails to bind -- ``grid``
    named the grid string then and names the descriptor now, so Python reports ``got
    multiple values for argument 'grid'``, or ``unexpected keyword argument 'nlat'`` for
    a keyword call -- or binds with a resolution or a name where a descriptor belongs.
    Neither says what to do instead.

    The old signature differs from layer to layer, so each one declares its own, and the
    guard binds the call against both: nothing is inferred from argument types, which
    would mistake a channel count for a resolution. A call that does not fit the new
    signature but fits the old one, carrying a resolution or a grid name, is answered
    with the replacement, assembled from the old arguments by name so it can be copied.
    It only rejects; a legacy call is never translated and run, so nothing silently
    changes meaning. A call that fits neither keeps Python's own error.

    Parameters
    ----------
    legacy : str
        The pre-v1.0.0 parameter list as it was written, without ``self``. Parsed, not
        evaluated: a default that is not a literal only marks the parameter optional.
    **grids
        For each descriptor parameter of the new signature, the old parameters it
        replaces: the name of a shape tuple (``grid_in="in_shape"``), or an
        ``(nlat, nlon)`` pair of names (``grid=("nlat", "nlon")``), where ``None`` for
        nlon means the old layer derived it as ``2 * nlat``. The grid *name* came from
        the old parameter of the same name, which every old signature had.

    Applied to ``__init__`` of the layers whose signature changed. Uses
    :func:`functools.wraps`, so ``inspect.signature`` and the documentation still report
    the real, descriptor-taking signature.
    """
    legacy_sig = _parse_signature(legacy)
    for name, spec in grids.items():
        named = (spec,) if isinstance(spec, str) else tuple(n for n in spec if n is not None)
        for old in (name,) + named:
            if old not in legacy_sig.parameters:
                raise ValueError(f"legacy signature ({legacy}) has no parameter '{old}' to recover '{name}' from")

    def decorate(init):
        new_sig = inspect.signature(init)
        missing = [name for name in grids if name not in new_sig.parameters]
        if missing:
            raise ValueError(f"{init.__qualname__} takes no descriptor parameter {missing}")

        @functools.wraps(init)
        def wrapper(self, *args, **kwargs):
            _reject_legacy_grid_call(type(self), new_sig, legacy_sig, grids, args, kwargs)
            return init(self, *args, **kwargs)

        # markers so a test can assert every grid-taking constructor carries the guard,
        # and build an old call from what it declares
        wrapper._rejects_legacy_signature = True
        wrapper._legacy_signature = legacy_sig
        wrapper._legacy_grids = dict(grids)
        return wrapper

    return decorate


class _NonLiteralDefault:
    """Stands in for a legacy default that is not a literal: it only marks the parameter optional."""

    def __repr__(self) -> str:
        return "..."


def _parse_signature(params: str) -> inspect.Signature:
    """``inspect.Signature`` of a parameter list given as source, without evaluating it."""
    args = ast.parse(f"def _({params}): pass").body[0].args
    if args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs:
        raise ValueError(f"legacy signatures are plain positional-or-keyword parameter lists, got ({params})")

    def default(node):
        try:
            return ast.literal_eval(node)
        except ValueError:
            return _NonLiteralDefault()

    defaults = [inspect.Parameter.empty] * (len(args.args) - len(args.defaults)) + [default(d) for d in args.defaults]
    return inspect.Signature([inspect.Parameter(a.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=d) for a, d in zip(args.args, defaults)])


def _is_resolution(value: Any) -> bool:
    """An ``nlat``/``nlon`` the old signature would have taken, rather than a descriptor."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_shape(value: Any) -> bool:
    """An ``(nlat, nlon)`` shape the old signature would have taken."""
    return isinstance(value, (tuple, list)) and len(value) == 2 and all(_is_resolution(v) for v in value)


def _resolution_of(spec: _GridSpec, old: Dict[str, Any]) -> Optional[Tuple[Any, Any]]:
    """The ``(nlat, nlon)`` an old call gave a grid, or None if it is not one."""
    if isinstance(spec, str):
        return tuple(old[spec]) if _is_shape(old[spec]) else None
    nlat_name, nlon_name = spec
    nlat = old[nlat_name]
    nlon = 2 * nlat if nlon_name is None and _is_resolution(nlat) else old.get(nlon_name)
    return (nlat, nlon) if _is_resolution(nlat) and _is_resolution(nlon) else None


def _reject_legacy_grid_call(
    cls: Type, new_sig: inspect.Signature, legacy_sig: inspect.Signature, grids: Dict[str, _GridSpec], args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> None:
    """Raise if ``args``/``kwargs`` are a pre-v1.0.0 call. See ``_rejects_legacy_signature``."""

    # the supported form: binds, and every grid slot that was filled holds something that
    # is not a resolution, a shape or a grid name (the descriptor guards judge the rest)
    try:
        bound = new_sig.bind(None, *args, **kwargs).arguments
        if not any(_is_resolution(v) or _is_shape(v) or isinstance(v, str) for v in (bound.get(g) for g in grids)):
            return
    except TypeError:
        pass

    try:
        old_bound = legacy_sig.bind(*args, **kwargs)
    except TypeError:
        return
    passed = set(old_bound.arguments)
    old_bound.apply_defaults()
    old = old_bound.arguments

    # the old call carried a resolution or a grid name: something only it could mean
    carries_resolution = any(_is_resolution(old[n]) or _is_shape(old[n]) for spec in grids.values() for n in ((spec,) if isinstance(spec, str) else spec) if n is not None)
    if not carries_resolution and not any(isinstance(old[g], str) and g in passed for g in grids):
        return

    def descriptor(name: str) -> str:
        grid_name = repr(old[name]) if isinstance(old[name], str) else "<grid name>"
        resolution = _resolution_of(grids[name], old)
        nlat, nlon = resolution if resolution is not None else ("...", "...")
        return f"as_grid({grid_name}, nlat={nlat}, nlon={nlon})"

    # the replacement: the new parameters in order, descriptors for the grids and the old
    # arguments carried over by name; positional while nothing is skipped and the caller
    # had not named it, keyword after
    consumed = set(grids)
    for spec in grids.values():
        consumed.update((spec,) if isinstance(spec, str) else (n for n in spec if n is not None))
    replacement, positional = [], True
    for name, param in list(new_sig.parameters.items())[1:]:
        if name in grids:
            value = descriptor(name)
        elif name in passed and name not in consumed:
            value = repr(old[name])
            if len(value) > 40:
                value = f"<{name}>"
        else:
            positional = False
            continue
        positional = positional and param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and (name in grids or name not in kwargs)
        replacement.append(value if positional else f"{name}={value}")

    # the old signature as the caller knew it: what it required or what held a grid,
    # with runs of the other optional parameters elided
    shown, elided = [], False
    for name, param in legacy_sig.parameters.items():
        if name in grids:
            shown.append(f"{name}=...")
        elif param.default is inspect.Parameter.empty or name in consumed:
            shown.append(name)
        else:
            if not elided:
                shown.append("...")
            elided = True
            continue
        elided = False
    if shown and shown[-1] == "...":
        shown.pop()

    raise TypeError(
        f"{cls.__name__} no longer takes ({', '.join(shown)}); since v1.0.0 it takes a grid descriptor, which carries "
        f"the resolution with it. Write {cls.__name__}({', '.join(replacement)}) instead. A descriptor also knows its "
        f"own parameters, so a grid family that is not described by (nlat, nlon) -- HEALPix, for instance -- fits the "
        f"same call."
    )
