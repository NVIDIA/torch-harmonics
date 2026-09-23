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

import difflib
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
    if grid.is_regular:
        # every ring carries the same longitudes, so tile rather than walking the rings
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

    The weakest contract in this module, and the base of the hierarchy. A point set
    is :attr:`npoints` locations on the sphere, each with an angular position and a
    quadrature weight, and nothing else. It says nothing about how those points are
    arranged, so it covers an unstructured mesh -- an icosahedral/triangular ICON
    grid, say -- as well as everything in :class:`GridS2` below it.

    What a consumer may assume grows down the hierarchy, and each level is exactly
    what some algorithm needs:

    * :class:`PointSetS2` -- points and weights. Enough to **integrate**.
    * :class:`GridS2` -- adds isolatitude rings with equispaced longitudes. Enough
      for an **FFT in longitude** (hence a fast SHT), for a neighbourhood search
      bounded by a binary search over rings rather than a spatial index, and for a
      contiguous polar decomposition.
    * :class:`RegularGridS2` -- adds a uniform longitude count. Enough for a dense
      ``(nlat, nlon)`` layout, the compiled kernels, and a 2D decomposition.

    Every subclass adds a constraint, so every :class:`RegularGridS2` is a
    :class:`GridS2` is a :class:`PointSetS2`.

    This class is abstract, as are :class:`GridS2` and :class:`RegularGridS2`.
    Instantiate one of the concrete grids, or build one by name with :func:`as_grid`.

    Each *concrete* subclass carries a class-level ``grid_type`` holding the
    historical grid string it corresponds to, e.g. ``"equiangular"``. That string
    is used for serialization and for the registry behind :func:`as_grid`. A class
    that does not define its own ``grid_type`` is treated as an abstract
    intermediate: it is neither registered nor instantiable.

    Notes
    -----
    Three properties of this type are load-bearing rather than incidental.

    The dataclass fields *are* the parameterization. :meth:`params`, :attr:`key`,
    :meth:`to_dict` and ``__repr__`` are all derived from them, so a family
    parameterized by something other than ``(nlat, nlon)`` -- a HEALPix ``nside``,
    an icosahedral refinement level -- gets correct construction, identity and
    serialization without registering anything. A subclass cannot forget to extend
    its own identity, which matters because forgetting would not raise: it would
    silently collide in every cache keyed on the descriptor.

    Node and weight tensors are deliberately not fields. A descriptor carrying
    tensors would fall back to identity hashing and defeat those same caches. A
    family whose nodes are *data* rather than a formula -- read from an ICON grid
    file, say -- therefore needs an identity that is not its data: a registered
    name, or a path plus a content hash. Getting that wrong does not raise either;
    two such grids would collide in every cached sparsity pattern.

    Descriptors stop at the Python layer: compiled kernels keep taking plain ints,
    and modules unpack the descriptor before calling into them.
    """

    #: historical grid string; set by each concrete subclass, absent on abstract ones
    grid_type: ClassVar[str]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # only a class that declares its own grid_type is concrete; an abstract
        # intermediate such as GridS2 or RegularGridS2 merely inherits the annotation
        grid_type = cls.__dict__.get("grid_type")
        if grid_type is None:
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
        """
        Names of this descriptor's constructor parameters, in declaration order.

        Derived from the dataclass fields, so it cannot drift from the constructor.
        """
        return tuple(f.name for f in fields(cls))

    @property
    def key(self) -> Tuple[Any, ...]:
        """
        Canonical, hashable identity of this descriptor.

        Contains only scalars. Everything that distinguishes two descriptors must
        appear here, and nothing that does not; this tuple backs both ``__hash__``
        and ``__eq__``, and therefore every cache keyed on a descriptor.

        Derived from :meth:`params` rather than listing the fields, so that a
        subclass which adds a parameter cannot forget to extend it.
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

        ``(npoints,)`` here, since an unstructured set has no second axis to speak
        of. :class:`RegularGridS2` overrides it with ``(nlat, nlon)``, where the
        sampling really is a product of two axes. Consumers that splat this into a
        ``reshape`` or compare it against ``x.shape[-len(grid.shape):]`` stay
        correct either way, while those that unpack it into two names do not
        silently misbehave.
        """
        return (self.npoints,)

    # -- geometry ------------------------------------------------------------

    @property
    def coords(self) -> torch.Tensor:
        r"""
        Angular position of every point, shape ``(npoints, 2)``.

        Column 0 is colatitude :math:`\theta \in [0, \pi]` measured from the north
        pole, column 1 is longitude :math:`\lambda \in [0, 2\pi)`. Colatitude rather
        than latitude because that is what the library computes with; see
        :attr:`GridS2.colats`.

        Row ``i`` describes element ``i`` of a field flattened to ``(npoints,)``.
        That correspondence is the load-bearing part of this contract -- getting it
        wrong transports the right values to the wrong places without raising -- so
        a family that defines its own flat order must document it. For a
        :class:`GridS2` the order is ring-major: point ``(ilat, ilon)`` sits at
        ``lon_offsets[ilat] + ilon``.

        Note that the pole-inclusive grids repeat a physical point: an equiangular
        grid carries ``nlon`` rows at :math:`\theta = 0`, differing only in a
        longitude that means nothing there.
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
            Not to be confused with :attr:`GridS2.colat_weights`, which is the
            *latitudinal* factor alone: shape ``(nrings,)`` and summing to 2, with
            the longitudinal :math:`2\pi / N_\lambda` excluded. The two differ by
            exactly that factor, so substituting one for the other scales an
            integral by :math:`2\pi` and raises nothing.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define quad_weights")

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

        Non-inclusive, i.e. degrees :math:`0 \le l < l_{\max}`. Determined by the
        exactness of the rule, so each family answers differently -- and some have
        no answer at all: an equal-area rule such as HEALPix integrates a constant
        exactly and nothing beyond, and should raise rather than invent a bound.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define max_exact_degree")

    @property
    def is_spectrally_accurate(self) -> bool:
        r"""
        Whether the quadrature rule converges spectrally.

        An SHT relies on the associated Legendre polynomials being *discretely*
        orthogonal under the rule. Interpolatory latitudinal rules --
        Gauss--Legendre, Gauss--Lobatto, Clenshaw--Curtis -- integrate the required
        degrees exactly, so orthogonality holds to machine precision. A rule that
        converges only algebraically does not, and refining buys back accuracy far
        more slowly than raising the truncation loses it.

        ``False`` here, which is the conservative answer and the right default for
        an arbitrary point set: nothing about a bag of points and weights implies
        discrete orthogonality. :class:`RegularGridS2` overrides it to ``True``,
        which is where the interpolatory rules actually live, so a ragged or
        unstructured family has to claim spectral accuracy deliberately rather than
        inherit the claim by accident.
        """
        return False

    @property
    def max_node_spacing(self) -> float:
        r"""
        Largest great-circle distance between neighbouring nodes, in radians.

        The grid's resolution expressed as an angle: how far apart its coarsest pair
        of adjacent samples sits on the sphere. A *fact* about the node distribution,
        which is why it is a spacing rather than a cutoff -- it neither applies a user
        override nor warns that a default moved. Turning it into the support radius of
        a localized operator is policy and lives in
        :func:`torch_harmonics.truncate_support`, which is what the layers call.

        Abstract here because "neighbouring" needs a definition, and an unstructured
        set has to supply its own -- typically a nearest-neighbour distance over a
        spatial index. :class:`GridS2` has rings and defines it in closed form.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define max_node_spacing")

    # -- decomposition -------------------------------------------------------

    @classmethod
    def shard_class(cls) -> Type["GridShardS2"]:
        """
        The :class:`GridShardS2` subclass that :meth:`shard` produces.

        Named by the descriptor rather than inferred, so that deserializing a shard
        can reconstruct the right type from the descriptor alone.
        """
        raise NotImplementedError(f"{cls.__name__} does not define shard_class")

    def shard(self, **decomposition: Any) -> "GridShardS2":
        """
        Return the piece of this descriptor held by one rank of a decomposition.

        How a sampling decomposes is a property of the sampling, which is why this
        is asked of the descriptor rather than computed by the caller. A regular
        latitude--longitude grid splits as a product of a latitude range and a
        longitude range, and :meth:`RegularGridS2.shard` takes ``polar`` and
        ``azimuth`` accordingly. A ragged grid has no single ``nlon`` to split, and
        an unstructured set has no rings either, so both would take something else;
        the signature is left to the subclass.

        Deliberately takes plain integers rather than process groups, so that the
        descriptors stay free of any dependency on :mod:`torch.distributed` and
        remain testable without a process group. The distributed layers translate
        their groups into these.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define shard")

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Plain-data representation, suitable for a config file or a checkpoint.

        The grid type plus its own parameters, so a family that takes ``nside``
        serializes as ``{"grid": ..., "nside": ...}`` without special-casing.
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

    Each ring sits at a colatitude :math:`\theta_k` and carries some number of
    longitudes, equispaced around the circle. That is the whole of what this level
    adds to :class:`PointSetS2`, and it deliberately says nothing about *how many*
    longitudes each ring carries, so it covers a reduced Gaussian or HEALPix grid --
    where the count varies from ring to ring -- as well as the regular
    latitude--longitude grids in :class:`RegularGridS2`.

    Still abstract; the node distribution is chosen by the concrete subclasses.

    Notes
    -----
    Ring structure is not bookkeeping, it is what several algorithms require:

    * **Separable quadrature.** The weight of a point factorizes as
      :math:`w_k \cdot 2\pi / N_{\lambda,k}`, so :attr:`colat_weights` stores
      :math:`O(\text{nrings})` numbers rather than :math:`O(\text{npoints})`.
    * **An FFT in longitude.** Since
      :math:`Y_l^m(\theta, \lambda) = P_l^m(\cos\theta)\, e^{im\lambda}`, equispaced
      longitudes turn the :math:`\lambda` integral into an FFT. This is what makes a
      fast SHT possible at all; without rings the transform becomes a dense
      least-squares solve.
    * **A bounded neighbourhood search.** Only rings within the cutoff of an output
      colatitude can contribute, which is a binary search rather than a spatial
      index.
    * **A contiguous polar decomposition**, and hence a halo exchange with immediate
      neighbours only.

    :attr:`nlon_per_lat` and :attr:`lon_offsets` exist on the regular grids too,
    where they are trivial. Consumers that cannot handle a ragged grid should demand
    a :class:`RegularGridS2` via :func:`require_regular_grid` rather than assume a
    uniform ``nlon`` stride, so that ragged grids become an additive change instead
    of a second API break.

    The flat order is ring-major: point ``(ilat, ilon)`` sits at flat index
    ``lon_offsets[ilat] + ilon``. :attr:`coords` and :attr:`quad_weights` follow it.
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

        One value per *ring*, not per point; :attr:`coords` is the per-point form.

        This is the primitive the library computes with: the associated Legendre
        functions are naturally expressed in :math:`\cos\theta`, and every quadrature
        rule here is formulated on that interval. :attr:`lats` converts to geographic
        latitude for the callers that want it.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define colats")

    @property
    def lats(self) -> torch.Tensor:
        r"""
        Geographic latitudes :math:`\phi_k = \pi/2 - \theta_k \in [-\pi/2, \pi/2]`, shape ``(nrings,)``.

        Descending, north pole first, because :attr:`colats` ascends from the north
        pole and latitude is its reflection. Provided because plotting, geographic
        data and anything user-facing want latitude, and because the conversion was
        previously open-coded at three call sites -- writing :math:`\pi - \theta`
        instead of :math:`\pi/2 - \theta` yields a number in :math:`[0, \pi]` that
        looks like a plausible angle and is wrong everywhere except the equator.

        Derived, never stored: :attr:`colats` is the primitive, so the two cannot
        drift apart.
        """
        return torch.pi / 2 - self.colats

    @property
    def colat_weights(self) -> torch.Tensor:
        r"""
        Latitudinal quadrature weights, shape ``(nrings,)``, paired with :attr:`colats`.

        Formulated in the :math:`\cos\theta` domain, so they already absorb the
        :math:`\sin\theta` Jacobian and **sum to 2**. The longitudinal factor is
        *not* included; on a regular grid it is the uniform :math:`2\pi / N_\lambda`.

        .. warning::
            The per-point :attr:`~PointSetS2.quad_weights` is a different tensor:
            shape ``(npoints,)`` and summing to :math:`4\pi`, with the longitudinal
            factor folded in. Substituting one for the other scales an integral by
            :math:`2\pi` and raises nothing.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define colat_weights")

    def lons(self, ilat: Optional[int] = None) -> torch.Tensor:
        r"""
        Longitudes :math:`\lambda_j \in [0, 2\pi)` of a latitude ring.

        Parameters
        ----------
        ilat : int, optional
            Index of the latitude ring. Ignored on regular grids, where every ring
            carries the same longitudes; accepted so that consumers can be written
            once and keep working on ragged grids.

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

        The separable form made explicit: ring :math:`k` contributes
        :math:`w_k \cdot 2\pi / N_{\lambda,k}` at each of its points, and since
        :attr:`colat_weights` sums to 2 the total is :math:`4\pi` by construction.
        """
        return _grid_quad_weights(self)

    def ring_weights(self, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        r"""
        Quadrature weight carried by a single point of each ring, shape ``(nrings,)``.

        The separable factorization of :attr:`quad_weights`: ring :math:`k` contributes
        :math:`w_k \cdot 2\pi / N_{\lambda,k}` at each of its points, so this is that
        per-point contribution indexed by ring. A consumer that already knows which ring
        a point belongs to -- a kernel walking a grid ring by ring -- wants this rather
        than the expanded form, and on a ragged grid it is the only compact form there is.

        Takes a dtype rather than returning float64 and leaving the caller to cast,
        because the two are not the same number. The arithmetic is done entirely in the
        requested dtype; rounding a float64 result instead lands about one ulp away, which
        is a visible shift in the quadrature weights of an already-trained model.

        Parameters
        ----------
        dtype : torch.dtype, optional
            Dtype to compute in, by default ``torch.float64``.

        Returns
        -------
        torch.Tensor
            Per-ring weights of shape ``(nrings,)``.
        """
        return _grid_ring_weights(self, dtype)

    def point_weights(self, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        r"""
        The same quadrature as :meth:`ring_weights`, one entry per point.

        Equal to :attr:`quad_weights` up to rounding -- that property is the descriptor's
        canonical per-point rule and keeps its own arithmetic, while this one is computed
        wholly in ``dtype`` so that it stays exactly consistent with
        :meth:`ring_weights`, which is what a consumer holding both forms needs.

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

        Zero on the product grids, where every ring starts at :math:`\lambda = 0`.
        HEALPix staggers successive rings by half a point, which is what makes its
        pixels rhombic and equal-area; a consumer assuming every ring starts at 0 would
        misplace half the grid.

        :meth:`lons` already includes the shift, so this exists for the consumers that
        need the offset *separately* -- reconstructing a point's index within its ring,
        as the neighbourhood kernels do.
        """
        return torch.zeros(self.nrings, dtype=torch.float64)

    @property
    def lon_offsets(self) -> torch.Tensor:
        """
        Exclusive prefix sum of :attr:`nlon_per_lat`, shape ``(nrings + 1,)``.

        A point ``(ilat, ilon)`` sits at flat index ``lon_offsets[ilat] + ilon``.
        On a regular grid this is just ``ilat * nlon``, but writing the flattening
        this way keeps consumers valid on ragged grids.
        """
        counts = self.nlon_per_lat
        return torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])

    @property
    def is_regular(self) -> bool:
        """
        Whether every latitude ring carries the same number of longitudes.

        Consumers backed by compiled kernels, which index with a uniform ``nlon``
        stride, should demand a :class:`RegularGridS2` outright rather than test
        this, so that the failure is a clear error at construction.
        """
        counts = self.nlon_per_lat
        return bool((counts == counts[0]).all())

    # -- derived quantities --------------------------------------------------

    @property
    def latitude_spacing(self) -> torch.Tensor:
        r"""
        Gaps :math:`\theta_{k+1} - \theta_k` between adjacent latitudes, shape ``(nrings - 1,)``.

        Taken on the colatitudes, but the gaps are the same either way: latitude is
        colatitude reflected, so the differences only change sign, and every consumer
        here wants the magnitude.
        """
        colats = self.colats
        return colats[1:] - colats[:-1]

    @property
    def max_latitude_spacing(self) -> float:
        r"""
        Largest gap between adjacent latitudes, :math:`\max_k (\theta_{k+1} - \theta_k)`.

        This is the grid's own notion of "one latitudinal grid spacing". Only
        :class:`EquiangularGrid` is uniform in :math:`\theta`, where it reduces to
        :math:`\pi / (N_\theta - 1)`.
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

        The great-circle arc, not the coordinate gap :math:`2\pi / N_{\lambda,k}`. The
        two differ by the :math:`\sin\theta` foreshortening:

        .. math::

            d_k = 2 \arcsin\!\left( \sin\theta_k \, \sin\frac{\pi}{N_{\lambda,k}} \right)

        Using the coordinate gap instead would report the polar rings as the widest,
        when their points are in fact nearly coincident -- on a ring at
        :math:`\theta \to 0` the gap tends to :math:`2\pi / N_\lambda` while the
        distance tends to 0.
        """
        colats = self.colats
        dlambda = 2.0 * torch.pi / self.nlon_per_lat.to(colats.dtype)
        return float((2.0 * torch.asin(torch.sin(colats) * torch.sin(0.5 * dlambda))).max())

    @property
    def max_node_spacing(self) -> float:
        r"""
        Largest great-circle distance between neighbouring nodes, in radians.

        The coarser of :attr:`max_latitude_spacing` and :attr:`max_longitude_spacing`,
        because a ring grid's neighbours run in both directions and the support of an
        operator has to reach the further one.

        Which of the two wins is not a formality. It is latitudinal on an equiangular
        grid at :math:`N_\lambda = 2 N_\theta`, but longitudinal on a Gauss grid at the
        same resolution, on any grid with :math:`N_\lambda < 2 N_\theta`, and on
        HEALPix, whose in-ring spacing exceeds its ring spacing by roughly 1.8x.
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

    The sampling is a product of a latitudinal rule and a uniform longitudinal one,
    which is what makes a field on it a dense ``(nlat, nlon)`` array, makes the
    spherical harmonic transform separable, and makes a 2D process decomposition
    meaningful. Everything that depends on those facts lives here rather than on
    :class:`GridS2`.

    Still abstract: the latitudinal rule is chosen by the concrete subclasses
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
        """
        ``True``: the latitudinal rules of this family are interpolatory.

        Overrides the conservative ``False`` on :class:`PointSetS2`. Declared at
        this level rather than higher up so that a ragged or unstructured family
        has to claim spectral accuracy deliberately -- HEALPix would inherit the
        wrong answer from a base that defaulted to ``True``.
        """
        return True

    @property
    def max_azimuthal_order(self) -> int:
        r"""
        Nyquist limit of the longitudinal sampling, :math:`\lfloor N_\lambda / 2 \rfloor + 1`.

        Non-inclusive. Well defined only because every ring is sampled alike; on a
        ragged grid each ring has its own limit.
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
        """
        Latitude counts held by each rank of a ``num_chunks``-way polar split.

        Balancing rings balances work because every ring is the same length here; see
        the note on :class:`GridS2` for why that reasoning does not survive contact
        with a ragged grid, and why this therefore lives on this class rather than on
        the base.
        """
        return tuple(compute_split_shapes(self.nlat, num_chunks))

    def lon_shapes(self, num_chunks: int) -> Tuple[int, ...]:
        """Longitude counts held by each rank of a ``num_chunks``-way azimuthal split."""
        return tuple(compute_split_shapes(self.nlon, num_chunks))


@dataclass(frozen=True, eq=False)
class GridShardS2:
    r"""
    One rank's piece of a decomposed :class:`GridS2`.

    A shard is deliberately **not** a :class:`GridS2`, because it is not a grid on the
    sphere. A band of latitudes does not cover :math:`S^2`, so:

    * its quadrature weights are partial. :attr:`RegularGridShardS2.colat_weights` does
      not sum to 2 and :attr:`RegularGridShardS2.quad_weights` does not sum to
      :math:`4\pi`; each is the local contribution to an integral that a collective
      reduction completes;
    * quantities that describe the quadrature *rule* rather than this piece of it --
      the spectral bounds, the angular support radius -- are global, and a shard does
      not define them at all. Ask :attr:`global_grid` for them. Absent is a stronger
      guarantee than forwarded: a support radius derived from a shard's own node
      spacing would differ between ranks, and ranks disagreeing about the support of
      an operator is a correctness bug rather than an inefficiency.

    Making this a separate type keeps that distinction enforceable: a shard cannot be
    passed where a global grid is required, and :func:`require_grid` says so.

    This class is abstract; which decomposition parameters exist depends on how the
    grid splits, so they live on the subclasses. See :class:`RegularGridShardS2`.

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
        """
        Canonical identity, including the global grid's own key.

        Derived from :meth:`params`, so a shard type with different decomposition
        parameters extends it automatically.
        """
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

        The local counterpart of :attr:`~PointSetS2.quad_weights`, and like everything
        else on a shard a *partial* quantity: these sum to :math:`4\pi` only once summed
        across every rank, which is what the collective in the layer completes.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define quad_weights")

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Plain-data representation, carrying the global grid with it."""
        return {name: self.grid.to_dict() if name == "grid" else getattr(self, name) for name in self.params()}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "GridShardS2":
        """
        Inverse of :meth:`to_dict`.

        The shard type follows from the grid type, so no separate tag is stored.
        """
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

    A regular grid is a product of a latitude axis and a longitude axis, so it
    splits as a product of a latitude range and a longitude range. Both halves of
    that are specific to this grid family: a ragged grid has no single ``nlon`` to
    split, and so no meaningful azimuthal rank.

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

        Built from the *global* ring lengths and the *local* point counts: splitting a
        ring across azimuth ranks divides its points up without changing how much solid
        angle each one covers. Sums to :math:`4\pi` only across all ranks.
        """
        return _shard_quad_weights(self)

    def lons(self, ilat: Optional[int] = None) -> torch.Tensor:
        """This rank's slice of the longitudes of a latitude ring."""
        return self.grid.lons(ilat)[self.lon_offset : self.lon_offset + self.nlon]


@dataclass(frozen=True, eq=False)
class EquiangularGrid(RegularGridS2):
    r"""
    Equiangular grid with Clenshaw--Curtis quadrature.

    Nodes are equally spaced in :math:`\theta` and include both poles, so the
    latitudinal spacing is exactly :math:`\pi / (N_\theta - 1)` everywhere. This is
    the default grid throughout torch-harmonics.
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

    Optimal quadrature accuracy per node, exact for polynomials up to degree
    :math:`2N - 1`, but the nodes exclude the poles and are not uniform in
    :math:`\theta`.
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

    Nodes cluster towards the equator, so the polar spacing is noticeably coarser
    than :math:`\pi / (N_\theta - 1)`.
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

    The nodes are equispaced in :math:`\cos\theta`, **not** in :math:`\theta`, which
    makes the spacing in :math:`\theta` strongly non-uniform: the polar spacing is a
    factor :math:`\sqrt{N_\theta - 1}` coarser than the equatorial one, so the
    disparity grows with resolution instead of staying fixed. At
    :math:`N_\theta = 17` the nodes sit up to 19 degrees away from
    :class:`EquiangularGrid`'s.

    Previously called ``"equiangular-trapezoidal"``, which named it after nodes it
    does not have. That string is no longer accepted.
    """

    grid_type: ClassVar[str] = "trapezoidal"

    @property
    def max_exact_degree(self) -> int:
        r"""
        Matches the equiangular grid, :math:`\lfloor (N_\theta + 1) / 2 \rfloor`.

        Retained for backwards compatibility, but see
        :attr:`is_spectrally_accurate`: the trapezoidal rule is not accurate
        enough to reach this degree, so the value is optimistic.
        """
        return (self.nlat + 1) // 2

    @property
    def is_spectrally_accurate(self) -> bool:
        """
        ``False``. The trapezoidal rule converges only algebraically, as :math:`O(h^2)`.

        The consequence for an SHT is severe, because the default truncation grows
        with resolution faster than the accuracy does. Measured round-trip relative
        error at ``nlat = 64``, against ~1e-15 for the interpolatory rules:

        ========  ========
        ``lmax``  rel. err
        ========  ========
        1         9e-16
        2         2.2e-4
        4         1.6e-3
        8         2.4e-2
        16        1.3e-1
        32        6.7e-1
        ========  ========

        Only ``lmax = 1`` -- the constant mode -- is exact; the rule integrates
        functions linear in :math:`\\cos\theta` without error, and nothing beyond.
        ``lmax = 32`` is the default this grid is assigned at ``nlat = 64``. Refining
        the grid helps only as :math:`n^{-2}`, so this grid is usable for a transform
        at very low truncation and not otherwise. It remains perfectly
        serviceable for plain quadrature and for the localized operators.
        """
        return False


def require_point_set(grid: Any, name: Optional[str] = "grid") -> PointSetS2:
    """
    Validate that a routine received a descriptor of *some* sampling of the sphere.

    The weakest of the three guards, for routines that need only points and weights
    -- integration, or asking for an angular support radius. Use it in preference to
    :func:`require_grid` wherever the routine genuinely does not care how the points
    are arranged, so that an unstructured family works without a second API break.

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

    Layers used to take a shape plus a grid name; they now take a descriptor.
    Passing either of the old arguments would otherwise fail deep inside the
    constructor with an opaque ``AttributeError``, so intercept it here and say what
    to write instead.

    This demands a :class:`GridS2` rather than a :class:`PointSetS2`, i.e. that the
    points are organized into isolatitude rings. That is what an FFT in longitude,
    a ring-bounded neighbourhood search and a contiguous polar decomposition all
    rely on; an unstructured set has none of it. Routines needing only points and
    weights should call :func:`require_point_set` instead.

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
        raise TypeError(
            f"{name} must be a grid descriptor, not the grid name {grid!r}. The descriptor carries the resolution too, so build one with " f"as_grid({grid!r}, nlat=..., nlon=...)."
        )
    if isinstance(grid, (tuple, list)) and len(grid) == 2:
        nlat, nlon = grid
        raise TypeError(
            f"{name} must be a grid descriptor, not a shape {tuple(grid)!r}. The descriptor carries the shape, so pass "
            f"as_grid(<grid name>, nlat={nlat!r}, nlon={nlon!r}) instead."
        )
    raise TypeError(f"{name} must be a grid descriptor, got {type(grid).__name__}. Build one with as_grid(<grid name>, nlat=..., nlon=...).")


def require_regular_grid(grid: Any, name: Optional[str] = "grid") -> RegularGridS2:
    """
    Validate that a routine received a grid it can actually handle.

    Most of torch-harmonics is backed by kernels that address a field as a dense
    ``(nlat, nlon)`` array with a uniform longitude stride. That is a property of
    :class:`RegularGridS2`, not of :class:`GridS2`: a ragged grid such as a reduced
    Gaussian or HEALPix grid has a different number of longitudes on each latitude
    ring, and handing one to those kernels would not raise -- it would silently
    index the wrong points.

    Every routine with that assumption calls this, so the assumption is stated once
    per routine and fails loudly at construction. As support for other grid
    families lands in a backend, the corresponding call relaxes to
    :func:`require_grid`; the guards are meant to be removed one at a time rather
    than all at once.

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

    Lets a caller -- or a config loader, or an error message -- ask what a grid
    takes without knowing its class, since the parameterization differs between
    grid families: a regular latitude--longitude grid takes ``(nlat, nlon)``, while
    a HEALPix or icosahedral grid takes a refinement level.

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

    The single construction entry point, so that a grid can also be built from a
    config file or a checkpoint where the type is a string. Parameters are passed
    by keyword and validated against the parameterization of the requested grid,
    which means a parameter that is meaningless for a grid family is rejected with
    a message naming what that family does take, rather than being silently
    ignored or misinterpreted.

    Parameters
    ----------
    spec : GridS2 or str or type
        A descriptor, which is returned unchanged, a grid type name such as
        ``"equiangular"``, or a :class:`GridS2` subclass.
    **params
        Parameters of the requested grid, by keyword. Which ones apply depends on
        the grid type; :func:`grid_params` reports them.

    Returns
    -------
    GridS2
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
        contradictions = {name: value for name, value in params.items() if getattr(spec, name, object()) != value}
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
