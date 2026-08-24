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
Grid descriptors for spherical latitude--longitude grids.

A :class:`GridS2` bundles everything a layer needs to know about the grid it
operates on -- node positions, quadrature weights, and the derived quantities
that used to be recomputed from ``nlat`` and a grid *string* at each call site.
The intent is that a layer takes one descriptor per side instead of the
``(nlat, nlon, grid)`` triple, so that new grid types can be added without
touching every consumer.

Design notes
------------
* **Hashing is by canonical key, never by tensor identity.** The node and weight
  tensors are looked up lazily through the ``lru_cache`` in
  :mod:`torch_harmonics.quadrature`, and are deliberately *not* dataclass fields.
  A descriptor that carried tensors would fall back to identity hashing and
  silently defeat every cache keyed on the grid.
* **Raggedness is expressible from day one.** :attr:`nlon_per_lat` and
  :attr:`lon_offsets` exist on the regular grids too, where they are trivial.
  Consumers that cannot handle a ragged grid should assert :attr:`is_regular`
  rather than assume a uniform ``nlon`` stride, so that reduced Gaussian grids
  become an additive change instead of a second API break.
* **Descriptors stop at the Python layer.** Compiled kernels keep taking plain
  ints; modules unpack the descriptor before calling into them.
"""

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional, Tuple, Type, Union

import torch

from torch_harmonics.quadrature import compute_latitude_spacing, compute_theta_cutoff, precompute_latitudes, precompute_longitudes

__all__ = [
    "GridS2",
    "EquiangularGrid",
    "LegendreGaussGrid",
    "LobattoGrid",
    "EquiangularTrapezoidalGrid",
    "as_grid",
    "grid_types",
    "require_grid",
]

# populated by __init_subclass__; maps the historical grid string to its class
_GRID_REGISTRY: Dict[str, Type["GridS2"]] = {}


@dataclass(frozen=True, eq=False)
class GridS2:
    r"""
    Descriptor for a latitude--longitude grid on :math:`S^2`.

    This is the abstract base; instantiate one of the concrete subclasses, or use
    :func:`as_grid` to coerce a legacy ``(grid_string, shape)`` pair.

    Each concrete subclass carries a class-level ``grid_type`` holding the
    historical grid string it corresponds to, e.g. ``"equiangular"``. That string
    is used for serialization and for the registry behind :func:`as_grid`.

    Parameters
    ----------
    nlat : int
        Number of latitudinal nodes. Must be at least 2.
    nlon : int
        Number of longitudinal nodes. Must be at least 1.
    """

    nlat: int
    nlon: int

    #: historical grid string; set by each concrete subclass
    grid_type: ClassVar[str]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        grid_type = getattr(cls, "grid_type", None)
        if grid_type is None:
            raise TypeError(f"{cls.__name__} must define a 'grid_type' class attribute")
        if grid_type in _GRID_REGISTRY:
            raise ValueError(f"grid_type '{grid_type}' is already registered to {_GRID_REGISTRY[grid_type].__name__}")
        _GRID_REGISTRY[grid_type] = cls

    def __post_init__(self):
        if type(self) is GridS2:
            raise TypeError("GridS2 is abstract; instantiate a concrete grid or use as_grid()")
        if not isinstance(self.nlat, int) or isinstance(self.nlat, bool):
            raise ValueError(f"nlat must be an int, got {type(self.nlat).__name__}")
        if not isinstance(self.nlon, int) or isinstance(self.nlon, bool):
            raise ValueError(f"nlon must be an int, got {type(self.nlon).__name__}")
        if self.nlat < 2:
            raise ValueError(f"nlat must be at least 2, got {self.nlat}")
        if self.nlon < 1:
            raise ValueError(f"nlon must be at least 1, got {self.nlon}")

    # -- identity ------------------------------------------------------------

    @property
    def key(self) -> Tuple[Any, ...]:
        """
        Canonical, hashable identity of this grid.

        Contains only scalars. Everything that distinguishes two grids must appear
        here, and nothing that does not; this tuple backs both ``__hash__`` and
        ``__eq__``, and therefore every cache keyed on a descriptor.
        """
        return (self.grid_type, self.nlat, self.nlon)

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GridS2):
            return NotImplemented
        return self.key == other.key

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nlat={self.nlat}, nlon={self.nlon})"

    @property
    def shape(self) -> Tuple[int, int]:
        """Spatial shape ``(nlat, nlon)`` of a field sampled on this grid."""
        return (self.nlat, self.nlon)

    # -- geometry ------------------------------------------------------------

    @property
    def lats(self) -> torch.Tensor:
        r"""
        Colatitudes :math:`\theta_k \in [0, \pi]`, ascending (north pole first), shape ``(nlat,)``.
        """
        lats, _ = precompute_latitudes(self.nlat, grid=self.grid_type)
        return lats

    @property
    def quad_weights(self) -> torch.Tensor:
        r"""
        Latitudinal quadrature weights, shape ``(nlat,)``, paired with :attr:`lats`.

        Formulated in the :math:`\cos\theta` domain, so they already absorb the
        :math:`\sin\theta` Jacobian and sum to 2.
        """
        _, w = precompute_latitudes(self.nlat, grid=self.grid_type)
        return w

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
        return precompute_longitudes(self.nlon)

    # -- raggedness ----------------------------------------------------------

    @property
    def is_regular(self) -> bool:
        """
        Whether every latitude ring carries the same number of longitudes.

        ``True`` for all currently implemented grids. Consumers backed by compiled
        kernels, which index with a uniform ``nlon`` stride, should assert this.
        """
        return True

    @property
    def nlon_per_lat(self) -> torch.Tensor:
        """Number of longitudes on each latitude ring, shape ``(nlat,)``."""
        return torch.full((self.nlat,), self.nlon, dtype=torch.int64)

    @property
    def lon_offsets(self) -> torch.Tensor:
        """
        Exclusive prefix sum of :attr:`nlon_per_lat`, shape ``(nlat + 1,)``.

        A point ``(ilat, ilon)`` sits at flat index ``lon_offsets[ilat] + ilon``.
        On a regular grid this is just ``ilat * nlon``, but writing the flattening
        this way keeps consumers valid on ragged grids.
        """
        return torch.arange(self.nlat + 1, dtype=torch.int64) * self.nlon

    @property
    def npoints(self) -> int:
        """Total number of grid points."""
        return self.nlat * self.nlon

    # -- derived quantities --------------------------------------------------

    @property
    def latitude_spacing(self) -> torch.Tensor:
        r"""Gaps :math:`\theta_{k+1} - \theta_k` between adjacent latitudes, shape ``(nlat - 1,)``."""
        lats = self.lats
        return lats[1:] - lats[:-1]

    @property
    def max_latitude_spacing(self) -> float:
        r"""
        Largest gap between adjacent latitudes, :math:`\max_k (\theta_{k+1} - \theta_k)`.

        This is the grid's own notion of "one latitudinal grid spacing". Only
        :class:`EquiangularGrid` is uniform in :math:`\theta`, where it reduces to
        :math:`\pi / (N_\theta - 1)`.
        """
        return compute_latitude_spacing(self.nlat, grid=self.grid_type)

    @property
    def is_uniform_in_theta(self) -> bool:
        r"""Whether the latitude nodes are equispaced in :math:`\theta`."""
        return False

    # -- spectral bounds -----------------------------------------------------
    #
    # These are facts about what the grid can represent, not decisions about
    # what an SHT should keep. The policy -- applying user overrides, enforcing
    # triangular truncation, warning about changed defaults -- lives in
    # :mod:`torch_harmonics.truncation`, so these properties stay silent.

    @property
    def max_exact_degree(self) -> int:
        r"""
        Highest spherical harmonic degree the quadrature rule integrates exactly.

        Non-inclusive, i.e. degrees :math:`0 \le l < l_{\max}`. Determined by the
        exactness of the latitudinal rule, so each grid type answers differently.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define max_exact_degree")

    @property
    def is_spectrally_accurate(self) -> bool:
        r"""
        Whether the latitudinal rule converges spectrally.

        An SHT relies on the associated Legendre polynomials being *discretely*
        orthogonal under the grid's quadrature. Interpolatory rules -- Gauss--Legendre,
        Gauss--Lobatto, Clenshaw--Curtis -- integrate the required polynomial degrees
        exactly, so orthogonality holds to machine precision. A rule that converges
        only algebraically does not, and refining the grid buys back accuracy far more
        slowly than raising the truncation loses it.
        """
        return True

    @property
    def max_azimuthal_order(self) -> int:
        r"""
        Nyquist limit of the longitudinal sampling, :math:`\lfloor N_\lambda / 2 \rfloor + 1`.

        Non-inclusive. On a ragged grid each latitude ring has its own limit; this
        returns the bound for the widest ring, which is the one a dense spectral
        representation has to accommodate.
        """
        return self.nlon // 2 + 1

    def theta_cutoff(self, scale: Optional[float] = 1.0) -> float:
        r"""
        Default angular cutoff for localized operators on this grid.

        Delegates to :func:`torch_harmonics.quadrature.compute_theta_cutoff`, so
        descriptor-based and legacy call sites cannot drift apart.

        Parameters
        ----------
        scale : float, optional
            Multiplier on the grid spacing, by default 1.0.

        Returns
        -------
        float
            Cutoff angle in radians.
        """
        return compute_theta_cutoff(self.nlat, grid=self.grid_type, scale=scale)

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Plain-data representation, suitable for a config file or a checkpoint."""
        return {"grid": self.grid_type, "nlat": self.nlat, "nlon": self.nlon}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "GridS2":
        """Inverse of :meth:`to_dict`."""
        missing = {"grid", "nlat", "nlon"} - set(data)
        if missing:
            raise ValueError(f"grid dict is missing {sorted(missing)}")
        return as_grid(data["grid"], (data["nlat"], data["nlon"]))


@dataclass(frozen=True, eq=False)
class EquiangularGrid(GridS2):
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
class LegendreGaussGrid(GridS2):
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
class LobattoGrid(GridS2):
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
class EquiangularTrapezoidalGrid(GridS2):
    r"""
    Trapezoidal rule applied on the :math:`\cos\theta` interval :math:`[-1, 1]`.

    Despite the name, the nodes are **not** equiangular in :math:`\theta`: they are
    equispaced in :math:`\cos\theta`, which makes the spacing in :math:`\theta`
    strongly non-uniform. The polar spacing is a factor :math:`\sqrt{N_\theta - 1}`
    coarser than the equatorial one, so the disparity grows with resolution instead
    of staying fixed.
    """

    grid_type: ClassVar[str] = "equiangular-trapezoidal"

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


def require_grid(grid: Any, name: Optional[str] = "grid") -> GridS2:
    """
    Validate that a layer received a grid descriptor, with a migration-friendly error.

    Layers used to take a shape plus a grid name; they now take a single
    :class:`GridS2`. Passing either of the old arguments would otherwise fail deep
    inside the constructor with an opaque ``AttributeError``, so intercept it here
    and say what to write instead.

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
    if isinstance(grid, str):
        raise TypeError(f"{name} must be a GridS2, not the grid name {grid!r}. The descriptor carries the resolution too, so build one with " f"as_grid({grid!r}, (nlat, nlon)).")
    if isinstance(grid, (tuple, list)):
        shape = tuple(grid)
        raise TypeError(f"{name} must be a GridS2, not a shape {shape!r}. The descriptor carries the shape, so pass as_grid(<grid name>, {shape!r}) instead.")
    raise TypeError(f"{name} must be a GridS2, got {type(grid).__name__}. Build one with as_grid(<grid name>, (nlat, nlon)).")


def grid_types() -> Tuple[str, ...]:
    """Names of all registered grid types, in registration order."""
    return tuple(_GRID_REGISTRY)


def as_grid(spec: Union[GridS2, str], shape: Optional[Tuple[int, int]] = None) -> GridS2:
    """
    Coerce a grid specification into a :class:`GridS2`.

    Lets a layer accept either a descriptor or the historical
    ``grid="equiangular"`` string plus a shape, so the descriptor API can be
    introduced without breaking existing call sites.

    Parameters
    ----------
    spec : GridS2 or str
        A descriptor, which is returned unchanged, or a grid type name.
    shape : tuple of int, optional
        ``(nlat, nlon)``. Required when ``spec`` is a string, and must agree with
        the descriptor when one is passed.

    Returns
    -------
    GridS2
        The corresponding descriptor.

    Raises
    ------
    ValueError
        If the grid type is unknown, if ``shape`` is missing for a string spec, or
        if ``shape`` contradicts a descriptor spec.

    Examples
    --------
    >>> from torch_harmonics import as_grid
    >>> as_grid("equiangular", (128, 256))
    EquiangularGrid(nlat=128, nlon=256)
    """
    if isinstance(spec, GridS2):
        if shape is not None and tuple(shape) != spec.shape:
            raise ValueError(f"shape {tuple(shape)} contradicts the grid descriptor {spec}")
        return spec

    if not isinstance(spec, str):
        raise ValueError(f"expected a GridS2 or a grid type name, got {type(spec).__name__}")

    if spec not in _GRID_REGISTRY:
        raise ValueError(f"Unknown grid type {spec}, expected one of {list(_GRID_REGISTRY)}")

    if shape is None:
        raise ValueError(f"shape is required when specifying a grid by name (got grid='{spec}')")
    if len(shape) != 2:
        raise ValueError(f"shape must be a 2-tuple (nlat, nlon), got length {len(shape)}")

    nlat, nlon = shape
    return _GRID_REGISTRY[spec](nlat=int(nlat), nlon=int(nlon))
