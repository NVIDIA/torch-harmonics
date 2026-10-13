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
Radial grids, and the product of a radial grid with a sampling of the sphere.

A volumetric field on a ball, a shell or the space outside a sphere is sampled on
:math:`r_k \times S^2`: the same angular grid at every radius. That is a direct product,
so it is described as one -- a :class:`RadialGrid` for the radial nodes and weights, and
a :class:`SphericalProductGrid` pairing it with any :class:`~torch_harmonics.grid.PointSetS2`
-- rather than by a radial variant of every angular grid.

The product is deliberately *not* a :class:`~torch_harmonics.grid.PointSetS2`. A layer on
the sphere that is handed one acts on every shell alike, with the radius as one more
batch axis; it builds on :attr:`SphericalProductGrid.angular` and never sees the radial
grid. A field on the product is laid out ``(..., nr, *angular.shape)``, the radial axis
directly in front of the angular ones, which for a regular angular grid is the
``(D, H, W)`` order of volumetric data.
"""

import difflib
import math
import numbers
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Dict, Optional, Tuple, Type, Union

import torch

from torch_harmonics.cache import lru_cache
from torch_harmonics.grid import GridShardS2, PointSetS2, _as_int, require_point_set
from torch_harmonics.quadrature import legendre_gauss_weights

__all__ = [
    "RadialGrid",
    "GeometricRadialGrid",
    "LegendreGaussRadialGrid",
    "SphericalProductGrid",
    "as_radial_grid",
    "radial_grid_types",
]

# populated by __init_subclass__, like the angular registry in torch_harmonics.grid but
# separate from it: "legendre-gauss" names an angular grid there and a radial rule here
_RADIAL_REGISTRY: Dict[str, Type["RadialGrid"]] = {}


def _as_float(owner: Any, name: str) -> None:
    """Validate a descriptor's real field in place, normalizing it to ``float`` like :func:`~torch_harmonics.grid._as_int` does for integers."""
    value = getattr(owner, name)
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {type(value).__name__}")
    if math.isnan(value):
        raise ValueError(f"{name} must not be NaN")
    if type(value) is not float:
        object.__setattr__(owner, name, float(value))


# Built once per descriptor and cached on its key, as the angular grids do; copy=True so a
# caller writing into the result cannot change it for the next one.
@lru_cache(typed=True, copy=True)
def _radial_nodes(grid: "RadialGrid") -> Tuple[torch.Tensor, torch.Tensor]:
    return grid._nodes_and_weights()


@dataclass(frozen=True, eq=False)
class RadialGrid:
    r"""
    Descriptor for the radial nodes and weights of a volumetric grid.

    ``nr`` nodes :math:`r_k` in the domain :math:`[r_\mathrm{inner}, r_\mathrm{outer}]`,
    with weights :math:`w_k` such that :math:`\sum_k w_k f(r_k) \approx \int f(r)\,dr`.
    The domain bounds are where boundary conditions apply, and they need not be nodes:
    ``r_outer`` may be infinite, and a rule may keep its nodes off the boundary.
    Together they say which domain the grid covers:

    ==========================  =========================  ======================
    ``r_inner``                 ``r_outer``                :attr:`domain`
    ==========================  =========================  ======================
    0                           finite                     ``"ball"``
    0                           ``inf``                    ``"half-line"``
    positive                    finite                     ``"shell"``
    positive                    ``inf``                    ``"exterior"``
    ==========================  =========================  ======================

    Abstract; subclassed once per radial rule, each with a class-level ``radial_type``
    string under which :func:`as_radial_grid` finds it. Descriptors are immutable and
    hashable, with equality defined by the rule and its parameters, as for the angular
    grids.
    """

    #: name of the rule; set by each concrete subclass, absent on abstract ones
    radial_type: ClassVar[str]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        radial_type = cls.__dict__.get("radial_type")
        if radial_type is None:
            # same reasoning as PointSetS2: a subclass inheriting a concrete rule's name
            # would share its identity, and with it every cache keyed on it
            inherited = next((base.__dict__["radial_type"] for base in cls.__mro__[1:] if "radial_type" in base.__dict__), None)
            if inherited is not None:
                raise TypeError(f"{cls.__name__} subclasses a concrete radial grid without declaring its own radial_type, so it would share '{inherited}''s identity")
            return
        if radial_type in _RADIAL_REGISTRY:
            raise ValueError(f"radial_type '{radial_type}' is already registered to {_RADIAL_REGISTRY[radial_type].__name__}")
        _RADIAL_REGISTRY[radial_type] = cls

    def __post_init__(self):
        if not hasattr(type(self), "radial_type"):
            raise TypeError(f"{type(self).__name__} is abstract; instantiate a concrete radial grid or use as_radial_grid()")
        _as_int(self, "nr")
        if self.nr < 1:
            raise ValueError(f"nr must be positive, got {self.nr}")
        for name in ("r_inner", "r_outer"):
            _as_float(self, name)
        if not 0.0 <= self.r_inner < self.r_outer:
            raise ValueError(f"expected 0 <= r_inner < r_outer, got r_inner={self.r_inner}, r_outer={self.r_outer}")

    # -- identity ------------------------------------------------------------

    @classmethod
    def params(cls) -> Tuple[str, ...]:
        """Names of this descriptor's constructor parameters, in declaration order."""
        return tuple(f.name for f in fields(cls))

    @property
    def key(self) -> Tuple[Any, ...]:
        """
        Canonical, hashable identity of this descriptor.

        A tuple of the rule's name followed by the constructor parameters; it defines both
        equality and hashing.
        """
        return (self.radial_type,) + tuple(getattr(self, name) for name in self.params())

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RadialGrid):
            return NotImplemented
        return self.key == other.key

    def __repr__(self) -> str:
        args = ", ".join(f"{name}={getattr(self, name)!r}" for name in self.params())
        return f"{type(self).__name__}({args})"

    # -- extent --------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int]:
        """``(nr,)``, the radial axis of a field."""
        return (self.nr,)

    @property
    def domain(self) -> str:
        """``"ball"``, ``"half-line"``, ``"shell"`` or ``"exterior"``, from the domain bounds."""
        if self.r_inner == 0.0:
            return "half-line" if math.isinf(self.r_outer) else "ball"
        return "exterior" if math.isinf(self.r_outer) else "shell"

    # -- nodes and weights ---------------------------------------------------

    def _nodes_and_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(r, w)`` in float64, both of shape ``(nr,)``; what a concrete rule defines."""
        raise NotImplementedError(f"{type(self).__name__} does not define its nodes")

    @property
    def r(self) -> torch.Tensor:
        """The radial nodes, shape ``(nr,)``, increasing, in float64."""
        return _radial_nodes(self)[0]

    @property
    def quad_weights(self) -> torch.Tensor:
        r"""Weights for :math:`\int f(r)\,dr`, shape ``(nr,)``, in float64; without the :math:`r^2` of the volume element."""
        return _radial_nodes(self)[1]

    @property
    def volume_weights(self) -> torch.Tensor:
        r"""Weights for :math:`\int f(r)\,r^2\,dr`, shape ``(nr,)``: :attr:`quad_weights` times :math:`r^2`."""
        r, w = _radial_nodes(self)
        return r**2 * w

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Plain-data representation: the rule under ``"radial"`` plus the parameters."""
        data = {"radial": self.radial_type}
        data.update({name: getattr(self, name) for name in self.params()})
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RadialGrid":
        """Inverse of :meth:`to_dict`."""
        if "radial" not in data:
            raise ValueError("radial grid dict is missing 'radial'")
        return as_radial_grid(data["radial"], **{k: v for k, v in data.items() if k != "radial"})


@dataclass(frozen=True, eq=False)
class GeometricRadialGrid(RadialGrid):
    r"""
    Nodes in geometric progression away from the inner boundary, with trapezoidal weights.

    The distance :math:`d = r - r_\mathrm{inner}` is sampled uniformly in :math:`\log d`
    between ``r_min - r_inner`` and ``r_max - r_inner``, so the nodes crowd towards the
    inner boundary and thin out geometrically away from it, and a few dozen nodes cover
    several decades in radius. The weights are the trapezoidal rule in :math:`\log d`
    times the Jacobian :math:`d`.

    The rule integrates only over ``[r_min, r_max]``. The gaps to the domain bounds, down
    to ``r_inner`` and up to ``r_outer``, are not sampled, so an integrand with mass there
    is truncated; on the half-line and the exterior domain that is the cost of an
    unbounded domain, and is why ``r_min`` and ``r_max`` should span the decades the field
    lives on.

    Parameters
    ----------
    nr : int
        Number of nodes, at least 2.
    r_min, r_max : float
        The innermost and the outermost node, ``r_inner < r_min < r_max <= r_outer``.
    r_inner : float, optional
        Inner domain bound, by default 0.
    r_outer : float, optional
        Outer domain bound, by default ``inf``.
    """

    radial_type: ClassVar[str] = "geometric"

    nr: int
    r_min: float
    r_max: float
    r_inner: float = 0.0
    r_outer: float = math.inf

    def __post_init__(self):
        super().__post_init__()
        if self.nr < 2:
            raise ValueError(f"a geometric radial grid needs at least 2 nodes, got nr={self.nr}")
        for name in ("r_min", "r_max"):
            _as_float(self, name)
            # nodes, unlike the domain bound r_outer, must be finite
            if math.isinf(getattr(self, name)):
                raise ValueError(f"{name} must be finite, got {getattr(self, name)}; only the domain bound r_outer may be infinite")
        if not self.r_inner < self.r_min < self.r_max <= self.r_outer:
            raise ValueError(f"expected r_inner < r_min < r_max <= r_outer, got r_inner={self.r_inner}, r_min={self.r_min}, r_max={self.r_max}, r_outer={self.r_outer}")

    def _nodes_and_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # trapezoidal rule in t = log d, then dr = dd = d dt
        t = torch.linspace(math.log(self.r_min - self.r_inner), math.log(self.r_max - self.r_inner), self.nr, dtype=torch.float64)
        h = (t[-1] - t[0]) / (self.nr - 1)
        wt = torch.full_like(t, h)
        wt[0] = wt[-1] = 0.5 * h
        d = torch.exp(t)
        return self.r_inner + d, d * wt


@dataclass(frozen=True, eq=False)
class LegendreGaussRadialGrid(RadialGrid):
    r"""
    Legendre-Gauss nodes and weights on a bounded domain ``[r_inner, r_outer]``.

    Integrates polynomials in :math:`r` of degree up to :math:`2\,n_r - 1` exactly, so the
    volume integral :math:`\int f\,r^2\,dr` of a polynomial of degree up to
    :math:`2\,n_r - 3`. The nodes are interior: neither bound is sampled.

    Parameters
    ----------
    nr : int
        Number of nodes.
    r_inner : float
        Inner domain bound, 0 for a ball.
    r_outer : float
        Outer domain bound, finite.
    """

    radial_type: ClassVar[str] = "legendre-gauss"

    nr: int
    r_inner: float
    r_outer: float

    def __post_init__(self):
        super().__post_init__()
        if math.isinf(self.r_outer):
            raise ValueError("a Legendre-Gauss radial grid needs a finite r_outer")

    def _nodes_and_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r, w = legendre_gauss_weights(self.nr, self.r_inner, self.r_outer)
        return r.to(torch.float64), w.to(torch.float64)


def radial_grid_types() -> Tuple[str, ...]:
    """Names of all registered radial rules, in registration order."""
    return tuple(_RADIAL_REGISTRY)


def as_radial_grid(spec: Union[RadialGrid, str, Type[RadialGrid]], **params: Any) -> RadialGrid:
    """
    Construct a radial grid from the name of its rule and its parameters.

    The radial counterpart of :func:`~torch_harmonics.grid.as_grid`, with a registry of its own,
    so a name such as ``"legendre-gauss"`` means the angular grid there and the radial
    rule here.

    Examples
    --------
    >>> from torch_harmonics import as_radial_grid
    >>> as_radial_grid("geometric", nr=32, r_min=0.1, r_max=100.0)
    GeometricRadialGrid(nr=32, r_min=0.1, r_max=100.0, r_inner=0.0, r_outer=inf)
    """
    if isinstance(spec, RadialGrid):
        if params:
            raise ValueError(f"parameters {sorted(params)} cannot be applied to the descriptor {spec!r}")
        return spec
    if isinstance(spec, type) and issubclass(spec, RadialGrid):
        cls = spec
    elif isinstance(spec, str):
        if spec not in _RADIAL_REGISTRY:
            message = f"Unknown radial grid type '{spec}', expected one of {list(_RADIAL_REGISTRY)}"
            close = difflib.get_close_matches(spec, _RADIAL_REGISTRY, n=1)
            if close:
                message += f". Did you mean '{close[0]}'?"
            raise ValueError(message)
        cls = _RADIAL_REGISTRY[spec]
    else:
        raise ValueError(f"expected a RadialGrid, a RadialGrid subclass or a radial grid type name, got {type(spec).__name__}")
    unknown = sorted(set(params) - set(cls.params()))
    if unknown:
        raise ValueError(f"{cls.__name__} takes parameters {list(cls.params())}, got unknown {unknown}")
    return cls(**params)


@dataclass(frozen=True, eq=False)
class SphericalProductGrid:
    r"""
    The direct product of a radial grid with a sampling of the sphere.

    Samples :math:`\{r_k\} \times \{x_i\}`: the angular grid ``angular`` repeated on every
    radial node of ``radial``. A field on it is laid out ``(..., nr, *angular.shape)``, so
    :attr:`shape` is ``(nr, *angular.shape)``.

    Every layer on the sphere accepts a product in place of its angular grid and acts on
    each shell alike, treating the radius as one more batch axis: an SHT transforms every
    shell, a DISCO convolution convolves every shell with the same filter. A layer taking
    an input and an output grid needs both on the same radial grid. Layers that couple the
    shells, such as a radial transform, take the product and use both factors.

    Not a :class:`~torch_harmonics.grid.PointSetS2`, deliberately: its points are not on the
    sphere, and a layer that read its :attr:`shape` or :attr:`quad_weights` as those of an
    angular grid would be silently wrong.

    Parameters
    ----------
    angular : PointSetS2
        The sampling of each shell.
    radial : RadialGrid
        The radial nodes.

    Examples
    --------
    >>> import torch_harmonics as th
    >>> grid = th.SphericalProductGrid(th.as_grid("legendre-gauss", nlat=32, nlon=64), th.as_radial_grid("legendre-gauss", nr=16, r_inner=1.0, r_outer=2.0))
    >>> grid.shape
    (16, 32, 64)
    """

    angular: PointSetS2
    radial: RadialGrid

    def __post_init__(self):
        if isinstance(self.angular, GridShardS2):
            raise TypeError("angular must be the global descriptor, not a shard of one")
        require_point_set(self.angular, "angular")
        if not isinstance(self.radial, RadialGrid):
            raise TypeError(f"radial must be a RadialGrid, got {type(self.radial).__name__}. Build one with as_radial_grid(...).")

    @property
    def key(self) -> Tuple[Any, ...]:
        """
        Canonical, hashable identity of this descriptor.

        Made of the two factors' keys; it defines both equality and hashing.
        """
        return ("spherical-product", self.angular.key, self.radial.key)

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SphericalProductGrid):
            return NotImplemented
        return self.key == other.key

    def __repr__(self) -> str:
        return f"SphericalProductGrid(angular={self.angular!r}, radial={self.radial!r})"

    @property
    def nr(self) -> int:
        """Number of radial nodes."""
        return self.radial.nr

    @property
    def npoints(self) -> int:
        """Total number of sample points, ``nr * angular.npoints``."""
        return self.radial.nr * self.angular.npoints

    @property
    def shape(self) -> Tuple[int, ...]:
        """Trailing shape of a field on this grid, ``(nr, *angular.shape)``."""
        return (self.radial.nr, *self.angular.shape)

    @property
    def quad_weights(self) -> torch.Tensor:
        r"""
        Volume weight of every point, shape ``(nr, angular.npoints)``.

        :math:`r_k^2\,w_k` times the angular weight, so that
        ``(f * grid.quad_weights).sum((-2, -1))`` approximates :math:`\int f\,dV` for a
        field whose angular axes are flattened, as :attr:`PointSetS2.quad_weights
        <torch_harmonics.grid.PointSetS2.quad_weights>` are.
        """
        return self.radial.volume_weights.unsqueeze(-1) * self.angular.quad_weights.to(torch.float64).unsqueeze(0)

    def to_dict(self) -> Dict[str, Any]:
        """Plain-data representation, nesting the two factors' own."""
        return {"grid": "spherical-product", "angular": self.angular.to_dict(), "radial": self.radial.to_dict()}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SphericalProductGrid":
        """Inverse of :meth:`to_dict`."""
        if data.get("grid") != "spherical-product":
            raise ValueError(f"not a spherical product grid dict: {data!r}")
        return SphericalProductGrid(PointSetS2.from_dict(data["angular"]), RadialGrid.from_dict(data["radial"]))


def split_radial(grid: Any) -> Tuple[Any, Optional[RadialGrid]]:
    """
    ``(angular, radial)`` of a product, ``(grid, None)`` of anything else.

    For a layer on the sphere taking a grid argument: it builds on the angular grid, and
    keeps the radial one only to state its field shape.
    """
    if isinstance(grid, SphericalProductGrid):
        return grid.angular, grid.radial
    return grid, None


def split_radial_pair(grid_in: Any, grid_out: Any) -> Tuple[Any, Any, Optional[RadialGrid]]:
    """
    :func:`split_radial` of an input and an output grid, which must share their radial grid.

    A layer on the sphere maps every shell to the same shell, so it cannot change the
    radial grid, and cannot take a product on one side only.
    """
    angular_in, radial_in = split_radial(grid_in)
    angular_out, radial_out = split_radial(grid_out)
    if radial_in != radial_out:
        raise ValueError(
            f"grid_in and grid_out must be on the same radial grid: this layer acts on each shell alike and cannot change the radial sampling. "
            f"Got {radial_in!r} and {radial_out!r}."
        )
    return angular_in, angular_out, radial_in


def fold_radius(x: torch.Tensor) -> torch.Tensor:
    """
    ``(batch, channels, nr, *spatial)`` to ``(batch * nr, channels, *spatial)``.

    For a layer that mixes channels and takes ``(batch, channels, *spatial)``: the radius
    moves into the batch. A copy, unless ``x`` came out of :func:`unfold_radius`, whose
    result already has this memory order, so a chain of such layers copies only once.
    """
    return x.movedim(2, 1).flatten(0, 1)


def unfold_radius(x: torch.Tensor, nr: int) -> torch.Tensor:
    """
    Inverse of :func:`fold_radius`, ``(batch * nr, channels, *spatial)`` to ``(batch, channels, nr, *spatial)``.

    A view, not a copy: the result keeps the folded memory order, which the next
    :func:`fold_radius` then gets for free. Elementwise ops in between preserve it.
    """
    return x.unflatten(0, (-1, nr)).movedim(1, 2)
