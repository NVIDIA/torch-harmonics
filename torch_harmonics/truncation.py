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

import math
import warnings
from numbers import Integral
from typing import NamedTuple, Optional

from torch_harmonics.grid import PointSetS2, RegularGridS2, require_point_set, require_regular_grid


def _warn_if_not_spectrally_accurate(grid: RegularGridS2) -> None:
    """
    Warn when an SHT is built on a grid whose quadrature does not round-trip.

    Shared by the serial and distributed transforms, which all construct from a grid and
    would otherwise each carry a copy. The measured errors are quoted at the resolution
    they were measured at, ``nlat = 64``, rather than paired with the lmax of the grid in
    hand.
    """
    if grid.is_spectrally_accurate:
        return
    warnings.warn(
        f"grid '{grid.grid_type}' must not be used for spherical harmonic transforms. Its quadrature converges only "
        "algebraically, so the associated Legendre polynomials are not discretely orthogonal on it and the transform does not "
        "round-trip. Measured relative round-trip error at nlat=64: 2e-4 at lmax=2, 2e-2 at lmax=8, and 6.7e-1 at lmax=32, "
        "the default there -- i.e. the result is of the same order as the signal. Only lmax=1 is exact. "
        "Use 'equiangular', 'legendre-gauss' or 'lobatto' instead. This grid remains appropriate for plain quadrature "
        "(QuadratureS2) and for the localized operators (DISCO convolutions, neighborhood attention), which make no "
        "orthogonality assumption.",
        UserWarning,
        # past this helper and the transform's __init__, to the line that built the transform
        stacklevel=3,
    )


class _SHTTruncation(NamedTuple):
    lmax: int
    mmax: int
    lmmax: Optional[int]


class _SHTTruncationMixin:
    """Expose a transform's single resolved truncation through its usual attributes."""

    @property
    def lmax(self) -> int:
        return self._trunc.lmax

    @property
    def mmax(self) -> int:
        return self._trunc.mmax

    @property
    def lmmax(self) -> Optional[int]:
        return self._trunc.lmmax


def truncate_sht(grid: RegularGridS2, lmax: Optional[int] = None, mmax: Optional[int] = None, lmmax: Optional[int] = None) -> _SHTTruncation:
    r"""
    Resolve the three non-inclusive spectral bounds of a regular-grid SHT.

    Missing ``lmax`` and ``mmax`` bounds are inferred from the grid resolution;
    an inferred ``mmax`` is limited by ``lmax`` so it cannot request nonexistent
    orders. The default truncation for each grid type is chosen so that the
    associated Legendre polynomials up to the returned degree can be
    square-integrated exactly by the corresponding quadrature rule:

    .. list-table:: Default latitudinal truncation :math:`l_{\max}` for :math:`N_\theta` latitude points
       :header-rows: 1
       :widths: 30 15 25 30

       * - Grid type
         - Includes poles?
         - Quadrature exactness
         - Default :math:`l_{\max}`
       * - ``"legendre-gauss"``
         - No
         - :math:`2 N_\theta - 1`
         - :math:`N_\theta`
       * - ``"lobatto"``
         - Yes
         - :math:`2 N_\theta - 3`
         - :math:`N_\theta - 1`
       * - ``"equiangular"`` / ``"trapezoidal"``
         - Yes
         - :math:`\approx N_\theta - 1`
         - :math:`\lfloor (N_\theta + 1) / 2 \rfloor`

    The default longitudinal truncation is the Nyquist limit of the uniform
    longitude grid: :math:`m_{\max} = \lfloor N_\lambda / 2 \rfloor + 1`.

    With no explicit bounds, the default remains triangular:
    :math:`l_{\max} = m_{\max} = \min(l_{\max},\, m_{\max})`.
    Otherwise degree and order limits are independent. A mode is retained when
    :math:`m < m_{\max}`, :math:`m \le l < l_{\max}`, and, when ``lmmax`` is
    given, :math:`l - m < lm_{\max}`. ``lmmax=None`` leaves the upper
    :math:`l-m` bandwidth unrestricted. ``lmmax=mmax`` gives the classical
    rhomboidal case when ``lmax`` does not clip it.

    The bounds are taken from
    :attr:`~torch_harmonics.grid.PointSetS2.max_exact_degree` and
    :attr:`~torch_harmonics.grid.RegularGridS2.max_azimuthal_order`.

    Parameters
    ----------
    grid : RegularGridS2
        Descriptor of the spatial grid the transform operates on.
    lmax : int, optional
        User-defined maximum spherical harmonic degree (non-inclusive).
        If not provided, the maximum degree is determined from the latitude
        grid as shown in the table above.
    mmax : int, optional
        User-defined maximum azimuthal harmonic order (non-inclusive).
        If not provided, use the smaller of ``lmax`` and the Nyquist limit
        :math:`\lfloor N_\lambda / 2 \rfloor + 1`.
    lmmax : int, optional
        Maximum degree-minus-order bandwidth (non-inclusive). ``None`` leaves
        this bandwidth unrestricted.

    Returns
    -------
    _SHTTruncation
        Resolved ``(lmax, mmax, lmmax)`` bounds.

    Examples
    --------
    >>> from torch_harmonics import as_grid, truncate_sht
    >>> truncate_sht(as_grid("legendre-gauss", nlat=128, nlon=256))
    _SHTTruncation(lmax=128, mmax=128, lmmax=None)
    >>> truncate_sht(as_grid("lobatto", nlat=128, nlon=256))
    _SHTTruncation(lmax=127, mmax=127, lmmax=None)
    >>> truncate_sht(as_grid("legendre-gauss", nlat=128, nlon=256), lmax=32)
    _SHTTruncation(lmax=32, mmax=32, lmmax=None)
    >>> truncate_sht(as_grid("legendre-gauss", nlat=128, nlon=256), lmax=85, mmax=43, lmmax=43)
    _SHTTruncation(lmax=85, mmax=43, lmmax=43)
    """

    # a shard has no spectral bounds of its own; say so with the migration message
    # rather than letting an AttributeError surface from deeper in
    grid = require_regular_grid(grid)

    for name, value in (("lmax", lmax), ("mmax", mmax), ("lmmax", lmmax)):
        if value is not None and (not isinstance(value, Integral) or isinstance(value, bool) or value <= 0):
            raise ValueError(f"{name} must be a positive integer, got {value!r}")

    # Resolve grid defaults without clamping any explicit spectral bound.
    default_lmax = grid.max_exact_degree
    default_mmax = grid.max_azimuthal_order
    no_bounds = lmax is None and mmax is None and lmmax is None
    if lmax is None:
        lmax = default_lmax
        if grid.grid_type in ("equiangular", "trapezoidal"):
            warnings.warn(
                "Default SHT truncation changed in v0.9.0: equiangular/trapezoidal grids now truncate to (nlat+1)//2. " "Specify lmax explicitly to override.",
                UserWarning,
                stacklevel=2,
            )
    if mmax is None:
        mmax = min(default_mmax, lmax)

    if no_bounds:
        lmax = min(lmax, mmax)
        mmax = lmax

    if mmax > lmax:
        raise ValueError(f"mmax={mmax} exceeds lmax={lmax}; orders m >= lmax cannot be represented")

    return _SHTTruncation(lmax, mmax, lmmax)


def _warn_if_default_moved(grid: PointSetS2) -> None:
    r"""
    Announce that the default support radius differs from the pre-v1.0.0 heuristic.

    Lives here rather than in :func:`torch_harmonics.quadrature.compute_theta_cutoff`
    because it is policy, not a fact about the nodes: the descriptor property stays
    silent, and only the routine that *chose* to default announces it. That split is
    pinned by ``test_the_descriptor_does_not_warn``.

    Only the latitude/longitude grids can have moved, since they are the only ones
    that existed before the change; a ragged grid has no ``pi / (nlat - 1)`` past to
    differ from and is passed over rather than warned about spuriously.
    """
    if not isinstance(grid, RegularGridS2):
        return

    spacing = grid.max_node_spacing

    # compare the numbers rather than trusting is_uniform_in_theta, so that a grid
    # family which sets that flag wrongly is caught here too
    legacy = math.pi / float(grid.nlat - 1)
    if abs(spacing - legacy) <= 1e-9 * legacy:
        return

    # two independent reasons the default can have moved, and a grid can hit both
    reasons = []
    if abs(grid.max_latitude_spacing - legacy) > 1e-9 * legacy:
        reasons.append(f"its nodes are not uniform in theta, so the latitudinal spacing is {grid.max_latitude_spacing:.6f} rather than pi/(nlat-1)")
    if grid.max_longitude_spacing > grid.max_latitude_spacing:
        reasons.append(
            f"its in-ring spacing ({grid.max_longitude_spacing:.6f}) exceeds its latitudinal spacing ({grid.max_latitude_spacing:.6f}), and the support has to reach the coarser neighbour"
        )

    consequence = "the previous value under-covered the grid" if spacing > legacy else "the previous value was wider than the grid warrants"
    warnings.warn(
        f"Default theta_cutoff changed in v1.0.0: on the '{grid.grid_type}' grid at nlat={grid.nlat}, nlon={grid.nlon} it is now "
        f"one node spacing ({spacing:.6f}) rather than pi/(nlat-1) ({legacy:.6f}), because " + " and ".join(reasons) + f". {consequence}. "
        "Specify theta_cutoff explicitly to override.",
        UserWarning,
        stacklevel=4,
    )


def truncate_support(grid: PointSetS2, theta_cutoff: Optional[float] = None, scale: Optional[float] = 1.0) -> float:
    r"""
    Determine the angular support radius of a localized operator on a grid.

    The spatial counterpart of :func:`truncate_sht`: it decides how far the filter
    of a DISCO convolution or neighborhood attention reaches. The default is one
    node spacing of the grid,
    :attr:`~torch_harmonics.grid.PointSetS2.max_node_spacing` (the distance to the
    coarsest neighbour, along or across rings), so that the supports of adjacent
    output points overlap.

    Parameters
    ----------
    grid : PointSetS2
        Descriptor of the grid that sets the cutoff. This is the output grid of a
        forward transform and the input grid of a transpose one, mirroring which
        of the two is the coarser. Must be a global grid, not a shard.
    theta_cutoff : float, optional
        Explicit cutoff in radians. If None (default), the grid's node spacing is
        used. Must be positive.
    scale : float, optional
        Multiplier applied to the default spacing, by default 1.0. Ignored when
        *theta_cutoff* is given.

    Returns
    -------
    float
        Cutoff angle in radians, always positive.

    Raises
    ------
    ValueError
        If the resulting radius is not positive, whether it came from an explicit
        *theta_cutoff* or from a non-positive *scale* applied to the default.

    Warns
    -----
    UserWarning
        On a latitude-longitude grid whose default differs from the
        ``pi / (nlat - 1)`` heuristic used before v1.0.0. That happens when its
        nodes are not uniform in :math:`\theta`, or when its in-ring spacing
        exceeds its latitudinal one -- on an equiangular grid, when ``nlon`` is
        below about ``2 * (nlat - 1)``. Ragged grids had no earlier default and do
        not warn.

    See Also
    --------
    truncate_sht : The spectral counterpart.

    Examples
    --------
    >>> from torch_harmonics import as_grid
    >>> from torch_harmonics.truncation import truncate_support
    >>> round(truncate_support(as_grid("equiangular", nlat=64, nlon=128)), 6)
    0.049867
    >>> truncate_support(as_grid("equiangular", nlat=64, nlon=128), theta_cutoff=0.2)
    0.2
    """

    if theta_cutoff is None:
        # a support radius taken from a shard would differ between ranks
        grid = require_point_set(grid)
        # ask the descriptor, not the grid string. The default is one grid spacing, and
        # a grid's spacing is the distance to its coarsest *neighbour* -- which may lie
        # along a ring rather than across rings. It is latitudinal on an equiangular grid
        # at nlon = 2 nlat, but longitudinal on a Gauss grid at the same resolution, on
        # anything with nlon < 2 nlat, and on HEALPix by a factor of ~1.8.
        radius = scale * grid.max_node_spacing
        origin = f"scale={scale} times the grid spacing"
        _warn_if_default_moved(grid)
    else:
        radius = theta_cutoff
        origin = f"theta_cutoff={theta_cutoff}"

    # guard the value that is returned rather than the argument it came from: a
    # non-positive radius reaches the kernels the same way whichever route made it
    if radius <= 0.0:
        raise ValueError(f"Error, the angular support radius has to be positive, got {radius} from {origin}.")

    return radius
