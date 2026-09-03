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

import warnings
from typing import Optional, Tuple

from torch_harmonics.grid import RegularGridS2, require_regular_grid
from torch_harmonics.quadrature import compute_theta_cutoff


def truncate_sht(grid: RegularGridS2, lmax: Optional[int] = None, mmax: Optional[int] = None) -> Tuple[int, int]:
    r"""
    Determine the maximum spherical harmonic degree and order for an SHT based
    on the spatial grid.

    When ``lmax`` or ``mmax`` are not provided, they are inferred from the grid
    resolution.  The default truncation for each grid type is chosen so that the
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

    Finally, a **triangular truncation** is applied:
    :math:`l_{\max} = m_{\max} = \min(l_{\max},\, m_{\max})`, so that every
    retained degree has a full set of orders.

    The bounds themselves come from the grid descriptor
    (:attr:`~torch_harmonics.grid.GridS2.max_exact_degree` and
    :attr:`~torch_harmonics.grid.RegularGridS2.max_azimuthal_order`), which reports what
    the grid can represent. This routine owns the *policy* on top of that: applying
    user overrides, enforcing the triangular truncation, and warning where the
    default changed.

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
        If not provided, set to the Nyquist limit
        :math:`\lfloor N_\lambda / 2 \rfloor + 1`.

    Returns
    -------
    lmax : int
        Maximum spherical harmonic degree (non-inclusive).
    mmax : int
        Maximum azimuthal harmonic order (non-inclusive).

    Examples
    --------
    >>> from torch_harmonics import as_grid, truncate_sht
    >>> truncate_sht(as_grid("legendre-gauss", nlat=128, nlon=256))
    (128, 128)
    >>> truncate_sht(as_grid("lobatto", nlat=128, nlon=256))
    (127, 127)
    >>> truncate_sht(as_grid("legendre-gauss", nlat=128, nlon=256), lmax=32)
    (32, 32)
    """

    # a shard has no spectral bounds of its own; say so with the migration message
    # rather than letting an AttributeError surface from deeper in
    grid = require_regular_grid(grid)

    # fall back to what the grid can actually represent. `is None` rather than a
    # falsy test: lmax=0 is meaningless but should not silently become the default.
    if lmax is None:
        lmax = grid.max_exact_degree
        if grid.grid_type in ("equiangular", "trapezoidal"):
            warnings.warn(
                "Default SHT truncation changed in v0.9.0: equiangular/trapezoidal grids now truncate to (nlat+1)//2. " "Specify lmax explicitly to override.",
                UserWarning,
                stacklevel=2,
            )
    if mmax is None:
        mmax = grid.max_azimuthal_order

    # perform triangular truncation
    lmax = min(lmax, mmax)
    mmax = lmax

    return lmax, mmax


def truncate_support(grid: RegularGridS2, theta_cutoff: Optional[float] = None, scale: Optional[float] = 1.0) -> float:
    r"""
    Determine the angular support radius of a localized operator on a grid.

    The spatial counterpart of :func:`truncate_sht`. Where that decides how far
    up in degree an SHT keeps, this decides how far out in angle the filter basis
    of a DISCO convolution or of neighborhood attention reaches. Both take the
    bound the grid can support, apply a user override if one is given, and warn
    when the default they pick differs from the one a previous release used.

    The default is one latitudinal node spacing of the grid, so that the basis
    functions of adjacent output points overlap and every output point sees more
    than the single latitude ring it sits on. That spacing is a fact about the
    node distribution, which the descriptor also reports as
    :attr:`~torch_harmonics.grid.GridS2.max_latitude_spacing`; the policy of
    turning it into a default, of rejecting a non-positive result, and of warning
    that the default moved, lives here.

    The spacing comes via :func:`torch_harmonics.quadrature.compute_theta_cutoff`
    rather than off the descriptor property, because the changed-default warning
    is raised there. The two agree by construction -- both are
    :func:`~torch_harmonics.quadrature.compute_latitude_spacing` of the same grid
    -- and ``test_default_is_one_grid_spacing`` pins that they keep agreeing.

    Parameters
    ----------
    grid : RegularGridS2
        Descriptor of the grid that sets the cutoff. This is the output grid of a
        forward transform and the input grid of a transpose one, mirroring which
        of the two is the coarser. It must be the global grid: a cutoff taken
        from a shard's own spacing would differ between ranks, and ranks
        disagreeing about the support of an operator is a correctness bug.
    theta_cutoff : float, optional
        Explicit cutoff in radians. If None (default), the grid's node spacing is
        used. Must be positive.
    scale : float, optional
        Multiplier applied to the default spacing, by default 1.0. Ignored when
        *theta_cutoff* is given, which is already a final value. Must leave the
        resulting radius positive.

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
        On grids whose node spacing is not uniform in :math:`\theta`, where the
        default differs from the ``pi / (nlat - 1)`` heuristic used before
        v0.9.3. Equiangular grids are unaffected and do not warn.

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
        grid = require_regular_grid(grid)
        radius = scale * compute_theta_cutoff(grid.nlat, grid=grid.grid_type)
        origin = f"scale={scale} times the grid spacing"
    else:
        radius = theta_cutoff
        origin = f"theta_cutoff={theta_cutoff}"

    # guard the value that is returned rather than the argument it came from: a
    # non-positive radius reaches the kernels the same way whichever route made it
    if radius <= 0.0:
        raise ValueError(f"Error, the angular support radius has to be positive, got {radius} from {origin}.")

    return radius
