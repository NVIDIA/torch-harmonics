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

from torch_harmonics.grid import GridS2


def truncate_sht(grid: GridS2, lmax: Optional[int] = None, mmax: Optional[int] = None) -> Tuple[int, int]:
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
       * - ``"equiangular"`` / ``"equiangular-trapezoidal"``
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
    :attr:`~torch_harmonics.grid.GridS2.max_azimuthal_order`), which reports what
    the grid can represent. This routine owns the *policy* on top of that: applying
    user overrides, enforcing the triangular truncation, and warning where the
    default changed.

    Parameters
    ----------
    grid : GridS2
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
    >>> from torch_harmonics import truncate_sht
    >>> from torch_harmonics.grid import as_grid
    >>> truncate_sht(as_grid("legendre-gauss", (128, 256)))
    (128, 128)
    >>> truncate_sht(as_grid("lobatto", (128, 256)))
    (127, 127)
    >>> truncate_sht(as_grid("legendre-gauss", (128, 256)), lmax=32)
    (32, 32)
    """

    # fall back to what the grid can actually represent. `is None` rather than a
    # falsy test: lmax=0 is meaningless but should not silently become the default.
    if lmax is None:
        lmax = grid.max_exact_degree
        if grid.grid_type in ("equiangular", "equiangular-trapezoidal"):
            warnings.warn(
                "Default SHT truncation changed in v0.9.0: equiangular/equiangular-trapezoidal grids now truncate to (nlat+1)//2. " "Specify lmax explicitly to override.",
                UserWarning,
                stacklevel=2,
            )
    if mmax is None:
        mmax = grid.max_azimuthal_order

    # perform triangular truncation
    lmax = min(lmax, mmax)
    mmax = lmax

    return lmax, mmax
