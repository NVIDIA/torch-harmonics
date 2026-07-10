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


def _truncate_lmax(nlat: int, grid: Optional[str] = "equiangular") -> int:
    """
    Truncate the maximum spherical harmonic degree based on the latitude grid. The maximum degree
    corresponds to the maximum degree of associated Legendre polynomials that can be square-integrated
    exactly.

    | Grid Type           | Includes Poles? | Exactness       | Heuristic ($L_{\text{max}}$) |
    | :---                | :---:           | :---:           | :---:                        |
    | Legendre-Gauss (GL) | No              | $2N - 1$        | $N - 1$                      |
    | Gauss-Lobatto (GLL) | Yes             | $2N - 3$        | $N - 2$                      |
    | Equiangular (CC)    | Yes             | $\approx N - 1$ | $\approx N/2$                |

    Parameters
    ----------
    nlat : int
        Number of latitude points
    grid : str, optional
        Grid type (``"legendre-gauss"``, ``"lobatto"``, ``"equiangular"``, ``"equiangular-trapezoidal"``), by default ``"equiangular"``

    Returns
    -------
    int
        Maximum spherical harmonic degree (non-inclusive)
    """
    if grid == "legendre-gauss":
        return nlat
    elif grid == "lobatto":
        return nlat - 1
    elif grid in ["equiangular", "equiangular-trapezoidal"]:
        warnings.warn(
            "Default SHT truncation changed in v0.9.0: equiangular/equiangular-trapezoidal grids now truncate to (nlat+1)//2. " "Specify lmax explicitly to override.",
            UserWarning,
            stacklevel=2,
        )
        return (nlat + 1) // 2
    else:
        raise ValueError(f"Unknown grid type {grid}")


def _truncate_mmax(nlon: int) -> int:
    """
    Truncate the maximum azimuthal harmonic degree based on the longitude grid. This is the same as the
    Nyquist frequency.

    Parameters
    ----------
    nlon : int
        Number of longitude points

    Returns
    -------
    int
        Maximum azimuthal harmonic degree (non-inclusive)
    """
    return nlon // 2 + 1


def truncate_sht(nlat: int, nlon: int, lmax: Optional[int] = None, mmax: Optional[int] = None, grid: Optional[str] = "equiangular") -> Tuple[int, int]:
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

    Parameters
    ----------
    nlat : int
        Number of latitude points :math:`N_\theta`.
    nlon : int
        Number of longitude points :math:`N_\lambda`.
    lmax : int, optional
        User-defined maximum spherical harmonic degree (non-inclusive).
        If not provided, the maximum degree is determined from the latitude
        grid as shown in the table above.
    mmax : int, optional
        User-defined maximum azimuthal harmonic order (non-inclusive).
        If not provided, set to the Nyquist limit
        :math:`\lfloor N_\lambda / 2 \rfloor + 1`.
    grid : str, optional
        Grid type (``"legendre-gauss"``, ``"lobatto"``, ``"equiangular"``,
        ``"equiangular-trapezoidal"``), by default ``"equiangular"``.

    Returns
    -------
    lmax : int
        Maximum spherical harmonic degree (non-inclusive).
    mmax : int
        Maximum azimuthal harmonic order (non-inclusive).

    Examples
    --------
    >>> from torch_harmonics import truncate_sht
    >>> truncate_sht(128, 256, grid="legendre-gauss")
    (128, 128)
    >>> truncate_sht(128, 256, grid="lobatto")
    (127, 127)
    >>> truncate_sht(128, 256, grid="equiangular")
    (64, 64)
    """

    # determine the maximum degrees based on user-defined values or the default values based on the grid type
    lmax = lmax or _truncate_lmax(nlat, grid)
    mmax = mmax or _truncate_mmax(nlon)

    # perform triangular truncation
    lmax = min(lmax, mmax)
    mmax = lmax

    return lmax, mmax
