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

from typing import Optional, Tuple

def _truncate_lmax(nlat: int, grid: Optional[str]="equiangular") -> int:
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
        Grid type ("legendre-gauss", "lobatto", "equiangular", "equidistant"), by default "equiangular"

    Returns
    -------
    int
        Maximum spherical harmonic degree (non-inclusive)
    """
    if grid == "legendre-gauss":
        return nlat
    elif grid == "lobatto":
        return nlat - 1
    elif grid in ["equiangular", "equidistant"]:
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

def truncate_sht(nlat: int, nlon: int, lmax: Optional[int]=None, mmax: Optional[int]=None, grid: Optional[str]="equiangular") -> Tuple[int, int]:
    """
    Truncate the maximum spherical harmonic degree and azimuthal harmonic degree based on the latitude and longitude grids.

    Parameters
    ----------
    nlat : int
        Number of latitude points
    nlon : int
        Number of longitude points
    lmax : int, optional
        User-defined maximum spherical harmonic degree (non-inclusive)
        If not provided, the maximum degree is determined based on the latitude grid.
    mmax : int, optional
        User-defined maximum azimuthal harmonic degree (non-inclusive)
        If not provided, the maximum degree is determined based on the longitude grid.
    grid : str, optional
        Grid type ("legendre-gauss", "lobatto", "equiangular", "equidistant"), by default "equiangular"

    Returns
    -------
    lmax : int
        Maximum spherical harmonic degree (non-inclusive)
    mmax : int
        Maximum azimuthal harmonic degree (non-inclusive)
    """

    # determine the maximum degrees based on user-defined values or the default values based on the grid type
    lmax = lmax or _truncate_lmax(nlat, grid)
    mmax = mmax or _truncate_mmax(nlon)

    # perform triangular truncation
    lmax = min(lmax, mmax)
    mmax = lmax

    return lmax, mmax