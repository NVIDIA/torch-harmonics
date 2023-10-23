# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
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

import numpy as np

def clm(l, m):
    """
    defines the normalization factor to orthonormalize the Spherical Harmonics
    """
    return np.sqrt((2*l + 1) / 4 / np.pi) * np.sqrt(np.math.factorial(l-m) / np.math.factorial(l+m))

def legpoly(mmax, lmax, x, norm="ortho", inverse=False, csphase=True):
    r"""
    Computes the values of (-1)^m c^l_m P^l_m(x) at the positions specified by x.
    The resulting tensor has shape (mmax, lmax, len(x)). The Condon-Shortley Phase (-1)^m
    can be turned off optionally.

    method of computation follows
    [1] Schaeffer, N.; Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Rapp, R.H.; A Fortran Program for the Computation of Gravimetric Quantities from High Degree Spherical Harmonic Expansions, Ohio State University Columbus; report; 1982;
        https://apps.dtic.mil/sti/citations/ADA123406
    [3] Schrama, E.; Orbit integration based upon interpolated gravitational gradients
    """

    # compute the tensor P^m_n:
    nmax = max(mmax,lmax)
    vdm = np.zeros((nmax, nmax, len(x)), dtype=np.float64)
        
    norm_factor = 1. if norm == "ortho" else np.sqrt(4 * np.pi)
    norm_factor = 1. / norm_factor if inverse else norm_factor

    # initial values to start the recursion
    vdm[0,0,:] = norm_factor / np.sqrt(4 * np.pi)

    # fill the diagonal and the lower diagonal
    for l in range(1, nmax):
        vdm[l-1, l, :] = np.sqrt(2*l + 1) * x * vdm[l-1, l-1, :]
        vdm[l, l, :] = np.sqrt( (2*l + 1) * (1 + x) * (1 - x) / 2 / l ) * vdm[l-1, l-1, :]

    # fill the remaining values on the upper triangle and multiply b
    for l in range(2, nmax):
        for m in range(0, l-1):
            vdm[m, l, :] = x * np.sqrt((2*l - 1) / (l - m) * (2*l + 1) / (l + m)) * vdm[m, l-1, :] \
                            - np.sqrt((l + m - 1) / (l - m) * (2*l + 1) / (2*l - 3) * (l - m - 1) / (l + m)) * vdm[m, l-2, :]

    if norm == "schmidt":
        for l in range(0, nmax):
            if inverse:
                vdm[:, l, : ] = vdm[:, l, : ] * np.sqrt(2*l + 1)
            else:
                vdm[:, l, : ] = vdm[:, l, : ] / np.sqrt(2*l + 1)

    vdm = vdm[:mmax, :lmax]

    if csphase:
        for m in range(1, mmax, 2):
            vdm[m] *= -1

    return vdm

def _precompute_legpoly(mmax, lmax, t, norm="ortho", inverse=False, csphase=True):
    r"""
    Computes the values of (-1)^m c^l_m P^l_m(\cos \theta) at the positions specified by t (theta).
    The resulting tensor has shape (mmax, lmax, len(x)). The Condon-Shortley Phase (-1)^m
    can be turned off optionally.

    method of computation follows
    [1] Schaeffer, N.; Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Rapp, R.H.; A Fortran Program for the Computation of Gravimetric Quantities from High Degree Spherical Harmonic Expansions, Ohio State University Columbus; report; 1982;
        https://apps.dtic.mil/sti/citations/ADA123406
    [3] Schrama, E.; Orbit integration based upon interpolated gravitational gradients
    """

    return legpoly(mmax, lmax, np.cos(t), norm=norm, inverse=inverse, csphase=csphase)

def _precompute_dlegpoly(mmax, lmax, t, norm="ortho", inverse=False, csphase=True):
    r"""
    Computes the values of the derivatives $\frac{d}{d \theta} P^m_l(\cos \theta)$
    at the positions specified by t (theta), as well as $\frac{1}{\sin \theta} P^m_l(\cos \theta)$,
    needed for the computation of the vector spherical harmonics. The resulting tensor has shape
    (2, mmax, lmax, len(t)).

    computation follows
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    pct = _precompute_legpoly(mmax+1, lmax+1, t, norm=norm, inverse=inverse, csphase=False)

    dpct = np.zeros((2, mmax, lmax, len(t)), dtype=np.float64)

    # fill the derivative terms wrt theta
    for l in range(0, lmax):

        # m = 0
        dpct[0, 0, l] = - np.sqrt(l*(l+1)) * pct[1, l]

        # 0 < m < l
        for m in range(1, min(l, mmax)):
            dpct[0, m, l] = 0.5 * ( np.sqrt((l+m)*(l-m+1)) * pct[m-1, l] - np.sqrt((l-m)*(l+m+1)) * pct[m+1, l] )

        # m == l
        if mmax > l:
            dpct[0, l, l] = np.sqrt(l/2) * pct[l-1, l]

        # fill the - 1j m P^m_l / sin(phi). as this component is purely imaginary,
        # we won't store it explicitly in a complex array
        for m in range(1, min(l+1, mmax)):
            # this component is implicitly complex
            # we do not divide by m here as this cancels with the derivative of the exponential
            dpct[1, m, l] = 0.5 * np.sqrt((2*l+1)/(2*l+3)) * \
                ( np.sqrt((l-m+1)*(l-m+2)) * pct[m-1, l+1] + np.sqrt((l+m+1)*(l+m+2)) * pct[m+1, l+1] )

    if csphase:
        for m in range(1, mmax, 2):
            dpct[:, m] *= -1

    return dpct