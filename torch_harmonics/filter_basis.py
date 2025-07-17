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

import abc
from typing import Tuple, Union, Optional
import math

import torch

from torch_harmonics.cache import lru_cache


def _circle_dist(x1: torch.Tensor, x2: torch.Tensor):
    return torch.minimum(torch.abs(x1 - x2), torch.abs(2 * math.pi - torch.abs(x1 - x2)))


def _log_factorial(x: torch.Tensor):
    return torch.lgamma(x + 1)


def _factorial(x: torch.Tensor):
    return torch.exp(_log_factorial(x))


class FilterBasis(metaclass=abc.ABCMeta):
    """Abstract base class for a filter basis"""

    def __init__(
        self,
        kernel_shape: Union[int, Tuple[int], Tuple[int, int]],
    ):

        self.kernel_shape = kernel_shape

    @property
    @abc.abstractmethod
    def kernel_size(self):

        raise NotImplementedError

    # @abc.abstractmethod
    # def compute_vals(self, r: torch.Tensor, phi: torch.Tensor, r_cutoff: float):
    #     """
    #     Computes the values of the filter basis
    #     """
    #     raise NotImplementedError

    @abc.abstractmethod
    def compute_support_vals(self, r: torch.Tensor, phi: torch.Tensor, r_cutoff: float):

        raise NotImplementedError


@lru_cache(typed=True, copy=False)
def get_filter_basis(kernel_shape: Union[int, Tuple[int], Tuple[int, int]], basis_type: str) -> FilterBasis:
    """Factory function to generate the appropriate filter basis"""

    if basis_type == "piecewise linear":
        return PiecewiseLinearFilterBasis(kernel_shape=kernel_shape)
    elif basis_type == "morlet":
        return MorletFilterBasis(kernel_shape=kernel_shape)
    elif basis_type == "zernike":
        return ZernikeFilterBasis(kernel_shape=kernel_shape)
    else:
        raise ValueError(f"Unknown basis_type {basis_type}")


class PiecewiseLinearFilterBasis(FilterBasis):
    """Tensor-product basis on a disk constructed from piecewise linear basis functions."""

    def __init__(
        self,
        kernel_shape: Union[int, Tuple[int], Tuple[int, int]],
    ):

        if isinstance(kernel_shape, int):
            kernel_shape = [kernel_shape]
        if len(kernel_shape) == 1:
            kernel_shape = [kernel_shape[0], 1]
        elif len(kernel_shape) != 2:
            raise ValueError(f"expected kernel_shape to be a list or tuple of length 1 or 2 but got {kernel_shape} instead.")

        super().__init__(kernel_shape=kernel_shape)

    @property
    def kernel_size(self):
        """Compute the number of basis functions in the kernel."""
        return (self.kernel_shape[0] // 2) * self.kernel_shape[1] + self.kernel_shape[0] % 2

    def _compute_support_vals_isotropic(self, r: torch.Tensor, phi: torch.Tensor, r_cutoff: float):

        # enumerator for basis function
        ikernel = torch.arange(self.kernel_size, device=r.device).reshape(-1, 1, 1)

        # collocation points
        nr = self.kernel_shape[0]
        dr = 2 * r_cutoff / (nr + 1)

        # compute the support
        if nr % 2 == 1:
            ir = ikernel * dr
        else:
            ir = (ikernel + 0.5) * dr

        # find the indices where the rotated position falls into the support of the kernel
        iidx = torch.argwhere(((r - ir).abs() <= dr) & (r <= r_cutoff))
        vals = 1 - (r[iidx[:, 1], iidx[:, 2]] - ir[iidx[:, 0], 0, 0]).abs() / dr

        return iidx, vals

    def _compute_support_vals_anisotropic(self, r: torch.Tensor, phi: torch.Tensor, r_cutoff: float):

        # enumerator for basis function
        ikernel = torch.arange(self.kernel_size, device=r.device).reshape(-1, 1, 1)

        # collocation points
        nr = self.kernel_shape[0]
        nphi = self.kernel_shape[1]
        dr = 2 * r_cutoff / (nr + 1)
        dphi = 2.0 * math.pi / nphi

        # disambiguate even and uneven cases and compute the support
        if nr % 2 == 1:
            ir = ((ikernel - 1) // nphi + 1) * dr
            iphi = ((ikernel - 1) % nphi) * dphi - math.pi
        else:
            ir = (ikernel // nphi + 0.5) * dr
            iphi = (ikernel % nphi) * dphi - math.pi

        # find the indices where the rotated position falls into the support of the kernel
        if nr % 2 == 1:
            # find the support
            cond_r = ((r - ir).abs() <= dr) & (r <= r_cutoff)
            cond_phi = (ikernel == 0) | (_circle_dist(phi, iphi).abs() <= dphi)
            # find indices where conditions are met
            iidx = torch.argwhere(cond_r & cond_phi)
            # compute the distance to the collocation points
            dist_r = (r[iidx[:, 1], iidx[:, 2]] - ir[iidx[:, 0], 0, 0]).abs()
            dist_phi = _circle_dist(phi[iidx[:, 1], iidx[:, 2]], iphi[iidx[:, 0], 0, 0])
            # compute the value of the basis functions
            vals  = 1 - dist_r / dr
            vals *= torch.where((iidx[:, 0] > 0), (1 - dist_phi / dphi), 1.0)

        else:
            # in the even case, the inner basis functions overlap into areas with a negative areas
            rn = -r
            phin = torch.where(phi + math.pi >= math.pi, phi - math.pi, phi + math.pi)
            # find the support
            cond_r = ((r - ir).abs() <= dr) & (r <= r_cutoff)
            cond_phi = _circle_dist(phi, iphi).abs() <= dphi
            cond_rn = ((rn - ir).abs() <= dr) & (rn <= r_cutoff)
            cond_phin = _circle_dist(phin, iphi) <= dphi
            # find indices where conditions are met
            iidx = torch.argwhere((cond_r & cond_phi) | (cond_rn & cond_phin))

            dist_r = (r[iidx[:, 1], iidx[:, 2]] - ir[iidx[:, 0], 0, 0]).abs()
            dist_phi = _circle_dist(phi[iidx[:, 1], iidx[:, 2]], iphi[iidx[:, 0], 0, 0])
            dist_rn = (rn[iidx[:, 1], iidx[:, 2]] - ir[iidx[:, 0], 0, 0]).abs()
            dist_phin = _circle_dist(phin[iidx[:, 1], iidx[:, 2]], iphi[iidx[:, 0], 0, 0])
            # compute the value of the basis functions
            vals = cond_r[iidx[:, 0], iidx[:, 1], iidx[:, 2]] * (1 - dist_r / dr)
            vals *= cond_phi[iidx[:, 0], iidx[:, 1], iidx[:, 2]] * (1 - dist_phi / dphi)
            valsn = cond_rn[iidx[:, 0], iidx[:, 1], iidx[:, 2]] * (1 - dist_rn / dr)
            valsn *= cond_phin[iidx[:, 0], iidx[:, 1], iidx[:, 2]] * (1 - dist_phin / dphi)
            vals += valsn

        return iidx, vals

    def compute_support_vals(self, r: torch.Tensor, phi: torch.Tensor, r_cutoff: float):

        if self.kernel_shape[1] > 1:
            return self._compute_support_vals_anisotropic(r, phi, r_cutoff=r_cutoff)
        else:
            return self._compute_support_vals_isotropic(r, phi, r_cutoff=r_cutoff)


class MorletFilterBasis(FilterBasis):
    """Morlet-style filter basis on the disk. A Gaussian is multiplied with a Fourier basis in x and y directions."""

    def __init__(
        self,
        kernel_shape: Union[int, Tuple[int], Tuple[int, int]],
    ):

        if isinstance(kernel_shape, int):
            kernel_shape = [kernel_shape, kernel_shape]
        if len(kernel_shape) != 2:
            raise ValueError(f"expected kernel_shape to be a list or tuple of 2 but got {kernel_shape} instead.")

        super().__init__(kernel_shape=kernel_shape)

    @property
    def kernel_size(self):

        return self.kernel_shape[0] * self.kernel_shape[1]

    def gaussian_window(self, r: torch.Tensor, width: float = 1.0):

        return 1 / (2 * math.pi * width**2) * torch.exp(-0.5 * r**2 / (width**2))

    def hann_window(self, r: torch.Tensor, width: float = 1.0):

        return torch.cos(0.5 * torch.pi * r / width) ** 2

    def compute_support_vals(self, r: torch.Tensor, phi: torch.Tensor, r_cutoff: float, width: float = 1.0):

        # enumerator for basis function
        ikernel = torch.arange(self.kernel_size, device=r.device).reshape(-1, 1, 1)
        nkernel = ikernel % self.kernel_shape[1]
        mkernel = ikernel // self.kernel_shape[1]

        # get relevant indices
        iidx = torch.argwhere((r <= r_cutoff) & torch.full_like(ikernel, True, dtype=torch.bool, device=r.device))

        # get corresponding r, phi, x and y coordinates
        r = r[iidx[:, 1], iidx[:, 2]] / r_cutoff
        phi = phi[iidx[:, 1], iidx[:, 2]]
        x = r * torch.sin(phi)
        y = r * torch.cos(phi)
        n = nkernel[iidx[:, 0], 0, 0]
        m = mkernel[iidx[:, 0], 0, 0]

        harmonic = torch.where(n % 2 == 1, torch.sin(torch.ceil(n / 2) * math.pi * x / width), torch.cos(torch.ceil(n / 2) * math.pi * x / width))
        harmonic *= torch.where(m % 2 == 1, torch.sin(torch.ceil(m / 2) * math.pi * y / width), torch.cos(torch.ceil(m / 2) * math.pi * y / width))

        # computes the envelope. To ensure that the curve is roughly 0 at the boundary, we rescale the Gaussian by 0.25
        vals = self.hann_window(r, width=width) * harmonic

        return iidx, vals


class ZernikeFilterBasis(FilterBasis):
    """Zernike polynomials which are defined on the disk. See https://en.wikipedia.org/wiki/Zernike_polynomials"""

    def __init__(
        self,
        kernel_shape: Union[int, Tuple[int]],
    ):

        if isinstance(kernel_shape, tuple) or isinstance(kernel_shape, list):
            kernel_shape = kernel_shape[0]
        if not isinstance(kernel_shape, int):
            raise ValueError(f"expected kernel_shape to be an integer but got {kernel_shape} instead.")

        super().__init__(kernel_shape=kernel_shape)

    @property
    def kernel_size(self):

        return (self.kernel_shape * (self.kernel_shape + 1)) // 2

    def zernikeradial(self, r: torch.Tensor, n: torch.Tensor, m: torch.Tensor):

        out = torch.zeros_like(r)
        bound = (n - m) // 2 + 1
        max_bound = bound.max().item()

        for k in range(max_bound):

            inc = (-1) ** k * _factorial(n - k) * r ** (n - 2 * k) / (math.factorial(k) * _factorial((n + m) // 2 - k) * _factorial((n - m) // 2 - k))
            out += torch.where(k < bound, inc, 0.0)

        return out

    def zernikepoly(self, r: torch.Tensor, phi: torch.Tensor, n: torch.Tensor, l: torch.Tensor):

        m = 2 * l - n
        return torch.where(m < 0, self.zernikeradial(r, n, -m) * torch.sin(m * phi), self.zernikeradial(r, n, m) * torch.cos(m * phi))

    def compute_support_vals(self, r: torch.Tensor, phi: torch.Tensor, r_cutoff: float, width: float = 0.25):

        # enumerator for basis function
        ikernel = torch.arange(self.kernel_size, device=r.device).reshape(-1, 1, 1)

        # get relevant indices
        iidx = torch.argwhere((r <= r_cutoff) & torch.full_like(ikernel, True, dtype=torch.bool, device=r.device))

        # indexing logic for zernike polynomials
        # the total index is given by (n * (n + 2) + l ) // 2 which needs to be reversed
        # precompute shifts in the level of the "pyramid"
        nshifts = torch.arange(self.kernel_shape, device=r.device)
        nshifts = (nshifts + 1) * nshifts // 2
        # find the level and position within the pyramid
        nkernel = torch.searchsorted(nshifts, ikernel, right=True) - 1
        lkernel = ikernel - nshifts[nkernel]
        # mkernel = 2 * ikernel - nkernel * (nkernel + 2)

        # get corresponding coordinates and n and l indices
        r = r[iidx[:, 1], iidx[:, 2]] / r_cutoff
        phi = phi[iidx[:, 1], iidx[:, 2]]
        n = nkernel[iidx[:, 0], 0, 0]
        l = lkernel[iidx[:, 0], 0, 0]

        # computes the Zernike polynomials using helper functions
        vals = self.zernikepoly(r, phi, n, l)

        return iidx, vals
