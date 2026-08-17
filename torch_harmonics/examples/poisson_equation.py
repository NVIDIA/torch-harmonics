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


import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

import torch_harmonics as th
from torch_harmonics.quadrature import geometric_weights, precompute_latitudes, precompute_longitudes


def radial_grid(
    nr: int, vmin: float, vmax: float, grid: str = "half-line", R: Optional[float] = None, dtype: torch.dtype = torch.float64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Radial grid and quadrature weights.

    Parameters
    -----------
    nr : int
        Number of radial nodes
    vmin : float
        Lower bound on r, or on rho / R = (r - R) / R for the exterior domain
    vmax : float
        Upper bound
    grid : str, optional
        Either "half-line" or "exterior", by default "half-line"
    R : float, optional
        Inner radius, required for domain="exterior", by default None
    dtype : torch.dtype, optional
        Floating point type, by default torch.float64

    Returns
    -------
    x : torch.Tensor
        Reduced coordinate of the nodes
    r : torch.Tensor
        Radial nodes
    w : torch.Tensor
        Trapezoidal weights for the integral over dr
    """

    if grid == "half-line":
        r, w = geometric_weights(nr, vmin, vmax)
        x = torch.log(r)
    elif grid == "exterior":
        if R is None:
            raise ValueError("R must be given for grid='exterior'")
        rho, wrho = geometric_weights(nr, vmin, vmax)
        x, r, w = torch.log(rho), R + R * rho, R * wrho
    else:
        raise ValueError(f"unknown grid: {grid}")

    return x.to(dtype), r.to(dtype), w.to(dtype)


class GreensOperator(nn.Module):
    """
    Exact per-degree Green's operator for the radial Poisson equation. Solves the radial ODE that lap u = f reduces to under a spherical harmonic
    transform.

    Parameters
    -----------
    r : torch.Tensor
        Radial nodes
    w : torch.Tensor
        Quadrature weights for the integral over dr
    lmax : int
        Number of spherical harmonic degrees
    domain : str, optional
        Either "half-line" or "exterior", by default "half-line"
    R : float, optional
        Inner radius for exterior domain, by default None
    """

    def __init__(self, r: torch.Tensor, w: torch.Tensor, lmax: int, domain: str = "half-line", R: Optional[float] = None):
        super().__init__()

        if domain not in ("half-line", "exterior"):
            raise NotImplementedError(f"Domain {domain} not implemented")
        if domain == "exterior" and R is None:
            raise ValueError("R must be given for domain='exterior'")

        self.lmax = lmax
        self.domain = domain
        self.R = R

        logr = torch.log(r)
        green = self._assemble(logr, r**2 * w)

        self.register_buffer("green", green)

        # harmonic lift for boundary data, shape (nr, lmax)
        if domain == "exterior":
            l = torch.arange(0, lmax, dtype=r.dtype).unsqueeze(0)
            self.register_buffer("blift", torch.exp((l + 1) * (math.log(R) - logr.unsqueeze(-1))))

    def _assemble(self, logr: torch.Tensor, quad: torch.Tensor) -> torch.Tensor:
        """Green's kernel of shape (lmax, nr, nr), assembled in log space."""

        # radial nodes broadcast against each other, shapes (nr, 1) and (1, nr)
        logrk = logr.unsqueeze(-1)
        logrj = logr.unsqueeze(0)
        loglo = torch.minimum(logrk, logrj)
        loghi = torch.maximum(logrk, logrj)

        # degrees, shape (lmax, 1, 1)
        l = torch.arange(0, self.lmax, dtype=logr.dtype, device=logr.device).reshape(-1, 1, 1)

        core = torch.exp(l * loglo - (l + 1) * loghi)
        if self.domain == "exterior":
            core = core - torch.exp((2 * l + 1) * math.log(self.R) - (l + 1) * (loglo + loghi))

        return -core * quad.reshape(1, 1, -1) / (2 * l + 1)

    def forward(self, fspec: torch.Tensor, v0spec: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Solve lap u = f in spectral space."""

        uspec = torch.complex(
            torch.einsum("lkj,...jlm->...klm", self.green, fspec.real),
            torch.einsum("lkj,...jlm->...klm", self.green, fspec.imag),
        )

        # handle boundary data for exterior domain
        if v0spec is not None:
            if self.domain != "exterior":
                raise ValueError("inner Dirichlet data is only defined on the exterior domain")
            uspec = uspec + v0spec.unsqueeze(-3) * self.blift.unsqueeze(-1).to(v0spec.dtype)

        return uspec


class RadialPoissonSolver(nn.Module):
    """
    Poisson solver on (0, inf) x S2 or [R, inf) x S2.

    Solves lap u = f subject to u -> 0 at infinity, using an exact per-degree Green's
    operator in the radial direction and a spherical harmonic transform in the angular
    directions.

    The Green's kernel is stored densely, so the solver holds an (lmax, nr, nr) float64
    buffer and each solve costs O(lmax * mmax * nr**2): 33 MB at nlat=64, nr=256, but
    537 MB at nlat=256, nr=512. Size the radial grid accordingly.

    Parameters
    -----------
    nlat : int
        Number of latitude points
    nlon : int
        Number of longitude points
    nr : int
        Number of radial points
    r_min, r_max : float, optional
        Bounds of the radial grid.
    lmax : int, optional
        Maximum l mode, by default None
    mmax : int, optional
        Maximum m mode, by default None
    grid : str, optional
        Grid type ("legendre-gauss", "lobatto", "equiangular"), by default "legendre-gauss"
    domain : str, optional
        Either "half-line" or "exterior", by default "half-line"
    R : float, optional
        Inner radius for exterior domain, by default None
    """

    def __init__(self, nlat, nlon, nr, r_min=None, r_max=None, lmax=None, mmax=None, grid="legendre-gauss", domain="half-line", R=None):
        super().__init__()

        # grid parameters
        self.nlat = nlat
        self.nlon = nlon
        self.nr = nr
        self.grid = grid
        self.domain = domain
        self.R = R

        # SHT
        self.sht = th.RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False)
        self.isht = th.InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False)

        self.lmax = self.sht.lmax
        self.mmax = self.sht.mmax

        # compute gridpoints; precompute_latitudes returns colatitudes ordered from the
        # north pole, matching the row order the SHT expects. Unsupported grids are
        # already rejected by the RealSHT constructor above.
        colats, _ = precompute_latitudes(self.nlat, grid=self.grid)
        lats = 0.5 * torch.pi - colats
        lons = precompute_longitudes(self.nlon)

        # radial grid and the exact radial Green's operator
        if domain == "half-line":
            if R is not None:
                raise ValueError("R is only meaningful on the exterior domain")
            defaults = (1e-1, 1e3)
        elif domain == "exterior":
            defaults = (1e-2, 1e2)
        else:
            raise ValueError(f"unknown domain: {domain}")

        self.r_min = defaults[0] if r_min is None else r_min
        self.r_max = defaults[1] if r_max is None else r_max

        x, r, w = radial_grid(nr, self.r_min, self.r_max, grid=domain, R=R)
        self.radial = GreensOperator(r, w, self.lmax, domain=domain, R=R)

        # register all
        self.register_buffer("lats", lats)
        self.register_buffer("lons", lons)
        self.register_buffer("x", x)
        self.register_buffer("r", r)
        self.register_buffer("w", w)

    def grid2spec(self, ugrid):
        """Convert spatial data to spectral coefficients."""
        return self.sht(ugrid)

    def spec2grid(self, uspec):
        """Convert spectral coefficients to spatial data."""
        return self.isht(uspec)

    def solve(self, f: torch.Tensor, v0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Solve poisson equation lap u = f on the grid."""
        v0spec = None if v0 is None else self.grid2spec(v0)
        return self.spec2grid(self.radial(self.grid2spec(f), v0spec))

    def _random_angular_spec(self, shape, l_src=8, decay=1.0) -> torch.Tensor:
        """Random band-limited complex coefficients of a real field on the sphere."""

        lsrc = min(l_src + 1, self.lmax)
        msrc = min(lsrc, self.mmax)

        aspec = torch.zeros(*shape, self.lmax, self.mmax, dtype=torch.complex128, device=self.r.device)
        scale = math.sqrt(4.0 * math.pi / lsrc / (lsrc + 1))
        aspec[..., :lsrc, :msrc] = scale * torch.randn_like(aspec[..., :lsrc, :msrc])

        # decay higher modes
        l = torch.arange(0, self.lmax, dtype=self.r.dtype, device=self.r.device).reshape(-1, 1)
        aspec = aspec * (1.0 + l).pow(-decay).to(aspec.dtype)

        aspec = torch.tril(aspec)
        aspec[..., 0].imag.zero_()

        return aspec

    def random_source(self, nblobs=(1, 8), l_src=8, decay=1.0, margin=None, width=(0.03, 0.10), positive=False) -> torch.Tensor:
        """
        Random multi-scale source.
        A sum of nblobs terms, each a radial bump in x = log r times a random band-limited angular field.

        Parameters
        -----------
        nblobs : int or tuple of int, optional
            Number of blobs, or an inclusive range to draw it from, by default (1, 8)
        l_src : int, optional
            Angular band limit of the source, by default 8
        decay : float, optional
            Decay exponent of angular spectrum, by default 1.0
        margin : float, optional
            Fraction of the log range kept free at each end, by default width[1] + 0.02
        width : tuple of float, optional
            Range of blob half-widths as a fraction of the log range, by default (0.03, 0.10)
        positive : bool, optional
            Make the source non-negative
        """

        device = self.r.device
        dtype = self.r.dtype

        if margin is None:
            margin = width[1] + 0.02
        if not isinstance(nblobs, int):
            nblobs = int(torch.randint(nblobs[0], nblobs[1] + 1, ()).item())

        # range of source support -> leave margin to avoid boundary effects
        x = self.x
        span = (x[-1] - x[0]).item()
        xlo = x[0].item() + margin * span
        xhi = x[-1].item() - margin * span

        # random half width of blobs, clamped so the support always fits
        half = width[0] * span + (width[1] - width[0]) * span * torch.rand(nblobs, dtype=dtype, device=device)
        half = torch.clamp(half, max=0.5 * (xhi - xlo))
        # random centers of blobs
        centers = xlo + half + (xhi - xlo - 2 * half) * torch.rand(nblobs, dtype=dtype, device=device)

        # dim random angular center towards blob boundary, shape (nblobs, nr)
        t = (x.unsqueeze(0) - centers.unsqueeze(-1)) / half.unsqueeze(-1)
        inside = t.abs() < 1.0
        radial = torch.zeros_like(t)
        radial[inside] = torch.exp(1.0 - 1.0 / (1.0 - t[inside] ** 2))

        # one angular field per blob, so a per-blob shift can enforce positivity
        angular = self.spec2grid(self._random_angular_spec((nblobs,), l_src=l_src, decay=decay))
        if positive:
            angular = angular - angular.amin(dim=(-1, -2), keepdim=True)

        return torch.einsum("bxy,bk->kxy", angular, radial)

    def random_boundary_data(self, l_src=8, decay=1.0) -> torch.Tensor:
        """Random band-limited Dirichlet data on the inner sphere."""

        if self.domain != "exterior":
            raise ValueError("random_boundary_data is only defined on the exterior domain")

        return self.spec2grid(self._random_angular_spec((), l_src=l_src, decay=decay))

    def plot_sphere(self, data, ax, title="", cmap=None, vmin=None, vmax=None, projection="mollweide", colorbar=True):
        """One radial level of a grid field, on the sphere. Supports the "mollweide" projection."""

        import matplotlib.pyplot as plt
        import numpy as np

        assert data.ndim == 2, f"data must be 2D (nlat, nlon), got {data.shape}"

        data = data.detach().cpu()
        lons = self.lons.cpu() - torch.pi
        lats = self.lats.cpu()

        if projection == "mollweide":
            Lons, Lats = np.meshgrid(lons, lats, indexing="ij")
            im = ax.pcolormesh(Lons, Lats, data.T, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(True, alpha=0.3)
        else:
            raise NotImplementedError(f"projection {projection!r} not implemented")

        if colorbar:
            plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
        ax.set_title(title)
        return im

    def plot_meridional(self, data, ax, ilon=0, title="", cmap=None, vmin=None, vmax=None, projection="log", colorbar=True, rmax=None):
        """Meridional slice of a grid field of shape (nr, nlat, nlon)."""

        import matplotlib.pyplot as plt
        import numpy as np

        data = data.detach().cpu()
        r = self.r.cpu()
        if rmax is not None:
            r, data = r[r <= rmax], data[r <= rmax]

        lats = self.lats.cpu()

        if projection == "polar":
            # the far side of the plane is the antipodal meridian, at angle 2 pi - colat
            colat = 0.5 * torch.pi - lats
            iopp = (ilon + self.nlon // 2) % self.nlon
            ang = torch.cat([colat, 2 * torch.pi - colat])
            slab = torch.cat([data[:, :, ilon], data[:, :, iopp]], dim=1)

            order = torch.argsort(ang)
            ang, slab = ang[order], slab[:, order]

            # repeat the first column one turn later, so the seam over the pole closes
            ang = torch.cat([ang, ang[:1] + 2 * torch.pi])
            slab = torch.cat([slab, slab[:, :1]], dim=1)

            # angle measured from the vertical, so the pole ends up at the top
            rr, aa = np.meshgrid(r, ang, indexing="ij")
            im = ax.pcolormesh(rr * np.sin(aa), rr * np.cos(aa), slab, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud")

            # stroke the walls; the inner one is R itself, which is never a grid node
            t = np.linspace(0.0, 2 * np.pi, 361)
            for radius in [self.R, r[-1]] if self.domain == "exterior" else [r[-1]]:
                ax.plot(radius * np.sin(t), radius * np.cos(t), c="k", lw=0.8)

            ax.set_aspect("equal")
            ax.set_axis_off()
        elif projection == "log":
            if self.domain == "exterior":
                x, label = (r - self.R) / self.R, r"$\rho/R$"
            else:
                x, label = r, "$r$"

            im = ax.pcolormesh(x, lats, data[:, :, ilon].T, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xscale("log")
            ax.set_xlabel(label)
            ax.set_ylabel("latitude")
        else:
            raise NotImplementedError(f"projection {projection!r} not implemented")

        if colorbar:
            plt.colorbar(im, ax=ax)
        ax.set_title(title)
        return im
