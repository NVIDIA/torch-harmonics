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

import torch
import torch.nn as nn

import torch_harmonics as th
from torch_harmonics.grid import require_regular_grid
from torch_harmonics.radial import RadialGrid, SphericalProductGrid


class GreensOperator(nn.Module):
    """
    Exact per-degree Green's operator for the radial Poisson equation. Solves the radial ODE that lap u = f reduces to under a spherical harmonic
    transform.

    Parameters
    -----------
    radial : RadialGrid
        Radial grid on the half-line (0, inf) or on the exterior domain [R, inf), whose
        nodes and volume weights r^2 dr discretize the radial integral
    lmax : int
        Number of spherical harmonic degrees

    References
    ----------
    .. [1] J. D. Jackson, *Classical Electrodynamics*, 3rd ed., Wiley, 1999, Sec. 3.9
    .. [2] R. Fitzpatrick, *Classical Electromagnetism*, lecture notes,
       https://farside.ph.utexas.edu/teaching/jk1/Electromagnetism/node31.html
    """

    def __init__(self, radial: RadialGrid, lmax: int):
        super().__init__()

        if radial.domain not in ("half-line", "exterior"):
            raise NotImplementedError(f"the Green's operator is implemented on the half-line and the exterior domain, got a {radial.domain} ({radial!r})")

        self.lmax = lmax
        self.domain = radial.domain
        self.inner_radius = radial.r_inner if self.domain == "exterior" else None

        green = self._assemble(torch.log(radial.r), radial.volume_weights)

        self.register_buffer("green", green)

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
            core = core - torch.exp((2 * l + 1) * math.log(self.inner_radius) - (l + 1) * (loglo + loghi))

        return -core * quad.reshape(1, 1, -1) / (2 * l + 1)

    def forward(self, fspec: torch.Tensor) -> torch.Tensor:
        """Solve lap u = f in spectral space."""

        # apply the operator in the precision of the input. the kernel is assembled in
        # float64 because it is built from exponentials over several decades in r, but
        # the caller decides what the solve runs in -- RadialPoissonDataset hands back float32
        # samples, and einsum rejects a float32 operand against a float64 kernel outright
        green = self.green.to(fspec.real.dtype)

        return torch.complex(
            torch.einsum("lkj,...jlm->...klm", green, fspec.real),
            torch.einsum("lkj,...jlm->...klm", green, fspec.imag),
        )


class RadialPoissonSolver(nn.Module):
    """
    Poisson solver on (0, inf) x S2 or [R, inf) x S2.

    Solves lap u = f subject to u -> 0 at infinity, using an exact per-degree Green's
    operator in the radial direction and a spherical harmonic transform in the angular
    directions.

    Parameters
    -----------
    grid : SphericalProductGrid
        Grid the fields live on, of shape (nr, nlat, nlon): a regular angular grid times a
        radial grid on the half-line (0, inf) or on the exterior domain [R, inf), e.g.
        ``SphericalProductGrid(as_grid("legendre-gauss", nlat=64, nlon=128), GeometricRadialGrid(256, 1e-1, 1e3))``.
        On the exterior domain R is the radial grid's ``r_inner``.
    lmax : int, optional
        Maximum l mode, by default None
    mmax : int, optional
        Maximum m mode, by default None

    Notes
    -----
    The solver is built in float64, since the Green's kernel spans several decades in r.
    Cast it with ``.float()`` before moving it to a device without float64, such as MPS,
    as ``RadialPoissonDataset`` does.
    """

    # new in v1.0.0rc1, so there is no signature before the grid descriptors to reject
    _legacy_signature_exempt = "new in v1.0.0rc1"

    def __init__(self, grid, lmax=None, mmax=None):
        super().__init__()

        if not isinstance(grid, SphericalProductGrid):
            raise TypeError(f"grid must be a SphericalProductGrid of an angular and a radial grid, got {grid!r}")
        require_regular_grid(grid.angular, "grid.angular")

        # grid parameters
        self.grid = grid
        self.nr, self.nlat, self.nlon = grid.shape
        self.domain = grid.radial.domain
        self.inner_radius = grid.radial.r_inner if self.domain == "exterior" else None

        # SHT, acting on every radial node alike
        self.sht = th.RealSHT(grid, lmax=lmax, mmax=mmax, csphase=False)
        self.isht = th.InverseRealSHT(grid, lmax=lmax, mmax=mmax, csphase=False)

        self.lmax = self.sht.lmax
        self.mmax = self.sht.mmax

        # exact radial Green's operator
        self.operator = GreensOperator(grid.radial, self.lmax)

        # compute gridpoints. x is the coordinate the radial nodes are uniform in for a
        # geometric grid, log(r) on the half-line and log(r - R) on the exterior domain
        lats = grid.angular.lats
        lons = grid.angular.lons()
        r = grid.radial.r
        w = grid.radial.quad_weights
        x = torch.log(r - grid.radial.r_inner)

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

    def solve(self, f: torch.Tensor) -> torch.Tensor:
        """Solve poisson equation lap u = f on the grid."""
        return self.spec2grid(self.operator(self.grid2spec(f)))

    def _random_angular_spec(self, shape, l_src=8) -> torch.Tensor:
        """Random band-limited complex coefficients of a real field on the sphere."""

        lsrc = min(l_src + 1, self.lmax)
        msrc = min(lsrc, self.mmax)

        ctype = torch.complex128 if self.r.dtype == torch.float64 else torch.complex64
        aspec = torch.zeros(*shape, self.lmax, self.mmax, dtype=ctype, device=self.r.device)
        scale = math.sqrt(4.0 * math.pi / lsrc / (lsrc + 1))
        aspec[..., :lsrc, :msrc] = scale * torch.randn_like(aspec[..., :lsrc, :msrc])

        aspec = torch.tril(aspec)
        aspec[..., 0].imag.zero_()

        return aspec

    def ball_source(self, radius=1.0, value=1.0) -> torch.Tensor:
        """Source for the Poisson equation on a ball of radius `radius` and value `value`."""
        f = torch.zeros(self.nr, self.nlat, self.nlon, dtype=self.r.dtype, device=self.r.device)
        f[self.r <= radius] = value
        return f

    def random_bump_source(self, nblobs=4, l_src=8, margin=0.05, width=(0.03, 0.10)) -> torch.Tensor:
        """Random multi-scale source. A sum of nblobs terms, each a radial bump in x = log r times a random band-limited angular field."""

        device = self.r.device
        dtype = self.r.dtype

        # range of source support
        x = self.x
        span = (x[-1] - x[0]).item()
        xlo = x[0].item() + margin * span
        xhi = x[-1].item() - margin * span

        # every blob must contain a node, or a sample can vanish on the grid and normalizing
        # it divides by zero. A blob is an open interval in x at least 2 * min_half wide, so
        # that holds exactly when no gap between neighbouring nodes is as wide
        min_half = min(width[0] * span, 0.5 * (xhi - xlo))
        gap = (x[1:] - x[:-1]).max().item()
        if gap >= 2 * min_half:
            raise ValueError(
                f"the radial grid is too coarse for bumps of half width {width[0]} of the log-radial range: its widest gap, "
                f"{gap / span:.3g} of the range, can hold a bump without a node. Use more radial nodes or a larger width[0]."
            )

        # random half width of blobs
        half = width[0] * span + (width[1] - width[0]) * span * torch.rand(nblobs, dtype=dtype, device=device)
        half = torch.clamp(half, max=0.5 * (xhi - xlo))
        # random centers of blobs
        centers = xlo + half + (xhi - xlo - 2 * half) * torch.rand(nblobs, dtype=dtype, device=device)

        # dim random angular center towards blob boundary, shape (nblobs, nr)
        t = (x.unsqueeze(0) - centers.unsqueeze(-1)) / half.unsqueeze(-1)
        inside = t.abs() < 1.0
        radial = torch.zeros_like(t)
        radial[inside] = torch.exp(1.0 - 1.0 / (1.0 - t[inside] ** 2))

        # one angular field per blob, combined with its radial profile before the inverse
        # transform, which is linear, so the transform runs once over the radial nodes
        aspec = self._random_angular_spec((nblobs,), l_src=l_src)
        fspec = torch.einsum("blm,bk->klm", aspec, radial.to(aspec.dtype))

        return self.spec2grid(fspec)

    def plot_sphere(self, data, ax, title="", cmap=None, vmin=None, vmax=None, projection="mollweide", colorbar=True):
        """One radial level of a grid field, on the sphere. Supports the "mollweide" projection."""

        import matplotlib.pyplot as plt
        import numpy as np

        if data.ndim != 2:
            raise ValueError(f"data must be 2D (nlat, nlon), got {tuple(data.shape)}")

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
            for radius in [self.inner_radius, r[-1]] if self.domain == "exterior" else [r[-1]]:
                ax.plot(radius * np.sin(t), radius * np.cos(t), c="k", lw=0.8)

            ax.set_aspect("equal")
            ax.set_axis_off()
        elif projection == "log":
            if self.domain == "exterior":
                x, label = (r - self.inner_radius) / self.inner_radius, r"$\rho/R$"
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
