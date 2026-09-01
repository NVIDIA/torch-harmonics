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
from typing import Optional

# import numpy as np
import torch
import torch.nn as nn

from torch_harmonics.quadrature import precompute_latitudes, precompute_longitudes


def _slerp_shortest_arc(start: torch.Tensor, end: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    r"""Interpolate angle-valued samples along the shorter arc of the circle.

    Spherical linear interpolation of two unit vectors
    :math:`\mathbf{u}_i = (\cos f_i, \sin f_i)` separated by
    :math:`\Omega = \arccos(\mathbf{u}_0 \cdot \mathbf{u}_1)` is

    .. math::

        \mathrm{slerp}(t) = \frac{\sin((1-t)\Omega)}{\sin\Omega}\,\mathbf{u}_0
                          + \frac{\sin(t\Omega)}{\sin\Omega}\,\mathbf{u}_1

    In two dimensions the angle of that interpolant collapses to a much simpler
    closed form -- walking a fraction :math:`t` along the shorter arc:

    .. math::

        f(t) = f_0 + t \cdot \mathrm{wrap}_\pi(f_1 - f_0),
        \qquad \mathrm{wrap}_\pi(\delta) \in (-\pi, \pi]

    so no trigonometry, no division, and no singularity are needed. Note the
    weights of the vector form are deliberately *not* a partition of unity --
    that is what re-normalises the interpolant back onto the sphere -- so they
    must never be applied to scalar samples directly.

    Because :math:`\mathrm{wrap}_\pi` is the identity whenever
    :math:`|f_1 - f_0| \le \pi`, this agrees exactly with plain linear
    interpolation everywhere except across a phase wrap, which is the case this
    mode exists to handle. It is exactly shift-equivariant, reproduces both
    endpoints, and never returns a value outside the range spanned by the arc.

    Parameters
    ----------
    start, end : torch.Tensor
        Angle-valued samples at the two ends of the arc, in radians.
    weight : torch.Tensor
        Interpolation weight :math:`t \in [0, 1]`.

    Returns
    -------
    torch.Tensor
        Interpolated angles, expressed as a continuous lift from ``start``
        (i.e. not re-wrapped), so the result stays adjacent to ``start``.
    """
    # antipodal samples (delta = +-pi) are a genuine tie: both arcs are equally
    # short. remainder() resolves it consistently towards -pi.
    delta = torch.remainder(end - start + math.pi, 2.0 * math.pi) - math.pi
    return start + weight * delta


def _circular_mean(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    r"""Mean *direction* of angle-valued samples, in radians.

    .. math::

        \bar{f} = \mathrm{atan2}\bigl(\overline{\sin f},\; \overline{\cos f}\bigr)

    The arithmetic mean is meaningless for angles: averaging :math:`+\pi` and
    :math:`-\pi`, which denote the same direction, gives :math:`0` -- the
    opposite one. Averaging the unit vectors instead and recovering the angle is
    invariant to where the branch cut is placed.

    If the samples are spread uniformly around the circle the resultant vector
    vanishes and the mean direction is genuinely undefined (a phase defect);
    ``atan2(0, 0)`` returns ``0`` in that case.
    """
    return torch.atan2(torch.sin(x).mean(dim=dim, keepdim=keepdim), torch.cos(x).mean(dim=dim, keepdim=keepdim))


class ResampleS2(nn.Module):
    r"""
    Resampling module for signals on the 2-sphere :math:`S^2`.

    This module resamples a spherical signal from one grid resolution (and type)
    to another.  Interpolation is performed independently along latitudes and
    longitudes, with proper handling of periodicity in :math:`\lambda` and pole
    expansion when the output grid extends beyond the input latitude range.

    Two interpolation modes are available:

    * ``"bilinear"`` -- Standard bilinear (linear-linear) interpolation.  For
      two neighbouring grid values :math:`f_0` and :math:`f_1` with interpolation
      weight :math:`t \in [0, 1]`, the interpolated value is

      .. math::

          f(t) = (1 - t)\, f_0 + t\, f_1

      This is applied first along the latitudinal (:math:`\theta`) and then
      along the longitudinal (:math:`\lambda`) direction.

    * ``"bilinear-spherical"`` -- Spherical linear interpolation (slerp) for
      fields whose *values* are angles (e.g.\ directions or phases, in
      radians).  Neighbouring samples are interpolated along the shorter arc of
      the circle rather than along a straight line in value space:

      .. math::

          f(t) = f_0 + t \cdot \mathrm{wrap}_\pi(f_1 - f_0),
          \qquad \mathrm{wrap}_\pi(\delta) \in (-\pi, \pi]

      This is the two-dimensional closed form of slerp; see the
      ``_slerp_shortest_arc`` helper in this module for the derivation.  Since
      :math:`\mathrm{wrap}_\pi` is the identity whenever
      :math:`|f_1 - f_0| \le \pi`, this mode is **identical to** ``"bilinear"``
      except across a :math:`2\pi` phase wrap -- precisely the discontinuity it
      exists to handle.  Interpolating :math:`f_0 = 3` and :math:`f_1 = -3`
      (nearly the same direction, either side of the branch cut) gives
      :math:`\pi` here, whereas ``"bilinear"`` gives :math:`0`, the opposite
      direction.

      The result is expressed as a continuous lift from :math:`f_0` and is not
      re-wrapped, so it stays adjacent to :math:`f_0` and may leave
      :math:`(-\pi, \pi]`.  Apply :func:`torch.remainder` afterwards if a
      canonical range is required.

    Parameters
    ----------
    nlat_in : int
        Number of latitude points in the input grid
    nlon_in : int
        Number of longitude points in the input grid
    nlat_out : int
        Number of latitude points in the output grid
    nlon_out : int
        Number of longitude points in the output grid
    grid_in : str, optional
        Input grid type (``"equiangular"``, ``"legendre-gauss"``, ``"lobatto"``),
        by default ``"equiangular"``
    grid_out : str, optional
        Output grid type (``"equiangular"``, ``"legendre-gauss"``, ``"lobatto"``),
        by default ``"equiangular"``
    mode : str, optional
        Interpolation mode (``"bilinear"``, ``"bilinear-spherical"``), by default
        ``"bilinear"``.  See above for a description of each mode.

    Examples
    --------
    >>> import torch
    >>> import torch_harmonics as th
    >>> resample = th.ResampleS2(64, 128, 128, 256).cuda()
    >>> x = torch.randn(1, 64, 128, device="cuda")
    >>> y = resample(x)
    >>> y.shape
    torch.Size([1, 128, 256])
    """

    def __init__(
        self,
        nlat_in: int,
        nlon_in: int,
        nlat_out: int,
        nlon_out: int,
        grid_in: Optional[str] = "equiangular",
        grid_out: Optional[str] = "equiangular",
        mode: Optional[str] = "bilinear",
    ):

        super().__init__()

        # currently only bilinear is supported
        if mode in ["bilinear", "bilinear-spherical"]:
            self.mode = mode
        else:
            raise NotImplementedError(f"unknown interpolation mode {mode}")

        self.nlat_in, self.nlon_in = nlat_in, nlon_in
        self.nlat_out, self.nlon_out = nlat_out, nlon_out

        self.grid_in = grid_in
        self.grid_out = grid_out

        # for upscaling the latitudes we will use interpolation
        self.lats_in, _ = precompute_latitudes(nlat_in, grid=grid_in)
        self.lons_in = precompute_longitudes(nlon_in)
        self.lats_out, _ = precompute_latitudes(nlat_out, grid=grid_out)
        self.lons_out = precompute_longitudes(nlon_out)

        # in the case where some points lie outside of the range spanned by lats_in,
        # we need to expand the solution to the poles before interpolating
        # bool(), not a 0-dim tensor: this is branched on in forward, and a tensor
        # there is data-dependent control flow that breaks torch.compile(fullgraph=True)
        self.expand_poles = bool((self.lats_out > self.lats_in[-1]).any() or (self.lats_out < self.lats_in[0]).any())
        if self.expand_poles:
            self.lats_in = torch.cat(
                [torch.as_tensor([0.0], dtype=torch.float64, device=self.lats_in.device), self.lats_in, torch.as_tensor([math.pi], dtype=torch.float64, device=self.lats_in.device)]
            ).contiguous()

        # prepare the interpolation by computing indices to the left and right of each output latitude
        lat_idx = torch.searchsorted(self.lats_in, self.lats_out, side="right") - 1
        # make sure that we properly treat the last point if they coincide with the pole
        lat_idx = torch.where(self.lats_out == self.lats_in[-1], lat_idx - 1, lat_idx)

        # lat_idx = np.where(self.lats_out > self.lats_in[-1], lat_idx - 1, lat_idx)
        # lat_idx = np.where(self.lats_out < self.lats_in[0], 0, lat_idx)

        # compute the interpolation weights along the latitude
        lat_weights = ((self.lats_out - self.lats_in[lat_idx]) / torch.diff(self.lats_in)[lat_idx]).to(torch.float32)
        lat_weights = lat_weights.unsqueeze(-1)

        # register buffers
        self.register_buffer("lat_idx", lat_idx, persistent=False)
        self.register_buffer("lat_weights", lat_weights, persistent=False)

        # get left and right indices but this time make sure periodicity in the longitude is handled
        lon_idx_left = torch.searchsorted(self.lons_in, self.lons_out, side="right") - 1
        lon_idx_right = torch.where(self.lons_out >= self.lons_in[-1], torch.zeros_like(lon_idx_left), lon_idx_left + 1)

        # get the difference
        diff = self.lons_in[lon_idx_right] - self.lons_in[lon_idx_left]
        diff = torch.where(diff < 0.0, diff + 2 * math.pi, diff)
        lon_weights = ((self.lons_out - self.lons_in[lon_idx_left]) / diff).to(torch.float32)

        # register buffers
        self.register_buffer("lon_idx_left", lon_idx_left, persistent=False)
        self.register_buffer("lon_idx_right", lon_idx_right, persistent=False)
        self.register_buffer("lon_weights", lon_weights, persistent=False)

        self.skip_resampling = (nlon_in == nlon_out) and (nlat_in == nlat_out) and (grid_in == grid_out)

    def extra_repr(self):
        return f"in_shape={(self.nlat_in, self.nlon_in)}, out_shape={(self.nlat_out, self.nlon_out)}"

    def _upscale_longitudes(self, x: torch.Tensor):
        # do the interpolation in precision of x
        lwgt = self.lon_weights.to(x.dtype)
        if self.mode == "bilinear":
            x = torch.lerp(x[..., self.lon_idx_left], x[..., self.lon_idx_right], lwgt)
        else:
            x = _slerp_shortest_arc(x[..., self.lon_idx_left], x[..., self.lon_idx_right], lwgt)

        return x

    def _expand_poles(self, x: torch.Tensor):
        # A pole is a single point, so its value cannot depend on longitude: reduce the
        # adjacent ring over phi. That reduction annihilates every m != 0 mode exactly,
        # which is the right answer there, since every m != 0 mode vanishes at the pole.
        #
        # Do NOT replace this with an f(theta, phi) -> f(-theta, phi + pi) continuation.
        # That is the correct way to reach *through* a pole (as a neighbourhood stencil
        # does), but here it would leave a phi-dependent -- i.e. multivalued -- value
        # *at* the pole: odd m cancels, but even m reinforces.
        if self.mode == "bilinear":
            x_north = x[..., 0, :].mean(dim=-1, keepdims=True)
            x_south = x[..., -1, :].mean(dim=-1, keepdims=True)
        else:
            # angle-valued field: must reduce as directions, not as numbers
            x_north = _circular_mean(x[..., 0, :], dim=-1, keepdim=True)
            x_south = _circular_mean(x[..., -1, :], dim=-1, keepdim=True)
        x = nn.functional.pad(x, pad=[0, 0, 1, 1], mode="constant")
        x[..., 0, :] = x_north[...]
        x[..., -1, :] = x_south[...]

        return x

    def _upscale_latitudes(self, x: torch.Tensor):
        # do the interpolation in precision of x
        lwgt = self.lat_weights.to(x.dtype)
        if self.mode == "bilinear":
            x = torch.lerp(x[..., self.lat_idx, :], x[..., self.lat_idx + 1, :], lwgt)
        else:
            x = _slerp_shortest_arc(x[..., self.lat_idx, :], x[..., self.lat_idx + 1, :], lwgt)

        return x

    def forward(self, x: torch.Tensor):
        """
        Resample a spherical signal onto the output grid.

        Parameters
        ----------
        x : torch.Tensor
            Input signal of shape ``(..., nlat_in, nlon_in)``. Resampling acts on the
            last two (spatial) dimensions; any leading batch/channel dimensions are
            preserved.

        Returns
        -------
        torch.Tensor
            Resampled signal of shape ``(..., nlat_out, nlon_out)``.
        """
        if self.skip_resampling:
            return x

        if self.expand_poles:
            x = self._expand_poles(x)

        x = self._upscale_latitudes(x)

        x = self._upscale_longitudes(x)

        return x
