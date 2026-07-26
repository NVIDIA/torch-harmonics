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


# Shared radial geometry for the 3D spherical solvers.
#
# Both the Poisson Green's operator and the heat Sturm-Liouville operator live on a
# radial grid that is uniform in some coordinate s, with r recovered through a
# domain-dependent map and Jacobian J = dr/ds:
#
#     linear             s = r            J = 1
#     geometric          s = log r        J = r
#     shifted-geometric  s = log(r - R)   J = rho = r - R
#
# The two geometric maps are what make a dilation an exact shift of the node index,
# which is why the dilation-group statements come out at roundoff rather than at
# discretization order.
#
# The Sturm-Liouville weights (w, m, q) = (r^2/J, r^2 J, J) put the radial Laplacian in
# flux form. Note that m = r^2 J is simply the volume measure: the heat solver's mass
# matrix is m ds, and the Poisson quadrature weight for the integral over r^2 dr is
# m ds times a trapezoid factor. Both solvers therefore integrate against the same
# measure, which is the invariant this module exists to keep true.


import math
from typing import Optional, Tuple

import torch

GRID_KINDS = ("linear", "geometric", "shifted-geometric")

# each domain's natural coordinate; callers may override
DEFAULT_GRID_KIND = {
    "shell": "linear",
    "half-line": "geometric",
    "exterior": "shifted-geometric",
}


def default_grid_kind(domain: str) -> str:
    """The radial coordinate a domain is naturally gridded in."""

    try:
        return DEFAULT_GRID_KIND[domain]
    except KeyError:
        raise NotImplementedError(f"domain={domain!r} not implemented") from None


def to_uniform(v_in: float, v_out: float, kind: str) -> Tuple[float, float]:
    """Map user-facing bounds to the uniform coordinate s.

    The bounds are on r for a linear or geometric grid, and on rho = r - R for a
    shifted-geometric one.
    """

    if kind == "linear":
        return v_in, v_out
    if kind in ("geometric", "shifted-geometric"):
        if v_in <= 0.0:
            raise ValueError(f"lower bound must be positive on a {kind} grid, got {v_in}")
        return math.log(v_in), math.log(v_out)
    raise ValueError(f"unknown grid kind: {kind}")


def coordinates(s: torch.Tensor, kind: str, R: Optional[float] = None, scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Radius r(s) and the Jacobian J = dr/ds.

    `scale` multiplies the geometric coordinate, i.e. rho = scale * exp(s), rather than
    folding the factor into the exponent as exp(s + log(scale)). Both are accurate; the
    difference is small, roughly 1 to 4 ulp for the folded form against 0 to 0.3 ulp for
    this one, and it does not show up end to end. The reason to prefer this form is
    structural: rho is then literally R times one fixed array, which is what the
    non-dimensional exterior grid claims to be.
    """

    if kind == "linear":
        return scale * s, torch.full_like(s, scale)
    if kind == "geometric":
        r = scale * torch.exp(s)
        return r, r
    if kind == "shifted-geometric":
        if R is None:
            raise ValueError("R must be given for a shifted-geometric grid")
        rho = scale * torch.exp(s)
        return R + rho, rho
    raise ValueError(f"unknown grid kind: {kind}")


def sturm_liouville_weights(r: torch.Tensor, jac: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(w, m, q) = (r^2 / J, r^2 J, J), putting the radial Laplacian in flux form."""

    return r**2 / jac, r**2 * jac, jac


def uniform_nodes(n: int, s_in: float, s_out: float, layout: str, dtype: torch.dtype = torch.float64):
    """Evolved-node coordinates, face coordinates and the uniform spacing ds.

    all  : n nodes spanning [s_in, s_out], every one evolved. Faces are not meaningful
           and come back as None; this is the layout the Poisson quadrature uses.
    node : nodes span [s_in, s_out]; nodes 0 and n-1 hold boundary data. Evolved nodes
           are 1..n-2, and the faces flanking them are the midpoints.
    cell : n cell centers, n + 1 faces, the outermost faces sitting exactly on s_in and
           s_out. Every cell is evolved.
    """

    if layout == "all":
        s = torch.linspace(s_in, s_out, n, dtype=dtype)
        return s, None, (s[1] - s[0]).clone()

    if layout == "node":
        s = torch.linspace(s_in, s_out, n, dtype=dtype)
        ds = (s[1] - s[0]).clone()
        s_ev = s[1:-1]
        # faces at the midpoints flanking the evolved nodes: s_ev -/+ ds/2
        return s_ev, torch.cat([s_ev - 0.5 * ds, s_ev[-1:] + 0.5 * ds]), ds

    if layout == "cell":
        ds = torch.as_tensor((s_out - s_in) / n, dtype=dtype)
        s_face = s_in + ds * torch.arange(n + 1, dtype=dtype)
        return 0.5 * (s_face[:-1] + s_face[1:]), s_face, ds

    raise ValueError(f"unknown layout: {layout}")
