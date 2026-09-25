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


import torch

from torch_harmonics.quadrature import QuadratureS2

from .poisson_equation import RadialPoissonSolver


class PoissonDataset(torch.utils.data.Dataset):
    """Custom Dataset class for Poisson training data

    Parameters
    ----------
    dims : tuple, optional
        Number of latitude, longitude and radial points, by default (64, 128, 256)
    grid : str, optional
        Angular grid type, by default "legendre-gauss"
    domain : str, optional
        Either "half-line" or "exterior", by default "half-line"
    inner_radius : float, optional
        Inner radius for exterior domain, by default None
    rmin, rmax : float, optional
        Radial bounds, by default (1e-1, 1e3)
    source : str, optional
        Source type, by default "bump". Either "bump" or "ball"
    num_examples : int, optional
        Number of examples, by default 32
    device : torch.device, optional
        Device to use, by default torch.device("cpu")
    normalize : bool, optional
        Whether to normalize the input and target, by default True

    Returns
    -------
    inp : torch.Tensor
        Source times r**2, shape (nr, nlat, nlon).
    tar : torch.Tensor
        Solution, shape (nr, nlat, nlon)
    """

    def __init__(
        self,
        dims=(64, 128, 256),
        grid="legendre-gauss",
        domain="half-line",
        inner_radius=None,
        rmin=1e-1,
        rmax=1e3,
        source="bump",
        num_examples=32,
        device=torch.device("cpu"),
        normalize=True,
    ):
        if source not in ("bump", "ball"):
            raise ValueError(f"unknown source: {source}")

        self.num_examples = num_examples
        self.device = device
        self.normalize = normalize
        self.source = source
        self.nlat, self.nlon, self.nr = dims

        self.solver = RadialPoissonSolver(
            self.nlat,
            self.nlon,
            self.nr,
            rmin=rmin,
            rmax=rmax,
            grid=grid,
            domain=domain,
            inner_radius=inner_radius,
        ).to(self.device)

        # mean over the sphere for the scale
        self.sphere_mean = QuadratureS2((self.nlat, self.nlon), grid=grid, normalize=True).to(self.device)

    def __len__(self):
        return self.num_examples

    def _get_sample(self):
        """Get one unscaled (r**2 f, u) pair in float64."""

        if self.source == "bump":
            nblobs = int(torch.randint(1, 9, ()).item())
            f = self.solver.random_bump_source(nblobs=nblobs, l_src=8)
        else:
            # random node in the middle of the grid
            ir = int(torch.randint(self.nr // 5, 4 * self.nr // 5, ()).item())
            r = self.solver.r[ir].item()
            f = self.solver.ball_source(radius=r)

        u = self.solver.solve(f)
        r2f = self.solver.r.reshape(-1, 1, 1) ** 2 * f

        return r2f, u

    def scale(self, g):
        """Scale factor as the RMS, averaged over the sphere and over the radial nodes."""
        return self.sphere_mean(g**2).mean().sqrt()

    def __getitem__(self, index):

        with torch.inference_mode():
            with torch.no_grad():
                inp, tar = self._get_sample()

                if self.normalize:
                    s = self.scale(inp)
                    inp, tar = inp / s, tar / s

        return inp.float(), tar.float()
