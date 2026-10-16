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

from torch_harmonics.integration import QuadratureS2

from .poisson_equation import RadialPoissonSolver


class RadialPoissonDataset(torch.utils.data.Dataset):
    """Custom Dataset class for Poisson training data

    Parameters
    ----------
    grid : SphericalProductGrid
        Grid the samples are generated on, of shape (nr, nlat, nlon), on the half-line or the
        exterior domain; see RadialPoissonSolver
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

    # new in v1.0.0rc1, so there is no signature before the grid descriptors to reject
    _legacy_signature_exempt = "new in v1.0.0rc1"

    def __init__(
        self,
        grid,
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

        # the solver keeps its buffers in float64 and solves in the precision of the source
        self.solver = RadialPoissonSolver(grid).to(self.device)

        self.grid = self.solver.grid
        self.nr, self.nlat, self.nlon = self.grid.shape

        # mean over the sphere, on every radial node alike, for the scale
        self.sphere_mean = QuadratureS2(self.grid, normalize=True).to(self.device)

    def __len__(self):
        return self.num_examples

    def _get_sample(self):
        """Get one unscaled (r**2 f, u) pair in float32."""

        # the samples are float32, so they are solved in float32 too, which agrees with a
        # float64 solve to a few 1e-7 of the solution's maximum
        dtype = torch.float32

        if self.source == "bump":
            nblobs = int(torch.randint(1, 9, ()).item())
            f = self.solver.random_bump_source(nblobs=nblobs, l_src=8, dtype=dtype)
        else:
            # random node in the middle of the grid
            ir = int(torch.randint(self.nr // 5, 4 * self.nr // 5, ()).item())
            r = self.solver.r[ir].item()
            f = self.solver.ball_source(radius=r, dtype=dtype)

        u = self.solver.solve(f)
        r2f = self.solver.r.to(dtype).reshape(-1, 1, 1) ** 2 * f

        return r2f, u

    def scale(self, g):
        """Scale factor as the RMS, averaged over the sphere and over the radial nodes."""
        # squared relative to the largest value, which cannot overflow: r**2 f squares to
        # beyond float32 from radii of about 4e9 on, which a grid in physical units can reach
        m = g.abs().amax().clamp_min(torch.finfo(g.dtype).tiny)
        return m * self.sphere_mean((g / m) ** 2).mean().sqrt()

    def __getitem__(self, index):

        with torch.inference_mode():
            with torch.no_grad():
                inp, tar = self._get_sample()

                if self.normalize:
                    s = self.scale(inp)
                    inp, tar = inp / s, tar / s

        return inp.clone(), tar.clone()
