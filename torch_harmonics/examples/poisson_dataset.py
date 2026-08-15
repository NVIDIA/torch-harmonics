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

from .poisson_equation import RadialPoissonSolver


class PoissonDataset(torch.utils.data.Dataset):
    """Custom Dataset class for Poisson training data

    Parameters
    ----------
    dims : tuple, optional
        Number of latitude, longitude and radial points, by default (64, 128, 256)
    grid : str, optional
        Angular grid type, by default "legendre-gauss"
    num_examples : int, optional
        Number of examples, by default 32
    device : torch.device, optional
        Device to use, by default torch.device("cpu")
    normalize : bool, optional
        Whether to normalize the input and target, by default True
    domain : str, optional
        Either "half-line" or "exterior", by default "half-line"
    R : float, optional
        Inner radius for exterior domain, by default None
    nblobs : int or tuple of int, optional
        Number of blobs in each source, by default (1, 8)
    l_src : int, optional
        Angular band limit of the source, by default 8
    positive : bool, optional
        Draw only positive sources, by default False

    Returns
    -------
    inp : torch.Tensor
        Source, shape (nr, nlat, nlon)
    tar : torch.Tensor
        Solution, shape (nr, nlat, nlon)
    """

    def __init__(
        self,
        dims=(64, 128, 256),
        grid="legendre-gauss",
        domain="half-line",
        R=None,
        nblobs=(1, 8),
        l_src=8,
        positive=False,
        num_examples=32,
        device=torch.device("cpu"),
        normalize=True,
    ):
        self.num_examples = num_examples
        self.device = device
        self.normalize = normalize
        self.nblobs = nblobs
        self.l_src = l_src
        self.positive = positive
        self.nlat, self.nlon, self.nr = dims

        self.solver = RadialPoissonSolver(
            self.nlat,
            self.nlon,
            self.nr,
            grid=grid,
            domain=domain,
            R=R,
        ).to(self.device)

    def __len__(self):
        return self.num_examples

    def _get_sample(self):
        """Get one unscaled source + solution pair."""

        f = self.solver.random_source(nblobs=self.nblobs, l_src=self.l_src, positive=self.positive)
        u = self.solver.solve(f)

        return f.float(), u.float()

    def scale(self, f):
        """Scale a pair by the L2 norm of the source in the measure r**2 dr dn."""

        w, r = self.solver.w, self.solver.r
        return ((w * r**2) * (f**2).mean(dim=(-1, -2))).sum().sqrt()

    def __getitem__(self, index):

        with torch.inference_mode():
            with torch.no_grad():
                inp, tar = self._get_sample()

                if self.normalize:
                    s = self.scale(inp)
                    inp, tar = inp / s, tar / s

        return inp.clone(), tar.clone()
