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

from math import ceil

import torch

from .heat_equation import SphericalHeatSolver
from .poisson_sphere import PoissonSolver
from .shallow_water_equations import ShallowWaterSolver


class PdeDataset(torch.utils.data.Dataset):
    """Custom Dataset class for PDE training data

    Parameters
    ----------
    dt : float
        Time step. Unused for "poisson", which is static
    nsteps : int
        Number of solver steps. Unused for "poisson"
    dims : tuple, optional
        Number of latitude and longitude points, by default (384, 768)
    grid : str, optional
        Grid type, by default "equiangular"
    pde : str, optional
        PDE type, "shallow water equations" (default), "heat equation" or "poisson"
    initial_condition : str, optional
        Initial condition type, by default "random"
    num_examples : int, optional
        Number of examples, by default 32
    device : torch.device, optional
        Device to use, by default torch.device("cpu")
    normalize : bool, optional
        Whether to normalize the input and target, by default True
    stream : torch.cuda.Stream, optional
        CUDA stream to use, by default None
    solver_kwargs : dict, optional
        Extra keyword arguments forwarded to the solver. For the heat equation this
        is where nr / domain / r_in / r_out / nu are set, and for Poisson nr / domain /
        R / rho_min / rho_max / gravity.
    source_kwargs : dict, optional
        Poisson only. Arguments for the random source, e.g. nblobs / l_src / margin,
        plus "boundary" ("zero" or "random") to pick the exterior inner Dirichlet data.
    seed : int, optional
        Poisson only. When given, sample i is drawn from seed + i, making the dataset
        deterministic. Left unset, every __getitem__ draws fresh randomness, which is
        the behaviour of the other PDEs.

    Returns
    -------
    inp : torch.Tensor
        Input tensor
    tar : torch.Tensor
        Target tensor
    """

    def __init__(
        self,
        dt=None,
        nsteps=None,
        dims=(384, 768),
        grid="equiangular",
        pde="shallow water equations",
        initial_condition="random",
        num_examples=32,
        device=torch.device("cpu"),
        normalize=True,
        stream=None,
        solver_kwargs=None,
        source_kwargs=None,
        seed=None,
    ):
        self.num_examples = num_examples
        self.device = device
        self.stream = stream

        self.nlat = dims[0]
        self.nlon = dims[1]
        self.grid = grid
        self.pde = pde

        # number of solver steps used to compute the target
        self.nsteps = nsteps
        self.normalize = normalize
        self.seed = seed

        solver_kwargs = dict(solver_kwargs or {})
        self.source_kwargs = dict(source_kwargs or {})
        self.boundary = self.source_kwargs.pop("boundary", "zero")

        # Poisson is static, so dt and nsteps carry no meaning there
        dt_solver = None if pde == "poisson" else dt / float(self.nsteps)

        if pde == "shallow water equations":
            # 2/3 dealiasing: the quadratic nonlinearity generates wavenumbers up to
            # 2*lmax, which would alias back onto the resolved band
            lmax = ceil(self.nlat / 3)
            mmax = lmax
            self.solver = ShallowWaterSolver(self.nlat, self.nlon, dt_solver, lmax=lmax, mmax=mmax, grid=grid, **solver_kwargs).to(self.device).float()
        elif pde == "heat equation":
            # the heat equation is linear, so there is no quadratic aliasing and no
            # need to dealias; lmax is limited only by the angular quadrature
            lmax = solver_kwargs.pop("lmax", self.nlat)
            mmax = solver_kwargs.pop("mmax", lmax)
            solver_kwargs.setdefault("nr", 48)
            # NOTE deliberately no .float(): the radial eigendecomposition spans many
            # decades (see SphericalHeatSolver.slow_mode_resolution) and must stay in
            # float64. Samples are cast to float32 on the way out of _get_sample.
            self.solver = SphericalHeatSolver(self.nlat, self.nlon, dt=dt_solver, lmax=lmax, mmax=mmax, grid=grid, **solver_kwargs).to(self.device)
        elif pde == "poisson":
            # static, and linear, so again no dealiasing is needed
            lmax = solver_kwargs.pop("lmax", self.nlat)
            mmax = solver_kwargs.pop("mmax", lmax)
            solver_kwargs.setdefault("nr", 256)
            if self.boundary == "random" and solver_kwargs.get("domain") != "exterior":
                raise ValueError("boundary='random' requires domain='exterior'")
            # as for the heat equation, deliberately no .float(): the Green's matrix spans
            # many decades and stays in float64. Samples are cast on the way out.
            self.solver = PoissonSolver(self.nlat, self.nlon, lmax=lmax, mmax=mmax, grid=grid, **solver_kwargs).to(self.device)
        else:
            raise NotImplementedError

        self.set_initial_condition(ictype=initial_condition)

        if self.normalize:
            inp0, tar0 = self._get_sample()
            self.inp_mean = torch.mean(inp0, dim=(-1, -2)).reshape(-1, 1, 1)
            self.inp_var = torch.var(inp0, dim=(-1, -2)).reshape(-1, 1, 1)
            if pde == "poisson":
                # unlike a time-stepped PDE, source and solution are different physical
                # quantities orders of magnitude apart, so the target needs its own stats
                self.tar_mean = torch.mean(tar0, dim=(-1, -2)).reshape(-1, 1, 1)
                self.tar_var = torch.var(tar0, dim=(-1, -2)).reshape(-1, 1, 1)

    def __len__(self):
        length = self.num_examples if self.ictype == "random" else 1
        return length

    def set_initial_condition(self, ictype="random"):
        self.ictype = ictype

    def set_num_examples(self, num_examples=32):
        self.num_examples = num_examples

    def _get_sample(self, index=0):
        if self.pde == "poisson":
            return self._get_poisson_sample(index)

        if self.pde == "heat equation":
            if self.ictype != "random":
                raise ValueError(f"the heat equation only supports initial_condition='random', got {self.ictype!r}")
            # (n_traj, n_int, lmax, mmax) -> drop the trajectory axis; the radial axis
            # occupies the channel slot the SWE solver uses for its physical fields
            inp = self.solver.random_initial_condition(n_traj=1)[0]
        elif self.ictype == "random":
            inp = self.solver.random_initial_condition(mach=0.2)
        elif self.ictype == "galewsky":
            inp = self.solver.galewsky_initial_condition()

        # solve pde for n steps to return the target
        tar = self.solver.timestep(inp, self.nsteps)
        inp = self.solver.spec2grid(inp)
        tar = self.solver.spec2grid(tar)

        if self.pde == "heat equation":
            inp, tar = inp.float(), tar.float()

        return inp, tar

    def _get_poisson_sample(self, index):
        """One (source, solution) pair. Static, so there is nothing to step."""

        seed = None if self.seed is None else self.seed + index
        f = self.solver.random_source(seed=seed, **self.source_kwargs)[0]

        if self.boundary == "random":
            v0 = self.solver.random_boundary_data(seed=seed, l_src=self.source_kwargs.get("l_src", 8))[0]
            u = self.solver.solve(f, v0)
            # provisional packing: the boundary field rides as the leading radial level
            f = torch.cat([v0.unsqueeze(0), f], dim=0)
        else:
            u = self.solver.solve(f)

        return f.float(), u.float()

    def __getitem__(self, index):

        # if self.stream is None:
        #     self.stream = torch.cuda.Stream()

        # with torch.cuda.stream(self.stream):
        #     with torch.inference_mode():
        #         with torch.no_grad():
        #             inp, tar = self._get_sample()

        #             if self.normalize:
        #                 inp = (inp - self.inp_mean) / torch.sqrt(self.inp_var)
        #                 tar = (tar - self.inp_mean) / torch.sqrt(self.inp_var)

        # self.stream.synchronize()

        with torch.inference_mode():
            with torch.no_grad():
                inp, tar = self._get_sample(index)

                if self.normalize:
                    if self.pde == "poisson":
                        # levels outside the compact support are identically zero, so the
                        # variance there vanishes and has to be guarded against
                        eps = torch.finfo(inp.dtype).tiny
                        inp = (inp - self.inp_mean) / torch.sqrt(self.inp_var + eps)
                        tar = (tar - self.tar_mean) / torch.sqrt(self.tar_var + eps)
                    else:
                        inp = (inp - self.inp_mean) / torch.sqrt(self.inp_var)
                        tar = (tar - self.inp_mean) / torch.sqrt(self.inp_var)

        return inp.clone(), tar.clone()
