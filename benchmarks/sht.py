# coding=utf-8
# SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
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

import torch
from bench import BenchmarkEntry, register

from torch_harmonics import InverseRealSHT, RealSHT

# ------------------------------------------------------------------------------
# Setup / forward / backward / reference
# ------------------------------------------------------------------------------


def _sht_setup(nlat, nlon, batch):
    def setup(device, dtype):
        x = torch.randn(batch, nlat, nlon, dtype=dtype, device=device, requires_grad=True)
        sht = RealSHT(nlat=nlat, nlon=nlon).to(device=device, dtype=dtype)
        return {"x": x, "sht": sht, "nlat": nlat, "nlon": nlon}

    return setup


def _sht_forward(state):
    return state["sht"](state["x"])


def _sht_backward(state, out):
    out.backward(torch.ones_like(out))


def _sht_reference(state):
    sht_ref = RealSHT(nlat=state["nlat"], nlon=state["nlon"]).to(dtype=torch.float64)
    return sht_ref(state["x"].detach().cpu().double())


# iSHT input is complex — map real compute dtype to its complex counterpart
_REAL_TO_COMPLEX = {
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}


def _isht_setup(nlat, nlon, batch):
    def setup(device, dtype):
        complex_dtype = _REAL_TO_COMPLEX[dtype]
        sht_ref = RealSHT(nlat=nlat, nlon=nlon)
        x_hat = sht_ref(torch.randn(batch, nlat, nlon, dtype=torch.float64)).to(dtype=complex_dtype, device=device).detach().requires_grad_(True)
        isht = InverseRealSHT(nlat=nlat, nlon=nlon).to(device=device, dtype=dtype)
        return {"x_hat": x_hat, "isht": isht, "nlat": nlat, "nlon": nlon}

    return setup


def _isht_forward(state):
    return state["isht"](state["x_hat"])


def _isht_backward(state, out):
    out.backward(torch.ones_like(out))


def _isht_reference(state):
    isht_ref = InverseRealSHT(nlat=state["nlat"], nlon=state["nlon"]).to(dtype=torch.float64)
    return isht_ref(state["x_hat"].detach().cpu().cdouble())


# ------------------------------------------------------------------------------
# Benchmark configs — all parameters explicit per entry
# ------------------------------------------------------------------------------

_SHT_CONFIGS = [
    # 1 degree, CPU
    dict(name="sht_fwd_bwd_1deg_b8_float32_cpu", device="cpu", dtype=torch.float32, nlat=180, nlon=360, batch=8, tags=["sht", "cpu"]),
    # 1 degree, CUDA
    dict(name="sht_fwd_bwd_1deg_b4096_float32_cuda", device="cuda", dtype=torch.float32, nlat=180, nlon=360, batch=8, tags=["sht"]),
    # half degree, CUDA
    dict(name="sht_fwd_bwd_hdeg_b1_float32_cuda", device="cuda", dtype=torch.float32, nlat=360, nlon=720, batch=1, tags=["sht"]),
]

_ISHT_CONFIGS = [
    # 1 degree, CPU
    dict(name="isht_fwd_bwd_1deg_b8_float32_cpu", device="cpu", dtype=torch.float32, nlat=180, nlon=360, batch=8, tags=["isht", "cpu"]),
    # 1 degree, CUDA
    dict(name="isht_fwd_bwd_1deg_b4096_float32_cuda", device="cuda", dtype=torch.float32, nlat=180, nlon=360, batch=8, tags=["isht"]),
    # half degree, CUDA
    dict(name="isht_fwd_bwd_hdeg_b1_float32_cuda", device="cuda", dtype=torch.float32, nlat=360, nlon=720, batch=1, tags=["isht"]),
]

for cfg in _SHT_CONFIGS:
    register(
        BenchmarkEntry(
            name=cfg["name"],
            device=cfg["device"],
            dtype=cfg["dtype"],
            setup=_sht_setup(nlat=cfg["nlat"], nlon=cfg["nlon"], batch=cfg["batch"]),
            forward=_sht_forward,
            backward=_sht_backward,
            reference=_sht_reference,
            tags=cfg["tags"],
        )
    )

for cfg in _ISHT_CONFIGS:
    register(
        BenchmarkEntry(
            name=cfg["name"],
            device=cfg["device"],
            dtype=cfg["dtype"],
            setup=_isht_setup(nlat=cfg["nlat"], nlon=cfg["nlon"], batch=cfg["batch"]),
            forward=_isht_forward,
            backward=_isht_backward,
            reference=_isht_reference,
            tags=cfg["tags"],
        )
    )
