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

from torch_harmonics import AttentionS2, NeighborhoodAttentionS2

# ------------------------------------------------------------------------------
# AttentionS2 (global)
# ------------------------------------------------------------------------------


def _attn_setup(batch, channels, num_heads, nlat, nlon):
    def setup(device, dtype):
        # keep model in fp32; use autocast for fp16/bf16 — matches real usage
        # (casting the full model to fp16 produces NaN from softmax overflow)
        fp16_types = {torch.float16, torch.bfloat16}
        model_dtype = torch.float32
        attn = AttentionS2(
            in_channels=channels,
            num_heads=num_heads,
            in_shape=(nlat, nlon),
            out_shape=(nlat, nlon),
        ).to(device=device, dtype=model_dtype)
        x = torch.randn(batch, channels, nlat, nlon, dtype=torch.float32, device=device, requires_grad=True)
        return {"attn": attn, "x": x, "dtype": dtype, "use_autocast": dtype in fp16_types, "device": device}

    return setup


def _attn_forward(state):
    if state["use_autocast"]:
        with torch.autocast(state["device"].type, dtype=state["dtype"]):
            return state["attn"](state["x"])
    return state["attn"](state["x"])


def _attn_backward(state, out):
    out.backward(torch.ones_like(out))


# ------------------------------------------------------------------------------
# NeighborhoodAttentionS2 (local)
# ------------------------------------------------------------------------------


def _nattn_setup(batch, channels, num_heads, nlat_in, nlon_in, nlat_out, nlon_out, theta_cutoff, optimized):
    def setup(device, dtype):
        fp16_types = {torch.float16, torch.bfloat16}
        attn = NeighborhoodAttentionS2(
            in_channels=channels,
            num_heads=num_heads,
            in_shape=(nlat_in, nlon_in),
            out_shape=(nlat_out, nlon_out),
            theta_cutoff=theta_cutoff,
            optimized_kernel=optimized,
        ).to(device=device, dtype=torch.float32)
        x = torch.randn(batch, channels, nlat_in, nlon_in, dtype=torch.float32, device=device, requires_grad=True)
        return {
            "attn": attn,
            "x": x,
            "channels": channels,
            "num_heads": num_heads,
            "nlat_in": nlat_in,
            "nlon_in": nlon_in,
            "nlat_out": nlat_out,
            "nlon_out": nlon_out,
            "theta_cutoff": theta_cutoff,
            "dtype": dtype,
            "use_autocast": dtype in fp16_types,
            "device": device,
        }

    return setup


def _nattn_forward(state):
    if state["use_autocast"]:
        with torch.autocast(state["device"].type, dtype=state["dtype"]):
            return state["attn"](state["x"])
    return state["attn"](state["x"])


def _nattn_backward(state, out):
    out.backward(torch.ones_like(out))


def _nattn_reference(state):
    attn_ref = NeighborhoodAttentionS2(
        in_channels=state["channels"],
        num_heads=state["num_heads"],
        in_shape=(state["nlat_in"], state["nlon_in"]),
        out_shape=(state["nlat_out"], state["nlon_out"]),
        theta_cutoff=state["theta_cutoff"],
        optimized_kernel=False,
    ).to(dtype=torch.float32)
    attn_ref.load_state_dict({k: v.cpu().float() for k, v in state["attn"].state_dict().items()})
    with torch.no_grad():
        return attn_ref(state["x"].detach().cpu().float())


# ------------------------------------------------------------------------------
# Benchmark configs — all parameters explicit per entry
# ------------------------------------------------------------------------------

_ATTN_CONFIGS = [
    # global attention — quadratic in nlat*nlon, keep resolution modest
    dict(
        name="attn_s2_global_1deg_b1_c64_h1_float32_cuda",
        device="cuda",
        dtype=torch.float32,
        batch=1,
        channels=64,
        num_heads=1,
        nlat=180,
        nlon=360,
        tags=["attention", "global", "self"],
    ),
    dict(
        name="attn_s2_global_1deg_b1_c64_h1_float16_cuda",
        device="cuda",
        dtype=torch.float16,
        batch=1,
        channels=64,
        num_heads=1,
        nlat=180,
        nlon=360,
        tags=["attention", "global", "self"],
    ),
    dict(
        name="attn_s2_global_1deg_b1_c64_h1_bfloat16_cuda",
        device="cuda",
        dtype=torch.bfloat16,
        batch=1,
        channels=64,
        num_heads=1,
        nlat=180,
        nlon=360,
        tags=["attention", "global", "self"],
    ),
]

_NATTN_CONFIGS = [
    # self-attention (same in/out grid), CPU
    dict(
        name="nattn_s2_opt_1deg_b1_c64_h1_tc0017_float32_cpu",
        device="cpu",
        dtype=torch.float32,
        batch=1,
        channels=64,
        num_heads=1,
        nlat_in=180,
        nlon_in=360,
        nlat_out=180,
        nlon_out=360,
        theta_cutoff=0.017,
        optimized=True,
        skip_correctness=False,
        tags=["attention", "neighborhood", "cpu", "self"],
    ),
    # self-attention (same in/out grid), CUDA
    dict(
        name="nattn_s2_opt_1deg_b1_c64_h1_tc0017_float32_cuda",
        device="cuda",
        dtype=torch.float32,
        batch=1,
        channels=64,
        num_heads=1,
        nlat_in=180,
        nlon_in=360,
        nlat_out=180,
        nlon_out=360,
        theta_cutoff=0.017,
        optimized=True,
        skip_correctness=False,
        tags=["attention", "neighborhood", "self"],
    ),
    dict(
        name="nattn_s2_opt_1deg_b1_c64_h1_tc0017_float16_cuda",
        device="cuda",
        dtype=torch.float16,
        batch=1,
        channels=64,
        num_heads=1,
        nlat_in=180,
        nlon_in=360,
        nlat_out=180,
        nlon_out=360,
        theta_cutoff=0.017,
        optimized=True,
        skip_correctness=False,
        tags=["attention", "neighborhood", "self"],
    ),
    dict(
        name="nattn_s2_opt_1deg_b1_c64_h1_tc0017_bfloat16_cuda",
        device="cuda",
        dtype=torch.bfloat16,
        batch=1,
        channels=64,
        num_heads=1,
        nlat_in=180,
        nlon_in=360,
        nlat_out=180,
        nlon_out=360,
        theta_cutoff=0.017,
        optimized=True,
        skip_correctness=False,
        tags=["attention", "neighborhood", "self"],
    ),
    dict(
        name="nattn_s2_opt_1deg_b1_c64_h1_tc003_float32_cuda",
        device="cuda",
        dtype=torch.float32,
        batch=1,
        channels=64,
        num_heads=1,
        nlat_in=180,
        nlon_in=360,
        nlat_out=180,
        nlon_out=360,
        theta_cutoff=0.03,
        optimized=True,
        skip_correctness=False,
        tags=["attention", "neighborhood", "self"],
    ),
    dict(
        name="nattn_s2_opt_1deg_b1_c64_h1_tc003_float16_cuda",
        device="cuda",
        dtype=torch.float16,
        batch=1,
        channels=64,
        num_heads=1,
        nlat_in=180,
        nlon_in=360,
        nlat_out=180,
        nlon_out=360,
        theta_cutoff=0.03,
        optimized=True,
        skip_correctness=False,
        tags=["attention", "neighborhood", "self"],
    ),
    dict(
        name="nattn_s2_opt_1deg_b1_c64_h1_tc003_bfloat16_cuda",
        device="cuda",
        dtype=torch.bfloat16,
        batch=1,
        channels=64,
        num_heads=1,
        nlat_in=180,
        nlon_in=360,
        nlat_out=180,
        nlon_out=360,
        theta_cutoff=0.03,
        optimized=True,
        skip_correctness=False,
        tags=["attention", "neighborhood", "self"],
    ),
    # self-attention (same in/out grid), half-degree, theta_cutoff=0.017, CUDA
    dict(
        name="nattn_s2_opt_hdeg_b1_c64_h1_tc0017_float32_cuda",
        device="cuda",
        dtype=torch.float32,
        batch=1,
        channels=64,
        num_heads=1,
        nlat_in=360,
        nlon_in=720,
        nlat_out=360,
        nlon_out=720,
        theta_cutoff=0.017,
        optimized=True,
        skip_correctness=True,
        tags=["attention", "neighborhood", "self"],
    ),
    dict(
        name="nattn_s2_opt_hdeg_b1_c64_h1_tc0017_float16_cuda",
        device="cuda",
        dtype=torch.float16,
        batch=1,
        channels=64,
        num_heads=1,
        nlat_in=360,
        nlon_in=720,
        nlat_out=360,
        nlon_out=720,
        theta_cutoff=0.017,
        optimized=True,
        skip_correctness=True,
        tags=["attention", "neighborhood", "self"],
    ),
    dict(
        name="nattn_s2_opt_hdeg_b1_c64_h1_tc0017_bfloat16_cuda",
        device="cuda",
        dtype=torch.bfloat16,
        batch=1,
        channels=64,
        num_heads=1,
        nlat_in=360,
        nlon_in=720,
        nlat_out=360,
        nlon_out=720,
        theta_cutoff=0.017,
        optimized=True,
        skip_correctness=True,
        tags=["attention", "neighborhood", "self"],
    ),
    # self-attention (same in/out grid), half-degree, theta_cutoff=0.03, CUDA
    dict(
        name="nattn_s2_opt_hdeg_b1_c64_h1_tc003_float32_cuda",
        device="cuda",
        dtype=torch.float32,
        batch=1,
        channels=64,
        num_heads=1,
        nlat_in=360,
        nlon_in=720,
        nlat_out=360,
        nlon_out=720,
        theta_cutoff=0.03,
        optimized=True,
        skip_correctness=True,
        tags=["attention", "neighborhood", "self"],
    ),
    dict(
        name="nattn_s2_opt_hdeg_b1_c64_h1_tc003_float16_cuda",
        device="cuda",
        dtype=torch.float16,
        batch=1,
        channels=64,
        num_heads=1,
        nlat_in=360,
        nlon_in=720,
        nlat_out=360,
        nlon_out=720,
        theta_cutoff=0.03,
        optimized=True,
        skip_correctness=True,
        tags=["attention", "neighborhood", "self"],
    ),
    dict(
        name="nattn_s2_opt_hdeg_b1_c64_h1_tc003_bfloat16_cuda",
        device="cuda",
        dtype=torch.bfloat16,
        batch=1,
        channels=64,
        num_heads=1,
        nlat_in=360,
        nlon_in=720,
        nlat_out=360,
        nlon_out=720,
        theta_cutoff=0.03,
        optimized=True,
        skip_correctness=True,
        tags=["attention", "neighborhood", "self"],
    ),
    # cross-attention (different in/out grid), half-degree to 1-degree, theta_cutoff=0.017, CUDA
    dict(
        name="nattn_s2_opt_h1deg_b1_c64_h1_tc0017_float32_cuda",
        device="cuda",
        dtype=torch.float32,
        batch=1,
        channels=64,
        num_heads=1,
        nlat_in=360,
        nlon_in=720,
        nlat_out=180,
        nlon_out=360,
        theta_cutoff=0.017,
        optimized=True,
        skip_correctness=True,
        tags=["attention", "neighborhood", "cross"],
    ),
    dict(
        name="nattn_s2_opt_h1deg_b1_c64_h1_tc0017_float16_cuda",
        device="cuda",
        dtype=torch.float16,
        batch=1,
        channels=64,
        num_heads=1,
        nlat_in=360,
        nlon_in=720,
        nlat_out=180,
        nlon_out=360,
        theta_cutoff=0.017,
        optimized=True,
        skip_correctness=True,
        tags=["attention", "neighborhood", "cross"],
    ),
    dict(
        name="nattn_s2_opt_h1deg_b1_c64_h1_tc0017_bfloat16_cuda",
        device="cuda",
        dtype=torch.bfloat16,
        batch=1,
        channels=64,
        num_heads=1,
        nlat_in=360,
        nlon_in=720,
        nlat_out=180,
        nlon_out=360,
        theta_cutoff=0.017,
        optimized=True,
        skip_correctness=True,
        tags=["attention", "neighborhood", "cross"],
    ),
    # cross-attention (different in/out grid), half-degree to 1-degree, theta_cutoff=0.03, CUDA
    dict(
        name="nattn_s2_opt_h1deg_b1_c64_h1_tc003_float32_cuda",
        device="cuda",
        dtype=torch.float32,
        batch=1,
        channels=64,
        num_heads=1,
        nlat_in=360,
        nlon_in=720,
        nlat_out=180,
        nlon_out=360,
        theta_cutoff=0.03,
        optimized=True,
        skip_correctness=True,
        tags=["attention", "neighborhood", "cross"],
    ),
    dict(
        name="nattn_s2_opt_h1deg_b1_c64_h1_tc003_float16_cuda",
        device="cuda",
        dtype=torch.float16,
        batch=1,
        channels=64,
        num_heads=1,
        nlat_in=360,
        nlon_in=720,
        nlat_out=180,
        nlon_out=360,
        theta_cutoff=0.03,
        optimized=True,
        skip_correctness=True,
        tags=["attention", "neighborhood", "cross"],
    ),
    dict(
        name="nattn_s2_opt_h1deg_b1_c64_h1_tc003_bfloat16_cuda",
        device="cuda",
        dtype=torch.bfloat16,
        batch=1,
        channels=64,
        num_heads=1,
        nlat_in=360,
        nlon_in=720,
        nlat_out=180,
        nlon_out=360,
        theta_cutoff=0.03,
        optimized=True,
        skip_correctness=True,
        tags=["attention", "neighborhood", "cross"],
    ),
    # cross-attention (different in/out grid), 1-degree to half-degree, theta_cutoff=0.017, CUDA
    dict(
        name="nattn_s2_opt_1hdeg_b1_c64_h1_tc0017_float32_cuda",
        device="cuda",
        dtype=torch.float32,
        batch=1,
        channels=64,
        num_heads=1,
        nlat_in=180,
        nlon_in=360,
        nlat_out=360,
        nlon_out=720,
        theta_cutoff=0.017,
        optimized=True,
        skip_correctness=True,
        tags=["attention", "neighborhood", "cross"],
    ),
    dict(
        name="nattn_s2_opt_1hdeg_b1_c64_h1_tc0017_float16_cuda",
        device="cuda",
        dtype=torch.float16,
        batch=1,
        channels=64,
        num_heads=1,
        nlat_in=180,
        nlon_in=360,
        nlat_out=360,
        nlon_out=720,
        theta_cutoff=0.017,
        optimized=True,
        skip_correctness=True,
        tags=["attention", "neighborhood", "cross"],
    ),
    dict(
        name="nattn_s2_opt_1hdeg_b1_c64_h1_tc0017_bfloat16_cuda",
        device="cuda",
        dtype=torch.bfloat16,
        batch=1,
        channels=64,
        num_heads=1,
        nlat_in=180,
        nlon_in=360,
        nlat_out=360,
        nlon_out=720,
        theta_cutoff=0.017,
        optimized=True,
        skip_correctness=True,
        tags=["attention", "neighborhood", "cross"],
    ),
    # cross-attention (different in/out grid), 1-degree to half-degree, theta_cutoff=0.03, CUDA
    dict(
        name="nattn_s2_opt_1hdeg_b1_c64_h1_tc003_float32_cuda",
        device="cuda",
        dtype=torch.float32,
        batch=1,
        channels=64,
        num_heads=1,
        nlat_in=180,
        nlon_in=360,
        nlat_out=360,
        nlon_out=720,
        theta_cutoff=0.03,
        optimized=True,
        skip_correctness=True,
        tags=["attention", "neighborhood", "cross"],
    ),
    dict(
        name="nattn_s2_opt_1hdeg_b1_c64_h1_tc003_float16_cuda",
        device="cuda",
        dtype=torch.float16,
        batch=1,
        channels=64,
        num_heads=1,
        nlat_in=180,
        nlon_in=360,
        nlat_out=360,
        nlon_out=720,
        theta_cutoff=0.03,
        optimized=True,
        skip_correctness=True,
        tags=["attention", "neighborhood", "cross"],
    ),
    dict(
        name="nattn_s2_opt_1hdeg_b1_c64_h1_tc003_bfloat16_cuda",
        device="cuda",
        dtype=torch.bfloat16,
        batch=1,
        channels=64,
        num_heads=1,
        nlat_in=180,
        nlon_in=360,
        nlat_out=360,
        nlon_out=720,
        theta_cutoff=0.03,
        optimized=True,
        skip_correctness=True,
        tags=["attention", "neighborhood", "cross"],
    ),
]

for cfg in _ATTN_CONFIGS:
    register(
        BenchmarkEntry(
            name=cfg["name"],
            device=cfg["device"],
            dtype=cfg["dtype"],
            setup=_attn_setup(batch=cfg["batch"], channels=cfg["channels"], num_heads=cfg["num_heads"], nlat=cfg["nlat"], nlon=cfg["nlon"]),
            forward=_attn_forward,
            backward=_attn_backward,
            reference=None,
            tags=cfg["tags"],
        )
    )

for cfg in _NATTN_CONFIGS:
    register(
        BenchmarkEntry(
            name=cfg["name"],
            device=cfg["device"],
            dtype=cfg["dtype"],
            setup=_nattn_setup(
                batch=cfg["batch"],
                channels=cfg["channels"],
                num_heads=cfg["num_heads"],
                nlat_in=cfg["nlat_in"],
                nlon_in=cfg["nlon_in"],
                nlat_out=cfg["nlat_out"],
                nlon_out=cfg["nlon_out"],
                theta_cutoff=cfg["theta_cutoff"],
                optimized=cfg["optimized"],
            ),
            forward=_nattn_forward,
            backward=_nattn_backward,
            reference=_nattn_reference,
            skip_correctness=cfg["skip_correctness"],
            tags=cfg["tags"],
        )
    )
