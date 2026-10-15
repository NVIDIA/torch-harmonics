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
from bench import EQUIANGULAR_GRID_1DEG, EQUIANGULAR_GRID_HDEG, EQUIANGULAR_GRID_QDEG, BenchmarkEntry, maybe_autocast, register

from torch_harmonics import AttentionS2, NeighborhoodAttentionS2

# ------------------------------------------------------------------------------
# AttentionS2 (global)
# ------------------------------------------------------------------------------


def _attn_setup(batch, channels, num_heads, grid):
    def setup(device, dtype):
        # keep model in fp32; use autocast for fp16/bf16 — matches real usage
        # (casting the full model to fp16 produces NaN from softmax overflow)
        model_dtype = torch.float32
        attn = AttentionS2(
            grid_in=grid,
            grid_out=grid,
            in_channels=channels,
            num_heads=num_heads,
        ).to(device=device, dtype=model_dtype)
        x = torch.randn(batch, channels, *grid.shape, dtype=torch.float32, device=device, requires_grad=True)
        return {"attn": attn, "x": x, "dtype": dtype, "device": device}

    return setup


def _attn_forward(state):
    with maybe_autocast(state["device"].type, state["dtype"]):
        return state["attn"](state["x"])


def _attn_backward(state, out):
    out.backward(torch.ones_like(out))


# ------------------------------------------------------------------------------
# NeighborhoodAttentionS2 (local)
# ------------------------------------------------------------------------------


def _nattn_setup(batch, channels, num_heads, grid_in, grid_out, theta_cutoff, optimized):
    def setup(device, dtype):
        attn = NeighborhoodAttentionS2(
            grid_in=grid_in,
            grid_out=grid_out,
            in_channels=channels,
            num_heads=num_heads,
            theta_cutoff=theta_cutoff,
            optimized_kernel=optimized,
        ).to(device=device, dtype=torch.float32)
        # query lives on the output grid; key/value on the input grid.
        # For self-attention (grid_in == grid_out) x_kv is unused (forward passes None).
        x_q = torch.randn(batch, channels, *grid_out.shape, dtype=torch.float32, device=device, requires_grad=True)
        x_kv = None
        if grid_in != grid_out:
            x_kv = torch.randn(batch, channels, *grid_in.shape, dtype=torch.float32, device=device, requires_grad=True)
        return {
            "attn": attn,
            "x_q": x_q,
            "x_kv": x_kv,
            "channels": channels,
            "num_heads": num_heads,
            "grid_in": grid_in,
            "grid_out": grid_out,
            "theta_cutoff": theta_cutoff,
            "dtype": dtype,
            "device": device,
        }

    return setup


def _nattn_forward(state):
    q, kv = state["x_q"], state["x_kv"]
    with maybe_autocast(state["device"].type, state["dtype"]):
        return state["attn"](q, kv, kv)


def _nattn_backward(state, out):
    out.backward(torch.ones_like(out))


def _nattn_reference(state):
    attn_ref = NeighborhoodAttentionS2(
        grid_in=state["grid_in"],
        grid_out=state["grid_out"],
        in_channels=state["channels"],
        num_heads=state["num_heads"],
        theta_cutoff=state["theta_cutoff"],
        optimized_kernel=False,
    ).to(dtype=torch.float32)
    attn_ref.load_state_dict({k: v.cpu().float() for k, v in state["attn"].state_dict().items()})
    q = state["x_q"].detach().cpu().float()
    kv = state["x_kv"].detach().cpu().float() if state["x_kv"] is not None else None
    with torch.no_grad():
        return attn_ref(q, kv, kv)


# ------------------------------------------------------------------------------
# Benchmark configs — all parameters explicit per entry
# ------------------------------------------------------------------------------

_ATTN_CONFIGS = [
    # global attention — quadratic in the number of grid points, keep resolution modest
    dict(
        name="attn_s2_global_1deg_b1_c64_h1_float32_cuda",
        device="cuda",
        dtype=torch.float32,
        batch=1,
        channels=64,
        num_heads=1,
        grid=EQUIANGULAR_GRID_1DEG,
        tags=["attention", "global", "self"],
    ),
    dict(
        name="attn_s2_global_1deg_b1_c64_h1_float16_cuda",
        device="cuda",
        dtype=torch.float16,
        batch=1,
        channels=64,
        num_heads=1,
        grid=EQUIANGULAR_GRID_1DEG,
        tags=["attention", "global", "self"],
    ),
    dict(
        name="attn_s2_global_1deg_b1_c64_h1_bfloat16_cuda",
        device="cuda",
        dtype=torch.bfloat16,
        batch=1,
        channels=64,
        num_heads=1,
        grid=EQUIANGULAR_GRID_1DEG,
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
        grid_in=EQUIANGULAR_GRID_1DEG,
        grid_out=EQUIANGULAR_GRID_1DEG,
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
        grid_in=EQUIANGULAR_GRID_1DEG,
        grid_out=EQUIANGULAR_GRID_1DEG,
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
        grid_in=EQUIANGULAR_GRID_1DEG,
        grid_out=EQUIANGULAR_GRID_1DEG,
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
        grid_in=EQUIANGULAR_GRID_1DEG,
        grid_out=EQUIANGULAR_GRID_1DEG,
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
        grid_in=EQUIANGULAR_GRID_1DEG,
        grid_out=EQUIANGULAR_GRID_1DEG,
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
        grid_in=EQUIANGULAR_GRID_1DEG,
        grid_out=EQUIANGULAR_GRID_1DEG,
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
        grid_in=EQUIANGULAR_GRID_1DEG,
        grid_out=EQUIANGULAR_GRID_1DEG,
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
        grid_in=EQUIANGULAR_GRID_HDEG,
        grid_out=EQUIANGULAR_GRID_HDEG,
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
        grid_in=EQUIANGULAR_GRID_HDEG,
        grid_out=EQUIANGULAR_GRID_HDEG,
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
        grid_in=EQUIANGULAR_GRID_HDEG,
        grid_out=EQUIANGULAR_GRID_HDEG,
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
        grid_in=EQUIANGULAR_GRID_HDEG,
        grid_out=EQUIANGULAR_GRID_HDEG,
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
        grid_in=EQUIANGULAR_GRID_HDEG,
        grid_out=EQUIANGULAR_GRID_HDEG,
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
        grid_in=EQUIANGULAR_GRID_HDEG,
        grid_out=EQUIANGULAR_GRID_HDEG,
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
        grid_in=EQUIANGULAR_GRID_HDEG,
        grid_out=EQUIANGULAR_GRID_1DEG,
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
        grid_in=EQUIANGULAR_GRID_HDEG,
        grid_out=EQUIANGULAR_GRID_1DEG,
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
        grid_in=EQUIANGULAR_GRID_HDEG,
        grid_out=EQUIANGULAR_GRID_1DEG,
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
        grid_in=EQUIANGULAR_GRID_HDEG,
        grid_out=EQUIANGULAR_GRID_1DEG,
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
        grid_in=EQUIANGULAR_GRID_HDEG,
        grid_out=EQUIANGULAR_GRID_1DEG,
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
        grid_in=EQUIANGULAR_GRID_HDEG,
        grid_out=EQUIANGULAR_GRID_1DEG,
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
        grid_in=EQUIANGULAR_GRID_1DEG,
        grid_out=EQUIANGULAR_GRID_HDEG,
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
        grid_in=EQUIANGULAR_GRID_1DEG,
        grid_out=EQUIANGULAR_GRID_HDEG,
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
        grid_in=EQUIANGULAR_GRID_1DEG,
        grid_out=EQUIANGULAR_GRID_HDEG,
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
        grid_in=EQUIANGULAR_GRID_1DEG,
        grid_out=EQUIANGULAR_GRID_HDEG,
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
        grid_in=EQUIANGULAR_GRID_1DEG,
        grid_out=EQUIANGULAR_GRID_HDEG,
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
        grid_in=EQUIANGULAR_GRID_1DEG,
        grid_out=EQUIANGULAR_GRID_HDEG,
        theta_cutoff=0.03,
        optimized=True,
        skip_correctness=True,
        tags=["attention", "neighborhood", "cross"],
    ),
    # quarter-degree self-attention. The resolution the production configs are
    # actually headed for, and the one where the kernels stop being latency-bound on
    # grid size alone: 721x1440 is ~1.04M query points, 16x the 1deg entries.
    # Correctness is skipped -- the dense torch reference is quadratic in the number of grid points
    # and does not fit at this size.
    dict(
        name="nattn_s2_opt_qdeg_b1_c64_h1_tc003_float32_cuda",
        device="cuda",
        dtype=torch.float32,
        batch=1,
        channels=64,
        num_heads=1,
        grid_in=EQUIANGULAR_GRID_QDEG,
        grid_out=EQUIANGULAR_GRID_QDEG,
        theta_cutoff=0.03,
        optimized=True,
        skip_correctness=True,
        tags=["attention", "neighborhood", "self"],
    ),
    dict(
        name="nattn_s2_opt_qdeg_b1_c64_h1_tc003_float16_cuda",
        device="cuda",
        dtype=torch.float16,
        batch=1,
        channels=64,
        num_heads=1,
        grid_in=EQUIANGULAR_GRID_QDEG,
        grid_out=EQUIANGULAR_GRID_QDEG,
        theta_cutoff=0.03,
        optimized=True,
        skip_correctness=True,
        tags=["attention", "neighborhood", "self"],
    ),
    dict(
        name="nattn_s2_opt_qdeg_b1_c64_h1_tc003_bfloat16_cuda",
        device="cuda",
        dtype=torch.bfloat16,
        batch=1,
        channels=64,
        num_heads=1,
        grid_in=EQUIANGULAR_GRID_QDEG,
        grid_out=EQUIANGULAR_GRID_QDEG,
        theta_cutoff=0.03,
        optimized=True,
        skip_correctness=True,
        tags=["attention", "neighborhood", "self"],
    ),
]

for cfg in _ATTN_CONFIGS:
    register(
        BenchmarkEntry(
            name=cfg["name"],
            device=cfg["device"],
            dtype=cfg["dtype"],
            setup=_attn_setup(batch=cfg["batch"], channels=cfg["channels"], num_heads=cfg["num_heads"], grid=cfg["grid"]),
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
                grid_in=cfg["grid_in"],
                grid_out=cfg["grid_out"],
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
