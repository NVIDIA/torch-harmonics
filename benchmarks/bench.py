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

import dataclasses
import platform
import re
import time
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

# ------------------------------------------------------------------------------
# Timer
# ------------------------------------------------------------------------------


class Timer:
    """Single timed region using CUDA events on GPU, perf_counter on CPU."""

    def __init__(self, device: torch.device):
        self.device = device
        self.elapsed_ms: float = 0.0
        self._start_event = None
        self._end_event = None
        self._t0: float = 0.0

    def __enter__(self):
        if self.device.type == "cuda":
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.device.type == "cuda":
            self._end_event.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self._start_event.elapsed_time(self._end_event)
        else:
            self.elapsed_ms = (time.perf_counter() - self._t0) * 1e3
        return False


# ------------------------------------------------------------------------------
# BenchmarkEntry — pure data, no behaviour
# ------------------------------------------------------------------------------


@dataclasses.dataclass
class BenchmarkEntry:
    name: str

    # "cpu" | "cuda" | "cuda:H100" | "cuda:A100" | "cuda:GB200" | ...
    # The runner checks availability and skips entries whose device isn't present.
    # "cuda:<arch>" matches if the arch string appears in the GPU name (case-insensitive).
    device: str

    dtype: torch.dtype

    # setup(device, dtype) -> state dict
    # Called once before warmup. Return value is passed to forward/backward/reference.
    # Tensors that need gradients should have requires_grad=True here.
    setup: Callable[[torch.device, torch.dtype], dict[str, Any]]

    # forward(state) -> output tensor
    # Timed independently. Should not zero grads; the runner does that.
    forward: Callable[[dict], torch.Tensor]

    # backward(state, fwd_output) -> None   (optional)
    # Timed independently from forward. The runner rebuilds the graph (untimed forward)
    # and zeros grads on state tensors before each timed backward call.
    backward: Optional[Callable[[dict, torch.Tensor], None]] = None

    # reference(state) -> tensor
    # Called once (untimed) after the timed loop. Should return the "ground truth"
    # output (e.g. float64 on CPU) so the runner can compute L-inf error.
    reference: Optional[Callable[[dict], torch.Tensor]] = None

    # Set True to opt out of correctness checks (e.g. large grids where the
    # CPU float64 reference run would dominate wall time).
    skip_correctness: bool = False

    tags: list = dataclasses.field(default_factory=list)


# ------------------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------------------

_REGISTRY: list[BenchmarkEntry] = []


def register(entry: BenchmarkEntry) -> BenchmarkEntry:
    _REGISTRY.append(entry)
    return entry


def get_entries(
    name_filter: Optional[str] = None,
    tags: Optional[list] = None,
    device_filter: Optional[str] = None,
) -> list[BenchmarkEntry]:
    entries = list(_REGISTRY)
    if name_filter:
        entries = [e for e in entries if name_filter in e.name]
    if tags:
        entries = [e for e in entries if any(t in e.tags for t in tags)]
    if device_filter:
        entries = [e for e in entries if e.device == device_filter]
    return entries


# ------------------------------------------------------------------------------
# Device availability
# ------------------------------------------------------------------------------


def _get_cpu_name() -> str:
    """Return a cleaned-up CPU model string, or 'Generic CPU' if not determinable."""
    import subprocess

    def _clean(name: str) -> str:
        name = re.sub(r"\s+@\s+[\d.]+\s*GHz.*", "", name)  # drop "@ 2.00GHz" suffix
        name = name.replace("(R)", "").replace("(TM)", "")  # drop trademark noise
        return re.sub(r" {2,}", " ", name).strip()

    # Linux: /proc/cpuinfo — works on x86; ARM has no "model name" here
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return _clean(line.split(":", 1)[1].strip())
    except OSError:
        pass

    # Linux: lscpu — works on both x86 and aarch64 (Grace shows "Neoverse-V2")
    try:
        out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if line.startswith("Model name"):
                return _clean(line.split(":", 1)[1].strip())
    except Exception:
        pass

    # macOS
    try:
        out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL).strip()
        if out:
            return _clean(out)
    except Exception:
        pass

    name = platform.processor()
    return _clean(name) if name else "Generic CPU"


def _resolve_device(spec: str) -> tuple[bool, Optional[torch.device]]:
    """
    Returns (available, device).

    Spec forms:
      "cpu"          always available
      "cuda"         any CUDA GPU
      "cuda:H100"    CUDA GPU whose name contains "H100" (case-insensitive)
    """
    if spec == "cpu":
        return True, torch.device("cpu")

    if not torch.cuda.is_available():
        return False, None

    current = torch.device("cuda", torch.cuda.current_device())

    if spec == "cuda":
        return True, current

    if spec.startswith("cuda:"):
        arch = spec[5:]
        gpu_name = torch.cuda.get_device_properties(current).name
        if arch.lower() in gpu_name.lower():
            return True, current
        return False, None

    return False, None


# ------------------------------------------------------------------------------
# Grad zeroing helper
# ------------------------------------------------------------------------------


def _zero_grads(state: dict) -> None:
    for v in state.values():
        if isinstance(v, torch.Tensor) and v.requires_grad and v.grad is not None:
            v.grad = None
        elif isinstance(v, nn.Module):
            for p in v.parameters():
                if p.grad is not None:
                    p.grad = None


# ------------------------------------------------------------------------------
# Result
# ------------------------------------------------------------------------------


@dataclasses.dataclass
class BenchmarkResult:
    name: str
    device: str
    arch: str  # GPU model name, or "cpu"
    dtype: str
    fwd_ms: float  # avg over iters
    bwd_ms: Optional[float]  # avg over iters; None if no backward
    ref_error: Optional[float]  # L-inf vs reference output; None if no reference


# ------------------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------------------


def run_entry(
    entry: BenchmarkEntry,
    warmup: int = 3,
    iters: int = 20,
    skip_reference: bool = False,
) -> Optional[BenchmarkResult]:
    """
    Returns None if the entry's device requirement is not met on this machine.
    """
    available, device = _resolve_device(entry.device)
    if not available:
        return None

    if device.type == "cuda":
        torch.cuda.empty_cache()
        gpu = torch.cuda.get_device_properties(device).name
    else:
        gpu = _get_cpu_name()

    state = entry.setup(device, entry.dtype)

    # --- warmup ---
    for _ in range(warmup):
        out = entry.forward(state)
        if entry.backward is not None:
            _zero_grads(state)
            entry.backward(state, out)
        del out

    # --- timed forward ---
    # Detach immediately after each iteration so the computation graph (and all
    # its stored intermediates) is freed before the next forward call.
    # Without this, graphs accumulate across iters and exhaust GPU memory.
    fwd_times = []
    last_out = None
    for _ in range(iters):
        _zero_grads(state)
        with Timer(device) as t:
            out = entry.forward(state)
        fwd_times.append(t.elapsed_ms)
        last_out = out.detach()
        del out
    fwd_ms = sum(fwd_times) / len(fwd_times)

    # --- timed backward ---
    bwd_ms = None
    if entry.backward is not None:
        bwd_times = []
        for _ in range(iters):
            out = entry.forward(state)  # rebuild graph (untimed)
            _zero_grads(state)
            with Timer(device) as t:
                entry.backward(state, out)  # backward frees the graph
            bwd_times.append(t.elapsed_ms)
            del out
        bwd_ms = sum(bwd_times) / len(bwd_times)

    # --- reference error ---
    ref_error = None
    if not skip_reference and not entry.skip_correctness and entry.reference is not None and last_out is not None:
        ref = entry.reference(state)
        cur = last_out.cpu()
        if torch.is_complex(cur):
            ref_error = (cur.cdouble() - ref.cdouble()).abs().max().item()
        else:
            ref_error = (cur.double() - ref.double()).abs().max().item()

    return BenchmarkResult(
        name=entry.name,
        device=str(device),
        arch=gpu,
        dtype=str(entry.dtype),
        fwd_ms=fwd_ms,
        bwd_ms=bwd_ms,
        ref_error=ref_error,
    )
