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

r"""
The plan of the regular DISCO backend with the frequency-space arc scatter, on sm_90a (H100) and
sm_100a (B200 / GB200).

Kernels:

* gather: the shipped kpacked forward for fp16/bf16 (as ``KpackedPlanS2``: WGMMA on sm_90a, tcgen05
  on sm_100a);
* scatter: the arc scatter in frequency space of ``kernels_cuda/regular/disco_cuda_bwd_fft.cu`` for
  fp16 and bf16, the shipped arc scatter for fp32 / fp64.

The frequency-space scatter is built into both arch-specific extensions, ``torch_harmonics.disco._C_sm90``
and ``_C_sm100`` (see ``disco_arch``), each registering two ops under its family's name, and selected on
compute capability 9.0 and 10.0 devices whose image the build holds (9.0a / 10.0a in
``TORCH_CUDA_ARCH_LIST``), the ones it was validated on:

``scatter_fft_<family>_plan``
    Everything the scatter derives from psi -- band entry lists, twiddle tables, the arcs' spectra --
    for a number of planes B * C and an input shape. This is the plan's call state, built once per
    key and kept.
``scatter_fft_<family>_run``
    The scatter of fp16 or bf16 activations through such a plan.

The backend gives the plan the ops of its device (``fft_plan_op``, ``fft_run_op``), as it does the
kpacked forward.

The spatial-first input gradient scatters each basis function on its own. The kernel needs those rows
as arcs of their own (offsets from 0, values of their own): on slices of the full arrays it would
walk arcs outside the slice. The backend builds them once, as static state.
"""

from typing import Optional, Tuple

import torch

from . import disco_arch
from .disco_plan import ARC_STATE, KpackedPlanS2, Tensors

#: the gradient dtypes the frequency-space scatter takes: it reads and writes them, and computes in fp32
FFT_SCATTER_DTYPES = (torch.float16, torch.bfloat16)

#: the fp32 values the frequency-space scatter reads (psi's own dtype may differ)
FFT_VALS = "psi_fft_vals"

#: per basis function k, its rows as arcs of their own: ``psi_fft_k{k}_{field}``
FFT_SPLIT_FIELDS = ("row_ker", "row_lat", "seg_off", "seg", "val_off", "vals")


def _split_name(k: int, field: str) -> str:
    return f"psi_fft_k{k}_{field}"


def _register_fakes(prefix: str) -> None:
    @torch.library.register_fake(f"disco_kernels::{prefix}_plan")
    def _(row_ker, row_lat, seg_off, seg, val_off, vals, bc, kernel_size, nlat_in, nlon_in, nlat_out, nlon_out):
        # a byte buffer whose size the configuration decides
        return row_ker.new_empty((torch.library.get_ctx().new_dynamic_size(),), dtype=torch.uint8)

    @torch.library.register_fake(f"disco_kernels::{prefix}_run")
    def _(inp, plan, row_ker, row_lat, seg_off, seg, val_off, vals, nlat_out, nlon_out):
        return inp.new_empty((inp.shape[0], inp.shape[1], nlat_out, nlon_out))


for _family, _prefix in disco_arch.FFT_SCATTER_OPS.items():
    if disco_arch.fft_scatter_registered(_family):
        _register_fakes(_prefix)


def fft_scatter_available(device: torch.device) -> bool:
    """
    Whether the frequency-space scatter can run on ``device``: a compute capability 9.0 or 10.0 GPU and a build holding its image.

    9.0 and 10.0 only, the devices it was validated on; a 10.3 (GB300) keeps the arc scatter.
    """
    if device.type != "cuda":
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    return disco_arch.fft_scatter_supported_on_device(index)


def build_fft_state(state: dict, split_row_offsets: Optional[Tuple[int, ...]]) -> dict:
    """
    The static state the frequency-space scatter adds to the kpacked backend's ``state``.

    The values in fp32, and with the spatial-first input gradient each basis function's rows as arcs of
    their own: arc and value offsets rebased to 0, ``row_ker`` zeros, the arcs and values they cover cut
    out. Built once, on the device of ``state``.
    """
    row_ker, row_lat, seg_off, seg, val_off, vals = (state[name] for name in ARC_STATE)
    vals32 = vals.to(torch.float32).contiguous()
    extra = {FFT_VALS: vals32}
    if split_row_offsets is None:
        return extra

    seg_off_host, val_off_host = seg_off.cpu(), val_off.cpu()
    for k in range(len(split_row_offsets) - 1):
        r0, r1 = split_row_offsets[k], split_row_offsets[k + 1]
        s0, s1 = int(seg_off_host[r0]), int(seg_off_host[r1])
        v0, v1 = int(val_off_host[r0]), int(val_off_host[r1])
        fields = {
            "row_ker": torch.zeros(r1 - r0, dtype=row_ker.dtype, device=row_ker.device),
            "row_lat": row_lat[r0:r1].contiguous(),
            "seg_off": (seg_off[r0 : r1 + 1] - s0).contiguous(),
            "seg": seg[s0:s1].contiguous(),
            "val_off": (val_off[r0 : r1 + 1] - v0).contiguous(),
            "vals": vals32[v0:v1].contiguous(),
        }
        for field in FFT_SPLIT_FIELDS:
            extra[_split_name(k, field)] = fields[field]
    return extra


class FftPlanS2(KpackedPlanS2):
    """
    The kpacked forward, and the frequency-space arc scatter for fp16 and bf16 gradients.

    The scatter's call state is its plan, a byte buffer from ``fft_plan_op`` per number of planes,
    dtype and input shape; for the spatial-first input gradient one per basis function. fp32 and fp64
    gradients take the arc scatter. ``fft_plan_op`` and ``fft_run_op`` are the ops of the arch-specific
    extension for the plan's device, which the backend sets (see ``disco_arch.fft_scatter_ops``).
    """

    has_call_state = True

    #: the frequency-space scatter's ops for the plan's device, e.g. ``torch.ops.disco_kernels.scatter_fft_sm90_plan.default``
    fft_plan_op = None
    fft_run_op = None

    def __init__(self, state_names, kernel_size, contract_shape, split_row_offsets=None):
        super().__init__(state_names, kernel_size, contract_shape, split_row_offsets)
        needed = [FFT_VALS]
        if self.has_split:
            needed += [_split_name(k, field) for k in range(kernel_size) for field in FFT_SPLIT_FIELDS]
        missing = [name for name in needed if name not in self._index]
        if missing:
            raise ValueError(f"{type(self).__name__} needs the state {missing[:6]}{' ...' if len(missing) > 6 else ''}")

    def _arcs32(self, t: Tensors) -> Tensors:
        """The arc arrays with the fp32 values."""
        return self.arcs(t)[:-1] + (self._get(t, FFT_VALS),)

    def _split_arcs(self, t: Tensors, k: int) -> Tensors:
        return tuple(self._get(t, _split_name(k, field)) for field in FFT_SPLIT_FIELDS)

    def has_scatter(self, dtype):
        return dtype in FFT_SCATTER_DTYPES

    def call_state(self, op, bc, dtype, in_shape, t):
        if dtype not in FFT_SCATTER_DTYPES or op == "gather":
            return ()
        nlat_in, nlon_in = in_shape
        plan = self.fft_plan_op
        if op == "scatter":
            return (plan(*self._arcs32(t), bc, self.kernel_size, self.nlat_out, self.nlon_out, nlat_in, nlon_in),)
        # scatter_split: one plan per basis function, over its own rows
        return tuple(plan(*self._split_arcs(t, k), bc, 1, self.nlat_out, self.nlon_out, nlat_in, nlon_in) for k in range(self.kernel_size))

    def scatter(self, g, t, tc, nlat_in, nlon_in):
        if g.dtype not in FFT_SCATTER_DTYPES or not tc:
            return super().scatter(g, t, tc, nlat_in, nlon_in)
        return self.fft_run_op(g.contiguous(), tc[0], *self._arcs32(t), nlat_in, nlon_in)

    def scatter_split(self, g, t, tc, nlat_in, nlon_in):
        if g.dtype not in FFT_SCATTER_DTYPES or not tc:
            return super().scatter_split(g, t, tc, nlat_in, nlon_in)
        g = g.contiguous()
        run = self.fft_run_op
        parts = []
        for k in range(self.kernel_size):
            if self.split_row_offsets[k + 1] == self.split_row_offsets[k]:
                parts.append(g.new_zeros((g.shape[0], g.shape[1], nlat_in, nlon_in)))
            else:
                parts.append(run(g, tc[k], *self._split_arcs(t, k), nlat_in, nlon_in))
        return torch.stack(parts, dim=2)
