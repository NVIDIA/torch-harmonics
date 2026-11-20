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
Experimental: the DISCO kernels from the Kernel Factory campaigns, for A/B comparisons.

The kernels live in ``kernels_cuda/kf/`` and build as one extension per kernel and
architecture (``TORCH_HARMONICS_BUILD_KF_KERNELS=1``, see ``setup.py``):

* ``_kf_fwd_sm90``, ``_kf_fwd_sm100`` -- the forward contraction on the kpacked psi
  (``disco_kernels::forward_kpacked``);
* ``_kf_bwd_sm90``, ``_kf_bwd_sm100`` -- the arc scatter (``disco_kernels::backward_regular``),
  i.e. the input gradient of the forward and the forward of the transpose.

They are wired in at the level of those two raw operators, so the autograd node, the
spatial-first input gradient and the recompute all stay as they are and only the kernel
underneath changes. :func:`install` routes a call to a Kernel Factory kernel when

* its psi belongs to a layer whose backend is ``regular-kf`` (see :func:`register`); every
  other layer, including the A arm of an A/B run, keeps the shipped kernels;
* the activations are fp16 (the kernels are fp16 only) and psi has the dtypes they read;
* an extension for the device's architecture was built.

Anything else falls through to the shipped operator. This is not the integration: the
kernels recompute their psi bookkeeping on every call and are routed by tensor identity
rather than by a backend of their own.
"""

import importlib
import warnings
import weakref
from typing import Optional

import torch

from .. import disco_kernels

#: compute capability -> architecture suffix of the extensions; sm_100a cubins do not load
#: on 10.3 (GB300), so only 10.0 is mapped
_ARCH = {(9, 0): "sm90", (10, 0): "sm100"}

_EXT = {}


def _extension(kind: str, device: torch.device):
    """The built extension for ``kind`` ("fwd" or "bwd") on ``device``, or None."""
    if device.type != "cuda":
        return None
    arch = _ARCH.get(torch.cuda.get_device_capability(device))
    if arch is None:
        return None
    name = f"{kind}_{arch}"
    if name not in _EXT:
        try:
            _EXT[name] = importlib.import_module(f"torch_harmonics.disco._kf_{name}")
        except ModuleNotFoundError:
            # not built
            _EXT[name] = None
        except ImportError as e:
            # built but not loadable: say so, or the backend would quietly decline
            warnings.warn(f"Kernel Factory DISCO kernel {name} is built but does not load: {e}")
            _EXT[name] = None
    return _EXT[name]


def available(device: torch.device) -> bool:
    """Whether a forward or a backward Kernel Factory kernel was built for ``device``."""
    return _extension("fwd", device) is not None or _extension("bwd", device) is not None


# storage pointer -> (weak reference to the psi tensor, rebased arc slices of it). Weak, so a
# layer that is freed stops being routed even if the allocator hands its memory to another one.
_OWNED = {}


def register(*tensors: torch.Tensor) -> None:
    """
    Route the operator calls that read these psi tensors to the Kernel Factory kernels.

    Called on every forward of a regular-kf layer, so a tensor that is already registered
    keeps its entry and with it the derived arrays cached for it (fp16 values, rebased arc
    slices); only a new tensor gets a fresh one. Entries of tensors that have been freed are
    dropped, so the caches do not outlive their layers.
    """
    for key in [k for k, (ref, _) in _OWNED.items() if ref() is None]:
        del _OWNED[key]
    for t in tensors:
        key = t.untyped_storage().data_ptr()
        entry = _OWNED.get(key)
        if entry is None or entry[0]() is not t:
            _OWNED[key] = (weakref.ref(t), {})


def _owner(t: torch.Tensor) -> Optional[dict]:
    """The per-tensor cache if ``t`` (or the tensor it is a slice of) is registered and alive."""
    key = t.untyped_storage().data_ptr()
    entry = _OWNED.get(key)
    if entry is None:
        return None
    ref, cache = entry
    if ref() is None:
        del _OWNED[key]
        return None
    return cache


def _rebased(cache: dict, seg_off, seg, val_off, vals):
    """
    The arcs of a row range as arrays of their own.

    The spatial-first input gradient passes each basis function's rows as slices of the
    offset arrays, with absolute offsets into the full ``seg`` and ``vals``. The Kernel Factory
    kernels were written against complete arrays, offsets starting at 0, so the range is cut
    out once per slice -- psi is static -- and cached.
    """
    key = (seg_off.storage_offset(), seg_off.numel())
    if key not in cache:
        s0, s1 = int(seg_off[0]), int(seg_off[-1])
        v0, v1 = int(val_off[0]), int(val_off[-1])
        if s0 == 0 and v0 == 0 and s1 == seg.shape[0] and v1 == vals.numel():
            cache[key] = None
        else:
            cache[key] = ((seg_off - s0).contiguous(), seg[s0:s1].contiguous(), (val_off - v0).contiguous(), vals[v0:v1].contiguous())
    rebased = cache[key]
    return (seg_off, seg, val_off, vals) if rebased is None else rebased


def _half_values(cache: dict, pack_val: torch.Tensor) -> torch.Tensor:
    """
    ``pack_val`` in fp16, converted once per layer.

    The kpacked layout keeps the values in the dtype of psi (fp32), and the shipped operator
    casts them to the activation dtype on every call; the Kernel Factory kernels read fp16.
    """
    if pack_val.dtype == torch.float16:
        return pack_val
    key = ("pack_val_fp16", pack_val.data_ptr())
    if key not in cache:
        cache[key] = pack_val.to(torch.float16).contiguous()
    return cache[key]


def forward_kpacked(inp, pack_idx, pack_val, pack_offset, kernel_size, nlat_out, nlon_out):
    ext = _extension("fwd", inp.device)
    cache = _owner(pack_idx) if ext is not None else None
    if cache is not None and inp.dtype == torch.float16:
        pack_val = _half_values(cache, pack_val)
        out = inp.new_empty((inp.shape[0], inp.shape[1], kernel_size, nlat_out, nlon_out))
        ext.run(inp.contiguous(), pack_idx, pack_val, pack_offset, kernel_size, nlon_out, out)
        return out
    return disco_kernels.forward_kpacked.default(inp, pack_idx, pack_val, pack_offset, kernel_size, nlat_out, nlon_out)


def backward_regular(inp, row_ker, row_lat, seg_off, seg, val_off, vals, kernel_size, nlat_out, nlon_out):
    ext = _extension("bwd", inp.device)
    cache = _owner(seg) if ext is not None else None
    if cache is not None and inp.dtype == torch.float16 and vals.dtype == torch.float32:
        seg_off, seg, val_off, vals = _rebased(cache, seg_off, seg, val_off, vals)
        out = inp.new_empty((inp.shape[0], inp.shape[1], nlat_out, nlon_out))
        ext.run(inp.contiguous(), row_ker, row_lat, seg_off, seg, val_off, vals, nlat_out, nlon_out, out)
        return out
    return disco_kernels.backward_regular.default(inp, row_ker, row_lat, seg_off, seg, val_off, vals, kernel_size, nlat_out, nlon_out)


def install() -> None:
    """Hook the two operators of :mod:`.disco_optimized` (idempotent)."""
    from . import disco_optimized

    disco_optimized._kernel_hooks["forward_kpacked"] = forward_kpacked
    disco_optimized._kernel_hooks["backward_regular"] = backward_regular
