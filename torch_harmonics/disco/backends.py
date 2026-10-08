# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
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

r"""
DISCO backends: an implementation of the psi contraction together with the state it needs.

The contraction of a DISCO layer can run three ways:

* ``kpacked`` -- the tensor-core forward (WGMMA on sm_90a, tcgen05 on sm_100a), for
  fp16/bf16 activations. psi in the blocked layout where every point carries all K filter
  values, plus the arcs its backward scatters with.
* ``optimized`` -- the compiled gather and scatter kernels, on CPU and CUDA alike: the
  operator dispatches to the device's kernel, so one backend serves both. psi in arc form.
* ``reference`` -- the torch implementation, in terms of a sparse COO psi. What a build
  without the kernels gets, or a layer with ``optimized_kernel=False``.

The layouts are described, and built, in :mod:`torch_harmonics.disco._psi`. A backend
registers the ones it reads and nothing else; see :mod:`torch_harmonics._backend` for how
selection works.

A backend serves every DISCO layer, serial or distributed, forward or transpose. The
layer describes its psi and leaves the kernels to the backend:

``transpose``
    Class attribute; whether psi is applied in the scatter direction.
``kernel_size``, ``nlon_out``
    As usual.
``_psi_coo()``
    The psi entries ``(ker_idx, row_idx, col_idx, vals)``. A distributed layer returns its
    rank's block.
``_psi_nlon``
    Longitudes of the grid psi's columns index: the input grid of the forward, the output
    grid of the transpose.
``_contract_shape``
    ``(nlat, nlon)`` of the contraction's result.
``_reference_psi(ker_idx, row_idx, col_idx, vals)``
    The sparse psi of the torch reference, whose extents are a property of the layer.
``_needs_split``
    Whether the layer contracts through the fused node with the spatial-first input
    gradient in play, which is what the per-basis-function row ranges are for.

The dtype is the one axis selection cannot settle, since fp16/bf16 -- under autocast in
particular -- is only known in ``forward``. The kpacked backend therefore checks it per
call and hands anything else to the arc kernels. That is a branch on a static dtype, so it
stays traceable.
"""

from typing import TYPE_CHECKING

import torch
from disco_helpers import optimized_kernels_is_available

from torch_harmonics._backend import BackendS2

from ._psi import build_arcs, build_kpacked, build_split
from .kernels_torch.disco_torch import _disco_s2_contraction_regular_torch, _disco_s2_transpose_contraction_regular_torch
from .optimized.disco_optimized import (
    _disco_s2_contraction_kpacked,
    _disco_s2_conv_optimized,
    _kpacked_build_available,
    _kpacked_k_pad,
    _kpacked_supported_on_device,
)

if optimized_kernels_is_available():
    from .optimized.disco_optimized import _disco_s2_contraction_regular_optimized, _disco_s2_transpose_contraction_regular_optimized

if TYPE_CHECKING:  # pragma: no cover
    from torch_harmonics.disco.convolution import DiscreteContinuousConv


#: the arc arrays, in the order the operators take them
_ARC_STATE = ("psi_row_ker", "psi_row_lat", "psi_seg_off", "psi_seg", "psi_val_off", "psi_vals")

_HALF_DTYPES = (torch.float16, torch.bfloat16)


def _amp_dtype(x: torch.Tensor) -> torch.dtype:
    """The dtype the contraction will actually run in: the autocast dtype if autocast is on for x's device."""
    device_type = x.device.type
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    return x.dtype


def _weight_contraction(x: torch.Tensor, weight: torch.Tensor, groups: int, groupsize: int) -> torch.Tensor:
    """(B, G*Cg, K, H, W) x (G, Og, Cg, K) -> (B, G*Og, H, W)."""
    B, _, K, H, W = x.shape
    x = x.reshape(B, groups, groupsize, K, H, W)
    out = torch.einsum("bgckxy,gock->bgoxy", x, weight).contiguous()
    return out.reshape(B, groups * weight.shape[1], H, W)


class DiscoBackendS2(BackendS2):
    """
    One way of evaluating the psi contraction of a DISCO layer.

    Besides ``available`` and ``prepare``, a backend provides the three operations the
    layers are built from. Each reads the prepared state back off the layer by name.

    ``contract(layer, x)``
        ``(B, C, H_in, W_in) -> (B, C, K, H_out, W_out)``, differentiable in ``x``.
    ``conv(layer, x, weight, groups, groupsize, recompute)``
        The contraction followed by the weight contraction,
        ``-> (B, groups * out_per_group, H_out, W_out)``. ``recompute`` asks for the
        K-expanded intermediate to be recomputed in backward rather than saved; a
        backend that cannot do that computes the same result the plain way.
    ``transpose(layer, x)``
        The scatter direction, ``(B, C, K, H_in, W_in) -> (B, C, H_out, W_out)``.
    """

    def contract(self, layer: "DiscreteContinuousConv", x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def conv(self, layer: "DiscreteContinuousConv", x: torch.Tensor, weight: torch.Tensor, groups: int, groupsize: int, recompute: bool = False) -> torch.Tensor:
        return _weight_contraction(self.contract(layer, x), weight, groups, groupsize)

    def transpose(self, layer: "DiscreteContinuousConv", x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReferenceBackend(DiscoBackendS2):
    """
    The torch reference, in terms of a sparse COO psi.

    Always available, so it is the last candidate. It contracts by repeatedly rolling the
    input -- the specification, not a fast path -- and has no recompute, so ``fused``
    changes nothing here but memory.
    """

    name = "reference"

    @classmethod
    def available(cls, layer, device):
        return True

    def prepare(self, layer, device):
        return {"psi": layer._reference_psi(*layer._psi_coo()).to(device)}

    def contract(self, layer, x):
        return _disco_s2_contraction_regular_torch(x, layer.psi, layer.nlon_out)

    def transpose(self, layer, x):
        return _disco_s2_transpose_contraction_regular_torch(x, layer.psi, layer.nlon_out)


class OptimizedBackend(DiscoBackendS2):
    """
    The compiled gather and scatter kernels, CPU or CUDA, on psi in arc form.

    The operators are registered for both devices, so the dispatcher picks the kernel
    and this one backend serves either -- the device is not a condition here, only
    whether the kernels were built and the layer wants them.
    """

    name = "optimized"

    @classmethod
    def available(cls, layer, device):
        return layer.optimized_kernel and optimized_kernels_is_available()

    def prepare(self, layer, device):
        return self._arc_state(layer, device)

    def _arc_state(self, layer, device):
        """The arc arrays, and the per-basis-function row ranges if the layer wants them."""
        arcs = build_arcs(*layer._psi_coo(), nlon=layer._psi_nlon)
        state = {name: t.to(device) for name, t in zip(_ARC_STATE, arcs)}

        self.split_row_offsets = None
        if layer._needs_split:
            split_ker, self.split_row_offsets = build_split(arcs, layer.kernel_size)
            state["psi_split_ker"] = split_ker.to(device)

        return state

    def _arcs(self, layer):
        return tuple(getattr(layer, name) for name in _ARC_STATE)

    def _split(self, layer):
        if self.split_row_offsets is None:
            return None
        return (layer.psi_split_ker, self.split_row_offsets)

    def contract(self, layer, x):
        return _disco_s2_contraction_regular_optimized(x, *self._arcs(layer), layer.kernel_size, *layer._contract_shape)

    def conv(self, layer, x, weight, groups, groupsize, recompute=False):
        # the contraction op follows autocast by itself; this path calls the kernels
        # directly, so it applies the cast the same way
        x = x.to(_amp_dtype(x))
        return _disco_s2_conv_optimized(x, weight, self._arcs(layer), None, self._split(layer), layer.kernel_size, *layer._contract_shape, groups, groupsize, recompute)

    def transpose(self, layer, x):
        return _disco_s2_transpose_contraction_regular_optimized(x, *self._arcs(layer), layer.kernel_size, *layer._contract_shape)


class KpackedBackend(OptimizedBackend):
    """
    The tensor-core forward (WGMMA on sm_90a, tcgen05 on sm_100a), with the arc scatter as backward.

    Serves fp16/bf16 activations and hands anything else to the arc kernels, which it
    also needs for the backward. The layout is built only on a device that can run it: it
    is padded per point to K_pad values, a sizeable buffer to hold for a kernel that could
    never launch.

    Available only for the forward direction, on a device and build the kernels were
    compiled for (see ``_kpacked_supported_on_device``), for K <= 16 and ``nlon_out`` a
    multiple of 8. Declines in ``prepare`` when the basis functions do not share one
    support, which the layout requires and which is only known once it is built.
    """

    name = "kpacked"

    @classmethod
    def available(cls, layer, device):
        return (
            super().available(layer, device)
            and not layer.transpose
            and device.type == "cuda"
            and _kpacked_build_available()
            and _kpacked_supported_on_device(device.index if device.index is not None else torch.cuda.current_device())
            and _kpacked_k_pad(layer.kernel_size) is not None
            and layer.nlon_out % 8 == 0
        )

    def prepare(self, layer, device):
        kpacked = build_kpacked(*layer._psi_coo(), layer.kernel_size, _kpacked_k_pad(layer.kernel_size), layer._contract_shape[0], layer._psi_nlon)
        if kpacked is None:
            return None

        state = self._arc_state(layer, device)
        for name, t in zip(("psi_kpacked_idx", "psi_kpacked_vals", "psi_kpacked_offset"), kpacked):
            state[name] = t.to(device)
        return state

    def _kpacked(self, layer):
        return (layer.psi_kpacked_idx, layer.psi_kpacked_vals, layer.psi_kpacked_offset)

    def contract(self, layer, x):
        dtype = _amp_dtype(x)
        if dtype not in _HALF_DTYPES:
            return super().contract(layer, x)
        return _disco_s2_contraction_kpacked(x.to(dtype), self._kpacked(layer), self._arcs(layer), layer.kernel_size, *layer._contract_shape)

    def conv(self, layer, x, weight, groups, groupsize, recompute=False):
        dtype = _amp_dtype(x)
        if dtype not in _HALF_DTYPES:
            return super().conv(layer, x, weight, groups, groupsize, recompute)
        return _disco_s2_conv_optimized(
            x.to(dtype), weight, self._arcs(layer), self._kpacked(layer), self._split(layer), layer.kernel_size, *layer._contract_shape, groups, groupsize, recompute
        )


#: Every backend, most specific first; see :mod:`torch_harmonics._backend`. The
#: transpose layers use the same list: the kpacked backend declines them itself.
BACKENDS = (KpackedBackend, OptimizedBackend, ReferenceBackend)
