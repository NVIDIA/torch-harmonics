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

The contraction of a DISCO layer on a regular grid can run three ways:

* ``regular-kpacked`` -- the tensor-core forward (WGMMA on sm_90a, tcgen05 on sm_100a), for
  fp16/bf16 activations. psi in the blocked layout where every point carries all K filter
  values, plus the arcs its backward scatters with.
* ``regular-optimized`` -- the compiled gather and scatter kernels, on CPU and CUDA alike:
  the operator dispatches to the device's kernel, so one backend serves both. psi in arc form.
* ``regular-reference`` -- the torch implementation, in terms of a sparse COO psi. What a
  build without the kernels gets, or a layer with ``optimized_kernel=False``.

and on a ragged one -- HEALPix, or a regular grid paired with one -- two ways, as for
neighborhood attention:

* ``ragged-optimized`` -- the compiled ragged gather and scatter, on CPU and CUDA. psi in
  arc form keyed per output point, plus the ring tables of the grid its columns index.
* ``ragged-reference`` -- the torch implementation, one sparse product per basis function.

The layouts are described, and built, in :mod:`torch_harmonics.disco._psi_layouts`. A
backend registers the ones it reads and nothing else; see :mod:`torch_harmonics._backend`
for how selection works.

A backend serves every DISCO layer, serial or distributed, forward or transpose. The
layer describes its psi and leaves the kernels to the backend:

``transpose``
    Class attribute; whether psi is applied in the scatter direction.
``ragged``
    Whether psi is keyed per point; the regular backends decline a ragged layer and the
    ragged ones a regular layer.
``kernel_size``, ``nlon_out``
    As usual; ``nlon_out`` on a regular layer only.
``npoints_out_local``
    Points of the ragged result: ``npoints_out`` on a serial layer, the rank's band of
    output points on a distributed one. Ragged layers only.
``_psi_coo()``
    The psi entries ``(ker_idx, row_idx, col_idx, vals)``. A distributed layer returns its
    rank's block.
``_psi_nlon``
    Longitudes of the grid psi's columns index: the input grid of the forward, the output
    grid of the transpose. Regular layers only.
``_psi_col_rings()``
    ``(ring_base, ring_size)`` of the rings psi's columns index, which the ragged arcs walk:
    those of the whole grid on a serial layer, of the rank's band on a distributed one.
    Ragged layers only.
``_contract_shape``
    ``(nlat, nlon)`` of the contraction's result. Regular layers only; a ragged result is
    ``npoints_out_local`` long.
``_reference_psi(ker_idx, row_idx, col_idx, vals)``
    The sparse psi of the torch reference, whose extents are a property of the layer.
``_needs_split``
    Whether the layer contracts through the fused node with the spatial-first input
    gradient in play, which is what the per-basis-function row ranges are for.

The dtype is the one axis selection cannot settle, since fp16/bf16 -- under autocast in
particular -- is only known in ``forward``. The kpacked plan therefore checks it per call
and hands anything else to the arc kernels. That is a branch on a static dtype, so it
stays traceable.

The compiled regular backends run their kernels through a plan
(:mod:`torch_harmonics.disco.optimized.disco_plan`): the kernels together with the names of
the state they read. The autograd nodes are written against the plan, so a backend with
new kernels supplies a plan and its state and inherits the forward, the fused recompute
and both orders of the input gradient.
"""

import os
from typing import TYPE_CHECKING

import torch
from disco_helpers import optimized_kernels_is_available

from torch_harmonics._backend import BackendS2, _kernel_device_types, _reference_state_dtype

from ._disco_utils import _weight_contraction
from ._psi_layouts import build_arcs, build_kpacked, build_split
from .kernels_torch.disco_ragged_torch import _disco_s2_contraction_ragged_torch, _disco_s2_transpose_contraction_ragged_torch
from .kernels_torch.disco_regular_torch import _disco_s2_contraction_regular_torch, _disco_s2_transpose_contraction_regular_torch
from .optimized import disco_arch
from .optimized.disco_optimized import (
    _disco_s2_contraction_planned,
    _disco_s2_conv_optimized,
    _disco_s2_conv_ragged_optimized,
    _kpacked_k_pad,
)
from .optimized.disco_plan import ARC_STATE, KPACKED_STATE, ArcPlanS2, KpackedPlanS2
from .optimized.disco_plan_fft import FftPlanS2, build_fft_state, fft_scatter_available

if optimized_kernels_is_available():
    from .optimized.disco_optimized import (
        _disco_s2_contraction_ragged_optimized,
        _disco_s2_contraction_regular_optimized,
        _disco_s2_transpose_contraction_ragged_optimized,
        _disco_s2_transpose_contraction_regular_optimized,
    )

if TYPE_CHECKING:  # pragma: no cover
    from torch_harmonics.disco.convolution import DiscreteContinuousConv


#: the arc arrays, in the order the operators take them
_ARC_STATE = ARC_STATE

#: the same on a ragged layer, where a row is a point, followed by the ring tables
_RAGGED_ARC_STATE = ("psi_row_ker", "psi_row_pt", "psi_seg_off", "psi_seg", "psi_val_off", "psi_vals", "psi_ring_base", "psi_ring_size")

#: the device types each compiled forward has a kernel for: CPU, and CUDA when built with
#: it. A CUDA layer under a CPU-only build falls through to the reference instead of
#: failing in forward.
_REGULAR_DEVICES = _kernel_device_types("disco_kernels::forward_regular")
_RAGGED_DEVICES = _kernel_device_types("disco_kernels::forward_ragged")


def _device_index(device: torch.device) -> int:
    """The index of a CUDA device, the current one for a bare ``cuda``."""
    return device.index if device.index is not None else torch.cuda.current_device()


def _amp_dtype(x: torch.Tensor) -> torch.dtype:
    """The dtype the contraction will actually run in: the autocast dtype if autocast is on for x's device."""
    device_type = x.device.type
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    return x.dtype


class DiscoBackendS2(BackendS2):
    """
    One way of evaluating the psi contraction of a DISCO layer.

    Besides ``available`` and ``prepare``, a backend provides the three operations the
    layers are built from. Each reads the prepared state back off the layer by name.

    ``contract(layer, x)``
        ``(B, C, H_in, W_in) -> (B, C, K, H_out, W_out)``, differentiable in ``x``; on a
        ragged layer ``(B, C, npoints_in) -> (B, C, K, npoints_out)``.
    ``conv(layer, x, weight, groups, groupsize, recompute)``
        The contraction followed by the weight contraction,
        ``-> (B, groups * out_per_group, H_out, W_out)``. ``recompute`` asks for the
        K-expanded intermediate to be recomputed in backward rather than saved; a
        backend that cannot do that computes the same result the plain way.
    ``transpose(layer, x)``
        The scatter direction, ``(B, C, K, H_in, W_in) -> (B, C, H_out, W_out)``; on a
        ragged layer ``(B, C, K, npoints_in) -> (B, C, npoints_out)``.
    """

    def contract(self, layer: "DiscreteContinuousConv", x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def conv(self, layer: "DiscreteContinuousConv", x: torch.Tensor, weight: torch.Tensor, groups: int, groupsize: int, recompute: bool = False) -> torch.Tensor:
        return _weight_contraction(self.contract(layer, x), weight)

    def transpose(self, layer: "DiscreteContinuousConv", x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RegularReferenceBackend(DiscoBackendS2):
    """
    The torch reference, in terms of a sparse COO psi.

    Always available on a regular layer, so it is that layer's last candidate. It
    contracts by repeatedly rolling the input -- the specification, not a fast path --
    and has no recompute, so ``fused`` changes nothing here but memory.
    """

    name = "regular-reference"

    @classmethod
    def available(cls, layer, device):
        return not layer.ragged

    def prepare(self, layer, device):
        return {"psi": layer._reference_psi(*layer._psi_coo()).to(device=device, dtype=_reference_state_dtype(layer))}

    def contract(self, layer, x):
        return _disco_s2_contraction_regular_torch(x, layer.psi, layer.nlon_out)

    def transpose(self, layer, x):
        return _disco_s2_transpose_contraction_regular_torch(x, layer.psi, layer.nlon_out)


class RegularOptimizedBackend(DiscoBackendS2):
    """
    The compiled gather and scatter kernels, CPU or CUDA, on psi in arc form.

    The operators are registered for both devices, so the dispatcher picks the kernel
    and this one backend serves either. The device test is only whether this build has a
    kernel for the device at all: MPS, XPU, or CUDA under a CPU-only build fall through
    to the reference.

    The contraction and the convolution run through a plan (see
    :mod:`torch_harmonics.disco.optimized.disco_plan`), the kernels together with the names
    of the state they read. This backend's is the arc plan; a backend with other kernels
    builds its own state in ``prepare`` and returns its own plan from :meth:`_make_plan`,
    and inherits ``contract`` and ``conv``. ``transpose`` stays on the arc contraction ops.
    """

    name = "regular-optimized"

    #: the plan class, constructed by :meth:`_make_plan`
    plan_class = ArcPlanS2

    @classmethod
    def available(cls, layer, device):
        return not layer.ragged and layer.optimized_kernel and device.type in _REGULAR_DEVICES

    def prepare(self, layer, device):
        state = self._arc_state(layer, device)
        self.plan = self._make_plan(layer, state)
        return state

    def _make_plan(self, layer, state):
        """The plan over ``state``, the tensors ``prepare`` built, in the order they were built."""
        return self.plan_class(tuple(state), layer.kernel_size, layer._contract_shape, self.split_row_offsets)

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

    def contract(self, layer, x):
        dtype = _amp_dtype(x)
        if not (self.plan.has_gather(dtype) or self.plan.has_scatter(dtype)):
            # no kernels of the plan's own: the arc contraction op, which follows autocast by itself
            return _disco_s2_contraction_regular_optimized(x, *self._arcs(layer), layer.kernel_size, *layer._contract_shape)
        return _disco_s2_contraction_planned(x.to(dtype), self.plan, self.plan.state(layer))

    def conv(self, layer, x, weight, groups, groupsize, recompute=False):
        # the contraction op follows autocast by itself; this path calls the kernels
        # directly, so it applies the cast the same way
        x = x.to(_amp_dtype(x))
        return _disco_s2_conv_optimized(x, weight, self.plan, self.plan.state(layer), groups, groupsize, recompute)

    def transpose(self, layer, x):
        return _disco_s2_transpose_contraction_regular_optimized(x, *self._arcs(layer), layer.kernel_size, *layer._contract_shape)


class RegularKpackedBackend(RegularOptimizedBackend):
    """
    The tensor-core forward (WGMMA on sm_90a, tcgen05 on sm_100a), with the arc scatter as backward.

    Serves fp16/bf16 activations and hands anything else to the arc kernels, which it
    also needs for the backward (see ``KpackedPlanS2``). The layout is built only on a
    device that can run it: it is padded per point to K_pad values, a sizeable buffer to
    hold for a kernel that could never launch.

    Available only for the forward direction, on a device and build the kernels were
    compiled for (see ``disco_arch``), for K <= 16 and ``nlon_out`` a
    multiple of 8. Declines in ``prepare`` when the basis functions do not share one
    support, which the layout requires and which is only known once it is built.
    """

    name = "regular-kpacked"

    plan_class = KpackedPlanS2

    @classmethod
    def available(cls, layer, device):
        return (
            super().available(layer, device)
            and not layer.transpose
            and device.type == "cuda"
            and disco_arch.kpacked_build_available()
            and disco_arch.kpacked_supported_on_device(_device_index(device))
            and _kpacked_k_pad(layer.kernel_size) is not None
            and layer.nlon_out % 8 == 0
        )

    def prepare(self, layer, device):
        state = self._kpacked_state(layer, device)
        if state is None:
            return None
        self.plan = self._kpacked_plan(layer, state, device)
        return state

    def _kpacked_plan(self, layer, state, device):
        """The plan over ``state``, with the kpacked forward of the arch-specific extension for ``device``."""
        plan = self._make_plan(layer, state)
        plan.kpacked_op = disco_arch.kpacked_op(_device_index(device))
        return plan

    def _kpacked_state(self, layer, device):
        """The arc arrays and the kpacked layout, or None if the basis functions do not share one support."""
        kpacked = build_kpacked(*layer._psi_coo(), layer.kernel_size, _kpacked_k_pad(layer.kernel_size), layer._contract_shape[0], layer._psi_nlon)
        if kpacked is None:
            return None

        state = self._arc_state(layer, device)
        for name, t in zip(KPACKED_STATE, kpacked):
            state[name] = t.to(device)
        return state


class RegularFftBackend(RegularKpackedBackend):
    """
    Experimental: the kpacked forward with the frequency-space arc scatter, on H100 and B200 / GB200.

    The scatter is a Kernel Factory candidate (``kernels_cuda/regular/disco_cuda_bwd_fft.cu``), built
    into both arch-specific extensions, and runs for fp16 and bf16 gradients; see ``FftPlanS2``.
    Available where the kpacked backend is, on a compute capability 9.0 or 10.0 device whose
    extension (``torch_harmonics.disco._C_sm90`` / ``_C_sm100``, 9.0a / 10.0a in
    ``TORCH_CUDA_ARCH_LIST``) was built, and when the input longitudes are a multiple of the output
    longitudes, which the scatter requires.

    Not in ``BACKENDS`` yet: layers select it only when ``EXPERIMENTAL_BACKENDS`` are enabled (see there).
    """

    name = "regular-fft"

    plan_class = FftPlanS2

    @classmethod
    def available(cls, layer, device):
        return super().available(layer, device) and fft_scatter_available(device) and layer._psi_nlon % layer.nlon_out == 0

    def prepare(self, layer, device):
        state = self._kpacked_state(layer, device)
        if state is None:
            return None
        state.update(build_fft_state(state, self.split_row_offsets))
        self.plan = self._kpacked_plan(layer, state, device)
        self.plan.fft_plan_op, self.plan.fft_run_op = disco_arch.fft_scatter_ops(_device_index(device))
        return state


class RaggedOptimizedBackend(DiscoBackendS2):
    """
    The compiled ragged gather and scatter, CPU or CUDA, on psi in arc form keyed per point.

    The counterpart of :class:`RegularOptimizedBackend` for a grid without a p-shift. Rows
    are (basis function, point) rather than (basis function, latitude), and the arcs walk
    rings of differing length, so the ring tables of the grid psi's columns index travel
    with them -- as for :class:`~torch_harmonics.attention.backends.RaggedOptimizedBackend`.
    No device test: the CPU and CUDA kernels are registered against the same operators.

    ``conv`` goes through the ragged counterpart of the regular fused node, with the same
    recompute and spatial-first input gradient.
    """

    name = "ragged-optimized"

    @classmethod
    def available(cls, layer, device):
        return layer.ragged and layer.optimized_kernel and device.type in _RAGGED_DEVICES

    def prepare(self, layer, device):
        ring_base, ring_size = layer._psi_col_rings()
        arcs = build_arcs(*layer._psi_coo(), ring_base=ring_base, ring_size=ring_size)
        state = {name: t.to(device) for name, t in zip(_RAGGED_ARC_STATE, (*arcs, ring_base, ring_size))}

        self.split_row_offsets = None
        if layer._needs_split:
            split_ker, self.split_row_offsets = build_split(arcs, layer.kernel_size)
            state["psi_split_ker"] = split_ker.to(device)

        return state

    def _arcs(self, layer):
        return tuple(getattr(layer, name) for name in _RAGGED_ARC_STATE)

    def _split(self, layer):
        if self.split_row_offsets is None:
            return None
        return (layer.psi_split_ker, self.split_row_offsets)

    def contract(self, layer, x):
        return _disco_s2_contraction_ragged_optimized(x, *self._arcs(layer), layer.kernel_size, layer.npoints_out_local)

    def conv(self, layer, x, weight, groups, groupsize, recompute=False):
        # the node calls the kernels directly, so it applies autocast's cast itself, as
        # RegularOptimizedBackend.conv does
        x = x.to(_amp_dtype(x))
        return _disco_s2_conv_ragged_optimized(x, weight, self._arcs(layer), self._split(layer), layer.kernel_size, layer.npoints_out_local, groups, groupsize, recompute)

    def transpose(self, layer, x):
        return _disco_s2_transpose_contraction_ragged_optimized(x, *self._arcs(layer), layer.kernel_size, layer.npoints_out_local)


class RaggedReferenceBackend(DiscoBackendS2):
    """
    The ragged torch reference, in terms of a sparse COO psi keyed per point.

    The fallback for a ragged layer: it asks nothing of the device, so it is what a build
    without the kernels gets, or a layer with ``optimized_kernel=False``. One sparse
    product per basis function; the transpose layer builds its psi transposed, so the
    same product serves it.
    """

    name = "ragged-reference"

    @classmethod
    def available(cls, layer, device):
        return layer.ragged

    def prepare(self, layer, device):
        return {"psi": layer._reference_psi(*layer._psi_coo()).to(device=device, dtype=_reference_state_dtype(layer))}

    def contract(self, layer, x):
        return _disco_s2_contraction_ragged_torch(x, layer.psi)

    def transpose(self, layer, x):
        return _disco_s2_transpose_contraction_ragged_torch(x, layer.psi)


#: Every backend, most specific first; see :mod:`torch_harmonics._backend`. A regular
#: and a ragged backend never both accept a layer, so the two families are ordered only
#: among themselves, as in :data:`torch_harmonics.attention.backends.BACKENDS`. The
#: transpose layers use the same list: the kpacked backend declines them itself.
#: backends with kernels still being validated, most specific first. Layers do not select them unless
#: ``TORCH_HARMONICS_DISCO_EXPERIMENTAL_BACKENDS=1`` is set when torch_harmonics is imported, which puts
#: them ahead of the others in ``BACKENDS``; a layer can also opt in on its own by setting its
#: ``_backends`` to ``EXPERIMENTAL_BACKENDS + BACKENDS`` and calling ``_select_backend()``.
EXPERIMENTAL_BACKENDS = (RegularFftBackend,)

BACKENDS = (
    RaggedOptimizedBackend,
    RaggedReferenceBackend,
    RegularKpackedBackend,
    RegularOptimizedBackend,
    RegularReferenceBackend,
)

if os.getenv("TORCH_HARMONICS_DISCO_EXPERIMENTAL_BACKENDS", "0") == "1":
    BACKENDS = EXPERIMENTAL_BACKENDS + BACKENDS
