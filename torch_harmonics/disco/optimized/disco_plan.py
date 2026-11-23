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
Plans: the kernels a regular DISCO backend runs, together with the psi-derived state they read.

The autograd nodes of :mod:`.disco_optimized` -- the contraction, and the contraction
fused with the weight contraction -- are written against a plan rather than against
particular kernels. They need three raw operations from it:

``gather(x, t, tc)``
    The forward contraction, ``(B, C, H_in, W_in) -> (B, C, K, H_out, W_out)``, in the dtype
    of ``x``. Runs in the forward and in the recompute of ``fused=True``.
``scatter(g, t, tc, nlat_in, nlon_in)``
    The arc scatter, ``(B, C, K, H_out, W_out) -> (B, C, H_in, W_in)``: the input gradient
    of the contraction, and of the convolution taken weight-first.
``scatter_split(g, t, tc, nlat_in, nlon_in)``
    The scatter of each basis function on its own, ``(B, C, 1, H_out, W_out) ->
    (B, C, K, H_in, W_in)``: the input gradient taken spatial-first. Only when the plan holds
    the per-basis-function row ranges (``has_split``).

and the arc arrays (``arcs(t)``), from which the differentiable contraction ops are built:
the backward under ``create_graph=True``, and the dtypes a plan's own kernels do not take.

The state comes in two levels.

``t``, the static state
    What depends on psi alone. A plan holds no tensors itself: it names them
    (``state_names``), its backend's ``prepare`` builds them, and the layer registers them as
    buffers, so they move with the layer and go when another backend is selected.
    ``state(layer)`` reads them off the layer in the order of ``state_names``.
``tc``, the call state
    What also depends on the call: the launch configuration a kernel picks from the
    number of planes it contracts (B * C) and from the dtype, and whatever it derives from
    psi under that configuration (tile schedules, psi laid out for those tiles). Built by
    :meth:`DiscoPlanS2.call_state` for a key ``(op, bc, dtype, in_shape)`` and cached on the
    plan, so it is built once per batch size, not once per call. A plan without it (the default,
    ``has_call_state = False``) costs nothing here.

The backend binds the call state before the autograd node (:meth:`DiscoPlanS2.bind`), for
exactly the operations the node will run, since the operations see different planes: the
gather and the weight-first scatter B * C_in, the spatial-first scatter B * C_out. The node
then receives ``t`` and ``tc`` as tensor arguments and passes each operation its part. That
keeps the node traceable as a whole under ``torch.compile`` -- the plan is configuration,
and Dynamo inlines its methods -- and the building, a side effect, outside it.

Call state is built outside CUDA graph capture: a call that would build it during capture
raises, so run the layer once eagerly at the shapes to be captured first (the usual warmup
before ``torch.cuda.graph``). All of a plan's tensors are allocated through torch, so its
memory is accounted for, and a kernel takes per-call scratch from torch's allocator too.

New kernels are a subclass: name the static tensors they read and build them in the
backend's ``prepare``; build the call state in ``call_state``; override the operations they
implement and claim them for the dtypes they take (``has_gather``, ``has_scatter``).
Anything they do not implement stays on the arc kernels.
"""

from collections import OrderedDict
from typing import Dict, Optional, Sequence, Tuple

import torch

from .. import disco_kernels
from .._disco_utils import _compute_dtype

#: the arc arrays, in the order the operators take them
ARC_STATE = ("psi_row_ker", "psi_row_lat", "psi_seg_off", "psi_seg", "psi_val_off", "psi_vals")

#: the kpacked layout, in the order the tensor-core forward takes it
KPACKED_STATE = ("psi_kpacked_idx", "psi_kpacked_vals", "psi_kpacked_offset")

#: the operations a plan runs, the keys of its call state
PLAN_OPS = ("gather", "scatter", "scatter_split")

_HALF_DTYPES = (torch.float16, torch.bfloat16)

Tensors = Tuple[torch.Tensor, ...]

#: where each operation's call state sits in the flat ``tc`` of a node: ``((op, start, stop), ...)``
CallLayout = Tuple[Tuple[str, int, int], ...]


def call_slice(layout: CallLayout, tc: Tensors, op: str) -> Tensors:
    """``op``'s part of the flat call state ``tc`` (empty if it was not bound)."""
    for name, start, stop in layout:
        if name == op:
            return tc[start:stop]
    return ()


class DiscoPlanS2:
    """
    The kernels of a regular DISCO backend and the names of the state they read.

    Parameters
    ----------
    state_names : Sequence[str]
        The layer buffers the kernels read, in the order they are passed as ``t``. Must
        include the arc arrays (``ARC_STATE``), and ``psi_split_ker`` when
        ``split_row_offsets`` is given.
    kernel_size : int
        Number of basis functions K.
    contract_shape : Tuple[int, int]
        ``(nlat_out, nlon_out)`` of the contraction's result.
    split_row_offsets : Optional[Tuple[int, ...]]
        Basis function k's rows are ``split_row_offsets[k]:split_row_offsets[k + 1]`` (see
        ``build_split``); None without the spatial-first input gradient.
    """

    #: whether the kernels read call state (see :meth:`call_state`); without it ``bind`` is free
    has_call_state = False

    #: call states kept per plan, least recently used dropped first (except those a captured CUDA
    #: graph uses): a training run needs one per operation, varying batch sizes one per size
    max_call_states = 16

    def __init__(self, state_names: Sequence[str], kernel_size: int, contract_shape: Tuple[int, int], split_row_offsets: Optional[Tuple[int, ...]] = None):
        self.state_names = tuple(state_names)
        self.kernel_size = kernel_size
        self.nlat_out, self.nlon_out = contract_shape
        self.split_row_offsets = split_row_offsets
        self._index: Dict[str, int] = {name: i for i, name in enumerate(self.state_names)}
        self._call_states: "OrderedDict[Tuple[str, int, torch.dtype, Tuple[int, int]], Tensors]" = OrderedDict()
        # key -> (event recorded after the build, the stream that built it), for state on a CUDA device
        self._call_ready: Dict[Tuple[str, int, torch.dtype, Tuple[int, int]], Tuple["torch.cuda.Event", "torch.cuda.Stream"]] = {}
        # keys a captured CUDA graph uses: never evicted, the graph replays their memory
        self._pinned: set = set()
        missing = [name for name in ARC_STATE + (("psi_split_ker",) if self.has_split else ()) if name not in self._index]
        if missing:
            raise ValueError(f"{type(self).__name__} needs the state {missing}")

    @property
    def has_split(self) -> bool:
        """Whether the plan can take the input gradient spatial-first."""
        return self.split_row_offsets is not None

    def state(self, layer: torch.nn.Module) -> Tensors:
        """The plan's static tensors, read off the layer, in the order of ``state_names``."""
        return tuple(getattr(layer, name) for name in self.state_names)

    def _get(self, t: Tensors, name: str) -> torch.Tensor:
        return t[self._index[name]]

    def arcs(self, t: Tensors) -> Tensors:
        """The six arc arrays, ``(row_ker, row_lat, seg_off, seg, val_off, vals)``."""
        return tuple(t[self._index[name]] for name in ARC_STATE)

    # ------------------------------------------------------------------ call state

    def call_state(self, op: str, bc: int, dtype: torch.dtype, in_shape: Tuple[int, int], t: Tensors) -> Tensors:
        """
        The call state of ``op`` (one of ``PLAN_OPS``) over ``bc`` planes of ``dtype``.

        ``in_shape`` is ``(nlat, nlon)`` of the contraction's input -- the gather's input, the
        scatters' output: on a distributed layer the rank's block as it is contracted, which
        the layer's own attributes need not describe. Called once per key, outside the
        autograd nodes and outside CUDA graph capture, with the static state ``t``; the result
        is cached and passed back to ``op`` as ``tc``. Allocate through torch. For
        ``scatter_split`` the planes are those of the output gradient, B * C_out, and the
        state covers every basis function. Only consulted when ``has_call_state``.
        """
        return ()

    def bind(self, ops: Sequence[Tuple[str, int]], dtype: torch.dtype, in_shape: Tuple[int, int], t: Tensors) -> Tuple[CallLayout, Tensors]:
        """
        The call state of the operations ``ops`` -- ``(op, bc)`` pairs -- for activations of ``dtype`` on an input of ``in_shape``.

        Returns the layout and the flat tuple of tensors a node takes (see ``call_slice``).
        Builds what is not cached yet; raises if that would happen during CUDA graph capture.

        Only eager calls fill the cache. A compiled call that misses builds the state for itself
        and leaves the cache alone: its tensors would carry the mode it runs in -- inference
        tensors under ``torch.inference_mode()``, which a later training call could not save
        for backward -- and Dynamo cannot query that mode. Once an eager call has built a key,
        compiled calls use it, so run the layer once eagerly per shape, as before CUDA graph
        capture.

        Cached state may be used on another stream than the one that built it: such a call
        waits for the build and marks the tensors in use on its stream. A key used during CUDA
        graph capture is pinned in the cache, since the graph replays its memory; the build
        is complete by then, as ``torch.cuda.graph`` synchronizes the device when capture
        begins. Compiled calls do neither, as Dynamo traces no stream handling: share a
        compiled layer across streams only after an eager call on each.
        """
        compiling = torch.compiler.is_compiling()
        if not self.has_call_state:
            return tuple((op, 0, 0) for op, _ in ops), ()

        in_shape = (int(in_shape[0]), int(in_shape[1]))
        layout, tc = [], []
        for op, bc in ops:
            if op not in PLAN_OPS:
                raise ValueError(f"unknown plan operation {op!r}; known: {PLAN_OPS}")
            key = (op, int(bc), dtype, in_shape)
            state = self._call_states.get(key)
            if state is None and compiling:
                # built for this call only, see above
                with torch.no_grad():
                    state = tuple(self.call_state(op, int(bc), dtype, in_shape, t))
            elif state is None:
                if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        f"{type(self).__name__} would build the call state of {op} for {bc} planes of {dtype} on {in_shape} during CUDA graph capture; "
                        "run the layer once eagerly with the shapes to be captured first"
                    )
                state = self._build_call_state(op, int(bc), dtype, in_shape, t)
                self._call_states[key] = state
                self._record_built(key, state)
                self._evict()
            elif not compiling:
                self._call_states.move_to_end(key)
                self._ready_on_current_stream(key, state)
            layout.append((op, len(tc), len(tc) + len(state)))
            tc.extend(state)
        return tuple(layout), tuple(tc)

    def _record_built(self, key, state: Tensors) -> None:
        """Remember when the build of ``state`` completes on the stream that built it."""
        device = next((x.device for x in state if x.is_cuda), None)
        if device is None:
            return
        stream = torch.cuda.current_stream(device)
        event = torch.cuda.Event()
        event.record(stream)
        self._call_ready[key] = (event, stream)

    def _ready_on_current_stream(self, key, state: Tensors) -> None:
        """Make cached ``state`` safe to use on the current stream (see ``bind``)."""
        ready = self._call_ready.get(key)
        if ready is None:
            return
        event, stream = ready
        if torch.cuda.is_current_stream_capturing():
            self._pinned.add(key)
            return
        current = torch.cuda.current_stream(stream.device)
        if current != stream:
            current.wait_event(event)
            for x in state:
                if x.is_cuda:
                    x.record_stream(current)

    def _evict(self) -> None:
        """Drop the least recently used unpinned call states beyond ``max_call_states``."""
        for key in [k for k in self._call_states if k not in self._pinned][: max(0, len(self._call_states) - self.max_call_states)]:
            del self._call_states[key]
            self._call_ready.pop(key, None)

    def _build_call_state(self, op: str, bc: int, dtype: torch.dtype, in_shape: Tuple[int, int], t: Tensors) -> Tensors:
        """
        ``call_state``, built in an eager call so that every later call can use it.

        The state is cached across calls, so it must not depend on the mode of the call that
        happens to build it: without autograd (it never needs gradients), and outside
        inference mode, whose tensors a later training call could not save for backward.
        """
        with torch.no_grad():
            if torch.is_inference_mode_enabled():
                with torch.inference_mode(False):
                    return tuple(self.call_state(op, bc, dtype, in_shape, t))
            return tuple(self.call_state(op, bc, dtype, in_shape, t))

    # ------------------------------------------------------------------ operations

    def has_gather(self, dtype: torch.dtype) -> bool:
        """
        Whether ``gather`` runs kernels of the plan's own for activations of ``dtype``.

        If not, the contraction goes through the arc contraction op, which has autograd and
        autocast of its own. The base plan's gather is that op's kernel, so it never needs to.
        """
        return False

    def has_scatter(self, dtype: torch.dtype) -> bool:
        """
        Whether ``scatter`` runs kernels of the plan's own for gradients of ``dtype``.

        Together with ``has_gather``, decides whether the contraction (``contract``, which
        the unfused distributed convolution runs) goes through the plan or through the arc
        contraction op. The convolution node runs the plan either way.
        """
        return False

    def gather(self, x: torch.Tensor, t: Tensors, tc: Tensors = ()) -> torch.Tensor:
        """The forward contraction of the contiguous ``x``, in its dtype."""
        row_ker, row_lat, seg_off, seg, val_off, vals = self.arcs(t)
        itype = x.dtype
        out = disco_kernels.forward_regular.default(x, row_ker, row_lat, seg_off, seg, val_off, vals.to(_compute_dtype(itype)), self.kernel_size, self.nlat_out, self.nlon_out)
        return out.to(itype)

    def scatter(self, g: torch.Tensor, t: Tensors, tc: Tensors, nlat_in: int, nlon_in: int) -> torch.Tensor:
        """The arc scatter of the contiguous ``g``, in its dtype or the compute dtype of it."""
        row_ker, row_lat, seg_off, seg, val_off, vals = self.arcs(t)
        return disco_kernels.backward_regular.default(g, row_ker, row_lat, seg_off, seg, val_off, vals.to(_compute_dtype(g.dtype)), self.kernel_size, nlat_in, nlon_in)

    def scatter_split(self, g: torch.Tensor, t: Tensors, tc: Tensors, nlat_in: int, nlon_in: int) -> torch.Tensor:
        """
        Every basis function's scatter of the contiguous ``(B, C, 1, H_out, W_out)`` ``g``, stacked to ``(B, C, K, H_in, W_in)``.

        In the dtype of ``g`` or its compute dtype. Here one K = 1 scatter per basis function:
        its arc and value offsets are absolute, so each call reads slices of the same arrays
        and needs only ``split_ker``, zeros, in place of ``row_ker``. See ``build_split``.
        Kernels that need each basis function's arcs as arrays of their own build them as
        static state.
        """
        _, row_lat, seg_off, seg, val_off, vals = self.arcs(t)
        split_ker = self._get(t, "psi_split_ker")
        vals = vals.to(_compute_dtype(g.dtype))
        offsets = self.split_row_offsets

        parts = []
        for k in range(self.kernel_size):
            r0, r1 = offsets[k], offsets[k + 1]
            if r1 == r0:
                parts.append(g.new_zeros((g.shape[0], g.shape[1], nlat_in, nlon_in), dtype=vals.dtype))
                continue
            part = disco_kernels.backward_regular.default(g, split_ker[: r1 - r0], row_lat[r0:r1], seg_off[r0 : r1 + 1], seg, val_off[r0 : r1 + 1], vals, 1, nlat_in, nlon_in)
            parts.append(part)
        return torch.stack(parts, dim=2)


class ArcPlanS2(DiscoPlanS2):
    """The compiled gather and scatter on psi in arc form, CPU or CUDA: the base plan as it is."""


class KpackedPlanS2(ArcPlanS2):
    """
    The tensor-core forward (WGMMA on sm_90a, tcgen05 on sm_100a / sm_103a) for fp16/bf16, on the kpacked layout.

    Everything else is the arc plan's: the scatter, and the gather of fp32/fp64. The forward is the op of the
    arch-specific extension for the plan's device, ``kpacked_op``, which the backend sets (see ``disco_arch``).
    """

    #: the kpacked forward for the plan's device, e.g. ``torch.ops.disco_kernels.forward_kpacked_sm90.default``
    kpacked_op = None

    def __init__(self, state_names, kernel_size, contract_shape, split_row_offsets=None):
        super().__init__(state_names, kernel_size, contract_shape, split_row_offsets)
        missing = [name for name in KPACKED_STATE if name not in self._index]
        if missing:
            raise ValueError(f"{type(self).__name__} needs the state {missing}")

    def has_gather(self, dtype):
        return dtype in _HALF_DTYPES

    def gather(self, x, t, tc=()):
        if x.dtype not in _HALF_DTYPES:
            return super().gather(x, t, tc)
        pack_idx, pack_val, pack_offset = (self._get(t, name) for name in KPACKED_STATE)
        return self.kpacked_op(x, pack_idx, pack_val, pack_offset, self.kernel_size, self.nlat_out, self.nlon_out)
