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
Backend selection shared by the layers that have more than one implementation.

A backend is an implementation together with the state it needs. Which one can run
depends on the device, and the device is not known when a layer is built --
constructing on CPU and moving with ``.to()`` is the normal thing to do. So selection
is driven by :meth:`~torch.nn.Module._apply`, which every one of ``.to()``,
``.cuda()``, ``.cpu()``, ``.float()`` funnels through. On a device change the layer
reselects and the new backend prepares its state there. Nothing about this is lazy and
nothing is decided in ``forward``, which is what keeps the forward pass traceable: the
backend is a plain Python attribute, fixed before tracing begins.

Backend state is always ``persistent=False``. Which backend is live is a property of
where the module happens to be, never of what was trained, so it must not reach a
checkpoint.

The layer lists its candidates in ``_backends``, most specific first. Order *is* the
decision procedure: the first whose ``available`` accepts the layer and device, and
whose ``prepare`` does not decline, wins. A backend narrows the case by returning
False rather than by sitting at a particular depth of a tree, so a new axis -- a
dtype, an SM version, a torch version -- is a new predicate, not a reshaped tree.
"""

from typing import Dict, Optional

import torch


def _kernel_device_types(op_name: str) -> frozenset:
    """
    The device types the compiled operator ``op_name`` has a kernel for.

    Being built is not the same as being usable on a device: the extension registers CPU
    and, when compiled with CUDA, CUDA kernels, and nothing for MPS or XPU. Callers test a
    tensor's ``device.type`` against this set before calling the operator and fall back to
    torch otherwise. It is computed once, at import, so the test is a set membership on a
    constant that dynamo traces without a graph break.
    """
    try:
        return frozenset(t for t, key in (("cpu", "CPU"), ("cuda", "CUDA")) if torch._C._dispatch_has_kernel_for_dispatch_key(op_name, key))
    except RuntimeError:
        # the extension is not built, so the operator does not exist
        return frozenset()


class BackendS2:
    """
    One way of evaluating a layer, and the state it needs to do it.

    Subclasses implement:

    ``available(layer, device)``
        Whether this backend can serve that layer on that device. Cheap: it is asked of
        every candidate on every device change.

    ``prepare(layer, device)``
        The tensors this backend reads, on that device, as a name -> tensor dict. The
        layer registers them as non-persistent buffers and removes them again when
        another backend is selected, so a backend gets exactly its own state and never
        sees another's. Returning ``None`` declines: for a condition that is only known
        once the state has been built, and too expensive to establish in ``available``.
        Python-side metadata (offset tables, padding) lives on the backend instance,
        which is created afresh for each selection.

    and whatever evaluation methods the layer calls. They read the prepared state back
    off the layer by name.
    """

    name = "?"

    #: a pure-torch reference rather than compiled kernels: a layer can warn when it lands
    #: here although the kernels were asked for (see ``_on_backend_selected``)
    reference = False

    @classmethod
    def available(cls, layer: torch.nn.Module, device: torch.device) -> bool:
        raise NotImplementedError

    def prepare(self, layer: torch.nn.Module, device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class BackendSelectionMixin:
    """
    Select a backend on construction and on every device or dtype change.

    Mix in ahead of :class:`torch.nn.Module`. The layer provides ``_backends``, the
    candidate classes in order, and a ``device`` property that follows every move -- a
    parameter's device, since buffers belong to backends and come and go with them. It
    calls :meth:`_select_backend` as the last step of ``__init__``, once everything a
    backend's ``prepare`` reads exists.
    """

    #: candidate backend classes, most specific first; the last must accept anything
    #: the layer can be built with, or selection raises
    _backends = ()

    #: names of the buffers the live backend registered
    _backend_state = ()

    #: the live backend
    backend = None

    def _select_backend(self) -> None:
        """
        Pick the backend for the current device and register exactly its state.

        The previous backend's buffers are removed first, so the module carries one
        backend's tensors and never a union of them -- not even transiently, which at
        high resolution would be a second copy of the largest thing the layer holds.
        """
        device = self.device

        for name in self._backend_state:
            delattr(self, name)
        self._backend_state = ()
        self.backend = None

        for cls in self._backends:
            if not cls.available(self, device):
                continue
            backend = cls()
            state = backend.prepare(self, device)
            if state is None:
                continue
            for name, tensor in state.items():
                self.register_buffer(name, tensor, persistent=False)
            self._backend_state = tuple(state)
            self.backend = backend
            self._on_backend_selected(backend, device)
            return

        raise RuntimeError(self._no_backend_message(device))

    def _no_backend_message(self, device: torch.device) -> str:
        return f"no {type(self).__name__} backend serves this configuration on {device}"

    def _on_backend_selected(self, backend: BackendS2, device: torch.device) -> None:
        """Called after every selection, with the new backend live; e.g. to warn about a fallback."""

    def _apply(self, fn, recurse: bool = True):
        """
        Reselect the backend when the module changes device, and restore its state when a
        dtype change has cast it.

        ``_apply`` rather than ``to``: ``.cuda()``, ``.cpu()``, ``.half()`` and
        ``.double()`` never call ``to``, so it is the only hook that sees every move.

        A dtype change casts every floating buffer, backend state included, but that state
        has a dtype its backend fixes -- quadrature weights and filter values are float32
        for the kernels, which read them as such, and for a reference float32 too unless
        the layer is float64. A dtype change can also change the backend, since a float64
        layer may take the reference. So a dtype change is answered by selecting and
        preparing again, just as a device move is -- which is also what keeps ``.half()``
        from handing the kernels a 16-bit buffer they would read as ``float``. The test is
        on what actually changed, the device, the layer's dtype (for a layer with a
        ``dtype`` property) or the state's dtypes, not on the call.

        The state of the outgoing backend is moved or cast by ``super()._apply`` and then
        thrown away -- once per move, against not having to know the target device or
        dtype before anything has been touched.
        """
        device_before, dtype_before = self.device, getattr(self, "dtype", None)
        dtypes_before = {name: getattr(self, name).dtype for name in self._backend_state}
        out = super()._apply(fn, recurse)
        state_cast = any(getattr(self, name).dtype != dtype for name, dtype in dtypes_before.items())
        # the layer's own dtype can decide the backend too (float64 may take the reference),
        # and can change without casting the state: a reference backend's state may already
        # be float32 when .float() brings the layer back from float64
        if self.device != device_before or getattr(self, "dtype", None) != dtype_before or state_cast:
            self._select_backend()
        return out
