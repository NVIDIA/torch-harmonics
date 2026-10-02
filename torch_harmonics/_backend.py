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
            return

        raise RuntimeError(self._no_backend_message(device))

    def _no_backend_message(self, device: torch.device) -> str:
        return f"no {type(self).__name__} backend serves this configuration on {device}"

    def _apply(self, fn, recurse: bool = True):
        """
        Reselect the backend when the module changes device, and restore its state when a
        dtype change has cast it.

        ``_apply`` rather than ``to``: ``.cuda()``, ``.cpu()``, ``.half()`` and
        ``.double()`` never call ``to``, so it is the only hook that sees every move.

        A dtype change casts every floating buffer, backend state included, but that state
        has a fixed dtype -- quadrature weights and filter values are float32 whatever the
        activations are, and the kernels read them as such. So a cast of the state is
        undone by preparing it again, just as a device move is. The test is on what
        actually changed, the device or the state's dtypes, not on the call.

        The state of the outgoing backend is moved or cast by ``super()._apply`` and then
        thrown away -- once per move, against not having to know the target device or
        dtype before anything has been touched.
        """
        device_before = self.device
        dtypes_before = {name: getattr(self, name).dtype for name in self._backend_state}
        out = super()._apply(fn, recurse)
        state_cast = any(getattr(self, name).dtype != dtype for name, dtype in dtypes_before.items())
        if self.device != device_before or state_cast:
            self._select_backend()
        return out
