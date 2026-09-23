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
#
# Workaround for pytorch/pytorch#132388: `torch.amp.custom_fwd` /
# `torch.amp.custom_bwd` assume the legacy autograd.Function API where
# ``forward(ctx, *args)`` — the decorators set ``_fwd_used_autocast`` /
# ``_dtype`` on ``args[0]`` (= ctx) in fwd and read them off ctx in bwd.
#
# Our autograd Functions use the new-style API with a separate
# ``setup_context(ctx, inputs, output)`` staticmethod. In that style ``args[0]``
# in fwd is the first input tensor, so the decorator's state lands on a tensor
# instead of ctx, and the bwd decorator AttributeErrors when it tries to read
# ``ctx._fwd_used_autocast``.
#
# This module provides ``_custom_setup_context`` — the missing decorator the
# PyTorch maintainers (soulitzer in #132388) said they'd accept as the fix:
# it sets the same attributes on ctx at setup_context time, bridging the gap.
#
# Because ``_custom_setup_context`` supplies the ctx state that ``custom_bwd``
# reads, ``custom_fwd`` has nothing left to do on a new-style ``forward``: with
# ``cast_inputs=None`` its whole body is those two assignments, which land on a
# tensor nobody reads.
#
# That dead store is not merely wasteful: ``args[0]._dtype = ...`` is a
# ``setattr`` on a Tensor, which Dynamo cannot trace, so it graph-breaks
# ``torch.compile(fullgraph=True)``. This module therefore also provides
# ``_custom_fwd``, a drop-in replacement that keeps the half of ``custom_fwd``
# that still has a job on the new-style API (``cast_inputs``) and drops the half
# that does not (recording state on a ctx that ``forward`` never receives).
#
# Remove this module and the imports once upstream lands the real
# ``torch.amp.custom_setup_context``.

import collections.abc
import functools

import torch


def _cast(value, device_type: str, dtype: torch.dtype):
    """Recursively cast autocast-eligible tensors in ``value`` to ``dtype``.

    Mirrors ``torch.amp.autocast_mode._cast`` (which is private, hence the
    copy): only floating-point tensors on ``device_type`` are cast, and float64
    is left alone so double-precision code paths are never silently narrowed.
    """
    if isinstance(value, torch.Tensor):
        eligible = value.is_floating_point() and value.device.type == device_type and value.dtype is not torch.float64
        return value.to(dtype) if eligible else value
    elif isinstance(value, (str, bytes)):
        return value
    elif isinstance(value, collections.abc.Mapping):
        return {k: _cast(v, device_type, dtype) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return type(value)(_cast(v, device_type, dtype) for v in value)
    else:
        return value


def _custom_fwd(fwd=None, *, device_type: str, cast_inputs=None):
    """Compile-safe ``torch.amp.custom_fwd`` for new-style autograd Functions.

    ``torch.amp.custom_fwd`` does two things: it records the autocast state on
    ``args[0]``, and -- if ``cast_inputs`` is given -- casts floating-point
    inputs and runs ``forward`` with autocast disabled.

    On the new-style API only the second job is still meaningful. ``forward``
    has no ``ctx`` parameter, so ``args[0]`` is the first input tensor and the
    recorded state is never read; ``_custom_setup_context`` records it on
    the real ``ctx`` instead. Dropping that write is what makes this decorator
    traceable under ``torch.compile(fullgraph=True)``.

    With ``cast_inputs=None`` this returns ``fwd`` unchanged, so the common case
    adds no wrapper and no overhead at all.

    Parameters
    ----------
    device_type : str
        Autocast device type, e.g. ``"cuda"``.
    cast_inputs : torch.dtype, optional
        If given, floating-point inputs are cast to this dtype and ``forward``
        runs with autocast disabled, by default ``None``.

    Notes
    -----
    When ``cast_inputs`` is used, the matching ``_custom_setup_context``
    must be given the same ``cast_inputs`` so that ``backward`` does not
    re-enter an autocast region that ``forward`` never ran in.
    """
    if fwd is None:
        return functools.partial(_custom_fwd, device_type=device_type, cast_inputs=cast_inputs)

    # nothing left to do: the state-recording half is _custom_setup_context's job
    if cast_inputs is None:
        return fwd

    @functools.wraps(fwd)
    def decorate_fwd(*args, **kwargs):
        if torch.is_autocast_enabled(device_type):
            with torch.amp.autocast(device_type=device_type, enabled=False):
                return fwd(*_cast(args, device_type, cast_inputs), **_cast(kwargs, device_type, cast_inputs))
        return fwd(*args, **kwargs)

    return decorate_fwd


def _custom_setup_context(setup_context_fn=None, *, device_type: str, cast_inputs=None):
    """Bridge for new-style autograd.Function + ``_custom_fwd``/custom_bwd.

    Records the current autocast state on ``ctx`` at setup_context time, so
    that the ``custom_bwd`` decorator on ``backward`` can find
    ``ctx._fwd_used_autocast`` / ``ctx._dtype`` and re-enter the same autocast
    context as ``forward`` was in.

    Parameters
    ----------
    device_type : str
        Autocast device type, e.g. ``"cuda"``.
    cast_inputs : torch.dtype, optional
        Must match the ``cast_inputs`` passed to ``_custom_fwd``. When it is
        given, ``forward`` ran with autocast disabled, so ``backward`` must not
        re-enter autocast and ``_fwd_used_autocast`` is recorded as ``False``.

    Usage::

        class MyFn(torch.autograd.Function):
            @staticmethod
            @_custom_fwd(device_type="cuda")
            def forward(x, ...):
                ...

            @staticmethod
            @_custom_setup_context(device_type="cuda")
            def setup_context(ctx, inputs, output):
                ...

            @staticmethod
            @torch.amp.custom_bwd(device_type="cuda")
            def backward(ctx, grad_out):
                ...
    """
    if setup_context_fn is None:
        return functools.partial(_custom_setup_context, device_type=device_type, cast_inputs=cast_inputs)

    @functools.wraps(setup_context_fn)
    def decorate_setup_context(ctx, *args, **kwargs):
        ctx._dtype = torch.get_autocast_dtype(device_type)
        ctx._fwd_used_autocast = torch.is_autocast_enabled(device_type) if cast_inputs is None else False
        return setup_context_fn(ctx, *args, **kwargs)

    return decorate_setup_context


def _cast_to_autocast_dtype(*tensors, device_type: str = "cuda"):
    """Cast floating-point Tensors to the active autocast dtype.

    Mirrors what PyTorch's built-in autocast-eligible ops (``F.linear``,
    ``F.conv2d``, ...) do internally: when autocast is enabled, transient
    copies of inputs are cast to the autocast dtype before the op runs. Use
    this in an orchestrator function immediately before calling ``.apply()``
    on a custom autograd.Function, so ``setup_context`` saves the cast
    tensors and the op runs in the autocast dtype throughout fwd and bwd.

    Non-Tensor arguments and non-floating-point Tensors are passed through
    unchanged. When autocast is not enabled, all arguments are passed
    through unchanged.

    Casting only creates transient copies — parameter storage on the calling
    module is not mutated. Gradients flowing back to fp32 parameters get
    cast to the parameter's dtype by autograd's ``AccumulateGrad`` node, so
    optimizer state stays fp32 even when the op's computation ran in bf16.

    Usage::

        x, weight = _cast_to_autocast_dtype(x, weight)
        out = _MyFn.apply(x, weight, ...)

    Returns a tuple in the same order the arguments were passed.
    """
    if not torch.is_autocast_enabled(device_type):
        return tensors
    cast_dtype = torch.get_autocast_dtype(device_type)
    return tuple(t.to(cast_dtype) if (isinstance(t, torch.Tensor) and t.is_floating_point()) else t for t in tensors)
