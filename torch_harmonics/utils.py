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

import functools
import os
import tempfile
import urllib.request
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

_MOLA_URL = (
    "https://astrogeology.usgs.gov/ckan/dataset/"
    "83c20dbd-e2b3-4e5b-b019-f13d4fdffa38/resource/"
    "57f84b24-d56c-42dd-a34d-cf9d61a82d2c/download/"
    "mars_mgs_mola_dem_mosaic_global_1024.jpg"
)


def load_mola_elevation(
    nlat: Optional[int] = None,
    nlon: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> torch.Tensor:
    """
    Download the NASA MOLA Mars digital elevation map and return it as a tensor.

    The image is downloaded once and cached locally for subsequent calls.

    Parameters
    ----------
    nlat : int, optional
        Target number of latitude points. If given together with *nlon*,
        the image is bilinearly interpolated to (nlat, nlon).
    nlon : int, optional
        Target number of longitude points.
    cache_dir : str, optional
        Directory for the cached download. Defaults to the system temp dir.

    Returns
    -------
    torch.Tensor
        Grayscale elevation map with shape (nlat, nlon), values in [0, 1].
    """
    from PIL import Image

    if cache_dir is None:
        cache_dir = tempfile.gettempdir()
    path = os.path.join(cache_dir, "mola_topo.jpg")

    if not os.path.exists(path):
        req = urllib.request.Request(_MOLA_URL, headers={"User-Agent": "torch-harmonics"})
        with urllib.request.urlopen(req) as resp, open(path, "wb") as f:
            f.write(resp.read())

    img = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    data = torch.from_numpy(img)

    if nlat is not None and nlon is not None:
        data = (
            F.interpolate(
                data.unsqueeze(0).unsqueeze(0),
                size=(nlat, nlon),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

    return data


def _contiguous(x: torch.Tensor) -> torch.Tensor:
    """
    ``x.contiguous()``, performed in real space for complex inputs.

    Inductor cannot generate a Triton kernel that reads or writes a complex buffer: complex
    dtypes have no Triton type, so the copy that ``contiguous()`` lowers to dies in codegen
    with ``KeyError: 'complex64'``. ``view_as_real``/``view_as_complex`` are pure views (the
    former is the one consumer for which inductor's complex-tensor check makes an exception),
    so routing the copy through them yields a bit-identical result from a real-dtype kernel.

    Nothing is written in place, so this does not reintroduce the autograd breakage that the
    old ``xout[..., 0] = ...`` / ``view_as_complex`` pattern caused.
    """

    if x.is_complex():
        return torch.view_as_complex(torch.view_as_real(x).contiguous())
    return x.contiguous()


class _EnsureContiguous(torch.autograd.Function):
    """Ensures the tensor is contiguous in both the forward and backward pass."""

    @staticmethod
    def forward(x):
        return _contiguous(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad):
        # load-bearing: an upstream layer can hand back a non-contiguous gradient, and the
        # CPU/MKL FFT backward rejects the resulting stride pattern (DftiCommitDescriptor).
        return _contiguous(grad)


def ensure_contiguous(x: torch.Tensor) -> torch.Tensor:
    """Ensures the tensor is contiguous in both the forward and backward pass."""
    return _EnsureContiguous.apply(x)


# How torch.compile words its refusal on an unsupported Python, in torch 2.8 to 2.10 at least.
_TORCH_COMPILE_REFUSAL = "torch.compile is not supported on Python"


@functools.lru_cache(maxsize=None)
def torch_compile_supported() -> bool:
    """
    Whether ``torch.compile`` can be used at all with this PyTorch and Python.

    False where torch refuses it outright -- torch 2.9 on Python 3.14, torch 2.10 on 3.15, a
    free-threaded build -- which it does with a RuntimeError starting with the prefix below in
    every supported version. Any other error is not a refusal and is raised, so a broken setup
    surfaces instead of silently running uncompiled. Wrapping a function does not compile
    anything yet, so this is cheap.
    """
    try:
        torch.compile(lambda x: x)
    except RuntimeError as e:
        if str(e).startswith(_TORCH_COMPILE_REFUSAL):
            return False
        raise
    return True


def compile_if_supported(fn):
    """
    ``torch.compile(fn)``, or ``fn`` unchanged where this torch cannot compile at all.

    ``torch.compile`` refuses outright when the running Python is newer than the torch
    release supports -- torch 2.9 on Python 3.14 raises ``torch.compile is not supported
    on Python 3.14+`` -- and it does so when called, not when the compiled function first
    runs. Used as a decorator in a class body, that made ``import torch_harmonics`` fail.
    Falling back to the plain function keeps the code working, eagerly.
    """
    return torch.compile(fn) if torch_compile_supported() else fn


#: device types that cannot hold a float64 tensor at all
_NO_FLOAT64_DEVICE_TYPES = frozenset({"mps"})


def recast_buffers_for(module: torch.nn.Module, fn) -> None:
    """
    Downcast ``module``'s own float64 buffers to float32 if ``module._apply(fn)`` is about to move them to a device without float64.

    The convention this supports: a layer builds the weights it derives from a grid
    (quadrature weights, Legendre polynomials, ...) in float64, keeps them as float64
    buffers, and casts them at use to ``torch.promote_types(x.dtype, torch.float32)``. A
    float64 input then computes with float64 weights, and every other dtype exactly as
    with float32 ones. MPS has no float64, and moving such a buffer there fails inside the
    move, before any forward could cast it. So a module with float64 buffers calls this
    first thing in its ``_apply``::

        def _apply(self, fn, recurse=True):
            recast_buffers_for(self, fn)
            return super()._apply(fn, recurse)

    On a move to MPS its float64 buffers become float32; on any other conversion nothing
    changes, and they keep their precision.

    **Why** ``_apply`` **and not** ``to``. Every conversion -- ``.to()``, ``.cuda()``,
    ``.cpu()``, ``.half()``, ``.double()``, ``.to_empty()`` -- ends in ``_apply``, and
    ``_apply`` is what recurses into the children. ``Module.to`` does not call the
    children's ``to``: a ``to`` override on a layer would never run when the layer sits
    inside a model and the model is moved, which is how layers are normally used.

    **How the destination is found.** ``_apply`` is only given ``fn``, not where it leads;
    PyTorch has no public way to ask. So ``fn`` is applied to an empty probe, and the
    probe's resulting device is the destination. The probe is float32 on CPU explicitly,
    whatever the default dtype and device: a float64 probe would itself fail to move to
    MPS, and a probe created on a default MPS device would make a dtype-only conversion
    such as ``.double()`` fail on it. From CPU, a move lands on its target and a dtype-only
    conversion stays on CPU, which is all the check needs to tell apart.

    Only ``module``'s own buffers are touched; ``_apply`` recurses through each child's
    ``_apply``, so a child with float64 buffers calls this itself. It runs on module
    conversions only, never in ``forward``, so ``torch.compile`` does not see it.

    Not covered: state a module creates directly on the device after the move, such as
    what a backend's ``prepare`` builds; that has to choose its dtype itself.

    Parameters
    ----------
    module : torch.nn.Module
        The module whose own buffers are about to be converted.
    fn : callable
        The conversion ``module._apply`` was called with.
    """
    # the probe's dtype and device are fixed rather than the defaults: a float64 default would fail
    # to move to MPS just as the buffers would, and an MPS default device would make a dtype-only
    # conversion such as .double() fail on the probe. From CPU, a move lands on its target and a
    # dtype-only conversion stays put, which is all this needs to tell apart
    if fn(torch.empty(0, dtype=torch.float32, device="cpu")).device.type not in _NO_FLOAT64_DEVICE_TYPES:
        return
    for name, buf in module._buffers.items():
        if buf is not None and buf.dtype == torch.float64:
            module._buffers[name] = buf.to(torch.float32)


def check(cond: bool, message) -> None:
    """
    ``torch._check`` with a deferred message, without blocking full-graph compilation.

    ``torch._check(cond, lambda: ...)`` is the documented way to keep an error
    message off the hot path, but it cannot be traced: Dynamo has no ``as_proxy()``
    for a closure, so a callable message fails ``torch.compile(fullgraph=True)`` with
    *Failed to convert args/kwargs to proxy*. That is true of any callable, including
    one returning a constant string, so keeping symbolic values out of the message is
    not enough to make it traceable.

    Eager execution keeps the deferred message. While tracing, the condition is
    checked without one: the check still fires, it just reports less.

    Parameters
    ----------
    cond : bool
        Condition to assert. May be a symbolic bool under dynamic shapes.
    message : callable
        Zero-argument callable returning the message, evaluated only on failure and
        only outside tracing.
    """

    if torch.compiler.is_compiling():
        torch._check(cond)
    else:
        torch._check(cond, message)
