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
import warnings
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


# How torch.compile words its refusal on an unsupported Python, in torch 2.6 to 2.10 at least.
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


# Before PyTorch 2.9, inductor mispredicts the strides of an intermediate in the scalar SHTs
# (and so in SpectralConvS2): its generated code asserts contiguous strides where eager
# produces a permuted einsum output, and fails with "AssertionError: expected size ...,
# stride ... at dim=..." from assert_size_stride. It is an inductor bug, fixed in 2.9, which
# no layout change on our side has reliably avoided -- so torch.compile of these layers needs
# PyTorch 2.9, and on older versions they warn when they are compiled.
TORCH_COMPILE_INDUCTOR_STRIDE_BUG = torch.__version__ < "2.9"

_compile_warning_issued = False


def _warn_compile_inductor_stride_bug() -> bool:
    global _compile_warning_issued
    if not _compile_warning_issued:
        _compile_warning_issued = True
        warnings.warn(
            f"torch.compile of the torch-harmonics SHT and spectral convolution layers needs PyTorch 2.9 or "
            f"newer; with PyTorch {torch.__version__}, an inductor bug makes the compiled code fail with "
            f"'AssertionError: expected size ..., stride ...' (assert_size_stride). Upgrade PyTorch, or run "
            f"these layers uncompiled.",
            UserWarning,
            stacklevel=3,
        )
    return True


# Warn, once, that torch.compile of the SHT and spectral layers needs PyTorch 2.9. Called from
# their forward under `TORCH_COMPILE_INDUCTOR_STRIDE_BUG and torch.compiler.is_compiling()`.
# Where the bug does not exist that condition is a constant False which dynamo folds away, so
# compilation is untouched, fullgraph=True included. Where it does, the function is marked
# assume_constant_result: dynamo runs it in Python while tracing and takes its result as a
# constant, so the warning appears without a graph break -- and before the compiled graph
# runs into the bug, which a warning deferred until after the graph would never reach.
warn_compile_inductor_stride_bug = (
    torch._dynamo.assume_constant_result(_warn_compile_inductor_stride_bug) if TORCH_COMPILE_INDUCTOR_STRIDE_BUG else _warn_compile_inductor_stride_bug
)


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
