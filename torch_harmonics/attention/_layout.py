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

"""
Layout conversion for the attention stack.

Every attention kernel -- CUDA, CPU and the pure-torch reference -- operates on
physical NHWC (channel-innermost) data. This module is the only place that
converts between NCHW and NHWC, so layout is always something the caller decides
and states, never something a kernel infers from strides.

The reason for the "never infer" rule: stride inspection cannot tell the two
layouts apart when a dimension is degenerate. A contiguous NCHW tensor has
``stride(1) == H*W``, so for ``H*W == 1`` it has ``stride(1) == 1`` and looks
exactly like NHWC; conversely ``x.is_contiguous(memory_format=torch.channels_last)``
returns True for any C == 1 tensor. Both layouts hold identical bytes in those
cases, so the ambiguity is harmless for the *input* of a kernel -- but not for
its output, where guessing wrong silently changes the returned shape.

On CUDA the conversion dispatches to the hand-written tiled transpose
(``permute_4D_to0231`` / ``permute_4D_to0312``), which is substantially faster
than ATen's generic strided copy for the channel-innermost case. fp32, fp16 and
bf16 all go through the same kernel -- it is templated on the element type and
only moves elements, so there is no dtype-specific arithmetic to specialize.

The two directions are exact inverses, which makes the autograd rule for each
simply the other direction. That matters: it means the layout conversion in the
*backward* pass uses the fast kernel too, rather than falling back to an ATen
copy.

When the optimized kernels are not built, everything falls back to
``permute(...).contiguous()`` behind the same interface, so the pure-torch
reference path keeps working with no build dependency.
"""

from typing import Tuple

import torch
from attention_helpers import optimized_kernels_is_available

from torch_harmonics.utils import check

__all__ = ["to_nhwc", "to_nchw"]

_OPTIMIZED = optimized_kernels_is_available()


if _OPTIMIZED:

    @torch.library.register_fake("attention_kernels::permute_to_nhwc")
    def _(x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        return x.new_empty((B, H, W, C))

    @torch.library.register_fake("attention_kernels::permute_to_nchw")
    def _(x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        return x.new_empty((B, C, H, W))

    # Autograd: the two ops are mutual inverses, so each one's backward is the
    # other applied to the incoming gradient. grad_output is forced contiguous
    # because the ops require a packed input; autograd does not guarantee it.
    #
    # Nothing is saved for backward: grad_output already has the output shape,
    # and applying the inverse conversion reproduces the input shape exactly, so
    # there is no forward state the backward needs.
    def _no_context(ctx, inputs, output):
        pass

    def _backward_to_nhwc(ctx, grad_output):
        return torch.ops.attention_kernels.permute_to_nchw.default(grad_output.contiguous())

    def _backward_to_nchw(ctx, grad_output):
        return torch.ops.attention_kernels.permute_to_nhwc.default(grad_output.contiguous())

    torch.library.register_autograd("attention_kernels::permute_to_nhwc", _backward_to_nhwc, setup_context=_no_context)
    torch.library.register_autograd("attention_kernels::permute_to_nchw", _backward_to_nchw, setup_context=_no_context)

    # No autocast rule is registered, deliberately. The autocast dispatch keys
    # fall through for ops without an explicit kernel, so these pass to the
    # backend with their input dtype untouched -- which is the correct behaviour
    # for a pure element permutation, and is what lets a conversion sit between
    # an autocast'd projection and the attention kernel without perturbing the
    # dtype either of them chose.
    #
    # This differs from the attention ops themselves (see the AutocastCUDA
    # registrations in optimized/attention_optimized.py and
    # kernels_torch/attention_torch.py), which *do* need a rule because they
    # must pull k/v/q to a common autocast dtype before the softmax. A layout
    # conversion has no such requirement: casting inside it would be a silent
    # precision change. test_autocast_preserves_dtype pins this.


def _permuted_copy(x: torch.Tensor, dims: Tuple[int, int, int, int]) -> torch.Tensor:
    """
    Permute ``x`` into a freshly allocated contiguous tensor.

    Not ``x.permute(dims).contiguous()``: when the permuted view is already
    contiguous -- which happens whenever ``C == 1`` or ``H * W == 1``, since the
    two layouts then hold identical bytes -- ``.contiguous()`` is a no-op and
    returns a view aliasing ``x``. The conversion must always hand back storage
    the caller owns, both to match the optimized path (which allocates) and so
    that writing into the result cannot corrupt the source.
    """

    view = x.permute(*dims)

    return view.clone() if view.is_contiguous() else view.contiguous()


def to_nhwc(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a channels-first tensor to physical channels-last.

    Parameters
    ----------
    x : torch.Tensor
        Contiguous tensor of shape ``(B, C, H, W)``.

    Returns
    -------
    torch.Tensor
        Contiguous tensor of shape ``(B, H, W, C)`` holding the same values.
    """

    # the message must not close over x: dynamo rejects a torch._check message
    # closure that captures anything other than Python constants
    check(x.dim() == 4, lambda: "to_nhwc expects a 4-dimensional (B, C, H, W) tensor")

    if _OPTIMIZED:
        return torch.ops.attention_kernels.permute_to_nhwc.default(x.contiguous())

    return _permuted_copy(x, (0, 2, 3, 1))


def to_nchw(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a physically channels-last tensor back to channels-first.

    Inverse of :func:`to_nhwc`.

    Parameters
    ----------
    x : torch.Tensor
        Contiguous tensor of shape ``(B, H, W, C)``.

    Returns
    -------
    torch.Tensor
        Contiguous tensor of shape ``(B, C, H, W)`` holding the same values.
    """

    check(x.dim() == 4, lambda: "to_nchw expects a 4-dimensional (B, H, W, C) tensor")

    if _OPTIMIZED:
        return torch.ops.attention_kernels.permute_to_nchw.default(x.contiguous())

    return _permuted_copy(x, (0, 3, 1, 2))
