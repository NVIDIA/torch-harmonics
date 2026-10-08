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


import torch

from torch_harmonics.utils import check


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


def _reciprocal_or_zero(alpha_sum: torch.Tensor) -> torch.Tensor:
    """
    ``1 / alpha_sum``, or zero where ``alpha_sum`` is zero.

    An output point whose neighbourhood is empty -- possible across two grids when the
    cutoff is below the input spacing -- has an empty softmax sum. Every path, the
    references, the kernels and the distributed finalize, returns zero output and zero
    gradient for it instead of dividing by that zero.
    """
    return torch.where(alpha_sum > 0, alpha_sum.reciprocal(), torch.zeros_like(alpha_sum))


# Input validation helpers.
#
# These exist because torch._check messages have to survive dynamo. A *callable*
# message never traces -- not even one returning a constant -- which is why the
# codebase routes these through torch_harmonics.utils.check, and why a message
# that has to interpolate a runtime value cannot simply be inlined here.
#
# The two cases differ in what can be reported:
#   - rank is static under dynamo, so the actual value is safe to interpolate
#   - an extent may be a SymInt under dynamic shapes, so it can be compared but
#     not put in the message; only the expected value (a plain int) can be


def _check_ndim(tensor: torch.Tensor, ndim: int, name: str) -> None:
    """Check a tensor's rank, with a dynamo-traceable error message."""

    # hoisted to a local int so the closure below captures a constant
    actual = tensor.dim()
    check(actual == ndim, lambda: f"Expected {ndim}-dimensional {name} tensor, got {actual} dimensions")


def _check_extent(tensor: torch.Tensor, dim: int, expected: int, name: str) -> None:
    """Check one dimension of a tensor, with a dynamo-traceable error message."""

    # int() pins expected to a constant even when it arrives as a tensor-derived
    # value; the actual extent is deliberately absent from the message, since it
    # is a SymInt whenever shapes are dynamic
    expected = int(expected)
    check(tensor.shape[dim] == expected, lambda: f"Expected {name} shape[{dim}] == {expected}")


def _check_dtypes_match(tensors) -> None:
    """
    Check that every tensor shares the first one's dtype.

    The kernels dispatch once, on q's scalar type, and then reinterpret_cast every
    activation pointer to that single element type. Mismatched inputs are therefore
    not merely rounded -- they are read as the wrong type, so a k/v tensor with a
    different dtype is silently misinterpreted rather than converted. This is the
    only place that turns it into an error.

    Like rank, dtype is static under dynamo (it is FakeTensor metadata, not data),
    so the comparison itself resolves at trace time into a guard rather than a graph
    break. Only the message needs care -- see below.
    """

    # The message is a plain literal: no f-string, no closure. Interpolating here
    # would put a dtype (and the loop's name variable) into the message, which is
    # what makes the enclosing forward untraceable. The identity of the offending
    # tensor is not lost -- the TORCH_CHECKs in the kernel entry points report it
    # with both actual dtypes, and this check only has to fire first.
    ref = tensors[0].dtype
    for tensor in tensors[1:]:
        check(tensor.dtype == ref, "all attention inputs must share a single dtype")


def _setup_context_attention_regular_optimized_backward(ctx, inputs, output):
    """
    Backward context for the compiled product-grid operator, which reads arcs.

    There were once one of these, shared with the torch reference, because both
    operators declared both forms of the neighbourhood and each ignored one. They now
    declare only what they read, so the input lists differ and so do these.
    """
    kw, vw, qw, ring_weights, seg, seg_off, nh, nlon_in, nlat_out, nlon_out = inputs
    ctx.save_for_backward(seg, seg_off, ring_weights, kw, vw, qw)
    ctx.nh = nh
    ctx.nlon_in = nlon_in
    ctx.nlat_out = nlat_out
    ctx.nlon_out = nlon_out


def _setup_context_attention_regular_reference_backward(ctx, inputs, output):
    """
    Backward context for the torch reference, which reads the column list.

    That it takes the columns and not the arcs is the whole of its value as a
    reference: the arcs are a derivation, and a reference that consumed them could not
    catch an error in deriving them. The signature now says so.
    """
    kw, vw, qw, ring_weights, col_idx, row_off, nh, nlon_in, nlat_out, nlon_out = inputs
    ctx.save_for_backward(col_idx, row_off, ring_weights, kw, vw, qw)
    ctx.nh = nh
    ctx.nlon_in = nlon_in
    ctx.nlat_out = nlat_out
    ctx.nlon_out = nlon_out


def _setup_context_attention_ragged_backward(ctx, inputs, output):
    """
    Backward context for the ragged optimized op.

    Two differences from the regular helper above. The neighbourhood is saved in arc
    form, with the ring tables -- both ragged backwards, CPU and CUDA, walk the arcs.

    The other is the three trailing outputs. The forward returns its softmax
    bookkeeping so the backward does not have to rebuild it, but that bookkeeping is
    not a function of the inputs in the differentiable sense: alpha_sum and qdotk_max
    are reduction statistics, and y_hi is the output over again, so a gradient routed
    through it would be counted twice. Marking them non-differentiable makes attempting
    any of that an error at the autograd level rather than a silently wrong number.
    """
    kw, vw, qw, ring_weights, psi_seg, psi_seg_off, ring_base, ring_size, nh, npoints_out = inputs
    y, y_hi, alpha_sum, qdotk_max = output

    ctx.save_for_backward(psi_seg, psi_seg_off, ring_base, ring_size, ring_weights, kw, vw, qw, y, y_hi, alpha_sum, qdotk_max)
    ctx.nh = nh
    ctx.npoints_out = npoints_out
    ctx.mark_non_differentiable(y_hi, alpha_sum, qdotk_max)
