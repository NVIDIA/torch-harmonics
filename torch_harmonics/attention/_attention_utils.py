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

# Input validation helpers.
#
# These exist because torch._check messages have to survive dynamo: the message
# argument must be a callable whose closure captures only Python constants. An
# inline ``lambda: f"... {tensor.dim()} ..."`` captures the tensor (and, when it
# mentions a module attribute, ``self``), which makes the enclosing function
# impossible to trace with fullgraph=True.
#
# The two cases differ in what can be reported:
#   - rank is static under dynamo, so the actual value is safe to interpolate
#   - an extent may be a SymInt under dynamic shapes, so it can be compared but
#     not put in the message; only the expected value (a plain int) can be


def _check_ndim(tensor: torch.Tensor, ndim: int, name: str) -> None:
    """Check a tensor's rank, with a dynamo-traceable error message."""

    # hoisted to a local int so the closure below captures a constant
    actual = tensor.dim()
    torch._check(actual == ndim, lambda: f"Expected {ndim}-dimensional {name} tensor, got {actual} dimensions")


def _check_extent(tensor: torch.Tensor, dim: int, expected: int, name: str) -> None:
    """Check one dimension of a tensor, with a dynamo-traceable error message."""

    # int() pins expected to a constant even when it arrives as a tensor-derived
    # value; the actual extent is deliberately absent from the message, since it
    # is a SymInt whenever shapes are dynamic
    expected = int(expected)
    torch._check(tensor.shape[dim] == expected, lambda: f"Expected {name} shape[{dim}] == {expected}")


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
        torch._check(tensor.dtype == ref, "all attention inputs must share a single dtype")


# Shared backward-context helper used by both the torch reference kernels
# (in kernels_torch/) and the optimized custom_op path (in optimized/).
def _setup_context_attention_backward(ctx, inputs, output):
    kw, vw, qw, quad_weights, col_idx, row_off, nh, nlon_in, nlat_out, nlon_out = inputs
    ctx.save_for_backward(col_idx, row_off, quad_weights, kw, vw, qw)
    ctx.nh = nh
    ctx.nlon_in = nlon_in
    ctx.nlat_out = nlat_out
    ctx.nlon_out = nlon_out
