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

r"""
Pure-PyTorch neighborhood S2 attention on a ragged grid, plus its
``torch.library`` custom_op and autograd registration.

The counterpart of :mod:`.attention_torch` for grids whose latitude rings differ
in length, such as HEALPix. Fields are flat ``(batch, channels, npoints)`` rather
than ``(batch, channels, nlat, nlon)``, and the neighbourhood is keyed per output
*point* rather than per output latitude.

That last difference is what makes this a separate path rather than a flag on the
existing one. The regular kernels store one neighbour list per output latitude and
reach the other output longitudes by shifting the stored input columns by
``pscale * wo`` -- valid because rotating a product grid about the polar axis by one
output longitude step maps the grid onto itself. On a ragged grid it does not:
rotating a HEALPix ring of :math:`n_k` pixels by one of its own pixels maps that
ring onto itself but not the rings above and below, whose pixel counts differ. So
there is no shift to apply, ``row_off`` is indexed by output point, and ``col_idx``
holds absolute flat input indices.

Removing that trick makes this code *shorter* than the regular path -- no
``pscale``, no decomposition of a column into ``(hi, wi)``, one loop over output
points instead of two over latitude and longitude -- at the cost of a neighbour list
that is ``nlon_out`` times larger, since it can no longer be shared across a ring.

Like :mod:`.attention_torch` this is a correctness reference, not a fast path. It is
written per output point with the neighbours of that point handled as a batch, which
is enough to make it usable at the resolutions the tests need without turning the
inner mathematics into something hard to check by eye.
"""

from typing import Tuple

import torch

from torch_harmonics.utils import check

__all__ = ["_neighborhood_s2_attention_ragged_torch"]


def _gather_neighbors(col_idx: torch.Tensor, row_off: torch.Tensor, ipoint: int) -> torch.Tensor:
    """Flat input indices of the neighbours of one output point."""
    return col_idx[int(row_off[ipoint]) : int(row_off[ipoint + 1])]


def _softmax_state(kx: torch.Tensor, qy: torch.Tensor, point_weights: torch.Tensor, cols: torch.Tensor, ipoint: int) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    The shared first half of every routine below: the unnormalized attention weights
    of one output point and their sum.

    Returns :math:`\alpha_{pj} = e^{q_p \cdot k_j - m_p} w_j` for the neighbours *j* of
    output point *p*, and :math:`A_p = \sum_j \alpha_{pj}`. Subtracting the row maximum
    :math:`m_p` is the standard softmax stabilization; it cancels between numerator and
    denominator, so the returned pair describes the same normalized weights as the
    unshifted one would.

    Parameters
    ----------
    kx : torch.Tensor
        Keys, ``(batch, channels, npoints_in)``.
    qy : torch.Tensor
        Queries, ``(batch, channels, npoints_out)``.
    point_weights : torch.Tensor
        Quadrature weight of each *input point*, ``(npoints_in,)``. Unlike the regular
        path, which indexes weights by input latitude, a ragged grid's points differ in
        weight within a latitude only through the ring length, so the weight is carried
        per point and the kernels need not know the ring structure at all.
    cols : torch.Tensor
        Flat input indices of the neighbours of ``ipoint``.
    ipoint : int
        Index of the output point.

    Returns
    -------
    alpha : torch.Tensor
        ``(batch, nnz)``.
    alpha_sum : torch.Tensor
        ``(batch, 1)``.
    """
    # (batch, channels, nnz) against (batch, channels, 1), contracted over channels
    qdotk = torch.sum(qy[:, :, ipoint].unsqueeze(-1) * kx[:, :, cols], dim=1)

    qdotk_max = qdotk.max(dim=-1, keepdim=True).values
    alpha = torch.exp(qdotk - qdotk_max) * point_weights[cols]

    return alpha, alpha.sum(dim=-1, keepdim=True)


def _neighborhood_s2_attention_ragged_fwd_torch(
    kx: torch.Tensor, vx: torch.Tensor, qy: torch.Tensor, point_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor, npoints_out: int
) -> torch.Tensor:
    r"""
    Forward pass, :math:`y_p = \frac{\sum_j \alpha_{pj} v_j}{\sum_j \alpha_{pj}}`.
    """
    y = torch.zeros((qy.shape[0], vx.shape[1], npoints_out), dtype=qy.dtype, device=qy.device)

    for ipoint in range(npoints_out):
        cols = _gather_neighbors(col_idx, row_off, ipoint)
        if cols.numel() == 0:
            continue

        alpha, alpha_sum = _softmax_state(kx, qy, point_weights, cols, ipoint)

        # (batch, channels, nnz) weighted by (batch, 1, nnz), summed over neighbours
        y[:, :, ipoint] = torch.sum(alpha.unsqueeze(1) * vx[:, :, cols], dim=-1) / alpha_sum

    return y


def _neighborhood_s2_attention_ragged_bwd_dv_torch(
    kx: torch.Tensor, vx: torch.Tensor, qy: torch.Tensor, dy: torch.Tensor, point_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor, npoints_out: int
) -> torch.Tensor:
    r"""
    Gradient with respect to the values,
    :math:`\frac{\partial L}{\partial v_j} = \sum_p \frac{\alpha_{pj}}{A_p} \, \partial_y L|_p`.

    The values enter the output linearly, so this is just the normalized attention
    weights transposed -- no softmax derivative appears.
    """
    dvx = torch.zeros_like(vx)

    for ipoint in range(npoints_out):
        cols = _gather_neighbors(col_idx, row_off, ipoint)
        if cols.numel() == 0:
            continue

        alpha, alpha_sum = _softmax_state(kx, qy, point_weights, cols, ipoint)
        alpha_norm = alpha / alpha_sum

        # index_add_ rather than `dvx[:, :, cols] +=`: advanced-index assignment
        # silently keeps only the last write for a repeated index, so it would be
        # correct only as long as the neighbour list stays duplicate-free
        dvx.index_add_(2, cols, alpha_norm.unsqueeze(1) * dy[:, :, ipoint].unsqueeze(-1))

    return dvx


def _neighborhood_s2_attention_ragged_bwd_dk_torch(
    kx: torch.Tensor, vx: torch.Tensor, qy: torch.Tensor, dy: torch.Tensor, point_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor, npoints_out: int
) -> torch.Tensor:
    r"""
    Gradient with respect to the keys,
    :math:`\frac{\partial L}{\partial k_j} = \sum_p q_p \frac{\alpha_{pj}}{A_p}
    \left( g_{pj} - I_p \right)`, with :math:`g_{pj} = \partial_y L|_p \cdot v_j` and
    :math:`I_p = \sum_j \frac{\alpha_{pj}}{A_p} g_{pj}`.

    The bracket is the softmax Jacobian: a key raises its own attention weight and
    lowers every other weight in the same row, so what survives is the deviation of
    its own contribution from the row mean :math:`I_p`.
    """
    dkx = torch.zeros_like(kx)

    for ipoint in range(npoints_out):
        cols = _gather_neighbors(col_idx, row_off, ipoint)
        if cols.numel() == 0:
            continue

        alpha, alpha_sum = _softmax_state(kx, qy, point_weights, cols, ipoint)
        alpha_norm = alpha / alpha_sum

        gdotv = torch.sum(dy[:, :, ipoint].unsqueeze(-1) * vx[:, :, cols], dim=1)
        integral = torch.sum(alpha_norm * gdotv, dim=-1, keepdim=True)

        dkx.index_add_(2, cols, qy[:, :, ipoint].unsqueeze(-1) * (alpha_norm * (gdotv - integral)).unsqueeze(1))

    return dkx


def _neighborhood_s2_attention_ragged_bwd_dq_torch(
    kx: torch.Tensor, vx: torch.Tensor, qy: torch.Tensor, dy: torch.Tensor, point_weights: torch.Tensor, col_idx: torch.Tensor, row_off: torch.Tensor, npoints_out: int
) -> torch.Tensor:
    r"""
    Gradient with respect to the queries,
    :math:`\frac{\partial L}{\partial q_p} = \sum_j \frac{\alpha_{pj}}{A_p}
    \left( g_{pj} - I_p \right) k_j`.

    The same softmax Jacobian as for the keys, contracted against the keys instead of
    against a single query, and needing no scatter: each output point owns its query.
    """
    dqy = torch.zeros_like(qy)

    for ipoint in range(npoints_out):
        cols = _gather_neighbors(col_idx, row_off, ipoint)
        if cols.numel() == 0:
            continue

        alpha, alpha_sum = _softmax_state(kx, qy, point_weights, cols, ipoint)
        alpha_norm = alpha / alpha_sum

        gdotv = torch.sum(dy[:, :, ipoint].unsqueeze(-1) * vx[:, :, cols], dim=1)
        integral = torch.sum(alpha_norm * gdotv, dim=-1, keepdim=True)

        dqy[:, :, ipoint] = torch.sum((alpha_norm * (gdotv - integral)).unsqueeze(1) * kx[:, :, cols], dim=-1)

    return dqy


def _to_channels_first(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
    """``(batch, npoints, heads * channels)`` to ``(batch * heads, channels, npoints)``."""
    batch, npoints, _ = tensor.shape
    return tensor.transpose(1, 2).reshape(batch * num_heads, -1, npoints)


def _to_channels_last(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Inverse of :func:`_to_channels_first`."""
    batch_heads, channels, npoints = tensor.shape
    return tensor.reshape(batch_heads // num_heads, num_heads * channels, npoints).transpose(1, 2).contiguous()


@torch.library.custom_op("attention_kernels::_neighborhood_s2_attention_ragged_torch", mutates_args=())
def _neighborhood_s2_attention_ragged_torch(
    kw: torch.Tensor,
    vw: torch.Tensor,
    qw: torch.Tensor,
    point_weights: torch.Tensor,
    col_idx: torch.Tensor,
    row_off: torch.Tensor,
    nh: int,
    npoints_out: int,
) -> torch.Tensor:
    check(row_off.numel() == npoints_out + 1, lambda: f"row_off must have npoints_out + 1 = {npoints_out + 1} entries, got {row_off.numel()}")

    # The op ABI is channels-last with heads packed along the channel dimension, for
    # the same reason as the regular path: every projection around it is 1x1, so
    # channels-last makes them plain GEMMs. The routines above are channels-first
    # because that is what makes the neighbour gathers read as slices, so convert at
    # the boundary and fold the heads into the batch while doing it.
    kx = _to_channels_first(kw, nh)
    vx = _to_channels_first(vw, nh)
    qy = _to_channels_first(qw, nh)

    # promote for the softmax, and return in the dtype register_fake commits to
    inp_dtype = kw.dtype
    kx = kx.to(torch.float32)
    vx = vx.to(torch.float32)
    qy = qy.to(torch.float32)

    output = _neighborhood_s2_attention_ragged_fwd_torch(kx, vx, qy, point_weights, col_idx, row_off, npoints_out)

    return _to_channels_last(output, nh).to(dtype=inp_dtype)


@torch.library.register_fake("attention_kernels::_neighborhood_s2_attention_ragged_torch")
def _(
    kw: torch.Tensor,
    vw: torch.Tensor,
    qw: torch.Tensor,
    point_weights: torch.Tensor,
    col_idx: torch.Tensor,
    row_off: torch.Tensor,
    nh: int,
    npoints_out: int,
) -> torch.Tensor:
    return torch.empty((kw.shape[0], npoints_out, vw.shape[2]), dtype=kw.dtype, device=kw.device)


def _setup_context_ragged(ctx, inputs, output):
    kw, vw, qw, point_weights, col_idx, row_off, nh, npoints_out = inputs
    ctx.save_for_backward(col_idx, row_off, point_weights, kw, vw, qw)
    ctx.nh = nh
    ctx.npoints_out = npoints_out


def _neighborhood_s2_attention_ragged_bwd(ctx, grad_output):
    col_idx, row_off, point_weights, kw, vw, qw = ctx.saved_tensors
    nh, npoints_out = ctx.nh, ctx.npoints_out

    kw_needs_grad, vw_needs_grad, qw_needs_grad = ctx.needs_input_grad[:3]

    kx = _to_channels_first(kw, nh).to(torch.float32)
    vx = _to_channels_first(vw, nh).to(torch.float32)
    qy = _to_channels_first(qw, nh).to(torch.float32)
    dy = _to_channels_first(grad_output, nh).to(torch.float32)

    dkw = dvw = dqw = None

    if vw_needs_grad:
        dvx = _neighborhood_s2_attention_ragged_bwd_dv_torch(kx, vx, qy, dy, point_weights, col_idx, row_off, npoints_out)
        dvw = _to_channels_last(dvx, nh).to(dtype=vw.dtype)

    if kw_needs_grad:
        dkx = _neighborhood_s2_attention_ragged_bwd_dk_torch(kx, vx, qy, dy, point_weights, col_idx, row_off, npoints_out)
        dkw = _to_channels_last(dkx, nh).to(dtype=kw.dtype)

    if qw_needs_grad:
        dqy = _neighborhood_s2_attention_ragged_bwd_dq_torch(kx, vx, qy, dy, point_weights, col_idx, row_off, npoints_out)
        dqw = _to_channels_last(dqy, nh).to(dtype=qw.dtype)

    # one gradient per forward input: kw, vw, qw, then None for point_weights,
    # col_idx, row_off, nh, npoints_out
    return dkw, dvw, dqw, None, None, None, None, None


torch.library.register_autograd("attention_kernels::_neighborhood_s2_attention_ragged_torch", _neighborhood_s2_attention_ragged_bwd, setup_context=_setup_context_ragged)


def _make_autocast_impl(device_type):
    @torch.library.impl("attention_kernels::_neighborhood_s2_attention_ragged_torch", f"Autocast{device_type.upper()}")
    def _(kw, vw, qw, point_weights, col_idx, row_off, nh, npoints_out):
        cast_dtype = torch.get_autocast_dtype(device_type)
        with torch.amp.autocast(device_type, enabled=False):
            return _neighborhood_s2_attention_ragged_torch(kw.to(cast_dtype), vw.to(cast_dtype), qw.to(cast_dtype), point_weights, col_idx, row_off, nh, npoints_out)

    return _


_make_autocast_impl("cuda")
_make_autocast_impl("cpu")
