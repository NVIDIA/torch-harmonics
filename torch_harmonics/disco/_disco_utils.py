# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
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

from typing import Optional

import torch


# compute dtype for the convolution operation
def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Upcast half types to float32 for numerics; pass float32/float64 through."""
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


# The weight contractions of the DISCO layers, as batched GEMMs over the groups. The
# K-expanded tensor (B, C, K, *spatial) is channel-first, so (C, K) and the spatial axes are
# already the two matrix axes of a GEMM. einsum instead makes the spatial axes the GEMM's rows
# and copies the K-expanded tensor into and out of that layout, which costs several times the
# GEMM itself. The forward and transpose layers, serial and distributed, regular and ragged,
# all contract through these three.
#
# A contraction over a single entry is an outer product and runs as a broadcast multiply:
# a GEMM with an inner dimension of 1 and an output over 2^31 elements hits an illegal memory
# access in cuBLAS (torch 2.13 nv26.06). The multiply is not autocast-eligible, so it casts
# both operands to the dtype the matmul would run in (see _outer).
def _outer(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``w * x`` in the dtype ``torch.matmul(w, x)`` would give: the autocast dtype if autocast is on, else the activation's."""
    device_type = x.device.type
    dtype = torch.get_autocast_dtype(device_type) if torch.is_autocast_enabled(device_type) else x.dtype
    return w.to(dtype) * x.to(dtype)


def _weight_contraction(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``(B, G*Cg, K, *spatial)`` x ``(G, Og, Cg, K)`` -> ``(B, G*Og, *spatial)``, summing over the channels and K."""
    G, Og, Cg, K = weight.shape
    B, spatial = x.shape[0], x.shape[3:]
    w, xf = weight.reshape(G, Og, Cg * K), x.reshape(B, G, Cg * K, -1)
    out = _outer(w, xf) if Cg * K == 1 else torch.matmul(w, xf)
    return out.reshape(B, G * Og, *spatial)


def _weight_expansion(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``(B, G*Og, *spatial)`` x ``(G, Og, Cg, K)`` -> ``(B, G*Cg, K, *spatial)``, the adjoint of :func:`_weight_contraction`."""
    G, Og, Cg, K = weight.shape
    B, spatial = x.shape[0], x.shape[2:]
    w, xf = weight.reshape(G, Og, Cg * K).transpose(1, 2), x.reshape(B, G, Og, -1)
    out = _outer(w, xf) if Og == 1 else torch.matmul(w, xf)
    return out.reshape(B, G * Cg, K, *spatial)


def _weight_gradient(grad_out: torch.Tensor, x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    The weight gradient of :func:`_weight_contraction`: ``(B, G*Og, *spatial)``, ``(B, G*Cg, K, *spatial)`` -> ``(G, Og, Cg, K)``.

    One bmm over the flat ``(B * G)`` batch, summed over B. Grouped layers run it as
    ``x @ grad_out^T``, which cuBLAS serves up to 3x faster for their narrow groups on H100
    and GB200; ungrouped ones as ``grad_out @ x^T``, the faster order there.
    """
    B, C, K = x.shape[:3]
    Og, Cg = grad_out.shape[1] // groups, C // groups
    go, xf = grad_out.reshape(B * groups, Og, -1), x.reshape(B * groups, Cg * K, -1)
    if groups > 1:
        grad = torch.bmm(xf, go.transpose(1, 2)).reshape(B, groups, Cg * K, Og).sum(0).transpose(1, 2)
    else:
        grad = torch.bmm(go, xf.transpose(1, 2)).reshape(B, groups, Og, Cg * K).sum(0)
    return grad.reshape(groups, Og, Cg, K)


# sparse-psi helper used by both the torch reference kernels (in kernels_torch/)
# and the optimized custom_op path (in optimized/).
def _get_psi(
    kernel_size: int,
    psi_idx: torch.Tensor,
    psi_vals: torch.Tensor,
    nlat_in: int,
    nlon_in: int,
    nlat_out: int,
    nlon_out: int,
    nlat_in_local: Optional[int] = None,
    nlat_out_local: Optional[int] = None,
    semi_transposed: Optional[bool] = False,
):
    """Creates a sparse tensor for spherical harmonic convolution operations."""
    nlat_in_local = nlat_in_local if nlat_in_local is not None else nlat_in
    nlat_out_local = nlat_out_local if nlat_out_local is not None else nlat_out

    # Indices originate from _precompute_convolution_tensor_s2 and are valid
    # by construction; skip the sparse-invariant pass (and silence the
    # implicit-default UserWarning).
    with torch.sparse.check_sparse_tensor_invariants(enable=False):
        if semi_transposed:
            # do partial transpose
            # we do a semi-transposition to faciliate the computation
            tout = psi_idx[2] // nlon_out
            pout = psi_idx[2] % nlon_out
            # flip the axis of longitudes
            pout = nlon_out - 1 - pout
            tin = psi_idx[1]
            idx = torch.stack([psi_idx[0], tout, tin * nlon_out + pout], dim=0)
            psi = torch.sparse_coo_tensor(idx, psi_vals, size=(kernel_size, nlat_out_local, nlat_in_local * nlon_out)).coalesce()
        else:
            psi = torch.sparse_coo_tensor(psi_idx, psi_vals, size=(kernel_size, nlat_out_local, nlat_in_local * nlon_in)).coalesce()
    return psi


def _get_psi_ragged(kernel_size: int, psi_idx: torch.Tensor, psi_vals: torch.Tensor, npoints_rows: int, npoints_cols: int, transposed: Optional[bool] = False):
    """
    The sparse psi of the ragged torch reference, ``(kernel_size, npoints_out, npoints_in)``.

    psi_idx holds ``(ker, row point, column point)``. The forward contracts it as given; the
    transpose scatters from the rows onto the columns, which is the same contraction with
    psi transposed, so ``transposed`` swaps the two point axes and the reference needs only
    one routine. There is no p-shift to build in, which is what makes this so much simpler
    than :func:`_get_psi`.
    """
    with torch.sparse.check_sparse_tensor_invariants(enable=False):
        if transposed:
            idx = torch.stack([psi_idx[0], psi_idx[2], psi_idx[1]], dim=0)
            size = (kernel_size, npoints_cols, npoints_rows)
        else:
            idx = psi_idx
            size = (kernel_size, npoints_rows, npoints_cols)
        psi = torch.sparse_coo_tensor(idx, psi_vals, size=size).coalesce()
    return psi
