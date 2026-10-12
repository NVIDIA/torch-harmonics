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

"""
Pure-PyTorch reference implementations of the DISCO contraction on a ragged grid.

The counterpart of ``disco_regular_torch`` for grids whose latitude rings differ in
length, such as HEALPix. Fields are flat ``(batch, channels, npoints)`` rather than
``(batch, channels, nlat, nlon)``, and psi is keyed per output *point* rather than per
output latitude.

That is what makes this so much shorter than the regular reference: with no p-shift there
is nothing to roll, and the contraction is a single sparse product per basis function.
The transpose is the same product with psi transposed, which the caller builds
(``_get_psi_ragged(..., transposed=True)``), followed by the sum over basis functions.

Like ``disco_regular_torch`` this is a correctness reference, independent of the arc
encoding the compiled kernels read, not a fast path.
"""

import torch

from torch_harmonics.utils import check

from .._disco_utils import _compute_dtype


def _contract_ragged(x: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Dense (K, npoints_in, B*C) by sparse psi (K, npoints_out, npoints_in), in the compute dtype: (K, npoints_out, B*C)."""
    xtype = x.dtype
    cdtype = _compute_dtype(xtype)
    x = x.to(cdtype).contiguous()
    psi = psi.to(device=x.device, dtype=cdtype)

    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        y = torch.bmm(psi, x)

    return y


def _disco_s2_contraction_ragged_torch(x: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation of the custom contraction on a ragged grid:
    ``(B, C, npoints_in) -> (B, C, K, npoints_out)``.
    """

    check(psi.dim() == 3, lambda: f"Expected 3-dimensional psi tensor, got {psi.dim()} dimensions")
    check(x.dim() == 3, lambda: f"Expected 3-dimensional input tensor, got {x.dim()} dimensions")

    batch_size, n_chans, npoints_in = x.shape
    kernel_size, npoints_out, _ = psi.shape

    check(psi.shape[-1] == npoints_in, lambda: f"Expected psi.shape[-1]=={npoints_in}, got {psi.shape[-1]}")

    # one copy of the input per basis function, points first and batch and channels last
    x = x.reshape(batch_size * n_chans, npoints_in).t().unsqueeze(0).expand(kernel_size, -1, -1)

    y = _contract_ragged(x, psi).to(x.dtype)

    # (K, npoints_out, B*C) -> (B, C, K, npoints_out)
    y = y.permute(2, 0, 1).reshape(batch_size, n_chans, kernel_size, npoints_out).contiguous()

    return y


def _disco_s2_transpose_contraction_ragged_torch(x: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation of the transpose contraction on a ragged grid:
    ``(B, C, K, npoints_in) -> (B, C, npoints_out)``, with psi already transposed to
    ``(K, npoints_out, npoints_in)``.
    """

    check(psi.dim() == 3, lambda: f"Expected 3-dimensional psi tensor, got {psi.dim()} dimensions")
    check(x.dim() == 4, lambda: f"Expected 4-dimensional input tensor, got {x.dim()} dimensions")

    batch_size, n_chans, kernel_size, npoints_in = x.shape
    _, npoints_out, _ = psi.shape

    check(psi.shape[0] == kernel_size, lambda: f"Expected psi.shape[0]=={kernel_size}, got {psi.shape[0]}")
    check(psi.shape[-1] == npoints_in, lambda: f"Expected psi.shape[-1]=={npoints_in}, got {psi.shape[-1]}")

    # (B, C, K, npoints_in) -> (K, npoints_in, B*C)
    x = x.reshape(batch_size * n_chans, kernel_size, npoints_in).permute(1, 2, 0)

    # the basis functions are summed in the compute dtype, before narrowing
    y = _contract_ragged(x, psi).sum(dim=0).to(x.dtype)

    # (npoints_out, B*C) -> (B, C, npoints_out)
    y = y.t().reshape(batch_size, n_chans, npoints_out).contiguous()

    return y
