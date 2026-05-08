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
