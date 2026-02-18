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
import torch.fft as fft
import torch.nn as nn


def _pad_dim_right(x: torch.Tensor, dim: int, target_size: int, value: float = 0.0) -> torch.Tensor:
    """Pad tensor along a single dimension to target_size (right-side only)."""
    ndim = x.ndim
    dim = dim if dim >= 0 else ndim + dim
    pad_amount = target_size - x.shape[dim]
    # F.pad expects (left, right) for last dim, then second-to-last, etc.
    pad_spec = [0] * (2 * ndim)
    pad_spec[(ndim - 1 - dim) * 2 + 1] = pad_amount
    return nn.functional.pad(x, tuple(pad_spec), value=value)


def rfft(x: torch.Tensor, nmodes: Optional[int] = None, dim: int = -1, **kwargs) -> torch.Tensor:
    """
    Real FFT with the correct padding behavior.
    If nmodes is given and larger than x.size(dim), x is zero-padded along dim before FFT.
    """

    if "n" in kwargs:
        raise ValueError("The 'n' argument is not allowed. Use 'nmodes' instead.")

    x = fft.rfft(x, dim=dim, **kwargs)

    if nmodes is not None and nmodes > x.shape[dim]:
        x = _pad_dim_right(x, dim, nmodes, value=0.0)
    elif nmodes is not None and nmodes < x.shape[dim]:
        x = x.narrow(dim, 0, nmodes)

    return x

def irfft(x: torch.Tensor, n: Optional[int] = None, dim: int = -1, **kwargs) -> torch.Tensor:
    """
    Torch version of IRFFT handles paddign and truncation correctly.
    This routine only applies Hermitian symmetry to avoid artifacts which occur depending on the backend.
    """

    if n is None:
        n = 2 * (x.size(dim) - 1)

    # ensure that imaginary part of 0 and nyquist components are zero
    # this is important because not all backend algorithms provided through the
    # irfft interface ensure that
    x[..., 0].imag = 0.0
    if (n % 2 == 0) and (n // 2 < x.size(dim)):
        x[..., n // 2].imag = 0.0

    x = fft.irfft(x, n=n, dim=dim, **kwargs)

    return x