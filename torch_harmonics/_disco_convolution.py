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

import math

import torch

try:
    import disco_cuda_extension
    _cuda_extension_available = True
except ImportError as err:
    disco_cuda_extension = None
    _cuda_extension_available = False


class _DiscoS2ContractionCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, roff_idx: torch.Tensor, ker_idx: torch.Tensor,
                row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor,
                kernel_size: int, nlat_out: int, nlon_out: int):
        ctx.save_for_backward(roff_idx, ker_idx, row_idx, col_idx, vals)
        ctx.kernel_size = kernel_size
        ctx.nlat_in = x.shape[-2]
        ctx.nlon_in = x.shape[-1]

        return disco_cuda_extension.forward(x.contiguous(), roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out)

    @staticmethod
    def backward(ctx, grad_output):
        roff_idx, ker_idx, row_idx, col_idx, vals = ctx.saved_tensors
        grad_input = disco_cuda_extension.backward(grad_output.contiguous(), roff_idx, ker_idx, row_idx, col_idx, vals,
                                         ctx.kernel_size, ctx.nlat_in, ctx.nlon_in)

        return grad_input, None, None, None, None, None, None, None, None


class _DiscoS2TransposeContractionCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, roff_idx: torch.Tensor, ker_idx: torch.Tensor,
                row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor,
                kernel_size: int, nlat_out: int, nlon_out: int):
        ctx.save_for_backward(roff_idx, ker_idx, row_idx, col_idx, vals)
        ctx.kernel_size = kernel_size
        ctx.nlat_in = x.shape[-2]
        ctx.nlon_in = x.shape[-1]

        return disco_cuda_extension.backward(x.contiguous(), roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out)

    @staticmethod
    def backward(ctx, grad_output):
        roff_idx, ker_idx, row_idx, col_idx, vals = ctx.saved_tensors
        grad_input = disco_cuda_extension.forward(grad_output.contiguous(), roff_idx, ker_idx, row_idx, col_idx, vals,
                                        ctx.kernel_size, ctx.nlat_in, ctx.nlon_in)

        return grad_input, None, None, None, None, None, None, None, None

# CUDA
def _disco_s2_contraction_cuda(x: torch.Tensor, roff_idx: torch.Tensor, ker_idx: torch.Tensor,
                               row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor,
                               kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
    return _DiscoS2ContractionCuda.apply(x, roff_idx, ker_idx, row_idx, col_idx, vals,
                                         kernel_size, nlat_out, nlon_out)

def _disco_s2_transpose_contraction_cuda(x: torch.Tensor, roff_idx: torch.Tensor, ker_idx: torch.Tensor,
                                         row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor,
                                         kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
    return _DiscoS2TransposeContractionCuda.apply(x, roff_idx, ker_idx, row_idx, col_idx, vals,
                                                  kernel_size, nlat_out, nlon_out)


def _disco_s2_contraction_torch(x: torch.Tensor, psi: torch.Tensor, nlon_out: int):
    """
    Reference implementation of the custom contraction as described in [1]. This requires repeated
    shifting of the input tensor, which can potentially be costly. For an efficient implementation
    on GPU, make sure to use the custom kernel written in CUDA.
    """
    assert len(psi.shape) == 3
    assert len(x.shape) == 4
    psi = psi.to(x.device)

    batch_size, n_chans, nlat_in, nlon_in = x.shape
    kernel_size, nlat_out, _ = psi.shape

    assert psi.shape[-1] == nlat_in * nlon_in
    assert nlon_in % nlon_out == 0
    assert nlon_in >= nlat_out
    pscale = nlon_in // nlon_out

    # add a dummy dimension for nkernel and move the batch and channel dims to the end
    x = x.reshape(1, batch_size * n_chans, nlat_in, nlon_in).permute(0, 2, 3, 1)
    x = x.expand(kernel_size, -1, -1, -1)

    y = torch.zeros(nlon_out, kernel_size, nlat_out, batch_size * n_chans, device=x.device, dtype=x.dtype)

    for pout in range(nlon_out):
        # sparse contraction with psi
        y[pout] = torch.bmm(psi, x.reshape(kernel_size, nlat_in * nlon_in, -1))
        # we need to repeatedly roll the input tensor to faciliate the shifted multiplication
        x = torch.roll(x, -pscale, dims=2)

    # reshape y back to expose the correct dimensions
    y = y.permute(3, 1, 2, 0).reshape(batch_size, n_chans, kernel_size, nlat_out, nlon_out)

    return y


def _disco_s2_transpose_contraction_torch(x: torch.Tensor, psi: torch.Tensor, nlon_out: int):
    """
    Reference implementation of the custom contraction as described in [1]. This requires repeated
    shifting of the input tensor, which can potentially be costly. For an efficient implementation
    on GPU, make sure to use the custom kernel written in CUDA.
    """
    assert len(psi.shape) == 3
    assert len(x.shape) == 5
    psi = psi.to(x.device)

    batch_size, n_chans, kernel_size, nlat_in, nlon_in = x.shape
    kernel_size, nlat_out, n_out = psi.shape

    assert n_out % nlon_out == 0
    assert nlon_out >= nlat_in
    pscale = nlon_out // nlon_in

    # interleave zeros along the longitude dimension to allow for fractional offsets to be considered
    x_ext = torch.zeros(kernel_size, nlat_in, nlon_out, batch_size * n_chans, device=x.device, dtype=x.dtype)
    x = x.reshape(batch_size * n_chans, kernel_size, nlat_in, nlon_in).permute(1, 2, 3, 0)

    # x has shape kernel_size x nlat_in x nlon_in x batch_size * n_chans
    # we only need to apoply the nlon stride here, since nlat stride is taken care of by the kernel
    x_ext[:, :, ::pscale, :] = x[...]

    # create output tensor
    y = torch.zeros(kernel_size, nlon_out, nlat_out, batch_size * n_chans, device=x.device, dtype=x.dtype)

    for pout in range(nlon_out):
        # we need to repeatedly roll the input tensor to faciliate the shifted multiplication
        # TODO: double-check why this has to happen first
        x_ext = torch.roll(x_ext, -1, dims=2)
        # sparse contraction with the modified psi
        y[:, pout, :, :] = torch.bmm(psi, x_ext.reshape(kernel_size, nlat_in * nlon_out, -1))

    # sum over the kernel dimension and reshape to the correct output size
    y = y.sum(dim=0).permute(2, 1, 0).reshape(batch_size, n_chans, nlat_out, nlon_out).contiguous()

    return y

