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

import triton
import triton.language as tl

BLOCK_SIZE_BATCH = 4
BLOCK_SIZE_NZ = 8
BLOCK_SIZE_POUT = 8


@triton.jit
def _disco_s2_contraction_kernel(
    inz_ptr,
    vnz_ptr,
    nnz,
    inz_stride_ii,
    inz_stride_nz,
    vnz_stride,
    x_ptr,
    batch_size,
    nlat_in,
    nlon_in,
    x_stride_b,
    x_stride_t,
    x_stride_p,
    y_ptr,
    kernel_size,
    nlat_out,
    nlon_out,
    y_stride_b,
    y_stride_f,
    y_stride_t,
    y_stride_p,
    pscale,
    backward: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_NZ: tl.constexpr,
    BLOCK_SIZE_POUT: tl.constexpr,
):
    """
    Kernel for the sparse-dense contraction for the S2 DISCO convolution.
    """

    pid_batch = tl.program_id(0)
    pid_pout = tl.program_id(2)

    # pid_nz should always be 0 as we do not account for larger grids in this dimension
    pid_nz = tl.program_id(1)  # should be always 0
    tl.device_assert(pid_nz == 0)

    # create the pointer block for pout
    pout = pid_pout * BLOCK_SIZE_POUT + tl.arange(0, BLOCK_SIZE_POUT)
    b = pid_batch * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)

    # create pointer blocks for the psi datastructure
    iinz = tl.arange(0, BLOCK_SIZE_NZ)

    # get the initial pointers
    fout_ptrs = inz_ptr + iinz * inz_stride_nz
    tout_ptrs = inz_ptr + iinz * inz_stride_nz + inz_stride_ii
    tpnz_ptrs = inz_ptr + iinz * inz_stride_nz + 2 * inz_stride_ii
    vals_ptrs = vnz_ptr + iinz * vnz_stride

    # iterate in a blocked fashion over the non-zero entries
    for offs_nz in range(0, nnz, BLOCK_SIZE_NZ):
        # load input output latitude coordinate pairs
        fout = tl.load(fout_ptrs + offs_nz * inz_stride_nz, mask=(offs_nz + iinz < nnz), other=-1)
        tout = tl.load(tout_ptrs + offs_nz * inz_stride_nz, mask=(offs_nz + iinz < nnz), other=-1)
        tpnz = tl.load(tpnz_ptrs + offs_nz * inz_stride_nz, mask=(offs_nz + iinz < nnz), other=-1)

        # load corresponding values
        vals = tl.load(vals_ptrs + offs_nz * vnz_stride, mask=(offs_nz + iinz < nnz), other=0.0)

        # compute the shifted longitude coordinates p+p' to read in a coalesced fashion
        tnz = tpnz // nlon_in
        pnz = tpnz % nlon_in

        # make sure the value is not out of bounds
        tl.device_assert(fout < kernel_size)
        tl.device_assert(tout < nlat_out)
        tl.device_assert(tnz < nlat_in)
        tl.device_assert(pnz < nlon_in)

        # load corresponding portion of the input array
        x_ptrs = (
            x_ptr
            + tnz[None, :, None] * x_stride_t
            + ((pnz[None, :, None] + pout[None, None, :] * pscale) % nlon_in) * x_stride_p
            + b[:, None, None] * x_stride_b
        )
        y_ptrs = (
            y_ptr
            + fout[None, :, None] * y_stride_f
            + tout[None, :, None] * y_stride_t
            + (pout[None, None, :] % nlon_out) * y_stride_p
            + b[:, None, None] * y_stride_b
        )

        # precompute the mask
        mask = ((b[:, None, None] < batch_size) and (offs_nz + iinz[None, :, None] < nnz)) and (
            pout[None, None, :] < nlon_out
        )

        # do the actual computation. Backward is essentially just the same operation with swapped tensors.
        if not backward:
            x = tl.load(x_ptrs, mask=mask, other=0.0)
            y = vals[None, :, None] * x

            # store it to the output array
            tl.atomic_add(y_ptrs, y, mask=mask)
        else:
            y = tl.load(y_ptrs, mask=mask, other=0.0)
            x = vals[None, :, None] * y

            # store it to the output array
            tl.atomic_add(x_ptrs, x, mask=mask)


def _disco_s2_contraction_fwd(x: torch.Tensor, psi: torch.Tensor, nlon_out: int):
    """
    Wrapper function for the triton implementation of the efficient DISCO convolution on the sphere.

    Parameters
    ----------
    x: torch.Tensor
        Input signal on the sphere. Expects a tensor of shape batch_size x channels x nlat_in x nlon_in).
    psi : torch.Tensor
        Pre-computed convolution tensor. Expects a sparse tensor of shape kernel_size x nlat_out x (nlat_in * nlon_in).
    nlon_out: int
        Number of longitude points the output should have.
    """

    # check the shapes of all input tensors
    assert len(psi.shape) == 3
    assert len(x.shape) == 4
    assert psi.is_sparse, "Psi must be a sparse COO tensor"

    # TODO: check that Psi is also coalesced

    # get the dimensions of the problem
    kernel_size, nlat_out, n_in = psi.shape
    nnz = psi.indices().shape[-1]
    batch_size, n_chans, nlat_in, nlon_in = x.shape
    assert nlat_in * nlon_in == n_in

    # TODO: check that Psi index vector is of type long

    # make sure that the grid-points of the output grid fall onto the grid points of the input grid
    assert nlon_in % nlon_out == 0
    pscale = nlon_in // nlon_out

    # to simplify things, we merge batch and channel dimensions
    x = x.reshape(batch_size * n_chans, nlat_in, nlon_in)

    # prepare the output tensor
    y = torch.zeros(batch_size * n_chans, kernel_size, nlat_out, nlon_out, device=x.device, dtype=x.dtype)

    # determine the grid for the computation
    grid = (
        triton.cdiv(batch_size * n_chans, BLOCK_SIZE_BATCH),
        1,
        triton.cdiv(nlon_out, BLOCK_SIZE_POUT),
    )

    # launch the kernel
    _disco_s2_contraction_kernel[grid](
        psi.indices(),
        psi.values(),
        nnz,
        psi.indices().stride(-2),
        psi.indices().stride(-1),
        psi.values().stride(-1),
        x,
        batch_size * n_chans,
        nlat_in,
        nlon_in,
        x.stride(0),
        x.stride(-2),
        x.stride(-1),
        y,
        kernel_size,
        nlat_out,
        nlon_out,
        y.stride(0),
        y.stride(1),
        y.stride(-2),
        y.stride(-1),
        pscale,
        False,
        BLOCK_SIZE_BATCH,
        BLOCK_SIZE_NZ,
        BLOCK_SIZE_POUT,
    )

    # reshape y back to expose the correct dimensions
    y = y.reshape(batch_size, n_chans, kernel_size, nlat_out, nlon_out)

    return y


def _disco_s2_contraction_bwd(grad_y: torch.Tensor, psi: torch.Tensor, nlon_in: int):
    """
    Backward pass for the triton implementation of the efficient DISCO convolution on the sphere.

    Parameters
    ----------
    grad_y: torch.Tensor
        Input gradient on the sphere. Expects a tensor of shape batch_size x channels x kernel_size x nlat_out x nlon_out.
    psi : torch.Tensor
        Pre-computed convolution tensor. Expects a sparse tensor of shape kernel_size x nlat_out x (nlat_in * nlon_in).
    nlon_in: int
        Number of longitude points the input used. Is required to infer the correct dimensions
    """

    # check the shapes of all input tensors
    assert len(psi.shape) == 3
    assert len(grad_y.shape) == 5
    assert psi.is_sparse, "psi must be a sparse COO tensor"

    # TODO: check that Psi is also coalesced

    # get the dimensions of the problem
    kernel_size, nlat_out, n_in = psi.shape
    nnz = psi.indices().shape[-1]
    assert grad_y.shape[-2] == nlat_out
    assert grad_y.shape[-3] == kernel_size
    assert n_in % nlon_in == 0
    nlat_in = n_in // nlon_in
    batch_size, n_chans, _, _, nlon_out = grad_y.shape

    # make sure that the grid-points of the output grid fall onto the grid points of the input grid
    assert nlon_in % nlon_out == 0
    pscale = nlon_in // nlon_out
    
    # to simplify things, we merge batch and channel dimensions
    grad_y = grad_y.reshape(batch_size * n_chans, kernel_size, nlat_out, nlon_out)

    # prepare the output tensor
    grad_x = torch.zeros(batch_size * n_chans, nlat_in, nlon_in, device=grad_y.device, dtype=grad_y.dtype)

    # determine the grid for the computation
    grid = (
        triton.cdiv(batch_size * n_chans, BLOCK_SIZE_BATCH),
        1,
        triton.cdiv(nlon_out, BLOCK_SIZE_POUT),
    )

    # launch the kernel
    _disco_s2_contraction_kernel[grid](
        psi.indices(),
        psi.values(),
        nnz,
        psi.indices().stride(-2),
        psi.indices().stride(-1),
        psi.values().stride(-1),
        grad_x,
        batch_size * n_chans,
        nlat_in,
        nlon_in,
        grad_x.stride(0),
        grad_x.stride(-2),
        grad_x.stride(-1),
        grad_y,
        kernel_size,
        nlat_out,
        nlon_out,
        grad_y.stride(0),
        grad_y.stride(1),
        grad_y.stride(-2),
        grad_y.stride(-1),
        pscale,
        True,
        BLOCK_SIZE_BATCH,
        BLOCK_SIZE_NZ,
        BLOCK_SIZE_POUT,
    )

    # reshape y back to expose the correct dimensions
    grad_x = grad_x.reshape(batch_size, n_chans, nlat_in, nlon_in)

    return grad_x


class _DiscoS2ContractionTriton(torch.autograd.Function):
    """
    Helper function to make the triton implementation work with PyTorch autograd functionality
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, psi: torch.Tensor, nlon_out: int):
        ctx.save_for_backward(psi)
        ctx.nlon_in = x.shape[-1]

        return _disco_s2_contraction_fwd(x, psi, nlon_out)

    @staticmethod
    def backward(ctx, grad_output):
        (psi,) = ctx.saved_tensors
        grad_input = _disco_s2_contraction_bwd(grad_output, psi, ctx.nlon_in)
        grad_x = grad_psi = None

        return grad_input, None, None

class _DiscoS2TransposeContractionTriton(torch.autograd.Function):
    """
    Helper function to make the triton implementation work with PyTorch autograd functionality
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, psi: torch.Tensor, nlon_out: int):
        ctx.save_for_backward(psi)
        ctx.nlon_in = x.shape[-1]

        return _disco_s2_contraction_bwd(x, psi, nlon_out)

    @staticmethod
    def backward(ctx, grad_output):
        (psi,) = ctx.saved_tensors
        grad_input = _disco_s2_contraction_fwd(grad_output, psi, ctx.nlon_in)
        grad_x = grad_psi = None

        return grad_input, None, None


def _disco_s2_contraction_triton(x: torch.Tensor, psi: torch.Tensor, nlon_out: int):
    return _DiscoS2ContractionTriton.apply(x, psi, nlon_out)

def _disco_s2_transpose_contraction_triton(x: torch.Tensor, psi: torch.Tensor, nlon_out: int):
    return _DiscoS2TransposeContractionTriton.apply(x, psi, nlon_out)


def _disco_s2_contraction_torch(x: torch.Tensor, psi: torch.Tensor, nlon_out: int):
    """
    Reference implementation of the custom contraction as described in [1]. This requires repeated
    shifting of the input tensor, which can potentially be costly. For an efficient implementation
    on GPU, make sure to use the custom kernel written in Triton.
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
    on GPU, make sure to use the custom kernel written in Triton.
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

