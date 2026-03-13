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
from disco_helpers import optimized_kernels_is_available
from . import disco_kernels
from torch_harmonics.fft import rfft, irfft

# custom kernels
if optimized_kernels_is_available():
    # raw forward fake
    @torch.library.register_fake("disco_kernels::forward")
    def _(inp: torch.Tensor, roff_idx: torch.Tensor, ker_idx: torch.Tensor, 
          row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor, 
          kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        out_shape = (inp.shape[0], inp.shape[1], kernel_size, nlat_out, nlon_out)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # raw backward fake
    @torch.library.register_fake("disco_kernels::backward")
    def _(inp: torch.Tensor, roff_idx: torch.Tensor, ker_idx: torch.Tensor, 
          row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor, 
          kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        out_shape = (inp.shape[0], inp.shape[1], nlat_out, nlon_out)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # forward
    @torch.library.custom_op("disco_kernels::_disco_s2_contraction_optimized", mutates_args=())
    def _disco_s2_contraction_optimized(
        inp: torch.Tensor, roff_idx: torch.Tensor, ker_idx: torch.Tensor, 
        row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor, 
        kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        itype = inp.dtype
        inp = inp.to(torch.float32).contiguous()
        out = disco_kernels.forward.default(inp, roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out)
        out = out.to(itype)
        return out

    # transpose
    @torch.library.custom_op("disco_kernels::_disco_s2_transpose_contraction_optimized", mutates_args=())
    def _disco_s2_transpose_contraction_optimized(
        inp: torch.Tensor, roff_idx: torch.Tensor, ker_idx: torch.Tensor,
        row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor,
        kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        itype = inp.dtype
        inp = inp.to(torch.float32).contiguous()
        out = disco_kernels.backward.default(inp, roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out)
        out = out.to(itype)
        return out
    
    # forward fake
    @torch.library.register_fake("disco_kernels::_disco_s2_contraction_optimized")
    def _(inp: torch.Tensor, roff_idx: torch.Tensor, ker_idx: torch.Tensor, 
          row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor, 
          kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        out_shape = (inp.shape[0], inp.shape[1], kernel_size, nlat_out, nlon_out)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # transpose fake
    @torch.library.register_fake("disco_kernels::_disco_s2_transpose_contraction_optimized")
    def _(inp: torch.Tensor, roff_idx: torch.Tensor, ker_idx: torch.Tensor,
          row_idx: torch.Tensor, col_idx: torch.Tensor, vals: torch.Tensor,
        kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        out_shape = (inp.shape[0], inp.shape[1], nlat_out, nlon_out)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

#general routines: this is the same for forward and transpose
def _setup_context_conv_backward(ctx, inputs, output):
    inp, roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out = inputs
    ctx.save_for_backward(roff_idx, ker_idx, row_idx, col_idx, vals)
    ctx.kernel_size = kernel_size
    ctx.nlat_in = inp.shape[-2]
    ctx.nlon_in = inp.shape[-1]

# convolution related
def _disco_s2_contraction_bwd_optimized(ctx, grad_output):
    roff_idx, ker_idx, row_idx, col_idx, vals = ctx.saved_tensors
    
    if ctx.needs_input_grad[0]:
        gtype =	grad_output.dtype
        grad_output = grad_output.to(torch.float32).contiguous()
        grad_input = disco_kernels.backward.default(grad_output, roff_idx, ker_idx, row_idx, col_idx, vals,
                                                    ctx.kernel_size, ctx.nlat_in, ctx.nlon_in)
        grad_input = grad_input.to(gtype)
    else:
        grad_input = None

    return grad_input, None, None, None, None, None, None, None, None

if optimized_kernels_is_available():
    torch.library.register_autograd(
        "disco_kernels::_disco_s2_contraction_optimized", _disco_s2_contraction_bwd_optimized, setup_context=_setup_context_conv_backward)

# Transpose convolution related
def _disco_s2_transpose_contraction_bwd_optimized(ctx, grad_output):
    roff_idx, ker_idx, row_idx, col_idx, vals = ctx.saved_tensors

    if ctx.needs_input_grad[0]:
        gtype = grad_output.dtype
        grad_output = grad_output.to(torch.float32).contiguous()
        grad_input = disco_kernels.forward.default(grad_output, roff_idx, ker_idx, row_idx, col_idx, vals,
                                                    ctx.kernel_size, ctx.nlat_in, ctx.nlon_in)
        grad_input = grad_input.to(gtype)
    else:
        grad_input = None

    return grad_input, None, None, None, None, None, None, None, None

if optimized_kernels_is_available():
    torch.library.register_autograd(
        "disco_kernels::_disco_s2_transpose_contraction_optimized", _disco_s2_transpose_contraction_bwd_optimized, setup_context=_setup_context_conv_backward)

# FFT-based contraction functions (banded representation)
def _precompute_psi_banded(psi_sparse: torch.Tensor, nlat_in: int, nlon: int):
    """Build a banded dense representation of psi directly from sparse COO data.

    Instead of densifying the full (K, nlat_out, nlat_in, nlon) tensor, this
    only stores the contiguous band of input latitudes that have nonzero entries
    for each output latitude, reducing memory by ~nlat_in/band_width.

    Returns
    -------
    psi_banded_fft_conj : (K, nlat_out, max_bw, nfreq) complex tensor
    gather_idx : (nlat_out, max_bw) long tensor of input latitude indices
    """
    K, nlat_out, _ = psi_sparse.shape
    psi = psi_sparse.coalesce()
    indices = psi.indices()  # (3, nnz)
    values = psi.values()    # (nnz,)

    ker_idx = indices[0]
    row_idx = indices[1]   # output lat
    col_idx = indices[2]   # input_lat * nlon + lon
    input_lat = col_idx // nlon
    input_lon = col_idx % nlon

    # Find min/max input lat per output lat (across all kernel indices)
    lat_min = torch.full((nlat_out,), nlat_in, dtype=torch.long)
    lat_max = torch.full((nlat_out,), 0, dtype=torch.long)
    lat_min.scatter_reduce_(0, row_idx, input_lat, reduce="amin")
    lat_max.scatter_reduce_(0, row_idx, input_lat, reduce="amax")

    # Handle empty rows
    empty = lat_min >= nlat_in
    lat_min[empty] = 0
    lat_max[empty] = 0

    max_bw = (lat_max - lat_min + 1).max().item()

    # Build banded tensor from sparse entries (no full densification)
    psi_banded = torch.zeros(K, nlat_out, max_bw, nlon, dtype=values.dtype)
    banded_lat = input_lat - lat_min[row_idx]
    psi_banded[ker_idx, row_idx, banded_lat, input_lon] = values

    # Precompute FFT and gather index (clamped to valid range for zero-padded bands)
    psi_banded_fft_conj = rfft(psi_banded, dim=-1).conj()
    gather_idx = lat_min.unsqueeze(1) + torch.arange(max_bw).unsqueeze(0)  # (nlat_out, max_bw)
    gather_idx = gather_idx.clamp(max=nlat_in - 1)

    return psi_banded_fft_conj, gather_idx


def _disco_s2_contraction_fft(x: torch.Tensor, psi_fft_conj: torch.Tensor, gather_idx: torch.Tensor, nlon_out: int):
    """
    FFT-based DISCO S2 contraction using banded psi representation.

    Parameters
    ----------
    x : torch.Tensor of shape (B, C, nlat_in, nlon_in)
    psi_fft_conj : torch.Tensor of shape (K, nlat_out, bw, nfreq)
        Precomputed conjugate FFT of the banded psi kernel.
    gather_idx : torch.Tensor of shape (nlat_out, bw)
        Input latitude indices for each output latitude band.
    nlon_out : int

    Returns
    -------
    torch.Tensor of shape (B, C, K, nlat_out, nlon_out)
    """
    batch_size, n_chans, nlat_in, nlon_in = x.shape
    kernel_size, nlat_out, bw, nfreq = psi_fft_conj.shape
    pscale = nlon_in // nlon_out

    # FFT of input along longitude
    X_f = rfft(x.to(torch.float32), dim=-1)  # (B, C, nlat_in, nfreq)
    X_f = X_f.reshape(batch_size * n_chans, nlat_in, nfreq)

    # Gather relevant input lats for each output lat
    X_f_gathered = X_f[:, gather_idx, :]  # (B*C, nlat_out, bw, nfreq)

    # Cross-correlate: einsum over band width and frequency, then irfft
    Y_f = torch.einsum("kowf,bowf->bkof", psi_fft_conj.to(X_f.dtype), X_f_gathered)

    # Inverse FFT
    y = irfft(Y_f, n=nlon_in, dim=-1)  # (B*C, K, nlat_out, nlon_in)

    # Subsample for stride
    y = y[..., ::pscale]  # (B*C, K, nlat_out, nlon_out)

    y = y.reshape(batch_size, n_chans, kernel_size, nlat_out, nlon_out).contiguous()
    return y.to(x.dtype)


def _disco_s2_transpose_contraction_fft(x: torch.Tensor, psi_fft_conj: torch.Tensor, gather_idx: torch.Tensor, nlon_out: int):
    """
    FFT-based DISCO S2 transpose contraction using banded psi representation.

    Parameters
    ----------
    x : torch.Tensor of shape (B, C, K, nlat_in, nlon_in)
    psi_fft_conj : torch.Tensor of shape (K, nlat_out, bw, nfreq)
        Precomputed conjugate FFT of the banded semi-transposed psi kernel.
    gather_idx : torch.Tensor of shape (nlat_out, bw)
        Input latitude indices for each output latitude band.
    nlon_out : int

    Returns
    -------
    torch.Tensor of shape (B, C, nlat_out, nlon_out)
    """
    batch_size, n_chans, kernel_size, nlat_in, nlon_in = x.shape
    _, nlat_out, bw, nfreq = psi_fft_conj.shape

    assert nlon_out >= nlon_in
    pscale = nlon_out // nlon_in

    # Upsample x by interleaving zeros along longitude
    x = x.reshape(batch_size * n_chans, kernel_size, nlat_in, nlon_in)
    x_ext = torch.zeros(batch_size * n_chans, kernel_size, nlat_in, nlon_out, device=x.device, dtype=x.dtype)
    x_ext[..., ::pscale] = x

    # Apply constant offset before FFT (matches loop-based roll(-1) convention)
    x_ext = torch.roll(x_ext, -1, dims=-1)

    # FFT of upsampled input along longitude
    X_f = rfft(x_ext.to(torch.float32), dim=-1)  # (B*C, K, nlat_in, nfreq)

    # Loop over kernel index to avoid materializing a large 5D gathered tensor
    Y_f = torch.zeros(batch_size * n_chans, nlat_out, nfreq, device=X_f.device, dtype=X_f.dtype)
    for k in range(kernel_size):
        X_f_k_gathered = X_f[:, k][:, gather_idx, :]  # (B*C, nlat_out, bw, nfreq)
        Y_f += torch.einsum("owf,bowf->bof", psi_fft_conj[k].to(X_f.dtype), X_f_k_gathered)

    # Inverse FFT
    y = irfft(Y_f, n=nlon_out, dim=-1)  # (B*C, nlat_out, nlon_out)

    y = y.reshape(batch_size, n_chans, nlat_out, nlon_out).contiguous()
    return y.to(x.dtype)


# torch kernel related functions
def _get_psi(kernel_size: int, psi_idx: torch.Tensor, psi_vals: torch.Tensor, nlat_in: int, nlon_in: int, nlat_out: int, nlon_out: int, nlat_in_local: Optional[int] = None, nlat_out_local: Optional[int] = None, semi_transposed: Optional[bool] = False):
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

# convolution
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
    xtype = x.dtype
    x = x.to(torch.float32).contiguous()

    y = torch.zeros(nlon_out, kernel_size, nlat_out, batch_size * n_chans, device=x.device, dtype=x.dtype)

    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        for pout in range(nlon_out):
            # sparse contraction with psi
            y[pout] = torch.bmm(psi, x.reshape(kernel_size, nlat_in * nlon_in, -1))
            # we need to repeatedly roll the input tensor to faciliate the shifted multiplication
            x = torch.roll(x, -pscale, dims=2)

    y = y.to(xtype)

    # reshape y back to expose the correct dimensions
    y = y.permute(3, 1, 2, 0).reshape(batch_size, n_chans, kernel_size, nlat_out, nlon_out).contiguous()

    return y

# transpose convolution
def _disco_s2_transpose_contraction_torch(x: torch.Tensor, psi: torch.Tensor, nlon_out: int):
    assert len(psi.shape) == 3
    assert len(x.shape) == 5
    psi = psi.to(x.device)

    batch_size, n_chans, kernel_size, nlat_in, nlon_in = x.shape
    kernel_size, nlat_out, n_out = psi.shape

    assert n_out % nlon_out == 0
    assert nlon_out >= nlon_in
    pscale = nlon_out // nlon_in

    # interleave zeros along the longitude dimension to allow for fractional offsets to be considered
    x_ext = torch.zeros(kernel_size, nlat_in, nlon_out, batch_size * n_chans, device=x.device, dtype=x.dtype)
    x = x.reshape(batch_size * n_chans, kernel_size, nlat_in, nlon_in).permute(1, 2, 3, 0)

    # x has shape kernel_size x nlat_in x nlon_in x batch_size * n_chans
    # we only need to apoply the nlon stride here, since nlat stride is taken care of by the kernel
    x_ext[:, :, ::pscale, :] = x[...]

    xtype = x_ext.dtype
    x_ext = x_ext.to(torch.float32).contiguous()

    # create output tensor
    y = torch.zeros(kernel_size, nlon_out, nlat_out, batch_size * n_chans, device=x.device, dtype=x.dtype)

    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        for pout in range(nlon_out):
            # we need to repeatedly roll the input tensor to faciliate the shifted multiplication
            # TODO: double-check why this has to happen first
            x_ext = torch.roll(x_ext, -1, dims=2)
            # sparse contraction with the modified psi
            y[:, pout, :, :] = torch.bmm(psi, x_ext.reshape(kernel_size, nlat_in * nlon_out, -1))

    # sum over the kernel dimension and reshape to the correct output size
    y = y.sum(dim=0).permute(2, 1, 0).reshape(batch_size, n_chans, nlat_out, nlon_out)
    
    # convert datatype back to input type
    y = y.to(xtype).contiguous()

    return y

