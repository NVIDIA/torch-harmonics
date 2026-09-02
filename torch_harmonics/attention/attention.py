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

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_helpers import optimized_kernels_is_available

from torch_harmonics.attention._attention_utils import _build_psi_segments, _check_dtypes_match, _check_extent, _check_ndim
from torch_harmonics.attention._layout import to_nchw, to_nhwc
from torch_harmonics.attention.kernels_torch.attention_torch import _neighborhood_s2_attention_torch
from torch_harmonics.attention.optimized.attention_optimized import _neighborhood_s2_attention_optimized
from torch_harmonics.disco.convolution import _precompute_convolution_tensor_s2
from torch_harmonics.filter_basis import get_filter_basis
from torch_harmonics.grid import GridS2, require_grid
from torch_harmonics.truncation import truncate_support


class AttentionS2(nn.Module):
    r"""
    (Global) attention on the 2-sphere.

    This is ordinary (global) scaled dot-product attention, made geometrically
    faithful on the sphere by folding the numerical quadrature weights of the
    grid into the attention. Following :cite:`Bonev2025`, the softmax over keys becomes a
    quadrature approximation of a continuous attention integral over the sphere:
    the *logarithms* of the spherical quadrature weights are added to the
    pre-softmax attention scores as an additive mask, so that after the softmax
    exponential they act as multiplicative quadrature weights in the
    normalization. Using log-weights lets them be passed directly as the
    ``attn_mask`` of :func:`torch.nn.functional.scaled_dot_product_attention`.

    Incorporating the quadrature weights this way makes the layer a
    resolution-agnostic neural operator (evaluable on arbitrary grids, though the
    learned features remain resolution dependent) and approximately
    :math:`SO(3)`-equivariant, since the underlying integral is invariant under
    rotations (the Haar measure). For the local variant that confines attention
    to a geodesic neighborhood, see
    :class:`~torch_harmonics.NeighborhoodAttentionS2`.

    Parameters
    ----------
    grid_in : GridS2
        Descriptor of the input grid; it carries the resolution as well as the
        quadrature rule.
    grid_out : GridS2
        Descriptor of the output grid.
    in_channels : int
        number of channels of the input signal (corresponds to embed_dim in MHA in PyTorch)
    num_heads : int
        number of attention heads
    scale : torch.Tensor or float, optional
        Scaling applied to the attention logits. If None (default), the usual
        :math:`1/\sqrt{d}` scaling is used, with :math:`d` the head dimension.
    use_qknorm : bool, optional
        if specified, applies a learnable per-head RMS normalization to the
        queries and keys before scaling, by default ``False``
    bias : bool, optional
        if specified, adds bias to input / output projection layers
    k_channels : int
        number of dimensions for interior inner product in the attention matrix (corresponds to kdim in MHA in PyTorch)
    out_channels : int, optional
        number of dimensions for interior inner product in the attention matrix (corresponds to vdim in MHA in PyTorch)
    drop_rate : float, optional
        Dropout probability applied to the attention weights during training,
        by default ``0.0``

    References
    ----------
    :cite:`Bonev2025`
    """

    def __init__(
        self,
        grid_in: GridS2,
        grid_out: GridS2,
        in_channels: int,
        num_heads: int,
        scale: Optional[Union[torch.Tensor, float]] = None,
        use_qknorm: Optional[bool] = False,
        bias: Optional[bool] = True,
        k_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        drop_rate: Optional[float] = 0.0,
    ):
        super().__init__()

        self.grid_in = require_grid(grid_in, "grid_in")
        self.grid_out = require_grid(grid_out, "grid_out")
        self.nlat_in, self.nlon_in = self.grid_in.shape
        self.nlat_out, self.nlon_out = self.grid_out.shape

        if self.nlon_in % self.nlon_out != 0:
            raise ValueError(f"nlon_in ({self.nlon_in}) must be an integer multiple of nlon_out ({self.nlon_out}) for the attention p-shift to be exact")

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.k_channels = in_channels if k_channels is None else k_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.drop_rate = drop_rate
        self.scale = scale

        # integration weights
        wgl = self.grid_in.quad_weights
        quad_weights = 2.0 * torch.pi * wgl.to(dtype=torch.float32) / self.nlon_in
        # we need to tile and flatten them accordingly
        quad_weights = torch.tile(quad_weights.reshape(-1, 1), (1, self.nlon_in)).flatten()

        # compute log because they are applied as an addition prior to the softmax ('attn_mask'), which includes an exponential.
        # see https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        # for info on how 'attn_mask' is applied to the attention weights
        log_quad_weights = torch.log(quad_weights).reshape(1, 1, -1)
        self.register_buffer("log_quad_weights", log_quad_weights, persistent=False)

        # learnable parameters — Xavier uniform init matching PyTorch MHA convention:
        # bound = sqrt(6 / (fan_in + fan_out)) for each projection
        if self.k_channels % self.num_heads != 0:
            raise ValueError(f"Please make sure that number of heads {self.num_heads} divides k_channels {self.k_channels} evenly.")
        if self.out_channels % self.num_heads != 0:
            raise ValueError(f"Please make sure that number of heads {self.num_heads} divides out_channels {self.out_channels} evenly.")
        scale_qk = math.sqrt(6.0 / (self.in_channels + self.k_channels))
        scale_v = math.sqrt(6.0 / (self.in_channels + self.out_channels))
        scale_proj = math.sqrt(3.0 / self.out_channels)
        self.q_weights = nn.Parameter(scale_qk * (2 * torch.rand(self.k_channels, self.in_channels, 1, 1) - 1))
        self.k_weights = nn.Parameter(scale_qk * (2 * torch.rand(self.k_channels, self.in_channels, 1, 1) - 1))
        self.v_weights = nn.Parameter(scale_v * (2 * torch.rand(self.out_channels, self.in_channels, 1, 1) - 1))
        self.proj_weights = nn.Parameter(scale_proj * (2 * torch.rand(self.out_channels, self.out_channels, 1, 1) - 1))

        if bias:
            self.q_bias = nn.Parameter(torch.zeros(self.k_channels))
            self.k_bias = nn.Parameter(torch.zeros(self.k_channels))
            self.v_bias = nn.Parameter(torch.zeros(self.out_channels))
            self.proj_bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
            self.proj_bias = None

        if use_qknorm:
            self.q_norm_weights = nn.Parameter(torch.zeros(self.k_channels // self.num_heads))
            self.k_norm_weights = nn.Parameter(torch.zeros(self.k_channels // self.num_heads))
        else:
            self.q_norm_weights = None
            self.k_norm_weights = None

    def extra_repr(self):
        return f"grid_in={self.grid_in!r},\ngrid_out={self.grid_out!r},\nin_channels={self.in_channels}, out_channels={self.out_channels}, k_channels={self.k_channels}"

    def forward(self, query: torch.Tensor, key: Optional[torch.Tensor] = None, value: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply global attention on the sphere.

        Parameters
        ----------
        query : torch.Tensor
            Query signal of shape ``(batch, in_channels, nlat_out, nlon_out)`` (sampled on the output grid).
        key : torch.Tensor, optional
            Key signal of shape ``(batch, in_channels, nlat_in, nlon_in)``. Defaults to ``query`` (self-attention).
        value : torch.Tensor, optional
            Value signal of shape ``(batch, in_channels, nlat_in, nlon_in)``. Defaults to ``query`` (self-attention).

        Returns
        -------
        torch.Tensor
            Attention output of shape ``(batch, out_channels, nlat_out, nlon_out)``.
        """

        # self attention simplification
        if key is None:
            key = query

        if value is None:
            value = query

        # change this later to allow arbitrary number of batch dims
        _check_ndim(query, 4, "query")
        _check_ndim(key, 4, "key")
        _check_ndim(value, 4, "value")
        _check_dtypes_match((query, key, value))

        # perform QKV projections
        query = nn.functional.conv2d(query, self.q_weights, bias=self.q_bias)
        key = nn.functional.conv2d(key, self.k_weights, bias=self.k_bias)
        value = nn.functional.conv2d(value, self.v_weights, bias=self.v_bias)

        # reshape
        B, _, H, W = query.shape
        query = query.reshape(B, self.num_heads, -1, H, W)
        B, _, H, W = key.shape
        key = key.reshape(B, self.num_heads, -1, H, W)
        B, _, H, W = value.shape
        value = value.reshape(B, self.num_heads, -1, H, W)

        # reshape to the right dimensions
        B, _, C, H, W = query.shape
        query = query.permute(0, 1, 3, 4, 2).reshape(B, self.num_heads, H * W, C)
        B, _, C, H, W = key.shape
        key = key.permute(0, 1, 3, 4, 2).reshape(B, self.num_heads, H * W, C)
        B, _, C, H, W = value.shape
        value = value.permute(0, 1, 3, 4, 2).reshape(B, self.num_heads, H * W, C)

        if self.q_norm_weights is not None:
            query = F.rms_norm(query, normalized_shape=self.q_norm_weights.shape, weight=1 + self.q_norm_weights)
        if self.k_norm_weights is not None:
            key = F.rms_norm(key, normalized_shape=self.k_norm_weights.shape, weight=1 + self.k_norm_weights)

        # apply scale — if scale is a tensor (e.g. learnable), multiply into query
        # directly since SDPA only accepts a float scale
        dropout_p = self.drop_rate if self.training else 0.0
        if isinstance(self.scale, torch.Tensor):
            query = query * self.scale
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=self.log_quad_weights, dropout_p=dropout_p, scale=1.0)
        else:
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=self.log_quad_weights, dropout_p=dropout_p, scale=self.scale)

        # reshape
        B, _, _, C = out.shape
        # (B, heads, H*W, C)
        out = out.permute(0, 1, 3, 2)
        # (B, heads, C, H*W)
        out = out.reshape(B, self.num_heads * C, self.nlat_out, self.nlon_out)
        # (B, heads*C, H, W)
        out = nn.functional.conv2d(out, self.proj_weights, bias=self.proj_bias)

        return out


class NeighborhoodAttentionS2(nn.Module):
    r"""
    Neighborhood attention on the 2-sphere.

    This is the local counterpart of :class:`~torch_harmonics.AttentionS2`.
    Instead of attending globally, every output location attends only to the
    input points inside a geodesic neighborhood around it -- the spherical disk
    :math:`D(x) = \{x' \in S^2 : d(x, x') \le \theta_\mathrm{cutoff}\}`, where
    :math:`d(\cdot, \cdot)` is the great-circle (Haversine) distance and
    :math:`\theta_\mathrm{cutoff}` the cutoff radius. Restricting attention to
    this disk adds an inductive bias for locality and lowers the cost from
    :math:`\mathcal{O}(N^2)` to :math:`\mathcal{O}(k N)`, where :math:`k` is the
    number of points in a neighborhood.

    Following :cite:`Bonev2025`, the attention softmax integrates over the neighborhood
    against the sphere's numerical quadrature weights. This makes the layer a
    resolution-agnostic neural operator -- it can be evaluated on arbitrary grid
    resolutions (though the learned features themselves remain resolution
    dependent) -- and approximately :math:`SO(3)`-equivariant, since the
    underlying integrals are invariant under rotations (the Haar measure).

    The sparse neighborhood structure is precomputed with the same
    discrete-continuous construction used for the DISCO convolutions
    (:class:`~torch_harmonics.DiscreteContinuousConvS2`):
    Here, only the suppot (index information) of the zero order DISCO kernel is
    used to define an indicator function of the cutoff disk, so that any
    input point contributes to an output location exactly when it lies within
    :math:`\theta_\mathrm{cutoff}` of it. The relative weight of each input point
    depends on their contribution to the softmax as well as their quadrature weights.

    Parameters
    ----------
    grid_in : GridS2
        Descriptor of the input grid; it carries the resolution as well as the
        quadrature rule.
    grid_out : GridS2
        Descriptor of the output grid.
    in_channels : int
        number of channels of the input signal (corresponds to embed_dim in MHA in PyTorch)
    num_heads : int, optional
        number of attention heads, by default ``1``
    scale : torch.Tensor or float, optional
        Scaling applied to the queries after normalization. If None (default),
        :math:`1/\sqrt{d}` is used, with :math:`d` the per-head dimension.
    use_qknorm : bool, optional
        if specified, applies a learnable per-head RMS normalization to the
        queries and keys before scaling, by default ``False``
    bias : bool, optional
        if specified, adds bias to input / output projection layers
    theta_cutoff : float, optional
        Angular radius of the geodesic neighborhood disk, in radians. Input points
        farther than this from an output location are excluded from its attention.
        If None (default), it is set to one latitudinal grid spacing of the coarser
        of the input and output grids, see
        :func:`torch_harmonics.truncate_support`. Must be positive.
    k_channels : int
        number of dimensions for interior inner product in the attention matrix (corresponds to kdim in MHA in PyTorch)
    out_channels : int, optional
        number of dimensions for interior inner product in the attention matrix (corresponds to vdim in MHA in PyTorch)
    optimized_kernel : Optional[bool]
        Whether to use the optimized kernel (if available)

    References
    ----------
    :cite:`Bonev2025`
    """

    def __init__(
        self,
        grid_in: GridS2,
        grid_out: GridS2,
        in_channels: int,
        num_heads: Optional[int] = 1,
        scale: Optional[Union[torch.Tensor, float]] = None,
        use_qknorm: Optional[bool] = False,
        bias: Optional[bool] = True,
        theta_cutoff: Optional[float] = None,
        k_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        optimized_kernel: Optional[bool] = True,
    ):
        super().__init__()

        self.grid_in = require_grid(grid_in, "grid_in")
        self.grid_out = require_grid(grid_out, "grid_out")
        self.nlat_in, self.nlon_in = self.grid_in.shape
        self.nlat_out, self.nlon_out = self.grid_out.shape

        # direction selection: gather (self / downsample) iff nlon_in is an integer
        # multiple of nlon_out; scatter (upsample) iff nlon_out is an integer multiple
        # of nlon_in. Self-attention (nlon_in == nlon_out) satisfies both and falls
        # through the gather path with pscale == 1.
        self.upsample = (self.nlon_out % self.nlon_in == 0) and (self.nlon_in % self.nlon_out != 0)
        if not (self.nlon_in % self.nlon_out == 0 or self.upsample):
            raise ValueError(f"either nlon_in ({self.nlon_in}) must be an integer multiple of nlon_out ({self.nlon_out}), or vice versa, for the attention p-shift to be exact")

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.k_channels = in_channels if k_channels is None else k_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.optimized_kernel = optimized_kernel and optimized_kernels_is_available()

        # heuristic to compute theta cutoff based on the bandlimit of the input field
        # and overlaps of the basis functions. For upsample we follow DISCO's transpose
        # convention and use the coarser (input) grid spacing.
        # the coarser of the two grids sets the support: the input when upsampling,
        # the output otherwise
        self.theta_cutoff = truncate_support(self.grid_in if self.upsample else self.grid_out, theta_cutoff)

        # integration weights live on the input grid
        wgl = self.grid_in.quad_weights
        quad_weights = 2.0 * torch.pi * wgl.to(dtype=torch.float32) / self.nlon_in
        self.register_buffer("quad_weights", quad_weights, persistent=False)

        # create a dummy filter basis to pass to the construction of the convolution tensor
        # this is to avoid code duplication as the logic of pre-computing the sparsity pattern
        # is identical to convolutions with a constant filter function
        fb = get_filter_basis(kernel_shape=1, basis_type="zernike")

        # precompute the neighborhood sparsity pattern. For upsample we mirror DISCO's
        # transpose module: pass shapes swapped + transpose_normalization=True so that
        # rows of psi index the (smaller) input grid and cols encode the (larger)
        # output grid as ho_big * nlon_out + wo_big_canonical.
        if self.upsample:
            idx, _, roff = _precompute_convolution_tensor_s2(
                self.grid_out,
                self.grid_in,
                fb,
                theta_cutoff=self.theta_cutoff,
                transpose_normalization=True,
                basis_norm_mode="none",
                merge_quadrature=True,
            )
        else:
            idx, _, roff = _precompute_convolution_tensor_s2(
                self.grid_in,
                self.grid_out,
                fb,
                theta_cutoff=self.theta_cutoff,
                transpose_normalization=False,
                basis_norm_mode="none",
                merge_quadrature=True,
            )

        # this is kept for legacy resons in case we want to resuse sorting of these entries
        row_idx = idx[1, ...].contiguous()
        col_idx = idx[2, ...].contiguous()
        roff_idx = roff.contiguous()

        self.register_buffer("psi_row_idx", row_idx, persistent=False)
        self.register_buffer("psi_col_idx", col_idx, persistent=False)
        self.register_buffer("psi_roff_idx", roff_idx, persistent=False)

        # Contiguous-arc form of the same sparsity, consumed by the CUDA kernels: it
        # lets them derive a neighbour's column by counting instead of recovering it
        # from col_idx with a per-neighbour 64-bit integer division, which the GPU has
        # no instruction for. col_idx is kept because the CPU and torch reference paths
        # still use it -- which is what keeps the reference independent of this
        # derivation. See _build_psi_segments and TestPsiArcStructure.
        #
        # For the scatter (upsample) psi the rows are keyed by input latitude and the
        # columns index the output grid, so the decode width differs.
        nlon_decode = self.nlon_out if (self.nlat_out > self.nlat_in or self.nlon_out > self.nlon_in) else self.nlon_in
        psi_seg, psi_seg_off = _build_psi_segments(col_idx, roff_idx, nlon_decode)
        self.register_buffer("psi_seg", psi_seg, persistent=False)
        self.register_buffer("psi_seg_off", psi_seg_off, persistent=False)

        # learnable parameters — Xavier uniform init matching PyTorch MHA convention:
        # bound = sqrt(6 / (fan_in + fan_out)) for each projection
        if self.k_channels % self.num_heads != 0:
            raise ValueError(f"Please make sure that number of heads {self.num_heads} divides k_channels {self.k_channels} evenly.")
        if self.out_channels % self.num_heads != 0:
            raise ValueError(f"Please make sure that number of heads {self.num_heads} divides out_channels {self.out_channels} evenly.")
        scale_qk = math.sqrt(6.0 / (self.in_channels + self.k_channels))
        scale_v = math.sqrt(6.0 / (self.in_channels + self.out_channels))
        scale_proj = math.sqrt(3.0 / self.out_channels)
        self.q_weights = nn.Parameter(scale_qk * (2 * torch.rand(self.k_channels, self.in_channels, 1, 1) - 1))
        self.k_weights = nn.Parameter(scale_qk * (2 * torch.rand(self.k_channels, self.in_channels, 1, 1) - 1))
        self.v_weights = nn.Parameter(scale_v * (2 * torch.rand(self.out_channels, self.in_channels, 1, 1) - 1))
        self.proj_weights = nn.Parameter(scale_proj * (2 * torch.rand(self.out_channels, self.out_channels, 1, 1) - 1))

        if scale is not None:
            self.scale = scale
        else:
            self.scale = 1 / math.sqrt(self.k_channels // self.num_heads)

        if bias:
            self.q_bias = nn.Parameter(torch.zeros(self.k_channels))
            self.k_bias = nn.Parameter(torch.zeros(self.k_channels))
            self.v_bias = nn.Parameter(torch.zeros(self.out_channels))
            self.proj_bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
            self.proj_bias = None

        if use_qknorm:
            self.q_norm_weights = nn.Parameter(torch.zeros(self.k_channels // self.num_heads))
            self.k_norm_weights = nn.Parameter(torch.zeros(self.k_channels // self.num_heads))
        else:
            self.q_norm_weights = None
            self.k_norm_weights = None

        if self.optimized_kernel:
            self.attention_handle = _neighborhood_s2_attention_optimized
        else:
            self.attention_handle = _neighborhood_s2_attention_torch

    def extra_repr(self):
        return f"grid_in={self.grid_in!r},\ngrid_out={self.grid_out!r},\nin_channels={self.in_channels}, out_channels={self.out_channels}, k_channels={self.k_channels}, theta_cutoff={self.theta_cutoff}"

    def forward(self, query: torch.Tensor, key: Optional[torch.Tensor] = None, value: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply neighborhood attention on the sphere.

        Parameters
        ----------
        query : torch.Tensor
            Query signal of shape ``(batch, in_channels, nlat_out, nlon_out)`` (sampled on the output grid).
        key : torch.Tensor, optional
            Key signal of shape ``(batch, in_channels, nlat_in, nlon_in)`` (sampled on the input grid).
            Defaults to ``query`` (self-attention, which requires matching input and output grids).
        value : torch.Tensor, optional
            Value signal of shape ``(batch, in_channels, nlat_in, nlon_in)`` (sampled on the input grid).
            Defaults to ``query`` (self-attention, which requires matching input and output grids).

        Returns
        -------
        torch.Tensor
            Attention output of shape ``(batch, out_channels, nlat_out, nlon_out)``.
        """

        # self attention simplification
        if key is None:
            key = query

        if value is None:
            value = query

        # change this later to allow arbitrary number of batch dims
        _check_ndim(query, 4, "query")
        _check_ndim(key, 4, "key")
        _check_ndim(value, 4, "value")
        _check_dtypes_match((query, key, value))
        _check_extent(query, -2, self.nlat_out, "query latitudes")
        _check_extent(query, -1, self.nlon_out, "query longitudes")
        _check_extent(key, -2, self.nlat_in, "key latitudes")
        _check_extent(key, -1, self.nlon_in, "key longitudes")
        _check_extent(value, -2, self.nlat_in, "value latitudes")
        _check_extent(value, -1, self.nlon_in, "value longitudes")

        # Convert to NHWC once, here, and stay in it for the whole module. Every
        # projection is 1x1 (see __init__), so in NHWC it is a plain GEMM over a
        # contiguous reduction dimension rather than a convolution; qk-norm's head
        # split becomes a free view; and the attention op already takes NHWC with
        # heads packed along the channel dimension. The only conversions left are
        # the two the channels-first public API forces: inputs in, output out.
        #
        # The shape checks above index dims -2/-1 as (lat, lon), so they have to
        # run before this point.
        #
        # Self-attention binds all three names to one tensor (see the `key is None`
        # handling above), which needs one conversion rather than three. Identity
        # rather than equality: that is how the caller expresses it, and it cannot
        # false-positive. Note query can only alias key/value when in_shape ==
        # out_shape -- with resampling the extents differ, so there is nothing to
        # share and the general path is already the right one.
        # Aliasing has to be sampled before any rebinding, or the conversion of the
        # first tensor would make every later identity test false.
        key_is_query = key is query
        value_is_query = value is query
        value_is_key = value is key

        query = to_nhwc(query)
        key = query if key_is_query else to_nhwc(key)
        if value_is_query:
            value = query
        elif value_is_key:
            value = key
        else:
            value = to_nhwc(value)

        # perform QKV projections. The stored weights keep their (C_out, C_in, 1, 1)
        # convolution shape so checkpoints stay loadable; the view to (C_out, C_in)
        # is free.
        query = F.linear(query, self.q_weights.reshape(self.q_weights.shape[0], -1), self.q_bias)
        key = F.linear(key, self.k_weights.reshape(self.k_weights.shape[0], -1), self.k_bias)
        value = F.linear(value, self.v_weights.reshape(self.v_weights.shape[0], -1), self.v_bias)

        # perform QK normalization (must come before scale). In NHWC the channel
        # axis is innermost, so splitting it into (heads, per-head channels) is a
        # reshape of contiguous memory -- a view, not a copy. The channels-first
        # form of this needed a 5D permute in and another back out.
        if self.q_norm_weights is not None:
            B, H, W, C = query.shape
            query = query.reshape(B, H, W, self.num_heads, -1)
            query = F.rms_norm(query, normalized_shape=self.q_norm_weights.shape, weight=1 + self.q_norm_weights)
            query = query.reshape(B, H, W, C)

        if self.k_norm_weights is not None:
            B, H, W, C = key.shape
            key = key.reshape(B, H, W, self.num_heads, -1)
            key = F.rms_norm(key, normalized_shape=self.k_norm_weights.shape, weight=1 + self.k_norm_weights)
            key = key.reshape(B, H, W, C)

        # scale after normalization
        query_scaled = query * self.scale

        out = self.attention_handle(
            key,
            value,
            query_scaled,
            self.quad_weights,
            self.psi_col_idx,
            self.psi_roff_idx,
            self.psi_seg,
            self.psi_seg_off,
            self.num_heads,
            self.nlon_in,
            self.nlat_out,
            self.nlon_out,
        )

        # output projection stays in NHWC for the same reason as the input ones;
        # only then back to channels-first. The matching backward conversion is
        # generated by autograd and uses the same tiled kernel.
        out = F.linear(out, self.proj_weights.reshape(self.proj_weights.shape[0], -1), self.proj_bias)

        return to_nchw(out)
