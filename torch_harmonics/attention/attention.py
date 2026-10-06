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
import warnings
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_helpers import optimized_kernels_is_available

from torch_harmonics.attention._attention_utils import _check_dtypes_match, _check_extent, _check_ndim
from torch_harmonics.attention._layout import to_nchw, to_nhwc
from torch_harmonics.attention.backends import BACKENDS
from torch_harmonics.grid import GridS2, RegularGridS2, _rejects_legacy_signature, require_grid
from torch_harmonics.neighborhood import precompute_neighborhood_arcs_s2
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

    Either side may be any :class:`~torch_harmonics.grid.GridS2`, including ragged
    grids such as HEALPix, and the two sides need not be of the same kind. A field on a regular grid is
    ``(batch, channels, nlat, nlon)``, on a ragged one ``(batch, channels, npoints)``;
    in general its trailing shape is ``grid.shape``. On an equal-area input grid
    (:attr:`~torch_harmonics.grid.PointSetS2.is_equal_area`, e.g. HEALPix) the weights
    cancel in the softmax and no mask is applied.

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

    @_rejects_legacy_signature(
        'in_channels, num_heads, in_shape, out_shape, grid_in="equiangular", grid_out="equiangular", scale=None, '
        "use_qknorm=False, bias=True, k_channels=None, out_channels=None, drop_rate=0.0",
        grid_in="in_shape",
        grid_out="out_shape",
    )
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

        # Any ring grid, ragged ones included: global attention reads only the points
        # and their weights. It does not go down to a bare PointSetS2 because the mask
        # below is GridS2.point_weights, computed in float32 so that it matches an
        # already-trained model bit for bit; the point set's float64 quad_weights would
        # round differently.
        #
        # There is no longitude constraint either, unlike the neighbourhood layer:
        # every output point attends to every input point, so there is no p-shift that
        # would need to be exact.
        self.grid_in = require_grid(grid_in, "grid_in")
        self.grid_out = require_grid(grid_out, "grid_out")

        self.npoints_in = self.grid_in.npoints
        self.npoints_out = self.grid_out.npoints

        # nlat/nlon exist only on a regular grid; keep them where they are defined, as
        # the neighbourhood layer does, so that a ragged side has none to misuse
        if isinstance(self.grid_in, RegularGridS2):
            self.nlat_in, self.nlon_in = self.grid_in.shape
        if isinstance(self.grid_out, RegularGridS2):
            self.nlat_out, self.nlon_out = self.grid_out.shape

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.k_channels = in_channels if k_channels is None else k_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.drop_rate = drop_rate
        self.scale = scale

        # integration weights
        # global attention has no neighbourhood to index by ring, so only the expanded
        # per-point form is ever needed here
        #
        # On an equal-area input grid there is no mask at all. The weights enter only
        # through the softmax, as the additive log w below, and a constant added to every
        # logit of a row cancels in it -- so dropping the mask is exact, not an
        # approximation. It is also what lets SDPA take its fused FlashAttention kernel,
        # which accepts no mask; with one it falls back to a slower path. None rather
        # than an absent attribute, so forward passes it straight through.
        if self.grid_in.is_equal_area:
            log_point_weights = None
        else:
            # compute log because they are applied as an addition prior to the softmax ('attn_mask'), which includes an exponential.
            # see https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
            # for info on how 'attn_mask' is applied to the attention weights
            point_weights = self.grid_in.point_weights(torch.float32)
            log_point_weights = torch.log(point_weights).reshape(1, 1, -1)
        self.register_buffer("log_point_weights", log_point_weights, persistent=False)

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

    def _check_inputs(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """
        Shape and dtype contract of the three inputs: ``(batch, channels, *grid.shape)``,
        query on the output grid and key/value on the input grid, each checked against
        its own grid since a mixed pair gives the two sides different ranks.
        """
        for tensor, name, grid in ((query, "query", self.grid_out), (key, "key", self.grid_in), (value, "value", self.grid_in)):
            _check_ndim(tensor, 2 + len(grid.shape), name)
            for axis, extent in enumerate(grid.shape, start=-len(grid.shape)):
                _check_extent(tensor, axis, extent, f"{name} spatial axis {axis}")
        _check_dtypes_match((query, key, value))

    @staticmethod
    def _to_points_last(tensor: torch.Tensor, grid: GridS2) -> torch.Tensor:
        """``(batch, channels, *grid.shape)`` to ``(batch, npoints, channels)``, contiguous."""
        # the type, not is_regular: only a RegularGridS2 lays a field out as (nlat, nlon); a
        # GridS2 whose rings happen to be equal is still stored flat
        if isinstance(grid, RegularGridS2):
            # the tiled NHWC kernel, then merging (nlat, nlon), which is a free view
            return to_nhwc(tensor).flatten(1, 2)
        # a ragged field has a single spatial axis, for which a transpose is the whole of it
        return tensor.transpose(1, 2).contiguous()

    @staticmethod
    def _to_channels_first(tensor: torch.Tensor, grid: GridS2) -> torch.Tensor:
        """Inverse of :meth:`_to_points_last`."""
        if isinstance(grid, RegularGridS2):
            return to_nchw(tensor.unflatten(1, grid.shape))
        return tensor.transpose(1, 2).contiguous()

    def forward(self, query: torch.Tensor, key: Optional[torch.Tensor] = None, value: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply global attention on the sphere.

        Parameters
        ----------
        query : torch.Tensor
            Query signal sampled on the output grid: ``(batch, in_channels, nlat_out, nlon_out)``
            on a :class:`~torch_harmonics.grid.RegularGridS2`, ``(batch, in_channels, npoints_out)``
            on any other :class:`~torch_harmonics.grid.GridS2` such as HEALPix -- in general
            ``(batch, in_channels, *grid_out.shape)``.
        key : torch.Tensor, optional
            Key signal sampled on the input grid: ``(batch, in_channels, nlat_in, nlon_in)``
            on a :class:`~torch_harmonics.grid.RegularGridS2`, ``(batch, in_channels, npoints_in)``
            on any other :class:`~torch_harmonics.grid.GridS2` -- in general
            ``(batch, in_channels, *grid_in.shape)``.
            Defaults to ``query`` (self-attention).
        value : torch.Tensor, optional
            Value signal sampled on the input grid: ``(batch, in_channels, nlat_in, nlon_in)``
            on a :class:`~torch_harmonics.grid.RegularGridS2`, ``(batch, in_channels, npoints_in)``
            on any other :class:`~torch_harmonics.grid.GridS2` -- in general
            ``(batch, in_channels, *grid_in.shape)``.
            Defaults to ``query`` (self-attention).

        Returns
        -------
        torch.Tensor
            Attention output on the output grid: ``(batch, out_channels, nlat_out, nlon_out)`` on a
            :class:`~torch_harmonics.grid.RegularGridS2`, ``(batch, out_channels, npoints_out)`` on any
            other :class:`~torch_harmonics.grid.GridS2` -- in general
            ``(batch, out_channels, *grid_out.shape)``.
        """

        # self attention simplification
        if key is None:
            key = query

        if value is None:
            value = query

        self._check_inputs(query, key, value)

        # Channels-last, as in NeighborhoodAttentionS2: converted once on the way in and
        # once on the way out, and in between every projection is a plain GEMM over a
        # contiguous channel axis. Self-attention binds all three names to one tensor,
        # which needs one conversion rather than three; aliasing is sampled before any
        # rebinding, or converting the first would make every later identity test false.
        key_is_query = key is query
        value_is_query = value is query
        value_is_key = value is key

        query = self._to_points_last(query, self.grid_out)
        key = query if key_is_query else self._to_points_last(key, self.grid_in)
        if value_is_query:
            value = query
        elif value_is_key:
            value = key
        else:
            value = self._to_points_last(value, self.grid_in)

        # QKV projections. The stored weights keep their (C_out, C_in, 1, 1) convolution
        # shape so checkpoints stay loadable; the view to (C_out, C_in) is free.
        query = F.linear(query, self.q_weights.reshape(self.q_weights.shape[0], -1), self.q_bias)
        key = F.linear(key, self.k_weights.reshape(self.k_weights.shape[0], -1), self.k_bias)
        value = F.linear(value, self.v_weights.reshape(self.v_weights.shape[0], -1), self.v_bias)

        # (batch, npoints, heads * C) -> (batch, heads, npoints, C), which is what SDPA
        # takes. A view: the channel axis stays innermost and contiguous, which is all
        # the fused SDPA kernels ask of the other strides.
        query = query.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        key = key.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        value = value.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)

        if self.q_norm_weights is not None:
            query = F.rms_norm(query, normalized_shape=self.q_norm_weights.shape, weight=1 + self.q_norm_weights)
        if self.k_norm_weights is not None:
            key = F.rms_norm(key, normalized_shape=self.k_norm_weights.shape, weight=1 + self.k_norm_weights)

        # apply scale — if scale is a tensor (e.g. learnable), multiply into query
        # directly since SDPA only accepts a float scale
        dropout_p = self.drop_rate if self.training else 0.0
        if isinstance(self.scale, torch.Tensor):
            query = query * self.scale
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=self.log_point_weights, dropout_p=dropout_p, scale=1.0)
        else:
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=self.log_point_weights, dropout_p=dropout_p, scale=self.scale)

        # (batch, heads, npoints, C) -> (batch, npoints, heads * C), heads outermost in the
        # channel axis as they were split
        out = out.transpose(1, 2).flatten(2)

        # output projection stays channels-last for the same reason as the input ones;
        # only then back to channels-first
        out = F.linear(out, self.proj_weights.reshape(self.proj_weights.shape[0], -1), self.proj_bias)

        return self._to_channels_first(out, self.grid_out)


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

    The neighborhood of each output point is the geodesic disk of radius
    :math:`\theta_\mathrm{cutoff}` around it, precomputed as contiguous longitude
    arcs (:func:`~torch_harmonics.neighborhood.precompute_neighborhood_arcs_s2`),
    so that an input point contributes to an output location exactly when it lies
    within :math:`\theta_\mathrm{cutoff}` of it. The relative weight of each input point
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

    #: The implementations this layer chooses from, in order; see
    #: :data:`.backends.BACKENDS`. A subclass that computes differently brings its own
    #: list -- DistributedNeighborhoodAttentionS2 lists the ring backends -- and inherits
    #: the selection, the device handling and the forward pass unchanged.
    _backends = BACKENDS

    @_rejects_legacy_signature(
        'in_channels, in_shape, out_shape, grid_in="equiangular", grid_out="equiangular", num_heads=1, scale=None, '
        "use_qknorm=False, bias=True, theta_cutoff=None, k_channels=None, out_channels=None, optimized_kernel=True",
        grid_in="in_shape",
        grid_out="out_shape",
    )
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

        # a ring grid, not necessarily a regular one: the ragged path below serves
        # HEALPix and other grids whose rings differ in length. This is the first of the
        # require_regular_grid guards to come back down, which is what they were put in
        # for -- one relaxed per backend that gains support, rather than all at once.
        self.grid_in = require_grid(grid_in, "grid_in")
        self.grid_out = require_grid(grid_out, "grid_out")

        # Raggedness is a property of each side on its own, and the two roles it plays
        # are separable. self.ragged picks the computation, and one ragged grid is
        # enough to force it for both sides, because the ragged path is the general one
        # and a regular grid is a case it admits. ragged_in and ragged_out pick only the
        # layout each side shows the caller, which is what lets a HEALPix field attend
        # onto a lat/lon one and come back shaped like a lat/lon field.
        # decided by type: only a RegularGridS2 has the (nlat, nlon) layout and uniform stride
        # the regular kernels assume, whatever its ring lengths happen to be
        self.ragged_in = not isinstance(self.grid_in, RegularGridS2)
        self.ragged_out = not isinstance(self.grid_out, RegularGridS2)
        self.ragged = self.ragged_in or self.ragged_out

        self.npoints_in = self.grid_in.npoints
        self.npoints_out = self.grid_out.npoints

        if self.ragged:
            # nlat/nlon exist only on a regular grid; keep them where they are defined so
            # that a consumer reaching for them on a ragged side fails rather than
            # silently taking the widest ring for a stride
            if not self.ragged_in:
                self.nlat_in, self.nlon_in = self.grid_in.shape
            if not self.ragged_out:
                self.nlat_out, self.nlon_out = self.grid_out.shape
            # there is no p-shift to be exact on a ragged grid, so direction is decided
            # by point count alone
            self.upsample = self.npoints_out > self.npoints_in
        else:
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
        # what was asked for, kept apart from what the build allows, so that landing on a
        # reference backend despite asking for the kernels can be reported
        self._optimized_kernel_requested = bool(optimized_kernel)
        self.optimized_kernel = optimized_kernel and optimized_kernels_is_available()

        # The coarser of the two grids sets the default support, judged by point count on
        # both paths, so that the default operator depends on the geometry and not on the
        # path or the direction chosen to compute it. The direction alone would not do:
        # the regular path reads it off the longitude counts, so a latitude-only upsample
        # such as (6, 12) -> (12, 12) takes the gather direction there while its input grid
        # is the coarser one. On equal point counts neither grid is coarser by this
        # measure, and the direction decides as before.
        if self.npoints_in != self.npoints_out:
            coarser = self.grid_in if self.npoints_in < self.npoints_out else self.grid_out
        else:
            coarser = self.grid_in if self.upsample else self.grid_out
        self.theta_cutoff = truncate_support(coarser, theta_cutoff)

        # The neighbourhood pattern is all attention needs: which input points lie
        # within theta_cutoff of each output point. It used to come from the DISCO
        # precompute with a fabricated one-function Zernike basis whose values were
        # thrown away, which cost roughly a third of the setup to produce nothing --
        # and forced the arc form to be recovered afterwards from the column list.
        #
        # fold_longitude keys the pattern by output ring rather than by output point,
        # which is what makes it the same size as DISCO's: on a regular grid shifting
        # the output longitude carries a point's neighbourhood onto the next point's,
        # so one row per ring suffices and the kernels shift. Without it the pattern
        # would be nlon times larger -- 8.8 million arcs instead of 8,632 at 512x1024.
        #
        # For upsample on a regular grid the grids swap, mirroring DISCO's transpose
        # module: rows then index the smaller input grid and columns encode the larger
        # output grid. That trick is the p-shift again, so the ragged path does not take
        # it -- there the pattern always runs input to output and the kernel reads it in
        # whichever direction it needs.
        if self.ragged:
            src, dst = self.grid_in, self.grid_out
        else:
            src, dst = (self.grid_out, self.grid_in) if self.upsample else (self.grid_in, self.grid_out)

        # the pattern itself is built by _neighborhood_arcs(), when a backend asks for it
        self._arcs_src, self._arcs_dst = src, dst

        # Every tensor derived from the neighbourhood belongs to a backend, which
        # registers exactly what it reads -- see backends.py. Selection happens last,
        # after the parameters exist, because a backend is handed the layer; and it
        # happens again on every device change, via _apply.
        self._backend_state = ()
        self.backend = None

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

        # last, so that a backend handed this layer finds it fully built
        self._setup()
        self._select_backend()

    def extra_repr(self):
        return f"grid_in={self.grid_in!r},\ngrid_out={self.grid_out!r},\nin_channels={self.in_channels}, out_channels={self.out_channels}, k_channels={self.k_channels}, theta_cutoff={self.theta_cutoff}"

    @property
    def device(self) -> torch.device:
        """
        Device of the module's parameters.
        """
        return self.q_weights.device

    @property
    def dtype(self) -> torch.dtype:
        """
        Dtype of the module's parameters.

        The inputs share it, since the projections reject any other.
        """
        return self.q_weights.dtype

    def _neighborhood_arcs(self):
        """The neighbourhood in arc form. Cached by the precompute, so backends share it."""
        # One call for both families; the only difference is whether the longitude axis
        # can be folded away. It can exactly when the grids are regular, which is what
        # fold_longitude checks -- so `not self.ragged` is not a shortcut here, it is the
        # same condition stated once.
        return precompute_neighborhood_arcs_s2(self._arcs_src, self._arcs_dst, theta_cutoff=self.theta_cutoff, fold_longitude=not self.ragged)

    def _setup(self) -> None:
        """
        Hook for a subclass to finish its own setup before a backend is selected.

        Selection is the last step of ``__init__``, and a backend's prepare() reads the
        layer, so anything a subclass's backends need has to exist by then -- which is
        before the subclass's own ``__init__`` would resume after ``super().__init__()``.
        Nothing to do here; the distributed layer settles its shards and halo.
        """
        pass

    def _select_backend(self) -> None:
        """
        Pick the backend for the current device and register exactly its state.

        The previous backend's buffers are removed first, so the module carries one
        backend's tensors and never a union of them.
        """
        device = self.device
        backend = next((b for b in self._backends if b.available(self, device)), None)
        if backend is None:
            raise RuntimeError(f"no attention backend serves {type(self.grid_in).__name__} -> {type(self.grid_out).__name__} on {device}")
        backend = backend()

        for name in self._backend_state:
            delattr(self, name)

        state = backend.prepare(self, device)
        for name, tensor in state.items():
            self.register_buffer(name, tensor, persistent=False)

        self._backend_state = tuple(state)
        self.backend = backend

        if backend.reference and self._optimized_kernel_requested:
            if not optimized_kernels_is_available():
                reason = "torch_harmonics was built without the compiled attention kernels"
            elif self.dtype == torch.float64:
                reason = "the compiled attention kernels compute in float32, and the layer is float64"
            else:
                reason = f"this build of the compiled attention kernels has no {device.type} implementation"
            warnings.warn(
                f"{type(self).__name__} on {device} falls back to the torch reference implementation ({backend.name}), because {reason}. "
                "It is considerably slower. Pass optimized_kernel=False to select the reference explicitly and silence this warning."
            )

    def _apply(self, fn, recurse: bool = True):
        """
        Reselect the backend when the module changes device, and restore its state when a
        dtype change has cast it.

        ``_apply`` rather than ``to``: ``.cuda()``, ``.cpu()``, ``.half()`` and
        ``.double()`` never call ``to``, so it is the only hook that sees every move.

        A dtype change casts every floating buffer, backend state included, but that state
        has a dtype its backend fixes: the quadrature weights are float32 for the kernels,
        which read them as such, and for the references float32 too unless the layer is
        float64. A dtype change can also change the backend, since a float64 layer takes
        the reference. So a dtype change is answered by selecting and preparing again, just
        as a device move is -- which is also what keeps ``.half()`` from handing the
        kernels a 16-bit buffer they would read as ``float``. The test is on what actually
        changed, the device, the layer's dtype or the state's dtypes, not on the call.

        The state of the outgoing backend is moved or cast by ``super()._apply`` and then
        thrown away -- a few MB, once per move, against not having to know the target
        device or dtype before anything has been touched.
        """
        device_before, dtype_before = self.device, self.dtype
        dtypes_before = {name: getattr(self, name).dtype for name in self._backend_state}
        out = super()._apply(fn, recurse)
        state_cast = any(getattr(self, name).dtype != dtype for name, dtype in dtypes_before.items())
        # the layer's own dtype decides the backend too (float64 takes the reference), and
        # can change without casting the state: a reference backend's state may already
        # be float32 when .float() brings the layer back from float64
        if self.device != device_before or self.dtype != dtype_before or state_cast:
            self._select_backend()
        return out

    def _check_inputs(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """
        Shape and dtype contract of the three inputs, for whichever grid kind this is.

        query is sampled on the output grid and key/value on the input grid, so a mixed
        pair gives the two sides different ranks; each is checked against its own grid
        rather than against one rank for the layer.

        The branches are on Python bools fixed at construction, so dynamo specialises on
        them rather than breaking the graph, and the checks themselves are the same
        traceable helpers the regular path has always used.
        """
        _check_ndim(query, 3 if self.ragged_out else 4, "query")
        for tensor, name in ((key, "key"), (value, "value")):
            _check_ndim(tensor, 3 if self.ragged_in else 4, name)
        _check_dtypes_match((query, key, value))

        if self.ragged_out:
            _check_extent(query, -1, self.npoints_out, "query points")
        else:
            _check_extent(query, -2, self.nlat_out, "query latitudes")
            _check_extent(query, -1, self.nlon_out, "query longitudes")

        if self.ragged_in:
            _check_extent(key, -1, self.npoints_in, "key points")
            _check_extent(value, -1, self.npoints_in, "value points")
        else:
            _check_extent(key, -2, self.nlat_in, "key latitudes")
            _check_extent(key, -1, self.nlon_in, "key longitudes")
            _check_extent(value, -2, self.nlat_in, "value latitudes")
            _check_extent(value, -1, self.nlon_in, "value longitudes")

    def _to_channels_last(self, tensor: torch.Tensor, ragged_layout: bool) -> torch.Tensor:
        """
        ``(batch, channels, *spatial)`` to ``(batch, points, channels)``.

        ``ragged_layout`` says whether *this tensor's* grid is ragged, which is not the
        same question as whether the computation is: with a mixed pair the query and the
        key/value sides answer it differently.
        """
        # to_nhwc is a tiled kernel for the two-axis case; a ragged field has a single
        # spatial axis, for which a transpose is the whole of it
        if not self.ragged:
            return to_nhwc(tensor)
        if not ragged_layout:
            # A regular grid entering the ragged path. Its two spatial axes collapse into
            # the one flat axis the neighbourhood indexes, and ring-major order is already
            # that flat order: lon_offsets places (ilat, ilon) at ilat * nlon + ilon on a
            # regular grid, which is exactly what this reshape gives.
            tensor = tensor.flatten(-2, -1)
        return tensor.transpose(1, 2).contiguous()

    def _to_channels_first(self, tensor: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`_to_channels_last`, on the output grid."""
        if not self.ragged:
            return to_nchw(tensor)
        tensor = tensor.transpose(1, 2).contiguous()
        if not self.ragged_out:
            # restore the two spatial axes the caller handed in
            tensor = tensor.unflatten(-1, (self.nlat_out, self.nlon_out))
        return tensor

    def forward(self, query: torch.Tensor, key: Optional[torch.Tensor] = None, value: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply neighborhood attention on the sphere.

        Parameters
        ----------
        query : torch.Tensor
            Query signal sampled on the output grid: ``(batch, in_channels, nlat_out, nlon_out)``
            on a :class:`~torch_harmonics.grid.RegularGridS2`, ``(batch, in_channels, npoints_out)``
            on any other :class:`~torch_harmonics.grid.GridS2` such as HEALPix -- in general
            ``(batch, in_channels, *grid_out.shape)``.
        key : torch.Tensor, optional
            Key signal sampled on the input grid: ``(batch, in_channels, nlat_in, nlon_in)``
            on a :class:`~torch_harmonics.grid.RegularGridS2`, ``(batch, in_channels, npoints_in)``
            on any other :class:`~torch_harmonics.grid.GridS2` -- in general
            ``(batch, in_channels, *grid_in.shape)``.
            Defaults to ``query`` (self-attention, which requires matching input and output grids).
        value : torch.Tensor, optional
            Value signal sampled on the input grid: ``(batch, in_channels, nlat_in, nlon_in)``
            on a :class:`~torch_harmonics.grid.RegularGridS2`, ``(batch, in_channels, npoints_in)``
            on any other :class:`~torch_harmonics.grid.GridS2` -- in general
            ``(batch, in_channels, *grid_in.shape)``.
            Defaults to ``query`` (self-attention, which requires matching input and output grids).

        Returns
        -------
        torch.Tensor
            Attention output on the output grid: ``(batch, out_channels, nlat_out, nlon_out)`` on a
            :class:`~torch_harmonics.grid.RegularGridS2`, ``(batch, out_channels, npoints_out)`` on any
            other :class:`~torch_harmonics.grid.GridS2` -- in general
            ``(batch, out_channels, *grid_out.shape)``.
        """

        # self attention simplification
        if key is None:
            key = query

        if value is None:
            value = query

        self._check_inputs(query, key, value)

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

        query = self._to_channels_last(query, self.ragged_out)
        key = query if key_is_query else self._to_channels_last(key, self.ragged_in)
        if value_is_query:
            value = query
        elif value_is_key:
            value = key
        else:
            value = self._to_channels_last(value, self.ragged_in)

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
            # splitting the channel axis into (heads, per-head) is a view in either
            # rank, so the spatial axes are carried through rather than named
            shape = query.shape
            query = query.reshape(*shape[:-1], self.num_heads, -1)
            query = F.rms_norm(query, normalized_shape=self.q_norm_weights.shape, weight=1 + self.q_norm_weights)
            query = query.reshape(shape)

        if self.k_norm_weights is not None:
            # splitting the channel axis into (heads, per-head) is a view in either
            # rank, so the spatial axes are carried through rather than named
            shape = key.shape
            key = key.reshape(*shape[:-1], self.num_heads, -1)
            key = F.rms_norm(key, normalized_shape=self.k_norm_weights.shape, weight=1 + self.k_norm_weights)
            key = key.reshape(shape)

        # scale after normalization
        query_scaled = query * self.scale

        # channels-last in and out, heads packed along the channel dimension; the
        # backend is fixed before tracing, see backends.py
        out = self.backend(self, key, value, query_scaled)

        # output projection stays in NHWC for the same reason as the input ones;
        # only then back to channels-first. The matching backward conversion is
        # generated by autograd and uses the same tiled kernel.
        out = F.linear(out, self.proj_weights.reshape(self.proj_weights.shape[0], -1), self.proj_bias)

        return self._to_channels_first(out)
