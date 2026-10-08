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

import torch
import torch.amp as amp
import torch.nn as nn

from torch_harmonics import AttentionS2, DiscreteContinuousConvS2, InverseRealSHT, NeighborhoodAttentionS2, RealSHT, ResampleS2
from torch_harmonics.examples.models._layers import MLP, DropPath, LayerNorm, LearnablePositionEmbedding, SequencePositionEmbedding, SpectralPositionEmbedding
from torch_harmonics.grid import require_regular_grid


# heuristic for finding theta_cutoff
def _compute_cutoff_radius(nlat, kernel_shape, basis_type):
    # "morlet" is the deprecated unnormalized alias of "harmonic", with the same support
    theta_cutoff_factor = {"piecewise linear": 0.5, "harmonic": 0.5, "morlet": 0.5, "zernike": math.sqrt(2.0)}

    return (kernel_shape[0] + 1) * theta_cutoff_factor[basis_type] * math.pi / float(nlat - 1)


class DiscreteContinuousEncoder(nn.Module):
    """
    Discrete-continuous encoder for spherical transformers.

    This module performs downsampling using discrete-continuous convolutions on the sphere,
    reducing the spatial resolution while maintaining the spectral properties of the data.

    Parameters
    ----------
    grid_in : RegularGridS2
        Grid of the input field.
    grid_out : RegularGridS2
        Grid of the output field.
    in_chans : int, optional
        Number of input channels, by default 2
    out_chans : int, optional
        Number of output channels, by default 2
    kernel_shape : tuple, optional
        Kernel shape for convolution, by default (3, 3)
    basis_type : str, optional
        Filter basis type, by default "harmonic"
    groups : int, optional
        Number of groups for grouped convolution, by default 1
    bias : bool, optional
        Whether to use bias, by default False
    """

    def __init__(
        self,
        grid_in,
        grid_out,
        in_chans=2,
        out_chans=2,
        kernel_shape=(3, 3),
        basis_type="harmonic",
        groups=1,
        bias=False,
    ):
        super().__init__()

        # set up local convolution
        self.conv = DiscreteContinuousConvS2(
            grid_in,
            grid_out,
            in_chans,
            out_chans,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            groups=groups,
            bias=bias,
            theta_cutoff=_compute_cutoff_radius(grid_in.nlat, kernel_shape, basis_type),
        )

    def forward(self, x):

        dtype = x.dtype

        with amp.autocast(device_type="cuda", enabled=False):
            x = x.float()
            x = self.conv(x)
            x = x.to(dtype=dtype)

        return x


class DiscreteContinuousDecoder(nn.Module):
    """
    Discrete-continuous decoder for spherical transformers.

    This module performs upsampling using either spherical harmonic transforms or resampling,
    followed by discrete-continuous convolutions to restore spatial resolution.

    Parameters
    ----------
    grid_in : RegularGridS2
        Grid of the input field.
    grid_out : RegularGridS2
        Grid of the output field.
    in_chans : int, optional
        Number of input channels, by default 2
    out_chans : int, optional
        Number of output channels, by default 2
    kernel_shape : tuple, optional
        Kernel shape for convolution, by default (3, 3)
    basis_type : str, optional
        Filter basis type, by default "harmonic"
    groups : int, optional
        Number of groups for grouped convolution, by default 1
    bias : bool, optional
        Whether to use bias, by default False
    upsample_sht : bool, optional
        Whether to use SHT for upsampling, by default False
    """

    def __init__(
        self,
        grid_in,
        grid_out,
        in_chans=2,
        out_chans=2,
        kernel_shape=(3, 3),
        basis_type="harmonic",
        groups=1,
        bias=False,
        upsample_sht=False,
    ):
        super().__init__()

        # set up upsampling
        if upsample_sht:
            self.sht = RealSHT(grid_in).float()
            self.isht = InverseRealSHT(grid_out, lmax=self.sht.lmax, mmax=self.sht.mmax).float()
            self.upsample = nn.Sequential(self.sht, self.isht)
        else:
            self.upsample = ResampleS2(grid_in, grid_out)

        # set up DISCO convolution
        self.conv = DiscreteContinuousConvS2(
            grid_out,
            grid_out,
            in_chans,
            out_chans,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            groups=groups,
            bias=False,
            theta_cutoff=_compute_cutoff_radius(grid_in.nlat, kernel_shape, basis_type),
        )

    def forward(self, x):

        dtype = x.dtype

        with amp.autocast(device_type="cuda", enabled=False):
            x = x.float()
            x = self.upsample(x)
            x = self.conv(x)
            x = x.to(dtype=dtype)

        return x


class SphericalAttentionBlock(nn.Module):
    """
    Spherical attention block for transformers on the sphere.

    This module implements a single attention block that can use either global attention
    or neighborhood attention on spherical data, followed by an optional MLP.

    Parameters
    ----------
    grid_in : RegularGridS2
        Grid of the input field.
    grid_out : RegularGridS2
        Grid of the output field.
    in_chans : int, optional
        Number of input channels, by default 2
    out_chans : int, optional
        Number of output channels, by default 2
    num_heads : int, optional
        Number of attention heads, by default 1
    mlp_ratio : float, optional
        Ratio of MLP hidden dimension to output dimension, by default 2.0
    drop_rate : float, optional
        Dropout rate, by default 0.0
    drop_path : float, optional
        Drop path rate, by default 0.0
    act_layer : nn.Module, optional
        Activation layer, by default nn.GELU
    norm_layer : str, optional
        Normalization layer type, by default "none"
    use_mlp : bool, optional
        Whether to use MLP after attention, by default True
    bias : bool, optional
        Whether to use bias, by default False
    attention_mode : str, optional
        Attention mode ("neighborhood" or "global"), by default "neighborhood"
    theta_cutoff : float, optional
        Cutoff radius for neighborhood attention, by default None
    """

    def __init__(
        self,
        grid_in,
        grid_out,
        in_chans=2,
        out_chans=2,
        num_heads=1,
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer="none",
        use_mlp=True,
        bias=False,
        attention_mode="neighborhood",
        theta_cutoff=None,
    ):
        super().__init__()

        # normalisation layer
        if norm_layer == "layer_norm":
            self.norm0 = LayerNorm(in_channels=in_chans, eps=1e-6)
            self.norm1 = LayerNorm(in_channels=out_chans, eps=1e-6)
        elif norm_layer == "instance_norm":
            self.norm0 = nn.InstanceNorm2d(num_features=in_chans, eps=1e-6, affine=True, track_running_stats=False)
            self.norm1 = nn.InstanceNorm2d(num_features=out_chans, eps=1e-6, affine=True, track_running_stats=False)
        elif norm_layer == "none":
            self.norm0 = nn.Identity()
            self.norm1 = nn.Identity()
        else:
            raise NotImplementedError(f"Error, normalization {norm_layer} not implemented.")

        # determine radius for neighborhood attention
        self.attention_mode = attention_mode
        if attention_mode == "neighborhood":
            if theta_cutoff is None:
                theta_cutoff = (7.0 / math.sqrt(math.pi)) * math.pi / (grid_in.nlat - 1)
            self.self_attn = NeighborhoodAttentionS2(
                grid_in=grid_in,
                grid_out=grid_out,
                in_channels=in_chans,
                num_heads=num_heads,
                bias=bias,
                theta_cutoff=theta_cutoff,
                k_channels=None,
                out_channels=out_chans,
            )
        else:
            self.self_attn = AttentionS2(
                grid_in=grid_in,
                grid_out=grid_out,
                in_channels=in_chans,
                num_heads=num_heads,
                bias=bias,
                out_channels=out_chans,
                drop_rate=drop_rate,
            )

        self.skip0 = nn.Identity()

        # dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if use_mlp == True:
            mlp_hidden_dim = int(out_chans * mlp_ratio)
            self.mlp = MLP(
                in_features=out_chans,
                out_features=out_chans,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop_rate=drop_rate,
                checkpointing=False,
                gain=0.5,
            )

        self.skip1 = nn.Identity()

    def forward(self, x):

        residual = x

        x = self.norm0(x)

        if self.attention_mode == "neighborhood":
            dtype = x.dtype
            with amp.autocast(device_type="cuda", enabled=False):
                x = x.float()
                x = self.self_attn(x).to(dtype=dtype)
        else:
            x = self.self_attn(x)

        if hasattr(self, "skip0"):
            x = x + self.skip0(residual)

        residual = x

        x = self.norm1(x)

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        x = self.drop_path(x)

        if hasattr(self, "skip1"):
            x = x + self.skip1(residual)

        return x


class SphericalTransformer(nn.Module):
    """
    Spherical transformer model designed to approximate mappings from spherical signals to spherical signals

    Parameters
    ----------
    grid : RegularGridS2
        Grid the input and output fields live on, e.g.
        ``as_grid("equiangular", nlat=128, nlon=256)``.
    grid_internal : RegularGridS2
        Grid the attention blocks work on, usually coarser than ``grid``, e.g.
        ``as_grid("legendre-gauss", nlat=43, nlon=86)``. The DISCO encoder maps
        ``grid`` onto it and the decoder back.
    in_chans : int, optional
        Number of input channels, by default 3
    out_chans : int, optional
        Number of output channels, by default 3
    embed_dim : int, optional
        Dimension of the embeddings, by default 256
    num_layers : int, optional
        Number of layers in the network, by default 4
    activation_function : str, optional
        Activation function to use, by default "gelu"
    encoder_kernel_shape : tuple, optional
        Kernel shape for encoder, by default (3, 3)
    filter_basis_type : str, optional
        Filter basis type, by default "harmonic"
    num_heads : int, optional
        Number of attention heads, by default 1
    use_mlp : bool, optional
        Whether to use MLPs in the SFNO blocks, by default True
    mlp_ratio : float, optional
        Ratio of MLP to use, by default 2.0
    drop_rate : float, optional
        Dropout rate, by default 0.0
    drop_path_rate : float, optional
        Dropout path rate, by default 0.0
    normalization_layer : str, optional
        Type of normalization layer to use ("layer_norm", "instance_norm", "none"), by default "none"
    hard_thresholding_fraction : float, optional
        Fraction of hard thresholding (frequency cutoff) to apply, by default 1.0
    residual_prediction : bool, optional
        Whether to add a single large skip connection, by default False
    pos_embed : str, optional
        Type of positional embedding to use, by default "spectral"
    upsample_sht : bool, optional
        Use SHT upsampling if true, else linear interpolation, by default False
    attention_mode : str, optional
        Attention mode ("neighborhood" or "global"), by default "neighborhood"
    bias : bool, optional
        Whether to use bias, by default False
    theta_cutoff : float, optional
        Cutoff radius for neighborhood attention, by default None

    Examples
    --------
    >>> from torch_harmonics import as_grid
    >>> model = SphericalTransformer(
    ...         grid=as_grid("equiangular", nlat=128, nlon=256),
    ...         grid_internal=as_grid("legendre-gauss", nlat=32, nlon=64),
    ...         in_chans=2,
    ...         out_chans=2,
    ...         embed_dim=16,
    ...         num_layers=4,
    ...         use_mlp=True,)
    >>> model(torch.randn(1, 2, 128, 256)).shape
    torch.Size([1, 2, 128, 256])
    """

    def __init__(
        self,
        grid,
        grid_internal,
        in_chans=3,
        out_chans=3,
        embed_dim=256,
        num_layers=4,
        activation_function="gelu",
        encoder_kernel_shape=(3, 3),
        filter_basis_type="harmonic",
        num_heads=1,
        use_mlp=True,
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        normalization_layer="none",
        hard_thresholding_fraction=1.0,
        residual_prediction=False,
        pos_embed="spectral",
        upsample_sht=False,
        attention_mode="neighborhood",
        bias=False,
        theta_cutoff=None,
    ):
        super().__init__()
        self.grid = require_regular_grid(grid, "grid")
        self.grid_internal = require_regular_grid(grid_internal, "grid_internal")
        self.img_size = self.grid.shape
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.encoder_kernel_shape = encoder_kernel_shape
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.normalization_layer = normalization_layer
        self.use_mlp = use_mlp
        self.residual_prediction = residual_prediction

        # activation function
        if activation_function == "relu":
            self.activation_function = nn.ReLU
        elif activation_function == "gelu":
            self.activation_function = nn.GELU
        # for debugging purposes
        elif activation_function == "identity":
            self.activation_function = nn.Identity
        else:
            raise ValueError(f"Unknown activation function {activation_function}")

        self.h, self.w = self.grid_internal.shape

        # dropout
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0.0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]

        if pos_embed == "sequence":
            self.pos_embed = SequencePositionEmbedding(self.grid_internal, num_chans=self.embed_dim)
        elif pos_embed == "spectral":
            self.pos_embed = SpectralPositionEmbedding(self.grid_internal, num_chans=self.embed_dim)
        elif pos_embed == "learnable lat":
            self.pos_embed = LearnablePositionEmbedding(self.grid_internal, num_chans=self.embed_dim, embed_type="lat")
        elif pos_embed == "learnable latlon":
            self.pos_embed = LearnablePositionEmbedding(self.grid_internal, num_chans=self.embed_dim, embed_type="latlon")
        elif pos_embed == "none":
            self.pos_embed = nn.Identity()
        else:
            raise ValueError(f"Unknown position embedding type {pos_embed}")

        # maybe keep for now becuase tr
        # encoder
        self.encoder = DiscreteContinuousEncoder(
            grid_in=self.grid,
            grid_out=self.grid_internal,
            in_chans=self.in_chans,
            out_chans=self.embed_dim,
            kernel_shape=self.encoder_kernel_shape,
            basis_type=filter_basis_type,
            groups=1,
            bias=False,
        )

        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):
            block = SphericalAttentionBlock(
                grid_in=self.grid_internal,
                grid_out=self.grid_internal,
                in_chans=self.embed_dim,
                out_chans=self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path=dpr[i],
                act_layer=self.activation_function,
                norm_layer=self.normalization_layer,
                use_mlp=use_mlp,
                bias=bias,
                attention_mode=attention_mode,
                theta_cutoff=theta_cutoff,
            )

            self.blocks.append(block)

        # decoder
        self.decoder = DiscreteContinuousDecoder(
            grid_in=self.grid_internal,
            grid_out=self.grid,
            in_chans=self.embed_dim,
            out_chans=self.out_chans,
            kernel_shape=self.encoder_kernel_shape,
            basis_type=filter_basis_type,
            groups=1,
            bias=False,
            upsample_sht=upsample_sht,
        )

    @torch.compiler.disable
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x):
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):

        if self.residual_prediction:
            residual = x

        x = self.encoder(x)

        if self.pos_embed is not None:
            # x = x + self.pos_embed
            x = self.pos_embed(x)

        x = self.forward_features(x)

        x = self.decoder(x)

        if self.residual_prediction:
            x = x + residual

        return x
