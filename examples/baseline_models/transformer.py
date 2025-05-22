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
import torch.nn as nn
from torch_harmonics.examples.models._layers import MLP, LayerNorm, DropPath, SequencePositionEmbedding, SpectralPositionEmbedding, LearnablePositionEmbedding
from natten import NeighborhoodAttention2D as NeighborhoodAttention
from functools import partial


class Encoder(nn.Module):
    def __init__(
        self,
        in_shape=(721, 1440),
        out_shape=(480, 960),
        in_chans=2,
        out_chans=2,
        kernel_shape=(3, 3),
        groups=1,
        bias=False,
    ):
        super().__init__()
        stride_h = in_shape[0] // out_shape[0]
        stride_w = in_shape[1] // out_shape[1]
        pad_h = math.ceil(((out_shape[0] - 1) * stride_h - in_shape[0] + kernel_shape[0]) / 2)
        pad_w = math.ceil(((out_shape[1] - 1) * stride_w - in_shape[1] + kernel_shape[1]) / 2)
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_shape, bias=bias, stride=(stride_h, stride_w), padding=(pad_h, pad_w), groups=groups)

    def forward(self, x):
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_shape=(480, 960), out_shape=(721, 1440), in_chans=2, out_chans=2, kernel_shape=(3, 3), groups=1, bias=False, upsampling_method="conv"):
        super().__init__()
        self.out_shape = out_shape
        self.upsampling_method = upsampling_method

        if upsampling_method == "conv":
            self.upsample = nn.Sequential(
                nn.Upsample(
                    size=out_shape,
                    mode="bilinear",
                ),
                nn.Conv2d(in_chans, out_chans, kernel_size=kernel_shape, bias=bias, padding="same", groups=groups),
            )
        elif upsampling_method == "pixel_shuffle":
            # check if it is possible to use PixelShuffle
            if out_shape[0] // in_shape[0] != out_shape[1] // in_shape[1]:
                raise Exception(f"out_shape {out_shape} and in_shape {in_shape} are incompatible for shuffle decoding")
            upsampling_factor = out_shape[0] // in_shape[0]
            self.upsample = nn.Sequential(
                nn.Conv2d(in_chans, out_chans * (upsampling_factor**2), kernel_size=1, bias=bias, padding=0, groups=groups), nn.PixelShuffle(upsampling_factor)
            )
        else:
            raise ValueError(f"Unknown upsampling method {upsampling_method}")

    def forward(self, x):
        x = self.upsample(x)
        return x


class GlobalAttention(nn.Module):
    """
    Global self-attention block over 2D inputs using MultiheadAttention.

    Input shape: (B, C, H, W)
    Output shape: (B, C, H, W) with residual skip.
    """

    def __init__(self, chans, num_heads=8, dropout=0.0, bias=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=chans, num_heads=num_heads, dropout=dropout, batch_first=True, bias=bias)

    def forward(self, x):
        # x: B, C, H, W
        B, H, W, C = x.shape
        # flatten spatial dims
        x_flat = x.reshape(B, H * W, C)  # B, N, C
        # self-attention
        out, _ = self.attn(x_flat, x_flat, x_flat)
        # reshape back
        out = out.view(B, H, W, C)
        return out


class AttentionBlock(nn.Module):
    """
    Neighborhood attention block based on Natten.
    """

    def __init__(
        self,
        in_shape=(480, 960),
        out_shape=(480, 960),
        chans=2,
        num_heads=1,
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer="none",
        use_mlp=True,
        bias=True,
        attention_mode="neighborhood",
        attn_kernel_shape=(7, 7),
    ):
        super().__init__()

        # normalisation layer
        if norm_layer == "layer_norm":
            self.norm0 = LayerNorm(in_channels=chans, eps=1e-6)
            self.norm1 = LayerNorm(in_channels=chans, eps=1e-6)
        elif norm_layer == "instance_norm":
            self.norm0 = nn.InstanceNorm2d(num_features=chans, eps=1e-6, affine=True, track_running_stats=False)
            self.norm1 = nn.InstanceNorm2d(num_features=chans, eps=1e-6, affine=True, track_running_stats=False)
        elif norm_layer == "none":
            self.norm0 = nn.Identity()
            self.norm1 = nn.Identity()
        else:
            raise NotImplementedError(f"Error, normalization {norm_layer} not implemented.")

        # determine shape for neighborhood attention
        if attention_mode == "neighborhood":
            self.self_attn = NeighborhoodAttention(
                chans,
                kernel_size=attn_kernel_shape,
                dilation=1,
                num_heads=num_heads,
                qkv_bias=bias,
                qk_scale=None,
                attn_drop=drop_rate,
                proj_drop=drop_rate,
            )
        else:
            self.self_attn = GlobalAttention(chans, num_heads=num_heads, dropout=drop_rate, bias=bias)

        self.skip0 = nn.Identity()

        # dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if use_mlp == True:
            mlp_hidden_dim = int(chans * mlp_ratio)
            self.mlp = MLP(
                in_features=chans,
                out_features=chans,
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

        x = x.permute(0, 2, 3, 1)
        x = self.self_attn(x).permute(0, 3, 1, 2)

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


class Transformer(nn.Module):
    """
    Parameters
    ----------
    img_size : tuple of int
        (latitude, longitude) size of the input tensor.
    scale_factor : int
        Ratio for down- and up-sampling between input and internal resolution.
    in_chans : int
        Number of channels in the input tensor.
    out_chans : int
        Number of channels in the output tensor.
    embed_dim : int
        Embedding dimension inside attention blocks.
    num_layers : int
        Number of attention blocks.
    activation_function : str
        "relu", "gelu", or "identity" specifying the activation.
    encoder_kernel_shape : tuple of int
        Kernel size for the encoder convolution.
    num_heads : int
        Number of heads in NeighborhoodAttention.
    use_mlp : bool
        If True, an MLP follows attention in each block.
    mlp_ratio : float
        Ratio of MLP hidden dim to input dim.
    drop_rate : float
        Dropout rate before positional embedding.
    drop_path_rate : float
        Stochastic depth rate across transformer blocks.
    normalization_layer : str
        "layer_norm", "instance_norm", or "none".
    residual_prediction : bool
        If True, add the input as a global skip connection.
    pos_embed : str
        "sequence", "spectral", "learnable lat", "learnable latlon", or "none".
    bias : bool
        Whether convolution and attention projections include bias.
    attention_mode: str
        "neighborhood" or "global"
    upsampling_method: str
        "conv" or "pixel_shuffle"
    attn_kernel_shape: tuple

    Example
    -------
    >>> model = Transformer(
    ...     img_size=(128, 256),
    ...     scale_factor=2,
    ...     in_chans=3,
    ...     out_chans=3,
    ...     embed_dim=256,
    ...     num_layers=4,
    ...     activation_function="gelu",
    ...     encoder_kernel_shape=(3, 3),
    ...     num_heads=1,
    ...     use_mlp=True,
    ...     mlp_ratio=2.0,
    ...     drop_rate=0.0,
    ...     drop_path_rate=0.0,
    ...     normalization_layer="none",
    ...     residual_prediction=False,
    ...     pos_embed="spectral",
    ...     bias=True,
    ...     attention_mode="neighborhood",
    ...     attn_kernel_shape=(7,7),
    ...     upsampling_method="conv"
    ... )
    >>> x = torch.randn(1, 3, 128, 256)
    >>> print(model(x).shape)
    torch.Size([1, 3, 128, 256])
    """

    def __init__(
        self,
        img_size=(128, 256),
        grid_internal="legendre-gauss",
        scale_factor=3,
        in_chans=3,
        out_chans=3,
        embed_dim=256,
        num_layers=4,
        activation_function="gelu",
        encoder_kernel_shape=(3, 3),
        num_heads=1,
        use_mlp=True,
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        normalization_layer="none",
        residual_prediction=False,
        pos_embed="spectral",
        bias=True,
        attention_mode="neighborhood",
        attn_kernel_shape=(7, 7),
        upsampling_method="conv",
    ):
        super().__init__()

        self.img_size = img_size
        self.scale_factor = scale_factor
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.encoder_kernel_shape = encoder_kernel_shape
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

        # compute downsampled image size. We assume that the latitude-grid includes both poles
        self.h = (self.img_size[0] - 1) // scale_factor + 1
        self.w = self.img_size[1] // scale_factor

        # dropout
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0.0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]

        if pos_embed == "sequence":
            self.pos_embed = SequencePositionEmbedding((self.h, self.w), num_chans=self.embed_dim, grid=grid_internal)
        elif pos_embed == "spectral":
            self.pos_embed = SpectralPositionEmbedding((self.h, self.w), num_chans=self.embed_dim, grid=grid_internal)
        elif pos_embed == "learnable lat":
            self.pos_embed = LearnablePositionEmbedding((self.h, self.w), num_chans=self.embed_dim, grid=grid_internal, embed_type="lat")
        elif pos_embed == "learnable latlon":
            self.pos_embed = LearnablePositionEmbedding((self.h, self.w), num_chans=self.embed_dim, grid=grid_internal, embed_type="latlon")
        elif pos_embed == "none":
            self.pos_embed = nn.Identity()
        else:
            raise ValueError(f"Unknown position embedding type {pos_embed}")

        # maybe keep for now becuase tr
        # encoder
        self.encoder = Encoder(
            in_shape=self.img_size,
            out_shape=(self.h, self.w),
            in_chans=self.in_chans,
            out_chans=self.embed_dim,
            kernel_shape=self.encoder_kernel_shape,
            groups=1,
            bias=False,
        )

        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):

            block = AttentionBlock(
                in_shape=(self.h, self.w),
                out_shape=(self.h, self.w),
                chans=self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path=dpr[i],
                act_layer=self.activation_function,
                norm_layer=self.normalization_layer,
                use_mlp=use_mlp,
                bias=bias,
                attention_mode=attention_mode,
                attn_kernel_shape=attn_kernel_shape,
            )

            self.blocks.append(block)

        # decoder
        self.decoder = Decoder(
            in_shape=(self.h, self.w),
            out_shape=self.img_size,
            in_chans=self.embed_dim,
            out_chans=self.out_chans,
            kernel_shape=self.encoder_kernel_shape,
            groups=1,
            bias=False,
            upsampling_method=upsampling_method,
        )

    @torch.jit.ignore
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
