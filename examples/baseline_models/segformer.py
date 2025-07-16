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

from natten import NeighborhoodAttention2D as NeighborhoodAttention
from torch_harmonics.examples.models._layers import MLP, LayerNorm, DropPath

from functools import partial


class OverlapPatchMerging(nn.Module):
    """
    OverlapPatchMerging layer for merging patches.
    
    Parameters
    -----------
    in_shape : tuple
        Input shape (height, width)
    out_shape : tuple
        Output shape (height, width)
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_shape : tuple
        Kernel shape for convolution
    bias : bool, optional
        Whether to use bias, by default False
    """
    def __init__(
        self,
        in_shape=(721, 1440),
        out_shape=(481, 960),
        in_channels=3,
        out_channels=64,
        kernel_shape=(3, 3),
        bias=False,
    ):
        super().__init__()

        # conv
        stride_h = in_shape[0] // out_shape[0]
        stride_w = in_shape[1] // out_shape[1]
        pad_h = math.ceil(((out_shape[0] - 1) * stride_h - in_shape[0] + kernel_shape[0]) / 2)
        pad_w = math.ceil(((out_shape[1] - 1) * stride_w - in_shape[1] + kernel_shape[1]) / 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_shape,
            bias=bias,
            stride=(stride_h, stride_w),
            padding=(pad_h, pad_w),
        )

        # layer norm
        self.norm = nn.LayerNorm((out_channels), eps=1e-05, elementwise_affine=True, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.conv(x)

        # permute
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        out = x.permute(0, 3, 1, 2)

        return out


class MixFFN(nn.Module):
    """
    MixFFN module combining MLP and depthwise convolution.
    
    Parameters
    -----------
    shape : tuple
        Input shape (height, width)
    inout_channels : int
        Number of input/output channels
    hidden_channels : int
        Number of hidden channels in MLP
    mlp_bias : bool, optional
        Whether to use bias in MLP layers, by default True
    kernel_shape : tuple, optional
        Kernel shape for depthwise convolution, by default (3, 3)
    conv_bias : bool, optional
        Whether to use bias in convolution, by default False
    activation : callable, optional
        Activation function, by default nn.GELU
    use_mlp : bool, optional
        Whether to use MLP instead of linear layers, by default False
    drop_path : float, optional
        Drop path rate, by default 0.0
    """
    def __init__(
        self,
        shape,
        inout_channels,
        hidden_channels,
        mlp_bias=True,
        kernel_shape=(3, 3),
        conv_bias=False,
        activation=nn.GELU,
        use_mlp=False,
        drop_path=0.0,
    ):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm = nn.LayerNorm((inout_channels), eps=1e-05, elementwise_affine=True, bias=True)

        if use_mlp:
            # although the paper says MLP, it uses a single linear layer
            self.mlp_in = MLP(inout_channels, hidden_features=hidden_channels, out_features=inout_channels, act_layer=activation, output_bias=False, drop_rate=0.0)
        else:
            self.mlp_in = nn.Conv2d(in_channels=inout_channels, out_channels=inout_channels, kernel_size=1, bias=True)

        self.conv = nn.Conv2d(inout_channels, inout_channels, kernel_size=kernel_shape, groups=inout_channels, bias=conv_bias, padding="same")

        if use_mlp:
            self.mlp_out = MLP(inout_channels, hidden_features=hidden_channels, out_features=inout_channels, act_layer=activation, output_bias=False, drop_rate=0.0)
        else:
            self.mlp_out = nn.Conv2d(in_channels=inout_channels, out_channels=inout_channels, kernel_size=1, bias=True)

        self.act = activation()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        residual = x

        # norm
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)

        # NOTE: we add another activation here
        # because in the paper the authors only use depthwise conv,
        # but without this activation it would just be a fused MM
        # with the disco conv
        x = self.mlp_in(x)

        # conv parth
        x = self.act(self.conv(x))

        # second linear
        x = self.mlp_out(x)

        return residual + self.drop_path(x)


class GlobalAttention(nn.Module):
    """
    Global self-attention block over 2D inputs using MultiheadAttention.

    Input shape: (B, C, H, W)
    Output shape: (B, C, H, W) with residual skip.
    
    Parameters
    -----------
    chans : int
        Number of channels
    num_heads : int, optional
        Number of attention heads, by default 8
    dropout : float, optional
        Dropout rate, by default 0.0
    bias : bool, optional
        Whether to use bias, by default True
    """

    def __init__(self, chans, num_heads=8, dropout=0.0, bias=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=chans, num_heads=num_heads, dropout=dropout, batch_first=True, bias=bias)

    def forward(self, x):

        # x: B, C, H, W
        B, H, W, C = x.shape
        # flatten spatial dims
        x_flat = x.view(B, H * W, C)  # B, N, C
        # self-attention
        out, _ = self.attn(x_flat, x_flat, x_flat)
        # reshape back
        out = out.view(B, H, W, C)
        return out


class AttentionWrapper(nn.Module):
    """
    Wrapper for different attention mechanisms.
    
    Parameters
    -----------
    channels : int
        Number of channels
    shape : tuple
        Input shape (height, width)
    heads : int
        Number of attention heads
    pre_norm : bool, optional
        Whether to apply normalization before attention, by default False
    attention_drop_rate : float, optional
        Attention dropout rate, by default 0.0
    drop_path : float, optional
        Drop path rate, by default 0.0
    attention_mode : str, optional
        Attention mode ("neighborhood", "global"), by default "neighborhood"
    kernel_shape : tuple, optional
        Kernel shape for neighborhood attention, by default (7, 7)
    bias : bool, optional
        Whether to use bias, by default True
    """
    def __init__(self, channels, shape, heads, pre_norm=False, attention_drop_rate=0.0, drop_path=0.0, attention_mode="neighborhood", kernel_shape=(7, 7), bias=True):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.attention_mode = attention_mode

        if attention_mode == "neighborhood":
            self.att = NeighborhoodAttention(
                channels, kernel_size=kernel_shape, dilation=1, num_heads=heads, qk_scale=None, attn_drop=attention_drop_rate, proj_drop=0.0, qkv_bias=bias
            )
        elif attention_mode == "global":
            self.att = GlobalAttention(channels, num_heads=heads, dropout=attention_drop_rate, bias=bias)
        else:
            raise ValueError(f"Unknown attention mode function {attention_mode}")

        self.norm = None
        if pre_norm:
            self.norm = nn.LayerNorm((channels), eps=1e-05, elementwise_affine=True, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        residual = x
        x = x.permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)

        x = self.att(x)
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and MLP.
    
    Parameters
    ----------
    in_shape : tuple
        Input shape (height, width)
    out_shape : tuple
        Output shape (height, width)
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    mlp_hidden_channels : int
        Number of hidden channels in MLP
    nrep : int, optional
        Number of repetitions of attention and MLP blocks, by default 1
    heads : int, optional
        Number of attention heads, by default 1
    kernel_shape : tuple, optional
        Kernel shape for neighborhood attention, by default (3, 3)
    activation : torch.nn.Module, optional
        Activation function to use, by default nn.GELU
    att_drop_rate : float, optional
        Attention dropout rate, by default 0.0
    drop_path_rates : float or list, optional
        Drop path rates for each block, by default 0.0
    attention_mode : str, optional
        Attention mode ("neighborhood", "global"), by default "neighborhood"
    attn_kernel_shape : tuple, optional
        Kernel shape for neighborhood attention, by default (7, 7)
    bias : bool, optional
        Whether to use bias, by default True
    """
    
    def __init__(
        self,
        in_shape,
        out_shape,
        in_channels,
        out_channels,
        mlp_hidden_channels,
        nrep=1,
        heads=1,
        kernel_shape=(3, 3),
        activation=nn.GELU,
        att_drop_rate=0.0,
        drop_path_rates=0.0,
        attention_mode="neighborhood",
        attn_kernel_shape=(7, 7),
        bias=True
    ):
        super().__init__()

        # ensure odd
        if attn_kernel_shape[0] % 2 == 0:
            raise ValueError(f"Attn Kernel shape {kernel_shape} is even, use odd kernel shape")
        if attn_kernel_shape[1] % 2 == 0:
            raise ValueError(f"Kernel shape {kernel_shape} is even, use odd kernel shape")

        attn_kernel_shape = list(attn_kernel_shape)
        orig_attn_kernel_shape = attn_kernel_shape.copy()

        # ensure that attn kernel shape is smaller than in_shape in both dimensions
        # if necessary fix kernel_shape to be 1 less (and odd) than in_shape
        if attn_kernel_shape[0] >= out_shape[0]:
            attn_kernel_shape[0] = out_shape[0] - 1
            # ensure odd
            if attn_kernel_shape[0] % 2 == 0:
                attn_kernel_shape[0] -= 1

            # make square if original was square
            if orig_attn_kernel_shape[0] == orig_attn_kernel_shape[1]:
                attn_kernel_shape[1] = attn_kernel_shape[0]
        if attn_kernel_shape[1] >= out_shape[1]:
            attn_kernel_shape[1] = out_shape[1] - 1
            # ensure odd
            if attn_kernel_shape[1] % 2 == 0:
                attn_kernel_shape[1] -= 1

        attn_kernel_shape = tuple(attn_kernel_shape)

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(drop_path_rates, float):
            drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rates, nrep)]

        assert len(drop_path_rates) == nrep

        self.fwd = [
            OverlapPatchMerging(
                in_shape=in_shape,
                out_shape=out_shape,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_shape=kernel_shape,
                bias=False,
            )
        ]

        for i in range(nrep):
            self.fwd.append(
                AttentionWrapper(
                    channels=out_channels,
                    shape=out_shape,
                    heads=heads,
                    pre_norm=True,
                    attention_drop_rate=att_drop_rate,
                    drop_path=drop_path_rates[i],
                    attention_mode=attention_mode,
                    kernel_shape=attn_kernel_shape,
                    bias=bias
                )
            )

            self.fwd.append(
                MixFFN(
                    out_shape,
                    inout_channels=out_channels,
                    hidden_channels=mlp_hidden_channels,
                    mlp_bias=True,
                    kernel_shape=kernel_shape,
                    conv_bias=False,
                    activation=activation,
                    use_mlp=False,
                    drop_path=drop_path_rates[i],
                )
            )

        # make sequential
        self.fwd = nn.Sequential(*self.fwd)

        # final norm
        self.norm = nn.LayerNorm((out_channels), eps=1e-05, elementwise_affine=True, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fwd(x)

        # apply norm
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)

        return x


class Upsampling(nn.Module):
    """
    Upsampling block for the Segformer model.
    
    Parameters
    ----------
    in_shape : tuple
        Input shape (height, width)
    out_shape : tuple
        Output shape (height, width)
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    hidden_channels : int
        Number of hidden channels in MLP
    mlp_bias : bool, optional
        Whether to use bias in MLP, by default True
    kernel_shape : tuple, optional
        Kernel shape for convolution, by default (3, 3)
    conv_bias : bool, optional
        Whether to use bias in convolution, by default False
    activation : torch.nn.Module, optional
        Activation function to use, by default nn.GELU
    use_mlp : bool, optional
        Whether to use MLP, by default False
    """
    
    def __init__(
        self,
        in_shape,
        out_shape,
        in_channels,
        out_channels,
        hidden_channels,
        mlp_bias=True,
        kernel_shape=(3, 3),
        conv_bias=False,
        activation=nn.GELU,
        use_mlp=False,
    ):
        super().__init__()
        self.out_shape = out_shape
        if use_mlp:
            self.mlp = MLP(in_channels, hidden_features=hidden_channels, out_features=out_channels, act_layer=activation, output_bias=False, drop_rate=0.0)
        else:
            self.mlp = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(self.mlp(x), size=self.out_shape, mode="bilinear")
        return x


class Segformer(nn.Module):
    """
    Spherical segformer model designed to approximate mappings from spherical signals to spherical segmentation masks

    Parameters
    ----------
    img_shape : tuple, optional
        Shape of the input channels, by default (128, 256)
    kernel_shape: tuple, int
    scale_factor: int, optional
        Scale factor to use, by default 2
    in_chans : int, optional
        Number of input channels, by default 3
    out_chans : int, optional
        Number of classes, by default 3
    embed_dims : List[int], optional
        Dimension of the embeddings for each block, has to be the same length as heads
    heads : List[int], optional
        Number of heads for each block in the network, has to be the same length as embed_dims
    depths: List[in], optional
        Number of repetitions of attentions blocks and ffn mixers per layer. Has to be the same length as embed_dims and heads
    activation_function : str, optional
        Activation function to use, by default "gelu"
    embedder_kernel_shape : int, optional
        size of the encoder kernel
    use_mlp : int, optional
        Whether to use MLPs in the SFNO blocks, by default True
    mlp_ratio : int, optional
        Ratio of MLP to use, by default 2.0
    drop_rate : float, optional
        Dropout rate, by default 0.0
    drop_path_rate : float, optional
        Dropout path rate, by default 0.0
    normalization_layer : str, optional
        Type of normalization layer to use ("layer_norm", "instance_norm", "none"), by default "instance_norm"

    Example
    ----------
    >>> model = Segformer(
    ...         img_size=(128, 256),
    ...         in_chans=3,
    ...         out_chans=3,
    ...         embed_dims=[64, 128, 256, 512],
    ...         heads=[1, 2, 4, 8],
    ...         depths=[3, 4, 6, 3],
    ...         scale_factor=2,
    ...         activation_function="gelu",
    ...         kernel_shape=(3, 3),
    ...         mlp_ratio=2.0,
    ...         att_drop_rate=0.0,
    ...         drop_path_rate=0.1,
    ...         attention_mode="global",
    ))
    >>> model(torch.randn(1, 2, 128, 256)).shape
    torch.Size([1, 2, 128, 256])
    """

    def __init__(
        self,
        img_size=(128, 256),
        in_chans=3,
        out_chans=3,
        embed_dims=[64, 128, 256, 512],
        heads=[1, 2, 4, 8],
        depths=[3, 4, 6, 3],
        scale_factor=2,
        activation_function="gelu",
        kernel_shape=(3, 3),
        mlp_ratio=2.0,
        att_drop_rate=0.0,
        drop_path_rate=0.1,
        attention_mode="neighborhood",
        attn_kernel_shape=(7, 7),
        bias=True
    ):
        super().__init__()

        self.img_size = img_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dims = embed_dims
        self.heads = heads
        self.num_blocks = len(self.embed_dims)
        self.depths = depths
        self.kernel_shape = kernel_shape

        assert len(self.heads) == self.num_blocks
        assert len(self.depths) == self.num_blocks

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

        # set up drop path rates
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]

        self.blocks = nn.ModuleList([])
        out_shape = img_size
        in_channels = in_chans
        cur = 0
        for i in range(self.num_blocks):
            out_shape_new = (out_shape[0] // scale_factor, out_shape[1] // scale_factor)
            out_channels = self.embed_dims[i]
            self.blocks.append(
                TransformerBlock(
                    in_shape=out_shape,
                    out_shape=out_shape_new,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    mlp_hidden_channels=int(mlp_ratio * out_channels),
                    nrep=self.depths[i],
                    heads=self.heads[i],
                    kernel_shape=kernel_shape,
                    activation=self.activation_function,
                    att_drop_rate=att_drop_rate,
                    drop_path_rates=dpr[cur : cur + self.depths[i]],
                    attention_mode=attention_mode,
                    attn_kernel_shape=attn_kernel_shape,
                    bias=bias
                )
            )
            cur += self.depths[i]
            out_shape = out_shape_new
            in_channels = out_channels

        self.upsamplers = nn.ModuleList([])
        out_shape = img_size
        for i in range(self.num_blocks):
            in_shape = self.blocks[i].out_shape
            self.upsamplers.append(
                Upsampling(
                    in_shape=in_shape,
                    out_shape=out_shape,
                    in_channels=self.embed_dims[i],
                    out_channels=self.embed_dims[i],
                    hidden_channels=int(mlp_ratio * self.embed_dims[i]),
                    mlp_bias=True,
                    kernel_shape=kernel_shape,
                    conv_bias=False,
                    activation=nn.GELU,
                )
            )

        segmentation_head_dim = sum(self.embed_dims)
        self.segmentation_head = nn.Conv2d(in_channels=segmentation_head_dim, out_channels=out_chans, kernel_size=1, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        # encoder:
        features = []
        feat = x
        for block in self.blocks:
            feat = block(feat)
            features.append(feat)

        # perform upsample
        upfeats = []
        for feat, upsampler in zip(features, self.upsamplers):
            upfeat = upsampler(feat)
            upfeats.append(upfeat)

        # perform concatenation
        upfeats = torch.cat(upfeats, dim=1)

        # final upsampling and prediction
        out = self.segmentation_head(upfeats)

        return out
