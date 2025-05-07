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
import torch.amp as amp

from torch_harmonics import DiscreteContinuousConvS2, DiscreteContinuousConvTransposeS2
from torch_harmonics import AttentionS2, NeighborhoodAttentionS2
from torch_harmonics import ResampleS2
from torch_harmonics import RealSHT, InverseRealSHT
from torch_harmonics.quadrature import _precompute_latitudes

from torch_harmonics.examples.models._layers import MLP, LayerNorm, DropPath

from functools import partial


# heuristic for finding theta_cutoff
def _compute_cutoff_radius(nlat, kernel_shape, basis_type):
    theta_cutoff_factor = {"piecewise linear": 0.5, "morlet": 0.5, "zernike": math.sqrt(2.0)}

    return (kernel_shape[0] + 1) * theta_cutoff_factor[basis_type] * math.pi / float(nlat - 1)


class OverlapPatchMerging(nn.Module):
    def __init__(
        self,
        in_shape=(721, 1440),
        out_shape=(481, 960),
        grid_in="equiangular",
        grid_out="equiangular",
        in_channels=3,
        out_channels=64,
        kernel_shape=(3, 3),
        basis_type="morlet",
        bias=False,
    ):
        super().__init__()

        # convolution for patches, curtoff radius inferred from kernel shape
        theta_cutoff = _compute_cutoff_radius(out_shape[0], kernel_shape, basis_type)
        self.conv = DiscreteContinuousConvS2(
            in_channels,
            out_channels,
            in_shape=in_shape,
            out_shape=out_shape,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            grid_in=grid_in,
            grid_out=grid_out,
            bias=bias,
            theta_cutoff=theta_cutoff,
        )

        # layer norm
        self.norm = nn.LayerNorm((out_channels), eps=1e-05, elementwise_affine=True, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        dtype = x.dtype

        with amp.autocast(device_type="cuda", enabled=False):
            x = x.float()
            x = self.conv(x).to(dtype=dtype)

        # permute
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        out = x.permute(0, 3, 1, 2)

        return out


class MixFFN(nn.Module):
    def __init__(
        self,
        shape,
        inout_channels,
        hidden_channels,
        mlp_bias=True,
        grid="equiangular",
        kernel_shape=(3, 3),
        basis_type="morlet",
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

        # convolution for patches, curtoff radius inferred from kernel shape
        theta_cutoff = _compute_cutoff_radius(shape[0], kernel_shape, basis_type)
        self.conv = DiscreteContinuousConvS2(
            inout_channels,
            inout_channels,
            in_shape=shape,
            out_shape=shape,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            grid_in=grid,
            grid_out=grid,
            groups=inout_channels,
            bias=conv_bias,
            theta_cutoff=theta_cutoff,
        )

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
        # because in the paper they only use depthwise conv,
        # but without this activation it would just be a fused MM
        # with the disco conv
        x = self.mlp_in(x)

        # conv parth
        x = self.act(self.conv(x))

        # second linear
        x = self.mlp_out(x)

        return residual + self.drop_path(x)


class AttentionWrapper(nn.Module):
    def __init__(
        self,
        channels,
        shape,
        grid,
        heads,
        pre_norm=False,
        attention_drop_rate=0.0,
        drop_path=0.0,
        attention_mode="neighborhood",
        theta_cutoff=None,
        bias=True
    ):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.attention_mode = attention_mode

        if attention_mode == "neighborhood":
            if theta_cutoff is None:
                theta_cutoff = (7.0 / math.sqrt(math.pi)) * math.pi / (shape[0] - 1)
            self.att = NeighborhoodAttentionS2(
                in_channels=channels,
                in_shape=shape,
                out_shape=shape,
                grid_in=grid,
                grid_out=grid,
                theta_cutoff=theta_cutoff,
                out_channels=channels,
                num_heads=heads,
                bias=bias
                # drop_rate=attention_drop_rate,
            )
        else:
            self.att = AttentionS2(
                in_channels=channels,
                num_heads=heads,
                in_shape=shape,
                out_shape=shape,
                grid_in=grid,
                grid_out=grid,
                out_channels=channels,
                drop_rate=attention_drop_rate,
                bias=bias
            )

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
        if self.norm is not None:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)

        if self.attention_mode == "neighborhood":
            dtype = x.dtype
            with amp.autocast(device_type="cuda", enabled=False):
                x = x.float()
                x = self.att(x).to(dtype=dtype)
        else:
            x = self.att(x)

        return residual + self.drop_path(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        in_shape,
        out_shape,
        in_channels,
        out_channels,
        mlp_hidden_channels,
        grid_in="equiangular",
        grid_out="equiangular",
        nrep=1,
        heads=1,
        kernel_shape=(3, 3),
        basis_type="morlet",
        activation=nn.GELU,
        att_drop_rate=0.0,
        drop_path_rates=0.0,
        attention_mode="neighborhood",
        theta_cutoff=None,
        bias=True
    ):
        super().__init__()

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
                grid_in=grid_in,
                grid_out=grid_out,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_shape=kernel_shape,
                basis_type=basis_type,
                bias=False,
            )
        ]

        for i in range(nrep):
            self.fwd.append(
                AttentionWrapper(
                    channels=out_channels,
                    shape=out_shape,
                    grid=grid_out,
                    heads=heads,
                    pre_norm=True,
                    attention_drop_rate=att_drop_rate,
                    drop_path=drop_path_rates[i],
                    attention_mode=attention_mode,
                    theta_cutoff=theta_cutoff,
                    bias=bias
                )
            )

            self.fwd.append(
                MixFFN(
                    out_shape,
                    inout_channels=out_channels,
                    hidden_channels=mlp_hidden_channels,
                    mlp_bias=True,
                    grid=grid_out,
                    kernel_shape=kernel_shape,
                    basis_type=basis_type,
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
    def __init__(
        self,
        in_shape,
        out_shape,
        in_channels,
        out_channels,
        hidden_channels,
        mlp_bias=True,
        grid_in="equiangular",
        grid_out="equiangular",
        kernel_shape=(3, 3),
        basis_type="morlet",
        conv_bias=False,
        activation=nn.GELU,
        use_mlp=False,
        upsampling_method="conv"
    ):
        super().__init__()

        if use_mlp:
            self.mlp = MLP(in_channels, hidden_features=hidden_channels, out_features=out_channels, act_layer=activation, output_bias=False, drop_rate=0.0)
        else:
            self.mlp = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)

        if upsampling_method == "conv":
            theta_cutoff = _compute_cutoff_radius(in_shape[0], kernel_shape, basis_type)
            self.upsample = DiscreteContinuousConvTransposeS2(
                out_channels,
                out_channels,
                in_shape=in_shape,
                out_shape=out_shape,
                kernel_shape=kernel_shape,
                basis_type=basis_type,
                grid_in=grid_in,
                grid_out=grid_out,
                bias=conv_bias,
                theta_cutoff=theta_cutoff,
            )
        elif upsampling_method == "bilinear":
            self.upsample = ResampleS2(*in_shape, *out_shape, grid_in=grid_in, grid_out=grid_out)
        else:
            raise ValueError(f"Unknown upsampling method {upsampling_method}")

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
        x = self.upsample(self.mlp(x))

        return x


class SphericalSegformer(nn.Module):
    """
    Spherical segformer model designed to approximate mappings from spherical signals to spherical segmentation masks

    Parameters
    -----------
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
    filter_basis_type: Optional[str]: str, optional
        filter basis type
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
    hard_thresholding_fraction : float, optional
        Fraction of hard thresholding (frequency cutoff) to apply, by default 1.0
    upsampling_method : str
        Conv, bilinear

    Example
    -----------
    >>> model = SphericalTransformer(
    ...         img_shape=(128, 256),
    ...         scale_factor=4,
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
        img_size=(128, 256),
        grid="equiangular",
        grid_internal="legendre-gauss",
        in_chans=3,
        out_chans=3,
        embed_dims=[64, 128, 256, 512],
        heads=[1, 2, 4, 8],
        depths=[3, 4, 6, 3],
        scale_factor=2,
        activation_function="gelu",
        kernel_shape=(3, 3),
        filter_basis_type="morlet",
        mlp_ratio=2.0,
        att_drop_rate=0.0,
        drop_path_rate=0.1,
        attention_mode="neighborhood",
        theta_cutoff=None,
        upsampling_method="bilinear",
        bias=True
    ):
        super().__init__()

        self.img_size = img_size
        self.grid = grid
        self.grid_internal = grid_internal
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
        grid_in = grid
        grid_out = grid_internal
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
                    grid_in=grid_in,
                    grid_out=grid_out,
                    nrep=self.depths[i],
                    heads=self.heads[i],
                    kernel_shape=kernel_shape,
                    basis_type=filter_basis_type,
                    activation=self.activation_function,
                    att_drop_rate=att_drop_rate,
                    drop_path_rates=dpr[cur : cur + self.depths[i]],
                    attention_mode=attention_mode,
                    theta_cutoff=theta_cutoff,
                    bias=bias
                )
            )
            cur += self.depths[i]
            out_shape = out_shape_new
            grid_in = grid_internal
            in_channels = out_channels

        self.upsamplers = nn.ModuleList([])
        out_shape = img_size
        grid_out = grid
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
                    grid_in=grid_internal,
                    grid_out=grid,
                    kernel_shape=kernel_shape,
                    basis_type=filter_basis_type,
                    conv_bias=False,
                    activation=nn.GELU,
                    upsampling_method=upsampling_method
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
            upfeats.append(upsampler(feat))

        # perform concatenation
        upfeats = torch.cat(upfeats, dim=1)

        # final upsampling and prediction
        out = self.segmentation_head(upfeats)

        return out
