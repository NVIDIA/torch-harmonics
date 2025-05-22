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
import torch.nn as nn

from torch_harmonics import RealSHT, InverseRealSHT

from torch_harmonics.examples.models._layers import MLP, SpectralConvS2, SequencePositionEmbedding, SpectralPositionEmbedding, LearnablePositionEmbedding

from functools import partial


class SphericalFourierNeuralOperatorBlock(nn.Module):
    """
    Helper module for a single SFNO/FNO block. Can use both FFTs and SHTs to represent either FNO or SFNO blocks.
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        input_dim,
        output_dim,
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer="none",
        inner_skip="none",
        outer_skip="identity",
        use_mlp=True,
        bias=False,
    ):
        super().__init__()

        if act_layer == nn.Identity:
            gain_factor = 1.0
        else:
            gain_factor = 2.0

        if inner_skip == "linear" or inner_skip == "identity":
            gain_factor /= 2.0

        self.global_conv = SpectralConvS2(forward_transform, inverse_transform, input_dim, output_dim, gain=gain_factor, bias=bias)

        if inner_skip == "linear":
            self.inner_skip = nn.Conv2d(input_dim, output_dim, 1, 1)
            nn.init.normal_(self.inner_skip.weight, std=math.sqrt(gain_factor / input_dim))
        elif inner_skip == "identity":
            assert input_dim == output_dim
            self.inner_skip = nn.Identity()
        elif inner_skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {inner_skip}")

        # normalisation layer
        if norm_layer == "layer_norm":
            self.norm = nn.LayerNorm(normalized_shape=(inverse_transform.nlat, inverse_transform.nlon), eps=1e-6)
        elif norm_layer == "instance_norm":
            self.norm = nn.InstanceNorm2d(num_features=output_dim, eps=1e-6, affine=True, track_running_stats=False)
        elif norm_layer == "none":
            self.norm = nn.Identity()
        else:
            raise NotImplementedError(f"Error, normalization {self.norm_layer} not implemented.")

        # dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        gain_factor = 1.0
        if outer_skip == "linear" or inner_skip == "identity":
            gain_factor /= 2.0

        if use_mlp == True:
            mlp_hidden_dim = int(output_dim * mlp_ratio)
            self.mlp = MLP(
                in_features=output_dim, out_features=input_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop_rate=drop_rate, checkpointing=False, gain=gain_factor
            )

        if outer_skip == "linear":
            self.outer_skip = nn.Conv2d(input_dim, input_dim, 1, 1)
            torch.nn.init.normal_(self.outer_skip.weight, std=math.sqrt(gain_factor / input_dim))
        elif outer_skip == "identity":
            assert input_dim == output_dim
            self.outer_skip = nn.Identity()
        elif outer_skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {outer_skip}")


    def forward(self, x):

        x, residual = self.global_conv(x)

        x = self.norm(x)

        if hasattr(self, "inner_skip"):
            x = x + self.inner_skip(residual)

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        x = self.drop_path(x)

        if hasattr(self, "outer_skip"):
            x = x + self.outer_skip(residual)

        return x


class SphericalFourierNeuralOperator(nn.Module):
    """
    SphericalFourierNeuralOperator module. Implements the 'linear' variant of the Spherical Fourier Neural Operator
    as presented in [1]. Spherical convolutions are applied via spectral transforms to apply a geometrically consistent
    and approximately equivariant architecture.

    Parameters
    ----------
    img_shape : tuple, optional
        Shape of the input channels, by default (128, 256)
    scale_factor : int, optional
        Scale factor to use, by default 3
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
    encoder_layers : int, optional
        Number of layers in the encoder, by default 1
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
    residual_prediction : bool, optional
        Whether to add a single large skip connection, by default True
    pos_embed : bool, optional
        Whether to use positional embedding, by default True
    bias : bool, optional
        Whether to use a bias, by default False

    Example:
    --------
    >>> model = SphericalFourierNeuralOperator(
    ...         img_shape=(128, 256),
    ...         scale_factor=4,
    ...         in_chans=2,
    ...         out_chans=2,
    ...         embed_dim=16,
    ...         num_layers=4,
    ...         use_mlp=True,)
    >>> model(torch.randn(1, 2, 128, 256)).shape
    torch.Size([1, 2, 128, 256])

    References
    -----------
    .. [1] Bonev B., Kurth T., Hundt C., Pathak, J., Baust M., Kashinath K., Anandkumar A.;
        "Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere" (2023).
        ICML 2023, https://arxiv.org/abs/2306.03838.
    """

    def __init__(
        self,
        img_size=(128, 256),
        grid="equiangular",
        grid_internal="legendre-gauss",
        scale_factor=3,
        in_chans=3,
        out_chans=3,
        embed_dim=256,
        num_layers=4,
        activation_function="gelu",
        encoder_layers=1,
        use_mlp=True,
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        normalization_layer="none",
        hard_thresholding_fraction=1.0,
        residual_prediction=False,
        pos_embed="none",
        bias=False,
    ):

        super().__init__()

        self.img_size = img_size
        self.grid = grid
        self.grid_internal = grid_internal
        self.scale_factor = scale_factor
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.normalization_layer = normalization_layer
        self.use_mlp = use_mlp
        self.encoder_layers = encoder_layers
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

        # construct an encoder with num_encoder_layers
        num_encoder_layers = 1
        encoder_hidden_dim = int(self.embed_dim * mlp_ratio)
        current_dim = self.in_chans
        encoder_layers = []
        for l in range(num_encoder_layers - 1):
            fc = nn.Conv2d(current_dim, encoder_hidden_dim, 1, bias=True)
            # initialize the weights correctly
            scale = math.sqrt(2.0 / current_dim)
            nn.init.normal_(fc.weight, mean=0.0, std=scale)
            if fc.bias is not None:
                nn.init.constant_(fc.bias, 0.0)
            encoder_layers.append(fc)
            encoder_layers.append(self.activation_function())
            current_dim = encoder_hidden_dim
        fc = nn.Conv2d(current_dim, self.embed_dim, 1, bias=bias)
        scale = math.sqrt(1.0 / current_dim)
        nn.init.normal_(fc.weight, mean=0.0, std=scale)
        if fc.bias is not None:
            nn.init.constant_(fc.bias, 0.0)
        encoder_layers.append(fc)
        self.encoder = nn.Sequential(*encoder_layers)

        # compute the modes for the sht
        modes_lat = self.h
        # due to some spectral artifacts with cufft, we substract one mode here
        modes_lon = (self.w // 2 + 1) - 1

        modes_lat = modes_lon = int(min(modes_lat, modes_lon) * self.hard_thresholding_fraction)

        self.trans_down = RealSHT(*self.img_size, lmax=modes_lat, mmax=modes_lon, grid=self.grid).float()
        self.itrans_up = InverseRealSHT(*self.img_size, lmax=modes_lat, mmax=modes_lon, grid=self.grid).float()
        self.trans = RealSHT(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=grid_internal).float()
        self.itrans = InverseRealSHT(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=grid_internal).float()

        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):

            first_layer = i == 0
            last_layer = i == self.num_layers - 1

            block = SphericalFourierNeuralOperatorBlock(
                self.trans_down if first_layer else self.trans,
                self.itrans_up if last_layer else self.itrans,
                self.embed_dim,
                self.embed_dim,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path=dpr[i],
                act_layer=self.activation_function,
                norm_layer=self.normalization_layer,
                use_mlp=use_mlp,
                bias=bias,
            )

            self.blocks.append(block)

        # construct an decoder with num_decoder_layers
        num_decoder_layers = 1
        decoder_hidden_dim = int(self.embed_dim * mlp_ratio)
        current_dim = self.embed_dim
        decoder_layers = []
        for l in range(num_decoder_layers - 1):
            fc = nn.Conv2d(current_dim, decoder_hidden_dim, 1, bias=True)
            # initialize the weights correctly
            scale = math.sqrt(2.0 / current_dim)
            nn.init.normal_(fc.weight, mean=0.0, std=scale)
            if fc.bias is not None:
                nn.init.constant_(fc.bias, 0.0)
            decoder_layers.append(fc)
            decoder_layers.append(self.activation_function())
            current_dim = decoder_hidden_dim
        fc = nn.Conv2d(current_dim, self.out_chans, 1, bias=bias)
        scale = math.sqrt(1.0 / current_dim)
        nn.init.normal_(fc.weight, mean=0.0, std=scale)
        if fc.bias is not None:
            nn.init.constant_(fc.bias, 0.0)
        decoder_layers.append(fc)
        self.decoder = nn.Sequential(*decoder_layers)

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
            x = self.pos_embed(x)

        x = self.forward_features(x)

        x = self.decoder(x)

        if self.residual_prediction:
            x = x + residual

        return x
