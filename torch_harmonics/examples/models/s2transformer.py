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
import torch.amp as amp

from torch_harmonics import DiscreteContinuousConvS2, DiscreteContinuousConvTransposeS2
from torch_harmonics import NeighborhoodAttentionS2
from torch_harmonics import ResampleS2
from torch_harmonics import InverseRealSHT
from torch_harmonics.quadrature import _precompute_latitudes

from ._layers import *

from functools import partial


class PositionEmbeddingS2(nn.Module):
    """
    Returns position embeddings on the torus
    """

    def __init__(self, img_shape=(480, 960), grid="equiangular", num_chans=1):

        super().__init__()

        self.img_shape = img_shape
        self.num_chans = num_chans

        with torch.no_grad():

            # alternating custom position embeddings
            lats, _ = _precompute_latitudes(img_shape[0], grid=grid)
            lats = lats.reshape(-1, 1)
            lons = torch.linspace(0, 2 * math.pi, img_shape[1] + 1)[:-1]
            lons = lons.reshape(1, -1)

            # channel index
            k = torch.arange(self.num_chans).reshape(1, -1, 1, 1)
            pos_embed = torch.where(k % 2 == 0, torch.sin(k * (lons + lats)), torch.cos(k * (lons - lats)))

        # register tensor
        self.register_buffer("position_embeddings", pos_embed.float())

    def forward(self, x: torch.Tensor):

        return x + self.position_embeddings


class SphericalHarmonicEmbedding(nn.Module):
    """
    Returns position embeddings for the spherical transformer
    """

    def __init__(self, img_shape=(480, 960), grid="equiangular", num_chans=1):

        super().__init__()

        self.img_shape = img_shape
        self.num_chans = num_chans

        # compute maximum required frequency and prepare isht
        lmax = math.floor(math.sqrt(self.num_chans)) + 1
        isht = InverseRealSHT(nlat=self.img_shape[0], nlon=self.img_shape[1], lmax=lmax, mmax=lmax, grid=grid)

        # fill position embedding
        with torch.no_grad():
            pos_embed_freq = torch.zeros(1, self.num_chans, isht.lmax, isht.mmax, dtype=torch.complex64)

            for i in range(self.num_chans):
                l = math.floor(math.sqrt(i))
                m = i - l**2 - l

                if m < 0:
                    pos_embed_freq[0, i, l, -m] = 1.0j
                else:
                    pos_embed_freq[0, i, l, m] = 1.0

        # compute spatial position embeddings
        pos_embed = isht(pos_embed_freq)

        # normalization
        pos_embed = pos_embed / torch.amax(pos_embed.abs(), dim=(-1, -2), keepdim=True)

        # register tensor
        self.register_buffer("position_embeddings", pos_embed)

    def forward(self, x: torch.Tensor):
        return x + self.position_embeddings


class DiscreteContinuousEncoder(nn.Module):
    def __init__(
        self,
        in_shape=(721, 1440),
        out_shape=(480, 960),
        grid_in="equiangular",
        grid_out="equiangular",
        in_chans=2,
        out_chans=2,
        kernel_shape=(3, 3),
        basis_type="morlet",
        groups=1,
        bias=False,
    ):
        super().__init__()

        # set up local convolution
        self.conv = DiscreteContinuousConvS2(
            in_chans,
            out_chans,
            in_shape=in_shape,
            out_shape=out_shape,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            grid_in=grid_in,
            grid_out=grid_out,
            groups=groups,
            bias=bias,
            theta_cutoff=4.0 * torch.pi / float(out_shape[0] - 1),
        )

    def forward(self, x):
        dtype = x.dtype

        with amp.autocast(device_type="cuda", enabled=False):
            x = x.float()
            x = self.conv(x)
            x = x.to(dtype=dtype)

        return x


class DiscreteContinuousDecoder(nn.Module):
    def __init__(
        self,
        in_shape=(480, 960),
        out_shape=(721, 1440),
        grid_in="equiangular",
        grid_out="equiangular",
        in_chans=2,
        out_chans=2,
        kernel_shape=(3, 3),
        basis_type="morlet",
        groups=1,
        bias=False,
        upsample_sht=False,
    ):
        super().__init__()

        # set up upsampling
        if upsample_sht:
            self.sht = RealSHT(*in_shape, grid=grid_in).float()
            self.isht = InverseRealSHT(*out_shape, lmax=self.sht.lmax, mmax=self.sht.mmax, grid=grid_out).float()
            self.upsample = nn.Sequential(self.sht, self.isht)
        else:
            self.upsample = ResampleS2(*in_shape, *out_shape, grid_in=grid_in, grid_out=grid_out)

        # set up DISCO convolution
        self.conv = DiscreteContinuousConvS2(
            in_chans,
            out_chans,
            in_shape=out_shape,
            out_shape=out_shape,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            grid_in=grid_out,
            grid_out=grid_out,
            groups=groups,
            bias=False,
            theta_cutoff=4.0 * torch.pi / float(in_shape[0] - 1),
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
    Helper module for a single SFNO/FNO block. Can use both FFTs and SHTs to represent either FNO or SFNO blocks.
    """

    def __init__(
        self,
        in_shape=(480, 960),
        out_shape=(480, 960),
        grid_in="equiangular",
        grid_out="equiangular",
        in_chans=2,
        out_chans=2,
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer="none",
        use_mlp=True,
    ):
        super().__init__()

        # normalisation layer
        if norm_layer == "layer_norm":
            self.norm0 = nn.LayerNorm(normalized_shape=in_shape, eps=1e-6)
            self.norm1 = nn.LayerNorm(normalized_shape=out_shape, eps=1e-6)
        elif norm_layer == "instance_norm":
            self.norm0 = nn.InstanceNorm2d(num_features=in_chans, eps=1e-6, affine=True, track_running_stats=False)
            self.norm1 = nn.InstanceNorm2d(num_features=out_chans, eps=1e-6, affine=True, track_running_stats=False)
        elif norm_layer == "none":
            self.norm0 = nn.Identity()
            self.norm1 = nn.Identity()
        else:
            raise NotImplementedError(f"Error, normalization {norm_layer} not implemented.")

        # determine radius for neighborhood attention
        theta_cutoff = 5 * torch.pi / (in_shape[0] - 1)

        self.self_attn = NeighborhoodAttentionS2(
            in_channels=in_chans,
            in_shape=in_shape,
            out_shape=out_shape,
            grid_in=grid_in,
            grid_out=grid_out,
            theta_cutoff=theta_cutoff,
            k_channels=None,
            out_channels=None,
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
    -----------
    img_shape : tuple, optional
        Shape of the input channels, by default (128, 256)
    kernel_shape: tuple, int
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
    encoder_kernel_shape : int, optional
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
    residual_prediction : bool, optional
        Whether to add a single large skip connection, by default True
    pos_embed : bool, optional
        Whether to use positional embedding, by default True
    upsample_sht : bool, optional
        Use SHT upsampling if true, else linear interpolation

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
        scale_factor=3,
        in_chans=3,
        out_chans=3,
        embed_dim=256,
        num_layers=4,
        activation_function="gelu",
        encoder_kernel_shape=(3, 3),
        filter_basis_type="morlet",
        use_mlp=True,
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        normalization_layer="none",
        hard_thresholding_fraction=1.0,
        use_complex_kernels=True,
        residual_prediction=False,
        pos_embed=False,
        upsample_sht=False,
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

        # compute downsampled image size. We assume that the latitude-grid includes both poles
        self.h = (self.img_size[0] - 1) // scale_factor + 1
        self.w = self.img_size[1] // scale_factor

        # dropout
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0.0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]

        # # for transformers this should be reingineered properly
        # if pos_embed == "latlon" or pos_embed == True:
        #     self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, self.h, self.w))
        #     nn.init.constant_(self.pos_embed, 0.0)
        # elif pos_embed == "lat":
        #     self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, self.h, 1))
        #     nn.init.constant_(self.pos_embed, 0.0)
        # elif pos_embed == "const":
        #     self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, 1, 1))
        #     nn.init.constant_(self.pos_embed, 0.0)
        # else:
        #     self.pos_embed = None
        if pos_embed == "spherical harmonic":
            self.pos_embed = SphericalHarmonicEmbedding((self.h, self.w), num_chans=self.embed_dim, grid=grid_internal)
        elif pos_embed == "s2":
            self.pos_embed = PositionEmbeddingS2((self.h, self.w), num_chans=self.embed_dim, grid=grid_internal)
        elif pos_embed == "none":
            self.pos_embed = nn.Identity()

        # maybe keep for now becuase tr
        # encoder
        self.encoder = DiscreteContinuousEncoder(
            in_shape=self.img_size,
            out_shape=(self.h, self.w),
            grid_in=grid,
            grid_out=grid_internal,
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
                in_shape=(self.h, self.w),
                out_shape=(self.h, self.w),
                grid_in=grid_internal,
                grid_out=grid_internal,
                in_chans=self.embed_dim,
                out_chans=self.embed_dim,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path=dpr[i],
                act_layer=self.activation_function,
                norm_layer=self.normalization_layer,
                use_mlp=use_mlp,
            )

            self.blocks.append(block)

        # decoder
        self.decoder = DiscreteContinuousDecoder(
            in_shape=(self.h, self.w),
            out_shape=self.img_size,
            grid_in=grid_internal,
            grid_out=grid,
            in_chans=self.embed_dim,
            out_chans=self.out_chans,
            kernel_shape=self.encoder_kernel_shape,
            basis_type=filter_basis_type,
            groups=1,
            bias=False,
            upsample_sht=upsample_sht,
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
