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
from torch_harmonics import NeighborhoodAttentionS2
from torch_harmonics import ResampleS2
from torch_harmonics import RealSHT, InverseRealSHT
from torch_harmonics.quadrature import _precompute_latitudes

from torch_harmonics.examples.models._layers import MLP, DropPath

from functools import partial


# heuristic for finding theta_cutoff
def _compute_cutoff_radius(nlat, kernel_shape, basis_type):
    theta_cutoff_factor = {"piecewise linear": 0.5, "morlet": 0.5, "zernike": math.sqrt(2.0)}

    return (kernel_shape[0] + 1) * theta_cutoff_factor[basis_type] * math.pi / float(nlat - 1)


class DownsamplingBlock(nn.Module):
    """
    Downsampling block for spherical U-Net architecture.
    
    This block performs convolution operations followed by downsampling on spherical data,
    using discrete-continuous convolutions to maintain spectral properties.
    
    Parameters
    -----------
    in_shape : tuple
        Input shape (nlat, nlon)
    out_shape : tuple
        Output shape (nlat, nlon)
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    grid_in : str, optional
        Input grid type, by default "equiangular"
    grid_out : str, optional
        Output grid type, by default "equiangular"
    nrep : int, optional
        Number of convolution repetitions, by default 1
    kernel_shape : tuple, optional
        Kernel shape for convolution, by default (3, 3)
    basis_type : str, optional
        Filter basis type, by default "morlet"
    activation : nn.Module, optional
        Activation function, by default nn.ReLU
    transform_skip : bool, optional
        Whether to transform skip connection, by default False
    drop_conv_rate : float, optional
        Dropout rate for convolutions, by default 0.0
    drop_path_rate : float, optional
        Drop path rate, by default 0.0
    drop_dense_rate : float, optional
        Dropout rate for dense layers, by default 0.0
    downsampling_mode : str, optional
        Downsampling mode ("bilinear", "conv"), by default "bilinear"
    """
    
    def __init__(
        self,
        in_shape,
        out_shape,
        in_channels,
        out_channels,
        grid_in="equiangular",
        grid_out="equiangular",
        nrep=1,
        kernel_shape=(3, 3),
        basis_type="morlet",
        activation=nn.ReLU,
        transform_skip=False,
        drop_conv_rate=0.0,
        drop_path_rate=0.0,
        drop_dense_rate=0.0,
        downsampling_mode="bilinear",
    ):
        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_in = grid_in
        self.grid_out = grid_out

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.fwd = []
        for i in range(nrep):
            # conv
            theta_cutoff = _compute_cutoff_radius(in_shape[0], kernel_shape, basis_type)
            self.fwd.append(
                DiscreteContinuousConvS2(
                    in_channels=(in_channels if i == 0 else out_channels),
                    out_channels=out_channels,
                    in_shape=in_shape,
                    out_shape=in_shape,
                    kernel_shape=kernel_shape,
                    basis_type=basis_type,
                    grid_in=grid_out,
                    grid_out=grid_out,
                    bias=False,
                    theta_cutoff=theta_cutoff,
                )
            )

            if drop_conv_rate > 0.0:
                self.fwd.append(nn.Dropout2d(p=drop_conv_rate))

            # batchnorm
            self.fwd.append(nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

            # activation
            self.fwd.append(
                activation(),
            )

        if downsampling_mode == "conv":
            theta_cutoff = _compute_cutoff_radius(out_shape[0], kernel_shape, basis_type)
            self.downsample = DiscreteContinuousConvS2(
                out_channels,
                out_channels,
                in_shape=in_shape,
                out_shape=out_shape,
                kernel_shape=kernel_shape,
                basis_type=basis_type,
                grid_in=grid_in,
                grid_out=grid_out,
                bias=False,
                theta_cutoff=theta_cutoff,
            )
        else:
            self.downsample = ResampleS2(
                nlat_in=in_shape[0],
                nlon_in=in_shape[1],
                nlat_out=out_shape[0],
                nlon_out=out_shape[1],
                grid_in=grid_in,
                grid_out=grid_out,
                mode=downsampling_mode,
            )

        # make sequential
        self.fwd = nn.Sequential(*self.fwd)

        # final norm
        if transform_skip or (in_channels != out_channels):
            self.transform_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

            if drop_dense_rate > 0.0:
                self.transform_skip = nn.Sequential(
                    self.transform_skip,
                    nn.Dropout2d(p=drop_dense_rate),
                )

        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # skip connection
        residual = x
        if hasattr(self, "transform_skip"):
            residual = self.transform_skip(residual)

        # main path
        x = self.fwd(x)

        # add residual connection
        x = residual + self.drop_path(x)

        # downsample
        x = self.downsample(x)

        return x


class UpsamplingBlock(nn.Module):
    """
    Upsampling block for spherical U-Net architecture.
    
    This block performs upsampling followed by convolution operations on spherical data,
    using discrete-continuous convolutions to maintain spectral properties.
    
    Parameters
    -----------
    in_shape : tuple
        Input shape (nlat, nlon)
    out_shape : tuple
        Output shape (nlat, nlon)
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    grid_in : str, optional
        Input grid type, by default "equiangular"
    grid_out : str, optional
        Output grid type, by default "equiangular"
    nrep : int, optional
        Number of convolution repetitions, by default 1
    kernel_shape : tuple, optional
        Kernel shape for convolution, by default (3, 3)
    basis_type : str, optional
        Filter basis type, by default "morlet"
    activation : nn.Module, optional
        Activation function, by default nn.ReLU
    transform_skip : bool, optional
        Whether to transform skip connection, by default False
    drop_conv_rate : float, optional
        Dropout rate for convolutions, by default 0.0
    drop_path_rate : float, optional
        Drop path rate, by default 0.0
    drop_dense_rate : float, optional
        Dropout rate for dense layers, by default 0.0
    upsampling_mode : str, optional
        Upsampling mode ("bilinear", "conv"), by default "bilinear"
    """

    def __init__(
        self,
        in_shape,
        out_shape,
        in_channels,
        out_channels,
        grid_in="equiangular",
        grid_out="equiangular",
        nrep=1,
        kernel_shape=(3, 3),
        basis_type="morlet",
        activation=nn.ReLU,
        transform_skip=False,
        drop_conv_rate=0.0,
        drop_path_rate=0.0,
        drop_dense_rate=0.0,
        upsampling_mode="bilinear",
    ):
        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        if in_shape != out_shape:
            if upsampling_mode == "conv":
                theta_cutoff = _compute_cutoff_radius(in_shape[0], kernel_shape, basis_type)
                self.upsample = nn.Sequential(
                    DiscreteContinuousConvTransposeS2(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        in_shape=in_shape,
                        out_shape=out_shape,
                        kernel_shape=kernel_shape,
                        basis_type=basis_type,
                        grid_in=grid_in,
                        grid_out=grid_out,
                        bias=False,
                        theta_cutoff=theta_cutoff,
                    ),
                    nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    activation(),
                    DiscreteContinuousConvS2(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        in_shape=out_shape,
                        out_shape=out_shape,
                        kernel_shape=kernel_shape,
                        basis_type=basis_type,
                        grid_in=grid_in,
                        grid_out=grid_out,
                        bias=False,
                        theta_cutoff=theta_cutoff,
                    ),
                )

            else:
                self.upsample = ResampleS2(
                    nlat_in=in_shape[0],
                    nlon_in=in_shape[1],
                    nlat_out=out_shape[0],
                    nlon_out=out_shape[1],
                    grid_in=grid_in,
                    grid_out=grid_out,
                    mode=upsampling_mode,
                )
        else:
            theta_cutoff = _compute_cutoff_radius(in_shape[0], kernel_shape, basis_type)
            self.upsample = DiscreteContinuousConvS2(
                in_channels=out_channels,
                out_channels=out_channels,
                in_shape=in_shape,
                out_shape=in_shape,
                kernel_shape=kernel_shape,
                basis_type=basis_type,
                grid_in=grid_in,
                grid_out=grid_out,
                bias=False,
                theta_cutoff=theta_cutoff,
            )

        self.fwd = []
        for i in range(nrep):
            # conv
            theta_cutoff = _compute_cutoff_radius(in_shape[0], kernel_shape, basis_type)
            self.fwd.append(
                DiscreteContinuousConvS2(
                    in_channels=in_channels,
                    out_channels=(out_channels if i == nrep - 1 else in_channels),
                    in_shape=in_shape,
                    out_shape=in_shape,
                    kernel_shape=kernel_shape,
                    basis_type=basis_type,
                    grid_in=grid_in,
                    grid_out=grid_in,
                    bias=False,
                    theta_cutoff=theta_cutoff,
                )
            )

            if drop_conv_rate > 0.0:
                self.fwd.append(nn.Dropout2d(p=drop_conv_rate))

            # batchnorm
            self.fwd.append(nn.BatchNorm2d((out_channels if i == nrep - 1 else in_channels), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

            # activation
            self.fwd.append(
                activation(),
            )

        # make sequential
        self.fwd = nn.Sequential(*self.fwd)

        # final norm
        if transform_skip or (in_channels != out_channels):
            self.transform_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
            if drop_dense_rate > 0.0:
                self.transform_skip = nn.Sequential(
                    self.transform_skip,
                    nn.Dropout2d(p=drop_dense_rate),
                )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # skip connection
        residual = x
        if hasattr(self, "transform_skip"):
            residual = self.transform_skip(residual)

        # main path
        x = residual + self.drop_path(self.fwd(x))

        # upsampling
        x = self.upsample(x)

        return x


class SphericalUNet(nn.Module):
    """
    Spherical U-Net model designed to approximate mappings from spherical signals to spherical segmentation masks

    Parameters
    -----------
    img_size : tuple, optional
        Shape of the input channels, by default (128, 256)
    grid : str, optional
        Grid type for input/output, by default "equiangular"
    grid_internal : str, optional
        Grid type for internal processing, by default "legendre-gauss"
    in_chans : int, optional
        Number of input channels, by default 3
    out_chans : int, optional
        Number of classes, by default 3
    embed_dims : List[int], optional
        Dimension of the embeddings for each block, has to be the same length as depths
    depths : List[int], optional
        Number of repetitions of conv blocks and ffn mixers per layer. Has to be the same length as embed_dims
    scale_factor : int, optional
        Scale factor to use, by default 2
    activation_function : str, optional
        Activation function to use, by default "relu"
    kernel_shape : tuple, optional
        Kernel shape for convolutions, by default (3, 3)
    filter_basis_type : str, optional
        Filter basis type, by default "morlet"
    transform_skip : bool, optional
        Whether to transform skip connection, by default False
    drop_conv_rate : float, optional
        Dropout rate for convolutions, by default 0.1
    drop_path_rate : float, optional
        Dropout path rate, by default 0.1
    drop_dense_rate : float, optional
        Dropout rate for dense layers, by default 0.5
    downsampling_mode : str, optional
        Downsampling mode ("bilinear", "conv"), by default "bilinear"
    upsampling_mode : str, optional
        Upsampling mode ("bilinear", "conv"), by default "bilinear"

    Example
    -----------
    >>> model = SphericalUNet(
    ...         img_size=(128, 256),
    ...         scale_factor=4,
    ...         in_chans=2,
    ...         out_chans=2,
    ...         embed_dims=[16, 32, 64, 128],
    ...         depths=[2, 2, 2, 2],
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
        depths=[2, 2, 2, 2],
        scale_factor=2,
        activation_function="relu",
        kernel_shape=(3, 3),
        filter_basis_type="morlet",
        transform_skip=False,
        drop_conv_rate=0.1,
        drop_path_rate=0.1,
        drop_dense_rate=0.5,
        downsampling_mode="bilinear",
        upsampling_mode="bilinear",
    ):
        super().__init__()

        self.img_size = img_size
        self.grid = grid
        self.grid_internal = grid_internal
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dims = embed_dims
        self.num_blocks = len(self.embed_dims)
        self.depths = depths
        self.kernel_shape = kernel_shape

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
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_blocks)]

        self.dblocks = nn.ModuleList([])
        out_shape = img_size
        grid_in = grid
        grid_out = grid_internal
        in_channels = in_chans
        for i in range(self.num_blocks):
            out_shape_new = (out_shape[0] // scale_factor, out_shape[1] // scale_factor)
            out_channels = self.embed_dims[i]
            self.dblocks.append(
                DownsamplingBlock(
                    in_shape=out_shape,
                    out_shape=out_shape_new,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    grid_in=grid_in,
                    grid_out=grid_out,
                    nrep=self.depths[i],
                    kernel_shape=kernel_shape,
                    basis_type=filter_basis_type,
                    activation=self.activation_function,
                    drop_conv_rate=drop_conv_rate,
                    drop_path_rate=dpr[i],
                    drop_dense_rate=drop_dense_rate,
                    transform_skip=transform_skip,
                    downsampling_mode=downsampling_mode,
                )
            )
            out_shape = out_shape_new
            grid_in = grid_internal
            in_channels = out_channels

        self.ublocks = nn.ModuleList([])
        for i in range(self.num_blocks - 1, -1, -1):
            in_shape = self.dblocks[i].out_shape
            out_shape = self.dblocks[i].in_shape
            in_channels = self.dblocks[i].out_channels
            if i != self.num_blocks - 1:
                in_channels = 2 * in_channels
            out_channels = self.dblocks[i].in_channels
            if i == 0:
                out_channels = self.embed_dims[0]
            grid_in = self.dblocks[i].grid_out
            grid_out = self.dblocks[i].grid_in
            self.ublocks.append(
                UpsamplingBlock(
                    in_shape=in_shape,
                    out_shape=out_shape,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    grid_in=grid_in,
                    grid_out=grid_out,
                    kernel_shape=kernel_shape,
                    basis_type=filter_basis_type,
                    activation=self.activation_function,
                    drop_conv_rate=drop_conv_rate,
                    drop_path_rate=0.0,
                    drop_dense_rate=drop_dense_rate,
                    transform_skip=transform_skip,
                    upsampling_mode=upsampling_mode,
                )
            )

        self.head = nn.Conv2d(self.embed_dims[0], self.out_chans, kernel_size=1, bias=True)

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
        for dblock in self.dblocks:
            feat = dblock(feat)
            features.append(feat)

        # reverse list
        features = features[::-1]

        # perform upsample
        ufeat = self.ublocks[0](features[0])
        for feat, ublock in zip(features[1:], self.ublocks[1:]):
            ufeat = ublock(torch.cat([feat, ufeat], dim=1))

        # last layer
        out = self.head(ufeat)

        return out
