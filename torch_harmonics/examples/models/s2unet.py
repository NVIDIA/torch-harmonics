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

from torch_harmonics import DiscreteContinuousConvS2, DiscreteContinuousConvTransposeS2, ResampleS2
from torch_harmonics.examples.models._layers import DropPath
from torch_harmonics.grid import require_regular_grid


# heuristic for finding theta_cutoff
def _compute_cutoff_radius(nlat, kernel_shape, basis_type):
    # "morlet" is the deprecated unnormalized alias of "harmonic", with the same support
    theta_cutoff_factor = {"piecewise linear": 0.5, "harmonic": 0.5, "morlet": 0.5, "zernike": math.sqrt(2.0)}

    return (kernel_shape[0] + 1) * theta_cutoff_factor[basis_type] * math.pi / float(nlat - 1)


class DownsamplingBlock(nn.Module):
    """
    Downsampling block for spherical U-Net architecture.

    This block performs convolution operations followed by downsampling on spherical data,
    using discrete-continuous convolutions to maintain spectral properties.

    Parameters
    ----------
    grid_in : RegularGridS2
        Grid of the block's input.
    grid_out : RegularGridS2
        Grid of the block's output.
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    nrep : int, optional
        Number of convolution repetitions, by default 1
    kernel_shape : tuple, optional
        Kernel shape for convolution, by default (3, 3)
    basis_type : str, optional
        Filter basis type, by default "harmonic"
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
        grid_in,
        grid_out,
        in_channels,
        out_channels,
        nrep=1,
        kernel_shape=(3, 3),
        basis_type="harmonic",
        activation=nn.ReLU,
        transform_skip=False,
        drop_conv_rate=0.0,
        drop_path_rate=0.0,
        drop_dense_rate=0.0,
        downsampling_mode="bilinear",
    ):
        super().__init__()

        self.grid_in = grid_in
        self.grid_out = grid_out
        self.in_shape = grid_in.shape
        self.out_shape = grid_out.shape
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.fwd = []
        for i in range(nrep):
            # conv, on the grid the input lives on
            theta_cutoff = _compute_cutoff_radius(grid_in.nlat, kernel_shape, basis_type)
            self.fwd.append(
                DiscreteContinuousConvS2(
                    grid_in=grid_in,
                    grid_out=grid_in,
                    in_channels=(in_channels if i == 0 else out_channels),
                    out_channels=out_channels,
                    kernel_shape=kernel_shape,
                    basis_type=basis_type,
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
            theta_cutoff = _compute_cutoff_radius(grid_out.nlat, kernel_shape, basis_type)
            self.downsample = DiscreteContinuousConvS2(
                grid_in,
                grid_out,
                out_channels,
                out_channels,
                kernel_shape=kernel_shape,
                basis_type=basis_type,
                bias=False,
                theta_cutoff=theta_cutoff,
            )
        else:
            self.downsample = ResampleS2(
                grid_in,
                grid_out,
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
    ----------
    grid_in : RegularGridS2
        Grid of the block's input.
    grid_out : RegularGridS2
        Grid of the block's output.
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    nrep : int, optional
        Number of convolution repetitions, by default 1
    kernel_shape : tuple, optional
        Kernel shape for convolution, by default (3, 3)
    basis_type : str, optional
        Filter basis type, by default "harmonic"
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
        grid_in,
        grid_out,
        in_channels,
        out_channels,
        nrep=1,
        kernel_shape=(3, 3),
        basis_type="harmonic",
        activation=nn.ReLU,
        transform_skip=False,
        drop_conv_rate=0.0,
        drop_path_rate=0.0,
        drop_dense_rate=0.0,
        upsampling_mode="bilinear",
    ):
        super().__init__()

        self.grid_in = grid_in
        self.grid_out = grid_out
        self.in_shape = grid_in.shape
        self.out_shape = grid_out.shape
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        if grid_in.shape != grid_out.shape:
            if upsampling_mode == "conv":
                theta_cutoff = _compute_cutoff_radius(grid_in.nlat, kernel_shape, basis_type)
                self.upsample = nn.Sequential(
                    DiscreteContinuousConvTransposeS2(
                        grid_in=grid_in,
                        grid_out=grid_out,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_shape=kernel_shape,
                        basis_type=basis_type,
                        bias=False,
                        theta_cutoff=theta_cutoff,
                    ),
                    nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    activation(),
                    # after the upsampling the data lives on grid_out
                    DiscreteContinuousConvS2(
                        grid_in=grid_out,
                        grid_out=grid_out,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_shape=kernel_shape,
                        basis_type=basis_type,
                        bias=False,
                        theta_cutoff=theta_cutoff,
                    ),
                )

            else:
                self.upsample = ResampleS2(
                    grid_in,
                    grid_out,
                    mode=upsampling_mode,
                )
        else:
            theta_cutoff = _compute_cutoff_radius(grid_in.nlat, kernel_shape, basis_type)
            self.upsample = DiscreteContinuousConvS2(
                grid_in=grid_in,
                grid_out=grid_out,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_shape=kernel_shape,
                basis_type=basis_type,
                bias=False,
                theta_cutoff=theta_cutoff,
            )

        self.fwd = []
        for i in range(nrep):
            # conv
            theta_cutoff = _compute_cutoff_radius(grid_in.nlat, kernel_shape, basis_type)
            self.fwd.append(
                DiscreteContinuousConvS2(
                    grid_in=grid_in,
                    grid_out=grid_in,
                    in_channels=in_channels,
                    out_channels=(out_channels if i == nrep - 1 else in_channels),
                    kernel_shape=kernel_shape,
                    basis_type=basis_type,
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
    ----------
    grid : RegularGridS2
        Grid the input and output fields live on, e.g.
        ``as_grid("equiangular", nlat=128, nlon=256)``.
    grids_internal : Sequence[RegularGridS2]
        One grid per stage, in order from the finest to the coarsest; stage ``i``
        downsamples onto ``grids_internal[i]`` and the matching upsampling stage
        returns from it. Must have the same length as ``embed_dims``.
    in_chans : int, optional
        Number of input channels, by default 3
    out_chans : int, optional
        Number of classes, by default 3
    embed_dims : List[int], optional
        Dimension of the embeddings for each block, has to be the same length as depths
    depths : List[int], optional
        Number of repetitions of conv blocks and ffn mixers per layer. Has to be the same length as embed_dims
    activation_function : str, optional
        Activation function to use, by default "relu"
    kernel_shape : tuple, optional
        Kernel shape for convolutions, by default (3, 3)
    filter_basis_type : str, optional
        Filter basis type, by default "harmonic"
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

    Examples
    --------
    >>> from torch_harmonics import as_grid
    >>> model = SphericalUNet(
    ...         grid=as_grid("equiangular", nlat=128, nlon=256),
    ...         grids_internal=[as_grid("legendre-gauss", nlat=128 // 2**i, nlon=256 // 2**i) for i in range(1, 5)],
    ...         in_chans=2,
    ...         out_chans=2,
    ...         embed_dims=[16, 32, 64, 128],
    ...         depths=[2, 2, 2, 2],)
    >>> model(torch.randn(1, 2, 128, 256)).shape
    torch.Size([1, 2, 128, 256])
    """

    def __init__(
        self,
        grid,
        grids_internal,
        in_chans=3,
        out_chans=3,
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2, 2, 2],
        activation_function="relu",
        kernel_shape=(3, 3),
        filter_basis_type="harmonic",
        transform_skip=False,
        drop_conv_rate=0.1,
        drop_path_rate=0.1,
        drop_dense_rate=0.5,
        downsampling_mode="bilinear",
        upsampling_mode="bilinear",
    ):
        super().__init__()

        self.grid = require_regular_grid(grid, "grid")
        self.grids_internal = [require_regular_grid(g, f"grids_internal[{i}]") for i, g in enumerate(grids_internal)]
        self.img_size = self.grid.shape
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dims = embed_dims
        self.num_blocks = len(self.embed_dims)
        self.depths = depths
        self.kernel_shape = kernel_shape

        if len(self.depths) != self.num_blocks:
            raise ValueError(f"depths must have length num_blocks={self.num_blocks}, got {len(self.depths)}")
        if len(self.grids_internal) != self.num_blocks:
            raise ValueError(f"grids_internal must have one grid per stage, num_blocks={self.num_blocks}, got {len(self.grids_internal)}")

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
        grid_in = self.grid
        in_channels = in_chans
        for i in range(self.num_blocks):
            grid_out = self.grids_internal[i]
            out_channels = self.embed_dims[i]
            self.dblocks.append(
                DownsamplingBlock(
                    grid_in=grid_in,
                    grid_out=grid_out,
                    in_channels=in_channels,
                    out_channels=out_channels,
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
            grid_in = grid_out
            in_channels = out_channels

        self.ublocks = nn.ModuleList([])
        for i in range(self.num_blocks - 1, -1, -1):
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
                    grid_in=grid_in,
                    grid_out=grid_out,
                    in_channels=in_channels,
                    out_channels=out_channels,
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
