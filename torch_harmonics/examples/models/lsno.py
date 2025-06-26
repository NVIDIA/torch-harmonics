# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2024 The torch-harmonics Authors. All rights reserved.
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

from torch_harmonics import RealSHT, InverseRealSHT
from torch_harmonics import DiscreteContinuousConvS2, DiscreteContinuousConvTransposeS2
from torch_harmonics import ResampleS2

from torch_harmonics.examples.models._layers import MLP, SpectralConvS2, SequencePositionEmbedding, SpectralPositionEmbedding, LearnablePositionEmbedding

from functools import partial

# heuristic for finding theta_cutoff
def _compute_cutoff_radius(nlat, kernel_shape, basis_type):
    theta_cutoff_factor = {"piecewise linear": 0.5, "morlet": 0.5, "zernike": math.sqrt(2.0)}

    return (kernel_shape[0] + 1) * theta_cutoff_factor[basis_type] * math.pi / float(nlat - 1)

class DiscreteContinuousEncoder(nn.Module):
    r"""
    Discrete-continuous encoder for spherical neural operators.
    
    This module performs downsampling using discrete-continuous convolutions on the sphere,
    reducing the spatial resolution while maintaining the spectral properties of the data.
    
    Parameters
    -----------
    in_shape : tuple, optional
        Input shape (nlat, nlon), by default (721, 1440)
    out_shape : tuple, optional
        Output shape (nlat, nlon), by default (480, 960)
    grid_in : str, optional
        Input grid type, by default "equiangular"
    grid_out : str, optional
        Output grid type, by default "equiangular"
    inp_chans : int, optional
        Number of input channels, by default 2
    out_chans : int, optional
        Number of output channels, by default 2
    kernel_shape : tuple, optional
        Kernel shape for convolution, by default (3, 3)
    basis_type : str, optional
        Filter basis type, by default "morlet"
    groups : int, optional
        Number of groups for grouped convolution, by default 1
    bias : bool, optional
        Whether to use bias, by default False
    """
    def __init__(
        self,
        in_shape=(721, 1440),
        out_shape=(480, 960),
        grid_in="equiangular",
        grid_out="equiangular",
        inp_chans=2,
        out_chans=2,
        kernel_shape=(3, 3),
        basis_type="morlet",
        groups=1,
        bias=False,
    ):
        super().__init__()

        # set up local convolution
        self.conv = DiscreteContinuousConvS2(
            inp_chans,
            out_chans,
            in_shape=in_shape,
            out_shape=out_shape,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            grid_in=grid_in,
            grid_out=grid_out,
            groups=groups,
            bias=bias,
            theta_cutoff=_compute_cutoff_radius(in_shape[0], kernel_shape, basis_type),
        )

    def forward(self, x):
        """
        Forward pass of the discrete-continuous encoder.
        
        Parameters
        -----------
        x : torch.Tensor
            Input tensor with shape (batch, channels, nlat, nlon)
            
        Returns
        -------
        torch.Tensor
            Encoded tensor with reduced spatial resolution
        """
        dtype = x.dtype

        with amp.autocast(device_type="cuda", enabled=False):
            x = x.float()
            x = self.conv(x)
            x = x.to(dtype=dtype)

        return x


class DiscreteContinuousDecoder(nn.Module):
    r"""
    Discrete-continuous decoder for spherical neural operators.
    
    This module performs upsampling using either spherical harmonic transforms or resampling,
    followed by discrete-continuous convolutions to restore spatial resolution.
    
    Parameters
    -----------
    in_shape : tuple, optional
        Input shape (nlat, nlon), by default (480, 960)
    out_shape : tuple, optional
        Output shape (nlat, nlon), by default (721, 1440)
    grid_in : str, optional
        Input grid type, by default "equiangular"
    grid_out : str, optional
        Output grid type, by default "equiangular"
    inp_chans : int, optional
        Number of input channels, by default 2
    out_chans : int, optional
        Number of output channels, by default 2
    kernel_shape : tuple, optional
        Kernel shape for convolution, by default (3, 3)
    basis_type : str, optional
        Filter basis type, by default "morlet"
    groups : int, optional
        Number of groups for grouped convolution, by default 1
    bias : bool, optional
        Whether to use bias, by default False
    upsample_sht : bool, optional
        Whether to use SHT for upsampling, by default False
    """
    def __init__(
        self,
        in_shape=(480, 960),
        out_shape=(721, 1440),
        grid_in="equiangular",
        grid_out="equiangular",
        inp_chans=2,
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
            inp_chans,
            out_chans,
            in_shape=out_shape,
            out_shape=out_shape,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            grid_in=grid_out,
            grid_out=grid_out,
            groups=groups,
            bias=False,
            theta_cutoff=_compute_cutoff_radius(in_shape[0], kernel_shape, basis_type),
        )

    def forward(self, x):
        """
        Forward pass of the discrete-continuous decoder.
        
        Parameters
        -----------
        x : torch.Tensor
            Input tensor with shape (batch, channels, nlat, nlon)
            
        Returns
        -------
        torch.Tensor
            Decoded tensor with restored spatial resolution
        """
        dtype = x.dtype

        with amp.autocast(device_type="cuda", enabled=False):
            x = x.float()
            x = self.upsample(x)
            x = self.conv(x)
            x = x.to(dtype=dtype)

        return x


class SphericalNeuralOperatorBlock(nn.Module):
    """
    Helper module for a single SFNO/FNO block. Can use both FFTs and SHTs to represent either FNO or SFNO blocks.
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        input_dim,
        output_dim,
        conv_type="local",
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer="none",
        inner_skip="none",
        outer_skip="identity",
        use_mlp=True,
        disco_kernel_shape=(3, 3),
        disco_basis_type="morlet",
        bias=False,
    ):
        super().__init__()

        if act_layer == nn.Identity:
            gain_factor = 1.0
        else:
            gain_factor = 2.0

        if inner_skip == "linear" or inner_skip == "identity":
            gain_factor /= 2.0

        # convolution layer
        if conv_type == "local":
            theta_cutoff = 2.0 * _compute_cutoff_radius(forward_transform.nlat, disco_kernel_shape, disco_basis_type)
            self.local_conv = DiscreteContinuousConvS2(
                input_dim,
                output_dim,
                in_shape=(forward_transform.nlat, forward_transform.nlon),
                out_shape=(inverse_transform.nlat, inverse_transform.nlon),
                kernel_shape=disco_kernel_shape,
                basis_type=disco_basis_type,
                grid_in=forward_transform.grid,
                grid_out=inverse_transform.grid,
                bias=bias,
                theta_cutoff=theta_cutoff,
            )
        elif conv_type == "global":
            self.global_conv = SpectralConvS2(forward_transform, inverse_transform, input_dim, output_dim, gain=gain_factor, bias=bias)
        else:
            raise ValueError(f"Unknown convolution type {conv_type}")

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
            raise NotImplementedError(f"Error, normalization {norm_layer} not implemented.")

        # dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        gain_factor = 1.0
        if outer_skip == "linear" or inner_skip == "identity":
            gain_factor /= 2.0

        if use_mlp == True:
            mlp_hidden_dim = int(output_dim * mlp_ratio)
            self.mlp = MLP(
                in_features=output_dim,
                out_features=input_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop_rate=drop_rate,
                checkpointing=False,
                gain=gain_factor,
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

        residual = x

        if hasattr(self, "global_conv"):
            x, _ = self.global_conv(x)
        elif hasattr(self, "local_conv"):
            x = self.local_conv(x)

        x = self.norm(x)

        if hasattr(self, "inner_skip"):
            x = x + self.inner_skip(residual)

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        x = self.drop_path(x)

        if hasattr(self, "outer_skip"):
            x = x + self.outer_skip(residual)

        return x


class LocalSphericalNeuralOperator(nn.Module):
    r"""
    LocalSphericalNeuralOperator module. A spherical neural operator which uses both local and global integral
    operators to accureately model both types of solution operators [1]. The architecture is based on the Spherical
    Fourier Neural Operator [2] and improves upon it with local integral operators in both the Neural Operator blocks,
    as well as in the encoder and decoders.
    
    Parameters
    -----------
    img_size : tuple, optional
        Input image size (nlat, nlon), by default (128, 256)
    grid : str, optional
        Grid type for input/output, by default "equiangular"
    grid_internal : str, optional
        Grid type for internal processing, by default "legendre-gauss"
    scale_factor : int, optional
        Scale factor for resolution changes, by default 3
    in_chans : int, optional
        Number of input channels, by default 3
    out_chans : int, optional
        Number of output channels, by default 3
    embed_dim : int, optional
        Embedding dimension, by default 256
    num_layers : int, optional
        Number of layers, by default 4
    activation_function : str, optional
        Activation function name, by default "gelu"
    kernel_shape : tuple, optional
        Kernel shape for convolutions, by default (3, 3)
    encoder_kernel_shape : tuple, optional
        Kernel shape for encoder, by default (3, 3)
    filter_basis_type : str, optional
        Filter basis type, by default "morlet"
    use_mlp : bool, optional
        Whether to use MLP layers, by default True
    mlp_ratio : float, optional
        MLP expansion ratio, by default 2.0
    drop_rate : float, optional
        Dropout rate, by default 0.0
    drop_path_rate : float, optional
        Drop path rate, by default 0.0
    normalization_layer : str, optional
        Type of normalization layer to use ("layer_norm", "instance_norm", "none"), by default "instance_norm"
    sfno_block_frequency : int, optional
        Frequency of SFNO blocks, by default 2
    hard_thresholding_fraction : float, optional
        Hard thresholding fraction, by default 1.0
    residual_prediction : bool, optional
        Whether to use residual prediction, by default False
    pos_embed : str, optional
        Position embedding type, by default "none"
    upsample_sht : bool, optional
        Use SHT upsampling if true, else linear interpolation
    bias : bool, optional
        Whether to use a bias, by default False

    Example
    -----------
    >>> model = LocalSphericalNeuralOperator(
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
    .. [1] Liu-Schiaffini M., Berner J., Bonev B., Kurth T., Azizzadenesheli K., Anandkumar A.;
        "Neural Operators with Localized Integral and Differential Kernels" (2024).
        ICML 2024, https://arxiv.org/pdf/2402.16845.

    .. [2] Bonev B., Kurth T., Hundt C., Pathak, J., Baust M., Kashinath K., Anandkumar A.;
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
        kernel_shape=(3, 3),
        encoder_kernel_shape=(3, 3),
        filter_basis_type="morlet",
        use_mlp=True,
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        normalization_layer="none",
        sfno_block_frequency=2,
        hard_thresholding_fraction=1.0,
        residual_prediction=False,
        pos_embed="none",
        upsample_sht=False,
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

        # encoder
        self.encoder = DiscreteContinuousEncoder(
            in_shape=self.img_size,
            out_shape=(self.h, self.w),
            grid_in=grid,
            grid_out=grid_internal,
            inp_chans=self.in_chans,
            out_chans=self.embed_dim,
            kernel_shape=self.encoder_kernel_shape,
            basis_type=filter_basis_type,
            groups=1,
            bias=False,
        )

        # compute the modes for the sht
        modes_lat = self.h
        # due to some spectral artifacts with cufft, we substract one mode here
        modes_lon = (self.w // 2 + 1) - 1

        modes_lat = modes_lon = int(min(modes_lat, modes_lon) * self.hard_thresholding_fraction)

        self.trans = RealSHT(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=grid_internal).float()
        self.itrans = InverseRealSHT(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=grid_internal).float()

        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):

            block = SphericalNeuralOperatorBlock(
                self.trans,
                self.itrans,
                self.embed_dim,
                self.embed_dim,
                conv_type="global" if i % sfno_block_frequency == (sfno_block_frequency-1) else "local",
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path=dpr[i],
                act_layer=self.activation_function,
                norm_layer=self.normalization_layer,
                use_mlp=use_mlp,
                disco_kernel_shape=kernel_shape,
                disco_basis_type=filter_basis_type,
                bias=bias,
            )

            self.blocks.append(block)

        # decoder
        self.decoder = DiscreteContinuousDecoder(
            in_shape=(self.h, self.w),
            out_shape=self.img_size,
            grid_in=grid_internal,
            grid_out=grid,
            inp_chans=self.embed_dim,
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
        """
        Forward pass through the complete LSNO model.
        
        Parameters
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_chans, height, width)
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_chans, height, width)
        """
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
