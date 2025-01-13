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

import torch
import torch.nn as nn
import torch.amp as amp

from torch_harmonics import RealSHT, InverseRealSHT
from torch_harmonics import DiscreteContinuousConvS2, DiscreteContinuousConvTransposeS2
from torch_harmonics import ResampleS2

from ._layers import *

from functools import partial


class DiscreteContinuousEncoder(nn.Module):
    def __init__(
        self,
        in_shape=(721, 1440),
        out_shape=(480, 960),
        grid_in="equiangular",
        grid_out="equiangular",
        inp_chans=2,
        out_chans=2,
        kernel_shape=[3, 4],
        basis_type="piecewise linear",
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
        inp_chans=2,
        out_chans=2,
        kernel_shape=[3, 4],
        basis_type="piecewise linear",
        groups=1,
        bias=False,
        upsample_sht=False
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
        operator_type="driscoll-healy",
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.ReLU,
        norm_layer=nn.Identity,
        inner_skip="None",
        outer_skip="linear",
        use_mlp=True,
        disco_kernel_shape=[3, 4],
        disco_basis_type="piecewise linear",
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
            self.local_conv = DiscreteContinuousConvS2(
                input_dim,
                output_dim,
                in_shape=(forward_transform.nlat, forward_transform.nlon),
                out_shape=(inverse_transform.nlat, inverse_transform.nlon),
                kernel_shape=disco_kernel_shape,
                basis_type=disco_basis_type,
                grid_in=forward_transform.grid,
                grid_out=inverse_transform.grid,
                bias=False,
                theta_cutoff=4.0 * (disco_kernel_shape[0] + 1) * torch.pi / float(inverse_transform.nlat - 1),
            )
        elif conv_type == "global":
            self.global_conv = SpectralConvS2(forward_transform, inverse_transform, input_dim, output_dim, gain=gain_factor, operator_type=operator_type, bias=False)
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

        # first normalisation layer
        self.norm0 = norm_layer()

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

        # second normalisation layer
        self.norm1 = norm_layer()

    def forward(self, x):

        residual = x

        if hasattr(self, "global_conv"):
            x, _ = self.global_conv(x)
        elif hasattr(self, "local_conv"):
            x = self.local_conv(x)

        x = self.norm0(x)

        if hasattr(self, "inner_skip"):
            x = x + self.inner_skip(residual)

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        x = self.norm1(x)

        x = self.drop_path(x)

        if hasattr(self, "outer_skip"):
            x = x + self.outer_skip(residual)

        return x


class LocalSphericalNeuralOperatorNet(nn.Module):
    """
    LocalSphericalNeuralOperator module. A spherical neural operator which uses both local and global integral
    operators to accureately model both types of solution operators [1]. The architecture is based on the Spherical
    Fourier Neural Operator [2] and improves upon it with local integral operators in both the Neural Operator blocks,
    as well as in the encoder and decoders.

    Parameters
    -----------
    img_shape : tuple, optional
        Shape of the input channels, by default (128, 256)
    operator_type : str, optional
        Type of operator to use ('driscoll-healy', 'diagonal'), by default "driscoll-healy"
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
    big_skip : bool, optional
        Whether to add a single large skip connection, by default True
    pos_embed : bool, optional
        Whether to use positional embedding, by default True
    upsample_sht : bool, optional
        Use SHT upsampling if true, else linear interpolation

    Example
    -----------
    >>> model = SphericalFourierNeuralOperatorNet(
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
        operator_type="driscoll-healy",
        grid="equiangular",
        grid_internal="legendre-gauss",
        scale_factor=4,
        in_chans=3,
        out_chans=3,
        embed_dim=256,
        num_layers=4,
        activation_function="relu",
        kernel_shape=[3, 4],
        encoder_kernel_shape=[3, 4],
        filter_basis_type="piecewise linear",
        use_mlp=True,
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        normalization_layer="none",
        hard_thresholding_fraction=1.0,
        use_complex_kernels=True,
        big_skip=False,
        pos_embed=False,
        upsample_sht=False,
    ):
        super().__init__()

        self.operator_type = operator_type
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
        self.big_skip = big_skip

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

        # pick norm layer
        if self.normalization_layer == "layer_norm":
            norm_layer0 = partial(nn.LayerNorm, normalized_shape=(self.img_size[0], self.img_size[1]), eps=1e-6)
            norm_layer1 = partial(nn.LayerNorm, normalized_shape=(self.h, self.w), eps=1e-6)
        elif self.normalization_layer == "instance_norm":
            norm_layer0 = partial(nn.InstanceNorm2d, num_features=self.embed_dim, eps=1e-6, affine=True, track_running_stats=False)
            norm_layer1 = partial(nn.InstanceNorm2d, num_features=self.embed_dim, eps=1e-6, affine=True, track_running_stats=False)
        elif self.normalization_layer == "none":
            norm_layer0 = nn.Identity
            norm_layer1 = norm_layer0
        else:
            raise NotImplementedError(f"Error, normalization {self.normalization_layer} not implemented.")

        if pos_embed == "latlon" or pos_embed == True:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, self.h, self.w))
            nn.init.constant_(self.pos_embed, 0.0)
        elif pos_embed == "lat":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, self.h, 1))
            nn.init.constant_(self.pos_embed, 0.0)
        elif pos_embed == "const":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, 1, 1))
            nn.init.constant_(self.pos_embed, 0.0)
        else:
            self.pos_embed = None

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

        # prepare the SHT
        modes_lat = int(self.h * self.hard_thresholding_fraction)
        modes_lon = int(self.w // 2 * self.hard_thresholding_fraction)
        modes_lat = modes_lon = min(modes_lat, modes_lon)

        self.trans = RealSHT(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=grid_internal).float()
        self.itrans = InverseRealSHT(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=grid_internal).float()

        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):
            first_layer = i == 0
            last_layer = i == self.num_layers - 1

            inner_skip = "none"
            outer_skip = "identity"

            if first_layer:
                norm_layer = norm_layer1
            elif last_layer:
                norm_layer = norm_layer0
            else:
                norm_layer = norm_layer1

            block = SphericalNeuralOperatorBlock(
                self.trans,
                self.itrans,
                self.embed_dim,
                self.embed_dim,
                conv_type="global" if i % 2 == 0 else "local",
                operator_type=self.operator_type,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path=dpr[i],
                act_layer=self.activation_function,
                norm_layer=norm_layer,
                inner_skip=inner_skip,
                outer_skip=outer_skip,
                use_mlp=use_mlp,
                disco_kernel_shape=kernel_shape,
                disco_basis_type=filter_basis_type,
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
            upsample_sht=upsample_sht
        )

        # # residual prediction
        # if self.big_skip:
        #     self.residual_transform = nn.Conv2d(self.out_chans, self.in_chans, 1, bias=False)
        #     self.residual_transform.weight.is_shared_mp = ["spatial"]
        #     self.residual_transform.weight.sharded_dims_mp = [None, None, None, None]
        #     scale = math.sqrt(0.5 / self.in_chans)
        #     nn.init.normal_(self.residual_transform.weight, mean=0.0, std=scale)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x):
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):
        if self.big_skip:
            residual = x

        x = self.encoder(x)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        x = self.forward_features(x)

        x = self.decoder(x)

        if self.big_skip:
            # x = x + self.residual_transform(residual)
            x = x + residual

        return x
