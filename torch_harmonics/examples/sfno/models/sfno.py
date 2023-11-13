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

from torch_harmonics import *

from .layers import *

from functools import partial


class SpectralFilterLayer(nn.Module):
    """
    Fourier layer. Contains the convolution part of the FNO/SFNO
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        input_dim,
        output_dim,
        gain = 2.,
        operator_type = "diagonal",
        hidden_size_factor = 2,
        factorization = None,
        separable = False,
        rank = 1e-2,
        bias = True):
        super(SpectralFilterLayer, self).__init__() 

        if factorization is None:
            self.filter = SpectralConvS2(forward_transform,
                                         inverse_transform,
                                         input_dim,
                                         output_dim,
                                         gain = gain,
                                         operator_type = operator_type,
                                         bias = bias)
            
        elif factorization is not None:
            self.filter = FactorizedSpectralConvS2(forward_transform,
                                                   inverse_transform,
                                                   input_dim,
                                                   output_dim,
                                                   gain = gain,
                                                   operator_type = operator_type,
                                                   rank = rank,
                                                   factorization = factorization,
                                                   separable = separable,
                                                   bias = bias)

        else:
            raise(NotImplementedError)

    def forward(self, x):
        return self.filter(x)

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
            operator_type = "driscoll-healy",
            mlp_ratio = 2.,
            drop_rate = 0.,
            drop_path = 0.,
            act_layer = nn.ReLU,
            norm_layer = nn.Identity,
            factorization = None,
            separable = False,
            rank = 128,
            inner_skip = "linear",
            outer_skip = None,
            use_mlp = True):
        super(SphericalFourierNeuralOperatorBlock, self).__init__()

        if act_layer == nn.Identity:
            gain_factor = 1.0
        else:
            gain_factor = 2.0

        if inner_skip == "linear" or inner_skip == "identity":
            gain_factor /= 2.0
        
        # convolution layer
        self.filter = SpectralFilterLayer(forward_transform,
                                          inverse_transform,
                                          input_dim,
                                          output_dim,
                                          gain = gain_factor,
                                          operator_type = operator_type,
                                          hidden_size_factor = mlp_ratio,
                                          factorization = factorization,
                                          separable = separable,
                                          rank = rank,
                                          bias = True)

        if inner_skip == "linear":
            self.inner_skip = nn.Conv2d(input_dim, output_dim, 1, 1)
            nn.init.normal_(self.inner_skip.weight, std=math.sqrt(gain_factor/input_dim))
        elif inner_skip == "identity":
            assert input_dim == output_dim
            self.inner_skip = nn.Identity()
        elif inner_skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {inner_skip}")

        self.act_layer = act_layer()

        # first normalisation layer
        self.norm0 = norm_layer()
        
        # dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        gain_factor = 1.0
        if outer_skip == "linear" or inner_skip == "identity":
            gain_factor /= 2.
        
        if use_mlp == True:
            mlp_hidden_dim = int(output_dim * mlp_ratio)
            self.mlp = MLP(in_features = output_dim,
                           out_features = input_dim,
                           hidden_features = mlp_hidden_dim,
                           act_layer = act_layer,
                           drop_rate = drop_rate,
                           checkpointing = False,
                           gain = gain_factor)

        if outer_skip == "linear":
            self.outer_skip = nn.Conv2d(input_dim, input_dim, 1, 1)
            torch.nn.init.normal_(self.outer_skip.weight, std=math.sqrt(gain_factor/input_dim))
        elif outer_skip == "identity":
            assert input_dim == output_dim
            self.outer_skip = nn.Identity()
        elif outer_skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {outer_skip}")

        # second normalisation layer
        self.norm1 = norm_layer()

    # def init_weights(self, scale):
    #     if hasattr(self, "inner_skip") and isinstance(self.inner_skip, nn.Conv2d):
    #         gain_factor = 1.
    #         scale = (gain_factor / embed_dim)**0.5
    #         nn.init.normal_(self.inner_skip.weight, mean=0., std=scale)
    #         self.filter.filter.init_weights(scale)
    #     else:
    #         gain_factor = 2.
    #         scale = (gain_factor / embed_dim)**0.5
    #         self.filter.filter.init_weights(scale)

    def forward(self, x):

        x, residual = self.filter(x)

        x = self.norm0(x)

        if hasattr(self, "inner_skip"):
            x = x + self.inner_skip(residual)

        if hasattr(self, "act_layer"):
            x = self.act_layer(x)

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        x = self.norm1(x)

        x = self.drop_path(x)

        if hasattr(self, "outer_skip"):
            x = x + self.outer_skip(residual)

        return x

class SphericalFourierNeuralOperatorNet(nn.Module):
    """
    SphericalFourierNeuralOperator module. Can use both FFTs and SHTs to represent either FNO or SFNO,
    both linear and non-linear variants.

    Parameters
    ----------
    spectral_transform : str, optional
        Type of spectral transformation to use, by default "sht"
    operator_type : str, optional
        Type of operator to use ('driscoll-healy', 'diagonal'), by default "driscoll-healy"
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
    big_skip : bool, optional
        Whether to add a single large skip connection, by default True
    rank : float, optional
        Rank of the approximation, by default 1.0
    factorization : Any, optional
        Type of factorization to use, by default None
    separable : bool, optional
        Whether to use separable convolutions, by default False
    rank : (int, Tuple[int]), optional
        If a factorization is used, which rank to use. Argument is passed to tensorly
    pos_embed : bool, optional
        Whether to use positional embedding, by default True

    Example:
    --------
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
    """

    def __init__(
            self,
            spectral_transform = "sht",
            operator_type = "driscoll-healy",
            img_size = (128, 256),
            grid = "equiangular",
            scale_factor = 3,
            in_chans = 3,
            out_chans = 3,
            embed_dim = 256,
            num_layers = 4,
            activation_function = "relu",
            encoder_layers = 1,
            use_mlp = True,
            mlp_ratio = 2.,
            drop_rate = 0.,
            drop_path_rate = 0.,
            normalization_layer = "none",
            hard_thresholding_fraction = 1.0,
            use_complex_kernels = True,
            big_skip = False,
            factorization = None,
            separable = False,
            rank = 128,
            pos_embed = False):

        super(SphericalFourierNeuralOperatorNet, self).__init__()

        self.spectral_transform = spectral_transform
        self.operator_type = operator_type
        self.img_size = img_size
        self.grid = grid
        self.scale_factor = scale_factor
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.normalization_layer = normalization_layer
        self.use_mlp = use_mlp
        self.encoder_layers = encoder_layers
        self.big_skip = big_skip
        self.factorization = factorization
        self.separable = separable,
        self.rank = rank

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

        # compute downsampled image size
        self.h = self.img_size[0] // scale_factor
        self.w = self.img_size[1] // scale_factor

        # dropout
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0. else nn.Identity()
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

        if pos_embed == "latlon" or pos_embed==True:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, self.img_size[0], self.img_size[1]))
            nn.init.constant_(self.pos_embed, 0.0)
        elif pos_embed == "lat":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, self.img_size[0], 1))
            nn.init.constant_(self.pos_embed, 0.0)
        elif pos_embed == "const":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, 1, 1))
            nn.init.constant_(self.pos_embed, 0.0)
        else:
            self.pos_embed = None

        # # encoder
        # encoder_hidden_dim = int(self.embed_dim * mlp_ratio)
        # encoder = MLP(in_features = self.in_chans,
        #               out_features = self.embed_dim,
        #               hidden_features = encoder_hidden_dim,
        #               act_layer = self.activation_function,
        #               drop_rate = drop_rate,
        #               checkpointing = False)
        # self.encoder = encoder


        # construct an encoder with num_encoder_layers
        num_encoder_layers = 1
        encoder_hidden_dim = int(self.embed_dim * mlp_ratio)
        current_dim = self.in_chans
        encoder_layers = []
        for l in range(num_encoder_layers-1):
            fc = nn.Conv2d(current_dim, encoder_hidden_dim, 1, bias=True)
            # initialize the weights correctly
            scale = math.sqrt(2. / current_dim)
            nn.init.normal_(fc.weight, mean=0., std=scale)
            if fc.bias is not None:
                nn.init.constant_(fc.bias, 0.0)
            encoder_layers.append(fc)
            encoder_layers.append(self.activation_function())
            current_dim = encoder_hidden_dim
        fc = nn.Conv2d(current_dim, self.embed_dim, 1, bias=False)
        scale = math.sqrt(1. / current_dim)
        nn.init.normal_(fc.weight, mean=0., std=scale)
        if fc.bias is not None:
            nn.init.constant_(fc.bias, 0.0)
        encoder_layers.append(fc)
        self.encoder = nn.Sequential(*encoder_layers)
        
        # prepare the spectral transform
        if self.spectral_transform == "sht":

            modes_lat = int(self.h * self.hard_thresholding_fraction)
            modes_lon = int(self.w//2 * self.hard_thresholding_fraction)
            modes_lat = modes_lon = min(modes_lat, modes_lon)

            self.trans_down = RealSHT(*self.img_size, lmax=modes_lat, mmax=modes_lon, grid=self.grid).float()
            self.itrans_up  = InverseRealSHT(*self.img_size, lmax=modes_lat, mmax=modes_lon, grid=self.grid).float()
            self.trans      = RealSHT(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss").float()
            self.itrans     = InverseRealSHT(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss").float()

        elif self.spectral_transform == "fft":

            modes_lat = int(self.h * self.hard_thresholding_fraction)
            modes_lon = int((self.w // 2 + 1) * self.hard_thresholding_fraction)

            self.trans_down = RealFFT2(*self.img_size, lmax=modes_lat, mmax=modes_lon).float()
            self.itrans_up  = InverseRealFFT2(*self.img_size, lmax=modes_lat, mmax=modes_lon).float()
            self.trans      = RealFFT2(self.h, self.w, lmax=modes_lat, mmax=modes_lon).float()
            self.itrans     = InverseRealFFT2(self.h, self.w, lmax=modes_lat, mmax=modes_lon).float()
            
        else:
            raise(ValueError("Unknown spectral transform"))

        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):

            first_layer = i == 0
            last_layer = i == self.num_layers-1

            forward_transform = self.trans_down if first_layer else self.trans
            inverse_transform = self.itrans_up if last_layer else self.itrans

            inner_skip = "none"
            outer_skip = "identity"

            if first_layer:
                norm_layer = norm_layer1
            elif last_layer:
                norm_layer = norm_layer0
            else:
                norm_layer = norm_layer1

            block = SphericalFourierNeuralOperatorBlock(forward_transform,
                                                        inverse_transform,
                                                        self.embed_dim,
                                                        self.embed_dim,
                                                        operator_type = self.operator_type,
                                                        mlp_ratio = mlp_ratio,
                                                        drop_rate = drop_rate,
                                                        drop_path = dpr[i],
                                                        act_layer = self.activation_function,
                                                        norm_layer = norm_layer,
                                                        inner_skip = inner_skip,
                                                        outer_skip = outer_skip,
                                                        use_mlp = use_mlp,
                                                        factorization = self.factorization,
                                                        separable = self.separable,
                                                        rank = self.rank)

            self.blocks.append(block)

        # # decoder
        # decoder_hidden_dim = int(self.embed_dim * mlp_ratio)
        # self.decoder = MLP(in_features = self.embed_dim + self.big_skip*self.in_chans,
        #                    out_features = self.out_chans,
        #                    hidden_features = decoder_hidden_dim,
        #                    act_layer = self.activation_function,
        #                    drop_rate = drop_rate,
        #                    checkpointing = False)

        # construct an decoder with num_decoder_layers
        num_decoder_layers = 1
        decoder_hidden_dim = int(self.embed_dim * mlp_ratio)
        current_dim = self.embed_dim + self.big_skip*self.in_chans
        decoder_layers = []
        for l in range(num_decoder_layers-1):
            fc = nn.Conv2d(current_dim, decoder_hidden_dim, 1, bias=True)
            # initialize the weights correctly
            scale = math.sqrt(2. / current_dim)
            nn.init.normal_(fc.weight, mean=0., std=scale)
            if fc.bias is not None:
                nn.init.constant_(fc.bias, 0.0)
            decoder_layers.append(fc)
            decoder_layers.append(self.activation_function())
            current_dim = decoder_hidden_dim
        fc = nn.Conv2d(current_dim, self.out_chans, 1, bias=False)
        scale = math.sqrt(1. / current_dim)
        nn.init.normal_(fc.weight, mean=0., std=scale)
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

        if self.big_skip:
            residual = x

        x = self.encoder(x)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        x = self.forward_features(x)

        if self.big_skip:
            x = torch.cat((x, residual), dim=1)

        x = self.decoder(x)

        return x
    
    
