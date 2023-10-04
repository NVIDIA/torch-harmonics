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
        embed_dim,
        filter_type = "non-linear",
        operator_type = "diagonal",
        sparsity_threshold = 0.0,
        use_complex_kernels = True,
        hidden_size_factor = 2,
        lr_scale_exponent = 0,
        factorization = None,
        separable = False,
        rank = 1e-2,
        complex_activation = "real",
        spectral_layers = 1,
        drop_rate = 0):
        super(SpectralFilterLayer, self).__init__() 

        if filter_type == "non-linear" and isinstance(forward_transform, RealSHT):
            self.filter = SpectralAttentionS2(forward_transform,
                                              inverse_transform,
                                              embed_dim,
                                              operator_type = operator_type,
                                              sparsity_threshold = sparsity_threshold,
                                              hidden_size_factor = hidden_size_factor,
                                              complex_activation = complex_activation,
                                              spectral_layers = spectral_layers,
                                              drop_rate = drop_rate,
                                              bias = False)

        elif filter_type == "non-linear" and isinstance(forward_transform, RealFFT2):
            self.filter = SpectralAttention2d(forward_transform,
                                              inverse_transform,
                                              embed_dim,
                                              sparsity_threshold = sparsity_threshold,
                                              use_complex_kernels = use_complex_kernels,
                                              hidden_size_factor = hidden_size_factor,
                                              complex_activation = complex_activation,
                                              spectral_layers = spectral_layers,
                                              drop_rate = drop_rate,
                                              bias = False)

        elif filter_type == "linear" and factorization is None:
            self.filter = SpectralConvS2(forward_transform,
                                         inverse_transform,
                                         embed_dim,
                                         embed_dim,
                                         operator_type = operator_type,
                                         lr_scale_exponent = lr_scale_exponent,
                                         bias = True)
            
        elif filter_type == "linear" and factorization is not None:
            self.filter = FactorizedSpectralConvS2(forward_transform,
                                                   inverse_transform,
                                                   embed_dim,
                                                   embed_dim,
                                                   operator_type = operator_type,
                                                   rank = rank,
                                                   factorization = factorization,
                                                   separable = separable,
                                                   bias = True)

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
            embed_dim,
            filter_type = "non-linear",
            operator_type = "driscoll-healy",
            mlp_ratio = 2.,
            drop_rate = 0.,
            drop_path = 0.,
            act_layer = nn.GELU,
            norm_layer = nn.Identity,
            sparsity_threshold = 0.0,
            use_complex_kernels = True,
            lr_scale_exponent = 0,
            factorization = None,
            separable = False,
            rank = 128,
            inner_skip = "linear",
            outer_skip = None,
            concat_skip = False,
            use_mlp = True,
            complex_activation = "real",
            spectral_layers = 3):
        super(SphericalFourierNeuralOperatorBlock, self).__init__()
        
        # convolution layer
        self.filter = SpectralFilterLayer(forward_transform,
                                          inverse_transform,
                                          embed_dim,
                                          filter_type,
                                          operator_type = operator_type,
                                          sparsity_threshold = sparsity_threshold,
                                          use_complex_kernels = use_complex_kernels,
                                          hidden_size_factor = mlp_ratio,
                                          lr_scale_exponent = lr_scale_exponent,
                                          factorization = factorization,
                                          separable = separable,
                                          rank = rank,
                                          complex_activation = complex_activation,
                                          spectral_layers = spectral_layers,
                                          drop_rate = drop_rate)

        if inner_skip == "linear":
            self.inner_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1)
        elif inner_skip == "identity":
            self.inner_skip = nn.Identity()
        elif inner_skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {inner_skip}")

        self.concat_skip = concat_skip

        if concat_skip and inner_skip is not None:
            self.inner_skip_conv = nn.Conv2d(2*embed_dim, embed_dim, 1, bias=False)

        if filter_type == "linear":
            self.act_layer = act_layer()

        # first normalisation layer
        self.norm0 = norm_layer()
        
        # dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        if use_mlp == True:
            mlp_hidden_dim = int(embed_dim * mlp_ratio)
            self.mlp = MLP(in_features = embed_dim,
                           hidden_features = mlp_hidden_dim,
                           act_layer = act_layer,
                           drop_rate = drop_rate,
                           checkpointing = False)

        if outer_skip == "linear":
            self.outer_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1)
        elif outer_skip == "identity":
            self.outer_skip = nn.Identity()
        elif outer_skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {outer_skip}")

        if concat_skip and outer_skip is not None:
            self.outer_skip_conv = nn.Conv2d(2*embed_dim, embed_dim, 1, bias=False)

        # second normalisation layer
        self.norm1 = norm_layer()

    def forward(self, x):

        x, residual = self.filter(x)

        if hasattr(self, "inner_skip"):
            if self.concat_skip:
                x = torch.cat((x, self.inner_skip(residual)), dim=1)
                x = self.inner_skip_conv(x)
            else:
                x = x + self.inner_skip(residual)

        if hasattr(self, "act_layer"):
            x = self.act_layer(x)

        x = self.norm0(x)

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        x = self.drop_path(x)

        if hasattr(self, "outer_skip"):
            if self.concat_skip:
                x = torch.cat((x, self.outer_skip(residual)), dim=1)
                x = self.outer_skip_conv(x)
            else:
                x = x + self.outer_skip(residual)

        x = self.norm1(x)

        return x

class SphericalFourierNeuralOperatorNet(nn.Module):
    """
    SphericalFourierNeuralOperator module. Can use both FFTs and SHTs to represent either FNO or SFNO,
    both linear and non-linear variants.

    Parameters
    ----------
    filter_type : str, optional
        Type of filter to use ('linear', 'non-linear'), by default "linear"
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
    sparsity_threshold : float, optional
        Threshold for sparsity, by default 0.0
    normalization_layer : str, optional
        Type of normalization layer to use ("layer_norm", "instance_norm", "none"), by default "instance_norm"
    hard_thresholding_fraction : float, optional
        Fraction of hard thresholding (frequency cutoff) to apply, by default 1.0
    use_complex_kernels : bool, optional
        Whether to use complex kernels, by default True
    big_skip : bool, optional
        Whether to add a single large skip connection, by default True
    rank : float, optional
        Rank of the approximation, by default 1.0
    lr_scale_exponent : float, optional
        exponential rescaling of spectral coefficients, by default 0.0 (no rescaling)
    factorization : Any, optional
        Type of factorization to use, by default None
    separable : bool, optional
        Whether to use separable convolutions, by default False
    rank : (int, Tuple[int]), optional
        If a factorization is used, which rank to use. Argument is passed to tensorly
    complex_activation : str, optional
        Type of complex activation function to use, by default "real"
    spectral_layers : int, optional
        Number of spectral layers, by default 3
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
            filter_type = "linear",
            spectral_transform = "sht",
            operator_type = "driscoll-healy",
            img_size = (128, 256),
            scale_factor = 3,
            in_chans = 3,
            out_chans = 3,
            embed_dim = 256,
            num_layers = 4,
            activation_function = "gelu",
            encoder_layers = 1,
            use_mlp = True,
            mlp_ratio = 2.,
            drop_rate = 0.,
            drop_path_rate = 0.,
            sparsity_threshold = 0.0,
            normalization_layer = "none",
            hard_thresholding_fraction = 1.0,
            use_complex_kernels = True,
            big_skip = True,
            lr_scale_exponent = 0,
            factorization = None,
            separable = False,
            rank = 128,
            complex_activation = "real",
            spectral_layers = 2,
            pos_embed = True):

        super(SphericalFourierNeuralOperatorNet, self).__init__()

        self.filter_type = filter_type
        self.spectral_transform = spectral_transform
        self.operator_type = operator_type
        self.img_size = img_size
        self.scale_factor = scale_factor
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = self.num_features = embed_dim
        self.pos_embed_dim = self.embed_dim
        self.num_layers = num_layers
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.normalization_layer = normalization_layer
        self.use_mlp = use_mlp
        self.encoder_layers = encoder_layers
        self.big_skip = big_skip
        self.lr_scale_exponent = lr_scale_exponent
        self.factorization = factorization
        self.separable = separable,
        self.rank = rank
        self.complex_activation = complex_activation
        self.spectral_layers = spectral_layers

        # activation function
        if activation_function == "relu":
            self.activation_function = nn.ReLU
        elif activation_function == "gelu":
            self.activation_function = nn.GELU
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
            norm_layer1 = norm_layer0
        elif self.normalization_layer == "none":
            norm_layer0 = nn.Identity
            norm_layer1 = norm_layer0
        else:
            raise NotImplementedError(f"Error, normalization {self.normalization_layer} not implemented.") 

        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, self.img_size[0], self.img_size[1]))
        else:
            self.pos_embed = None

        # encoder
        encoder_hidden_dim = int(self.embed_dim * mlp_ratio)
        encoder = MLP(in_features = self.in_chans,
                      out_features = self.embed_dim,
                      hidden_features = encoder_hidden_dim,
                      act_layer = self.activation_function,
                      drop_rate = drop_rate,
                      checkpointing = False)
        self.encoder = encoder
        # self.encoder = nn.Sequential(encoder, norm_layer0())
        
        # prepare the spectral transform
        if self.spectral_transform == "sht":

            modes_lat = int(self.h * self.hard_thresholding_fraction)
            modes_lon = int((self.w // 2 + 1) * self.hard_thresholding_fraction)

            self.trans_down = RealSHT(*self.img_size, lmax=modes_lat, mmax=modes_lon, grid="equiangular").float()
            self.itrans_up  = InverseRealSHT(*self.img_size, lmax=modes_lat, mmax=modes_lon, grid="equiangular").float()
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

            inner_skip = 'linear'
            outer_skip = 'identity'

            if first_layer:
                norm_layer = norm_layer1
            elif last_layer:
                norm_layer = norm_layer0
            else:
                norm_layer = norm_layer1

            block = SphericalFourierNeuralOperatorBlock(forward_transform,
                                                        inverse_transform,
                                                        self.embed_dim,
                                                        filter_type = filter_type,
                                                        operator_type = self.operator_type,
                                                        mlp_ratio = mlp_ratio,
                                                        drop_rate = drop_rate,
                                                        drop_path = dpr[i],
                                                        act_layer = self.activation_function,
                                                        norm_layer = norm_layer,
                                                        sparsity_threshold = sparsity_threshold,
                                                        use_complex_kernels = use_complex_kernels,
                                                        inner_skip = inner_skip,
                                                        outer_skip = outer_skip,
                                                        use_mlp = use_mlp,
                                                        lr_scale_exponent = self.lr_scale_exponent,
                                                        factorization = self.factorization,
                                                        separable = self.separable,
                                                        rank = self.rank,
                                                        complex_activation = self.complex_activation,
                                                        spectral_layers = self.spectral_layers)

            self.blocks.append(block)

        # decoder
        encoder_hidden_dim = int(self.embed_dim * mlp_ratio)
        self.decoder = MLP(in_features = self.embed_dim + self.big_skip*self.in_chans,
                           out_features = self.out_chans,
                           hidden_features = encoder_hidden_dim,
                           act_layer = self.activation_function,
                           drop_rate = drop_rate,
                           checkpointing = False)

        # trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            #nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

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
    
    
