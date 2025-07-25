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

import os, sys

from functools import partial
import torch.nn as nn

# import baseline models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from baseline_models import Transformer, UNet, Segformer, EGTransformer
from torch_harmonics.examples.models import SphericalFourierNeuralOperator, LocalSphericalNeuralOperator, SphericalTransformer, SphericalUNet, SphericalSegformer

def get_baseline_models(img_size=(128, 256), in_chans=3, out_chans=3, residual_prediction=False, drop_path_rate=0., grid="equiangular"):
    
    # prepare dicts containing models and corresponding metrics
    model_registry = dict(
        sfno_sc2_layers4_e32 = partial(
            SphericalFourierNeuralOperator,
            img_size=img_size,
            grid=grid,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=2,
            embed_dim=32,
            activation_function="gelu",
            residual_prediction=residual_prediction,
            use_mlp=True,
            normalization_layer="instance_norm",
        ),
        lsno_sc2_layers4_e32 = partial(
            LocalSphericalNeuralOperator,
            img_size=img_size,
            grid=grid,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=2,
            embed_dim=32,
            activation_function="gelu",
            residual_prediction=residual_prediction,
            use_mlp=True,
            normalization_layer="instance_norm",
            kernel_shape=(5, 4),
            encoder_kernel_shape=(5, 4),
            filter_basis_type="piecewise linear",
            upsample_sht=False,
        ),
        s2unet_sc2_layers4_e128 = partial(
            SphericalUNet,
            img_size=img_size,
            grid=grid,
            grid_internal="equiangular",
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dims=[16, 32, 64, 128],
            depths=[2, 2, 2, 2],
            scale_factor=2,
            activation_function="gelu",
            kernel_shape=(5, 4),
            filter_basis_type="piecewise linear",
            drop_path_rate=0.1,
            drop_conv_rate=0.2,
            drop_dense_rate=0.5,
            transform_skip=False,
            upsampling_mode="conv",
            downsampling_mode="conv",
        ),
        
        s2transformer_sc2_layers4_e128 = partial(
            SphericalTransformer,
            img_size=img_size,
            grid=grid,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=2,
            embed_dim=128,
            activation_function="gelu",
            residual_prediction=residual_prediction,
            pos_embed="spectral",
            use_mlp=True,
            normalization_layer="instance_norm",
            encoder_kernel_shape=(5, 4),
            filter_basis_type="piecewise linear",
            drop_path_rate=drop_path_rate,
            upsample_sht=False,
            attention_mode="global",
            bias=False
        ),
        s2transformer_sc2_layers4_e256 = partial(
            SphericalTransformer,
            img_size=img_size,
            grid=grid,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=2,
            embed_dim=256,
            activation_function="gelu",
            residual_prediction=residual_prediction,
            pos_embed="spectral",
            use_mlp=True,
            normalization_layer="instance_norm",
            encoder_kernel_shape=(5, 4),
            filter_basis_type="piecewise linear",
            drop_path_rate=drop_path_rate,
            upsample_sht=False,
            attention_mode="global",
            bias=False
        ),
        
        s2ntransformer_sc2_layers4_e128 = partial(
            SphericalTransformer,
            img_size=img_size,
            grid=grid,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=2,
            embed_dim=128,
            activation_function="gelu",
            residual_prediction=residual_prediction,
            pos_embed="spectral",
            use_mlp=True,
            normalization_layer="instance_norm",
            encoder_kernel_shape=(5, 4),
            filter_basis_type="piecewise linear",
            drop_path_rate=drop_path_rate,
            upsample_sht=False,
            attention_mode="neighborhood",
            bias=False
        ),
        s2ntransformer_sc2_layers4_e256 = partial(
            SphericalTransformer,
            img_size=img_size,
            grid=grid,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=2,
            embed_dim=256,
            activation_function="gelu",
            residual_prediction=residual_prediction,
            pos_embed="spectral",
            use_mlp=True,
            normalization_layer="instance_norm",
            encoder_kernel_shape=(5, 4),
            filter_basis_type="piecewise linear",
            drop_path_rate=drop_path_rate,
            upsample_sht=False,
            attention_mode="neighborhood",
            bias=False
        ),
        
        transformer_sc2_layers4_e128 = partial(
            Transformer,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=2,
            embed_dim=128,
            activation_function="gelu",
            residual_prediction=residual_prediction,
            pos_embed="spectral",
            use_mlp=True,
            normalization_layer="instance_norm",
            encoder_kernel_shape=(3, 3),
            drop_path_rate=drop_path_rate,
            attention_mode="global",
            upsampling_method="conv",
            bias=False
        ),
        transformer_sc2_layers4_e256 = partial(
            Transformer,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=2,
            embed_dim=256,
            activation_function="gelu",
            residual_prediction=residual_prediction,
            pos_embed="spectral",
            use_mlp=True,
            normalization_layer="instance_norm",
            encoder_kernel_shape=(3, 3),
            drop_path_rate=drop_path_rate,
            attention_mode="global",
            upsampling_method="conv",
            bias=False
        ),
        
        ntransformer_sc2_layers4_e128 = partial(
            Transformer,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=2,
            embed_dim=128,
            activation_function="gelu",
            residual_prediction=residual_prediction,
            pos_embed="spectral",
            use_mlp=True,
            normalization_layer="instance_norm",
            encoder_kernel_shape=(3, 3),
            drop_path_rate=drop_path_rate,
            attention_mode="neighborhood",
            attn_kernel_shape=(7, 7),
            bias=False
        ),
        ntransformer_sc2_layers4_e256 = partial(
            Transformer,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=2,
            embed_dim=256,
            activation_function="gelu",
            residual_prediction=residual_prediction,
            pos_embed="spectral",
            use_mlp=True,
            normalization_layer="instance_norm",
            encoder_kernel_shape=(3, 3),
            drop_path_rate=drop_path_rate,
            attention_mode="neighborhood",
            attn_kernel_shape=(7, 7),
	    bias=False
	),
        
        s2segformer_sc2_layers4_e128 = partial(
            SphericalSegformer,
            img_size=img_size,
            grid=grid,
            grid_internal="equiangular",
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dims=[16, 32, 64, 128],
            heads=[1, 2, 4, 8],
            depths=[3, 4, 6, 3],
            scale_factor=2,
            activation_function="gelu",
            kernel_shape=(5, 4),
            filter_basis_type="piecewise linear",
            mlp_ratio=4.0,
            att_drop_rate=0.0,
            drop_path_rate=0.1,
            attention_mode="global",
            bias=False
        ),
        s2segformer_sc2_layers4_e256 = partial(
            SphericalSegformer,
            img_size=img_size,
            grid=grid,
            grid_internal="equiangular",
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dims=[32, 64, 128, 256],
            heads=[1, 2, 4, 8],
            depths=[3, 4, 6, 3],
            scale_factor=2,
            activation_function="gelu",
            kernel_shape=(5, 4),
            filter_basis_type="piecewise linear",
            mlp_ratio=4.0,
            att_drop_rate=0.0,
            drop_path_rate=0.1,
            attention_mode="global",
            bias=False
        ),
        
        s2nsegformer_sc2_layers4_e128 = partial(
            SphericalSegformer,
            img_size=img_size,
            grid=grid,
            grid_internal="equiangular",
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dims=[16, 32, 64, 128],
            heads=[1, 2, 4, 8],
            depths=[3, 4, 6, 3],
            scale_factor=2,
            activation_function="gelu",
            kernel_shape=(5, 4),
            filter_basis_type="piecewise linear",
            mlp_ratio=4.0,
            att_drop_rate=0.0,
            drop_path_rate=0.1,
            attention_mode="neighborhood",
            bias=False
        ),
        s2nsegformer_sc2_layers4_e256 = partial(
            SphericalSegformer,
            img_size=img_size,
            grid=grid,
            grid_internal="equiangular",
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dims=[32, 64, 128, 256],
            heads=[1, 2, 4, 8],
            depths=[3, 4, 6, 3],
            scale_factor=2,
            activation_function="gelu",
            kernel_shape=(5, 4),
            filter_basis_type="piecewise linear",
            mlp_ratio=4.0,
            att_drop_rate=0.0,
            drop_path_rate=0.1,
            attention_mode="neighborhood",
            bias=False
        ),
        
        segformer_sc2_layers4_e128 = partial(
            Segformer,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dims=[16, 32, 64, 128],
            heads=[1, 2, 4, 8],
            depths=[3, 4, 6, 3],
            scale_factor=2,
            activation_function="gelu",
            kernel_shape=(4, 4),
            mlp_ratio=4.0,
            att_drop_rate=0.0,
            drop_path_rate=0.1,
            attention_mode="global",
            bias=False
        ),
        segformer_sc2_layers4_e256 = partial(
            Segformer,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dims=[32, 64, 128, 256],
            heads=[1, 2, 4, 8],
            depths=[3, 4, 6, 3],
            scale_factor=2,
            activation_function="gelu",
            kernel_shape=(4, 4),
            mlp_ratio=4.0,
            att_drop_rate=0.0,
            drop_path_rate=0.1,
            attention_mode="global",
            bias=False
        ),
        
        nsegformer_sc2_layers4_e128 = partial(
            Segformer,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dims=[16, 32, 64, 128],
            heads=[1, 2, 4, 8],
            depths=[3, 4, 6, 3],
            scale_factor=2,
            activation_function="gelu",
            kernel_shape=(4, 4),
            mlp_ratio=4.0,
            att_drop_rate=0.0,
            drop_path_rate=0.1,
            attention_mode="neighborhood",
            attn_kernel_shape=(7, 7),
            bias=False
        ),
        nsegformer_sc2_layers4_e256 = partial(
            Segformer,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dims=[32, 64, 128, 256],
            heads=[1, 2, 4, 8],
            depths=[3, 4, 6, 3],
            scale_factor=2,
            activation_function="gelu",
            kernel_shape=(4, 4),
            mlp_ratio=4.0,
            att_drop_rate=0.0,
            drop_path_rate=0.1,
            attention_mode="neighborhood",
            attn_kernel_shape=(7, 7),
            bias=False
        ),
        
        vit_sc2_layers4_e128 = partial(
            Transformer,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=2,
            embed_dim=128,
            activation_function="gelu",
            residual_prediction=residual_prediction,
            pos_embed="learnable latlon",
            use_mlp=True,
            normalization_layer="layer_norm",
            encoder_kernel_shape=(2, 2),
            attention_mode="global",
            upsampling_method="pixel_shuffle",
            bias=False
        ),
        
        egformer = partial(
            EGTransformer,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dim=32,
            depth=[2, 2, 2, 2, 2, 2, 2, 2, 2],
            split_size=[1, 1, 1, 1, 1, 1, 1, 1, 1],
            num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            norm_layer=nn.LayerNorm,
        ),
    )

    return model_registry
