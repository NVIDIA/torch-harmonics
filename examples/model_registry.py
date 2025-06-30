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

# import baseline models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from baseline_models import Transformer, UNet, Segformer
from torch_harmonics.examples.models import SphericalFourierNeuralOperator, LocalSphericalNeuralOperator, SphericalTransformer, SphericalUNet, SphericalSegformer

def get_baseline_models(img_size=(128, 256), in_chans=3, out_chans=3, residual_prediction=False, drop_path_rate=0., grid="equiangular"):
    """
    Get a registry of baseline models for spherical and planar neural networks.
    
    This function returns a dictionary containing pre-configured model architectures
    for various tasks including spherical Fourier neural operators (SFNO), local
    spherical neural operators (LSNO), spherical transformers, U-Nets, and Segformers.
    Each model is configured with specific hyperparameters optimized for different
    computational budgets and performance requirements.
    
    Parameters
    ----------
    img_size : tuple, optional
        Input image size as (height, width), by default (128, 256)
    in_chans : int, optional
        Number of input channels, by default 3
    out_chans : int, optional
        Number of output channels, by default 3
    residual_prediction : bool, optional
        Whether to use residual prediction (add input to output), by default False
    drop_path_rate : float, optional
        Dropout path rate for regularization, by default 0.0
    grid : str, optional
        Grid type for spherical models ("equiangular", "legendre-gauss", etc.), by default "equiangular"
    
    Returns
    ----------
    dict
        Dictionary mapping model names to partial functions that can be called
        to instantiate the corresponding model with the specified parameters.
        
        Available models include:
        
        **Spherical Models:**
        - sfno_sc2_layers4_e32: Spherical Fourier Neural Operator (small)
        - lsno_sc2_layers4_e32: Local Spherical Neural Operator (small)
        - s2unet_sc2_layers4_e128: Spherical U-Net (medium)
        - s2transformer_sc2_layers4_e128: Spherical Transformer (global attention, medium)
        - s2transformer_sc2_layers4_e256: Spherical Transformer (global attention, large)
        - s2ntransformer_sc2_layers4_e128: Spherical Transformer (neighborhood attention, medium)
        - s2ntransformer_sc2_layers4_e256: Spherical Transformer (neighborhood attention, large)
        - s2segformer_sc2_layers4_e128: Spherical Segformer (global attention, medium)
        - s2segformer_sc2_layers4_e256: Spherical Segformer (global attention, large)
        - s2nsegformer_sc2_layers4_e128: Spherical Segformer (neighborhood attention, medium)
        - s2nsegformer_sc2_layers4_e256: Spherical Segformer (neighborhood attention, large)
        
        **Planar Models:**
        - transformer_sc2_layers4_e128: Planar Transformer (global attention, medium)
        - transformer_sc2_layers4_e256: Planar Transformer (global attention, large)
        - ntransformer_sc2_layers4_e128: Planar Transformer (neighborhood attention, medium)
        - ntransformer_sc2_layers4_e256: Planar Transformer (neighborhood attention, large)
        - segformer_sc2_layers4_e128: Planar Segformer (global attention, medium)
        - segformer_sc2_layers4_e256: Planar Segformer (global attention, large)
        - nsegformer_sc2_layers4_e128: Planar Segformer (neighborhood attention, medium)
        - nsegformer_sc2_layers4_e256: Planar Segformer (neighborhood attention, large)
        - vit_sc2_layers4_e128: Vision Transformer variant (medium)
    
    Examples
    ----------
    >>> model_registry = get_baseline_models(img_size=(64, 128), in_chans=2, out_chans=1)
    >>> model = model_registry['sfno_sc2_layers4_e32']()
    >>> print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    Notes
    ----------
    - Model names follow the pattern: {model_type}_{scale_factor}_{layers}_{embed_dim}
    - 'sc2' indicates scale factor of 2 (downsampling by 2)
    - 'e32', 'e128', 'e256' indicate embedding dimensions
    - 'n' prefix indicates neighborhood attention instead of global attention
    - All models use GELU activation and instance normalization by default
    """

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
    )

    return model_registry
