from torch_harmonics.examples.models._layers import (
    MLP, 
    LayerNorm, 
    DropPath, 
    SequencePositionEmbedding, 
    SpectralPositionEmbedding, 
    LearnablePositionEmbedding,
)
from torch_harmonics.examples.models.s2transformer import (
    _compute_cutoff_radius,
    DiscreteContinuousEncoder,
    DiscreteContinuousDecoder,
    SphericalAttentionBlock
)
from .transformer import (
    Encoder,
    Decoder,
    AttentionBlock
)

import torch
import torch.nn as nn

class AblationTransformer(nn.Module):
    def __init__(
        self,
        sph_encoding=False,
        sph_attention=False,
        # common params
        img_size=(128, 256),
        scale_factor=3,
        in_chans=3,
        out_chans=3,
        embed_dim=256,
        num_layers=4,
        activation_function="gelu",
        encoder_kernel_shape=(3, 3),
        num_heads=1,
        use_mlp=True,
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        normalization_layer="none",
        residual_prediction=False,
        pos_embed="spectral",
        bias=True,
        attention_mode="neighborhood",
        # euclidean-only params
        attn_kernel_shape=(7, 7),
        upsampling_method="conv",
        # spherical-only params
        grid="equiangular",
        grid_internal="legendre-gauss",
        upsample_sht=False,
        theta_cutoff=None,
        filter_basis_type="morlet",
        hard_thresholding_fraction=1.0
    ):
        super().__init__()
    
        self.img_size = img_size
        self.scale_factor = scale_factor
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.encoder_kernel_shape = encoder_kernel_shape
        self.normalization_layer = normalization_layer
        self.use_mlp = use_mlp
        self.residual_prediction = residual_prediction

        self.grid = grid
        self.grid_internal = grid_internal
        self.hard_thresholding_fraction = hard_thresholding_fraction

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

        # maybe keep for now becuase tr
        # encoder
        if sph_encoding:
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
        else:
            self.encoder = Encoder(
                in_shape=self.img_size,
                out_shape=(self.h, self.w),
                in_chans=self.in_chans,
                out_chans=self.embed_dim,
                kernel_shape=self.encoder_kernel_shape,
                groups=1,
                bias=False,
            )
            self.decoder = Decoder(
                in_shape=(self.h, self.w),
                out_shape=self.img_size,
                in_chans=self.embed_dim,
                out_chans=self.out_chans,
                kernel_shape=self.encoder_kernel_shape,
                groups=1,
                bias=False,
                upsampling_method=upsampling_method,
            )

        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):
            if sph_attention:
                block = SphericalAttentionBlock(
                    in_shape=(self.h, self.w),
                    out_shape=(self.h, self.w),
                    grid_in=grid_internal,
                    grid_out=grid_internal,
                    in_chans=self.embed_dim,
                    out_chans=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    drop_path=dpr[i],
                    act_layer=self.activation_function,
                    norm_layer=self.normalization_layer,
                    use_mlp=use_mlp,
                    bias=bias,
                    attention_mode=attention_mode,
                    theta_cutoff=theta_cutoff,
                )
            else:
                block = AttentionBlock(
                    in_shape=(self.h, self.w),
                    out_shape=(self.h, self.w),
                    chans=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    drop_path=dpr[i],
                    act_layer=self.activation_function,
                    norm_layer=self.normalization_layer,
                    use_mlp=use_mlp,
                    bias=bias,
                    attention_mode=attention_mode,
                    attn_kernel_shape=attn_kernel_shape,
                )

            self.blocks.append(block)

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
