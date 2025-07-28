import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Additional imports...
from einops.layers.torch import Rearrange
from einops import rearrange
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from typing import List, Optional
from torch.utils.checkpoint import checkpoint

class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, resolution, norm_layer=nn.LayerNorm,scale_factor=0.5):
        super().__init__()

        if scale_factor < 1.:
            self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        elif scale_factor > 1.:
            self.conv = nn.Conv2d(dim, dim_out, 3, 1, 1)
        elif scale_factor == 1.:
            self.conv = nn.Conv2d(dim, dim_out, 3, 1, 1)
        self.scale_factor = scale_factor   
        self.norm = norm_layer(dim_out)
        self.resolution = resolution
        self.gelu = nn.GELU()
 
    def forward(self, x):
        B, new_HW, C = x.shape
        # H = W = int(np.sqrt(new_HW))
        H, W = self.resolution[0], self.resolution[1]
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        if self.scale_factor >1.:
            x = F.interpolate(x,scale_factor=self.scale_factor)
        x = self.conv(x)
        x = self.gelu(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)
        
        return x

class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel, input_resolution=None):
        super(Downsample, self).__init__()
        self.input_resolution = input_resolution
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=0),

        )

    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H, W = self.input_resolution
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = F.pad(x, (1, 1, 0, 0), mode='circular')  # width
        x = F.pad(x, (0, 0, 1, 1))
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel, input_resolution=None):
        super(Upsample, self).__init__()
        self.input_resolution = input_resolution
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)  # B, H//H_sp, W//W_sp, H_sp * W_sp, C
    return img_perm


class To_BCHW(nn.Module):
    def __init__(self, resolution, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.resolution = resolution
        self.gelu = nn.GELU()
    def forward(self, x):
        B, new_HW, C = x.shape
        # H = W = int(np.sqrt(new_HW))
        H, W = self.resolution[0], self.resolution[1]
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        
        return x

class Tune_Block_Final(nn.Module):
    def __init__(self, dim, dim_out, resolution, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 1, 1)
        self.norm = norm_layer(dim_out)
        self.resolution = resolution
        self.gelu = nn.GELU()
    def forward(self, x):
        B, new_HW, C = x.shape
#        # H = W = int(np.sqrt(new_HW))
        H, W = self.resolution[0], self.resolution[1]
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.gelu(x)
        x = self.norm(x)
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        
        return x

class Tune_Block(nn.Module):
    def __init__(self, dim, dim_out, resolution, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 1, 1)
        self.norm = norm_layer(dim_out)
        self.resolution = resolution
        self.gelu = nn.GELU()
    def forward(self, x):
        H, W = self.resolution[0], self.resolution[1]
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.gelu(x)
        x = self.norm(x)
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        
        return x

class GridGenerator:
  def __init__(self, height: int, width: int, kernel_size, stride=1):
    self.height = height
    self.width = width
    self.kernel_size = kernel_size  # (Kh, Kw)
    self.stride = stride  # (H, W)

  def createSamplingPattern(self):
    """
    :return: (1, H*Kh, W*Kw, (Lat, Lon)) sampling pattern
    """
    kerX, kerY = self.createKernel()  # (Kh, Kw)

    # create some values using in generating lat/lon sampling pattern
    rho = np.sqrt(kerX ** 2 + kerY ** 2)
    Kh, Kw = self.kernel_size
    # when the value of rho at center is zero, some lat values explode to `nan`.
    if Kh % 2 and Kw % 2:
      rho[Kh // 2][Kw // 2] = 1e-8

    nu = np.arctan(rho)
    cos_nu = np.cos(nu)
    sin_nu = np.sin(nu)

    stride_h, stride_w = self.stride, self.stride
    h_range = np.arange(0, self.height, stride_h)
    w_range = np.arange(0, self.width, stride_w)

    lat_range = ((h_range / self.height) - 0.5) * np.pi
    lon_range = ((w_range / self.width) - 0.5) * (2 * np.pi)

    # generate latitude sampling pattern
    lat = np.array([
      np.arcsin(cos_nu * np.sin(_lat) + kerY * sin_nu * np.cos(_lat) / rho) for _lat in lat_range
    ])  # (H, Kh, Kw)

    lat = np.array([lat for _ in lon_range])  # (W, H, Kh, Kw)
    lat = lat.transpose((1, 0, 2, 3))  # (H, W, Kh, Kw)

    # generate longitude sampling pattern
    lon = np.array([
      np.arctan(kerX * sin_nu / (rho * np.cos(_lat) * cos_nu - kerY * np.sin(_lat) * sin_nu)) for _lat in lat_range
    ])  # (H, Kh, Kw)

    lon = np.array([lon + _lon for _lon in lon_range])  # (W, H, Kh, Kw)
    lon = lon.transpose((1, 0, 2, 3))  # (H, W, Kh, Kw)

    # (radian) -> (index of pixel)
    lat = (lat / np.pi + 0.5) * self.height
    lon = ((lon / (2 * np.pi) + 0.5) * self.width) % self.width

    LatLon = np.stack((lat, lon))  # (2, H, W, Kh, Kw) = ((lat, lon), H, W, Kh, Kw)
    LatLon = LatLon.transpose((1, 2, 3, 4, 0))  # (H, Kh, W, Kw, 2) = (H, Kh, W, Kw, (lat, lon))

    H, W, Kh, Kw, d = LatLon.shape
    LatLon = LatLon.reshape((1, H, W, Kh*Kw, d))  # (1, H*Kh, W*Kw, 2)

    return LatLon

  def createKernel(self):
    """
    :return: (Ky, Kx) kernel pattern
    """
    Kh, Kw = self.kernel_size

    delta_lat = np.pi / self.height
    delta_lon = 2 * np.pi / self.width

    range_x = np.arange(-(Kw // 2), Kw // 2 + 1)
    if not Kw % 2:
      range_x = np.delete(range_x, Kw // 2)

    range_y = np.arange(-(Kh // 2), Kh // 2 + 1)
    if not Kh % 2:
      range_y = np.delete(range_y, Kh // 2)

    kerX = np.tan(range_x * delta_lon)
    kerY = np.tan(range_y * delta_lat) / np.cos(range_y * delta_lon)

    return np.meshgrid(kerX, kerY)  # (Kh, Kw)

def genSamplingPattern(h, w, kh, kw, stride=1):
    gridGenerator = GridGenerator(h, w, (kh, kw), stride)
    LonLatSamplingPattern = gridGenerator.createSamplingPattern()

    # generate grid to use `F.grid_sample`
    # lat_grid = (LonLatSamplingPattern[:, :, :, 0] / h) * 2 - 1
    # lon_grid = (LonLatSamplingPattern[:, :, :, 1] / w) * 2 - 1

    grid = LonLatSamplingPattern#np.stack((lon_grid, lat_grid), axis=-1)
    with torch.no_grad():
      grid = torch.FloatTensor(grid)
      grid.requires_grad = False

    return grid

########### feed-forward network #############
class LeFF(nn.Module):
    def __init__(self, dim, hidden_dim=None, act_layer=nn.GELU, drop=0., flag = 0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4  # Default to 4x expansion if not specified
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=0),
            act_layer())
        #self.hw = StripPooling(hidden_dim)
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))

    def forward(self, x, H, W):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = H

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh * 2)
        # bs,hidden_dim,32x32
        #att = self.hw(x)
        
        x = F.pad(x, (1, 1, 0, 0), mode='circular')  # width
        x = F.pad(x, (0, 0, 1, 1))

        x = self.dwconv(x)

        #x = x * att

        #x = self.active(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh * 2)

        x = self.linear2(x)

        return x
class PanoSelfAttention(nn.Module):
    def __init__(self, h,
                 d_model,
                 k,
                 last_feat_height,
                 last_feat_width,
                 scales=1,
                 dropout=0.1,
                 need_attn=False):
        """
        :param h: number of self attention head
        :param d_model: dimension of model
        :param dropout:
        :param k: number of keys
        """
        super(PanoSelfAttention, self).__init__()
        #assert h == 8  # currently header is fixed 8 in paper
        assert d_model % h == 0
        # We assume d_v always equals d_k, d_q = d_k = d_v = d_m / h
        self.d_k = int(d_model / h)
        self.h = h

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)

        self.scales_hw = []
        for i in range(scales):
            self.scales_hw.append([last_feat_height * 2 ** i,
                                   last_feat_width * 2 ** i])

        self.dropout = None
        if self.dropout:
            self.dropout = nn.Dropout(p=dropout)

        self.k = k
        self.scales = scales
        self.last_feat_height = last_feat_height
        self.last_feat_width = last_feat_width

        self.offset_dims = 2 * self.h * self.k * self.scales
        self.A_dims = self.h * self.k * self.scales

        # 2MLK for offsets MLK for A_mlqk
        self.offset_proj = nn.Linear(d_model, self.offset_dims)
        self.A_proj = nn.Linear(d_model, self.A_dims)

        self.wm_proj = nn.Linear(d_model, d_model)
        self.need_attn = need_attn
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.offset_proj.weight, 0.0)
        torch.nn.init.constant_(self.A_proj.weight, 0.0)

        torch.nn.init.constant_(self.A_proj.bias, 1 / (self.scales * self.k))

        def init_xy(bias, x, y):
            torch.nn.init.constant_(bias[:, 0], float(x))
            torch.nn.init.constant_(bias[:, 1], float(y))

        # caution: offset layout will be  M, L, K, 2
        bias = self.offset_proj.bias.view(self.h, self.scales, self.k, 2)

        # init_xy(bias[0], x=-self.k, y=-self.k)
        # init_xy(bias[1], x=-self.k, y=0)
        # init_xy(bias[2], x=-self.k, y=self.k)
        # init_xy(bias[3], x=0, y=-self.k)
        # init_xy(bias[4], x=0, y=self.k)
        # init_xy(bias[5], x=self.k, y=-self.k)
        # init_xy(bias[6], x=self.k, y=0)
        # init_xy(bias[7], x=self.k, y=self.k)

    def forward(self,
                query: torch.Tensor,
                keys: List[torch.Tensor],
                ref_point: torch.Tensor,
                query_mask: torch.Tensor = None,
                key_masks: Optional[torch.Tensor] = None,
                ):
        """
        :param key_masks:
        :param query_mask:
        :param query: B, H, W, C
        :param keys: List[B, H, W, C]
        :param ref_point: B, H, W, 2
        :return:
        """
        if key_masks is None:
            key_masks = [None] * len(keys)

        assert len(keys) == self.scales

        attns = {'attns': None, 'offsets': None}

        nbatches, query_height, query_width, _ = query.shape

        # B, H, W, C
        query = self.q_proj(query)

        # B, H, W, 2MLK
        offset = self.offset_proj(query)
        # B, H, W, M, 2LK
        offset = offset.view(nbatches, query_height, query_width, self.h, -1)

        # B, H, W, MLK
        A = self.A_proj(query)

        # B, H, W, 1, mask before softmax
        if query_mask is not None:
            query_mask_ = query_mask.unsqueeze(dim=-1)
            _, _, _, mlk = A.shape
            query_mask_ = query_mask_.expand(nbatches, query_height, query_width, mlk)
            A = torch.masked_fill(A, mask=query_mask_, value=float('-inf'))

        # B, H, W, M, LK
        A = A.view(nbatches, query_height, query_width, self.h, -1)
        A = F.softmax(A, dim=-1)

        # mask nan position
        if query_mask is not None:
            # B, H, W, 1, 1
            query_mask_ = query_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
            A = torch.masked_fill(A, query_mask_.expand_as(A), 0.0)

        if self.need_attn:
            attns['attns'] = A
            attns['offsets'] = offset

        offset = offset.view(nbatches, query_height, query_width, self.h, self.scales, self.k, 2)
        offset = offset.permute(0, 3, 4, 5, 1, 2, 6).contiguous()
        # B*M, L, K, H, W, 2
        offset = offset.view(nbatches * self.h, self.scales, self.k, query_height, query_width, 2)

        A = A.permute(0, 3, 1, 2, 4).contiguous()
        # B*M, H*W, LK
        A = A.view(nbatches * self.h, query_height * query_width, -1)

        scale_features = []
        for l in range(self.scales):
            feat_map = keys[l]
            _, h, w, _ = feat_map.shape

            key_mask = key_masks[l]

            #ref_point = generate_ref_points(query_width, query_height).repeat(nbatches, 1, 1, 1)

            # B, K, H, W, 2
            # Scale reference points to match current feature map dimensions
            if ref_point.shape[2] != h or ref_point.shape[3] != w:
                # Interpolate reference points to match current spatial dimensions
                ref_point_resized = F.interpolate(
                    ref_point.view(-1, 2, ref_point.shape[2], ref_point.shape[3]), 
                    size=(h, w), 
                    mode='bilinear', 
                    align_corners=False
                ).view(ref_point.shape[0], ref_point.shape[1], h, w, 2)
                reversed_ref_point = ref_point_resized
            else:
                reversed_ref_point = ref_point

            #reversed_ref_point = restore_scale(w, h)

            # B, K, H, W, 2 -> B*M, K, H, W, 2
            reversed_ref_point = reversed_ref_point.repeat(self.h, 1, 1, 1, 1)

            #equi_offset = ref_point_offset.unsqueeze(1)

            # B, h, w, M, C_v
            scale_feature = self.k_proj(feat_map).view(nbatches, h, w, self.h, self.d_k)

            if key_mask is not None:
                # B, h, w, 1, 1
                key_mask = key_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
                key_mask = key_mask.expand(nbatches, h, w, self.h, self.d_k)
                scale_feature = torch.masked_fill(scale_feature, mask=key_mask, value=0)

            # B, M, C_v, h, w
            scale_feature = scale_feature.permute(0, 3, 4, 1, 2).contiguous()
            # B*M, C_v, h, w
            scale_feature = scale_feature.view(-1, self.d_k, h, w)

            k_features = []

            #show_feature_map(scale_feature)



            for k in range(self.k):
                # Scale offset to match feature map dimensions if needed
                current_offset = offset[:, l, k, :, :, :]
                
                # Debug: print tensor shapes
                # print(f"reversed_ref_point shape: {reversed_ref_point.shape}")
                # print(f"current_offset shape: {current_offset.shape}")
                # print(f"h, w: {h}, {w}")
                
                if current_offset.shape[2] != h or current_offset.shape[3] != w:
                    # Interpolate offset to match current spatial dimensions
                    # current_offset has shape [B*M, H, W, 2] where H, W are query dimensions
                    # We need to reshape it to [B*M, 2, H, W] for interpolation, then back to [B*M, h, w, 2]
                    offset_reshaped = current_offset.permute(0, 3, 1, 2).contiguous()  # [B*M, 2, H, W]
                    offset_resized = F.interpolate(
                        offset_reshaped, 
                        size=(h, w), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    current_offset = offset_resized.permute(0, 2, 3, 1).contiguous()  # [B*M, h, w, 2]
                
                # Ensure both tensors have the same spatial dimensions
                # reversed_ref_point has shape [B*M, 1, h, w, 2] where the second dimension is 1
                # We need to extract the reference point for the current k
                ref_point_k = reversed_ref_point[:, 0, :, :, :]  # [B*M, h, w, 2]
                
                # Debug: print final tensor shapes
                # print(f"ref_point_k shape: {ref_point_k.shape}")
                # print(f"current_offset shape after scaling: {current_offset.shape}")
                
                points = ref_point_k + current_offset
                vgrid_x = 2.0 * points[:, :, :, 1] / max(w - 1, 1) - 1.0
                vgrid_y = 2.0 * points[:, :, :, 0] / max(h - 1, 1) - 1.0
                vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
                #print(points)

                # B*M, C_v, H, W
                feat = F.grid_sample(scale_feature, vgrid_scaled, mode='bilinear', padding_mode='zeros', align_corners=False)
                #show_feature_map(feat)



                k_features.append(feat)

            # B*M, k, C_v, H, W
            k_features = torch.stack(k_features, dim=1)
            scale_features.append(k_features)

        # B*M, L, K, C_v, H, W
        scale_features = torch.stack(scale_features, dim=1)

        # B*M, H*W, C_v, LK
        scale_features = scale_features.permute(0, 4, 5, 3, 1, 2).contiguous()
        scale_features = scale_features.view(nbatches * self.h, query_height * query_width, self.d_k, -1)

        # B*M, H*W, C_v
        feat = torch.einsum('nlds, nls -> nld', scale_features, A)

        # B*M, H*W, C_v -> B, M, H, W, C_v
        feat = feat.view(nbatches, self.h, query_height, query_width, self.d_k)
        # B, M, H, W, C_v -> B, H, W, M, C_v
        feat = feat.permute(0, 2, 3, 1, 4).contiguous()
        # B, H, W, M, C_v -> B, H, W, M * C_v
        feat = feat.view(nbatches, query_height, query_width, self.d_k * self.h)

        feat = self.wm_proj(feat)
        if self.dropout:
            feat = self.dropout(feat)

        return feat

class PanoformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=4,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear', token_mlp='leff',
                 se_layer=False, ref_point = None, flag = 0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.ref_point = ref_point #generate_ref_points(self.input_resolution[1], self.input_resolution[0])
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)

        self.dattn = PanoSelfAttention(num_heads, dim, k=9, last_feat_height=self.input_resolution[0], last_feat_width=self.input_resolution[1], scales=1, dropout=0, need_attn=False)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop, flag = flag)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # W-MSA/SW-MSA
        if self.ref_point is not None:
            self.ref_point = self.ref_point.to(x.get_device())
            # Scale reference points to match current input resolution
            if self.ref_point.shape[2] != H or self.ref_point.shape[3] != W:
                # Interpolate reference points to match current spatial dimensions
                ref_point_resized = F.interpolate(
                    self.ref_point.view(-1, 2, self.ref_point.shape[2], self.ref_point.shape[3]), 
                    size=(H, W), 
                    mode='bilinear', 
                    align_corners=False
                ).view(self.ref_point.shape[0], self.ref_point.shape[1], H, W, 2)
                ref_point_to_use = ref_point_resized
            else:
                ref_point_to_use = self.ref_point
            x = self.dattn(x, x.unsqueeze(0), ref_point_to_use.repeat(B, 1, 1, 1, 1))  # nW*B, win_size*win_size, C
        else:
            # Fallback: create a simple reference point grid
            ref_point = torch.zeros(1, 1, H, W, 2, device=x.get_device())
            x = self.dattn(x, x.unsqueeze(0), ref_point.repeat(B, 1, 1, 1, 1))

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

########### Basic layer of Uformer ################
class BasicPanoformerLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear', token_mlp='leff', se_layer=False, ref_point = None, flag = 0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList([
            PanoformerBlock(dim=dim, input_resolution=input_resolution,
                                  num_heads=num_heads, win_size=win_size,
                                  shift_size=0 if (i % 2 == 0) else win_size // 2,
                                  mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  drop=drop, attn_drop=attn_drop,
                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                  norm_layer=norm_layer, token_projection=token_projection, token_mlp=token_mlp,
                                  se_layer=se_layer, ref_point=ref_point)
            for i in range(depth)])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x



VRPE_LUT=[]
HRPE_LUT=[]


def VRPE(num_heads, height,width,split_size): ## Used for vertical attention  / (theta,phi) -> (theta,phi') / This part is coded assuming that the split_size (stripe_with) is 1 to reduce the computational cost. To use larger split_size, ERPE must be calculated accoringly.

    H = height // split_size # Base height
    pi = torch.acos(torch.zeros(1)).item() * 2
    
    base_x = torch.linspace(0,H,H) * pi / H
    base_x = base_x.unsqueeze(0).repeat(H,1)

    base_y = torch.linspace(0,H,H) * pi / H
    base_y = base_y.unsqueeze(1).repeat(1,H)

    base = base_x - base_y
    pn = torch.where(base>0,1,-1)
    
    base =  torch.sqrt(2 * (1 - torch.cos(base))) # H x H 
    base = pn * base
    return (base.unsqueeze(0).unsqueeze(0)).repeat(width * split_size,num_heads,1,1) 

def HRPE(num_heads, height, width, split_size): ## Used for Horizontal attention  / (theta,phi) -> (theta',phi) / This part is coded assuming that the split_size (stripe_with) is 1 to reduce the computational cost. To use larger split_size, ERPE must be calculated accoringly.


    W = width // split_size # Base width
    pi = torch.acos(torch.zeros(1)).item() * 2

    base_x = torch.linspace(0,W,W) *2*pi / W
    base_x = base_x.unsqueeze(0).repeat(W,1)

    base_y = torch.linspace(0,W,W)*2*pi / W
    base_y = base_y.unsqueeze(1).repeat(1,W)
    base = base_x - base_y
    pn = torch.where(base>0,1,-1)
    base = base.unsqueeze(0).repeat(height,1,1)

    for k in range(0,height):
        base[k,:,:] = torch.sin(torch.as_tensor(k*pi/height)) * torch.sqrt(2 * (1 - torch.cos(base[k,:,:]))) # height x W x W  
    
    if True: # Unlike the vertical direction, EIs are cyclic along the horizontal direction. Set to 'False' to reflect this cyclic characteristic / Refer to discussions in repo for more details. 
        base = pn * base
    return base.unsqueeze(1).repeat(split_size,num_heads,1,1) 


# RPE functions are now used dynamically - no need for hardcoded LUTs


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EGAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0., qk_scale=None,attention=0,depth_index=0):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        
        self.scale=1
        self.bias_level = 0.1

        self.sigmoid = nn.Sigmoid()
        self.d_idx = depth_index
        self.idx = idx 
        self.relu = nn.ReLU()
        if attention == 0:
            self.attention = 'L'
        
        if self.attention == 'L':
            # We assume split_size (stripe_with) is 1   
            assert self.split_size == 1, "split_size is not 1" 

            if idx == 0:  # Horizontal Self-Attention
                W_sp, H_sp = self.resolution[1], self.split_size
                # Compute RPE dynamically instead of using pre-computed LUT
                self.RPE = HRPE(self.num_heads, self.resolution[0], self.resolution[1], self.split_size)
            elif idx == 1:  # Vertical Self-Attention
                H_sp, W_sp = self.resolution[0], self.split_size
                # Compute RPE dynamically instead of using pre-computed LUT
                self.RPE = VRPE(self.num_heads, self.resolution[0], self.resolution[1], self.split_size)
            else:
                print ("ERROR MODE", idx)
                exit(0)


        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.attn_drop = nn.Dropout(attn_drop)

    def im2hvwin(self, x):
        B, N, C = x.shape
        # H = W = int(np.sqrt(N))
        H, W = self.resolution[0], self.resolution[1]
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()  # B, H//H_sp, W//W_sp, H_sp * W_sp, C -> B, H//H_sp, W//W_sp, H_sp*W_sp, heads, C//heads
        return x

    def get_v(self, x): # LePE is not used for EGformer
        B, N, C = x.shape
        # H = W = int(np.sqrt(N))
        H, W = self.resolution[0], self.resolution[1]
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp) ### B', C, H', W'

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp* self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x

    def forward(self, qkv,res_x):
        """
        x: B L C
        """
        q,k,v = qkv[0], qkv[1], qkv[2]
        
        pi = torch.acos(torch.zeros(1)).item() * 2

        ### Img2Window
        # H = W = self.resolution
        H, W = self.resolution[0], self.resolution[1]
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        
        q = self.im2hvwin(q)
        k = self.im2hvwin(k)
        v = self.get_v(v)

        if self.attention == 'L': 
            self.RPE = self.RPE.cuda(q.get_device())
            
            Re = int(q.size(0) / self.RPE.size(0))
    
            attn = q @ k.transpose(-2, -1)
            
            # ERPE
            attn = attn + self.bias_level * self.RPE.repeat(Re,1,1,1)
 
            M = torch.abs(attn) # Importance level of each local attention

            # DAS
            attn = F.normalize(attn,dim=-1) * pi/2
            attn = (1 - torch.cos(attn)) # Square of the distance from the baseline point. By setting the baseline point as (1/sqrt(2),0,pi/2), DAS can get equal score range (0,1) for both vertical & horizontal direction. 

            # EaAR
            M = torch.mean(M,dim=(1,2,3),keepdim=True)   # Check this part to utilize batch size > 1 per GPU.
           
            M = M / torch.max(M)
            M = torch.clamp(M, min=0.5)

            attn = attn * M  

            attn = self.attn_drop(attn)

            x = (attn @ v) 

        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C
        
        # EaAR
        res_x = res_x.reshape(-1,self.H_sp*self.W_sp,C).unsqueeze(1)
        res_x = res_x * (1 - M)
        res_x = res_x.view(B,-1,C)


        return x + res_x


class EGBlock(nn.Module):

    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False,attention=0,idx=0,depth_index=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)
        
        self.branch_num = 1
       
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.idx = idx
        self.attention = attention
       
        self.attns = nn.ModuleList([
            EGAttention(
                dim, resolution=self.patches_resolution, idx = self.idx,
                split_size=split_size, num_heads=num_heads, dim_out=dim,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,attention=self.attention, depth_index = depth_index)
            for i in range(self.branch_num)])
        

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        # H = W = self.patches_resolution
        H, W = self.patches_resolution[0], self.patches_resolution[1]
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        
        attened_x = self.attns[0](qkv,x)

        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class EGTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    Args:
        depth       : Number of blocks in each stage
        split_size  : Width(Height) of stripe size in each stage
        num_heads   : Number of heads in each stage
        hybrid      : Whether to use hybrid patch embedding (ResNet50)/ Not used
    """
    def __init__(self, img_size=[512, 1024], patch_size=16, in_chans=3, out_chans=1, num_classes=1000, embed_dim=96, depth=[2,2,6,2], split_size = [3,5,7],
                 num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2], mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, use_chk=False, hybrid=False):
        super().__init__()
        self.use_chk = use_chk
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        heads=num_heads
        ## Fine until here        

        self.patch_embed= nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 3, 2, 1),
            Rearrange('b c h w -> b (h w) c', h = img_size[0]//2, w = img_size[1]//2),
            #lambda x: x.view(x.shape[0], x.shape[1], -1).transpose(1, 2),  # b c h w -> b (h w) c
            nn.LayerNorm(embed_dim)
        )

        #### Panoformer variables - use parameters instead of hardcoded values
        img_size_pano=256; depths=[2, 2, 2, 2, 2, 2, 2, 2, 2]; num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2]
        win_size=8; qkv_bias=True; qk_scale=None; drop_rate=0.; attn_drop_rate=0.; drop_path_rate=0.1
        norm_layer=nn.LayerNorm; patch_norm=True; use_checkpoint=False; token_projection='linear'; token_mlp='leff'; se_layer=False
        dowsample=Downsample; upsample=Upsample

        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim  # Use the parameter, not hardcoded value
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio  # Use the parameter, not hardcoded value
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size
        ## Fine until here
        # self.ref_point256x512 = genSamplingPattern(256, 512, 3, 3).cuda()
        # self.ref_point128x256 = genSamplingPattern(128, 256, 3, 3).cuda()
        # self.ref_point64x128 = genSamplingPattern(64, 128, 3, 3).cuda()
        # self.ref_point32x64 = genSamplingPattern(32, 64, 3, 3).cuda()
        # self.ref_point16x32 = genSamplingPattern(16, 32, 3, 3).cuda()
        # Generate reference points dynamically based on image size
        self.img_size = img_size
        self.ref_point256x512 = None
        self.ref_point128x256 = None
        self.ref_point64x128 = None
        self.ref_point32x64 = None
        self.ref_point16x32 = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]


        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule
        
        ## Fine until here
        self.stage1 = nn.ModuleList([
            BasicPanoformerLayer(dim=curr_dim, output_dim=curr_dim,
                                                input_resolution=(img_size[0]//2, img_size[1]//2),
                                                depth=depths[0],
                                                num_heads=num_heads[0],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[int(sum(depths[:0])):int(sum(depths[:1]))],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer, ref_point=self.ref_point256x512, flag = 0) if i%2==0 else 
            ## Fine until here
            EGBlock(
                dim=curr_dim, num_heads=heads[0], reso=[img_size[0]//2, img_size[1]//2], mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer,attention = 0, idx= i%2, depth_index = 0)
            for i in range(depth[0])])
        self.downsample1 = Merge_Block(curr_dim, curr_dim *2 , resolution = [img_size[0]//2, img_size[1]//2])
       
        # Tuning into decoder dimension

        curr_dim = curr_dim*2
        self.stage2 = nn.ModuleList([
            BasicPanoformerLayer(dim=curr_dim, output_dim=curr_dim,
                                                input_resolution=(img_size[0]//4, img_size[1]//4),
                                                depth=depths[1],
                                                num_heads=num_heads[1],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer, ref_point=self.ref_point128x256, flag = 0) if i%2==0 else 
            EGBlock(
                dim=curr_dim, num_heads=heads[1], reso=[img_size[0]//4, img_size[1]//4], mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1])+i], norm_layer=norm_layer,attention = 0, idx= i%2, depth_index = 1)
            for i in range(depth[1])])
        self.downsample2 = Merge_Block(curr_dim, curr_dim*2, resolution = [img_size[0]//4, img_size[1]//4])
       
        # Update curr_dim after downsample2
        curr_dim = curr_dim * 2
        
        # Skip stage3 and stage4 - go directly to bottleneck
        # curr_dim is now 256 (64 -> 128 -> 256)
        self.bottleneck = nn.ModuleList([
            EGBlock(
                dim=curr_dim, num_heads=heads[2], reso=[img_size[0]//8, img_size[1]//8], mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2])+i], norm_layer=norm_layer, last_stage=False,attention = 0, idx= i%2, depth_index = 2)
            for i in range(depth[2])])
 
        self.upsample5 = Merge_Block(curr_dim, curr_dim // 2, resolution = [img_size[0]//8, img_size[1]//8],scale_factor=2.)
        curr_dim = curr_dim // 2



        self.red_ch = []
        self.set_dim = []
        self.rearrange = []
        curr_dim = curr_dim
        self.dec_stage5 = nn.ModuleList(
            [EGBlock(
                dim=curr_dim, num_heads=heads[4], reso=[img_size[0]//4, img_size[1]//4], mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2])+i], norm_layer=norm_layer, last_stage=False, attention = 0, idx= i%2, depth_index = 3)
            for i in range(depth[2])])

        self.upsample6 = Merge_Block(curr_dim, curr_dim // 2, resolution = [img_size[0]//4, img_size[1]//4],scale_factor=2.)

        self.tune5 = Tune_Block(curr_dim * 2  ,curr_dim, resolution = [img_size[0]//4, img_size[1]//4]) # Tune_5
        self.set_dim.append(To_BCHW(resolution = [img_size[0]//4, img_size[1]//4])) # BCHW_5
        self.rearrange.append(Rearrange('b c h w -> b (h w) c', h = img_size[0]//4, w = img_size[1]//4))


        
        curr_dim = curr_dim // 2
        self.dec_stage6 = nn.ModuleList(
            [EGBlock(
                dim=curr_dim, num_heads=heads[5], reso=[img_size[0]//2, img_size[1]//2], mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:3])+i], norm_layer=norm_layer,attention = 0, idx= i%2, depth_index = 2)
            for i in range(depth[3])])

        self.upsample7 = Merge_Block(curr_dim , curr_dim //2, resolution = [img_size[0]//2, img_size[1]//2],scale_factor=2.)
 
        self.tune6 = Tune_Block(curr_dim * 2  ,curr_dim, resolution = [img_size[0]//2, img_size[1]//2]) # Tune_6

        self.set_dim.append(To_BCHW(resolution = [img_size[0]//2, img_size[1]//2])) # BCHW_6
        self.rearrange.append(Rearrange('b c h w -> b (h w) c', h = img_size[0]//2, w = img_size[1]//2))
        
        
        # Skip dec_stage7 and dec_stage8 - removed to reduce parameters
    
        self.tune_final = Tune_Block_Final(curr_dim,curr_dim, resolution = [img_size[0]//2, img_size[1]//2])
        # Tuning into decoder dimension

        self.norm = norm_layer(curr_dim)

        self.output_conv = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(curr_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_chans, kernel_size=1, stride=1, padding=0),
        )

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _generate_ref_points(self, device):
        """Generate reference points dynamically based on image size"""
        if self.ref_point256x512 is None:
            h, w = self.img_size[0]//2, self.img_size[1]//2
            self.ref_point256x512 = genSamplingPattern(h, w, 3, 3).to(device)
        if self.ref_point128x256 is None:
            h, w = self.img_size[0]//4, self.img_size[1]//4
            self.ref_point128x256 = genSamplingPattern(h, w, 3, 3).to(device)
        if self.ref_point64x128 is None:
            h, w = self.img_size[0]//8, self.img_size[1]//8
            self.ref_point64x128 = genSamplingPattern(h, w, 3, 3).to(device)
        if self.ref_point32x64 is None:
            h, w = self.img_size[0]//16, self.img_size[1]//16
            self.ref_point32x64 = genSamplingPattern(h, w, 3, 3).to(device)
        if self.ref_point16x32 is None:
            h, w = self.img_size[0]//32, self.img_size[1]//32
            self.ref_point16x32 = genSamplingPattern(h, w, 3, 3).to(device)

    def forward_features(self, x):
        features = []
        B = x.shape[0]
        features= []
        
        # Generate reference points for the current device
        device = x.device
        self._generate_ref_points(device)
       

    ########## Encoder (Reduced to 3 stages)       
        x = self.patch_embed(x)
        
        for blk in self.stage1:
            x = blk(x)
        features.append(x)
        x = self.downsample1(x)

        for blk in self.stage2:
            x = blk(x)
        features.append(x)
        x = self.downsample2(x)

        # Skip stage3 and stage4, go directly to bottleneck
        for blk in self.bottleneck:
            x = blk(x)

    ######## Decoder (Reduced to 2 stages)
        x = self.upsample5(x)
        for blk in self.dec_stage5:
            x = blk(x)
        x = torch.cat((self.set_dim[0](features[1]), self.set_dim[0](x)), dim=1)
        x = self.tune5(x)
        x = self.rearrange[0](x)

        x = self.upsample6(x)
        for blk in self.dec_stage6:
            x = blk(x)
        x = torch.cat((self.set_dim[1](features[0]), self.set_dim[1](x)), dim=1)
        x = self.tune6(x)
        x = self.rearrange[1](x)

        # Skip dec_stage7 and dec_stage8

        # EGformer Output Projection
        x = self.tune_final(x)
        x = self.output_conv(x)

        return x

    def forward(self, x):
        out = self.forward_features(x)
        return out


# Model registration for timm
@register_model
def egformer(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    """EGformer tiny model"""
    # Filter out timm-specific parameters
    egformer_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['cache_dir', 'features_only', 'out_indices', 'scriptable', 'exportable']}
    model = EGTransformer(**egformer_kwargs)
    return model
