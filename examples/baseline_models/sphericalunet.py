import torch 
from deepsphere.models.spherical_unet.unet_model import SphericalUNet

class SUNet(torch.nn.Module):
    def __init__(self, shape, input_channels, output_channels, embed_size, depth, laplacian_type, kernel_size, ratio, pool_factor):
        super(SUNet, self).__init__()
        if isinstance(shape, int):
            shape = [shape, shape]
        N = shape[0] * shape[1]
        self.output_channels = output_channels
        self.unet = SphericalUNet(pooling_class="equiangular", N=N, depth=depth, laplacian_type=laplacian_type, kernel_size=kernel_size, ratio=ratio, pool_factor=pool_factor, output_channels=output_channels, input_channels=input_channels, embed_size=embed_size)

    def forward(self, x):
        B, C, H, W = x.shape 
        x = x.view(B, C, H*W)
        x = x.permute(0, 2, 1)
        output = self.unet(x).permute(0, 2, 1)
        return output.view(B, self.output_channels, H, W)