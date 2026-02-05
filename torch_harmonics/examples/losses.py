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

import torch
import torch.nn as nn
import torch.amp as amp
import torch.nn.functional as F
from typing import Optional
from abc import ABC, abstractmethod

from torch_harmonics.quadrature import precompute_latitudes


def get_quadrature_weights(nlat: int, nlon: int, grid: str, tile: bool = False, normalized: bool = True) -> torch.Tensor:
    # area weights
    _, q = precompute_latitudes(nlat=nlat, grid=grid)
    q = q.reshape(-1, 1) * 2 * torch.pi / nlon

    # numerical precision can be an issue here, make sure it sums to 1:
    if normalized:
        q = q / torch.sum(q) / float(nlon)

    if tile:
        q = torch.tile(q, (1, nlon)).contiguous()

    return q.to(torch.float32)


class DiceLossS2(nn.Module):
    """
    Dice loss for spherical segmentation tasks.

    Parameters
    -----------
    nlat : int
        Number of latitude points
    nlon : int
        Number of longitude points
    grid : str, optional
        Grid type, by default "equiangular"
    weight : torch.Tensor, optional
        Class weights, by default None
    smooth : float, optional
        Smoothing factor, by default 0
    ignore_index : int, optional
        Index to ignore in loss computation, by default -100
    mode : str, optional
        Aggregation mode ("micro" or "macro"), by default "micro"
    """

    def __init__(self, nlat: int, nlon: int, grid: str = "equiangular", weight: torch.Tensor = None, smooth: float = 0, ignore_index: int = -100, mode: str = "micro"):

        super().__init__()

        self.smooth = smooth
        self.ignore_index = ignore_index
        self.mode = mode

        # area weights
        q = get_quadrature_weights(nlat=nlat, nlon=nlon, grid=grid)
        self.register_buffer("quad_weights", q)

        if weight is None:
            self.weight = None
        else:
            self.register_buffer("weight", weight.unsqueeze(0))

    def forward(self, prd: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
        prd = nn.functional.softmax(prd, dim=1)

        # mask values
        if self.ignore_index is not None:
            mask = torch.where(tar == self.ignore_index, 0, 1)
            prd = prd * mask.unsqueeze(1)
            tar = tar * mask

        # one hot encode
        taroh = nn.functional.one_hot(tar, num_classes=prd.shape[1]).permute(0, 3, 1, 2)

        # compute numerator and denominator
        intersection = torch.sum((prd * taroh) * self.quad_weights, dim=(-2, -1))
        union = torch.sum((prd + taroh) * self.quad_weights, dim=(-2, -1))

        if self.mode == "micro":
            if self.weight is not None:
                intersection = torch.sum(intersection * self.weight, dim=1)
                union = torch.sum(union * self.weight, dim=1)
            else:
                intersection = torch.mean(intersection, dim=1)
                union = torch.mean(union, dim=1)

        # compute score
        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        # compute average over classes
        if self.mode == "macro":
            if self.weight is not None:
                dice = torch.sum(dice * self.weight, dim=1)
            else:
                dice = torch.mean(dice, dim=1)

        # average over batch
        dice = torch.mean(dice)

        return 1 - dice


class CrossEntropyLossS2(nn.Module):
    """
    Cross-entropy loss for spherical classification tasks.

    Parameters
    -----------
    nlat : int
        Number of latitude points
    nlon : int
        Number of longitude points
    grid : str, optional
        Grid type, by default "equiangular"
    weight : torch.Tensor, optional
        Class weights, by default None
    smooth : float, optional
        Label smoothing factor, by default 0
    ignore_index : int, optional
        Index to ignore in loss computation, by default -100
    """

    def __init__(self, nlat: int, nlon: int, grid: str = "equiangular", weight: torch.Tensor = None, smooth: float = 0, ignore_index: int = -100):

        super().__init__()

        self.smooth = smooth
        self.ignore_index = ignore_index

        if weight is None:
            self.weight = None
        else:
            self.register_buffer("weight", weight)

        q = get_quadrature_weights(nlat=nlat, nlon=nlon, grid=grid)
        self.register_buffer("quad_weights", q)

    def forward(self, prd: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:

        # compute log softmax
        logits = nn.functional.log_softmax(prd, dim=1)
        ce = nn.functional.cross_entropy(logits, tar, weight=self.weight, reduction="none", ignore_index=self.ignore_index, label_smoothing=self.smooth)
        ce = (ce * self.quad_weights).sum(dim=(-1, -2))
        ce = torch.mean(ce)

        return ce


class FocalLossS2(nn.Module):
    """
    Focal loss for spherical classification tasks.

    Parameters
    -----------
    nlat : int
        Number of latitude points
    nlon : int
        Number of longitude points
    grid : str, optional
        Grid type, by default "equiangular"
    weight : torch.Tensor, optional
        Class weights, by default None
    smooth : float, optional
        Label smoothing factor, by default 0
    ignore_index : int, optional
        Index to ignore in loss computation, by default -100
    """

    def __init__(self, nlat: int, nlon: int, grid: str = "equiangular", weight: torch.Tensor = None, smooth: float = 0, ignore_index: int = -100):

        super().__init__()

        self.smooth = smooth
        self.ignore_index = ignore_index

        if weight is None:
            self.weight = None
        else:
            self.register_buffer("weight", weight)

        q = get_quadrature_weights(nlat=nlat, nlon=nlon, grid=grid)
        self.register_buffer("quad_weights", q)

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, alpha: float = 0.25, gamma: float = 2):

        # compute logits
        logits = nn.functional.log_softmax(prd, dim=1)

        # w = (1.0 - nn.functional.softmax(prd, dim=-3)).pow(gamma)
        # w = torch.where(tar == self.ignore_index, 0.0, w.gather(-3, tar.unsqueeze(-3)).squeeze(-3))
        ce = nn.functional.cross_entropy(logits, tar, weight=self.weight, reduction="none", ignore_index=self.ignore_index, label_smoothing=self.smooth)
        fl = alpha * (1 - torch.exp(-ce)) ** gamma * ce
        # fl = w * ce
        fl = (fl * self.quad_weights).sum(dim=(-1, -2))
        fl = fl.mean()

        return fl


class SphericalLossBase(nn.Module, ABC):
    """Abstract base class for spherical losses that handles common initialization and integration."""

    def __init__(self, nlat: int, nlon: int, grid: str = "equiangular", normalized: bool = True):
        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid

        # get quadrature weights - these sum to 1!
        q = get_quadrature_weights(nlat=nlat, nlon=nlon, grid=grid, normalized=normalized)
        self.register_buffer("quad_weights", q)

    def _integrate_sphere(self, ugrid, mask=None):
        if mask is None:
            out = torch.sum(ugrid * self.quad_weights, dim=(-2, -1))
        elif mask is not None:
            out = torch.sum(mask * ugrid * self.quad_weights, dim=(-2, -1)) / torch.sum(mask * self.quad_weights, dim=(-2, -1))
        return out

    @abstractmethod
    def _compute_loss_term(self, prd: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
        """Abstract method that must be implemented by child classes to compute loss terms.

        Args:
            prd (torch.Tensor): Prediction tensor
            tar (torch.Tensor): Target tensor

        Returns:
            torch.Tensor: Computed loss term before integration
        """
        pass

    def _post_integration_hook(self, loss: torch.Tensor) -> torch.Tensor:
        """Post-integration hook. Commonly used for the roots in Lp norms"""
        return loss

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Common forward pass that handles masking and reduction.

        Args:
            prd (torch.Tensor): Prediction tensor
            tar (torch.Tensor): Target tensor
            mask (Optional[torch.Tensor], optional): Mask tensor. Defaults to None.

        Returns:
            torch.Tensor: Final loss value
        """
        loss_term = self._compute_loss_term(prd, tar)
        # Integrate over the sphere for each item in the batch
        loss = self._integrate_sphere(loss_term, mask)
        # potentially call root
        loss = self._post_integration_hook(loss)
        # Average the loss over the batch dimension
        return torch.mean(loss)


class SquaredL2LossS2(SphericalLossBase):
    def _compute_loss_term(self, prd: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
        return torch.square(prd - tar)


class L1LossS2(SphericalLossBase):
    def _compute_loss_term(self, prd: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
        return torch.abs(prd - tar)


class L2LossS2(SquaredL2LossS2):
    def _post_integration_hook(self, loss: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(loss)


class W11LossS2(SphericalLossBase):
    def __init__(self, nlat: int, nlon: int, grid: str = "equiangular"):
        super().__init__(nlat=nlat, nlon=nlon, grid=grid)
        # Set up grid and domain for FFT
        l_phi = 2 * torch.pi  # domain size
        l_theta = torch.pi  # domain size

        k_phi = torch.fft.fftfreq(nlon, d=l_phi / (2 * torch.pi * nlon))
        k_theta = torch.fft.fftfreq(nlat, d=l_theta / (2 * torch.pi * nlat))
        k_theta_mesh, k_phi_mesh = torch.meshgrid(k_theta, k_phi, indexing="ij")
        self.register_buffer("k_phi_mesh", k_phi_mesh)
        self.register_buffer("k_theta_mesh", k_theta_mesh)

    def _compute_loss_term(self, prd: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
        prdtype = prd.dtype
        with amp.autocast(device_type="cuda", enabled=False):
            prd = prd.to(torch.float32)
            prd_prime_fft2_phi_h = torch.fft.ifft2(1j * self.k_phi_mesh * torch.fft.fft2(prd)).real
            prd_prime_fft2_theta_h = torch.fft.ifft2(1j * self.k_theta_mesh * torch.fft.fft2(prd)).real

            tar_prime_fft2_phi_h = torch.fft.ifft2(1j * self.k_phi_mesh * torch.fft.fft2(tar)).real
            tar_prime_fft2_theta_h = torch.fft.ifft2(1j * self.k_theta_mesh * torch.fft.fft2(tar)).real

        # Return the element-wise loss term
        return torch.abs(prd_prime_fft2_phi_h - tar_prime_fft2_phi_h) + torch.abs(prd_prime_fft2_theta_h - tar_prime_fft2_theta_h)


class NormalLossS2(SphericalLossBase):
    """Combined L1 and Surface Normal Consistency Loss for spherical data.

    This loss function combines an L1 loss term with a surface normal alignment term.

    The loss consists of:
    1. L1 Loss: Absolute difference between predicted and target values
    2. Normal Consistency Loss: 1 - cosine similarity between surface normals
       (equivalent to cosine distance between normal vectors)

    Surface normals are computed by calculating gradients in latitude and longitude
    directions using FFT, then constructing 3D normal vectors that are normalized.

    Parameters
    ----------
    nlat : int
        Number of latitude points
    nlon : int
        Number of longitude points
    grid : str, optional
        Grid type, by default "equiangular"

    Returns
    -------
    torch.Tensor
        Combined loss term
    """

    def __init__(self, nlat: int, nlon: int, grid: str = "equiangular"):
        super().__init__(nlat=nlat, nlon=nlon, grid=grid)
        # Set up grid and domain for FFT
        l_phi = 2 * torch.pi  # domain size
        l_theta = torch.pi  # domain size

        k_phi = torch.fft.fftfreq(nlon, d=l_phi / (2 * torch.pi * nlon))
        k_theta = torch.fft.fftfreq(nlat, d=l_theta / (2 * torch.pi * nlat))
        k_theta_mesh, k_phi_mesh = torch.meshgrid(k_theta, k_phi, indexing="ij")
        self.register_buffer("k_phi_mesh", k_phi_mesh)
        self.register_buffer("k_theta_mesh", k_theta_mesh)

    def compute_gradients(self, x):
        # Make sure x is reshaped to have a batch dimension if it's missing
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension

        x_prime_fft2_phi_h = torch.fft.ifft2(1j * self.k_phi_mesh * torch.fft.fft2(x)).real
        x_prime_fft2_theta_h = torch.fft.ifft2(1j * self.k_theta_mesh * torch.fft.fft2(x)).real
        return x_prime_fft2_theta_h, x_prime_fft2_phi_h

    def compute_normals(self, x):
        x = x.to(torch.float32)
        # Ensure x has a batch dimension
        if x.dim() == 2:
            x = x.unsqueeze(0)

        grad_lat, grad_lon = self.compute_gradients(x)

        # Create 3D normal vectors
        ones = torch.ones_like(x)
        normals = torch.stack([-grad_lon, -grad_lat, ones], dim=1)

        # Normalize along component dimension
        normals = F.normalize(normals, p=2, dim=1)
        return normals

    def _compute_loss_term(self, prd: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
        # Handle dimensions for both prediction and target
        # Ensure we have at least a batch dimension
        if prd.dim() == 2:
            prd = prd.unsqueeze(0)
        if tar.dim() == 2:
            tar = tar.unsqueeze(0)

        # For 4D tensors (batch, channel, height, width), remove channel if it's 1
        if prd.dim() == 4 and prd.size(1) == 1:
            prd = prd.squeeze(1)
        if tar.dim() == 4 and tar.size(1) == 1:
            tar = tar.squeeze(1)

        pred_normals = self.compute_normals(prd)
        tar_normals = self.compute_normals(tar)

        # Compute cosine similarity
        normal_loss = 1 - torch.sum(pred_normals * tar_normals, dim=1, keepdim=True)
        return normal_loss
