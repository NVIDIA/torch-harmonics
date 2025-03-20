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
import torch.nn.functional as F
from typing import Optional

from torch_harmonics.quadrature import _precompute_latitudes

def get_quadrature_weights(nlat: int, nlon: int, grid: str) -> torch.Tensor:
    # area weights
    _, q = _precompute_latitudes(nlat=nlat, grid=grid)
    q = q.reshape(-1, 1) * 2 * torch.pi / nlon

    # numerical precision can be an issue here, make sure it sums to 1:
    q = q / torch.sum(q) / float(nlon)

    return q.to(torch.float32)


def label_smoothed_nll_loss(
    lprobs: torch.Tensor,
    target: torch.Tensor,
    epsilon: float,
    ignore_index=None,
    reduction="mean",
    dim=-1,
) -> torch.Tensor:
    """NLL loss with label smoothing

    References:
        https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py

    Args:
        lprobs (torch.Tensor): Log-probabilities of predictions (e.g after log_softmax)

    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(dim)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        target = target.masked_fill(pad_mask, 0)
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        # nll_loss.masked_fill_(pad_mask, 0.0)
        # smooth_loss.masked_fill_(pad_mask, 0.0)
        nll_loss = nll_loss.masked_fill(pad_mask, 0.0)
        smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)
    else:
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        nll_loss = nll_loss.squeeze(dim)
        smooth_loss = smooth_loss.squeeze(dim)

    if reduction == "sum":
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == "mean":
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()

    eps_i = epsilon / lprobs.size(dim)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss

class SoftCrossEntropyLoss(nn.Module):
    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(
        self,
        reduction: str = "mean",
        smooth_factor: float = 0.0,
        ignore_index: Optional[int] = -100,
        dim: int = 1,
    ):
        """Drop-in replacement for torch.nn.CrossEntropyLoss with label_smoothing

        Args:
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 0] -> [0.9, 0.05, 0.05])

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(y_pred, dim=self.dim)
        return label_smoothed_nll_loss(
            log_prob,
            y_true,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )

class DiceLossS2(nn.Module):
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

    def __init__(self, nlat: int, nlon: int, grid: str = "equiangular", ignore_index: int = -100):

        super().__init__()

        self.ignore_index = ignore_index
        
        q = get_quadrature_weights(nlat=nlat, nlon=nlon, grid=grid)
        self.register_buffer("quad_weights", q)

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, alpha: float = 0.25, gamma: float = 2):

        # compute logits
        logits = nn.functional.log_softmax(prd, dim=1)
        
        # w = (1.0 - nn.functional.softmax(prd, dim=-3)).pow(gamma)
        # w = torch.where(tar == self.ignore_index, 0.0, w.gather(-3, tar.unsqueeze(-3)).squeeze(-3))
        ce = nn.functional.cross_entropy(logits, tar, weight=None, reduction="none", ignore_index=self.ignore_index)
        fl = alpha * (1 - torch.exp(-ce)) ** gamma * ce
        # fl = w * ce
        fl = (fl * self.quad_weights).sum(dim=(-1, -2))
        fl = fl.mean()

        return fl
