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

from torch_harmonics.quadrature import _precompute_latitudes

class DiceLossS2(nn.Module):
    def __init__(self, nlat: int, nlon: int, grid: str = "equiangular", weight: torch.Tensor = None, smooth: float = 0, ignore_index: int = -100, mode: str = "micro"):

        super().__init__()

        self.smooth = smooth
        self.ignore_index = ignore_index
        self.mode = mode

        # area weights
        _, q = _precompute_latitudes(nlat=nlat, grid=grid)
        q = q.reshape(-1, 1) * 2 * torch.pi / nlon

        # numerical precision can be an issue here, make sure it sums to 1:
        q = q / torch.sum(q)
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

        _, q = _precompute_latitudes(nlat=nlat, grid=grid)

        q = q.reshape(-1, 1) * 2 * torch.pi / nlon

        # numerical precision can be an issue here, make sure it sums to 1:
        q = q / torch.sum(q)

        if weight is None:
            self.weight = None
        else:
            self.register_buffer("weight", weight)

        self.register_buffer("quad_weights", q)

    def forward(self, prd: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:

        ce = nn.functional.cross_entropy(prd, tar, weight=self.weight, reduction="none", ignore_index=self.ignore_index, label_smoothing=self.smooth)
        ce = (ce * self.quad_weights).sum(dim=(-1, -2))
        ce = torch.mean(ce)

        return ce


class FocalLossS2(nn.Module):

    def __init__(self, nlat: int, nlon: int, grid: str = "equiangular", ignore_index: int = -100):

        super().__init__()

        self.ignore_index = ignore_index

        _, q = _precompute_latitudes(nlat=nlat, grid=grid)

        q = q.reshape(-1, 1) * 2 * torch.pi / nlon

        self.register_buffer("quad_weights", q)

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, alpha: float = 0.25, gamma: float = 2):

        # w = (1.0 - nn.functional.softmax(prd, dim=-3)).pow(gamma)
        # w = torch.where(tar == self.ignore_index, 0.0, w.gather(-3, tar.unsqueeze(-3)).squeeze(-3))
        ce = nn.functional.cross_entropy(prd, tar, weight=None, reduction="none", ignore_index=self.ignore_index)
        fl = alpha * (1 - torch.exp(-ce)) ** gamma * ce
        # fl = w * ce
        fl = (fl * self.quad_weights).sum(dim=(-1, -2)) / 4 / torch.pi
        fl = fl.mean()

        return fl
