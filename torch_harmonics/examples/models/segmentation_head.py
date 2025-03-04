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

import abc

import math

import torch
import torch.nn as nn
import torch.amp as amp

from torch_harmonics.examples.models.s2transformer import SphericalTransformer
from torch_harmonics.examples.models.lsno import LocalSphericalNeuralOperator
from torch_harmonics.examples.models.sfno import SphericalFourierNeuralOperator

from ._layers import MLP

from functools import partial


class SegmentationWrapper(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, num_classes, out_chans=3, activation_function=nn.ReLU):
        super().__init__()
        self.num_classes = num_classes

        self.activation_function = activation_function

        self.softmax = nn.Softmax(dim=-3)  # assumes dim = (...,C,H,W), so dim=-3 gets C

        print(num_classes)

        self.mlp = MLP(in_features=out_chans, out_features=num_classes, hidden_features=out_chans, act_layer=self.activation_function)

        # apply softmax to the output of the MLP
        self.segmentation_head = nn.Sequential(self.mlp, self.softmax)

    def forward(self, x):
        x = self.backbone(x)
        x = self.segmentation_head(x)
        return x


class SphericalTransformerForSegmentation(SegmentationWrapper):

    def __init__(self, num_classes: int, embed_dim: int = 256, **kwargs):

        kwargs["out_chans"] = embed_dim
        kwargs["embed_dim"] = embed_dim

        super().__init__(num_classes, kwargs["out_chans"])

        self.backbone = SphericalTransformer(**kwargs)


class LocalSphericalNeuralOperatorForSegmentation(SegmentationWrapper):

    def __init__(self, num_classes: int, embed_dim: int = 256, **kwargs):
        kwargs["out_chans"] = embed_dim
        kwargs["embed_dim"] = embed_dim

        super().__init__(num_classes, kwargs["out_chans"])

        self.backbone = LocalSphericalNeuralOperator(**kwargs)


class SphericalFourierNeuralOperatorForSegmentation(SegmentationWrapper):

    def __init__(self, num_classes: int, embed_dim: int = 256, **kwargs):
        kwargs["out_chans"] = embed_dim
        kwargs["embed_dim"] = embed_dim

        super().__init__(num_classes, kwargs["out_chans"])

        self.backbone = SphericalFourierNeuralOperator(**kwargs)
