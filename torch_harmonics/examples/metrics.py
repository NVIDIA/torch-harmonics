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

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .losses import get_quadrature_weights


# routine to compute multiclass labels on the sphere
# the routine follows the implementation in
# https://github.com/qubvel-org/segmentation_models.pytorch/blob/4aa36c6ad13f8a12552e4ea4131af2a86e564962/segmentation_models_pytorch/metrics/functional.py
# but uses quadrature weights
def _get_stats_multiclass(
    output: torch.LongTensor,
    target: torch.LongTensor,
    num_classes: int,
    quad_weights: torch.Tensor,
    ignore_index: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute multiclass statistics (TP, FP, FN, TN) on the sphere using quadrature weights.
    
    This function computes true positives, false positives, false negatives, and true negatives
    for multiclass classification on spherical data, properly weighted by quadrature weights
    to account for the spherical geometry.
    
    Parameters
    -----------
    output : torch.LongTensor
        Predicted class labels
    target : torch.LongTensor
        Ground truth class labels
    num_classes : int
        Number of classes in the classification task
    quad_weights : torch.Tensor
        Quadrature weights for spherical integration
    ignore_index : Optional[int]
        Index to ignore in the computation (e.g., for padding or invalid regions)
        
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Tuple containing (tp_count, fp_count, fn_count, tn_count) for each class
    """
    batch_size, *dims = output.shape
    num_elements = torch.prod(torch.tensor(dims)).long()

    if ignore_index is not None:
        ignore = target == ignore_index
        output = torch.where(ignore, -1, output)
        target = torch.where(ignore, -1, target)
        ignore_per_sample = ignore.view(batch_size, -1).sum(1)

    tp_count = torch.zeros(batch_size, num_classes, dtype=torch.float32, device=output.device)
    fp_count = torch.zeros(batch_size, num_classes, dtype=torch.float32, device=output.device)
    fn_count = torch.zeros(batch_size, num_classes, dtype=torch.float32, device=output.device)
    tn_count = torch.zeros(batch_size, num_classes, dtype=torch.float32, device=output.device)

    matched = target == output
    not_matched = target != output
    for i in range(batch_size):
        matched_i = matched[i, ...]
        not_matched_i = not_matched[i, ...]
        target_i = target[i, ...]
        output_i = output[i, ...]
        for c in range(num_classes):
            # compute weights
            qwt_c = quad_weights[target_i == c]
            qwo_c = quad_weights[output_i == c]

            # true positives
            tp_count[i, c] = torch.sum(matched_i[target_i == c] * qwt_c)
            # false positives
            fp_count[i, c] = torch.sum(not_matched_i[output_i == c] * qwo_c)
            # false negatives
            fn_count[i, c] = torch.sum(not_matched_i[target_i == c] * qwt_c)

    # true negatives is the leftovers
    tn_count = torch.sum(quad_weights) - tp_count - fp_count - fn_count
    return tp_count, fp_count, fn_count, tn_count


def _predict_classes(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert logits to class predictions using softmax and argmax.
    
    Parameters
    -----------
    logits : torch.Tensor
        Input logits tensor
        
    Returns
    -------
    torch.Tensor
        Predicted class labels
    """
    return torch.argmax(torch.softmax(logits, dim=1), dim=1, keepdim=False)


class BaseMetricS2(nn.Module):
    """
    Base class for spherical metrics that properly handle spherical geometry.
    
    This class provides the foundation for computing metrics on spherical data
    by using quadrature weights to account for the non-uniform area distribution
    on the sphere.
    
    Parameters
    -----------
    nlat : int
        Number of latitude points
    nlon : int
        Number of longitude points
    grid : str, optional
        Grid type ("equiangular", "legendre-gauss", etc.), by default "equiangular"
    weight : torch.Tensor, optional
        Class weights for weighted averaging, by default None
    ignore_index : int, optional
        Index to ignore in computations, by default -100
    mode : str, optional
        Averaging mode ("micro" or "macro"), by default "micro"
    """
    
    def __init__(self, nlat: int, nlon: int, grid: str = "equiangular", weight: torch.Tensor = None, ignore_index: int = -100, mode: str = "micro"):
        super().__init__()

        self.ignore_index = ignore_index
        self.mode = mode

        # area weights
        q = get_quadrature_weights(nlat=nlat, nlon=nlon, grid=grid, tile=True)
        self.register_buffer("quad_weights", q)

        if weight is None:
            self.weight = None
        else:
            self.register_buffer("weight", weight.unsqueeze(0))

    def _forward(self, pred: torch.Tensor, truth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # convert logits to class predictions
        pred_class = _predict_classes(pred)

        # get true positive, false positive, etc
        tp, fp, fn, tn = _get_stats_multiclass(pred_class, truth, pred.shape[1], self.quad_weights, self.ignore_index)

        # compute averages:
        if self.mode == "micro":
            if self.weight is not None:
                # weighted average
                tp = torch.sum(tp * self.weight)
                fp = torch.sum(fp * self.weight)
                fn = torch.sum(fn * self.weight)
                tn = torch.sum(tn * self.weight)
            else:
                # normal average
                tp = torch.mean(tp)
                fp = torch.mean(fp)
                fn = torch.mean(fn)
                tn = torch.mean(tn)
        else:
            tp = torch.mean(tp, dim=0)
            fp = torch.mean(fp, dim=0)
            fn = torch.mean(fn, dim=0)
            tn = torch.mean(tn, dim=0)

        return tp, fp, fn, tn


class IntersectionOverUnionS2(BaseMetricS2):
    """
    Intersection over Union (IoU) metric for spherical data.
    
    Computes the IoU score for multiclass classification on the sphere,
    properly weighted by quadrature weights to account for spherical geometry.
    
    Parameters
    -----------
    nlat : int
        Number of latitude points
    nlon : int
        Number of longitude points
    grid : str, optional
        Grid type ("equiangular", "legendre-gauss", etc.), by default "equiangular"
    weight : torch.Tensor, optional
        Class weights for weighted averaging, by default None
    ignore_index : int, optional
        Index to ignore in computations, by default -100
    mode : str, optional
        Averaging mode ("micro" or "macro"), by default "micro"
    """
    
    def __init__(self, nlat: int, nlon: int, grid: str = "equiangular", weight: torch.Tensor = None, ignore_index: int = -100, mode: str = "micro"):
        super().__init__(nlat, nlon, grid, weight, ignore_index, mode)

    def forward(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:

        tp, fp, fn, tn = self._forward(pred, truth)

        # compute score
        score = tp / (tp + fp + fn)

        if self.mode == "macro":
            # we need to do some averaging still:
            # be careful with zeros
            score = torch.where(torch.isnan(score), 0.0, score)

            if self.weight is not None:
                score = torch.sum(score * self.weight)
            else:
                score = torch.mean(score)

        return score


class AccuracyS2(BaseMetricS2):
    """
    Accuracy metric for spherical data.
    
    Computes the accuracy score for multiclass classification on the sphere,
    properly weighted by quadrature weights to account for spherical geometry.
    
    Parameters
    -----------
    nlat : int
        Number of latitude points
    nlon : int
        Number of longitude points
    grid : str, optional
        Grid type ("equiangular", "legendre-gauss", etc.), by default "equiangular"
    weight : torch.Tensor, optional
        Class weights for weighted averaging, by default None
    ignore_index : int, optional
        Index to ignore in computations, by default -100
    mode : str, optional
        Averaging mode ("micro" or "macro"), by default "micro"
    """
    
    def __init__(self, nlat: int, nlon: int, grid: str = "equiangular", weight: torch.Tensor = None, ignore_index: int = -100, mode: str = "micro"):
        super().__init__(nlat, nlon, grid, weight, ignore_index, mode)

    def forward(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:

        tp, fp, fn, tn = self._forward(pred, truth)

        # compute score
        score = (tp + tn) / (tp + fp + fn + tn)

        if self.mode == "macro":
            # we need to do some averaging still:
            # be careful with zeros
            score = torch.where(torch.isnan(score), 0.0, score)

            if self.weight is not None:
                score = torch.sum(score * self.weight)
            else:
                score = torch.mean(score)

        return score
