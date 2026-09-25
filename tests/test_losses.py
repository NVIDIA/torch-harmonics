# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
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

import itertools
import unittest

import torch
from parameterized import parameterized

from torch_harmonics.examples.losses import DiceLossS2


def valid_pixel_reference(loss, logits, target):
    """Accumulate each class using valid pixels only, without one-hot replacement."""
    probabilities = logits.softmax(dim=1)
    batch_scores = []
    area = loss.quad_weights.expand_as(target[0])
    for batch in range(target.shape[0]):
        valid = torch.ones_like(target[batch], dtype=torch.bool) if loss.ignore_index is None else target[batch] != loss.ignore_index
        labels = target[batch][valid]
        weights = area[valid]
        intersections = []
        unions = []
        for channel in range(logits.shape[1]):
            prediction = probabilities[batch, channel][valid]
            truth = (labels == channel).to(prediction.dtype)
            intersections.append((weights * prediction * truth).sum())
            unions.append((weights * (prediction + truth)).sum())
        intersection = torch.stack(intersections)
        union = torch.stack(unions)
        if loss.mode == "micro":
            if loss.weight is None:
                intersection, union = intersection.mean(), union.mean()
            else:
                intersection = (intersection * loss.weight[0]).sum()
                union = (union * loss.weight[0]).sum()
        score = (2 * intersection + loss.smooth) / (union + loss.smooth)
        if loss.mode == "macro":
            score = score.mean() if loss.weight is None else (score * loss.weight[0]).sum()
        batch_scores.append(score)
    return 1 - torch.stack(batch_scores).mean()


class TestDiceLossIgnoredPixels(unittest.TestCase):
    @parameterized.expand(list(itertools.product(["micro", "macro"], [-100, 1], [False, True], [0.0, 0.1])))
    def test_matches_valid_pixel_loss_and_gradient(self, mode, ignore_index, weighted, smooth):
        generator = torch.Generator().manual_seed(812)
        logits = torch.randn((2, 3, 6, 8), dtype=torch.float64, generator=generator, requires_grad=True)
        target = torch.randint(0, 3, (2, 6, 8), generator=generator)
        target[:, ::2, ::3] = ignore_index
        class_weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64) if weighted else None
        loss = DiceLossS2(6, 8, weight=class_weights, smooth=smooth, ignore_index=ignore_index, mode=mode).double()
        before = target.clone()
        actual = loss(logits, target)
        expected = valid_pixel_reference(loss, logits, target)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        actual_gradient = torch.autograd.grad(actual, logits, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, logits)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-11, atol=1e-12)
        ignored = (target == ignore_index).unsqueeze(1).expand_as(logits)
        self.assertEqual(torch.count_nonzero(actual_gradient[ignored]).item(), 0)
        self.assertTrue(torch.equal(target, before))

    @parameterized.expand([("equiangular",), ("legendre-gauss",), ("lobatto",)])
    def test_perfect_valid_predictions_have_zero_loss(self, grid):
        logits = torch.empty((1, 2, 6, 8), dtype=torch.float64)
        logits[:, 0] = 100
        logits[:, 1] = -100
        target = torch.zeros((1, 6, 8), dtype=torch.long)
        target[:, 1:3, :] = -100
        loss = DiceLossS2(6, 8, grid=grid).double()
        torch.testing.assert_close(loss(logits, target), torch.tensor(0.0, dtype=torch.float64), rtol=0, atol=1e-12)

    @parameterized.expand(list(itertools.product(["micro", "macro"], [None, -100])))
    def test_unmasked_inputs_retain_existing_result(self, mode, ignore_index):
        generator = torch.Generator().manual_seed(46)
        logits = torch.randn((1, 3, 6, 8), dtype=torch.float64, generator=generator)
        target = torch.randint(0, 3, (1, 6, 8), generator=generator)
        loss = DiceLossS2(6, 8, ignore_index=ignore_index, mode=mode, smooth=0.1).double()
        torch.testing.assert_close(loss(logits, target), valid_pixel_reference(loss, logits, target), rtol=1e-12, atol=1e-12)

    def test_ignored_prediction_values_do_not_affect_loss(self):
        generator = torch.Generator().manual_seed(91)
        logits = torch.randn((1, 3, 6, 8), dtype=torch.float64, generator=generator)
        target = torch.randint(0, 3, (1, 6, 8), generator=generator)
        target[:, 2:4, :] = -100
        changed = logits.clone()
        changed[:, :, 2:4, :] = 100 * torch.randn((1, 3, 2, 8), dtype=torch.float64, generator=generator)
        loss = DiceLossS2(6, 8, smooth=0.1).double()
        torch.testing.assert_close(loss(logits, target), loss(changed, target), rtol=0, atol=0)
