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

from torch_harmonics.examples.losses import (
    CrossEntropyLossS2,
    DiceLossS2,
    FocalLossS2,
    L1LossS2,
    L2LossS2,
    NormalLossS2,
    SquaredL2LossS2,
    W11LossS2,
)


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


def area_weighted_mean(quad_weights, values):
    """Integrate a per-pixel quantity against the normalized quadrature weights."""
    return (values * quad_weights).sum(dim=(-2, -1))


def cross_entropy_reference(loss, logits, target, class_weights=None):
    """Per-pixel cross entropy built from log-probabilities directly.

    Ignored pixels contribute nothing, which is what ``ignore_index`` means; the
    integral is then taken over the whole sphere, so masking lowers the result.
    """
    log_probabilities = logits.log_softmax(dim=1)
    gathered = torch.zeros_like(target, dtype=logits.dtype)
    for batch in range(target.shape[0]):
        for i in range(target.shape[1]):
            for j in range(target.shape[2]):
                label = target[batch, i, j].item()
                if label == loss.ignore_index:
                    continue
                scale = 1.0 if class_weights is None else class_weights[label].item()
                gathered[batch, i, j] = -scale * log_probabilities[batch, label, i, j]
    return area_weighted_mean(loss.quad_weights, gathered).mean()


class TestCrossEntropyLossS2(unittest.TestCase):
    """CrossEntropyLossS2, which delegates masking to nn.functional.cross_entropy."""

    @parameterized.expand(list(itertools.product([-100, 1], [False, True])))
    def test_matches_log_probability_reference(self, ignore_index, weighted):
        generator = torch.Generator().manual_seed(17)
        logits = torch.randn((2, 3, 6, 8), dtype=torch.float64, generator=generator)
        target = torch.randint(0, 3, (2, 6, 8), generator=generator)
        target[:, ::3, ::2] = ignore_index
        class_weights = torch.tensor([0.4, 0.1, 0.5], dtype=torch.float64) if weighted else None
        loss = CrossEntropyLossS2(6, 8, weight=class_weights, ignore_index=ignore_index).double()

        actual = loss(logits, target)
        expected = cross_entropy_reference(loss, logits, target, class_weights)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_ignored_prediction_values_do_not_affect_loss(self):
        generator = torch.Generator().manual_seed(18)
        logits = torch.randn((1, 3, 6, 8), dtype=torch.float64, generator=generator)
        target = torch.randint(0, 3, (1, 6, 8), generator=generator)
        target[:, 2:4, :] = -100
        changed = logits.clone()
        changed[:, :, 2:4, :] = 50 * torch.randn((1, 3, 2, 8), dtype=torch.float64, generator=generator)
        loss = CrossEntropyLossS2(6, 8).double()
        torch.testing.assert_close(loss(logits, target), loss(changed, target), rtol=0, atol=0)

    def test_gradients_vanish_on_ignored_pixels(self):
        generator = torch.Generator().manual_seed(19)
        logits = torch.randn((1, 3, 6, 8), dtype=torch.float64, generator=generator, requires_grad=True)
        target = torch.randint(0, 3, (1, 6, 8), generator=generator)
        target[:, 1:3, :] = -100
        loss = CrossEntropyLossS2(6, 8).double()
        gradient = torch.autograd.grad(loss(logits, target), logits)[0]
        self.assertEqual(torch.count_nonzero(gradient[:, :, 1:3, :]).item(), 0)

    @parameterized.expand([("equiangular",), ("legendre-gauss",), ("lobatto",)])
    def test_confident_correct_prediction_approaches_zero(self, grid):
        target = torch.randint(0, 2, (1, 6, 8))
        logits = torch.where(
            torch.stack([target == 0, target == 1], dim=1),
            torch.tensor(60.0, dtype=torch.float64),
            torch.tensor(-60.0, dtype=torch.float64),
        )
        loss = CrossEntropyLossS2(6, 8, grid=grid).double()
        torch.testing.assert_close(loss(logits, target), torch.tensor(0.0, dtype=torch.float64), rtol=0, atol=1e-12)

    def test_ignored_area_lowers_the_integral(self):
        """Characterizes the current normalization: the integral runs over the whole
        sphere, so an ignored region lowers the loss rather than being divided out.

        This differs from ``SphericalLossBase._integrate_sphere``, which divides by the
        valid area, and from ``reduction="mean"`` in torch, which divides by the valid
        count. Encoded here so that changing the convention is a deliberate decision.
        """
        generator = torch.Generator().manual_seed(20)
        logits = torch.randn((1, 2, 6, 8), dtype=torch.float64, generator=generator)
        target = torch.randint(0, 2, (1, 6, 8), generator=generator)
        loss = CrossEntropyLossS2(6, 8).double()

        unmasked = loss(logits, target)
        masked_target = target.clone()
        masked_target[:, 3:, :] = -100
        masked = loss(logits, masked_target)

        kept = area_weighted_mean(loss.quad_weights, (masked_target != -100).to(torch.float64)).item()
        self.assertLess(masked.item(), unmasked.item())
        self.assertLess(kept, 1.0)


class TestFocalLossS2(unittest.TestCase):
    """FocalLossS2, which modulates the same per-pixel cross entropy."""

    @parameterized.expand([(0.25, 2.0), (1.0, 1.0), (0.5, 3.0)])
    def test_matches_focal_formula(self, alpha, gamma):
        generator = torch.Generator().manual_seed(21)
        logits = torch.randn((2, 3, 6, 8), dtype=torch.float64, generator=generator)
        target = torch.randint(0, 3, (2, 6, 8), generator=generator)
        target[:, ::3, ::2] = -100
        loss = FocalLossS2(6, 8).double()

        entropy = torch.nn.functional.cross_entropy(logits, target, reduction="none", ignore_index=-100)
        expected = area_weighted_mean(loss.quad_weights, alpha * (1 - torch.exp(-entropy)) ** gamma * entropy).mean()
        torch.testing.assert_close(loss(logits, target, alpha=alpha, gamma=gamma), expected, rtol=1e-12, atol=1e-12)

    def test_zero_gamma_reduces_to_scaled_cross_entropy(self):
        """(1 - p)^0 == 1, so the modulation disappears and only alpha remains."""
        generator = torch.Generator().manual_seed(22)
        logits = torch.randn((1, 3, 6, 8), dtype=torch.float64, generator=generator)
        target = torch.randint(0, 3, (1, 6, 8), generator=generator)
        target[:, 2:4, :] = -100
        alpha = 0.25

        focal = FocalLossS2(6, 8).double()
        entropy = CrossEntropyLossS2(6, 8).double()
        torch.testing.assert_close(focal(logits, target, alpha=alpha, gamma=0.0), alpha * entropy(logits, target), rtol=1e-12, atol=1e-12)

    def test_ignored_prediction_values_do_not_affect_loss(self):
        generator = torch.Generator().manual_seed(23)
        logits = torch.randn((1, 3, 6, 8), dtype=torch.float64, generator=generator)
        target = torch.randint(0, 3, (1, 6, 8), generator=generator)
        target[:, 2:4, :] = -100
        changed = logits.clone()
        changed[:, :, 2:4, :] = 50 * torch.randn((1, 3, 2, 8), dtype=torch.float64, generator=generator)
        loss = FocalLossS2(6, 8).double()
        torch.testing.assert_close(loss(logits, target), loss(changed, target), rtol=0, atol=0)

    def test_gradients_vanish_on_ignored_pixels(self):
        generator = torch.Generator().manual_seed(24)
        logits = torch.randn((1, 3, 6, 8), dtype=torch.float64, generator=generator, requires_grad=True)
        target = torch.randint(0, 3, (1, 6, 8), generator=generator)
        target[:, 1:3, :] = -100
        loss = FocalLossS2(6, 8).double()
        gradient = torch.autograd.grad(loss(logits, target), logits)[0]
        self.assertEqual(torch.count_nonzero(gradient[:, :, 1:3, :]).item(), 0)


class TestSphericalRegressionLosses(unittest.TestCase):
    """The SphericalLossBase family, which integrates a pointwise term over the sphere."""

    @parameterized.expand([(SquaredL2LossS2,), (L1LossS2,), (L2LossS2,), (W11LossS2,), (NormalLossS2,)])
    def test_identical_inputs_give_zero_loss(self, loss_type):
        generator = torch.Generator().manual_seed(25)
        field = torch.randn((1, 1, 8, 16), generator=generator)
        loss = loss_type(8, 16)
        torch.testing.assert_close(loss(field, field.clone()), torch.tensor(0.0), rtol=0, atol=1e-6)

    @parameterized.expand([("equiangular",), ("legendre-gauss",), ("lobatto",)])
    def test_squared_l2_matches_area_weighted_reference(self, grid):
        generator = torch.Generator().manual_seed(26)
        prd = torch.randn((2, 1, 8, 16), dtype=torch.float64, generator=generator)
        tar = torch.randn((2, 1, 8, 16), dtype=torch.float64, generator=generator)
        loss = SquaredL2LossS2(8, 16, grid=grid).double()
        expected = area_weighted_mean(loss.quad_weights, torch.square(prd - tar)).mean()
        torch.testing.assert_close(loss(prd, tar), expected, rtol=1e-12, atol=1e-12)

    def test_l1_matches_area_weighted_reference(self):
        generator = torch.Generator().manual_seed(27)
        prd = torch.randn((2, 1, 8, 16), dtype=torch.float64, generator=generator)
        tar = torch.randn((2, 1, 8, 16), dtype=torch.float64, generator=generator)
        loss = L1LossS2(8, 16).double()
        expected = area_weighted_mean(loss.quad_weights, torch.abs(prd - tar)).mean()
        torch.testing.assert_close(loss(prd, tar), expected, rtol=1e-12, atol=1e-12)

    def test_l2_is_the_root_of_the_squared_loss(self):
        generator = torch.Generator().manual_seed(28)
        prd = torch.randn((1, 1, 8, 16), dtype=torch.float64, generator=generator)
        tar = torch.randn((1, 1, 8, 16), dtype=torch.float64, generator=generator)
        squared = SquaredL2LossS2(8, 16).double()(prd, tar)
        torch.testing.assert_close(L2LossS2(8, 16).double()(prd, tar), torch.sqrt(squared), rtol=1e-12, atol=1e-12)

    @parameterized.expand([(SquaredL2LossS2,), (L1LossS2,)])
    def test_mask_normalizes_by_the_valid_area(self, loss_type):
        """A constant error over the valid region integrates to that error whatever the
        mask covers, because _integrate_sphere divides by the valid area."""
        loss = loss_type(8, 16).double()
        tar = torch.zeros((1, 1, 8, 16), dtype=torch.float64)
        prd = torch.full((1, 1, 8, 16), 3.0, dtype=torch.float64)

        for rows in (2, 4, 6):
            mask = torch.zeros((1, 1, 8, 16), dtype=torch.float64)
            mask[..., :rows, :] = 1.0
            prd_noisy = prd.clone()
            prd_noisy[..., rows:, :] = 100.0  # ignored region must not leak in
            expected = torch.tensor(9.0 if loss_type is SquaredL2LossS2 else 3.0, dtype=torch.float64)
            torch.testing.assert_close(loss(prd_noisy, tar, mask=mask), expected, rtol=1e-12, atol=1e-12)
