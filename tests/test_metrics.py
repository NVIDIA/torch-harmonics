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

from torch_harmonics.examples.metrics import AccuracyS2, IntersectionOverUnionS2, _get_stats_multiclass


def confusion_reference(output, target, num_classes, quad_weights, ignore_index):
    """Weighted confusion matrix accumulated directly, in double precision.

    Ignored pixels carry no area, so they contribute to none of the four counts. This
    is the definition the metrics are checked against.
    """
    batch_size = output.shape[0]
    counts = [torch.zeros(batch_size, num_classes, dtype=torch.float64) for _ in range(4)]
    true_positive, false_positive, false_negative, true_negative = counts
    area = quad_weights.to(torch.float64).flatten()

    for batch in range(batch_size):
        labels, predictions = target[batch].flatten(), output[batch].flatten()
        weights = area if ignore_index is None else torch.where(labels == ignore_index, 0.0, area)
        for channel in range(num_classes):
            is_label, is_prediction = labels == channel, predictions == channel
            true_positive[batch, channel] = torch.sum(weights * (is_label & is_prediction))
            false_positive[batch, channel] = torch.sum(weights * (~is_label & is_prediction))
            false_negative[batch, channel] = torch.sum(weights * (is_label & ~is_prediction))
            true_negative[batch, channel] = torch.sum(weights * (~is_label & ~is_prediction))

    return true_positive, false_positive, false_negative, true_negative


def confident_logits(target, num_classes):
    """Logits that predict the given labels with certainty."""
    channels = [torch.where(target == c, 60.0, -60.0) for c in range(num_classes)]
    return torch.stack(channels, dim=1).to(torch.float64)


class TestMulticlassStats(unittest.TestCase):
    """The weighted confusion matrix underlying both metrics."""

    @parameterized.expand(list(itertools.product([None, -100, 0, 2], ["equiangular", "legendre-gauss"])))
    def test_matches_confusion_matrix(self, ignore_index, grid):
        generator = torch.Generator().manual_seed(31)
        target = torch.randint(0, 4, (3, 8, 16), generator=generator)
        output = torch.randint(0, 4, (3, 8, 16), generator=generator)
        if ignore_index is not None:
            target[0, :, ::3] = ignore_index
            target[-1] = ignore_index  # a fully ignored sample

        metric = AccuracyS2(8, 16, grid=grid, ignore_index=ignore_index)
        actual = _get_stats_multiclass(output, target, 4, metric.quad_weights, ignore_index)
        expected = confusion_reference(output, target, 4, metric.quad_weights, ignore_index)

        for count, reference, name in zip(actual, expected, ["tp", "fp", "fn", "tn"]):
            torch.testing.assert_close(count.double(), reference, rtol=1e-6, atol=1e-7, msg=f"{name} mismatch")

    def test_counts_sum_to_the_scored_area(self):
        """Every class partitions the scored area into the four counts."""
        generator = torch.Generator().manual_seed(32)
        target = torch.randint(0, 3, (2, 8, 16), generator=generator)
        output = torch.randint(0, 3, (2, 8, 16), generator=generator)
        target[:, 5:, :] = -100

        metric = AccuracyS2(8, 16)
        counts = _get_stats_multiclass(output, target, 3, metric.quad_weights, -100)
        total = sum(counts)
        scored = torch.where(target == -100, 0.0, metric.quad_weights.expand_as(target.double()).double())
        expected = scored.flatten(1).sum(1, keepdim=True).expand_as(total)
        torch.testing.assert_close(total.double(), expected, rtol=1e-6, atol=1e-7)


class TestAccuracyS2(unittest.TestCase):

    @parameterized.expand(list(itertools.product(["micro", "macro"], ["equiangular", "legendre-gauss", "lobatto"])))
    def test_perfect_prediction_scores_one(self, mode, grid):
        generator = torch.Generator().manual_seed(33)
        target = torch.randint(0, 3, (2, 8, 16), generator=generator)
        target[:, 6:, :] = -100
        metric = AccuracyS2(8, 16, grid=grid, mode=mode)
        score = metric(confident_logits(target.clamp(min=0), 3), target)
        torch.testing.assert_close(score.double(), torch.tensor(1.0, dtype=torch.float64), rtol=0, atol=1e-6)

    def test_ignored_area_is_not_credited_as_correct(self):
        """A prediction wrong on every scored pixel scores zero, whatever is masked out.

        Before the true-negative fix the ignored area was counted as correctly
        classified, so this returned the ignored fraction instead.
        """
        target = torch.full((1, 8, 16), -100, dtype=torch.long)
        target[..., :4] = 0  # only a quarter of the sphere is scored
        logits = confident_logits(torch.ones_like(target), 2)  # always predicts class 1
        metric = AccuracyS2(8, 16, mode="micro")
        torch.testing.assert_close(metric(logits, target).double(), torch.tensor(0.0, dtype=torch.float64), rtol=0, atol=1e-6)

    def test_ignored_predictions_do_not_change_the_score(self):
        generator = torch.Generator().manual_seed(34)
        target = torch.randint(0, 3, (1, 8, 16), generator=generator)
        target[:, 3:5, :] = -100
        logits = torch.randn((1, 3, 8, 16), dtype=torch.float64, generator=generator)
        changed = logits.clone()
        changed[:, :, 3:5, :] = 50 * torch.randn((1, 3, 2, 16), dtype=torch.float64, generator=generator)
        metric = AccuracyS2(8, 16)
        torch.testing.assert_close(metric(logits, target), metric(changed, target), rtol=0, atol=0)

    def test_unmasked_score_is_the_area_weighted_correct_fraction(self):
        generator = torch.Generator().manual_seed(35)
        target = torch.randint(0, 2, (1, 8, 16), generator=generator)
        output = torch.randint(0, 2, (1, 8, 16), generator=generator)
        metric = AccuracyS2(8, 16, mode="micro")
        expected = (metric.quad_weights * (target == output)).sum().double()
        torch.testing.assert_close(metric(confident_logits(output, 2), target).double(), expected, rtol=1e-6, atol=1e-7)


class TestIntersectionOverUnionS2(unittest.TestCase):

    @parameterized.expand(["micro", "macro"])
    def test_perfect_prediction_scores_one(self, mode):
        generator = torch.Generator().manual_seed(36)
        target = torch.randint(0, 3, (2, 8, 16), generator=generator)
        target[:, 6:, :] = -100
        metric = IntersectionOverUnionS2(8, 16, mode=mode)
        score = metric(confident_logits(target.clamp(min=0), 3), target)
        torch.testing.assert_close(score.double(), torch.tensor(1.0, dtype=torch.float64), rtol=0, atol=1e-6)

    def test_matches_the_iou_definition(self):
        """IoU has no true-negative term, so it reads straight off the reference."""
        generator = torch.Generator().manual_seed(37)
        target = torch.randint(0, 2, (1, 8, 16), generator=generator)
        output = torch.randint(0, 2, (1, 8, 16), generator=generator)
        target[:, 5:, :] = -100

        metric = IntersectionOverUnionS2(8, 16, mode="micro")
        tp, fp, fn, _ = confusion_reference(output, target, 2, metric.quad_weights, -100)
        expected = (tp.mean(dim=1) / (tp + fp + fn).mean(dim=1)).mean()
        torch.testing.assert_close(metric(confident_logits(output, 2), target).double(), expected, rtol=1e-6, atol=1e-7)
