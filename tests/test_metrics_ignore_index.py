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

import unittest

import torch
from parameterized import parameterized
from testutils import compare_tensors

from torch_harmonics.examples.metrics import AccuracyS2, IntersectionOverUnionS2, _get_stats_multiclass


def _reference_counts(output, target, num_classes, weights, ignore_index):
    counts = [torch.zeros(output.shape[0], num_classes) for _ in range(4)]
    for batch in range(output.shape[0]):
        valid = torch.ones_like(target[batch], dtype=torch.bool) if ignore_index is None else target[batch] != ignore_index
        for label in range(num_classes):
            predicted = output[batch] == label
            actual = target[batch] == label
            for result, mask in zip(counts, (predicted & actual, predicted & ~actual, ~predicted & actual, ~predicted & ~actual)):
                result[batch, label] = weights[valid & mask].sum()
    return counts


class TestIgnoredSphericalMetrics(unittest.TestCase):
    @parameterized.expand([[-100], [3], [None]])
    def test_counts_match_weighted_confusion_matrix(self, ignore_index, verbose=False):
        target = (torch.arange(24).reshape(2, 3, 4) % 3).long()
        output = (target + 1) % 3
        output[:, 0, ::2] = target[:, 0, ::2]
        if ignore_index is not None:
            target[0, 0, :] = ignore_index
            target[1, 1:, 1:] = ignore_index
        weights = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4) / 78
        expected = _reference_counts(output, target, 3, weights, ignore_index)
        actual = _get_stats_multiclass(output, target, 3, weights, ignore_index)
        for label, wanted, got in zip(("tp", "fp", "fn", "tn"), expected, actual):
            self.assertTrue(compare_tensors(label, wanted, got, atol=1e-6, rtol=1e-5, verbose=verbose))

    @parameterized.expand([(grid, mode, weighted) for grid in ("equiangular", "legendre-gauss") for mode in ("micro", "macro") for weighted in (False, True)])
    def test_ignored_area_does_not_reward_wrong_predictions(self, grid, mode, weighted, verbose=False):
        target = (torch.arange(4)[:, None] % 2).expand(4, 4)[None].clone()
        predictions = 1 - target
        target[..., 1:] = -100
        logits = 10 * torch.nn.functional.one_hot(predictions, num_classes=2).permute(0, 3, 1, 2).float()
        weight = torch.tensor([0.25, 0.75]) if weighted else None
        accuracy = AccuracyS2(4, 4, grid=grid, mode=mode, weight=weight)
        iou = IntersectionOverUnionS2(4, 4, grid=grid, mode=mode, weight=weight)
        # All valid binary predictions are wrong. Masking 3/4 of the sphere
        # must not turn that into a 75% accuracy score.
        self.assertTrue(compare_tensors("accuracy", torch.tensor(0.0), accuracy(logits, target), atol=1e-6, rtol=0, verbose=verbose))
        self.assertTrue(compare_tensors("IoU", torch.tensor(0.0), iou(logits, target), atol=1e-6, rtol=0, verbose=verbose))

    def test_fully_ignored_sample_has_no_true_negatives(self, verbose=False):
        output = torch.zeros((1, 3, 4), dtype=torch.long)
        target = torch.full_like(output, -100)
        weights = torch.ones((3, 4)) / 12
        for result in _get_stats_multiclass(output, target, 2, weights, -100):
            self.assertTrue(compare_tensors("empty confusion matrix", torch.zeros_like(result), result, atol=0, rtol=0, verbose=verbose))

    def test_no_ignored_labels_preserves_counts(self, verbose=False):
        target = (torch.arange(12).reshape(1, 3, 4) % 3).long()
        output = torch.flip(target, dims=(-1,))
        weights = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4) / 78
        without_ignore = _get_stats_multiclass(output, target, 3, weights, None)
        unused_ignore = _get_stats_multiclass(output, target, 3, weights, -100)
        for wanted, got in zip(without_ignore, unused_ignore):
            self.assertTrue(compare_tensors("unmasked counts", wanted, got, atol=0, rtol=0, verbose=verbose))


if __name__ == "__main__":
    unittest.main()
