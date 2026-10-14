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
from testutils import compare_tensors

from torch_harmonics import as_grid
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
        loss = DiceLossS2(as_grid("equiangular", nlat=6, nlon=8), weight=class_weights, smooth=smooth, ignore_index=ignore_index, mode=mode).double()
        before = target.clone()
        actual = loss(logits, target)
        expected = valid_pixel_reference(loss, logits, target)
        self.assertTrue(compare_tensors("matches valid pixel loss and gradient", actual, expected, atol=1e-12, rtol=1e-12))
        actual_gradient = torch.autograd.grad(actual, logits, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, logits)[0]
        self.assertTrue(compare_tensors("matches valid pixel loss and gradient", actual_gradient, expected_gradient, atol=1e-12, rtol=1e-11))
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
        loss = DiceLossS2(as_grid(grid, nlat=6, nlon=8)).double()
        self.assertTrue(compare_tensors("perfect valid predictions have zero loss", loss(logits, target), torch.tensor(0.0, dtype=torch.float64), atol=1e-12, rtol=0))

    @parameterized.expand(list(itertools.product(["micro", "macro"], [None, -100])))
    def test_unmasked_inputs_retain_existing_result(self, mode, ignore_index):
        generator = torch.Generator().manual_seed(46)
        logits = torch.randn((1, 3, 6, 8), dtype=torch.float64, generator=generator)
        target = torch.randint(0, 3, (1, 6, 8), generator=generator)
        loss = DiceLossS2(as_grid("equiangular", nlat=6, nlon=8), ignore_index=ignore_index, mode=mode, smooth=0.1).double()
        self.assertTrue(compare_tensors("unmasked inputs retain existing result", loss(logits, target), valid_pixel_reference(loss, logits, target), atol=1e-12, rtol=1e-12))

    def test_ignored_prediction_values_do_not_affect_loss(self):
        generator = torch.Generator().manual_seed(91)
        logits = torch.randn((1, 3, 6, 8), dtype=torch.float64, generator=generator)
        target = torch.randint(0, 3, (1, 6, 8), generator=generator)
        target[:, 2:4, :] = -100
        changed = logits.clone()
        changed[:, :, 2:4, :] = 100 * torch.randn((1, 3, 2, 8), dtype=torch.float64, generator=generator)
        loss = DiceLossS2(as_grid("equiangular", nlat=6, nlon=8), smooth=0.1).double()
        self.assertTrue(compare_tensors("ignored prediction values do not affect loss", loss(logits, target), loss(changed, target), atol=0, rtol=0))


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
        loss = CrossEntropyLossS2(as_grid("equiangular", nlat=6, nlon=8), weight=class_weights, ignore_index=ignore_index).double()

        actual = loss(logits, target)
        expected = cross_entropy_reference(loss, logits, target, class_weights)
        self.assertTrue(compare_tensors("matches log probability reference", actual, expected, atol=1e-12, rtol=1e-12))

    def test_ignored_prediction_values_do_not_affect_loss(self):
        generator = torch.Generator().manual_seed(18)
        logits = torch.randn((1, 3, 6, 8), dtype=torch.float64, generator=generator)
        target = torch.randint(0, 3, (1, 6, 8), generator=generator)
        target[:, 2:4, :] = -100
        changed = logits.clone()
        changed[:, :, 2:4, :] = 50 * torch.randn((1, 3, 2, 8), dtype=torch.float64, generator=generator)
        loss = CrossEntropyLossS2(as_grid("equiangular", nlat=6, nlon=8)).double()
        self.assertTrue(compare_tensors("ignored prediction values do not affect loss", loss(logits, target), loss(changed, target), atol=0, rtol=0))

    def test_gradients_vanish_on_ignored_pixels(self):
        generator = torch.Generator().manual_seed(19)
        logits = torch.randn((1, 3, 6, 8), dtype=torch.float64, generator=generator, requires_grad=True)
        target = torch.randint(0, 3, (1, 6, 8), generator=generator)
        target[:, 1:3, :] = -100
        loss = CrossEntropyLossS2(as_grid("equiangular", nlat=6, nlon=8)).double()
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
        loss = CrossEntropyLossS2(as_grid(grid, nlat=6, nlon=8)).double()
        self.assertTrue(compare_tensors("confident correct prediction approaches zero", loss(logits, target), torch.tensor(0.0, dtype=torch.float64), atol=1e-12, rtol=0))

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
        loss = CrossEntropyLossS2(as_grid("equiangular", nlat=6, nlon=8)).double()

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
        loss = FocalLossS2(as_grid("equiangular", nlat=6, nlon=8)).double()

        entropy = torch.nn.functional.cross_entropy(logits, target, reduction="none", ignore_index=-100)
        expected = area_weighted_mean(loss.quad_weights, alpha * (1 - torch.exp(-entropy)) ** gamma * entropy).mean()
        self.assertTrue(compare_tensors("matches focal formula", loss(logits, target, alpha=alpha, gamma=gamma), expected, atol=1e-12, rtol=1e-12))

    def test_zero_gamma_reduces_to_scaled_cross_entropy(self):
        """(1 - p)^0 == 1, so the modulation disappears and only alpha remains."""
        generator = torch.Generator().manual_seed(22)
        logits = torch.randn((1, 3, 6, 8), dtype=torch.float64, generator=generator)
        target = torch.randint(0, 3, (1, 6, 8), generator=generator)
        target[:, 2:4, :] = -100
        alpha = 0.25

        focal = FocalLossS2(as_grid("equiangular", nlat=6, nlon=8)).double()
        entropy = CrossEntropyLossS2(as_grid("equiangular", nlat=6, nlon=8)).double()
        self.assertTrue(
            compare_tensors("zero gamma reduces to scaled cross entropy", focal(logits, target, alpha=alpha, gamma=0.0), alpha * entropy(logits, target), atol=1e-12, rtol=1e-12)
        )

    def test_ignored_prediction_values_do_not_affect_loss(self):
        generator = torch.Generator().manual_seed(23)
        logits = torch.randn((1, 3, 6, 8), dtype=torch.float64, generator=generator)
        target = torch.randint(0, 3, (1, 6, 8), generator=generator)
        target[:, 2:4, :] = -100
        changed = logits.clone()
        changed[:, :, 2:4, :] = 50 * torch.randn((1, 3, 2, 8), dtype=torch.float64, generator=generator)
        loss = FocalLossS2(as_grid("equiangular", nlat=6, nlon=8)).double()
        self.assertTrue(compare_tensors("ignored prediction values do not affect loss", loss(logits, target), loss(changed, target), atol=0, rtol=0))

    def test_gradients_vanish_on_ignored_pixels(self):
        generator = torch.Generator().manual_seed(24)
        logits = torch.randn((1, 3, 6, 8), dtype=torch.float64, generator=generator, requires_grad=True)
        target = torch.randint(0, 3, (1, 6, 8), generator=generator)
        target[:, 1:3, :] = -100
        loss = FocalLossS2(as_grid("equiangular", nlat=6, nlon=8)).double()
        gradient = torch.autograd.grad(loss(logits, target), logits)[0]
        self.assertEqual(torch.count_nonzero(gradient[:, :, 1:3, :]).item(), 0)


class TestSphericalRegressionLosses(unittest.TestCase):
    """The SphericalLossBase family, which integrates a pointwise term over the sphere."""

    @parameterized.expand([(SquaredL2LossS2,), (L1LossS2,), (L2LossS2,), (W11LossS2,), (NormalLossS2,)])
    def test_identical_inputs_give_zero_loss(self, loss_type):
        generator = torch.Generator().manual_seed(25)
        field = torch.randn((1, 1, 8, 16), generator=generator)
        loss = loss_type(as_grid("equiangular", nlat=8, nlon=16))
        self.assertTrue(compare_tensors("identical inputs give zero loss", loss(field, field.clone()), torch.tensor(0.0), atol=1e-6, rtol=0))

    @parameterized.expand([("equiangular",), ("legendre-gauss",), ("lobatto",)])
    def test_squared_l2_matches_area_weighted_reference(self, grid):
        generator = torch.Generator().manual_seed(26)
        prd = torch.randn((2, 1, 8, 16), dtype=torch.float64, generator=generator)
        tar = torch.randn((2, 1, 8, 16), dtype=torch.float64, generator=generator)
        loss = SquaredL2LossS2(as_grid(grid, nlat=8, nlon=16)).double()
        expected = area_weighted_mean(loss.quad_weights, torch.square(prd - tar)).mean()
        self.assertTrue(compare_tensors("squared l2 matches area weighted reference", loss(prd, tar), expected, atol=1e-12, rtol=1e-12))

    def test_l1_matches_area_weighted_reference(self):
        generator = torch.Generator().manual_seed(27)
        prd = torch.randn((2, 1, 8, 16), dtype=torch.float64, generator=generator)
        tar = torch.randn((2, 1, 8, 16), dtype=torch.float64, generator=generator)
        loss = L1LossS2(as_grid("equiangular", nlat=8, nlon=16)).double()
        expected = area_weighted_mean(loss.quad_weights, torch.abs(prd - tar)).mean()
        self.assertTrue(compare_tensors("l1 matches area weighted reference", loss(prd, tar), expected, atol=1e-12, rtol=1e-12))

    def test_l2_is_the_root_of_the_squared_loss(self):
        generator = torch.Generator().manual_seed(28)
        prd = torch.randn((1, 1, 8, 16), dtype=torch.float64, generator=generator)
        tar = torch.randn((1, 1, 8, 16), dtype=torch.float64, generator=generator)
        squared = SquaredL2LossS2(as_grid("equiangular", nlat=8, nlon=16)).double()(prd, tar)
        self.assertTrue(
            compare_tensors("l2 is the root of the squared loss", L2LossS2(as_grid("equiangular", nlat=8, nlon=16)).double()(prd, tar), torch.sqrt(squared), atol=1e-12, rtol=1e-12)
        )

    @parameterized.expand([(SquaredL2LossS2,), (L1LossS2,)])
    def test_mask_normalizes_by_the_valid_area(self, loss_type):
        """A constant error over the valid region integrates to that error whatever the
        mask covers, because _integrate_sphere divides by the valid area."""
        loss = loss_type(as_grid("equiangular", nlat=8, nlon=16)).double()
        tar = torch.zeros((1, 1, 8, 16), dtype=torch.float64)
        prd = torch.full((1, 1, 8, 16), 3.0, dtype=torch.float64)

        for rows in (2, 4, 6):
            mask = torch.zeros((1, 1, 8, 16), dtype=torch.float64)
            mask[..., :rows, :] = 1.0
            prd_noisy = prd.clone()
            prd_noisy[..., rows:, :] = 100.0  # ignored region must not leak in
            expected = torch.tensor(9.0 if loss_type is SquaredL2LossS2 else 3.0, dtype=torch.float64)
            self.assertTrue(compare_tensors("mask normalizes by the valid area", loss(prd_noisy, tar, mask=mask), expected, atol=1e-12, rtol=1e-12))


def longitude_wave(nlat, nlon, order, dtype=torch.float64):
    """A field varying only in longitude, whose derivative the FFT resolves exactly.

    ``k_phi`` comes out of ``fftfreq`` as integer wavenumbers, so d/dphi of
    sin(order * phi) is order * cos(order * phi) to machine precision. That gives the
    gradient-based losses an analytic reference independent of their own FFT code.
    """
    phi = 2 * torch.pi * torch.arange(nlon, dtype=dtype) / nlon
    field = torch.sin(order * phi).expand(nlat, nlon)
    derivative = order * torch.cos(order * phi).expand(nlat, nlon)
    return field.unsqueeze(0).unsqueeze(0).contiguous(), derivative.unsqueeze(0).unsqueeze(0).contiguous()


class TestW11LossS2(unittest.TestCase):
    """W11LossS2 compares FFT-derived first derivatives."""

    @parameterized.expand([(1,), (2,), (3,)])
    def test_matches_analytic_derivative(self, order):
        field, derivative = longitude_wave(8, 16, order)
        loss = W11LossS2(as_grid("equiangular", nlat=8, nlon=16))
        # the target is flat, so the theta term drops out and only |d/dphi| survives
        expected = area_weighted_mean(loss.quad_weights.double(), derivative.abs()).mean()
        self.assertTrue(compare_tensors("matches analytic derivative", loss(field.float(), torch.zeros_like(field).float()).double(), expected, atol=1e-6, rtol=1e-5))

    def test_a_constant_offset_is_invisible(self):
        """Characterizes a real consequence of the definition: the loss term is built
        only from derivative differences, with no value term, so it is a W^{1,1}
        seminorm. Two fields differing by a DC offset score zero. Encoded so that
        adding a value term is a deliberate change rather than a silent one.
        """
        generator = torch.Generator().manual_seed(41)
        tar = torch.randn((1, 1, 8, 16), generator=generator)
        loss = W11LossS2(as_grid("equiangular", nlat=8, nlon=16))
        self.assertTrue(compare_tensors("a constant offset is invisible", loss(tar + 5.0, tar), torch.tensor(0.0), atol=1e-5, rtol=0))

    def test_scales_linearly_with_the_derivative_difference(self):
        field, _ = longitude_wave(8, 16, 2)
        loss = W11LossS2(as_grid("equiangular", nlat=8, nlon=16))
        single = loss(field.float(), torch.zeros_like(field).float())
        double = loss(2.0 * field.float(), torch.zeros_like(field).float())
        self.assertTrue(compare_tensors("scales linearly with the derivative difference", double, 2.0 * single, atol=1e-6, rtol=1e-5))


class TestNormalLossS2(unittest.TestCase):
    """NormalLossS2 builds surface normals from the same derivatives and compares directions."""

    @parameterized.expand([(1,), (2,)])
    def test_matches_analytic_cosine_distance(self, order):
        field, derivative = longitude_wave(8, 16, order)
        loss = NormalLossS2(as_grid("equiangular", nlat=8, nlon=16))
        # normals are [-d/dphi, -d/dtheta, 1] normalized; against a flat target whose
        # normal is [0, 0, 1] the cosine reduces to 1 / sqrt(1 + (d/dphi)^2)
        cosine = 1.0 / torch.sqrt(1.0 + derivative**2)
        expected = area_weighted_mean(loss.quad_weights.double(), 1.0 - cosine).mean()
        self.assertTrue(compare_tensors("matches analytic cosine distance", loss(field.float(), torch.zeros_like(field).float()).double(), expected, atol=1e-6, rtol=1e-5))

    def test_a_constant_offset_leaves_the_normals_unchanged(self):
        generator = torch.Generator().manual_seed(42)
        tar = torch.randn((1, 1, 8, 16), generator=generator)
        loss = NormalLossS2(as_grid("equiangular", nlat=8, nlon=16))
        self.assertTrue(compare_tensors("a constant offset leaves the normals unchanged", loss(tar + 5.0, tar), torch.tensor(0.0), atol=1e-6, rtol=0))


class TestSphericalLossIntegration(unittest.TestCase):
    """Normalization and gradient flow shared by the SphericalLossBase family."""

    @parameterized.expand([(SquaredL2LossS2,), (L1LossS2,), (L2LossS2,)])
    def test_unnormalized_weights_scale_by_the_sphere_area(self, loss_type):
        """normalized=False leaves the quadrature weights summing to 4*pi instead of 1,
        which scales the integral (and, for L2, its square root)."""
        generator = torch.Generator().manual_seed(43)
        prd = torch.randn((1, 1, 8, 16), dtype=torch.float64, generator=generator)
        tar = torch.randn((1, 1, 8, 16), dtype=torch.float64, generator=generator)

        normalized = loss_type(as_grid("equiangular", nlat=8, nlon=16)).double()(prd, tar)
        unnormalized = loss_type(as_grid("equiangular", nlat=8, nlon=16), normalized=False).double()(prd, tar)
        ratio = (4 * torch.pi) ** (0.5 if loss_type is L2LossS2 else 1.0)
        # the quadrature rules are float64 but the weights are cast to float32 at the
        # layer boundary, the same convention QuadratureS2 follows and documents. the
        # two weight sets therefore agree with each other only to float32 precision
        # (~5e-8), even though both losses are evaluated in float64
        self.assertTrue(compare_tensors("unnormalized weights scale by the sphere area", unnormalized, normalized * ratio, atol=1e-12, rtol=1e-6))

    @parameterized.expand([(SquaredL2LossS2,), (L1LossS2,), (L2LossS2,), (W11LossS2,), (NormalLossS2,)])
    def test_gradients_reach_the_prediction(self, loss_type):
        generator = torch.Generator().manual_seed(44)
        prd = torch.randn((1, 1, 8, 16), generator=generator, requires_grad=True)
        tar = torch.randn((1, 1, 8, 16), generator=generator)
        gradient = torch.autograd.grad(loss_type(as_grid("equiangular", nlat=8, nlon=16))(prd, tar), prd)[0]
        self.assertEqual(gradient.shape, prd.shape)
        self.assertGreater(torch.count_nonzero(gradient).item(), 0)
        self.assertTrue(torch.isfinite(gradient).all())

    @parameterized.expand([(SquaredL2LossS2,), (L1LossS2,)])
    def test_masked_integration_ignores_the_masked_region(self, loss_type):
        generator = torch.Generator().manual_seed(45)
        tar = torch.zeros((1, 1, 8, 16), dtype=torch.float64)
        prd = torch.randn((1, 1, 8, 16), dtype=torch.float64, generator=generator)
        mask = torch.zeros((1, 1, 8, 16), dtype=torch.float64)
        mask[..., :4, :] = 1.0

        loss = loss_type(as_grid("equiangular", nlat=8, nlon=16)).double()
        polluted = prd.clone()
        polluted[..., 4:, :] = 1e3
        self.assertTrue(compare_tensors("masked integration ignores the masked region", loss(prd, tar, mask=mask), loss(polluted, tar, mask=mask), atol=0, rtol=0))
