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

import contextlib
import io
import math
import unittest
from unittest import mock

import testutils
import torch
from parameterized import parameterized
from testutils import compare_tensors, param_grad_ulps, range_atol


def _cancelled_reference(dtype, scale=100.0):
    """
    A reference with one large entry and one exactly cancelled one, the shape of a parameter
    gradient that the range-scaled floor exists for: the floor is set by ``scale`` and is
    tested at the zero entry, where neither an rtol nor a small atol can help.
    """
    return torch.tensor([scale, 0.0, -0.5 * scale], dtype=dtype)


def _perturbed(ref, delta):
    """``ref`` with ``delta`` added to its cancelled (zero) entry."""
    out = ref.clone()
    out[1] += delta
    return out


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class TestRangeAtol(unittest.TestCase):
    """range_atol: atol raised to ulps ULPs of max|ref|, in the reference's or a given precision."""

    @parameterized.expand([[torch.float32], [torch.float64]])
    def test_the_floor_is_ulps_of_the_largest_entry(self, dtype):
        ref = _cancelled_reference(dtype)
        expected = 64 * torch.finfo(dtype).eps * 100.0
        self.assertAlmostEqual(range_atol(ref, 0.0, 64), expected, delta=1e-6 * expected)

    def test_an_explicit_eps_dtype_replaces_the_reference_dtype(self):
        # fp32 gradients accumulated from bf16 products: the budget is in bf16 ULPs
        ref = _cancelled_reference(torch.float32)
        expected = 2 * torch.finfo(torch.bfloat16).eps * 100.0
        self.assertAlmostEqual(range_atol(ref, 0.0, 2, eps_dtype=torch.bfloat16), expected, delta=1e-6 * expected)

    def test_atol_stays_a_floor(self):
        ref = _cancelled_reference(torch.float32)
        self.assertEqual(range_atol(ref, 1.0, 64), 1.0)

    def test_a_complex_reference_uses_its_real_precision_and_modulus(self):
        ref = torch.tensor([3.0 + 4.0j, 0.0], dtype=torch.complex64)
        expected = 8 * torch.finfo(torch.float32).eps * 5.0
        self.assertAlmostEqual(range_atol(ref, 0.0, 8), expected, delta=1e-6 * expected)


class TestParamGradUlps(unittest.TestCase):
    """param_grad_ulps: sqrt(N) for fp32/fp64, floored at 64; a constant 2 under autocast."""

    @parameterized.expand([[torch.float32], [torch.float64]])
    def test_it_grows_like_the_root_of_the_reduction(self, dtype):
        n = 32 * 64 * 128
        for factor in (1, 4, 16):
            with self.subTest(n=factor * n):
                self.assertEqual(param_grad_ulps(factor * n, dtype), math.sqrt(factor) * math.sqrt(n))

    @parameterized.expand([[torch.float32], [torch.float64]])
    def test_small_reductions_get_the_floor(self, dtype):
        for n in (1, 100, 64**2):
            with self.subTest(n=n):
                self.assertEqual(param_grad_ulps(n, dtype), 64.0)

    @parameterized.expand([[torch.float16], [torch.bfloat16]])
    def test_autocast_does_not_grow_with_the_reduction(self, dtype):
        for n in (1, 10**4, 10**8):
            with self.subTest(n=n):
                self.assertEqual(param_grad_ulps(n, dtype), 2.0)


class TestCompareTensors(unittest.TestCase):
    """
    compare_tensors accepts an error just inside its bound and rejects one just outside, for each
    way the bound is set. The bound is |t1 - t2| <= atol_eff + rtol * |t2|, and every case below
    puts the error on the cancelled entry with rtol=0, so it is atol_eff alone that is measured.
    """

    # how far inside or outside the bound a case sits; well above rounding of the perturbation
    margin = 0.01

    def _assert_boundary(self, ref, limit, **kwargs):
        for side, factor, expected in (("inside", 1 - self.margin, True), ("outside", 1 + self.margin, False)):
            with self.subTest(side):
                error = factor * limit
                ok = _quiet(compare_tensors, side, _perturbed(ref, error), ref, rtol=0.0, **kwargs)
                self.assertEqual(ok, expected, f"{'rejected' if expected else 'accepted'} an error of {error:.3e} against a limit of {limit:.3e}")

    @parameterized.expand([[torch.float32], [torch.float64]])
    def test_without_ulps_it_is_a_plain_atol(self, dtype):
        ref = _cancelled_reference(dtype)
        self._assert_boundary(ref, 1e-3, atol=1e-3)

    @parameterized.expand([[torch.float32, 64], [torch.float64, 64], [torch.float32, 512]])
    def test_the_floor_in_the_reference_precision(self, dtype, ulps):
        ref = _cancelled_reference(dtype)
        self._assert_boundary(ref, ulps * torch.finfo(dtype).eps * 100.0, atol=0.0, ulps=ulps)

    @parameterized.expand([[torch.float16], [torch.bfloat16]])
    def test_the_floor_in_an_explicit_precision(self, eps_dtype):
        ref = _cancelled_reference(torch.float32)
        self._assert_boundary(ref, 2 * torch.finfo(eps_dtype).eps * 100.0, atol=0.0, ulps=2, eps_dtype=eps_dtype)

    def test_an_explicit_precision_is_not_ignored(self):
        # 2 bf16 ULPs of 100 is ~1.6; the same budget in the fp32 reference's own ULPs is ~2.4e-5
        ref = _cancelled_reference(torch.float32)
        got = _perturbed(ref, 1.0)
        for eps_dtype, expected in ((torch.bfloat16, True), (None, False)):
            with self.subTest(eps_dtype=eps_dtype):
                self.assertEqual(_quiet(compare_tensors, "eps_dtype", got, ref, atol=0.0, rtol=0.0, ulps=2, eps_dtype=eps_dtype), expected)

    def test_the_floor_scales_with_the_second_argument(self):
        # The reference is tensor2, and a tested result that came out larger must not widen its
        # own bound. Here the result's range is 10% above the reference's (its large entry still
        # within rtol), and the cancelled entry is 5% above the reference-based limit: a floor
        # taken from tensor1 would be 10% wider and accept it.
        ref = torch.tensor([1000.0, 0.0], dtype=torch.float32)
        limit = 64 * torch.finfo(torch.float32).eps * 1000.0
        cases = (
            ("wider result", torch.tensor([1100.0, 1.05 * limit]), False),
            # and an error just inside the reference-based limit is accepted
            ("matched range", torch.tensor([1000.0, 0.95 * limit]), True),
        )
        for name, got, expected in cases:
            with self.subTest(name):
                self.assertEqual(_quiet(compare_tensors, name, got, ref, atol=0.0, rtol=0.2, ulps=64), expected)

    def test_atol_stays_a_floor_under_ulps(self):
        ref = _cancelled_reference(torch.float32)
        self._assert_boundary(ref, 1e-2, atol=1e-2, ulps=1)

    def test_rtol_still_applies_on_large_entries(self):
        ref = _cancelled_reference(torch.float32)
        got = ref.clone()
        got[0] += 0.5  # 0.5% of the entry
        for rtol, expected in ((1e-2, True), (1e-3, False)):
            with self.subTest(rtol=rtol):
                self.assertEqual(_quiet(compare_tensors, "rtol", got, ref, atol=0.0, rtol=rtol, ulps=1), expected)

    def test_integers_compare_exactly(self):
        ref = torch.tensor([1, 2, 3])
        for name, got, expected in (("equal", ref.clone(), True), ("off by one", ref + torch.tensor([0, 0, 1]), False)):
            with self.subTest(name):
                self.assertEqual(bool(_quiet(compare_tensors, "ints", got, ref, atol=10, rtol=10, verbose=True)), expected)

    def test_none_matches_only_none(self):
        ref = torch.zeros(2)
        for name, t1, t2, expected in (("both", None, None, True), ("first", None, ref, False), ("second", ref, None, False)):
            with self.subTest(name):
                self.assertEqual(_quiet(compare_tensors, "none", t1, t2, verbose=True), expected)


class TestCompareTensorsReport(unittest.TestCase):
    """The margin line, which is what a ULP budget is calibrated from."""

    def _output(self, *args, **kwargs):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ok = compare_tensors(*args, **kwargs)
        return ok, buf.getvalue()

    def test_a_passing_check_is_silent_by_default(self):
        ref = _cancelled_reference(torch.float32)
        with mock.patch.object(testutils, "_REPORT_MARGINS", False):
            ok, out = self._output("quiet", ref.clone(), ref, atol=0.0, rtol=0.0)
        self.assertTrue(ok)
        self.assertEqual(out, "")

    def test_report_prints_the_error_in_ulps_of_the_reference(self):
        ref = _cancelled_reference(torch.float32)
        got = _perturbed(ref, 10 * torch.finfo(torch.float32).eps * 100.0)
        ok, out = self._output("reported", got, ref, atol=1.0, rtol=0.0, report=True)
        self.assertTrue(ok)
        self.assertIn("Margin on reported", out)
        self.assertIn("10.0 ULPs of max|ref|", out)

    def test_the_environment_switch_reports_every_check(self):
        ref = _cancelled_reference(torch.float32)
        with mock.patch.object(testutils, "_REPORT_MARGINS", True):
            ok, out = self._output("env", ref.clone(), ref, atol=0.0, rtol=0.0)
        self.assertTrue(ok)
        self.assertIn("Margin on env", out)

    def test_a_verbose_failure_reports_the_enforced_atol(self):
        ref = _cancelled_reference(torch.float32)
        got = _perturbed(ref, 1.0)
        ok, out = self._output("failed", got, ref, atol=0.0, rtol=0.0, ulps=64, verbose=True)
        self.assertFalse(ok)
        # the floored atol, not the atol=0 that was passed in
        self.assertIn(f"atol {64 * torch.finfo(torch.float32).eps * 100.0:.1e}", out)


if __name__ == "__main__":
    unittest.main()
