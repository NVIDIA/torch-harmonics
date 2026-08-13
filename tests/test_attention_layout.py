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

"""
Tests for the attention layout conversion helpers (``attention/_layout.py``).

A layout conversion is a pure element permutation, so every check here is
**bitwise**: there is no arithmetic to lose precision, and any tolerance would
be hiding a bug rather than accommodating one.
"""

import unittest

import torch
from parameterized import parameterized, parameterized_class
from testutils import set_seed

from torch_harmonics.attention import optimized_kernels_is_available
from torch_harmonics.attention._layout import to_nchw, to_nhwc

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))

# fp16/bf16 are only exercised on CUDA: the tiled transpose dispatches all three
# dtypes through the same templated kernel, and that dispatch is what we want to
# pin. On CPU the fallback is ATen's copy, which needs no dtype coverage here.
_dtypes_cpu = [torch.float32]
_dtypes_cuda = [torch.float32, torch.float16, torch.bfloat16]

# (B, C, H, W). The degenerate entries are the cases where stride inspection
# cannot identify a layout: C == 1 makes a plain NCHW tensor report
# ``is_contiguous(channels_last) == True``, and H*W == 1 makes it report
# ``stride(1) == 1``. Both must round-trip regardless.
_shapes = [
    (2, 8, 6, 12),
    (1, 8, 6, 12),
    (4, 33, 7, 13),  # non-power-of-two channels, exercises the tile remainder path
    (2, 1, 6, 12),  # single channel
    (2, 8, 1, 12),  # single latitude
    (2, 8, 6, 1),  # single longitude
    (2, 8, 1, 1),  # single spatial point
    (1, 1, 1, 1),  # fully degenerate
]


@parameterized_class(("device"), _devices)
class TestAttentionLayout(unittest.TestCase):
    """Round-trip, gradient, dtype and torch.compile behaviour of to_nhwc/to_nchw."""

    def setUp(self):
        set_seed(333)

    def _dtypes(self):
        return _dtypes_cuda if self.device.type == "cuda" else _dtypes_cpu

    @parameterized.expand(_shapes)
    def test_forward_matches_permute(self, B, C, H, W):
        """to_nhwc/to_nchw agree bitwise with the equivalent ATen permute."""

        for dtype in self._dtypes():
            with self.subTest(dtype=dtype):
                x = torch.randn(B, C, H, W, device=self.device, dtype=dtype)

                y = to_nhwc(x)
                self.assertEqual(tuple(y.shape), (B, H, W, C))
                self.assertTrue(y.is_contiguous())
                self.assertEqual(y.dtype, dtype)
                self.assertTrue(torch.equal(y, x.permute(0, 2, 3, 1).contiguous()))

                z = torch.randn(B, H, W, C, device=self.device, dtype=dtype)
                zc = to_nchw(z)
                self.assertEqual(tuple(zc.shape), (B, C, H, W))
                self.assertTrue(zc.is_contiguous())
                self.assertEqual(zc.dtype, dtype)
                self.assertTrue(torch.equal(zc, z.permute(0, 3, 1, 2).contiguous()))

    @parameterized.expand(_shapes)
    def test_roundtrip(self, B, C, H, W):
        """to_nchw(to_nhwc(x)) is the identity, including on degenerate shapes."""

        for dtype in self._dtypes():
            with self.subTest(dtype=dtype):
                x = torch.randn(B, C, H, W, device=self.device, dtype=dtype)
                self.assertTrue(torch.equal(to_nchw(to_nhwc(x)), x))

    @parameterized.expand(_shapes)
    def test_output_does_not_alias_input(self, B, C, H, W):
        """The conversion always returns storage the caller owns.

        For C == 1 and H*W == 1 the two layouts hold identical bytes, so a
        naive ``permute(...).contiguous()`` returns a view of the input instead
        of a copy. That violates the op schema (which declares no aliasing) and
        would let a write into the converted tensor corrupt the source.
        """

        x = torch.randn(B, C, H, W, device=self.device, dtype=torch.float32)
        x_ref = x.clone()

        y = to_nhwc(x)
        y.zero_()
        self.assertTrue(torch.equal(x, x_ref), "to_nhwc output aliases its input")

        z = torch.randn(B, H, W, C, device=self.device, dtype=torch.float32)
        z_ref = z.clone()

        zc = to_nchw(z)
        zc.zero_()
        self.assertTrue(torch.equal(z, z_ref), "to_nchw output aliases its input")

    @parameterized.expand(_shapes)
    def test_backward(self, B, C, H, W):
        """The backward pass is the opposite conversion applied to grad_output."""

        for dtype in self._dtypes():
            with self.subTest(dtype=dtype):
                x = torch.randn(B, C, H, W, device=self.device, dtype=dtype, requires_grad=True)
                grad = torch.randn(B, H, W, C, device=self.device, dtype=dtype)

                to_nhwc(x).backward(grad)

                self.assertEqual(tuple(x.grad.shape), (B, C, H, W))
                self.assertTrue(torch.equal(x.grad, grad.permute(0, 3, 1, 2).contiguous()))

                # and the other direction
                z = torch.randn(B, H, W, C, device=self.device, dtype=dtype, requires_grad=True)
                grad_z = torch.randn(B, C, H, W, device=self.device, dtype=dtype)

                to_nchw(z).backward(grad_z)

                self.assertEqual(tuple(z.grad.shape), (B, H, W, C))
                self.assertTrue(torch.equal(z.grad, grad_z.permute(0, 2, 3, 1).contiguous()))

    @parameterized.expand(_shapes)
    def test_backward_noncontiguous_grad(self, B, C, H, W):
        """A non-contiguous incoming gradient is handled, not rejected."""

        x = torch.randn(B, C, H, W, device=self.device, dtype=torch.float32, requires_grad=True)
        # transposing the two spatial dims of a (B, H, W, C) tensor makes the
        # gradient non-contiguous without changing its logical shape (it stays
        # contiguous for the degenerate shapes where H == W == 1, which is fine
        # -- the point is that neither case is rejected)
        grad = torch.randn(B, W, H, C, device=self.device, dtype=torch.float32).transpose(1, 2)

        to_nhwc(x).backward(grad)
        self.assertTrue(torch.equal(x.grad, grad.permute(0, 3, 1, 2).contiguous()))

    @parameterized.expand(_shapes)
    def test_compile(self, B, C, H, W):
        """Both directions compile in a single graph and match eager bitwise."""

        def fn(t):
            return to_nchw(to_nhwc(t) * 2.0)

        x = torch.randn(B, C, H, W, device=self.device, dtype=torch.float32)
        expected = fn(x)

        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        self.assertTrue(torch.equal(compiled(x), expected))

    @parameterized.expand(_shapes)
    def test_autocast_preserves_dtype(self, B, C, H, W):
        """Under autocast the conversion is dtype-preserving, not dtype-casting.

        No autocast kernel is registered for these ops (the autocast keys fall
        through), which is deliberate: a layout conversion is a pure element
        permutation and must not silently change precision. If someone later
        adds an AutocastCUDA rule by analogy with the attention ops, this fails.
        """

        if self.device.type != "cuda":
            self.skipTest("autocast dtype behaviour is only meaningful on CUDA here")

        for autocast_dtype in (torch.float16, torch.bfloat16):
            for dtype in self._dtypes():
                with self.subTest(autocast_dtype=autocast_dtype, dtype=dtype):
                    x = torch.randn(B, C, H, W, device=self.device, dtype=dtype)
                    with torch.autocast("cuda", dtype=autocast_dtype):
                        y = to_nhwc(x)
                    self.assertEqual(y.dtype, dtype)
                    self.assertTrue(torch.equal(y, x.permute(0, 2, 3, 1).contiguous()))

    @unittest.skipUnless(optimized_kernels_is_available(), "optimized kernels not available")
    @parameterized.expand(_shapes)
    def test_opcheck(self, B, C, H, W):
        """Schema, fake impl and autograd registration are mutually consistent."""

        for dtype in self._dtypes():
            with self.subTest(dtype=dtype):
                x = torch.randn(B, C, H, W, device=self.device, dtype=dtype, requires_grad=True)
                torch.library.opcheck(torch.ops.attention_kernels.permute_to_nhwc.default, (x,))

                z = torch.randn(B, H, W, C, device=self.device, dtype=dtype, requires_grad=True)
                torch.library.opcheck(torch.ops.attention_kernels.permute_to_nchw.default, (z,))


if __name__ == "__main__":
    unittest.main()
