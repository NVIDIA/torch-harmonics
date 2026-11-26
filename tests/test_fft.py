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
from parameterized import parameterized, parameterized_class
from testutils import compare_tensors, requires_torch_compile

from torch_harmonics.fft import irfft, rfft

# Error budget in ULPs of the largest reference entry (see testutils.range_atol): an FFT of
# length n carries ~log2(n) roundings per mode (measured: 1.5).
_FFT_ULPS = 16

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


@parameterized_class(("device",), _devices)
class TestFFTWrappers(unittest.TestCase):
    """Tests for the rfft/irfft wrappers in torch_harmonics.fft."""

    @parameterized.expand(
        [
            # batch, nlon, nmodes (nmodes >= nlon//2+1 or None to keep all modes)
            [1, 64, None],
            [4, 64, None],
            [4, 64, 48],
            [4, 128, None],
        ]
    )
    def test_rfft_irfft_roundtrip(self, batch, nlon, nmodes):
        """irfft(rfft(x)) should recover the original signal."""
        x = torch.randn(batch, nlon, device=self.device, dtype=torch.float64)
        coeffs = rfft(x, nmodes=nmodes, dim=-1, norm="forward")
        x_rec = irfft(coeffs, n=nlon, dim=-1, norm="forward")
        self.assertTrue(compare_tensors("roundtrip", x_rec, x, atol=0.0, rtol=0.0, ulps=_FFT_ULPS))

    @parameterized.expand(
        [
            # batch, nlon, nmodes, nlon_out
            [4, 64, 16, 64],
            [4, 64, 16, 128],
            [4, 128, 33, 128],
            [4, 128, 33, 256],
        ]
    )
    def test_rfft_irfft_bandlimited_roundtrip(self, batch, nlon, nmodes, nlon_out):
        """A band-limited signal (only nmodes frequencies) must survive a
        roundtrip through rfft(truncate to nmodes) -> irfft(upsample to nlon_out)
        -> rfft(truncate to nmodes) with the same spectral coefficients."""
        # build a band-limited signal: only the first nmodes frequencies are nonzero
        coeffs = torch.randn(batch, nmodes, device=self.device, dtype=torch.complex128)
        coeffs[:, 0] = coeffs[:, 0].real  # DC must be real
        x = irfft(coeffs, n=nlon, dim=-1, norm="forward")

        # roundtrip: forward FFT with truncation, then inverse to a (possibly different) grid
        coeffs_rt = rfft(x, nmodes=nmodes, dim=-1, norm="forward")
        x_rt = irfft(coeffs_rt, n=nlon_out, dim=-1, norm="forward")
        coeffs_rt2 = rfft(x_rt, nmodes=nmodes, dim=-1, norm="forward")

        self.assertTrue(compare_tensors("band-limited roundtrip", coeffs_rt2, coeffs_rt, atol=0.0, rtol=0.0, ulps=_FFT_ULPS))

    @parameterized.expand(
        [
            # batch, nlon, nmodes
            [1, 64, None],
            [4, 64, None],
            [4, 64, 16],
            [4, 128, 33],
        ]
    )
    def test_rfft_backward(self, batch, nlon, nmodes):
        """Backward through rfft must not error (covers stride-0 grad from sum)."""
        x = torch.randn(batch, nlon, device=self.device, dtype=torch.float32, requires_grad=True)
        coeffs = rfft(x, nmodes=nmodes, dim=-1, norm="forward")
        loss = coeffs.abs().sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

    @parameterized.expand(
        [
            # batch, nmodes, nlon
            [1, 33, 64],
            [4, 33, 64],
            [4, 17, 64],
            [4, 65, 128],
        ]
    )
    def test_irfft_backward(self, batch, nmodes, nlon):
        """Backward through irfft must not error (covers stride-0 grad from sum)."""
        x = torch.randn(batch, nmodes, device=self.device, dtype=torch.complex64, requires_grad=True)
        out = irfft(x, n=nlon, dim=-1, norm="forward")
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

    @parameterized.expand(
        [
            # batch, nmodes, nlon
            [4, 33, 64],
            [4, 65, 128],
        ]
    )
    def test_irfft_hermitian_symmetry(self, batch, nmodes, nlon):
        """irfft must zero the imaginary part of DC and Nyquist bins,
        so injecting imaginary noise at those bins should not change the output."""
        x = torch.randn(batch, nmodes, device=self.device, dtype=torch.complex64)
        out_clean = irfft(x, n=nlon, dim=-1, norm="forward")

        # inject nonzero imaginary parts at DC and Nyquist
        x_dirty = x.clone()
        x_dirty[:, 0] = x_dirty[:, 0] + 1j
        if nlon % 2 == 0 and nlon // 2 < nmodes:
            x_dirty[:, nlon // 2] = x_dirty[:, nlon // 2] + 1j
        out_dirty = irfft(x_dirty, n=nlon, dim=-1, norm="forward")

        self.assertTrue(compare_tensors("hermitian fix", out_dirty, out_clean, atol=0.0, rtol=0.0))

    @parameterized.expand(
        [
            # batch, nmodes, nlon
            [4, 33, 64],
            [4, 65, 128],
        ]
    )
    def test_rfft_irfft_grad_chain(self, batch, nmodes, nlon):
        """Gradient must flow through rfft -> linear op -> irfft without error."""
        x = torch.randn(batch, nlon, device=self.device, dtype=torch.float32, requires_grad=True)
        coeffs = rfft(x, nmodes=nmodes, dim=-1, norm="forward")
        # simple linear op in spectral space
        coeffs = coeffs * 0.5
        out = irfft(coeffs, n=nlon, dim=-1, norm="forward")
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

    @parameterized.expand(
        [
            # shape, dim, nmodes, norm: even and odd lengths, truncated, all bins and padded
            [(3, 16), -1, 5, "forward"],
            [(3, 16), -1, 9, "backward"],
            [(3, 15), -1, 8, "ortho"],
            [(3, 15), -1, 4, None],
            [(3, 16, 4), 1, 6, "forward"],
            [(3, 16), -1, 12, "forward"],
        ]
    )
    def test_rfft_matches_torch_fft(self, shape, dim, nmodes, norm):
        """Values and gradients agree with torch.fft.rfft cut (or zero-padded) to nmodes, also for a strided gradient."""
        x = torch.randn(*shape, device=self.device, dtype=torch.float64)
        x1 = x.clone().requires_grad_()
        x2 = x.clone().requires_grad_()

        y1 = rfft(x1, nmodes=nmodes, dim=dim, norm=norm)
        y2 = torch.fft.rfft(x2, dim=dim, norm=norm)
        nbins = y2.shape[dim]
        if nmodes <= nbins:
            y2 = y2.narrow(dim, 0, nmodes)
        else:
            y2 = torch.cat([y2, y2.new_zeros(*y2.shape[:-1], nmodes - nbins)], dim=-1)
        self.assertTrue(y1.is_contiguous())
        self.assertTrue(compare_tensors("rfft", y1, y2, atol=0.0, rtol=0.0, ulps=_FFT_ULPS))

        # the upstream gradient arrives transposed, as from the SHT
        g = torch.randn_like(y1)
        y1.backward(g.transpose(0, -1).contiguous().transpose(0, -1))
        y2.backward(g)
        self.assertTrue(x1.grad.is_contiguous())
        self.assertTrue(compare_tensors("rfft grad", x1.grad, x2.grad, atol=0.0, rtol=0.0, ulps=_FFT_ULPS))

    @parameterized.expand([[16, 9, "forward"], [16, 6, "ortho"], [15, 8, "backward"], [15, 4, None]])
    def test_rfft_gradcheck(self, nlon, nmodes, norm):
        """First and second derivatives of rfft and irfft."""
        x = torch.randn(2, nlon, device=self.device, dtype=torch.float64, requires_grad=True)
        fn = lambda t: torch.view_as_real(rfft(t, nmodes=nmodes, norm=norm))
        self.assertTrue(torch.autograd.gradcheck(fn, (x,)))
        self.assertTrue(torch.autograd.gradgradcheck(fn, (x,)))
        c = torch.randn(2, nmodes, device=self.device, dtype=torch.complex128, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda t: irfft(t, n=nlon, norm=norm), (c,)))

    def test_rfft_saves_nothing(self):
        """The backward of rfft keeps nothing of its input: torch.fft.rfft's keeps all of it."""
        saved = []

        def pack(t):
            saved.append(t.shape)
            return t

        x = torch.randn(4, 32, 64, device=self.device, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            rfft(x, nmodes=20, norm="forward")
        self.assertEqual(saved, [])

    def test_no_host_sync(self):
        """Neither wrapper synchronizes with the host, forward or backward: irfft's mask is built on the device."""
        if self.device.type != "cuda":
            self.skipTest("stream synchronization is a CUDA notion")
        x = torch.randn(4, 32, 64, device=self.device, requires_grad=True)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            y = irfft(rfft(x, nmodes=20, norm="forward"), n=64, norm="forward")
            y.sum().backward()
        finally:
            torch.cuda.set_sync_debug_mode(0)

    @requires_torch_compile
    def test_compile(self):
        """rfft and irfft compile as one graph and agree with eager, values and gradients."""
        fn = lambda t: irfft(rfft(t, nmodes=20, norm="forward") * 0.5, n=64, norm="forward")
        x = torch.randn(4, 64, device=self.device, dtype=torch.float32, requires_grad=True)
        expected = fn(x)
        expected.pow(2).sum().backward()
        grad, x.grad = x.grad, None
        actual = torch.compile(fn, fullgraph=True)(x)
        actual.pow(2).sum().backward()
        self.assertTrue(compare_tensors("compiled", actual, expected, atol=1e-6, rtol=0.0))
        self.assertTrue(compare_tensors("compiled grad", x.grad, grad, atol=1e-6, rtol=0.0))

    @parameterized.expand([[16, 9], [16, 5], [15, 8]])
    def test_conjugated_spectrum(self, nlon, nmodes):
        """irfft accepts a conjugate view, as torch.fft.irfft does, and differentiates through it."""
        x = torch.randn(3, nmodes, device=self.device, dtype=torch.complex128)
        xc = x.clone().requires_grad_()
        out = irfft(xc.conj(), n=nlon, norm="forward")

        # reference: the resolved conjugate with the DC and Nyquist imaginary parts zeroed
        ref = x.conj().resolve_conj().clone()
        ref[:, 0] = ref[:, 0].real
        if nlon % 2 == 0 and nlon // 2 < nmodes:
            ref[:, nlon // 2] = ref[:, nlon // 2].real
        ref = ref.requires_grad_()
        expected = torch.fft.irfft(ref, n=nlon, norm="forward")
        self.assertTrue(compare_tensors("irfft conj", out, expected, atol=0.0, rtol=0.0, ulps=_FFT_ULPS))

        # the gradient through the conjugate view is the conjugate of the plain one, up to the zeroed imaginary parts
        g = torch.randn_like(out)
        out.backward(g)
        expected.backward(g)
        grad = ref.grad.conj()
        grad[:, 0] = grad[:, 0].real
        if nlon % 2 == 0 and nlon // 2 < nmodes:
            grad[:, nlon // 2] = grad[:, nlon // 2].real
        self.assertTrue(compare_tensors("irfft conj grad", xc.grad, grad, atol=0.0, rtol=0.0, ulps=_FFT_ULPS))

    def test_torch_func(self):
        """torch.func.grad and vmap through rfft agree with autograd."""
        x = torch.randn(4, 16, device=self.device, dtype=torch.float64)
        fn = lambda t: rfft(t, nmodes=6, norm="forward").abs().pow(2).sum()
        xg = x.clone().requires_grad_()
        fn(xg).backward()
        self.assertTrue(compare_tensors("func.grad", torch.func.grad(fn)(x), xg.grad, atol=0.0, rtol=0.0, ulps=_FFT_ULPS))
        per_row = torch.func.vmap(torch.func.grad(fn))(x)
        self.assertTrue(compare_tensors("vmap(grad)", per_row, xg.grad, atol=0.0, rtol=0.0, ulps=_FFT_ULPS))
