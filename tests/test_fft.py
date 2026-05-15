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
from parameterized import parameterized, parameterized_class
import torch

from torch_harmonics.fft import rfft, irfft

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


@parameterized_class(("device",), _devices)
class TestFFTWrappers(unittest.TestCase):
    """Tests for the rfft/irfft wrappers in torch_harmonics.fft."""

    @parameterized.expand([
        # batch, nlon, nmodes (nmodes >= nlon//2+1 or None to keep all modes)
        [1, 64, None],
        [4, 64, None],
        [4, 64, 48],
        [4, 128, None],
    ])
    def test_rfft_irfft_roundtrip(self, batch, nlon, nmodes):
        """irfft(rfft(x)) should recover the original signal."""
        x = torch.randn(batch, nlon, device=self.device, dtype=torch.float64)
        coeffs = rfft(x, nmodes=nmodes, dim=-1, norm="forward")
        x_rec = irfft(coeffs, n=nlon, dim=-1, norm="forward")
        self.assertTrue(
            torch.allclose(x, x_rec, atol=1e-10),
            f"Roundtrip failed: max error {(x - x_rec).abs().max().item():.2e}",
        )

    @parameterized.expand([
        # batch, nlon, nmodes, nlon_out
        [4, 64, 16, 64],
        [4, 64, 16, 128],
        [4, 128, 33, 128],
        [4, 128, 33, 256],
    ])
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

        self.assertTrue(
            torch.allclose(coeffs_rt, coeffs_rt2, atol=1e-10),
            f"Band-limited roundtrip failed: max error {(coeffs_rt - coeffs_rt2).abs().max().item():.2e}",
        )

    @parameterized.expand([
        # batch, nlon, nmodes
        [1, 64, None],
        [4, 64, None],
        [4, 64, 16],
        [4, 128, 33],
    ])
    def test_rfft_backward(self, batch, nlon, nmodes):
        """Backward through rfft must not error (covers stride-0 grad from sum)."""
        x = torch.randn(batch, nlon, device=self.device, dtype=torch.float32, requires_grad=True)
        coeffs = rfft(x, nmodes=nmodes, dim=-1, norm="forward")
        loss = coeffs.abs().sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

    @parameterized.expand([
        # batch, nmodes, nlon
        [1, 33, 64],
        [4, 33, 64],
        [4, 17, 64],
        [4, 65, 128],
    ])
    def test_irfft_backward(self, batch, nmodes, nlon):
        """Backward through irfft must not error (covers stride-0 grad from sum)."""
        x = torch.randn(batch, nmodes, device=self.device, dtype=torch.complex64, requires_grad=True)
        out = irfft(x, n=nlon, dim=-1, norm="forward")
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

    @parameterized.expand([
        # batch, nmodes, nlon
        [4, 33, 64],
        [4, 65, 128],
    ])
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

        self.assertTrue(
            torch.allclose(out_clean, out_dirty, atol=1e-6),
            f"Hermitian fix failed: max diff {(out_clean - out_dirty).abs().max().item():.2e}",
        )

    @parameterized.expand([
        # batch, nmodes, nlon
        [4, 33, 64],
        [4, 65, 128],
    ])
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


if __name__ == "__main__":
    unittest.main()
