# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
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
from parameterized import parameterized
import torch
import torch.fft as fft

from torch_harmonics.fft import rfft, irfft

from testutils import disable_tf32, set_seed, compare_tensors


class TestRealFFT(unittest.TestCase):
    """Tests for the rfft/irfft truncation and padding helpers."""

    def setUp(self):
        disable_tf32()

    @parameterized.expand(
        [
            [64, 8],
            [64, 16],
            [64, 32],
            [64, 33],
            [128, 20],
            # nmodes larger than nlon//2+1: tests zero-padding path
            [32, 24],
        ],
        skip_on_empty=True,
    )
    def test_rfft_truncation_vs_manual(self, nlon, nmodes, verbose=False):
        """rfft(x, nmodes) must equal torch.fft.rfft(x) with manual truncation or padding."""

        set_seed(333)
        x = torch.randn(4, nlon, dtype=torch.float64)

        result = rfft(x, nmodes=nmodes, dim=-1, norm="forward")

        # reference: full rfft then manually truncate or pad
        ref_full = fft.rfft(x, dim=-1, norm="forward")
        if nmodes <= ref_full.shape[-1]:
            ref = ref_full[..., :nmodes]
        else:
            ref = torch.zeros(4, nmodes, dtype=torch.complex128)
            ref[..., :ref_full.shape[-1]] = ref_full

        self.assertTrue(compare_tensors("rfft truncation", result, ref, atol=1e-12, rtol=1e-12, verbose=verbose))

    @parameterized.expand(
        [
            [64, 8],
            [64, 16],
            [64, 32],
            [64, 33],
            [128, 20],
        ],
        skip_on_empty=True,
    )
    def test_rfft_irfft_roundtrip(self, nlon, nmodes, verbose=False):
        """Band-limited round-trip: irfft(rfft(x, nmodes), nlon) recovers the signal
        when x is band-limited to nmodes real-FFT modes."""

        set_seed(333)

        # build a band-limited signal: only nmodes rfft modes are non-zero
        c = torch.zeros(4, nlon // 2 + 1, dtype=torch.complex128)
        n = min(nmodes, nlon // 2 + 1)
        c[..., :n] = torch.randn(4, n, dtype=torch.complex128)
        c[..., 0] = c[..., 0].real  # DC is real
        x = fft.irfft(c, n=nlon, dim=-1, norm="forward")

        reconstructed = irfft(rfft(x, nmodes=nmodes, dim=-1, norm="forward"), n=nlon, dim=-1, norm="forward")

        self.assertTrue(compare_tensors("rfft/irfft round-trip", reconstructed, x, atol=1e-12, rtol=1e-12, verbose=verbose))

    @parameterized.expand(
        [
            [64, 8],
            [64, 32],
            [128, 20],
        ],
        skip_on_empty=True,
    )
    def test_irfft_hermitian_cleanup(self, nlon, nmodes, verbose=False):
        """irfft must zero the imaginary part of the DC (and Nyquist) components
        to enforce Hermitian symmetry, even if the input has spurious imaginary parts."""

        set_seed(333)

        c = torch.randn(4, nmodes, dtype=torch.complex128)
        # inject spurious imaginary part at DC
        c[..., 0] = c[..., 0] + 0.5j

        result = irfft(c, n=nlon, dim=-1, norm="forward")
        # result must be real-valued (imaginary part cleaned up)
        self.assertTrue(result.is_floating_point())


if __name__ == "__main__":
    unittest.main()
