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

from torch_harmonics.spectral_convolution import SpectralConvS2

from testutils import disable_tf32, set_seed, compare_tensors

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


@parameterized_class(("device",), _devices)
class TestSpectralConvS2(unittest.TestCase):

    def setUp(self):
        disable_tf32()

    # -----------------------------------------------------------------------
    # Test 1: Zonal rotation equivariance
    #
    # A cyclic shift of the input along the longitude axis (dim=-1) by k steps
    # must produce an identically shifted output.  This holds because:
    #   - equiangular grids have uniform longitude spacing, so a cyclic shift
    #     is a rotation by an exact grid multiple
    #   - the forward SHT decomposes via DFT in longitude, so a shift by k
    #     introduces a phase factor exp(i*m*2π*k/nlon) on each order m
    #   - SpectralConvS2 weights depend only on degree l, not on order m, so
    #     the phase factor passes through the contraction unchanged
    #   - the inverse SHT maps the phase-shifted coefficients back to a
    #     cyclically shifted spatial field
    # -----------------------------------------------------------------------
    @parameterized.expand(
        [
            # nlat, nlon, in_channels, out_channels, num_groups, lon_shift
            [32, 64, 4, 4, 1, 8],
            [32, 64, 4, 4, 2, 16],
            [32, 64, 1, 1, 1, 7],
            [16, 32, 8, 4, 2, 3],
        ],
        skip_on_empty=True,
    )
    def test_zonal_rotation_equivariance(self, nlat, nlon, in_channels, out_channels, num_groups, lon_shift, verbose=False):
        """Conv(Roll(x, k)) ≈ Roll(Conv(x), k) for any longitude shift k."""

        # set seed
        set_seed(333)

        conv = SpectralConvS2(
            in_shape=(nlat, nlon),
            out_shape=(nlat, nlon),
            in_channels=in_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            grid_in="equiangular",
            grid_out="equiangular",
        ).to(self.device)
        conv.eval()

        x = torch.randn(2, in_channels, nlat, nlon, device=self.device)

        with torch.no_grad():
            y = conv(x)
            x_shifted = torch.roll(x, lon_shift, dims=-1)
            y_of_shifted = conv(x_shifted)
            shift_of_y = torch.roll(y, lon_shift, dims=-1)

        self.assertTrue(
            compare_tensors(
                "zonal rotation equivariance",
                y_of_shifted, shift_of_y,
                atol=1e-5, rtol=1e-4,
                verbose=verbose,
            )
        )

    # -----------------------------------------------------------------------
    # Test 4: Grouped convolution consistency
    #
    # A SpectralConvS2 with num_groups=G must produce the same result as
    # running G independent SpectralConvS2 layers (each with num_groups=1)
    # on the corresponding channel slices, using the same weight values.
    # -----------------------------------------------------------------------
    @parameterized.expand(
        [
            # nlat, nlon, in_channels, out_channels, num_groups
            [32, 64, 4, 4, 2],
            [32, 64, 4, 4, 4],  # depthwise (num_groups == in_channels)
            [16, 32, 6, 6, 3],
        ],
        skip_on_empty=True,
    )
    def test_grouped_convolution_consistency(self, nlat, nlon, in_channels, out_channels, num_groups, verbose=False):
        """num_groups > 1 equals running separate num_groups=1 convs on channel subsets."""

        # set seed
        set_seed(333)

        conv_grouped = SpectralConvS2(
            in_shape=(nlat, nlon),
            out_shape=(nlat, nlon),
            in_channels=in_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            grid_in="equiangular",
            grid_out="equiangular",
        ).to(self.device)
        conv_grouped.eval()

        ic_per_group = in_channels // num_groups
        oc_per_group = out_channels // num_groups

        # Build one num_groups=1 conv per group and copy the matching weight slice
        group_convs = []
        for g in range(num_groups):
            gc = SpectralConvS2(
                in_shape=(nlat, nlon),
                out_shape=(nlat, nlon),
                in_channels=ic_per_group,
                out_channels=oc_per_group,
                num_groups=1,
                grid_in="equiangular",
                grid_out="equiangular",
            ).to(self.device)
            # weight shape: [num_groups, ic_per_group, oc_per_group, lmax]
            gc.weight.data = conv_grouped.weight[g : g + 1].clone()
            gc.eval()
            group_convs.append(gc)

        x = torch.randn(2, in_channels, nlat, nlon, device=self.device)

        with torch.no_grad():
            y_grouped = conv_grouped(x)
            y_manual = torch.cat(
                [gc(x[:, g * ic_per_group : (g + 1) * ic_per_group]) for g, gc in enumerate(group_convs)],
                dim=1,
            )

        self.assertTrue(
            compare_tensors(
                "grouped convolution consistency",
                y_grouped, y_manual,
                atol=1e-5, rtol=1e-4,
                verbose=verbose,
            )
        )

    # -----------------------------------------------------------------------
    # Test 6: Dtype preservation
    #
    # The module casts inputs to float32 internally for the SHT, but must
    # restore the original dtype on output via the final .to(dtype=dtype) call.
    # -----------------------------------------------------------------------
    @parameterized.expand(
        [
            # nlat, nlon, in_channels, out_channels, dtype
            [32, 64, 4, 4, torch.bfloat16],
            [32, 64, 4, 4, torch.float16],
        ],
        skip_on_empty=True,
    )
    def test_dtype_preservation(self, nlat, nlon, in_channels, out_channels, dtype, verbose=False):
        """Output dtype must match input dtype after the internal float32 cast."""

        # set seed
        set_seed(333)

        conv = SpectralConvS2(
            in_shape=(nlat, nlon),
            out_shape=(nlat, nlon),
            in_channels=in_channels,
            out_channels=out_channels,
            grid_in="equiangular",
            grid_out="equiangular",
        ).to(self.device)
        conv.eval()

        x = torch.randn(2, in_channels, nlat, nlon, device=self.device, dtype=dtype)
        with torch.no_grad():
            y = conv(x)

        self.assertEqual(
            y.dtype, dtype,
            msg=f"Expected output dtype {dtype}, got {y.dtype}",
        )

    # -----------------------------------------------------------------------
    # Test 7: Batch independence
    #
    # Stacking two identical inputs must yield identical outputs up to float32
    # non-determinism from @torch.compile.  The compiled _contract_lwise einsum
    # parallelises the reduction over threads; different batch positions can be
    # assigned to different threads, so partial sums accumulate in a different
    # order.  Differences are O(1e-7) — not a bug, just float32 non-associativity.
    # The tolerance is set to 1e-6 to stay safely above this noise floor while
    # still catching any real cross-sample mixing.
    # -----------------------------------------------------------------------
    @parameterized.expand(
        [
            # nlat, nlon, in_channels, out_channels, num_groups
            [32, 64, 4, 4, 1, 1e-6, 1e-5],
            [32, 64, 4, 4, 2, 1e-6, 1e-5],
        ],
        skip_on_empty=True,
    )
    def test_batch_independence(self, nlat, nlon, in_channels, out_channels, num_groups, atol, rtol, verbose=True):
        """Two identical samples in a batch must produce bitwise-identical outputs."""

        # set seed
        set_seed(333)

        conv = SpectralConvS2(
            in_shape=(nlat, nlon),
            out_shape=(nlat, nlon),
            in_channels=in_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            grid_in="equiangular",
            grid_out="equiangular",
        ).to(self.device)
        conv.eval()

        x_single = torch.randn(1, in_channels, nlat, nlon, device=self.device)
        # two identical samples
        x_batch = x_single.expand(2, -1, -1, -1).contiguous()

        with torch.no_grad():
            y_single = conv(x_single)
            y_batch = conv(x_batch)

        # the two batched outputs must be identical to each other …
        self.assertTrue(
            compare_tensors("batch independence [0 vs 1]", y_batch[0], y_batch[1], atol=atol, rtol=rtol, verbose=verbose)
        )
        # … and identical to running the single sample alone
        self.assertTrue(
            compare_tensors("batch independence [single vs batched]", y_single[0], y_batch[0], atol=atol, rtol=rtol, verbose=verbose)
        )

    # -----------------------------------------------------------------------
    # Test 8: Zero weights → zero output
    #
    # With all weights set to zero the output must be exactly zero for any
    # input (the bias-free path is purely linear in the weights).
    # -----------------------------------------------------------------------
    @parameterized.expand(
        [
            # nlat, nlon, in_channels, out_channels, num_groups
            [32, 64, 4, 4, 1],
            [32, 64, 4, 4, 2],
            [16, 32, 1, 1, 1],
        ],
        skip_on_empty=True,
    )
    def test_zero_weights_zero_output(self, nlat, nlon, in_channels, out_channels, num_groups, verbose=False):
        """Zeroing all weights must produce an exactly-zero output."""

        # set seed
        set_seed(333)

        conv = SpectralConvS2(
            in_shape=(nlat, nlon),
            out_shape=(nlat, nlon),
            in_channels=in_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            grid_in="equiangular",
            grid_out="equiangular",
        ).to(self.device)
        conv.eval()

        conv.weight.data.zero_()

        x = torch.randn(2, in_channels, nlat, nlon, device=self.device)
        with torch.no_grad():
            y = conv(x)

        self.assertTrue(
            compare_tensors(
                "zero weights zero output",
                y, torch.zeros_like(y),
                atol=0.0, rtol=0.0,
                verbose=verbose,
            )
        )


    # -----------------------------------------------------------------------
    # Test 9: Spectral bias
    #
    # With bias=True the layer computes:
    #   x_spectral += integral(x_spatial) * spectral_bias
    # before the weight contraction, where integral is the quadrature-weighted
    # spatial sum (a scalar per channel per batch element).
    #
    # Three properties to verify:
    #  a) zero spectral_bias (the initialised state) is a no-op: the output
    #     matches a bias=False layer with identical weights.
    #  b) setting spectral_bias to a nonzero value changes the output.
    #  c) gradients flow back through spectral_bias after a backward pass.
    # -----------------------------------------------------------------------
    @parameterized.expand(
        [
            # nlat, nlon, channels, num_groups
            [32, 64, 4, 1],
            [32, 64, 4, 2],
        ],
        skip_on_empty=True,
    )
    def test_spectral_bias_zero_equals_no_bias(self, nlat, nlon, channels, num_groups, verbose=False):
        """Zero spectral_bias must produce the same output as bias=False."""
        set_seed(333)

        conv_no_bias = SpectralConvS2(
            in_shape=(nlat, nlon),
            out_shape=(nlat, nlon),
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            grid_in="equiangular",
            grid_out="equiangular",
            bias=False,
        ).to(self.device)

        set_seed(333)
        conv_bias = SpectralConvS2(
            in_shape=(nlat, nlon),
            out_shape=(nlat, nlon),
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            grid_in="equiangular",
            grid_out="equiangular",
            bias=True,
        ).to(self.device)
        # spectral_bias is zero-initialised; weights must match
        conv_bias.weight.data.copy_(conv_no_bias.weight.data)
        # confirm bias is zero
        self.assertTrue(conv_bias.spectral_bias.data.abs().max().item() == 0.0)

        x = torch.randn(2, channels, nlat, nlon, device=self.device)
        with torch.no_grad():
            y_no_bias = conv_no_bias(x)
            y_bias = conv_bias(x)

        self.assertTrue(
            compare_tensors("zero bias equals no bias", y_bias, y_no_bias, atol=1e-6, rtol=1e-5, verbose=verbose)
        )

    @parameterized.expand(
        [
            # nlat, nlon, channels, num_groups
            [32, 64, 4, 1],
            [32, 64, 4, 2],
        ],
        skip_on_empty=True,
    )
    def test_spectral_bias_nonzero_changes_output(self, nlat, nlon, channels, num_groups, verbose=False):
        """A nonzero spectral_bias must change the output relative to zero bias."""
        set_seed(333)

        conv = SpectralConvS2(
            in_shape=(nlat, nlon),
            out_shape=(nlat, nlon),
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            grid_in="equiangular",
            grid_out="equiangular",
            bias=True,
        ).to(self.device)

        x = torch.randn(2, channels, nlat, nlon, device=self.device)

        with torch.no_grad():
            y_zero_bias = conv(x)
            # set a clearly nonzero bias and re-run
            conv.spectral_bias.data.fill_(1.0)
            y_nonzero_bias = conv(x)

        self.assertFalse(
            torch.allclose(y_zero_bias, y_nonzero_bias, atol=1e-6),
            msg="Expected output to change when spectral_bias is nonzero",
        )

    @parameterized.expand(
        [
            # nlat, nlon, channels, num_groups
            [32, 64, 4, 1],
            [32, 64, 4, 2],
        ],
        skip_on_empty=True,
    )
    def test_spectral_bias_gradient(self, nlat, nlon, channels, num_groups, verbose=False):
        """Gradients must flow back to spectral_bias after a backward pass."""
        set_seed(333)

        conv = SpectralConvS2(
            in_shape=(nlat, nlon),
            out_shape=(nlat, nlon),
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            grid_in="equiangular",
            grid_out="equiangular",
            bias=True,
        ).to(self.device)

        with torch.no_grad():
            # give the bias a nonzero value so it has an effect on the output
            conv.spectral_bias.data.fill_(0.1)

        x = torch.randn(2, channels, nlat, nlon, device=self.device)
        loss = conv(x).sum()
        loss.backward()

        self.assertIsNotNone(conv.spectral_bias.grad, msg="spectral_bias.grad should not be None")
        self.assertTrue(
            conv.spectral_bias.grad.abs().max().item() > 0.0,
            msg="spectral_bias.grad should contain non-zero entries",
        )


if __name__ == "__main__":
    unittest.main()
