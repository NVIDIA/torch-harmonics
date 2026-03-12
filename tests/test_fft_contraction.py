# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest
from parameterized import parameterized

import torch
from torch_harmonics import DiscreteContinuousConvS2, DiscreteContinuousConvTransposeS2

from testutils import disable_tf32, set_seed


class TestFFTContraction(unittest.TestCase):
    """Test FFT-based contraction against the loop-and-roll torch reference."""

    @parameterized.expand(
        [
            # same resolution
            [2, 4, 2, (16, 32), (16, 32), 3, "piecewise linear", "mean", "equiangular", "equiangular", 1e-5],
            # downsampling
            [2, 4, 2, (16, 32), (8, 16), 3, "piecewise linear", "mean", "equiangular", "equiangular", 1e-5],
            [2, 4, 2, (24, 48), (12, 24), (3, 3), "piecewise linear", "mean", "equiangular", "equiangular", 1e-5],
            # different basis types
            [2, 4, 2, (24, 48), (12, 24), (2, 2), "morlet", "mean", "equiangular", "equiangular", 1e-5],
            [2, 4, 2, (24, 48), (12, 24), 3, "zernike", "mean", "equiangular", "equiangular", 1e-5],
            # different grids
            [2, 4, 2, (16, 32), (8, 16), 5, "piecewise linear", "mean", "equiangular", "legendre-gauss", 1e-5],
            [2, 4, 2, (16, 32), (8, 16), 5, "piecewise linear", "mean", "legendre-gauss", "equiangular", 1e-5],
            # 3x stride
            [2, 4, 2, (18, 36), (6, 12), 7, "piecewise linear", "mean", "equiangular", "equiangular", 1e-5],
        ],
        skip_on_empty=True,
    )
    def test_fft_forward_against_torch(
        self, batch_size, in_channels, out_channels, in_shape, out_shape,
        kernel_shape, basis_type, basis_norm_mode, grid_in, grid_out, atol,
    ):
        disable_tf32()
        set_seed(42)

        nlat_in = in_shape[0]
        if isinstance(kernel_shape, int):
            theta_cutoff = (kernel_shape + 1) * torch.pi / float(nlat_in - 1)
        else:
            theta_cutoff = (kernel_shape[0] + 1) * torch.pi / float(nlat_in - 1)

        conv_ref = DiscreteContinuousConvS2(
            in_channels, out_channels, in_shape, out_shape, kernel_shape,
            basis_type=basis_type, basis_norm_mode=basis_norm_mode,
            grid_in=grid_in, grid_out=grid_out, bias=False,
            theta_cutoff=theta_cutoff, optimized_kernel=False, use_fft_contraction=False,
        )
        conv_fft = DiscreteContinuousConvS2(
            in_channels, out_channels, in_shape, out_shape, kernel_shape,
            basis_type=basis_type, basis_norm_mode=basis_norm_mode,
            grid_in=grid_in, grid_out=grid_out, bias=False,
            theta_cutoff=theta_cutoff, optimized_kernel=False, use_fft_contraction=True,
        )
        with torch.no_grad():
            conv_fft.weight.copy_(conv_ref.weight)

        x = torch.randn(batch_size, in_channels, *in_shape)
        y_ref = conv_ref(x)
        y_fft = conv_fft(x)

        self.assertEqual(y_ref.shape, y_fft.shape)
        torch.testing.assert_close(y_ref, y_fft, atol=atol, rtol=1e-4)

    @parameterized.expand(
        [
            # same resolution
            [2, 4, 2, (16, 32), (16, 32), 3, "piecewise linear", "mean", "equiangular", "equiangular", 1e-5],
            # upsampling
            [2, 4, 2, (8, 16), (16, 32), 5, "piecewise linear", "mean", "equiangular", "equiangular", 1e-5],
            [2, 4, 2, (12, 24), (24, 48), (3, 3), "piecewise linear", "mean", "equiangular", "equiangular", 1e-5],
            # different basis types
            [2, 4, 2, (12, 24), (24, 48), (2, 2), "morlet", "mean", "equiangular", "equiangular", 1e-5],
            [2, 4, 2, (12, 24), (24, 48), 3, "zernike", "mean", "equiangular", "equiangular", 1e-5],
            # different grids
            [2, 4, 2, (8, 16), (16, 32), 5, "piecewise linear", "mean", "equiangular", "legendre-gauss", 1e-5],
            [2, 4, 2, (8, 16), (16, 32), 5, "piecewise linear", "mean", "legendre-gauss", "equiangular", 1e-5],
            # 3x stride
            [2, 4, 2, (6, 12), (18, 36), 7, "piecewise linear", "mean", "equiangular", "equiangular", 1e-5],
        ],
        skip_on_empty=True,
    )
    def test_fft_transpose_against_torch(
        self, batch_size, in_channels, out_channels, in_shape, out_shape,
        kernel_shape, basis_type, basis_norm_mode, grid_in, grid_out, atol,
    ):
        disable_tf32()
        set_seed(42)

        nlat_in = in_shape[0]
        if isinstance(kernel_shape, int):
            theta_cutoff = (kernel_shape + 1) * torch.pi / float(nlat_in - 1)
        else:
            theta_cutoff = (kernel_shape[0] + 1) * torch.pi / float(nlat_in - 1)

        tconv_ref = DiscreteContinuousConvTransposeS2(
            in_channels, out_channels, in_shape, out_shape, kernel_shape,
            basis_type=basis_type, basis_norm_mode=basis_norm_mode,
            grid_in=grid_in, grid_out=grid_out, bias=False,
            theta_cutoff=theta_cutoff, optimized_kernel=False, use_fft_contraction=False,
        )
        tconv_fft = DiscreteContinuousConvTransposeS2(
            in_channels, out_channels, in_shape, out_shape, kernel_shape,
            basis_type=basis_type, basis_norm_mode=basis_norm_mode,
            grid_in=grid_in, grid_out=grid_out, bias=False,
            theta_cutoff=theta_cutoff, optimized_kernel=False, use_fft_contraction=True,
        )
        with torch.no_grad():
            tconv_fft.weight.copy_(tconv_ref.weight)

        x = torch.randn(batch_size, in_channels, *in_shape)
        y_ref = tconv_ref(x)
        y_fft = tconv_fft(x)

        self.assertEqual(y_ref.shape, y_fft.shape)
        torch.testing.assert_close(y_ref, y_fft, atol=atol, rtol=1e-4)

    def test_fft_forward_gradient(self):
        """Test that gradients flow correctly through the FFT path."""
        disable_tf32()
        set_seed(42)

        conv = DiscreteContinuousConvS2(
            4, 2, (16, 32), (8, 16), 3, bias=False,
            optimized_kernel=False, use_fft_contraction=True,
        )

        x = torch.randn(2, 4, 16, 32, requires_grad=True)
        y = conv(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(conv.weight.grad)
        self.assertFalse(torch.all(x.grad == 0))
        self.assertFalse(torch.all(conv.weight.grad == 0))

    def test_fft_transpose_gradient(self):
        """Test that gradients flow correctly through the FFT transpose path."""
        disable_tf32()
        set_seed(42)

        tconv = DiscreteContinuousConvTransposeS2(
            4, 2, (8, 16), (16, 32), 3, bias=False,
            optimized_kernel=False, use_fft_contraction=True,
        )

        x = torch.randn(2, 4, 8, 16, requires_grad=True)
        y = tconv(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(tconv.weight.grad)
        self.assertFalse(torch.all(x.grad == 0))
        self.assertFalse(torch.all(tconv.weight.grad == 0))


if __name__ == "__main__":
    unittest.main()
