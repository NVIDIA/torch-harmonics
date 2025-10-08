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

import os
from time import perf_counter_ns
import unittest
from parameterized import parameterized, parameterized_class
import math

import torch
from torch.library import opcheck
from torch_harmonics import DiscreteContinuousConvS2, DiscreteContinuousConvTransposeS2

from torch_harmonics.quadrature import _precompute_latitudes, _precompute_longitudes
from torch_harmonics.disco import cuda_kernels_is_available, optimized_kernels_is_available

from testutils import compare_tensors

if not optimized_kernels_is_available():
    print(f"Warning: Couldn't import optimized disco convolution kernels")


_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))

# perf thresholds
# CPU results normalized to 16 OpenMP threads,
# GPU results normalized to V100 16 GB GPU
# this is just to detect performance regressions, not for absolute performance
_perf_test_thresholds = {"cpu": {"fwd_ms": 100, "bwd_ms": 90}, 
                         "cuda": {"fwd_ms": 2, "bwd_ms": 3}}
_run_perf_tests = (os.getenv("TORCH_HARMONICS_RUN_PERF_TESTS", "0") == "1")


def _normalize_convolution_tensor_dense(psi, quad_weights, transpose_normalization=False, basis_norm_mode="none", merge_quadrature=False, eps=1e-9):
    """Discretely normalizes the convolution tensor."""

    kernel_size, nlat_out, nlon_out, nlat_in, nlon_in = psi.shape
    correction_factor = nlon_out / nlon_in

    if basis_norm_mode == "individual":
        if transpose_normalization:
            # the normalization is not quite symmetric due to the compressed way psi is stored in the main code
            # look at the normalization code in the actual implementation
            psi_norm = torch.sum(quad_weights.reshape(1, -1, 1, 1, 1) * psi[:, :, :1].abs(), dim=(1, 4), keepdim=True)
        else:
            psi_norm = torch.sum(quad_weights.reshape(1, 1, 1, -1, 1) * psi.abs(), dim=(3, 4), keepdim=True)

    elif basis_norm_mode == "mean":
        if transpose_normalization:
            # the normalization is not quite symmetric due to the compressed way psi is stored in the main code
            # look at the normalization code in the actual implementation
            psi_norm = torch.sum(quad_weights.reshape(1, -1, 1, 1, 1) * psi[:, :, :1].abs(), dim=(1, 4), keepdim=True)
            psi_norm = psi_norm.mean(dim=3, keepdim=True)
        else:
            psi_norm = torch.sum(quad_weights.reshape(1, 1, 1, -1, 1) * psi.abs(), dim=(3, 4), keepdim=True)
            psi_norm = psi_norm.mean(dim=1, keepdim=True)
    elif basis_norm_mode == "none":
        psi_norm = 1.0
    else:
        raise ValueError(f"Unknown basis normalization mode {basis_norm_mode}.")

    if transpose_normalization:
        if merge_quadrature:
            psi = quad_weights.reshape(1, -1, 1, 1, 1) * psi / correction_factor
    else:
        if merge_quadrature:
            psi = quad_weights.reshape(1, 1, 1, -1, 1) * psi

    return psi / (psi_norm + eps)


def _precompute_convolution_tensor_dense(
    in_shape,
    out_shape,
    filter_basis,
    grid_in="equiangular",
    grid_out="equiangular",
    theta_cutoff=0.01 * math.pi,
    theta_eps=1e-3,
    transpose_normalization=False,
    basis_norm_mode="none",
    merge_quadrature=False,
):
    """Helper routine to compute the convolution Tensor in a dense fashion."""
    assert len(in_shape) == 2
    assert len(out_shape) == 2

    kernel_size = filter_basis.kernel_size

    nlat_in, nlon_in = in_shape
    nlat_out, nlon_out = out_shape

    lats_in, win = _precompute_latitudes(nlat_in, grid=grid_in)
    lats_out, wout = _precompute_latitudes(nlat_out, grid=grid_out)

    # compute the phi differences.
    lons_in = _precompute_longitudes(nlon_in)
    lons_out = _precompute_longitudes(nlon_out)

    # effective theta cutoff if multiplied with a fudge factor to avoid aliasing with grid width (especially near poles)
    theta_cutoff_eff = (1.0 + theta_eps) * theta_cutoff

    # compute quadrature weights that will be merged into the Psi tensor
    if transpose_normalization:
        quad_weights = wout.reshape(-1, 1) / nlon_in / 2.0
    else:
        quad_weights = win.reshape(-1, 1) / nlon_in / 2.0

    # array for accumulating non-zero indices
    out = torch.zeros(kernel_size, nlat_out, nlon_out, nlat_in, nlon_in, dtype=torch.float64, device=lons_in.device)

    for t in range(nlat_out):
        for p in range(nlon_out):
            alpha = -lats_out[t]
            beta = lons_in - lons_out[p]
            gamma = lats_in.reshape(-1, 1)

            # compute latitude of the rotated position
            z = -torch.cos(beta) * torch.sin(alpha) * torch.sin(gamma) + torch.cos(alpha) * torch.cos(gamma)

            # compute cartesian coordinates of the rotated position
            x = torch.cos(alpha) * torch.cos(beta) * torch.sin(gamma) + torch.cos(gamma) * torch.sin(alpha)
            y = torch.sin(beta) * torch.sin(gamma) * torch.ones_like(alpha)

            # normalize instead of clipping to ensure correct range
            norm = torch.sqrt(x * x + y * y + z * z)
            x = x / norm
            y = y / norm
            z = z / norm

            # compute spherical coordinates
            theta = torch.arccos(z)
            phi = torch.arctan2(y, x)
            phi = torch.where(phi < 0.0, phi + 2 * torch.pi, phi)

            # find the indices where the rotated position falls into the support of the kernel
            iidx, vals = filter_basis.compute_support_vals(theta, phi, r_cutoff=theta_cutoff_eff)
            out[iidx[:, 0], t, p, iidx[:, 1], iidx[:, 2]] = vals

    # take care of normalization and cast to float
    out = _normalize_convolution_tensor_dense(
        out, quad_weights=quad_weights, transpose_normalization=transpose_normalization, basis_norm_mode=basis_norm_mode, merge_quadrature=merge_quadrature
    )
    out = out.to(dtype=torch.float32)

    return out


@parameterized_class(("device"), _devices)
class TestDiscreteContinuousConvolution(unittest.TestCase):
    """Test the discrete-continuous convolution module (CPU/CUDA if available)."""

    def setUp(self):
        torch.manual_seed(333)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(333)

    @parameterized.expand(
        [
            # regular convolution
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", False, 1e-4, False],
            [8, 4, 2, (16, 32), (8, 16), (3), "piecewise linear", "mean", "equiangular", "equiangular", False, 1e-4, False],
            [8, 4, 2, (24, 48), (12, 24), (3, 3), "piecewise linear", "mean", "equiangular", "equiangular", False, 1e-4, False],
            [8, 4, 2, (24, 48), (12, 24), (4, 3), "piecewise linear", "mean", "equiangular", "equiangular", False, 1e-4, False],
            [8, 4, 2, (24, 48), (12, 24), (2, 2), "morlet", "mean", "equiangular", "equiangular", False, 1e-4, False],
            [8, 4, 2, (24, 48), (12, 24), (2, 1), "morlet", "mean", "equiangular", "equiangular", False, 1e-4, False],
            [8, 4, 2, (24, 48), (12, 24), (3), "zernike", "mean", "equiangular", "equiangular", False, 1e-4, False],
            [8, 4, 2, (16, 24), (8, 8), (3), "piecewise linear", "mean", "equiangular", "equiangular", False, 1e-4, False],
            [8, 4, 2, (18, 36), (6, 12), (7), "piecewise linear", "mean", "equiangular", "equiangular", False, 1e-4, False],
            [8, 4, 2, (16, 32), (8, 16), (5), "piecewise linear", "mean", "equiangular", "legendre-gauss", False, 1e-4, False],
            [8, 4, 2, (16, 32), (8, 16), (5), "piecewise linear", "mean", "legendre-gauss", "equiangular", False, 1e-4, False],
            [8, 4, 2, (16, 32), (8, 16), (5), "piecewise linear", "mean", "legendre-gauss", "legendre-gauss", False, 1e-4, False],
            # transpose convolution
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", True, 1e-4, False],
            [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "mean", "equiangular", "equiangular", True, 1e-4, False],
            [8, 4, 2, (12, 24), (24, 48), (3, 3), "piecewise linear", "mean", "equiangular", "equiangular", True, 1e-4, False],
            [8, 4, 2, (12, 24), (24, 48), (4, 3), "piecewise linear", "mean", "equiangular", "equiangular", True, 1e-4, False],
            [8, 4, 2, (12, 24), (24, 48), (2, 2), "morlet", "mean", "equiangular", "equiangular", True, 1e-4, False],
            [8, 4, 2, (12, 24), (24, 48), (2, 1), "morlet", "mean", "equiangular", "equiangular", True, 1e-4, False],
            [8, 4, 2, (12, 24), (24, 48), (3), "zernike", "mean", "equiangular", "equiangular", True, 1e-4, False],
            [8, 4, 2, (8, 8), (16, 24), (3), "piecewise linear", "mean", "equiangular", "equiangular", True, 1e-4, False],
            [8, 4, 2, (6, 12), (18, 36), (7), "piecewise linear", "mean", "equiangular", "equiangular", True, 1e-4, False],
            [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "mean", "equiangular", "legendre-gauss", True, 1e-4, False],
            [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "mean", "legendre-gauss", "equiangular", True, 1e-4, False],
            [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "mean", "legendre-gauss", "legendre-gauss", True, 1e-4, False],
        ],
        skip_on_empty=True,
    )
    def test_sparse_against_dense(
        self,
        batch_size,
        in_channels,
        out_channels,
        in_shape,
        out_shape,
        kernel_shape,
        basis_type,
        basis_norm_mode,
        grid_in,
        grid_out,
        transpose,
        tol,
        verbose,
    ):

        if verbose:
            print(f"Testing DISCO convolution on {in_shape[0]}x{in_shape[1]} {grid_in} grid to {out_shape[0]}x{out_shape[1]} {grid_out} grid on {self.device.type} device")

        use_optimized_kernels = optimized_kernels_is_available()
        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            use_optimized_kernels = False
        
        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        if isinstance(kernel_shape, int):
            theta_cutoff = (kernel_shape + 1) * torch.pi / float(nlat_in - 1)
        else:
            theta_cutoff = (kernel_shape[0] + 1) * torch.pi / float(nlat_in - 1)

        Conv = DiscreteContinuousConvTransposeS2 if transpose else DiscreteContinuousConvS2
        conv = Conv(
            in_channels,
            out_channels,
            in_shape,
            out_shape,
            kernel_shape,
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            groups=1,
            grid_in=grid_in,
            grid_out=grid_out,
            bias=False,
            theta_cutoff=theta_cutoff,
            optimized_kernel=use_optimized_kernels,
        ).to(self.device)

        filter_basis = conv.filter_basis

        if transpose:
            psi_dense = _precompute_convolution_tensor_dense(
                out_shape,
                in_shape,
                filter_basis,
                grid_in=grid_out,
                grid_out=grid_in,
                theta_cutoff=theta_cutoff,
                transpose_normalization=transpose,
                basis_norm_mode=basis_norm_mode,
                merge_quadrature=True,
            ).to(self.device)

            psi = torch.sparse_coo_tensor(conv.psi_idx, conv.psi_vals, size=(conv.kernel_size, conv.nlat_in, conv.nlat_out * conv.nlon_out)).to_dense()

            self.assertTrue(compare_tensors(psi, psi_dense[:, :, 0].reshape(-1, nlat_in, nlat_out * nlon_out)))
        else:
            psi_dense = _precompute_convolution_tensor_dense(
                in_shape,
                out_shape,
                filter_basis,
                grid_in=grid_in,
                grid_out=grid_out,
                theta_cutoff=theta_cutoff,
                transpose_normalization=transpose,
                basis_norm_mode=basis_norm_mode,
                merge_quadrature=True,
            ).to(self.device)

            psi = torch.sparse_coo_tensor(conv.psi_idx, conv.psi_vals, size=(conv.kernel_size, conv.nlat_out, conv.nlat_in * conv.nlon_in)).to_dense()

            self.assertTrue(compare_tensors(psi, psi_dense[:, :, 0].reshape(-1, nlat_out, nlat_in * nlon_in)))

        # create a copy of the weight
        w_ref = torch.empty_like(conv.weight)
        with torch.no_grad():
            w_ref.copy_(conv.weight)
        w_ref = w_ref.reshape(-1, w_ref.shape[2], w_ref.shape[3])
        w_ref.requires_grad = True

        # create an input signal
        x = torch.randn(batch_size, in_channels, *in_shape, device=self.device)

        # FWD and BWD pass
        x.requires_grad = True
        y = conv(x)
        grad_input = torch.randn_like(y)
        y.backward(grad_input)
        x_grad = x.grad.clone()

        # perform the reference computation
        x_ref = x.clone().detach()
        x_ref.requires_grad = True
        if transpose:
            y_ref = torch.einsum("oif,biqr->bofqr", w_ref, x_ref)
            y_ref = torch.einsum("fqrtp,bofqr->botp", psi_dense, y_ref)
        else:
            y_ref = torch.einsum("ftpqr,bcqr->bcftp", psi_dense, x_ref)
            y_ref = torch.einsum("oif,biftp->botp", w_ref, y_ref)
        y_ref.backward(grad_input)
        x_ref_grad = x_ref.grad.clone()

        # compare results
        print(y, y_ref)
        self.assertTrue(compare_tensors(y, y_ref, rtol=tol, atol=tol, verbose=verbose))

        # compare
        self.assertTrue(compare_tensors(x_grad, x_ref_grad, rtol=tol, atol=tol, verbose=verbose))
        self.assertTrue(compare_tensors(conv.weight.grad, w_ref.grad.unsqueeze(0), rtol=tol, atol=tol, verbose=verbose))


    @parameterized.expand(
        [
            # regular convolution
            [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", False, 1e-4, False],
            [8, 4, 2, (41, 80), (41, 80), (2, 2), "morlet", "mean", "equiangular", "equiangular", False, 1e-4, False],
            [8, 4, 2, (41, 80), (41, 80), (2, 3), "morlet", "mean", "equiangular", "equiangular", False, 1e-4, False],
            [8, 4, 2, (41, 80), (41, 80), (3), "zernike", "mean", "equiangular", "equiangular", False, 1e-4, False],
            [8, 4, 2, (41, 80), (21, 40), (3), "piecewise linear", "mean", "equiangular", "equiangular", False, 1e-4, False],
            [8, 4, 2, (41, 80), (21, 40), (2, 2), "morlet", "mean", "equiangular", "equiangular", False, 1e-4, False],
            [8, 4, 2, (41, 80), (21, 40), (2, 1), "morlet", "mean", "equiangular", "equiangular", False, 1e-4, False],
            [8, 4, 2, (41, 80), (21, 40), (3), "zernike", "mean", "equiangular", "equiangular", False, 1e-4, False],
            # transpose convolution
            [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", True, 1e-4, False],
            [8, 4, 2, (41, 80), (41, 80), (2, 2), "morlet", "mean", "equiangular", "equiangular", True, 1e-4, False],
            [8, 4, 2, (41, 80), (41, 80), (2, 3), "morlet", "mean", "equiangular", "equiangular", True, 1e-4, False],
            [8, 4, 2, (41, 80), (41, 80), (3), "zernike", "mean", "equiangular", "equiangular", True, 1e-4, False],
            [8, 4, 2, (21, 40), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", True, 1e-4, False],
            [8, 4, 2, (21, 40), (41, 80), (2, 2), "morlet", "mean", "equiangular", "equiangular", True, 1e-4, False],
            [8, 4, 2, (21, 40), (41, 80), (2, 1), "morlet", "mean", "equiangular", "equiangular", True, 1e-4, False],
            [8, 4, 2, (21, 40), (41, 80), (3), "zernike", "mean", "equiangular", "equiangular", True, 1e-4, False],
        ],
        skip_on_empty=True,
    )
    @unittest.skipUnless((optimized_kernels_is_available()), "skipping test because optimized kernels are not available")
    def test_optimized_against_torch(
        self,
        batch_size,
        in_channels,
        out_channels,
        in_shape,
        out_shape,
        kernel_shape,
        basis_type,
        basis_norm_mode,
        grid_in,
        grid_out,
        transpose,
        tol,
        verbose,
    ):
        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

        if verbose:
            print(f"Testing DISCO convolution on {in_shape[0]}x{in_shape[1]} {grid_in} grid to {out_shape[0]}x{out_shape[1]} {grid_out} grid on {self.device.type} device")
        
        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        if isinstance(kernel_shape, int):
            theta_cutoff = (kernel_shape + 1) * torch.pi / float(nlat_in - 1)
        else:
            theta_cutoff = (kernel_shape[0] + 1) * torch.pi / float(nlat_in - 1)

        Conv = DiscreteContinuousConvTransposeS2 if transpose else DiscreteContinuousConvS2

        conv_naive = Conv(
            in_channels,
            out_channels,
            in_shape,
            out_shape,
            kernel_shape,
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            groups=1,
            grid_in=grid_in,
            grid_out=grid_out,
            bias=False,
            theta_cutoff=theta_cutoff,
            optimized_kernel=False,
        ).to(self.device)
    
        conv_opt = Conv(
            in_channels,
            out_channels,
            in_shape,
            out_shape,
            kernel_shape,
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            groups=1,
            grid_in=grid_in,
            grid_out=grid_out,
            bias=False,
            theta_cutoff=theta_cutoff,
            optimized_kernel=True,
        ).to(self.device)

        # create a copy of the weight
        with torch.no_grad():
            conv_naive.weight.copy_(conv_opt.weight)

        # create an input signal
        inp = torch.randn(batch_size, in_channels, *in_shape, device=self.device)

        # FWD and BWD pass
        inp.requires_grad = True
        out_naive = conv_naive(inp)
        grad_input = torch.randn_like(out_naive)
        out_naive.backward(grad_input)
        inp_grad_naive = inp.grad.clone()

        # perform the reference computation
        inp.grad = None
        out_opt = conv_opt(inp)
        out_opt.backward(grad_input)
        inp_grad_opt = inp.grad.clone()

        # compare results
        self.assertTrue(compare_tensors(out_naive, out_opt, rtol=tol, atol=tol, verbose=verbose))

        # compare
        self.assertTrue(compare_tensors(inp_grad_naive, inp_grad_opt, rtol=tol, atol=tol, verbose=verbose))
        self.assertTrue(compare_tensors(conv_naive.weight.grad, conv_opt.weight.grad, rtol=tol, atol=tol, verbose=verbose))


    @parameterized.expand(
        [
            # [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", False, 1e-4, False],
            # [8, 4, 2, (16, 32), (8, 16), (5), "piecewise linear", "mean", "legendre-gauss", "legendre-gauss", False, 1e-4, False],
            # [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", True, 1e-4, False],
            # [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "mean", "legendre-gauss", "legendre-gauss", True, 1e-4, False],
        ],
        skip_on_empty=True,
    )
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_device_instantiation(self, batch_size, in_channels, out_channels, in_shape, out_shape, kernel_shape, basis_type, basis_norm_mode, grid_in, grid_out, transpose, tol, verbose):

        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        if isinstance(kernel_shape, int):
            theta_cutoff = (kernel_shape + 1) * torch.pi / float(nlat_in - 1)
        else:
            theta_cutoff = (kernel_shape[0] + 1) * torch.pi / float(nlat_in - 1)

        # get handle
        Conv = DiscreteContinuousConvTransposeS2 if transpose else DiscreteContinuousConvS2

        # init on cpu
        conv_host = Conv(
            in_channels,
            out_channels,
            in_shape,
            out_shape,
            kernel_shape,
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            groups=1,
            grid_in=grid_in,
            grid_out=grid_out,
            bias=False,
            theta_cutoff=theta_cutoff,
        )

        #torch.set_default_device(self.device)
        with torch.device(self.device):
            conv_device = Conv(
                in_channels,
                out_channels,
                in_shape,
                out_shape,
                kernel_shape,
                basis_type=basis_type,
                basis_norm_mode=basis_norm_mode,
                groups=1,
                grid_in=grid_in,
                grid_out=grid_out,
                bias=False,
                theta_cutoff=theta_cutoff,
            )

        # since we specified the device specifier everywhere, it should always
        # use the cpu and it should be the same everywhere
        self.assertTrue(torch.allclose(conv_host.psi_col_idx.cpu(), conv_device.psi_col_idx.cpu()))
        self.assertTrue(torch.allclose(conv_host.psi_row_idx.cpu(), conv_device.psi_row_idx.cpu()))
        self.assertTrue(torch.allclose(conv_host.psi_roff_idx.cpu(), conv_device.psi_roff_idx.cpu()))
        self.assertTrue(torch.allclose(conv_host.psi_vals.cpu(), conv_device.psi_vals.cpu()))
        self.assertTrue(torch.allclose(conv_host.psi_idx.cpu(), conv_device.psi_idx.cpu()))


    @parameterized.expand(
        [
            # [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", False, 1e-4, False],
            # [8, 4, 2, (16, 32), (8,  16), (3), "piecewise linear", "mean", "equiangular", "equiangular", False, 1e-4, False],
            # [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", True, 1e-4, False],
            # [8, 4, 2, (8,  16), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", True, 1e-4, False],

        ], 
        skip_on_empty=True,
    )
    @unittest.skipUnless((optimized_kernels_is_available()), "skipping test because optimized kernels are not available")
    def test_optimized_pt2_compatibility(
        self,
        batch_size,
        in_channels,
        out_channels,
        in_shape,
        out_shape,
        kernel_shape,
        basis_type,
        basis_norm_mode,
        grid_in,
        grid_out,
        transpose,
        tol,
        verbose,
    ):  
        """Tests whether the optimized kernels are PyTorch 2 compatible"""

        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping GPU test because CUDA kernels are not available")
        
        if verbose:
            print(f"Testing DISCO convolution on {in_shape[0]}x{in_shape[1]} {grid_in} grid to {out_shape[0]}x{out_shape[1]} {grid_out} grid on {self.device.type} device")
        
        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        if isinstance(kernel_shape, int):
            theta_cutoff = (kernel_shape + 1) * torch.pi / float(nlat_in - 1)
        else:
            theta_cutoff = (kernel_shape[0] + 1) * torch.pi / float(nlat_in - 1)

        Conv = DiscreteContinuousConvTransposeS2 if transpose else DiscreteContinuousConvS2
        conv = Conv(
            in_channels,
            out_channels,
            in_shape,
            out_shape,
            kernel_shape,
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            groups=1,
            grid_in=grid_in,
            grid_out=grid_out,
            bias=False,
            theta_cutoff=theta_cutoff,
        ).to(self.device)

        # forward test
        if not transpose:
            inp = torch.randn(batch_size, in_channels, *in_shape, device=self.device)
        else:
            inp = torch.randn(batch_size, conv.kernel_size, in_channels, *in_shape, device=self.device)

        test_inputs = (inp, conv.psi_roff_idx, conv.psi_ker_idx, conv.psi_row_idx, conv.psi_col_idx, conv.psi_vals, 
                       conv.kernel_size, conv.nlat_out, conv.nlon_out)

        if not transpose:
            opcheck(torch.ops.disco_kernels._disco_s2_contraction_optimized, test_inputs)
        else:
            opcheck(torch.ops.disco_kernels._disco_s2_transpose_contraction_optimized, test_inputs)

        # if a test fails, those help to disambiguate the error
        # schema
        #opcheck(torch.ops.disco_kernels._disco_s2_contraction_optimized, test_inputs, test_utils="test_schema")
        # fake tensor
        #opcheck(torch.ops.disco_kernels._disco_s2_contraction_optimized, test_inputs, test_utils="test_faketensor")
        # test AOT stuff
        # this is expected to fail if the output shapes are dependent on input shapes (which is the case for DISCO)
        #opcheck(torch.ops.disco_kernels._disco_s2_contraction_optimized, test_inputs, test_utils="test_aot_dispatch_static")
        # this one should pass
        #opcheck(torch.ops.disco_kernels._disco_s2_contraction_optimized, test_inputs, test_utils="test_aot_dispatch_dynamic")


    @parameterized.expand(
        [
            # [8, 4, 2, (91, 180), (91, 180), (3), "piecewise linear", "mean", "equiangular", "equiangular", False, 1e-4],
        ],
        skip_on_empty=True,
    )
    @unittest.skipUnless(optimized_kernels_is_available() and _run_perf_tests, "skipping performance test because optimized kernels are not available or perf tests are disabled")
    def test_perf(self, batch_size, in_channels, out_channels, in_shape, out_shape, kernel_shape, basis_type, basis_norm_mode, grid_in, grid_out, transpose, tol, verbose=True):
        
        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")
        
        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        if isinstance(kernel_shape, int):
            theta_cutoff = (kernel_shape + 1) * torch.pi / float(nlat_in - 1)
        else:
            theta_cutoff = (kernel_shape[0] + 1) * torch.pi / float(nlat_in - 1)

        # get handle
        Conv = DiscreteContinuousConvTransposeS2 if transpose else DiscreteContinuousConvS2

        # init on cpu
        conv_optimized = Conv(
            in_channels,
            out_channels,
            in_shape,
            out_shape,
            kernel_shape,
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            groups=1,
            grid_in=grid_in,
            grid_out=grid_out,
            bias=True,
            theta_cutoff=theta_cutoff,
            optimized_kernel=True,
        ).to(self.device)

        # random weights
        with torch.no_grad():
            conv_optimized.weight.normal_()
            conv_optimized.bias.normal_()

        # create an input signal
        inp = torch.randn(batch_size, in_channels, *in_shape, device=self.device)
        inp.requires_grad = True

        # forward test
        # warmup
        for i in range(2):
            out_optimized = conv_optimized(inp)

        # start timer
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = perf_counter_ns()
        out_optimized = conv_optimized(inp)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        end = perf_counter_ns()
        duration = (end - start) / 1e6
        if verbose:
            print(f"Forward execution time on device {self.device.type}: {duration:.2f} ms")
        self.assertTrue(duration <= _perf_test_thresholds[self.device.type]["fwd_ms"])

        # backward test
        out_optimized = conv_optimized(inp)
        out_grad = torch.randn(out_optimized.shape, dtype=torch.float32, device=self.device)
        
        # warmup
        for _ in range(2):
            out_optimized.backward(out_grad, retain_graph=True)

        # start timer
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = perf_counter_ns()
        out_optimized.backward(out_grad)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        end = perf_counter_ns()
        duration = (end - start) / 1e6
        if verbose:
            print(f"Backward execution time on device {self.device.type}: {duration:.2f} ms")
        self.assertTrue(duration <= _perf_test_thresholds[self.device.type]["bwd_ms"])

if __name__ == "__main__":
    unittest.main()
