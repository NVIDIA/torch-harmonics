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
from functools import partial
import math
import numpy as np
import torch
from torch.autograd import gradcheck
from torch_harmonics import quadrature, DiscreteContinuousConvS2, DiscreteContinuousConvTransposeS2

from torch_harmonics.quadrature import _precompute_grid, _precompute_latitudes, _precompute_longitudes


def _normalize_convolution_tensor_dense(psi, quad_weights, transpose_normalization=False, basis_norm_mode="none", merge_quadrature=False, eps=1e-9):
    """
    Discretely normalizes the convolution tensor.
    """

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
    """
    Helper routine to compute the convolution Tensor in a dense fashion
    """

    assert len(in_shape) == 2
    assert len(out_shape) == 2

    kernel_size = filter_basis.kernel_size

    nlat_in, nlon_in = in_shape
    nlat_out, nlon_out = out_shape

    lats_in, win = quadrature._precompute_latitudes(nlat_in, grid=grid_in)
    lats_out, wout = quadrature._precompute_latitudes(nlat_out, grid=grid_out)

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
    out = torch.zeros(kernel_size, nlat_out, nlon_out, nlat_in, nlon_in, dtype=torch.float64)

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


class TestDiscreteContinuousConvolution(unittest.TestCase):
    def setUp(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device.index)
            torch.cuda.manual_seed(333)
        else:
            self.device = torch.device("cpu")
        torch.manual_seed(333)

    @parameterized.expand(
        [
            # regular convolution
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", False, 1e-4, False],
            [8, 4, 2, (16, 32), (8, 16), (3), "piecewise linear", "mean", "equiangular", "equiangular", False, 1e-4, False],
            [8, 4, 2, (24, 48), (12, 24), (3, 3), "piecewise linear", "mean", "equiangular", "equiangular", False, 1e-4, False],
            [8, 4, 2, (24, 48), (12, 24), (4, 3), "piecewise linear", "mean", "equiangular", "equiangular", False, 1e-4, False],
            [8, 4, 2, (24, 48), (12, 24), (2, 2), "morlet", "mean", "equiangular", "equiangular", False, 1e-4, False],
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
            [8, 4, 2, (12, 24), (24, 48), (3), "zernike", "mean", "equiangular", "equiangular", True, 1e-4, False],
            [8, 4, 2, (8, 8), (16, 24), (3), "piecewise linear", "mean", "equiangular", "equiangular", True, 1e-4, False],
            [8, 4, 2, (6, 12), (18, 36), (7), "piecewise linear", "mean", "equiangular", "equiangular", True, 1e-4, False],
            [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "mean", "equiangular", "legendre-gauss", True, 1e-4, False],
            [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "mean", "legendre-gauss", "equiangular", True, 1e-4, False],
            [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "mean", "legendre-gauss", "legendre-gauss", True, 1e-4, False],
        ]
    )
    def test_disco_convolution(
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

            self.assertTrue(torch.allclose(psi, psi_dense[:, :, 0].reshape(-1, nlat_in, nlat_out * nlon_out)))
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

            self.assertTrue(torch.allclose(psi, psi_dense[:, :, 0].reshape(-1, nlat_out, nlat_in * nlon_in)))

        # create a copy of the weight
        w_ref = torch.empty_like(conv.weight)
        with torch.no_grad():
            w_ref.copy_(conv.weight)
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
        self.assertTrue(torch.allclose(y, y_ref, rtol=tol, atol=tol))

        # compare
        self.assertTrue(torch.allclose(x_grad, x_ref_grad, rtol=tol, atol=tol))
        self.assertTrue(torch.allclose(conv.weight.grad, w_ref.grad, rtol=tol, atol=tol))


if __name__ == "__main__":
    unittest.main()
