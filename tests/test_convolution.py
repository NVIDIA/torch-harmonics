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

from torch_harmonics.quadrature import precompute_latitudes, precompute_longitudes
from torch_harmonics.disco import cuda_kernels_is_available, optimized_kernels_is_available
from torch_harmonics.disco.convolution import (
    _precompute_convolution_tensor_s2,
    _normalize_convolution_tensor_s2,
    _normalize_convolution_tensor_s2_legacy,
)
from torch_harmonics.filter_basis import get_filter_basis
from disco_helpers import preprocess_psi

from testutils import disable_tf32, set_seed, compare_tensors, maybe_autocast

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


def _normalize_convolution_tensor_dense(
    psi,
    quad_weights,
    transpose_normalization=False,
    basis_norm_mode="none",
    merge_quadrature=False,
    isotropic_mask=None,
    theta_cutoff=None,
    in_support=None,
    eps=1e-9,
):
    """Discretely normalizes the convolution tensor.

    Mirrors the normalization logic in _normalize_convolution_tensor_s2
    for all supported normalization modes.
    """

    kernel_size, nlat_out, nlon_out, nlat_in, nlon_in = psi.shape
    correction_factor = nlon_out / nlon_in

    if transpose_normalization:
        n_olat = nlat_in
    else:
        n_olat = nlat_out

    bias_arr = torch.zeros(kernel_size, n_olat, dtype=psi.dtype, device=psi.device)
    scale_arr = torch.zeros(kernel_size, n_olat, dtype=psi.dtype, device=psi.device)
    support_arr = torch.zeros(kernel_size, n_olat, dtype=psi.dtype, device=psi.device)

    for ik in range(kernel_size):
        for ilat in range(n_olat):
            if transpose_normalization:
                entries = psi[ik, :, 0, ilat, :]
                q = quad_weights[:nlat_out, 0].unsqueeze(1).expand_as(entries)
                smask = in_support[ik, :, 0, ilat, :] if in_support is not None else (entries.abs() > 0)
            else:
                entries = psi[ik, ilat, 0, :, :]
                q = quad_weights[:nlat_in, 0].unsqueeze(1).expand_as(entries)
                smask = in_support[ik, ilat, 0, :, :] if in_support is not None else (entries.abs() > 0)

            q_masked = q * smask
            support_arr[ik, ilat] = q_masked.sum()

            is_isotropic = isotropic_mask[ik] if isotropic_mask is not None else (ik == 0)
            if basis_norm_mode == "modal" and not is_isotropic and support_arr[ik, ilat].abs() > eps:
                bias_arr[ik, ilat] = (entries * q_masked).sum() / support_arr[ik, ilat]

            scale_arr[ik, ilat] = ((entries - bias_arr[ik, ilat]).abs() * q_masked).sum()

    # The sparse implementation stores one longitude slice and reuses it for all
    # output longitudes via rolling during contraction. We mirror this: normalize
    # only the r=0 slice, then fill other slices with cyclic shifts. Normalizing
    # each slice independently would amplify floating-point noise at near-zero
    # entries (e.g. anisotropic modes at the poles where scale ≈ 0).
    pscale = nlon_in // nlon_out

    # precompute the per-ik mean for "mean" mode so we don't rely on Python function-scope
    # reuse of b/s across ilat iterations inside the loop below
    if basis_norm_mode == "mean":
        bias_per_ik = bias_arr.mean(dim=1)
        scale_per_ik = scale_arr.mean(dim=1)

    # precompute the "geometric" scalar once; it's ik/ilat-independent
    if basis_norm_mode == "geometric":
        geometric_scale = (1.0 - math.cos(theta_cutoff)) / 2.0 / 2.0

    for ik in range(kernel_size):
        for ilat in range(n_olat):
            if basis_norm_mode in ["nodal", "modal"]:
                b = bias_arr[ik, ilat]
                s = scale_arr[ik, ilat]
            elif basis_norm_mode == "mean":
                b = bias_per_ik[ik]
                s = scale_per_ik[ik]
            elif basis_norm_mode == "support":
                b = 0.0
                s = support_arr[ik, ilat]
            elif basis_norm_mode == "geometric":
                b = 0.0
                s = geometric_scale
            elif basis_norm_mode == "none":
                b = 0.0
                s = 1.0
            else:
                raise ValueError(f"Unknown basis normalization mode {basis_norm_mode}.")

            if transpose_normalization:
                slc0 = psi[ik, :, 0, ilat, :]
                mask0 = in_support[ik, :, 0, ilat, :] if in_support is not None else (slc0 != 0)
                psi[ik, :, 0, ilat, :] = torch.where(mask0, (slc0 - b) / max(s, eps), slc0)
                for r in range(1, nlon_out):
                    psi[ik, :, r, ilat, :] = torch.roll(psi[ik, :, 0, ilat, :], r * pscale, dims=-1)
            else:
                slc0 = psi[ik, ilat, 0, :, :]
                mask0 = in_support[ik, ilat, 0, :, :] if in_support is not None else (slc0 != 0)
                psi[ik, ilat, 0, :, :] = torch.where(mask0, (slc0 - b) / max(s, eps), slc0)
                for r in range(1, nlon_out):
                    psi[ik, ilat, r, :, :] = torch.roll(psi[ik, ilat, 0, :, :], r * pscale, dims=-1)

    if transpose_normalization:
        if merge_quadrature:
            psi = quad_weights.reshape(1, -1, 1, 1, 1) * psi / correction_factor
    else:
        if merge_quadrature:
            psi = quad_weights.reshape(1, 1, 1, -1, 1) * psi

    return psi


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

    lats_in, win = precompute_latitudes(nlat_in, grid=grid_in)
    lats_out, wout = precompute_latitudes(nlat_out, grid=grid_out)

    # compute the phi differences.
    lons_in = precompute_longitudes(nlon_in)
    lons_out = precompute_longitudes(nlon_out)

    # effective theta cutoff if multiplied with a fudge factor to avoid aliasing with grid width (especially near poles)
    theta_cutoff_eff = (1.0 + theta_eps) * theta_cutoff

    # compute quadrature weights that will be merged into the Psi tensor
    if transpose_normalization:
        quad_weights = wout.reshape(-1, 1) / nlon_in / 2.0
    else:
        quad_weights = win.reshape(-1, 1) / nlon_in / 2.0

    # array for accumulating non-zero indices and tracking filter support
    out = torch.zeros(kernel_size, nlat_out, nlon_out, nlat_in, nlon_in, dtype=torch.float64, device=lons_in.device)
    in_support = torch.zeros_like(out, dtype=torch.bool)

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
            in_support[iidx[:, 0], t, p, iidx[:, 1], iidx[:, 2]] = True

    # take care of normalization
    out = _normalize_convolution_tensor_dense(
        out,
        quad_weights=quad_weights,
        transpose_normalization=transpose_normalization,
        basis_norm_mode=basis_norm_mode,
        merge_quadrature=merge_quadrature,
        isotropic_mask=filter_basis.isotropic_mask,
        theta_cutoff=theta_cutoff,
        in_support=in_support,
    )

    return out


@parameterized_class(("device"), _devices)
class TestDiscreteContinuousConvolution(unittest.TestCase):
    """Test the discrete-continuous convolution module (CPU/CUDA if available)."""

    def setUp(self):
        disable_tf32()

    @parameterized.expand(
        [
            # harmonic
            [(16, 32), (16, 32), (1, 1), "harmonic", "mean", "equiangular", "equiangular"],
            [(16, 32), (16, 32), (3, 3), "harmonic", "mean", "equiangular", "equiangular"],
            [(17, 32), (17, 32), (3, 3), "harmonic", "mean", "equiangular", "equiangular"],
            [(16, 32), (16, 32), (3, 4), "harmonic", "mean", "equiangular", "equiangular"],
            [(16, 32), (16, 32), (3, 2), "harmonic", "mean", "equiangular", "equiangular"],
            [(16, 32), (8, 16), (3, 3), "harmonic", "mean", "equiangular", "equiangular"],
            [(16, 32), (8, 16), (3, 4), "harmonic", "mean", "equiangular", "equiangular"],
            # zernike
            [(16, 32), (16, 32), (1), "zernike", "mean", "equiangular", "equiangular"],
            [(16, 32), (16, 32), (3), "zernike", "mean", "equiangular", "equiangular"],
            [(17, 32), (17, 32), (3), "zernike", "mean", "equiangular", "equiangular"],
            [(16, 32), (8, 16), (3), "zernike", "mean", "equiangular", "equiangular"],
            # fourier-bessel
            [(16, 32), (16, 32), (3, 3), "fourier-bessel", "mean", "equiangular", "equiangular"],
            [(17, 32), (17, 32), (3, 3), "fourier-bessel", "mean", "equiangular", "equiangular"],
            [(16, 32), (8, 16), (3, 3), "fourier-bessel", "mean", "equiangular", "equiangular"],
            # exercise each normalization mode at least once
            [(16, 32), (16, 32), (3, 3), "harmonic", "nodal", "equiangular", "equiangular"],
            [(16, 32), (16, 32), (3, 3), "harmonic", "modal", "equiangular", "equiangular"],
            [(16, 32), (16, 32), (3, 3), "harmonic", "support", "equiangular", "equiangular"],
            [(16, 32), (16, 32), (3, 3), "harmonic", "geometric", "equiangular", "equiangular"],
            [(16, 32), (16, 32), (3, 3), "harmonic", "none", "equiangular", "equiangular"],
            # mixed grid
            [(16, 32), (8, 16), (3, 3), "harmonic", "mean", "legendre-gauss", "equiangular"],
            [(16, 32), (8, 16), (3, 3), "harmonic", "mean", "equiangular", "legendre-gauss"],
        ],
        skip_on_empty=True,
    )
    def test_convolution_tensor_integrity(self, in_shape, out_shape, kernel_shape, basis_type, basis_norm_mode, grid_in, grid_out, verbose=False):
        """Structural invariants of the sparse psi datastructure after precompute + preprocess_psi.

        Note: intentionally excludes the "piecewise linear" basis, whose per-kernel radial support
        yields non-uniform (row, col) sets across kernel indices. The remaining bases share a
        full-disk support across all kernel basis functions and therefore satisfy the invariants
        the optimized DISCO kernel relies on.
        """

        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        filter_basis = get_filter_basis(kernel_shape=kernel_shape, basis_type=basis_type)

        theta_cutoff = torch.pi / float(nlat_out - 1)

        idx, vals, _ = _precompute_convolution_tensor_s2(
            in_shape=in_shape,
            out_shape=out_shape,
            filter_basis=filter_basis,
            grid_in=grid_in,
            grid_out=grid_out,
            theta_cutoff=theta_cutoff,
            transpose_normalization=False,
            basis_norm_mode=basis_norm_mode,
            merge_quadrature=True,
        )

        ker_idx = idx[0, ...].contiguous()
        row_idx = idx[1, ...].contiguous()
        col_idx = idx[2, ...].contiguous()
        vals = vals.contiguous()

        # sort + row offsets (preprocess_psi mutates ker/row/col/vals in place)
        roff_idx = preprocess_psi(filter_basis.kernel_size, nlat_out, ker_idx, row_idx, col_idx, vals).contiguous()

        # 1) shape consistency
        self.assertEqual(ker_idx.shape[0], row_idx.shape[0])
        self.assertEqual(ker_idx.shape[0], col_idx.shape[0])
        self.assertEqual(ker_idx.shape[0], vals.shape[0])

        # 2) roff_idx covers every (kernel, output-latitude) row exactly once
        self.assertEqual(roff_idx.shape[0] - 1, filter_basis.kernel_size * nlat_out)

        # 3) same number of nnz per kernel basis function
        _, counts = torch.unique(ker_idx, return_counts=True)
        self.assertTrue(torch.all(counts == counts[0]), f"multiplicity in ker_idx is not uniform: counts={counts.tolist()}")

        # 4) same (row, col) support pattern across all kernel basis functions
        row_idx_ref = row_idx[ker_idx == 0]
        col_idx_ref = col_idx[ker_idx == 0]
        for k in range(1, filter_basis.kernel_size):
            self.assertTrue(torch.equal(row_idx_ref, row_idx[ker_idx == k]), f"row_idx differs for kernel index {k}")
            self.assertTrue(torch.equal(col_idx_ref, col_idx[ker_idx == k]), f"col_idx differs for kernel index {k}")

        if verbose:
            print(f"\nintegrity OK: nnz={ker_idx.shape[0]}, per-kernel={counts[0].item()}, nrows={roff_idx.shape[0]-1}")


    # Vectorized _normalize_convolution_tensor_s2 vs the loop-based legacy reference.
    # One downsampling harmonic config (anisotropic kernels exercise the "modal" path);
    # full cross-product of basis_norm_mode x merge_quadrature x transpose_normalization.
    @parameterized.expand(
        [
            # in_shape, out_shape, kernel_shape, basis_type, grid_in, grid_out, mode, merge, transpose
            [(16, 32), (8, 16), (3, 3), "harmonic", "equiangular", "equiangular", "none",      False, False],
            [(16, 32), (8, 16), (3, 3), "harmonic", "equiangular", "equiangular", "none",      False, True ],
            [(16, 32), (8, 16), (3, 3), "harmonic", "equiangular", "equiangular", "none",      True,  False],
            [(16, 32), (8, 16), (3, 3), "harmonic", "equiangular", "equiangular", "none",      True,  True ],
            [(16, 32), (8, 16), (3, 3), "harmonic", "equiangular", "equiangular", "nodal",     False, False],
            [(16, 32), (8, 16), (3, 3), "harmonic", "equiangular", "equiangular", "nodal",     False, True ],
            [(16, 32), (8, 16), (3, 3), "harmonic", "equiangular", "equiangular", "nodal",     True,  False],
            [(16, 32), (8, 16), (3, 3), "harmonic", "equiangular", "equiangular", "nodal",     True,  True ],
            [(16, 32), (8, 16), (3, 3), "harmonic", "equiangular", "equiangular", "modal",     False, False],
            [(16, 32), (8, 16), (3, 3), "harmonic", "equiangular", "equiangular", "modal",     False, True ],
            [(16, 32), (8, 16), (3, 3), "harmonic", "equiangular", "equiangular", "modal",     True,  False],
            [(16, 32), (8, 16), (3, 3), "harmonic", "equiangular", "equiangular", "modal",     True,  True ],
            [(16, 32), (8, 16), (3, 3), "harmonic", "equiangular", "equiangular", "mean",      False, False],
            [(16, 32), (8, 16), (3, 3), "harmonic", "equiangular", "equiangular", "mean",      False, True ],
            [(16, 32), (8, 16), (3, 3), "harmonic", "equiangular", "equiangular", "mean",      True,  False],
            [(16, 32), (8, 16), (3, 3), "harmonic", "equiangular", "equiangular", "mean",      True,  True ],
            [(16, 32), (8, 16), (3, 3), "harmonic", "equiangular", "equiangular", "support",   False, False],
            [(16, 32), (8, 16), (3, 3), "harmonic", "equiangular", "equiangular", "support",   False, True ],
            [(16, 32), (8, 16), (3, 3), "harmonic", "equiangular", "equiangular", "support",   True,  False],
            [(16, 32), (8, 16), (3, 3), "harmonic", "equiangular", "equiangular", "support",   True,  True ],
            [(16, 32), (8, 16), (3, 3), "harmonic", "equiangular", "equiangular", "geometric", False, False],
            [(16, 32), (8, 16), (3, 3), "harmonic", "equiangular", "equiangular", "geometric", False, True ],
            [(16, 32), (8, 16), (3, 3), "harmonic", "equiangular", "equiangular", "geometric", True,  False],
            [(16, 32), (8, 16), (3, 3), "harmonic", "equiangular", "equiangular", "geometric", True,  True ],
        ],
        skip_on_empty=True,
    )
    def test_normalize_matches_legacy(self, in_shape, out_shape, kernel_shape, basis_type,
                                      grid_in, grid_out, basis_norm_mode, merge_quadrature,
                                      transpose_normalization, verbose=False):
        """Vectorized _normalize_convolution_tensor_s2 must agree with the legacy
        loop-based reference for every (basis_norm_mode, merge_quadrature, transpose_normalization)."""

        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        filter_basis = get_filter_basis(kernel_shape=kernel_shape, basis_type=basis_type)
        kernel_size = filter_basis.kernel_size
        theta_cutoff = torch.pi / float(nlat_out - 1)

        # build raw (un-normalized) psi via precompute with mode="none" + merge_quadrature=False;
        # the internal normalize call is then a no-op so out_vals are the raw filter values.
        idx, vals_raw, _ = _precompute_convolution_tensor_s2(
            in_shape=in_shape,
            out_shape=out_shape,
            filter_basis=filter_basis,
            grid_in=grid_in,
            grid_out=grid_out,
            theta_cutoff=theta_cutoff,
            transpose_normalization=False,
            basis_norm_mode="none",
            merge_quadrature=False,
        )

        # quadrature weights mirror the precompute's choice: win for non-transpose, wout for transpose
        _, win  = precompute_latitudes(nlat_in,  grid=grid_in)
        _, wout = precompute_latitudes(nlat_out, grid=grid_out)
        quad_weights = (wout if transpose_normalization else win).reshape(-1, 1) / nlon_in / 2.0

        ref = _normalize_convolution_tensor_s2_legacy(
            idx, vals_raw.clone(), in_shape, out_shape, kernel_size,
            quad_weights, theta_cutoff,
            transpose_normalization=transpose_normalization,
            basis_norm_mode=basis_norm_mode,
            merge_quadrature=merge_quadrature,
            isotropic_mask=filter_basis.isotropic_mask,
        )
        new = _normalize_convolution_tensor_s2(
            idx, vals_raw.clone(), in_shape, out_shape, kernel_size,
            quad_weights, theta_cutoff,
            transpose_normalization=transpose_normalization,
            basis_norm_mode=basis_norm_mode,
            merge_quadrature=merge_quadrature,
            isotropic_mask=filter_basis.isotropic_mask,
        )
        self.assertTrue(compare_tensors(
            f"normalize(mode={basis_norm_mode}, merge={merge_quadrature}, transpose={transpose_normalization})",
            new, ref, atol=1e-9, rtol=1e-9, verbose=verbose,
        ))


    @parameterized.expand(
        [
            # fp32 tests
            # regular convolution
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (16, 32), (8, 16), (3), "piecewise linear", "nodal", "equiangular", "equiangular",  torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (3, 3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (4, 3), "piecewise linear", "none", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (2, 1), "harmonic", "mean", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (3), "zernike", "nodal", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (3, 3), "harmonic", "nodal", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (3, 3), "fourier-bessel", "mean", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (16, 24), (8, 8), (3), "piecewise linear", "mean", "equiangular", "equiangular",  torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (18, 36), (6, 12), (7), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            # pscale=4 (exercises the default/fallback PSCALE=0 dispatch branch)
            [8, 4, 2, (16, 32), (4, 8), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (16, 32), (8, 16), (5), "piecewise linear", "mean", "equiangular", "legendre-gauss", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (16, 32), (8, 16), (5), "piecewise linear", "mean", "legendre-gauss", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (16, 32), (8, 16), (5), "piecewise linear", "nodal", "legendre-gauss", "legendre-gauss", torch.float32, False, 1e-4, 1e-4],
            # regular convolution — modal, support, geometric normalization
            [8, 4, 2, (24, 48), (12, 24), (3, 3), "piecewise linear", "modal", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "modal", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (3), "zernike", "modal", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (3, 3), "fourier-bessel", "modal", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (3), "piecewise linear", "support", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "support", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (3, 3), "piecewise linear", "geometric", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (3, 3), "fourier-bessel", "geometric", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "geometric", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            # transpose convolution
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "nodal", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (3, 3), "piecewise linear", "mean", "equiangular", "equiangular",  torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (4, 3), "piecewise linear", "none", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (2, 1), "harmonic", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (3), "zernike", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (3, 3), "fourier-bessel", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (8, 8), (16, 24), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (6, 12), (18, 36), (7), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            # pscale=4 (exercises the default/fallback PSCALE=0 dispatch branch)
            [8, 4, 2, (4, 8), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "mean", "equiangular", "legendre-gauss", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "mean", "legendre-gauss", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "mean", "legendre-gauss", "legendre-gauss", torch.float32, True, 1e-4, 1e-4],
            # transpose convolution — modal, support, geometric normalization
            [8, 4, 2, (12, 24), (24, 48), (3, 3), "piecewise linear", "modal", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (2, 2), "harmonic", "modal", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (3), "zernike", "modal", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (3, 3), "fourier-bessel", "modal", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (3), "piecewise linear", "support", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (2, 2), "harmonic", "support", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (3, 3), "piecewise linear", "geometric", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (3, 3), "fourier-bessel", "geometric", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (2, 2), "harmonic", "geometric", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            # fp64 tests
            # regular convolution
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float64, False, 1e-9, 1e-9],
            [8, 4, 2, (16, 32), (8, 16), (3), "piecewise linear", "nodal", "equiangular", "equiangular", torch.float64, False, 1e-9, 1e-9],
            [8, 4, 2, (24, 48), (12, 24), (3, 3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float64, False, 1e-9, 1e-9],
            [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float64, False, 1e-9, 1e-9],
            [8, 4, 2, (24, 48), (12, 24), (3), "zernike", "mean", "equiangular", "equiangular", torch.float64, False, 1e-9, 1e-9],
            [8, 4, 2, (24, 48), (12, 24), (3, 3), "fourier-bessel", "nodal", "equiangular", "equiangular", torch.float64, False, 1e-9, 1e-9],
            # transpose convolution
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float64, True, 1e-9, 1e-9],
            [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "mean", "equiangular", "equiangular", torch.float64, True, 1e-9, 1e-9],
            [8, 4, 2, (12, 24), (24, 48), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float64, True, 1e-9, 1e-9],
            [8, 4, 2, (12, 24), (24, 48), (3), "zernike", "mean", "equiangular", "equiangular", torch.float64, True, 1e-9, 1e-9],
            [8, 4, 2, (12, 24), (24, 48), (3, 3), "fourier-bessel", "mean", "equiangular", "equiangular", torch.float64, True, 1e-9, 1e-9],
            # fp16 tests (AMP)
            # regular convolution
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float16, False, 2e-2, 1e-2],
            [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float16, False, 2e-2, 1e-2],
            # transpose convolution
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float16, True, 2e-2, 1e-2],
            [8, 4, 2, (12, 24), (24, 48), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float16, True, 2e-2, 1e-2],
            # bf16 tests (AMP)
            # regular convolution
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.bfloat16, False, 5e-2, 5e-2],
            [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.bfloat16, False, 5e-2, 5e-2],
            # transpose convolution
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.bfloat16, True, 5e-2, 5e-2],
            [8, 4, 2, (12, 24), (24, 48), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.bfloat16, True, 5e-2, 5e-2],
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
        dtype,
        transpose,
        atol,
        rtol,
        verbose=True,
    ):
        # for AMP dtypes, the module and input stay in float32; autocast handles the rest
        is_amp = dtype in (torch.float16, torch.bfloat16)
        module_dtype = torch.float32 if is_amp else dtype

        # set seed
        set_seed(333)

        # use optimized kernels
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

        # psi comparison in float64 (both sides come from precompute in float64)
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

        # cast module to the target dtype for forward/backward
        if module_dtype != torch.float32:
            conv = conv.to(dtype=module_dtype)

        # create a copy of the weight
        w_ref = torch.empty_like(conv.weight)
        with torch.no_grad():
            w_ref.copy_(conv.weight)
        w_ref.requires_grad = True

        # create an input signal
        x = torch.randn(batch_size, in_channels, *in_shape, dtype=module_dtype, device=self.device)

        # FWD and BWD pass
        x.requires_grad = True
        with maybe_autocast(self.device.type, dtype):
            y = conv(x)
        grad_input = torch.randn_like(y)
        y.backward(grad_input)
        x_grad = x.grad.clone()

        # perform the reference computation
        x_ref = x.clone().detach()
        x_ref.requires_grad = True
        psi_ref = psi_dense.to(dtype=module_dtype)
        if transpose:
            y_ref = torch.einsum("oif,biqr->bofqr", w_ref, x_ref)
            y_ref = torch.einsum("fqrtp,bofqr->botp", psi_ref, y_ref)
        else:
            y_ref = torch.einsum("ftpqr,bcqr->bcftp", psi_ref, x_ref)
            y_ref = torch.einsum("oif,biftp->botp", w_ref, y_ref)
        y_ref.backward(grad_input)
        x_ref_grad = x_ref.grad.clone()

        # compare results
        self.assertTrue(compare_tensors(f"output", y.to(y_ref.dtype), y_ref, atol=atol, rtol=rtol, verbose=verbose))

        # compare
        self.assertTrue(compare_tensors(f"input grad", x_grad.to(x_ref_grad.dtype), x_ref_grad, atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors(f"weight grad", conv.weight.grad.to(w_ref.grad.dtype), w_ref.grad, atol=atol, rtol=rtol, verbose=verbose))


    @parameterized.expand(
        [
            # fp32 tests
            # regular convolution
            [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (41, 80), (2, 2), "harmonic", "nodal", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (41, 80), (2, 3), "harmonic", "mean", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (41, 80), (3), "zernike", "modal", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (41, 80), (3, 3), "fourier-bessel", "geometric", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (41, 80), (3, 3), "harmonic", "modal", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (21, 40), (3), "piecewise linear", "nodal", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (21, 40), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (21, 40), (2, 1), "harmonic", "geometric", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (21, 40), (3), "zernike", "mean", "equiangular", "equiangular", torch.float32, False, 1e-4, 1e-4],
            # transpose convolution
            [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "modal", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (41, 80), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (41, 80), (2, 3), "harmonic", "nodal", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (41, 80), (3), "zernike", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (41, 80), (3, 3), "fourier-bessel", "modal", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (41, 80), (3, 3), "harmonic", "geometric", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (21, 40), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (21, 40), (41, 80), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (21, 40), (41, 80), (2, 1), "harmonic", "mean", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            [8, 4, 2, (21, 40), (41, 80), (3), "zernike", "nodal", "equiangular", "equiangular", torch.float32, True, 1e-4, 1e-4],
            # fp64 tests
            # regular convolution
            [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float64, False, 1e-9, 1e-9],
            [8, 4, 2, (41, 80), (41, 80), (2, 2), "harmonic", "modal", "equiangular", "equiangular", torch.float64, False, 1e-9, 1e-9],
            [8, 4, 2, (41, 80), (21, 40), (3), "piecewise linear", "nodal", "equiangular", "equiangular", torch.float64, False, 1e-9, 1e-9],
            # transpose convolution
            [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "geometric", "equiangular", "equiangular", torch.float64, True, 1e-9, 1e-9],
            [8, 4, 2, (21, 40), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float64, True, 1e-9, 1e-9],
            [8, 4, 2, (21, 40), (41, 80), (2, 2), "harmonic", "modal", "equiangular", "equiangular", torch.float64, True, 1e-9, 1e-9],
            # fp16 tests (AMP)
            # regular convolution
            [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float16, False, 1e-2, 1e-2],
            [8, 4, 2, (41, 80), (41, 80), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float16, False, 1e-2, 1e-2],
            # transpose convolution
            [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float16, True, 1e-2, 1e-2],
            [8, 4, 2, (41, 80), (41, 80), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float16, True, 1e-2, 1e-2],
            # bf16 tests (AMP)
            # regular convolution
            [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.bfloat16, False, 1e-2, 1e-2],
            [8, 4, 2, (41, 80), (41, 80), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.bfloat16, False, 1e-2, 1e-2],
            # transpose convolution
            [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.bfloat16, True, 1e-2, 1e-2],
            [8, 4, 2, (41, 80), (41, 80), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.bfloat16, True, 1e-2, 1e-2],
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
        dtype,
        transpose,
        atol,
        rtol,
        verbose=False,
    ):
        # for AMP dtypes, the module and input stay in float32; autocast handles the rest
        is_amp = dtype in (torch.float16, torch.bfloat16)
        module_dtype = torch.float32 if is_amp else dtype

        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

        # set seed
        set_seed(333)

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
        ).to(dtype=module_dtype, device=self.device)

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
        ).to(dtype=module_dtype, device=self.device)

        # create a copy of the weight
        with torch.no_grad():
            conv_naive.weight.copy_(conv_opt.weight)

        # create an input signal
        inp = torch.randn(batch_size, in_channels, *in_shape, dtype=module_dtype, device=self.device)

        # FWD and BWD pass
        inp.requires_grad = True
        with maybe_autocast(self.device.type, dtype):
            out_naive = conv_naive(inp)
        grad_input = torch.randn_like(out_naive)
        out_naive.backward(grad_input)
        inp_grad_naive = inp.grad.clone()

        # perform the reference computation
        inp.grad = None
        with maybe_autocast(self.device.type, dtype):
            out_opt = conv_opt(inp)
        out_opt.backward(grad_input)
        inp_grad_opt = inp.grad.clone()

        # compare results
        self.assertTrue(compare_tensors(f"output", out_naive, out_opt, atol=atol, rtol=rtol, verbose=verbose))

        # compare
        self.assertTrue(compare_tensors(f"input grad", inp_grad_naive, inp_grad_opt, atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors(f"weight grad", conv_naive.weight.grad, conv_opt.weight.grad, atol=atol, rtol=rtol, verbose=verbose))


    @parameterized.expand(
        [
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", False, 1e-4, 1e-4],
            [8, 4, 2, (16, 32), (8, 16), (5), "piecewise linear", "mean", "legendre-gauss", "legendre-gauss", False, 1e-4, 1e-4],
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", True, 1e-4, 1e-4],
            [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "mean", "legendre-gauss", "legendre-gauss", True, 1e-4, 1e-4],
        ],
        skip_on_empty=True,
    )
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_device_instantiation(self, batch_size, in_channels, out_channels, in_shape, out_shape, kernel_shape, basis_type, basis_norm_mode, grid_in, grid_out, transpose, atol, rtol, verbose=False):

        set_seed(333)

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
        self.assertTrue(compare_tensors(f"psi col idx", conv_host.psi_col_idx.cpu(), conv_device.psi_col_idx.cpu(), atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors(f"psi row idx", conv_host.psi_row_idx.cpu(), conv_device.psi_row_idx.cpu(), atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors(f"psi roff idx", conv_host.psi_roff_idx.cpu(), conv_device.psi_roff_idx.cpu(), atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors(f"psi vals", conv_host.psi_vals.cpu(), conv_device.psi_vals.cpu(), atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors(f"psi idx", conv_host.psi_idx.cpu(), conv_device.psi_idx.cpu(), atol=atol, rtol=rtol, verbose=verbose))


    @parameterized.expand(
        [
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", False],
            [8, 4, 2, (16, 32), (8,  16), (3), "piecewise linear", "mean", "equiangular", "equiangular", False],
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", True],
            [8, 4, 2, (8,  16), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", True],

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
        verbose=False,
    ):
        """Tests whether the optimized kernels are PyTorch 2 compatible"""

        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping GPU test because CUDA kernels are not available")

        if verbose:
            print(f"Testing DISCO convolution on {in_shape[0]}x{in_shape[1]} {grid_in} grid to {out_shape[0]}x{out_shape[1]} {grid_out} grid on {self.device.type} device")

        set_seed(333)

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
            # (in_shape, out_shape, kernel_shape, transpose)
            # one row per dispatcher direction; the op-input has no other differentiable
            # input than `inp`, so freezing it covers the full contract surface.
            [(16, 32), (8, 16), (3, 3), False],  # standard conv
            [(8, 16), (16, 32), (3, 3), True ],  # transpose conv
        ],
        skip_on_empty=True,
    )
    @unittest.skipUnless(optimized_kernels_is_available(), "skipping test because optimized kernels are not available")
    def test_no_input_grad(self, in_shape, out_shape, kernel_shape, transpose, verbose=False):
        """Verifies the disco autograd contract when the module input does not require gradients.

        The disco custom op only has one differentiable input (``inp``, slot 0); the rest of the
        schema are int index buffers, float ``vals`` registered as a non-grad buffer, and Python
        ints. So the contract reduces to: when ``inp.requires_grad=False``, the backward must not
        crash and ``inp.grad`` must remain ``None`` — while the conv's learnable weight still gets
        a gradient via the einsum that sits outside the custom op.

        Baseline (``inp.requires_grad=True``) is also exercised as a sanity check.
        """
        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

        set_seed(333)

        batch_size, in_channels, out_channels = 2, 4, 4
        basis_type = "piecewise linear"
        nlat_in = in_shape[0]
        theta_cutoff = (kernel_shape[0] + 1) * torch.pi / float(nlat_in - 1)

        Conv = DiscreteContinuousConvTransposeS2 if transpose else DiscreteContinuousConvS2
        conv = Conv(
            in_channels, out_channels, in_shape, out_shape,
            kernel_shape, basis_type=basis_type, basis_norm_mode="mean",
            groups=1, grid_in="equiangular", grid_out="equiangular",
            bias=True, theta_cutoff=theta_cutoff,
        ).to(self.device)

        # --- baseline: inp requires grad ---
        inp = torch.randn(batch_size, in_channels, *in_shape, device=self.device, requires_grad=True)
        out = conv(inp)
        out.sum().backward()
        self.assertIsNotNone(inp.grad, "baseline: inp.grad should be populated when requires_grad=True")
        self.assertIsNotNone(conv.weight.grad, "baseline: weight.grad should be populated")

        # --- contract: inp does NOT require grad ---
        conv.zero_grad()
        inp_nograd = torch.randn(batch_size, in_channels, *in_shape, device=self.device, requires_grad=False)
        out = conv(inp_nograd)
        out.sum().backward()
        self.assertIsNone(inp_nograd.grad,
                          "contract violation: inp.grad must be None when requires_grad=False")
        self.assertIsNotNone(conv.weight.grad,
                             "weight.grad should still be populated via the einsum outside the op")

        # --- contract: psi_* buffers must never accumulate gradients ---
        # (they are non-learnable index/value tensors registered via register_buffer)
        for name in ("psi_roff_idx", "psi_ker_idx", "psi_row_idx", "psi_col_idx", "psi_vals"):
            buf = getattr(conv, name)
            self.assertIsNone(buf.grad,
                              f"buffer {name} should not accumulate a gradient (requires_grad={buf.requires_grad})")


    @parameterized.expand(
        [
            [8, 4, 2, (91, 180), (91, 180), (3), "piecewise linear", "mean", "equiangular", "equiangular", False, 1e-4],
        ],
        skip_on_empty=True,
    )
    @unittest.skipUnless(optimized_kernels_is_available() and _run_perf_tests, "skipping performance test because optimized kernels are not available or perf tests are disabled")
    def test_perf(self, batch_size, in_channels, out_channels, in_shape, out_shape, kernel_shape, basis_type, basis_norm_mode, grid_in, grid_out, transpose, tol, verbose=True):

        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

        set_seed(333)

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
