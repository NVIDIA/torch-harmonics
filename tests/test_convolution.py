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

import math
import types
import unittest
from unittest import mock

import torch
from parameterized import parameterized, parameterized_class
from testutils import _is_sm90, _is_sm100, _ProductGridAsRagged, compare_tensors, maybe_autocast, param_grad_ulps, requires_torch_compile, set_seed
from torch.library import opcheck

from torch_harmonics import DiscreteContinuousConvS2, DiscreteContinuousConvTransposeS2, HealpixGrid, as_grid
from torch_harmonics.disco import backends as disco_backends
from torch_harmonics.disco import cuda_kernels_is_available, optimized_kernels_is_available
from torch_harmonics.disco._disco_utils import _weight_contraction, _weight_expansion, _weight_gradient
from torch_harmonics.disco._psi_layouts import arcs_to_coo, build_arcs, build_kpacked
from torch_harmonics.disco.backends import RaggedOptimizedBackend, RaggedReferenceBackend, RegularOptimizedBackend, RegularReferenceBackend
from torch_harmonics.disco.convolution import (
    _precompute_convolution_tensor_s2,
)
from torch_harmonics.disco.optimized import disco_arch
from torch_harmonics.disco.optimized.disco_optimized import _kpacked_k_pad
from torch_harmonics.disco.optimized.disco_plan import ArcPlanS2
from torch_harmonics.filter_basis import get_filter_basis
from torch_harmonics.grid import RegularGridS2
from torch_harmonics.quadrature import compute_theta_cutoff, precompute_latitudes, precompute_longitudes

if not optimized_kernels_is_available():
    print("Warning: Couldn't import optimized disco convolution kernels")


# The layer's psi and the dense reference agree up to the float64 rounding of the basis
# normalization, a sum over each kernel's support (measured: up to 17 ULPs of max|psi|).
_PSI_ULPS = 64

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


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

    colats_in, win = precompute_latitudes(nlat_in, grid=grid_in)
    colats_out, wout = precompute_latitudes(nlat_out, grid=grid_out)

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
            alpha = -colats_out[t]
            beta = lons_in - lons_out[p]
            gamma = colats_in.reshape(-1, 1)

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
            # non-equiangular output grids, where the default theta_cutoff is driven by a
            # node distribution that is not uniform in theta (lobatto clusters towards the
            # equator, trapezoidal is equispaced in cos(theta))
            [(16, 32), (16, 32), (3, 3), "harmonic", "mean", "lobatto", "lobatto"],
            [(16, 32), (8, 16), (3, 3), "harmonic", "mean", "lobatto", "lobatto"],
            [(16, 32), (8, 16), (3, 3), "harmonic", "mean", "equiangular", "lobatto"],
            [(16, 32), (16, 32), (3, 3), "harmonic", "mean", "trapezoidal", "trapezoidal"],
            [(16, 32), (8, 16), (3, 3), "harmonic", "mean", "equiangular", "trapezoidal"],
        ],
        skip_on_empty=True,
    )
    def test_convolution_tensor_integrity(self, in_shape, out_shape, kernel_shape, basis_type, basis_norm_mode, grid_in, grid_out, verbose=False):
        """Structural invariants of psi that the kpacked layout relies on.

        Note: intentionally excludes the "piecewise linear" basis, whose per-kernel radial support
        yields non-uniform (row, col) sets across kernel indices. The remaining bases share a
        full-disk support across all kernel basis functions and therefore satisfy the invariants
        the optimized DISCO kernel relies on.
        """

        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        filter_basis = get_filter_basis(kernel_shape=kernel_shape, basis_type=basis_type)

        # use the same default DiscreteContinuousConvS2 would pick, rather than a
        # hardcoded pi/(nlat_out-1), which is only the node spacing of an equiangular grid
        theta_cutoff = compute_theta_cutoff(nlat_out, grid=grid_out)

        idx, vals, _ = _precompute_convolution_tensor_s2(
            as_grid(grid_in, nlat=in_shape[0], nlon=in_shape[1]),
            as_grid(grid_out, nlat=out_shape[0], nlon=out_shape[1]),
            filter_basis=filter_basis,
            theta_cutoff=theta_cutoff,
            transpose_normalization=False,
            basis_norm_mode=basis_norm_mode,
            merge_quadrature=True,
        )

        ker_idx = idx[0, ...].contiguous()
        row_idx = idx[1, ...].contiguous()
        col_idx = idx[2, ...].contiguous()
        vals = vals.contiguous()

        # 1) shape consistency
        self.assertEqual(ker_idx.shape[0], row_idx.shape[0])
        self.assertEqual(ker_idx.shape[0], col_idx.shape[0])
        self.assertEqual(ker_idx.shape[0], vals.shape[0])

        # 2) the arc form has one row per (kernel, output latitude)
        arcs = build_arcs(ker_idx, row_idx, col_idx, vals, nlon=nlon_in)
        self.assertEqual(arcs.row_ker.numel(), filter_basis.kernel_size * nlat_out)

        # 3) same number of nnz per kernel basis function
        _, counts = torch.unique(ker_idx, return_counts=True)
        self.assertTrue(torch.all(counts == counts[0]), f"multiplicity in ker_idx is not uniform: counts={counts.tolist()}")

        # 4) same (row, col) support pattern across all kernel basis functions
        row_idx_ref = row_idx[ker_idx == 0]
        col_idx_ref = col_idx[ker_idx == 0]
        for k in range(1, filter_basis.kernel_size):
            self.assertTrue(torch.equal(row_idx_ref, row_idx[ker_idx == k]), f"row_idx differs for kernel index {k}")
            self.assertTrue(torch.equal(col_idx_ref, col_idx[ker_idx == k]), f"col_idx differs for kernel index {k}")

        # 5) which is what the kpacked layout needs; K_pad only has to cover K here
        k_pad = ((filter_basis.kernel_size + 7) // 8) * 8
        self.assertIsNotNone(build_kpacked(ker_idx, row_idx, col_idx, vals, filter_basis.kernel_size, k_pad, nlat_out, nlon_in), "a shared support must pack")

    @parameterized.expand(
        [
            # in_shape, out_shape, kernel_shape, basis_type, grid_in, grid_out, transpose, theta_cutoff_scale
            [(16, 32), (16, 32), (3, 3), "harmonic", "equiangular", "equiangular", False, 1.0],
            [(16, 32), (8, 16), (3,), "piecewise linear", "equiangular", "equiangular", False, 1.0],
            # a wide cutoff: full rings at the poles, several rings per row, annuli with gaps
            [(24, 48), (12, 24), (3,), "piecewise linear", "equiangular", "equiangular", False, 4.0],
            [(16, 32), (8, 16), (3, 3), "harmonic", "legendre-gauss", "equiangular", False, 2.0],
            # the transpose's columns index the output grid
            [(8, 16), (16, 32), (3, 3), "harmonic", "equiangular", "equiangular", True, 1.0],
            [(8, 16), (16, 32), (3,), "piecewise linear", "equiangular", "legendre-gauss", True, 3.0],
        ],
        skip_on_empty=True,
    )
    def test_psi_arcs(self, in_shape, out_shape, kernel_shape, basis_type, grid_in, grid_out, transpose, theta_cutoff_scale, verbose=False):
        """The arc form of psi holds exactly psi's entries, in the shape the kernels assume."""

        Conv = DiscreteContinuousConvTransposeS2 if transpose else DiscreteContinuousConvS2
        grid_in_desc = as_grid(grid_in, nlat=in_shape[0], nlon=in_shape[1])
        grid_out_desc = as_grid(grid_out, nlat=out_shape[0], nlon=out_shape[1])
        theta_cutoff = theta_cutoff_scale * compute_theta_cutoff((in_shape if transpose else out_shape)[0], grid=grid_in if transpose else grid_out)
        # the reference backend needs no kernels; the arcs are built from the layer's description
        conv = Conv(grid_in_desc, grid_out_desc, 2, 2, kernel_shape, basis_type=basis_type, theta_cutoff=theta_cutoff, optimized_kernel=False)

        ker_idx, row_idx, col_idx, vals = conv._psi_coo()
        nlon = conv._psi_nlon
        arcs = build_arcs(ker_idx, row_idx, col_idx, vals, nlon=nlon)

        # lossless: the same entries with the same values, whatever the order
        def canon(k, r, c, v):
            key = (k.to(torch.int64) * (int(r.max()) + 1) + r.to(torch.int64)) * (int(c.max()) + 1) + c.to(torch.int64)
            order = torch.argsort(key)
            return key[order], v[order]

        key_ref, vals_ref = canon(ker_idx, row_idx, col_idx, vals)
        key_arc, vals_arc = canon(*arcs_to_coo(arcs, nlon))
        self.assertTrue(torch.equal(key_ref, key_arc), "the arcs changed the sparsity pattern")
        self.assertTrue(torch.equal(vals_ref, vals_arc), "the arcs changed the values")

        # every arc lies on one ring and wraps at most once
        seg = arcs.seg.to(torch.int64)
        ring, start, length = seg[:, 0], seg[:, 1], seg[:, 2]
        self.assertTrue(bool(((start >= 0) & (start < nlon) & (length >= 1) & (length <= nlon)).all()))

        # the offsets agree with the arcs: each row's values are exactly its arcs' lengths
        nrows = arcs.row_ker.numel()
        row_of_arc = torch.repeat_interleave(torch.arange(nrows), arcs.seg_off[1:] - arcs.seg_off[:-1])
        row_len = torch.zeros(nrows, dtype=torch.int64).index_add_(0, row_of_arc, length)
        self.assertTrue(torch.equal(row_len, arcs.val_off[1:] - arcs.val_off[:-1]))

        # rows sorted by basis function, which the spatial-first gradient slices by
        self.assertTrue(bool((arcs.row_ker[1:] >= arcs.row_ker[:-1]).all()))

        # rings ascend within a row, so the kernels restage or flush once per ring
        same_row = row_of_arc[1:] == row_of_arc[:-1]
        self.assertTrue(bool((ring[1:] >= ring[:-1])[same_row].all()))

        # arcs are maximal: no arc continues where the previous one on its ring ended, and a
        # run across the seam is one wrapping arc, not an arc ending at nlon and one at 0
        end = (start + length) % nlon
        same_ring = same_row & (ring[1:] == ring[:-1])
        self.assertFalse(bool((same_ring & (end[:-1] == start[1:])).any()), "adjacent arcs were not merged")

        # psi's row is the neighbourhood of output longitude 0, centred on the seam: every
        # ring the disk crosses without covering it has to wrap
        self.assertTrue(bool((start + length > nlon).any()), "no arc crosses the seam, so the merge went unexercised")

    @parameterized.expand(
        [
            # fp32 tests
            # regular convolution
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (16, 32), (8, 16), (3), "piecewise linear", "nodal", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (3, 3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (4, 3), "piecewise linear", "none", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (2, 1), "harmonic", "mean", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (3), "zernike", "nodal", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (3, 3), "harmonic", "nodal", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (3, 3), "fourier-bessel", "mean", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (16, 24), (8, 8), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (18, 36), (6, 12), (7), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            # pscale=4 (exercises the default/fallback PSCALE=0 dispatch branch)
            [8, 4, 2, (16, 32), (4, 8), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (16, 32), (8, 16), (5), "piecewise linear", "mean", "equiangular", "legendre-gauss", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (16, 32), (8, 16), (5), "piecewise linear", "mean", "legendre-gauss", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (16, 32), (8, 16), (5), "piecewise linear", "nodal", "legendre-gauss", "legendre-gauss", torch.float32, False, False, 1e-4, 1e-4],
            # regular convolution — modal, support, geometric normalization
            [8, 4, 2, (24, 48), (12, 24), (3, 3), "piecewise linear", "modal", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "modal", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (3), "zernike", "modal", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (3, 3), "fourier-bessel", "modal", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (3), "piecewise linear", "support", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "support", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (3, 3), "piecewise linear", "geometric", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (3, 3), "fourier-bessel", "geometric", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "geometric", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            # transpose convolution
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "nodal", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (3, 3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (4, 3), "piecewise linear", "none", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (2, 1), "harmonic", "mean", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (3), "zernike", "mean", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (3, 3), "fourier-bessel", "mean", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (8, 8), (16, 24), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (6, 12), (18, 36), (7), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            # pscale=4 (exercises the default/fallback PSCALE=0 dispatch branch)
            [8, 4, 2, (4, 8), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "mean", "equiangular", "legendre-gauss", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "mean", "legendre-gauss", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "mean", "legendre-gauss", "legendre-gauss", torch.float32, True, False, 1e-4, 1e-4],
            # transpose convolution — modal, support, geometric normalization
            [8, 4, 2, (12, 24), (24, 48), (3, 3), "piecewise linear", "modal", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (2, 2), "harmonic", "modal", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (3), "zernike", "modal", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (3, 3), "fourier-bessel", "modal", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (3), "piecewise linear", "support", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (2, 2), "harmonic", "support", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (3, 3), "piecewise linear", "geometric", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (3, 3), "fourier-bessel", "geometric", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (12, 24), (24, 48), (2, 2), "harmonic", "geometric", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            # fp64 tests
            # regular convolution
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float64, False, False, 1e-9, 1e-9],
            [8, 4, 2, (16, 32), (8, 16), (3), "piecewise linear", "nodal", "equiangular", "equiangular", torch.float64, False, False, 1e-9, 1e-9],
            [8, 4, 2, (24, 48), (12, 24), (3, 3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float64, False, False, 1e-9, 1e-9],
            [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float64, False, False, 1e-9, 1e-9],
            [8, 4, 2, (24, 48), (12, 24), (3), "zernike", "mean", "equiangular", "equiangular", torch.float64, False, False, 1e-9, 1e-9],
            [8, 4, 2, (24, 48), (12, 24), (3, 3), "fourier-bessel", "nodal", "equiangular", "equiangular", torch.float64, False, False, 1e-9, 1e-9],
            # transpose convolution
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float64, True, False, 1e-9, 1e-9],
            [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "mean", "equiangular", "equiangular", torch.float64, True, False, 1e-9, 1e-9],
            [8, 4, 2, (12, 24), (24, 48), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float64, True, False, 1e-9, 1e-9],
            [8, 4, 2, (12, 24), (24, 48), (3), "zernike", "mean", "equiangular", "equiangular", torch.float64, True, False, 1e-9, 1e-9],
            [8, 4, 2, (12, 24), (24, 48), (3, 3), "fourier-bessel", "mean", "equiangular", "equiangular", torch.float64, True, False, 1e-9, 1e-9],
            # fp16 tests (AMP)
            # regular convolution
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float16, False, False, 2e-2, 1e-2],
            [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float16, False, False, 2e-2, 1e-2],
            # transpose convolution
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float16, True, False, 2e-2, 1e-2],
            [8, 4, 2, (12, 24), (24, 48), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float16, True, False, 2e-2, 1e-2],
            # bf16 tests (AMP)
            # regular convolution
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.bfloat16, False, False, 2e-1, 5e-2],
            [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.bfloat16, False, False, 2e-1, 5e-2],
            # transpose convolution
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.bfloat16, True, False, 2e-1, 5e-2],
            [8, 4, 2, (12, 24), (24, 48), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.bfloat16, True, False, 2e-1, 5e-2],
            # fused convolution (forward conv only — compares fused against dense reference)
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, False, True, 1e-4, 1e-4],
            [8, 4, 2, (16, 32), (8, 16), (3), "piecewise linear", "nodal", "equiangular", "equiangular", torch.float32, False, True, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float32, False, True, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (3), "zernike", "nodal", "equiangular", "equiangular", torch.float32, False, True, 1e-4, 1e-4],
            [8, 4, 2, (24, 48), (12, 24), (3, 3), "fourier-bessel", "mean", "equiangular", "equiangular", torch.float32, False, True, 1e-4, 1e-4],
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float64, False, True, 1e-9, 1e-9],
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float16, False, True, 2e-2, 1e-2],
            [8, 4, 2, (24, 48), (12, 24), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.bfloat16, False, True, 5e-2, 5e-2],
            # The fused input gradient takes the spatial-first path only when the output is less than
            # half as wide as the input, as in the 4 -> 2 rows above; equal and widening channel
            # counts take the weight-first path, with the weight matmul ahead of the sparse transpose
            [8, 4, 4, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, False, True, 1e-4, 1e-4],
            [8, 2, 4, (24, 48), (12, 24), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float32, False, True, 1e-4, 1e-4],
            [8, 4, 4, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float16, False, True, 2e-2, 1e-2],
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
        fused,
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
        # fused is only supported for forward (non-transpose) convolution
        fused_kwarg = {"fused": fused} if (fused and not transpose) else {}
        conv = Conv(
            as_grid(grid_in, nlat=in_shape[0], nlon=in_shape[1]),
            as_grid(grid_out, nlat=out_shape[0], nlon=out_shape[1]),
            in_channels,
            out_channels,
            kernel_shape,
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            groups=1,
            bias=False,
            theta_cutoff=theta_cutoff,
            optimized_kernel=use_optimized_kernels,
            **fused_kwarg,
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

            # psi as the layer describes it, whichever backend holds it
            ker_idx, row_idx, col_idx, vals = conv._psi_coo()
            with torch.sparse.check_sparse_tensor_invariants(enable=False):
                psi = torch.sparse_coo_tensor(torch.stack([ker_idx, row_idx, col_idx]), vals, size=(conv.kernel_size, conv.nlat_in, conv.nlat_out * conv.nlon_out)).to_dense()
            psi = psi.to(self.device)

            self.assertTrue(compare_tensors("psi", psi, psi_dense[:, :, 0].reshape(-1, nlat_in, nlat_out * nlon_out), atol=0.0, rtol=0.0, ulps=_PSI_ULPS))
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

            ker_idx, row_idx, col_idx, vals = conv._psi_coo()
            with torch.sparse.check_sparse_tensor_invariants(enable=False):
                psi = torch.sparse_coo_tensor(torch.stack([ker_idx, row_idx, col_idx]), vals, size=(conv.kernel_size, conv.nlat_out, conv.nlat_in * conv.nlon_in)).to_dense()
            psi = psi.to(self.device)

            self.assertTrue(compare_tensors("psi", psi, psi_dense[:, :, 0].reshape(-1, nlat_out, nlat_in * nlon_in), atol=0.0, rtol=0.0, ulps=_PSI_ULPS))

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
        self.assertTrue(compare_tensors("output", y.to(y_ref.dtype), y_ref, atol=atol, rtol=rtol, verbose=verbose))

        # compare
        self.assertTrue(compare_tensors("input grad", x_grad.to(x_ref_grad.dtype), x_ref_grad, atol=atol, rtol=rtol, verbose=verbose))
        # the weight gradient is a sum over batch and grid, see param_grad_ulps
        pg_ulps = param_grad_ulps(batch_size * max(in_shape[0] * in_shape[1], out_shape[0] * out_shape[1]), dtype)
        self.assertTrue(compare_tensors("weight grad", conv.weight.grad.to(w_ref.grad.dtype), w_ref.grad, atol=atol, rtol=rtol, verbose=verbose, ulps=pg_ulps, eps_dtype=dtype))

    @parameterized.expand(
        [
            # fp32 tests
            # regular convolution
            [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (41, 80), (2, 2), "harmonic", "nodal", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (41, 80), (2, 3), "harmonic", "mean", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (41, 80), (3), "zernike", "modal", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (41, 80), (3, 3), "fourier-bessel", "geometric", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (41, 80), (3, 3), "harmonic", "modal", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (21, 40), (3), "piecewise linear", "nodal", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (21, 40), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (21, 40), (2, 1), "harmonic", "geometric", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (21, 40), (3), "zernike", "mean", "equiangular", "equiangular", torch.float32, False, False, 1e-4, 1e-4],
            # transpose convolution
            [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "modal", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (41, 80), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (41, 80), (2, 3), "harmonic", "nodal", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (41, 80), (3), "zernike", "mean", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (41, 80), (3, 3), "fourier-bessel", "modal", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (41, 80), (3, 3), "harmonic", "geometric", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (21, 40), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (21, 40), (41, 80), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (21, 40), (41, 80), (2, 1), "harmonic", "mean", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            [8, 4, 2, (21, 40), (41, 80), (3), "zernike", "nodal", "equiangular", "equiangular", torch.float32, True, False, 1e-4, 1e-4],
            # fp64 tests
            # regular convolution
            [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float64, False, False, 1e-9, 1e-9],
            [8, 4, 2, (41, 80), (41, 80), (2, 2), "harmonic", "modal", "equiangular", "equiangular", torch.float64, False, False, 1e-9, 1e-9],
            [8, 4, 2, (41, 80), (21, 40), (3), "piecewise linear", "nodal", "equiangular", "equiangular", torch.float64, False, False, 1e-9, 1e-9],
            # transpose convolution
            [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "geometric", "equiangular", "equiangular", torch.float64, True, False, 1e-9, 1e-9],
            [8, 4, 2, (21, 40), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float64, True, False, 1e-9, 1e-9],
            [8, 4, 2, (21, 40), (41, 80), (2, 2), "harmonic", "modal", "equiangular", "equiangular", torch.float64, True, False, 1e-9, 1e-9],
            # fp16 tests (AMP)
            # regular convolution
            [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float16, False, False, 5e-2, 1e-2],
            [8, 4, 2, (41, 80), (41, 80), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float16, False, False, 5e-2, 1e-2],
            # SM90 kpacked proxy-fence regression coverage:
            # large output grid with K=9 -> K_PAD=16.
            [8, 4, 2, (41, 80), (41, 80), (3, 3), "harmonic", "mean", "equiangular", "equiangular", torch.float16, False, False, 5e-2, 1e-2],
            # BC-heavy large-grid kpacked path; stresses multiple BC CTAs and weight contraction.
            [8, 32, 32, (41, 80), (41, 80), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float16, False, False, 5e-2, 1e-2],
            # transpose convolution
            [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float16, True, False, 5e-2, 1e-2],
            [8, 4, 2, (41, 80), (41, 80), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float16, True, False, 5e-2, 1e-2],
            # bf16 tests (AMP)
            # regular convolution
            [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.bfloat16, False, False, 3e-1, 1e-2],
            [8, 4, 2, (41, 80), (41, 80), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.bfloat16, False, False, 3e-1, 1e-2],
            [8, 4, 2, (41, 80), (41, 80), (3, 3), "harmonic", "mean", "equiangular", "equiangular", torch.bfloat16, False, False, 3e-1, 1e-2],
            # 32 channels make the weight grad large (|w.grad| ~ 70, where one bf16 step is 0.5), so
            # cancellation leaves near-zero entries ~2^-8 * 70 off in both paths; the weight grad's
            # range-scaled atol (param_grad_ulps) covers that, the row tolerance does not need to
            [8, 32, 32, (41, 80), (41, 80), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.bfloat16, False, False, 3e-1, 5e-2],
            # transpose convolution
            [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.bfloat16, True, False, 3e-1, 1e-2],
            [8, 4, 2, (41, 80), (41, 80), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.bfloat16, True, False, 3e-1, 1e-2],
            # fused convolution (forward conv only — compares fused optimized against torch reference)
            [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, False, True, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (21, 40), (3), "piecewise linear", "nodal", "equiangular", "equiangular", torch.float32, False, True, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (21, 40), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float32, False, True, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (21, 40), (3), "zernike", "mean", "equiangular", "equiangular", torch.float32, False, True, 1e-4, 1e-4],
            [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float64, False, True, 1e-9, 1e-9],
            [8, 4, 2, (41, 80), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float16, False, True, 1e-2, 1e-2],
            [8, 4, 2, (41, 80), (41, 80), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float16, False, True, 5e-2, 1e-2],
            # same tolerance as the unfused bf16 row: CPU autocast now reaches the fused path too,
            # which used to run it in fp32 and so passed a tighter bound on CPU only
            [8, 4, 2, (41, 80), (41, 80), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.bfloat16, False, True, 3e-1, 5e-2],
            # equal and widening channel counts: the weight-first input gradient (see test_sparse_against_dense)
            [8, 4, 4, (41, 80), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float32, False, True, 1e-4, 1e-4],
            [8, 2, 4, (41, 80), (21, 40), (2, 2), "harmonic", "mean", "equiangular", "equiangular", torch.float32, False, True, 1e-4, 1e-4],
            # fp16 like the unfused rows: CPU autocast now reaches the fused path, which used to
            # run it in fp32 and so passed a tighter bound on CPU only
            [8, 4, 4, (41, 80), (41, 80), (3), "piecewise linear", "mean", "equiangular", "equiangular", torch.float16, False, True, 5e-2, 1e-2],
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
        fused,
        atol,
        rtol,
        verbose=True,
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
            as_grid(grid_in, nlat=in_shape[0], nlon=in_shape[1]),
            as_grid(grid_out, nlat=out_shape[0], nlon=out_shape[1]),
            in_channels,
            out_channels,
            kernel_shape,
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            groups=1,
            bias=False,
            theta_cutoff=theta_cutoff,
            optimized_kernel=False,
        ).to(dtype=module_dtype, device=self.device)

        # fused is only supported for forward (non-transpose) convolution
        fused_kwarg = {"fused": fused} if (fused and not transpose) else {}
        conv_opt = Conv(
            as_grid(grid_in, nlat=in_shape[0], nlon=in_shape[1]),
            as_grid(grid_out, nlat=out_shape[0], nlon=out_shape[1]),
            in_channels,
            out_channels,
            kernel_shape,
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            groups=1,
            bias=False,
            theta_cutoff=theta_cutoff,
            optimized_kernel=True,
            **fused_kwarg,
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
        self.assertTrue(compare_tensors("output", out_opt, out_naive, atol=atol, rtol=rtol, verbose=verbose))

        # compare; the weight gradient is a sum over batch and grid, see param_grad_ulps
        pg_ulps = param_grad_ulps(batch_size * max(nlat_in * nlon_in, nlat_out * nlon_out), dtype)
        self.assertTrue(compare_tensors("input grad", inp_grad_opt, inp_grad_naive, atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors("weight grad", conv_opt.weight.grad, conv_naive.weight.grad, atol=atol, rtol=rtol, verbose=verbose, ulps=pg_ulps, eps_dtype=dtype))

    @parameterized.expand(
        [
            # [transpose, in_shape, out_shape, autocast_dtype]
            # regular DISCO (downsample): in > out
            [False, (16, 32), (8, 16), torch.float16],
            [False, (16, 32), (8, 16), torch.bfloat16],
            # transpose DISCO (upsample): in < out
            [True, (8, 16), (16, 32), torch.float16],
            [True, (8, 16), (16, 32), torch.bfloat16],
        ],
        skip_on_empty=True,
    )
    def test_optimized_autocast_dtype(self, transpose, in_shape, out_shape, autocast_dtype):
        """Direct check that the autocast registration on the optimized DISCO custom_ops
        produces output in the active autocast dtype.

        Uses bias=False so the final op of the conv module is the weight matmul, which is
        autocast-eligible and preserves the autocast dtype. A bias add of bf16+fp32
        would dtype-promote to fp32 and mask the autocast contract we're testing.
        """
        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")
        if not optimized_kernels_is_available():
            raise unittest.SkipTest("skipping test because optimized kernels are not available")

        set_seed(333)

        Conv = DiscreteContinuousConvTransposeS2 if transpose else DiscreteContinuousConvS2
        nlat_in = in_shape[0]
        theta_cutoff = 4 * torch.pi / float(nlat_in - 1)

        conv = Conv(
            grid_in=as_grid("equiangular", nlat=in_shape[0], nlon=in_shape[1]),
            grid_out=as_grid("equiangular", nlat=out_shape[0], nlon=out_shape[1]),
            in_channels=4,
            out_channels=4,
            kernel_shape=(3,),
            basis_type="piecewise linear",
            basis_norm_mode="mean",
            groups=1,
            bias=False,
            theta_cutoff=theta_cutoff,
            optimized_kernel=True,
        ).to(self.device)

        # Module + input in fp32; autocast handles the cast inside fwd.
        x = torch.randn(2, 4, *in_shape, device=self.device, dtype=torch.float32)

        with torch.autocast(self.device.type, dtype=autocast_dtype):
            out = conv(x)

        self.assertEqual(
            out.dtype,
            autocast_dtype,
            f"{Conv.__name__} output dtype {out.dtype} != autocast dtype {autocast_dtype}",
        )

    @parameterized.expand(
        [
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", False],
            [8, 4, 2, (16, 32), (8, 16), (5), "piecewise linear", "mean", "legendre-gauss", "legendre-gauss", False],
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", True],
            [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "mean", "legendre-gauss", "legendre-gauss", True],
        ],
        skip_on_empty=True,
    )
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_device_instantiation(
        self, batch_size, in_channels, out_channels, in_shape, out_shape, kernel_shape, basis_type, basis_norm_mode, grid_in, grid_out, transpose, verbose=False
    ):

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
            as_grid(grid_in, nlat=in_shape[0], nlon=in_shape[1]),
            as_grid(grid_out, nlat=out_shape[0], nlon=out_shape[1]),
            in_channels,
            out_channels,
            kernel_shape,
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            groups=1,
            bias=False,
            theta_cutoff=theta_cutoff,
        )

        # torch.set_default_device(self.device)
        with torch.device(self.device):
            conv_device = Conv(
                as_grid(grid_in, nlat=in_shape[0], nlon=in_shape[1]),
                as_grid(grid_out, nlat=out_shape[0], nlon=out_shape[1]),
                in_channels,
                out_channels,
                kernel_shape,
                basis_type=basis_type,
                basis_norm_mode=basis_norm_mode,
                groups=1,
                bias=False,
                theta_cutoff=theta_cutoff,
            )

        # since we specified the device specifier everywhere, it should always
        # use the cpu and it should be the same everywhere
        for name in ("psi_row_ker", "psi_row_lat", "psi_seg_off", "psi_seg", "psi_val_off", "psi_vals"):
            self.assertTrue(compare_tensors(name, getattr(conv_device, name).cpu(), getattr(conv_host, name).cpu(), atol=0.0, rtol=0.0, verbose=verbose))

    @parameterized.expand(
        [
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", False, False],
            [8, 4, 2, (16, 32), (8, 16), (3), "piecewise linear", "mean", "equiangular", "equiangular", False, False],
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", True, False],
            [8, 4, 2, (8, 16), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", True, False],
            # fused forward convolution
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", False, True],
            [8, 4, 2, (16, 32), (8, 16), (3), "piecewise linear", "mean", "equiangular", "equiangular", False, True],
        ],
        skip_on_empty=True,
    )
    @unittest.skipUnless((optimized_kernels_is_available()), "skipping test because optimized kernels are not available")
    @requires_torch_compile
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
        fused,
        verbose=False,
    ):
        """Tests whether the optimized kernels are PyTorch 2 compatible"""

        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping GPU test because CUDA kernels are not available")

        set_seed(333)

        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        if isinstance(kernel_shape, int):
            theta_cutoff = (kernel_shape + 1) * torch.pi / float(nlat_in - 1)
        else:
            theta_cutoff = (kernel_shape[0] + 1) * torch.pi / float(nlat_in - 1)

        Conv = DiscreteContinuousConvTransposeS2 if transpose else DiscreteContinuousConvS2
        fused_kwarg = {"fused": fused} if (fused and not transpose) else {}
        conv = Conv(
            as_grid(grid_in, nlat=in_shape[0], nlon=in_shape[1]),
            as_grid(grid_out, nlat=out_shape[0], nlon=out_shape[1]),
            in_channels,
            out_channels,
            kernel_shape,
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            groups=1,
            bias=False,
            theta_cutoff=theta_cutoff,
            **fused_kwarg,
        ).to(self.device)

        inp = torch.randn(batch_size, in_channels, *in_shape, device=self.device)

        if not transpose:
            # The convolution, fused or not, is an autograd.Function around the raw kernels
            # rather than an op of its own, so check that it traces as a whole -- forward and
            # backward, in one graph -- and agrees with eager. aot_eager exercises the fake
            # kernels and the joint graph without needing a codegen toolchain.
            compiled = torch.compile(conv, backend="aot_eager", fullgraph=True)
            inp_eager = inp.clone().requires_grad_(True)
            inp_compiled = inp.clone().requires_grad_(True)
            out_eager = conv(inp_eager)
            out_compiled = compiled(inp_compiled)
            self.assertTrue(compare_tensors("output", out_compiled, out_eager, atol=1e-5, rtol=1e-5, verbose=verbose))
            grad = torch.randn_like(out_eager)
            out_eager.backward(grad)
            weight_grad_eager = conv.weight.grad.clone()
            conv.weight.grad = None
            out_compiled.backward(grad)
            self.assertTrue(compare_tensors("input grad", inp_compiled.grad, inp_eager.grad, atol=1e-5, rtol=1e-5, verbose=verbose))
            self.assertTrue(compare_tensors("weight grad", conv.weight.grad, weight_grad_eager, atol=1e-5, rtol=1e-5, verbose=verbose))

            # and the op it contracts with satisfies the op contract
            test_inputs = (inp, *_arc_state(conv), conv.kernel_size, conv.nlat_out, conv.nlon_out)
            opcheck(torch.ops.disco_kernels._disco_s2_contraction_regular_optimized, test_inputs)
        else:
            # the scatter op reads (B, C, K, H, W): one plane per basis function per channel
            inp = torch.randn(batch_size, in_channels, conv.kernel_size, *in_shape, device=self.device)
            test_inputs = (inp, *_arc_state(conv), conv.kernel_size, conv.nlat_out, conv.nlon_out)
            opcheck(torch.ops.disco_kernels._disco_s2_transpose_contraction_regular_optimized, test_inputs)

    @parameterized.expand(
        [
            # (in_shape, out_shape, kernel_shape, transpose)
            # one row per dispatcher direction; the op-input has no other differentiable
            # input than `inp`, so freezing it covers the full contract surface.
            [(16, 32), (8, 16), (3, 3), False],  # standard conv
            [(8, 16), (16, 32), (3, 3), True],  # transpose conv
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
        a gradient via the weight matmul that sits outside the custom op.

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
            as_grid("equiangular", nlat=in_shape[0], nlon=in_shape[1]),
            as_grid("equiangular", nlat=out_shape[0], nlon=out_shape[1]),
            in_channels,
            out_channels,
            kernel_shape,
            basis_type=basis_type,
            basis_norm_mode="mean",
            groups=1,
            bias=True,
            theta_cutoff=theta_cutoff,
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
        self.assertIsNone(inp_nograd.grad, "contract violation: inp.grad must be None when requires_grad=False")
        self.assertIsNotNone(conv.weight.grad, "weight.grad should still be populated via the weight matmul outside the op")

        # --- contract: psi_* buffers must never accumulate gradients ---
        # (they are non-learnable index/value tensors registered via register_buffer)
        for name in conv._backend_state:
            buf = getattr(conv, name)
            self.assertIsNone(buf.grad, f"buffer {name} should not accumulate a gradient (requires_grad={buf.requires_grad})")

    @parameterized.expand(
        [
            # (in_channels, out_channels, groups, transpose, fused, optimized_kernel, contiguous)
            # the optimized convolution: weight-first (4->4) and spatial-first (8->2) input
            # gradient, saved and recomputed (fused) intermediate, groups, a strided input
            [4, 4, 1, False, False, True, True],
            [8, 2, 1, False, False, True, True],
            [4, 4, 1, False, True, True, True],
            [8, 2, 1, False, True, True, True],
            [4, 6, 2, False, False, True, True],
            [4, 4, 1, False, False, True, False],
            # the torch reference, and the transpose convolution
            [4, 4, 1, False, False, False, True],
            [8, 2, 1, False, True, False, True],
            [4, 4, 1, True, False, True, True],
            [4, 4, 1, True, False, False, True],
        ],
        skip_on_empty=True,
    )
    def test_double_backward(self, in_channels, out_channels, groups, transpose, fused, optimized_kernel, contiguous, verbose=False):
        """Gradients taken with create_graph=True are differentiable, and equal the ordinary ones."""
        if optimized_kernel and not optimized_kernels_is_available():
            raise unittest.SkipTest("skipping test because optimized kernels are not available")
        if optimized_kernel and (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

        set_seed(333)

        grid = as_grid("equiangular", nlat=8, nlon=16)
        Conv = DiscreteContinuousConvTransposeS2 if transpose else DiscreteContinuousConvS2
        fused_kwarg = {} if transpose else {"fused": fused}
        conv = Conv(grid, grid, in_channels, out_channels, (3), basis_type="piecewise linear", groups=groups, bias=False, optimized_kernel=optimized_kernel, **fused_kwarg)
        conv = conv.to(device=self.device, dtype=torch.float64)

        if contiguous:
            inp = torch.randn(1, in_channels, 8, 16, device=self.device, dtype=torch.float64)
        else:
            inp = torch.randn(1, 16, 8, in_channels, device=self.device, dtype=torch.float64).permute(0, 3, 2, 1)
        inp.requires_grad_(True)
        weight = conv.weight.detach().clone().requires_grad_(True)

        def fn(x, w):
            return torch.func.functional_call(conv, {"weight": w}, (x,))

        # the create_graph backward is a different code path: it must give the same gradients
        grad = torch.randn_like(fn(inp, weight))
        plain = torch.autograd.grad(fn(inp, weight), (inp, weight), grad)
        graphed = torch.autograd.grad(fn(inp, weight), (inp, weight), grad, create_graph=True)
        self.assertTrue(compare_tensors("input grad", graphed[0], plain[0], atol=1e-12, rtol=1e-10, verbose=verbose))
        self.assertTrue(compare_tensors("weight grad", graphed[1], plain[1], atol=1e-12, rtol=1e-10, verbose=verbose))

        # the CUDA scatter kernel accumulates with atomics, so two backward passes differ
        # in the last bits; gradcheck's reentrancy check wants them bit-identical otherwise
        self.assertTrue(torch.autograd.gradgradcheck(fn, (inp, weight), nondet_tol=1e-12))


# A supported device is not sufficient: the kpacked buffers are only built when
# this build actually contains the matching kernel (an arch-specific extension holding
# the image of the device's target, see disco_arch). A build targeting an architecture newer than
# the ones with kpacked kernels reports e.g. major == 10 while carrying no
# sm_100a cubin, so a device-only guard runs these tests against buffers that
# were deliberately never constructed. Ask the same question the library asks.
def _kpacked_built_for_sm90():
    """Hopper device AND an sm_90a kpacked kernel this device can load."""
    return _is_sm90() and _kpacked_runnable_here()


def _kpacked_built_for_sm100():
    """Blackwell device AND an sm_100a / sm_103a kpacked kernel this device can load."""
    return _is_sm100() and _kpacked_runnable_here()


def _kpacked_runnable_here():
    """Defer to the library, so the tests and the dispatch cannot disagree."""
    if not torch.cuda.is_available():
        return False
    return disco_arch.kpacked_supported_on_device(torch.cuda.current_device())


def _is_kpacked_supported():
    """Return True when the kpacked forward can actually run here (device AND build)."""
    return _kpacked_built_for_sm90() or _kpacked_built_for_sm100()


class _CountingPlan(ArcPlanS2):
    """
    The arc plan with a state tensor of its own and a count of the calls it serves.

    Stands in for a drop-in kernel: its gather and scatter read their values from
    ``psi_vals_plan``, which its backend builds in ``prepare``, rather than from the arcs, so
    the tests see that the plan's own state reaches it through the autograd nodes. It claims
    the gather for every dtype, so ``contract`` runs it too.
    """

    calls = None

    def has_gather(self, dtype):
        return True

    def has_scatter(self, dtype):
        return True

    def _count(self, op):
        # a side effect Dynamo will not trace inside the autograd node; the compile test does not count
        if not torch.compiler.is_compiling():
            type(self).calls[op] += 1

    def _with_own_vals(self, t):
        own = dict(zip(self.state_names, t))
        own["psi_vals"] = own["psi_vals_plan"]
        return tuple(own[name] for name in self.state_names)

    def gather(self, x, t, tc=()):
        self._count("gather")
        return super().gather(x, self._with_own_vals(t), tc)

    def scatter(self, g, t, tc, nlat_in, nlon_in):
        self._count("scatter")
        return super().scatter(g, self._with_own_vals(t), tc, nlat_in, nlon_in)

    def scatter_split(self, g, t, tc, nlat_in, nlon_in):
        self._count("scatter_split")
        return super().scatter_split(g, self._with_own_vals(t), tc, nlat_in, nlon_in)


class _KeyedPlan(ArcPlanS2):
    """
    The arc plan with call state: each operation reads its values from state built for the
    planes it contracts, as a drop-in kernel reads psi laid out for the launch configuration
    it picks from B * C. Records the keys it builds and the keys each operation is run with.
    """

    has_call_state = True
    built = None
    shapes = None
    used = None

    def has_gather(self, dtype):
        return True

    def call_state(self, op, bc, dtype, in_shape, t):
        type(self).built.append((op, bc, dtype))
        type(self).shapes.append(in_shape)
        # the values, and the key they were built for
        return (self.arcs(t)[-1].clone(), torch.tensor([bc], device=t[0].device))

    def _use(self, op, x, t, tc):
        vals, bc = tc
        if not torch.compiler.is_compiling():
            type(self).used.append((op, int(bc.item()), x.shape[0] * x.shape[1]))
        return t[: self._index["psi_vals"]] + (vals,) + t[self._index["psi_vals"] + 1 :]

    def gather(self, x, t, tc=()):
        return super().gather(x, self._use("gather", x, t, tc), tc)

    def scatter(self, g, t, tc, nlat_in, nlon_in):
        return super().scatter(g, self._use("scatter", g, t, tc), tc, nlat_in, nlon_in)

    def scatter_split(self, g, t, tc, nlat_in, nlon_in):
        return super().scatter_split(g, self._use("scatter_split", g, t, tc), tc, nlat_in, nlon_in)


class _KeyedBackend(RegularOptimizedBackend):
    """A backend whose plan has call state."""

    name = "regular-keyed"
    plan_class = _KeyedPlan


class _CountingBackend(RegularOptimizedBackend):
    """A backend that differs from the arc one only in its plan and the state that plan adds."""

    name = "regular-counting"
    plan_class = _CountingPlan

    def prepare(self, layer, device):
        state = self._arc_state(layer, device)
        state["psi_vals_plan"] = state["psi_vals"].clone()
        self.plan = self._make_plan(layer, state)
        return state


@unittest.skipUnless(optimized_kernels_is_available(), "skipping plan tests: optimized kernels not available")
@parameterized_class(("device",), _devices)
class TestDiscoPlan(unittest.TestCase):
    """A backend brings its kernels as a plan and inherits the autograd nodes around them."""

    def setUp(self):
        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")
        _CountingPlan.calls = {"gather": 0, "scatter": 0, "scatter_split": 0}
        _KeyedPlan.built, _KeyedPlan.shapes, _KeyedPlan.used = [], [], []

    def _pair(self, in_channels, out_channels, fused, backend=None):
        """The same layer twice, with the counting (or the given) backend and with the arc backend."""
        convs = []
        for backend in (backend or _CountingBackend, RegularOptimizedBackend):
            set_seed(333)
            conv = DiscreteContinuousConvS2(
                as_grid("equiangular", nlat=16, nlon=32),
                as_grid("equiangular", nlat=8, nlon=16),
                in_channels,
                out_channels,
                (3, 2),
                basis_type="harmonic",
                bias=False,
                theta_cutoff=0.3,
                fused=fused,
            )
            conv._backends = (backend, RegularReferenceBackend)
            conv._select_backend()
            convs.append(conv.to(self.device))
        self.assertEqual(convs[1].backend.name, "regular-optimized")
        return convs

    @parameterized.expand(
        [
            # in_channels, out_channels, fused: weight-first and spatial-first input gradient
            [4, 4, False],
            [4, 4, True],
            [8, 2, False],
            [8, 2, True],
        ]
    )
    def test_drop_in_plan_runs_the_conv(self, in_channels, out_channels, fused):
        """The plan's gather, scatter and recompute run in place of the arc kernels, with the same result."""
        conv, conv_arc = self._pair(in_channels, out_channels, fused)
        self.assertEqual(conv.backend.name, "regular-counting")
        self.assertIn("psi_vals_plan", dict(conv.named_buffers()))

        x = torch.randn(2, in_channels, 16, 32, device=self.device)
        x1, x2 = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
        out, out_arc = conv(x1), conv_arc(x2)
        grad = torch.randn_like(out)
        out.backward(grad)
        out_arc.backward(grad)

        self.assertTrue(compare_tensors("output", out, out_arc, atol=1e-6, rtol=1e-6))
        self.assertTrue(compare_tensors("input grad", x1.grad, x2.grad, atol=1e-6, rtol=1e-6))
        self.assertTrue(compare_tensors("weight grad", conv.weight.grad, conv_arc.weight.grad, atol=1e-6, rtol=1e-6))

        spatial_first = conv._needs_split
        self.assertEqual(spatial_first, out_channels * 2 <= in_channels)
        # the forward, and the recompute of the fused layer
        self.assertEqual(_CountingPlan.calls["gather"], 2 if fused else 1)
        self.assertEqual(_CountingPlan.calls["scatter_split"], 1 if spatial_first else 0)
        self.assertEqual(_CountingPlan.calls["scatter"], 0 if spatial_first else 1)

    @parameterized.expand([["both"], ["scatter"]])
    def test_drop_in_plan_runs_the_contraction(self, claims):
        """A plan with kernels of its own serves ``contract`` (the unfused distributed convolution) too, forward and backward."""
        conv, conv_arc = self._pair(4, 4, False)
        if claims == "scatter":
            # a plan with only a scatter of its own still takes the contraction's backward
            conv.backend.plan.has_gather = lambda dtype: False
        x = torch.randn(2, 4, 16, 32, device=self.device)
        x1, x2 = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
        y, y_arc = conv.backend.contract(conv, x1), conv_arc.backend.contract(conv_arc, x2)
        grad = torch.randn_like(y)
        y.backward(grad)
        y_arc.backward(grad)

        self.assertEqual(_CountingPlan.calls["gather"], 1)
        self.assertEqual(_CountingPlan.calls["scatter"], 1)
        self.assertTrue(compare_tensors("contraction", y, y_arc, atol=1e-6, rtol=1e-6))
        self.assertTrue(compare_tensors("input grad", x1.grad, x2.grad, atol=1e-6, rtol=1e-6))

    def test_drop_in_plan_contraction_double_backward(self):
        """Under ``create_graph=True`` the contraction's backward is built from ops autograd sees, and stays differentiable."""
        conv, conv_arc = self._pair(4, 4, False)
        x = torch.randn(2, 4, 16, 32, device=self.device, dtype=torch.float64)
        conv, conv_arc = conv.double(), conv_arc.double()
        grads = []
        for c in (conv, conv_arc):
            xi = x.clone().requires_grad_(True)
            y = c.backend.contract(c, xi)
            (g,) = torch.autograd.grad(y.square().sum(), xi, create_graph=True)
            if c is conv:
                # the differentiable path, not the plan's raw scatter
                self.assertEqual(_CountingPlan.calls["scatter"], 0)
            (gg,) = torch.autograd.grad(g.square().sum(), xi)
            if c is conv:
                # the second pass builds no graph: back through the node, it is the plan's scatter
                self.assertEqual(_CountingPlan.calls["scatter"], 1)
            grads.append((g.detach(), gg))
        self.assertEqual(conv.backend.name, "regular-counting")
        self.assertTrue(compare_tensors("first derivative", grads[0][0], grads[1][0], atol=1e-10, rtol=1e-10))
        self.assertTrue(compare_tensors("second derivative", grads[0][1], grads[1][1], atol=1e-10, rtol=1e-10))

    def test_plan_state_goes_with_the_backend(self):
        """The plan's own tensors are the backend's state: registered with it, removed when another backend is selected."""
        conv, _ = self._pair(4, 4, False)
        self.assertIn("psi_vals_plan", dict(conv.named_buffers()))
        conv._backends = (RegularOptimizedBackend, RegularReferenceBackend)
        conv._select_backend()
        self.assertEqual(conv.backend.name, "regular-optimized")
        self.assertNotIn("psi_vals_plan", dict(conv.named_buffers()))

    @parameterized.expand(
        [
            # in_channels, out_channels, fused: weight-first and spatial-first input gradient
            [4, 4, False],
            [4, 4, True],
            [8, 2, False],
        ]
    )
    def test_call_state_is_built_once_per_key(self, in_channels, out_channels, fused):
        """Call state is built per operation and number of planes, reused across calls, and each operation gets its own."""
        conv, conv_arc = self._pair(in_channels, out_channels, fused, backend=_KeyedBackend)
        self.assertEqual(conv.backend.name, "regular-keyed")
        spatial_first = conv._needs_split
        dgrad_op = "scatter_split" if spatial_first else "scatter"

        for batch in (2, 2, 3, 2):
            x = torch.randn(batch, in_channels, 16, 32, device=self.device)
            x1, x2 = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
            out, out_arc = conv(x1), conv_arc(x2)
            grad = torch.randn_like(out)
            out.backward(grad)
            out_arc.backward(grad)
            self.assertTrue(compare_tensors("output", out, out_arc, atol=1e-6, rtol=1e-6))
            self.assertTrue(compare_tensors("input grad", x1.grad, x2.grad, atol=1e-6, rtol=1e-6))
            self.assertTrue(compare_tensors("weight grad", conv.weight.grad, conv_arc.weight.grad, atol=1e-6, rtol=1e-6))
            conv.weight.grad = conv_arc.weight.grad = None

        # built once per (op, planes), for batch 2 and for batch 3; the input gradient's over
        # the input's planes weight-first, over the output gradient's spatial-first
        dgrad_channels = out_channels if spatial_first else in_channels
        expected = [(op, b * c, torch.float32) for b in (2, 3) for op, c in (("gather", in_channels), (dgrad_op, dgrad_channels))]
        self.assertEqual(_KeyedPlan.built, expected)
        # with the shape of the contraction's input, the gather's and the scatters' alike
        self.assertEqual(set(_KeyedPlan.shapes), {(16, 32)})
        # every operation ran with the state built for the planes it contracted
        for op, bc_built, bc_seen in _KeyedPlan.used:
            self.assertEqual(bc_built, bc_seen, f"{op} ran with the call state of another key")
        gathers = 4 * (2 if fused else 1)
        self.assertEqual(sum(op == "gather" for op, _, _ in _KeyedPlan.used), gathers)
        self.assertEqual(sum(op == dgrad_op for op, _, _ in _KeyedPlan.used), 4)

    def test_call_state_built_under_inference_serves_training(self):
        """Call state first built under inference mode is ordinary state: a later training call reuses it and runs backward."""
        conv, conv_arc = self._pair(4, 4, False, backend=_KeyedBackend)
        x = torch.randn(2, 4, 16, 32, device=self.device)
        with torch.inference_mode():
            conv(x)
        self.assertEqual(_KeyedPlan.built, [("gather", 8, torch.float32)])
        state = conv.backend.plan._call_states[("gather", 8, torch.float32, (16, 32))]
        self.assertFalse(any(s.is_inference() or s.requires_grad for s in state))

        x1, x2 = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
        out, out_arc = conv(x1), conv_arc(x2)
        grad = torch.randn_like(out)
        out.backward(grad)
        out_arc.backward(grad)
        # the gather's state was reused, the scatter's built now
        self.assertEqual(_KeyedPlan.built, [("gather", 8, torch.float32), ("scatter", 8, torch.float32)])
        self.assertTrue(compare_tensors("input grad", x1.grad, x2.grad, atol=1e-6, rtol=1e-6))
        self.assertTrue(compare_tensors("weight grad", conv.weight.grad, conv_arc.weight.grad, atol=1e-6, rtol=1e-6))

    def test_call_state_cache_is_bounded(self):
        """The least recently used call state goes once the plan holds max_call_states."""
        conv, _ = self._pair(4, 4, False, backend=_KeyedBackend)
        plan = conv.backend.plan
        t = plan.state(conv)
        for bc in range(1, plan.max_call_states + 2):
            plan.bind((("gather", bc),), torch.float32, (16, 32), t)
        self.assertEqual(len(plan._call_states), plan.max_call_states)
        self.assertNotIn(("gather", 1, torch.float32, (16, 32)), plan._call_states)
        # a hit refreshes, a miss rebuilds
        plan.bind((("gather", 2),), torch.float32, (16, 32), t)
        plan.bind((("gather", 1),), torch.float32, (16, 32), t)
        self.assertIn(("gather", 2, torch.float32, (16, 32)), plan._call_states)
        self.assertEqual(_KeyedPlan.built.count(("gather", 1, torch.float32)), 2)

    def test_call_state_is_not_built_during_capture(self):
        """A call that would build call state during CUDA graph capture raises; one that finds it cached does not."""
        conv, _ = self._pair(4, 4, False, backend=_KeyedBackend)
        plan = conv.backend.plan
        t = plan.state(conv)
        plan.bind((("gather", 8),), torch.float32, (16, 32), t)
        with mock.patch.object(torch.cuda, "is_available", return_value=True), mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            plan.bind((("gather", 8),), torch.float32, (16, 32), t)
            with self.assertRaisesRegex(RuntimeError, "during CUDA graph capture"):
                plan.bind((("gather", 16),), torch.float32, (16, 32), t)

    def test_contraction_binds_its_operations(self):
        """``contract`` binds the gather's call state over the input's planes, and the scatter's only when the gradient is needed."""
        conv, conv_arc = self._pair(4, 4, False, backend=_KeyedBackend)
        x = torch.randn(3, 4, 16, 32, device=self.device)
        y, y_arc = conv.backend.contract(conv, x), conv_arc.backend.contract(conv_arc, x)
        self.assertTrue(compare_tensors("contraction", y, y_arc, atol=1e-6, rtol=1e-6))
        self.assertEqual(_KeyedPlan.built, [("gather", 12, torch.float32)])

        x1, x2 = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
        y, y_arc = conv.backend.contract(conv, x1), conv_arc.backend.contract(conv_arc, x2)
        grad = torch.randn_like(y)
        y.backward(grad)
        y_arc.backward(grad)
        self.assertTrue(compare_tensors("input grad", x1.grad, x2.grad, atol=1e-6, rtol=1e-6))
        self.assertEqual(_KeyedPlan.built, [("gather", 12, torch.float32), ("scatter", 12, torch.float32)])
        self.assertIn(("scatter", 12, 12), _KeyedPlan.used)

    def test_conv_without_input_grad_binds_no_scatter(self):
        """Inference binds only the gather: no call state is built for a gradient that is never taken."""
        conv, _ = self._pair(4, 4, False, backend=_KeyedBackend)
        with torch.no_grad():
            conv(torch.randn(2, 4, 16, 32, device=self.device))
        self.assertEqual(_KeyedPlan.built, [("gather", 8, torch.float32)])

    @parameterized.expand([[False], [True]])
    @requires_torch_compile
    def test_keyed_plan_compiles(self, fused):
        """
        A keyed plan's conv traces as a whole. A compiled call that misses the cache builds its
        call state for itself without caching it; once an eager call has cached it, compiled
        calls use the cache.
        """
        conv, _ = self._pair(8, 2, fused, backend=_KeyedBackend)
        compiled = torch.compile(conv, backend="aot_eager", fullgraph=True)
        x = torch.randn(2, 8, 16, 32, device=self.device)
        grad = None
        results = []
        # compiled before any eager call, then eager, then compiled again
        for run in (compiled, conv, compiled):
            xi = x.clone().requires_grad_(True)
            out = run(xi)
            if grad is None:
                grad = torch.randn_like(out)
            out.backward(grad)
            results.append((out.detach(), xi.grad, conv.weight.grad.clone()))
            conv.weight.grad = None
            if run is compiled and len(results) == 1:
                self.assertEqual(len(conv.backend.plan._call_states), 0, "a compiled call must not fill the cache")
        self.assertEqual(len(conv.backend.plan._call_states), 2)
        for (out, gx, gw), name in zip(results[1:], ("eager", "compiled after eager")):
            self.assertTrue(compare_tensors(f"output, {name}", out, results[0][0], atol=1e-6, rtol=1e-6))
            self.assertTrue(compare_tensors(f"input grad, {name}", gx, results[0][1], atol=1e-6, rtol=1e-6))
            self.assertTrue(compare_tensors(f"weight grad, {name}", gw, results[0][2], atol=1e-6, rtol=1e-6))

    @requires_torch_compile
    def test_compiled_inference_does_not_poison_training(self):
        """Call state a compiled call builds under inference mode is not cached, so a later training call builds and trains."""
        conv, conv_arc = self._pair(4, 4, False, backend=_KeyedBackend)
        compiled = torch.compile(conv, backend="aot_eager", fullgraph=True)
        x = torch.randn(2, 4, 16, 32, device=self.device)
        with torch.inference_mode():
            compiled(x)
        self.assertEqual(len(conv.backend.plan._call_states), 0)

        x1, x2 = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
        out, out_arc = conv(x1), conv_arc(x2)
        grad = torch.randn_like(out)
        out.backward(grad)
        out_arc.backward(grad)
        self.assertTrue(compare_tensors("input grad", x1.grad, x2.grad, atol=1e-6, rtol=1e-6))
        self.assertFalse(any(s.is_inference() for state in conv.backend.plan._call_states.values() for s in state))

    @parameterized.expand([[False], [True]])
    @requires_torch_compile
    def test_drop_in_plan_compiles(self, fused):
        """The conv node traces as a whole with a plan's kernels in it, forward and backward."""
        conv, _ = self._pair(8, 2, fused)
        compiled = torch.compile(conv, backend="aot_eager", fullgraph=True)
        x = torch.randn(2, 8, 16, 32, device=self.device)
        x1, x2 = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
        out, out_compiled = conv(x1), compiled(x2)
        grad = torch.randn_like(out)
        out.backward(grad)
        weight_grad = conv.weight.grad.clone()
        conv.weight.grad = None
        out_compiled.backward(grad)
        self.assertTrue(compare_tensors("output", out_compiled, out, atol=1e-6, rtol=1e-6))
        self.assertTrue(compare_tensors("input grad", x2.grad, x1.grad, atol=1e-6, rtol=1e-6))
        self.assertTrue(compare_tensors("weight grad", conv.weight.grad, weight_grad, atol=1e-6, rtol=1e-6))


@parameterized_class(("device"), _devices)
class TestWeightContraction(unittest.TestCase):
    """The matmul weight contractions of the DISCO layers against einsum, including the outer-product cases."""

    @parameterized.expand(
        [
            # [batch, groups, in_per_group, out_per_group, kernel_size, spatial]
            [2, 1, 4, 6, 3, (5, 7)],
            [2, 3, 4, 2, 3, (5, 7)],
            [1, 4, 2, 3, 1, (11,)],
            # one in-channel per group and K = 1: the contraction is an outer product
            [2, 4, 1, 3, 1, (5, 7)],
            # one out-channel per group: the expansion is an outer product
            [2, 4, 3, 1, 2, (5, 7)],
        ],
        skip_on_empty=True,
    )
    def test_against_einsum(self, B, G, Cg, Og, K, spatial):
        set_seed(333)
        kw = dict(device=self.device, dtype=torch.float64)
        w = torch.randn(G, Og, Cg, K, **kw)
        x = torch.randn(B, G * Cg, K, *spatial, **kw)
        y = torch.randn(B, G * Og, *spatial, **kw)
        xg, yg = x.reshape(B, G, Cg, K, -1), y.reshape(B, G, Og, -1)

        out = torch.einsum("bgckn,gock->bgon", xg, w).reshape(B, G * Og, *spatial)
        self.assertTrue(torch.allclose(_weight_contraction(x, w), out))
        exp = torch.einsum("bgon,gock->bgckn", yg, w).reshape(B, G * Cg, K, *spatial)
        self.assertTrue(torch.allclose(_weight_expansion(y, w), exp))
        grad = torch.einsum("bgon,bgckn->gock", yg, xg)
        self.assertTrue(torch.allclose(_weight_gradient(y, x, G), grad))

    @parameterized.expand([[1], [2]], skip_on_empty=True)
    def test_gradient_cancels_across_batch(self, groups):
        """Half-precision partial sums over each batch entry overflow float16 but cancel across the batch: the gradient stays finite."""
        dtype = torch.float16
        # each batch entry contributes +-8 * 16384 = +-131072 to every weight entry, beyond float16's 65504
        x = torch.full((2, groups, 1, 8), 128.0, device=self.device, dtype=dtype)
        go = torch.full((2, groups, 8), 128.0, device=self.device, dtype=dtype)
        go[1] = -128.0
        grad = _weight_gradient(go, x, groups)
        self.assertTrue(torch.equal(grad, torch.zeros_like(grad)), f"{grad}")

    @parameterized.expand([[1, 3], [3, 1], [3, 3]], skip_on_empty=True)
    def test_autocast_keeps_float64(self, Cg, Og):
        """Autocast leaves float64 alone, in the outer-product cases as in the matmul."""
        w = torch.randn(2, Og, Cg, 1, device=self.device, dtype=torch.float64)
        x = torch.randn(2, 2 * Cg, 1, 4, 8, device=self.device, dtype=torch.float64)
        y = torch.randn(2, 2 * Og, 4, 8, device=self.device, dtype=torch.float64)
        out = torch.einsum("bgckn,gock->bgon", x.reshape(2, 2, Cg, 1, -1), w).reshape(2, 2 * Og, 4, 8)
        exp = torch.einsum("bgon,gock->bgckn", y.reshape(2, 2, Og, -1), w).reshape(2, 2 * Cg, 1, 4, 8)
        dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16
        with torch.autocast(self.device.type, dtype=dtype):
            got_out, got_exp = _weight_contraction(x, w), _weight_expansion(y, w)
        self.assertEqual((got_out.dtype, got_exp.dtype), (torch.float64, torch.float64))
        self.assertTrue(torch.allclose(got_out, out) and torch.allclose(got_exp, exp))

    @parameterized.expand([[1, 3], [3, 1], [3, 3]], skip_on_empty=True)
    def test_autocast_dtype(self, Cg, Og):
        """Under autocast the outer-product cases give the autocast dtype, as the matmul does."""
        if self.device.type != "cuda":
            raise unittest.SkipTest("autocast to float16 is checked on CUDA")
        w = torch.randn(2, Og, Cg, 1, device=self.device)
        with torch.autocast("cuda", dtype=torch.float16):
            self.assertEqual(_weight_contraction(torch.randn(2, 2 * Cg, 1, 4, 8, device=self.device), w).dtype, torch.float16)
            self.assertEqual(_weight_expansion(torch.randn(2, 2 * Og, 4, 8, device=self.device), w).dtype, torch.float16)


class TestKpackedDeviceGate(unittest.TestCase):
    """
    The kpacked images are arch-specific, not forward compatible: each device needs its own
    build target, and a 10.0a-only build must not select kpacked on a 10.3 device, which cannot
    load an sm_100a image (#337). The targets the extensions report and the device capability
    are mocked, so this runs without the hardware, and on every CI runner.
    """

    _TARGETS = {"9.0a": (9, 0), "10.0a": (10, 0), "10.3a": (10, 3)}

    def _supported(self, built, capability):
        with mock.patch.object(disco_arch, "built_targets", return_value=frozenset(built)), mock.patch("torch.cuda.get_device_capability", return_value=capability):
            return disco_arch.kpacked_supported_on_device(0)

    @parameterized.expand(
        [
            # build targets, device capability, kpacked selected
            [("10.0a",), (10, 0), True],
            [("10.0a",), (10, 3), False],
            [("10.3a",), (10, 3), True],
            [("10.3a",), (10, 0), False],
            [("10.0a", "10.3a"), (10, 0), True],
            [("10.0a", "10.3a"), (10, 3), True],
            [("10.0a", "10.3a"), (9, 0), False],
            [("9.0a",), (9, 0), True],
            [("9.0a",), (10, 0), False],
            # devices without a kpacked kernel: Rubin (10.7) is not a 10.0, nor are Ampere and RTX Blackwell
            [("9.0a", "10.0a", "10.3a"), (10, 7), False],
            [("9.0a", "10.0a", "10.3a"), (8, 0), False],
            [("9.0a", "10.0a", "10.3a"), (12, 0), False],
            [(), (10, 0), False],
        ]
    )
    def test_each_device_needs_its_own_target(self, built, capability, expected):
        self.assertEqual(self._supported(built, capability), expected)

    def test_the_table_covers_every_kpacked_target(self):
        self.assertEqual({target: capability for capability, (_, target) in disco_arch.DEVICE_TARGETS.items()}, self._TARGETS)
        # every family a device maps to has a kpacked forward and an extension
        self.assertTrue(all(family in disco_arch.KPACKED_OPS and family in disco_arch.EXTENSIONS for family, _ in disco_arch.DEVICE_TARGETS.values()))

    def test_any_target_builds_the_kpacked_buffers(self):
        # construction cannot know the device yet, so any built target is enough there
        for built in ((), ("9.0a",), ("10.0a",), ("10.3a",), ("9.0a", "10.0a", "10.3a")):
            with self.subTest(built=built), mock.patch.object(disco_arch, "built_targets", return_value=frozenset(built)):
                self.assertEqual(disco_arch.kpacked_build_available(), bool(built))

    def test_the_device_gets_its_own_op(self):
        """Each device is handed the kpacked op of its own family, and a device the build has no image for none."""
        # spelled out here rather than read from disco_arch's tables, so a wrong mapping there is caught
        expected = {(9, 0): "forward_kpacked_sm90", (10, 0): "forward_kpacked_sm100", (10, 3): "forward_kpacked_sm100"}
        ops = types.SimpleNamespace(**{name: types.SimpleNamespace(default=object()) for name in set(expected.values())})
        built = frozenset(self._TARGETS)
        for capability, name in expected.items():
            with (
                self.subTest(capability=capability),
                mock.patch.object(torch.ops, "disco_kernels", ops, create=True),
                mock.patch.object(disco_arch, "built_targets", return_value=built),
                mock.patch("torch.cuda.get_device_capability", return_value=capability),
            ):
                self.assertIs(disco_arch.kpacked_op(0), getattr(ops, name).default)
        with mock.patch.object(disco_arch, "built_targets", return_value=frozenset({"9.0a"})), mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)):
            with self.assertRaisesRegex(RuntimeError, "no kpacked kernel"):
                disco_arch.kpacked_op(0)


def _arc_state(conv):
    """The arc arrays of conv's backend, in the order the operators take them."""
    return tuple(getattr(conv, name) for name in ("psi_row_ker", "psi_row_lat", "psi_seg_off", "psi_seg", "psi_val_off", "psi_vals"))


def _without_kpacked(conv):
    """Reselect conv's backend with the kpacked one ruled out, so it runs the arc kernels."""
    conv._backends = (RegularOptimizedBackend, RegularReferenceBackend)
    conv._select_backend()
    return conv


@unittest.skipUnless(
    optimized_kernels_is_available() and torch.cuda.is_available(),
    "skipping kpacked tests: optimized kernels or CUDA not available",
)
class TestKpackedPath(unittest.TestCase):
    """Tests specific to the tensor-core kpacked forward, whose backward is the arc scatter."""

    device = torch.device("cuda")

    def _make_conv(self, batch, channels, in_shape, out_shape=None, theta_cutoff=0.05, grid_in="legendre-gauss", grid_out="legendre-gauss", fused=False):
        if out_shape is None:
            out_shape = in_shape
        conv = DiscreteContinuousConvS2(
            grid_in=as_grid(grid_in, nlat=in_shape[0], nlon=in_shape[1]),
            grid_out=as_grid(grid_out, nlat=out_shape[0], nlon=out_shape[1]),
            in_channels=channels,
            out_channels=channels,
            kernel_shape=(3, 3),
            basis_type="harmonic",
            basis_norm_mode="nodal",
            groups=1,
            bias=False,
            theta_cutoff=theta_cutoff,
            fused=fused,
        ).to(device=self.device, dtype=torch.bfloat16)
        return conv

    @unittest.skipUnless(_kpacked_built_for_sm90(), "kpacked forward requires SM_90a (Hopper) and an sm_90a build")
    def test_kpacked_forward_activates_on_sm90(self):
        """forward_kpacked_sm90 is chosen for bf16/fp16 on Hopper."""
        conv = self._make_conv(1, 8, (16, 32))
        self.assertEqual(conv.backend.name, "regular-kpacked", "the harmonic basis should select the kpacked backend")
        self.assertIn(conv.psi_kpacked_vals.shape[1], (8, 16), "K_pad must be 8 or 16 for the WGMMA kernel")
        inp = torch.randn(1, 8, 16, 32, dtype=torch.bfloat16, device=self.device)
        out = conv(inp)
        self.assertEqual(out.dtype, torch.bfloat16)

    @unittest.skipUnless(_kpacked_built_for_sm100(), "kpacked forward on Blackwell requires a 10.0 or 10.3 device and the matching 10.0a / 10.3a build")
    def test_kpacked_forward_activates_on_sm100(self):
        """tcgen05 kpacked path is chosen for bf16/fp16 on Blackwell."""
        conv = self._make_conv(1, 8, (16, 32))
        self.assertEqual(conv.backend.name, "regular-kpacked", "the harmonic basis should select the kpacked backend")
        self.assertIn(conv.psi_kpacked_vals.shape[1], (8, 16), "K_pad must be 8 or 16 for the tcgen05 kernel")
        inp = torch.randn(1, 8, 16, 32, dtype=torch.bfloat16, device=self.device)
        out = conv(inp)
        self.assertEqual(out.dtype, torch.bfloat16)

    @unittest.skipUnless(_is_kpacked_supported(), "kpacked forward requires SM_90a, SM_100a or SM_103a")
    def test_kpacked_matches_optimized(self, verbose=True):
        """The kpacked tensor-core forward matches the arc kernels numerically."""
        set_seed(123)
        in_shape = (16, 32)
        conv_kpacked = self._make_conv(1, 8, in_shape).float()
        conv_opt = _without_kpacked(self._make_conv(1, 8, in_shape).float())
        self.assertEqual(conv_kpacked.backend.name, "regular-kpacked", "kpacked reference test requires the kpacked backend")
        self.assertEqual(conv_opt.backend.name, "regular-optimized")

        conv_opt.weight.data.copy_(conv_kpacked.weight.data)

        inp = torch.randn(1, 8, *in_shape, dtype=torch.float32, device=self.device, requires_grad=True)
        inp_ref = inp.detach().clone().requires_grad_(True)

        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            out_kpacked = conv_kpacked(inp)
            out_opt = conv_opt(inp_ref)
        self.assertTrue(compare_tensors("output", out_kpacked.float(), out_opt.float(), atol=5e-2, rtol=5e-2))

        grad = torch.randn_like(out_kpacked)
        out_kpacked.backward(grad)
        out_opt.backward(grad.clone())

        self.assertTrue(compare_tensors("inp grad", inp.grad.float(), inp_ref.grad.float(), atol=5e-2, rtol=5e-2, verbose=verbose))
        self.assertTrue(compare_tensors("weight grad", conv_kpacked.weight.grad.float(), conv_opt.weight.grad.float(), atol=5e-2, rtol=5e-2, verbose=verbose))

    @unittest.skipUnless(_is_kpacked_supported(), "kpacked forward requires SM_90a, SM_100a or SM_103a")
    def test_kpacked_contraction_double_backward(self):
        """The kpacked contraction's backward is the transpose op under create_graph, so it is differentiable: its derivative is the arc contraction."""
        from torch_harmonics.disco.optimized.disco_optimized import _disco_s2_contraction_regular_optimized

        set_seed(7)
        conv = self._make_conv(1, 8, (16, 32))
        self.assertEqual(conv.backend.name, "regular-kpacked")
        arcs = _arc_state(conv)

        inp = torch.randn(1, 8, 16, 32, dtype=torch.bfloat16, device=self.device, requires_grad=True)
        # the backend's contraction: the kpacked plan's gather through the planned node
        out = conv.backend.contract(conv, inp)
        grad = torch.randn_like(out, requires_grad=True)
        grad_inp = torch.autograd.grad(out, inp, grad, create_graph=True)[0]

        # grad_inp = J^T grad is linear in grad, so differentiating <grad_inp, u> by grad gives J u
        u = torch.randn_like(inp)
        ggrad = torch.autograd.grad((grad_inp * u).sum(), grad)[0]
        ref = _disco_s2_contraction_regular_optimized(u, *arcs, conv.kernel_size, conv.nlat_out, conv.nlon_out)
        self.assertTrue(compare_tensors("double backward", ggrad.float(), ref.float(), atol=1e-2, rtol=1e-2))

    @unittest.skipUnless(_is_kpacked_supported(), "kpacked forward requires SM_90a, SM_100a or SM_103a")
    def test_kpacked_fused_matches_unfused(self):
        """
        fused=True kpacked path gives the same result as fused=False.

        The weight gradient is the part that reads the recomputed intermediate. The recompute
        runs the forward's own kpacked contraction on the saved input, so it reproduces the
        intermediate the unfused layer saved bit for bit, and the weight gradients are equal,
        not merely close. A recompute on another kernel (the arc gather) would differ.
        """
        set_seed(42)
        conv_unfused = self._make_conv(1, 8, (16, 32), fused=False)
        conv_fused = self._make_conv(1, 8, (16, 32), fused=True)
        self.assertEqual(conv_fused.backend.name, "regular-kpacked")
        # share weights so outputs are identical
        conv_fused.weight.data.copy_(conv_unfused.weight.data)

        inp = torch.randn(1, 8, 16, 32, dtype=torch.bfloat16, device=self.device, requires_grad=True)
        inp2 = inp.detach().clone().requires_grad_(True)

        out_u = conv_unfused(inp)
        out_f = conv_fused(inp2)
        self.assertTrue(compare_tensors("output", out_u, out_f, atol=1e-2, rtol=1e-2))

        grad = torch.randn_like(out_u)
        out_u.backward(grad)
        out_f.backward(grad.clone())
        self.assertTrue(compare_tensors("inp grad", inp.grad, inp2.grad, atol=1e-2, rtol=1e-2))
        self.assertTrue(torch.equal(conv_unfused.weight.grad, conv_fused.weight.grad), "the fused recompute must reproduce the saved intermediate exactly")

    @unittest.skipUnless(_is_kpacked_supported(), "kpacked forward requires SM_90a, SM_100a or SM_103a")
    def test_kpacked_bwd_bc_tile_boundaries(self):
        """BC_TILE selection: exercises BC_TILE=1 (BC=3), BC_TILE=4 (BC=5), BC_TILE=8 (BC=16).

        All three should produce the same gradients as fp32 (within bf16 tolerance),
        confirming the tail-CTA zero-padding path is correct.
        """
        in_shape = (16, 32)
        for batch, channels, expected_bc_tile in [(3, 1, 1), (1, 5, 4), (2, 8, 8)]:
            with self.subTest(batch=batch, channels=channels, bc_tile=expected_bc_tile):
                set_seed(0)
                conv_bf16 = self._make_conv(batch, channels, in_shape)
                conv_fp32 = DiscreteContinuousConvS2(
                    grid_in=as_grid("legendre-gauss", nlat=in_shape[0], nlon=in_shape[1]),
                    grid_out=as_grid("legendre-gauss", nlat=in_shape[0], nlon=in_shape[1]),
                    in_channels=channels,
                    out_channels=channels,
                    kernel_shape=(3, 3),
                    basis_type="harmonic",
                    basis_norm_mode="nodal",
                    groups=1,
                    bias=False,
                    theta_cutoff=0.05,
                ).to(device=self.device, dtype=torch.float32)
                conv_fp32.weight.data.copy_(conv_bf16.weight.data.float())

                inp_bf16 = torch.randn(batch, channels, *in_shape, dtype=torch.bfloat16, device=self.device, requires_grad=True)
                inp_fp32 = inp_bf16.detach().float().requires_grad_(True)
                grad = torch.randn(batch, channels, *in_shape, dtype=torch.bfloat16, device=self.device)

                conv_bf16(inp_bf16).backward(grad)
                conv_fp32(inp_fp32).backward(grad.float())

                self.assertTrue(compare_tensors("inp grad", inp_bf16.grad.float(), inp_fp32.grad, atol=1e-1, rtol=1e-1))

    def test_kpacked_disabled_for_unsupported_k_pad(self):
        """A basis the kpacked kernels have no instantiation for must select the optimized backend, not crash."""
        # the kernels take K as the MMA's N dimension, instantiated for N = 8 and 16
        self.assertEqual(_kpacked_k_pad(3), 8)
        self.assertEqual(_kpacked_k_pad(15), 16)
        self.assertIsNone(_kpacked_k_pad(20))

        conv = _without_kpacked(self._make_conv(1, 4, (16, 32)))
        inp = torch.randn(1, 4, 16, 32, dtype=torch.bfloat16, device=self.device)
        out = conv(inp)
        self.assertEqual(out.shape[0], 1)
        self.assertFalse(any(name.startswith("psi_kpacked") for name in conv._backend_state), "the optimized backend must not hold the kpacked layout")

    def test_kpacked_disabled_fused_fallback(self):
        """fused=True on the optimized backend must match fused=False."""
        set_seed(77)
        conv_unfused = self._make_conv(1, 8, (16, 32), fused=False)
        conv_fused = self._make_conv(1, 8, (16, 32), fused=True)
        conv_fused.weight.data.copy_(conv_unfused.weight.data)

        # Rule kpacked out on both so both take the arc kernels.
        _without_kpacked(conv_unfused)
        _without_kpacked(conv_fused)

        inp = torch.randn(1, 8, 16, 32, dtype=torch.bfloat16, device=self.device, requires_grad=True)
        inp2 = inp.detach().clone().requires_grad_(True)

        out_u = conv_unfused(inp)
        out_f = conv_fused(inp2)
        self.assertTrue(compare_tensors("output", out_u, out_f, atol=1e-3, rtol=1e-3))

        grad = torch.randn_like(out_u)
        out_u.backward(grad)
        out_f.backward(grad.clone())
        self.assertTrue(compare_tensors("inp grad", inp.grad, inp2.grad, atol=1e-3, rtol=1e-3))

    @unittest.skipUnless(_is_kpacked_supported(), "kpacked forward requires SM_90a, SM_100a or SM_103a")
    def test_kpacked_opcheck(self):
        """The kpacked op of this device's extension satisfies the PT2 opcheck contract."""
        conv = self._make_conv(1, 8, (16, 32))
        inp = torch.randn(1, 8, 16, 32, dtype=torch.bfloat16, device=self.device)
        test_inputs = (
            inp,
            conv.psi_kpacked_idx,
            conv.psi_kpacked_vals,
            conv.psi_kpacked_offset,
            conv.kernel_size,
            conv.nlat_out,
            conv.nlon_out,
        )
        opcheck(disco_arch.kpacked_op(torch.cuda.current_device()), test_inputs)


# ---------------------------------------------------------------------------
# Ragged grids (HEALPix, and a regular grid paired with one)
# ---------------------------------------------------------------------------


def _grid(spec):
    """``("healpix", nside)``, ``("product-as-ragged", nlat, nlon)`` or ``(grid_type, nlat, nlon)``."""
    if spec[0] == "healpix":
        return HealpixGrid(nside=spec[1])
    if spec[0] == "product-as-ragged":
        return _ProductGridAsRagged(nlat=spec[1], nlon=spec[2])
    return as_grid(spec[0], nlat=spec[1], nlon=spec[2])


def _field(batch_size, channels, grid, dtype, device):
    """A random field in the layout the layer takes on that grid."""
    # the layer's criterion: only a RegularGridS2 has the (nlat, nlon) layout
    shape = grid.shape if isinstance(grid, RegularGridS2) else (grid.npoints,)
    return torch.randn(batch_size, channels, *shape, dtype=dtype, device=device)


@parameterized_class(("device"), _devices)
class TestDiscreteContinuousConvRaggedS2(unittest.TestCase):
    """DISCO on ragged grids: the per-point psi, the ragged backends and their agreement."""

    def setUp(self):
        set_seed(333)

    def _make(
        self,
        transpose,
        grid_in,
        grid_out,
        kernel_shape=(3, 4),
        basis_type="piecewise linear",
        basis_norm_mode="nodal",
        optimized_kernel=True,
        dtype=torch.float32,
        in_channels=4,
        out_channels=2,
        **kwargs,
    ):
        Conv = DiscreteContinuousConvTransposeS2 if transpose else DiscreteContinuousConvS2
        return Conv(
            _grid(grid_in),
            _grid(grid_out),
            in_channels,
            out_channels,
            kernel_shape,
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            bias=True,
            optimized_kernel=optimized_kernel,
            **kwargs,
        ).to(dtype=dtype, device=self.device)

    @parameterized.expand(
        [
            [False, ("healpix", 4), ("healpix", 4)],
            [True, ("healpix", 4), ("healpix", 4)],
            [False, ("healpix", 4), ("equiangular", 9, 16)],
            [False, ("equiangular", 9, 16), ("healpix", 4)],
            [True, ("legendre-gauss", 8, 16), ("healpix", 4)],
        ]
    )
    def test_layout_and_backend(self, transpose, grid_in, grid_out):
        """A ragged side is flat, a regular side keeps (nlat, nlon), and a ragged backend serves the layer."""
        conv = self._make(transpose, grid_in, grid_out)
        self.assertTrue(conv.ragged)
        expected = "ragged-optimized" if optimized_kernels_is_available() else "ragged-reference"
        self.assertEqual(conv.backend.name, expected)

        inp = _field(2, 4, conv.grid_in, torch.float32, self.device)
        out = conv(inp)
        out_shape = (conv.grid_out.npoints,) if conv.ragged_out else conv.grid_out.shape
        self.assertEqual(tuple(out.shape), (2, 2, *out_shape))

    def test_regular_layers_stay_regular(self):
        """A pair of regular grids keeps the regular backends, whatever the ragged ones would accept."""
        conv = self._make(False, ("equiangular", 9, 16), ("equiangular", 9, 16))
        self.assertFalse(conv.ragged)
        self.assertTrue(conv.backend.name.startswith("regular-"))

    @parameterized.expand([["regular"], ["ragged"]])
    def test_optimized_only_on_devices_with_a_kernel(self, family):
        """
        The compiled backend accepts only a device this build has a kernel for, so a layer
        moved to MPS, or to CUDA under a CPU-only build, takes the reference instead of
        failing in the dispatcher on its first forward. A build without a kernel for this
        device is simulated by emptying the backend's device set.
        """
        grid = ("healpix", 4) if family == "ragged" else ("equiangular", 9, 16)
        optimized, reference = (RaggedOptimizedBackend, RaggedReferenceBackend) if family == "ragged" else (RegularOptimizedBackend, RegularReferenceBackend)
        conv = self._make(False, grid, grid)

        self.assertFalse(optimized.available(conv, torch.device("mps")))
        self.assertEqual(optimized.available(conv, torch.device("cuda")), optimized_kernels_is_available() and cuda_kernels_is_available())
        self.assertEqual(optimized.available(conv, torch.device("cpu")), optimized_kernels_is_available())
        self.assertTrue(reference.available(conv, torch.device("mps")))

        devices = "_RAGGED_DEVICES" if family == "ragged" else "_REGULAR_DEVICES"
        with mock.patch.object(disco_backends, devices, frozenset()):
            conv = self._make(False, grid, grid)
        self.assertEqual(conv.backend.name, reference.name)

    @parameterized.expand(
        [
            # transpose, grid_in, grid_out, in_channels, out_channels, kernel_shape, basis_type, basis_norm_mode, dtype, fused, atol, rtol
            [False, ("healpix", 4), ("healpix", 4), 4, 2, (3, 4), "piecewise linear", "nodal", torch.float32, False, 1e-5, 1e-4],
            [False, ("healpix", 8), ("healpix", 4), 4, 2, (2, 3), "harmonic", "modal", torch.float32, False, 1e-5, 1e-4],
            [False, ("healpix", 4), ("equiangular", 9, 16), 4, 2, 3, "zernike", "mean", torch.float32, False, 1e-5, 1e-4],
            [False, ("equiangular", 17, 32), ("healpix", 4), 4, 2, (3, 4), "piecewise linear", "support", torch.float32, False, 1e-5, 1e-4],
            [False, ("healpix", 8), ("healpix", 4), 4, 4, (3, 4), "piecewise linear", "nodal", torch.float32, False, 1e-5, 1e-4],
            [True, ("healpix", 4), ("healpix", 4), 4, 2, (3, 4), "piecewise linear", "nodal", torch.float32, False, 1e-5, 1e-4],
            [True, ("healpix", 4), ("healpix", 8), 4, 2, (2, 3), "harmonic", "mean", torch.float32, False, 1e-5, 1e-4],
            [True, ("healpix", 4), ("equiangular", 17, 32), 4, 2, (3, 4), "piecewise linear", "geometric", torch.float32, False, 1e-5, 1e-4],
            [False, ("healpix", 8), ("healpix", 4), 4, 2, (3, 4), "piecewise linear", "nodal", torch.float64, False, 1e-10, 1e-10],
            [True, ("healpix", 4), ("healpix", 8), 4, 2, (2, 3), "harmonic", "modal", torch.float64, False, 1e-10, 1e-10],
            [False, ("healpix", 4), ("healpix", 4), 4, 2, (3, 4), "piecewise linear", "nodal", torch.float16, False, 5e-2, 1e-2],
            [True, ("healpix", 4), ("healpix", 8), 4, 2, (2, 3), "harmonic", "mean", torch.bfloat16, False, 5e-2, 5e-2],
            # fused, with the spatial-first input gradient (2 * out_channels <= in_channels) and without
            [False, ("healpix", 8), ("healpix", 4), 8, 2, (3, 4), "piecewise linear", "nodal", torch.float32, True, 1e-5, 1e-4],
            [False, ("healpix", 8), ("healpix", 4), 4, 4, (2, 3), "harmonic", "modal", torch.float32, True, 1e-5, 1e-4],
            [False, ("healpix", 4), ("equiangular", 9, 16), 8, 2, (3, 4), "piecewise linear", "nodal", torch.float32, True, 1e-5, 1e-4],
            [False, ("equiangular", 17, 32), ("healpix", 4), 4, 4, (3, 4), "piecewise linear", "nodal", torch.float64, True, 1e-10, 1e-10],
            [False, ("healpix", 4), ("healpix", 4), 8, 2, (3, 4), "piecewise linear", "nodal", torch.bfloat16, True, 5e-2, 5e-2],
        ]
    )
    @unittest.skipUnless(optimized_kernels_is_available(), "skipping test because optimized kernels are not available")
    def test_optimized_against_torch(
        self, transpose, grid_in, grid_out, in_channels, out_channels, kernel_shape, basis_type, basis_norm_mode, dtype, fused, atol, rtol, verbose=True
    ):
        """The compiled ragged kernels against the sparse torch reference: output and both gradients."""
        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

        # for AMP dtypes, the module and input stay in float32; autocast handles the rest
        is_amp = dtype in (torch.float16, torch.bfloat16)
        module_dtype = torch.float32 if is_amp else dtype

        kw = dict(kernel_shape=kernel_shape, basis_type=basis_type, basis_norm_mode=basis_norm_mode, dtype=module_dtype, in_channels=in_channels, out_channels=out_channels)
        # fused is only supported for forward (non-transpose) convolution
        fused_kwarg = {"fused": fused} if (fused and not transpose) else {}
        conv_naive = self._make(transpose, grid_in, grid_out, optimized_kernel=False, **kw)
        conv_opt = self._make(transpose, grid_in, grid_out, optimized_kernel=True, **kw, **fused_kwarg)
        self.assertEqual(conv_naive.backend.name, "ragged-reference")
        self.assertEqual(conv_opt.backend.name, "ragged-optimized")

        with torch.no_grad():
            conv_naive.weight.copy_(conv_opt.weight)

        inp = _field(3, in_channels, conv_opt.grid_in, module_dtype, self.device)

        inp.requires_grad = True
        with maybe_autocast(self.device.type, dtype):
            out_naive = conv_naive(inp)
        grad_output = torch.randn_like(out_naive)
        out_naive.backward(grad_output)
        inp_grad_naive = inp.grad.clone()

        inp.grad = None
        with maybe_autocast(self.device.type, dtype):
            out_opt = conv_opt(inp)
        out_opt.backward(grad_output)
        inp_grad_opt = inp.grad.clone()

        # the weight gradient is a sum over batch and points, see param_grad_ulps
        pg_ulps = param_grad_ulps(inp.shape[0] * max(conv_opt.grid_in.npoints, conv_opt.grid_out.npoints), dtype)
        self.assertTrue(compare_tensors("output", out_opt, out_naive, atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors("input grad", inp_grad_opt, inp_grad_naive, atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors("weight grad", conv_opt.weight.grad, conv_naive.weight.grad, atol=atol, rtol=rtol, verbose=verbose, ulps=pg_ulps, eps_dtype=dtype))

    @parameterized.expand(
        [
            # transpose, (nlat_in, nlon_in), (nlat_out, nlon_out), kernel_shape, basis_type, basis_norm_mode, optimized_kernel
            [False, (17, 32), (17, 32), (3, 4), "piecewise linear", "nodal", True],
            [False, (17, 32), (9, 16), (2, 3), "harmonic", "modal", True],
            [False, (17, 32), (9, 32), 3, "zernike", "support", False],
            [False, (17, 32), (9, 16), (3, 4), "piecewise linear", "support", True],
            [False, (17, 32), (17, 32), (2, 3), "piecewise linear", "modal", False],
            [True, (17, 32), (17, 32), (3, 4), "piecewise linear", "nodal", True],
            [True, (9, 32), (17, 32), (2, 3), "harmonic", "modal", True],
            # a transpose across differing nlon matches only in the modes that do not
            # normalize per group: the regular path pools the column phases of a latitude
            [True, (9, 16), (17, 32), (3, 4), "piecewise linear", "mean", True],
            [True, (9, 16), (17, 32), (2, 3), "harmonic", "geometric", False],
        ]
    )
    def test_ragged_path_matches_regular(self, transpose, in_shape, out_shape, kernel_shape, basis_type, basis_norm_mode, optimized_kernel, verbose=True):
        """
        Identical geometry down both paths: an equiangular grid as itself and presented as
        ragged. The per-point psi has to reproduce the p-shifted one, so any difference is
        the ragged precompute or the ragged kernels.
        """
        if optimized_kernel and not optimized_kernels_is_available():
            raise unittest.SkipTest("skipping test because optimized kernels are not available")
        if (self.device.type == "cuda") and optimized_kernel and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

        kw = dict(kernel_shape=kernel_shape, basis_type=basis_type, basis_norm_mode=basis_norm_mode, optimized_kernel=optimized_kernel, dtype=torch.float64)
        conv_reg = self._make(transpose, ("equiangular", *in_shape), ("equiangular", *out_shape), **kw)
        conv_rag = self._make(transpose, ("product-as-ragged", *in_shape), ("product-as-ragged", *out_shape), **kw)
        self.assertFalse(conv_reg.ragged)
        self.assertTrue(conv_rag.ragged)

        with torch.no_grad():
            conv_rag.weight.copy_(conv_reg.weight)
            conv_rag.bias.copy_(conv_reg.bias)

        inp = torch.randn(2, 4, *in_shape, dtype=torch.float64, device=self.device, requires_grad=True)
        out_reg = conv_reg(inp)
        grad_output = torch.randn_like(out_reg)
        out_reg.backward(grad_output)
        inp_grad_reg = inp.grad.clone()

        inp.grad = None
        out_rag = conv_rag(inp.flatten(-2, -1)).unflatten(-1, out_shape)
        out_rag.backward(grad_output)
        inp_grad_rag = inp.grad.clone()

        self.assertTrue(compare_tensors("output", out_reg, out_rag, atol=1e-10, rtol=1e-8, verbose=verbose))
        self.assertTrue(compare_tensors("input grad", inp_grad_reg, inp_grad_rag, atol=1e-10, rtol=1e-8, verbose=verbose))
        self.assertTrue(compare_tensors("weight grad", conv_reg.weight.grad, conv_rag.weight.grad, atol=1e-10, rtol=1e-8, verbose=verbose))

    @parameterized.expand([[False], [True]])
    @unittest.skipUnless(optimized_kernels_is_available(), "skipping test because optimized kernels are not available")
    def test_ragged_opcheck(self, transpose):
        """The ragged custom ops satisfy the PT2 opcheck contract: schema, fakes and autograd registration."""
        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

        conv = self._make(transpose, ("healpix", 2), ("healpix", 2))
        arcs = conv.backend._arcs(conv)
        if transpose:
            op = torch.ops.disco_kernels._disco_s2_transpose_contraction_ragged_optimized
            inp = torch.randn(2, 3, conv.kernel_size, conv.npoints_in, device=self.device, requires_grad=True)
        else:
            op = torch.ops.disco_kernels._disco_s2_contraction_ragged_optimized
            inp = torch.randn(2, 3, conv.npoints_in, device=self.device, requires_grad=True)
        opcheck(op, (inp, *arcs, conv.kernel_size, conv.npoints_out))

    @parameterized.expand(
        [
            # transpose, fused
            [False, False],
            [False, True],
            [True, False],
        ]
    )
    @unittest.skipUnless(optimized_kernels_is_available(), "skipping test because optimized kernels are not available")
    @requires_torch_compile
    def test_ragged_pt2_compatibility(self, transpose, fused, verbose=False):
        """The ragged convolution traces as a whole, forward and backward in one graph, and agrees with eager."""
        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

        fused_kwarg = {"fused": fused} if not transpose else {}
        conv = self._make(transpose, ("healpix", 4), ("healpix", 2) if not transpose else ("healpix", 8), **fused_kwarg)
        self.assertEqual(conv.backend.name, "ragged-optimized")

        # aot_eager exercises the fake kernels and the joint graph without needing a codegen toolchain
        compiled = torch.compile(conv, backend="aot_eager", fullgraph=True)
        inp = _field(2, 4, conv.grid_in, torch.float32, self.device)
        inp_eager = inp.clone().requires_grad_(True)
        inp_compiled = inp.clone().requires_grad_(True)
        out_eager = conv(inp_eager)
        out_compiled = compiled(inp_compiled)
        self.assertTrue(compare_tensors("output", out_compiled, out_eager, atol=1e-5, rtol=1e-5, verbose=verbose))
        grad = torch.randn_like(out_eager)
        out_eager.backward(grad)
        weight_grad_eager = conv.weight.grad.clone()
        conv.weight.grad = None
        out_compiled.backward(grad)
        self.assertTrue(compare_tensors("input grad", inp_compiled.grad, inp_eager.grad, atol=1e-5, rtol=1e-5, verbose=verbose))
        self.assertTrue(compare_tensors("weight grad", conv.weight.grad, weight_grad_eager, atol=1e-5, rtol=1e-5, verbose=verbose))

    @parameterized.expand([[False], [True]])
    @unittest.skipUnless(optimized_kernels_is_available(), "skipping test because optimized kernels are not available")
    def test_no_input_grad(self, transpose):
        """With an input that does not require grad, backward leaves it alone and still reaches the weight."""
        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

        conv = self._make(transpose, ("healpix", 4), ("healpix", 4))
        inp = _field(2, 4, conv.grid_in, torch.float32, self.device)
        conv(inp).sum().backward()
        self.assertIsNone(inp.grad)
        self.assertIsNotNone(conv.weight.grad)
        for name in conv._backend_state:
            self.assertIsNone(getattr(conv, name).grad, f"buffer {name} should not accumulate a gradient")

    @parameterized.expand(
        [
            # (grid_in, grid_out, in_channels, out_channels, groups, transpose, fused, optimized_kernel, contiguous)
            # the optimized convolution: weight-first (4->4) and spatial-first (8->2) input
            # gradient, saved and recomputed (fused) intermediate, groups, a strided input
            [("healpix", 2), ("healpix", 2), 4, 4, 1, False, False, True, True],
            [("healpix", 2), ("healpix", 2), 8, 2, 1, False, False, True, True],
            [("healpix", 2), ("healpix", 2), 4, 4, 1, False, True, True, True],
            [("healpix", 2), ("healpix", 2), 8, 2, 1, False, True, True, True],
            [("healpix", 2), ("healpix", 2), 4, 6, 2, False, False, True, True],
            [("healpix", 2), ("healpix", 2), 4, 4, 1, False, False, True, False],
            # a regular grid on one side
            [("equiangular", 9, 16), ("healpix", 2), 4, 4, 1, False, True, True, True],
            # the torch reference, and the transpose convolution
            [("healpix", 2), ("healpix", 2), 4, 4, 1, False, False, False, True],
            [("healpix", 2), ("healpix", 2), 8, 2, 1, False, True, False, True],
            [("healpix", 2), ("healpix", 2), 4, 4, 1, True, False, True, True],
            [("healpix", 2), ("equiangular", 9, 16), 4, 4, 1, True, False, True, True],
            [("healpix", 2), ("healpix", 2), 4, 4, 1, True, False, False, True],
        ]
    )
    def test_double_backward(self, grid_in, grid_out, in_channels, out_channels, groups, transpose, fused, optimized_kernel, contiguous, verbose=False):
        """Gradients taken with create_graph=True are differentiable, and equal the ordinary ones."""
        if optimized_kernel and not optimized_kernels_is_available():
            raise unittest.SkipTest("skipping test because optimized kernels are not available")
        if optimized_kernel and (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

        fused_kwarg = {} if transpose else {"fused": fused}
        conv = self._make(
            transpose,
            grid_in,
            grid_out,
            kernel_shape=(3),
            optimized_kernel=optimized_kernel,
            dtype=torch.float64,
            in_channels=in_channels,
            out_channels=out_channels,
            groups=groups,
            **fused_kwarg,
        )
        self.assertTrue(conv.ragged)

        inp = _field(1, in_channels, conv.grid_in, torch.float64, self.device)
        if not contiguous:
            inp = inp.transpose(1, -1).contiguous().transpose(1, -1)
            self.assertFalse(inp.is_contiguous())
        inp.requires_grad_(True)
        weight = conv.weight.detach().clone().requires_grad_(True)

        def fn(x, w):
            return torch.func.functional_call(conv, {"weight": w}, (x,))

        # the create_graph backward is a different code path: it must give the same gradients
        grad = torch.randn_like(fn(inp, weight))
        plain = torch.autograd.grad(fn(inp, weight), (inp, weight), grad)
        graphed = torch.autograd.grad(fn(inp, weight), (inp, weight), grad, create_graph=True)
        self.assertTrue(compare_tensors("input grad", graphed[0], plain[0], atol=1e-12, rtol=1e-10, verbose=verbose))
        self.assertTrue(compare_tensors("weight grad", graphed[1], plain[1], atol=1e-12, rtol=1e-10, verbose=verbose))

        # the CUDA scatter kernel accumulates with atomics, so two backward passes differ
        # in the last bits; gradcheck's reentrancy check wants them bit-identical otherwise
        self.assertTrue(torch.autograd.gradgradcheck(fn, (inp, weight), nondet_tol=1e-12))

    def test_ragged_arcs_roundtrip(self):
        """build_arcs on ring tables is lossless: expanding the arcs gives back psi's entries."""
        conv = self._make(False, ("healpix", 4), ("healpix", 4))
        ker, row, col, vals = conv._psi_coo()
        grid = conv.grid_in
        ring_base, ring_size = grid.lon_offsets[:-1].to(torch.int64), grid.nlon_per_lat.to(torch.int64)
        arcs = build_arcs(ker, row, col, vals, ring_base=ring_base, ring_size=ring_size)

        def canon(k, r, c, v):
            key = (k * conv.npoints_out + r) * conv.npoints_in + c
            order = torch.argsort(key)
            return key[order], v[order]

        key_ref, vals_ref = canon(ker, row, col, vals)
        key_arc, vals_arc = canon(*arcs_to_coo(arcs, ring_base=ring_base, ring_size=ring_size))
        self.assertTrue(torch.equal(key_ref, key_arc))
        self.assertTrue(torch.equal(vals_ref, vals_arc))


class TestDiscreteContinuousConvWithoutFloat64Device(unittest.TestCase):
    """The torch reference backends on a device without float64 (MPS), where psi must be float32."""

    _GRIDS = {"regular": (("equiangular", 9, 16), ("equiangular", 5, 8)), "ragged": (("healpix", 4), ("healpix", 4))}

    def setUp(self):
        set_seed(333)

    def _conv(self, transpose, family):
        grid_in, grid_out = self._GRIDS[family]
        Conv = DiscreteContinuousConvTransposeS2 if transpose else DiscreteContinuousConvS2
        if transpose:
            grid_in, grid_out = grid_out, grid_in
        return Conv(_grid(grid_in), _grid(grid_out), 3, 2, (3, 4), basis_type="piecewise linear", optimized_kernel=False)

    @parameterized.expand([[False, "regular"], [True, "regular"], [False, "ragged"], [True, "ragged"]])
    def test_reference_psi_follows_layer_dtype(self, transpose, family):
        """The reference psi is float64 for a float64 layer and float32 again after .float()."""
        conv = self._conv(transpose, family).double()
        self.assertTrue(conv.backend.name.endswith("-reference"))
        self.assertEqual(conv.psi.dtype, torch.float64)
        conv = conv.float()
        self.assertTrue(conv.backend.name.endswith("-reference"))
        self.assertEqual(conv.psi.dtype, torch.float32)

    @parameterized.expand([[False, "regular"], [True, "regular"], [False, "ragged"], [True, "ragged"]])
    @unittest.skipUnless(torch.backends.mps.is_available(), "needs an MPS device")
    def test_conv_moves_to_mps(self, transpose, family):
        """A layer moved to MPS selects the reference and agrees with its CPU result."""
        conv = self._conv(transpose, family)
        inp = _field(2, 3, conv.grid_in, torch.float32, "cpu")
        expected = conv(inp)
        conv = conv.to("mps")
        self.assertTrue(conv.backend.name.endswith("-reference"))
        out = conv(inp.to("mps")).cpu()
        self.assertTrue(compare_tensors("mps output", out, expected, atol=1e-5, rtol=1e-4))
