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
import unittest

import torch
from parameterized import parameterized, parameterized_class
from testutils import _is_sm90, _is_sm100, compare_tensors, disable_tf32, maybe_autocast, set_seed
from torch.library import opcheck

from torch_harmonics import DiscreteContinuousConvS2, DiscreteContinuousConvTransposeS2, as_grid
from torch_harmonics.disco import cuda_kernels_is_available, optimized_kernels_is_available
from torch_harmonics.disco._psi import arcs_to_coo, build_arcs, build_kpacked
from torch_harmonics.disco.backends import OptimizedBackend, ReferenceBackend
from torch_harmonics.disco.convolution import (
    _precompute_convolution_tensor_s2,
)
from torch_harmonics.disco.optimized.disco_optimized import _kpacked_k_pad, _kpacked_supported_on_device
from torch_harmonics.filter_basis import get_filter_basis
from torch_harmonics.quadrature import compute_theta_cutoff, precompute_latitudes, precompute_longitudes

if not optimized_kernels_is_available():
    print("Warning: Couldn't import optimized disco convolution kernels")


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

            ker_idx, row_idx, col_idx, vals = conv._psi_coo()
            with torch.sparse.check_sparse_tensor_invariants(enable=False):
                psi = torch.sparse_coo_tensor(torch.stack([ker_idx, row_idx, col_idx]), vals, size=(conv.kernel_size, conv.nlat_out, conv.nlat_in * conv.nlon_in)).to_dense()
            psi = psi.to(self.device)

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
        self.assertTrue(compare_tensors("output", y.to(y_ref.dtype), y_ref, atol=atol, rtol=rtol, verbose=verbose))

        # compare
        self.assertTrue(compare_tensors("input grad", x_grad.to(x_ref_grad.dtype), x_ref_grad, atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors("weight grad", conv.weight.grad.to(w_ref.grad.dtype), w_ref.grad, atol=atol, rtol=rtol, verbose=verbose))

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
        self.assertTrue(compare_tensors("output", out_naive, out_opt, atol=atol, rtol=rtol, verbose=verbose))

        # compare
        self.assertTrue(compare_tensors("input grad", inp_grad_naive, inp_grad_opt, atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors("weight grad", conv_naive.weight.grad, conv_opt.weight.grad, atol=atol, rtol=rtol, verbose=verbose))

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

        Uses bias=False so the final op of the conv module is the einsum, which is
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
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", False, 1e-4, 1e-4],
            [8, 4, 2, (16, 32), (8, 16), (5), "piecewise linear", "mean", "legendre-gauss", "legendre-gauss", False, 1e-4, 1e-4],
            [8, 4, 2, (16, 32), (16, 32), (3), "piecewise linear", "mean", "equiangular", "equiangular", True, 1e-4, 1e-4],
            [8, 4, 2, (8, 16), (16, 32), (5), "piecewise linear", "mean", "legendre-gauss", "legendre-gauss", True, 1e-4, 1e-4],
        ],
        skip_on_empty=True,
    )
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_device_instantiation(
        self, batch_size, in_channels, out_channels, in_shape, out_shape, kernel_shape, basis_type, basis_norm_mode, grid_in, grid_out, transpose, atol, rtol, verbose=False
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
            self.assertTrue(compare_tensors(name, getattr(conv_host, name).cpu(), getattr(conv_device, name).cpu(), atol=atol, rtol=rtol, verbose=verbose))

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

        if fused and not transpose:
            # The fused path is an autograd.Function around the raw kernels rather than an
            # op of its own, so check that it traces as a whole -- forward and backward, in
            # one graph -- and agrees with eager. aot_eager exercises the fake kernels and
            # the joint graph without needing a codegen toolchain.
            compiled = torch.compile(conv, backend="aot_eager", fullgraph=True)
            inp_eager = inp.clone().requires_grad_(True)
            inp_compiled = inp.clone().requires_grad_(True)
            out_eager = conv(inp_eager)
            out_compiled = compiled(inp_compiled)
            self.assertTrue(compare_tensors("fused output", out_compiled, out_eager, atol=1e-5, rtol=1e-5, verbose=verbose))
            grad = torch.randn_like(out_eager)
            out_eager.backward(grad)
            out_compiled.backward(grad)
            self.assertTrue(compare_tensors("fused input grad", inp_compiled.grad, inp_eager.grad, atol=1e-5, rtol=1e-5, verbose=verbose))

            # and the op it contracts with satisfies the op contract
            test_inputs = (inp, *_arc_state(conv), conv.kernel_size, conv.nlat_out, conv.nlon_out)
            opcheck(torch.ops.disco_kernels._disco_s2_contraction_regular_optimized, test_inputs)
        else:
            if transpose:
                # the scatter op reads (B, C, K, H, W): one plane per basis function per channel
                inp = torch.randn(batch_size, in_channels, conv.kernel_size, *in_shape, device=self.device)
            test_inputs = (inp, *_arc_state(conv), conv.kernel_size, conv.nlat_out, conv.nlon_out)
            if not transpose:
                opcheck(torch.ops.disco_kernels._disco_s2_contraction_regular_optimized, test_inputs)
            else:
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
        self.assertIsNotNone(conv.weight.grad, "weight.grad should still be populated via the einsum outside the op")

        # --- contract: psi_* buffers must never accumulate gradients ---
        # (they are non-learnable index/value tensors registered via register_buffer)
        for name in conv._backend_state:
            buf = getattr(conv, name)
            self.assertIsNone(buf.grad, f"buffer {name} should not accumulate a gradient (requires_grad={buf.requires_grad})")


# A supported device is not sufficient: the kpacked buffers are only built when
# this build actually contains the matching kernel (BUILD_KPACKED_SM90 / SM100,
# set from TORCH_CUDA_ARCH_LIST). A build targeting an architecture newer than
# the ones with kpacked kernels reports e.g. major == 10 while carrying no
# sm_100a cubin, so a device-only guard runs these tests against buffers that
# were deliberately never constructed. Ask the same question the library asks.
def _kpacked_built_for_sm90():
    """Hopper device AND an sm_90a kpacked kernel this device can load."""
    return _is_sm90() and _kpacked_runnable_here()


def _kpacked_built_for_sm100():
    """Blackwell device AND an sm_100a kpacked kernel this device can load."""
    return _is_sm100() and _kpacked_runnable_here()


def _kpacked_runnable_here():
    """Defer to the library, so the tests and the dispatch cannot disagree."""
    if not torch.cuda.is_available():
        return False
    return _kpacked_supported_on_device(torch.cuda.current_device())


def _is_kpacked_supported():
    """Return True when the kpacked forward can actually run here (device AND build)."""
    return _kpacked_built_for_sm90() or _kpacked_built_for_sm100()


def _arc_state(conv):
    """The arc arrays of conv's backend, in the order the operators take them."""
    return tuple(getattr(conv, name) for name in ("psi_row_ker", "psi_row_lat", "psi_seg_off", "psi_seg", "psi_val_off", "psi_vals"))


def _without_kpacked(conv):
    """Reselect conv's backend with the kpacked one ruled out, so it runs the arc kernels."""
    conv._backends = (OptimizedBackend, ReferenceBackend)
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
        """forward_kpacked is chosen for bf16/fp16 on Hopper."""
        conv = self._make_conv(1, 8, (16, 32))
        self.assertEqual(conv.backend.name, "kpacked", "the harmonic basis should select the kpacked backend")
        self.assertIn(conv.psi_kpacked_vals.shape[1], (8, 16), "K_pad must be 8 or 16 for the WGMMA kernel")
        inp = torch.randn(1, 8, 16, 32, dtype=torch.bfloat16, device=self.device)
        out = conv(inp)
        self.assertEqual(out.dtype, torch.bfloat16)

    @unittest.skipUnless(_kpacked_built_for_sm100(), "kpacked forward on Blackwell requires SM_100a (GB200/B200) and an sm_100a build")
    def test_kpacked_forward_activates_on_sm100(self):
        """tcgen05 kpacked path is chosen for bf16/fp16 on Blackwell."""
        conv = self._make_conv(1, 8, (16, 32))
        self.assertEqual(conv.backend.name, "kpacked", "the harmonic basis should select the kpacked backend")
        self.assertIn(conv.psi_kpacked_vals.shape[1], (8, 16), "K_pad must be 8 or 16 for the tcgen05 kernel")
        inp = torch.randn(1, 8, 16, 32, dtype=torch.bfloat16, device=self.device)
        out = conv(inp)
        self.assertEqual(out.dtype, torch.bfloat16)

    @unittest.skipUnless(_is_kpacked_supported(), "kpacked forward requires SM_90a or SM_100a")
    def test_kpacked_matches_optimized(self, verbose=True):
        """The kpacked tensor-core forward matches the arc kernels numerically."""
        set_seed(123)
        in_shape = (16, 32)
        conv_kpacked = self._make_conv(1, 8, in_shape).float()
        conv_opt = _without_kpacked(self._make_conv(1, 8, in_shape).float())
        self.assertEqual(conv_kpacked.backend.name, "kpacked", "kpacked reference test requires the kpacked backend")
        self.assertEqual(conv_opt.backend.name, "optimized")

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

    @unittest.skipUnless(_is_kpacked_supported(), "kpacked forward requires SM_90a or SM_100a")
    def test_kpacked_fused_matches_unfused(self):
        """fused=True kpacked path gives the same result as fused=False."""
        set_seed(42)
        conv_unfused = self._make_conv(1, 8, (16, 32), fused=False)
        conv_fused = self._make_conv(1, 8, (16, 32), fused=True)
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

    @unittest.skipUnless(_is_kpacked_supported(), "kpacked forward requires SM_90a or SM_100a")
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

    @unittest.skipUnless(_is_kpacked_supported(), "kpacked forward requires SM_90a or SM_100a")
    def test_kpacked_opcheck(self):
        """forward_kpacked op satisfies the PT2 opcheck contract."""
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
        opcheck(torch.ops.disco_kernels.forward_kpacked, test_inputs)


if __name__ == "__main__":
    unittest.main()
