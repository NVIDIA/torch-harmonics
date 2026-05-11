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

"""
Tests for the discrete-continuous (DISCO) convolution tensor itself —
structural integrity of psi (forward CSR) and operator-transpose correctness
of psi_T. No conv module instantiation, no CUDA/CPU custom kernels: pure
checks on the tensor data structures produced by `convolution_tensor_s2`.
"""

import unittest
from parameterized import parameterized

import torch

from torch_harmonics.convolution_tensor_s2 import (
    _precompute_convolution_tensor_s2,
    _transpose_convolution_tensor_s2,
    _transpose_psi_kpacked,
)
from torch_harmonics.filter_basis import get_filter_basis
from disco_helpers import preprocess_psi

from testutils import disable_tf32


class TestConvolutionTensor(unittest.TestCase):
    """Integrity / transpose-correctness checks on the DISCO convolution tensor."""

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


    @parameterized.expand(
        [
            # in_shape, out_shape, kernel_shape, basis_type, basis_norm_mode, grid_in, grid_out
            # equiangular sweep over pscale and bases
            [(8, 16),  (8, 16),  (3, 3), "harmonic",         "mean", "equiangular", "equiangular"],   # pscale=1
            [(8, 16),  (4, 8),   (3, 3), "harmonic",         "mean", "equiangular", "equiangular"],   # pscale=2
            [(12, 24), (4, 8),   (3, 3), "harmonic",         "mean", "equiangular", "equiangular"],   # pscale=3
            [(16, 32), (8,  8),  (3, 3), "harmonic",         "mean", "equiangular", "equiangular"],   # pscale=4
            [(8, 16),  (4, 8),   (3,),   "piecewise linear", "mean", "equiangular", "equiangular"],
            [(8, 16),  (4, 8),   (3, 3), "fourier-bessel",   "mean", "equiangular", "equiangular"],
            [(8, 16),  (4, 8),   (3,),   "zernike",          "mean", "equiangular", "equiangular"],
            # K=1 edge case (single basis function — exercises off-by-one in row_T encoding / nrows_T = Hi*pscale)
            [(8, 16),  (4, 8),   (1, 1), "harmonic",         "mean", "equiangular", "equiangular"],
            # legendre-gauss grid (covers grid asymmetry in the precompute step)
            [(8, 16),  (4, 8),   (3, 3), "harmonic",         "mean", "legendre-gauss", "equiangular"],
            [(8, 16),  (4, 8),   (3, 3), "harmonic",         "mean", "equiangular",    "legendre-gauss"],
            [(8, 16),  (4, 8),   (3, 3), "harmonic",         "mean", "legendre-gauss", "legendre-gauss"],
        ],
        skip_on_empty=True,
    )
    def test_psi_T_matches_dense_transpose(
        self, in_shape, out_shape, kernel_shape, basis_type, basis_norm_mode, grid_in, grid_out, verbose=False,
    ):
        """psi_T must encode the operator-transpose of psi.

        The forward csr kernel realizes a linear map A : [Hi, Wi] -> [K, Ho, Wo]:
            out[k, ho, wo] = sum_{hi, wi} A[k, ho, wo, hi, wi] * inp[hi, wi]
        with A[k, ho, wo, hi, wi] = val whenever a psi entry exists at
        (k, ho, hi, wi_offset) and wi == (wi_offset + pscale*wo) mod Wi.

        psi_T must encode A^T : [K, Ho, Wo] -> [Hi, Wi] — i.e., the same operator
        with output/input axes swapped. We materialize A and B (the operator
        induced by psi_T) as dense tensors using `sparse_coo_tensor + coalesce`
        (vectorized; coalesce defensively merges any accidental duplicates) and
        check `B == A.permute(0, 3, 4, 1, 2)`.
        """
        Hi, Wi = in_shape
        Ho, Wo = out_shape
        assert Wi % Wo == 0
        pscale = Wi // Wo

        filter_basis = get_filter_basis(kernel_shape=kernel_shape, basis_type=basis_type)
        K = filter_basis.kernel_size

        if isinstance(kernel_shape, int):
            theta_cutoff = (kernel_shape + 1) * torch.pi / float(Hi - 1)
        else:
            theta_cutoff = (kernel_shape[0] + 1) * torch.pi / float(Hi - 1)

        idx, vals, _ = _precompute_convolution_tensor_s2(
            in_shape=in_shape, out_shape=out_shape, filter_basis=filter_basis,
            grid_in=grid_in, grid_out=grid_out,
            theta_cutoff=theta_cutoff,
            transpose_normalization=False, basis_norm_mode=basis_norm_mode,
            merge_quadrature=True,
        )
        ker_idx = idx[0].contiguous()
        row_idx = idx[1].contiguous()
        col_idx = idx[2].contiguous()
        vals = vals.contiguous()

        # ----- Forward operator A : [K, Ho, Wo, Hi, Wi] -----
        # Expand each psi entry over the wo axis: for each (k, ho, hi, wi_offset)
        # and each wo, contribute val to A[k, ho, wo, hi, (wi_offset+pscale*wo)%Wi].
        nnz = ker_idx.numel()
        wo_axis = torch.arange(Wo, dtype=torch.int64)

        ker_e       = ker_idx.repeat_interleave(Wo)
        ho_e        = row_idx.repeat_interleave(Wo)
        hi_e        = (col_idx // Wi).repeat_interleave(Wo)
        wi_offset_e = (col_idx %  Wi).repeat_interleave(Wo)
        wo_e        = wo_axis.repeat(nnz)
        wi_e        = (wi_offset_e + pscale * wo_e) % Wi
        vals_e      = vals.repeat_interleave(Wo)

        A_idx = torch.stack([ker_e, ho_e, wo_e, hi_e, wi_e], dim=0)
        A = torch.sparse_coo_tensor(A_idx, vals_e, size=(K, Ho, Wo, Hi, Wi)).coalesce().to_dense()

        # ----- psi_T operator B : [K, Hi, Wi, Ho, Wo] -----
        ker_T, col_T, vals_T, roff_T = _transpose_convolution_tensor_s2(
            ker_idx, row_idx, col_idx, vals,
            in_shape=in_shape, out_shape=out_shape,
        )
        nnz_T = ker_T.numel()

        # recover hi for each entry via roff_T (each row_T bucket has implicit hi = row_T // pscale)
        row_T_per_entry = torch.repeat_interleave(
            torch.arange(roff_T.numel() - 1, dtype=torch.int64),
            torch.diff(roff_T),
        )
        hi_T = row_T_per_entry // pscale  # [nnz_T]

        ker_T_e       = ker_T.repeat_interleave(Wo)
        hi_T_e        = hi_T.repeat_interleave(Wo)
        ho_T_e        = (col_T // Wi).repeat_interleave(Wo)
        wi_offset_T_e = (col_T %  Wi).repeat_interleave(Wo)
        wo_T_e        = wo_axis.repeat(nnz_T)
        wi_T_e        = (wi_offset_T_e + pscale * wo_T_e) % Wi
        vals_T_e      = vals_T.repeat_interleave(Wo)

        B_idx = torch.stack([ker_T_e, hi_T_e, wi_T_e, ho_T_e, wo_T_e], dim=0)
        B = torch.sparse_coo_tensor(B_idx, vals_T_e, size=(K, Hi, Wi, Ho, Wo)).coalesce().to_dense()

        A_T = A.permute(0, 3, 4, 1, 2).contiguous()
        if verbose:
            print(f"\npsi_T transpose check: nnz(A)={(A != 0).sum().item()}, nnz(B)={(B != 0).sum().item()}, "
                  f"max|B - A^T|={(B - A_T).abs().max().item():.3e}")
        self.assertTrue(torch.allclose(B, A_T, atol=1e-12, rtol=0))


    @parameterized.expand(
        [
            # Only K-shared bases (harmonic, fourier-bessel, zernike, morlet).
            # Piecewise-linear has per-k radial support → K-packing doesn't apply.
            # in_shape, out_shape, kernel_shape, basis_type, basis_norm_mode, grid_in, grid_out
            [(8, 16),  (8, 16),  (3, 3), "harmonic",       "mean", "equiangular", "equiangular"],  # pscale=1
            [(8, 16),  (4, 8),   (3, 3), "harmonic",       "mean", "equiangular", "equiangular"],  # pscale=2
            [(12, 24), (4, 8),   (3, 3), "harmonic",       "mean", "equiangular", "equiangular"],  # pscale=3
            [(16, 32), (8,  8),  (3, 3), "harmonic",       "mean", "equiangular", "equiangular"],  # pscale=4
            [(8, 16),  (4, 8),   (3, 3), "fourier-bessel", "mean", "equiangular", "equiangular"],
            [(8, 16),  (4, 8),   (3,),   "zernike",        "mean", "equiangular", "equiangular"],
            # K=1 edge case
            [(8, 16),  (4, 8),   (1, 1), "harmonic",       "mean", "equiangular", "equiangular"],
            # legendre-gauss grid sweeps
            [(8, 16),  (4, 8),   (3, 3), "harmonic",       "mean", "legendre-gauss", "equiangular"],
            [(8, 16),  (4, 8),   (3, 3), "harmonic",       "mean", "equiangular",    "legendre-gauss"],
        ],
        skip_on_empty=True,
    )
    def test_psi_T_kpacked_matches_dense_transpose(
        self, in_shape, out_shape, kernel_shape, basis_type, basis_norm_mode, grid_in, grid_out, verbose=False,
    ):
        """K-packed psi_T must encode the operator-transpose of psi.

        Mirrors `test_psi_T_matches_dense_transpose` but feeds K-packed forward
        psi (one entry per (ho, nz), with K vals stacked) into
        `_transpose_psi_kpacked` and materializes the resulting K-packed psi_T
        operator. Restricted to bases where all K basis functions share the
        same (hi, wi_offset) support per ho (the K-shared invariant), which
        the forward K-packed layout requires.
        """
        Hi, Wi = in_shape
        Ho, Wo = out_shape
        assert Wi % Wo == 0
        pscale = Wi // Wo

        filter_basis = get_filter_basis(kernel_shape=kernel_shape, basis_type=basis_type)
        K = filter_basis.kernel_size

        if isinstance(kernel_shape, int):
            theta_cutoff = (kernel_shape + 1) * torch.pi / float(Hi - 1)
        else:
            theta_cutoff = (kernel_shape[0] + 1) * torch.pi / float(Hi - 1)

        idx, vals, _ = _precompute_convolution_tensor_s2(
            in_shape=in_shape, out_shape=out_shape, filter_basis=filter_basis,
            grid_in=grid_in, grid_out=grid_out,
            theta_cutoff=theta_cutoff,
            transpose_normalization=False, basis_norm_mode=basis_norm_mode,
            merge_quadrature=True,
        )
        ker_idx = idx[0].contiguous()
        row_idx = idx[1].contiguous()
        col_idx = idx[2].contiguous()
        vals = vals.contiguous()

        # ----- Forward operator A : [K, Ho, Wo, Hi, Wi] (reference, from CSR) -----
        nnz = ker_idx.numel()
        wo_axis = torch.arange(Wo, dtype=torch.int64)

        ker_e       = ker_idx.repeat_interleave(Wo)
        ho_e        = row_idx.repeat_interleave(Wo)
        hi_e        = (col_idx // Wi).repeat_interleave(Wo)
        wi_offset_e = (col_idx %  Wi).repeat_interleave(Wo)
        wo_e        = wo_axis.repeat(nnz)
        wi_e        = (wi_offset_e + pscale * wo_e) % Wi
        vals_e      = vals.repeat_interleave(Wo)

        A_idx = torch.stack([ker_e, ho_e, wo_e, hi_e, wi_e], dim=0)
        A = torch.sparse_coo_tensor(A_idx, vals_e, size=(K, Ho, Wo, Hi, Wi)).coalesce().to_dense()

        # ----- Build forward K-packed psi from CSR (inline; expects K-shared support) -----
        psi_kp_idx, psi_kp_vals, psi_kp_count, K_pad = self._csr_to_kpacked(
            ker_idx, row_idx, col_idx, vals, K=K, Ho=Ho, Wi=Wi,
        )
        # Sanity: kpacked must be K-shared. Skip rather than fail if the basis
        # doesn't actually satisfy the invariant (defensive — shouldn't trip
        # for the bases in this parameterization).
        if psi_kp_idx is None:
            self.skipTest(f"basis '{basis_type}' does not satisfy K-shared support; K-packing inapplicable")

        # ----- Build K-packed psi_T from K-packed forward psi -----
        idx_T_kp, vals_T_kp, count_T_kp = _transpose_psi_kpacked(
            psi_kp_idx, psi_kp_vals, psi_kp_count, in_shape=in_shape, out_shape=out_shape,
        )

        # ----- Materialize operator B : [K_pad, Hi, Wi, Ho, Wo] from K-packed psi_T -----
        # Per (row_T, nz_slot) valid slot: (ho_orig, wi_offset) + K_pad vals.
        # Expand over (k, wo) to get the operator entries.
        Nrows_T, NBR_PAD_T, _ = idx_T_kp.shape
        nz_axis = torch.arange(NBR_PAD_T, dtype=torch.int64)
        valid = nz_axis.view(1, NBR_PAD_T) < count_T_kp.view(Nrows_T, 1)  # [Nrows_T, NBR_PAD_T]

        row_T_axis = torch.arange(Nrows_T, dtype=torch.int64).view(Nrows_T, 1).expand(Nrows_T, NBR_PAD_T)
        hi_T = row_T_axis // pscale       # [Nrows_T, NBR_PAD_T]   (bigger grid lat from row bucket)
        ho_T = idx_T_kp[:, :, 0]
        wi_offset_T = idx_T_kp[:, :, 1]

        valid_flat       = valid.reshape(-1)
        hi_T_flat        = hi_T.reshape(-1)[valid_flat]
        ho_T_flat        = ho_T.reshape(-1)[valid_flat]
        wi_offset_T_flat = wi_offset_T.reshape(-1)[valid_flat]
        vals_T_flat      = vals_T_kp.reshape(Nrows_T * NBR_PAD_T, K_pad)[valid_flat]  # [N_valid, K_pad]
        N_valid          = hi_T_flat.numel()

        # Expand each entry over (k, wo):
        #   hi_T,wi_T  : [K_pad * N_valid * Wo]
        #   ho_T,wo_T  : same
        #   k_axis     : [K_pad * N_valid * Wo]
        #   vals_kew   : [K_pad * N_valid * Wo]  flattened with K_pad as outer
        k_axis     = torch.arange(K_pad, dtype=torch.int64)

        hi_e_T     = hi_T_flat.repeat_interleave(Wo).unsqueeze(0).expand(K_pad, -1).reshape(-1)
        ho_e_T     = ho_T_flat.repeat_interleave(Wo).unsqueeze(0).expand(K_pad, -1).reshape(-1)
        wi_off_e_T = wi_offset_T_flat.repeat_interleave(Wo).unsqueeze(0).expand(K_pad, -1).reshape(-1)
        wo_e_T     = wo_axis.repeat(N_valid).unsqueeze(0).expand(K_pad, -1).reshape(-1)
        wi_e_T     = (wi_off_e_T + pscale * wo_e_T) % Wi
        k_e_T      = k_axis.unsqueeze(1).expand(K_pad, N_valid * Wo).reshape(-1)
        # vals stored as [N_valid, K_pad] → repeat_interleave by Wo → [N_valid*Wo, K_pad]
        # → transpose → [K_pad, N_valid*Wo] → reshape to flat
        vals_e_T   = vals_T_flat.repeat_interleave(Wo, dim=0).transpose(0, 1).reshape(-1)

        B_idx = torch.stack([k_e_T, hi_e_T, wi_e_T, ho_e_T, wo_e_T], dim=0)
        B_full = torch.sparse_coo_tensor(B_idx, vals_e_T,
                                         size=(K_pad, Hi, Wi, Ho, Wo)).coalesce().to_dense()

        # Drop padding along K to compare against operator A.
        B = B_full[:K]

        # Padded K slots must be exactly zero (vals_T_kp[:, :, K:] is zero by construction).
        if K_pad > K:
            self.assertEqual(B_full[K:].abs().max().item(), 0.0)

        A_T = A.permute(0, 3, 4, 1, 2).contiguous()
        if verbose:
            print(f"\npsi_T kpacked transpose check: K={K}, K_pad={K_pad}, "
                  f"max|B - A^T|={(B - A_T).abs().max().item():.3e}")
        self.assertTrue(torch.allclose(B, A_T, atol=1e-12, rtol=0))


    @staticmethod
    def _csr_to_kpacked(ker_idx, row_idx, col_idx, vals, K, Ho, Wi, n_align=8):
        """Build K-packed forward psi from CSR (inline test helper).

        Verifies the K-shared invariant (same per-ho count + same (hi, wi_offset)
        sequence across all k) and returns (idx, vals, count, K_pad) on success,
        or (None, None, None, None) on failure. K_pad is K rounded up to
        n_align — matches the production wiring's WGMMA alignment.
        """
        device = ker_idx.device
        K_pad = ((K + n_align - 1) // n_align) * n_align

        # Per-(k, ho) count check via bincount.
        # Reference counts come from k=0.
        mask_k0 = (ker_idx == 0)
        counts_ref = torch.bincount(row_idx[mask_k0], minlength=Ho).to(torch.int64)
        NBR_PAD = int(counts_ref.max().item()) if counts_ref.numel() else 0
        if NBR_PAD == 0:
            # Trivial degenerate case
            return (
                torch.zeros(Ho, 0, 2, dtype=torch.int64, device=device),
                torch.zeros(Ho, 0, K_pad, dtype=vals.dtype, device=device),
                counts_ref, K_pad,
            )

        idx_kp  = torch.zeros(Ho, NBR_PAD, 2, dtype=torch.int64, device=device)
        vals_kp = torch.zeros(Ho, NBR_PAD, K_pad, dtype=vals.dtype, device=device)

        # CSR after preprocess_psi is sorted by (ker, row=ho, col). Within each
        # (k, ho) the entries are in col-sorted order. For K-shared support the
        # col sequence matches across all k.
        for k in range(K):
            mask_k = (ker_idx == k)
            rows_k = row_idx[mask_k]
            cols_k = col_idx[mask_k]
            vals_k = vals[mask_k]

            counts_k = torch.bincount(rows_k, minlength=Ho).to(torch.int64)
            if not torch.equal(counts_k, counts_ref):
                return None, None, None, None  # Not K-shared

            # nz_pos within (k, ho) = z - cumulative-prefix(counts_k)[ho].
            offsets = torch.zeros(Ho + 1, dtype=torch.int64, device=device)
            offsets[1:] = torch.cumsum(counts_k, dim=0)
            z_axis = torch.arange(rows_k.numel(), device=device)
            nz_pos = z_axis - offsets[rows_k]

            if k == 0:
                # Fill (hi, wi_offset).
                hi_k0        = cols_k // Wi
                wi_offset_k0 = cols_k %  Wi
                idx_kp[rows_k, nz_pos, 0] = hi_k0
                idx_kp[rows_k, nz_pos, 1] = wi_offset_k0
            else:
                # Verify col sequence matches k=0 (K-shared invariant beyond
                # just count match).
                expected_col = idx_kp[rows_k, nz_pos, 0] * Wi + idx_kp[rows_k, nz_pos, 1]
                if not torch.equal(cols_k, expected_col):
                    return None, None, None, None

            vals_kp[rows_k, nz_pos, k] = vals_k

        return idx_kp, vals_kp, counts_ref, K_pad


if __name__ == "__main__":
    unittest.main()
