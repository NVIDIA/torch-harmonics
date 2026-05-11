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


if __name__ == "__main__":
    unittest.main()
