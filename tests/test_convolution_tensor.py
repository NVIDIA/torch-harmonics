# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Tests for the construction of the disco psi convolution tensor:
#   - structural invariants of the sparse (CSR-form) psi after preprocess_psi
#   - the dense repacker (pack_psi_dense) against a Python reference
#   - equivalence between the existing CUDA-CSR forward and a Python reference
#     forward consuming the packed layout

import unittest
from parameterized import parameterized, parameterized_class

import torch

from torch_harmonics.disco import cuda_kernels_is_available, optimized_kernels_is_available
from torch_harmonics.disco.convolution import _precompute_convolution_tensor_s2
from torch_harmonics.filter_basis import get_filter_basis

from disco_helpers import preprocess_psi, pack_psi_dense

from disco_test_utils import python_pack_psi, python_disco_fwd_from_packed
from testutils import disable_tf32, set_seed


_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


def _build_psi(in_shape, out_shape, kernel_shape, basis_type, basis_norm_mode,
               grid_in, grid_out, theta_cutoff, dtype=torch.float64):
    """Build a preprocessed (CSR-form) psi for the requested configuration.

    Returns: (K, Ho, Wi, ker_idx, row_idx, col_idx, vals, roff_idx).
    """
    nlat_in, nlon_in = in_shape
    nlat_out, nlon_out = out_shape

    filter_basis = get_filter_basis(kernel_shape=kernel_shape, basis_type=basis_type)

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
    vals = vals.to(dtype).contiguous()

    K = filter_basis.kernel_size
    Ho = nlat_out
    Wi = nlon_in

    roff_idx = preprocess_psi(K, Ho, ker_idx, row_idx, col_idx, vals).contiguous()

    return K, Ho, Wi, ker_idx, row_idx, col_idx, vals, roff_idx


def _theta_cutoff(in_shape, out_shape, kernel_shape, factor):
    """Pick a theta cutoff. factor=1.0 reproduces the integrity-test default;
    factor>1 enlarges the support to exercise the dense-psi regime."""
    nlat_in = in_shape[0]
    nlat_out = out_shape[0]
    if factor == 1.0:
        return torch.pi / float(nlat_out - 1)
    if isinstance(kernel_shape, int):
        return factor * (kernel_shape + 1) * torch.pi / float(nlat_in - 1)
    return factor * (kernel_shape[0] + 1) * torch.pi / float(nlat_in - 1)


# Shared parameter set for repack tests. These cover the basis types and grid
# combinations that the dense kernel will need to support, plus a couple of
# enlarged-theta cases for the high-NBR regime.
_REPACK_CONFIGS = [
    # in_shape, out_shape, kernel_shape, basis, norm, grid_in, grid_out, theta_factor
    [(16, 32), (16, 32), (3,),    "piecewise linear", "mean", "equiangular", "equiangular", 1.0],
    [(16, 32), (8, 16),  (3,),    "piecewise linear", "mean", "equiangular", "equiangular", 1.0],
    [(24, 48), (12, 24), (3, 3),  "piecewise linear", "mean", "equiangular", "equiangular", 1.0],
    [(16, 32), (16, 32), (2, 2),  "harmonic",         "mean", "equiangular", "equiangular", 1.0],
    [(16, 32), (16, 32), (3,),    "zernike",          "mean", "equiangular", "equiangular", 1.0],
    [(16, 32), (16, 32), (3, 3),  "fourier-bessel",   "mean", "equiangular", "equiangular", 1.0],
    # larger theta cutoff -> denser psi (more neighbors per row)
    [(16, 32), (16, 32), (3,),    "piecewise linear", "mean", "equiangular", "equiangular", 4.0],
    [(24, 48), (12, 24), (3, 3),  "harmonic",         "mean", "equiangular", "equiangular", 4.0],
]


class TestConstruction(unittest.TestCase):
    """Tests covering construction of the disco psi tensor:
    structural integrity of the sparse form and correctness of the dense repacker.
    """

    def setUp(self):
        disable_tf32()
        set_seed(333)

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

    @parameterized.expand(_REPACK_CONFIGS, skip_on_empty=True)
    def test_pack_psi_matches_python_reference(
        self, in_shape, out_shape, kernel_shape, basis_type, basis_norm_mode,
        grid_in, grid_out, theta_factor,
    ):
        """C++ pack_psi_dense must produce the same packed psi as the Python reference."""

        theta_cutoff = _theta_cutoff(in_shape, out_shape, kernel_shape, theta_factor)

        K, Ho, Wi, ker_idx, row_idx, col_idx, vals, roff_idx = _build_psi(
            in_shape, out_shape, kernel_shape, basis_type, basis_norm_mode,
            grid_in, grid_out, theta_cutoff,
        )

        # auto-pad
        idx_cpp, val_cpp, count_cpp = pack_psi_dense(
            K, Ho, Wi, 0, ker_idx, row_idx, col_idx, vals, roff_idx,
        )

        idx_py, val_py, count_py = python_pack_psi(
            K, Ho, Wi, 0, ker_idx, row_idx, col_idx, vals, roff_idx,
        )

        self.assertEqual(tuple(idx_cpp.shape),   tuple(idx_py.shape),   "idx shape mismatch")
        self.assertEqual(tuple(val_cpp.shape),   tuple(val_py.shape),   "val shape mismatch")
        self.assertEqual(tuple(count_cpp.shape), tuple(count_py.shape), "count shape mismatch")

        self.assertTrue(torch.equal(idx_cpp.cpu(),   idx_py),   "idx_packed differs from python reference")
        self.assertTrue(torch.equal(count_cpp.cpu(), count_py), "count_packed differs from python reference")
        # vals: bit-exact since we just copy floats through with no arithmetic
        self.assertTrue(torch.equal(val_cpp.cpu(), val_py), "val_packed differs from python reference")

    @parameterized.expand(_REPACK_CONFIGS, skip_on_empty=True)
    def test_pack_psi_explicit_pad_matches_auto(
        self, in_shape, out_shape, kernel_shape, basis_type, basis_norm_mode,
        grid_in, grid_out, theta_factor,
    ):
        """Explicit nbr_pad >= max_nbr must produce the same content with extra zero-padded slots."""

        theta_cutoff = _theta_cutoff(in_shape, out_shape, kernel_shape, theta_factor)

        K, Ho, Wi, ker_idx, row_idx, col_idx, vals, roff_idx = _build_psi(
            in_shape, out_shape, kernel_shape, basis_type, basis_norm_mode,
            grid_in, grid_out, theta_cutoff,
        )

        idx_auto, val_auto, count_auto = pack_psi_dense(
            K, Ho, Wi, 0, ker_idx, row_idx, col_idx, vals, roff_idx,
        )
        nbr_pad = idx_auto.shape[2]
        bigger = nbr_pad + 7

        idx_big, val_big, count_big = pack_psi_dense(
            K, Ho, Wi, bigger, ker_idx, row_idx, col_idx, vals, roff_idx,
        )

        self.assertEqual(idx_big.shape[2], bigger)
        self.assertEqual(val_big.shape[2], bigger)
        self.assertTrue(torch.equal(idx_big[..., :nbr_pad, :].cpu(), idx_auto.cpu()))
        self.assertTrue(torch.equal(val_big[..., :nbr_pad].cpu(),    val_auto.cpu()))
        self.assertTrue(torch.equal(count_big.cpu(),                 count_auto.cpu()))
        # tail must be zero
        self.assertTrue(torch.all(idx_big[..., nbr_pad:, :] == 0))
        self.assertTrue(torch.all(val_big[..., nbr_pad:] == 0))


@parameterized_class(("device"), _devices)
class TestPackedPsiAgainstCSRKernel(unittest.TestCase):
    """Confirm the existing CPU/CUDA disco forward kernel agrees with a Python
    reference forward that consumes the packed psi. This pins down the semantics
    of the packed layout against the kernel that's already in production."""

    def setUp(self):
        if not optimized_kernels_is_available():
            self.skipTest("optimized disco kernels not available")
        if self.device.type == "cuda" and not cuda_kernels_is_available():
            self.skipTest("CUDA disco kernel not available")
        disable_tf32()
        set_seed(333)

    @parameterized.expand(
        [
            # smaller / faster configs only — the python reference fwd is O(K*Ho*NBR*B*C*Wo)
            # B=1 to match the production target
            # in_shape, out_shape, kernel_shape, basis, norm, grid_in, grid_out, theta_factor, B, C
            [(16, 32), (16, 32), (3,),   "piecewise linear", "mean", "equiangular", "equiangular", 1.0, 1, 2],
            [(16, 32), (8, 16),  (3,),   "piecewise linear", "mean", "equiangular", "equiangular", 1.0, 1, 2],
            [(16, 32), (16, 32), (2, 2), "harmonic",         "mean", "equiangular", "equiangular", 1.0, 1, 3],
            [(24, 48), (12, 24), (3, 3), "harmonic",         "mean", "equiangular", "equiangular", 1.0, 1, 2],
            # denser psi
            [(16, 32), (16, 32), (3,),   "piecewise linear", "mean", "equiangular", "equiangular", 4.0, 1, 2],
        ],
        skip_on_empty=True,
    )
    def test_csr_kernel_matches_python_from_packed(
        self, in_shape, out_shape, kernel_shape, basis_type, basis_norm_mode,
        grid_in, grid_out, theta_factor, B, C,
    ):
        theta_cutoff = _theta_cutoff(in_shape, out_shape, kernel_shape, theta_factor)

        K, Ho, Wi, ker_idx, row_idx, col_idx, vals, roff_idx = _build_psi(
            in_shape, out_shape, kernel_shape, basis_type, basis_norm_mode,
            grid_in, grid_out, theta_cutoff, dtype=torch.float64,
        )

        Hi, _ = in_shape
        _, Wo = out_shape

        # pack
        idx_packed, val_packed, count_packed = pack_psi_dense(
            K, Ho, Wi, 0, ker_idx, row_idx, col_idx, vals, roff_idx,
        )

        # move CSR data + packed data to the test device
        ker_idx_d  = ker_idx.to(self.device)
        row_idx_d  = row_idx.to(self.device)
        col_idx_d  = col_idx.to(self.device)
        vals_d     = vals.to(self.device)
        roff_idx_d = roff_idx.to(self.device)

        idx_packed_d   = idx_packed.to(self.device)
        val_packed_d   = val_packed.to(self.device)
        count_packed_d = count_packed.to(self.device)

        # input
        inp = torch.randn(B, C, Hi, Wi, dtype=torch.float64, device=self.device)

        # CSR kernel path: torch.ops.disco_kernels.forward
        out_csr = torch.ops.disco_kernels.forward(
            inp, roff_idx_d, ker_idx_d, row_idx_d, col_idx_d, vals_d, K, Ho, Wo
        )

        # Python reference from packed psi
        out_ref = python_disco_fwd_from_packed(
            inp, idx_packed_d, val_packed_d, count_packed_d, K, Ho, Wo
        )

        self.assertEqual(tuple(out_csr.shape), (B, C, K, Ho, Wo))
        self.assertEqual(tuple(out_ref.shape), (B, C, K, Ho, Wo))

        # both paths run in fp64 on identical data; the only differences are operation order.
        torch.testing.assert_close(out_csr, out_ref, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
