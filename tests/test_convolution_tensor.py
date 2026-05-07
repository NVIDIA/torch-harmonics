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
from testutils import compare_tensors, disable_tf32, set_seed


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
    [(25, 48), (12, 24), (3, 3),  "piecewise linear", "mean", "equiangular", "legendre-gauss", 1.0],
    [(16, 32), (16, 32), (2, 2),  "harmonic",         "mean", "equiangular", "equiangular", 1.0],
    [(17, 32), (17, 32), (2, 2),  "harmonic",         "mean", "equiangular", "equiangular", 1.0],
    [(17, 32), ( 8, 16), (2, 2),  "harmonic",         "mean", "equiangular", "legendre-gauss", 1.0],
    [(16, 32), (16, 16), (2, 2),  "harmonic",         "mean", "legendre-gauss", "legendre-gauss", 1.0],
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
        self.assertTrue(compare_tensors("val_packed (cpp vs python ref)", val_cpp.cpu(), val_py))

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
        self.assertTrue(compare_tensors("val_packed (auto-pad prefix)", val_big[..., :nbr_pad].cpu(), val_auto.cpu()))
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

        # CSR kernel path: torch.ops.disco_kernels.forward_csr
        out_csr = torch.ops.disco_kernels.forward_csr(
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


class TestDenseKernelWiring(unittest.TestCase):
    """Smoke tests for the use_dense_kernel construction path: when the flag is on,
    the planner must register the packed buffers with the right shapes and contents.
    When the flag is off, the packed buffers must NOT be registered."""

    def setUp(self):
        if not optimized_kernels_is_available():
            self.skipTest("optimized disco kernels not available")
        disable_tf32()
        set_seed(333)

    @parameterized.expand(
        [
            # in_shape, out_shape, kernel_shape, basis, norm, transpose
            [(16, 32), (16, 32), (3,),   "piecewise linear", "mean", False],
            [(16, 32), (8, 16),  (3,),   "piecewise linear", "mean", False],
            [(16, 32), (16, 32), (2, 2), "harmonic",         "mean", False],
            [(16, 32), (16, 32), (3,),   "piecewise linear", "mean", True],
            [(8, 16),  (16, 32), (3,),   "piecewise linear", "mean", True],
        ],
        skip_on_empty=True,
    )
    def test_packed_buffers_match_python_reference(
        self, in_shape, out_shape, kernel_shape, basis_type, basis_norm_mode, transpose,
    ):
        """When use_dense_kernel=True, psi_packed_{idx,vals,count} must be registered
        and equal to the python_pack_psi reference. We get the CSR form by also
        instantiating a parallel use_dense_kernel=False module — those two share
        the same psi up to the packing step."""
        from torch_harmonics import DiscreteContinuousConvS2, DiscreteContinuousConvTransposeS2

        Conv = DiscreteContinuousConvTransposeS2 if transpose else DiscreteContinuousConvS2

        if isinstance(kernel_shape, int):
            theta_cutoff = (kernel_shape + 1) * torch.pi / float(in_shape[0] - 1)
        else:
            theta_cutoff = (kernel_shape[0] + 1) * torch.pi / float(in_shape[0] - 1)

        common_kwargs = dict(
            in_channels=2, out_channels=2,
            in_shape=in_shape, out_shape=out_shape,
            kernel_shape=kernel_shape,
            basis_type=basis_type, basis_norm_mode=basis_norm_mode,
            bias=False, theta_cutoff=theta_cutoff,
            optimized_kernel=True,
        )

        conv_dense = Conv(**common_kwargs, use_dense_kernel=True)
        conv_csr   = Conv(**common_kwargs, use_dense_kernel=False)

        # dense path registers only the packed buffers
        for name in ("psi_packed_idx", "psi_packed_vals", "psi_packed_count"):
            self.assertTrue(hasattr(conv_dense, name), f"missing buffer {name} on dense module")

        # for transpose, col is decoded against nlon_out; for forward, against nlon_in.
        # Ho is nlat_in for transpose, nlat_out for forward.
        Ho_pack = conv_dense.nlat_in if transpose else conv_dense.nlat_out
        Wi_pack = conv_dense.nlon_out if transpose else conv_dense.nlon_in

        idx_ref, val_ref, count_ref = python_pack_psi(
            conv_dense.kernel_size, Ho_pack, Wi_pack, 0,
            conv_csr.psi_ker_idx, conv_csr.psi_row_idx, conv_csr.psi_col_idx,
            conv_csr.psi_vals, conv_csr.psi_roff_idx,
        )

        # shapes
        self.assertEqual(tuple(conv_dense.psi_packed_idx.shape),   tuple(idx_ref.shape))
        self.assertEqual(tuple(conv_dense.psi_packed_vals.shape),  tuple(val_ref.shape))
        self.assertEqual(tuple(conv_dense.psi_packed_count.shape), tuple(count_ref.shape))

        # contents (compare on cpu)
        self.assertTrue(torch.equal(conv_dense.psi_packed_idx.cpu(),   idx_ref))
        self.assertTrue(compare_tensors("psi_packed_vals (planner vs python ref)", conv_dense.psi_packed_vals.cpu(), val_ref))
        self.assertTrue(torch.equal(conv_dense.psi_packed_count.cpu(), count_ref))

    @parameterized.expand(
        [
            [False],  # forward
            [True],   # transpose
        ],
        skip_on_empty=True,
    )
    def test_buffer_set_matches_kernel_mode(self, transpose):
        """Buffer hygiene: CSR-only mode registers only CSR buffers, dense-only
        mode registers only packed buffers. The two sets should never coexist."""
        from torch_harmonics import DiscreteContinuousConvS2, DiscreteContinuousConvTransposeS2

        Conv = DiscreteContinuousConvTransposeS2 if transpose else DiscreteContinuousConvS2
        in_shape  = (8, 16) if transpose else (16, 32)
        out_shape = (16, 32) if transpose else (16, 32)

        common_kwargs = dict(
            in_channels=2, out_channels=2,
            in_shape=in_shape, out_shape=out_shape,
            kernel_shape=(3,), basis_type="piecewise linear",
            basis_norm_mode="mean", bias=False,
            theta_cutoff=4 * torch.pi / float(in_shape[0] - 1),
            optimized_kernel=True,
        )

        csr_buffers    = ("psi_roff_idx", "psi_ker_idx", "psi_row_idx", "psi_col_idx", "psi_vals")
        packed_buffers = ("psi_packed_idx", "psi_packed_vals", "psi_packed_count")

        # CSR mode: CSR buffers present, packed absent
        conv_csr = Conv(**common_kwargs, use_dense_kernel=False)
        for name in csr_buffers:
            self.assertTrue(hasattr(conv_csr, name), f"buffer {name} should be registered when use_dense_kernel=False")
        for name in packed_buffers:
            self.assertFalse(hasattr(conv_csr, name), f"buffer {name} should not be registered when use_dense_kernel=False")

        # dense mode: packed buffers present, CSR absent
        conv_dense = Conv(**common_kwargs, use_dense_kernel=True)
        for name in packed_buffers:
            self.assertTrue(hasattr(conv_dense, name), f"buffer {name} should be registered when use_dense_kernel=True")
        for name in csr_buffers:
            self.assertFalse(hasattr(conv_dense, name), f"buffer {name} should not be registered when use_dense_kernel=True")


class TestKPackedKernel(unittest.TestCase):
    """Tests for the K-packed buffers and the scalar K-packed CUDA kernel
    (`forward_dense_kpacked`). The kernel reads psi laid out as
    [Ho, NBR_PAD, K_PAD] and produces output identical to the per-k_kern dense
    kernel for bases with shared support across k_kern (everything except
    piecewise linear)."""

    def setUp(self):
        if not optimized_kernels_is_available():
            self.skipTest("optimized disco kernels not available")
        disable_tf32()
        set_seed(333)

    @parameterized.expand(
        [
            # (basis_type, expects_kpacked)
            # Bases with shared support across k_kern → kpacked buffers should be built.
            ("harmonic",        True),
            ("morlet",          True),
            ("zernike",         True),
            ("fourier-bessel",  True),
            # Piecewise linear has per-k_kern support → no kpacked buffers.
            ("piecewise linear", False),
        ],
        skip_on_empty=True,
    )
    def test_kpacked_buffer_registration(self, basis_type, expects_kpacked):
        """psi_kpacked_* must be registered for shared-support bases and absent
        for piecewise linear."""
        from torch_harmonics import DiscreteContinuousConvS2

        in_shape, out_shape = (16, 32), (16, 32)
        kernel_shape = (3, 3) if basis_type != "piecewise linear" else (3,)
        theta_cutoff = (kernel_shape[0] + 1) * torch.pi / float(in_shape[0] - 1) \
            if isinstance(kernel_shape, tuple) else \
            (kernel_shape + 1) * torch.pi / float(in_shape[0] - 1)

        conv = DiscreteContinuousConvS2(
            in_channels=2, out_channels=2,
            in_shape=in_shape, out_shape=out_shape,
            kernel_shape=kernel_shape,
            basis_type=basis_type, basis_norm_mode="mean",
            bias=False, theta_cutoff=theta_cutoff,
            optimized_kernel=True, use_dense_kernel=True,
        )

        kpacked_names = ("psi_kpacked_idx", "psi_kpacked_vals", "psi_kpacked_count")
        if expects_kpacked:
            for name in kpacked_names:
                self.assertTrue(hasattr(conv, name),
                                f"{name} should be registered for basis '{basis_type}'")
            self.assertIsNotNone(conv.psi_kpacked_K_pad,
                                 f"psi_kpacked_K_pad should be set for '{basis_type}'")
            # K_pad should be a multiple of 8 and >= K
            self.assertEqual(conv.psi_kpacked_K_pad % 8, 0)
            self.assertGreaterEqual(conv.psi_kpacked_K_pad, conv.kernel_size)
            # Shape sanity
            Ho      = conv.nlat_out
            NBR_PAD = conv.psi_packed_idx.shape[2]
            K_pad   = conv.psi_kpacked_K_pad
            self.assertEqual(tuple(conv.psi_kpacked_idx.shape),   (Ho, NBR_PAD, 2))
            self.assertEqual(tuple(conv.psi_kpacked_vals.shape),  (Ho, NBR_PAD, K_pad))
            self.assertEqual(tuple(conv.psi_kpacked_count.shape), (Ho,))
        else:
            for name in kpacked_names:
                self.assertFalse(hasattr(conv, name),
                                 f"{name} should NOT be registered for basis '{basis_type}'")
            self.assertIsNone(conv.psi_kpacked_K_pad,
                              f"psi_kpacked_K_pad should be None for '{basis_type}'")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for forward_dense_kpacked kernel")
    @parameterized.expand(
        [
            # (in_shape, out_shape, kernel_shape, basis, dtype, atol, rtol)
            [(24, 48), (12, 24), (3, 3), "harmonic", torch.float32, 1e-5, 1e-5],
            [(24, 48), (12, 24), (3, 4), "morlet",   torch.float32, 1e-5, 1e-5],
            [(24, 48), (12, 24), (3,),   "zernike",  torch.float32, 1e-5, 1e-5],
            # bf16 — looser tolerance
            [(24, 48), (12, 24), (3, 3), "harmonic", torch.bfloat16, 5e-2, 5e-2],
            # WGMMA-firing configs: B*C=8, Wo divisible by 8. Symmetric sweep
            # over (dtype ∈ {bf16, fp16}) × (K-class) × (pscale ∈ {1,2,3,4}):
            #   - K=8 (harmonic(4,2)) → m64n8k16 wgmma shape
            #   - K=16 exact (harmonic(4,4)) → m64n16k16 wgmma shape, no padding
            #   - K=9 (harmonic(3,3)) → K_PAD=16, exercises k_o ≥ K writeback skip
            # On Hopper this routes through the WGMMA fast path; on V100/A100
            # the host runtime check falls back to the scalar K-packed kernel.
            # bf16 × K=8 × pscale 1..4
            [(16, 32),  (16, 32), (4, 2), "harmonic", torch.bfloat16, 5e-2, 5e-2],
            [(32, 64),  (16, 32), (4, 2), "harmonic", torch.bfloat16, 5e-2, 5e-2],
            [(32, 96),  (16, 32), (4, 2), "harmonic", torch.bfloat16, 5e-2, 5e-2],
            [(32, 128), (16, 32), (4, 2), "harmonic", torch.bfloat16, 5e-2, 5e-2],
            # bf16 × K=16 × pscale 1..4
            [(16, 32),  (16, 32), (4, 4), "harmonic", torch.bfloat16, 5e-2, 5e-2],
            [(32, 64),  (16, 32), (4, 4), "harmonic", torch.bfloat16, 5e-2, 5e-2],
            [(32, 96),  (16, 32), (4, 4), "harmonic", torch.bfloat16, 5e-2, 5e-2],
            [(32, 128), (16, 32), (4, 4), "harmonic", torch.bfloat16, 5e-2, 5e-2],
            # bf16 × K=9 (K_PAD=16) × pscale 1..4
            [(16, 32),  (16, 32), (3, 3), "harmonic", torch.bfloat16, 5e-2, 5e-2],
            [(32, 64),  (16, 32), (3, 3), "harmonic", torch.bfloat16, 5e-2, 5e-2],
            [(32, 96),  (16, 32), (3, 3), "harmonic", torch.bfloat16, 5e-2, 5e-2],
            [(32, 128), (16, 32), (3, 3), "harmonic", torch.bfloat16, 5e-2, 5e-2],
            # fp16 × K=8 × pscale 1..4
            [(16, 32),  (16, 32), (4, 2), "harmonic", torch.float16,  5e-2, 5e-2],
            [(32, 64),  (16, 32), (4, 2), "harmonic", torch.float16,  5e-2, 5e-2],
            [(32, 96),  (16, 32), (4, 2), "harmonic", torch.float16,  5e-2, 5e-2],
            [(32, 128), (16, 32), (4, 2), "harmonic", torch.float16,  5e-2, 5e-2],
            # fp16 × K=16 × pscale 1..4
            [(16, 32),  (16, 32), (4, 4), "harmonic", torch.float16,  5e-2, 5e-2],
            [(32, 64),  (16, 32), (4, 4), "harmonic", torch.float16,  5e-2, 5e-2],
            [(32, 96),  (16, 32), (4, 4), "harmonic", torch.float16,  5e-2, 5e-2],
            [(32, 128), (16, 32), (4, 4), "harmonic", torch.float16,  5e-2, 5e-2],
            # fp16 × K=9 (K_PAD=16) × pscale 1..4
            [(16, 32),  (16, 32), (3, 3), "harmonic", torch.float16,  5e-2, 5e-2],
            [(32, 64),  (16, 32), (3, 3), "harmonic", torch.float16,  5e-2, 5e-2],
            [(32, 96),  (16, 32), (3, 3), "harmonic", torch.float16,  5e-2, 5e-2],
            [(32, 128), (16, 32), (3, 3), "harmonic", torch.float16,  5e-2, 5e-2],
        ],
        skip_on_empty=True,
    )
    def test_kpacked_kernel_matches_per_k_kernel(
        self, in_shape, out_shape, kernel_shape, basis_type, dtype, atol, rtol,
    ):
        """The K-packed CUDA kernel must produce identical output to the existing
        per-k_kern dense kernel (within precision tolerance) when both are
        invoked on the same input and matching psi data."""
        if not cuda_kernels_is_available():
            self.skipTest("CUDA disco kernels not available")
        from torch_harmonics import DiscreteContinuousConvS2

        device = torch.device("cuda")
        nlat_in, nlon_in = in_shape

        if isinstance(kernel_shape, int) or len(kernel_shape) == 1:
            k0 = kernel_shape if isinstance(kernel_shape, int) else kernel_shape[0]
            theta_cutoff = (k0 + 1) * torch.pi / float(nlat_in - 1)
        else:
            theta_cutoff = (kernel_shape[0] + 1) * torch.pi / float(nlat_in - 1)

        conv = DiscreteContinuousConvS2(
            in_channels=8, out_channels=4,
            in_shape=in_shape, out_shape=out_shape,
            kernel_shape=kernel_shape,
            basis_type=basis_type, basis_norm_mode="mean",
            bias=False, theta_cutoff=theta_cutoff,
            optimized_kernel=True, use_dense_kernel=True,
        ).to(device)

        if conv.psi_kpacked_K_pad is None:
            self.skipTest(f"basis '{basis_type}' did not produce K-packed buffers")

        # Wo divisible by WO_TILE=8 is enforced by the kernel; B*C alignment is
        # not required (out-of-range bc threads early-return in the scalar path
        # and are zero-padded / writeback-skipped in the WGMMA path).
        B, C = 1, 8
        x = torch.randn(B, C, nlat_in, nlon_in, device=device, dtype=dtype)

        # Per-k_kern dense kernel output (reference).
        try:
            from torch_harmonics.disco import disco_kernels
        except Exception:
            self.skipTest("disco_kernels op namespace not loadable")
        if not hasattr(disco_kernels, "forward_dense_kpacked"):
            self.skipTest("forward_dense_kpacked op not registered (rebuild needed?)")

        kernel_size = conv.kernel_size
        out_ref = disco_kernels.forward_dense.default(
            x.contiguous(),
            conv.psi_packed_idx, conv.psi_packed_vals, conv.psi_packed_count,
            kernel_size, conv.nlat_out, conv.nlon_out,
        )

        out_kpacked = disco_kernels.forward_dense_kpacked.default(
            x.contiguous(),
            conv.psi_kpacked_idx, conv.psi_kpacked_vals, conv.psi_kpacked_count,
            kernel_size, conv.nlat_out, conv.nlon_out,
        )

        self.assertEqual(tuple(out_ref.shape), tuple(out_kpacked.shape))
        self.assertTrue(compare_tensors(
            "kpacked vs per-k_kern forward_dense",
            out_kpacked, out_ref, atol=atol, rtol=rtol, verbose=True))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for forward_dense_kpacked kernel")
    @parameterized.expand(
        [
            # (B, C, in_shape, out_shape, kernel_shape, dtype) — exercise B*C
            # values not divisible by BC_TILE=8. The kernel_shape sets K_PAD
            # (4,2)→K=8, (4,4)→K=16. Wi/Wo > 1 sets pscale.
            #
            # Sweep C ∈ {9..15} at B=1 to hit all 7 nonzero remainders mod 8.
            [1,  9, (16, 32), (16, 32), (4, 2), torch.bfloat16],   # B*C=9   rem 1
            [1, 10, (16, 32), (16, 32), (4, 2), torch.bfloat16],   # B*C=10  rem 2
            [1, 11, (16, 32), (16, 32), (4, 2), torch.bfloat16],   # B*C=11  rem 3
            [1, 12, (16, 32), (16, 32), (4, 2), torch.bfloat16],   # B*C=12  rem 4
            [1, 13, (16, 32), (16, 32), (4, 2), torch.bfloat16],   # B*C=13  rem 5
            [1, 14, (16, 32), (16, 32), (4, 2), torch.bfloat16],   # B*C=14  rem 6
            [1, 15, (16, 32), (16, 32), (4, 2), torch.bfloat16],   # B*C=15  rem 7
            # fp16 spot-check (same dispatch logic as bf16, different opcode)
            [1,  9, (16, 32), (16, 32), (4, 2), torch.float16],
            [1, 13, (16, 32), (16, 32), (4, 2), torch.float16],
            # K_PAD=16 (m64n16k16 path) for both dtypes
            [1,  9, (16, 32), (16, 32), (4, 4), torch.bfloat16],
            [1,  9, (16, 32), (16, 32), (4, 4), torch.float16],
            # pscale=2 with unaligned BC
            [1,  9, (32, 64), (16, 32), (4, 2), torch.bfloat16],
            # Multi-batch unaligned: B*C=10 via B=2,C=5
            [2,  5, (16, 32), (16, 32), (4, 2), torch.bfloat16],
            # fp32 → scalar K-packed path (no WGMMA)
            [1,  9, (16, 32), (16, 32), (4, 2), torch.float32],
            [1, 12, (16, 32), (16, 32), (4, 2), torch.float32],
        ],
        skip_on_empty=True,
    )
    def test_kpacked_kernel_handles_unaligned_bc(self, B, C, in_shape, out_shape, kernel_shape, dtype):
        """B*C not divisible by BC_TILE=8 must still produce correct output —
        last grid.y CTA straddles the BC boundary and is masked off."""
        if not cuda_kernels_is_available():
            self.skipTest("CUDA disco kernels not available")
        from torch_harmonics import DiscreteContinuousConvS2

        device = torch.device("cuda")
        nlat_in, _ = in_shape
        theta_cutoff = (kernel_shape[0] + 1) * torch.pi / float(nlat_in - 1)

        conv = DiscreteContinuousConvS2(
            in_channels=C, out_channels=C,
            in_shape=in_shape, out_shape=out_shape,
            kernel_shape=kernel_shape,
            basis_type="harmonic", basis_norm_mode="mean",
            bias=False, theta_cutoff=theta_cutoff,
            optimized_kernel=True, use_dense_kernel=True,
        ).to(device)

        if conv.psi_kpacked_K_pad is None:
            self.skipTest("kpacked buffers not built for harmonic basis (unexpected)")

        x = torch.randn(B, C, *in_shape, device=device, dtype=dtype)
        try:
            from torch_harmonics.disco import disco_kernels
        except Exception:
            self.skipTest("disco_kernels op namespace not loadable")
        if not hasattr(disco_kernels, "forward_dense_kpacked"):
            self.skipTest("forward_dense_kpacked op not registered")

        kernel_size = conv.kernel_size
        out_ref = disco_kernels.forward_dense.default(
            x.contiguous(),
            conv.psi_packed_idx, conv.psi_packed_vals, conv.psi_packed_count,
            kernel_size, conv.nlat_out, conv.nlon_out,
        )
        out_kpacked = disco_kernels.forward_dense_kpacked.default(
            x.contiguous(),
            conv.psi_kpacked_idx, conv.psi_kpacked_vals, conv.psi_kpacked_count,
            kernel_size, conv.nlat_out, conv.nlon_out,
        )

        atol = rtol = (1e-5 if dtype == torch.float32 else 5e-2)
        self.assertEqual(tuple(out_ref.shape), tuple(out_kpacked.shape))
        self.assertTrue(compare_tensors(
            f"kpacked vs ref @ B={B},C={C},dtype={dtype}",
            out_kpacked, out_ref, atol=atol, rtol=rtol, verbose=True))


if __name__ == "__main__":
    unittest.main()
