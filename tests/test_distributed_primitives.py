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
#

import unittest
from parameterized import parameterized

import torch
import torch.distributed as dist
import torch_harmonics.distributed as thd
from torch_harmonics.distributed import (
    compute_split_shapes,
    scatter_to_polar_region,
    gather_from_polar_region,
    distributed_transpose_polar,
    reduce_from_polar_region,
    reduce_from_azimuth_region,
)

from testutils import (
    set_seed,
    setup_module,
    teardown_module,
    setup_class_from_context,
    split_tensor_dim,
    compare_tensors,
)

_DIST_CTX = {}

def setUpModule():
    setup_module(_DIST_CTX)

def tearDownModule():
    teardown_module(_DIST_CTX)


class TestComputeSplitShapes(unittest.TestCase):

    def test_single_chunk(self):
        self.assertEqual(compute_split_shapes(1, 1), [1])
        self.assertEqual(compute_split_shapes(7, 1), [7])
        self.assertEqual(compute_split_shapes(100, 1), [100])

    def test_even_split(self):
        self.assertEqual(compute_split_shapes(10, 2), [5, 5])
        self.assertEqual(compute_split_shapes(12, 3), [4, 4, 4])
        self.assertEqual(compute_split_shapes(8, 4), [2, 2, 2, 2])

    def test_uneven_split_is_balanced(self):
        shapes = compute_split_shapes(10, 3)
        self.assertEqual(sum(shapes), 10)
        self.assertEqual(len(shapes), 3)
        self.assertLessEqual(max(shapes) - min(shapes), 1)

        shapes = compute_split_shapes(7, 3)
        self.assertEqual(sum(shapes), 7)
        self.assertEqual(len(shapes), 3)
        self.assertLessEqual(max(shapes) - min(shapes), 1)

    def test_all_chunks_nonempty(self):
        for size in range(1, 33):
            for num_chunks in range(1, size + 1):
                shapes = compute_split_shapes(size, num_chunks)
                self.assertEqual(sum(shapes), size)
                self.assertEqual(len(shapes), num_chunks)
                self.assertTrue(all(s > 0 for s in shapes))
                self.assertLessEqual(max(shapes) - min(shapes), 1)

    def test_size_equals_num_chunks(self):
        shapes = compute_split_shapes(4, 4)
        self.assertEqual(shapes, [1, 1, 1, 1])

    def test_fails_when_size_less_than_num_chunks(self):
        with self.assertRaises(AssertionError):
            compute_split_shapes(2, 5)

        with self.assertRaises(AssertionError):
            compute_split_shapes(1, 2)

        with self.assertRaises(AssertionError):
            compute_split_shapes(0, 1)


class TestDistributedScatterGather(unittest.TestCase):
    """Test scatter and gather primitives across the polar group."""

    @classmethod
    def setUpClass(cls):
        setup_class_from_context(cls, _DIST_CTX)

    @parameterized.expand(
        [
            # B,  C,  H,  W, split_dim
            [ 4,  8, 32, 64, -2],
            [ 4,  8, 33, 64, -2],
            [ 4,  8, 32, 64, -1],
            [ 4,  8, 32, 65, -1],
            [ 1,  1,  7, 13, -2],
        ],
        skip_on_empty=True,
    )
    def test_scatter(self, B, C, H, W, split_dim):
        set_seed(333)
        x_full = torch.randn(B, C, H, W, device=self.device)

        comm_size = thd.polar_group_size()
        x_expected = split_tensor_dim(
            x_full, dim=split_dim, dimsize=comm_size, dimrank=self.hrank,
        )

        x_local = scatter_to_polar_region(x_full, split_dim)

        self.assertEqual(x_local.shape, x_expected.shape)
        self.assertTrue(torch.equal(x_local, x_expected))

    @parameterized.expand(
        [
            # B,  C,  H,  W, split_dim
            [ 4,  8, 32, 64, -2],
            [ 4,  8, 33, 64, -2],
            [ 4,  8, 32, 64, -1],
            [ 4,  8, 32, 65, -1],
            [ 1,  1,  7, 13, -2],
        ],
        skip_on_empty=True,
    )
    def test_gather(self, B, C, H, W, split_dim):
        set_seed(333)
        x_full = torch.randn(B, C, H, W, device=self.device)

        comm_size = thd.polar_group_size()
        dim = split_dim if split_dim >= 0 else x_full.dim() + split_dim
        shapes = compute_split_shapes(x_full.shape[dim], comm_size)

        x_local = scatter_to_polar_region(x_full, split_dim)
        x_gathered = gather_from_polar_region(x_local, split_dim, shapes)

        self.assertEqual(x_gathered.shape, x_full.shape)
        self.assertTrue(torch.equal(x_gathered, x_full))


class TestDistributedTranspose(unittest.TestCase):
    """Test distributed transpose across the polar group."""

    @classmethod
    def setUpClass(cls):
        setup_class_from_context(cls, _DIST_CTX)

    @parameterized.expand(
        [
            # B,  C,  H,  W, dim0, dim1
            [ 4,  8,  32,  64,   2,   3],
            [ 4,  8,  33,  64,   2,   3],
            [ 4,  8,  32,  65,   2,   3],
            [ 4,  8,  33,  65,   2,   3],
            [ 1,  1,   7,  13,   2,   3],
            [ 4,  9,  33,  65,   2,   3],
            [ 1,  4, 361, 361,   2,   3],
        ],
        skip_on_empty=True,
    )
    def test_transpose_polar(self, B, C, H, W, dim0, dim1):
        """Transpose from split-on-dim1 to split-on-dim0, then gather dim0 to verify."""
        set_seed(333)
        x_full = torch.randn(B, C, H, W, device=self.device)

        comm_size = thd.polar_group_size()
        dim1_shapes = compute_split_shapes(x_full.shape[dim1], comm_size)

        x_local = split_tensor_dim(
            x_full, dim=dim1, dimsize=comm_size, dimrank=self.hrank,
        )

        x_transposed = distributed_transpose_polar(x_local, (dim0, dim1), dim1_shapes)

        dim0_shapes = compute_split_shapes(x_full.shape[dim0], comm_size)
        x_gathered = gather_from_polar_region(x_transposed, dim0, dim0_shapes)

        self.assertEqual(x_gathered.shape, x_full.shape)
        self.assertTrue(torch.equal(x_gathered, x_full))


class TestDistributedReduce(unittest.TestCase):
    """
    Test reduce_from_polar_region and reduce_from_azimuth_region.

    Forward: each rank holds a different local tensor.  The reduce primitive
    must produce the same result as manually gathering all local tensors and
    summing them (the "local sum + global sum" path).

    Backward: the adjoint of a broadcast-sum (all_reduce) is the identity –
    the upstream gradient flows through each rank unchanged.  We verify this
    by comparing the computed input gradient against the upstream gradient dy.
    """

    @classmethod
    def setUpClass(cls):
        setup_class_from_context(cls, _DIST_CTX)

    # ------------------------------------------------------------------
    # Reference helpers – use dist.all_gather so we never call the op
    # under test as part of the reference computation.
    # ------------------------------------------------------------------

    def _polar_gather_sum(self, x_local):
        """Global sum over the polar group via all_gather + Python sum."""
        polar_size = thd.polar_group_size()
        if polar_size > 1:
            x_all = [torch.empty_like(x_local) for _ in range(polar_size)]
            dist.all_gather(x_all, x_local.detach().contiguous(), group=thd.polar_group())
            return torch.stack(x_all, dim=0).sum(dim=0)
        else:
            return x_local.detach().clone()

    def _azimuth_gather_sum(self, x_local):
        """Global sum over the azimuth group via all_gather + Python sum."""
        az_size = thd.azimuth_group_size()
        if az_size > 1:
            x_all = [torch.empty_like(x_local) for _ in range(az_size)]
            dist.all_gather(x_all, x_local.detach().contiguous(), group=thd.azimuth_group())
            return torch.stack(x_all, dim=0).sum(dim=0)
        else:
            return x_local.detach().clone()

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    @parameterized.expand(
        [
            # B,  C,   H,  W
            [2,   8,  16, 32],
            [1,   4,   7, 13],
            [3,  16,   8, 16],
        ],
        skip_on_empty=True,
    )
    def test_reduce_from_polar_region_fwd_bwd(self, B, C, H, W):
        set_seed(333)

        # Give each rank a distinct contribution so that a missing reduce is
        # immediately visible: x_local = randn + hrank.
        x_local = torch.randn(B, C, H, W, device=self.device) + float(self.hrank)

        # --- Forward reference: gather-and-sum without using the primitive ---
        ref = self._polar_gather_sum(x_local)

        # --- Distributed forward ---
        x = x_local.clone().requires_grad_(True)
        out = reduce_from_polar_region(x)

        self.assertTrue(
            compare_tensors("reduce_from_polar_region fwd", ref, out,
                            atol=1e-5, rtol=1e-4, verbose=True),
            "forward output does not match the reference global sum",
        )

        # --- Backward: the gradient must pass through unchanged ---
        dy = torch.randn_like(out)
        out.backward(dy)

        self.assertTrue(
            compare_tensors("reduce_from_polar_region bwd", dy, x.grad,
                            atol=1e-5, rtol=1e-4, verbose=True),
            "input gradient does not match the upstream gradient (expected pass-through)",
        )

    @parameterized.expand(
        [
            # B,  C,   H,  W
            [2,   8,  16, 32],
            [1,   4,   7, 13],
            [3,  16,   8, 16],
        ],
        skip_on_empty=True,
    )
    def test_reduce_from_azimuth_region_fwd_bwd(self, B, C, H, W):
        set_seed(333)

        x_local = torch.randn(B, C, H, W, device=self.device) + float(self.wrank)

        # --- Forward reference ---
        ref = self._azimuth_gather_sum(x_local)

        # --- Distributed forward ---
        x = x_local.clone().requires_grad_(True)
        out = reduce_from_azimuth_region(x)

        self.assertTrue(
            compare_tensors("reduce_from_azimuth_region fwd", ref, out,
                            atol=1e-5, rtol=1e-4, verbose=True),
            "forward output does not match the reference global sum",
        )

        # --- Backward: pass-through ---
        dy = torch.randn_like(out)
        out.backward(dy)

        self.assertTrue(
            compare_tensors("reduce_from_azimuth_region bwd", dy, x.grad,
                            atol=1e-5, rtol=1e-4, verbose=True),
            "input gradient does not match the upstream gradient (expected pass-through)",
        )


if __name__ == "__main__":
    unittest.main()
