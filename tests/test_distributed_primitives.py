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
import torch_harmonics.distributed as thd
from torch_harmonics.distributed import (
    compute_split_shapes,
    scatter_to_polar_region,
    gather_from_polar_region,
    distributed_transpose_polar,
)

from testutils import (
    set_seed,
    setup_module,
    teardown_module,
    setup_class_from_context,
    split_tensor_dim,
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
            [ 4,  8, 32, 64,    2,    3],
            [ 4,  8, 33, 64,    2,    3],
            [ 4,  8, 32, 65,    2,    3],
            [ 4,  8, 33, 65,    2,    3],
            [ 1,  1,  7, 13,    2,    3],
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


if __name__ == "__main__":
    unittest.main()
