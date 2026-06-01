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

import torch
import torch.distributed as dist
from parameterized import parameterized
from testutils import (
    compare_tensors,
    set_seed,
    setup_class_from_context,
    setup_module,
    split_tensor_dim,
    teardown_module,
)

import torch_harmonics.distributed as thd
from torch_harmonics.distributed import (
    compute_split_shapes,
    distributed_transpose_polar,
    gather_from_polar_region,
    reduce_from_azimuth_region,
    reduce_from_polar_region,
    reduce_from_scatter_to_azimuth_region,
    reduce_from_scatter_to_polar_region,
    scatter_to_polar_region,
    split_tensor_along_dim,
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
        with self.assertRaises(RuntimeError):
            compute_split_shapes(2, 5)

        with self.assertRaises(RuntimeError):
            compute_split_shapes(1, 2)

        with self.assertRaises(RuntimeError):
            compute_split_shapes(0, 1)


class TestDistributedScatterGather(unittest.TestCase):
    """Test scatter and gather primitives across the polar group."""

    @classmethod
    def setUpClass(cls):
        setup_class_from_context(cls, _DIST_CTX)

    @parameterized.expand(
        [
            # B,  C,  H,  W, split_dim
            [4, 8, 32, 64, -2],
            [4, 8, 33, 64, -2],
            [4, 8, 32, 64, -1],
            [4, 8, 32, 65, -1],
            [1, 1, 7, 13, -2],
        ],
        skip_on_empty=True,
    )
    def test_scatter(self, B, C, H, W, split_dim):
        set_seed(333)
        x_full = torch.randn(B, C, H, W, device=self.device)

        comm_size = thd.polar_group_size()
        x_expected = split_tensor_dim(
            x_full,
            dim=split_dim,
            dimsize=comm_size,
            dimrank=self.hrank,
        )

        x_local = scatter_to_polar_region(x_full, split_dim)

        self.assertEqual(x_local.shape, x_expected.shape)
        self.assertTrue(torch.equal(x_local, x_expected))

    @parameterized.expand(
        [
            # B,  C,  H,  W, split_dim
            [4, 8, 32, 64, -2],
            [4, 8, 33, 64, -2],
            [4, 8, 32, 64, -1],
            [4, 8, 32, 65, -1],
            [1, 1, 7, 13, -2],
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
            [4, 8, 32, 64, 2, 3],
            [4, 8, 33, 64, 2, 3],
            [4, 8, 32, 65, 2, 3],
            [4, 8, 33, 65, 2, 3],
            [1, 1, 7, 13, 2, 3],
            [4, 9, 33, 65, 2, 3],
            [1, 4, 361, 361, 2, 3],
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
            x_full,
            dim=dim1,
            dimsize=comm_size,
            dimrank=self.hrank,
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
            [2, 8, 16, 32],
            [1, 4, 7, 13],
            [3, 16, 8, 16],
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
            compare_tensors("reduce_from_polar_region fwd", ref, out, atol=1e-5, rtol=1e-4, verbose=True),
            "forward output does not match the reference global sum",
        )

        # --- Backward: the gradient must pass through unchanged ---
        dy = torch.randn_like(out)
        out.backward(dy)

        self.assertTrue(
            compare_tensors("reduce_from_polar_region bwd", dy, x.grad, atol=1e-5, rtol=1e-4, verbose=True),
            "input gradient does not match the upstream gradient (expected pass-through)",
        )

    @parameterized.expand(
        [
            # B,  C,   H,  W
            [2, 8, 16, 32],
            [1, 4, 7, 13],
            [3, 16, 8, 16],
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
            compare_tensors("reduce_from_azimuth_region fwd", ref, out, atol=1e-5, rtol=1e-4, verbose=True),
            "forward output does not match the reference global sum",
        )

        # --- Backward: pass-through ---
        dy = torch.randn_like(out)
        out.backward(dy)

        self.assertTrue(
            compare_tensors("reduce_from_azimuth_region bwd", dy, x.grad, atol=1e-5, rtol=1e-4, verbose=True),
            "input gradient does not match the upstream gradient (expected pass-through)",
        )


class TestReduceScatter(unittest.TestCase):
    """
    Test reduce_from_scatter_to_polar_region and
    reduce_from_scatter_to_azimuth_region (fused all_reduce + scatter via
    NCCL reduce_scatter) for both even and uneven splits along the scattered
    dim.

    Forward semantics: each rank holds a different x_local with the same
    shape; the op sums them across the polar (resp. azimuth) group and
    returns each rank's slice along ``dim_``.

    Backward semantics: the adjoint of reduce-scatter is all-gather, so the
    full input gradient on each rank must be the concatenation of all ranks'
    upstream gradients along ``dim_``.
    """

    @classmethod
    def setUpClass(cls):
        setup_class_from_context(cls, _DIST_CTX)

    def _polar_all_gather(self, x_local, dim, shapes):
        """All-gather x_local (per-rank tensor) along ``dim`` using
        per-rank ``shapes``. Used to build the forward reference and to
        verify the backward gradient."""
        polar_size = thd.polar_group_size()
        if polar_size <= 1:
            return x_local.detach().clone()

        out_list = []
        local_shape = list(x_local.shape)
        for s in shapes:
            local_shape[dim] = s
            out_list.append(torch.empty(local_shape, dtype=x_local.dtype, device=x_local.device))
        dist.all_gather(out_list, x_local.detach().contiguous(), group=thd.polar_group())
        return torch.cat(out_list, dim=dim)

    def _azimuth_all_gather(self, x_local, dim, shapes):
        """All-gather across the azimuth group; mirrors _polar_all_gather."""
        az_size = thd.azimuth_group_size()
        if az_size <= 1:
            return x_local.detach().clone()

        out_list = []
        local_shape = list(x_local.shape)
        for s in shapes:
            local_shape[dim] = s
            out_list.append(torch.empty(local_shape, dtype=x_local.dtype, device=x_local.device))
        dist.all_gather(out_list, x_local.detach().contiguous(), group=thd.azimuth_group())
        return torch.cat(out_list, dim=dim)

    @parameterized.expand(
        [
            # B, C,  H,  W, split_dim   ── H and W chosen to cover both
            #                              even and uneven splits for polar
            #                              group sizes typical at launch (2 or 4).
            [2, 4, 16, 32, -2],  # H=16  → even for P∈{1,2,4,8}
            [2, 4, 17, 32, -2],  # H=17  → uneven for P>1
            [2, 4, 33, 32, -2],  # H=33  → uneven for P∈{2,4}
            [2, 4, 16, 32, -1],  # W=32  → even
            [2, 4, 16, 33, -1],  # W=33  → uneven
            [1, 1, 7, 13, -2],  # tiny + uneven for P∈{2,3,4}
            [1, 1, 7, 13, -1],
            [2, 4, 11, 9, -2],  # both dims uneven across most P
        ],
        skip_on_empty=True,
    )
    def test_reduce_from_scatter_to_polar(self, B, C, H, W, split_dim):
        set_seed(333)
        polar_size = thd.polar_group_size()

        # x_local shape is identical on all ranks; the +hrank offset makes
        # the per-rank contribution distinguishable so a missing reduce
        # can't hide.
        x_local = torch.randn(B, C, H, W, device=self.device) + float(self.hrank)

        ndim = x_local.dim()
        dim = split_dim if split_dim >= 0 else ndim + split_dim

        expected_shapes = compute_split_shapes(x_local.shape[dim], polar_size)
        my_expected = expected_shapes[self.hrank]

        # Forward reference: gather-and-sum, then slice my own chunk.
        if polar_size > 1:
            x_all = [torch.empty_like(x_local) for _ in range(polar_size)]
            dist.all_gather(x_all, x_local.detach().contiguous(), group=thd.polar_group())
            x_full = torch.stack(x_all, dim=0).sum(dim=0)
        else:
            x_full = x_local.detach().clone()

        offsets = [0]
        for s in expected_shapes:
            offsets.append(offsets[-1] + s)
        slicer = [slice(None)] * ndim
        slicer[dim] = slice(offsets[self.hrank], offsets[self.hrank + 1])
        ref = x_full[tuple(slicer)].contiguous()

        x = x_local.clone().requires_grad_(True)
        out = reduce_from_scatter_to_polar_region(x, split_dim)

        # Shape contract: my_expected along scattered dim, others unchanged.
        self.assertEqual(
            out.shape[dim],
            my_expected,
            f"reduce_scatter output dim {dim} = {out.shape[dim]}, " f"expected {my_expected} from compute_split_shapes",
        )
        ref_shape = list(x_local.shape)
        ref_shape[dim] = my_expected
        self.assertEqual(tuple(out.shape), tuple(ref_shape))

        self.assertTrue(
            compare_tensors("reduce_from_scatter_to_polar fwd", ref, out, atol=1e-5, rtol=1e-4, verbose=True),
            "forward output does not match the gather-then-slice reference",
        )

        # Backward: adjoint = all_gather along ``dim``.
        dy = torch.randn_like(out)
        out.backward(dy)

        if polar_size > 1:
            grad_ref = self._polar_all_gather(dy, dim, expected_shapes)
        else:
            grad_ref = dy.clone()

        self.assertEqual(
            tuple(x.grad.shape),
            tuple(x_local.shape),
            "backward gradient shape must equal the forward input shape",
        )
        self.assertTrue(
            compare_tensors("reduce_from_scatter_to_polar bwd", grad_ref, x.grad, atol=1e-5, rtol=1e-4, verbose=True),
            "input gradient does not match the all-gathered upstream gradient",
        )

    @parameterized.expand(
        [
            # B, C,  H,  W, split_dim   ── azimuth-natural cases scatter
            #                              along W; we also exercise H to
            #                              confirm the primitive is dim-agnostic.
            [2, 4, 16, 32, -1],  # W=32  → even for P∈{1,2,4,8}
            [2, 4, 16, 33, -1],  # W=33  → uneven for P>1
            [2, 4, 16, 65, -1],  # W=65  → uneven for P∈{2,4}
            [2, 4, 16, 32, -2],  # H=16  → even
            [2, 4, 17, 32, -2],  # H=17  → uneven for P>1
            [1, 1, 7, 13, -1],  # tiny + uneven for P∈{2,3,4}
            [1, 1, 7, 13, -2],
            [2, 4, 11, 9, -1],
        ],
        skip_on_empty=True,
    )
    def test_reduce_from_scatter_to_azimuth(self, B, C, H, W, split_dim):
        set_seed(333)
        az_size = thd.azimuth_group_size()

        x_local = torch.randn(B, C, H, W, device=self.device) + float(self.wrank)

        ndim = x_local.dim()
        dim = split_dim if split_dim >= 0 else ndim + split_dim

        expected_shapes = compute_split_shapes(x_local.shape[dim], az_size)
        my_expected = expected_shapes[self.wrank]

        if az_size > 1:
            x_all = [torch.empty_like(x_local) for _ in range(az_size)]
            dist.all_gather(x_all, x_local.detach().contiguous(), group=thd.azimuth_group())
            x_full = torch.stack(x_all, dim=0).sum(dim=0)
        else:
            x_full = x_local.detach().clone()

        offsets = [0]
        for s in expected_shapes:
            offsets.append(offsets[-1] + s)
        slicer = [slice(None)] * ndim
        slicer[dim] = slice(offsets[self.wrank], offsets[self.wrank + 1])
        ref = x_full[tuple(slicer)].contiguous()

        x = x_local.clone().requires_grad_(True)
        out = reduce_from_scatter_to_azimuth_region(x, split_dim)

        self.assertEqual(
            out.shape[dim],
            my_expected,
            f"reduce_scatter output dim {dim} = {out.shape[dim]}, " f"expected {my_expected} from compute_split_shapes",
        )
        ref_shape = list(x_local.shape)
        ref_shape[dim] = my_expected
        self.assertEqual(tuple(out.shape), tuple(ref_shape))

        self.assertTrue(
            compare_tensors("reduce_from_scatter_to_azimuth fwd", ref, out, atol=1e-5, rtol=1e-4, verbose=True),
            "forward output does not match the gather-then-slice reference",
        )

        dy = torch.randn_like(out)
        out.backward(dy)

        if az_size > 1:
            grad_ref = self._azimuth_all_gather(dy, dim, expected_shapes)
        else:
            grad_ref = dy.clone()

        self.assertEqual(
            tuple(x.grad.shape),
            tuple(x_local.shape),
            "backward gradient shape must equal the forward input shape",
        )
        self.assertTrue(
            compare_tensors("reduce_from_scatter_to_azimuth bwd", grad_ref, x.grad, atol=1e-5, rtol=1e-4, verbose=True),
            "input gradient does not match the all-gathered upstream gradient",
        )


class TestRingExchange(unittest.TestCase):
    """Ring exchange across the azimuth group, as used by
    DistributedNeighborhoodAttentionS2.

    Strategy:
      1. Build a global tensor that is identical on every rank (fixed seed)
         and split it deterministically into `az_size` chunks along its last
         dimension.
      2. Each rank starts holding its own chunk (`az_rank`).
      3. Walk the ring for `az_size` steps. At step t, every rank should be
         holding the chunk that originally belonged to rank
         `(az_rank + t) mod az_size`. We compare against the corresponding
         slice of the global tensor before each exchange.
      4. After verifying, post a non-blocking ring exchange (send to prev,
         recv from next) and continue.
    """

    @classmethod
    def setUpClass(cls):
        setup_class_from_context(cls, _DIST_CTX)
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Ring exchange uses dist.batch_isend_irecv which currently " "requires the NCCL backend (CUDA)")

    @parameterized.expand(
        [
            # B,  C,  H, nlon_global
            [2, 16, 8, 64],
            [2, 16, 8, 65],  # uneven split: shapes differ across ranks
            [1, 4, 3, 16],
            [4, 8, 16, 128],
        ],
        skip_on_empty=True,
    )
    def test_ring_kv_full_cycle(self, B, C, H, nlon_global):
        # _ring_kv lives in distributed_attention because it's the only caller;
        # the function itself is a generic ring-exchange primitive.
        from torch_harmonics.distributed.distributed_attention import _ring_kv

        az_group = thd.azimuth_group()
        az_size = thd.azimuth_group_size()
        az_rank = thd.azimuth_group_rank()

        # az_size == 1 is intentionally not skipped: the ring is then a no-op
        # (the loop runs once, verifies the local chunk equals the reference,
        # and exits without posting any P2POp). That is itself worth confirming.

        # Same seed everywhere → every rank reconstructs the identical global
        # tensor and so the per-source reference chunks all agree.
        set_seed(333)
        # Use different channel counts for kw and vw to catch any accidental
        # cross-tensor bleed (e.g. wrong send/recv ordering inside _ring_kv).
        kw_full = torch.randn(B, C, H, nlon_global, device=self.device, dtype=torch.float32)
        vw_full = torch.randn(B, C + 1, H, nlon_global, device=self.device, dtype=torch.float32)

        kw_ref = split_tensor_along_dim(kw_full, dim=-1, num_chunks=az_size)
        vw_ref = split_tensor_along_dim(vw_full, dim=-1, num_chunks=az_size)

        # Each rank starts holding its own chunk.
        kw_chunk = kw_ref[az_rank].clone()
        vw_chunk = vw_ref[az_rank].clone()

        for step in range(az_size):
            src_rank = (az_rank + step) % az_size

            self.assertTrue(
                torch.equal(kw_chunk, kw_ref[src_rank]),
                f"step {step} on rank {az_rank}: kw_chunk does not match " f"reference chunk from source rank {src_rank}",
            )
            self.assertTrue(
                torch.equal(vw_chunk, vw_ref[src_rank]),
                f"step {step} on rank {az_rank}: vw_chunk does not match " f"reference chunk from source rank {src_rank}",
            )

            # No exchange after the last verification.
            if step == az_size - 1:
                break

            # Next source rank determines the receive-buffer width because
            # the global split can be uneven.
            next_src = (az_rank + step + 1) % az_size
            next_nlon = kw_ref[next_src].shape[-1]
            recv_kw, recv_vw, reqs = _ring_kv(
                kw_chunk,
                vw_chunk,
                az_group,
                next_nlon,
                next_nlon,
            )
            for req in reqs:
                req.wait()
            kw_chunk = recv_kw.clone()
            vw_chunk = recv_vw.clone()


if __name__ == "__main__":
    unittest.main()
