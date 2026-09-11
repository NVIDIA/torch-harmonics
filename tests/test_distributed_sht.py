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

import unittest

import torch
from parameterized import parameterized
from testutils import (
    compare_tensors,
    disable_tf32,
    gather_tensor_hw,
    reduce_success,
    set_seed,
    setup_class_from_context,
    setup_module,
    split_tensor_hw,
    teardown_module,
)

import torch_harmonics as th
import torch_harmonics.distributed as thd

# shared state
_DIST_CTX = {}


def setUpModule():
    setup_module(_DIST_CTX)


def tearDownModule():
    teardown_module(_DIST_CTX)


class TestDistributedSphericalHarmonicTransform(unittest.TestCase):
    """Test the distributed spherical harmonic transform module (CPU/CUDA if available)."""

    @classmethod
    def setUpClass(cls):
        setup_class_from_context(cls, _DIST_CTX)
        disable_tf32()

    def _split_helper(self, tensor):
        return split_tensor_hw(tensor, hdim=-2, wdim=-1, hsize=self.grid_size_h, wsize=self.grid_size_w, hrank=self.hrank, wrank=self.wrank)

    def _gather_helper_fwd(self, tensor, transform_dist):
        tensor_gather = gather_tensor_hw(
            tensor,
            hdim=-2,
            wdim=-1,
            hshapes=transform_dist.l_shapes,
            wshapes=transform_dist.m_shapes,
            hsize=self.grid_size_h,
            wsize=self.grid_size_w,
            hrank=self.hrank,
            wrank=self.wrank,
            hgroup=self.h_group,
            wgroup=self.w_group,
        )

        return tensor_gather

    def _gather_helper_bwd(self, tensor, transform_dist):

        tensor_gather = gather_tensor_hw(
            tensor,
            hdim=-2,
            wdim=-1,
            hshapes=transform_dist.lat_shapes,
            wshapes=transform_dist.lon_shapes,
            hsize=self.grid_size_h,
            wsize=self.grid_size_w,
            hrank=self.hrank,
            wrank=self.wrank,
            hgroup=self.h_group,
            wgroup=self.w_group,
        )

        return tensor_gather

    # Tolerances are atol=1e-5, rtol=1e-6, loosened from the near-exact agreement these rows
    # used to demand. The Legendre contraction is a distributed matmul: each polar rank
    # contracts the latitudes (forward) or degrees (inverse) it owns, and the partial sums are
    # combined by a reduce-scatter. Reassociating a float32 sum across ranks perturbs the result
    # by a couple of ULP of the *summands*.
    #
    # atol is what carries this, not rtol, and the reason matters: the worst offenders are
    # near-cancellation outputs -- elements of magnitude ~0.015 in a tensor whose scale is ~2,
    # produced by summing terms two orders larger. Their error is inherited from those terms,
    # so relative to the element itself it is ~1e-4 no matter how the sum is arranged. No rtol
    # can cover that without also asserting something false about small outputs. The criterion
    # that does match the numerics is absolute and scaled to the tensor's dynamic range,
    # eps_fp32 * max|x|, which is ~1e-6 here; 1e-5 leaves roughly 8x headroom over the largest
    # difference measured at H=2.
    #
    # This is a ULP-level allowance, not a general loosening: a genuine defect in the splitting
    # or the collective shows up as O(1e-3) or worse, far outside this band. Worth re-measuring
    # at large polar counts, since the spread grows slowly with the number of partial sums.
    @parameterized.expand(
        [
            # lmax automatically determined
            # Scalar SHT
            [32, 64, None, 32, 8, "equiangular", False, 1e-5, 1e-6],
            [32, 64, None, 32, 8, "legendre-gauss", False, 1e-5, 1e-6],
            [32, 64, None, 32, 8, "equiangular", False, 1e-5, 1e-6],
            [32, 64, None, 32, 8, "legendre-gauss", False, 1e-5, 1e-6],
            [32, 64, None, 32, 8, "equiangular", False, 1e-5, 1e-6],
            [32, 64, None, 32, 8, "legendre-gauss", False, 1e-5, 1e-6],
            [33, 64, None, 1, 10, "equiangular", False, 1e-5, 1e-6],
            [33, 64, None, 1, 10, "legendre-gauss", False, 1e-5, 1e-6],
            [8, 16, None, 1, 10, "equiangular", False, 1e-5, 1e-6],
            # fewer channels than the polar group size (gh #207): B*C must be padded
            # up to the group size before the channel-axis transposes. Looser tolerance
            # here: padding pushes the (inert) channel GEMM from the batch-1 algorithm the
            # serial reference uses into the batch>=2 regime, a ~1e-6 fp32 rounding shift.
            # Divisible-channel cases never cross that boundary, so they keep the default rtol.
            [32, 64, None, 1, 1, "equiangular", False, 1e-5, 1e-5],
            [32, 64, None, 1, 1, "legendre-gauss", False, 1e-5, 1e-5],
            [32, 64, None, 1, 2, "equiangular", False, 1e-5, 1e-5],
            [32, 64, None, 1, 1, "equiangular", True, 1e-5, 1e-5],
            [32, 64, None, 1, 2, "equiangular", True, 1e-5, 1e-5],
            # Vector SHT
            [32, 64, None, 32, 8, "equiangular", True, 1e-5, 1e-6],
            [32, 64, None, 32, 8, "legendre-gauss", True, 1e-5, 1e-6],
            [32, 64, None, 32, 8, "equiangular", True, 1e-5, 1e-6],
            [32, 64, None, 32, 8, "legendre-gauss", True, 1e-5, 1e-6],
            [32, 64, None, 32, 8, "equiangular", True, 1e-5, 1e-6],
            [32, 64, None, 32, 8, "legendre-gauss", True, 1e-5, 1e-6],
            [32, 64, None, 1, 10, "equiangular", True, 1e-5, 1e-6],
            [33, 64, None, 1, 10, "legendre-gauss", True, 1e-5, 1e-6],
            # downsampling:
            # Scalar SHT
            [32, 64, 8, 32, 8, "equiangular", False, 1e-5, 1e-6],
            [32, 64, 8, 32, 8, "legendre-gauss", False, 1e-5, 1e-6],
            [33, 64, 9, 1, 10, "equiangular", False, 1e-5, 1e-6],
            [33, 64, 8, 1, 10, "legendre-gauss", False, 1e-5, 1e-6],
            # Vector SHT
            [32, 64, 8, 32, 8, "equiangular", True, 1e-5, 1e-6],
            [32, 64, 8, 32, 8, "legendre-gauss", True, 1e-5, 1e-6],
            [33, 64, 9, 1, 10, "equiangular", True, 1e-5, 1e-6],
            [33, 64, 8, 1, 10, "legendre-gauss", True, 1e-5, 1e-6],
            # upsampling
            # Scalar SHT
            [32, 64, 64, 32, 8, "equiangular", False, 1e-5, 1e-6],
            [32, 64, 64, 32, 8, "legendre-gauss", False, 1e-5, 1e-6],
            [33, 64, 65, 1, 10, "equiangular", False, 1e-5, 1e-6],
            [33, 64, 64, 1, 10, "legendre-gauss", False, 1e-5, 1e-6],
            # Vector SHT
            [32, 64, 64, 32, 8, "equiangular", True, 1e-5, 1e-6],
            [32, 64, 64, 32, 8, "legendre-gauss", True, 1e-5, 1e-6],
            [33, 64, 65, 1, 10, "equiangular", True, 1e-5, 1e-6],
            [33, 64, 64, 1, 10, "legendre-gauss", True, 1e-5, 1e-6],
        ],
        skip_on_empty=True,
    )
    def test_distributed_sht(self, nlat, nlon, lmax, batch_size, num_chan, grid, vector, atol, rtol, verbose=False):

        set_seed(333)

        B, C, H, W = batch_size, num_chan, nlat, nlon

        # set up handles
        if vector:
            forward_transform_local = th.RealVectorSHT(nlat=H, nlon=W, lmax=lmax, mmax=lmax, grid=grid).to(self.device)
            forward_transform_dist = thd.DistributedRealVectorSHT(nlat=H, nlon=W, lmax=lmax, mmax=lmax, grid=grid).to(self.device)
        else:
            forward_transform_local = th.RealSHT(nlat=H, nlon=W, lmax=lmax, mmax=lmax, grid=grid).to(self.device)
            forward_transform_dist = thd.DistributedRealSHT(nlat=H, nlon=W, lmax=lmax, mmax=lmax, grid=grid).to(self.device)

        # create tensors
        if vector:
            inp_full = torch.randn((B, C, 2, H, W), dtype=torch.float32, device=self.device)
        else:
            inp_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device)

        # local transform
        # FWD pass
        inp_full.requires_grad = True
        out_full = forward_transform_local(inp_full)

        # create grad for backward
        with torch.no_grad():
            # create full grad
            ograd_full = torch.randn_like(out_full)

        # BWD pass
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        # distributed transform
        # FWD pass
        inp_local = self._split_helper(inp_full.detach().clone())
        inp_local.requires_grad = True
        out_local = forward_transform_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full)
        out_local = forward_transform_dist(inp_local)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        # Print diagnostics from rank 0 only; assert the all-reduced verdict on every
        # rank so a failure on any rank fails the test consistently (see reduce_success).
        verbose = verbose and self.world_rank == 0

        # evaluate FWD pass
        out_gather_full = self._gather_helper_fwd(out_local, forward_transform_dist)
        ok = compare_tensors("output", out_full, out_gather_full, atol=atol, rtol=rtol, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "output")

        # evaluate BWD pass
        igrad_gather_full = self._gather_helper_bwd(igrad_local, forward_transform_dist)
        ok = compare_tensors("gradients", igrad_full, igrad_gather_full, atol=atol, rtol=rtol, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "gradients")

    # Tolerances are atol=1e-5, rtol=1e-6, loosened from the near-exact agreement these rows
    # used to demand. The Legendre contraction is a distributed matmul: each polar rank
    # contracts the latitudes (forward) or degrees (inverse) it owns, and the partial sums are
    # combined by a reduce-scatter. Reassociating a float32 sum across ranks perturbs the result
    # by a couple of ULP of the *summands*.
    #
    # atol is what carries this, not rtol, and the reason matters: the worst offenders are
    # near-cancellation outputs -- elements of magnitude ~0.015 in a tensor whose scale is ~2,
    # produced by summing terms two orders larger. Their error is inherited from those terms,
    # so relative to the element itself it is ~1e-4 no matter how the sum is arranged. No rtol
    # can cover that without also asserting something false about small outputs. The criterion
    # that does match the numerics is absolute and scaled to the tensor's dynamic range,
    # eps_fp32 * max|x|, which is ~1e-6 here; 1e-5 leaves roughly 8x headroom over the largest
    # difference measured at H=2.
    #
    # This is a ULP-level allowance, not a general loosening: a genuine defect in the splitting
    # or the collective shows up as O(1e-3) or worse, far outside this band. Worth re-measuring
    # at large polar counts, since the spread grows slowly with the number of partial sums.
    @parameterized.expand(
        [
            # lmax automatically determined
            # Scalar SHT
            [32, 64, None, 32, 8, "equiangular", False, 1e-5, 1e-6],
            [32, 64, None, 32, 8, "legendre-gauss", False, 1e-5, 1e-6],
            [32, 64, None, 32, 8, "equiangular", False, 1e-5, 1e-6],
            [32, 64, None, 32, 8, "legendre-gauss", False, 1e-5, 1e-6],
            [32, 64, None, 32, 8, "equiangular", False, 1e-5, 1e-6],
            [32, 64, None, 32, 8, "legendre-gauss", False, 1e-5, 1e-6],
            [33, 64, None, 1, 10, "equiangular", False, 1e-5, 1e-6],
            [33, 64, None, 1, 10, "legendre-gauss", False, 1e-5, 1e-6],
            # fewer channels than the polar group size (gh #207): B*C must be padded
            # up to the group size before the channel-axis transposes. Looser tolerance
            # here: padding pushes the (inert) channel GEMM from the batch-1 algorithm the
            # serial reference uses into the batch>=2 regime, a ~1e-6 fp32 rounding shift.
            # Divisible-channel cases never cross that boundary, so they keep the default rtol.
            [32, 64, None, 1, 1, "equiangular", False, 1e-5, 1e-5],
            [32, 64, None, 1, 1, "legendre-gauss", False, 1e-5, 1e-5],
            [32, 64, None, 1, 2, "equiangular", False, 1e-5, 1e-5],
            [32, 64, None, 1, 1, "equiangular", True, 1e-5, 1e-5],
            [32, 64, None, 1, 2, "equiangular", True, 1e-5, 1e-5],
            # Vector SHT
            [32, 64, None, 32, 8, "equiangular", True, 1e-5, 1e-6],
            [32, 64, None, 32, 8, "legendre-gauss", True, 1e-5, 1e-6],
            [32, 64, None, 32, 8, "equiangular", True, 1e-5, 1e-6],
            [32, 64, None, 32, 8, "legendre-gauss", True, 1e-5, 1e-6],
            [32, 64, None, 32, 8, "equiangular", True, 1e-5, 1e-6],
            [32, 64, None, 32, 8, "legendre-gauss", True, 1e-5, 1e-6],
            [33, 64, None, 1, 10, "equiangular", True, 1e-5, 1e-6],
            [33, 64, None, 1, 10, "legendre-gauss", True, 1e-5, 1e-6],
            # downsampling (SHT is upsampling)
            # Scalar SHT
            [32, 64, 64, 32, 8, "equiangular", False, 1e-5, 1e-6],
            [32, 64, 64, 32, 8, "legendre-gauss", False, 1e-5, 1e-6],
            [33, 64, 65, 1, 10, "equiangular", False, 1e-5, 1e-6],
            [33, 64, 64, 1, 10, "legendre-gauss", False, 1e-5, 1e-6],
            # Vector SHT
            [32, 64, 64, 32, 8, "equiangular", True, 1e-5, 1e-6],
            [32, 64, 64, 32, 8, "legendre-gauss", True, 1e-5, 1e-6],
            [33, 64, 65, 1, 10, "equiangular", True, 1e-5, 1e-6],
            [33, 64, 64, 1, 10, "legendre-gauss", True, 1e-5, 1e-6],
            # upsampling (SHT is downsampling)
            # Scalar SHT
            [32, 64, 8, 32, 8, "equiangular", False, 1e-5, 1e-6],
            [32, 64, 8, 32, 8, "legendre-gauss", False, 1e-5, 1e-6],
            [33, 64, 9, 1, 10, "equiangular", False, 1e-5, 1e-6],
            [33, 64, 8, 1, 10, "legendre-gauss", False, 1e-5, 1e-6],
            # Vector SHT
            [32, 64, 8, 32, 8, "equiangular", True, 1e-5, 1e-6],
            [32, 64, 8, 32, 8, "legendre-gauss", True, 1e-5, 1e-6],
            [33, 64, 9, 1, 10, "equiangular", True, 1e-5, 1e-6],
            [33, 64, 8, 1, 10, "legendre-gauss", True, 1e-5, 1e-6],
        ],
        skip_on_empty=True,
    )
    def test_distributed_isht(self, nlat, nlon, lmax, batch_size, num_chan, grid, vector, atol, rtol, verbose=True):

        set_seed(333)

        B, C, H, W = batch_size, num_chan, nlat, nlon

        if vector:
            forward_transform_local = th.RealVectorSHT(nlat=H, nlon=W, lmax=lmax, mmax=lmax, grid=grid).to(self.device)
            backward_transform_local = th.InverseRealVectorSHT(nlat=H, nlon=W, lmax=lmax, mmax=lmax, grid=grid).to(self.device)
            backward_transform_dist = thd.DistributedInverseRealVectorSHT(nlat=H, nlon=W, lmax=lmax, mmax=lmax, grid=grid).to(self.device)
        else:
            forward_transform_local = th.RealSHT(nlat=H, nlon=W, lmax=lmax, mmax=lmax, grid=grid).to(self.device)
            backward_transform_local = th.InverseRealSHT(nlat=H, nlon=W, lmax=lmax, mmax=lmax, grid=grid).to(self.device)
            backward_transform_dist = thd.DistributedInverseRealSHT(nlat=H, nlon=W, lmax=lmax, mmax=lmax, grid=grid).to(self.device)

        # create tensors
        if vector:
            dummy_full = torch.randn((B, C, 2, H, W), dtype=torch.float32, device=self.device)
        else:
            dummy_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device)
        inp_full = forward_transform_local(dummy_full)

        #############################################################
        # local transform
        #############################################################
        # FWD pass
        inp_full.requires_grad = True
        out_full = backward_transform_local(inp_full)

        # create grad for backward
        with torch.no_grad():
            # create full grad
            ograd_full = torch.randn_like(out_full)

        # BWD pass
        out_full.backward(ograd_full)

        # repeat once due to known irfft bug
        inp_full.grad = None
        out_full = backward_transform_local(inp_full)
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        # distributed transform
        # FWD pass
        inp_local = self._split_helper(inp_full.detach().clone())
        inp_local.requires_grad = True
        out_local = backward_transform_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full)
        out_local = backward_transform_dist(inp_local)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        # Print diagnostics from rank 0 only; assert the all-reduced verdict on every
        # rank so a failure on any rank fails the test consistently (see reduce_success).
        verbose = verbose and self.world_rank == 0

        # evaluate FWD pass
        out_gather_full = self._gather_helper_bwd(out_local, backward_transform_dist)
        ok = compare_tensors("output", out_full, out_gather_full, atol=atol, rtol=rtol, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "output")

        # evaluate BWD pass
        igrad_gather_full = self._gather_helper_fwd(igrad_local, backward_transform_dist)
        ok = compare_tensors("gradients", igrad_full, igrad_gather_full, atol=atol, rtol=rtol, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "gradients")

    @parameterized.expand(
        [
            # nlat, nlon, lmax, grid, vector
            [32, 64, None, "equiangular", False],
            [32, 64, None, "legendre-gauss", False],
            [33, 64, None, "equiangular", False],
            [32, 64, 8, "equiangular", False],
            [32, 64, None, "equiangular", True],
            [33, 64, None, "legendre-gauss", True],
            [32, 64, 8, "equiangular", True],
        ],
        skip_on_empty=True,
    )
    def test_legendre_blocks(self, nlat, nlon, lmax, grid, vector, verbose=False):
        """Each rank's precomputed Legendre buffer equals its slice of the serial one.

        The distributed transforms build only the block they keep rather than the whole
        table and discarding most of it, which means the recurrences are entered at an
        offset: the sectoral seed is walked up to the rank's first order and the three-term
        recurrence up to its first degree, storing nothing until then. This test isolates
        that from the transform itself, so an off-by-one in the offsets shows up here rather
        than as a diffuse accuracy failure in the round trip.

        Each rank checks only its own block against the corresponding cut-out of the serial
        construction, so there is no collective involved and a failure identifies the rank.

        Agreement is expected to be exact -- the restricted build performs the same
        elementwise operations on the same values -- so the tolerance is only insurance
        against a last-bit difference, far tighter than anything an indexing error survives.
        """

        set_seed(333)

        if vector:
            fwd_dist = thd.DistributedRealVectorSHT(nlat=nlat, nlon=nlon, lmax=lmax, mmax=lmax, grid=grid).to(self.device)
            fwd_local = th.RealVectorSHT(nlat=nlat, nlon=nlon, lmax=lmax, mmax=lmax, grid=grid).to(self.device)
            inv_dist = thd.DistributedInverseRealVectorSHT(nlat=nlat, nlon=nlon, lmax=lmax, mmax=lmax, grid=grid).to(self.device)
            inv_local = th.InverseRealVectorSHT(nlat=nlat, nlon=nlon, lmax=lmax, mmax=lmax, grid=grid).to(self.device)
            fwd_buf, inv_buf = "weights", "dpct"
        else:
            fwd_dist = thd.DistributedRealSHT(nlat=nlat, nlon=nlon, lmax=lmax, mmax=lmax, grid=grid).to(self.device)
            fwd_local = th.RealSHT(nlat=nlat, nlon=nlon, lmax=lmax, mmax=lmax, grid=grid).to(self.device)
            inv_dist = thd.DistributedInverseRealSHT(nlat=nlat, nlon=nlon, lmax=lmax, mmax=lmax, grid=grid).to(self.device)
            inv_local = th.InverseRealSHT(nlat=nlat, nlon=nlon, lmax=lmax, mmax=lmax, grid=grid).to(self.device)
            fwd_buf, inv_buf = "weights", "pct"

        # offsets are recomputed here from the per-rank shape lists rather than read off the
        # transform, so a wrong offset in the construction is not masked by reusing it
        lat_off = sum(fwd_dist.lat_shapes[: self.hrank])
        lat_loc = fwd_dist.lat_shapes[self.hrank]
        l_off = sum(inv_dist.l_shapes[: self.hrank])
        l_loc = inv_dist.l_shapes[self.hrank]
        m_off = sum(fwd_dist.m_shapes[: self.wrank])
        m_loc = fwd_dist.m_shapes[self.wrank]

        # forward: local orders, all degrees, local latitudes -- (..., m, l, k)
        got = getattr(fwd_dist, fwd_buf)
        ref = getattr(fwd_local, fwd_buf)[..., m_off : m_off + m_loc, :, lat_off : lat_off + lat_loc]

        if verbose:
            print(f"forward block on rank ({self.hrank},{self.wrank}): {tuple(got.shape)} vs {tuple(ref.shape)}")

        self.assertEqual(tuple(got.shape), tuple(ref.shape), "forward block shape")
        ok = compare_tensors("forward legendre block", got, ref.contiguous(), atol=1e-14, rtol=1e-14, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "forward legendre block")

        # inverse: local orders, all latitudes, local degrees -- (..., m, k, l)
        got = getattr(inv_dist, inv_buf)
        ref = getattr(inv_local, inv_buf)[..., m_off : m_off + m_loc, :, l_off : l_off + l_loc]

        self.assertEqual(tuple(got.shape), tuple(ref.shape), "inverse block shape")
        ok = compare_tensors("inverse legendre block", got, ref.contiguous(), atol=1e-14, rtol=1e-14, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "inverse legendre block")


if __name__ == "__main__":
    unittest.main()
