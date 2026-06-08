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

import os
import unittest
from unittest import mock

import torch
import torch.distributed as dist
from parameterized import parameterized
from testutils import (
    compare_tensors,
    disable_tf32,
    gather_tensor_hw,
    maybe_autocast,
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


class TestDistributedDiscreteContinuousConvolution(unittest.TestCase):
    """Test the distributed discrete-continuous convolution module."""

    @classmethod
    def setUpClass(cls):
        setup_class_from_context(cls, _DIST_CTX)
        disable_tf32()

    def setUp(self):
        # Sync at test entry too — belt and braces. If a preceding test
        # raised mid-flight (e.g., an assertion failure right after
        # backward), its module locals may be GC'd while NCCL writes are
        # still in flight, and Python's destruction order between the
        # exception unwind and our tearDown isn't guaranteed. Starting
        # each test from a fully drained device avoids inheriting that
        # state.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # Rank-sync at test entry. All parametrized tests run the same
        # code on all ranks (no rank-divergent skipTest paths), so the
        # barrier won't deadlock — and it catches NCCL-level state leaks
        # that cuda.synchronize() alone doesn't reset.
        if dist.is_initialized():
            dist.barrier()

    def tearDown(self):
        # Each test creates a fresh conv module whose module-owned recv
        # buffers go out of scope at end-of-test. The caching allocator
        # then marks that memory free using only the compute stream's
        # view of liveness — but NCCL writes on its internal stream may
        # still be in flight, so the next test's allocations can recycle
        # memory that NCCL is still writing to → cross-test corruption.
        # torch.cuda.synchronize() waits on ALL streams on the device,
        # including NCCL's internal stream, which is sufficient to
        # ensure each test starts from a fully clean local state.
        #
        # In production (long-lived stacked-layer models) module-owned
        # buffers persist for the model's lifetime, so this scenario
        # doesn't arise and no in-code sync is needed.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _split_helper(self, tensor):
        return split_tensor_hw(tensor, hdim=-2, wdim=-1, hsize=self.grid_size_h, wsize=self.grid_size_w, hrank=self.hrank, wrank=self.wrank)

    def _gather_helper_fwd(self, tensor, convolution_dist):

        tensor_gather = gather_tensor_hw(
            tensor,
            hdim=-2,
            wdim=-1,
            hshapes=convolution_dist.lat_out_shapes,
            wshapes=convolution_dist.lon_out_shapes,
            hsize=self.grid_size_h,
            wsize=self.grid_size_w,
            hrank=self.hrank,
            wrank=self.wrank,
            hgroup=self.h_group,
            wgroup=self.w_group,
        )

        return tensor_gather

    def _allreduce_param_grad(self, tensor):
        """Sum a per-rank parameter gradient across the polar (h) and
        azimuth (w) groups to recover the global gradient.

        Each rank's parameter gradient holds the local-spatial contribution
        to the global gradient (see e.g. the einsum in the bwd grad_w
        path: it reduces over local b/H_out/W_out only). To compare against
        a serial reference we have to sum across the two split dimensions
        — independently, so this works for any (h, w) grid layout and
        doesn't implicitly assume the world group equals h × w.

        Returns a clone (allreduce is in-place) so the caller doesn't
        mutate ``conv_dist.weight.grad`` and friends.
        """
        out = tensor.clone()
        if self.grid_size_h > 1:
            dist.all_reduce(out, group=self.h_group)
        if self.grid_size_w > 1:
            dist.all_reduce(out, group=self.w_group)
        return out

    def _gather_helper_bwd(self, tensor, convolution_dist):

        tensor_gather = gather_tensor_hw(
            tensor,
            hdim=-2,
            wdim=-1,
            hshapes=convolution_dist.lat_in_shapes,
            wshapes=convolution_dist.lon_in_shapes,
            hsize=self.grid_size_h,
            wsize=self.grid_size_w,
            hrank=self.hrank,
            wrank=self.wrank,
            hgroup=self.h_group,
            wgroup=self.w_group,
        )

        return tensor_gather

    @parameterized.expand(
        [
            # fp32 tests
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "a2a", False, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3, 2), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "a2a", False, 1e-6, 1e-5],
            [64, 128, 32, 64, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "a2a", False, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 2, "equiangular", "equiangular", torch.float32, False, "a2a", False, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 6, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "a2a", False, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, True, "a2a", False, 1e-6, 1e-5],
            [65, 128, 65, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, True, "a2a", False, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3, 2), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, True, "a2a", False, 1e-6, 1e-5],
            [64, 128, 128, 256, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, True, "a2a", False, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 2, "equiangular", "equiangular", torch.float32, True, "a2a", False, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 6, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, True, "a2a", False, 1e-6, 1e-5],
            [65, 128, 65, 128, 32, 8, (3, 4), "harmonic", "mean", 1, "equiangular", "equiangular", torch.float32, False, "a2a", False, 1e-6, 1e-5],
            [65, 128, 65, 128, 32, 8, (3, 4), "harmonic", "mean", 1, "equiangular", "equiangular", torch.float32, True, "a2a", False, 1e-6, 1e-5],
            [33, 64, 65, 128, 32, 8, (3, 4), "harmonic", "mean", 1, "equiangular", "equiangular", torch.float32, True, "a2a", False, 1e-6, 1e-5],
            [65, 128, 33, 64, 32, 8, (3, 4), "harmonic", "mean", 1, "equiangular", "equiangular", torch.float32, False, "a2a", False, 1e-6, 1e-5],
            # fp64 tests
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float64, False, "a2a", False, 1e-6, 1e-6],
            [64, 128, 32, 64, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float64, False, "a2a", False, 1e-6, 1e-6],
            [64, 128, 64, 128, 32, 8, (3, 2), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float64, True, "a2a", False, 1e-6, 1e-6],
            [64, 128, 128, 256, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float64, True, "a2a", False, 1e-6, 1e-6],
            [65, 128, 65, 128, 32, 8, (3, 4), "harmonic", "mean", 1, "equiangular", "equiangular", torch.float64, False, "a2a", False, 1e-6, 1e-6],
            [65, 128, 65, 128, 32, 8, (3, 4), "harmonic", "mean", 1, "equiangular", "equiangular", torch.float64, True, "a2a", False, 1e-6, 1e-6],
            # ring tests (non-transpose only; ring is CUDA-only and the
            # transpose class doesn't accept method=). Each ring scenario
            # runs twice: once with the legacy per-step recv allocation
            # (p2p_buffer=False), once with the pre-allocated recv pool
            # opted in via TORCH_HARMONICS_P2P_BUFFER=1 (p2p_buffer=True).
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring", False, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring", True, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3, 2), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring", False, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3, 2), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring", True, 1e-6, 1e-5],
            [64, 128, 32, 64, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring", False, 1e-6, 1e-5],
            [64, 128, 32, 64, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring", True, 1e-6, 1e-5],
            [65, 128, 65, 128, 32, 8, (3, 4), "harmonic", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring", False, 1e-6, 1e-5],
            [65, 128, 65, 128, 32, 8, (3, 4), "harmonic", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring", True, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring_fused", False, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring_fused", True, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3, 2), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring_fused", False, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3, 2), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring_fused", True, 1e-6, 1e-5],
            [64, 128, 32, 64, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring_fused", False, 1e-6, 1e-5],
            [64, 128, 32, 64, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring_fused", True, 1e-6, 1e-5],
            [65, 128, 65, 128, 32, 8, (3, 4), "harmonic", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring_fused", False, 1e-6, 1e-5],
            [65, 128, 65, 128, 32, 8, (3, 4), "harmonic", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring_fused", True, 1e-6, 1e-5],
            # ring_stream / ring_fused_stream: same algorithms with a real
            # dedicated CUDA stream for the ring P2P. Validates that the
            # explicit comm-stream + event-sync path matches the
            # single-stream path numerically.
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring_stream", False, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring_stream", True, 1e-6, 1e-5],
            [64, 128, 32, 64, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring_stream", True, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring_fused_stream", False, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring_fused_stream", True, 1e-6, 1e-5],
            [64, 128, 32, 64, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring_fused_stream", True, 1e-6, 1e-5],
            # ring_masked / ring_fused_masked: force the masked kernel
            # via TORCH_HARMONICS_RING_FWD_BAND=0 so the masked branch is
            # covered on small test grids (the default path is the band kernel).
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring_masked", False, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring_masked", True, 1e-6, 1e-5],
            [64, 128, 32, 64, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring_masked", True, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring_fused_masked", False, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring_fused_masked", True, 1e-6, 1e-5],
            [64, 128, 32, 64, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, "ring_fused_masked", True, 1e-6, 1e-5],
            # fp16 / bf16 AMP coverage. Tolerances are looser than the fp32
            # rows because the bf16/fp16 cuBLAS rounding at each einsum's
            # output dominates the abs error, and cross-rank reductions
            # (polar reduce_scatter for ring; polar all_reduce for a2a)
            # amplify cancellation at near-zero output positions — see the
            # cross-rank-cancellation note in
            # feedback_amp_fp16_cancellation. Starting tolerances mirror the
            # serial test's sparse-against-dense AMP rows; tighten/loosen
            # per-row if empirics warrant.
            #
            # a2a — covers both non-transpose and transpose branches.
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float16, False, "a2a", False, 2e-2, 1e-2],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.bfloat16, False, "a2a", False, 5e-2, 5e-2],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float16, True, "a2a", False, 2e-2, 1e-2],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.bfloat16, True, "a2a", False, 5e-2, 5e-2],
            # ring (K-expanded activation saved across fwd/bwd).
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float16, False, "ring", True, 2e-2, 1e-2],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.bfloat16, False, "ring", True, 5e-2, 5e-2],
            # ring_fused (K-expanded activation dropped, recomputed in bwd).
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float16, False, "ring_fused", True, 2e-2, 1e-2],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.bfloat16, False, "ring_fused", True, 5e-2, 5e-2],
            # ring_stream / ring_fused_stream AMP rows — same tolerances.
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float16, False, "ring_stream", True, 2e-2, 1e-2],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.bfloat16, False, "ring_fused_stream", True, 5e-2, 5e-2],
        ],
        skip_on_empty=True,
    )
    def test_distributed_disco_conv(
        self,
        nlat_in,
        nlon_in,
        nlat_out,
        nlon_out,
        batch_size,
        num_chan,
        kernel_shape,
        basis_type,
        basis_norm_mode,
        groups,
        grid_in,
        grid_out,
        dtype,
        transpose,
        algorithm,
        p2p_buffer,
        atol,
        rtol,
        verbose=True,
    ):

        # Translate the single ``algorithm`` test parameter into the
        # distributed-conv kwargs. ``a2a`` is the default A2A path;
        # ``ring`` selects the ring algorithm with the K-expanded
        # activation saved across fwd/bwd; ``ring_fused`` additionally
        # drops that activation (bwd recomputes it via a second ring fwd).
        # The ``_stream`` variants pass a dedicated CUDA stream for the
        # ring P2P (so send/recv overlaps with the per-step kernel on
        # the compute stream); the non-``_stream`` variants pass
        # comm_stream=None (single-stream fallback).
        # The transpose class does not accept ``method=`` yet, so ignore
        # algorithm for the transpose branch.
        if algorithm == "a2a":
            method, fused, use_comm_stream, force_masked = "a2a", False, False, False
        elif algorithm == "ring":
            method, fused, use_comm_stream, force_masked = "ring", False, False, False
        elif algorithm == "ring_fused":
            method, fused, use_comm_stream, force_masked = "ring", True, False, False
        elif algorithm == "ring_stream":
            method, fused, use_comm_stream, force_masked = "ring", False, True, False
        elif algorithm == "ring_fused_stream":
            method, fused, use_comm_stream, force_masked = "ring", True, True, False
        elif algorithm == "ring_masked":
            # Force the masked-kernel fallback path via the env var so this
            # branch is covered on the existing small test grids.
            method, fused, use_comm_stream, force_masked = "ring", False, False, True
        elif algorithm == "ring_fused_masked":
            method, fused, use_comm_stream, force_masked = "ring", True, False, True
        else:
            raise ValueError(f"unknown algorithm={algorithm!r}")

        # Ring path is CUDA-only — skip gracefully when CUDA isn't there.
        if method == "ring" and not torch.cuda.is_available():
            self.skipTest("method='ring' is CUDA-only")

        # Allocate the dedicated comm stream for the _stream variants.
        # Stays None for the single-stream variants.
        comm_stream = torch.cuda.Stream(device=self.device) if use_comm_stream else None

        # For AMP dtypes the modules + inputs stay in float32 and autocast
        # handles the downcast inside fwd/bwd — same pattern as the serial
        # convolution tests in test_convolution.py.
        is_amp = dtype in (torch.float16, torch.bfloat16)
        module_dtype = torch.float32 if is_amp else dtype

        set_seed(333)

        B, C, H, W = batch_size, num_chan, nlat_in, nlon_in

        disco_args = dict(
            in_channels=C,
            out_channels=C,
            in_shape=(nlat_in, nlon_in),
            out_shape=(nlat_out, nlon_out),
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            kernel_shape=kernel_shape,
            groups=groups,
            grid_in=grid_in,
            grid_out=grid_out,
            bias=True,
        )

        # The distributed conv reads TORCH_HARMONICS_P2P_BUFFER once in
        # __init__ and stores the flag on the instance.
        # TORCH_HARMONICS_RING_FWD_BAND is read by the C++ ring fwd kernel
        # every launch — setting it to "0" selects the masked kernel instead
        # of the default band kernel so that branch is covered on small grids.
        env_override = {"TORCH_HARMONICS_P2P_BUFFER": "1" if p2p_buffer else "0"}
        if force_masked:
            env_override["TORCH_HARMONICS_RING_FWD_BAND"] = "0"

        # set up handles
        if transpose:
            # Transpose class does not yet accept method= / fused= — the
            # ``algorithm`` and ``p2p_buffer`` test parameters are ignored
            # on this branch.
            conv_local = th.DiscreteContinuousConvTransposeS2(**disco_args).to(dtype=module_dtype, device=self.device)
            with mock.patch.dict(os.environ, env_override):
                conv_dist = thd.DistributedDiscreteContinuousConvTransposeS2(**disco_args).to(dtype=module_dtype, device=self.device)
        else:
            conv_local = th.DiscreteContinuousConvS2(**disco_args).to(dtype=module_dtype, device=self.device)
            with mock.patch.dict(os.environ, env_override):
                conv_dist = thd.DistributedDiscreteContinuousConvS2(
                    **disco_args,
                    method=method,
                    fused=fused,
                    comm_stream=comm_stream,
                ).to(dtype=module_dtype, device=self.device)

        # copy the weights from the local conv into the dist conv
        with torch.no_grad():
            conv_dist.weight.copy_(conv_local.weight)
            if disco_args["bias"]:
                conv_dist.bias.copy_(conv_local.bias)

        # create tensors
        inp_full = torch.randn((B, C, H, W), dtype=module_dtype, device=self.device)

        # local conv
        # FWD pass
        inp_full.requires_grad = True
        with maybe_autocast(self.device.type, dtype):
            out_full = conv_local(inp_full)

        # create grad for backward
        with torch.no_grad():
            # create full grad
            ograd_full = torch.randn_like(out_full)

        # BWD pass
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        # distributed conv. The env patch is re-applied around the
        # fwd/bwd because TORCH_HARMONICS_RING_FWD_BAND is read by
        # the C++ kernel at each launch (whereas TORCH_HARMONICS_P2P_BUFFER
        # was already consumed during __init__).
        # FWD pass
        inp_local = self._split_helper(inp_full.detach().clone())
        inp_local.requires_grad = True
        with mock.patch.dict(os.environ, env_override):
            with maybe_autocast(self.device.type, dtype):
                out_local = conv_dist(inp_local)

            # BWD pass
            ograd_local = self._split_helper(ograd_full)
            with maybe_autocast(self.device.type, dtype):
                out_local = conv_dist(inp_local)
            out_local.backward(ograd_local)
            igrad_local = inp_local.grad.clone()

        # evaluate FWD pass
        out_gather_full = self._gather_helper_fwd(out_local, conv_dist)
        self.assertTrue(compare_tensors("output", out_full, out_gather_full, atol=atol, rtol=rtol, verbose=verbose))

        # evaluate BWD pass
        igrad_gather_full = self._gather_helper_bwd(igrad_local, conv_dist)
        self.assertTrue(compare_tensors("gradients", igrad_full, igrad_gather_full, atol=atol, rtol=rtol, verbose=verbose))

        # evaluate parameter gradients — local per-rank contributions
        # summed across the h and w groups must match the serial gradient.
        # Parameter grads are sums over batch + local spatial → accumulate
        # reduction noise that per-element output / input-grad checks
        # don't see. Use a higher factor for fp32 (where the base tolerance
        # is tight at 1e-6/1e-5 and H100's a2a reductions can drift up to
        # ~5e-3 absolute) than for AMP (where the base is already 4–5
        # orders of magnitude looser; the same factor would make the
        # bound meaninglessly large).
        param_grad_tol_factor = 10.0 if is_amp else 1000.0
        pg_atol, pg_rtol = atol * param_grad_tol_factor, rtol * param_grad_tol_factor
        if conv_dist.weight.grad is not None:
            wgrad = self._allreduce_param_grad(conv_dist.weight.grad)
            self.assertTrue(compare_tensors("weight grad", conv_local.weight.grad, wgrad, atol=pg_atol, rtol=pg_rtol, verbose=verbose))
        if getattr(conv_dist, "bias", None) is not None and conv_dist.bias.grad is not None:
            bgrad = self._allreduce_param_grad(conv_dist.bias.grad)
            self.assertTrue(compare_tensors("bias grad", conv_local.bias.grad, bgrad, atol=pg_atol, rtol=pg_rtol, verbose=verbose))

    @parameterized.expand(
        [
            # Stacked ring convs in series. The point of this test is to
            # validate that module-owned recv pools prevent cross-layer
            # corruption (one layer's NCCL writes leaking into the next
            # layer's allocations). No tearDown sync intervenes within a
            # single forward pass, so this exercises the production
            # stacked-layer scenario that the per-test sync would mask.
            #
            # Tolerances are explicit per row (looser than the single-conv
            # test to absorb the noise of chaining two convolutions and
            # their backwards). Per-row atol/rtol applies to the output
            # and input-gradient checks; parameter-gradient checks scale
            # those by ``pgrad_tol_factor`` inside the test.
            #
            # Each row: (method, fused, kernel_shape, basis, dtype,
            #           p2p_buffer, use_comm_stream, force_masked,
            #           atol, rtol).
            ["ring", False, (3, 4), "harmonic", torch.float32, True, False, False, 1e-5, 1e-4],
            ["ring", True, (3, 4), "harmonic", torch.float32, True, False, False, 1e-5, 1e-4],
            ["ring", True, (3, 4), "harmonic", torch.float32, True, True, False, 1e-5, 1e-4],
            ["ring", False, (3), "piecewise linear", torch.float32, True, False, False, 1e-5, 1e-4],
            ["ring", True, (3), "piecewise linear", torch.float32, True, False, False, 1e-5, 1e-4],
            # AMP rows — base tolerances loosened above the single-conv
            # AMP rows to absorb two-layer fp16/bf16 cast accumulation.
            # bf16's weight-grad reduction noise can reach ~1 absolute
            # at elements where the reference is 0; pg_atol = atol * 10
            # then covers it.
            ["ring", False, (3, 4), "harmonic", torch.float16, True, False, False, 5e-2, 5e-2],
            ["ring", True, (3, 4), "harmonic", torch.bfloat16, True, False, False, 2e-1, 2e-1],
        ],
        skip_on_empty=True,
    )
    def test_distributed_disco_conv_stacked(
        self,
        method,
        fused,
        kernel_shape,
        basis_type,
        dtype,
        p2p_buffer,
        use_comm_stream,
        force_masked,
        atol,
        rtol,
        verbose=True,
    ):
        """Two ring convs chained in series.

        Validates that module-owned recv buffers prevent cross-layer
        memory aliasing (layer 1's NCCL writes leaking into layer 2's
        allocations). With per-call recv pools, layer 2's torch.empty
        could land on memory NCCL was still writing to from layer 1;
        module ownership keeps the pool alive across layers.

        Compares the chained forward output against a serial reference
        and the input-gradient flowing back through both layers.
        """
        if method == "ring" and not torch.cuda.is_available():
            self.skipTest("method='ring' is CUDA-only")

        is_amp = dtype in (torch.float16, torch.bfloat16)
        module_dtype = torch.float32 if is_amp else dtype

        set_seed(444)

        # Canonical 65x128 shape (matches the previously-broken harmonic
        # config from earlier in this branch's debug history). Two layers
        # keep the same spatial shape so they can be chained directly.
        nlat, nlon = 65, 128
        B, C = 32, 8

        common_args = dict(
            in_channels=C,
            out_channels=C,
            in_shape=(nlat, nlon),
            out_shape=(nlat, nlon),
            basis_type=basis_type,
            basis_norm_mode="mean",
            kernel_shape=kernel_shape,
            groups=1,
            grid_in="equiangular",
            grid_out="equiangular",
            bias=True,
        )

        env_override = {"TORCH_HARMONICS_P2P_BUFFER": "1" if p2p_buffer else "0"}
        if force_masked:
            env_override["TORCH_HARMONICS_RING_FWD_BAND"] = "0"

        # Serial reference: two stacked local convs.
        conv1_local = th.DiscreteContinuousConvS2(**common_args).to(dtype=module_dtype, device=self.device)
        conv2_local = th.DiscreteContinuousConvS2(**common_args).to(dtype=module_dtype, device=self.device)

        # Distributed: two stacked ring convs. Same comm_stream is shared
        # across both layers to mirror typical training-script usage.
        comm_stream = torch.cuda.Stream(device=self.device) if use_comm_stream else None
        with mock.patch.dict(os.environ, env_override):
            conv1_dist = thd.DistributedDiscreteContinuousConvS2(**common_args, method=method, fused=fused, comm_stream=comm_stream).to(dtype=module_dtype, device=self.device)
            conv2_dist = thd.DistributedDiscreteContinuousConvS2(**common_args, method=method, fused=fused, comm_stream=comm_stream).to(dtype=module_dtype, device=self.device)

        # Sync weights into the dist convs (per-layer).
        with torch.no_grad():
            conv1_dist.weight.copy_(conv1_local.weight)
            conv2_dist.weight.copy_(conv2_local.weight)
            conv1_dist.bias.copy_(conv1_local.bias)
            conv2_dist.bias.copy_(conv2_local.bias)

        # Inputs + grad targets.
        inp_full = torch.randn((B, C, nlat, nlon), dtype=module_dtype, device=self.device)

        # Serial reference: forward through both convs.
        inp_full.requires_grad = True
        with maybe_autocast(self.device.type, dtype):
            mid_full = conv1_local(inp_full)
            out_full = conv2_local(mid_full)

        with torch.no_grad():
            ograd_full = torch.randn_like(out_full)

        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        # Distributed: forward through both convs in series.
        inp_local = self._split_helper(inp_full.detach().clone())
        inp_local.requires_grad = True
        with mock.patch.dict(os.environ, env_override):
            with maybe_autocast(self.device.type, dtype):
                mid_local = conv1_dist(inp_local)
                out_local = conv2_dist(mid_local)

            ograd_local = self._split_helper(ograd_full)
            # Re-run for autograd graph (matches the single-conv test).
            with maybe_autocast(self.device.type, dtype):
                mid_local = conv1_dist(inp_local)
                out_local = conv2_dist(mid_local)
            out_local.backward(ograd_local)
            igrad_local = inp_local.grad.clone()

        # Compare chained output.
        out_gather_full = self._gather_helper_fwd(out_local, conv2_dist)
        self.assertTrue(compare_tensors("stacked output", out_full, out_gather_full, atol=atol, rtol=rtol, verbose=verbose))

        # Compare input gradient flowing back through both layers.
        igrad_gather_full = self._gather_helper_bwd(igrad_local, conv1_dist)
        self.assertTrue(compare_tensors("stacked gradients", igrad_full, igrad_gather_full, atol=atol, rtol=rtol, verbose=verbose))

        # Compare per-layer parameter gradients (allreduced over h, w).
        # Param grads accumulate reduction noise over batch + local
        # spatial; loosen relative to the row's per-element tolerance.
        # Use a higher factor for fp32 (where the base atol/rtol is tight
        # at 1e-5 / 1e-4) than for AMP (where the base is already large
        # enough that a high factor makes the bound meaningless).
        pgrad_tol_factor = 10.0 if is_amp else 100.0
        pg_atol, pg_rtol = atol * pgrad_tol_factor, rtol * pgrad_tol_factor
        for layer_idx, (conv_dist, conv_local_ref) in enumerate([(conv1_dist, conv1_local), (conv2_dist, conv2_local)], start=1):
            if conv_dist.weight.grad is not None:
                wgrad = self._allreduce_param_grad(conv_dist.weight.grad)
                self.assertTrue(compare_tensors(f"layer{layer_idx} weight grad", conv_local_ref.weight.grad, wgrad, atol=pg_atol, rtol=pg_rtol, verbose=verbose))
            if getattr(conv_dist, "bias", None) is not None and conv_dist.bias.grad is not None:
                bgrad = self._allreduce_param_grad(conv_dist.bias.grad)
                self.assertTrue(compare_tensors(f"layer{layer_idx} bias grad", conv_local_ref.bias.grad, bgrad, atol=pg_atol, rtol=pg_rtol, verbose=verbose))


if __name__ == "__main__":
    unittest.main()
