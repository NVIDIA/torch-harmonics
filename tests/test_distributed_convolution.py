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

import torch
import torch.distributed as dist
from parameterized import parameterized
from testutils import (
    compare_tensors,
    disable_tf32,
    gather_tensor_hw,
    maybe_autocast,
    reduce_success,
    set_seed,
    setup_class_from_context,
    setup_module,
    split_tensor_hw,
    teardown_module,
)

import torch_harmonics as th
import torch_harmonics.distributed as thd

# Opt-in gate for slow / large-grid parameterized cases (e.g. 721x1440 ERA5-like
# shapes). Mirrors the TORCH_HARMONICS_RUN_PERF_TESTS pattern in
# tests/test_attention.py and tests/test_convolution.py, and the slow gate in
# tests/test_distributed_attention.py.
_run_slow_tests = os.getenv("TORCH_HARMONICS_RUN_SLOW_TESTS", "0") == "1"

# (nlat_in, nlon_in, nlat_out, nlon_out) shapes whose parameterized cases are
# gated behind TORCH_HARMONICS_RUN_SLOW_TESTS=1.
_SLOW_DISCO_SHAPES = frozenset(
    {
        (721, 1440, 721, 1440),
        (721, 1440, 360, 720),
    }
)

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
            # ---- fused=False : standard a2a (K-expanded saved for backward) ----
            # fp32
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, False, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3, 2), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, False, 1e-6, 1e-5],
            [64, 128, 32, 64, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, False, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 2, "equiangular", "equiangular", torch.float32, False, False, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 6, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, False, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, True, False, 1e-6, 1e-5],
            [65, 128, 65, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, True, False, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3, 2), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, True, False, 1e-6, 1e-5],
            [64, 128, 128, 256, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, True, False, 1e-6, 1e-5],
            [65, 128, 65, 128, 32, 8, (3, 4), "harmonic", "mean", 1, "equiangular", "equiangular", torch.float32, False, False, 1e-6, 1e-5],
            [65, 128, 65, 128, 32, 8, (3, 4), "harmonic", "mean", 1, "equiangular", "equiangular", torch.float32, True, False, 1e-6, 1e-5],
            [65, 128, 33, 64, 32, 8, (3, 4), "harmonic", "mean", 1, "equiangular", "equiangular", torch.float32, False, False, 1e-6, 1e-5],
            # group coverage: depthwise (groups == n_channels) and a groupsize>1
            # split (C=12, groups=3 -> groupsize=4).
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 8, "equiangular", "equiangular", torch.float32, False, False, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 12, (3), "piecewise linear", "mean", 3, "equiangular", "equiangular", torch.float32, False, False, 1e-6, 1e-5],
            # fp64
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float64, False, False, 1e-6, 1e-6],
            [64, 128, 32, 64, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float64, False, False, 1e-6, 1e-6],
            [64, 128, 64, 128, 32, 8, (3, 2), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float64, True, False, 1e-6, 1e-6],
            [65, 128, 65, 128, 32, 8, (3, 4), "harmonic", "mean", 1, "equiangular", "equiangular", torch.float64, False, False, 1e-6, 1e-6],
            # ERA5-like grids, gated behind TORCH_HARMONICS_RUN_SLOW_TESTS=1.
            # batch_size and num_chan dialed down (2, 8) vs the rest of the suite (32, 8)
            # to keep the working set under a few GB at these resolutions.
            [721, 1440, 721, 1440, 2, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, False, 1e-6, 1e-5],
            [721, 1440, 360, 720, 2, 8, (3), "piecewise linear", "mean", 1, "equiangular", "legendre-gauss", torch.float32, False, False, 1e-6, 1e-5],
            # ---- fused=True : reordered a2a (CUDA + optimized kernels) ----
            # non-transpose only; downsample + harmonic.
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, True, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3, 2), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, True, 1e-6, 1e-5],
            [64, 128, 32, 64, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, True, 1e-6, 1e-5],
            [65, 128, 65, 128, 32, 8, (3, 4), "harmonic", "mean", 1, "equiangular", "equiangular", torch.float32, False, True, 1e-6, 1e-5],
            [65, 128, 33, 64, 32, 8, (3, 4), "harmonic", "mean", 1, "equiangular", "equiangular", torch.float32, False, True, 1e-6, 1e-5],
            # group coverage for the padded grouped path:
            #  groups=1            -> within-group channel split (no padding)
            #  groups=2 (gs=4)     -> split cuts a group at az>=4 (padding)
            #  groups=3,C=12 (gs=4)-> split cuts a group at az=2 and az=4 (padding)
            #  groups=C (gs=1)     -> depthwise; every channel a group (no-pad fast path)
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 2, "equiangular", "equiangular", torch.float32, False, True, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 12, (3), "piecewise linear", "mean", 3, "equiangular", "equiangular", torch.float32, False, True, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 8, "equiangular", "equiangular", torch.float32, False, True, 1e-6, 1e-5],
            # ---- AMP (fp16/bf16) ----
            # Each dtype runs fused off AND on (non-transpose). The transpose
            # class has no ``fused`` argument, so it runs fused=False only
            # (fused=True there would be an identical duplicate).
            # fp16
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float16, False, False, 2e-2, 1e-2],
            # Harmonic AMP rows exercise distributed kpacked tensor-core paths:
            # (2,2) -> K=4 -> K_PAD=8; (3,3) -> K=9 -> K_PAD=16.
            [64, 128, 64, 128, 8, 8, (2, 2), "harmonic", "mean", 1, "equiangular", "equiangular", torch.float16, False, False, 5e-2, 1e-2],
            [64, 128, 64, 128, 8, 8, (3, 3), "harmonic", "mean", 1, "equiangular", "equiangular", torch.float16, False, False, 5e-2, 1e-2],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float16, False, True, 2e-2, 1e-2],
            [64, 128, 64, 128, 8, 8, (2, 2), "harmonic", "mean", 1, "equiangular", "equiangular", torch.float16, False, True, 5e-2, 1e-2],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float16, True, False, 2e-2, 1e-2],
            # bf16
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.bfloat16, False, False, 5e-2, 5e-2],
            [64, 128, 64, 128, 8, 8, (2, 2), "harmonic", "mean", 1, "equiangular", "equiangular", torch.bfloat16, False, False, 3e-1, 5e-2],
            [64, 128, 64, 128, 8, 8, (3, 3), "harmonic", "mean", 1, "equiangular", "equiangular", torch.bfloat16, False, False, 3e-1, 5e-2],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.bfloat16, False, True, 5e-2, 5e-2],
            [64, 128, 64, 128, 8, 8, (2, 2), "harmonic", "mean", 1, "equiangular", "equiangular", torch.bfloat16, False, True, 3e-1, 5e-2],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.bfloat16, True, False, 5e-2, 5e-2],
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
        fused,
        atol,
        rtol,
        verbose=True,
    ):
        if (nlat_in, nlon_in, nlat_out, nlon_out) in _SLOW_DISCO_SHAPES and not _run_slow_tests:
            self.skipTest("slow test; set TORCH_HARMONICS_RUN_SLOW_TESTS=1 to run")

        # ``fused`` mirrors the serial conv: False -> standard a2a (K-expanded
        # saved), True -> reordered a2a (einsum-first, K-expanded recomputed in
        # backward). fused=True is CUDA + optimized-kernel only, and the
        # transpose class has no ``fused`` argument, so it is ignored there.
        if fused and not torch.cuda.is_available():
            self.skipTest("fused=True is CUDA-only")

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

        # set up handles
        if transpose:
            # Transpose class has no ``fused`` argument; it is ignored here.
            conv_local = th.DiscreteContinuousConvTransposeS2(**disco_args).to(dtype=module_dtype, device=self.device)
            conv_dist = thd.DistributedDiscreteContinuousConvTransposeS2(**disco_args).to(dtype=module_dtype, device=self.device)
        else:
            conv_local = th.DiscreteContinuousConvS2(**disco_args).to(dtype=module_dtype, device=self.device)
            conv_dist = thd.DistributedDiscreteContinuousConvS2(
                **disco_args,
                fused=fused,
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

        # distributed conv.
        # FWD pass
        inp_local = self._split_helper(inp_full.detach().clone())
        inp_local.requires_grad = True
        with maybe_autocast(self.device.type, dtype):
            out_local = conv_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        # Print diagnostics from rank 0 only; assert the all-reduced verdict on every
        # rank so a failure on any rank fails the test consistently (see reduce_success).
        verbose = verbose and self.world_rank == 0

        # evaluate FWD pass
        out_gather_full = self._gather_helper_fwd(out_local, conv_dist)
        ok = compare_tensors("output", out_full, out_gather_full, atol=atol, rtol=rtol, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "output")

        # evaluate BWD pass
        igrad_gather_full = self._gather_helper_bwd(igrad_local, conv_dist)
        ok = compare_tensors("gradients", igrad_full, igrad_gather_full, atol=atol, rtol=rtol, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "gradients")

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
        if dtype == torch.bfloat16 and basis_type == "piecewise linear":
            # bf16 parameter gradients accumulate over batch and distributed
            # spatial shards. Keep output/dgrad tolerances tight, but allow
            # small absolute drift near zero-valued weight-gradient entries.
            pg_atol = max(pg_atol, 3.0)
        if conv_dist.weight.grad is not None:
            wgrad = self._allreduce_param_grad(conv_dist.weight.grad)
            self.assertTrue(compare_tensors("weight grad", conv_local.weight.grad, wgrad, atol=pg_atol, rtol=pg_rtol, verbose=verbose))
        if getattr(conv_dist, "bias", None) is not None and conv_dist.bias.grad is not None:
            bgrad = self._allreduce_param_grad(conv_dist.bias.grad)
            self.assertTrue(compare_tensors("bias grad", conv_local.bias.grad, bgrad, atol=pg_atol, rtol=pg_rtol, verbose=verbose))

    def _run_kpacked_fallback(self, fused: bool, dtype: torch.dtype, atol: float, rtol: float):
        """Shared body for kpacked-fallback tests.

        Monkeypatches psi_kpacked_K_pad to an ineligible value (24) so that
        the distributed conv falls back to the CSR path even with bf16/fp16
        input, and verifies fwd+bwd correctness against the serial reference
        (which also has kpacked disabled via the same monkeypatch).
        """
        if fused and not torch.cuda.is_available():
            self.skipTest("fused=True is CUDA-only")

        set_seed(555)
        nlat, nlon = 64, 128
        B, C = 8, 8

        args = dict(
            in_channels=C,
            out_channels=C,
            in_shape=(nlat, nlon),
            out_shape=(nlat, nlon),
            basis_type="piecewise linear",
            basis_norm_mode="mean",
            kernel_shape=(3,),
            groups=1,
            grid_in="equiangular",
            grid_out="equiangular",
            bias=True,
        )

        conv_local = th.DiscreteContinuousConvS2(**args).to(dtype=torch.float32, device=self.device)
        conv_dist = thd.DistributedDiscreteContinuousConvS2(**args, fused=fused).to(dtype=torch.float32, device=self.device)

        with torch.no_grad():
            conv_dist.weight.copy_(conv_local.weight)
            conv_dist.bias.copy_(conv_local.bias)

        # Force both to CSR fallback by making K_PAD ineligible.
        conv_local.psi_kpacked_K_pad = 24
        conv_dist.psi_kpacked_K_pad = 24

        inp_full = torch.randn((B, C, nlat, nlon), dtype=torch.float32, device=self.device)

        inp_full.requires_grad = True
        with maybe_autocast(self.device.type, dtype):
            out_full = conv_local(inp_full)
        ograd_full = torch.randn_like(out_full)
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        inp_local = self._split_helper(inp_full.detach().clone())
        inp_local.requires_grad = True
        with maybe_autocast(self.device.type, dtype):
            out_local = conv_dist(inp_local)
        ograd_local = self._split_helper(ograd_full)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        verbose = self.world_rank == 0
        out_gather = self._gather_helper_fwd(out_local, conv_dist)
        ok = compare_tensors("output", out_full, out_gather, atol=atol, rtol=rtol, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "output")

        igrad_gather = self._gather_helper_bwd(igrad_local, conv_dist)
        ok = compare_tensors("gradients", igrad_full, igrad_gather, atol=atol, rtol=rtol, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "gradients")

    def test_kpacked_fallback_bf16_unfused(self):
        """bf16 + kpacked disabled (K_PAD=24) → CSR path, fused=False."""
        self._run_kpacked_fallback(fused=False, dtype=torch.bfloat16, atol=5e-2, rtol=5e-2)

    def test_kpacked_fallback_bf16_fused(self):
        """bf16 + kpacked disabled (K_PAD=24) → CSR path, fused=True."""
        self._run_kpacked_fallback(fused=True, dtype=torch.bfloat16, atol=5e-2, rtol=5e-2)

    def test_kpacked_fallback_fp16_unfused(self):
        """fp16 + kpacked disabled (K_PAD=24) → CSR path, fused=False."""
        self._run_kpacked_fallback(fused=False, dtype=torch.float16, atol=2e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
