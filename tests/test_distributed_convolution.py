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
from torch_harmonics.distributed import compute_polar_halo_radius, compute_split_shapes
from torch_harmonics.quadrature import compute_theta_cutoff, effective_theta_cutoff, precompute_latitudes

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
            # non-equiangular grids at regular resolution. These are the rows where the
            # default theta_cutoff is derived from a node distribution that is not uniform
            # in theta, so the polar halo differs from the pi/(nlat-1) assumption. The only
            # other non-equiangular case in this file is ERA5-sized and slow-gated, which
            # left the distributed halo/split bookkeeping untested on these grids.
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "lobatto", "lobatto", torch.float32, False, False, 1e-6, 1e-5],
            [64, 128, 32, 64, 32, 8, (3), "piecewise linear", "mean", 1, "lobatto", "lobatto", torch.float32, False, False, 1e-6, 1e-5],
            [64, 128, 32, 64, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "legendre-gauss", torch.float32, False, False, 1e-6, 1e-5],
            [64, 128, 128, 256, 32, 8, (3), "piecewise linear", "mean", 1, "lobatto", "lobatto", torch.float32, True, False, 1e-6, 1e-5],
            # equiangular-trapezoidal is equispaced in cos(theta), so its default cutoff is
            # ~5x wider than the other grids at the same nlat and psi is correspondingly
            # denser. batch_size/num_chan dialed down to keep the working set bounded.
            [64, 128, 64, 128, 2, 8, (3), "piecewise linear", "mean", 1, "equiangular-trapezoidal", "equiangular-trapezoidal", torch.float32, False, False, 1e-6, 1e-5],
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

    @parameterized.expand(
        [
            # nlat_in, nlon_in, nlat_out, nlon_out, kernel_shape, grid_in, grid_out, transpose, polar_mode, theta_cutoff_scale
            # even resolutions, where every rank gets the same number of latitudes
            [32, 64, 32, 64, (3, 3), "equiangular", "equiangular", False, "halo-exchange", 1.0],
            [32, 64, 32, 64, (3, 3), "legendre-gauss", "legendre-gauss", False, "halo-exchange", 1.0],
            [32, 64, 16, 32, (3, 3), "equiangular", "equiangular", False, "halo-exchange", 1.0],
            [32, 64, 32, 64, (3, 3), "equiangular", "equiangular", True, "halo-exchange", 1.0],
            [16, 32, 32, 64, (3, 3), "equiangular", "equiangular", True, "halo-exchange", 1.0],
            # odd nlat, so compute_split_shapes hands ranks different counts and the two axes
            # are skewed against each other -- e.g. 33 and 32 over 4 ranks split [9,8,8,8] and
            # [8,8,8,8], which puts every rank's input and output bands at a different offset
            [33, 64, 33, 64, (3, 3), "equiangular", "equiangular", False, "halo-exchange", 1.0],
            [33, 64, 33, 64, (3, 3), "legendre-gauss", "legendre-gauss", False, "halo-exchange", 1.0],
            [33, 64, 32, 64, (3, 3), "equiangular", "equiangular", False, "halo-exchange", 1.0],
            [32, 64, 33, 64, (3, 3), "equiangular", "equiangular", False, "halo-exchange", 1.0],
            [33, 64, 33, 64, (3, 3), "equiangular", "equiangular", True, "halo-exchange", 1.0],
            [17, 64, 33, 64, (3, 3), "equiangular", "equiangular", True, "halo-exchange", 1.0],
            # coarse, odd resolution ratios: the support then spans more than one input ring per
            # output ring, so these are the rows where the latitude band is genuinely wider than
            # the nearest neighbour and an off-by-one in it would not cancel out
            [33, 64, 17, 64, (3, 3), "equiangular", "equiangular", False, "halo-exchange", 1.0],
            [33, 64, 11, 64, (3, 3), "equiangular", "equiangular", False, "halo-exchange", 1.0],
            [33, 64, 17, 64, (3, 3), "legendre-gauss", "legendre-gauss", False, "halo-exchange", 1.0],
            # the reduce-scatter fallback keys psi the other way round; it stays reachable for
            # cutoffs a halo cannot serve, so it needs to stay covered
            [32, 64, 32, 64, (3, 3), "equiangular", "equiangular", False, "reduce-scatter", 1.0],
            [33, 64, 17, 64, (3, 3), "equiangular", "equiangular", False, "reduce-scatter", 1.0],
            # a cutoff far wider than the grid spacing: the support reaches past a neighbour once
            # the polar group is fine enough, so halo-exchange must refuse and reduce-scatter must
            # still work. Whether it actually refuses depends on the grid size at runtime, so the
            # test derives the expectation from the geometry rather than asserting it here.
            [32, 64, 32, 64, (3, 3), "equiangular", "equiangular", False, "halo-exchange", 12.0],
            [32, 64, 32, 64, (3, 3), "equiangular", "equiangular", False, "reduce-scatter", 12.0],
        ],
        skip_on_empty=True,
    )
    def test_psi_blocks(self, nlat_in, nlon_in, nlat_out, nlon_out, kernel_shape, grid_in, grid_out, transpose, polar_mode, theta_cutoff_scale, verbose=False):
        """Each rank's local psi equals the serial psi restricted to its input latitudes.

        The sparsity pattern is built from the latitude band that can fall inside the angular
        cutoff, and the distributed module keeps only the entries whose input latitude it owns.
        This checks the two agree entry for entry, which isolates the pattern from the
        convolution: an off-by-one in the band or in the index remapping shows up here rather
        than as a diffuse accuracy failure in the forward.

        Each rank compares against its own slice of the serial construction, so no collective
        is involved and a failure identifies the rank. The comparison is order-insensitive --
        the local build re-keys and re-sorts into CSR -- so entries are sorted by
        (kernel, output latitude, global input index) on both sides first.
        """

        set_seed(333)

        # the transpose convolution still reduces via all-gather and has no polar_mode, so only
        # the halo rows are meaningful for it
        if transpose and polar_mode != "halo-exchange":
            self.skipTest("the transpose convolution has no polar_mode")

        theta_cutoff = theta_cutoff_scale * compute_theta_cutoff(nlat_out if not transpose else nlat_in, grid=grid_out if not transpose else grid_in)

        # Whether a halo can serve this configuration is a property of the geometry and the
        # decomposition, so derive it rather than hard-coding it per row: the same cutoff is
        # servable on a coarse polar split and not on a fine one. The assertion is then that the
        # constructor agrees -- refusing exactly when the support outruns a neighbour, and saying
        # which mode to use instead.
        lats_in, _ = precompute_latitudes(nlat_in, grid=grid_in)
        lats_out, _ = precompute_latitudes(nlat_out, grid=grid_out)
        try:
            compute_polar_halo_radius(
                lats_in,
                lats_out,
                effective_theta_cutoff(theta_cutoff),
                compute_split_shapes(nlat_in, self.grid_size_h),
                compute_split_shapes(nlat_out, self.grid_size_h),
            )
            halo_servable = True
        except ValueError:
            halo_servable = False

        if not transpose and polar_mode == "halo-exchange" and not halo_servable:
            with self.assertRaises(ValueError) as ctx:
                thd.DistributedDiscreteContinuousConvS2(
                    1,
                    1,
                    (nlat_in, nlon_in),
                    (nlat_out, nlon_out),
                    kernel_shape=kernel_shape,
                    grid_in=grid_in,
                    grid_out=grid_out,
                    theta_cutoff=theta_cutoff,
                    polar_mode=polar_mode,
                )
            self.assertIn("reduce-scatter", str(ctx.exception), "the refusal must name the mode that does work")
            return

        if transpose:
            conv_local = th.DiscreteContinuousConvTransposeS2(
                1, 1, (nlat_in, nlon_in), (nlat_out, nlon_out), kernel_shape=kernel_shape, grid_in=grid_in, grid_out=grid_out, theta_cutoff=theta_cutoff
            ).to(self.device)
            conv_dist = thd.DistributedDiscreteContinuousConvTransposeS2(
                1, 1, (nlat_in, nlon_in), (nlat_out, nlon_out), kernel_shape=kernel_shape, grid_in=grid_in, grid_out=grid_out, theta_cutoff=theta_cutoff
            ).to(self.device)
            # the transpose module's psi indexes the output grid along the split axis
            nlon_split = nlon_out
            shapes = conv_dist.lat_out_shapes
        else:
            conv_local = th.DiscreteContinuousConvS2(
                1, 1, (nlat_in, nlon_in), (nlat_out, nlon_out), kernel_shape=kernel_shape, grid_in=grid_in, grid_out=grid_out, theta_cutoff=theta_cutoff
            ).to(self.device)
            conv_dist = thd.DistributedDiscreteContinuousConvS2(
                1,
                1,
                (nlat_in, nlon_in),
                (nlat_out, nlon_out),
                kernel_shape=kernel_shape,
                grid_in=grid_in,
                grid_out=grid_out,
                theta_cutoff=theta_cutoff,
                polar_mode=polar_mode,
            ).to(self.device)
            nlon_split = nlon_in
            shapes = conv_dist.lat_in_shapes

        def sorted_entries(ker, row, col, vals):
            """Canonical ordering so the two builds are comparable regardless of CSR layout."""
            key = (ker.to(torch.int64) * (nlat_out + nlat_in) + row.to(torch.int64)) * (nlat_in * nlat_out * nlon_split) + col.to(torch.int64)
            order = torch.argsort(key)
            return ker[order], row[order], col[order], vals[order]

        # The two polar strategies key psi differently, so both the un-keying and the predicate
        # for "which serial entries should this rank hold" differ. Lift the local tensor back to
        # global coordinates and select the matching serial entries, then compare entry for entry.
        use_halo = getattr(conv_dist, "use_halo", False)

        # Assert the mode rather than only reading it back: everything below adapts to whichever
        # keying the constructor chose, so a silent fall back to reduce-scatter would satisfy the
        # comparison while leaving the halo path untested. The transpose class has no polar_mode.
        if not transpose:
            self.assertEqual(use_halo, polar_mode == "halo-exchange", f"constructor did not honour polar_mode={polar_mode!r}")

        if use_halo:
            # rows are this rank's own output latitudes, columns index a halo-padded input band
            r_lat = conv_dist.r_lat
            out_start = sum(conv_dist.lat_out_shapes[: self.hrank])
            halo_start = sum(conv_dist.lat_in_shapes[: self.hrank]) - r_lat

            lat_loc = conv_dist.psi_col_idx // nlon_split
            lon_loc = conv_dist.psi_col_idx % nlon_split
            col_global = (lat_loc + halo_start) * nlon_split + lon_loc
            row_global = conv_dist.psi_row_idx + out_start

            keep = (conv_local.psi_row_idx >= out_start) & (conv_local.psi_row_idx < out_start + conv_dist.nlat_out_local)
        else:
            # rows stay global, columns index the local input slice
            lat_start = sum(shapes[: self.hrank])
            lat_local = shapes[self.hrank]

            lat_loc = conv_dist.psi_col_idx // nlon_split
            lon_loc = conv_dist.psi_col_idx % nlon_split
            col_global = (lat_loc + lat_start) * nlon_split + lon_loc
            row_global = conv_dist.psi_row_idx

            lat_ser = conv_local.psi_col_idx // nlon_split
            keep = (lat_ser >= lat_start) & (lat_ser < lat_start + lat_local)

        got = sorted_entries(conv_dist.psi_ker_idx, row_global, col_global, conv_dist.psi_vals)
        ref = sorted_entries(conv_local.psi_ker_idx[keep], conv_local.psi_row_idx[keep], conv_local.psi_col_idx[keep], conv_local.psi_vals[keep])

        if verbose:
            print(f"psi block on rank ({self.hrank},{self.wrank}), use_halo={use_halo}: {got[0].numel()} vs {ref[0].numel()} entries")

        self.assertEqual(got[0].numel(), ref[0].numel(), "number of local psi entries")
        names = ("kernel index", "row index", "column index", "values")
        for name, g, r in zip(names, got, ref):
            ok = compare_tensors(f"psi {name}", g, r, atol=1e-14, rtol=1e-14, verbose=verbose)
            self.assertTrue(reduce_success(ok, self.device), f"psi {name}")

        # Completeness. The per-rank check above compares against a predicate, so a predicate
        # wrong in the same way as the implementation would pass it; summing the local entry
        # counts over the polar group and comparing to the serial total catches a partition that
        # drops or duplicates entries, which is the failure that would quietly change results.
        local_nnz = torch.tensor([conv_dist.psi_vals.numel()], device=self.device, dtype=torch.int64)
        if self.grid_size_h > 1:
            dist.all_reduce(local_nnz, group=self.h_group)
        self.assertEqual(int(local_nnz.item()), int(conv_local.psi_vals.numel()), "polar ranks together must hold every serial psi entry exactly once")

    def test_polar_mode_rejects_unknown_value(self):
        """An unrecognised mode is a typo, not a request for a default."""
        with self.assertRaises(ValueError) as ctx:
            thd.DistributedDiscreteContinuousConvS2(1, 1, (32, 64), (32, 64), kernel_shape=(3, 3), polar_mode="halo")
        self.assertIn("halo-exchange", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
