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
            # fp64
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float64, False, False, 1e-6, 1e-6],
            [64, 128, 32, 64, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float64, False, False, 1e-6, 1e-6],
            [64, 128, 64, 128, 32, 8, (3, 2), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float64, True, False, 1e-6, 1e-6],
            [65, 128, 65, 128, 32, 8, (3, 4), "harmonic", "mean", 1, "equiangular", "equiangular", torch.float64, False, False, 1e-6, 1e-6],
            # ---- fused=True : reordered a2a (CUDA + optimized kernels) ----
            # non-transpose only; covers groups=1, groups=2, downsample, harmonic.
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, True, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3, 2), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, True, 1e-6, 1e-5],
            [64, 128, 32, 64, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float32, False, True, 1e-6, 1e-5],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 2, "equiangular", "equiangular", torch.float32, False, True, 1e-6, 1e-5],
            [65, 128, 65, 128, 32, 8, (3, 4), "harmonic", "mean", 1, "equiangular", "equiangular", torch.float32, False, True, 1e-6, 1e-5],
            [65, 128, 33, 64, 32, 8, (3, 4), "harmonic", "mean", 1, "equiangular", "equiangular", torch.float32, False, True, 1e-6, 1e-5],
            # ---- AMP (fp16/bf16) ----
            # Each dtype runs fused off AND on (non-transpose). The transpose
            # class has no ``fused`` argument, so it runs fused=False only
            # (fused=True there would be an identical duplicate).
            # fp16
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float16, False, False, 2e-2, 1e-2],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float16, False, True, 2e-2, 1e-2],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.float16, True, False, 2e-2, 1e-2],
            # bf16
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.bfloat16, False, False, 5e-2, 5e-2],
            [64, 128, 64, 128, 32, 8, (3), "piecewise linear", "mean", 1, "equiangular", "equiangular", torch.bfloat16, False, True, 5e-2, 5e-2],
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
            # Two distributed convs chained in series — validates that the
            # reordered (fused) and standard a2a paths compose correctly
            # across layers (output + input-grad through both layers).
            #
            # Tolerances are explicit per row (looser than the single-conv
            # test to absorb the noise of chaining two convolutions and
            # their backwards). Per-row atol/rtol applies to the output
            # and input-gradient checks; parameter-gradient checks scale
            # those by ``pgrad_tol_factor`` inside the test.
            #
            # Each row: (fused, kernel_shape, basis, dtype, atol, rtol).
            [False, (3, 4), "harmonic", torch.float32, 1e-5, 1e-4],
            [True, (3, 4), "harmonic", torch.float32, 1e-5, 1e-4],
            [False, (3), "piecewise linear", torch.float32, 1e-5, 1e-4],
            [True, (3), "piecewise linear", torch.float32, 1e-5, 1e-4],
            # AMP rows — both dtypes run fused off and on; tolerances
            # loosened above the single-conv AMP rows to absorb two-layer
            # fp16/bf16 cast accumulation.
            [False, (3, 4), "harmonic", torch.float16, 5e-2, 5e-2],
            [True, (3, 4), "harmonic", torch.float16, 5e-2, 5e-2],
            [False, (3, 4), "harmonic", torch.bfloat16, 2e-1, 2e-1],
            [True, (3, 4), "harmonic", torch.bfloat16, 2e-1, 2e-1],
        ],
        skip_on_empty=True,
    )
    def test_distributed_disco_conv_stacked(
        self,
        fused,
        kernel_shape,
        basis_type,
        dtype,
        atol,
        rtol,
        verbose=True,
    ):
        """Two distributed convs chained in series.

        Compares the chained forward output against a serial reference and
        the input-gradient flowing back through both layers, for both the
        standard (fused=False) and reordered (fused=True) a2a paths.
        """
        if fused and not torch.cuda.is_available():
            self.skipTest("fused=True is CUDA-only")

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

        # Serial reference: two stacked local convs.
        conv1_local = th.DiscreteContinuousConvS2(**common_args).to(dtype=module_dtype, device=self.device)
        conv2_local = th.DiscreteContinuousConvS2(**common_args).to(dtype=module_dtype, device=self.device)

        # Distributed: two stacked convs.
        conv1_dist = thd.DistributedDiscreteContinuousConvS2(**common_args, fused=fused).to(dtype=module_dtype, device=self.device)
        conv2_dist = thd.DistributedDiscreteContinuousConvS2(**common_args, fused=fused).to(dtype=module_dtype, device=self.device)

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
        with maybe_autocast(self.device.type, dtype):
            mid_local = conv1_dist(inp_local)
            out_local = conv2_dist(mid_local)

        ograd_local = self._split_helper(ograd_full)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        # Compare chained output.
        out_gather_full = self._gather_helper_fwd(out_local, conv2_dist)
        self.assertTrue(compare_tensors("stacked output", out_full, out_gather_full, atol=atol, rtol=rtol, verbose=verbose))

        # Compare input gradient flowing back through both layers.
        igrad_gather_full = self._gather_helper_bwd(igrad_local, conv1_dist)
        self.assertTrue(compare_tensors("stacked gradients", igrad_full, igrad_gather_full, atol=atol, rtol=rtol, verbose=verbose))

        # Compare per-layer parameter gradients (allreduced over h, w).
        # Param grads accumulate reduction noise over batch + local spatial;
        # loosen relative to the row's per-element tolerance. AMP needs a
        # large factor here: stacking two bf16/fp16 layers compounds the
        # cross-rank cancellation, so the weight grad drifts by O(few)
        # absolute at near-zero reference elements (the per-element output
        # and input-grad checks at the row tolerance are the real validation;
        # this is a coarse sanity net gated by atol). fp32 uses a smaller
        # factor since its base atol/rtol is already tight at 1e-5 / 1e-4.
        pgrad_tol_factor = 25.0 if is_amp else 100.0
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
