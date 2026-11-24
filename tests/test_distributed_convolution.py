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
from itertools import accumulate

import torch
import torch.distributed as dist
from parameterized import parameterized
from testutils import (
    ShardedFieldMixin,
    amp_ulps,
    compare_tensors,
    make_grid,
    maybe_autocast,
    param_grad_ulps,
    reduce_success,
    set_seed,
    setup_class_from_context,
    setup_module,
    teardown_module,
)

import torch_harmonics as th
import torch_harmonics.distributed as thd
from torch_harmonics.disco.backends import RegularFftBackend, RegularKpackedBackend, RegularOptimizedBackend, RegularReferenceBackend
from torch_harmonics.disco.optimized.disco_plan_fft import fft_scatter_available
from torch_harmonics.distributed import compute_split_shapes
from torch_harmonics.quadrature import compute_theta_cutoff

# Opt-in gate for slow / large-grid parameterized cases (e.g. 721x1440 ERA5-like
# shapes), as in tests/test_distributed_attention.py.
_run_slow_tests = os.getenv("TORCH_HARMONICS_RUN_SLOW_TESTS", "0") == "1"

# (*in_shape, *out_shape) of the parameterized cases that are
# gated behind TORCH_HARMONICS_RUN_SLOW_TESTS=1.
_SLOW_DISCO_SHAPES = frozenset(
    {
        (721, 1440, 721, 1440),
        (721, 1440, 360, 720),
    }
)

# shared state
_DIST_CTX = {}

_DOUBLE_BACKWARD_SHAPES = [
    # grid_in, in_shape, grid_out, out_shape, kernel_shape, basis_type, transpose, fused, polar_mode
    ["equiangular", (33, 64), "equiangular", (33, 64), (3), "piecewise linear", False, False, "halo-exchange"],
    ["equiangular", (33, 64), "equiangular", (33, 64), (3), "piecewise linear", False, True, "halo-exchange"],
    ["equiangular", (33, 64), "equiangular", (17, 32), (3, 4), "harmonic", False, False, "halo-exchange"],
    ["equiangular", (33, 64), "equiangular", (17, 32), (3, 4), "harmonic", False, True, "halo-exchange"],
    ["equiangular", (33, 64), "equiangular", (33, 64), (3), "piecewise linear", False, False, "reduce-scatter"],
    ["equiangular", (33, 64), "equiangular", (33, 64), (3), "piecewise linear", False, True, "reduce-scatter"],
    ["equiangular", (17, 32), "equiangular", (33, 64), (3), "piecewise linear", True, False, "halo-exchange"],
    ["equiangular", (33, 64), "equiangular", (33, 64), (3, 4), "harmonic", True, False, "halo-exchange"],
    ["equiangular", (33, 64), "equiangular", (33, 64), (3), "piecewise linear", True, False, "reduce-scatter"],
    ["healpix", (8,), "healpix", (8,), (3), "piecewise linear", False, False, "halo-exchange"],
    ["healpix", (8,), "healpix", (4,), (3, 3), "morlet", False, True, "halo-exchange"],
    ["healpix", (8,), "healpix", (8,), (3), "piecewise linear", False, False, "reduce-scatter"],
    ["healpix", (4,), "healpix", (8,), (3), "piecewise linear", True, False, "halo-exchange"],
    ["healpix", (4,), "healpix", (8,), (3), "piecewise linear", True, False, "reduce-scatter"],
]

# float64, and fp16/bf16 under autocast of an fp32 layer
_DOUBLE_BACKWARD_CASES = [shape + [dtype] for shape in _DOUBLE_BACKWARD_SHAPES for dtype in (torch.float64, torch.float16, torch.bfloat16)]


def _ulps(dtype):
    """
    ULP budget, of max|ref| in ``dtype``, for the output and input gradient. fp32/fp64: the
    halo or reduce-scatter only reassociates the per-point contraction (measured: up to 3.9
    ULPs on CPU and GB200). fp16/bf16: see amp_ulps. Parameter gradients use param_grad_ulps.
    """
    return amp_ulps(dtype) or 16.0


def setUpModule():
    setup_module(_DIST_CTX)


def tearDownModule():
    teardown_module(_DIST_CTX)


class TestDistributedDiscreteContinuousConvolution(ShardedFieldMixin, unittest.TestCase):
    """Test the distributed discrete-continuous convolution module."""

    @classmethod
    def setUpClass(cls):
        setup_class_from_context(cls, _DIST_CTX)

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

    def _allreduce_param_grad(self, tensor):
        """Sum a per-rank parameter gradient across the polar (h) and
        azimuth (w) groups to recover the global gradient.

        Each rank's parameter gradient holds the local-spatial contribution
        to the global gradient (see e.g. the einsum in the bwd grad_w
        path: it reduces over local b/H_out/W_out only). To compare against
        a serial reference we have to sum across the two split dimensions
        — independently, so this works for any (h, w) grid layout and
        doesn't implicitly assume the world group equals h × w.

        Returns a contiguous clone: allreduce is in-place, so the caller's
        ``conv_dist.weight.grad`` and friends must not be mutated, and NCCL
        rejects non-contiguous tensors. A gradient from ``torch.autograd.grad``
        keeps the layout of the backward that produced it (the transpose
        convolution's weight gradient is a permuted view); only ``.grad`` is
        laid out like the parameter.
        """
        out = tensor.contiguous().clone()
        if self.grid_size_h > 1:
            dist.all_reduce(out, group=self.h_group)
        if self.grid_size_w > 1:
            dist.all_reduce(out, group=self.w_group)
        return out

    @parameterized.expand(
        [
            # ---- fused=False : standard a2a (K-expanded saved for backward) ----
            # fp32
            ["equiangular", (64, 128), "equiangular", (64, 128), 32, 8, (3), "piecewise linear", "mean", 1, torch.float32, False, False, "halo-exchange"],
            ["equiangular", (64, 128), "equiangular", (64, 128), 32, 8, (3, 2), "piecewise linear", "mean", 1, torch.float32, False, False, "halo-exchange"],
            ["equiangular", (64, 128), "equiangular", (32, 64), 32, 8, (3), "piecewise linear", "mean", 1, torch.float32, False, False, "halo-exchange"],
            ["equiangular", (64, 128), "equiangular", (64, 128), 32, 8, (3), "piecewise linear", "mean", 2, torch.float32, False, False, "halo-exchange"],
            # a channel count that is not a multiple of the azimuth split, so the a2a has to
            # cope with unequal per-rank shares. 10 rather than a smaller odd number because
            # the split has to stay non-empty on every rank: 6 over 8 azimuth ranks leaves the
            # last two with nothing and the split itself raises.
            ["equiangular", (64, 128), "equiangular", (64, 128), 32, 10, (3), "piecewise linear", "mean", 1, torch.float32, False, False, "halo-exchange"],
            ["equiangular", (64, 128), "equiangular", (64, 128), 32, 8, (3), "piecewise linear", "mean", 1, torch.float32, True, False, "halo-exchange"],
            ["equiangular", (65, 128), "equiangular", (65, 128), 32, 8, (3), "piecewise linear", "mean", 1, torch.float32, True, False, "halo-exchange"],
            ["equiangular", (64, 128), "equiangular", (64, 128), 32, 8, (3, 2), "piecewise linear", "mean", 1, torch.float32, True, False, "halo-exchange"],
            ["equiangular", (64, 128), "equiangular", (128, 256), 32, 8, (3), "piecewise linear", "mean", 1, torch.float32, True, False, "halo-exchange"],
            ["equiangular", (65, 128), "equiangular", (65, 128), 32, 8, (3, 4), "harmonic", "mean", 1, torch.float32, False, False, "halo-exchange"],
            ["equiangular", (65, 128), "equiangular", (65, 128), 32, 8, (3, 4), "harmonic", "mean", 1, torch.float32, True, False, "halo-exchange"],
            ["equiangular", (65, 128), "equiangular", (33, 64), 32, 8, (3, 4), "harmonic", "mean", 1, torch.float32, False, False, "halo-exchange"],
            # group coverage: depthwise (groups == n_channels) and a groupsize>1
            # split (C=12, groups=3 -> groupsize=4).
            ["equiangular", (64, 128), "equiangular", (64, 128), 32, 8, (3), "piecewise linear", "mean", 8, torch.float32, False, False, "halo-exchange"],
            ["equiangular", (64, 128), "equiangular", (64, 128), 32, 12, (3), "piecewise linear", "mean", 3, torch.float32, False, False, "halo-exchange"],
            # fp64
            ["equiangular", (64, 128), "equiangular", (64, 128), 32, 8, (3), "piecewise linear", "mean", 1, torch.float64, False, False, "halo-exchange"],
            ["equiangular", (64, 128), "equiangular", (32, 64), 32, 8, (3), "piecewise linear", "mean", 1, torch.float64, False, False, "halo-exchange"],
            ["equiangular", (64, 128), "equiangular", (64, 128), 32, 8, (3, 2), "piecewise linear", "mean", 1, torch.float64, True, False, "halo-exchange"],
            ["equiangular", (65, 128), "equiangular", (65, 128), 32, 8, (3, 4), "harmonic", "mean", 1, torch.float64, False, False, "halo-exchange"],
            # non-equiangular grids at regular resolution. These are the rows where the
            # default theta_cutoff is derived from a node distribution that is not uniform
            # in theta, so the polar halo differs from the pi/(nlat-1) assumption. The only
            # other non-equiangular case in this file is ERA5-sized and slow-gated, which
            # left the distributed halo/split bookkeeping untested on these grids.
            ["lobatto", (64, 128), "lobatto", (64, 128), 32, 8, (3), "piecewise linear", "mean", 1, torch.float32, False, False, "halo-exchange"],
            ["lobatto", (64, 128), "lobatto", (32, 64), 32, 8, (3), "piecewise linear", "mean", 1, torch.float32, False, False, "halo-exchange"],
            ["equiangular", (64, 128), "legendre-gauss", (32, 64), 32, 8, (3), "piecewise linear", "mean", 1, torch.float32, False, False, "halo-exchange"],
            ["lobatto", (64, 128), "lobatto", (128, 256), 32, 8, (3), "piecewise linear", "mean", 1, torch.float32, True, False, "halo-exchange"],
            # trapezoidal is equispaced in cos(theta), so its default cutoff is
            # ~5x wider than the other grids at the same nlat and psi is correspondingly
            # denser. batch_size/num_chan dialed down to keep the working set bounded.
            ["trapezoidal", (64, 128), "trapezoidal", (64, 128), 2, 8, (3), "piecewise linear", "mean", 1, torch.float32, False, False, "halo-exchange"],
            # ERA5-like grids, gated behind TORCH_HARMONICS_RUN_SLOW_TESTS=1.
            # batch_size and num_chan dialed down (2, 8) vs the rest of the suite (32, 8)
            # to keep the working set under a few GB at these resolutions.
            ["equiangular", (721, 1440), "equiangular", (721, 1440), 2, 8, (3), "piecewise linear", "mean", 1, torch.float32, False, False, "halo-exchange"],
            ["equiangular", (721, 1440), "legendre-gauss", (360, 720), 2, 8, (3), "piecewise linear", "mean", 1, torch.float32, False, False, "halo-exchange"],
            # ---- fused=True : reordered a2a (CUDA + optimized kernels) ----
            # non-transpose only; downsample + harmonic.
            ["equiangular", (64, 128), "equiangular", (64, 128), 32, 8, (3), "piecewise linear", "mean", 1, torch.float32, False, True, "halo-exchange"],
            ["equiangular", (64, 128), "equiangular", (64, 128), 32, 8, (3, 2), "piecewise linear", "mean", 1, torch.float32, False, True, "halo-exchange"],
            ["equiangular", (64, 128), "equiangular", (32, 64), 32, 8, (3), "piecewise linear", "mean", 1, torch.float32, False, True, "halo-exchange"],
            ["equiangular", (65, 128), "equiangular", (65, 128), 32, 8, (3, 4), "harmonic", "mean", 1, torch.float32, False, True, "halo-exchange"],
            ["equiangular", (65, 128), "equiangular", (33, 64), 32, 8, (3, 4), "harmonic", "mean", 1, torch.float32, False, True, "halo-exchange"],
            # group coverage for the padded grouped path:
            #  groups=1            -> within-group channel split (no padding)
            #  groups=2 (gs=4)     -> split cuts a group at az>=4 (padding)
            #  groups=3,C=12 (gs=4)-> split cuts a group at az=2 and az=4 (padding)
            #  groups=C (gs=1)     -> depthwise; every channel a group (no-pad fast path)
            ["equiangular", (64, 128), "equiangular", (64, 128), 32, 8, (3), "piecewise linear", "mean", 2, torch.float32, False, True, "halo-exchange"],
            ["equiangular", (64, 128), "equiangular", (64, 128), 32, 12, (3), "piecewise linear", "mean", 3, torch.float32, False, True, "halo-exchange"],
            ["equiangular", (64, 128), "equiangular", (64, 128), 32, 8, (3), "piecewise linear", "mean", 8, torch.float32, False, True, "halo-exchange"],
            # ---- AMP (fp16/bf16) ----
            # Each dtype runs fused off AND on (non-transpose). The transpose
            # class has no ``fused`` argument, so it runs fused=False only
            # (fused=True there would be an identical duplicate).
            # fp16
            ["equiangular", (64, 128), "equiangular", (64, 128), 32, 8, (3), "piecewise linear", "mean", 1, torch.float16, False, False, "halo-exchange"],
            # Harmonic AMP rows exercise distributed kpacked tensor-core paths:
            # (2,2) -> K=4 -> K_PAD=8; (3,3) -> K=9 -> K_PAD=16.
            ["equiangular", (64, 128), "equiangular", (64, 128), 8, 8, (2, 2), "harmonic", "mean", 1, torch.float16, False, False, "halo-exchange"],
            ["equiangular", (64, 128), "equiangular", (64, 128), 8, 8, (3, 3), "harmonic", "mean", 1, torch.float16, False, False, "halo-exchange"],
            ["equiangular", (64, 128), "equiangular", (64, 128), 32, 8, (3), "piecewise linear", "mean", 1, torch.float16, False, True, "halo-exchange"],
            ["equiangular", (64, 128), "equiangular", (64, 128), 8, 8, (2, 2), "harmonic", "mean", 1, torch.float16, False, True, "halo-exchange"],
            ["equiangular", (64, 128), "equiangular", (64, 128), 32, 8, (3), "piecewise linear", "mean", 1, torch.float16, True, False, "halo-exchange"],
            # bf16
            ["equiangular", (64, 128), "equiangular", (64, 128), 32, 8, (3), "piecewise linear", "mean", 1, torch.bfloat16, False, False, "halo-exchange"],
            ["equiangular", (64, 128), "equiangular", (64, 128), 8, 8, (2, 2), "harmonic", "mean", 1, torch.bfloat16, False, False, "halo-exchange"],
            ["equiangular", (64, 128), "equiangular", (64, 128), 8, 8, (3, 3), "harmonic", "mean", 1, torch.bfloat16, False, False, "halo-exchange"],
            ["equiangular", (64, 128), "equiangular", (64, 128), 32, 8, (3), "piecewise linear", "mean", 1, torch.bfloat16, False, True, "halo-exchange"],
            ["equiangular", (64, 128), "equiangular", (64, 128), 8, 8, (2, 2), "harmonic", "mean", 1, torch.bfloat16, False, True, "halo-exchange"],
            ["equiangular", (64, 128), "equiangular", (64, 128), 32, 8, (3), "piecewise linear", "mean", 1, torch.bfloat16, True, False, "halo-exchange"],
            # The reduce-scatter fallback needs forward/backward coverage of its own: it is the
            # mode a wide cutoff forces, and before the halo became the default these rows were
            # the only thing exercising it end to end.
            ["equiangular", (64, 128), "equiangular", (64, 128), 32, 8, (3), "piecewise linear", "mean", 1, torch.float32, False, False, "reduce-scatter"],
            ["equiangular", (64, 128), "equiangular", (32, 64), 32, 8, (3, 3), "piecewise linear", "mean", 1, torch.float32, False, False, "reduce-scatter"],
            ["equiangular", (64, 128), "equiangular", (64, 128), 32, 8, (3), "piecewise linear", "mean", 1, torch.float32, False, True, "reduce-scatter"],
            ["equiangular", (64, 128), "equiangular", (64, 128), 32, 8, (3), "piecewise linear", "mean", 1, torch.float32, True, False, "reduce-scatter"],
            ["equiangular", (32, 64), "equiangular", (64, 128), 32, 8, (3, 3), "piecewise linear", "mean", 2, torch.float32, True, False, "reduce-scatter"],
            ["equiangular", (64, 128), "equiangular", (64, 128), 32, 8, (3), "piecewise linear", "mean", 1, torch.bfloat16, True, False, "reduce-scatter"],
            # ---- optimized_kernel=False : the pure-PyTorch fallback ----
            # Every row above runs with the extension built, so the fallback is untested in CI
            # even though the base class selects it automatically on any install without the
            # extension. It is also the only consumer that builds a torch.sparse_coo_tensor and
            # therefore has to declare psi's column extent, which the halo mode re-keys onto a
            # padded input band -- a mismatch the arc and kpacked kernels cannot see because they
            # just index into whatever x they are handed. One row per polar mode: halo-exchange
            # for the halo-padded extent, reduce-scatter to pin that no halo still yields the
            # unpadded one.
            #
            # Resolution and batch_size/num_chan are dialed down relative to the rest of the
            # suite: the fallback contracts a torch.sparse_coo_tensor instead of the arc kernel
            # and is orders of magnitude slower. What these rows check is index bookkeeping,
            # which is resolution-independent, so the smallest grid that still splits across
            # polar ranks is enough.
            ["equiangular", (32, 64), "equiangular", (16, 32), 2, 8, (3), "piecewise linear", "mean", 1, torch.float32, False, False, "halo-exchange", False],
            ["equiangular", (32, 64), "equiangular", (32, 64), 2, 8, (3), "piecewise linear", "mean", 1, torch.float32, False, False, "reduce-scatter", False],
            # fused=True on the reference: the reordered all-to-all is kept, and the intermediate
            # is saved rather than recomputed, which only the compiled kernels can do
            ["equiangular", (32, 64), "equiangular", (16, 32), 2, 8, (3), "piecewise linear", "mean", 1, torch.float32, False, True, "halo-exchange", False],
            # The transpose convolution's fallback is a separate code path (it contracts a
            # semi-transposed COO psi over the output grid, whose rows the halo cuts to the
            # input band), and the rows above never reach it. Upsampling and same-shape, and
            # one row per polar mode.
            ["equiangular", (16, 32), "equiangular", (32, 64), 2, 8, (3), "piecewise linear", "mean", 1, torch.float32, True, False, "halo-exchange", False],
            ["equiangular", (32, 64), "equiangular", (32, 64), 2, 8, (3), "piecewise linear", "mean", 1, torch.float32, True, False, "halo-exchange", False],
            ["equiangular", (16, 32), "equiangular", (32, 64), 2, 8, (3), "piecewise linear", "mean", 1, torch.float32, True, False, "reduce-scatter", False],
            # ---- HEALPix: the serial ragged backends on each polar rank's band of rings ----
            # Polar ranks only; the azimuth configurations skip these. The forward borrows a
            # halo of whole input rings, or reduce-scatters; the transpose borrows the input
            # rings that scatter onto its own output points.
            ["healpix", (8,), "healpix", (8,), 4, 8, (3), "piecewise linear", "mean", 1, torch.float32, False, False, "halo-exchange"],
            ["healpix", (8,), "healpix", (8,), 4, 8, (3), "piecewise linear", "mean", 1, torch.float32, False, True, "halo-exchange"],
            ["healpix", (8,), "healpix", (4,), 4, 8, (3, 3), "morlet", "mean", 1, torch.float32, False, False, "halo-exchange"],
            ["healpix", (8,), "healpix", (8,), 4, 12, (3), "piecewise linear", "mean", 3, torch.float32, False, True, "halo-exchange"],
            ["healpix", (8,), "healpix", (8,), 4, 8, (3), "piecewise linear", "mean", 1, torch.float32, False, False, "reduce-scatter"],
            ["healpix", (8,), "healpix", (4,), 4, 8, (3, 3), "morlet", "mean", 1, torch.float32, False, True, "reduce-scatter"],
            ["healpix", (4,), "healpix", (8,), 4, 8, (3), "piecewise linear", "mean", 1, torch.float32, True, False, "halo-exchange"],
            ["healpix", (8,), "healpix", (8,), 4, 8, (3, 3), "morlet", "mean", 1, torch.float32, True, False, "halo-exchange"],
            ["healpix", (4,), "healpix", (8,), 4, 8, (3), "piecewise linear", "mean", 1, torch.float32, True, False, "reduce-scatter"],
            # a regular grid paired with HEALPix keeps its (nlat, nlon) layout at the interface
            ["healpix", (8,), "equiangular", (17, 32), 4, 8, (3), "piecewise linear", "mean", 1, torch.float32, False, False, "halo-exchange"],
            ["equiangular", (17, 32), "healpix", (8,), 4, 8, (3), "piecewise linear", "mean", 1, torch.float32, False, False, "halo-exchange"],
            ["equiangular", (9, 16), "healpix", (8,), 4, 8, (3), "piecewise linear", "mean", 1, torch.float32, True, False, "halo-exchange"],
            ["equiangular", (9, 16), "healpix", (8,), 4, 8, (3), "piecewise linear", "mean", 1, torch.float32, True, False, "reduce-scatter"],
            # a product grid convolves every shell alike
            ["healpix+radial", (8,), "healpix+radial", (8,), 4, 8, (3), "piecewise linear", "mean", 1, torch.float32, False, False, "halo-exchange"],
            ["healpix+radial", (4,), "healpix+radial", (8,), 4, 8, (3), "piecewise linear", "mean", 1, torch.float32, True, False, "halo-exchange"],
            # float64 and the torch reference
            ["healpix", (8,), "healpix", (8,), 4, 8, (3, 3), "morlet", "mean", 1, torch.float64, False, False, "halo-exchange"],
            ["healpix", (8,), "healpix", (8,), 2, 8, (3), "piecewise linear", "mean", 1, torch.float64, False, False, "halo-exchange", False],
            ["healpix", (8,), "healpix", (8,), 2, 8, (3), "piecewise linear", "mean", 1, torch.float64, False, False, "reduce-scatter", False],
            ["healpix", (4,), "healpix", (8,), 2, 8, (3), "piecewise linear", "mean", 1, torch.float64, True, False, "halo-exchange", False],
            # AMP
            ["healpix", (8,), "healpix", (8,), 4, 8, (3), "piecewise linear", "mean", 1, torch.bfloat16, False, True, "halo-exchange"],
            ["healpix", (8,), "healpix", (8,), 4, 8, (3), "piecewise linear", "mean", 1, torch.float16, True, False, "halo-exchange"],
        ],
        skip_on_empty=True,
    )
    def test_distributed_disco_conv(
        self,
        grid_in,
        in_shape,
        grid_out,
        out_shape,
        batch_size,
        num_chan,
        kernel_shape,
        basis_type,
        basis_norm_mode,
        groups,
        dtype,
        transpose,
        fused,
        polar_mode,
        optimized_kernel=True,
        verbose=True,
    ):
        if (*in_shape, *out_shape) in _SLOW_DISCO_SHAPES and not _run_slow_tests:
            self.skipTest("slow test; set TORCH_HARMONICS_RUN_SLOW_TESTS=1 to run")
        grid_in, grid_out = make_grid(grid_in, in_shape), make_grid(grid_out, out_shape)
        self.skip_unless_decomposable(grid_in, grid_out)

        # ``fused`` mirrors the serial conv: False -> standard a2a (K-expanded
        # saved), True -> reordered a2a (einsum-first, K-expanded recomputed in
        # backward). The transpose class has no ``fused`` argument, so it is
        # ignored there.

        # For AMP dtypes the modules + inputs stay in float32 and autocast
        # handles the downcast inside fwd/bwd — same pattern as the serial
        # convolution tests in test_convolution.py.
        is_amp = dtype in (torch.float16, torch.bfloat16)
        module_dtype = torch.float32 if is_amp else dtype

        set_seed(333)

        B, C = batch_size, num_chan

        disco_args = dict(
            in_channels=C,
            out_channels=C,
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            kernel_shape=kernel_shape,
            groups=groups,
            grid_in=grid_in,
            grid_out=grid_out,
            bias=True,
            # applied to the reference too, so the comparison stays between the same two
            # algorithms and a fallback row is not silently checking the arc kernel
            optimized_kernel=optimized_kernel,
        )

        # set up handles
        if transpose:
            # Transpose class has no ``fused`` argument; it is ignored here.
            conv_local = th.DiscreteContinuousConvTransposeS2(**disco_args).to(dtype=module_dtype, device=self.device)
            conv_dist = thd.DistributedDiscreteContinuousConvTransposeS2(**disco_args, polar_mode=polar_mode).to(dtype=module_dtype, device=self.device)
        else:
            conv_local = th.DiscreteContinuousConvS2(**disco_args).to(dtype=module_dtype, device=self.device)
            conv_dist = thd.DistributedDiscreteContinuousConvS2(
                **disco_args,
                fused=fused,
                polar_mode=polar_mode,
            ).to(dtype=module_dtype, device=self.device)

        # the transpose must declare itself so: the kpacked backend serves the forward only,
        # and would otherwise pack the transpose's psi, keyed by the other grid's latitudes
        self.assertEqual(conv_dist.transpose, transpose)
        self.assertEqual(conv_local.transpose, transpose)
        if transpose:
            self.assertNotEqual(conv_dist.backend.name, "regular-kpacked")

        # copy the weights from the local conv into the dist conv
        with torch.no_grad():
            conv_dist.weight.copy_(conv_local.weight)
            if disco_args["bias"]:
                conv_dist.bias.copy_(conv_local.bias)

        # create tensors
        inp_full = torch.randn((B, C, *grid_in.shape), dtype=module_dtype, device=self.device)

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
        inp_local = self.split_field(inp_full.detach().clone(), conv_dist.shard_in)
        inp_local.requires_grad = True
        with maybe_autocast(self.device.type, dtype):
            out_local = conv_dist(inp_local)

        # BWD pass
        ograd_local = self.split_field(ograd_full, conv_dist.shard_out)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        # Print diagnostics from rank 0 only; assert the all-reduced verdict on every
        # rank so a failure on any rank fails the test consistently (see reduce_success).
        verbose = verbose and self.world_rank == 0

        # evaluate FWD pass
        out_gather_full = self.gather_field(out_local, conv_dist.shard_out)
        ok = compare_tensors("output", out_gather_full, out_full, atol=0.0, rtol=0.0, ulps=_ulps(dtype), eps_dtype=dtype, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "output")

        # evaluate BWD pass
        igrad_gather_full = self.gather_field(igrad_local, conv_dist.shard_in)
        ok = compare_tensors("gradients", igrad_gather_full, igrad_full, atol=0.0, rtol=0.0, ulps=_ulps(dtype), eps_dtype=dtype, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "gradients")

        # evaluate parameter gradients — local per-rank contributions
        # summed across the h and w groups must match the serial gradient.
        # Parameter grads are sums over batch and the whole grid, reassociated
        # across ranks, so their absolute rounding error scales with the
        # largest entries rather than the entry it lands on: entries that
        # cancel to ~0 carry the same error as their neighbours. The atol is
        # therefore floored at ULPs of max|ref|, see param_grad_ulps.
        pg_ulps = param_grad_ulps(B * max(grid_in.npoints, grid_out.npoints), dtype)
        if conv_dist.weight.grad is not None:
            wgrad = self._allreduce_param_grad(conv_dist.weight.grad)
            ok = compare_tensors("weight grad", wgrad, conv_local.weight.grad, atol=0.0, rtol=0.0, verbose=verbose, ulps=pg_ulps, eps_dtype=dtype)
            self.assertTrue(reduce_success(ok, self.device), "weight grad")
        if getattr(conv_dist, "bias", None) is not None and conv_dist.bias.grad is not None:
            bgrad = self._allreduce_param_grad(conv_dist.bias.grad)
            ok = compare_tensors("bias grad", bgrad, conv_local.bias.grad, atol=0.0, rtol=0.0, verbose=verbose, ulps=pg_ulps, eps_dtype=dtype)
            self.assertTrue(reduce_success(ok, self.device), "bias grad")

    def _run_kpacked_fallback(self, fused: bool, dtype: torch.dtype):
        """Shared body for kpacked-fallback tests.

        Rules the kpacked backend out, so the distributed conv runs the arc kernels even
        with bf16/fp16 input, and verifies fwd+bwd correctness against the serial reference
        (which has kpacked ruled out the same way).
        """

        set_seed(555)
        nlat, nlon = 64, 128
        B, C = 8, 8

        args = dict(
            in_channels=C,
            out_channels=C,
            basis_type="piecewise linear",
            basis_norm_mode="mean",
            kernel_shape=(3,),
            groups=1,
            grid_in=th.as_grid("equiangular", nlat=nlat, nlon=nlon),
            grid_out=th.as_grid("equiangular", nlat=nlat, nlon=nlon),
            bias=True,
        )

        conv_local = th.DiscreteContinuousConvS2(**args).to(dtype=torch.float32, device=self.device)
        conv_dist = thd.DistributedDiscreteContinuousConvS2(**args, fused=fused).to(dtype=torch.float32, device=self.device)

        with torch.no_grad():
            conv_dist.weight.copy_(conv_local.weight)
            conv_dist.bias.copy_(conv_local.bias)

        # Force both onto the arc kernels.
        for conv in (conv_local, conv_dist):
            conv._backends = (RegularOptimizedBackend, RegularReferenceBackend)
            conv._select_backend()

        inp_full = torch.randn((B, C, nlat, nlon), dtype=torch.float32, device=self.device)

        inp_full.requires_grad = True
        with maybe_autocast(self.device.type, dtype):
            out_full = conv_local(inp_full)
        ograd_full = torch.randn_like(out_full)
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        inp_local = self.split_field(inp_full.detach().clone(), conv_dist.shard_in)
        inp_local.requires_grad = True
        with maybe_autocast(self.device.type, dtype):
            out_local = conv_dist(inp_local)
        ograd_local = self.split_field(ograd_full, conv_dist.shard_out)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        verbose = self.world_rank == 0
        out_gather = self.gather_field(out_local, conv_dist.shard_out)
        ok = compare_tensors("output", out_gather, out_full, atol=0.0, rtol=0.0, ulps=_ulps(dtype), eps_dtype=dtype, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "output")

        igrad_gather = self.gather_field(igrad_local, conv_dist.shard_in)
        ok = compare_tensors("gradients", igrad_gather, igrad_full, atol=0.0, rtol=0.0, ulps=_ulps(dtype), eps_dtype=dtype, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "gradients")

    def test_kpacked_fallback_bf16_unfused(self):
        """bf16 + kpacked ruled out → arc kernels, fused=False."""
        self._run_kpacked_fallback(fused=False, dtype=torch.bfloat16)

    def test_kpacked_fallback_bf16_fused(self):
        """bf16 + kpacked ruled out → arc kernels, fused=True."""
        self._run_kpacked_fallback(fused=True, dtype=torch.bfloat16)

    def test_kpacked_fallback_fp16_unfused(self):
        """fp16 + kpacked ruled out → arc kernels, fused=False."""
        self._run_kpacked_fallback(fused=False, dtype=torch.float16)

    @parameterized.expand(
        [
            # in_shape, out_shape, in_channels, out_channels, fused, polar_mode, dtype: pscale 1 and 2, the standard
            # and the reordered a2a, halo exchange and reduce-scatter, and with the reordered a2a and fewer output than
            # input channels the spatial-first input gradient (one scatter per basis function)
            [in_shape, out_shape, cin, cout, fused, polar_mode, dtype]
            for in_shape, out_shape, cin, cout, fused, polar_mode in (
                ((64, 128), (64, 128), 8, 8, False, "halo-exchange"),
                ((64, 128), (64, 128), 8, 8, True, "halo-exchange"),
                ((64, 128), (32, 64), 8, 8, False, "halo-exchange"),
                ((64, 128), (32, 64), 8, 8, True, "halo-exchange"),
                ((64, 128), (64, 128), 8, 2, True, "halo-exchange"),
                ((64, 128), (32, 64), 8, 2, True, "halo-exchange"),
                ((64, 128), (64, 128), 8, 8, False, "reduce-scatter"),
                ((64, 128), (32, 64), 8, 2, True, "reduce-scatter"),
                # odd latitudes: ranks hold bands of different heights, so their plans differ
                ((65, 128), (33, 64), 8, 8, False, "halo-exchange"),
            )
            for dtype in (torch.float16, torch.bfloat16)
        ]
    )
    def test_regular_fft_backend(self, in_shape, out_shape, in_channels, out_channels, fused, polar_mode, dtype):
        """
        The distributed conv on the regular-fft backend (the kpacked forward, the frequency-space
        scatter) against the serial conv on the kpacked one (the same forward, the arc scatter).

        Each rank runs the scatter on its own block, so each builds plans of its own shapes. The
        backend is experimental and not selected by default; the test selects it explicitly.
        """
        if not fft_scatter_available(self.device):
            # the same verdict on every rank: they run on the same kind of device and build
            self.skipTest("the frequency-space scatter needs a 9.0 or 10.0 GPU and a build with 9.0a / 10.0a")

        set_seed(333)
        B = 2
        grid_in = th.as_grid("equiangular", nlat=in_shape[0], nlon=in_shape[1])
        grid_out = th.as_grid("equiangular", nlat=out_shape[0], nlon=out_shape[1])
        args = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            basis_type="harmonic",
            basis_norm_mode="mean",
            kernel_shape=(3, 3),
            groups=1,
            grid_in=grid_in,
            grid_out=grid_out,
            bias=True,
            theta_cutoff=2.5 * torch.pi / float(out_shape[0] - 1),
        )

        conv_local = th.DiscreteContinuousConvS2(**args).to(dtype=torch.float32, device=self.device)
        conv_dist = thd.DistributedDiscreteContinuousConvS2(**args, fused=fused, polar_mode=polar_mode).to(dtype=torch.float32, device=self.device)
        with torch.no_grad():
            conv_dist.weight.copy_(conv_local.weight)
            conv_dist.bias.copy_(conv_local.bias)

        conv_local._backends = (RegularKpackedBackend, RegularOptimizedBackend, RegularReferenceBackend)
        conv_local._select_backend()
        conv_dist._backends = (RegularFftBackend, RegularOptimizedBackend, RegularReferenceBackend)
        conv_dist._select_backend()
        self.assertEqual(conv_local.backend.name, "regular-kpacked")
        self.assertEqual(conv_dist.backend.name, "regular-fft")
        self.assertEqual(conv_dist.backend.plan.has_split, fused and 2 * out_channels <= in_channels)

        inp_full = torch.randn((B, in_channels, *in_shape), dtype=torch.float32, device=self.device)
        inp_full.requires_grad = True
        with maybe_autocast(self.device.type, dtype):
            out_full = conv_local(inp_full)
        ograd_full = torch.randn_like(out_full)
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        inp_local = self.split_field(inp_full.detach().clone(), conv_dist.shard_in)
        inp_local.requires_grad = True
        with maybe_autocast(self.device.type, dtype):
            out_local = conv_dist(inp_local)
        ograd_local = self.split_field(ograd_full, conv_dist.shard_out)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        # the input gradient went through the frequency-space scatter: it built a plan for this dtype
        scatter_states = [state for (op, _, state_dtype, _), state in conv_dist.backend.plan._call_states.items() if op.startswith("scatter") and state_dtype == dtype]
        self.assertTrue(scatter_states and all(len(state) > 0 for state in scatter_states), "no frequency-space scatter plan was built")

        verbose = self.world_rank == 0
        out_gather = self.gather_field(out_local, conv_dist.shard_out)
        ok = compare_tensors("output", out_gather, out_full, atol=0.0, rtol=0.0, ulps=_ulps(dtype), eps_dtype=dtype, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "output")

        # frequency space (fp32 FFTs, rounded once to the dtype) against the arc scatter: each within about half an
        # ULP of the exact result, so within the budget of two implementations of the same contraction
        igrad_gather = self.gather_field(igrad_local, conv_dist.shard_in)
        ok = compare_tensors("gradients", igrad_gather, igrad_full, atol=0.0, rtol=0.0, ulps=_ulps(dtype), eps_dtype=dtype, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "gradients")

        # the weight gradient does not go through the scatter
        pg_ulps = param_grad_ulps(B * max(grid_in.npoints, grid_out.npoints), dtype)
        wgrad = self._allreduce_param_grad(conv_dist.weight.grad)
        ok = compare_tensors("weight grad", wgrad, conv_local.weight.grad, atol=0.0, rtol=0.0, ulps=pg_ulps, eps_dtype=dtype, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "weight grad")

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
            # the transpose plans its halo from psi, in both directions of resampling, and
            # falls back to the same reduce-scatter
            [33, 64, 17, 64, (3, 3), "equiangular", "equiangular", True, "halo-exchange", 1.0],
            [32, 64, 32, 64, (3, 3), "legendre-gauss", "legendre-gauss", True, "halo-exchange", 1.0],
            [32, 64, 32, 64, (3, 3), "equiangular", "equiangular", True, "reduce-scatter", 1.0],
            [17, 64, 33, 64, (3, 3), "equiangular", "equiangular", True, "reduce-scatter", 1.0],
            [32, 64, 32, 64, (3, 3), "equiangular", "equiangular", True, "halo-exchange", 12.0],
            [32, 64, 32, 64, (3, 3), "equiangular", "equiangular", True, "reduce-scatter", 12.0],
        ],
        skip_on_empty=True,
    )
    def test_psi_blocks(self, nlat_in, nlon_in, nlat_out, nlon_out, kernel_shape, grid_in, grid_out, transpose, polar_mode, theta_cutoff_scale, verbose=False):
        """Each rank's local psi equals the serial psi restricted to the latitudes it owns.

        The sparsity pattern is built from the latitude band that can fall inside the angular
        cutoff, and the distributed module keeps only the entries whose latitude it owns: on the
        output side under halo-exchange, on the input side under reduce-scatter.
        This checks the two agree entry for entry, which isolates the pattern from the
        convolution: an off-by-one in the band or in the index remapping shows up here rather
        than as a diffuse accuracy failure in the forward.

        Each rank compares against its own slice of the serial construction, so no collective
        is involved and a failure identifies the rank. The comparison is order-insensitive --
        the local build re-keys and re-sorts psi -- so entries are sorted by
        (kernel, output latitude, global input index) on both sides first.
        """

        set_seed(333)

        # psi does not depend on the channels, but the layer has to be buildable: its azimuth
        # all-to-all needs at least one channel per azimuth rank
        C = self.grid_size_w

        g_in, g_out = th.as_grid(grid_in, nlat=nlat_in, nlon=nlon_in), th.as_grid(grid_out, nlat=nlat_out, nlon=nlon_out)
        theta_cutoff = theta_cutoff_scale * compute_theta_cutoff(nlat_out if not transpose else nlat_in, grid=grid_out if not transpose else grid_in)
        args = dict(kernel_shape=kernel_shape, theta_cutoff=theta_cutoff)
        if transpose:
            conv_local = th.DiscreteContinuousConvTransposeS2(g_in, g_out, C, C, **args).to(self.device)

            def build():
                return thd.DistributedDiscreteContinuousConvTransposeS2(g_in, g_out, C, C, polar_mode=polar_mode, **args).to(self.device)

        else:
            conv_local = th.DiscreteContinuousConvS2(g_in, g_out, C, C, **args).to(self.device)

            def build():
                return thd.DistributedDiscreteContinuousConvS2(g_in, g_out, C, C, polar_mode=polar_mode, **args).to(self.device)

        ser_ker, ser_row, ser_col, ser_vals = conv_local._psi_coo()

        # psi's rows are latitudes of the grid it is keyed by, its columns flat points of the
        # other one: output rows over input columns in the forward, input rows over output
        # columns in the transpose
        nlon_col = nlon_out if transpose else nlon_in
        row_shapes = compute_split_shapes(nlat_in if transpose else nlat_out, self.grid_size_h)
        col_shapes = compute_split_shapes(nlat_out if transpose else nlat_in, self.grid_size_h)
        row_starts, col_starts = [0, *accumulate(row_shapes)], [0, *accumulate(col_shapes)]
        ser_col_lat = ser_col // nlon_col

        # Whether a halo can serve this configuration is a property of the geometry and the
        # decomposition, so derive it rather than hard-coding it per row: the same cutoff is
        # servable on a coarse polar split and not on a fine one. The assertion is then that
        # the constructor agrees -- refusing exactly when the support outruns a neighbour, and
        # saying which mode to use instead.
        if polar_mode == "halo-exchange":
            # the halo is planned from psi: a rank keeps its output latitudes -- psi's rows in
            # the forward, its columns in the transpose -- and reads the input latitudes their
            # entries reach, which have to lie within its immediate polar neighbours
            owned_lat, reached_lat = (ser_col_lat, ser_row) if transpose else (ser_row, ser_col_lat)
            owned_starts, reached_starts = (col_starts, row_starts) if transpose else (row_starts, col_starts)
            halo_servable = True
            for r in range(self.grid_size_h):
                sel = (owned_lat >= owned_starts[r]) & (owned_lat < owned_starts[r + 1])
                if sel.any():
                    lo, hi = int(reached_lat[sel].min()), int(reached_lat[sel].max()) + 1
                    halo_servable &= lo >= reached_starts[max(r - 1, 0)] and hi <= reached_starts[min(r + 2, self.grid_size_h)]
            if not halo_servable:
                with self.assertRaises(ValueError) as ctx:
                    build()
                self.assertIn("reduce-scatter", str(ctx.exception), "the refusal must name the mode that does work")
                return

        conv_dist = build()

        # Assert the mode rather than only reading it back: a silent fall back to the other
        # keying would satisfy the comparison while leaving this mode untested.
        self.assertEqual(conv_dist.use_halo, polar_mode == "halo-exchange", f"constructor did not honour polar_mode={polar_mode!r}")

        # The entries this rank owns, read off the decomposition rather than the layer's own
        # band: the output latitudes it computes under halo-exchange, the input latitudes it
        # holds under reduce-scatter. The output is psi's row in the forward and its column
        # in the transpose.
        owned_is_row = conv_dist.use_halo != transpose
        owned_lat = ser_row if owned_is_row else ser_col_lat
        starts = row_starts if owned_is_row else col_starts
        keep = (owned_lat >= starts[self.hrank]) & (owned_lat < starts[self.hrank + 1])

        # the layer's block, lifted back to global coordinates
        dist_ker, dist_row, dist_col, dist_vals = conv_dist._psi_coo()
        row_global = dist_row + conv_dist._psi_rows[0]
        col_global = dist_col + conv_dist._psi_cols[0]

        def sorted_entries(ker, row, col, vals):
            """Canonical ordering so the two builds are comparable regardless of layout."""
            key = (ker.to(torch.int64) * (nlat_out + nlat_in) + row.to(torch.int64)) * (nlat_in * nlat_out * max(nlon_in, nlon_out)) + col.to(torch.int64)
            order = torch.argsort(key)
            return ker[order], row[order], col[order], vals[order]

        got = sorted_entries(dist_ker, row_global, col_global, dist_vals)
        ref = sorted_entries(ser_ker[keep], ser_row[keep], ser_col[keep], ser_vals[keep])
        use_halo = conv_dist.use_halo

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
        local_nnz = torch.tensor([dist_vals.numel()], device=self.device, dtype=torch.int64)
        if self.grid_size_h > 1:
            dist.all_reduce(local_nnz, group=self.h_group)
        self.assertEqual(int(local_nnz.item()), int(ser_vals.numel()), "polar ranks together must hold every serial psi entry exactly once")

    @parameterized.expand(
        [
            ["forward", "reduce-scatter", "equiangular", (32, 64), (16, 32)],
            ["forward", "halo-exchange", "equiangular", (32, 64), (16, 32)],
            ["transpose", "halo-exchange", "equiangular", (32, 64), (16, 32)],
            ["transpose", "reduce-scatter", "equiangular", (32, 64), (16, 32)],
            ["forward", "reduce-scatter", "healpix", (8,), (4,)],
            ["forward", "halo-exchange", "healpix", (8,), (4,)],
            ["transpose", "halo-exchange", "healpix", (8,), (4,)],
        ]
    )
    def test_identical_layers_share_their_block(self, direction, polar_mode, grid, fine_shape, coarse_shape):
        """
        Two identical distributed DISCO layers share this rank's backend state, the backend
        included; one with a different support does not. See BackendSelectionMixin.
        """
        g_in, g_out = make_grid(grid, fine_shape), make_grid(grid, coarse_shape)
        self.skip_unless_decomposable(g_in)
        if direction == "transpose":

            def build(**kwargs):
                return thd.DistributedDiscreteContinuousConvTransposeS2(g_out, g_in, 4, 4, kernel_shape=3, polar_mode=polar_mode, **kwargs).to(self.device)

        else:

            def build(**kwargs):
                return thd.DistributedDiscreteContinuousConvS2(g_in, g_out, 4, 4, kernel_shape=3, polar_mode=polar_mode, **kwargs).to(self.device)

        a, b = build(), build()
        self.assertIs(a.backend, b.backend)
        for name in a._backend_state:
            self.assertIs(getattr(a, name), getattr(b, name), msg=name)
        other = build(theta_cutoff=1.5 * a.theta_cutoff)
        self.assertTrue(any(getattr(a, name) is not getattr(other, name) for name in a._backend_state))

    def test_fused_runs_on_every_backend(self):
        """
        fused=True is accepted wherever the layer can be built, the torch reference included:
        there the reordered all-to-all is kept and the intermediate is saved instead of
        recomputed. The numerics are checked by the fused reference row of
        test_distributed_disco_conv.
        """
        grid = th.as_grid("equiangular", nlat=32, nlon=64)
        conv = thd.DistributedDiscreteContinuousConvS2(grid, grid, 4, 4, kernel_shape=(3, 3), fused=True, optimized_kernel=False)
        self.assertTrue(conv.fused)
        self.assertEqual(conv.backend.name, "regular-reference")

    @parameterized.expand([["forward"], ["transpose"]])
    def test_polar_mode_rejects_unknown_value(self, direction):
        """An unrecognised mode is a typo, not a request for a default."""
        layer = thd.DistributedDiscreteContinuousConvTransposeS2 if direction == "transpose" else thd.DistributedDiscreteContinuousConvS2
        grid = th.as_grid("equiangular", nlat=32, nlon=64)
        with self.assertRaises(ValueError) as ctx:
            layer(grid, grid, 1, 1, kernel_shape=(3, 3), polar_mode="halo")
        self.assertIn("halo-exchange", str(ctx.exception))

    @parameterized.expand([["forward", "halo-exchange"], ["forward", "reduce-scatter"], ["transpose", "halo-exchange"], ["transpose", "reduce-scatter"]])
    def test_healpix_psi_blocks(self, direction, polar_mode, verbose=False):
        """
        Each rank holds exactly the serial psi entries it owns, and the polar ranks together hold each once.

        The counterpart of test_psi_blocks on HEALPix, where psi is keyed per point. Which
        entries a rank owns is read off the shards, independently of the band the layer
        plans: its own output points under halo-exchange, its own input points under
        reduce-scatter; for the transpose the output points it scatters onto under
        halo-exchange, and its own input points under reduce-scatter.
        """
        transpose = direction == "transpose"
        fine, coarse = th.HealpixGrid(nside=8), th.HealpixGrid(nside=4)
        self.skip_unless_decomposable(fine)
        g_in, g_out = (coarse, fine) if transpose else (fine, coarse)
        if transpose:
            conv_local = th.DiscreteContinuousConvTransposeS2(g_in, g_out, 1, 1, kernel_shape=(3, 3), basis_type="morlet")
            conv_dist = thd.DistributedDiscreteContinuousConvTransposeS2(g_in, g_out, 1, 1, kernel_shape=(3, 3), basis_type="morlet", polar_mode=polar_mode)
        else:
            conv_local = th.DiscreteContinuousConvS2(g_in, g_out, 1, 1, kernel_shape=(3, 3), basis_type="morlet")
            conv_dist = thd.DistributedDiscreteContinuousConvS2(g_in, g_out, 1, 1, kernel_shape=(3, 3), basis_type="morlet", polar_mode=polar_mode)

        ser_ker, ser_row, ser_col, ser_vals = conv_local._psi_coo()
        dist_ker, dist_row, dist_col, dist_vals = conv_dist._psi_coo()
        row_global = dist_row + conv_dist._psi_rows[0]
        col_global = dist_col + conv_dist._psi_cols[0]

        # the serial rows are output points of the forward, input points of the transpose,
        # and its columns the other grid's
        if polar_mode == "halo-exchange":
            shard, owned = conv_dist.shard_out, (ser_col if transpose else ser_row)
        else:
            shard, owned = conv_dist.shard_in, (ser_row if transpose else ser_col)
        lo = sum(shard.pixel_shapes[: self.hrank])
        keep = (owned >= lo) & (owned < lo + shard.pixel_shapes[self.hrank])

        def sorted_entries(ker, row, col, vals):
            key = (ker.to(torch.int64) * fine.npoints + row.to(torch.int64)) * fine.npoints + col.to(torch.int64)
            order = torch.argsort(key)
            return ker[order], row[order], col[order], vals[order]

        got = sorted_entries(dist_ker, row_global, col_global, dist_vals)
        ref = sorted_entries(ser_ker[keep], ser_row[keep], ser_col[keep], ser_vals[keep])
        self.assertEqual(got[0].numel(), ref[0].numel(), "number of local psi entries")
        for name, g, r in zip(("kernel index", "row index", "column index", "values"), got, ref):
            ok = compare_tensors(f"psi {name}", g, r, atol=0.0, rtol=0.0, verbose=verbose and self.world_rank == 0)
            self.assertTrue(reduce_success(ok, self.device), f"psi {name}")

        local_nnz = torch.tensor([dist_vals.numel()], device=self.device, dtype=torch.int64)
        if self.grid_size_h > 1:
            dist.all_reduce(local_nnz, group=self.h_group)
        self.assertEqual(int(local_nnz.item()), int(ser_vals.numel()), "polar ranks together must hold every serial psi entry exactly once")

    @parameterized.expand([["forward"], ["transpose"]])
    def test_healpix_cutoff_too_wide_for_the_decomposition(self, direction):
        """
        A support reaching past the neighbouring bands is refused on every rank, at construction,
        naming reduce-scatter, which serves it.
        """
        if self.grid_size_h < 3:
            self.skipTest("with fewer than three polar ranks every band is a neighbour")
        grid = th.HealpixGrid(nside=4)
        self.skip_unless_decomposable(grid)
        layer = thd.DistributedDiscreteContinuousConvTransposeS2 if direction == "transpose" else thd.DistributedDiscreteContinuousConvS2
        with self.assertRaisesRegex(ValueError, "too wide for this polar decomposition(.|\n)*reduce-scatter"):
            layer(grid, grid, 1, 1, kernel_shape=3, theta_cutoff=1.5)
        conv = layer(grid, grid, 1, 1, kernel_shape=3, theta_cutoff=1.5, polar_mode="reduce-scatter")
        self.assertEqual(conv.npoints_out_local, grid.npoints)

    @parameterized.expand(
        [
            ["forward", "healpix", "healpix"],
            ["forward", "healpix", "equiangular"],
            ["forward", "equiangular", "healpix"],
            ["transpose", "healpix", "healpix"],
        ]
    )
    def test_healpix_refuses_an_azimuthal_decomposition(self, direction, grid_in, grid_out):
        """A HEALPix grid on either side cannot be cut along the azimuth; the layer says so at construction."""
        if self.grid_size_w == 1:
            self.skipTest("needs an azimuth group of more than one rank")
        gin = make_grid(grid_in, (8,) if grid_in == "healpix" else (17, 32))
        gout = make_grid(grid_out, (8,) if grid_out == "healpix" else (17, 32))
        layer = thd.DistributedDiscreteContinuousConvTransposeS2 if direction == "transpose" else thd.DistributedDiscreteContinuousConvS2
        with self.assertRaises(NotImplementedError):
            layer(gin, gout, 4, 4, kernel_shape=3)

    @parameterized.expand(_DOUBLE_BACKWARD_CASES, skip_on_empty=True)
    def test_distributed_disco_conv_double_backward(self, grid_in, in_shape, grid_out, out_shape, kernel_shape, basis_type, transpose, fused, polar_mode, dtype, verbose=False):
        """Gradient of the penalty |dL/dx|^2 through the distributed convolution, against the serial one.

        L is a sum of tanh of the output, so its Hessian couples output points through the
        convolution and the penalty gradient crosses every halo, reduce-scatter and
        all-to-all the layer communicates through. The bounded tanh keeps the second-order
        gradients within fp16 range; a cubic overflows the penalty weight gradient there. On uneven latitude splits, in float64,
        and for fp16 and bf16 with an fp32 layer and input under ``maybe_autocast``, as the
        first-order autocast tests run. The loss is evaluated under autocast and both
        gradients outside it, and the gradients must come back in the dtype of the input
        and of the weight.
        """
        grid_in, grid_out = make_grid(grid_in, in_shape), make_grid(grid_out, out_shape)
        self.skip_unless_decomposable(grid_in, grid_out)
        set_seed(333)
        B, C = 2, 4
        # the layer and the input stay fp32 for the autocast cases; only maybe_autocast lowers the precision
        module_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        disco_args = dict(
            in_channels=C,
            out_channels=C,
            basis_type=basis_type,
            kernel_shape=kernel_shape,
            grid_in=grid_in,
            grid_out=grid_out,
            bias=False,
        )
        if transpose:
            conv_local = th.DiscreteContinuousConvTransposeS2(**disco_args).to(dtype=module_dtype, device=self.device)
            conv_dist = thd.DistributedDiscreteContinuousConvTransposeS2(**disco_args, polar_mode=polar_mode).to(dtype=module_dtype, device=self.device)
        else:
            conv_local = th.DiscreteContinuousConvS2(**disco_args).to(dtype=module_dtype, device=self.device)
            conv_dist = thd.DistributedDiscreteContinuousConvS2(**disco_args, fused=fused, polar_mode=polar_mode).to(dtype=module_dtype, device=self.device)
        with torch.no_grad():
            conv_dist.weight.copy_(conv_local.weight)

        def penalty_grads(conv, x, reduce_fn):
            with maybe_autocast(self.device.type, dtype):
                loss = reduce_fn(torch.sum(torch.tanh(conv(x))))
            (g,) = torch.autograd.grad(loss, x, create_graph=True)
            hx, hw = torch.autograd.grad(reduce_fn(torch.sum(g * g)), (x, conv.weight))
            return g.detach(), hx, hw

        def reduce_hw(t):
            return thd.reduce_from_azimuth_region(thd.reduce_from_polar_region(t))

        inp_full = torch.randn((B, C, *grid_in.shape), dtype=module_dtype, device=self.device)
        g_full, hx_full, hw_full = penalty_grads(conv_local, inp_full.clone().requires_grad_(True), lambda t: t)
        g_local, hx_local, hw_local = penalty_grads(conv_dist, self.split_field(inp_full, conv_dist.shard_in).clone().requires_grad_(True), reduce_hw)

        # the gradients come back in the dtype of what they are taken with respect to
        for name, t in (("gradients", g_local), ("penalty input grad", hx_local)):
            self.assertEqual(t.dtype, module_dtype, f"{name} dtype")
        self.assertEqual(hw_local.dtype, conv_dist.weight.dtype, "penalty weight grad dtype")

        verbose = verbose and self.world_rank == 0
        ulps = _ulps(dtype)
        ok = compare_tensors("gradients", self.gather_field(g_local, conv_dist.shard_in), g_full, atol=0.0, rtol=0.0, ulps=ulps, eps_dtype=dtype, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "gradients")
        ok = compare_tensors("penalty input grad", self.gather_field(hx_local, conv_dist.shard_in), hx_full, atol=0.0, rtol=0.0, ulps=ulps, eps_dtype=dtype, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "penalty input grad")
        pg_ulps = param_grad_ulps(B * max(grid_in.npoints, grid_out.npoints), dtype)
        ok = compare_tensors("penalty weight grad", self._allreduce_param_grad(hw_local), hw_full, atol=0.0, rtol=0.0, ulps=pg_ulps, eps_dtype=dtype, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "penalty weight grad")
