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
from parameterized import parameterized
from testutils import (
    compare_tensors,
    disable_tf32,
    gather_tensor_hw,
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
        # The transpose class does not accept ``method=`` yet, so ignore
        # algorithm for the transpose branch.
        if algorithm == "a2a":
            method, fused = "a2a", False
        elif algorithm == "ring":
            method, fused = "ring", False
        elif algorithm == "ring_fused":
            method, fused = "ring", True
        else:
            raise ValueError(f"unknown algorithm={algorithm!r}")

        # Ring path is CUDA-only — skip gracefully when CUDA isn't there.
        if method == "ring" and not torch.cuda.is_available():
            self.skipTest("method='ring' is CUDA-only")

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
        # __init__ and stores the flag on the instance. Patch the env
        # around construction so each parametrize row exercises the
        # requested code path. mock.patch.dict restores the prior value
        # automatically when the context exits.
        env_override = {"TORCH_HARMONICS_P2P_BUFFER": "1" if p2p_buffer else "0"}

        # set up handles
        if transpose:
            # Transpose class does not yet accept method= / fused= — the
            # ``algorithm`` and ``p2p_buffer`` test parameters are ignored
            # on this branch.
            conv_local = th.DiscreteContinuousConvTransposeS2(**disco_args).to(dtype=dtype, device=self.device)
            with mock.patch.dict(os.environ, env_override):
                conv_dist = thd.DistributedDiscreteContinuousConvTransposeS2(**disco_args).to(dtype=dtype, device=self.device)
        else:
            conv_local = th.DiscreteContinuousConvS2(**disco_args).to(dtype=dtype, device=self.device)
            with mock.patch.dict(os.environ, env_override):
                conv_dist = thd.DistributedDiscreteContinuousConvS2(
                    **disco_args,
                    method=method,
                    fused=fused,
                ).to(dtype=dtype, device=self.device)

        # copy the weights from the local conv into the dist conv
        with torch.no_grad():
            conv_dist.weight.copy_(conv_local.weight)
            if disco_args["bias"]:
                conv_dist.bias.copy_(conv_local.bias)

        # create tensors
        inp_full = torch.randn((B, C, H, W), dtype=dtype, device=self.device)

        # local conv
        # FWD pass
        inp_full.requires_grad = True
        out_full = conv_local(inp_full)

        # create grad for backward
        with torch.no_grad():
            # create full grad
            ograd_full = torch.randn_like(out_full)

        # BWD pass
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        # distributed conv
        # FWD pass
        inp_local = self._split_helper(inp_full.detach().clone())
        inp_local.requires_grad = True
        out_local = conv_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full)
        out_local = conv_dist(inp_local)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        # evaluate FWD pass
        out_gather_full = self._gather_helper_fwd(out_local, conv_dist)
        self.assertTrue(compare_tensors("output", out_full, out_gather_full, atol=atol, rtol=rtol, verbose=verbose))

        # evaluate BWD pass
        igrad_gather_full = self._gather_helper_bwd(igrad_local, conv_dist)
        self.assertTrue(compare_tensors("gradients", igrad_full, igrad_gather_full, atol=atol, rtol=rtol, verbose=verbose))


if __name__ == "__main__":
    unittest.main()
