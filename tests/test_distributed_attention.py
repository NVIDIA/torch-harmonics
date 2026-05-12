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
import torch_harmonics as th
import torch_harmonics.distributed as thd

from testutils import (
    set_seed,
    setup_module,
    teardown_module,
    setup_class_from_context,
    split_tensor_hw,
    gather_tensor_hw,
    compare_tensors,
)

# shared state
_DIST_CTX = {}

def setUpModule():
    setup_module(_DIST_CTX)

def tearDownModule():
    teardown_module(_DIST_CTX)


class TestDistributedNeighborhoodAttention(unittest.TestCase):
    """
    Compare serial NeighborhoodAttentionS2 against DistributedNeighborhoodAttentionS2.

    CPU-only runs are skipped: distributed attention requires CUDA (NCCL + custom kernels).
    """

    @classmethod
    def setUpClass(cls):
        setup_class_from_context(cls, _DIST_CTX)
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Distributed neighborhood attention requires CUDA")

    def _split_helper(self, tensor):
        return split_tensor_hw(
            tensor,
            hdim=-2,
            wdim=-1,
            hsize=self.grid_size_h,
            wsize=self.grid_size_w,
            hrank=self.hrank,
            wrank=self.wrank,
        )

    def _gather_helper_fwd(self, tensor, attn_dist):
        return gather_tensor_hw(
            tensor,
            hdim=-2,
            wdim=-1,
            hshapes=attn_dist.lat_out_shapes,
            wshapes=attn_dist.lon_out_shapes,
            hsize=self.grid_size_h,
            wsize=self.grid_size_w,
            hrank=self.hrank,
            wrank=self.wrank,
            hgroup=self.h_group,
            wgroup=self.w_group,
        )

    def _gather_helper_bwd(self, tensor, attn_dist, use_out_shapes=False):
        hshapes = attn_dist.lat_out_shapes if use_out_shapes else attn_dist.lat_in_shapes
        wshapes = attn_dist.lon_out_shapes if use_out_shapes else attn_dist.lon_in_shapes
        return gather_tensor_hw(
            tensor,
            hdim=-2,
            wdim=-1,
            hshapes=hshapes,
            wshapes=wshapes,
            hsize=self.grid_size_h,
            wsize=self.grid_size_w,
            hrank=self.hrank,
            wrank=self.wrank,
            hgroup=self.h_group,
            wgroup=self.w_group,
        )

    @parameterized.expand(
        [
            # nlat_in, nlon_in, nlat_out, nlon_out, batch_size, in_channels, num_heads, k_channels, out_channels, grid_in, grid_out, use_qknorm, atol, rtol
            # same shape tests
            [64, 128, 64, 128, 2, 16, 1, None, None, "equiangular", "equiangular", False, 1e-5, 1e-4],
            [64, 128, 64, 128, 2, 16, 2, None, None, "equiangular", "equiangular", False, 1e-5, 1e-4],
            [64, 128, 64, 128, 2, 16, 1,   8,    8, "equiangular", "equiangular", False, 1e-5, 1e-4],
            [65, 128, 65, 128, 2, 16, 1, None, None, "equiangular", "equiangular", False, 1e-5, 1e-4],
            # downsampling tests, pscale=2 (lat+lon)
            [64, 128, 32,  64, 2, 16, 1, None, None, "equiangular", "equiangular", False, 1e-5, 1e-4],
            [65, 128, 33,  64, 2, 16, 1, None, None, "equiangular", "equiangular", False, 1e-5, 1e-4],
            # pscale=3 downsampling (exercises pscale*wo kernel arithmetic)
            [64,  96, 32,  32, 2, 16, 1, None, None, "equiangular", "equiangular", False, 1e-5, 1e-4],
            # pscale=4 downsampling
            [64, 128, 32,  32, 2, 16, 1, None, None, "equiangular", "equiangular", False, 1e-5, 1e-4],
            # lon-only downsampling (same nlat; isolates azimuth ring with pscale_lon=2, no lat halo)
            [64, 128, 64,  64, 2, 16, 1, None, None, "equiangular", "equiangular", False, 1e-5, 1e-4],
            # lat-only downsampling (same nlon, pscale_lon=1; isolates polar halo exchange)
            [64, 128, 32, 128, 2, 16, 1, None, None, "equiangular", "equiangular", False, 1e-5, 1e-4],
            # odd nlat_in -> even nlat_out, pscale_lon=2
            [65, 128, 32,  64, 2, 16, 1, None, None, "equiangular", "equiangular", False, 1e-5, 1e-4],
            # legendre-gauss grid (same-shape + downsampling)
            [64, 128, 64, 128, 2, 16, 1, None, None, "legendre-gauss", "legendre-gauss", False, 1e-5, 1e-4],
            [64, 128, 32,  64, 2, 16, 1, None, None, "legendre-gauss", "legendre-gauss", False, 1e-5, 1e-4],
            # mixed grid: equiangular input -> legendre-gauss output
            [64, 128, 64, 128, 2, 16, 1, None, None, "equiangular",    "legendre-gauss", False, 1e-5, 1e-4],
            [64, 128, 32,  64, 2, 16, 1, None, None, "equiangular",    "legendre-gauss", False, 1e-5, 1e-4],
            # heads=4 with asymmetric channels (k=32, out=16; in=16)
            [64, 128, 64, 128, 2, 16, 4,  32,   16, "equiangular", "equiangular", False, 1e-5, 1e-4],
            [64, 128, 32,  64, 2, 16, 4,  32,   16, "equiangular", "equiangular", False, 1e-5, 1e-4],
            # same cases with QK norm enabled
            [64, 128, 64, 128, 2, 16, 1, None, None, "equiangular", "equiangular", True, 1e-5, 1e-4],
            [64, 128, 64, 128, 2, 16, 2, None, None, "equiangular", "equiangular", True, 1e-5, 1e-4],
            [64, 128, 64, 128, 2, 16, 1,   8,    8, "equiangular", "equiangular", True, 1e-5, 1e-4],
            [65, 128, 65, 128, 2, 16, 1, None, None, "equiangular", "equiangular", True, 1e-5, 1e-4],
            # downsampling tests with QK norm enabled, pscale=2
            [64, 128, 32,  64, 2, 16, 1, None, None, "equiangular", "equiangular", True, 1e-5, 1e-4],
            [65, 128, 33,  64, 2, 16, 1, None, None, "equiangular", "equiangular", True, 1e-5, 1e-4],
            # pscale=3 / pscale=4 downsampling with QK norm
            [64,  96, 32,  32, 2, 16, 1, None, None, "equiangular", "equiangular", True, 1e-5, 1e-4],
            [64, 128, 32,  32, 2, 16, 1, None, None, "equiangular", "equiangular", True, 1e-5, 1e-4],
            # lon-only / lat-only downsampling with QK norm
            [64, 128, 64,  64, 2, 16, 1, None, None, "equiangular", "equiangular", True, 1e-5, 1e-4],
            [64, 128, 32, 128, 2, 16, 1, None, None, "equiangular", "equiangular", True, 1e-5, 1e-4],
            # odd nlat_in -> even nlat_out with QK norm
            [65, 128, 32,  64, 2, 16, 1, None, None, "equiangular", "equiangular", True, 1e-5, 1e-4],
            # legendre-gauss grid with QK norm
            [64, 128, 64, 128, 2, 16, 1, None, None, "legendre-gauss", "legendre-gauss", True, 1e-5, 1e-4],
            [64, 128, 32,  64, 2, 16, 1, None, None, "legendre-gauss", "legendre-gauss", True, 1e-5, 1e-4],
            # upsampling is not supported by the kernel yet (serial layer asserts nlon_in % nlon_out == 0)
        ],
        skip_on_empty=True,
    )
    def test_distributed_neighborhood_attention(
        self,
        nlat_in, nlon_in, nlat_out, nlon_out,
        batch_size, in_channels, num_heads, k_channels, out_channels,
        grid_in, grid_out, use_qknorm,
        atol, rtol,
        verbose=True,
    ):
        set_seed(333)

        B, C, Hi, Wi, Ho, Wo = batch_size, in_channels, nlat_in, nlon_in, nlat_out, nlon_out

        attn_args = dict(
            in_channels=C,
            in_shape=(nlat_in, nlon_in),
            out_shape=(nlat_out, nlon_out),
            grid_in=grid_in,
            grid_out=grid_out,
            num_heads=num_heads,
            bias=True,
            use_qknorm=use_qknorm,
            k_channels=k_channels,
            out_channels=out_channels,
        )

        # build serial and distributed modules with identical weights
        attn_serial = th.NeighborhoodAttentionS2(**attn_args).to(self.device)
        attn_dist   = thd.DistributedNeighborhoodAttentionS2(**attn_args).to(self.device)

        with torch.no_grad():
            attn_dist.k_weights.copy_(attn_serial.k_weights)
            attn_dist.v_weights.copy_(attn_serial.v_weights)
            attn_dist.q_weights.copy_(attn_serial.q_weights)
            attn_dist.proj_weights.copy_(attn_serial.proj_weights)
            if attn_args["bias"]:
                attn_dist.k_bias.copy_(attn_serial.k_bias)
                attn_dist.v_bias.copy_(attn_serial.v_bias)
                attn_dist.q_bias.copy_(attn_serial.q_bias)
                attn_dist.proj_bias.copy_(attn_serial.proj_bias)
            if use_qknorm:
                attn_dist.q_norm_weights.copy_(attn_serial.q_norm_weights)
                attn_dist.k_norm_weights.copy_(attn_serial.k_norm_weights)

        # Helper: create inputs
        inp_full = {
            "k": torch.randn(B, C, Hi, Wi, requires_grad=True, device=self.device, dtype=torch.float32),
            "v": torch.randn(B, C, Hi, Wi, requires_grad=True, device=self.device, dtype=torch.float32),
            "q": torch.randn(B, C, Ho, Wo, requires_grad=True, device=self.device, dtype=torch.float32),
        }

        # ---- serial forward ----
        out_full = attn_serial(inp_full["q"], inp_full["k"], inp_full["v"])

        torch.cuda.synchronize()

        # ---- serial backward ----
        with torch.no_grad():
            ograd_full = torch.randn_like(out_full)
        out_full.backward(ograd_full)

        igrad_full = {}
        for inp in ["q", "k", "v"]:
            igrad_full[inp] = inp_full[inp].grad.clone()

        torch.cuda.synchronize()

        # ---- distributed forward ----
        inp_local = {}
        for inp in ["q", "k", "v"]:
            inp_local[inp] = self._split_helper(inp_full[inp].detach())
            inp_local[inp].requires_grad_(True)
        out_local = attn_dist(inp_local["q"], inp_local["k"], inp_local["v"])

        torch.cuda.synchronize()

        # ---- distributed backward ----
        ograd_local = self._split_helper(ograd_full)
        out_local.backward(ograd_local)

        igrad_local = {}
        for inp in ["q", "k", "v"]:
            igrad_local[inp] = inp_local[inp].grad.clone()

        torch.cuda.synchronize()

        # ---- compare forward ----
        out_gather = self._gather_helper_fwd(out_local, attn_dist)
        self.assertTrue(compare_tensors("forward output", out_full, out_gather, atol=atol, rtol=rtol, verbose=verbose))

        # ---- compare backward ----
        for inp in ["q", "k", "v"]:
            use_out = (inp == "q")
            igrad_gather = self._gather_helper_bwd(igrad_local[inp], attn_dist, use_out_shapes=use_out)
            self.assertTrue(compare_tensors(f"input gradient {inp}", igrad_full[inp], igrad_gather, atol=atol, rtol=rtol, verbose=verbose))


    @parameterized.expand(
        [
            # (nlat_in, nlon_in, nlat_out, nlon_out, batch, in_chans, heads, k_chans, out_chans, grid_in, grid_out, frozen)
            # one same-shape config and one downsample config (pscale=2), each with each branch frozen in turn.
            [64, 128, 64, 128, 2, 16, 1, None, None, "equiangular", "equiangular", "k"],
            [64, 128, 64, 128, 2, 16, 1, None, None, "equiangular", "equiangular", "v"],
            [64, 128, 64, 128, 2, 16, 1, None, None, "equiangular", "equiangular", "q"],
            [64, 128, 32,  64, 2, 16, 1, None, None, "equiangular", "equiangular", "k"],
            [64, 128, 32,  64, 2, 16, 1, None, None, "equiangular", "equiangular", "v"],
            [64, 128, 32,  64, 2, 16, 1, None, None, "equiangular", "equiangular", "q"],
        ],
        skip_on_empty=True,
    )
    def test_distributed_neighborhood_attention_selective_requires_grad(
        self,
        nlat_in, nlon_in, nlat_out, nlon_out,
        batch_size, in_channels, num_heads, k_channels, out_channels,
        grid_in, grid_out, frozen,
        verbose=True,
    ):
        """Freezes one of {k, v, q} at a time and verifies:
        - frozen input's `.grad` is None on both serial and distributed,
        - frozen projection's weight/bias `.grad` is None on both modules,
        - the remaining (unfrozen) input gradients match between serial and distributed.

        Both the raw module input AND the corresponding projection (weight + bias) are
        frozen, so the custom op's input tensor (kw / vw / qw) is genuinely a
        non-requires_grad leaf and `ctx.needs_input_grad[0:3]` reflects the frozen branch.
        This is what exercises the per-branch gates added to _RingNeighborhoodAttentionFn.backward.
        """
        atol, rtol = 1e-5, 1e-4
        set_seed(333)

        B, C, Hi, Wi, Ho, Wo = batch_size, in_channels, nlat_in, nlon_in, nlat_out, nlon_out

        attn_args = dict(
            in_channels=C,
            in_shape=(nlat_in, nlon_in),
            out_shape=(nlat_out, nlon_out),
            grid_in=grid_in,
            grid_out=grid_out,
            num_heads=num_heads,
            bias=True,
            use_qknorm=False,
            k_channels=k_channels,
            out_channels=out_channels,
        )

        attn_serial = th.NeighborhoodAttentionS2(**attn_args).to(self.device)
        attn_dist   = thd.DistributedNeighborhoodAttentionS2(**attn_args).to(self.device)

        with torch.no_grad():
            attn_dist.k_weights.copy_(attn_serial.k_weights)
            attn_dist.v_weights.copy_(attn_serial.v_weights)
            attn_dist.q_weights.copy_(attn_serial.q_weights)
            attn_dist.proj_weights.copy_(attn_serial.proj_weights)
            attn_dist.k_bias.copy_(attn_serial.k_bias)
            attn_dist.v_bias.copy_(attn_serial.v_bias)
            attn_dist.q_bias.copy_(attn_serial.q_bias)
            attn_dist.proj_bias.copy_(attn_serial.proj_bias)

        # Freeze the chosen branch's projection (weights + bias) on BOTH modules.
        for m in (attn_serial, attn_dist):
            getattr(m, f"{frozen}_weights").requires_grad_(False)
            bias = getattr(m, f"{frozen}_bias")
            if bias is not None:
                bias.requires_grad_(False)

        # Build full inputs; the frozen branch's input has requires_grad=False.
        inp_full = {}
        for name, shape in (("k", (B, C, Hi, Wi)), ("v", (B, C, Hi, Wi)), ("q", (B, C, Ho, Wo))):
            t = torch.randn(*shape, device=self.device, dtype=torch.float32)
            t.requires_grad_(name != frozen)
            inp_full[name] = t

        # ---- serial forward / backward ----
        out_full = attn_serial(inp_full["q"], inp_full["k"], inp_full["v"])
        torch.cuda.synchronize()
        with torch.no_grad():
            ograd_full = torch.randn_like(out_full)
        out_full.backward(ograd_full)
        igrad_full = {n: (inp_full[n].grad.clone() if inp_full[n].grad is not None else None)
                      for n in ("k", "v", "q")}
        torch.cuda.synchronize()

        # ---- distributed forward / backward ----
        inp_local = {}
        for n in ("k", "v", "q"):
            inp_local[n] = self._split_helper(inp_full[n].detach())
            inp_local[n].requires_grad_(n != frozen)
        out_local = attn_dist(inp_local["q"], inp_local["k"], inp_local["v"])
        torch.cuda.synchronize()

        ograd_local = self._split_helper(ograd_full)
        out_local.backward(ograd_local)
        igrad_local = {n: (inp_local[n].grad.clone() if inp_local[n].grad is not None else None)
                       for n in ("k", "v", "q")}
        torch.cuda.synchronize()

        # ---- compare forward ----
        out_gather = self._gather_helper_fwd(out_local, attn_dist)
        self.assertTrue(compare_tensors("forward output", out_full, out_gather,
                                        atol=atol, rtol=rtol, verbose=verbose))

        # ---- contract: frozen input grads must be None on both ----
        self.assertIsNone(igrad_full[frozen],
                          f"serial: frozen input {frozen} must have None grad")
        self.assertIsNone(igrad_local[frozen],
                          f"distributed: frozen input {frozen} must have None grad")

        # ---- contract: frozen projection's weight + bias grads must be None on both ----
        for m, label in ((attn_serial, "serial"), (attn_dist, "distributed")):
            w = getattr(m, f"{frozen}_weights")
            b = getattr(m, f"{frozen}_bias")
            self.assertIsNone(w.grad, f"{label}: {frozen}_weights frozen, grad must be None")
            if b is not None:
                self.assertIsNone(b.grad, f"{label}: {frozen}_bias frozen, grad must be None")

        # ---- unfrozen input grads must match between serial and distributed ----
        for n in ("k", "v", "q"):
            if n == frozen:
                continue
            self.assertIsNotNone(igrad_full[n],
                                 f"serial: unfrozen input {n} should have a grad")
            self.assertIsNotNone(igrad_local[n],
                                 f"distributed: unfrozen input {n} should have a grad")
            use_out = (n == "q")
            igrad_gather = self._gather_helper_bwd(igrad_local[n], attn_dist, use_out_shapes=use_out)
            self.assertTrue(compare_tensors(
                f"input grad {n} (frozen={frozen})",
                igrad_full[n], igrad_gather, atol=atol, rtol=rtol, verbose=verbose,
            ))


    def test_wrong_shape_assertions(self):
        """Verify that forward raises ValueError on spatial-shape mismatches."""
        B, C = 2, 16
        in_shape  = (64, 128)
        out_shape = (32,  64)

        attn = thd.DistributedNeighborhoodAttentionS2(
            in_channels=C,
            in_shape=in_shape,
            out_shape=out_shape,
            grid_in="equiangular",
            grid_out="equiangular",
            num_heads=1,
            bias=False,
        ).to(self.device)

        # Build correctly-shaped local tensors using the module's own local extents.
        q_local = torch.randn(B, C, attn.nlat_out_local, attn.nlon_out_local, device=self.device)
        k_local = torch.randn(B, C, attn.nlat_in_local,  attn.nlon_in_local,  device=self.device)

        # 1. Self-attention on an up/downsampling module: a single tensor cannot
        #    simultaneously satisfy in_shape (for k/v) and out_shape (for q).
        with self.assertRaises(ValueError):
            attn(q_local)  # key defaults to query, but key must have in_shape

        # 2. q_shape == k_shape != v_shape: key carries out_shape instead of in_shape.
        with self.assertRaises(ValueError):
            attn(q_local, q_local, k_local)

        # 3. q_shape == v_shape != k_shape: value carries out_shape instead of in_shape.
        with self.assertRaises(ValueError):
            attn(q_local, k_local, q_local)


if __name__ == "__main__":
    unittest.main()
