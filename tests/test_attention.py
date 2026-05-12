# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
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

import math
import os
from time import perf_counter_ns
import unittest
from parameterized import parameterized, parameterized_class

import torch
import torch.nn.functional as F
from torch.library import opcheck

# from torch.autograd import gradcheck
from torch_harmonics import AttentionS2, NeighborhoodAttentionS2
from torch_harmonics.attention.kernels_torch.attention_torch import (
    _neighborhood_s2_attention_torch,
    _neighborhood_s2_attention_fwd_torch,
    _neighborhood_s2_attention_bwd_dv_torch,
    _neighborhood_s2_attention_bwd_dk_torch,
    _neighborhood_s2_attention_bwd_dq_torch,
    _neighborhood_s2_attention_upsample_fwd_torch,
    _neighborhood_s2_attention_upsample_bwd_dv_torch,
    _neighborhood_s2_attention_upsample_bwd_dk_torch,
    _neighborhood_s2_attention_upsample_bwd_dq_torch,
)
from torch_harmonics.attention import cuda_kernels_is_available, optimized_kernels_is_available
from torch_harmonics.quadrature import precompute_latitudes
from torch_harmonics.disco.convolution import _precompute_convolution_tensor_s2
from torch_harmonics.filter_basis import get_filter_basis

from testutils import disable_tf32, set_seed, compare_tensors

if not optimized_kernels_is_available():
    print(f"Warning: Couldn't import optimized disco convolution kernels")

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))

# perf thresholds
# CPU results normalized to 16 OpenMP threads,
# GPU results normalized to V100 16 GB GPU
# this is just to detect performance regressions, not for absolute performance
_perf_test_thresholds = {"cpu": {"fwd_ms": 1000, "bwd_ms": 8000}, 
                         "cuda": {"fwd_ms": 50, "bwd_ms": 150}}
_run_perf_tests = (os.getenv("TORCH_HARMONICS_RUN_PERF_TESTS", "0") == "1")


@parameterized_class(("device"), _devices)
class TestNeighborhoodAttentionS2(unittest.TestCase):
    """Test the neighborhood attention module (CPU/CUDA if available)."""
    
    def setUp(self):
        disable_tf32()
        torch.manual_seed(333)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(333)

    @parameterized.expand(
        [
            # Format: [batch_size, channels, channels_out, heads, in_shape, out_shape, grid_in, grid_out, use_qknorm, atol, rtol]
            [4, 4, 4, 1, (6, 12), (6, 12), "equiangular", "equiangular", False, 1e-5, 1e-3],
            [4, 4, 4, 2, (6, 12), (6, 12), "equiangular", "equiangular", False, 1e-5, 1e-3],
            [4, 4, 4, 4, (6, 12), (6, 12), "equiangular", "equiangular", False, 1e-5, 1e-3],
            [4, 4, 8, 4, (6, 12), (6, 12), "equiangular", "equiangular", False, 1e-5, 1e-3],
            [4, 8, 4, 4, (6, 12), (6, 12), "equiangular", "equiangular", False, 1e-5, 1e-3],
            [4, 1, 1, 1, (2, 4), (2, 4), "equiangular", "equiangular", False, 1e-5, 1e-3],
            [4, 1, 4, 1, (2, 4), (2, 4), "equiangular", "equiangular", False, 1e-5, 1e-3],
            [4, 4, 4, 4, (6, 12), (6, 12), "legendre-gauss", "legendre-gauss", False, 1e-5, 1e-3],
            [4, 4, 4, 1, (6, 12), (6, 12), "lobatto", "lobatto", False, 1e-5, 1e-3],
            # downsampling: nlon_in must be an integer multiple of nlon_out (pscale = nlon_in / nlon_out)
            [4, 8, 4, 4, (12, 24), (6, 12), "equiangular",    "equiangular",    False, 1e-5, 1e-3],  # lat 2x, lon 2x (pscale=2)
            [4, 4, 8, 4, (12, 24), (6, 12), "equiangular",    "equiangular",    False, 1e-5, 1e-3],  # C_in<C_out asym, pscale=2
            [4, 4, 4, 1, (6, 12),  (6, 6),  "equiangular",    "equiangular",    False, 1e-5, 1e-3],  # lon-only, pscale=2
            [4, 4, 4, 1, (12, 24), (6, 8),  "equiangular",    "equiangular",    False, 1e-5, 1e-3],  # pscale=3
            [4, 4, 4, 1, (12, 24), (3, 6),  "equiangular",    "equiangular",    False, 1e-5, 1e-3],  # pscale=4
            [4, 4, 4, 1, (12, 12), (6, 12), "equiangular",    "equiangular",    False, 1e-5, 1e-3],  # lat-only, pscale=1
            [4, 4, 4, 1, (12, 24), (6, 12), "legendre-gauss", "legendre-gauss", False, 1e-5, 1e-3],  # LG grid, pscale=2
            # odd latitude sizes
            [4, 4, 4, 1, (7, 12),  (5, 6),  "equiangular",    "equiangular",    False, 1e-5, 1e-3],  # odd-odd lat, pscale=2
            [4, 4, 4, 1, (9, 12),  (5, 4),  "equiangular",    "equiangular",    False, 1e-5, 1e-3],  # odd-odd lat, pscale=3
            [4, 4, 4, 1, (11, 24), (7, 12), "equiangular",    "equiangular",    False, 1e-5, 1e-3],  # odd-odd lat, pscale=2
            [4, 4, 4, 1, (12, 24), (11, 24),"equiangular",    "equiangular",    False, 1e-5, 1e-3],  # odd nlat_out only, pscale=1
            # upsampling: mirror of the downsampling rows above (in_shape ↔ out_shape, grid_in ↔ grid_out)
            [4, 8, 4, 4, (6, 12),  (12, 24),"equiangular",    "equiangular",    False, 1e-5, 1e-3],  # pscale_out=2
            [4, 4, 8, 4, (6, 12),  (12, 24),"equiangular",    "equiangular",    False, 1e-5, 1e-3],  # C_in<C_out asym, pscale_out=2
            [4, 4, 4, 1, (6, 6),   (6, 12), "equiangular",    "equiangular",    False, 1e-5, 1e-3],  # lon-only, pscale_out=2
            [4, 4, 4, 1, (6, 8),   (12, 24),"equiangular",    "equiangular",    False, 1e-5, 1e-3],  # pscale_out=3
            [4, 4, 4, 1, (3, 6),   (12, 24),"equiangular",    "equiangular",    False, 1e-5, 1e-3],  # pscale_out=4
            [4, 4, 4, 1, (6, 12),  (12, 12),"equiangular",    "equiangular",    False, 1e-5, 1e-3],  # lat-only, pscale_out=1
            [4, 4, 4, 1, (6, 12),  (12, 24),"legendre-gauss", "legendre-gauss", False, 1e-5, 1e-3],  # LG grid, pscale_out=2
            [4, 4, 4, 1, (5, 6),   (7, 12), "equiangular",    "equiangular",    False, 1e-5, 1e-3],  # odd-odd lat, pscale_out=2
            [4, 4, 4, 1, (5, 4),   (9, 12), "equiangular",    "equiangular",    False, 1e-5, 1e-3],  # odd-odd lat, pscale_out=3
            [4, 4, 4, 1, (7, 12),  (11, 24),"equiangular",    "equiangular",    False, 1e-5, 1e-3],  # odd-odd lat, pscale_out=2
            [4, 4, 4, 1, (11, 24), (12, 24),"equiangular",    "equiangular",    False, 1e-5, 1e-3],  # odd nlat_in only, pscale_out=1
            # same cases with QK norm enabled
            [4, 4, 4, 1, (6, 12), (6, 12), "equiangular", "equiangular", True, 1e-5, 1e-3],
            [4, 4, 4, 2, (6, 12), (6, 12), "equiangular", "equiangular", True, 1e-5, 1e-3],
            [4, 4, 4, 4, (6, 12), (6, 12), "equiangular", "equiangular", True, 1e-5, 1e-3],
            [4, 4, 8, 4, (6, 12), (6, 12), "equiangular", "equiangular", True, 1e-5, 1e-3],
            [4, 8, 4, 4, (6, 12), (6, 12), "equiangular", "equiangular", True, 1e-5, 1e-3],
            [4, 8, 4, 4, (12, 24), (6, 12), "equiangular", "equiangular", True, 1e-5, 1e-3],
            [4, 1, 1, 1, (2, 4), (2, 4), "equiangular", "equiangular", True, 1e-5, 1e-3],
            [4, 1, 4, 1, (2, 4), (2, 4), "equiangular", "equiangular", True, 1e-5, 1e-3],
            [4, 4, 4, 4, (6, 12), (6, 12), "legendre-gauss", "legendre-gauss", True, 1e-5, 1e-3],
            [4, 4, 4, 1, (6, 12), (6, 12), "lobatto", "lobatto", True, 1e-5, 1e-3],
            # downsampling: nlon_in must be an integer multiple of nlon_out (pscale = nlon_in / nlon_out)
            [4, 8, 4, 4, (12, 24), (6, 12), "equiangular",    "equiangular",    True, 1e-5, 1e-3],  # lat 2x, lon 2x (pscale=2)
            [4, 4, 8, 4, (12, 24), (6, 12), "equiangular",    "equiangular",    True, 1e-5, 1e-3],  # C_in<C_out asym, pscale=2
            [4, 4, 4, 1, (6, 12),  (6, 6),  "equiangular",    "equiangular",    True, 1e-5, 1e-3],  # lon-only, pscale=2
            [4, 4, 4, 1, (12, 24), (6, 8),  "equiangular",    "equiangular",    True, 1e-5, 1e-3],  # pscale=3
            [4, 4, 4, 1, (12, 24), (3, 6),  "equiangular",    "equiangular",    True, 1e-5, 1e-3],  # pscale=4
            [4, 4, 4, 1, (12, 12), (6, 12), "equiangular",    "equiangular",    True, 1e-5, 1e-3],  # lat-only, pscale=1
            [4, 4, 4, 1, (12, 24), (6, 12), "legendre-gauss", "legendre-gauss", True, 1e-5, 1e-3],  # LG grid, pscale=2
            # odd latitude sizes
            [4, 4, 4, 1, (7, 12),  (5, 6),  "equiangular",    "equiangular",    True, 1e-5, 1e-3],  # odd-odd lat, pscale=2
            [4, 4, 4, 1, (9, 12),  (5, 4),  "equiangular",    "equiangular",    True, 1e-5, 1e-3],  # odd-odd lat, pscale=3
            [4, 4, 4, 1, (11, 24), (7, 12), "equiangular",    "equiangular",    True, 1e-5, 1e-3],  # odd-odd lat, pscale=2
            [4, 4, 4, 1, (12, 24), (11, 24),"equiangular",    "equiangular",    True, 1e-5, 1e-3],  # odd nlat_out only, pscale=1
            # upsampling: mirror of the downsampling rows above (in_shape ↔ out_shape, grid_in ↔ grid_out)
            [4, 8, 4, 4, (6, 12),  (12, 24),"equiangular",    "equiangular",    True, 1e-5, 1e-3],  # pscale_out=2
            [4, 4, 8, 4, (6, 12),  (12, 24),"equiangular",    "equiangular",    True, 1e-5, 1e-3],  # C_in<C_out asym, pscale_out=2
            [4, 4, 4, 1, (6, 6),   (6, 12), "equiangular",    "equiangular",    True, 1e-5, 1e-3],  # lon-only, pscale_out=2
            [4, 4, 4, 1, (6, 8),   (12, 24),"equiangular",    "equiangular",    True, 1e-5, 1e-3],  # pscale_out=3
            [4, 4, 4, 1, (3, 6),   (12, 24),"equiangular",    "equiangular",    True, 1e-5, 1e-3],  # pscale_out=4
            [4, 4, 4, 1, (6, 12),  (12, 12),"equiangular",    "equiangular",    True, 1e-5, 1e-3],  # lat-only, pscale_out=1
            [4, 4, 4, 1, (6, 12),  (12, 24),"legendre-gauss", "legendre-gauss", True, 1e-5, 1e-3],  # LG grid, pscale_out=2
            [4, 4, 4, 1, (5, 6),   (7, 12), "equiangular",    "equiangular",    True, 1e-5, 1e-3],  # odd-odd lat, pscale_out=2
            [4, 4, 4, 1, (5, 4),   (9, 12), "equiangular",    "equiangular",    True, 1e-5, 1e-3],  # odd-odd lat, pscale_out=3
            [4, 4, 4, 1, (7, 12),  (11, 24),"equiangular",    "equiangular",    True, 1e-5, 1e-3],  # odd-odd lat, pscale_out=2
            [4, 4, 4, 1, (11, 24), (12, 24),"equiangular",    "equiangular",    True, 1e-5, 1e-3],  # odd nlat_in only, pscale_out=1
        ],
        skip_on_empty=True,
    )
    @unittest.skipUnless(optimized_kernels_is_available(), "skipping test because optimized kernels are not available")
    def test_custom_implementation(self, batch_size, channels, channels_out, heads, in_shape, out_shape, grid_in, grid_out, use_qknorm, atol, rtol, verbose=False):
        """Tests numerical equivalence between the custom (CUDA) implementation and the reference torch implementation"""

        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

        # set seed
        set_seed(333)

        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        # Helper: create inputs
        inputs_ref = {
            "k": torch.randn(batch_size, channels, nlat_in, nlon_in, requires_grad=True, device=self.device, dtype=torch.float32),
            "v": torch.randn(batch_size, channels, nlat_in, nlon_in, requires_grad=True, device=self.device, dtype=torch.float32),
            "q": torch.randn(batch_size, channels, nlat_out, nlon_out, requires_grad=True, device=self.device, dtype=torch.float32),
        }
        inputs_opt = {k: v.detach().clone().to(self.device).requires_grad_() for k, v in inputs_ref.items()}

        # reference input and model
        model_ref = NeighborhoodAttentionS2(in_channels=channels, out_channels=channels_out, num_heads=heads, in_shape=in_shape, out_shape=out_shape, grid_in=grid_in, grid_out=grid_out, bias=True, use_qknorm=use_qknorm, optimized_kernel=False).to(self.device)

        # Device model and inputs
        model_opt = NeighborhoodAttentionS2(in_channels=channels, out_channels=channels_out, num_heads=heads, in_shape=in_shape, out_shape=out_shape, grid_in=grid_in, grid_out=grid_out, bias=True, use_qknorm=use_qknorm, optimized_kernel=True).to(self.device)

        # Synchronize parameters of model
        model_opt.load_state_dict(model_ref.state_dict())
        for (name_ref, p_ref), (name_opt, p_opt) in zip(model_ref.named_parameters(), model_opt.named_parameters()):
            self.assertTrue(torch.allclose(p_ref.cpu(), p_opt.cpu()))

        # reference forward passes
        out_ref = model_ref(inputs_ref["q"], inputs_ref["k"], inputs_ref["v"])

        # Device output
        out_opt = model_opt(inputs_opt["q"], inputs_opt["k"], inputs_opt["v"])

        # Check forward equivalence
        self.assertTrue(torch.allclose(out_opt, out_ref, atol=atol, rtol=rtol), "Forward outputs differ between torch reference and custom implementation")

        # Backward passes
        grad = torch.randn_like(out_ref)
        out_ref.backward(grad)
        out_opt.backward(grad)

        # Check input gradient equivalence
        for inp in ["q", "v", "k"]:
            grad_ref = inputs_ref[inp].grad.cpu()
            grad_opt = inputs_opt[inp].grad.cpu()
            self.assertTrue(compare_tensors(f"input grad {inp}", grad_opt, grad_ref, atol=atol, rtol=rtol, verbose=verbose))

        # Check parameter gradient equivalence
        for (name_ref, p_ref), (name_opt, p_opt) in zip(model_ref.named_parameters(), model_opt.named_parameters()):
            pgrad_opt = p_opt.grad.cpu()
            pgrad_ref = p_ref.grad.cpu()
            self.assertTrue(compare_tensors(f"parameter grad {name_ref}", pgrad_opt, pgrad_ref, atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            # Format: [in_shape, out_shape, frozen]  -- which of {k, v, q} has no requires_grad
            # one downsample (pscale=2) and one upsample (pscale_out=2) row, each
            # exercised three times to freeze each input branch in turn.
            [(12, 24), (6, 12), "k"],  # downsample, frozen k
            [(12, 24), (6, 12), "v"],  # downsample, frozen v
            [(12, 24), (6, 12), "q"],  # downsample, frozen q
            [(6, 12), (12, 24), "k"],  # upsample,   frozen k
            [(6, 12), (12, 24), "v"],  # upsample,   frozen v
            [(6, 12), (12, 24), "q"],  # upsample,   frozen q
        ],
        skip_on_empty=True,
    )
    @unittest.skipUnless(optimized_kernels_is_available(), "skipping test because optimized kernels are not available")
    def test_selective_requires_grad(self, in_shape, out_shape, frozen, verbose=False):
        """Verifies the autograd contract when exactly one of {k, v, q} doesn't require gradients.

        Freezing the raw input AND its projection weight+bias makes the op's input tensor a
        non-requires_grad leaf, so ctx.needs_input_grad reflects the intended frozen branch.
        We then confirm:
          - forward outputs still match between torch ref and optimized,
          - the frozen input + its projection params have .grad == None,
          - the remaining input/parameter grads match between ref and optimized.
        """
        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

        set_seed(333)

        batch_size, channels, heads = 4, 4, 1
        atol, rtol = 1e-5, 1e-3
        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        def make_inputs(device):
            ins = {
                "k": torch.randn(batch_size, channels, nlat_in,  nlon_in,  device=device, dtype=torch.float32),
                "v": torch.randn(batch_size, channels, nlat_in,  nlon_in,  device=device, dtype=torch.float32),
                "q": torch.randn(batch_size, channels, nlat_out, nlon_out, device=device, dtype=torch.float32),
            }
            for name in ("k", "v", "q"):
                ins[name].requires_grad_(name != frozen)
            return ins

        inputs_ref = make_inputs(self.device)
        inputs_opt = {n: t.detach().clone().requires_grad_(t.requires_grad) for n, t in inputs_ref.items()}

        model_ref = NeighborhoodAttentionS2(
            in_channels=channels, num_heads=heads, in_shape=in_shape, out_shape=out_shape,
            grid_in="equiangular", grid_out="equiangular", bias=True, optimized_kernel=False,
        ).to(self.device)
        model_opt = NeighborhoodAttentionS2(
            in_channels=channels, num_heads=heads, in_shape=in_shape, out_shape=out_shape,
            grid_in="equiangular", grid_out="equiangular", bias=True, optimized_kernel=True,
        ).to(self.device)
        model_opt.load_state_dict(model_ref.state_dict())

        # freeze the chosen branch's projection so kw/vw/qw (the op inputs) are non-requires_grad
        for model in (model_ref, model_opt):
            getattr(model, f"{frozen}_weights").requires_grad_(False)
            bias = getattr(model, f"{frozen}_bias")
            if bias is not None:
                bias.requires_grad_(False)

        # forward
        out_ref = model_ref(inputs_ref["q"], inputs_ref["k"], inputs_ref["v"])
        out_opt = model_opt(inputs_opt["q"], inputs_opt["k"], inputs_opt["v"])
        self.assertTrue(torch.allclose(out_opt, out_ref, atol=atol, rtol=rtol),
                        f"Forward outputs differ between torch ref and optimized (frozen={frozen})")

        # backward
        grad = torch.randn_like(out_ref)
        out_ref.backward(grad)
        out_opt.backward(grad)

        # input grads: frozen one must be None, the other two must match
        for name in ("k", "v", "q"):
            g_ref = inputs_ref[name].grad
            g_opt = inputs_opt[name].grad
            if name == frozen:
                self.assertIsNone(g_ref, f"ref: expected None grad for frozen input {name}")
                self.assertIsNone(g_opt, f"opt: expected None grad for frozen input {name}")
            else:
                self.assertIsNotNone(g_ref, f"ref: missing grad for input {name}")
                self.assertIsNotNone(g_opt, f"opt: missing grad for input {name}")
                self.assertTrue(compare_tensors(
                    f"input grad {name} (frozen={frozen})",
                    g_opt.cpu(), g_ref.cpu(), atol=atol, rtol=rtol, verbose=verbose,
                ))

        # parameter grads: frozen-branch projection (weights + bias) must be None, others must match
        for (n_ref, p_ref), (n_opt, p_opt) in zip(model_ref.named_parameters(), model_opt.named_parameters()):
            if n_ref.startswith(f"{frozen}_"):
                self.assertIsNone(p_ref.grad, f"ref: expected None grad for frozen param {n_ref}")
                self.assertIsNone(p_opt.grad, f"opt: expected None grad for frozen param {n_opt}")
            else:
                self.assertIsNotNone(p_ref.grad, f"ref: missing grad for param {n_ref}")
                self.assertIsNotNone(p_opt.grad, f"opt: missing grad for param {n_opt}")
                self.assertTrue(compare_tensors(
                    f"parameter grad {n_ref} (frozen={frozen})",
                    p_opt.grad.cpu(), p_ref.grad.cpu(), atol=atol, rtol=rtol, verbose=verbose,
                ))

    # caution: multihead-implementation between full and neighborhood attention still seem to differ. tests are only done for single head
    @parameterized.expand(
        [
            # Format: [batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol]
            # same shape
            [2, 64, 1, (25, 48), (25, 48),    "equiangular",    "equiangular",    5e-2, 1e-4],
            # downsampling: nlon_in must be an integer multiple of nlon_out (pscale = nlon_in / nlon_out)
            [2, 16, 1, (24, 48), (12, 24),    "equiangular",    "equiangular",    5e-2, 1e-4],  # lat 2x, lon 2x (pscale=2)
            [2, 16, 1, (12, 24), (12, 12),    "equiangular",    "equiangular",    5e-2, 1e-4],  # lon-only, pscale=2
            [2, 16, 1, (12, 24), (6, 8),      "equiangular",    "equiangular",    5e-2, 1e-4],  # pscale=3
            [2, 16, 1, (24, 48), (6, 12),     "equiangular",    "equiangular",    5e-2, 1e-4],  # pscale=4
            [2, 16, 1, (24, 48), (12, 24),    "legendre-gauss", "legendre-gauss", 5e-2, 1e-4],  # LG grid, pscale=2
            # odd latitude sizes
            [2, 16, 1, (11, 24), (7, 12),     "equiangular",    "equiangular",    5e-2, 1e-4],  # odd-odd lat, pscale=2
            [2, 16, 1, (13, 24), (9, 8),      "equiangular",    "equiangular",    5e-2, 1e-4],  # odd-odd lat, pscale=3
            # upsampling: mirror of the downsampling rows above (in_shape ↔ out_shape, grid_in ↔ grid_out)
            [2, 16, 1, (12, 24), (24, 48),    "equiangular",    "equiangular",    5e-2, 1e-4],  # pscale_out=2
            [2, 16, 1, (12, 12), (12, 24),    "equiangular",    "equiangular",    5e-2, 1e-4],  # lon-only, pscale_out=2
            [2, 16, 1, (6, 8),   (12, 24),    "equiangular",    "equiangular",    5e-2, 1e-4],  # pscale_out=3
            [2, 16, 1, (6, 12),  (24, 48),    "equiangular",    "equiangular",    5e-2, 1e-4],  # pscale_out=4
            [2, 16, 1, (12, 24), (24, 48),    "legendre-gauss", "legendre-gauss", 5e-2, 1e-4],  # LG grid, pscale_out=2
            [2, 16, 1, (7, 12),  (11, 24),    "equiangular",    "equiangular",    5e-2, 1e-4],  # odd-odd lat, pscale_out=2
            [2, 16, 1, (9, 8),   (13, 24),    "equiangular",    "equiangular",    5e-2, 1e-4],  # odd-odd lat, pscale_out=3
        ],
        skip_on_empty=True,
    )
    @unittest.skipUnless(cuda_kernels_is_available(), "skipping test because CUDA kernels are not available")
    def test_device_vs_cpu(self, batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol, verbose=False):
        """Tests numerical equivalence between optimized CUDA and CPU implementations"""

        if (self.device.type == "cpu"):
            # comparing CPU with itself does not make sense
            return

        # set seed
        set_seed(333)

        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        # Helper: create inputs
        inputs_host = {
            "k": torch.randn(batch_size, channels, nlat_in, nlon_in, requires_grad=True, dtype=torch.float32),
            "v": torch.randn(batch_size, channels, nlat_in, nlon_in, requires_grad=True, dtype=torch.float32),
            "q": torch.randn(batch_size, channels, nlat_out, nlon_out, requires_grad=True, dtype=torch.float32),
        }
        inputs_device = {k: v.detach().clone().to(self.device).requires_grad_() for k, v in inputs_host.items()}

        # reference input and model (use default local theta_cutoff so the test is sensitive
        # to the (wi + pscale*wo) % nlon_in shift; a global cutoff makes every input a neighbor
        # of every output and collapses the shift to a permutation the result is invariant to)
        att_host = NeighborhoodAttentionS2(
            in_channels=channels, num_heads=heads, in_shape=in_shape, out_shape=out_shape, grid_in=grid_in, grid_out=grid_out, bias=True
        )

        # Device model and inputs
        att_device = NeighborhoodAttentionS2(
            in_channels=channels, num_heads=heads, in_shape=in_shape, out_shape=out_shape, grid_in=grid_in, grid_out=grid_out, bias=True
        ).to(self.device)

        # Synchronize parameters of model
        att_device.load_state_dict(att_host.state_dict())
        for (name_host, p_host), (name_device, p_device) in zip(att_host.named_parameters(), att_device.named_parameters()):
            p_host_copy = p_host.detach().clone().cpu()
            p_device_copy = p_device.detach().clone().cpu()
            self.assertTrue(compare_tensors(f"weight {name_host}", p_device_copy, p_host_copy, atol=atol, rtol=rtol, verbose=verbose))

        # reference forward passes
        out_host = att_host(inputs_host["q"], inputs_host["k"], inputs_host["v"])
        out_device = att_device(inputs_device["q"], inputs_device["k"], inputs_device["v"])
        self.assertTrue(compare_tensors(f"output", out_device.cpu(), out_host.cpu(), atol=atol, rtol=rtol, verbose=verbose))

        # Backward passes
        grad = torch.randn_like(out_host)
        out_host.backward(grad)
        out_device.backward(grad.to(self.device))

        for inp in ["q", "k", "v"]:
            igrad_host = inputs_host[inp].grad.cpu()
            igrad_device = inputs_device[inp].grad.cpu()
            self.assertTrue(compare_tensors(f"input grad {inp}", igrad_device, igrad_host, atol=atol, rtol=rtol, verbose=verbose))

        # Check parameter gradient equivalence - check only q,k, v weights
        for (name_host, p_host), (name_device, p_device) in zip(att_host.named_parameters(), att_device.named_parameters()):
            grad_host = p_host.grad.cpu()
            grad_device = p_device.grad.cpu()
            self.assertTrue(compare_tensors(f"parameter grad {name_host}", grad_device, grad_host, atol=atol, rtol=rtol, verbose=verbose))


    @parameterized.expand(
        [
            # Format: [batch_size, channels, channels_out, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol]
            [4, 4, 4, 1, (6, 12), (6, 12), "equiangular", "equiangular", 1e-2, 0],
            [4, 4, 8, 1, (6, 12), (6, 12), "equiangular", "equiangular", 1e-2, 0],
            [4, 8, 4, 1, (6, 12), (6, 12), "equiangular", "equiangular", 1e-2, 0],
            [4, 4, 4, 1, (6, 12), (6, 12), "legendre-gauss", "legendre-gauss", 1e-2, 0],
            [4, 4, 4, 1, (6, 12), (6, 12), "lobatto", "lobatto", 1e-2, 0],
        ],
        skip_on_empty=True,
    )
    def test_neighborhood_global_equivalence(self, batch_size, channels, channels_out, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol, verbose=False):
        """Tests that NeighborhoodAttentionS2 reduces to the global AttentionS2 when its neighborhood covers the whole sphere.

        Passing ``theta_cutoff = 2 * pi`` forces every input point into the support of every output point,
        so the sparse psi mechanism in NeighborhoodAttentionS2 becomes mathematically identical to the dense
        softmax(Q Kt) V computation in AttentionS2 (with the same quadrature weights applied).

        Cases are restricted to ``in_shape == out_shape`` because the neighborhood kernel advances the input
        column index by ``pscale * wo`` where ``pscale = nlon_in / nlon_out``; only when the two grids match
        does that shift reproduce the translation-invariant behavior of the global attention. With
        downsampling the shift maps multiple outputs to the same input column set, and the two modules are
        not expected to agree numerically."""

        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

        # set seed
        set_seed(333)

        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        # Helper: create inputs
        inputs_ref = {
            "k": torch.randn(batch_size, channels, nlat_in, nlon_in, requires_grad=True, device=self.device, dtype=torch.float32),
            "v": torch.randn(batch_size, channels, nlat_in, nlon_in, requires_grad=True, device=self.device, dtype=torch.float32),
            "q": torch.randn(batch_size, channels, nlat_out, nlon_out, requires_grad=True, device=self.device, dtype=torch.float32),
        }
        inputs = {k: v.detach().clone().to(self.device).requires_grad_() for k, v in inputs_ref.items()}

        # reference input and model
        model_ref = AttentionS2(in_channels=channels, out_channels=channels_out, num_heads=heads, in_shape=in_shape, out_shape=out_shape, grid_in=grid_in, grid_out=grid_out, bias=False).to(self.device)

        # Device model and inputs
        model = NeighborhoodAttentionS2(
            in_channels=channels, num_heads=heads, out_channels=channels_out, in_shape=in_shape, out_shape=out_shape, grid_in=grid_in, grid_out=grid_out, bias=False, theta_cutoff=2 * torch.pi
        )

        # Synchronize parameters of model
        model.load_state_dict(model_ref.state_dict())
        model = model.to(self.device)
        for (name_ref, p_ref), (name, p) in zip(model_ref.named_parameters(), model.named_parameters()):
            self.assertTrue(compare_tensors(f"weight {name_ref}", p, p_ref, atol=atol, rtol=rtol, verbose=verbose))

        # reference forward passes
        out_ref = model_ref(inputs_ref["q"], inputs_ref["k"], inputs_ref["v"])
        out = model(inputs["q"], inputs["k"], inputs["v"])

        # Check forward equivalence
        self.assertTrue(compare_tensors(f"output", out, out_ref, atol=atol, rtol=rtol, verbose=verbose))

        # Backward passes
        grad = torch.randn_like(out_ref)
        out_ref.backward(grad)
        out.backward(grad.to(self.device))

        # Check input gradient equivalence
        for inp in ["q", "k", "v"]:
            grad_ref = inputs_ref[inp].grad
            grad = inputs[inp].grad
            self.assertTrue(compare_tensors(f"input grad {inp}", grad, grad_ref, atol=atol, rtol=rtol, verbose=verbose))

        # Check parameter gradient equivalence - check only q,k, v weights
        for key in ["q_weights", "k_weights", "v_weights"]:
            grad_ref = getattr(model_ref, key).grad
            grad = getattr(model, key).grad
            self.assertTrue(compare_tensors(f"parameter grad {key}", grad, grad_ref, atol=atol, rtol=rtol, verbose=verbose))


    @parameterized.expand(
        [
            # Format: [batch_size, channels_in, channels_out, shape, grid, atol, rtol]
            [2, 4, 4, (4,  8),  "equiangular",    1e-5, 1e-4],
            [2, 4, 8, (4,  8),  "equiangular",    1e-5, 1e-4],
            [2, 8, 4, (4,  8),  "equiangular",    1e-5, 1e-4],
            [2, 4, 4, (6, 12),  "legendre-gauss", 1e-5, 1e-4],
            [2, 4, 4, (6, 12),  "lobatto",        1e-5, 1e-4],
        ],
        skip_on_empty=True,
    )
    def test_self_attention_kernel_equivalence(self, batch_size, channels_in, channels_out, shape, grid, atol, rtol, verbose=False):
        """For nlat_in == nlat_out and nlon_in == nlon_out the gather (downsample) and
        scatter (upsample) torch reference math kernels must produce numerically identical
        forward and backward results: the p-shift collapses to pscale = pscale_out = 1, so
        both formulations describe the same self-attention computation, just with the psi
        sparsity pattern stored as either rows-by-output (gather) or rows-by-input (scatter).

        This is a pure-Python equivalence check on the math fns themselves; the C++/CUDA
        dispatcher always routes self-attention through the gather kernel, so this test
        exercises the upsample math that the dispatcher would otherwise hide.

        Restricted to CPU: a CPU-vs-CUDA divergence on these pure-PyTorch math fns would
        indicate a PyTorch bug, not a bug in our code.
        """

        if self.device.type != "cpu":
            raise unittest.SkipTest("kernel-equivalence check runs only on CPU")

        set_seed(333)

        nlat, nlon = shape
        # head-folded shapes; pass directly to the math fns (which expect [B, C, H, W]).
        kw = torch.randn(batch_size, channels_in,  nlat, nlon, dtype=torch.float32, device=self.device)
        vw = torch.randn(batch_size, channels_out, nlat, nlon, dtype=torch.float32, device=self.device)
        qw = torch.randn(batch_size, channels_in,  nlat, nlon, dtype=torch.float32, device=self.device)

        # quadrature weights on the (input == output) grid
        _, wgl = precompute_latitudes(nlat, grid=grid)
        quad_weights = (2.0 * torch.pi * wgl.to(torch.float32) / nlon).to(self.device)

        # neighborhood pattern; theta_cutoff heuristics for self-attention agree (nlat_in == nlat_out)
        fb = get_filter_basis(kernel_shape=1, basis_type="zernike")
        theta_cutoff = math.pi / float(nlat - 1)

        # gather psi (rows by ho, cols by hi*nlon + wi_canonical)
        idx_g, _, roff_g = _precompute_convolution_tensor_s2(
            shape, shape, fb,
            grid_in=grid, grid_out=grid,
            theta_cutoff=theta_cutoff,
            transpose_normalization=False,
            basis_norm_mode="none",
            merge_quadrature=True,
        )
        col_g  = idx_g[2].contiguous().to(self.device)
        roff_g = roff_g.contiguous().to(self.device)

        # scatter psi (rows by hi, cols by ho*nlon + wo_canonical) — shapes swapped + transpose_normalization=True
        idx_s, _, roff_s = _precompute_convolution_tensor_s2(
            shape, shape, fb,
            grid_in=grid, grid_out=grid,
            theta_cutoff=theta_cutoff,
            transpose_normalization=True,
            basis_norm_mode="none",
            merge_quadrature=True,
        )
        col_s  = idx_s[2].contiguous().to(self.device)
        roff_s = roff_s.contiguous().to(self.device)

        # ---- forward ----
        y_g = _neighborhood_s2_attention_fwd_torch(
            kw, vw, qw, quad_weights, col_g, roff_g, nlon, nlat, nlon)
        y_s = _neighborhood_s2_attention_upsample_fwd_torch(
            kw, vw, qw, quad_weights, col_s, roff_s, nlon, nlat, nlon)

        self.assertTrue(compare_tensors("fwd output (gather vs scatter)", y_s, y_g, atol=atol, rtol=rtol, verbose=verbose))

        # ---- backward (dvx, dkx, dqy individually) ----
        dy = torch.randn_like(y_g)

        dvx_g = _neighborhood_s2_attention_bwd_dv_torch(kw, vw, qw, dy, quad_weights, col_g, roff_g, nlon, nlat, nlon)
        dkx_g = _neighborhood_s2_attention_bwd_dk_torch(kw, vw, qw, dy, quad_weights, col_g, roff_g, nlon, nlat, nlon)
        dqy_g = _neighborhood_s2_attention_bwd_dq_torch(kw, vw, qw, dy, quad_weights, col_g, roff_g, nlon, nlat, nlon)

        dvx_s = _neighborhood_s2_attention_upsample_bwd_dv_torch(kw, vw, qw, dy, quad_weights, col_s, roff_s, nlon, nlat, nlon)
        dkx_s = _neighborhood_s2_attention_upsample_bwd_dk_torch(kw, vw, qw, dy, quad_weights, col_s, roff_s, nlon, nlat, nlon)
        dqy_s = _neighborhood_s2_attention_upsample_bwd_dq_torch(kw, vw, qw, dy, quad_weights, col_s, roff_s, nlon, nlat, nlon)

        self.assertTrue(compare_tensors("bwd dv (gather vs scatter)", dvx_s, dvx_g, atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors("bwd dk (gather vs scatter)", dkx_s, dkx_g, atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors("bwd dq (gather vs scatter)", dqy_s, dqy_g, atol=atol, rtol=rtol, verbose=verbose))


    @parameterized.expand(
        [
            # Format: [batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol]
            # one row per dispatcher code path (gather / scatter); pscale=2 covers
            # the wip = (wi + pscale*wo) % nlon_in shift, which the trivial self-attention
            # case (pscale=1) would not.
            [4, 4, 1, (12, 24),(6, 12),  "equiangular", "equiangular", 1e-2, 0],  # downsample, pscale=2
            [4, 4, 1, (6, 12), (12, 24), "equiangular", "equiangular", 1e-2, 0],  # upsample, pscale_out=2
        ],
        skip_on_empty=True,
    )
    def test_optimized_pt2_compatibility(self, batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol, verbose=False):
        """Tests whether the optimized kernels are PyTorch 2 compatible"""

        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping GPU test because CUDA kernels are not available")

        set_seed(333)

        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        att = NeighborhoodAttentionS2(
            in_channels=channels, num_heads=heads, in_shape=in_shape, out_shape=out_shape, grid_in=grid_in, grid_out=grid_out, bias=False, optimized_kernel=True
        ).to(self.device)

        inputs = {
            "k": torch.randn(batch_size, channels, nlat_in, nlon_in, requires_grad=True, device=self.device, dtype=torch.float32),
            "v": torch.randn(batch_size, channels, nlat_in, nlon_in, requires_grad=True, device=self.device, dtype=torch.float32),
            "q": torch.randn(batch_size, channels, nlat_out, nlon_out, requires_grad=True, device=self.device, dtype=torch.float32),
        }

        kw = F.conv2d(inputs["k"], att.k_weights, att.k_bias)
        vw = F.conv2d(inputs["v"], att.v_weights, att.v_bias)
        qw = F.conv2d(inputs["q"], att.q_weights, att.q_bias) * att.scale

        test_inputs = (kw, vw, qw,
                       att.quad_weights, att.psi_col_idx, att.psi_roff_idx,
                       att.num_heads, nlon_in, nlat_out, nlon_out)

        opcheck(torch.ops.attention_kernels._neighborhood_s2_attention_optimized, test_inputs)
        # opcheck(torch.ops.attention_kernels._neighborhood_s2_attention_optimized, test_inputs, test_utils="test_schema")
        # opcheck(torch.ops.attention_kernels._neighborhood_s2_attention_optimized, test_inputs, test_utils="test_faketensor")
        #opcheck(torch.ops.attention_kernels._neighborhood_s2_attention_optimized, test_inputs, test_utils="test_aot_dispatch_dynamic")


    @parameterized.expand(
        [
            # Format: [batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out]
            # same-shape (pscale=1) and downsampling (pscale=2) cases
            [4, 4, 1, (6, 12),  (6, 12), "equiangular", "equiangular"],
            [4, 8, 4, (6, 12),  (6, 12), "equiangular", "equiangular"],
            [4, 4, 1, (12, 24), (6, 12), "equiangular", "equiangular"],
        ],
        skip_on_empty=True,
    )
    def test_ring_kernels_pt2_compatibility(self, batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, verbose=False):
        """Tests whether the ring-step CUDA kernels (used by DistributedNeighborhoodAttentionS2)
        are PyTorch 2 compatible.

        Only the local CUDA kernels are exercised — the ring exchange itself (NCCL P2P) is not
        tested here. With az_size = 1 a single ring step covers the full longitude, so we can call
        the kernels in single-rank mode (lon_lo_kx=0, lat_halo_start=0, no halo padding).

        opcheck only verifies the op contract (schema, fake tensors, AOT dispatch); the input
        tensor values do not need to be numerically meaningful, so kw/vw/qw and the gradient /
        state buffers are allocated directly with the right shapes."""

        if self.device.type != "cuda":
            raise unittest.SkipTest("ring kernels are only registered for CUDA")
        if not cuda_kernels_is_available():
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

        set_seed(333)

        nlat_in,  nlon_in  = in_shape
        nlat_out, nlon_out = out_shape
        pscale             = nlon_in // nlon_out

        # Build the module just to get a consistent (quad_weights, psi_col_idx, psi_roff_idx)
        # for the chosen grid; we do not exercise its forward.
        att = NeighborhoodAttentionS2(
            in_channels=channels, num_heads=heads, in_shape=in_shape, out_shape=out_shape,
            grid_in=grid_in, grid_out=grid_out, bias=False, optimized_kernel=True,
        ).to(self.device)

        # Channel counts after the (head-folded) projections in DistributedNeighborhoodAttentionS2.
        Bnh = batch_size * heads
        C_k = channels // heads
        C_v = channels // heads

        # Synthetic projected k/v/q with the correct kernel-side shapes.
        kw = torch.randn(Bnh, C_k, nlat_in,  nlon_in,  device=self.device, dtype=torch.float32)
        vw = torch.randn(Bnh, C_v, nlat_in,  nlon_in,  device=self.device, dtype=torch.float32)
        qw = torch.randn(Bnh, C_k, nlat_out, nlon_out, device=self.device, dtype=torch.float32)

        # The kernel expects row_idx to be the sorted permutation of output rows (by nnz),
        # not the row index buffer registered on the serial module. Build it the same way
        # _build_local_psi does in distributed_attention.py.
        nnz_per_row = (att.psi_roff_idx[1:] - att.psi_roff_idx[:-1]).cpu()
        row_idx_kernel = torch.argsort(nnz_per_row, descending=True).to(torch.int32).to(self.device)

        # ---- forward ring step ----
        # State buffers in channels-last layout, as expected by the CUDA kernels.
        y_acc     = torch.zeros(Bnh, nlat_out, nlon_out, C_v, device=self.device, dtype=torch.float32)
        alpha_sum = torch.zeros(Bnh, nlat_out, nlon_out,      device=self.device, dtype=torch.float32)
        qdotk_max = torch.full ((Bnh, nlat_out, nlon_out), float('-inf'), device=self.device, dtype=torch.float32)

        fwd_inputs = (kw, vw, qw,
                      y_acc, alpha_sum, qdotk_max,
                      att.quad_weights, att.psi_col_idx, att.psi_roff_idx, row_idx_kernel,
                      nlon_in, pscale, 0, 0, nlat_out, nlon_out)

        opcheck(torch.ops.attention_kernels.forward_ring_step, fwd_inputs)

        # ---- backward pass 1: re-accumulate softmax stats + alpha_k / alpha_kvw ----
        dy = torch.randn(Bnh, C_v, nlat_out, nlon_out, device=self.device, dtype=torch.float32)

        bwd_alpha_sum = torch.zeros(Bnh, nlat_out, nlon_out,      device=self.device, dtype=torch.float32)
        bwd_qdotk_max = torch.full ((Bnh, nlat_out, nlon_out), float('-inf'), device=self.device, dtype=torch.float32)
        integral_buf  = torch.zeros(Bnh, nlat_out, nlon_out,      device=self.device, dtype=torch.float32)
        alpha_k_buf   = torch.zeros(Bnh, nlat_out, nlon_out, C_k, device=self.device, dtype=torch.float32)
        alpha_kvw_buf = torch.zeros(Bnh, nlat_out, nlon_out, C_k, device=self.device, dtype=torch.float32)

        bwd1_inputs = (kw, vw, qw, dy,
                       bwd_alpha_sum, bwd_qdotk_max, integral_buf, alpha_k_buf, alpha_kvw_buf,
                       att.quad_weights, att.psi_col_idx, att.psi_roff_idx, row_idx_kernel,
                       nlon_in, pscale, 0, 0, nlat_out, nlon_out)

        opcheck(torch.ops.attention_kernels.backward_ring_step_pass1, bwd1_inputs)

        # ---- backward pass 2: scatter dkx/dvx using finalized stats ----
        # Synthetic but well-formed stats from a previous pass1: avoid -inf in qdotk_max and 0 in
        # alpha_sum so the kernel doesn't produce NaNs (opcheck doesn't check numerics, but NaNs
        # can interact badly with AOT dispatch comparisons).
        fwd_alpha_sum = torch.ones (Bnh, nlat_out, nlon_out, device=self.device, dtype=torch.float32)
        fwd_qdotk_max = torch.zeros(Bnh, nlat_out, nlon_out, device=self.device, dtype=torch.float32)
        integral_norm = torch.zeros(Bnh, nlat_out, nlon_out, device=self.device, dtype=torch.float32)

        dkw = torch.zeros(Bnh, nlat_in, nlon_in, C_k, device=self.device, dtype=torch.float32)
        dvw = torch.zeros(Bnh, nlat_in, nlon_in, C_v, device=self.device, dtype=torch.float32)

        bwd2_inputs = (kw, vw, qw, dy,
                       fwd_alpha_sum, fwd_qdotk_max, integral_norm,
                       dkw, dvw,
                       att.quad_weights, att.psi_col_idx, att.psi_roff_idx, row_idx_kernel,
                       nlon_in, pscale, 0, 0, nlat_out, nlon_out)

        opcheck(torch.ops.attention_kernels.backward_ring_step_pass2, bwd2_inputs)


    @parameterized.expand(
        [
            # self attention
            [1, 256, 1, (91, 180), (91, 180), "equiangular", "equiangular", 1e-5, 1e-5],
        ],
        skip_on_empty=True,
    )
    @unittest.skipUnless(optimized_kernels_is_available() and _run_perf_tests, "skipping performance test because optimized kernels are not available or perf tests are disabled")
    def test_perf(self, batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol, verbose=False):
        
        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

        # set seed
        set_seed(333)

        # extract some parameters
        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        # TODO: this test seems hardcoded for GPU. Is this necessary?
        k_inp = torch.randn(batch_size, channels, nlat_in, nlon_in, dtype=torch.float32, device=self.device)
        k_inp.requires_grad = False
        v_inp = torch.randn(batch_size, channels, nlat_in, nlon_in, dtype=torch.float32, device=self.device)
        v_inp.requires_grad = False
        q_inp = torch.randn(batch_size, channels, nlat_out, nlon_out, dtype=torch.float32, device=self.device)
        q_inp.requires_grad = False

        att_optimized = NeighborhoodAttentionS2(in_channels=channels, num_heads=heads,
                                                in_shape=in_shape, out_shape=out_shape,
                                                grid_in=grid_in, grid_out=grid_out, bias=True,
                                                optimized_kernel=True).to(self.device)

        # random weights
        with torch.no_grad():
            att_optimized.q_weights.normal_()
            att_optimized.k_weights.normal_()
            att_optimized.v_weights.normal_()
            att_optimized.q_bias.normal_()
            att_optimized.k_bias.normal_()
            att_optimized.v_bias.normal_()

        # forward test
        # warmup
        for i in range(2):
            out_optimized = att_optimized(q_inp, k_inp, v_inp)

        # start timer
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = perf_counter_ns()
        out_optimized = att_optimized(q_inp, k_inp, v_inp)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        end = perf_counter_ns()
        duration = (end - start) / 1e6
        if verbose:
            print(f"Forward execution time on device {self.device.type}: {duration:.2f} ms")
        threshold = _perf_test_thresholds[self.device.type]["fwd_ms"]
        self.assertTrue(duration <= threshold, msg=f"Forward execution time on device {self.device.type} is too high: {duration:.2f} ms > {threshold:.2f} ms")

        # # backward test
        out_optimized = att_optimized(q_inp, k_inp, v_inp)
        out_grad = torch.randn(out_optimized.shape, dtype=torch.float32, device=self.device)
        
        # # warmup
        for i in range(2):
            out_optimized.backward(out_grad, retain_graph=True)

        # start timer
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = perf_counter_ns()
        out_optimized.backward(out_grad)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        end = perf_counter_ns()
        duration = (end - start) / 1e6
        if verbose:
            print(f"Backward execution time on device {self.device.type}: {duration:.2f} ms")
        threshold = _perf_test_thresholds[self.device.type]["bwd_ms"]
        self.assertTrue(duration <= threshold, msg=f"Backward execution time on device {self.device.type} is too high: {duration:.2f} ms > {threshold:.2f} ms")

    def test_wrong_shape_assertions(self):
        """Verify that forward raises ValueError on spatial-shape mismatches."""
        B, C = 2, 16
        in_shape  = (12, 24)
        out_shape = (6,  12)
        nlat_in, nlon_in   = in_shape
        nlat_out, nlon_out = out_shape

        model = NeighborhoodAttentionS2(
            in_channels=C,
            in_shape=in_shape,
            out_shape=out_shape,
            grid_in="equiangular",
            grid_out="equiangular",
            num_heads=1,
            bias=False,
        ).to(self.device)

        q  = torch.randn(B, C, nlat_out, nlon_out, device=self.device)
        kv = torch.randn(B, C, nlat_in,  nlon_in,  device=self.device)

        # 1. Self-attention on an up/downsampling module: a single tensor cannot
        #    simultaneously satisfy in_shape (for k/v) and out_shape (for q).
        with self.assertRaises(ValueError):
            model(q)  # key defaults to query, but key must have in_shape

        # 2. q_shape == k_shape != v_shape: key carries out_shape instead of in_shape.
        with self.assertRaises(ValueError):
            model(q, q, kv)

        # 3. q_shape == v_shape != k_shape: value carries out_shape instead of in_shape.
        with self.assertRaises(ValueError):
            model(q, kv, q)

if __name__ == "__main__":
    unittest.main()
