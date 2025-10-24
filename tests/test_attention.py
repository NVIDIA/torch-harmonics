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

import os
from time import perf_counter_ns
import unittest
from parameterized import parameterized, parameterized_class

import torch
from torch.library import opcheck
import torch.nn as nn

# from torch.autograd import gradcheck
from torch_harmonics import AttentionS2, NeighborhoodAttentionS2
from torch_harmonics.attention._attention_utils import _neighborhood_s2_attention_torch
from torch_harmonics.attention import cuda_kernels_is_available, optimized_kernels_is_available

if not optimized_kernels_is_available():
    print(f"Warning: Couldn't import optimized disco convolution kernels")

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))

# perf thresholds
# CPU results normalized to 16 OpenMP threads,
# GPU results normalized to V100 16 GB GPU
# this is just to detect performance regressions, not for absolute performance
_perf_test_thresholds = {"cpu": {"fwd_ms": 800, "bwd_ms": 6000}, 
                         "cuda": {"fwd_ms": 10, "bwd_ms": 30}}
_run_perf_tests = (os.getenv("TORCH_HARMONICS_RUN_PERF_TESTS", "0") == "1")


@parameterized_class(("device"), _devices)
class TestNeighborhoodAttentionS2(unittest.TestCase):
    """Test the neighborhood attention module (CPU/CUDA if available)."""
    
    def setUp(self):
        torch.manual_seed(333)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(333)

    @parameterized.expand(
        [
            # Format: [batch_size, channels, channels_out, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol]
            [4, 4, 4, 1, (6, 12), (6, 12), "equiangular", "equiangular", 1e-5, 1e-3],
            [4, 4, 4, 2, (6, 12), (6, 12), "equiangular", "equiangular", 1e-5, 1e-3],
            [4, 4, 4, 4, (6, 12), (6, 12), "equiangular", "equiangular", 1e-5, 1e-3],
            [4, 4, 8, 4, (6, 12), (6, 12), "equiangular", "equiangular", 1e-5, 1e-3],
            [4, 8, 4, 4, (6, 12), (6, 12), "equiangular", "equiangular", 1e-5, 1e-3],
            [4, 1, 1, 1, (2, 4), (2, 4), "equiangular", "equiangular", 1e-5, 1e-3],
            [4, 1, 4, 1, (2, 4), (2, 4), "equiangular", "equiangular", 1e-5, 1e-3],
            [4, 4, 4, 4, (6, 12), (6, 12), "legendre-gauss", "legendre-gauss", 1e-5, 1e-3],
            [4, 4, 4, 1, (6, 12), (6, 12), "lobatto", "lobatto", 1e-5, 1e-3],
        ],
        skip_on_empty=True,
    )
    @unittest.skipUnless(optimized_kernels_is_available(), "skipping test because optimized kernels are not available")
    def test_custom_implementation(self, batch_size, channels, channels_out, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol, verbose=False):
        """Tests numerical equivalence between the custom (CUDA) implementation and the reference torch implementation"""

        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

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
        model_ref = NeighborhoodAttentionS2(in_channels=channels, out_channels=channels_out, num_heads=heads, in_shape=in_shape, out_shape=out_shape, grid_in=grid_in, grid_out=grid_out, bias=True, optimized_kernel=False).to(self.device)

        # Device model and inputs
        model_opt = NeighborhoodAttentionS2(in_channels=channels, out_channels=channels_out, num_heads=heads, in_shape=in_shape, out_shape=out_shape, grid_in=grid_in, grid_out=grid_out, bias=True, optimized_kernel=True).to(self.device)

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
            if verbose:
                print(f"input grad comparison for {inp}: grad_ref={grad_ref}, grad_opt={grad_opt}, diff={torch.abs(grad_ref - grad_opt)}")
            self.assertTrue(torch.allclose(grad_opt, grad_ref, atol=atol, rtol=rtol), f"Input gradient mismatch in {inp}")

        # Check parameter gradient equivalence
        for (name_ref, p_ref), (name_opt, p_opt) in zip(model_ref.named_parameters(), model_opt.named_parameters()):
            pgrad_opt = p_opt.grad.cpu()
            pgrad_ref = p_ref.grad.cpu()
            if verbose:
                print(f"parameter grad comparison for {name_ref}: grad_ref={pgrad_ref}, grad_opt={pgrad_opt}, diff={torch.abs(pgrad_ref - pgrad_opt)}")
            self.assertTrue(torch.allclose(pgrad_opt, pgrad_ref, atol=atol, rtol=rtol), f"Parameter gradient mismatch: {name_ref}")

    # caution: multihead-implementation between full and neighborhood attention still seem to differ. tests are only done for single head
    @parameterized.expand(
        [
            # Format: [batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol]
            [2, 64, 1, (25, 48), (25, 48), "equiangular", "equiangular", 5e-2, 1e-4],
        ],
        skip_on_empty=True,
    )
    @unittest.skipUnless(cuda_kernels_is_available(), "skipping test because CUDA kernels are not available")
    def test_device_vs_cpu(self, batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol, verbose=False):
        """Tests numerical equivalence between optimized CUDA and CPU implementations"""

        if (self.device.type == "cpu"):
            # comparing CPU with itself does not make sense
            return

        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        # Helper: create inputs
        inputs_host = {
            "k": torch.randn(batch_size, channels, nlat_in, nlon_in, requires_grad=True, dtype=torch.float32),
            "v": torch.randn(batch_size, channels, nlat_in, nlon_in, requires_grad=True, dtype=torch.float32),
            "q": torch.randn(batch_size, channels, nlat_out, nlon_out, requires_grad=True, dtype=torch.float32),
        }
        inputs_device = {k: v.detach().clone().to(self.device).requires_grad_() for k, v in inputs_host.items()}

        # reference input and model
        att_host = NeighborhoodAttentionS2(
            in_channels=channels, num_heads=heads, in_shape=in_shape, out_shape=out_shape, grid_in=grid_in, grid_out=grid_out, bias=True, theta_cutoff=2 * torch.pi
        )

        # Device model and inputs
        att_device = NeighborhoodAttentionS2(
            in_channels=channels, num_heads=heads, in_shape=in_shape, out_shape=out_shape, grid_in=grid_in, grid_out=grid_out, bias=True, theta_cutoff=2 * torch.pi
        ).to(self.device)

        # Synchronize parameters of model
        att_device.load_state_dict(att_host.state_dict())
        for (name_host, p_host), (name_device, p_device) in zip(att_host.named_parameters(), att_device.named_parameters()):
            p_host_copy = p_host.detach().clone().cpu()
            p_device_copy = p_device.detach().clone().cpu()
            if verbose:
                print(f"weight comparison for {name_host}: host_weight={p_host_copy}, device_weight={p_device_copy}, diff={torch.abs(p_host_copy - p_device_copy)}")
            self.assertTrue(torch.allclose(p_host_copy, p_device_copy))

        # reference forward passes
        out_host = att_host(inputs_host["q"], inputs_host["k"], inputs_host["v"])
        out_device = att_device(inputs_device["q"], inputs_device["k"], inputs_device["v"])

        # Check forward equivalence
        if verbose:
            print(f"output comparison: out_host={out_host}, out_device={out_device}, diff={torch.abs(out_host.cpu() - out_device.cpu())}")
        self.assertTrue(torch.allclose(out_host.cpu(), out_device.cpu(), atol=atol, rtol=rtol), "Forward outputs differ between host and device implementation")

        # Backward passes
        grad = torch.randn_like(out_host)
        out_host.backward(grad)
        out_device.backward(grad.to(self.device))

        for inp in ["q", "k", "v"]:
            igrad_host = inputs_host[inp].grad.cpu()
            igrad_device = inputs_device[inp].grad.cpu()
            if verbose:
                print(f"input grad comparison for {inp}: host_grad={igrad_host}, device_grad={igrad_device}, diff={torch.abs(igrad_host - igrad_device)}")
            self.assertTrue(torch.allclose(igrad_host, igrad_device, atol=atol, rtol=rtol), f"Input gradient mismatch in {inp}")

        # Check parameter gradient equivalence - check only q,k, v weights
        for (name_host, p_host), (name_device, p_device) in zip(att_host.named_parameters(), att_device.named_parameters()):
            grad_host = p_host.grad.cpu()
            grad_device = p_device.grad.cpu()
            if verbose:
                print(f"grad comparison for {name_host}: host_grad={grad_host}, device_grad={grad_device}, diff={torch.abs(grad_host - grad_device)}")
            self.assertTrue(torch.allclose(grad_host, grad_device, atol=atol, rtol=rtol), f"Parameter gradient mismatch: {name_host}")


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
        """Tests numerical equivalence between the global spherical attention module and the neighborhood spherical attention module with the neighborhood set ot the whole sphere"""

        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")

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
            self.assertTrue(torch.allclose(p_ref, p))

        # reference forward passes
        out_ref = model_ref(inputs_ref["q"], inputs_ref["k"], inputs_ref["v"])
        out = model(inputs["q"], inputs["k"], inputs["v"])

        # Check forward equivalence
        self.assertTrue(torch.allclose(out, out_ref, atol=atol, rtol=rtol), "Forward outputs differ between torch reference and custom implementation")

        # Backward passes
        grad = torch.randn_like(out_ref)
        out_ref.backward(grad)
        out.backward(grad.to(self.device))

        # Check input gradient equivalence
        for inp in ["q", "k", "v"]:
            grad_ref = inputs_ref[inp].grad
            grad = inputs[inp].grad
            self.assertTrue(torch.allclose(grad, grad_ref, atol=atol, rtol=rtol), f"Input gradient mismatch in {inp}")

        # Check parameter gradient equivalence - check only q,k, v weights
        for key in ["q_weights", "k_weights", "v_weights"]:
            grad_ref = getattr(model_ref, key).grad
            grad = getattr(model, key).grad
            self.assertTrue(torch.allclose(grad, grad_ref, atol=atol, rtol=rtol), f"Parameter gradient mismatch")


    @parameterized.expand(
        [
            # Format: [batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol]
            [4, 4, 1, (6, 12), (6, 12), "equiangular", "equiangular", 1e-2, 0],
        ],
        skip_on_empty=True,
    )
    def test_optimized_pt2_compatibility(self, batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol, verbose=False):
        """Tests whether the optimized kernels are PyTorch 2 compatible"""

        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping GPU test because CUDA kernels are not available")

        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        att = NeighborhoodAttentionS2(
            in_channels=channels, num_heads=heads, in_shape=in_shape, out_shape=out_shape, grid_in=grid_in, grid_out=grid_out, bias=False, theta_cutoff=2 * torch.pi, optimized_kernel=True
        ).to(self.device)

        inputs = {
            "k": torch.randn(batch_size, channels, nlat_in, nlon_in, requires_grad=True, device=self.device, dtype=torch.float32),
            "v": torch.randn(batch_size, channels, nlat_in, nlon_in, requires_grad=True, device=self.device, dtype=torch.float32),
            "q": torch.randn(batch_size, channels, nlat_out, nlon_out, requires_grad=True, device=self.device, dtype=torch.float32),
        }

        test_inputs = (inputs["k"], inputs["v"], inputs["q"], 
                       att.k_weights, att.v_weights, att.q_weights, 
                       att.k_bias, att.v_bias, att.q_bias, 
                       att.quad_weights, att.psi_col_idx, att.psi_roff_idx, 
                       att.psi_max_nnz, att.num_heads, nlon_in, nlat_out, nlon_out)

        opcheck(torch.ops.attention_kernels._neighborhood_s2_attention_optimized, test_inputs)
        # opcheck(torch.ops.attention_kernels._neighborhood_s2_attention_optimized, test_inputs, test_utils="test_schema")
        # opcheck(torch.ops.attention_kernels._neighborhood_s2_attention_optimized, test_inputs, test_utils="test_faketensor")
        #opcheck(torch.ops.attention_kernels._neighborhood_s2_attention_optimized, test_inputs, test_utils="test_aot_dispatch_dynamic")


    @parameterized.expand(
        [
            # self attention
            [1, 256, 1, (91, 180), (91, 180), "equiangular", "equiangular"],
        ],
        skip_on_empty=True,
    )
    @unittest.skipUnless(optimized_kernels_is_available() and _run_perf_tests, "skipping performance test because optimized kernels are not available or perf tests are disabled")
    def test_perf(self, batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, verbose=False):
        
        if (self.device.type == "cuda") and (not cuda_kernels_is_available()):
            raise unittest.SkipTest("skipping test because CUDA kernels are not available")
        
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

if __name__ == "__main__":
    unittest.main()
