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
from parameterized import parameterized

# import math
import numpy as np
import torch
import torch.nn as nn

# from torch.autograd import gradcheck
from torch_harmonics import AttentionS2, NeighborhoodAttentionS2

from torch_harmonics._neighborhood_attention import (
    _neighborhood_attention_s2_torch,
    _neighborhood_attention_s2_fwd_torch,
    _neighborhood_attention_s2_bwd_dv_torch,
    _neighborhood_attention_s2_bwd_dk_torch,
    _neighborhood_attention_s2_bwd_dq_torch,
)

# import custom C++/CUDA extensions
try:
    import attention_cuda_extension

    _cuda_extension_available = True
except ImportError as err:
    print(f"Warning: Couldn't Import cuda attention: {err}")
    attention_cuda_extension = None
    _cuda_extension_available = False


class TestNeighborhoodAttentionS2(unittest.TestCase):
    def setUp(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device.index)
            torch.cuda.manual_seed(333)
            torch.manual_seed(333)
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(333)

    @parameterized.expand(
        [
            # Format: [batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol]
            [4, 4, 1, (6, 12), (6, 12), "equiangular", "equiangular", 1e-5, 1e-3],
            [4, 4, 2, (6, 12), (6, 12), "equiangular", "equiangular", 1e-5, 1e-3],
            [4, 4, 4, (6, 12), (6, 12), "equiangular", "equiangular", 1e-5, 1e-3],
            [4, 4, 1, (6, 12), (6, 12), "legendre-gauss", "legendre-gauss", 1e-5, 1e-3],
            [4, 4, 1, (6, 12), (6, 12), "lobatto", "lobatto", 1e-5, 1e-3],
        ]
    )
    def test_custom_implementation(self, batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol, verbose=True):
        """Tests numerical equivalence between the custom (CUDA) implementation and the reference torch implementation"""

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
        model_ref = NeighborhoodAttentionS2(in_channels=channels, num_heads=heads, in_shape=in_shape, out_shape=out_shape, grid_in=grid_in, grid_out=grid_out, bias=True).to(
            self.device
        )

        # Device model and inputs
        model = NeighborhoodAttentionS2(in_channels=channels, num_heads=heads, in_shape=in_shape, out_shape=out_shape, grid_in=grid_in, grid_out=grid_out, bias=True)

        # Synchronize parameters of model
        model.load_state_dict(model_ref.state_dict())
        model = model.to(self.device)
        for (name_ref, p_ref), (name, p) in zip(model_ref.named_parameters(), model.named_parameters()):
            assert torch.allclose(p_ref, p), f"Parameter mismatch: {name_ref} vs {name}"

        # reference forward passes
        out_ref = _neighborhood_attention_s2_torch(
            inputs_ref["k"],
            inputs_ref["v"],
            inputs_ref["q"] * model_ref.scale,
            model_ref.k_weights,
            model_ref.v_weights,
            model_ref.q_weights,
            model_ref.k_bias,
            model_ref.v_bias,
            model_ref.q_bias,
            model_ref.quad_weights,
            model_ref.psi_col_idx,
            model_ref.psi_roff_idx,
            model_ref.num_heads,
            model_ref.nlon_in,
            model_ref.nlat_out,
            model_ref.nlon_out,
        )
        out_ref = nn.functional.conv2d(out_ref, model_ref.proj_weights, bias=model_ref.proj_bias)
        out = model(inputs["q"], inputs["k"], inputs["v"])

        # Check forward equivalence
        self.assertTrue(torch.allclose(out, out_ref, atol=atol, rtol=rtol), "Forward outputs differ between torch reference and custom implementation")

        # Backward passes
        grad = torch.randn_like(out_ref)
        out_ref.backward(grad)
        out.backward(grad.to(self.device))

        # Check input gradient equivalence
        for inp in ["q", "k", "v"]:
            grad_ref = inputs_ref[inp].grad.cpu()
            grad = inputs[inp].grad.cpu()
            self.assertTrue(torch.allclose(grad, grad_ref, atol=atol, rtol=rtol), f"Input gradient mismatch in {inp}")

        # Check parameter gradient equivalence
        for p_ref, p in zip(model_ref.parameters(), model.parameters()):
            self.assertTrue(torch.allclose(p.grad, p_ref.grad, atol=atol, rtol=rtol), f"Parameter gradient mismatch: {type(p_ref).__name__}")

    # caution: multihead-implementation between full and neighborhood attention still seem to differ. tests are only done for single head
    @parameterized.expand(
        [
            # Format: [batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol]
            [4, 4, 1, (6, 12), (6, 12), "equiangular", "equiangular", 1e-2, 0],
            # [4, 4, 2, (6, 12), (6, 12), "equiangular", "equiangular", 1e-5, 1e-3],
            # [4, 4, 4, (6, 12), (6, 12), "equiangular", "equiangular", 1e-5, 1e-3],
            [4, 4, 1, (6, 12), (6, 12), "legendre-gauss", "legendre-gauss", 1e-2, 0],
            [4, 4, 1, (6, 12), (6, 12), "lobatto", "lobatto", 1e-2, 0],
        ]
    )
    def test_neighborhood_global_equivalence(self, batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol, verbose=True):
        """Tests numerical equivalence between the global spherical attention module and the neighborhood spherical attention module with the neighborhood set ot the whole sphere"""

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
        model_ref = AttentionS2(in_channels=channels, num_heads=heads, in_shape=in_shape, out_shape=out_shape, grid_in=grid_in, grid_out=grid_out, bias=False).to(self.device)

        # Device model and inputs
        model = NeighborhoodAttentionS2(
            in_channels=channels, num_heads=heads, in_shape=in_shape, out_shape=out_shape, grid_in=grid_in, grid_out=grid_out, bias=False, theta_cutoff=2 * torch.pi
        )

        # Synchronize parameters of model
        model.load_state_dict(model_ref.state_dict())
        model = model.to(self.device)
        for (name_ref, p_ref), (name, p) in zip(model_ref.named_parameters(), model.named_parameters()):
            assert torch.allclose(p_ref, p), f"Parameter mismatch: {name_ref} vs {name}"

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


if __name__ == "__main__":
    unittest.main()
