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


# this routine is only supposed to be used in this test, since it is numerically not stable but supports
# autograd which some of the better kernels do not
def _neighborhood_attention_s2_torch_test(
    kx: torch.Tensor, vx: torch.Tensor, qy: torch.Tensor, quad_weights: torch.Tensor, col_idx: torch.Tensor, row_idx: torch.Tensor, nlon_in: int, nlat_out: int, nlon_out: int
):

    out = torch.zeros_like(qy)

    for ho in range(nlat_out):

        # get nonzero indices in output row
        idx_ho = col_idx[row_idx == ho]

        for wo in range(nlon_out):
            alpha_sum = torch.zeros((out.shape[0],), dtype=out.dtype, device=out.device)
            alpha = torch.zeros((out.shape[0], len(idx_ho)), dtype=out.dtype, device=out.device)
            for inz, nz_col_idx in enumerate(idx_ho):
                # compute input indices from psi datastructure
                hi = nz_col_idx // nlon_in
                # account for output shift and ensure positive index due to circular condition
                wi = nz_col_idx % nlon_in
                wip = (wi + wo) % nlon_in

                # compute correlation & softmax numerator
                q_ho_wo = qy[:, :, ho, wo]
                k_hi_wip = kx[:, :, hi, wip]
                alpha[:, inz] = torch.exp(torch.sum(q_ho_wo * k_hi_wip, dim=1)) * quad_weights[hi]
                # softmax denominator
                alpha_sum[:] = alpha_sum[:] + alpha[:, inz]

            for inz, nz_col_idx in enumerate(idx_ho):
                # compute input indices from psi datastructure
                hi = nz_col_idx // nlon_in
                # account for output shift and ensure positive index due to circular condition
                wi = nz_col_idx % nlon_in
                wip = (wi + wo) % nlon_in

                # compute matmul of attention matrix with V-vector
                out[:, :, ho, wo] = out[:, :, ho, wo] + (alpha[:, None, inz] / alpha_sum[:, None]) * vx[:, :, hi, wip]

    return out


class TestNeighborhoodAttention(unittest.TestCase):

    def setUp(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device.index)
            torch.cuda.manual_seed(333)
        else:
            self.device = torch.device("cpu")

        torch.manual_seed(333)

    @parameterized.expand(
        [
            # regular convolution
            [8, 4, 6, (17, 32), 1e-6, 1e-5],
        ]
    )
    def test_batched_linear(self, batch_size, in_channels, out_channels, shape, atol, rtol):
        # weight
        weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, 1, 1, dtype=torch.float32, device=self.device))
        bias = torch.nn.Parameter(torch.randn(out_channels, dtype=torch.float32, device=self.device))

        # input
        inp = torch.randn(batch_size, in_channels, *shape, dtype=torch.float32, device=self.device)
        inp.requires_grad = True

        # operation
        out = torch.nn.functional.conv2d(inp, weight=weight, bias=bias)
        out_grad = torch.randn(batch_size, out_channels, *shape, dtype=torch.float32, device=self.device)
        out.backward(out_grad)

        # store for comparison
        wgrad = weight.grad.clone()
        bgrad = bias.grad.clone()
        igrad = inp.grad.clone()

        # explicit layers
        igrad_explicit = torch.nn.functional.conv2d(out_grad, weight=weight.permute([1, 0, 2, 3]), bias=None)
        wgrad_explicit = torch.einsum("bchw,bfhw->cf", out_grad, inp).reshape(out_channels, in_channels, 1, 1).contiguous()
        bgrad_explicit = torch.sum(out_grad, dim=(0, 2, 3))

        # check consistency
        self.assertTrue(torch.allclose(igrad, igrad_explicit, atol=atol, rtol=rtol))
        self.assertTrue(torch.allclose(wgrad, wgrad_explicit, atol=atol, rtol=rtol))
        self.assertTrue(torch.allclose(bgrad, bgrad_explicit, atol=atol, rtol=rtol))

    @parameterized.expand(
        [
            # self attention
            [8, 4, 1, (17, 32), (17, 32), "equiangular", "equiangular", 1e-6, 1e-4],
        ]
    )
    def test_fwd(self, batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol):

        # extract some parameters
        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        # set up neighbor matrix
        att = NeighborhoodAttentionS2(in_channels=channels, num_heads=heads,
                                      in_shape=in_shape, out_shape=out_shape,
                                      grid_in=grid_in, grid_out=grid_out, bias=False).to(self.device)

        # Execute and compare
        k_inp = torch.randn(batch_size, channels, nlat_in, nlon_in, dtype=torch.float32, device=self.device)
        k_inp.requires_grad = False
        v_inp = torch.randn(batch_size, channels, nlat_in, nlon_in, dtype=torch.float32, device=self.device)
        v_inp.requires_grad = False
        q_inp = torch.randn(batch_size, channels, nlat_out, nlon_out, dtype=torch.float32, device=self.device)
        q_inp.requires_grad = False

        out_torch = _neighborhood_attention_s2_torch_test(k_inp, v_inp, q_inp, att.quad_weights, att.psi_col_idx, att.psi_row_idx, nlon_in, nlat_out, nlon_out)

        with torch.no_grad():
            out_torch_explicit = _neighborhood_attention_s2_fwd_torch(k_inp, v_inp, q_inp, att.quad_weights, att.psi_col_idx, att.psi_roff_idx, nlon_in, nlat_out, nlon_out)

            self.assertTrue(torch.allclose(out_torch_explicit, out_torch, atol=atol, rtol=rtol))

        if _cuda_extension_available:

            out_cuda = attention_cuda_extension.forward(k_inp, v_inp, q_inp, att.quad_weights, att.psi_col_idx, att.psi_roff_idx, nlon_in, nlat_out, nlon_out)

            self.assertTrue(torch.allclose(out_torch, out_cuda, atol=atol, rtol=rtol))

    @parameterized.expand(
        [
            # regular convolution
            [8, 4, 1, (17, 32), (17, 32), "equiangular", "equiangular", 1e-6, 1e-4],
        ]
    )
    def test_bwd_dv(self, batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol):

        # extract some parameters
        _, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        # set up neighbor matrix
        att = NeighborhoodAttentionS2(in_channels=channels, num_heads=heads,
                                      in_shape=in_shape, out_shape=out_shape,
                                      grid_in=grid_in, grid_out=grid_out, bias=False).to(self.device)

        # Execute and compare
        k_inp = torch.randn(batch_size, channels, *in_shape, dtype=torch.float32, device=self.device)
        k_inp.requires_grad = False
        v_inp = torch.randn(batch_size, channels, *in_shape, dtype=torch.float32, device=self.device)
        v_inp.requires_grad = True
        q_inp = torch.randn(batch_size, channels, *out_shape, dtype=torch.float32, device=self.device)
        q_inp.requires_grad = False
        out_grad = torch.randn(batch_size, channels, *out_shape, dtype=torch.float32, device=self.device)

        out_torch = _neighborhood_attention_s2_torch_test(k_inp, v_inp, q_inp, att.quad_weights, att.psi_col_idx, att.psi_row_idx, nlon_in, nlat_out, nlon_out)

        # need 'retain_graph' to avoid an error in the tests after this one
        out_torch.backward(out_grad)
        dv_inp_torch = v_inp.grad.clone()

        with torch.no_grad():
            dv_inp_torch_explicit = _neighborhood_attention_s2_bwd_dv_torch(
                k_inp, v_inp, q_inp, out_grad, att.quad_weights, att.psi_col_idx, att.psi_roff_idx, nlon_in, nlat_out, nlon_out
            )

        self.assertTrue(torch.allclose(dv_inp_torch_explicit, dv_inp_torch, atol=atol, rtol=rtol))

        if _cuda_extension_available:

            dv_inp_cuda_explicit = attention_cuda_extension.backward_dv(
                k_inp, v_inp, q_inp, out_grad, att.quad_weights, att.psi_col_idx, att.psi_roff_idx, nlon_in, nlat_out, nlon_out
            )

            self.assertTrue(torch.allclose(dv_inp_cuda_explicit, dv_inp_torch, atol=atol, rtol=rtol))

    @parameterized.expand(
        [
            # regular convolution
            [8, 4, 1, (17, 32), (17, 32), "equiangular", "equiangular", 1e-6, 1e-3],
        ]
    )
    def test_bwd_dk(self, batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol):

        # extract some parameters
        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        # set up neighbor matrix
        att = NeighborhoodAttentionS2(in_channels=channels, num_heads=heads,
                                      in_shape=in_shape, out_shape=out_shape,
                                      grid_in=grid_in, grid_out=grid_out, bias=False).to(self.device)

        # Execute and compare
        k_inp = torch.randn(batch_size, channels, nlat_in, nlon_in, dtype=torch.float32, device=self.device)
        k_inp.requires_grad = True
        v_inp = torch.randn(batch_size, channels, nlat_in, nlon_in, dtype=torch.float32, device=self.device)
        v_inp.requires_grad = False
        q_inp = torch.randn(batch_size, channels, nlat_out, nlon_out, dtype=torch.float32, device=self.device)
        q_inp.requires_grad = False
        out_grad = torch.randn(batch_size, channels, nlat_out, nlon_out, dtype=torch.float32, device=self.device)

        out_torch = _neighborhood_attention_s2_torch_test(k_inp, v_inp, q_inp, att.quad_weights, att.psi_col_idx, att.psi_row_idx, nlon_in, nlat_out, nlon_out)

        # need 'retain_graph' to avoid an error in the tests after this one
        out_torch.backward(out_grad)
        dk_inp_torch = k_inp.grad.clone()

        with torch.no_grad():
            dk_inp_torch_explicit = _neighborhood_attention_s2_bwd_dk_torch(
                k_inp, v_inp, q_inp, out_grad, att.quad_weights, att.psi_col_idx, att.psi_roff_idx, nlon_in, nlat_out, nlon_out
            )

            self.assertTrue(torch.allclose(dk_inp_torch_explicit, dk_inp_torch, atol=atol, rtol=rtol))

        if _cuda_extension_available:

            dk_inp_cuda_explicit = attention_cuda_extension.backward_dk(
                k_inp, v_inp, q_inp, out_grad, att.quad_weights, att.psi_col_idx, att.psi_roff_idx, nlon_in, nlat_out, nlon_out
            )

            self.assertTrue(torch.allclose(dk_inp_cuda_explicit, dk_inp_torch, atol=atol, rtol=rtol))

    @parameterized.expand(
        [
            # regular convolution
            [8, 4, 1, (17, 32), (17, 32), "equiangular", "equiangular", 4e-6, 1e-3],
        ]
    )
    def test_bwd_dq(self, batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol):

        # extract some parameters
        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        # set up neighbor matrix
        att = NeighborhoodAttentionS2(in_channels=channels, num_heads=heads,
                                      in_shape=in_shape, out_shape=out_shape,
                                      grid_in=grid_in, grid_out=grid_out, bias=False).to(self.device)

        # Execute and compare
        k_inp = torch.randn(batch_size, channels, nlat_in, nlon_in, dtype=torch.float32, device=self.device)
        k_inp.requires_grad = False
        v_inp = torch.randn(batch_size, channels, nlat_in, nlon_in, dtype=torch.float32, device=self.device)
        v_inp.requires_grad = False
        q_inp = torch.randn(batch_size, channels, nlat_out, nlon_out, dtype=torch.float32, device=self.device)
        q_inp.requires_grad = True
        out_grad = torch.randn(batch_size, channels, nlat_out, nlon_out, dtype=torch.float32, device=self.device)

        out_torch = _neighborhood_attention_s2_torch_test(k_inp, v_inp, q_inp, att.quad_weights, att.psi_col_idx, att.psi_row_idx, nlon_in, nlat_out, nlon_out)

        # need 'retain_graph' to avoid an error in the tests after this one
        out_torch.backward(out_grad)
        dq_inp_torch = q_inp.grad.clone()

        with torch.no_grad():
            dq_inp_torch_explicit = _neighborhood_attention_s2_bwd_dq_torch(
                k_inp, v_inp, q_inp, out_grad, att.quad_weights, att.psi_col_idx, att.psi_roff_idx, nlon_in, nlat_out, nlon_out
            )

            self.assertTrue(torch.allclose(dq_inp_torch_explicit, dq_inp_torch, atol=atol, rtol=rtol))

        if _cuda_extension_available:

            dq_inp_cuda_explicit = attention_cuda_extension.backward_dq(
                k_inp, v_inp, q_inp, out_grad, att.quad_weights, att.psi_col_idx, att.psi_roff_idx, nlon_in, nlat_out, nlon_out
            )

            self.assertTrue(torch.allclose(dq_inp_cuda_explicit, dq_inp_torch, atol=atol, rtol=rtol))

    @parameterized.expand(
        [
            # self attention
            [1, 73, 1, (721, 1440), (721, 1440), "equiangular", "equiangular", 1e-5, 1e-5],
        ]
    )
    def test_big(self, batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol):

        # this test only makes sense when CUDA version is available
        if torch.cuda.is_available():
            if not _cuda_extension_available:
                print("WARNING: Problem loading CUDA attention module")
                return
        # extract some parameters
        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        # TODO: this test seems hardcoded for GPU. Is this necessary?
        k_gpu = torch.randn(batch_size, channels, nlat_in, nlon_in, dtype=torch.float32, device="cuda:0")
        k_gpu.requires_grad = False
        v_gpu = torch.randn(batch_size, channels, nlat_in, nlon_in, dtype=torch.float32, device="cuda:0")
        v_gpu.requires_grad = False
        q_gpu = torch.randn(batch_size, channels, nlat_out, nlon_out, dtype=torch.float32, device="cuda:0")
        q_gpu.requires_grad = False

        # set up layers
        att_gpu = NeighborhoodAttentionS2(in_channels=channels, num_heads=heads,
                                          in_shape=in_shape, out_shape=out_shape,
                                          grid_in=grid_in, grid_out=grid_out, bias=True).to("cuda:0")

        # random weights
        with torch.no_grad():
            att_gpu.q_weights.normal_()
            att_gpu.k_weights.normal_()
            att_gpu.v_weights.normal_()
            att_gpu.q_bias.normal_()
            att_gpu.k_bias.normal_()
            att_gpu.v_bias.normal_()

            out_gpu = att_gpu(q_gpu, k_gpu, v_gpu)

        # sync weights:
        with torch.no_grad():
            att_gpu.q_weights.copy_(att_gpu.q_weights)
            att_gpu.k_weights.copy_(att_gpu.k_weights)
            att_gpu.v_weights.copy_(att_gpu.v_weights)
            att_gpu.q_bias.copy_(att_gpu.q_bias)
            att_gpu.k_bias.copy_(att_gpu.k_bias)
            att_gpu.v_bias.copy_(att_gpu.v_bias)

        q_gpu = q_gpu.detach().clone().to(self.device)
        q_gpu.requires_grad = True
        k_gpu = k_gpu.detach().clone().to(self.device)
        k_gpu.requires_grad = True
        v_gpu = v_gpu.detach().clone().to(self.device)
        v_gpu.requires_grad = True

        out_gpu = att_gpu(q_gpu, k_gpu, v_gpu)
        out_grad = torch.randn(out_gpu.shape, dtype=torch.float32, device="cuda:0")
        out_gpu.backward(out_grad.to("cuda:0"))

    @parameterized.expand(
        [
            # self attention
            [10, 2, 1, (17, 32), (17, 32), "equiangular", "equiangular", 1e-5, 1e-5],
        ]
    )
    def test_neighborhood(self, batch_size, num_channels, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol):
        """
        This test sets a specific q[ho,wo] value to 1.0 (elsewhere 0), and then a neighborhood of k around ho,wo to 1.0 (else 0.0). Also vi is set to a sinusoidal input. We also run it with fully 0 q,k. We test that the output of the nonzero q,k is only different to the zero q,k in a single point. We also check the value of this difference (as a regression test).
        """

        # extract some parameters
        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape
        from torch_harmonics import _neighborhood_attention_s2_fwd_torch

        device = "cpu"
        nas2_2 = NeighborhoodAttentionS2(in_channels=num_channels, num_heads=heads,
                                         in_shape=(nlat_in, nlon_in), out_shape=(nlat_out, nlon_out),
                                         theta_cutoff=torch.pi / 128 * 10)
        nas2_2.to(device)
        qo = torch.zeros((batch_size, num_channels, nlat_in, nlon_in)).to(device)
        x = torch.linspace(0, 2 * np.pi, nlat_in)  # 100 points in x direction
        y = torch.linspace(0, 2 * np.pi, nlon_in)  # 100 points in y direction
        # Create a meshgrid
        X, Y = torch.meshgrid(x, y, indexing="ij")
        vi = torch.ones((batch_size, num_channels, nlat_in, nlon_in)).to(device)
        vi[:, :, :, :] = (torch.sin(X) + torch.sin(Y))[None, None, :, :]

        ki = torch.zeros((batch_size, num_channels, nlat_in, nlon_in)).to(device)
        ki2 = torch.zeros((batch_size, num_channels, nlat_in, nlon_in)).to(device)

        ho = 10
        wo = 15
        qo[:, 0, ho, wo] = 1.0
        nas3 = NeighborhoodAttentionS2(in_channels=num_channels, num_heads=heads,
                                       in_shape=(nlat_in, nlon_in), out_shape=(nlat_out, nlon_out),
                                       theta_cutoff=torch.pi / 128 * 7)
        zstart = nas3.psi_roff_idx[ho]
        zend = nas3.psi_roff_idx[ho + 1]

        # set a small neighborhood of k around (ho,wo) to 1
        for idz in range(zstart, zend):
            nz_col_idx = nas3.psi_col_idx[idz]
            # compute input indices from psi datastructure
            hi = nz_col_idx // nlon_in
            # account for output shift and ensure positive index due to circular condition
            wi = nz_col_idx % nlon_in
            wip = (wi + wo) % nlon_in
            ki2[:, 0, hi, wip] = 1.0

        # run with k zero
        y = _neighborhood_attention_s2_fwd_torch(ki, vi, qo, nas2_2.quad_weights, nas2_2.psi_col_idx, nas2_2.psi_roff_idx, nlon_in, nlat_out, nlon_out)
        # run with k 1 at neighborhood of ho,wo
        y2 = _neighborhood_attention_s2_fwd_torch(ki2, vi, qo, nas2_2.quad_weights, nas2_2.psi_col_idx, nas2_2.psi_roff_idx, nlon_in, nlat_out, nlon_out)

        # for viz if desired
        # plt.matshow((y[0,0,:,:]-y2[0,0,:,:]).detach().cpu())#, vmin=0, vmax=2.0)
        # plt.matshow((y2[0,0,:,:]).detach().cpu())#, vmin=0, vmax=2.0)

        # compare zero k vs. nonzero k and ensure difference only occurs at ho,wo
        nz_x, nz_y = torch.where((y[0, 0, :, :] - y2[0, 0, :, :]).abs() > 0)
        self.assertTrue(nz_x.item() == ho)
        self.assertTrue(nz_y.item() == wo)
        h, w = nz_x.item(), nz_y.item()
        diff_hw = y[0, 0, h, w] - y2[0, 0, h, w]
        # print("diff_hw=", diff_hw.item())

        # regression test the difference. Unfortunately difficult to come up with an
        # analytical value, so we just have it hardcoded.
        self.assertTrue(torch.allclose(diff_hw, torch.tensor([0.00753], device=device), rtol=rtol, atol=atol))

    @parameterized.expand(
        [
            # self attention
            [8, 4, 1, (17, 32), (17, 32), "equiangular", "equiangular", 2e-4, 1e-5],
            [8, 4, 2, (17, 32), (17, 32), "equiangular", "equiangular", 2e-4, 1e-5],
        ]
    )
    def test_full(self, batch_size, channels, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol):

        # extract some parameters
        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        k_cpu = torch.randn(batch_size, channels, nlat_in, nlon_in, dtype=torch.float32, device="cpu")
        k_cpu.requires_grad = True
        v_cpu = torch.randn(batch_size, channels, nlat_in, nlon_in, dtype=torch.float32, device="cpu")
        v_cpu.requires_grad = True
        q_cpu = torch.randn(batch_size, channels, nlat_out, nlon_out, dtype=torch.float32, device="cpu")
        q_cpu.requires_grad = True

        # set up layers
        att_cpu = NeighborhoodAttentionS2(in_channels=channels, num_heads=heads,
                                          in_shape=in_shape, out_shape=out_shape,
                                          grid_in=grid_in, grid_out=grid_out, bias=True)

        # random weights
        with torch.no_grad():
            att_cpu.q_weights.normal_()
            att_cpu.k_weights.normal_()
            att_cpu.v_weights.normal_()
            att_cpu.q_bias.normal_()
            att_cpu.k_bias.normal_()
            att_cpu.v_bias.normal_()

        out_cpu = att_cpu(q_cpu, k_cpu, v_cpu)
        out_grad = torch.randn(out_cpu.shape, dtype=torch.float32, device="cpu")
        out_cpu.backward(out_grad)

        att_gpu = NeighborhoodAttentionS2(in_channels=channels, num_heads=heads,
                                          in_shape=in_shape, out_shape=out_shape,
                                          grid_in=grid_in, grid_out=grid_out, bias=True).to(self.device)

        # sync weights:
        with torch.no_grad():
            att_gpu.q_weights.copy_(att_cpu.q_weights)
            att_gpu.k_weights.copy_(att_cpu.k_weights)
            att_gpu.v_weights.copy_(att_cpu.v_weights)
            att_gpu.proj_weights.copy_(att_cpu.proj_weights)
            att_gpu.q_bias.copy_(att_cpu.q_bias)
            att_gpu.k_bias.copy_(att_cpu.k_bias)
            att_gpu.v_bias.copy_(att_cpu.v_bias)
            att_gpu.proj_bias.copy_(att_cpu.proj_bias)
            

        q_gpu = q_cpu.detach().clone().to(self.device)
        q_gpu.requires_grad = True
        k_gpu = k_cpu.detach().clone().to(self.device)
        k_gpu.requires_grad = True
        v_gpu = v_cpu.detach().clone().to(self.device)
        v_gpu.requires_grad = True

        out_gpu = att_gpu(q_gpu, k_gpu, v_gpu)
        out_gpu.backward(out_grad.to(self.device))

        # check forward
        self.assertTrue(torch.allclose(out_cpu.to(self.device), out_gpu, atol=atol, rtol=rtol))

        # check input gradients:
        self.assertTrue(torch.allclose(q_cpu.grad.to(self.device), q_gpu.grad, atol=atol, rtol=rtol))
        self.assertTrue(torch.allclose(k_cpu.grad.to(self.device), k_gpu.grad, atol=atol, rtol=rtol))
        self.assertTrue(torch.allclose(v_cpu.grad.to(self.device), v_gpu.grad, atol=atol, rtol=rtol))

        # check weight gradients
        self.assertTrue(torch.allclose(att_cpu.q_weights.grad.to(self.device), att_gpu.q_weights.grad, atol=atol, rtol=rtol))
        self.assertTrue(torch.allclose(att_cpu.k_weights.grad.to(self.device), att_gpu.k_weights.grad, atol=atol, rtol=rtol))
        self.assertTrue(torch.allclose(att_cpu.v_weights.grad.to(self.device), att_gpu.v_weights.grad, atol=atol, rtol=rtol))
        self.assertTrue(torch.allclose(att_cpu.proj_weights.grad.to(self.device), att_gpu.proj_weights.grad, atol=atol, rtol=rtol))

        # check bias gradients
        self.assertTrue(torch.allclose(att_cpu.q_bias.grad.to(self.device), att_gpu.q_bias.grad, atol=atol, rtol=rtol))
        self.assertTrue(torch.allclose(att_cpu.k_bias.grad.to(self.device), att_gpu.k_bias.grad, atol=atol, rtol=rtol))
        self.assertTrue(torch.allclose(att_cpu.v_bias.grad.to(self.device), att_gpu.v_bias.grad, atol=atol, rtol=rtol))
        self.assertTrue(torch.allclose(att_cpu.proj_bias.grad.to(self.device), att_gpu.proj_bias.grad, atol=atol, rtol=rtol))

    @parameterized.expand(
        [
            # self attention
            [8, 8, 8, 2, (17, 32), (17, 32), "equiangular", "equiangular", 2e-4, 1e-5],
            [8, 8, 8, 2, (17, 32), (17, 32), "legendre-gauss", "legendre-gauss", 2e-4, 1e-5],
            [8, 8, 8, 2, (17, 32), (17, 32), "lobatto", "lobatto", 2e-4, 1e-5],
            [8, 8, 4, 2, (17, 32), (17, 32), "equiangular", "equiangular", 2e-4, 1e-5],
        ]
    )
    def test_full_attention(self, batch_size, channels_in, channels_out, heads, in_shape, out_shape, grid_in, grid_out, atol, rtol):
        # extract some parameters
        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        k_cpu = torch.randn(batch_size, channels_in, nlat_in, nlon_in, dtype=torch.float32, device="cpu")
        k_cpu.requires_grad = True
        v_cpu = torch.randn(batch_size, channels_in, nlat_in, nlon_in, dtype=torch.float32, device="cpu")
        v_cpu.requires_grad = True
        q_cpu = torch.randn(batch_size, channels_in, nlat_out, nlon_out, dtype=torch.float32, device="cpu")
        q_cpu.requires_grad = True

        att_cpu = AttentionS2(in_channels=channels_in, out_channels=channels_out, num_heads=heads, in_shape=in_shape, out_shape=out_shape, grid_in=grid_in, grid_out=grid_out, bias=True)

        out = att_cpu(q_cpu, k_cpu, v_cpu)

        # check if output is sane
        self.assertFalse(torch.isnan(out).any())
        

if __name__ == "__main__":
    unittest.main()
