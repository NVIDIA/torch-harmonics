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
import math
import numpy as np
import torch
from torch.autograd import gradcheck
from torch_harmonics import *

import importlib

import torch_harmonics
from torch_harmonics.attention import _disco_att_v3_torch, _disco_att_bwd_dv_torch,\
    _disco_att_bwd_dk_torch, _disco_att_bwd_dq_torch
from torch_harmonics import NeighborhoodAttentionS2
import attention_cuda_extension as att_cuda
import matplotlib.pyplot as plt
import ipywidgets as widgets
import torch.nn.functional as F
import numpy as np

# a simple torch fwd implementation used to test gradient
def disco_att_torch(kx: torch.Tensor, vx: torch.Tensor, qy: torch.Tensor,
                    quad_weights: torch.Tensor, col_idx: torch.Tensor, row_idx: torch.Tensor, 
                    nlat_in: int, nlon_in: int, nlat_out: int, nlon_out: int):

    out = torch.zeros_like(qy)

    for ho in range(nlat_out):

        # get nonzero indices in output row
        idx_ho = col_idx[row_idx==ho]

        for wo in range(nlon_out):
            alpha_sum = torch.zeros((out.shape[0],), dtype=out.dtype, device=out.device)
            alpha = torch.zeros((out.shape[0], len(idx_ho)), dtype=out.dtype, device=out.device)
            for (inz, nz_col_idx) in enumerate(idx_ho):
                # compute input indices from psi datastructure
                hi = nz_col_idx // nlon_in
                # account for output shift and ensure positive index due to circular condition
                wi = nz_col_idx % nlon_in
                wip = (wi+wo) % nlon_in

                # compute correlation & softmax numerator
                q_ho_wo = qy[:, :, ho, wo]
                k_hi_wi = kx[:, :, hi, wip]
                # k_hi_wi = kx[:, :, hi, wi]
                alpha[:, inz] = torch.exp(torch.sum(q_ho_wo * k_hi_wi, dim=1))
                # softmax denominator
                alpha_sum[:] = alpha_sum[:] + alpha[:, inz]

            for (inz, nz_col_idx) in enumerate(idx_ho):
                # compute input indices from psi datastructure
                hi = nz_col_idx // nlon_in
                # account for output shift and ensure positive index due to circular condition
                wi = nz_col_idx % nlon_in
                wip = (wi + wo) % nlon_in

                # compute matmul of attention matrix with V-vector
                out[:,:,ho,wo] = out[:,:,ho,wo] + (alpha[:, inz]/alpha_sum[:])[:,None] * vx[:,:,hi,wip] * quad_weights[hi]

    return out


def test_dvx(k_inp_d, v_inp_d, q_inp_d,
             quad_weights_d, col_idx_d, row_off, row_idx_d, max_psi_nnz,
             nlat_in, nlon_in, nlat_out, nlon_out,
             grad_d):
    # Execute and compare
    k_inp_d = k_inp_d.detach()
    k_inp_d.requires_grad = False
    v_inp_d = v_inp_d.detach()
    v_inp_d.requires_grad = True
    q_inp_d = q_inp_d.detach()
    q_inp_d.requires_grad = False
    out_torch_v3_d = disco_att_torch(k_inp_d, v_inp_d, q_inp_d,
                                     quad_weights_d, col_idx_d, row_idx_d,
                                     nlat_in, nlon_in, nlat_out, nlon_out)

    # need 'retain_graph' to avoid an error in the tests after this one
    out_torch_v3_d.backward(grad_d, retain_graph=True)
    dv_inp_torch_v3_d = v_inp_d.grad.clone()


    with torch.no_grad():
        dv_inp_torch_explicit_d = _disco_att_bwd_dv_torch(k_inp_d, v_inp_d, q_inp_d, grad_d,
                                                          quad_weights_d, col_idx_d, row_off,
                                                          nlat_in, nlon_out, nlat_out, nlon_out)

    # print("dvx:")
    # print("by hand:   ", dv_inp_torch_explicit_d[0,0,:3,:3], "\n via torch: ", dv_inp_torch_v3_d[0,0,:3,:3])
    assert torch.allclose(dv_inp_torch_explicit_d, dv_inp_torch_v3_d)

    dydv_cuda = torch.zeros_like(grad_d)
    att_cuda.s2_attention_bwd_dv_cuda(k_inp_d, v_inp_d, q_inp_d, quad_weights_d, col_idx_d, row_off, max_psi_nnz, grad_d, nlon_in, nlat_out, nlon_out, dydv_cuda)
    # print("dydv_cuda:\n", dydv_cuda[0,0,:3,:3], "\n vs torch autograd: \n", dv_inp_torch_v3_d[0,0,:3,:3], "\n vs torch hand-rolled: \n", dv_inp_torch_explicit_d[0,0,:3,:3])
    assert torch.allclose(dydv_cuda, dv_inp_torch_v3_d)



def test_dkx(k_inp_d, v_inp_d, q_inp_d,
             quad_weights_d, col_idx_d, row_off,
             row_idx, max_psi_nnz,
             nlat_in, nlon_in, nlat_out, nlon_out,
             grad_d):

    # Execute and compare
    torch.autograd.set_detect_anomaly(False)
    k_inp_d = k_inp_d.detach()
    k_inp_d.requires_grad = True
    v_inp_d = v_inp_d.detach()
    v_inp_d.requires_grad = False
    q_inp_d = q_inp_d.detach()
    q_inp_d.requires_grad = False
    out_torch_v3_d = disco_att_torch(k_inp_d, v_inp_d, q_inp_d,
                                     quad_weights_d, col_idx_d, row_idx,
                                     nlat_in, nlon_in, nlat_out, nlon_out)
    # out_torch_v3_d = _disco_att_v3_torch(k_inp_d, v_inp_d, q_inp_d,
    # quad_weights_d, col_idx_d, row_off,
    # nlon_in, nlat_out, nlon_out)

    # need 'retain_graph' to avoid an error in the tests after this one
    out_torch_v3_d.backward(grad_d, retain_graph=True)
    dk_inp_torch_v3_d = k_inp_d.grad.clone()
    with torch.no_grad():
        dk_inp_torch_explicit_d = _disco_att_bwd_dk_torch(k_inp_d, v_inp_d, q_inp_d, grad_d,
                                                         quad_weights_d, col_idx_d, row_off,
                                                         nlat_in, nlon_out, nlat_out, nlon_out)

    # print("dkx:")
    # print("by hand:   ", dk_inp_torch_explicit_d[0,0,:3,:3], "\n via torch: ", dk_inp_torch_v3_d[0,0,:3,:3])
    assert torch.allclose(dk_inp_torch_explicit_d, dk_inp_torch_v3_d)

    dydk_cuda = torch.zeros_like(grad_d)
    att_cuda.s2_attention_bwd_dk_cuda(k_inp_d, v_inp_d, q_inp_d, quad_weights_d, col_idx_d, row_off, max_psi_nnz, grad_d, nlon_in, nlat_out, nlon_out, dydk_cuda)
    # print("dydk_cuda:\n", dydk_cuda[0,0,:3,:3], "\n vs torch autograd: \n", dk_inp_torch_v3_d[0,0,:3,:3], "\n vs torch hand-rolled: \n", dk_inp_torch_explicit_d[0,0,:3,:3])
    assert torch.allclose(dydk_cuda, dk_inp_torch_v3_d)

def test_dqy(k_inp_d, v_inp_d, q_inp_d,
             quad_weights_d, col_idx_d, row_off,
             row_idx, max_psi_nnz,
             nlat_in, nlon_in, nlat_out, nlon_out,
             grad_d):

    # Execute and compare
    torch.autograd.set_detect_anomaly(False)
    k_inp_d = k_inp_d.detach()
    k_inp_d.requires_grad = False
    v_inp_d = v_inp_d.detach()
    v_inp_d.requires_grad = False
    q_inp_d = q_inp_d.detach()
    q_inp_d.requires_grad = True
    out_torch_v3_d = disco_att_torch(k_inp_d, v_inp_d, q_inp_d,
                                     quad_weights_d, col_idx_d, row_idx,
                                     nlat_in, nlon_in, nlat_out, nlon_out)
    # out_torch_v3_d = _disco_att_v3_torch(k_inp_d, v_inp_d, q_inp_d,
    # quad_weights_d, col_idx_d, row_off,
    # nlon_in, nlat_out, nlon_out)

    # need 'retain_graph' to avoid an error in the tests after this one
    out_torch_v3_d.backward(grad_d, retain_graph=True)
    dq_inp_torch_v3_d = q_inp_d.grad.clone()
    with torch.no_grad():
        dq_inp_torch_explicit_d = _disco_att_bwd_dq_torch(k_inp_d, v_inp_d, q_inp_d, grad_d,
                                                          quad_weights_d, col_idx_d, row_off,
                                                          nlat_in, nlon_out, nlat_out, nlon_out)

    # print("dqy:")
    # print("by hand: ", dq_inp_torch_explicit_d[0,0,:3,:3], "\n via torch: ", dq_inp_torch_v3_d[0,0,:3,:3])
    # print("ratio: ", dq_inp_torch_explicit_d[0,0,:3,:3] / dq_inp_torch_v3_d[0,0,:3,:3])
    assert torch.allclose(dq_inp_torch_explicit_d, dq_inp_torch_v3_d, atol=1e-7) # current diff.abs().max()==2e-8

    dydq_cuda = torch.zeros_like(grad_d)
    att_cuda.s2_attention_bwd_dq_cuda(k_inp_d, v_inp_d, q_inp_d, quad_weights_d, col_idx_d, row_off, max_psi_nnz, grad_d, nlon_in, nlat_out, nlon_out, dydq_cuda)
    # print("dydq_cuda:\n", dydq_cuda[0,0,:3,:3], "\n vs torch autograd: \n", dq_inp_torch_v3_d[0,0,:3,:3], "\n vs torch hand-rolled: \n", dq_inp_torch_explicit_d[0,0,:3,:3])
    assert torch.allclose(dydq_cuda, dq_inp_torch_v3_d, atol=1e-7)


def test_row_offset_cuda(psi_col_idx, psi_row_idx):
        # compute row offsets for more structured traversal.
    # only works if rows are sorted but they are by construction
    row_off = np.empty(nlat_out+1, dtype=np.int64)
    row_off[0] = 0
    row = psi_row_idx[0]
    for idz, z in enumerate(range(psi_col_idx.shape[0])):
        if psi_row_idx[z] != row:
            row_off[row+1] = idz
            row = psi_row_idx[z]

    # set the last value
    row_off[row+1] = idz+1

    psi_row_offset = torch.zeros(nlat_out+1, dtype=torch.int64).to(device)
    psi_row_count = torch.zeros(nlat_out+1, dtype=torch.int64).to(device)
    att_cuda.s2_row_offset(nas2.psi_col_idx, nas2.psi_row_idx, psi_row_offset, psi_row_count)
    
    assert((row_off == psi_row_offset.cpu().numpy()).all())

def test_s2_attention_fwd(nas2, nlon_in, nlat_out, nlon_out):
    qo = torch.rand((10, num_channels, nlat_in,nlon_in)).to(device)
    ki = torch.rand((10, num_channels, nlat_in,nlon_in)).to(device)
    vi = torch.rand((10, num_channels, nlat_in,nlon_in)).to(device)
    k = F.conv2d(ki, weight=nas2.k_weights, bias=None)
    q = F.conv2d(qo, weight=nas2.q_weights, bias=None)
    v = F.conv2d(vi, weight=nas2.v_weights, bias=None)
    y = torch.zeros_like(qo)
    att_cuda.s2_attention_fwd(k, v, q, nas2.quad_weights, nas2.psi_col_idx, nas2.psi_roff_idx, nas2.max_psi_nnz, nlon_in, nlat_out, nlon_out, y)
    with torch.no_grad():
        y_torch = nas2(qo,ki,vi)
        # print("cuda: ", y[0,0,:3,:3], " torch: ", y_torch[0,0,:3,:3])
        assert(torch.allclose(y, y_torch))
        # print("WARNING: s2_attention_fwd_cuda didn't match closely enough!")

if __name__ == "__main__":
    # for test purposes fix input output grids:
    #nlat_in, nlon_in = 129, 256
    # nlat_in, nlon_in = int(33), int(64)
    nlat_in, nlon_in = int(17), int(32)
    scale_factor = 1
    num_channels = 3
    num_batches = 10
    nlat_out, nlon_out = int(nlat_in // scale_factor), int(nlon_in // scale_factor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(42)
    nas2 = NeighborhoodAttentionS2(channels=num_channels, in_shape=(nlat_in, nlon_in), out_shape=(nlat_out, nlon_out), theta_cutoff=torch.pi / 128 * 10)
    nas2.to(device)

    test_row_offset_cuda(nas2.psi_col_idx, nas2.psi_row_idx)

    test_s2_attention_fwd(nas2, nlon_in, nlat_out, nlon_out)

    qo = torch.rand((num_batches, num_channels, nlat_in,nlon_in)).to(device)
    ki = torch.rand((num_batches, num_channels, nlat_in,nlon_in)).to(device)
    vi = torch.rand((num_batches, num_channels, nlat_in,nlon_in)).to(device)
    grad_d_k = torch.rand((num_batches, num_channels, nlat_in,nlon_in)).to(device)
    grad_d_v = torch.rand((num_batches, num_channels, nlat_in,nlon_in)).to(device)
    grad_d_q = torch.rand((num_batches, num_channels, nlat_in,nlon_in)).to(device)

    k = F.conv2d(ki, weight=nas2.k_weights, bias=None)
    q = F.conv2d(qo, weight=nas2.q_weights, bias=None)
    v = F.conv2d(vi, weight=nas2.v_weights, bias=None)

    test_dkx(k, v, q, nas2.quad_weights, nas2.psi_col_idx, nas2.psi_roff_idx, nas2.psi_row_idx, nas2.max_psi_nnz, nlat_in, nlon_in, nlat_out, nlon_out, grad_d_k)
    test_dvx(k, v, q, nas2.quad_weights, nas2.psi_col_idx, nas2.psi_roff_idx, nas2.psi_row_idx, nas2.max_psi_nnz, nlat_in, nlon_in, nlat_out, nlon_out, grad_d_v)
    test_dqy(k, v, q, nas2.quad_weights, nas2.psi_col_idx, nas2.psi_roff_idx, nas2.psi_row_idx, nas2.max_psi_nnz, nlat_in, nlon_in, nlat_out, nlon_out, grad_d_q)
    
