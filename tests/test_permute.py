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
import unittest
from parameterized import parameterized, parameterized_class


import torch
from torch.library import opcheck
from torch_harmonics.utils import permute_to_0231, permute_to_0312

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))

@parameterized_class(("device"), _devices)
class TestPermutation(unittest.TestCase):
    """Test the optimized convolution module (CPU/CUDA if available)."""

    def setUp(self):
        torch.manual_seed(333)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(333)

    @parameterized.expand(
        [
            [8, 8, 16, 32, "0231"], 
            [8, 1, 16, 32, "0231"], 
            [1, 8, 16, 32, "0231"],
            [8, 8, 16, 32, "0312"], 
            [8, 1, 16, 32, "0312"], 
            [1, 8, 16, 32, "0312"],
        ],
        skip_on_empty=True,
    )
    def test_permutation(
        self, batch_size, channels, nlat, nlon, mode
    ):
        # create input
        if mode == "0231":
            inp = torch.randn(batch_size, channels, nlat, nlon, device=self.device)
            permute_fn = permute_to_0231
            permute_shape = (0, 2, 3, 1)
        else:
            inp = torch.randn(batch_size, nlat, nlon, channels, device=self.device)
            permute_fn = permute_to_0312 
            permute_shape = (0, 3, 1, 2)
        inp.requires_grad = True

        # forward test
        out_opt = permute_fn(inp)
        out_naive = inp.permute(*permute_shape).contiguous().clone()
        self.assertTrue(torch.allclose(out_opt, out_naive))

        # backward test
        ograd = torch.randn_like(out_opt)
        out_opt.backward(ograd)
        igrad_opt = inp.grad.clone()
        inp.grad = None
        out_naive.backward(ograd)
        igrad_naive = inp.grad.clone()
        self.assertTrue(torch.allclose(igrad_opt, igrad_naive))


    @parameterized.expand(
        [
            [8,  8, 16, 32, "0231"], 
            [8,  8, 16, 32, "0312"], 
        ],
        skip_on_empty=True,
    )
    def test_pt2_compatibility(self, batch_size, channels, nlat, nlon, mode):

        if mode == "0231":
            inp = torch.randn(batch_size, channels, nlat, nlon, device=self.device)
            permute_fn = torch.ops.utility_kernels.permute_to_0231
        else:
            inp = torch.randn(batch_size, nlat, nlon, channels, device=self.device)
            permute_fn = torch.ops.utility_kernels.permute_to_0312

        test_inputs = (inp, )

        opcheck(permute_fn, test_inputs)


if __name__ == "__main__":
    unittest.main()
