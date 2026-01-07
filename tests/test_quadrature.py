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


class TestQuadrature(unittest.TestCase):
    """Serial QuadratureS2 integration tests."""

    @parameterized.expand(
        [
            [64, 128, 2, 3, "equiangular", False, 1e-6],
            [64, 128, 2, 3, "equiangular", True, 1e-6],
            [65, 128, 1, 1, "legendre-gauss", False, 1e-6],
            [65, 128, 1, 1, "legendre-gauss", True, 1e-6],
            [65, 128, 2, 2, "lobatto", False, 1e-6],
            [65, 128, 2, 2, "lobatto", True, 1e-6],
        ]
    )
    def test_constant_integral(self, nlat, nlon, batch_size, num_chan, grid, normalize, tol):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        quad = th.QuadratureS2(img_shape=(nlat, nlon), grid=grid, normalize=normalize).to(device)

        data = torch.ones((batch_size, num_chan, nlat, nlon), dtype=torch.float32, device=device)
        out = quad(data)

        expected_value = 1.0 if normalize else 4.0 * torch.pi
        expected = torch.full((batch_size, num_chan), expected_value, device=device)

        self.assertTrue(torch.allclose(out, expected, atol=tol, rtol=tol))


if __name__ == "__main__":
    unittest.main()

