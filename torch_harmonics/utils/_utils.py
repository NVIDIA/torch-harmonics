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

from typing import Optional

import torch
from utility_helpers import optimized_kernels_is_available
from . import utility_kernels

# custom kernels
if optimized_kernels_is_available():

    # fake permutations
    @torch.library.register_fake("utility_kernels::permute_to_0231")
    def _(inp: torch.Tensor) -> torch.Tensor:
        B, C, H, W = inp.shape
        out_shape = (B, H, W, C)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    @torch.library.register_fake("utility_kernels::permute_to_0312")
    def _(inp: torch.Tensor) -> torch.Tensor:
        B, H, W, C = inp.shape
        out_shape = (B, C, H, W)
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # autograds: shallow wrappers around the default kernels
    def _permute_to_0231_bwd(ctx, grad_output):
        return utility_kernels.permute_to_0312.default(grad_output)

    def _permute_to_0312_bwd(ctx, grad_output):
        return utility_kernels.permute_to_0231.default(grad_output)

    torch.library.register_autograd(
        "utility_kernels::permute_to_0231", _permute_to_0231_bwd)

    torch.library.register_autograd(
        "utility_kernels::permute_to_0312", _permute_to_0312_bwd)

if optimized_kernels_is_available():
    def permute_to_0231(inp: torch.Tensor) -> torch.Tensor:
        return utility_kernels.permute_to_0231.default(inp)
    
    def permute_to_0312(inp: torch.Tensor) -> torch.Tensor:
        return utility_kernels.permute_to_0312.default(inp)
else:
    def permute_to_0231(inp: torch.Tensor) -> torch.Tensor:
        return inp.permute(0, 2, 3, 1).contiguous()

    def permute_to_0312(inp: torch.Tensor) -> torch.Tensor:
        return inp.permute(0, 3, 1, 2).contiguous()


