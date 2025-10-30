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

import torch

def compare_tensors(msg, tensor, tensor_ref, rtol=1e-8, atol=1e-5,  verbose=False):
    allclose = torch.allclose(tensor, tensor_ref, rtol=rtol, atol=atol)
    if (not allclose) and verbose:
        diff = torch.abs(tensor - tensor_ref)
        print(f"{msg} absolute tensor diff: min = {torch.min(diff)}, mean = {torch.mean(diff)}, max = {torch.max(diff)}.")
        reldiff = diff / torch.abs(tensor_ref)
        print(f"{msg} relative tensor diff: min = {torch.min(reldiff)}, mean = {torch.mean(reldiff)}, max = {torch.max(reldiff)}.")
        # find element with maximum difference
        index = torch.argmax(diff)
        print(f"{msg} element {index} with maximum difference: value = {tensor.flatten()[index]}, reference value = {tensor_ref.flatten()[index]}, diff = {diff.flatten()[index]}.")
    return allclose
