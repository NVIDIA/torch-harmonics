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

import torch

"""
Contains complex contractions wrapped into jit for harmonic layers
"""

@torch.jit.script
def contract_diagonal(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    res = torch.einsum("bixy,kixy->bkxy", ac, bc)
    return torch.view_as_real(res)

@torch.jit.script
def contract_dhconv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    res = torch.einsum("bixy,kix->bkxy", ac, bc)
    return torch.view_as_real(res)

@torch.jit.script
def contract_blockdiag(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    res = torch.einsum("bixy,kixyz->bkxz", ac, bc)
    return torch.view_as_real(res)

# Helper routines for the non-linear FNOs (Attention-like)
@torch.jit.script
def compl_mul1d_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    tmp = torch.einsum("bixs,ior->srbox", a, b)
    res = torch.stack([tmp[0,0,...] - tmp[1,1,...], tmp[1,0,...] + tmp[0,1,...]], dim=-1) 
    return res

@torch.jit.script
def compl_mul1d_fwd_c(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bix,io->box", ac, bc)
    res = torch.view_as_real(resc)
    return res

@torch.jit.script
def compl_muladd1d_fwd(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    res = compl_mul1d_fwd(a, b) + c
    return res

@torch.jit.script
def compl_muladd1d_fwd_c(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    tmpcc = torch.view_as_complex(compl_mul1d_fwd_c(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)

# Helper routines for FFT MLPs

@torch.jit.script
def compl_mul2d_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    tmp = torch.einsum("bixys,ior->srboxy", a, b)
    res = torch.stack([tmp[0,0,...] - tmp[1,1,...], tmp[1,0,...] + tmp[0,1,...]], dim=-1) 
    return res


@torch.jit.script
def compl_mul2d_fwd_c(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,io->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res

@torch.jit.script
def compl_muladd2d_fwd(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    res = compl_mul2d_fwd(a, b) + c
    return res

@torch.jit.script
def compl_muladd2d_fwd_c(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    tmpcc = torch.view_as_complex(compl_mul2d_fwd_c(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)

@torch.jit.script
def real_mul2d_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.einsum("bixy,io->boxy", a, b)
    return out

@torch.jit.script
def real_muladd2d_fwd(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return compl_mul2d_fwd_c(a, b) + c

