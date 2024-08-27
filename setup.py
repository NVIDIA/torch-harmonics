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

import sys

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

import torch
from torch.utils import cpp_extension


def get_ext_modules(argv):

    compile_cuda_extension = "--cuda_ext" in argv
    if "--cuda_ext" in argv:
        argv.remove("--cuda_ext")

    print(compile_cuda_extension)

    ext_modules = [
        cpp_extension.CppExtension("disco_helpers", ["torch_harmonics/csrc/disco/disco_helpers.cpp"]),
    ]

    if torch.cuda.is_available() or compile_cuda_extension:
        ext_modules.append(
            cpp_extension.CUDAExtension(
                "disco_cuda_extension",
                [
                    "torch_harmonics/csrc/disco/disco_interface.cu",
                    "torch_harmonics/csrc/disco/disco_cuda_fwd.cu",
                    "torch_harmonics/csrc/disco/disco_cuda_bwd.cu",
                ],
            )
        )

    return ext_modules

if __name__ == "__main__":

    ext_modules = get_ext_modules(sys.argv)

    setup(
        packages=find_packages(),
        ext_modules=ext_modules,
        cmdclass={"build_ext": cpp_extension.BuildExtension},
    )