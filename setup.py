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

import os, sys
import warnings

from setuptools import setup, find_packages
from setuptools.command.install import install

# some code to handle the building of custom modules
FORCE_CUDA_EXTENSION = os.getenv("FORCE_CUDA_EXTENSION", "0") == "1"
BUILD_CPP = BUILD_CUDA = False

# try to import torch
try:
    import torch

    print(f"setup.py with torch {torch.__version__}")
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    BUILD_CPP = True
    from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension

    BUILD_CUDA = FORCE_CUDA_EXTENSION or (torch.cuda.is_available() and (CUDA_HOME is not None))
except (ImportError, TypeError, AssertionError, AttributeError) as e:
    warnings.warn(f"building custom extensions skipped: {e}")

def get_ext_modules():

    ext_modules = []
    cmdclass = {}

    if BUILD_CPP:
        print(f"Compiling helper routines for torch-harmonics.")
        ext_modules.append(CppExtension("disco_helpers", ["torch_harmonics/csrc/disco/disco_helpers.cpp"]))
        cmdclass["build_ext"] = BuildExtension

    if BUILD_CUDA:
        print(f"Compiling custom CUDA kernels for torch-harmonics.")
        ext_modules.append(
            CUDAExtension(
                "disco_cuda_extension",
                [
                    "torch_harmonics/csrc/disco/disco_interface.cu",
                    "torch_harmonics/csrc/disco/disco_cuda_fwd.cu",
                    "torch_harmonics/csrc/disco/disco_cuda_bwd.cu",
                ],
            )
        )
        cmdclass["build_ext"] = BuildExtension

    return ext_modules, cmdclass

if __name__ == "__main__":

    ext_modules, cmdclass = get_ext_modules()

    setup(
        packages=find_packages(),
        ext_modules=ext_modules,
        cmdclass=cmdclass,
    )