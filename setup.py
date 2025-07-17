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

def get_compile_args(module_name):
    """If user runs build with TORCH_HARMONICS_DEBUG=1 set, it will use debugging flags to build"""
    
    debug_mode = os.environ.get('TORCH_HARMONICS_DEBUG', '0') == '1'
    profile_mode = os.environ.get('TORCH_HARMONICS_PROFILE', '0') == '1'
    openmp_mode = os.getenv('TORCH_HARMONICS_ENABLE_OPENMP', '0') == '1'

    cpp_extra_flags = []
    if openmp_mode:
        cpp_extra_flags.append("-fopenmp")

    nvcc_extra_flags = []
    if profile_mode:
        nvcc_extra_flags.append("-lineinfo")
        nvcc_extra_flags.append("-Xptxas=-v")
        
    if debug_mode:
        print(f"WARNING: Compiling {module_name} with debugging flags")
        return {
            'cxx': ['-g', '-O0', '-Wall'],
            'nvcc': ['-g', '-G', '-O0'] + nvcc_extra_flags
        }
    else:
        print(f"NOTE: Compiling {module_name} with release flags")
        return {
            'cxx': ['-O3', "-DNDEBUG"] + cpp_extra_flags,
            'nvcc': ['-O3', "-DNDEBUG"] + nvcc_extra_flags
        }

def get_helpers_compile_args():
    return {
        'cxx': [
            f'-DBUILD_CPP={1 if BUILD_CPP else 0}', 
            f'-DBUILD_CUDA={1 if BUILD_CUDA else 0}'
        ], 
    }

def get_ext_modules():
    """Get list of extension modules to compile."""
    
    ext_modules = []
    cmdclass = {}

    print(f"Compiling helper routines for torch-harmonics.")
    ext_modules.append(
        CppExtension(
            "disco_helpers", 
            [
                "torch_harmonics/disco/csrc/disco_helpers.cpp",
            ],
            extra_compile_args=get_helpers_compile_args(),
        )
    )

    if BUILD_CPP:
        # Create a single extension that includes both CPU and CUDA code
        sources = [
            "torch_harmonics/disco/csrc/disco_interface.cpp",
            "torch_harmonics/disco/csrc/disco_cpu.cpp"
        ]
        
        if BUILD_CUDA:
            print(f"Compiling custom CUDA kernels for torch-harmonics.")
            sources.extend([
                "torch_harmonics/disco/csrc/disco_cuda_fwd.cu",
                "torch_harmonics/disco/csrc/disco_cuda_bwd.cu",
            ])
            ext_modules.append(
                CUDAExtension(
                    "torch_harmonics.disco._C",
                    sources,
                    extra_compile_args=get_compile_args("disco")
                )
            )
        else:
            ext_modules.append(
                CppExtension(
                    "torch_harmonics.disco._C", 
                    sources,
                    extra_compile_args=get_compile_args("disco")
                )
            )
        )
        cmdclass["build_ext"] = BuildExtension

    if BUILD_CUDA:
        print(f"Compiling custom CUDA kernels for torch-harmonics.")
        ext_modules.append(
            CUDAExtension(
                "torch_harmonics.disco._C",
                [
                    "torch_harmonics/disco/csrc/disco_cuda_fwd.cu",
                    "torch_harmonics/disco/csrc/disco_cuda_bwd.cu",
                ],
                extra_compile_args=get_compile_args("disco")
            )
        )
        ext_modules.append(
            CUDAExtension(
                name="attention_cuda_extension",
                sources=[
                    "torch_harmonics/attention/csrc/attention_utils.cu",
                    "torch_harmonics/attention/csrc/attention_fwd_cuda.cu",
                    "torch_harmonics/attention/csrc/attention_bwd_cuda.cu",
                    "torch_harmonics/attention/csrc/attention_interface.cu",
                ],
                extra_compile_args=get_compile_args("neighborhood_attention")
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
