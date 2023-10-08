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
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

import re
from pathlib import Path


def version(root_path):
    """Returns the version taken from __init__.py

    Parameters
    ----------
    root_path : pathlib.Path
        path to the root of the package

    Reference
    ---------
    https://packaging.python.org/guides/single-sourcing-package-version/
    """
    version_path = root_path.joinpath('torch_harmonics', '__init__.py')
    with version_path.open() as f:
        version_file = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def readme(root_path):
    """Returns the text content of the README.md of the package

    Parameters
    ----------
    root_path : pathlib.Path
        path to the root of the package
    """
    with root_path.joinpath('README.md').open(encoding='UTF-8') as f:
        return f.read()


root_path = Path(__file__).parent
README = readme(root_path)
VERSION = version(root_path)


config = {
    'name': 'torch_harmonics',
    'packages': find_packages(),
    'description': 'A differentiable spherical harmonic transform for PyTorch.',
    'long_description': README,
    'long_description_content_type' : 'text/markdown',
    'url' : 'https://github.com/NVIDIA/torch-harmonics',
    'author': 'Boris Bonev',
    'author_email': 'bbonev@nvidia.com',
    'version': VERSION,
    'install_requires': ['torch', 'numpy'],
    'extras_require': {
        'sfno':  ['tensorly', 'tensorly-torch'],
    },
    'license': 'Modified BSD',
    'scripts': [],
    'include_package_data': True,
    'classifiers': [
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3'
    ],
}

setup(**config)
