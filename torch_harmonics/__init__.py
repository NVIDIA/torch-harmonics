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

# Generated at build time by setuptools-scm from the git tag; see
# [tool.setuptools_scm] in pyproject.toml. Not checked in, so an unbuilt source
# tree falls back to the sentinel rather than a stale hardcoded string.
try:
    from ._version import version as __version__
except ImportError:  # pragma: no cover - source tree that was never built
    __version__ = "0.0.0"

from . import examples, grid, integration, partition, quadrature, random_fields
from .attention import AttentionS2, NeighborhoodAttentionS2
from .disco import DiscreteContinuousConvS2, DiscreteContinuousConvTransposeS2
from .grid import EquiangularGrid, EquiangularTrapezoidalGrid, GridS2, GridShardS2, LegendreGaussGrid, LobattoGrid, RegularGridS2, RegularGridShardS2, as_grid, grid_params, grid_types, require_grid, require_regular_grid
from .integration import QuadratureS2
from .resample import ResampleS2
from .sht import InverseRealSHT, InverseRealVectorSHT, RealSHT, RealVectorSHT
from .spectral_convolution import SpectralConvS2
from .truncation import truncate_sht, truncate_support
