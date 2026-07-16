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

import os
import tempfile
import urllib.request
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

_MOLA_URL = (
    "https://astrogeology.usgs.gov/ckan/dataset/"
    "83c20dbd-e2b3-4e5b-b019-f13d4fdffa38/resource/"
    "57f84b24-d56c-42dd-a34d-cf9d61a82d2c/download/"
    "mars_mgs_mola_dem_mosaic_global_1024.jpg"
)


def load_mola_elevation(
    nlat: Optional[int] = None,
    nlon: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> torch.Tensor:
    """
    Download the NASA MOLA Mars digital elevation map and return it as a tensor.

    The image is downloaded once and cached locally for subsequent calls.

    Parameters
    ----------
    nlat : int, optional
        Target number of latitude points. If given together with *nlon*,
        the image is bilinearly interpolated to (nlat, nlon).
    nlon : int, optional
        Target number of longitude points.
    cache_dir : str, optional
        Directory for the cached download. Defaults to the system temp dir.

    Returns
    -------
    torch.Tensor
        Grayscale elevation map with shape (nlat, nlon), values in [0, 1].
    """
    from PIL import Image

    if cache_dir is None:
        cache_dir = tempfile.gettempdir()
    path = os.path.join(cache_dir, "mola_topo.jpg")

    if not os.path.exists(path):
        req = urllib.request.Request(_MOLA_URL, headers={"User-Agent": "torch-harmonics"})
        with urllib.request.urlopen(req) as resp, open(path, "wb") as f:
            f.write(resp.read())

    img = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    data = torch.from_numpy(img)

    if nlat is not None and nlon is not None:
        data = (
            F.interpolate(
                data.unsqueeze(0).unsqueeze(0),
                size=(nlat, nlon),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

    return data


class _EnsureContiguous(torch.autograd.Function):
    """Ensures the tensor is contiguous in both the forward and backward pass."""

    @staticmethod
    def forward(x):
        return x.contiguous()

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad):
        return grad.contiguous()


def ensure_contiguous(x: torch.Tensor) -> torch.Tensor:
    """Ensures the tensor is contiguous in both the forward and backward pass."""
    return _EnsureContiguous.apply(x)
