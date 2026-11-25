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

r"""
The DISCO kernels that need an arch-specific target, and which device can run them.

They live in extensions of their own, one per target family, built for exactly the targets
``TORCH_CUDA_ARCH_LIST`` asks for (see ``_DISCO_ARCH_EXTENSIONS`` in ``setup.py``):

``torch_harmonics.disco._C_sm90``
    ``sm_90a``: the WGMMA kpacked forward, op ``forward_kpacked_sm90``, the Hankel/Toeplitz
    tensor-core forward, ops ``forward_hankel_sm90_{plan,run,config}``, and the frequency-space
    arc scatter, ops ``scatter_fft_sm90_{plan,run,config}``.
``torch_harmonics.disco._C_sm100``
    ``sm_100a`` and/or ``sm_103a``: the tcgen05 kpacked forward, op ``forward_kpacked_sm100``, the
    Hankel/Toeplitz tensor-core forward, ops ``forward_hankel_sm100_{plan,run,config}``, and the
    frequency-space arc scatter, ops ``scatter_fft_sm100_{plan,run,config}``.

The frequency-space scatter is one source (``kernels_cuda/regular/disco_cuda_bwd_fft.cu``) built
into both; each registers its ops under its family's name, so a build with both loads both.

Each reports the targets it holds images for in its ``TARGETS`` attribute. Arch-specific images
are not forward compatible -- an ``sm_100a`` image does not load on a 10.3 device, nor on Rubin
(10.7) -- so a device runs an extension only if the build holds the image of its own target
(#337). An extension that was not built is simply absent.
"""

import functools
import importlib
import warnings
from types import ModuleType
from typing import Dict, FrozenSet, Optional, Tuple

import torch

#: the arch-specific extensions, by family
EXTENSIONS: Dict[str, str] = {
    "sm90": "torch_harmonics.disco._C_sm90",
    "sm100": "torch_harmonics.disco._C_sm100",
}

#: compute capability -> (extension family, the target whose image runs on it)
DEVICE_TARGETS: Dict[Tuple[int, int], Tuple[str, str]] = {
    (9, 0): ("sm90", "9.0a"),
    (10, 0): ("sm100", "10.0a"),
    (10, 3): ("sm100", "10.3a"),
}

#: the kpacked forward of each extension family
KPACKED_OPS: Dict[str, str] = {
    "sm90": "forward_kpacked_sm90",
    "sm100": "forward_kpacked_sm100",
}

#: the prefix of the frequency-space scatter's ops (``<prefix>_plan``, ``_run``, ``_config``) of each extension family
FFT_SCATTER_OPS: Dict[str, str] = {
    "sm90": "scatter_fft_sm90",
    "sm100": "scatter_fft_sm100",
}

#: the prefix of the Hankel/Toeplitz forward's ops (``<prefix>_plan``, ``_run``, ``_config``) of each extension family
HANKEL_FORWARD_OPS: Dict[str, str] = {
    "sm90": "forward_hankel_sm90",
    "sm100": "forward_hankel_sm100",
}

#: the compute capabilities the Hankel/Toeplitz forward runs on: H100 and B200 / GB200
HANKEL_FORWARD_CAPABILITIES: FrozenSet[Tuple[int, int]] = frozenset({(9, 0), (10, 0)})

#: the compute capabilities the frequency-space scatter was validated on: H100 and B200 / GB200; a 10.3 (GB300) keeps
#: the arc scatter
FFT_SCATTER_CAPABILITIES: FrozenSet[Tuple[int, int]] = frozenset({(9, 0), (10, 0)})


@functools.lru_cache(maxsize=None)
def extension(family: str) -> Optional[ModuleType]:
    """The extension of ``family``, imported (which registers its operators), or None if it was not built."""
    try:
        return importlib.import_module(EXTENSIONS[family])
    except ModuleNotFoundError:
        return None
    except ImportError as e:
        # built but not loadable: say so, or the backends would quietly decline
        warnings.warn(f"the DISCO extension {EXTENSIONS[family]} is built but does not load: {e}")
        return None


def built_targets() -> FrozenSet[str]:
    """The arch-specific targets this build holds images for, e.g. ``{"9.0a", "10.0a"}``."""
    targets = set()
    for family in EXTENSIONS:
        module = extension(family)
        if module is not None:
            targets.update(getattr(module, "TARGETS", ()))
    return frozenset(targets)


def device_family(device_index: int) -> Optional[str]:
    """The extension family that can run on CUDA device ``device_index``, or None."""
    family, target = DEVICE_TARGETS.get(torch.cuda.get_device_capability(device_index), (None, None))
    return family if target in built_targets() else None


def kpacked_build_available() -> bool:
    """
    Whether this build holds a kpacked kernel for any device.

    Purely a build-time question, deliberately not a device check: layers are normally built on the
    CPU and moved afterwards, so the device is checked when a backend is selected for it.
    """
    return bool(built_targets())


def kpacked_supported_on_device(device_index: int) -> bool:
    """Whether the kpacked forward can run on CUDA device ``device_index`` with this build."""
    return device_family(device_index) in KPACKED_OPS


def kpacked_op(device_index: int):
    """The kpacked forward for CUDA device ``device_index`` (see ``kpacked_supported_on_device``)."""
    family = device_family(device_index)
    if family not in KPACKED_OPS:
        raise RuntimeError(f"this build has no kpacked kernel for CUDA device {device_index} (compute capability {torch.cuda.get_device_capability(device_index)})")
    return getattr(torch.ops.disco_kernels, KPACKED_OPS[family]).default


def loaded_kpacked_ops() -> Tuple[str, ...]:
    """The qualified names of the kpacked ops of the extensions this build has."""
    return tuple(f"disco_kernels::{KPACKED_OPS[family]}" for family in EXTENSIONS if family in KPACKED_OPS and extension(family) is not None)


def fft_scatter_registered(family: str) -> bool:
    """Whether the extension of ``family`` was built and registered the frequency-space scatter's ops."""
    return family in FFT_SCATTER_OPS and extension(family) is not None and hasattr(torch.ops.disco_kernels, FFT_SCATTER_OPS[family] + "_run")


def fft_scatter_supported_on_device(device_index: int) -> bool:
    """Whether the frequency-space scatter can run on CUDA device ``device_index`` with this build."""
    if torch.cuda.get_device_capability(device_index) not in FFT_SCATTER_CAPABILITIES:
        return False
    family = device_family(device_index)
    return family is not None and fft_scatter_registered(family)


def fft_scatter_ops(device_index: int):
    """The frequency-space scatter's ``(plan, run)`` ops for CUDA device ``device_index`` (see ``fft_scatter_supported_on_device``)."""
    if not fft_scatter_supported_on_device(device_index):
        raise RuntimeError(f"this build has no frequency-space scatter for CUDA device {device_index} (compute capability {torch.cuda.get_device_capability(device_index)})")
    prefix = FFT_SCATTER_OPS[device_family(device_index)]
    ops = torch.ops.disco_kernels
    return getattr(ops, prefix + "_plan").default, getattr(ops, prefix + "_run").default


def fft_scatter_config_op(device_index: int):
    """The op reporting the path the frequency-space scatter takes for given sizes on CUDA device ``device_index`` (for tests)."""
    if not fft_scatter_supported_on_device(device_index):
        raise RuntimeError(f"this build has no frequency-space scatter for CUDA device {device_index}")
    return getattr(torch.ops.disco_kernels, FFT_SCATTER_OPS[device_family(device_index)] + "_config").default


def hankel_forward_registered(family: str) -> bool:
    """Whether the extension of ``family`` was built and registered the Hankel/Toeplitz forward's ops."""
    return family in HANKEL_FORWARD_OPS and extension(family) is not None and hasattr(torch.ops.disco_kernels, HANKEL_FORWARD_OPS[family] + "_run")


def hankel_forward_supported_on_device(device_index: int) -> bool:
    """Whether the Hankel/Toeplitz forward can run on CUDA device ``device_index`` with this build."""
    if torch.cuda.get_device_capability(device_index) not in HANKEL_FORWARD_CAPABILITIES:
        return False
    family = device_family(device_index)
    return family is not None and hankel_forward_registered(family)


def _hankel_op(device_index: int, suffix: str):
    if not hankel_forward_supported_on_device(device_index):
        raise RuntimeError(f"this build has no Hankel/Toeplitz forward for CUDA device {device_index} (compute capability {torch.cuda.get_device_capability(device_index)})")
    return getattr(torch.ops.disco_kernels, HANKEL_FORWARD_OPS[device_family(device_index)] + suffix).default


def hankel_forward_ops(device_index: int):
    """The Hankel/Toeplitz forward's ``(plan, run)`` ops for CUDA device ``device_index`` (see ``hankel_forward_supported_on_device``)."""
    return _hankel_op(device_index, "_plan"), _hankel_op(device_index, "_run")


def hankel_forward_config_op(device_index: int):
    """The op reporting the configuration the Hankel/Toeplitz forward takes for given sizes on CUDA device ``device_index`` (for tests)."""
    return _hankel_op(device_index, "_config")
