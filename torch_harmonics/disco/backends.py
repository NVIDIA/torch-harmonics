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

r"""
DISCO backends: an implementation of the psi contraction together with the state it needs.

The contraction of a DISCO layer can run three ways, and each wants psi in a different
form:

* ``kpacked`` -- the tensor-core kernels (WGMMA on sm_90a, tcgen05 on sm_100a), for
  fp16/bf16 activations. psi in a blocked CSR where every neighbour carries all K filter
  values, plus the plain CSR its backward scatters with.
* ``csr`` -- the compiled CSR gather/scatter kernels, on CPU and CUDA alike: the
  operator dispatches to the device's kernel, so one backend serves both.
* ``reference`` -- the torch implementation, in terms of a sparse COO psi. What a build
  without the kernels gets, or a layer with ``optimized_kernel=False``.

The layer used to register the union -- COO, CSR, a second per-basis-function copy of the
CSR, the kpacked layout whenever the build had one, and the sparse tensor for the
reference -- and pick among them in ``forward``. A backend now registers what it reads,
and nothing else; see :mod:`torch_harmonics._backend` for how selection works.

A backend serves every DISCO layer, serial or distributed, forward or transpose. The
layer describes its psi and leaves the kernels to the backend:

``transpose``
    Class attribute; whether psi is applied in the scatter direction.
``kernel_size``, ``nlon_in``, ``nlon_out``
    As usual.
``_psi_coo()``
    The psi entries ``(ker_idx, row_idx, col_idx, vals)`` on the CPU, fresh -- the CSR
    preprocessing sorts them in place. A distributed layer returns its rank's block.
``_psi_nrows``
    The rows psi is keyed by: the output latitudes of the forward, the input latitudes
    of the transpose; local ones on a distributed layer.
``_contract_shape``
    ``(nlat, nlon)`` of the contraction's result.
``_reference_psi(ker_idx, row_idx, col_idx, vals)``
    The sparse psi of the torch reference, whose extents are a property of the layer.
``_needs_split``
    Whether the layer contracts through the fused node with the spatial-first input
    gradient in play, which is what the per-basis-function CSR tables are for.

The dtype is the one axis selection cannot settle, since fp16/bf16 -- under autocast in
particular -- is only known in ``forward``. The kpacked backend therefore checks it per
call and falls back to its CSR half for anything else. That is a branch on a static
dtype, so it stays traceable.
"""

from typing import TYPE_CHECKING

import torch
from disco_helpers import optimized_kernels_is_available, pack_psi_dense, preprocess_psi

from torch_harmonics._backend import BackendS2

from .kernels_torch.disco_torch import _disco_s2_contraction_regular_torch, _disco_s2_transpose_contraction_regular_torch
from .optimized.disco_optimized import (
    _build_kernel_split_csr,
    _disco_s2_contraction_kpacked,
    _disco_s2_conv_optimized,
    _kpacked_build_available,
    _kpacked_k_pad,
    _kpacked_supported_on_device,
    _maybe_kpack_psi,
)

if optimized_kernels_is_available():
    from .optimized.disco_optimized import _disco_s2_contraction_regular_optimized, _disco_s2_transpose_contraction_regular_optimized

if TYPE_CHECKING:  # pragma: no cover
    from torch_harmonics.disco.convolution import DiscreteContinuousConv


_HALF_DTYPES = (torch.float16, torch.bfloat16)


def _amp_dtype(x: torch.Tensor) -> torch.dtype:
    """The dtype the contraction will actually run in: the autocast dtype if autocast is on for x's device."""
    device_type = x.device.type
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    return x.dtype


def _weight_contraction(x: torch.Tensor, weight: torch.Tensor, groups: int, groupsize: int) -> torch.Tensor:
    """(B, G*Cg, K, H, W) x (G, Og, Cg, K) -> (B, G*Og, H, W)."""
    B, _, K, H, W = x.shape
    x = x.reshape(B, groups, groupsize, K, H, W)
    out = torch.einsum("bgckxy,gock->bgoxy", x, weight).contiguous()
    return out.reshape(B, groups * weight.shape[1], H, W)


class DiscoBackendS2(BackendS2):
    """
    One way of evaluating the psi contraction of a DISCO layer.

    Besides ``available`` and ``prepare``, a backend provides the three operations the
    layers are built from. Each reads the prepared state back off the layer by name.

    ``contract(layer, x)``
        ``(B, C, H_in, W_in) -> (B, C, K, H_out, W_out)``, differentiable in ``x``.
    ``conv(layer, x, weight, groups, groupsize, recompute)``
        The contraction followed by the weight contraction,
        ``-> (B, groups * out_per_group, H_out, W_out)``. ``recompute`` asks for the
        K-expanded intermediate to be recomputed in backward rather than saved; a
        backend that cannot do that computes the same result the plain way.
    ``transpose(layer, x)``
        The scatter direction, ``(B, C, K, H_in, W_in) -> (B, C, H_out, W_out)``.
    """

    def contract(self, layer: "DiscreteContinuousConv", x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def conv(self, layer: "DiscreteContinuousConv", x: torch.Tensor, weight: torch.Tensor, groups: int, groupsize: int, recompute: bool = False) -> torch.Tensor:
        return _weight_contraction(self.contract(layer, x), weight, groups, groupsize)

    def transpose(self, layer: "DiscreteContinuousConv", x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReferenceBackend(DiscoBackendS2):
    """
    The torch reference, in terms of a sparse COO psi.

    Always available, so it is the last candidate. It contracts by repeatedly rolling the
    input -- the specification, not a fast path -- and has no recompute, so ``fused``
    changes nothing here but memory.
    """

    name = "reference"

    @classmethod
    def available(cls, layer, device):
        return True

    def prepare(self, layer, device):
        return {"psi": layer._reference_psi(*layer._psi_coo()).to(device)}

    def contract(self, layer, x):
        return _disco_s2_contraction_regular_torch(x, layer.psi, layer.nlon_out)

    def transpose(self, layer, x):
        return _disco_s2_transpose_contraction_regular_torch(x, layer.psi, layer.nlon_out)


class CSRBackend(DiscoBackendS2):
    """
    The compiled CSR kernels, CPU or CUDA.

    The operators are registered for both devices, so the dispatcher picks the kernel
    and this one backend serves either -- the device is not a condition here, only
    whether the kernels were built and the layer wants them.
    """

    name = "csr"

    @classmethod
    def available(cls, layer, device):
        return layer.optimized_kernel and optimized_kernels_is_available()

    def prepare(self, layer, device):
        return self._csr_state(layer, device)[0]

    def _csr_state(self, layer, device):
        """The CSR buffers, the per-basis-function tables if the layer wants them, and the sorted COO on the CPU."""
        # on the CPU explicitly: a layer built under a torch.device context gets its psi
        # there, and the preprocessing below would sort a temporary host copy of it
        ker_idx, row_idx, col_idx, vals = (t.cpu().contiguous() for t in layer._psi_coo())

        # sorts the four arrays in place, by basis function, which is the order the
        # kernels and the per-basis-function tables below both rely on
        roff_idx = preprocess_psi(layer.kernel_size, layer._psi_nrows, ker_idx, row_idx, col_idx, vals).contiguous()

        state = {
            "psi_roff_idx": roff_idx.to(device),
            "psi_ker_idx": ker_idx.to(device),
            "psi_row_idx": row_idx.to(device),
            "psi_col_idx": col_idx.to(device),
            "psi_vals": vals.to(device),
        }

        self.split_offsets = None
        if layer._needs_split:
            split_roff_idx, split_ker_idx, row_offsets, nnz_offsets = _build_kernel_split_csr(roff_idx, ker_idx, layer.kernel_size)
            state["psi_split_roff_idx"] = split_roff_idx.to(device)
            state["psi_split_ker_idx"] = split_ker_idx.to(device)
            self.split_offsets = (row_offsets, nnz_offsets)

        return state, (ker_idx, row_idx, col_idx, vals, roff_idx)

    def _csr(self, layer):
        return (layer.psi_roff_idx, layer.psi_ker_idx, layer.psi_row_idx, layer.psi_col_idx, layer.psi_vals)

    def _split(self, layer):
        if self.split_offsets is None:
            return None
        return (layer.psi_split_roff_idx, layer.psi_split_ker_idx, *self.split_offsets)

    def contract(self, layer, x):
        return _disco_s2_contraction_regular_optimized(x, *self._csr(layer), layer.kernel_size, *layer._contract_shape)

    def conv(self, layer, x, weight, groups, groupsize, recompute=False):
        # the contraction op follows autocast by itself; this path calls the kernels
        # directly, so it applies the cast the same way
        x = x.to(_amp_dtype(x))
        return _disco_s2_conv_optimized(x, weight, self._csr(layer), None, self._split(layer), layer.kernel_size, *layer._contract_shape, groups, groupsize, recompute)

    def transpose(self, layer, x):
        return _disco_s2_transpose_contraction_regular_optimized(x, *self._csr(layer), layer.kernel_size, *layer._contract_shape)


class KpackedBackend(CSRBackend):
    """
    The tensor-core forward (WGMMA on sm_90a, tcgen05 on sm_100a), with the CSR backward.

    Serves fp16/bf16 activations and hands anything else to its CSR half, which it also
    needs for the backward. The layout is built only on a device that can run it: it is
    padded per neighbour to K_pad values, which at quarter degree is a sizeable buffer to
    hold for a kernel that could never launch.

    Available only for the forward direction, on a device and build the kernels were
    compiled for (see ``_kpacked_supported_on_device``), for K <= 16 and ``nlon_out`` a
    multiple of 8. Declines in ``prepare`` when the basis functions do not share one
    support set, which the layout requires and which is only known once it is built.
    """

    name = "kpacked"

    @classmethod
    def available(cls, layer, device):
        return (
            super().available(layer, device)
            and not layer.transpose
            and device.type == "cuda"
            and _kpacked_build_available()
            and _kpacked_supported_on_device(device.index if device.index is not None else torch.cuda.current_device())
            and _kpacked_k_pad(layer.kernel_size) is not None
            and layer.nlon_out % 8 == 0
        )

    def prepare(self, layer, device):
        state, (ker_idx, row_idx, col_idx, vals, roff_idx) = self._csr_state(layer, device)

        packed_idx, packed_vals, packed_count = pack_psi_dense(layer.kernel_size, layer._psi_nrows, layer.nlon_in, 0, ker_idx, row_idx, col_idx, vals, roff_idx)
        kpack = _maybe_kpack_psi(packed_idx.contiguous(), packed_vals.contiguous(), packed_count.contiguous())
        if kpack is None or kpack[3] != _kpacked_k_pad(layer.kernel_size):
            return None

        kpacked_idx, kpacked_vals, kpacked_offset, _ = kpack
        state["psi_kpacked_idx"] = kpacked_idx.to(device)
        state["psi_kpacked_vals"] = kpacked_vals.to(device)
        state["psi_kpacked_offset"] = kpacked_offset.to(device)
        return state

    def _kpacked(self, layer):
        return (layer.psi_kpacked_idx, layer.psi_kpacked_vals, layer.psi_kpacked_offset)

    def contract(self, layer, x):
        dtype = _amp_dtype(x)
        if dtype not in _HALF_DTYPES:
            return super().contract(layer, x)
        return _disco_s2_contraction_kpacked(x.to(dtype), *self._kpacked(layer), *self._csr(layer), layer.kernel_size, *layer._contract_shape)

    def conv(self, layer, x, weight, groups, groupsize, recompute=False):
        dtype = _amp_dtype(x)
        if dtype not in _HALF_DTYPES:
            return super().conv(layer, x, weight, groups, groupsize, recompute)
        return _disco_s2_conv_optimized(
            x.to(dtype), weight, self._csr(layer), self._kpacked(layer), self._split(layer), layer.kernel_size, *layer._contract_shape, groups, groupsize, recompute
        )


#: Every backend, most specific first; see :mod:`torch_harmonics._backend`. The
#: transpose layers use the same list: the kpacked backend declines them itself.
BACKENDS = (KpackedBackend, CSRBackend, ReferenceBackend)
