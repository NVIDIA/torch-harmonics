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
Attention backends: an implementation together with the state it needs.

A neighbourhood is one piece of geometry, but no two implementations want it in the
same form. The CUDA ragged kernels walk contiguous arcs and index weights by ring; the
torch reference gathers an explicit column list and indexes weights by point; the
regular CUDA kernels want both, because their schema is shared with the CPU kernels;
the distributed ring kernels want a sharded column list plus a row permutation. A
FlexAttention backend would want a ``BlockMask``, which is not a tensor at all.

Left to the layer, that becomes a union: every buffer any implementation might need,
registered always, with the surplus deleted or rebuilt lazily by whoever knows better.
That is what this replaces. A backend declares when it can run and what it needs, and
the layer registers exactly that.

Which backend can run depends on the device, and the device is not known when the
layer is built -- constructing on CPU and moving with ``.to()`` is the normal thing to
do. So selection is driven by :meth:`~torch.nn.Module._apply`, which every one of
``.to()``, ``.cuda()``, ``.cpu()``, ``.float()`` funnels through. On a device change
the layer reselects and the new backend prepares its state there. Nothing about this
is lazy and nothing is decided in ``forward``, which is what keeps the forward pass
traceable: the backend is a plain Python attribute, fixed before tracing begins.

Backend state is always ``persistent=False``. Which backend is live is a property of
where the module happens to be, never of what was trained, so it must not reach a
checkpoint.
"""

from typing import TYPE_CHECKING, Dict

import torch

from torch_harmonics.attention.kernels_torch.attention_ragged_torch import _neighborhood_s2_attention_ragged_torch
from torch_harmonics.attention.kernels_torch.attention_regular_torch import _neighborhood_s2_attention_regular_torch
from torch_harmonics.attention.optimized.attention_optimized import _neighborhood_s2_attention_ragged_optimized, _neighborhood_s2_attention_regular_optimized
from torch_harmonics.neighborhood import precompute_neighborhood_csr_s2

if TYPE_CHECKING:  # pragma: no cover
    from torch_harmonics.attention.attention import NeighborhoodAttentionS2


class AttentionBackendS2:
    """
    One way of evaluating neighbourhood attention, and the state it needs to do it.

    Subclasses implement three things:

    ``available(layer, device)``
        Whether this backend can serve that layer on that device. Checked in the order
        the layer lists its backends, so the most specific comes first and the torch
        reference -- which is always available -- comes last.

    ``prepare(layer, device)``
        The tensors this backend reads, on that device. The layer registers them as
        non-persistent buffers and removes them again when another backend is selected,
        so a backend gets exactly its own state and never sees another's.

    ``__call__(layer, key, value, query_scaled)``
        Evaluate. Reads the prepared state back off the layer by name.
    """

    name = "?"

    @classmethod
    def available(cls, layer: "NeighborhoodAttentionS2", device: torch.device) -> bool:
        raise NotImplementedError

    def prepare(self, layer: "NeighborhoodAttentionS2", device: torch.device) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def __call__(self, layer: "NeighborhoodAttentionS2", key: torch.Tensor, value: torch.Tensor, query_scaled: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RaggedOptimizedBackend(AttentionBackendS2):
    """
    The CUDA ragged kernels.

    Takes the arc form of the neighbourhood and the ring tables, and no column list:
    the kernel derives a neighbour's index by counting along an arc, so the columns
    would be several MB of tensor it never reads. At nside=64 that is 3.6 MB against
    2.8 MB for everything else the layer holds.
    """

    name = "ragged-optimized"

    @classmethod
    def available(cls, layer: "NeighborhoodAttentionS2", device: torch.device) -> bool:
        # there is no CPU ragged kernel; see optimized/kernels_cpu/ragged/README.md
        return layer.ragged and layer.optimized_kernel and device.type == "cuda"

    def prepare(self, layer: "NeighborhoodAttentionS2", device: torch.device) -> Dict[str, torch.Tensor]:
        arcs = layer._neighborhood_arcs()
        return {
            "psi_seg": arcs.segments.contiguous().to(device),
            "psi_seg_off": arcs.offsets.contiguous().to(device),
            "psi_ring_base": arcs.ring_base.contiguous().to(device),
            "psi_ring_size": arcs.ring_size.contiguous().to(device),
        }

    def __call__(self, layer, key, value, query_scaled):
        # the kernel also returns the softmax bookkeeping its backward consumes; only
        # the output is the layer's result
        return _neighborhood_s2_attention_ragged_optimized(
            key,
            value,
            query_scaled,
            layer.ring_weights,
            layer.psi_seg,
            layer.psi_seg_off,
            layer.psi_ring_base,
            layer.psi_ring_size,
            layer.num_heads,
            layer.npoints_out,
        )[0]


class RaggedReferenceBackend(AttentionBackendS2):
    """
    The torch reference, in terms of an explicit column list.

    The fallback for a ragged grid: it asks nothing of the device, so it is what a CPU
    module gets and what any device without the kernels gets. It walks a neighbour list
    in Python, which is what makes it readable and also what makes it slow -- it is the
    specification, not a fast path.
    """

    name = "ragged-reference"

    @classmethod
    def available(cls, layer: "NeighborhoodAttentionS2", device: torch.device) -> bool:
        return layer.ragged

    def prepare(self, layer: "NeighborhoodAttentionS2", device: torch.device) -> Dict[str, torch.Tensor]:
        col_idx, roff_idx = precompute_neighborhood_csr_s2(layer.grid_in, layer.grid_out, layer.theta_cutoff)
        return {"psi_col_idx": col_idx.to(device), "psi_roff_idx": roff_idx.to(device)}

    def __call__(self, layer, key, value, query_scaled):
        return _neighborhood_s2_attention_ragged_torch(
            key,
            value,
            query_scaled,
            layer.point_weights,
            layer.psi_col_idx,
            layer.psi_roff_idx,
            layer.num_heads,
            layer.npoints_out,
        )


class _RegularBackend(AttentionBackendS2):
    """
    Shared state for the product-grid backends.

    The optimized and reference regular paths prepare identically, which is not an
    accident of this refactor: they share one operator schema, because the CPU kernels
    and the torch reference both consume the column list while the CUDA kernels consume
    the arcs, and one schema has to carry both. So unlike the ragged pair there is no
    memory to be saved by choosing between them -- the whole point of splitting them
    here is that the *call* differs, and that distributed can later substitute a third
    preparation that is sharded rather than global.
    """

    @classmethod
    def available(cls, layer: "NeighborhoodAttentionS2", device: torch.device) -> bool:
        raise NotImplementedError

    def prepare(self, layer: "NeighborhoodAttentionS2", device: torch.device) -> Dict[str, torch.Tensor]:
        arcs = layer._neighborhood_arcs()
        col_idx, roff_idx = arcs.to_csr()
        col_idx = col_idx.contiguous()
        roff_idx = roff_idx.to(torch.int64).contiguous()
        # the row index is the expansion of the offsets, not a sort order -- see the
        # distributed layer, which rebuilds its own by neighbour count
        row_idx = torch.repeat_interleave(torch.arange(roff_idx.numel() - 1, dtype=torch.int64), roff_idx.diff()).contiguous()
        return {
            "psi_row_idx": row_idx.to(device),
            "psi_col_idx": col_idx.to(device),
            "psi_roff_idx": roff_idx.to(device),
            "psi_seg": arcs.segments.contiguous().to(device),
            "psi_seg_off": arcs.offsets.contiguous().to(device),
        }

    def _args(self, layer, key, value, query_scaled):
        return (
            key,
            value,
            query_scaled,
            layer.ring_weights,
            layer.psi_col_idx,
            layer.psi_roff_idx,
            layer.psi_seg,
            layer.psi_seg_off,
            layer.num_heads,
            layer.nlon_in,
            layer.nlat_out,
            layer.nlon_out,
        )


class RegularOptimizedBackend(_RegularBackend):
    """The compiled product-grid kernels. Unlike the ragged pair these exist for CPU too."""

    name = "regular-optimized"

    @classmethod
    def available(cls, layer: "NeighborhoodAttentionS2", device: torch.device) -> bool:
        return not layer.ragged and layer.optimized_kernel

    def __call__(self, layer, key, value, query_scaled):
        return _neighborhood_s2_attention_regular_optimized(*self._args(layer, key, value, query_scaled))


class RegularReferenceBackend(_RegularBackend):
    """The product-grid torch reference: the fallback when the kernels were not built."""

    name = "regular-reference"

    @classmethod
    def available(cls, layer: "NeighborhoodAttentionS2", device: torch.device) -> bool:
        return not layer.ragged

    def __call__(self, layer, key, value, query_scaled):
        return _neighborhood_s2_attention_regular_torch(*self._args(layer, key, value, query_scaled))


#: Every backend, most specific first. Order *is* the decision procedure: the first
#: whose ``available`` accepts the layer and device wins, so a backend narrows the case
#: by returning False rather than by sitting at a particular depth of a tree.
#:
#: The axes that actually decide this are not a fixed three. Grid family, whether the
#: kernels were compiled, and the device are the ones in play today; the regular path
#: adds direction (downsample vs upsample), the distributed path adds dense vs ring, a
#: FlexAttention backend would add a torch version, and DISCO's kpacked equivalent turns
#: on dtype and SM version together. A nested tree has to be reshaped each time one of
#: those appears. A predicate does not -- which is the whole reason the condition lives
#: on the backend instead of in the layer.
#:
#: Ordering resolves overlap, as in any dispatcher: put the narrower backend first. The
#: last entry must accept anything the layer can be built with, or selection raises.
BACKENDS = (
    RaggedOptimizedBackend,
    RaggedReferenceBackend,
    RegularOptimizedBackend,
    RegularReferenceBackend,
    # The distributed ring path is not here yet: it still slices a global pattern in
    # DistributedNeighborhoodAttentionS2 and opts out via _backend_managed. It becomes
    # a backend whose prepare() takes a shard, which is what lets the global pattern
    # never be built at all.
)
