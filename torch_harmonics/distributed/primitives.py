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

from dataclasses import dataclass
from itertools import accumulate
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from torch_harmonics.partition import compute_split_shapes
from torch_harmonics.quadrature import latitude_support_band
from torch_harmonics.utils import check

from ._amp_utils import _custom_fwd, _custom_setup_context
from .utils import _point_shapes, _ring_shapes, _start_polar_group, azimuth_group, azimuth_group_size, is_distributed_azimuth, is_distributed_polar, polar_group, polar_group_size
from .utils import config as thd_config


@torch.compiler.disable()
def get_group_neighbors(group):
    """Return the ``(prev_rank, next_rank)`` global ranks of the immediate neighbours in ``group``.

    Ranks wrap around cyclically, so rank 0's predecessor is the last rank in
    the group.

    Parameters
    ----------
    group : torch.distributed.ProcessGroup
        The process group to query.

    Returns
    -------
    tuple[int, int]
        ``(prev_rank, next_rank)`` as global ranks.
    """
    group_size = dist.get_world_size(group)
    global_rank = dist.get_rank()
    group_ranks = dist.get_process_group_ranks(group)
    my_rank_id = group_ranks.index(global_rank)
    prev_rank = group_ranks[(my_rank_id - 1) % group_size]
    next_rank = group_ranks[(my_rank_id + 1) % group_size]

    return prev_rank, next_rank


def _check_shapes(msg, shapes_gather, shapes_expected):
    for idx, (size_gather, size_expected) in enumerate(zip(shapes_gather, shapes_expected)):
        if size_gather != size_expected:
            raise ValueError(f"{msg} shapes are not correct. Expected {size_expected}, got {size_gather} for index {idx}. Please check that the number of chunks is correct.")


# helper routine to compute uneven splitting in balanced way:
def split_tensor_along_dim(tensor, dim, num_chunks):
    r"""
    Split a tensor along a given dimension into balanced chunks.

    Uses :func:`compute_split_shapes` to determine chunk sizes, so the split
    is consistent with the partitioning used by all distributed modules in
    torch-harmonics.  Chunk sizes differ by at most one element.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to split.
    dim : int
        The dimension along which to split.
    num_chunks : int
        Number of chunks (typically the process-group size).

    Returns
    -------
    tuple[torch.Tensor, ...]
        A tuple of ``num_chunks`` tensor views.

    Raises
    ------
    RuntimeError
        If ``dim`` is out of range or ``tensor.shape[dim] < num_chunks``.

    Examples
    --------
    >>> import torch
    >>> from torch_harmonics.distributed import split_tensor_along_dim
    >>> x = torch.arange(10).unsqueeze(0)          # shape (1, 10)
    >>> parts = split_tensor_along_dim(x, dim=1, num_chunks=3)
    >>> [p.shape[1] for p in parts]
    [4, 3, 3]
    """

    check(dim < tensor.dim(), lambda: f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}")
    check(tensor.shape[dim] >= num_chunks, lambda: f"Error, cannot split dim {dim} of size {tensor.shape[dim]} into {num_chunks} chunks. Empty slices are currently not supported.")

    # get split
    sections = compute_split_shapes(tensor.shape[dim], num_chunks)
    tensor_list = torch.split(tensor, sections, dim=dim)

    return tensor_list


def flatten_and_pad_leading_dims(tensor: torch.Tensor, min_leading_size: int, num_trailing_dims: int = 2):
    """Collapse all but the trailing ``num_trailing_dims`` dims into a single leading dim, padding it to at least ``min_leading_size``.

    The distributed transforms split this leading (batch/channel) axis across ranks,
    so every rank needs at least one entry. The axis is zero-padded only when it is
    shorter than ``min_leading_size``; uneven splits are allowed.
    :func:`unpad_and_unflatten_leading_dims` restores the original layout.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor whose last ``num_trailing_dims`` dims are the transform dims (everything
        before them is flattened into the leading axis).
    min_leading_size : int
        Minimum size the flattened leading dim must reach. Pass
        ``max(comm_size_polar, comm_size_azimuth)`` -- both transpose directions split
        this same axis, so it must be at least as large as the larger group.
    num_trailing_dims : int
        Number of trailing dims to keep intact. ``2`` for the scalar SHT (``nlat, nlon``);
        ``3`` for the vector SHT (``2, nlat, nlon``), so the component axis is preserved.

    Returns
    -------
    tensor : torch.Tensor
        Flattened (and possibly zero-padded) contiguous tensor with shape
        ``(M_pad, *trailing)``.
    lead_shape : torch.Size
        The original leading dims, used to restore the shape later.
    lead_size : int
        The true (pre-pad) flattened leading size, used to slice off the padding.
    """
    # complex inputs (the inverse SHTs call this on their spectral input) are processed as
    # their real view, see unpad_and_unflatten_leading_dims for the rationale
    is_complex = tensor.is_complex()
    if is_complex:
        tensor = torch.view_as_real(tensor)
        num_trailing_dims += 1

    lead_shape = tensor.shape[:-num_trailing_dims]
    tensor = tensor.reshape(-1, *tensor.shape[-num_trailing_dims:])
    lead_size = tensor.shape[0]

    if lead_size < min_leading_size:
        # new_zeros preserves dtype and device
        zeros = tensor.new_zeros((min_leading_size - lead_size, *tensor.shape[1:]))
        tensor = torch.cat([tensor, zeros], dim=0)

    tensor = tensor.contiguous()

    return (torch.view_as_complex(tensor) if is_complex else tensor), lead_shape, lead_size


def unpad_and_unflatten_leading_dims(tensor: torch.Tensor, lead_shape, lead_size: int, num_trailing_dims: int = 2) -> torch.Tensor:
    """Inverse of :func:`flatten_and_pad_leading_dims`: drop the padding rows and restore the leading dims.

    The trailing ``num_trailing_dims`` dims are taken from ``tensor`` as-is, so this is
    valid even when the transform changed them (e.g. ``nlat, nlon`` -> ``lmax, mmax``).
    ``num_trailing_dims`` must match the value passed to the flatten call.
    """

    # complex inputs (the forward SHTs call this on their spectral output) are processed as
    # their real view: this is pure layout code touching only the leading dims, and a copy
    # over a complex buffer cannot be codegen'd by inductor -- triton has no complex type,
    # so it fails with KeyError: 'complex64'. view_as_real appends the re/im pair as a
    # trailing dim, which is simply carried along; the contiguous() right before the
    # view_as_complex supplies the unit last-dim stride the latter requires.
    is_complex = tensor.is_complex()
    if is_complex:
        tensor = torch.view_as_real(tensor)
        num_trailing_dims += 1

    tensor = tensor.narrow(0, 0, lead_size)
    out = tensor.reshape(*lead_shape, *tensor.shape[-num_trailing_dims:]).contiguous()

    return torch.view_as_complex(out) if is_complex else out


def _needs_equal_shapes(group) -> bool:
    """
    Whether the group's backend requires every rank's tensor to have the same shape.

    gloo's all_gather and all_to_all reject tensor lists whose entries differ in shape, and
    the decompositions here split unevenly whenever an extent does not divide by the group
    size. NCCL handles such lists natively, so its calls must stay untouched: the padded
    path below is for gloo alone, and only when the shapes actually differ.
    """
    return dist.get_backend(group) == dist.Backend.GLOO


def _pad_to(tensor: torch.Tensor, sizes: dict) -> torch.Tensor:
    """Zero-pad ``tensor`` at the end of each dimension ``d`` in ``sizes`` up to ``sizes[d]``."""
    for dim, size in sizes.items():
        missing = size - tensor.shape[dim]
        if missing > 0:
            pad_shape = list(tensor.shape)
            pad_shape[dim] = missing
            tensor = torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=dim)
    return tensor.contiguous()


def _empty_padded(like: torch.Tensor, sizes: dict) -> torch.Tensor:
    """An uninitialised receive buffer shaped like ``like``, but ``sizes[d]`` long along each dimension ``d`` in ``sizes``."""
    shape = list(like.shape)
    for dim, size in sizes.items():
        shape[dim % like.dim()] = size
    return like.new_empty(shape)


def _trim_to(tensor: torch.Tensor, sizes: dict) -> torch.Tensor:
    """The leading ``sizes[d]`` entries of ``tensor`` along each dimension ``d`` in ``sizes``; the inverse of :func:`_pad_to`."""
    slicer = [slice(None)] * tensor.dim()
    for dim, size in sizes.items():
        slicer[dim] = slice(0, size)
    return tensor[tuple(slicer)].contiguous()


def _transpose(tensor, dim0, dim1, dim1_split_sizes, group=None, async_op=False, verify_shapes=None, dim0_split_sizes=None):

    if verify_shapes is None:
        verify_shapes = thd_config.debug

    # get comm params
    comm_size = dist.get_world_size(group=group)
    comm_rank = dist.get_rank(group=group)

    if comm_size == 1:
        return [tensor], [tensor.size(dim0)], None

    # verify_shapes: check that dim1_split_sizes are correct:
    if verify_shapes:
        dim0_size = tensor.size(dim0)
        stens = torch.as_tensor([tensor.size(dim1)], dtype=torch.int64, device=tensor.device)
        stens_gather = [torch.empty_like(stens) for _ in range(comm_size)]
        stens_gather[comm_rank] = stens
        dist.all_gather(stens_gather, stens, group=group)
        sizes_gather = [stens.item() for stens in stens_gather]
        _check_shapes("_transpose: error, dim1_split_sizes", sizes_gather, dim1_split_sizes)

    # split and local transposition: the natural split of dim0, or the given one
    tsplit = _split_chunks(tensor, dim0, comm_size, dim0_split_sizes)
    x_send = [y.contiguous() for y in tsplit]
    x_send_shapes = [x.shape for x in x_send]
    x_recv = []
    x_shape = list(x_send_shapes[comm_rank])
    for dim1_len in dim1_split_sizes:
        x_shape[dim1] = dim1_len
        x_recv.append(torch.empty(x_shape, dtype=tensor.dtype, device=tensor.device))

    # get dim0 split sizes
    dim0_split_sizes = [x[dim0] for x in x_send_shapes]

    # global transposition
    uneven = len(set(dim0_split_sizes)) > 1 or len(set(dim1_split_sizes)) > 1
    if uneven and _needs_equal_shapes(group):
        # gloo: pad every chunk to the largest extent along both split dimensions, exchange,
        # and trim each received chunk back. The maxima come from the global split shapes,
        # so every rank pads identically. Synchronous, since the trim needs the data.
        max_sizes = {dim0: max(dim0_split_sizes), dim1: max(dim1_split_sizes)}
        x_send_padded = [_pad_to(x, max_sizes) for x in x_send]
        x_recv_padded = [_empty_padded(x, max_sizes) for x in x_recv]
        dist.all_to_all(x_recv_padded, x_send_padded, group=group)
        x_recv = [_trim_to(xp, {dim0: x.shape[dim0], dim1: x.shape[dim1]}) for xp, x in zip(x_recv_padded, x_recv)]
        req = None
    else:
        req = dist.all_to_all(x_recv, x_send, group=group, async_op=async_op)

    if verify_shapes:
        stens = torch.as_tensor([x_send[comm_rank].size(dim0)], dtype=torch.int64, device=tensor.device)
        stens_gather = [torch.empty_like(stens) for _ in range(comm_size)]
        stens_gather[comm_rank] = stens
        dist.all_gather(stens_gather, stens, group=group)
        sizes_gather = [stens.item() for stens in stens_gather]
        _check_shapes("_transpose: error, dim0_split_sizes", sizes_gather, dim0_split_sizes)
        if sum(sizes_gather) != dim0_size:
            raise ValueError(f"_transpose: error, dim0_split_sizes do not sum to the correct size. Expected {dim0_size}, got {torch.sum(sizes_gather)}")

    return x_recv, dim0_split_sizes, req


class _DistributeTransposeAzimuth(torch.autograd.Function):

    @staticmethod
    @_custom_fwd(device_type="cuda")
    def forward(x, dims, dim1_split_sizes, dim0_split_sizes=None):
        x = x.contiguous()

        if not is_distributed_azimuth():
            return x

        xlist, _, _ = _transpose(x, dims[0], dims[1], dim1_split_sizes, group=azimuth_group(), dim0_split_sizes=dim0_split_sizes)
        return torch.cat(xlist, dim=dims[1]).contiguous()

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        x, dims, dim1_split_sizes, dim0_split_sizes = inputs
        ctx.dims = dims
        if is_distributed_azimuth():
            if dim0_split_sizes is None:
                dim0_split_sizes = compute_split_shapes(x.shape[dims[0]], azimuth_group_size())
            ctx.dim0_split_sizes = list(dim0_split_sizes)
            ctx.dim1_split_sizes = list(dim1_split_sizes)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, go):
        go = go.contiguous()

        if not is_distributed_azimuth():
            return go, None, None, None

        # the reverse transpose: dim0 comes back whole in the chunks it was split into, and dim1
        # goes back out in the chunks it arrived in, which need not be the natural split
        dims = ctx.dims
        return _DistributeTransposeAzimuth.apply(go, (dims[1], dims[0]), ctx.dim0_split_sizes, ctx.dim1_split_sizes), None, None, None


class _DistributeTransposePolar(torch.autograd.Function):

    @staticmethod
    @_custom_fwd(device_type="cuda")
    def forward(x, dims, dim1_split_sizes, dim0_split_sizes=None):
        x = x.contiguous()

        if not is_distributed_polar():
            return x

        xlist, _, _ = _transpose(x, dims[0], dims[1], dim1_split_sizes, group=polar_group(), dim0_split_sizes=dim0_split_sizes)
        return torch.cat(xlist, dim=dims[1]).contiguous()

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        x, dims, dim1_split_sizes, dim0_split_sizes = inputs
        ctx.dims = dims
        if is_distributed_polar():
            if dim0_split_sizes is None:
                dim0_split_sizes = compute_split_shapes(x.shape[dims[0]], polar_group_size())
            ctx.dim0_split_sizes = list(dim0_split_sizes)
            ctx.dim1_split_sizes = list(dim1_split_sizes)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, go):
        go = go.contiguous()

        if not is_distributed_polar():
            return go, None, None, None

        # the reverse transpose: dim0 comes back whole in the chunks it was split into, and dim1
        # goes back out in the chunks it arrived in, which need not be the natural split
        dims = ctx.dims
        return _DistributeTransposePolar.apply(go, (dims[1], dims[0]), ctx.dim0_split_sizes, ctx.dim1_split_sizes), None, None, None


# we need those additional primitives for distributed matrix multiplications
# dtypes whose accumulation error a cross-rank reduction would make worse than the local
# one, so the collective is run in fp32 instead
_LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)


def _reduction_dtype(dtype, use_fp32):
    """Working dtype for a collective reduction.

    ``use_fp32`` promotes low-precision inputs so that accumulating across ranks does not
    lose precision the local reduction had kept. It must only ever *promote*: reducing an
    fp64 tensor in fp32 would silently cap the result at fp32 accuracy, which is the
    opposite of what the flag is for. Anything already at fp32 or wider is left alone.
    """

    if use_fp32 and dtype in _LOW_PRECISION_DTYPES:
        return torch.float32
    return dtype


def _reduce(input_, use_fp32=True, group=None):

    # Bypass the function if we are using only 1 GPU.
    if dist.get_world_size(group=group) == 1:
        return input_

    dtype = input_.dtype
    work_dtype = _reduction_dtype(dtype, use_fp32)

    # dist.all_reduce is in-place on its tensor argument; the .clone() forces a fresh
    # buffer so we never mutate the caller's tensor (which would alias an autograd-
    # tracked value when the input is fp32 + contiguous).
    if work_dtype != dtype:
        inputf_ = input_.to(work_dtype).contiguous().clone()
        dist.all_reduce(inputf_, group=group)
        input_ = inputf_.to(dtype)
    else:
        input_ = input_.contiguous().clone()
        dist.all_reduce(input_, group=group)

    return input_


def _split(input_, dim_, group=None):
    # Bypass the function if we are using only 1 GPU.
    comm_size = dist.get_world_size(group=group)
    if comm_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_dim(input_, dim_, comm_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = dist.get_rank(group=group)
    output = input_list[rank]

    return output


def _gather(input_, dim_, shapes_, group=None, verify_shapes=None):

    if verify_shapes is None:
        verify_shapes = thd_config.debug

    comm_size = dist.get_world_size(group=group)

    if comm_size == 1:
        return input_

    if (shapes_ is not None) and (len(shapes_) != comm_size):
        raise ValueError(f"Error, shapes_ is not correct. Expected {comm_size}, got {len(shapes_)}. Please check that the number of chunks is correct.")
    if dim_ >= input_.dim():
        raise ValueError(f"Error, dim_ is not correct. Expected {input_.dim()}, got {dim_}. Please check that the dimension is correct.")

    # verify shapes:
    if verify_shapes and shapes_ is not None:
        comm_rank = dist.get_rank(group=group)
        stens = torch.as_tensor([input_.size(dim_)], dtype=torch.int64, device=input_.device)
        stens_gather = [torch.empty_like(stens) for _ in range(comm_size)]
        stens_gather[comm_rank] = stens
        dist.all_gather(stens_gather, stens, group=group)
        sizes_gather = [stens.item() for stens in stens_gather]
        _check_shapes("_gather: error, shapes_", sizes_gather, shapes_)

    # make contiguous:
    input_ = input_.contiguous()
    input_shape = list(input_.shape)

    if shapes_ is None:
        # gather shapes across ranks
        comm_rank = dist.get_rank(group=group)
        stens = torch.as_tensor([input_.size(dim_)], dtype=torch.int64, device=input_.device)
        stens_gather = [torch.empty_like(stens) for _ in range(comm_size)]
        stens_gather[comm_rank] = stens
        dist.all_gather(stens_gather, stens, group=group)
        shapes_ = [stens.item() for stens in stens_gather]

    # now create the recv list
    input_list = []
    for src in range(comm_size):
        input_shape[dim_] = shapes_[src]
        input_list.append(torch.empty(input_shape, dtype=input_.dtype, device=input_.device))

    # gather data across ranks
    if len(set(shapes_)) > 1 and _needs_equal_shapes(group):
        # gloo: pad every rank's piece to the largest along dim_, gather, trim each back
        max_size = {dim_: max(shapes_)}
        gathered = [_empty_padded(x, max_size) for x in input_list]
        dist.all_gather(gathered, _pad_to(input_, max_size), group=group)
        input_list = [_trim_to(g, {dim_: size}) for g, size in zip(gathered, shapes_)]
    else:
        dist.all_gather(input_list, input_, group=group)

    # concatenate along dim
    output = torch.cat(input_list, dim=dim_)

    return output


def _split_chunks(input_, dim_, comm_size, shapes):
    """The per-rank chunks of ``input_`` along ``dim_``: the natural split, or ``shapes``."""
    if shapes is None:
        return split_tensor_along_dim(input_, dim_, comm_size)
    return torch.split(input_, [int(n) for n in shapes], dim=dim_)


def _reduce_scatter(input_, dim_, use_fp32=True, group=None, shapes=None):
    """Reduce-scatter along ``dim_`` across the given group.

    ``shapes`` are the per-rank chunk sizes along ``dim_``, by default the natural split of
    :func:`compute_split_shapes`. A HEALPix band of rings is cut by pixel count instead, so
    its chunks can differ by more than one.

    Handles uneven splits by padding each chunk to max_chunk before the
    collective and trimming the per-rank output after. NCCL's reduce_scatter
    requires equal-sized chunks across ranks; without padding the call
    would silently read past short chunks' allocations, which manifests as
    corruption that surfaces many steps later as ``invalid memory address``
    errors in unrelated kernels.

    ``use_fp32`` promotes fp16/bf16 inputs for the reduction, see
    :func:`_reduction_dtype`; wider dtypes are reduced as-is.
    """

    # Bypass the function if we are using only 1 GPU.
    if dist.get_world_size(group=group) == 1:
        return input_

    comm_size = dist.get_world_size(group=group)
    comm_rank = dist.get_rank(group=group)

    # Per-rank chunk sizes, by default from the natural (possibly uneven) split.
    if shapes is None:
        orig_shapes = compute_split_shapes(input_.shape[dim_], comm_size)
    else:
        orig_shapes = [int(n) for n in shapes]
        if len(orig_shapes) != comm_size or sum(orig_shapes) != input_.shape[dim_]:
            raise ValueError(f"shapes {orig_shapes} do not split dimension {dim_} of size {input_.shape[dim_]} over {comm_size} ranks")
    max_chunk = max(orig_shapes)
    my_chunk = orig_shapes[comm_rank]
    is_even = min(orig_shapes) == max_chunk

    dtype = input_.dtype
    work_dtype = _reduction_dtype(dtype, use_fp32)

    if is_even:
        # Fast path: all chunks the same size; call reduce_scatter directly.
        input_list = _split_chunks(input_, dim_, comm_size, shapes)
        if work_dtype != dtype:
            input_list = [x.to(work_dtype) for x in input_list]
        input_list = [x.contiguous() for x in input_list]
        output = torch.empty_like(input_list[comm_rank])
        dist.reduce_scatter(output, input_list, group=group)
        return output.to(dtype) if work_dtype != dtype else output

    # Uneven split: zero-pad every short chunk to max_chunk along ``dim_``,
    # run reduce_scatter on the now-equal-sized chunks, then trim back to
    # the local rank's true chunk size. Padding zeros sum to zero across
    # ranks and are discarded by the slice after the collective.
    chunks = list(_split_chunks(input_, dim_, comm_size, shapes))
    padded_chunks = []
    for c in chunks:
        if c.shape[dim_] < max_chunk:
            pad_shape = list(c.shape)
            pad_shape[dim_] = max_chunk - c.shape[dim_]
            pad = torch.zeros(pad_shape, dtype=c.dtype, device=c.device)
            c = torch.cat([c, pad], dim=dim_)
        if work_dtype != dtype:
            c = c.to(work_dtype)
        padded_chunks.append(c.contiguous())

    output = torch.empty_like(padded_chunks[comm_rank])
    dist.reduce_scatter(output, padded_chunks, group=group)

    if my_chunk < max_chunk:
        slicer = [slice(None)] * output.dim()
        slicer[dim_] = slice(0, my_chunk)
        output = output[tuple(slicer)].contiguous()

    return output.to(dtype) if work_dtype != dtype else output


class _CopyToPolarRegion(torch.autograd.Function):

    @staticmethod
    @_custom_fwd(device_type="cuda")
    def forward(input_):
        return input_

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        if is_distributed_polar():
            return _ReduceFromPolarRegion.apply(grad_output)
        else:
            return grad_output


class _CopyToAzimuthRegion(torch.autograd.Function):

    @staticmethod
    @_custom_fwd(device_type="cuda")
    def forward(input_):
        return input_

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        if is_distributed_azimuth():
            return _ReduceFromAzimuthRegion.apply(grad_output)
        else:
            return grad_output


class _ScatterToPolarRegion(torch.autograd.Function):

    @staticmethod
    @_custom_fwd(device_type="cuda")
    def forward(input_, dim_):
        if is_distributed_polar():
            return _split(input_, dim_, group=polar_group())
        else:
            return input_

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        input_, dim_ = inputs
        ctx.dim = dim_
        if is_distributed_polar():
            ctx.split_shapes = compute_split_shapes(input_.shape[dim_], polar_group_size())

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        if is_distributed_polar():
            return _GatherFromPolarRegion.apply(grad_output, ctx.dim, ctx.split_shapes), None
        else:
            return grad_output, None


class _GatherFromPolarRegion(torch.autograd.Function):

    @staticmethod
    @_custom_fwd(device_type="cuda")
    def forward(input_, dim_, shapes_):
        if is_distributed_polar():
            return _gather(input_, dim_, shapes_, group=polar_group())
        else:
            return input_

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        _, dim_, _ = inputs
        ctx.dim = dim_

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        if is_distributed_polar():
            return _ScatterToPolarRegion.apply(grad_output, ctx.dim), None, None
        else:
            return grad_output, None, None


class _ReduceFromPolarRegion(torch.autograd.Function):

    @staticmethod
    @_custom_fwd(device_type="cuda")
    def forward(input_):
        if is_distributed_polar():
            return _reduce(input_, group=polar_group())
        else:
            return input_

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        return _CopyToPolarRegion.apply(grad_output)


class _ReduceFromAzimuthRegion(torch.autograd.Function):

    @staticmethod
    @_custom_fwd(device_type="cuda")
    def forward(input_):
        if is_distributed_azimuth():
            return _reduce(input_, group=azimuth_group())
        else:
            return input_

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        return _CopyToAzimuthRegion.apply(grad_output)


class _ReduceFromScatterToPolarRegion(torch.autograd.Function):

    @staticmethod
    @_custom_fwd(device_type="cuda")
    def forward(input_, dim_, shapes_=None):
        if is_distributed_polar():
            return _reduce_scatter(input_, dim_, group=polar_group(), shapes=shapes_)
        else:
            return input_

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        input_, dim_, shapes_ = inputs
        ctx.dim = dim_
        if is_distributed_polar():
            ctx.split_shapes = compute_split_shapes(input_.shape[dim_], polar_group_size()) if shapes_ is None else list(shapes_)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        if is_distributed_polar():
            return _GatherFromCopyToPolarRegion.apply(grad_output, ctx.dim, ctx.split_shapes), None, None
        else:
            return grad_output, None, None


class _ReduceFromScatterToAzimuthRegion(torch.autograd.Function):
    """Fused reduce_scatter on the azimuth group: forward sums partial values
    across azimuth ranks and scatters along ``dim_``; backward is all_gather."""

    @staticmethod
    @_custom_fwd(device_type="cuda")
    def forward(input_, dim_):
        if is_distributed_azimuth():
            return _reduce_scatter(input_, dim_, group=azimuth_group())
        else:
            return input_

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        input_, dim_ = inputs
        ctx.dim = dim_
        if is_distributed_azimuth():
            ctx.split_shapes = compute_split_shapes(input_.shape[dim_], azimuth_group_size())

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        if is_distributed_azimuth():
            return _GatherFromCopyToAzimuthRegion.apply(grad_output, ctx.dim, ctx.split_shapes), None
        else:
            return grad_output, None


class _GatherFromCopyToPolarRegion(torch.autograd.Function):

    @staticmethod
    @_custom_fwd(device_type="cuda")
    def forward(input_, dim_, shapes_):
        if is_distributed_polar():
            return _gather(input_, dim_, shapes_, group=polar_group())
        else:
            return input_

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        _, dim_, shapes_ = inputs
        ctx.dim = dim_
        ctx.shapes = list(shapes_)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        if is_distributed_polar():
            # the adjoint scatters back the chunks the forward gathered, whatever their sizes
            return _ReduceFromScatterToPolarRegion.apply(grad_output, ctx.dim, ctx.shapes), None, None
        else:
            return grad_output, None, None


class _GatherFromCopyToAzimuthRegion(torch.autograd.Function):
    """All-gather along ``dim_`` across azimuth ranks; reduce-scatter in the backward pass.

    The adjoint of :class:`_ReduceFromScatterToAzimuthRegion`, which uses it as its backward.
    """

    @staticmethod
    @_custom_fwd(device_type="cuda")
    def forward(input_, dim_, shapes_):
        if is_distributed_azimuth():
            return _gather(input_, dim_, shapes_, group=azimuth_group())
        else:
            return input_

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        _, dim_, _ = inputs
        ctx.dim = dim_

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        if is_distributed_azimuth():
            return _ReduceFromScatterToAzimuthRegion.apply(grad_output, ctx.dim), None, None
        else:
            return grad_output, None, None


@torch.compiler.disable()
def distributed_transpose_azimuth(input_, dims_, shapes_, split_shapes_=None):
    """All-to-all transpose across the azimuth process group.

    Redistributes ``input_`` so that data sharded along ``dims_[1]`` becomes
    sharded along ``dims_[0]``: every rank scatters its ``dims_[0]`` in chunks and
    gathers the whole of ``dims_[1]``. This is the core communication pattern used
    when switching between spatial and spectral partitioning of the longitude
    axis.

    Parameters
    ----------
    input_ : torch.Tensor
        Input tensor, partitioned along ``dims_[1]``.
    dims_ : tuple[int, int]
        ``(scatter_dim, gather_dim)`` — the dimension to scatter and the
        dimension to gather.
    shapes_ : list[int]
        Per-rank sizes along ``dims_[1]`` (i.e. the expected receive sizes).
    split_shapes_ : list[int], optional
        Per-rank sizes to scatter ``dims_[0]`` in, by default its natural split
        (:func:`compute_split_shapes`).

    Returns
    -------
    torch.Tensor
        Transposed tensor, now partitioned along ``dims_[1]``.
    """
    return _DistributeTransposeAzimuth.apply(input_, dims_, shapes_, split_shapes_)


@torch.compiler.disable()
def distributed_transpose_polar(input_, dims_, shapes_, split_shapes_=None):
    """All-to-all transpose across the polar process group.

    Same semantics as :func:`distributed_transpose_azimuth` but operates on
    the polar (latitudinal) process group.

    Parameters
    ----------
    input_ : torch.Tensor
        Input tensor, partitioned along ``dims_[1]``.
    dims_ : tuple[int, int]
        ``(scatter_dim, gather_dim)``.
    shapes_ : list[int]
        Per-rank sizes along ``dims_[1]``.
    split_shapes_ : list[int], optional
        Per-rank sizes to scatter ``dims_[0]`` in, by default its natural split.
        A HEALPix band of rings is cut by pixel count, so its chunks can differ
        by more than one.

    Returns
    -------
    torch.Tensor
        Transposed tensor, now partitioned along ``dims_[0]``.
    """
    return _DistributeTransposePolar.apply(input_, dims_, shapes_, split_shapes_)


@torch.compiler.disable()
def copy_to_polar_region(input_):
    """Identity in the forward pass; all-reduce across polar ranks in the backward pass.

    Use this to broadcast a replicated tensor into a region where each polar
    rank will compute a partial result.  The backward pass sums the partial
    gradients so that the replicated parameter receives the correct total
    gradient.

    Parameters
    ----------
    input_ : torch.Tensor
        Replicated tensor (same value on every polar rank).

    Returns
    -------
    torch.Tensor
        Same tensor (forward is a no-op).
    """
    return _CopyToPolarRegion.apply(input_)


@torch.compiler.disable()
def copy_to_azimuth_region(input_):
    """Identity in the forward pass; all-reduce across azimuth ranks in the backward pass.

    Azimuth counterpart of :func:`copy_to_polar_region`.

    Parameters
    ----------
    input_ : torch.Tensor
        Replicated tensor (same value on every azimuth rank).

    Returns
    -------
    torch.Tensor
        Same tensor (forward is a no-op).
    """
    return _CopyToAzimuthRegion.apply(input_)


@torch.compiler.disable()
def reduce_from_polar_region(input_):
    """All-reduce across polar ranks in the forward pass; identity in the backward pass.

    Use this to aggregate partial results computed independently on each polar
    rank.

    Parameters
    ----------
    input_ : torch.Tensor
        Partial result on the local polar rank.

    Returns
    -------
    torch.Tensor
        Sum of ``input_`` across all polar ranks.
    """
    return _ReduceFromPolarRegion.apply(input_)


@torch.compiler.disable()
def reduce_from_azimuth_region(input_):
    """All-reduce across azimuth ranks in the forward pass; identity in the backward pass.

    Azimuth counterpart of :func:`reduce_from_polar_region`.

    Parameters
    ----------
    input_ : torch.Tensor
        Partial result on the local azimuth rank.

    Returns
    -------
    torch.Tensor
        Sum of ``input_`` across all azimuth ranks.
    """
    return _ReduceFromAzimuthRegion.apply(input_)


@torch.compiler.disable()
def scatter_to_polar_region(input_, dim_):
    """Split ``input_`` along ``dim_`` and keep only the local polar rank's chunk.

    The backward pass is an all-gather that reconstructs the full tensor.

    Parameters
    ----------
    input_ : torch.Tensor
        Full (non-partitioned) tensor.
    dim_ : int
        Dimension along which to scatter.

    Returns
    -------
    torch.Tensor
        The local rank's slice of the input.
    """
    return _ScatterToPolarRegion.apply(input_, dim_)


@torch.compiler.disable()
def gather_from_polar_region(input_, dim_, shapes_):
    """All-gather along ``dim_`` across polar ranks to reconstruct the full tensor.

    The backward pass is a split (scatter) that distributes gradients back to
    owning ranks.

    Parameters
    ----------
    input_ : torch.Tensor
        Local partition of the tensor.
    dim_ : int
        Dimension along which to gather.
    shapes_ : list[int]
        Per-rank sizes along ``dim_``.

    Returns
    -------
    torch.Tensor
        Fully gathered tensor.
    """
    return _GatherFromPolarRegion.apply(input_, dim_, shapes_)


@torch.compiler.disable()
def reduce_from_scatter_to_polar_region(input_, dim_, shapes_=None):
    """Fused reduce-scatter across polar ranks along ``dim_``.

    Equivalent to an all-reduce followed by keeping only the local rank's
    chunk, but performed in a single collective for efficiency.  The backward
    pass is an all-gather.

    Parameters
    ----------
    input_ : torch.Tensor
        Tensor with partial contributions from the local rank.
    dim_ : int
        Dimension along which to scatter after reducing.
    shapes_ : list[int], optional
        Per-rank sizes along ``dim_``, by default the natural split of
        :func:`compute_split_shapes`.

    Returns
    -------
    torch.Tensor
        Reduced and scattered tensor (local chunk only).
    """
    return _ReduceFromScatterToPolarRegion.apply(input_, dim_, shapes_)


@torch.compiler.disable()
def reduce_from_scatter_to_azimuth_region(input_, dim_):
    """Fused reduce-scatter across azimuth ranks along ``dim_``.

    Azimuth counterpart of :func:`reduce_from_scatter_to_polar_region`.

    Parameters
    ----------
    input_ : torch.Tensor
        Tensor with partial contributions from the local rank.
    dim_ : int
        Dimension along which to scatter after reducing.

    Returns
    -------
    torch.Tensor
        Reduced and scattered tensor (local chunk only).
    """
    return _ReduceFromScatterToAzimuthRegion.apply(input_, dim_)


@torch.compiler.disable()
def gather_from_copy_to_polar_region(input_, dim_, shapes_):
    """All-gather along ``dim_`` across polar ranks; reduce-scatter in the backward pass.

    Similar to :func:`gather_from_polar_region`, but the backward pass uses
    reduce-scatter instead of split, making it the adjoint of a copy-then-gather
    pattern.

    Parameters
    ----------
    input_ : torch.Tensor
        Local partition of the tensor.
    dim_ : int
        Dimension along which to gather.
    shapes_ : list[int]
        Per-rank sizes along ``dim_``.

    Returns
    -------
    torch.Tensor
        Fully gathered tensor.
    """
    return _GatherFromCopyToPolarRegion.apply(input_, dim_, shapes_)


# ---------------------------------------------------------------------------
# nearest neighbor exchange algorithms
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PolarHaloPlan:
    r"""
    What one polar rank exchanges with its two neighbours along the axis its shard splits.

    A halo exchange lets a rank read entries its neighbours own: it receives ``recv_prev``
    entries from the end of rank ``r - 1``'s block and ``recv_next`` from the start of rank
    ``r + 1``'s, and sends ``send_prev`` entries from the start of its own block to
    ``r - 1`` and ``send_next`` from its end to ``r + 1``. ``pad_prev`` and ``pad_next``
    zero entries go outside the received ones, so a layout can stay uniform across ranks
    where there is no neighbour to receive from. The halo-padded block is::

        [pad_prev zeros | recv_prev | own block | recv_next | pad_next zeros]

    Only immediate neighbours are involved, so every count is at most the extent the
    neighbour holds. The counts of two neighbours have to match: rank ``r``'s ``send_prev``
    is rank ``r - 1``'s ``recv_next``. The constructors build the plans of all ranks from
    the same global data, which guarantees that; a side whose counts are zero posts no
    message at all.

    An entry is whatever the axis indexes: a latitude row of a regular grid, or a pixel of
    the flat axis of a HEALPix field, whose bands of rings differ in length.

    Parameters
    ----------
    recv_prev, recv_next : int
        Entries received from the previous and the next polar rank.
    send_prev, send_next : int
        Entries sent to the previous and the next polar rank.
    pad_prev, pad_next : int
        Zero entries added before and after the received ones.
    """

    recv_prev: int = 0
    recv_next: int = 0
    send_prev: int = 0
    send_next: int = 0
    pad_prev: int = 0
    pad_next: int = 0

    def __post_init__(self):
        for name in ("recv_prev", "recv_next", "send_prev", "send_next", "pad_prev", "pad_next"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"PolarHaloPlan.{name} must be a non-negative integer, got {value!r}")

    @property
    def before(self) -> int:
        """Entries in front of the own block in the halo-padded one."""
        return self.pad_prev + self.recv_prev

    @property
    def after(self) -> int:
        """Entries behind the own block in the halo-padded one."""
        return self.recv_next + self.pad_next

    @property
    def is_identity(self) -> bool:
        """Whether the exchange leaves the block as it is and posts no message."""
        return self.before == 0 and self.after == 0 and self.send_prev == 0 and self.send_next == 0

    @classmethod
    def symmetric(cls, r_lat: int, rank: int, size: int) -> "PolarHaloPlan":
        """
        ``r_lat`` entries from each neighbour, zero-padded where there is none.

        Every rank's halo-padded block is then ``2 * r_lat`` longer than its own, at the
        poles too, which is the layout of the regular grids' localized operators. A single
        polar rank holds the whole axis and exchanges nothing, so its plan is the identity,
        as the exchange is without a polar split.

        Parameters
        ----------
        r_lat : int
            Entries borrowed from each neighbour.
        rank, size : int
            This rank's position in the polar group, and the group's size.

        Returns
        -------
        PolarHaloPlan
            The plan of ``rank``.
        """
        r_lat = int(r_lat)
        if size == 1:
            return cls()
        has_prev, has_next = rank > 0, rank < size - 1
        return cls(
            recv_prev=r_lat if has_prev else 0,
            recv_next=r_lat if has_next else 0,
            send_prev=r_lat if has_prev else 0,
            send_next=r_lat if has_next else 0,
            pad_prev=0 if has_prev else r_lat,
            pad_next=0 if has_next else r_lat,
        )

    @classmethod
    def from_needs(cls, shapes: List[int], needs: List[Tuple[int, int]], rank: int) -> "PolarHaloPlan":
        r"""
        The plan of ``rank`` when every rank names the range it has to read.

        Parameters
        ----------
        shapes : List[int]
            Entries every polar rank holds, ordered by rank. The blocks are contiguous and
            in rank order, so they tile ``[0, sum(shapes))``.
        needs : List[Tuple[int, int]]
            For every rank, the half-open global range ``[lo, hi)`` its operator reads. The
            part of it beyond the rank's own block has to lie within the neighbours' blocks;
            a range within the own block, or an empty one, needs no halo.
        rank : int
            The rank to build the plan of.

        Returns
        -------
        PolarHaloPlan
            Without padding: at the ends of the axis nothing is received, and the
            halo-padded block is exactly ``[lo, hi)``.

        Raises
        ------
        ValueError
            If any rank's range reaches past an immediate neighbour, or past either end of
            the axis. The exchange talks to immediate neighbours only, so such a range would
            need data from a rank further out; the decomposition is too fine for the operator
            then. Every rank checks every range, so all of them raise together.
        """
        size = len(shapes)
        if len(needs) != size:
            raise ValueError(f"needs one range per rank: got {len(needs)} for {size} ranks")
        if not 0 <= rank < size:
            raise ValueError(f"rank must lie in [0, {size}), got {rank}")

        starts = [0]
        for n in shapes:
            starts.append(starts[-1] + int(n))

        def overhang(r):
            lo, hi = (int(v) for v in needs[r])
            start, end = starts[r], starts[r + 1]
            if hi <= lo:
                # reads nothing beyond its block
                return 0, 0
            before, after = max(start - lo, 0), max(hi - end, 0)
            prev_extent = shapes[r - 1] if r > 0 else 0
            next_extent = shapes[r + 1] if r < size - 1 else 0
            if before > prev_extent or after > next_extent:
                raise ValueError(
                    f"polar rank {r} holds [{start}, {end}) and needs [{lo}, {hi}), which reaches past its "
                    f"immediate neighbours (blocks of {prev_extent} before and {next_extent} after; extents "
                    f"{list(shapes)}). The halo is exchanged with immediate neighbours only, so the polar "
                    "decomposition is too fine for the operator's support: use fewer polar ranks, a finer "
                    "grid or a smaller cutoff."
                )
            return before, after

        # every rank checks every range, not only its own and its neighbours': a range that
        # one rank refused and another accepted would leave the second waiting for a message
        overhangs = [overhang(r) for r in range(size)]
        before, after = overhangs[rank]
        send_prev = overhangs[rank - 1][1] if rank > 0 else 0
        send_next = overhangs[rank + 1][0] if rank < size - 1 else 0
        return cls(recv_prev=before, recv_next=after, send_prev=send_prev, send_next=send_next)


def _plan_ring_band_halo(shard, reach: List[Optional[Tuple[int, int]]], rank: int, per_ring: bool = False) -> Tuple[PolarHaloPlan, Tuple[int, int], Tuple[int, int]]:
    r"""
    The halo of an operator that reads whole rings, planned from the rings every rank reaches.

    The layer-side half of a halo over a ragged field, shared by the distributed layers that
    read their input through ring tables, such as neighborhood attention and DISCO on
    HEALPix. Each rank's band is its own rings, widened to the rings its operator reaches.
    Bands are whole rings because the kernels address a ring through its tables and wrap at
    its end, and the shard cuts on ring boundaries anyway. Every rank plans every rank's band
    from the same global data, which :meth:`PolarHaloPlan.from_needs` needs for the ranks to
    agree on their messages; the band in flat points is the halo-padded block the plan
    produces.

    Parameters
    ----------
    shard : GridShardS2
        The shard of the field the halo is exchanged on, split across the polar group only:
        whole rings of a HEALPix grid, or whole latitude rows of a regular one.
    reach : List[Optional[Tuple[int, int]]]
        For every polar rank, the half-open range ``[lo, hi)`` of global rings its operator
        reads, or ``None`` if it reads none.
    rank : int
        The rank to plan for.
    per_ring : bool, optional
        Plan the exchange in whole rings rather than flat points, for a field that keeps a
        latitude axis of its own, as a regular grid's ``(..., nlat, nlon)`` layout does. By
        default ``False``: the field is flat, as on HEALPix.

    Returns
    -------
    halo : PolarHaloPlan
        The plan of ``rank``, without padding, in rings if ``per_ring`` and points otherwise.
    ring_range : Tuple[int, int]
        ``rank``'s band as global rings ``[lo, hi)``.
    point_range : Tuple[int, int]
        ``rank``'s band as global flat points ``[lo, hi)``, the extent of the halo-padded block.

    Raises
    ------
    ValueError
        If a band reaches past an immediate neighbour, see :meth:`PolarHaloPlan.from_needs`.
    """
    ring_starts = list(accumulate([0] + _ring_shapes(shard)))
    # where every ring starts in flat order, and where the last one ends
    ring_bounds = shard.grid.lon_offsets.tolist()

    ring_bands = []
    for r in range(len(ring_starts) - 1):
        lo, hi = ring_starts[r], ring_starts[r + 1]
        if reach[r] is not None:
            lo, hi = min(lo, int(reach[r][0])), max(hi, int(reach[r][1]))
        ring_bands.append((lo, hi))
    needs = [(int(ring_bounds[lo]), int(ring_bounds[hi])) for lo, hi in ring_bands]

    halo = PolarHaloPlan.from_needs(_ring_shapes(shard), ring_bands, rank) if per_ring else PolarHaloPlan.from_needs(_point_shapes(shard), needs, rank)
    return halo, ring_bands[rank], needs[rank]


def _halo_slice(x, dim, start, n):
    """``n`` entries of ``x`` from ``start`` along ``dim``, contiguous for sending, or ``None`` if there are none."""
    return x.narrow(dim, start, n).contiguous() if n > 0 else None


def _halo_zeros(x, dim, n):
    """``n`` zero entries shaped like ``x`` along ``dim``, as padding or a receive buffer, or ``None`` if there are none."""
    if n == 0:
        return None
    shape = list(x.shape)
    shape[dim] = n
    return torch.zeros(shape, device=x.device, dtype=x.dtype)


def _halo_exchange_neighbours(sends, recvs):
    """
    Post the point-to-point messages of one halo exchange and wait for them.

    ``sends`` and ``recvs`` are ``(prev, next)`` pairs of tensors or ``None``. A side with
    nothing to move posts nothing: every message has to be matched by the neighbour, and an
    unmatched one could be delivered to a later exchange on the polar communicator.
    """
    prev_rank, next_rank = get_group_neighbors(polar_group())
    ops = []
    for peer, send, recv in ((prev_rank, sends[0], recvs[0]), (next_rank, sends[1], recvs[1])):
        if send is not None:
            ops.append(dist.P2POp(dist.isend, send, peer, polar_group()))
        if recv is not None:
            ops.append(dist.P2POp(dist.irecv, recv, peer, polar_group()))
    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()


def _halo_gather(x, plan: PolarHaloPlan, lat_dim=-2):
    r"""
    Pad ``x`` along ``lat_dim`` with the entries the plan borrows from the polar neighbours.

    ``[B, C, H, W]`` -> ``[B, C, plan.before + H + plan.after, W]`` for the default
    ``lat_dim``; any other dimension can be named, which is how a channels-last
    ``[B, H, W, C]`` tensor, or the flat pixel axis of a HEALPix field, is exchanged
    without a layout round trip.

    This is the linear map whose adjoint is :func:`_halo_reduce`. The two are used as each
    other's forward and backward by :class:`_PolarHaloExchangeFn` (gather forward) and
    :class:`_PolarHaloReduceFn` (reduce forward), so they must stay exact adjoints -- see
    the inner-product identity asserted in the distributed primitives tests.
    """

    # Nothing to borrow, so post nothing: every op below is a P2P the neighbour has to match.
    # Up to v0.9.3 the bottom slice was x[..., -r_lat:, :], which at r_lat == 0 is the whole
    # tensor; sent against the neighbour's 0-row receive, its surplus could be delivered to the
    # next halo receive on the polar communicator and corrupt a later layer. narrow() below no
    # longer degenerates that way, and a side with a zero count posts nothing. That can leave a
    # rank out of the exchange altogether, which NCCL refuses in a group's first P2P call, so
    # every rank starts the group first
    _start_polar_group(x.device)
    if plan.is_identity:
        return x

    lat_dim = lat_dim % x.dim()
    H = x.shape[lat_dim]
    if plan.send_prev > H or plan.send_next > H:
        raise ValueError(f"halo plan {plan} sends more entries than the {H} this rank holds along dim {lat_dim}")

    # the edges of the own block lent to the neighbours, and room for what they lend
    sends = (_halo_slice(x, lat_dim, 0, plan.send_prev), _halo_slice(x, lat_dim, H - plan.send_next, plan.send_next))
    recvs = (_halo_zeros(x, lat_dim, plan.recv_prev), _halo_zeros(x, lat_dim, plan.recv_next))
    _halo_exchange_neighbours(sends, recvs)

    parts = [_halo_zeros(x, lat_dim, plan.pad_prev), recvs[0], x, recvs[1], _halo_zeros(x, lat_dim, plan.pad_next)]
    return torch.cat([t for t in parts if t is not None], dim=lat_dim).contiguous()


def _halo_reduce(y, plan: PolarHaloPlan, lat_dim=-2):
    r"""
    Adjoint of :func:`_halo_gather`: send the halo entries of ``y`` home and accumulate them.

    ``[B, C, plan.before + H + plan.after, W]`` -> ``[B, C, H, W]`` for the default ``lat_dim``.
    The halo slices of ``y`` hold contributions to entries owned by the neighbours, so they are
    sent back and added onto the owner's entries; symmetrically, each rank receives the
    contributions its neighbours hold for the entries it lent them.

    The padding has no owner and is dropped, which is exactly the adjoint of
    :func:`_halo_gather` zero-padding there.
    """

    # no halo entries to send home; skip the P2P exchange entirely, as _halo_gather does, once
    # the group has been started by every rank
    _start_polar_group(y.device)
    if plan.is_identity:
        return y

    lat_dim = lat_dim % y.dim()
    H = y.shape[lat_dim] - plan.before - plan.after
    if H < max(plan.send_prev, plan.send_next):
        raise ValueError(f"halo plan {plan} does not fit a halo-padded extent of {y.shape[lat_dim]} along dim {lat_dim}")

    # the own entries pass straight through
    out = y.narrow(lat_dim, plan.before, H).contiguous().clone()

    # the received halo goes back where it came from, and what was lent comes back
    sends = (_halo_slice(y, lat_dim, plan.pad_prev, plan.recv_prev), _halo_slice(y, lat_dim, plan.before + H, plan.recv_next))
    recvs = (_halo_zeros(y, lat_dim, plan.send_prev), _halo_zeros(y, lat_dim, plan.send_next))
    _halo_exchange_neighbours(sends, recvs)

    if recvs[0] is not None:
        out.narrow(lat_dim, 0, plan.send_prev).add_(recvs[0])
    if recvs[1] is not None:
        out.narrow(lat_dim, H - plan.send_next, plan.send_next).add_(recvs[1])

    return out


class _PolarHaloExchangeFn(torch.autograd.Function):
    """Differentiable halo exchange for polar-distributed tensors.

    Forward: gathers the entries the plan borrows from the neighbouring polar ranks along
             ``lat_dim``, e.g. [B, C, H_local, W] -> [B, C, H_local + 2*r_lat, W]
             channels-first for a symmetric plan.
    Backward: communicates halo gradient contributions back to their owning
              ranks and accumulates them onto the local input gradient.

    The padding the plan adds where a rank has no neighbour holds zeros in the forward
    pass; the corresponding gradient is discarded in the backward, which is the correct
    adjoint of padding with zeros.
    """

    @staticmethod
    @_custom_fwd(device_type="cuda")
    def forward(x, plan, lat_dim):

        if not is_distributed_polar():
            return x

        return _halo_gather(x, plan, lat_dim)

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        _, plan, lat_dim = inputs
        ctx.plan = plan
        ctx.lat_dim = lat_dim

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout):

        if not is_distributed_polar():
            return dout, None, None

        # The halo slices of the incoming gradient belong to the neighbouring ranks, and each
        # neighbour holds the gradient it owes us for the entries we lent it: exactly the
        # adjoint of the gather performed in the forward.
        return _PolarHaloReduceFn.apply(dout, ctx.plan, ctx.lat_dim), None, None


class _PolarHaloReduceFn(torch.autograd.Function):
    """Adjoint of :class:`_PolarHaloExchangeFn`: halo reduce forward, halo gather backward.

    Forward: takes a halo-padded tensor whose halo entries hold contributions to entries
             owned by the neighbours, sends them home and accumulates, returning the own
             entries only along ``lat_dim`` -- ``[B, C, H_local + 2*r_lat, W]`` -> ``[B, C, H_local, W]``
             channels-first for a symmetric plan.
    Backward: gathers the halo back, the same operation the exchange performs forward.

    This is the direction a *scatter*-shaped operator needs: the transpose DISCO convolution
    writes into output latitudes that may belong to a neighbour, so its forward accumulates
    across the boundary rather than borrowing across it.
    """

    @staticmethod
    @_custom_fwd(device_type="cuda")
    def forward(y, plan, lat_dim):

        if not is_distributed_polar():
            return y

        return _halo_reduce(y, plan, lat_dim)

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        _, plan, lat_dim = inputs
        ctx.plan = plan
        ctx.lat_dim = lat_dim

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout):

        if not is_distributed_polar():
            return dout, None, None

        return _PolarHaloExchangeFn.apply(dout, ctx.plan, ctx.lat_dim), None, None


@torch.compiler.disable()
def compute_polar_halo_radius(
    colats_in: torch.Tensor,
    colats_out: torch.Tensor,
    theta_cutoff: float,
    lat_in_shapes: List[int],
    lat_out_shapes: List[int],
) -> int:
    r"""
    Latitude halo radius a polar shard needs from its neighbours, derived analytically.

    A localized operator couples an output latitude only to input latitudes within
    ``theta_cutoff``, a contiguous band given by :func:`~torch_harmonics.quadrature.latitude_support_band`.
    A polar rank owns a contiguous range of both axes, so the rows it must borrow are exactly
    those where the band of its output range runs past its own input range. This returns the
    largest such overhang over all ranks: the radius of the halo the regular grids' layers
    exchange, as :meth:`PolarHaloPlan.symmetric` lays it out.

    The result is computed from the geometry alone and is identical on every rank; no
    communication is involved.

    Parameters
    ----------
    colats_in : torch.Tensor
        Input colatitudes in radians, ascending, shape ``(nlat_in,)``.
    colats_out : torch.Tensor
        Output colatitudes in radians, ascending, shape ``(nlat_out,)``.
    theta_cutoff : float
        Effective angular support radius, i.e. including the widening the sparsity pattern is
        built with, see :func:`~torch_harmonics.quadrature.effective_theta_cutoff`.
    lat_in_shapes : List[int]
        Input latitudes held by each polar rank, ordered by rank.
    lat_out_shapes : List[int]
        Output latitudes held by each polar rank, ordered by rank.

    Returns
    -------
    int
        Halo radius in latitude rows; ``0`` when the polar group holds a single rank.

    Raises
    ------
    ValueError
        If the radius exceeds the smallest local input chunk. :func:`polar_halo_exchange`
        only communicates with immediate neighbours, so a halo wider than a neighbour's entire
        share would have to reach past it to the next rank out.
    """

    comm_size = len(lat_out_shapes)
    if comm_size <= 1:
        return 0

    lo, hi = latitude_support_band(colats_in, colats_out, theta_cutoff)

    in_starts = [0]
    out_starts = [0]
    for n in lat_in_shapes[:-1]:
        in_starts.append(in_starts[-1] + n)
    for n in lat_out_shapes[:-1]:
        out_starts.append(out_starts[-1] + n)

    r_lat = 0
    for rank in range(comm_size):
        out_lo, out_n = out_starts[rank], lat_out_shapes[rank]
        if out_n <= 0:
            continue
        in_lo, in_n = in_starts[rank], lat_in_shapes[rank]

        # the band is monotone in the output latitude, so the extremes of this rank's output
        # range bracket every input latitude it can reach
        need_lo = int(lo[out_lo])
        need_hi = int(hi[out_lo + out_n - 1])
        if need_hi < need_lo:
            continue

        r_lat = max(r_lat, in_lo - need_lo, need_hi - (in_lo + in_n - 1))

    r_lat = max(r_lat, 0)

    min_chunk = min(lat_in_shapes)
    if r_lat > min_chunk:
        raise ValueError(
            f"Latitude halo of {r_lat} rows exceeds the smallest local input chunk of {min_chunk} rows "
            f"(input latitudes split as {list(lat_in_shapes)} over {comm_size} polar ranks). The halo is "
            "exchanged with immediate neighbours only, so a halo this wide would have to reach past a "
            "neighbour to the rank beyond it. This means the polar decomposition is too fine for the "
            "angular cutoff: use fewer polar ranks, more latitudes, or a smaller theta_cutoff."
        )

    return r_lat


def _require_halo_plan(halo, r_lat, name: str) -> PolarHaloPlan:
    """The plan a halo function was given, or a TypeError naming the replacement for a radius."""
    if r_lat is not None:
        raise TypeError(
            f"{name} takes a PolarHaloPlan as halo since v1.0.0; r_lat is gone. For the symmetric halo of "
            f"radius {r_lat!r}, pass halo=PolarHaloPlan.symmetric({r_lat!r}, polar_group_rank(), polar_group_size())."
        )
    if not isinstance(halo, PolarHaloPlan):
        raise TypeError(
            f"{name} takes a PolarHaloPlan as halo since v1.0.0, got {halo!r}. For the symmetric halo of a "
            "radius r_lat, pass PolarHaloPlan.symmetric(r_lat, polar_group_rank(), polar_group_size())."
        )
    return halo


@torch.compiler.disable()
def polar_halo_exchange(x, halo=None, lat_dim=-2, *, r_lat=None):
    """Exchange halo entries with the neighbouring polar ranks.

    Gathers the entries this rank borrows from each polar neighbour along ``lat_dim``, as
    ``halo`` lays them out, and returns the halo-padded tensor. The operation is fully
    differentiable: the backward pass sends halo gradients back to their owning ranks and
    accumulates them.

    Every rank of the polar group has to call this together, also a rank whose plan moves
    nothing: before the first exchange after :func:`~torch_harmonics.distributed.init`, all
    of them join one collective on the polar group, since NCCL requires every member of a
    group in its first point-to-point call.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape ``(B, C, H_local, W)``, or any shape whose split axis is
        ``lat_dim``.
    halo : PolarHaloPlan
        What this rank receives from and sends to each neighbour, and how much it pads.
        :meth:`PolarHaloPlan.symmetric` gives ``r_lat`` rows on each side, zero-padded at
        the poles; :meth:`PolarHaloPlan.from_needs` counts that differ per side and per
        rank, as on the bands of rings of a HEALPix grid.
    lat_dim : int, optional
        The split axis, by default ``-2`` (channels-first). A channels-last
        ``(B, H_local, W, C)`` tensor passes ``1``, a flat HEALPix field ``-1``.

    Returns
    -------
    torch.Tensor
        Halo-padded tensor, ``halo.before`` entries longer in front and ``halo.after``
        behind along ``lat_dim``.

    Raises
    ------
    TypeError
        If ``halo`` is not a :class:`PolarHaloPlan`, or the removed ``r_lat`` is passed.
    """
    halo = _require_halo_plan(halo, r_lat, "polar_halo_exchange")
    if not is_distributed_polar():
        return x
    return _PolarHaloExchangeFn.apply(x, halo, lat_dim)


@torch.compiler.disable()
def polar_halo_reduce(y, halo=None, lat_dim=-2, *, r_lat=None):
    """Accumulate halo entries onto their owning polar ranks.

    The adjoint of :func:`polar_halo_exchange`, and its mirror image: where the exchange
    *borrows* entries from the neighbours before a gather-shaped operator reads them, this
    *returns* entries to the neighbours after a scatter-shaped operator has written them.
    The padding is dropped, matching the zeros the exchange puts there. Fully
    differentiable; the backward is the exchange.

    Parameters
    ----------
    y : torch.Tensor
        Halo-padded tensor, ``halo.before + nlat_local + halo.after`` long along
        ``lat_dim``.
    halo : PolarHaloPlan
        The halo ``y`` was padded with, as for :func:`polar_halo_exchange`.
    lat_dim : int, optional
        The split axis, by default ``-2`` (channels-first). A channels-last
        ``(B, H, W, C)`` tensor passes ``1``.

    Returns
    -------
    torch.Tensor
        This rank's completed entries along ``lat_dim``, ``nlat_local`` long.

    Raises
    ------
    TypeError
        If ``halo`` is not a :class:`PolarHaloPlan`, or the removed ``r_lat`` is passed.
    """
    halo = _require_halo_plan(halo, r_lat, "polar_halo_reduce")
    if not is_distributed_polar():
        return y
    return _PolarHaloReduceFn.apply(y, halo, lat_dim)
