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
from typing import List

import torch
import torch.distributed as dist

from ._amp_utils import _custom_setup_context
from .utils import azimuth_group, azimuth_group_size, is_distributed_azimuth, is_distributed_polar, polar_group, polar_group_rank, polar_group_size
from .utils import config as thd_config


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
def compute_split_shapes(size: int, num_chunks: int) -> List[int]:
    r"""
    Compute balanced chunk sizes for distributing a dimension across ranks.

    Divides ``size`` elements into ``num_chunks`` pieces that differ by at most
    one element.  The first ``size % num_chunks`` chunks receive one extra
    element; the remaining chunks get the base size ``size // num_chunks``.

    This is used internally by every distributed module to determine how
    latitudes, longitudes, and spectral modes are partitioned across process
    groups.

    Parameters
    ----------
    size : int
        Total number of elements to split (e.g.\ ``nlat`` or ``nlon``).
    num_chunks : int
        Number of chunks (typically the process-group size).

    Returns
    -------
    List[int]
        Per-rank chunk sizes, ordered by rank.

    Raises
    ------
    RuntimeError
        If ``size < num_chunks`` (every chunk must be non-empty).

    Examples
    --------
    >>> from torch_harmonics.distributed import compute_split_shapes
    >>> compute_split_shapes(256, 4)
    [64, 64, 64, 64]
    >>> compute_split_shapes(128, 3)
    [43, 43, 42]
    >>> compute_split_shapes(10, 4)
    [3, 3, 2, 2]
    """

    torch._check(size >= num_chunks, lambda: f"Cannot split {size} elements into {num_chunks} chunks; every chunk must be non-empty.")

    base, remainder = divmod(size, num_chunks)
    return [base + 1] * remainder + [base] * (num_chunks - remainder)


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

    torch._check(dim < tensor.dim(), lambda: f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}")
    torch._check(
        tensor.shape[dim] >= num_chunks, lambda: f"Error, cannot split dim {dim} of size {tensor.shape[dim]} into {num_chunks} chunks. Empty slices are currently not supported."
    )

    # get split
    sections = compute_split_shapes(tensor.shape[dim], num_chunks)
    tensor_list = torch.split(tensor, sections, dim=dim)

    return tensor_list


def flatten_and_pad_leading_dims(tensor: torch.Tensor, min_leading_size: int, num_trailing_dims: int = 2):
    """Collapse all but the trailing ``num_trailing_dims`` dims into a single leading dim, padding it to at least ``min_leading_size``.

    The distributed (S)HT redistributes this leading ("channel/batch") axis across
    the process grid via all-to-all transposes, which require every rank to receive
    a non-empty chunk -- i.e. the leading dim must be at least the (largest) group
    size. Uneven splits are fine (e.g. 5 elements over 4 ranks -> [2, 1, 1, 1]), so
    we only pad when the leading dim is *smaller* than the group size, never up to a
    multiple of it. Since the transforms are linear, zero-padding leaves the real
    entries untouched; :func:`unpad_and_unflatten_leading_dims` restores the original
    layout afterwards.

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
    lead_shape = tensor.shape[:-num_trailing_dims]
    tensor = tensor.reshape(-1, *tensor.shape[-num_trailing_dims:])
    lead_size = tensor.shape[0]

    if lead_size < min_leading_size:
        # new_zeros preserves dtype (incl. complex) and device
        zeros = tensor.new_zeros((min_leading_size - lead_size, *tensor.shape[1:]))
        tensor = torch.cat([tensor, zeros], dim=0)

    return tensor.contiguous(), lead_shape, lead_size


def unpad_and_unflatten_leading_dims(tensor: torch.Tensor, lead_shape, lead_size: int, num_trailing_dims: int = 2) -> torch.Tensor:
    """Inverse of :func:`flatten_and_pad_leading_dims`: drop the padding rows and restore the leading dims.

    The trailing ``num_trailing_dims`` dims are taken from ``tensor`` as-is, so this is
    valid even when the transform changed them (e.g. ``nlat, nlon`` -> ``lmax, mmax``).
    ``num_trailing_dims`` must match the value passed to the flatten call.
    """
    tensor = tensor.narrow(0, 0, lead_size)
    return tensor.reshape(*lead_shape, *tensor.shape[-num_trailing_dims:]).contiguous()


def _transpose(tensor, dim0, dim1, dim1_split_sizes, group=None, async_op=False, verify_shapes=None):

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

    # split and local transposition
    tsplit = split_tensor_along_dim(tensor, num_chunks=comm_size, dim=dim0)
    x_send = [y.contiguous() for y in tsplit]
    x_send_shapes = [x.shape for x in x_send]
    x_recv = []
    x_shape = list(x_send_shapes[comm_rank])
    for dim1_len in dim1_split_sizes:
        x_shape[dim1] = dim1_len
        x_recv.append(torch.empty(x_shape, dtype=tensor.dtype, device=tensor.device))

    # global transposition
    req = dist.all_to_all(x_recv, x_send, group=group, async_op=async_op)

    # get dim0 split sizes
    dim0_split_sizes = [x[dim0] for x in x_send_shapes]

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
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(x, dims, dim1_split_sizes):
        x = x.contiguous()

        if not is_distributed_azimuth():
            return x

        xlist, _, _ = _transpose(x, dims[0], dims[1], dim1_split_sizes, group=azimuth_group())
        return torch.cat(xlist, dim=dims[1]).contiguous()

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        x, dims, _ = inputs
        ctx.dims = dims
        if is_distributed_azimuth():
            ctx.dim0_split_sizes = compute_split_shapes(x.shape[dims[0]], azimuth_group_size())

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, go):
        go = go.contiguous()

        if not is_distributed_azimuth():
            return go, None, None

        dims = ctx.dims
        dim0_split_sizes = ctx.dim0_split_sizes
        gilist, _, _ = _transpose(go, dims[1], dims[0], dim0_split_sizes, group=azimuth_group())
        gi = torch.cat(gilist, dim=dims[0]).contiguous()
        return gi, None, None


class _DistributeTransposePolar(torch.autograd.Function):

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(x, dims, dim1_split_sizes):
        x = x.contiguous()

        if not is_distributed_polar():
            return x

        xlist, _, _ = _transpose(x, dims[0], dims[1], dim1_split_sizes, group=polar_group())
        return torch.cat(xlist, dim=dims[1]).contiguous()

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        x, dims, _ = inputs
        ctx.dims = dims
        if is_distributed_polar():
            ctx.dim0_split_sizes = compute_split_shapes(x.shape[dims[0]], polar_group_size())

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, go):
        go = go.contiguous()

        if not is_distributed_polar():
            return go, None, None

        dims = ctx.dims
        dim0_split_sizes = ctx.dim0_split_sizes
        gilist, _, _ = _transpose(go, dims[1], dims[0], dim0_split_sizes, group=polar_group())
        gi = torch.cat(gilist, dim=dims[0]).contiguous()
        return gi, None, None


# we need those additional primitives for distributed matrix multiplications
def _reduce(input_, use_fp32=True, group=None):

    # Bypass the function if we are using only 1 GPU.
    if dist.get_world_size(group=group) == 1:
        return input_

    # dist.all_reduce is in-place on its tensor argument; the .clone() forces a fresh
    # buffer so we never mutate the caller's tensor (which would alias an autograd-
    # tracked value when the input is fp32 + contiguous).
    if use_fp32:
        dtype = input_.dtype
        inputf_ = input_.float().contiguous().clone()
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
    dist.all_gather(input_list, input_, group=group)

    # concatenate along dim
    output = torch.cat(input_list, dim=dim_)

    return output


def _reduce_scatter(input_, dim_, use_fp32=True, group=None):
    """Reduce-scatter along ``dim_`` across the given group.

    Handles uneven splits (compute_split_shapes can produce per-rank chunk
    sizes that differ by one) by padding each chunk to max_chunk before the
    collective and trimming the per-rank output after. NCCL's reduce_scatter
    requires equal-sized chunks across ranks; without padding the call
    would silently read past short chunks' allocations, which manifests as
    corruption that surfaces many steps later as ``invalid memory address``
    errors in unrelated kernels.
    """

    # Bypass the function if we are using only 1 GPU.
    if dist.get_world_size(group=group) == 1:
        return input_

    comm_size = dist.get_world_size(group=group)
    comm_rank = dist.get_rank(group=group)

    # Per-rank chunk sizes from the natural (possibly uneven) split.
    orig_shapes = compute_split_shapes(input_.shape[dim_], comm_size)
    max_chunk = max(orig_shapes)
    my_chunk = orig_shapes[comm_rank]
    is_even = min(orig_shapes) == max_chunk

    dtype = input_.dtype
    work_dtype = torch.float32 if (use_fp32 and dtype != torch.float32) else dtype

    if is_even:
        # Fast path: all chunks the same size; call reduce_scatter directly.
        input_list = split_tensor_along_dim(input_, dim_, comm_size)
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
    chunks = list(split_tensor_along_dim(input_, dim_, comm_size))
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
    def symbolic(graph, input_):
        return input_

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
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
            return _reduce(grad_output, group=polar_group())
        else:
            return grad_output


class _CopyToAzimuthRegion(torch.autograd.Function):

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
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
            return _reduce(grad_output, group=azimuth_group())
        else:
            return grad_output


class _ScatterToPolarRegion(torch.autograd.Function):

    @staticmethod
    def symbolic(graph, input_, dim_):
        return _split(input_, dim_, group=polar_group())

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
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
            return _gather(grad_output, ctx.dim, ctx.split_shapes, polar_group()), None
        else:
            return grad_output, None


class _GatherFromPolarRegion(torch.autograd.Function):

    @staticmethod
    def symbolic(graph, input_, dim_, shapes_):
        return _gather(input_, dim_, shapes_, polar_group())

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
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
            return _split(grad_output, ctx.dim, group=polar_group()), None, None
        else:
            return grad_output, None, None


class _ReduceFromPolarRegion(torch.autograd.Function):

    @staticmethod
    def symbolic(graph, input_):
        if is_distributed_polar():
            return _reduce(input_, group=polar_group())
        else:
            return input_

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
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
        return grad_output


class _ReduceFromAzimuthRegion(torch.autograd.Function):

    @staticmethod
    def symbolic(graph, input_):
        if is_distributed_azimuth():
            return _reduce(input_, group=azimuth_group())
        else:
            return input_

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
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
        return grad_output


class _ReduceFromScatterToPolarRegion(torch.autograd.Function):

    @staticmethod
    def symbolic(graph, input_, dim_):
        if is_distributed_polar():
            return _reduce_scatter(input_, dim_, group=polar_group())
        else:
            return input_

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(input_, dim_):
        if is_distributed_polar():
            return _reduce_scatter(input_, dim_, group=polar_group())
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
            return _gather(grad_output, ctx.dim, ctx.split_shapes, polar_group()), None
        else:
            return grad_output, None


class _ReduceFromScatterToAzimuthRegion(torch.autograd.Function):
    """Fused reduce_scatter on the azimuth group: forward sums partial values
    across azimuth ranks and scatters along ``dim_``; backward is all_gather."""

    @staticmethod
    def symbolic(graph, input_, dim_):
        if is_distributed_azimuth():
            return _reduce_scatter(input_, dim_, group=azimuth_group())
        else:
            return input_

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
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
            return _gather(grad_output, ctx.dim, ctx.split_shapes, azimuth_group()), None
        else:
            return grad_output, None


class _GatherFromCopyToPolarRegion(torch.autograd.Function):

    @staticmethod
    def symbolic(graph, input_, dim_, shapes_):
        if is_distributed_polar():
            return _gather(input_, dim_, shapes_, polar_group())
        else:
            return input_

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
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
            return _reduce_scatter(grad_output, ctx.dim, use_fp32=True, group=polar_group()), None, None
        else:
            return grad_output, None, None


@torch.compiler.disable()
def distributed_transpose_azimuth(input_, dims_, shapes_):
    """All-to-all transpose across the azimuth process group.

    Redistributes ``input_`` so that data sharded along ``dims_[0]`` becomes
    sharded along ``dims_[1]``.  This is the core communication pattern used
    when switching between spatial and spectral partitioning of the longitude
    axis.

    Parameters
    ----------
    input_ : torch.Tensor
        Input tensor, partitioned along ``dims_[0]``.
    dims_ : tuple[int, int]
        ``(source_dim, target_dim)`` — the dimension to scatter from and the
        dimension to gather into.
    shapes_ : list[int]
        Per-rank sizes along ``dims_[1]`` (i.e. the expected receive sizes).

    Returns
    -------
    torch.Tensor
        Transposed tensor, now partitioned along ``dims_[1]``.
    """
    return _DistributeTransposeAzimuth.apply(input_, dims_, shapes_)


@torch.compiler.disable()
def distributed_transpose_polar(input_, dims_, shapes_):
    """All-to-all transpose across the polar process group.

    Same semantics as :func:`distributed_transpose_azimuth` but operates on
    the polar (latitudinal) process group.

    Parameters
    ----------
    input_ : torch.Tensor
        Input tensor, partitioned along ``dims_[0]``.
    dims_ : tuple[int, int]
        ``(source_dim, target_dim)``.
    shapes_ : list[int]
        Per-rank sizes along ``dims_[1]``.

    Returns
    -------
    torch.Tensor
        Transposed tensor, now partitioned along ``dims_[1]``.
    """
    return _DistributeTransposePolar.apply(input_, dims_, shapes_)


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
def reduce_from_scatter_to_polar_region(input_, dim_):
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

    Returns
    -------
    torch.Tensor
        Reduced and scattered tensor (local chunk only).
    """
    return _ReduceFromScatterToPolarRegion.apply(input_, dim_)


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
class _PolarHaloExchangeFn(torch.autograd.Function):
    """Differentiable lat halo exchange for polar-distributed tensors.

    Forward: gathers r_lat rows from neighbouring polar ranks and returns a
             halo-padded tensor of shape [B, C, H_local + 2*r_lat, W].
    Backward: communicates halo gradient contributions back to their owning
              ranks and accumulates them onto the local input gradient.

    Ranks at the polar boundary (rank 0 / rank group_size-1) receive
    zero-padding on the missing side in the forward pass; the corresponding
    halo-gradient portion is discarded in the backward (no neighbour to send
    it to), which is the correct adjoint of padding with zeros.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(x, r_lat):

        if not is_distributed_polar():
            return x

        group_size = polar_group_size()
        group_rank = polar_group_rank()
        prev_rank, next_rank = get_group_neighbors(polar_group())

        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        # setup send buffers
        send_top = x[:, :, :r_lat, :].contiguous()  # top r_lat rows → rank-1
        send_bot = x[:, :, -r_lat:, :].contiguous()  # bottom r_lat rows → rank+1

        # setup recv buffers
        recv_top = torch.zeros(B, C, r_lat, W, device=device, dtype=dtype)
        recv_bot = torch.zeros(B, C, r_lat, W, device=device, dtype=dtype)

        ops = []
        if group_rank > 0:
            ops.append(dist.P2POp(dist.isend, send_top, prev_rank, polar_group()))
            ops.append(dist.P2POp(dist.irecv, recv_top, prev_rank, polar_group()))
        if group_rank < group_size - 1:
            ops.append(dist.P2POp(dist.isend, send_bot, next_rank, polar_group()))
            ops.append(dist.P2POp(dist.irecv, recv_bot, next_rank, polar_group()))

        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        return torch.cat([recv_top, x, recv_bot], dim=2).contiguous()

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        x, r_lat = inputs
        ctx.r_lat = r_lat
        ctx.H = x.shape[2]
        if is_distributed_polar():
            ctx.group_size = polar_group_size()
            ctx.group_rank = polar_group_rank()
            prev_rank, next_rank = get_group_neighbors(polar_group())
            ctx.prev_rank = prev_rank
            ctx.next_rank = next_rank

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout):

        if not is_distributed_polar():
            return dout, None

        r_lat = ctx.r_lat
        group_size = ctx.group_size
        group_rank = ctx.group_rank
        H = ctx.H
        prev_rank = ctx.prev_rank
        next_rank = ctx.next_rank

        B, C, _, W = dout.shape
        device, dtype = dout.device, dout.dtype

        # Direct gradient for the local (non-halo) rows.
        dx = dout[:, :, r_lat : r_lat + H, :].contiguous().clone()

        # The halo slices carry gradients that belong to neighbouring ranks:
        #   dout[:, :, :r_lat, :]       → came FROM rank-1; send gradient back to rank-1
        #   dout[:, :, r_lat + H:, :]   → came FROM rank+1; send gradient back to rank+1
        # Simultaneously receive from each neighbour the gradient they owe us
        # for the rows we sent them in the forward pass.
        send_to_prev = dout[:, :, :r_lat, :].contiguous()
        send_to_next = dout[:, :, r_lat + H :, :].contiguous()

        recv_from_prev = torch.zeros(B, C, r_lat, W, device=device, dtype=dtype)
        recv_from_next = torch.zeros(B, C, r_lat, W, device=device, dtype=dtype)

        ops = []
        if group_rank > 0:
            ops.append(dist.P2POp(dist.isend, send_to_prev, prev_rank, polar_group()))
            ops.append(dist.P2POp(dist.irecv, recv_from_prev, prev_rank, polar_group()))
        if group_rank < group_size - 1:
            ops.append(dist.P2POp(dist.isend, send_to_next, next_rank, polar_group()))
            ops.append(dist.P2POp(dist.irecv, recv_from_next, next_rank, polar_group()))

        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # Accumulate gradient contributions for rows we sent in the forward.
        # recv_from_prev = gradient for our top r_lat rows (sent as prev rank's recv_bot)
        # recv_from_next = gradient for our bottom r_lat rows (sent as next rank's recv_top)
        if group_rank > 0:
            dx[:, :, :r_lat, :] = dx[:, :, :r_lat, :] + recv_from_prev
        if group_rank < group_size - 1:
            dx[:, :, H - r_lat :, :] = dx[:, :, H - r_lat :, :] + recv_from_next

        # Gradients for r_lat is None (not tensors / non-differentiable)
        return dx, None


@torch.compiler.disable()
def polar_halo_exchange(x, r_lat):
    """Exchange ``r_lat`` halo rows with neighbouring polar ranks.

    Gathers ``r_lat`` latitude rows from each polar neighbour and returns a
    halo-padded tensor.  Boundary ranks receive zero-padding on the missing
    side.  The operation is fully differentiable: the backward pass sends
    halo gradients back to their owning ranks and accumulates them.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape ``(B, C, H_local, W)``.
    r_lat : int
        Number of halo rows to exchange on each side.

    Returns
    -------
    torch.Tensor
        Halo-padded tensor of shape ``(B, C, H_local + 2 * r_lat, W)``.
    """
    return _PolarHaloExchangeFn.apply(x, r_lat)
