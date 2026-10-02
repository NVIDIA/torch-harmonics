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

import contextlib
import math
import os
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.distributed as dist
from packaging import version

import torch_harmonics.distributed as thd
from torch_harmonics import GridS2, as_grid
from torch_harmonics.grid import _GRID_REGISTRY


def _is_sm90():
    """Return True when the default CUDA device is Hopper (SM 9.0)."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 9


def _is_sm100():
    """Return True when the default CUDA device is Blackwell (SM 10.0)."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 10


def regular_grid_types():
    """
    Registered grid families whose descriptors are :class:`RegularGridS2`.

    Most of the library addresses a field as a dense ``(nlat, nlon)`` array and is
    guarded by ``require_regular_grid``, so a test that sweeps "every grid" means every
    grid those routines accept -- and constructs them with ``nlat``/``nlon``, which a
    ragged family does not take.

    Derived from the registry by subclass rather than by listing names, so a grid family
    added later lands on the correct side of this without anyone remembering to come
    back. HEALPix is excluded here and exercised where it is actually supported, which
    today is attention; as other backends gain ragged support their tests should sweep
    the full registry instead of this.
    """
    from torch_harmonics.grid import _GRID_REGISTRY, RegularGridS2

    return tuple(name for name, cls in _GRID_REGISTRY.items() if issubclass(cls, RegularGridS2))


def set_seed(seed=333):
    """Set the torch + CUDA random seed.

    The ``seed`` argument can be overridden by the ``TORCH_HARMONICS_TEST_SEED_OVERRIDE``
    environment variable. Useful for distinguishing statistical (precision)
    failures from systematic (bug) failures: if a test fails with seed=333 but
    passes with seed=666 (and vice versa across rows), the failure is
    statistical — the per-row data distribution happens to land just over the
    AMP noise floor. A systematic bug fails consistently across seeds.

    Usage:
        TORCH_HARMONICS_TEST_SEED_OVERRIDE=666 pytest tests/test_convolution.py ...
    """
    env_seed = os.environ.get("TORCH_HARMONICS_TEST_SEED_OVERRIDE")
    if env_seed is not None:
        seed = int(env_seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return


def disable_tf32():
    # the api for this was changed lately in pytorch
    if torch.cuda.is_available():
        if version.parse(torch.__version__) >= version.parse("2.9.0"):
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.fp32_precision = "ieee"
            torch.backends.cudnn.conv.fp32_precision = "ieee"
            torch.backends.cudnn.rnn.fp32_precision = "ieee"
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    return


@contextlib.contextmanager
def maybe_autocast(device_type, dtype):
    """Unified precision context for tests that parameterize over multiple dtypes.

    - fp16 / bf16 → torch.autocast(device_type, dtype).
    - fp32 / fp64 → no-op. The caller is expected to have already cast module + inputs
      to the target dtype; PyTorch's autocast only supports fp16/bf16 as target dtypes,
      and entering it with fp32/fp64 triggers a "target dtype is not supported" warning.
    """
    if dtype in (torch.float16, torch.bfloat16):
        with torch.autocast(device_type=device_type, dtype=dtype):
            yield
    else:
        yield


def setup_distributed_context(ctx):
    ctx.world_rank = int(os.getenv("WORLD_RANK", 0))
    ctx.grid_size_h = int(os.getenv("GRID_H", 1))
    ctx.grid_size_w = int(os.getenv("GRID_W", 1))
    port = int(os.getenv("MASTER_PORT", "29501"))
    master_address = os.getenv("MASTER_ADDR", "localhost")
    ctx.world_size = ctx.grid_size_h * ctx.grid_size_w

    if torch.cuda.is_available():
        if ctx.world_rank == 0:
            print("Running test on GPU")
        local_rank = ctx.world_rank % torch.cuda.device_count()
        ctx.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
        proc_backend = "nccl"
    else:
        if ctx.world_rank == 0:
            print("Running test on CPU")
        ctx.device = torch.device("cpu")
        proc_backend = "gloo"

    init_kwargs = dict(
        backend=proc_backend,
        init_method=f"tcp://{master_address}:{port}",
        rank=ctx.world_rank,
        world_size=ctx.world_size,
    )
    if torch.cuda.is_available():
        init_kwargs["device_id"] = ctx.device
    dist.init_process_group(**init_kwargs)

    ctx.wrank = ctx.world_rank % ctx.grid_size_w
    ctx.hrank = ctx.world_rank // ctx.grid_size_w

    ctx.w_group = None
    ctx.h_group = None

    wgroups = []
    for w in range(0, ctx.world_size, ctx.grid_size_w):
        start = w
        end = w + ctx.grid_size_w
        wgroups.append(list(range(start, end)))

    if ctx.world_rank == 0:
        print("w-groups:", wgroups)
    for grp in wgroups:
        if len(grp) == 1:
            continue
        tmp_group = dist.new_group(ranks=grp)
        if ctx.world_rank in grp:
            ctx.w_group = tmp_group

    hgroups = [sorted(list(i)) for i in zip(*wgroups)]

    if ctx.world_rank == 0:
        print("h-groups:", hgroups)
    for grp in hgroups:
        if len(grp) == 1:
            continue
        tmp_group = dist.new_group(ranks=grp)
        if ctx.world_rank in grp:
            ctx.h_group = tmp_group

    if ctx.world_rank == 0:
        print(f"Running distributed tests on grid H x W = {ctx.grid_size_h} x {ctx.grid_size_w}")

    thd.init(ctx.h_group, ctx.w_group)
    # gloo on a CPU-only host gives every rank a plain "cpu" device, whose index is
    # None -- set_device would reject it and take the whole module down at setup.
    if ctx.device.type == "cuda":
        torch.cuda.set_device(ctx.device.index)

    return


def teardown_distributed_context(ctx):
    device_ids = [ctx.device.index] if ctx.device.type == "cuda" else None
    dist.barrier(device_ids=device_ids)
    thd.finalize()
    dist.destroy_process_group()

    return


def setup_class_from_context(cls, ctx_dict):

    ctx = ctx_dict.get("ctx", None)

    if ctx is None:
        raise ValueError("Context not found")

    cls.device = ctx.device
    cls.world_rank = ctx.world_rank
    cls.grid_size_h = ctx.grid_size_h
    cls.grid_size_w = ctx.grid_size_w
    cls.h_group = ctx.h_group
    cls.w_group = ctx.w_group
    cls.hrank = ctx.hrank
    cls.wrank = ctx.wrank


def setup_module(ctx_dict={}):
    # set up once per module
    class _Dummy:
        pass

    ctx = _Dummy()
    setup_distributed_context(ctx)
    ctx_dict["ctx"] = ctx


def teardown_module(ctx_dict):
    ctx = ctx_dict.get("ctx")
    if ctx:
        teardown_distributed_context(ctx)
    ctx_dict.clear()


def split_tensor_dim(tensor, dim=-2, dimsize=1, dimrank=0):
    """Split tensor along dim according to process grid ranks in that dim."""
    with torch.no_grad():
        if dimsize > 1:
            tensor_list_local = thd.split_tensor_along_dim(tensor, dim=dim, num_chunks=dimsize)
            tensor_local = tensor_list_local[dimrank]
        else:
            tensor_local = tensor
    return tensor_local


def split_tensor_hw(tensor, hdim=-2, wdim=-1, hsize=1, wsize=1, hrank=0, wrank=0):
    """Split tensor along height/width according to process grid ranks."""
    with torch.no_grad():
        tensor_local = split_tensor_dim(tensor, dim=wdim, dimsize=wsize, dimrank=wrank)
        tensor_local = split_tensor_dim(tensor_local, dim=hdim, dimsize=hsize, dimrank=hrank)

    return tensor_local


def gather_tensor_hw(tensor, hdim=-2, wdim=-1, hshapes=[], wshapes=[], hsize=1, wsize=1, hrank=0, wrank=0, hgroup=None, wgroup=None):
    """Gather tensor along height/width according to process grid ranks and shapes."""
    with torch.no_grad():
        tensor = tensor.contiguous()
        if wsize > 1:
            local_shape = list(tensor.shape)
            gather_shapes = []
            for w in wshapes:
                local_shape[wdim] = w
                gather_shapes.append(tuple(local_shape))
            olist = [torch.empty(shape, dtype=tensor.dtype, device=tensor.device) for shape in gather_shapes]
            olist[wrank] = tensor
            dist.all_gather(olist, tensor, group=wgroup)
            tensor = torch.cat(olist, dim=wdim)

        if hsize > 1:
            local_shape = list(tensor.shape)
            gather_shapes = []
            for h in hshapes:
                local_shape[hdim] = h
                gather_shapes.append(tuple(local_shape))
            olist = [torch.empty(shape, dtype=tensor.dtype, device=tensor.device) for shape in gather_shapes]
            olist[hrank] = tensor
            dist.all_gather(olist, tensor, group=hgroup)
            tensor = torch.cat(olist, dim=hdim)

    return tensor


def reduce_success(success, device, group=None):
    """All-reduce a per-rank boolean check result with logical AND.

    Returns the *global* verdict on every rank so that all ranks assert
    consistently: a failure on any single rank fails the test on all ranks.
    This avoids one rank raising (and bailing out of the test) while the
    others are still waiting on a subsequent collective -- which would
    otherwise hang the job at the next all-gather / teardown barrier.

    Only the reporting (e.g. ``compare_tensors(verbose=...)``) should be
    rank-0-gated; the assert itself must run on every rank using the value
    returned here.
    """
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return bool(success)
    flag = torch.tensor([1 if success else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def compare_tensors(msg, tensor1, tensor2, atol=1e-8, rtol=1e-5, verbose=False):

    # some None checks
    if tensor1 is None and tensor2 is None:
        allclose = True
    elif tensor1 is None and tensor2 is not None:
        allclose = False
        if verbose:
            print("tensor1 is None and tensor2 is not None")
    elif tensor1 is not None and tensor2 is None:
        allclose = False
        if verbose:
            print("tensor1 is not None and tensor2 is None")
    elif not (tensor1.is_floating_point() or tensor1.is_complex()) and not (tensor2.is_floating_point() or tensor2.is_complex()):
        # integers of any width (or bools): exact, and no mean/relative error, which
        # integer tensors do not support
        allclose = torch.all(tensor1 == tensor2)
        if not allclose and verbose:
            diff = torch.abs(tensor1 - tensor2)
            print(f"Element values with max difference on {msg}: {tensor1.flatten()[diff.argmax()]} and {tensor2.flatten()[diff.argmax()]}")
    else:
        diff = torch.abs(tensor1 - tensor2)
        abs_diff = torch.mean(diff, dim=0)
        rel_diff = torch.mean(diff / torch.clamp(torch.abs(tensor2), min=1e-6), dim=0)
        allclose = torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol)
        if not allclose and verbose:
            print(f"Absolute difference on {msg}: min = {abs_diff.min()}, mean = {abs_diff.mean()}, max = {abs_diff.max()}")
            print(f"Relative difference on {msg}: min = {rel_diff.min()}, mean = {rel_diff.mean()}, max = {rel_diff.max()}")
            print(f"Element values with max difference on {msg}: {tensor1.flatten()[diff.argmax()]} and {tensor2.flatten()[diff.argmax()]}")
            # find violating entry
            worst_diff = torch.argmax(diff - (atol + rtol * torch.abs(tensor2)))
            diff_bad = diff.flatten()[worst_diff].item()
            tensor2_abs_bad = torch.abs(tensor2).flatten()[worst_diff].item()
            print(f"Worst allclose condition violation: {diff_bad} <= {atol} + {rtol} * {tensor2_abs_bad} = {atol + rtol * tensor2_abs_bad}")

    return allclose


def build_psi_segments(col_idx: torch.Tensor, roff_idx: torch.Tensor, nlon: int):
    """
    Re-express a column list as contiguous longitude arcs, by brute force.

    A test oracle. The library computes the arcs natively in
    :func:`torch_harmonics.neighborhood.precompute_neighborhood_arcs_s2`; this recovers
    them from a column list instead, sharing none of that code, so the two can be held
    against each other.

    psi's sparsity is a union of arcs: for a given output row and input latitude, the
    neighbor longitudes are contiguous on the circle (possibly wrapping). This is
    geometric -- a geodesic ball meets a latitude circle in one arc -- and is pinned by
    TestPsiArcStructure.

    That lets a kernel iterate (hi, lo, len) segments and derive each neighbor's column
    by counting, instead of loading it from col_idx and recovering hi with a 64-bit
    integer division. The GPU has no integer divide instruction, so that division costs
    ~70-100 emulated instructions per neighbor against roughly four instructions of
    useful math; profiling showed the forward kernel at 80% compute throughput while
    delivering ~2.4% of peak FLOPs.

    Returns
    -------
    seg : int32 tensor of shape (nsegs, 3), columns (hi, lo, len)
    seg_off : int32 tensor of shape (nrows + 1,), row -> segment range

    Notes
    -----
    Relies on col_idx being sorted ascending within each row, which is how both
    _precompute_convolution_tensor_s2 and NeighborhoodArcsS2.to_csr emit it. A wrapping arc therefore appears as
    two runs at the ends of the sorted list, which is handled explicitly.
    """

    col = col_idx.cpu().to(torch.int64)
    roff = roff_idx.cpu().to(torch.int64)
    nrows = roff.numel() - 1

    seg_rows = []
    segs = []
    for row in range(nrows):
        beg, end = int(roff[row]), int(roff[row + 1])
        n_before = len(segs)
        if end > beg:
            cols = col[beg:end]
            hi = torch.div(cols, nlon, rounding_mode="floor")
            wi = cols - hi * nlon
            for h in torch.unique(hi):
                w = torch.unique(wi[hi == h]).sort().values
                count = int(w.numel())
                lo, hi_w = int(w[0]), int(w[-1])
                if hi_w - lo + 1 == count:
                    # plain arc
                    start, length = lo, count
                else:
                    # wraps the seam: sorted as [0..a] u [b..nlon-1]; the arc starts at
                    # b, which is one past the single interior gap
                    gaps = torch.diff(w)
                    split = int(torch.argmax(gaps))
                    start = int(w[split + 1])
                    length = count
                segs.append((int(h), start, length))
        seg_rows.append(len(segs) - n_before)

    seg = torch.tensor(segs, dtype=torch.int32).reshape(-1, 3)
    seg_off = torch.zeros(nrows + 1, dtype=torch.int32)
    seg_off[1:] = torch.tensor(seg_rows, dtype=torch.int32).cumsum(0)
    return seg, seg_off


def expand_psi_segments(seg: torch.Tensor, seg_off: torch.Tensor, nlon: int):
    """Expand segments back to a per-row column list. Inverse of build_psi_segments,
    used to verify the two representations describe the same sparsity."""

    out = []
    for row in range(seg_off.numel() - 1):
        cols = []
        for s in range(int(seg_off[row]), int(seg_off[row + 1])):
            hi, lo, length = (int(x) for x in seg[s])
            for j in range(length):
                cols.append(hi * nlon + (lo + j) % nlon)
        out.append(sorted(cols))
    return out


@dataclass(frozen=True, eq=False)
class _ProductGridAsRagged(GridS2):
    """
    An equiangular grid presented through the ragged interface.

    Every ring carries the same number of longitudes, so this *is* a product grid --
    it simply declines to say so, which routes it down the ragged path. That makes the
    two implementations comparable on identical geometry: any difference between them
    is the implementation, since the points, the weights and the neighbourhood are the
    same tensors either way.

    It is a test fixture rather than a library grid because nothing in the library
    would want it: a real product grid should be a RegularGridS2 and take the faster
    path. Its whole purpose is to be the control in that comparison. Shared by the
    attention and DISCO tests, whose ragged paths it controls alike.
    """

    nlat: int
    nlon: int
    grid_type: ClassVar[str] = "test-product-as-ragged"

    @property
    def is_regular(self):
        # Deliberately false. GridS2 computes this from the geometry -- every ring the
        # same length means regular -- so a uniform grid cannot be ragged by accident,
        # and saying so here is the only way to route identical geometry down the other
        # path. That is the whole point of the fixture: the lie is the experiment.
        return False

    @property
    def nrings(self):
        return self.nlat

    @property
    def nlon_per_lat(self):
        return torch.full((self.nlat,), self.nlon, dtype=torch.int64)

    @property
    def colats(self):
        return as_grid("equiangular", nlat=self.nlat, nlon=self.nlon).colats

    @property
    def colat_weights(self):
        return as_grid("equiangular", nlat=self.nlat, nlon=self.nlon).colat_weights

    def lons(self, ilat=None):
        return torch.arange(self.nlon, dtype=torch.float64) * (2.0 * math.pi / self.nlon)


# Defining a GridS2 subclass with a grid_type registers it, and the registry is global:
# left in place this fixture would be swept up by every test elsewhere that parameterizes
# over grid_types() and constructs with nlat/nlon. It is only ever built directly, by name
# it has no business being discoverable, so it is withdrawn immediately -- the class object
# keeps working, only as_grid("test-product-as-ragged") stops resolving.
_GRID_REGISTRY.pop(_ProductGridAsRagged.grid_type, None)
