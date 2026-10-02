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

# The operators behind the DISCO backends: fake kernels and autocast for the raw ops, the
# contraction and its transpose as custom ops with autograd, and the autograd node that
# fuses the contraction with the weight contraction. psi reaches them in the layouts
# built by torch_harmonics.disco._psi_layouts; which ones a layer holds is its backend's
# choice.

import functools
from typing import Optional

import torch
from disco_helpers import (
    kpacked_sm90_kernels_is_available,
    kpacked_sm100_kernels_is_available,
    optimized_kernels_is_available,
)

from torch_harmonics.utils import check

from .. import disco_kernels
from .._disco_utils import _compute_dtype


@functools.lru_cache(maxsize=None)
def _kpacked_build_available() -> bool:
    """Return True if this build contains any kpacked kernel at all.

    Purely a build-time question: BUILD_KPACKED_SM90 / SM100 come from whether
    TORCH_CUDA_ARCH_LIST asked for 9.0a / 10.0a. If neither is present no device
    can ever run the kpacked path, so the (padded, and at high resolution large)
    kpacked buffers need not be constructed.

    Deliberately NOT a device check. Modules are normally built on CPU and moved
    with .to(device) afterwards, so at construction time the runtime device is
    unknown; keying on it would disable kpacked for the ordinary flow. The device is
    checked when a backend is selected for it, see RegularKpackedBackend.available.
    """
    return kpacked_sm90_kernels_is_available() or kpacked_sm100_kernels_is_available()


# Minor versions the kpacked kernels are actually compiled for, mirroring the
# arch targets in setup.py (9.0a, and 10.0a / 10.3a). These cubins are
# arch-CONDITIONAL, not forward compatible: an sm_100a cubin will not load on
# sm_103 or sm_107. So the major version alone is not a sufficient test --
# Rubin reports 10.7 and would pass a `major == 10` check while no cubin in the
# build can run on it.
_KPACKED_SM90_MINORS = (0,)
_KPACKED_SM100_MINORS = (0, 3)


def _kpacked_supported_on_device(device_index: int) -> bool:
    """Return True if the kpacked kernel is supported on this CUDA device.

    SM_90a (Hopper)  — WGMMA path in disco_cuda_fwd_dense_kpacked_sm90.cu.
    SM_100a (Blackwell) — tcgen05 path in disco_cuda_fwd_dense_kpacked_sm100.cu.

    Checks the minor version too; see the note on the tables above. A device
    outside those targets falls back to the arc kernels, which are correct
    everywhere.
    """
    major, minor = torch.cuda.get_device_capability(device_index)
    if major == 9 and minor in _KPACKED_SM90_MINORS:
        return kpacked_sm90_kernels_is_available()
    if major == 10 and minor in _KPACKED_SM100_MINORS:
        return kpacked_sm100_kernels_is_available()
    return False


def _kpacked_k_pad(kernel_size: int, n_align: int = 8) -> Optional[int]:
    """The K padding the kpacked kernels would use, or None if they have no instantiation for it.

    The tensor-core kernels take k as the MMA's N dimension and are instantiated for
    N = 8 and N = 16 only, so a basis with more than 16 functions stays on the arc kernels.
    """
    k_pad = ((kernel_size + n_align - 1) // n_align) * n_align
    return k_pad if k_pad in (8, 16) else None


def _use_spatial_first_dgrad(out_per_group: int, in_per_group: int, kernel_size: int) -> bool:
    """
    Whether the input gradient should run the sparse transpose before the weight contraction.

    Spatial-first dgrad runs the sparse transpose over output channels and applies the
    weight contraction afterwards. It wins when output channels are substantially fewer
    than input channels; near parity the weight-first order is better because it needs a
    single sparse launch rather than one per basis function.
    """
    return kernel_size > 1 and out_per_group * 2 <= in_per_group


def _spatial_first_dgrad(grad_output_r, weight, row_lat, seg_off, seg, val_off, vals, split_ker, kernel_size, nlat_in, nlon_in, row_offsets):
    """
    Input gradient with the sparse transpose first, one scatter per basis function.

    Basis function k's rows are ``row_offsets[k]:row_offsets[k + 1]``. Their arc and value
    offsets are absolute, so each call reads slices of the same arrays and needs only
    ``split_ker``, zeros, in place of row_ker for a K = 1 call. See build_split.
    """
    B, G, Og, H, W = grad_output_r.shape
    Cg = weight.shape[2]
    grad_small = grad_output_r.reshape(B, G * Og, 1, H, W).contiguous()

    parts = []
    for k in range(kernel_size):
        r0, r1 = row_offsets[k], row_offsets[k + 1]
        if r1 == r0:
            parts.append(grad_output_r.new_zeros((B, G, Og, nlat_in, nlon_in)))
            continue
        part = disco_kernels.backward_regular.default(grad_small, split_ker[: r1 - r0], row_lat[r0:r1], seg_off[r0 : r1 + 1], seg, val_off[r0 : r1 + 1], vals, 1, nlat_in, nlon_in)
        parts.append(part.reshape(B, G, Og, nlat_in, nlon_in))

    grad_spatial = torch.stack(parts, dim=3)
    grad_inp = torch.einsum("bgokxy,gock->bgcxy", grad_spatial, weight).contiguous()
    return grad_inp.reshape(B, G * Cg, nlat_in, nlon_in)


def _register_autocast(qualname: str, device_types):
    """
    Make an op follow ``torch.autocast`` by casting its activation, the first argument.

    Registered at the dispatcher's Autocast keys rather than through
    ``torch.library.register_autocast``, whose ``cast_inputs`` hard-codes one dtype and
    cannot follow the active autocast dtype. The inner call runs with autocast disabled,
    which drops the Autocast key from the dispatch set, so it reaches the device kernel
    instead of recursing into this one.

    One helper for every op and both devices, so the ops cannot drift apart. Without the
    CPU key ``torch.autocast("cpu")`` would be a silent no-op for the op.
    """
    op = getattr(torch.ops.disco_kernels, qualname.split("::")[-1]).default

    for device_type in device_types:

        def _autocast_impl(inp, *args, _device_type=device_type):
            cast_dtype = torch.get_autocast_dtype(_device_type)
            with torch.amp.autocast(_device_type, enabled=False):
                return op(inp.to(cast_dtype), *args)

        torch.library.impl(qualname, f"Autocast{device_type.upper()}")(_autocast_impl)


# Operand checks for the fake implementations. They mirror the host-side TORCH_CHECKs in
# disco_checks.h -- devices, dtypes, ranks, sizes -- so that under torch.compile a malformed
# call fails at trace time with the same message instead of only when the graph runs.
# Strides stay on the C++ side: with dynamic shapes they are symbolic, and the fakes promise
# nothing about layout.
#
# Every check goes through `check`, i.e. torch._check, and holds at most one comparison of
# sizes: those may be symbolic, and combining them with `and`, all() or a tuple comparison
# would call bool() on a SymBool, which makes Dynamo guard -- specialize -- on it. Checks on
# plain Python values (dtypes, ranks, int arguments) are combined freely.


def _check_size(t: torch.Tensor, dim: int, expected, what: str) -> None:
    # one symbolic comparison per call, see above
    check(t.shape[dim] == expected, lambda: f"{what}: expected {expected}, got {t.shape[dim]} (shape {tuple(t.shape)})")


def _check_index_vector(t: torch.Tensor, inp: torch.Tensor, dtype: torch.dtype, name: str) -> None:
    check(t.device == inp.device, lambda: f"{name} must be on the same device as the activations ({inp.device}), got {t.device}")
    check(t.dtype == dtype and t.dim() == 1, lambda: f"{name} must be a 1-D {dtype} tensor, got {t.dtype} of shape {tuple(t.shape)}")


def _check_arc_psi(inp, row_ker, row_lat, seg_off, seg, val_off, vals, kernel_size: int, vals_dtype_exact: bool) -> None:
    """
    psi in arc form, see check_arc_psi in disco_checks.h.

    The raw operators read vals in the compute dtype of the activations, so for them the
    dtype is exact; the custom ops cast vals themselves and only need it floating.
    """
    check(inp.is_floating_point(), lambda: f"inp must be a floating-point tensor, got {inp.dtype}")
    check(kernel_size > 0, lambda: f"kernel_size must be positive, got {kernel_size}")
    _check_index_vector(row_ker, inp, torch.int32, "row_ker")
    _check_index_vector(row_lat, inp, torch.int32, "row_lat")
    _check_index_vector(seg_off, inp, torch.int64, "seg_off")
    _check_index_vector(val_off, inp, torch.int64, "val_off")
    _check_size(row_lat, 0, row_ker.shape[0], "row_lat, one per row")
    _check_size(seg_off, 0, row_ker.shape[0] + 1, "seg_off, one more than the rows")
    _check_size(val_off, 0, row_ker.shape[0] + 1, "val_off, one more than the rows")
    check(seg.device == inp.device, lambda: f"seg must be on the same device as the activations ({inp.device}), got {seg.device}")
    check(seg.dtype == torch.int32 and seg.dim() == 2, lambda: f"seg must be an int32 (nsegs, 3) tensor, got {seg.dtype} of shape {tuple(seg.shape)}")
    _check_size(seg, 1, 3, "seg columns (ring, start, length)")
    check(vals.device == inp.device, lambda: f"vals must be on the same device as the activations ({inp.device}), got {vals.device}")
    check(vals.dim() == 1, lambda: f"vals must be 1-D, got shape {tuple(vals.shape)}")
    if vals_dtype_exact:
        check(vals.dtype == _compute_dtype(inp.dtype), lambda: f"vals must be in the compute dtype of the input ({_compute_dtype(inp.dtype)} for {inp.dtype}), got {vals.dtype}")
    else:
        check(vals.is_floating_point(), lambda: f"vals must be floating point, got {vals.dtype}")


def _check_forward_inputs(inp, row_ker, row_lat, seg_off, seg, val_off, vals, kernel_size, nlat_out, nlon_out, vals_dtype_exact) -> None:
    """The gather: inp (B, C, Hi, Wi) -> (B, C, K, Ho, Wo); see check_forward_inputs."""
    check(inp.dim() == 4, lambda: f"inp must be (B, C, Hi, Wi), got shape {tuple(inp.shape)}")
    check(nlat_out > 0 and nlon_out > 0, lambda: f"nlat_out and nlon_out must be positive, got {nlat_out} and {nlon_out}")
    check(inp.shape[3] % nlon_out == 0, lambda: f"Wi ({inp.shape[3]}) must be an integer multiple of Wo ({nlon_out}) for the p-shift to be exact")
    _check_arc_psi(inp, row_ker, row_lat, seg_off, seg, val_off, vals, kernel_size, vals_dtype_exact)


def _check_backward_inputs(inp, row_ker, row_lat, seg_off, seg, val_off, vals, kernel_size, nlat_out, nlon_out, vals_dtype_exact) -> None:
    """The scatter: inp (B, C, K, Hi, Wi) -> (B, C, Ho, Wo); see check_backward_inputs."""
    check(inp.dim() == 5, lambda: f"inp must be (B, C, K, Hi, Wi), got shape {tuple(inp.shape)}")
    _check_size(inp, 2, kernel_size, "inp basis-function planes (kernel_size)")
    check(nlat_out > 0 and nlon_out > 0, lambda: f"nlat_out and nlon_out must be positive, got {nlat_out} and {nlon_out}")
    check(nlon_out % inp.shape[4] == 0, lambda: f"Wo ({nlon_out}) must be an integer multiple of Wi ({inp.shape[4]}) for the p-shift to be exact")
    _check_arc_psi(inp, row_ker, row_lat, seg_off, seg, val_off, vals, kernel_size, vals_dtype_exact)


def _check_ring_tables(inp, ring_base, ring_size) -> None:
    """The ragged ops' ring tables; see check_ring_tables."""
    _check_index_vector(ring_base, inp, torch.int64, "ring_base")
    _check_index_vector(ring_size, inp, torch.int64, "ring_size")
    _check_size(ring_size, 0, ring_base.shape[0], "ring_size, one per ring")


def _check_ragged_forward_inputs(inp, row_ker, row_pt, seg_off, seg, val_off, vals, ring_base, ring_size, kernel_size, npoints_out, vals_dtype_exact) -> None:
    """The ragged gather: inp (B, C, npoints_in) -> (B, C, K, npoints_out); see check_ragged_forward_inputs."""
    check(inp.dim() == 3, lambda: f"inp must be (B, C, npoints_in), got shape {tuple(inp.shape)}")
    check(npoints_out > 0, lambda: f"npoints_out must be positive, got {npoints_out}")
    _check_arc_psi(inp, row_ker, row_pt, seg_off, seg, val_off, vals, kernel_size, vals_dtype_exact)
    _check_ring_tables(inp, ring_base, ring_size)


def _check_ragged_backward_inputs(inp, row_ker, row_pt, seg_off, seg, val_off, vals, ring_base, ring_size, kernel_size, npoints_out, vals_dtype_exact) -> None:
    """The ragged scatter: inp (B, C, K, npoints_in) -> (B, C, npoints_out); see check_ragged_backward_inputs."""
    check(inp.dim() == 4, lambda: f"inp must be (B, C, K, npoints_in), got shape {tuple(inp.shape)}")
    _check_size(inp, 2, kernel_size, "inp basis-function planes (kernel_size)")
    check(npoints_out > 0, lambda: f"npoints_out must be positive, got {npoints_out}")
    _check_arc_psi(inp, row_ker, row_pt, seg_off, seg, val_off, vals, kernel_size, vals_dtype_exact)
    _check_ring_tables(inp, ring_base, ring_size)


def _check_kpacked_inputs(inp, pack_idx, pack_val, pack_offset, kernel_size, nlat_out, nlon_out) -> None:
    """The tensor-core forward's blocked layout; see check_kpacked_inputs."""
    check(inp.dim() == 4, lambda: f"inp must be (B, C, Hi, Wi), got shape {tuple(inp.shape)}")
    check(inp.dtype in (torch.float16, torch.bfloat16), lambda: f"forward_kpacked requires fp16 or bf16 activations, got {inp.dtype}")
    check(nlat_out > 0 and nlon_out > 0 and nlon_out % 8 == 0, lambda: f"nlat_out must be positive and nlon_out a positive multiple of 8, got {nlat_out} and {nlon_out}")
    check(inp.shape[3] % nlon_out == 0, lambda: f"Wi ({inp.shape[3]}) must be an integer multiple of Wo ({nlon_out})")
    for t, name in ((pack_idx, "pack_idx"), (pack_val, "pack_val"), (pack_offset, "pack_offset")):
        check(t.device == inp.device, lambda t=t, name=name: f"{name} must be on the same device as the activations ({inp.device}), got {t.device}")
    check(pack_idx.dtype == torch.int64 and pack_idx.dim() == 2, lambda: f"pack_idx must be an int64 (npoints, 2) tensor, got {pack_idx.dtype} of shape {tuple(pack_idx.shape)}")
    _check_size(pack_idx, 1, 2, "pack_idx columns (ring, lon)")
    check(
        pack_val.is_floating_point() and pack_val.dim() == 2, lambda: f"pack_val must be a floating (npoints, K_pad) tensor, got {pack_val.dtype} of shape {tuple(pack_val.shape)}"
    )
    _check_size(pack_val, 0, pack_idx.shape[0], "pack_val rows, one per pack_idx entry")
    # K_pad in {8, 16}, as three single comparisons rather than one `or` of two
    check(pack_val.shape[1] % 8 == 0, lambda: f"pack_val must be padded to K_pad 8 or 16, got {pack_val.shape[1]}")
    check(pack_val.shape[1] >= 8, lambda: f"pack_val must be padded to K_pad 8 or 16, got {pack_val.shape[1]}")
    check(pack_val.shape[1] <= 16, lambda: f"pack_val must be padded to K_pad 8 or 16, got {pack_val.shape[1]}")
    check(kernel_size <= pack_val.shape[1], lambda: f"pack_val's K_pad ({pack_val.shape[1]}) must cover kernel_size ({kernel_size})")
    _check_index_vector(pack_offset, inp, torch.int64, "pack_offset")
    _check_size(pack_offset, 0, nlat_out + 1, "pack_offset, nlat_out + 1 offsets")


# The operators take psi in arc form (see torch_harmonics.disco._psi_layouts): row_ker and
# row_lat int32 per row, seg_off and val_off int64 row offsets into seg (int32 (nsegs, 3)
# arcs) and vals, the values in the compute dtype of the activations.
if optimized_kernels_is_available():

    @torch.library.register_fake("disco_kernels::forward_regular")
    def _(
        inp: torch.Tensor,
        row_ker: torch.Tensor,
        row_lat: torch.Tensor,
        seg_off: torch.Tensor,
        seg: torch.Tensor,
        val_off: torch.Tensor,
        vals: torch.Tensor,
        kernel_size: int,
        nlat_out: int,
        nlon_out: int,
    ) -> torch.Tensor:
        _check_forward_inputs(inp, row_ker, row_lat, seg_off, seg, val_off, vals, kernel_size, nlat_out, nlon_out, vals_dtype_exact=True)
        return inp.new_empty((inp.shape[0], inp.shape[1], kernel_size, nlat_out, nlon_out))

    @torch.library.register_fake("disco_kernels::backward_regular")
    def _(
        inp: torch.Tensor,
        row_ker: torch.Tensor,
        row_lat: torch.Tensor,
        seg_off: torch.Tensor,
        seg: torch.Tensor,
        val_off: torch.Tensor,
        vals: torch.Tensor,
        kernel_size: int,
        nlat_out: int,
        nlon_out: int,
    ) -> torch.Tensor:
        _check_backward_inputs(inp, row_ker, row_lat, seg_off, seg, val_off, vals, kernel_size, nlat_out, nlon_out, vals_dtype_exact=True)
        return inp.new_empty((inp.shape[0], inp.shape[1], nlat_out, nlon_out))

    @torch.library.register_fake("disco_kernels::forward_kpacked")
    def _(inp: torch.Tensor, pack_idx: torch.Tensor, pack_val: torch.Tensor, pack_offset: torch.Tensor, kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        _check_kpacked_inputs(inp, pack_idx, pack_val, pack_offset, kernel_size, nlat_out, nlon_out)
        return inp.new_empty((inp.shape[0], inp.shape[1], kernel_size, nlat_out, nlon_out))

    # The contraction and its transpose, as custom ops with their own autograd: each is
    # the other's backward. The activation stays in its storage dtype, so fp16/bf16 reach
    # the kernel, which accumulates in fp32; vals is cast to the compute dtype, which the
    # kernel reads it as. fp32/fp64 pass through unchanged.
    @torch.library.custom_op("disco_kernels::_disco_s2_contraction_regular_optimized", mutates_args=())
    def _disco_s2_contraction_regular_optimized(
        inp: torch.Tensor,
        row_ker: torch.Tensor,
        row_lat: torch.Tensor,
        seg_off: torch.Tensor,
        seg: torch.Tensor,
        val_off: torch.Tensor,
        vals: torch.Tensor,
        kernel_size: int,
        nlat_out: int,
        nlon_out: int,
    ) -> torch.Tensor:
        itype = inp.dtype
        vals = vals.to(_compute_dtype(itype))
        out = disco_kernels.forward_regular.default(inp.contiguous(), row_ker, row_lat, seg_off, seg, val_off, vals, kernel_size, nlat_out, nlon_out)
        return out.to(itype)

    @torch.library.custom_op("disco_kernels::_disco_s2_transpose_contraction_regular_optimized", mutates_args=())
    def _disco_s2_transpose_contraction_regular_optimized(
        inp: torch.Tensor,
        row_ker: torch.Tensor,
        row_lat: torch.Tensor,
        seg_off: torch.Tensor,
        seg: torch.Tensor,
        val_off: torch.Tensor,
        vals: torch.Tensor,
        kernel_size: int,
        nlat_out: int,
        nlon_out: int,
    ) -> torch.Tensor:
        itype = inp.dtype
        vals = vals.to(_compute_dtype(itype))
        out = disco_kernels.backward_regular.default(inp.contiguous(), row_ker, row_lat, seg_off, seg, val_off, vals, kernel_size, nlat_out, nlon_out)
        return out.to(itype)

    @torch.library.register_fake("disco_kernels::_disco_s2_contraction_regular_optimized")
    def _(
        inp: torch.Tensor,
        row_ker: torch.Tensor,
        row_lat: torch.Tensor,
        seg_off: torch.Tensor,
        seg: torch.Tensor,
        val_off: torch.Tensor,
        vals: torch.Tensor,
        kernel_size: int,
        nlat_out: int,
        nlon_out: int,
    ) -> torch.Tensor:
        _check_forward_inputs(inp, row_ker, row_lat, seg_off, seg, val_off, vals, kernel_size, nlat_out, nlon_out, vals_dtype_exact=False)
        return inp.new_empty((inp.shape[0], inp.shape[1], kernel_size, nlat_out, nlon_out))

    @torch.library.register_fake("disco_kernels::_disco_s2_transpose_contraction_regular_optimized")
    def _(
        inp: torch.Tensor,
        row_ker: torch.Tensor,
        row_lat: torch.Tensor,
        seg_off: torch.Tensor,
        seg: torch.Tensor,
        val_off: torch.Tensor,
        vals: torch.Tensor,
        kernel_size: int,
        nlat_out: int,
        nlon_out: int,
    ) -> torch.Tensor:
        _check_backward_inputs(inp, row_ker, row_lat, seg_off, seg, val_off, vals, kernel_size, nlat_out, nlon_out, vals_dtype_exact=False)
        return inp.new_empty((inp.shape[0], inp.shape[1], nlat_out, nlon_out))


def _setup_context_contraction(ctx, inputs, output):
    inp, row_ker, row_lat, seg_off, seg, val_off, vals, kernel_size, nlat_out, nlon_out = inputs
    ctx.save_for_backward(row_ker, row_lat, seg_off, seg, val_off, vals)
    ctx.kernel_size = kernel_size
    ctx.nlat_in = inp.shape[-2]
    ctx.nlon_in = inp.shape[-1]


def _contraction_bwd(ctx, grad_output):
    grad_input = None
    if ctx.needs_input_grad[0]:
        grad_input = _disco_s2_transpose_contraction_regular_optimized(grad_output, *ctx.saved_tensors, ctx.kernel_size, ctx.nlat_in, ctx.nlon_in)
    return (grad_input,) + (None,) * 9


def _transpose_contraction_bwd(ctx, grad_output):
    grad_input = None
    if ctx.needs_input_grad[0]:
        grad_input = _disco_s2_contraction_regular_optimized(grad_output, *ctx.saved_tensors, ctx.kernel_size, ctx.nlat_in, ctx.nlon_in)
    return (grad_input,) + (None,) * 9


if optimized_kernels_is_available():
    torch.library.register_autograd("disco_kernels::_disco_s2_contraction_regular_optimized", _contraction_bwd, setup_context=_setup_context_contraction)
    torch.library.register_autograd("disco_kernels::_disco_s2_transpose_contraction_regular_optimized", _transpose_contraction_bwd, setup_context=_setup_context_contraction)

    _register_autocast("disco_kernels::_disco_s2_contraction_regular_optimized", ("cuda", "cpu"))
    _register_autocast("disco_kernels::_disco_s2_transpose_contraction_regular_optimized", ("cuda", "cpu"))
    # the kpacked kernel exists on CUDA only
    _register_autocast("disco_kernels::forward_kpacked", ("cuda",))


# The ragged counterparts, for a grid whose rings differ in length (HEALPix, or a regular
# grid paired with one); registered for CPU and CUDA, both reading the arc form keyed per
# point (kernels_cpu/ragged, kernels_cuda/ragged). The same structure as the regular ops
# above, with npoints_out for (nlat_out, nlon_out) and the ring tables added; without the
# compiled kernels a ragged layer falls back to the torch reference.
if optimized_kernels_is_available():

    @torch.library.register_fake("disco_kernels::forward_ragged")
    def _(
        inp: torch.Tensor,
        row_ker: torch.Tensor,
        row_pt: torch.Tensor,
        seg_off: torch.Tensor,
        seg: torch.Tensor,
        val_off: torch.Tensor,
        vals: torch.Tensor,
        ring_base: torch.Tensor,
        ring_size: torch.Tensor,
        kernel_size: int,
        npoints_out: int,
    ) -> torch.Tensor:
        _check_ragged_forward_inputs(inp, row_ker, row_pt, seg_off, seg, val_off, vals, ring_base, ring_size, kernel_size, npoints_out, vals_dtype_exact=True)
        return inp.new_empty((inp.shape[0], inp.shape[1], kernel_size, npoints_out))

    @torch.library.register_fake("disco_kernels::backward_ragged")
    def _(
        inp: torch.Tensor,
        row_ker: torch.Tensor,
        row_pt: torch.Tensor,
        seg_off: torch.Tensor,
        seg: torch.Tensor,
        val_off: torch.Tensor,
        vals: torch.Tensor,
        ring_base: torch.Tensor,
        ring_size: torch.Tensor,
        kernel_size: int,
        npoints_out: int,
    ) -> torch.Tensor:
        _check_ragged_backward_inputs(inp, row_ker, row_pt, seg_off, seg, val_off, vals, ring_base, ring_size, kernel_size, npoints_out, vals_dtype_exact=True)
        return inp.new_empty((inp.shape[0], inp.shape[1], npoints_out))

    @torch.library.custom_op("disco_kernels::_disco_s2_contraction_ragged_optimized", mutates_args=())
    def _disco_s2_contraction_ragged_optimized(
        inp: torch.Tensor,
        row_ker: torch.Tensor,
        row_pt: torch.Tensor,
        seg_off: torch.Tensor,
        seg: torch.Tensor,
        val_off: torch.Tensor,
        vals: torch.Tensor,
        ring_base: torch.Tensor,
        ring_size: torch.Tensor,
        kernel_size: int,
        npoints_out: int,
    ) -> torch.Tensor:
        itype = inp.dtype
        vals = vals.to(_compute_dtype(itype))
        out = disco_kernels.forward_ragged.default(inp.contiguous(), row_ker, row_pt, seg_off, seg, val_off, vals, ring_base, ring_size, kernel_size, npoints_out)
        return out.to(itype)

    @torch.library.custom_op("disco_kernels::_disco_s2_transpose_contraction_ragged_optimized", mutates_args=())
    def _disco_s2_transpose_contraction_ragged_optimized(
        inp: torch.Tensor,
        row_ker: torch.Tensor,
        row_pt: torch.Tensor,
        seg_off: torch.Tensor,
        seg: torch.Tensor,
        val_off: torch.Tensor,
        vals: torch.Tensor,
        ring_base: torch.Tensor,
        ring_size: torch.Tensor,
        kernel_size: int,
        npoints_out: int,
    ) -> torch.Tensor:
        itype = inp.dtype
        vals = vals.to(_compute_dtype(itype))
        out = disco_kernels.backward_ragged.default(inp.contiguous(), row_ker, row_pt, seg_off, seg, val_off, vals, ring_base, ring_size, kernel_size, npoints_out)
        return out.to(itype)

    @torch.library.register_fake("disco_kernels::_disco_s2_contraction_ragged_optimized")
    def _(
        inp: torch.Tensor,
        row_ker: torch.Tensor,
        row_pt: torch.Tensor,
        seg_off: torch.Tensor,
        seg: torch.Tensor,
        val_off: torch.Tensor,
        vals: torch.Tensor,
        ring_base: torch.Tensor,
        ring_size: torch.Tensor,
        kernel_size: int,
        npoints_out: int,
    ) -> torch.Tensor:
        _check_ragged_forward_inputs(inp, row_ker, row_pt, seg_off, seg, val_off, vals, ring_base, ring_size, kernel_size, npoints_out, vals_dtype_exact=False)
        return inp.new_empty((inp.shape[0], inp.shape[1], kernel_size, npoints_out))

    @torch.library.register_fake("disco_kernels::_disco_s2_transpose_contraction_ragged_optimized")
    def _(
        inp: torch.Tensor,
        row_ker: torch.Tensor,
        row_pt: torch.Tensor,
        seg_off: torch.Tensor,
        seg: torch.Tensor,
        val_off: torch.Tensor,
        vals: torch.Tensor,
        ring_base: torch.Tensor,
        ring_size: torch.Tensor,
        kernel_size: int,
        npoints_out: int,
    ) -> torch.Tensor:
        _check_ragged_backward_inputs(inp, row_ker, row_pt, seg_off, seg, val_off, vals, ring_base, ring_size, kernel_size, npoints_out, vals_dtype_exact=False)
        return inp.new_empty((inp.shape[0], inp.shape[1], npoints_out))


def _setup_context_ragged_contraction(ctx, inputs, output):
    inp, row_ker, row_pt, seg_off, seg, val_off, vals, ring_base, ring_size, kernel_size, npoints_out = inputs
    ctx.save_for_backward(row_ker, row_pt, seg_off, seg, val_off, vals, ring_base, ring_size)
    ctx.kernel_size = kernel_size
    ctx.npoints_in = inp.shape[-1]


def _ragged_contraction_bwd(ctx, grad_output):
    grad_input = None
    if ctx.needs_input_grad[0]:
        grad_input = _disco_s2_transpose_contraction_ragged_optimized(grad_output, *ctx.saved_tensors, ctx.kernel_size, ctx.npoints_in)
    return (grad_input,) + (None,) * 10


def _ragged_transpose_contraction_bwd(ctx, grad_output):
    grad_input = None
    if ctx.needs_input_grad[0]:
        grad_input = _disco_s2_contraction_ragged_optimized(grad_output, *ctx.saved_tensors, ctx.kernel_size, ctx.npoints_in)
    return (grad_input,) + (None,) * 10


if optimized_kernels_is_available():
    torch.library.register_autograd("disco_kernels::_disco_s2_contraction_ragged_optimized", _ragged_contraction_bwd, setup_context=_setup_context_ragged_contraction)
    torch.library.register_autograd(
        "disco_kernels::_disco_s2_transpose_contraction_ragged_optimized", _ragged_transpose_contraction_bwd, setup_context=_setup_context_ragged_contraction
    )

    _register_autocast("disco_kernels::_disco_s2_contraction_ragged_optimized", ("cuda", "cpu"))
    _register_autocast("disco_kernels::_disco_s2_transpose_contraction_ragged_optimized", ("cuda", "cpu"))


def _contract(inp, row_ker, row_lat, seg_off, seg, val_off, vals, pack_idx, pack_val, pack_offset, kernel_size, nlat_out, nlon_out):
    """The raw forward contraction: tensor-core kpacked when its layout is given, the arc gather otherwise."""
    inp = inp.contiguous()
    if pack_idx is not None:
        return disco_kernels.forward_kpacked.default(inp, pack_idx, pack_val, pack_offset, kernel_size, nlat_out, nlon_out)
    itype = inp.dtype
    out = disco_kernels.forward_regular.default(inp, row_ker, row_lat, seg_off, seg, val_off, vals.to(_compute_dtype(itype)), kernel_size, nlat_out, nlon_out)
    return out.to(itype)


class _DiscoKpackedFn(torch.autograd.Function):
    """
    Kpacked forward contraction, arc scatter backward.

    The backward stays on the scatter kernel: it is input-pixel-parallel with no cross-CTA
    atomics, the right algorithm for overlapping support sets (the reason cuDNN uses an
    implicit GEMM rather than col2im for strided convolutions).
    """

    @staticmethod
    def forward(ctx, inp, pack_idx, pack_val, pack_offset, row_ker, row_lat, seg_off, seg, val_off, vals, kernel_size, nlat_out, nlon_out):
        ctx.save_for_backward(row_ker, row_lat, seg_off, seg, val_off, vals)
        ctx.kernel_size = kernel_size
        ctx.nlat_in = inp.shape[-2]
        ctx.nlon_in = inp.shape[-1]
        return _contract(inp, None, None, None, None, None, None, pack_idx, pack_val, pack_offset, kernel_size, nlat_out, nlon_out)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            # the transpose contraction op rather than the raw scatter kernel: the op has
            # autograd, so this backward is itself differentiable (double backward works)
            grad_input = _disco_s2_transpose_contraction_regular_optimized(grad_output, *ctx.saved_tensors, ctx.kernel_size, ctx.nlat_in, ctx.nlon_in)
        # inp, pack_idx, pack_val, pack_offset, the six arc arrays, kernel_size, nlat_out, nlon_out
        return (grad_input,) + (None,) * 12


def _disco_s2_contraction_kpacked(inp, kpacked, arcs, kernel_size, nlat_out, nlon_out):
    """Kpacked forward through :class:`_DiscoKpackedFn`; ``kpacked`` is (pack_idx, pack_val, pack_offset), ``arcs`` the six arc arrays."""
    return _DiscoKpackedFn.apply(inp, *kpacked, *arcs, kernel_size, nlat_out, nlon_out)


class _DiscoConvFn(torch.autograd.Function):
    """
    The DISCO contraction followed by the weight contraction, as one autograd node.

    One node rather than the contraction op followed by an einsum, for what the node can
    do with both halves in view:

    * ``recompute=True`` saves the input instead of the K-expanded intermediate
      ``(B, C, K, H, W)`` and recomputes it in backward: K times less activation memory for
      one extra contraction. This is the layer's ``fused=True``.
    * the input gradient can run the sparse transpose before the weight contraction
      (spatial-first, see :func:`_use_spatial_first_dgrad`), which it can only choose when
      it sees the weight. ``split_ker`` and ``split_row_offsets`` (see build_split) are its
      state, and it is considered exactly when they are passed.

    The forward contraction is the kpacked tensor-core kernel when its layout is passed,
    the arc gather otherwise; the backward is the arc scatter either way, and so is the
    recompute.

    The backward calls the raw kernels, which autograd cannot see. Under
    ``create_graph=True`` it is built from the contraction ops and einsums instead (see
    :func:`_conv_backward_differentiable`), so double backward works. That needs the input
    itself, which is why it is saved even when the K-expanded intermediate is.
    """

    @staticmethod
    def forward(
        ctx,
        inp,
        weight,
        row_ker,
        row_lat,
        seg_off,
        seg,
        val_off,
        vals,
        pack_idx,
        pack_val,
        pack_offset,
        split_ker,
        kernel_size,
        nlat_out,
        nlon_out,
        groups,
        groupsize,
        recompute,
        split_row_offsets,
    ):
        itype = inp.dtype
        x_expanded = _contract(inp, row_ker, row_lat, seg_off, seg, val_off, vals, pack_idx, pack_val, pack_offset, kernel_size, nlat_out, nlon_out)

        # the input as passed, not a contiguous copy made here: only the input carries its
        # graph into a create_graph backward. Saving it only keeps a reference.
        ctx.save_for_backward(inp, None if recompute else x_expanded, weight, row_ker, row_lat, seg_off, seg, val_off, vals, split_ker)
        ctx.recompute = recompute
        ctx.kernel_size = kernel_size
        ctx.nlat_in = inp.shape[-2]
        ctx.nlon_in = inp.shape[-1]
        ctx.nlat_out = nlat_out
        ctx.nlon_out = nlon_out
        ctx.groups = groups
        ctx.groupsize = groupsize
        ctx.split_row_offsets = split_row_offsets

        B, C, K, H, W = x_expanded.shape
        x_expanded = x_expanded.reshape(B, groups, groupsize, K, H, W)
        out = torch.einsum("bgckxy,gock->bgoxy", x_expanded, weight.to(itype)).contiguous()
        return out.reshape(B, groups * weight.shape[1], H, W)

    @staticmethod
    def backward(ctx, grad_output):
        inp, x_expanded, weight, row_ker, row_lat, seg_off, seg, val_off, vals, split_ker = ctx.saved_tensors

        itype = grad_output.dtype
        vals_c = vals.to(_compute_dtype(itype))

        K = ctx.kernel_size
        G, Cg = ctx.groups, ctx.groupsize
        H, W = ctx.nlat_out, ctx.nlon_out
        Og = weight.shape[1]
        B = grad_output.shape[0]
        grad_output_r = grad_output.reshape(B, G, Og, H, W)

        # inp, weight, then the six arc arrays, the three kpacked ones, split_ker,
        # kernel_size, nlat_out, nlon_out, groups, groupsize, recompute, split_row_offsets
        nones = (None,) * 17

        # create_graph=True: the gradients must be differentiable themselves. Eager only, a
        # compiled backward rejects create_graph by itself.
        if torch.is_grad_enabled() and not torch.compiler.is_compiling():
            arcs = (row_ker, row_lat, seg_off, seg, val_off, vals)
            return _conv_backward_differentiable(ctx, grad_output_r, inp, weight, arcs) + nones

        grad_inp = None
        grad_weight = None

        if ctx.needs_input_grad[0]:
            if split_ker is not None and _use_spatial_first_dgrad(Og, Cg, K):
                grad_inp = _spatial_first_dgrad(
                    grad_output_r, weight.to(itype), row_lat, seg_off, seg, val_off, vals_c, split_ker, K, ctx.nlat_in, ctx.nlon_in, ctx.split_row_offsets
                )
            else:
                grad_x_expanded = torch.einsum("bgoxy,gock->bgckxy", grad_output_r, weight.to(itype))
                grad_x_expanded = grad_x_expanded.reshape(B, G * Cg, K, H, W).contiguous()
                grad_inp = disco_kernels.backward_regular.default(grad_x_expanded, row_ker, row_lat, seg_off, seg, val_off, vals_c, K, ctx.nlat_in, ctx.nlon_in)
            grad_inp = grad_inp.to(itype)

        if ctx.needs_input_grad[1]:
            if ctx.recompute:
                x_expanded = disco_kernels.forward_regular.default(inp.contiguous(), row_ker, row_lat, seg_off, seg, val_off, vals_c, K, H, W)
            x_expanded = x_expanded.to(itype).reshape(B, G, Cg, K, H, W)
            grad_weight = torch.einsum("bgoxy,bgckxy->gock", grad_output_r, x_expanded)

        return (grad_inp, grad_weight) + nones


def _conv_backward_differentiable(ctx, grad_output_r, inp, weight, arcs):
    """
    The backward of :class:`_DiscoConvFn` from ops autograd can see, for ``create_graph=True``.

    The contraction op and its transpose are each other's backward, so the gradients built
    from them and the two einsums are differentiable to any order. The input gradient takes
    the weight-first order and the recompute the arc kernels: the same values as the fast
    path, without its choices of kernel.
    """
    B, G, Og, H, W = grad_output_r.shape
    K, Cg = ctx.kernel_size, ctx.groupsize
    itype = grad_output_r.dtype

    grad_inp = None
    grad_weight = None

    if ctx.needs_input_grad[0]:
        grad_x_expanded = torch.einsum("bgoxy,gock->bgckxy", grad_output_r, weight.to(itype)).reshape(B, G * Cg, K, H, W)
        grad_inp = _disco_s2_transpose_contraction_regular_optimized(grad_x_expanded, *arcs, K, ctx.nlat_in, ctx.nlon_in)

    if ctx.needs_input_grad[1]:
        x_expanded = _disco_s2_contraction_regular_optimized(inp.to(itype), *arcs, K, H, W).reshape(B, G, Cg, K, H, W)
        grad_weight = torch.einsum("bgoxy,bgckxy->gock", grad_output_r, x_expanded)

    return grad_inp, grad_weight


def _disco_s2_conv_optimized(inp, weight, arcs, kpacked, split, kernel_size, nlat_out, nlon_out, groups, groupsize, recompute=False):
    """
    Contraction plus weight contraction through :class:`_DiscoConvFn`.

    Parameters
    ----------
    inp : torch.Tensor
        ``(B, groups * groupsize, H_in, W_in)``.
    weight : torch.Tensor
        ``(groups, out_per_group, groupsize, kernel_size)``.
    arcs : Tuple[torch.Tensor, ...]
        ``(row_ker, row_lat, seg_off, seg, val_off, vals)``, always: the backward reads it.
    kpacked : Optional[Tuple[torch.Tensor, ...]]
        ``(pack_idx, pack_val, pack_offset)`` to run the forward on the tensor cores.
    split : Optional[Tuple]
        ``(split_ker, row_offsets)`` from build_split, to allow the spatial-first input
        gradient.
    recompute : bool
        Recompute the K-expanded intermediate in backward rather than saving it.
    """
    pack_idx, pack_val, pack_offset = kpacked if kpacked is not None else (None, None, None)
    split_ker, row_offsets = split if split is not None else (None, ())
    return _DiscoConvFn.apply(inp, weight, *arcs, pack_idx, pack_val, pack_offset, split_ker, kernel_size, nlat_out, nlon_out, groups, groupsize, recompute, row_offsets)


def _spatial_first_dgrad_ragged(grad_output_r, weight, row_pt, seg_off, seg, val_off, vals, ring_base, ring_size, split_ker, kernel_size, npoints_in, row_offsets):
    """The ragged counterpart of :func:`_spatial_first_dgrad`: one K = 1 scatter per basis function."""
    B, G, Og, N = grad_output_r.shape
    Cg = weight.shape[2]
    grad_small = grad_output_r.reshape(B, G * Og, 1, N).contiguous()

    parts = []
    for k in range(kernel_size):
        r0, r1 = row_offsets[k], row_offsets[k + 1]
        if r1 == r0:
            parts.append(grad_output_r.new_zeros((B, G, Og, npoints_in)))
            continue
        part = disco_kernels.backward_ragged.default(
            grad_small, split_ker[: r1 - r0], row_pt[r0:r1], seg_off[r0 : r1 + 1], seg, val_off[r0 : r1 + 1], vals, ring_base, ring_size, 1, npoints_in
        )
        parts.append(part.reshape(B, G, Og, npoints_in))

    grad_spatial = torch.stack(parts, dim=3)
    grad_inp = torch.einsum("bgokn,gock->bgcn", grad_spatial, weight).contiguous()
    return grad_inp.reshape(B, G * Cg, npoints_in)


class _DiscoRaggedConvFn(torch.autograd.Function):
    """
    The ragged DISCO contraction followed by the weight contraction, as one autograd node.

    The counterpart of :class:`_DiscoConvFn` for psi keyed per point, with the same two
    things the node is for -- ``recompute`` and the spatial-first input gradient -- on the
    ragged gather and scatter. There is no tensor-core forward here, so the forward is the
    gather and the backward and the recompute the scatter and the gather.
    """

    @staticmethod
    def forward(
        ctx,
        inp,
        weight,
        row_ker,
        row_pt,
        seg_off,
        seg,
        val_off,
        vals,
        ring_base,
        ring_size,
        split_ker,
        kernel_size,
        npoints_out,
        groups,
        groupsize,
        recompute,
        split_row_offsets,
    ):
        itype = inp.dtype
        inp = inp.contiguous()
        vals_c = vals.to(_compute_dtype(itype))
        x_expanded = disco_kernels.forward_ragged.default(inp, row_ker, row_pt, seg_off, seg, val_off, vals_c, ring_base, ring_size, kernel_size, npoints_out).to(itype)

        ctx.save_for_backward(inp if recompute else x_expanded, weight, row_ker, row_pt, seg_off, seg, val_off, vals, ring_base, ring_size, split_ker)
        ctx.recompute = recompute
        ctx.kernel_size = kernel_size
        ctx.npoints_in = inp.shape[-1]
        ctx.npoints_out = npoints_out
        ctx.groups = groups
        ctx.groupsize = groupsize
        ctx.split_row_offsets = split_row_offsets

        B, C, K, N = x_expanded.shape
        x_expanded = x_expanded.reshape(B, groups, groupsize, K, N)
        out = torch.einsum("bgckn,gock->bgon", x_expanded, weight.to(itype)).contiguous()
        return out.reshape(B, groups * weight.shape[1], N)

    @staticmethod
    def backward(ctx, grad_output):
        saved, weight, row_ker, row_pt, seg_off, seg, val_off, vals, ring_base, ring_size, split_ker = ctx.saved_tensors

        itype = grad_output.dtype
        vals_c = vals.to(_compute_dtype(itype))

        K = ctx.kernel_size
        G, Cg = ctx.groups, ctx.groupsize
        N = ctx.npoints_out
        Og = weight.shape[1]
        B = grad_output.shape[0]
        grad_output_r = grad_output.reshape(B, G, Og, N)

        grad_inp = None
        grad_weight = None

        if ctx.needs_input_grad[0]:
            if split_ker is not None and _use_spatial_first_dgrad(Og, Cg, K):
                grad_inp = _spatial_first_dgrad_ragged(
                    grad_output_r, weight.to(itype), row_pt, seg_off, seg, val_off, vals_c, ring_base, ring_size, split_ker, K, ctx.npoints_in, ctx.split_row_offsets
                )
            else:
                grad_x_expanded = torch.einsum("bgon,gock->bgckn", grad_output_r, weight.to(itype))
                grad_x_expanded = grad_x_expanded.reshape(B, G * Cg, K, N).contiguous()
                grad_inp = disco_kernels.backward_ragged.default(grad_x_expanded, row_ker, row_pt, seg_off, seg, val_off, vals_c, ring_base, ring_size, K, ctx.npoints_in)
            grad_inp = grad_inp.to(itype)

        if ctx.needs_input_grad[1]:
            if ctx.recompute:
                x_expanded = disco_kernels.forward_ragged.default(saved, row_ker, row_pt, seg_off, seg, val_off, vals_c, ring_base, ring_size, K, N)
            else:
                x_expanded = saved
            x_expanded = x_expanded.to(itype).reshape(B, G, Cg, K, N)
            grad_weight = torch.einsum("bgon,bgckn->gock", grad_output_r, x_expanded)

        # inp, weight, then the eight ragged arc arrays, split_ker, kernel_size,
        # npoints_out, groups, groupsize, recompute, split_row_offsets
        return (grad_inp, grad_weight) + (None,) * 15


def _disco_s2_conv_ragged_optimized(inp, weight, arcs, split, kernel_size, npoints_out, groups, groupsize, recompute=False):
    """
    Ragged contraction plus weight contraction through :class:`_DiscoRaggedConvFn`.

    Parameters
    ----------
    inp : torch.Tensor
        ``(B, groups * groupsize, npoints_in)``.
    weight : torch.Tensor
        ``(groups, out_per_group, groupsize, kernel_size)``.
    arcs : Tuple[torch.Tensor, ...]
        ``(row_ker, row_pt, seg_off, seg, val_off, vals, ring_base, ring_size)``.
    split : Optional[Tuple]
        ``(split_ker, row_offsets)`` from build_split, to allow the spatial-first input
        gradient.
    recompute : bool
        Recompute the K-expanded intermediate in backward rather than saving it.
    """
    split_ker, row_offsets = split if split is not None else (None, ())
    return _DiscoRaggedConvFn.apply(inp, weight, *arcs, split_ker, kernel_size, npoints_out, groups, groupsize, recompute, row_offsets)
