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

import functools
from typing import Optional

import torch
from disco_helpers import (
    kpacked_sm90_kernels_is_available,
    kpacked_sm100_kernels_is_available,
    optimized_kernels_is_available,
)

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
    checked when a backend is selected for it, see KpackedBackend.available.
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
    outside those targets falls back to the CSR path, which is correct
    everywhere.
    """
    major, minor = torch.cuda.get_device_capability(device_index)
    if major == 9 and minor in _KPACKED_SM90_MINORS:
        return kpacked_sm90_kernels_is_available()
    if major == 10 and minor in _KPACKED_SM100_MINORS:
        return kpacked_sm100_kernels_is_available()
    return False


def _maybe_kpack_psi(psi_packed_idx, psi_packed_vals, psi_packed_count, n_align: int = 8):
    """Convert pack_psi_dense output [K,Ho,NBR_PAD,*] to the blocked-CSR kpacked layout.

    Returns (kpacked_idx, kpacked_vals, kpacked_offset, K_pad) or None if the
    per-k support sets differ across k_kern (layout mismatch).

    Inputs (from pack_psi_dense):
        psi_packed_idx   : [K, Ho, NBR_PAD, 2]   int64
        psi_packed_vals  : [K, Ho, NBR_PAD]      fp32
        psi_packed_count : [K, Ho]               int64
    Outputs:
        kpacked_idx      : [nnz, 2]              int64
        kpacked_vals     : [nnz, K_pad]          fp32   (zero-padded in k)
        kpacked_offset   : [Ho + 1]              int64  (prefix sum of per-ho counts)
        K_pad            : int  (K rounded up to next multiple of n_align)

    Blocked CSR: row offsets over ho, and each neighbour carries all K_pad values
    contiguously as one block. The block layout is what makes the tensor-core
    kernel possible -- it contracts over nz with k as the MMA's N dimension -- so
    this is not the serial path's CSR, where k is a *row* dimension and each
    nonzero holds a single scalar.

    The rows used to be padded to NBR_PAD = max_ho cnt(ho), a stride set by the
    polar rows where the cutoff spans the whole longitude circle while the mean
    row is far shorter. That cost ~33 MB of pack_val at half degree and ~274 MB at
    1080x2160 -> 360x720, against ~2 MB and ~16 MB of real data. The padding was
    never read by the kernel, so compacting is a footprint fix rather than a
    speed one -- but at quarter degree and finer the padded form approaches a
    gigabyte per layer, which stops being merely wasteful.

    kpacked_offset replaces the previous kpacked_offset: cnt(ho) is recoverable as
    offset[ho+1] - offset[ho], so the op keeps its arity.

    No alignment padding is needed between rows. Both wide accesses in the kernel
    land on 16-byte boundaries for any offset, because each neighbour occupies
    K_pad*sizeof(T) bytes (32 at K_pad=16, 16 at K_pad=8) in pack_val and 16 bytes
    in pack_idx -- all multiples of 16.
    """
    K = int(psi_packed_count.shape[0])
    K_pad = ((K + n_align - 1) // n_align) * n_align

    if psi_packed_count.shape[0] > 1:
        # The K-packed layout needs one idx/count per ho shared across all k.
        if not torch.equal(psi_packed_count, psi_packed_count[0:1].expand_as(psi_packed_count)):
            return None
        if not torch.equal(psi_packed_idx, psi_packed_idx[0:1].expand_as(psi_packed_idx)):
            return None

    counts = psi_packed_count[0].contiguous()  # [Ho]
    Ho = int(counts.numel())
    NBR_PAD = int(psi_packed_vals.shape[2])

    kpacked_offset = torch.zeros(Ho + 1, dtype=counts.dtype, device=counts.device)
    kpacked_offset[1:] = torch.cumsum(counts, dim=0)

    # Row-major mask over [Ho, NBR_PAD] selecting each row's first cnt(ho) entries,
    # so the gathered order is exactly ho-major then nz -- i.e. the CSR order the
    # offsets describe.
    valid = torch.arange(NBR_PAD, device=counts.device).unsqueeze(0) < counts.unsqueeze(1)

    kpacked_idx = psi_packed_idx[0][valid].contiguous()  # [nnz, 2]

    vals_perm = psi_packed_vals.permute(1, 2, 0)  # [Ho, NBR_PAD, K]
    vals_sel = vals_perm[valid]  # [nnz, K]
    if K_pad == K:
        kpacked_vals = vals_sel.contiguous()
    else:
        kpacked_vals = torch.zeros(vals_sel.shape[0], K_pad, dtype=vals_sel.dtype, device=vals_sel.device)
        kpacked_vals[:, :K] = vals_sel
        kpacked_vals = kpacked_vals.contiguous()

    return kpacked_idx, kpacked_vals, kpacked_offset, K_pad


def _kpacked_k_pad(kernel_size: int, n_align: int = 8) -> Optional[int]:
    """The K padding the kpacked kernels would use, or None if they have no instantiation for it.

    The tensor-core kernels take k as the MMA's N dimension and are instantiated for
    N = 8 and N = 16 only, so a basis with more than 16 functions stays on the CSR path.
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


def _build_kernel_split_csr(roff_idx: torch.Tensor, ker_idx: torch.Tensor, kernel_size: int):
    """
    Index the CSR psi per basis function, for the spatial-first input gradient.

    That gradient calls the scatter kernel once per basis function k, each time with a
    psi holding only k's rows and K = 1. :func:`preprocess_psi` has already sorted the
    entries by k, so k's entries are one contiguous block of the full arrays and nothing
    needs copying: all that is new is a row-offset table relative to each block, and the
    block boundaries. The kernel also reads ``ker_idx``, which for a K = 1 call has to be
    zero throughout; one shared zero vector, as long as the largest block, serves every k.

    Returns
    -------
    split_roff_idx : torch.Tensor
        The per-k row offsets, concatenated; k's table is
        ``split_roff_idx[row_offsets[k]:row_offsets[k + 1]]`` and starts at 0.
    split_ker_idx : torch.Tensor
        Zeros, as long as the largest per-k block.
    row_offsets, nnz_offsets : Tuple[int, ...]
        Python block boundaries into ``split_roff_idx`` and into the entry arrays. Python
        ints on purpose: they slice tensors inside the backward, where a tensor-valued
        bound would force a device sync, or a graph break under torch.compile.
    """
    roff_idx = roff_idx.cpu()
    ker_idx = ker_idx.cpu()
    nrows = roff_idx.numel() - 1

    # the basis function of every row, and how many rows and entries each has
    row_ker = ker_idx[roff_idx[:-1]] if nrows > 0 else ker_idx.new_empty((0,))
    rows_per_k = torch.bincount(row_ker, minlength=kernel_size).tolist()
    nnz_per_k = torch.bincount(ker_idx, minlength=kernel_size).tolist()

    parts, row_offsets, nnz_offsets = [], [0], [0]
    row_start = 0
    for k in range(kernel_size):
        # k's rows are contiguous, so its table is a slice of the full one, rebased to 0;
        # an empty k gets the one-entry table [0], i.e. zero rows
        table = roff_idx[row_start : row_start + rows_per_k[k] + 1]
        parts.append(table - table[0])
        row_start += rows_per_k[k]
        row_offsets.append(row_offsets[-1] + rows_per_k[k] + 1)
        nnz_offsets.append(nnz_offsets[-1] + nnz_per_k[k])

    split_roff_idx = torch.cat(parts).contiguous()
    split_ker_idx = torch.zeros(max(nnz_per_k, default=0), dtype=ker_idx.dtype, device=ker_idx.device)
    return split_roff_idx, split_ker_idx, tuple(row_offsets), tuple(nnz_offsets)


def _spatial_first_dgrad(grad_output_r, weight, split_roff_idx, split_ker_idx, row_idx, col_idx, vals, kernel_size, nlat_in, nlon_in, row_offsets, nnz_offsets):
    """Input gradient with the sparse transpose first, one launch per basis function; see _use_spatial_first_dgrad."""
    B, G, Og, H, W = grad_output_r.shape
    Cg = weight.shape[2]
    grad_small = grad_output_r.reshape(B, G * Og, 1, H, W).contiguous()

    parts = []
    for k in range(kernel_size):
        roff_k = split_roff_idx[row_offsets[k] : row_offsets[k + 1]]
        if roff_k.numel() <= 1:
            parts.append(grad_output_r.new_zeros((B, G, Og, nlat_in, nlon_in)))
            continue
        start, end = nnz_offsets[k], nnz_offsets[k + 1]
        part = disco_kernels.backward_regular.default(
            grad_small, roff_k, split_ker_idx[: end - start], row_idx[start:end], col_idx[start:end], vals[start:end], 1, nlat_in, nlon_in
        )
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


# custom kernels
if optimized_kernels_is_available():

    @torch.library.register_fake("disco_kernels::forward_regular")
    def _(
        inp: torch.Tensor,
        roff_idx: torch.Tensor,
        ker_idx: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        vals: torch.Tensor,
        kernel_size: int,
        nlat_out: int,
        nlon_out: int,
    ) -> torch.Tensor:
        return inp.new_empty((inp.shape[0], inp.shape[1], kernel_size, nlat_out, nlon_out))

    @torch.library.register_fake("disco_kernels::backward_regular")
    def _(
        inp: torch.Tensor,
        roff_idx: torch.Tensor,
        ker_idx: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        vals: torch.Tensor,
        kernel_size: int,
        nlat_out: int,
        nlon_out: int,
    ) -> torch.Tensor:
        return inp.new_empty((inp.shape[0], inp.shape[1], nlat_out, nlon_out))

    @torch.library.register_fake("disco_kernels::forward_kpacked")
    def _(inp: torch.Tensor, pack_idx: torch.Tensor, pack_val: torch.Tensor, pack_offset: torch.Tensor, kernel_size: int, nlat_out: int, nlon_out: int) -> torch.Tensor:
        return inp.new_empty((inp.shape[0], inp.shape[1], kernel_size, nlat_out, nlon_out))

    # The contraction and its transpose, as custom ops with their own autograd: each is
    # the other's backward. The activation stays in its storage dtype, so fp16/bf16 reach
    # the kernel, which accumulates in fp32; vals is cast to the compute dtype, matching
    # the kernel's val.data_ptr<compute_t>(). fp32/fp64 pass through unchanged.
    @torch.library.custom_op("disco_kernels::_disco_s2_contraction_regular_optimized", mutates_args=())
    def _disco_s2_contraction_regular_optimized(
        inp: torch.Tensor,
        roff_idx: torch.Tensor,
        ker_idx: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        vals: torch.Tensor,
        kernel_size: int,
        nlat_out: int,
        nlon_out: int,
    ) -> torch.Tensor:
        itype = inp.dtype
        vals = vals.to(_compute_dtype(itype))
        out = disco_kernels.forward_regular.default(inp.contiguous(), roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out)
        return out.to(itype)

    @torch.library.custom_op("disco_kernels::_disco_s2_transpose_contraction_regular_optimized", mutates_args=())
    def _disco_s2_transpose_contraction_regular_optimized(
        inp: torch.Tensor,
        roff_idx: torch.Tensor,
        ker_idx: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        vals: torch.Tensor,
        kernel_size: int,
        nlat_out: int,
        nlon_out: int,
    ) -> torch.Tensor:
        itype = inp.dtype
        vals = vals.to(_compute_dtype(itype))
        out = disco_kernels.backward_regular.default(inp.contiguous(), roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out)
        return out.to(itype)

    @torch.library.register_fake("disco_kernels::_disco_s2_contraction_regular_optimized")
    def _(
        inp: torch.Tensor,
        roff_idx: torch.Tensor,
        ker_idx: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        vals: torch.Tensor,
        kernel_size: int,
        nlat_out: int,
        nlon_out: int,
    ) -> torch.Tensor:
        return inp.new_empty((inp.shape[0], inp.shape[1], kernel_size, nlat_out, nlon_out))

    @torch.library.register_fake("disco_kernels::_disco_s2_transpose_contraction_regular_optimized")
    def _(
        inp: torch.Tensor,
        roff_idx: torch.Tensor,
        ker_idx: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        vals: torch.Tensor,
        kernel_size: int,
        nlat_out: int,
        nlon_out: int,
    ) -> torch.Tensor:
        return inp.new_empty((inp.shape[0], inp.shape[1], nlat_out, nlon_out))


def _setup_context_contraction(ctx, inputs, output):
    inp, roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out = inputs
    ctx.save_for_backward(roff_idx, ker_idx, row_idx, col_idx, vals)
    ctx.kernel_size = kernel_size
    ctx.nlat_in = inp.shape[-2]
    ctx.nlon_in = inp.shape[-1]


def _contraction_bwd(ctx, grad_output):
    roff_idx, ker_idx, row_idx, col_idx, vals = ctx.saved_tensors
    grad_input = None
    if ctx.needs_input_grad[0]:
        grad_input = _disco_s2_transpose_contraction_regular_optimized(grad_output, roff_idx, ker_idx, row_idx, col_idx, vals, ctx.kernel_size, ctx.nlat_in, ctx.nlon_in)
    return grad_input, None, None, None, None, None, None, None, None


def _transpose_contraction_bwd(ctx, grad_output):
    roff_idx, ker_idx, row_idx, col_idx, vals = ctx.saved_tensors
    grad_input = None
    if ctx.needs_input_grad[0]:
        grad_input = _disco_s2_contraction_regular_optimized(grad_output, roff_idx, ker_idx, row_idx, col_idx, vals, ctx.kernel_size, ctx.nlat_in, ctx.nlon_in)
    return grad_input, None, None, None, None, None, None, None, None


if optimized_kernels_is_available():
    torch.library.register_autograd("disco_kernels::_disco_s2_contraction_regular_optimized", _contraction_bwd, setup_context=_setup_context_contraction)
    torch.library.register_autograd("disco_kernels::_disco_s2_transpose_contraction_regular_optimized", _transpose_contraction_bwd, setup_context=_setup_context_contraction)

    _register_autocast("disco_kernels::_disco_s2_contraction_regular_optimized", ("cuda", "cpu"))
    _register_autocast("disco_kernels::_disco_s2_transpose_contraction_regular_optimized", ("cuda", "cpu"))
    # the kpacked kernel exists on CUDA only
    _register_autocast("disco_kernels::forward_kpacked", ("cuda",))


def _contract(inp, roff_idx, ker_idx, row_idx, col_idx, vals, pack_idx, pack_val, pack_offset, kernel_size, nlat_out, nlon_out):
    """The raw forward contraction: tensor-core kpacked when its layout is given, CSR otherwise."""
    inp = inp.contiguous()
    if pack_idx is not None:
        return disco_kernels.forward_kpacked.default(inp, pack_idx, pack_val, pack_offset, kernel_size, nlat_out, nlon_out)
    itype = inp.dtype
    out = disco_kernels.forward_regular.default(inp, roff_idx, ker_idx, row_idx, col_idx, vals.to(_compute_dtype(itype)), kernel_size, nlat_out, nlon_out)
    return out.to(itype)


class _DiscoKpackedFn(torch.autograd.Function):
    """
    Kpacked forward contraction, CSR backward.

    The backward stays on the CSR scatter: it is input-pixel-parallel with no cross-CTA
    atomics, the right algorithm for overlapping support sets (the reason cuDNN uses an
    implicit GEMM rather than col2im for strided convolutions).
    """

    @staticmethod
    def forward(ctx, inp, pack_idx, pack_val, pack_offset, roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out):
        ctx.save_for_backward(roff_idx, ker_idx, row_idx, col_idx, vals)
        ctx.kernel_size = kernel_size
        ctx.nlat_in = inp.shape[-2]
        ctx.nlon_in = inp.shape[-1]
        return _contract(inp, None, None, None, None, None, pack_idx, pack_val, pack_offset, kernel_size, nlat_out, nlon_out)

    @staticmethod
    def backward(ctx, grad_output):
        roff_idx, ker_idx, row_idx, col_idx, vals = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            gtype = grad_output.dtype
            grad_input = disco_kernels.backward_regular.default(
                grad_output.contiguous(), roff_idx, ker_idx, row_idx, col_idx, vals.to(_compute_dtype(gtype)), ctx.kernel_size, ctx.nlat_in, ctx.nlon_in
            ).to(gtype)
        return (grad_input,) + (None,) * 11


def _disco_s2_contraction_kpacked(inp, pack_idx, pack_val, pack_offset, roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out):
    return _DiscoKpackedFn.apply(inp, pack_idx, pack_val, pack_offset, roff_idx, ker_idx, row_idx, col_idx, vals, kernel_size, nlat_out, nlon_out)


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
      it sees the weight -- the split-CSR tables are its state, and it is used exactly
      when they are passed.

    The forward contraction is the kpacked tensor-core kernel when its layout is passed,
    the CSR kernel otherwise; the backward is the CSR scatter either way, and so is the
    recompute. Every combination of those used to be a separate autograd path.
    """

    @staticmethod
    def forward(
        ctx,
        inp,
        weight,
        roff_idx,
        ker_idx,
        row_idx,
        col_idx,
        vals,
        pack_idx,
        pack_val,
        pack_offset,
        split_roff_idx,
        split_ker_idx,
        kernel_size,
        nlat_out,
        nlon_out,
        groups,
        groupsize,
        recompute,
        split_row_offsets,
        split_nnz_offsets,
    ):
        itype = inp.dtype
        inp = inp.contiguous()
        x_expanded = _contract(inp, roff_idx, ker_idx, row_idx, col_idx, vals, pack_idx, pack_val, pack_offset, kernel_size, nlat_out, nlon_out)

        ctx.save_for_backward(inp if recompute else x_expanded, weight, roff_idx, ker_idx, row_idx, col_idx, vals, split_roff_idx, split_ker_idx)
        ctx.recompute = recompute
        ctx.kernel_size = kernel_size
        ctx.nlat_in = inp.shape[-2]
        ctx.nlon_in = inp.shape[-1]
        ctx.nlat_out = nlat_out
        ctx.nlon_out = nlon_out
        ctx.groups = groups
        ctx.groupsize = groupsize
        ctx.split_row_offsets = split_row_offsets
        ctx.split_nnz_offsets = split_nnz_offsets

        B, C, K, H, W = x_expanded.shape
        x_expanded = x_expanded.reshape(B, groups, groupsize, K, H, W)
        out = torch.einsum("bgckxy,gock->bgoxy", x_expanded, weight.to(itype)).contiguous()
        return out.reshape(B, groups * weight.shape[1], H, W)

    @staticmethod
    def backward(ctx, grad_output):
        saved, weight, roff_idx, ker_idx, row_idx, col_idx, vals, split_roff_idx, split_ker_idx = ctx.saved_tensors

        itype = grad_output.dtype
        vals_c = vals.to(_compute_dtype(itype))

        K = ctx.kernel_size
        G, Cg = ctx.groups, ctx.groupsize
        H, W = ctx.nlat_out, ctx.nlon_out
        Og = weight.shape[1]
        B = grad_output.shape[0]
        grad_output_r = grad_output.reshape(B, G, Og, H, W)

        grad_inp = None
        grad_weight = None

        if ctx.needs_input_grad[0]:
            if split_roff_idx is not None and _use_spatial_first_dgrad(Og, Cg, K):
                grad_inp = _spatial_first_dgrad(
                    grad_output_r,
                    weight.to(itype),
                    split_roff_idx,
                    split_ker_idx,
                    row_idx,
                    col_idx,
                    vals_c,
                    K,
                    ctx.nlat_in,
                    ctx.nlon_in,
                    ctx.split_row_offsets,
                    ctx.split_nnz_offsets,
                )
            else:
                grad_x_expanded = torch.einsum("bgoxy,gock->bgckxy", grad_output_r, weight.to(itype))
                grad_x_expanded = grad_x_expanded.reshape(B, G * Cg, K, H, W).contiguous()
                grad_inp = disco_kernels.backward_regular.default(grad_x_expanded, roff_idx, ker_idx, row_idx, col_idx, vals_c, K, ctx.nlat_in, ctx.nlon_in)
            grad_inp = grad_inp.to(itype)

        if ctx.needs_input_grad[1]:
            if ctx.recompute:
                x_expanded = disco_kernels.forward_regular.default(saved, roff_idx, ker_idx, row_idx, col_idx, vals_c, K, H, W)
            else:
                x_expanded = saved
            x_expanded = x_expanded.to(itype).reshape(B, G, Cg, K, H, W)
            grad_weight = torch.einsum("bgoxy,bgckxy->gock", grad_output_r, x_expanded)

        return (grad_inp, grad_weight) + (None,) * 18


def _disco_s2_conv_optimized(inp, weight, csr, kpacked, split, kernel_size, nlat_out, nlon_out, groups, groupsize, recompute=False):
    """
    Contraction plus weight contraction through :class:`_DiscoConvFn`.

    Parameters
    ----------
    inp : torch.Tensor
        ``(B, groups * groupsize, H_in, W_in)``.
    weight : torch.Tensor
        ``(groups, out_per_group, groupsize, kernel_size)``.
    csr : Tuple[torch.Tensor, ...]
        ``(roff_idx, ker_idx, row_idx, col_idx, vals)``, always: the backward reads it.
    kpacked : Optional[Tuple[torch.Tensor, ...]]
        ``(pack_idx, pack_val, pack_offset)`` to run the forward on the tensor cores.
    split : Optional[Tuple]
        ``(split_roff_idx, split_ker_idx, row_offsets, nnz_offsets)`` from
        :func:`_build_kernel_split_csr`, to allow the spatial-first input gradient.
    recompute : bool
        Recompute the K-expanded intermediate in backward rather than saving it.
    """
    pack_idx, pack_val, pack_offset = kpacked if kpacked is not None else (None, None, None)
    split_roff_idx, split_ker_idx, row_offsets, nnz_offsets = split if split is not None else (None, None, (), ())
    return _DiscoConvFn.apply(
        inp,
        weight,
        *csr,
        pack_idx,
        pack_val,
        pack_offset,
        split_roff_idx,
        split_ker_idx,
        kernel_size,
        nlat_out,
        nlon_out,
        groups,
        groupsize,
        recompute,
        row_offsets,
        nnz_offsets,
    )
