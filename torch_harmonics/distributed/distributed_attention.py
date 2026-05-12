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

from itertools import accumulate
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist

from torch_harmonics.attention.attention import NeighborhoodAttentionS2

from .utils import azimuth_group, polar_group
from .utils import polar_group_size, polar_group_rank
from .utils import azimuth_group_size, azimuth_group_rank
from .primitives import compute_split_shapes, get_group_neighbors, polar_halo_exchange

from attention_helpers import optimized_kernels_is_available
from torch_harmonics.attention import attention_kernels


# ---------------------------------------------------------------------------
# autograd.Function for the ring-step attention kernel calls
# ---------------------------------------------------------------------------

@torch.compiler.disable()
def _ring_kv(kw_chunk, vw_chunk, az_group, next_nlon_kw, next_nlon_kv):
    """Async send current chunks, receive next chunks with known shapes."""
    send_to, recv_from = get_group_neighbors(az_group)
    B, C_k, H, _ = kw_chunk.shape
    B, C_v, H, _ = vw_chunk.shape
    recv_kw = torch.empty(B, C_k, H, next_nlon_kw, device=kw_chunk.device, dtype=kw_chunk.dtype)
    recv_vw = torch.empty(B, C_v, H, next_nlon_kv, device=vw_chunk.device, dtype=vw_chunk.dtype)
    ops = [
        dist.P2POp(dist.isend, kw_chunk, send_to,   az_group),
        dist.P2POp(dist.irecv, recv_kw,  recv_from, az_group),
        dist.P2POp(dist.isend, vw_chunk, send_to,   az_group),
        dist.P2POp(dist.irecv, recv_vw,  recv_from, az_group),
    ]
    reqs = dist.batch_isend_irecv(ops)
    return recv_kw, recv_vw, reqs


class _RingNeighborhoodAttentionFn(torch.autograd.Function):
    """Forward ring attention + backward ring for one attention head group.

    kw, vw : [B*nh, C_k/C_v, H_halo, W_local]  channels-first, lat-halo-padded
    qw     : [B*nh, C_k,     H_out_local, W_out_local]  channels-first

    State buffers use channels-last layout as required by the CUDA kernels:
      y_acc        : [B, H_out, W_out, C_v]
      alpha_k/kvw  : [B, H_out, W_out, C_k]
      alpha_sum/qdotk_max/integral : [B, H_out, W_out]
    """

    @staticmethod
    def forward(
        kw, vw, qw,
        psi_col_idx, psi_roff_idx, psi_row_idx,
        quad_weights,
        nlon_in: int,
        pscale: int,
        lon_chunk_starts: list,
        nlon_kx_list: list,
        lat_halo_start: int,
        nlat_out_local: int,
        nlon_out_local: int,
        r_lat: int,
        az_group,
        az_rank: int,
        az_size: int,
    ):
        B, _, _, _  = kw.shape
        _, C_v, _,      _  = vw.shape
        device = kw.device

        # Allocate state buffers in formats expected by the CUDA kernels:
        # y_acc: channels-last [B, H, W, C_v];  scalars: [B, H, W]
        y_acc     = torch.zeros(B, nlat_out_local, nlon_out_local, C_v,
                                device=device, dtype=torch.float32)
        alpha_sum = torch.zeros(B, nlat_out_local, nlon_out_local,
                                device=device, dtype=torch.float32)
        qdotk_max = torch.full ((B, nlat_out_local, nlon_out_local), float('-inf'),
                                device=device, dtype=torch.float32)

        kw_chunk = kw.contiguous()
        vw_chunk = vw.contiguous()

        for step in range(az_size):
            src_rank  = (az_rank + step) % az_size
            lon_lo_kx = lon_chunk_starts[src_rank]

            # Pre-allocate receive buffers for the NEXT chunk (correct shape)
            if step < az_size - 1:
                next_src = (az_rank + step + 1) % az_size
                recv_kw, recv_vw, reqs = _ring_kv(
                    kw_chunk, vw_chunk, az_group,
                    nlon_kx_list[next_src], nlon_kx_list[next_src])

            attention_kernels.forward_ring_step.default(
                kw_chunk, vw_chunk, qw,
                y_acc, alpha_sum, qdotk_max,
                quad_weights, psi_col_idx, psi_roff_idx, psi_row_idx,
                nlon_in, pscale, lon_lo_kx, lat_halo_start,
                nlat_out_local, nlon_out_local,
            )

            if step < az_size - 1:
                for req in reqs:
                    req.wait()
                kw_chunk = recv_kw.clone()
                vw_chunk = recv_vw.clone()

        # Finalize: y = y_acc / alpha_sum  (both channels-last layout)
        y_out = y_acc / alpha_sum.unsqueeze(-1)           # [B, H, W, C_v]
        y_out = y_out.permute(0, 3, 1, 2).contiguous()   # [B, C_v, H, W]

        # alpha_sum and qdotk_max are returned so setup_context can save them;
        # they are marked non-differentiable there, so backward still only
        # receives one gradient argument (dy for y_out).
        return y_out, alpha_sum, qdotk_max

    @staticmethod
    def setup_context(ctx, inputs, output):
        (kw, vw, qw,
         psi_col_idx, psi_roff_idx, psi_row_idx,
         quad_weights,
         nlon_in, pscale, lon_chunk_starts, nlon_kx_list,
         lat_halo_start, nlat_out_local, nlon_out_local,
         r_lat, az_group, az_rank, az_size) = inputs
        y_out, alpha_sum, qdotk_max = output
        # alpha_sum and qdotk_max are internal accumulators, not true outputs;
        # marking them non-differentiable keeps backward's signature as (ctx, dy).
        ctx.mark_non_differentiable(alpha_sum, qdotk_max)
        ctx.save_for_backward(kw, vw, qw, psi_col_idx, psi_roff_idx, psi_row_idx,
                              quad_weights, alpha_sum, qdotk_max)
        ctx.nlon_in          = nlon_in
        ctx.pscale           = pscale
        ctx.lon_chunk_starts = lon_chunk_starts
        ctx.nlon_kx_list     = nlon_kx_list
        ctx.lat_halo_start   = lat_halo_start
        ctx.nlat_out_local   = nlat_out_local
        ctx.nlon_out_local   = nlon_out_local
        ctx.az_group         = az_group
        ctx.az_rank          = az_rank
        ctx.az_size          = az_size

    @staticmethod
    def backward(ctx, dy, _dalpha_sum, _dqdotk_max):
        # _dalpha_sum and _dqdotk_max are always None (non-differentiable outputs)
        (kw, vw, qw,
         psi_col_idx, psi_roff_idx, psi_row_idx,
         quad_weights,
         fwd_alpha_sum, fwd_qdotk_max) = ctx.saved_tensors

        nlon_in          = ctx.nlon_in
        pscale           = ctx.pscale
        lon_chunk_starts = ctx.lon_chunk_starts
        nlon_kx_list     = ctx.nlon_kx_list
        lat_halo_start   = ctx.lat_halo_start
        nlat_out_local   = ctx.nlat_out_local
        nlon_out_local   = ctx.nlon_out_local
        az_group         = ctx.az_group
        az_rank          = ctx.az_rank
        az_size          = ctx.az_size

        # Autograd contract: skip per-branch work (kernel calls, allreduces) for any
        # of {kw, vw, qw} that doesn't need a gradient, and return None in those slots.
        # This is what lets torch.compile / AOTAutograd prune dead subgraphs (including
        # the NCCL allreduces) from the compiled backward.
        kw_needs = ctx.needs_input_grad[0]
        vw_needs = ctx.needs_input_grad[1]
        qw_needs = ctx.needs_input_grad[2]

        # Defensive: if somehow none of (kw, vw, qw) need grad (e.g., user wired
        # requires_grad onto one of the index buffers), there's nothing to compute.
        if not (kw_needs or vw_needs or qw_needs):
            return (None,) * 18

        B, C_k, H_halo, _ = kw.shape
        _, C_v, _,      _ = vw.shape
        device = kw.device

        dy_cf = dy.contiguous()  # channels-first [B, C_v, H, W]

        # ----------------------------------------------------------------
        # Backward pass 1: re-accumulate {alpha_sum, qdotk_max, integral,
        #                                  alpha_k, alpha_kvw} via ring.
        # Required whenever any of (kw, vw, qw) needs grad: integral feeds
        # pass-2's integral_norm and dqy reads alpha_k/alpha_kvw. The kernel
        # writes all three buffers in one call, so pass-1 cannot be pruned
        # per-branch.
        # ----------------------------------------------------------------
        bwd_alpha_sum = torch.zeros(B, nlat_out_local, nlon_out_local,
                                    device=device, dtype=torch.float32)
        bwd_qdotk_max = torch.full ((B, nlat_out_local, nlon_out_local), float('-inf'),
                                    device=device, dtype=torch.float32)
        integral_buf  = torch.zeros_like(bwd_alpha_sum)
        alpha_k_buf   = torch.zeros(B, nlat_out_local, nlon_out_local, C_k,
                                    device=device, dtype=torch.float32)
        alpha_kvw_buf = torch.zeros_like(alpha_k_buf)

        kw_chunk = kw.contiguous()
        vw_chunk = vw.contiguous()

        for step in range(az_size):
            src_rank  = (az_rank + step) % az_size
            lon_lo_kx = lon_chunk_starts[src_rank]

            if step < az_size - 1:
                next_src = (az_rank + step + 1) % az_size
                recv_kw, recv_vw, reqs = _ring_kv(
                    kw_chunk, vw_chunk, az_group,
                    nlon_kx_list[next_src], nlon_kx_list[next_src])

            attention_kernels.backward_ring_step_pass1.default(
                kw_chunk, vw_chunk, qw, dy_cf,
                bwd_alpha_sum, bwd_qdotk_max, integral_buf,
                alpha_k_buf, alpha_kvw_buf,
                quad_weights, psi_col_idx, psi_roff_idx, psi_row_idx,
                nlon_in, pscale, lon_lo_kx, lat_halo_start,
                nlat_out_local, nlon_out_local,
            )

            if step < az_size - 1:
                for req in reqs:
                    req.wait()
                kw_chunk = recv_kw.clone()
                vw_chunk = recv_vw.clone()

        # ----------------------------------------------------------------
        # Finalize pass-1 outputs.
        # Use the SAVED forward alpha_sum/qdotk_max (same values, but authoritative).
        # ----------------------------------------------------------------
        alpha_sum_inv = 1.0 / fwd_alpha_sum                              # [B, H, W]

        # integral_norm only feeds pass-2; skip if neither kw nor vw needs grad.
        if kw_needs or vw_needs:
            integral_norm = integral_buf * alpha_sum_inv                  # [B, H, W]

        # dqy[b,h,w,c] = inv_sq*(alpha_sum*alpha_kvw - integral*alpha_k)
        if qw_needs:
            alpha_sum_inv_sq = alpha_sum_inv ** 2
            dqy_cl = alpha_sum_inv_sq.unsqueeze(-1) * (
                fwd_alpha_sum.unsqueeze(-1) * alpha_kvw_buf
                - integral_buf.unsqueeze(-1) * alpha_k_buf
            )                                                              # [B, H, W, C_k]
            dqy = dqy_cl.permute(0, 3, 1, 2).contiguous()                # [B, C_k, H, W]
        else:
            dqy = None

        # ----------------------------------------------------------------
        # Backward pass 2: scatter dkw/dvw contributions.
        # Each GPU computes its contribution to every lon chunk it visits;
        # then allreduce across azimuth ranks, extract local chunk.
        # Skip entirely if neither kw nor vw needs grad. The fused kernel
        # writes both dkw_chunk_cl and dvw_chunk_cl in one call, so the
        # per-chunk allocations stay; we just gate the accumulation /
        # allreduce / extract per branch.
        # TODO: replace allreduce with ring reduce-scatter for efficiency.
        # ----------------------------------------------------------------
        if kw_needs or vw_needs:
            kw_chunk      = kw.contiguous()
            vw_chunk      = vw.contiguous()
            nlon_in_total = sum(nlon_kx_list)
            dkw_full_cl = torch.zeros(B, H_halo, nlon_in_total, C_k,
                                      device=device, dtype=torch.float32) if kw_needs else None
            dvw_full_cl = torch.zeros(B, H_halo, nlon_in_total, C_v,
                                      device=device, dtype=torch.float32) if vw_needs else None

            for step in range(az_size):
                src_rank  = (az_rank + step) % az_size
                lon_lo_kx = lon_chunk_starts[src_rank]
                nlon_kx   = nlon_kx_list[src_rank]

                # Channels-last gradient buffers for this chunk (both required by the
                # fused kernel signature; we discard the one we don't need).
                dkw_chunk_cl = torch.zeros(B, H_halo, nlon_kx, C_k,
                                           device=device, dtype=torch.float32)
                dvw_chunk_cl = torch.zeros(B, H_halo, nlon_kx, C_v,
                                           device=device, dtype=torch.float32)

                attention_kernels.backward_ring_step_pass2.default(
                    kw_chunk, vw_chunk, qw, dy_cf,
                    fwd_alpha_sum, fwd_qdotk_max, integral_norm,
                    dkw_chunk_cl, dvw_chunk_cl,
                    quad_weights, psi_col_idx, psi_roff_idx, psi_row_idx,
                    nlon_in, pscale, lon_lo_kx, lat_halo_start,
                    nlat_out_local, nlon_out_local,
                )

                if kw_needs:
                    dkw_full_cl[:, :, lon_lo_kx:lon_lo_kx + nlon_kx, :].add_(dkw_chunk_cl)
                if vw_needs:
                    dvw_full_cl[:, :, lon_lo_kx:lon_lo_kx + nlon_kx, :].add_(dvw_chunk_cl)

                if step < az_size - 1:
                    next_src = (az_rank + step + 1) % az_size
                    recv_kw, recv_vw, reqs = _ring_kv(
                        kw_chunk, vw_chunk, az_group,
                        nlon_kx_list[next_src], nlon_kx_list[next_src])
                    for req in reqs:
                        req.wait()
                    kw_chunk = recv_kw.clone()
                    vw_chunk = recv_vw.clone()

            # Per-branch allreduce — only the branches we'll return.
            if az_size > 1 and az_group is not None:
                if kw_needs:
                    dist.all_reduce(dkw_full_cl, group=az_group)
                if vw_needs:
                    dist.all_reduce(dvw_full_cl, group=az_group)

            my_lo   = lon_chunk_starts[az_rank]
            my_nlon = nlon_kx_list[az_rank]
            # Extract local chunk and convert channels-last → channels-first.
            # No halo stripping: dkw/dvw must match kw/vw shape (= key_halo/value_halo).
            # The autograd through torch.cat in _exchange_lat_halo extracts the
            # middle H_in rows as the gradient for key_proj/value_proj.
            if kw_needs:
                dkw_cl = dkw_full_cl[:, :, my_lo:my_lo + my_nlon, :].contiguous()
                dkw    = dkw_cl.permute(0, 3, 1, 2).contiguous()  # [B, C_k, H_halo, W_local]
            else:
                dkw = None
            if vw_needs:
                dvw_cl = dvw_full_cl[:, :, my_lo:my_lo + my_nlon, :].contiguous()
                dvw    = dvw_cl.permute(0, 3, 1, 2).contiguous()  # [B, C_v, H_halo, W_local]
            else:
                dvw = None
        else:
            dkw = None
            dvw = None

        # Return grads for (kw, vw, qw, psi_col, psi_roff, psi_row, quad_weights,
        #                   nlon_in, pscale, lon_chunk_starts, nlon_kx_list, lat_halo_start,
        #                   nlat_out_local, nlon_out_local, r_lat,
        #                   az_group, az_rank, az_size)
        return dkw, dvw, dqy, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Distributed Neighborhood Attention on the 2-sphere
# ---------------------------------------------------------------------------

class DistributedNeighborhoodAttentionS2(NeighborhoodAttentionS2):
    """
    Distributed neighborhood attention on the 2-sphere using a ring exchange
    strategy for the longitude dimension and halo exchange for the latitude
    dimension.

    Data is assumed to be split along both the latitude (polar group) and
    longitude (azimuth group) dimensions.  The forward pass uses ring exchange
    of key/value chunks over the azimuth group so that every output point can
    attend to its full spherical neighborhood.

    Inherits learnable parameters from :class:`NeighborhoodAttentionS2`.
    """

    def __init__(
        self,
        in_channels: int,
        in_shape: Tuple[int, int],
        out_shape: Tuple[int, int],
        grid_in: Optional[str] = "equiangular",
        grid_out: Optional[str] = "equiangular",
        num_heads: Optional[int] = 1,
        scale: Optional[Union[torch.Tensor, float]] = None,
        use_qknorm: Optional[bool] = False,
        bias: Optional[bool] = True,
        theta_cutoff: Optional[float] = None,
        k_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        optimized_kernel: Optional[bool] = True,
    ):
        if not optimized_kernels_is_available():
            raise RuntimeError("Optimized kernels are required to run DistributedNeighborhoodAttentionS2.")

        # initialise base class (builds global psi, creates parameters)
        super().__init__(
            in_channels, in_shape, out_shape,
            grid_in=grid_in, grid_out=grid_out,
            num_heads=num_heads, scale=scale, use_qknorm=use_qknorm, bias=bias,
            theta_cutoff=theta_cutoff, k_channels=k_channels,
            out_channels=out_channels,
            optimized_kernel=True,
        )

        # ---- distributed info ----
        self.comm_size_polar   = polar_group_size()
        self.comm_rank_polar   = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # split shapes
        self.lat_in_shapes  = compute_split_shapes(self.nlat_in,  self.comm_size_polar)
        self.lon_in_shapes  = compute_split_shapes(self.nlon_in,  self.comm_size_azimuth)
        self.lat_out_shapes = compute_split_shapes(self.nlat_out, self.comm_size_polar)
        self.lon_out_shapes = compute_split_shapes(self.nlon_out, self.comm_size_azimuth)

        # local sizes for this rank
        self.nlat_in_local  = self.lat_in_shapes[self.comm_rank_polar]
        self.nlon_in_local  = self.lon_in_shapes[self.comm_rank_azimuth]
        self.nlat_out_local = self.lat_out_shapes[self.comm_rank_polar]
        self.nlon_out_local = self.lon_out_shapes[self.comm_rank_azimuth]

        # Downsampling invariant: every azimuth rank must carry the same lon pscale.
        # The global `nlon_in % nlon_out == 0` check is inherited from the serial
        # NeighborhoodAttentionS2.__init__, but that is not sufficient in distributed:
        # if compute_split_shapes hands different ranks different local pscales
        # (e.g. nlon_in=12, nlon_out=4, comm_size_azimuth=3 -> [4,4,4] vs [2,1,1]),
        # the p-shift mapping in the ring exchange is ill-defined.
        pscale_lon = self.nlon_in // self.nlon_out
        for r, (lon_in_r, lon_out_r) in enumerate(zip(self.lon_in_shapes, self.lon_out_shapes)):
            if lon_in_r != pscale_lon * lon_out_r:
                raise ValueError(
                    f"DistributedNeighborhoodAttentionS2: inconsistent azimuth split at rank {r}: "
                    f"nlon_in_local={lon_in_r}, nlon_out_local={lon_out_r}. "
                    f"Every azimuth rank must satisfy nlon_in_local == (nlon_in // nlon_out) * nlon_out_local "
                    f"= {pscale_lon} * nlon_out_local. "
                    f"Choose (nlon_in, nlon_out, comm_size_azimuth) so that compute_split_shapes "
                    f"produces uniform local pscale."
                )

        # global lon offsets
        self.lon_in_starts  = list(accumulate([0] + self.lon_in_shapes[:-1]))
        self.lon_out_starts = list(accumulate([0] + self.lon_out_shapes[:-1]))
        self.lat_out_starts = list(accumulate([0] + self.lat_out_shapes[:-1]))

        self.lon_lo_out = self.lon_out_starts[self.comm_rank_azimuth]
        self.lat_lo_out = self.lat_out_starts[self.comm_rank_polar]

        # ---- build local psi ----
        # The global psi built by the base class covers all output lat rows.
        # We filter to only the rows owned by this rank and shift the wi
        # component of col_idx by lon_lo_out so that the kernel can use
        # local wo directly without knowing the global lon offset.
        self._build_local_psi()

        # ---- lat halo size ----
        # Compute r_lat from the global psi: maximum |hi_global - ho_global|
        # over all (ho, hi) pairs in the neighbourhood.
        # Use the lat_out_lo of our polar rank to compute ho_global.
        self.r_lat = self._compute_r_lat()

    # -----------------------------------------------------------------------

    def _build_local_psi(self):
        """Filter global psi to local output lat rows and shift col_idx wi."""

        lat_lo = self.lat_lo_out
        lat_hi = lat_lo + self.nlat_out_local

        # global psi from the base class (built over all nlat_out rows)
        col_idx_global  = self.psi_col_idx   # [nnz]        int64
        row_idx_global  = self.psi_row_idx   # [nnz]        int32
        roff_global     = self.psi_roff_idx  # [nlat_out+1] int64

        # psi_row_idx stores the sorted permutation: value is the row index.
        # psi_roff_idx[ho] .. psi_roff_idx[ho+1] gives entries for row ho.
        # (The row_idx buffer is the *sort order*, not the row indices directly.)
        # For the distributed case we rebuild roff for the local rows only.

        # Build local roff: select rows lat_lo..lat_hi-1
        roff_local = roff_global[lat_lo:lat_hi + 1] - roff_global[lat_lo]  # offset by first entry

        # Select the corresponding col_idx entries
        start = roff_global[lat_lo].item()
        end   = roff_global[lat_hi].item()
        col_idx_local = col_idx_global[start:end].clone()

        # Shift wi by pscale * lon_lo_out so the kernel can reconstruct wip from wo_local:
        # col stores hi_global * nlon_in + wi_canonical. For global wo_global = lon_lo_out + wo_local,
        # the target input column is (wi_canonical + pscale * wo_global) % nlon_in. The kernel evaluates
        # (wi_shifted + pscale * wo_local) % nlon_in, so pre-shifting by pscale * lon_lo_out absorbs
        # the rank-offset piece. pscale = 1 when nlon_in == nlon_out (same-shape case).
        nlon_in   = self.nlon_in
        lon_lo    = self.lon_lo_out
        pscale    = self.nlon_in // self.nlon_out
        hi_global = col_idx_local // nlon_in
        wi_canon  = col_idx_local - hi_global * nlon_in
        wi_shifted = (wi_canon + pscale * lon_lo) % nlon_in
        col_idx_shifted = hi_global * nlon_in + wi_shifted

        # Build sorted row_idx for local output rows (0-indexed within local range)
        # Reuse the serial sort order: just re-sort by nnz per local row
        nnz_per_row = (roff_local[1:] - roff_local[:-1]).cpu()
        row_idx_local = torch.argsort(nnz_per_row, descending=True).to(torch.int32)

        self.register_buffer("psi_col_idx_local",  col_idx_shifted, persistent=False)
        self.register_buffer("psi_roff_idx_local", roff_local,      persistent=False)
        self.register_buffer("psi_row_idx_local",  row_idx_local,   persistent=False)

    def _compute_r_lat(self) -> int:
        """Max lat halo radius needed across all polar ranks.

        Computed locally from the global psi (built identically on every rank
        by the base class), so no communication is required.
        """

        if polar_group_size() == 1:
            return 0

        col_idx = self.psi_col_idx  # global, all nlat_out rows
        if col_idx.numel() == 0:
            return 0

        lat_in_starts = list(accumulate([0] + self.lat_in_shapes[:-1]))
        roff = self.psi_roff_idx

        r = 0
        for rank in range(self.comm_size_polar):
            lat_in_lo  = lat_in_starts[rank]
            lat_in_hi  = lat_in_lo + self.lat_in_shapes[rank]
            lat_out_lo = self.lat_out_starts[rank]
            lat_out_hi = lat_out_lo + self.lat_out_shapes[rank]

            start = roff[lat_out_lo].item()
            end   = roff[lat_out_hi].item()
            if start == end:
                continue

            hi = (col_idx[start:end] // self.nlon_in).long()
            r_top = max(0, lat_in_lo - int(hi.min().item()))
            r_bot = max(0, int(hi.max().item()) - (lat_in_hi - 1))
            r = max(r, r_top, r_bot)

        return r

    # -----------------------------------------------------------------------

    def forward(
        self,
        query: torch.Tensor,
        key:   Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if key is None:
            key = query
        if value is None:
            value = query

        assert query.dim() == 4

        if query.shape[-2] != self.nlat_out_local or query.shape[-1] != self.nlon_out_local:
            raise ValueError(f"query spatial shape {(query.shape[-2], query.shape[-1])} does not match local out_shape {(self.nlat_out_local, self.nlon_out_local)}")
        if key.shape[-2] != self.nlat_in_local or key.shape[-1] != self.nlon_in_local:
            raise ValueError(f"key spatial shape {(key.shape[-2], key.shape[-1])} does not match local in_shape {(self.nlat_in_local, self.nlon_in_local)}")
        if value.shape[-2] != self.nlat_in_local or value.shape[-1] != self.nlon_in_local:
            raise ValueError(f"value spatial shape {(value.shape[-2], value.shape[-1])} does not match local in_shape {(self.nlat_in_local, self.nlon_in_local)}")

        # ---- 1. project to k/v/q ----
        key_proj   = nn.functional.conv2d(key,   self.k_weights, bias=self.k_bias)
        value_proj = nn.functional.conv2d(value, self.v_weights, bias=self.v_bias)
        query_proj = nn.functional.conv2d(query, self.q_weights, bias=self.q_bias)

        # QK normalization (must come before scale)
        if self.q_norm_weights is not None:
            B, C, H, W = query_proj.shape
            query_proj = query_proj.reshape(B, self.num_heads, -1, H, W).permute(0,1,3,4,2)
            query_proj = nn.functional.rms_norm(query_proj, normalized_shape=self.q_norm_weights.shape, weight=1 + self.q_norm_weights)
            query_proj = query_proj.permute(0,1,4,2,3).reshape(B, C, H, W).contiguous()

        if self.k_norm_weights is not None:
            B, C, H, W = key_proj.shape
            key_proj = key_proj.reshape(B, self.num_heads, -1, H, W).permute(0,1,3,4,2)
            key_proj = nn.functional.rms_norm(key_proj, normalized_shape=self.k_norm_weights.shape, weight=1 + self.k_norm_weights)
            key_proj = key_proj.permute(0,1,4,2,3).reshape(B, C, H, W).contiguous()

        # scale after normalization
        query_proj = query_proj * self.scale

        # fold num_heads into batch
        B, _, H, W = key_proj.shape
        key_proj   = key_proj.reshape(B * self.num_heads, -1, H, W)
        B, _, H, W = value_proj.shape
        value_proj = value_proj.reshape(B * self.num_heads, -1, H, W)
        B, _, H, W = query_proj.shape
        query_proj = query_proj.reshape(B * self.num_heads, -1, H, W)

        Bnh = B  # B*nh after reshape

        # ---- 2. lat halo exchange ----
        # key_proj/value_proj: [Bnh, C, H_in_local, W_in_local]
        # Use differentiable halo exchange when there is an actual polar split;
        # otherwise fall through to the identity (no-op).
        if self.r_lat > 0 and self.comm_size_polar > 1:
            key_halo   = polar_halo_exchange(key_proj,   self.r_lat)
            value_halo = polar_halo_exchange(value_proj, self.r_lat)
        else:
            key_halo   = key_proj
            value_halo = value_proj

        # global lat index of first halo row
        lat_in_starts = list(accumulate([0] + self.lat_in_shapes[:-1]))
        lat_halo_start = lat_in_starts[self.comm_rank_polar] - self.r_lat

        # ---- 3. ring attention ----
        # Global pscale — the kernel must not infer this from local shapes,
        # because kernel `nlon_out` is nlon_out_local which differs when az_size > 1.
        pscale = self.nlon_in // self.nlon_out
        out, _, _ = _RingNeighborhoodAttentionFn.apply(
            key_halo,
            value_halo,
            query_proj,
            self.psi_col_idx_local,
            self.psi_roff_idx_local,
            self.psi_row_idx_local,
            self.quad_weights,
            self.nlon_in,
            pscale,
            self.lon_in_starts,         # lon chunk starts for kv (same as lon_in)
            self.lon_in_shapes,         # lon chunk sizes for kv
            lat_halo_start,
            self.nlat_out_local,
            self.nlon_out_local,
            self.r_lat,
            azimuth_group(),
            self.comm_rank_azimuth,
            self.comm_size_azimuth,
        )  # [Bnh, C_v, H_out_local, W_out_local]

        # unfold num_heads
        B_nh, C_v, H_out, W_out = out.shape
        B_orig = B_nh // self.num_heads
        out = out.reshape(B_orig, self.num_heads * C_v, H_out, W_out)

        # ---- 4. output projection ----
        out = nn.functional.conv2d(out, self.proj_weights, bias=self.proj_bias)

        return out
