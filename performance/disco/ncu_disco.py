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

"""
Standalone ncu driver for the DISCO CUDA kernels.

Calls the raw torch ops (``disco_kernels::forward_kpacked``, ``::forward``,
``::backward``) in a tight loop with nothing else on the stream, so a profiler
sees exactly one kernel of interest per iteration. The psi data structures are
built by constructing a real ``DiscreteContinuousConvS2`` and lifting its
buffers; the module's forward (and its weight einsum) is never invoked.

Configs are the half- and quarter-degree shapes from ``benchmarks/disco.py``,
in both pscale = 1 (self-conv) and pscale = 2 (downsampling) flavours, since
those are the two p-shifts the kpacked path is tuned for.

Usage
-----
Sanity check + psi arc statistics, no profiler (fast, do this first):

    python performance/disco/ncu_disco.py --list
    python performance/disco/ncu_disco.py --config hdeg_self_tc003 --stats-only
    python performance/disco/ncu_disco.py --config hdeg_down_tc003 --stats-only

Profile the kpacked forward. ``--profile-from-start off`` pairs with the
cudaProfilerStart/Stop this script issues around the measured loop, so warmup
and psi construction are excluded:

    ncu --profile-from-start off \\
        --kernel-name regex:"disco_fwd_dense_kpacked_(wgmma|tcgen05)_blk_k" \\
        --set full -o disco_kpacked_hdeg \\
        python performance/disco/ncu_disco.py --config hdeg_self_tc003 --op fwd_kpacked --iters 3

The targeted metric set for the staging question (cheaper than --set full, and
these are the counters that decide whether the A-tile staging is bound on L1
wavefronts or on DRAM):

    ncu --profile-from-start off \\
        --kernel-name regex:"disco_fwd_dense_kpacked_(wgmma|tcgen05)_blk_k" \\
        --metrics $(python performance/disco/ncu_disco.py --print-metrics) \\
        python performance/disco/ncu_disco.py --config hdeg_self_tc003 --op fwd_kpacked --iters 3

Kernel names to filter on:
    fwd_kpacked     : disco_fwd_dense_kpacked_wgmma_blk_k    (SM_90a)
                      disco_fwd_dense_kpacked_tcgen05_blk_k  (SM_100a)
    fwd_kpacked_exp : disco_fwd_dense_kpacked_wgmma_exp_blk_k (SM_90a)
    fwd             : disco_fwd_blk_k
    bwd             : disco_bwd_blk_k

pack_idx staging experiment
---------------------------
``--op fwd_kpacked_exp`` drives the experimental kernel, whose only difference
from production is how pack_idx reaches each thread. pack_idx is ~19% of global
load instructions but ~27% of L1 sectors, and L1 is the measured ceiling.

    --variant 0   direct — the production staging, verbatim (control)
    --variant 1   vec    — one vector load of the (hi, wi) pair per thread
    --variant 2   shfl   — 16 lanes load, rest shuffle; no extra barrier
    --variant 3   smem   — 16 threads load per CTA; +1 __syncthreads per chunk
    --variant 4   astore — vec idx + A tile as one STS.128 instead of 8 STS.U16
    --idx-dtype {int64,int32}   orthogonal; int32 halves sectors for all variants

Measured on H100, hdeg_self_tc003, bf16, BC=64, int32 idx:

    variant        inst      sectors    L1 %   duration
    production  113.3M       650.8M     79.9   3.22 ms
    0 direct    113.3M       567.8M     79.9   3.25 ms
    1 vec       102.3M       525.6M     79.3   3.13 ms
    2 shfl      102.3M       525.6M     79.6   3.34 ms
    3 smem       94.1M       494.0M     81.8   3.24 ms

Instruction counts matched the model exactly (v1 saves 4 x 2,737,440, v3 a
further 3 x 2,737,440). Time did not follow: sectors fell 19%, duration 2.8%.

Two things that says. shfl saves no sectors at all -- the memory system already
deduplicates repeated addresses across lanes within a warp, so 32 lanes hitting
16 distinct entries touch the same lines as 16 lanes do; it only pays for the
shuffles. And smem's extra barrier converts long-scoreboard stall into barrier
stall almost one for one (4.51 -> 2.46 vs 2.54 -> 4.73), for no net gain.

More important: L1 throughput stayed pinned at ~80% while global sectors fell
24%. If global-load sectors were the saturated resource that number would have
dropped. So the saturated l1tex sub-pipe is one these variants do not touch --
and l1tex serves shared memory too. The A staging issues 8 STS.U16 per thread
per chunk alongside its 8 LDG.U16. Variant 4 and the shared-memory metrics
above exist to test exactly that.

Verify before believing any timing — the variants must be bit-identical:

    for v in 0 1 2 3; do
      python performance/disco/ncu_disco.py --config hdeg_self_tc003 \\
          --op fwd_kpacked_exp --variant $v --idx-dtype int32 --check --iters 1
    done

Then sweep under ncu:

    for v in 0 1 2 3; do
      ncu --profile-from-start off \\
          --kernel-name regex:"disco_fwd_dense_kpacked_wgmma_exp_blk_k" \\
          --metrics $(python performance/disco/ncu_disco.py --print-metrics) \\
          python performance/disco/ncu_disco.py --config hdeg_self_tc003 \\
              --op fwd_kpacked_exp --variant $v --iters 3
    done

Variant 0 should reproduce the production kernel's counters exactly; if it does
not, the harness is adding something and the rest of the sweep is untrustworthy.

Because psi construction is slow at quarter degree and you will want several
ncu runs over the same config, the built buffers are cached under --cache-dir.
"""

import argparse
import os
import sys
import time

import torch

import torch_harmonics  # noqa: F401  (registers the disco_kernels ops)
from torch_harmonics import DiscreteContinuousConvS2

# ------------------------------------------------------------------------------
# Configs — half / quarter degree, pscale 1 and 2
#
# pscale = nlon_in / nlon_out. The kpacked kernels take it as a runtime argument
# and only pscale 1 and 2 are of interest, so every entry here is one or the
# other. theta_cutoff values match benchmarks/disco.py.
# ------------------------------------------------------------------------------

_PROD_TC = 0.02771993517873347

CONFIGS = {
    # --- pscale = 1, self-conv -------------------------------------------------
    "1deg_self_tc003": dict(nlat_in=180, nlon_in=360, nlat_out=180, nlon_out=360, theta_cutoff=0.03),
    "hdeg_self_tc0017": dict(nlat_in=360, nlon_in=720, nlat_out=360, nlon_out=720, theta_cutoff=0.017),
    "hdeg_self_tc003": dict(nlat_in=360, nlon_in=720, nlat_out=360, nlon_out=720, theta_cutoff=0.03),
    "qdeg_self_tc0017": dict(nlat_in=720, nlon_in=1440, nlat_out=720, nlon_out=1440, theta_cutoff=0.017),
    "qdeg_self_tc003": dict(nlat_in=720, nlon_in=1440, nlat_out=720, nlon_out=1440, theta_cutoff=0.03),
    "prod_decoder": dict(nlat_in=721, nlon_in=1440, nlat_out=721, nlon_out=1440, theta_cutoff=_PROD_TC),
    # --- pscale = 2, downsampling ---------------------------------------------
    "hdeg_down_tc0017": dict(nlat_in=720, nlon_in=1440, nlat_out=360, nlon_out=720, theta_cutoff=0.017),
    "hdeg_down_tc003": dict(nlat_in=720, nlon_in=1440, nlat_out=360, nlon_out=720, theta_cutoff=0.03),
    "1deg_down_tc003": dict(nlat_in=360, nlon_in=720, nlat_out=180, nlon_out=360, theta_cutoff=0.03),
    "prod_encoder": dict(nlat_in=721, nlon_in=1440, nlat_out=360, nlon_out=720, theta_cutoff=_PROD_TC),
}

# The counters that answer "is the A-tile staging bound on L1 wavefronts or on
# DRAM". Kept as a list so --print-metrics can emit them for the ncu CLI.
METRICS = [
    # load instructions vs. sectors actually moved: the redundancy ratio
    "smsp__inst_executed_op_global_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    # shared-memory side of l1tex. The A-tile staging issues 8 STS.U16 per thread
    # per chunk alongside its 8 LDG.U16, and both go through l1tex — so the global
    # counters above only see half the traffic. L1 throughput stayed pinned at
    # ~80% across every pack_idx variant while global sectors fell 24%, which says
    # the saturated sub-pipe is one those variants do not touch.
    "smsp__inst_executed_op_shared_st.sum",
    "smsp__inst_executed_op_shared_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    # is L1 the ceiling?
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "lts__throughput.avg.pct_of_peak_sustained_active",
    "dram__bytes_read.sum",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    # where the warps actually sit
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    # occupancy: distinguishes "latency, not enough CTAs" from "request bound"
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__time_duration.sum",
]


# ------------------------------------------------------------------------------
# psi construction (cached — the CPU precompute is slow at quarter degree)
# ------------------------------------------------------------------------------


def _cache_key(cfg, kernel_shape, basis_type, basis_norm_mode):
    return "disco_psi_{}x{}_to_{}x{}_ks{}_{}_{}_tc{:.10f}.pt".format(
        cfg["nlat_in"],
        cfg["nlon_in"],
        cfg["nlat_out"],
        cfg["nlon_out"],
        "x".join(str(k) for k in kernel_shape),
        basis_type.replace(" ", "-"),
        basis_norm_mode,
        cfg["theta_cutoff"],
    )


def build_psi(cfg, kernel_shape, basis_type, basis_norm_mode, cache_dir, verbose=True):
    """Construct a conv module and lift its psi buffers. Returns a plain dict of
    CPU tensors; the caller moves what it needs to the device."""

    path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, _cache_key(cfg, kernel_shape, basis_type, basis_norm_mode))
        if os.path.exists(path):
            if verbose:
                print(f"[psi] loading cached buffers from {path}")
            return torch.load(path, weights_only=True)

    if verbose:
        print(f"[psi] building {cfg['nlat_in']}x{cfg['nlon_in']} -> {cfg['nlat_out']}x{cfg['nlon_out']}, " f"theta_cutoff={cfg['theta_cutoff']} (this is the slow part)")
    t0 = time.perf_counter()

    # in/out channels are irrelevant here: we call the raw ops and synthesise the
    # activation ourselves, so keep them small to avoid a pointless weight alloc.
    conv = DiscreteContinuousConvS2(
        in_channels=8,
        out_channels=8,
        in_shape=(cfg["nlat_in"], cfg["nlon_in"]),
        out_shape=(cfg["nlat_out"], cfg["nlon_out"]),
        kernel_shape=kernel_shape,
        basis_type=basis_type,
        basis_norm_mode=basis_norm_mode,
        theta_cutoff=cfg["theta_cutoff"],
        optimized_kernel=True,
        groups=1,
    )
    if verbose:
        print(f"[psi] built in {time.perf_counter() - t0:.1f} s")

    if not conv.optimized_kernel:
        raise RuntimeError("optimized kernels are not available in this build (disco_helpers/_C missing)")
    if conv.psi_kpacked_K_pad is None:
        raise RuntimeError(
            "kpacked buffers were not produced for this config: pack_psi_dense returned a psi whose "
            "neighbour list is not identical across k, so the K-packed layout does not apply."
        )

    out = dict(
        kernel_size=int(conv.kernel_size),
        K_pad=int(conv.psi_kpacked_K_pad),
        nlat_in=int(cfg["nlat_in"]),
        nlon_in=int(cfg["nlon_in"]),
        nlat_out=int(cfg["nlat_out"]),
        nlon_out=int(cfg["nlon_out"]),
        pack_idx=conv.psi_kpacked_idx.cpu(),
        pack_val=conv.psi_kpacked_vals.cpu(),
        pack_count=conv.psi_kpacked_count.cpu(),
        roff_idx=conv.psi_roff_idx.cpu(),
        ker_idx=conv.psi_ker_idx.cpu(),
        row_idx=conv.psi_row_idx.cpu(),
        col_idx=conv.psi_col_idx.cpu(),
        vals=conv.psi_vals.cpu(),
    )

    if path:
        torch.save(out, path)
        if verbose:
            print(f"[psi] cached to {path}")
    return out


# ------------------------------------------------------------------------------
# Arc statistics
#
# The proposed staging rewrite hinges on two properties of pack_idx, and both are
# cheap to check here rather than discovering them from a wrong kernel:
#
#   (a) for a given (ho, hi) the neighbour longitudes form a single contiguous
#       arc on the circle (possibly wrapping the seam), so a run-length encoding
#       is exact;
#   (b) the arcs are long enough that loading their union once beats the current
#       8 scalar loads per neighbour.
#
# For an arc of length L the current kernel issues WO_TILE*L element loads while
# touching only L + (WO_TILE-1)*pscale distinct elements; that ratio is the
# headroom. The chunk-aware variant accounts for NZ_CHUNK=16 tiling, which is
# what the kernel actually does.
# ------------------------------------------------------------------------------

WO_TILE = 8
NZ_CHUNK = 16


def _runs_of(w, nlon):
    """Run-length encode a sorted, unique longitude list into arc lengths.

    Consecutive runs are found by differencing. A run pair that touches both ends
    of the circle (0 and nlon-1) is one wrapping arc, so it is merged: sorted
    ascending, an arc such as {718, 719, 0, 1} appears as two runs at the ends of
    the list. Merging only on that condition matters — a single interior gap on
    its own means two genuinely separate arcs, not a wrap.
    """
    count = int(w.numel())
    d = torch.diff(w)
    breaks = (d != 1).nonzero().flatten()
    if breaks.numel() == 0:
        return [count]
    starts = torch.cat([torch.zeros(1, dtype=torch.int64), breaks + 1])
    ends = torch.cat([breaks + 1, torch.tensor([count], dtype=torch.int64)])
    runs = (ends - starts).tolist()
    if len(runs) > 1 and int(w[0]) == 0 and int(w[-1]) == nlon - 1:
        merged = runs[-1] + runs[0]
        runs = [merged] + runs[1:-1]
    return runs


def arc_stats(pack_idx, pack_count, nlon_in, pscale, verbose=True):
    Ho = int(pack_count.numel())
    idx = pack_idx.to(torch.int64)

    n_arcs_total = 0
    n_multiarc_groups = 0
    n_groups = 0
    arc_lens = []
    nnz_total = 0
    chunks_flat = 0  # ceil(cnt/16) — what the kernel launches today
    chunks_arcs = 0  # sum_arc ceil(len/16) — cost of iterating arcs with padding
    loads_now = 0  # element loads issued today
    loads_dedup = 0  # distinct elements, chunk-aware

    for ho in range(Ho):
        cnt = int(pack_count[ho])
        nnz_total += cnt
        if cnt == 0:
            continue
        chunks_flat += (cnt + NZ_CHUNK - 1) // NZ_CHUNK

        his = idx[ho, :cnt, 0]
        wis = idx[ho, :cnt, 1]
        for h in torch.unique(his):
            n_groups += 1
            w = torch.unique(wis[his == h])  # torch.unique returns sorted
            arcs = _runs_of(w, nlon_in)
            if len(arcs) > 1:
                n_multiarc_groups += 1

            for L in arcs:
                n_arcs_total += 1
                arc_lens.append(L)
                for c in range(0, L, NZ_CHUNK):
                    s = min(NZ_CHUNK, L - c)
                    loads_now += WO_TILE * s
                    loads_dedup += s + (WO_TILE - 1) * pscale
                    chunks_arcs += 1

    arc_lens_t = torch.tensor(arc_lens, dtype=torch.float64)
    stats = dict(
        Ho=Ho,
        nnz_total=nnz_total,
        nnz_per_ho=nnz_total / max(Ho, 1),
        n_groups=n_groups,
        n_arcs=n_arcs_total,
        n_multiarc_groups=n_multiarc_groups,
        arcs_per_ho=n_arcs_total / max(Ho, 1),
        arc_len_mean=float(arc_lens_t.mean()) if arc_lens else 0.0,
        arc_len_median=float(arc_lens_t.median()) if arc_lens else 0.0,
        arc_len_min=int(arc_lens_t.min()) if arc_lens else 0,
        arc_len_max=int(arc_lens_t.max()) if arc_lens else 0,
        chunks_flat=chunks_flat,
        chunks_arcs=chunks_arcs,
        load_ratio=loads_now / max(loads_dedup, 1),
    )

    if verbose:
        print("--- psi arc structure (per output row ho, over all input latitudes) ---")
        print(f"  nnz per ho              : {stats['nnz_per_ho']:.1f}  (total {stats['nnz_total']})")
        print(f"  (ho, hi) groups         : {stats['n_groups']}")
        print(f"  arcs                    : {stats['n_arcs']}  ({stats['arcs_per_ho']:.1f} per ho)")
        print(f"  groups that are NOT one arc : {stats['n_multiarc_groups']}", end="")
        if stats["n_multiarc_groups"] == 0:
            print("   <- single-arc property holds")
        else:
            print("   <- RLE still exact, but more segments than one per (ho, hi)")
        print(f"  arc length              : mean {stats['arc_len_mean']:.1f}, median {stats['arc_len_median']:.0f}, " f"min {stats['arc_len_min']}, max {stats['arc_len_max']}")
        print(f"  A-tile load redundancy  : {stats['load_ratio']:.2f}x  (element loads issued / distinct elements)")
        print(
            f"  MMA chunks: {stats['chunks_flat']} today vs {stats['chunks_arcs']} if iterating arcs "
            f"({stats['chunks_arcs'] / max(stats['chunks_flat'], 1):.2f}x — the zero-padding cost of arc iteration)"
        )
    return stats


# ------------------------------------------------------------------------------
# Op drivers
# ------------------------------------------------------------------------------


def make_fwd_kpacked(psi, device, dtype, args):
    """disco_kernels::forward_kpacked — the WGMMA / tcgen05 kernel."""
    inp = torch.randn(args.batch, args.channels, psi["nlat_in"], psi["nlon_in"], device=device, dtype=dtype)
    pack_idx = psi["pack_idx"].to(device)
    # Pre-cast: the op does pack_val.to(inp_dtype) internally, which would add a
    # cast kernel to every iteration and pollute the profile.
    pack_val = psi["pack_val"].to(device=device, dtype=dtype).contiguous()
    pack_count = psi["pack_count"].to(device)
    K, Ho, Wo = psi["kernel_size"], psi["nlat_out"], psi["nlon_out"]

    def run():
        return torch.ops.disco_kernels.forward_kpacked.default(inp, pack_idx, pack_val, pack_count, K, Ho, Wo)

    return run


def make_fwd(psi, device, dtype, args):
    """disco_kernels::forward — the scalar CSR kernel (disco_fwd_blk_k)."""
    inp = torch.randn(args.batch, args.channels, psi["nlat_in"], psi["nlon_in"], device=device, dtype=dtype)
    roff = psi["roff_idx"].to(device)
    ker = psi["ker_idx"].to(device)
    row = psi["row_idx"].to(device)
    col = psi["col_idx"].to(device)
    vals = psi["vals"].to(device)
    K, Ho, Wo = psi["kernel_size"], psi["nlat_out"], psi["nlon_out"]

    def run():
        return torch.ops.disco_kernels.forward.default(inp, roff, ker, row, col, vals, K, Ho, Wo)

    return run


def make_bwd(psi, device, dtype, args):
    """disco_kernels::backward — disco_bwd_blk_k.

    The backward transposes the grids: its input lives on the forward's *output*
    grid (B, C, K, nlat_out, nlon_out) and it writes the forward's *input* grid,
    so the Ho/Wo arguments are the forward's nlat_in/nlon_in. pscale = Wo/Wi is
    the same ratio as in the forward. vals must be in compute precision (fp32).
    """
    inp = torch.randn(args.batch, args.channels, psi["kernel_size"], psi["nlat_out"], psi["nlon_out"], device=device, dtype=dtype)
    roff = psi["roff_idx"].to(device)
    ker = psi["ker_idx"].to(device)
    row = psi["row_idx"].to(device)
    col = psi["col_idx"].to(device)
    vals = psi["vals"].to(device=device, dtype=torch.float32)
    K, Ho, Wo = psi["kernel_size"], psi["nlat_in"], psi["nlon_in"]

    def run():
        return torch.ops.disco_kernels.backward.default(inp, roff, ker, row, col, vals, K, Ho, Wo)

    return run


VARIANTS = {
    0: "direct — production staging, 2 scalar loads/thread (control)",
    1: "vec    — one vector load of the (hi, wi) pair per thread",
    2: "shfl   — lanes 0..15 load, rest take it by shuffle; no extra barrier",
    3: "smem   — threads 0..15 load for the whole CTA; +1 __syncthreads/chunk",
    4: "astore — vec idx staging + A tile written as one STS.128 instead of 8 STS.U16",
}


def make_fwd_kpacked_exp(psi, device, dtype, args):
    """disco_kernels::forward_kpacked_exp — pack_idx staging variants (SM_90a only).

    Variant 0 reproduces the production staging exactly, so it is the control:
    compare it against --op fwd_kpacked to confirm the harness adds nothing, then
    against variants 1-3 to isolate the staging change.
    """
    inp = torch.randn(args.batch, args.channels, psi["nlat_in"], psi["nlon_in"], device=device, dtype=dtype)
    idx_dtype = torch.int32 if args.idx_dtype == "int32" else torch.int64
    pack_idx = psi["pack_idx"].to(device=device, dtype=idx_dtype).contiguous()
    # The exp op refuses to cast pack_val itself, precisely so a conversion kernel
    # can never land inside the profiled loop.
    pack_val = psi["pack_val"].to(device=device, dtype=dtype).contiguous()
    pack_count = psi["pack_count"].to(device)
    K, Ho, Wo = psi["kernel_size"], psi["nlat_out"], psi["nlon_out"]
    variant = args.variant

    def run():
        return torch.ops.disco_kernels.forward_kpacked_exp.default(inp, pack_idx, pack_val, pack_count, K, Ho, Wo, variant)

    return run


OPS = {
    "fwd_kpacked": make_fwd_kpacked,
    "fwd_kpacked_exp": make_fwd_kpacked_exp,
    "fwd": make_fwd,
    "bwd": make_bwd,
}

KERNEL_NAMES = {
    "fwd_kpacked": "disco_fwd_dense_kpacked_wgmma_blk_k (SM_90a) / disco_fwd_dense_kpacked_tcgen05_blk_k (SM_100a)",
    "fwd_kpacked_exp": "disco_fwd_dense_kpacked_wgmma_exp_blk_k (SM_90a)",
    "fwd": "disco_fwd_blk_k",
    "bwd": "disco_bwd_blk_k",
}


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="hdeg_self_tc003", help="config name (see --list)")
    p.add_argument("--op", default="fwd_kpacked", choices=sorted(OPS), help="which op to drive")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--channels", type=int, default=64, help="BC = batch*channels sets grid.y of the kpacked kernel")
    p.add_argument(
        "--variant",
        type=int,
        default=0,
        choices=sorted(VARIANTS),
        help="fwd_kpacked_exp pack_idx staging mode: " + "; ".join(f"{k}={v.split(chr(32))[0]}" for k, v in VARIANTS.items()),
    )
    p.add_argument(
        "--idx-dtype",
        default="int64",
        choices=["int64", "int32"],
        help="fwd_kpacked_exp pack_idx precision; int32 halves the sectors of every variant",
    )
    p.add_argument("--iters", type=int, default=3, help="profiled iterations; keep small under ncu")
    p.add_argument("--warmup", type=int, default=5, help="iterations before cudaProfilerStart")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the synthetic activation")
    p.add_argument("--kernel-shape", type=int, nargs="+", default=[3, 3])
    p.add_argument("--basis-type", default="harmonic")
    p.add_argument("--basis-norm-mode", default="mean")
    p.add_argument(
        "--cache-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".psi_cache"),
        help="where to cache built psi buffers; empty string disables caching",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="for fwd_kpacked_exp: assert bit-identical output vs forward_kpacked before profiling",
    )
    p.add_argument("--stats", action="store_true", help="also print psi arc statistics (O(nnz) on CPU, not free)")
    p.add_argument("--stats-only", action="store_true", help="print shapes and arc statistics, run no kernels")
    p.add_argument("--time", action="store_true", help="also report wall time per iteration (not for use under ncu)")
    p.add_argument("--list", action="store_true", help="list configs and exit")
    p.add_argument("--print-metrics", action="store_true", help="print the ncu --metrics list and exit")
    args = p.parse_args()

    if args.print_metrics:
        print(",".join(METRICS))
        return 0

    if args.list:
        print("{:<20} {:>12} {:>12} {:>7} {:>14}".format("config", "in", "out", "pscale", "theta_cutoff"))
        for name, cfg in CONFIGS.items():
            shape_in = "{}x{}".format(cfg["nlat_in"], cfg["nlon_in"])
            shape_out = "{}x{}".format(cfg["nlat_out"], cfg["nlon_out"])
            pscale = cfg["nlon_in"] // cfg["nlon_out"]
            print("{:<20} {:>12} {:>12} {:>7} {:>14.6f}".format(name, shape_in, shape_out, pscale, cfg["theta_cutoff"]))
        return 0

    if args.config not in CONFIGS:
        print(f"unknown config '{args.config}'; use --list", file=sys.stderr)
        return 1
    cfg = CONFIGS[args.config]
    pscale = cfg["nlon_in"] // cfg["nlon_out"]
    dtype = getattr(torch, args.dtype)

    psi = build_psi(
        cfg,
        tuple(args.kernel_shape),
        args.basis_type,
        args.basis_norm_mode,
        args.cache_dir or None,
    )

    print(f"=== {args.config} / {args.op} ===")
    print(f"  grid      : {psi['nlat_in']}x{psi['nlon_in']} -> {psi['nlat_out']}x{psi['nlon_out']}  (pscale={pscale})")
    print(f"  K         : {psi['kernel_size']}   K_PAD: {psi['K_pad']}   NBR_PAD: {psi['pack_idx'].shape[1]}")
    print(f"  BC        : {args.batch * args.channels}  (batch={args.batch}, channels={args.channels})")
    print(f"  dtype     : {args.dtype}")
    if args.op == "fwd_kpacked":
        wo_per_ho = psi["nlon_out"] // WO_TILE
        bc_blocks = (args.batch * args.channels + 7) // 8
        print(f"  grid dims : ({psi['nlat_out'] * wo_per_ho}, {bc_blocks}) x 128 threads")
    if args.op == "fwd_kpacked_exp":
        print(f"  variant   : {args.variant} — {VARIANTS[args.variant]}")
        print(f"  idx dtype : {args.idx_dtype}")
    print(f"  kernel    : {KERNEL_NAMES[args.op]}")

    if args.stats or args.stats_only:
        arc_stats(psi["pack_idx"], psi["pack_count"], psi["nlon_in"], pscale)

    if args.stats_only:
        return 0

    if not torch.cuda.is_available():
        print("no CUDA device available", file=sys.stderr)
        return 1
    device = torch.device("cuda")
    major, minor = torch.cuda.get_device_capability(device)
    print(f"  device    : {torch.cuda.get_device_name(device)} (sm_{major}{minor})")
    if args.op == "fwd_kpacked" and major not in (9, 10):
        print(
            f"  WARNING: forward_kpacked requires SM_90a or SM_100a; this device is sm_{major}{minor} " "and the op will raise.",
            file=sys.stderr,
        )
    if args.op == "fwd_kpacked_exp" and major != 9:
        print(
            f"  WARNING: forward_kpacked_exp is SM_90a only; this device is sm_{major}{minor} and the op will raise.",
            file=sys.stderr,
        )
    if args.op.startswith("fwd_kpacked") and dtype == torch.float32:
        print("  WARNING: the kpacked kernels require bf16/fp16 input; use --dtype bfloat16", file=sys.stderr)

    # Every driver draws its input with torch.randn as its first act, so seeding
    # here and again before the reference gives both ops the same activation.
    torch.manual_seed(args.seed)
    run = OPS[args.op](psi, device, dtype, args)

    # The staging variants only change how (hi, wi) reaches the thread, never the
    # arithmetic, so they must agree with the production kernel bit for bit. A
    # variant that is fast because it stages the wrong neighbours would otherwise
    # look like a win.
    if args.check:
        if args.op != "fwd_kpacked_exp":
            print("  --check only applies to --op fwd_kpacked_exp", file=sys.stderr)
        else:
            torch.manual_seed(args.seed)
            ref = OPS["fwd_kpacked"](psi, device, dtype, args)()
            got = run()
            # Both kernel bodies are gated on __CUDA_ARCH_FEAT_SM90_ALL and compile
            # to nothing without TORCH_CUDA_ARCH_LIST="9.0a". Then both return the
            # zero-filled output tensor and a plain equality check passes while
            # nothing ran at all, so test for that first.
            if not ref.any():
                print(
                    "  check     : reference output is entirely zero — the kernel body was almost certainly " 'compiled out. Rebuild with TORCH_CUDA_ARCH_LIST="9.0a".',
                    file=sys.stderr,
                )
                return 1
            if torch.equal(ref, got):
                print(f"  check     : variant {args.variant}/{args.idx_dtype} bit-identical to forward_kpacked")
            else:
                bad = (ref != got).sum().item()
                maxdiff = (ref.float() - got.float()).abs().max().item()
                print(
                    f"  check     : MISMATCH — {bad} of {ref.numel()} elements differ, max |delta| {maxdiff}",
                    file=sys.stderr,
                )
                return 1
            del ref, got

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    # Everything before this point is excluded when ncu is invoked with
    # --profile-from-start off.
    torch.cuda.profiler.start()
    t0 = time.perf_counter()
    for i in range(args.iters):
        torch.cuda.nvtx.range_push(f"{args.op}_{i}")
        run()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    torch.cuda.profiler.stop()

    if args.time:
        print(f"  wall time : {1e3 * elapsed / args.iters:.3f} ms/iter over {args.iters} iters")
        print("              (meaningless under ncu — it serialises and replays kernels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
