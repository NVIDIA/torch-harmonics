# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal correctness/timing probe for DISCO kpacked WGMMA.

This script bypasses DiscreteContinuousConvS2 construction and calls
disco_kernels.forward_kpacked directly with synthetic packed data.  It is meant
to isolate the Hopper m64n8k16 path (N_PAD=8) from the known-good m64n16k16
path (N_PAD=16).

Usage:
    python performance/disco/minibench_kpacked_wgmma.py
    python performance/disco/minibench_kpacked_wgmma.py --dtype fp16 --n-pad 8
"""

import argparse
import statistics

import torch
from disco_helpers import cuda_kernels_is_available, optimized_kernels_is_available

from torch_harmonics import DiscreteContinuousConvS2
from torch_harmonics.disco import disco_kernels
from torch_harmonics.disco._disco_utils import _compute_dtype, _get_psi
from torch_harmonics.disco.kernels_torch.disco_torch import _disco_s2_contraction_torch

_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def _disable_tf32():
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cudnn.rnn.fp32_precision = "ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "conv", "csr", "einsum", "module"], default="synthetic")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bf16")
    parser.add_argument("--n-pad", type=int, nargs="+", default=[8, 16], choices=[8, 16])
    parser.add_argument("--kernel-size", type=int, default=None, help="Logical K passed to forward_kpacked; defaults to N_PAD.")
    parser.add_argument("--bc", type=int, default=8, help="Flattened B*C row count.")
    parser.add_argument("--hi", type=int, default=4)
    parser.add_argument("--wi", type=int, default=16)
    parser.add_argument("--ho", type=int, default=3)
    parser.add_argument("--wo", type=int, default=8)
    parser.add_argument("--nbr", type=int, default=16, help="Packed neighbor count per output latitude.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--dump", action="store_true", help="Print a small output/reference slice.")
    parser.add_argument("--debug", action="store_true", help="Print detailed synthetic-mode error localization diagnostics.")
    parser.add_argument("--debug-topk", type=int, default=8, help="Synthetic debug: number of worst elements/CTAs to print.")
    parser.add_argument("--debug-repeats", type=int, default=0, help="Synthetic debug: rerun the kernel N times and compare with the first output.")
    parser.add_argument("--debug-isolate-ho", action="store_true", help="Synthetic debug: rerun the worst output latitude as Ho=1.")
    parser.add_argument("--time-csr", action="store_true", help="Conv mode: also time raw CSR forward.")
    parser.add_argument("--time-module", action="store_true", help="Conv mode: also time DiscreteContinuousConvS2 forward.")
    parser.add_argument("--ncu", action="store_true", help="Conv mode: run an isolated Nsight Compute target launch.")
    parser.add_argument(
        "--ncu-target",
        choices=["raw_kpacked", "raw_csr", "module"],
        default="raw_kpacked",
        help="Conv mode --ncu: operation to profile.",
    )
    parser.add_argument("--ncu-launches", type=int, default=1, help="Conv mode --ncu: profiled launches after warmup.")

    parser.add_argument("--batch", type=int, default=8, help="Module mode batch size.")
    parser.add_argument("--channels", type=int, default=8, help="Conv/module mode input channel count.")
    parser.add_argument("--out-channels", type=int, default=2, help="Module mode output channel count.")
    parser.add_argument("--fused", action="store_true", help="Module mode: request fused optimized conv.")
    parser.add_argument("--in-shape", type=int, nargs=2, default=(16, 32))
    parser.add_argument("--out-shape", type=int, nargs=2, default=None)
    parser.add_argument("--kernel-shape-conv", type=int, nargs="+", default=[3, 3])
    parser.add_argument("--basis-type", default="harmonic")
    parser.add_argument("--basis-norm-mode", default="nodal")
    parser.add_argument("--grid-in", default="legendre-gauss")
    parser.add_argument("--grid-out", default="legendre-gauss")
    parser.add_argument("--theta-cutoff", type=float, default=0.05)
    parser.add_argument("--groups", type=int, default=1)
    return parser.parse_args()


def _check_runtime(device):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not optimized_kernels_is_available() or not cuda_kernels_is_available():
        raise RuntimeError("optimized CUDA DISCO kernels are not available")
    major, minor = torch.cuda.get_device_capability(device)
    if major not in (9, 10):
        raise RuntimeError(f"kpacked tensor-core path requires SM_90a or SM_100a, got SM_{major}{minor}")


def _make_inputs(args, dtype, device, n_pad, first8_vals=None):
    torch.manual_seed(args.seed)

    if args.wi % args.wo != 0:
        raise ValueError(f"--wi ({args.wi}) must be divisible by --wo ({args.wo})")
    if args.wo % 8 != 0:
        raise ValueError(f"--wo ({args.wo}) must be divisible by 8")
    if args.nbr <= 0:
        raise ValueError("--nbr must be positive")

    # Use B=1 and C=bc so each WGMMA M row maps directly to one channel row.
    inp = torch.randn(1, args.bc, args.hi, args.wi, device=device, dtype=torch.float32)
    inp = (inp * 0.25 + 0.1).to(dtype)

    pack_idx = torch.empty(args.ho, args.nbr, 2, device=device, dtype=torch.int64)
    for ho in range(args.ho):
        for nz in range(args.nbr):
            pack_idx[ho, nz, 0] = (nz + ho) % args.hi
            pack_idx[ho, nz, 1] = (3 * nz + 5 * ho) % args.wi

    vals = torch.randn(args.ho, args.nbr, n_pad, device=device, dtype=torch.float32)
    vals = vals * 0.2
    if first8_vals is not None:
        vals[:, :, :8] = first8_vals
    if n_pad == 16:
        # Keep the upper half nonzero so the n16 path still exercises both N groups.
        vals[:, :, 8:] += 0.05 * torch.arange(8, device=device, dtype=torch.float32)
    pack_val = vals.to(dtype).contiguous()
    pack_count = torch.full((args.ho,), args.nbr, device=device, dtype=torch.int64)

    return inp.contiguous(), pack_idx.contiguous(), pack_val, pack_count


def _reference(inp, pack_idx, pack_val, pack_count, kernel_size, ho, wo):
    bsz, channels, hi, wi = inp.shape
    pscale = wi // wo
    inp_f = inp.float().reshape(bsz * channels, hi, wi)
    val_f = pack_val.float()
    out = torch.zeros(bsz * channels, kernel_size, ho, wo, device=inp.device, dtype=torch.float32)

    for h in range(ho):
        count = int(pack_count[h].item())
        for nz in range(count):
            src_h = int(pack_idx[h, nz, 0].item())
            src_w_base = int(pack_idx[h, nz, 1].item())
            coeff = val_f[h, nz, :kernel_size]
            for w in range(wo):
                src_w = src_w_base + w * pscale
                if src_w >= wi:
                    src_w -= wi
                out[:, :, h, w] += inp_f[:, src_h, src_w].unsqueeze(1) * coeff.unsqueeze(0)

    return out.reshape(bsz, channels, kernel_size, ho, wo)


def _logical_kernel_size(args, n_pad):
    kernel_size = n_pad if args.kernel_size is None else args.kernel_size
    if kernel_size <= 0:
        raise ValueError("--kernel-size must be positive")
    if kernel_size > n_pad:
        raise ValueError(f"--kernel-size ({kernel_size}) must be <= N_PAD ({n_pad})")
    return kernel_size


def _conv_kernel_shape(args):
    if len(args.kernel_shape_conv) == 1:
        return args.kernel_shape_conv[0]
    if len(args.kernel_shape_conv) == 2:
        return tuple(args.kernel_shape_conv)
    raise ValueError("--kernel-shape-conv expects one or two integers")


def _run_once(args, dtype, device, n_pad, first8_vals=None):
    kernel_size = _logical_kernel_size(args, n_pad)
    inp, pack_idx, pack_val, pack_count = _make_inputs(args, dtype, device, n_pad, first8_vals=first8_vals)
    out = disco_kernels.forward_kpacked.default(inp, pack_idx, pack_val, pack_count, kernel_size, args.ho, args.wo)
    ref = _reference(inp, pack_idx, pack_val, pack_count, kernel_size, args.ho, args.wo).to(dtype)
    torch.cuda.synchronize(device)

    diff = (out.float() - ref.float()).abs()
    max_abs = float(diff.max().item())
    max_ref = float(ref.float().abs().max().item())
    max_rel = max_abs / max(max_ref, 1e-12)

    if args.dump:
        diff = (out.float() - ref.float()).abs()
        flat_idx = int(diff.nan_to_num(float("inf")).reshape(-1).argmax().item())
        idx = torch.unravel_index(torch.tensor(flat_idx, device=device), diff.shape)
        idx_tuple = tuple(int(x.item()) for x in idx)
        b, c, _, h, w = idx_tuple
        got = out[b, c, :, h, w].float().tolist()
        exp = ref[b, c, :, h, w].float().tolist()
        mid_h = args.ho // 2
        mid_w = args.wo // 2
        first_got = out[0, 0, :, 0, 0].float().tolist()
        first_exp = ref[0, 0, :, 0, 0].float().tolist()
        mid_got = out[0, 0, :, mid_h, mid_w].float().tolist()
        mid_exp = ref[0, 0, :, mid_h, mid_w].float().tolist()
        print(f"N_PAD={n_pad} max diff index={idx_tuple}")
        print(f"N_PAD={n_pad} sample out[{b},{c},:,{h},{w}] = {[round(x, 5) for x in got]}")
        print(f"N_PAD={n_pad} sample ref[{b},{c},:,{h},{w}] = {[round(x, 5) for x in exp]}")
        print(f"N_PAD={n_pad} sample out[0,0,:,0,0] = {[round(x, 5) for x in first_got]}")
        print(f"N_PAD={n_pad} sample ref[0,0,:,0,0] = {[round(x, 5) for x in first_exp]}")
        print(f"N_PAD={n_pad} midlat out[0,0,:,{mid_h},{mid_w}] = {[round(x, 5) for x in mid_got]}")
        print(f"N_PAD={n_pad} midlat ref[0,0,:,{mid_h},{mid_w}] = {[round(x, 5) for x in mid_exp]}")

    times = []
    for _ in range(args.warmup):
        disco_kernels.forward_kpacked.default(inp, pack_idx, pack_val, pack_count, kernel_size, args.ho, args.wo)
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(args.iters):
        start.record()
        disco_kernels.forward_kpacked.default(inp, pack_idx, pack_val, pack_count, kernel_size, args.ho, args.wo)
        end.record()
        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end))

    return {
        "out": out,
        "ref": ref,
        "inp": inp,
        "pack_idx": pack_idx,
        "pack_val": pack_val,
        "pack_count": pack_count,
        "kernel_size": kernel_size,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "max_ref": max_ref,
        "median_ms": statistics.median(times),
    }


def _unravel(flat_idx, shape, device):
    idx = torch.unravel_index(torch.tensor(int(flat_idx), device=device), shape)
    return tuple(int(x.item()) for x in idx)


def _print_worst_elements(label, diff, out, ref, pack_count, topk, threshold):
    device = diff.device
    diff_key = diff.nan_to_num(float("inf")).reshape(-1)
    topk = min(int(topk), diff_key.numel())
    if topk <= 0:
        return

    vals, flats = torch.topk(diff_key, topk)
    print(f"{label}: worst {topk} elements")
    for rank, (val, flat) in enumerate(zip(vals.tolist(), flats.tolist()), start=1):
        b, c, k, h, w = _unravel(flat, diff.shape, device)
        wo_strip = w // 8
        wo_local = w - wo_strip * 8
        bc_tile = c // 8
        bc_local = c - bc_tile * 8
        m_row = bc_local * 8 + wo_local
        count = int(pack_count[h].item())
        status = "bad" if val > threshold else "ok"
        print(
            f"  #{rank}: idx=(b={b}, c={c}, k={k}, h={h}, w={w}) "
            f"block=(x={h * (diff.shape[-1] // 8) + wo_strip}, y={bc_tile}) "
            f"locals=(bc={bc_local}, wo={wo_local}, m={m_row}) count={count} "
            f"abs={val:.6e} got={float(out[b, c, k, h, w].float().item()):.6e} "
            f"ref={float(ref[b, c, k, h, w].float().item()):.6e} [{status}]"
        )


def _print_synthetic_debug(args, n_pad, result, atol, rtol, device):
    out = result["out"]
    ref = result["ref"]
    inp = result["inp"]
    pack_idx = result["pack_idx"]
    pack_val = result["pack_val"]
    pack_count = result["pack_count"]
    kernel_size = result["kernel_size"]

    diff = (out.float() - ref.float()).abs()
    diff_key = diff.nan_to_num(float("inf"))
    threshold = atol + rtol * result["max_ref"]
    bad = diff_key > threshold
    nonfinite = ~torch.isfinite(out)

    print(f"debug N_PAD={n_pad}: threshold={threshold:.6e} " f"bad_elements={int(bad.sum().item())}/{bad.numel()} " f"nonfinite_out={int(nonfinite.sum().item())}")
    _print_tensor_stats("debug out", out)
    _print_tensor_stats("debug ref", ref)
    _print_tensor_stats("debug diff", diff)
    _print_worst_elements("debug element", diff, out, ref, pack_count, args.debug_topk, threshold)

    ho_error = diff_key.amax(dim=(0, 1, 2, 4))
    ho_bad = bad.sum(dim=(0, 1, 2, 4))
    ho_top = min(args.debug_topk, ho_error.numel())
    vals, hs = torch.topk(ho_error, ho_top)
    print(f"debug latitude: worst {ho_top} rows")
    for rank, (val, h) in enumerate(zip(vals.tolist(), hs.tolist()), start=1):
        h = int(h)
        count = int(pack_count[h].item())
        print(f"  #{rank}: h={h} count={count} max_abs={val:.6e} bad_elements={int(ho_bad[h].item())}")

    wo_per_ho = args.wo // 8
    block_error = diff_key.reshape(diff.shape[0], diff.shape[1], diff.shape[2], args.ho, wo_per_ho, 8).amax(dim=(0, 1, 2, 5))
    block_bad = bad.reshape(diff.shape[0], diff.shape[1], diff.shape[2], args.ho, wo_per_ho, 8).sum(dim=(0, 1, 2, 5))
    block_top = min(args.debug_topk, block_error.numel())
    vals, flats = torch.topk(block_error.reshape(-1), block_top)
    print(f"debug CTA x: worst {block_top} (h, wo_strip) tiles")
    for rank, (val, flat) in enumerate(zip(vals.tolist(), flats.tolist()), start=1):
        h = int(flat) // wo_per_ho
        wo_strip = int(flat) - h * wo_per_ho
        print(
            f"  #{rank}: block_x={h * wo_per_ho + wo_strip} h={h} "
            f"wo=[{wo_strip * 8},{wo_strip * 8 + 7}] "
            f"count={int(pack_count[h].item())} max_abs={val:.6e} "
            f"bad_elements={int(block_bad[h, wo_strip].item())}"
        )

    if args.debug_repeats > 0:
        print(f"debug repeats: comparing {args.debug_repeats} reruns to initial output")
        for rep in range(args.debug_repeats):
            out_rep = disco_kernels.forward_kpacked.default(inp, pack_idx, pack_val, pack_count, kernel_size, args.ho, args.wo)
            torch.cuda.synchronize(device)
            delta = (out_rep.float() - out.float()).abs().nan_to_num(float("inf"))
            print(f"  repeat {rep + 1}: max_delta={float(delta.max().item()):.6e} " f"changed_elements={int((delta > 0).sum().item())}/{delta.numel()}")

    if args.debug_isolate_ho:
        worst_flat = int(diff_key.reshape(-1).argmax().item())
        _, _, _, worst_h, _ = _unravel(worst_flat, diff.shape, device)
        iso_idx = pack_idx[worst_h : worst_h + 1].contiguous()
        iso_val = pack_val[worst_h : worst_h + 1].contiguous()
        iso_count = pack_count[worst_h : worst_h + 1].contiguous()
        iso_out = disco_kernels.forward_kpacked.default(inp, iso_idx, iso_val, iso_count, kernel_size, 1, args.wo)
        iso_ref = _reference(inp, iso_idx, iso_val, iso_count, kernel_size, 1, args.wo).to(out.dtype)
        torch.cuda.synchronize(device)
        iso_diff = (iso_out.float() - iso_ref.float()).abs()
        iso_max = float(iso_diff.nan_to_num(float("inf")).max().item())
        iso_ref_max = float(iso_ref.float().abs().max().item())
        iso_threshold = atol + rtol * iso_ref_max
        iso_status = "PASS" if iso_max <= iso_threshold else "CHECK"
        full_slice_diff = (out[:, :, :, worst_h : worst_h + 1, :].float() - iso_out.float()).abs().nan_to_num(float("inf"))
        print(
            f"debug isolate ho={worst_h}: max_abs={iso_max:.6e} "
            f"threshold={iso_threshold:.6e} [{iso_status}] "
            f"max |full_slice - isolated|={float(full_slice_diff.max().item()):.6e}"
        )


def _run_conv_mode(args, dtype, device, atol, rtol):
    out_shape = tuple(args.out_shape) if args.out_shape is not None else tuple(args.in_shape)
    kernel_shape = _conv_kernel_shape(args)
    out_channels = args.out_channels if (args.time_module or args.ncu_target == "module") else args.channels
    module_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    conv = DiscreteContinuousConvS2(
        in_channels=args.channels,
        out_channels=out_channels,
        in_shape=tuple(args.in_shape),
        out_shape=out_shape,
        kernel_shape=kernel_shape,
        basis_type=args.basis_type,
        basis_norm_mode=args.basis_norm_mode,
        groups=args.groups,
        grid_in=args.grid_in,
        grid_out=args.grid_out,
        bias=False,
        theta_cutoff=args.theta_cutoff,
        optimized_kernel=True,
        fused=False,
    ).to(device=device, dtype=module_dtype)

    if conv.psi_kpacked_K_pad is None:
        raise RuntimeError("conv did not produce kpacked buffers")

    torch.manual_seed(args.seed)
    inp = torch.randn(args.batch, args.channels, *tuple(args.in_shape), device=device, dtype=dtype).contiguous()
    vals = conv.psi_vals.to(_compute_dtype(dtype))

    def run_kpacked():
        return disco_kernels.forward_kpacked.default(
            inp,
            conv.psi_kpacked_idx,
            conv.psi_kpacked_vals,
            conv.psi_kpacked_count,
            conv.kernel_size,
            conv.nlat_out,
            conv.nlon_out,
        )

    def run_csr():
        return disco_kernels.forward.default(
            inp,
            conv.psi_roff_idx,
            conv.psi_ker_idx,
            conv.psi_row_idx,
            conv.psi_col_idx,
            vals,
            conv.kernel_size,
            conv.nlat_out,
            conv.nlon_out,
        ).to(dtype)

    def run_module():
        with torch.autocast(device.type, dtype=dtype, enabled=dtype in (torch.float16, torch.bfloat16)):
            return conv(inp.float())

    if args.ncu:
        targets = {
            "raw_kpacked": run_kpacked,
            "raw_csr": run_csr,
            "module": run_module,
        }
        target_fn = targets[args.ncu_target]
        count = conv.psi_kpacked_count
        print(
            "conv ncu config: "
            f"target={args.ncu_target} launches={args.ncu_launches} warmup={args.warmup} "
            f"batch={args.batch} channels={args.channels}->{out_channels} groups={args.groups} "
            f"in_shape={tuple(args.in_shape)} out_shape={out_shape} "
            f"kernel_shape={kernel_shape} basis={args.basis_type}/{args.basis_norm_mode} "
            f"K={conv.kernel_size} K_PAD={conv.psi_kpacked_K_pad} "
            f"count[min,max]=({int(count.min().item())},{int(count.max().item())}) "
            f"nnz={int(count.sum().item())} nlon_in/out={conv.nlon_in}/{conv.nlon_out}",
            flush=True,
        )
        for _ in range(args.warmup):
            target_fn()
        torch.cuda.synchronize(device)

        for launch_idx in range(args.ncu_launches):
            range_name = f"disco_{args.ncu_target}_launch_{launch_idx}"
            torch.cuda.nvtx.range_push(range_name)
            target_fn()
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize(device)
        print(f"conv ncu done: target={args.ncu_target} launches={args.ncu_launches}", flush=True)
        return {"ncu": True}

    out_kp = run_kpacked()
    ref = run_csr()
    torch.cuda.synchronize(device)

    py_kpack_ref = None
    if args.debug:
        py_kpack_ref = _reference(
            inp,
            conv.psi_kpacked_idx,
            conv.psi_kpacked_vals,
            conv.psi_kpacked_count,
            conv.kernel_size,
            conv.nlat_out,
            conv.nlon_out,
        ).to(dtype)

    diff = (out_kp.float() - ref.float()).abs()
    max_abs = float(diff.max().item())
    max_ref = float(ref.float().abs().max().item())
    max_rel = max_abs / max(max_ref, 1e-12)
    flat_idx = int(diff.reshape(-1).argmax().item())
    idx = torch.unravel_index(torch.tensor(flat_idx, device=device), diff.shape)
    idx_tuple = tuple(int(x.item()) for x in idx)

    if args.dump:
        b, c, _, h, w = idx_tuple
        got = out_kp[b, c, :, h, w].float().tolist()
        exp = ref[b, c, :, h, w].float().tolist()
        print(f"max diff index={idx_tuple}")
        print(f"conv sample out[{b},{c},:,{h},{w}] = {[round(x, 5) for x in got]}")
        print(f"conv sample ref[{b},{c},:,{h},{w}] = {[round(x, 5) for x in exp]}")

    def time_fn(fn):
        samples = []
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(args.iters):
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize(device)
            samples.append(start.elapsed_time(end))
        return samples

    times = time_fn(run_kpacked)
    csr_times = time_fn(run_csr) if args.time_csr else None
    module_times = time_fn(run_module) if args.time_module else None

    count = conv.psi_kpacked_count
    print(
        "conv config: "
        f"batch={args.batch} channels={args.channels}->{out_channels} groups={args.groups} "
        f"in_shape={tuple(args.in_shape)} out_shape={out_shape} "
        f"kernel_shape={kernel_shape} basis={args.basis_type}/{args.basis_norm_mode} "
        f"K={conv.kernel_size} K_PAD={conv.psi_kpacked_K_pad} "
        f"count[min,max]=({int(count.min().item())},{int(count.max().item())}) "
        f"nlon_in/out={conv.nlon_in}/{conv.nlon_out}"
    )
    print(f"conv timing: raw_kpacked median={statistics.median(times):.4f} ms")
    if csr_times is not None:
        print(f"conv timing: raw_csr     median={statistics.median(csr_times):.4f} ms")
    if module_times is not None:
        print(f"conv timing: module      median={statistics.median(module_times):.4f} ms")
    if args.debug:
        _print_compare("kpack py", py_kpack_ref, ref, atol, rtol)
        _print_compare("wgmma py", out_kp, py_kpack_ref, atol, rtol)
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "max_ref": max_ref,
        "median_ms": statistics.median(times),
        "csr_median_ms": statistics.median(csr_times) if csr_times is not None else None,
        "module_median_ms": statistics.median(module_times) if module_times is not None else None,
    }


def _run_csr_mode(args, dtype, device, atol, rtol):
    out_shape = tuple(args.out_shape) if args.out_shape is not None else tuple(args.in_shape)
    kernel_shape = _conv_kernel_shape(args)
    conv = DiscreteContinuousConvS2(
        in_channels=args.channels,
        out_channels=args.channels,
        in_shape=tuple(args.in_shape),
        out_shape=out_shape,
        kernel_shape=kernel_shape,
        basis_type=args.basis_type,
        basis_norm_mode=args.basis_norm_mode,
        groups=1,
        grid_in=args.grid_in,
        grid_out=args.grid_out,
        bias=False,
        theta_cutoff=args.theta_cutoff,
        optimized_kernel=True,
        fused=False,
    ).to(device=device, dtype=torch.float32)

    torch.manual_seed(args.seed)
    inp = torch.randn(args.batch, args.channels, *tuple(args.in_shape), device=device, dtype=torch.float32).to(dtype).contiguous()
    vals = conv.psi_vals.to(_compute_dtype(dtype))

    out_csr = disco_kernels.forward.default(
        inp,
        conv.psi_roff_idx,
        conv.psi_ker_idx,
        conv.psi_row_idx,
        conv.psi_col_idx,
        vals,
        conv.kernel_size,
        conv.nlat_out,
        conv.nlon_out,
    ).to(dtype)

    psi = _get_psi(
        conv.kernel_size,
        conv.psi_idx,
        conv.psi_vals,
        conv.nlat_in,
        conv.nlon_in,
        conv.nlat_out,
        conv.nlon_out,
    )
    ref = _disco_s2_contraction_torch(inp, psi, conv.nlon_out)
    torch.cuda.synchronize(device)

    print(
        "csr config: "
        f"batch={args.batch} channels={args.channels} "
        f"in_shape={tuple(args.in_shape)} out_shape={out_shape} "
        f"kernel_shape={kernel_shape} basis={args.basis_type}/{args.basis_norm_mode} "
        f"K={conv.kernel_size} kpacked_K_pad={conv.psi_kpacked_K_pad} "
        f"input_dtype={inp.dtype} vals_dtype={vals.dtype}"
    )
    _print_compare("csr fwd", out_csr, ref, atol, rtol)

    if args.dump:
        diff = (out_csr.float() - ref.float()).abs()
        flat_idx = int(diff.reshape(-1).argmax().item())
        idx = torch.unravel_index(torch.tensor(flat_idx, device=device), diff.shape)
        idx_tuple = tuple(int(x.item()) for x in idx)
        b, c, _, h, w = idx_tuple
        print(f"max csr diff index={idx_tuple}")
        print(f"csr out[{b},{c},:,{h},{w}] = {[round(x, 5) for x in out_csr[b, c, :, h, w].float().tolist()]}")
        print(f"ref out[{b},{c},:,{h},{w}] = {[round(x, 5) for x in ref[b, c, :, h, w].float().tolist()]}")

    times = []
    for _ in range(args.warmup):
        disco_kernels.forward.default(
            inp,
            conv.psi_roff_idx,
            conv.psi_ker_idx,
            conv.psi_row_idx,
            conv.psi_col_idx,
            vals,
            conv.kernel_size,
            conv.nlat_out,
            conv.nlon_out,
        )
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(args.iters):
        start.record()
        disco_kernels.forward.default(
            inp,
            conv.psi_roff_idx,
            conv.psi_ker_idx,
            conv.psi_row_idx,
            conv.psi_col_idx,
            vals,
            conv.kernel_size,
            conv.nlat_out,
            conv.nlon_out,
        )
        end.record()
        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end))
    print(f"csr median={statistics.median(times):.4f} ms")


def _max_stats(a, b):
    af = a.float()
    bf = b.float()
    diff = (af - bf).abs()
    max_abs = float(diff.max().item())
    max_ref = float(bf.abs().max().item())
    max_rel = max_abs / max(max_ref, 1e-12)
    return max_abs, max_rel, max_ref


def _print_compare(label, got, ref, atol, rtol):
    max_abs, max_rel, max_ref = _max_stats(got, ref)
    threshold = atol + rtol * max_ref
    status = "PASS" if max_abs <= threshold else "CHECK"
    print(f"{label:<12} got_dtype={got.dtype} ref_dtype={ref.dtype} " f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} threshold={threshold:.6e} [{status}]")


def _print_tensor_stats(label, tensor):
    finite = torch.isfinite(tensor)
    n_total = tensor.numel()
    n_finite = int(finite.sum().item())
    n_nan = int(torch.isnan(tensor).sum().item())
    n_inf = int(torch.isinf(tensor).sum().item())
    if n_finite:
        vals = tensor.detach().float()[finite]
        vmin = float(vals.min().item())
        vmax = float(vals.max().item())
        vmean = float(vals.abs().mean().item())
        print(f"{label:<18} dtype={tensor.dtype} finite={n_finite}/{n_total} " f"nan={n_nan} inf={n_inf} min={vmin:.6e} max={vmax:.6e} mean_abs={vmean:.6e}")
    else:
        print(f"{label:<18} dtype={tensor.dtype} finite=0/{n_total} nan={n_nan} inf={n_inf}")


def _weight_contract(x_expanded, weight, groups, groupsize):
    bsz, _, kernel_size, height, width = x_expanded.shape
    weight_r = weight.reshape(groups, -1, weight.shape[1], weight.shape[2])
    x_r = x_expanded.reshape(bsz, groups, groupsize, kernel_size, height, width)
    out = torch.einsum("bgckxy,gock->bgoxy", x_r, weight_r).contiguous()
    return out.reshape(bsz, -1, height, width)


def _run_einsum_mode(args, dtype, device, atol, rtol):
    out_shape = tuple(args.out_shape) if args.out_shape is not None else tuple(args.in_shape)
    kernel_shape = _conv_kernel_shape(args)
    module_dtype = torch.float32

    torch.manual_seed(args.seed)
    conv_ref = DiscreteContinuousConvS2(
        in_channels=args.channels,
        out_channels=args.out_channels,
        in_shape=tuple(args.in_shape),
        out_shape=out_shape,
        kernel_shape=kernel_shape,
        basis_type=args.basis_type,
        basis_norm_mode=args.basis_norm_mode,
        groups=1,
        grid_in=args.grid_in,
        grid_out=args.grid_out,
        bias=False,
        theta_cutoff=args.theta_cutoff,
        optimized_kernel=False,
    ).to(device=device, dtype=module_dtype)
    conv_opt = DiscreteContinuousConvS2(
        in_channels=args.channels,
        out_channels=args.out_channels,
        in_shape=tuple(args.in_shape),
        out_shape=out_shape,
        kernel_shape=kernel_shape,
        basis_type=args.basis_type,
        basis_norm_mode=args.basis_norm_mode,
        groups=1,
        grid_in=args.grid_in,
        grid_out=args.grid_out,
        bias=False,
        theta_cutoff=args.theta_cutoff,
        optimized_kernel=True,
    ).to(device=device, dtype=module_dtype)
    with torch.no_grad():
        conv_ref.weight.copy_(conv_opt.weight)

    inp = torch.randn(args.batch, args.channels, *tuple(args.in_shape), dtype=module_dtype, device=device)
    inp_narrow = inp.to(dtype).contiguous()

    vals = conv_opt.psi_vals.to(_compute_dtype(dtype))
    if conv_opt.psi_kpacked_K_pad in (8, 16):
        x_opt = disco_kernels.forward_kpacked.default(
            inp_narrow,
            conv_opt.psi_kpacked_idx,
            conv_opt.psi_kpacked_vals,
            conv_opt.psi_kpacked_count,
            conv_opt.kernel_size,
            conv_opt.nlat_out,
            conv_opt.nlon_out,
        )
        x_path = "kpacked"
    else:
        x_opt = disco_kernels.forward.default(
            inp_narrow,
            conv_opt.psi_roff_idx,
            conv_opt.psi_ker_idx,
            conv_opt.psi_row_idx,
            conv_opt.psi_col_idx,
            vals,
            conv_opt.kernel_size,
            conv_opt.nlat_out,
            conv_opt.nlon_out,
        ).to(dtype)
        x_path = "csr"

    psi = _get_psi(
        conv_ref.kernel_size,
        conv_ref.psi_idx,
        conv_ref.psi_vals,
        conv_ref.nlat_in,
        conv_ref.nlon_in,
        conv_ref.nlat_out,
        conv_ref.nlon_out,
    )
    x_ref = _disco_s2_contraction_torch(inp, psi, conv_ref.nlon_out)
    torch.cuda.synchronize(device)

    print(
        "einsum config: "
        f"batch={args.batch} channels={args.channels}->{args.out_channels} "
        f"in_shape={tuple(args.in_shape)} out_shape={out_shape} "
        f"kernel_shape={kernel_shape} basis={args.basis_type}/{args.basis_norm_mode} "
        f"K={conv_opt.kernel_size} kpacked_K_pad={conv_opt.psi_kpacked_K_pad} x_path={x_path}"
    )
    _print_tensor_stats("x_opt csr", x_opt)
    _print_tensor_stats("x_ref torch", x_ref)
    _print_compare("expanded", x_opt, x_ref, atol, rtol)

    with torch.autocast(device.type, dtype=dtype):
        out_opt_ac = _weight_contract(x_opt, conv_opt.weight, conv_opt.groups, conv_opt.groupsize)
        out_ref_ac = _weight_contract(x_ref, conv_ref.weight, conv_ref.groups, conv_ref.groupsize)
    with torch.autocast(device.type, enabled=False):
        out_opt_fp32 = _weight_contract(x_opt.float(), conv_opt.weight.float(), conv_opt.groups, conv_opt.groupsize)
        out_ref_fp32 = _weight_contract(x_ref.float(), conv_ref.weight.float(), conv_ref.groups, conv_ref.groupsize)
    torch.cuda.synchronize(device)

    _print_tensor_stats("out opt autocast", out_opt_ac)
    _print_tensor_stats("out ref autocast", out_ref_ac)
    _print_compare("einsum ac", out_opt_ac, out_ref_ac, atol, rtol)
    _print_tensor_stats("out opt fp32", out_opt_fp32)
    _print_tensor_stats("out ref fp32", out_ref_fp32)
    _print_compare("einsum fp32", out_opt_fp32, out_ref_fp32, atol, rtol)

    if args.dump:
        diff = (out_opt_ac.float() - out_ref_ac.float()).abs()
        flat_idx = int(diff.nan_to_num(float("inf")).reshape(-1).argmax().item())
        idx = torch.unravel_index(torch.tensor(flat_idx, device=device), diff.shape)
        idx_tuple = tuple(int(x.item()) for x in idx)
        b, c, h, w = idx_tuple
        print(f"max autocast output diff index={idx_tuple}")
        print(f"out opt ac[{b},:,{h},{w}] = {[round(x, 5) for x in out_opt_ac[b, :, h, w].float().tolist()]}")
        print(f"out ref ac[{b},:,{h},{w}] = {[round(x, 5) for x in out_ref_ac[b, :, h, w].float().tolist()]}")


def _run_module_mode(args, dtype, device, atol, rtol):
    out_shape = tuple(args.out_shape) if args.out_shape is not None else tuple(args.in_shape)
    kernel_shape = _conv_kernel_shape(args)

    is_amp = dtype in (torch.float16, torch.bfloat16)
    module_dtype = torch.float32 if is_amp else dtype
    fused_kwarg = {"fused": args.fused}

    torch.manual_seed(args.seed)
    conv_naive = DiscreteContinuousConvS2(
        in_channels=args.channels,
        out_channels=args.out_channels,
        in_shape=tuple(args.in_shape),
        out_shape=out_shape,
        kernel_shape=kernel_shape,
        basis_type=args.basis_type,
        basis_norm_mode=args.basis_norm_mode,
        groups=1,
        grid_in=args.grid_in,
        grid_out=args.grid_out,
        bias=False,
        theta_cutoff=args.theta_cutoff,
        optimized_kernel=False,
    ).to(device=device, dtype=module_dtype)

    conv_opt = DiscreteContinuousConvS2(
        in_channels=args.channels,
        out_channels=args.out_channels,
        in_shape=tuple(args.in_shape),
        out_shape=out_shape,
        kernel_shape=kernel_shape,
        basis_type=args.basis_type,
        basis_norm_mode=args.basis_norm_mode,
        groups=1,
        grid_in=args.grid_in,
        grid_out=args.grid_out,
        bias=False,
        theta_cutoff=args.theta_cutoff,
        optimized_kernel=True,
        **fused_kwarg,
    ).to(device=device, dtype=module_dtype)

    with torch.no_grad():
        conv_naive.weight.copy_(conv_opt.weight)

    inp = torch.randn(args.batch, args.channels, *tuple(args.in_shape), dtype=module_dtype, device=device)
    inp_naive = inp.detach().clone().requires_grad_(True)
    inp_opt = inp.detach().clone().requires_grad_(True)

    with torch.autocast(device.type, dtype=dtype, enabled=is_amp):
        out_naive = conv_naive(inp_naive)
        out_opt = conv_opt(inp_opt)

    grad = torch.randn_like(out_naive)
    out_naive.backward(grad)
    out_opt.backward(grad.clone())
    torch.cuda.synchronize(device)

    print(
        "module config: "
        f"batch={args.batch} channels={args.channels}->{args.out_channels} "
        f"in_shape={tuple(args.in_shape)} out_shape={out_shape} "
        f"kernel_shape={kernel_shape} basis={args.basis_type}/{args.basis_norm_mode} "
        f"fused={args.fused} optimized_kpacked_K_pad={conv_opt.psi_kpacked_K_pad}"
    )
    print("module dtypes: " f"module={module_dtype} autocast={dtype} " f"inp={inp.dtype} opt_weight={conv_opt.weight.dtype} naive_weight={conv_naive.weight.dtype}")
    _print_compare("output", out_opt, out_naive, atol, rtol)
    _print_compare("input grad", inp_opt.grad, inp_naive.grad, atol, rtol)
    _print_compare("weight grad", conv_opt.weight.grad, conv_naive.weight.grad, atol, rtol)

    if args.dump:
        diff = (out_opt.float() - out_naive.float()).abs()
        flat_idx = int(diff.reshape(-1).argmax().item())
        idx = torch.unravel_index(torch.tensor(flat_idx, device=device), diff.shape)
        idx_tuple = tuple(int(x.item()) for x in idx)
        b, c, h, w = idx_tuple
        print(f"max output diff index={idx_tuple}")
        print(f"out_opt[{b},:,{h},{w}]   = {[round(x, 5) for x in out_opt[b, :, h, w].float().tolist()]}")
        print(f"out_naive[{b},:,{h},{w}] = {[round(x, 5) for x in out_naive[b, :, h, w].float().tolist()]}")


def main():
    args = parse_args()
    _disable_tf32()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    _check_runtime(device)

    dtype = _DTYPES[args.dtype]
    atol = args.atol if args.atol is not None else (2e-2 if dtype == torch.float16 else 8e-2)
    rtol = args.rtol if args.rtol is not None else (2e-2 if dtype == torch.float16 else 8e-2)

    major, minor = torch.cuda.get_device_capability(device)
    print(f"device={torch.cuda.get_device_name(device)} SM_{major}{minor} dtype={args.dtype}")
    if args.mode == "csr":
        _run_csr_mode(args, dtype, device, atol, rtol)
        return

    if args.mode == "einsum":
        _run_einsum_mode(args, dtype, device, atol, rtol)
        return

    if args.mode == "module":
        _run_module_mode(args, dtype, device, atol, rtol)
        return

    if args.mode == "conv":
        result = _run_conv_mode(args, dtype, device, atol, rtol)
        if result.get("ncu", False):
            return
        threshold = atol + rtol * result["max_ref"]
        status = "PASS" if result["max_abs"] <= threshold else "CHECK"
        print(
            f"conv actual buffers: max_abs={result['max_abs']:.6e} " f"max_rel={result['max_rel']:.6e} threshold={threshold:.6e} " f"median={result['median_ms']:.4f} ms [{status}]"
        )
        return

    k_label = "N_PAD" if args.kernel_size is None else str(args.kernel_size)
    print(f"shape: B=1 C={args.bc} Hi={args.hi} Wi={args.wi} Ho={args.ho} Wo={args.wo} nbr={args.nbr} logical_K={k_label}")

    first8 = None
    results = {}
    if 8 in args.n_pad and 16 in args.n_pad:
        _, _, pack_val8, _ = _make_inputs(args, dtype, device, 8)
        first8 = pack_val8.float()

    for n_pad in args.n_pad:
        use_first8 = first8 if n_pad == 16 else None
        result = _run_once(args, dtype, device, n_pad, first8_vals=use_first8)
        results[n_pad] = result
        threshold = atol + rtol * result["max_ref"]
        status = "PASS" if result["max_abs"] <= threshold else "CHECK"
        print(
            f"N_PAD={n_pad:2d} K={result['kernel_size']:2d}: max_abs={result['max_abs']:.6e} "
            f"max_rel={result['max_rel']:.6e} threshold={threshold:.6e} "
            f"median={result['median_ms']:.4f} ms [{status}]"
        )
        if args.debug:
            _print_synthetic_debug(args, n_pad, result, atol, rtol, device)

    if 8 in results and 16 in results:
        k8 = results[8]["kernel_size"]
        k16 = results[16]["kernel_size"]
        k_cmp = min(k8, k16)
        paired = (results[8]["out"][:, :, :k_cmp].float() - results[16]["out"][:, :, :k_cmp].float()).abs().max().item()
        print(f"paired check: max |N_PAD=8[:{k_cmp}] - N_PAD=16[:{k_cmp}]| = {paired:.6e}")


if __name__ == "__main__":
    main()
