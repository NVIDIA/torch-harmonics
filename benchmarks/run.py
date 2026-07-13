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

import os
import sys

# Add this directory to sys.path so the sibling modules (bench, sht, disco,
# attention) are importable regardless of where the script is invoked from.
# The repo root is deliberately NOT added, so the installed torch_harmonics
# package takes priority over any local source tree.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import csv
import json
from typing import Optional

import attention  # noqa: F401
import disco  # noqa: F401

# trigger registration of all built-in entries
import sht  # noqa: F401
import torch
from bench import BenchmarkResult, get_entries, run_entry

# ------------------------------------------------------------------------------
# Reference CSV
# ------------------------------------------------------------------------------


def load_reference_csv(path: str) -> list[dict]:
    """Load a previously saved CSV as a list of row dicts (multiple rows per name allowed)."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "name": row["name"],
                    "architecture": row.get("architecture", ""),
                    "fwd_ms": float(row["fwd_ms"]) if row.get("fwd_ms") not in (None, "", "None") else None,
                    "bwd_ms": float(row["bwd_ms"]) if row.get("bwd_ms") not in (None, "", "None") else None,
                    "ref_error": float(row["ref_error"]) if row.get("ref_error") not in (None, "", "None") else None,
                }
            )
    return rows


def _lookup_reference(reference: Optional[list[dict]], result: "BenchmarkResult") -> Optional[dict]:
    """Find the reference row matching result's name and architecture."""
    if reference is None:
        return None
    # prefer an arch-matching row; fall back to a row with no arch recorded
    fallback = None
    for row in reference:
        if row["name"] != result.name:
            continue
        if _arch_match(row, result):
            return row
        if not row.get("architecture"):
            fallback = row
    return fallback


def _arch_match(ref: dict, result: "BenchmarkResult") -> bool:
    """Return False if the reference entry was produced on a different GPU."""
    arch = ref.get("architecture", "")
    return not arch or _fmt_arch(arch) == _fmt_arch(result.arch)


def _speedup(ref_ms: Optional[float], cur_ms: Optional[float]) -> Optional[float]:
    """ref / cur — >1 means current is faster, <1 means regression."""
    if ref_ms is None or cur_ms is None or cur_ms == 0.0:
        return None
    return ref_ms / cur_ms


def _is_regression(speedup: Optional[float], tol: float) -> bool:
    return speedup is not None and speedup < (1.0 - tol)


# ------------------------------------------------------------------------------
# Table printing
# ------------------------------------------------------------------------------

# Vendor prefixes to strip from GPU names for brevity
_ARCH_STRIP = ("NVIDIA ", "AMD ", "Intel ")


def _fmt_arch(arch: str) -> str:
    for prefix in _ARCH_STRIP:
        if arch.startswith(prefix):
            return arch[len(prefix) :]
    return arch


def _fmt_dtype(dtype: str) -> str:
    return dtype.replace("torch.", "")


def _fmt_ms(v: Optional[float]) -> str:
    return f"{v:.2f}" if v is not None else "n/a"


def _fmt_err(v: Optional[float]) -> str:
    return f"{v:.3e}" if v is not None else "n/a"


def _fmt_speedup(speedup: Optional[float], tol: float) -> str:
    if speedup is None:
        return "n/a"
    flag = " !" if _is_regression(speedup, tol) else ""
    return f"{speedup:.2f}x{flag}"


def _render_rows(
    results: list[BenchmarkResult],
    reference: Optional[dict[str, dict]],
    tol: float,
    with_err: bool,
) -> tuple[list[str], list[list[str]]]:
    """Return (headers, data_rows) as lists of pre-formatted cell strings."""
    headers = ["name", "architecture", "device", "dtype", "fwd_ms", "bwd_ms"]
    if with_err:
        headers.append("ref_l_inf")
    if reference is not None:
        headers += ["fwd_spd", "bwd_spd"]

    rows = []
    for r in results:
        ref = _lookup_reference(reference, r)
        cells = [
            r.name,
            _fmt_arch(r.arch),
            r.device,
            _fmt_dtype(r.dtype),
            _fmt_ms(r.fwd_ms),
            _fmt_ms(r.bwd_ms),
        ]
        if with_err:
            cells.append(_fmt_err(r.ref_error))
        if ref is not None:
            cells.append(_fmt_speedup(_speedup(ref.get("fwd_ms"), r.fwd_ms), tol))
            cells.append(_fmt_speedup(_speedup(ref.get("bwd_ms"), r.bwd_ms), tol))
        elif reference is not None:
            cells += ["n/a", "n/a"]
        rows.append(cells)

    return headers, rows


def _group_key(name: str) -> str:
    """Derive a display group from benchmark name prefix."""
    if name.startswith("sht") or name.startswith("isht"):
        return "sht"
    if name.startswith("disco"):
        return "disco"
    return "attention"


def _format_table(headers: list[str], rows: list[list[str]], separator_after: set) -> list[str]:
    """Format headers + rows into fixed-width lines with dynamic column widths.

    separator_after: set of row indices (0-based) after which to insert a blank separator line.
    """
    # left-aligned columns by index (name, arch, device, dtype); rest right-aligned
    LEFT = {0, 1, 2, 3}
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells):
        parts = []
        for i, (cell, w) in enumerate(zip(cells, widths)):
            parts.append(f"{cell:<{w}}" if i in LEFT else f"{cell:>{w}}")
        return "  ".join(parts)

    total_width = len(fmt_row(headers))
    header_line = fmt_row(headers)
    lines = [header_line, "-" * total_width]
    for i, row in enumerate(rows):
        lines.append(fmt_row(row))
        if i in separator_after:
            lines.append("")
    return lines


def print_table(
    results: list[BenchmarkResult],
    skipped: list[tuple[str, str]],
    reference: Optional[dict[str, dict]] = None,
    tol: float = 0.05,
) -> int:
    """Print results table. Returns number of flagged regressions."""
    with_err = any(r.ref_error is not None for r in results)
    headers, rows = _render_rows(results, reference, tol, with_err)

    # insert blank lines between benchmark groups (sht / disco / attention)
    separator_after: set = set()
    prev_group = _group_key(results[0].name) if results else ""
    for i, r in enumerate(results[1:], start=0):
        g = _group_key(results[i + 1].name)
        if g != prev_group:
            separator_after.add(i)
        prev_group = g

    for line in _format_table(headers, rows, separator_after):
        print(line)

    n_regressions = 0
    if reference is not None:
        for r in results:
            ref = _lookup_reference(reference, r)
            if ref is not None:
                fwd_spd = _speedup(ref.get("fwd_ms"), r.fwd_ms)
                bwd_spd = _speedup(ref.get("bwd_ms"), r.bwd_ms)
                if _is_regression(fwd_spd, tol) or _is_regression(bwd_spd, tol):
                    n_regressions += 1

    if skipped:
        print()
        for name, reason in skipped:
            print(f"  [skip] {name}  ({reason})")

    if reference is not None and n_regressions:
        print(f"\n  ! {n_regressions} regression(s) detected (tolerance {tol*100:.0f}%)")

    return n_regressions


# ------------------------------------------------------------------------------
# Output serialisation
# ------------------------------------------------------------------------------


def _result_to_dict(r: BenchmarkResult) -> dict:
    return {
        "name": r.name,
        "architecture": r.arch,
        "device": r.device,
        "dtype": r.dtype,
        "fwd_ms": r.fwd_ms,
        "bwd_ms": r.bwd_ms,
        "ref_error": r.ref_error,
    }


def save_json(results: list[BenchmarkResult], path: str) -> None:
    with open(path, "w") as f:
        json.dump([_result_to_dict(r) for r in results], f, indent=2)
    print(f"results saved to {path}")


def save_csv(results: list[BenchmarkResult], path: str) -> None:
    if not results:
        return
    fields = list(_result_to_dict(results[0]).keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(_result_to_dict(r) for r in results)
    print(f"results saved to {path}")


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="torch-harmonics benchmark suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", default=None, help="substring filter on benchmark name")
    parser.add_argument("--tags", nargs="*", default=None, help="keep only entries whose tags list contains any of these")
    parser.add_argument("--device", default=None, help="exact device spec filter, e.g. 'cuda' or 'cpu'")
    parser.add_argument("--warmup", type=int, default=3, help="number of warmup iterations (not timed)")
    parser.add_argument("--iters", type=int, default=20, help="number of timed iterations per entry")
    parser.add_argument("--save-json", default=None, metavar="PATH", help="write results to a JSON file")
    parser.add_argument("--save-csv", default=None, metavar="PATH", help="write results to a CSV file")
    parser.add_argument("--reference-csv", default=None, metavar="PATH", help="CSV produced by a previous --save-csv run; " "adds speedup columns and flags regressions")
    parser.add_argument("--regression-tol", type=float, default=0.05, metavar="F", help="slowdowns within this fraction of 1x are not flagged " "(default 0.05 = 5%%)")
    parser.add_argument("--check-outputs", action="store_true", help="run float64/CPU reference and report ref_l_inf error")
    args = parser.parse_args()

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0).name
        print(f"GPU : {gpu}")
    else:
        print("GPU : none (CPU only)")
    print(f"PyTorch : {torch.__version__}")
    print()

    reference = None
    if args.reference_csv:
        reference = load_reference_csv(args.reference_csv)
        ref_archs = sorted({r["architecture"] for r in reference if r.get("architecture")})
        arch_str = ", ".join(_fmt_arch(a) for a in ref_archs) if ref_archs else "unknown"
        print(f"reference : {args.reference_csv}  ({len(reference)} rows, arch: {arch_str})")
        print(f"tolerance : {args.regression_tol*100:.0f}%")
        print()

    entries = get_entries(
        name_filter=args.name,
        tags=args.tags,
        device_filter=args.device,
    )
    if not entries:
        print("no matching benchmark entries found")
        return 1

    results, skipped = [], []
    for entry in entries:
        print(f"  {entry.name} ...", end=" ", flush=True)
        result = run_entry(entry, warmup=args.warmup, iters=args.iters, skip_reference=not args.check_outputs)
        if result is None:
            reason = f"device '{entry.device}' not available"
            print(f"skipped ({reason})")
            skipped.append((entry.name, reason))
        else:
            parts = [f"fwd={result.fwd_ms:.2f}ms"]
            if result.bwd_ms is not None:
                parts.append(f"bwd={result.bwd_ms:.2f}ms")
            if result.ref_error is not None:
                parts.append(f"err={result.ref_error:.2e}")
            print("  ".join(parts))
            results.append(result)

    print()
    n_regressions = print_table(results, skipped, reference=reference, tol=args.regression_tol)

    if args.save_json:
        save_json(results, args.save_json)
    if args.save_csv:
        save_csv(results, args.save_csv)

    # non-zero exit when regressions are detected — useful in CI
    return 1 if n_regressions else 0


if __name__ == "__main__":
    sys.exit(main())
