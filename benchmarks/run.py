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


def load_reference_csv(path: str) -> dict[str, dict]:
    """Load a previously saved CSV into a dict keyed by benchmark name."""
    ref = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["name"]
            ref[name] = {
                "fwd_ms": float(row["fwd_ms"]) if row.get("fwd_ms") not in (None, "", "None") else None,
                "bwd_ms": float(row["bwd_ms"]) if row.get("bwd_ms") not in (None, "", "None") else None,
                "ref_error": float(row["ref_error"]) if row.get("ref_error") not in (None, "", "None") else None,
            }
    return ref


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

_COL_NAME = 52
_COL_DEV = 10
_COL_DTYPE = 10
_COL_FWD = 9
_COL_BWD = 9
_COL_ERR = 12
_COL_SPD = 9  # speedup column width (shared for fwd and bwd)


def _fmt_ms(v: Optional[float]) -> str:
    return f"{v:8.2f}" if v is not None else "     n/a"


def _fmt_err(v: Optional[float]) -> str:
    return f"{v:.3e}" if v is not None else "         n/a"


def _fmt_speedup(speedup: Optional[float], tol: float) -> str:
    if speedup is None:
        return "     n/a"
    flag = " !" if _is_regression(speedup, tol) else "  "
    return f"{speedup:6.2f}x{flag}"


def _header(with_ref: bool) -> str:
    h = f"{'name':<{_COL_NAME}} " f"{'device':<{_COL_DEV}} " f"{'dtype':<{_COL_DTYPE}} " f"{'fwd_ms':>{_COL_FWD}} " f"{'bwd_ms':>{_COL_BWD}} " f"{'ref_l_inf':>{_COL_ERR}}"
    if with_ref:
        h += f"  {'fwd_spd':>{_COL_SPD}}  {'bwd_spd':>{_COL_SPD}}"
    return h


def _row(
    r: BenchmarkResult,
    ref: Optional[dict],
    tol: float,
) -> str:
    line = (
        f"{r.name:<{_COL_NAME}} "
        f"{r.device:<{_COL_DEV}} "
        f"{r.dtype:<{_COL_DTYPE}} "
        f"{_fmt_ms(r.fwd_ms):>{_COL_FWD}} "
        f"{_fmt_ms(r.bwd_ms):>{_COL_BWD}} "
        f"{_fmt_err(r.ref_error):>{_COL_ERR}}"
    )
    if ref is not None:
        fwd_spd = _speedup(ref.get("fwd_ms"), r.fwd_ms)
        bwd_spd = _speedup(ref.get("bwd_ms"), r.bwd_ms)
        line += f"  {_fmt_speedup(fwd_spd, tol):>{_COL_SPD}}  {_fmt_speedup(bwd_spd, tol):>{_COL_SPD}}"
    return line


def print_table(
    results: list[BenchmarkResult],
    skipped: list[tuple[str, str]],
    reference: Optional[dict[str, dict]] = None,
    tol: float = 0.05,
) -> int:
    """Print results table. Returns number of flagged regressions."""
    with_ref = reference is not None
    h = _header(with_ref)
    print(h)
    print("-" * len(h))

    n_regressions = 0
    for r in results:
        ref = reference.get(r.name) if reference else None
        print(_row(r, ref, tol))
        if ref is not None:
            fwd_spd = _speedup(ref.get("fwd_ms"), r.fwd_ms)
            bwd_spd = _speedup(ref.get("bwd_ms"), r.bwd_ms)
            if _is_regression(fwd_spd, tol) or _is_regression(bwd_spd, tol):
                n_regressions += 1

    if skipped:
        print()
        for name, reason in skipped:
            print(f"  [skip] {name}  ({reason})")

    if with_ref and n_regressions:
        print(f"\n  ! {n_regressions} regression(s) detected (tolerance {tol*100:.0f}%)")

    return n_regressions


# ------------------------------------------------------------------------------
# Output serialisation
# ------------------------------------------------------------------------------


def _result_to_dict(r: BenchmarkResult) -> dict:
    return {
        "name": r.name,
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
        print(f"reference : {args.reference_csv}  ({len(reference)} entries)")
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
        result = run_entry(entry, warmup=args.warmup, iters=args.iters)
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
