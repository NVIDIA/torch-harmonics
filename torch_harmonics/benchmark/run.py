import argparse
import dataclasses
import json
import logging
import pathlib
import subprocess
import sys

import torch

from torch_harmonics.benchmark.benchmark import get_benchmarks
from torch_harmonics.benchmark.hardware import get_batch_size_factor, set_device

_GIT_COMMIT: str | None = None


def get_git_commit() -> str:
    global _GIT_COMMIT
    if _GIT_COMMIT is None:
        try:
            commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
            dirty = (
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
            if dirty:
                commit = f"{commit}-dirty"
        except (subprocess.CalledProcessError, FileNotFoundError):
            commit = "unknown"
        _GIT_COMMIT = commit
    return _GIT_COMMIT


def get_device_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).name
    else:
        return "CPU"


def main(
    benchmark_name: str | None,
    iters: int,
    output_dir: pathlib.Path,
    device: str,
    batch_size_factors: list[float],
) -> int:
    set_device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_name = get_device_name()
    safe_device_name = device_name.replace(" ", "_").replace("/", "_").lower()
    hardware_factor = get_batch_size_factor()

    logging.info(f"Running benchmarks on device: {device_name}")
    logging.info(f"Hardware batch size factor: {hardware_factor}")
    benchmarks = get_benchmarks()
    if benchmark_name is not None:
        if benchmark_name not in benchmarks:
            logging.error(
                f"Specified benchmark {benchmark_name} not found. "
                f"Available benchmarks: {', '.join(benchmarks.keys())}"
            )
            return 1
        benchmarks_to_run = {benchmark_name: benchmarks[benchmark_name]}
    else:
        benchmarks_to_run = benchmarks

    def get_filename(name, factor, extension) -> pathlib.Path:
        safe_name = name.replace("/", "_").replace(".", "_").lower()
        factor_str = f"{factor:g}x"
        return (
            output_dir
            / f"{safe_name}_{factor_str}_{safe_device_name}_{get_git_commit()}.{extension}"
        )

    for name, cls in benchmarks_to_run.items():
        for factor in batch_size_factors:
            combined_factor = hardware_factor * factor
            logging.info(f"Running benchmark: {name} (batch size factor: {factor:g}x)")
            result = cls.run_benchmark(iters=iters, batch_size_factor=combined_factor)
            result_data = json.dumps(dataclasses.asdict(result), indent=2)
            logging.info(f"Result:\n{result_data}")
            json_path = get_filename(name, factor, "json")
            with open(json_path, "w") as f:
                logging.info(f"Saving result json to {f.name}")
                f.write(result_data)

    return 0


def cli() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser(description="Run registered benchmarks.")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "Name of the benchmark to run. If not provided, "
            "all benchmarks will be run."
        ),
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="Number of iterations to run each benchmark for.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark results in.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run benchmarks on, e.g. 'cpu', 'cuda', 'cuda:1'. "
             "Defaults to 'cuda' if available, otherwise 'cpu'.",
    )
    parser.add_argument(
        "--batch-size-factors",
        type=float,
        nargs="+",
        default=[1, 2],
        help="Batch size scale factors to run each benchmark with. "
             "Each benchmark will be run once per factor. "
             "These are multiplied with the hardware-specific factor. "
             "Defaults to [1, 2].",
    )
    args = parser.parse_args()
    return main(
        benchmark_name=args.name,
        iters=args.iters,
        output_dir=pathlib.Path(args.output_dir),
        device=args.device,
        batch_size_factors=args.batch_size_factors,
    )


if __name__ == "__main__":
    sys.exit(cli())
