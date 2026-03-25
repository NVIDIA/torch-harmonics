import argparse
import dataclasses
import json
import logging
import pathlib
import subprocess
import sys

import torch

from torch_harmonics.benchmark.benchmark import get_benchmarks

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
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    device_name = get_device_name()
    safe_device_name = device_name.replace(" ", "_").replace("/", "_").lower()

    logging.info(f"Running benchmarks on device: {device_name}")
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

    def get_filename(name, extension) -> pathlib.Path:
        safe_name = name.replace("/", "_").replace(".", "_").lower()
        return (
            output_dir
            / f"{safe_name}_{safe_device_name}_{get_git_commit()}.{extension}"
        )

    for name, cls in benchmarks_to_run.items():
        logging.info(f"Running benchmark: {name}")
        result = cls.run_benchmark(iters=iters)
        result_data = json.dumps(dataclasses.asdict(result), indent=2)
        logging.info(f"Result: {result_data}")
        json_path = get_filename(name, "json")
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
    args = parser.parse_args()
    return main(
        benchmark_name=args.name,
        iters=args.iters,
        output_dir=pathlib.Path(args.output_dir),
    )


if __name__ == "__main__":
    sys.exit(cli())
