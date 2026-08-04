# Benchmarking

torch-harmonics ships a benchmark suite in the `benchmarks/` directory that
measures forward and backward pass latency for SHT, DISCO convolutions, and
spherical attention layers across devices and dtypes.

## Running benchmarks

From the repository root:

```bash
python benchmarks/run.py
```

This runs every registered benchmark entry with default settings (3 warm-up
iterations, 20 timed iterations) and prints a summary table.

### Useful options

| Flag                       | Description                                                                  |
| -------------------------- | ---------------------------------------------------------------------------- |
| `--name <substring>`       | Run only benchmarks whose name contains the substring (e.g. `--name disco`). |
| `--tags <tag> [<tag> ...]` | Filter by tags (e.g. `--tags sht`, `--tags attention`).                      |
| `--device <spec>`          | Restrict to a device (`cuda`, `cpu`).                                        |
| `--warmup <N>`             | Number of untimed warm-up iterations (default 3).                            |
| `--iters <N>`              | Number of timed iterations (default 20).                                     |
| `--check-outputs`          | Also run a float64/CPU reference and report the $L_\infty$ error.            |

### Example: run only DISCO benchmarks on GPU

```bash
python benchmarks/run.py --name disco --device cuda
```

## Saving results

Results can be saved to CSV or JSON for later comparison:

```bash
python benchmarks/run.py --save-csv results.csv
python benchmarks/run.py --save-json results.json
```

The CSV contains one row per benchmark entry with columns: `name`,
`architecture`, `device`, `dtype`, `fwd_ms`, `bwd_ms`, and (if
`--check-outputs` was used) `ref_error`.

## Comparing against reference numbers

To detect performance regressions, compare a new run against a previously
saved CSV:

```bash
python benchmarks/run.py --reference-csv benchmarks/reference_results.csv
```

This adds `fwd_spd` and `bwd_spd` columns to the output table showing the
speedup relative to the reference ($>1$ means the current run is faster).
Regressions beyond the tolerance threshold (default 5%) are flagged with `!`.

By default, reference rows are matched by the GPU architecture of the current
machine. To compare against a different architecture (e.g. run on GB200 but
compare against H100 numbers), use `--reference-arch`:

```bash
python benchmarks/run.py --reference-csv benchmarks/reference_results.csv \
    --reference-arch "H100 80GB HBM3"
```

The tolerance can be adjusted:

```bash
python benchmarks/run.py --reference-csv benchmarks/reference_results.csv --regression-tol 0.10
```

The script exits with a non-zero status when regressions are detected, making
it suitable for CI gates.

## Reference results

The file `benchmarks/reference_results.csv` contains reference timings on
NVIDIA H100 and GB200 GPUs. To update it after a verified improvement:

```bash
python benchmarks/run.py --save-csv benchmarks/reference_results.csv
```
