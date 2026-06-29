# Contributing to torch-harmonics

Thank you for your interest in contributing. torch-harmonics implements differentiable
signal processing on the sphere (SHT, DISCO convolutions, spherical attention, and
distributed variants). We are grateful for contributions that improve correctness, performance,
documentation, or test coverage.

## Table of contents

- [Getting in touch](#getting-in-touch)
- [Development setup](#development-setup)
- [Building C++/CUDA extensions](#building-ccuda-extensions)
- [Running tests](#running-tests)
- [Running benchmarks](#running-benchmarks)
- [Code style and pre-commit](#code-style-and-pre-commit)
- [Project structure](#project-structure)
- [Guidelines by area](#guidelines-by-area)
- [Pull requests](#pull-requests)
- [Release and packaging](#release-and-packaging)

## Getting in touch

We're happy to discuss ideas before you spend time on a large change.

- Open a [GitHub issue](https://github.com/NVIDIA/torch-harmonics/issues) for bugs,
  feature proposals, or design questions. A similar effort may already be in progress.
- For larger changes (new APIs, breaking behavior, distributed design), an issue first
  helps align on approach. Small fixes and test improvements can often go straight to a PR.
- Please be respectful and constructive in issues and reviews.

## Development setup

**Requirements**

- Python 3.9+
- PyTorch 2.6+ (install before building; extensions compile against your local `torch`)
- NumPy 1.22.4+
- A C++17 compiler; CUDA toolkit optional but recommended for GPU kernel work

**Editable install**

```bash
git clone https://github.com/NVIDIA/torch-harmonics.git
cd torch-harmonics

# Install PyTorch first (CPU example; use the CUDA wheel that matches your system)
python3 -m pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

# Editable install with dev dependencies
python3 -m pip install -e ".[dev]" --no-build-isolation
```

Optional extras:

```bash
pip install -e ".[dev,filter_basis]"   # scipy for filter-basis tests
```

Install [pre-commit](https://pre-commit.com/) hooks:

```bash
pre-commit install
```

## Building C++/CUDA extensions

torch-harmonics ships several compiled extensions:

| Module | Purpose |
|--------|---------|
| `torch_harmonics.attention._C` | Optimized neighborhood attention (CPU/CUDA) |
| `torch_harmonics.disco._C` | Optimized DISCO convolution (CPU/CUDA) |
| `attention_helpers`, `disco_helpers` | Runtime availability checks |

**You must rebuild after changing any `.cpp`, `.cu`, or `.h` file under
`torch_harmonics/attention/optimized/` or `torch_harmonics/disco/optimized/`.** Python-only
edits do not require a rebuild, but stale `.so` files are a common source of confusing test
failures (e.g. old shape checks still firing from an outdated binary).

```bash
pip install -e . --no-build-isolation
```

Force a clean extension rebuild if needed:

```bash
rm -f torch_harmonics/attention/_C*.so torch_harmonics/disco/_C*.so
pip install -e . --no-build-isolation
```

Verify optimized kernels are available:

```bash
python3 -c "
from torch_harmonics.attention import optimized_kernels_is_available
from torch_harmonics.disco import optimized_kernels_is_available as disco_ok
print('attention:', optimized_kernels_is_available())
print('disco:', disco_ok())
"
```

### CUDA builds

If CUDA is not detected automatically (containers, headless nodes):

```bash
export FORCE_CUDA_EXTENSION=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0"   # set to GPUs you target; reduces compile time
pip install -e . --no-build-isolation
```

Custom CUDA extensions require compute capability **≥ 8.0**.

### Build environment variables

| Variable | Effect |
|----------|--------|
| `FORCE_CUDA_EXTENSION=1` | Build CUDA kernels even if CUDA is not detected at configure time |
| `TORCH_CUDA_ARCH_LIST` | Limit NVCC target architectures |
| `TORCH_HARMONICS_DEBUG=1` | Debug flags (`-O0`, `-g`) for extensions |
| `TORCH_HARMONICS_PROFILE=1` | NVCC lineinfo / PTXAS verbose (CUDA) |
| `TORCH_HARMONICS_ENABLE_OPENMP=1` | Enable OpenMP in CPU kernels |
| `TORCH_HARMONICS_NATIVE_CPU_ARCH=1` | `-march=native` (local dev only; not for wheels) |

### Installing without a local build

Most users install prebuilt wheels:

- **NVIDIA PyPI** (CUDA): `torch-harmonics-cu126`, `cu128`, etc., or
  `torch-harmonics-cuda-latest` / `torch-harmonics-cpu-latest`. See
  [README.md](README.md#installation).
- **PyPI** (`torch-harmonics`): CPU-only wheel for the latest supported PyTorch release.

Install PyTorch first, then the wheel that matches your CUDA toolkit (`nvidia-smi` for the
driver CUDA version).

### Building wheels locally

```bash
python3 -m pip install build
python3 -m build --wheel --no-isolation
```

Wheels follow the PyTorch ecosystem pattern
`torch_harmonics-{version}+{cuda}-{python}-{abi}-{platform}.whl`, e.g.
`torch_harmonics-0.9.1+cu126-cp310-cp310-linux_x86_64.whl`.

Sanity-check an install:

```bash
python3 -c "import torch_harmonics; print('Import successful')"
```

### Docker

```bash
docker build . -t torch_harmonics
docker run --gpus all -it --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 torch_harmonics
```

### Build troubleshooting

| Problem | What to try |
|---------|-------------|
| CUDA version mismatch | Match the CUDA toolkit to your installed PyTorch CUDA build |
| Extensions not found / stale behavior | Rebuild with `pip install -e . --no-build-isolation`; remove old `_C*.so` if needed |
| No matching wheel on install | Install PyTorch first, then torch-harmonics; or build from source |
| ABI mismatch | Rebuild from source against the same PyTorch version |
| CUDA not detected at build time | `export FORCE_CUDA_EXTENSION=1` and set `TORCH_CUDA_ARCH_LIST` |

## Running tests

CI runs pytest on CPU with coverage over `torch_harmonics`, excluding distributed tests:

```bash
python3 -m pytest -ra \
  --cov-report term \
  --cov-config=.coveragerc \
  --cov=torch_harmonics \
  --ignore-glob="**/test_distributed_*" \
  ./tests/
```

Run a single test file or case:

```bash
python3 -m pytest tests/test_attention.py -x
python3 -m pytest tests/test_attention.py::TestNeighborhoodAttentionS2_0::test_custom_implementation_20 -x
```

Many tests for DISCO and attention are gated on `optimized_kernels_is_available()`. If
extensions failed to build, those tests are skipped rather than failed—confirm your rebuild
succeeded.

Distributed tests live under `tests/test_distributed_*` and require a multi-process launch;
they are not part of the default CI job above.

### What to run for your change

| Change type | Suggested tests |
|-------------|-----------------|
| SHT / quadrature / resampling | `tests/test_sht.py`, related modules |
| DISCO convolution | `tests/test_convolution.py`, `tests/test_filter_basis.py` |
| Attention | `tests/test_attention.py` |
| Distributed | `tests/test_distributed_*.py` (manual multi-GPU setup) |
| C++/CUDA kernels | Relevant test file **and** compare against torch reference paths where they exist; add or extend `opcheck` PT2 tests |

CI workflows: **style** (pre-commit on PRs) and **tests** (pytest on push to `main`). Both
should pass before merge.

### Running distributed tests
Distributed tests need process groups and do not run in the CI due to the need for MPI. They compare distributed implementations
to the sequential implementation, assuming its correctness. To run the distributed tests, use the helper script:
```bash
bash tests/run_tests.sh -d --grid_size_lat 2 --grid_size_lon 2   # 2x2 = 4 ranks
```
Tests pick up MASTER_ADDR, MASTER_PORT, WORLD_RANK, WORLD_SIZE from the environment (see tests/testutils.py). If you make any
modifications to distributed routines, we kindly ask you to run these tests for various combinations of `grid_size_lat` and
`grid_size_lon`.

## Running benchmarks

The `benchmarks/` directory contains a self-contained benchmark suite covering SHT, DISCO
convolution, and spherical attention across multiple resolutions, channel counts, and dtypes.

Run the full suite on the current GPU:

```bash
python benchmarks/run.py
```

Filter by name substring or tag:

```bash
python benchmarks/run.py --name disco
python benchmarks/run.py --tags attention neighborhood
```

Save a baseline CSV, then compare after your change:

```bash
python benchmarks/run.py --save-csv baseline.csv
# ... make your changes and rebuild ...
python benchmarks/run.py --reference-csv baseline.csv
```

The comparison table shows per-entry speedup (`fwd_spd`, `bwd_spd`) and flags regressions
with `!` if throughput drops by more than 5% (adjustable via `--regression-tol`).

Run the float64/CPU reference error check (slower, opt-in):

```bash
python benchmarks/run.py --check-outputs --name disco_s2_opt_1deg
```

## Code style and pre-commit

We use [pre-commit](https://pre-commit.com/) on pull requests (`.github/workflows/style.yml`).
Running it locally before you push avoids CI surprises:

```bash
pre-commit run --all-files
```

Hooks include:

- **black** (line length 180, see `pyproject.toml`)
- **ruff** (lint + import sorting; notebooks excluded)
- **clang-format** for C, C++, and CUDA
- **SPDX license header** check on Python files (`scripts/check_license_header.py`)

### License headers

New Python files must include the BSD-3-Clause SPDX header block used elsewhere in the repo
(`SPDX-FileCopyrightText` and `SPDX-License-Identifier: BSD-3-Clause`). Copy the header
from an existing file in the same directory.

C/C++/CUDA sources use the same SPDX comment style at the top of the file.

### Python conventions

- Match existing naming, module layout, and documentation level in the area you edit.
- Prefer extending existing helpers over duplicating logic.
- Use `parameterized` for multi-configuration unit tests (see `tests/test_attention.py`).
- Shared test utilities live in `tests/testutils.py`.

### Naming conventions

- Modules: lower_snake_case; prefix with _ for internal-only modules (e.g. _layers.py, _disco_utils.py).
- Public classes: PascalCase. Sphere-valued classes carry the suffix S2 (e.g. DiscreteContinuousConvS2, NeighborhoodAttentionS2). Distributed counterparts prefix Distributed (e.g. DistributedDiscreteContinuousConvS2). Transpose counterparts append TransposeS2.
- Public functions: lower_snake_case (e.g. compute_split_shapes).
- Internal helpers: leading underscore (e.g. _compute_dtype, _get_psi).
- Low-level ops follow _<op-family>_<direction>_<variant> (e.g. _disco_s2_contraction_optimized, _neighborhood_s2_attention_bwd_dq_torch).
- Module-level constants: UPPER_SNAKE_CASE (public), _UPPER_SNAKE_CASE (internal — e.g. distributed-state globals like _POLAR_PARALLEL_GROUP).
- Tests: file test_<area>.py mirroring the source module; class Test<PascalCase>; method test_<lower_snake_case>.
- Prefer verbose names that read like English. Abbreviate only when the short form is mathematical convention (l, m, n for orders/degrees). When in doubt, write it out.

## Project structure

```
torch_harmonics/
├── sht.py, legendre.py, quadrature.py, truncation.py, fft.py, ...  # Core transforms
├── filter_basis.py                                                  # DISCO filter bases
├── disco/                                                           # DISCO convolution
│   ├── convolution.py                                               # High-level API
│   ├── optimized/                                                   # C++/CUDA kernels
│   └── kernels_torch/                                               # Differentiable reference
├── attention/                                                       # Spherical attention
│   ├── attention.py                                                 # AttentionS2, NeighborhoodAttentionS2
│   ├── optimized/                                                   # C++/CUDA kernels
│   └── kernels_torch/                                               # Torch reference implementations
└── distributed/                                                     # Multi-GPU primitives and layers

tests/              # Pytest suite (shared helpers in testutils.py)
benchmarks/         # Performance benchmark suite (run.py entry point)
examples/           # Training and usage examples (not run in default CI)
notebooks/          # Exploratory notebooks (not run in CI)
```

## Guidelines by area


### Spherical harmonic transforms

- Grid types (`equiangular`, `legendre-gauss`, `lobatto`, …) and normalization modes
  (`ortho`, `schmidt`, `four-pi`) have different truncation rules; see `truncation.py` and
  existing SHT tests.
- Document breaking changes to truncation or grid conventions in `Changelog.md`.

### DISCO convolution

- Filter basis types and normalization modes (`nodal`, `modal`, `geometric`, …) are
  covered by `tests/test_filter_basis.py`.
- Deprecated API names may still work with `DeprecationWarning`; prefer the new names in
  new code.

### Attention

- Longitude grid ratios must be compatible with the p-shift indexing:
  - **Downsample / self-attention:** `nlon_in % nlon_out == 0`
  - **Upsample:** `nlon_out % nlon_in == 0` (serial `NeighborhoodAttentionS2`; distributed
    upsample not implemented yet)
- After editing attention C++ or CUDA sources, rebuild and run `tests/test_attention.py`
  (including upsample cases).

### Custom operators

- If optimized kernels are implemented, torch reference implementations must be provided to ensure
 readability and their outputs must agree within test tolerances.
- Test tolerances can be adjusted, however this needs to be justified and clearly documented.
- If you change kernel semantics, update both paths (or the shared sparsity / indexing logic) and add or extend tests.

**Performance requirements for kernel rewrites.** Any PR that rewrites or significantly
modifies a CUDA/C++ kernel must demonstrate a speedup on the relevant benchmark entries
and must not regress existing ones:

1. Check whether `benchmarks/` already has entries that cover the parameter regime of your
   change (resolution, channel count, dtype). If not, add benchmark entries for the cases
   of interest.
2. Run the benchmark on a representative GPU, save a baseline from `main`, then measure
   your branch:
   ```bash
   git stash   # or checkout main
   python benchmarks/run.py --name <relevant_prefix> --save-csv before.csv
   git stash pop   # or checkout your branch + rebuild
   python benchmarks/run.py --name <relevant_prefix> --reference-csv before.csv
   ```
3. Include the comparison table (copy-paste the terminal output) in your PR description.
   Report both forward and backward speedups. If any existing entry regresses by more than
   5%, explain why or fix it before requesting review.

### PT2 / `torch.compile` compatibility

torch-harmonics targets PyTorch 2.x and `torch.compile`. New or changed custom CUDA kernels and autograd
paths must stay PT2-safe.

**Custom ops.** Register kernels with `@torch.library.custom_op` and a matching
`@torch.library.register_fake` (see `torch_harmonics/attention/optimized/attention_optimized.py` and
`torch_harmonics/disco/optimized/disco_optimized.py`). Add an `opcheck`-based test for each registered op:

```python
from torch.library import opcheck

opcheck(torch.ops.<namespace>.<op_name>, test_inputs)
```

Templates in the test suite:

- `tests/test_attention.py::TestNeighborhoodAttentionS2::test_optimized_pt2_compatibility` — main attention op
- `tests/test_attention.py::TestNeighborhoodAttentionS2::test_ring_kernels_pt2_compatibility` — ring-step kernels (one `opcheck` per op; single-rank mode avoids NCCL)
- `tests/test_convolution.py` — DISCO ops

`opcheck` verifies the op contract (schema, fake tensors, AOT dispatch); inputs need correct shapes but not
numerically meaningful values.

**Autograd backward contract.** In `torch.autograd.Function.backward` and `register_autograd` handlers,
return `None` for every input position where `ctx.needs_input_grad[i]` is `False`. Do not return zero
tensors or omit slots with the wrong arity. This is required for `torch.compile` / AOTAutograd to prune
dead subgraphs correctly — not optional. Check `ctx.needs_input_grad` before expensive kernel or collective
work (see `_neighborhood_s2_attention_bwd_torch` in `torch_harmonics/attention/kernels_torch/attention_torch.py`).

**Untraceable boundaries.** Wrap Python entry points that call NCCL P2P (`dist.batch_isend_irecv`),
process-group setup, or other code Dynamo cannot trace in `@torch.compiler.disable()`. That forces a clean
graph break instead of an opaque compile failure. See `torch_harmonics/distributed/primitives.py`.

### Distributed

- Distributed modules are checked against their local counterparts.
- Use `TORCH_HARMONICS_DISTRIBUTED_DEBUG` for extra shape checks (see distributed module
  docs).

## Pull requests

1. **Branch** from `main` using `username/feature-name` (e.g. `jdoe/fourier-bessel-basis`).
2. **Keep PRs focused.** One logical change per PR is easier to review.
3. **Add tests** for bug fixes and new behavior.
4. **Update `Changelog.md`** for user-visible changes, especially breaking ones.
5. **Describe the PR clearly:** what problem it solves, how you tested it (commands,
   CPU/GPU), and any API or numerical behavior changes.
6. **Ensure CI passes:** style (pre-commit) and tests workflows.
7. **For kernel rewrites:** include a benchmark comparison table showing speedup on the
   affected entries and no regression on existing ones. See
   [Running benchmarks](#running-benchmarks) and
   [Custom operators](#custom-operators) for the required workflow.

Linking to an open issue when one exists is helpful but not required.

Reviewers may ask for reference-kernel parity checks or justification for tolerance changes.

## Release and packaging

Maintainers handle tagged releases and wheel publishing. Contributors do not need to run
the wheel pipeline locally.

Prebuilt manylinux wheels are built in CI only, via
[`.github/workflows/build_wheels.yml`](.github/workflows/build_wheels.yml). That workflow
runs when a version tag matching `v*` is pushed, or when triggered manually from the
GitHub Actions UI (`workflow_dispatch`). It is not part of the default PR checks
(`style` and `tests`).

---

Questions? Open a [GitHub issue](https://github.com/NVIDIA/torch-harmonics/issues) or
contact the maintainers listed in [README.md](README.md#contributors).
