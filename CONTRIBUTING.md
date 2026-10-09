# Contributing to torch-harmonics

Thank you for your interest in contributing. torch-harmonics implements differentiable
signal processing on the sphere: spherical harmonic transforms, DISCO convolutions, spherical
(neighborhood) attention, resampling, spectral convolutions, and quadrature, on regular
latitude-longitude grids and, where supported, HEALPix. It also provides distributed (multi-GPU)
variants of these layers and reference neural operator models. We are grateful for
contributions that improve correctness, performance, documentation, or test coverage.

## Table of contents

- [Getting in touch](#getting-in-touch)
- [Contribution policy](#contribution-policy)
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
- Changes start with an issue, so that we can agree on the approach before you write code.
  Only small documentation and typo fixes can go straight to a PR; see
  [Contribution policy](#contribution-policy).
- Please be respectful and constructive in issues and reviews.

## Contribution policy

We welcome contributions from anyone with a genuine interest in the project, including work
done with the help of AI tools. What we ask is that a person stands behind every pull request.

- **A human is accountable for every PR.** Whoever opens the PR must understand the change,
  be able to explain and defend it in review, and respond to feedback themselves. Pull
  requests opened autonomously by bots or AI agents are closed without review.
- **Disclose AI assistance.** If AI tools wrote a substantial part of the code, tests, or
  description, say so in the PR description and name the tool. Disclosure does not count
  against a PR; it tells reviewers where to look carefully.
- **Open an issue first.** Describe the problem and your intended approach in an issue (or
  comment on an existing one), and wait for a maintainer to confirm the change is wanted
  before opening a PR. Small documentation and typo fixes are exempt. PRs without an agreed
  issue may be closed.
- **Review time is limited.** Maintainers may close PRs that are out of scope, do not follow
  these guidelines, or cannot be reviewed with reasonable effort, without a detailed
  justification. Large changes produced with little human input fall into that category.
- **Credit.** [AUTHORS](AUTHORS) and the [Contributors](README.md#contributors) list in the
  README are maintained by hand. We add people who have made sustained, substantive
  contributions; individual merged PRs are credited through the git history.

## Development setup

**Requirements**

- Python 3.11+
- PyTorch 2.9+ (install before building; extensions compile against your local `torch`)
- NumPy 1.22.4+
- A C++17 compiler; CUDA toolkit optional but recommended for GPU kernel work

**Editable install**

```bash
git clone https://github.com/NVIDIA/torch-harmonics.git
cd torch-harmonics

# Install PyTorch first (CPU example; use the CUDA wheel that matches your system)
python3 -m pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

# Build requirements; without setuptools-scm the version silently resolves to 0.0.0
python3 -m pip install setuptools setuptools-scm wheel

# Editable install with dev dependencies
python3 -m pip install -e ".[dev]" --no-build-isolation
```

Optional extras:

```bash
# earth2grid, to check the HEALPix grid against an independent implementation; it is not
# on PyPI and builds against the installed torch, hence the separate pinned requirements.
# --no-deps keeps pip from replacing your torch to satisfy earth2grid's exact torch pin.
pip install numpy packaging einops setuptools wheel
pip install --no-deps --no-build-isolation -r .github/requirements/test-tools.txt
```

The `dev` extra already includes scipy, which the `filter_basis` extra provides for
non-development installs.

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
| `attention_helpers`, `disco_helpers` | Runtime availability checks; top-level modules built at the repository root |

**You must rebuild after changing any `.cpp`, `.cu`, `.h`, or `.cuh` file under `torch_harmonics/csrc/`,
`torch_harmonics/attention/optimized/`, or `torch_harmonics/disco/optimized/`.** Python-only
edits do not require a rebuild, but stale `.so` files are a common source of confusing test
failures (e.g. old shape checks still firing from an outdated binary).

```bash
pip install -e . --no-build-isolation
```

Force a clean extension rebuild if needed:

```bash
rm -f torch_harmonics/attention/_C*.so torch_harmonics/disco/_C*.so attention_helpers*.so disco_helpers*.so
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
export TORCH_HARMONICS_BUILD_CUDA_EXTENSION=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0a"   # set to GPUs you target; reduces compile time
pip install -e . --no-build-isolation
```

Custom CUDA extensions require compute capability **≥ 8.0**. The tensor-core DISCO forward
kernels are only compiled when `TORCH_CUDA_ARCH_LIST` names the arch-specific targets: `9.0a`
for Hopper, `10.0a` / `10.3a` for datacenter Blackwell. With plain `9.0` or `10.0` they compile
to stubs and those GPUs fall back to the generic path.

### Build environment variables

| Variable | Effect |
|----------|--------|
| `TORCH_HARMONICS_BUILD_CUDA_EXTENSION=1` | Build CUDA kernels even if CUDA is not detected at configure time |
| `TORCH_CUDA_ARCH_LIST` | Limit NVCC target architectures; use `9.0a`, `10.0a`, `10.3a` to build the tensor-core DISCO kernels |
| `TORCH_HARMONICS_DEBUG=1` | Debug flags (`-O0`, `-g`) for extensions |
| `TORCH_HARMONICS_PROFILE=1` | NVCC lineinfo / PTXAS verbose (CUDA) |
| `TORCH_HARMONICS_ENABLE_OPENMP=1` | Enable OpenMP in CPU kernels |
| `TORCH_HARMONICS_NATIVE_CPU_ARCH=1` | `-march=native` (local dev only; not for wheels) |

### Installing without a local build

Most users install prebuilt wheels:

- **NVIDIA PyPI** (CUDA): `torch-harmonics-cu126`, `cu128`, `cu129`, etc., or
  `torch-harmonics-cuda-latest`. See [README.md](README.md#installation).
- **PyPI** (`torch-harmonics`): CPU-only wheel for the latest supported PyTorch release.

Install PyTorch first, then the wheel that matches your CUDA toolkit (`nvidia-smi` for the
driver CUDA version).

### Building wheels locally

```bash
python3 -m pip install build setuptools-scm
python3 -m build --wheel --no-isolation
```

The version comes from the git tag through setuptools-scm, so a local build between releases
is named like `torch_harmonics-0.9.4.dev5+g<hash>-cp311-cp311-linux_x86_64.whl`. Release
wheels built in CI carry a `+torch<version>.<cuda>` local tag (e.g. `0.9.3+torch2.9.1.cu129`).
The release scripts strip it when they repack the wheels under the per-CUDA package names, and
pin each package to the torch minor version its wheel was built against.

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
| Extensions not found / stale behavior | Rebuild with `pip install -e . --no-build-isolation`; remove old `_C*.so` and `*_helpers*.so` if needed |
| No matching wheel on install | Install PyTorch first, then torch-harmonics; or build from source |
| ABI mismatch | Rebuild from source against the same PyTorch version |
| CUDA not detected at build time | `export TORCH_HARMONICS_BUILD_CUDA_EXTENSION=1` and set `TORCH_CUDA_ARCH_LIST` |

## Running tests

The serial test suite runs on CPU, with coverage over `torch_harmonics` and without the
distributed tests. With an editable install, the local equivalent of the CI job is:

```bash
python3 -m pytest -ra \
  --cov-report term \
  --cov-config=.coveragerc \
  --cov=torch_harmonics \
  --ignore-glob="**/test_distributed_*" \
  ./tests/
```

CI also runs the docstring examples:

```bash
python3 -m pytest -ra -c pyproject.toml --doctest-modules --pyargs torch_harmonics
```

Run a single test file or case:

```bash
python3 -m pytest tests/test_attention.py -x
python3 -m pytest tests/test_attention.py::TestNeighborhoodAttentionRegularS2_0::test_custom_implementation_20 -x
```

Many tests for DISCO and attention are gated on `optimized_kernels_is_available()`. If
extensions failed to build, those tests are skipped rather than failed—confirm your rebuild
succeeded.

Distributed tests live under `tests/test_distributed_*` and need a multi-process launch. CI
runs them in a separate job; see [Running distributed tests](#running-distributed-tests).

### What to run for your change

| Change type | Suggested tests |
|-------------|-----------------|
| Grids / HEALPix | `tests/test_grid_assumptions.py`, `tests/test_neighborhood.py`, `tests/test_cache.py` |
| SHT / quadrature / resampling | `tests/test_sht.py`, `tests/test_truncation.py`, `tests/test_quadrature.py`, `tests/test_resample.py`, `tests/test_spectral_convolution.py` |
| DISCO convolution | `tests/test_convolution.py`, `tests/test_filter_basis.py`, `tests/test_neighborhood.py` |
| Attention | `tests/test_attention.py`, `tests/test_attention_layout.py`, `tests/test_neighborhood.py` |
| Example models, losses, metrics | `tests/test_example_models.py`, `tests/test_losses.py`, `tests/test_metrics.py` |
| Distributed | `tests/test_distributed_*.py` (CPU process grids run in CI; run on GPUs yourself for the NCCL/CUDA paths) |
| C++/CUDA kernels | Relevant test file **and** compare against the torch reference path; add or extend `opcheck`, autocast-dtype, and `torch.compile` tests (see [PT2 / `torch.compile` compatibility](#pt2--torchcompile-compatibility)) |

CI workflows on pull requests:

- **style**: pre-commit.
- **tests**: the serial suite and doctests on every push and pull request; the distributed
  suite and combined coverage on pull requests and merges to `main`.
- **docs**: Sphinx build with warnings treated as errors.

All of them must pass before merge.

### Running distributed tests

Distributed tests compare the distributed implementations to the serial ones, assuming the
serial ones are correct. CI runs every `tests/test_distributed_*.py` file on CPU with the gloo
backend, on process grids from 1x1 to 4x2 (up to 8 ranks). CUDA and NCCL paths are skipped
there, so run the tests on GPUs yourself when you change them.

Each rank is a separate pytest process. Tests read `WORLD_RANK`, `GRID_H`, `GRID_W`,
`MASTER_ADDR`, and `MASTER_PORT` from the environment; the world size is `GRID_H * GRID_W`
(see `tests/testutils.py`). Set `WORLD_RANK` for every process: it defaults to 0, and `RANK`
is not read in its place. To run one file on a 2x2 grid the way CI does:

```bash
export GRID_H=2 GRID_W=2 MASTER_ADDR=localhost MASTER_PORT=29501
for r in $(seq 0 $((GRID_H * GRID_W - 1))); do
  WORLD_RANK=$r python3 -m pytest -q tests/test_distributed_sht.py > rank$r.log 2>&1 &
done
wait
```

`tests/run_tests.sh -d --grid_size_lat 2 --grid_size_lon 2` launches the same through `mpirun`
(Open MPI), but covers only the SHT, convolution, and resample suites. If you change distributed
routines, run the tests for several combinations of grid sizes.

## Running benchmarks

The `benchmarks/` directory contains a self-contained benchmark suite covering SHT, DISCO
convolution, and spherical attention across multiple resolutions, channel counts, and dtypes.

Run the full suite (GPU and CPU entries; `--device cuda` or `--device cpu` selects one):

```bash
python benchmarks/run.py
```

Filter by name substring or tag:

```bash
python benchmarks/run.py --name disco
python benchmarks/run.py --tags neighborhood   # entries matching any of the given tags
```

Save a baseline CSV, then compare after your change:

```bash
python benchmarks/run.py --save-csv baseline.csv
# ... make your changes and rebuild ...
python benchmarks/run.py --reference-csv baseline.csv
```

The comparison table shows per-entry speedup (`fwd_spd`, `bwd_spd`) and flags regressions
with `!` if throughput drops by more than 5% (adjustable via `--regression-tol`). Rows are
matched by GPU name; `--reference-arch` compares against results from a different GPU, such as
the checked-in `benchmarks/reference_results.csv`.

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

- **pre-commit-hooks** basics: trailing whitespace, end of file, YAML syntax, merge
  conflicts, and large files (over 2.5 MB)
- **black** (line length 180, see `pyproject.toml`)
- **ruff** (lint + import sorting; notebooks excluded)
- **clang-format** for C, C++, and CUDA
- **mdformat** for Markdown under `docs/`
- **SPDX license header** check on Python and C/C++/CUDA files (`scripts/check_license_header.py`)

### License headers

New Python and C/C++/CUDA files must include the BSD-3-Clause SPDX header block used
elsewhere in the repo (`SPDX-FileCopyrightText` and `SPDX-License-Identifier: BSD-3-Clause`)
within their first ten lines. Copy the header from an existing file in the same directory;
C/C++/CUDA sources use `//` comments instead of `#`.

### Python conventions

- Match existing naming, module layout, and documentation level in the area you edit.
- Prefer extending existing helpers over duplicating logic.
- Use `parameterized` for multi-configuration unit tests (see `tests/test_attention.py`).
- Shared test utilities live in `tests/testutils.py`.

### Naming conventions

- Modules: lower_snake_case; prefix with _ for internal-only modules (e.g. _disco_utils.py, attention/_layout.py).
- Public classes: PascalCase. Sphere-valued classes carry the suffix S2 (e.g. DiscreteContinuousConvS2, NeighborhoodAttentionS2). Distributed counterparts prefix Distributed (e.g. DistributedDiscreteContinuousConvS2). Transpose counterparts append TransposeS2.
- Public functions: lower_snake_case (e.g. compute_split_shapes).
- Internal helpers: leading underscore (e.g. _compute_dtype, _get_psi).
- Low-level ops follow _<op-family>_<grid-family>_<direction>_<variant> (e.g. _disco_s2_contraction_regular_optimized,
  _neighborhood_s2_attention_regular_bwd_dq_torch). The grid family is spelled out on both sides -- `regular` for a grid
  with uniform ring length, `ragged` for one without (HEALPix, reduced Gaussian) -- rather than leaving `regular` implied,
  so that neither reads as the default. The same holds for the registered operator names (`forward_regular` /
  `forward_ragged`). Variants already qualified by something else (`kpacked`, `ring_step`, `upsample`) are
  regular-only and keep their names.
- Module-level constants: UPPER_SNAKE_CASE (public), _UPPER_SNAKE_CASE (internal — e.g. distributed-state globals like _POLAR_PARALLEL_GROUP).
- Tests: file test_<area>.py, normally mirroring the source module; class Test<PascalCase>; method test_<lower_snake_case>.
- Prefer verbose names that read like English. Abbreviate only when the short form is mathematical convention (l, m, n for orders/degrees). When in doubt, write it out.

## Project structure

```
torch_harmonics/
├── grid.py, healpix.py                       # Grid descriptors (RegularGridS2, HealpixGrid, as_grid)
├── sht.py, legendre.py, quadrature.py,       # Core transforms
│   truncation.py, fft.py, integration.py
├── resample.py, spectral_convolution.py, random_fields.py
├── neighborhood.py, cache.py, partition.py   # Neighborhoods, caching, domain decomposition
├── _backend.py                               # Backend selection shared by the layers
├── filter_basis.py                           # DISCO filter bases
├── plotting.py, utils.py
├── csrc/                                     # Shared C++/CUDA headers
├── disco/                                    # DISCO convolution
│   ├── convolution.py                        # High-level API
│   ├── backends.py                           # Torch, optimized, and tensor-core backends
│   ├── optimized/                            # C++/CUDA kernels and their Python registration
│   └── kernels_torch/                        # Differentiable reference
├── attention/                                # Spherical attention
│   ├── attention.py                          # AttentionS2, NeighborhoodAttentionS2
│   ├── backends.py                           # Regular- and ragged-grid backends
│   ├── optimized/                            # C++/CUDA kernels and their Python registration
│   └── kernels_torch/                        # Torch references (regular and ragged grids)
├── distributed/                              # Multi-GPU primitives, layers, and kernels
└── examples/                                 # Importable example models, losses, metrics, datasets

tests/              # Pytest suite (shared helpers in testutils.py)
benchmarks/         # Performance benchmark suite (run.py entry point)
examples/           # Training and usage scripts (not run in CI)
notebooks/          # Exploratory notebooks (not run in CI)
```

## Guidelines by area


### Spherical harmonic transforms

- Transforms take a `RegularGridS2` descriptor (see `grid.py` and `as_grid`). Ragged grids such
  as `HealpixGrid` are not supported by the SHT.
- The regular grid types (`equiangular`, `legendre-gauss`, `lobatto`, `trapezoidal`) have
  different truncation rules; see `truncate_sht` in `truncation.py` and
  `tests/test_truncation.py`. The normalization mode does not affect truncation.
- Document breaking changes to truncation or grid conventions in `Changelog.md`.

### DISCO convolution

- DISCO convolutions take `RegularGridS2` grids only.
- Filter basis types (`piecewise linear`, `harmonic`, `zernike`, `fourier-bessel`) are covered
  by `tests/test_filter_basis.py`; basis normalization modes (`none`, `nodal`, `modal`, `mean`,
  `support`, `geometric`) by `tests/test_convolution.py`.
- Deprecated API names may still work with `DeprecationWarning`; prefer the new names in
  new code.

### Attention

- Between two `RegularGridS2` grids, longitude counts must be compatible with the p-shift
  indexing:
  - **Downsample / self-attention:** `nlon_in % nlon_out == 0`
  - **Upsample:** `nlon_out % nlon_in == 0`
- On ragged grids such as `HealpixGrid` no ratio is required; the direction follows the point
  counts. `DistributedNeighborhoodAttentionS2` accepts regular grids only and supports both
  directions.
- After editing attention C++ or CUDA sources, rebuild and run `tests/test_attention.py` and
  `tests/test_attention_layout.py`, covering regular and ragged grids and upsampling.

### Custom operators

- Every optimized kernel needs a torch reference implementation, for readability, and the
  outputs of both must agree within test tolerances. For distributed kernels, the serial layer
  is the reference.
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

torch-harmonics targets PyTorch 2.9+ and `torch.compile`. New or changed custom C++/CUDA
operators (CPU and CUDA) and autograd paths must stay PT2-safe. Use the existing operators in
`attention/optimized/attention_optimized.py`, `attention/_layout.py`, `attention/kernels_torch/`,
and `disco/optimized/disco_optimized.py` as templates.

**Operator registration.** Operators are registered in layers:

1. **Schemas** in C++ with `TORCH_LIBRARY(<namespace>, m)` in the `*_interface.cpp` file. Tag
   each with `at::Tag::pt2_compliant_tag`, and annotate outputs written in place with
   `Tensor(a!)`.
2. **Device kernels** with `TORCH_LIBRARY_IMPL(<namespace>, CPU, m)` or `CUDA` in the kernel's
   source file.
3. **A fake** for every C++ operator, with `torch.library.register_fake` in Python. The fake
   must promise the same output shapes, dtypes, and memory layout as the real kernel. Mirror
   the C++ input checks with `torch_harmonics.utils.check`, which keeps working under
   `fullgraph=True` and with symbolic shapes, and do not branch on sizes.
4. **Differentiable entry points.** Raw C++ operators have no autograd or autocast of their
   own and are only called from a Python `@torch.library.custom_op(..., mutates_args=())`
   wrapper or a `torch.autograd.Function` that provides both. Wrappers register
   `register_fake`, `register_autograd`, and autocast.

**Autocast.** Every differentiable operator needs an autocast implementation at the
`AutocastCUDA` and `AutocastCPU` keys (only `AutocastCUDA` for CUDA-only operators), registered
with `torch.library.impl`. Do not use `torch.library.register_autocast`: it fixes the cast
dtype and cannot follow the active autocast dtype. Layout-only operators (`attention/_layout.py`)
deliberately have no autocast and preserve the input dtype. A new-style `autograd.Function`
(with a separate `setup_context`) that needs AMP uses `_custom_fwd` / `_custom_setup_context`
from `torch_harmonics/distributed/_amp_utils.py`; `torch.amp.custom_fwd` only works with the
legacy API.

**Backend selection.** Layers choose their implementation at construction and when the
module moves (`_apply`, which `.to()`, `.cuda()`, and `.cpu()` go through), never in
`forward`; see `torch_harmonics/_backend.py`. Keep `forward` free of device- or
availability-dependent branching, so that it traces into a single graph.

**Tests.** Add an `opcheck` test for every registered operator, the wrapper **and** the raw C++
operators. Raw operators without autograd get detached inputs. `opcheck` verifies the operator
contract (schema, fake tensors, AOT dispatch); inputs need correct shapes but not numerically
meaningful values.

```python
from torch.library import opcheck

opcheck(torch.ops.<namespace>.<op_name>, test_inputs)
```

Also add an autocast-dtype test, and a `torch.compile(fullgraph=True)` forward and backward
test for a new layer or `autograd.Function` (decorate it with `requires_torch_compile` from
`tests/testutils.py`). Templates in the test suite:

- `tests/test_attention.py::TestNeighborhoodAttentionRegularS2::test_optimized_pt2_compatibility`: regular attention, wrapper and raw operators
- `tests/test_attention.py::TestNeighborhoodAttentionRegularS2::test_ring_kernels_pt2_compatibility` and `::test_ring_upsample_kernels_pt2_compatibility`: ring-step operators (CUDA; single-rank mode avoids NCCL)
- `tests/test_attention.py::TestNeighborhoodAttentionRaggedS2::test_optimized_pt2_compatibility`: ragged attention
- `tests/test_attention_layout.py::TestAttentionLayout::test_opcheck` and `::test_compile`: layout operators
- `tests/test_convolution.py::TestDiscreteContinuousConvolution::test_optimized_pt2_compatibility`: DISCO operators and a fullgraph compile of the layer
- `tests/test_convolution.py::TestKpackedPath::test_kpacked_opcheck`: tensor-core DISCO operator (sm90/sm100)
- `test_optimized_autocast_dtype` in `tests/test_attention.py` and `tests/test_convolution.py`: autocast

**Autograd backward contract.** In `torch.autograd.Function.backward` and `register_autograd`
handlers, return `None` for every input position where `ctx.needs_input_grad[i]` is `False`. Do
not return zero tensors or omit slots with the wrong arity. This is required for
`torch.compile` / AOTAutograd to prune dead subgraphs correctly, and is not optional. Check
`ctx.needs_input_grad` before expensive kernel or collective work: see
`_neighborhood_s2_attention_regular_bwd_torch` in
`torch_harmonics/attention/kernels_torch/attention_regular_torch.py`, and
`_RingNeighborhoodAttentionFn.backward` in `torch_harmonics/distributed/distributed_attention.py`
for skipping collectives. A fused kernel that computes several gradients in one launch still
returns `None` for the ones that are not needed.

**Untraceable boundaries.** Wrap Python entry points that call NCCL P2P
(`dist.batch_isend_irecv`, `req.wait()`), process-group setup, or other code Dynamo cannot trace
in `@torch.compiler.disable()`. That forces a clean graph break instead of an opaque compile
failure. See `torch_harmonics/distributed/primitives.py`. Two related patterns:

- When the graph would break at every collective anyway, disable the whole `forward`, as the
  distributed SHTs do (`torch_harmonics/distributed/distributed_sht.py`).
- Skip collectives when the communicator has a single rank, so that the single-rank path stays
  compilable as one graph (see `torch_harmonics/distributed/distributed_convolution.py`).

### Distributed

- Distributed modules are checked against their local counterparts.
- Use `TORCH_HARMONICS_DISTRIBUTED_DEBUG` for extra shape checks (see distributed module
  docs).

## Pull requests

1. **Link the issue** the PR addresses, agreed with a maintainer beforehand (see
   [Contribution policy](#contribution-policy)). Small documentation and typo fixes are exempt.
2. **Branch** from `main` using `username/feature-name` (e.g. `jdoe/fourier-bessel-basis`).
3. **Keep PRs focused.** One logical change per PR is easier to review.
4. **Add tests** for bug fixes and new behavior.
5. **Update `Changelog.md`** for user-visible changes, especially breaking ones.
6. **Describe the PR clearly:** what problem it solves, how you tested it (commands,
   CPU/GPU), any API or numerical behavior changes, and which AI tools you used, if any.
7. **Ensure CI passes:** the style, tests, and docs workflows.
8. **For kernel rewrites:** include a benchmark comparison table showing speedup on the
   affected entries and no regression on existing ones. See
   [Running benchmarks](#running-benchmarks) and
   [Custom operators](#custom-operators) for the required workflow.

The [pull request template](.github/pull_request_template.md) walks through these points.

Reviewers may ask for reference-kernel parity checks or justification for tolerance changes.

## Release and packaging

Maintainers handle tagged releases and wheel publishing. Contributors do not need to run
the wheel pipeline locally.

Prebuilt manylinux wheels are built in CI only, via
[`.github/workflows/build_wheels.yml`](.github/workflows/build_wheels.yml). That workflow
runs when a version tag matching `v*` is pushed, or when triggered manually from the
GitHub Actions UI (`workflow_dispatch`). It is not part of the default PR checks
(`style`, `tests`, and `docs`). The workflow only uploads the wheels as artifacts; publishing
happens outside it. The documentation is deployed when a full GitHub release is published.

---

Questions? Open a [GitHub issue](https://github.com/NVIDIA/torch-harmonics/issues) or
contact the maintainers listed in [README.md](README.md#contributors).
