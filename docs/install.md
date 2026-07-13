# Installation

## Prebuilt wheels (GPU and CPU)

Prebuilt Linux wheels with compiled CUDA extensions are published on
[pypi.nvidia.com](https://pypi.nvidia.com), one per CUDA toolkit version. The
packages are named `torch-harmonics-cu<CUDA>` — for example
`torch-harmonics-cu126` for CUDA 12.6. Check
[pypi.nvidia.com](https://pypi.nvidia.com) for the packages currently available
and the PyTorch release each one targets, then install the one matching your
CUDA toolkit:

```bash
pip install torch-harmonics-cu<CUDA> --extra-index-url https://pypi.nvidia.com
```

If you don't need a specific CUDA version, use one of the rolling aliases, which
always track the newest build:

```bash
# latest CUDA build
pip install torch-harmonics-cuda-latest --extra-index-url https://pypi.nvidia.com

# CPU only
pip install torch-harmonics-cpu-latest --extra-index-url https://pypi.nvidia.com
```

```{tip}
Run `nvidia-smi` to check the CUDA version supported by your driver.
```

## PyPI (CPU only)

The vanilla [`torch-harmonics`](https://pypi.org/project/torch_harmonics/)
package on PyPI ships a CPU-only prebuilt wheel, built for the newest PyTorch
release. For GPU support, use the NVIDIA packages above.

```bash
pip install torch-harmonics
```

## Building from source

If your OS, PyTorch, or CUDA toolkit version is not covered by the available
wheels, we recommend building from the GitHub repository. Use
`--no-build-isolation` so that the custom CPU and CUDA kernels compile against
your existing PyTorch installation:

```bash
git clone https://github.com/NVIDIA/torch-harmonics.git
cd torch-harmonics
pip install --no-build-isolation -e .
```

The custom CUDA kernels are built automatically when a CUDA toolkit is detected;
otherwise the pure-PyTorch fallbacks are used. If CUDA devices are not detected
automatically (e.g. inside a container), set `TORCH_HARMONICS_BUILD_CUDA_EXTENSION`. Set
`TORCH_CUDA_ARCH_LIST` to only the architectures you need to reduce compilation
time:

```bash
export TORCH_HARMONICS_BUILD_CUDA_EXTENSION=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0 10.0+PTX"
pip install --no-build-isolation -e .
```

```{warning}
The custom CUDA extensions require compute capability >= 7.0.
```

```{warning}
The custom C++/CUDA extensions are compiled against the PyTorch C++ ABI at
build time.  If you upgrade PyTorch after installing torch-harmonics, the
compiled extensions may become incompatible and cause crashes or undefined
behaviour at runtime.  **After upgrading PyTorch, always rebuild
torch-harmonics** (``pip install --no-build-isolation -e .`` for source
installs, or reinstall the matching prebuilt wheel).
```

## Docker

Alternatively, build and run a Docker container:

```bash
git clone https://github.com/NVIDIA/torch-harmonics.git
cd torch-harmonics
docker build . -t torch_harmonics
docker run --gpus all -it --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 torch_harmonics
```

## Requirements

`torch-harmonics` requires a recent Python and PyTorch; see the `requires-python`
and `dependencies` fields in
[`pyproject.toml`](https://github.com/NVIDIA/torch-harmonics/blob/main/pyproject.toml)
for the exact minimum versions.

## Building the documentation

```bash
pip install -e ".[docs]"
cd docs
make html
```

The rendered site is written to `docs/_build/html/index.html`.
