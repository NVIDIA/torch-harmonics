# Installation

## From PyPI

```bash
pip install torch-harmonics
```

## From source

```bash
git clone https://github.com/NVIDIA/torch-harmonics.git
cd torch-harmonics
pip install -e .
```

`torch-harmonics` requires Python ≥ 3.9 and PyTorch ≥ 2.6. The optional custom
CUDA kernels are built automatically when a CUDA toolkit is available at install
time; otherwise the pure-PyTorch fallbacks are used.

## Building the documentation

```bash
pip install -e ".[docs]"
cd docs
make html
```

The rendered site is written to `docs/_build/html/index.html`.
