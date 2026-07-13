# torch-harmonics

**Differentiable signal processing on the sphere for PyTorch.**

`torch-harmonics` implements differentiable spherical harmonic transforms (SHT),
discrete-continuous (DISCO) convolutions, spherical attention, and related
operators as PyTorch modules. All operators are autograd-compatible and run on
CPU and GPU, with optional custom CUDA kernels for the performance-critical
paths.

```{toctree}
---
maxdepth: 1
caption: Getting started
---
install
tutorials/index
```

```{toctree}
---
maxdepth: 2
caption: User guide
---
guide/index
```

```{toctree}
---
maxdepth: 2
caption: Reference
---
api/index
```

## Quick example

```python
import torch
import torch_harmonics as th

# forward / inverse real spherical harmonic transform on an equiangular grid
sht = th.RealSHT(nlat=128, nlon=256, grid="equiangular")
isht = th.InverseRealSHT(nlat=128, nlon=256, grid="equiangular")

signal = torch.randn(1, 128, 256)
coeffs = sht(signal)          # -> spherical harmonic coefficients
reconstructed = isht(coeffs)  # -> back to grid space
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
