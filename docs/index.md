# torch-harmonics

**Differentiable signal processing on the sphere for PyTorch.**

`torch-harmonics` implements differentiable spherical harmonic transforms (SHT),
discrete-continuous (DISCO) convolutions, spherical attention, and related
operators as PyTorch modules. All operators are autograd-compatible and run on
CPU and GPU, with optional custom CUDA kernels for the performance-critical
paths.

```{toctree}
---
maxdepth: 2
caption: Getting started
---
install
benchmarking
tutorials/index
```

```{toctree}
---
maxdepth: 1
caption: User guide
---
guide/spherical_harmonic_transforms
guide/spectral_convolutions
guide/disco_convolutions
guide/spherical_attention
guide/distributed
```

```{toctree}
---
maxdepth: 2
caption: API reference
---
api/serial
api/distributed_helpers
api/distributed_layers
api/distributed_primitives
api/utilities
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

```{toctree}
---
maxdepth: 1
caption: Bibliography
---
references
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
