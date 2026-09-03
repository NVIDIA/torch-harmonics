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

# the grid descriptor carries the resolution and the quadrature rule together
grid = th.as_grid("equiangular", nlat=128, nlon=256)

# forward / inverse real spherical harmonic transform on that grid
sht = th.RealSHT(grid)
isht = th.InverseRealSHT(grid)

signal = torch.randn(1, 128, 256)
coeffs = sht(signal)          # -> spherical harmonic coefficients
reconstructed = isht(coeffs)  # -> back to grid space
```

Every operator takes a grid descriptor rather than a resolution and a grid
name. The descriptor carries both, together with everything that follows from
where the nodes sit: the quadrature weights, the angular cutoff localized
operators default to, the degree an SHT can be truncated to, and how the grid
decomposes across ranks. Operators mapping between two grids take `grid_in` and
`grid_out` in that leading position. See {ref}`grids`.

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
