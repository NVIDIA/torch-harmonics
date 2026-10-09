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
guide/grids
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
where the nodes sit: the quadrature weights, the node spacing localized
operators take their default cutoff from, the degree an SHT can be truncated to,
and how the grid decomposes across ranks. Operators mapping between two grids
take `grid_in` and `grid_out` in that leading position. Besides the
latitude-longitude grids there is HEALPix, which the attention layers and the DISCO
convolutions accept on either side. See {ref}`grids`.

## Citing torch-harmonics

If you use `torch-harmonics` in your work, please cite the paper describing the
library {cite:p}`Kurth2026`:

```bibtex
@misc{kurth2026library,
      title={A library for differentiable signal processing and machine learning on the sphere},
      author={Thorsten Kurth and Max Rietmann and Mauro Bisson and Andrea Paris and Alberto Carpentieri and Jean Kossaifi and Anima Anandkumar and Christian Hundt and Boris Bonev},
      year={2026},
      eprint={2609.39737},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.39737}
}
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
