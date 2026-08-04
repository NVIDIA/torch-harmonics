# Utilities

Plotting, quadrature, and helper functions.

## Quadrature

torch-harmonics supports several quadrature rules for the latitudinal
direction. Each corresponds to a `grid` keyword accepted by the SHT and
convolution layers:

| Grid string                 | Quadrature rule | Nodes                                           | Key properties                                                                                      |
| --------------------------- | --------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `"equiangular"`             | Clenshaw–Curtis | Equally spaced in $\theta$ (including poles)    | Default grid. Exact for polynomials up to degree $N-1$. Simple, FFT-friendly.                       |
| `"legendre-gauss"`          | Gauss–Legendre  | Roots of $P_N(\cos\theta)$                      | Exact for polynomials up to degree $2N-1$. Optimal accuracy per node, but nodes are non-uniform.    |
| `"lobatto"`                 | Gauss–Lobatto   | Roots of $P'_{N-1}(\cos\theta)$, plus endpoints | Exact for polynomials up to degree $2N-3$. Includes both poles, useful when pole values are needed. |
| `"equiangular-trapezoidal"` | Trapezoidal     | Equally spaced                                  | Supports periodic grids. Lower-order accuracy but simplest structure.                               |

The longitudinal direction always uses equispaced nodes (see
`precompute_longitudes`).

```{eval-rst}
.. currentmodule:: torch_harmonics.quadrature

.. autosummary::
   :toctree: generated
   :nosignatures:

   precompute_longitudes
   precompute_latitudes
   legendre_gauss_weights
   lobatto_weights
   clenshaw_curtiss_weights
   trapezoidal_weights
```

## Plotting

```{eval-rst}
.. currentmodule:: torch_harmonics.plotting

.. autosummary::
   :toctree: generated
   :nosignatures:

   plot_sphere
   imshow_sphere
```

## Truncation

```{eval-rst}
.. currentmodule:: torch_harmonics

.. autosummary::
   :toctree: generated
   :nosignatures:

   truncate_sht
```

## Debugging

```{eval-rst}
.. currentmodule:: torch_harmonics.distributed

.. autodata:: config
   :no-value:
```

The `config` object exposes a single boolean property, `debug`.
When enabled, the distributed primitives perform extra shape-verification
checks on every collective call, which is useful for diagnosing partitioning
mismatches.

```python
from torch_harmonics.distributed import config

# enable programmatically
config.debug = True

# or via environment variable (before importing)
# TORCH_HARMONICS_DISTRIBUTED_DEBUG=1
```
