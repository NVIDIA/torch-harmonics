# Utilities

Plotting, quadrature, and helper functions.

## Quadrature

torch-harmonics supports several quadrature rules for the latitudinal
direction. Each corresponds to a `grid` keyword accepted by the SHT and
convolution layers:

| Grid string                 | Quadrature rule | Nodes                                            | Key properties                                                                                                                                                                                                                                                                                                                                                                                              |
| --------------------------- | --------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"equiangular"`             | Clenshaw–Curtis | Equally spaced in $\theta$ (including poles)     | Default grid. Exact for polynomials up to degree $N-1$. Simple, FFT-friendly.                                                                                                                                                                                                                                                                                                                               |
| `"legendre-gauss"`          | Gauss–Legendre  | Roots of $P_N(\cos\theta)$                       | Exact for polynomials up to degree $2N-1$. Optimal accuracy per node, but nodes are non-uniform.                                                                                                                                                                                                                                                                                                            |
| `"lobatto"`                 | Gauss–Lobatto   | Roots of $P'_{N-1}(\cos\theta)$, plus endpoints  | Exact for polynomials up to degree $2N-3$. Includes both poles, useful when pole values are needed.                                                                                                                                                                                                                                                                                                         |
| `"equiangular-trapezoidal"` | Trapezoidal     | Equally spaced in $\cos\theta$ (including poles) | Supports periodic grids. Lower-order accuracy but simplest structure. Despite the name, the nodes are *not* equiangular in $\theta$: the rule is applied on the $\cos\theta$ interval $[-1, 1]$, so the spacing in $\theta$ is strongly non-uniform. The polar spacing is a factor $\sqrt{N_\theta - 1}$ coarser than the equatorial one, a disparity that grows with resolution rather than staying fixed. |

The longitudinal direction always uses equispaced nodes (see
`precompute_longitudes`).

Because only `"equiangular"` has uniform spacing in $\theta$, quantities derived
from "one latitudinal grid spacing" must come from the grid's actual node
distribution rather than from $\pi / (N_\theta - 1)$; see
`compute_latitude_spacing` and `compute_theta_cutoff`. The gap is largest for
`"equiangular-trapezoidal"`, whose maximum spacing exceeds $\pi / (N_\theta - 1)$
by a factor of about $2\sqrt{N_\theta - 1} / \pi$ — roughly $5\times$ at
$N_\theta = 65$ and $17\times$ at $N_\theta = 721$. For `"lobatto"` the excess is
a resolution-independent ~21%.

```{eval-rst}
.. currentmodule:: torch_harmonics.quadrature

.. autosummary::
   :toctree: generated
   :nosignatures:

   precompute_longitudes
   precompute_latitudes
   compute_latitude_spacing
   compute_theta_cutoff
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
