# Utilities

Plotting, quadrature, and helper functions.

## Quadrature

torch-harmonics supports several quadrature rules for the latitudinal
direction. Each corresponds to a `grid` keyword accepted by the SHT and
convolution layers:

| Grid string                 | Quadrature rule | Nodes                                            | Key properties                                                                                                                                                                                                                                                                                                        |
| --------------------------- | --------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"equiangular"`             | Clenshaw–Curtis | Equally spaced in $\theta$ (including poles)     | Default grid. Exact for polynomials up to degree $N-1$. Simple, FFT-friendly.                                                                                                                                                                                                                                         |
| `"legendre-gauss"`          | Gauss–Legendre  | Roots of $P_N(\cos\theta)$                       | Exact for polynomials up to degree $2N-1$. Optimal accuracy per node, but nodes are non-uniform.                                                                                                                                                                                                                      |
| `"lobatto"`                 | Gauss–Lobatto   | Roots of $P'_{N-1}(\cos\theta)$, plus endpoints  | Exact for polynomials up to degree $2N-3$. Includes both poles, useful when pole values are needed.                                                                                                                                                                                                                   |
| `"equiangular-trapezoidal"` | Trapezoidal     | Equally spaced in $\cos\theta$ (including poles) | Supports periodic grids. Lower-order accuracy but simplest structure. Despite the name, the nodes are *not* equiangular in $\theta$: the rule is applied on the $\cos\theta$ interval $[-1, 1]$, so the spacing in $\theta$ is strongly non-uniform and roughly $5\times$ coarser near the poles than at the equator. |

The longitudinal direction always uses equispaced nodes (see
`precompute_longitudes`).

`geometric_weights` is not a latitudinal rule: it returns nodes that are
equispaced in $\log x$ on a positive interval, together with the corresponding
trapezoidal weights for $\int f \, \mathrm{d}x$. It is intended for radial
directions spanning several decades. `precompute_radii` builds on it to return
the radial grid of either the half-line $(0, \infty)$ or the exterior domain
$[R, \infty)$, where the geometric spacing is applied to the reduced coordinate
$(r - R)/R$.

Because only `"equiangular"` has uniform spacing in $\theta$, quantities derived
from "one latitudinal grid spacing" must come from the grid's actual node
distribution rather than from $\pi / (N_\theta - 1)$; see
`compute_latitude_spacing` and `compute_theta_cutoff`. The localized operators
additionally need to know which latitudes a cutoff can reach, which is what
`latitude_support_band` returns; `effective_theta_cutoff` applies the widening
that the sparsity patterns are actually built with.

```{eval-rst}
.. currentmodule:: torch_harmonics.quadrature

.. autosummary::
   :toctree: generated
   :nosignatures:

   precompute_longitudes
   precompute_latitudes
   precompute_radii
   compute_latitude_spacing
   compute_theta_cutoff
   effective_theta_cutoff
   latitude_support_band
   legendre_gauss_weights
   lobatto_weights
   clenshaw_curtiss_weights
   trapezoidal_weights
   geometric_weights
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
