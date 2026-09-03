# Utilities

Grids, plotting, quadrature, and helper functions.

(grids)=

## Grids

A `GridS2` describes a grid on the sphere: where its nodes sit, what quadrature
weights go with them, and the quantities derived from that node distribution,
such as the default angular cutoff and the spectral bounds an SHT can be
truncated to. It is the single argument that replaces a
`(nlat, nlon, grid_string)` triple, so a new grid type can be added without
editing every consumer.

The hierarchy has two levels. `GridS2` describes a stack of latitude rings and
says nothing about how many longitudes each ring carries, so it covers ragged
grids such as reduced Gaussian or HEALPix. `RegularGridS2` adds the assumption
that every ring is sampled alike, which is what makes a field a dense
`(nlat, nlon)` array, makes the SHT separable, and makes a 2D process
decomposition meaningful. Every grid implemented today is a `RegularGridS2`, and
routines that rely on that call `require_regular_grid`.

Use `as_grid` to build one, from a grid type name and the parameters that type
takes:

```python
from torch_harmonics import as_grid, grid_params

grid = as_grid("legendre-gauss", nlat=128, nlon=256)
grid.lats, grid.quad_weights     # nodes and weights
grid.theta_cutoff()              # default support radius for localized operators
grid.max_exact_degree            # highest degree the quadrature integrates exactly

grid_params("legendre-gauss")    # ('nlat', 'nlon') -- what this grid type takes
```

The parameters are validated against the grid type, so a parameter that is
meaningless for a grid family is rejected rather than ignored. A grid family
parameterized by a refinement level instead of `(nlat, nlon)` therefore needs no
changes here: identity, hashing, `to_dict`/`from_dict` and the error messages are
all derived from its own fields.

```{eval-rst}
.. currentmodule:: torch_harmonics.grid

.. autosummary::
   :toctree: generated
   :nosignatures:

   as_grid
   grid_params
   grid_types
   require_grid
   require_regular_grid
   GridS2
   RegularGridS2
   GridShardS2
   RegularGridShardS2
   EquiangularGrid
   LegendreGaussGrid
   LobattoGrid
   TrapezoidalGrid
```

## Quadrature

torch-harmonics supports several quadrature rules for the latitudinal
direction. Each is named by one of the strings below, which is what `as_grid`
takes to build the descriptor the layers are given:

| Grid string        | Quadrature rule | Nodes                                            | Key properties                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ------------------ | --------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"equiangular"`    | Clenshaw–Curtis | Equally spaced in $\theta$ (including poles)     | Default grid. Exact for polynomials up to degree $N-1$. Simple, FFT-friendly.                                                                                                                                                                                                                                                                                                                                                                              |
| `"legendre-gauss"` | Gauss–Legendre  | Roots of $P_N(\cos\theta)$                       | Exact for polynomials up to degree $2N-1$. Optimal accuracy per node, but nodes are non-uniform.                                                                                                                                                                                                                                                                                                                                                           |
| `"lobatto"`        | Gauss–Lobatto   | Roots of $P'_{N-1}(\cos\theta)$, plus endpoints  | Exact for polynomials up to degree $2N-3$. Includes both poles, useful when pole values are needed.                                                                                                                                                                                                                                                                                                                                                        |
| `"trapezoidal"`    | Trapezoidal     | Equally spaced in $\cos\theta$ (including poles) | Supports periodic grids. Lower-order accuracy but simplest structure. The nodes are equispaced in $\cos\theta$, *not* in $\theta$, so the spacing in $\theta$ is strongly non-uniform: the polar spacing is a factor $\sqrt{N_\theta - 1}$ coarser than the equatorial one, a disparity that grows with resolution rather than staying fixed. Formerly named `"equiangular-trapezoidal"`, after nodes it does not have; that string is no longer accepted. |

The longitudinal direction always uses equispaced nodes (see
`precompute_longitudes`).

Because only `"equiangular"` has uniform spacing in $\theta$, quantities derived
from "one latitudinal grid spacing" must come from the grid's actual node
distribution rather than from $\pi / (N_\theta - 1)$; see
`compute_latitude_spacing` and `compute_theta_cutoff`. The gap is largest for
`"trapezoidal"`, whose maximum spacing exceeds $\pi / (N_\theta - 1)$
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

Both routines answer the same question in dual spaces: given a grid, how much
of an operator should be kept by default, and what happens when the caller says
otherwise. `truncate_sht` bounds the spectrum an SHT retains; `truncate_support`
bounds the angular radius a DISCO convolution or neighborhood attention reaches
over. Each takes the bound the grid can support, applies an explicit override if
one is given, and warns when the default it picks differs from a previous
release's. The grid descriptor states the facts these decisions rest on --
`max_exact_degree`, `max_azimuthal_order`, `max_latitude_spacing` -- and makes no
decisions itself.

```{eval-rst}
.. currentmodule:: torch_harmonics

.. autosummary::
   :toctree: generated
   :nosignatures:

   truncate_sht
   truncate_support
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
