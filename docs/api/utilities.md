# Utilities

Grids, plotting, quadrature, and helper functions.

(grids)=

## Grids

Grid descriptors say where the sample points sit and which quadrature weights go
with them; every layer takes one. See the {doc}`grids guide </guide/grids>` for
the available grids, field shapes and migration from earlier versions.

The descriptors form a hierarchy, each level adding a constraint on how the points
are arranged:

| level           | adds                                       | enables                                                        |
| --------------- | ------------------------------------------ | -------------------------------------------------------------- |
| `PointSetS2`    | points and weights                         | integration                                                    |
| `GridS2`        | isolatitude rings, equispaced in longitude | an FFT in longitude (hence a fast SHT), ragged grids (HEALPix) |
| `RegularGridS2` | every ring the same length                 | a dense `(nlat, nlon)` layout and a 2D decomposition           |

Today `QuadratureS2` accepts any `PointSetS2`, `AttentionS2` and
`NeighborhoodAttentionS2` any `GridS2`, and all other layers a `RegularGridS2`.
`SpectralGrid` is separate from this spatial hierarchy: it describes the
coefficient support of an SHT.

```{eval-rst}
.. currentmodule:: torch_harmonics.grid

.. autosummary::
   :toctree: generated
   :nosignatures:

   as_grid
   grid_params
   grid_types
   require_point_set
   require_grid
   require_regular_grid
   PointSetS2
   GridS2
   RegularGridS2
   SpectralGrid
   GridShardS2
   RegularGridShardS2
   EquiangularGrid
   Fejer1Grid
   LegendreGaussGrid
   LobattoGrid
   TrapezoidalGrid

.. currentmodule:: torch_harmonics.healpix

.. autosummary::
   :toctree: generated
   :nosignatures:

   HealpixGrid
```

### Radial and volumetric grids

A field on a ball, a shell or the space outside a sphere is sampled on the same angular
grid at every radius, which is the direct product of a radial grid and a sampling of the
sphere. A `SphericalProductGrid` pairs a `RadialGrid` with any of the grids above, and a
field on it is laid out `(..., nr, *angular.shape)`, the `(D, H, W)` order of volumetric
data on a regular angular grid. Every layer above accepts a product in place of its
angular grid and acts on each shell alike, with the radius as one more batch axis; a layer
with an input and an output grid needs both on the same radial grid. `QuadratureS2` on a
product integrates each shell over the sphere; `QuadratureVolume` integrates over the volume. Radial rules have a
registry of their own, so `as_radial_grid("legendre-gauss", ...)` is the radial rule and
`as_grid("legendre-gauss", ...)` the angular grid.

```{eval-rst}
.. currentmodule:: torch_harmonics.radial

.. autosummary::
   :toctree: generated
   :nosignatures:

   as_radial_grid
   radial_grid_types
   RadialGrid
   GeometricRadialGrid
   LegendreGaussRadialGrid
   SphericalProductGrid
```

## Quadrature

torch-harmonics supports several quadrature rules for the latitudinal
direction. Each is named by one of the strings below, which is what `as_grid`
takes to build the descriptor the layers are given:

| Grid string        | Quadrature rule | Nodes                                            | Key properties                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ------------------ | --------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"equiangular"`    | Clenshaw–Curtis | Equally spaced in $\theta$ (including poles)     | Default grid. Exact through degree $N-1$ for even $N$ and $N$ for odd $N$. Native direct-analysis bound `(N+1)//2`; folded analysis can recover degrees through $N-2$.                                                                                                                                                                                                                                                                                     |
| `"fejer1"`         | Fejér first     | Equally spaced in $\theta$ (poles excluded)      | Exact through degree $N-1$ for even $N$ and $N$ for odd $N$. Native direct-analysis bound `(N+1)//2`; folded analysis can recover degrees through $N-1$.                                                                                                                                                                                                                                                                                                   |
| `"legendre-gauss"` | Gauss–Legendre  | Roots of $P_N(\cos\theta)$                       | Exact for polynomials up to degree $2N-1$. Optimal accuracy per node, but nodes are non-uniform.                                                                                                                                                                                                                                                                                                                                                           |
| `"lobatto"`        | Gauss–Lobatto   | Roots of $P'_{N-1}(\cos\theta)$, plus endpoints  | Exact for polynomials up to degree $2N-3$. Includes both poles, useful when pole values are needed.                                                                                                                                                                                                                                                                                                                                                        |
| `"trapezoidal"`    | Trapezoidal     | Equally spaced in $\cos\theta$ (including poles) | Supports periodic grids. Lower-order accuracy but simplest structure. The nodes are equispaced in $\cos\theta$, *not* in $\theta$, so the spacing in $\theta$ is strongly non-uniform: the polar spacing is a factor $\sqrt{N_\theta - 1}$ coarser than the equatorial one, a disparity that grows with resolution rather than staying fixed. Formerly named `"equiangular-trapezoidal"`, after nodes it does not have; that string is no longer accepted. |

The longitudinal direction always uses equispaced nodes (see
`precompute_longitudes`). Folded scalar and vector SHT analysis on equiangular
and Fejér-1 grids requires `mmax <= (nlon + 1) // 2`, excluding the longitude
Nyquist order when `nlon` is even. See the {doc}`grids guide </guide/grids>`
for the latitude limits and construction costs.

Both `"equiangular"` and `"fejer1"` have uniform spacing in $\theta$:
$\pi/(N_\theta-1)$ and $\pi/N_\theta$, respectively. Other grids have
non-uniform latitude spacing. Use `compute_latitude_spacing` or
`compute_theta_cutoff` to obtain the spacing for each grid. The gap is largest for
`"trapezoidal"`, whose maximum spacing exceeds $\pi / (N_\theta - 1)$
by a factor of about $2\sqrt{N_\theta - 1} / \pi$ — roughly $5\times$ at
$N_\theta = 65$ and $17\times$ at $N_\theta = 721$. For `"lobatto"` the excess is
a resolution-independent ~21%. `truncate_support` also considers longitude
spacing when choosing the default support of localized operators.

The localized operators additionally need to know which latitudes a cutoff can
reach, which is what `latitude_support_band` returns; `effective_theta_cutoff`
applies the widening that the sparsity patterns are actually built with.

```{eval-rst}
.. currentmodule:: torch_harmonics.quadrature

.. autosummary::
   :toctree: generated
   :nosignatures:

   precompute_longitudes
   precompute_latitudes
   compute_latitude_spacing
   compute_theta_cutoff
   effective_theta_cutoff
   latitude_support_band
   legendre_gauss_weights
   lobatto_weights
   clenshaw_curtiss_weights
   fejer1_weights
   trapezoidal_weights
```

## Neighborhoods

Which input points lie within an angular cutoff of each output point -- the
pattern neighborhood attention attends over. A geodesic disk meets a latitude
ring in a single contiguous arc, so the pattern is computed and stored as
`(ring, start, length)` arcs, on any `GridS2`; on a latitude-longitude grid it
holds one row per output ring, the other longitudes being reached by a shift.
`precompute_neighborhood_csr_s2` expands the arcs to an explicit column list for
the consumers that want one.

```{eval-rst}
.. currentmodule:: torch_harmonics.neighborhood

.. autosummary::
   :toctree: generated
   :nosignatures:

   precompute_neighborhood_arcs_s2
   precompute_neighborhood_csr_s2
   NeighborhoodArcsS2
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
`max_exact_degree`, `max_azimuthal_order`, `max_node_spacing` -- and makes no
decisions itself.

`truncate_sht` returns the resolved bounds as a `SpectralGrid`.

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
