# Grids

Every layer in torch-harmonics takes a *grid descriptor*: a small, immutable
object that says where the sample points sit on the sphere and which quadrature
weights go with them. Everything that depends on the sampling -- the default
truncation of an SHT, the default support radius of a DISCO convolution or
neighborhood attention, the shape of a field -- is derived from it, so the
resolution and the grid type are always passed together.

## Building a grid

Build a grid by name with `as_grid`, passing the parameters that grid type takes
by keyword:

```python
import torch_harmonics as th

grid = th.as_grid("legendre-gauss", nlat=128, nlon=256)
sht = th.RealSHT(grid)

th.grid_types()                   # ('equiangular', 'legendre-gauss', 'lobatto', 'trapezoidal', 'healpix')
th.grid_params("healpix")         # ('nside',)
```

The grid classes can also be instantiated directly, e.g.
`th.LegendreGaussGrid(nlat=128, nlon=256)` or `th.HealpixGrid(nside=64)`.
Parameters are checked against the grid type, so passing `nside` to an
equiangular grid raises instead of being ignored.

## Available grids

| grid type          | class               | parameters    | nodes                                                      | poles | spherical harmonic transform |
| ------------------ | ------------------- | ------------- | ---------------------------------------------------------- | ----- | ---------------------------- |
| `"equiangular"`    | `EquiangularGrid`   | `nlat`,`nlon` | equispaced in colatitude                                   | yes   | yes                          |
| `"legendre-gauss"` | `LegendreGaussGrid` | `nlat`,`nlon` | Gauss-Legendre nodes                                       | no    | yes                          |
| `"lobatto"`        | `LobattoGrid`       | `nlat`,`nlon` | Gauss-Lobatto nodes                                        | yes   | yes                          |
| `"trapezoidal"`    | `TrapezoidalGrid`   | `nlat`,`nlon` | equispaced in `cos(colatitude)`                            | yes   | only at low truncation       |
| `"healpix"`        | `HealpixGrid`       | `nside`       | `12 * nside**2` equal-area pixels on `4 * nside - 1` rings | no    | no                           |

The first four are latitude-longitude grids: `nlat` latitude rings with `nlon`
equispaced longitudes each. They differ only in where the rings sit and in the
quadrature rule that goes with them. All layers accept them.

**Equiangular.** Rings equally spaced in colatitude, including both poles, with
Clenshaw-Curtis quadrature. This is the layout of most reanalysis and climate
data, for example ERA5 at 0.25 degrees (`nlat=721, nlon=1440`). Its default SHT
truncation is `lmax = (nlat + 1) // 2`.

**Legendre-Gauss.** Rings at the Gauss-Legendre nodes, which exclude the poles.
The quadrature is exact up to degree `2 * nlat - 1`, the best achievable with
`nlat` rings, so it supports the highest truncation, `lmax = nlat`. This is the
regular Gaussian grid used by spectral weather models.

**Lobatto.** Rings at the Gauss-Lobatto nodes, which include both poles. Exact up
to degree `2 * nlat - 3`, with default `lmax = nlat - 1`. The rings cluster
toward the equator, so the polar spacing is coarser than on an equiangular grid
with the same `nlat`.

**Trapezoidal.** Rings equally spaced in `cos(colatitude)`, which cuts the
sphere into bands of equal area, with the trapezoidal rule. The rule converges
only algebraically, so an SHT on this grid is accurate only at a truncation far
below the default; pass a small `lmax` explicitly. It works well for quadrature
and the localized operators. This grid was called `"equiangular-trapezoidal"`
before v1.0.0.

**HEALPix.** The HEALPix pixelization in RING order: equal-area pixels on rings
whose length grows from 4 at the poles to `4 * nside` at the equator. Its
quadrature integrates only constants exactly, so it does not support an SHT.
It is accepted by `AttentionS2`, `NeighborhoodAttentionS2`,
`DiscreteContinuousConvS2`, `DiscreteContinuousConvTransposeS2` and
`QuadratureS2`; the SHTs, spectral convolutions, resampling, random fields and
the distributed layers require a latitude-longitude grid. `HealpixGrid.from_level(k)`
builds the grid with `nside = 2**k`.

## Field shapes

On a latitude-longitude grid a field has shape `(..., nlat, nlon)`. A HEALPix
field cannot be a rectangle, so it is stored flat, ring after ring, with shape
`(..., npix)`. `grid.shape` gives the trailing shape in both cases:

```python
ll = th.as_grid("equiangular", nlat=33, nlon=64)
hp = th.HealpixGrid(nside=16)

ll.shape                  # (33, 64)
hp.shape                  # (3072,)
hp.nlon_per_lat           # pixels per ring: 4, 8, 12, ..., 64, ..., 8, 4
hp.lon_offsets            # flat index of the first pixel of each ring
```

To write code that works on any grid, use `grid.shape` as a whole, e.g.
`x.reshape(*batch, *grid.shape)` or `len(grid.shape)`, rather than unpacking it
into `nlat, nlon`.

Rings are always ordered from the north pole southward. `grid.colats` gives the
ring colatitudes in radians and `grid.lats` the geographic latitudes. Two kinds
of quadrature weights are available, and they are not interchangeable:
`grid.quad_weights` has one entry per point and sums to `4*pi`, while
`grid.colat_weights` has one entry per ring and sums to 2.

## Mapping between grids

Layers that map one grid onto another take `grid_in` and `grid_out`, which do not
have to be of the same type. For example, attention can encode on HEALPix and
decode onto a latitude-longitude grid:

```python
hp = th.HealpixGrid(nside=16)
ll = th.as_grid("equiangular", nlat=33, nlon=64)

attn = th.NeighborhoodAttentionS2(grid_in=hp, grid_out=hp, in_channels=32, num_heads=4)
decode = th.AttentionS2(grid_in=hp, grid_out=ll, in_channels=32, num_heads=4)
```

The distributed layers take the *global* grid; each rank derives its own part
of it.

## Migrating from earlier versions

Before v1.0.0, layers took a resolution and a grid name. They now take a
descriptor, usually as the first argument, and calling them the old way raises a
`TypeError` that shows the replacement.

| before v1.0.0                                                                                         | now                                                                      |
| ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `RealSHT(nlat, nlon, grid="legendre-gauss")`                                                          | `RealSHT(th.as_grid("legendre-gauss", nlat=nlat, nlon=nlon))`            |
| `QuadratureS2((nlat, nlon), grid="equiangular")`                                                      | `QuadratureS2(th.as_grid("equiangular", nlat=nlat, nlon=nlon))`          |
| `ResampleS2(nlat_in, nlon_in, nlat_out, nlon_out, grid_in=..., grid_out=...)`                         | `ResampleS2(grid_in, grid_out)`                                          |
| `DiscreteContinuousConvS2(c_in, c_out, in_shape, out_shape, kernel_shape, grid_in=..., grid_out=...)` | `DiscreteContinuousConvS2(grid_in, grid_out, c_in, c_out, kernel_shape)` |
| `NeighborhoodAttentionS2(c_in, in_shape, out_shape, grid_in=..., grid_out=...)`                       | `NeighborhoodAttentionS2(grid_in, grid_out, c_in)`                       |
| `GaussianRandomFieldS2(nlat, grid=...)` (with `nlon = 2 * nlat`)                                      | `GaussianRandomFieldS2(th.as_grid(..., nlat=nlat, nlon=2 * nlat))`       |

The same pattern applies to the other layers and to their distributed versions.
Other changes to look out for:

- The grid name `"equiangular-trapezoidal"` is now `"trapezoidal"`.
- The default `theta_cutoff` of the DISCO convolutions and neighborhood attention
  is now one node spacing of the grid, `grid.max_node_spacing`, rather than
  `pi / (nlat - 1)`. The two agree on an equiangular grid with
  `nlon = 2 * (nlat - 1)` or more; elsewhere a warning tells you the default
  changed. Pass `theta_cutoff` explicitly to keep the old value, or use
  `th.truncate_support(grid)` to see the new one.

## Storing a grid in a config

A grid converts to plain data and back, so it can be stored in a config file or
a checkpoint:

```python
from torch_harmonics.grid import PointSetS2

grid = th.as_grid("legendre-gauss", nlat=64, nlon=128)
grid.to_dict()                          # {'grid': 'legendre-gauss', 'nlat': 64, 'nlon': 128}
PointSetS2.from_dict(grid.to_dict())    # LegendreGaussGrid(nlat=64, nlon=128)

th.HealpixGrid(nside=8).to_dict()       # {'grid': 'healpix', 'nside': 8}
```

Grids can also be pickled. Two grids are equal, and hash equally, when they
have the same type and parameters.

See {ref}`grids` in the API reference for every property and method.
