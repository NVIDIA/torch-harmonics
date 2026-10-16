# Grids

Layers that operate on sampled fields take a *spatial grid descriptor*: a small,
immutable object that says where the sample points sit on the sphere and which
quadrature weights go with them. The SHTs also expose the spectral coefficient
domain at their input or output through a `SpectralGrid`.

The {doc}`grid descriptors tutorial <../tutorials/grid_descriptors>` walks
through building descriptors, what they know, and how layers use them.

## Building a grid

Build a grid by name with `as_grid`, passing the parameters that grid type takes
by keyword:

```python
import torch_harmonics as th

grid = th.as_grid("legendre-gauss", nlat=128, nlon=256)
sht = th.RealSHT(grid)

th.grid_types()                   # ('equiangular', 'fejer1', 'legendre-gauss', 'lobatto', 'trapezoidal', 'healpix')
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
| `"fejer1"`         | `Fejer1Grid`        | `nlat`,`nlon` | equispaced half a ring from each pole                      | no    | yes                          |
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

**Fejér-1.** Rings are equally spaced in colatitude at
`theta[j] = pi * (j + 0.5) / nlat`, so neither pole is sampled. Fejér's first
quadrature rule integrates polynomials through degree `nlat - 1` for even
`nlat` and `nlat` for odd `nlat`. Ordinary direct SHT analysis therefore uses
the exclusive native bound `grid.max_exact_degree = (nlat + 1) // 2` (T89 on a
180-ring grid).

`RealSHT`, `RealVectorSHT`, `DistributedRealSHT` and
`DistributedRealVectorSHT` support folded high-bandwidth analysis on both
equiangular and Fejér-1 grids when explicit `lmax` exceeds
`grid.max_exact_degree`. The construction interleaves the input rings with
complementary Clenshaw-Curtis nodes from a fine grid with `2*nlat + 1` rings for
F1, then folds the quadrature and interpolation into real Legendre projection
weights, following the doubled-meridian method of {cite}`Reinecke2023`.
For F1 the exclusive recovery limits are `lmax <= nlat` and
`mmax <= (nlon + 1) // 2`; for equiangular they remain `lmax <= nlat - 1`.
The order limit excludes the even-longitude Nyquist mode, which cannot recover
a complex coefficient. For example, `lmax=mmax=180` on a 180 by 360 F1 grid
represents degrees and orders through 179. Larger requested dimensions,
including an inferred `mmax` beyond the recovery limit, retain direct
quadrature without clamping `lmax`, `mmax` or `lmmax`; accepted dimensions do
not imply accurate recovery. Independent `lmax`, `mmax` and optional `lmmax`
domains are supported.

Writing the folded latitude operator as $U_m = Q_n + A_m^* Q_c A_m$, analysis
is $P_m^T U_m x_m = W_m^T x_m$ with $W_m = U_m^T P_m$. Here $Q_n$ and $Q_c$
are the fine Clenshaw-Curtis weights on native and complementary rings,
respectively; their roles reverse between CC and F1. Half-grid interpolation
suppresses the unpaired meridional Nyquist cosine in the complementary term.
This makes $U_m$ real and symmetric. Scalar continuation uses $(-1)^m$ and
tangential vector continuation uses $(-1)^{m+1}$. Both derivative components
use the same vector parity. Precomputation runs in float64 degree/order blocks;
modules retain only real, nonpersistent weights and cast them when requested.
A two-entry cache can retain up to two completed float64 projection tables in
host memory, in addition to the independent projection buffers owned by each
module.

This increases construction time while removing meridional FFTs from every
analysis call. Persistent storage remains one scalar projection table or two
vector projection tables. Distributed ranks generate their local global-order
range on the full meridian, then retain their latitude shard. Polar ranks repeat
that initialization work, but forward communication remains the existing two
azimuth redistributions and polar reduce-scatter. There is no latitude gather.

For even `nlon`, the inferred azimuthal bound can include the Nyquist order and
disable extended analysis for the whole transform. For example, `mmax` is
inferred as 33 below, including order 32, so the transform uses direct quadrature
even though `lmax=70` is within the latitude recovery limit. Set `mmax=32` to
exclude Nyquist and select extended analysis:

```python
grid = th.as_grid("equiangular", nlat=129, nlon=64)
direct = th.RealSHT(grid, lmax=70)  # inferred mmax=33 includes the Nyquist order
extended = th.RealSHT(grid, lmax=70, mmax=32)
```

```python
grid = th.as_grid("equiangular", nlat=73, nlon=144)
sht = th.RealSHT(grid, lmax=72, mmax=72)
isht = th.InverseRealSHT(grid, lmax=72, mmax=72)
# Degrees and orders through 71; sht.grid_out == th.SpectralGrid(72, 72).
```

The default remains degree/order limits of 37 on this equiangular grid. Its
`max_exact_degree` continues to describe the native direct quadrature limit.
Analysis within that limit uses the original direct path, and synthesis
continues to evaluate harmonics directly on the requested grid. The extended
analysis paths are intended for float32 and float64. Their constructor
precomputation uses a latitude FFT of length `2 * (nlat - 1)` for equiangular
and `2 * nlat` for F1.

For either grid, folded analysis uses more constructor work and projection
memory than native quadrature, while avoiding meridional resampling during each
forward call. At `nlat=180`, one float64 table is about 44.5 MiB for scalar
analysis and the two vector tables together are about 89 MiB; temporary
construction memory is block bounded. The cache may retain two projections.
Synthesis evaluates Legendre functions directly on the requested grid.

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

## Spatial and spectral domains

A spatial grid describes physical sample locations and quadrature weights. A
`SpectralGrid` describes the spherical harmonic modes an SHT retains. It is
independent of the spatial grid hierarchy and stores only the exclusive bounds
`lmax`, `mmax`, and optional `lmmax`. The bounds must satisfy
`0 <= mmax <= lmax`; exceeding `lmax` raises `ValueError`.

`SpectralGrid.shape` is the dense coefficient shape `(lmax, mmax)`. When
`lmmax` is set, it masks modes inside that rectangle using the exclusive bound
`l - m < lmmax`; it does not change the storage shape. For example,
`SpectralGrid(lmax=85, mmax=43, lmmax=43)` is R42 and retains modes satisfying
`0 <= m < 43`, `m <= l < 85`, and `l - m < 43`.

Forward SHTs map from a spatial grid to a spectral grid. Inverse SHTs map back
to the spatial grid. The `.grid` attribute is the spatial descriptor in both
directions; `grid_in` and `grid_out` identify the
actual input and output domains:

For a `SphericalProductGrid`, these descriptors describe the angular domains.
The radial shells remain a separate axis in the field and are available on the
layer through `.radial`.

```python
grid = th.as_grid("legendre-gauss", nlat=128, nlon=256)
sht = th.RealSHT(grid, lmax=85, mmax=43, lmmax=43)
isht = th.InverseRealSHT(grid, lmax=85, mmax=43, lmmax=43)

assert sht.grid_out == isht.grid_in
assert sht.grid_in == isht.grid_out
```

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
