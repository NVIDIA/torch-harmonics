# Distributed spherical transforms

This tutorial explains how to set up and use the distributed (multi-GPU)
modules in `torch_harmonics.distributed`. All distributed operators partition
the sphere across a **2-D process grid** with a *polar* axis (latitudes) and an
*azimuthal* axis (longitudes). Each GPU owns a contiguous tile of the sphere
and the modules handle the necessary communication (all-to-all transposes, halo
exchanges, reductions) internally.

## 1. The process grid

The distributed backend organises ranks into two orthogonal
[process groups](https://pytorch.org/docs/stable/distributed.html#groups):

- **Polar group** — ranks that share the same azimuthal index. Together they
  hold all latitude chunks for a fixed set of longitudes.
- **Azimuth group** — ranks that share the same polar index. Together they
  hold all longitude chunks for a fixed set of latitudes.

For a grid with `num_polar` polar ranks and `num_azimuth` azimuth ranks the
total world size is `num_polar × num_azimuth`. The layout is row-major: global
rank `r` maps to polar index `r // num_azimuth` and azimuth index
`r % num_azimuth`.

```
          azimuth index
          0       1       2       3
       ┌───────┬───────┬───────┬───────┐
  p  0 │ GPU 0 │ GPU 1 │ GPU 2 │ GPU 3 │   ← azimuth group (row)
  o    ├───────┼───────┼───────┼───────┤
  l  1 │ GPU 4 │ GPU 5 │ GPU 6 │ GPU 7 │   ← azimuth group (row)
  a    └───────┴───────┴───────┴───────┘
  r        │               │
  i    polar group      polar group
  n    (column)         (column)
  d
  e
  x
```

## 2. Creating the communicator grid

After initialising PyTorch distributed as usual, build the two groups and pass
them to {func}`~torch_harmonics.distributed.init`:

```python
import torch
import torch.distributed as dist
import torch_harmonics.distributed as thd

dist.init_process_group(backend="nccl")
world_rank = dist.get_rank()
world_size = dist.get_world_size()

# choose the decomposition
num_polar = 2      # split latitudes across 2 ranks
num_azimuth = 4    # split longitudes across 4 ranks
assert num_polar * num_azimuth == world_size

# --- build orthogonal groups ---
# Ranks in the same row share a polar index → they form an azimuth group.
azimuth_groups, azimuth_group = None, None
for p in range(num_polar):
    ranks = list(range(p * num_azimuth, (p + 1) * num_azimuth))
    grp = dist.new_group(ranks=ranks)
    if world_rank in ranks:
        azimuth_group = grp

# Ranks in the same column share an azimuth index → they form a polar group.
polar_groups, polar_group = None, None
for a in range(num_azimuth):
    ranks = list(range(a, world_size, num_azimuth))
    grp = dist.new_group(ranks=ranks)
    if world_rank in ranks:
        polar_group = grp

# Register with torch-harmonics
thd.init(polar_group, azimuth_group)
```

From this point on every distributed module
(`DistributedRealSHT`, `DistributedSpectralConvS2`, etc.) will use these groups
for its internal communication.

```{note}
Both groups must be created on **all** ranks (even those that are not members),
because ``dist.new_group`` is a collective call.  Each rank only keeps the
handle it actually belongs to.
```

## 3. Splitting the data with `compute_split_shapes`

{func}`~torch_harmonics.distributed.compute_split_shapes` determines how a
dimension of length `N` is divided across `P` ranks. The split is balanced:
chunk sizes differ by at most one element.

```python
from torch_harmonics.distributed import compute_split_shapes

# 256 longitudes split across 4 azimuth ranks → perfectly even
compute_split_shapes(256, 4)
# [64, 64, 64, 64]

# 128 latitudes split across 3 polar ranks → one rank gets an extra row
compute_split_shapes(128, 3)
# [43, 43, 42]
```

Every distributed module calls this function internally to determine local
slice sizes, so the user normally does not need to call it directly. It is
useful however when preparing input data: each rank must hold only its local
tile.

## 4. Preparing local input data

Given a global signal of shape `(batch, channels, nlat, nlon)`, each rank
needs the sub-tensor that corresponds to its polar and azimuth indices:

```python
nlat, nlon = 512, 1024

# compute per-rank chunk sizes
lat_shapes = compute_split_shapes(nlat, num_polar)    # e.g. [256, 256]
lon_shapes = compute_split_shapes(nlon, num_azimuth)   # e.g. [256, 256, 256, 256]

# local sizes for this rank
polar_rank = thd.polar_group_rank()
azimuth_rank = thd.azimuth_group_rank()
nlat_local = lat_shapes[polar_rank]
nlon_local = lon_shapes[azimuth_rank]

# slice the global tensor (only for illustration — in practice each rank
# typically loads or generates only its own tile)
lat_offsets = [0] + list(torch.cumsum(torch.tensor(lat_shapes), 0).tolist())
lon_offsets = [0] + list(torch.cumsum(torch.tensor(lon_shapes), 0).tolist())

x_local = x_global[
    ...,
    lat_offsets[polar_rank] : lat_offsets[polar_rank + 1],
    lon_offsets[azimuth_rank] : lon_offsets[azimuth_rank + 1],
]
```

## 5. Calling distributed modules

Once the communicator grid is set up and each rank holds its local tile, the
distributed modules are drop-in replacements for their serial counterparts:

```python
import torch_harmonics.distributed as thd

# create the distributed forward / inverse SHT with global grid sizes
sht  = thd.DistributedRealSHT(nlat, nlon, grid="equiangular").cuda()
isht = thd.DistributedInverseRealSHT(nlat, nlon, grid="equiangular").cuda()

# each rank passes only its local tile
coeffs = sht(x_local)            # local spectral coefficients
x_recon = isht(coeffs)           # back to the local spatial tile
```

The modules handle all internal communication (distributed transposes,
reductions, halo exchanges) transparently. The local output shapes are
determined by the same splitting logic.

## 6. Clean-up

When done, tear down the torch-harmonics state and the PyTorch process group:

```python
thd.finalize()
dist.destroy_process_group()
```

## Complete example

Putting it all together as a script that can be launched with
`torchrun --nproc_per_node=8 distributed_sht.py`:

```python
"""Distributed forward + inverse SHT on 8 GPUs (2 polar × 4 azimuth)."""

import torch
import torch.distributed as dist
import torch_harmonics.distributed as thd
from torch_harmonics.distributed import compute_split_shapes

def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    world_rank = dist.get_rank()
    world_size = dist.get_world_size()

    # --- 1. Define the 2-D process grid ---
    num_polar, num_azimuth = 2, 4
    assert num_polar * num_azimuth == world_size

    azimuth_group = None
    for p in range(num_polar):
        ranks = list(range(p * num_azimuth, (p + 1) * num_azimuth))
        grp = dist.new_group(ranks=ranks)
        if world_rank in ranks:
            azimuth_group = grp

    polar_group = None
    for a in range(num_azimuth):
        ranks = list(range(a, world_size, num_azimuth))
        grp = dist.new_group(ranks=ranks)
        if world_rank in ranks:
            polar_group = grp

    thd.init(polar_group, azimuth_group)

    # --- 2. Prepare local data ---
    nlat, nlon = 512, 1024
    lat_shapes = compute_split_shapes(nlat, num_polar)
    lon_shapes = compute_split_shapes(nlon, num_azimuth)
    nlat_local = lat_shapes[thd.polar_group_rank()]
    nlon_local = lon_shapes[thd.azimuth_group_rank()]

    x_local = torch.randn(1, 1, nlat_local, nlon_local, device="cuda")

    # --- 3. Distributed SHT round-trip ---
    sht  = thd.DistributedRealSHT(nlat, nlon, grid="equiangular").cuda()
    isht = thd.DistributedInverseRealSHT(nlat, nlon, grid="equiangular").cuda()

    coeffs  = sht(x_local)
    x_recon = isht(coeffs)

    # --- 4. Check reconstruction error ---
    err = (x_local - x_recon).abs().max().item()
    if world_rank == 0:
        print(f"Max reconstruction error: {err:.2e}")

    # --- 5. Clean up ---
    thd.finalize()
    dist.destroy_process_group()

if __name__ == "__main__":
    import os
    main()
```
