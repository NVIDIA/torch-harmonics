# Ragged CPU attention kernels — not implemented

This directory is empty on purpose, and the emptiness is the point.

`kernels_cuda/ragged/` holds a forward and a backward kernel for neighbourhood
attention on a grid whose latitude rings differ in length — a HEALPix or reduced
Gaussian grid. There is no CPU counterpart, so on CPU a ragged grid falls back to
the reference implementation in
`torch_harmonics/attention/kernels_torch/attention_ragged_torch.py`.

That fallback is correct but slow, and the gap is easy to miss when the kernels
sit in one flat directory: nothing announces that the optimized path covers only
half the grid families. An empty directory next to a full one does.

## Why the regular kernels cannot be reused

The regular kernels exploit longitudinal translation invariance. Every ring has
the same length and starts at the same longitude, so one sparsity pattern serves
all `nlon` columns, shifted — which is what makes their psi `O(nlat_out * band)`
rather than `O(npoints_out * neighbours)`, and what `pscale = nlon_in / nlon_out`
in the kernel signature is for.

HEALPix has neither property: ring lengths vary by a factor of `4 * nside`
between the poles and the equator, and successive rings are staggered by half a
pixel. So the shift is not an optimization the ragged path happens to skip; it is
a thing that does not exist there. The ragged kernels take `npoints_in`,
`npoints_out`, and explicit `ring_base` / `ring_size` tables instead.

A CPU implementation would follow the CUDA ragged kernels rather than the CPU
regular ones.
