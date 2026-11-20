# Kernel Factory DISCO kernels (experimental)

Candidates from Kernel Factory campaigns on the DISCO contraction at the benchmark sizes
(721x1440 and 360x720), kept as the campaigns produced them so they can be compared against the shipped kernels
with `performance/disco/disco_ab.py`. They are not integrated: they build only with
`TORCH_HARMONICS_BUILD_KF_KERNELS=1`, are fp16 only, and run under the experimental
`regular-kf` backend (see `torch_harmonics/disco/optimized/disco_kf.py`).

| directory | operator | GPU | campaign | candidate | campaign speedup |
| --- | --- | --- | --- | --- | --- |
| `fwd_sm90` | forward (kpacked psi) | H100, sm_90a | disco-fwd-h100-v2 | `101252ed` | 7.14x |
| `fwd_sm100` | forward (kpacked psi) | B200, sm_100a | disco-fwd-b200-v2 | `91ed7617` | 18.8x |
| `bwd_sm90` | arc scatter | H100, sm_90a | disco-bwd-h100-v2 | `3a79600d` | 8.15x |
| `bwd_sm100` | arc scatter | B200, sm_100a | disco-bwd-b200-v2 | `aea62ac1` | 8.29x |

Speedups are Kernel Factory's geometric mean over its four workloads (encoder, aux encoder,
decoder, local conv, batch 1, fp16) against the shipped kernels, kernel time only.

All four were checked for dispatch on exact sizes (channel count, batch, nlat, nlon, ...),
which the campaigns forbid; they dispatch on the longitude ratio, divisibility and size
thresholds only.

Each directory holds the candidate's sources (`kernel.cu` and its includes), changed only by the
repository's license header and formatting, and a
`kf_entry.cu` that calls its `Run`. `shim/` maps the part of the TVM FFI API the sources use
onto ATen, and `kf_bind_fwd.cpp` / `kf_bind_bwd.cpp` bind the entry points to Python.
