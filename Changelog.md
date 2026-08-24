# Changelog

## Versioning

### v0.9.3b1 (unreleased)

* Unified the default `theta_cutoff` heuristic for DISCO convolutions and neighborhood attention into `torch_harmonics.quadrature.compute_theta_cutoff`, backed by the new `compute_latitude_spacing`. The cutoff was previously computed as `pi / (nlat - 1)` in six independent places; that expression is the exact latitudinal node spacing of an equiangular grid, but not of the others. Gauss-Lobatto nodes cluster towards the equator (polar spacing ~21% larger) and `"equiangular-trapezoidal"` nodes are equispaced in `cos(theta)` rather than in `theta` (polar spacing larger by a factor `2*sqrt(nlat-1)/pi`, so ~5x at `nlat = 65` and growing with resolution), so on both grids the old default under-covered the poles: at `nlat = 129`, 6 of 129 output latitudes on a lobatto grid and 30 of 129 on an equiangular-trapezoidal grid saw only the single latitude ring they sit on, silently degenerating the stencil there. The cutoff is now taken from the grid's actual node distribution.
* **Breaking**: the default `theta_cutoff` of `DiscreteContinuousConvS2`, `DiscreteContinuousConvTransposeS2`, `NeighborhoodAttentionS2`, and their distributed counterparts now depends on the grid's node distribution rather than on `nlat` alone. Equiangular grids, which are the default, are unaffected in practice — the value moves by ~1e-15 and the resulting `psi` has an identical sparsity pattern — but `"lobatto"` widens by ~21% and `"legendre-gauss"` narrows by ~2.3%, both independently of resolution. For `"equiangular-trapezoidal"` the widening is resolution-dependent, about `2*sqrt(nlat-1)/pi`: ~5x at `nlat = 65` but ~17x at `nlat = 721`. Since the number of nonzeros in `psi` grows with the square of the cutoff radius, DISCO and neighborhood attention on that grid become substantially more expensive in both time and memory at high resolution; models trained on any non-equiangular grid will see a different stencil unless `theta_cutoff` is passed explicitly, which also remains the way to ask for the old, pole-under-covering support. A `UserWarning` is emitted on the grids whose default changed, following the `truncate_sht` precedent from v0.9.0.
* **Breaking**: the spherical harmonic transforms and the quadrature layer now take a `GridS2` descriptor in place of a resolution and a grid name. `RealSHT(nlat, nlon, lmax=..., grid="legendre-gauss")` becomes `RealSHT(as_grid("legendre-gauss", (nlat, nlon)), lmax=...)`, and likewise for `InverseRealSHT`, `RealVectorSHT`, `InverseRealVectorSHT`, `QuadratureS2`, `SpectralConvS2`, `ResampleS2`, `DiscreteContinuousConvS2`, `DiscreteContinuousConvTransposeS2`, `AttentionS2`, `NeighborhoodAttentionS2`, `GaussianRandomFieldS2` and their distributed counterparts. Where a layer took a pair of shapes and a pair of grid names, it now takes `grid_in` and `grid_out` in the slot the shapes occupied. `SpectralConvS2(in_shape, out_shape, in_channels, out_channels, ..., grid_in=..., grid_out=...)` becomes `SpectralConvS2(grid_in, grid_out, in_channels, out_channels, ...)`. The descriptor carries the resolution, so `nlat`/`nlon`/`img_shape` are gone from those signatures; both remain available as attributes on the module. Passing the old arguments raises a `TypeError` naming the replacement rather than failing deep inside the constructor.
* **Breaking**: `GaussianRandomFieldS2` no longer assumes `nlon = 2 * nlat`. It takes a grid descriptor, so any longitudinal resolution is expressible; previously only the 2:1 aspect ratio was reachable.
* `QuadratureS2` moved from `torch_harmonics.quadrature` to the new `torch_harmonics.integration`, so that the quadrature *rules* stay below the grid descriptor and the *layer* sits above it. The documented path, `torch_harmonics.QuadratureS2`, is unchanged.
* `GridS2.shard()` returns a `GridShardS2` describing one rank's piece of a decomposed grid, so that how a grid decomposes is a property of the grid rather than an assumption repeated in each distributed layer. `compute_split_shapes` moved to `torch_harmonics.partition`, shared by the descriptors and the distributed primitives; `torch_harmonics.distributed.compute_split_shapes` still resolves.
* Replaced the per-class grid dispatch in the SHT and quadrature layers with lookups on the grid descriptor. The same `if grid == "legendre-gauss" / "lobatto" / "equiangular"` chain, followed by a hand-rolled `flip(arccos(...))`, appeared twelve times across `sht.py`, `distributed_sht.py`, `QuadratureS2`, `DistributedQuadratureS2` and two examples; all of them now read `grid.lats` / `grid.quad_weights`, or `precompute_latitudes` where importing the descriptor would be circular. Numerically inert: every quadrature and Legendre buffer, and the transform outputs, are bit-identical.
* `DistributedQuadratureS2` now accepts the `"equiangular-trapezoidal"` grid, which the serial `QuadratureS2` already supported.
* The SHT classes now emit a prominent `UserWarning` when constructed on an `"equiangular-trapezoidal"` grid, in place of the previous `ValueError("Unknown quadrature mode")`. That rule converges only algebraically, so the associated Legendre polynomials are not discretely orthogonal on it and the transform does not round-trip: measured relative error at `nlat = 64` is 2e-4 at `lmax = 2`, 2e-2 at `lmax = 8`, and 6.7e-1 at the default `lmax = 32`, where the error is of the same order as the signal. Only `lmax = 1` is exact. The grid remains appropriate for plain quadrature and for the localized operators, which make no orthogonality assumption.
* Fixed the copying `lru_cache` decorator in `torch_harmonics.cache` hiding the function it decorates. It did not apply `functools.wraps`, so every cached routine presented itself as `wrapper(*args, **kwargs)` with no docstring, and autodoc silently dropped the API documentation of `precompute_latitudes`, `precompute_longitudes` and the other cached helpers. The wrapper now preserves the wrapped function's metadata and forwards the `cache_info`, `cache_clear` and `cache_parameters` accessors of the underlying `functools.lru_cache`, which were previously unreachable.
* Fixed `trapezoidal_weights` returning float32 weights alongside float64 nodes, because the underlying `torch.ones(n)` inherited the default dtype. It was the only rule in `torch_harmonics.quadrature` doing so, and it capped the accuracy of everything derived from it at roughly 1e-7, including the latitude weights of the `"equiangular-trapezoidal"` grid. Weights are now float64 like the other rules, and the tolerances of the affected tests have been tightened to match the other grids.

### v0.9.2

* Added upsampling support to `DistributedNeighborhoodAttentionS2` (`nlon_out % nlon_in == 0`): new upsample (scatter) ring-step CUDA kernels for forward and backward, matching the serial upsample attention. K/V rotate around the azimuth ring while queries and the softmax state stay local; all three directions (self-attention, downsampling, upsampling) are now supported by the distributed layer.
* Consistent CUDA kernel launch error checking: all attention and DISCO CUDA host wrappers now call `C10_CUDA_KERNEL_LAUNCH_CHECK()` and explicitly include `<c10/cuda/CUDAException.h>` instead of relying on transitive includes.
* Added `benchmarks/` suite covering SHT, DISCO convolution (self and downsampling), and spherical attention (global, neighborhood self, neighborhood cross-resolution) across resolutions, dtypes, and channel counts. Entry point: `python benchmarks/run.py`. Supports baseline CSV comparison for regression tracking in performance PRs.
* Fused DISCO kernel for serial convolution: fuses the sparse psi contraction and weight multiplication into a single autograd region, avoiding the K-expanded intermediate activation in the graph and reducing memory footprint by a factor of K. Enabled via `fused=True` on `DiscreteContinuousConvS2`.
* Distributed DISCO convolution now uses `reduce_scatter` instead of `all_reduce` + `scatter` for the polar reduction, cutting communicated data volume in half.
* Added fused distributed DISCO convolution variant which reduces activation storage.
* Adding WGMMA (tensor core) support to DISCO forward kernels on H100 (SM90) architectures. The kernel will be selected automatically if shapes allow, no specific action from the user is required.
* Improved performance for CPU based attention kernels.
* Fixed autocast on CPU for attention and DISCO: the custom ops registered an autocast kernel only at the `AutocastCUDA` dispatch key, so under `torch.autocast("cpu", ...)` nothing reconciled the inputs. For attention this could hand the kernel an fp32 query alongside fp16/bf16 keys and values, tripping the kernel's dtype check (`v dtype (Half) must match q dtype (Float)`); for DISCO it silently meant CPU autocast had no effect at all. Both now register `AutocastCPU` alongside `AutocastCUDA`. Whether the mismatch surfaced depended on the PyTorch version, so it was invisible on newer builds.
* Fixed stride problems in SHT under torch.compile on CPU.
* Converted all Python assert statements to torch._check for better torch.compile friendliness. All asserts in cosntructors were converted into ValueErrors for streamlined and clear error handling. In C++ and CUDA compiled code, all dynamic asserts were changed to TORCH_CHECK calls.
* Improved performance of distributed attention kernels achieved by splitting the kernel into two different ones for dense and less dense rows. This happens behind the scenes and the distributed attention API is unchanged.
* Improved distributed tests: tests now only print on rank 0 and test states are broadcast to all ranks before being triggered, to ensure clean failure on all ranks in case of failing tests.
* Splitting logic in distributed SHT improved. Now the SHT splits all leading dims up to the spatial dims when performing the all-to-all. It also automatically applies padding if the split tensor dim is smaller than the number of ranks it is split across. Tests were added to cover these cases.
* Fixed `SpectralConvS2` and `DistributedSpectralConvS2` spectral bias handling when input and output channel counts differ.
* Fixed a problem with squeezing singleton dimensions in the `GaussianRandomFieldS2` noise tensor.
* Fixed `AttentionS2` to disable SDPA dropout in eval mode.
* Hardened the 2D3DS example dataset downloader: archives are verified against SHA-256 checksums and tar members which resolve outside the target directory are rejected. Also fixes HTTP error handling, the temporary file location and resuming interrupted downloads, and makes the `2d3ds` extra installable again.
* Pinned the CI build tooling and restricted workflow token permissions to read-only.
* **Breaking**: minimum supported Python version is now 3.10, since 3.9 has reached its end of life. No cp39 wheels are built anymore.
* **Breaking**: default `basis_norm_mode` for `DistributedDiscreteContinuousConvS2` and `DistributedDiscreteContinuousConvTransposeS2` changed from `"mean"` to `"nodal"` to match the serial `DiscreteContinuousConvS2` / `DiscreteContinuousConvTransposeS2` defaults. Distributed and serial DISCO layers now share the same default normalization unless explicitly overridden.

### v0.9.1

* Fourier-Bessel filter basis; Hann window basis with per-type init factors via `get_init_factors`
* Standardized L2 normalization on the unit disk (harmonic, Zernike, Fourier-Bessel); on a disk of radius R the norm equals R via the Jacobian
* New DISCO basis normalization modes `modal` (mean-subtracted, reduces spectral leakage) and `geometric` (spherical cap area measure)
* Deprecated `basis_norm_mode="individual"` → `"nodal"` and `"area ratio"` → `"geometric"` (old names emit `DeprecationWarning`)
* Faster DISCO sparsity-pattern setup; OpenMP forward/backward kernels with up to ~55x speedup in some configurations
* Cross-attention (`key != value != query`) in `AttentionS2`, `NeighborhoodAttentionS2`, and `DistributedNeighborhoodAttentionS2`
* Serial attention upsampling when `nlon_out % nlon_in == 0`: CPU/CUDA/torch upsample kernels and matching reference
* `DistributedNeighborhoodAttentionS2` for self-attention and downsampling (distributed upsample not yet implemented)
* Optional per-head QK RMS norm (`use_qknorm`) for `AttentionS2` and `NeighborhoodAttentionS2`; shape checks across attention layers
* Fixed Q/K/V projection gain when input dim != embedding dim
* **Breaking**: default `NeighborhoodAttentionS2` scale changed from `1/sqrt(k_channels)` to `1/sqrt(k_channels // num_heads)` to match standard MHA head-dim scaling (`num_heads > 1`)
* Faster Legendre coefficient precomputation for SHT layers
* Differentiable `polar_halo_exchange` and `get_group_neighbors` for distributed attention
* More robust distributed transpose; `_reduce` clones before `all_reduce` for `torch.compile` compatibility
* Fixed Galewsky initial condition NaN from overflow; convolution adapter for mismatched residual channel counts
* Midpoint rule for filter-basis L2 norm integration (O(h^2)); improved `_precompute_convolution_tensor_s2` docstring
* Expanded attention tests (including upsample); new `tests/test_filter_basis.py`; broader layer integrity coverage

### v0.9.0

* New CPU backend (OpenMP-accelerated) for both DISCO convolution and attention layers
* Pre-compiled manylinux wheels for multiple PyTorch and CUDA versions, available on PyPI and pypi.nvidia.com
* Revised truncation logic for the SHT: centralized in new `truncation.py` module, enforcing triangular truncation (`lmax = min(lmax, mmax)`) across all SHT classes. Note: truncation for equiangular/equiangular-trapezoidal grids changed from `nlat` to `(nlat+1)//2`
* SHT performance improvements: contraction dimensions are now transposed to be stride-1 before einsum, and real/imaginary parts are split into separate contiguous tensors
* New `fft.py` wrapper module with proper Hermitian symmetry enforcement in `irfft` and explicit mode truncation in `rfft`
* Full PyTorch 2 custom operator compatibility for DISCO and attention layers using `torch.library` registration, enabling `torch.compile` and `torch.export`
* Restructured DISCO convolution and attention code into proper subpackages (`torch_harmonics/disco/`, `torch_harmonics/attention/`)
* Added double precision support for DISCO convolution
* Fixed Schmidt normalization for derivatives of associated Legendre polynomials
* Fixed up/downsampling in attention layers when input and output shapes differ
* Fixed `GaussianRandomFieldS2` to use `isht.lmax`/`isht.mmax` for compatibility with revised truncation logic
* Distributed module: added shape verification for transpose and gather operations, controllable via `TORCH_HARMONICS_DISTRIBUTED_DEBUG`
* Distributed module: fixed `finalize()` bug where process group was not properly destroyed
* Query functions `torch_harmonics.disco.optimized_kernels_is_available` and `torch_harmonics.attention.optimized_kernels_is_available` for checking optimized layer availability
* Quadrature helper functions `precompute_latitudes` and `precompute_longitudes` are now public API
* added new tests:
    * Comprehensive SHT test suite now covering vector SHT, Schmidt normalization, batch dimensions, and multiple grid types
    * New test suites for `SpectralConvS2`, `QuadratureS2`, `GaussianRandomFieldS2`, and `ResampleS2`
     Enhanced DISCO convolution tests covering different input/output channel counts and double precision
    * Enhanced attention tests with up/downsampling and `opcheck` integration
    * New distributed tests for primitives, quadrature, and spectral convolution
    * Shared test utilities module (`testutils.py`)

### v0.8.2

* Adding Driscoll-Healy (spectral) convolutions
* Adding QuadratureS2 method which allows to integrate a spherical field over one of the supported grids
* Adding tests for QuadratureS2 and Driscoll-Healy spectral convolutions
* Improving setup for distributed tests, refactoring and code re-use for distributed and serial tests
* Decreasing problem sizes for some tests, allowing for faster execution
* Adding an additional caching test based of contents of a torch tensor
* DistributedRealVectorSHT now does truncation correctly, previously this was not guaranteed
* Double precision support for DISCO convolution

### v0.8.1

* Revised the truncation logic for the SHT
* Restructuring torch-harmonics module to streamline usage of the new attention and DISCO layers
* Full PyTorch 2 custom operator compatibility, allowing for exporting models with torch-harmonics layers using torch.export
* Added OpenMP accelerated CPU backend for DISCO and attention layers
* New query functions `torch_harmonics.disco.optimized_kernels_is_available` and `torch_harmonics.attention.optimized_kernels_is_available` for optimized layers availability
* More tests for DISCO and attention layers
* Cleaned up notebooks

### v0.8.0

* Adding spherical attention and spherical neighborhood attention
* Custom CUDA kerneles for spherical neighborhood attention
* New datasets for segmentation and depth estimation on the sphere based on the 2D3DS dataset
* added new spherical architectures and corresponding baselines
    * S2 Transformer
    * S2 Segformer
    * S2 U-Net
* Reorganized examples folder, including new examples based on the 2d3ds dataset
* Added spherical loss functions to examples
* Added plotting module
* Updated docstrings

### v0.7.6

* Adding cache for precomoputed tensors such as weight tensors for DISCO and SHT
* Cache is returning copies of tensors and not references. Users are still encouraged to re-use
  those tensors manually in their models because this will also save memory. However,
  the cache will help with model setup speed.
* Adding test which ensures that cache is working correctly

### v0.7.5

* New normalization mode `support` for DISCO convolutions
* More efficient computation of Morlet filter basis
* Changed default for Morlet filter basis to a Hann window function

### v0.7.4

* New filter basis normalization in DISCO convolutions
* More robust pre-computation of DISCO convolution tensor
* Reworked DISCO filter basis datastructure
* Support for new filter basis types
* Added Zernike polynomial basis on a disk
* Added Morlet wavelet basis functions on a spherical disk
* Cleaning up the SFNO example and adding new Local Spherical Neural Operator model
* Updated resampling module to extend input signal to the poles if needed
* Added slerp interpolation to the resampling module
* Added distributed resampling module

### v0.7.3

* Changing default grid in all SHT routines to `equiangular`
* Hotfix to the numpy version requirements

### v0.7.2

* Added resampling modules for convenience
* Changing behavior of distributed SHT to use `dim=-3` as channel dimension
* Fixing SHT unittests to test SHT and ISHT individually, rather than the roundtrip
* Changing the way custom CUDA extensions are handled

### v0.7.1

* Hotfix to AMP in SFNO example

### v0.7.0

* CUDA-accelerated DISCO convolutions
* Updated DISCO convolutions to support even number of collocation points across the diameter
* Distributed DISCO convolutions
* Fused quadrature into multiplication with the Psi tensor to lower memory footprint
* Removed DISCO convolution in the plane to focus on the sphere
* Updated unit tests which now include tests for the distributed convolutions

### v0.6.5

* Discrete-continuous (DISCO) convolutions on the sphere and in two dimensions
* DISCO supports isotropic and anisotropic kernel functions parameterized as hat functions
* Supports regular and transpose convolutions
* Accelerated spherical DISCO convolutions on GPU via Triton implementation
* Unittests for DISCO convolutions in `tests/test_convolution.py`

### v0.6.4

* Reworking distributed to allow for uneven split tensors, effectively removing the necessity of padding the transformed tensors
* Distributed SHT tests are now using unittest. Test extended to vector SHT versions
* Tests are defined in `torch_harmonics/distributed/distributed_tests.py`
* Base pytorch container version bumped up to 23.11 in Dockerfile

### v0.6.3

* Adding gradient check in unit tests
* Temporary work-around for NCCL contiguous issues with distributed SHT
* Refactored examples and documentation
* Updated SFNO example

### v0.6.2

* Adding github CI
* Changed SHT modules to convert dtype dynamically when computing the SHT/ISHT
* Bugfixes to fix importing examples

### v0.6.1

* Minor bugfixes to export SFNO code
* Readme should now render correctly in PyPI

### v0.6.0

* Added SFNO example
* Added Shallow Water Equations Dataset for SFNO training
* Cleanup of the repository and added PyPI
* Updated Readme

### v0.5.0

* Reworked distributed SHT
* Module for sampling Gaussian Random Fields on the sphere

### v0.4.0

* Computation of associated Legendre polynomials
    * changed algorithm to compute the associated Legendre polynomials for improved stability
* Improved Readme

### v0.3.0

* Vector Spherical Harmonic Transforms
    * projects vector-valued fields onto the vector Spherical Harmonics
    * supports computation of div and curl on the sphere
* New quadrature rules
    * Clenshaw-Curtis quadrature rule
    * Fejér quadrature rule
    * Legendre-Gauss-Lobatto quadrature
* New notebooks
    * complete with differentiable Shallow Water Solver
    * notebook on quadrature and interpolation
* Unit tests
* Refactor of the API

### v0.2.0

* Renaming from torch_sht to torch_harmonics
* Adding distributed SHT support
* New logo

### v0.1.0

* Single GPU forward and backward transform
* Minimal code example and notebook
