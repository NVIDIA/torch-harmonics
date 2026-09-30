# Changelog

## Versioning

### v1.0.0rc1 (unreleased)

* Faster DISCO CUDA kernels on H100 and GB200: kpacked forward up to 1.28x, backward up to 2.1x.
* Smaller DISCO psi memory footprint through a blocked-CSR layout instead of padding every row to the maximum.
* Fixed a DISCO backward launch failure for `nlon_in > 2048` with an integer scale factor of 3 or more.
* **Breaking**: `DistributedDiscreteContinuousConvS2` gains `polar_mode`, default `"halo-exchange"`, which computes only locally owned rows; `"reduce-scatter"` restores the old behaviour and is needed when the support reaches past the neighbouring rank.
* Fixed single-longitude inputs producing non-finite values and gradients in `ResampleS2` and `DistributedResampleS2`.
* Fixed Gaussian random-field sampling keeping stale dtype or device buffers after module conversions, including a parent module's.
* Fixed `DiceLossS2` counting ignored pixels in the class-zero denominator.
* Added tests for the example losses and metrics.
* Added `polar_halo_reduce`, the adjoint of `polar_halo_exchange`, and exported `compute_polar_halo_radius`.
* `polar_halo_exchange` and `polar_halo_reduce` take a `lat_dim` argument, so channels-last tensors can be exchanged directly.
* The sparsity patterns of DISCO and neighborhood attention are built only on the latitude band within `theta_cutoff`, making construction about 25x faster at nlat=256.
* **Breaking**: the distributed SHTs contract as a distributed matmul plus reduce-scatter, partitioning the Legendre coefficients over the whole process grid; results now differ from the serial transform by a few float32 ULP.
* The Legendre recurrences stream over degree, so a distributed rank builds only the block it stores and peak construction memory no longer grows as `O(nlat^3)`.
* Fixed the Legendre tables being rebuilt by every layer instead of being reused from the cache.
* Fixed `use_fp32` in the distributed reductions demoting float64 inputs to float32.
* Added `torch.compile(fullgraph=True)` support to the serial and distributed layers.
* The forward SHTs fold the `2*pi` longitudinal factor into their quadrature weights instead of scaling the FFT output on every call.
* `SpectralConvS2` and `DistributedSpectralConvS2` apply the spectral bias on the real view of their coefficients, removing the last complex pointwise ops.
* Fixed `torch.compile` of the SHT layers failing in inductor with `KeyError: 'complex64'`.
* Removed a redundant autocast decorator from the distributed autograd Functions that blocked full-graph compilation.
* Distributed DISCO convolution skips its polar collectives when the polar group holds a single rank.
* **Breaking**: layers take a grid descriptor instead of a resolution and a grid name, e.g. `RealSHT(as_grid("legendre-gauss", nlat=n, nlon=2*n))`, and `grid_in`/`grid_out` when mapping between grids; the old arguments raise a `TypeError`.
* **Breaking**: renamed or moved since v0.9.2: `lats_in`/`lats_out` are now `colats_in`/`colats_out` (`ResampleS2`, `DistributedResampleS2`, `latitude_support_band`, `compute_polar_halo_radius`); `QuadratureS2` moved from `torch_harmonics.quadrature` to `torch_harmonics.integration`; `NeighborhoodAttentionS2.quad_weights` became the backend buffer `ring_weights`; `AttentionS2.log_quad_weights` is now `log_point_weights`; the shallow-water example's `quad_weights` buffer is per-point and no longer checkpointed.
* New `as_grid` builds a grid descriptor from a grid type and its parameters by keyword, e.g. `as_grid("equiangular", nlat=128, nlon=256)`, validated against the type; `grid_params` lists what a type takes.
* `as_grid(descriptor, **params)` accepts only the descriptor type's own parameters and raises on anything else.
* Grid descriptors form a hierarchy of `PointSetS2` (points and weights), `GridS2` (isolatitude rings) and `RegularGridS2` (rings of equal length), with matching `GridShardS2`/`RegularGridShardS2`.
* New `require_point_set`, `require_grid` and `require_regular_grid` guards state which of these levels each routine needs, so an unsupported grid fails at construction.
* New `HealpixGrid(nside)` (also `as_grid("healpix", nside=...)`), the HEALPix pixelization in RING order, accepted today by `QuadratureS2`, `AttentionS2` and `NeighborhoodAttentionS2`.
* New `HealpixGrid.from_level(level)` builds the grid at refinement level `level`, i.e. `nside = 2**level`.
* New `PointSetS2.is_equal_area`, `True` for HEALPix and `False` for every latitude-longitude grid.
* New `PointSetS2.coords` gives every point's `(colat, lon)` in the order a field is stored in.
* New `GridS2.ring_weights(dtype)` and `GridS2.point_weights(dtype)` give the quadrature weights per ring and per point, computed in the requested dtype.
* `plot_sphere` takes a `grid` argument that places the samples at the grid's actual latitudes, and draws ragged grids such as HEALPix by nearest-point resampling.
* The example losses' `get_quadrature_weights(tile=False)` raises on grids that are not regular.
* `NeighborhoodAttentionS2` accepts any `GridS2` on either side, including HEALPix and mixed pairs, with new compiled CPU and CUDA kernels for ragged grids.
* `AttentionS2` accepts any `GridS2` on either side, no longer requires `nlon_in` to be a multiple of `nlon_out`, and projects channels-last like `NeighborhoodAttentionS2`.
* `AttentionS2` passes no weight mask on equal-area input grids, which is exact and lets SDPA use FlashAttention.
* New `torch_harmonics.neighborhood` module computing the neighborhood pattern of any `GridS2` directly as contiguous longitude arcs, replacing the DISCO-based precompute neighborhood attention used before.
* Neighborhood attention picks its implementation through backends selected per device, and each layer registers only the buffers its backend reads.
* `DistributedNeighborhoodAttentionS2` shares the serial forward pass and uses ring backends whose kernels take the serial layout and arc form, walking only the neighbours in each key/value chunk.
* `DistributedNeighborhoodAttentionS2` builds only its rank's slice of the sparsity pattern and reduces key/value gradients with a ring reduce-scatter, so its memory shrinks as ranks are added.
* The distributed neighborhood attention derives its latitude halo from the grid geometry and raises when it would exceed a local chunk.
* `DistributedNeighborhoodAttentionS2` raises on `optimized_kernel=False` instead of ignoring it, since it has no reference implementation.
* Fixed attention CUDA kernel launches failing when a large channel count needs more than 48 KiB of shared memory.
* Fixed the attention CUDA ops running on the current device rather than their inputs' when a module lives on another GPU.
* The compiled attention operators changed: `forward`/`backward` became `forward_regular`/`backward_regular`, `forward_ragged`/`backward_ragged` were added, the ring operators take arcs, and `split_csr_rows` was removed.
* Attention's layout conversions use the dedicated 4-D helpers, fixing a stride mismatch with the fake kernel.
* Grid descriptors give ring colatitudes as `colats` and geographic latitudes as `lats`.
* Grid descriptors carry the quadrature per ring as `colat_weights` (summing to 2) and per point as `quad_weights` (summing to `4*pi`).
* **Breaking**: the default `theta_cutoff` of DISCO and neighborhood attention is one node spacing (`PointSetS2.max_node_spacing`) instead of a function of `nlat`, changing it on every grid but equiangular with `nlon = 2*nlat`; neighborhood attention takes it from the grid with fewer points.
* New `truncate_support` returns the default support radius of DISCO and neighborhood attention from the grid descriptor, with override, validation and a warning when the default changed.
* `QuadratureS2` takes any `PointSetS2` and reads its per-point weights, which is also correct on ragged grids.
* Grid resolutions may be any integer type, e.g. `numpy.int64`.
* **Breaking**: the example losses and metrics take a grid descriptor instead of `(nlat, nlon, grid)`.
* **Breaking**: the example models, solvers and `PdeDataset` take grid descriptors: `grid` replaces `img_size`/`dims` plus a grid name, and `grid_internal` (or `grids_internal`, one per stage, for `SphericalUNet` and `SphericalSegformer`) replaces `scale_factor`.
* Fixed `SphericalUNet` convolving its input and output stages on the internal grid type instead of the grid the data lives on.
* The shallow-water solver skips transforms whose results it discards, and its time step is a `forward` that `ShallowWaterSolver.compile()` or `PdeDataset(compile=True)` can compile.
* `Stanford2D3DSDownloader` downloads archives concurrently (`max_workers`), in larger chunks (`chunk_size`), and hashes them while downloading.
* **Breaking**: the example models default to the `"harmonic"` filter basis, the L2-normalized form of the deprecated `"morlet"`, which changes their results; pass `filter_basis_type="morlet"` for the previous behaviour.
* **Breaking**: the `"equiangular-trapezoidal"` grid is renamed `"trapezoidal"` (class `TrapezoidalGrid`), since its nodes are equispaced in `cos(theta)`.
* **Breaking**: `GaussianRandomFieldS2` no longer assumes `nlon = 2 * nlat`.
* **Breaking**: fixed the `"bilinear-spherical"` resampling mode applying vector interpolation weights to scalars; it now interpolates along the shorter arc, changing only fields with a phase wrap.
* Fixed pole expansion in `"bilinear-spherical"` resampling for fields crossing the branch cut.
* New `RegularGridS2.shard()` describes one rank's piece of a grid, and the distributed layers take their decomposition from it; `compute_split_shapes` moved to `torch_harmonics.partition`.
* `DistributedQuadratureS2` builds only its rank's weights and now accepts the `"trapezoidal"` grid.
* The SHT and quadrature layers read nodes and weights from the grid descriptor instead of dispatching on the grid string, with bit-identical results.
* The SHTs warn on a `"trapezoidal"` grid, which does not round-trip, instead of raising.
* Fixed the caching decorator hiding the docstrings and signatures of cached routines.
* Fixed `trapezoidal_weights` returning float32 weights alongside float64 nodes.
* Fixed `AccuracyS2` counting ignored area as correctly classified; `IntersectionOverUnionS2` has no true-negative term and is unaffected.

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
