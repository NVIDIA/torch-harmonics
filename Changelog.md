# Changelog

## Versioning

### v0.9.1

* Added Fourier-Bessel basis functions
* New Hann filter basis; filter basis types can now specify their own initialization factors via `get_init_factors`
* Filter basis L2 normalization standardized on the unit disk for harmonic, Zernike, and Fourier-Bessel bases; on a disk of radius R the norm equals R via the Jacobian
* New `modal` filter basis normalization mode subtracts the mean to reduce spectral leakage
* New `geometric` mode uses the theoretical area measure of the spherical cap to normalize
* Deprecated `basis_norm_mode="individual"` (use `"nodal"`) and `basis_norm_mode="area ratio"` (use `"geometric"`); old names still work and emit `DeprecationWarning`
* Vectorized DISCO convolution tensor normalization: the per-(ikernel, ilat_out) Python double loop is replaced by a single `scatter_add_` per reduced quantity, materially speeding up module construction
* Specialized DISCO forward CUDA kernel for pscale = 1, 2, 3
* Vectorized routines for the associated Legendre polynomial computation, replacing the per-mode loop
* Refactored attention custom autograd to perform QKV projections outside the custom op, letting torch handle conv2d gradients natively
* **Breaking**: default attention scale in `NeighborhoodAttentionS2` changed from `1/sqrt(k_channels)` to `1/sqrt(k_channels // num_heads)` to match standard MHA head-dim scaling; affects users relying on the default with `num_heads > 1`
* Cross-attention (`key != value != query`) supported in `AttentionS2`, `NeighborhoodAttentionS2`, and `DistributedNeighborhoodAttentionS2`
* Attention upsampling (`nlon_out > nlon_in`) supported in serial `NeighborhoodAttentionS2`: new `s2_attn_fwd_upsample` / `s2_attn_bwd_upsample` kernels (CPU and CUDA) and matching torch reference
* Support for DistributedNeighborhoodAttentionS2. This layer uses a 2-stage kernel to compute the attention per spatial parallel rank and performs an online update using ring exchange. Neighboring points in latitude are gathered using halo exchange. Currently supports the gather/downsample direction.
* Added proper shape checks in all attention layers
* Optional QK normalization (`use_qknorm=True`) for `AttentionS2` and `NeighborhoodAttentionS2`, applying per-head RMS normalization to Q and K projections
* Fixed weight initialization in `AttentionS2` and `NeighborhoodAttentionS2`: Q/K/V projections now use correct gain factors when input dim != embedding dim
* DISCO and distributed neighborhood attention backwards now honor the autograd contract: per-input `ctx.needs_input_grad` branches return `None` and skip the corresponding kernel calls and NCCL allreduces, enabling AOTAutograd to prune dead subgraphs from the compiled backward
* New distributed primitives: differentiable `polar_halo_exchange` and `get_group_neighbors` to support distributed attention
* New ring-step CUDA kernels for distributed attention: forward (`s2_attn_fwd_ring_step`) and two-pass backward (`s2_attn_bwd_ring_step_pass1/2`)
* Improved robustness of distributed transpose and better `torch.compile` compatibility; `_reduce` now clones before `all_reduce` to avoid mutating its input in place
* PT2 compatibility tags and `opcheck` coverage added for the ring-step attention kernels
* Minor fixes and cleanups:
    * Fixed Galewsky initial condition NaN caused by overflowing values
    * Added convolution adapter for residual paths when input and output channel counts differ
    * Midpoint rule applied to the radial integral in basis L2-norm computation for O(h^2) convergence
    * Removed legacy `_f`-suffixed function names in favor of descriptive names
    * Improved docstring for `_precompute_convolution_tensor_s2`
* added new tests:
    * expanded attention tests cover cross-attention, QK normalization, downsampling, and serial upsampling; distributed attention tests cover the gather/downsample path
    * new `tests/test_filter_basis.py` suite for filter basis analytical properties: L2 normalization, orthogonality (where applicable), Dirichlet boundary for Fourier-Bessel, partition-of-unity for piecewise-linear, support and isotropy invariants
    * gradient-contract tests for DISCO (`test_no_input_grad`) and attention (`test_selective_requires_grad`, `test_distributed_neighborhood_attention_selective_requires_grad`) verify that frozen inputs yield `None` gradients

### v0.9.0

* New CPU backend (OpenMP-accelerated) for both DISCO convolution and attention layers
* Pre-compiled manylinux wheels for multiple PyTorch and CUDA versions, available on PyPI and pypi.nvidia.com
* Revised truncation logic for the SHT: centralized in new `truncation.py` module, enforcing triangular truncation (`lmax = min(lmax, mmax)`) across all SHT classes. Note: truncation for equiangular/equidistant grids changed from `nlat` to `(nlat+1)//2`
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
