# Changelog

## Versioning

### v0.9.2b

* Fused DISCO kernel for serial convolution: fuses the sparse psi contraction and weight multiplication into a single autograd region, avoiding the K-expanded intermediate activation in the graph and reducing memory footprint by a factor of K. Enabled via `fused=True` on `DiscreteContinuousConvS2`.
* Distributed DISCO convolution now uses `reduce_scatter` instead of `all_reduce` + `scatter` for the polar reduction, cutting communicated data volume in half.
* Added fused distributed DISCO convolution variant which reduces activation storage.
* Improved performance for CPU based attention kernels.
* Fixed stride problems in SHT under torch.compile on CPU.
* Converted all Python assert statements to torch._check for better torch.compile friendliness. All asserts in cosntructors were converted into ValueErrors for streamlined and clear error handling. In C++ and CUDA compiled code, all dynamic asserts were changed to TORCH_CHECK calls.
* Improved performance of distributed attention kernels achieved by splitting the kernel into two different ones for dense and less dense rows. This happens behind the scenes and the distributed attention API is unchanged.
* Improved distributed tests: tests now only print on rank 0 and test states are broadcast to all ranks before being triggered, to ensure clean failure on all ranks in case of failing tests.
* Splitting logic in distributed SHT improved. Now the SHT splits all leading dims up to the spatial dims when performing the all-to-all. It also automatically applies padding if the split tensor dim is smaller than the number of ranks it is split across. Tests were added to cover these cases.
* Fixed `SpectralConvS2` and `DistributedSpectralConvS2` spectral bias handling when input and output channel counts differ.
* Fixed a problem with squeezing singleton dimensions in the `GaussianRandomFieldS2` noise tensor.
* Fixed `AttentionS2` to disable SDPA dropout in eval mode.
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
