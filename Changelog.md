# Changelog

## Versioning

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
