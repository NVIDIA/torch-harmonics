# Changelog

## Versioning

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