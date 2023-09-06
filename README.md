<!-- 
SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
 
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

<p align="center">
    <img src="https://raw.githubusercontent.com/NVIDIA/torch-harmonics/main/images/logo/logo.png"  width="568">
    <br>
    <a href="https://github.com/NVIDIA/torch-harmonics/actions/workflows/tests.yml"><img src="https://github.com/NVIDIA/torch-harmonics/actions/workflows/tests.yml/badge.svg"></a>
    <a href="https://pypi.org/project/torch_harmonics/"><img src="https://img.shields.io/pypi/v/torch_harmonics"></a>
</p>

<!--
[![pypi](https://img.shields.io/pypi/v/torch_harmonics)](https://pypi.org/project/torch_harmonics/)
-->

<!-- # torch-harmonics: differentiable harmonic transforms -->

<!-- ## What is torch-harmonics? -->

torch-harmonics is a differentiable implementation of the Spherical Harmonic transform in PyTorch. It was originally implemented to enable Spherical Fourier Neural Operators (SFNO). It uses quadrature rules to compute the projection onto the associated Legendre polynomials and FFTs for the projection onto the harmonic basis. This algorithm tends to outperform others with better asymptotic scaling for most practical purposes.

torch-harmonics uses PyTorch primitives to implement these operations, making it fully differentiable. Moreover, the quadrature can be distributed onto multiple ranks making it spatially distributed.

torch-harmonics has been used to implement a variety of differentiable PDE solvers which generated the animations below. Moreover, it has enabled the development of Spherical Fourier Neural Operators (SFNOs) [1].


<table border="0" cellspacing="0" cellpadding="0">
    <tr>
        <td><img src="https://media.githubusercontent.com/media/NVIDIA/torch-harmonics/main/images/sfno.gif"  width="240"></td>
        <td><img src="https://media.githubusercontent.com/media/NVIDIA/torch-harmonics/main/images/zonal_jet.gif"  width="240"></td>
        <td><img src="https://media.githubusercontent.com/media/NVIDIA/torch-harmonics/main/images/allen-cahn.gif"  width="240"></td>
    </tr>
<!--     <tr>
        <td style="text-align:center; border-style : hidden!important;">Shallow Water Eqns.</td>
        <td style="text-align:center; border-style : hidden!important;">Ginzburg-Landau Eqn.</td>
        <td style="text-align:center; border-style : hidden!important;">Allen-Cahn Eqn.</td>
    </tr>  -->
</table>


## Installation
Download directyly from PyPI:

```
pip install torch-harmonics
```

Build in your environment using the Python package:

```
git clone git@github.com:NVIDIA/torch-harmonics.git
cd torch-harmonics
pip install -e .
```

Alternatively, use the Dockerfile to build your custom container after cloning:

```
git clone git@github.com:NVIDIA/torch-harmonics.git
cd torch-harmonics
docker build . -t torch_harmonics
docker run --gpus all -it --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 torch_harmonics
```

## Contributors

 - [Boris Bonev](https://bonevbs.github.io) (bbonev@nvidia.com)
 - [Thorsten Kurth](https://github.com/azrael417) (tkurth@nvidia.com)
 - [Christian Hundt](https://github.com/gravitino) (chundt@nvidia.com)
 - [Nikola Kovachki](https://kovachki.github.io) (nkovachki@nvidia.com)
 - [Jean Kossaifi](http://jeankossaifi.com) (jkossaifi@nvidia.com)

## Implementation
The implementation follows the algorithm as presented in [2].

### Spherical harmonic transform

The truncated series expansion of a function $f$ defined on the surface of a sphere can be written as

$$
f(\theta, \lambda) = \sum_{m=-M}^{M} \exp(im\lambda) \sum_{n=|m|}^{M} F_n^m \bar{P}_n^m (\cos \theta),
$$

where $\theta$ is the colatitude, $\lambda$ the longitude, $\bar{P}_n^m$ the normalized, associated Legendre polynomials and $F_n^m$, the expansion coefficient associated to the mode $(m,n)$.

A direct spherical harmonic transform can be accomplished by a Fourier transform

$$
F^m(\theta) = \frac{1}{2 \pi} \int_{0}^{2\pi} f(\theta, \lambda) \exp(-im\lambda)  \mathrm{d}\lambda
$$

in longitude and a Legendre transform

$$
F_n^m = \frac{1}{2} \int_{-1}^1 F^m(\theta) \bar{P}_n^m(\cos \theta)  \mathrm{d} \cos \theta
$$

in latitude.

### Discrete Legendre transform

in order to apply the Legendre transfor, we shall use Gauss-Legendre points in the latitudinal direction. The integral

$$
F_n^m = \int_{0}^\pi F^m(\theta) \bar{P}_n^m(\cos \theta) \sin \theta \mathrm{d} \theta
$$

is approximated by the sum

$$
F_n^m = \sum_{j=1}^{N_\theta} F^m(\theta_j) \bar{P}_n^m(\cos \theta_j) w_j
$$

## Usage

### Getting started

The main functionality of `torch_harmonics` is provided in the form of `torch.nn.Modules` for composability. A minimum example is given by:

```python
import torch
import torch_harmonics as th

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nlat = 512
nlon = 2*nlat
batch_size = 32
signal = torch.randn(batch_size, nlat, nlon)

# transform data on an equiangular grid
sht = th.RealSHT(nlat, nlon, grid="equiangular").to(device)

coeffs = sht(signal)
```

To enable scalable model-parallelism, `torch-harmonics` implements a distributed variant of the SHT located in `torch_harmonics.distributed`.

### Cite us

If you use `torch-harmonics` in an academic paper, please cite [1]

```
@misc{bonev2023spherical,
      title={Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere}, 
      author={Boris Bonev and Thorsten Kurth and Christian Hundt and Jaideep Pathak and Maximilian Baust and Karthik Kashinath and Anima Anandkumar},
      year={2023},
      eprint={2306.03838},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## References

<a id="1">[1]</a> 
Bonev B., Kurth T., Hundt C., Pathak, J., Baust M., Kashinath K., Anandkumar A.;
Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere;
arXiv 2306.0383, 2023.

<a id="1">[2]</a> 
Schaeffer N.;
Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations;
G3: Geochemistry, Geophysics, Geosystems, 2013.

<a id="1">[3]</a> 
Wang B., Wang L., Xie Z.;
Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids;
Adv Comput Math, 2018.
