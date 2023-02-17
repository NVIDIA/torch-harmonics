<p align="center">
    <img src="./images/logo/logo.png"  width="568">
</p>

<!-- # torch-harmonics: differentiable harmonic transforms -->

<!-- ## What is torch-harmonics? -->

`torch_harmonics` is a differentiable implementation of the Spherical Harmonic transform in PyTorch. It uses quadrature to compute the projection onto the associated Legendre polynomials and FFTs for the projection onto the harmonic basis. This algorithm tends to outperform others with better asymptotic scaling for most practical purposes.



<table border="0" cellspacing="0" cellpadding="0">
    <tr>
        <td><img src="./images/zonal_jet.gif"  width="288"></td>
        <td><img src="./images/ginzburg-landau.gif"  width="288"></td>
        <td><img src="./images/allen-cahn.gif"  width="288"></td>
    </tr> 
    <tr>
        <td style="text-align:center; border-style : hidden!important;">Shallow Water Eqns.</td>
        <td style="text-align:center; border-style : hidden!important;">Ginzburg-Landau Eqn.</td>
        <td style="text-align:center; border-style : hidden!important;">Allen-Cahn Eqn.</td>
    </tr> 
</table>


<!-- <p align="left">
    <img src="./images/zonal_jet.gif"  width="288">
    <img src="./images/allen-cahn.gif"  width="288">
</p> -->

## Installation
Build in your environment using the Python package:

```
git clone git@github.com:NVIDIA/torch-harmonics.git
pip install ./torch_harmonics
```

Alternatively, use the Dockerfile to build your custom container after cloning:


```
git clone git@github.com:NVIDIA/torch-harmonics.git
cd torch_harmonics
docker build . -t torch_harmonics
docker run --gpus all -it --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 torch_harmonics
```

## Contributors

 - Boris Bonev (bbonev@nvidia.com)
 - Christian Hundt (chundt@nvidia.com)
 - Thorsten Kurth (tkurth@nvidia.com)

## Implementation
The implementation follows the paper "Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations", N. Schaeffer, G3: Geochemistry, Geophysics, Geosystems. 

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
import torch_harmonics as harmonics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nlat = 512
nlon = 2*nlat
batch_size = 32
signal = torch.randn(batch_size, nlat, nlon)

# transform data on an equiangular grid
sht = harmonics.RealSHT(nlat, nlon, grid="equiangular").to(device).float()

coeffs = sht(signal)
```
