# Distributed layers

Distributed (multi-GPU) counterparts of the serial layers. These are
available in the `torch_harmonics.distributed` subpackage.

## Coverage

The table below shows which serial layers have a distributed counterpart and
which do not.

| Serial layer                        | Distributed counterpart                        | Notes                                    |
| ----------------------------------- | ---------------------------------------------- | ---------------------------------------- |
| `RealSHT`                           | `DistributedRealSHT`                           | HEALPix, polar ranks only                |
| `InverseRealSHT`                    | `DistributedInverseRealSHT`                    | HEALPix, polar ranks only                |
| `RealVectorSHT`                     | `DistributedRealVectorSHT`                     | HEALPix, polar ranks only                |
| `InverseRealVectorSHT`              | `DistributedInverseRealVectorSHT`              | HEALPix, polar ranks only                |
| `SpectralConvS2`                    | `DistributedSpectralConvS2`                    |                                          |
| `DiscreteContinuousConvS2`          | `DistributedDiscreteContinuousConvS2`          |                                          |
| `DiscreteContinuousConvTransposeS2` | `DistributedDiscreteContinuousConvTransposeS2` |                                          |
| `NeighborhoodAttentionS2`           | `DistributedNeighborhoodAttentionS2`           |                                          |
| `ResampleS2`                        | `DistributedResampleS2`                        |                                          |
| `QuadratureS2`                      | `DistributedQuadratureS2`                      | HEALPix, polar ranks only                |
| `QuadratureVolume`                  | `DistributedQuadratureVolume`                  |                                          |
| `AttentionS2`                       | —                                              | Global attention; no distributed version |
| `GaussianRandomFieldS2`             | —                                              | Sampling utility                         |

HEALPix grids decompose along latitude only: each polar rank holds a band of
whole rings, and a layer built on a process grid with more than one azimuth rank
raises `NotImplementedError`. The other distributed layers require a
latitude-longitude grid for now.

## Layer reference

Distributed SHT endpoint properties describe global domains; rank-local blocks
continue to follow the process-grid partitioning. `.grid` remains the global
spatial descriptor.

```{eval-rst}
.. currentmodule:: torch_harmonics.distributed

.. autosummary::
   :toctree: generated
   :nosignatures:

   DistributedRealSHT
   DistributedInverseRealSHT
   DistributedRealVectorSHT
   DistributedInverseRealVectorSHT
   DistributedSpectralConvS2
   DistributedDiscreteContinuousConvS2
   DistributedDiscreteContinuousConvTransposeS2
   DistributedNeighborhoodAttentionS2
   DistributedResampleS2
   DistributedQuadratureS2
   DistributedQuadratureVolume
```

```{note}
The custom C++/CUDA kernels are an implementation detail invoked from these
Python modules; they have no separately documented API.
```
