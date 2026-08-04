# Distributed layers

Distributed (multi-GPU) counterparts of the serial layers. These are
available in the `torch_harmonics.distributed` subpackage.

## Coverage

The table below shows which serial layers have a distributed counterpart and
which do not.

| Serial layer                        | Distributed counterpart                        | Notes                                    |
| ----------------------------------- | ---------------------------------------------- | ---------------------------------------- |
| `RealSHT`                           | `DistributedRealSHT`                           |                                          |
| `InverseRealSHT`                    | `DistributedInverseRealSHT`                    |                                          |
| `RealVectorSHT`                     | `DistributedRealVectorSHT`                     |                                          |
| `InverseRealVectorSHT`              | `DistributedInverseRealVectorSHT`              |                                          |
| `SpectralConvS2`                    | `DistributedSpectralConvS2`                    |                                          |
| `DiscreteContinuousConvS2`          | `DistributedDiscreteContinuousConvS2`          |                                          |
| `DiscreteContinuousConvTransposeS2` | `DistributedDiscreteContinuousConvTransposeS2` |                                          |
| `NeighborhoodAttentionS2`           | `DistributedNeighborhoodAttentionS2`           |                                          |
| `ResampleS2`                        | `DistributedResampleS2`                        |                                          |
| `QuadratureS2`                      | `DistributedQuadratureS2`                      |                                          |
| `AttentionS2`                       | —                                              | Global attention; no distributed version |
| `GaussianRandomFieldS2`             | —                                              | Sampling utility                         |

## Layer reference

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
```

```{note}
The custom C++/CUDA kernels are an implementation detail invoked from these
Python modules; they have no separately documented API.
```
