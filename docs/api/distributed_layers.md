# Distributed layers

Distributed (multi-GPU) counterparts of the serial layers. These are
available in the `torch_harmonics.distributed` subpackage.

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
