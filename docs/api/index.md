# API reference

The public API mirrors the top-level `torch_harmonics` namespace. Each entry
below is generated automatically from the module docstrings.

## Spherical harmonic transforms

```{eval-rst}
.. currentmodule:: torch_harmonics

.. autosummary::
   :toctree: generated
   :nosignatures:

   RealSHT
   InverseRealSHT
   RealVectorSHT
   InverseRealVectorSHT
```

## Convolutions

```{eval-rst}
.. currentmodule:: torch_harmonics

.. autosummary::
   :toctree: generated
   :nosignatures:

   DiscreteContinuousConvS2
   DiscreteContinuousConvTransposeS2
   SpectralConvS2
```

## Filter basis

```{eval-rst}
.. currentmodule:: torch_harmonics.filter_basis

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_filter_basis
```

## Attention

```{eval-rst}
.. currentmodule:: torch_harmonics

.. autosummary::
   :toctree: generated
   :nosignatures:

   AttentionS2
   NeighborhoodAttentionS2
```

## Resampling and quadrature

```{eval-rst}
.. currentmodule:: torch_harmonics

.. autosummary::
   :toctree: generated
   :nosignatures:

   ResampleS2
   QuadratureS2
```

## Random fields

```{eval-rst}
.. currentmodule:: torch_harmonics.random_fields

.. autosummary::
   :toctree: generated
   :nosignatures:

   GaussianRandomFieldS2
```

## Utilities

```{eval-rst}
.. currentmodule:: torch_harmonics

.. autosummary::
   :toctree: generated
   :nosignatures:

   truncate_sht
```

## Distributed

Distributed (multi-GPU) counterparts of the modules above. These are available
in the `torch_harmonics.distributed` subpackage. See the
{doc}`distributed guide </guide/distributed>` for a complete walkthrough.

### Setup and teardown

```{eval-rst}
.. currentmodule:: torch_harmonics.distributed

.. autosummary::
   :toctree: generated
   :nosignatures:

   init
   finalize
   is_initialized
```

### Process group accessors

```{eval-rst}
.. currentmodule:: torch_harmonics.distributed

.. autosummary::
   :toctree: generated
   :nosignatures:

   polar_group
   polar_group_rank
   polar_group_size
   azimuth_group
   azimuth_group_rank
   azimuth_group_size
```

### Data partitioning helpers

```{eval-rst}
.. currentmodule:: torch_harmonics.distributed

.. autosummary::
   :toctree: generated
   :nosignatures:

   compute_split_shapes
   split_tensor_along_dim
```

### Distributed modules

```{eval-rst}
.. currentmodule:: torch_harmonics.distributed

.. autosummary::
   :toctree: generated
   :nosignatures:

   DistributedRealSHT
   DistributedInverseRealSHT
   DistributedRealVectorSHT
   DistributedInverseRealVectorSHT
   DistributedDiscreteContinuousConvS2
   DistributedDiscreteContinuousConvTransposeS2
   DistributedSpectralConvS2
   DistributedNeighborhoodAttentionS2
   DistributedResampleS2
   DistributedQuadratureS2
```

```{note}
The custom C++/CUDA kernels are an implementation detail invoked from these
Python modules; they have no separately documented API.
```
