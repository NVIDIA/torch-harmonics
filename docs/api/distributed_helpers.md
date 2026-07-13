# Distributed helpers

Distributed (multi-GPU) setup, process group accessors, and data
partitioning utilities. These are available in the
`torch_harmonics.distributed` subpackage.
See the {doc}`distributed guide </guide/distributed>` for a complete
walkthrough.

## Setup and teardown

```{eval-rst}
.. currentmodule:: torch_harmonics.distributed

.. autosummary::
   :toctree: generated
   :nosignatures:

   init
   finalize
   is_initialized
   is_distributed_polar
   is_distributed_azimuth
```

## Process group accessors

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

## Data partitioning

```{eval-rst}
.. currentmodule:: torch_harmonics.distributed

.. autosummary::
   :toctree: generated
   :nosignatures:

   compute_split_shapes
   split_tensor_along_dim
```
