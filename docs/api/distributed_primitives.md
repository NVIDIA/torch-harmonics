# Distributed primitives

Low-level communication primitives used internally by the distributed
layers. They are documented here so that advanced users can build custom
distributed operators on top of the same infrastructure.

All primitives are autograd-compatible: each one defines both a forward
and backward communication pattern so that gradients flow correctly
through distributed computations.

## Transpose (all-to-all)

```{eval-rst}
.. currentmodule:: torch_harmonics.distributed

.. autosummary::
   :toctree: generated
   :nosignatures:

   distributed_transpose_azimuth
   distributed_transpose_polar
```

## Copy / Reduce

```{eval-rst}
.. currentmodule:: torch_harmonics.distributed

.. autosummary::
   :toctree: generated
   :nosignatures:

   copy_to_polar_region
   copy_to_azimuth_region
   reduce_from_polar_region
   reduce_from_azimuth_region
```

## Scatter / Gather

```{eval-rst}
.. currentmodule:: torch_harmonics.distributed

.. autosummary::
   :toctree: generated
   :nosignatures:

   scatter_to_polar_region
   gather_from_polar_region
   gather_from_copy_to_polar_region
   reduce_from_scatter_to_polar_region
   reduce_from_scatter_to_azimuth_region
```

## Halo exchange

```{eval-rst}
.. currentmodule:: torch_harmonics.distributed

.. autosummary::
   :toctree: generated
   :nosignatures:

   polar_halo_exchange
   get_group_neighbors
```

## Tensor reshaping

```{eval-rst}
.. currentmodule:: torch_harmonics.distributed

.. autosummary::
   :toctree: generated
   :nosignatures:

   flatten_and_pad_leading_dims
   unpad_and_unflatten_leading_dims
```
