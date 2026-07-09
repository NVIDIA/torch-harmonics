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

## Utilities

```{eval-rst}
.. currentmodule:: torch_harmonics

.. autosummary::
   :toctree: generated
   :nosignatures:

   truncate_sht
```

```{note}
The custom C++/CUDA kernels are an implementation detail invoked from these
Python modules; they have no separately documented API.
```
