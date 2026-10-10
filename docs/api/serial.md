# Serial layers

## Spherical harmonic transforms

SHTs expose their input and output domains as `grid_in` and `grid_out`; `.grid`
remains the spatial descriptor.

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

   SpectralConvS2
   DiscreteContinuousConvS2
   DiscreteContinuousConvTransposeS2
```

## Filter basis

```{eval-rst}
.. currentmodule:: torch_harmonics.filter_basis

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_filter_basis
   FilterBasis
   PiecewiseLinearFilterBasis
   HarmonicFilterBasis
   ZernikeFilterBasis
   FourierBesselFilterBasis
```

## Attention mechanism

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
