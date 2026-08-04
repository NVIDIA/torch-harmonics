# Tutorials

These tutorials are rendered from the notebooks in the repository's
`notebooks/` directory. Their stored outputs are shown as-is; the notebooks are
not re-executed at documentation build time (they require CUDA capable devices
and access to datasets).

```{toctree}
---
maxdepth: 1
caption: Fundamentals
---
getting_started
plot_spherical_harmonics
quadrature
partial_derivatives
gradient_analysis
```

```{toctree}
---
maxdepth: 1
caption: Applications
---
helmholtz
shallow_water_equations
train_spherical_neural_operator
stanford_2d3ds_dataset
```

```{toctree}
---
maxdepth: 1
caption: Advanced topics
---
filter_basis_functions
resample_sphere
conditioning_sht
equivariance_test
```
