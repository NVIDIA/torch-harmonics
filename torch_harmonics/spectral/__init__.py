# coding=utf-8

import warnings

def cuda_kernels_is_available() -> bool:
    return False

def optimized_kernels_is_available() -> bool:
    return False

try:
    from spectral_helpers import cuda_kernels_is_available, optimized_kernels_is_available
except Exception:
    pass

if optimized_kernels_is_available():
    try:
        from . import _C  # noqa: F401
        from torch.ops import spectral_kernels
    except Exception:
        spectral_kernels = None
        warnings.warn("Failed to load spectral extension module; fused spectral contraction is disabled.")
else:
    spectral_kernels = None
    warnings.warn(
        "No optimized spectral kernels are available. Compile extensions with BUILD_CPP/BUILD_CUDA to enable fused spectral contraction."
    )
