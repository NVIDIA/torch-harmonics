import logging

import torch

logger = logging.getLogger(__name__)

# Batch size scale factors relative to Tesla T4 (the default baseline).
# To add a new GPU, add an entry mapping its device name (as returned by
# torch.cuda.get_device_properties(...).name) to a float scale factor.
# Values > 1.0 mean the GPU is faster than a T4 and can use larger batches;
# values < 1.0 mean it is slower.
_BATCH_SIZE_FACTORS: dict[str, float] = {
    "Tesla T4": 1.0,
}

_DEFAULT_BATCH_SIZE_FACTOR = 1.0

_device: torch.device | None = None


def set_device(device: str | torch.device) -> None:
    """Override the device used by all benchmarks."""
    global _device
    _device = torch.device(device)


def get_device() -> torch.device:
    if _device is not None:
        return _device
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def get_batch_size_factor() -> float:
    """Return a hardware-dependent scale factor for benchmark batch sizes.

    Benchmarks define a base batch size tuned for a Tesla T4. This function
    returns a multiplier so that benchmarks take a similar wall-clock time
    on other hardware. If the batch size is too small, the GPU will not be fully
    occupied, and the benchmarks cannot be used to tune performance.
    
    Unknown devices fall back to the T4 default (1.0).
    """
    if not torch.cuda.is_available():
        return _DEFAULT_BATCH_SIZE_FACTOR
    name = torch.cuda.get_device_properties(torch.cuda.current_device()).name
    factor = _BATCH_SIZE_FACTORS.get(name)
    if factor is None:
        logger.warning(
            f"Unknown GPU '{name}', using default batch size factor "
            f"{_DEFAULT_BATCH_SIZE_FACTOR}. Add an entry to "
            f"_BATCH_SIZE_FACTORS in hardware.py to tune for this device."
        )
        return _DEFAULT_BATCH_SIZE_FACTOR
    return factor


def scale_batch_size(base: int, factor: float) -> int:
    """Scale a base batch size by the given factor.

    Always returns at least 1.
    """
    return max(1, round(base * factor))
