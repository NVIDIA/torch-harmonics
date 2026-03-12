from torch_harmonics.benchmark.benchmark import (
    BenchmarkABC,
    BenchmarkResult,
    get_benchmarks,
    register_benchmark,
)
from torch_harmonics.benchmark.timer import (
    CUDATimer,
    NullTimer,
    Timer,
    TimerResult,
)

# Import to trigger registration of built-in benchmarks.
import torch_harmonics.benchmark.sht  # noqa: F401
