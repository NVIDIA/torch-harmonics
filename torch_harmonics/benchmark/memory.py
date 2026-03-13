import dataclasses
from typing import Literal

import torch

_benchmark_memory_started = False


@dataclasses.dataclass
class MemoryResult:
    max_alloc: int
    max_reserved: int


class MemoryBenchmark:
    def __init__(self):
        self._ended = False

    def __enter__(self) -> "MemoryBenchmark":
        global _benchmark_memory_started
        if _benchmark_memory_started:
            raise RuntimeError(
                "benchmark_memory cannot be nested due to its use of globals"
            )
        try:
            if self._ended:
                raise RuntimeError(
                    "MemoryBenchmark cannot be reused after it has ended."
                )
            _benchmark_memory_started = True
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self._max_alloc = 0
            self._max_reserved = 0
        except Exception as e:
            _benchmark_memory_started = False
            raise e
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        torch.cuda.synchronize()
        global _benchmark_memory_started
        _benchmark_memory_started = False
        self._ended = True
        self._max_alloc = torch.cuda.max_memory_allocated()
        self._max_reserved = torch.cuda.max_memory_reserved()
        return False

    @property
    def result(self) -> MemoryResult:
        if _benchmark_memory_started:
            raise RuntimeError(
                "MemoryBenchmark is still running. "
                "Please exit the context before getting results."
            )
        if not self._ended:
            raise RuntimeError(
                "MemoryBenchmark has not been run yet. "
                "Please enter and exit the context before getting results."
            )
        return MemoryResult(max_alloc=self._max_alloc, max_reserved=self._max_reserved)


def benchmark_memory() -> MemoryBenchmark:
    return MemoryBenchmark()
