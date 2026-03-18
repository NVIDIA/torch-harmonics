import abc
import dataclasses
from collections.abc import Callable
from typing import Self

import torch

from torch_harmonics.benchmark.hardware import get_device
from torch_harmonics.benchmark.timer import (
    CPUEventPair,
    CPUTimer,
    CUDATimer,
    NullTimer,
    Timer,
    TimerResult,
)

TensorDict = dict[str, torch.Tensor]


@dataclasses.dataclass
class BenchmarkResult:
    phase: str
    device: str
    timer: TimerResult
    cpu_time: float

    def __repr__(self) -> str:
        return f"BenchmarkResult(phase={self.phase}, device={self.device}, timer={self.timer}, cpu_time={self.cpu_time})"

    def asdict(self) -> dict:
        return dataclasses.asdict(self)

    def get_logs(self, max_depth: int) -> dict[str, float]:
        logs = {"phase": self.phase, "device": self.device, "cpu_time": self.cpu_time}
        logs.update(self.timer.get_logs(max_depth=max_depth))
        return logs


class BenchmarkABC(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def new(cls: type[Self]) -> Self:
        """
        Initialize any state needed for the benchmark.
        This will be called once before the benchmark is run.
        """
        pass

    @classmethod
    def _make_timer(cls) -> CUDATimer | CPUTimer:
        if torch.cuda.is_available():
            return CUDATimer()
        return CPUTimer()

    @classmethod
    def _sync(cls) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @classmethod
    def run_forward_benchmark(cls, iters=10, warmup=1) -> BenchmarkResult:
        null_timer = NullTimer()
        benchmark = cls.new()
        for _ in range(warmup):
            benchmark.run_instance_forward(null_timer)
        timer = cls._make_timer()
        cpu_timer = CPUEventPair()
        cpu_timer.record_start()
        for _ in range(iters):
            with timer:
                benchmark.run_instance_forward(timer)
        cls._sync()
        cpu_timer.record_end()
        return BenchmarkResult(
            phase="forward",
            device=str(get_device()),
            timer=timer.result,
            cpu_time=cpu_timer.elapsed_time_ms(),
        )

    @classmethod
    def run_backward_benchmark(cls, iters=10, warmup=1) -> BenchmarkResult:
        null_timer = NullTimer()
        benchmark = cls.new()
        benchmark.run_instance_forward(null_timer)
        for _ in range(warmup):
            benchmark.run_instance_backward(null_timer)
        timer = cls._make_timer()
        cpu_timer = CPUEventPair()
        cpu_timer.record_start()
        for _ in range(iters):
            with timer:
                benchmark.run_instance_backward(timer)
        cls._sync()
        cpu_timer.record_end()
        return BenchmarkResult(
            phase="backward",
            device=str(get_device()),
            timer=timer.result,
            cpu_time=cpu_timer.elapsed_time_ms(),
        )

    @abc.abstractmethod
    def run_instance_forward(self: Self, timer: Timer) -> TensorDict:
        """
        Run the benchmark in backward pass. This will be called multiple times,
        and should return a TensorDict of results.

        This must not mutate any state on self, since the same instance may be
        used across multiple iterations.
        """
        pass

    @abc.abstractmethod
    def run_instance_backward(self: Self, timer: Timer) -> TensorDict:
        """
        Run the benchmark in forward pass. This will be called multiple times,
        and should return a TensorDict of results.

        This must not mutate any state on self, since the same instance may be
        used across multiple iterations.
        """
        pass


_BENCHMARKS: dict[str, type[BenchmarkABC]] = {}


def register_benchmark(name: str) -> Callable[[type[BenchmarkABC]], type[BenchmarkABC]]:
    def _register(fn: type[BenchmarkABC]) -> type[BenchmarkABC]:
        if name in _BENCHMARKS:
            raise ValueError(f"Benchmark with name '{name}' is already registered.")
        _BENCHMARKS[name] = fn
        return fn

    return _register


def get_benchmarks() -> dict[str, type[BenchmarkABC]]:
    return _BENCHMARKS.copy()
