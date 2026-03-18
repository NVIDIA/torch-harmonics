import collections
import dataclasses
import time
from typing import Literal, Protocol, Self

import torch


@dataclasses.dataclass
class TimerResult:
    count: int
    avg_time: float
    unit: str = "ms"
    children: dict[str, "TimerResult"] = dataclasses.field(default_factory=dict)

    def get_logs(self, max_depth: int) -> dict[str, float]:
        logs = {
            "avg_time": self.avg_time,
            "unit": self.unit,
        }
        if max_depth > 0:
            for child_name, child in self.children.items():
                for log_name, value in child.get_logs(max_depth=max_depth - 1).items():
                    logs[f"{child_name}/{log_name}"] = value
        return logs

    def assert_close(self, other: "TimerResult", rtol=0.02, children_rtol=0.02) -> None:
        if self.count != other.count:
            raise AssertionError(f"count differ: {self.count} vs {other.count}")
        if not torch.isclose(
            torch.tensor(self.avg_time), torch.tensor(other.avg_time), rtol=rtol
        ):
            raise AssertionError(
                f"avg_time differ: {self.avg_time} vs "
                f"{other.avg_time} given rtol={rtol}"
            )
        if self.children.keys() != other.children.keys():
            raise AssertionError(
                f"children keys differ: {self.children.keys()} vs "
                f"{other.children.keys()}"
            )
        for key in self.children.keys():
            try:
                self.children[key].assert_close(
                    other.children[key], rtol=children_rtol, children_rtol=children_rtol
                )
            except AssertionError as e:
                raise AssertionError(f"child '{key}' differ: {e}") from e


class Timer(Protocol):
    def child(self, name: str) -> Self: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]: ...


class NullTimer:
    def child(self, name: str) -> "NullTimer":
        return self

    def __enter__(self) -> "NullTimer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        return False


_: Timer = NullTimer()
del _


class EventPair:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self._stream = None
        self._start_recorded = False
        self._end_recorded = False

    def record_start(self):
        if self._start_recorded:
            raise RuntimeError(
                "record_start has already been called on this EventPair."
            )
        self._stream = torch.cuda.current_stream()
        self.start.record(self._stream)
        self._start_recorded = True

    def record_end(self):
        if not self._start_recorded:
            raise RuntimeError("record_start must be called before record_end")
        if self._end_recorded:
            raise RuntimeError("record_end has already been called on this EventPair.")
        if self._stream is None:
            raise RuntimeError("record_start must be called before record_end")
        self.end.record(self._stream)
        self._end_recorded = True

    def elapsed_time_ms(self) -> float:
        if not self._start_recorded or not self._end_recorded:
            raise RuntimeError(
                "Both record_start and record_end must be called "
                "before elapsed_time_ms can be called."
            )
        return self.start.elapsed_time(self.end)


class CPUEventPair:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def record_start(self):
        if self.start_time is not None:
            raise RuntimeError(
                "record_start has already been called on this CPUEventPair."
            )
        self.start_time = time.time()

    def record_end(self):
        if self.start_time is None:
            raise RuntimeError("record_start must be called before record_end")
        if self.end_time is not None:
            raise RuntimeError(
                "record_end has already been called on this CPUEventPair."
            )
        self.end_time = time.time()

    def elapsed_time_ms(self) -> float:
        if self.start_time is None or self.end_time is None:
            raise RuntimeError(
                "Both record_start and record_end must be called "
                "before elapsed_time_ms can be called."
            )
        return (self.end_time - self.start_time) * 1000


class CUDATimer:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot use CUDATimer.")
        self._children: collections.defaultdict[str, CUDATimer] = (
            collections.defaultdict(CUDATimer)
        )
        self._event_pairs: list[EventPair] = []
        self._entered = False
        self._result: TimerResult | None = None

    @classmethod
    def new_if_available(cls) -> "CUDATimer | NullTimer":
        if torch.cuda.is_available():
            return cls()
        else:
            return NullTimer()

    def __enter__(self):
        if self._entered:
            raise RuntimeError("CUDATimer is already entered.")
        self._entered = True
        self._event_pairs.append(EventPair())
        self._event_pairs[-1].record_start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._event_pairs:
            raise RuntimeError("CUDATimer context was not properly entered.")
        self._event_pairs[-1].record_end()
        self._entered = False
        return False

    def child(self, name: str) -> "CUDATimer":
        if not self._entered:
            raise RuntimeError(
                "CUDATimer child cannot be used before entering the timer."
            )
        return self._children[name]

    @property
    def _avg_time(self) -> float:
        if len(self._event_pairs) == 0:
            raise RuntimeError(
                "CUDATimer report cannot be generated before entering the timer."
            )
        total_time = sum(
            event_pair.elapsed_time_ms() for event_pair in self._event_pairs
        )
        return total_time / len(self._event_pairs)

    def _child_reports(self) -> dict[str, TimerResult]:
        return {name: child.result for name, child in self._children.items()}

    @property
    def result(self) -> TimerResult:
        if self._result is None:
            torch.cuda.synchronize()
            self._result = TimerResult(
                count=len(self._event_pairs),
                avg_time=self._avg_time,
                children=self._child_reports(),
            )
        return self._result


__: type[Timer] = CUDATimer
del __


class CPUTimer:
    """Wall-clock timer with the same interface as CUDATimer, for CPU benchmarks."""

    def __init__(self):
        self._children: collections.defaultdict[str, "CPUTimer"] = (
            collections.defaultdict(CPUTimer)
        )
        self._event_pairs: list[CPUEventPair] = []
        self._entered = False
        self._result: TimerResult | None = None

    def __enter__(self):
        if self._entered:
            raise RuntimeError("CPUTimer is already entered.")
        self._entered = True
        self._event_pairs.append(CPUEventPair())
        self._event_pairs[-1].record_start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._event_pairs:
            raise RuntimeError("CPUTimer context was not properly entered.")
        self._event_pairs[-1].record_end()
        self._entered = False
        return False

    def child(self, name: str) -> "CPUTimer":
        if not self._entered:
            raise RuntimeError(
                "CPUTimer child cannot be used before entering the timer."
            )
        return self._children[name]

    @property
    def _avg_time(self) -> float:
        if not self._event_pairs:
            raise RuntimeError(
                "CPUTimer report cannot be generated before entering the timer."
            )
        total_time = sum(ep.elapsed_time_ms() for ep in self._event_pairs)
        return total_time / len(self._event_pairs)

    def _child_reports(self) -> dict[str, TimerResult]:
        return {name: child.result for name, child in self._children.items()}

    @property
    def result(self) -> TimerResult:
        if self._result is None:
            self._result = TimerResult(
                count=len(self._event_pairs),
                avg_time=self._avg_time,
                children=self._child_reports(),
            )
        return self._result


_: type[Timer] = CPUTimer
del _
