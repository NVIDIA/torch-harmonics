import abc
from typing import Self, final

import torch

from torch_harmonics.benchmark.benchmark import (
    BenchmarkABC,
    TensorDict,
    register_benchmark,
)
from torch_harmonics.benchmark.hardware import get_device, scale_batch_size
from torch_harmonics.benchmark.timer import Timer
from torch_harmonics.sht import InverseRealSHT, RealSHT


class RealSHTBenchmark(BenchmarkABC):

    @final
    def __init__(self, forward_sht: RealSHT, x: torch.Tensor):
        self.forward_sht = forward_sht
        self.x = x

    @classmethod
    @abc.abstractmethod
    def new(cls) -> "RealSHTBenchmark": ...

    @classmethod
    @final
    def new_with_shape(cls: type[Self], B: int, H: int, L: int) -> Self:
        device = get_device()
        x = torch.randn(B, H, L, device=device, requires_grad=True)
        forward_sht = RealSHT(nlat=H, nlon=L).to(device)
        return cls(forward_sht=forward_sht, x=x)

    @final
    def run_instance(self, timer: Timer) -> TensorDict:
        if self.x.grad is not None:
            self.x.grad.zero_()
        with timer.child("forward"):
            result = self.forward_sht(self.x)
        with timer.child("backward"):
            g = torch.randn_like(result)
            result.backward(g)
        return {"output": result.detach()}

# predefined benchmarks
@register_benchmark("real_sht_1deg")
class RealSHTBenchmark1Degree(RealSHTBenchmark):

    @classmethod
    def new(cls) -> "RealSHTBenchmark1Degree":
        return cls.new_with_shape(B=scale_batch_size(4096), H=180, L=360)

@register_benchmark("real_sht_quarter_deg")
class RealSHTBenchmarkQuarterDegree(RealSHTBenchmark):

    @classmethod
    def new(cls) -> "RealSHTBenchmarkQuarterDegree":
        return cls.new_with_shape(B=scale_batch_size(1), H=721, L=1440)


class InverseRealSHTBenchmark(BenchmarkABC):

    @final
    def __init__(self, inverse_sht: InverseRealSHT, x_hat: torch.Tensor):
        self.inverse_sht = inverse_sht
        self.x_hat = x_hat

    @classmethod
    @abc.abstractmethod
    def new(cls) -> "InverseRealSHTBenchmark": ...

    @classmethod
    @final
    def new_with_shape(cls: type[Self], B: int, H: int, L: int) -> Self:
        device = get_device()
        x = torch.randn(B, H, L, device=device)
        forward_sht = RealSHT(nlat=H, nlon=L).to(device)
        x_hat = forward_sht(x)
        x_hat = x_hat.detach().requires_grad_(True)
        inverse_sht = InverseRealSHT(nlat=H, nlon=L).to(device)
        return cls(inverse_sht=inverse_sht, x_hat=x_hat)

    @final
    def run_instance(self, timer: Timer) -> TensorDict:
        if self.x_hat.grad is not None:
            self.x_hat.grad.zero_()
        with timer.child("forward"):
            result = self.inverse_sht(self.x_hat)
        with timer.child("backward"):
            g = torch.randn_like(result)
            result.backward(g)
        return {"output": result.detach()}

# predefined benchmarks
@register_benchmark("inverse_real_sht_1deg")
class InverseRealSHTBenchmark1Degree(InverseRealSHTBenchmark):

    @classmethod
    def new(cls) -> "InverseRealSHTBenchmark1Degree":
        return cls.new_with_shape(B=scale_batch_size(4096), H=180, L=360)

@register_benchmark("inverse_real_sht_quarter_deg")
class InverseRealSHTBenchmarkQuarterDegree(InverseRealSHTBenchmark):

    @classmethod
    def new(cls) -> "InverseRealSHTBenchmarkQuarterDegree":
        return cls.new_with_shape(B=scale_batch_size(1), H=721, L=1440)
