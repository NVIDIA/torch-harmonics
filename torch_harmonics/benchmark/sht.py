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
        cls.device = get_device()
        x = torch.randn(B, H, L, device=cls.device)
        x.requires_grad = True
        forward_sht = RealSHT(nlat=H, nlon=L).to(cls.device)
        return cls(forward_sht=forward_sht, x=x)

    @final
    def run_instance_forward(self, timer: Timer) -> TensorDict:
        result = self.forward_sht(self.x)
        self.output = result
        return {"output": result.detach()}

    @final
    def run_instance_backward(self, timer: Timer) -> TensorDict:
        g = torch.randn_like(self.output)
        self.output.backward(g, retain_graph=True)
        return {"gradient": self.x.grad}

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
        cls.device = get_device()
        x = torch.randn(B, H, L, device=cls.device)
        forward_sht = RealSHT(nlat=H, nlon=L).to(cls.device)
        x_hat = forward_sht(x)
        x_hat.requires_grad = True
        inverse_sht = InverseRealSHT(nlat=H, nlon=L).to(cls.device)
        return cls(inverse_sht=inverse_sht, x_hat=x_hat)

    @final
    def run_instance_forward(self, timer: Timer) -> TensorDict:
        result = self.inverse_sht(self.x_hat)
        self.output = result
        return {"output": result.detach()}

    @final
    def run_instance_backward(self, timer: Timer) -> TensorDict:
        g = torch.randn_like(self.output)
        self.output.backward(g, retain_graph=True)
        return {"gradient": self.x_hat.grad}

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
