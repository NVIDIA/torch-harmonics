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
from torch_harmonics.disco import DiscreteContinuousConvS2


class DiscreteContinuousConvS2Benchmark(BenchmarkABC):

    @final
    def __init__(self, conv: DiscreteContinuousConvS2, x: torch.Tensor):
        self.conv = conv
        self.x = x

    @classmethod
    @abc.abstractmethod
    def new(cls) -> "DiscreteContinuousConvS2Benchmark": ...

    @classmethod
    @final
    def new_with_shape(
        cls: type[Self],
        B: int,
        in_channels: int,
        out_channels: int,
        nlat: int,
        nlon: int,
        kernel_shape: int = 3,
    ) -> Self:
        device = get_device()
        conv = DiscreteContinuousConvS2(
            in_channels=in_channels,
            out_channels=out_channels,
            in_shape=(nlat, nlon),
            out_shape=(nlat, nlon),
            kernel_shape=kernel_shape,
            theta_cutoff=None,
            optimized_kernel=False,
        ).to(device)
        x = torch.randn(B, in_channels, nlat, nlon, dtype=torch.float32, device=device, requires_grad=True)
        return cls(conv=conv, x=x)

    @final
    def run_instance(self, timer: Timer) -> TensorDict:
        if self.x.grad is not None:
            self.x.grad.zero_()
        with timer.child("forward"):
            result = self.conv(self.x)
        with timer.child("backward"):
            g = torch.randn_like(result)
            result.backward(g)
        return {"output": result.detach()}


@register_benchmark("disco_conv_s2_torch_1deg")
class DiscreteContinuousConvS2TorchBenchmark1Degree(DiscreteContinuousConvS2Benchmark):

    @classmethod
    def new(cls) -> "DiscreteContinuousConvS2TorchBenchmark1Degree":
        return cls.new_with_shape(
            B=scale_batch_size(4), in_channels=4, out_channels=4, nlat=180, nlon=360,
        )
