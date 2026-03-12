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
        optimized_kernel: bool = False,
        use_fft_contraction: bool = False,
    ) -> Self:
        device = get_device()
        theta_cutoff = (kernel_shape + 1) * torch.pi / float(nlat - 1)
        conv = DiscreteContinuousConvS2(
            in_channels=in_channels,
            out_channels=out_channels,
            in_shape=(nlat, nlon),
            out_shape=(nlat, nlon),
            kernel_shape=kernel_shape,
            theta_cutoff=theta_cutoff,
            optimized_kernel=optimized_kernel,
            use_fft_contraction=use_fft_contraction,
        ).to(device)
        x = torch.randn(B, in_channels, nlat, nlon, device=device)
        return cls(conv=conv, x=x)

    @final
    def run_instance(self, timer: Timer) -> TensorDict:
        result = self.conv(self.x)
        return {"output": result.detach()}


@register_benchmark("disco_conv_s2_torch_1deg")
class DiscreteContinuousConvS2TorchBenchmark1Degree(DiscreteContinuousConvS2Benchmark):

    @classmethod
    def new(cls) -> "DiscreteContinuousConvS2TorchBenchmark1Degree":
        return cls.new_with_shape(
            B=scale_batch_size(16), in_channels=4, out_channels=4, nlat=180, nlon=360,
        )


@register_benchmark("disco_conv_s2_cuda_1deg")
class DiscreteContinuousConvS2CUDABenchmark1Degree(DiscreteContinuousConvS2Benchmark):

    @classmethod
    def new(cls) -> "DiscreteContinuousConvS2CUDABenchmark1Degree":
        return cls.new_with_shape(
            B=scale_batch_size(64), in_channels=4, out_channels=4, nlat=180, nlon=360,
            optimized_kernel=True,
        )


@register_benchmark("disco_conv_s2_fft_1deg")
class DiscreteContinuousConvS2FFTBenchmark1Degree(DiscreteContinuousConvS2Benchmark):

    @classmethod
    def new(cls) -> "DiscreteContinuousConvS2FFTBenchmark1Degree":
        return cls.new_with_shape(
            B=scale_batch_size(64), in_channels=4, out_channels=4, nlat=180, nlon=360,
            use_fft_contraction=True,
        )
