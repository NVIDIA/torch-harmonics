# coding=utf-8

import argparse
import time

import torch

from torch_harmonics.spectral_convolution import SpectralConvS2


def benchmark_once(module: SpectralConvS2, x: torch.Tensor, steps: int, warmup: int) -> float:
    module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = module(x)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(steps):
            _ = module(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / steps


def benchmark_contraction_only(
    module: SpectralConvS2,
    x_spatial: torch.Tensor,
    steps: int,
    warmup: int,
    fused: bool,
    fused_precision: str,
) -> float:
    module.eval()
    with torch.no_grad():
        x_spec = module.sht(x_spatial.float()).contiguous()
        if fused:
            from torch_harmonics.spectral._spectral_utils import fused_spectral_contract

            fn = lambda t: fused_spectral_contract(
                t,
                module.weight,
                module.num_groups,
                gemm_dtype=fused_precision,
                accum_fp32=True,
            )
        else:
            fn = lambda t: module._contract_lwise(
                t.reshape(t.shape[0], module.num_groups, t.shape[1] // module.num_groups, t.shape[2], t.shape[3]),
                module.weight,
            ).reshape(t.shape[0], module.out_channels, t.shape[2], t.shape[3]).contiguous()

        for _ in range(warmup):
            _ = fn(x_spec)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(steps):
            _ = fn(x_spec)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark baseline vs fused SpectralConvS2.")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--channels", type=int, default=384)
    parser.add_argument("--nlat", type=int, default=180)
    parser.add_argument("--nlon", type=int, default=360)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--dtype", type=str, choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--fused-precision", type=str, choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    device = torch.device("cuda")
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    base = SpectralConvS2(
        in_shape=(args.nlat, args.nlon),
        out_shape=(args.nlat, args.nlon),
        in_channels=args.channels,
        out_channels=args.channels,
        num_groups=args.groups,
        grid_in="equiangular",
        grid_out="equiangular",
        use_fused_contract=False,
    ).to(device)

    fused = SpectralConvS2(
        in_shape=(args.nlat, args.nlon),
        out_shape=(args.nlat, args.nlon),
        in_channels=args.channels,
        out_channels=args.channels,
        num_groups=args.groups,
        grid_in="equiangular",
        grid_out="equiangular",
        use_fused_contract=True,
        fused_bf16=True,
        fused_precision=args.fused_precision,
        fused_accum_fp32=True,
    ).to(device)
    fused.weight.data.copy_(base.weight.data)

    x = torch.randn(args.batch, args.channels, args.nlat, args.nlon, device=device, dtype=dtype)

    end_to_end_base = benchmark_once(base, x, args.steps, args.warmup)
    end_to_end_fused = benchmark_once(fused, x, args.steps, args.warmup)
    contract_base = benchmark_contraction_only(base, x, args.steps, args.warmup, fused=False, fused_precision=args.fused_precision)
    contract_fused = benchmark_contraction_only(fused, x, args.steps, args.warmup, fused=True, fused_precision=args.fused_precision)

    print("=== Fused SpectralConvS2 Benchmark ===")
    print(f"Shape: B={args.batch} C={args.channels} H={args.nlat} W={args.nlon} groups={args.groups} dtype={args.dtype}")
    print(f"End-to-end baseline: {end_to_end_base:.3f} ms/iter")
    print(f"End-to-end fused:    {end_to_end_fused:.3f} ms/iter")
    print(f"End-to-end speedup:  {end_to_end_base / end_to_end_fused:.3f}x")
    print(f"Contract baseline:   {contract_base:.3f} ms/iter")
    print(f"Contract fused:      {contract_fused:.3f} ms/iter")
    print(f"Contract speedup:    {contract_base / contract_fused:.3f}x")


if __name__ == "__main__":
    main()
