# coding=utf-8

import argparse
import sys
import types

import torch

from torch_harmonics.spectral_convolution import SpectralConvS2


def _eager_contract(self, ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    # Keep baseline in eager mode to avoid torch.compile reduction-order noise.
    return torch.einsum("bgixy,giox->bgoxy", ac, bc)


def _build_modules(
    in_shape,
    out_shape,
    channels,
    num_groups,
    device,
):
    base = SpectralConvS2(
        in_shape=in_shape,
        out_shape=out_shape,
        in_channels=channels,
        out_channels=channels,
        num_groups=num_groups,
        grid_in="equiangular",
        grid_out="equiangular",
        use_fused_contract=False,
    ).to(device)
    # Force eager baseline contraction for deterministic comparison.
    base._contract_lwise = types.MethodType(_eager_contract, base)

    fused = SpectralConvS2(
        in_shape=in_shape,
        out_shape=out_shape,
        in_channels=channels,
        out_channels=channels,
        num_groups=num_groups,
        grid_in="equiangular",
        grid_out="equiangular",
        use_fused_contract=True,
        fused_bf16=False,
        fused_accum_fp32=True,
    ).to(device)
    fused.weight.data.copy_(base.weight.data)
    return base, fused


def _check_fp32_exact(args) -> bool:
    device = torch.device(args.device)
    base, fused = _build_modules(
        in_shape=(args.nlat, args.nlon),
        out_shape=(args.nlat, args.nlon),
        channels=args.channels,
        num_groups=args.num_groups,
        device=device,
    )

    x0 = torch.randn(
        args.batch_size,
        args.channels,
        args.nlat,
        args.nlon,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    x1 = x0.detach().clone().requires_grad_(True)

    y0 = base(x0)
    y1 = fused(x1)

    forward_exact = torch.equal(y0, y1)
    grad_out = torch.randn_like(y0)
    y0.backward(grad_out)
    y1.backward(grad_out)
    x_grad_exact = torch.equal(x0.grad, x1.grad)
    w_grad_exact = torch.equal(base.weight.grad, fused.weight.grad)

    if not forward_exact:
        diff = (y0 - y1).abs().max().item()
        print(f"[FP32 exact] forward mismatch, max_abs_diff={diff:e}")
    if not x_grad_exact:
        diff = (x0.grad - x1.grad).abs().max().item()
        print(f"[FP32 exact] input-grad mismatch, max_abs_diff={diff:e}")
    if not w_grad_exact:
        diff = (base.weight.grad - fused.weight.grad).abs().max().item()
        print(f"[FP32 exact] weight-grad mismatch, max_abs_diff={diff:e}")

    ok = forward_exact and x_grad_exact and w_grad_exact
    print(f"[FP32 exact] PASS={ok}")
    return ok


def _check_bf16_tolerance(args) -> bool:
    device = torch.device(args.device)
    base, fused = _build_modules(
        in_shape=(args.nlat, args.nlon),
        out_shape=(args.nlat, args.nlon),
        channels=args.channels,
        num_groups=args.num_groups,
        device=device,
    )
    fused.fused_bf16 = True

    x = torch.randn(
        args.batch_size,
        args.channels,
        args.nlat,
        args.nlon,
        device=device,
        dtype=torch.bfloat16,
    )

    with torch.no_grad():
        y0 = base(x)
        y1 = fused(x)

    max_abs = (y0 - y1).abs().max().item()
    mean_rel = (y0 - y1).abs().mean().item() / (y0.abs().mean().item() + 1e-12)
    ok = (max_abs <= args.bf16_atol) and (mean_rel <= args.bf16_rtol)
    print(
        f"[BF16 tolerance] max_abs_diff={max_abs:e}, mean_rel_diff={mean_rel:e}, "
        f"thresholds=(atol={args.bf16_atol}, rtol={args.bf16_rtol}), PASS={ok}"
    )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify fused spectral conv parity: exact FP32 and tolerance BF16."
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--nlat", type=int, default=180)
    parser.add_argument("--nlon", type=int, default=360)
    parser.add_argument("--bf16-atol", type=float, default=5e-2)
    parser.add_argument("--bf16-rtol", type=float, default=5e-2)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available.")
        return 2

    torch.manual_seed(1234)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(1234)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    fp32_ok = _check_fp32_exact(args)
    bf16_ok = _check_bf16_tolerance(args)
    return 0 if (fp32_ok and bf16_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
