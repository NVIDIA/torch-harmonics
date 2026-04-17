# coding=utf-8

from typing import Tuple

import torch

from . import optimized_kernels_is_available, spectral_kernels


def _contract_lwise_torch(x: torch.Tensor, weight: torch.Tensor, num_groups: int) -> torch.Tensor:
    bsz, in_chans, lmax, mmax = x.shape
    xg = x.reshape(bsz, num_groups, in_chans // num_groups, lmax, mmax)
    out = torch.einsum("bgixy,giox->bgoxy", xg, weight)
    return out.reshape(bsz, num_groups * weight.shape[2], lmax, mmax).contiguous()


def _gemm_dtype_to_code(gemm_dtype: str) -> int:
    if gemm_dtype == "fp32":
        return 0
    if gemm_dtype == "bf16":
        return 1
    if gemm_dtype == "fp16":
        return 2
    raise ValueError(f"Unsupported fused gemm dtype '{gemm_dtype}'")


def can_use_fused_spectral_contract(x: torch.Tensor, weight: torch.Tensor, num_groups: int) -> bool:
    if not optimized_kernels_is_available() or spectral_kernels is None:
        return False
    if x.ndim != 4 or weight.ndim != 4:
        return False
    if not x.is_complex() or not weight.is_complex():
        return False
    if x.shape[1] % num_groups != 0:
        return False
    return True


class _FusedSpectralContractFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        num_groups: int,
        gemm_dtype: str,
        accum_fp32: bool,
    ) -> torch.Tensor:
        if not can_use_fused_spectral_contract(x, weight, num_groups):
            out = _contract_lwise_torch(x, weight, num_groups)
            ctx.used_fallback = True
            ctx.num_groups = num_groups
            ctx.save_for_backward(x, weight)
            return out

        out = spectral_kernels.forward.default(
            x.contiguous(),
            weight.contiguous(),
            int(num_groups),
            int(_gemm_dtype_to_code(gemm_dtype)),
            bool(accum_fp32),
        )
        ctx.used_fallback = False
        ctx.num_groups = num_groups
        ctx.save_for_backward(x, weight)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None, None, None]:  # type: ignore[override]
        x, weight = ctx.saved_tensors
        num_groups = ctx.num_groups

        needs_x = ctx.needs_input_grad[0]
        needs_w = ctx.needs_input_grad[1]
        if not (needs_x or needs_w):
            return None, None, None, None, None

        with torch.enable_grad():
            x_req = x.detach().requires_grad_(needs_x)
            w_req = weight.detach().requires_grad_(needs_w)
            y = _contract_lwise_torch(x_req, w_req, num_groups)

            grad_inputs = []
            if needs_x:
                grad_inputs.append(x_req)
            if needs_w:
                grad_inputs.append(w_req)

            grad_results = torch.autograd.grad(
                outputs=y,
                inputs=tuple(grad_inputs),
                grad_outputs=grad_output,
                allow_unused=True,
            )

        grad_x = grad_results[0] if needs_x else None
        grad_w = grad_results[1 if needs_x else 0] if needs_w else None
        return grad_x, grad_w, None, None, None


def fused_spectral_contract(
    x: torch.Tensor,
    weight: torch.Tensor,
    num_groups: int,
    gemm_dtype: str = "bf16",
    accum_fp32: bool = True,
) -> torch.Tensor:
    return _FusedSpectralContractFn.apply(x, weight, num_groups, gemm_dtype, accum_fp32)


def fused_spectral_contract_prepacked(
    x: torch.Tensor,
    weight_re: torch.Tensor,
    weight_im: torch.Tensor,
    num_groups: int,
    accum_fp32: bool = True,
) -> torch.Tensor:
    if spectral_kernels is None:
        raise RuntimeError("spectral extension kernels are not available")
    return spectral_kernels.forward_prepacked.default(
        x.contiguous(),
        weight_re.contiguous(),
        weight_im.contiguous(),
        int(num_groups),
        bool(accum_fp32),
    )
