// coding=utf-8
//
// SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>

#include "spectral.h"

namespace spectral_kernels {

namespace {

void check_cublas(cublasStatus_t status, const char* msg)
{
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, msg, " (cublas status=", static_cast<int>(status), ")");
}

void run_batched_real_gemm_rowmajor(
    cublasHandle_t handle,
    const void* a_ptr,  // [batch, m, k] row-major
    const void* b_ptr,  // [batch, k, n] row-major
    float* c_ptr,       // [batch, m, n] row-major, fp32 output
    int batch_count,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t stride_a,
    int64_t stride_b,
    int64_t stride_c,
    cublasDataType_t in_type,
    cublasComputeType_t compute_type,
    float alpha,
    float beta,
    bool /*accum_fp32*/)
{
    // Row-major GEMM C = A @ B (m x n) is expressed as column-major:
    // C_col(n x m) = B_col(n x k) @ A_col(k x m)
    // with lda=n, ldb=k, ldc=n.
    check_cublas(
        cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            static_cast<int>(n),
            static_cast<int>(m),
            static_cast<int>(k),
            &alpha,
            b_ptr,
            in_type,
            static_cast<int>(n),
            static_cast<long long>(stride_b),
            a_ptr,
            in_type,
            static_cast<int>(k),
            static_cast<long long>(stride_a),
            &beta,
            c_ptr,
            CUDA_R_32F,
            static_cast<int>(n),
            static_cast<long long>(stride_c),
            batch_count,
            compute_type,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmStridedBatchedEx failed");
}

torch::Tensor spectral_contract_cuda_impl(
    torch::Tensor x,
    torch::Tensor w_re_mm,  // (GH, CoutG, CinG) float/bfloat16
    torch::Tensor w_im_mm,  // (GH, CoutG, CinG) float/bfloat16
    int64_t num_groups,
    bool accum_fp32)
{
    TORCH_CHECK(x.scalar_type() == at::kComplexFloat, "spectral CUDA kernel currently supports complex64 x");
    TORCH_CHECK(w_re_mm.dim() == 3, "w_re must have shape (G*H, CoutG, CinG)");
    TORCH_CHECK(w_im_mm.sizes() == w_re_mm.sizes(), "w_im shape must match w_re shape");
    TORCH_CHECK(
        w_re_mm.scalar_type() == at::kBFloat16 || w_re_mm.scalar_type() == at::kHalf || w_re_mm.scalar_type() == at::kFloat,
        "w_re must be bfloat16, float16, or float32");
    TORCH_CHECK(w_im_mm.scalar_type() == w_re_mm.scalar_type(), "w_re and w_im must have same dtype");

    const int64_t B = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);
    const int64_t GH = w_re_mm.size(0);
    const int64_t CoutG = w_re_mm.size(1);
    const int64_t CinG = w_re_mm.size(2);

    TORCH_CHECK(num_groups > 0, "num_groups must be positive");
    TORCH_CHECK(GH % H == 0, "w_re first dimension must be divisible by H");
    const int64_t G = GH / H;
    TORCH_CHECK(G == num_groups, "w_re first dimension must be num_groups * H");
    TORCH_CHECK(C == G * CinG, "x channels must equal num_groups * CinG");

    // x_g: (B, GH, CinG, W)
    auto x_g = x.view({B, G, CinG, H, W}).permute({0, 1, 3, 2, 4}).reshape({B, GH, CinG, W});

    const auto gemm_dtype = w_re_mm.scalar_type();
    auto x_re_mm = torch::real(x_g).to(gemm_dtype).contiguous();
    auto x_im_mm = torch::imag(x_g).to(gemm_dtype).contiguous();
    auto wr = w_re_mm.contiguous();
    auto wi = w_im_mm.contiguous();

    // fp32 output buffers: (B, GH, CoutG, W)
    auto y_re = torch::empty({B, GH, CoutG, W}, x.options().dtype(torch::kFloat));
    auto y_im = torch::empty({B, GH, CoutG, W}, x.options().dtype(torch::kFloat));

    auto handle = at::cuda::getCurrentCUDABlasHandle();
    check_cublas(cublasSetStream(handle, at::cuda::getDefaultCUDAStream()), "cublasSetStream failed");

    cublasDataType_t in_type = CUDA_R_32F;
    if (gemm_dtype == at::kBFloat16) {
        in_type = CUDA_R_16BF;
    } else if (gemm_dtype == at::kHalf) {
        in_type = CUDA_R_16F;
    }
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    if (accum_fp32) {
        if (gemm_dtype == at::kBFloat16) {
            compute_type = CUBLAS_COMPUTE_32F_FAST_16BF;
        } else if (gemm_dtype == at::kHalf) {
            compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
        }
    }

    const int batch_count = static_cast<int>(GH);
    const int64_t stride_a = CoutG * CinG;
    const int64_t stride_b = CinG * W;
    const int64_t stride_c = CoutG * W;
    const int64_t x_batch_stride = GH * stride_b;
    const int64_t y_batch_stride = GH * stride_c;

    for (int64_t b = 0; b < B; ++b) {
        const char* xre_b = static_cast<const char*>(x_re_mm.data_ptr()) + b * x_batch_stride * x_re_mm.element_size();
        const char* xim_b = static_cast<const char*>(x_im_mm.data_ptr()) + b * x_batch_stride * x_im_mm.element_size();
        float* yre_b = y_re.data_ptr<float>() + b * y_batch_stride;
        float* yim_b = y_im.data_ptr<float>() + b * y_batch_stride;

        // Re = Wr*Xr - Wi*Xi
        run_batched_real_gemm_rowmajor(
            handle, wr.data_ptr(), xre_b, yre_b, batch_count, CoutG, W, CinG, stride_a, stride_b, stride_c,
            in_type, compute_type, 1.0f, 0.0f, accum_fp32);
        run_batched_real_gemm_rowmajor(
            handle, wi.data_ptr(), xim_b, yre_b, batch_count, CoutG, W, CinG, stride_a, stride_b, stride_c,
            in_type, compute_type, -1.0f, 1.0f, accum_fp32);

        // Im = Wr*Xi + Wi*Xr
        run_batched_real_gemm_rowmajor(
            handle, wr.data_ptr(), xim_b, yim_b, batch_count, CoutG, W, CinG, stride_a, stride_b, stride_c,
            in_type, compute_type, 1.0f, 0.0f, accum_fp32);
        run_batched_real_gemm_rowmajor(
            handle, wi.data_ptr(), xre_b, yim_b, batch_count, CoutG, W, CinG, stride_a, stride_b, stride_c,
            in_type, compute_type, 1.0f, 1.0f, accum_fp32);
    }

    auto y_complex = torch::complex(y_re, y_im);
    auto y = y_complex
                 .reshape({B, G, H, CoutG, W})
                 .permute({0, 1, 3, 2, 4})
                 .reshape({B, G * CoutG, H, W})
                 .contiguous();
    return y;
}

}  // namespace

torch::Tensor spectral_contract_cuda(
    torch::Tensor x,
    torch::Tensor w,
    int64_t num_groups,
    int64_t gemm_dtype_code,
    bool accum_fp32)
{
    CHECK_CUDA_INPUT_TENSOR(x);
    CHECK_CUDA_INPUT_TENSOR(w);
    TORCH_CHECK(x.dim() == 4, "x must have shape (B, C, H, W)");
    TORCH_CHECK(w.dim() == 4, "w must have shape (G, CinG, CoutG, H)");
    TORCH_CHECK(w.scalar_type() == at::kComplexFloat, "spectral CUDA kernel currently supports complex64 w");

    const int64_t H = x.size(2);

    const int64_t G = w.size(0);
    const int64_t H_w = w.size(3);

    TORCH_CHECK(H == H_w, "x and w must agree on H/l dimension");
    TORCH_CHECK(num_groups > 0, "num_groups must be positive");
    TORCH_CHECK(G == num_groups, "w.shape[0] must equal num_groups");

    const at::cuda::CUDAGuard device_guard(x.device());
    const int64_t GH = G * H;
    auto w_g = w.permute({0, 3, 2, 1}).reshape({GH, w.size(2), w.size(1)});
    torch::ScalarType gemm_dtype = torch::kFloat;
    if (gemm_dtype_code == 1) {
        gemm_dtype = torch::kBFloat16;
    } else if (gemm_dtype_code == 2) {
        gemm_dtype = torch::kHalf;
    } else {
        gemm_dtype = torch::kFloat;
    }
    auto w_re_mm = torch::real(w_g).to(gemm_dtype);
    auto w_im_mm = torch::imag(w_g).to(gemm_dtype);
    return spectral_contract_cuda_impl(x, w_re_mm, w_im_mm, num_groups, accum_fp32);
}

torch::Tensor spectral_contract_cuda_prepacked(
    torch::Tensor x,
    torch::Tensor w_re,
    torch::Tensor w_im,
    int64_t num_groups,
    bool accum_fp32)
{
    CHECK_CUDA_INPUT_TENSOR(x);
    CHECK_CUDA_INPUT_TENSOR(w_re);
    CHECK_CUDA_INPUT_TENSOR(w_im);
    TORCH_CHECK(x.dim() == 4, "x must have shape (B, C, H, W)");
    TORCH_CHECK(w_re.dim() == 3, "w_re must have shape (G*H, CoutG, CinG)");
    TORCH_CHECK(w_im.sizes() == w_re.sizes(), "w_im shape must match w_re shape");
    return spectral_contract_cuda_impl(x, w_re, w_im, num_groups, accum_fp32);
}

TORCH_LIBRARY_IMPL(spectral_kernels, CUDA, m)
{
    m.impl("forward", &spectral_contract_cuda);
    m.impl("forward_prepacked", &spectral_contract_cuda_prepacked);
}

}  // namespace spectral_kernels
