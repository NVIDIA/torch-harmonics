// coding=utf-8
//
// SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "spectral.h"

namespace spectral_kernels {

torch::Tensor spectral_contract_cpu(
    torch::Tensor x,
    torch::Tensor w,
    int64_t num_groups,
    int64_t /*gemm_dtype_code*/,
    bool /*accum_fp32*/)
{
    CHECK_CPU_INPUT_TENSOR(x);
    CHECK_CPU_INPUT_TENSOR(w);
    TORCH_CHECK(x.dim() == 4, "x must have shape (B, C, H, W)");
    TORCH_CHECK(w.dim() == 4, "w must have shape (G, CinG, CoutG, H)");
    TORCH_CHECK(x.scalar_type() == w.scalar_type(), "x and w dtypes must match");
    TORCH_CHECK(
        x.scalar_type() == at::kComplexFloat || x.scalar_type() == at::kComplexDouble,
        "spectral CPU kernel expects complex64 or complex128 tensors");

    const int64_t B = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    const int64_t G = w.size(0);
    const int64_t CinG = w.size(1);
    const int64_t CoutG = w.size(2);
    const int64_t H_w = w.size(3);

    TORCH_CHECK(H == H_w, "x and w must agree on H/l dimension");
    TORCH_CHECK(num_groups > 0, "num_groups must be positive");
    TORCH_CHECK(G == num_groups, "w.shape[0] must equal num_groups");
    TORCH_CHECK(C == num_groups * CinG, "x channels must equal num_groups * w.shape[1]");

    auto y = torch::zeros({B, num_groups * CoutG, H, W}, x.options());

    AT_DISPATCH_COMPLEX_TYPES(x.scalar_type(), "spectral_contract_cpu", [&] {
        auto x_acc = x.accessor<scalar_t, 4>();
        auto w_acc = w.accessor<scalar_t, 4>();
        auto y_acc = y.accessor<scalar_t, 4>();

        for (int64_t b = 0; b < B; ++b) {
            for (int64_t g = 0; g < num_groups; ++g) {
                for (int64_t o = 0; o < CoutG; ++o) {
                    const int64_t out_chan = g * CoutG + o;
                    for (int64_t h = 0; h < H; ++h) {
                        for (int64_t m = 0; m < W; ++m) {
                            scalar_t acc = scalar_t(0.0, 0.0);
                            for (int64_t i = 0; i < CinG; ++i) {
                                const int64_t in_chan = g * CinG + i;
                                acc += x_acc[b][in_chan][h][m] * w_acc[g][i][o][h];
                            }
                            y_acc[b][out_chan][h][m] = acc;
                        }
                    }
                }
            }
        }
    });

    return y;
}

torch::Tensor spectral_contract_cpu_prepacked(
    torch::Tensor x,
    torch::Tensor w_re,
    torch::Tensor w_im,
    int64_t num_groups,
    bool /*accum_fp32*/)
{
    CHECK_CPU_INPUT_TENSOR(x);
    CHECK_CPU_INPUT_TENSOR(w_re);
    CHECK_CPU_INPUT_TENSOR(w_im);
    TORCH_CHECK(x.dim() == 4, "x must have shape (B, C, H, W)");
    TORCH_CHECK(w_re.dim() == 3, "w_re must have shape (G*H, CoutG, CinG)");
    TORCH_CHECK(w_im.sizes() == w_re.sizes(), "w_im shape must match w_re shape");
    TORCH_CHECK(x.scalar_type() == at::kComplexFloat || x.scalar_type() == at::kComplexDouble, "x must be complex");
    TORCH_CHECK(w_re.scalar_type() == at::kFloat || w_re.scalar_type() == at::kBFloat16, "w_re must be float/bfloat16");
    TORCH_CHECK(w_im.scalar_type() == w_re.scalar_type(), "w_re and w_im must share dtype");

    const int64_t B = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);
    const int64_t GH = w_re.size(0);
    const int64_t CoutG = w_re.size(1);
    const int64_t CinG = w_re.size(2);

    TORCH_CHECK(num_groups > 0, "num_groups must be positive");
    TORCH_CHECK(GH % H == 0, "w_re first dimension must be divisible by H");
    const int64_t G = GH / H;
    TORCH_CHECK(G == num_groups, "w_re first dimension must equal num_groups * H");
    TORCH_CHECK(C == G * CinG, "x channels must equal num_groups * CinG");

    auto wr_f = w_re.to(torch::kFloat).view({G, H, CoutG, CinG});
    auto wi_f = w_im.to(torch::kFloat).view({G, H, CoutG, CinG});
    auto w_complex = torch::complex(wr_f, wi_f).permute({0, 3, 2, 1}).contiguous();
    return spectral_contract_cpu(x, w_complex, num_groups, 0, true);
}

TORCH_LIBRARY_IMPL(spectral_kernels, CPU, m)
{
    m.impl("forward", &spectral_contract_cpu);
    m.impl("forward_prepacked", &spectral_contract_cpu_prepacked);
}

}  // namespace spectral_kernels
