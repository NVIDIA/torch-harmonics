---
name: disco-kernel-dev
description: >
  Expert guide for adding, optimizing, or debugging DISCO convolution kernels
  in torch-harmonics. Use this skill whenever the user mentions: adding a new
  kernel variant or GPU arch path, wiring up a kernel through the Python dispatch
  layer, fixing bf16/fp16/autocast issues in DISCO, profiling or benchmarking
  the CSR or kpacked forward/backward, propagating kernel changes to the
  distributed DISCO path, or writing tests that cover all dispatch branches.
  Also use for debugging correctness issues in the DISCO fwd/bwd path or distributed a2a collectives.
---

# DISCO Kernel Development Guide

## Architecture snapshot

```
disco_interface.cpp          TORCH_LIBRARY("disco_kernels") — raw op schema
  ├── forward(inp, …)        CSR sparse contraction inp → (B,C,K,H,W)
  ├── backward(inp, …)       CSR transpose contraction (B,C,K,H,W) → inp
  └── forward_kpacked(…)     WGMMA kpacked forward (SM_90a + bf16/fp16 only)

disco_optimized.py           Python dispatch layer
  ├── _disco_s2_contraction_optimized         custom_op wrapping forward
  ├── _disco_s2_transpose_contraction_optimized  custom_op wrapping backward
  ├── _disco_s2_fused_conv_optimized          custom_op: contraction + einsum
  ├── _DiscoKpackedFn(autograd.Function)      WGMMA fwd + CSR bwd (unfused)
  ├── _DiscoKpackedFusedFn(autograd.Function) WGMMA fwd + CSR bwd (fused)
  └── _maybe_kpack_psi(…)    converts CSR psi to kpacked layout at init time if required

convolution.py               DiscreteContinuousConvS2._forward() dispatch:
  _kpacked_ok = optimized_kernel and psi_kpacked_K_pad in (8,16)
                and x.dtype in (float16, bfloat16)
  fused + kpacked  →  _disco_s2_fused_conv_kpacked
  fused only       →  _disco_s2_fused_conv_optimized
  kpacked only     →  _disco_s2_contraction_kpacked
  CSR default      →  _disco_s2_contraction_optimized / torch

distributed_convolution_kernels.py   mirrors serial dispatch for a2a paths
distributed_convolution.py           builds kpacked buffers in _build_local_psi
```

### Key file locations

| Purpose | Path |
|---------|------|
| PyTorch reference kernels | `torch_harmonics/disco/kernels_torch/disco_torch.py` |
| CUDA kernel headers | `torch_harmonics/disco/optimized/kernels_cuda/disco_cuda.cuh` |
| CSR forward kernel | `torch_harmonics/disco/optimized/kernels_cuda/disco_cuda_fwd.cu` |
| CSR backward kernel | `torch_harmonics/disco/optimized/kernels_cuda/disco_cuda_bwd.cu` (BC_TILE optimized) |
| SM_90 kpacked kernel | `torch_harmonics/disco/optimized/kernels_cuda/disco_cuda_fwd_dense_kpacked_sm90.cu` |
| PTX helpers (WGMMA) | `torch_harmonics/disco/optimized/kernels_cuda/disco_cuda_ptx.cuh` |
| CPU OpenMP forward kernel | `torch_harmonics/disco/optimized/kernels_cpu/disco_cpu_fwd.py` |
| CPU OpenMP backward kernel | `torch_harmonics/disco/optimized/kernels_cpu/disco_cpu_bwd.py` |
| C++ interface | `torch_harmonics/disco/optimized/disco_interface.cpp` |
| Python dispatch | `torch_harmonics/disco/optimized/disco_optimized.py` |
| Serial conv (dispatch) | `torch_harmonics/disco/convolution.py` |
| Distributed conv | `torch_harmonics/distributed/distributed_convolution.py` |
| Distributed kernels | `torch_harmonics/distributed/kernels/distributed_convolution_kernels.py` |
| Build config | `setup.py` (CUDA sources list) |
| Serial tests | `tests/test_convolution.py` (`TestKpackedPath`) |
| Distributed tests | `tests/test_distributed_convolution.py` |

---

## Adding a new kernel variant — checklist

Walk through every layer in order. Each layer has a hard dependency on the previous one.

### 1. CUDA kernel (`.cu` / `.cuh`)

- Declare the host function in `disco_cuda.cuh`.
- Implement in a new `.cu` file named `disco_cuda_{direction}_{arch}.cu`
  (direction: `fwd` or `bwd`; arch: `sm90`, `sm100`, …).
- Add a `#if defined(__CUDA_ARCH_FEAT_SM{NN}_ALL)` guard so the kernel body
  compiles to empty on other arches — the host launcher enforces the arch check
  at runtime via `TORCH_CHECK(props.major == N, …)`.
- Add the new `.cu` to `setup.py` CUDA sources.

### 2. C++ interface (`disco_interface.cpp`)

- Register the new op schema in `TORCH_LIBRARY("disco_kernels", m)`.
- Implement the CUDA dispatch in `TORCH_LIBRARY_IMPL("disco_kernels", CUDA, m)`.

### 3. Python dispatch layer (`disco_optimized.py`)

Every op that participates in autograd needs **all four** of these:

| What | How |
|------|-----|
| Fake kernel (shape inference) | `@torch.library.register_fake("disco_kernels::op_name")` |
| AutocastCUDA handler | `@torch.library.impl("disco_kernels::op_name", "AutocastCUDA")` — cast float inputs to `torch.get_autocast_dtype("cuda")`, call `.default` inside `autocast(enabled=False)` |
| Backward | `torch.library.register_autograd(…)` for simple ops; `torch.autograd.Function` subclass when forward and backward use *different* kernel paths |
| Public wrapper | thin Python function that calls `op.apply(…)` or `op.default(…)` |

**When to use `autograd.Function` vs `register_autograd`:**
Use `autograd.Function` whenever forward and backward run *different* kernels — the canonical case is WGMMA forward + CSR backward or when composing ops with other PyTorch kernels, for example the fused kernel variants. `register_autograd` assumes the backward is structurally parallel to the forward; mixing kernel types breaks that assumption.

**AutocastCUDA pattern** (copy-paste template):
```python
@torch.library.impl("disco_kernels::my_op", "AutocastCUDA")
def _(inp, ...):
    cast_dtype = torch.get_autocast_dtype("cuda")
    with torch.amp.autocast("cuda", enabled=False):
        return my_op(inp.to(cast_dtype), ...)
```

### 4. Serial conv dispatch (`convolution.py`)

- Add `_kpacked_ok` (or equivalent arch gate) before the dispatch branch.
- The 4-way dispatch pattern (fused×kpacked) should be kept symmetric:
  ```
  fused + new_path → new fused variant
  fused only       → existing fused CSR variant
  new_path only    → new unfused variant
  default          → CSR
  ```

### 5. Distributed path (`distributed_convolution.py` + `_kernels.py`)

The distributed path **must mirror** the serial dispatch exactly:

1. In `_build_local_psi`: call the same psi-preparation helpers as the serial
   `_build_local_psi` (e.g. `pack_psi_dense` + `_maybe_kpack_psi` for kpacked).
   Set `self.{feature}_K_pad = None` unconditionally before the `if optimized_kernel:` block so `forward` always has the attribute.
2. In `distributed_convolution_kernels.py`: add the same `_feature_ok` guard and
   dispatch branches as the serial path.
3. In `forward`: thread the new buffers through via `getattr(self, "buf", None)`.

---

## Backward direction — gather vs scatter

This is the key architectural insight for DISCO; get it wrong and you silently
lose performance or correctness.

- **CSR forward** (`disco_kernels::forward`): inp → K-expanded. Gather direction — each output pixel reads from a bounded neighbourhood. *Input-pixel-parallel*, no atomics.
- **CSR backward** (`disco_kernels::backward`): K-expanded grad → inp grad. Also gather direction — each input pixel accumulates from its neighbourhood. *Input-pixel-parallel*, no atomics. This is the correct backward for any convolution with overlapping support sets, for the same reason cuDNN uses implicit GEMM not col2im scatter.
- **WGMMA forward_kpacked**: accelerated CSR forward using Tensor Cores, restricted to SM_90a + bf16/fp16 + K_PAD ∈ {8, 16}.
- **WGMMA backward (retired)**: scatter direction — output → input — causes massive atomicAdd contention when support sets overlap. Do NOT reintroduce. This is why it is currently not in the active code base.

The `_DiscoKpackedFn` / `_DiscoKpackedFusedFn` autograd.Function classes exist precisely to pair WGMMA forward with CSR backward.

---

## BC_TILE optimization (CSR backward)

The CSR backward had poor FMA utilisation (L1/TEX bound, ~12% FMA) because each CTA covered one channel and redundantly loaded the psi index arrays. BC_TILE amortises the index loads: one CTA processes BC_TILE channels, loading psi indices once.

- BC_TILE is selected at runtime: `BC >= 8 → 8`, `BC >= 4 → 4`, else `1`.
- Non-divisible BC is handled by ceiling division + `if (bc >= BC_total) continue` guards; invalid slots get zero-filled registers (harmless FMAs).
- `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)` is required for BC_TILE=8 (49152 bytes, exactly at the default carveout limit).

---

## kpacked layout

`_maybe_kpack_psi` converts `pack_psi_dense` output to the K-packed format:

```
pack_idx    [Ho, NBR_PAD, 2]      int64   (shared across all K — verify with torch.equal)
pack_val    [Ho, NBR_PAD, K_PAD]  fp32    (permuted + zero-padded to K_PAD = ceil(K/8)*8)
pack_count  [Ho]                  int64
```

Returns `None` if the per-K support sets differ (layout mismatch; CSR fallback activates). `K_PAD` must be 8 or 16 for the WGMMA kernel; store as `psi_kpacked_K_pad` on the module.

---

## Test coverage requirements

Every kernel variant needs tests in **both** the enabled and disabled states. The monkeypatch pattern forces the fallback path without needing a non-Hopper machine:

```python
conv.psi_kpacked_K_pad = 24   # ineligible → _kpacked_ok = False → CSR fallback
```

### Serial tests (`TestKpackedPath` in `test_convolution.py`)

| Test | Gate | What it checks |
|------|------|----------------|
| `test_kpacked_forward_activates_on_sm90` | SM_90 | kpacked path selected, output dtype preserved |
| `test_kpacked_fused_matches_unfused` | SM_90 | fused==unfused output + grad |
| `test_kpacked_bwd_bc_tile_boundaries` | SM_90 | BC_TILE=1/4/8 correctness vs fp32 |
| `test_kpacked_disabled_for_unsupported_k_pad` | none | K_PAD=24 → no crash (CSR fallback) |
| `test_kpacked_disabled_fused_fallback` | none | fused=True + K_PAD=24 → CSR fused path, fwd+bwd match |
| `test_kpacked_opcheck` | SM_90 | PT2 opcheck contract |

### Distributed tests (`TestDistributedDiscreteContinuousConvolution`)

The parameterised AMP rows (dtype=float16/bfloat16) exercise kpacked-enabled on Hopper.
Three dedicated methods cover the fallback:

| Test | Covers |
|------|--------|
| `test_kpacked_fallback_bf16_unfused` | fused=False + K_PAD=24 |
| `test_kpacked_fallback_bf16_fused` | fused=True + K_PAD=24 |
| `test_kpacked_fallback_fp16_unfused` | fp16 + K_PAD=24 |

---

## Profiling

### Serial Benchmarks

The code base currently does not have a benchmark. For writing profiling scripts for serial kernels, stick to a minimal implementation. Aim at running the kernel in question in isolation, comparing to existing kernels when possible. Incorporate all 3 precisions (fp32, bf16, fp16) into the benchmark and also compare the results of the kernel against its fp32 variant. Relevant shape combinations are:

```python
CONFIGS = {
    "self_256x360x720":  dict(in_channels=256, out_channels=256, in_shape=(360, 720), out_shape=(360, 720), grid_in="legendre-gauss", grid_out="legendre-gauss", theta_cutoff=0.017, kernel_shape=(3,3), basis_type="harmonic", basis_norm_mode="mean"),
    "self_512x360x720":  dict(in_channels=512, out_channels=512, in_shape=(360, 720), out_shape=(360, 720), grid_in="legendre-gauss", grid_out="legendre-gauss", theta_cutoff=0.017,
    kernel_shape=(3,3), basis_type="harmonic", basis_norm_mode="mean"),
    "down_73x721x1440":  dict(in_channels=80, out_channels=512,, in_shape=(721, 1440), out_shape=(360, 720), grid_in="equiangular", grid_out="legendre-gauss", theta_cutoff=0.017,
    kernel_shape=(3,3), basis_type="harmonic", basis_norm_mode="mean"),
}
```

The specific type of grid does not affect kernel performance but might affect accuracy.
For detailed profiling, run ncu

```bash
# Capture light sections (avoid --set full which hangs on multi-replay kernels)
ncu --kernel-name disco_bwd_blk_k \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    -o profiles/my_profile \
    python performance/disco/my_benchmark.py
```

Key metrics to watch:
- **FMA utilisation** (ComputeWorkloadAnalysis) — target near-saturation; low FMA on an L1-bound kernel means the index load is the bottleneck → BC_TILE fixes this.
- **L1/TEX hit rate** (MemoryWorkloadAnalysis) — high is good for psi values that fit in L2.
- **DRAM utilisation** — should be low for typical model sizes (psi fits in L2).

### Distributed Benchmarks
Write a small benchmark using mock communicators, imitating the actual communication pattern. Allow for splitting in latitude (h) and longitude(w). Only measure the serial custom kernel performance on the individual shards for the various precisions. This yields a floor for the expected kernel execution time.
