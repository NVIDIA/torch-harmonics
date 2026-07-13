---
name: test-writing
description: >
  Guide for writing tests in torch-harmonics. Use this skill whenever the user
  asks to add tests, check or extend test coverage, write a new test class, test a new
  layer or kernel, or add distributed tests. Also use when the user asks about
  tolerance values, how to compare tensors, how to structure distributed test
  infrastructure, or how to test gradients. Covers serial tests in
  test_convolution.py and distributed tests in test_distributed_*.py.
---

# Test Writing Guide

## General principles

- **Match existing style exactly.** Before writing any test, read the relevant
  existing test class in `tests/` to pick up naming conventions, helper usage,
  and `parameterized.expand` patterns.
- **Numerical tests against a reference baseline.** Every forward and backward
  check must compare against a deterministic reference (typically fp32 serial or
  the torch reference path). Never just check shapes or dtypes alone.
- **Test every code path for every precision.** If a layer has 4 dispatch
  branches (e.g. fused×kpacked), run each one. If it supports fp32/fp16/bf16,
  run all three.

---

## Naming Conventions

The files for serial tests should be named `test_<feature_name>.py`
and distributed tests `test_distributed_<feature_name>.py` respectively.

---

## Default tolerances

| dtype | atol | rtol |
|-------|------|------|
| fp64 | 1e-6 | 1e-6 |
| fp32 | 1e-6 | 1e-5 |
| fp16 | 1e-3 | 1e-2 |
| bf16 | 1e-2 | 5e-2 |

Parameter gradients accumulate noise over batch × spatial dims, so loosen their
tolerance by a factor of 10–1000× relative to per-element output tolerances
(use judgment based on reduction size). For stacked/chained layers, loosen
further (2–5×).

---

## Comparison helpers

**Tensor comparisons** — always use `compare_tensors`. The first argument is a
name string used in failure messages; it is required:
```python
ok = compare_tensors("output", ref, got, atol=atol, rtol=rtol, verbose=verbose)
self.assertTrue(ok, "output")
```

**Python logic checks** — use plain `assertTrue` / `assertFalse`:
```python
self.assertIsNotNone(conv.psi_kpacked_K_pad)
self.assertIn(conv.psi_kpacked_K_pad, (8, 16))
self.assertEqual(out.dtype, torch.bfloat16)
```

Never use `torch.allclose` or `torch.testing.assert_close` directly — these
don't give the diagnostic output that `compare_tensors` provides.

### verbose flag

Every test method that calls `compare_tensors` must accept a `verbose=False`
parameter and forward it:

```python
def test_my_feature(self, ..., verbose=False):
    ok = compare_tensors("output", ref, got, atol=atol, rtol=rtol, verbose=verbose)
    self.assertTrue(ok)
```

In distributed tests, only rank 0 should print:
```python
verbose = verbose and self.world_rank == 0
```

---

## Autocast / AMP

Wrap forward (and any backward that needs grad under the same dtype) with the
`maybe_autocast` helper from `testutils`:

```python
with maybe_autocast(self.device.type, dtype):
    out = module(inp)
```

For AMP dtypes (float16, bfloat16), keep module parameters and inputs in
`float32` and let autocast downcast inside the forward. Match this pattern
exactly — do not manually cast module weights or inputs before the call.

```python
is_amp = dtype in (torch.float16, torch.bfloat16)
module_dtype = torch.float32 if is_amp else dtype
module = MyLayer(...).to(dtype=module_dtype, device=device)
inp = torch.randn(..., dtype=module_dtype, device=device)
with maybe_autocast(device.type, dtype):
    out = module(inp)
```

---

## What to test in every test method

Test **all three** of these in a single test method; do not split them across
separate tests unless there is a strong reason:

1. **Forward output** — compare against reference with `compare_tensors`.
2. **Input gradient** — run `.backward(ograd)`, compare `inp.grad` with `compare_tensors`.
3. **Weight (parameter) gradients** — compare each parameter's `.grad` with `compare_tensors`.
   See distributed section for the all-reduce needed there.

```python
inp.requires_grad_(True)
with maybe_autocast(device.type, dtype):
    out = module(inp)
ograd = torch.randn_like(out)
out.backward(ograd)

ok = compare_tensors("output",   ref_out,   out,      atol=atol, rtol=rtol, verbose=verbose)
self.assertTrue(ok, "output")
ok = compare_tensors("inp grad", ref_igrad, inp.grad, atol=atol, rtol=rtol, verbose=verbose)
self.assertTrue(ok, "inp grad")
if module.weight.grad is not None:
    ok = compare_tensors("weight grad", ref_wgrad, module.weight.grad,
                         atol=wgrad_atol, rtol=wgrad_rtol, verbose=verbose)
    self.assertTrue(ok, "weight grad")
```

---

## Serial test structure

```python
class TestMyFeature(unittest.TestCase):
    device = torch.device("cuda")  # or "cpu"

    @classmethod
    def setUpClass(cls):
        disable_tf32()

    def test_my_case(self, ..., verbose=False):
        set_seed(42)
        ...
```

`disable_tf32()` must be called in `setUpClass` for any test class that runs on
CUDA — TF32 reduces fp32 matmul precision and causes spurious failures at the
default tolerances.

For parameterised cases use `@parameterized.expand([...], skip_on_empty=True)`:

```python
@parameterized.expand([
    # [param1, param2, ..., atol, rtol]
    [64, 128, torch.float32, 1e-6, 1e-5],
    [64, 128, torch.bfloat16, 1e-2, 5e-2],
], skip_on_empty=True)
def test_my_feature(self, nlat, nlon, dtype, atol, rtol, verbose=False):
    ...
```

---

## Distributed test structure

### Module-level wireup (required in every distributed test file)

```python
_DIST_CTX = {}

def setUpModule():
    setup_module(_DIST_CTX)

def tearDownModule():
    teardown_module(_DIST_CTX)
```

### Class setup

```python
class TestMyDistributedFeature(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_class_from_context(cls, _DIST_CTX)
        disable_tf32()

    def setUp(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

    def tearDown(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
```

`setup_class_from_context` sets `cls.device`, `cls.world_rank`,
`cls.grid_size_h`, `cls.grid_size_w`, `cls.hrank`, `cls.wrank`,
`cls.h_group`, `cls.w_group` on the class.

`disable_tf32()` is called in `setUpClass` for the same reason as serial tests.

### Comparing distributed output against serial baseline

The pattern is always:
1. Build a serial (local) module and a distributed module with identical weights.
2. Run serial forward + backward on the full tensor.
3. Split the input, run distributed forward + backward on the local shard.
4. Gather distributed outputs/grads back to the full tensor.
5. Compare with `compare_tensors` on all ranks; use `reduce_success` so a
   failure on any rank fails the test.

```python
# 1. build modules
conv_local = MyLayer(**args).to(dtype=module_dtype, device=self.device)
conv_dist  = DistributedMyLayer(**args).to(dtype=module_dtype, device=self.device)

# 2. sync weights — every rank must have identical parameters
with torch.no_grad():
    conv_dist.weight.copy_(conv_local.weight)
    if hasattr(conv_dist, "bias") and conv_dist.bias is not None:
        conv_dist.bias.copy_(conv_local.bias)

# 3. serial fwd + bwd
inp_full = torch.randn((B, C, H, W), dtype=module_dtype, device=self.device)
inp_full.requires_grad_(True)
with maybe_autocast(self.device.type, dtype):
    out_full = conv_local(inp_full)
ograd_full = torch.randn_like(out_full)
out_full.backward(ograd_full)
igrad_full = inp_full.grad.clone()

# 4. distributed fwd + bwd on local shard
inp_local = self._split_helper(inp_full.detach().clone())
inp_local.requires_grad_(True)
with maybe_autocast(self.device.type, dtype):
    out_local = conv_dist(inp_local)
ograd_local = self._split_helper(ograd_full)
out_local.backward(ograd_local)
igrad_local = inp_local.grad.clone()
```

### Gather helpers

Use the `gather_tensor_hw` utility (from `testutils`) to gather distributed
tensors back to the full shape on all ranks:

```python
def _split_helper(self, tensor):
    return split_tensor_hw(tensor,
        hdim=-2, wdim=-1,
        hsize=self.grid_size_h, wsize=self.grid_size_w,
        hrank=self.hrank, wrank=self.wrank)

def _gather_helper_fwd(self, tensor, conv_dist):
    return gather_tensor_hw(tensor,
        hdim=-2, wdim=-1,
        hshapes=conv_dist.lat_out_shapes, wshapes=conv_dist.lon_out_shapes,
        hsize=self.grid_size_h, wsize=self.grid_size_w,
        hrank=self.hrank, wrank=self.wrank,
        hgroup=self.h_group, wgroup=self.w_group)

def _gather_helper_bwd(self, tensor, conv_dist):
    return gather_tensor_hw(tensor,
        hdim=-2, wdim=-1,
        hshapes=conv_dist.lat_in_shapes, wshapes=conv_dist.lon_in_shapes,
        hsize=self.grid_size_h, wsize=self.grid_size_w,
        hrank=self.hrank, wrank=self.wrank,
        hgroup=self.h_group, wgroup=self.w_group)
```

Use `_gather_helper_fwd` for output tensors (indexed by output spatial shape)
and `_gather_helper_bwd` for input-space tensors (input gradient).

### Comparing gathered results

Always use `reduce_success` so a test failure on any rank propagates to all:

```python
verbose = verbose and self.world_rank == 0

out_gather = self._gather_helper_fwd(out_local, conv_dist)
ok = compare_tensors("output", out_full, out_gather, atol=atol, rtol=rtol, verbose=verbose)
self.assertTrue(reduce_success(ok, self.device), "output")

igrad_gather = self._gather_helper_bwd(igrad_local, conv_dist)
ok = compare_tensors("gradients", igrad_full, igrad_gather, atol=atol, rtol=rtol, verbose=verbose)
self.assertTrue(reduce_success(ok, self.device), "gradients")
```

### Weight gradient reductions for distributed layers

Distributed layers compute parameter gradients as partial sums over the local
spatial tile. To compare against the serial reference, all-reduce across the
polar (h) and azimuth (w) groups independently:

```python
def _allreduce_param_grad(self, tensor):
    out = tensor.clone()
    if self.grid_size_h > 1:
        dist.all_reduce(out, group=self.h_group)
    if self.grid_size_w > 1:
        dist.all_reduce(out, group=self.w_group)
    return out
```

Then compare:
```python
param_tol = 1000.0 if not is_amp else 10.0   # param grads accumulate more noise
pg_atol, pg_rtol = atol * param_tol, rtol * param_tol
if conv_dist.weight.grad is not None:
    wgrad = self._allreduce_param_grad(conv_dist.weight.grad)
    ok = compare_tensors("weight grad", conv_local.weight.grad, wgrad,
                         atol=pg_atol, rtol=pg_rtol, verbose=verbose)
    self.assertTrue(reduce_success(ok, self.device), "weight grad")
```

The tighter tolerance factor for AMP is intentional: the base atol/rtol for
fp16/bf16 is already 3–4 orders of magnitude looser than fp32, so multiplying
by 1000× makes the bound meaninglessly large.

---

## Checklist before committing a new test

- [ ] `disable_tf32()` called in `setUpClass`
- [ ] `set_seed(N)` called at the top of every test method that uses random tensors
- [ ] `verbose=False` parameter on every method that calls `compare_tensors`
- [ ] All three of forward output, input gradient, weight gradients are tested
- [ ] All relevant precisions are covered (fp32, fp16, bf16 at minimum for CUDA layers)
- [ ] All dispatch branches are covered (e.g. fused=True/False, kpacked-enabled/disabled)
- [ ] Distributed tests use `setUpModule` / `tearDownModule` wireup
- [ ] Distributed weight grads are all-reduced before comparison
- [ ] `reduce_success` used for every assertion in distributed tests
- [ ] AMP tests use `maybe_autocast` — not manual `.to(dtype)` on module weights
