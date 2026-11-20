# Grouped MoE for V4.1-Flash and GLM-5.3-Flash

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make one RTX PRO 6000 expert layer for DeepSeek-V4.1-Flash and GLM-5.3-Flash a single grouped low-precision launch, instead of a per-expert dequant-to-BF16 loop.

**Architecture:** V4.1 already has the grouped kernel, `v4_moe_forward_sm120_pro6000`. The serving wrapper in moe-store still calls `v4_expert_forward` once per expert, after a host `bincount().tolist()`. GLM has a per-expert native FP8 kernel (`fused_fp8_moe_ffn_sm120_pro6000_into`) that skips the BF16 copy and still launches once per expert. The missing piece is a grouped block-128 FP8 layer kernel with the same pointer-table contract, called once from `ExpertDispatcher`.

**Tech Stack:** CUDA sm_120a block-scaled MMA, the existing `_v4_fp4` and `_store` extensions, moe-store `official_offload_adapter.py`, `benchmarks/kernel_audit` on GPU 5.

**Measured gap (2026-11-11, GPU 5, one layer):** GLM batch 64 is 26.2 ms here and 4.5 ms in vLLM 0.30.0 / SGLang 0.5.23. V4.1 batch 64 is 43.4 ms of 244 serial `v4_expert_forward` calls, about 178 µs each. Tracking issue #357.

The two tracks ship separately. Track A does not wait on Track B.

---

### Task 1: Call the existing V4 grouped kernel from the layer

**Files:**
- Modify: `/mnt/raid0nvme0/leyang/moe-store/moe_store/wrappers/deepseek_v4/official_offload_adapter.py` (the `make_forward` loop at lines 226–249)
- Test: `moe-store` unit test next to that adapter, or `tests/python/unit/test_v4_fp4_hw_gate.py` if the hook is mirrored here
- Gate already registered from `moe_infinity/runtime/engine_ops.py:32-94`

The installed adapter always does this, which is the timed path:

```python
counts = torch.bincount(indices.flatten(), minlength=n_routed).tolist()
active = [e for e in range(start_idx, end_idx) if counts[e] > 0]
for e in active:
    idx, top = torch.where(indices == e)
    out = ext.v4_expert_forward(x[idx], ...)
    y[idx] += out.float() * weights[idx, top, None].float()
```

- [ ] **Step 1: Write the failing test**

In the moe-store tree, add a test that a fake extension records one grouped call and zero per-expert calls when the gate returns true.

```python
def test_pro6000_forward_is_one_grouped_call(monkeypatch):
    calls = []

    class Ext:
        def v4_moe_forward_sm120_pro6000(self, x, ids, w, table, inter, limit, act_block):
            calls.append(("grouped", tuple(x.shape), act_block))
            return torch.zeros(x.shape, dtype=torch.float32)

        def v4_expert_forward(self, *args):
            calls.append(("expert",))
            raise AssertionError("per-expert path")

    # build the smallest moe whose forward is make_forward(...), with
    # grouped_native_available patched to True and store.get returning
    # CUDA tensors kept alive for the pointer table.
    y = forward(x, input_ids)
    assert calls == [("grouped", (x.shape[0], dim), 32)]
    assert y.shape == x.shape
```

V4.1-Flash must pass `act_block=32`. The pybind default is 128, which is V4-Flash (`v4_fp4_binding.cpp`, `py::arg("act_block") = 128`).

- [ ] **Step 2: Run the test and confirm it fails**

Run: `pytest tests/python/unit/test_v4_grouped_forward.py::test_pro6000_forward_is_one_grouped_call -q`

Expected: FAIL because `make_forward` still calls `v4_expert_forward`.

- [ ] **Step 3: Replace the expert loop when the gate passes**

Inside `make_forward`, before the Python expert loop:

```python
ext = _load_native_fp4()
use_grouped = (
    use_native
    and hasattr(ext, "v4_moe_forward_sm120_pro6000")
    and v4_moe_sm120_pro6000_supported(x.device)
)
if use_grouped:
    # indices, weights stay on device. No .tolist(), no torch.where.
    table = _expert_pointer_table(store, layer_id, start_idx, end_idx)
    y = ext.v4_moe_forward_sm120_pro6000(
        x, indices, weights, table, moe.moe_inter_dim,
        float(swiglu_limit), 32,
    )
    if official_module.world_size > 1:
        official_module.dist.all_reduce(y)
    y = y + shared(x)
    return y.type_as(x).view(shape)
```

`_expert_pointer_table` returns a CUDA int64 `[E, 6]` of `(w1, s1, w3, s3, w2, s2)` data pointers, the contract in `v4_fp4_binding.cpp`. Prefetch the experts this token can touch before building the table, using the ids already on device (`torch.unique` is allowed; copying the count vector to the host is not).

On any other GPU, leave the existing `v4_expert_forward` loop unchanged.

- [ ] **Step 4: Re-run the test**

Expected: PASS. Also run `pytest tests/python/unit/test_v4_fp4_hw_gate.py -q` in this repo. The 5090 and H100 cases must stay on Path B.

- [ ] **Step 5: Time the grouped kernel on GPU 5**

The audit harness already builds one grouped call (`_v4_fp4_native_layer_builder` in `benchmarks/kernel_audit/suites.py`). Run, with the extension built so `v4_moe_forward_sm120_pro6000` exists:

```bash
cd /mnt/raid0nvme0/leyang/moeinf-wt/v41-glm53-audit
CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m benchmarks.kernel_audit.run_cases \
  --suite expert_layer --mode time --iters 10 \
  --models deepseek-v4.1-flash \
  --impl '^v4_fp4_native$' \
  --match '\|(b1|b64|1K|12K)\|' \
  --out benchmarks/kernel_audit/results/framework_bench/v41_grouped.jsonl
```

Expected: batch 64 well below the 43.4 ms Path B number. Record the four medians in `REPORT_V41_GLM53_VS_VLLM_SGLANG.md`.

- [ ] **Step 6: Commit** in moe-store and, if `engine_ops.py` changed, in this repo.

```bash
git commit -m "$(cat <<'EOF'
perf(v4): run V4.1-Flash routed experts as one grouped FP4 launch

Path B's per-expert dequant is 178 µs times the active-expert count.
The sm_120a grouped kernel already exists; the layer was not calling it.

EOF
)"
```

---

### Task 2: Grouped block-128 FP8 for GLM-5.3-Flash

**Files:**
- Create: `extensions/kernel/fp8_moe/fp8_moe_grouped_sm120_pro6000.cu`
- Create: `extensions/kernel/fp8_moe/fp8_moe_grouped_sm120_pro6000.h`
- Modify: `core/python/py_archer_prefetch.cpp` (bind next to `fused_fp8_moe_ffn_sm120_pro6000_into`)
- Modify: `core/parallel/expert_dispatcher.cpp` (one layer call on the RTX PRO 6000 FP8 path)
- Test: `tests/python/ops/test_fp8_grouped_expert.py`

Per-expert `fused_fp8_moe_ffn_sm120_pro6000_into` stays. It removes the BF16 dequant and does not remove the 241 launches at batch 64. The grouped kernel is what the 4.5 ms column is.

- [ ] **Step 1: Write the failing parity test**

```python
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")


def test_grouped_fp8_matches_per_expert_native():
    from moe_infinity import _store

    if not hasattr(_store, "fp8_moe_grouped_forward_sm120_pro6000"):
        pytest.fail("grouped FP8 entry is not built")
    T, H, I, E, K = 4, 256, 256, 8, 2
    torch.manual_seed(0)
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, E, (T, K), device="cuda", dtype=torch.int32)
    wts = torch.full((T, K), 1.0 / K, device="cuda")
    weights, scales = [], []
    for _ in range(E):
        weights.append([
            torch.randint(0, 120, (n, k), device="cuda", dtype=torch.uint8)
            for n, k in ((I, H), (I, H), (H, I))
        ])
        scales.append([
            torch.rand((n + 127) // 128, (k + 127) // 128, device="cuda")
            for n, k in ((I, H), (I, H), (H, I))
        ])
    table = torch.tensor(
        [[t.data_ptr() for t in (w[0], s[0], w[1], s[1], w[2], s[2])]
         for w, s in zip(weights, scales)],
        dtype=torch.int64, device="cuda",
    )
    out = _store.fp8_moe_grouped_forward_sm120_pro6000(
        x, ids, wts, table, I
    )
    acc = torch.zeros(T, H, device="cuda", dtype=torch.float32)
    for slot in range(K):
        for token in range(T):
            e = int(ids[token, slot])
            one = _store.fused_fp8_moe_ffn_sm120_pro6000_into(
                x[token:token + 1],
                weights[e][0], scales[e][0],
                weights[e][1], scales[e][1],
                weights[e][2], scales[e][2],
                torch.empty(1, H, device="cuda", dtype=torch.bfloat16),
            )
            acc[token] += one.float() * wts[token, slot]
    torch.testing.assert_close(out, acc, rtol=2e-2, atol=2e-2)
```

H and I in the test are 256 so the block-128 scales are 2×2. The production shape (H=4096, I=2048, E=288, top-8) is the timing case, not this test.

- [ ] **Step 2: Run it and confirm failure**

Run: `CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m pytest tests/python/ops/test_fp8_grouped_expert.py -q`

Expected: FAIL with `grouped FP8 entry is not built`.

- [ ] **Step 3: Add the entry and the kernel**

Host signature, same pointer order as the V4 table but three weight/scale pairs (gate, up, down):

```cpp
// fp8_moe_grouped_sm120_pro6000.h
void fp8_moe_grouped_forward_sm120_pro6000_cuda(
    const at::Tensor& x,          // bf16 [T, H]
    const at::Tensor& topk_ids,   // int32 [T, K]
    const at::Tensor& topk_w,     // fp32 [T, K]
    const at::Tensor& table,      // int64 [E, 6] = w_g, s_g, w_u, s_u, w_d, s_d
    int64_t inter,
    at::Tensor& output);          // fp32 [T, H]
```

Device side, follow `v4_moe_forward_sm120_pro6000_cuda` in `extensions/kernel/v4_fp4/v4_fp4_moe_sm120_pro6000.cu`:

1. One prep kernel: histogram and exclusive scan of `topk_ids`, tile list, no host sync.
2. If mean rows per expert ≤ 8, one persistent kernel over all experts (the decode grid of 188 CTAs used by the V4 kernel). Otherwise the prefill grouped GEMM.
3. Each MMA reads e4m3 weights and the fp32 128×128 scales. Do not write a BF16 copy of the expert.
4. Epilogue is the GLM SiLU (`silu(gate) * up`), multiply by `topk_w`, and accumulate in fp32 in ascending expert order.
5. Refuse the launch unless `fp8_moe_sm120_pro6000_supported(hidden, inter, device)` is true. Other devices keep `DequantFp8Params`.

Bind it beside the existing per-expert symbol in `core/python/py_archer_prefetch.cpp`.

- [ ] **Step 4: Re-run the parity test**

Expected: PASS on GPU 5.

- [ ] **Step 5: Call it once per layer from the dispatcher**

In `ExpertDispatcher`, when every expert of the layer is FP8 and `fp8_sm120_pro6000_` is true, do not enqueue `MoEMLP::forward` per expert. Build the `[E, 6]` table from the resident FP8 tensors and scales, and launch `fp8_moe_grouped_forward_sm120_pro6000_cuda` once with the layer's token matrix and top-k. The per-expert function remains the fallback when the gate is false (`MOE_FP8_NATIVE=0` or a non-PRO-6000 GPU).

- [ ] **Step 6: Retime the four points**

```bash
CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m benchmarks.kernel_audit.run_cases \
  --suite expert_layer --mode time --iters 10 \
  --models glm-5.3-flash \
  --impl '^fp8_grouped$' \
  --match '\|(b1|b64|1K|12K)\|' \
  --out benchmarks/kernel_audit/results/framework_bench/glm_grouped.jsonl
```

Add the `fp8_grouped` impl in `benchmarks/kernel_audit/suites.py` next to `_fp8_native_builder`, using the same layer-builder pattern as `_v4_fp4_native_layer_builder`.

Expected on GPU 5: batch 1 near 0.2 ms and batch 64 near 5 ms, the grouped column in the 2026-11-11 table. If batch 64 is still tens of milliseconds, the launch is still per expert.

- [ ] **Step 7: Commit**

```bash
git commit -m "$(cat <<'EOF'
perf(glm): one grouped FP8 launch per GLM-5.3-Flash layer

The per-expert dequant costs about 440 µs an expert, so batch 64 is 26 ms.
A single block-128 FP8 kernel over the expert table is the path vLLM and
SGLang already take on this GPU.

EOF
)"
```

---

### Task 3: Record the new times

**Files:**
- Modify: `benchmarks/kernel_audit/REPORT_V41_GLM53_VS_VLLM_SGLANG.md` (the 2026-11-11 table)

- [ ] **Step 1:** Fill the V4.1 vLLM/SGLang cells only if those kernels were actually run. Do not copy the GLM Triton numbers across.
- [ ] **Step 2:** Replace the MoE-Infinity column with the grouped medians and leave the old dequant numbers in one sentence so the before/after stays readable.
- [ ] **Step 3:** Comment the four medians on #357.
