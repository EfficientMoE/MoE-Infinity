# Issue 221 Phase 6 — Lightning Indexer attention

Draft for review. This does not add a kernel and does not enable the native engine. Phase 4's eager run remains the correctness reference.

**Goal:** Serve MiniMax-M3 block-sparse attention with a kernel that matches the current Hugging Face eager path, while `use_native_engine` stays false for `minimax_m3_vl`.

**Why the native engine stays off:** `MoE.__init__` forces `use_native_engine = False` when `model_type` is `minimax_m3_vl`, and `arch == "minimaxm3"` forces `is_flash_attn_available = False`. Those guards exist because the native paged backend has no Lightning Indexer implementation. Re-enabling either flag is a separate change from landing a kernel inside the Hugging Face module.

## Decision checkpoint (not yet accepted)

Installed evidence is `transformers==5.12.0`. The shipped file is `/home/leyang/anaconda3/lib/python3.13/site-packages/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py`; the import path is `transformers.models.minimax_m3_vl.modeling_minimax_m3_vl`, the attention class is `MiniMaxM3VLAttention`, and the indexer is `MiniMaxM3VLIndexer`. The hook is `MiniMaxM3VLAttention.forward`, immediately after `block_indices = self.indexer(...)` and at the existing `attention_interface(self, query_states, key_states, value_states, attention_mask, ..., block_indices=block_indices)` call. The installed package therefore supplies the source of truth; no upstream fallback is needed.

Run this checkpoint command and paste its complete output on the implementation PR before choosing FlashInfer or a local kernel:

```bash
python -c "import inspect, transformers; from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig as C; from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLAttention as A, MiniMaxM3VLIndexer as I; c=C(); af,al=inspect.getsourcelines(A.forward); ix=inspect.getsource(I); print('transformers',transformers.__version__); print('module',inspect.getsourcefile(A)); print('hook',f'{A.__module__}.{A.__name__}.forward:{al}'); print('block_size',c.index_block_size,'topk_blocks',c.index_topk_blocks,'local_blocks',c.index_local_blocks,'is_causal',inspect.getsource(A.__init__).find('self.is_causal = True')>=0); print('index_output [B,S_q,index_topk_blocks], valid left-packed, -1 padded'); print('attention_inputs query_states,key_states,value_states [B,H,S,D]; block_indices selects key blocks'); print('causal_selection',next(x.strip() for x in ix.splitlines() if 'token_future =' in x)); print('kept_selection',next(x.strip() for x in ix.splitlines() if 'topk = min(' in x)); print('mask_expansion',next(x.strip() for x in ix.splitlines() if 'repeat_interleave(self.block_size' in x))"
```

Recorded output: block size `128`; causal token mask `key_position > position_id`; at most `16` selected key blocks per query with `1` immediately preceding/local block forced visible; valid block IDs are left-packed in `block_indices: [B,S_q,16]` and unused/future slots are `-1`. The indexer scores `idx_q: [B,4,S_q,128]` against `idx_k: [B,1,S_k,128]`, max-pools over heads and 128-token key blocks, and returns block IDs. Main attention consumes `query_states`, `key_states`, and `value_states: [B,H,S,D]`; eager expands selected IDs to `[B,1,S_q,S_k]`, composes that mask with causal/padding visibility, and passes it to the standard attention interface.

1. Choose FlashInfer versus a local kernel only after checking the recorded contract above against the available provider API; record that choice on the PR.
2. Eager stays the correctness reference and default. Add `ArcherConfig.enable_minimax_m3_indexer_kernel: bool = False`; copy it onto the HF model config before construction. The replacement attention reads only that flag.
3. The first patch hooks only `MiniMaxM3VLAttention.forward` through the existing install/restore lifecycle in `transformers_model_overrides.py`. It does not flip `use_native_engine` or clear the `arch == "minimaxm3"` flash-attn disable in `big_modeling.py`.
4. Qwen3.5 hybrid attention and GLM-DSA are not part of this patch.

## File map (after the checkpoint)

| File | Change |
|---|---|
| `moe_infinity/kernel/minimax_m3_sparse_attention.py` | Provider selected at the checkpoint; consume Q/K/V plus the recorded left-packed block IDs. Eager fallback remains. |
| `moe_infinity/utils/config.py` | Add `enable_minimax_m3_indexer_kernel`, default `False`. |
| `moe_infinity/entrypoints/big_modeling.py` | Copy the opt-in flag to `model_config`; keep native engine and flash-attn guards unchanged. |
| `moe_infinity/boundary/transformers_model_overrides.py` | Install/restore the `MiniMaxM3VLAttention` replacement at the exact hook above. |
| `examples/vl_example.py` | Add `--enable-minimax-m3-indexer-kernel` and pass it into the `MoE` config for the smoke gate. |
| `tests/python/unit/test_minimax_m3_indexer.py` | Tiny eager-versus-kernel and default-off no-op tests; skip only when the installed module is absent. |
| `benchmarks/minimax_m3/gates.py` | Pure 20% speedup gate and recorded sweep-row builder. |
| `benchmarks/minimax_m3/bench_indexer_attention.py` | No-download, module-level eager-versus-kernel prefill microbenchmark. |
| `tests/python/unit/test_minimax_m3_bench_gates.py` | CPU gate boundary, validation, row, and CLI-help tests. |

## Kernel performance microbenchmark

Installed `transformers==5.12.0` defaults are `hidden_size=6144`, `num_attention_heads=64`, `num_key_value_heads=4`, and `head_dim=128` (`index_n_heads=4`). Use this production geometry; set layer 0 to `"minimax_m3_sparse"` so `MiniMaxM3VLAttention(config, 0)` constructs its indexer.

1. First write `tests/python/unit/test_minimax_m3_bench_gates.py`: assert `kernel_speedup_passes(100.0, 80.0)` and reject `80.0 + 2**-20`; assert `speedup_row(8192, 100.0, 75.0) == {"seq_len": 8192, "t_eager_ms": 100.0, "t_kernel_ms": 75.0, "speedup": 4.0 / 3.0}`; loop over zero, negative, NaN, and infinity in every applicable argument and require `ValueError`; run the benchmark with `--help` in a subprocess and require exit 0. Run `pytest -q tests/python/unit/test_minimax_m3_bench_gates.py`; before implementation expect import failure.
2. Create `benchmarks/minimax_m3/gates.py`. `kernel_speedup_passes(t_eager_ms, t_kernel_ms)` validates finite positive timings and returns `t_kernel_ms <= 0.8 * t_eager_ms`. `speedup_row(seq_len, t_eager_ms, t_kernel_ms)` validates a finite positive sequence length and timings, then returns the exact dict above with `speedup=t_eager_ms / t_kernel_ms`.
3. Create `benchmarks/minimax_m3/bench_indexer_attention.py`. Its module docstring states the manual GPU command below. Seed random initialization, instantiate one BF16 eval-mode attention module without downloading a checkpoint, generate BF16 hidden states plus rotary inputs, and run prefill forward at `S in (2048, 8192, 16384)` through eager and the selected kernel with identical weights and inputs. Time each path using CUDA events with 5 warmups, 20 measured iterations, and synchronization around each sample; report `speedup_row` JSON per length. Only `S=8192` is enforced. `--enforce --t-eager-ms X --t-kernel-ms Y` prints `pass` or `fail` and exits 0 only on pass; `--help` exits 0 without CUDA.
4. Run `pytest -q tests/python/unit/test_minimax_m3_bench_gates.py`; expected: `4 passed`.

## QA gate

The CPU performance-gate tests always run:

```bash
pytest -q tests/python/unit/test_minimax_m3_bench_gates.py
```

Expected: `4 passed`. Manually run the GPU sweep and paste all three rows plus the enforcement output on the implementation PR:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/minimax_m3/bench_indexer_attention.py
```

The sweep prints a ready-to-run `--enforce --t-eager-ms X --t-kernel-ms Y` command populated from its `S=8192` row; run and paste it. The `S=2048` and `S=16384` rows are record-only. If the `S=8192` kernel time is not at most 80% of eager, the kernel does not land and the provider choice from the decision checkpoint is revisited.

Use `torch.manual_seed(0)`, BF16, `B=1`, `S_q=S_k=17`, `H_q=4`, `H_kv=2`, `D=16`, fixture block size `4`, top-k blocks `2`, and local blocks `1`. Build `block_indices` with the installed `MiniMaxM3VLIndexer`; compare the kernel output to installed HF eager attention using `rtol=1e-2, atol=1e-2`. The fixture must include `-1` padding and a query whose future block would otherwise score highest.

```bash
CUDA_VISIBLE_DEVICES=0 pytest -q tests/python/unit/test_minimax_m3_indexer.py
```

Expected: `2 passed`. `test_tiny_kernel_matches_hf_eager` checks the stated shape/tolerances and causal exclusion. `test_kernel_flag_defaults_off_and_is_noop` monkeypatches the kernel entry point to raise if called, omits `enable_minimax_m3_indexer_kernel`, runs `MiniMaxM3VLAttention.forward`, asserts no raise, and asserts `torch.equal` with the unpatched eager output.

Phase 4 cat smoke, greedy and kernel flag on:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/vl_example.py --checkpoint MiniMaxAI/MiniMax-M3-preview --image tests/fixtures/images/cat.jpg --prompt "What animal is in the picture? Answer with one lowercase word." --offload_dir /tmp/moe-minimax-m3-phase6 --device_memory_ratio 0.5 --max_new_tokens 16 --enable-minimax-m3-indexer-kernel --expect cat
```

Expected: output contains `ANSWER:` with case-insensitive `cat`, routed-expert evidence, and final line `PASS`.

## Non-goals

Paged native KV for M3, flash-attn on `minimaxm3`, MXFP8, video, and multi-image.
