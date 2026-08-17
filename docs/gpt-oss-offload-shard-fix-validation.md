# GPT-OSS offload: shard-agnostic expert expansion — validation

## Root cause

`_expand_gpt_oss_packed_experts` (`moe_infinity/runtime/model_offload.py`) ran
inside the per-shard checkpoint loop (`for ckpt in self.ckpt_files`), operating
on a single shard's `state_dict` slice. It discovered layers via
`gate_up_proj_blocks` and required all six packed components
(`gate_up_proj_{blocks,scales,bias}`, `down_proj_{blocks,scales,bias}`) to be
co-resident in that one slice.

`openai/gpt-oss-20b` splits a layer's components across safetensors shards. Per
`model.safetensors.index.json`:

| key                                              | shard |
| ------------------------------------------------ | ----- |
| `model.layers.6.mlp.experts.gate_up_proj_bias`   | 00001 |
| `model.layers.6.mlp.experts.gate_up_proj_blocks` | 00002 |
| `model.layers.6.mlp.experts.gate_up_proj_scales` | 00002 |
| `model.layers.6.mlp.experts.down_proj_*`         | 00002 |

Expanding shard 00002 raised
`Incomplete GPT-OSS packed expert layer model.layers.6.mlp.experts: ['gate_up_proj_bias']`.
A secondary bug: shard 00001 (bias only, no `gate_up_proj_blocks`) was skipped
entirely, so the bias was never expanded.

## Fix

Expand each packed component independently by suffix, so every per-shard slice
expands the components it holds. Global per-`(layer, expert)` completeness is
still enforced over the merged `name_id_map` by `_gpt_oss_expert_groups`, so
genuinely corrupt checkpoints are still rejected. Expansion stays zero-copy
(per-expert views alias the packed storage).

Tests (`tests/test_gpt_oss_offload_topology.py`):
- `test_expansion_handles_components_split_across_shards` — split slices expand.
- `test_incomplete_checkpoint_rejected_globally` — corrupt checkpoint rejected
  at the merged-`name_id_map` layer.

## Validation

Unit suite (green):
`test_gpt_oss_offload_topology.py`, `test_gpt_oss_config.py`,
`tests/python/unit/test_gpt_oss_mxfp4_dispatch.py`,
`test_gpt_oss_offload_policy.py`, `test_gpt_oss_wrapper.py` — 29 passed.
`ruff` clean; LSP clean.

Real model (`openai/gpt-oss-20b`, offload on, `device_memory_ratio` 0.2 and
0.75): all three shards load, expansion no longer raises, offload topology
builds (24 sparse layers; experts moved to CPU, 9153 MB + 3978 MB partitions),
`set_topology done`. The layer-6 loading blocker is resolved.

Resident reference (plain HF Transformers, experts resident):
`"The capital of France is Paris."` (coherent) — confirms the environment and
checkpoint are healthy outside MoE.

## Out-of-scope downstream blocker (not this fix)

With loading fixed, offloaded **generation** segfaults inside the native
post-forward offload hook `archer_engine.end`
(`_post_forward_module_hook`, `model_offload.py:1831`) during the GptOss expert
forward. This is independent of the expansion fix:

- It reproduces on a **cached** offload load
  (`Loading model from offload_path ...`), which never calls
  `_expand_gpt_oss_packed_experts`.
- It is identical across the native and HF (`use_native_engine=False`) engine
  paths and across `device_memory_ratio` values.
- It was previously unreachable: the unmodified stack cannot load gpt-oss-20b
  past layer 6, so generation was never exercised. The fix unmasks a pre-existing
  latent native-execution bug.

Greedy agreement vs resident could therefore not be measured. Per scope, the
native-engine segfault is reported rather than worked around; the agreement gate
was not weakened.
