# Model compatibility matrix

This guide is the source of truth for the families registered in
`moe_infinity/common/constants.py` and the runtime adapters wired in
`moe_infinity/runtime/model_offload.py`.

Legend:
- `validated`, covered by a repo test on a real checkpoint or a close smoke harness
- `implemented/experimental`, the code path exists, but the repo only has unit or fixture coverage
- `not validated`, the repo has no harness for the claim
- `unsupported`, no runtime path or a fail-fast guard blocks it

Conditional registry keys, `deepseekv4`, `qwen3_5`, and `glmmoedsa`, only
register when the matching Transformers class imports cleanly. The repo floor
is `transformers>=5.3.0,<6`, but some families need newer builds.

| Family / HF class | Example checkpoint | Minimum Transformers | Sync generation | Continuous serving | Expert offload / quantization | Speculative decoding | Validated topology | Limitations |
|---|---|---|---|---|---|---|---|---|
| DeepSeek-V2 (`DeepseekV2ForCausalLM`) | `deepseek-ai/DeepSeek-V2-Lite-Chat` | `>= 5.3.0` | validated | implemented/experimental | validated | not recorded | `1x GPU` | FlashAttention is excluded in the offload path, eager attention is used in the consistency harness |
| DeepSeek-V3 (`DeepseekV3ForCausalLM`) | `deepseek-ai/DeepSeek-V3` | `>= 5.3.0` | implemented/experimental | implemented/experimental | implemented/experimental | not recorded | Not recorded | Only routing and paged-attention parity are covered in repo tests |
| DeepSeek-V4 (`DeepseekV4ForCausalLM`) | `deepseek-ai/DeepSeek-V4-Flash` | Not recorded, guarded import | validated for the official offload path | not validated | validated | unsupported | `4x GPU mp4` | The official checkpoint must be mp-sharded; repo validation covers the mp4 path, while mp1 is not covered by repo tests |
| Mixtral (`MixtralForCausalLM`) | `mistralai/Mixtral-8x7B-Instruct-v0.1` | `>= 5.3.0` | implemented/experimental | implemented/experimental | implemented/experimental | not recorded | Not recorded | No real-model harness is recorded in this repo |
| Qwen3 (`Qwen3MoeForCausalLM`) | `Qwen/Qwen3-30B-A3B` | `>= 5.3.0` | validated | implemented/experimental | validated | not recorded | `1x GPU` | `Qwen3PagedAttention` exists, but the repo does not ship a real-model serving harness for it |
| Qwen3.5 (`Qwen3_5MoeForConditionalGeneration`) | `Qwen/Qwen3.5-35B-A3B` | `>= 5.12` | validated on tiny fixtures through deprecated `MoE.generate()` | serving wiring covered by tiny fixtures; no real-checkpoint serving validation recorded | validated on tiny fixtures | validated on tiny fixtures | Tiny CPU fixtures only | Text-only path, vision and MTP weights stay unused, batch>1 speculative draft is blocked in deprecated `MoE.generate()` |
| GLM-5.2 (`GlmMoeDsaForCausalLM`) | `zai-org/GLM-5.2-FP8` | `>= 5.12` | validated | validated on the tiny serving harness | validated | built-in MTP only | `1x GPU` | Native engine is forced off, no GLM DFlash drafter is registered, and non-zero temperature falls back to greedy in MTP |
| GPT-OSS (`GptOssForCausalLM`) | `openai/gpt-oss-20b` | `>= 5.3.0` | validated for 20B | validated for 20B | validated | validated for greedy batch-1, sampled batch-1 implemented/experimental | GPU-gated | Route-ahead is not wired because the path never attaches an `expert_executor`; batch>1 `speculative_draft` is blocked |
| DBRX (`DbrxForCausalLM`) | `databricks/dbrx-instruct` | `>= 5.3.0` | implemented/experimental | not validated | implemented/experimental | not recorded | Not recorded | Registry and adapter code exist, but there is no real-model harness in repo tests |
| Jamba (`JambaForCausalLM`) | `ai21labs/Jamba-*` | `>= 5.3.0` | implemented/experimental | not validated | implemented/experimental | not recorded | Not recorded | Registry and adapter code exist, but there is no real-model harness in repo tests |
| OLMoE (`OlmoeForCausalLM`) | `allenai/OLMoE-*` | `>= 5.3.0` | implemented/experimental | not validated | implemented/experimental | not recorded | Not recorded | Registry and adapter code exist, but there is no real-model harness in repo tests |
| NLLB-MoE (`NllbMoeForConditionalGeneration`) | `facebook/nllb-moe-54b` | `>= 5.3.0` | implemented/experimental | not validated | implemented/experimental | not recorded | Not recorded | Encoder-decoder sparse parsing is covered, but there is no repo end-to-end harness |
| OPT (`OPTForCausalLM`) | Not recorded | `>= 5.3.0` | unsupported | unsupported | unsupported | unsupported | Not recorded | Registry entry only, no adapter or test coverage beyond parser dispatch |

Repo evidence:
- `moe_infinity/common/constants.py` for the registry and conditional imports
- `moe_infinity/runtime/model_offload.py` for the adapter wiring
- `tests/test_gpt_oss_*`, `tests/python/v4/*`, `tests/python/integration/test_glm_*`,
  `tests/python/unit/test_qwen3_5_moe.py`, `tests/python/unit/test_model_registry.py`,
  `tests/python/unit/test_glm_*`, and `tests/python/integration/test_model_consistency.py`

Use `Not recorded` when the repo has no direct evidence for a version,
topology, or capability claim. Use `implemented/experimental` when the code path
exists but only unit or fixture tests cover it.
