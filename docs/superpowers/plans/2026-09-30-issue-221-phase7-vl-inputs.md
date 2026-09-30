# Issue 221 Phase 7 — multi-image, video rejection, VL prefix cache

Draft for review. This does not block closing #221. The maintainer close list is the Phase 4 baseline, Phase 5 MXFP8, and Phase 6 indexer. This phase is the remaining input follow-up from the issue body.

**Goal:** Make image-token identity safe in the prefix cache, reject video until a processor path exists, and lock multi-image admission with a test.

## What the code does now

- `_normalize_multimodal_messages` rewrites every content part whose type is `image_url`. A second image in the same message already goes through that loop. There is no two-image test.
- Any other part type, including `video_url`, is copied through unchanged and can reach the text tokenizer or the processor with no stable error.
- `Scheduler._allocate_with_prefix_cache` hashes prompt token ids only (`hash_block_tokens`). Vision placeholder ids repeat across different images, so two requests can share a block hash and reuse KV computed for the wrong pixels. Prefix caching itself is still the Qwen3 + FlashInfer path documented in `docs/serving.md`.

## Decision checkpoint (proposed)

Accept or replace before the code diff:

1. Multi-image needs a test, not a new parser, unless that test shows the normalizer or the scheduler token count dropping an image. The test posts two `image_url` parts at a VL mock and checks that admitted tokens equal the sum of the two expansions.
2. A `video_url` part returns HTTP 400 with a stable message that video input is unsupported. It is not tokenized as text and not passed to `apply_chat_template`.
3. When a request carries `multimodal_inputs`, prefix-cache lookup and insert are skipped for that request. Text-only requests keep today's hash. A cache key that hashes pixel bytes is a later patch.
4. Qwen3 text prefix caching stays enabled under `--enable-prefix-caching`. This phase adds a bypass for multimodal requests. It does not turn prefix caching on for MiniMax-M3.

## File map (after the checkpoint)

| File | Change |
|---|---|
| `moe_infinity/entrypoints/openai/api_server_v2.py` | 400 on `video_url`. Multi-image behavior covered by the test below. |
| `moe_infinity/engine/scheduler.py` | Skip `_allocate_with_prefix_cache` reuse and insert when the request has multimodal inputs. |
| `tests/python/serving/test_multimodal_inputs.py` | Two-image token sum. `video_url` returns 400. A multimodal request does not reuse a block hashed from the same placeholder ids. |

## QA gate

- CPU, no checkpoint: the three tests above.
- Text-only prefix-cache tests stay green. A request with no `multimodal_inputs` still hashes token ids.

## Non-goals

Video decode, pixel-hash cache keys, MXFP8, and Lightning Indexer.
