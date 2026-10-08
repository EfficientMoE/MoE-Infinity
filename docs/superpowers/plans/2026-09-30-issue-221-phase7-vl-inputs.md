# Issue 221 Phase 7 — multi-image, video rejection, VL prefix cache

Draft for review. This does not block closing #221. The maintainer close list is the Phase 4 baseline, Phase 5 MXFP8, and Phase 6 indexer. This phase is the remaining input follow-up from the issue body.

**Goal:** Make image-token identity safe in the prefix cache, reject video until a processor path exists, and lock multi-image admission with a test.

## What the code does now

- `_normalize_multimodal_messages` rewrites every content part whose type is `image_url`. A second image in the same message already goes through that loop. There is no two-image test.
- Any other part type, including `video_url`, is copied through unchanged and can reach the text tokenizer or the processor with no stable error.
- The OpenAI path is `api_server_v2._chat_request_inputs` → `ContinuousBatchingEngine.add_request` → `SequenceData.multimodal_inputs` → `serving.scheduler.Scheduler._acquire_prefill_lease`; committed blocks are published by `ContinuousBatchingEngine._publish_committed_prefix`. Both prefix operations currently use prompt token ids only, so repeated vision placeholders can reuse KV for different pixels.
- `moe_infinity/engine/scheduler.py` is not this path: its `Request` has no `multimodal_inputs`; do not change it.

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
| `moe_infinity/serving/scheduler.py` | In `_acquire_prefill_lease`, return `PrefixLease.empty()` when `sequence.multimodal_inputs is not None`; do not call `prefix_lease_provider.acquire_prefix_lease`. Text-only leasing is unchanged. |
| `moe_infinity/serving/engine.py` | In `_publish_committed_prefix`, return before `prefix_cache.insert` when `sequence.multimodal_inputs is not None`. Text-only publication is unchanged. |
| `tests/python/serving/test_multimodal_inputs.py` | Add the two-image token-sum, `video_url` 400, and serving-path prefix lookup/publication bypass tests. |

## QA gate

CPU only; no checkpoint download.

1. Two-image admission: extend `_vision_payload`/`_StubProcessor` with two `image_url` parts, verify both decoded images reach `apply_chat_template`, return both image expansions, and assert `scheduler_output.num_prefill_tokens` equals their summed expanded token count.
   - Run: `pytest -q tests/python/serving/test_multimodal_inputs.py::test_vl_request_admits_two_images_and_sums_expanded_tokens`
   - Expected: `1 passed`.
2. Video rejection: post a `video_url` content part, assert HTTP 400 with the stable unsupported-video message, and assert the processor/chat template was not called.
   - Run: `pytest -q tests/python/serving/test_multimodal_inputs.py::test_video_url_returns_400_without_processor_call`
   - Expected: `1 passed`.
3. Multimodal prefix safety: create a serving `SequenceData` with prompt ids matching a seeded text prefix and `multimodal_inputs={"pixel_values": ...}`. Schedule it with `RecordingPrefixLeaseProvider` and assert no `pin:` event (lookup/lease was not called). Then call `_publish_committed_prefix` on a prefix-capable engine sequence carrying the same metadata and assert `PrefixCache.num_entries` is unchanged (no block hash inserted).
   - Run: `pytest -q tests/python/serving/test_multimodal_inputs.py::test_multimodal_request_bypasses_prefix_lookup_and_publication`
   - Expected: `1 passed`.
4. Text-only regression: requests without `multimodal_inputs` must still lease and publish token-id prefixes.
   - Run: `pytest -q tests/python/serving/test_prefix_cache.py tests/python/serving/test_prefix_group_admission.py tests/python/serving/test_prefix_publication.py`
   - Expected: all collected tests pass.

## Non-goals

Video decode, pixel-hash cache keys, MXFP8, and Lightning Indexer.
