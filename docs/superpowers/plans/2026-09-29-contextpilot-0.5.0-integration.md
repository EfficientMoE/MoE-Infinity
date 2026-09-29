# ContextPilot 0.5.0 Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pin MoE-Infinity to ContextPilot 0.5.0 and make the in-process middleware call the real 0.5.0 reorder, dedup, and eviction methods without flattening chat prompts.

**Architecture:** Keep one `ContextPilot(use_gpu=False)` inside `ContextPilotMiddleware`. Chat requests call `reorder` and, only for an explicit conversation key, `deduplicate` before that reorder. Map returned document strings back onto the original role-bearing messages. Record the ContextPilot request ids created by that call and drop them later with `remove_requests`. Completions stay unchanged. Phase C overlap scores stay local and zero; 0.5.0 has no score API.

**Tech Stack:** Python 3.10+, ContextPilot 0.5.0 (`EfficientContext/ContextPilot` `main` `7f3b62b`, PyPI `contextpilot` 0.5.0), FastAPI OpenAI server v2, pytest.

**Spec:** this file, section Contract. Upstream references: `contextpilot/server/live_index.py` on `main` at `7f3b62b`, and `docs/contextpilot/README.md` in this repo.

## Global Constraints

- ContextPilot package: `contextpilot>=0.5.0,<0.6` (PyPI project `contextpilot`, homepage `https://github.com/EfficientContext/ContextPilot`).
- Python floor stays `>=3.10`. Do not reintroduce 3.8/3.9.
- Runtime kill switch stays `CONTEXTPILOT_ENABLED=0`. Do not set `CONTEXTPILOT` or `CONTEXTPILOT_INDEX_URL`; those arm upstream SGLang, vLLM, and OpenAI import hooks.
- Chat message count, roles, and non-string `content` values are preserved. `ContextPilot.optimize` is not used on the serving path.
- Cross-turn dedup runs only when `ChatCompletionRequest.user` is a non-empty string. That string is the `conversation_id`.
- Anonymous HTTP requests pass `conversation_id=serving_request_id` into `reorder` so they do not share ContextPilot's `_default` conversation. Direct `process_chat_request` calls that omit both ids may pass `conversation_id=None` and are not the serving path.
- Existing constructor flags `reorder_enabled` and `dedup_enabled` still gate those steps. `reorder_enabled=False` skips `reorder`. `dedup_enabled=False` skips both upstream `deduplicate` and the exact-string fallback.
- Eviction calls `ContextPilot.remove_requests(set[str])` with ids observed from `get_all_request_ids()` around that reorder. It does not call `on_request_complete`, `evict`, `remove_request`, `remove`, `delete`, or `live_index.pop` on the upstream object.
- Swap events still do not evict ContextPilot state (`EvictionSyncAdapter` already skips `EvictionEvent.SWAPPED`).
- Any ContextPilot exception, or a reorder result that is not a same-multiset permutation of the input document strings, returns the original messages or prompt.
- Phase C does not gain a `predict_prefix_reuse` implementation in this plan.

## Contract

Upstream 0.5.0 class `contextpilot.ContextPilot`:

- `__init__(self, alpha: float = 0.001, use_gpu: bool = False, linkage_method: str = "average", batch_size: int = 10000)`
- `reorder(self, contexts, initial_tokens_per_context: int = 0, conversation_id: str | None = None) -> tuple[list, list]`
  - A flat `list[str]` is wrapped to one context. `result[0][0]` is the reordered document list for that context.
- `optimize(self, docs, query, *, conversation_id=None, system_instruction=None) -> list[dict]`
  - Always returns a new two-message RAG prompt: one `system` blob plus one `user` query. This destroys assistant, tool, and earlier-turn roles. Serving must not call it.
- `deduplicate(self, contexts: list[list], conversation_id: str, hint_template: str | None = None) -> list[dict]`
  - `conversation_id` is required. Raises `ValueError` when that id has no earlier `reorder`.
  - Each result dict has `new_docs`, `overlapping_docs`, `reference_hints`, `deduplicated_docs`.
  - Calling it in the same turn after `reorder` marks the docs just registered as overlapping. Dedup therefore runs before reorder, and only when the caller passed an explicit user id.
- `get_all_request_ids(self) -> set[str]`
- `remove_requests(self, request_ids: set[str]) -> dict`
- `remove_request_by_id(self, request_id: str) -> bool`
- There is no `live_index` dict and no `predict_prefix_reuse`.

Serving ids come from `random_uuid()` in `api_server_v2.py` and are the same ids `ContinuousBatchingEngine.add_request` and `EvictionSyncAdapter` already use. ContextPilot's own ids look like `req-{12 hex chars}` and are a different namespace.

## Review Focus

- Mixed string and non-string message content in one chat request must come back unchanged, including list-valued multimodal content.
- The first turn for a `user` id must keep full document text. Dedup must not replace that turn with reference hints.
- Two anonymous requests must not share one ContextPilot conversation id.
- `on_request_complete(serving_request_id)` must pass only the ContextPilot ids created for that serving id, not the serving id itself.
- A reorder result that inserts, drops, or duplicates a document string must leave the original messages in place.

---

### Task 1: Probe ContextPilot 0.5.0 and record the contract

**Files:**
- Create: `docs/superpowers/reports/2026-09-29-contextpilot-0.5.0-probe.md`
- Test: `tests/python/contextpilot/test_upstream_050_contract.py`

**Interfaces:**
- Consumes: PyPI `contextpilot==0.5.0` in a throwaway venv only.
- Produces: a skipped-or-passing contract test, plus a probe report. Later tasks assume the Contract section above still matches that report. If the report disagrees, stop and amend this plan before Task 2.

- [ ] **Step 1: Write the failing contract test**

```python
import inspect

import pytest

contextpilot = pytest.importorskip("contextpilot")


def test_installed_contextpilot_is_0_5():
    assert contextpilot.__version__.startswith("0.5.")


def test_serving_methods_match_0_5_contract():
    cp = contextpilot.ContextPilot(use_gpu=False)
    assert "conversation_id" in inspect.signature(cp.reorder).parameters
    assert "conversation_id" in inspect.signature(cp.deduplicate).parameters
    assert callable(cp.get_all_request_ids)
    assert callable(cp.remove_requests)
    assert not isinstance(getattr(cp, "live_index", None), dict)
    assert not hasattr(cp, "predict_prefix_reuse")


def test_optimize_replaces_chat_with_two_messages():
    cp = contextpilot.ContextPilot(use_gpu=False)
    messages = cp.optimize(
        ["alpha doc", "beta doc"],
        "what changed?",
    )
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[1]["content"] == "what changed?"
```

- [ ] **Step 2: Run the test without the package and confirm the skip**

Run: `pytest tests/python/contextpilot/test_upstream_050_contract.py -v`

Expected: SKIP with `contextpilot package not installed`, or PASS if 0.5.x is already installed. FAIL on `AttributeError` or an assertion means the installed package is not 0.5.x; do not continue.

- [ ] **Step 3: Probe a throwaway venv and fill the report**

Create `docs/superpowers/reports/2026-09-29-contextpilot-0.5.0-probe.md` with these headings, and paste command output under each one:

- `version` — `python -c "import contextpilot; print(contextpilot.__version__)"` after `pip install 'contextpilot==0.5.0'` in a fresh venv. Expected: `0.5.0`.
- `pip check` — output of `pip check` in that venv. Record any conflict with `elasticsearch==8.18.1` or `transformers`. Do not add those packages to MoE-Infinity `install_requires`.
- `hook delta` — subprocess with `CONTEXTPILOT` and `CONTEXTPILOT_INDEX_URL` unset: `import sys, contextpilot; print(contextpilot.__version__)`. Expected: process starts, version prints, and MoE-Infinity is not imported.
- `dedup first turn` — `ContextPilot(use_gpu=False).deduplicate([["doc"]], conversation_id="user-a")`. Expected: `ValueError`.
- `remove unknown` — `remove_requests({"serving-id"})` on a fresh instance. Expected: a dict whose `not_found` contains `serving-id`.
- `verdict` — one line, `MATCH` or `STOP`. `STOP` if any expected value above is absent.

Use that venv only for the probe. Do not install ContextPilot into the repo environment in this step.

- [ ] **Step 4: Re-run the contract test inside the probe venv**

Run: `pytest tests/python/contextpilot/test_upstream_050_contract.py -v`

Expected: PASS, version assertion included.

- [ ] **Step 5: Commit**

```bash
git add tests/python/contextpilot/test_upstream_050_contract.py \
  docs/superpowers/reports/2026-09-29-contextpilot-0.5.0-probe.md
git commit -m "test(contextpilot): pin the 0.5.0 public contract"
```

### Task 2: Pin the dependency to 0.5.x

**Files:**
- Modify: `requirements-contextpilot.txt`
- Modify: `setup.py` (`extras_require["contextpilot"]`)
- Modify: `.github/workflows/ci-pr.yml` (the `Install contextpilot` step)
- Modify: `docs/contextpilot/README.md` (installation and the PyPI-name troubleshooting paragraph only)

**Interfaces:**
- Consumes: Task 1 verdict `MATCH`.
- Produces: install spec `contextpilot>=0.5.0,<0.6` in the requirements file, the setuptools extra, and CI.

- [ ] **Step 1: Write the pin into the three install sites**

Set each of these to the same specifier `contextpilot>=0.5.0,<0.6`:

- `requirements-contextpilot.txt` replaces `contextpilot>=0.4.0,<1.0`
- `setup.py` extra `"contextpilot"` replaces `contextpilot>=0.4.0`
- `.github/workflows/ci-pr.yml` `pip install contextpilot` becomes `pip install 'contextpilot>=0.5.0,<0.6'`

- [ ] **Step 2: Correct the PyPI troubleshooting paragraph**

In `docs/contextpilot/README.md`, replace the paragraph that says `pip install contextpilot` may pull a different project. State that PyPI `contextpilot` 0.5.x is `EfficientContext/ContextPilot`, that it depends on `elasticsearch==8.18.1`, and that MoE-Infinity does not import Elasticsearch. Keep the Python 3.10+ requirement.

- [ ] **Step 3: Check the pin text**

Run: `grep -n "contextpilot" requirements-contextpilot.txt setup.py .github/workflows/ci-pr.yml`

Expected: all three lines contain `contextpilot>=0.5.0,<0.6` and none contain `>=0.4.0`.

- [ ] **Step 4: Commit**

```bash
git add requirements-contextpilot.txt setup.py .github/workflows/ci-pr.yml docs/contextpilot/README.md
git commit -m "chore(contextpilot): require 0.5.x below 0.6"
```

### Task 3: Reorder chat documents without dropping roles

**Files:**
- Modify: `moe_infinity/serving/contextpilot_middleware.py` (`process_chat_request`, `_reorder_messages`)
- Modify: `tests/python/contextpilot/test_middleware.py`
- Modify: `tests/python/contextpilot/test_load.py` (`_FakeContextPilot`)

**Interfaces:**
- Consumes: `ContextPilot.reorder(docs, conversation_id=...) -> tuple[list, list]`.
- Produces: `ContextPilotMiddleware.process_chat_request(messages: list[dict], *, serving_request_id: str | None = None, conversation_id: str | None = None) -> list[dict]`.

  Keyword-only ids. `serving_request_id` is stored for Task 5 and is not sent as `conversation_id` when `conversation_id` is set. When `conversation_id` is `None` and `serving_request_id` is a non-empty string, `reorder` receives `conversation_id=serving_request_id`. When both are missing, `reorder` receives `conversation_id=None`. `reorder_enabled=False` skips `reorder` and still runs the exact-string fallback when `dedup_enabled` is true. Returned messages keep the input role on each string. The last user string whose content is `str` stays the query and is not part of the document list.

- [ ] **Step 1: Replace the chat tests that expect `optimize` to rewrite the transcript**

In `tests/python/contextpilot/test_middleware.py`:

- `test_process_chat_request_preserves_roles_and_reorders_docs` — fake `reorder` returns `([["reply-a", "rule", "ctx-a"]], [0])`. Input messages are system `rule`, user `ctx-a`, assistant `reply-a`, user `final query`. Assert roles stay `system`, `user`, `assistant`, `user` in that returned order only after contents are matched back: the three non-query messages are rebuilt in returned-doc order with the original role attached to each exact string, and the last message is still `{role: user, content: final query}`. Assert the fake received `conversation_id="srv-1"` when `conversation_id` is omitted and `serving_request_id="srv-1"`.
- `test_reorder_non_permutation_keeps_original` — fake `reorder` returns `([["only-one"]], [0])` for two docs. Assert the output equals the input.
- `test_non_string_content_skips_reorder` — one message has `content=["image"]`. Assert `reorder` is not called and the list content is unchanged.
- `test_graceful_fallback_on_exception` and `test_thread_safety` — move the fake behavior from `optimize` to `reorder`. `reorder` still raises `RuntimeError` or records concurrency. Fallback still returns the original messages. The lock assertion stays on one in-flight `reorder`.

In `tests/python/contextpilot/test_load.py`, replace `optimize` with `reorder(self, contexts, conversation_id=None)` that sleeps `0.002` seconds and returns `([list(contexts)], [0])`.

- [ ] **Step 2: Run the new tests and confirm they fail**

Run: `pytest tests/python/contextpilot/test_middleware.py::test_process_chat_request_preserves_roles_and_reorders_docs tests/python/contextpilot/test_middleware.py::test_reorder_non_permutation_keeps_original tests/python/contextpilot/test_middleware.py::test_non_string_content_skips_reorder -v`

Expected: FAIL because `process_chat_request` does not accept `serving_request_id` or still calls `optimize`.

- [ ] **Step 3: Implement reorder mapping in `ContextPilotMiddleware`**

`process_chat_request` signature is the one in this task's Interfaces. Delete the chat-path `optimize` calls. Under the existing `_lock`:

- Collect document strings from every message except the last user message whose `content` is `str`.
- If any collected message, or any message between the first and last document slot, has non-string `content`, return copies of the input messages.
- Call `self._cp.reorder(docs, conversation_id=conversation_key)` when `reorder_enabled` is true. `conversation_key` is the public `conversation_id` when that argument is a non-empty string, else `serving_request_id` when that is a non-empty string, else `None`.
- Require `reordered[0]` to be a list whose string multiset equals `docs`. Rebuild non-query messages by popping the original message dict for each returned string so the role stays with that string. Append the original query message unchanged.
- On `AttributeError`, `TypeError`, `ValueError`, `IndexError`, or a multiset mismatch, return copies of the input messages and do not warn at warning level for the mismatch. Keep the existing warning for unexpected exceptions.

Do not call `deduplicate` in this task.

- [ ] **Step 4: Run the chat and load tests**

Run: `pytest tests/python/contextpilot/test_middleware.py tests/python/contextpilot/test_load.py -v -m "not gpu and not integration"`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/serving/contextpilot_middleware.py \
  tests/python/contextpilot/test_middleware.py \
  tests/python/contextpilot/test_load.py
git commit -m "fix(contextpilot): reorder documents without flattening chat roles"
```

### Task 4: Dedup only across turns that share `request.user`

**Files:**
- Modify: `moe_infinity/serving/contextpilot_middleware.py` (`process_chat_request`, `_deduplicate_messages`)
- Modify: `moe_infinity/entrypoints/openai/api_server_v2.py` (`_process_chat_messages_with_contextpilot` and its chat call site)
- Test: `tests/python/contextpilot/test_middleware.py`

**Interfaces:**
- Consumes: `process_chat_request(..., *, serving_request_id, conversation_id)` from Task 3. `deduplicate(contexts: list[list[str]], conversation_id: str) -> list[dict]`.
- Produces: the same `process_chat_request` return. When `conversation_id` is a non-empty string, call `deduplicate([docs], conversation_id=conversation_id)` before `reorder`. On `ValueError`, keep the original doc strings and still reorder. On success, replace each document string that appears in `overlapping_docs` with the matching `reference_hints` entry, then reorder the rewritten strings. When `conversation_id` is omitted, do not call `deduplicate`. Intra-request exact-string fallback still runs after reorder.

  `api_server_v2._process_chat_messages_with_contextpilot(messages, *, request_id: str, conversation_id: str | None = None)` passes `serving_request_id=request_id` and that `conversation_id`. The chat handler passes `conversation_id=request.user` only when `request.user` is a non-empty string.

- [ ] **Step 1: Write the dedup tests**

- `test_explicit_conversation_dedup_waits_until_second_turn` — fake `deduplicate` raises `ValueError` on the first call and on the second returns `[{"new_docs": ["new"], "overlapping_docs": ["shared"], "reference_hints": ["Please refer to [Doc shared] from the previous conversation."], "deduplicated_docs": ["new"]}]`. Fake `reorder` returns the docs unchanged. First `process_chat_request` for messages system `shared`, user `q1` with `conversation_id="user-a"` still contains `shared`. Second call with system `shared`, system `new`, user `q2` contains the hint text and `new`, and does not contain the raw `shared` string.
- `test_anonymous_requests_do_not_call_deduplicate` — two calls with `serving_request_id="srv-1"` and `srv-2` and no `conversation_id`. Assert `deduplicate` was not called, and `reorder` saw `conversation_id="srv-1"` then `"srv-2"`.
- `test_same_turn_dedup_does_not_run_after_reorder` — assert one request makes at most one `deduplicate` call and that call happens before `reorder` (record call order in the fake).

Keep `test_dedup_removes_duplicates` and `test_dedup_without_reorder` asserting the exact-string fallback marker `[Deduplicated content; same as message #`.

- [ ] **Step 2: Run those tests and confirm they fail**

Run: `pytest tests/python/contextpilot/test_middleware.py::test_explicit_conversation_dedup_waits_until_second_turn tests/python/contextpilot/test_middleware.py::test_anonymous_requests_do_not_call_deduplicate tests/python/contextpilot/test_middleware.py::test_same_turn_dedup_does_not_run_after_reorder -v`

Expected: FAIL because `deduplicate` is still invoked as `deduplicate(messages)` or not at all.

- [ ] **Step 3: Implement ordered dedup and pass `request.user`**

In the middleware, call `deduplicate` only inside the branch where the public `conversation_id` argument is a non-empty string, before `reorder`, with `contexts=[docs]`. Map hints back by exact string match. Leave the exact-string fallback after reorder.

In `api_server_v2.py`, thread `conversation_id` from `request.user` at the chat call around the existing `request_id = random_uuid()` site. Do not add a new request field.

- [ ] **Step 4: Run middleware tests**

Run: `pytest tests/python/contextpilot/test_middleware.py tests/python/contextpilot/test_load.py -v -m "not gpu and not integration"`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/serving/contextpilot_middleware.py \
  moe_infinity/entrypoints/openai/api_server_v2.py \
  tests/python/contextpilot/test_middleware.py
git commit -m "fix(contextpilot): dedup follow-up turns only for request.user"
```

### Task 5: Evict ContextPilot ids and report the real index size

**Files:**
- Modify: `moe_infinity/serving/contextpilot_middleware.py` (`process_chat_request`, `on_request_complete`)
- Modify: `moe_infinity/entrypoints/openai/api_server_v2.py` (`_contextpilot_index_size`)
- Test: `tests/python/contextpilot/test_middleware.py`

**Interfaces:**
- Consumes: `get_all_request_ids() -> set[str]`, `remove_requests(request_ids: set[str]) -> dict`.
- Produces: `ContextPilotMiddleware.cp_index_size() -> int`. `on_request_complete(serving_request_id: str) -> None` calls `remove_requests` with the set difference of `get_all_request_ids()` taken immediately before and after the matching reorder. Unknown serving ids are a no-op. `cp_index_size` returns `len(get_all_request_ids())`, or `0` when ContextPilot is absent.

- [ ] **Step 1: Write the eviction tests**

- `test_on_request_complete_removes_only_new_contextpilot_ids` — fake `get_all_request_ids` returns `{"req-old"}` before reorder and `{"req-old", "req-new"}` after. `process_chat_request(..., serving_request_id="srv-9")` then `on_request_complete("srv-9")` calls `remove_requests` once with `{"req-new"}`. A second complete for `srv-9` does not call `remove_requests` again.
- `test_unrelated_serving_id_is_not_removed` — `on_request_complete("other")` does not call `remove_requests` with `req-new`.
- `test_cp_index_size_uses_get_all_request_ids` — fake returns three ids, `cp_index_size()` is `3`. No `live_index` attribute.

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest tests/python/contextpilot/test_middleware.py::test_on_request_complete_removes_only_new_contextpilot_ids tests/python/contextpilot/test_middleware.py::test_cp_index_size_uses_get_all_request_ids -v`

Expected: FAIL because cleanup still probes missing method names.

- [ ] **Step 3: Implement id diff, `remove_requests`, and `cp_index_size`**

Snapshot ids under `_lock` around `reorder`. Store the new set on `serving_request_id` when that id is non-empty. `on_request_complete` pops that set and calls `remove_requests` only when the set is non-empty. Delete the probes for `on_request_complete`, `evict`, `remove_request`, `remove`, `delete`, and `live_index` on the upstream object.

`_contextpilot_index_size` calls `middleware.cp_index_size()` when that method exists, otherwise `0`.

- [ ] **Step 4: Run middleware and eviction tests**

Run: `pytest tests/python/contextpilot/test_middleware.py tests/python/contextpilot/test_eviction_sync.py tests/python/contextpilot/test_eviction_parity.py -v -m "not gpu and not integration"`

Expected: PASS. Swap events still do not call middleware completion.

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/serving/contextpilot_middleware.py \
  moe_infinity/entrypoints/openai/api_server_v2.py \
  tests/python/contextpilot/test_middleware.py
git commit -m "fix(contextpilot): evict upstream request ids from the live index"
```

### Task 6: Leave completion prompts unchanged

**Files:**
- Modify: `moe_infinity/serving/contextpilot_middleware.py` (`process_completion_request`)
- Modify: `tests/python/contextpilot/test_middleware.py`

**Interfaces:**
- Consumes: nothing from ContextPilot.
- Produces: `process_completion_request(self, prompt: str) -> str` returns `prompt` and does not call `optimize` or `reorder`.

- [ ] **Step 1: Replace the completion IndexError test**

Replace `test_process_completion_request_survives_optimize_index_error` with `test_completion_prompt_does_not_call_contextpilot`. A fake whose `optimize` or `reorder` raises `AssertionError` must not be called. `process_completion_request("the capital of france is")` returns that same string.

- [ ] **Step 2: Run it and confirm it fails if optimize is still called**

Run: `pytest tests/python/contextpilot/test_middleware.py::test_completion_prompt_does_not_call_contextpilot -v`

Expected: FAIL while `process_completion_request` still calls `optimize`.

- [ ] **Step 3: Return the prompt immediately when the middleware is enabled**

Do not call ContextPilot. Keep the disabled short-circuit. Still increment `requests_processed` the way the current success path does, with zero token savings, so status counters stay monotonic.

- [ ] **Step 4: Run middleware tests**

Run: `pytest tests/python/contextpilot/test_middleware.py -v -m "not gpu and not integration"`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/serving/contextpilot_middleware.py tests/python/contextpilot/test_middleware.py
git commit -m "fix(contextpilot): skip completion prompts that have no documents"
```

### Task 7: Document the 0.5.0 serving behavior

**Files:**
- Modify: `docs/contextpilot/README.md`
- Modify: `docs/contextpilot/eviction_lifecycle.md` (the `remove_requests()` rows only)

**Interfaces:**
- Consumes: the method names from Tasks 3–6.
- Produces: docs that match those methods. No new runtime behavior.

- [ ] **Step 1: Update the integration guide**

In `docs/contextpilot/README.md`, state:

- Chat uses `ContextPilot.reorder`, not `optimize`.
- Cross-turn dedup uses OpenAI `user` as `conversation_id`.
- Completions are not rewritten.
- `/contextpilot/status` field `cp_index_size` is `len(get_all_request_ids())`.
- Phase C waiting-queue scores stay `0.0` on ContextPilot 0.5.0 because that class has no `predict_prefix_reuse`.

In `docs/contextpilot/eviction_lifecycle.md`, say the adapter calls middleware `on_request_complete`, which calls `ContextPilot.remove_requests` with ContextPilot's own ids. Do not say the serving request id is passed through as a ContextPilot id.

- [ ] **Step 2: Search for the stale method names**

Run: `grep -n "live_index\\|ContextPilot.optimize\\|pip install contextpilot>" docs/contextpilot/*.md`

Expected: no remaining claim that serving calls `optimize`, reads `live_index`, or installs an unpinned `contextpilot`.

- [ ] **Step 3: Commit**

```bash
git add docs/contextpilot/README.md docs/contextpilot/eviction_lifecycle.md
git commit -m "docs(contextpilot): describe the 0.5.0 reorder and eviction contract"
```
