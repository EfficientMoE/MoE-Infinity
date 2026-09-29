from __future__ import annotations

import threading
import time

import pytest
from _pytest.monkeypatch import MonkeyPatch

import moe_infinity.serving.contextpilot_middleware as middleware_module
from moe_infinity.serving.contextpilot_middleware import ContextPilotMiddleware

# Skip (not fail) live-middleware tests that need the optional real package.
requires_contextpilot = pytest.mark.skipif(
    middleware_module.ContextPilot is None,
    reason="contextpilot package not installed",
)


def test_process_chat_request_preserves_roles_and_reorders_docs(
    monkeypatch: MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeCP:
        def __init__(self, use_gpu: bool = False) -> None:
            captured["use_gpu"] = use_gpu

        def reorder(
            self,
            contexts: list[str],
            conversation_id: str | None = None,
        ) -> tuple[list[list[str]], list[int]]:
            captured["contexts"] = list(contexts)
            captured["conversation_id"] = conversation_id
            return ([["reply-a", "rule", "ctx-a"]], [0])

    monkeypatch.setattr(middleware_module, "ContextPilot", FakeCP)
    middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)
    messages = [
        {"role": "system", "content": "rule"},
        {"role": "user", "content": "ctx-a"},
        {"role": "assistant", "content": "reply-a"},
        {"role": "user", "content": "final query"},
    ]

    output = middleware.process_chat_request(
        messages, serving_request_id="srv-1"
    )

    assert output == [
        {"role": "assistant", "content": "reply-a"},
        {"role": "system", "content": "rule"},
        {"role": "user", "content": "ctx-a"},
        {"role": "user", "content": "final query"},
    ]
    assert captured["use_gpu"] is False
    assert captured["contexts"] == ["rule", "ctx-a", "reply-a"]
    assert captured["conversation_id"] == "srv-1"


def test_reorder_non_permutation_keeps_original(
    monkeypatch: MonkeyPatch,
) -> None:
    class FakeCP:
        def __init__(self, use_gpu: bool = False) -> None:
            _ = use_gpu

        def optimize(
            self, contexts: list[str], query: str
        ) -> list[dict[str, str]]:
            _ = contexts
            _ = query
            return [{"role": "system", "content": "only-one"}]

        def reorder(
            self,
            contexts: list[str],
            conversation_id: str | None = None,
        ) -> tuple[list[list[str]], list[int]]:
            _ = contexts
            _ = conversation_id
            return ([["only-one"]], [0])

    monkeypatch.setattr(middleware_module, "ContextPilot", FakeCP)
    middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)
    messages = [
        {"role": "system", "content": "alpha"},
        {"role": "assistant", "content": "beta"},
        {"role": "user", "content": "query"},
    ]

    output = middleware.process_chat_request(messages)

    assert output == messages


def test_non_string_content_skips_reorder(monkeypatch: MonkeyPatch) -> None:
    called = {"reorder": False}

    class FakeCP:
        def __init__(self, use_gpu: bool = False) -> None:
            _ = use_gpu

        def optimize(
            self, contexts: list[str], query: str
        ) -> list[dict[str, str]]:
            _ = contexts
            _ = query
            return [{"role": "user", "content": "flattened"}]

        def reorder(
            self,
            contexts: list[str],
            conversation_id: str | None = None,
        ) -> tuple[list[list[str]], list[int]]:
            _ = contexts
            _ = conversation_id
            called["reorder"] = True
            return ([list(contexts)], [0])

    monkeypatch.setattr(middleware_module, "ContextPilot", FakeCP)
    middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)
    messages = [
        {"role": "system", "content": "rule"},
        {"role": "user", "content": ["image"]},
    ]

    output = middleware.process_chat_request(messages)

    assert called["reorder"] is False
    assert output[1]["content"] == ["image"]


def test_graceful_fallback_on_exception(monkeypatch: MonkeyPatch) -> None:
    class ExplodingCP:
        def __init__(self, use_gpu: bool = False) -> None:
            _ = use_gpu

        def reorder(
            self,
            contexts: list[str],
            conversation_id: str | None = None,
        ) -> tuple[list[list[str]], list[int]]:
            _ = contexts
            _ = conversation_id
            raise RuntimeError("boom")

    monkeypatch.setattr(middleware_module, "ContextPilot", ExplodingCP)
    middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)
    original = [
        {"role": "system", "content": "doc"},
        {"role": "user", "content": "hello"},
    ]

    output = middleware.process_chat_request(original)

    assert output == original


def test_thread_safety(monkeypatch: MonkeyPatch) -> None:
    cp_holder: dict[str, object] = {}

    class ConcurrencySensitiveCP:
        _guard: threading.Lock
        _active: int
        max_active: int

        def __init__(self, use_gpu: bool = False) -> None:
            _ = use_gpu
            self._guard = threading.Lock()
            self._active = 0
            self.max_active = 0
            cp_holder["instance"] = self

        def reorder(
            self,
            contexts: list[str],
            conversation_id: str | None = None,
        ) -> tuple[list[list[str]], list[int]]:
            _ = conversation_id
            with self._guard:
                self._active += 1
                self.max_active = max(self.max_active, self._active)
                if self._active > 1:
                    raise RuntimeError("concurrent reorder detected")

            try:
                time.sleep(0.01)
                return ([list(contexts)], [0])
            finally:
                with self._guard:
                    self._active -= 1

    monkeypatch.setattr(
        middleware_module, "ContextPilot", ConcurrencySensitiveCP
    )
    middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)
    errors: list[Exception] = []
    outputs: list[list[dict[str, str]]] = []

    def _worker(i: int) -> None:
        try:
            output = middleware.process_chat_request(
                [
                    {"role": "system", "content": f"doc-{i}"},
                    {"role": "user", "content": f"query-{i}"},
                ]
            )
            outputs.append(output)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(outputs) == 10
    assert all(
        output
        and output[0]["role"] == "system"
        and output[0]["content"].startswith("doc-")
        for output in outputs
    )
    cp = cp_holder["instance"]
    assert isinstance(cp, ConcurrencySensitiveCP)
    assert cp.max_active == 1


def test_status_metrics_nonblocking_during_slow_optimize(
    monkeypatch: MonkeyPatch,
) -> None:
    """Regression test for /contextpilot/status hang.

    get_last_request_metrics() must not block on the CP-call lock
    while an in-flight process_chat_request is holding it during a slow
    optimize(). Uses the stats-only lock introduced to split counter
    reads from external-call serialization.
    """
    optimize_started = threading.Event()
    optimize_release = threading.Event()

    class SlowCP:
        def __init__(self, use_gpu: bool = False) -> None:
            _ = use_gpu

        def reorder(
            self,
            contexts: list[str],
            conversation_id: str | None = None,
        ) -> tuple[list[list[str]], list[int]]:
            _ = conversation_id
            optimize_started.set()
            _ = optimize_release.wait(timeout=5.0)
            return ([list(contexts)], [0])

    monkeypatch.setattr(middleware_module, "ContextPilot", SlowCP)
    middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)

    worker = threading.Thread(
        target=lambda: middleware.process_chat_request(
            [
                {"role": "system", "content": "slow-doc"},
                {"role": "user", "content": "hi"},
            ]
        )
    )
    worker.start()
    assert optimize_started.wait(timeout=2.0), "worker did not reach optimize"

    metrics_deadline_s = 0.5
    started = time.monotonic()
    metrics = middleware.get_last_request_metrics()
    elapsed = time.monotonic() - started

    optimize_release.set()
    worker.join(timeout=5.0)

    assert elapsed < metrics_deadline_s, (
        f"get_last_request_metrics blocked for {elapsed:.3f}s "
        f"while optimize() was in-flight (budget {metrics_deadline_s}s)"
    )
    assert "reorder_latency_ms" in metrics


def test_on_request_complete_removes_only_new_contextpilot_ids(
    monkeypatch: MonkeyPatch,
) -> None:
    class FakeCP:
        def __init__(self, use_gpu: bool = False) -> None:
            _ = use_gpu
            self._ids = {"req-old"}
            self.removed: list[set[str]] = []

        def reorder(
            self,
            contexts: list[str],
            conversation_id: str | None = None,
        ) -> tuple[list[list[str]], list[int]]:
            _ = conversation_id
            self._ids = {"req-old", "req-new"}
            return ([list(contexts)], [0])

        def get_all_request_ids(self) -> set[str]:
            return set(self._ids)

        def remove_requests(self, request_ids: set[str]) -> dict[str, object]:
            self.removed.append(set(request_ids))
            return {"removed_count": len(request_ids)}

    monkeypatch.setattr(middleware_module, "ContextPilot", FakeCP)
    middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)
    _ = middleware.process_chat_request(
        [
            {"role": "system", "content": "doc"},
            {"role": "user", "content": "q"},
        ],
        serving_request_id="srv-9",
    )
    cp = middleware._cp
    assert isinstance(cp, FakeCP)
    middleware.on_request_complete("srv-9")
    middleware.on_request_complete("srv-9")

    assert cp.removed == [{"req-new"}]


def test_unrelated_serving_id_is_not_removed(
    monkeypatch: MonkeyPatch,
) -> None:
    class FakeCP:
        def __init__(self, use_gpu: bool = False) -> None:
            _ = use_gpu
            self._ids = {"req-old"}
            self.removed: list[set[str]] = []

        def reorder(
            self,
            contexts: list[str],
            conversation_id: str | None = None,
        ) -> tuple[list[list[str]], list[int]]:
            _ = conversation_id
            self._ids = {"req-old", "req-new"}
            return ([list(contexts)], [0])

        def get_all_request_ids(self) -> set[str]:
            return set(self._ids)

        def remove_requests(self, request_ids: set[str]) -> dict[str, object]:
            self.removed.append(set(request_ids))
            return {}

    monkeypatch.setattr(middleware_module, "ContextPilot", FakeCP)
    middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)
    _ = middleware.process_chat_request(
        [
            {"role": "system", "content": "doc"},
            {"role": "user", "content": "q"},
        ],
        serving_request_id="srv-9",
    )
    middleware.on_request_complete("other")
    cp = middleware._cp
    assert isinstance(cp, FakeCP)
    assert cp.removed == []


def test_cp_index_size_uses_get_all_request_ids(
    monkeypatch: MonkeyPatch,
) -> None:
    class FakeCP:
        def __init__(self, use_gpu: bool = False) -> None:
            _ = use_gpu

        def get_all_request_ids(self) -> set[str]:
            return {"req-a", "req-b", "req-c"}

    monkeypatch.setattr(middleware_module, "ContextPilot", FakeCP)
    middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)

    assert not hasattr(middleware._cp, "live_index")
    assert middleware.cp_index_size() == 3


def test_snapshot_failure_after_reorder_still_evicts(
    monkeypatch: MonkeyPatch,
) -> None:
    class FakeCP:
        def __init__(self, use_gpu: bool = False) -> None:
            _ = use_gpu
            self._ids: set[str] = {"req-old"}
            self._snapshots = 0
            self.removed: list[set[str]] = []

        def reorder(
            self,
            contexts: list[str],
            conversation_id: str | None = None,
        ) -> tuple[list[list[str]], list[int]]:
            _ = conversation_id
            self._ids.add("req-new")
            return ([list(contexts)], [0])

        def get_all_request_ids(self) -> set[str]:
            self._snapshots += 1
            if self._snapshots == 2:
                raise RuntimeError("snapshot failed")
            return set(self._ids)

        def remove_requests(self, request_ids: set[str]) -> dict[str, int]:
            self.removed.append(set(request_ids))
            return {"removed_count": len(request_ids)}

    monkeypatch.setattr(middleware_module, "ContextPilot", FakeCP)
    middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)
    output = middleware.process_chat_request(
        [
            {"role": "system", "content": "doc"},
            {"role": "user", "content": "q"},
        ],
        serving_request_id="srv-9",
    )
    middleware.on_request_complete("srv-9")
    cp = middleware._cp
    assert isinstance(cp, FakeCP)

    assert output[0]["content"] == "doc"
    assert cp.removed == [{"req-new"}]


def test_on_request_complete_doesnt_raise() -> None:
    middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)

    middleware.on_request_complete("request-123")


@requires_contextpilot
def test_is_enabled_respects_flag() -> None:
    disabled = ContextPilotMiddleware(enabled=False)
    enabled = ContextPilotMiddleware(enabled=True)

    assert disabled.is_enabled() is False
    assert enabled.is_enabled() is True


def test_empty_messages_handled() -> None:
    middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)

    output = middleware.process_chat_request([])

    assert output == []


def test_process_completion_request_returns_string() -> None:
    middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)

    output = middleware.process_completion_request("explain this")

    assert isinstance(output, str)


def test_completion_prompt_does_not_call_contextpilot(
    monkeypatch: MonkeyPatch,
) -> None:
    called = {"optimize": 0, "reorder": 0}

    class AssertingCP:
        def __init__(self, use_gpu: bool = False) -> None:
            _ = use_gpu

        def optimize(
            self, contexts: list[str], query: str
        ) -> list[dict[str, str]]:
            _ = contexts
            _ = query
            called["optimize"] += 1
            raise AssertionError("optimize must not be called")

        def reorder(
            self,
            contexts: list[str],
            conversation_id: str | None = None,
        ) -> tuple[list[list[str]], list[int]]:
            _ = contexts
            _ = conversation_id
            called["reorder"] += 1
            raise AssertionError("reorder must not be called")

    monkeypatch.setattr(middleware_module, "ContextPilot", AssertingCP)
    middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)

    output = middleware.process_completion_request("the capital of france is")

    assert output == "the capital of france is"
    assert called == {"optimize": 0, "reorder": 0}


def test_explicit_conversation_dedup_waits_until_second_turn(
    monkeypatch: MonkeyPatch,
) -> None:
    hint = "Please refer to [Doc shared] from the previous conversation."

    class FakeCP:
        dedup_calls: int

        def __init__(self, use_gpu: bool = False) -> None:
            _ = use_gpu
            self.dedup_calls = 0

        def deduplicate(
            self,
            contexts: list[list[str]],
            conversation_id: str,
            hint_template: str | None = None,
        ) -> list[dict[str, object]]:
            _ = contexts
            _ = conversation_id
            _ = hint_template
            self.dedup_calls += 1
            if self.dedup_calls == 1:
                raise ValueError("no history")
            return [
                {
                    "new_docs": ["new"],
                    "overlapping_docs": ["shared"],
                    "reference_hints": [hint],
                    "deduplicated_docs": ["new"],
                }
            ]

        def reorder(
            self,
            contexts: list[str],
            conversation_id: str | None = None,
        ) -> tuple[list[list[str]], list[int]]:
            _ = conversation_id
            return ([list(contexts)], [0])

    monkeypatch.setattr(middleware_module, "ContextPilot", FakeCP)
    middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)
    first = middleware.process_chat_request(
        [
            {"role": "system", "content": "shared"},
            {"role": "user", "content": "q1"},
        ],
        conversation_id="user-a",
    )
    second = middleware.process_chat_request(
        [
            {"role": "system", "content": "shared"},
            {"role": "system", "content": "new"},
            {"role": "user", "content": "q2"},
        ],
        conversation_id="user-a",
    )

    assert any(message["content"] == "shared" for message in first)
    assert any(message["content"] == hint for message in second)
    assert any(message["content"] == "new" for message in second)
    assert all(message["content"] != "shared" for message in second)


def test_anonymous_requests_do_not_call_deduplicate(
    monkeypatch: MonkeyPatch,
) -> None:
    seen: list[object] = []

    class FakeCP:
        def __init__(self, use_gpu: bool = False) -> None:
            _ = use_gpu

        def deduplicate(
            self,
            contexts: list[list[str]],
            conversation_id: str,
            hint_template: str | None = None,
        ) -> list[dict[str, object]]:
            _ = contexts
            _ = hint_template
            seen.append(("dedup", conversation_id))
            raise ValueError("should not be called")

        def reorder(
            self,
            contexts: list[str],
            conversation_id: str | None = None,
        ) -> tuple[list[list[str]], list[int]]:
            seen.append(("reorder", conversation_id))
            return ([list(contexts)], [0])

    monkeypatch.setattr(middleware_module, "ContextPilot", FakeCP)
    middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)
    for serving_id in ("srv-1", "srv-2"):
        _ = middleware.process_chat_request(
            [
                {"role": "system", "content": "doc"},
                {"role": "user", "content": "q"},
            ],
            serving_request_id=serving_id,
        )

    assert [item for item in seen if item[0] == "dedup"] == []
    assert [item for item in seen if item[0] == "reorder"] == [
        ("reorder", "srv-1"),
        ("reorder", "srv-2"),
    ]


def test_same_turn_dedup_does_not_run_after_reorder(
    monkeypatch: MonkeyPatch,
) -> None:
    order: list[str] = []

    class FakeCP:
        def __init__(self, use_gpu: bool = False) -> None:
            _ = use_gpu

        def deduplicate(
            self,
            contexts: list[list[str]],
            conversation_id: str,
            hint_template: str | None = None,
        ) -> list[dict[str, object]]:
            _ = contexts
            _ = conversation_id
            _ = hint_template
            order.append("dedup")
            raise ValueError("first turn")

        def reorder(
            self,
            contexts: list[str],
            conversation_id: str | None = None,
        ) -> tuple[list[list[str]], list[int]]:
            _ = conversation_id
            order.append("reorder")
            return ([list(contexts)], [0])

    monkeypatch.setattr(middleware_module, "ContextPilot", FakeCP)
    middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)
    _ = middleware.process_chat_request(
        [
            {"role": "system", "content": "doc"},
            {"role": "user", "content": "q"},
        ],
        serving_request_id="srv",
        conversation_id="user-a",
    )

    assert order.count("dedup") == 1
    assert order == ["dedup", "reorder"]


def test_dedup_removes_duplicates(monkeypatch: MonkeyPatch) -> None:
    class FakeCP:
        def __init__(self, use_gpu: bool = False) -> None:
            _ = use_gpu

        def reorder(
            self,
            contexts: list[str],
            conversation_id: str | None = None,
        ) -> tuple[list[list[str]], list[int]]:
            _ = conversation_id
            return ([list(contexts)], [0])

    monkeypatch.setattr(middleware_module, "ContextPilot", FakeCP)
    middleware = ContextPilotMiddleware(
        use_gpu=False,
        enabled=True,
        dedup_enabled=True,
        reorder_enabled=True,
    )
    repeated = "duplicate-system-block " * 8
    messages = [
        {"role": "system", "content": repeated},
        {"role": "system", "content": repeated},
        {"role": "user", "content": "final query"},
    ]

    output = middleware.process_chat_request(messages)
    stats = middleware.get_token_savings()

    assert any(
        "Deduplicated content" in str(message.get("content", ""))
        for message in output
    )
    assert stats["total_tokens_saved"] > 0


@requires_contextpilot
def test_dedup_without_reorder() -> None:
    middleware = ContextPilotMiddleware(
        use_gpu=False,
        enabled=True,
        reorder_enabled=False,
        dedup_enabled=True,
    )
    repeated = "same-system-context " * 8
    messages = [
        {"role": "system", "content": repeated},
        {"role": "system", "content": repeated},
        {"role": "user", "content": "query"},
    ]

    output = middleware.process_chat_request(messages)
    stats = middleware.get_token_savings()

    assert any(
        "Deduplicated content" in str(message.get("content", ""))
        for message in output
    )
    assert stats["total_tokens_saved"] > 0


@requires_contextpilot
def test_token_savings_tracked() -> None:
    middleware = ContextPilotMiddleware(
        use_gpu=False,
        enabled=True,
        reorder_enabled=False,
        dedup_enabled=True,
    )
    repeated = "dedup-me " * 12
    request = [
        {"role": "system", "content": repeated},
        {"role": "system", "content": repeated},
        {"role": "user", "content": "q"},
    ]

    _ = middleware.process_chat_request(request)
    _ = middleware.process_chat_request(request)
    stats = middleware.get_token_savings()

    assert set(stats.keys()) == {
        "total_tokens_saved",
        "avg_savings_pct",
        "requests_processed",
    }
    assert isinstance(stats["total_tokens_saved"], int)
    assert isinstance(stats["avg_savings_pct"], float)
    assert isinstance(stats["requests_processed"], int)
    assert stats["requests_processed"] == 2
    assert stats["total_tokens_saved"] > 0
