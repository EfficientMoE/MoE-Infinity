from __future__ import annotations

import threading
import time

from _pytest.monkeypatch import MonkeyPatch

import moe_infinity.serving.contextpilot_middleware as middleware_module
from moe_infinity.serving.contextpilot_middleware import ContextPilotMiddleware


def test_process_chat_request_returns_messages(
    monkeypatch: MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeCP:
        def __init__(self, use_gpu: bool = False) -> None:
            captured["use_gpu"] = use_gpu

        def optimize(
            self, contexts: list[str], query: str
        ) -> list[dict[str, str]]:
            captured["contexts"] = list(contexts)
            captured["query"] = query
            return [{"role": "assistant", "content": "optimized"}]

    monkeypatch.setattr(middleware_module, "ContextPilot", FakeCP)
    middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)
    messages = [
        {"role": "system", "content": "rule"},
        {"role": "user", "content": "ctx-a"},
        {"role": "assistant", "content": "reply-a"},
        {"role": "user", "content": "final query"},
    ]

    output = middleware.process_chat_request(messages)

    assert isinstance(output, list)
    assert output and isinstance(output[0], dict)
    assert captured["use_gpu"] is False
    assert captured["contexts"] == ["rule", "ctx-a", "reply-a"]
    assert captured["query"] == "final query"


def test_graceful_fallback_on_exception(monkeypatch: MonkeyPatch) -> None:
    class ExplodingCP:
        def __init__(self, use_gpu: bool = False) -> None:
            _ = use_gpu

        def optimize(
            self, contexts: list[str], query: str
        ) -> list[dict[str, str]]:
            _ = contexts
            _ = query
            raise RuntimeError("boom")

    monkeypatch.setattr(middleware_module, "ContextPilot", ExplodingCP)
    middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)
    original = [{"role": "user", "content": "hello"}]

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

        def optimize(
            self, contexts: list[str], query: str
        ) -> list[dict[str, str]]:
            _ = contexts
            with self._guard:
                self._active += 1
                self.max_active = max(self.max_active, self._active)
                if self._active > 1:
                    raise RuntimeError("concurrent optimize detected")

            try:
                time.sleep(0.01)
                return [{"role": "assistant", "content": f"optimized:{query}"}]
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
                [{"role": "user", "content": f"query-{i}"}]
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
        output and output[0]["content"].startswith("optimized:")
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

        def optimize(
            self, contexts: list[str], query: str
        ) -> list[dict[str, str]]:
            optimize_started.set()
            _ = optimize_release.wait(timeout=5.0)
            return [{"role": "user", "content": query}]

    monkeypatch.setattr(middleware_module, "ContextPilot", SlowCP)
    middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)

    worker = threading.Thread(
        target=lambda: middleware.process_chat_request(
            [{"role": "user", "content": "hi"}]
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


def test_on_request_complete_doesnt_raise() -> None:
    middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)

    middleware.on_request_complete("request-123")


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


def test_dedup_removes_duplicates(monkeypatch: MonkeyPatch) -> None:
    class FakeCP:
        def __init__(self, use_gpu: bool = False) -> None:
            _ = use_gpu

        def optimize(
            self, contexts: list[str], query: str
        ) -> list[dict[str, str]]:
            output = [{"role": "system", "content": ctx} for ctx in contexts]
            if query:
                output.append({"role": "user", "content": query})
            return output

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
