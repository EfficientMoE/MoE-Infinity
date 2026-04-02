from __future__ import annotations

import gc
import importlib
import logging
import threading
import time
import tracemalloc
from typing import Protocol, cast

import pytest
from _pytest.monkeypatch import MonkeyPatch

import moe_infinity.serving.contextpilot_middleware as middleware_module
from moe_infinity.serving.contextpilot_middleware import ContextPilotMiddleware


class _EvictionSyncAdapterLike(Protocol):
    def __init__(self, cp_middleware: object | None = None) -> None: ...

    def on_request_finished(self, request_id: str) -> None: ...

    def on_request_aborted(self, request_id: str) -> None: ...

    def on_kv_blocks_freed(self, request_id: str) -> None: ...

    def on_kv_blocks_swapped(self, request_id: str) -> None: ...

    def get_counters(self) -> dict[str, int]: ...


EvictionSyncAdapter = cast(
    type[_EvictionSyncAdapterLike],
    getattr(
        importlib.import_module("moe_infinity.serving.eviction_sync"),
        "EvictionSyncAdapter",
    ),
)


def test_concurrent_reorder_same_context(monkeypatch: MonkeyPatch) -> None:
    cp_holder: dict[str, object] = {}

    class ConcurrencyGuardCP:
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
            with self._guard:
                self._active += 1
                self.max_active = max(self.max_active, self._active)
                if self._active > 1:
                    raise RuntimeError("concurrent optimize detected")

            try:
                time.sleep(0.005)
                output = [
                    {"role": "system", "content": value} for value in contexts
                ]
                output.append({"role": "user", "content": query})
                output.append({"role": "assistant", "content": "ok"})
                return output
            finally:
                with self._guard:
                    self._active -= 1

    monkeypatch.setattr(
        middleware_module,
        "ContextPilot",
        ConcurrencyGuardCP,
    )
    middleware = ContextPilotMiddleware(
        use_gpu=False,
        enabled=True,
        reorder_enabled=True,
        dedup_enabled=False,
    )
    shared_messages = [
        {"role": "system", "content": "policy"},
        {"role": "assistant", "content": "context-a"},
        {"role": "user", "content": "same final query"},
    ]
    expected = [
        {"role": "system", "content": "policy"},
        {"role": "system", "content": "context-a"},
        {"role": "user", "content": "same final query"},
        {"role": "assistant", "content": "ok"},
    ]

    barrier = threading.Barrier(10)
    errors: list[Exception] = []
    outputs: list[list[dict[str, str]]] = []
    outputs_lock = threading.Lock()

    def _worker() -> None:
        try:
            _ = barrier.wait(timeout=2.0)
            output = middleware.process_chat_request(shared_messages)
            with outputs_lock:
                outputs.append(output)
        except Exception as exc:
            with outputs_lock:
                errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert not any(thread.is_alive() for thread in threads)
    assert not errors
    assert len(outputs) == 10
    assert all(output == expected for output in outputs)
    assert shared_messages == [
        {"role": "system", "content": "policy"},
        {"role": "assistant", "content": "context-a"},
        {"role": "user", "content": "same final query"},
    ]
    cp = cp_holder["instance"]
    assert isinstance(cp, ConcurrencyGuardCP)
    assert cp.max_active == 1


def test_abort_mid_reorder(monkeypatch: MonkeyPatch) -> None:
    class AbortOnceCP:
        calls: int

        def __init__(self, use_gpu: bool = False) -> None:
            _ = use_gpu
            self.calls = 0

        def optimize(
            self, contexts: list[str], query: str
        ) -> list[dict[str, str]]:
            _ = contexts
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("reorder aborted")
            return [{"role": "assistant", "content": f"ok:{query}"}]

    monkeypatch.setattr(middleware_module, "ContextPilot", AbortOnceCP)
    middleware = ContextPilotMiddleware(
        use_gpu=False,
        enabled=True,
        reorder_enabled=True,
        dedup_enabled=False,
    )
    messages = [{"role": "user", "content": "recover me"}]

    first = middleware.process_chat_request(messages)
    second = middleware.process_chat_request(messages)
    stats = middleware.get_token_savings()

    assert first == messages
    assert second == [{"role": "assistant", "content": "ok:recover me"}]
    assert stats["requests_processed"] == 2


def test_cp_restart_recovery(
    monkeypatch: MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class RestartingCP:
        def __init__(self, use_gpu: bool = False) -> None:
            _ = use_gpu

        def on_request_complete(self, request_id: str) -> None:
            _ = request_id
            raise ConnectionError("contextpilot sidecar unavailable")

    monkeypatch.setattr(middleware_module, "ContextPilot", RestartingCP)
    cp_middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)
    adapter = EvictionSyncAdapter(cp_middleware)

    with caplog.at_level(logging.WARNING):
        adapter.on_request_finished("req-restart")

    assert adapter.get_counters() == {
        "evict_incoming": 1,
        "evict_removed": 1,
        "evict_not_found": 0,
    }
    assert any(
        "ContextPilot request cleanup failed" in record.getMessage()
        for record in caplog.records
    )


def test_empty_context_list(monkeypatch: MonkeyPatch) -> None:
    class FakeCP:
        def __init__(self, use_gpu: bool = False) -> None:
            _ = use_gpu

        def optimize(
            self, contexts: list[str], query: str
        ) -> list[dict[str, str]]:
            _ = contexts
            _ = query
            return [{"role": "assistant", "content": "unused"}]

    monkeypatch.setattr(middleware_module, "ContextPilot", FakeCP)
    middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)

    output = middleware.process_chat_request([])

    assert output == []


def test_unicode_documents(monkeypatch: MonkeyPatch) -> None:
    class EchoCP:
        def __init__(self, use_gpu: bool = False) -> None:
            _ = use_gpu

        def optimize(
            self, contexts: list[str], query: str
        ) -> list[dict[str, str]]:
            output = [
                {"role": "system", "content": value} for value in contexts
            ]
            output.append({"role": "user", "content": query})
            return output

    monkeypatch.setattr(middleware_module, "ContextPilot", EchoCP)
    middleware = ContextPilotMiddleware(
        use_gpu=False,
        enabled=True,
        reorder_enabled=True,
        dedup_enabled=False,
    )
    messages = [
        {"role": "system", "content": "中文段落：你好世界🌏"},
        {"role": "user", "content": "emoji 😀🔥✨ and العربية"},
        {"role": "assistant", "content": "mixed: English-日本語-русский"},
        {"role": "user", "content": "请总结并保留原文🙂"},
    ]

    output = middleware.process_chat_request(messages)

    assert [msg["content"] for msg in output] == [
        "中文段落：你好世界🌏",
        "emoji 😀🔥✨ and العربية",
        "mixed: English-日本語-русский",
        "请总结并保留原文🙂",
    ]


def test_very_large_context(monkeypatch: MonkeyPatch) -> None:
    class EchoCP:
        def __init__(self, use_gpu: bool = False) -> None:
            _ = use_gpu

        def optimize(
            self, contexts: list[str], query: str
        ) -> list[dict[str, str]]:
            output = [
                {"role": "system", "content": value} for value in contexts
            ]
            output.append({"role": "user", "content": query})
            return output

    monkeypatch.setattr(middleware_module, "ContextPilot", EchoCP)
    middleware = ContextPilotMiddleware(
        use_gpu=False,
        enabled=True,
        reorder_enabled=True,
        dedup_enabled=False,
    )
    large_a = "A" * 60000
    large_b = "B" * 50064
    messages = [
        {"role": "system", "content": large_a},
        {"role": "assistant", "content": large_b},
        {"role": "user", "content": "summarize large context"},
    ]

    start = time.monotonic()
    output = middleware.process_chat_request(messages)
    elapsed = time.monotonic() - start

    assert elapsed < 30.0
    assert isinstance(output, list) and len(output) >= 3
    assert any(msg["content"] == large_a for msg in output)
    assert any(msg["content"] == large_b for msg in output)


def test_clock_skew_eviction_race() -> None:
    class MockCP:
        _lock: threading.Lock

        def __init__(self) -> None:
            self.removed: list[str] = []
            self._lock = threading.Lock()

        def on_request_complete(self, request_id: str) -> None:
            with self._lock:
                self.removed.append(request_id)

    cp = MockCP()
    adapter = EvictionSyncAdapter(cp)
    request_id = "race-id"
    barrier = threading.Barrier(2)

    def _finish() -> None:
        _ = barrier.wait(timeout=2.0)
        adapter.on_request_finished(request_id)

    def _abort() -> None:
        _ = barrier.wait(timeout=2.0)
        adapter.on_request_aborted(request_id)

    t1 = threading.Thread(target=_finish)
    t2 = threading.Thread(target=_abort)
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert not t1.is_alive()
    assert not t2.is_alive()
    assert cp.removed == [request_id]
    assert adapter.get_counters() == {
        "evict_incoming": 2,
        "evict_removed": 1,
        "evict_not_found": 1,
    }


def test_memory_pressure_large_batch(monkeypatch: MonkeyPatch) -> None:
    class EchoCP:
        def __init__(self, use_gpu: bool = False) -> None:
            _ = use_gpu

        def optimize(
            self, contexts: list[str], query: str
        ) -> list[dict[str, str]]:
            output = [
                {"role": "system", "content": value} for value in contexts
            ]
            output.append({"role": "user", "content": query})
            return output

    monkeypatch.setattr(middleware_module, "ContextPilot", EchoCP)
    middleware = ContextPilotMiddleware(
        use_gpu=False,
        enabled=True,
        reorder_enabled=True,
        dedup_enabled=False,
    )
    tracemalloc.start()
    baseline_current, _ = tracemalloc.get_traced_memory()

    for i in range(50):
        output = middleware.process_chat_request(
            [
                {"role": "system", "content": f"ctx-{i}"},
                {"role": "assistant", "content": f"history-{i}"},
                {"role": "user", "content": f"query-{i}"},
            ]
        )
        assert output

    _ = gc.collect()
    after_current, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    delta_bytes = after_current - baseline_current

    assert delta_bytes < 50 * 1024 * 1024


def test_duplicate_request_id() -> None:
    class MockCP:
        def __init__(self) -> None:
            self.removed: list[str] = []

        def on_request_complete(self, request_id: str) -> None:
            self.removed.append(request_id)

    cp = MockCP()
    adapter = EvictionSyncAdapter(cp)

    adapter.on_request_finished("same-id")
    adapter.on_request_finished("same-id")

    assert cp.removed == ["same-id"]
    assert adapter.get_counters() == {
        "evict_incoming": 2,
        "evict_removed": 1,
        "evict_not_found": 1,
    }


def test_swap_then_evict_then_swap() -> None:
    class MockCP:
        def __init__(self) -> None:
            self.removed: list[str] = []

        def on_request_complete(self, request_id: str) -> None:
            self.removed.append(request_id)

    cp = MockCP()
    adapter = EvictionSyncAdapter(cp)

    adapter.on_kv_blocks_swapped("id")
    adapter.on_request_finished("id")
    adapter.on_kv_blocks_swapped("id")

    assert cp.removed == ["id"]
    assert adapter.get_counters() == {
        "evict_incoming": 1,
        "evict_removed": 1,
        "evict_not_found": 0,
    }
