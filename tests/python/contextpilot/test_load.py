from __future__ import annotations

import gc
import threading
import time
from typing import Protocol, cast

import psutil
from _pytest.monkeypatch import MonkeyPatch

import moe_infinity.serving.contextpilot_middleware as middleware_module
from moe_infinity.serving.contextpilot_middleware import ContextPilotMiddleware


class _MemoryInfo(Protocol):
    rss: int


def _build_messages(index: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": f"context-{index}-a"},
        {"role": "assistant", "content": f"context-{index}-b"},
        {"role": "system", "content": f"context-{index}-c"},
        {"role": "user", "content": f"query-{index}"},
    ]


def _is_valid_message_batch(messages: object) -> bool:
    if not isinstance(messages, list) or not messages:
        return False

    message_items = cast(list[object], messages)
    for message_obj in message_items:
        if not isinstance(message_obj, dict):
            return False
        message = cast(dict[str, str], message_obj)
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            return False
    return True


def _rss_bytes(process: psutil.Process) -> int:
    memory_info = cast(_MemoryInfo, process.memory_info())
    return int(memory_info.rss)


class _FakeContextPilot:
    def __init__(self, use_gpu: bool = False) -> None:
        _ = use_gpu

    def optimize(self, contexts: list[str], query: str) -> list[dict[str, str]]:
        time.sleep(0.002)
        optimized = [{"role": "system", "content": ctx} for ctx in contexts]
        optimized.append({"role": "user", "content": query})
        return optimized


def test_10_concurrent_requests_no_errors(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(middleware_module, "ContextPilot", _FakeContextPilot)
    middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)
    errors: list[Exception] = []
    outputs: list[list[dict[str, str]]] = []
    results_lock = threading.Lock()

    def _worker(index: int) -> None:
        try:
            output = middleware.process_chat_request(_build_messages(index))
            with results_lock:
                outputs.append(output)
        except Exception as exc:
            with results_lock:
                errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert not errors
    assert len(outputs) == 10
    assert all(_is_valid_message_batch(output) for output in outputs)


def test_50_concurrent_requests_throughput(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(middleware_module, "ContextPilot", _FakeContextPilot)
    middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)
    errors: list[Exception] = []
    outputs: list[list[dict[str, str]]] = []
    results_lock = threading.Lock()

    def _worker(index: int) -> None:
        try:
            output = middleware.process_chat_request(_build_messages(index))
            with results_lock:
                outputs.append(output)
        except Exception as exc:
            with results_lock:
                errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(50)]

    started_at = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    duration_seconds = time.perf_counter() - started_at

    assert all(not thread.is_alive() for thread in threads)
    assert duration_seconds < 30.0
    assert not errors
    assert len(outputs) == 50
    assert all(_is_valid_message_batch(output) for output in outputs)


def test_sustained_100_requests_no_memory_leak(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(middleware_module, "ContextPilot", _FakeContextPilot)
    middleware = ContextPilotMiddleware(use_gpu=False, enabled=True)
    process = psutil.Process()

    _ = gc.collect()
    rss_before = _rss_bytes(process)

    for i in range(100):
        output = middleware.process_chat_request(_build_messages(i))
        assert _is_valid_message_batch(output)

    _ = gc.collect()
    rss_after = _rss_bytes(process)
    rss_delta = rss_after - rss_before

    assert rss_delta < 100 * 1024 * 1024
