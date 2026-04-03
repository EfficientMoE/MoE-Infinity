from __future__ import annotations

import asyncio
import importlib
import sys
from typing import Any

import pytest

from moe_infinity.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)

try:
    from fastapi.testclient import TestClient

    MODULE_NAME = "moe_infinity.entrypoints.openai.api_server_v2"
    server_module = importlib.import_module(MODULE_NAME)
except TypeError:
    pytest.skip(
        "Pydantic v1 incompatible with Python 3.12+", allow_module_level=True
    )


class _DummyRequest:
    async def is_disconnected(self) -> bool:
        return False


def test_cli_flag_recognized(monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setattr(sys, "argv", ["api_server_v2.py", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        server_module.parse_args()

    captured = capsys.readouterr()
    assert excinfo.value.code == 0
    assert "--enable-contextpilot" in captured.out
    assert "--contextpilot-debug" in captured.out


def test_env_var_overrides_flag(monkeypatch: Any) -> None:
    monkeypatch.setenv("CONTEXTPILOT_ENABLED", "0")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "api_server_v2.py",
            "--model",
            "demo/model",
            "--offload-dir",
            "/tmp/offload",
            "--enable-contextpilot",
        ],
    )

    args = server_module.parse_args()

    assert args.enable_contextpilot is True
    assert (
        server_module._resolve_contextpilot_enabled(args.enable_contextpilot)
        is False
    )


def test_toggle_endpoint_exists() -> None:
    route_exists = any(
        route.path == "/contextpilot/toggle"
        and "POST" in (route.methods or set())
        for route in server_module.app.routes
    )

    assert route_exists


def test_status_endpoint_includes_required_fields() -> None:
    required_fields = {
        "enabled",
        "circuit_breaker_state",
        "requests_processed",
        "reorder_count",
        "dedup_count",
        "avg_reorder_latency_ms",
        "p99_reorder_latency_ms",
        "token_savings_total",
        "token_savings_avg_pct",
        "eviction_sync",
        "cp_index_size",
        "last_fallback_count",
    }

    with TestClient(server_module.app) as client:
        response = client.get("/contextpilot/status")

    assert response.status_code == 200
    payload = response.json()
    assert required_fields.issubset(payload)
    assert payload["circuit_breaker_state"] in {"closed", "open", "half_open"}
    assert set(payload["eviction_sync"]) == {"incoming", "removed", "not_found"}


def test_inject_fault_endpoint_disabled_without_debug(monkeypatch: Any) -> None:
    old_debug = server_module._contextpilot_debug
    old_fault = server_module._contextpilot_fault
    monkeypatch.setattr(server_module, "_contextpilot_debug", False)
    monkeypatch.setattr(server_module, "_contextpilot_fault", "none")

    try:
        with TestClient(server_module.app) as client:
            response = client.post(
                "/contextpilot/inject-fault",
                json={"fault": "reorder_exception", "duration_s": 10},
            )

        assert response.status_code == 403
        assert server_module._contextpilot_fault == "none"
    finally:
        monkeypatch.setattr(server_module, "_contextpilot_debug", old_debug)
        monkeypatch.setattr(server_module, "_contextpilot_fault", old_fault)


def test_completion_uses_contextpilot_before_tokenization(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}

    class FakeMiddleware:
        def process_completion_request(self, prompt: str) -> str:
            captured["middleware_prompt"] = prompt
            return f"optimized::{prompt}"

    async def _fake_wait_non_stream_result(
        **_: Any,
    ) -> tuple[str, dict[str, int], str]:
        return (
            "ok",
            {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            "stop",
        )

    def _fake_tokenize(prompt: str) -> list[int]:
        captured["tokenized_prompt"] = prompt
        return [1, 2]

    monkeypatch.setenv("CONTEXTPILOT_ENABLED", "1")
    monkeypatch.setattr(server_module, "engine", object())
    monkeypatch.setattr(server_module, "model_name_global", "unit-test-model")
    monkeypatch.setattr(server_module, "runtime_max_seq_length", 32)
    monkeypatch.setattr(server_module, "_contextpilot_enabled", True)
    monkeypatch.setattr(server_module, "_contextpilot_fault", "none")
    monkeypatch.setattr(server_module, "_cp_middleware", FakeMiddleware())
    monkeypatch.setattr(server_module, "_tokenize_text", _fake_tokenize)
    monkeypatch.setattr(
        server_module,
        "_ensure_runtime_ready",
        lambda: (object(), object()),
    )
    monkeypatch.setattr(
        server_module,
        "_wait_non_stream_result",
        _fake_wait_non_stream_result,
    )

    request = CompletionRequest(
        model="unit-test-model",
        prompt="hello",
        max_tokens=8,
        stream=False,
    )
    response = asyncio.run(server_module.completion(request, _DummyRequest()))

    assert captured["middleware_prompt"] == "hello"
    assert captured["tokenized_prompt"] == "optimized::hello"
    assert response.choices[0].text == "ok"


def test_chat_middleware_failure_falls_back_to_original_messages(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}

    class FailingMiddleware:
        def process_chat_request(
            self, messages: list[dict[str, str]]
        ) -> list[dict[str, str]]:
            _ = messages
            raise RuntimeError("boom")

    async def _fake_wait_non_stream_result(
        **_: Any,
    ) -> tuple[str, dict[str, int], str]:
        return (
            "ok",
            {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            "stop",
        )

    def _fake_chat_prompt_to_token_ids(request: Any) -> list[int]:
        captured["messages"] = request.messages
        return [1, 2]

    monkeypatch.setenv("CONTEXTPILOT_ENABLED", "1")
    monkeypatch.setattr(server_module, "engine", object())
    monkeypatch.setattr(server_module, "model_name_global", "unit-test-model")
    monkeypatch.setattr(server_module, "runtime_max_seq_length", 32)
    monkeypatch.setattr(server_module, "_contextpilot_enabled", True)
    monkeypatch.setattr(server_module, "_contextpilot_fault", "none")
    monkeypatch.setattr(server_module, "_cp_middleware", FailingMiddleware())
    monkeypatch.setattr(server_module, "_contextpilot_fallback_count", 0)
    monkeypatch.setattr(server_module, "_contextpilot_last_fallback_count", 0)
    monkeypatch.setattr(
        server_module,
        "_chat_prompt_to_token_ids",
        _fake_chat_prompt_to_token_ids,
    )
    monkeypatch.setattr(
        server_module,
        "_ensure_runtime_ready",
        lambda: (object(), object()),
    )
    monkeypatch.setattr(
        server_module,
        "_wait_non_stream_result",
        _fake_wait_non_stream_result,
    )

    original_messages = [{"role": "user", "content": "hello"}]
    request = ChatCompletionRequest(
        model="unit-test-model",
        messages=original_messages,
        max_tokens=8,
        stream=False,
    )

    response = asyncio.run(
        server_module.chat_completion(request, _DummyRequest())
    )

    assert captured["messages"] == original_messages
    assert response.choices[0].message.content == "ok"
    assert server_module._contextpilot_last_fallback_count == 1
