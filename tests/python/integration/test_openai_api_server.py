# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportUnusedVariable=false

import asyncio
from typing import Any, cast

import pytest

from moe_infinity import MoE
from moe_infinity.entrypoints.openai import api_server
from moe_infinity.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)


class _DummyRequest:
    async def is_disconnected(self) -> bool:
        return False


def _raw_request() -> Any:
    return cast(Any, _DummyRequest())


class _FakeTokenizer:
    def encode(self, prompt: str) -> list[int]:
        return [max(1, len(prompt) % 17), 2, 3]

    def decode(
        self, token_ids: list[int], skip_special_tokens: bool = False
    ) -> str:
        _ = skip_special_tokens
        return "".join(chr(96 + token_id) for token_id in token_ids)

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        _ = tokenize
        _ = add_generation_prompt
        return "\n".join(message.get("content", "") for message in conversation)


def _collect_stream_events(response: Any) -> list[str]:
    async def _collect() -> list[str]:
        iterator = getattr(response, "body_iterator", None)
        if iterator is None:
            iterator = getattr(response, "content", None)
        if iterator is None:
            return []

        items: list[str] = []
        async for chunk in iterator:
            if isinstance(chunk, bytes):
                items.append(chunk.decode("utf-8"))
            else:
                items.append(str(chunk))
        return items

    return asyncio.run(_collect())


def _setup_runtime() -> None:
    api_server.model = object()
    api_server.model_name = "unit-test-model"
    api_server.tokenizer = _FakeTokenizer()
    api_server.runtime_max_seq_length = 16


def test_moe_class_exposes_serve() -> None:
    assert hasattr(MoE, "serve")


def test_api_server_routes_exist() -> None:
    route_paths = {
        (tuple(sorted(route.methods or [])), route.path)
        for route in getattr(api_server.app, "routes", [])
    }
    assert (("POST",), "/v1/completions") in route_paths
    assert (("POST",), "/v1/chat/completions") in route_paths


def test_completion_streaming_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_runtime()

    async def _fake_submit_generation(**_: Any) -> dict[str, Any]:
        return {
            "output_text": "ab",
            "token_texts": ["a", "b"],
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }

    monkeypatch.setattr(
        api_server, "_submit_generation", _fake_submit_generation
    )

    request = CompletionRequest(
        model="unit-test-model", prompt="hello", stream=True
    )
    response = asyncio.run(api_server.completion(request, _raw_request()))

    assert getattr(response, "media_type", None) == "text/event-stream"
    events = _collect_stream_events(response)
    assert events
    assert events[0].startswith('data: {"choices"')
    assert events[-1].strip() == "data: [DONE]"


def test_chat_streaming_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_runtime()

    async def _fake_submit_generation(**_: Any) -> dict[str, Any]:
        return {
            "output_text": "xy",
            "token_texts": ["x", "y"],
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "total_tokens": 6,
        }

    monkeypatch.setattr(
        api_server, "_submit_generation", _fake_submit_generation
    )

    request = ChatCompletionRequest(
        model="unit-test-model",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    response = asyncio.run(api_server.chat_completion(request, _raw_request()))

    assert getattr(response, "media_type", None) == "text/event-stream"
    events = _collect_stream_events(response)
    assert events[0].startswith('data: {"choices"')
    assert events[-1].strip() == "data: [DONE]"


def test_long_context_rejected_with_http_400() -> None:
    _setup_runtime()
    api_server.runtime_max_seq_length = 1

    request = CompletionRequest(model="unit-test-model", prompt="hello")
    with pytest.raises(Exception) as exc_info:
        _ = asyncio.run(api_server.completion(request, _raw_request()))

    assert getattr(exc_info.value, "status_code", None) == 400


def test_oom_is_mapped_to_http_503(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_runtime()

    queue_http_error = getattr(api_server, "_QueueHTTPError")

    async def _fake_submit_generation(**_: Any) -> dict[str, Any]:
        raise queue_http_error(503, "Failed to allocate KV blocks")

    monkeypatch.setattr(
        api_server, "_submit_generation", _fake_submit_generation
    )

    request = CompletionRequest(model="unit-test-model", prompt="hello")
    with pytest.raises(Exception) as exc_info:
        _ = asyncio.run(api_server.completion(request, _raw_request()))

    assert getattr(exc_info.value, "status_code", None) == 503


def test_malformed_request_returns_422() -> None:
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    _setup_runtime()
    with TestClient(api_server.app) as client:
        response = client.post(
            "/v1/completions",
            json={"model": "unit-test-model"},
        )
    assert response.status_code == 422
