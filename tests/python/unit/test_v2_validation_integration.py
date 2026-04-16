# pyright: reportAny=false, reportExplicitAny=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportMissingParameterType=false
from __future__ import annotations

import importlib
from typing import Any

import pytest

try:
    from fastapi.testclient import TestClient

    MODULE_NAME = "moe_infinity.entrypoints.openai.api_server_v2"
    importlib.import_module(MODULE_NAME)
except TypeError:
    pytest.skip(
        "Pydantic v1 incompatible with Python 3.12+", allow_module_level=True
    )


class _FakeTokenizer:
    def encode(self, prompt: str) -> list[int]:
        length = max(1, len(prompt))
        return [1] * length

    def decode(
        self, token_ids: list[int], skip_special_tokens: bool = False
    ) -> str:
        _ = token_ids
        _ = skip_special_tokens
        return "ok"

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        _ = tokenize
        _ = add_generation_prompt
        return "\n".join(message.get("content", "") for message in conversation)


class _Output:
    def __init__(self) -> None:
        self.token_id = 1
        self.token_text = "ok"
        self.seq_id = 0
        self.token_logprob = None
        self.top_logprobs = None
        self.finished = True
        self.finish_reason = "stop"
        self.usage = {
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "total_tokens": 4,
        }


class _FakeEngine:
    def __init__(self) -> None:
        self.config: dict[str, Any] = {"max_seq_length": 64}

    def add_request(self, **kwargs: Any) -> None:
        on_token = kwargs["on_token"]
        on_token(_Output())

    def abort_request(self, request_id: str) -> None:
        _ = request_id

    def has_pending_requests(self) -> bool:
        return False

    def step(self) -> list[Any]:
        return []

    def shutdown(self) -> None:
        return None


def _snapshot_runtime_state(module: Any) -> dict[str, Any]:
    return {
        "engine": module.engine,
        "stream_manager": module.stream_manager,
        "tokenizer": module.tokenizer,
        "model_name_global": module.model_name_global,
        "runtime_max_seq_length": getattr(
            module, "runtime_max_seq_length", 4096
        ),
        "_engine_task": module._engine_task,
        "_engine_shutdown_event": module._engine_shutdown_event,
        "_startup_args": getattr(module, "_startup_args", None),
        "_model_init_task": getattr(module, "_model_init_task", None),
    }


def _restore_runtime_state(module: Any, state: dict[str, Any]) -> None:
    for key, value in state.items():
        setattr(module, key, value)


def _setup_runtime(module: Any, *, max_seq_length: int) -> None:
    module.engine = _FakeEngine()
    module.stream_manager = object()
    module.tokenizer = _FakeTokenizer()
    module.model_name_global = "unit-test-model"
    module.runtime_max_seq_length = max_seq_length
    setattr(module, "_startup_args", None)


def test_completion_missing_max_tokens_returns_openai_invalid_request_error() -> (
    None
):
    module: Any = importlib.import_module(MODULE_NAME)
    original_state = _snapshot_runtime_state(module)
    _setup_runtime(module, max_seq_length=64)

    try:
        with TestClient(module.app) as client:
            response = client.post(
                "/v1/completions",
                json={"model": "unit-test-model", "prompt": "hello"},
            )

        assert response.status_code == 400
        assert response.json()["error"] == {
            "message": "max_tokens is required. Please provide a positive integer.",
            "type": "invalid_request_error",
            "param": "max_tokens",
            "code": "invalid_request_error",
        }
    finally:
        _restore_runtime_state(module, original_state)


def test_completion_invalid_temperature_returns_param_field() -> None:
    module: Any = importlib.import_module(MODULE_NAME)
    original_state = _snapshot_runtime_state(module)
    _setup_runtime(module, max_seq_length=64)

    try:
        with TestClient(module.app) as client:
            response = client.post(
                "/v1/completions",
                json={
                    "model": "unit-test-model",
                    "prompt": "hello",
                    "max_tokens": 8,
                    "temperature": 5.0,
                },
            )

        assert response.status_code == 400
        payload = response.json()
        assert payload["error"]["type"] == "invalid_request_error"
        assert payload["error"]["code"] == "invalid_request_error"
        assert payload["error"]["param"] == "temperature"
    finally:
        _restore_runtime_state(module, original_state)


def test_completion_over_context_returns_context_length_exceeded() -> None:
    module: Any = importlib.import_module(MODULE_NAME)
    original_state = _snapshot_runtime_state(module)
    _setup_runtime(module, max_seq_length=6)

    try:
        with TestClient(module.app) as client:
            response = client.post(
                "/v1/completions",
                json={
                    "model": "unit-test-model",
                    "prompt": "hello",
                    "max_tokens": 2,
                },
            )

        assert response.status_code == 400
        payload = response.json()
        assert payload["error"]["type"] == "invalid_request_error"
        assert payload["error"]["code"] == "context_length_exceeded"
    finally:
        _restore_runtime_state(module, original_state)


def test_completion_valid_request_returns_200() -> None:
    module: Any = importlib.import_module(MODULE_NAME)
    original_state = _snapshot_runtime_state(module)
    _setup_runtime(module, max_seq_length=64)

    try:
        with TestClient(module.app) as client:
            response = client.post(
                "/v1/completions",
                json={
                    "model": "unit-test-model",
                    "prompt": "hello",
                    "max_tokens": 8,
                    "temperature": 1.0,
                    "top_p": 1.0,
                },
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["choices"][0]["text"] == "ok"
    finally:
        _restore_runtime_state(module, original_state)
