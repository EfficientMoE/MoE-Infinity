# pyright: reportAny=false, reportExplicitAny=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportMissingParameterType=false
from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace
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
        return [len(prompt), 1]

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
        self.usage = {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }
        self.finished = True
        self.finish_reason = "stop"


class _FakeEngine:
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
        "_engine_task": module._engine_task,
        "_engine_shutdown_event": module._engine_shutdown_event,
        "_startup_args": getattr(module, "_startup_args", None),
        "_model_init_task": getattr(module, "_model_init_task", None),
    }


def _restore_runtime_state(module: Any, state: dict[str, Any]) -> None:
    for key, value in state.items():
        setattr(module, key, value)


def test_completion_returns_503_while_service_starting(
    monkeypatch: Any,
) -> None:
    module: Any = importlib.import_module(MODULE_NAME)
    original_state = _snapshot_runtime_state(module)

    async def _slow_initialize_model() -> None:
        await asyncio.sleep(0.2)

    monkeypatch.setattr(
        module,
        "_initialize_model",
        _slow_initialize_model,
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "_startup_args",
        SimpleNamespace(model="unit-test-model"),
        raising=False,
    )
    module.engine = None
    module.stream_manager = None
    module.tokenizer = None
    module.model_name_global = None

    try:
        with TestClient(module.app) as client:
            response = client.post(
                "/v1/completions",
                json={"model": "unit-test-model", "prompt": "hello"},
            )

        body = response.json()
        assert response.status_code == 503
        assert body["error"]["code"] == "service_starting"
        assert (
            body["error"]["message"]
            == "Service is starting. Please retry shortly."
        )
        assert body["error"]["type"] == "server_error"
    finally:
        _restore_runtime_state(module, original_state)


def test_chat_completion_returns_503_while_service_starting(
    monkeypatch: Any,
) -> None:
    module: Any = importlib.import_module(MODULE_NAME)
    original_state = _snapshot_runtime_state(module)

    async def _slow_initialize_model() -> None:
        await asyncio.sleep(0.2)

    monkeypatch.setattr(
        module,
        "_initialize_model",
        _slow_initialize_model,
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "_startup_args",
        SimpleNamespace(model="unit-test-model"),
        raising=False,
    )
    module.engine = None
    module.stream_manager = None
    module.tokenizer = None
    module.model_name_global = None

    try:
        with TestClient(module.app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "unit-test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        body = response.json()
        assert response.status_code == 503
        assert body["error"]["code"] == "service_starting"
        assert (
            body["error"]["message"]
            == "Service is starting. Please retry shortly."
        )
        assert body["error"]["type"] == "server_error"
    finally:
        _restore_runtime_state(module, original_state)


def test_handlers_work_normally_after_engine_initialized() -> None:
    module: Any = importlib.import_module(MODULE_NAME)
    original_state = _snapshot_runtime_state(module)

    module.engine = _FakeEngine()
    module.stream_manager = object()
    module.tokenizer = _FakeTokenizer()
    module.model_name_global = "unit-test-model"
    setattr(module, "_startup_args", None)

    try:
        with TestClient(module.app) as client:
            completion_response = client.post(
                "/v1/completions",
                json={
                    "model": "unit-test-model",
                    "prompt": "hello",
                    "max_tokens": 5,
                },
            )
            chat_response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "unit-test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 5,
                },
            )

        assert completion_response.status_code == 200
        assert completion_response.json()["choices"][0]["text"] == "ok"
        assert chat_response.status_code == 200
        assert chat_response.json()["choices"][0]["message"]["content"] == "ok"
    finally:
        _restore_runtime_state(module, original_state)


def test_initialize_model_failure_sets_unhealthy(monkeypatch: Any) -> None:
    module: Any = importlib.import_module(MODULE_NAME)
    original_state = _snapshot_runtime_state(module)
    original_health = module._health_state

    from moe_infinity.serving.health import ServerHealthState

    module._health_state = ServerHealthState()
    module.engine = None
    setattr(
        module,
        "_startup_args",
        SimpleNamespace(
            model="unit-test-model",
            offload_dir="/tmp/unit-test-offload",
            device_memory_ratio=0.5,
            enable_prefix_caching=False,
            speculative_draft=None,
            startup_timeout=None,
            decode_step_timeout=None,
        ),
    )

    import transformers

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom: tokenizer load failed")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", _boom)

    try:
        asyncio.run(module._initialize_model())

        status = module._health_state.get_status_dict()
        assert status["status"] == "unhealthy"
        assert status["reason"] is not None
        assert "boom" in status["reason"]
        assert module.engine is None
        assert not module._health_state.is_healthy()
    finally:
        module._health_state = original_health
        _restore_runtime_state(module, original_state)
