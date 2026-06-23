# pyright: reportAny=false, reportCallIssue=false, reportExplicitAny=false, reportMissingParameterType=false, reportMissingTypeArgument=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false
from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

try:
    from fastapi.testclient import TestClient

    import moe_infinity.entrypoints.openai.api_server_v2 as srv
except TypeError:
    pytest.skip(
        "Pydantic v1 incompatible with Python 3.12+", allow_module_level=True
    )


def _snapshot_runtime_state() -> dict[str, Any]:
    return {
        "engine": srv.engine,
        "stream_manager": srv.stream_manager,
        "tokenizer": srv.tokenizer,
        "model_name_global": srv.model_name_global,
        "_engine_task": srv._engine_task,
        "_engine_shutdown_event": srv._engine_shutdown_event,
        "_startup_args": getattr(srv, "_startup_args", None),
        "_model_init_task": getattr(srv, "_model_init_task", None),
    }


def _restore_runtime_state(state: dict[str, Any]) -> None:
    for key, value in state.items():
        setattr(srv, key, value)


def test_reload_valid_module(monkeypatch: pytest.MonkeyPatch) -> None:
    original_state = _snapshot_runtime_state()

    def _reload(module: Any) -> Any:
        return module

    monkeypatch.setattr(importlib, "reload", _reload)

    try:
        with TestClient(srv.app) as client:
            response = client.post("/v1/reload", json={"modules": ["json"]})

        body = response.json()
        assert response.status_code == 200
        assert body["status"] == "ok"
        assert body["reloaded"] == ["json"]
        assert body["errors"] == []
    finally:
        _restore_runtime_state(original_state)


def test_reload_invalid_module() -> None:
    original_state = _snapshot_runtime_state()

    try:
        with TestClient(srv.app) as client:
            response = client.post(
                "/v1/reload", json={"modules": ["nonexistent.module"]}
            )

        body = response.json()
        assert response.status_code == 200
        assert body["status"] == "partial"
        assert body["reloaded"] == []
        assert body["errors"][0]["module"] == "nonexistent.module"
    finally:
        _restore_runtime_state(original_state)


def test_reload_bad_payload() -> None:
    original_state = _snapshot_runtime_state()

    try:
        with TestClient(srv.app) as client:
            response = client.post("/v1/reload", json={"modules": "not_a_list"})

        assert response.status_code == 400
    finally:
        _restore_runtime_state(original_state)


def test_config_get_no_engine() -> None:
    original_state = _snapshot_runtime_state()
    srv.engine = None

    try:
        with TestClient(srv.app) as client:
            response = client.get("/v1/config")

        assert response.status_code == 503
    finally:
        _restore_runtime_state(original_state)


def test_config_post_no_engine() -> None:
    original_state = _snapshot_runtime_state()
    srv.engine = None

    try:
        with TestClient(srv.app) as client:
            response = client.post("/v1/config", json={"foo": "bar"})

        assert response.status_code == 503
    finally:
        _restore_runtime_state(original_state)
