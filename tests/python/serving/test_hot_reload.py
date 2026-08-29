# pyright: reportAny=false, reportCallIssue=false, reportExplicitAny=false, reportMissingParameterType=false, reportMissingTypeArgument=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false
from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

try:
    from fastapi.testclient import TestClient

    import moe_infinity.entrypoints.openai.api_server_v2 as srv
    from moe_infinity.entrypoints.openai.api_server_v2 import (
        _format_prometheus_metrics,
    )
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


def test_hot_disable_restores_static_targets() -> None:
    original_state = _snapshot_runtime_state()
    fake = SimpleNamespace(
        get_config=Mock(return_value={"adaptive_memory_enabled": True}),
        restore_static_memory_targets=Mock(),
        has_pending_requests=Mock(return_value=False),
        step=Mock(return_value=[]),
    )

    def update(values: dict[str, object]) -> dict[str, object]:
        if values.get("adaptive_memory_enabled") is False:
            fake.restore_static_memory_targets(transactional=True)
        return dict(values)

    fake.update_config = Mock(side_effect=update)
    srv.engine = fake
    try:
        with TestClient(srv.app) as client:
            response = client.post(
                "/v1/config", json={"adaptive_memory_enabled": False}
            )
        assert response.status_code == 200
        fake.restore_static_memory_targets.assert_called_once_with(
            transactional=True
        )
    finally:
        _restore_runtime_state(original_state)


def test_prometheus_contains_controller_state() -> None:
    text = _format_prometheus_metrics(
        {
            "memory": {
                "adaptive": {
                    "devices": {
                        0: {
                            "enabled": True,
                            "expert_target_bytes": 10,
                            "kv_target_blocks": 4,
                            "resize_failures": 1,
                        }
                    }
                }
            }
        }
    )
    assert 'moe_adaptive_memory_enabled{device="0"} 1' in text
    assert 'moe_adaptive_memory_resize_failures_total{device="0"} 1' in text
