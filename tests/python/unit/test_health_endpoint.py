# pyright: reportAny=false, reportCallIssue=false, reportExplicitAny=false, reportMissingParameterType=false, reportMissingTypeArgument=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false
from __future__ import annotations

from typing import Any

import pytest

try:
    from fastapi.testclient import TestClient

    import moe_infinity.entrypoints.openai.api_server_v2 as server_module
except TypeError:
    pytest.skip(
        "Pydantic v1 incompatible with Python 3.12+", allow_module_level=True
    )
from moe_infinity.serving.health import ServerHealthState


def _client_with_health_state(
    monkeypatch: Any, state: ServerHealthState
) -> TestClient:
    monkeypatch.setattr(server_module, "_health_state", state, raising=False)
    return TestClient(server_module.app)


def test_health_endpoint_returns_healthy_json(monkeypatch: Any) -> None:
    state = ServerHealthState()
    state.set_healthy()

    with _client_with_health_state(monkeypatch, state) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "reason": None}


def test_health_endpoint_returns_starting_json(monkeypatch: Any) -> None:
    state = ServerHealthState()
    state.set_starting()

    with _client_with_health_state(monkeypatch, state) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "starting", "reason": None}


def test_health_endpoint_returns_unhealthy_json(monkeypatch: Any) -> None:
    state = ServerHealthState()
    state.set_unhealthy("boom")

    with _client_with_health_state(monkeypatch, state) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "unhealthy", "reason": "boom"}


def test_health_endpoint_has_no_extra_fields(monkeypatch: Any) -> None:
    state = ServerHealthState()
    state.set_healthy()

    with _client_with_health_state(monkeypatch, state) as client:
        response = client.get("/health")

    body = response.json()
    assert "engine_status" not in body
    assert "free_blocks" not in body
