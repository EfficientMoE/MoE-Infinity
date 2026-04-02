from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest

try:
    from fastapi.testclient import TestClient

    MODULE_NAME = "moe_infinity.entrypoints.openai.api_server_v2"
    server_module = importlib.import_module(MODULE_NAME)
except TypeError:
    pytest.skip(
        "Pydantic v1 incompatible with Python 3.12+", allow_module_level=True
    )


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
