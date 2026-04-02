# pyright: reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnusedCallResult=false
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib import error, request

import pytest

_ALLOWED_HEALTH_STATUS = {"ready", "not_ready", "healthy"}


def _parse_simple_yaml_mapping(text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"invalid yaml line: {raw_line!r}")
        key, value = line.split(":", 1)
        parsed[key.strip()] = _parse_scalar(value.strip())
    return parsed


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.isdigit():
        return int(value)
    return value


def _health_payload_valid(payload: dict[str, Any]) -> bool:
    status = payload.get("status")
    return isinstance(status, str) and status in _ALLOWED_HEALTH_STATUS


def _evict_payload_valid(payload: dict[str, Any]) -> bool:
    if set(payload) != {"request_ids"}:
        return False
    request_ids = payload.get("request_ids")
    return isinstance(request_ids, list) and all(
        isinstance(request_id, str) for request_id in request_ids
    )


def _json_http(
    *,
    method: str,
    url: str,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url=url, method=method, data=body, headers=headers)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw) if raw else {}
            return int(resp.status), data
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        data = json.loads(raw) if raw else {}
        return int(exc.code), data


def _extract_generated_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AssertionError("completion payload missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise AssertionError("completion choice has invalid shape")
    text = first.get("text")
    if isinstance(text, str):
        return text
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    raise AssertionError("completion payload has no text/message content")


def _server_required(url: str, timeout: float) -> None:
    try:
        _json_http(method="GET", url=f"{url}/health", timeout=timeout)
    except error.URLError:
        pytest.skip(f"server is not reachable: {url}")


def test_sidecar_script_exists(repo_root: Path) -> None:
    script = repo_root / "scripts" / "contextpilot_sidecar.sh"
    assert script.exists(), "sidecar launcher script must exist"
    assert script.is_file(), "sidecar launcher path must be a file"
    assert os.access(script, os.X_OK), "sidecar launcher must be executable"


def test_sidecar_config_valid(repo_root: Path) -> None:
    config_path = repo_root / "configs" / "contextpilot_sidecar.yaml"
    assert config_path.exists(), "sidecar config file must exist"

    config = _parse_simple_yaml_mapping(config_path.read_text(encoding="utf-8"))
    required = {"port", "backend_url", "reorder_enabled", "dedup_enabled"}
    assert required.issubset(config.keys())
    assert isinstance(config["port"], int)
    assert isinstance(config["backend_url"], str)
    assert isinstance(config["reorder_enabled"], bool)
    assert isinstance(config["dedup_enabled"], bool)


def test_sidecar_health_contract() -> None:
    for status in _ALLOWED_HEALTH_STATUS:
        assert _health_payload_valid({"status": status})

    assert not _health_payload_valid({"status": "starting"})
    assert not _health_payload_valid({"reason": None})


def test_sidecar_evict_contract() -> None:
    assert _evict_payload_valid({"request_ids": []})
    assert _evict_payload_valid({"request_ids": ["req-a", "req-b"]})

    assert not _evict_payload_valid({"request_ids": "req-a"})
    assert not _evict_payload_valid({"request_ids": [123]})
    assert not _evict_payload_valid({"request_ids": [], "extra": True})


@pytest.mark.integration
@pytest.mark.gpu
def test_pass_through_deterministic(
    sidecar_base_url: str,
    backend_base_url: str,
    integration_timeout_seconds: float,
) -> None:
    _server_required(sidecar_base_url, integration_timeout_seconds)
    _server_required(backend_base_url, integration_timeout_seconds)

    model = os.getenv(
        "CONTEXTPILOT_TEST_MODEL", "deepseek-ai/DeepSeek-V2-Lite-Chat"
    )
    payload = {
        "model": model,
        "prompt": "Say exactly: deterministic pass-through check.",
        "max_tokens": 24,
        "temperature": 0,
    }

    sidecar_status, sidecar_json = _json_http(
        method="POST",
        url=f"{sidecar_base_url}/v1/completions",
        timeout=integration_timeout_seconds,
        payload=payload,
    )
    backend_status, backend_json = _json_http(
        method="POST",
        url=f"{backend_base_url}/v1/completions",
        timeout=integration_timeout_seconds,
        payload=payload,
    )

    assert sidecar_status == 200
    assert backend_status == 200
    assert _extract_generated_text(sidecar_json) == _extract_generated_text(
        backend_json
    )


@pytest.mark.integration
@pytest.mark.gpu
def test_evict_idempotent(
    sidecar_base_url: str, integration_timeout_seconds: float
) -> None:
    _server_required(sidecar_base_url, integration_timeout_seconds)

    status, _ = _json_http(
        method="POST",
        url=f"{sidecar_base_url}/evict",
        timeout=integration_timeout_seconds,
        payload={"request_ids": ["unknown-request-id"]},
    )
    assert status == 200
