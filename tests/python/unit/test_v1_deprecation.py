# pyright: reportAny=false, reportCallIssue=false, reportMissingParameterType=false, reportMissingTypeArgument=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false
from __future__ import annotations

import importlib
import sys
from unittest import mock

import pytest

try:
    from fastapi.testclient import TestClient
except TypeError:
    pytest.skip(
        "Pydantic v1 incompatible with Python 3.12+", allow_module_level=True
    )

MODULE_NAME = "moe_infinity.entrypoints.openai.api_server"


def _import_fresh_api_server():
    _ = sys.modules.pop(MODULE_NAME, None)
    import moe_infinity

    if not hasattr(moe_infinity, "MoE"):
        setattr(moe_infinity, "MoE", mock.MagicMock())
    with mock.patch("logging.warning") as warning_mock:
        module = importlib.import_module(MODULE_NAME)
    return module, warning_mock


def test_import_logs_v1_deprecation_warning() -> None:
    _, warning_mock = _import_fresh_api_server()

    warning_mock.assert_called_once_with(
        "MoE-Infinity v1 API (api_server.py) is deprecated. Please migrate to v2 (api_server_v2.py). This server will be removed in a future release."
    )


def test_completion_response_has_deprecation_header() -> None:
    module, _ = _import_fresh_api_server()
    module.__dict__["_tokenize_text"] = mock.Mock(return_value=[1])
    module.__dict__["_submit_generation"] = mock.AsyncMock(
        return_value={
            "output_text": "hello",
            "token_texts": ["hello"],
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }
    )

    with TestClient(module.app) as client:
        response = client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "hi"},
        )

    assert response.status_code == 200
    assert response.headers["Deprecation"] == "true"


def test_chat_completion_response_has_deprecation_header() -> None:
    module, _ = _import_fresh_api_server()
    module.__dict__["_chat_prompt_to_token_ids"] = mock.Mock(return_value=[1])
    module.__dict__["_submit_generation"] = mock.AsyncMock(
        return_value={
            "output_text": "hello",
            "token_texts": ["hello"],
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }
    )

    with TestClient(module.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    assert response.headers["Deprecation"] == "true"


def test_health_response_has_deprecation_header() -> None:
    module, _ = _import_fresh_api_server()

    with TestClient(module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["Deprecation"] == "true"
