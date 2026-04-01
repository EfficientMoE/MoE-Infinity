# pyright: reportAny=false, reportCallIssue=false, reportMissingParameterType=false, reportMissingTypeArgument=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false
from __future__ import annotations

import json

import pytest

try:
    from moe_infinity.entrypoints.openai.protocol import (
        ErrorResponse,
        create_error_response,
    )
except TypeError:
    pytest.skip(
        "Pydantic v1 incompatible with Python 3.12+", allow_module_level=True
    )


def _decode_body(response) -> dict:
    return json.loads(response.body.decode())


def test_create_error_response_serializes_nested_error_structure() -> None:
    response = create_error_response(
        400,
        "Prompt too long",
        "invalid_request_error",
        "context_length_exceeded",
    )

    assert response.status_code == 400
    assert _decode_body(response) == {
        "error": {
            "message": "Prompt too long",
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "param": None,
        }
    }


def test_error_payload_fields_live_inside_error_dict() -> None:
    response = create_error_response(
        500,
        "OOM",
        "server_error",
        "server_error",
        param="max_tokens",
    )

    body = _decode_body(response)

    assert body["error"] == {
        "message": "OOM",
        "type": "server_error",
        "code": "server_error",
        "param": "max_tokens",
    }


def test_request_id_is_top_level_when_provided() -> None:
    response = create_error_response(
        500,
        "OOM",
        "server_error",
        "server_error",
        request_id="req-123",
    )

    body = _decode_body(response)

    assert body["request_id"] == "req-123"
    assert "debug" not in body


def test_request_id_and_debug_are_omitted_when_none() -> None:
    response = create_error_response(
        400,
        "Prompt too long",
        "invalid_request_error",
        "context_length_exceeded",
    )

    body = _decode_body(response)

    assert "request_id" not in body
    assert "debug" not in body


@pytest.mark.parametrize(
    ("error_type", "code"),
    [
        ("invalid_request_error", "context_length_exceeded"),
        ("server_error", "server_error"),
        ("context_length_exceeded", "context_length_exceeded"),
    ],
)
def test_supported_error_types_round_trip(error_type: str, code: str) -> None:
    response = create_error_response(400, "msg", error_type, code)

    assert _decode_body(response)["error"]["type"] == error_type


def test_legacy_flat_errorresponse_init_still_builds_nested_shape() -> None:
    response = ErrorResponse(
        message="Prompt too long",
        type="invalid_request_error",
        code="context_length_exceeded",
        param=None,
    )

    assert response.dict() == {
        "error": {
            "message": "Prompt too long",
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "param": None,
        }
    }
