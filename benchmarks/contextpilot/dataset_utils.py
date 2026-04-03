from __future__ import annotations

# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
import json
from pathlib import Path

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_WORKLOAD_FILES = {
    "shared_prefix_rag": "shared_prefix_rag.json",
    "multi_turn_conversation": "multi_turn_conversation.json",
    "batch_with_overlap": "batch_with_overlap.json",
    "no_overlap_baseline": "no_overlap_baseline.json",
}


def get_workload_names() -> list[str]:
    return list(_WORKLOAD_FILES.keys())


def load_workload(name: str) -> list[dict[str, object]]:
    if name not in _WORKLOAD_FILES:
        valid = ", ".join(get_workload_names())
        raise ValueError(f"Unknown workload '{name}'. Valid names: {valid}")

    fixture_path = _FIXTURES_DIR / _WORKLOAD_FILES[name]
    with fixture_path.open("r", encoding="utf-8") as handle:
        payload_raw: object = json.load(handle)

    if not isinstance(payload_raw, dict):
        raise ValueError(
            f"Fixture '{fixture_path}' is invalid: root must be an object"
        )
    payload: dict[str, object] = payload_raw

    requests_raw = payload.get("requests")
    if not isinstance(requests_raw, list):
        raise ValueError(
            f"Fixture '{fixture_path}' is invalid: 'requests' must be a list"
        )

    validated_requests: list[dict[str, object]] = []
    for req_index, request_raw in enumerate(requests_raw):
        if not isinstance(request_raw, dict):
            raise ValueError(
                f"Fixture '{fixture_path}' request[{req_index}] must be an object"
            )

        messages_raw = request_raw.get("messages")
        if not isinstance(messages_raw, list):
            raise ValueError(
                f"Fixture '{fixture_path}' request[{req_index}] must have list 'messages'"
            )

        messages: list[dict[str, str]] = []
        for msg_index, message_raw in enumerate(messages_raw):
            if not isinstance(message_raw, dict):
                raise ValueError(
                    f"Fixture '{fixture_path}' request[{req_index}].messages[{msg_index}] must be an object"
                )
            role = message_raw.get("role")
            content = message_raw.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError(
                    f"Fixture '{fixture_path}' request[{req_index}].messages[{msg_index}] must include string 'role' and 'content'"
                )
            messages.append({"role": role, "content": content})

        expected_token_count = request_raw.get("expected_token_count")
        if not isinstance(expected_token_count, int) or isinstance(
            expected_token_count, bool
        ):
            raise ValueError(
                f"Fixture '{fixture_path}' request[{req_index}] must include integer 'expected_token_count'"
            )

        overlap = request_raw.get("context_overlap_with_prev")
        if not isinstance(overlap, (int, float)) or isinstance(overlap, bool):
            raise ValueError(
                f"Fixture '{fixture_path}' request[{req_index}] must include numeric 'context_overlap_with_prev'"
            )

        validated_requests.append(
            {
                "messages": messages,
                "expected_token_count": expected_token_count,
                "context_overlap_with_prev": float(overlap),
            }
        )

    return validated_requests
