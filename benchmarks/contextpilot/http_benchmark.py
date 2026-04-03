from __future__ import annotations

import json
import time
from collections.abc import Iterable
from typing import Optional, cast

import requests

from benchmarks.contextpilot.benchmark_utils import MetricsCollector
from benchmarks.contextpilot.dataset_utils import (
    get_workload_names,
    load_workload,
)


def _json_object(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    return None


def _json_load_object(value: str | bytes) -> dict[str, object] | None:
    try:
        loaded = cast(object, json.loads(value))
    except Exception:
        return None
    return _json_object(loaded)


def _response_json_object(
    response: requests.Response,
) -> dict[str, object] | None:
    try:
        payload = cast(object, response.json())
    except Exception:
        return None
    return _json_object(payload)


def _validate_messages(value: object) -> list[dict[str, str]] | None:
    if not isinstance(value, list):
        return None

    validated: list[dict[str, str]] = []
    for item in cast(list[object], value):
        item_obj = _json_object(item)
        if item_obj is None:
            return None
        role = item_obj.get("role")
        content = item_obj.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            return None
        validated.append({"role": role, "content": content})
    return validated


def _to_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _to_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    return None


def _extract_usage_from_header(response: requests.Response) -> dict[str, int]:
    usage_header = response.headers.get("X-Usage")
    if not usage_header:
        return {}

    usage_obj = _json_load_object(usage_header)
    if usage_obj is None:
        return {}

    prompt_tokens = _to_int(usage_obj.get("prompt_tokens"))
    completion_tokens = _to_int(usage_obj.get("completion_tokens"))
    result: dict[str, int] = {}
    if prompt_tokens is not None:
        result["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        result["completion_tokens"] = completion_tokens
    return result


def _extract_status_metric(
    status: dict[str, object], keys: tuple[str, ...], default: float = 0.0
) -> float:
    for key in keys:
        value = _to_float(status.get(key))
        if value is not None:
            return value
    return float(default)


def _extract_expert_cache_hit_rate(status: dict[str, object]) -> float:
    return _extract_status_metric(
        status,
        (
            "expert_cache_hit_rate",
            "expert_hit_rate",
            "expert_cache_hit_rate_avg",
        ),
        default=0.0,
    )


def _extract_kv_cache_hit_rate(status: dict[str, object]) -> float:
    ratio = _extract_status_metric(
        status,
        (
            "kv_cache_hit_rate",
            "kv_hit_rate",
            "prefix_cache_hit_rate",
            "cache_hit_rate",
        ),
        default=0.0,
    )
    return max(0.0, min(1.0, ratio))


def _extract_token_savings_pct(status: dict[str, object]) -> float:
    return max(
        0.0,
        min(
            100.0,
            _extract_status_metric(
                status,
                (
                    "token_savings_avg_pct",
                    "avg_savings_pct",
                    "token_savings_pct",
                ),
                default=0.0,
            ),
        ),
    )


def check_server_health(base_url: str, timeout_s: float = 3.0) -> bool:
    health_url = f"{base_url.rstrip('/')}/health"
    try:
        response = requests.get(health_url, timeout=timeout_s)
    except requests.RequestException:
        return False
    return response.status_code == 200


def set_contextpilot_enabled(
    base_url: str, enabled: bool, timeout_s: float = 5.0
) -> bool:
    toggle_url = f"{base_url.rstrip('/')}/contextpilot/toggle"
    try:
        response = requests.post(
            toggle_url,
            json={"enabled": enabled},
            timeout=timeout_s,
        )
        if response.status_code != 200:
            return False
        payload = _response_json_object(response)
        if payload is not None and isinstance(payload.get("enabled"), bool):
            return bool(payload["enabled"]) == enabled
    except Exception:
        return False
    return False


def measure_request_streaming(
    url: str,
    messages: list[dict[str, str]],
    model: str = "default",
    max_tokens: int = 64,
    temperature: float = 0.0,
) -> dict[str, float | int]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    start = time.perf_counter()
    response = requests.post(url, json=payload, stream=True, timeout=(10, 300))
    response.raise_for_status()

    ttft: float | None = None
    usage: dict[str, int] = {}

    for line in cast(Iterable[bytes], response.iter_lines()):
        if not line:
            continue
        if not line.startswith(b"data: "):
            continue

        raw_chunk = line[len(b"data: ") :].strip()
        if raw_chunk == b"[DONE]":
            break

        if ttft is None:
            ttft = time.perf_counter() - start

        chunk = _json_load_object(raw_chunk)
        if chunk is None:
            continue

        chunk_usage_obj = _json_object(chunk.get("usage"))
        if chunk_usage_obj is not None:
            prompt_tokens = _to_int(chunk_usage_obj.get("prompt_tokens"))
            completion_tokens = _to_int(
                chunk_usage_obj.get("completion_tokens")
            )
            if prompt_tokens is not None:
                usage["prompt_tokens"] = prompt_tokens
            if completion_tokens is not None:
                usage["completion_tokens"] = completion_tokens

        choices_obj = chunk.get("choices")
        if isinstance(choices_obj, list) and choices_obj:
            first_choice_obj = _json_object(cast(list[object], choices_obj)[0])
            if first_choice_obj is not None:
                finish_reason = first_choice_obj.get("finish_reason")
                if finish_reason is not None:
                    break

    e2e_latency = time.perf_counter() - start
    if ttft is None:
        ttft = e2e_latency

    if not usage:
        usage = _extract_usage_from_header(response)

    return {
        "ttft": float(ttft),
        "e2e_latency": float(e2e_latency),
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
    }


def fetch_cp_status(base_url: str) -> dict[str, object]:
    status_url = f"{base_url.rstrip('/')}/contextpilot/status"
    try:
        response = requests.get(status_url, timeout=5)
        response.raise_for_status()
        payload = _response_json_object(response)
        if payload is not None:
            return payload
    except Exception:
        return {}
    return {}


def run_workload_benchmark(
    server_url: str,
    model: str,
    max_tokens: int = 64,
    workload_names: Optional[list[str]] = None,
) -> dict[str, dict[str, float]]:
    base_url = server_url.rstrip("/")
    request_url = f"{base_url}/v1/chat/completions"
    names = (
        workload_names if workload_names is not None else get_workload_names()
    )

    results: dict[str, dict[str, float]] = {}

    for workload_name in names:
        status_before = fetch_cp_status(base_url)
        if bool(status_before.get("enabled", False)):
            _ = set_contextpilot_enabled(base_url, False)
            _ = set_contextpilot_enabled(base_url, True)

        collector = MetricsCollector()
        workload_requests = load_workload(workload_name)

        for request in workload_requests:
            messages_obj = _validate_messages(request.get("messages"))
            if messages_obj is None:
                continue

            measurement: dict[str, float | int] = measure_request_streaming(
                url=request_url,
                messages=messages_obj,
                model=model,
                max_tokens=max_tokens,
                temperature=0.0,
            )

            ttft = _to_float(measurement.get("ttft")) or 0.0
            e2e_latency = _to_float(measurement.get("e2e_latency")) or 0.0
            prompt_tokens = _to_int(measurement.get("prompt_tokens")) or 0

            if prompt_tokens <= 0:
                expected = request.get("expected_token_count")
                if isinstance(expected, int) and expected > 0:
                    prompt_tokens = expected

            prefill_throughput = (
                float(prompt_tokens / ttft)
                if prompt_tokens > 0 and ttft > 0.0
                else 0.0
            )

            collector.add(
                ttft=ttft,
                prefill_throughput=prefill_throughput,
                kv_cache_hit_rate=0.0,
                token_savings_pct=0.0,
                e2e_latency=e2e_latency,
                expert_cache_hit_rate=0.0,
            )

        summary = collector.summarize_for_baseline()
        status = fetch_cp_status(base_url)
        summary["kv_cache_hit_rate"] = _extract_kv_cache_hit_rate(status)
        summary["token_savings_pct"] = _extract_token_savings_pct(status)
        summary["expert_cache_hit_rate"] = _extract_expert_cache_hit_rate(
            status
        )
        results[workload_name] = summary

    return results
