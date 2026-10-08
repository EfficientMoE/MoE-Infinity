"""Pure percentile and performance-summary math for the MiniMax-M3 BF16
eager baseline (issue #221 phase 4).

This module imports no torch, transformers, or moe_infinity code so it can
run and be unit-tested on CPU. It mirrors the linear-interpolation
percentile used by ``benchmarks/serving/latency.py`` and produces the
machine-readable BF16-baseline summary consumed by
``examples/vl_keyword_eval.py``.
"""

from __future__ import annotations

import math


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("values must be a non-empty list")
    if not math.isfinite(p) or p < 0.0 or p > 100.0:
        raise ValueError(f"p must be a finite value in [0, 100], got {p!r}")
    for value in values:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"samples must be finite and non-negative, got {value!r}"
            )

    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])

    position = (p / 100.0) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])

    lower_value = ordered[lower]
    upper_value = ordered[upper]
    fraction = position - lower
    return float(lower_value + (upper_value - lower_value) * fraction)


def summarize_performance(
    ttft_ms: list[float],
    decode_tokens_per_s: list[float],
) -> dict[str, float | int]:
    if len(ttft_ms) != len(decode_tokens_per_s):
        raise ValueError(
            "ttft_ms and decode_tokens_per_s must have equal length, got "
            f"{len(ttft_ms)} and {len(decode_tokens_per_s)}"
        )
    if not ttft_ms:
        raise ValueError("performance sample lists must be non-empty")

    return {
        "sample_count": len(ttft_ms),
        "ttft_p50_ms": percentile(ttft_ms, 50.0),
        "ttft_p99_ms": percentile(ttft_ms, 99.0),
        "decode_tok_s_p50": percentile(decode_tokens_per_s, 50.0),
        "decode_tok_s_p99": percentile(decode_tokens_per_s, 99.0),
    }
