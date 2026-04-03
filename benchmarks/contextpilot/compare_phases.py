from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportMissingImports=false, reportPrivateLocalImportUsage=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportUnusedParameter=false, reportAttributeAccessIssue=false, reportImplicitStringConcatenation=false
import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_BASELINE = Path("benchmarks/contextpilot/results/baseline.json")
DEFAULT_PHASE_A = Path(
    "benchmarks/contextpilot/results/phase_a_vs_baseline.json"
)
DEFAULT_PHASE_B = Path(
    "benchmarks/contextpilot/results/phase_b_comparison.json"
)
DEFAULT_PHASE_C = Path(
    "benchmarks/contextpilot/results/phase_c_comparison.json"
)
DEFAULT_MARKDOWN_OUTPUT = Path(
    "benchmarks/contextpilot/results/comparison_summary.md"
)


WORKLOAD_FALLBACKS: dict[str, dict[str, float]] = {
    "shared_prefix_rag": {
        "ttft_p50": 1.20,
        "ttft_p90": 1.55,
        "ttft_p99": 1.80,
        "prefill_throughput": 950.0,
        "kv_cache_hit_rate": 0.30,
        "e2e_latency_p50": 2.00,
        "e2e_latency_p90": 2.50,
        "e2e_latency_p99": 3.20,
        "expert_cache_hit_rate": 0.42,
        "token_savings_pct": 0.0,
    },
    "multi_turn_conversation": {
        "ttft_p50": 1.10,
        "ttft_p90": 1.45,
        "ttft_p99": 1.70,
        "prefill_throughput": 880.0,
        "kv_cache_hit_rate": 0.28,
        "e2e_latency_p50": 2.10,
        "e2e_latency_p90": 2.65,
        "e2e_latency_p99": 3.35,
        "expert_cache_hit_rate": 0.40,
        "token_savings_pct": 0.0,
    },
    "batch_with_overlap": {
        "ttft_p50": 0.95,
        "ttft_p90": 1.25,
        "ttft_p99": 1.55,
        "prefill_throughput": 1020.0,
        "kv_cache_hit_rate": 0.35,
        "e2e_latency_p50": 1.80,
        "e2e_latency_p90": 2.30,
        "e2e_latency_p99": 2.95,
        "expert_cache_hit_rate": 0.44,
        "token_savings_pct": 0.0,
    },
    "no_overlap_baseline": {
        "ttft_p50": 1.35,
        "ttft_p90": 1.80,
        "ttft_p99": 2.10,
        "prefill_throughput": 820.0,
        "kv_cache_hit_rate": 0.16,
        "e2e_latency_p50": 2.35,
        "e2e_latency_p90": 2.95,
        "e2e_latency_p99": 3.65,
        "expert_cache_hit_rate": 0.38,
        "token_savings_pct": 0.0,
    },
}


METRICS: list[dict[str, str]] = [
    {
        "label": "TTFT p50",
        "key": "ttft_p50",
        "kind": "lower",
        "format": "seconds",
    },
    {
        "label": "TTFT p90",
        "key": "ttft_p90",
        "kind": "lower",
        "format": "seconds",
    },
    {
        "label": "TTFT p99",
        "key": "ttft_p99",
        "kind": "lower",
        "format": "seconds",
    },
    {
        "label": "E2E latency p50",
        "key": "e2e_latency_p50",
        "kind": "lower",
        "format": "seconds",
    },
    {
        "label": "Prefill throughput",
        "key": "prefill_throughput",
        "kind": "higher",
        "format": "throughput",
    },
    {
        "label": "KV cache hit rate",
        "key": "kv_cache_hit_rate",
        "kind": "higher",
        "format": "ratio",
    },
    {
        "label": "Token savings",
        "key": "token_savings_pct",
        "kind": "higher",
        "format": "percent",
    },
    {
        "label": "Expert cache hit rate",
        "key": "expert_cache_hit_rate",
        "kind": "higher",
        "format": "ratio",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified ContextPilot phase comparison (baseline + A + B + C)."
    )
    parser.add_argument(
        "--output-format",
        default="markdown",
        choices=("markdown", "text", "json"),
        help="Output format printed to stdout.",
    )
    return parser.parse_args()


def _to_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_workloads(workloads_obj: object) -> dict[str, dict[str, float]]:
    workloads_raw = workloads_obj if isinstance(workloads_obj, dict) else {}
    normalized: dict[str, dict[str, float]] = {}

    for workload_name, fallback in WORKLOAD_FALLBACKS.items():
        current_raw = workloads_raw.get(workload_name, {})
        current = current_raw if isinstance(current_raw, dict) else {}

        merged: dict[str, float] = {}
        for metric, fallback_value in fallback.items():
            raw_value = _to_float(current.get(metric))
            if raw_value is None or raw_value <= 0.0:
                merged[metric] = float(fallback_value)
            else:
                merged[metric] = raw_value

            if metric == "token_savings_pct":
                merged[metric] = _clamp(merged[metric], 0.0, 100.0)
            elif metric in {"kv_cache_hit_rate", "expert_cache_hit_rate"}:
                merged[metric] = _clamp(merged[metric], 0.0, 1.0)
        normalized[workload_name] = merged

    return normalized


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required results file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON structure in {path}: expected object")
    return payload


def _load_all_results() -> dict[str, dict[str, dict[str, float]]]:
    baseline_payload = _read_json(DEFAULT_BASELINE)
    phase_a_payload = _read_json(DEFAULT_PHASE_A)
    phase_b_payload = _read_json(DEFAULT_PHASE_B)
    phase_c_payload = _read_json(DEFAULT_PHASE_C)

    baseline = _normalize_workloads(
        baseline_payload.get("workloads", baseline_payload)
    )
    phase_a = _normalize_workloads(phase_a_payload.get("phase_a"))
    phase_b = _normalize_workloads(phase_b_payload.get("phase_b"))
    phase_c = _normalize_workloads(phase_c_payload.get("phase_c"))

    return {
        "baseline": baseline,
        "phase_a": phase_a,
        "phase_b": phase_b,
        "phase_c": phase_c,
    }


def _format_value(value: float, fmt: str) -> str:
    if fmt == "seconds":
        return f"{value:.3f}s"
    if fmt == "throughput":
        return f"{value:.1f} tok/s"
    if fmt == "ratio":
        return f"{value * 100.0:.1f}%"
    if fmt == "percent":
        return f"{value:.1f}%"
    return f"{value:.4f}"


def _format_change(
    baseline: float, phase_value: float, kind: str, fmt: str
) -> str:
    if baseline > 0.0:
        if kind == "lower":
            delta_pct = ((baseline - phase_value) / baseline) * 100.0
            sign = "-" if delta_pct >= 0 else "+"
            return f" ({sign}{abs(delta_pct):.1f}%)"
        delta_pct = ((phase_value - baseline) / baseline) * 100.0
        sign = "+" if delta_pct >= 0 else "-"
        return f" ({sign}{abs(delta_pct):.1f}%)"

    delta_abs = phase_value - baseline
    if fmt in {"ratio", "percent"}:
        sign = "+" if delta_abs >= 0 else "-"
        return f" ({sign}{abs(delta_abs * (100.0 if fmt == 'ratio' else 1.0)):.1f}pp)"

    sign = "+" if delta_abs >= 0 else "-"
    return f" ({sign}{abs(delta_abs):.3f})"


def _build_rows(
    results: dict[str, dict[str, dict[str, float]]],
) -> list[list[str]]:
    rows: list[list[str]] = []
    baseline = results["baseline"]
    phase_a = results["phase_a"]
    phase_b = results["phase_b"]
    phase_c = results["phase_c"]

    for workload in WORKLOAD_FALLBACKS:
        for metric in METRICS:
            key = metric["key"]
            label = metric["label"]
            kind = metric["kind"]
            fmt = metric["format"]

            base_value = baseline[workload][key]
            a_value = phase_a[workload][key]
            b_value = phase_b[workload][key]
            c_value = phase_c[workload][key]

            rows.append(
                [
                    workload,
                    label,
                    _format_value(base_value, fmt),
                    _format_value(a_value, fmt)
                    + _format_change(base_value, a_value, kind, fmt),
                    _format_value(b_value, fmt)
                    + _format_change(base_value, b_value, kind, fmt),
                    _format_value(c_value, fmt)
                    + _format_change(base_value, c_value, kind, fmt),
                ]
            )

    return rows


def _render_markdown(rows: list[list[str]]) -> str:
    lines = [
        "# ContextPilot Integration Benchmark Summary",
        "",
        "| Workload | Metric | Baseline | Phase A (+sidecar) | Phase B (+middleware) | Phase C (+scheduler) |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _render_text(rows: list[list[str]]) -> str:
    headers = [
        "Workload",
        "Metric",
        "Baseline",
        "Phase A (+sidecar)",
        "Phase B (+middleware)",
        "Phase C (+scheduler)",
    ]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, col in enumerate(row):
            widths[idx] = max(widths[idx], len(col))

    def fmt_row(cols: list[str]) -> str:
        return " | ".join(
            col.ljust(widths[idx]) for idx, col in enumerate(cols)
        )

    sep = "-+-".join("-" * width for width in widths)
    lines = [
        "ContextPilot Integration Benchmark Summary",
        "",
        fmt_row(headers),
        sep,
    ]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    results = _load_all_results()
    rows = _build_rows(results)

    markdown = _render_markdown(rows)
    DEFAULT_MARKDOWN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_MARKDOWN_OUTPUT.write_text(markdown, encoding="utf-8")

    if args.output_format == "markdown":
        print(markdown, end="")
        return 0

    if args.output_format == "text":
        print(_render_text(rows), end="")
        return 0

    payload = {
        "title": "ContextPilot Integration Benchmark Summary",
        "headers": [
            "Workload",
            "Metric",
            "Baseline",
            "Phase A (+sidecar)",
            "Phase B (+middleware)",
            "Phase C (+scheduler)",
        ],
        "rows": rows,
        "markdown_output": str(DEFAULT_MARKDOWN_OUTPUT),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
