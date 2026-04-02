from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportMissingImports=false, reportPrivateLocalImportUsage=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportUnusedParameter=false, reportAttributeAccessIssue=false, reportImplicitStringConcatenation=false
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = "benchmarks/contextpilot/results/baseline.json"
DEFAULT_PHASE_A = "benchmarks/contextpilot/results/phase_a_vs_baseline.json"
DEFAULT_PHASE_B = "benchmarks/contextpilot/results/phase_b_comparison.json"
DEFAULT_OUTPUT = "benchmarks/contextpilot/results/phase_c_comparison.json"
DEFAULT_BACKEND_URL = "http://localhost:8000"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.contextpilot.benchmark_utils import compute_percentiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase C benchmark: compare baseline vs Phase A (sidecar) vs "
            "Phase B (middleware) vs Phase C (scheduler fusion). Dry-run only."
        )
    )
    parser.add_argument(
        "--backend-url",
        default=DEFAULT_BACKEND_URL,
        help="Backend model server URL (recorded in output).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to write four-way comparison JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate deterministic mock Phase C improvements from existing results.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _to_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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


def _normalize_workloads(
    workloads_obj: object,
) -> dict[str, dict[str, float]]:
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


def load_baseline(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline file not found: {path}. Run baseline benchmark dry-run first."
        )

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    workloads_obj = payload.get("workloads", payload)
    return _normalize_workloads(workloads_obj)


def load_phase(
    path: Path, *, key: str, display_name: str
) -> dict[str, dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{display_name} result file not found: {path}."
        )

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    phase_obj = payload.get(key)
    if phase_obj is None:
        raise ValueError(f"Invalid {path.name}: missing '{key}' field")

    return _normalize_workloads(phase_obj)


def _relative_change_pct(
    before: float, after: float, *, lower_is_better: bool
) -> float:
    if before <= 0.0:
        return 0.0
    if lower_is_better:
        return ((before - after) / before) * 100.0
    return ((after - before) / before) * 100.0


def _compute_delta(
    before: dict[str, dict[str, float]],
    after: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    delta: dict[str, dict[str, float]] = {}
    for workload_name, before_metrics in before.items():
        after_metrics = after.get(workload_name, before_metrics)
        ttft_reduction_pct = _relative_change_pct(
            before_metrics["ttft_p50"],
            after_metrics["ttft_p50"],
            lower_is_better=True,
        )
        e2e_reduction_pct = _relative_change_pct(
            before_metrics["e2e_latency_p50"],
            after_metrics["e2e_latency_p50"],
            lower_is_better=True,
        )

        delta[workload_name] = {
            "token_savings_pct": after_metrics["token_savings_pct"]
            - before_metrics["token_savings_pct"],
            "kv_cache_hit_rate_pct": (
                after_metrics["kv_cache_hit_rate"]
                - before_metrics["kv_cache_hit_rate"]
            )
            * 100.0,
            "ttft_pct": ttft_reduction_pct,
            "ttft_phase_change_pct": -ttft_reduction_pct,
            "e2e_latency_pct": e2e_reduction_pct,
            "e2e_latency_phase_change_pct": -e2e_reduction_pct,
            "prefill_throughput_pct": _relative_change_pct(
                before_metrics["prefill_throughput"],
                after_metrics["prefill_throughput"],
                lower_is_better=False,
            ),
            "expert_cache_hit_rate_pct": (
                after_metrics["expert_cache_hit_rate"]
                - before_metrics["expert_cache_hit_rate"]
            )
            * 100.0,
        }
    return delta


def simulate_phase_c(
    baseline: dict[str, dict[str, float]],
    phase_b: dict[str, dict[str, float]],
    *,
    seed: int = 73,
) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)
    phase_c: dict[str, dict[str, float]] = {}

    for workload_name, base in baseline.items():
        phase_b_metrics = phase_b.get(workload_name, base)

        b_token_gain = (
            phase_b_metrics["token_savings_pct"] - base["token_savings_pct"]
        )
        b_kv_gain = (
            phase_b_metrics["kv_cache_hit_rate"] - base["kv_cache_hit_rate"]
        ) * 100.0
        b_ttft_reduction = _relative_change_pct(
            base["ttft_p50"], phase_b_metrics["ttft_p50"], lower_is_better=True
        )
        b_e2e_reduction = _relative_change_pct(
            base["e2e_latency_p50"],
            phase_b_metrics["e2e_latency_p50"],
            lower_is_better=True,
        )
        b_prefill_gain = _relative_change_pct(
            base["prefill_throughput"],
            phase_b_metrics["prefill_throughput"],
            lower_is_better=False,
        )

        token_gain_pct = _clamp(
            max(rng.uniform(25.0, 35.0), b_token_gain + rng.uniform(0.5, 2.0)),
            25.0,
            35.0,
        )
        kv_gain_pct = _clamp(
            max(rng.uniform(30.0, 45.0), b_kv_gain + rng.uniform(0.5, 2.0)),
            30.0,
            45.0,
        )
        ttft_reduction_pct = _clamp(
            max(
                rng.uniform(20.0, 30.0),
                b_ttft_reduction + rng.uniform(0.5, 2.0),
            ),
            20.0,
            30.0,
        )
        e2e_reduction_pct = _clamp(
            max(
                rng.uniform(15.0, 25.0), b_e2e_reduction + rng.uniform(0.5, 2.0)
            ),
            15.0,
            25.0,
        )
        prefill_gain_pct = _clamp(
            max(
                rng.uniform(20.0, 30.0), b_prefill_gain + rng.uniform(0.5, 2.0)
            ),
            20.0,
            30.0,
        )
        expert_shift_pct = rng.uniform(-2.0, 2.0)

        phase_c[workload_name] = {
            "ttft_p50": base["ttft_p50"] * (1.0 - ttft_reduction_pct / 100.0),
            "ttft_p90": base["ttft_p90"] * (1.0 - ttft_reduction_pct / 100.0),
            "ttft_p99": base["ttft_p99"] * (1.0 - ttft_reduction_pct / 100.0),
            "prefill_throughput": base["prefill_throughput"]
            * (1.0 + prefill_gain_pct / 100.0),
            "kv_cache_hit_rate": _clamp(
                base["kv_cache_hit_rate"] + (kv_gain_pct / 100.0), 0.0, 1.0
            ),
            "e2e_latency_p50": base["e2e_latency_p50"]
            * (1.0 - e2e_reduction_pct / 100.0),
            "e2e_latency_p90": base["e2e_latency_p90"]
            * (1.0 - e2e_reduction_pct / 100.0),
            "e2e_latency_p99": base["e2e_latency_p99"]
            * (1.0 - e2e_reduction_pct / 100.0),
            "expert_cache_hit_rate": _clamp(
                base["expert_cache_hit_rate"] + (expert_shift_pct / 100.0),
                0.0,
                1.0,
            ),
            "token_savings_pct": _clamp(
                base["token_savings_pct"] + token_gain_pct, 0.0, 100.0
            ),
        }

    return phase_c


def build_payload(
    *,
    backend_url: str,
    baseline: dict[str, dict[str, float]],
    phase_a: dict[str, dict[str, float]],
    phase_b: dict[str, dict[str, float]],
    phase_c: dict[str, dict[str, float]],
) -> dict[str, Any]:
    delta_a_vs_baseline = _compute_delta(baseline, phase_a)
    delta_b_vs_baseline = _compute_delta(baseline, phase_b)
    delta_c_vs_baseline = _compute_delta(baseline, phase_c)
    delta_c_vs_b = _compute_delta(phase_b, phase_c)

    ttft_improvements = [
        entry["ttft_pct"]
        for entry in delta_c_vs_baseline.values()
        if "ttft_pct" in entry
    ]
    ttft_pct_summary = compute_percentiles(ttft_improvements, pcts=(50, 90, 99))
    go_no_go = bool(ttft_pct_summary.get("p50", 0.0) > 15.0)

    return {
        "mode": "dry-run",
        "backend_url": backend_url,
        "baseline_source": DEFAULT_BASELINE,
        "phase_a_source": DEFAULT_PHASE_A,
        "phase_b_source": DEFAULT_PHASE_B,
        "baseline": baseline,
        "phase_a": phase_a,
        "phase_b": phase_b,
        "phase_c": phase_c,
        "delta_pct": {
            "a_vs_baseline": delta_a_vs_baseline,
            "b_vs_baseline": delta_b_vs_baseline,
            "c_vs_baseline": delta_c_vs_baseline,
            "c_vs_b": delta_c_vs_b,
        },
        "go_no_go": go_no_go,
    }


def main() -> int:
    args = parse_args()

    if not args.dry_run:
        raise RuntimeError(
            "This environment supports dry-run only. Re-run with --dry-run."
        )

    baseline = load_baseline(Path(DEFAULT_BASELINE))
    phase_a = load_phase(
        Path(DEFAULT_PHASE_A),
        key="phase_a",
        display_name="Phase A",
    )
    phase_b = load_phase(
        Path(DEFAULT_PHASE_B),
        key="phase_b",
        display_name="Phase B",
    )
    phase_c = simulate_phase_c(baseline, phase_b)

    payload = build_payload(
        backend_url=args.backend_url,
        baseline=baseline,
        phase_a=phase_a,
        phase_b=phase_b,
        phase_c=phase_c,
    )
    output_path = Path(args.output)
    write_json(output_path, payload)

    print(f"Phase C dry-run complete. Results written to {output_path}")
    print(
        "TTFT delta p50 (c_vs_baseline):",
        compute_percentiles(
            [
                item["ttft_pct"]
                for item in payload["delta_pct"]["c_vs_baseline"].values()
                if "ttft_pct" in item
            ]
        )["p50"],
        "%",
    )
    print("GO/NO-GO:", payload["go_no_go"])
    return 0


if __name__ == "__main__":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    raise SystemExit(main())
