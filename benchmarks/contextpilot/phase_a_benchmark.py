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
DEFAULT_OUTPUT = "benchmarks/contextpilot/results/phase_a_vs_baseline.json"
DEFAULT_SIDECAR_URL = "http://localhost:8765"
DEFAULT_BACKEND_URL = "http://localhost:8000"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.contextpilot.http_benchmark import (
    check_server_health,
    run_workload_benchmark,
)

DEFAULT_MODEL = "default"
DEFAULT_MAX_TOKENS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase A benchmark: compare baseline vs ContextPilot sidecar "
            "(supports dry-run and real HTTP measurement)."
        )
    )
    parser.add_argument(
        "--sidecar-url",
        default=DEFAULT_SIDECAR_URL,
        help="ContextPilot sidecar URL (used by real runs; recorded in output).",
    )
    parser.add_argument(
        "--backend-url",
        default=DEFAULT_BACKEND_URL,
        help="Backend model server URL (used by real runs; recorded in output).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to write Phase A comparison JSON.",
    )
    parser.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE,
        help="Baseline JSON input path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate mock sidecar improvements from baseline without live servers.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model identifier sent in OpenAI-compatible request payload.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="max_tokens value sent in benchmark requests.",
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


def load_baseline_workloads(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline file not found: {path}. Run baseline benchmark dry-run first."
        )

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    workloads_obj = payload.get("workloads", payload)
    if not isinstance(workloads_obj, dict):
        raise ValueError(
            "Invalid baseline JSON: expected top-level 'workloads' object"
        )

    baseline: dict[str, dict[str, float]] = {}
    for workload_name, fallback in WORKLOAD_FALLBACKS.items():
        current_raw = workloads_obj.get(workload_name, {})
        current = current_raw if isinstance(current_raw, dict) else {}

        merged: dict[str, float] = {}
        for metric, fallback_value in fallback.items():
            raw_value = _to_float(current.get(metric))
            if raw_value is None or raw_value <= 0.0:
                merged[metric] = float(fallback_value)
            else:
                merged[metric] = raw_value
        baseline[workload_name] = merged

    return baseline


def simulate_phase_a(
    baseline: dict[str, dict[str, float]], *, seed: int = 13
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    rng = random.Random(seed)
    phase_a: dict[str, dict[str, float]] = {}
    delta_pct: dict[str, dict[str, float]] = {}

    for workload_name, base in baseline.items():
        token_gain_pct = rng.uniform(15.0, 25.0)
        kv_gain_pct = rng.uniform(20.0, 35.0)
        ttft_reduction_pct = rng.uniform(10.0, 20.0)
        e2e_reduction_pct = rng.uniform(8.0, 15.0)
        prefill_gain_pct = rng.uniform(10.0, 20.0)
        expert_shift_pct = rng.uniform(-3.0, 3.0)

        phase_entry = {
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

        delta_entry = {
            "token_savings_pct": token_gain_pct,
            "kv_cache_hit_rate_pct": kv_gain_pct,
            "ttft_pct": ttft_reduction_pct,
            "ttft_phase_change_pct": -ttft_reduction_pct,
            "e2e_latency_pct": e2e_reduction_pct,
            "e2e_latency_phase_change_pct": -e2e_reduction_pct,
            "prefill_throughput_pct": prefill_gain_pct,
            "expert_cache_hit_rate_pct": expert_shift_pct,
        }

        phase_a[workload_name] = phase_entry
        delta_pct[workload_name] = delta_entry

    return phase_a, delta_pct


def build_payload(
    *,
    baseline: dict[str, dict[str, float]],
    phase_a: dict[str, dict[str, float]],
    delta_pct: dict[str, dict[str, float]],
    sidecar_url: str,
    backend_url: str,
) -> dict[str, Any]:
    shared_prefix_ttft_delta = delta_pct.get("shared_prefix_rag", {}).get(
        "ttft_pct", 0.0
    )
    go_no_go = bool(shared_prefix_ttft_delta > -5.0)

    return {
        "mode": "dry-run",
        "baseline_source": DEFAULT_BASELINE,
        "backend_url": backend_url,
        "sidecar_url": sidecar_url,
        "baseline": baseline,
        "phase_a": phase_a,
        "delta_pct": delta_pct,
        "go_no_go": go_no_go,
    }


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    baseline_path = Path(args.baseline)

    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0")

    if not args.dry_run:
        if not check_server_health(args.sidecar_url):
            print(
                f"Server not reachable at {args.sidecar_url}. Start MoE-Infinity first."
            )
            return 1
        if not check_server_health(args.backend_url):
            print(
                f"Server not reachable at {args.backend_url}. Start MoE-Infinity first."
            )
            return 1

        backend_results = run_workload_benchmark(
            server_url=args.backend_url,
            model=args.model,
            max_tokens=args.max_tokens,
        )
        sidecar_results = run_workload_benchmark(
            server_url=args.sidecar_url,
            model=args.model,
            max_tokens=args.max_tokens,
        )
        delta_pct = _compute_delta(backend_results, sidecar_results)

        ttft_delta = delta_pct.get("shared_prefix_rag", {}).get("ttft_pct", 0.0)
        payload: dict[str, Any] = {
            "mode": "real",
            "baseline_source": str(baseline_path),
            "backend_url": args.backend_url,
            "sidecar_url": args.sidecar_url,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "baseline": backend_results,
            "phase_a": sidecar_results,
            "delta_pct": delta_pct,
            "go_no_go": bool(ttft_delta > -5.0),
        }

        if baseline_path.exists():
            try:
                baseline_reference = load_baseline_workloads(baseline_path)
                payload["baseline_reference"] = baseline_reference
                payload["delta_pct_reference"] = {
                    "phase_a_vs_reference": _compute_delta(
                        baseline_reference, sidecar_results
                    ),
                    "backend_vs_reference": _compute_delta(
                        baseline_reference, backend_results
                    ),
                }
            except Exception:
                pass

        write_json(output_path, payload)
        print(
            f"Phase A real benchmark complete. Results written to {output_path}"
        )
        print(
            "TTFT delta (shared_prefix_rag):",
            payload["delta_pct"].get("shared_prefix_rag", {}).get("ttft_pct"),
            "%",
        )
        print("GO/NO-GO:", payload.get("go_no_go"))
        return 0

    baseline = load_baseline_workloads(baseline_path)
    phase_a, delta_pct = simulate_phase_a(baseline)
    payload = build_payload(
        baseline=baseline,
        phase_a=phase_a,
        delta_pct=delta_pct,
        sidecar_url=args.sidecar_url,
        backend_url=args.backend_url,
    )
    write_json(output_path, payload)

    print(f"Phase A dry-run complete. Results written to {output_path}")
    print(
        "TTFT delta (shared_prefix_rag):",
        payload["delta_pct"].get("shared_prefix_rag", {}).get("ttft_pct"),
        "%",
    )
    print("GO/NO-GO:", payload.get("go_no_go"))
    return 0


if __name__ == "__main__":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    raise SystemExit(main())
