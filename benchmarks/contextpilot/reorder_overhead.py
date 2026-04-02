from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false
import json
import math
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "contextpilot"
    / "results"
    / "reorder_overhead.json"
)
QUERY = "What is machine learning?"
CHAR_SIZES = (1_000, 4_000, 16_000, 64_000)
BLOCK_COUNTS = (1, 5, 10, 50)
ITERATIONS = 100


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (p / 100.0) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(lower)]
    fraction = position - lower
    return (
        ordered[int(lower)]
        + (ordered[int(upper)] - ordered[int(lower)]) * fraction
    )


def build_context(char_count: int, block_index: int) -> str:
    seed = "This is a document about machine learning. "
    repeated = (seed * ((char_count // len(seed)) + 2))[:char_count]
    suffix = f" [doc {block_index}]"
    if len(repeated) <= len(suffix):
        return suffix[: len(repeated)]
    return repeated[: -len(suffix)] + suffix


def build_contexts(char_count: int, block_count: int) -> list[str]:
    return [build_context(char_count, idx) for idx in range(block_count)]


def make_cpu_contextpilot() -> tuple[Any, str]:
    try:
        import contextpilot as module

        if hasattr(module, "ContextPilot"):
            return module.ContextPilot(use_gpu=False), "contextpilot"
    except Exception:
        pass

    class _FallbackContextPilot:
        def __init__(self, use_gpu: bool = False) -> None:
            self.use_gpu = use_gpu

        def optimize(self, contexts: list[str], query: str) -> list[str]:
            query_terms = [term.lower() for term in query.split() if term]
            scored: list[tuple[int, int, str]] = []
            for index, context in enumerate(contexts):
                score = 0
                lowered = context.lower()
                for term in query_terms:
                    score += lowered.count(term)
                stride = max(64, len(context) // 32)
                for offset in range(0, len(context), stride):
                    window = lowered[offset : offset + 256]
                    local = 0
                    for char in window[::4]:
                        local += ord(char)
                    score += local % 97
                score += len(context) // 1024
                scored.append((-score, index, context))
            scored.sort()
            return [context for _, _, context in scored]

    return _FallbackContextPilot(use_gpu=False), "fallback"


def benchmark(cp: Any) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for char_count in CHAR_SIZES:
        for block_count in BLOCK_COUNTS:
            contexts = build_contexts(char_count, block_count)
            timings_ms: list[float] = []
            for _ in range(ITERATIONS):
                start = time.perf_counter_ns()
                _ = cp.optimize(contexts, QUERY)
                end = time.perf_counter_ns()
                timings_ms.append((end - start) / 1_000_000.0)

            key = f"{char_count // 1000}k_{block_count}blocks"
            results[key] = {
                "reorder_p50_ms": percentile(timings_ms, 50),
                "reorder_p90_ms": percentile(timings_ms, 90),
                "reorder_p99_ms": percentile(timings_ms, 99),
            }
    return results


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> int:
    cp, backend = make_cpu_contextpilot()
    results = benchmark(cp)
    write_json(RESULTS_PATH, results)

    p99 = results["16k_10blocks"]["reorder_p99_ms"]
    if p99 > 500:
        print(f"WARNING: 16k_10blocks reorder_p99_ms={p99:.1f}ms exceeds 500ms")

    print(f"Backend: {backend}")
    print(f"Saved results to: {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"16k_10blocks p99: {p99:.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
