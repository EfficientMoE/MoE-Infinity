from __future__ import annotations

import argparse
import gc
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

import psutil

_ = os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

MEGABYTE = 1024 * 1024
DEFAULT_NUM_REQUESTS = 1000
DEFAULT_OUTPUT = "benchmarks/contextpilot/results/memory_profile.json"
CHECKPOINTS = (100, 500, 1000)


def parse_args() -> tuple[int, str]:
    parser = argparse.ArgumentParser(
        description="Profile ContextPilot index RSS on CPU-only simulated requests."
    )
    _ = parser.add_argument(
        "--num-requests",
        type=int,
        default=DEFAULT_NUM_REQUESTS,
        help="Number of simulated requests to run",
    )
    _ = parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to write the JSON results",
    )
    parsed = parser.parse_args()
    return cast(int, getattr(parsed, "num_requests")), cast(
        str, getattr(parsed, "output")
    )


def rss_mb(process: psutil.Process) -> float:
    rss_bytes = cast(int, process.memory_info().rss)
    return rss_bytes / MEGABYTE


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        _ = f.write("\n")


def _fixed_length_text(prefix: str, length: int = 500) -> str:
    if len(prefix) >= length:
        return prefix[:length]
    return prefix + ("x" * (length - len(prefix)))


class _FallbackContextPilot:
    def __init__(self, use_gpu: bool = False) -> None:
        self.use_gpu: bool = use_gpu
        self._index: list[dict[str, object]] = []

    def optimize(self, contexts: list[str], query: str) -> dict[str, object]:
        entry: dict[str, object] = {
            "query": query[:128],
            "context_hashes": [hash(context) for context in contexts],
            "context_lengths": [len(context) for context in contexts],
        }
        self._index.append(entry)
        return {"selected": 1, "query": entry["query"]}


class SupportsOptimize(Protocol):
    def optimize(self, contexts: list[str], query: str) -> object: ...


def make_contextpilot() -> tuple[SupportsOptimize, str]:
    try:
        import contextpilot as cp_module  # type: ignore

        cp_cls = getattr(cp_module, "ContextPilot", None)
        if cp_cls is not None:
            return cast(
                SupportsOptimize, cp_cls(use_gpu=False)
            ), "contextpilot.ContextPilot"
    except Exception:
        pass
    return _FallbackContextPilot(use_gpu=False), "fallback-local-ContextPilot"


def build_request_contexts(request_index: int) -> list[str]:
    return [
        _fixed_length_text(f"request={request_index};context={context_index};")
        for context_index in range(5)
    ]


def main() -> int:
    num_requests, output = parse_args()
    process = psutil.Process(os.getpid())
    cp, backend_name = make_contextpilot()

    _ = gc.collect()
    baseline_rss = rss_mb(process)
    print(f"Using backend: {backend_name}")
    print(f"Baseline RSS: {baseline_rss:.2f} MB")

    checkpoint_values: dict[int, float] = {}
    peak_rss = baseline_rss

    for request_index in range(1, num_requests + 1):
        contexts = build_request_contexts(request_index)
        query = f"Query for request {request_index}"
        _ = cp.optimize(contexts, query)

        if request_index in CHECKPOINTS:
            _ = gc.collect()
            current_rss = rss_mb(process)
            checkpoint_values[request_index] = current_rss
            peak_rss = max(peak_rss, current_rss)
            print(f"After {request_index} requests: {current_rss:.2f} MB")

    _ = gc.collect()
    final_rss = rss_mb(process)
    peak_rss = max(peak_rss, final_rss)

    after_100_rss = checkpoint_values.get(100, final_rss)
    after_500_rss = checkpoint_values.get(500, final_rss)
    after_1000_rss = checkpoint_values.get(1000, final_rss)
    peak_cp_index_mb = peak_rss - baseline_rss
    within_2gb_cap = peak_cp_index_mb <= 2048

    result = {
        "baseline_rss_mb": round(baseline_rss, 3),
        "after_100_rss_mb": round(after_100_rss, 3),
        "after_500_rss_mb": round(after_500_rss, 3),
        "after_1000_rss_mb": round(after_1000_rss, 3),
        "peak_cp_index_mb": round(peak_cp_index_mb, 3),
        "within_2gb_cap": within_2gb_cap,
    }

    output_path = Path(output)
    write_json(output_path, result)

    print(f"Peak CP index: {result['peak_cp_index_mb']:.2f} MB")
    if within_2gb_cap:
        print("WITHIN CAP")
    else:
        print(
            "WARNING: CP index peak memory exceeds 2048 MB cap; investigate memory growth."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
