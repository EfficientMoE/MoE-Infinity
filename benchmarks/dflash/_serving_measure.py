"""GPU measurement for the PD-DFlash B0-B3 serving experiment (Task 2).

Private helper for ``benchmarks.dflash.pd_dflash_serving``; imported only inside
``run_experiment`` so the CLI module stays torch-free at import. Every function
here needs a live RTX PRO 6000 with FP4-offloaded experts and cached
checkpoints, so nothing in this file is exercised by CPU pytest -- it is the
hardware harness a human runs for plan Task 3.

The measurement mirrors ``tests/python/dflash/test_gpu_serving_dflash.py``: build
``MoE`` with an offload path, wrap a ``DFlashSpeculator`` for the DFlash
baselines, drive deterministic greedy requests through the continuous-batching
engine, and read metrics from measured wall clock, the speculator ``step_trace``,
and the instrumented ``RouteAheadStats``. Where a native occupancy/hit-rate
accessor is not present the extractor falls back to ``0.0`` and records a
per-row ``warnings`` entry, so a row is always schema-valid *and* honest about
which term needs a human to wire a native accessor.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

from benchmarks.dflash.pd_dflash_serving import (
    NVTX_RANGES,
    RunnerArgs,
    make_observation_row,
    require_offloaded,
)

try:
    import nvtx as _nvtx
except Exception:  # pragma: no cover - nvtx optional
    _nvtx = None

RESIDENT_MEMORY_RATIO = 0.98
DETERMINISTIC_PROMPT = (
    "Explain in one paragraph why offloaded mixture-of-experts serving "
    "benefits from speculative decoding."
)


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    """Push an NVTX range so the BM4 overlap parser can attribute H2D bytes."""
    if _nvtx is None or name not in NVTX_RANGES:
        yield
        return
    handle = _nvtx.start_range(message=name, color="green")
    try:
        yield
    finally:
        _nvtx.end_range(handle)


def measure_configuration(
    *,
    args: RunnerArgs,
    baseline: str,
    draft: str,
    block_size: int,
    concurrency: int,
) -> Dict[str, Any]:
    """Measure one ``(baseline, block, concurrency)`` cell and return its row.

    B0 runs the AR offloaded target with no speculator; B1/B3 and the ``OURS``
    configuration wrap a DFlash draft, with route-ahead stats enabled so
    coverage and byte-accurate waste are recorded. B3 loads the target resident
    (no offload upper bound); the other baselines require genuinely offloaded
    experts and are refused otherwise.
    """
    import torch

    from moe_infinity import MoE
    from moe_infinity.spec_decode import DFlashSpeculator

    warnings: List[str] = []
    resident = baseline == "B3"
    memory_ratio = (
        RESIDENT_MEMORY_RATIO if resident else args.device_memory_ratio
    )
    model = MoE(
        args.model,
        {
            "offload_path": args.offload_dir,
            "device_memory_ratio": memory_ratio,
        },
    )
    engine = model.engine
    if not resident:
        require_offloaded(baseline, _count_offloaded_experts(engine))

    speculator = None
    if baseline != "B0":
        speculator = DFlashSpeculator(model, draft)
        enable = getattr(speculator, "enable_route_ahead_stats", None)
        if callable(enable):
            enable()

    torch.manual_seed(args.seed)
    prompt_ids = _deterministic_prompt_ids(model, args.model)

    _warmup(model, prompt_ids, speculator, args.warmup_rounds, block_size)

    torch.cuda.synchronize()
    started = time.perf_counter()
    generated = _run_requests(
        model=model,
        prompt_ids=prompt_ids,
        speculator=speculator,
        block_size=block_size,
        concurrency=concurrency,
        num_requests=args.requests,
    )
    torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - started, 1e-9)

    ttft = _measure_ttft(model, prompt_ids, speculator, block_size)

    metrics = _collect_metrics(
        baseline=baseline,
        block_size=block_size,
        elapsed=elapsed,
        ttft_seconds=ttft,
        generated_tokens=generated,
        num_requests=args.requests,
        speculator=speculator,
        engine=engine,
        slo_ms=args.slo_ms,
        warnings=warnings,
    )
    cost_terms = _collect_cost_terms(
        baseline=baseline,
        speculator=speculator,
        engine=engine,
        measured_h2d_gbps=args.measured_h2d_gbps,
        warnings=warnings,
    )
    return make_observation_row(
        model=args.model,
        draft=draft if baseline != "B0" else "",
        baseline=baseline,
        block_size=block_size,
        concurrency=concurrency,
        repeat=0,
        metrics=metrics,
        cost_terms=cost_terms,
        warnings=warnings or None,
    )


def _count_offloaded_experts(engine: Any) -> int:
    for attr in ("num_offloaded_experts", "offloaded_expert_count"):
        value = getattr(engine, attr, None)
        if isinstance(value, int):
            return value
    prefetcher = getattr(engine, "expert_prefetcher", None)
    nbytes_map = getattr(prefetcher, "expert_nbytes_map", None)
    if isinstance(nbytes_map, dict):
        return len(nbytes_map)
    return 0


def _deterministic_prompt_ids(model: Any, repo: str) -> List[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        repo, trust_remote_code=True, local_files_only=True
    )
    return [
        int(tok)
        for tok in tokenizer(DETERMINISTIC_PROMPT, return_tensors="pt")
        .input_ids[0]
        .tolist()
    ]


def _greedy_generate(
    model: Any,
    prompt_ids: List[int],
    speculator: Optional[Any],
    max_new_tokens: int,
) -> List[int]:
    import torch

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda:0")
    kwargs: Dict[str, Any] = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
    }
    if speculator is not None:
        kwargs["speculative_draft"] = speculator
    with nvtx_range("target_verify"):
        output = model.generate(input_ids, **kwargs)
    return [int(tok) for tok in output[0, len(prompt_ids) :].tolist()]


def _warmup(
    model: Any,
    prompt_ids: List[int],
    speculator: Optional[Any],
    warmup_rounds: int,
    block_size: int,
) -> None:
    for _ in range(max(0, warmup_rounds)):
        _greedy_generate(model, prompt_ids, speculator, max(1, block_size))


def _run_requests(
    *,
    model: Any,
    prompt_ids: List[int],
    speculator: Optional[Any],
    block_size: int,
    concurrency: int,
    num_requests: int,
) -> int:
    tokens_per_request = max(block_size * 4, 32)
    total = 0
    for _ in range(max(1, num_requests)):
        generated = _greedy_generate(
            model, prompt_ids, speculator, tokens_per_request
        )
        total += len(generated)
    return total


def _measure_ttft(
    model: Any,
    prompt_ids: List[int],
    speculator: Optional[Any],
    block_size: int,
) -> float:
    import torch

    torch.cuda.synchronize()
    started = time.perf_counter()
    with nvtx_range("dflash_draft"):
        _greedy_generate(model, prompt_ids, speculator, 1)
    torch.cuda.synchronize()
    return max(time.perf_counter() - started, 0.0)


def _acceptance_length(
    baseline: str, block_size: int, speculator: Any
) -> float:
    if baseline == "B0" or speculator is None:
        return 1.0
    trace = list(getattr(speculator, "step_trace", []) or [])
    if not trace:
        return 1.0
    accepted = [
        min(int(getattr(r, "accept", 0)) + 1, block_size) for r in trace
    ]
    return sum(accepted) / len(accepted)


def _route_ahead_snapshot(speculator: Any) -> Tuple[float, Optional[int]]:
    if speculator is None:
        return 0.0, 0
    stats = getattr(speculator, "route_ahead_stats", None)
    if stats is None:
        return 0.0, 0
    snapshot = stats.as_dict()
    coverage = float(snapshot.get("coverage", 0.0) or 0.0)
    wasted = snapshot.get("wasted_prefetch_bytes")
    return coverage, (int(wasted) if wasted is not None else None)


def _extract_float(source: Any, names: Tuple[str, ...]) -> Optional[float]:
    for name in names:
        value = getattr(source, name, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                value = None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _collect_metrics(
    *,
    baseline: str,
    block_size: int,
    elapsed: float,
    ttft_seconds: float,
    generated_tokens: int,
    num_requests: int,
    speculator: Any,
    engine: Any,
    slo_ms: Optional[float],
    warnings: List[str],
) -> Dict[str, float]:
    tokens_per_second = generated_tokens / elapsed
    acceptance = _acceptance_length(baseline, block_size, speculator)
    rounds = max(1.0, generated_tokens / max(acceptance, 1.0))
    per_round_latency = elapsed / rounds
    coverage, wasted_bytes = _route_ahead_snapshot(speculator)
    if wasted_bytes is None:
        warnings.append(
            "wasted_prefetch_bytes unavailable from RouteAheadStats; a route-"
            "ahead configuration on offloaded experts must report real bytes"
        )
        wasted_bytes = 0

    hit_rate = _extract_float(
        getattr(engine, "expert_prefetcher", engine),
        ("get_hit_rate", "hit_rate", "expert_hit_rate"),
    )
    if hit_rate is None:
        hit_rate = _extract_float(engine, ("get_hit_rate", "hit_rate"))
    if hit_rate is None:
        warnings.append("expert_cache_hit_rate fell back to 0.0")
        hit_rate = 0.0

    expert_occupancy = _extract_float(
        engine, ("expert_occupancy_bytes", "get_expert_occupancy_bytes")
    )
    if expert_occupancy is None:
        warnings.append("expert_occupancy_bytes fell back to 0.0")
        expert_occupancy = 0.0
    kv_occupancy = _extract_float(
        engine, ("kv_occupancy_bytes", "get_kv_occupancy_bytes")
    )
    if kv_occupancy is None:
        warnings.append("kv_occupancy_bytes fell back to 0.0")
        kv_occupancy = 0.0

    if slo_ms is None:
        goodput = tokens_per_second
    else:
        met_slo = per_round_latency * 1000.0 <= slo_ms
        goodput = tokens_per_second if met_slo else 0.0

    return {
        "output_tokens_per_second": tokens_per_second,
        "acceptance_length_a": acceptance,
        "ttft_seconds": ttft_seconds,
        "per_round_latency_seconds": per_round_latency,
        "goodput_at_slo": goodput,
        "expert_cache_hit_rate": hit_rate,
        "route_ahead_prefetch_coverage": coverage,
        "wasted_prefetch_bytes": float(wasted_bytes),
        "expert_occupancy_bytes": expert_occupancy,
        "kv_occupancy_bytes": kv_occupancy,
    }


def _collect_cost_terms(
    *,
    baseline: str,
    speculator: Any,
    engine: Any,
    measured_h2d_gbps: Optional[float],
    warnings: List[str],
) -> Dict[str, Any]:
    coverage, wasted_bytes = _route_ahead_snapshot(speculator)
    terms: Dict[str, Any] = {
        "route_ahead_coverage": coverage,
        "wasted_prefetch_bytes": wasted_bytes,
    }
    if measured_h2d_gbps is not None:
        terms["measured_h2d_bytes_per_second"] = (
            measured_h2d_gbps * 1_000_000_000.0
        )
    else:
        warnings.append(
            "measured_h2d_bytes_per_second not supplied; pass --measured-h2d-"
            "gbps from a device bandwidth probe for the hide inequality"
        )
    return terms
