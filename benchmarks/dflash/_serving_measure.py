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
    effective_h2d_gbps = args.measured_h2d_gbps
    if effective_h2d_gbps is None and getattr(args, "probe_h2d", False):
        effective_h2d_gbps = _probe_h2d_gbps()
    cost_terms = _collect_cost_terms(
        baseline=baseline,
        speculator=speculator,
        engine=engine,
        measured_h2d_gbps=effective_h2d_gbps,
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


def measure_configuration_serving(
    *,
    args: RunnerArgs,
    baseline: str,
    draft: str,
    block_size: int,
    concurrency: int,
) -> Dict[str, Any]:
    """Measure a serving-path baseline (B2) via ``ContinuousBatchingEngine``.

    Unlike ``measure_configuration`` (which drives ``MoE.generate`` and has no
    serving KV manager, so ``kv_occupancy_bytes`` falls back to ``0.0``), this
    builds the continuous-batching engine with the four Task-6 verify budgets so
    ``_step_speculative_session`` and the 2-D admission scheduler govern every
    VERIFY round, and reads a real ``kv_occupancy_bytes`` from the serving
    ``PagedKVCache``. Acceptance is captured from the per-round ``VerifyResult``
    and native expert-cache counters are cleared so the hit rate reflects only
    the timed run.
    """
    import math

    import torch
    from transformers import AutoTokenizer

    from benchmarks.dflash.pd_dflash_serving import build_b2_serving_config
    from moe_infinity import MoE
    from moe_infinity.serving.engine import ContinuousBatchingEngine
    from moe_infinity.serving.sequence import SamplingParams
    from moe_infinity.spec_decode import DFlashSpeculator

    warnings: List[str] = []
    model = MoE(
        args.model,
        {
            "offload_path": args.offload_dir,
            "device_memory_ratio": args.device_memory_ratio,
        },
    )
    engine = model.engine
    require_offloaded(baseline, _count_offloaded_experts(engine))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    speculator = DFlashSpeculator(model, draft)
    enable = getattr(speculator, "enable_route_ahead_stats", None)
    if callable(enable):
        enable()

    torch.manual_seed(args.seed)
    prompt_ids = _deterministic_prompt_ids(model, args.model)
    max_new = max(block_size * 4, 32)

    model_config = model.model.config
    num_layers = _model_int(model_config, "num_hidden_layers", "num_layers")
    num_kv_heads = _model_int(
        model_config,
        "num_key_value_heads",
        "num_kv_heads",
        "num_attention_heads",
    )
    head_dim = _model_int(model_config, "head_dim")
    dtype_str = str(getattr(model.model, "dtype", torch.bfloat16)).replace(
        "torch.", ""
    )
    blocks_per_seq = math.ceil(
        (len(prompt_ids) + max_new + block_size) / block_size
    )
    num_kv_blocks = blocks_per_seq * max(1, concurrency) + max(1, concurrency)

    serving_config = build_b2_serving_config(
        block_size=block_size,
        concurrency=concurrency,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype_str,
        eos_token_id=getattr(model_config, "eos_token_id", None),
        num_kv_blocks=num_kv_blocks,
        device_memory_ratio=args.device_memory_ratio,
    )
    serving = ContinuousBatchingEngine(
        model=model.model,
        engine=model.engine,
        config=serving_config,
        tokenizer=tokenizer,
        speculative_draft=speculator,
    )
    if not serving.scheduler.verify_scheduling_enabled:
        raise RuntimeError(
            "B2 requires the 2-D verify scheduler; verify budgets did not "
            "enable it -- check build_b2_serving_config"
        )

    accepts: List[int] = []
    _wrap_verify_round(speculator, accepts)
    peak_used_blocks = _install_kv_peak_probe(serving.kv_cache)
    _reset_expert_cache_counts(engine)

    sampling = SamplingParams(
        temperature=0.0, top_k=0, top_p=1.0, max_tokens=max_new
    )

    for warmup_index in range(max(0, args.warmup_rounds)):
        _run_one_serving_request(
            serving, f"b2-warmup-{warmup_index}", prompt_ids, sampling
        )
    accepts.clear()
    peak_used_blocks["value"] = 0
    _reset_expert_cache_counts(engine)

    torch.cuda.synchronize()
    started = time.perf_counter()
    generated = 0
    for index in range(max(1, args.requests)):
        tokens = _run_one_serving_request(
            serving, f"b2-{index}", prompt_ids, sampling
        )
        generated += len(tokens)
    torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - started, 1e-9)

    ttft = _measure_serving_ttft(serving, prompt_ids)

    kv_occupancy = _kv_occupancy_bytes(
        serving.kv_cache,
        peak_used_blocks["value"],
        num_layers=num_layers,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        prompt_tokens=len(prompt_ids),
        max_new=max_new,
        warnings=warnings,
    )

    metrics = _collect_serving_metrics(
        block_size=block_size,
        elapsed=elapsed,
        ttft_seconds=ttft,
        generated_tokens=generated,
        accepts=accepts,
        speculator=speculator,
        engine=engine,
        kv_occupancy=kv_occupancy,
        slo_ms=args.slo_ms,
        warnings=warnings,
    )
    effective_h2d_gbps = args.measured_h2d_gbps
    if effective_h2d_gbps is None and getattr(args, "probe_h2d", False):
        effective_h2d_gbps = _probe_h2d_gbps()
    cost_terms = _collect_cost_terms(
        baseline=baseline,
        speculator=speculator,
        engine=engine,
        measured_h2d_gbps=effective_h2d_gbps,
        warnings=warnings,
    )
    return make_observation_row(
        model=args.model,
        draft=draft,
        baseline=baseline,
        block_size=block_size,
        concurrency=concurrency,
        repeat=0,
        metrics=metrics,
        cost_terms=cost_terms,
        warnings=warnings or None,
    )


def _model_int(config: Any, *names: str) -> int:
    get_text = getattr(config, "get_text_config", None)
    text_config = (
        get_text()
        if callable(get_text)
        else getattr(config, "text_config", None)
    )
    for candidate in (config, text_config):
        if candidate is None:
            continue
        for name in names:
            value = getattr(candidate, name, None)
            if isinstance(value, int):
                return value
    raise RuntimeError(f"unable to resolve any of {names!r} from model config")


def _wrap_verify_round(speculator: Any, accepts: List[int]) -> None:
    original = speculator.verify_round

    def traced(session: Any) -> Any:
        with nvtx_range("target_verify"):
            result = original(session)
        accepts.append(int(getattr(result, "accept", 0)))
        return result

    speculator.verify_round = traced


def _install_kv_peak_probe(kv_cache: Any) -> Dict[str, int]:
    allocator = kv_cache.block_allocator
    original = allocator.allocate
    peak = {"value": 0}

    def tracked(num_blocks: int) -> Any:
        block_ids = original(num_blocks)
        used = kv_cache.num_blocks - allocator.num_free_blocks
        if used > peak["value"]:
            peak["value"] = used
        return block_ids

    allocator.allocate = tracked
    return peak


def _reset_expert_cache_counts(engine: Any) -> None:
    prefetcher = getattr(engine, "expert_prefetcher", None)
    dispatcher = getattr(prefetcher, "expert_dispatcher", None)
    clear = getattr(dispatcher, "clear_expert_cache_counts", None)
    if callable(clear):
        try:
            clear()
        except Exception:
            pass


def _run_one_serving_request(
    serving: Any,
    request_id: str,
    prompt_ids: List[int],
    sampling: Any,
) -> List[int]:
    serving.add_request(
        request_id=request_id,
        prompt_token_ids=list(prompt_ids),
        sampling_params=sampling,
    )
    result = serving.run_until_done()
    tokens = result.get(request_id, [])
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    return [int(t) for t in tokens]


def _measure_serving_ttft(serving: Any, prompt_ids: List[int]) -> float:
    import torch

    from moe_infinity.serving.sequence import SamplingParams

    sampling = SamplingParams(temperature=0.0, top_k=0, top_p=1.0, max_tokens=1)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with nvtx_range("dflash_draft"):
        _run_one_serving_request(serving, "b2-ttft", prompt_ids, sampling)
    torch.cuda.synchronize()
    return max(time.perf_counter() - started, 0.0)


def _kv_occupancy_bytes(
    kv_cache: Any,
    peak_used_blocks: int,
    *,
    num_layers: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    prompt_tokens: int,
    max_new: int,
    warnings: List[str],
) -> float:
    import math

    element_size = int(kv_cache._kv_cache.element_size())
    per_block_bytes = (
        num_layers * 2 * block_size * num_kv_heads * head_dim * element_size
    )
    if peak_used_blocks > 0:
        return float(peak_used_blocks * per_block_bytes)
    resident_blocks = math.ceil((prompt_tokens + max_new) / block_size)
    warnings.append(
        "kv_occupancy_bytes: serving allocator reported no peak; used the "
        "serving PagedKVCache geometry for a resident sequence"
    )
    return float(resident_blocks * per_block_bytes)


def _collect_serving_metrics(
    *,
    block_size: int,
    elapsed: float,
    ttft_seconds: float,
    generated_tokens: int,
    accepts: List[int],
    speculator: Any,
    engine: Any,
    kv_occupancy: float,
    slo_ms: Optional[float],
    warnings: List[str],
) -> Dict[str, float]:
    tokens_per_second = generated_tokens / elapsed
    if accepts:
        acceptance = sum(min(a + 1, block_size) for a in accepts) / len(accepts)
    else:
        acceptance = 1.0
    rounds = max(1.0, generated_tokens / max(acceptance, 1.0))
    per_round_latency = elapsed / rounds

    coverage, wasted_bytes = _route_ahead_snapshot(speculator)
    if wasted_bytes is None:
        native_wasted = _native_wasted_prefetch_bytes(engine)
        if native_wasted is not None:
            wasted_bytes = int(native_wasted)
        else:
            warnings.append(
                "wasted_prefetch_bytes unavailable from RouteAheadStats; a "
                "route-ahead configuration on offloaded experts must report "
                "real bytes"
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
        "kv_occupancy_bytes": float(kv_occupancy),
    }


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


def _probe_h2d_gbps(
    nbytes: int = 256 * 1024 * 1024, iters: int = 20
) -> Optional[float]:
    import torch

    if not torch.cuda.is_available():
        return None
    try:
        elements = max(1, nbytes // 2)
        host = torch.empty(elements, dtype=torch.float16, pin_memory=True)
        device = torch.empty(elements, dtype=torch.float16, device="cuda:0")
        torch.cuda.synchronize()
        device.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(max(1, iters)):
            device.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        elapsed = max(time.perf_counter() - started, 1e-9)
    except Exception:
        return None
    moved = float(elements * 2 * max(1, iters))
    return moved / elapsed / 1_000_000_000.0


def _native_wasted_prefetch_bytes(engine: Any) -> Optional[float]:
    prefetcher = getattr(engine, "expert_prefetcher", None)
    getter = getattr(prefetcher, "wasted_prefetch_bytes", None)
    if callable(getter):
        try:
            return float(getter())
        except Exception:
            return None
    return None


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
        native_wasted = _native_wasted_prefetch_bytes(engine)
        if native_wasted is not None:
            wasted_bytes = int(native_wasted)
        else:
            warnings.append(
                "wasted_prefetch_bytes unavailable from RouteAheadStats; a "
                "route-ahead configuration on offloaded experts must report "
                "real bytes"
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
    if wasted_bytes is None:
        native_wasted = _native_wasted_prefetch_bytes(engine)
        if native_wasted is not None:
            wasted_bytes = int(native_wasted)
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
