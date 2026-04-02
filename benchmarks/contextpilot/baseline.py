from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportMissingImports=false, reportPrivateLocalImportUsage=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportUnusedParameter=false, reportAttributeAccessIssue=false, reportImplicitStringConcatenation=false
import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"
DEFAULT_OUTPUT = "benchmarks/contextpilot/results/baseline.json"
DEFAULT_MAX_NEW_TOKENS = 32
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from benchmarks.contextpilot.benchmark_utils import MetricsCollector, StopWatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline benchmark infrastructure for ContextPilot integration."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name or local checkpoint path. Real runs require this model to be downloaded.",
    )
    parser.add_argument(
        "--offload-dir",
        default="./offload_dir",
        help="Directory used for MoE expert offload storage.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to write benchmark JSON results.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Max generated tokens per request.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate benchmark structure and output schema without loading model/GPU.",
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


def _extract_hits_misses(values: object) -> tuple[float, float] | None:
    if isinstance(values, dict):
        hit = None
        miss = None
        for key in ("hit", "hits", "cache_hits"):
            hit = _to_float(values.get(key))
            if hit is not None:
                break
        for key in ("miss", "misses", "cache_misses"):
            miss = _to_float(values.get(key))
            if miss is not None:
                break
        if hit is not None and miss is not None:
            return hit, miss
        return None

    if isinstance(values, (list, tuple)) and len(values) >= 2:
        hit = _to_float(values[0])
        miss = _to_float(values[1])
        if hit is not None and miss is not None:
            return hit, miss
        return None

    hit = _to_float(getattr(values, "hit", None))
    if hit is None:
        hit = _to_float(getattr(values, "hits", None))
    miss = _to_float(getattr(values, "miss", None))
    if miss is None:
        miss = _to_float(getattr(values, "misses", None))
    if hit is not None and miss is not None:
        return hit, miss
    return None


def extract_expert_hit_rate(dispatcher: object) -> float:
    if dispatcher is None:
        return 0.0

    for method_name in (
        "get_expert_cache_hit_rate",
        "get_cache_hit_rate",
        "expert_cache_hit_rate",
        "cache_hit_rate",
        "hit_rate",
    ):
        member = getattr(dispatcher, method_name, None)
        value = member() if callable(member) else member
        rate = _to_float(value)
        if rate is not None and 0.0 <= rate <= 1.0:
            return rate

    for counts_name in (
        "get_expert_cache_counts",
        "get_cache_counts",
        "expert_cache_counts",
    ):
        member = getattr(dispatcher, counts_name, None)
        values = member() if callable(member) else member
        hits_misses = _extract_hits_misses(values)
        if hits_misses is None:
            continue
        hits, misses = hits_misses
        total = hits + misses
        if total <= 0:
            return 0.0
        return hits / total
    return 0.0


def extract_kv_cache_hit_rate(model: object) -> float:
    engine = getattr(model, "engine", None)
    if engine is None:
        return 0.0

    for holder_name in ("prefix_cache", "kv_cache", "cache_manager"):
        holder = getattr(engine, holder_name, None)
        if holder is None:
            continue
        for metric_name in ("hit_rate", "kv_cache_hit_rate", "cache_hit_rate"):
            member = getattr(holder, metric_name, None)
            value = member() if callable(member) else member
            rate = _to_float(value)
            if rate is not None and 0.0 <= rate <= 1.0:
                return rate
    return 0.0


def environment_info() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {
            "torch_version": None,
            "torch_cuda_version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count()
    info: dict[str, Any] = {
        "torch_version": getattr(torch, "__version__", "unknown"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
    }
    if cuda_available and cuda_device_count > 0:
        info["cuda_device_names"] = [
            torch.cuda.get_device_name(idx) for idx in range(cuda_device_count)
        ]
    return info


def shared_prefix_rag_prompts() -> list[str]:
    docs = [
        "[DocA] ContextPilot deduplicates repeated retrieval passages.",
        "[DocB] Prefix reuse can reduce prefill compute and latency.",
        "[DocC] Baseline run should report token_savings_pct as 0.0.",
    ]
    prefix = "\n".join(docs)
    return [
        f"{prefix}\n\nQuestion {idx}: summarize the documents with focus {idx}."
        for idx in range(1, 6)
    ]


def multi_turn_conversation_prompts() -> list[str]:
    turns: list[str] = []
    memory_blocks: list[str] = []
    for turn in range(1, 11):
        memory_blocks.append(
            f"[Memory-{turn}] User preference and summary from turn {turn}."
        )
        history = "\n".join(memory_blocks)
        turns.append(
            f"System: You are a helpful assistant.\n{history}\nUser turn {turn}: continue planning."
        )
    return turns


def batch_with_overlap_prompts() -> list[str]:
    shared = (
        "Common brief: project constraints, shared incident timeline, and root-cause notes. "
        "This segment is intentionally reused across requests."
    )
    prompts: list[str] = []
    for req_id in range(1, 9):
        unique = (
            f" Unique tail for request {req_id} with scenario-specific details."
        )
        prompts.append(f"{shared}\n{shared}\n{unique}")
    return prompts


def no_overlap_baseline_prompts() -> list[str]:
    return [
        f"Independent request {req_id}: completely unrelated prompt body {req_id * 17}."
        for req_id in range(1, 9)
    ]


def workload_prompts() -> dict[str, list[str]]:
    return {
        "shared_prefix_rag": shared_prefix_rag_prompts(),
        "multi_turn_conversation": multi_turn_conversation_prompts(),
        "batch_with_overlap": batch_with_overlap_prompts(),
        "no_overlap_baseline": no_overlap_baseline_prompts(),
    }


def run_dry_run() -> dict[str, dict[str, float]]:
    workloads = workload_prompts()
    results: dict[str, dict[str, float]] = {}
    for workload_name, prompts in workloads.items():
        collector = MetricsCollector()
        for _ in prompts:
            collector.add(
                ttft=0.0,
                prefill_throughput=0.0,
                kv_cache_hit_rate=0.0,
                token_savings_pct=0.0,
                e2e_latency=0.0,
                expert_cache_hit_rate=0.0,
            )
        results[workload_name] = collector.summarize_for_baseline()
    return results


def load_model_and_tokenizer(
    model_name: str, offload_dir: str
) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoTokenizer

    from moe_infinity import MoE

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = {
        "offload_path": offload_dir,
        "device_memory_ratio": 0.75,
    }
    model = MoE(model_name, config)
    return model, tokenizer, torch


def run_real_benchmark(
    model: Any,
    tokenizer: Any,
    torch: Any,
    *,
    max_new_tokens: int,
) -> dict[str, dict[str, float]]:
    workloads = workload_prompts()
    results: dict[str, dict[str, float]] = {}

    for workload_name, prompts in workloads.items():
        collector = MetricsCollector()
        dispatcher = getattr(
            getattr(model, "engine", None), "expert_dispatcher", None
        )

        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt")
            input_ids = encoded.input_ids.to("cuda:0")

            stopwatch = StopWatch(dispatcher=dispatcher)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_e2e = time.perf_counter()

            _ = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                streamer=stopwatch,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_e2e = time.perf_counter()
            stopwatch.end()

            ttft = float(stopwatch.prefilling_time or 0.0)
            prefill_tokens = int(input_ids.shape[-1])
            prefill_throughput = (
                float(prefill_tokens / ttft) if ttft > 0.0 else 0.0
            )
            kv_cache_hit_rate = extract_kv_cache_hit_rate(model)
            expert_cache_hit_rate = extract_expert_hit_rate(dispatcher)
            collector.add(
                ttft=ttft,
                prefill_throughput=prefill_throughput,
                kv_cache_hit_rate=kv_cache_hit_rate,
                token_savings_pct=0.0,
                e2e_latency=float(end_e2e - start_e2e),
                expert_cache_hit_rate=expert_cache_hit_rate,
            )

        results[workload_name] = collector.summarize_for_baseline()

    return results


def main() -> int:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0")

    output_path = Path(args.output)
    env = environment_info()

    if args.dry_run:
        workloads = run_dry_run()
        payload: dict[str, Any] = {
            "status": "PASS",
            "mode": "dry-run",
            "model_requirement": "Real benchmark requires model download and CUDA-capable GPU.",
            "requested_model": args.model,
            "offload_dir": args.offload_dir,
            "environment": env,
            "workloads": workloads,
        }
        write_json(output_path, payload)
        print(f"Dry-run complete. Results written to {output_path}")
        return 0

    if not env.get("cuda_available", False):
        payload = {
            "status": "BLOCKED",
            "reason": "No CUDA device for real benchmark run",
            "model_requirement": "Real benchmark requires model download and CUDA-capable GPU.",
            "requested_model": args.model,
            "offload_dir": args.offload_dir,
            "environment": env,
            "workloads": run_dry_run(),
        }
        write_json(output_path, payload)
        print("BLOCKED: CUDA unavailable. Use --dry-run for schema validation.")
        return 0

    try:
        model, tokenizer, torch = load_model_and_tokenizer(
            args.model, args.offload_dir
        )
    except Exception as exc:
        payload = {
            "status": "BLOCKED",
            "reason": f"{type(exc).__name__}: {exc}",
            "model_requirement": "Real benchmark requires model download and CUDA-capable GPU.",
            "requested_model": args.model,
            "offload_dir": args.offload_dir,
            "environment": env,
            "workloads": run_dry_run(),
        }
        write_json(output_path, payload)
        print(f"BLOCKED: {type(exc).__name__}: {exc}")
        return 0

    workloads = run_real_benchmark(
        model,
        tokenizer,
        torch,
        max_new_tokens=args.max_new_tokens,
    )
    payload = {
        "status": "PASS",
        "mode": "real",
        "model_requirement": "Real benchmark requires model download and CUDA-capable GPU.",
        "requested_model": args.model,
        "offload_dir": args.offload_dir,
        "environment": env,
        "workloads": workloads,
    }
    write_json(output_path, payload)
    print(f"Benchmark complete. Results written to {output_path}")
    return 0


if __name__ == "__main__":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    raise SystemExit(main())
