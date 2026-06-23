#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEGABYTE = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KV offload benchmark")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--offload-dir", type=str, required=True)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--prompt-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--enable-kv-offload", action="store_true")
    parser.add_argument("--kv-cache-ratio", type=float, default=0.05)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def environment_info() -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count()
    torch_version_ns = getattr(torch, "version", object())
    info: dict[str, Any] = {
        "torch_version": getattr(torch, "__version__", "unknown"),
        "torch_cuda_version": getattr(torch_version_ns, "cuda", None),
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
    }
    if cuda_available and cuda_device_count > 0:
        info["cuda_device_names"] = [
            torch.cuda.get_device_name(idx) for idx in range(cuda_device_count)
        ]
    return info


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _resolve_int_attr(config: object, *names: str) -> int | None:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int):
            return value
    return None


def _resolve_dtype(model: object) -> torch.dtype:
    model_dtype = getattr(model, "dtype", None)
    if isinstance(model_dtype, torch.dtype):
        return model_dtype

    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            parameter_source = parameters()
            if isinstance(parameter_source, Iterator):
                first_param = next(parameter_source, None)
            elif isinstance(parameter_source, Iterable):
                first_param = next(iter(parameter_source), None)
            else:
                first_param = None
        except Exception:
            first_param = None
        if isinstance(first_param, torch.Tensor):
            return first_param.dtype

    cfg = getattr(model, "config", None)
    cfg_dtype = getattr(cfg, "torch_dtype", None)
    if isinstance(cfg_dtype, torch.dtype):
        return cfg_dtype
    if isinstance(cfg_dtype, str):
        mapping = {
            "float16": torch.float16,
            "half": torch.float16,
            "float32": torch.float32,
            "float": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        normalized = cfg_dtype.replace("torch.", "")
        if normalized in mapping:
            return mapping[normalized]

    return torch.float16


def _build_engine_config(
    model: object,
    kv_cache_ratio: float,
) -> dict[str, object]:
    model_config = getattr(model, "config", None)
    if model_config is None:
        raise RuntimeError("model.config is required")

    num_layers = _resolve_int_attr(
        model_config,
        "num_hidden_layers",
        "num_layers",
        "n_layer",
    )
    num_attention_heads = _resolve_int_attr(
        model_config,
        "num_attention_heads",
        "n_head",
    )
    num_kv_heads = _resolve_int_attr(
        model_config,
        "num_key_value_heads",
        "num_kv_heads",
        "n_head_kv",
    )
    hidden_size = _resolve_int_attr(model_config, "hidden_size", "n_embd")
    head_dim = _resolve_int_attr(model_config, "head_dim")
    eos_token_id = _resolve_int_attr(model_config, "eos_token_id")

    if num_layers is None or num_attention_heads is None:
        raise RuntimeError("unable to resolve model layer/head config")
    if num_kv_heads is None:
        num_kv_heads = num_attention_heads
    if head_dim is None:
        if hidden_size is None:
            raise RuntimeError("unable to resolve model head_dim")
        head_dim = hidden_size // max(1, num_attention_heads)

    config: dict[str, object] = {
        "device_memory_ratio": 0.75,
        "kv_cache_ratio": kv_cache_ratio,
        "max_batch_size": 64,
        "max_tokens_per_step": 4096,
        "block_size": 16,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype": _resolve_dtype(model),
    }
    if isinstance(eos_token_id, int):
        config["eos_token_id"] = eos_token_id
    return config


def _repeat_to_length(token_ids: list[int], target_length: int) -> list[int]:
    if target_length <= 0:
        raise ValueError(f"target_length must be > 0, got {target_length}")
    if not token_ids:
        return [0] * target_length
    output: list[int] = []
    while len(output) < target_length:
        output.extend(token_ids)
    return output[:target_length]


def _build_prompt_batches(
    tokenizer: Any, num_requests: int, prompt_length: int
) -> list[list[int]]:
    base_text = (
        "MoE-Infinity KV offload benchmark prompt. "
        "Keep this text deterministic for stable measurements."
    )
    encoded = tokenizer.encode(base_text, add_special_tokens=False)
    prompt_ids = _repeat_to_length(encoded, prompt_length)
    return [list(prompt_ids) for _ in range(num_requests)]


def load_model_and_tokenizer(
    model_name: str, offload_dir: str
) -> tuple[Any, Any]:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError(f"transformers import failed: {exc}") from exc

    try:
        from moe_infinity import MoE
    except Exception as exc:
        raise RuntimeError(f"moe_infinity import failed: {exc}") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    offload_path = Path(offload_dir)
    offload_path.mkdir(parents=True, exist_ok=True)

    model = MoE(
        model_name,
        {
            "offload_path": str(offload_path),
            "device_memory_ratio": 0.75,
        },
    )
    return model, tokenizer


def run_benchmark(
    engine: Any,
    prompt_batches: list[list[int]],
    max_new_tokens: int,
) -> dict[str, float | int | None]:
    from moe_infinity.serving.sequence import SamplingParams

    psutil_module: Any | None
    try:
        import psutil as psutil_module
    except Exception:
        psutil_module = None

    process = psutil_module.Process() if psutil_module is not None else None
    cpu_rss_before = (
        float(process.memory_info().rss / MEGABYTE)
        if process is not None
        else None
    )

    swap_count = [0]
    original_swap_out = engine.kv_cache.swap_out

    def counting_swap_out(seq_id: int) -> None:
        swap_count[0] += 1
        original_swap_out(seq_id)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    engine.eos_token_id = None
    start = time.perf_counter()
    with patch.object(
        engine.kv_cache,
        "swap_out",
        side_effect=counting_swap_out,
    ):
        for req_idx, prompt_ids in enumerate(prompt_batches):
            engine.add_request(
                request_id=f"bench-{req_idx}",
                prompt_token_ids=list(prompt_ids),
                sampling_params=SamplingParams(
                    temperature=0.0,
                    max_tokens=max_new_tokens,
                ),
            )
        outputs = engine.run_until_done()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start

    cpu_rss_after = (
        float(process.memory_info().rss / MEGABYTE)
        if process is not None
        else None
    )
    cpu_rss_mb = None
    if cpu_rss_before is not None and cpu_rss_after is not None:
        cpu_rss_mb = max(cpu_rss_before, cpu_rss_after)

    gpu_peak_mb = (
        float(torch.cuda.max_memory_allocated() / MEGABYTE)
        if torch.cuda.is_available()
        else None
    )

    generated_tokens = sum(len(token_ids) for token_ids in outputs.values())
    num_requests = len(prompt_batches)
    throughput_toks_per_s = (
        0.0 if elapsed_s <= 0 else float(generated_tokens) / elapsed_s
    )
    latency_ms = (
        0.0 if num_requests <= 0 else (elapsed_s * 1000.0) / num_requests
    )

    return {
        "latency_ms": latency_ms,
        "throughput_toks_per_s": throughput_toks_per_s,
        "gpu_peak_mb": gpu_peak_mb,
        "cpu_rss_mb": cpu_rss_mb,
        "swap_count": int(swap_count[0]),
        "generated_tokens": int(generated_tokens),
        "elapsed_s": elapsed_s,
    }


def print_table(
    measurement: dict[str, float | int | None],
    *,
    model: str,
    kv_cache_ratio: float,
    kv_offload_enabled: bool,
) -> None:
    print("=== MoE-Infinity KV Offload Benchmark ===")
    print(f"model                : {model}")
    print(f"kv_offload_enabled   : {kv_offload_enabled}")
    print(f"kv_cache_ratio       : {kv_cache_ratio}")
    print(f"latency_ms           : {measurement['latency_ms']}")
    print(f"throughput_toks_per_s: {measurement['throughput_toks_per_s']}")
    print(f"gpu_peak_mb          : {measurement['gpu_peak_mb']}")
    print(f"cpu_rss_mb           : {measurement['cpu_rss_mb']}")
    print(f"swap_count           : {measurement['swap_count']}")


def main() -> int:
    args = parse_args()
    if args.num_requests <= 0:
        raise ValueError("--num-requests must be > 0")
    if args.prompt_length <= 0:
        raise ValueError("--prompt-length must be > 0")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0")
    if args.kv_cache_ratio <= 0:
        raise ValueError("--kv-cache-ratio must be > 0")

    env = environment_info()

    kv_cache_ratio = args.kv_cache_ratio if args.enable_kv_offload else 0.35
    payload: dict[str, Any] = {
        "status": "BLOCKED",
        "reason": None,
        "environment": env,
        "requested_model": args.model,
        "offload_dir": args.offload_dir,
        "num_requests": args.num_requests,
        "prompt_length": args.prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "enable_kv_offload": bool(args.enable_kv_offload),
        "kv_cache_ratio": kv_cache_ratio,
        "measurement": None,
    }

    if not env["cuda_available"]:
        payload["reason"] = "No CUDA device"
        print("BLOCKED: No CUDA. Run on GPU hardware.")
        if args.output_json:
            write_json(Path(args.output_json), payload)
        return 0

    try:
        model, tokenizer = load_model_and_tokenizer(
            args.model, args.offload_dir
        )
    except Exception as exc:
        payload["reason"] = f"{type(exc).__name__}: {exc}"
        print(f"BLOCKED: {payload['reason']}")
        if args.output_json:
            write_json(Path(args.output_json), payload)
        return 0

    try:
        from moe_infinity.serving.engine import ContinuousBatchingEngine
    except Exception as exc:
        payload["reason"] = f"ContinuousBatchingEngine import failed: {exc}"
        print(f"BLOCKED: {payload['reason']}")
        if args.output_json:
            write_json(Path(args.output_json), payload)
        return 0

    engine = ContinuousBatchingEngine(
        model=model.model,
        engine=model.engine,
        config=_build_engine_config(model.model, kv_cache_ratio=kv_cache_ratio),
        tokenizer=tokenizer,
    )
    prompt_batches = _build_prompt_batches(
        tokenizer,
        num_requests=args.num_requests,
        prompt_length=args.prompt_length,
    )

    measurement = run_benchmark(
        engine,
        prompt_batches,
        max_new_tokens=args.max_new_tokens,
    )
    payload["status"] = "PASS"
    payload["reason"] = None
    payload["measurement"] = measurement

    print_table(
        measurement,
        model=args.model,
        kv_cache_ratio=kv_cache_ratio,
        kv_offload_enabled=bool(args.enable_kv_offload),
    )

    if args.output_json:
        write_json(Path(args.output_json), payload)
    return 0


if __name__ == "__main__":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    raise SystemExit(main())
