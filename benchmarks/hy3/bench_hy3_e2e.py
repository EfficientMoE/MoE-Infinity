#!/usr/bin/env python3
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Hy3-FP8 end-to-end TTFT + decode throughput benchmark.

Loads the real tencent/Hy3-FP8 checkpoint through the offloaded MoE
runtime (hy_v3 / glm_fp8_block128 dequant-on-copy path) and measures,
per prompt:

* TTFT: wall time of a cuda-synced ``generate(max_new_tokens=1)``
  (the v4flash_ab convention), after a short warmup generate.
* E2E: wall time of a full greedy ``generate(max_new_tokens=N)``.
* decode tok/s = (N - 1) / (e2e - ttft).
* expert cache hit rate plus ``get_expert_h2d_bytes_total`` deltas
  (per-token H2D bytes during the full generate when the native
  counter is available; counters are reset between phases).

Usage (first run builds the ~290 GB offload store, ~10-12 min):
    CUDA_VISIBLE_DEVICES=<gpu> \
    PYTHONPATH=/path/to/moe-store \
    python benchmarks/hy3/bench_hy3_e2e.py \
        --out benchmarks/hy3/results/hy3_e2e.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_CKPT = os.environ.get(
    "HY3_FP8_CKPT",
    "/mnt/raid0nvme0/public/huggingface/hub/"
    "models--tencent--Hy3-FP8/snapshots/"
    "ecc1d8e194e093f33177f2f0ef7ce8f397b2d68b",
)
DEFAULT_OFFLOAD = "/mnt/raid0nvme0/leyang/hy3-offload-bench"

PROMPTS = [
    "The capital of France is a city known for its museums, cafes, and",
    "In distributed systems, the CAP theorem states that a system can",
    "Write a short haiku about graphics processors working late at night:",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--offload-dir", default=DEFAULT_OFFLOAD)
    parser.add_argument("--device-memory-ratio", type=float, default=0.75)
    parser.add_argument("--prompts", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--out", default="results.json")
    return parser.parse_args()


def _archer(model: Any) -> Any:
    """Native ArcherPrefetchHandle (see bench_store_parity.py)."""
    eng = getattr(model, "engine", model)
    for attr in ("archer_engine", "engine"):
        cand = getattr(eng, attr, None)
        if cand is not None and hasattr(cand, "get_hit_rate"):
            return cand
    return getattr(eng, "archer_engine", None)


def _call(obj: Any, name: str) -> Any:
    fn = getattr(obj, name, None)
    if callable(fn):
        try:
            return fn()
        except Exception:
            return None
    return None


def _scalar(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    return value


def _timed_generate(model: Any, input_ids: Any, max_new_tokens: int) -> float:
    import torch

    torch.cuda.synchronize()
    start = time.perf_counter()
    model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def _p50(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def main() -> int:
    args = parse_args()

    import torch

    if not torch.cuda.is_available():
        print("BLOCKED: CUDA is not available")
        return 1

    from transformers import AutoTokenizer

    from moe_infinity import MoE

    env = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ckpt": args.ckpt,
        "offload_dir": args.offload_dir,
        "device_memory_ratio": args.device_memory_ratio,
        "max_new_tokens": args.max_new_tokens,
        "warmup_tokens": args.warmup_tokens,
    }
    print(json.dumps(env, indent=2))

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    load_start = time.perf_counter()
    model = MoE(
        args.ckpt,
        {
            "offload_path": args.offload_dir,
            "device_memory_ratio": args.device_memory_ratio,
        },
    )
    load_s = time.perf_counter() - load_start
    print(f"model load (incl. store build if cold): {load_s:.1f} s")

    archer = _archer(model)
    has_reset = archer is not None and callable(
        getattr(archer, "reset_expert_transfer_stats", None)
    )

    # Warmup (outside measurement, per docs/benchmarking.md).
    warm_ids = tokenizer(PROMPTS[0], return_tensors="pt").input_ids.to("cuda:0")
    warm_s = _timed_generate(model, warm_ids, args.warmup_tokens)
    print(f"warmup generate ({args.warmup_tokens} tokens): {warm_s:.1f} s")

    n = args.max_new_tokens
    per_prompt: list[dict[str, Any]] = []
    for i in range(args.prompts):
        prompt = PROMPTS[i % len(PROMPTS)]
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
            "cuda:0"
        )
        prompt_tokens = int(input_ids.shape[-1])

        if has_reset:
            _call(archer, "reset_expert_transfer_stats")
        ttft = _timed_generate(model, input_ids, 1)
        ttft_h2d = (
            _scalar(_call(archer, "get_expert_h2d_bytes_total"))
            if has_reset
            else None
        )

        if has_reset:
            _call(archer, "reset_expert_transfer_stats")
        e2e = _timed_generate(model, input_ids, n)
        e2e_h2d = (
            _scalar(_call(archer, "get_expert_h2d_bytes_total"))
            if has_reset
            else None
        )

        decode_window = max(1e-6, e2e - ttft)
        decode_tok_s = (n - 1) / decode_window
        hit_rate = _scalar(getattr(model.engine, "expert_cache_hit_rate", None))
        row = {
            "prompt_tokens": prompt_tokens,
            "ttft_s": ttft,
            "e2e_s": e2e,
            "decode_tok_s": decode_tok_s,
            "ttft_h2d_bytes": ttft_h2d,
            "e2e_h2d_bytes": e2e_h2d,
            "e2e_h2d_bytes_per_token": (
                e2e_h2d / n if isinstance(e2e_h2d, (int, float)) else None
            ),
            "expert_cache_hit_rate": hit_rate,
        }
        per_prompt.append(row)
        print(
            f"prompt {i + 1}/{args.prompts}: tokens={prompt_tokens} "
            f"ttft={ttft:.2f}s e2e={e2e:.2f}s "
            f"decode={decode_tok_s:.3f} tok/s hit_rate={hit_rate}"
        )

    summary = {
        "ttft_p50_s": _p50([r["ttft_s"] for r in per_prompt]),
        "e2e_p50_s": _p50([r["e2e_s"] for r in per_prompt]),
        "decode_tok_s_p50": _p50([r["decode_tok_s"] for r in per_prompt]),
        "h2d_bytes_per_token_p50": _p50(
            [
                r["e2e_h2d_bytes_per_token"]
                for r in per_prompt
                if r["e2e_h2d_bytes_per_token"] is not None
            ]
        ),
        "hit_rate_last": per_prompt[-1]["expert_cache_hit_rate"]
        if per_prompt
        else None,
    }

    print("\n| Metric | p50 |")
    print("| --- | ---: |")
    print(f"| TTFT (s) | {summary['ttft_p50_s']:.2f} |")
    print(f"| E2E {n} tokens (s) | {summary['e2e_p50_s']:.2f} |")
    print(f"| Decode tok/s | {summary['decode_tok_s_p50']:.3f} |")
    hbpt = summary["h2d_bytes_per_token_p50"]
    if hbpt is not None:
        print(f"| Expert H2D MB/token | {hbpt / 2**20:.1f} |")
    print(f"| Expert cache hit rate | {summary['hit_rate_last']} |")

    payload = {
        "environment": env,
        "model_load_s": load_s,
        "warmup_s": warm_s,
        "per_prompt": per_prompt,
        "summary": summary,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
