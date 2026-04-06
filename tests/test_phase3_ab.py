#!/usr/bin/env python3
"""A/B benchmark: speculative_prefetch OFF vs ON."""

import json
import os
import sys
import time

import torch
from transformers import AutoTokenizer

from moe_infinity import MoE

CHECKPOINT = "deepseek-ai/DeepSeek-V2-Lite-Chat"
OFFLOAD_DIR = "/tmp/moe_test_offload"
RESULTS_DIR = "/workspace/MoE-Infinity/benchmarks/expert_io_microbench"
NUM_TOKENS = 20
WARMUP = 3
ITERS = 8


def bench(model, tokenizer, label):
    prompt = "Explain the concept of machine learning in simple terms."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

    for _ in range(WARMUP):
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=5, do_sample=False)

    times = []
    for i in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=NUM_TOKENS, do_sample=False
            )
        torch.cuda.synchronize()
        t = time.perf_counter() - t0
        times.append(t)
        actual = out.shape[-1] - input_ids.shape[-1]
        print(f"  [{label}] iter {i + 1}: {t:.3f}s ({actual} tok)")

    avg = sum(times) / len(times)
    per_tok = avg / NUM_TOKENS * 1000
    print(f"  [{label}] avg per-token: {per_tok:.2f} ms\n")
    return {
        "label": label,
        "avg_total_s": round(avg, 4),
        "per_token_ms": round(per_tok, 2),
        "min_s": round(min(times), 4),
        "max_s": round(max(times), 4),
        "iters": ITERS,
        "tokens": NUM_TOKENS,
    }


def main():
    os.makedirs(OFFLOAD_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Model: {CHECKPOINT}\n")

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT, trust_remote_code=True
    )

    # --- A: speculative_prefetch OFF ---
    print("=== Loading model: speculative_prefetch=False ===")
    model_off = MoE(
        CHECKPOINT,
        {
            "offload_path": OFFLOAD_DIR,
            "device_memory_ratio": 0.75,
            "speculative_prefetch": False,
        },
    )
    result_off = bench(model_off, tokenizer, "prefetch_OFF")
    del model_off
    torch.cuda.empty_cache()

    # --- B: speculative_prefetch ON ---
    print("=== Loading model: speculative_prefetch=True ===")
    model_on = MoE(
        CHECKPOINT,
        {
            "offload_path": OFFLOAD_DIR,
            "device_memory_ratio": 0.75,
            "speculative_prefetch": True,
        },
    )
    result_on = bench(model_on, tokenizer, "prefetch_ON")
    del model_on
    torch.cuda.empty_cache()

    # --- Summary ---
    delta_ms = result_off["per_token_ms"] - result_on["per_token_ms"]
    delta_pct = delta_ms / result_off["per_token_ms"] * 100

    summary = {
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "model": CHECKPOINT,
        "prefetch_off": result_off,
        "prefetch_on": result_on,
        "improvement_ms": round(delta_ms, 2),
        "improvement_pct": round(delta_pct, 2),
    }

    out_path = os.path.join(RESULTS_DIR, "phase3_ab_test.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print("PHASE 3 A/B RESULTS")
    print(f"  OFF: {result_off['per_token_ms']:.2f} ms/token")
    print(f"   ON: {result_on['per_token_ms']:.2f} ms/token")
    print(f"  Δ:   {delta_ms:+.2f} ms ({delta_pct:+.1f}%)")
    print(f"  Saved: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
