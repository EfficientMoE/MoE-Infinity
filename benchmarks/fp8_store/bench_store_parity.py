#!/usr/bin/env python3
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Paired-store parity harness for synthetic FP8 expert stores (#234 Phase 1).

Runs the same offloaded model on a bf16 store (--store-a) and an FP8 store
(--store-b) and asserts the Phase 1 release gates in-script:

  * expert transfer bytes (FP8) <= 55% of bf16  [hard gate]
  * teacher-forced mean-NLL delta <= 1% relative [hard gate]
  * resident-expert count at fixed budget >= 1.8x bf16 [hard gate]
  * TTFT / TPOT percentiles and expert-wait time [characterization only]

NLL reuses ``benchmarks/eval/perplexity.py`` (``load_texts`` /
``evaluate_perplexity``); transfer bytes reuse the native
``get_expert_h2d_bytes_total`` / ``reset_expert_transfer_stats`` accessors.
Results are written as JSON next to each store.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.eval.perplexity import (  # noqa: E402
    evaluate_perplexity,
    load_texts,
)

BYTE_GATE = 0.55
NLL_GATE = 0.01
RESIDENT_GATE = 1.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--store-a", required=True, help="bf16 baseline store")
    parser.add_argument("--store-b", required=True, help="FP8 store")
    parser.add_argument(
        "--prompts",
        default="wikitext",
        help="dataset name for the fixed prompt set (see benchmarks/eval)",
    )
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--gen-tokens", type=int, default=32)
    parser.add_argument("--device-memory-ratio", type=float, default=0.35)
    parser.add_argument(
        "--skip-gates",
        action="store_true",
        help="report metrics without asserting release gates",
    )
    return parser.parse_args()


def _engine(model):
    return getattr(model, "engine", model)


def _archer(model):
    eng = _engine(model)
    for attr in ("archer_engine", "engine"):
        cand = getattr(eng, attr, None)
        if cand is not None and hasattr(cand, "get_hit_rate"):
            return cand
    return getattr(eng, "archer_engine", None)


def _call(obj, name):
    fn = getattr(obj, name, None)
    if callable(fn):
        try:
            return fn()
        except Exception:
            return None
    return None


def _percentiles(values):
    if not values:
        return {}
    values = sorted(values)

    def pct(p):
        idx = min(len(values) - 1, max(0, int(round(p * (len(values) - 1)))))
        return values[idx]

    return {"p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99)}


def run_store(checkpoint, store_dir, args):
    import torch
    from transformers import AutoTokenizer

    from moe_infinity import MoE

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = MoE(
        checkpoint,
        {
            "offload_path": store_dir,
            "device_memory_ratio": args.device_memory_ratio,
        },
    )
    archer = _archer(model)
    _call(archer, "reset_expert_transfer_stats")

    texts = load_texts(args.prompts, "test", args.num_prompts)

    ttft, tpot = [], []
    for text in texts[: max(4, args.num_prompts // 4)]:
        input_ids = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        ).input_ids.to("cuda:0")
        start = time.perf_counter()
        out = model.generate(
            input_ids,
            max_new_tokens=args.gen_tokens,
            do_sample=False,
        )
        elapsed = time.perf_counter() - start
        new_tokens = int(out.shape[-1] - input_ids.shape[-1])
        if new_tokens > 0:
            tpot.append(elapsed / new_tokens)
        ttft.append(elapsed / max(1, new_tokens))

    ppl, nll, n = evaluate_perplexity(
        model, tokenizer, texts, args.seq_length, args.num_prompts, 1
    )

    result = {
        "store": str(store_dir),
        "exit_status": 0,
        "nll": nll,
        "perplexity": ppl,
        "samples": n,
        "expert_h2d_bytes_total": _call(archer, "get_expert_h2d_bytes_total"),
        "resident_expert_count": _call(archer, "get_resident_expert_count")
        or _call(archer, "get_expert_cache_size"),
        "cache_hit_rate": _call(archer, "get_hit_rate"),
        "wasted_prefetch_bytes": _call(archer, "get_wasted_prefetch_bytes"),
        "ttft_s": _percentiles(ttft),
        "tpot_s": _percentiles(tpot),
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    results = {}
    for label, store in (("bf16", args.store_a), ("fp8", args.store_b)):
        print(f"[bench_store_parity] running {label} store: {store}")
        results[label] = run_store(args.checkpoint, store, args)
        out_path = Path(store) / "bench_store_parity.json"
        out_path.write_text(json.dumps(results[label], indent=2) + "\n")
        print(f"[bench_store_parity] wrote {out_path}")

    a, b = results["bf16"], results["fp8"]
    summary = {"bf16": a, "fp8": b, "gates": {}}

    bytes_a, bytes_b = a["expert_h2d_bytes_total"], b["expert_h2d_bytes_total"]
    byte_ratio = (bytes_b / bytes_a) if bytes_a and bytes_b else None
    nll_delta = (b["nll"] - a["nll"]) / a["nll"]
    res_a, res_b = a["resident_expert_count"], b["resident_expert_count"]
    resident_ratio = (res_b / res_a) if res_a and res_b else None

    summary["gates"] = {
        "byte_ratio": byte_ratio,
        "byte_gate": BYTE_GATE,
        "nll_delta_rel": nll_delta,
        "nll_gate": NLL_GATE,
        "resident_ratio": resident_ratio,
        "resident_gate": RESIDENT_GATE,
    }
    out_path = Path(args.store_b) / "bench_store_parity_summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["gates"], indent=2))

    if args.skip_gates:
        return
    failures = []
    if byte_ratio is None:
        failures.append("expert_h2d_bytes_total accessor unavailable")
    elif byte_ratio > BYTE_GATE:
        failures.append(f"byte ratio {byte_ratio:.3f} > {BYTE_GATE}")
    if nll_delta > NLL_GATE:
        failures.append(f"NLL delta {nll_delta:.4f} > {NLL_GATE}")
    if resident_ratio is None:
        failures.append("resident-expert accessor unavailable")
    elif resident_ratio < RESIDENT_GATE:
        failures.append(
            f"resident ratio {resident_ratio:.2f} < {RESIDENT_GATE}"
        )
    if failures:
        print("[bench_store_parity] GATE FAILURES:", "; ".join(failures))
        raise SystemExit(1)
    print("[bench_store_parity] all gates passed")


if __name__ == "__main__":
    main()
