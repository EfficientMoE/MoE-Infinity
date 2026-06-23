#!/usr/bin/env python3
"""MXFP4 vs BF16-dequant benchmark for GPT-OSS models.

Compares two offloading strategies for MXFP4-native checkpoints:
  MXFP4 fused:  keep packed uint8 weights, fused dequant+GEMM Triton kernel
  BF16 dequant: dequantize at load time, standard F.linear (4x more I/O)

Usage inside Docker:
  # MXFP4 fused (new)
  python benchmarks/mxfp4_benchmark.py --model openai/gpt-oss-20b --mode mxfp4

  # BF16 dequant (baseline)
  python benchmarks/mxfp4_benchmark.py --model openai/gpt-oss-20b --mode bf16
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev

import torch

MEGABYTE = 1024 * 1024

PROMPTS = [
    "What is the capital of Japan?",
    "Explain quantum mechanics in one sentence.",
    "If a train travels 120 km in 2 hours, what is its average speed?",
    "Describe one practical strategy to reduce memory usage during inference.",
    "Summarize the difference between supervised and unsupervised learning.",
]


@dataclass
class BenchResult:
    mode: str
    model: str
    num_requests: int
    warmup_requests: int
    max_new_tokens: int
    avg_ttft_ms: float | None
    avg_per_token_ms: float | None
    std_per_token_ms: float | None
    peak_gpu_mb: float
    expert_weight_mb: float | None
    gpu_name: str
    total_time_s: float


class StopWatch:
    def __init__(self):
        self.start_time: float | None = None
        self.first_token_time: float | None = None
        self.end_time: float | None = None
        self.token_count = 0
        self._seen_prompt = False

    def put(self, value):
        if not self._seen_prompt:
            self._seen_prompt = True
            return
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        self.token_count += 1

    def end(self):
        self.end_time = time.perf_counter()


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--mode", choices=["mxfp4", "bf16", "both"], default="both")
    p.add_argument("--offload-dir", default="/tmp/moe-offload")
    p.add_argument("--device-memory-ratio", type=float, default=0.75)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--num-requests", type=int, default=10)
    p.add_argument("--output-json", default="mxfp4_benchmark_results.json")
    p.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove offload data after each run",
    )
    return p.parse_args()


def get_gpu_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "Unknown"


def estimate_expert_weight_size(model_name: str, mode: str) -> float | None:
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        h = cfg.hidden_size
        inter = cfg.intermediate_size
        n_experts = cfg.num_local_experts
        if mode == "mxfp4":
            gate_up_bytes = h * (2 * inter) // 2
            down_bytes = inter * h // 2
            scale_gate_up = h * (2 * inter) // 32
            scale_down = inter * h // 32
            per_expert = gate_up_bytes + down_bytes + scale_gate_up + scale_down
        else:
            gate_up_bytes = h * (2 * inter) * 2
            down_bytes = inter * h * 2
            per_expert = gate_up_bytes + down_bytes
        total = per_expert * n_experts
        return total / MEGABYTE
    except Exception:
        return None


def run_single_mode(mode: str, args) -> BenchResult:
    offload_path = os.path.join(args.offload_dir, f"{mode}")
    os.makedirs(offload_path, exist_ok=True)

    if mode == "bf16":
        os.environ["MOE_INFINITY_MXFP4_DEQUANT"] = "1"
    else:
        os.environ.pop("MOE_INFINITY_MXFP4_DEQUANT", None)

    from importlib import reload

    import moe_infinity.runtime.model_offload

    reload(moe_infinity.runtime.model_offload)

    from transformers import AutoTokenizer

    from moe_infinity import MoE

    print(f"\n{'=' * 60}")
    print(f"  Mode: {mode.upper()}")
    print(f"  Model: {args.model}")
    print(f"  Offload: {offload_path}")
    print(f"{'=' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    t_load_start = time.perf_counter()
    model = MoE(
        args.model,
        {
            "offload_path": offload_path,
            "device_memory_ratio": args.device_memory_ratio,
        },
    )
    t_load = time.perf_counter() - t_load_start
    print(f"  Model loaded in {t_load:.1f}s")

    print(f"  Warming up ({args.warmup} requests)...")
    with torch.no_grad():
        for i in range(args.warmup):
            ids = tokenizer("Hello", return_tensors="pt").input_ids.to("cuda:0")
            _ = model.generate(
                ids,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    ttfts: list[float] = []
    per_token_latencies: list[float] = []
    peak_mems: list[float] = []

    print(f"  Running {args.num_requests} measured requests...")
    total_start = time.perf_counter()
    with torch.no_grad():
        for i in range(args.num_requests):
            prompt = PROMPTS[i % len(PROMPTS)]
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                "cuda:0"
            )

            torch.cuda.reset_peak_memory_stats()
            sw = StopWatch()
            sw.start_time = time.perf_counter()

            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                streamer=sw,
                pad_token_id=tokenizer.eos_token_id,
            )
            torch.cuda.synchronize()
            end_t = time.perf_counter()

            gen_tokens = max(output_ids.shape[-1] - input_ids.shape[-1], 1)
            total_req_ms = (end_t - sw.start_time) * 1000
            per_tok_ms = total_req_ms / max(gen_tokens, 1)

            if sw.first_token_time and sw.start_time:
                ttfts.append((sw.first_token_time - sw.start_time) * 1000)
                decode_time = end_t - sw.first_token_time
                per_tok_ms = decode_time * 1000 / max(gen_tokens - 1, 1)

            per_token_latencies.append(per_tok_ms)
            peak_mems.append(torch.cuda.max_memory_allocated() / MEGABYTE)

            ttft_str = f"{ttfts[-1]:.0f}" if ttfts else "N/A"
            print(
                f"    [{i + 1}/{args.num_requests}] tokens={gen_tokens} "
                f"ttft={ttft_str}ms per_tok={per_tok_ms:.0f}ms "
                f"peak_gpu={peak_mems[-1]:.0f}MB"
            )

    total_time = time.perf_counter() - total_start

    del model
    gc.collect()
    torch.cuda.empty_cache()

    if args.cleanup and os.path.isdir(offload_path):
        import shutil

        shutil.rmtree(offload_path, ignore_errors=True)

    result = BenchResult(
        mode=mode,
        model=args.model,
        num_requests=args.num_requests,
        warmup_requests=args.warmup,
        max_new_tokens=args.max_new_tokens,
        avg_ttft_ms=mean(ttfts) if ttfts else None,
        avg_per_token_ms=mean(per_token_latencies)
        if per_token_latencies
        else None,
        std_per_token_ms=stdev(per_token_latencies)
        if len(per_token_latencies) > 1
        else None,
        peak_gpu_mb=max(peak_mems) if peak_mems else 0,
        expert_weight_mb=estimate_expert_weight_size(args.model, mode),
        gpu_name=get_gpu_name(),
        total_time_s=total_time,
    )

    print(f"\n  Results ({mode.upper()}):")
    print(
        f"    Avg TTFT:           {result.avg_ttft_ms:.1f} ms"
        if result.avg_ttft_ms
        else "    Avg TTFT:           N/A"
    )
    print(
        f"    Avg per-token:      {result.avg_per_token_ms:.1f} ms"
        if result.avg_per_token_ms
        else "    Avg per-token:      N/A"
    )
    print(f"    Peak GPU memory:    {result.peak_gpu_mb:.0f} MB")
    print(
        f"    Expert weight size: {result.expert_weight_mb:.0f} MB"
        if result.expert_weight_mb
        else "    Expert weight size: N/A"
    )
    return result


def print_comparison(mxfp4: BenchResult, bf16: BenchResult):
    print(f"\n{'=' * 60}")
    print(f"  COMPARISON: MXFP4 fused vs BF16 dequant")
    print(f"  GPU: {mxfp4.gpu_name}")
    print(f"{'=' * 60}")
    print(f"{'Metric':<30s} {'MXFP4':>12s} {'BF16':>12s} {'Speedup':>10s}")
    print(f"{'-' * 64}")

    if mxfp4.avg_ttft_ms and bf16.avg_ttft_ms:
        sp = bf16.avg_ttft_ms / mxfp4.avg_ttft_ms
        print(
            f"{'Avg TTFT (ms)':<30s} {mxfp4.avg_ttft_ms:>12.1f} {bf16.avg_ttft_ms:>12.1f} {sp:>9.2f}x"
        )

    if mxfp4.avg_per_token_ms and bf16.avg_per_token_ms:
        sp = bf16.avg_per_token_ms / mxfp4.avg_per_token_ms
        print(
            f"{'Avg per-token (ms)':<30s} {mxfp4.avg_per_token_ms:>12.1f} {bf16.avg_per_token_ms:>12.1f} {sp:>9.2f}x"
        )

    sp = bf16.peak_gpu_mb / max(mxfp4.peak_gpu_mb, 1)
    print(
        f"{'Peak GPU memory (MB)':<30s} {mxfp4.peak_gpu_mb:>12.0f} {bf16.peak_gpu_mb:>12.0f} {sp:>9.2f}x"
    )

    if mxfp4.expert_weight_mb and bf16.expert_weight_mb:
        sp = bf16.expert_weight_mb / max(mxfp4.expert_weight_mb, 1)
        print(
            f"{'Expert weight size (MB)':<30s} {mxfp4.expert_weight_mb:>12.0f} {bf16.expert_weight_mb:>12.0f} {sp:>9.2f}x"
        )


def main():
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device available.")
        return 1

    results = {}

    modes = ["mxfp4", "bf16"] if args.mode == "both" else [args.mode]
    for mode in modes:
        results[mode] = run_single_mode(mode, args)

    if "mxfp4" in results and "bf16" in results:
        print_comparison(results["mxfp4"], results["bf16"])

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: asdict(v) for k, v in results.items()}
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nResults saved to {output}")
    os._exit(0)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
