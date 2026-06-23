#!/usr/bin/env python3

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


def ensure_dirs():
    os.makedirs(OFFLOAD_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def load_model():
    config = {
        "offload_path": OFFLOAD_DIR,
        "device_memory_ratio": 0.75,
    }
    model = MoE(CHECKPOINT, config)
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT, trust_remote_code=True
    )
    return model, tokenizer


def test_correctness(model, tokenizer, phase_name="phase1"):
    prompts = [
        "What is 2+2?",
        "The capital of France is",
        "def hello_world():",
    ]

    results = []
    all_pass = True

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
            "cuda:0"
        )

        with torch.no_grad():
            out1 = model.generate(input_ids, max_new_tokens=20, do_sample=False)
            out2 = model.generate(input_ids, max_new_tokens=20, do_sample=False)

        match = torch.equal(out1, out2)
        decoded1 = tokenizer.decode(out1[0], skip_special_tokens=True)
        decoded2 = tokenizer.decode(out2[0], skip_special_tokens=True)

        result = {
            "prompt": prompt,
            "idempotent": match,
            "output1_preview": decoded1[:100],
            "output2_preview": decoded2[:100],
            "output_length": out1.shape[-1],
        }
        results.append(result)
        if not match:
            all_pass = False
            print(f"  FAIL: '{prompt}' - outputs differ")
        else:
            print(f"  PASS: '{prompt}' - {out1.shape[-1]} tokens")

    report = {
        "status": "PASS" if all_pass else "FAIL",
        "phase": phase_name,
        "description": f"{phase_name} numerical correctness (idempotency)",
        "results": results,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }

    output_path = os.path.join(RESULTS_DIR, f"{phase_name}_correctness.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {output_path}")

    return all_pass


def test_benchmark(model, tokenizer, phase_name="phase1"):
    prompt = "Explain the concept of machine learning in simple terms."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

    print(f"  Warming up ({phase_name})...")
    for _ in range(3):
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=5, do_sample=False)

    num_tokens = 20
    iters = 5
    times = []

    print(f"  Measuring {iters} iterations × {num_tokens} tokens...")
    for i in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=num_tokens, do_sample=False
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        actual_tokens = output.shape[-1] - input_ids.shape[-1]
        times.append(elapsed)
        print(f"    iter {i + 1}: {elapsed:.3f}s ({actual_tokens} tokens)")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    per_token_ms = (avg_time / num_tokens) * 1000

    report = {
        "status": "MEASURED",
        "phase": phase_name,
        "description": f"{phase_name} latency benchmark",
        "avg_total_time_s": round(avg_time, 4),
        "min_total_time_s": round(min_time, 4),
        "max_total_time_s": round(max_time, 4),
        "per_token_latency_ms": round(per_token_ms, 2),
        "num_tokens": num_tokens,
        "num_iters": iters,
        "baseline_per_step_ms": 947.5,
        "baseline_bubble_ratio": 0.8715,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }

    output_path = os.path.join(RESULTS_DIR, f"{phase_name}_benchmark.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {output_path}")

    return report


def main():
    ensure_dirs()

    print("=" * 60)
    print("Expert I/O Overlap Optimization — GPU Verification")
    print("=" * 60)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Model: {CHECKPOINT}")
    print()

    print("Loading model...")
    model, tokenizer = load_model()
    print("Model loaded.\n")

    print("[Phase 1+2+3 Correctness Gate]")
    correct = test_correctness(model, tokenizer, "combined_all_phases")
    print(f"  Result: {'PASS' if correct else 'FAIL'}\n")

    print("[Phase 1+2+3 Benchmark]")
    bench = test_benchmark(model, tokenizer, "combined_all_phases")
    print(f"  Avg per-token latency: {bench['per_token_latency_ms']:.2f} ms")
    print(f"  Baseline per-step: {bench['baseline_per_step_ms']} ms\n")

    print("=" * 60)
    print("SUMMARY")
    print(f"  Correctness: {'PASS' if correct else 'FAIL'}")
    print(f"  Per-token latency: {bench['per_token_latency_ms']:.2f} ms")
    print(
        f"  Total time (avg): {bench['avg_total_time_s']:.3f} s for {bench['num_tokens']} tokens"
    )
    print("=" * 60)

    return 0 if correct else 1


if __name__ == "__main__":
    sys.exit(main())
