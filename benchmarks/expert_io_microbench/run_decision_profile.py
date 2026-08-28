"""IBP feasibility decision-profile runner (Wave 1+ orchestrator).

Wraps a real `MoE.generate` workload inside `nsys profile -t cuda,nvtx
--capture-range=cudaProfilerApi`, then invokes `nsys_parser.summarise`
to emit a verdict against the FROZEN criterion in
.sisyphus/plans/ibp-feasibility-profile.md.

This script must be invoked with `nsys profile ... python run_decision_profile.py
<args>`; it expects to be running INSIDE the nsys-instrumented process. It
toggles cudaProfilerStart/Stop around the measured decode steps so warmup is
not counted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--offload-dir", required=True)
    p.add_argument("--hardware-tag", required=True)
    p.add_argument("--mode", choices=["disk", "host-only"], required=True)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--warmup-tokens", type=int, default=8)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--device-memory-ratio", type=float, default=0.5)
    p.add_argument("--speculative-prefetch", action="store_true")
    p.add_argument("--speculative-prefetch-overlap", action="store_true")
    p.add_argument(
        "--overlap-prefetch-policy",
        choices=["off", "observe", "enforce"],
        default="off",
    )
    p.add_argument("--overlap-prefetch-ewma-alpha", type=float, default=0.2)
    p.add_argument("--overlap-prefetch-safety-factor", type=float, default=0.8)
    p.add_argument("--overlap-prefetch-cold-start-experts", type=int, default=1)
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def sample_pcie_link() -> tuple[int, int]:
    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=pcie.link.width.current,pcie.link.gen.current",
                    "--format=csv,noheader,nounits",
                    "-i",
                    "0",
                ],
                text=True,
                timeout=10,
            )
            .strip()
            .splitlines()[0]
        )
        width_s, gen_s = out.split(",")
        return int(width_s.strip()), int(gen_s.strip())
    except Exception as e:
        print(f"[WARN] could not sample PCIe link: {e}", file=sys.stderr)
        return 16, 4


def main() -> int:
    args = parse_args()
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[runner] mode={args.mode} model={args.model}", flush=True)
    print(
        f"[runner] offload={args.offload_dir} ratio={args.device_memory_ratio}",
        flush=True,
    )

    import torch
    from transformers import AutoTokenizer

    from moe_infinity import MoE

    t0 = time.time()
    print(f"[{time.time() - t0:.1f}s] loading model", flush=True)
    m = MoE(
        args.model,
        {
            "offload_path": args.offload_dir,
            "device_memory_ratio": args.device_memory_ratio,
            "speculative_prefetch": args.speculative_prefetch,
            "speculative_prefetch_overlap": args.speculative_prefetch_overlap,
            "overlap_prefetch_policy": args.overlap_prefetch_policy,
            "overlap_prefetch_ewma_alpha": args.overlap_prefetch_ewma_alpha,
            "overlap_prefetch_safety_factor": (
                args.overlap_prefetch_safety_factor
            ),
            "overlap_prefetch_cold_start_experts": (
                args.overlap_prefetch_cold_start_experts
            ),
            "num_threads": args.num_threads,
        },
    )
    print(f"[{time.time() - t0:.1f}s] model loaded", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base_text = (
        "MoE-Infinity transfer timing benchmark prompt. "
        "Use deterministic content for stable measurements."
    )
    enc = tok.encode(base_text, add_special_tokens=False)
    while len(enc) < 128:
        enc = enc + enc
    enc = enc[:128]
    ids = torch.tensor([enc], dtype=torch.long, device="cuda")

    print(
        f"[{time.time() - t0:.1f}s] warmup {args.warmup_tokens} tokens",
        flush=True,
    )
    _ = m.generate(
        ids,
        max_new_tokens=max(args.warmup_tokens, 1),
        temperature=0.0,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    torch.cuda.synchronize()
    print(f"[{time.time() - t0:.1f}s] warmup done", flush=True)

    link_width, link_gen = sample_pcie_link()
    print(
        f"[runner] PCIe before profile: width={link_width} gen={link_gen}",
        flush=True,
    )

    print(f"[{time.time() - t0:.1f}s] profiler START", flush=True)
    torch.cuda.cudart().cudaProfilerStart()

    decode_step_times_ns: list[int] = []
    for it in range(args.iters):
        t_iter = time.perf_counter_ns()
        _ = m.generate(
            ids,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        torch.cuda.synchronize()
        decode_step_times_ns.append(time.perf_counter_ns() - t_iter)

    torch.cuda.cudart().cudaProfilerStop()
    print(f"[{time.time() - t0:.1f}s] profiler STOP", flush=True)

    link_width_after, link_gen_after = sample_pcie_link()
    print(
        f"[runner] PCIe after profile: width={link_width_after} gen={link_gen_after}",
        flush=True,
    )

    use_width = max(link_width, link_width_after)
    use_gen = max(link_gen, link_gen_after)

    overlap_stats = {}
    try:
        prefetcher = m.engine.expert_prefetcher
        stats_getter = getattr(prefetcher, "overlap_prefetch_stats", None)
        if callable(stats_getter):
            overlap_stats = stats_getter()
    except Exception:
        overlap_stats = {}

    out = {
        "model": args.model,
        "mode": args.mode,
        "hardware_tag": args.hardware_tag,
        "offload_dir": args.offload_dir,
        "max_new_tokens": args.max_new_tokens,
        "warmup_tokens": args.warmup_tokens,
        "iters": args.iters,
        "device_memory_ratio": args.device_memory_ratio,
        "speculative_prefetch": args.speculative_prefetch,
        "speculative_prefetch_overlap": args.speculative_prefetch_overlap,
        "overlap_prefetch_policy": args.overlap_prefetch_policy,
        "overlap_prefetch_stats": overlap_stats,
        "num_threads": args.num_threads,
        "decode_step_times_ns": decode_step_times_ns,
        "decode_step_total_ns": sum(decode_step_times_ns),
        "decode_step_count": args.iters * args.max_new_tokens,
        "pcie_link_width_observed": use_width,
        "pcie_link_gen_observed": use_gen,
        "pcie_link_width_pre": link_width,
        "pcie_link_gen_pre": link_gen,
        "pcie_link_width_post": link_width_after,
        "pcie_link_gen_post": link_gen_after,
    }
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"[runner] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
