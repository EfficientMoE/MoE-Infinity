"""Phase 0 trace capture: one off-policy decode run per model.

Runs the frozen 24-prompt x 128-token greedy set through native GPU-only
routing with MOE_EXPERT_DROP_TRACE enabled, dumps the pre-drop
routing/residency trace, converts it to a compact .npz, and records the
measured decode-ITL percentiles for the same run (the off-policy calibration
anchor). With --policy on_miss it instead measures the fused drop arm, used as
the second calibration anchor.
"""

from __future__ import annotations

import argparse
import json
import os
import time

PROMPTS = [
    "Explain why the sky is blue in simple terms.",
    "Summarize the plot of Romeo and Juliet.",
    "What are the main causes of the French Revolution?",
    "Describe how photosynthesis works.",
    "Write a short story about a lost dog finding its way home.",
    "What is the difference between speed and velocity?",
    "Explain the water cycle step by step.",
    "How does a refrigerator keep food cold?",
    "Describe the rules of chess for a beginner.",
    "What happened during the Apollo 11 mission?",
    "Explain supply and demand with an example.",
    "How do vaccines protect the human body?",
    "Describe the life cycle of a butterfly.",
    "What makes the Roman Empire historically important?",
    "Explain how tides are caused by the moon.",
    "What is the role of DNA in living organisms?",
    "Describe how earthquakes happen.",
    "Explain the difference between weather and climate.",
    "How does the human heart pump blood?",
    "What are black holes and how do they form?",
    "Explain how bees make honey.",
    "Describe the invention of the printing press.",
    "What causes rainbows to appear?",
    "Explain how sound travels through air.",
]


def _find_dispatcher(model):
    engine = getattr(model, "engine", None) or getattr(model, "_engine", None)
    for holder in (model, engine, getattr(model, "archer_engine", None)):
        executor = getattr(holder, "expert_executor", None)
        if executor is not None:
            dispatcher = getattr(executor, "expert_dispatcher", None)
            if dispatcher is not None:
                return dispatcher
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--offload-dir", required=True)
    parser.add_argument("--policy", choices=["off", "on_miss"], default="off")
    parser.add_argument("--mass-budget", type=float, default=0.05)
    parser.add_argument("--head-budget", type=float, default=0.0)
    parser.add_argument("--adaptive-slope", type=float, default=0.0)
    parser.add_argument("--adaptive-miss0", type=int, default=0)
    parser.add_argument("--device-memory-ratio", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=24)
    parser.add_argument("--trace-out", default="")
    parser.add_argument("--npz-out", default="")
    parser.add_argument("--latency-out", required=True)
    args = parser.parse_args()

    if args.trace_out:
        os.environ["MOE_EXPERT_DROP_TRACE"] = args.trace_out
    os.environ.setdefault("MOE_FORCE_GPU_ONLY_ROUTING", "1")
    if args.policy == "on_miss":
        os.environ["MOE_EXPERT_DROP_FUSED"] = "1"

    import torch
    from transformers import AutoTokenizer

    from moe_infinity import MoE

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    config = {
        "offload_path": args.offload_dir,
        "device_memory_ratio": args.device_memory_ratio,
        "expert_drop_policy": args.policy,
    }
    if args.policy == "on_miss":
        config["expert_drop_mass_budget"] = args.mass_budget
        config["expert_drop_head_budget"] = args.head_budget
        config["expert_drop_adaptive_slope"] = args.adaptive_slope
        config["expert_drop_adaptive_miss0"] = args.adaptive_miss0
        config["gpu_only_expert_routing"] = True
    if os.environ.get("MOE_FORCE_GPU_ONLY_ROUTING", "0") == "1":
        config["gpu_only_expert_routing"] = True

    model = MoE(args.checkpoint, config)

    token_times = []
    from moe_infinity.engine.generation_loop import GenerationEngine

    original = GenerationEngine._sample

    def timed_sample(self, logits, params):
        token = original(self, logits, params)
        token_times.append(time.perf_counter())
        return token

    GenerationEngine._sample = timed_sample

    itl_ms = []
    gen_tokens = []
    for prompt in PROMPTS[: args.num_prompts]:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
        token_times.clear()
        out = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        ts = list(token_times)
        itl_ms.extend((b - a) * 1e3 for a, b in zip(ts[1:], ts[2:]))
        gen_tokens.append([int(t) for t in out[0][-args.max_new_tokens :].tolist()])

    dispatcher = _find_dispatcher(model)
    fused_drop_stats = None
    if dispatcher is not None and hasattr(dispatcher, "get_fused_drop_stats"):
        fused_drop_stats = dispatcher.get_fused_drop_stats()
    if args.trace_out and dispatcher is not None and hasattr(
        dispatcher, "dump_expert_drop_trace"
    ):
        dispatcher.dump_expert_drop_trace(args.trace_out)

    t = torch.tensor(itl_ms, dtype=torch.float64)
    latency = {
        "checkpoint": args.checkpoint,
        "policy": args.policy,
        "mass_budget": args.mass_budget,
        "device_memory_ratio": args.device_memory_ratio,
        "samples": len(itl_ms),
        "mean_ms": float(t.mean()),
        "p50_ms": float(t.quantile(0.5)),
        "p90_ms": float(t.quantile(0.9)),
        "p99_ms": float(t.quantile(0.99)),
        "head_budget": args.head_budget,
        "adaptive_slope": args.adaptive_slope,
        "adaptive_miss0": args.adaptive_miss0,
        "fused_drop_stats": fused_drop_stats,
        "gen_tokens": gen_tokens,
    }
    with open(args.latency_out, "w") as fh:
        json.dump(latency, fh, indent=2)
    print(
        json.dumps(
            {k: v for k, v in latency.items()
             if k not in ("fused_drop_stats", "gen_tokens")}
        )
    )

    if args.trace_out and args.npz_out:
        try:
            from benchmarks.expert_drop.trace_parser import parse_trace_to_npz
        except ImportError:
            import sys

            sys.path.insert(0, os.path.dirname(__file__))
            from trace_parser import parse_trace_to_npz

        info = parse_trace_to_npz(args.trace_out, args.npz_out)
        print(
            f"trace npz {args.npz_out}: kept {int(info['records_kept'])}/"
            f"{int(info['records_total'])} decode records, "
            f"ne={int(info['num_experts'])} nl={int(info['num_layers'])}"
        )


if __name__ == "__main__":
    main()
