"""Single-arm OLMoE decode-latency run for the expert-drop A/B gate.

Runs one policy arm per process (fresh runtime state), streams greedy
decode, and records inter-token latencies via streamer callbacks. The
first token of each prompt (prefill) is excluded. Drive it once per arm
and compare p99 with benchmarks/expert_drop/gates.py.
"""

import argparse
import json
import time

import torch
from transformers import AutoTokenizer

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


class _ITLStreamer:
    def __init__(self):
        self.timestamps = []

    def put(self, value):
        self.timestamps.append(time.perf_counter())

    def end(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default="allenai/OLMoE-1B-7B-0924-Instruct"
    )
    parser.add_argument("--offload-dir", required=True)
    parser.add_argument("--policy", choices=["off", "on_miss"], required=True)
    parser.add_argument("--mass-budget", type=float, default=0.05)
    parser.add_argument("--device-memory-ratio", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=24)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from moe_infinity import MoE

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    config = {
        "offload_path": args.offload_dir,
        "device_memory_ratio": args.device_memory_ratio,
        "expert_drop_policy": args.policy,
    }
    if args.policy == "on_miss":
        config["expert_drop_mass_budget"] = args.mass_budget
    model = MoE(args.checkpoint, config)

    itl_ms = []
    outputs = []
    for prompt in PROMPTS[: args.num_prompts]:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
            "cuda:0"
        )
        streamer = _ITLStreamer()
        with torch.inference_mode():
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.max_new_tokens,
                do_sample=False,
                streamer=streamer,
            )
        ts = streamer.timestamps
        itl_ms.extend((b - a) * 1e3 for a, b in zip(ts[1:], ts[2:]))
        outputs.append(tokenizer.decode(out[0][-16:], skip_special_tokens=True))

    t = torch.tensor(itl_ms, dtype=torch.float64)
    result = {
        "policy": args.policy,
        "mass_budget": args.mass_budget,
        "samples": len(itl_ms),
        "mean_ms": float(t.mean()),
        "p50_ms": float(t.quantile(0.5)),
        "p90_ms": float(t.quantile(0.9)),
        "p99_ms": float(t.quantile(0.99)),
        "tail_outputs": outputs[:3],
    }
    with open(args.output, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "tail_outputs"}))


if __name__ == "__main__":
    main()
