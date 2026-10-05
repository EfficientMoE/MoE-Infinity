"""Teacher-forced (drift-free) fidelity for a drop policy.

Feeds the off baseline's greedy tokens back as forced decode context, so every
position is scored on the SAME context as the baseline. This isolates per-step
policy quality from the autoregressive drift that collapses free-running greedy
agreement (where one early flip desynchronizes the rest). Reports, over the
frozen 24x128 set:
  - tf_agreement: fraction of positions where the policy's greedy argmax equals
    the off token (given off context),
  - tf_nll_per_tok: mean negative log-prob the policy assigns to the off token
    (surprisal of the baseline continuation under the policy).
The off policy teacher-forced on its own tokens is the control (agreement ~1.0).
"""

from __future__ import annotations

import argparse
import json
import os

try:
    from benchmarks.expert_drop.trace_capture import PROMPTS
except ImportError:
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from trace_capture import PROMPTS


class _Forcer:
    def __init__(self):
        self.it = iter(())
        self.records = []

    def start(self, tokens):
        self.it = iter(tokens)
        self.records = []


FORCER = _Forcer()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--offload-dir", required=True)
    ap.add_argument("--ref", required=True, help="off-run json with gen_tokens")
    ap.add_argument("--policy", choices=["off", "on_miss"], default="on_miss")
    ap.add_argument("--mass-budget", type=float, default=0.20)
    ap.add_argument("--head-budget", type=float, default=0.0)
    ap.add_argument("--adaptive-slope", type=float, default=0.0)
    ap.add_argument("--adaptive-miss0", type=int, default=0)
    ap.add_argument("--device-memory-ratio", type=float, default=0.05)
    ap.add_argument("--num-prompts", type=int, default=24)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    os.environ.setdefault("MOE_FORCE_GPU_ONLY_ROUTING", "1")
    if args.policy == "on_miss":
        os.environ["MOE_EXPERT_DROP_FUSED"] = "1"

    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from moe_infinity import MoE

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    config = {
        "offload_path": args.offload_dir,
        "device_memory_ratio": args.device_memory_ratio,
        "expert_drop_policy": args.policy,
        "gpu_only_expert_routing": True,
    }
    if args.policy == "on_miss":
        config["expert_drop_mass_budget"] = args.mass_budget
        config["expert_drop_head_budget"] = args.head_budget
        config["expert_drop_adaptive_slope"] = args.adaptive_slope
        config["expert_drop_adaptive_miss0"] = args.adaptive_miss0

    ref = json.load(open(args.ref))["gen_tokens"]
    model = MoE(args.checkpoint, config)

    from moe_infinity.engine.generation_loop import GenerationEngine

    def tf_sample(self, logits, params):
        final = logits[-1]
        pred = int(final.argmax().item())
        try:
            forced = int(next(FORCER.it))
        except StopIteration:
            forced = pred
        logp = float(F.log_softmax(final.float(), dim=-1)[forced].item())
        FORCER.records.append((1 if pred == forced else 0, logp))
        return forced

    GenerationEngine._sample = tf_sample

    per_prompt = []
    tot_match = 0
    tot_n = 0
    tot_nll = 0.0
    for p in range(min(args.num_prompts, len(ref))):
        toks = ref[p]
        if not toks:
            continue
        FORCER.start(toks)
        input_ids = tokenizer(PROMPTS[p], return_tensors="pt").input_ids.to(
            "cuda:0"
        )
        model.generate(
            input_ids,
            max_new_tokens=len(toks),
            min_new_tokens=len(toks),
            do_sample=False,
        )
        recs = FORCER.records[: len(toks)]
        m = sum(r[0] for r in recs)
        n = len(recs)
        nll = -sum(r[1] for r in recs)
        per_prompt.append(
            {
                "agree": round(m / max(n, 1), 4),
                "nll": round(nll / max(n, 1), 4),
                "n": n,
            }
        )
        tot_match += m
        tot_n += n
        tot_nll += nll

    res = {
        "checkpoint": args.checkpoint,
        "policy": args.policy,
        "mass_budget": args.mass_budget,
        "head_budget": args.head_budget,
        "adaptive_slope": args.adaptive_slope,
        "positions": tot_n,
        "tf_agreement": round(tot_match / max(tot_n, 1), 4),
        "tf_nll_per_tok": round(tot_nll / max(tot_n, 1), 4),
        "per_prompt": per_prompt,
    }
    with open(args.out_json, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "per_prompt"}))


if __name__ == "__main__":
    main()
