from __future__ import annotations

import argparse
import json
from typing import List


def token_agreement_rate(a: List[int], b: List[int]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    same = sum(1 for i in range(n) if a[i] == b[i])
    return same / n


def load_prompts(path: str) -> List[str]:
    with open(path) as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = data.get("prompts", [])
    return [str(p) for p in data]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="openai/gpt-oss-120b")
    parser.add_argument("--draft", default="z-lab/gpt-oss-120b-DFlash")
    parser.add_argument("--offload-dir", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device-memory-ratio", type=float, default=0.75)
    parser.add_argument("--out", default="dflash_agreement.json")
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer

    from moe_infinity import MoE
    from moe_infinity.spec_decode import DFlashSpeculator

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    moe = MoE(
        args.target,
        {
            "offload_path": args.offload_dir,
            "device_memory_ratio": args.device_memory_ratio,
        },
    )
    speculator = DFlashSpeculator(moe, args.draft)

    prompts = load_prompts(args.prompts)
    results = []
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        n_prompt = input_ids.shape[1]

        baseline = moe.generate(
            input_ids, max_new_tokens=args.max_new_tokens, do_sample=False
        )[0].tolist()
        with torch.inference_mode():
            speculative = speculator.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
            )[0].tolist()

        agreement = token_agreement_rate(
            baseline[n_prompt:], speculative[n_prompt:]
        )
        results.append(
            {
                "prompt": prompt,
                "agreement": agreement,
                "baseline_new": len(baseline) - n_prompt,
                "speculative_new": len(speculative) - n_prompt,
            }
        )
        print(f"agreement={agreement:.3f}  prompt={prompt[:60]!r}")

    mean_agreement = sum(r["agreement"] for r in results) / max(len(results), 1)
    print(f"MEAN_AGREEMENT {mean_agreement:.4f}")
    with open(args.out, "w") as handle:
        json.dump(
            {"mean_agreement": mean_agreement, "per_prompt": results},
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
