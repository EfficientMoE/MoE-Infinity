"""Real-checkpoint smoke for GLM-5.3-Flash (glm5_next family).

GPU-gated rollout evidence for docs/model-compatibility.md promotion: loads
the checkpoint through the drop-in MoE class (routed FP8 experts offloaded;
KDA/DSA/mHC/vision/MTP resident), renders a text-only prompt through the
tokenizer chat template, and asserts a non-empty greedy completion. Paste the
output into the PR under "Local validation evidence".
"""

import argparse
import os
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="zai-org/GLM-5.3-Flash")
    parser.add_argument("--offload-dir", required=True)
    parser.add_argument("--device-memory-ratio", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--prompt",
        default="Explain in one sentence what a mixture-of-experts model is.",
    )
    args = parser.parse_args()

    os.environ.setdefault(
        "HF_HUB_CACHE", "/mnt/raid0nvme0/public/huggingface/hub"
    )

    import torch
    from transformers import AutoTokenizer

    from moe_infinity import MoE

    tok = AutoTokenizer.from_pretrained(args.model)
    model = MoE(
        args.model,
        {
            "offload_path": args.offload_dir,
            "device_memory_ratio": args.device_memory_ratio,
        },
    )

    messages = [{"role": "user", "content": args.prompt}]
    ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda:0")

    start = time.monotonic()
    out = model.generate(ids, max_new_tokens=args.max_new_tokens)
    elapsed = time.monotonic() - start

    completion = tok.decode(
        out[0][ids.shape[1] :], skip_special_tokens=True
    ).strip()

    assert out.shape[1] >= ids.shape[1] + 1, "no tokens generated"
    assert len(completion) > 0, "empty completion"

    resident = torch.cuda.memory_allocated() / (1 << 30)
    peak = torch.cuda.max_memory_allocated() / (1 << 30)
    print(f"model: {args.model}")
    print(f"prompt tokens: {ids.shape[1]}")
    print(f"generated tokens: {out.shape[1] - ids.shape[1]} in {elapsed:.1f}s")
    print(f"resident GPU memory: {resident:.1f} GiB (peak {peak:.1f} GiB)")
    print(f"completion: {completion}")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
