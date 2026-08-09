"""Native DFlash speculative decoding for GPT-OSS, greedy batch-1 path.

This script keeps one end-to-end flow in view: a greedy,
batch-1 ``model.generate(..., speculative_draft=...)`` call routes through the
native DFlash draft, verify, rollback loop in
``moe_infinity/spec_decode/dflash.py``. The drafter loads via
``trust_remote_code=True`` and reuses the target's ``embed_tokens`` and
``lm_head``. Omit ``speculative_draft`` or switch to a non-greedy config, and
``generate`` stays on the standard autoregressive path.

``MoE.generate()`` is the current in-process synchronous path but emits
``DeprecationWarning`` and is scheduled for removal. ``MoE.serve()`` is the
recommended continuous-batching HTTP path, not a drop-in tensor-return API.
Because ``trust_remote_code=True`` executes checkpoint repository code, use
only a trusted and pinned drafter revision.

The direct speculator API also supports batch-1 sampled decoding, and the
batch>1, serving, and route-ahead variants are documented in
``docs/dflash.md``.

Validation is covered by the CPU tiny losslessness gate
(``tests/python/dflash/test_native_e2e.py``) plus the GPU-gated GPT-OSS-20B,
GPT-OSS-120B, and serving harnesses under ``tests/python/dflash/``.
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="openai/gpt-oss-120b")
    parser.add_argument(
        "--draft_model_path", default="z-lab/gpt-oss-120b-DFlash"
    )
    parser.add_argument("--offload_dir", required=True)
    parser.add_argument("--device_memory_ratio", type=float, default=0.75)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--prompt",
        default="Question: What is the capital of France?\nAnswer:",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from moe_infinity import MoE
    from moe_infinity.spec_decode import DFlashSpeculator

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = MoE(
        args.model_name_or_path,
        {
            "offload_path": args.offload_dir,
            "device_memory_ratio": args.device_memory_ratio,
        },
    )
    speculator = DFlashSpeculator(model, args.draft_model_path)
    print(
        f"DFlash drafter ready: block_size={speculator.config.block_size}, "
        f"mask_token_id={speculator.config.mask_token_id}, "
        f"target_layer_ids={speculator.config.target_layer_ids}"
    )

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids

    # Engine path: greedy batch-1 generate with a drafter routes through the
    # native DFlash loop.
    start = time.time()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        speculative_draft=speculator,
    )
    elapsed = time.time() - start

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    new_tokens = output_ids.shape[1] - input_ids.shape[1]
    print(f"[dflash] {new_tokens} tokens in {elapsed:.1f}s")
    print(text)


if __name__ == "__main__":
    main()
