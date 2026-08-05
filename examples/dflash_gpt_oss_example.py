"""Native DFlash speculative decoding for gpt-oss-120b (v1: greedy, resident).

Drives the **native** DFlash draft->verify->rollback loop through the
MoE-Infinity engine: a greedy, batch-1 ``model.generate(..., speculative_draft=...)``
routes through ``GenerationEngine.spec_strategy`` (the native loop in
``moe_infinity/spec_decode/dflash.py``). The drafter (``z-lab/gpt-oss-120b-DFlash``)
loads via ``trust_remote_code=True`` and reuses the target's ``embed_tokens`` +
``lm_head``. Omit ``speculative_draft`` (or use a non-greedy config / batch>1) and
``generate`` uses the standard autoregressive path, byte-identical to before.

v1 scope (everything else is deferred):
  * Greedy only (``do_sample=False``); sampled speculative decoding is deferred.
  * Resident by default. Expert offload is a tunable knob (``device_memory_ratio``);
    v1 does not couple expert prefetch to the speculative loop.
  * Sync path only; the async serving path is not spec-enabled.
  * batch == 1.

Hardware: gpt-oss-120b is validated resident with TP=2 on Blackwell/SM120
(RTX PRO 6000). Correctness is proven autonomously on a tiny CPU model
(``tests/python/dflash/test_native_e2e.py``: native == plain greedy,
token-identical); the 120B agreement-rate / acceptance-length / tok-s harness is
GPU-gated (``tests/python/dflash/test_gpu_120b.py``, enabled via ``MOE_DFLASH_GPU=1``
with the checkpoints cached). See ``docs/dflash.md``.

Run:
    python examples/dflash_gpt_oss_example.py --offload_dir /ssd/moe-infinity/gpt-oss-120b
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="openai/gpt-oss-120b")
    parser.add_argument("--draft_model_path", default="z-lab/gpt-oss-120b-DFlash")
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

    # Engine path: greedy batch-1 generate with a drafter routes through
    # GenerationEngine.spec_strategy (the native DFlash loop).
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
