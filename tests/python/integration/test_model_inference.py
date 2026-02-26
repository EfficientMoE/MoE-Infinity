"""Functional inference test for MoE-Infinity.

Verifies that a model can receive a text question and produce non-empty text
output via MoE-Infinity without errors.

Usage::

    python tests/python/integration/test_model_inference.py \\
        --model deepseek-ai/DeepSeek-V2-Lite-Chat \\
        --offload_dir ~/moe-infinity/deepseek-v2-lite

Assertions:
    1. model.generate() completes without error.
    2. output_ids has more tokens than input_ids (new tokens were generated).
    3. Decoded output_text is a non-empty string.
    4. output_text differs from the input prompt (something was generated).
"""

import argparse
import os
import sys

import torch
from transformers import AutoTokenizer

from moe_infinity import MoE


def main():
    parser = argparse.ArgumentParser(
        description="MoE-Infinity functional inference test"
    )
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-V2-Lite-Chat",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--offload_dir",
        default=os.path.join(
            os.path.expanduser("~"), "moe-infinity", "test-model"
        ),
        help="Directory for offloading expert weights",
    )
    parser.add_argument(
        "--prompt",
        default="What is 2 + 2?",
        help="Input prompt for the model",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Maximum number of new tokens to generate",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )

    config = {
        "offload_path": os.path.join(
            args.offload_dir, args.model.split("/")[-1]
        ),
        "device_memory_ratio": 0.75,
    }

    print(f"Loading model {args.model} via MoE-Infinity ...", flush=True)
    model = MoE(args.model, config)

    # Build prompt with chat template if available.
    if (
        hasattr(tokenizer, "chat_template")
        and tokenizer.chat_template is not None
    ):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": args.prompt},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        input_text = args.prompt

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(
        "cuda:0"
    )

    print("Running model.generate() ...", flush=True)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # --- Assertions ---
    # 1. New tokens were generated.
    assert output_ids.shape[1] > input_ids.shape[1], (
        f"FAIL: no new tokens generated. "
        f"input length={input_ids.shape[1]}, output length={output_ids.shape[1]}"
    )

    # 2. Decode only new tokens.
    new_tokens = output_ids[0][input_ids.shape[1]:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # 3. Output is non-empty.
    assert output_text.strip(), "FAIL: decoded output_text is empty."

    # 4. Output differs from the raw input prompt.
    assert output_text.strip() != args.prompt.strip(), (
        "FAIL: output_text is identical to the input prompt."
    )

    print("PASS: all assertions satisfied.")
    print(f"Prompt   : {args.prompt}")
    print(f"Response : {output_text}")


if __name__ == "__main__":
    main()
