"""HuggingFace consistency test for MoE-Infinity.

Verifies that MoE-Infinity produces token-for-token identical output to the
HuggingFace native model under greedy (do_sample=False) decoding.

Usage::

    python tests/python/integration/test_hf_consistency.py \\
        --model deepseek-ai/DeepSeek-V2-Lite-Chat \\
        --offload_dir ~/moe-infinity/deepseek-v2-lite

Steps:
    1. Load model via MoE-Infinity; run generate with greedy decoding.
    2. Load model via HF AutoModelForCausalLM with device_map="auto".
    3. Run generate with identical parameters.
    4. Assert torch.equal(moe_output_ids, hf_output_ids).

NOTE: This test requires enough GPU memory to hold the full model for the HF
baseline run (step 2).  For large models, use a smaller model or adjust
device_memory_ratio.
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from moe_infinity import MoE

MAX_NEW_TOKENS = 32


def build_input(tokenizer, prompt):
    if (
        hasattr(tokenizer, "chat_template")
        and tokenizer.chat_template is not None
    ):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


def main():
    parser = argparse.ArgumentParser(
        description="MoE-Infinity vs HuggingFace consistency test"
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
        default=MAX_NEW_TOKENS,
        help="Maximum number of new tokens to generate",
    )
    args = parser.parse_args()

    generate_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

    print(f"Loading tokenizer from {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    generate_kwargs["pad_token_id"] = tokenizer.eos_token_id

    input_text = build_input(tokenizer, args.prompt)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(
        "cuda:0"
    )

    # --- MoE-Infinity run ---
    print(f"Loading {args.model} via MoE-Infinity ...", flush=True)
    moe_config = {
        "offload_path": os.path.join(
            args.offload_dir, args.model.split("/")[-1]
        ),
        "device_memory_ratio": 0.75,
    }
    moe_model = MoE(args.model, moe_config)

    with torch.no_grad():
        moe_output_ids = moe_model.generate(input_ids, **generate_kwargs)

    del moe_model
    torch.cuda.empty_cache()

    # --- HuggingFace native run ---
    print(
        f"Loading {args.model} via HuggingFace (device_map=auto) ...",
        flush=True,
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        trust_remote_code=True,
    )
    hf_model.eval()

    with torch.no_grad():
        hf_output_ids = hf_model.generate(
            input_ids.to(next(hf_model.parameters()).device), **generate_kwargs
        )

    del hf_model
    torch.cuda.empty_cache()

    # Bring both to CPU for comparison.
    moe_out = moe_output_ids.cpu()
    hf_out = hf_output_ids.cpu()

    # --- Assertion ---
    assert torch.equal(moe_out, hf_out), (
        f"FAIL: MoE-Infinity and HuggingFace outputs differ.\n"
        f"MoE output : {tokenizer.decode(moe_out[0], skip_special_tokens=True)!r}\n"
        f"HF  output : {tokenizer.decode(hf_out[0], skip_special_tokens=True)!r}"
    )

    print("PASS: MoE-Infinity output matches HuggingFace output token-for-token.")
    output_text = tokenizer.decode(
        moe_out[0][input_ids.shape[1]:], skip_special_tokens=True
    )
    print(f"Prompt   : {args.prompt}")
    print(f"Response : {output_text}")


if __name__ == "__main__":
    main()
