# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from moe_infinity import MoE

CHECKPOINT = "deepseek-ai/DeepSeek-V2-Lite-Chat"
PROMPT = "What is 2+3? Answer with one short sentence."
SEED = 20260809


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offload-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT, trust_remote_code=True
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda:0")
    model = MoE(
        CHECKPOINT,
        {"offload_path": args.offload_dir, "device_memory_ratio": 0.5},
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    payload = {
        "checkpoint": CHECKPOINT,
        "prompt": PROMPT,
        "seed": SEED,
        "input_ids": input_ids[0].cpu().tolist(),
        "output_ids": output_ids[0].cpu().tolist(),
        "decoded": tokenizer.decode(output_ids[0], skip_special_tokens=True),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
