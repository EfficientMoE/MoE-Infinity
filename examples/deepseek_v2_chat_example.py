# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import torch
from transformers import AutoTokenizer

from moe_infinity import MoE

parser = argparse.ArgumentParser(description="DeepSeek-V2 chat example")
parser.add_argument(
    "--checkpoint",
    default="deepseek-ai/DeepSeek-V2-Lite-Chat",
    help="HuggingFace model checkpoint",
)
parser.add_argument(
    "--offload_dir",
    default=os.path.join(os.path.expanduser("~"), "moe-infinity"),
    help="Directory for offloading expert weights",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=64,
    help="Maximum tokens to generate",
)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(
    args.checkpoint, trust_remote_code=True
)
config = {
    "offload_path": args.offload_dir,
    "device_memory_ratio": 0.75,
}
model = MoE(args.checkpoint, config)

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is 2+3? Answer briefly."}],
    tokenize=False,
    add_generation_prompt=True,
)
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda:0")

with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)
