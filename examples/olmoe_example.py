# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import torch
from transformers import AutoTokenizer

from moe_infinity import MoE

parser = argparse.ArgumentParser(description="OLMoE inference example")
parser.add_argument(
    "--checkpoint",
    default="allenai/OLMoE-1B-7B-0924-Instruct",
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

tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
config = {
    "offload_path": args.offload_dir,
    "device_memory_ratio": 0.75,
}
model = MoE(args.checkpoint, config)

input_ids = tokenizer(
    "The capital of France is", return_tensors="pt"
).input_ids.to("cuda:0")

with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)
