# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Minimal MoE-Infinity inference example (DeepSeek-V2-Lite-Chat).

This is the simplest entry point: a HuggingFace-native MoE checkpoint loaded
through the drop-in ``MoE`` class with expert weights offloaded to host
memory/SSD. It runs a single chat prompt and prints the answer.

Suggested hardware
------------------
- 1x GPU with >= 16 GB VRAM (e.g. RTX 4090 / A5000 / A100). The example
  defaults to ``device_memory_ratio=0.75``; lower it on OOM.
- DeepSeek-V2-Lite-Chat is ~16 B params (2.4 B active). With offloading it
  fits comfortably on a single 16-24 GB GPU; smaller GPUs work by lowering
  ``--device_memory_ratio``.
- Fast local SSD for ``--offload_dir`` (the offload format is written there
  on first run and reused afterwards).

For the large FP4 expert-offloading path (DeepSeek-V4-Flash, multi-GPU), see
``examples/deepseek_v4_flash_example.py``.
"""

import argparse
import os

import torch
from transformers import AutoTokenizer

from moe_infinity import MoE

parser = argparse.ArgumentParser(
    description="MoE-Infinity minimal inference example (DeepSeek-V2-Lite-Chat)"
)
parser.add_argument(
    "--checkpoint",
    default="deepseek-ai/DeepSeek-V2-Lite-Chat",
    help="HuggingFace MoE model checkpoint",
)
parser.add_argument(
    "--offload_dir",
    default=os.path.join(os.path.expanduser("~"), "moe-infinity"),
    help=(
        "Directory for offloading expert weights (use a fast local SSD; "
        "must be unique per model)"
    ),
)
parser.add_argument(
    "--device_memory_ratio",
    type=float,
    default=0.75,
    help="Fraction of GPU memory used for expert caching (lower on OOM)",
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

# The offload_path must be unique per model; namespace it by checkpoint name
# so reusing --offload_dir across models does not collide (a common cause of
# partially-loaded weights).
model_name = args.checkpoint.split("/")[-1]
config = {
    "offload_path": os.path.join(args.offload_dir, model_name),
    "device_memory_ratio": args.device_memory_ratio,  # lower on OOM
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
