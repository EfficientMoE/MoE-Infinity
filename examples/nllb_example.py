# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import torch
from transformers import AutoTokenizer

from moe_infinity import MoE

parser = argparse.ArgumentParser(description="NLLB-MoE translation example")
parser.add_argument(
    "--checkpoint",
    default="facebook/nllb-moe-54b",
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

tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, src_lang="eng_Latn")
config = {
    "offload_path": args.offload_dir,
    "device_memory_ratio": 0.75,
}
model = MoE(args.checkpoint, config)

input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids.to(
    "cuda:0"
)
forced_bos_token_id = tokenizer.convert_tokens_to_ids("fra_Latn")

with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        forced_bos_token_id=forced_bos_token_id,
    )

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)
