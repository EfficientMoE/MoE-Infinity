import argparse
import os

from transformers import AutoTokenizer

from moe_infinity import MoE

parser = argparse.ArgumentParser(
    description="MoE-Infinity minimal inference example"
)
parser.add_argument(
    "--checkpoint",
    default="Qwen/Qwen3-30B-A3B",
    help="HuggingFace model checkpoint",
)
parser.add_argument(
    "--offload_dir",
    default=os.path.join(os.path.expanduser("~"), "moe-infinity"),
    help="Directory for offloading expert weights",
)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote=True)

config = {
    "offload_path": args.offload_dir,
    "device_memory_ratio": 0.75,  # lower on OOM
}

model = MoE(args.checkpoint, config)

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda:0")

output_ids = model.generate(input_ids)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(output_text)
